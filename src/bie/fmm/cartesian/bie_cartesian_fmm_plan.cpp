#include "bie_cartesian_fmm_plan.h"

#include "../../../tabi/tabi_fmm_kifmm_hierarchy.h"
#include "../../../titan_executor.h"
#include "bie_cartesian_fmm_checked_size.h"
#include "bie_cartesian_fmm_translations.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace titan::bie::fmm::cartesian {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point begin) noexcept {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

bool finite_vec(Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

bool has_source_child(
    const titan::tabi::fmm_kifmm::KifmmHierarchyTree &tree,
    const titan::tabi::fmm_kifmm::KifmmHierarchyNode &node) noexcept {
  for (const int child : node.child) {
    if (child >= 0 &&
        !tree.nodes[static_cast<std::size_t>(child)].source_indices.empty())
      return true;
  }
  return false;
}

bool has_target_child(
    const titan::tabi::fmm_kifmm::KifmmHierarchyTree &tree,
    const titan::tabi::fmm_kifmm::KifmmHierarchyNode &node) noexcept {
  for (const int child : node.child) {
    if (child >= 0 &&
        !tree.nodes[static_cast<std::size_t>(child)].target_indices.empty())
      return true;
  }
  return false;
}

bool cartesian_local_mac(
    const titan::tabi::fmm_kifmm::KifmmHierarchyNode &source,
    const titan::tabi::fmm_kifmm::KifmmHierarchyNode &target,
    double theta) noexcept {
  const double dx = target.center.x - source.center.x;
  const double dy = target.center.y - source.center.y;
  const double dz = target.center.z - source.center.z;
  const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
  const double source_radius = std::sqrt(3.0) * source.half_width;
  const double target_radius = std::sqrt(3.0) * target.half_width;
  return distance > source_radius + target_radius &&
         (source_radius + target_radius) / distance <= theta;
}

void refine_cartesian_far_pair(
    const titan::tabi::fmm_kifmm::KifmmHierarchyTree &tree, int source_id,
    int target_id, double theta, std::vector<std::pair<int, int>> &far_pairs,
    std::vector<std::pair<int, int>> &direct_pairs) {
  const auto &source = tree.nodes[static_cast<std::size_t>(source_id)];
  const auto &target = tree.nodes[static_cast<std::size_t>(target_id)];
  if (cartesian_local_mac(source, target, theta)) {
    far_pairs.emplace_back(target_id, source_id);
    return;
  }

  const bool source_has_children = has_source_child(tree, source);
  const bool target_has_children = has_target_child(tree, target);
  const bool split_target =
      target_has_children &&
      (!source_has_children || target.half_width >= source.half_width);
  if (split_target) {
    for (const int child_id : target.child) {
      if (child_id >= 0 && !tree.nodes[static_cast<std::size_t>(child_id)]
                                .target_indices.empty()) {
        refine_cartesian_far_pair(tree, source_id, child_id, theta, far_pairs,
                                  direct_pairs);
      }
    }
    return;
  }
  if (source_has_children) {
    for (const int child_id : source.child) {
      if (child_id >= 0 && !tree.nodes[static_cast<std::size_t>(child_id)]
                                .source_indices.empty()) {
        refine_cartesian_far_pair(tree, child_id, target_id, theta, far_pairs,
                                  direct_pairs);
      }
    }
    return;
  }
  direct_pairs.emplace_back(target_id, source_id);
}

std::uint64_t
hash_refined_pairs(std::span<const std::pair<int, int>> far_pairs,
                   std::span<const std::pair<int, int>> direct_pairs) noexcept {
  constexpr std::uint64_t offset = 1469598103934665603ULL;
  constexpr std::uint64_t prime = 1099511628211ULL;
  std::uint64_t hash = offset;
  auto mix = [&](std::uint64_t value) {
    hash ^= value;
    hash *= prime;
  };
  for (const auto &[target, source] : far_pairs) {
    mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(target)));
    mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(source)));
    mix(0x4d324cULL);
  }
  for (const auto &[target, source] : direct_pairs) {
    mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(target)));
    mix(static_cast<std::uint64_t>(static_cast<std::uint32_t>(source)));
    mix(0x444952454354ULL);
  }
  return hash == 0 ? 1 : hash;
}

using TranslationKey = std::array<std::uint64_t, 5>;

TranslationKey translation_key(const CartesianFmmNode &source,
                               const CartesianFmmNode &target) noexcept {
  // Nodes in the strict octree lie on a dyadic lattice.  Keying raw doubles
  // prevents otherwise identical translations from sharing a cache entry
  // when the same center is reached through a different sequence of floating
  // point additions.  Express the displacement in units of the finer box;
  // source/target levels retain the physical lattice scale unambiguously.
  const double lattice_unit = std::min(source.half_width, target.half_width);
  const auto lattice_coordinate = [lattice_unit](double delta) noexcept {
    const auto coordinate = static_cast<std::int64_t>(
        std::llround(delta / std::max(lattice_unit, 1.0e-300)));
    return std::bit_cast<std::uint64_t>(coordinate);
  };
  return {static_cast<std::uint64_t>(source.level),
          static_cast<std::uint64_t>(target.level),
          lattice_coordinate(target.center.x - source.center.x),
          lattice_coordinate(target.center.y - source.center.y),
          lattice_coordinate(target.center.z - source.center.z)};
}

void build_pair_offsets(std::size_t node_count,
                        std::span<const CartesianFmmPair> pairs,
                        std::vector<std::size_t> &offsets,
                        std::vector<std::uint32_t> &targets) {
  const auto offset_count = detail::checked_add_or_error(
      node_count, std::size_t{1}, "allocating Cartesian pair offsets");
  if (!offset_count)
    throw std::overflow_error(offset_count.error());
  const auto offset_bytes = detail::checked_container_allocation(
      *offset_count, offsets, "allocating Cartesian pair offsets");
  const auto target_bytes = detail::checked_container_allocation(
      node_count, targets, "reserving Cartesian pair target nodes");
  if (!offset_bytes || !target_bytes) {
    throw std::length_error(!offset_bytes ? offset_bytes.error()
                                          : target_bytes.error());
  }
  offsets.assign(*offset_count, 0);
  for (const CartesianFmmPair &pair : pairs) {
    if (pair.target_node < node_count) {
      if (offsets[static_cast<std::size_t>(pair.target_node) + 1] ==
          std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(
            "BIE Cartesian full FMM pair-offset counter overflow");
      }
      ++offsets[static_cast<std::size_t>(pair.target_node) + 1];
    }
  }
  for (std::size_t i = 1; i < offsets.size(); ++i) {
    const auto summed = detail::checked_add_or_error(
        offsets[i], offsets[i - 1], "building Cartesian pair offsets");
    if (!summed)
      throw std::overflow_error(summed.error());
    offsets[i] = *summed;
  }
  targets.clear();
  targets.reserve(node_count);
  for (std::size_t node = 0; node < node_count; ++node) {
    if (offsets[node + 1] > offsets[node])
      targets.push_back(static_cast<std::uint32_t>(node));
  }
}

void sort_pairs_by_target(std::vector<CartesianFmmPair> &pairs) {
  std::sort(pairs.begin(), pairs.end(),
            [](const CartesianFmmPair &left, const CartesianFmmPair &right) {
              if (left.target_node != right.target_node)
                return left.target_node < right.target_node;
              if (left.source_node != right.source_node)
                return left.source_node < right.source_node;
              return left.translation < right.translation;
            });
}

} // namespace

std::span<const double>
CartesianFmmPlan::coulomb_translation(std::size_t index) const noexcept {
  if (green_term_count_ == 0 ||
      index > coulomb_translations_.size() / green_term_count_)
    return {};
  const std::size_t begin = index * green_term_count_;
  if (green_term_count_ > coulomb_translations_.size() - begin)
    return {};
  return {coulomb_translations_.data() + begin, green_term_count_};
}

std::span<const double>
CartesianFmmPlan::yukawa_translation(std::size_t index) const noexcept {
  if (green_term_count_ == 0 ||
      index > yukawa_translations_.size() / green_term_count_)
    return {};
  const std::size_t begin = index * green_term_count_;
  if (green_term_count_ > yukawa_translations_.size() - begin)
    return {};
  return {yukawa_translations_.data() + begin, green_term_count_};
}

std::expected<std::shared_ptr<const CartesianFmmPlan>, std::string>
prepare_cartesian_fmm_plan(std::span<const PanelGeometry> panels,
                           const CartesianFmmPlanOptions &options,
                           TitanExecutor &executor) {
  const auto setup_begin = Clock::now();
  if (panels.empty())
    return std::unexpected("BIE Cartesian full FMM requires panels");
  if (options.expansion_order != 1 && options.expansion_order != 4)
    return std::unexpected(
        "BIE Cartesian full FMM expansion order must be 1 or 4");
  if (options.leaf_size < 4 || options.leaf_size > 2048 ||
      options.max_depth < 1 || options.max_depth > 12 ||
      !std::isfinite(options.theta) || options.theta <= 0.0 ||
      options.theta > 1.0 || !std::isfinite(options.eps_ratio) ||
      options.eps_ratio <= 0.0 || !std::isfinite(options.kappa) ||
      options.kappa < 0.0) {
    return std::unexpected("BIE Cartesian full FMM plan options are invalid");
  }
  for (const PanelGeometry &panel : panels) {
    if (!finite_vec(panel.target) || !finite_vec(panel.source) ||
        !finite_vec(panel.normal) || !std::isfinite(panel.area) ||
        panel.area <= 0.0) {
      return std::unexpected(
          "BIE Cartesian full FMM panel geometry is invalid");
    }
  }

  if (panels.size() > std::numeric_limits<std::uint64_t>::max()) {
    return std::unexpected(
        "BIE Cartesian full FMM panel count exceeds 64-bit accounting");
  }
  const std::uint64_t panel_count_u64 =
      static_cast<std::uint64_t>(panels.size());
  const auto expected_panel_pairs = detail::checked_multiply_or_error(
      panel_count_u64, panel_count_u64,
      "counting Cartesian full-FMM panel pairs");
  if (!expected_panel_pairs)
    return std::unexpected(expected_panel_pairs.error());

  try {
  auto plan = std::shared_ptr<CartesianFmmPlan>(new CartesianFmmPlan());
  plan->options_ = options;
  const auto plan_panel_bytes = detail::checked_container_allocation(
      panels.size(), plan->panels_, "allocating Cartesian plan panels");
  if (!plan_panel_bytes)
    return std::unexpected(plan_panel_bytes.error());
  plan->panels_.assign(panels.begin(), panels.end());
  plan->stats_.panels = panels.size();

  std::vector<titan::tabi::fmm_kifmm::Vec3> source_points;
  std::vector<titan::tabi::fmm_kifmm::Vec3> target_points;
  const auto source_point_bytes = detail::checked_container_allocation(
      panels.size(), source_points, "allocating Cartesian source points");
  const auto target_point_bytes = detail::checked_container_allocation(
      panels.size(), target_points, "allocating Cartesian target points");
  if (!source_point_bytes || !target_point_bytes) {
    return std::unexpected(!source_point_bytes ? source_point_bytes.error()
                                                : target_point_bytes.error());
  }
  source_points.reserve(panels.size());
  target_points.reserve(panels.size());
  for (const PanelGeometry &panel : panels) {
    source_points.push_back({panel.source.x, panel.source.y, panel.source.z});
    target_points.push_back({panel.target.x, panel.target.y, panel.target.z});
  }

  titan::tabi::fmm_kifmm::KifmmHierarchyPolicy policy;
  policy.order = 4;
  policy.max_depth = options.max_depth;
  policy.leaf_size = std::clamp(2 * options.leaf_size, 8, 4096);
  policy.theta = options.theta;
  policy.tree_stop_policy = "baseline";
  policy.m2l_matrix_cache_policy = "off";
  const auto tree_begin = Clock::now();
  const auto tree = titan::tabi::fmm_kifmm::build_strict_octree(
      source_points, target_points, policy);
  plan->stats_.tree_seconds = seconds_since(tree_begin);
  if (tree.status != "ok" || tree.nodes.empty()) {
    return std::unexpected("BIE Cartesian full FMM octree failed: " +
                           tree.status);
  }
  if (tree.nodes.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return std::unexpected(
        "BIE Cartesian full FMM octree exceeds 32-bit node addressing");
  }
  plan->stats_.tree_hash = tree.tree_hash;

  const auto worklist_begin = Clock::now();
  const auto hierarchy_worklist =
      titan::tabi::fmm_kifmm::build_hierarchy_worklist(tree, policy);
  std::vector<std::pair<int, int>> m2l_pairs;
  std::vector<std::pair<int, int>> direct_leaf_pairs =
      hierarchy_worklist.direct_leaf_pairs;
  m2l_pairs.reserve(hierarchy_worklist.m2l_pairs.size());
  for (const auto &[target_id, source_id] : hierarchy_worklist.m2l_pairs) {
    refine_cartesian_far_pair(tree, source_id, target_id, options.theta,
                              m2l_pairs, direct_leaf_pairs);
  }
  const auto by_target_source = [](const auto &left, const auto &right) {
    if (left.first != right.first)
      return left.first < right.first;
    return left.second < right.second;
  };
  std::sort(m2l_pairs.begin(), m2l_pairs.end(), by_target_source);
  std::sort(direct_leaf_pairs.begin(), direct_leaf_pairs.end(),
            by_target_source);
  plan->stats_.worklist_seconds = seconds_since(worklist_begin);
  plan->stats_.worklist_hash = hash_refined_pairs(m2l_pairs, direct_leaf_pairs);

  const auto node_bytes = detail::checked_container_allocation(
      tree.nodes.size(), plan->nodes_, "allocating Cartesian plan nodes");
  const auto source_index_bytes = detail::checked_container_allocation(
      panels.size(), plan->source_panel_indices_,
      "reserving Cartesian source panel indices");
  const auto target_index_bytes = detail::checked_container_allocation(
      panels.size(), plan->target_panel_indices_,
      "reserving Cartesian target panel indices");
  const auto source_leaf_bytes = detail::checked_container_allocation(
      tree.nodes.size(), plan->source_leaves_,
      "reserving Cartesian source leaves");
  const auto target_leaf_bytes = detail::checked_container_allocation(
      tree.nodes.size(), plan->target_leaves_,
      "reserving Cartesian target leaves");
  if (!node_bytes || !source_index_bytes || !target_index_bytes ||
      !source_leaf_bytes || !target_leaf_bytes) {
    const std::string message = !node_bytes
                                    ? node_bytes.error()
                                    : !source_index_bytes
                                          ? source_index_bytes.error()
                                          : !target_index_bytes
                                                ? target_index_bytes.error()
                                                : !source_leaf_bytes
                                                      ? source_leaf_bytes.error()
                                                      : target_leaf_bytes.error();
    return std::unexpected(message);
  }
  plan->nodes_.resize(tree.nodes.size());
  plan->source_panel_indices_.reserve(panels.size());
  plan->target_panel_indices_.reserve(panels.size());
  plan->source_leaves_.reserve(tree.nodes.size());
  plan->target_leaves_.reserve(tree.nodes.size());
  int max_level = 0;
  for (std::size_t node_id = 0; node_id < tree.nodes.size(); ++node_id) {
    const auto &source_node = tree.nodes[node_id];
    CartesianFmmNode &node = plan->nodes_[node_id];
    node.parent = source_node.parent;
    node.child = source_node.child;
    node.level = source_node.level;
    node.center = {source_node.center.x, source_node.center.y,
                   source_node.center.z};
    node.half_width = source_node.half_width;
    node.source_population = source_node.source_indices.size();
    node.target_population = source_node.target_indices.size();
    max_level = std::max(max_level, node.level);
    if (!source_node.source_indices.empty() &&
        !has_source_child(tree, source_node)) {
      node.source_begin = plan->source_panel_indices_.size();
      node.source_count = source_node.source_indices.size();
      plan->source_panel_indices_.insert(plan->source_panel_indices_.end(),
                                         source_node.source_indices.begin(),
                                         source_node.source_indices.end());
      plan->source_leaves_.push_back(static_cast<std::uint32_t>(node_id));
    }
    if (!source_node.target_indices.empty() &&
        !has_target_child(tree, source_node)) {
      node.target_begin = plan->target_panel_indices_.size();
      node.target_count = source_node.target_indices.size();
      plan->target_panel_indices_.insert(plan->target_panel_indices_.end(),
                                         source_node.target_indices.begin(),
                                         source_node.target_indices.end());
      plan->target_leaves_.push_back(static_cast<std::uint32_t>(node_id));
    }
  }

  const auto level_count = detail::checked_add_or_error(
      static_cast<std::size_t>(max_level), std::size_t{2},
      "allocating Cartesian level offsets");
  if (!level_count)
    return std::unexpected(level_count.error());
  const auto level_offset_bytes = detail::checked_container_allocation(
      *level_count, plan->level_offsets_, "allocating Cartesian level offsets");
  const auto nodes_by_level_bytes = detail::checked_container_allocation(
      plan->nodes_.size(), plan->nodes_by_level_,
      "allocating Cartesian nodes-by-level");
  if (!level_offset_bytes || !nodes_by_level_bytes) {
    return std::unexpected(!level_offset_bytes ? level_offset_bytes.error()
                                               : nodes_by_level_bytes.error());
  }
  plan->level_offsets_.assign(*level_count, 0);
  for (const CartesianFmmNode &node : plan->nodes_) {
    if (plan->level_offsets_[static_cast<std::size_t>(node.level) + 1] ==
        std::numeric_limits<std::size_t>::max()) {
      return std::unexpected(
          "BIE Cartesian full FMM level population counter overflow");
    }
    ++plan->level_offsets_[static_cast<std::size_t>(node.level) + 1];
  }
  for (std::size_t level = 1; level < plan->level_offsets_.size(); ++level) {
    const auto offset = detail::checked_add_or_error(
        plan->level_offsets_[level], plan->level_offsets_[level - 1],
        "building Cartesian level offsets");
    if (!offset)
      return std::unexpected(offset.error());
    plan->level_offsets_[level] = *offset;
  }
  plan->nodes_by_level_.resize(plan->nodes_.size());
  std::vector<std::size_t> next_level = plan->level_offsets_;
  for (std::size_t node = 0; node < plan->nodes_.size(); ++node) {
    const std::size_t level =
        static_cast<std::size_t>(plan->nodes_[node].level);
    plan->nodes_by_level_[next_level[level]++] =
        static_cast<std::uint32_t>(node);
  }

  plan->green_term_count_ =
      green_jet_coefficient_count(options.expansion_order);
  std::map<TranslationKey, std::uint32_t> translation_lookup;
  std::vector<std::pair<std::uint32_t, std::uint32_t>>
      translation_node_pairs;
  const auto cache_begin = Clock::now();
  const auto m2l_pair_bytes = detail::checked_container_allocation(
      m2l_pairs.size(), plan->m2l_pairs_, "reserving Cartesian far pairs");
  const auto translation_node_pair_bytes = detail::checked_container_allocation(
      m2l_pairs.size(), translation_node_pairs,
      "reserving Cartesian translation node pairs");
  if (!m2l_pair_bytes || !translation_node_pair_bytes) {
    return std::unexpected(!m2l_pair_bytes
                               ? m2l_pair_bytes.error()
                               : translation_node_pair_bytes.error());
  }
  plan->m2l_pairs_.reserve(m2l_pairs.size());
  translation_node_pairs.reserve(m2l_pairs.size());
  for (const auto &[target_id, source_id] : m2l_pairs) {
    if (target_id < 0 || source_id < 0 ||
        static_cast<std::size_t>(target_id) >= plan->nodes_.size() ||
        static_cast<std::size_t>(source_id) >= plan->nodes_.size()) {
      return std::unexpected(
          "BIE Cartesian full FMM worklist contains an invalid far pair");
    }
    const CartesianFmmNode &target =
        plan->nodes_[static_cast<std::size_t>(target_id)];
    const CartesianFmmNode &source =
        plan->nodes_[static_cast<std::size_t>(source_id)];
    const TranslationKey key = translation_key(source, target);
    auto found = translation_lookup.find(key);
    std::uint32_t translation = 0;
    if (found == translation_lookup.end()) {
      if (translation_lookup.size() >=
          static_cast<std::size_t>(
              std::numeric_limits<std::uint32_t>::max())) {
        return std::unexpected(
            "BIE Cartesian full FMM translation cache exceeds 32-bit indexing");
      }
      translation = static_cast<std::uint32_t>(translation_lookup.size());
      translation_lookup.emplace(key, translation);
      translation_node_pairs.emplace_back(
          static_cast<std::uint32_t>(source_id),
          static_cast<std::uint32_t>(target_id));
    } else {
      translation = found->second;
      ++plan->stats_.translation_cache_hits;
    }
    plan->m2l_pairs_.push_back({static_cast<std::uint32_t>(target_id),
                                static_cast<std::uint32_t>(source_id),
                                translation});
    const auto far_pairs = detail::checked_multiply_or_error(
        static_cast<std::uint64_t>(target.target_population),
        static_cast<std::uint64_t>(source.source_population),
        "counting Cartesian far panel pairs");
    if (!far_pairs)
      return std::unexpected(far_pairs.error());
    const auto total_far_pairs = detail::checked_add_or_error(
        plan->stats_.far_panel_pairs, *far_pairs,
        "accumulating Cartesian far panel pairs");
    if (!total_far_pairs)
      return std::unexpected(total_far_pairs.error());
    plan->stats_.far_panel_pairs = *total_far_pairs;
  }
  const std::size_t translation_count = translation_node_pairs.size();
  const auto translation_coefficients = detail::checked_multiply_or_error(
      translation_count, plan->green_term_count_,
      "allocating Cartesian translation coefficients");
  if (!translation_coefficients)
    return std::unexpected(translation_coefficients.error());
  const auto coulomb_bytes = detail::checked_container_allocation(
      *translation_coefficients, plan->coulomb_translations_,
      "allocating Cartesian Coulomb translation cache");
  const auto yukawa_bytes = detail::checked_container_allocation(
      *translation_coefficients, plan->yukawa_translations_,
      "allocating Cartesian Yukawa translation cache");
  if (!coulomb_bytes || !yukawa_bytes) {
    return std::unexpected(!coulomb_bytes ? coulomb_bytes.error()
                                          : yukawa_bytes.error());
  }
  plan->coulomb_translations_.resize(*translation_coefficients);
  plan->yukawa_translations_.resize(*translation_coefficients);
  executor.parallel_for(
      translation_count, executor.capacity(), 1,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t translation = begin; translation < end;
             ++translation) {
          const auto [source_id, target_id] =
              translation_node_pairs[translation];
          const CartesianFmmNode &source = plan->nodes_[source_id];
          const CartesianFmmNode &target = plan->nodes_[target_id];
          const std::size_t coefficient_begin =
              translation * plan->green_term_count_;
          build_green_translation(
              source.center, target.center, options.kappa,
              options.expansion_order,
              std::span<double>(plan->coulomb_translations_.data() +
                                    coefficient_begin,
                                plan->green_term_count_),
              std::span<double>(plan->yukawa_translations_.data() +
                                    coefficient_begin,
                                plan->green_term_count_));
        }
      });
  plan->stats_.cache_seconds = seconds_since(cache_begin);

  const auto direct_pair_bytes = detail::checked_container_allocation(
      direct_leaf_pairs.size(), plan->direct_pairs_,
      "reserving Cartesian direct pairs");
  if (!direct_pair_bytes)
    return std::unexpected(direct_pair_bytes.error());
  plan->direct_pairs_.reserve(direct_leaf_pairs.size());
  for (const auto &[target_id, source_id] : direct_leaf_pairs) {
    if (target_id < 0 || source_id < 0 ||
        static_cast<std::size_t>(target_id) >= plan->nodes_.size() ||
        static_cast<std::size_t>(source_id) >= plan->nodes_.size()) {
      return std::unexpected(
          "BIE Cartesian full FMM worklist contains an invalid direct pair");
    }
    const CartesianFmmNode &target =
        plan->nodes_[static_cast<std::size_t>(target_id)];
    const CartesianFmmNode &source =
        plan->nodes_[static_cast<std::size_t>(source_id)];
    if (target.target_count == 0 || source.source_count == 0) {
      return std::unexpected(
          "BIE Cartesian full FMM direct pair is not leaf-addressable");
    }
    plan->direct_pairs_.push_back({static_cast<std::uint32_t>(target_id),
                                   static_cast<std::uint32_t>(source_id), 0});
    const auto direct_pairs = detail::checked_multiply_or_error(
        static_cast<std::uint64_t>(target.target_population),
        static_cast<std::uint64_t>(source.source_population),
        "counting Cartesian direct panel pairs");
    if (!direct_pairs)
      return std::unexpected(direct_pairs.error());
    const auto total_direct_pairs = detail::checked_add_or_error(
        plan->stats_.direct_panel_pairs, *direct_pairs,
        "accumulating Cartesian direct panel pairs");
    if (!total_direct_pairs)
      return std::unexpected(total_direct_pairs.error());
    plan->stats_.direct_panel_pairs = *total_direct_pairs;
  }

  // The hierarchy builder currently emits target-major worklists, but the BIE
  // plan deliberately does not depend on that implementation detail.  The
  // offset tables below require contiguous target groups and the canonical
  // ordering also keeps repeated plans bitwise deterministic.
  sort_pairs_by_target(plan->m2l_pairs_);
  sort_pairs_by_target(plan->direct_pairs_);
  build_pair_offsets(plan->nodes_.size(), plan->m2l_pairs_, plan->m2l_offsets_,
                     plan->m2l_target_nodes_);
  build_pair_offsets(plan->nodes_.size(), plan->direct_pairs_,
                     plan->direct_offsets_, plan->direct_target_nodes_);

  const auto audit_begin = Clock::now();
  plan->stats_.expected_panel_pairs = *expected_panel_pairs;
  const auto accounted_panel_pairs = detail::checked_add_or_error(
      plan->stats_.far_panel_pairs, plan->stats_.direct_panel_pairs,
      "auditing Cartesian near/far panel-pair coverage");
  if (!accounted_panel_pairs)
    return std::unexpected(accounted_panel_pairs.error());
  plan->stats_.exact_pair_coverage =
      *accounted_panel_pairs == plan->stats_.expected_panel_pairs;
  const TranslationParityAudit parity =
      run_translation_parity_audit(options.expansion_order);
  plan->stats_.m2m_parity_relative_error = parity.m2m_relative_error;
  plan->stats_.l2l_parity_relative_error = parity.l2l_relative_error;
  plan->stats_.audit_seconds = seconds_since(audit_begin);
  if (!plan->stats_.exact_pair_coverage) {
    std::ostringstream message;
    message << "BIE Cartesian full FMM near/far coverage mismatch: far="
            << plan->stats_.far_panel_pairs
            << " direct=" << plan->stats_.direct_panel_pairs
            << " expected=" << plan->stats_.expected_panel_pairs;
    return std::unexpected(message.str());
  }
  if (!parity.finite_outputs || parity.m2m_relative_error > 1.0e-11 ||
      parity.l2l_relative_error > 1.0e-11) {
    std::ostringstream message;
    message << "BIE Cartesian full FMM translation parity failed: m2m="
            << parity.m2m_relative_error
            << " l2l=" << parity.l2l_relative_error;
    return std::unexpected(message.str());
  }

  plan->stats_.nodes = plan->nodes_.size();
  plan->stats_.source_leaves = plan->source_leaves_.size();
  plan->stats_.target_leaves = plan->target_leaves_.size();
  plan->stats_.far_pairs = plan->m2l_pairs_.size();
  plan->stats_.direct_pairs = plan->direct_pairs_.size();
  plan->stats_.translation_cache_entries = translation_lookup.size();
  const auto coulomb_cache_bytes = detail::checked_container_capacity_bytes(
      plan->coulomb_translations_,
      "accounting Cartesian Coulomb translation cache");
  const auto yukawa_cache_bytes = detail::checked_container_capacity_bytes(
      plan->yukawa_translations_,
      "accounting Cartesian Yukawa translation cache");
  if (!coulomb_cache_bytes || !yukawa_cache_bytes) {
    return std::unexpected(!coulomb_cache_bytes ? coulomb_cache_bytes.error()
                                                 : yukawa_cache_bytes.error());
  }
  const auto translation_cache_bytes = detail::checked_add_or_error(
      *coulomb_cache_bytes, *yukawa_cache_bytes,
      "summing Cartesian translation cache bytes");
  if (!translation_cache_bytes)
    return std::unexpected(translation_cache_bytes.error());
  plan->stats_.translation_cache_bytes = *translation_cache_bytes;

  std::size_t retained_plan_bytes = 0;
  std::string retained_error;
  const auto append_retained_bytes = [&](const auto &buffer,
                                         std::string_view context) {
    const auto bytes = detail::checked_container_capacity_bytes(buffer, context);
    if (!bytes) {
      retained_error = bytes.error();
      return false;
    }
    const auto total = detail::checked_add_or_error(
        retained_plan_bytes, *bytes, "summing retained Cartesian plan bytes");
    if (!total) {
      retained_error = total.error();
      return false;
    }
    retained_plan_bytes = *total;
    return true;
  };
  if (!append_retained_bytes(plan->panels_, "accounting Cartesian plan panels") ||
      !append_retained_bytes(plan->nodes_, "accounting Cartesian plan nodes") ||
      !append_retained_bytes(plan->source_panel_indices_,
                             "accounting Cartesian source panel indices") ||
      !append_retained_bytes(plan->target_panel_indices_,
                             "accounting Cartesian target panel indices") ||
      !append_retained_bytes(plan->source_leaves_,
                             "accounting Cartesian source leaves") ||
      !append_retained_bytes(plan->target_leaves_,
                             "accounting Cartesian target leaves") ||
      !append_retained_bytes(plan->nodes_by_level_,
                             "accounting Cartesian nodes-by-level") ||
      !append_retained_bytes(plan->level_offsets_,
                             "accounting Cartesian level offsets") ||
      !append_retained_bytes(plan->m2l_pairs_,
                             "accounting Cartesian far pairs") ||
      !append_retained_bytes(plan->m2l_offsets_,
                             "accounting Cartesian far offsets") ||
      !append_retained_bytes(plan->m2l_target_nodes_,
                             "accounting Cartesian far target nodes") ||
      !append_retained_bytes(plan->direct_pairs_,
                             "accounting Cartesian direct pairs") ||
      !append_retained_bytes(plan->direct_offsets_,
                             "accounting Cartesian direct offsets") ||
      !append_retained_bytes(plan->direct_target_nodes_,
                             "accounting Cartesian direct target nodes") ||
      !append_retained_bytes(plan->coulomb_translations_,
                             "accounting Cartesian Coulomb translations") ||
      !append_retained_bytes(plan->yukawa_translations_,
                             "accounting Cartesian Yukawa translations")) {
    return std::unexpected(retained_error);
  }
  plan->stats_.retained_plan_bytes = retained_plan_bytes;
  plan->stats_.finite_outputs = true;
  plan->stats_.status = "ok";
  plan->stats_.setup_seconds = seconds_since(setup_begin);
  return std::shared_ptr<const CartesianFmmPlan>(std::move(plan));
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        "BIE Cartesian full FMM allocation failed while preparing the plan");
  } catch (const std::length_error &error) {
    return std::unexpected(error.what());
  } catch (const std::overflow_error &error) {
    return std::unexpected(error.what());
  }
}

} // namespace titan::bie::fmm::cartesian
