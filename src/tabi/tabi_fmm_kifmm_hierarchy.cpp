#include "tabi_fmm_kifmm_hierarchy.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>

namespace titan::tabi::fmm_kifmm {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

bool finite_vec(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

double norm(Vec3 v) noexcept {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3 sub(Vec3 a, Vec3 b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

void hash_u64(std::uint64_t &hash, std::uint64_t value) noexcept {
  hash ^= value;
  hash *= kFnvPrime;
}

void hash_int(std::uint64_t &hash, int value) noexcept {
  hash_u64(hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
}

void hash_double(std::uint64_t &hash, double value) noexcept {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(hash, bits);
}

void hash_string(std::uint64_t &hash, const std::string &value) noexcept {
  for (const char ch : value) {
    hash_u64(hash, static_cast<unsigned char>(ch));
  }
}

bool valid_policy(const KifmmHierarchyPolicy &policy) noexcept {
  return policy.order >= 4 && policy.order <= 12 && policy.order % 2 == 0 &&
         policy.max_depth >= 1 && policy.max_depth <= 12 &&
         policy.leaf_size >= 1 && policy.leaf_size <= 4096 &&
         std::isfinite(policy.theta) && policy.theta > 0.0 &&
         std::isfinite(policy.equiv_scale) && policy.equiv_scale > 1.0 &&
         std::isfinite(policy.upward_check_scale) &&
         policy.upward_check_scale > policy.equiv_scale &&
         std::isfinite(policy.local_equiv_scale) &&
         policy.local_equiv_scale > 1.0 &&
         std::isfinite(policy.local_check_scale) &&
         policy.local_check_scale > 1.0 &&
         policy.local_equiv_scale > policy.local_check_scale &&
         policy.source_box_policy == "strict-octree" &&
         (policy.m2l_eval_policy == "scalar-existing" ||
          policy.m2l_eval_policy == "target-check-batched") &&
         (policy.m2l_matrix_cache_policy == "off" ||
         policy.m2l_matrix_cache_policy == "relative-geometry-cache" ||
         policy.m2l_matrix_cache_policy == "static-translation-table-cache" ||
         policy.m2l_matrix_cache_policy == "canonical-offset-cache" ||
          policy.m2l_matrix_cache_policy == "target-node-streaming-cache" ||
          policy.m2l_matrix_cache_policy == "bounded-lru-cache") &&
         std::isfinite(policy.m2l_matrix_cache_limit_mb) &&
         policy.m2l_matrix_cache_limit_mb >= 0.0 &&
         (policy.m2l_gemm_batch_policy == "per-pair" ||
          policy.m2l_gemm_batch_policy == "group-by-cache-key") &&
         (policy.m2l_parallel_policy == "serial" ||
          policy.m2l_parallel_policy == "target-node-key-group") &&
         (policy.tree_stop_policy == "baseline" ||
          policy.tree_stop_policy == "panel-radius" ||
          policy.tree_stop_policy == "panel-radius-occupancy") &&
         (policy.jcp_channel_policy == "all" ||
          policy.jcp_channel_policy == "k123-only") &&
         std::isfinite(policy.tree_panel_radius_factor) &&
         policy.tree_panel_radius_factor >= 0.0 &&
         policy.tree_occupancy_stop >= 0 &&
         policy.tree_occupancy_stop <= 4096 &&
         policy.m2l_parallel_workers >= 1 &&
         policy.m2l_parallel_workers <= 256;
}

std::uint64_t hash_policy(const KifmmHierarchyPolicy &policy) noexcept {
  std::uint64_t hash = kFnvOffset;
  hash_int(hash, policy.order);
  hash_int(hash, policy.max_depth);
  hash_int(hash, policy.leaf_size);
  hash_double(hash, policy.theta);
  hash_double(hash, policy.equiv_scale);
  hash_double(hash, policy.upward_check_scale);
  hash_double(hash, policy.local_equiv_scale);
  hash_double(hash, policy.local_check_scale);
  hash_string(hash, policy.condition_policy);
  hash_string(hash, policy.source_box_policy);
  hash_string(hash, policy.m2l_eval_policy);
  hash_string(hash, policy.m2l_matrix_cache_policy);
  hash_double(hash, policy.m2l_matrix_cache_limit_mb);
  hash_string(hash, policy.m2l_gemm_batch_policy);
  hash_string(hash, policy.m2l_parallel_policy);
  hash_string(hash, policy.tree_stop_policy);
  hash_string(hash, policy.jcp_channel_policy);
  hash_double(hash, policy.tree_panel_radius_factor);
  hash_int(hash, policy.tree_occupancy_stop);
  hash_int(hash, policy.m2l_parallel_workers);
  return hash == 0 ? 1 : hash;
}

int octant_for(Vec3 point, Vec3 center) noexcept {
  int octant = 0;
  if (point.x >= center.x) {
    octant |= 1;
  }
  if (point.y >= center.y) {
    octant |= 2;
  }
  if (point.z >= center.z) {
    octant |= 4;
  }
  return octant;
}

Vec3 child_center(Vec3 parent_center, double child_half_width,
                  int octant) noexcept {
  const double sx = (octant & 1) != 0 ? 1.0 : -1.0;
  const double sy = (octant & 2) != 0 ? 1.0 : -1.0;
  const double sz = (octant & 4) != 0 ? 1.0 : -1.0;
  return {parent_center.x + sx * child_half_width,
          parent_center.y + sy * child_half_width,
          parent_center.z + sz * child_half_width};
}

std::uint64_t hash_tree_nodes(const std::vector<KifmmHierarchyNode> &nodes) {
  std::uint64_t hash = kFnvOffset;
  for (const KifmmHierarchyNode &node : nodes) {
    hash_int(hash, node.id);
    hash_int(hash, node.parent);
    hash_int(hash, node.level);
    hash_double(hash, node.center.x);
    hash_double(hash, node.center.y);
    hash_double(hash, node.center.z);
    hash_double(hash, node.half_width);
    for (const int child : node.child) {
      hash_int(hash, child);
    }
    for (const int index : node.source_indices) {
      hash_int(hash, index);
    }
    hash_u64(hash, 0x9e3779b97f4a7c15ULL);
    for (const int index : node.target_indices) {
      hash_int(hash, index);
    }
    hash_u64(hash, 0xbf58476d1ce4e5b9ULL);
  }
  return hash == 0 ? 1 : hash;
}

double max_radius_for_indices(const std::vector<int> &indices,
                              std::span<const double> radii) noexcept {
  if (radii.empty()) {
    return 0.0;
  }
  double out = 0.0;
  for (const int index : indices) {
    const std::size_t i = static_cast<std::size_t>(index);
    if (i < radii.size() && std::isfinite(radii[i])) {
      out = std::max(out, radii[i]);
    }
  }
  return out;
}

double node_max_panel_radius(const KifmmHierarchyNode &node,
                             std::span<const double> source_radii,
                             std::span<const double> target_radii) noexcept {
  return std::max(max_radius_for_indices(node.source_indices, source_radii),
                  max_radius_for_indices(node.target_indices, target_radii));
}

bool uses_panel_radius_stop(const KifmmHierarchyPolicy &policy) noexcept {
  return policy.tree_stop_policy == "panel-radius" ||
         policy.tree_stop_policy == "panel-radius-occupancy";
}

bool uses_occupancy_stop(const KifmmHierarchyPolicy &policy) noexcept {
  return policy.tree_stop_policy == "panel-radius-occupancy" &&
         policy.tree_occupancy_stop > 0;
}

bool sparse_side_stop(const KifmmHierarchyNode &node,
                      const KifmmHierarchyPolicy &policy) noexcept {
  if (!uses_occupancy_stop(policy)) {
    return false;
  }
  const int source_count = static_cast<int>(node.source_indices.size());
  const int target_count = static_cast<int>(node.target_indices.size());
  return source_count <= policy.tree_occupancy_stop ||
         target_count <= policy.tree_occupancy_stop;
}

void update_stats(KifmmHierarchyTree &tree) {
  std::set<int> observed_levels;
  std::set<double> observed_half_widths;
  tree.stats.node_count = static_cast<int>(tree.nodes.size());
  tree.stats.leaf_count = 0;
  tree.stats.max_observed_depth = 0;
  std::uint64_t leaf_source_total = 0;
  std::uint64_t leaf_target_total = 0;
  for (const KifmmHierarchyNode &node : tree.nodes) {
    observed_levels.insert(node.level);
    observed_half_widths.insert(node.half_width);
    tree.stats.max_observed_depth =
        std::max(tree.stats.max_observed_depth, node.level);
    if (node.is_leaf) {
      ++tree.stats.leaf_count;
      leaf_source_total +=
          static_cast<std::uint64_t>(node.source_indices.size());
      leaf_target_total +=
          static_cast<std::uint64_t>(node.target_indices.size());
    }
  }
  if (tree.stats.leaf_count > 0) {
    tree.stats.mean_leaf_source_count =
        static_cast<double>(leaf_source_total) /
        static_cast<double>(tree.stats.leaf_count);
    tree.stats.mean_leaf_target_count =
        static_cast<double>(leaf_target_total) /
        static_cast<double>(tree.stats.leaf_count);
  }
  tree.stats.observed_level_count = static_cast<int>(observed_levels.size());
  tree.stats.unique_half_widths =
      static_cast<int>(observed_half_widths.size());
  tree.stats.exact_pair_coverage =
      static_cast<std::uint64_t>(tree.stats.source_count) *
      static_cast<std::uint64_t>(tree.stats.target_count);
}

void update_leaf_radius_stats(KifmmHierarchyTree &tree,
                              const KifmmHierarchyNode &node,
                              std::span<const double> source_radii,
                              std::span<const double> target_radii) {
  const double radius =
      node_max_panel_radius(node, source_radii, target_radii);
  tree.stats.max_leaf_panel_radius =
      std::max(tree.stats.max_leaf_panel_radius, radius);
  if (radius > 0.0) {
    tree.stats.min_leaf_half_width_over_panel_radius =
        std::min(tree.stats.min_leaf_half_width_over_panel_radius,
                 node.half_width / radius);
  }
}

void mark_leaf(KifmmHierarchyTree &tree, int node_id, const char *reason,
               std::span<const double> source_radii,
               std::span<const double> target_radii) {
  KifmmHierarchyNode &node = tree.nodes[static_cast<std::size_t>(node_id)];
  node.is_leaf = true;
  if (std::strcmp(reason, "max_depth") == 0) {
    ++tree.stats.max_depth_stop_count;
  } else if (std::strcmp(reason, "leaf_size") == 0) {
    ++tree.stats.leaf_size_stop_count;
  } else if (std::strcmp(reason, "radius") == 0) {
    ++tree.stats.radius_stop_leaf_count;
  } else if (std::strcmp(reason, "occupancy") == 0) {
    ++tree.stats.occupancy_stop_leaf_count;
  } else if (std::strcmp(reason, "unsplittable") == 0) {
    ++tree.stats.unsplittable_stop_count;
  }
  update_leaf_radius_stats(tree, node, source_radii, target_radii);
}

void split_node(KifmmHierarchyTree &tree, int node_id,
                std::span<const Vec3> sources, std::span<const Vec3> targets,
                std::span<const double> source_radii,
                std::span<const double> target_radii,
                const KifmmHierarchyPolicy &policy) {
  const KifmmHierarchyNode node =
      tree.nodes[static_cast<std::size_t>(node_id)];
  const int total_count = static_cast<int>(node.source_indices.size() +
                                          node.target_indices.size());
  if (node.level >= policy.max_depth) {
    mark_leaf(tree, node_id, "max_depth", source_radii, target_radii);
    return;
  }
  if (total_count <= policy.leaf_size) {
    mark_leaf(tree, node_id, "leaf_size", source_radii, target_radii);
    return;
  }
  const double child_half_width = 0.5 * node.half_width;
  const double max_panel_radius =
      node_max_panel_radius(node, source_radii, target_radii);
  if (uses_panel_radius_stop(policy) && max_panel_radius > 0.0 &&
      child_half_width <
          policy.tree_panel_radius_factor * max_panel_radius) {
    mark_leaf(tree, node_id, "radius", source_radii, target_radii);
    return;
  }
  if (sparse_side_stop(node, policy)) {
    mark_leaf(tree, node_id, "occupancy", source_radii, target_radii);
    return;
  }

  std::array<std::vector<int>, 8> child_sources;
  std::array<std::vector<int>, 8> child_targets;
  for (const int index : node.source_indices) {
    child_sources[static_cast<std::size_t>(
                      octant_for(sources[static_cast<std::size_t>(index)],
                                 node.center))]
        .push_back(index);
  }
  for (const int index : node.target_indices) {
    child_targets[static_cast<std::size_t>(
                      octant_for(targets[static_cast<std::size_t>(index)],
                                 node.center))]
        .push_back(index);
  }

  bool created_child = false;
  for (int octant = 0; octant < 8; ++octant) {
    if (child_sources[static_cast<std::size_t>(octant)].empty() &&
        child_targets[static_cast<std::size_t>(octant)].empty()) {
      continue;
    }
    const int child_id = static_cast<int>(tree.nodes.size());
    tree.nodes[static_cast<std::size_t>(node_id)]
        .child[static_cast<std::size_t>(octant)] = child_id;
    KifmmHierarchyNode child;
    child.id = child_id;
    child.parent = node_id;
    child.level = node.level + 1;
    child.center = child_center(node.center, child_half_width, octant);
    child.half_width = child_half_width;
    child.source_indices =
        std::move(child_sources[static_cast<std::size_t>(octant)]);
    child.target_indices =
        std::move(child_targets[static_cast<std::size_t>(octant)]);
    tree.nodes.push_back(std::move(child));
    created_child = true;
  }

  tree.nodes[static_cast<std::size_t>(node_id)].is_leaf = !created_child;
  if (!created_child) {
    mark_leaf(tree, node_id, "unsplittable", source_radii, target_radii);
    return;
  }
  const std::array<int, 8> child_ids =
      tree.nodes[static_cast<std::size_t>(node_id)].child;
  for (const int child_id : child_ids) {
    if (child_id >= 0) {
      split_node(tree, child_id, sources, targets, source_radii, target_radii,
                 policy);
    }
  }
}

bool well_separated(const KifmmHierarchyNode &source,
                    const KifmmHierarchyNode &target,
                    const KifmmHierarchyPolicy &policy) noexcept {
  const double radius = std::sqrt(3.0) * source.half_width;
  const double distance = norm(sub(target.center, source.center));
  return distance > 0.0 && radius / distance <= policy.theta;
}

bool target_box_outside_source_check_box(
    const KifmmHierarchyNode &source, const KifmmHierarchyNode &target,
    const KifmmHierarchyPolicy &policy) noexcept {
  const double source_check_half_width =
      source.half_width * policy.upward_check_scale;
  const double dx = std::abs(target.center.x - source.center.x);
  const double dy = std::abs(target.center.y - source.center.y);
  const double dz = std::abs(target.center.z - source.center.z);
  const double target_half_width = target.half_width;
  return (dx - target_half_width > source_check_half_width) ||
         (dy - target_half_width > source_check_half_width) ||
         (dz - target_half_width > source_check_half_width);
}

bool has_source_children(const KifmmHierarchyNode &node,
                         const KifmmHierarchyTree &tree) {
  for (const int child_id : node.child) {
    if (child_id >= 0 &&
        !tree.nodes[static_cast<std::size_t>(child_id)].source_indices.empty()) {
      return true;
    }
  }
  return false;
}

bool has_target_children(const KifmmHierarchyNode &node,
                         const KifmmHierarchyTree &tree) {
  for (const int child_id : node.child) {
    if (child_id >= 0 &&
        !tree.nodes[static_cast<std::size_t>(child_id)].target_indices.empty()) {
      return true;
    }
  }
  return false;
}

void classify_pair(const KifmmHierarchyTree &tree,
                   const KifmmHierarchyPolicy &policy, int source_node_id,
                   int target_node_id, KifmmHierarchyWorklist &worklist) {
  const KifmmHierarchyNode &source =
      tree.nodes[static_cast<std::size_t>(source_node_id)];
  const KifmmHierarchyNode &target =
      tree.nodes[static_cast<std::size_t>(target_node_id)];
  if (source.source_indices.empty() || target.target_indices.empty()) {
    return;
  }
  if (well_separated(source, target, policy)) {
    if (target_box_outside_source_check_box(source, target, policy)) {
      worklist.m2l_pairs.push_back({target_node_id, source_node_id});
      ++worklist.mac_accepted_pairs;
      return;
    }
    ++worklist.check_guard_rejected_pairs;
  }

  const bool split_target =
      has_target_children(target, tree) &&
      (!has_source_children(source, tree) ||
       target.half_width >= source.half_width);
  if (split_target) {
    for (const int child_id : target.child) {
      if (child_id >= 0 &&
          !tree.nodes[static_cast<std::size_t>(child_id)]
               .target_indices.empty()) {
        classify_pair(tree, policy, source_node_id, child_id, worklist);
      }
    }
    return;
  }

  if (has_source_children(source, tree)) {
    for (const int child_id : source.child) {
      if (child_id >= 0 &&
          !tree.nodes[static_cast<std::size_t>(child_id)]
               .source_indices.empty()) {
        classify_pair(tree, policy, child_id, target_node_id, worklist);
      }
    }
    return;
  }

  worklist.direct_leaf_pairs.push_back({target_node_id, source_node_id});
  ++worklist.leaf_direct_pairs;
}

std::uint64_t hash_pairs(const KifmmHierarchyWorklist &worklist) {
  std::uint64_t hash = kFnvOffset;
  for (const auto &[target, source] : worklist.m2l_pairs) {
    hash_int(hash, target);
    hash_int(hash, source);
    hash_u64(hash, 0x1234ULL);
  }
  hash_u64(hash, 0xabcdefULL);
  for (const auto &[target, source] : worklist.direct_leaf_pairs) {
    hash_int(hash, target);
    hash_int(hash, source);
    hash_u64(hash, 0x5678ULL);
  }
  return hash == 0 ? 1 : hash;
}

} // namespace

KifmmHierarchyTree build_strict_octree(std::span<const Vec3> sources,
                                       std::span<const Vec3> targets,
                                       const KifmmHierarchyPolicy &policy) {
  return build_strict_octree(sources, targets, {}, {}, policy);
}

KifmmHierarchyTree build_strict_octree(
    std::span<const Vec3> sources, std::span<const Vec3> targets,
    std::span<const double> source_radii,
    std::span<const double> target_radii, const KifmmHierarchyPolicy &policy) {
  KifmmHierarchyTree tree;
  tree.stats.source_count = static_cast<int>(sources.size());
  tree.stats.target_count = static_cast<int>(targets.size());
  tree.stats.min_leaf_half_width_over_panel_radius =
      std::numeric_limits<double>::infinity();
  if (!valid_policy(policy)) {
    tree.status = "invalid_policy";
    return tree;
  }
  if ((!source_radii.empty() && source_radii.size() != sources.size()) ||
      (!target_radii.empty() && target_radii.size() != targets.size())) {
    tree.status = "invalid_radius_geometry";
    return tree;
  }
  if (sources.empty() || targets.empty()) {
    tree.status = "empty_geometry";
    return tree;
  }

  Vec3 min_corner{std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity()};
  Vec3 max_corner{-std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()};
  const auto include_point = [&](Vec3 p) {
    min_corner.x = std::min(min_corner.x, p.x);
    min_corner.y = std::min(min_corner.y, p.y);
    min_corner.z = std::min(min_corner.z, p.z);
    max_corner.x = std::max(max_corner.x, p.x);
    max_corner.y = std::max(max_corner.y, p.y);
    max_corner.z = std::max(max_corner.z, p.z);
  };
  for (const Vec3 point : sources) {
    if (!finite_vec(point)) {
      tree.status = "invalid_geometry";
      return tree;
    }
    include_point(point);
  }
  for (const Vec3 point : targets) {
    if (!finite_vec(point)) {
      tree.status = "invalid_geometry";
      return tree;
    }
    include_point(point);
  }

  const Vec3 center{0.5 * (min_corner.x + max_corner.x),
                    0.5 * (min_corner.y + max_corner.y),
                    0.5 * (min_corner.z + max_corner.z)};
  const double max_extent =
      std::max({max_corner.x - min_corner.x, max_corner.y - min_corner.y,
                max_corner.z - min_corner.z, 1.0e-9});

  KifmmHierarchyNode root;
  root.id = 0;
  root.parent = -1;
  root.level = 0;
  root.center = center;
  root.half_width = 0.5 * max_extent * (1.0 + 1.0e-12);
  root.source_indices.reserve(sources.size());
  root.target_indices.reserve(targets.size());
  for (std::size_t i = 0; i < sources.size(); ++i) {
    root.source_indices.push_back(static_cast<int>(i));
  }
  for (std::size_t i = 0; i < targets.size(); ++i) {
    root.target_indices.push_back(static_cast<int>(i));
  }
  tree.nodes.push_back(std::move(root));
  split_node(tree, 0, sources, targets, source_radii, target_radii, policy);
  update_stats(tree);
  if (!std::isfinite(tree.stats.min_leaf_half_width_over_panel_radius)) {
    tree.stats.min_leaf_half_width_over_panel_radius =
        std::numeric_limits<double>::quiet_NaN();
  }
  tree.tree_hash = hash_tree_nodes(tree.nodes);
  tree.status = "ok";
  return tree;
}

KifmmHierarchyWorklist
build_hierarchy_worklist(const KifmmHierarchyTree &tree,
                         const KifmmHierarchyPolicy &policy) {
  KifmmHierarchyWorklist worklist;
  worklist.tree_hash = tree.tree_hash;
  worklist.policy_hash = hash_policy(policy);
  if (tree.status != "ok" || tree.nodes.empty() || !valid_policy(policy)) {
    return worklist;
  }
  classify_pair(tree, policy, 0, 0, worklist);
  finalize_hierarchy_worklist(worklist, policy);
  return worklist;
}

void finalize_hierarchy_worklist(KifmmHierarchyWorklist &worklist,
                                 const KifmmHierarchyPolicy &policy) {
  worklist.policy_hash = hash_policy(policy);
  const auto by_target_source = [](const auto &a, const auto &b) {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    return a.second < b.second;
  };
  std::sort(worklist.m2l_pairs.begin(), worklist.m2l_pairs.end(),
            by_target_source);
  std::sort(worklist.direct_leaf_pairs.begin(),
            worklist.direct_leaf_pairs.end(), by_target_source);
  worklist.worklist_hash = hash_pairs(worklist);
}

} // namespace titan::tabi::fmm_kifmm
