#include "tabi_fmm_kifmm_sampled_panel_matvec.h"

#include "tabi_near.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace titan::tabi::fmm_kifmm {
namespace {

using TitanPBE::Tabi::JcpNearPairKernel;
using TitanPBE::Tabi::Panel;

struct SampledNode {
  std::vector<std::size_t> indices{};
  Vec3 center{};
  double half_width{0.0};
  double radius{0.0};
  std::size_t node_id{0};
  int level{0};
  int left{-1};
  int right{-1};
  std::array<int, 8> children{{-1, -1, -1, -1, -1, -1, -1, -1}};
};

struct SampledTree {
  std::vector<SampledNode> nodes{};
  std::vector<int> leaves{};
};

struct SampledWorkPair {
  int target_node{-1};
  int source_node{-1};
  bool far{false};
};

struct BoxBounds {
  Vec3 min{};
  Vec3 max{};
  Vec3 center{};
  double half_width{0.0};
};

struct ProductionNearStats {
  std::size_t valid_pairs{0};
  std::size_t invalid_pairs{0};
};

using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

std::size_t tree_memory_bytes(const SampledTree &tree) {
  std::size_t bytes = sizeof(SampledTree);
  bytes += tree.nodes.size() * sizeof(SampledNode);
  bytes += tree.leaves.size() * sizeof(int);
  for (const SampledNode &node : tree.nodes) {
    bytes += node.indices.capacity() * sizeof(std::size_t);
  }
  return bytes;
}

std::size_t equivalent_charge_bytes(
    const KifmmEquivalentSolveResult &result) noexcept {
  return result.charges.size() * sizeof(double);
}

std::size_t equivalent_charge_storage_bytes(
    const std::unordered_map<int, KifmmJcpChannels> &cache) {
  std::size_t bytes = 0;
  for (const auto &entry : cache) {
    const KifmmJcpChannels &channels = entry.second;
    bytes += equivalent_charge_bytes(channels.phin_laplace);
    bytes += equivalent_charge_bytes(channels.phin_yukawa);
    for (const KifmmEquivalentSolveResult &solve :
         channels.phi_normal_laplace) {
      bytes += equivalent_charge_bytes(solve);
    }
    for (const KifmmEquivalentSolveResult &solve :
         channels.phi_normal_yukawa) {
      bytes += equivalent_charge_bytes(solve);
    }
  }
  return bytes;
}

std::size_t channel_cache_memory_bytes(
    const std::unordered_map<int, KifmmJcpChannels> &cache) {
  return cache.size() * (sizeof(int) + sizeof(KifmmJcpChannels)) +
         equivalent_charge_storage_bytes(cache);
}

struct EquivalentCacheDiagnostics {
  int max_rank{0};
  double min_sigma_min{0.0};
  double max_sigma_max{0.0};
  double max_rhs_norm{0.0};
  double max_charge_norm{0.0};
  double max_charge_to_rhs_norm{0.0};
};

void update_equivalent_cache_diagnostics(
    EquivalentCacheDiagnostics &diag,
    const KifmmEquivalentSolveResult &solve) noexcept {
  diag.max_rank = std::max(diag.max_rank, solve.rank);
  if (solve.diagnostics.sigma_min > 0.0 &&
      std::isfinite(solve.diagnostics.sigma_min)) {
    if (diag.min_sigma_min == 0.0) {
      diag.min_sigma_min = solve.diagnostics.sigma_min;
    } else {
      diag.min_sigma_min =
          std::min(diag.min_sigma_min, solve.diagnostics.sigma_min);
    }
  }
  if (std::isfinite(solve.diagnostics.sigma_max)) {
    diag.max_sigma_max =
        std::max(diag.max_sigma_max, solve.diagnostics.sigma_max);
  }
  if (std::isfinite(solve.rhs_norm)) {
    diag.max_rhs_norm = std::max(diag.max_rhs_norm, solve.rhs_norm);
  }
  if (std::isfinite(solve.charge_norm)) {
    diag.max_charge_norm =
        std::max(diag.max_charge_norm, solve.charge_norm);
  }
  const double denom = std::max(solve.rhs_norm, 1.0e-300);
  if (std::isfinite(solve.charge_norm) && std::isfinite(denom)) {
    diag.max_charge_to_rhs_norm =
        std::max(diag.max_charge_to_rhs_norm, solve.charge_norm / denom);
  }
}

EquivalentCacheDiagnostics equivalent_cache_diagnostics(
    const std::unordered_map<int, KifmmJcpChannels> &cache) {
  EquivalentCacheDiagnostics diag;
  for (const auto &entry : cache) {
    const KifmmJcpChannels &channels = entry.second;
    update_equivalent_cache_diagnostics(diag, channels.phin_laplace);
    update_equivalent_cache_diagnostics(diag, channels.phin_yukawa);
    for (const KifmmEquivalentSolveResult &solve :
         channels.phi_normal_laplace) {
      update_equivalent_cache_diagnostics(diag, solve);
    }
    for (const KifmmEquivalentSolveResult &solve :
         channels.phi_normal_yukawa) {
      update_equivalent_cache_diagnostics(diag, solve);
    }
  }
  return diag;
}

void fill_equivalent_cache_geometry(
    KifmmSampledPanelMatvecResult &result,
    const KifmmEquivalentSystemCache &cache) {
  result.equivalent_system_cache_entries = cache.entries.size();
  result.equivalent_laplace_systems = cache.stats.laplace_systems;
  result.equivalent_yukawa_systems = cache.stats.yukawa_systems;
  const std::size_t lookups = cache.stats.hits + cache.stats.misses;
  result.equivalent_cache_hit_rate =
      lookups == 0 ? 0.0 : static_cast<double>(cache.stats.hits) /
                              static_cast<double>(lookups);
  std::vector<std::uint64_t> half_width_keys;
  std::vector<std::uint64_t> kappa_h_keys;
  half_width_keys.reserve(cache.entries.size());
  kappa_h_keys.reserve(cache.entries.size());
  result.equivalent_half_width_min =
      cache.entries.empty() ? 0.0 : std::numeric_limits<double>::infinity();
  result.equivalent_half_width_max = 0.0;
  for (const KifmmEquivalentSystemCacheEntry &entry : cache.entries) {
    half_width_keys.push_back(std::bit_cast<std::uint64_t>(entry.half_width));
    kappa_h_keys.push_back(
        std::bit_cast<std::uint64_t>(entry.kappa * entry.half_width));
    result.equivalent_half_width_min =
        std::min(result.equivalent_half_width_min, entry.half_width);
    result.equivalent_half_width_max =
        std::max(result.equivalent_half_width_max, entry.half_width);
  }
  std::sort(half_width_keys.begin(), half_width_keys.end());
  half_width_keys.erase(
      std::unique(half_width_keys.begin(), half_width_keys.end()),
      half_width_keys.end());
  std::sort(kappa_h_keys.begin(), kappa_h_keys.end());
  kappa_h_keys.erase(std::unique(kappa_h_keys.begin(), kappa_h_keys.end()),
                     kappa_h_keys.end());
  result.unique_equivalent_half_widths = half_width_keys.size();
  result.unique_equivalent_kappa_h_values = kappa_h_keys.size();
  result.equivalent_systems_per_source_node =
      result.unique_far_source_nodes == 0
          ? 0.0
          : static_cast<double>(cache.entries.size()) /
                static_cast<double>(result.unique_far_source_nodes);
  for (const KifmmEquivalentSystemCacheEntry &entry : cache.entries) {
    result.condition_computed =
        result.condition_computed ||
        entry.canonical_system.diagnostics.condition_computed;
    result.condition_gate_enabled =
        result.condition_gate_enabled ||
        entry.canonical_system.diagnostics.condition_gate_enabled;
  }
  if (!std::isfinite(result.equivalent_half_width_min)) {
    result.equivalent_half_width_min = 0.0;
  }
}

double coord(Vec3 p, int axis) noexcept {
  if (axis == 0) {
    return p.x;
  }
  if (axis == 1) {
    return p.y;
  }
  return p.z;
}

Vec3 position_of(const Panel &panel) noexcept {
  return {panel.x, panel.y, panel.z};
}

Vec3 normal_of(const Panel &panel) noexcept {
  return {panel.nx, panel.ny, panel.nz};
}

std::array<Vec3, 3> vertices_of(const Panel &panel) noexcept {
  return {{{panel.v0x, panel.v0y, panel.v0z},
           {panel.v1x, panel.v1y, panel.v1z},
           {panel.v2x, panel.v2y, panel.v2z}}};
}

Vec3 subtract(Vec3 a, Vec3 b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

double dot(Vec3 a, Vec3 b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(Vec3 v) noexcept {
  return std::sqrt(std::max(0.0, dot(v, v)));
}

bool finite_vec(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool valid_panel(const Panel &panel, bool require_unit_normals) noexcept {
  const Vec3 position = position_of(panel);
  const Vec3 normal = normal_of(panel);
  const auto vertices = vertices_of(panel);
  const double normal_norm = norm(normal);
  if (!finite_vec(position) || !finite_vec(normal) ||
      !finite_vec(vertices[0]) || !finite_vec(vertices[1]) ||
      !finite_vec(vertices[2]) || normal_norm <= 1.0e-12 ||
      !std::isfinite(panel.area) || panel.area <= 0.0 ||
      !panel.has_triangle_geometry) {
    return false;
  }
  if (require_unit_normals && std::abs(normal_norm - 1.0) > 1.0e-6) {
    return false;
  }
  const Vec3 e01 = subtract(vertices[1], vertices[0]);
  const Vec3 e02 = subtract(vertices[2], vertices[0]);
  const Vec3 cross{e01.y * e02.z - e01.z * e02.y,
                   e01.z * e02.x - e01.x * e02.z,
                   e01.x * e02.y - e01.y * e02.x};
  const double triangle_area = 0.5 * norm(cross);
  return triangle_area > 1.0e-14 && std::isfinite(triangle_area);
}

bool valid_unknowns(std::span<const double> unknowns) noexcept {
  for (const double value : unknowns) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

bool valid_target_indices(std::span<const std::size_t> target_indices,
                          std::size_t panel_count) {
  if (target_indices.empty()) {
    return false;
  }
  std::vector<std::size_t> sorted(target_indices.begin(),
                                  target_indices.end());
  std::sort(sorted.begin(), sorted.end());
  for (const std::size_t index : sorted) {
    if (index >= panel_count) {
      return false;
    }
  }
  return std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
}

BoxBounds tight_bounds(const std::vector<std::size_t> &indices,
                       std::span<const Panel> panels) {
  Vec3 min_p{std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
  Vec3 max_p{-std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()};
  for (const std::size_t index : indices) {
    const Vec3 p = position_of(panels[index]);
    min_p.x = std::min(min_p.x, p.x);
    min_p.y = std::min(min_p.y, p.y);
    min_p.z = std::min(min_p.z, p.z);
    max_p.x = std::max(max_p.x, p.x);
    max_p.y = std::max(max_p.y, p.y);
    max_p.z = std::max(max_p.z, p.z);
  }
  const double hx = (max_p.x - min_p.x) * 0.5;
  const double hy = (max_p.y - min_p.y) * 0.5;
  const double hz = (max_p.z - min_p.z) * 0.5;
  BoxBounds bounds;
  bounds.min = min_p;
  bounds.max = max_p;
  bounds.center = {(min_p.x + max_p.x) * 0.5, (min_p.y + max_p.y) * 0.5,
                   (min_p.z + max_p.z) * 0.5};
  bounds.half_width = std::max({hx, hy, hz, 1.0e-9}) * (1.0 + 1.0e-12);
  return bounds;
}

SampledNode make_node(const std::vector<std::size_t> &indices,
                      std::span<const Panel> panels, int level,
                      std::size_t node_id) {
  const BoxBounds bounds = tight_bounds(indices, panels);
  SampledNode node;
  node.indices = indices;
  node.center = bounds.center;
  node.half_width = bounds.half_width;
  node.radius = std::sqrt(3.0) * node.half_width;
  node.node_id = node_id;
  node.level = level;
  return node;
}

double enclosing_level_half_width(double root_half_width,
                                  double tight_half_width) noexcept {
  double h = std::max(root_half_width, tight_half_width);
  while (h * 0.5 >= tight_half_width && h > 1.0e-12) {
    h *= 0.5;
  }
  return std::max(h, 1.0e-12);
}

SampledNode make_level_stable_node(const std::vector<std::size_t> &indices,
                                   std::span<const Panel> panels,
                                   double root_half_width, int level,
                                   std::size_t node_id) {
  const BoxBounds bounds = tight_bounds(indices, panels);
  SampledNode node;
  node.indices = indices;
  node.center = bounds.center;
  node.half_width =
      enclosing_level_half_width(root_half_width, bounds.half_width);
  node.radius = std::sqrt(3.0) * node.half_width;
  node.node_id = node_id;
  node.level = level;
  return node;
}

int build_node(SampledTree &tree, std::vector<std::size_t> indices,
               std::span<const Panel> panels, int level, int leaf_size) {
  const int node_index = static_cast<int>(tree.nodes.size());
  SampledNode node =
      make_node(indices, panels, level, static_cast<std::size_t>(node_index));
  tree.nodes.push_back(std::move(node));
  if (static_cast<int>(indices.size()) <= leaf_size) {
    tree.leaves.push_back(node_index);
    return node_index;
  }

  Vec3 min_p{std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
  Vec3 max_p{-std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()};
  for (const std::size_t index : tree.nodes[node_index].indices) {
    const Vec3 p = position_of(panels[index]);
    min_p.x = std::min(min_p.x, p.x);
    min_p.y = std::min(min_p.y, p.y);
    min_p.z = std::min(min_p.z, p.z);
    max_p.x = std::max(max_p.x, p.x);
    max_p.y = std::max(max_p.y, p.y);
    max_p.z = std::max(max_p.z, p.z);
  }
  const double extent[3] = {max_p.x - min_p.x, max_p.y - min_p.y,
                            max_p.z - min_p.z};
  int axis = 0;
  if (extent[1] > extent[axis]) {
    axis = 1;
  }
  if (extent[2] > extent[axis]) {
    axis = 2;
  }
  std::sort(indices.begin(), indices.end(), [&](std::size_t a,
                                                std::size_t b) {
    const double ca = coord(position_of(panels[a]), axis);
    const double cb = coord(position_of(panels[b]), axis);
    if (ca != cb) {
      return ca < cb;
    }
    return a < b;
  });
  const auto mid = indices.begin() +
                   static_cast<std::ptrdiff_t>(indices.size() / 2);
  std::vector<std::size_t> left(indices.begin(), mid);
  std::vector<std::size_t> right(mid, indices.end());
  tree.nodes[node_index].left =
      build_node(tree, std::move(left), panels, level + 1, leaf_size);
  tree.nodes[node_index].right =
      build_node(tree, std::move(right), panels, level + 1, leaf_size);
  return node_index;
}

int build_level_stable_binary_node(SampledTree &tree,
                                   std::vector<std::size_t> indices,
                                   std::span<const Panel> panels,
                                   double root_half_width, int level,
                                   int leaf_size) {
  const int node_index = static_cast<int>(tree.nodes.size());
  SampledNode node = make_level_stable_node(
      indices, panels, root_half_width, level,
      static_cast<std::size_t>(node_index));
  tree.nodes.push_back(std::move(node));
  if (static_cast<int>(indices.size()) <= leaf_size) {
    tree.leaves.push_back(node_index);
    return node_index;
  }

  const BoxBounds bounds =
      tight_bounds(tree.nodes[static_cast<std::size_t>(node_index)].indices,
                   panels);
  const double extent[3] = {bounds.max.x - bounds.min.x,
                            bounds.max.y - bounds.min.y,
                            bounds.max.z - bounds.min.z};
  int axis = 0;
  if (extent[1] > extent[axis]) {
    axis = 1;
  }
  if (extent[2] > extent[axis]) {
    axis = 2;
  }
  std::sort(indices.begin(), indices.end(), [&](std::size_t a,
                                                std::size_t b) {
    const double ca = coord(position_of(panels[a]), axis);
    const double cb = coord(position_of(panels[b]), axis);
    if (ca != cb) {
      return ca < cb;
    }
    return a < b;
  });
  const auto mid = indices.begin() +
                   static_cast<std::ptrdiff_t>(indices.size() / 2);
  std::vector<std::size_t> left(indices.begin(), mid);
  std::vector<std::size_t> right(mid, indices.end());
  tree.nodes[static_cast<std::size_t>(node_index)].left =
      build_level_stable_binary_node(tree, std::move(left), panels,
                                     root_half_width, level + 1, leaf_size);
  tree.nodes[static_cast<std::size_t>(node_index)].right =
      build_level_stable_binary_node(tree, std::move(right), panels,
                                     root_half_width, level + 1, leaf_size);
  return node_index;
}

SampledTree build_tree(std::span<const Panel> panels, int leaf_size,
                       const KifmmLiteConfig &config) {
  SampledTree tree;
  std::vector<std::size_t> indices(panels.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  if (config.source_box_policy == "level-cube") {
    const BoxBounds root = tight_bounds(indices, panels);
    build_level_stable_binary_node(tree, std::move(indices), panels,
                                   root.half_width, 0, leaf_size);
  } else {
    build_node(tree, std::move(indices), panels, 0, leaf_size);
  }
  return tree;
}

bool accept_far(const SampledNode &target, const SampledNode &source,
                double theta) noexcept {
  const double center_distance = norm(subtract(target.center, source.center));
  const double separation =
      center_distance - target.radius - source.radius;
  if (!(separation > 0.0)) {
    return false;
  }
  return source.radius / separation <= theta;
}

bool target_node_outside_source_check_box(const SampledNode &target,
                                          const SampledNode &source,
                                          std::span<const Panel> target_panels,
                                          double check_scale) noexcept {
  const double h = source.half_width * check_scale;
  if (!(h > 0.0) || !std::isfinite(h)) {
    return false;
  }
  for (const std::size_t target_index : target.indices) {
    for (const Vec3 p : vertices_of(target_panels[target_index])) {
      if (std::abs(p.x - source.center.x) <= h &&
          std::abs(p.y - source.center.y) <= h &&
          std::abs(p.z - source.center.z) <= h) {
        return false;
      }
    }
  }
  return true;
}

bool share_vertex(const Panel &a, const Panel &b) noexcept {
  const std::size_t invalid = static_cast<std::size_t>(-1);
  const std::array<std::size_t, 3> av{a.v0_index, a.v1_index, a.v2_index};
  const std::array<std::size_t, 3> bv{b.v0_index, b.v1_index, b.v2_index};
  for (const std::size_t x : av) {
    if (x == invalid) {
      continue;
    }
    for (const std::size_t y : bv) {
      if (x == y) {
        return true;
      }
    }
  }
  return false;
}

bool node_pairs_share_vertex(const SampledNode &target,
                             const SampledNode &source,
                             std::span<const Panel> target_panels,
                             std::span<const Panel> source_panels) noexcept {
  for (const std::size_t ti : target.indices) {
    for (const std::size_t si : source.indices) {
      if (share_vertex(target_panels[ti], source_panels[si])) {
        return true;
      }
    }
  }
  return false;
}

void collect_work(const SampledTree &target_tree,
                  const SampledTree &source_tree, int target_node,
                  int source_node, double theta, double check_scale,
                  std::span<const Panel> target_panels,
                  std::span<const Panel> source_panels,
                  std::vector<SampledWorkPair> &pairs) {
  const SampledNode &target =
      target_tree.nodes[static_cast<std::size_t>(target_node)];
  const SampledNode &source =
      source_tree.nodes[static_cast<std::size_t>(source_node)];
  if (accept_far(target, source, theta) &&
      target_node_outside_source_check_box(target, source, target_panels,
                                           check_scale) &&
      !node_pairs_share_vertex(target, source, target_panels, source_panels)) {
    pairs.push_back({target_node, source_node, true});
    return;
  }
  bool has_octree_children = false;
  for (const int child : source.children) {
    if (child < 0) {
      continue;
    }
    has_octree_children = true;
    collect_work(target_tree, source_tree, target_node, child, theta,
                 check_scale, target_panels, source_panels, pairs);
  }
  if (has_octree_children) {
    return;
  }
  if (source.left < 0 || source.right < 0) {
    pairs.push_back({target_node, source_node, false});
    return;
  }
  collect_work(target_tree, source_tree, target_node, source.left, theta,
               check_scale, target_panels, source_panels, pairs);
  collect_work(target_tree, source_tree, target_node, source.right, theta,
               check_scale, target_panels, source_panels, pairs);
}

std::vector<SampledWorkPair>
build_worklist(const SampledTree &target_tree, const SampledTree &source_tree,
               const KifmmLiteConfig &config,
               std::span<const Panel> target_panels,
               std::span<const Panel> source_panels) {
  std::vector<SampledWorkPair> pairs;
  for (const int target_leaf : target_tree.leaves) {
    collect_work(target_tree, source_tree, target_leaf, 0,
                 config.opening_theta, config.check_scale, target_panels,
                 source_panels, pairs);
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const SampledWorkPair &a, const SampledWorkPair &b) {
              if (a.target_node != b.target_node) {
                return a.target_node < b.target_node;
              }
              if (a.source_node != b.source_node) {
                return a.source_node < b.source_node;
              }
              return a.far > b.far;
            });
  return pairs;
}

void hash_u64(std::uint64_t &hash, std::uint64_t value) noexcept {
  constexpr std::uint64_t prime = 1099511628211ull;
  hash ^= value;
  hash *= prime;
}

void hash_double(std::uint64_t &hash, double value) noexcept {
  hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

void hash_string(std::uint64_t &hash, const std::string &value) noexcept {
  hash_u64(hash, static_cast<std::uint64_t>(value.size()));
  for (const char c : value) {
    hash_u64(hash, static_cast<unsigned char>(c));
  }
}

void hash_vec3(std::uint64_t &hash, Vec3 value) noexcept {
  hash_double(hash, value.x);
  hash_double(hash, value.y);
  hash_double(hash, value.z);
}

std::uint64_t panel_geometry_hash(std::span<const Panel> panels) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  hash_u64(hash, static_cast<std::uint64_t>(panels.size()));
  for (const Panel &panel : panels) {
    hash_vec3(hash, position_of(panel));
    hash_vec3(hash, normal_of(panel));
    hash_double(hash, panel.area);
    hash_vec3(hash, {panel.v0x, panel.v0y, panel.v0z});
    hash_vec3(hash, {panel.v1x, panel.v1y, panel.v1z});
    hash_vec3(hash, {panel.v2x, panel.v2y, panel.v2z});
    hash_u64(hash, static_cast<std::uint64_t>(panel.v0_index));
    hash_u64(hash, static_cast<std::uint64_t>(panel.v1_index));
    hash_u64(hash, static_cast<std::uint64_t>(panel.v2_index));
    hash_u64(hash, panel.has_triangle_geometry ? 1ull : 0ull);
  }
  return hash;
}

std::uint64_t unknown_vector_hash(std::span<const double> unknowns) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  hash_u64(hash, static_cast<std::uint64_t>(unknowns.size()));
  for (const double value : unknowns) {
    hash_double(hash, value);
  }
  return hash;
}

std::uint64_t
sampled_target_hash(std::span<const std::size_t> indices) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  hash_u64(hash, static_cast<std::uint64_t>(indices.size()));
  for (const std::size_t index : indices) {
    hash_u64(hash, static_cast<std::uint64_t>(index));
  }
  return hash;
}

std::uint64_t
worklist_hash(const std::vector<SampledWorkPair> &pairs,
              const SampledTree &target_tree, const SampledTree &source_tree,
              std::span<const std::size_t> sampled_target_indices,
              const KifmmLiteConfig &config) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  hash_u64(hash, static_cast<std::uint64_t>(config.order));
  hash_u64(hash, static_cast<std::uint64_t>(config.leaf_size));
  hash_double(hash, config.opening_theta);
  hash_double(hash, config.equiv_scale);
  hash_double(hash, config.check_scale);
  hash_double(hash, config.kappa);
  hash_double(hash, config.eps_ratio);
  hash_double(hash, config.max_condition_estimate);
  hash_double(hash, config.max_equivalent_relative_residual);
  hash_double(hash, config.min_far_panel_fraction);
  hash_u64(hash, config.require_far_work ? 1ull : 0ull);
  hash_u64(hash, config.require_unit_normals ? 1ull : 0ull);
  hash_string(hash, config.solver);
  hash_string(hash, config.condition_policy);
  hash_string(hash, config.source_box_policy);
  hash_u64(hash, static_cast<std::uint64_t>(sampled_target_indices.size()));
  for (const std::size_t index : sampled_target_indices) {
    hash_u64(hash, static_cast<std::uint64_t>(index));
  }
  hash_u64(hash, static_cast<std::uint64_t>(pairs.size()));
  for (const SampledWorkPair &pair : pairs) {
    const SampledNode &target =
        target_tree.nodes[static_cast<std::size_t>(pair.target_node)];
    const SampledNode &source =
        source_tree.nodes[static_cast<std::size_t>(pair.source_node)];
    hash_u64(hash, static_cast<std::uint64_t>(pair.target_node));
    hash_u64(hash, static_cast<std::uint64_t>(pair.source_node));
    hash_u64(hash, pair.far ? 1ull : 0ull);
    hash_u64(hash, static_cast<std::uint64_t>(target.indices.size()));
    hash_u64(hash, static_cast<std::uint64_t>(source.indices.size()));
    hash_vec3(hash, target.center);
    hash_double(hash, target.half_width);
    hash_vec3(hash, source.center);
    hash_double(hash, source.half_width);
  }
  return hash;
}

std::uint64_t policy_hash(const KifmmRealPanelCloudConfig &config,
                          std::uint64_t worklist_hash_value) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  hash_u64(hash, worklist_hash_value);
  hash_u64(hash, static_cast<std::uint64_t>(config.lite.order));
  hash_u64(hash, static_cast<std::uint64_t>(config.lite.leaf_size));
  hash_double(hash, config.lite.opening_theta);
  hash_double(hash, config.lite.equiv_scale);
  hash_double(hash, config.lite.check_scale);
  hash_double(hash, config.lite.kappa);
  hash_double(hash, config.lite.eps_ratio);
  hash_double(hash, config.lite.max_condition_estimate);
  hash_double(hash, config.lite.max_equivalent_relative_residual);
  hash_double(hash, config.lite.max_aggregate_relative_rms);
  hash_double(hash, config.lite.max_channel_relative_rms);
  hash_double(hash, config.lite.min_far_panel_fraction);
  hash_u64(hash, config.lite.require_far_work ? 1ull : 0ull);
  hash_u64(hash, config.lite.require_unit_normals ? 1ull : 0ull);
  hash_string(hash, config.lite.solver);
  hash_string(hash, config.lite.condition_policy);
  hash_string(hash, config.lite.source_box_policy);
  hash_string(hash, config.near_integration);
  hash_string(hash, config.quad_buffer);
  return hash;
}

KifmmPanelSource source_from_panel(const Panel &panel, double phi,
                                   double phin) noexcept {
  return {position_of(panel), normal_of(panel), panel.area, phi, phin};
}

KifmmLiteTargetPanel target_from_panel(const Panel &panel) noexcept {
  return {position_of(panel), normal_of(panel)};
}

std::vector<KifmmPanelSource>
gather_sources(std::span<const KifmmPanelSource> sources,
               const SampledNode &node) {
  std::vector<KifmmPanelSource> subset;
  subset.reserve(node.indices.size());
  for (const std::size_t index : node.indices) {
    subset.push_back(sources[index]);
  }
  return subset;
}

KifmmBox node_box(const SampledNode &node) noexcept {
  return {node.center, node.half_width, node.level, node.node_id};
}

std::size_t tree_levels(const SampledTree &tree) {
  int max_level = -1;
  for (const SampledNode &node : tree.nodes) {
    max_level = std::max(max_level, node.level);
  }
  return max_level < 0 ? 0 : static_cast<std::size_t>(max_level + 1);
}

std::size_t unique_node_half_widths(const SampledTree &tree) {
  std::vector<std::uint64_t> values;
  values.reserve(tree.nodes.size());
  for (const SampledNode &node : tree.nodes) {
    values.push_back(std::bit_cast<std::uint64_t>(node.half_width));
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values.size();
}

void add_row(KifmmJcpRow &dst, const KifmmJcpRow &src) noexcept {
  dst.phi_row += src.phi_row;
  dst.phin_row += src.phin_row;
}

void add_channel_row(KifmmJcpChannelRow &dst,
                     const KifmmJcpChannelRow &src) noexcept {
  dst.k1 += src.k1;
  dst.k2 += src.k2;
  dst.k3 += src.k3;
  dst.k4 += src.k4;
}

KifmmJcpChannelRow production_channel_row(
    const Panel &target_panel, std::size_t target_original_index,
    std::span<const Panel> source_panels, std::span<const double> unknowns,
    const SampledNode &source_node, double eps_ratio, double kappa,
    const std::string &near_integration, const std::string &quad_buffer,
    ProductionNearStats *stats) {
  KifmmJcpChannelRow row;
  const std::size_t n = source_panels.size();
  for (const std::size_t source_index : source_node.indices) {
    const JcpNearPairKernel kernel =
        TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
            target_panel, target_original_index, source_panels[source_index],
            source_index, eps_ratio, kappa, true, near_integration,
            quad_buffer, nullptr);
    if (!kernel.valid) {
      if (stats != nullptr) {
        ++stats->invalid_pairs;
      }
      continue;
    }
    if (stats != nullptr) {
      ++stats->valid_pairs;
    }
    const double phi = unknowns[source_index];
    const double phin = unknowns[n + source_index];
    row.k1 += kernel.k1 * phin;
    row.k2 += kernel.k2 * phi;
    row.k3 += kernel.k3 * phin;
    row.k4 += kernel.k4 * phi;
  }
  return row;
}

KifmmJcpRow row_from_channel(const KifmmJcpChannelRow &channel) noexcept {
  return {channel.k1 + channel.k2, channel.k3 + channel.k4};
}

double rel_rms_rows(std::span<const KifmmJcpRow> approx,
                    std::span<const KifmmJcpRow> direct) noexcept {
  double err_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double d_phi = approx[i].phi_row - direct[i].phi_row;
    const double d_phin = approx[i].phin_row - direct[i].phin_row;
    err_sq += d_phi * d_phi + d_phin * d_phin;
    ref_sq += direct[i].phi_row * direct[i].phi_row +
              direct[i].phin_row * direct[i].phin_row;
  }
  return std::sqrt(err_sq / std::max(ref_sq, 1.0e-300));
}

double abs_rms_rows(std::span<const KifmmJcpRow> approx,
                    std::span<const KifmmJcpRow> direct) noexcept {
  double err_sq = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double d_phi = approx[i].phi_row - direct[i].phi_row;
    const double d_phin = approx[i].phin_row - direct[i].phin_row;
    err_sq += d_phi * d_phi + d_phin * d_phin;
  }
  const double denom =
      std::max(1.0, 2.0 * static_cast<double>(approx.size()));
  return std::sqrt(err_sq / denom);
}

enum class Channel { K1, K2, K3, K4 };

double channel_value(const KifmmJcpChannelRow &row,
                     Channel channel) noexcept {
  if (channel == Channel::K1) {
    return row.k1;
  }
  if (channel == Channel::K2) {
    return row.k2;
  }
  if (channel == Channel::K3) {
    return row.k3;
  }
  return row.k4;
}

double rel_rms_channel(std::span<const KifmmJcpChannelRow> approx,
                       std::span<const KifmmJcpChannelRow> direct,
                       Channel channel) noexcept {
  double err_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double a = channel_value(approx[i], channel);
    const double d = channel_value(direct[i], channel);
    const double delta = a - d;
    err_sq += delta * delta;
    ref_sq += d * d;
  }
  return std::sqrt(err_sq / std::max(ref_sq, 1.0e-300));
}

double abs_rms_channel(std::span<const KifmmJcpChannelRow> approx,
                       std::span<const KifmmJcpChannelRow> direct,
                       Channel channel) noexcept {
  double err_sq = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double delta =
        channel_value(approx[i], channel) - channel_value(direct[i], channel);
    err_sq += delta * delta;
  }
  const double denom = std::max(1.0, static_cast<double>(approx.size()));
  return std::sqrt(err_sq / denom);
}

double reference_rms_channel(std::span<const KifmmJcpChannelRow> direct,
                             Channel channel) noexcept {
  double ref_sq = 0.0;
  for (const KifmmJcpChannelRow &row : direct) {
    const double value = channel_value(row, channel);
    ref_sq += value * value;
  }
  const double denom = std::max(1.0, static_cast<double>(direct.size()));
  return std::sqrt(ref_sq / denom);
}

bool finite_rows(std::span<const KifmmJcpRow> rows) noexcept {
  for (const KifmmJcpRow &row : rows) {
    if (!std::isfinite(row.phi_row) || !std::isfinite(row.phin_row)) {
      return false;
    }
  }
  return true;
}

bool finite_channel_rows(std::span<const KifmmJcpChannelRow> rows) noexcept {
  for (const KifmmJcpChannelRow &row : rows) {
    if (!std::isfinite(row.k1) || !std::isfinite(row.k2) ||
        !std::isfinite(row.k3) || !std::isfinite(row.k4)) {
      return false;
    }
  }
  return true;
}

KifmmSampledPanelMatvecResult failed(std::string status, int order = 0) {
  KifmmSampledPanelMatvecResult result;
  result.order = order;
  result.status = std::move(status);
  result.lite.status = result.status;
  result.finite = false;
  result.lite.finite = false;
  return result;
}

} // namespace

KifmmSampledPanelMatvecResult run_kifmm_lite_sampled_panel_matvec(
    std::span<const Panel> panels, std::span<const double> unknowns,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmRealPanelCloudConfig &config) {
  const auto total_start = Clock::now();
  if (panels.empty()) {
    return failed("empty_cloud", config.lite.order);
  }
  if (unknowns.size() != 2 * panels.size() || !valid_unknowns(unknowns)) {
    return failed("invalid_unknowns", config.lite.order);
  }
  if (!valid_target_indices(sampled_target_indices, panels.size())) {
    return failed("invalid_target_indices", config.lite.order);
  }
  if (config.near_integration != "analytic_singular" ||
      config.quad_buffer != "off") {
    return failed("unsupported_near_policy", config.lite.order);
  }
  if (config.lite.order < 2 || (config.lite.order % 2) != 0 ||
      config.lite.leaf_size <= 0 ||
      !std::isfinite(config.lite.opening_theta) ||
      config.lite.opening_theta < 0.0 ||
      !std::isfinite(config.lite.equiv_scale) ||
      !std::isfinite(config.lite.check_scale) ||
      config.lite.equiv_scale <= 1.0 ||
      config.lite.equiv_scale >= config.lite.check_scale ||
      !std::isfinite(config.lite.kappa) || config.lite.kappa < 0.0 ||
      !std::isfinite(config.lite.eps_ratio) || config.lite.eps_ratio <= 0.0 ||
      !std::isfinite(config.lite.max_condition_estimate) ||
      config.lite.max_condition_estimate <= 0.0 ||
      !std::isfinite(config.lite.max_equivalent_relative_residual) ||
      config.lite.max_equivalent_relative_residual <= 0.0 ||
      !std::isfinite(config.lite.max_aggregate_relative_rms) ||
      config.lite.max_aggregate_relative_rms <= 0.0 ||
      !std::isfinite(config.lite.max_channel_relative_rms) ||
      config.lite.max_channel_relative_rms <= 0.0 ||
      !std::isfinite(config.lite.min_far_panel_fraction) ||
      config.lite.min_far_panel_fraction < 0.0 ||
      config.lite.min_far_panel_fraction > 1.0 ||
      !(config.lite.condition_policy == "svd-full-debug" ||
        config.lite.condition_policy == "svd-values" ||
        config.lite.condition_policy == "qr-rank-only" ||
        config.lite.condition_policy == "off") ||
      !(config.lite.source_box_policy == "tight-bbox" ||
        config.lite.source_box_policy == "level-cube")) {
    return failed("invalid_config", config.lite.order);
  }
  for (const Panel &panel : panels) {
    if (!valid_panel(panel, config.lite.require_unit_normals)) {
      return failed("invalid_panel", config.lite.order);
    }
  }

  std::vector<Panel> target_panels;
  target_panels.reserve(sampled_target_indices.size());
  for (const std::size_t index : sampled_target_indices) {
    target_panels.push_back(panels[index]);
  }
  for (const Panel &panel : target_panels) {
    if (!valid_panel(panel, config.lite.require_unit_normals)) {
      return failed("invalid_target_panel", config.lite.order);
    }
  }

  KifmmSampledPanelMatvecResult result;
  result.order = config.lite.order;
  result.source_box_policy = config.lite.source_box_policy;
  result.sampled_target_indices.assign(sampled_target_indices.begin(),
                                       sampled_target_indices.end());
  result.panel_geometry_hash = panel_geometry_hash(panels);
  result.unknown_vector_hash = unknown_vector_hash(unknowns);
  result.sampled_target_hash = sampled_target_hash(sampled_target_indices);
  result.lite.source_panels = panels.size();
  result.lite.target_panels = target_panels.size();
  result.lite.min_far_panel_fraction = config.lite.min_far_panel_fraction;
  result.lite.rows.assign(target_panels.size(), {});
  result.lite.channel_rows.assign(target_panels.size(), {});
  result.lite.direct_rows.assign(target_panels.size(), {});
  result.lite.direct_channel_rows.assign(target_panels.size(), {});
  auto finish_failure = [&](std::string status) {
    result.status = std::move(status);
    result.lite.status = result.status;
    result.finite = false;
    result.lite.finite = false;
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  };

  std::vector<KifmmPanelSource> sources;
  sources.reserve(panels.size());
  for (std::size_t i = 0; i < panels.size(); ++i) {
    sources.push_back(source_from_panel(panels[i], unknowns[i],
                                        unknowns[panels.size() + i]));
  }
  std::vector<KifmmLiteTargetPanel> targets;
  targets.reserve(target_panels.size());
  for (const Panel &panel : target_panels) {
    targets.push_back(target_from_panel(panel));
  }

  const double inv_eps_ratio = 1.0 / config.lite.eps_ratio;
  const std::size_t expected_pairs = panels.size() * target_panels.size();
  SampledNode all_sources;
  all_sources.indices.resize(panels.size());
  std::iota(all_sources.indices.begin(), all_sources.indices.end(),
            std::size_t{0});
  ProductionNearStats direct_reference_stats;
  const auto direct_reference_start = Clock::now();
  for (std::size_t target_local = 0; target_local < target_panels.size();
       ++target_local) {
    result.lite.direct_channel_rows[target_local] = production_channel_row(
        target_panels[target_local], sampled_target_indices[target_local],
        panels, unknowns, all_sources, config.lite.eps_ratio,
        config.lite.kappa, config.near_integration, config.quad_buffer,
        &direct_reference_stats);
    result.lite.direct_rows[target_local] =
        row_from_channel(result.lite.direct_channel_rows[target_local]);
  }
  result.direct_reference_seconds =
      elapsed_seconds(direct_reference_start, Clock::now());
  result.production_direct_reference_valid_pairs +=
      direct_reference_stats.valid_pairs;
  result.production_invalid_pairs += direct_reference_stats.invalid_pairs;
  if (result.production_invalid_pairs != 0) {
    return finish_failure("invalid_production_near_kernel");
  }
  if (result.production_direct_reference_valid_pairs != expected_pairs) {
    return finish_failure("direct_reference_coverage_gap");
  }
  if (!finite_rows(result.lite.direct_rows) ||
      !finite_channel_rows(result.lite.direct_channel_rows)) {
    return finish_failure("nonfinite_direct_reference");
  }

  const auto tree_build_start = Clock::now();
  const SampledTree source_tree =
      build_tree(panels, config.lite.leaf_size, config.lite);
  const int target_leaf_size = std::min(config.lite.leaf_size, 8);
  KifmmLiteConfig target_tree_config = config.lite;
  target_tree_config.source_box_policy = "tight-bbox";
  const SampledTree target_tree =
      build_tree(target_panels, target_leaf_size, target_tree_config);
  result.tree_build_seconds = elapsed_seconds(tree_build_start, Clock::now());
  result.source_tree_bytes = tree_memory_bytes(source_tree);
  result.target_tree_bytes = tree_memory_bytes(target_tree);
  result.lite.source_nodes = source_tree.nodes.size();
  result.lite.target_nodes = target_tree.nodes.size();
  result.lite.source_leaves = source_tree.leaves.size();
  result.lite.target_leaves = target_tree.leaves.size();
  result.lite.source_tree_levels = tree_levels(source_tree);
  result.lite.unique_source_node_half_widths =
      unique_node_half_widths(source_tree);
  result.lite.source_root_half_width =
      source_tree.nodes.empty() ? 0.0 : source_tree.nodes.front().half_width;
  result.lite.level_cube_geometry =
      config.lite.source_box_policy == "level-cube";
  result.source_tree_levels = result.lite.source_tree_levels;
  result.unique_source_node_half_widths =
      result.lite.unique_source_node_half_widths;
  result.source_root_half_width = result.lite.source_root_half_width;
  result.level_cube_geometry = result.lite.level_cube_geometry;

  const auto worklist_build_start = Clock::now();
  const std::vector<SampledWorkPair> pairs =
      build_worklist(target_tree, source_tree, config.lite, target_panels,
                     panels);
  result.worklist_build_seconds =
      elapsed_seconds(worklist_build_start, Clock::now());
  result.worklist_bytes = pairs.capacity() * sizeof(SampledWorkPair);
  result.lite.worklist_hash =
      worklist_hash(pairs, target_tree, source_tree, sampled_target_indices,
                    config.lite);
  result.policy_hash = policy_hash(config, result.lite.worklist_hash);

  const KifmmSurfaceRule surface_rule =
      make_cube_surface_rule(config.lite.order, config.lite.equiv_scale,
                             config.lite.check_scale);
  std::unordered_map<int, KifmmJcpChannels> channel_cache;
  KifmmEquivalentSystemCache equivalent_system_cache;

  for (const SampledWorkPair &pair : pairs) {
    const SampledNode &target_node =
        target_tree.nodes[static_cast<std::size_t>(pair.target_node)];
    const SampledNode &source_node =
        source_tree.nodes[static_cast<std::size_t>(pair.source_node)];
    const std::size_t panel_pairs =
        target_node.indices.size() * source_node.indices.size();
    if (pair.far) {
      ++result.lite.accepted_far_cell_pairs;
      result.lite.accepted_far_panel_pairs += panel_pairs;
      auto channel_it = channel_cache.find(pair.source_node);
      if (channel_it == channel_cache.end()) {
        const KifmmBox box = node_box(source_node);
        const std::vector<KifmmPanelSource> subset =
            gather_sources(sources, source_node);
        const auto equivalent_start = Clock::now();
        const KifmmJcpChannels channels = assemble_kifmm_jcp_channels(
            surface_rule, box, subset, config.lite.kappa,
            config.lite.eps_ratio, inv_eps_ratio, config.lite.solver,
            &equivalent_system_cache, config.lite.condition_policy);
        result.equivalent_build_solve_seconds +=
            elapsed_seconds(equivalent_start, Clock::now());
        result.equivalent_matrix_build_seconds +=
            channels.equivalent_matrix_build_seconds;
        result.equivalent_condition_seconds +=
            channels.equivalent_condition_seconds;
        result.equivalent_factor_seconds += channels.equivalent_factor_seconds;
        result.equivalent_rhs_build_seconds +=
            channels.equivalent_rhs_build_seconds;
        result.equivalent_solve_seconds += channels.equivalent_solve_seconds;
        result.equivalent_factor_cache_hits +=
            channels.equivalent_factor_cache_hits;
        result.equivalent_factor_cache_misses +=
            channels.equivalent_factor_cache_misses;
        result.lite.equivalent_system_builds +=
            channels.equivalent_system_builds;
        result.lite.equivalent_rhs_solves += channels.equivalent_rhs_solves;
        channel_it = channel_cache.emplace(pair.source_node, channels).first;
      }
      const KifmmJcpChannels &channels = channel_it->second;
      result.lite.max_condition_estimate = std::max(
          result.lite.max_condition_estimate,
          channels.max_condition_estimate);
      result.lite.max_equivalent_relative_residual =
          std::max(result.lite.max_equivalent_relative_residual,
                   channels.max_relative_residual);
      if (!channels.finite || channels.status != "pass") {
        return finish_failure("far_equivalent_failure");
      }
      if (channels.max_condition_estimate >
          config.lite.max_condition_estimate) {
        return finish_failure("ill_conditioned_equivalent_system");
      }
      if (channels.max_relative_residual >
          config.lite.max_equivalent_relative_residual) {
        return finish_failure("equivalent_residual_too_large");
      }
      const KifmmBox box = node_box(source_node);
      const auto far_eval_start = Clock::now();
      for (const std::size_t target_local : target_node.indices) {
        const KifmmJcpChannelRow channel = assemble_kifmm_jcp_channel_row(
            surface_rule, box, channels, config.lite.kappa,
            config.lite.eps_ratio, inv_eps_ratio, targets[target_local].position,
            targets[target_local].normal);
        add_channel_row(result.lite.channel_rows[target_local], channel);
        add_row(result.lite.rows[target_local], row_from_channel(channel));
      }
      result.far_eval_seconds +=
          elapsed_seconds(far_eval_start, Clock::now());
    } else {
      ++result.lite.direct_fallback_cell_pairs;
      result.lite.direct_fallback_panel_pairs += panel_pairs;
      ProductionNearStats fallback_stats;
      const auto fallback_eval_start = Clock::now();
      for (const std::size_t target_local : target_node.indices) {
        const KifmmJcpChannelRow channel = production_channel_row(
            target_panels[target_local], sampled_target_indices[target_local],
            panels, unknowns, source_node, config.lite.eps_ratio,
            config.lite.kappa, config.near_integration, config.quad_buffer,
            &fallback_stats);
        add_channel_row(result.lite.channel_rows[target_local], channel);
        add_row(result.lite.rows[target_local], row_from_channel(channel));
      }
      result.fallback_eval_seconds +=
          elapsed_seconds(fallback_eval_start, Clock::now());
      result.production_fallback_valid_pairs += fallback_stats.valid_pairs;
      result.production_invalid_pairs += fallback_stats.invalid_pairs;
      if (result.production_invalid_pairs != 0) {
        return finish_failure("invalid_production_near_kernel");
      }
    }
  }

  if (result.nonproduction_fallback_pairs != 0) {
    return finish_failure("nonproduction_fallback_used");
  }
  if (result.lite.accepted_far_panel_pairs +
          result.lite.direct_fallback_panel_pairs != expected_pairs) {
    return finish_failure("worklist_coverage_gap");
  }
  if (result.production_fallback_valid_pairs !=
      result.lite.direct_fallback_panel_pairs) {
    return finish_failure("fallback_coverage_gap");
  }

  result.unique_far_source_nodes = channel_cache.size();
  result.cached_source_nodes = channel_cache.size();
  fill_equivalent_cache_geometry(result, equivalent_system_cache);
  if (!result.condition_computed) {
    result.lite.max_condition_estimate = -1.0;
  }
  result.equivalent_charge_storage_bytes =
      equivalent_charge_storage_bytes(channel_cache);
  result.cached_channel_object_estimated_bytes =
      channel_cache_memory_bytes(channel_cache);
  const EquivalentCacheDiagnostics equivalent_diag =
      equivalent_cache_diagnostics(channel_cache);
  result.max_equivalent_rank = equivalent_diag.max_rank;
  result.min_equivalent_sigma_min = equivalent_diag.min_sigma_min;
  result.max_equivalent_sigma_max = equivalent_diag.max_sigma_max;
  result.max_equivalent_rhs_norm = equivalent_diag.max_rhs_norm;
  result.max_equivalent_charge_norm = equivalent_diag.max_charge_norm;
  result.max_equivalent_charge_to_rhs_norm =
      equivalent_diag.max_charge_to_rhs_norm;

  result.aggregate_abs_rms =
      abs_rms_rows(result.lite.rows, result.lite.direct_rows);
  result.k1_abs_rms =
      abs_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K1);
  result.k2_abs_rms =
      abs_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K2);
  result.k3_abs_rms =
      abs_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K3);
  result.k4_abs_rms =
      abs_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K4);
  result.k1_reference_rms =
      reference_rms_channel(result.lite.direct_channel_rows, Channel::K1);
  result.k2_reference_rms =
      reference_rms_channel(result.lite.direct_channel_rows, Channel::K2);
  result.k3_reference_rms =
      reference_rms_channel(result.lite.direct_channel_rows, Channel::K3);
  result.k4_reference_rms =
      reference_rms_channel(result.lite.direct_channel_rows, Channel::K4);
  result.lite.aggregate_relative_rms =
      rel_rms_rows(result.lite.rows, result.lite.direct_rows);
  result.lite.k1_relative_rms =
      rel_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K1);
  result.lite.k2_relative_rms =
      rel_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K2);
  result.lite.k3_relative_rms =
      rel_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K3);
  result.lite.k4_relative_rms =
      rel_rms_channel(result.lite.channel_rows, result.lite.direct_channel_rows,
                      Channel::K4);

  if (!finite_rows(result.lite.rows) ||
      !finite_channel_rows(result.lite.channel_rows)) {
    return finish_failure("nonfinite_kifmm_rows");
  }
  if (result.lite.aggregate_relative_rms >
      config.lite.max_aggregate_relative_rms) {
    return finish_failure("aggregate_rms_gate_failed");
  }
  const double max_channel_rms =
      std::max({result.lite.k1_relative_rms, result.lite.k2_relative_rms,
                result.lite.k3_relative_rms, result.lite.k4_relative_rms});
  if (max_channel_rms > config.lite.max_channel_relative_rms) {
    return finish_failure("channel_rms_gate_failed");
  }
  if (config.lite.require_far_work &&
      (result.lite.accepted_far_cell_pairs == 0 ||
       result.lite.accepted_far_panel_pairs == 0)) {
    return finish_failure("no_far_work");
  }
  if (config.lite.require_far_work &&
      config.lite.min_far_panel_fraction > 0.0) {
    const std::size_t total_panel_pairs =
        result.lite.accepted_far_panel_pairs +
        result.lite.direct_fallback_panel_pairs;
    const double far_fraction =
        total_panel_pairs == 0
            ? 0.0
            : static_cast<double>(result.lite.accepted_far_panel_pairs) /
                  static_cast<double>(total_panel_pairs);
    if (far_fraction < config.lite.min_far_panel_fraction) {
      return finish_failure("fallback_dominated");
    }
  }

  result.lite.finite = std::isfinite(result.lite.aggregate_relative_rms) &&
                       std::isfinite(result.lite.k1_relative_rms) &&
                       std::isfinite(result.lite.k2_relative_rms) &&
                       std::isfinite(result.lite.k3_relative_rms) &&
                       std::isfinite(result.lite.k4_relative_rms);
  result.finite = result.lite.finite;
  result.used_nonproduction_fallback = result.nonproduction_fallback_pairs > 0;
  result.status = result.finite ? "pass" : "non_finite";
  if (result.finite && result.lite.accepted_far_cell_pairs == 0 &&
      result.lite.accepted_far_panel_pairs == 0 &&
      !config.lite.require_far_work) {
    result.status = "direct_only_reference";
  }
  result.total_seconds = elapsed_seconds(total_start, Clock::now());
  result.lite.status = result.status;
  return result;
}

} // namespace titan::tabi::fmm_kifmm
