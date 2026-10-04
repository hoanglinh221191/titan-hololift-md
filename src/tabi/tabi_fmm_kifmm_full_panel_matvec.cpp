#include "tabi_fmm_kifmm_full_panel_matvec.h"

#include "tabi_near.h"
#include "titan_executor.h"

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
#include <stdexcept>
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

struct TargetNodeTask {
  int target_node{-1};
  std::size_t begin_pair{0};
  std::size_t end_pair{0};
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
    KifmmFullPanelMatvecResult &result,
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

std::uint64_t full_output_hash_phi(
    std::span<const KifmmJcpRow> rows) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  hash_u64(hash, static_cast<std::uint64_t>(rows.size()));
  for (const KifmmJcpRow &row : rows) {
    hash_double(hash, row.phi_row);
  }
  return hash;
}

std::uint64_t full_output_hash_phin(
    std::span<const KifmmJcpRow> rows) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  hash_u64(hash, static_cast<std::uint64_t>(rows.size()));
  for (const KifmmJcpRow &row : rows) {
    hash_double(hash, row.phin_row);
  }
  return hash;
}

std::uint64_t full_channel_output_hash(
    std::span<const KifmmJcpChannelRow> rows) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  hash_u64(hash, static_cast<std::uint64_t>(rows.size()));
  for (const KifmmJcpChannelRow &row : rows) {
    hash_double(hash, row.k1);
    hash_double(hash, row.k2);
    hash_double(hash, row.k3);
    hash_double(hash, row.k4);
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
  hash_string(hash, config.full_apply_policy);
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

KifmmFullPanelMatvecResult failed(std::string status, int order = 0) {
  KifmmFullPanelMatvecResult result;
  result.order = order;
  result.status = std::move(status);
  result.lite.status = result.status;
  result.finite = false;
  result.lite.finite = false;
  return result;
}

} // namespace

KifmmFullPanelMatvecResult run_kifmm_lite_full_panel_matvec(
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
        config.lite.source_box_policy == "level-cube") ||
      !(config.full_apply_policy == "serial" ||
        config.full_apply_policy == "target-node-parallel")) {
    return failed("invalid_config", config.lite.order);
  }
  for (const Panel &panel : panels) {
    if (!valid_panel(panel, config.lite.require_unit_normals)) {
      return failed("invalid_panel", config.lite.order);
    }
  }

  std::vector<Panel> target_panels(panels.begin(), panels.end());
  for (const Panel &panel : target_panels) {
    if (!valid_panel(panel, config.lite.require_unit_normals)) {
      return failed("invalid_target_panel", config.lite.order);
    }
  }

  KifmmFullPanelMatvecResult result;
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
  result.lite.direct_rows.assign(sampled_target_indices.size(), {});
  result.lite.direct_channel_rows.assign(sampled_target_indices.size(), {});
  auto finish_failure = [&](std::string status) {
    result.status = std::move(status);
    result.lite.status = result.status;
    result.finite = false;
    result.lite.finite = false;
    result.setup_seconds = result.tree_build_seconds +
                           result.worklist_build_seconds +
                           result.equivalent_build_solve_seconds;
    result.apply_seconds =
        result.full_far_eval_seconds + result.full_fallback_eval_seconds;
    result.first_apply_seconds = result.setup_seconds + result.apply_seconds;
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
  const std::size_t expected_full_pairs = panels.size() * target_panels.size();
  const std::size_t expected_sampled_pairs =
      panels.size() * sampled_target_indices.size();
  SampledNode all_sources;
  all_sources.indices.resize(panels.size());
  std::iota(all_sources.indices.begin(), all_sources.indices.end(),
            std::size_t{0});
  ProductionNearStats direct_reference_stats;
  const auto direct_reference_start = Clock::now();
  for (std::size_t target_local = 0;
       target_local < sampled_target_indices.size(); ++target_local) {
    const std::size_t target_index = sampled_target_indices[target_local];
    result.lite.direct_channel_rows[target_local] = production_channel_row(
        target_panels[target_index], target_index, panels, unknowns,
        all_sources, config.lite.eps_ratio, config.lite.kappa,
        config.near_integration, config.quad_buffer, &direct_reference_stats);
    result.lite.direct_rows[target_local] =
        row_from_channel(result.lite.direct_channel_rows[target_local]);
  }
  result.sampled_dense_audit_seconds =
      elapsed_seconds(direct_reference_start, Clock::now());
  result.sampled_direct_reference_valid_pairs +=
      direct_reference_stats.valid_pairs;
  result.sampled_invalid_pairs += direct_reference_stats.invalid_pairs;
  if (result.sampled_invalid_pairs != 0) {
    return finish_failure("invalid_production_near_kernel");
  }
  if (result.sampled_direct_reference_valid_pairs != expected_sampled_pairs) {
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
      worklist_hash(pairs, target_tree, source_tree,
                    std::span<const std::size_t>{}, config.lite);
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
      result.full_far_eval_seconds +=
          elapsed_seconds(far_eval_start, Clock::now());
    } else {
      ++result.lite.direct_fallback_cell_pairs;
      result.lite.direct_fallback_panel_pairs += panel_pairs;
      ProductionNearStats fallback_stats;
      const auto fallback_eval_start = Clock::now();
      for (const std::size_t target_local : target_node.indices) {
        const KifmmJcpChannelRow channel = production_channel_row(
            target_panels[target_local], target_local, panels, unknowns,
            source_node, config.lite.eps_ratio,
            config.lite.kappa, config.near_integration, config.quad_buffer,
            &fallback_stats);
        add_channel_row(result.lite.channel_rows[target_local], channel);
        add_row(result.lite.rows[target_local], row_from_channel(channel));
      }
      result.full_fallback_eval_seconds +=
          elapsed_seconds(fallback_eval_start, Clock::now());
      result.production_fallback_valid_pairs += fallback_stats.valid_pairs;
      result.sampled_invalid_pairs += fallback_stats.invalid_pairs;
      if (result.sampled_invalid_pairs != 0) {
        return finish_failure("invalid_production_near_kernel");
      }
    }
  }

  if (result.nonproduction_fallback_pairs != 0) {
    return finish_failure("nonproduction_fallback_used");
  }
  if (result.lite.accepted_far_panel_pairs +
          result.lite.direct_fallback_panel_pairs != expected_full_pairs) {
    return finish_failure("worklist_coverage_gap");
  }
  result.full_pair_coverage =
      result.lite.accepted_far_panel_pairs +
      result.lite.direct_fallback_panel_pairs;
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

  std::vector<KifmmJcpRow> sampled_rows;
  std::vector<KifmmJcpChannelRow> sampled_channel_rows;
  sampled_rows.reserve(sampled_target_indices.size());
  sampled_channel_rows.reserve(sampled_target_indices.size());
  for (const std::size_t target_index : sampled_target_indices) {
    sampled_rows.push_back(result.lite.rows[target_index]);
    sampled_channel_rows.push_back(result.lite.channel_rows[target_index]);
  }

  result.aggregate_abs_rms = abs_rms_rows(sampled_rows, result.lite.direct_rows);
  result.full_output_hash_phi = full_output_hash_phi(result.lite.rows);
  result.full_output_hash_phin = full_output_hash_phin(result.lite.rows);
  result.full_channel_output_hash =
      full_channel_output_hash(result.lite.channel_rows);
  result.k1_abs_rms =
      abs_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                      Channel::K1);
  result.k2_abs_rms =
      abs_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                      Channel::K2);
  result.k3_abs_rms =
      abs_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                      Channel::K3);
  result.k4_abs_rms =
      abs_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                      Channel::K4);
  result.k1_reference_rms =
      reference_rms_channel(result.lite.direct_channel_rows, Channel::K1);
  result.k2_reference_rms =
      reference_rms_channel(result.lite.direct_channel_rows, Channel::K2);
  result.k3_reference_rms =
      reference_rms_channel(result.lite.direct_channel_rows, Channel::K3);
  result.k4_reference_rms =
      reference_rms_channel(result.lite.direct_channel_rows, Channel::K4);
  result.aggregate_relative_rms = rel_rms_rows(sampled_rows, result.lite.direct_rows);
  result.k1_relative_rms =
      rel_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                      Channel::K1);
  result.k2_relative_rms =
      rel_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                      Channel::K2);
  result.k3_relative_rms =
      rel_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                      Channel::K3);
  result.k4_relative_rms =
      rel_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                      Channel::K4);
  result.lite.aggregate_relative_rms = result.aggregate_relative_rms;
  result.lite.k1_relative_rms = result.k1_relative_rms;
  result.lite.k2_relative_rms = result.k2_relative_rms;
  result.lite.k3_relative_rms = result.k3_relative_rms;
  result.lite.k4_relative_rms = result.k4_relative_rms;

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
  result.setup_seconds = result.tree_build_seconds +
                         result.worklist_build_seconds +
                         result.equivalent_build_solve_seconds;
  result.apply_seconds =
      result.full_far_eval_seconds + result.full_fallback_eval_seconds;
  result.first_apply_seconds = result.setup_seconds + result.apply_seconds;
  result.total_seconds = elapsed_seconds(total_start, Clock::now());
  result.lite.status = result.status;
  return result;
}

struct KifmmFullPanelPreparedState {
  std::vector<Panel> panels{};
  std::vector<Panel> target_panels{};
  std::vector<std::size_t> sampled_target_indices{};
  std::vector<KifmmLiteTargetPanel> targets{};
  KifmmRealPanelCloudConfig config{};
  SampledTree source_tree{};
  SampledTree target_tree{};
  std::vector<SampledWorkPair> pairs{};
  KifmmSurfaceRule surface_rule{};
  KifmmEquivalentSystemCache equivalent_system_cache{};
  std::uint64_t panel_geometry_hash{0};
  std::uint64_t sampled_target_hash{0};
  std::uint64_t worklist_hash{0};
  std::uint64_t policy_hash{0};
  std::size_t source_tree_bytes{0};
  std::size_t target_tree_bytes{0};
  std::size_t worklist_bytes{0};
  std::size_t source_tree_levels{0};
  std::size_t unique_source_node_half_widths{0};
  double source_root_half_width{0.0};
  bool level_cube_geometry{false};
  std::size_t prepared_far_source_nodes{0};
  double tree_build_seconds{0.0};
  double worklist_build_seconds{0.0};
  double equivalent_prepare_seconds{0.0};
};

void require_preparable_full_panel_operator(
    std::span<const Panel> panels,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmRealPanelCloudConfig &config) {
  if (panels.empty()) {
    throw std::runtime_error("empty_cloud");
  }
  if (!valid_target_indices(sampled_target_indices, panels.size())) {
    throw std::runtime_error("invalid_target_indices");
  }
  if (config.near_integration != "analytic_singular" ||
      config.quad_buffer != "off") {
    throw std::runtime_error("unsupported_near_policy");
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
        config.lite.source_box_policy == "level-cube") ||
      !(config.full_apply_policy == "serial" ||
        config.full_apply_policy == "target-node-parallel")) {
    throw std::runtime_error("invalid_config");
  }
  for (const Panel &panel : panels) {
    if (!valid_panel(panel, config.lite.require_unit_normals)) {
      throw std::runtime_error("invalid_panel");
    }
  }
}

std::vector<int> unique_far_source_nodes(
    std::span<const SampledWorkPair> pairs) {
  std::vector<int> nodes;
  nodes.reserve(pairs.size());
  for (const SampledWorkPair &pair : pairs) {
    if (pair.far) {
      nodes.push_back(pair.source_node);
    }
  }
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  return nodes;
}

std::vector<TargetNodeTask> group_target_node_tasks(
    std::span<const SampledWorkPair> pairs) {
  std::vector<TargetNodeTask> tasks;
  std::size_t begin = 0;
  while (begin < pairs.size()) {
    const int target_node = pairs[begin].target_node;
    std::size_t end = begin + 1;
    while (end < pairs.size() && pairs[end].target_node == target_node) {
      ++end;
    }
    tasks.push_back({target_node, begin, end});
    begin = end;
  }
  return tasks;
}

void prebuild_equivalent_system_cache(KifmmFullPanelPreparedState &state) {
  const auto start = Clock::now();
  const std::vector<int> far_source_nodes = unique_far_source_nodes(state.pairs);
  for (const int source_node_index : far_source_nodes) {
    const SampledNode &source_node =
        state.source_tree.nodes[static_cast<std::size_t>(source_node_index)];
    const KifmmBox box = node_box(source_node);
    const KifmmEquivalentSystem laplace = build_equivalent_system_cached(
        state.equivalent_system_cache, state.surface_rule, box,
        KifmmKernel::Laplace, 0.0, state.config.lite.solver,
        state.config.lite.condition_policy);
    const KifmmEquivalentSystem yukawa = build_equivalent_system_cached(
        state.equivalent_system_cache, state.surface_rule, box,
        KifmmKernel::Yukawa, state.config.lite.kappa,
        state.config.lite.solver, state.config.lite.condition_policy);
    if (!laplace.finite || !yukawa.finite) {
      throw std::runtime_error("equivalent_prepare_failure");
    }
  }
  state.prepared_far_source_nodes = far_source_nodes.size();
  state.equivalent_prepare_seconds = elapsed_seconds(start, Clock::now());
}

KifmmFullPanelPreparedOperator prepare_kifmm_lite_full_panel_operator(
    std::span<const Panel> panels,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmRealPanelCloudConfig &config) {
  const auto start = Clock::now();
  require_preparable_full_panel_operator(panels, sampled_target_indices,
                                         config);

  auto state = std::make_shared<KifmmFullPanelPreparedState>();
  state->panels.assign(panels.begin(), panels.end());
  state->target_panels.assign(panels.begin(), panels.end());
  state->sampled_target_indices.assign(sampled_target_indices.begin(),
                                       sampled_target_indices.end());
  state->config = config;
  state->panel_geometry_hash = panel_geometry_hash(state->panels);
  state->sampled_target_hash =
      sampled_target_hash(state->sampled_target_indices);
  state->targets.reserve(state->target_panels.size());
  for (const Panel &panel : state->target_panels) {
    state->targets.push_back(target_from_panel(panel));
  }

  const auto tree_start = Clock::now();
  state->source_tree =
      build_tree(state->panels, state->config.lite.leaf_size,
                 state->config.lite);
  const int target_leaf_size = std::min(state->config.lite.leaf_size, 8);
  KifmmLiteConfig target_tree_config = state->config.lite;
  target_tree_config.source_box_policy = "tight-bbox";
  state->target_tree =
      build_tree(state->target_panels, target_leaf_size, target_tree_config);
  state->tree_build_seconds = elapsed_seconds(tree_start, Clock::now());
  state->source_tree_bytes = tree_memory_bytes(state->source_tree);
  state->target_tree_bytes = tree_memory_bytes(state->target_tree);
  state->source_tree_levels = tree_levels(state->source_tree);
  state->unique_source_node_half_widths =
      unique_node_half_widths(state->source_tree);
  state->source_root_half_width = state->source_tree.nodes.empty()
                                      ? 0.0
                                      : state->source_tree.nodes.front().half_width;
  state->level_cube_geometry =
      state->config.lite.source_box_policy == "level-cube";

  const auto worklist_start = Clock::now();
  state->pairs = build_worklist(state->target_tree, state->source_tree,
                                state->config.lite, state->target_panels,
                                state->panels);
  state->worklist_build_seconds = elapsed_seconds(worklist_start, Clock::now());
  state->worklist_bytes = state->pairs.capacity() * sizeof(SampledWorkPair);
  state->worklist_hash =
      worklist_hash(state->pairs, state->target_tree, state->source_tree,
                    state->sampled_target_indices, state->config.lite);
  state->policy_hash = policy_hash(state->config, state->worklist_hash);
  state->surface_rule =
      make_cube_surface_rule(state->config.lite.order,
                             state->config.lite.equiv_scale,
                             state->config.lite.check_scale);
  prebuild_equivalent_system_cache(*state);

  KifmmFullPanelPreparedOperator prepared;
  prepared.panels = state->panels;
  prepared.sampled_target_indices = state->sampled_target_indices;
  prepared.config = state->config;
  prepared.panel_geometry_hash = state->panel_geometry_hash;
  prepared.sampled_target_hash = state->sampled_target_hash;
  prepared.worklist_hash = state->worklist_hash;
  prepared.policy_hash = state->policy_hash;
  prepared.source_panels = state->panels.size();
  prepared.target_panels = state->target_panels.size();
  prepared.prepare_build_count = 1;
  prepared.frozen_worklist = !state->pairs.empty();
  prepared.frozen_equivalent_system_cache =
      !state->equivalent_system_cache.entries.empty();
  prepared.prepared_worklist_pairs = state->pairs.size();
  prepared.prepared_far_source_nodes = state->prepared_far_source_nodes;
  prepared.prepared_equivalent_system_cache_entries =
      state->equivalent_system_cache.entries.size();
  prepared.prepare_seconds = elapsed_seconds(start, Clock::now());
  prepared.state = std::move(state);
  return prepared;
}

KifmmFullPanelMatvecResult apply_prepared_kifmm_lite_full_panel_operator(
    KifmmFullPanelPreparedOperator &prepared,
    std::span<const double> unknowns,
    bool compute_sampled_direct_audit) {
  return apply_prepared_kifmm_lite_full_panel_operator(
      prepared, unknowns, nullptr, compute_sampled_direct_audit);
}

KifmmFullPanelMatvecResult apply_prepared_kifmm_lite_full_panel_operator(
    KifmmFullPanelPreparedOperator &prepared,
    std::span<const double> unknowns,
    TitanExecutor *executor,
    bool compute_sampled_direct_audit) {
  ++prepared.apply_count;
  if (!prepared.state) {
    return failed("missing_prepared_state", prepared.config.lite.order);
  }
  KifmmFullPanelPreparedState &state = *prepared.state;
  const auto total_start = Clock::now();
  if (unknowns.size() != 2 * state.panels.size() || !valid_unknowns(unknowns)) {
    return failed("invalid_unknowns", state.config.lite.order);
  }

  KifmmFullPanelMatvecResult result;
  result.order = state.config.lite.order;
  result.source_box_policy = state.config.lite.source_box_policy;
  result.kifmm_full_apply_policy = state.config.full_apply_policy;
  result.sampled_target_indices = state.sampled_target_indices;
  result.panel_geometry_hash = state.panel_geometry_hash;
  result.unknown_vector_hash = unknown_vector_hash(unknowns);
  result.sampled_target_hash = state.sampled_target_hash;
  result.lite.source_panels = state.panels.size();
  result.lite.target_panels = state.target_panels.size();
  result.lite.min_far_panel_fraction =
      state.config.lite.min_far_panel_fraction;
  result.lite.rows.assign(state.target_panels.size(), {});
  result.lite.channel_rows.assign(state.target_panels.size(), {});
  result.lite.direct_rows.assign(state.sampled_target_indices.size(), {});
  result.lite.direct_channel_rows.assign(state.sampled_target_indices.size(),
                                         {});
  result.source_tree_bytes = state.source_tree_bytes;
  result.target_tree_bytes = state.target_tree_bytes;
  result.worklist_bytes = state.worklist_bytes;
  result.lite.source_nodes = state.source_tree.nodes.size();
  result.lite.target_nodes = state.target_tree.nodes.size();
  result.lite.source_leaves = state.source_tree.leaves.size();
  result.lite.target_leaves = state.target_tree.leaves.size();
  result.lite.source_tree_levels = state.source_tree_levels;
  result.lite.unique_source_node_half_widths =
      state.unique_source_node_half_widths;
  result.lite.source_root_half_width = state.source_root_half_width;
  result.lite.level_cube_geometry = state.level_cube_geometry;
  result.source_tree_levels = state.source_tree_levels;
  result.unique_source_node_half_widths =
      state.unique_source_node_half_widths;
  result.source_root_half_width = state.source_root_half_width;
  result.level_cube_geometry = state.level_cube_geometry;
  result.lite.worklist_hash = state.worklist_hash;
  result.policy_hash = state.policy_hash;

  auto finish_failure = [&](std::string status) {
    result.status = std::move(status);
    result.lite.status = result.status;
    result.finite = false;
    result.lite.finite = false;
    result.setup_seconds = 0.0;
    result.kifmm_far_eval_seconds = result.full_far_eval_seconds;
    result.kifmm_fallback_eval_seconds = result.full_fallback_eval_seconds;
    result.apply_seconds =
        result.kifmm_channel_build_seconds + result.full_far_eval_seconds +
        result.full_fallback_eval_seconds;
    result.first_apply_seconds = result.setup_seconds + result.apply_seconds;
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  };

  std::vector<KifmmPanelSource> sources;
  sources.reserve(state.panels.size());
  for (std::size_t i = 0; i < state.panels.size(); ++i) {
    sources.push_back(source_from_panel(state.panels[i], unknowns[i],
                                        unknowns[state.panels.size() + i]));
  }

  const double inv_eps_ratio = 1.0 / state.config.lite.eps_ratio;
  const std::size_t expected_full_pairs =
      state.panels.size() * state.target_panels.size();
  const std::size_t expected_sampled_pairs =
      state.panels.size() * state.sampled_target_indices.size();
  SampledNode all_sources;
  all_sources.indices.resize(state.panels.size());
  std::iota(all_sources.indices.begin(), all_sources.indices.end(),
            std::size_t{0});
  if (compute_sampled_direct_audit) {
    ProductionNearStats direct_reference_stats;
    const auto direct_reference_start = Clock::now();
    for (std::size_t target_local = 0;
         target_local < state.sampled_target_indices.size(); ++target_local) {
      const std::size_t target_index =
          state.sampled_target_indices[target_local];
      result.lite.direct_channel_rows[target_local] = production_channel_row(
          state.target_panels[target_index], target_index, state.panels,
          unknowns, all_sources, state.config.lite.eps_ratio,
          state.config.lite.kappa, state.config.near_integration,
          state.config.quad_buffer, &direct_reference_stats);
      result.lite.direct_rows[target_local] =
          row_from_channel(result.lite.direct_channel_rows[target_local]);
    }
    result.sampled_dense_audit_seconds =
        elapsed_seconds(direct_reference_start, Clock::now());
    result.sampled_direct_reference_valid_pairs +=
        direct_reference_stats.valid_pairs;
    result.sampled_invalid_pairs += direct_reference_stats.invalid_pairs;
    if (result.sampled_invalid_pairs != 0) {
      return finish_failure("invalid_production_near_kernel");
    }
    if (result.sampled_direct_reference_valid_pairs != expected_sampled_pairs) {
      return finish_failure("direct_reference_coverage_gap");
    }
    if (!finite_rows(result.lite.direct_rows) ||
        !finite_channel_rows(result.lite.direct_channel_rows)) {
      return finish_failure("nonfinite_direct_reference");
    }
  }

  std::unordered_map<int, KifmmJcpChannels> channel_cache;
  const auto channel_build_start = Clock::now();
  for (const SampledWorkPair &pair : state.pairs) {
    const SampledNode &target_node =
        state.target_tree.nodes[static_cast<std::size_t>(pair.target_node)];
    const SampledNode &source_node =
        state.source_tree.nodes[static_cast<std::size_t>(pair.source_node)];
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
            state.surface_rule, box, subset, state.config.lite.kappa,
            state.config.lite.eps_ratio, inv_eps_ratio,
            state.config.lite.solver, &state.equivalent_system_cache,
            state.config.lite.condition_policy);
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
          state.config.lite.max_condition_estimate) {
        return finish_failure("ill_conditioned_equivalent_system");
      }
      if (channels.max_relative_residual >
          state.config.lite.max_equivalent_relative_residual) {
        return finish_failure("equivalent_residual_too_large");
      }
    } else {
      ++result.lite.direct_fallback_cell_pairs;
      result.lite.direct_fallback_panel_pairs += panel_pairs;
    }
  }
  result.kifmm_channel_build_seconds =
      elapsed_seconds(channel_build_start, Clock::now());

  const std::vector<TargetNodeTask> target_tasks =
      group_target_node_tasks(state.pairs);
  result.kifmm_target_tasks = target_tasks.size();
  std::vector<int> grouped_target_nodes;
  grouped_target_nodes.reserve(target_tasks.size());
  for (const TargetNodeTask &task : target_tasks) {
    if (task.begin_pair >= task.end_pair ||
        task.end_pair > state.pairs.size()) {
      return finish_failure("invalid_target_node_task_range");
    }
    if (task.target_node < 0 ||
        static_cast<std::size_t>(task.target_node) >=
            state.target_tree.nodes.size()) {
      return finish_failure("invalid_target_node_task");
    }
    if (state.pairs[task.begin_pair].target_node != task.target_node ||
        state.pairs[task.end_pair - 1].target_node != task.target_node) {
      return finish_failure("noncontiguous_target_node_worklist");
    }
    grouped_target_nodes.push_back(task.target_node);
  }
  std::sort(grouped_target_nodes.begin(), grouped_target_nodes.end());
  if (std::adjacent_find(grouped_target_nodes.begin(),
                         grouped_target_nodes.end()) !=
      grouped_target_nodes.end()) {
    return finish_failure("noncontiguous_target_node_worklist");
  }
  const bool use_parallel =
      state.config.full_apply_policy == "target-node-parallel" &&
      executor != nullptr && executor->capacity() > 1 && target_tasks.size() > 1;
  const std::size_t worker_count =
      use_parallel
          ? std::min<std::size_t>(executor->capacity(), target_tasks.size())
          : std::size_t{1};
  result.kifmm_parallel_workers = worker_count;

  std::vector<ProductionNearStats> worker_fallback_stats(worker_count);
  auto evaluate_task_range = [&](std::size_t worker_id, std::size_t begin,
                                 std::size_t end, bool far_pass) {
    ProductionNearStats local_fallback_stats;
    for (std::size_t task_index = begin; task_index < end; ++task_index) {
      const TargetNodeTask &task = target_tasks[task_index];
      const SampledNode &target_node =
          state.target_tree.nodes[static_cast<std::size_t>(task.target_node)];
      for (std::size_t pair_index = task.begin_pair;
           pair_index < task.end_pair; ++pair_index) {
        const SampledWorkPair &pair = state.pairs[pair_index];
        if (pair.far != far_pass) {
          continue;
        }
        const SampledNode &source_node =
            state.source_tree.nodes[static_cast<std::size_t>(pair.source_node)];
        if (pair.far) {
          const KifmmBox box = node_box(source_node);
          const KifmmJcpChannels &channels = channel_cache.at(pair.source_node);
          for (const std::size_t target_local : target_node.indices) {
            const KifmmJcpChannelRow channel = assemble_kifmm_jcp_channel_row(
                state.surface_rule, box, channels, state.config.lite.kappa,
                state.config.lite.eps_ratio, inv_eps_ratio,
                state.targets[target_local].position,
                state.targets[target_local].normal);
            add_channel_row(result.lite.channel_rows[target_local], channel);
            add_row(result.lite.rows[target_local], row_from_channel(channel));
          }
        } else {
          for (const std::size_t target_local : target_node.indices) {
            const KifmmJcpChannelRow channel = production_channel_row(
                state.target_panels[target_local], target_local, state.panels,
                unknowns, source_node, state.config.lite.eps_ratio,
                state.config.lite.kappa, state.config.near_integration,
                state.config.quad_buffer, &local_fallback_stats);
            add_channel_row(result.lite.channel_rows[target_local], channel);
            add_row(result.lite.rows[target_local], row_from_channel(channel));
          }
        }
      }
    }
    if (worker_id < worker_fallback_stats.size()) {
      worker_fallback_stats[worker_id].valid_pairs +=
          local_fallback_stats.valid_pairs;
      worker_fallback_stats[worker_id].invalid_pairs +=
          local_fallback_stats.invalid_pairs;
    }
  };

  const auto far_eval_start = Clock::now();
  if (use_parallel) {
    executor->parallel_for(target_tasks.size(), worker_count, 1,
                           [&](std::size_t worker_id, std::size_t begin,
                               std::size_t end) {
                             evaluate_task_range(worker_id, begin, end, true);
                           });
  } else {
    evaluate_task_range(0, 0, target_tasks.size(), true);
  }
  result.full_far_eval_seconds = elapsed_seconds(far_eval_start, Clock::now());

  const auto fallback_eval_start = Clock::now();
  if (use_parallel) {
    executor->parallel_for(target_tasks.size(), worker_count, 1,
                           [&](std::size_t worker_id, std::size_t begin,
                               std::size_t end) {
                             evaluate_task_range(worker_id, begin, end, false);
                           });
  } else {
    evaluate_task_range(0, 0, target_tasks.size(), false);
  }
  result.full_fallback_eval_seconds =
      elapsed_seconds(fallback_eval_start, Clock::now());
  result.kifmm_far_eval_seconds = result.full_far_eval_seconds;
  result.kifmm_fallback_eval_seconds = result.full_fallback_eval_seconds;
  for (const ProductionNearStats &stats : worker_fallback_stats) {
    result.production_fallback_valid_pairs += stats.valid_pairs;
    result.sampled_invalid_pairs += stats.invalid_pairs;
  }
  if (result.sampled_invalid_pairs != 0) {
    return finish_failure("invalid_production_near_kernel");
  }

  if (result.nonproduction_fallback_pairs != 0) {
    return finish_failure("nonproduction_fallback_used");
  }
  if (result.lite.accepted_far_panel_pairs +
          result.lite.direct_fallback_panel_pairs != expected_full_pairs) {
    return finish_failure("worklist_coverage_gap");
  }
  result.full_pair_coverage =
      result.lite.accepted_far_panel_pairs +
      result.lite.direct_fallback_panel_pairs;
  if (result.production_fallback_valid_pairs !=
      result.lite.direct_fallback_panel_pairs) {
    return finish_failure("fallback_coverage_gap");
  }

  result.unique_far_source_nodes = channel_cache.size();
  result.cached_source_nodes = channel_cache.size();
  fill_equivalent_cache_geometry(result, state.equivalent_system_cache);
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

  std::vector<KifmmJcpRow> sampled_rows;
  std::vector<KifmmJcpChannelRow> sampled_channel_rows;
  if (compute_sampled_direct_audit) {
    sampled_rows.reserve(state.sampled_target_indices.size());
    sampled_channel_rows.reserve(state.sampled_target_indices.size());
    for (const std::size_t target_index : state.sampled_target_indices) {
      sampled_rows.push_back(result.lite.rows[target_index]);
      sampled_channel_rows.push_back(result.lite.channel_rows[target_index]);
    }
  }

  result.aggregate_abs_rms = compute_sampled_direct_audit
                                 ? abs_rms_rows(sampled_rows,
                                                result.lite.direct_rows)
                                 : 0.0;
  result.full_output_hash_phi = full_output_hash_phi(result.lite.rows);
  result.full_output_hash_phin = full_output_hash_phin(result.lite.rows);
  result.full_channel_output_hash =
      full_channel_output_hash(result.lite.channel_rows);
  if (compute_sampled_direct_audit) {
    result.k1_abs_rms =
        abs_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                        Channel::K1);
    result.k2_abs_rms =
        abs_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                        Channel::K2);
    result.k3_abs_rms =
        abs_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                        Channel::K3);
    result.k4_abs_rms =
        abs_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                        Channel::K4);
    result.k1_reference_rms =
        reference_rms_channel(result.lite.direct_channel_rows, Channel::K1);
    result.k2_reference_rms =
        reference_rms_channel(result.lite.direct_channel_rows, Channel::K2);
    result.k3_reference_rms =
        reference_rms_channel(result.lite.direct_channel_rows, Channel::K3);
    result.k4_reference_rms =
        reference_rms_channel(result.lite.direct_channel_rows, Channel::K4);
    result.aggregate_relative_rms =
        rel_rms_rows(sampled_rows, result.lite.direct_rows);
    result.k1_relative_rms =
        rel_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                        Channel::K1);
    result.k2_relative_rms =
        rel_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                        Channel::K2);
    result.k3_relative_rms =
        rel_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                        Channel::K3);
    result.k4_relative_rms =
        rel_rms_channel(sampled_channel_rows, result.lite.direct_channel_rows,
                        Channel::K4);
  }
  result.lite.aggregate_relative_rms = result.aggregate_relative_rms;
  result.lite.k1_relative_rms = result.k1_relative_rms;
  result.lite.k2_relative_rms = result.k2_relative_rms;
  result.lite.k3_relative_rms = result.k3_relative_rms;
  result.lite.k4_relative_rms = result.k4_relative_rms;

  if (!finite_rows(result.lite.rows) ||
      !finite_channel_rows(result.lite.channel_rows)) {
    return finish_failure("nonfinite_kifmm_rows");
  }
  if (state.config.lite.require_far_work &&
      (result.lite.accepted_far_cell_pairs == 0 ||
       result.lite.accepted_far_panel_pairs == 0)) {
    return finish_failure("no_far_work");
  }
  if (state.config.lite.require_far_work &&
      state.config.lite.min_far_panel_fraction > 0.0) {
    const std::size_t total_panel_pairs =
        result.lite.accepted_far_panel_pairs +
        result.lite.direct_fallback_panel_pairs;
    const double far_fraction =
        total_panel_pairs == 0
            ? 0.0
            : static_cast<double>(result.lite.accepted_far_panel_pairs) /
                  static_cast<double>(total_panel_pairs);
    if (far_fraction < state.config.lite.min_far_panel_fraction) {
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
  result.setup_seconds = 0.0;
  result.apply_seconds =
      result.kifmm_channel_build_seconds + result.full_far_eval_seconds +
      result.full_fallback_eval_seconds;
  result.first_apply_seconds = result.setup_seconds + result.apply_seconds;
  result.total_seconds = elapsed_seconds(total_start, Clock::now());
  result.lite.status = result.status;
  return result;
}

std::vector<std::vector<std::size_t>>
build_kifmm_lite_fallback_panel_graph(
    const KifmmFullPanelPreparedOperator &prepared,
    std::size_t neighbor_cap) {
  if (!prepared.state) {
    throw std::invalid_argument("kifmm_fallback_graph_unprepared_operator");
  }
  const KifmmFullPanelPreparedState &state = *prepared.state;
  const std::size_t n = state.panels.size();
  if (n == 0) {
    return {};
  }
  const std::size_t cap = std::max<std::size_t>(1, neighbor_cap);
  std::vector<std::vector<std::size_t>> graph(n);
  auto append_unique = [cap](std::vector<std::size_t> &row,
                             std::size_t value) {
    if (row.size() >= cap) {
      return;
    }
    if (std::find(row.begin(), row.end(), value) == row.end()) {
      row.push_back(value);
    }
  };
  for (const SampledWorkPair &pair : state.pairs) {
    if (pair.far) {
      continue;
    }
    if (pair.target_node < 0 || pair.source_node < 0 ||
        static_cast<std::size_t>(pair.target_node) >=
            state.target_tree.nodes.size() ||
        static_cast<std::size_t>(pair.source_node) >=
            state.source_tree.nodes.size()) {
      throw std::runtime_error("kifmm_fallback_graph_invalid_node");
    }
    const SampledNode &target_node =
        state.target_tree.nodes[static_cast<std::size_t>(pair.target_node)];
    const SampledNode &source_node =
        state.source_tree.nodes[static_cast<std::size_t>(pair.source_node)];
    for (const std::size_t target : target_node.indices) {
      if (target >= n) {
        throw std::runtime_error("kifmm_fallback_graph_invalid_target");
      }
      auto &row = graph[target];
      for (const std::size_t source : source_node.indices) {
        if (source >= n) {
          throw std::runtime_error("kifmm_fallback_graph_invalid_source");
        }
        append_unique(row, source);
      }
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    auto &row = graph[i];
    if (std::find(row.begin(), row.end(), i) == row.end()) {
      if (row.size() < cap) {
        row.push_back(i);
      } else {
        row.back() = i;
      }
    }
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    if (row.size() > cap) {
      row.resize(cap);
      if (!std::binary_search(row.begin(), row.end(), i)) {
        row.back() = i;
        std::sort(row.begin(), row.end());
      }
    }
  }
  return graph;
}

void assemble_kifmm_lite_tabi_system_matvec(
    const KifmmFullPanelMatvecResult &applied,
    std::span<const double> unknowns,
    double lhs_phi,
    double lhs_phin,
    std::vector<double> &out) {
  const std::size_t n = applied.lite.rows.size();
  if (n == 0 || unknowns.size() != 2 * n) {
    throw std::invalid_argument("kifmm_lite_tabi_matvec_size_mismatch");
  }
  if (!std::isfinite(lhs_phi) || !std::isfinite(lhs_phin) ||
      !valid_unknowns(unknowns)) {
    throw std::invalid_argument("kifmm_lite_tabi_matvec_nonfinite_input");
  }
  out.assign(2 * n, 0.0);
  for (std::size_t panel = 0; panel < n; ++panel) {
    const KifmmJcpRow &row = applied.lite.rows[panel];
    if (!std::isfinite(row.phi_row) || !std::isfinite(row.phin_row)) {
      throw std::invalid_argument("kifmm_lite_tabi_matvec_nonfinite_row");
    }
    out[panel] = lhs_phi * unknowns[panel] - row.phi_row;
    out[n + panel] = lhs_phin * unknowns[n + panel] - row.phin_row;
  }
}

} // namespace titan::tabi::fmm_kifmm
