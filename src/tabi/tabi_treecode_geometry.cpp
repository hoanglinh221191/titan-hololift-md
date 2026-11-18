#include "tabi_treecode_internal.h"
#include "tabi_node_patch_curved.h"
#include "tabi_near.h"

#include "../titan_executor.h"
#include "../titan_simd_dispatch.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TitanPBE::Tabi::detail {
std::uint64_t expand_bits_21(std::uint32_t value) noexcept {
  std::uint64_t out = 0;
  for (std::uint32_t bit = 0; bit < 21; ++bit)
    out |= static_cast<std::uint64_t>((value >> bit) & 1u) << (3u * bit);
  return out;
}

std::uint64_t morton_code(std::uint32_t x, std::uint32_t y,
                          std::uint32_t z) noexcept {
  return (expand_bits_21(x) << 2u) | (expand_bits_21(y) << 1u) |
         expand_bits_21(z);
}

std::vector<MortonEntry> radix_sort_morton(std::vector<MortonEntry> entries) {
  constexpr std::size_t kBits = 11;
  constexpr std::size_t kRadix = 1u << kBits;
  constexpr std::uint64_t kMask = kRadix - 1u;
  std::vector<MortonEntry> scratch(entries.size());
  for (std::size_t shift = 0; shift < 63; shift += kBits) {
    std::array<std::size_t, kRadix> counts{};
    for (const MortonEntry &entry : entries)
      ++counts[static_cast<std::size_t>((entry.code >> shift) & kMask)];
    std::size_t offset = 0;
    for (std::size_t bucket = 0; bucket < kRadix; ++bucket) {
      const std::size_t count = counts[bucket];
      counts[bucket] = offset;
      offset += count;
    }
    for (const MortonEntry &entry : entries) {
      const std::size_t bucket =
          static_cast<std::size_t>((entry.code >> shift) & kMask);
      scratch[counts[bucket]++] = entry;
    }
    entries.swap(scratch);
  }
  return entries;
}

void include_bound_point(TreeNode &node, double x, double y, double z,
                         bool &initialized) noexcept {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
    return;
  if (!initialized) {
    node.min_x = node.max_x = x;
    node.min_y = node.max_y = y;
    node.min_z = node.max_z = z;
    initialized = true;
    return;
  }
  node.min_x = std::min(node.min_x, x);
  node.min_y = std::min(node.min_y, y);
  node.min_z = std::min(node.min_z, z);
  node.max_x = std::max(node.max_x, x);
  node.max_y = std::max(node.max_y, y);
  node.max_z = std::max(node.max_z, z);
}

void include_panel_bounds(TreeNode &node, const Panel &panel,
                          bool &initialized) noexcept {
  if(const auto*cell=curved_dual_cell(panel)) {
    include_bound_point(node,panel.x-cell->radius,panel.y-cell->radius,panel.z-cell->radius,initialized);
    include_bound_point(node,panel.x+cell->radius,panel.y+cell->radius,panel.z+cell->radius,initialized);
  }
  if (panel.has_triangle_geometry) {
    include_bound_point(node, panel.v0x, panel.v0y, panel.v0z, initialized);
    include_bound_point(node, panel.v1x, panel.v1y, panel.v1z, initialized);
    include_bound_point(node, panel.v2x, panel.v2y, panel.v2z, initialized);
  }
  include_bound_point(node, panel.x, panel.y, panel.z, initialized);
}

double aabb_radius_from_center(const TreeNode &node) noexcept {
  double radius = 0.0;
  for (int xi = 0; xi < 2; ++xi) {
    const double x = xi == 0 ? node.min_x : node.max_x;
    for (int yi = 0; yi < 2; ++yi) {
      const double y = yi == 0 ? node.min_y : node.max_y;
      for (int zi = 0; zi < 2; ++zi) {
        const double z = zi == 0 ? node.min_z : node.max_z;
        const double dx = x - node.cx;
        const double dy = y - node.cy;
        const double dz = z - node.cz;
        radius = std::max(radius, std::sqrt(dx * dx + dy * dy + dz * dz));
      }
    }
  }
  return radius;
}

void finalize_tight_bounds(TreeNode &node, bool initialized,
                           bool use_aabb_center) noexcept {
  if (!initialized) {
    node.min_x = node.max_x = node.cx;
    node.min_y = node.max_y = node.cy;
    node.min_z = node.max_z = node.cz;
  }
  const double aabb_cx = 0.5 * (node.min_x + node.max_x);
  const double aabb_cy = 0.5 * (node.min_y + node.max_y);
  const double aabb_cz = 0.5 * (node.min_z + node.max_z);
  if (use_aabb_center || !std::isfinite(node.cx) ||
      !std::isfinite(node.cy) || !std::isfinite(node.cz)) {
    node.cx = aabb_cx;
    node.cy = aabb_cy;
    node.cz = aabb_cz;
  }
  node.radius = aabb_radius_from_center(node);
}

void set_moment_bounds_from_radius(TreeNode &node) noexcept {
  node.min_x = node.cx - node.radius;
  node.min_y = node.cy - node.radius;
  node.min_z = node.cz - node.radius;
  node.max_x = node.cx + node.radius;
  node.max_y = node.cy + node.radius;
  node.max_z = node.cz + node.radius;
}

TreeNode make_leaf_node(const std::vector<Panel> &panels,
                        const std::vector<std::size_t> &order,
                        std::size_t first_panel, std::size_t panel_count,
                        bool tight_bounds,
                        const std::string &expansion_center) noexcept {
  TreeNode node;
  node.first_panel = first_panel;
  node.panel_count = panel_count;
  bool bounds_initialized = false;
  double weight = 0.0;
  for (std::size_t pos = first_panel; pos < first_panel + panel_count; ++pos) {
    const Panel &panel = panels[order[pos]];
    include_panel_bounds(node, panel, bounds_initialized);
    node.max_edge_length =
        std::max(node.max_edge_length, tree_panel_max_edge_length(panel));
    const double area = std::max(0.0, panel.area);
    node.cx += area * panel.x;
    node.cy += area * panel.y;
    node.cz += area * panel.z;
    node.area_sum += area;
    weight += area;
  }
  if (weight > 0.0) {
    node.cx /= weight;
    node.cy /= weight;
    node.cz /= weight;
  } else if (panel_count > 0) {
    for (std::size_t pos = first_panel; pos < first_panel + panel_count; ++pos) {
      const Panel &panel = panels[order[pos]];
      node.cx += panel.x;
      node.cy += panel.y;
      node.cz += panel.z;
    }
    const double inv_count = 1.0 / static_cast<double>(panel_count);
    node.cx *= inv_count;
    node.cy *= inv_count;
    node.cz *= inv_count;
  }
  for (std::size_t pos = first_panel; pos < first_panel + panel_count; ++pos) {
    const Panel &panel = panels[order[pos]];
    const double dx = panel.x - node.cx;
    const double dy = panel.y - node.cy;
    const double dz = panel.z - node.cz;
    const auto*curved=curved_dual_cell(panel);
    node.radius = std::max(node.radius, std::sqrt(dx * dx + dy * dy + dz * dz) +
                          (curved?curved->radius:0.0));
  }
  if (tight_bounds)
    finalize_tight_bounds(node, bounds_initialized,
                          expansion_center != "area-centroid");
  else
    set_moment_bounds_from_radius(node);
  return node;
}

TreeNode make_parent_node(const std::vector<TreeNode> &nodes,
                          const std::vector<std::size_t> &children,
                          bool tight_bounds,
                          const std::string &expansion_center) noexcept {
  TreeNode node;
  node.child_count = children.size();
  if (!children.empty())
    node.first_panel = nodes[children.front()].first_panel;
  bool bounds_initialized = false;
  double weight = 0.0;
  for (std::size_t child_index : children) {
    const TreeNode &child_node = nodes[child_index];
    node.max_edge_length =
        std::max(node.max_edge_length, child_node.max_edge_length);
    node.depth = std::max(node.depth, child_node.depth + 1);
    include_bound_point(node, child_node.min_x, child_node.min_y,
                        child_node.min_z, bounds_initialized);
    include_bound_point(node, child_node.max_x, child_node.max_y,
                        child_node.max_z, bounds_initialized);
    const double area = std::max(0.0, child_node.area_sum);
    node.cx += area * child_node.cx;
    node.cy += area * child_node.cy;
    node.cz += area * child_node.cz;
    node.area_sum += area;
    node.panel_count += child_node.panel_count;
    weight += area;
  }
  if (weight > 0.0) {
    node.cx /= weight;
    node.cy /= weight;
    node.cz /= weight;
  } else if (!children.empty()) {
    node.cx = 0.0;
    node.cy = 0.0;
    node.cz = 0.0;
    for (std::size_t child_index : children) {
      const TreeNode &child_node = nodes[child_index];
      node.cx += child_node.cx;
      node.cy += child_node.cy;
      node.cz += child_node.cz;
    }
    const double inv_count = 1.0 / static_cast<double>(children.size());
    node.cx *= inv_count;
    node.cy *= inv_count;
    node.cz *= inv_count;
  }
  for (std::size_t child_index : children) {
    const TreeNode &child_node = nodes[child_index];
    const double dx = child_node.cx - node.cx;
    const double dy = child_node.cy - node.cy;
    const double dz = child_node.cz - node.cz;
    node.radius =
        std::max(node.radius,
                 std::sqrt(dx * dx + dy * dy + dz * dz) + child_node.radius);
  }
  if (tight_bounds)
    finalize_tight_bounds(node, bounds_initialized,
                          expansion_center != "area-centroid");
  else
    set_moment_bounds_from_radius(node);
  return node;
}

double point_aabb_gap(double x, double y, double z,
                      const TreeNode &node) noexcept {
  const double dx = std::max({node.min_x - x, x - node.max_x, 0.0});
  const double dy = std::max({node.min_y - y, y - node.max_y, 0.0});
  const double dz = std::max({node.min_z - z, z - node.max_z, 0.0});
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void annotate_fragmented_nodes(TreeWorkspace &tree) noexcept {
  for (TreeNode &node : tree.nodes) {
    node.fragmented = false;
    node.fragmentation_center_gap_ratio = 0.0;
    if (node.child_count < 2)
      continue;
    double min_child_gap = std::numeric_limits<double>::infinity();
    for (std::size_t child = 0; child < node.child_count; ++child) {
      const std::size_t child_pos = node.first_child + child;
      const std::size_t child_index =
          child_pos < tree.child_indices.size()
              ? tree.child_indices[child_pos]
              : static_cast<std::size_t>(-1);
      if (child_index >= tree.nodes.size())
        continue;
      min_child_gap =
          std::min(min_child_gap,
                   point_aabb_gap(node.cx, node.cy, node.cz,
                                  tree.nodes[child_index]));
    }
    if (!std::isfinite(min_child_gap))
      continue;
    const double scale = std::max(node.max_edge_length, 1.0e-12);
    node.fragmentation_center_gap_ratio = min_child_gap / scale;
    node.fragmented =
        node.fragmentation_center_gap_ratio >=
        kFragmentationGapRatioThreshold;
  }
}

std::size_t child_node_index(const TreeWorkspace &tree, const TreeNode &node,
                             std::size_t child) noexcept {
  const std::size_t pos = node.first_child + child;
  if (pos < tree.child_indices.size())
    return tree.child_indices[pos];
  return static_cast<std::size_t>(-1);
}

void push_child_nodes(const TreeWorkspace &tree, const TreeNode &node,
                      std::vector<std::size_t> &stack) {
  for (std::size_t child = 0; child < node.child_count; ++child) {
    const std::size_t child_index = child_node_index(tree, node, child);
    if (child_index < tree.nodes.size())
      stack.push_back(child_index);
  }
}

void push_child_nodes(const TreeWorkspace &tree, const TreeNode &node,
                      std::vector<TraversalItem> &stack,
                      LeafWorkReason reason) {
  for (std::size_t child = 0; child < node.child_count; ++child) {
    const std::size_t child_index = child_node_index(tree, node, child);
    if (child_index < tree.nodes.size())
      stack.push_back({child_index, reason});
  }
}

void attach_children(TreeWorkspace &tree, TreeNode &node,
                     const std::vector<std::size_t> &children) {
  node.first_child = tree.child_indices.size();
  node.child_count = children.size();
  tree.child_indices.insert(tree.child_indices.end(), children.begin(),
                            children.end());
}

void panel_centroid_bounds(const std::vector<Panel> &panels,
                           const std::vector<std::size_t> &order,
                           std::size_t first, std::size_t count,
                           double &min_x, double &min_y, double &min_z,
                           double &max_x, double &max_y,
                           double &max_z) noexcept {
  min_x = min_y = min_z = std::numeric_limits<double>::infinity();
  max_x = max_y = max_z = -std::numeric_limits<double>::infinity();
  for (std::size_t pos = first; pos < first + count; ++pos) {
    const Panel &panel = panels[order[pos]];
    min_x = std::min(min_x, panel.x);
    min_y = std::min(min_y, panel.y);
    min_z = std::min(min_z, panel.z);
    max_x = std::max(max_x, panel.x);
    max_y = std::max(max_y, panel.y);
    max_z = std::max(max_z, panel.z);
  }
  if (!std::isfinite(min_x)) {
    min_x = min_y = min_z = 0.0;
    max_x = max_y = max_z = 0.0;
  }
}

std::size_t build_topdown_temp_node(
    const std::vector<Panel> &panels, std::vector<std::size_t> &order,
    std::size_t first, std::size_t count, std::size_t leaf_size,
    std::size_t depth, std::vector<TopDownTempNode> &temp_nodes) {
  const std::size_t node_index = temp_nodes.size();
  TopDownTempNode node;
  node.first_panel = first;
  node.panel_count = count;
  node.depth = depth;
  temp_nodes.push_back(node);
  if (count <= leaf_size || count <= 1 || depth >= 32)
    return node_index;

  double min_x = 0.0;
  double min_y = 0.0;
  double min_z = 0.0;
  double max_x = 0.0;
  double max_y = 0.0;
  double max_z = 0.0;
  panel_centroid_bounds(panels, order, first, count, min_x, min_y, min_z,
                        max_x, max_y, max_z);
  const double mid_x = 0.5 * (min_x + max_x);
  const double mid_y = 0.5 * (min_y + max_y);
  const double mid_z = 0.5 * (min_z + max_z);
  std::array<std::vector<std::size_t>, kBranchFactor> buckets;
  for (std::size_t pos = first; pos < first + count; ++pos) {
    const Panel &panel = panels[order[pos]];
    std::size_t bucket = 0;
    if (panel.x >= mid_x)
      bucket |= 4u;
    if (panel.y >= mid_y)
      bucket |= 2u;
    if (panel.z >= mid_z)
      bucket |= 1u;
    buckets[bucket].push_back(order[pos]);
  }
  std::size_t nonempty = 0;
  std::size_t largest = 0;
  for (const auto &bucket : buckets) {
    if (!bucket.empty()) {
      ++nonempty;
      largest = std::max(largest, bucket.size());
    }
  }
  if (nonempty <= 1 || largest == count)
    return node_index;

  std::size_t write = first;
  std::array<std::pair<std::size_t, std::size_t>, kBranchFactor> ranges{};
  for (std::size_t bucket = 0; bucket < kBranchFactor; ++bucket) {
    ranges[bucket] = {write, buckets[bucket].size()};
    for (std::size_t panel_index : buckets[bucket])
      order[write++] = panel_index;
  }

  for (std::size_t bucket = 0; bucket < kBranchFactor; ++bucket) {
    const auto [child_first, child_count] = ranges[bucket];
    if (child_count == 0)
      continue;
    const std::size_t child = build_topdown_temp_node(
        panels, order, child_first, child_count, leaf_size, depth + 1,
        temp_nodes);
    TopDownTempNode &stored = temp_nodes[node_index];
    stored.children[stored.child_count++] = child;
  }
  return node_index;
}

void flatten_topdown_tree(const std::vector<Panel> &panels,
                          TreeWorkspace &tree,
                          const std::vector<TopDownTempNode> &temp_nodes,
                          std::size_t temp_root, bool tight_bounds,
                          const std::string &expansion_center) {
  std::vector<std::size_t> temp_to_final(temp_nodes.size(),
                                         static_cast<std::size_t>(-1));
  std::size_t max_depth = 0;
  for (const TopDownTempNode &node : temp_nodes)
    max_depth = std::max(max_depth, node.depth);

  tree.nodes.clear();
  tree.child_indices.clear();
  tree.moment_levels.clear();
  tree.nodes.reserve(temp_nodes.size());
  for (std::size_t temp = 0; temp < temp_nodes.size(); ++temp) {
    const TopDownTempNode &node = temp_nodes[temp];
    if (node.child_count != 0)
      continue;
    temp_to_final[temp] = tree.nodes.size();
    tree.nodes.push_back(make_leaf_node(panels, tree.order, node.first_panel,
                                        node.panel_count, tight_bounds,
                                        expansion_center));
  }
  tree.leaf_count = tree.nodes.size();
  tree.moment_levels.push_back({0, tree.leaf_count});

  for (std::size_t reverse_depth = max_depth + 1; reverse_depth-- > 0;) {
    const std::size_t level_begin = tree.nodes.size();
    for (std::size_t temp = 0; temp < temp_nodes.size(); ++temp) {
      const TopDownTempNode &node = temp_nodes[temp];
      if (node.child_count == 0 || node.depth != reverse_depth)
        continue;
      std::vector<std::size_t> children;
      children.reserve(node.child_count);
      bool mapped = true;
      for (std::size_t child = 0; child < node.child_count; ++child) {
        const std::size_t temp_child = node.children[child];
        if (temp_child >= temp_to_final.size() ||
            temp_to_final[temp_child] == static_cast<std::size_t>(-1)) {
          mapped = false;
          break;
        }
        children.push_back(temp_to_final[temp_child]);
      }
      if (!mapped || children.empty())
        continue;
      TreeNode parent = make_parent_node(tree.nodes, children, tight_bounds,
                                         expansion_center);
      parent.first_panel = node.first_panel;
      parent.panel_count = node.panel_count;
      attach_children(tree, parent, children);
      temp_to_final[temp] = tree.nodes.size();
      tree.nodes.push_back(parent);
    }
    if (tree.nodes.size() > level_begin)
      tree.moment_levels.push_back({level_begin, tree.nodes.size()});
    if (reverse_depth == 0)
      break;
  }
  tree.root_index =
      temp_root < temp_to_final.size() ? temp_to_final[temp_root] : 0;
  tree.depth_max = max_depth;
}

bool valid_vertex_index(std::size_t index) noexcept {
  return index != static_cast<std::size_t>(-1);
}

bool panel_has_vertex(const Panel &panel, std::size_t vertex) noexcept {
  if (!valid_vertex_index(vertex))
    return false;
  return panel.v0_index == vertex || panel.v1_index == vertex ||
         panel.v2_index == vertex;
}

int shared_vertex_count_treecode(const Panel &a, const Panel &b) noexcept {
  if (!a.has_triangle_geometry || !b.has_triangle_geometry)
    return 0;
  int count = 0;
  if (panel_has_vertex(b, a.v0_index))
    ++count;
  if (a.v1_index != a.v0_index && panel_has_vertex(b, a.v1_index))
    ++count;
  if (a.v2_index != a.v0_index && a.v2_index != a.v1_index &&
      panel_has_vertex(b, a.v2_index))
    ++count;
  return count;
}

bool is_singular_near_pair(const Panel &target, std::size_t target_index,
                           const Panel &source,
                           std::size_t source_index) noexcept {
  if (source_index == target_index)
    return true;
  return shared_vertex_count_treecode(target, source) > 0;
}

std::string normalize_leaf_near_policy(const std::string &policy) {
  if (policy == "adaptive-large")
    return "adaptive-large";
  if (policy == "large-protein")
    return "large-protein";
  if (policy == "smooth-node")
    return "smooth-node";
  return "exact";
}

std::string normalize_leaf_panel_kernel(const std::string &kernel) {
  if (kernel == "tile" || kernel == "tile-scalar" || kernel == "tile-avx2" ||
      kernel == "auto")
    return kernel;
  if (kernel == "avx2-gather" || kernel == "avx2")
    return "avx2-gather";
  return "scalar";
}

bool is_tile_leaf_panel_kernel(const std::string &kernel) noexcept {
  return kernel == "tile" || kernel == "tile-scalar" ||
         kernel == "tile-avx2";
}

bool is_effective_tile_leaf_panel_kernel(const std::string &kernel) noexcept {
  return kernel == "tile-scalar" || kernel == "tile-avx2";
}

AdaptivePruneParams adaptive_prune_params(const std::string &level) noexcept {
  if (level == "aggressive")
    return {0.45, 0.60, 0.0400};
  if (level == "balanced")
    return {0.34, 0.45, 0.0120};
  return {};
}

std::vector<std::pair<std::size_t, std::size_t>>
make_leaf_spans(const TreeNode &node, const std::vector<std::size_t> &order,
                const std::vector<std::size_t> &excluded_original) {
  std::vector<std::pair<std::size_t, std::size_t>> spans;
  const std::size_t end = node.first_panel + node.panel_count;
  std::size_t span_begin = end;
  std::size_t span_count = 0;
  auto is_excluded = [&](std::size_t original) noexcept {
    for (std::size_t value : excluded_original) {
      if (value == original)
        return true;
    }
    return false;
  };
  for (std::size_t pos = node.first_panel; pos < end; ++pos) {
    const std::size_t original = pos < order.size() ? order[pos] : pos;
    if (is_excluded(original)) {
      if (span_count > 0)
        spans.push_back({span_begin, span_count});
      span_begin = end;
      span_count = 0;
      continue;
    }
    if (span_count == 0)
      span_begin = pos;
    ++span_count;
  }
  if (span_count > 0)
    spans.push_back({span_begin, span_count});
  return spans;
}

std::string tile_fallback_status(const std::string &kernel,
                                 std::size_t tile_nodes) {
  if (tile_nodes > 0)
    return "active";
  if (is_effective_tile_leaf_panel_kernel(kernel))
    return "fallback";
  return "off";
}

TreeInteractionLists build_tree_interactions(const std::vector<Panel> &panels,
                                             const TreeWorkspace &tree,
                                             TitanExecutor &executor) {
  const auto begin_time = std::chrono::steady_clock::now();
  TreeInteractionLists lists;
  const std::size_t n = panels.size();
  lists.singular_offsets.assign(n + 1, 0);
  lists.leaf_offsets.assign(n + 1, 0);
  lists.leaf_tile_offsets.assign(n + 1, 0);
  lists.smooth_leaf_offsets.assign(n + 1, 0);
  lists.far_offsets.assign(n + 1, 0);
  if (n == 0 || tree.nodes.empty() || tree.order.size() != n) {
    lists.build_ms =
        elapsed_ms(begin_time, std::chrono::steady_clock::now());
    return lists;
  }

  std::vector<std::size_t> rank(n, 0);
  for (std::size_t pos = 0; pos < tree.order.size(); ++pos)
    rank[tree.order[pos]] = pos;

  std::vector<std::vector<std::size_t>> singular_by_target(n);
  std::vector<std::vector<std::size_t>> leaf_by_target(n);
  std::vector<std::vector<std::size_t>> leaf_tile_by_target(n);
  std::vector<std::vector<std::size_t>> leaf_tile_span_counts_by_target(n);
  std::vector<std::vector<std::size_t>> leaf_tile_span_firsts_by_target(n);
  std::vector<std::vector<std::size_t>> leaf_tile_span_lengths_by_target(n);
  std::vector<std::vector<std::size_t>> smooth_leaf_by_target(n);
  std::vector<std::vector<std::size_t>> far_by_target(n);
  std::vector<std::size_t> leaf_pruned_by_target(n, 0);
  std::vector<std::size_t> adaptive_smooth_accepted_by_target(n, 0);
  std::vector<std::size_t> adaptive_smooth_rejected_by_target(n, 0);
  std::vector<std::size_t> adaptive_span_accepted_by_target(n, 0);
  std::vector<std::size_t> adaptive_span_rejected_by_target(n, 0);
  std::vector<std::size_t> adaptive_exact_fallbacks_by_target(n, 0);
  std::vector<std::size_t> adaptive_reject_one_ring_by_target(n, 0);
  std::vector<std::size_t> adaptive_reject_error_by_target(n, 0);
  std::vector<std::size_t> adaptive_reject_p4_by_target(n, 0);
  std::vector<std::size_t> adaptive_reject_too_close_by_target(n, 0);
  std::vector<std::size_t> target_cluster_interactions_by_target(n, 0);
  std::vector<std::size_t> target_cluster_targets_by_target(n, 0);
  std::vector<std::size_t> target_cluster_panels_by_target(n, 0);
  std::vector<std::size_t> target_cluster_rejected_by_target(n, 0);
  std::vector<std::size_t> mac_reject_by_target(n, 0);
  std::vector<std::size_t> mac_convergence_rho_reject_by_target(n, 0);
  std::vector<std::size_t> mac_convergence_aspect_reject_by_target(n, 0);
  std::vector<std::size_t> mac_convergence_rho_panel_by_target(n, 0);
  std::vector<std::size_t> mac_convergence_aspect_panel_by_target(n, 0);
  std::vector<std::size_t> mac_surface_margin_reject_by_target(n, 0);
  std::vector<std::size_t> mac_surface_margin_panel_by_target(n, 0);
  std::vector<std::size_t> leaf_mac_candidates_by_target(n, 0);
  std::vector<std::size_t> leaf_mac_accepted_by_target(n, 0);
  std::vector<std::size_t> leaf_mac_rejected_by_target(n, 0);
  std::vector<std::size_t> leaf_mac_far_panels_by_target(n, 0);
  std::vector<std::size_t> leaf_mac_saved_panels_by_target(n, 0);
  std::vector<MacAuditTally> mac_audit_by_target(n);
  std::vector<FragmentationTally> fragmentation_by_target(n);
  const AdaptivePruneParams adaptive_params =
      adaptive_prune_params(tree.adaptive_level);
  const std::size_t workers = effective_workers(executor, n);
  executor.parallel_for(
      n, workers, 8,
      [&](std::size_t, std::size_t target_begin, std::size_t target_end) {
        std::vector<std::size_t> stack;
        stack.reserve(64);
        for (std::size_t i = target_begin; i < target_end; ++i) {
          const Panel &target = panels[i];
          const std::size_t target_rank = rank[i];
          auto &singular = singular_by_target[i];
          auto &leaf = leaf_by_target[i];
          auto &leaf_tile = leaf_tile_by_target[i];
          auto &leaf_tile_span_counts = leaf_tile_span_counts_by_target[i];
          auto &leaf_tile_span_firsts = leaf_tile_span_firsts_by_target[i];
          auto &leaf_tile_span_lengths = leaf_tile_span_lengths_by_target[i];
          auto &smooth_leaf = smooth_leaf_by_target[i];
          auto &far = far_by_target[i];
          stack.clear();
          stack.push_back(tree.root_index);
          while (!stack.empty()) {
            const std::size_t node_index = stack.back();
            stack.pop_back();
            const TreeNode &node = tree.nodes[node_index];
            const bool is_leaf = node.child_count == 0;
            const bool contains_target =
                target_rank >= node.first_panel &&
                target_rank < node.first_panel + node.panel_count;
            const bool leaf_mac_enabled = tree.leaf_mac != "off";
            const bool evaluate_node_mac =
                !contains_target && (!is_leaf || leaf_mac_enabled);
            if (evaluate_node_mac) {
              if (is_leaf)
                ++leaf_mac_candidates_by_target[i];
              const double dx = target.x - node.cx;
              const double dy = target.y - node.cy;
              const double dz = target.z - node.cz;
              const double distance =
                  std::sqrt(std::max(1.0e-16, dx * dx + dy * dy + dz * dz));
              const MacDecision mac = evaluate_mac(target, node, tree, distance);
              if (is_leaf) {
                if (mac.accepted)
                  ++leaf_mac_accepted_by_target[i];
                else
                  ++leaf_mac_rejected_by_target[i];
              }
              if (mac.ratio_accept && tree.mac_safety != "off") {
                record_far_geometric_audit(mac_audit_by_target[i], node,
                                           mac.gap, mac.gap_ratio,
                                           mac.unsafe);
                if (mac.physical_gap_limited)
                  ++mac_audit_by_target[i].physical_gap_rejects;
              }
              if (mac.convergence_rho_limited) {
                ++mac_convergence_rho_reject_by_target[i];
                mac_convergence_rho_panel_by_target[i] += node.panel_count;
              }
              if (mac.convergence_aspect_limited) {
                ++mac_convergence_aspect_reject_by_target[i];
                mac_convergence_aspect_panel_by_target[i] += node.panel_count;
              }
              if (mac.surface_margin_limited) {
                ++mac_surface_margin_reject_by_target[i];
                mac_surface_margin_panel_by_target[i] += node.panel_count;
              }
              if (mac.accepted && (!is_leaf || tree.leaf_mac == "on")) {
                bool leaf_mac_fallback_to_direct = false;
                const bool fragmented_far =
                    tree.fragmentation_safety != "off" && node.fragmented;
                if (fragmented_far) {
                  const bool split =
                      tree.fragmentation_safety == "split";
                  record_fragmentation_audit(fragmentation_by_target[i], node,
                                             split);
                  if (split) {
                    if (is_leaf) {
                      leaf_mac_fallback_to_direct = true;
                    } else {
                      push_child_nodes(tree, node, stack);
                      continue;
                    }
                  }
                }
                if (!leaf_mac_fallback_to_direct) {
                far.push_back(node_index);
                if (is_leaf) {
                  leaf_mac_far_panels_by_target[i] += node.panel_count;
                  leaf_mac_saved_panels_by_target[i] += node.panel_count;
                }
                continue;
                }
              }
              if (!is_leaf)
                ++mac_reject_by_target[i];
            }
            if (is_leaf) {
              bool has_singular_panel = false;
              std::vector<std::size_t> singular_candidates;
              std::vector<std::size_t> residual_candidates;
              singular_candidates.reserve(4);
              residual_candidates.reserve(node.panel_count);
              for (std::size_t pos = node.first_panel;
                   pos < node.first_panel + node.panel_count; ++pos) {
                const std::size_t source_index = tree.order[pos];
                const Panel &source = panels[source_index];
                if (curved_dual_cell(source) || is_singular_near_pair(target, i, source, source_index) ||
                    classify_geometric_near_buffer_pair(
                        target, source, tree.quad_buffer)
                        .active) {
                  has_singular_panel = true;
                  singular_candidates.push_back(source_index);
                } else {
                  residual_candidates.push_back(source_index);
                }
              }
              bool accepts_smooth_leaf = false;
              const bool adaptive_policy =
                  tree.leaf_near_policy == "adaptive-large";
              if (tree.leaf_near_policy == "smooth-node" &&
                  !contains_target && node.panel_count > 0 &&
                  !has_singular_panel) {
                  const double dx = target.x - node.cx;
                  const double dy = target.y - node.cy;
                  const double dz = target.z - node.cz;
                  const double distance = std::sqrt(
                      std::max(1.0e-16, dx * dx + dy * dy + dz * dz));
                  const double smooth_opening_theta =
                      std::min(0.35, 0.5 * tree.opening_theta);
                  accepts_smooth_leaf =
                      node.radius / distance < smooth_opening_theta;
              }
              if (adaptive_policy && !contains_target && node.panel_count > 0) {
                if (has_singular_panel) {
                  ++adaptive_smooth_rejected_by_target[i];
                  ++adaptive_reject_one_ring_by_target[i];
                } else {
                  const double dx = target.x - node.cx;
                  const double dy = target.y - node.cy;
                  const double dz = target.z - node.cz;
                  const double distance = std::sqrt(
                      std::max(1.0e-16, dx * dx + dy * dy + dz * dz));
                  const double ratio = node.radius / distance;
                  const double p4_proxy = ratio * ratio * ratio * ratio;
                  const double kAdaptiveSmoothTheta =
                      std::min(adaptive_params.smooth_theta_cap,
                               adaptive_params.smooth_theta_factor *
                                   tree.opening_theta);
                  const double kAdaptiveP4ProxyLimit =
                      adaptive_params.p4_proxy_limit;
                  if (ratio < kAdaptiveSmoothTheta &&
                      p4_proxy < kAdaptiveP4ProxyLimit) {
                    accepts_smooth_leaf = true;
                    ++adaptive_smooth_accepted_by_target[i];
                    ++target_cluster_interactions_by_target[i];
                    ++target_cluster_targets_by_target[i];
                    target_cluster_panels_by_target[i] += node.panel_count;
                  } else {
                    ++adaptive_smooth_rejected_by_target[i];
                    ++target_cluster_rejected_by_target[i];
                    if (ratio >= 0.5) {
                      ++adaptive_reject_too_close_by_target[i];
                    } else if (p4_proxy >= kAdaptiveP4ProxyLimit) {
                      ++adaptive_reject_p4_by_target[i];
                    } else {
                      ++adaptive_reject_error_by_target[i];
                    }
                  }
                }
              }
              if (accepts_smooth_leaf) {
                smooth_leaf.push_back(node_index);
                leaf_pruned_by_target[i] += node.panel_count;
                continue;
              }
              if ((tree.leaf_near_policy == "large-protein" ||
                   adaptive_policy) &&
                  is_tile_leaf_panel_kernel(tree.leaf_panel_kernel) &&
                  !residual_candidates.empty() && node.panel_count >= 16 &&
                  singular_candidates.size() * 4 <= node.panel_count) {
                const auto spans =
                    make_leaf_spans(node, tree.order, singular_candidates);
                std::size_t total_span_panels = 0;
                for (const auto &span : spans)
                  total_span_panels += span.second;
                const bool span_ok =
                    adaptive_policy
                        ? (spans.size() <= 4 && total_span_panels >= 64 &&
                           total_span_panels >= 32 * spans.size())
                        : (spans.size() <= 4 && total_span_panels >= 16);
                if (span_ok) {
                  if (adaptive_policy)
                    ++adaptive_span_accepted_by_target[i];
                  singular.insert(singular.end(), singular_candidates.begin(),
                                  singular_candidates.end());
                  leaf_tile.push_back(node_index);
                  leaf_tile_span_counts.push_back(spans.size());
                  for (const auto &span : spans) {
                    leaf_tile_span_firsts.push_back(span.first);
                    leaf_tile_span_lengths.push_back(span.second);
                  }
                  continue;
                }
                if (adaptive_policy)
                  ++adaptive_span_rejected_by_target[i];
              }
              if (adaptive_policy)
                ++adaptive_exact_fallbacks_by_target[i];
              if (is_tile_leaf_panel_kernel(tree.leaf_panel_kernel) &&
                  !contains_target && !has_singular_panel &&
                  node.panel_count > 0) {
                leaf_tile.push_back(node_index);
                leaf_tile_span_counts.push_back(1);
                leaf_tile_span_firsts.push_back(node.first_panel);
                leaf_tile_span_lengths.push_back(node.panel_count);
                continue;
              }
              singular.insert(singular.end(), singular_candidates.begin(),
                              singular_candidates.end());
              leaf.insert(leaf.end(), residual_candidates.begin(),
                          residual_candidates.end());
            } else {
              for (std::size_t child = 0; child < node.child_count; ++child) {
                const std::size_t child_index =
                    child_node_index(tree, node, child);
                if (child_index < tree.nodes.size())
                  stack.push_back(child_index);
              }
            }
          }
        }
      });

  for (std::size_t i = 0; i < n; ++i) {
    lists.singular_offsets[i + 1] =
        lists.singular_offsets[i] + singular_by_target[i].size();
    lists.leaf_offsets[i + 1] =
        lists.leaf_offsets[i] + leaf_by_target[i].size();
    lists.leaf_tile_offsets[i + 1] =
        lists.leaf_tile_offsets[i] + leaf_tile_by_target[i].size();
    lists.smooth_leaf_offsets[i + 1] =
        lists.smooth_leaf_offsets[i] + smooth_leaf_by_target[i].size();
    lists.far_offsets[i + 1] = lists.far_offsets[i] + far_by_target[i].size();
    lists.leaf_pruned_count += leaf_pruned_by_target[i];
    lists.adaptive_smooth_leaf_accepted +=
        adaptive_smooth_accepted_by_target[i];
    lists.adaptive_smooth_leaf_rejected +=
        adaptive_smooth_rejected_by_target[i];
    lists.adaptive_span_tile_accepted += adaptive_span_accepted_by_target[i];
    lists.adaptive_span_tile_rejected += adaptive_span_rejected_by_target[i];
    lists.adaptive_exact_leaf_fallbacks += adaptive_exact_fallbacks_by_target[i];
    lists.adaptive_reject_one_ring_overlap +=
        adaptive_reject_one_ring_by_target[i];
    lists.adaptive_reject_error_bound_failed +=
        adaptive_reject_error_by_target[i];
    lists.adaptive_reject_p4_check_failed += adaptive_reject_p4_by_target[i];
    lists.adaptive_reject_too_close += adaptive_reject_too_close_by_target[i];
    lists.target_cluster_interactions +=
        target_cluster_interactions_by_target[i];
    lists.target_cluster_targets += target_cluster_targets_by_target[i];
    lists.target_cluster_panels_pruned += target_cluster_panels_by_target[i];
    lists.target_cluster_rejected += target_cluster_rejected_by_target[i];
    lists.mac_reject_radius_limited += mac_reject_by_target[i];
    lists.mac_convergence_rho_rejects +=
        mac_convergence_rho_reject_by_target[i];
    lists.mac_convergence_aspect_rejects +=
        mac_convergence_aspect_reject_by_target[i];
    lists.mac_convergence_rho_reject_panels +=
        mac_convergence_rho_panel_by_target[i];
    lists.mac_convergence_aspect_reject_panels +=
        mac_convergence_aspect_panel_by_target[i];
    lists.mac_surface_margin_rejects += mac_surface_margin_reject_by_target[i];
    lists.mac_surface_margin_reject_panels +=
        mac_surface_margin_panel_by_target[i];
    lists.leaf_mac_candidates += leaf_mac_candidates_by_target[i];
    lists.leaf_mac_accepted += leaf_mac_accepted_by_target[i];
    lists.leaf_mac_rejected += leaf_mac_rejected_by_target[i];
    lists.leaf_mac_far_panels += leaf_mac_far_panels_by_target[i];
    lists.leaf_mac_direct_panels_saved_estimate +=
        leaf_mac_saved_panels_by_target[i];
    MacAuditTally tally;
    tally.unsafe_count = lists.far_geometric_unsafe_count;
    tally.unsafe_panels = lists.far_geometric_unsafe_panels;
    tally.min_gap = lists.far_geometric_min_gap;
    tally.min_gap_ratio = lists.far_geometric_gap_ratio_min;
    tally.sum_gap_ratio = lists.far_geometric_gap_ratio_sum;
    tally.samples = lists.far_geometric_audit_samples;
    tally.physical_gap_rejects = lists.mac_physical_gap_rejects;
    merge_far_geometric_audit(tally, mac_audit_by_target[i]);
    lists.far_geometric_unsafe_count = tally.unsafe_count;
    lists.far_geometric_unsafe_panels = tally.unsafe_panels;
    lists.far_geometric_min_gap = tally.min_gap;
    lists.far_geometric_gap_ratio_min = tally.min_gap_ratio;
    lists.far_geometric_gap_ratio_sum = tally.sum_gap_ratio;
    lists.far_geometric_audit_samples = tally.samples;
    lists.mac_physical_gap_rejects = tally.physical_gap_rejects;
    FragmentationTally fragmentation;
    fragmentation.far_count = lists.fragmented_far_count;
    fragmentation.far_panels = lists.fragmented_far_panels;
    fragmentation.split_rejects = lists.fragmented_split_rejects;
    fragmentation.gap_ratio_sum =
        lists.fragmentation_center_gap_ratio_sum;
    fragmentation.gap_ratio_max =
        lists.fragmentation_center_gap_ratio_max;
    merge_fragmentation_audit(fragmentation, fragmentation_by_target[i]);
    lists.fragmented_far_count = fragmentation.far_count;
    lists.fragmented_far_panels = fragmentation.far_panels;
    lists.fragmented_split_rejects = fragmentation.split_rejects;
    lists.fragmentation_center_gap_ratio_sum = fragmentation.gap_ratio_sum;
    lists.fragmentation_center_gap_ratio_max = fragmentation.gap_ratio_max;
  }
  lists.singular_indices.reserve(lists.singular_offsets.back());
  lists.leaf_indices.reserve(lists.leaf_offsets.back());
  lists.leaf_tile_node_indices.reserve(lists.leaf_tile_offsets.back());
  lists.smooth_leaf_node_indices.reserve(lists.smooth_leaf_offsets.back());
  lists.far_node_indices.reserve(lists.far_offsets.back());
  lists.leaf_tile_exclusion_offsets.clear();
  lists.leaf_tile_exclusion_offsets.reserve(lists.leaf_tile_offsets.back() + 1);
  lists.leaf_tile_exclusion_offsets.push_back(0);
  lists.leaf_tile_exclusion_indices.clear();
  lists.leaf_tile_span_offsets.clear();
  lists.leaf_tile_span_offsets.reserve(lists.leaf_tile_offsets.back() + 1);
  lists.leaf_tile_span_offsets.push_back(0);
  lists.leaf_tile_span_firsts.clear();
  lists.leaf_tile_span_counts.clear();
  for (std::size_t i = 0; i < n; ++i) {
    lists.singular_indices.insert(lists.singular_indices.end(),
                                  singular_by_target[i].begin(),
                                  singular_by_target[i].end());
    lists.leaf_indices.insert(lists.leaf_indices.end(),
                              leaf_by_target[i].begin(),
                              leaf_by_target[i].end());
    lists.leaf_tile_node_indices.insert(
        lists.leaf_tile_node_indices.end(), leaf_tile_by_target[i].begin(),
        leaf_tile_by_target[i].end());
    const auto &span_counts = leaf_tile_span_counts_by_target[i];
    const auto &span_firsts = leaf_tile_span_firsts_by_target[i];
    const auto &span_lengths = leaf_tile_span_lengths_by_target[i];
    std::size_t span_pos = 0;
    for (std::size_t entry = 0; entry < leaf_tile_by_target[i].size();
         ++entry) {
      lists.leaf_tile_exclusion_offsets.push_back(
          lists.leaf_tile_exclusion_indices.size());
      const std::size_t count =
          entry < span_counts.size() ? span_counts[entry] : 0;
      if (count > 0 && span_pos + count <= span_firsts.size() &&
          span_pos + count <= span_lengths.size()) {
        lists.leaf_tile_span_firsts.insert(
            lists.leaf_tile_span_firsts.end(),
            span_firsts.begin() + static_cast<std::ptrdiff_t>(span_pos),
            span_firsts.begin() + static_cast<std::ptrdiff_t>(span_pos + count));
        lists.leaf_tile_span_counts.insert(
            lists.leaf_tile_span_counts.end(),
            span_lengths.begin() + static_cast<std::ptrdiff_t>(span_pos),
            span_lengths.begin() +
                static_cast<std::ptrdiff_t>(span_pos + count));
      }
      span_pos += count;
      lists.leaf_tile_span_offsets.push_back(
          lists.leaf_tile_span_firsts.size());
    }
    lists.smooth_leaf_node_indices.insert(
        lists.smooth_leaf_node_indices.end(),
        smooth_leaf_by_target[i].begin(), smooth_leaf_by_target[i].end());
    lists.far_node_indices.insert(lists.far_node_indices.end(),
                                  far_by_target[i].begin(),
                                  far_by_target[i].end());
  }
  lists.singular_pair_count = lists.singular_indices.size();
  lists.leaf_pair_count = lists.leaf_indices.size();
  lists.leaf_tile_node_count = lists.leaf_tile_node_indices.size();
  lists.leaf_tile_panel_count = 0;
  for (std::size_t entry = 0; entry < lists.leaf_tile_node_indices.size();
       ++entry) {
    const std::size_t node_index = lists.leaf_tile_node_indices[entry];
    if (node_index < tree.nodes.size()) {
      const std::size_t span_begin =
          entry + 1 < lists.leaf_tile_span_offsets.size()
              ? lists.leaf_tile_span_offsets[entry]
              : 0;
      const std::size_t span_end =
          entry + 1 < lists.leaf_tile_span_offsets.size()
              ? lists.leaf_tile_span_offsets[entry + 1]
              : span_begin;
      for (std::size_t span = span_begin; span < span_end; ++span) {
        if (span < lists.leaf_tile_span_counts.size())
          lists.leaf_tile_panel_count += lists.leaf_tile_span_counts[span];
      }
    }
  }
  lists.leaf_pair_count += lists.leaf_tile_panel_count;
  lists.smooth_leaf_node_count = lists.smooth_leaf_node_indices.size();
  lists.far_node_count = lists.far_node_indices.size();
  lists.build_ms = elapsed_ms(begin_time, std::chrono::steady_clock::now());
  return lists;
}


} // namespace TitanPBE::Tabi::detail

#include "tabi_treecode_internal.h"
#include "tabi_near.h"

#include "../titan_executor.h"
#include "../titan_simd_dispatch.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TitanPBE::Tabi::detail {
TreeWorkspace build_tree_workspace_impl(const std::vector<Panel> &panels,
                                   TitanExecutor &executor,
                                   std::size_t leaf_size,
                                   double opening_theta,
                                   const std::string &leaf_near_policy,
                                   const std::string &leaf_panel_kernel,
                                   const std::string &adaptive_level,
                                   const std::string &quad_buffer,
                                   const std::string &interaction_mode,
                                   const std::string &node_bounds_mode,
                                   const std::string &tree_builder,
                                   const std::string &mac_safety,
                                   double mac_gap_factor,
                                   const std::string &moment_audit,
                                   const std::string &expansion_center,
                                   const std::string &fragmentation_safety,
                                   const std::string &expansion_audit,
                                   int expansion_audit_sample_count,
                                   int expansion_audit_max_panels,
                                   const std::string &moment_construction,
                                   const std::string &moment_sum,
                                   const std::string &convergence_audit,
                                   const std::string &convergence_safety,
                                   double convergence_rho_cap,
                                   double mac_aspect_cap,
                                   const std::string &convergence_estimator,
                                   double convergence_tail_threshold,
                                   double convergence_estimator_error_cap,
                                   double convergence_unstable_rho_cap,
                                   const std::string
                                       &convergence_calibration,
                                   const std::string &cluster_policy,
                                   const std::string
                                       &convergence_dielectric_amp,
                                   int target_batch_size,
                                   int leaf_pair_min_source_panels,
                                   int leaf_pair_min_targets,
                                   const std::string
                                       &convergence_estimator_model,
                                   const std::string
                                       &expansion_audit_channel,
                                   const std::string
                                       &convergence_channel_calibration,
                                   const std::string
                                       &channel_missed_semantics,
                                   const std::string
                                       &convergence_split_policy,
                                   const std::string
                                       &accepted_miss_diagnostics,
                                   double convergence_k3_accepted_rho_cap,
                                   const std::string
                                       &mac_distance_policy,
                                   double mac_surface_margin_factor,
                                   const std::string
                                       &mac_surface_margin_policy,
                                   double mac_surface_margin_aspect_threshold,
                                   double mac_surface_margin_rho_trigger,
                                   double accepted_miss_absolute_floor,
                                   const std::string &leaf_pair_kernel,
                                   const std::string &leaf_kernel_profile,
                                   const std::string &leaf_work_profile,
                                   const std::string &jcp_direct_kernel,
                                   const std::string
                                       &leaf_pair_selection_profile,
                                   const std::string &leaf_pair_probe_cache,
                                   const std::string &leaf_pair_target_group,
                                   int leaf_pair_max_lookahead,
                                   const std::string
                                       &leaf_pair_directness_probe,
                                   const std::string
                                       &leaf_pair_directness_cache,
                                   const std::string
                                       &leaf_pair_directness_verify,
                                   const std::string &leaf_pair_scheduler,
                                   int leaf_pair_worklist_max_active,
                                   const std::string
                                       &leaf_pair_worklist_verify,
                                   const std::string &leaf_block_verify,
                                   double mac_surface_margin_soft_aspect_threshold,
                                   double mac_surface_margin_soft_rho_trigger,
                                   double mac_surface_margin_soft_factor,
                                   const std::string
                                       &convergence_split_selectivity,
                                   double convergence_split_absolute_floor,
                                   const std::string
                                       &convergence_overreject_audit,
                                   const std::string
                                       &convergence_overreject_detail,
                                   const std::string
                                       &convergence_shape_audit,
                                   const std::string
                                       &convergence_kernel_response_audit,
                                   const std::string &leaf_mac,
                                   const std::string &fmm_m2l) {
  const auto build_begin = std::chrono::steady_clock::now();
  TreeWorkspace tree;
  tree.taylor_order = kTaylorOrderDefault;
  const std::size_t effective_leaf_size =
      std::clamp<std::size_t>(leaf_size == 0 ? kDefaultLeafSize : leaf_size, 8,
                              512);
  tree.leaf_size = effective_leaf_size;
  tree.opening_theta =
      std::clamp(std::isfinite(opening_theta) && opening_theta > 0.0
                     ? opening_theta
                     : kDefaultOpeningTheta,
                 0.05, 2.0);
  tree.leaf_near_policy = normalize_leaf_near_policy(leaf_near_policy);
  tree.leaf_panel_kernel = normalize_leaf_panel_kernel(leaf_panel_kernel);
  tree.adaptive_level =
      (adaptive_level == "balanced" || adaptive_level == "aggressive")
          ? adaptive_level
          : std::string("strict");
  tree.cluster_policy =
      (cluster_policy == "leaf-pair-strict" ||
       cluster_policy == "leaf-pair-balanced")
          ? cluster_policy
          : std::string("off");
  tree.leaf_pair_kernel = normalize_leaf_pair_kernel(leaf_pair_kernel);
  tree.leaf_kernel_profile =
      leaf_kernel_profile == "summary" ? std::string("summary")
                                       : std::string("off");
  tree.leaf_work_profile =
      leaf_work_profile == "summary" ? std::string("summary")
                                     : std::string("off");
  tree.jcp_direct_kernel =
      (jcp_direct_kernel == "soa-avx2" ||
       jcp_direct_kernel == "fused-soa-avx2")
          ? jcp_direct_kernel
          : std::string("baseline");
  tree.target_batch_size = static_cast<std::size_t>(
      std::clamp(target_batch_size, 1, 8));
  if (tree.target_batch_size != 4 && tree.target_batch_size != 8)
    tree.target_batch_size = 1;
  tree.leaf_pair_min_source_panels = static_cast<std::size_t>(
      std::clamp(leaf_pair_min_source_panels, 16, 512));
  tree.leaf_pair_min_targets = static_cast<std::size_t>(
      std::clamp(leaf_pair_min_targets, 2, 16));
  tree.leaf_pair_selection_profile =
      leaf_pair_selection_profile == "summary" ? std::string("summary")
                                               : std::string("off");
  tree.leaf_pair_probe_cache =
      leaf_pair_probe_cache == "per-matvec" ? std::string("per-matvec")
                                            : std::string("off");
  tree.leaf_pair_target_group =
      leaf_pair_target_group == "same-target-leaf"
          ? std::string("same-target-leaf")
          : std::string("scan");
  tree.leaf_pair_directness_probe =
      leaf_pair_directness_probe == "ancestor-path"
          ? std::string("ancestor-path")
          : std::string("full-traversal");
  tree.leaf_pair_directness_cache =
      leaf_pair_directness_cache == "target-leaf-source-leaf"
          ? std::string("target-leaf-source-leaf")
          : std::string("off");
  tree.leaf_pair_directness_verify =
      (leaf_pair_directness_verify == "sampled" ||
       leaf_pair_directness_verify == "full-small")
          ? leaf_pair_directness_verify
          : std::string("off");
  tree.leaf_pair_scheduler =
      (leaf_pair_scheduler == "deferred-worklist" ||
       leaf_pair_scheduler == "target-leaf-block" ||
       leaf_pair_scheduler == "cost-tile-worklist")
          ? leaf_pair_scheduler
          : std::string("inline-select");
  tree.leaf_pair_worklist_max_active = static_cast<std::size_t>(
      std::clamp(leaf_pair_worklist_max_active, 64, 8192));
  tree.leaf_pair_worklist_verify =
      (leaf_pair_worklist_verify == "sampled" ||
       leaf_pair_worklist_verify == "full-small")
          ? leaf_pair_worklist_verify
          : std::string("off");
  tree.leaf_block_verify =
      (leaf_block_verify == "sampled" || leaf_block_verify == "full-small")
          ? leaf_block_verify
          : std::string("off");
  tree.leaf_mac =
      (leaf_mac == "audit" || leaf_mac == "on") ? leaf_mac
                                                 : std::string("off");
  tree.fmm_m2l = fmm_m2l == "single-matvec-analytic"
                     ? std::string("single-matvec-analytic")
                     : std::string("off");
  tree.leaf_pair_max_lookahead = static_cast<std::size_t>(
      std::clamp(leaf_pair_max_lookahead, 4, 128));
  tree.convergence_split_selectivity =
      (convergence_split_selectivity == "absolute-aware" ||
       convergence_split_selectivity == "channel-absolute-aware")
          ? convergence_split_selectivity
          : std::string("legacy");
  tree.convergence_split_absolute_floor =
      std::clamp(std::isfinite(convergence_split_absolute_floor)
                     ? convergence_split_absolute_floor
                     : 0.0,
                 0.0, 1.0e-4);
  tree.convergence_overreject_audit =
      (convergence_overreject_audit == "sampled" ||
       convergence_overreject_audit == "full-small")
          ? convergence_overreject_audit
          : std::string("off");
  tree.convergence_overreject_detail =
      (convergence_overreject_detail == "summary" ||
       convergence_overreject_detail == "sampled-detail")
          ? convergence_overreject_detail
          : std::string("off");
  tree.convergence_shape_audit =
      (convergence_shape_audit == "summary" ||
       convergence_shape_audit == "sampled-detail")
          ? convergence_shape_audit
          : std::string("off");
  tree.convergence_kernel_response_audit =
      (convergence_kernel_response_audit == "summary" ||
       convergence_kernel_response_audit == "sampled-detail")
          ? convergence_kernel_response_audit
          : std::string("off");
  tree.interaction_mode =
      interaction_mode == "onthefly" ? std::string("onthefly")
                                     : std::string("cached");
  tree.node_bounds_mode =
      node_bounds_mode == "tight-aabb" ? std::string("tight-aabb")
                                       : std::string("moment");
  tree.tree_builder = tree_builder == "topdown-aabb"
                          ? std::string("topdown-aabb")
                          : std::string("grouped-morton");
  tree.mac_safety =
      (mac_safety == "audit" || mac_safety == "physical-gap")
          ? mac_safety
          : std::string("off");
  tree.mac_gap_factor =
      std::clamp(std::isfinite(mac_gap_factor) ? mac_gap_factor : 1.5, 0.5,
                 4.0);
  tree.moment_audit =
      (moment_audit == "sampled" || moment_audit == "full-small")
          ? moment_audit
          : std::string("off");
  tree.moment_construction =
      moment_construction == "direct-node" ? std::string("direct-node")
                                           : std::string("translated");
  tree.moment_sum =
      moment_sum == "kahan" ? std::string("kahan") : std::string("naive");
  tree.convergence_audit =
      (convergence_audit == "sampled" || convergence_audit == "full-small")
          ? convergence_audit
          : std::string("off");
  tree.convergence_safety =
      (convergence_safety == "audit" || convergence_safety == "split")
          ? convergence_safety
          : std::string("off");
  tree.convergence_rho_cap =
      std::clamp(std::isfinite(convergence_rho_cap) ? convergence_rho_cap
                                                    : 0.0,
                 0.0, 2.0);
  tree.mac_aspect_cap =
      std::clamp(std::isfinite(mac_aspect_cap) ? mac_aspect_cap : 0.0, 0.0,
                 50.0);
  tree.convergence_estimator =
      (convergence_estimator == "moment-tail" ||
       convergence_estimator == "hybrid")
          ? convergence_estimator
          : std::string("off");
  tree.convergence_tail_threshold =
      std::clamp(std::isfinite(convergence_tail_threshold)
                     ? convergence_tail_threshold
                     : 1.0e4,
                 1.0, 1.0e12);
  tree.convergence_estimator_error_cap =
      std::clamp(std::isfinite(convergence_estimator_error_cap)
                     ? convergence_estimator_error_cap
                     : 1.0e-3,
                 1.0e-8, 1.0);
  tree.convergence_unstable_rho_cap =
      std::clamp(std::isfinite(convergence_unstable_rho_cap)
                     ? convergence_unstable_rho_cap
                     : 0.25,
                 0.05, 2.0);
  tree.convergence_calibration =
      (convergence_calibration == "consecutive" ||
       convergence_calibration == "aspect" ||
       convergence_calibration == "consecutive-aspect")
          ? convergence_calibration
          : std::string("stage3j");
  tree.convergence_dielectric_amp =
      convergence_dielectric_amp == "on" ? std::string("on")
                                         : std::string("off");
  tree.convergence_estimator_model =
      (convergence_estimator_model == "channel-tail" ||
       convergence_estimator_model == "channel-tail-eps")
          ? convergence_estimator_model
          : std::string("stage3l");
  tree.convergence_channel_calibration =
      (convergence_channel_calibration == "k3-sqrt-eps" ||
       convergence_channel_calibration == "k3-eps" ||
       convergence_channel_calibration == "k4-sqrt-eps" ||
       convergence_channel_calibration == "k4-eps" ||
       convergence_channel_calibration == "all-derivative-sqrt-eps" ||
       convergence_channel_calibration == "polynomial-eps")
          ? convergence_channel_calibration
          : std::string("stage3m");
  tree.convergence_split_policy =
      convergence_split_policy == "k3-hot" ? std::string("k3-hot")
                                           : std::string("global");
  tree.channel_missed_semantics =
      channel_missed_semantics == "accepted-only" ? std::string("accepted-only")
                                                  : std::string("legacy");
  tree.accepted_miss_diagnostics =
      (accepted_miss_diagnostics == "summary" ||
       accepted_miss_diagnostics == "sampled-detail")
          ? accepted_miss_diagnostics
          : std::string("off");
  tree.accepted_miss_absolute_floor =
      std::clamp(std::isfinite(accepted_miss_absolute_floor)
                     ? accepted_miss_absolute_floor
                     : 1.0e-7,
                 0.0, 1.0e-4);
  tree.convergence_k3_accepted_rho_cap = std::clamp(
      std::isfinite(convergence_k3_accepted_rho_cap)
          ? convergence_k3_accepted_rho_cap
          : 0.0,
      0.0, 2.0);
  tree.mac_distance_policy =
      mac_distance_policy == "surface-margin" ? std::string("surface-margin")
                                              : std::string("center");
  tree.mac_surface_margin_factor = std::clamp(
      std::isfinite(mac_surface_margin_factor) ? mac_surface_margin_factor
                                               : 0.25,
      0.0, 0.75);
  tree.mac_surface_margin_policy =
      (mac_surface_margin_policy == "adaptive" ||
       mac_surface_margin_policy == "risk-scaled")
          ? mac_surface_margin_policy
          : std::string("all");
  tree.mac_surface_margin_aspect_threshold = std::clamp(
      std::isfinite(mac_surface_margin_aspect_threshold)
          ? mac_surface_margin_aspect_threshold
          : 3.0,
      1.0, 50.0);
  tree.mac_surface_margin_rho_trigger = std::clamp(
      std::isfinite(mac_surface_margin_rho_trigger)
          ? mac_surface_margin_rho_trigger
          : 0.55,
      0.0, 2.0);
  tree.mac_surface_margin_soft_aspect_threshold = std::clamp(
      std::isfinite(mac_surface_margin_soft_aspect_threshold)
          ? mac_surface_margin_soft_aspect_threshold
          : 0.0,
      0.0, 50.0);
  tree.mac_surface_margin_soft_rho_trigger = std::clamp(
      std::isfinite(mac_surface_margin_soft_rho_trigger)
          ? mac_surface_margin_soft_rho_trigger
          : 0.0,
      0.0, 2.0);
  tree.mac_surface_margin_soft_factor = std::clamp(
      std::isfinite(mac_surface_margin_soft_factor)
          ? mac_surface_margin_soft_factor
          : 0.10,
      0.0, 0.75);
  tree.expansion_center =
      (expansion_center == "area-centroid" ||
       expansion_center == "aabb-center")
          ? expansion_center
          : std::string("legacy");
  tree.fragmentation_safety =
      (fragmentation_safety == "audit" || fragmentation_safety == "split")
          ? fragmentation_safety
          : std::string("off");
  tree.expansion_audit =
      (expansion_audit == "sampled" || expansion_audit == "full-small")
          ? expansion_audit
          : std::string("off");
  tree.expansion_audit_channel =
      (expansion_audit_channel == "summary" ||
       expansion_audit_channel == "sampled-detail")
          ? expansion_audit_channel
          : std::string("off");
  tree.expansion_audit_sample_count = static_cast<std::size_t>(
      std::clamp(expansion_audit_sample_count, 64, 2048));
  tree.expansion_audit_max_panels = static_cast<std::size_t>(
      std::clamp(expansion_audit_max_panels, 128, 8192));
  tree.quad_buffer =
      (quad_buffer == "gauss7" || quad_buffer == "adaptive") ? quad_buffer
                                                             : "off";
  if (panels.empty())
    return tree;

  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  for (const Panel &panel : panels) {
    min_x = std::min(min_x, panel.x);
    min_y = std::min(min_y, panel.y);
    min_z = std::min(min_z, panel.z);
    max_x = std::max(max_x, panel.x);
    max_y = std::max(max_y, panel.y);
    max_z = std::max(max_z, panel.z);
  }
  const double extent =
      std::max({max_x - min_x, max_y - min_y, max_z - min_z, 1.0e-9});
  auto quantize = [&](double value, double origin) noexcept {
    const double scaled =
        ((value - origin) / extent) * static_cast<double>(kMortonMax);
    const double clamped =
        std::clamp(scaled, 0.0, static_cast<double>(kMortonMax));
    return static_cast<std::uint32_t>(clamped);
  };

  std::vector<MortonEntry> entries(panels.size());
  const std::size_t workers = effective_workers(executor, panels.size());
  executor.parallel_for(
      panels.size(), workers, 256,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          const Panel &panel = panels[i];
          entries[i].code =
              morton_code(quantize(panel.x, min_x), quantize(panel.y, min_y),
                          quantize(panel.z, min_z));
          entries[i].index = i;
        }
      });
  entries = radix_sort_morton(std::move(entries));

  std::vector<std::uint64_t> panel_morton(panels.size(), 0);
  for (const MortonEntry &entry : entries)
    panel_morton[entry.index] = entry.code;
  tree.order.reserve(entries.size());
  tree.morton.reserve(entries.size());
  for (const MortonEntry &entry : entries) {
    tree.order.push_back(entry.index);
    tree.morton.push_back(entry.code);
  }

  const bool tight_bounds = tree.node_bounds_mode == "tight-aabb";
  if (tree.tree_builder == "topdown-aabb") {
    std::vector<TopDownTempNode> temp_nodes;
    temp_nodes.reserve((panels.size() + effective_leaf_size - 1) /
                           effective_leaf_size * 2 +
                       8);
    const std::size_t temp_root = build_topdown_temp_node(
        panels, tree.order, 0, panels.size(), effective_leaf_size, 0,
        temp_nodes);
    flatten_topdown_tree(panels, tree, temp_nodes, temp_root, tight_bounds,
                         tree.expansion_center);
  } else {
    std::vector<std::size_t> level_indices;
    const std::size_t leaf_count =
        (panels.size() + effective_leaf_size - 1) / effective_leaf_size;
    tree.nodes.reserve(leaf_count * 2 + 8);
    level_indices.reserve(leaf_count);
    for (std::size_t leaf = 0; leaf < leaf_count; ++leaf) {
      const std::size_t first_panel = leaf * effective_leaf_size;
      const std::size_t count =
          std::min(effective_leaf_size, panels.size() - first_panel);
      level_indices.push_back(tree.nodes.size());
      tree.nodes.push_back(make_leaf_node(panels, tree.order, first_panel,
                                          count, tight_bounds,
                                          tree.expansion_center));
    }
    tree.leaf_count = tree.nodes.size();
    tree.moment_levels.clear();
    tree.moment_levels.push_back({0, tree.leaf_count});

    while (level_indices.size() > 1) {
      std::vector<std::size_t> next_level;
      next_level.reserve((level_indices.size() + kBranchFactor - 1) /
                         kBranchFactor);
      const std::size_t level_begin = tree.nodes.size();
      for (std::size_t begin = 0; begin < level_indices.size();
           begin += kBranchFactor) {
        const std::size_t child_count =
            std::min(kBranchFactor, level_indices.size() - begin);
        std::vector<std::size_t> children;
        children.reserve(child_count);
        for (std::size_t child = 0; child < child_count; ++child)
          children.push_back(level_indices[begin + child]);
        TreeNode parent = make_parent_node(tree.nodes, children, tight_bounds,
                                           tree.expansion_center);
        attach_children(tree, parent, children);
        next_level.push_back(tree.nodes.size());
        tree.nodes.push_back(parent);
      }
      tree.moment_levels.push_back({level_begin, tree.nodes.size()});
      level_indices.swap(next_level);
    }
    tree.root_index = level_indices.empty() ? 0 : level_indices.front();
    tree.depth_max =
        tree.moment_levels.empty() ? 0 : tree.moment_levels.size() - 1;
  }
  tree.morton.clear();
  tree.morton.reserve(tree.order.size());
  for (std::size_t panel_index : tree.order)
    tree.morton.push_back(panel_index < panel_morton.size()
                              ? panel_morton[panel_index]
                              : std::uint64_t{0});

  if (!tree.nodes.empty()) {
    double radius_sum = 0.0;
    double aspect_sum = 0.0;
    for (const TreeNode &node : tree.nodes) {
      tree.node_radius_max = std::max(tree.node_radius_max, node.radius);
      radius_sum += node.radius;
      const double aspect = node_aspect_ratio(node);
      tree.node_aspect_ratio_max =
          std::max(tree.node_aspect_ratio_max, aspect);
      aspect_sum += aspect;
    }
    tree.node_radius_mean = radius_sum / static_cast<double>(tree.nodes.size());
    tree.node_aspect_ratio_mean =
        aspect_sum / static_cast<double>(tree.nodes.size());
  }
  annotate_fragmented_nodes(tree);
  const std::size_t invalid_node = std::numeric_limits<std::size_t>::max();
  tree.parent_index_by_node.assign(tree.nodes.size(), invalid_node);
  for (std::size_t node_index = 0; node_index < tree.nodes.size();
       ++node_index) {
    const TreeNode &node = tree.nodes[node_index];
    for (std::size_t child = 0; child < node.child_count; ++child) {
      const std::size_t child_index = child_node_index(tree, node, child);
      if (child_index < tree.parent_index_by_node.size())
        tree.parent_index_by_node[child_index] = node_index;
    }
  }
  tree.leaf_ancestor_path_by_node.assign(tree.nodes.size(), {});
  for (std::size_t node_index = 0; node_index < tree.nodes.size();
       ++node_index) {
    const TreeNode &node = tree.nodes[node_index];
    if (node.child_count != 0)
      continue;
    std::vector<std::size_t> reverse_path;
    for (std::size_t cursor = node_index; cursor < tree.nodes.size();) {
      reverse_path.push_back(cursor);
      const std::size_t parent =
          cursor < tree.parent_index_by_node.size()
              ? tree.parent_index_by_node[cursor]
              : invalid_node;
      if (parent == invalid_node || parent == cursor ||
          parent >= tree.nodes.size())
        break;
      cursor = parent;
    }
    tree.leaf_ancestor_path_by_node[node_index].assign(
        reverse_path.rbegin(), reverse_path.rend());
  }
  tree.target_leaf_index_by_panel.assign(
      panels.size(), std::numeric_limits<std::size_t>::max());
  tree.target_panels_by_leaf.assign(tree.nodes.size(), {});
  for (std::size_t node_index = 0; node_index < tree.nodes.size();
       ++node_index) {
    const TreeNode &node = tree.nodes[node_index];
    if (node.child_count != 0)
      continue;
    const std::size_t end = node.first_panel + node.panel_count;
    for (std::size_t pos = node.first_panel; pos < end; ++pos) {
      if (pos >= tree.order.size())
        continue;
      const std::size_t panel_index = tree.order[pos];
      if (panel_index < tree.target_leaf_index_by_panel.size()) {
        tree.target_leaf_index_by_panel[panel_index] = node_index;
        tree.target_panels_by_leaf[node_index].push_back(panel_index);
      }
    }
  }
  for (std::vector<std::size_t> &leaf_panels : tree.target_panels_by_leaf)
    std::sort(leaf_panels.begin(), leaf_panels.end());
  tree.build_ms = elapsed_ms(build_begin, std::chrono::steady_clock::now());
  tree.panel_cache = build_tree_panel_cache(panels, tree.order);
  if (tree.interaction_mode != "onthefly")
    tree.interactions = build_tree_interactions(panels, tree, executor);
  return tree;
}


} // namespace TitanPBE::Tabi::detail
