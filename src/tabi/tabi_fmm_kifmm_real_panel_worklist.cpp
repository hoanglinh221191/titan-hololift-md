#include "tabi_fmm_kifmm_real_panel_worklist.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace titan::tabi::fmm_kifmm {
namespace {

using TitanPBE::Tabi::Panel;

[[maybe_unused]] constexpr const char *kStrictLocalCheckGuardMarker =
    "strict-local-check";
[[maybe_unused]] constexpr const char *kLocalCheckViolationTelemetryMarker =
    "negative_target_local_check_excess_pairs";

double axis_gap(double t_min, double t_max, double s_center,
                double s_half) noexcept {
  if (t_max < s_center - s_half) {
    return (s_center - s_half) - t_max;
  }
  if (t_min > s_center + s_half) {
    return t_min - (s_center + s_half);
  }
  return -std::min(t_max - (s_center - s_half),
                   (s_center + s_half) - t_min);
}

double target_envelope_check_excess(
    const KifmmRealPanelNodeEnvelope &target_envelope,
    const KifmmHierarchyNode &source_node,
    double source_check_half_width) noexcept {
  const double gx = axis_gap(target_envelope.vertex_min.x,
                             target_envelope.vertex_max.x,
                             source_node.center.x, source_check_half_width);
  const double gy = axis_gap(target_envelope.vertex_min.y,
                             target_envelope.vertex_max.y,
                             source_node.center.y, source_check_half_width);
  const double gz = axis_gap(target_envelope.vertex_min.z,
                             target_envelope.vertex_max.z,
                             source_node.center.z, source_check_half_width);
  return std::max({gx, gy, gz});
}

bool real_panel_center_mac(const KifmmHierarchyNode &source,
                           const KifmmHierarchyNode &target,
                           const KifmmHierarchyPolicy &policy) noexcept {
  const double radius = std::sqrt(3.0) * source.half_width;
  const double d = norm(sub(target.center, source.center));
  return d > 0.0 && radius / d <= policy.theta;
}

bool target_node_box_outside_source_check_box(
    const KifmmHierarchyNode &source, const KifmmHierarchyNode &target,
    const KifmmHierarchyPolicy &policy) noexcept {
  const double source_check_half_width =
      source.half_width * policy.upward_check_scale;
  const double dx = std::abs(target.center.x - source.center.x);
  const double dy = std::abs(target.center.y - source.center.y);
  const double dz = std::abs(target.center.z - source.center.z);
  return (dx - target.half_width > source_check_half_width) ||
         (dy - target.half_width > source_check_half_width) ||
         (dz - target.half_width > source_check_half_width);
}

double child_local_check_containment_margin(
    const KifmmHierarchyNode &parent, const KifmmHierarchyNode &child,
    const KifmmHierarchyPolicy &policy) noexcept {
  const double parent_local_check_half_width =
      parent.half_width * policy.local_check_scale;
  const double child_local_check_half_width =
      child.half_width * policy.local_check_scale;
  const double mx = parent_local_check_half_width -
                    std::abs(child.center.x - parent.center.x) -
                    child_local_check_half_width;
  const double my = parent_local_check_half_width -
                    std::abs(child.center.y - parent.center.y) -
                    child_local_check_half_width;
  const double mz = parent_local_check_half_width -
                    std::abs(child.center.z - parent.center.z) -
                    child_local_check_half_width;
  return std::min({mx, my, mz});
}

bool node_has_source_children(const KifmmHierarchyNode &node,
                              const KifmmHierarchyTree &tree) {
  for (const int child_id : node.child) {
    if (child_id >= 0 &&
        !tree.nodes[static_cast<std::size_t>(child_id)]
             .source_indices.empty()) {
      return true;
    }
  }
  return false;
}

bool node_has_target_children(const KifmmHierarchyNode &node,
                              const KifmmHierarchyTree &tree) {
  for (const int child_id : node.child) {
    if (child_id >= 0 &&
        !tree.nodes[static_cast<std::size_t>(child_id)]
             .target_indices.empty()) {
      return true;
    }
  }
  return false;
}

void add_real_guard_direct_pair(KifmmHierarchyWorklist &worklist,
                                KifmmHierarchyRealPanelResult &result,
                                int target_node_id, int source_node_id) {
  worklist.direct_leaf_pairs.push_back({target_node_id, source_node_id});
  ++worklist.leaf_direct_pairs;
  ++result.real_guard_direct_pairs;
}

void classify_real_panel_pair(
    const KifmmHierarchyTree &tree, const KifmmHierarchyPolicy &policy,
    std::span<const KifmmRealPanelNodeEnvelope> source_envelopes,
    std::span<const KifmmRealPanelNodeEnvelope> target_envelopes,
    double guard_margin_factor, bool local_check_guard, int source_node_id,
    int target_node_id, KifmmHierarchyWorklist &worklist,
    KifmmHierarchyRealPanelResult &result) {
  const KifmmHierarchyNode &source =
      tree.nodes[static_cast<std::size_t>(source_node_id)];
  const KifmmHierarchyNode &target =
      tree.nodes[static_cast<std::size_t>(target_node_id)];
  if (source.source_indices.empty() || target.target_indices.empty()) {
    return;
  }

  if (real_panel_center_mac(source, target, policy)) {
    const KifmmRealPanelNodeEnvelope &source_envelope =
        source_envelopes[static_cast<std::size_t>(source_node_id)];
    const KifmmRealPanelNodeEnvelope &target_envelope =
        target_envelopes[static_cast<std::size_t>(target_node_id)];
    const double margin = guard_margin_factor *
                          std::max(source_envelope.max_panel_diagonal,
                                   target_envelope.max_panel_diagonal);
    result.real_panel_guard_margin_max =
        std::max(result.real_panel_guard_margin_max, margin);
    const double envelope_excess = target_envelope_check_excess(
        target_envelope, source, source.half_width * policy.upward_check_scale);
    const double local_excess =
        target_local_check_excess(source, target, policy);
    const bool admissible =
        local_check_guard
            ? local_excess > margin
            : target_node_box_outside_source_check_box(source, target,
                                                       policy) &&
                  envelope_excess > margin;
    if (admissible) {
      worklist.m2l_pairs.push_back({target_node_id, source_node_id});
      ++worklist.mac_accepted_pairs;
      return;
    }
    ++result.real_guard_rejected_pairs;
  }

  const bool can_split_source = node_has_source_children(source, tree);
  const bool can_split_target = node_has_target_children(target, tree);
  if (!can_split_source && !can_split_target) {
    add_real_guard_direct_pair(worklist, result, target_node_id,
                               source_node_id);
    return;
  }

  const bool split_target =
      can_split_target &&
      (!can_split_source || target.half_width > source.half_width ||
       (target.half_width == source.half_width &&
        target.target_indices.size() >= source.source_indices.size()));
  ++result.real_guard_split_pairs;
  if (split_target) {
    for (const int child_id : target.child) {
      if (child_id >= 0 &&
          !tree.nodes[static_cast<std::size_t>(child_id)]
               .target_indices.empty()) {
        classify_real_panel_pair(tree, policy, source_envelopes,
                                 target_envelopes, guard_margin_factor,
                                 local_check_guard, source_node_id, child_id,
                                 worklist, result);
      }
    }
    return;
  }

  for (const int child_id : source.child) {
    if (child_id >= 0 &&
        !tree.nodes[static_cast<std::size_t>(child_id)]
             .source_indices.empty()) {
      classify_real_panel_pair(tree, policy, source_envelopes,
                               target_envelopes, guard_margin_factor,
                               local_check_guard, child_id, target_node_id,
                               worklist, result);
    }
  }
}

} // namespace

int mac_bin(double ratio) noexcept {
  if (ratio < 0.4) {
    return 0;
  }
  if (ratio < 0.6) {
    return 1;
  }
  if (ratio < 0.8) {
    return 2;
  }
  if (ratio < 1.0) {
    return 3;
  }
  return 4;
}

double target_vertex_check_excess(const Panel &target,
                                  const KifmmHierarchyNode &source,
                                  double check_scale) noexcept {
  const double check_half = source.half_width * check_scale;
  const Vec3 center = source.center;
  const double e0 = max_abs_component(sub(vertex0(target), center)) - check_half;
  const double e1 = max_abs_component(sub(vertex1(target), center)) - check_half;
  const double e2 = max_abs_component(sub(vertex2(target), center)) - check_half;
  return std::min({e0, e1, e2});
}

double source_boundary_margin(const KifmmPanelSource &source,
                              const KifmmHierarchyNode &node) noexcept {
  return node.half_width - max_abs_component(sub(source.position, node.center));
}

double target_local_check_excess(const KifmmHierarchyNode &source,
                                 const KifmmHierarchyNode &target,
                                 const KifmmHierarchyPolicy &policy) noexcept {
  const double source_check_half_width =
      source.half_width * policy.upward_check_scale;
  const double target_local_check_half_width =
      target.half_width * policy.local_check_scale;
  const double dx = std::abs(target.center.x - source.center.x);
  const double dy = std::abs(target.center.y - source.center.y);
  const double dz = std::abs(target.center.z - source.center.z);
  return std::max({dx - source_check_half_width - target_local_check_half_width,
                   dy - source_check_half_width - target_local_check_half_width,
                   dz - source_check_half_width -
                       target_local_check_half_width});
}

KifmmHierarchyWorklist build_real_panel_guarded_worklist(
    const KifmmHierarchyTree &tree, const KifmmHierarchyPolicy &policy,
    std::span<const KifmmRealPanelNodeEnvelope> source_envelopes,
    std::span<const KifmmRealPanelNodeEnvelope> target_envelopes,
    double guard_margin_factor, bool local_check_guard,
    KifmmHierarchyRealPanelResult &result) {
  KifmmHierarchyWorklist worklist;
  worklist.tree_hash = tree.tree_hash;
  if (tree.nodes.empty() || tree.status != "ok") {
    return worklist;
  }
  classify_real_panel_pair(tree, policy, source_envelopes, target_envelopes,
                           guard_margin_factor, local_check_guard, 0, 0,
                           worklist, result);
  finalize_hierarchy_worklist(worklist, policy);
  return worklist;
}

void update_leaf_panel_telemetry(
    const KifmmHierarchyTree &tree,
    std::span<const KifmmRealPanelNodeEnvelope> source_envelopes,
    std::span<const KifmmRealPanelNodeEnvelope> target_envelopes,
    KifmmHierarchyRealPanelResult &result) {
  result.min_leaf_half_width_over_panel_radius =
      std::numeric_limits<double>::infinity();
  result.panels_crossing_node_boundary = 0;
  for (std::size_t node_id = 0; node_id < tree.nodes.size(); ++node_id) {
    const KifmmHierarchyNode &node = tree.nodes[node_id];
    bool is_leaf = true;
    for (const int child_id : node.child) {
      if (child_id >= 0) {
        is_leaf = false;
        break;
      }
    }
    if (!is_leaf) {
      continue;
    }
    for (const KifmmRealPanelNodeEnvelope *envelope :
         {&source_envelopes[node_id], &target_envelopes[node_id]}) {
      if (envelope->panel_count == 0) {
        continue;
      }
      if (envelope->max_panel_radius > 0.0) {
        result.min_leaf_half_width_over_panel_radius =
            std::min(result.min_leaf_half_width_over_panel_radius,
                     node.half_width / envelope->max_panel_radius);
      }
      if (envelope->vertex_min.x < node.center.x - node.half_width ||
          envelope->vertex_min.y < node.center.y - node.half_width ||
          envelope->vertex_min.z < node.center.z - node.half_width ||
          envelope->vertex_max.x > node.center.x + node.half_width ||
          envelope->vertex_max.y > node.center.y + node.half_width ||
          envelope->vertex_max.z > node.center.z + node.half_width) {
        result.panels_crossing_node_boundary +=
            static_cast<std::uint64_t>(envelope->panel_count);
      }
    }
  }
  if (!std::isfinite(result.min_leaf_half_width_over_panel_radius)) {
    result.min_leaf_half_width_over_panel_radius =
        std::numeric_limits<double>::quiet_NaN();
  }
}

void update_l2l_containment_telemetry(const KifmmHierarchyTree &tree,
                                      const KifmmHierarchyPolicy &policy,
                                      KifmmHierarchyRealPanelResult &result) {
  result.child_local_check_inside_parent_local_check_min_margin =
      std::numeric_limits<double>::infinity();
  result.negative_l2l_local_containment_pairs = 0;
  for (const KifmmHierarchyNode &parent : tree.nodes) {
    for (const int child_id : parent.child) {
      if (child_id < 0) {
        continue;
      }
      const KifmmHierarchyNode &child =
          tree.nodes[static_cast<std::size_t>(child_id)];
      const double margin =
          child_local_check_containment_margin(parent, child, policy);
      result.child_local_check_inside_parent_local_check_min_margin = std::min(
          result.child_local_check_inside_parent_local_check_min_margin,
          margin);
      if (margin < 0.0) {
        ++result.negative_l2l_local_containment_pairs;
      }
    }
  }
  if (!std::isfinite(
          result.child_local_check_inside_parent_local_check_min_margin)) {
    result.child_local_check_inside_parent_local_check_min_margin =
        std::numeric_limits<double>::quiet_NaN();
  }
}

} // namespace titan::tabi::fmm_kifmm
