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
const char *jcp_expansion_channel_label(std::size_t channel) noexcept {
  switch (channel) {
  case kJcpK1PhiChannel:
    return "K1_phi";
  case kJcpK2PhiNormalChannel:
    return "K2_phi_normal";
  case kJcpK3PhinChannel:
    return "K3_phin";
  case kJcpK4MixedChannel:
    return "K4_mixed";
  default:
    return "none";
  }
}

const char *jcp_kernel_source_family(std::size_t channel) noexcept {
  switch (channel) {
  case kJcpK1PhiChannel:
  case kJcpK3PhinChannel:
    return "phin_scalar";
  case kJcpK2PhiNormalChannel:
  case kJcpK4MixedChannel:
    return "phi_normal";
  default:
    return "mixed";
  }
}

std::size_t effective_workers(const TitanExecutor &executor,
                              std::size_t work_items) noexcept {
  if (work_items == 0)
    return 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(executor.capacity(), work_items));
}

double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) noexcept {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

double point_distance(double ax, double ay, double az, double bx, double by,
                      double bz) noexcept {
  const double dx = ax - bx;
  const double dy = ay - by;
  const double dz = az - bz;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double tree_panel_max_edge_length(const Panel &panel) noexcept {
  if (!panel.has_triangle_geometry)
    return std::sqrt(std::max(0.0, panel.area));
  const double e01 = point_distance(panel.v0x, panel.v0y, panel.v0z, panel.v1x,
                                    panel.v1y, panel.v1z);
  const double e12 = point_distance(panel.v1x, panel.v1y, panel.v1z, panel.v2x,
                                    panel.v2y, panel.v2z);
  const double e20 = point_distance(panel.v2x, panel.v2y, panel.v2z, panel.v0x,
                                    panel.v0y, panel.v0z);
  return std::max({e01, e12, e20, std::sqrt(std::max(0.0, panel.area))});
}

void panel_aabb(const Panel &panel, double &min_x, double &min_y, double &min_z,
                double &max_x, double &max_y, double &max_z) noexcept {
  if (!panel.has_triangle_geometry) {
    min_x = max_x = panel.x;
    min_y = max_y = panel.y;
    min_z = max_z = panel.z;
    return;
  }
  min_x = std::min({panel.v0x, panel.v1x, panel.v2x});
  min_y = std::min({panel.v0y, panel.v1y, panel.v2y});
  min_z = std::min({panel.v0z, panel.v1z, panel.v2z});
  max_x = std::max({panel.v0x, panel.v1x, panel.v2x});
  max_y = std::max({panel.v0y, panel.v1y, panel.v2y});
  max_z = std::max({panel.v0z, panel.v1z, panel.v2z});
}

double aabb_gap(double amin_x, double amin_y, double amin_z, double amax_x,
                double amax_y, double amax_z, double bmin_x, double bmin_y,
                double bmin_z, double bmax_x, double bmax_y,
                double bmax_z) noexcept {
  const double dx = std::max({bmin_x - amax_x, amin_x - bmax_x, 0.0});
  const double dy = std::max({bmin_y - amax_y, amin_y - bmax_y, 0.0});
  const double dz = std::max({bmin_z - amax_z, amin_z - bmax_z, 0.0});
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double panel_node_aabb_gap(const Panel &target,
                           const TreeNode &node) noexcept {
  double min_x = 0.0;
  double min_y = 0.0;
  double min_z = 0.0;
  double max_x = 0.0;
  double max_y = 0.0;
  double max_z = 0.0;
  panel_aabb(target, min_x, min_y, min_z, max_x, max_y, max_z);
  return aabb_gap(min_x, min_y, min_z, max_x, max_y, max_z, node.min_x,
                  node.min_y, node.min_z, node.max_x, node.max_y, node.max_z);
}

double node_aspect_ratio(const TreeNode &node) noexcept {
  const double ex = std::max(0.0, node.max_x - node.min_x);
  const double ey = std::max(0.0, node.max_y - node.min_y);
  const double ez = std::max(0.0, node.max_z - node.min_z);
  const double hi = std::max({ex, ey, ez});
  const double lo = std::max(1.0e-12, std::min({ex > 0.0 ? ex : hi,
                                                 ey > 0.0 ? ey : hi,
                                                 ez > 0.0 ? ez : hi}));
  return hi > 0.0 ? hi / lo : 1.0;
}

void add_leaf_work(LeafWorkTally &tally, LeafWorkReason reason,
                   std::size_t panels) noexcept {
  if (panels == 0)
    return;
  switch (reason) {
  case LeafWorkReason::PhysicalGap:
    tally.physical_gap_panels += panels;
    break;
  case LeafWorkReason::Fragmentation:
    tally.fragmentation_panels += panels;
    break;
  case LeafWorkReason::Convergence:
    tally.convergence_panels += panels;
    break;
  case LeafWorkReason::ConvergenceEstimator:
    tally.convergence_panels += panels;
    tally.convergence_estimator_panels += panels;
    break;
  case LeafWorkReason::ConvergenceRho:
    tally.convergence_panels += panels;
    tally.convergence_rho_panels += panels;
    break;
  case LeafWorkReason::ConvergenceAspect:
    tally.convergence_panels += panels;
    tally.convergence_aspect_panels += panels;
    break;
  case LeafWorkReason::SurfaceMargin:
    tally.surface_margin_panels += panels;
    tally.surface_margin_hard_panels += panels;
    break;
  case LeafWorkReason::SurfaceMarginHard:
    tally.surface_margin_panels += panels;
    tally.surface_margin_hard_panels += panels;
    break;
  case LeafWorkReason::SurfaceMarginSoft:
    tally.surface_margin_panels += panels;
    tally.surface_margin_soft_panels += panels;
    break;
  case LeafWorkReason::K3RhoCap:
    tally.k3_rho_cap_panels += panels;
    break;
  case LeafWorkReason::SingularFallback:
    tally.singular_fallback_panels += panels;
    break;
  case LeafWorkReason::ScalarTail:
    tally.scalar_tail_panels += panels;
    break;
  case LeafWorkReason::MacTerminal:
  default:
    tally.mac_terminal_panels += panels;
    break;
  }
}

std::size_t leaf_work_tally_total(const LeafWorkTally &tally) noexcept {
  return tally.mac_terminal_panels + tally.physical_gap_panels +
         tally.fragmentation_panels + tally.convergence_panels +
         tally.surface_margin_panels + tally.k3_rho_cap_panels +
         tally.singular_fallback_panels + tally.scalar_tail_panels;
}

void record_far_geometric_audit(MacAuditTally &tally, const TreeNode &node,
                                double gap, double gap_ratio,
                                bool unsafe) noexcept {
  ++tally.samples;
  if (!unsafe)
    return;
  ++tally.unsafe_count;
  tally.unsafe_panels += node.panel_count;
  if (tally.unsafe_count == 1 || gap < tally.min_gap)
    tally.min_gap = gap;
  if (tally.unsafe_count == 1 || gap_ratio < tally.min_gap_ratio)
    tally.min_gap_ratio = gap_ratio;
  tally.sum_gap_ratio += gap_ratio;
}

void merge_far_geometric_audit(MacAuditTally &dst,
                               const MacAuditTally &src) noexcept {
  if (src.samples == 0)
    return;
  dst.samples += src.samples;
  dst.physical_gap_rejects += src.physical_gap_rejects;
  if (src.unsafe_count == 0)
    return;
  if (dst.unsafe_count == 0 || src.min_gap < dst.min_gap)
    dst.min_gap = src.min_gap;
  if (dst.unsafe_count == 0 || src.min_gap_ratio < dst.min_gap_ratio)
    dst.min_gap_ratio = src.min_gap_ratio;
  dst.unsafe_count += src.unsafe_count;
  dst.unsafe_panels += src.unsafe_panels;
  dst.sum_gap_ratio += src.sum_gap_ratio;
}

void copy_far_geometric_audit_to_stats(TreecodeStats &stats,
                                       const MacAuditTally &tally) noexcept {
  stats.far_geometric_audit_samples += tally.samples;
  stats.mac_physical_gap_rejects += tally.physical_gap_rejects;
  if (tally.unsafe_count == 0)
    return;
  if (stats.far_geometric_unsafe_count == 0 ||
      tally.min_gap < stats.far_geometric_min_gap)
    stats.far_geometric_min_gap = tally.min_gap;
  if (stats.far_geometric_unsafe_count == 0 ||
      tally.min_gap_ratio < stats.far_geometric_gap_ratio_min)
    stats.far_geometric_gap_ratio_min = tally.min_gap_ratio;
  const double old_count =
      static_cast<double>(stats.far_geometric_unsafe_count);
  const double new_count = old_count + static_cast<double>(tally.unsafe_count);
  const double old_sum = stats.far_geometric_gap_ratio_mean * old_count;
  stats.far_geometric_unsafe_count += tally.unsafe_count;
  stats.far_geometric_unsafe_panels += tally.unsafe_panels;
  stats.far_geometric_gap_ratio_mean =
      new_count > 0.0 ? (old_sum + tally.sum_gap_ratio) / new_count : 0.0;
}

void record_fragmentation_audit(FragmentationTally &tally,
                                const TreeNode &node,
                                bool split_reject) noexcept {
  ++tally.far_count;
  tally.far_panels += node.panel_count;
  if (split_reject)
    ++tally.split_rejects;
  tally.gap_ratio_sum += node.fragmentation_center_gap_ratio;
  tally.gap_ratio_max =
      std::max(tally.gap_ratio_max, node.fragmentation_center_gap_ratio);
}

void merge_fragmentation_audit(FragmentationTally &dst,
                               const FragmentationTally &src) noexcept {
  if (src.far_count == 0)
    return;
  dst.far_count += src.far_count;
  dst.far_panels += src.far_panels;
  dst.split_rejects += src.split_rejects;
  dst.gap_ratio_sum += src.gap_ratio_sum;
  dst.gap_ratio_max = std::max(dst.gap_ratio_max, src.gap_ratio_max);
}

void copy_fragmentation_audit_to_stats(
    TreecodeStats &stats, const FragmentationTally &tally) noexcept {
  if (tally.far_count == 0)
    return;
  const double old_count = static_cast<double>(stats.fragmented_far_count);
  const double new_count = old_count + static_cast<double>(tally.far_count);
  const double old_sum =
      stats.fragmentation_center_gap_ratio_mean * old_count;
  stats.fragmented_far_count += tally.far_count;
  stats.fragmented_far_panels += tally.far_panels;
  stats.fragmented_split_rejects += tally.split_rejects;
  stats.fragmentation_center_gap_ratio_max =
      std::max(stats.fragmentation_center_gap_ratio_max,
               tally.gap_ratio_max);
  stats.fragmentation_center_gap_ratio_mean =
      new_count > 0.0 ? (old_sum + tally.gap_ratio_sum) / new_count : 0.0;
}

bool should_audit_far_expansion(
    const TreeWorkspace &tree, std::size_t panel_count, const TreeNode &node,
    std::atomic_size_t &sampled_candidates) noexcept {
  if (tree.expansion_audit == "off" || node.panel_count == 0)
    return false;
  if (node.panel_count > tree.expansion_audit_max_panels)
    return false;
  if (tree.expansion_audit == "full-small" &&
      panel_count <= tree.expansion_audit_max_panels)
    return true;
  const std::size_t slot =
      sampled_candidates.fetch_add(1, std::memory_order_relaxed);
  return slot < tree.expansion_audit_sample_count;
}

bool should_audit_far_expansion_mode(
    const TreeWorkspace &tree, const std::string &mode,
    std::size_t panel_count, const TreeNode &node,
    std::atomic_size_t &sampled_candidates) noexcept {
  if (mode == "off" || node.panel_count == 0)
    return false;
  if (node.panel_count > tree.expansion_audit_max_panels)
    return false;
  if (mode == "full-small" && panel_count <= tree.expansion_audit_max_panels)
    return true;
  const std::size_t slot =
      sampled_candidates.fetch_add(1, std::memory_order_relaxed);
  return slot < tree.expansion_audit_sample_count;
}

void record_accepted_miss_diagnostic(
    ExpansionAuditTally &tally, std::size_t target_index,
    std::size_t source_node_index, std::size_t channel, std::size_t panels,
    double rho, double tail_ratio, double estimated_error, double rel_error,
    double abs_error, bool is_phantom, const std::string &mode) {
  if (channel >= kJcpExpansionChannelCount || mode == "off")
    return;
  if (!is_phantom) {
    ++tally.accepted_miss_channel_count[channel];
    tally.accepted_miss_channel_panels[channel] += panels;
    tally.accepted_miss_channel_rho_sum[channel] += rho;
    if (tally.accepted_miss_channel_count[channel] == 1 ||
        rho < tally.accepted_miss_channel_rho_min[channel])
      tally.accepted_miss_channel_rho_min[channel] = rho;
    tally.accepted_miss_channel_rho_max[channel] =
        std::max(tally.accepted_miss_channel_rho_max[channel], rho);
    tally.accepted_miss_channel_tail_ratio_sum[channel] += tail_ratio;
    tally.accepted_miss_channel_tail_ratio_max[channel] =
        std::max(tally.accepted_miss_channel_tail_ratio_max[channel],
                 tail_ratio);
    tally.accepted_miss_channel_estimated_error_sum[channel] +=
        estimated_error;
    tally.accepted_miss_channel_estimated_error_max[channel] =
        std::max(tally.accepted_miss_channel_estimated_error_max[channel],
                 estimated_error);
  }
  if (mode == "sampled-detail" && tally.accepted_miss_records.size() < 4096) {
    tally.accepted_miss_records.push_back(
        {target_index, source_node_index, channel, rho, tail_ratio,
         estimated_error, rel_error, abs_error, is_phantom});
  }
}

void merge_far_expansion_audit(ExpansionAuditTally &dst,
                               const ExpansionAuditTally &src) noexcept {
  bool has_split_caught = src.estimator_high_error_caught_by_split > 0 ||
                          src.estimator_phantom_high_error_count > 0 ||
                          src.convergence_overreject_count > 0 ||
                          src.convergence_overreject_caught_count > 0;
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel)
    has_split_caught = has_split_caught ||
                       src.channel_high_error_caught_by_split[channel] > 0 ||
                       src.channel_phantom_high_error_count[channel] > 0 ||
                       src.convergence_channel_overreject_count[channel] > 0 ||
                       src.convergence_channel_overreject_caught_count[channel] > 0;
  if (src.pairs == 0 && !has_split_caught)
    return;
  if (src.pairs == 0) {
    for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
         ++channel) {
      dst.channel_high_error_caught_by_split[channel] +=
          src.channel_high_error_caught_by_split[channel];
      dst.channel_phantom_high_error_count[channel] +=
          src.channel_phantom_high_error_count[channel];
      dst.channel_phantom_high_error_abs_max[channel] =
          std::max(dst.channel_phantom_high_error_abs_max[channel],
                   src.channel_phantom_high_error_abs_max[channel]);
      dst.convergence_channel_overreject_count[channel] +=
          src.convergence_channel_overreject_count[channel];
      dst.convergence_channel_overreject_caught_count[channel] +=
          src.convergence_channel_overreject_caught_count[channel];
      dst.convergence_channel_overreject_panels[channel] +=
          src.convergence_channel_overreject_panels[channel];
      dst.convergence_channel_overreject_abs_error_sum[channel] +=
          src.convergence_channel_overreject_abs_error_sum[channel];
      dst.convergence_channel_overreject_abs_error_max[channel] =
          std::max(dst.convergence_channel_overreject_abs_error_max[channel],
                   src.convergence_channel_overreject_abs_error_max[channel]);
      dst.convergence_channel_overreject_required_scale_samples[channel].insert(
          dst.convergence_channel_overreject_required_scale_samples[channel].end(),
          src.convergence_channel_overreject_required_scale_samples[channel].begin(),
          src.convergence_channel_overreject_required_scale_samples[channel].end());
    }
    dst.estimator_high_error_caught_by_split +=
        src.estimator_high_error_caught_by_split;
    dst.estimator_phantom_high_error_count +=
        src.estimator_phantom_high_error_count;
    dst.estimator_phantom_high_error_abs_max =
        std::max(dst.estimator_phantom_high_error_abs_max,
                 src.estimator_phantom_high_error_abs_max);
    dst.convergence_overreject_count += src.convergence_overreject_count;
    dst.convergence_overreject_panels += src.convergence_overreject_panels;
    dst.convergence_overreject_caught_count +=
        src.convergence_overreject_caught_count;
    dst.convergence_overreject_caught_panels +=
        src.convergence_overreject_caught_panels;
    dst.convergence_overreject_abs_error_sum +=
        src.convergence_overreject_abs_error_sum;
    dst.convergence_overreject_abs_error_max =
        std::max(dst.convergence_overreject_abs_error_max,
                 src.convergence_overreject_abs_error_max);
    dst.convergence_overreject_estimated_abs_error_sum +=
        src.convergence_overreject_estimated_abs_error_sum;
    dst.convergence_overreject_estimated_abs_error_max =
        std::max(dst.convergence_overreject_estimated_abs_error_max,
                 src.convergence_overreject_estimated_abs_error_max);
    dst.convergence_overreject_abs_error_samples.insert(
        dst.convergence_overreject_abs_error_samples.end(),
        src.convergence_overreject_abs_error_samples.begin(),
        src.convergence_overreject_abs_error_samples.end());
    dst.convergence_overreject_required_scale_samples.insert(
        dst.convergence_overreject_required_scale_samples.end(),
        src.convergence_overreject_required_scale_samples.begin(),
        src.convergence_overreject_required_scale_samples.end());
    const std::size_t calibration_remaining =
        dst.overreject_calibration_records.size() < 20480
            ? 20480 - dst.overreject_calibration_records.size()
            : 0;
    const std::size_t calibration_take = std::min<std::size_t>(
        calibration_remaining, src.overreject_calibration_records.size());
    dst.overreject_calibration_records.insert(
        dst.overreject_calibration_records.end(),
        src.overreject_calibration_records.begin(),
        src.overreject_calibration_records.begin() + calibration_take);
    const std::size_t shape_remaining =
        dst.convergence_shape_records.size() < 20480
            ? 20480 - dst.convergence_shape_records.size()
            : 0;
    const std::size_t shape_take = std::min<std::size_t>(
        shape_remaining, src.convergence_shape_records.size());
    dst.convergence_shape_records.insert(
        dst.convergence_shape_records.end(),
        src.convergence_shape_records.begin(),
        src.convergence_shape_records.begin() + shape_take);
    const std::size_t kernel_remaining =
        dst.kernel_response_records.size() < 20480
            ? 20480 - dst.kernel_response_records.size()
            : 0;
    const std::size_t kernel_take = std::min<std::size_t>(
        kernel_remaining, src.kernel_response_records.size());
    dst.kernel_response_records.insert(
        dst.kernel_response_records.end(), src.kernel_response_records.begin(),
        src.kernel_response_records.begin() + kernel_take);
    return;
  }
  dst.pairs += src.pairs;
  dst.panels += src.panels;
  dst.direct_ms += src.direct_ms;
  dst.rel_error_sum += src.rel_error_sum;
  dst.rel_error_max = std::max(dst.rel_error_max, src.rel_error_max);
  dst.rel_errors.insert(dst.rel_errors.end(), src.rel_errors.begin(),
                        src.rel_errors.end());
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel) {
    dst.channel_rel_error_sum[channel] +=
        src.channel_rel_error_sum[channel];
    dst.channel_rel_error_max[channel] =
        std::max(dst.channel_rel_error_max[channel],
                 src.channel_rel_error_max[channel]);
    dst.channel_rel_errors[channel].insert(
        dst.channel_rel_errors[channel].end(),
        src.channel_rel_errors[channel].begin(),
        src.channel_rel_errors[channel].end());
    dst.channel_high_error_caught[channel] +=
        src.channel_high_error_caught[channel];
    dst.channel_high_error_missed[channel] +=
        src.channel_high_error_missed[channel];
    dst.channel_high_error_caught_by_split[channel] +=
        src.channel_high_error_caught_by_split[channel];
    dst.channel_phantom_high_error_count[channel] +=
        src.channel_phantom_high_error_count[channel];
    dst.channel_phantom_high_error_abs_max[channel] =
        std::max(dst.channel_phantom_high_error_abs_max[channel],
                 src.channel_phantom_high_error_abs_max[channel]);
    dst.convergence_channel_overreject_count[channel] +=
        src.convergence_channel_overreject_count[channel];
    dst.convergence_channel_overreject_caught_count[channel] +=
        src.convergence_channel_overreject_caught_count[channel];
    dst.convergence_channel_overreject_panels[channel] +=
        src.convergence_channel_overreject_panels[channel];
    dst.convergence_channel_overreject_abs_error_sum[channel] +=
        src.convergence_channel_overreject_abs_error_sum[channel];
    dst.convergence_channel_overreject_abs_error_max[channel] =
        std::max(dst.convergence_channel_overreject_abs_error_max[channel],
                 src.convergence_channel_overreject_abs_error_max[channel]);
    dst.convergence_channel_overreject_required_scale_samples[channel].insert(
        dst.convergence_channel_overreject_required_scale_samples[channel].end(),
        src.convergence_channel_overreject_required_scale_samples[channel].begin(),
        src.convergence_channel_overreject_required_scale_samples[channel].end());
    if (src.accepted_miss_channel_count[channel] > 0) {
      if (dst.accepted_miss_channel_count[channel] == 0 ||
          src.accepted_miss_channel_rho_min[channel] <
              dst.accepted_miss_channel_rho_min[channel])
        dst.accepted_miss_channel_rho_min[channel] =
            src.accepted_miss_channel_rho_min[channel];
      dst.accepted_miss_channel_count[channel] +=
          src.accepted_miss_channel_count[channel];
      dst.accepted_miss_channel_panels[channel] +=
          src.accepted_miss_channel_panels[channel];
      dst.accepted_miss_channel_rho_sum[channel] +=
          src.accepted_miss_channel_rho_sum[channel];
      dst.accepted_miss_channel_rho_max[channel] =
          std::max(dst.accepted_miss_channel_rho_max[channel],
                   src.accepted_miss_channel_rho_max[channel]);
      dst.accepted_miss_channel_tail_ratio_sum[channel] +=
          src.accepted_miss_channel_tail_ratio_sum[channel];
      dst.accepted_miss_channel_tail_ratio_max[channel] =
          std::max(dst.accepted_miss_channel_tail_ratio_max[channel],
                   src.accepted_miss_channel_tail_ratio_max[channel]);
      dst.accepted_miss_channel_estimated_error_sum[channel] +=
          src.accepted_miss_channel_estimated_error_sum[channel];
      dst.accepted_miss_channel_estimated_error_max[channel] =
          std::max(dst.accepted_miss_channel_estimated_error_max[channel],
                   src.accepted_miss_channel_estimated_error_max[channel]);
    }
  }
  const std::size_t remaining =
      dst.accepted_miss_records.size() < 4096
          ? 4096 - dst.accepted_miss_records.size()
          : 0;
  const std::size_t take =
      std::min<std::size_t>(remaining, src.accepted_miss_records.size());
  dst.accepted_miss_records.insert(dst.accepted_miss_records.end(),
                                   src.accepted_miss_records.begin(),
                                   src.accepted_miss_records.begin() + take);
  if (src.worst_channel < kJcpExpansionChannelCount &&
      (dst.worst_channel >= kJcpExpansionChannelCount ||
       src.channel_rel_error_max[src.worst_channel] >
           dst.channel_rel_error_max[dst.worst_channel])) {
    dst.worst_channel = src.worst_channel;
  }
  dst.estimator_high_error_caught += src.estimator_high_error_caught;
  dst.estimator_high_error_missed += src.estimator_high_error_missed;
  dst.estimator_high_error_caught_by_split +=
      src.estimator_high_error_caught_by_split;
  dst.estimator_phantom_high_error_count +=
      src.estimator_phantom_high_error_count;
  dst.estimator_phantom_high_error_abs_max =
      std::max(dst.estimator_phantom_high_error_abs_max,
               src.estimator_phantom_high_error_abs_max);
  dst.convergence_overreject_count += src.convergence_overreject_count;
  dst.convergence_overreject_panels += src.convergence_overreject_panels;
  dst.convergence_overreject_caught_count +=
      src.convergence_overreject_caught_count;
  dst.convergence_overreject_caught_panels +=
      src.convergence_overreject_caught_panels;
  dst.convergence_overreject_abs_error_sum +=
      src.convergence_overreject_abs_error_sum;
  dst.convergence_overreject_abs_error_max =
      std::max(dst.convergence_overreject_abs_error_max,
               src.convergence_overreject_abs_error_max);
  dst.convergence_overreject_estimated_abs_error_sum +=
      src.convergence_overreject_estimated_abs_error_sum;
  dst.convergence_overreject_estimated_abs_error_max =
      std::max(dst.convergence_overreject_estimated_abs_error_max,
               src.convergence_overreject_estimated_abs_error_max);
  dst.convergence_overreject_abs_error_samples.insert(
      dst.convergence_overreject_abs_error_samples.end(),
      src.convergence_overreject_abs_error_samples.begin(),
      src.convergence_overreject_abs_error_samples.end());
  dst.convergence_overreject_required_scale_samples.insert(
      dst.convergence_overreject_required_scale_samples.end(),
      src.convergence_overreject_required_scale_samples.begin(),
      src.convergence_overreject_required_scale_samples.end());
  const std::size_t calibration_remaining =
      dst.overreject_calibration_records.size() < 20480
          ? 20480 - dst.overreject_calibration_records.size()
          : 0;
  const std::size_t calibration_take = std::min<std::size_t>(
      calibration_remaining, src.overreject_calibration_records.size());
  dst.overreject_calibration_records.insert(
      dst.overreject_calibration_records.end(),
      src.overreject_calibration_records.begin(),
      src.overreject_calibration_records.begin() + calibration_take);
  const std::size_t shape_remaining =
      dst.convergence_shape_records.size() < 20480
          ? 20480 - dst.convergence_shape_records.size()
          : 0;
  const std::size_t shape_take = std::min<std::size_t>(
      shape_remaining, src.convergence_shape_records.size());
  dst.convergence_shape_records.insert(
      dst.convergence_shape_records.end(),
      src.convergence_shape_records.begin(),
      src.convergence_shape_records.begin() + shape_take);
  const std::size_t kernel_remaining =
      dst.kernel_response_records.size() < 20480
          ? 20480 - dst.kernel_response_records.size()
          : 0;
  const std::size_t kernel_take = std::min<std::size_t>(
      kernel_remaining, src.kernel_response_records.size());
  dst.kernel_response_records.insert(
      dst.kernel_response_records.end(), src.kernel_response_records.begin(),
      src.kernel_response_records.begin() + kernel_take);
}

double percentile_copy(std::vector<double> values, double q) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const double clamped = std::clamp(q, 0.0, 1.0);
  const std::size_t index = std::min<std::size_t>(
      values.size() - 1,
      static_cast<std::size_t>(
          std::ceil(clamped * static_cast<double>(values.size())) - 1.0));
  return values[index];
}

void copy_overreject_calibration_to_stats(
    TreecodeStats &stats, const ExpansionAuditTally &tally) noexcept {
  if (tally.convergence_overreject_abs_error_samples.empty())
    return;
  const double old_count =
      static_cast<double>(stats.convergence_overreject_abs_error_samples.size());
  const double new_samples = static_cast<double>(
      tally.convergence_overreject_abs_error_samples.size());
  const double total_count = old_count + new_samples;
  stats.convergence_overreject_estimated_abs_error_mean =
      total_count > 0.0
          ? (stats.convergence_overreject_estimated_abs_error_mean * old_count +
             tally.convergence_overreject_estimated_abs_error_sum) /
                total_count
          : 0.0;
  stats.convergence_overreject_estimated_abs_error_max =
      std::max(stats.convergence_overreject_estimated_abs_error_max,
               tally.convergence_overreject_estimated_abs_error_max);
  stats.convergence_overreject_abs_error_samples.insert(
      stats.convergence_overreject_abs_error_samples.end(),
      tally.convergence_overreject_abs_error_samples.begin(),
      tally.convergence_overreject_abs_error_samples.end());
  stats.convergence_overreject_required_scale_samples.insert(
      stats.convergence_overreject_required_scale_samples.end(),
      tally.convergence_overreject_required_scale_samples.begin(),
      tally.convergence_overreject_required_scale_samples.end());
  stats.convergence_overreject_abs_error_p95 =
      percentile_copy(stats.convergence_overreject_abs_error_samples, 0.95);
  stats.convergence_overreject_abs_error_p99 =
      percentile_copy(stats.convergence_overreject_abs_error_samples, 0.99);
  stats.convergence_overreject_required_scale_max =
      std::max(stats.convergence_overreject_required_scale_max,
               percentile_copy(tally.convergence_overreject_required_scale_samples,
                               1.0));
  stats.convergence_overreject_required_scale_p95 =
      percentile_copy(stats.convergence_overreject_required_scale_samples, 0.95);
  stats.convergence_overreject_required_scale_p99 =
      percentile_copy(stats.convergence_overreject_required_scale_samples, 0.99);
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel) {
    const std::size_t sample_count =
        tally.convergence_channel_overreject_required_scale_samples[channel].size();
    if (sample_count == 0)
      continue;
    const double old_channel_count = static_cast<double>(
        stats.convergence_channel_overreject_count[channel]);
    const double new_channel_count =
        old_channel_count + static_cast<double>(sample_count);
    stats.convergence_channel_overreject_abs_error_mean[channel] =
        new_channel_count > 0.0
            ? (stats.convergence_channel_overreject_abs_error_mean[channel] *
                   old_channel_count +
               tally.convergence_channel_overreject_abs_error_sum[channel]) /
                  new_channel_count
            : 0.0;
    stats.convergence_channel_overreject_abs_error_max[channel] =
        std::max(stats.convergence_channel_overreject_abs_error_max[channel],
                 tally.convergence_channel_overreject_abs_error_max[channel]);
    stats.convergence_channel_overreject_panels[channel] +=
        tally.convergence_channel_overreject_panels[channel];
    stats.convergence_channel_overreject_required_scale_samples[channel].insert(
        stats.convergence_channel_overreject_required_scale_samples[channel].end(),
        tally.convergence_channel_overreject_required_scale_samples[channel].begin(),
        tally.convergence_channel_overreject_required_scale_samples[channel].end());
    stats.convergence_channel_overreject_required_scale_max[channel] =
        std::max(stats.convergence_channel_overreject_required_scale_max[channel],
                 percentile_copy(
                     tally.convergence_channel_overreject_required_scale_samples[channel],
                     1.0));
    stats.convergence_channel_overreject_required_scale_p99[channel] =
        percentile_copy(
            stats.convergence_channel_overreject_required_scale_samples[channel],
            0.99);
  }
}

void copy_far_expansion_audit_to_stats(
    TreecodeStats &stats, const ExpansionAuditTally &tally) noexcept {
  bool has_split_caught = tally.estimator_high_error_caught_by_split > 0 ||
                          tally.estimator_phantom_high_error_count > 0 ||
                          tally.convergence_overreject_count > 0 ||
                          tally.convergence_overreject_caught_count > 0;
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel)
    has_split_caught = has_split_caught ||
                       tally.channel_high_error_caught_by_split[channel] > 0 ||
                       tally.channel_phantom_high_error_count[channel] > 0 ||
                       tally.convergence_channel_overreject_count[channel] > 0 ||
                       tally.convergence_channel_overreject_caught_count[channel] > 0;
  if (tally.pairs == 0 && !has_split_caught)
    return;
  copy_overreject_calibration_to_stats(stats, tally);
  if (tally.pairs == 0) {
    for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
         ++channel) {
      stats.convergence_channel_high_error_caught_by_split[channel] +=
          tally.channel_high_error_caught_by_split[channel];
      stats.phantom_high_error_channel_count[channel] +=
          tally.channel_phantom_high_error_count[channel];
      stats.phantom_high_error_channel_abs_max[channel] =
          std::max(stats.phantom_high_error_channel_abs_max[channel],
                   tally.channel_phantom_high_error_abs_max[channel]);
      stats.convergence_channel_overreject_count[channel] +=
          tally.convergence_channel_overreject_count[channel];
      stats.convergence_channel_overreject_caught_count[channel] +=
          tally.convergence_channel_overreject_caught_count[channel];
    }
    stats.convergence_estimator_high_error_caught_by_split +=
        tally.estimator_high_error_caught_by_split;
    stats.convergence_estimator_phantom_high_error_count +=
        tally.estimator_phantom_high_error_count;
    stats.convergence_estimator_phantom_high_error_abs_max =
        std::max(stats.convergence_estimator_phantom_high_error_abs_max,
                 tally.estimator_phantom_high_error_abs_max);
    stats.convergence_overreject_count += tally.convergence_overreject_count;
    stats.convergence_overreject_panels += tally.convergence_overreject_panels;
    stats.convergence_overreject_caught_count +=
        tally.convergence_overreject_caught_count;
    stats.convergence_overreject_caught_panels +=
        tally.convergence_overreject_caught_panels;
    stats.convergence_overreject_abs_error_max =
        std::max(stats.convergence_overreject_abs_error_max,
                 tally.convergence_overreject_abs_error_max);
    const double total_overreject =
        static_cast<double>(stats.convergence_overreject_count);
    stats.convergence_overreject_abs_error_mean =
        total_overreject > 0.0
            ? ((stats.convergence_overreject_abs_error_mean *
                (total_overreject -
                 static_cast<double>(tally.convergence_overreject_count))) +
               tally.convergence_overreject_abs_error_sum) /
                  total_overreject
            : 0.0;
    return;
  }
  const double old_count =
      static_cast<double>(stats.far_expansion_audit_pairs);
  const double new_count = old_count + static_cast<double>(tally.pairs);
  const double old_sum =
      stats.far_expansion_rel_error_mean * old_count;
  stats.far_expansion_audit_pairs += tally.pairs;
  stats.far_expansion_audit_panels += tally.panels;
  stats.far_expansion_direct_ms += tally.direct_ms;
  stats.far_expansion_rel_error_max =
      std::max(stats.far_expansion_rel_error_max, tally.rel_error_max);
  stats.far_expansion_rel_error_mean =
      new_count > 0.0 ? (old_sum + tally.rel_error_sum) / new_count : 0.0;
  stats.far_expansion_rel_error_samples.insert(
      stats.far_expansion_rel_error_samples.end(), tally.rel_errors.begin(),
      tally.rel_errors.end());
  stats.far_expansion_rel_error_p95 =
      percentile_copy(stats.far_expansion_rel_error_samples, 0.95);
  stats.far_expansion_rel_error_p99 =
      percentile_copy(stats.far_expansion_rel_error_samples, 0.99);
  stats.expansion_audit_channel =
      stats.expansion_audit_channel.empty() ? std::string("off")
                                            : stats.expansion_audit_channel;
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel) {
    const double old_channel_sum =
        stats.far_expansion_channel_rel_error_mean[channel] * old_count;
    stats.far_expansion_channel_rel_error_max[channel] =
        std::max(stats.far_expansion_channel_rel_error_max[channel],
                 tally.channel_rel_error_max[channel]);
    stats.far_expansion_channel_rel_error_mean[channel] =
        new_count > 0.0
            ? (old_channel_sum + tally.channel_rel_error_sum[channel]) /
                  new_count
            : 0.0;
    stats.far_expansion_channel_rel_error_samples[channel].insert(
        stats.far_expansion_channel_rel_error_samples[channel].end(),
        tally.channel_rel_errors[channel].begin(),
        tally.channel_rel_errors[channel].end());
    stats.far_expansion_channel_rel_error_p99[channel] =
        percentile_copy(stats.far_expansion_channel_rel_error_samples[channel],
                        0.99);
    stats.convergence_channel_high_error_caught[channel] +=
        tally.channel_high_error_caught[channel];
    stats.convergence_channel_high_error_missed[channel] +=
        tally.channel_high_error_missed[channel];
    stats.convergence_channel_high_error_caught_by_split[channel] +=
        tally.channel_high_error_caught_by_split[channel];
    stats.phantom_high_error_channel_count[channel] +=
        tally.channel_phantom_high_error_count[channel];
    stats.phantom_high_error_channel_abs_max[channel] =
        std::max(stats.phantom_high_error_channel_abs_max[channel],
                 tally.channel_phantom_high_error_abs_max[channel]);
    stats.convergence_channel_overreject_count[channel] +=
        tally.convergence_channel_overreject_count[channel];
    stats.convergence_channel_overreject_caught_count[channel] +=
        tally.convergence_channel_overreject_caught_count[channel];
    const std::size_t accepted_miss_count =
        tally.accepted_miss_channel_count[channel];
    if (accepted_miss_count > 0) {
      const double old_miss_count =
          static_cast<double>(stats.accepted_miss_channel_count[channel]);
      const double new_miss_count =
          old_miss_count + static_cast<double>(accepted_miss_count);
      const double old_rho_sum =
          stats.accepted_miss_channel_rho_mean[channel] * old_miss_count;
      const double old_tail_sum =
          stats.accepted_miss_channel_tail_ratio_mean[channel] *
          old_miss_count;
      const double old_est_sum =
          stats.accepted_miss_channel_estimated_error_mean[channel] *
          old_miss_count;
      if (stats.accepted_miss_channel_count[channel] == 0 ||
          tally.accepted_miss_channel_rho_min[channel] <
              stats.accepted_miss_channel_rho_min[channel])
        stats.accepted_miss_channel_rho_min[channel] =
            tally.accepted_miss_channel_rho_min[channel];
      stats.accepted_miss_channel_count[channel] += accepted_miss_count;
      stats.accepted_miss_channel_panels[channel] +=
          tally.accepted_miss_channel_panels[channel];
      stats.accepted_miss_channel_rho_max[channel] =
          std::max(stats.accepted_miss_channel_rho_max[channel],
                   tally.accepted_miss_channel_rho_max[channel]);
      stats.accepted_miss_channel_tail_ratio_max[channel] =
          std::max(stats.accepted_miss_channel_tail_ratio_max[channel],
                   tally.accepted_miss_channel_tail_ratio_max[channel]);
      stats.accepted_miss_channel_estimated_error_max[channel] =
          std::max(stats.accepted_miss_channel_estimated_error_max[channel],
                   tally.accepted_miss_channel_estimated_error_max[channel]);
      stats.accepted_miss_channel_rho_mean[channel] =
          new_miss_count > 0.0
              ? (old_rho_sum + tally.accepted_miss_channel_rho_sum[channel]) /
                    new_miss_count
              : 0.0;
      stats.accepted_miss_channel_tail_ratio_mean[channel] =
          new_miss_count > 0.0
              ? (old_tail_sum +
                 tally.accepted_miss_channel_tail_ratio_sum[channel]) /
                    new_miss_count
              : 0.0;
      stats.accepted_miss_channel_estimated_error_mean[channel] =
          new_miss_count > 0.0
              ? (old_est_sum +
                 tally.accepted_miss_channel_estimated_error_sum[channel]) /
                    new_miss_count
              : 0.0;
    }
  }
  std::size_t worst_miss_channel = kJcpExpansionChannelCount;
  std::size_t worst_miss_count = 0;
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel) {
    if (stats.accepted_miss_channel_count[channel] > worst_miss_count) {
      worst_miss_count = stats.accepted_miss_channel_count[channel];
      worst_miss_channel = channel;
    }
  }
  stats.accepted_miss_worst_channel =
      worst_miss_channel < kJcpExpansionChannelCount
          ? jcp_expansion_channel_label(worst_miss_channel)
          : "none";
  std::size_t worst_channel = kJcpExpansionChannelCount;
  double worst_rel = 0.0;
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel) {
    if (stats.far_expansion_channel_rel_error_max[channel] > worst_rel) {
      worst_rel = stats.far_expansion_channel_rel_error_max[channel];
      worst_channel = channel;
    }
  }
  stats.far_expansion_worst_channel =
      worst_channel < kJcpExpansionChannelCount
          ? jcp_expansion_channel_label(worst_channel)
          : "none";
  stats.convergence_estimator_high_error_caught +=
      tally.estimator_high_error_caught;
  stats.convergence_estimator_high_error_missed +=
      tally.estimator_high_error_missed;
  stats.convergence_estimator_high_error_caught_by_split +=
      tally.estimator_high_error_caught_by_split;
  stats.convergence_estimator_phantom_high_error_count +=
      tally.estimator_phantom_high_error_count;
  stats.convergence_estimator_phantom_high_error_abs_max =
      std::max(stats.convergence_estimator_phantom_high_error_abs_max,
               tally.estimator_phantom_high_error_abs_max);
  const double old_overreject_count =
      static_cast<double>(stats.convergence_overreject_count);
  stats.convergence_overreject_count += tally.convergence_overreject_count;
  stats.convergence_overreject_panels += tally.convergence_overreject_panels;
  stats.convergence_overreject_caught_count +=
      tally.convergence_overreject_caught_count;
  stats.convergence_overreject_caught_panels +=
      tally.convergence_overreject_caught_panels;
  stats.convergence_overreject_abs_error_max =
      std::max(stats.convergence_overreject_abs_error_max,
               tally.convergence_overreject_abs_error_max);
  const double new_overreject_count =
      static_cast<double>(stats.convergence_overreject_count);
  stats.convergence_overreject_abs_error_mean =
      new_overreject_count > 0.0
          ? (stats.convergence_overreject_abs_error_mean *
                 old_overreject_count +
             tally.convergence_overreject_abs_error_sum) /
                new_overreject_count
          : 0.0;
  stats.far_expansion_mean_gate_pass =
      stats.far_expansion_audit_pairs > 0 &&
      stats.far_expansion_rel_error_mean < 5.0e-4 &&
      stats.convergence_estimator_high_error_missed == 0;
}

void write_accepted_miss_detail_csv(const ExpansionAuditTally &tally) {
  std::ofstream out("pbe_tabi_accepted_miss_diagnostics.csv",
                    std::ios::out | std::ios::trunc);
  if (!out.is_open())
    return;
  out << "target_index,source_node,channel,rho,tail_ratio,estimated_error,"
         "rel_error,abs_error,is_phantom\n";
  for (const auto &record : tally.accepted_miss_records) {
    out << record.target_index << ',' << record.source_node_index << ','
        << jcp_expansion_channel_label(record.channel) << ',' << record.rho
        << ',' << record.tail_ratio << ',' << record.estimated_error << ','
        << record.rel_error << ',' << record.abs_error << ','
        << (record.is_phantom ? 1 : 0) << '\n';
  }
}

void write_overreject_calibration_detail_csv(const ExpansionAuditTally &tally) {
  std::ofstream out("pbe_tabi_convergence_overreject_calibration.csv",
                    std::ios::out | std::ios::trunc);
  if (!out.is_open())
    return;
  out << "target_index,source_node,channel,node_panels,rho,aspect_ratio,"
         "estimated_rel_error,estimated_abs_error,oracle_rel_error,"
         "oracle_abs_error,required_scale,classification\n";
  for (const auto &record : tally.overreject_calibration_records) {
    out << record.target_index << ',' << record.source_node_index << ','
        << (record.channel < kJcpExpansionChannelCount
                ? jcp_expansion_channel_label(record.channel)
                : "global")
        << ',' << record.node_panels << ',' << record.rho << ','
        << record.aspect_ratio << ',' << record.estimated_rel_error << ','
        << record.estimated_abs_error << ',' << record.oracle_rel_error << ','
        << record.oracle_abs_error << ',' << record.required_scale << ','
        << (record.caught ? "caught" : "overreject") << '\n';
  }
}

void write_convergence_shape_calibration_csv(
    const ExpansionAuditTally &tally) {
  std::ofstream out("pbe_tabi_convergence_shape_calibration.csv",
                    std::ios::out | std::ios::trunc);
  if (!out.is_open())
    return;
  out << "target_index,source_node,channel,node_panels,rho,rho_power,"
         "aspect_ratio,node_radius,target_distance,source_norm,tail_norm_p,"
         "tail_norm_pm1,tail_ratio,effective_amp,estimated_rel_error,"
         "estimated_abs_error,oracle_rel_error,oracle_abs_error,"
         "oracle_abs_over_source_norm,oracle_abs_over_tail_norm,"
         "oracle_abs_over_tail_rho,classification\n";
  for (const auto &record : tally.convergence_shape_records) {
    out << record.target_index << ',' << record.source_node_index << ',';
    if (record.channel >= kJcpExpansionChannelCount) {
      out << "global";
    } else {
      out << jcp_expansion_channel_label(record.channel);
    }
    out << ',' << record.node_panels << ',' << record.rho << ','
        << record.rho_power << ',' << record.aspect_ratio << ','
        << record.node_radius << ',' << record.target_distance << ','
        << record.source_norm << ',' << record.tail_norm_p << ','
        << record.tail_norm_pm1 << ',' << record.tail_ratio << ','
        << record.effective_amp << ',' << record.estimated_rel_error << ','
        << record.estimated_abs_error << ',' << record.oracle_rel_error << ','
        << record.oracle_abs_error << ','
        << record.oracle_abs_over_source_norm << ','
        << record.oracle_abs_over_tail_norm << ','
        << record.oracle_abs_over_tail_rho << ','
        << (record.caught ? "caught" : "overreject") << '\n';
  }
}

void write_kernel_response_calibration_csv(const ExpansionAuditTally &tally) {
  std::ofstream out("pbe_tabi_kernel_response_calibration.csv",
                    std::ios::out | std::ios::trunc);
  if (!out.is_open())
    return;
  out << "target_index,source_node,channel,node_panels,rho,aspect_ratio,"
         "source_family,source_norm,tail_norm_p,tail_norm_pm1,"
         "estimated_abs_error,kernel_tail_p_abs,kernel_tail_pm1_abs,"
         "kernel_tail_envelope_abs,oracle_rel_error,oracle_abs_error,"
         "oracle_abs_over_kernel_tail_envelope,classification\n";
  for (const auto &record : tally.kernel_response_records) {
    out << record.target_index << ',' << record.source_node_index << ','
        << (record.channel < kJcpExpansionChannelCount
                ? jcp_expansion_channel_label(record.channel)
                : "global")
        << ',' << record.node_panels << ',' << record.rho << ','
        << record.aspect_ratio << ',' << record.source_family << ','
        << record.source_norm << ',' << record.tail_norm_p << ','
        << record.tail_norm_pm1 << ',' << record.estimated_abs_error << ','
        << record.kernel_tail_p_abs << ',' << record.kernel_tail_pm1_abs << ','
        << record.kernel_tail_envelope_abs << ',' << record.oracle_rel_error
        << ',' << record.oracle_abs_error << ','
        << record.oracle_abs_over_kernel_tail_envelope << ','
        << (record.caught ? "caught" : "overreject") << '\n';
  }
}

void record_convergence_estimator(ConvergenceEstimatorTally &tally,
                                  const TreeNode &node,
                                  const ConvergenceEstimatorDecision &decision)
    noexcept {
  if (!decision.active)
    return;
  ++tally.samples;
  tally.tail_norm_p_sum += decision.tail_norm_p;
  tally.tail_norm_p_max =
      std::max(tally.tail_norm_p_max, decision.tail_norm_p);
  tally.tail_norm_pm1_sum += decision.tail_norm_pm1;
  tally.tail_norm_pm1_max =
      std::max(tally.tail_norm_pm1_max, decision.tail_norm_pm1);
  tally.tail_ratio_sum += decision.tail_ratio;
  tally.tail_ratio_max = std::max(tally.tail_ratio_max, decision.tail_ratio);
  tally.estimated_error_sum += decision.estimated_error;
  tally.estimated_error_max =
      std::max(tally.estimated_error_max, decision.estimated_error);
  tally.estimated_abs_error_sum += decision.estimated_abs_error;
  tally.estimated_abs_error_max =
      std::max(tally.estimated_abs_error_max, decision.estimated_abs_error);
  tally.dielectric_jcp_amp_sum += decision.dielectric_jcp_amp;
  tally.dielectric_jcp_amp_max =
      std::max(tally.dielectric_jcp_amp_max, decision.dielectric_jcp_amp);
  tally.effective_jcp_amp_sum += decision.effective_jcp_amp;
  tally.effective_jcp_amp_max =
      std::max(tally.effective_jcp_amp_max, decision.effective_jcp_amp);
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel) {
    tally.channel_estimated_error_sum[channel] +=
        decision.channel_estimated_error[channel];
    tally.channel_estimated_error_max[channel] =
        std::max(tally.channel_estimated_error_max[channel],
                 decision.channel_estimated_error[channel]);
    tally.channel_effective_jcp_amp_sum[channel] +=
        decision.channel_effective_jcp_amp[channel];
    tally.channel_effective_jcp_amp_max[channel] =
        std::max(tally.channel_effective_jcp_amp_max[channel],
                 decision.channel_effective_jcp_amp[channel]);
    if (decision.channel_unstable[channel]) {
      ++tally.channel_unstable_count[channel];
      tally.channel_unstable_panels[channel] += node.panel_count;
      if (decision.split_reject) {
        ++tally.channel_split_rejects[channel];
        tally.channel_split_reject_panels[channel] += node.panel_count;
      }
      if (decision.channel_low_abs_reaccepted[channel]) {
        ++tally.channel_low_abs_reaccepted_count[channel];
        tally.channel_low_abs_reaccepted_panels[channel] += node.panel_count;
      }
    }
  }
  if (decision.unstable) {
    ++tally.unstable_count;
    tally.unstable_panels += node.panel_count;
  }
  if (decision.split_reject) {
    ++tally.split_rejects;
    tally.split_reject_panels += node.panel_count;
  }
  if (decision.low_abs_reaccepted) {
    ++tally.low_abs_reaccepted_count;
    tally.low_abs_reaccepted_panels += node.panel_count;
  }
}

void merge_convergence_estimator(ConvergenceEstimatorTally &dst,
                                 const ConvergenceEstimatorTally &src)
    noexcept {
  if (src.samples == 0)
    return;
  dst.samples += src.samples;
  dst.unstable_count += src.unstable_count;
  dst.unstable_panels += src.unstable_panels;
  dst.split_rejects += src.split_rejects;
  dst.split_reject_panels += src.split_reject_panels;
  dst.tail_norm_p_sum += src.tail_norm_p_sum;
  dst.tail_norm_p_max = std::max(dst.tail_norm_p_max, src.tail_norm_p_max);
  dst.tail_norm_pm1_sum += src.tail_norm_pm1_sum;
  dst.tail_norm_pm1_max =
      std::max(dst.tail_norm_pm1_max, src.tail_norm_pm1_max);
  dst.dielectric_jcp_amp_sum += src.dielectric_jcp_amp_sum;
  dst.dielectric_jcp_amp_max =
      std::max(dst.dielectric_jcp_amp_max, src.dielectric_jcp_amp_max);
  dst.tail_ratio_sum += src.tail_ratio_sum;
  dst.tail_ratio_max = std::max(dst.tail_ratio_max, src.tail_ratio_max);
  dst.estimated_error_sum += src.estimated_error_sum;
  dst.estimated_error_max =
      std::max(dst.estimated_error_max, src.estimated_error_max);
  dst.estimated_abs_error_sum += src.estimated_abs_error_sum;
  dst.estimated_abs_error_max =
      std::max(dst.estimated_abs_error_max, src.estimated_abs_error_max);
  dst.low_abs_reaccepted_count += src.low_abs_reaccepted_count;
  dst.low_abs_reaccepted_panels += src.low_abs_reaccepted_panels;
  dst.effective_jcp_amp_sum += src.effective_jcp_amp_sum;
  dst.effective_jcp_amp_max =
      std::max(dst.effective_jcp_amp_max, src.effective_jcp_amp_max);
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel) {
    dst.channel_estimated_error_sum[channel] +=
        src.channel_estimated_error_sum[channel];
    dst.channel_estimated_error_max[channel] =
        std::max(dst.channel_estimated_error_max[channel],
                 src.channel_estimated_error_max[channel]);
    dst.channel_effective_jcp_amp_sum[channel] +=
        src.channel_effective_jcp_amp_sum[channel];
    dst.channel_effective_jcp_amp_max[channel] =
        std::max(dst.channel_effective_jcp_amp_max[channel],
                 src.channel_effective_jcp_amp_max[channel]);
    dst.channel_unstable_count[channel] += src.channel_unstable_count[channel];
    dst.channel_unstable_panels[channel] += src.channel_unstable_panels[channel];
    dst.channel_split_rejects[channel] += src.channel_split_rejects[channel];
    dst.channel_split_reject_panels[channel] +=
        src.channel_split_reject_panels[channel];
    dst.channel_low_abs_reaccepted_count[channel] +=
        src.channel_low_abs_reaccepted_count[channel];
    dst.channel_low_abs_reaccepted_panels[channel] +=
        src.channel_low_abs_reaccepted_panels[channel];
  }
}

void copy_convergence_estimator_to_stats(
    TreecodeStats &stats, const ConvergenceEstimatorTally &tally) noexcept {
  if (tally.samples == 0)
    return;
  const double old_count =
      static_cast<double>(stats.convergence_estimator_samples);
  const double new_count = old_count + static_cast<double>(tally.samples);
  const double old_tail_sum =
      stats.convergence_estimator_tail_ratio_mean * old_count;
  const double old_error_sum =
      stats.convergence_estimator_error_mean * old_count;
  const double old_tail_p_sum =
      stats.convergence_tail_norm_p_mean * old_count;
  const double old_tail_pm1_sum =
      stats.convergence_tail_norm_pm1_mean * old_count;
  const double old_dielectric_amp_sum =
      stats.convergence_dielectric_amp_mean * old_count;
  const double old_amp_sum =
      stats.convergence_effective_jcp_amp_mean * old_count;
  stats.convergence_estimator_samples += tally.samples;
  stats.convergence_estimator_unstable_far_count += tally.unstable_count;
  stats.convergence_estimator_unstable_far_panels += tally.unstable_panels;
  stats.convergence_estimator_split_rejects += tally.split_rejects;
  stats.convergence_estimator_split_reject_panels +=
      tally.split_reject_panels;
  stats.convergence_low_abs_reaccepted_far_count +=
      tally.low_abs_reaccepted_count;
  stats.convergence_low_abs_reaccepted_far_panels +=
      tally.low_abs_reaccepted_panels;
  stats.convergence_estimator_tail_ratio_max =
      std::max(stats.convergence_estimator_tail_ratio_max,
               tally.tail_ratio_max);
  stats.convergence_estimator_error_max =
      std::max(stats.convergence_estimator_error_max,
               tally.estimated_error_max);
  stats.convergence_estimator_abs_error_max =
      std::max(stats.convergence_estimator_abs_error_max,
               tally.estimated_abs_error_max);
  stats.convergence_tail_norm_p_max =
      std::max(stats.convergence_tail_norm_p_max, tally.tail_norm_p_max);
  stats.convergence_tail_norm_pm1_max =
      std::max(stats.convergence_tail_norm_pm1_max, tally.tail_norm_pm1_max);
  stats.convergence_dielectric_amp_max =
      std::max(stats.convergence_dielectric_amp_max,
               tally.dielectric_jcp_amp_max);
  stats.convergence_effective_jcp_amp_max =
      std::max(stats.convergence_effective_jcp_amp_max,
               tally.effective_jcp_amp_max);
  stats.convergence_estimator_tail_ratio_mean =
      new_count > 0.0 ? (old_tail_sum + tally.tail_ratio_sum) / new_count
                      : 0.0;
  stats.convergence_estimator_error_mean =
      new_count > 0.0 ? (old_error_sum + tally.estimated_error_sum) / new_count
                      : 0.0;
  const double old_abs_error_sum =
      stats.convergence_estimator_abs_error_mean * old_count;
  stats.convergence_estimator_abs_error_mean =
      new_count > 0.0
          ? (old_abs_error_sum + tally.estimated_abs_error_sum) / new_count
          : 0.0;
  stats.convergence_tail_norm_p_mean =
      new_count > 0.0 ? (old_tail_p_sum + tally.tail_norm_p_sum) / new_count
                      : 0.0;
  stats.convergence_tail_norm_pm1_mean =
      new_count > 0.0
          ? (old_tail_pm1_sum + tally.tail_norm_pm1_sum) / new_count
          : 0.0;
  stats.convergence_dielectric_amp_mean =
      new_count > 0.0
          ? (old_dielectric_amp_sum + tally.dielectric_jcp_amp_sum) /
                new_count
          : 0.0;
  stats.convergence_effective_jcp_amp_mean =
      new_count > 0.0 ? (old_amp_sum + tally.effective_jcp_amp_sum) / new_count
                      : 0.0;
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel) {
    const double old_channel_sum =
        stats.convergence_channel_estimated_error_mean[channel] * old_count;
    const double old_channel_amp_sum =
        stats.convergence_channel_effective_jcp_amp_mean[channel] * old_count;
    stats.convergence_channel_estimated_error_max[channel] =
        std::max(stats.convergence_channel_estimated_error_max[channel],
                 tally.channel_estimated_error_max[channel]);
    stats.convergence_channel_estimated_error_mean[channel] =
        new_count > 0.0
            ? (old_channel_sum + tally.channel_estimated_error_sum[channel]) /
                  new_count
            : 0.0;
    stats.convergence_channel_effective_jcp_amp_max[channel] =
        std::max(stats.convergence_channel_effective_jcp_amp_max[channel],
                 tally.channel_effective_jcp_amp_max[channel]);
    stats.convergence_channel_effective_jcp_amp_mean[channel] =
        new_count > 0.0
            ? (old_channel_amp_sum +
               tally.channel_effective_jcp_amp_sum[channel]) /
                  new_count
            : 0.0;
    stats.convergence_channel_unstable_far_count[channel] +=
        tally.channel_unstable_count[channel];
    stats.convergence_channel_unstable_far_panels[channel] +=
        tally.channel_unstable_panels[channel];
    stats.convergence_channel_split_rejects[channel] +=
        tally.channel_split_rejects[channel];
    stats.convergence_channel_split_reject_panels[channel] +=
        tally.channel_split_reject_panels[channel];
    stats.convergence_channel_low_abs_reaccepted_count[channel] +=
        tally.channel_low_abs_reaccepted_count[channel];
    stats.convergence_channel_low_abs_reaccepted_panels[channel] +=
        tally.channel_low_abs_reaccepted_panels[channel];
  }
}

void record_far_convergence_audit(ConvergenceAuditTally &tally,
                                  const TreeNode &node,
                                  double rho) noexcept {
  if (rho <= 0.0 || !std::isfinite(rho))
    return;
  const double aspect = node_aspect_ratio(node);
  ++tally.pairs;
  tally.rho_sum += rho;
  tally.rho_max = std::max(tally.rho_max, rho);
  tally.aspect_sum += aspect;
  tally.aspect_max = std::max(tally.aspect_max, aspect);
}

void merge_far_convergence_audit(ConvergenceAuditTally &dst,
                                 const ConvergenceAuditTally &src) noexcept {
  if (src.pairs == 0)
    return;
  dst.pairs += src.pairs;
  dst.rho_sum += src.rho_sum;
  dst.rho_max = std::max(dst.rho_max, src.rho_max);
  dst.aspect_sum += src.aspect_sum;
  dst.aspect_max = std::max(dst.aspect_max, src.aspect_max);
}

void copy_far_convergence_audit_to_stats(
    TreecodeStats &stats, const ConvergenceAuditTally &tally) noexcept {
  if (tally.pairs == 0)
    return;
  const std::size_t previous_pairs =
      stats.far_node_interactions >= tally.pairs
          ? stats.far_node_interactions - tally.pairs
          : std::size_t{0};
  const double old_count = static_cast<double>(previous_pairs);
  const double add_count = static_cast<double>(tally.pairs);
  const double new_count = old_count + add_count;
  stats.far_convergence_rho_max =
      std::max(stats.far_convergence_rho_max, tally.rho_max);
  stats.accepted_far_aspect_ratio_max =
      std::max(stats.accepted_far_aspect_ratio_max, tally.aspect_max);
  stats.far_convergence_rho_mean =
      new_count > 0.0
          ? (stats.far_convergence_rho_mean * old_count + tally.rho_sum) /
                new_count
          : 0.0;
  stats.accepted_far_aspect_ratio_mean =
      new_count > 0.0
          ? (stats.accepted_far_aspect_ratio_mean * old_count +
             tally.aspect_sum) /
                new_count
          : 0.0;
}

MacDecision evaluate_mac(const Panel &target, const TreeNode &node,
                         const TreeWorkspace &tree,
                         double distance) noexcept {
  MacDecision decision;
  const double center_rho =
      distance > 0.0 && std::isfinite(distance)
          ? node.radius / distance
          : std::numeric_limits<double>::infinity();
  decision.rho = center_rho;
  decision.aspect = node_aspect_ratio(node);
  if (tree.mac_distance_policy == "surface-margin") {
    const bool adaptive = tree.mac_surface_margin_policy == "adaptive";
    const bool risk_scaled =
        tree.mac_surface_margin_policy == "risk-scaled";
    const bool high_risk =
        (!adaptive && !risk_scaled) ||
        center_rho >= tree.mac_surface_margin_rho_trigger ||
        decision.aspect >= tree.mac_surface_margin_aspect_threshold;
    if (high_risk) {
      decision.surface_margin_applied = true;
      decision.surface_margin_hard_applied = true;
      const double margin = tree.mac_surface_margin_factor * node.radius;
      const double effective_distance =
          std::max(1.0e-12, std::isfinite(distance) ? distance - margin : 0.0);
      decision.rho = node.radius / effective_distance;
    } else if (risk_scaled &&
               ((tree.mac_surface_margin_soft_rho_trigger > 0.0 &&
                 center_rho >= tree.mac_surface_margin_soft_rho_trigger) ||
                (tree.mac_surface_margin_soft_aspect_threshold > 0.0 &&
                 decision.aspect >=
                     tree.mac_surface_margin_soft_aspect_threshold))) {
      decision.surface_margin_applied = true;
      decision.surface_margin_soft_applied = true;
      const double margin =
          tree.mac_surface_margin_soft_factor * node.radius;
      const double effective_distance =
          std::max(1.0e-12, std::isfinite(distance) ? distance - margin : 0.0);
      decision.rho = node.radius / effective_distance;
    } else {
      decision.surface_margin_skipped = true;
    }
  }
  decision.ratio_accept = decision.rho < tree.opening_theta;
  decision.surface_margin_limited =
      decision.surface_margin_applied &&
      center_rho < tree.opening_theta && !decision.ratio_accept;
  decision.surface_margin_adaptive_limited =
      decision.surface_margin_limited &&
      tree.mac_surface_margin_policy == "adaptive";
  decision.surface_margin_hard_limited =
      decision.surface_margin_limited && decision.surface_margin_hard_applied;
  decision.surface_margin_soft_limited =
      decision.surface_margin_limited && decision.surface_margin_soft_applied;
  decision.accepted = decision.ratio_accept;
  if (!decision.ratio_accept)
    return decision;
  if (tree.mac_safety != "off") {
    const double target_edge = tree_panel_max_edge_length(target);
    const double guard_edge =
        std::max({target_edge, node.max_edge_length, 1.0e-12});
    decision.gap = panel_node_aabb_gap(target, node);
    decision.gap_ratio = decision.gap / guard_edge;
    decision.unsafe = decision.gap <= tree.mac_gap_factor * guard_edge;
    if (tree.mac_safety == "physical-gap" && decision.unsafe) {
      decision.physical_gap_limited = true;
      decision.accepted = false;
    }
  }
  if (!decision.accepted || tree.convergence_safety == "off")
    return decision;
  decision.convergence_rho_limited =
      tree.convergence_rho_cap > 0.0 && decision.rho > tree.convergence_rho_cap;
  decision.convergence_aspect_limited =
      tree.mac_aspect_cap > 0.0 && decision.aspect > tree.mac_aspect_cap;
  if (tree.convergence_safety == "split" &&
      (decision.convergence_rho_limited ||
       decision.convergence_aspect_limited))
    decision.accepted = false;
  return decision;
}



double jcp_rel_error(const JcpAccum &taylor, const JcpAccum &direct,
                     bool coupled) noexcept {
  const double dphi = taylor.phi - direct.phi;
  const double dphin = coupled ? (taylor.phin - direct.phin) : 0.0;
  const double diff_norm = std::sqrt(dphi * dphi + dphin * dphin);
  const double ref_norm = std::sqrt(
      direct.phi * direct.phi + (coupled ? direct.phin * direct.phin : 0.0));
  return diff_norm / std::max(1.0e-300, ref_norm);
}

double jcp_abs_error(const JcpAccum &taylor, const JcpAccum &direct,
                     bool coupled) noexcept {
  const double dphi = taylor.phi - direct.phi;
  const double dphin = coupled ? (taylor.phin - direct.phin) : 0.0;
  return std::sqrt(dphi * dphi + dphin * dphin);
}

bool is_real_high_error(double rel_error, double abs_error, double rel_cap,
                        double abs_floor) noexcept {
  return rel_cap > 0.0 && rel_error > rel_cap &&
         abs_error > std::max(0.0, abs_floor);
}

bool is_phantom_high_error(double rel_error, double abs_error, double rel_cap,
                           double abs_floor) noexcept {
  return rel_cap > 0.0 && rel_error > rel_cap &&
         abs_error <= std::max(0.0, abs_floor);
}

void record_far_expansion_audit(ExpansionAuditTally &tally,
                                const JcpAccum &taylor,
                                const JcpAccum &direct, bool coupled,
                                std::size_t panel_count,
                                double direct_ms,
                                bool estimator_active = false,
                                bool estimator_unstable = false,
                                bool estimator_split_reject = false,
                                double estimator_error_cap = 0.0,
                                double accepted_miss_absolute_floor = 1.0e-7,
                                const std::string
                                    &channel_missed_semantics =
                                        "legacy") noexcept {
  const double rel = jcp_rel_error(taylor, direct, coupled);
  const double abs_error = jcp_abs_error(taylor, direct, coupled);
  ++tally.pairs;
  tally.panels += panel_count;
  tally.direct_ms += direct_ms;
  tally.rel_error_sum += rel;
  tally.rel_error_max = std::max(tally.rel_error_max, rel);
  tally.rel_errors.push_back(rel);
  if (estimator_active &&
      is_phantom_high_error(rel, abs_error, estimator_error_cap,
                            accepted_miss_absolute_floor)) {
    ++tally.estimator_phantom_high_error_count;
    tally.estimator_phantom_high_error_abs_max =
        std::max(tally.estimator_phantom_high_error_abs_max, abs_error);
  } else if (estimator_active &&
             is_real_high_error(rel, abs_error, estimator_error_cap,
                                accepted_miss_absolute_floor)) {
    if (channel_missed_semantics == "accepted-only") {
      if (estimator_split_reject)
        ++tally.estimator_high_error_caught_by_split;
      else
        ++tally.estimator_high_error_missed;
    } else if (estimator_unstable) {
      ++tally.estimator_high_error_caught;
    } else {
      ++tally.estimator_high_error_missed;
    }
  }
}

void record_far_expansion_channel_audit(
    ExpansionAuditTally &tally, const JcpChannelAccum &taylor,
    const JcpChannelAccum &direct,
    const ConvergenceEstimatorDecision &estimator,
    double estimator_error_cap,
    double accepted_miss_absolute_floor = 1.0e-7,
    const std::string &channel_missed_semantics = "legacy",
    const std::string &accepted_miss_diagnostics = "off",
    std::size_t target_index = 0, std::size_t source_node_index = 0,
    std::size_t panel_count = 0, double rho = 0.0) {
  std::size_t pair_worst = kJcpExpansionChannelCount;
  double pair_worst_rel = 0.0;
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel) {
    const bool channel_coupled =
        channel == kJcpK3PhinChannel || channel == kJcpK4MixedChannel;
    const double rel = jcp_rel_error(taylor.channel[channel],
                                     direct.channel[channel],
                                     channel_coupled);
    const double abs_error =
        jcp_abs_error(taylor.channel[channel], direct.channel[channel],
                      channel_coupled);
    tally.channel_rel_error_sum[channel] += rel;
    tally.channel_rel_error_max[channel] =
        std::max(tally.channel_rel_error_max[channel], rel);
    tally.channel_rel_errors[channel].push_back(rel);
    if (estimator.active &&
        is_phantom_high_error(rel, abs_error, estimator_error_cap,
                              accepted_miss_absolute_floor)) {
      ++tally.channel_phantom_high_error_count[channel];
      tally.channel_phantom_high_error_abs_max[channel] =
          std::max(tally.channel_phantom_high_error_abs_max[channel],
                   abs_error);
      if (channel_missed_semantics == "accepted-only" &&
          !estimator.split_reject) {
        record_accepted_miss_diagnostic(
            tally, target_index, source_node_index, channel, panel_count, rho,
            estimator.channel_tail_ratio[channel],
            estimator.channel_estimated_error[channel], rel, abs_error, true,
            accepted_miss_diagnostics);
      }
    } else if (estimator.active &&
               is_real_high_error(rel, abs_error, estimator_error_cap,
                                  accepted_miss_absolute_floor)) {
      if (channel_missed_semantics == "accepted-only") {
        if (estimator.split_reject) {
          ++tally.channel_high_error_caught_by_split[channel];
        } else {
          ++tally.channel_high_error_missed[channel];
          record_accepted_miss_diagnostic(
              tally, target_index, source_node_index, channel, panel_count,
              rho, estimator.channel_tail_ratio[channel],
              estimator.channel_estimated_error[channel], rel, abs_error,
              false,
              accepted_miss_diagnostics);
        }
      } else if (estimator.channel_unstable[channel]) {
        ++tally.channel_high_error_caught[channel];
      } else {
        ++tally.channel_high_error_missed[channel];
      }
    }
    if (rel > pair_worst_rel) {
      pair_worst_rel = rel;
      pair_worst = channel;
    }
  }
  if (pair_worst < kJcpExpansionChannelCount &&
      (tally.worst_channel >= kJcpExpansionChannelCount ||
       pair_worst_rel > tally.channel_rel_error_max[tally.worst_channel])) {
    tally.worst_channel = pair_worst;
  }
}

void record_split_far_expansion_channel_audit(
    ExpansionAuditTally &tally, const JcpAccum &taylor,
    const JcpAccum &direct, const JcpChannelAccum &taylor_channels,
    const JcpChannelAccum &direct_channels,
    const ConvergenceEstimatorDecision &estimator, bool coupled,
    double estimator_error_cap,
    double accepted_miss_absolute_floor = 1.0e-7) noexcept {
  if (!estimator.active || !estimator.split_reject ||
      estimator_error_cap <= 0.0)
    return;
  const double rel = jcp_rel_error(taylor, direct, coupled);
  const double abs_error = jcp_abs_error(taylor, direct, coupled);
  if (is_real_high_error(rel, abs_error, estimator_error_cap,
                         accepted_miss_absolute_floor)) {
    ++tally.estimator_high_error_caught_by_split;
  } else if (is_phantom_high_error(rel, abs_error, estimator_error_cap,
                                   accepted_miss_absolute_floor)) {
    ++tally.estimator_phantom_high_error_count;
    tally.estimator_phantom_high_error_abs_max =
        std::max(tally.estimator_phantom_high_error_abs_max, abs_error);
  }
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel) {
    const bool channel_coupled =
        channel == kJcpK3PhinChannel || channel == kJcpK4MixedChannel;
    const double channel_rel =
        jcp_rel_error(taylor_channels.channel[channel],
                      direct_channels.channel[channel], channel_coupled);
    const double channel_abs =
        jcp_abs_error(taylor_channels.channel[channel],
                      direct_channels.channel[channel], channel_coupled);
    if (is_real_high_error(channel_rel, channel_abs, estimator_error_cap,
                           accepted_miss_absolute_floor)) {
      ++tally.channel_high_error_caught_by_split[channel];
    } else if (is_phantom_high_error(channel_rel, channel_abs,
                                     estimator_error_cap,
                                     accepted_miss_absolute_floor)) {
      ++tally.channel_phantom_high_error_count[channel];
      tally.channel_phantom_high_error_abs_max[channel] =
          std::max(tally.channel_phantom_high_error_abs_max[channel],
                   channel_abs);
    }
  }
}

void record_convergence_overreject_audit(
    ExpansionAuditTally &tally, const JcpAccum &taylor,
    const JcpAccum &direct, const JcpChannelAccum &taylor_channels,
    const JcpChannelAccum &direct_channels,
    const ConvergenceEstimatorDecision &estimator, const TreeNode &node,
    bool coupled, double estimator_error_cap,
    double accepted_miss_absolute_floor, const std::string &detail_mode,
    const std::string &shape_mode, const std::string &kernel_response_mode,
    std::size_t target_index,
    std::size_t source_node_index, double rho, double target_distance,
    int taylor_order, const JcpChannelAccum &kernel_tail_p,
    const JcpChannelAccum &kernel_tail_pm1) {
  if (!estimator.active || !estimator.split_reject ||
      estimator_error_cap <= 0.0)
    return;
  const bool record_shape =
      shape_mode == "summary" || shape_mode == "sampled-detail";
  const bool record_kernel_response =
      kernel_response_mode == "summary" ||
      kernel_response_mode == "sampled-detail";
  const double rho_power = pow_int(rho, std::max(0, taylor_order + 1));
  auto append_shape_record =
      [&](std::size_t channel, double source_norm, double tail_norm_p,
          double tail_norm_pm1, double tail_ratio, double effective_amp,
          double estimated_rel, double estimated_abs, double oracle_rel,
          double oracle_abs, bool caught) {
        if (!record_shape || tally.convergence_shape_records.size() >= 20480)
          return;
        const double tail_norm =
            std::max(tail_norm_p, tail_norm_pm1 * std::max(0.0, rho));
        const double tail_rho = tail_norm * std::max(1.0e-300, rho_power);
        tally.convergence_shape_records.push_back(
            {target_index,
             source_node_index,
             channel,
             node.panel_count,
             rho,
             rho_power,
             node_aspect_ratio(node),
             node.radius,
             target_distance,
             source_norm,
             tail_norm_p,
             tail_norm_pm1,
             tail_ratio,
             effective_amp,
             estimated_rel,
             estimated_abs,
             oracle_rel,
             oracle_abs,
             oracle_abs / std::max(1.0e-300, source_norm),
             oracle_abs / std::max(1.0e-300, tail_norm),
             oracle_abs / std::max(1.0e-300, tail_rho),
             caught});
      };
  auto append_kernel_response_record =
      [&](std::size_t channel, double source_norm, double tail_norm_p,
          double tail_norm_pm1, double estimated_abs, double oracle_rel,
          double oracle_abs, bool caught) {
        if (!record_kernel_response ||
            tally.kernel_response_records.size() >= 20480)
          return;
        const bool channel_coupled =
            channel >= kJcpExpansionChannelCount ||
            channel == kJcpK3PhinChannel || channel == kJcpK4MixedChannel;
        const JcpAccum zero{};
        const JcpAccum response_p =
            channel >= kJcpExpansionChannelCount
                ? sum_jcp_channels(kernel_tail_p)
                : kernel_tail_p.channel[channel];
        const JcpAccum response_pm1 =
            channel >= kJcpExpansionChannelCount
                ? sum_jcp_channels(kernel_tail_pm1)
                : kernel_tail_pm1.channel[channel];
        const double tail_p_abs =
            jcp_abs_error(response_p, zero, channel_coupled);
        const double tail_pm1_abs =
            jcp_abs_error(response_pm1, zero, channel_coupled);
        const double envelope =
            std::max(tail_p_abs, std::max(0.0, rho) * tail_pm1_abs);
        tally.kernel_response_records.push_back(
            {target_index,
             source_node_index,
             channel,
             node.panel_count,
             rho,
             node_aspect_ratio(node),
             jcp_kernel_source_family(channel),
             source_norm,
             tail_norm_p,
             tail_norm_pm1,
             estimated_abs,
             tail_p_abs,
             tail_pm1_abs,
             envelope,
             oracle_rel,
             oracle_abs,
             oracle_abs / std::max(1.0e-300, envelope),
             caught});
      };
  const double rel = jcp_rel_error(taylor, direct, coupled);
  const double abs_error = jcp_abs_error(taylor, direct, coupled);
  const detail::ConvergenceOverrejectAuditDecision global_decision =
      detail::classify_convergence_overreject_audit(
          estimator.split_reject, rel, abs_error, estimator_error_cap,
          accepted_miss_absolute_floor);
  const double global_scale =
      abs_error / std::max(1.0e-300, estimator.estimated_abs_error);
  if (global_decision.caught_high_error) {
    ++tally.convergence_overreject_caught_count;
    tally.convergence_overreject_caught_panels += node.panel_count;
  } else if (global_decision.low_error_overreject) {
    ++tally.convergence_overreject_count;
    tally.convergence_overreject_panels += node.panel_count;
    tally.convergence_overreject_abs_error_sum += abs_error;
    tally.convergence_overreject_abs_error_max =
        std::max(tally.convergence_overreject_abs_error_max, abs_error);
    if (detail_mode != "off") {
      tally.convergence_overreject_estimated_abs_error_sum +=
          estimator.estimated_abs_error;
      tally.convergence_overreject_estimated_abs_error_max =
          std::max(tally.convergence_overreject_estimated_abs_error_max,
                   estimator.estimated_abs_error);
      tally.convergence_overreject_abs_error_samples.push_back(abs_error);
      tally.convergence_overreject_required_scale_samples.push_back(
          global_scale);
    }
  }
  if (detail_mode == "sampled-detail" &&
      (global_decision.caught_high_error ||
       global_decision.low_error_overreject) &&
      tally.overreject_calibration_records.size() < 20480) {
    tally.overreject_calibration_records.push_back(
        {target_index, source_node_index, kJcpExpansionChannelCount,
         node.panel_count, rho, node_aspect_ratio(node),
         estimator.estimated_error, estimator.estimated_abs_error, rel,
         abs_error, global_scale, global_decision.caught_high_error});
  }
  if (global_decision.caught_high_error ||
      global_decision.low_error_overreject) {
    append_shape_record(kJcpExpansionChannelCount, estimator.source_norm,
                        estimator.tail_norm_p, estimator.tail_norm_pm1,
                        estimator.tail_ratio, estimator.effective_jcp_amp,
                        estimator.estimated_error,
                        estimator.estimated_abs_error, rel, abs_error,
                        global_decision.caught_high_error);
    append_kernel_response_record(
        kJcpExpansionChannelCount, estimator.source_norm,
        estimator.tail_norm_p, estimator.tail_norm_pm1,
        estimator.estimated_abs_error, rel, abs_error,
        global_decision.caught_high_error);
  }
  for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
       ++channel) {
    const bool channel_coupled =
        channel == kJcpK3PhinChannel || channel == kJcpK4MixedChannel;
    const double channel_rel =
        jcp_rel_error(taylor_channels.channel[channel],
                      direct_channels.channel[channel], channel_coupled);
    const double channel_abs =
        jcp_abs_error(taylor_channels.channel[channel],
                      direct_channels.channel[channel], channel_coupled);
    const detail::ConvergenceOverrejectAuditDecision channel_decision =
        detail::classify_convergence_overreject_audit(
            estimator.split_reject, channel_rel, channel_abs,
            estimator_error_cap, accepted_miss_absolute_floor);
    const double channel_scale =
        channel_abs / std::max(1.0e-300,
                               estimator.channel_estimated_abs_error[channel]);
    if (channel_decision.caught_high_error) {
      ++tally.convergence_channel_overreject_caught_count[channel];
    } else if (channel_decision.low_error_overreject) {
      ++tally.convergence_channel_overreject_count[channel];
      if (detail_mode != "off") {
        tally.convergence_channel_overreject_panels[channel] +=
            node.panel_count;
        tally.convergence_channel_overreject_abs_error_sum[channel] +=
            channel_abs;
        tally.convergence_channel_overreject_abs_error_max[channel] =
            std::max(tally.convergence_channel_overreject_abs_error_max[channel],
                     channel_abs);
        tally.convergence_channel_overreject_required_scale_samples[channel]
            .push_back(channel_scale);
      }
    }
    if (detail_mode == "sampled-detail" &&
        (channel_decision.caught_high_error ||
         channel_decision.low_error_overreject) &&
        tally.overreject_calibration_records.size() < 20480) {
      tally.overreject_calibration_records.push_back(
          {target_index, source_node_index, channel, node.panel_count, rho,
           node_aspect_ratio(node), estimator.channel_estimated_error[channel],
           estimator.channel_estimated_abs_error[channel], channel_rel,
           channel_abs, channel_scale, channel_decision.caught_high_error});
    }
    if (channel_decision.caught_high_error ||
        channel_decision.low_error_overreject) {
      append_shape_record(
          channel, estimator.channel_source_norm[channel],
          estimator.channel_tail_norm_p[channel],
          estimator.channel_tail_norm_pm1[channel],
          estimator.channel_tail_ratio[channel],
          estimator.channel_effective_jcp_amp[channel],
          estimator.channel_estimated_error[channel],
          estimator.channel_estimated_abs_error[channel], channel_rel,
          channel_abs, channel_decision.caught_high_error);
      append_kernel_response_record(
          channel, estimator.channel_source_norm[channel],
          estimator.channel_tail_norm_p[channel],
          estimator.channel_tail_norm_pm1[channel],
          estimator.channel_estimated_abs_error[channel], channel_rel,
          channel_abs, channel_decision.caught_high_error);
    }
  }
}



double channel_moment_norm_by_degree(const NodeMoments &moments,
                                     const std::vector<MomentIndex> &indices,
                                     int degree, double radius) noexcept {
  double out = 0.0;
  const double safe_radius = std::max(1.0e-12, radius);
  const double scale = pow_int(safe_radius, degree);
  for (std::size_t m = 0; m < indices.size(); ++m) {
    const MomentIndex &index = indices[m];
    if (index.ax + index.ay + index.az != degree)
      continue;
    out = std::max(out, std::abs(moments.values[m]) /
                            std::max(1.0e-300, scale));
  }
  return out;
}

ConvergenceEstimatorDecision evaluate_convergence_estimator(
    const TreeWorkspace &tree, const TreeNode &node,
    const PackedJcpMoments &packed, std::size_t node_index, int taylor_order,
    bool coupled, double rho, double eps_ratio) noexcept {
  ConvergenceEstimatorDecision decision;
  if (tree.convergence_estimator == "off" || node_index >= packed.phin.size() ||
      taylor_order <= 0 || rho <= 0.0 || !std::isfinite(rho))
    return decision;
  decision.active = true;
  const auto &indices = moment_indices(taylor_order);
  const bool use_consecutive =
      tree.convergence_calibration == "consecutive" ||
      tree.convergence_calibration == "consecutive-aspect";
  const bool use_aspect = tree.convergence_calibration == "aspect" ||
                          tree.convergence_calibration == "consecutive-aspect";
  const double safe_eps_ratio =
      std::isfinite(eps_ratio) ? std::abs(eps_ratio) : 1.0;
  const double aspect = std::max(1.0, node_aspect_ratio(node));
  const double rho_factor = pow_int(rho, taylor_order + 1);
  const JcpConvergenceNorms *cached_norms = nullptr;
  if (node_index < packed.convergence_norms.size()) {
    const JcpConvergenceNorms &candidate = packed.convergence_norms[node_index];
    if (candidate.valid && candidate.taylor_order == taylor_order)
      cached_norms = &candidate;
  }

  auto norms_for_moments = [&](const NodeMoments &moments, double &source_norm,
                               double &tail_norm_p,
                               double &tail_norm_pm1) {
    source_norm = std::max(
        source_norm,
        channel_moment_norm_by_degree(moments, indices, 0, node.radius));
    source_norm = std::max(
        source_norm,
        channel_moment_norm_by_degree(moments, indices, 1, node.radius));
    tail_norm_p = std::max(
        tail_norm_p,
        channel_moment_norm_by_degree(moments, indices, taylor_order,
                                      node.radius));
    tail_norm_pm1 = std::max(
        tail_norm_pm1,
        channel_moment_norm_by_degree(moments, indices, taylor_order - 1,
                                      node.radius));
  };
  auto finish_global = [&]() {
    const bool tail_limited =
        decision.tail_ratio > tree.convergence_tail_threshold;
    const bool error_limited =
        tree.convergence_estimator == "hybrid" &&
        decision.estimated_error > tree.convergence_estimator_error_cap;
    decision.unstable = tail_limited || error_limited;
    for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
         ++channel) {
      decision.channel_estimated_error[channel] = decision.estimated_error;
      decision.channel_estimated_abs_error[channel] =
          decision.estimated_abs_error;
      decision.channel_effective_jcp_amp[channel] =
          decision.effective_jcp_amp;
      decision.channel_source_norm[channel] = decision.source_norm;
      decision.channel_tail_norm_p[channel] = decision.tail_norm_p;
      decision.channel_tail_norm_pm1[channel] = decision.tail_norm_pm1;
      decision.channel_tail_ratio[channel] = decision.tail_ratio;
      decision.channel_unstable[channel] = decision.unstable;
    }
  };

  if (tree.convergence_estimator_model == "stage3l") {
    double source_norm = 0.0;
    double tail_norm_p = 0.0;
    double tail_norm_pm1 = 0.0;
    if (cached_norms != nullptr) {
      source_norm = cached_norms->normal_source_norm;
      tail_norm_p = cached_norms->normal_tail_norm_p;
      tail_norm_pm1 = cached_norms->normal_tail_norm_pm1;
      if (coupled) {
        source_norm = std::max(source_norm, cached_norms->scalar_source_norm);
        tail_norm_p = std::max(tail_norm_p, cached_norms->scalar_tail_norm_p);
        tail_norm_pm1 =
            std::max(tail_norm_pm1, cached_norms->scalar_tail_norm_pm1);
      }
    } else {
      if (coupled)
        norms_for_moments(packed.phin[node_index], source_norm, tail_norm_p,
                          tail_norm_pm1);
      if (node_index < packed.phi_normal[0].size()) {
        norms_for_moments(packed.phi_normal[0][node_index], source_norm,
                          tail_norm_p, tail_norm_pm1);
        norms_for_moments(packed.phi_normal[1][node_index], source_norm,
                          tail_norm_p, tail_norm_pm1);
        norms_for_moments(packed.phi_normal[2][node_index], source_norm,
                          tail_norm_p, tail_norm_pm1);
      }
    }
    decision.tail_norm_p = tail_norm_p;
    decision.tail_norm_pm1 = tail_norm_pm1;
    decision.source_norm = source_norm;
    const double tail_norm =
        use_consecutive ? std::max(tail_norm_p, tail_norm_pm1 * rho)
                        : tail_norm_p;
    decision.tail_ratio = tail_norm / std::max(source_norm, 1.0e-300);
    double jcp_amp =
        coupled ? (tree.convergence_dielectric_amp == "on"
                       ? std::max(30.0, safe_eps_ratio)
                       : 30.0)
                : 1.0;
    decision.dielectric_jcp_amp = jcp_amp;
    if (use_aspect && coupled)
      jcp_amp *= aspect;
    decision.effective_jcp_amp = jcp_amp;
    decision.estimated_error = decision.tail_ratio * rho_factor * jcp_amp;
    decision.estimated_abs_error = tail_norm * rho_factor * jcp_amp;
    finish_global();
  } else {
    std::array<double, kJcpExpansionChannelCount> source_norm{};
    std::array<double, kJcpExpansionChannelCount> tail_norm_p{};
    std::array<double, kJcpExpansionChannelCount> tail_norm_pm1{};
    auto observe_channel = [&](std::size_t channel,
                               const NodeMoments &moments) {
      norms_for_moments(moments, source_norm[channel], tail_norm_p[channel],
                        tail_norm_pm1[channel]);
    };
    if (cached_norms != nullptr) {
      auto copy_channel = [&](std::size_t channel) {
        source_norm[channel] = cached_norms->channel_source_norm[channel];
        tail_norm_p[channel] = cached_norms->channel_tail_norm_p[channel];
        tail_norm_pm1[channel] = cached_norms->channel_tail_norm_pm1[channel];
      };
      if (coupled) {
        copy_channel(kJcpK1PhiChannel);
        copy_channel(kJcpK3PhinChannel);
      }
      copy_channel(kJcpK2PhiNormalChannel);
      if (coupled)
        copy_channel(kJcpK4MixedChannel);
    } else {
      if (coupled) {
        observe_channel(kJcpK1PhiChannel, packed.phin[node_index]);
        observe_channel(kJcpK3PhinChannel, packed.phin[node_index]);
      }
      if (node_index < packed.phi_normal[0].size()) {
        for (int component = 0; component < 3; ++component) {
          const NodeMoments &moments =
              packed.phi_normal[static_cast<std::size_t>(component)]
                               [node_index];
          observe_channel(kJcpK2PhiNormalChannel, moments);
          if (coupled)
            observe_channel(kJcpK4MixedChannel, moments);
        }
      }
    }
    const bool eps_channel_model =
        tree.convergence_estimator_model == "channel-tail-eps";
    const bool k3_sqrt_eps =
        tree.convergence_channel_calibration == "k3-sqrt-eps";
    const bool k3_eps = tree.convergence_channel_calibration == "k3-eps";
    const bool k4_sqrt_eps =
        tree.convergence_channel_calibration == "k4-sqrt-eps";
    const bool k4_eps = tree.convergence_channel_calibration == "k4-eps";
    const bool all_derivative_sqrt_eps =
        tree.convergence_channel_calibration == "all-derivative-sqrt-eps";
    const bool polynomial_eps = eps_channel_model &&
                                tree.convergence_channel_calibration ==
                                    "polynomial-eps";
    const double sqrt_eps_amp =
        coupled ? std::max(30.0, 30.0 * std::sqrt(safe_eps_ratio)) : 1.0;
    const double polynomial_d1_amp = static_cast<double>(taylor_order) + 1.0;
    const double polynomial_d2_amp =
        polynomial_d1_amp * (static_cast<double>(taylor_order) + 2.0);
    const double polynomial_d1_eps_amp =
        coupled ? std::max(30.0, polynomial_d1_amp * 30.0 *
                                     std::sqrt(safe_eps_ratio))
                : 1.0;
    const double polynomial_d2_eps_amp =
        coupled ? std::max(30.0, polynomial_d2_amp * safe_eps_ratio) : 1.0;
    double dielectric_amp = coupled ? 30.0 : 1.0;
    if (coupled && tree.convergence_dielectric_amp == "on")
      dielectric_amp = std::max(30.0, safe_eps_ratio);
    decision.dielectric_jcp_amp = dielectric_amp;
    double effective_amp_max = 0.0;
    for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
         ++channel) {
      const double tail_norm =
          use_consecutive
              ? std::max(tail_norm_p[channel], tail_norm_pm1[channel] * rho)
              : tail_norm_p[channel];
      const double tail_ratio =
          tail_norm / std::max(source_norm[channel], 1.0e-300);
      double amp = coupled ? 30.0 : 1.0;
      if (channel == kJcpK1PhiChannel)
        amp = 1.0;
      if (!polynomial_eps && eps_channel_model &&
          channel == kJcpK2PhiNormalChannel && coupled &&
          tree.convergence_dielectric_amp == "on")
        amp = std::max(30.0, safe_eps_ratio);
      if (coupled) {
        if (polynomial_eps &&
            (channel == kJcpK2PhiNormalChannel ||
             channel == kJcpK3PhinChannel)) {
          amp = std::max(amp, polynomial_d1_eps_amp);
        } else if (polynomial_eps && channel == kJcpK4MixedChannel) {
          amp = std::max(amp, polynomial_d2_eps_amp);
        } else if ((k3_sqrt_eps || all_derivative_sqrt_eps) &&
            channel == kJcpK3PhinChannel) {
          amp = std::max(amp, sqrt_eps_amp);
        } else if (k3_eps && channel == kJcpK3PhinChannel) {
          amp = std::max(amp, safe_eps_ratio);
        } else if ((k4_sqrt_eps || all_derivative_sqrt_eps) &&
                   channel == kJcpK4MixedChannel) {
          amp = std::max(amp, sqrt_eps_amp);
        } else if (k4_eps && channel == kJcpK4MixedChannel) {
          amp = std::max(amp, safe_eps_ratio);
        } else if (all_derivative_sqrt_eps &&
                   channel == kJcpK2PhiNormalChannel) {
          amp = std::max(amp, sqrt_eps_amp);
        }
      }
      if (use_aspect && coupled &&
          (channel == kJcpK2PhiNormalChannel ||
           channel == kJcpK4MixedChannel ||
           (polynomial_eps && channel == kJcpK3PhinChannel)))
        amp *= aspect;
      const double estimated_error = tail_ratio * rho_factor * amp;
      const double estimated_abs_error = tail_norm * rho_factor * amp;
      decision.channel_estimated_error[channel] = estimated_error;
      decision.channel_estimated_abs_error[channel] = estimated_abs_error;
      decision.channel_effective_jcp_amp[channel] = amp;
      decision.channel_source_norm[channel] = source_norm[channel];
      decision.channel_tail_norm_p[channel] = tail_norm_p[channel];
      decision.channel_tail_norm_pm1[channel] = tail_norm_pm1[channel];
      decision.channel_tail_ratio[channel] = tail_ratio;
      decision.channel_unstable[channel] =
          tail_ratio > tree.convergence_tail_threshold ||
          (tree.convergence_estimator == "hybrid" &&
           estimated_error > tree.convergence_estimator_error_cap);
      decision.tail_norm_p = std::max(decision.tail_norm_p,
                                      tail_norm_p[channel]);
      decision.tail_norm_pm1 = std::max(decision.tail_norm_pm1,
                                        tail_norm_pm1[channel]);
      decision.source_norm = std::max(decision.source_norm,
                                      source_norm[channel]);
      decision.tail_ratio = std::max(decision.tail_ratio, tail_ratio);
      decision.estimated_error =
          std::max(decision.estimated_error, estimated_error);
      decision.estimated_abs_error =
          std::max(decision.estimated_abs_error, estimated_abs_error);
      effective_amp_max = std::max(effective_amp_max, amp);
      decision.unstable = decision.unstable || decision.channel_unstable[channel];
    }
    decision.effective_jcp_amp = effective_amp_max;
  }
  detail::ConvergenceSplitSelectivityInput split_input;
  split_input.unstable = decision.unstable;
  split_input.safety_split = tree.convergence_safety == "split";
  split_input.rho = rho;
  split_input.unstable_rho_cap = tree.convergence_unstable_rho_cap;
  split_input.split_selectivity = tree.convergence_split_selectivity;
  split_input.split_absolute_floor = tree.convergence_split_absolute_floor;
  split_input.split_policy = tree.convergence_split_policy;
  split_input.estimator_model = tree.convergence_estimator_model;
  split_input.estimated_abs_error = decision.estimated_abs_error;
  split_input.channel_unstable = decision.channel_unstable;
  split_input.channel_estimated_abs_error =
      decision.channel_estimated_abs_error;
  const detail::ConvergenceSplitSelectivityResult split_result =
      detail::evaluate_convergence_split_selectivity(split_input);
  decision.split_reject = split_result.split_reject;
  decision.low_abs_reaccepted = split_result.low_abs_reaccepted;
  decision.channel_low_abs_reaccepted =
      split_result.channel_low_abs_reaccepted;
  return decision;
}


} // namespace TitanPBE::Tabi::detail
