#include "tabi_fmm_kifmm_hierarchy_real_panel.h"

#include "tabi_fmm_cartesian_sampled_panel_matvec.h"
#include "tabi_fmm_kifmm_hierarchy.h"
#include "tabi_fmm_kifmm_real_panel_geometry.h"
#include "tabi_fmm_kifmm_real_panel_worklist.h"
#include "tabi_fmm_m2l_cartesian.h"
#include "tabi_near.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <cstdio>
#endif

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace titan::tabi::fmm_kifmm {
namespace {

using Clock = std::chrono::steady_clock;
using TitanPBE::Tabi::JcpNearPairKernel;
using TitanPBE::Tabi::Panel;

double elapsed_seconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

struct ProcessMemorySample {
  std::uint64_t rss_bytes{0};
  std::uint64_t peak_working_set_bytes{0};
};

ProcessMemorySample current_process_memory_sample() noexcept {
  ProcessMemorySample sample;
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters,
                              sizeof(counters))) {
    sample.rss_bytes = static_cast<std::uint64_t>(counters.WorkingSetSize);
    sample.peak_working_set_bytes =
        static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
  }
#else
  FILE *file = std::fopen("/proc/self/status", "r");
  if (file != nullptr) {
    char line[256];
    char label[64];
    long value_kb = 0;
    while (std::fgets(line, sizeof(line), file) != nullptr) {
      if (std::sscanf(line, "%63s %ld", label, &value_kb) != 2) {
        continue;
      }
      if (std::string(label) == "VmRSS:") {
        sample.rss_bytes =
            static_cast<std::uint64_t>(std::max<long>(0, value_kb)) * 1024u;
      } else if (std::string(label) == "VmHWM:") {
        sample.peak_working_set_bytes =
            static_cast<std::uint64_t>(std::max<long>(0, value_kb)) * 1024u;
      }
    }
    std::fclose(file);
  }
#endif
  return sample;
}

void update_apply_peak_delta(KifmmHierarchyRealPanelResult &result) {
  if (result.apply_peak_working_set_after_finalize_bytes >=
      result.apply_peak_working_set_start_bytes) {
    result.apply_peak_working_set_delta_bytes =
        result.apply_peak_working_set_after_finalize_bytes -
        result.apply_peak_working_set_start_bytes;
  } else {
    result.apply_peak_working_set_delta_bytes = 0;
  }
}

void record_apply_memory_start(KifmmHierarchyRealPanelResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.apply_rss_start_bytes = sample.rss_bytes;
  result.apply_peak_working_set_start_bytes = sample.peak_working_set_bytes;
  result.apply_rss_after_finalize_bytes = sample.rss_bytes;
  result.apply_peak_working_set_after_finalize_bytes =
      sample.peak_working_set_bytes;
  update_apply_peak_delta(result);
}

void record_apply_memory_after_source_build(
    KifmmHierarchyRealPanelResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.apply_rss_after_source_build_bytes = sample.rss_bytes;
  result.apply_peak_working_set_after_source_build_bytes =
      sample.peak_working_set_bytes;
}

void record_apply_memory_after_hierarchy(
    KifmmHierarchyRealPanelResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.apply_rss_after_hierarchy_bytes = sample.rss_bytes;
  result.apply_peak_working_set_after_hierarchy_bytes =
      sample.peak_working_set_bytes;
}

void record_apply_memory_after_guard_audit(
    KifmmHierarchyRealPanelResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.apply_rss_after_guard_audit_bytes = sample.rss_bytes;
  result.apply_peak_working_set_after_guard_audit_bytes =
      sample.peak_working_set_bytes;
}

void record_apply_memory_after_fallback(
    KifmmHierarchyRealPanelResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.apply_rss_after_fallback_bytes = sample.rss_bytes;
  result.apply_peak_working_set_after_fallback_bytes =
      sample.peak_working_set_bytes;
}

void record_apply_memory_after_cartesian_k4(
    KifmmHierarchyRealPanelResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.apply_rss_after_cartesian_k4_bytes = sample.rss_bytes;
  result.apply_peak_working_set_after_cartesian_k4_bytes =
      sample.peak_working_set_bytes;
}

void record_apply_memory_after_dense_oracle(
    KifmmHierarchyRealPanelResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.apply_rss_after_dense_oracle_bytes = sample.rss_bytes;
  result.apply_peak_working_set_after_dense_oracle_bytes =
      sample.peak_working_set_bytes;
}

void record_apply_memory_after_finalize(
    KifmmHierarchyRealPanelResult &result) {
  const ProcessMemorySample sample = current_process_memory_sample();
  result.apply_rss_after_finalize_bytes = sample.rss_bytes;
  result.apply_peak_working_set_after_finalize_bytes =
      sample.peak_working_set_bytes;
  update_apply_peak_delta(result);
}

double compute_apply_unattributed_seconds(
    const KifmmHierarchyRealPanelResult &result) noexcept {
  const double attributed = result.apply_source_build_seconds +
                            result.apply_hierarchy_seconds +
                            result.apply_guard_audit_seconds +
                            result.apply_production_fallback_seconds +
                            result.apply_cartesian_k4_seconds +
                            result.apply_dense_oracle_seconds +
                            result.apply_finalize_seconds;
  if (!std::isfinite(result.total_seconds) || !std::isfinite(attributed)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::max(0.0, result.total_seconds - attributed);
}

double pair_fraction(std::uint64_t numerator,
                     std::uint64_t denominator) noexcept {
  if (denominator == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

std::uint64_t mix_hash(std::uint64_t seed, std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31;
  return seed ^ value;
}

std::uint64_t hash_double(std::uint64_t seed, double value) noexcept {
  return mix_hash(seed, std::bit_cast<std::uint64_t>(value));
}

void copy_hierarchy_closeout_fields(KifmmHierarchyRealPanelResult &result) {
  result.hierarchy_inner_status = result.hierarchy.status;
  result.hierarchy_inner_accuracy_failure = result.hierarchy.accuracy_failure;
  result.hybrid_mode = result.hierarchy.hybrid_mode;
  result.m2e_max_relative_residual =
      result.hierarchy.m2e_max_relative_residual;
  result.local_equiv_max_relative_residual =
      result.hierarchy.local_equiv_max_relative_residual;
  result.e2l_coefficient_relative_rms =
      result.hierarchy.e2l_coefficient_relative_rms;
  result.e2l_coefficient_status = result.hierarchy.e2l_coefficient_status;
  result.k4_decomposition_status = result.hierarchy.k4_decomposition_status;
  result.split_k4_laplace_relative_rms =
      result.hierarchy.split_k4_laplace_relative_rms;
  result.split_k4_yukawa_relative_rms =
      result.hierarchy.split_k4_yukawa_relative_rms;
  result.split_k4_combined_relative_rms =
      result.hierarchy.split_k4_combined_relative_rms;
  result.split_vs_dense_k4_relative_rms =
      result.hierarchy.split_vs_dense_k4_relative_rms;
  result.l2p_gradient_method = result.hierarchy.l2p_gradient_method;
  result.m2l_eval_policy = result.hierarchy.m2l_eval_policy;
  result.m2l_matrix_cache_policy = result.hierarchy.m2l_matrix_cache_policy;
  result.m2l_gemm_batch_policy = result.hierarchy.m2l_gemm_batch_policy;
  result.m2l_parallel_policy = result.hierarchy.m2l_parallel_policy;
  result.m2l_parallel_workers = result.hierarchy.m2l_parallel_workers;
  result.jcp_channel_policy = result.hierarchy.jcp_channel_policy;
  result.hierarchy_k4_policy = result.hierarchy.hierarchy_k4_policy;
  result.hierarchy_k4_evaluation_count =
      result.hierarchy.hierarchy_k4_evaluation_count;
  result.m2l_matrix_cache_entries =
      result.hierarchy.m2l_matrix_cache_entries;
  result.m2l_matrix_cache_peak_entries =
      result.hierarchy.m2l_matrix_cache_peak_entries;
  result.m2l_matrix_cache_hits = result.hierarchy.m2l_matrix_cache_hits;
  result.m2l_matrix_cache_misses = result.hierarchy.m2l_matrix_cache_misses;
  result.m2l_matrix_cache_hit_rate =
      result.hierarchy.m2l_matrix_cache_hit_rate;
  result.m2l_matrix_cache_limit_bytes =
      result.hierarchy.m2l_matrix_cache_limit_bytes;
  result.m2l_matrix_cache_evictions =
      result.hierarchy.m2l_matrix_cache_evictions;
  result.m2l_matrix_cache_estimated_bytes =
      result.hierarchy.m2l_matrix_cache_estimated_bytes;
  result.m2l_matrix_cache_peak_estimated_bytes =
      result.hierarchy.m2l_matrix_cache_peak_estimated_bytes;
  result.m2l_matrix_cache_total_built_bytes =
      result.hierarchy.m2l_matrix_cache_total_built_bytes;
  result.m2l_matrix_cache_build_seconds =
      result.hierarchy.m2l_matrix_cache_build_seconds;
  result.m2l_matrix_gemm_seconds = result.hierarchy.m2l_matrix_gemm_seconds;
  result.m2l_matrix_cache_key_status =
      result.hierarchy.m2l_matrix_cache_key_status;
  result.m2l_gemm_group_count = result.hierarchy.m2l_gemm_group_count;
  result.m2l_gemm_grouped_pair_count =
      result.hierarchy.m2l_gemm_grouped_pair_count;
  result.m2l_gemm_grouped_rhs_columns =
      result.hierarchy.m2l_gemm_grouped_rhs_columns;
  result.m2l_gemm_max_group_size = result.hierarchy.m2l_gemm_max_group_size;
  result.m2l_parallel_task_count =
      result.hierarchy.m2l_parallel_task_count;
  result.m2l_parallel_group_count =
      result.hierarchy.m2l_parallel_group_count;
  result.m2l_parallel_max_task_pairs =
      result.hierarchy.m2l_parallel_max_task_pairs;
  result.m2l_parallel_prepare_seconds =
      result.hierarchy.m2l_parallel_prepare_seconds;
  result.m2l_parallel_eval_seconds =
      result.hierarchy.m2l_parallel_eval_seconds;
  result.m2l_parallel_merge_seconds =
      result.hierarchy.m2l_parallel_merge_seconds;
  result.m2l_target_nodes = result.hierarchy.m2l_target_nodes;
  result.m2l_source_nodes = result.hierarchy.m2l_source_nodes;
  result.m2l_check_points_total = result.hierarchy.m2l_check_points_total;
  result.m2l_equivalent_points = result.hierarchy.m2l_equivalent_points;
  result.m2l_kernel_evals_estimated =
      result.hierarchy.m2l_kernel_evals_estimated;
  result.m2l_seconds_per_translation =
      result.hierarchy.m2l_seconds_per_translation;
  result.m2l_seconds_per_check_point =
      result.hierarchy.m2l_seconds_per_check_point;
}

void copy_tree_policy_and_stats(KifmmHierarchyRealPanelResult &result,
                                const KifmmHierarchyPolicy &policy,
                                const KifmmHierarchyTree &tree) {
  result.m2l_matrix_cache_policy = policy.m2l_matrix_cache_policy;
  result.m2l_gemm_batch_policy = policy.m2l_gemm_batch_policy;
  result.m2l_parallel_policy = policy.m2l_parallel_policy;
  result.m2l_parallel_workers = policy.m2l_parallel_workers;
  result.tree_stop_policy = policy.tree_stop_policy;
  result.jcp_channel_policy = policy.jcp_channel_policy;
  result.tree_panel_radius_factor = policy.tree_panel_radius_factor;
  result.tree_occupancy_stop = policy.tree_occupancy_stop;
  result.tree_node_count = tree.stats.node_count;
  result.tree_leaf_count = tree.stats.leaf_count;
  result.tree_max_depth_observed = tree.stats.max_observed_depth;
  result.tree_radius_stop_leaf_count = tree.stats.radius_stop_leaf_count;
  result.tree_occupancy_stop_leaf_count =
      tree.stats.occupancy_stop_leaf_count;
  result.tree_leaf_size_stop_count = tree.stats.leaf_size_stop_count;
  result.tree_max_depth_stop_count = tree.stats.max_depth_stop_count;
  result.tree_unsplittable_stop_count = tree.stats.unsplittable_stop_count;
  result.tree_min_leaf_half_width_over_panel_radius =
      tree.stats.min_leaf_half_width_over_panel_radius;
  result.tree_max_leaf_panel_radius = tree.stats.max_leaf_panel_radius;
  result.tree_mean_leaf_source_count = tree.stats.mean_leaf_source_count;
  result.tree_mean_leaf_target_count = tree.stats.mean_leaf_target_count;
}

std::string classify_hierarchy_execution_gap(
    const std::string &mode, const KifmmHierarchyJcpResult &hierarchy) {
  if (mode == "hybrid-source-local-bridge") {
    if (hierarchy.status == "p2m_field_solve_failed") {
      return "hybrid_source_m2e_failed";
    }
    if (hierarchy.status == "m2m_system_gap" ||
        hierarchy.status == "m2m_field_solve_failed") {
      return "hybrid_source_m2m_failed";
    }
    if (hierarchy.status == "m2l_missing_upward" ||
        hierarchy.status == "m2l_local_system_gap" ||
        hierarchy.status == "m2l_dense_local_system_gap") {
      return "hybrid_source_m2l_failed";
    }
    if (hierarchy.status == "local_field_solve_failed" ||
        hierarchy.status == "l2l_local_system_gap") {
      return "hybrid_source_local_system_gap";
    }
    if (hierarchy.status == "hybrid_e2l_failed") {
      return "hybrid_source_e2l_failed";
    }
    return "hybrid_source_execution_gap";
  }
  if (mode == "hybrid-local-bridge" &&
      hierarchy.status == "hybrid_e2l_failed") {
    return "hybrid_local_e2l_failed";
  }
  return "hierarchy_execution_gap";
}

std::uint64_t hash_channel_rows(
    std::span<const KifmmJcpChannelRow> rows) noexcept {
  std::uint64_t h = 0xE255C4A11E10CA1ull;
  h = mix_hash(h, static_cast<std::uint64_t>(rows.size()));
  for (const KifmmJcpChannelRow &row : rows) {
    h = hash_double(h, row.k1);
    h = hash_double(h, row.k2);
    h = hash_double(h, row.k3);
    h = hash_double(h, row.k4);
  }
  return h;
}

bool finite_channels(std::span<const KifmmJcpChannelRow> rows) noexcept {
  for (const KifmmJcpChannelRow &row : rows) {
    if (!std::isfinite(row.k1) || !std::isfinite(row.k2) ||
        !std::isfinite(row.k3) || !std::isfinite(row.k4)) {
      return false;
    }
  }
  return true;
}

std::vector<titan::tabi::fmm_m2l::PrototypePanel>
to_prototype_panels(std::span<const Panel> panels) {
  std::vector<titan::tabi::fmm_m2l::PrototypePanel> out;
  out.reserve(panels.size());
  for (const Panel &panel : panels) {
    titan::tabi::fmm_m2l::PrototypePanel converted;
    converted.x = panel.x;
    converted.y = panel.y;
    converted.z = panel.z;
    converted.nx = panel.nx;
    converted.ny = panel.ny;
    converted.nz = panel.nz;
    converted.area = panel.area;
    converted.scalar_density = 0.0;
    out.push_back(converted);
  }
  return out;
}

std::vector<std::size_t> source_indices_as_size_t(
    const KifmmHierarchyNode &node) {
  std::vector<std::size_t> indices;
  indices.reserve(node.source_indices.size());
  for (const int index : node.source_indices) {
    indices.push_back(static_cast<std::size_t>(index));
  }
  return indices;
}

KifmmJcpChannelRow production_single_pair_channel(
    const Panel &target, std::size_t target_index, const Panel &source,
    std::size_t source_index, double phi, double phin, double eps_ratio,
    double kappa, const std::string &near_integration,
    const std::string &quad_buffer, std::uint64_t &valid_pairs,
    std::uint64_t &invalid_pairs) {
  KifmmJcpChannelRow row;
  const JcpNearPairKernel kernel =
      TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
          target, target_index, source, source_index, eps_ratio, kappa, true,
          near_integration, quad_buffer, nullptr);
  if (!kernel.valid) {
    ++invalid_pairs;
    return row;
  }
  ++valid_pairs;
  row.k1 = kernel.k1 * phin;
  row.k2 = kernel.k2 * phi;
  row.k3 = kernel.k3 * phin;
  row.k4 = kernel.k4 * phi;
  return row;
}

void add_channel(KifmmJcpChannelRow &dst,
                 const KifmmJcpChannelRow &src) noexcept {
  dst.k1 += src.k1;
  dst.k2 += src.k2;
  dst.k3 += src.k3;
  dst.k4 += src.k4;
}

double channel_value(const KifmmJcpChannelRow &row, int channel) noexcept {
  if (channel == 0) {
    return row.k1;
  }
  if (channel == 1) {
    return row.k2;
  }
  if (channel == 2) {
    return row.k3;
  }
  return row.k4;
}

double channel_relative_rms(std::span<const KifmmJcpChannelRow> approx,
                            std::span<const KifmmJcpChannelRow> direct,
                            int channel) noexcept {
  double err = 0.0;
  double ref = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double a = channel_value(approx[i], channel);
    const double d = channel_value(direct[i], channel);
    const double delta = a - d;
    err += delta * delta;
    ref += d * d;
  }
  return std::sqrt(err / std::max(ref, 1.0e-300));
}

double aggregate_relative_rms(std::span<const KifmmJcpChannelRow> approx,
                              std::span<const KifmmJcpChannelRow> direct)
    noexcept {
  double err = 0.0;
  double ref = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double a_phi = approx[i].k1 + approx[i].k2;
    const double a_phin = approx[i].k3 + approx[i].k4;
    const double d_phi = direct[i].k1 + direct[i].k2;
    const double d_phin = direct[i].k3 + direct[i].k4;
    const double d0 = a_phi - d_phi;
    const double d1 = a_phin - d_phin;
    err += d0 * d0 + d1 * d1;
    ref += d_phi * d_phi + d_phin * d_phin;
  }
  return std::sqrt(err / std::max(ref, 1.0e-300));
}

double channel_abs_rms(std::span<const KifmmJcpChannelRow> approx,
                       std::span<const KifmmJcpChannelRow> direct,
                       int channel) noexcept {
  double err = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double delta =
        channel_value(approx[i], channel) - channel_value(direct[i], channel);
    err += delta * delta;
  }
  return std::sqrt(err / std::max<std::size_t>(approx.size(), 1));
}

double channel_ref_rms(std::span<const KifmmJcpChannelRow> direct,
                       int channel) noexcept {
  double ref = 0.0;
  for (const KifmmJcpChannelRow &row : direct) {
    const double value = channel_value(row, channel);
    ref += value * value;
  }
  return std::sqrt(ref / std::max<std::size_t>(direct.size(), 1));
}

void set_accuracy(KifmmHierarchyRealPanelResult &result) {
  result.aggregate_relative_rms =
      aggregate_relative_rms(result.hierarchy.approximate_rows,
                             result.hierarchy.direct_rows);
  result.k1_relative_rms =
      channel_relative_rms(result.hierarchy.approximate_rows,
                           result.hierarchy.direct_rows, 0);
  result.k2_relative_rms =
      channel_relative_rms(result.hierarchy.approximate_rows,
                           result.hierarchy.direct_rows, 1);
  result.k3_relative_rms =
      channel_relative_rms(result.hierarchy.approximate_rows,
                           result.hierarchy.direct_rows, 2);
  result.k4_relative_rms =
      channel_relative_rms(result.hierarchy.approximate_rows,
                           result.hierarchy.direct_rows, 3);
  result.max_channel_relative_rms = result.k1_relative_rms;
  result.max_channel_name = "K1";
  if (result.k2_relative_rms > result.max_channel_relative_rms) {
    result.max_channel_relative_rms = result.k2_relative_rms;
    result.max_channel_name = "K2";
  }
  if (result.k3_relative_rms > result.max_channel_relative_rms) {
    result.max_channel_relative_rms = result.k3_relative_rms;
    result.max_channel_name = "K3";
  }
  if (result.k4_relative_rms > result.max_channel_relative_rms) {
    result.max_channel_relative_rms = result.k4_relative_rms;
    result.max_channel_name = "K4";
  }
  result.k4_abs_rms =
      channel_abs_rms(result.hierarchy.approximate_rows,
                      result.hierarchy.direct_rows, 3);
  result.k4_reference_rms = channel_ref_rms(result.hierarchy.direct_rows, 3);

  const bool aggregate_pass =
      result.aggregate_relative_rms <= result.aggregate_relative_rms_gate;
  const bool channel_pass =
      result.max_channel_relative_rms <= result.max_channel_relative_rms_gate;
  result.accuracy_status = aggregate_pass && channel_pass ? "pass" : "fail";
  if (!aggregate_pass && !channel_pass) {
    result.accuracy_failure = "aggregate_and_channel";
  } else if (!aggregate_pass) {
    result.accuracy_failure = "aggregate";
  } else if (!channel_pass) {
    result.accuracy_failure = "channel";
  } else {
    result.accuracy_failure = "none";
  }
}

void set_hierarchy_k123_telemetry(KifmmHierarchyRealPanelResult &result) {
  result.hierarchy_k123_max_channel_relative_rms = result.k1_relative_rms;
  result.hierarchy_k123_max_channel_name = "K1";
  if (result.k2_relative_rms >
      result.hierarchy_k123_max_channel_relative_rms) {
    result.hierarchy_k123_max_channel_relative_rms = result.k2_relative_rms;
    result.hierarchy_k123_max_channel_name = "K2";
  }
  if (result.k3_relative_rms >
      result.hierarchy_k123_max_channel_relative_rms) {
    result.hierarchy_k123_max_channel_relative_rms = result.k3_relative_rms;
    result.hierarchy_k123_max_channel_name = "K3";
  }
}

void copy_cartesian_k4_telemetry(
    KifmmHierarchyRealPanelResult &result,
    const CartesianSampledPanelMatvecResult &cartesian) {
  result.cartesian_k4_status = cartesian.status;
  result.cartesian_k4_execution_status = cartesian.execution_status;
  result.cartesian_k4_relative_rms = cartesian.k4_relative_rms;
  result.cartesian_k4_abs_rms = cartesian.k4_abs_rms;
  result.cartesian_k4_reference_rms = cartesian.k4_reference_rms;
  const bool k4_pass = cartesian.execution_status == "pass" &&
                       cartesian.k4_relative_rms <=
                           result.max_channel_relative_rms_gate;
  result.cartesian_k4_accuracy_status = k4_pass ? "pass" : "fail";
  result.cartesian_k4_accuracy_failure = k4_pass ? "none" : "k4_channel";
  result.cartesian_k4_far_panel_pairs =
      static_cast<std::uint64_t>(cartesian.far_panel_pairs);
  result.cartesian_k4_fallback_panel_pairs =
      static_cast<std::uint64_t>(cartesian.fallback_panel_pairs);
  result.cartesian_k4_total_seconds = cartesian.total_seconds;
  result.cartesian_k4_policy_hash = cartesian.policy_hash;
}

CartesianSampledPanelMatvecConfig e29_cartesian_k4_config(
    const KifmmHierarchyJcpConfig &config) {
  CartesianSampledPanelMatvecConfig cartesian;
  cartesian.order = 5;
  cartesian.theta = 0.6;
  cartesian.tail_tol = 0.002;
  cartesian.strict_geometry_theta = 0.4;
  cartesian.eps_ratio = config.eps_ratio;
  cartesian.kappa = config.kappa;
  cartesian.near_integration = "analytic_singular";
  cartesian.quad_buffer = "off";
  cartesian.fallback_kernel = "packed-avx2";
  cartesian.refine_mode = "operator-tail-static";
  cartesian.cache_mode = "green-jet-compressed";
  cartesian.plan_builder = "metric-cache";
  cartesian.cache_budget_mb = 256;
  cartesian.max_aggregate_relative_rms = config.aggregate_relative_rms_gate;
  cartesian.max_channel_relative_rms = config.max_channel_relative_rms_gate;
  return cartesian;
}

void apply_channel_hybrid_rows(
    KifmmHierarchyRealPanelResult &result,
    const CartesianSampledPanelMatvecResult &cartesian,
    bool k4_only_diagnostic) {
  std::vector<KifmmJcpChannelRow> recombined =
      k4_only_diagnostic ? result.hierarchy.direct_rows
                         : result.hierarchy.approximate_rows;
  if (recombined.size() != cartesian.approximate_rows.size() ||
      result.hierarchy.direct_rows.size() != cartesian.direct_rows.size()) {
    result.status = "channel_hybrid_row_count_gap";
    result.execution_status = "fail";
    result.accuracy_status = "fail";
    result.accuracy_failure = "row_count";
    result.channel_hybrid_double_count_guard = "fail";
    return;
  }
  for (std::size_t i = 0; i < recombined.size(); ++i) {
    recombined[i].k4 = cartesian.approximate_rows[i].k4;
  }
  result.hierarchy.approximate_rows = std::move(recombined);
  const std::string hierarchy_source =
      "hierarchy_q" + std::to_string(result.order);
  result.k1_source = k4_only_diagnostic ? "dense_direct" : hierarchy_source;
  result.k2_source = k4_only_diagnostic ? "dense_direct" : hierarchy_source;
  result.k3_source = k4_only_diagnostic ? "dense_direct" : hierarchy_source;
  result.k4_source = "cartesian_p5_geometry0p4";
  result.hybrid_mode = k4_only_diagnostic
                           ? "cartesian-p5-geometry0p4-k4-only"
                           : "channel-hybrid-k123-hierarchy-k4-cartesian";
  result.k4_decomposition_status = "channel_hybrid_k4_cartesian";
  result.l2p_gradient_method = k4_only_diagnostic
                                    ? "cartesian_k4_only"
                                    : "hierarchy_k123_cartesian_k4";
  result.channel_hybrid_double_count_guard =
      k4_only_diagnostic ? "not_applicable" : "pass";
  result.output_hash = hash_channel_rows(result.hierarchy.approximate_rows);
  result.channel_hybrid_recombined_hash = result.output_hash;
  set_accuracy(result);
  set_hierarchy_k123_telemetry(result);
  result.hybrid_recombined_aggregate_relative_rms =
      result.aggregate_relative_rms;
  result.hybrid_recombined_max_channel_relative_rms =
      result.max_channel_relative_rms;
  result.hybrid_recombined_max_channel_name = result.max_channel_name;
  result.status = "ok";
  result.execution_status = "pass";
}

void annotate_dual_worklist_channel_hybrid(
    KifmmHierarchyRealPanelResult &result) {
  if (result.execution_status == "fail") {
    result.dual_worklist_status = "fail";
    result.channel_recombination_status = "fail";
    result.worklist_independence_status = "expected";
    return;
  }
  const std::string hierarchy_source =
      "kifmm_hierarchy_q" + std::to_string(result.order);
  result.k1_source = hierarchy_source;
  result.k2_source = hierarchy_source;
  result.k3_source = hierarchy_source;
  result.k4_source = "cartesian_p5_geometry0p4";
  result.hybrid_mode =
      "dual-worklist-channel-hybrid-k123-kifmm-k4-cartesian";
  result.k4_decomposition_status = "dual_worklist_cartesian_k4";
  result.l2p_gradient_method = "kifmm_k123_cartesian_k4_dual_worklist";
  result.kifmm_worklist_hash = result.worklist_hash;
  result.kifmm_far_pairs = result.far_panel_pairs;
  result.kifmm_direct_pairs = result.direct_panel_pairs;
  result.kifmm_far_fraction = result.far_fraction;
  result.dual_worklist_status = "pass";
  result.channel_recombination_status =
      result.channel_hybrid_double_count_guard == "pass" ? "pass" : "fail";
  result.worklist_independence_status = "expected";
}

void apply_shared_worklist_channel_hybrid(
    KifmmHierarchyRealPanelResult &result, std::span<const Panel> panels,
    std::span<const double> unknowns,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmHierarchyTree &tree, const KifmmHierarchyWorklist &worklist,
    const KifmmHierarchyJcpConfig &config) {
  result.shared_worklist_hash = result.worklist_hash;
  result.shared_theta = config.hierarchy.theta;
  result.shared_guard = config.real_panel_guard;
  result.shared_far_pairs = result.far_panel_pairs;
  result.shared_direct_pairs = result.direct_panel_pairs;
  result.shared_far_fraction = result.far_fraction;
  result.worklist_consistency_status = "not_evaluated";

  if (result.hierarchy.approximate_rows.size() !=
          sampled_target_indices.size() ||
      result.hierarchy.direct_rows.size() != sampled_target_indices.size()) {
    result.status = "shared_worklist_row_count_gap";
    result.execution_status = "fail";
    result.accuracy_status = "fail";
    result.accuracy_failure = "row_count";
    result.worklist_consistency_status = "row_count_gap";
    return;
  }

  std::vector<KifmmJcpChannelRow> recombined =
      result.hierarchy.approximate_rows;
  std::vector<KifmmJcpChannelRow> shared_k4(sampled_target_indices.size());
  const std::vector<titan::tabi::fmm_m2l::PrototypePanel> prototype_panels =
      to_prototype_panels(panels);
  std::vector<double> phi(panels.size());
  std::vector<double> phin(panels.size());
  for (std::size_t i = 0; i < panels.size(); ++i) {
    phi[i] = unknowns[i];
    phin[i] = unknowns[panels.size() + i];
  }

  std::uint64_t far_pairs = 0;
  try {
    for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
      const KifmmHierarchyNode &target_node =
          tree.nodes[static_cast<std::size_t>(target_node_id)];
      const KifmmHierarchyNode &source_node =
          tree.nodes[static_cast<std::size_t>(source_node_id)];
      const std::vector<std::size_t> source_indices =
          source_indices_as_size_t(source_node);
      const titan::tabi::fmm_m2l::CartesianJcpMoments moments =
          titan::tabi::fmm_m2l::build_cartesian_jcp_moments(
              prototype_panels, phi, phin, source_indices,
              source_node.center.x, source_node.center.y,
              source_node.center.z, 5);
      for (const int target_local_int : target_node.target_indices) {
        const std::size_t target_local =
            static_cast<std::size_t>(target_local_int);
        const std::size_t target_panel_index =
            sampled_target_indices[target_local];
        const std::array<double, 4> channels =
            titan::tabi::fmm_m2l::evaluate_cartesian_jcp_m2p(
                prototype_panels[target_panel_index], source_node.center.x,
                source_node.center.y, source_node.center.z, moments,
                config.eps_ratio, config.kappa);
        shared_k4[target_local].k4 += channels[3];
        far_pairs += static_cast<std::uint64_t>(
            source_node.source_indices.size());
      }
    }
  } catch (const std::exception &) {
    result.status = "cartesian_k4_shared_far_gap";
    result.execution_status = "fail";
    result.accuracy_status = "fail";
    result.accuracy_failure = "cartesian_k4_shared_far";
    result.worklist_consistency_status = "cartesian_k4_far_gap";
    return;
  }

  std::uint64_t direct_pairs = 0;
  std::uint64_t invalid_pairs = 0;
  for (const auto &[target_node_id, source_node_id] :
       worklist.direct_leaf_pairs) {
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    for (const int target_local_int : target_node.target_indices) {
      const std::size_t target_local =
          static_cast<std::size_t>(target_local_int);
      const std::size_t target_panel_index =
          sampled_target_indices[target_local];
      for (const int source_int : source_node.source_indices) {
        const std::size_t source_index = static_cast<std::size_t>(source_int);
        KifmmJcpChannelRow contribution = production_single_pair_channel(
            panels[target_panel_index], target_panel_index,
            panels[source_index], source_index, unknowns[source_index],
            unknowns[panels.size() + source_index], result.eps_ratio,
            result.kappa, result.near_integration, result.quad_buffer,
            direct_pairs, invalid_pairs);
        shared_k4[target_local].k4 += contribution.k4;
      }
    }
  }

  result.cartesian_k4_shared_far_pairs = far_pairs;
  result.cartesian_k4_shared_direct_pairs = direct_pairs;
  result.direct_avx2_all_channel_pairs = direct_pairs;
  result.cartesian_k4_far_panel_pairs = far_pairs;
  result.cartesian_k4_fallback_panel_pairs = direct_pairs;
  result.cartesian_k4_policy_hash =
      mix_hash(result.policy_hash, 0xE30CA7E514A11C4ull);
  result.cartesian_k4_status = "ok";
  result.cartesian_k4_execution_status = "pass";
  result.cartesian_k4_total_seconds = 0.0;

  if (far_pairs != result.shared_far_pairs ||
      direct_pairs != result.shared_direct_pairs || invalid_pairs != 0 ||
      result.exact_pair_coverage !=
          result.shared_far_pairs + result.shared_direct_pairs) {
    result.status = "shared_worklist_consistency_gap";
    result.execution_status = "fail";
    result.accuracy_status = "fail";
    result.accuracy_failure = "shared_worklist_consistency";
    result.worklist_consistency_status = "coverage_gap";
    result.production_invalid_pairs += invalid_pairs;
    return;
  }

  for (std::size_t i = 0; i < recombined.size(); ++i) {
    recombined[i].k4 = shared_k4[i].k4;
  }
  result.hierarchy.approximate_rows = std::move(recombined);
  result.k1_source = "hierarchy_q8_shared_worklist";
  result.k2_source = "hierarchy_q8_shared_worklist";
  result.k3_source = "hierarchy_q8_shared_worklist";
  result.k4_source = "cartesian_p5_shared_worklist";
  result.hybrid_mode =
      "channel-hybrid-shared-worklist-k123-hierarchy-k4-cartesian";
  result.k4_decomposition_status = "shared_worklist_cartesian_k4";
  result.l2p_gradient_method = "hierarchy_k123_cartesian_k4_shared_worklist";
  result.channel_hybrid_double_count_guard = "pass";
  result.worklist_consistency_status = "pass";
  result.output_hash = hash_channel_rows(result.hierarchy.approximate_rows);
  result.channel_hybrid_recombined_hash = result.output_hash;
  set_accuracy(result);
  set_hierarchy_k123_telemetry(result);
  result.hybrid_recombined_aggregate_relative_rms =
      result.aggregate_relative_rms;
  result.hybrid_recombined_max_channel_relative_rms =
      result.max_channel_relative_rms;
  result.hybrid_recombined_max_channel_name = result.max_channel_name;
  result.cartesian_k4_relative_rms = result.k4_relative_rms;
  result.cartesian_k4_abs_rms = result.k4_abs_rms;
  result.cartesian_k4_reference_rms = result.k4_reference_rms;
  const bool k4_pass =
      result.cartesian_k4_relative_rms <= result.max_channel_relative_rms_gate;
  result.cartesian_k4_accuracy_status = k4_pass ? "pass" : "fail";
  result.cartesian_k4_accuracy_failure = k4_pass ? "none" : "k4_channel";
  result.status = "ok";
  result.execution_status = "pass";
}

} // namespace

KifmmHierarchyRealPanelResult run_kifmm_hierarchy_real_panel_sampled(
    std::span<const Panel> panels, std::span<const double> unknowns,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmHierarchyJcpConfig &config) {
  const auto total_start = Clock::now();
  KifmmHierarchyRealPanelResult result;
  result.execution_status = "fail";
  result.source_panels = panels.size();
  result.target_panels = sampled_target_indices.size();
  result.sampled_rows = sampled_target_indices.size();
  result.full_coverage_expected =
      static_cast<std::uint64_t>(panels.size()) *
      static_cast<std::uint64_t>(sampled_target_indices.size());
  result.sampled_coverage_expected = result.full_coverage_expected;
  result.timing_scope = "sampled_rows_only";
  result.order = config.hierarchy.order;
  result.theta = config.hierarchy.theta;
  result.eps_ratio = config.eps_ratio;
  result.kappa = config.kappa;
  result.condition_policy = config.hierarchy.condition_policy;
  result.source_box_policy = config.hierarchy.source_box_policy;
  result.m2l_eval_policy = config.hierarchy.m2l_eval_policy;
  result.m2l_matrix_cache_policy = config.hierarchy.m2l_matrix_cache_policy;
  result.m2l_gemm_batch_policy = config.hierarchy.m2l_gemm_batch_policy;
  result.m2l_parallel_policy = config.hierarchy.m2l_parallel_policy;
  result.m2l_parallel_workers = config.hierarchy.m2l_parallel_workers;
  result.tree_stop_policy = config.hierarchy.tree_stop_policy;
  result.jcp_channel_policy = config.hierarchy.jcp_channel_policy;
  result.tree_panel_radius_factor =
      config.hierarchy.tree_panel_radius_factor;
  result.tree_occupancy_stop = config.hierarchy.tree_occupancy_stop;
  result.real_panel_guard = config.real_panel_guard;
  result.hier_debug_mode =
      config.hier_debug_mode.empty() ? std::string("full")
                                     : config.hier_debug_mode;
  const bool e29_cartesian_k4_only =
      result.hier_debug_mode == "cartesian-p5-geometry0p4-k4-only";
  const bool e29_channel_hybrid =
      result.hier_debug_mode ==
      "channel-hybrid-k123-hierarchy-k4-cartesian";
  const bool e30_shared_channel_hybrid =
      result.hier_debug_mode ==
      "channel-hybrid-shared-worklist-k123-hierarchy-k4-cartesian";
  const bool e31_dual_channel_hybrid =
      result.hier_debug_mode ==
      "dual-worklist-channel-hybrid-k123-kifmm-k4-cartesian";
  result.real_panel_guard_margin_factor =
      config.real_panel_guard_margin_factor;
  result.aggregate_relative_rms_gate = config.aggregate_relative_rms_gate;
  result.max_channel_relative_rms_gate =
      config.max_channel_relative_rms_gate;

  if (panels.empty() || sampled_target_indices.empty()) {
    result.status = "invalid_input_size";
    return result;
  }
  if (unknowns.size() != 2 * panels.size()) {
    result.status = "invalid_unknown_size";
    return result;
  }
  if (!std::isfinite(config.eps_ratio) || config.eps_ratio == 0.0 ||
      !std::isfinite(config.kappa) || config.kappa < 0.0) {
    result.status = "invalid_physics";
    return result;
  }
  if (config.real_panel_guard != "off" &&
      config.real_panel_guard != "strict" &&
      config.real_panel_guard != "strict-local-check") {
    result.status = "invalid_real_panel_guard";
    return result;
  }
  if (result.hier_debug_mode != "full" &&
      result.hier_debug_mode != "m2l-direct-to-targets" &&
      result.hier_debug_mode != "no-m2m" &&
      result.hier_debug_mode != "no-l2l" &&
      result.hier_debug_mode != "local-check-oracle" &&
      result.hier_debug_mode != "l2p-oracle" &&
      result.hier_debug_mode != "hybrid-local-bridge" &&
      result.hier_debug_mode != "hybrid-local-split-bridge" &&
      result.hier_debug_mode != "hybrid-source-local-bridge" &&
      !e29_cartesian_k4_only && !e29_channel_hybrid &&
      !e30_shared_channel_hybrid && !e31_dual_channel_hybrid) {
    result.status = "invalid_hier_debug_mode";
    return result;
  }
  if (!std::isfinite(config.real_panel_guard_margin_factor) ||
      config.real_panel_guard_margin_factor < 0.0) {
    result.status = "invalid_real_panel_guard_margin";
    return result;
  }
  std::set<std::size_t> unique_targets;
  for (const std::size_t index : sampled_target_indices) {
    if (index >= panels.size() || !unique_targets.insert(index).second) {
      result.status = "invalid_target_indices";
      return result;
    }
  }
  for (double value : unknowns) {
    if (!std::isfinite(value)) {
      result.status = "invalid_unknown";
      return result;
    }
  }
  for (const Panel &panel : panels) {
    if (!valid_panel(panel)) {
      result.status = "invalid_panel_geometry";
      return result;
    }
  }

  result.panel_geometry_hash = hash_panels(panels);
  result.unknown_vector_hash = hash_doubles(unknowns);
  result.sampled_target_hash = hash_indices(sampled_target_indices);

  std::vector<KifmmPanelSource> sources;
  sources.reserve(panels.size());
  for (std::size_t i = 0; i < panels.size(); ++i) {
    sources.push_back(source_from_panel(panels[i], unknowns[i],
                                        unknowns[panels.size() + i]));
  }
  std::vector<Vec3> targets;
  std::vector<Vec3> target_normals;
  targets.reserve(sampled_target_indices.size());
  target_normals.reserve(sampled_target_indices.size());
  for (const std::size_t target_index : sampled_target_indices) {
    targets.push_back(position_of(panels[target_index]));
    target_normals.push_back(normal_of(panels[target_index]));
  }

  const std::vector<Vec3> source_positions = positions_from_sources(sources);
  const std::vector<double> source_radii = panel_radii(panels);
  const std::vector<double> target_radii =
      sampled_panel_radii(panels, sampled_target_indices);
  const KifmmHierarchyTree tree =
      build_strict_octree(source_positions, targets, source_radii,
                          target_radii, config.hierarchy);
  result.tree_hash = tree.tree_hash;
  if (tree.status != "ok") {
    result.status = tree.status;
    return result;
  }
  copy_tree_policy_and_stats(result, config.hierarchy, tree);
  const std::vector<KifmmRealPanelNodeEnvelope> source_envelopes =
      build_source_envelopes(tree, panels);
  const std::vector<KifmmRealPanelNodeEnvelope> target_envelopes =
      build_target_envelopes(tree, panels, sampled_target_indices);
  update_leaf_panel_telemetry(tree, source_envelopes, target_envelopes,
                              result);
  update_l2l_containment_telemetry(tree, config.hierarchy, result);
  const KifmmHierarchyWorklist worklist =
      (config.real_panel_guard == "strict" ||
       config.real_panel_guard == "strict-local-check")
          ? build_real_panel_guarded_worklist(
                tree, config.hierarchy, source_envelopes, target_envelopes,
                config.real_panel_guard_margin_factor,
                config.real_panel_guard == "strict-local-check", result)
          : build_hierarchy_worklist(tree, config.hierarchy);
  result.worklist_hash = worklist.worklist_hash;
  result.policy_hash = worklist.policy_hash;
  if (worklist.worklist_hash == 0) {
    result.status = "worklist_gap";
    return result;
  }

  const auto hierarchy_start = Clock::now();
  KifmmHierarchyJcpConfig hierarchy_config = config;
  if (e29_cartesian_k4_only || e29_channel_hybrid ||
      e30_shared_channel_hybrid || e31_dual_channel_hybrid) {
    hierarchy_config.hier_debug_mode = "full";
  }
  hierarchy_config.build_dense_oracle = true;
  hierarchy_config.include_direct_fallback = false;
  result.hierarchy = run_full_hierarchical_jcp_kifmm_with_worklist(
      sources, targets, target_normals, hierarchy_config, tree, worklist);
  result.hierarchy_seconds = elapsed_seconds(hierarchy_start, Clock::now());
  result.apply_hierarchy_seconds = result.hierarchy_seconds;
  record_apply_memory_after_hierarchy(result);
  copy_hierarchy_closeout_fields(result);
  if (result.hierarchy.execution_status != "pass" ||
      result.hierarchy.status != "ok") {
    result.status =
        classify_hierarchy_execution_gap(result.hier_debug_mode,
                                         result.hierarchy);
    return result;
  }
  if (result.hierarchy.worklist_hash != result.worklist_hash ||
      result.hierarchy.policy_hash != result.policy_hash ||
      result.hierarchy.tree_hash != result.tree_hash) {
    result.status = "hierarchy_hash_gap";
    return result;
  }
  result.local_check_rhs_relative_rms =
      result.hierarchy.local_check_rhs_relative_rms;
  result.local_check_rhs_max_abs_error =
      result.hierarchy.local_check_rhs_max_abs_error;
  result.l2p_oracle_aggregate_relative_rms =
      result.hierarchy.l2p_oracle_aggregate_relative_rms;
  result.l2p_oracle_k4_relative_rms =
      result.hierarchy.l2p_oracle_k4_relative_rms;
  result.k4_laplace_grad_relative_rms =
      result.hierarchy.k4_laplace_grad_relative_rms;
  result.k4_yukawa_grad_relative_rms =
      result.hierarchy.k4_yukawa_grad_relative_rms;
  result.k4_combined_relative_rms =
      result.hierarchy.k4_combined_relative_rms;
  result.field_attribution = result.hierarchy.field_attribution;
  result.flat = run_flat_jcp_kifmm_with_worklist(sources, targets,
                                                 target_normals,
                                                 hierarchy_config, tree,
                                                 worklist);
  if (result.flat.execution_status != "pass" ||
      result.flat.status != "ok") {
    result.status = "flat_execution_gap";
    return result;
  }
  if (result.flat.worklist_hash != result.worklist_hash ||
      result.flat.policy_hash != result.policy_hash ||
      result.flat.tree_hash != result.tree_hash) {
    result.status = "flat_hash_gap";
    return result;
  }

  std::vector<KifmmJcpChannelRow> approx = result.hierarchy.approximate_rows;
  std::vector<KifmmJcpChannelRow> flat_approx = result.flat.approximate_rows;
  std::vector<KifmmJcpChannelRow> far_approx = approx;
  std::vector<KifmmJcpChannelRow> far_direct(sampled_target_indices.size());
  std::vector<KifmmJcpChannelRow> fallback_direct(
      sampled_target_indices.size());
  std::vector<KifmmJcpChannelRow> direct(sampled_target_indices.size());
  if (approx.size() != sampled_target_indices.size() ||
      flat_approx.size() != sampled_target_indices.size()) {
    result.status = "hierarchy_row_count_gap";
    return result;
  }

  result.far_panel_pairs = result.hierarchy.stats.far_panel_pairs;
  result.direct_panel_pairs = result.hierarchy.stats.direct_panel_pairs;
  result.exact_pair_coverage =
      result.far_panel_pairs + result.direct_panel_pairs;
  const std::uint64_t expected_pairs =
      static_cast<std::uint64_t>(panels.size()) *
      static_cast<std::uint64_t>(sampled_target_indices.size());
  result.far_fraction =
      expected_pairs == 0 ? 0.0
                          : static_cast<double>(result.far_panel_pairs) /
                                static_cast<double>(expected_pairs);
  if (result.exact_pair_coverage != expected_pairs) {
    result.status = "coverage_gap";
    return result;
  }

  const auto guard_audit_start = Clock::now();
  result.source_box_boundary_margin_min =
      std::numeric_limits<double>::infinity();
  result.target_check_excess_min = std::numeric_limits<double>::infinity();
  result.target_panel_envelope_excess_min =
      std::numeric_limits<double>::infinity();
  result.target_local_check_excess_min =
      std::numeric_limits<double>::infinity();
  result.mac_bin_min_target_check_excess.fill(
      std::numeric_limits<double>::infinity());

  std::uint64_t far_invalid_pairs = 0;
  for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const double distance = norm(sub(target_node.center, source_node.center));
    const double ratio =
        distance > 0.0 ? (std::sqrt(3.0) * source_node.half_width) / distance
                       : std::numeric_limits<double>::infinity();
    const int bin = mac_bin(ratio);
    ++result.mac_bin_cell_pairs[static_cast<std::size_t>(bin)];
    result.mac_bin_panel_pairs[static_cast<std::size_t>(bin)] +=
        static_cast<std::uint64_t>(target_node.target_indices.size()) *
        static_cast<std::uint64_t>(source_node.source_indices.size());
    for (const int source_int : source_node.source_indices) {
      const std::size_t source_index = static_cast<std::size_t>(source_int);
      result.source_box_boundary_margin_min = std::min(
          result.source_box_boundary_margin_min,
          source_boundary_margin(sources[source_index], source_node));
    }
    for (const int target_local_int : target_node.target_indices) {
      const std::size_t target_local =
          static_cast<std::size_t>(target_local_int);
      const std::size_t target_panel_index =
          sampled_target_indices[target_local];
      const double check_excess = target_vertex_check_excess(
          panels[target_panel_index], source_node,
          config.hierarchy.upward_check_scale);
      const double local_check_excess =
          target_local_check_excess(source_node, target_node, config.hierarchy);
      result.target_check_excess_min =
          std::min(result.target_check_excess_min, check_excess);
      result.target_panel_envelope_excess_min =
          std::min(result.target_panel_envelope_excess_min, check_excess);
      result.target_local_check_excess_min =
          std::min(result.target_local_check_excess_min, local_check_excess);
      result.mac_bin_min_target_check_excess[static_cast<std::size_t>(bin)] =
          std::min(result.mac_bin_min_target_check_excess
                       [static_cast<std::size_t>(bin)],
                   check_excess);
      if (check_excess < 0.0) {
        const auto source_panel_pairs =
            static_cast<std::uint64_t>(source_node.source_indices.size());
        result.negative_target_check_excess_pairs += source_panel_pairs;
        result.negative_target_check_excess_panel_pairs += source_panel_pairs;
      }
      if (local_check_excess < 0.0) {
        const auto source_panel_pairs =
            static_cast<std::uint64_t>(source_node.source_indices.size());
        result.negative_target_local_check_excess_pairs += source_panel_pairs;
        result.negative_target_local_check_excess_panel_pairs +=
            source_panel_pairs;
      }
      for (const int source_int : source_node.source_indices) {
        const std::size_t source_index = static_cast<std::size_t>(source_int);
        KifmmJcpChannelRow contribution = production_single_pair_channel(
            panels[target_panel_index], target_panel_index,
            panels[source_index], source_index, unknowns[source_index],
            unknowns[panels.size() + source_index], result.eps_ratio,
            result.kappa, result.near_integration, result.quad_buffer,
            result.far_direct_valid_pairs, far_invalid_pairs);
        add_channel(far_direct[target_local], contribution);
      }
    }
  }
  if (result.far_direct_valid_pairs != result.far_panel_pairs ||
      far_invalid_pairs != 0) {
    result.status = "production_far_coverage_gap";
    result.production_invalid_pairs += far_invalid_pairs;
    return result;
  }
  if (!std::isfinite(result.source_box_boundary_margin_min)) {
    result.source_box_boundary_margin_min = std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(result.target_check_excess_min)) {
    result.target_check_excess_min = std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(result.target_panel_envelope_excess_min)) {
    result.target_panel_envelope_excess_min =
        std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(result.target_local_check_excess_min)) {
    result.target_local_check_excess_min =
        std::numeric_limits<double>::quiet_NaN();
  }
  for (double &value : result.mac_bin_min_target_check_excess) {
    if (!std::isfinite(value)) {
      value = std::numeric_limits<double>::quiet_NaN();
    }
  }
  result.apply_guard_audit_seconds =
      elapsed_seconds(guard_audit_start, Clock::now());
  record_apply_memory_after_guard_audit(result);

  const auto fallback_start = Clock::now();
  for (const auto &[target_node_id, source_node_id] :
       worklist.direct_leaf_pairs) {
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    for (const int target_local_int : target_node.target_indices) {
      const std::size_t target_local =
          static_cast<std::size_t>(target_local_int);
      const std::size_t target_panel_index =
          sampled_target_indices[target_local];
      for (const int source_int : source_node.source_indices) {
        const std::size_t source_index = static_cast<std::size_t>(source_int);
        KifmmJcpChannelRow contribution = production_single_pair_channel(
            panels[target_panel_index], target_panel_index,
            panels[source_index], source_index, unknowns[source_index],
            unknowns[panels.size() + source_index], result.eps_ratio,
            result.kappa, result.near_integration, result.quad_buffer,
            result.production_fallback_pairs, result.production_invalid_pairs);
        add_channel(approx[target_local], contribution);
        add_channel(flat_approx[target_local], contribution);
        add_channel(fallback_direct[target_local], contribution);
      }
    }
  }
  result.production_fallback_seconds =
      elapsed_seconds(fallback_start, Clock::now());
  result.apply_production_fallback_seconds =
      result.production_fallback_seconds;
  record_apply_memory_after_fallback(result);
  if (result.production_fallback_pairs != result.direct_panel_pairs) {
    result.status = "production_fallback_coverage_gap";
    return result;
  }

  result.far_aggregate_relative_rms =
      aggregate_relative_rms(far_approx, far_direct);
  result.far_k1_relative_rms =
      channel_relative_rms(far_approx, far_direct, 0);
  result.far_k2_relative_rms =
      channel_relative_rms(far_approx, far_direct, 1);
  result.far_k3_relative_rms =
      channel_relative_rms(far_approx, far_direct, 2);
  result.far_k4_relative_rms =
      channel_relative_rms(far_approx, far_direct, 3);
  result.fallback_k4_relative_rms =
      channel_relative_rms(fallback_direct, fallback_direct, 3);

  const auto oracle_start = Clock::now();
  std::uint64_t dense_invalid = 0;
  for (std::size_t target_local = 0;
       target_local < sampled_target_indices.size(); ++target_local) {
    const std::size_t target_panel_index =
        sampled_target_indices[target_local];
    for (std::size_t source_index = 0; source_index < panels.size();
         ++source_index) {
      KifmmJcpChannelRow contribution = production_single_pair_channel(
          panels[target_panel_index], target_panel_index, panels[source_index],
          source_index, unknowns[source_index],
          unknowns[panels.size() + source_index], result.eps_ratio,
          result.kappa, result.near_integration, result.quad_buffer,
          result.sampled_dense_reference_valid_pairs, dense_invalid);
      add_channel(direct[target_local], contribution);
    }
  }
  result.dense_oracle_seconds = elapsed_seconds(oracle_start, Clock::now());
  result.production_invalid_pairs += dense_invalid;
  if (result.sampled_dense_reference_valid_pairs != expected_pairs) {
    result.status = "dense_reference_coverage_gap";
    return result;
  }
  if (result.production_invalid_pairs != 0) {
    result.status = "production_invalid_pairs";
    return result;
  }

  if (!finite_channels(approx) || !finite_channels(flat_approx) ||
      !finite_channels(direct)) {
    result.status = "nonfinite_output";
    return result;
  }
  result.hierarchy.approximate_rows = std::move(approx);
  result.hierarchy.direct_rows = std::move(direct);
  result.flat.approximate_rows = std::move(flat_approx);
  result.flat.direct_rows = result.hierarchy.direct_rows;
  result.flat_vs_dense_aggregate_relative_rms =
      aggregate_relative_rms(result.flat.approximate_rows,
                             result.hierarchy.direct_rows);
  const double flat_k1 = channel_relative_rms(
      result.flat.approximate_rows, result.hierarchy.direct_rows, 0);
  const double flat_k2 = channel_relative_rms(
      result.flat.approximate_rows, result.hierarchy.direct_rows, 1);
  const double flat_k3 = channel_relative_rms(
      result.flat.approximate_rows, result.hierarchy.direct_rows, 2);
  result.flat_vs_dense_k4_relative_rms = channel_relative_rms(
      result.flat.approximate_rows, result.hierarchy.direct_rows, 3);
  result.flat_vs_dense_max_channel_relative_rms =
      std::max({flat_k1, flat_k2, flat_k3,
                result.flat_vs_dense_k4_relative_rms});
  result.hierarchy_vs_dense_aggregate_relative_rms =
      aggregate_relative_rms(result.hierarchy.approximate_rows,
                             result.hierarchy.direct_rows);
  result.hierarchy_vs_dense_k4_relative_rms =
      channel_relative_rms(result.hierarchy.approximate_rows,
                           result.hierarchy.direct_rows, 3);
  result.hierarchy_vs_flat_aggregate_relative_rms =
      aggregate_relative_rms(result.hierarchy.approximate_rows,
                             result.flat.approximate_rows);
  result.hierarchy_vs_flat_k4_relative_rms =
      channel_relative_rms(result.hierarchy.approximate_rows,
                           result.flat.approximate_rows, 3);
  if (result.hier_debug_mode == "hybrid-local-split-bridge") {
    result.split_vs_flat_k4_relative_rms =
        result.hierarchy_vs_flat_k4_relative_rms;
  }
  result.k4_combined_relative_rms = result.hierarchy_vs_dense_k4_relative_rms;
  if (result.hier_debug_mode == "l2p-oracle") {
    result.l2p_oracle_aggregate_relative_rms =
        result.hierarchy_vs_dense_aggregate_relative_rms;
    result.l2p_oracle_k4_relative_rms =
        result.hierarchy_vs_dense_k4_relative_rms;
  }
  result.flat_output_hash = hash_channel_rows(result.flat.approximate_rows);
  if (e30_shared_channel_hybrid) {
    apply_shared_worklist_channel_hybrid(
        result, panels, unknowns, sampled_target_indices, tree, worklist,
        config);
    if (result.execution_status == "fail") {
      result.total_seconds = elapsed_seconds(total_start, Clock::now());
      return result;
    }
  } else if (e29_cartesian_k4_only || e29_channel_hybrid ||
             e31_dual_channel_hybrid) {
    const CartesianSampledPanelMatvecConfig cartesian_config =
        e29_cartesian_k4_config(config);
    const CartesianSampledPanelMatvecResult cartesian =
        run_guarded_cartesian_p4_sampled_panel_matvec(
            panels, unknowns, sampled_target_indices, cartesian_config);
    copy_cartesian_k4_telemetry(result, cartesian);
    if (cartesian.execution_status != "pass" ||
        cartesian.approximate_rows.size() != sampled_target_indices.size() ||
        cartesian.direct_rows.size() != sampled_target_indices.size()) {
      result.status = "cartesian_k4_execution_gap";
      result.execution_status = "fail";
      result.accuracy_status = "fail";
      result.accuracy_failure = "cartesian_k4_execution";
      result.total_seconds = elapsed_seconds(total_start, Clock::now());
      return result;
    }
    if (cartesian.production_direct_reference_valid_pairs != expected_pairs ||
        cartesian.invalid_pairs != 0 ||
        cartesian.nonproduction_fallback_pairs != 0) {
      result.status = "cartesian_k4_coverage_gap";
      result.execution_status = "fail";
      result.accuracy_status = "fail";
      result.accuracy_failure = "cartesian_k4_coverage";
      result.total_seconds = elapsed_seconds(total_start, Clock::now());
      return result;
    }
    apply_channel_hybrid_rows(result, cartesian, e29_cartesian_k4_only);
    if (e31_dual_channel_hybrid) {
      annotate_dual_worklist_channel_hybrid(result);
    }
    if (result.execution_status == "fail") {
      result.total_seconds = elapsed_seconds(total_start, Clock::now());
      return result;
    }
  } else {
    result.output_hash = hash_channel_rows(result.hierarchy.approximate_rows);
    set_accuracy(result);
    set_hierarchy_k123_telemetry(result);
  }
  if (result.hier_debug_mode == "local-check-oracle") {
    const bool rhs_pass = std::isfinite(result.local_check_rhs_relative_rms) &&
                          result.local_check_rhs_relative_rms <=
                              result.aggregate_relative_rms_gate;
    result.accuracy_status = rhs_pass ? "pass" : "fail";
    result.accuracy_failure = rhs_pass ? "none" : "local_check_rhs";
  }
  if (config.real_panel_guard == "strict-local-check" &&
      (result.negative_target_local_check_excess_pairs > 0 ||
       (std::isfinite(result.target_local_check_excess_min) &&
        result.target_local_check_excess_min < 0.0))) {
    result.status = "local_check_guard_internal_error";
    result.execution_status = "fail";
    result.accuracy_status = "fail";
    result.accuracy_failure = "target_local_check_excess";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  if (config.real_panel_guard != "strict-local-check" &&
      (result.negative_target_check_excess_pairs > 0 ||
       (std::isfinite(result.target_check_excess_min) &&
        result.target_check_excess_min < 0.0))) {
    result.accuracy_status = "fail";
    result.accuracy_failure = "negative_target_check_excess";
  }
  result.total_seconds = elapsed_seconds(total_start, Clock::now());
  result.status = "ok";
  result.execution_status = "pass";
  return result;
}

KifmmHierarchyRealPanelResult run_dual_worklist_channel_hybrid_with_geometry(
    KifmmHierarchyRealPanelResult result, Clock::time_point total_start,
    std::span<const Panel> panels, std::span<const double> unknowns,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmHierarchyJcpConfig &config,
    std::span<const KifmmPanelSource> sources,
    std::span<const Vec3> targets, std::span<const Vec3> target_normals,
    std::span<const std::size_t> all_target_indices,
    const KifmmHierarchyTree &tree,
    const KifmmHierarchyWorklist &worklist) {
  if (tree.status != "ok") {
    result.status = tree.status;
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  if (worklist.worklist_hash == 0) {
    result.status = "worklist_gap";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }

  const auto hierarchy_start = Clock::now();
  KifmmHierarchyJcpConfig hierarchy_config = config;
  hierarchy_config.hier_debug_mode = "full";
  hierarchy_config.build_dense_oracle = false;
  hierarchy_config.include_direct_fallback = false;
  result.hierarchy = run_full_hierarchical_jcp_kifmm_with_worklist(
      sources, targets, target_normals, hierarchy_config, tree, worklist);
  result.hierarchy_seconds = elapsed_seconds(hierarchy_start, Clock::now());
  result.apply_hierarchy_seconds = result.hierarchy_seconds;
  record_apply_memory_after_hierarchy(result);
  copy_hierarchy_closeout_fields(result);
  if (result.hierarchy.execution_status != "pass" ||
      result.hierarchy.status != "ok") {
    result.status =
        classify_hierarchy_execution_gap(result.hier_debug_mode,
                                         result.hierarchy);
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  if (result.hierarchy.worklist_hash != result.worklist_hash ||
      result.hierarchy.policy_hash != result.policy_hash ||
      result.hierarchy.tree_hash != result.tree_hash) {
    result.status = "hierarchy_hash_gap";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  if (result.hierarchy.approximate_rows.size() != panels.size()) {
    result.status = "hierarchy_row_count_gap";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  result.local_check_rhs_relative_rms =
      result.hierarchy.local_check_rhs_relative_rms;
  result.local_check_rhs_max_abs_error =
      result.hierarchy.local_check_rhs_max_abs_error;
  result.l2p_oracle_aggregate_relative_rms =
      result.hierarchy.l2p_oracle_aggregate_relative_rms;
  result.l2p_oracle_k4_relative_rms =
      result.hierarchy.l2p_oracle_k4_relative_rms;
  result.k4_laplace_grad_relative_rms =
      result.hierarchy.k4_laplace_grad_relative_rms;
  result.k4_yukawa_grad_relative_rms =
      result.hierarchy.k4_yukawa_grad_relative_rms;
  result.k4_combined_relative_rms =
      result.hierarchy.k4_combined_relative_rms;
  result.field_attribution = result.hierarchy.field_attribution;

  std::vector<KifmmJcpChannelRow> full_approx =
      result.hierarchy.approximate_rows;
  result.far_panel_pairs = result.hierarchy.stats.far_panel_pairs;
  result.direct_panel_pairs = result.hierarchy.stats.direct_panel_pairs;
  result.exact_pair_coverage =
      result.far_panel_pairs + result.direct_panel_pairs;
  result.kifmm_compression_ratio =
      pair_fraction(result.far_panel_pairs, result.full_coverage_expected);
  result.far_fraction =
      result.full_coverage_expected == 0
          ? 0.0
          : static_cast<double>(result.far_panel_pairs) /
                static_cast<double>(result.full_coverage_expected);
  if (result.exact_pair_coverage != result.full_coverage_expected) {
    result.status = "coverage_gap";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }

  const auto guard_audit_start = Clock::now();
  result.source_box_boundary_margin_min =
      std::numeric_limits<double>::infinity();
  result.target_check_excess_min = std::numeric_limits<double>::infinity();
  result.target_panel_envelope_excess_min =
      std::numeric_limits<double>::infinity();
  result.target_local_check_excess_min =
      std::numeric_limits<double>::infinity();
  result.mac_bin_min_target_check_excess.fill(
      std::numeric_limits<double>::infinity());
  for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const double distance = norm(sub(target_node.center, source_node.center));
    const double ratio =
        distance > 0.0 ? (std::sqrt(3.0) * source_node.half_width) / distance
                       : std::numeric_limits<double>::infinity();
    const int bin = mac_bin(ratio);
    ++result.mac_bin_cell_pairs[static_cast<std::size_t>(bin)];
    result.mac_bin_panel_pairs[static_cast<std::size_t>(bin)] +=
        static_cast<std::uint64_t>(target_node.target_indices.size()) *
        static_cast<std::uint64_t>(source_node.source_indices.size());
    for (const int source_int : source_node.source_indices) {
      const std::size_t source_index = static_cast<std::size_t>(source_int);
      result.source_box_boundary_margin_min = std::min(
          result.source_box_boundary_margin_min,
          source_boundary_margin(sources[source_index], source_node));
    }
    for (const int target_local_int : target_node.target_indices) {
      const std::size_t target_panel_index =
          static_cast<std::size_t>(target_local_int);
      const double check_excess = target_vertex_check_excess(
          panels[target_panel_index], source_node,
          config.hierarchy.upward_check_scale);
      const double local_check_excess =
          target_local_check_excess(source_node, target_node, config.hierarchy);
      result.target_check_excess_min =
          std::min(result.target_check_excess_min, check_excess);
      result.target_panel_envelope_excess_min =
          std::min(result.target_panel_envelope_excess_min, check_excess);
      result.target_local_check_excess_min =
          std::min(result.target_local_check_excess_min, local_check_excess);
      result.mac_bin_min_target_check_excess[static_cast<std::size_t>(bin)] =
          std::min(result.mac_bin_min_target_check_excess
                       [static_cast<std::size_t>(bin)],
                   check_excess);
      if (check_excess < 0.0) {
        const auto source_panel_pairs =
            static_cast<std::uint64_t>(source_node.source_indices.size());
        result.negative_target_check_excess_pairs += source_panel_pairs;
        result.negative_target_check_excess_panel_pairs += source_panel_pairs;
      }
      if (local_check_excess < 0.0) {
        const auto source_panel_pairs =
            static_cast<std::uint64_t>(source_node.source_indices.size());
        result.negative_target_local_check_excess_pairs += source_panel_pairs;
        result.negative_target_local_check_excess_panel_pairs +=
            source_panel_pairs;
      }
    }
  }
  if (!std::isfinite(result.source_box_boundary_margin_min)) {
    result.source_box_boundary_margin_min =
        std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(result.target_check_excess_min)) {
    result.target_check_excess_min =
        std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(result.target_panel_envelope_excess_min)) {
    result.target_panel_envelope_excess_min =
        std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(result.target_local_check_excess_min)) {
    result.target_local_check_excess_min =
        std::numeric_limits<double>::quiet_NaN();
  }
  for (double &value : result.mac_bin_min_target_check_excess) {
    if (!std::isfinite(value)) {
      value = std::numeric_limits<double>::quiet_NaN();
    }
  }
  result.apply_guard_audit_seconds =
      elapsed_seconds(guard_audit_start, Clock::now());
  record_apply_memory_after_guard_audit(result);

  const auto fallback_start = Clock::now();
  for (const auto &[target_node_id, source_node_id] :
       worklist.direct_leaf_pairs) {
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    for (const int target_local_int : target_node.target_indices) {
      const std::size_t target_panel_index =
          static_cast<std::size_t>(target_local_int);
      for (const int source_int : source_node.source_indices) {
        const std::size_t source_index = static_cast<std::size_t>(source_int);
        KifmmJcpChannelRow contribution = production_single_pair_channel(
            panels[target_panel_index], target_panel_index,
            panels[source_index], source_index, unknowns[source_index],
            unknowns[panels.size() + source_index], result.eps_ratio,
            result.kappa, result.near_integration, result.quad_buffer,
            result.production_fallback_pairs, result.production_invalid_pairs);
        add_channel(full_approx[target_panel_index], contribution);
      }
    }
  }
  result.production_fallback_seconds =
      elapsed_seconds(fallback_start, Clock::now());
  result.apply_production_fallback_seconds =
      result.production_fallback_seconds;
  record_apply_memory_after_fallback(result);
  if (result.production_fallback_pairs != result.direct_panel_pairs) {
    result.status = "production_fallback_coverage_gap";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }

  CartesianSampledPanelMatvecConfig cartesian_config =
      e29_cartesian_k4_config(config);
  cartesian_config.build_direct_reference = false;
  const CartesianSampledPanelMatvecResult cartesian =
      run_guarded_cartesian_p4_sampled_panel_matvec(
          panels, unknowns, all_target_indices, cartesian_config);
  result.cartesian_k4_status = cartesian.status;
  result.cartesian_k4_execution_status = cartesian.execution_status;
  result.cartesian_k4_accuracy_status = cartesian.accuracy_status;
  result.cartesian_k4_accuracy_failure = cartesian.accuracy_failure;
  result.cartesian_k4_far_panel_pairs =
      static_cast<std::uint64_t>(cartesian.far_panel_pairs);
  result.cartesian_k4_fallback_panel_pairs =
      static_cast<std::uint64_t>(cartesian.fallback_panel_pairs);
  const std::uint64_t cartesian_k4_total_pairs =
      result.cartesian_k4_far_panel_pairs +
      result.cartesian_k4_fallback_panel_pairs;
  result.cartesian_k4_compression_ratio =
      pair_fraction(result.cartesian_k4_far_panel_pairs,
                    cartesian_k4_total_pairs);
  result.cartesian_k4_total_seconds = cartesian.total_seconds;
  result.apply_cartesian_k4_seconds = result.cartesian_k4_total_seconds;
  record_apply_memory_after_cartesian_k4(result);
  result.cartesian_k4_policy_hash = cartesian.policy_hash;
  if (cartesian.execution_status != "pass" ||
      cartesian.approximate_rows.size() != panels.size() ||
      cartesian.invalid_pairs != 0 ||
      cartesian.nonproduction_fallback_pairs != 0) {
    result.status = "cartesian_k4_execution_gap";
    result.execution_status = "fail";
    result.accuracy_status = "fail";
    result.accuracy_failure = "cartesian_k4_execution";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  for (std::size_t panel_index = 0; panel_index < panels.size();
       ++panel_index) {
    full_approx[panel_index].k4 = cartesian.approximate_rows[panel_index].k4;
  }

  const auto oracle_start = Clock::now();
  std::vector<KifmmJcpChannelRow> sampled_approx(
      sampled_target_indices.size());
  std::vector<KifmmJcpChannelRow> direct(sampled_target_indices.size());
  std::uint64_t dense_invalid = 0;
  for (std::size_t target_local = 0;
       target_local < sampled_target_indices.size(); ++target_local) {
    const std::size_t target_panel_index =
        sampled_target_indices[target_local];
    sampled_approx[target_local] = full_approx[target_panel_index];
    for (std::size_t source_index = 0; source_index < panels.size();
         ++source_index) {
      KifmmJcpChannelRow contribution = production_single_pair_channel(
          panels[target_panel_index], target_panel_index, panels[source_index],
          source_index, unknowns[source_index],
          unknowns[panels.size() + source_index], result.eps_ratio,
          result.kappa, result.near_integration, result.quad_buffer,
          result.sampled_dense_reference_valid_pairs, dense_invalid);
      add_channel(direct[target_local], contribution);
    }
  }
  result.dense_oracle_seconds = elapsed_seconds(oracle_start, Clock::now());
  result.apply_dense_oracle_seconds = result.dense_oracle_seconds;
  record_apply_memory_after_dense_oracle(result);
  result.production_invalid_pairs += dense_invalid;
  if (result.sampled_dense_reference_valid_pairs !=
      result.sampled_coverage_expected) {
    result.status = "dense_reference_coverage_gap";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  if (result.production_invalid_pairs != 0) {
    result.status = "production_invalid_pairs";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  if (!finite_channels(full_approx) || !finite_channels(sampled_approx) ||
      !finite_channels(direct)) {
    result.status = "nonfinite_output";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }

  const auto finalize_start = Clock::now();
  result.output_hash = hash_channel_rows(full_approx);
  result.channel_hybrid_recombined_hash = result.output_hash;
  result.full_approximate_rows = full_approx;
  result.hierarchy.approximate_rows = std::move(sampled_approx);
  result.hierarchy.direct_rows = std::move(direct);
  result.hierarchy.output_hash = result.output_hash;
  result.hierarchy_vs_dense_aggregate_relative_rms =
      aggregate_relative_rms(result.hierarchy.approximate_rows,
                             result.hierarchy.direct_rows);
  result.hierarchy_vs_dense_k4_relative_rms =
      channel_relative_rms(result.hierarchy.approximate_rows,
                           result.hierarchy.direct_rows, 3);
  set_accuracy(result);
  set_hierarchy_k123_telemetry(result);
  result.hybrid_recombined_aggregate_relative_rms =
      result.aggregate_relative_rms;
  result.hybrid_recombined_max_channel_relative_rms =
      result.max_channel_relative_rms;
  result.hybrid_recombined_max_channel_name = result.max_channel_name;
  result.cartesian_k4_relative_rms = result.k4_relative_rms;
  result.cartesian_k4_abs_rms = result.k4_abs_rms;
  result.cartesian_k4_reference_rms = result.k4_reference_rms;
  const bool k4_pass =
      result.cartesian_k4_relative_rms <= result.max_channel_relative_rms_gate;
  result.cartesian_k4_accuracy_status = k4_pass ? "pass" : "fail";
  result.cartesian_k4_accuracy_failure = k4_pass ? "none" : "k4_channel";
  result.channel_hybrid_double_count_guard = "pass";
  result.status = "ok";
  result.execution_status = "pass";
  result.full_apply_stage_status = "pass";
  result.compression_status =
      result.accuracy_status == "pass"
          ? (result.far_fraction >= 0.10 ? "pass" : "fallback_dominated")
          : "accuracy_gap";
  annotate_dual_worklist_channel_hybrid(result);
  if (config.real_panel_guard == "strict-local-check" &&
      (result.negative_target_local_check_excess_pairs > 0 ||
       (std::isfinite(result.target_local_check_excess_min) &&
        result.target_local_check_excess_min < 0.0))) {
    result.status = "local_check_guard_internal_error";
    result.execution_status = "fail";
    result.accuracy_status = "fail";
    result.accuracy_failure = "target_local_check_excess";
    result.full_apply_stage_status = "fail";
    result.compression_status = "accuracy_gap";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  result.apply_finalize_seconds = elapsed_seconds(finalize_start, Clock::now());
  result.total_seconds = elapsed_seconds(total_start, Clock::now());
  record_apply_memory_after_finalize(result);
  result.apply_unattributed_seconds = compute_apply_unattributed_seconds(result);
  result.runtime_memory_attribution_status = "pass";
  return result;
}

KifmmHierarchyRealPanelResult
run_kifmm_hierarchy_real_panel_full_apply_sampled_audit(
    std::span<const Panel> panels, std::span<const double> unknowns,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmHierarchyJcpConfig &config) {
  const auto total_start = Clock::now();
  KifmmHierarchyRealPanelResult result;
  result.execution_status = "fail";
  result.compression_status = "not_evaluated";
  result.full_apply_stage_status = "fail";
  result.source_panels = panels.size();
  result.target_panels = panels.size();
  result.sampled_rows = sampled_target_indices.size();
  result.full_coverage_expected =
      static_cast<std::uint64_t>(panels.size()) *
      static_cast<std::uint64_t>(panels.size());
  result.sampled_coverage_expected =
      static_cast<std::uint64_t>(panels.size()) *
      static_cast<std::uint64_t>(sampled_target_indices.size());
  result.timing_scope = "full_panel_apply_sampled_audit";
  result.order = config.hierarchy.order;
  result.theta = config.hierarchy.theta;
  result.eps_ratio = config.eps_ratio;
  result.kappa = config.kappa;
  result.condition_policy = config.hierarchy.condition_policy;
  result.source_box_policy = config.hierarchy.source_box_policy;
  result.m2l_eval_policy = config.hierarchy.m2l_eval_policy;
  result.m2l_matrix_cache_policy = config.hierarchy.m2l_matrix_cache_policy;
  result.m2l_gemm_batch_policy = config.hierarchy.m2l_gemm_batch_policy;
  result.m2l_parallel_policy = config.hierarchy.m2l_parallel_policy;
  result.m2l_parallel_workers = config.hierarchy.m2l_parallel_workers;
  result.tree_stop_policy = config.hierarchy.tree_stop_policy;
  result.jcp_channel_policy = config.hierarchy.jcp_channel_policy;
  result.tree_panel_radius_factor =
      config.hierarchy.tree_panel_radius_factor;
  result.tree_occupancy_stop = config.hierarchy.tree_occupancy_stop;
  result.real_panel_guard = config.real_panel_guard;
  result.hier_debug_mode =
      config.hier_debug_mode.empty() ? std::string("full")
                                     : config.hier_debug_mode;
  result.real_panel_guard_margin_factor =
      config.real_panel_guard_margin_factor;
  result.aggregate_relative_rms_gate = config.aggregate_relative_rms_gate;
  result.max_channel_relative_rms_gate =
      config.max_channel_relative_rms_gate;

  if (panels.empty() || sampled_target_indices.empty()) {
    result.status = "invalid_input_size";
    return result;
  }
  if (unknowns.size() != 2 * panels.size()) {
    result.status = "invalid_unknown_size";
    return result;
  }
  if (!std::isfinite(config.eps_ratio) || config.eps_ratio == 0.0 ||
      !std::isfinite(config.kappa) || config.kappa < 0.0) {
    result.status = "invalid_physics";
    return result;
  }
  if (config.real_panel_guard != "strict-local-check") {
    result.status = "unsupported_full_apply_guard";
    return result;
  }
  if (result.hier_debug_mode !=
      "dual-worklist-channel-hybrid-k123-kifmm-k4-cartesian") {
    result.status = "unsupported_full_apply_mode";
    return result;
  }
  if (!std::isfinite(config.real_panel_guard_margin_factor) ||
      config.real_panel_guard_margin_factor < 0.0) {
    result.status = "invalid_real_panel_guard_margin";
    return result;
  }
  std::set<std::size_t> unique_targets;
  for (const std::size_t index : sampled_target_indices) {
    if (index >= panels.size() || !unique_targets.insert(index).second) {
      result.status = "invalid_target_indices";
      return result;
    }
  }
  for (double value : unknowns) {
    if (!std::isfinite(value)) {
      result.status = "invalid_unknown";
      return result;
    }
  }
  for (const Panel &panel : panels) {
    if (!valid_panel(panel)) {
      result.status = "invalid_panel_geometry";
      return result;
    }
  }

  result.panel_geometry_hash = hash_panels(panels);
  result.unknown_vector_hash = hash_doubles(unknowns);
  result.sampled_target_hash = hash_indices(sampled_target_indices);

  std::vector<KifmmPanelSource> sources;
  sources.reserve(panels.size());
  for (std::size_t i = 0; i < panels.size(); ++i) {
    sources.push_back(source_from_panel(panels[i], unknowns[i],
                                        unknowns[panels.size() + i]));
  }
  std::vector<Vec3> targets;
  std::vector<Vec3> target_normals;
  targets.reserve(panels.size());
  target_normals.reserve(panels.size());
  std::vector<std::size_t> all_target_indices(panels.size());
  std::iota(all_target_indices.begin(), all_target_indices.end(), 0);
  for (const Panel &panel : panels) {
    targets.push_back(position_of(panel));
    target_normals.push_back(normal_of(panel));
  }

  const std::vector<Vec3> source_positions = positions_from_sources(sources);
  const std::vector<double> radii = panel_radii(panels);
  const KifmmHierarchyTree tree =
      build_strict_octree(source_positions, targets, radii, radii,
                          config.hierarchy);
  result.tree_hash = tree.tree_hash;
  if (tree.status != "ok") {
    result.status = tree.status;
    return result;
  }
  copy_tree_policy_and_stats(result, config.hierarchy, tree);
  const std::vector<KifmmRealPanelNodeEnvelope> source_envelopes =
      build_source_envelopes(tree, panels);
  const std::vector<KifmmRealPanelNodeEnvelope> target_envelopes =
      build_target_envelopes(tree, panels, all_target_indices);
  update_leaf_panel_telemetry(tree, source_envelopes, target_envelopes,
                              result);
  update_l2l_containment_telemetry(tree, config.hierarchy, result);
  const KifmmHierarchyWorklist worklist = build_real_panel_guarded_worklist(
      tree, config.hierarchy, source_envelopes, target_envelopes,
      config.real_panel_guard_margin_factor, true, result);
  result.worklist_hash = worklist.worklist_hash;
  result.policy_hash = worklist.policy_hash;
  if (worklist.worklist_hash == 0) {
    result.status = "worklist_gap";
    return result;
  }

  const auto hierarchy_start = Clock::now();
  KifmmHierarchyJcpConfig hierarchy_config = config;
  hierarchy_config.hier_debug_mode = "full";
  hierarchy_config.build_dense_oracle = false;
  hierarchy_config.include_direct_fallback = false;
  result.hierarchy = run_full_hierarchical_jcp_kifmm_with_worklist(
      sources, targets, target_normals, hierarchy_config, tree, worklist);
  result.hierarchy_seconds = elapsed_seconds(hierarchy_start, Clock::now());
  copy_hierarchy_closeout_fields(result);
  if (result.hierarchy.execution_status != "pass" ||
      result.hierarchy.status != "ok") {
    result.status =
        classify_hierarchy_execution_gap(result.hier_debug_mode,
                                         result.hierarchy);
    return result;
  }
  if (result.hierarchy.worklist_hash != result.worklist_hash ||
      result.hierarchy.policy_hash != result.policy_hash ||
      result.hierarchy.tree_hash != result.tree_hash) {
    result.status = "hierarchy_hash_gap";
    return result;
  }
  if (result.hierarchy.approximate_rows.size() != panels.size()) {
    result.status = "hierarchy_row_count_gap";
    return result;
  }
  result.local_check_rhs_relative_rms =
      result.hierarchy.local_check_rhs_relative_rms;
  result.local_check_rhs_max_abs_error =
      result.hierarchy.local_check_rhs_max_abs_error;
  result.l2p_oracle_aggregate_relative_rms =
      result.hierarchy.l2p_oracle_aggregate_relative_rms;
  result.l2p_oracle_k4_relative_rms =
      result.hierarchy.l2p_oracle_k4_relative_rms;
  result.k4_laplace_grad_relative_rms =
      result.hierarchy.k4_laplace_grad_relative_rms;
  result.k4_yukawa_grad_relative_rms =
      result.hierarchy.k4_yukawa_grad_relative_rms;
  result.k4_combined_relative_rms =
      result.hierarchy.k4_combined_relative_rms;
  result.field_attribution = result.hierarchy.field_attribution;

  std::vector<KifmmJcpChannelRow> full_approx =
      result.hierarchy.approximate_rows;
  result.far_panel_pairs = result.hierarchy.stats.far_panel_pairs;
  result.direct_panel_pairs = result.hierarchy.stats.direct_panel_pairs;
  result.exact_pair_coverage =
      result.far_panel_pairs + result.direct_panel_pairs;
  result.kifmm_compression_ratio =
      pair_fraction(result.far_panel_pairs, result.full_coverage_expected);
  result.far_fraction =
      result.full_coverage_expected == 0
          ? 0.0
          : static_cast<double>(result.far_panel_pairs) /
                static_cast<double>(result.full_coverage_expected);
  if (result.exact_pair_coverage != result.full_coverage_expected) {
    result.status = "coverage_gap";
    return result;
  }

  result.source_box_boundary_margin_min =
      std::numeric_limits<double>::infinity();
  result.target_check_excess_min = std::numeric_limits<double>::infinity();
  result.target_panel_envelope_excess_min =
      std::numeric_limits<double>::infinity();
  result.target_local_check_excess_min =
      std::numeric_limits<double>::infinity();
  result.mac_bin_min_target_check_excess.fill(
      std::numeric_limits<double>::infinity());
  for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const double distance = norm(sub(target_node.center, source_node.center));
    const double ratio =
        distance > 0.0 ? (std::sqrt(3.0) * source_node.half_width) / distance
                       : std::numeric_limits<double>::infinity();
    const int bin = mac_bin(ratio);
    ++result.mac_bin_cell_pairs[static_cast<std::size_t>(bin)];
    result.mac_bin_panel_pairs[static_cast<std::size_t>(bin)] +=
        static_cast<std::uint64_t>(target_node.target_indices.size()) *
        static_cast<std::uint64_t>(source_node.source_indices.size());
    for (const int source_int : source_node.source_indices) {
      const std::size_t source_index = static_cast<std::size_t>(source_int);
      result.source_box_boundary_margin_min = std::min(
          result.source_box_boundary_margin_min,
          source_boundary_margin(sources[source_index], source_node));
    }
    for (const int target_local_int : target_node.target_indices) {
      const std::size_t target_panel_index =
          static_cast<std::size_t>(target_local_int);
      const double check_excess = target_vertex_check_excess(
          panels[target_panel_index], source_node,
          config.hierarchy.upward_check_scale);
      const double local_check_excess =
          target_local_check_excess(source_node, target_node, config.hierarchy);
      result.target_check_excess_min =
          std::min(result.target_check_excess_min, check_excess);
      result.target_panel_envelope_excess_min =
          std::min(result.target_panel_envelope_excess_min, check_excess);
      result.target_local_check_excess_min =
          std::min(result.target_local_check_excess_min, local_check_excess);
      result.mac_bin_min_target_check_excess[static_cast<std::size_t>(bin)] =
          std::min(result.mac_bin_min_target_check_excess
                       [static_cast<std::size_t>(bin)],
                   check_excess);
      if (check_excess < 0.0) {
        const auto source_panel_pairs =
            static_cast<std::uint64_t>(source_node.source_indices.size());
        result.negative_target_check_excess_pairs += source_panel_pairs;
        result.negative_target_check_excess_panel_pairs += source_panel_pairs;
      }
      if (local_check_excess < 0.0) {
        const auto source_panel_pairs =
            static_cast<std::uint64_t>(source_node.source_indices.size());
        result.negative_target_local_check_excess_pairs += source_panel_pairs;
        result.negative_target_local_check_excess_panel_pairs +=
            source_panel_pairs;
      }
    }
  }
  if (!std::isfinite(result.source_box_boundary_margin_min)) {
    result.source_box_boundary_margin_min =
        std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(result.target_check_excess_min)) {
    result.target_check_excess_min =
        std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(result.target_panel_envelope_excess_min)) {
    result.target_panel_envelope_excess_min =
        std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(result.target_local_check_excess_min)) {
    result.target_local_check_excess_min =
        std::numeric_limits<double>::quiet_NaN();
  }
  for (double &value : result.mac_bin_min_target_check_excess) {
    if (!std::isfinite(value)) {
      value = std::numeric_limits<double>::quiet_NaN();
    }
  }

  const auto fallback_start = Clock::now();
  for (const auto &[target_node_id, source_node_id] :
       worklist.direct_leaf_pairs) {
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    for (const int target_local_int : target_node.target_indices) {
      const std::size_t target_panel_index =
          static_cast<std::size_t>(target_local_int);
      for (const int source_int : source_node.source_indices) {
        const std::size_t source_index = static_cast<std::size_t>(source_int);
        KifmmJcpChannelRow contribution = production_single_pair_channel(
            panels[target_panel_index], target_panel_index,
            panels[source_index], source_index, unknowns[source_index],
            unknowns[panels.size() + source_index], result.eps_ratio,
            result.kappa, result.near_integration, result.quad_buffer,
            result.production_fallback_pairs, result.production_invalid_pairs);
        add_channel(full_approx[target_panel_index], contribution);
      }
    }
  }
  result.production_fallback_seconds =
      elapsed_seconds(fallback_start, Clock::now());
  if (result.production_fallback_pairs != result.direct_panel_pairs) {
    result.status = "production_fallback_coverage_gap";
    return result;
  }

  CartesianSampledPanelMatvecConfig cartesian_config =
      e29_cartesian_k4_config(config);
  cartesian_config.build_direct_reference = false;
  const CartesianSampledPanelMatvecResult cartesian =
      run_guarded_cartesian_p4_sampled_panel_matvec(
          panels, unknowns, all_target_indices, cartesian_config);
  result.cartesian_k4_status = cartesian.status;
  result.cartesian_k4_execution_status = cartesian.execution_status;
  result.cartesian_k4_accuracy_status = cartesian.accuracy_status;
  result.cartesian_k4_accuracy_failure = cartesian.accuracy_failure;
  result.cartesian_k4_far_panel_pairs =
      static_cast<std::uint64_t>(cartesian.far_panel_pairs);
  result.cartesian_k4_fallback_panel_pairs =
      static_cast<std::uint64_t>(cartesian.fallback_panel_pairs);
  const std::uint64_t cartesian_k4_total_pairs =
      result.cartesian_k4_far_panel_pairs +
      result.cartesian_k4_fallback_panel_pairs;
  result.cartesian_k4_compression_ratio =
      pair_fraction(result.cartesian_k4_far_panel_pairs,
                    cartesian_k4_total_pairs);
  result.cartesian_k4_total_seconds = cartesian.total_seconds;
  result.cartesian_k4_policy_hash = cartesian.policy_hash;
  if (cartesian.execution_status != "pass" ||
      cartesian.approximate_rows.size() != panels.size() ||
      cartesian.invalid_pairs != 0 ||
      cartesian.nonproduction_fallback_pairs != 0) {
    result.status = "cartesian_k4_execution_gap";
    result.execution_status = "fail";
    result.accuracy_status = "fail";
    result.accuracy_failure = "cartesian_k4_execution";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  for (std::size_t panel_index = 0; panel_index < panels.size();
       ++panel_index) {
    full_approx[panel_index].k4 = cartesian.approximate_rows[panel_index].k4;
  }

  const auto oracle_start = Clock::now();
  std::vector<KifmmJcpChannelRow> sampled_approx(
      sampled_target_indices.size());
  std::vector<KifmmJcpChannelRow> direct(sampled_target_indices.size());
  std::uint64_t dense_invalid = 0;
  for (std::size_t target_local = 0;
       target_local < sampled_target_indices.size(); ++target_local) {
    const std::size_t target_panel_index =
        sampled_target_indices[target_local];
    sampled_approx[target_local] = full_approx[target_panel_index];
    for (std::size_t source_index = 0; source_index < panels.size();
         ++source_index) {
      KifmmJcpChannelRow contribution = production_single_pair_channel(
          panels[target_panel_index], target_panel_index, panels[source_index],
          source_index, unknowns[source_index],
          unknowns[panels.size() + source_index], result.eps_ratio,
          result.kappa, result.near_integration, result.quad_buffer,
          result.sampled_dense_reference_valid_pairs, dense_invalid);
      add_channel(direct[target_local], contribution);
    }
  }
  result.dense_oracle_seconds = elapsed_seconds(oracle_start, Clock::now());
  result.production_invalid_pairs += dense_invalid;
  if (result.sampled_dense_reference_valid_pairs !=
      result.sampled_coverage_expected) {
    result.status = "dense_reference_coverage_gap";
    return result;
  }
  if (result.production_invalid_pairs != 0) {
    result.status = "production_invalid_pairs";
    return result;
  }
  if (!finite_channels(full_approx) || !finite_channels(sampled_approx) ||
      !finite_channels(direct)) {
    result.status = "nonfinite_output";
    return result;
  }

  result.output_hash = hash_channel_rows(full_approx);
  result.channel_hybrid_recombined_hash = result.output_hash;
  result.full_approximate_rows = full_approx;
  result.hierarchy.approximate_rows = std::move(sampled_approx);
  result.hierarchy.direct_rows = std::move(direct);
  result.hierarchy.output_hash = result.output_hash;
  result.hierarchy_vs_dense_aggregate_relative_rms =
      aggregate_relative_rms(result.hierarchy.approximate_rows,
                             result.hierarchy.direct_rows);
  result.hierarchy_vs_dense_k4_relative_rms =
      channel_relative_rms(result.hierarchy.approximate_rows,
                           result.hierarchy.direct_rows, 3);
  set_accuracy(result);
  set_hierarchy_k123_telemetry(result);
  result.hybrid_recombined_aggregate_relative_rms =
      result.aggregate_relative_rms;
  result.hybrid_recombined_max_channel_relative_rms =
      result.max_channel_relative_rms;
  result.hybrid_recombined_max_channel_name = result.max_channel_name;
  result.cartesian_k4_relative_rms = result.k4_relative_rms;
  result.cartesian_k4_abs_rms = result.k4_abs_rms;
  result.cartesian_k4_reference_rms = result.k4_reference_rms;
  const bool k4_pass =
      result.cartesian_k4_relative_rms <= result.max_channel_relative_rms_gate;
  result.cartesian_k4_accuracy_status = k4_pass ? "pass" : "fail";
  result.cartesian_k4_accuracy_failure = k4_pass ? "none" : "k4_channel";
  result.channel_hybrid_double_count_guard = "pass";
  result.status = "ok";
  result.execution_status = "pass";
  result.full_apply_stage_status = "pass";
  result.compression_status =
      result.accuracy_status == "pass"
          ? (result.far_fraction >= 0.10 ? "pass" : "fallback_dominated")
          : "accuracy_gap";
  annotate_dual_worklist_channel_hybrid(result);
  if (config.real_panel_guard == "strict-local-check" &&
      (result.negative_target_local_check_excess_pairs > 0 ||
       (std::isfinite(result.target_local_check_excess_min) &&
        result.target_local_check_excess_min < 0.0))) {
    result.status = "local_check_guard_internal_error";
    result.execution_status = "fail";
    result.accuracy_status = "fail";
    result.accuracy_failure = "target_local_check_excess";
    result.full_apply_stage_status = "fail";
    result.compression_status = "accuracy_gap";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  result.total_seconds = elapsed_seconds(total_start, Clock::now());
  return result;
}

DualWorklistChannelHybridPreparedOperator
prepare_dual_worklist_channel_hybrid_operator_impl(
    std::span<const Panel> panels,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmHierarchyJcpConfig &config) {
  const auto start = Clock::now();
  DualWorklistChannelHybridPreparedOperator prepared;
  prepared.panels.assign(panels.begin(), panels.end());
  prepared.sampled_target_indices.assign(sampled_target_indices.begin(),
                                         sampled_target_indices.end());
  prepared.config = config;
  prepared.source_panels = prepared.panels.size();
  prepared.target_panels = prepared.panels.size();
  prepared.panel_geometry_hash = hash_panels(prepared.panels);
  prepared.sampled_target_hash = hash_indices(prepared.sampled_target_indices);
  prepared.prepare_build_count = 1;
  prepared.frozen_worklist = false;
  prepared.frozen_equivalent_system_cache = false;
  prepared.prepared_rebuild_status = "prepare_failed";
  if (panels.empty() || sampled_target_indices.empty()) {
    prepared.prepared_rebuild_status = "invalid_input_size";
    prepared.prepare_seconds = elapsed_seconds(start, Clock::now());
    return prepared;
  }
  if (!std::isfinite(config.eps_ratio) || config.eps_ratio == 0.0 ||
      !std::isfinite(config.kappa) || config.kappa < 0.0) {
    prepared.prepared_rebuild_status = "invalid_physics";
    prepared.prepare_seconds = elapsed_seconds(start, Clock::now());
    return prepared;
  }
  if (config.real_panel_guard != "strict-local-check") {
    prepared.prepared_rebuild_status = "unsupported_full_apply_guard";
    prepared.prepare_seconds = elapsed_seconds(start, Clock::now());
    return prepared;
  }
  const std::string mode = config.hier_debug_mode.empty()
                               ? std::string("full")
                               : config.hier_debug_mode;
  if (mode != "dual-worklist-channel-hybrid-k123-kifmm-k4-cartesian") {
    prepared.prepared_rebuild_status = "unsupported_full_apply_mode";
    prepared.prepare_seconds = elapsed_seconds(start, Clock::now());
    return prepared;
  }
  if (!std::isfinite(config.real_panel_guard_margin_factor) ||
      config.real_panel_guard_margin_factor < 0.0) {
    prepared.prepared_rebuild_status = "invalid_real_panel_guard_margin";
    prepared.prepare_seconds = elapsed_seconds(start, Clock::now());
    return prepared;
  }
  std::set<std::size_t> unique_targets;
  for (const std::size_t index : sampled_target_indices) {
    if (index >= panels.size() || !unique_targets.insert(index).second) {
      prepared.prepared_rebuild_status = "invalid_target_indices";
      prepared.prepare_seconds = elapsed_seconds(start, Clock::now());
      return prepared;
    }
  }
  for (const Panel &panel : panels) {
    if (!valid_panel(panel)) {
      prepared.prepared_rebuild_status = "invalid_panel_geometry";
      prepared.prepare_seconds = elapsed_seconds(start, Clock::now());
      return prepared;
    }
  }
  prepared.targets.reserve(panels.size());
  prepared.target_normals.reserve(panels.size());
  prepared.all_target_indices.resize(panels.size());
  std::iota(prepared.all_target_indices.begin(),
            prepared.all_target_indices.end(), 0);
  std::vector<Vec3> source_positions;
  source_positions.reserve(panels.size());
  for (const Panel &panel : panels) {
    source_positions.push_back(position_of(panel));
    prepared.targets.push_back(position_of(panel));
    prepared.target_normals.push_back(normal_of(panel));
  }
  const std::vector<double> radii = panel_radii(panels);
  prepared.tree =
      build_strict_octree(source_positions, prepared.targets, radii, radii,
                          config.hierarchy);
  prepared.tree_hash = prepared.tree.tree_hash;
  if (prepared.tree.status != "ok") {
    prepared.prepared_rebuild_status = prepared.tree.status;
    prepared.prepare_seconds = elapsed_seconds(start, Clock::now());
    return prepared;
  }
  KifmmHierarchyRealPanelResult prepare_result;
  prepare_result.real_panel_guard = config.real_panel_guard;
  prepare_result.real_panel_guard_margin_factor =
      config.real_panel_guard_margin_factor;
  copy_tree_policy_and_stats(prepare_result, config.hierarchy, prepared.tree);
  const std::vector<KifmmRealPanelNodeEnvelope> source_envelopes =
      build_source_envelopes(prepared.tree, panels);
  const std::vector<KifmmRealPanelNodeEnvelope> target_envelopes =
      build_target_envelopes(prepared.tree, panels,
                             prepared.all_target_indices);
  update_leaf_panel_telemetry(prepared.tree, source_envelopes,
                              target_envelopes, prepare_result);
  update_l2l_containment_telemetry(prepared.tree, config.hierarchy,
                                   prepare_result);
  prepared.worklist = build_real_panel_guarded_worklist(
      prepared.tree, config.hierarchy, source_envelopes, target_envelopes,
      config.real_panel_guard_margin_factor, true, prepare_result);
  prepared.worklist_hash = prepared.worklist.worklist_hash;
  prepared.policy_hash = prepared.worklist.policy_hash;
  if (prepared.worklist_hash == 0) {
    prepared.prepared_rebuild_status = "worklist_gap";
    prepared.prepare_seconds = elapsed_seconds(start, Clock::now());
    return prepared;
  }
  std::uint64_t prepared_panel_pairs = 0;
  for (const auto &[target_node_id, source_node_id] :
       prepared.worklist.m2l_pairs) {
    const KifmmHierarchyNode &target_node =
        prepared.tree.nodes[static_cast<std::size_t>(target_node_id)];
    const KifmmHierarchyNode &source_node =
        prepared.tree.nodes[static_cast<std::size_t>(source_node_id)];
    prepared_panel_pairs +=
        static_cast<std::uint64_t>(target_node.target_indices.size()) *
        static_cast<std::uint64_t>(source_node.source_indices.size());
  }
  for (const auto &[target_node_id, source_node_id] :
       prepared.worklist.direct_leaf_pairs) {
    const KifmmHierarchyNode &target_node =
        prepared.tree.nodes[static_cast<std::size_t>(target_node_id)];
    const KifmmHierarchyNode &source_node =
        prepared.tree.nodes[static_cast<std::size_t>(source_node_id)];
    prepared_panel_pairs +=
        static_cast<std::uint64_t>(target_node.target_indices.size()) *
        static_cast<std::uint64_t>(source_node.source_indices.size());
  }
  prepared.prepared_worklist_pairs =
      static_cast<std::size_t>(prepared_panel_pairs);
  prepared.prepared_equivalent_system_cache_entries =
      prepared.worklist.m2l_pairs.empty() ? 0u : 1u;
  prepared.frozen_worklist = true;
  prepared.frozen_equivalent_system_cache =
      prepared.prepared_equivalent_system_cache_entries > 0;
  prepared.prepared_rebuild_status = "frozen_prepared_apply";
  prepared.prepare_seconds = elapsed_seconds(start, Clock::now());
  return prepared;
}

KifmmHierarchyRealPanelResult
apply_prepared_dual_worklist_channel_hybrid_operator_impl(
    DualWorklistChannelHybridPreparedOperator &prepared,
    std::span<const double> unknowns,
    bool compute_sampled_direct_audit) {
  KifmmHierarchyJcpConfig config = prepared.config;
  config.build_dense_oracle = compute_sampled_direct_audit;
  ++prepared.apply_count;
  const auto total_start = Clock::now();
  KifmmHierarchyRealPanelResult result;
  result.execution_status = "fail";
  result.compression_status = "not_evaluated";
  result.full_apply_stage_status = "fail";
  result.source_panels = prepared.panels.size();
  result.target_panels = prepared.panels.size();
  result.sampled_rows = prepared.sampled_target_indices.size();
  result.full_coverage_expected =
      static_cast<std::uint64_t>(prepared.panels.size()) *
      static_cast<std::uint64_t>(prepared.panels.size());
  result.sampled_coverage_expected =
      static_cast<std::uint64_t>(prepared.panels.size()) *
      static_cast<std::uint64_t>(prepared.sampled_target_indices.size());
  result.timing_scope = "full_panel_apply_sampled_audit";
  result.order = config.hierarchy.order;
  result.theta = config.hierarchy.theta;
  result.eps_ratio = config.eps_ratio;
  result.kappa = config.kappa;
  result.condition_policy = config.hierarchy.condition_policy;
  result.source_box_policy = config.hierarchy.source_box_policy;
  result.m2l_eval_policy = config.hierarchy.m2l_eval_policy;
  result.m2l_matrix_cache_policy = config.hierarchy.m2l_matrix_cache_policy;
  result.m2l_gemm_batch_policy = config.hierarchy.m2l_gemm_batch_policy;
  result.m2l_parallel_policy = config.hierarchy.m2l_parallel_policy;
  result.m2l_parallel_workers = config.hierarchy.m2l_parallel_workers;
  result.tree_stop_policy = config.hierarchy.tree_stop_policy;
  result.jcp_channel_policy = config.hierarchy.jcp_channel_policy;
  result.tree_panel_radius_factor =
      config.hierarchy.tree_panel_radius_factor;
  result.tree_occupancy_stop = config.hierarchy.tree_occupancy_stop;
  result.real_panel_guard = config.real_panel_guard;
  result.hier_debug_mode =
      config.hier_debug_mode.empty() ? std::string("full")
                                     : config.hier_debug_mode;
  result.real_panel_guard_margin_factor =
      config.real_panel_guard_margin_factor;
  result.aggregate_relative_rms_gate = config.aggregate_relative_rms_gate;
  result.max_channel_relative_rms_gate =
      config.max_channel_relative_rms_gate;
  result.panel_geometry_hash = prepared.panel_geometry_hash;
  result.sampled_target_hash = prepared.sampled_target_hash;
  result.tree_hash = prepared.tree_hash;
  result.worklist_hash = prepared.worklist_hash;
  result.policy_hash = prepared.policy_hash;
  copy_tree_policy_and_stats(result, config.hierarchy, prepared.tree);
  record_apply_memory_start(result);

  if (!prepared.frozen_worklist || !prepared.frozen_equivalent_system_cache ||
      prepared.prepared_rebuild_status != "frozen_prepared_apply") {
    result.status = prepared.prepared_rebuild_status.empty()
                        ? "prepared_context_not_frozen"
                        : prepared.prepared_rebuild_status;
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  if (unknowns.size() != 2 * prepared.panels.size()) {
    result.status = "invalid_unknown_size";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  for (double value : unknowns) {
    if (!std::isfinite(value)) {
      result.status = "invalid_unknown";
      result.total_seconds = elapsed_seconds(total_start, Clock::now());
      return result;
    }
  }
  result.unknown_vector_hash = hash_doubles(unknowns);
  const auto source_build_start = Clock::now();
  std::vector<KifmmPanelSource> sources;
  sources.reserve(prepared.panels.size());
  for (std::size_t i = 0; i < prepared.panels.size(); ++i) {
    sources.push_back(source_from_panel(prepared.panels[i], unknowns[i],
                                        unknowns[prepared.panels.size() + i]));
  }
  result.apply_source_build_seconds =
      elapsed_seconds(source_build_start, Clock::now());
  record_apply_memory_after_source_build(result);
  result = run_dual_worklist_channel_hybrid_with_geometry(
      std::move(result), total_start, prepared.panels, unknowns,
      prepared.sampled_target_indices, config, sources, prepared.targets,
      prepared.target_normals, prepared.all_target_indices, prepared.tree,
      prepared.worklist);
  if (result.worklist_hash != prepared.worklist_hash ||
      result.policy_hash != prepared.policy_hash ||
      result.tree_hash != prepared.tree_hash ||
      result.panel_geometry_hash != prepared.panel_geometry_hash ||
      result.sampled_target_hash != prepared.sampled_target_hash) {
    result.status = "prepared_lifecycle_hash_drift";
    result.execution_status = "fail";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }
  if (prepared.apply_count == 1) {
    prepared.prepared_equivalent_system_cache_entries =
        static_cast<std::size_t>(result.m2l_matrix_cache_entries);
  }
  return result;
}

void assemble_dual_worklist_channel_hybrid_tabi_system_matvec_impl(
    const KifmmHierarchyRealPanelResult &applied,
    std::span<const double> unknowns,
    double lhs_phi,
    double lhs_phin,
    std::vector<double> &out) {
  const std::size_t n = applied.full_approximate_rows.size();
  if (n == 0 || unknowns.size() != 2 * n) {
    throw std::invalid_argument(
        "dual_worklist_channel_hybrid_matvec_size_mismatch");
  }
  if (!std::isfinite(lhs_phi) || !std::isfinite(lhs_phin)) {
    throw std::invalid_argument(
        "dual_worklist_channel_hybrid_matvec_nonfinite_diagonal");
  }
  for (double value : unknowns) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "dual_worklist_channel_hybrid_matvec_nonfinite_input");
    }
  }
  out.assign(2 * n, 0.0);
  for (std::size_t panel = 0; panel < n; ++panel) {
    const KifmmJcpChannelRow &row = applied.full_approximate_rows[panel];
    if (!std::isfinite(row.k1) || !std::isfinite(row.k2) ||
        !std::isfinite(row.k3) || !std::isfinite(row.k4)) {
      throw std::invalid_argument(
          "dual_worklist_channel_hybrid_matvec_nonfinite_row");
    }
    out[panel] = lhs_phi * unknowns[panel] - (row.k1 + row.k2);
    out[n + panel] =
        lhs_phin * unknowns[n + panel] - (row.k3 + row.k4);
  }
}

} // namespace titan::tabi::fmm_kifmm
