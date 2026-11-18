#include "tabi_fmm_kifmm_channel_hybrid_operator.h"
#include "tabi_fmm_kifmm_full_panel_matvec.h"
#include "tabi_fmm_m2l_preconditioner.h"
#include "tabi_fmm_m2l_prototype.h"
#include "tabi_preconditioner.h"
#include "tabi_cartesian_retry_foundation.h"
#include "tabi_types.h"
#include "../bie/bie_cartesian_fmm_solver_adapter.h"
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

#include "tabi_near.h"
#include "tabi_charge_resolution.h"
#include "../titan_ui.h"
#include "tabi_node_patch.h"
#include "tabi_node_patch_curved.h"

#include "../aligned_allocator.h"
#include "../titan_executor.h"
#include "../titan_simd_dispatch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace TitanPBE::Tabi {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kCoulombKcalPerMol = 332.063713299;
constexpr double kKcalToKJ = 4.184;
constexpr double kKBKJPerMolK = 0.00831446261815324;
constexpr std::size_t kParallelVectorThreshold = 32768;

constexpr std::uint64_t kFmmAuditHashOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFmmAuditHashPrime = 1099511628211ULL;

#if defined(TITAN_CARTESIAN_FMM_TESTING)
void maybe_write_p3c0_packed_plan_inspection(
    const titan::bie::CartesianFullFmmSolverOperator &op) {
  const char *const destination =
      std::getenv("TITAN_P3C0_PACKED_PLAN_INSPECT_PATH");
  if (destination == nullptr || *destination == '\0')
    return;
  const auto inspection = op.packed_direct_plan_inspection_for_testing();
  const auto &stats = op.stats();
  std::ofstream out(destination, std::ios::out | std::ios::trunc);
  if (!out.is_open())
    throw std::runtime_error(
        "P3C0 test-only packed-plan inspector could not open output");
  out << std::setprecision(17);
  out << "available,target_batches,full_four_lane_batches,tail_batches,"
         "full_four_lane_task_count,tail_task_count,"
         "target_lanes,task_count,source_span_count,source_spans_per_task,"
         "tasks_per_batch_mean,tasks_per_batch_min,tasks_per_batch_p50,"
         "tasks_per_batch_p95,tasks_per_batch_max,"
         "source_panels_per_task_mean,source_panels_per_task_min,"
         "source_panels_per_task_p50,source_panels_per_task_p95,"
         "source_panels_per_task_max,metadata_bytes,total_pairs,"
         "plan_direct_panel_pairs,plan_direct_pairs,plan_tree_hash,"
         "plan_worklist_hash,leaf_size,theta\n";
  out << (inspection.available ? "true" : "false") << ','
      << inspection.target_batches << ',' << inspection.full_four_lane_batches
      << ',' << inspection.tail_batches << ','
      << inspection.full_four_lane_task_count << ','
      << inspection.tail_task_count << ',' << inspection.target_lanes << ','
      << inspection.task_count << ',' << inspection.source_span_count << ','
      << inspection.source_spans_per_task << ','
      << inspection.tasks_per_batch_mean << ','
      << inspection.tasks_per_batch_min << ','
      << inspection.tasks_per_batch_p50 << ','
      << inspection.tasks_per_batch_p95 << ','
      << inspection.tasks_per_batch_max << ','
      << inspection.source_panels_per_task_mean << ','
      << inspection.source_panels_per_task_min << ','
      << inspection.source_panels_per_task_p50 << ','
      << inspection.source_panels_per_task_p95 << ','
      << inspection.source_panels_per_task_max << ','
      << inspection.metadata_bytes << ',' << inspection.total_pairs << ','
      << stats.plan.direct_panel_pairs << ',' << stats.plan.direct_pairs << ','
      << stats.plan.tree_hash << ',' << stats.plan.worklist_hash << ','
      << stats.leaf_size << ',' << stats.opening_theta << '\n';
  if (!out)
    throw std::runtime_error(
        "P3C0 test-only packed-plan inspector could not write output");
}

bool p3d0_timing_value_valid(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

double p3d0_timing_tolerance(double parent_seconds,
                              std::size_t apply_count) noexcept {
  return std::max(1.0e-6 * static_cast<double>(apply_count),
                  1.0e-5 * std::max(0.0, parent_seconds));
}

void maybe_write_p3d0_direct_phase_attribution(
    const titan::bie::CartesianFullFmmSolverOperator &op,
    const Result &result) {
  const char *const destination =
      std::getenv("TITAN_P3D0_DIRECT_PHASE_PATH");
  if (destination == nullptr || *destination == '\0')
    return;

  const auto &stats = op.stats();
  const auto &timing = stats.testing_direct_phase_timing;
  const auto inspection = op.packed_direct_plan_inspection_for_testing();
  const auto fail = [](const char *reason) {
    throw std::runtime_error(
        std::string("P3D0 test-only direct phase sidecar invariant failed: ") +
        reason);
  };
  if (!timing.available || !timing.finite || !inspection.available)
    fail("testing_timing_or_packed_plan_unavailable");
  if (stats.direct_kernel != "packed-avx2-vector-exp-node-patch-corrected")
    fail("unexpected_direct_kernel");
  if (stats.apply_calls == 0 || timing.counted_apply_calls != stats.apply_calls ||
      result.fmm_local_gmres_matvec_calls != stats.apply_calls)
    fail("counted_apply_call_mismatch");
  if (!stats.finite_outputs || !stats.last_apply.finite_outputs ||
      stats.plan.direct_panel_pairs == 0 || inspection.total_pairs == 0 ||
      stats.plan.direct_panel_pairs != inspection.total_pairs ||
      timing.expected_direct_panel_pairs != timing.applied_direct_panel_pairs ||
      timing.expected_direct_panel_pairs !=
          stats.total_apply.expected_direct_panel_pairs ||
      timing.applied_direct_panel_pairs != stats.total_apply.direct_panel_pairs ||
      stats.total_apply.expected_direct_panel_pairs !=
          stats.total_apply.direct_panel_pairs)
    fail("direct_coverage_mismatch");
  const std::uint64_t apply_count =
      static_cast<std::uint64_t>(stats.apply_calls);
  if (apply_count != stats.apply_calls ||
      inspection.total_pairs >
          std::numeric_limits<std::uint64_t>::max() / apply_count ||
      timing.expected_direct_panel_pairs != inspection.total_pairs * apply_count)
    fail("direct_coverage_count_overflow_or_drift");
  if (result.bie_cartesian_full_fmm_tree_hash == 0 ||
      result.bie_cartesian_full_fmm_worklist_hash == 0 ||
      result.bie_cartesian_full_fmm_policy_hash == 0 ||
      result.bie_cartesian_full_fmm_tree_hash != stats.plan.tree_hash ||
      result.bie_cartesian_full_fmm_worklist_hash != stats.plan.worklist_hash ||
      result.bie_cartesian_full_fmm_direct_panel_pairs_applied_total !=
          timing.applied_direct_panel_pairs)
    fail("result_identity_or_coverage_drift");

  const std::array<double, 15> nonnegative_timing{
      timing.packed_unknown_cache_reorder_seconds,
      timing.packed_worker_scratch_prep_seconds,
      timing.packed_parallel_region_seconds,
      timing.packed_worker_metric_reduction_seconds,
      timing.packed_finite_scan_seconds,
      timing.packed_wrapper_residual_seconds,
      timing.packed_phase_sum_seconds,
      timing.packed_total_seconds,
      timing.node_patch_correction_seconds,
      timing.adapter_direct_total_seconds,
      timing.adapter_direct_residual_seconds,
      timing.adapter_coverage_bookkeeping_seconds,
      timing.adapter_residual_coverage_seconds,
      timing.adapter_direct_phase_sum_seconds,
      stats.total_apply.direct_seconds};
  for (const double value : nonnegative_timing) {
    if (!p3d0_timing_value_valid(value))
      fail("nonfinite_or_negative_timing");
  }
  if (timing.adapter_direct_total_seconds <= 0.0)
    fail("zero_adapter_direct_total");
  if (!std::isfinite(timing.packed_closure_seconds) ||
      !std::isfinite(timing.adapter_direct_closure_seconds))
    fail("nonfinite_timing_closure");
  const double packed_tolerance =
      p3d0_timing_tolerance(timing.packed_total_seconds, stats.apply_calls);
  const double adapter_tolerance = p3d0_timing_tolerance(
      timing.adapter_direct_total_seconds, stats.apply_calls);
  if (timing.packed_phase_sum_seconds >
          timing.packed_total_seconds + packed_tolerance ||
      std::abs(timing.packed_closure_seconds) > packed_tolerance ||
      timing.adapter_direct_phase_sum_seconds >
          timing.adapter_direct_total_seconds + adapter_tolerance ||
      std::abs(timing.adapter_direct_closure_seconds) > adapter_tolerance ||
      timing.adapter_direct_total_seconds != stats.total_apply.direct_seconds)
    fail("phase_parent_or_closure_mismatch");

  std::ofstream out(destination, std::ios::out | std::ios::trunc);
  if (!out.is_open())
    throw std::runtime_error(
        "P3D0 test-only direct phase sidecar could not open output");
  out << std::setprecision(17);
  out << "status,apply_count,stats_apply_calls,panels,iterations,matvec_calls,"
         "fmm_gmres_matvec_calls,direct_kernel,tree_hash,worklist_hash,"
         "policy_hash,plan_direct_leaf_pairs,plan_direct_panel_pairs,"
         "packed_target_batches,packed_task_count,packed_metadata_bytes,"
         "packed_total_pairs,direct_pairs_expected_total,"
         "direct_pairs_applied_total,packed_unknown_cache_reorder_seconds,"
         "packed_worker_scratch_prep_seconds,packed_parallel_region_seconds,"
         "packed_worker_metric_reduction_seconds,"
         "packed_finite_scan_seconds,packed_wrapper_residual_seconds,"
         "packed_phase_sum_seconds,packed_total_seconds,"
         "packed_closure_seconds,node_patch_correction_seconds,"
         "adapter_direct_total_seconds,adapter_direct_residual_seconds,"
         "adapter_coverage_bookkeeping_seconds,"
         "adapter_residual_coverage_seconds,"
         "adapter_direct_phase_sum_seconds,adapter_direct_closure_seconds,"
         "packed_unknown_cache_reorder_fraction,"
         "packed_worker_scratch_prep_fraction,"
         "packed_parallel_region_fraction,"
         "packed_worker_metric_reduction_fraction,"
         "packed_finite_scan_fraction,packed_wrapper_residual_fraction,"
         "node_patch_correction_fraction,adapter_direct_residual_fraction,"
         "adapter_coverage_bookkeeping_fraction,finite_outputs\n";
  const double direct_total = timing.adapter_direct_total_seconds;
  out << result.status << ',' << timing.counted_apply_calls << ','
      << stats.apply_calls << ',' << result.panel_count << ','
      << result.iterations
      << ',' << result.matvec_calls << ',' << result.fmm_local_gmres_matvec_calls
      << ',' << stats.direct_kernel << ',' << stats.plan.tree_hash << ','
      << stats.plan.worklist_hash << ','
      << result.bie_cartesian_full_fmm_policy_hash << ','
      << stats.plan.direct_pairs << ',' << stats.plan.direct_panel_pairs << ','
      << inspection.target_batches << ',' << inspection.task_count << ','
      << inspection.metadata_bytes << ',' << inspection.total_pairs << ','
      << timing.expected_direct_panel_pairs << ','
      << timing.applied_direct_panel_pairs << ','
      << timing.packed_unknown_cache_reorder_seconds << ','
      << timing.packed_worker_scratch_prep_seconds << ','
      << timing.packed_parallel_region_seconds << ','
      << timing.packed_worker_metric_reduction_seconds << ','
      << timing.packed_finite_scan_seconds << ','
      << timing.packed_wrapper_residual_seconds << ','
      << timing.packed_phase_sum_seconds << ',' << timing.packed_total_seconds
      << ',' << timing.packed_closure_seconds << ','
      << timing.node_patch_correction_seconds << ','
      << timing.adapter_direct_total_seconds << ','
      << timing.adapter_direct_residual_seconds << ','
      << timing.adapter_coverage_bookkeeping_seconds << ','
      << timing.adapter_residual_coverage_seconds << ','
      << timing.adapter_direct_phase_sum_seconds << ','
      << timing.adapter_direct_closure_seconds << ','
      << timing.packed_unknown_cache_reorder_seconds / direct_total << ','
      << timing.packed_worker_scratch_prep_seconds / direct_total << ','
      << timing.packed_parallel_region_seconds / direct_total << ','
      << timing.packed_worker_metric_reduction_seconds / direct_total << ','
      << timing.packed_finite_scan_seconds / direct_total << ','
      << timing.packed_wrapper_residual_seconds / direct_total << ','
      << timing.node_patch_correction_seconds / direct_total << ','
      << timing.adapter_direct_residual_seconds / direct_total << ','
      << timing.adapter_coverage_bookkeeping_seconds / direct_total << ','
      << (stats.finite_outputs ? "true" : "false") << '\n';
  if (!out)
    throw std::runtime_error(
        "P3D0 test-only direct phase sidecar could not write output");
}
#endif

void fmm_audit_hash_mix(std::uint64_t &hash, std::uint64_t value) noexcept {
  hash ^= value;
  hash *= kFmmAuditHashPrime;
}

std::uint64_t fmm_audit_probe_hash(const std::vector<double> &values,
                                   std::uint64_t tag) noexcept {
  std::uint64_t hash = kFmmAuditHashOffset;
  fmm_audit_hash_mix(hash, tag);
  fmm_audit_hash_mix(hash, static_cast<std::uint64_t>(values.size()));
  for (const double value : values)
    fmm_audit_hash_mix(hash, std::bit_cast<std::uint64_t>(value));
  return hash == 0 ? 1 : hash;
}

std::uint64_t
fmm_audit_target_hash(const std::vector<std::size_t> &targets) noexcept {
  std::uint64_t hash = kFmmAuditHashOffset;
  fmm_audit_hash_mix(hash, 0x54415247455453ULL); // "TARGETS"
  fmm_audit_hash_mix(hash, static_cast<std::uint64_t>(targets.size()));
  for (const std::size_t target : targets)
    fmm_audit_hash_mix(hash, static_cast<std::uint64_t>(target));
  return hash == 0 ? 1 : hash;
}

struct AttributionAggregate {
  double error_sq{0.0};
  double reference_sq{0.0};
  double max_abs_error{0.0};
  std::size_t count{0};
};

struct AttributionSummaryKey {
  std::string approximation_mode;
  std::string probe;
  std::string component;
  std::string region;
  std::string channel;

  bool operator<(const AttributionSummaryKey &other) const noexcept {
    return std::tie(approximation_mode, probe, component, region, channel) <
           std::tie(other.approximation_mode, other.probe, other.component,
                    other.region, other.channel);
  }
};

struct AttributionLevelKey {
  std::string approximation_mode;
  std::string probe;
  std::string component;
  std::string channel;
  int source_depth{0};
  std::string mac_bin;

  bool operator<(const AttributionLevelKey &other) const noexcept {
    return std::tie(approximation_mode, probe, component, channel, source_depth,
                    mac_bin) < std::tie(other.approximation_mode, other.probe,
                                        other.component, other.channel,
                                        other.source_depth, other.mac_bin);
  }
};

void add_attribution_value(AttributionAggregate &aggregate, double approximate,
                           double direct) noexcept {
  const double error = approximate - direct;
  aggregate.error_sq += error * error;
  aggregate.reference_sq += direct * direct;
  aggregate.max_abs_error = std::max(aggregate.max_abs_error, std::abs(error));
  ++aggregate.count;
}

double attribution_rel_rms(const AttributionAggregate &aggregate) noexcept {
  return std::sqrt(std::max(0.0, aggregate.error_sq)) /
         std::max(1.0e-300, std::sqrt(std::max(0.0, aggregate.reference_sq)));
}

std::string attribution_mac_bin(double rho) {
  if (rho < 0.2)
    return "[0.0;0.2)";
  if (rho < 0.3)
    return "[0.2;0.3)";
  if (rho < 0.4)
    return "[0.3;0.4)";
  if (rho < 0.5)
    return "[0.4;0.5)";
  if (rho < 0.6)
    return "[0.5;0.6)";
  return "[0.6;inf)";
}

void write_operator_attribution_outputs(
    const std::vector<titan::tabi::fmm_m2l::JcpOperatorAttributionResult>
        &results,
    double aggregate_audit_rel, double aggregate_direct_sq) {
  static constexpr std::array<const char *, 4> channel_names{"K1", "K2", "K3",
                                                             "K4"};
  std::ofstream samples("pbe_tabi_operator_attribution_samples.csv",
                        std::ios::out | std::ios::trunc);
  std::ofstream summary("pbe_tabi_operator_attribution_summary.csv",
                        std::ios::out | std::ios::trunc);
  std::ofstream levels("pbe_tabi_operator_attribution_levels.csv",
                       std::ios::out | std::ios::trunc);
  if (!samples.is_open() || !summary.is_open() || !levels.is_open()) {
    throw std::runtime_error(
        "PBE BIE: failed to open sampled-detail attribution outputs.");
  }
  samples
      << "approximation_mode,configured_expansion_order,implemented_far_moment_"
         "degree,probe,"
         "target_panel,row_component,region,channel,source_node,"
         "source_depth,source_panels,distance_A,source_radius_A,mac_ratio,"
         "expansion_distance_A,target_expansion_radius_A,source_expansion_"
         "radius_A,"
         "target_center_offset_A,source_center_offset_A,expansion_mac_ratio,"
         "approximate,direct,error,relative_error\n";

  std::map<AttributionSummaryKey, AttributionAggregate> aggregates;
  std::map<AttributionLevelKey, AttributionAggregate> level_aggregates;
  std::map<std::pair<std::string, std::string>, double> probe_error_sq;
  std::map<std::tuple<std::string, std::string, std::size_t>,
           std::pair<std::array<double, 4>, std::array<double, 4>>>
      target_totals;
  std::map<std::pair<std::string, std::string>,
           const titan::tabi::fmm_m2l::JcpOperatorAttributionResult *>
      result_by_probe;

  for (const auto &result : results) {
    result_by_probe[{result.approximation_mode, result.probe}] = &result;
    for (const auto &record : result.records) {
      auto &target_total = target_totals[{result.approximation_mode,
                                          record.probe, record.target_panel}];
      for (std::size_t channel = 0; channel < channel_names.size(); ++channel) {
        const std::string component = channel < 2 ? "phi" : "phin";
        const double approximate = record.approximate[channel];
        const double direct = record.direct[channel];
        const double error = approximate - direct;
        const double relative =
            std::abs(error) / std::max(1.0e-300, std::abs(direct));
        samples << result.approximation_mode << ','
                << result.configured_expansion_order << ','
                << result.implemented_far_moment_degree << ',' << record.probe
                << ',' << record.target_panel << ',' << component << ','
                << record.region << ',' << channel_names[channel] << ','
                << record.source_node << ',' << record.source_depth << ','
                << record.source_panels << ',' << record.distance << ','
                << record.source_radius << ',' << record.mac_ratio << ','
                << record.expansion_distance << ','
                << record.target_expansion_radius << ','
                << record.source_expansion_radius << ','
                << record.target_center_offset << ','
                << record.source_center_offset << ','
                << record.expansion_mac_ratio << ',' << approximate << ','
                << direct << ',' << error << ',' << relative << '\n';
        AttributionSummaryKey key{result.approximation_mode, record.probe,
                                  component, record.region,
                                  channel_names[channel]};
        add_attribution_value(aggregates[key], approximate, direct);
        probe_error_sq[{result.approximation_mode, record.probe}] +=
            error * error;
        target_total.first[channel] += approximate;
        target_total.second[channel] += direct;
        if (record.region == "far") {
          AttributionLevelKey level_key{result.approximation_mode,
                                        record.probe,
                                        component,
                                        channel_names[channel],
                                        record.source_depth,
                                        attribution_mac_bin(record.mac_ratio)};
          add_attribution_value(level_aggregates[level_key], approximate,
                                direct);
        }
      }
    }
  }

  for (const auto &[target_key, values] : target_totals) {
    const auto &[approximation_mode, probe, target_panel] = target_key;
    (void)target_panel;
    for (std::size_t component_index = 0; component_index < 2;
         ++component_index) {
      const std::size_t first = component_index == 0 ? 0 : 2;
      const double approximate = values.first[first] + values.first[first + 1];
      const double direct = values.second[first] + values.second[first + 1];
      add_attribution_value(aggregates[{approximation_mode, probe,
                                        component_index == 0 ? "phi" : "phin",
                                        "total", "aggregate"}],
                            approximate, direct);
      add_attribution_value(
          aggregates[{approximation_mode, probe, "both", "total", "aggregate"}],
          approximate, direct);
    }
  }

  summary
      << "approximation_mode,configured_expansion_order,implemented_far_moment_"
         "degree,probe,"
         "row_component,region,channel,samples,error_rms,reference_rms,"
         "relative_rms,max_abs_error,error_fraction,recomposition_rel_error,"
         "aggregate_audit_rel,reference_gap,moment_build_seconds,far_"
         "evaluation_seconds,"
         "m2l_seconds,l2p_seconds,local_target_leaves,local_m2l_pairs,"
         "fmm_local_mode,fmm_local_source_order,fmm_local_target_order,"
         "fmm_local_kernel_derivative_order,fmm_local_target_leaves,"
         "fmm_local_m2l_pairs,fmm_local_moment_seconds,fmm_local_m2l_seconds,"
         "fmm_local_l2p_seconds,fmm_local_finite_outputs,"
         "fmm_local_refine_mode,fmm_local_refine_tolerance,"
         "fmm_local_tail_ratio_before,fmm_local_tail_ratio_after,"
         "fmm_local_refined_cells,fmm_local_split_children,"
         "fmm_local_direct_fallback_panels,fmm_local_direct_fallback_pairs,"
         "fmm_local_far_pairs_before,"
         "fmm_local_far_pairs_after,fmm_local_refinement_seconds,"
         "fmm_local_plan_hash,fmm_local_plan_build_count,"
         "fmm_local_static_probe_count,fmm_local_refine_probe_mode,"
         "fmm_local_solution_probe_used,"
         "fmm_local_linf_max_abs_tail,"
         "fmm_local_linf_max_relative_tail,"
         "fmm_local_linf_outlier_panels,"
         "fmm_local_geometry_guard,fmm_local_geometry_theta,"
         "fmm_local_geometry_guard_rejected_cells,"
         "fmm_local_geometry_guard_direct_fallback_pairs,"
         "fmm_local_geometry_gap_min,fmm_local_geometry_ratio_max,"
         "fmm_local_cache_mode,"
         "fmm_local_cache_layout,fmm_local_cache_operator_bytes,"
         "fmm_local_cache_uncompressed_estimated_bytes,"
         "fmm_local_cache_compression_ratio,"
         "fmm_local_cache_far_pair_offsets,"
         "fmm_local_cache_operators,fmm_local_cache_bytes,"
         "fmm_local_cache_build_count,fmm_local_cache_hits,"
         "fmm_local_cache_tile_bytes,fmm_local_cache_tile_count,"
         "fmm_local_cache_tile_builds,fmm_local_cache_tile_seconds,"
         "fmm_local_cache_build_seconds,"
         "fmm_local_plan_seconds,fmm_local_plan_builder,"
         "fmm_local_plan_probe_moment_seconds,"
         "fmm_local_plan_pair_metric_seconds,"
         "fmm_local_plan_refinement_seconds,"
         "fmm_local_plan_pair_metric_evaluations,"
         "fmm_local_plan_pair_metric_cache_hits,"
         "fmm_local_plan_refinement_iterations,"
         "fmm_local_plan_metric_cache_peak_bytes,"
         "fmm_local_plan_parallel_target_leaves,"
         "fmm_local_cache_seconds,"
         "fmm_local_candidate_apply_seconds,"
         "fmm_local_candidate_repeat_count,"
         "fmm_local_candidate_median_apply_seconds,"
         "fmm_local_candidate_direct_reference_seconds,"
         "fmm_local_fallback_requested,fmm_local_fallback_effective,"
         "fmm_local_fallback_worklist_hash,"
         "fmm_local_fallback_worklist_build_count,"
         "fmm_local_fallback_tasks,fmm_local_fallback_metadata_bytes,"
         "fmm_local_fallback_base_near_pairs,"
         "fmm_local_fallback_refined_pairs,"
         "fmm_local_fallback_vectorized_sources,"
         "fmm_local_fallback_vectorized_lanes,"
         "fmm_local_fallback_scalar_self_pairs,"
         "fmm_local_fallback_seconds\n";
  for (const auto &[key, aggregate] : aggregates) {
    const auto *result =
        result_by_probe.at({key.approximation_mode, key.probe});
    const double error_rms =
        aggregate.count > 0 ? std::sqrt(aggregate.error_sq /
                                        static_cast<double>(aggregate.count))
                            : 0.0;
    const double reference_rms =
        aggregate.count > 0 ? std::sqrt(aggregate.reference_sq /
                                        static_cast<double>(aggregate.count))
                            : 0.0;
    const double fraction =
        probe_error_sq[{key.approximation_mode, key.probe}] > 0.0
            ? aggregate.error_sq /
                  probe_error_sq[{key.approximation_mode, key.probe}]
            : 0.0;
    const double attribution_audit_rel =
        key.probe == "solution" && key.component == "both" &&
                key.region == "total" && key.channel == "aggregate"
            ? std::sqrt(std::max(0.0, aggregate.error_sq)) /
                  std::max(1.0e-300,
                           std::sqrt(std::max(0.0, aggregate_direct_sq)))
            : 0.0;
    const double reference_gap =
        key.probe == "solution" && key.component == "both" &&
                key.region == "total" && key.channel == "aggregate" &&
                key.approximation_mode == "legacy_degree1"
            ? std::abs(attribution_audit_rel - aggregate_audit_rel)
            : 0.0;
    const bool local_p4 = result->approximation_mode.find("local_m2l_p4") == 0;
    summary << result->approximation_mode << ','
            << result->configured_expansion_order << ','
            << result->implemented_far_moment_degree << ',' << key.probe << ','
            << key.component << ',' << key.region << ',' << key.channel << ','
            << aggregate.count << ',' << error_rms << ',' << reference_rms
            << ',' << attribution_rel_rms(aggregate) << ','
            << aggregate.max_abs_error << ',' << fraction << ','
            << result->recomposition_rel_error << ',' << attribution_audit_rel
            << ',' << reference_gap << ',' << result->moment_build_seconds
            << ',' << result->far_evaluation_seconds << ','
            << result->m2l_seconds << ',' << result->l2p_seconds << ','
            << result->local_target_leaves << ',' << result->local_m2l_pairs
            << ',' << (local_p4 ? "audit-p4" : "off") << ','
            << (local_p4 ? result->local_source_order : 0) << ','
            << (local_p4 ? result->local_target_order : 0) << ','
            << (local_p4 ? result->local_kernel_derivative_order : 0) << ','
            << result->local_target_leaves << ',' << result->local_m2l_pairs
            << ',' << result->moment_build_seconds << ',' << result->m2l_seconds
            << ',' << result->l2p_seconds << ','
            << (result->finite_outputs ? "true" : "false") << ','
            << result->local_refine_mode << ','
            << result->local_refine_tolerance << ','
            << result->local_tail_ratio_before << ','
            << result->local_tail_ratio_after << ','
            << result->local_refined_cells << ','
            << result->local_split_children << ','
            << result->local_direct_fallback_panels << ','
            << result->local_direct_fallback_pairs << ','
            << result->local_far_pairs_before << ','
            << result->local_far_pairs_after << ','
            << result->local_refinement_seconds << ','
            << result->local_plan_hash << ',' << result->local_plan_build_count
            << ',' << result->local_static_probe_count << ','
            << result->local_refine_probe_mode << ','
            << (result->local_solution_probe_used ? "true" : "false") << ','
            << result->local_linf_max_abs_tail << ','
            << result->local_linf_max_relative_tail << ','
            << result->local_linf_outlier_panels << ','
            << result->local_geometry_guard << ','
            << result->local_geometry_theta << ','
            << result->local_geometry_guard_rejected_cells << ','
            << result->local_geometry_guard_direct_fallback_pairs << ','
            << result->local_geometry_gap_min << ','
            << result->local_geometry_ratio_max << ','
            << result->local_cache_mode << ',' << result->local_cache_layout
            << ',' << result->local_cache_operator_bytes << ','
            << result->local_cache_uncompressed_estimated_bytes << ','
            << result->local_cache_compression_ratio << ','
            << result->local_cache_far_pair_offsets << ','
            << result->local_cache_operators << ',' << result->local_cache_bytes
            << ',' << result->local_cache_build_count << ','
            << result->local_cache_hits << ',' << result->local_cache_tile_bytes
            << ',' << result->local_cache_tile_count << ','
            << result->local_cache_tile_builds << ','
            << result->local_cache_tile_seconds << ','
            << result->local_cache_seconds << ',' << result->local_plan_seconds
            << ',' << result->local_plan_builder << ','
            << result->local_plan_probe_moment_seconds << ','
            << result->local_plan_pair_metric_seconds << ','
            << result->local_plan_refinement_seconds << ','
            << result->local_plan_pair_metric_evaluations << ','
            << result->local_plan_pair_metric_cache_hits << ','
            << result->local_plan_refinement_iterations << ','
            << result->local_plan_metric_cache_peak_bytes << ','
            << result->local_plan_parallel_target_leaves << ','
            << result->local_cache_seconds << ','
            << result->local_candidate_apply_seconds << ','
            << result->local_candidate_repeat_count << ','
            << result->local_candidate_median_apply_seconds << ','
            << result->local_candidate_direct_reference_seconds << ','
            << result->local_fallback_requested << ','
            << result->local_fallback_effective << ','
            << result->local_fallback_worklist_hash << ','
            << result->local_fallback_worklist_build_count << ','
            << result->local_fallback_tasks << ','
            << result->local_fallback_metadata_bytes << ','
            << result->local_fallback_base_near_pairs << ','
            << result->local_fallback_refined_pairs << ','
            << result->local_fallback_vectorized_sources << ','
            << result->local_fallback_vectorized_lanes << ','
            << result->local_fallback_scalar_self_pairs << ','
            << result->local_fallback_seconds << '\n';
  }

  levels << "approximation_mode,configured_expansion_order,implemented_far_"
            "moment_degree,probe,"
            "row_component,channel,source_depth,mac_bin,samples,error_rms,"
            "reference_rms,relative_rms,max_abs_error\n";
  for (const auto &[key, aggregate] : level_aggregates) {
    const auto *result =
        result_by_probe.at({key.approximation_mode, key.probe});
    const double error_rms =
        aggregate.count > 0 ? std::sqrt(aggregate.error_sq /
                                        static_cast<double>(aggregate.count))
                            : 0.0;
    const double reference_rms =
        aggregate.count > 0 ? std::sqrt(aggregate.reference_sq /
                                        static_cast<double>(aggregate.count))
                            : 0.0;
    levels << result->approximation_mode << ','
           << result->configured_expansion_order << ','
           << result->implemented_far_moment_degree << ',' << key.probe << ','
           << key.component << ',' << key.channel << ',' << key.source_depth
           << ',' << key.mac_bin << ',' << aggregate.count << ',' << error_rms
           << ',' << reference_rms << ',' << attribution_rel_rms(aggregate)
           << ',' << aggregate.max_abs_error << '\n';
  }
}

struct FormulationConfig {
  std::string formulation{"jcp2013"};
  std::string energy_projection{"boundary"};
  double rhs_scale{1.0};
  double operator_contrast_scale{1.0};
  double boundary_energy_scale{1.0};
};

// kappa^2 = 2 N_A e^2 I / (eps_0 eps_r k_B T), so kappa scales as
// 1/sqrt(eps_r * T).  The 0.3291 A^-1 M^-1/2 prefactor is water at 298.15 K,
// which already contains eps_r = 78.5; without the dielectric factor below a
// run with any other solvent silently kept water's screening length.
double debye_kappa_inv_a(double salt_m, double temp_k,
                         double eps_out) noexcept {
  if (salt_m <= 0.0)
    return 0.0;
  constexpr double kReferenceEpsOut = 78.5;
  const double temp_scale = std::sqrt(298.15 / std::max(1.0, temp_k));
  const double diel_scale =
      std::sqrt(kReferenceEpsOut / std::max(1.0e-6, eps_out));
  return 0.3291 * std::sqrt(salt_m) * temp_scale * diel_scale;
}

double dot(const std::vector<double> &a, const std::vector<double> &b) {
  double value = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    value += a[i] * b[i];
  return value;
}

double l2_norm(const std::vector<double> &values) {
  return std::sqrt(std::max(0.0, dot(values, values)));
}

double rms_norm(const std::vector<double> &values) {
  if (values.empty())
    return 0.0;
  return l2_norm(values) / std::sqrt(static_cast<double>(values.size()));
}

std::size_t solver_workers(const TitanExecutor &executor,
                           std::size_t work_items) noexcept {
  if (work_items == 0)
    return 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(executor.capacity(), work_items));
}

double dot_parallel(const std::vector<double> &a, const std::vector<double> &b,
                    TitanExecutor &executor) {
  const std::size_t n = a.size();
  if (n < kParallelVectorThreshold || executor.capacity() <= 1)
    return dot(a, b);
  const std::size_t workers = solver_workers(executor, n);
  std::vector<double> partial(workers, 0.0);
  executor.parallel_for(
      n, workers, kParallelVectorThreshold,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        double value = 0.0;
        for (std::size_t i = begin; i < end; ++i)
          value += a[i] * b[i];
        partial[worker] = value;
      });
  return std::accumulate(partial.begin(), partial.end(), 0.0);
}

double l2_norm_parallel(const std::vector<double> &values,
                        TitanExecutor &executor) {
  return std::sqrt(std::max(0.0, dot_parallel(values, values, executor)));
}

void parallel_for_each_index(std::size_t n, TitanExecutor &executor,
                             const std::function<void(std::size_t)> &fn) {
  if (n < kParallelVectorThreshold || executor.capacity() <= 1) {
    for (std::size_t i = 0; i < n; ++i)
      fn(i);
    return;
  }
  const std::size_t workers = solver_workers(executor, n);
  executor.parallel_for(n, workers, kParallelVectorThreshold,
                        [&](std::size_t, std::size_t begin, std::size_t end) {
                          for (std::size_t i = begin; i < end; ++i)
                            fn(i);
                        });
}

double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) noexcept {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::string normalize_jcp_direct_kernel_name(const std::string &value) {
  if (value == "soa-avx2" || value == "fused-soa-avx2")
    return value;
  return "baseline";
}

void refresh_matvec_cost_summary(Result &result) noexcept {
  const double traversal_seconds = result.tree_traversal_ms / 1000.0;
  result.leaf_panels_per_second =
      traversal_seconds > 0.0
          ? static_cast<double>(result.leaf_panel_interactions) /
                traversal_seconds
          : 0.0;
  result.clean_leaf_panels_per_second =
      result.leaf_pair_batch_clean_source_panels_per_second;
  result.tree_traversal_fraction =
      result.matvec_ms_total > 0.0
          ? result.tree_traversal_ms / result.matvec_ms_total
          : 0.0;
}

std::size_t current_working_set_bytes() noexcept {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
    return static_cast<std::size_t>(counters.WorkingSetSize);
#else
  FILE *file = std::fopen("/proc/self/status", "r");
  if (file != nullptr) {
    char line[256];
    char label[64];
    long value_kb = 0;
    while (std::fgets(line, sizeof(line), file) != nullptr) {
      if (std::sscanf(line, "%63s %ld", label, &value_kb) == 2 &&
          std::string(label) == "VmRSS:") {
        std::fclose(file);
        return static_cast<std::size_t>(std::max<long>(0, value_kb)) * 1024u;
      }
    }
    std::fclose(file);
  }
#endif
  return 0;
}

std::string csv_escape_progress(const std::string &value) {
  bool needs_quotes = false;
  for (char c : value) {
    if (c == ',' || c == '"' || c == '\n' || c == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes)
    return value;
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (char c : value) {
    if (c == '"')
      escaped.push_back('"');
    escaped.push_back(c);
  }
  escaped.push_back('"');
  return escaped;
}

class LongRunProgressWriter {
public:
  LongRunProgressWriter(const Options &options, std::size_t threads)
      : enabled_(options.progress != "off"),
        interval_(std::chrono::seconds(
            std::clamp(options.progress_interval_seconds, 1, 300))),
        start_(std::chrono::steady_clock::now()),
        last_write_(start_ - interval_), threads_(threads) {
    if (!enabled_)
      return;
    out_.open("pbe_tabi_longrun_progress.csv", std::ios::out | std::ios::trunc);
    if (!out_.is_open()) {
      enabled_ = false;
      return;
    }
    out_ << "elapsed_seconds,phase,gmres_iter,restart_iter,matvec_calls,"
            "residual,gmres_restart_requested,gmres_restart_effective,"
            "best_residual,best_residual_iter,final_residual,"
            "best_preconditioned_residual,final_preconditioned_residual,"
            "true_residual_abs,true_residual_rel,"
            "true_residual_probe_mode,true_residual_probe_interval,"
            "true_residual_probe_calls,true_residual_probe_seconds,"
            "true_residual_probe_scope,true_residual_sample_count,"
            "sampled_true_residual_rel,sampled_true_residual_rows,"
            "sampled_probe_calls,sampled_probe_seconds,"
            "full_true_residual_confirm_calls,full_probe_seconds,"
            "panels,unknown_count,taylor_order,tree_nodes,"
            "interaction_mode,tree_builder,node_bounds_mode,mac_safety,"
            "mac_gap_factor,expansion_center,"
            "fragmentation_safety,moment_construction,moment_sum,"
            "convergence_audit,convergence_safety,"
            "convergence_rho_cap,mac_aspect_cap,"
            "convergence_estimator,convergence_tail_threshold,"
            "convergence_estimator_error_cap,"
            "convergence_unstable_rho_cap,"
            "convergence_calibration,"
            "convergence_dielectric_amp,"
            "convergence_estimator_model,"
            "convergence_channel_calibration,"
            "convergence_split_policy,"
            "convergence_split_selectivity,"
            "convergence_split_absolute_floor,"
            "convergence_overreject_audit,"
            "convergence_overreject_detail,"
            "convergence_shape_audit,"
            "convergence_kernel_response_audit,"
            "channel_missed_semantics,"
            "accepted_miss_diagnostics,"
            "accepted_miss_absolute_floor,"
            "convergence_k3_accepted_rho_cap,"
            "mac_distance_policy,mac_surface_margin_factor,"
            "mac_surface_margin_policy,"
            "mac_surface_margin_aspect_threshold,"
            "mac_surface_margin_rho_trigger,"
            "mac_surface_margin_soft_aspect_threshold,"
            "mac_surface_margin_soft_rho_trigger,"
            "mac_surface_margin_soft_factor,"
            "convergence_estimator_samples,"
            "convergence_estimator_unstable_far_count,"
            "convergence_estimator_unstable_far_panels,"
            "convergence_estimator_split_rejects,"
            "convergence_estimator_split_reject_panels,"
            "convergence_low_abs_reaccepted_far_count,"
            "convergence_low_abs_reaccepted_far_panels,"
            "convergence_estimator_tail_ratio_max,"
            "convergence_estimator_tail_ratio_mean,"
            "convergence_estimator_error_max,"
            "convergence_estimator_error_mean,"
            "convergence_estimator_abs_error_max,"
            "convergence_estimator_abs_error_mean,"
            "convergence_tail_norm_p_max,"
            "convergence_tail_norm_p_mean,"
            "convergence_tail_norm_pm1_max,"
            "convergence_tail_norm_pm1_mean,"
            "convergence_effective_jcp_amp_max,"
            "convergence_effective_jcp_amp_mean,"
            "convergence_dielectric_amp_max,"
            "convergence_dielectric_amp_mean,"
            "convergence_k1_effective_jcp_amp_max,"
            "convergence_k2_effective_jcp_amp_max,"
            "convergence_k3_effective_jcp_amp_max,"
            "convergence_k4_effective_jcp_amp_max,"
            "convergence_k1_effective_jcp_amp_mean,"
            "convergence_k2_effective_jcp_amp_mean,"
            "convergence_k3_effective_jcp_amp_mean,"
            "convergence_k4_effective_jcp_amp_mean,"
            "convergence_k1_estimated_error_max,"
            "convergence_k2_estimated_error_max,"
            "convergence_k3_estimated_error_max,"
            "convergence_k4_estimated_error_max,"
            "convergence_k1_unstable_far_count,"
            "convergence_k2_unstable_far_count,"
            "convergence_k3_unstable_far_count,"
            "convergence_k4_unstable_far_count,"
            "convergence_k1_unstable_far_panels,"
            "convergence_k2_unstable_far_panels,"
            "convergence_k3_unstable_far_panels,"
            "convergence_k4_unstable_far_panels,"
            "convergence_k1_split_rejects,"
            "convergence_k2_split_rejects,"
            "convergence_k3_split_rejects,"
            "convergence_k4_split_rejects,"
            "convergence_k1_split_reject_panels,"
            "convergence_k2_split_reject_panels,"
            "convergence_k3_split_reject_panels,"
            "convergence_k4_split_reject_panels,"
            "convergence_k1_low_abs_reaccepted,"
            "convergence_k2_low_abs_reaccepted,"
            "convergence_k3_low_abs_reaccepted,"
            "convergence_k4_low_abs_reaccepted,"
            "convergence_k1_low_abs_reaccepted_panels,"
            "convergence_k2_low_abs_reaccepted_panels,"
            "convergence_k3_low_abs_reaccepted_panels,"
            "convergence_k4_low_abs_reaccepted_panels,"
            "convergence_k1_high_error_caught,"
            "convergence_k2_high_error_caught,"
            "convergence_k3_high_error_caught,"
            "convergence_k4_high_error_caught,"
            "convergence_k1_high_error_missed,"
            "convergence_k2_high_error_missed,"
            "convergence_k3_high_error_missed,"
            "convergence_k4_high_error_missed,"
            "convergence_k1_high_error_caught_by_split,"
            "convergence_k2_high_error_caught_by_split,"
            "convergence_k3_high_error_caught_by_split,"
            "convergence_k4_high_error_caught_by_split,"
            "convergence_estimator_high_error_caught,"
            "convergence_estimator_high_error_missed,"
            "convergence_estimator_high_error_caught_by_split,"
            "convergence_estimator_phantom_high_error_count,"
            "convergence_estimator_phantom_high_error_abs_max,"
            "convergence_overreject_count,"
            "convergence_overreject_panels,"
            "convergence_overreject_caught_count,"
            "convergence_overreject_caught_panels,"
            "convergence_overreject_abs_error_max,"
            "convergence_overreject_abs_error_mean,"
            "convergence_overreject_abs_error_p95,"
            "convergence_overreject_abs_error_p99,"
            "convergence_overreject_estimated_abs_error_max,"
            "convergence_overreject_estimated_abs_error_mean,"
            "convergence_overreject_required_scale_max,"
            "convergence_overreject_required_scale_p95,"
            "convergence_overreject_required_scale_p99,"
            "convergence_k1_overreject_count,"
            "convergence_k2_overreject_count,"
            "convergence_k3_overreject_count,"
            "convergence_k4_overreject_count,"
            "convergence_k1_overreject_caught_count,"
            "convergence_k2_overreject_caught_count,"
            "convergence_k3_overreject_caught_count,"
            "convergence_k4_overreject_caught_count,"
            "convergence_k1_overreject_panels,convergence_k2_overreject_panels,"
            "convergence_k3_overreject_panels,convergence_k4_overreject_panels,"
            "convergence_k1_overreject_abs_error_max,"
            "convergence_k2_overreject_abs_error_max,"
            "convergence_k3_overreject_abs_error_max,"
            "convergence_k4_overreject_abs_error_max,"
            "convergence_k1_overreject_abs_error_mean,"
            "convergence_k2_overreject_abs_error_mean,"
            "convergence_k3_overreject_abs_error_mean,"
            "convergence_k4_overreject_abs_error_mean,"
            "convergence_k1_overreject_required_scale_max,"
            "convergence_k2_overreject_required_scale_max,"
            "convergence_k3_overreject_required_scale_max,"
            "convergence_k4_overreject_required_scale_max,"
            "convergence_k1_overreject_required_scale_p99,"
            "convergence_k2_overreject_required_scale_p99,"
            "convergence_k3_overreject_required_scale_p99,"
            "convergence_k4_overreject_required_scale_p99,"
            "phantom_high_error_k1_count,phantom_high_error_k2_count,"
            "phantom_high_error_k3_count,phantom_high_error_k4_count,"
            "phantom_high_error_k1_abs_max,"
            "phantom_high_error_k2_abs_max,"
            "phantom_high_error_k3_abs_max,"
            "phantom_high_error_k4_abs_max,"
            "accepted_miss_worst_channel,"
            "accepted_miss_k1_count,accepted_miss_k2_count,"
            "accepted_miss_k3_count,accepted_miss_k4_count,"
            "accepted_miss_k1_panels,accepted_miss_k2_panels,"
            "accepted_miss_k3_panels,accepted_miss_k4_panels,"
            "accepted_miss_k1_rho_min,accepted_miss_k2_rho_min,"
            "accepted_miss_k3_rho_min,accepted_miss_k4_rho_min,"
            "accepted_miss_k1_rho_mean,accepted_miss_k2_rho_mean,"
            "accepted_miss_k3_rho_mean,accepted_miss_k4_rho_mean,"
            "accepted_miss_k1_rho_max,accepted_miss_k2_rho_max,"
            "accepted_miss_k3_rho_max,accepted_miss_k4_rho_max,"
            "accepted_miss_k1_tail_ratio_mean,"
            "accepted_miss_k2_tail_ratio_mean,"
            "accepted_miss_k3_tail_ratio_mean,"
            "accepted_miss_k4_tail_ratio_mean,"
            "accepted_miss_k1_tail_ratio_max,"
            "accepted_miss_k2_tail_ratio_max,"
            "accepted_miss_k3_tail_ratio_max,"
            "accepted_miss_k4_tail_ratio_max,"
            "accepted_miss_k1_estimated_error_mean,"
            "accepted_miss_k2_estimated_error_mean,"
            "accepted_miss_k3_estimated_error_mean,"
            "accepted_miss_k4_estimated_error_mean,"
            "accepted_miss_k1_estimated_error_max,"
            "accepted_miss_k2_estimated_error_max,"
            "accepted_miss_k3_estimated_error_max,"
            "accepted_miss_k4_estimated_error_max,"
            "convergence_k3_rho_cap_rejects,"
            "convergence_k3_rho_cap_reject_panels,"
            "expansion_audit,expansion_audit_channel,"
            "far_geometric_unsafe_count,far_geometric_unsafe_panels,"
            "far_geometric_audit_samples,mac_physical_gap_rejects,"
            "mac_convergence_rho_rejects,"
            "mac_convergence_aspect_rejects,"
            "mac_convergence_rho_reject_panels,"
            "mac_convergence_aspect_reject_panels,"
            "mac_surface_margin_rejects,"
            "mac_surface_margin_reject_panels,"
            "mac_surface_margin_applied,"
            "mac_surface_margin_skipped,"
            "mac_surface_margin_adaptive_rejects,"
            "mac_surface_margin_adaptive_reject_panels,"
            "mac_surface_margin_skipped_accepted_far,"
            "mac_surface_margin_hard_applied,"
            "mac_surface_margin_soft_applied,"
            "mac_surface_margin_hard_rejects,"
            "mac_surface_margin_hard_reject_panels,"
            "mac_surface_margin_soft_rejects,"
            "mac_surface_margin_soft_reject_panels,"
            "fragmented_far_count,fragmented_far_panels,"
            "fragmented_split_rejects,"
            "fragmentation_center_gap_ratio_max,"
            "fragmentation_center_gap_ratio_mean,"
            "moment_audit,moment_audit_nodes,"
            "moment_translation_rel_error_max,"
            "moment_translation_rel_error_mean,"
            "moment_translation_offset_max,"
            "moment_translation_offset_mean,"
            "moment_cancellation_ratio_max,"
            "moment_cancellation_ratio_mean,"
            "moment_dynamic_range_max,"
            "far_expansion_audit_pairs,far_expansion_audit_panels,"
            "far_expansion_direct_seconds,"
            "far_expansion_rel_error_max,far_expansion_rel_error_mean,"
            "far_expansion_rel_error_p95,far_expansion_rel_error_p99,"
            "far_expansion_worst_channel,"
            "far_expansion_k1_rel_error_max,"
            "far_expansion_k2_rel_error_max,"
            "far_expansion_k3_rel_error_max,"
            "far_expansion_k4_rel_error_max,"
            "far_expansion_k1_rel_error_mean,"
            "far_expansion_k2_rel_error_mean,"
            "far_expansion_k3_rel_error_mean,"
            "far_expansion_k4_rel_error_mean,"
            "far_expansion_k1_rel_error_p99,"
            "far_expansion_k2_rel_error_p99,"
            "far_expansion_k3_rel_error_p99,"
            "far_expansion_k4_rel_error_p99,"
            "far_expansion_mean_gate_pass,"
            "far_convergence_rho_max,far_convergence_rho_mean,"
            "node_aspect_ratio_max,node_aspect_ratio_mean,"
            "accepted_far_aspect_ratio_max,"
            "accepted_far_aspect_ratio_mean,"
            "leaf_panel_interactions,leaf_tile_panel_interactions,"
            "leaf_residual_panel_interactions,far_node_interactions,"
            "leaf_mac_candidates,leaf_mac_accepted,leaf_mac_rejected,"
            "leaf_mac_far_panels,leaf_mac_direct_panels_saved_estimate,"
            "matvec_cost_profile,leaf_panels_per_second,"
            "clean_leaf_panels_per_second,tree_traversal_fraction,"
            "leaf_work_profile,leaf_work_mac_terminal_panels,"
            "leaf_work_physical_gap_panels,"
            "leaf_work_fragmentation_panels,"
            "leaf_work_convergence_panels,"
            "leaf_work_convergence_estimator_panels,"
            "leaf_work_convergence_rho_panels,"
            "leaf_work_convergence_aspect_panels,"
            "leaf_work_surface_margin_panels,"
            "leaf_work_surface_margin_hard_panels,"
            "leaf_work_surface_margin_soft_panels,"
            "leaf_work_k3_rho_cap_panels,"
            "leaf_work_singular_fallback_panels,"
            "leaf_work_scalar_tail_panels,"
            "leaf_work_profile_accounted_panels,"
            "leaf_work_mac_terminal_frac,"
            "leaf_work_convergence_frac,"
            "leaf_work_surface_margin_frac,"
            "leaf_work_singular_fallback_frac,"
            "leaf_work_scalar_tail_frac,"
            "cluster_policy,target_batch_size,"
            "leaf_pair_min_source_panels,leaf_pair_min_targets,"
            "leaf_pair_batch_interactions,leaf_pair_batch_targets,"
            "leaf_pair_batch_source_panels,"
            "leaf_pair_batch_residual_panels,"
            "leaf_pair_batch_fallbacks,leaf_pair_batch_seconds,"
            "leaf_pair_batch_kernel,leaf_pair_batch_vectorized_sources,"
            "leaf_pair_batch_vector_lanes,"
            "leaf_pair_batch_pairs_per_second,"
            "leaf_kernel_profile,jcp_direct_kernel,"
            "leaf_kernel_scalar_tasks,leaf_kernel_batch_tasks,"
            "leaf_kernel_source_panel_evals,"
            "leaf_kernel_scalar_seconds,"
            "leaf_kernel_target_avx2_seconds,"
            "leaf_kernel_tail_seconds,"
            "leaf_pair_batch_soa_sources,"
            "leaf_pair_batch_soa_seconds,"
            "leaf_pair_batch_scalar_tail_seconds,"
            "leaf_pair_batch_clean_source_panels_per_second,"
            "leaf_pair_selection_profile,leaf_pair_probe_cache,"
            "leaf_pair_target_group,leaf_pair_max_lookahead,"
            "leaf_pair_directness_probe,leaf_pair_directness_cache,"
            "leaf_pair_directness_verify,"
            "leaf_pair_selection_candidate_scans,"
            "leaf_pair_selection_clean_rejects,"
            "leaf_pair_selection_probe_calls,"
            "leaf_pair_selection_probe_node_visits,"
            "leaf_pair_selection_probe_seconds,"
            "leaf_pair_selection_seconds,"
            "leaf_pair_selection_probe_cache_hits,"
            "leaf_pair_selection_probe_cache_misses,"
            "leaf_pair_selection_insufficient_targets,"
            "leaf_pair_selection_same_target_leaf_rejects,"
            "leaf_pair_directness_ancestor_probe_calls,"
            "leaf_pair_directness_ancestor_probe_node_visits,"
            "leaf_pair_directness_ancestor_probe_seconds,"
            "leaf_pair_directness_pair_cache_hits,"
            "leaf_pair_directness_pair_cache_misses,"
            "leaf_pair_directness_all_direct_proofs,"
            "leaf_pair_directness_mixed_fallbacks,"
            "leaf_pair_directness_verify_checks,"
            "leaf_pair_directness_verify_mismatches,"
            "leaf_pair_scheduler,leaf_pair_worklist_max_active,"
            "leaf_pair_worklist_verify,"
            "leaf_pair_worklist_tasks,leaf_pair_worklist_groups,"
            "leaf_pair_worklist_batched_groups,"
            "leaf_pair_worklist_scalar_flush_tasks,"
            "leaf_pair_worklist_forced_flushes,"
            "leaf_pair_worklist_active_groups_max,"
            "leaf_pair_worklist_seconds,"
            "leaf_pair_worklist_verify_checks,"
            "leaf_pair_worklist_verify_mismatches,"
            "leaf_block_verify,leaf_block_target_leaves,"
            "leaf_block_targets,leaf_block_source_groups,"
            "leaf_block_batched_groups,leaf_block_scalar_tail_tasks,"
            "leaf_block_group_seconds,leaf_block_flush_seconds,"
            "leaf_block_writeback_seconds,leaf_block_verify_checks,"
            "leaf_block_verify_mismatches,"
            "singular_near_interactions,preconditioner_effective,"
            "preconditioner_block_panel_cap,"
            "preconditioner_overlap_panel_cap,"
            "preconditioner_ilu_neighbor_cap_requested,"
            "preconditioner_ilu_neighbor_cap,"
            "preconditioner_ilu_drop_tol,"
            "preconditioner_ilu_regularization,"
            "preconditioner_ilu_regularization_max,"
            "preconditioner_ilu_regularization_selected,"
            "preconditioner_ilu_regularization_absolute_max,"
            "preconditioner_ilu_regularization_attempts,"
            "preconditioner_ilu_regularization_rejected_attempts,"
            "preconditioner_ilu_regularization_shifted_rows,"
            "preconditioner_ilu_unshifted_failure_stage,"
            "preconditioner_memory_budget_bytes,"
            "preconditioner_estimated_bytes,"
            "preconditioner_valid_blocks,preconditioner_valid_panels,"
            "preconditioner_ill_conditioned_blocks,"
            "preconditioner_quality_rejected_blocks,"
            "preconditioner_quality_probe_max_gain,"
            "preconditioner_quality_probe_rel_residual,"
            "preconditioner_pivot_ratio_min,"
            "preconditioner_factor_growth_max,"
            "preconditioner_pivot_ratio_min_row,"
            "preconditioner_pivot_ratio_min_column,"
            "preconditioner_factor_growth_row,"
            "preconditioner_factor_growth_column,"
            "preconditioner_quality_failure_stage,"
            "preconditioner_quality_failure_row,"
            "preconditioner_quality_failure_column,"
            "preconditioner_quality_failure_probe,"
            "preconditioner_quality_failure_probe_gain,"
            "preconditioner_quality_failure_probe_rel_residual,"
            "preconditioner_quality_status,"
            "preconditioner_setup_seconds,"
            "preconditioner_graph_setup_seconds,"
            "preconditioner_graph_cache_reused,"
            "preconditioner_near_assembly_setup_seconds,"
            "preconditioner_ilu_factor_setup_seconds,"
            "preconditioner_apply_seconds,"
            "preconditioner_apply_calls,"
            "preconditioner_sparse_nnz_blocks,"
            "preconditioner_sparse_avg_degree,"
            "preconditioner_sparse_max_degree,"
            "preconditioner_coarse_aggregate_cap,"
            "preconditioner_coarse_aggregates,"
            "preconditioner_coarse_dimension,"
            "preconditioner_coarse_base,"
            "preconditioner_coarse_matrix_bytes,"
            "preconditioner_coarse_pivot_ratio,"
            "preconditioner_coarse_status,"
            "preconditioner_coarse_setup_matvec_calls,"
            "preconditioner_coarse_setup_seconds,"
            "preconditioner_coarse_moments_setup_seconds,"
            "preconditioner_coarse_far_setup_seconds,"
            "preconditioner_coarse_audit_setup_seconds,"
            "preconditioner_coarse_apply_seconds,"
            "preconditioner_coarse_apply_calls,"
            "preconditioner_coarse_rp_identity_error_max,"
            "preconditioner_coarse_verify_rel_error_max,"
            "gmres_orthogonalization_seconds,gmres_preconditioner_seconds,"
            "fmm_local_gmres_active,fmm_local_gmres_matvec_calls,"
            "fmm_local_gmres_apply_seconds_total,"
            "fmm_local_gmres_apply_seconds_mean,"
            "fmm_local_moment_seconds_total,fmm_local_m2l_seconds_total,"
            "fmm_local_fallback_seconds_total,fmm_local_l2p_seconds_total,"
            "threads,"
            "rss_bytes,status\n";
    out_.flush();
  }

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  void write(const std::string &phase, const Result &result,
             const TreecodeStats *stats, std::size_t matvec_calls,
             int gmres_iter, int restart_iter, double residual,
             const std::string &status, bool force) {
    if (!enabled_)
      return;
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - last_write_ < interval_)
      return;
    last_write_ = now;

    const std::size_t tree_nodes = stats != nullptr && stats->tree_nodes > 0
                                       ? stats->tree_nodes
                                       : result.tree_nodes;
    const std::size_t leaf_panel = stats != nullptr
                                       ? stats->leaf_panel_interactions
                                       : result.leaf_panel_interactions;
    const std::size_t leaf_tile_panel =
        stats != nullptr ? stats->leaf_tile_panel_interactions
                         : result.leaf_tile_panel_interactions;
    const std::size_t leaf_residual =
        stats != nullptr ? stats->leaf_residual_panel_interactions
                         : result.leaf_residual_panel_interactions;
    const std::size_t far_nodes = stats != nullptr
                                      ? stats->far_node_interactions
                                      : result.far_node_interactions;
    const std::size_t leaf_mac_candidates = stats != nullptr
                                                ? stats->leaf_mac_candidates
                                                : result.leaf_mac_candidates;
    const std::size_t leaf_mac_accepted =
        stats != nullptr ? stats->leaf_mac_accepted : result.leaf_mac_accepted;
    const std::size_t leaf_mac_rejected =
        stats != nullptr ? stats->leaf_mac_rejected : result.leaf_mac_rejected;
    const std::size_t leaf_mac_far_panels = stats != nullptr
                                                ? stats->leaf_mac_far_panels
                                                : result.leaf_mac_far_panels;
    const std::size_t leaf_mac_direct_panels_saved_estimate =
        stats != nullptr ? stats->leaf_mac_direct_panels_saved_estimate
                         : result.leaf_mac_direct_panels_saved_estimate;
    const double traversal_ms =
        stats != nullptr ? stats->tree_traversal_ms : result.tree_traversal_ms;
    const double leaf_panels_per_second =
        traversal_ms > 0.0
            ? static_cast<double>(leaf_panel) / (traversal_ms / 1000.0)
            : result.leaf_panels_per_second;
    const std::string matvec_cost_profile = result.matvec_cost_profile;
    const double tree_traversal_fraction =
        result.matvec_ms_total > 0.0 ? traversal_ms / result.matvec_ms_total
                                     : result.tree_traversal_fraction;
    const std::size_t singular = stats != nullptr
                                     ? stats->singular_near_interactions
                                     : result.singular_near_interactions;
    const std::string tree_builder =
        stats != nullptr ? stats->tree_builder : result.tree_builder;
    const std::string interaction_mode =
        stats != nullptr ? stats->interaction_mode : result.interaction_mode;
    const std::string node_bounds =
        stats != nullptr ? stats->node_bounds_mode : result.node_bounds_mode;
    const std::string mac_safety =
        stats != nullptr ? stats->mac_safety : result.mac_safety;
    const std::string expansion_center =
        stats != nullptr ? stats->expansion_center : result.expansion_center;
    const std::string fragmentation_safety = stats != nullptr
                                                 ? stats->fragmentation_safety
                                                 : result.fragmentation_safety;
    const std::string expansion_audit =
        stats != nullptr ? stats->expansion_audit : result.expansion_audit;
    const double mac_gap_factor =
        stats != nullptr ? stats->mac_gap_factor : result.mac_gap_factor;
    const std::size_t unsafe_count = stats != nullptr
                                         ? stats->far_geometric_unsafe_count
                                         : result.far_geometric_unsafe_count;
    const std::size_t unsafe_panels = stats != nullptr
                                          ? stats->far_geometric_unsafe_panels
                                          : result.far_geometric_unsafe_panels;
    const std::size_t gap_samples = stats != nullptr
                                        ? stats->far_geometric_audit_samples
                                        : result.far_geometric_audit_samples;
    const std::size_t gap_rejects = stats != nullptr
                                        ? stats->mac_physical_gap_rejects
                                        : result.mac_physical_gap_rejects;
    const std::size_t fragmented_count = stats != nullptr
                                             ? stats->fragmented_far_count
                                             : result.fragmented_far_count;
    const std::size_t fragmented_panels = stats != nullptr
                                              ? stats->fragmented_far_panels
                                              : result.fragmented_far_panels;
    const std::size_t fragmented_rejects =
        stats != nullptr ? stats->fragmented_split_rejects
                         : result.fragmented_split_rejects;
    const double fragmentation_gap_max =
        stats != nullptr ? stats->fragmentation_center_gap_ratio_max
                         : result.fragmentation_center_gap_ratio_max;
    const double fragmentation_gap_mean =
        stats != nullptr ? stats->fragmentation_center_gap_ratio_mean
                         : result.fragmentation_center_gap_ratio_mean;
    const std::string moment_audit =
        stats != nullptr ? stats->moment_audit : result.moment_audit;
    const std::string moment_construction = stats != nullptr
                                                ? stats->moment_construction
                                                : result.moment_construction;
    const std::string moment_sum =
        stats != nullptr ? stats->moment_sum : result.moment_sum;
    const std::string convergence_audit =
        stats != nullptr ? stats->convergence_audit : result.convergence_audit;
    const std::string convergence_safety = stats != nullptr
                                               ? stats->convergence_safety
                                               : result.convergence_safety;
    const double convergence_rho_cap = stats != nullptr
                                           ? stats->convergence_rho_cap
                                           : result.convergence_rho_cap;
    const double mac_aspect_cap =
        stats != nullptr ? stats->mac_aspect_cap : result.mac_aspect_cap;
    const std::string convergence_estimator =
        stats != nullptr ? stats->convergence_estimator
                         : result.convergence_estimator;
    const double convergence_tail_threshold =
        stats != nullptr ? stats->convergence_tail_threshold
                         : result.convergence_tail_threshold;
    const double convergence_estimator_error_cap =
        stats != nullptr ? stats->convergence_estimator_error_cap
                         : result.convergence_estimator_error_cap;
    const double convergence_unstable_rho_cap =
        stats != nullptr ? stats->convergence_unstable_rho_cap
                         : result.convergence_unstable_rho_cap;
    const std::string convergence_calibration =
        stats != nullptr ? stats->convergence_calibration
                         : result.convergence_calibration;
    const std::string convergence_dielectric_amp =
        stats != nullptr ? stats->convergence_dielectric_amp
                         : result.convergence_dielectric_amp;
    const std::string convergence_estimator_model =
        stats != nullptr ? stats->convergence_estimator_model
                         : result.convergence_estimator_model;
    const std::string convergence_channel_calibration =
        stats != nullptr ? stats->convergence_channel_calibration
                         : result.convergence_channel_calibration;
    const std::string convergence_split_policy =
        stats != nullptr ? stats->convergence_split_policy
                         : result.convergence_split_policy;
    const std::string convergence_split_selectivity =
        stats != nullptr ? stats->convergence_split_selectivity
                         : result.convergence_split_selectivity;
    const double convergence_split_absolute_floor =
        stats != nullptr ? stats->convergence_split_absolute_floor
                         : result.convergence_split_absolute_floor;
    const std::string convergence_overreject_audit =
        stats != nullptr ? stats->convergence_overreject_audit
                         : result.convergence_overreject_audit;
    const std::string convergence_overreject_detail =
        stats != nullptr ? stats->convergence_overreject_detail
                         : result.convergence_overreject_detail;
    const std::string convergence_shape_audit =
        stats != nullptr ? stats->convergence_shape_audit
                         : result.convergence_shape_audit;
    const std::string convergence_kernel_response_audit =
        stats != nullptr ? stats->convergence_kernel_response_audit
                         : result.convergence_kernel_response_audit;
    const std::string channel_missed_semantics =
        stats != nullptr ? stats->channel_missed_semantics
                         : result.channel_missed_semantics;
    const std::string accepted_miss_diagnostics =
        stats != nullptr ? stats->accepted_miss_diagnostics
                         : result.accepted_miss_diagnostics;
    const double accepted_miss_absolute_floor =
        stats != nullptr ? stats->accepted_miss_absolute_floor
                         : result.accepted_miss_absolute_floor;
    const double convergence_k3_accepted_rho_cap =
        stats != nullptr ? stats->convergence_k3_accepted_rho_cap
                         : result.convergence_k3_accepted_rho_cap;
    const std::string mac_distance_policy = stats != nullptr
                                                ? stats->mac_distance_policy
                                                : result.mac_distance_policy;
    const double mac_surface_margin_factor =
        stats != nullptr ? stats->mac_surface_margin_factor
                         : result.mac_surface_margin_factor;
    const std::string mac_surface_margin_policy =
        stats != nullptr ? stats->mac_surface_margin_policy
                         : result.mac_surface_margin_policy;
    const double mac_surface_margin_aspect_threshold =
        stats != nullptr ? stats->mac_surface_margin_aspect_threshold
                         : result.mac_surface_margin_aspect_threshold;
    const double mac_surface_margin_rho_trigger =
        stats != nullptr ? stats->mac_surface_margin_rho_trigger
                         : result.mac_surface_margin_rho_trigger;
    const double mac_surface_margin_soft_aspect_threshold =
        stats != nullptr ? stats->mac_surface_margin_soft_aspect_threshold
                         : result.mac_surface_margin_soft_aspect_threshold;
    const double mac_surface_margin_soft_rho_trigger =
        stats != nullptr ? stats->mac_surface_margin_soft_rho_trigger
                         : result.mac_surface_margin_soft_rho_trigger;
    const double mac_surface_margin_soft_factor =
        stats != nullptr ? stats->mac_surface_margin_soft_factor
                         : result.mac_surface_margin_soft_factor;
    const std::size_t convergence_estimator_samples =
        stats != nullptr ? stats->convergence_estimator_samples
                         : result.convergence_estimator_samples;
    const std::size_t convergence_estimator_unstable_count =
        stats != nullptr ? stats->convergence_estimator_unstable_far_count
                         : result.convergence_estimator_unstable_far_count;
    const std::size_t convergence_estimator_unstable_panels =
        stats != nullptr ? stats->convergence_estimator_unstable_far_panels
                         : result.convergence_estimator_unstable_far_panels;
    const std::size_t convergence_estimator_split_rejects =
        stats != nullptr ? stats->convergence_estimator_split_rejects
                         : result.convergence_estimator_split_rejects;
    const std::size_t convergence_estimator_split_reject_panels =
        stats != nullptr ? stats->convergence_estimator_split_reject_panels
                         : result.convergence_estimator_split_reject_panels;
    const std::size_t convergence_low_abs_reaccepted_count =
        stats != nullptr ? stats->convergence_low_abs_reaccepted_far_count
                         : result.convergence_low_abs_reaccepted_far_count;
    const std::size_t convergence_low_abs_reaccepted_panels =
        stats != nullptr ? stats->convergence_low_abs_reaccepted_far_panels
                         : result.convergence_low_abs_reaccepted_far_panels;
    const double convergence_estimator_tail_ratio_max =
        stats != nullptr ? stats->convergence_estimator_tail_ratio_max
                         : result.convergence_estimator_tail_ratio_max;
    const double convergence_estimator_tail_ratio_mean =
        stats != nullptr ? stats->convergence_estimator_tail_ratio_mean
                         : result.convergence_estimator_tail_ratio_mean;
    const double convergence_estimator_error_max =
        stats != nullptr ? stats->convergence_estimator_error_max
                         : result.convergence_estimator_error_max;
    const double convergence_estimator_error_mean =
        stats != nullptr ? stats->convergence_estimator_error_mean
                         : result.convergence_estimator_error_mean;
    const double convergence_estimator_abs_error_max =
        stats != nullptr ? stats->convergence_estimator_abs_error_max
                         : result.convergence_estimator_abs_error_max;
    const double convergence_estimator_abs_error_mean =
        stats != nullptr ? stats->convergence_estimator_abs_error_mean
                         : result.convergence_estimator_abs_error_mean;
    const double convergence_tail_norm_p_max =
        stats != nullptr ? stats->convergence_tail_norm_p_max
                         : result.convergence_tail_norm_p_max;
    const double convergence_tail_norm_p_mean =
        stats != nullptr ? stats->convergence_tail_norm_p_mean
                         : result.convergence_tail_norm_p_mean;
    const double convergence_tail_norm_pm1_max =
        stats != nullptr ? stats->convergence_tail_norm_pm1_max
                         : result.convergence_tail_norm_pm1_max;
    const double convergence_tail_norm_pm1_mean =
        stats != nullptr ? stats->convergence_tail_norm_pm1_mean
                         : result.convergence_tail_norm_pm1_mean;
    const double convergence_effective_jcp_amp_max =
        stats != nullptr ? stats->convergence_effective_jcp_amp_max
                         : result.convergence_effective_jcp_amp_max;
    const double convergence_effective_jcp_amp_mean =
        stats != nullptr ? stats->convergence_effective_jcp_amp_mean
                         : result.convergence_effective_jcp_amp_mean;
    const double convergence_dielectric_amp_max =
        stats != nullptr ? stats->convergence_dielectric_amp_max
                         : result.convergence_dielectric_amp_max;
    const double convergence_dielectric_amp_mean =
        stats != nullptr ? stats->convergence_dielectric_amp_mean
                         : result.convergence_dielectric_amp_mean;
    const auto convergence_channel_estimated_error_max =
        stats != nullptr ? stats->convergence_channel_estimated_error_max
                         : result.convergence_channel_estimated_error_max;
    const auto convergence_channel_effective_jcp_amp_max =
        stats != nullptr ? stats->convergence_channel_effective_jcp_amp_max
                         : result.convergence_channel_effective_jcp_amp_max;
    const auto convergence_channel_effective_jcp_amp_mean =
        stats != nullptr ? stats->convergence_channel_effective_jcp_amp_mean
                         : result.convergence_channel_effective_jcp_amp_mean;
    const auto convergence_channel_unstable_count =
        stats != nullptr ? stats->convergence_channel_unstable_far_count
                         : result.convergence_channel_unstable_far_count;
    const auto convergence_channel_unstable_panels =
        stats != nullptr ? stats->convergence_channel_unstable_far_panels
                         : result.convergence_channel_unstable_far_panels;
    const auto convergence_channel_split_rejects =
        stats != nullptr ? stats->convergence_channel_split_rejects
                         : result.convergence_channel_split_rejects;
    const auto convergence_channel_split_reject_panels =
        stats != nullptr ? stats->convergence_channel_split_reject_panels
                         : result.convergence_channel_split_reject_panels;
    const auto convergence_channel_low_abs_reaccepted_count =
        stats != nullptr ? stats->convergence_channel_low_abs_reaccepted_count
                         : result.convergence_channel_low_abs_reaccepted_count;
    const auto convergence_channel_low_abs_reaccepted_panels =
        stats != nullptr ? stats->convergence_channel_low_abs_reaccepted_panels
                         : result.convergence_channel_low_abs_reaccepted_panels;
    const auto convergence_channel_high_error_caught =
        stats != nullptr ? stats->convergence_channel_high_error_caught
                         : result.convergence_channel_high_error_caught;
    const auto convergence_channel_high_error_missed =
        stats != nullptr ? stats->convergence_channel_high_error_missed
                         : result.convergence_channel_high_error_missed;
    const auto convergence_channel_high_error_caught_by_split =
        stats != nullptr
            ? stats->convergence_channel_high_error_caught_by_split
            : result.convergence_channel_high_error_caught_by_split;
    const std::size_t convergence_estimator_high_error_caught =
        stats != nullptr ? stats->convergence_estimator_high_error_caught
                         : result.convergence_estimator_high_error_caught;
    const std::size_t convergence_estimator_high_error_missed =
        stats != nullptr ? stats->convergence_estimator_high_error_missed
                         : result.convergence_estimator_high_error_missed;
    const std::size_t convergence_estimator_high_error_caught_by_split =
        stats != nullptr
            ? stats->convergence_estimator_high_error_caught_by_split
            : result.convergence_estimator_high_error_caught_by_split;
    const std::size_t convergence_estimator_phantom_high_error_count =
        stats != nullptr
            ? stats->convergence_estimator_phantom_high_error_count
            : result.convergence_estimator_phantom_high_error_count;
    const double convergence_estimator_phantom_high_error_abs_max =
        stats != nullptr
            ? stats->convergence_estimator_phantom_high_error_abs_max
            : result.convergence_estimator_phantom_high_error_abs_max;
    const std::size_t convergence_overreject_count =
        stats != nullptr ? stats->convergence_overreject_count
                         : result.convergence_overreject_count;
    const std::size_t convergence_overreject_panels =
        stats != nullptr ? stats->convergence_overreject_panels
                         : result.convergence_overreject_panels;
    const std::size_t convergence_overreject_caught_count =
        stats != nullptr ? stats->convergence_overreject_caught_count
                         : result.convergence_overreject_caught_count;
    const std::size_t convergence_overreject_caught_panels =
        stats != nullptr ? stats->convergence_overreject_caught_panels
                         : result.convergence_overreject_caught_panels;
    const double convergence_overreject_abs_error_max =
        stats != nullptr ? stats->convergence_overreject_abs_error_max
                         : result.convergence_overreject_abs_error_max;
    const double convergence_overreject_abs_error_mean =
        stats != nullptr ? stats->convergence_overreject_abs_error_mean
                         : result.convergence_overreject_abs_error_mean;
    const double convergence_overreject_abs_error_p95 =
        stats != nullptr ? stats->convergence_overreject_abs_error_p95
                         : result.convergence_overreject_abs_error_p95;
    const double convergence_overreject_abs_error_p99 =
        stats != nullptr ? stats->convergence_overreject_abs_error_p99
                         : result.convergence_overreject_abs_error_p99;
    const double convergence_overreject_estimated_abs_error_max =
        stats != nullptr
            ? stats->convergence_overreject_estimated_abs_error_max
            : result.convergence_overreject_estimated_abs_error_max;
    const double convergence_overreject_estimated_abs_error_mean =
        stats != nullptr
            ? stats->convergence_overreject_estimated_abs_error_mean
            : result.convergence_overreject_estimated_abs_error_mean;
    const double convergence_overreject_required_scale_max =
        stats != nullptr ? stats->convergence_overreject_required_scale_max
                         : result.convergence_overreject_required_scale_max;
    const double convergence_overreject_required_scale_p95 =
        stats != nullptr ? stats->convergence_overreject_required_scale_p95
                         : result.convergence_overreject_required_scale_p95;
    const double convergence_overreject_required_scale_p99 =
        stats != nullptr ? stats->convergence_overreject_required_scale_p99
                         : result.convergence_overreject_required_scale_p99;
    const auto convergence_channel_overreject_count =
        stats != nullptr ? stats->convergence_channel_overreject_count
                         : result.convergence_channel_overreject_count;
    const auto convergence_channel_overreject_caught_count =
        stats != nullptr ? stats->convergence_channel_overreject_caught_count
                         : result.convergence_channel_overreject_caught_count;
    const auto convergence_channel_overreject_panels =
        stats != nullptr ? stats->convergence_channel_overreject_panels
                         : result.convergence_channel_overreject_panels;
    const auto convergence_channel_overreject_abs_error_max =
        stats != nullptr ? stats->convergence_channel_overreject_abs_error_max
                         : result.convergence_channel_overreject_abs_error_max;
    const auto convergence_channel_overreject_abs_error_mean =
        stats != nullptr ? stats->convergence_channel_overreject_abs_error_mean
                         : result.convergence_channel_overreject_abs_error_mean;
    const auto convergence_channel_overreject_required_scale_max =
        stats != nullptr
            ? stats->convergence_channel_overreject_required_scale_max
            : result.convergence_channel_overreject_required_scale_max;
    const auto convergence_channel_overreject_required_scale_p99 =
        stats != nullptr
            ? stats->convergence_channel_overreject_required_scale_p99
            : result.convergence_channel_overreject_required_scale_p99;
    const auto phantom_high_error_count =
        stats != nullptr ? stats->phantom_high_error_channel_count
                         : result.phantom_high_error_channel_count;
    const auto phantom_high_error_abs_max =
        stats != nullptr ? stats->phantom_high_error_channel_abs_max
                         : result.phantom_high_error_channel_abs_max;
    const auto accepted_miss_count = stats != nullptr
                                         ? stats->accepted_miss_channel_count
                                         : result.accepted_miss_channel_count;
    const auto accepted_miss_panels = stats != nullptr
                                          ? stats->accepted_miss_channel_panels
                                          : result.accepted_miss_channel_panels;
    const auto accepted_miss_rho_min =
        stats != nullptr ? stats->accepted_miss_channel_rho_min
                         : result.accepted_miss_channel_rho_min;
    const auto accepted_miss_rho_mean =
        stats != nullptr ? stats->accepted_miss_channel_rho_mean
                         : result.accepted_miss_channel_rho_mean;
    const auto accepted_miss_rho_max =
        stats != nullptr ? stats->accepted_miss_channel_rho_max
                         : result.accepted_miss_channel_rho_max;
    const auto accepted_miss_tail_ratio_mean =
        stats != nullptr ? stats->accepted_miss_channel_tail_ratio_mean
                         : result.accepted_miss_channel_tail_ratio_mean;
    const auto accepted_miss_tail_ratio_max =
        stats != nullptr ? stats->accepted_miss_channel_tail_ratio_max
                         : result.accepted_miss_channel_tail_ratio_max;
    const auto accepted_miss_estimated_error_mean =
        stats != nullptr ? stats->accepted_miss_channel_estimated_error_mean
                         : result.accepted_miss_channel_estimated_error_mean;
    const auto accepted_miss_estimated_error_max =
        stats != nullptr ? stats->accepted_miss_channel_estimated_error_max
                         : result.accepted_miss_channel_estimated_error_max;
    const std::string accepted_miss_worst_channel =
        stats != nullptr ? stats->accepted_miss_worst_channel
                         : result.accepted_miss_worst_channel;
    const std::size_t convergence_k3_rho_cap_rejects =
        stats != nullptr ? stats->convergence_k3_rho_cap_rejects
                         : result.convergence_k3_rho_cap_rejects;
    const std::size_t convergence_k3_rho_cap_reject_panels =
        stats != nullptr ? stats->convergence_k3_rho_cap_reject_panels
                         : result.convergence_k3_rho_cap_reject_panels;
    const std::size_t moment_nodes = stats != nullptr
                                         ? stats->moment_audit_nodes
                                         : result.moment_audit_nodes;
    const double moment_rel_max = stats != nullptr
                                      ? stats->moment_translation_rel_error_max
                                      : result.moment_translation_rel_error_max;
    const double moment_rel_mean =
        stats != nullptr ? stats->moment_translation_rel_error_mean
                         : result.moment_translation_rel_error_mean;
    const double moment_offset_max = stats != nullptr
                                         ? stats->moment_translation_offset_max
                                         : result.moment_translation_offset_max;
    const double moment_offset_mean =
        stats != nullptr ? stats->moment_translation_offset_mean
                         : result.moment_translation_offset_mean;
    const double moment_cancel_max = stats != nullptr
                                         ? stats->moment_cancellation_ratio_max
                                         : result.moment_cancellation_ratio_max;
    const double moment_cancel_mean =
        stats != nullptr ? stats->moment_cancellation_ratio_mean
                         : result.moment_cancellation_ratio_mean;
    const double moment_dynamic_range = stats != nullptr
                                            ? stats->moment_dynamic_range_max
                                            : result.moment_dynamic_range_max;
    const std::size_t expansion_pairs = stats != nullptr
                                            ? stats->far_expansion_audit_pairs
                                            : result.far_expansion_audit_pairs;
    const std::size_t expansion_panels =
        stats != nullptr ? stats->far_expansion_audit_panels
                         : result.far_expansion_audit_panels;
    const std::string expansion_audit_channel =
        stats != nullptr ? stats->expansion_audit_channel
                         : result.expansion_audit_channel;
    const double expansion_direct_ms = stats != nullptr
                                           ? stats->far_expansion_direct_ms
                                           : result.far_expansion_direct_ms;
    const double expansion_rel_max = stats != nullptr
                                         ? stats->far_expansion_rel_error_max
                                         : result.far_expansion_rel_error_max;
    const double expansion_rel_mean = stats != nullptr
                                          ? stats->far_expansion_rel_error_mean
                                          : result.far_expansion_rel_error_mean;
    const double expansion_rel_p95 = stats != nullptr
                                         ? stats->far_expansion_rel_error_p95
                                         : result.far_expansion_rel_error_p95;
    const double expansion_rel_p99 = stats != nullptr
                                         ? stats->far_expansion_rel_error_p99
                                         : result.far_expansion_rel_error_p99;
    const std::string expansion_worst_channel =
        stats != nullptr ? stats->far_expansion_worst_channel
                         : result.far_expansion_worst_channel;
    const auto expansion_channel_rel_max =
        stats != nullptr ? stats->far_expansion_channel_rel_error_max
                         : result.far_expansion_channel_rel_error_max;
    const auto expansion_channel_rel_mean =
        stats != nullptr ? stats->far_expansion_channel_rel_error_mean
                         : result.far_expansion_channel_rel_error_mean;
    const auto expansion_channel_rel_p99 =
        stats != nullptr ? stats->far_expansion_channel_rel_error_p99
                         : result.far_expansion_channel_rel_error_p99;
    const bool expansion_mean_gate_pass =
        stats != nullptr ? stats->far_expansion_mean_gate_pass
                         : result.far_expansion_mean_gate_pass;
    const double convergence_rho_max = stats != nullptr
                                           ? stats->far_convergence_rho_max
                                           : result.far_convergence_rho_max;
    const double convergence_rho_mean = stats != nullptr
                                            ? stats->far_convergence_rho_mean
                                            : result.far_convergence_rho_mean;
    const double node_aspect_max = stats != nullptr
                                       ? stats->node_aspect_ratio_max
                                       : result.node_aspect_ratio_max;
    const double node_aspect_mean = stats != nullptr
                                        ? stats->node_aspect_ratio_mean
                                        : result.node_aspect_ratio_mean;
    const double accepted_aspect_max =
        stats != nullptr ? stats->accepted_far_aspect_ratio_max
                         : result.accepted_far_aspect_ratio_max;
    const double accepted_aspect_mean =
        stats != nullptr ? stats->accepted_far_aspect_ratio_mean
                         : result.accepted_far_aspect_ratio_mean;
    const std::size_t mac_convergence_rho_rejects =
        stats != nullptr ? stats->mac_convergence_rho_rejects
                         : result.mac_convergence_rho_rejects;
    const std::size_t mac_convergence_aspect_rejects =
        stats != nullptr ? stats->mac_convergence_aspect_rejects
                         : result.mac_convergence_aspect_rejects;
    const std::size_t mac_convergence_rho_reject_panels =
        stats != nullptr ? stats->mac_convergence_rho_reject_panels
                         : result.mac_convergence_rho_reject_panels;
    const std::size_t mac_convergence_aspect_reject_panels =
        stats != nullptr ? stats->mac_convergence_aspect_reject_panels
                         : result.mac_convergence_aspect_reject_panels;
    const std::size_t mac_surface_margin_rejects =
        stats != nullptr ? stats->mac_surface_margin_rejects
                         : result.mac_surface_margin_rejects;
    const std::size_t mac_surface_margin_reject_panels =
        stats != nullptr ? stats->mac_surface_margin_reject_panels
                         : result.mac_surface_margin_reject_panels;
    const std::size_t mac_surface_margin_applied =
        stats != nullptr ? stats->mac_surface_margin_applied
                         : result.mac_surface_margin_applied;
    const std::size_t mac_surface_margin_skipped =
        stats != nullptr ? stats->mac_surface_margin_skipped
                         : result.mac_surface_margin_skipped;
    const std::size_t mac_surface_margin_adaptive_rejects =
        stats != nullptr ? stats->mac_surface_margin_adaptive_rejects
                         : result.mac_surface_margin_adaptive_rejects;
    const std::size_t mac_surface_margin_adaptive_reject_panels =
        stats != nullptr ? stats->mac_surface_margin_adaptive_reject_panels
                         : result.mac_surface_margin_adaptive_reject_panels;
    const std::size_t mac_surface_margin_skipped_accepted_far =
        stats != nullptr ? stats->mac_surface_margin_skipped_accepted_far
                         : result.mac_surface_margin_skipped_accepted_far;
    const std::size_t mac_surface_margin_hard_applied =
        stats != nullptr ? stats->mac_surface_margin_hard_applied
                         : result.mac_surface_margin_hard_applied;
    const std::size_t mac_surface_margin_soft_applied =
        stats != nullptr ? stats->mac_surface_margin_soft_applied
                         : result.mac_surface_margin_soft_applied;
    const std::size_t mac_surface_margin_hard_rejects =
        stats != nullptr ? stats->mac_surface_margin_hard_rejects
                         : result.mac_surface_margin_hard_rejects;
    const std::size_t mac_surface_margin_hard_reject_panels =
        stats != nullptr ? stats->mac_surface_margin_hard_reject_panels
                         : result.mac_surface_margin_hard_reject_panels;
    const std::size_t mac_surface_margin_soft_rejects =
        stats != nullptr ? stats->mac_surface_margin_soft_rejects
                         : result.mac_surface_margin_soft_rejects;
    const std::size_t mac_surface_margin_soft_reject_panels =
        stats != nullptr ? stats->mac_surface_margin_soft_reject_panels
                         : result.mac_surface_margin_soft_reject_panels;
    const std::string leaf_work_profile =
        stats != nullptr ? stats->leaf_work_profile : result.leaf_work_profile;
    const std::size_t leaf_work_mac_terminal_panels =
        stats != nullptr ? stats->leaf_work_mac_terminal_panels
                         : result.leaf_work_mac_terminal_panels;
    const std::size_t leaf_work_physical_gap_panels =
        stats != nullptr ? stats->leaf_work_physical_gap_panels
                         : result.leaf_work_physical_gap_panels;
    const std::size_t leaf_work_fragmentation_panels =
        stats != nullptr ? stats->leaf_work_fragmentation_panels
                         : result.leaf_work_fragmentation_panels;
    const std::size_t leaf_work_convergence_panels =
        stats != nullptr ? stats->leaf_work_convergence_panels
                         : result.leaf_work_convergence_panels;
    const std::size_t leaf_work_convergence_estimator_panels =
        stats != nullptr ? stats->leaf_work_convergence_estimator_panels
                         : result.leaf_work_convergence_estimator_panels;
    const std::size_t leaf_work_convergence_rho_panels =
        stats != nullptr ? stats->leaf_work_convergence_rho_panels
                         : result.leaf_work_convergence_rho_panels;
    const std::size_t leaf_work_convergence_aspect_panels =
        stats != nullptr ? stats->leaf_work_convergence_aspect_panels
                         : result.leaf_work_convergence_aspect_panels;
    const std::size_t leaf_work_surface_margin_panels =
        stats != nullptr ? stats->leaf_work_surface_margin_panels
                         : result.leaf_work_surface_margin_panels;
    const std::size_t leaf_work_surface_margin_hard_panels =
        stats != nullptr ? stats->leaf_work_surface_margin_hard_panels
                         : result.leaf_work_surface_margin_hard_panels;
    const std::size_t leaf_work_surface_margin_soft_panels =
        stats != nullptr ? stats->leaf_work_surface_margin_soft_panels
                         : result.leaf_work_surface_margin_soft_panels;
    const std::size_t leaf_work_k3_rho_cap_panels =
        stats != nullptr ? stats->leaf_work_k3_rho_cap_panels
                         : result.leaf_work_k3_rho_cap_panels;
    const std::size_t leaf_work_singular_fallback_panels =
        stats != nullptr ? stats->leaf_work_singular_fallback_panels
                         : result.leaf_work_singular_fallback_panels;
    const std::size_t leaf_work_scalar_tail_panels =
        stats != nullptr ? stats->leaf_work_scalar_tail_panels
                         : result.leaf_work_scalar_tail_panels;
    const std::size_t leaf_work_profile_accounted_panels =
        stats != nullptr ? stats->leaf_work_profile_accounted_panels
                         : result.leaf_work_profile_accounted_panels;
    const double leaf_work_mac_terminal_frac =
        stats != nullptr ? stats->leaf_work_mac_terminal_frac
                         : result.leaf_work_mac_terminal_frac;
    const double leaf_work_convergence_frac =
        stats != nullptr ? stats->leaf_work_convergence_frac
                         : result.leaf_work_convergence_frac;
    const double leaf_work_surface_margin_frac =
        stats != nullptr ? stats->leaf_work_surface_margin_frac
                         : result.leaf_work_surface_margin_frac;
    const double leaf_work_singular_fallback_frac =
        stats != nullptr ? stats->leaf_work_singular_fallback_frac
                         : result.leaf_work_singular_fallback_frac;
    const double leaf_work_scalar_tail_frac =
        stats != nullptr ? stats->leaf_work_scalar_tail_frac
                         : result.leaf_work_scalar_tail_frac;
    const std::string cluster_policy =
        stats != nullptr ? stats->cluster_policy : result.cluster_policy;
    const int target_batch_size =
        stats != nullptr ? stats->target_batch_size : result.target_batch_size;
    const int leaf_pair_min_source_panels =
        stats != nullptr ? stats->leaf_pair_min_source_panels
                         : result.leaf_pair_min_source_panels;
    const int leaf_pair_min_targets = stats != nullptr
                                          ? stats->leaf_pair_min_targets
                                          : result.leaf_pair_min_targets;
    const std::size_t leaf_pair_batch_interactions =
        stats != nullptr ? stats->leaf_pair_batch_interactions
                         : result.leaf_pair_batch_interactions;
    const std::size_t leaf_pair_batch_targets =
        stats != nullptr ? stats->leaf_pair_batch_targets
                         : result.leaf_pair_batch_targets;
    const std::size_t leaf_pair_batch_source_panels =
        stats != nullptr ? stats->leaf_pair_batch_source_panels
                         : result.leaf_pair_batch_source_panels;
    const std::size_t leaf_pair_batch_residual_panels =
        stats != nullptr ? stats->leaf_pair_batch_residual_panels
                         : result.leaf_pair_batch_residual_panels;
    const std::size_t leaf_pair_batch_fallbacks =
        stats != nullptr ? stats->leaf_pair_batch_fallbacks
                         : result.leaf_pair_batch_fallbacks;
    const double leaf_pair_batch_ms = stats != nullptr
                                          ? stats->leaf_pair_batch_ms
                                          : result.leaf_pair_batch_ms;
    const std::string leaf_pair_batch_kernel =
        stats != nullptr ? stats->leaf_pair_batch_kernel
                         : result.leaf_pair_batch_kernel;
    const std::size_t leaf_pair_batch_vectorized_sources =
        stats != nullptr ? stats->leaf_pair_batch_vectorized_sources
                         : result.leaf_pair_batch_vectorized_sources;
    const std::size_t leaf_pair_batch_vector_lanes =
        stats != nullptr ? stats->leaf_pair_batch_vector_lanes
                         : result.leaf_pair_batch_vector_lanes;
    const double leaf_pair_batch_pairs_per_second =
        stats != nullptr ? stats->leaf_pair_batch_pairs_per_second
                         : result.leaf_pair_batch_pairs_per_second;
    const std::string leaf_kernel_profile = stats != nullptr
                                                ? stats->leaf_kernel_profile
                                                : result.leaf_kernel_profile;
    const std::string jcp_direct_kernel =
        stats != nullptr ? stats->jcp_direct_kernel : result.jcp_direct_kernel;
    const std::size_t leaf_kernel_scalar_tasks =
        stats != nullptr ? stats->leaf_kernel_scalar_tasks
                         : result.leaf_kernel_scalar_tasks;
    const std::size_t leaf_kernel_batch_tasks =
        stats != nullptr ? stats->leaf_kernel_batch_tasks
                         : result.leaf_kernel_batch_tasks;
    const std::size_t leaf_kernel_source_panel_evals =
        stats != nullptr ? stats->leaf_kernel_source_panel_evals
                         : result.leaf_kernel_source_panel_evals;
    const double leaf_kernel_scalar_ms = stats != nullptr
                                             ? stats->leaf_kernel_scalar_ms
                                             : result.leaf_kernel_scalar_ms;
    const double leaf_kernel_target_avx2_ms =
        stats != nullptr ? stats->leaf_kernel_target_avx2_ms
                         : result.leaf_kernel_target_avx2_ms;
    const double leaf_kernel_tail_ms = stats != nullptr
                                           ? stats->leaf_kernel_tail_ms
                                           : result.leaf_kernel_tail_ms;
    const std::size_t leaf_pair_batch_soa_sources =
        stats != nullptr ? stats->leaf_pair_batch_soa_sources
                         : result.leaf_pair_batch_soa_sources;
    const double leaf_pair_batch_soa_ms = stats != nullptr
                                              ? stats->leaf_pair_batch_soa_ms
                                              : result.leaf_pair_batch_soa_ms;
    const double leaf_pair_batch_scalar_tail_ms =
        stats != nullptr ? stats->leaf_pair_batch_scalar_tail_ms
                         : result.leaf_pair_batch_scalar_tail_ms;
    const double leaf_pair_batch_clean_source_panels_per_second =
        stats != nullptr
            ? stats->leaf_pair_batch_clean_source_panels_per_second
            : result.leaf_pair_batch_clean_source_panels_per_second;
    const std::string leaf_pair_selection_profile =
        stats != nullptr ? stats->leaf_pair_selection_profile
                         : result.leaf_pair_selection_profile;
    const std::string leaf_pair_probe_cache =
        stats != nullptr ? stats->leaf_pair_probe_cache
                         : result.leaf_pair_probe_cache;
    const std::string leaf_pair_target_group =
        stats != nullptr ? stats->leaf_pair_target_group
                         : result.leaf_pair_target_group;
    const std::string leaf_pair_directness_probe =
        stats != nullptr ? stats->leaf_pair_directness_probe
                         : result.leaf_pair_directness_probe;
    const std::string leaf_pair_directness_cache =
        stats != nullptr ? stats->leaf_pair_directness_cache
                         : result.leaf_pair_directness_cache;
    const std::string leaf_pair_directness_verify =
        stats != nullptr ? stats->leaf_pair_directness_verify
                         : result.leaf_pair_directness_verify;
    const int leaf_pair_max_lookahead = stats != nullptr
                                            ? stats->leaf_pair_max_lookahead
                                            : result.leaf_pair_max_lookahead;
    const std::size_t leaf_pair_selection_candidate_scans =
        stats != nullptr ? stats->leaf_pair_selection_candidate_scans
                         : result.leaf_pair_selection_candidate_scans;
    const std::size_t leaf_pair_selection_clean_rejects =
        stats != nullptr ? stats->leaf_pair_selection_clean_rejects
                         : result.leaf_pair_selection_clean_rejects;
    const std::size_t leaf_pair_selection_probe_calls =
        stats != nullptr ? stats->leaf_pair_selection_probe_calls
                         : result.leaf_pair_selection_probe_calls;
    const std::size_t leaf_pair_selection_probe_node_visits =
        stats != nullptr ? stats->leaf_pair_selection_probe_node_visits
                         : result.leaf_pair_selection_probe_node_visits;
    const double leaf_pair_selection_probe_ms =
        stats != nullptr ? stats->leaf_pair_selection_probe_ms
                         : result.leaf_pair_selection_probe_ms;
    const double leaf_pair_selection_ms = stats != nullptr
                                              ? stats->leaf_pair_selection_ms
                                              : result.leaf_pair_selection_ms;
    const std::size_t leaf_pair_selection_probe_cache_hits =
        stats != nullptr ? stats->leaf_pair_selection_probe_cache_hits
                         : result.leaf_pair_selection_probe_cache_hits;
    const std::size_t leaf_pair_selection_probe_cache_misses =
        stats != nullptr ? stats->leaf_pair_selection_probe_cache_misses
                         : result.leaf_pair_selection_probe_cache_misses;
    const std::size_t leaf_pair_selection_insufficient_targets =
        stats != nullptr ? stats->leaf_pair_selection_insufficient_targets
                         : result.leaf_pair_selection_insufficient_targets;
    const std::size_t leaf_pair_selection_same_target_leaf_rejects =
        stats != nullptr ? stats->leaf_pair_selection_same_target_leaf_rejects
                         : result.leaf_pair_selection_same_target_leaf_rejects;
    const std::size_t leaf_pair_directness_ancestor_probe_calls =
        stats != nullptr ? stats->leaf_pair_directness_ancestor_probe_calls
                         : result.leaf_pair_directness_ancestor_probe_calls;
    const std::size_t leaf_pair_directness_ancestor_probe_node_visits =
        stats != nullptr
            ? stats->leaf_pair_directness_ancestor_probe_node_visits
            : result.leaf_pair_directness_ancestor_probe_node_visits;
    const double leaf_pair_directness_ancestor_probe_ms =
        stats != nullptr ? stats->leaf_pair_directness_ancestor_probe_ms
                         : result.leaf_pair_directness_ancestor_probe_ms;
    const std::size_t leaf_pair_directness_pair_cache_hits =
        stats != nullptr ? stats->leaf_pair_directness_pair_cache_hits
                         : result.leaf_pair_directness_pair_cache_hits;
    const std::size_t leaf_pair_directness_pair_cache_misses =
        stats != nullptr ? stats->leaf_pair_directness_pair_cache_misses
                         : result.leaf_pair_directness_pair_cache_misses;
    const std::size_t leaf_pair_directness_all_direct_proofs =
        stats != nullptr ? stats->leaf_pair_directness_all_direct_proofs
                         : result.leaf_pair_directness_all_direct_proofs;
    const std::size_t leaf_pair_directness_mixed_fallbacks =
        stats != nullptr ? stats->leaf_pair_directness_mixed_fallbacks
                         : result.leaf_pair_directness_mixed_fallbacks;
    const std::size_t leaf_pair_directness_verify_checks =
        stats != nullptr ? stats->leaf_pair_directness_verify_checks
                         : result.leaf_pair_directness_verify_checks;
    const std::size_t leaf_pair_directness_verify_mismatches =
        stats != nullptr ? stats->leaf_pair_directness_verify_mismatches
                         : result.leaf_pair_directness_verify_mismatches;
    const std::string leaf_pair_scheduler = stats != nullptr
                                                ? stats->leaf_pair_scheduler
                                                : result.leaf_pair_scheduler;
    const int leaf_pair_worklist_max_active =
        stats != nullptr ? stats->leaf_pair_worklist_max_active
                         : result.leaf_pair_worklist_max_active;
    const std::string leaf_pair_worklist_verify =
        stats != nullptr ? stats->leaf_pair_worklist_verify
                         : result.leaf_pair_worklist_verify;
    const std::size_t leaf_pair_worklist_tasks =
        stats != nullptr ? stats->leaf_pair_worklist_tasks
                         : result.leaf_pair_worklist_tasks;
    const std::size_t leaf_pair_worklist_groups =
        stats != nullptr ? stats->leaf_pair_worklist_groups
                         : result.leaf_pair_worklist_groups;
    const std::size_t leaf_pair_worklist_batched_groups =
        stats != nullptr ? stats->leaf_pair_worklist_batched_groups
                         : result.leaf_pair_worklist_batched_groups;
    const std::size_t leaf_pair_worklist_scalar_flush_tasks =
        stats != nullptr ? stats->leaf_pair_worklist_scalar_flush_tasks
                         : result.leaf_pair_worklist_scalar_flush_tasks;
    const std::size_t leaf_pair_worklist_forced_flushes =
        stats != nullptr ? stats->leaf_pair_worklist_forced_flushes
                         : result.leaf_pair_worklist_forced_flushes;
    const std::size_t leaf_pair_worklist_active_groups_max =
        stats != nullptr ? stats->leaf_pair_worklist_active_groups_max
                         : result.leaf_pair_worklist_active_groups_max;
    const double leaf_pair_worklist_ms = stats != nullptr
                                             ? stats->leaf_pair_worklist_ms
                                             : result.leaf_pair_worklist_ms;
    const std::size_t leaf_pair_worklist_verify_checks =
        stats != nullptr ? stats->leaf_pair_worklist_verify_checks
                         : result.leaf_pair_worklist_verify_checks;
    const std::size_t leaf_pair_worklist_verify_mismatches =
        stats != nullptr ? stats->leaf_pair_worklist_verify_mismatches
                         : result.leaf_pair_worklist_verify_mismatches;
    const std::string leaf_block_verify =
        stats != nullptr ? stats->leaf_block_verify : result.leaf_block_verify;
    const std::size_t leaf_block_target_leaves =
        stats != nullptr ? stats->leaf_block_target_leaves
                         : result.leaf_block_target_leaves;
    const std::size_t leaf_block_targets = stats != nullptr
                                               ? stats->leaf_block_targets
                                               : result.leaf_block_targets;
    const std::size_t leaf_block_source_groups =
        stats != nullptr ? stats->leaf_block_source_groups
                         : result.leaf_block_source_groups;
    const std::size_t leaf_block_batched_groups =
        stats != nullptr ? stats->leaf_block_batched_groups
                         : result.leaf_block_batched_groups;
    const std::size_t leaf_block_scalar_tail_tasks =
        stats != nullptr ? stats->leaf_block_scalar_tail_tasks
                         : result.leaf_block_scalar_tail_tasks;
    const double leaf_block_group_ms = stats != nullptr
                                           ? stats->leaf_block_group_ms
                                           : result.leaf_block_group_ms;
    const double leaf_block_flush_ms = stats != nullptr
                                           ? stats->leaf_block_flush_ms
                                           : result.leaf_block_flush_ms;
    const double leaf_block_writeback_ms = stats != nullptr
                                               ? stats->leaf_block_writeback_ms
                                               : result.leaf_block_writeback_ms;
    const std::size_t leaf_block_verify_checks =
        stats != nullptr ? stats->leaf_block_verify_checks
                         : result.leaf_block_verify_checks;
    const std::size_t leaf_block_verify_mismatches =
        stats != nullptr ? stats->leaf_block_verify_mismatches
                         : result.leaf_block_verify_mismatches;

    out_ << std::fixed << std::setprecision(6)
         << std::chrono::duration<double>(now - start_).count() << ','
         << csv_escape_progress(phase) << ',' << gmres_iter << ','
         << restart_iter << ',' << matvec_calls << ',' << std::scientific
         << std::setprecision(12) << residual << ','
         << result.gmres_restart_requested << ','
         << result.gmres_restart_effective << ',' << result.best_residual << ','
         << result.best_residual_iter << ',' << result.final_residual << ','
         << result.best_preconditioned_residual << ','
         << result.final_preconditioned_residual << ','
         << result.true_residual_abs << ',' << result.true_residual_rel << ','
         << csv_escape_progress(result.true_residual_probe_mode) << ','
         << result.true_residual_probe_interval << ','
         << result.true_residual_probe_calls << ',' << std::fixed
         << std::setprecision(6) << result.true_residual_probe_ms / 1000.0
         << ',' << csv_escape_progress(result.true_residual_probe_scope) << ','
         << result.true_residual_sample_count << ','
         << result.sampled_true_residual_rel << ','
         << result.sampled_true_residual_rows << ','
         << result.sampled_probe_calls << ',' << std::fixed
         << std::setprecision(6) << result.sampled_probe_ms / 1000.0 << ','
         << result.full_true_residual_confirm_calls << ',' << std::fixed
         << std::setprecision(6)
         << result.full_true_residual_confirm_ms / 1000.0 << std::fixed
         << std::setprecision(6) << ',' << result.panel_count << ','
         << result.unknown_count << ',' << result.taylor_order << ','
         << tree_nodes << ',' << csv_escape_progress(interaction_mode) << ','
         << csv_escape_progress(tree_builder) << ','
         << csv_escape_progress(node_bounds) << ','
         << csv_escape_progress(mac_safety) << ',' << mac_gap_factor << ','
         << csv_escape_progress(expansion_center) << ','
         << csv_escape_progress(fragmentation_safety) << ','
         << csv_escape_progress(moment_construction) << ','
         << csv_escape_progress(moment_sum) << ','
         << csv_escape_progress(convergence_audit) << ','
         << csv_escape_progress(convergence_safety) << ','
         << convergence_rho_cap << ',' << mac_aspect_cap << ','
         << csv_escape_progress(convergence_estimator) << ','
         << convergence_tail_threshold << ',' << convergence_estimator_error_cap
         << ',' << convergence_unstable_rho_cap << ','
         << csv_escape_progress(convergence_calibration) << ','
         << csv_escape_progress(convergence_dielectric_amp) << ','
         << csv_escape_progress(convergence_estimator_model) << ','
         << csv_escape_progress(convergence_channel_calibration) << ','
         << csv_escape_progress(convergence_split_policy) << ','
         << csv_escape_progress(convergence_split_selectivity) << ','
         << convergence_split_absolute_floor << ','
         << csv_escape_progress(convergence_overreject_audit) << ','
         << csv_escape_progress(convergence_overreject_detail) << ','
         << csv_escape_progress(convergence_shape_audit) << ','
         << csv_escape_progress(convergence_kernel_response_audit) << ','
         << csv_escape_progress(channel_missed_semantics) << ','
         << csv_escape_progress(accepted_miss_diagnostics) << ','
         << accepted_miss_absolute_floor << ','
         << convergence_k3_accepted_rho_cap << ','
         << csv_escape_progress(mac_distance_policy) << ','
         << mac_surface_margin_factor << ','
         << csv_escape_progress(mac_surface_margin_policy) << ','
         << mac_surface_margin_aspect_threshold << ','
         << mac_surface_margin_rho_trigger << ','
         << mac_surface_margin_soft_aspect_threshold << ','
         << mac_surface_margin_soft_rho_trigger << ','
         << mac_surface_margin_soft_factor << ','
         << convergence_estimator_samples << ','
         << convergence_estimator_unstable_count << ','
         << convergence_estimator_unstable_panels << ','
         << convergence_estimator_split_rejects << ','
         << convergence_estimator_split_reject_panels << ','
         << convergence_low_abs_reaccepted_count << ','
         << convergence_low_abs_reaccepted_panels << ','
         << convergence_estimator_tail_ratio_max << ','
         << convergence_estimator_tail_ratio_mean << ','
         << convergence_estimator_error_max << ','
         << convergence_estimator_error_mean << ','
         << convergence_estimator_abs_error_max << ','
         << convergence_estimator_abs_error_mean << ','
         << convergence_tail_norm_p_max << ',' << convergence_tail_norm_p_mean
         << ',' << convergence_tail_norm_pm1_max << ','
         << convergence_tail_norm_pm1_mean << ','
         << convergence_effective_jcp_amp_max << ','
         << convergence_effective_jcp_amp_mean << ','
         << convergence_dielectric_amp_max << ','
         << convergence_dielectric_amp_mean << ','
         << convergence_channel_effective_jcp_amp_max[0] << ','
         << convergence_channel_effective_jcp_amp_max[1] << ','
         << convergence_channel_effective_jcp_amp_max[2] << ','
         << convergence_channel_effective_jcp_amp_max[3] << ','
         << convergence_channel_effective_jcp_amp_mean[0] << ','
         << convergence_channel_effective_jcp_amp_mean[1] << ','
         << convergence_channel_effective_jcp_amp_mean[2] << ','
         << convergence_channel_effective_jcp_amp_mean[3] << ','
         << convergence_channel_estimated_error_max[0] << ','
         << convergence_channel_estimated_error_max[1] << ','
         << convergence_channel_estimated_error_max[2] << ','
         << convergence_channel_estimated_error_max[3] << ','
         << convergence_channel_unstable_count[0] << ','
         << convergence_channel_unstable_count[1] << ','
         << convergence_channel_unstable_count[2] << ','
         << convergence_channel_unstable_count[3] << ','
         << convergence_channel_unstable_panels[0] << ','
         << convergence_channel_unstable_panels[1] << ','
         << convergence_channel_unstable_panels[2] << ','
         << convergence_channel_unstable_panels[3] << ','
         << convergence_channel_split_rejects[0] << ','
         << convergence_channel_split_rejects[1] << ','
         << convergence_channel_split_rejects[2] << ','
         << convergence_channel_split_rejects[3] << ','
         << convergence_channel_split_reject_panels[0] << ','
         << convergence_channel_split_reject_panels[1] << ','
         << convergence_channel_split_reject_panels[2] << ','
         << convergence_channel_split_reject_panels[3] << ','
         << convergence_channel_low_abs_reaccepted_count[0] << ','
         << convergence_channel_low_abs_reaccepted_count[1] << ','
         << convergence_channel_low_abs_reaccepted_count[2] << ','
         << convergence_channel_low_abs_reaccepted_count[3] << ','
         << convergence_channel_low_abs_reaccepted_panels[0] << ','
         << convergence_channel_low_abs_reaccepted_panels[1] << ','
         << convergence_channel_low_abs_reaccepted_panels[2] << ','
         << convergence_channel_low_abs_reaccepted_panels[3] << ','
         << convergence_channel_high_error_caught[0] << ','
         << convergence_channel_high_error_caught[1] << ','
         << convergence_channel_high_error_caught[2] << ','
         << convergence_channel_high_error_caught[3] << ','
         << convergence_channel_high_error_missed[0] << ','
         << convergence_channel_high_error_missed[1] << ','
         << convergence_channel_high_error_missed[2] << ','
         << convergence_channel_high_error_missed[3] << ','
         << convergence_channel_high_error_caught_by_split[0] << ','
         << convergence_channel_high_error_caught_by_split[1] << ','
         << convergence_channel_high_error_caught_by_split[2] << ','
         << convergence_channel_high_error_caught_by_split[3] << ','
         << convergence_estimator_high_error_caught << ','
         << convergence_estimator_high_error_missed << ','
         << convergence_estimator_high_error_caught_by_split << ','
         << convergence_estimator_phantom_high_error_count << ','
         << convergence_estimator_phantom_high_error_abs_max << ','
         << convergence_overreject_count << ',' << convergence_overreject_panels
         << ',' << convergence_overreject_caught_count << ','
         << convergence_overreject_caught_panels << ','
         << convergence_overreject_abs_error_max << ','
         << convergence_overreject_abs_error_mean << ','
         << convergence_overreject_abs_error_p95 << ','
         << convergence_overreject_abs_error_p99 << ','
         << convergence_overreject_estimated_abs_error_max << ','
         << convergence_overreject_estimated_abs_error_mean << ','
         << convergence_overreject_required_scale_max << ','
         << convergence_overreject_required_scale_p95 << ','
         << convergence_overreject_required_scale_p99 << ','
         << convergence_channel_overreject_count[0] << ','
         << convergence_channel_overreject_count[1] << ','
         << convergence_channel_overreject_count[2] << ','
         << convergence_channel_overreject_count[3] << ','
         << convergence_channel_overreject_caught_count[0] << ','
         << convergence_channel_overreject_caught_count[1] << ','
         << convergence_channel_overreject_caught_count[2] << ','
         << convergence_channel_overreject_caught_count[3] << ','
         << convergence_channel_overreject_panels[0] << ','
         << convergence_channel_overreject_panels[1] << ','
         << convergence_channel_overreject_panels[2] << ','
         << convergence_channel_overreject_panels[3] << ','
         << convergence_channel_overreject_abs_error_max[0] << ','
         << convergence_channel_overreject_abs_error_max[1] << ','
         << convergence_channel_overreject_abs_error_max[2] << ','
         << convergence_channel_overreject_abs_error_max[3] << ','
         << convergence_channel_overreject_abs_error_mean[0] << ','
         << convergence_channel_overreject_abs_error_mean[1] << ','
         << convergence_channel_overreject_abs_error_mean[2] << ','
         << convergence_channel_overreject_abs_error_mean[3] << ','
         << convergence_channel_overreject_required_scale_max[0] << ','
         << convergence_channel_overreject_required_scale_max[1] << ','
         << convergence_channel_overreject_required_scale_max[2] << ','
         << convergence_channel_overreject_required_scale_max[3] << ','
         << convergence_channel_overreject_required_scale_p99[0] << ','
         << convergence_channel_overreject_required_scale_p99[1] << ','
         << convergence_channel_overreject_required_scale_p99[2] << ','
         << convergence_channel_overreject_required_scale_p99[3] << ','
         << phantom_high_error_count[0] << ',' << phantom_high_error_count[1]
         << ',' << phantom_high_error_count[2] << ','
         << phantom_high_error_count[3] << ',' << phantom_high_error_abs_max[0]
         << ',' << phantom_high_error_abs_max[1] << ','
         << phantom_high_error_abs_max[2] << ','
         << phantom_high_error_abs_max[3] << ','
         << csv_escape_progress(accepted_miss_worst_channel) << ','
         << accepted_miss_count[0] << ',' << accepted_miss_count[1] << ','
         << accepted_miss_count[2] << ',' << accepted_miss_count[3] << ','
         << accepted_miss_panels[0] << ',' << accepted_miss_panels[1] << ','
         << accepted_miss_panels[2] << ',' << accepted_miss_panels[3] << ','
         << accepted_miss_rho_min[0] << ',' << accepted_miss_rho_min[1] << ','
         << accepted_miss_rho_min[2] << ',' << accepted_miss_rho_min[3] << ','
         << accepted_miss_rho_mean[0] << ',' << accepted_miss_rho_mean[1] << ','
         << accepted_miss_rho_mean[2] << ',' << accepted_miss_rho_mean[3] << ','
         << accepted_miss_rho_max[0] << ',' << accepted_miss_rho_max[1] << ','
         << accepted_miss_rho_max[2] << ',' << accepted_miss_rho_max[3] << ','
         << accepted_miss_tail_ratio_mean[0] << ','
         << accepted_miss_tail_ratio_mean[1] << ','
         << accepted_miss_tail_ratio_mean[2] << ','
         << accepted_miss_tail_ratio_mean[3] << ','
         << accepted_miss_tail_ratio_max[0] << ','
         << accepted_miss_tail_ratio_max[1] << ','
         << accepted_miss_tail_ratio_max[2] << ','
         << accepted_miss_tail_ratio_max[3] << ','
         << accepted_miss_estimated_error_mean[0] << ','
         << accepted_miss_estimated_error_mean[1] << ','
         << accepted_miss_estimated_error_mean[2] << ','
         << accepted_miss_estimated_error_mean[3] << ','
         << accepted_miss_estimated_error_max[0] << ','
         << accepted_miss_estimated_error_max[1] << ','
         << accepted_miss_estimated_error_max[2] << ','
         << accepted_miss_estimated_error_max[3] << ','
         << convergence_k3_rho_cap_rejects << ','
         << convergence_k3_rho_cap_reject_panels << ','
         << csv_escape_progress(expansion_audit) << ','
         << csv_escape_progress(expansion_audit_channel) << ',' << unsafe_count
         << ',' << unsafe_panels << ',' << gap_samples << ',' << gap_rejects
         << ',' << mac_convergence_rho_rejects << ','
         << mac_convergence_aspect_rejects << ','
         << mac_convergence_rho_reject_panels << ','
         << mac_convergence_aspect_reject_panels << ','
         << mac_surface_margin_rejects << ','
         << mac_surface_margin_reject_panels << ','
         << mac_surface_margin_applied << ',' << mac_surface_margin_skipped
         << ',' << mac_surface_margin_adaptive_rejects << ','
         << mac_surface_margin_adaptive_reject_panels << ','
         << mac_surface_margin_skipped_accepted_far << ','
         << mac_surface_margin_hard_applied << ','
         << mac_surface_margin_soft_applied << ','
         << mac_surface_margin_hard_rejects << ','
         << mac_surface_margin_hard_reject_panels << ','
         << mac_surface_margin_soft_rejects << ','
         << mac_surface_margin_soft_reject_panels << ',' << fragmented_count
         << ',' << fragmented_panels << ',' << fragmented_rejects << ','
         << fragmentation_gap_max << ',' << fragmentation_gap_mean << ','
         << csv_escape_progress(moment_audit) << ',' << moment_nodes << ','
         << moment_rel_max << ',' << moment_rel_mean << ',' << moment_offset_max
         << ',' << moment_offset_mean << ',' << moment_cancel_max << ','
         << moment_cancel_mean << ',' << moment_dynamic_range << ','
         << expansion_pairs << ',' << expansion_panels << ','
         << expansion_direct_ms / 1000.0 << ',' << expansion_rel_max << ','
         << expansion_rel_mean << ',' << expansion_rel_p95 << ','
         << expansion_rel_p99 << ','
         << csv_escape_progress(expansion_worst_channel) << ','
         << expansion_channel_rel_max[0] << ',' << expansion_channel_rel_max[1]
         << ',' << expansion_channel_rel_max[2] << ','
         << expansion_channel_rel_max[3] << ',' << expansion_channel_rel_mean[0]
         << ',' << expansion_channel_rel_mean[1] << ','
         << expansion_channel_rel_mean[2] << ','
         << expansion_channel_rel_mean[3] << ',' << expansion_channel_rel_p99[0]
         << ',' << expansion_channel_rel_p99[1] << ','
         << expansion_channel_rel_p99[2] << ',' << expansion_channel_rel_p99[3]
         << ',' << (expansion_mean_gate_pass ? 1 : 0) << ','
         << convergence_rho_max << ',' << convergence_rho_mean << ','
         << node_aspect_max << ',' << node_aspect_mean << ','
         << accepted_aspect_max << ',' << accepted_aspect_mean << ','
         << leaf_panel << ',' << leaf_tile_panel << ',' << leaf_residual << ','
         << far_nodes << ',' << leaf_mac_candidates << ',' << leaf_mac_accepted
         << ',' << leaf_mac_rejected << ',' << leaf_mac_far_panels << ','
         << leaf_mac_direct_panels_saved_estimate << ','
         << csv_escape_progress(matvec_cost_profile) << ','
         << leaf_panels_per_second << ','
         << leaf_pair_batch_clean_source_panels_per_second << ','
         << tree_traversal_fraction << ','
         << csv_escape_progress(leaf_work_profile) << ','
         << leaf_work_mac_terminal_panels << ','
         << leaf_work_physical_gap_panels << ','
         << leaf_work_fragmentation_panels << ','
         << leaf_work_convergence_panels << ','
         << leaf_work_convergence_estimator_panels << ','
         << leaf_work_convergence_rho_panels << ','
         << leaf_work_convergence_aspect_panels << ','
         << leaf_work_surface_margin_panels << ','
         << leaf_work_surface_margin_hard_panels << ','
         << leaf_work_surface_margin_soft_panels << ','
         << leaf_work_k3_rho_cap_panels << ','
         << leaf_work_singular_fallback_panels << ','
         << leaf_work_scalar_tail_panels << ','
         << leaf_work_profile_accounted_panels << ','
         << leaf_work_mac_terminal_frac << ',' << leaf_work_convergence_frac
         << ',' << leaf_work_surface_margin_frac << ','
         << leaf_work_singular_fallback_frac << ','
         << leaf_work_scalar_tail_frac << ','
         << csv_escape_progress(cluster_policy) << ',' << target_batch_size
         << ',' << leaf_pair_min_source_panels << ',' << leaf_pair_min_targets
         << ',' << leaf_pair_batch_interactions << ','
         << leaf_pair_batch_targets << ',' << leaf_pair_batch_source_panels
         << ',' << leaf_pair_batch_residual_panels << ','
         << leaf_pair_batch_fallbacks << ',' << leaf_pair_batch_ms / 1000.0
         << ',' << csv_escape_progress(leaf_pair_batch_kernel) << ','
         << leaf_pair_batch_vectorized_sources << ','
         << leaf_pair_batch_vector_lanes << ','
         << leaf_pair_batch_pairs_per_second << ','
         << csv_escape_progress(leaf_kernel_profile) << ','
         << csv_escape_progress(jcp_direct_kernel) << ','
         << leaf_kernel_scalar_tasks << ',' << leaf_kernel_batch_tasks << ','
         << leaf_kernel_source_panel_evals << ','
         << leaf_kernel_scalar_ms / 1000.0 << ','
         << leaf_kernel_target_avx2_ms / 1000.0 << ','
         << leaf_kernel_tail_ms / 1000.0 << ',' << leaf_pair_batch_soa_sources
         << ',' << leaf_pair_batch_soa_ms / 1000.0 << ','
         << leaf_pair_batch_scalar_tail_ms / 1000.0 << ','
         << leaf_pair_batch_clean_source_panels_per_second << ','
         << csv_escape_progress(leaf_pair_selection_profile) << ','
         << csv_escape_progress(leaf_pair_probe_cache) << ','
         << csv_escape_progress(leaf_pair_target_group) << ','
         << leaf_pair_max_lookahead << ','
         << csv_escape_progress(leaf_pair_directness_probe) << ','
         << csv_escape_progress(leaf_pair_directness_cache) << ','
         << csv_escape_progress(leaf_pair_directness_verify) << ','
         << leaf_pair_selection_candidate_scans << ','
         << leaf_pair_selection_clean_rejects << ','
         << leaf_pair_selection_probe_calls << ','
         << leaf_pair_selection_probe_node_visits << ','
         << leaf_pair_selection_probe_ms / 1000.0 << ','
         << leaf_pair_selection_ms / 1000.0 << ','
         << leaf_pair_selection_probe_cache_hits << ','
         << leaf_pair_selection_probe_cache_misses << ','
         << leaf_pair_selection_insufficient_targets << ','
         << leaf_pair_selection_same_target_leaf_rejects << ','
         << leaf_pair_directness_ancestor_probe_calls << ','
         << leaf_pair_directness_ancestor_probe_node_visits << ','
         << leaf_pair_directness_ancestor_probe_ms / 1000.0 << ','
         << leaf_pair_directness_pair_cache_hits << ','
         << leaf_pair_directness_pair_cache_misses << ','
         << leaf_pair_directness_all_direct_proofs << ','
         << leaf_pair_directness_mixed_fallbacks << ','
         << leaf_pair_directness_verify_checks << ','
         << leaf_pair_directness_verify_mismatches << ','
         << csv_escape_progress(leaf_pair_scheduler) << ','
         << leaf_pair_worklist_max_active << ','
         << csv_escape_progress(leaf_pair_worklist_verify) << ','
         << leaf_pair_worklist_tasks << ',' << leaf_pair_worklist_groups << ','
         << leaf_pair_worklist_batched_groups << ','
         << leaf_pair_worklist_scalar_flush_tasks << ','
         << leaf_pair_worklist_forced_flushes << ','
         << leaf_pair_worklist_active_groups_max << ','
         << leaf_pair_worklist_ms / 1000.0 << ','
         << leaf_pair_worklist_verify_checks << ','
         << leaf_pair_worklist_verify_mismatches << ','
         << csv_escape_progress(leaf_block_verify) << ','
         << leaf_block_target_leaves << ',' << leaf_block_targets << ','
         << leaf_block_source_groups << ',' << leaf_block_batched_groups << ','
         << leaf_block_scalar_tail_tasks << ',' << leaf_block_group_ms / 1000.0
         << ',' << leaf_block_flush_ms / 1000.0 << ','
         << leaf_block_writeback_ms / 1000.0 << ',' << leaf_block_verify_checks
         << ',' << leaf_block_verify_mismatches << ',' << singular << ','
         << csv_escape_progress(result.preconditioner_effective) << ','
         << result.preconditioner_block_panel_cap << ','
         << result.preconditioner_overlap_panel_cap << ','
         << result.preconditioner_ilu_neighbor_cap_requested << ','
         << result.preconditioner_ilu_neighbor_cap << ','
         << result.preconditioner_ilu_drop_tol << ','
         << csv_escape_progress(result.preconditioner_ilu_regularization) << ','
         << result.preconditioner_ilu_regularization_max << ','
         << result.preconditioner_ilu_regularization_selected << ','
         << result.preconditioner_ilu_regularization_absolute_max << ','
         << result.preconditioner_ilu_regularization_attempts << ','
         << result.preconditioner_ilu_regularization_rejected_attempts << ','
         << result.preconditioner_ilu_regularization_shifted_rows << ','
         << csv_escape_progress(
                result.preconditioner_ilu_unshifted_failure_stage)
         << ',' << result.preconditioner_memory_budget_bytes << ','
         << result.preconditioner_estimated_bytes << ','
         << result.preconditioner_valid_blocks << ','
         << result.preconditioner_valid_panels << ','
         << result.preconditioner_ill_conditioned_blocks << ','
         << result.preconditioner_quality_rejected_blocks << ','
         << result.preconditioner_quality_probe_max_gain << ','
         << result.preconditioner_quality_probe_rel_residual << ','
         << result.preconditioner_pivot_ratio_min << ','
         << result.preconditioner_factor_growth_max << ','
         << result.preconditioner_pivot_ratio_min_row << ','
         << result.preconditioner_pivot_ratio_min_column << ','
         << result.preconditioner_factor_growth_row << ','
         << result.preconditioner_factor_growth_column << ','
         << csv_escape_progress(result.preconditioner_quality_failure_stage)
         << ',' << result.preconditioner_quality_failure_row << ','
         << result.preconditioner_quality_failure_column << ','
         << result.preconditioner_quality_failure_probe << ','
         << result.preconditioner_quality_failure_probe_gain << ','
         << result.preconditioner_quality_failure_probe_rel_residual << ','
         << csv_escape_progress(result.preconditioner_quality_status) << ','
         << result.preconditioner_setup_ms / 1000.0 << ','
         << result.preconditioner_graph_setup_ms / 1000.0 << ','
         << (result.preconditioner_graph_cache_reused ? "true" : "false")
         << ','
         << result.preconditioner_near_assembly_setup_ms / 1000.0 << ','
         << result.preconditioner_ilu_factor_setup_ms / 1000.0 << ','
         << result.preconditioner_apply_ms / 1000.0 << ','
         << result.preconditioner_apply_calls << ','
         << result.preconditioner_sparse_nnz_blocks << ','
         << result.preconditioner_sparse_avg_degree << ','
         << result.preconditioner_sparse_max_degree << ','
         << result.preconditioner_coarse_aggregate_cap << ','
         << result.preconditioner_coarse_aggregates << ','
         << result.preconditioner_coarse_dimension << ','
         << csv_escape_progress(result.preconditioner_coarse_base) << ','
         << result.preconditioner_coarse_matrix_bytes << ','
         << result.preconditioner_coarse_pivot_ratio << ','
         << csv_escape_progress(result.preconditioner_coarse_status) << ','
         << result.preconditioner_coarse_setup_matvec_calls << ','
         << result.preconditioner_coarse_setup_ms / 1000.0 << ','
         << result.preconditioner_coarse_moments_setup_ms / 1000.0 << ','
         << result.preconditioner_coarse_far_setup_ms / 1000.0 << ','
         << result.preconditioner_coarse_audit_setup_ms / 1000.0 << ','
         << result.preconditioner_coarse_apply_ms / 1000.0 << ','
         << result.preconditioner_coarse_apply_calls << ','
         << result.preconditioner_coarse_rp_identity_error_max << ','
         << result.preconditioner_coarse_verify_rel_error_max << ','
         << result.orthogonalization_ms / 1000.0 << ','
         << result.preconditioner_ms / 1000.0 << ','
         << (result.fmm_local_gmres_active ? "true" : "false") << ','
         << result.fmm_local_gmres_matvec_calls << ','
         << result.fmm_local_gmres_apply_seconds_total << ','
         << result.fmm_local_gmres_apply_seconds_mean << ','
         << result.fmm_local_moment_seconds_total << ','
         << result.fmm_local_m2l_seconds_total << ','
         << result.fmm_local_fallback_seconds_total << ','
         << result.fmm_local_l2p_seconds_total << ',' << threads_ << ','
         << current_working_set_bytes() << ',' << csv_escape_progress(status)
         << '\n';
    out_.flush();
  }

private:
  bool enabled_{false};
  std::chrono::seconds interval_{5};
  std::chrono::steady_clock::time_point start_{};
  std::chrono::steady_clock::time_point last_write_{};
  std::size_t threads_{1};
  std::ofstream out_;
};

std::string residual_history_string(const std::vector<double> &history) {
  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss.precision(6);
  for (std::size_t i = 0; i < history.size(); ++i) {
    if (i != 0)
      oss << ';';
    oss << history[i];
  }
  return oss.str();
}

FormulationConfig make_formulation_config(const Options &options) {
  FormulationConfig config;
  config.formulation = options.formulation;
  config.energy_projection = options.energy_projection;
  if (config.formulation.empty())
    config.formulation = "jcp2013-centroid";
  if (config.formulation == "jcp2013")
    config.formulation = "jcp2013-centroid";
  if (config.energy_projection.empty())
    config.energy_projection = "boundary";
  if (config.formulation == "jcp2013-centroid" ||
      config.formulation == "jcp2013-treecode" ||
      config.formulation == "tabi2-nodepatch") {
    // The jcp2013 path is intentionally explicit even while it shares the
    // current scale factors. Accuracy gates use the audit CSV to keep future
    // sign/scale changes visible instead of silently drifting the formulation.
    config.rhs_scale = 1.0;
    config.operator_contrast_scale = 1.0;
    config.boundary_energy_scale = 1.0;
  } else if (config.formulation == "calibrated-debug") {
    config.rhs_scale = 1.0;
    config.operator_contrast_scale = 1.0;
    config.boundary_energy_scale = 1.0;
  }
  return config;
}

struct LinearSolveResult {
  std::vector<double> x;
  std::vector<double> best_x;
  int iterations{0};
  double residual{0.0};
  int restart_requested{32};
  int restart_effective{32};
  double best_residual{0.0};
  int best_residual_iter{0};
  double final_residual{0.0};
  double best_preconditioned_residual{0.0};
  double final_preconditioned_residual{0.0};
  double true_residual_abs{0.0};
  double true_residual_rel{0.0};
  std::string true_residual_probe_mode{"restart"};
  std::string true_residual_probe_scope{"full"};
  int true_residual_probe_interval{8};
  int true_residual_sample_count{1024};
  std::size_t true_residual_probe_calls{0};
  double true_residual_probe_ms{0.0};
  double sampled_true_residual_rel{0.0};
  std::size_t sampled_true_residual_rows{0};
  std::size_t sampled_probe_calls{0};
  double sampled_probe_ms{0.0};
  std::size_t full_true_residual_confirm_calls{0};
  double full_true_residual_confirm_ms{0.0};
  bool converged{false};
  std::string convergence_reason{"not_started"};
  std::vector<double> residual_history;
  double orthogonalization_ms{0.0};
  double preconditioner_ms{0.0};
  double preconditioner_apply_ms{0.0};
  std::size_t preconditioner_apply_calls{0};
};

using GmresProgressCallback = std::function<void(
    int gmres_iter, int restart_iter, double true_residual_rel,
    double preconditioned_residual_rel, bool force, const std::string &phase,
    const LinearSolveResult &state)>;

struct SampledResidualResult {
  double abs{0.0};
  double rel{std::numeric_limits<double>::infinity()};
  std::size_t rows{0};
  bool finite{false};
};

using SampledTrueResidualCallback = std::function<SampledResidualResult(
    const std::vector<double> &, const std::vector<std::size_t> &)>;

LinearSolveResult gmres_solve(
    const std::function<void(const std::vector<double> &,
                             std::vector<double> &)> &matvec,
    const std::vector<double> &b, const std::vector<double> &diagonal,
    int max_iter, double tol, int restart_requested, TitanExecutor &executor,
    const std::function<void(const std::vector<double> &,
                             std::vector<double> &)> &custom_preconditioner =
        {},
    const GmresProgressCallback &progress_callback = {},
    const std::string &true_residual_probe_mode = "restart",
    int true_residual_probe_interval = 8,
    const std::string &true_residual_probe_scope = "full",
    int true_residual_sample_count = 1024,
    const SampledTrueResidualCallback &sampled_true_residual = {}) {
  LinearSolveResult result;
  const std::size_t n = b.size();
  result.x.assign(n, 0.0);
  result.restart_requested = std::clamp(restart_requested, 8, 128);
  result.restart_effective = result.restart_requested;
  result.true_residual_probe_mode =
      true_residual_probe_mode == "interval"
          ? std::string("interval")
          : (true_residual_probe_mode == "off" ? std::string("off")
                                               : std::string("restart"));
  result.true_residual_probe_scope =
      true_residual_probe_scope == "sampled"
          ? std::string("sampled")
          : (true_residual_probe_scope == "adaptive" ? std::string("adaptive")
                                                     : std::string("full"));
  result.true_residual_probe_interval =
      std::clamp(true_residual_probe_interval, 1, 128);
  result.true_residual_sample_count =
      std::clamp(true_residual_sample_count, 128, 4096);
  if (n == 0) {
    result.converged = true;
    result.convergence_reason = "converged";
    result.best_residual = 0.0;
    result.best_residual_iter = 0;
    result.final_residual = 0.0;
    result.best_preconditioned_residual = 0.0;
    result.final_preconditioned_residual = 0.0;
    result.true_residual_abs = 0.0;
    result.true_residual_rel = 0.0;
    result.true_residual_probe_calls = 0;
    result.true_residual_probe_ms = 0.0;
    result.sampled_true_residual_rel = 0.0;
    result.sampled_true_residual_rows = 0;
    result.sampled_probe_calls = 0;
    result.sampled_probe_ms = 0.0;
    result.full_true_residual_confirm_calls = 0;
    result.full_true_residual_confirm_ms = 0.0;
    return result;
  }

  std::vector<double> inv_precond(n, 1.0);
  const auto precond_begin = std::chrono::steady_clock::now();
  std::atomic_bool preconditioner_ok{true};
  parallel_for_each_index(n, executor, [&](std::size_t i) {
    const double diag = i < diagonal.size() ? std::abs(diagonal[i]) : 1.0;
    if (!std::isfinite(diag) || diag <= 1.0e-12) {
      inv_precond[i] = 1.0;
      preconditioner_ok.store(false, std::memory_order_relaxed);
    } else {
      inv_precond[i] = 1.0 / diag;
    }
  });
  result.preconditioner_ms +=
      elapsed_ms(precond_begin, std::chrono::steady_clock::now());
  if (!preconditioner_ok.load(std::memory_order_relaxed))
    result.convergence_reason = "preconditioner_breakdown";

  auto apply_preconditioner = [&](const std::vector<double> &in,
                                  std::vector<double> &out) {
    const auto begin = std::chrono::steady_clock::now();
    if (custom_preconditioner) {
      custom_preconditioner(in, out);
      if (out.size() != n) {
        out.resize(n);
        for (std::size_t i = 0; i < n; ++i)
          out[i] = in[i] * inv_precond[i];
      }
    } else {
      out.resize(n);
      parallel_for_each_index(
          n, executor, [&](std::size_t i) { out[i] = in[i] * inv_precond[i]; });
    }
    const double apply_ms = elapsed_ms(begin, std::chrono::steady_clock::now());
    result.preconditioner_ms += apply_ms;
    result.preconditioner_apply_ms += apply_ms;
    ++result.preconditioner_apply_calls;
  };

  auto finite_vector = [](const std::vector<double> &values) noexcept {
    for (double value : values) {
      if (!std::isfinite(value))
        return false;
    }
    return true;
  };

  const int restart = result.restart_effective;
  std::vector<double> ax(n, 0.0);
  std::vector<double> residual = b;
  std::vector<double> precond_residual(n, 0.0);
  apply_preconditioner(residual, precond_residual);
  if (!finite_vector(precond_residual)) {
    result.convergence_reason = "preconditioner_breakdown";
    return result;
  }
  const double b_norm_raw = l2_norm_parallel(b, executor);
  const double b_norm = std::max(1.0e-300, b_norm_raw);
  const double precond_norm_raw = l2_norm_parallel(precond_residual, executor);
  const double precond_norm = std::max(1.0e-300, precond_norm_raw);
  auto relative_true_residual = [&](const std::vector<double> &r) {
    const double abs_value = l2_norm_parallel(r, executor);
    return std::pair<double, double>{abs_value, abs_value / b_norm};
  };
  auto preconditioned_relative = [&](const std::vector<double> &r) {
    return l2_norm_parallel(r, executor) / precond_norm;
  };
  auto build_sample_rows = [](std::size_t row_count, int requested) {
    std::vector<std::size_t> rows;
    if (row_count == 0 || requested <= 0)
      return rows;
    const std::size_t m =
        std::min<std::size_t>(row_count, static_cast<std::size_t>(requested));
    rows.reserve(m);
    std::size_t previous = std::numeric_limits<std::size_t>::max();
    for (std::size_t k = 0; k < m; ++k) {
      std::size_t row =
          ((2u * k + 1u) * row_count) / std::max<std::size_t>(1, 2u * m);
      row = std::min(row, row_count - 1u);
      if (row != previous) {
        rows.push_back(row);
        previous = row;
      }
    }
    if (rows.empty())
      rows.push_back(0);
    return rows;
  };
  const std::vector<std::size_t> sampled_rows =
      build_sample_rows(n, result.true_residual_sample_count);
  auto sampled_relative_from_full_ax =
      [&](const std::vector<double> &ax_values) {
        double residual_sq = 0.0;
        double b_sq = 0.0;
        for (std::size_t row : sampled_rows) {
          const double r = b[row] - ax_values[row];
          residual_sq += r * r;
          b_sq += b[row] * b[row];
        }
        const double abs_value = std::sqrt(std::max(0.0, residual_sq));
        const double denom = std::max(1.0e-300, std::sqrt(std::max(0.0, b_sq)));
        SampledResidualResult sample;
        sample.abs = abs_value;
        sample.rel = abs_value / denom;
        sample.rows = sampled_rows.size();
        sample.finite = std::isfinite(sample.rel);
        return sample;
      };

  auto [initial_abs, initial_rel] = relative_true_residual(residual);
  result.residual = initial_rel;
  result.best_residual = initial_rel;
  result.best_residual_iter = 0;
  result.final_residual = initial_rel;
  result.true_residual_abs = initial_abs;
  result.true_residual_rel = initial_rel;
  result.best_x = result.x;
  double precond_rel = preconditioned_relative(precond_residual);
  result.best_preconditioned_residual = precond_rel;
  result.final_preconditioned_residual = precond_rel;
  result.residual_history.push_back(initial_rel);
  if (result.residual <= tol) {
    result.converged = true;
    result.convergence_reason = "converged_true_residual";
    return result;
  }

  const int iter_cap = std::max(1, max_iter);
  double best_true_abs = initial_abs;
  int non_improving_restarts = 0;
  bool saw_phantom_preconditioned = false;
  // Confirm an apparently converged left-preconditioned residual immediately.
  // If the full residual rejects it, wait for another decade of estimated
  // reduction before paying for the next confirmation matvec.
  double next_preconditioned_confirmation = tol;
  // Retain the Krylov basis and work vectors across restarts. On large p4
  // solves these buffers are tens of MiB; reallocating them at every restart
  // adds allocator and page-fault noise without changing GMRES arithmetic.
  std::vector<std::vector<double>> krylov_basis;
  std::vector<double> h_workspace;
  std::vector<double> cs_workspace;
  std::vector<double> sn_workspace;
  std::vector<double> g_workspace;
  std::vector<double> w_workspace(n, 0.0);
  std::vector<double> precond_w_workspace(n, 0.0);
  while (result.iterations < iter_cap) {
    const double beta = l2_norm_parallel(precond_residual, executor);
    if (!std::isfinite(beta) || beta <= 1.0e-300) {
      result.convergence_reason = "preconditioner_breakdown";
      return result;
    }
    if (beta <= tol * precond_norm) {
      saw_phantom_preconditioned = true;
      auto [true_abs, true_rel] = relative_true_residual(residual);
      result.final_residual = true_rel;
      result.true_residual_abs = true_abs;
      result.true_residual_rel = true_rel;
      result.final_preconditioned_residual = beta / precond_norm;
      if (std::isfinite(result.final_preconditioned_residual) &&
          result.final_preconditioned_residual <
              result.best_preconditioned_residual) {
        result.best_preconditioned_residual =
            result.final_preconditioned_residual;
      }
      if (true_rel <= tol) {
        result.converged = true;
        result.residual = true_rel;
        result.convergence_reason = "converged_true_residual";
        break;
      }
    }
    if (result.converged)
      break;

    const int window = std::min(restart, iter_cap - result.iterations);
    if (krylov_basis.size() < static_cast<std::size_t>(window) + 1)
      krylov_basis.resize(static_cast<std::size_t>(window) + 1);
    for (int basis = 0; basis <= window; ++basis) {
      std::vector<double> &vector =
          krylov_basis[static_cast<std::size_t>(basis)];
      if (vector.size() != n)
        vector.assign(n, 0.0);
    }
    auto &v = krylov_basis;
    parallel_for_each_index(n, executor, [&](std::size_t i) {
      v[0][i] = precond_residual[i] / beta;
    });
    h_workspace.assign((static_cast<std::size_t>(window) + 1) *
                           static_cast<std::size_t>(window),
                       0.0);
    cs_workspace.assign(static_cast<std::size_t>(window), 0.0);
    sn_workspace.assign(static_cast<std::size_t>(window), 0.0);
    g_workspace.assign(static_cast<std::size_t>(window) + 1, 0.0);
    auto &h = h_workspace;
    auto &cs = cs_workspace;
    auto &sn = sn_workspace;
    auto &g = g_workspace;
    g[0] = beta;

    auto solve_window_y = [&](int cols) {
      std::vector<double> y(static_cast<std::size_t>(cols), 0.0);
      for (int rev = 0; rev < cols; ++rev) {
        const int row = cols - 1 - rev;
        double value = g[static_cast<std::size_t>(row)];
        for (int col = row + 1; col < cols; ++col) {
          value -= h[static_cast<std::size_t>(row) *
                         static_cast<std::size_t>(window) +
                     static_cast<std::size_t>(col)] *
                   y[static_cast<std::size_t>(col)];
        }
        const double diag =
            h[static_cast<std::size_t>(row) * static_cast<std::size_t>(window) +
              static_cast<std::size_t>(row)];
        y[static_cast<std::size_t>(row)] =
            std::abs(diag) > 1.0e-30 ? value / diag : 0.0;
      }
      return y;
    };

    auto build_candidate_x = [&](const std::vector<double> &base,
                                 const std::vector<double> &y, int cols,
                                 std::vector<double> &candidate) {
      candidate = base;
      parallel_for_each_index(n, executor, [&](std::size_t k) {
        double update = 0.0;
        for (int basis = 0; basis < cols; ++basis)
          update += y[static_cast<std::size_t>(basis)] *
                    v[static_cast<std::size_t>(basis)][k];
        candidate[k] += update;
      });
    };

    int used_cols = 0;
    bool happy_breakdown = false;
    bool preconditioned_requested_check = false;
    bool probe_converged = false;
    for (int j = 0; j < window; ++j) {
      std::vector<double> &w = w_workspace;
      matvec(v[static_cast<std::size_t>(j)], w);
      if (!finite_vector(w)) {
        result.convergence_reason = "nan_guard";
        return result;
      }
      std::vector<double> &precond_w = precond_w_workspace;
      apply_preconditioner(w, precond_w);
      if (!finite_vector(precond_w)) {
        result.convergence_reason = "preconditioner_breakdown";
        return result;
      }
      w.swap(precond_w);

      const auto ortho_begin = std::chrono::steady_clock::now();
      for (int i = 0; i <= j; ++i) {
        const double hij =
            dot_parallel(w, v[static_cast<std::size_t>(i)], executor);
        h[static_cast<std::size_t>(i) * static_cast<std::size_t>(window) +
          static_cast<std::size_t>(j)] = hij;
        parallel_for_each_index(n, executor, [&](std::size_t k) {
          w[k] -= hij * v[static_cast<std::size_t>(i)][k];
        });
      }
      const double h_next = l2_norm_parallel(w, executor);
      h[(static_cast<std::size_t>(j) + 1) * static_cast<std::size_t>(window) +
        static_cast<std::size_t>(j)] = h_next;
      if (h_next > 1.0e-20 && j + 1 < window + 1) {
        parallel_for_each_index(n, executor, [&](std::size_t k) {
          v[static_cast<std::size_t>(j) + 1][k] = w[k] / h_next;
        });
      } else {
        happy_breakdown = true;
      }
      result.orthogonalization_ms +=
          elapsed_ms(ortho_begin, std::chrono::steady_clock::now());

      for (int i = 0; i < j; ++i) {
        const double h_i =
            h[static_cast<std::size_t>(i) * static_cast<std::size_t>(window) +
              static_cast<std::size_t>(j)];
        const double h_next_i = h[(static_cast<std::size_t>(i) + 1) *
                                      static_cast<std::size_t>(window) +
                                  static_cast<std::size_t>(j)];
        h[static_cast<std::size_t>(i) * static_cast<std::size_t>(window) +
          static_cast<std::size_t>(j)] =
            cs[static_cast<std::size_t>(i)] * h_i +
            sn[static_cast<std::size_t>(i)] * h_next_i;
        h[(static_cast<std::size_t>(i) + 1) * static_cast<std::size_t>(window) +
          static_cast<std::size_t>(j)] =
            -sn[static_cast<std::size_t>(i)] * h_i +
            cs[static_cast<std::size_t>(i)] * h_next_i;
      }

      const double h_j =
          h[static_cast<std::size_t>(j) * static_cast<std::size_t>(window) +
            static_cast<std::size_t>(j)];
      const double h_next_j = h[(static_cast<std::size_t>(j) + 1) *
                                    static_cast<std::size_t>(window) +
                                static_cast<std::size_t>(j)];
      const double rho = std::hypot(h_j, h_next_j);
      if (rho <= 1.0e-30 || !std::isfinite(rho)) {
        happy_breakdown = true;
        h[static_cast<std::size_t>(j) * static_cast<std::size_t>(window) +
          static_cast<std::size_t>(j)] = h_j;
        h[(static_cast<std::size_t>(j) + 1) * static_cast<std::size_t>(window) +
          static_cast<std::size_t>(j)] = 0.0;
      } else {
        cs[static_cast<std::size_t>(j)] = h_j / rho;
        sn[static_cast<std::size_t>(j)] = h_next_j / rho;
        h[static_cast<std::size_t>(j) * static_cast<std::size_t>(window) +
          static_cast<std::size_t>(j)] = rho;
        h[(static_cast<std::size_t>(j) + 1) * static_cast<std::size_t>(window) +
          static_cast<std::size_t>(j)] = 0.0;
        const double g_j = g[static_cast<std::size_t>(j)];
        const double g_next = g[static_cast<std::size_t>(j) + 1];
        g[static_cast<std::size_t>(j)] =
            cs[static_cast<std::size_t>(j)] * g_j +
            sn[static_cast<std::size_t>(j)] * g_next;
        g[static_cast<std::size_t>(j) + 1] =
            -sn[static_cast<std::size_t>(j)] * g_j +
            cs[static_cast<std::size_t>(j)] * g_next;
      }

      ++result.iterations;
      used_cols = j + 1;
      precond_rel = std::abs(g[static_cast<std::size_t>(j) + 1]) / precond_norm;
      result.final_preconditioned_residual = precond_rel;
      if (std::isfinite(precond_rel) &&
          precond_rel < result.best_preconditioned_residual) {
        result.best_preconditioned_residual = precond_rel;
      }
      if (progress_callback)
        progress_callback(result.iterations, j + 1, result.final_residual,
                          precond_rel, false, "gmres_iter", result);
      bool full_probe_ran_this_iteration = false;
      if (result.true_residual_probe_mode == "interval" &&
          result.true_residual_probe_interval > 0 &&
          (result.iterations % result.true_residual_probe_interval) == 0) {
        std::vector<double> y_probe = solve_window_y(used_cols);
        std::vector<double> probe_x;
        build_candidate_x(result.x, y_probe, used_cols, probe_x);
        std::vector<double> probe_ax;
        bool probe_ax_ready = false;
        bool run_full_probe = result.true_residual_probe_scope == "full";
        if (result.true_residual_probe_scope == "sampled" ||
            result.true_residual_probe_scope == "adaptive") {
          const auto sample_begin = std::chrono::steady_clock::now();
          SampledResidualResult sample;
          if (sampled_true_residual) {
            sample = sampled_true_residual(probe_x, sampled_rows);
          } else {
            probe_ax.assign(n, 0.0);
            matvec(probe_x, probe_ax);
            probe_ax_ready = true;
            if (!finite_vector(probe_ax)) {
              result.convergence_reason = "nan_guard";
              return result;
            }
            sample = sampled_relative_from_full_ax(probe_ax);
          }
          ++result.sampled_probe_calls;
          result.sampled_probe_ms +=
              elapsed_ms(sample_begin, std::chrono::steady_clock::now());
          result.sampled_true_residual_rel = sample.rel;
          result.sampled_true_residual_rows = sample.rows;
          if (!sample.finite) {
            result.convergence_reason = "nan_guard";
            return result;
          }
          if (progress_callback)
            progress_callback(result.iterations, j + 1, sample.rel, precond_rel,
                              true, "gmres_sampled_true_probe", result);
          if (result.true_residual_probe_scope == "adaptive" &&
              sample.rel <= std::max(5.0 * tol, 1.0e-3)) {
            run_full_probe = true;
          }
        }
        if (run_full_probe) {
          full_probe_ran_this_iteration = true;
          const auto probe_begin = std::chrono::steady_clock::now();
          if (!probe_ax_ready) {
            probe_ax.assign(n, 0.0);
            matvec(probe_x, probe_ax);
            probe_ax_ready = true;
            if (!finite_vector(probe_ax)) {
              result.convergence_reason = "nan_guard";
              return result;
            }
          }
          std::vector<double> probe_residual(n, 0.0);
          parallel_for_each_index(n, executor, [&](std::size_t k) {
            probe_residual[k] = b[k] - probe_ax[k];
          });
          auto [probe_abs, probe_rel] = relative_true_residual(probe_residual);
          ++result.true_residual_probe_calls;
          ++result.full_true_residual_confirm_calls;
          const double probe_ms =
              elapsed_ms(probe_begin, std::chrono::steady_clock::now());
          result.true_residual_probe_ms += probe_ms;
          result.full_true_residual_confirm_ms += probe_ms;
          result.final_residual = probe_rel;
          result.true_residual_abs = probe_abs;
          result.true_residual_rel = probe_rel;
          if (std::isfinite(probe_rel) && probe_rel < result.best_residual) {
            result.best_residual = probe_rel;
            result.best_residual_iter = result.iterations;
            best_true_abs = probe_abs;
            result.best_x = probe_x;
          }
          result.residual_history.push_back(probe_rel);
          if (progress_callback)
            progress_callback(result.iterations, j + 1, probe_rel, precond_rel,
                              true, "gmres_true_probe", result);
          if (probe_rel <= tol) {
            result.x = std::move(probe_x);
            result.residual = probe_rel;
            result.converged = true;
            result.convergence_reason = "converged_true_residual_probe";
            probe_converged = true;
            break;
          }
          next_preconditioned_confirmation = std::max(
              1.0e-300, std::min(next_preconditioned_confirmation,
                                 0.1 * std::max(precond_rel, 1.0e-300)));
        }
      }
      if (!probe_converged && !full_probe_ran_this_iteration &&
          std::isfinite(precond_rel) &&
          precond_rel <= next_preconditioned_confirmation) {
        preconditioned_requested_check = true;
        saw_phantom_preconditioned = true;

        std::vector<double> y_confirm = solve_window_y(used_cols);
        std::vector<double> confirm_x;
        build_candidate_x(result.x, y_confirm, used_cols, confirm_x);
        std::vector<double> confirm_ax(n, 0.0);
        const auto confirm_begin = std::chrono::steady_clock::now();
        matvec(confirm_x, confirm_ax);
        if (!finite_vector(confirm_ax)) {
          result.convergence_reason = "nan_guard";
          return result;
        }
        std::vector<double> confirm_residual(n, 0.0);
        parallel_for_each_index(n, executor, [&](std::size_t k) {
          confirm_residual[k] = b[k] - confirm_ax[k];
        });
        auto [confirm_abs, confirm_rel] =
            relative_true_residual(confirm_residual);
        ++result.true_residual_probe_calls;
        ++result.full_true_residual_confirm_calls;
        const double confirm_ms =
            elapsed_ms(confirm_begin, std::chrono::steady_clock::now());
        result.true_residual_probe_ms += confirm_ms;
        result.full_true_residual_confirm_ms += confirm_ms;
        result.final_residual = confirm_rel;
        result.true_residual_abs = confirm_abs;
        result.true_residual_rel = confirm_rel;
        if (std::isfinite(confirm_rel) && confirm_rel < result.best_residual) {
          result.best_residual = confirm_rel;
          result.best_residual_iter = result.iterations;
          best_true_abs = confirm_abs;
          result.best_x = confirm_x;
        }
        result.residual_history.push_back(confirm_rel);
        if (progress_callback)
          progress_callback(result.iterations, j + 1, confirm_rel, precond_rel,
                            true, "gmres_early_true_confirmation", result);
        if (confirm_rel <= tol) {
          result.x = std::move(confirm_x);
          result.residual = confirm_rel;
          result.converged = true;
          result.convergence_reason = "converged_early_true_residual";
          probe_converged = true;
          break;
        }
        next_preconditioned_confirmation =
            std::max(1.0e-300, 0.1 * std::max(precond_rel, 1.0e-300));
      }
      if (precond_rel <= tol) {
        preconditioned_requested_check = true;
        saw_phantom_preconditioned = true;
      }
      if (result.iterations >= iter_cap)
        break;
      if (happy_breakdown)
        break;
    }

    if (probe_converged)
      break;
    if (used_cols == 0)
      break;

    std::vector<double> y = solve_window_y(used_cols);
    build_candidate_x(result.x, y, used_cols, result.x);

    matvec(result.x, ax);
    if (!finite_vector(ax)) {
      result.convergence_reason = "nan_guard";
      return result;
    }
    parallel_for_each_index(n, executor,
                            [&](std::size_t k) { residual[k] = b[k] - ax[k]; });
    auto [true_abs, true_rel] = relative_true_residual(residual);
    result.final_residual = true_rel;
    result.true_residual_abs = true_abs;
    result.true_residual_rel = true_rel;
    result.residual = true_rel;
    bool improved_true = false;
    if (std::isfinite(true_rel) && true_rel < result.best_residual) {
      result.best_residual = true_rel;
      result.best_residual_iter = result.iterations;
      best_true_abs = true_abs;
      result.best_x = result.x;
      improved_true = true;
    }
    result.residual_history.push_back(true_rel);
    apply_preconditioner(residual, precond_residual);
    if (!finite_vector(precond_residual)) {
      result.convergence_reason = "preconditioner_breakdown";
      return result;
    }
    precond_rel = preconditioned_relative(precond_residual);
    result.final_preconditioned_residual = precond_rel;
    if (std::isfinite(precond_rel) &&
        precond_rel < result.best_preconditioned_residual) {
      result.best_preconditioned_residual = precond_rel;
    }
    if (progress_callback)
      progress_callback(result.iterations, 0, true_rel, precond_rel, true,
                        "gmres_restart", result);
    if (true_rel <= tol) {
      result.converged = true;
      result.convergence_reason =
          preconditioned_requested_check
              ? "converged_true_residual_after_preconditioned_check"
              : "converged_true_residual";
      break;
    }
    if (preconditioned_requested_check && true_rel > tol) {
      saw_phantom_preconditioned = true;
      non_improving_restarts = 0;
    } else if (!improved_true) {
      ++non_improving_restarts;
    } else {
      non_improving_restarts = 0;
    }
    if (non_improving_restarts >= 2) {
      result.convergence_reason =
          result.iterations >= iter_cap ? "max_iter" : "stagnated";
      break;
    }
  }
  if (!result.best_x.empty()) {
    result.x = result.best_x;
    result.residual = result.best_residual;
    result.true_residual_abs = best_true_abs;
    result.true_residual_rel = result.best_residual;
  }
  if (!result.converged && result.best_residual <= tol) {
    result.converged = true;
    result.convergence_reason = result.final_residual <= tol
                                    ? "converged_true_residual"
                                    : "converged_true_residual_best";
  }
  if (!result.converged && result.convergence_reason == "not_started")
    result.convergence_reason =
        saw_phantom_preconditioned
            ? "phantom_preconditioned"
            : (result.iterations >= iter_cap ? "max_iter" : "stagnated");
  return result;
}

// TABI v1 convention: linear boundary-integral solve for the induced
// surface density. Kernels use Angstrom distances, elementary-charge inputs,
// inverse-Angstrom kappa, outward normals, and Coulomb conversion only in the
// final reaction-field energy projection.
std::vector<double> build_rhs(const std::vector<Atom> &atoms,
                              const std::vector<Panel> &panels, double contrast,
                              double kappa, double rhs_scale) {
  std::vector<double> rhs(panels.size(), 0.0);
  for (std::size_t p = 0; p < panels.size(); ++p) {
    const Panel &panel = panels[p];
    double normal_field = 0.0;
    for (const Atom &atom : atoms) {
      const double dx = panel.x - atom.x;
      const double dy = panel.y - atom.y;
      const double dz = panel.z - atom.z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= 1.0e-16)
        continue;
      const double r = std::sqrt(r2);
      const double normal_projection =
          (panel.nx * dx + panel.ny * dy + panel.nz * dz) / r;
      const double screened = std::exp(-kappa * r);
      normal_field +=
          atom.q * screened * normal_projection / std::max(1.0e-12, r2);
    }
    rhs[p] = -rhs_scale * contrast * normal_field / (4.0 * kPi);
  }
  return rhs;
}

double boundary_energy_kcal(const std::vector<Atom> &atoms,
                            const std::vector<Panel> &panels,
                            const std::vector<double> &density, double kappa) {
  double energy = 0.0;
  for (const Atom &atom : atoms) {
    double phi = 0.0;
    for (std::size_t p = 0; p < panels.size(); ++p) {
      const Panel &panel = panels[p];
      const double dx = atom.x - panel.x;
      const double dy = atom.y - panel.y;
      const double dz = atom.z - panel.z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= 1.0e-16)
        continue;
      const double r = std::sqrt(r2);
      phi += density[p] * panel.area * std::exp(-kappa * r) / (4.0 * kPi * r);
    }
    energy += 0.5 * kCoulombKcalPerMol * atom.q * phi;
  }
  return energy;
}

double born_reference_kcal(const std::vector<Atom> &atoms, double eps_in,
                           double eps_out, double kappa) {
  double energy = 0.0;
  for (const Atom &atom : atoms) {
    const double radius = std::max(0.5, atom.radius);
    const double outside =
        1.0 / (std::max(1.0e-12, eps_out) * (1.0 + kappa * radius));
    const double inside = 1.0 / std::max(1.0e-12, eps_in);
    energy += 0.5 * kCoulombKcalPerMol * atom.q * atom.q * (outside - inside) /
              radius;
  }
  return energy;
}

std::vector<double> build_jcp_rhs(const std::vector<Atom> &atoms,
                                  const std::vector<Panel> &panels,
                                  double eps_in, bool coupled,
                                  TitanExecutor &executor, double *s1_norm,
                                  double *s2_norm) {
  const std::size_t n = panels.size();
  std::vector<double> rhs(coupled ? 2 * n : n, 0.0);
  const double inv_eps_in = 1.0 / std::max(1.0e-12, eps_in);
  AlignedDoubleVector atom_x(atoms.size());
  AlignedDoubleVector atom_y(atoms.size());
  AlignedDoubleVector atom_z(atoms.size());
  AlignedDoubleVector atom_q_over_eps(atoms.size());
  for (std::size_t atom_index = 0; atom_index < atoms.size(); ++atom_index) {
    atom_x[atom_index] = atoms[atom_index].x;
    atom_y[atom_index] = atoms[atom_index].y;
    atom_z[atom_index] = atoms[atom_index].z;
    atom_q_over_eps[atom_index] = atoms[atom_index].q * inv_eps_in;
  }

  auto scalar_panel = [&](const Panel &panel, double &s1, double &s2) noexcept {
    constexpr double inv_four_pi = 1.0 / (4.0 * kPi);
    s1 = 0.0;
    s2 = 0.0;
    for (std::size_t atom_index = 0; atom_index < atom_x.size(); ++atom_index) {
      const double dx = panel.x - atom_x[atom_index];
      const double dy = panel.y - atom_y[atom_index];
      const double dz = panel.z - atom_z[atom_index];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= 1.0e-28)
        continue;
      const double r = std::sqrt(r2);
      const double weighted_kernel =
          atom_q_over_eps[atom_index] * inv_four_pi;
      s1 += weighted_kernel / r;
      s2 -= weighted_kernel *
            (panel.nx * dx + panel.ny * dy + panel.nz * dz) / (r2 * r);
    }
  };

#if TITAN_SIMD_X86
  auto avx2_panel = [&] TITAN_LAMBDA_TARGET_AVX2(const Panel &panel, double &s1,
                                                 double &s2) noexcept {
    constexpr double inv_four_pi_scalar = 1.0 / (4.0 * kPi);
    const __m256d px = _mm256_set1_pd(panel.x);
    const __m256d py = _mm256_set1_pd(panel.y);
    const __m256d pz = _mm256_set1_pd(panel.z);
    const __m256d nx = _mm256_set1_pd(panel.nx);
    const __m256d ny = _mm256_set1_pd(panel.ny);
    const __m256d nz = _mm256_set1_pd(panel.nz);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d tiny = _mm256_set1_pd(1.0e-28);
    const __m256d inv_four_pi = _mm256_set1_pd(inv_four_pi_scalar);
    s1 = 0.0;
    s2 = 0.0;
    std::size_t atom_index = 0;
    alignas(32) double s1_lane[4];
    alignas(32) double s2_lane[4];
    for (; atom_index + 4 <= atom_x.size(); atom_index += 4) {
      const __m256d dx =
          _mm256_sub_pd(px, _mm256_load_pd(atom_x.data() + atom_index));
      const __m256d dy =
          _mm256_sub_pd(py, _mm256_load_pd(atom_y.data() + atom_index));
      const __m256d dz =
          _mm256_sub_pd(pz, _mm256_load_pd(atom_z.data() + atom_index));
      const __m256d r2 = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
          _mm256_mul_pd(dz, dz));
      const __m256d valid = _mm256_cmp_pd(r2, tiny, _CMP_GT_OQ);
      const __m256d safe_r2 = _mm256_blendv_pd(one, r2, valid);
      const __m256d r = _mm256_sqrt_pd(safe_r2);
      const __m256d weighted_kernel = _mm256_mul_pd(
          _mm256_load_pd(atom_q_over_eps.data() + atom_index), inv_four_pi);
      __m256d value_s1 = _mm256_div_pd(weighted_kernel, r);
      const __m256d normal_dot = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(nx, dx), _mm256_mul_pd(ny, dy)),
          _mm256_mul_pd(nz, dz));
      __m256d value_s2 = _mm256_div_pd(
          _mm256_mul_pd(_mm256_set1_pd(-1.0),
                        _mm256_mul_pd(weighted_kernel, normal_dot)),
          _mm256_mul_pd(safe_r2, r));
      value_s1 = _mm256_and_pd(value_s1, valid);
      value_s2 = _mm256_and_pd(value_s2, valid);
      _mm256_store_pd(s1_lane, value_s1);
      _mm256_store_pd(s2_lane, value_s2);
      for (int lane = 0; lane < 4; ++lane) {
        s1 += s1_lane[lane];
        s2 += s2_lane[lane];
      }
    }
    for (; atom_index < atom_x.size(); ++atom_index) {
      const double dx = panel.x - atom_x[atom_index];
      const double dy = panel.y - atom_y[atom_index];
      const double dz = panel.z - atom_z[atom_index];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 <= 1.0e-28)
        continue;
      const double r = std::sqrt(r2);
      const double weighted_kernel =
          atom_q_over_eps[atom_index] * inv_four_pi_scalar;
      s1 += weighted_kernel / r;
      s2 -= weighted_kernel *
            (panel.nx * dx + panel.ny * dy + panel.nz * dz) / (r2 * r);
    }
  };
#endif

#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  const bool use_avx2 = titan_simd_dispatch::runtime_has_avx2();
#else
  const bool use_avx2 = false;
#endif
  const std::size_t workers = std::max<std::size_t>(
      1, std::min<std::size_t>(executor.capacity(), panels.size()));
  constexpr std::size_t panel_tile = 8;
  std::atomic<std::size_t> next_panel{0};
  executor.parallel_do(workers, [&](std::size_t) {
    for (;;) {
      const std::size_t begin = next_panel.fetch_add(panel_tile);
      if (begin >= panels.size())
        break;
      const std::size_t end = std::min(panels.size(), begin + panel_tile);
      for (std::size_t i = begin; i < end; ++i) {
        double s1 = 0.0;
        double s2 = 0.0;
#if TITAN_SIMD_X86
        if (use_avx2)
          avx2_panel(panels[i], s1, s2);
        else
#endif
          scalar_panel(panels[i], s1, s2);
        rhs[i] = s1;
        if (coupled)
          rhs[n + i] = s2;
      }
    }
  });

  auto rhs_rms = [&](std::size_t begin, std::size_t end) {
    double sum_sq = 0.0;
    for (std::size_t i = begin; i < end; ++i)
      sum_sq += rhs[i] * rhs[i];
    return begin == end
               ? 0.0
               : std::sqrt(sum_sq / static_cast<double>(end - begin));
  };
  if (s1_norm)
    *s1_norm = rhs_rms(0, n);
  if (s2_norm)
    *s2_norm = coupled ? rhs_rms(n, 2 * n) : 0.0;
  return rhs;
}

void apply_jcp_linear_pb_operator(
    const std::vector<Panel> &panels, const std::vector<double> &x,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    const std::string &near_integration, const std::string &quad_buffer,
    TitanExecutor &executor, std::vector<double> &out, TreecodeStats *stats) {
  const std::size_t n = panels.size();
  out.assign(2 * n, 0.0);
  std::vector<std::size_t> order(n, 0);
  std::iota(order.begin(), order.end(), std::size_t{0});
  const std::size_t workers =
      std::max<std::size_t>(1, std::min<std::size_t>(executor.capacity(), n));
  std::vector<TreecodeStats> local_stats(workers);
  const auto near_begin = std::chrono::steady_clock::now();
  const NearSourceCache source_cache = build_near_source_cache(panels);
  executor.parallel_for(
      n, workers, 4,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          TreecodeStats &worker_stats = local_stats[worker];
          const JcpNearAccum sum = accumulate_jcp_near_ordered(
              panels[i], i, panels, x, eps_ratio, kappa, true, order, 0, n,
              near_integration, quad_buffer, &worker_stats, &source_cache);
          double phi_row =
              jcp_lhs_phi_for_panel(panels[i], eps_ratio, lhs_phi) * x[i];
          double phin_row =
              jcp_lhs_phin_for_panel(panels[i], eps_ratio, lhs_phin) *
              x[n + i];
          phi_row -= sum.phi;
          phin_row -= sum.phin;
          out[i] = phi_row;
          out[n + i] = phin_row;
        }
      });
  if (stats != nullptr) {
    stats->near_integration = near_integration;
    stats->quad_buffer = quad_buffer;
    const double near_ms =
        elapsed_ms(near_begin, std::chrono::steady_clock::now());
    stats->near_integration_ms += near_ms;
    stats->near_eval_ms += near_ms;
    std::size_t local_centroid = 0;
    std::size_t local_duffy = 0;
    std::size_t local_geometric = 0;
    for (const TreecodeStats &worker_stats : local_stats) {
      const std::size_t previous_duffy = stats->near_duffy_pairs;
      stats->near_kernel_policy = worker_stats.near_kernel_policy;
      stats->near_self_pairs += worker_stats.near_self_pairs;
      stats->near_one_ring_pairs += worker_stats.near_one_ring_pairs;
      stats->near_duffy_pairs += worker_stats.near_duffy_pairs;
      stats->near_centroid_pairs += worker_stats.near_centroid_pairs;
      stats->near_fallbacks += worker_stats.near_fallbacks;
      merge_near_kernel_block_stats(*stats, worker_stats);
      local_centroid += worker_stats.near_centroid_pairs;
      local_duffy += worker_stats.near_duffy_pairs;
      local_geometric += worker_stats.near_geometric_buffer_pairs;
      stats->duffy_subdivisions_max = std::max(
          stats->duffy_subdivisions_max, worker_stats.duffy_subdivisions_max);
      if (worker_stats.near_duffy_pairs > 0) {
        const double old_count = static_cast<double>(previous_duffy);
        const double new_count = static_cast<double>(stats->near_duffy_pairs);
        stats->duffy_subdivisions_mean =
            new_count <= 0.0
                ? 0.0
                : (stats->duffy_subdivisions_mean * old_count +
                   worker_stats.duffy_subdivisions_mean *
                       static_cast<double>(worker_stats.near_duffy_pairs)) /
                      new_count;
      }
    }
    stats->near_interactions += local_centroid + local_duffy + local_geometric;
  }
}

void apply_jcp_poisson_operator(
    const std::vector<Panel> &panels, const std::vector<double> &x,
    double eps_ratio, double lhs_phi, const std::string &near_integration,
    const std::string &quad_buffer, TitanExecutor &executor,
    std::vector<double> &out, TreecodeStats *stats) {
  const std::size_t n = panels.size();
  out.assign(n, 0.0);
  std::vector<std::size_t> order(n, 0);
  std::iota(order.begin(), order.end(), std::size_t{0});
  const std::size_t workers =
      std::max<std::size_t>(1, std::min<std::size_t>(executor.capacity(), n));
  std::vector<TreecodeStats> local_stats(workers);
  const auto near_begin = std::chrono::steady_clock::now();
  const NearSourceCache source_cache = build_near_source_cache(panels);
  executor.parallel_for(
      n, workers, 4,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          TreecodeStats &worker_stats = local_stats[worker];
          const JcpNearAccum sum = accumulate_jcp_near_ordered(
              panels[i], i, panels, x, eps_ratio, 0.0, false, order, 0, n,
              near_integration, quad_buffer, &worker_stats, &source_cache);
          double row =
              jcp_lhs_phi_for_panel(panels[i], eps_ratio, lhs_phi) * x[i];
          row -= sum.phi;
          out[i] = row;
        }
      });
  if (stats != nullptr) {
    stats->near_integration = near_integration;
    stats->quad_buffer = quad_buffer;
    const double near_ms =
        elapsed_ms(near_begin, std::chrono::steady_clock::now());
    stats->near_integration_ms += near_ms;
    stats->near_eval_ms += near_ms;
    std::size_t local_centroid = 0;
    std::size_t local_duffy = 0;
    std::size_t local_geometric = 0;
    for (const TreecodeStats &worker_stats : local_stats) {
      const std::size_t previous_duffy = stats->near_duffy_pairs;
      stats->near_kernel_policy = worker_stats.near_kernel_policy;
      stats->near_self_pairs += worker_stats.near_self_pairs;
      stats->near_one_ring_pairs += worker_stats.near_one_ring_pairs;
      stats->near_duffy_pairs += worker_stats.near_duffy_pairs;
      stats->near_centroid_pairs += worker_stats.near_centroid_pairs;
      stats->near_fallbacks += worker_stats.near_fallbacks;
      merge_near_kernel_block_stats(*stats, worker_stats);
      local_centroid += worker_stats.near_centroid_pairs;
      local_duffy += worker_stats.near_duffy_pairs;
      local_geometric += worker_stats.near_geometric_buffer_pairs;
      stats->duffy_subdivisions_max = std::max(
          stats->duffy_subdivisions_max, worker_stats.duffy_subdivisions_max);
      if (worker_stats.near_duffy_pairs > 0) {
        const double old_count = static_cast<double>(previous_duffy);
        const double new_count = static_cast<double>(stats->near_duffy_pairs);
        stats->duffy_subdivisions_mean =
            new_count <= 0.0
                ? 0.0
                : (stats->duffy_subdivisions_mean * old_count +
                   worker_stats.duffy_subdivisions_mean *
                       static_cast<double>(worker_stats.near_duffy_pairs)) /
                      new_count;
      }
    }
    stats->near_interactions += local_centroid + local_duffy + local_geometric;
  }
}

struct EnergyProjectionSources {
  AlignedDoubleVector x;
  AlignedDoubleVector y;
  AlignedDoubleVector z;
  AlignedDoubleVector nx;
  AlignedDoubleVector ny;
  AlignedDoubleVector nz;
  AlignedDoubleVector area;
  AlignedDoubleVector phi;
  AlignedDoubleVector phin;
};

EnergyProjectionSources
build_energy_projection_sources(const std::vector<Panel> &panels,
                                const std::vector<double> &unknowns,
                                bool coupled) {
  const std::size_t n = panels.size();
  if (n && curved_dual_cell(panels.front())) {
    EnergyProjectionSources sources;
    std::size_t count=0;
    for(const auto&p:panels) count+=curved_dual_cell(p)->samples.size();
    sources.x.reserve(count);sources.y.reserve(count);sources.z.reserve(count);
    sources.nx.reserve(count);sources.ny.reserve(count);sources.nz.reserve(count);
    sources.area.reserve(count);sources.phi.reserve(count);
    if(coupled)sources.phin.reserve(count);
    for(std::size_t j=0;j<n;++j)for(const auto&q:curved_dual_cell(panels[j])->samples){
      sources.x.push_back(q.xyz[0]);sources.y.push_back(q.xyz[1]);sources.z.push_back(q.xyz[2]);
      sources.nx.push_back(q.normal[0]);sources.ny.push_back(q.normal[1]);sources.nz.push_back(q.normal[2]);
      sources.area.push_back(q.weight);sources.phi.push_back(unknowns[j]);
      if(coupled)sources.phin.push_back(unknowns[n+j]);
    }
    return sources;
  }
  EnergyProjectionSources sources;
  sources.x.resize(n);
  sources.y.resize(n);
  sources.z.resize(n);
  sources.nx.resize(n);
  sources.ny.resize(n);
  sources.nz.resize(n);
  sources.area.resize(n);
  sources.phi.resize(n);
  if (coupled)
    sources.phin.resize(n);
  for (std::size_t j = 0; j < n; ++j) {
    const Panel &panel = panels[j];
    sources.x[j] = panel.has_source_quadrature_center ? panel.source_x : panel.x;
    sources.y[j] = panel.has_source_quadrature_center ? panel.source_y : panel.y;
    sources.z[j] = panel.has_source_quadrature_center ? panel.source_z : panel.z;
    sources.nx[j] = panel_source_nx(panel);
    sources.ny[j] = panel_source_ny(panel);
    sources.nz[j] = panel_source_nz(panel);
    sources.area[j] = panel.area;
    sources.phi[j] = unknowns[j];
    if (coupled)
      sources.phin[j] = unknowns[n + j];
  }
  return sources;
}

double linear_pb_reaction_phi_scalar(const Atom &atom,
                                     const EnergyProjectionSources &sources,
                                     double eps_ratio, double kappa) noexcept {
  constexpr double inv_four_pi = 1.0 / (4.0 * kPi);
  double reaction_phi = 0.0;
  for (std::size_t j = 0; j < sources.x.size(); ++j) {
    const double dx = atom.x - sources.x[j];
    const double dy = atom.y - sources.y[j];
    const double dz = atom.z - sources.z[j];
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 <= 1.0e-28)
      continue;
    const double r = std::sqrt(r2);
    const double screened = std::exp(-kappa * r);
    const double g0 = inv_four_pi / r;
    const double normal_dot =
        sources.nx[j] * dx + sources.ny[j] * dy + sources.nz[j] * dz;
    const double dg0 = inv_four_pi * normal_dot / (r2 * r);
    const double k1 = g0 * (1.0 - screened);
    const double k2 =
        dg0 * (eps_ratio * screened * (1.0 + kappa * r) - 1.0);
    reaction_phi += sources.area[j] *
                    (k1 * sources.phin[j] + k2 * sources.phi[j]);
  }
  return reaction_phi;
}

double poisson_reaction_phi_scalar(const Atom &atom,
                                   const EnergyProjectionSources &sources,
                                   double eps_ratio) noexcept {
  constexpr double inv_four_pi = 1.0 / (4.0 * kPi);
  const double k2_scale = eps_ratio - 1.0;
  double reaction_phi = 0.0;
  for (std::size_t j = 0; j < sources.x.size(); ++j) {
    const double dx = atom.x - sources.x[j];
    const double dy = atom.y - sources.y[j];
    const double dz = atom.z - sources.z[j];
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 <= 1.0e-28)
      continue;
    const double r = std::sqrt(r2);
    const double normal_dot =
        sources.nx[j] * dx + sources.ny[j] * dy + sources.nz[j] * dz;
    const double k2 = k2_scale * inv_four_pi * normal_dot / (r2 * r);
    reaction_phi += sources.area[j] * k2 * sources.phi[j];
  }
  return reaction_phi;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
double linear_pb_reaction_phi_avx2(const Atom &atom,
                                   const EnergyProjectionSources &sources,
                                   double eps_ratio, double kappa) noexcept {
  constexpr double inv_four_pi_scalar = 1.0 / (4.0 * kPi);
  const __m256d tx = _mm256_set1_pd(atom.x);
  const __m256d ty = _mm256_set1_pd(atom.y);
  const __m256d tz = _mm256_set1_pd(atom.z);
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d tiny = _mm256_set1_pd(1.0e-28);
  const __m256d inv_four_pi = _mm256_set1_pd(inv_four_pi_scalar);
  const __m256d eps = _mm256_set1_pd(eps_ratio);
  const __m256d kap = _mm256_set1_pd(kappa);
  double reaction_phi = 0.0;
  std::size_t j = 0;
  alignas(32) double exponent[4];
  alignas(32) double screened_lane[4];
  alignas(32) double contribution[4];
  for (; j + 4 <= sources.x.size(); j += 4) {
    const __m256d dx = _mm256_sub_pd(tx, _mm256_load_pd(sources.x.data() + j));
    const __m256d dy = _mm256_sub_pd(ty, _mm256_load_pd(sources.y.data() + j));
    const __m256d dz = _mm256_sub_pd(tz, _mm256_load_pd(sources.z.data() + j));
    const __m256d r2 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
        _mm256_mul_pd(dz, dz));
    const __m256d valid = _mm256_cmp_pd(r2, tiny, _CMP_GT_OQ);
    const __m256d safe_r2 = _mm256_blendv_pd(one, r2, valid);
    const __m256d r = _mm256_sqrt_pd(safe_r2);
    _mm256_store_pd(exponent, _mm256_mul_pd(_mm256_set1_pd(-kappa), r));
    for (int lane = 0; lane < 4; ++lane)
      screened_lane[lane] = std::exp(exponent[lane]);
    const __m256d screened = _mm256_load_pd(screened_lane);
    const __m256d g0 = _mm256_div_pd(inv_four_pi, r);
    const __m256d normal_dot = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(_mm256_load_pd(sources.nx.data() + j), dx),
                      _mm256_mul_pd(_mm256_load_pd(sources.ny.data() + j), dy)),
        _mm256_mul_pd(_mm256_load_pd(sources.nz.data() + j), dz));
    const __m256d dg0 = _mm256_div_pd(
        _mm256_mul_pd(inv_four_pi, normal_dot), _mm256_mul_pd(safe_r2, r));
    const __m256d k1 = _mm256_mul_pd(g0, _mm256_sub_pd(one, screened));
    const __m256d k2_factor = _mm256_sub_pd(
        _mm256_mul_pd(_mm256_mul_pd(eps, screened),
                      _mm256_add_pd(one, _mm256_mul_pd(kap, r))),
        one);
    const __m256d k2 = _mm256_mul_pd(dg0, k2_factor);
    __m256d value = _mm256_mul_pd(
        _mm256_load_pd(sources.area.data() + j),
        _mm256_add_pd(
            _mm256_mul_pd(k1, _mm256_load_pd(sources.phin.data() + j)),
            _mm256_mul_pd(k2, _mm256_load_pd(sources.phi.data() + j))));
    value = _mm256_and_pd(value, valid);
    _mm256_store_pd(contribution, value);
    reaction_phi += contribution[0];
    reaction_phi += contribution[1];
    reaction_phi += contribution[2];
    reaction_phi += contribution[3];
  }
  for (; j < sources.x.size(); ++j) {
    const double dx = atom.x - sources.x[j];
    const double dy = atom.y - sources.y[j];
    const double dz = atom.z - sources.z[j];
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 <= 1.0e-28)
      continue;
    const double r = std::sqrt(r2);
    const double screened = std::exp(-kappa * r);
    const double g0 = inv_four_pi_scalar / r;
    const double normal_dot =
        sources.nx[j] * dx + sources.ny[j] * dy + sources.nz[j] * dz;
    const double dg0 = inv_four_pi_scalar * normal_dot / (r2 * r);
    reaction_phi +=
        sources.area[j] *
        (g0 * (1.0 - screened) * sources.phin[j] +
         dg0 * (eps_ratio * screened * (1.0 + kappa * r) - 1.0) *
             sources.phi[j]);
  }
  return reaction_phi;
}

TITAN_TARGET_AVX2
double poisson_reaction_phi_avx2(const Atom &atom,
                                 const EnergyProjectionSources &sources,
                                 double eps_ratio) noexcept {
  constexpr double inv_four_pi_scalar = 1.0 / (4.0 * kPi);
  const __m256d tx = _mm256_set1_pd(atom.x);
  const __m256d ty = _mm256_set1_pd(atom.y);
  const __m256d tz = _mm256_set1_pd(atom.z);
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d tiny = _mm256_set1_pd(1.0e-28);
  const __m256d scale = _mm256_set1_pd((eps_ratio - 1.0) * inv_four_pi_scalar);
  double reaction_phi = 0.0;
  std::size_t j = 0;
  alignas(32) double contribution[4];
  for (; j + 4 <= sources.x.size(); j += 4) {
    const __m256d dx = _mm256_sub_pd(tx, _mm256_load_pd(sources.x.data() + j));
    const __m256d dy = _mm256_sub_pd(ty, _mm256_load_pd(sources.y.data() + j));
    const __m256d dz = _mm256_sub_pd(tz, _mm256_load_pd(sources.z.data() + j));
    const __m256d r2 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
        _mm256_mul_pd(dz, dz));
    const __m256d valid = _mm256_cmp_pd(r2, tiny, _CMP_GT_OQ);
    const __m256d safe_r2 = _mm256_blendv_pd(one, r2, valid);
    const __m256d r = _mm256_sqrt_pd(safe_r2);
    const __m256d normal_dot = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(_mm256_load_pd(sources.nx.data() + j), dx),
                      _mm256_mul_pd(_mm256_load_pd(sources.ny.data() + j), dy)),
        _mm256_mul_pd(_mm256_load_pd(sources.nz.data() + j), dz));
    __m256d value = _mm256_mul_pd(
        _mm256_load_pd(sources.area.data() + j),
        _mm256_mul_pd(_mm256_load_pd(sources.phi.data() + j),
                      _mm256_div_pd(_mm256_mul_pd(scale, normal_dot),
                                    _mm256_mul_pd(safe_r2, r))));
    value = _mm256_and_pd(value, valid);
    _mm256_store_pd(contribution, value);
    reaction_phi += contribution[0];
    reaction_phi += contribution[1];
    reaction_phi += contribution[2];
    reaction_phi += contribution[3];
  }
  for (; j < sources.x.size(); ++j) {
    const double dx = atom.x - sources.x[j];
    const double dy = atom.y - sources.y[j];
    const double dz = atom.z - sources.z[j];
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 <= 1.0e-28)
      continue;
    const double r = std::sqrt(r2);
    const double normal_dot =
        sources.nx[j] * dx + sources.ny[j] * dy + sources.nz[j] * dz;
    reaction_phi += sources.area[j] * sources.phi[j] *
                    ((eps_ratio - 1.0) * inv_four_pi_scalar * normal_dot /
                     (r2 * r));
  }
  return reaction_phi;
}
#endif

template <typename ScalarKernel, typename Avx2Kernel>
double project_reaction_energy(const std::vector<Atom> &atoms,
                               TitanExecutor &executor,
                               ScalarKernel scalar_kernel,
                               Avx2Kernel avx2_kernel,
                               std::vector<double> *reaction_phi_by_atom,
                               std::string *kernel_policy) {
  std::vector<double> reaction_phi(atoms.size(), 0.0);
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  const bool use_avx2 = titan_simd_dispatch::runtime_has_avx2();
#else
  const bool use_avx2 = false;
#endif
  const std::size_t workers = std::max<std::size_t>(
      1, std::min<std::size_t>(executor.capacity(), atoms.size()));
  constexpr std::size_t atom_tile = 4;
  std::atomic<std::size_t> next_atom{0};
  executor.parallel_do(workers, [&](std::size_t) {
    for (;;) {
      const std::size_t begin = next_atom.fetch_add(atom_tile);
      if (begin >= atoms.size())
        break;
      const std::size_t end = std::min(atoms.size(), begin + atom_tile);
      for (std::size_t atom_index = begin; atom_index < end; ++atom_index) {
        reaction_phi[atom_index] =
            use_avx2 ? avx2_kernel(atoms[atom_index])
                     : scalar_kernel(atoms[atom_index]);
      }
    }
  });
  double energy = 0.0;
  for (std::size_t atom_index = 0; atom_index < atoms.size(); ++atom_index)
    energy += 0.5 * kCoulombKcalPerMol * atoms[atom_index].q *
              reaction_phi[atom_index];
  if (reaction_phi_by_atom != nullptr)
    *reaction_phi_by_atom = std::move(reaction_phi);
  if (kernel_policy != nullptr) {
    *kernel_policy = use_avx2 ? "dynamic-atom-tile4-avx2-exact-exp"
                              : "dynamic-atom-tile4-scalar";
  }
  return energy;
}

// Curved energy projection with far cells aggregated.  Every curved dual cell
// is first summed as one point - its area centroid, total area and
// area-weighted normal - for all atoms.  Cells whose centroid lies within
// far_eta cell radii of an atom then have that point replaced by all of their
// samples.  The single-point error decays as (radius/distance)^2, so far_eta
// sets the accuracy; as far_eta grows the result tends to the exact sample
// sum.  No per-sample copy is made, which is most of the projection memory.
struct CurvedProjectionSplit {
  EnergyProjectionSources aggregate;
  std::vector<double> radius;
  std::vector<std::uint64_t> keys;
  std::vector<std::uint32_t> order;
  double cell_size{0.0};
  double ox{0.0}, oy{0.0}, oz{0.0};
};

[[nodiscard]] std::uint64_t projection_grid_key(std::int64_t ix,
                                                std::int64_t iy,
                                                std::int64_t iz) noexcept {
  constexpr std::int64_t bias = std::int64_t{1} << 20;
  constexpr std::uint64_t mask = (std::uint64_t{1} << 21) - 1U;
  return ((static_cast<std::uint64_t>(ix + bias) & mask) << 42) |
         ((static_cast<std::uint64_t>(iy + bias) & mask) << 21) |
         (static_cast<std::uint64_t>(iz + bias) & mask);
}

CurvedProjectionSplit build_curved_projection_split(
    const std::vector<Panel> &panels, const std::vector<double> &unknowns,
    bool coupled, double far_eta) {
  const std::size_t n = panels.size();
  CurvedProjectionSplit split;
  auto &a = split.aggregate;
  a.x.resize(n);
  a.y.resize(n);
  a.z.resize(n);
  a.nx.resize(n);
  a.ny.resize(n);
  a.nz.resize(n);
  a.area.resize(n);
  a.phi.resize(n);
  if (coupled)
    a.phin.resize(n);
  split.radius.assign(n, 0.0);
  double radius_max = 0.0;
  split.ox = split.oy = split.oz = std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < n; ++j) {
    const auto &cell = *curved_dual_cell(panels[j]);
    const double area = cell.area;
    a.x[j] = cell.center[0];
    a.y[j] = cell.center[1];
    a.z[j] = cell.center[2];
    const double inverse = area > 0.0 ? 1.0 / area : 0.0;
    a.nx[j] = cell.normal_integral[0] * inverse;
    a.ny[j] = cell.normal_integral[1] * inverse;
    a.nz[j] = cell.normal_integral[2] * inverse;
    a.area[j] = area;
    a.phi[j] = unknowns[j];
    if (coupled)
      a.phin[j] = unknowns[n + j];
    double radius = 0.0;
    for (const auto &q : cell.samples) {
      const double dx = q.xyz[0] - cell.center[0];
      const double dy = q.xyz[1] - cell.center[1];
      const double dz = q.xyz[2] - cell.center[2];
      radius = std::max(radius, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    split.radius[j] = radius;
    radius_max = std::max(radius_max, radius);
    split.ox = std::min(split.ox, a.x[j]);
    split.oy = std::min(split.oy, a.y[j]);
    split.oz = std::min(split.oz, a.z[j]);
  }
  split.cell_size = std::max(1.0e-6, far_eta * radius_max);
  split.keys.resize(n);
  split.order.resize(n);
  for (std::size_t j = 0; j < n; ++j) {
    split.keys[j] = projection_grid_key(
        static_cast<std::int64_t>(std::floor((a.x[j] - split.ox) / split.cell_size)),
        static_cast<std::int64_t>(std::floor((a.y[j] - split.oy) / split.cell_size)),
        static_cast<std::int64_t>(std::floor((a.z[j] - split.oz) / split.cell_size)));
    split.order[j] = static_cast<std::uint32_t>(j);
  }
  std::sort(split.order.begin(), split.order.end(),
            [&](std::uint32_t l, std::uint32_t r) {
              return split.keys[l] != split.keys[r] ? split.keys[l] < split.keys[r]
                                                    : l < r;
            });
  std::vector<std::uint64_t> sorted(n);
  for (std::size_t k = 0; k < n; ++k)
    sorted[k] = split.keys[split.order[k]];
  split.keys = std::move(sorted);
  return split;
}

// Sum of `kernel(source point)` over the samples of every cell near `atom`,
// minus the same kernel at that cell's aggregate point.
template <typename PointKernel>
double curved_projection_near_correction(const Atom &atom,
                                         const std::vector<Panel> &panels,
                                         const CurvedProjectionSplit &split,
                                         const std::vector<double> &unknowns,
                                         bool coupled, double far_eta,
                                         PointKernel kernel) {
  const auto &a = split.aggregate;
  const std::size_t n = panels.size();
  const auto cx = static_cast<std::int64_t>(
      std::floor((atom.x - split.ox) / split.cell_size));
  const auto cy = static_cast<std::int64_t>(
      std::floor((atom.y - split.oy) / split.cell_size));
  const auto cz = static_cast<std::int64_t>(
      std::floor((atom.z - split.oz) / split.cell_size));
  double correction = 0.0;
  for (std::int64_t dz = -1; dz <= 1; ++dz)
    for (std::int64_t dy = -1; dy <= 1; ++dy)
      for (std::int64_t dx = -1; dx <= 1; ++dx) {
        const std::uint64_t key = projection_grid_key(cx + dx, cy + dy, cz + dz);
        auto first = std::lower_bound(split.keys.begin(), split.keys.end(), key);
        for (auto it = first; it != split.keys.end() && *it == key; ++it) {
          const std::size_t j = split.order[static_cast<std::size_t>(
              it - split.keys.begin())];
          const double rx = atom.x - a.x[j], ry = atom.y - a.y[j],
                       rz = atom.z - a.z[j];
          const double limit = far_eta * split.radius[j];
          if (rx * rx + ry * ry + rz * rz >= limit * limit)
            continue;
          const double phi = unknowns[j];
          const double phin = coupled ? unknowns[n + j] : 0.0;
          for (const auto &q : curved_dual_cell(panels[j])->samples)
            correction += kernel(q.xyz[0], q.xyz[1], q.xyz[2], q.normal[0],
                                 q.normal[1], q.normal[2], q.weight, phi, phin);
          correction -= kernel(a.x[j], a.y[j], a.z[j], a.nx[j], a.ny[j],
                               a.nz[j], a.area[j], phi, phin);
        }
      }
  return correction;
}

double jcp_linear_pb_energy_kcal(const std::vector<Atom> &atoms,
                                 const std::vector<Panel> &panels,
                                 const std::vector<double> &x, double eps_ratio,
                                 double kappa, TitanExecutor &executor,
                                 std::vector<double> *reaction_phi_by_atom,
                                 std::string *kernel_policy,
                                 double far_eta = 0.0) {
  if (far_eta > 0.0 && !panels.empty() && curved_dual_cell(panels.front())) {
    const CurvedProjectionSplit split =
        build_curved_projection_split(panels, x, true, far_eta);
    constexpr double inv_four_pi = 1.0 / (4.0 * kPi);
    const auto point = [&](const Atom &atom) {
      return [&atom, eps_ratio, kappa](double sx, double sy, double sz,
                                       double nx, double ny, double nz,
                                       double area, double phi, double phin) {
        const double dx = atom.x - sx, dy = atom.y - sy, dz = atom.z - sz;
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= 1.0e-28)
          return 0.0;
        const double r = std::sqrt(r2);
        const double screened = std::exp(-kappa * r);
        const double g0 = inv_four_pi / r;
        const double normal_dot = nx * dx + ny * dy + nz * dz;
        const double dg0 = inv_four_pi * normal_dot / (r2 * r);
        const double k1 = g0 * (1.0 - screened);
        const double k2 = dg0 * (eps_ratio * screened * (1.0 + kappa * r) - 1.0);
        return area * (k1 * phin + k2 * phi);
      };
    };
    const auto scalar = [&](const Atom &atom) {
      return linear_pb_reaction_phi_scalar(atom, split.aggregate, eps_ratio,
                                           kappa) +
             curved_projection_near_correction(atom, panels, split, x, true,
                                               far_eta, point(atom));
    };
    const auto avx2 = [&](const Atom &atom) {
#if TITAN_SIMD_X86
      return linear_pb_reaction_phi_avx2(atom, split.aggregate, eps_ratio,
                                         kappa) +
             curved_projection_near_correction(atom, panels, split, x, true,
                                               far_eta, point(atom));
#else
      return scalar(atom);
#endif
    };
    const double energy = project_reaction_energy(
        atoms, executor, scalar, avx2, reaction_phi_by_atom, kernel_policy);
    if (kernel_policy != nullptr)
      *kernel_policy += "-cell-aggregate-eta" + std::to_string(far_eta);
    return energy;
  }
  const EnergyProjectionSources sources =
      build_energy_projection_sources(panels, x, true);
  const auto scalar = [&](const Atom &atom) {
    return linear_pb_reaction_phi_scalar(atom, sources, eps_ratio, kappa);
  };
  const auto avx2 = [&](const Atom &atom) {
#if TITAN_SIMD_X86
    return linear_pb_reaction_phi_avx2(atom, sources, eps_ratio, kappa);
#else
    return scalar(atom);
#endif
  };
  return project_reaction_energy(atoms, executor, scalar, avx2,
                                 reaction_phi_by_atom, kernel_policy);
}

double jcp_poisson_energy_kcal(const std::vector<Atom> &atoms,
                               const std::vector<Panel> &panels,
                               const std::vector<double> &phi, double eps_ratio,
                               TitanExecutor &executor,
                               std::vector<double> *reaction_phi_by_atom,
                               std::string *kernel_policy,
                               double far_eta = 0.0) {
  if (far_eta > 0.0 && !panels.empty() && curved_dual_cell(panels.front())) {
    const CurvedProjectionSplit split =
        build_curved_projection_split(panels, phi, false, far_eta);
    constexpr double inv_four_pi = 1.0 / (4.0 * kPi);
    const double k2_scale = eps_ratio - 1.0;
    const auto point = [&](const Atom &atom) {
      return [&atom, k2_scale](double sx, double sy, double sz, double nx,
                               double ny, double nz, double area, double value,
                               double) {
        const double dx = atom.x - sx, dy = atom.y - sy, dz = atom.z - sz;
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= 1.0e-28)
          return 0.0;
        const double r = std::sqrt(r2);
        const double normal_dot = nx * dx + ny * dy + nz * dz;
        return area * k2_scale * inv_four_pi * normal_dot / (r2 * r) * value;
      };
    };
    const auto scalar = [&](const Atom &atom) {
      return poisson_reaction_phi_scalar(atom, split.aggregate, eps_ratio) +
             curved_projection_near_correction(atom, panels, split, phi, false,
                                               far_eta, point(atom));
    };
    const auto avx2 = [&](const Atom &atom) {
#if TITAN_SIMD_X86
      return poisson_reaction_phi_avx2(atom, split.aggregate, eps_ratio) +
             curved_projection_near_correction(atom, panels, split, phi, false,
                                               far_eta, point(atom));
#else
      return scalar(atom);
#endif
    };
    const double energy = project_reaction_energy(
        atoms, executor, scalar, avx2, reaction_phi_by_atom, kernel_policy);
    if (kernel_policy != nullptr)
      *kernel_policy += "-cell-aggregate-eta" + std::to_string(far_eta);
    return energy;
  }
  const EnergyProjectionSources sources =
      build_energy_projection_sources(panels, phi, false);
  const auto scalar = [&](const Atom &atom) {
    return poisson_reaction_phi_scalar(atom, sources, eps_ratio);
  };
  const auto avx2 = [&](const Atom &atom) {
#if TITAN_SIMD_X86
    return poisson_reaction_phi_avx2(atom, sources, eps_ratio);
#else
    return scalar(atom);
#endif
  };
  return project_reaction_energy(atoms, executor, scalar, avx2,
                                 reaction_phi_by_atom, kernel_policy);
}

// The certificate of a curved node-patch surface as "key=value;...": for the
// PN lift the vertices that took the star normal (and of them those turned
// further toward their faces), the repair rounds, the triangles still flat,
// the straightened neighbour edges, the corner vertices that collocate off
// the vertex and their largest offset, touching flat non-adjacent patches and
// the largest shared-edge gap; for the near pairs by admissible subdivision
// their number, the unresolved ones and the smallest target-to-cell gap
// ("n/a" when there were none); and the highest Gauss order a self block
// needed (an unconverged one refuses the lift).  The near counts grow while
// pairs are integrated, so it is refreshed when the solve ends.
std::string node_patch_certificate(const std::vector<Panel> &panels) {
  if (panels.empty() || !panels.front().node_patch_geometry ||
      !panels.front().node_patch_geometry->curved)
    return "n/a";
  const CurvedNodePatchGeometry &c = *panels.front().node_patch_geometry->curved;
  std::ostringstream out;
  out << std::setprecision(6);
  if (c.representation == "curved_pn")
    out << "pn_star_normals=" << c.pn_star_normal_vertices
        << ";pn_turned_normals=" << c.pn_turned_normal_vertices
        << ";pn_repair_rounds=" << c.pn_repair_rounds
        << ";pn_residual_flat=" << c.pn_residual_flat_triangles
        << ";pn_straightened_edges=" << c.pn_straightened_edges
        << ";pn_corner_vertices=" << c.pn_corner_vertices
        << ";pn_max_offset_A=" << c.pn_max_collocation_offset
        << ";pn_flat_contacts=" << c.pn_flat_contacts
        << ";pn_edge_gap_A=" << c.pn_max_shared_edge_gap << ';';
  out << "near_adaptive=" << c.near_adaptive_evaluations.load()
      << ";near_unresolved=" << c.near_unresolved_evaluations.load()
      << ";near_min_gap_A=";
  const double gap = c.near_min_gap.load();
  if (std::isfinite(gap))
    out << gap;
  else
    out << "n/a";
  out << ";self_max_order=" << c.self_max_order
      << ";self_base_order=" << c.self_base_order
      << ";self_fallback=" << c.self_fallback_blocks
      << ";self_fallback_step=" << c.self_max_fallback_step;
  return out.str();
}

// The shape of the solved normal derivative q on the node patches, appended
// to the certificate: q_rms, the largest |q| over the median |q|, and the
// adjacent node pairs of opposite sign that both exceed 50 median |q|.  A
// spurious oscillating mode - 451c at h = 1.25 A: |q| up to 573 against a
// median of 0.008, alternating between neighbouring nodes of 40 small cells
// refined in a deep pocket - shows as a ratio of order 1e4 and many such
// pairs, where a smooth q gives a ratio of tens to hundreds and none.  It
// does not stop the run; the audit of the solution (-pbe-bie-matvec-audit=
// targets) is the gate that the mode trips.
std::string node_patch_solution_shape(const std::vector<Panel> &panels,
                                      const std::vector<double> &x,
                                      bool coupled) {
  const std::size_t n = panels.size();
  if (!coupled || n == 0 || x.size() < 2 * n || !panels.front().node_patch_geometry)
    return "";
  const NodePatchGeometry &g = *panels.front().node_patch_geometry;
  std::vector<double> magnitude(n);
  double sum_sq = 0.0, largest = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double q = x[n + i];
    magnitude[i] = std::abs(q);
    sum_sq += q * q;
    largest = std::max(largest, magnitude[i]);
  }
  std::vector<double> sorted = magnitude;
  std::nth_element(sorted.begin(), sorted.begin() + n / 2, sorted.end());
  const double median = sorted[n / 2];
  std::size_t pairs = 0;
  if (g.adjacent_offsets.size() == n + 1 && median > 0.0)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t p = g.adjacent_offsets[i]; p < g.adjacent_offsets[i + 1];
           ++p) {
        const std::size_t j = g.adjacent_nodes[p];
        if (j > i && j < n && x[n + i] * x[n + j] < 0.0 &&
            magnitude[i] > 50 * median && magnitude[j] > 50 * median)
          ++pairs;
      }
  std::ostringstream out;
  out << std::setprecision(6) << ";q_rms=" << std::sqrt(sum_sq / n)
      << ";q_max_median=";
  if (median > 0.0)
    out << largest / median;
  else
    out << "n/a";
  out << ";q_checkerboard_pairs=" << pairs;
  return out.str();
}

} // namespace

static Result solve_linear_pb_single_attempt(const std::vector<Atom> &atoms,
                                             const Options &options,
                                             TitanExecutor &executor) {
  Result result;
  const bool cartesian_full_requested =
      options.fmm_engine == "cartesian-full-experimental";
  const bool cartesian_full_order4 =
      cartesian_full_requested && options.bie_fmm_cartesian_order == 4;
  const int effective_tree_leaf_size =
      cartesian_full_requested && !options.tree_leaf_size_explicit
          ? (cartesian_full_order4 ? 80 : 64)
          : options.tree_leaf_size;
  const double effective_tree_opening_theta =
      cartesian_full_requested && !options.tree_opening_theta_explicit
          ? (cartesian_full_order4 ? 0.65 : 0.20)
          : options.tree_opening_theta;
  result.tree_preset = (options.tree_preset == "stage3u-safe" ||
                        options.tree_preset == "stage3u-perf" ||
                        options.tree_preset == "stage3u-perf2")
                           ? options.tree_preset
                           : std::string("off");
  const FormulationConfig formulation = make_formulation_config(options);
  result.formulation = formulation.formulation;
  result.energy_projection = formulation.energy_projection;
  result.preconditioner_requested = options.preconditioner.empty()
                                        ? std::string("auto")
                                        : options.preconditioner;
  result.preconditioner_block_near =
      options.preconditioner_block_near == "buffer-aware" ||
              options.preconditioner_block_near == "operator-near"
          ? options.preconditioner_block_near
          : std::string("singular");
  const int preconditioner_block_cap_max =
      result.preconditioner_requested == "fmm-block-jacobi" ? 128 : 32;
  result.preconditioner_block_panel_cap = std::clamp(
      options.preconditioner_block_panel_cap, 8, preconditioner_block_cap_max);
  result.preconditioner_overlap_panel_cap =
      std::clamp(options.preconditioner_overlap_panel_cap, 0, 128);
  result.preconditioner_ilu_neighbor_cap_requested =
      std::clamp(options.preconditioner_ilu_neighbor_cap, 32, 1024);
  result.preconditioner_ilu_neighbor_cap =
      result.preconditioner_ilu_neighbor_cap_requested;
  result.preconditioner_ilu_drop_tol =
      std::clamp(options.preconditioner_ilu_drop_tol, 0.0, 1.0e-2);
  const bool sparse_ilu_pipeline =
      result.preconditioner_requested == "fmm-nearfield-ilu0" ||
      result.preconditioner_requested == "fmm-nearfield-ilu0-coarse" ||
      result.preconditioner_requested == "distance-knn-ilu0" ||
      result.preconditioner_requested == "classic-nearfield-ilu0";
  result.preconditioner_ilu_regularization =
      sparse_ilu_pipeline &&
              options.preconditioner_ilu_regularization != "off"
          ? std::string("adaptive")
          : std::string("off");
  result.preconditioner_ilu_regularization_max =
      sparse_ilu_pipeline
          ? std::clamp(options.preconditioner_ilu_regularization_max, 0.0,
                       1.0e-2)
          : 0.0;
  result.preconditioner_memory_budget_bytes =
      static_cast<std::size_t>(
          std::clamp(options.preconditioner_memory_budget_mb, 16, 8192)) *
      1024u * 1024u;
  result.preconditioner_coarse_aggregate_cap =
      std::clamp(options.preconditioner_coarse_aggregate_cap, 4, 64);
  result.preconditioner_coarse_base =
      options.preconditioner_coarse_base == "block-budgeted"
          ? std::string("block-budgeted")
          : std::string("diagonal");
  result.preconditioner_effective = "diagonal";
  result.gmres_restart_requested = std::clamp(options.gmres_restart, 8, 128);
  result.gmres_restart_effective = result.gmres_restart_requested;
  result.best_residual = 0.0;
  result.best_residual_iter = 0;
  result.final_residual = 0.0;
  result.best_preconditioned_residual = 0.0;
  result.final_preconditioned_residual = 0.0;
  result.true_residual_abs = 0.0;
  result.true_residual_rel = 0.0;
  result.true_residual_probe_mode =
      options.true_residual_probe == "interval"
          ? std::string("interval")
          : (options.true_residual_probe == "off" ? std::string("off")
                                                  : std::string("restart"));
  result.true_residual_probe_scope =
      options.true_residual_probe_scope == "sampled"
          ? std::string("sampled")
          : (options.true_residual_probe_scope == "adaptive"
                 ? std::string("adaptive")
                 : std::string("full"));
  result.true_residual_probe_interval =
      std::clamp(options.true_residual_probe_interval, 1, 128);
  result.true_residual_sample_count =
      std::clamp(options.true_residual_sample_count, 128, 4096);
  result.true_residual_probe_calls = 0;
  result.true_residual_probe_ms = 0.0;
  result.sampled_true_residual_rel = 0.0;
  result.sampled_true_residual_rows = 0;
  result.sampled_probe_calls = 0;
  result.sampled_probe_ms = 0.0;
  result.full_true_residual_confirm_calls = 0;
  result.full_true_residual_confirm_ms = 0.0;
  result.adaptive_level = options.adaptive_level.empty()
                              ? std::string("strict")
                              : options.adaptive_level;
  result.cluster_policy = (options.cluster_policy == "leaf-pair-strict" ||
                           options.cluster_policy == "leaf-pair-balanced")
                              ? options.cluster_policy
                              : std::string("off");
  result.target_batch_size = std::clamp(options.target_batch_size, 1, 8);
  if (result.target_batch_size != 4 && result.target_batch_size != 8)
    result.target_batch_size = 1;
  result.leaf_pair_min_source_panels =
      std::clamp(options.leaf_pair_min_source_panels, 16, 512);
  result.leaf_pair_min_targets =
      std::clamp(options.leaf_pair_min_targets, 2, 16);
  result.leaf_kernel_profile =
      options.leaf_kernel_profile == "summary" ? "summary" : "off";
  result.leaf_work_profile =
      options.leaf_work_profile == "summary" ? "summary" : "off";
  result.matvec_cost_profile =
      options.matvec_cost_profile == "summary" ? "summary" : "off";
  result.jcp_direct_kernel =
      normalize_jcp_direct_kernel_name(options.jcp_direct_kernel);
  result.leaf_pair_selection_profile =
      options.leaf_pair_selection_profile == "summary" ? "summary" : "off";
  result.leaf_pair_probe_cache =
      options.leaf_pair_probe_cache == "per-matvec" ? "per-matvec" : "off";
  result.leaf_pair_target_group =
      options.leaf_pair_target_group == "same-target-leaf" ? "same-target-leaf"
                                                           : "scan";
  result.leaf_pair_directness_probe =
      options.leaf_pair_directness_probe == "ancestor-path" ? "ancestor-path"
                                                            : "full-traversal";
  result.leaf_pair_directness_cache =
      options.leaf_pair_directness_cache == "target-leaf-source-leaf"
          ? "target-leaf-source-leaf"
          : "off";
  result.leaf_pair_directness_verify =
      (options.leaf_pair_directness_verify == "sampled" ||
       options.leaf_pair_directness_verify == "full-small")
          ? options.leaf_pair_directness_verify
          : "off";
  result.leaf_pair_scheduler =
      (options.leaf_pair_scheduler == "deferred-worklist" ||
       options.leaf_pair_scheduler == "target-leaf-block" ||
       options.leaf_pair_scheduler == "cost-tile-worklist")
          ? options.leaf_pair_scheduler
          : "inline-select";
  result.leaf_pair_worklist_max_active =
      std::clamp(options.leaf_pair_worklist_max_active, 64, 8192);
  result.leaf_pair_worklist_verify =
      (options.leaf_pair_worklist_verify == "sampled" ||
       options.leaf_pair_worklist_verify == "full-small")
          ? options.leaf_pair_worklist_verify
          : "off";
  result.leaf_block_verify = (options.leaf_block_verify == "sampled" ||
                              options.leaf_block_verify == "full-small")
                                 ? options.leaf_block_verify
                                 : "off";
  result.leaf_mac = (options.leaf_mac == "audit" || options.leaf_mac == "on")
                        ? options.leaf_mac
                        : "off";
  result.leaf_pair_max_lookahead =
      std::clamp(options.leaf_pair_max_lookahead, 4, 128);
  result.interaction_mode =
      options.interaction_mode == "onthefly" ? "onthefly" : "cached";
  result.node_bounds_mode = options.node_bounds == "tight-aabb"
                                ? std::string("tight-aabb")
                                : std::string("moment");
  result.tree_builder = options.tree_builder == "topdown-aabb"
                            ? std::string("topdown-aabb")
                            : std::string("grouped-morton");
  result.expansion_center = (options.expansion_center == "area-centroid" ||
                             options.expansion_center == "aabb-center")
                                ? options.expansion_center
                                : std::string("legacy");
  result.fragmentation_safety = (options.fragmentation_safety == "audit" ||
                                 options.fragmentation_safety == "split")
                                    ? options.fragmentation_safety
                                    : std::string("off");
  result.mac_safety =
      (options.mac_safety == "audit" || options.mac_safety == "physical-gap")
          ? options.mac_safety
          : std::string("off");
  result.mac_gap_factor = std::clamp(
      std::isfinite(options.mac_gap_factor) ? options.mac_gap_factor : 1.5, 0.5,
      4.0);
  result.moment_audit = (options.moment_audit == "sampled" ||
                         options.moment_audit == "full-small")
                            ? options.moment_audit
                            : std::string("off");
  result.moment_construction = options.moment_construction == "direct-node"
                                   ? std::string("direct-node")
                                   : std::string("translated");
  result.moment_sum = options.moment_sum == "kahan" ? std::string("kahan")
                                                    : std::string("naive");
  result.convergence_audit = (options.convergence_audit == "sampled" ||
                              options.convergence_audit == "full-small")
                                 ? options.convergence_audit
                                 : std::string("off");
  result.convergence_safety = (options.convergence_safety == "audit" ||
                               options.convergence_safety == "split")
                                  ? options.convergence_safety
                                  : std::string("off");
  result.convergence_rho_cap = std::clamp(
      std::isfinite(options.convergence_rho_cap) ? options.convergence_rho_cap
                                                 : 0.0,
      0.0, 2.0);
  result.mac_aspect_cap = std::clamp(
      std::isfinite(options.mac_aspect_cap) ? options.mac_aspect_cap : 0.0, 0.0,
      50.0);
  result.convergence_estimator =
      (options.convergence_estimator == "moment-tail" ||
       options.convergence_estimator == "hybrid")
          ? options.convergence_estimator
          : std::string("off");
  result.convergence_tail_threshold =
      std::clamp(std::isfinite(options.convergence_tail_threshold)
                     ? options.convergence_tail_threshold
                     : 1.0e4,
                 1.0, 1.0e12);
  result.convergence_estimator_error_cap =
      std::clamp(std::isfinite(options.convergence_estimator_error_cap)
                     ? options.convergence_estimator_error_cap
                     : 1.0e-3,
                 1.0e-8, 1.0);
  result.convergence_unstable_rho_cap =
      std::clamp(std::isfinite(options.convergence_unstable_rho_cap)
                     ? options.convergence_unstable_rho_cap
                     : 0.25,
                 0.05, 2.0);
  result.convergence_calibration =
      (options.convergence_calibration == "consecutive" ||
       options.convergence_calibration == "aspect" ||
       options.convergence_calibration == "consecutive-aspect")
          ? options.convergence_calibration
          : std::string("stage3j");
  result.convergence_dielectric_amp = options.convergence_dielectric_amp == "on"
                                          ? std::string("on")
                                          : std::string("off");
  result.convergence_estimator_model =
      (options.convergence_estimator_model == "channel-tail" ||
       options.convergence_estimator_model == "channel-tail-eps")
          ? options.convergence_estimator_model
          : std::string("stage3l");
  result.convergence_channel_calibration =
      (options.convergence_channel_calibration == "k3-sqrt-eps" ||
       options.convergence_channel_calibration == "k3-eps" ||
       options.convergence_channel_calibration == "k4-sqrt-eps" ||
       options.convergence_channel_calibration == "k4-eps" ||
       options.convergence_channel_calibration == "all-derivative-sqrt-eps" ||
       options.convergence_channel_calibration == "polynomial-eps")
          ? options.convergence_channel_calibration
          : std::string("stage3m");
  result.convergence_split_policy = options.convergence_split_policy == "k3-hot"
                                        ? std::string("k3-hot")
                                        : std::string("global");
  result.convergence_split_selectivity =
      (options.convergence_split_selectivity == "absolute-aware" ||
       options.convergence_split_selectivity == "channel-absolute-aware")
          ? options.convergence_split_selectivity
          : std::string("legacy");
  result.convergence_split_absolute_floor =
      std::clamp(std::isfinite(options.convergence_split_absolute_floor)
                     ? options.convergence_split_absolute_floor
                     : 0.0,
                 0.0, 1.0e-4);
  result.convergence_overreject_audit =
      (options.convergence_overreject_audit == "sampled" ||
       options.convergence_overreject_audit == "full-small")
          ? options.convergence_overreject_audit
          : std::string("off");
  result.convergence_overreject_detail =
      (options.convergence_overreject_detail == "summary" ||
       options.convergence_overreject_detail == "sampled-detail")
          ? options.convergence_overreject_detail
          : std::string("off");
  result.convergence_shape_audit =
      (options.convergence_shape_audit == "summary" ||
       options.convergence_shape_audit == "sampled-detail")
          ? options.convergence_shape_audit
          : std::string("off");
  result.convergence_kernel_response_audit =
      (options.convergence_kernel_response_audit == "summary" ||
       options.convergence_kernel_response_audit == "sampled-detail")
          ? options.convergence_kernel_response_audit
          : std::string("off");
  result.channel_missed_semantics =
      options.channel_missed_semantics == "accepted-only"
          ? std::string("accepted-only")
          : std::string("legacy");
  result.accepted_miss_diagnostics =
      (options.accepted_miss_diagnostics == "summary" ||
       options.accepted_miss_diagnostics == "sampled-detail")
          ? options.accepted_miss_diagnostics
          : std::string("off");
  result.accepted_miss_absolute_floor =
      std::clamp(std::isfinite(options.accepted_miss_absolute_floor)
                     ? options.accepted_miss_absolute_floor
                     : 1.0e-7,
                 0.0, 1.0e-4);
  result.convergence_k3_accepted_rho_cap =
      std::clamp(std::isfinite(options.convergence_k3_accepted_rho_cap)
                     ? options.convergence_k3_accepted_rho_cap
                     : 0.0,
                 0.0, 2.0);
  result.mac_distance_policy = options.mac_distance_policy == "surface-margin"
                                   ? std::string("surface-margin")
                                   : std::string("center");
  result.mac_surface_margin_factor =
      std::clamp(std::isfinite(options.mac_surface_margin_factor)
                     ? options.mac_surface_margin_factor
                     : 0.25,
                 0.0, 0.75);
  result.mac_surface_margin_policy =
      (options.mac_surface_margin_policy == "adaptive" ||
       options.mac_surface_margin_policy == "risk-scaled")
          ? options.mac_surface_margin_policy
          : "all";
  result.mac_surface_margin_aspect_threshold =
      std::clamp(std::isfinite(options.mac_surface_margin_aspect_threshold)
                     ? options.mac_surface_margin_aspect_threshold
                     : 3.0,
                 1.0, 50.0);
  result.mac_surface_margin_rho_trigger =
      std::clamp(std::isfinite(options.mac_surface_margin_rho_trigger)
                     ? options.mac_surface_margin_rho_trigger
                     : 0.55,
                 0.0, 2.0);
  result.mac_surface_margin_soft_aspect_threshold =
      std::clamp(std::isfinite(options.mac_surface_margin_soft_aspect_threshold)
                     ? options.mac_surface_margin_soft_aspect_threshold
                     : 0.0,
                 0.0, 50.0);
  result.mac_surface_margin_soft_rho_trigger =
      std::clamp(std::isfinite(options.mac_surface_margin_soft_rho_trigger)
                     ? options.mac_surface_margin_soft_rho_trigger
                     : 0.0,
                 0.0, 2.0);
  result.mac_surface_margin_soft_factor =
      std::clamp(std::isfinite(options.mac_surface_margin_soft_factor)
                     ? options.mac_surface_margin_soft_factor
                     : 0.10,
                 0.0, 0.75);
  result.expansion_audit = (options.expansion_audit == "sampled" ||
                            options.expansion_audit == "full-small")
                               ? options.expansion_audit
                               : std::string("off");
  result.expansion_audit_channel =
      (options.expansion_audit_channel == "summary" ||
       options.expansion_audit_channel == "sampled-detail")
          ? options.expansion_audit_channel
          : std::string("off");
  result.quad_buffer_auto_candidate =
      (options.quad_buffer == "gauss7" || options.quad_buffer == "adaptive")
          ? std::string("explicit")
          : std::string("off");
  result.derivative_mode = options.derivative_mode.empty()
                               ? std::string("analytic")
                               : options.derivative_mode;
  result.surface_generator = options.surface_geometry == "point-panel"
                                 ? std::string("titan_surface_point_panel")
                             : options.primary_surface == "grid-ses"
                                 ? std::string("titan_grid_watertight_ses")
                             : options.primary_surface == "analytic-ses"
                                 ? std::string("titan_analytic_patch_ray_ses")
                                 : std::string("titan_sphere_patch");
  const bool node_patch_discretization =
      options.discretization == "node-patch" ||
      formulation.formulation == "tabi2-nodepatch";
  if (node_patch_discretization &&
      formulation.formulation != "jcp2013-centroid" &&
      formulation.formulation != "jcp2013-treecode" &&
      formulation.formulation != "tabi2-nodepatch") {
    throw std::runtime_error(
        "PBE BIE: node-patch discretization is supported only by the JCP "
        "2013 coupled formulations.");
  }
  result.discretization =
      node_patch_discretization ? std::string("node_patch")
                                : std::string("centroid");
  result.rhs_scale = formulation.rhs_scale;
  result.coulomb_factor_kcal_mol = kCoulombKcalPerMol;
  result.total_charge = std::accumulate(
      atoms.begin(), atoms.end(), 0.0,
      [](double sum, const Atom &atom) { return sum + atom.q; });

  LongRunProgressWriter progress(options, executor.capacity());
  progress.write("surface_start", result, nullptr, 0, 0, 0, 0.0, "running",
                 true);
  std::vector<Panel> panels = build_surface(atoms, options, &result, &executor);
  // The `current` analytic-SES engine takes its inside/outside sign from a
  // 26-direction sampling of the probe ball.  That sampling does not refine
  // with the grid, so the mesh converges to the sampled closing rather than
  // to the SES, and no vertex is ever checked against the analytic patches.
  // Measured on an isolated 1.7 A sphere with probe 1.4 A: vertices sit up to
  // 0.086 A outside the surface at every spacing from 0.75 A down to
  // 0.094 A, while the area error grows from -1.5% to +7.2%.  The engine is
  // kept because it is what published TITAN runs used, and its status is a
  // closed-mesh check only, so nothing else in the output says this.
  if (options.primary_surface == "analytic-ses" &&
      result.surface_analytic_ses_engine == "current") {
    TitanUI::print_warning(
        "TABI analytic-ses engine 'current': vertices are not certified"
        " against the analytic SES.  Its sign comes from a 26-direction"
        " sampling of the probe ball, whose boundary lies outside the true"
        " surface between sample directions by up to 0.09 A at probe 1.4 A,"
        " and refining -pbe-bie-ses-grid-spacing does not reduce it (measured"
        " 8.3e-2, 8.1e-2, 8.6e-2, 8.6e-2 A at h=0.75, 0.375, 0.1875,"
        " 0.09375 on a one-atom sphere, with area error -1.5% to +7.2%)."
        "  Energies can still improve with h through error cancellation."
        "  Use -pbe-bie-analytic-ses-engine=patch-ray-experimental for a"
        " surface whose vertices are certified on the analytic patches,"
        " or current-dense for the same engine at a finer angular"
        " resolution; keep 'current' only to reproduce earlier runs.");
  }
  // current-dense samples 98 directions rather than 26.  That is a finer
  // angular resolution and not a finer grid, so it lowers the floor by about
  // four and does not remove it, and no vertex is certified either way.
  if (options.primary_surface == "analytic-ses" &&
      result.surface_analytic_ses_engine == "current-dense") {
    TitanUI::print_warning(
        "TABI analytic-ses engine 'current-dense': vertices are not certified"
        " against the analytic SES.  Its sign comes from a 98-direction"
        " sampling of the probe ball, which lowers the sampling error rather"
        " than removing it: the boundary still lies outside the true surface"
        " between sample directions, and refining -pbe-bie-ses-grid-spacing"
        " does not reduce that part (measured mean radial error 4.9e-3 and"
        " 6.7e-3 A at h=0.3 and 0.15 on a one-atom sphere, against 2.3e-2"
        " and 2.5e-2 for 'current').  At h=0.6 and coarser the grid dominates"
        " and the two are comparable.  Use"
        " -pbe-bie-analytic-ses-engine=patch-ray-experimental for a surface"
        " whose vertices are certified on the analytic patches, and 'current'"
        " to reproduce earlier runs.");
  }
  // The convergence estimator, the far-expansion audit and the accepted-miss
  // diagnostics are called from one of the two far-field traversals in
  // tabi_treecode_operator.cpp, the one that reports interaction_mode
  // "onthefly".  The cached traversal carries none of them, and analytical
  // FMM/M2L does not send the far field through either.  Asking for those
  // audits in any other configuration is not an error and changes no answer,
  // but every counter they publish comes back zero -- which reads exactly
  // like "the far field is fine".  Say so instead of leaving it silent.
  {
    const auto requested = [](const std::string &mode) {
      return !mode.empty() && mode != "off";
    };
    const bool wants_instrumented_far =
        requested(options.expansion_audit) ||
        requested(options.convergence_audit) ||
        requested(options.accepted_miss_diagnostics) ||
        requested(options.convergence_overreject_audit) ||
        options.convergence_estimator == "moment-tail" ||
        options.convergence_estimator == "hybrid";
    if (wants_instrumented_far) {
      const bool analytic_m2l = options.fmm_m2l == "single-matvec-analytic";
      const bool cached_traversal = options.interaction_mode != "onthefly";
      if (analytic_m2l || cached_traversal) {
        std::ostringstream message;
        message
            << "TABI far-field audits were requested but cannot observe this"
               " configuration, so their counters will read zero rather than"
               " clean.  ";
        if (analytic_m2l)
          message << "Analytical FMM/M2L carries the far field outside the"
                     " instrumented traversal, so it sees no far"
                     " interactions at all; add -pbe-bie-fmm-m2l=off after"
                     " -pbe-bie-fast-operator to route through the treecode.";
        if (analytic_m2l && cached_traversal)
          message << "  ";
        if (cached_traversal)
          message << "The cached interaction traversal carries none of these"
                     " audits; add -pbe-bie-interaction-mode=onthefly.  Note"
                     " that this measures the on-the-fly traversal, which is"
                     " not the default one.";
        TitanUI::print_warning(message.str());
      }
    }
  }
  result.charge_resolution = diagnose_charge_resolution(atoms, panels, options);
  if (result.charge_resolution.min_ratio >= 0.0) {
    std::ostringstream message;
    message << std::setprecision(5) << "TABI charge/mesh min d/h="
            << result.charge_resolution.min_ratio
            << " (retained curved panels; h=max edge; heuristic)";
    if (result.charge_resolution.warning)
      TitanUI::print_warning(message.str() +
          " < 1.5: check local resolution and energy convergence.");
    else
      TitanUI::print_info(message.str() + "; no resolution flag, not an accuracy certificate.");
  }
  std::size_t node_patch_topology_bytes = 0;
  if (node_patch_discretization) {
    if (options.surface_geometry != "triangle")
      throw std::runtime_error(
          "PBE BIE: node-patch discretization requires "
          "-pbe-bie-geometry=triangle.");
    auto parent_panels = std::move(panels);
    panels = build_node_patch_panels(parent_panels, options.diagonal_mode);
    if (!panels.empty() && panels.front().node_patch_geometry) {
      const NodePatchGeometry &g = *panels.front().node_patch_geometry;
      result.node_patch_normal_source =
          "analytic=" + std::to_string(g.analytic_normal_nodes) +
          ";averaged=" + std::to_string(g.averaged_normal_nodes) +
          ";rejected=" + std::to_string(g.analytic_normal_rejected);
    }
    // A node-patch cell is the barycentric dual of a vertex, which is well
    // defined for an interior vertex of a closed mesh.  A closed triangulation
    // has V/F = 0.5; every open patch boundary pushes the ratio up, and those
    // boundary vertices get cells that do not close.  Test the measured ratio
    // rather than the surface name: an isolated sphere built by sphere-patch
    // is closed and works, while the same builder on a 43-atom ligand at 24
    // points gave 6345 nodes to 5162 triangles, V/F 1.229, where GMRES
    // stalled near residual 0.3 and centroid reached 1.8e-11 on the same
    // input.  node-patch on a watertight grid-ses reached 1.9e-10 there in
    // half the unknowns.
    if (!parent_panels.empty()) {
      const double vertex_face_ratio =
          static_cast<double>(panels.size()) /
          static_cast<double>(parent_panels.size());
      if (vertex_face_ratio > 0.6) {
        std::ostringstream message;
        message << std::setprecision(4)
                << "TABI node-patch on a surface that is not closed: "
                << panels.size() << " nodes to " << parent_panels.size()
                << " triangles, V/F=" << vertex_face_ratio
                << " where a closed mesh gives 0.5.  The excess is dual cells"
                   " on patch boundaries, which do not close; on a 43-atom"
                   " test this stalled GMRES near residual 0.3 where centroid"
                   " converged.  Prefer a watertight surface"
                   " (-pbe-bie-primary-surface grid-ses or analytic-ses with a"
                   " positive -pbe-probe) for node-patch, or"
                   " -pbe-bie-discretization centroid on this one.  They are"
                   " different surfaces, not two routes to one answer.";
        TitanUI::print_warning(message.str());
      }
    }
    const auto curved_begin=std::chrono::steady_clock::now();
    // The PN lift is only as good as the triangles under it.  The marching
    // Grid-SES keeps its slivers and needles (it is not remeshed), and at
    // h = 1.25 A on 1SH1 the lift there gave the normal-derivative unknown a
    // spurious mode 2000 times its size and an energy four times too large,
    // with GMRES converged.  So Grid-SES keeps the flat cell, as in v0.5; the
    // curved cell is built on the remeshed analytic SES.
    const bool curved_surface=options.primary_surface!="grid-ses";
    if(!curved_surface)
      std::cerr<<"TABI node-patch: Grid-SES keeps the flat cell (the curved"
                 " PN cell needs the remeshed analytic SES)\n";
    const bool curved=curved_surface&&prepare_curved_node_patches(parent_panels,panels,
      std::max(1e-12,options.eps_out)/std::max(1e-12,options.eps_in),
      debye_kappa_inv_a(options.salt_m,options.temperature_k,options.eps_out),
      options.surface_only?0u:static_cast<std::size_t>(
        std::max(0,options.curved_pair_cache_mb))*1024u*1024u,executor,options.sum_mode!="treecode",
      options.curved_sample_order,
      // The analytic FMM sums one point per cell beyond its near list, so
      // the curved cell is integrated only inside a near zone there.
      options.fmm_m2l=="single-matvec-analytic"?options.curved_near_zone_radii:0.0);
    result.node_patch_representation=
        curved && !panels.empty() && panels.front().node_patch_geometry &&
                panels.front().node_patch_geometry->curved
            ? panels.front().node_patch_geometry->curved->representation
            : std::string("flat_legacy");
    result.node_patch_certificate = node_patch_certificate(panels);
    if(curved){
      if(!options.surface_only && options.fmm_m2l!="off" &&
         options.fmm_m2l!="single-matvec-analytic")
        throw std::runtime_error("curved_node_patch_requires_direct_classic_treecode_or_p1:"+options.fmm_m2l);
      // The experimental Cartesian full adapter corrects one-ring pairs only,
      // not the curved near zone the p1 and local p4 near loops correct.
      if(!options.surface_only && cartesian_full_requested)
        throw std::runtime_error("curved_node_patch_unsupported_on_cartesian_full_experimental");
      result.surface_mesh_ms+=elapsed_ms(curved_begin,std::chrono::steady_clock::now());
      result.surface_area_a2=std::accumulate(panels.begin(),panels.end(),0.0,[](double a,const Panel&p){return a+p.area;});
    } else if(!options.surface_only) {
      // A curved dual cell is normally always available: exact spherical
      // parents on sphere-patch/vdW, and a PN-triangle lift from vertices and
      // node normals on any other closed mesh.  Reaching this branch means the
      // lift was refused - unusable node normals, or TITAN_CURVED_NODE_PATCH
      // set to off - and the flat model is first order where the curved one is
      // second, so say so rather than leave it to a policy field.
      TitanUI::print_warning(
          "TABI node-patch on surface '" + options.primary_surface + "' (" +
          options.surface_mode + "): the curved dual cell was not built "
          "(node normals unusable, or TITAN_CURVED_NODE_PATCH=off), so the "
          "flat legacy cell is used.  It converges one order slower than "
          "either curved representation.  Reported as "
          "node_patch_representation=flat_legacy.");
    }
    if (!panels.empty() && panels.front().node_patch_geometry != nullptr) {
      const NodePatchGeometry &geometry = *panels.front().node_patch_geometry;
      node_patch_topology_bytes =
          sizeof(NodePatchGeometry) +
          geometry.vertex_xyz.size() * sizeof(std::array<double, 3>) +
          geometry.triangles.size() * sizeof(NodePatchTriangle) +
          (geometry.incident_offsets.size() +
           geometry.incident_triangles.size() +
           geometry.adjacent_offsets.size() +
           geometry.adjacent_nodes.size()) *
              sizeof(std::size_t);
      result.solid_angle_fallbacks = geometry.solid_angle_fallbacks;
      result.solid_angle_clamps += geometry.solid_angle_clamps;
      result.diag_min = std::numeric_limits<double>::infinity();
      result.diag_max = -std::numeric_limits<double>::infinity();
      for (const Panel &panel : panels) {
        result.diag_min = std::min(result.diag_min, panel.diagonal);
        result.diag_max = std::max(result.diag_max, panel.diagonal);
      }
    }
  }
  result.panel_count = panels.size();
  result.boundary_nodes = panels.size();
  if (options.fmm_cartesian_retry_runtime) {
    const cartesian_retry::CartesianRetryGeometryFingerprint fingerprint =
        cartesian_retry::fingerprint_cartesian_retry_geometry(panels);
    if (!fingerprint.valid || fingerprint.value == 0) {
      throw std::runtime_error(
          "cartesian_full_fmm_retry_geometry_fingerprint_failed");
    }
    result.fmm_cartesian_retry_geometry_hash = fingerprint.value;
    result.fmm_cartesian_retry_geometry_valid = true;
  }
  const int surface_point_max = options.highres_surface ? 65536 : 4096;
  result.surface_point_count_requested =
      std::clamp(options.surface_point_count, 8, surface_point_max);
  result.panel_density_per_a2 =
      result.surface_area_a2 > 1.0e-12
          ? static_cast<double>(result.panel_count) / result.surface_area_a2
          : 0.0;
  if (result.panel_count == 0) {
    result.panel_budget_status = "empty";
  } else if (result.panel_density_per_a2 < 1.5) {
    result.panel_budget_status = "coarse";
  } else if (result.panel_density_per_a2 <= 3.0) {
    result.panel_budget_status = "target_density";
  } else {
    result.panel_budget_status = "dense";
  }
  progress.write("surface_end", result, nullptr, 0, 0, 0, 0.0, "running", true);
  result.taylor_order = std::clamp(options.taylor_order, 2, 5);
  result.near_integration = resolve_near_integration_mode(
      node_patch_discretization && options.near_integration == "auto"
          ? std::string("analytic_singular")
          : options.near_integration,
      panels);
  result.quad_buffer = resolve_quad_buffer_mode(
      options.quad_buffer, result.near_integration, panels);
  if (options.quad_buffer == "auto")
    result.quad_buffer_auto_candidate = "off";
  result.memory_bytes =
      panels.size() * sizeof(Panel) + panels.size() * sizeof(double) * 8 +
      node_patch_topology_bytes + sphere_patch_quadrature_storage_bytes(panels) +
      curved_node_storage_bytes(panels);

  if (options.surface_only) {
    result.surface_only = true;
    result.converged = true;
    result.status = "ok";
    result.reaction_energy_source = "not_computed_surface_only";
    result.convergence_reason = "surface_only";
    result.sum_mode_effective = "surface_only";
    result.matvec_mode = "not_built_surface_only";
    result.fmm_m2l_mode = "off";
    result.preconditioner_requested = options.preconditioner;
    result.preconditioner_effective = "not_built_surface_only";
    result.preconditioner_quality_status = "not_built_surface_only";
    result.iterations = 0;
    result.residual = 0.0;
    result.final_residual = 0.0;
    result.unknown_count = 0;
    result.peak_memory_bytes =
        std::max(result.memory_bytes, current_working_set_bytes());
    progress.write("surface_only_done", result, nullptr, 0, 0, 0, 0.0,
                   result.status, true);
    return result;
  }

  if (panels.empty()) {
    result.converged = true;
    result.status = "ok";
    result.reaction_energy_source = "tabi_boundary_integral";
    result.convergence_reason = "converged";
    result.sum_mode_effective = "direct";
    progress.write("solve_done", result, nullptr, 0, 0, 0, 0.0, result.status,
                   true);
    return result;
  }

  const double kappa = debye_kappa_inv_a(options.salt_m, options.temperature_k,
                                        options.eps_out);
  const double contrast = (options.eps_out - options.eps_in) /
                          std::max(1.0e-12, options.eps_out + options.eps_in);
  const double operator_contrast =
      contrast * formulation.operator_contrast_scale;
  result.kappa_inv_a = kappa;
  result.dielectric_contrast = contrast;
  result.operator_contrast = operator_contrast;
  std::string mode = options.sum_mode;
  const bool paper_bie_formulation =
      formulation.formulation == "jcp2013-centroid" ||
      formulation.formulation == "jcp2013-treecode" ||
      formulation.formulation == "tabi2-nodepatch";
  if (mode == "auto") {
    mode = paper_bie_formulation
               ? "direct"
               : (result.surface_geometry == "triangle"
                      ? "direct"
                      : (panels.size() > 1024 ? "treecode" : "direct"));
  }
  result.sum_mode_effective = mode;
  result.matvec_mode = mode;

  if (paper_bie_formulation) {
    if (mode == "treecode" && formulation.formulation != "jcp2013-treecode") {
      throw std::runtime_error(
          "PBE: -pbe=bie JCP/TABI2 direct formulations do not support "
          "-pbe-bie-sum=treecode yet; use auto/direct, pin "
          "-pbe-bie-formulation=jcp2013-treecode for the experimental "
          "paper-treecode validation path, or pin titan-v1 for the legacy "
          "scalar treecode.");
    }
    const bool experimental_treecode =
        formulation.formulation == "jcp2013-treecode" && mode == "treecode";
    result.matvec_mode = experimental_treecode ? "jcp2013_treecode" : "direct";
    result.sum_mode_effective = experimental_treecode ? "treecode" : "direct";
    result.taylor_order =
        experimental_treecode ? std::clamp(options.taylor_order, 2, 5) : 0;
    if (result.preconditioner_requested == "none")
      result.preconditioner_effective = "none";
    else if (result.preconditioner_requested == "block" ||
             result.preconditioner_requested == "block-budgeted")
      result.preconditioner_effective = "diagonal_block_pending";
    else if (result.preconditioner_requested == "coarse-galerkin-probe")
      result.preconditioner_effective = "diagonal_coarse_pending";
    else if (result.preconditioner_requested ==
             "fmm-nearfield-ilu0-coarse")
      result.preconditioner_effective = "fmm_ilu0_coarse_pending";
    else
      result.preconditioner_effective = "diagonal";
    const double eps_ratio =
        std::max(1.0e-12, options.eps_out) / std::max(1.0e-12, options.eps_in);
    const double lhs_phi = 0.5 * (1.0 + eps_ratio);
    const double lhs_phin = 0.5 * (1.0 + 1.0 / eps_ratio);
    const bool coupled = kappa > 1.0e-14;
    if (!coupled && (options.preconditioner == "fmm-block-jacobi" ||
                     options.preconditioner == "fmm-overlap-schwarz" ||
                     options.preconditioner == "fmm-nearfield-ilu0" ||
                     options.preconditioner == "distance-knn-overlap-schwarz" ||
                     options.preconditioner == "distance-knn-ilu0" ||
                     options.preconditioner == "classic-nearfield-ilu0")) {
      throw std::invalid_argument(
          "FMM-native BIE preconditioners require positive ionic strength and "
          "the coupled JCP system");
    }
    if (options.fmm_local == "gmres-p4" && !options.fmm_local_geometry_sweep &&
        (options.fmm_local_geometry_guard != "strict-geometry-mac" ||
         std::abs(options.fmm_local_geometry_theta - 0.5) > 1.0e-15)) {
      throw std::invalid_argument(
          "gmres-p4 requires strict-geometry-mac with geometry theta 0.5");
    }
    result.operator_contrast = eps_ratio;
    result.unknown_count = coupled ? 2 * panels.size() : panels.size();
    result.bie_system = coupled ? "linear_pb_2x2" : "poisson";
    result.lhs_diag_phi = lhs_phi;
    result.lhs_diag_phin = coupled ? lhs_phin : 0.0;
    result.energy_equation = coupled ? "jcp2013_eq14" : "jcp2013_eq22";
    result.kernel_convention = "G0_Gk_source_normal_target_normal_outward_" +
                               result.near_integration + "_near";

    const auto rhs_begin = std::chrono::steady_clock::now();
    double s1_norm = 0.0;
    double s2_norm = 0.0;
    const std::vector<double> rhs =
        build_jcp_rhs(atoms, panels, options.eps_in, coupled, executor,
                      &s1_norm, &s2_norm);
    result.rhs_ms = elapsed_ms(rhs_begin, std::chrono::steady_clock::now());
    result.source_norm_s1 = s1_norm;
    result.source_norm_s2 = s2_norm;

    const int size_factor = coupled ? 8 : 4;
    const int max_iter =
        std::min(std::max(1, options.max_iter),
                 std::max(64, static_cast<int>(panels.size()) * size_factor));
    const double tol = std::max(1.0e-12, options.residual_tol);
    std::vector<double> diagonal(result.unknown_count, 1.0);
    if (result.preconditioner_effective != "none") {
      for (std::size_t i = 0; i < panels.size(); ++i) {
        diagonal[i] = jcp_lhs_phi_for_panel(panels[i], eps_ratio, lhs_phi);
        if (coupled)
          diagonal[panels.size() + i] =
              jcp_lhs_phin_for_panel(panels[i], eps_ratio, lhs_phin);
      }
    }

    double matvec_ms_total = 0.0;
    std::size_t matvec_calls = 0;
    TreeWorkspace jcp_tree;
    TreecodeStats jcp_tree_stats;
    jcp_tree_stats.near_integration = result.near_integration;
    jcp_tree_stats.quad_buffer = result.quad_buffer;
    jcp_tree_stats.interaction_mode = result.interaction_mode;
    jcp_tree_stats.node_bounds_mode = result.node_bounds_mode;
    jcp_tree_stats.tree_builder = result.tree_builder;
    jcp_tree_stats.expansion_center = result.expansion_center;
    jcp_tree_stats.fragmentation_safety = result.fragmentation_safety;
    jcp_tree_stats.mac_safety = result.mac_safety;
    jcp_tree_stats.mac_gap_factor = result.mac_gap_factor;
    jcp_tree_stats.moment_audit = result.moment_audit;
    jcp_tree_stats.moment_construction = result.moment_construction;
    jcp_tree_stats.moment_sum = result.moment_sum;
    jcp_tree_stats.convergence_audit = result.convergence_audit;
    jcp_tree_stats.convergence_safety = result.convergence_safety;
    jcp_tree_stats.convergence_rho_cap = result.convergence_rho_cap;
    jcp_tree_stats.mac_aspect_cap = result.mac_aspect_cap;
    jcp_tree_stats.convergence_estimator = result.convergence_estimator;
    jcp_tree_stats.convergence_tail_threshold =
        result.convergence_tail_threshold;
    jcp_tree_stats.convergence_estimator_error_cap =
        result.convergence_estimator_error_cap;
    jcp_tree_stats.convergence_unstable_rho_cap =
        result.convergence_unstable_rho_cap;
    jcp_tree_stats.convergence_calibration = result.convergence_calibration;
    jcp_tree_stats.convergence_dielectric_amp =
        result.convergence_dielectric_amp;
    jcp_tree_stats.convergence_estimator_model =
        result.convergence_estimator_model;
    jcp_tree_stats.convergence_channel_calibration =
        result.convergence_channel_calibration;
    jcp_tree_stats.convergence_split_policy = result.convergence_split_policy;
    jcp_tree_stats.convergence_split_selectivity =
        result.convergence_split_selectivity;
    jcp_tree_stats.convergence_split_absolute_floor =
        result.convergence_split_absolute_floor;
    jcp_tree_stats.convergence_overreject_audit =
        result.convergence_overreject_audit;
    jcp_tree_stats.convergence_overreject_detail =
        result.convergence_overreject_detail;
    jcp_tree_stats.convergence_shape_audit = result.convergence_shape_audit;
    jcp_tree_stats.convergence_kernel_response_audit =
        result.convergence_kernel_response_audit;
    jcp_tree_stats.channel_missed_semantics = result.channel_missed_semantics;
    jcp_tree_stats.accepted_miss_diagnostics = result.accepted_miss_diagnostics;
    jcp_tree_stats.accepted_miss_absolute_floor =
        result.accepted_miss_absolute_floor;
    jcp_tree_stats.convergence_k3_accepted_rho_cap =
        result.convergence_k3_accepted_rho_cap;
    jcp_tree_stats.mac_distance_policy = result.mac_distance_policy;
    jcp_tree_stats.mac_surface_margin_factor = result.mac_surface_margin_factor;
    jcp_tree_stats.mac_surface_margin_policy = result.mac_surface_margin_policy;
    jcp_tree_stats.mac_surface_margin_aspect_threshold =
        result.mac_surface_margin_aspect_threshold;
    jcp_tree_stats.mac_surface_margin_rho_trigger =
        result.mac_surface_margin_rho_trigger;
    jcp_tree_stats.mac_surface_margin_soft_aspect_threshold =
        result.mac_surface_margin_soft_aspect_threshold;
    jcp_tree_stats.mac_surface_margin_soft_rho_trigger =
        result.mac_surface_margin_soft_rho_trigger;
    jcp_tree_stats.mac_surface_margin_soft_factor =
        result.mac_surface_margin_soft_factor;
    jcp_tree_stats.cluster_policy = result.cluster_policy;
    jcp_tree_stats.target_batch_size = result.target_batch_size;
    jcp_tree_stats.leaf_pair_min_source_panels =
        result.leaf_pair_min_source_panels;
    jcp_tree_stats.leaf_pair_min_targets = result.leaf_pair_min_targets;
    jcp_tree_stats.leaf_kernel_profile = result.leaf_kernel_profile;
    jcp_tree_stats.leaf_work_profile = result.leaf_work_profile;
    jcp_tree_stats.matvec_cost_profile = result.matvec_cost_profile;
    jcp_tree_stats.jcp_direct_kernel = result.jcp_direct_kernel;
    jcp_tree_stats.leaf_pair_selection_profile =
        result.leaf_pair_selection_profile;
    jcp_tree_stats.leaf_pair_probe_cache = result.leaf_pair_probe_cache;
    jcp_tree_stats.leaf_pair_target_group = result.leaf_pair_target_group;
    jcp_tree_stats.leaf_pair_directness_probe =
        result.leaf_pair_directness_probe;
    jcp_tree_stats.leaf_pair_directness_cache =
        result.leaf_pair_directness_cache;
    jcp_tree_stats.leaf_pair_directness_verify =
        result.leaf_pair_directness_verify;
    jcp_tree_stats.leaf_pair_scheduler = result.leaf_pair_scheduler;
    jcp_tree_stats.leaf_pair_worklist_max_active =
        result.leaf_pair_worklist_max_active;
    jcp_tree_stats.leaf_pair_worklist_verify = result.leaf_pair_worklist_verify;
    jcp_tree_stats.leaf_block_verify = result.leaf_block_verify;
    jcp_tree_stats.leaf_pair_max_lookahead = result.leaf_pair_max_lookahead;
    jcp_tree_stats.expansion_audit = result.expansion_audit;
    jcp_tree_stats.expansion_audit_channel = result.expansion_audit_channel;
    const bool fmm_m2l_single_matvec =
        experimental_treecode && options.fmm_m2l == "single-matvec-analytic";
    const bool preconditioner_needs_jcp_tree =
        result.preconditioner_requested == "block" ||
        result.preconditioner_requested == "block-budgeted" ||
        result.preconditioner_requested == "coarse-galerkin-probe";
    const bool needs_jcp_tree =
        experimental_treecode || preconditioner_needs_jcp_tree;
    if (needs_jcp_tree) {
      progress.write("tree_build_start", result, &jcp_tree_stats, matvec_calls,
                     0, 0, 0.0, "running", true);
      const std::string jcp_tree_interaction_mode =
          fmm_m2l_single_matvec && !preconditioner_needs_jcp_tree
              ? std::string("onthefly")
              : (experimental_treecode ? options.interaction_mode
                                       : std::string("onthefly"));
      jcp_tree = build_tree_workspace(
          panels, executor,
          static_cast<std::size_t>(std::max(8, effective_tree_leaf_size)),
          effective_tree_opening_theta, options.leaf_near_policy,
          options.leaf_panel_kernel, options.adaptive_level, result.quad_buffer,
          jcp_tree_interaction_mode, result.node_bounds_mode,
          result.tree_builder, options.mac_safety, options.mac_gap_factor,
          options.moment_audit, result.expansion_center,
          result.fragmentation_safety, options.expansion_audit,
          options.expansion_audit_sample_count,
          options.expansion_audit_max_panels, options.moment_construction,
          options.moment_sum, options.convergence_audit,
          options.convergence_safety, options.convergence_rho_cap,
          options.mac_aspect_cap, options.convergence_estimator,
          options.convergence_tail_threshold,
          options.convergence_estimator_error_cap,
          options.convergence_unstable_rho_cap, options.convergence_calibration,
          options.cluster_policy, options.convergence_dielectric_amp,
          options.target_batch_size, options.leaf_pair_min_source_panels,
          options.leaf_pair_min_targets, options.convergence_estimator_model,
          options.expansion_audit_channel,
          options.convergence_channel_calibration,
          options.channel_missed_semantics, options.convergence_split_policy,
          options.accepted_miss_diagnostics,
          options.convergence_k3_accepted_rho_cap, options.mac_distance_policy,
          options.mac_surface_margin_factor, options.mac_surface_margin_policy,
          options.mac_surface_margin_aspect_threshold,
          options.mac_surface_margin_rho_trigger,
          options.accepted_miss_absolute_floor, options.leaf_pair_kernel,
          options.leaf_kernel_profile, options.leaf_work_profile,
          options.jcp_direct_kernel, options.leaf_pair_selection_profile,
          options.leaf_pair_probe_cache, options.leaf_pair_target_group,
          options.leaf_pair_max_lookahead, options.leaf_pair_directness_probe,
          options.leaf_pair_directness_cache,
          options.leaf_pair_directness_verify, options.leaf_pair_scheduler,
          options.leaf_pair_worklist_max_active,
          options.leaf_pair_worklist_verify, options.leaf_block_verify,
          options.mac_surface_margin_soft_aspect_threshold,
          options.mac_surface_margin_soft_rho_trigger,
          options.mac_surface_margin_soft_factor,
          options.convergence_split_selectivity,
          options.convergence_split_absolute_floor,
          options.convergence_overreject_audit,
          options.convergence_overreject_detail,
          options.convergence_shape_audit,
          options.convergence_kernel_response_audit, result.leaf_mac,
          options.fmm_m2l);
      jcp_tree.leaf_mac = result.leaf_mac;
      jcp_tree.taylor_order = result.taylor_order;
      jcp_tree.derivative_mode = result.derivative_mode;
      result.interaction_mode = jcp_tree.interaction_mode;
      result.node_bounds_mode = jcp_tree.node_bounds_mode;
      result.tree_builder = jcp_tree.tree_builder;
      result.expansion_center = jcp_tree.expansion_center;
      result.fragmentation_safety = jcp_tree.fragmentation_safety;
      result.mac_safety = jcp_tree.mac_safety;
      result.mac_gap_factor = jcp_tree.mac_gap_factor;
      result.moment_audit = jcp_tree.moment_audit;
      result.moment_construction = jcp_tree.moment_construction;
      result.moment_sum = jcp_tree.moment_sum;
      result.convergence_audit = jcp_tree.convergence_audit;
      result.convergence_safety = jcp_tree.convergence_safety;
      result.convergence_rho_cap = jcp_tree.convergence_rho_cap;
      result.mac_aspect_cap = jcp_tree.mac_aspect_cap;
      result.convergence_estimator = jcp_tree.convergence_estimator;
      result.convergence_tail_threshold = jcp_tree.convergence_tail_threshold;
      result.convergence_estimator_error_cap =
          jcp_tree.convergence_estimator_error_cap;
      result.convergence_unstable_rho_cap =
          jcp_tree.convergence_unstable_rho_cap;
      result.convergence_calibration = jcp_tree.convergence_calibration;
      result.convergence_dielectric_amp = jcp_tree.convergence_dielectric_amp;
      result.convergence_estimator_model = jcp_tree.convergence_estimator_model;
      result.convergence_channel_calibration =
          jcp_tree.convergence_channel_calibration;
      result.convergence_split_policy = jcp_tree.convergence_split_policy;
      result.convergence_split_selectivity =
          jcp_tree.convergence_split_selectivity;
      result.convergence_split_absolute_floor =
          jcp_tree.convergence_split_absolute_floor;
      result.convergence_overreject_audit =
          jcp_tree.convergence_overreject_audit;
      result.convergence_overreject_detail =
          jcp_tree.convergence_overreject_detail;
      result.convergence_shape_audit = jcp_tree.convergence_shape_audit;
      result.convergence_kernel_response_audit =
          jcp_tree.convergence_kernel_response_audit;
      result.channel_missed_semantics = jcp_tree.channel_missed_semantics;
      result.accepted_miss_diagnostics = jcp_tree.accepted_miss_diagnostics;
      result.accepted_miss_absolute_floor =
          jcp_tree.accepted_miss_absolute_floor;
      result.convergence_k3_accepted_rho_cap =
          jcp_tree.convergence_k3_accepted_rho_cap;
      result.mac_distance_policy = jcp_tree.mac_distance_policy;
      result.mac_surface_margin_factor = jcp_tree.mac_surface_margin_factor;
      result.mac_surface_margin_policy = jcp_tree.mac_surface_margin_policy;
      result.mac_surface_margin_aspect_threshold =
          jcp_tree.mac_surface_margin_aspect_threshold;
      result.mac_surface_margin_rho_trigger =
          jcp_tree.mac_surface_margin_rho_trigger;
      result.mac_surface_margin_soft_aspect_threshold =
          jcp_tree.mac_surface_margin_soft_aspect_threshold;
      result.mac_surface_margin_soft_rho_trigger =
          jcp_tree.mac_surface_margin_soft_rho_trigger;
      result.mac_surface_margin_soft_factor =
          jcp_tree.mac_surface_margin_soft_factor;
      result.cluster_policy = jcp_tree.cluster_policy;
      result.target_batch_size = static_cast<int>(jcp_tree.target_batch_size);
      result.leaf_pair_min_source_panels =
          static_cast<int>(jcp_tree.leaf_pair_min_source_panels);
      result.leaf_pair_min_targets =
          static_cast<int>(jcp_tree.leaf_pair_min_targets);
      result.leaf_kernel_profile = jcp_tree.leaf_kernel_profile;
      result.leaf_work_profile = jcp_tree.leaf_work_profile;
      result.jcp_direct_kernel = jcp_tree.jcp_direct_kernel;
      result.leaf_pair_selection_profile = jcp_tree.leaf_pair_selection_profile;
      result.leaf_pair_probe_cache = jcp_tree.leaf_pair_probe_cache;
      result.leaf_pair_target_group = jcp_tree.leaf_pair_target_group;
      result.leaf_pair_directness_probe = jcp_tree.leaf_pair_directness_probe;
      result.leaf_pair_directness_cache = jcp_tree.leaf_pair_directness_cache;
      result.leaf_pair_directness_verify = jcp_tree.leaf_pair_directness_verify;
      result.leaf_pair_scheduler = jcp_tree.leaf_pair_scheduler;
      result.leaf_pair_worklist_max_active =
          static_cast<int>(jcp_tree.leaf_pair_worklist_max_active);
      result.leaf_pair_worklist_verify = jcp_tree.leaf_pair_worklist_verify;
      result.leaf_block_verify = jcp_tree.leaf_block_verify;
      result.leaf_pair_max_lookahead =
          static_cast<int>(jcp_tree.leaf_pair_max_lookahead);
      result.expansion_audit = jcp_tree.expansion_audit;
      result.expansion_audit_channel = jcp_tree.expansion_audit_channel;
      jcp_tree_stats.interaction_mode = jcp_tree.interaction_mode;
      jcp_tree_stats.node_bounds_mode = jcp_tree.node_bounds_mode;
      jcp_tree_stats.tree_builder = jcp_tree.tree_builder;
      jcp_tree_stats.expansion_center = jcp_tree.expansion_center;
      jcp_tree_stats.fragmentation_safety = jcp_tree.fragmentation_safety;
      jcp_tree_stats.mac_safety = jcp_tree.mac_safety;
      jcp_tree_stats.mac_gap_factor = jcp_tree.mac_gap_factor;
      jcp_tree_stats.moment_audit = jcp_tree.moment_audit;
      jcp_tree_stats.moment_construction = jcp_tree.moment_construction;
      jcp_tree_stats.moment_sum = jcp_tree.moment_sum;
      jcp_tree_stats.convergence_audit = jcp_tree.convergence_audit;
      jcp_tree_stats.convergence_safety = jcp_tree.convergence_safety;
      jcp_tree_stats.convergence_rho_cap = jcp_tree.convergence_rho_cap;
      jcp_tree_stats.mac_aspect_cap = jcp_tree.mac_aspect_cap;
      jcp_tree_stats.convergence_estimator = jcp_tree.convergence_estimator;
      jcp_tree_stats.convergence_tail_threshold =
          jcp_tree.convergence_tail_threshold;
      jcp_tree_stats.convergence_estimator_error_cap =
          jcp_tree.convergence_estimator_error_cap;
      jcp_tree_stats.convergence_unstable_rho_cap =
          jcp_tree.convergence_unstable_rho_cap;
      jcp_tree_stats.convergence_calibration = jcp_tree.convergence_calibration;
      jcp_tree_stats.convergence_dielectric_amp =
          jcp_tree.convergence_dielectric_amp;
      jcp_tree_stats.convergence_estimator_model =
          jcp_tree.convergence_estimator_model;
      jcp_tree_stats.convergence_channel_calibration =
          jcp_tree.convergence_channel_calibration;
      jcp_tree_stats.convergence_split_policy =
          jcp_tree.convergence_split_policy;
      jcp_tree_stats.convergence_split_selectivity =
          jcp_tree.convergence_split_selectivity;
      jcp_tree_stats.convergence_split_absolute_floor =
          jcp_tree.convergence_split_absolute_floor;
      jcp_tree_stats.convergence_overreject_audit =
          jcp_tree.convergence_overreject_audit;
      jcp_tree_stats.convergence_overreject_detail =
          jcp_tree.convergence_overreject_detail;
      jcp_tree_stats.convergence_shape_audit = jcp_tree.convergence_shape_audit;
      jcp_tree_stats.convergence_kernel_response_audit =
          jcp_tree.convergence_kernel_response_audit;
      jcp_tree_stats.channel_missed_semantics =
          jcp_tree.channel_missed_semantics;
      jcp_tree_stats.accepted_miss_diagnostics =
          jcp_tree.accepted_miss_diagnostics;
      jcp_tree_stats.accepted_miss_absolute_floor =
          jcp_tree.accepted_miss_absolute_floor;
      jcp_tree_stats.convergence_k3_accepted_rho_cap =
          jcp_tree.convergence_k3_accepted_rho_cap;
      jcp_tree_stats.mac_distance_policy = jcp_tree.mac_distance_policy;
      jcp_tree_stats.mac_surface_margin_factor =
          jcp_tree.mac_surface_margin_factor;
      jcp_tree_stats.mac_surface_margin_policy =
          jcp_tree.mac_surface_margin_policy;
      jcp_tree_stats.mac_surface_margin_aspect_threshold =
          jcp_tree.mac_surface_margin_aspect_threshold;
      jcp_tree_stats.mac_surface_margin_rho_trigger =
          jcp_tree.mac_surface_margin_rho_trigger;
      jcp_tree_stats.mac_surface_margin_soft_aspect_threshold =
          jcp_tree.mac_surface_margin_soft_aspect_threshold;
      jcp_tree_stats.mac_surface_margin_soft_rho_trigger =
          jcp_tree.mac_surface_margin_soft_rho_trigger;
      jcp_tree_stats.mac_surface_margin_soft_factor =
          jcp_tree.mac_surface_margin_soft_factor;
      jcp_tree_stats.cluster_policy = jcp_tree.cluster_policy;
      jcp_tree_stats.target_batch_size =
          static_cast<int>(jcp_tree.target_batch_size);
      jcp_tree_stats.leaf_pair_min_source_panels =
          static_cast<int>(jcp_tree.leaf_pair_min_source_panels);
      jcp_tree_stats.leaf_pair_min_targets =
          static_cast<int>(jcp_tree.leaf_pair_min_targets);
      jcp_tree_stats.leaf_kernel_profile = jcp_tree.leaf_kernel_profile;
      jcp_tree_stats.leaf_work_profile = jcp_tree.leaf_work_profile;
      jcp_tree_stats.matvec_cost_profile = result.matvec_cost_profile;
      jcp_tree_stats.jcp_direct_kernel = jcp_tree.jcp_direct_kernel;
      jcp_tree_stats.leaf_pair_selection_profile =
          jcp_tree.leaf_pair_selection_profile;
      jcp_tree_stats.leaf_pair_probe_cache = jcp_tree.leaf_pair_probe_cache;
      jcp_tree_stats.leaf_pair_target_group = jcp_tree.leaf_pair_target_group;
      jcp_tree_stats.leaf_pair_directness_probe =
          jcp_tree.leaf_pair_directness_probe;
      jcp_tree_stats.leaf_pair_directness_cache =
          jcp_tree.leaf_pair_directness_cache;
      jcp_tree_stats.leaf_pair_directness_verify =
          jcp_tree.leaf_pair_directness_verify;
      jcp_tree_stats.leaf_pair_scheduler = jcp_tree.leaf_pair_scheduler;
      jcp_tree_stats.leaf_pair_worklist_max_active =
          static_cast<int>(jcp_tree.leaf_pair_worklist_max_active);
      jcp_tree_stats.leaf_pair_worklist_verify =
          jcp_tree.leaf_pair_worklist_verify;
      jcp_tree_stats.leaf_block_verify = jcp_tree.leaf_block_verify;
      jcp_tree_stats.leaf_pair_max_lookahead =
          static_cast<int>(jcp_tree.leaf_pair_max_lookahead);
      jcp_tree_stats.expansion_audit = jcp_tree.expansion_audit;
      jcp_tree_stats.expansion_audit_channel = jcp_tree.expansion_audit_channel;
      jcp_tree_stats.tree_depth_max = jcp_tree.depth_max;
      jcp_tree_stats.node_radius_max = jcp_tree.node_radius_max;
      jcp_tree_stats.node_radius_mean = jcp_tree.node_radius_mean;
      jcp_tree_stats.node_aspect_ratio_max = jcp_tree.node_aspect_ratio_max;
      jcp_tree_stats.node_aspect_ratio_mean = jcp_tree.node_aspect_ratio_mean;
      result.tree_build_ms = jcp_tree.build_ms;
      result.tree_nodes = jcp_tree.nodes.size();
      result.tree_leaf_count = jcp_tree.leaf_count;
      result.tree_depth_max = jcp_tree.depth_max;
      result.memory_bytes +=
          jcp_tree.nodes.size() * sizeof(TreeNode) +
          jcp_tree.child_indices.size() * sizeof(std::size_t) +
          jcp_tree.order.size() * sizeof(std::size_t) +
          jcp_tree.morton.size() * sizeof(std::uint64_t) +
          panels.size() * 14u * sizeof(double) +
          panels.size() * sizeof(std::size_t);
      jcp_tree_stats.tree_build_ms = result.tree_build_ms;
      jcp_tree_stats.tree_nodes = result.tree_nodes;
      jcp_tree_stats.tree_leaf_count = result.tree_leaf_count;
      progress.write("tree_build_end", result, &jcp_tree_stats, matvec_calls, 0,
                     0, 0.0, "running", true);
    }
    auto build_sampled_target_rows = [](std::size_t row_count, int requested) {
      std::vector<std::size_t> rows;
      if (row_count == 0)
        return rows;
      const std::size_t m = std::min<std::size_t>(
          row_count,
          static_cast<std::size_t>(std::clamp(requested, 128, 4096)));
      rows.reserve(m);
      std::size_t previous = std::numeric_limits<std::size_t>::max();
      for (std::size_t k = 0; k < m; ++k) {
        std::size_t row =
            ((2u * k + 1u) * row_count) / std::max<std::size_t>(1, 2u * m);
        row = std::min(row, row_count - 1u);
        if (row != previous) {
          rows.push_back(row);
          previous = row;
        }
      }
      if (rows.empty())
        rows.push_back(0);
      return rows;
    };

    const bool cartesian_full_gmres = cartesian_full_requested;
    const bool gmres_p4 = options.fmm_local == "gmres-p4";
    const bool kifmm_gmres = options.fmm_engine == "kifmm-lite-experimental";
    const bool channel_hybrid_gmres =
        options.fmm_engine == "kifmm-channel-hybrid-experimental";
    const bool kifmm_any_gmres = kifmm_gmres || channel_hybrid_gmres;
    const bool kifmm_fallback_ilu_graph =
        kifmm_gmres && options.kifmm_ilu_graph == "kifmm-fallback-capped";
    [[maybe_unused]] const std::string kifmm_ilu0_graph_source =
        kifmm_fallback_ilu_graph ? "kifmm_fallback_capped_panel_graph"
                                 : "cartesian_fmm_m2l_near_panel_graph";
    std::unique_ptr<titan::bie::CartesianFullFmmSolverOperator>
        cartesian_full_operator;
    bool cartesian_full_workspace_accounted = false;
    std::uint64_t cartesian_full_tree_hash = 0;
    std::uint64_t cartesian_full_worklist_hash = 0;
    std::uint64_t cartesian_full_policy_hash = 0;
    if (cartesian_full_gmres) {
      titan::bie::CartesianFullFmmPrepareOptions full_options;
      full_options.expansion_order = options.bie_fmm_cartesian_order;
      full_options.leaf_size = std::max(8, effective_tree_leaf_size);
      full_options.max_depth = 10;
      full_options.theta = effective_tree_opening_theta;
      full_options.eps_ratio = eps_ratio;
      full_options.kappa = kappa;
      full_options.near_integration = result.near_integration;
      full_options.quad_buffer = result.quad_buffer;
      auto prepared =
          titan::bie::CartesianFullFmmSolverOperator::prepare(panels,
                                                               full_options,
                                                               executor);
      if (!prepared)
        throw std::runtime_error("cartesian_full_prepare_failed: " +
                                 prepared.error());
      cartesian_full_operator = std::move(*prepared);
      const auto &full_stats = cartesian_full_operator->stats();
#if defined(TITAN_CARTESIAN_FMM_TESTING)
      maybe_write_p3c0_packed_plan_inspection(*cartesian_full_operator);
#endif
      result.bie_cartesian_full_fmm_active = true;
      result.bie_cartesian_full_fmm_requested =
          "cartesian-full-experimental";
      result.bie_cartesian_full_fmm_effective =
          full_stats.effective_engine;
      result.bie_cartesian_full_fmm_order = full_stats.expansion_order;
      result.bie_cartesian_full_fmm_leaf_size = full_stats.leaf_size;
      result.bie_cartesian_full_fmm_opening_theta =
          full_stats.opening_theta;
      cartesian_full_tree_hash = full_stats.plan.tree_hash;
      cartesian_full_worklist_hash = full_stats.plan.worklist_hash;
      std::uint64_t policy_hash = kFmmAuditHashOffset;
      auto mix_policy_text = [&](const std::string &text) {
        for (const unsigned char character : text)
          fmm_audit_hash_mix(policy_hash,
                             static_cast<std::uint64_t>(character));
        fmm_audit_hash_mix(policy_hash, 0xffu);
      };
      fmm_audit_hash_mix(policy_hash, 0x43415254464d4dULL); // "CARTFMM"
      fmm_audit_hash_mix(policy_hash,
                         static_cast<std::uint64_t>(full_stats.expansion_order));
      fmm_audit_hash_mix(policy_hash,
                         static_cast<std::uint64_t>(full_stats.leaf_size));
      fmm_audit_hash_mix(
          policy_hash, std::bit_cast<std::uint64_t>(full_stats.opening_theta));
      fmm_audit_hash_mix(policy_hash, std::bit_cast<std::uint64_t>(eps_ratio));
      fmm_audit_hash_mix(policy_hash, std::bit_cast<std::uint64_t>(kappa));
      fmm_audit_hash_mix(policy_hash, coupled ? 1u : 0u);
      mix_policy_text(result.near_integration);
      mix_policy_text(result.quad_buffer);
      mix_policy_text(result.discretization);
      cartesian_full_policy_hash = policy_hash == 0 ? 1 : policy_hash;
      result.bie_cartesian_full_fmm_tree_hash = cartesian_full_tree_hash;
      result.bie_cartesian_full_fmm_worklist_hash =
          cartesian_full_worklist_hash;
      result.bie_cartesian_full_fmm_policy_hash = cartesian_full_policy_hash;
      result.bie_cartesian_full_fmm_nodes = full_stats.plan.nodes;
      result.bie_cartesian_full_fmm_far_pairs = full_stats.plan.far_pairs;
      result.bie_cartesian_full_fmm_direct_pairs =
          full_stats.plan.direct_pairs;
      result.bie_cartesian_full_fmm_direct_panel_pairs_expected =
          full_stats.plan.direct_panel_pairs;
      result.bie_cartesian_full_fmm_cache_entries =
          full_stats.plan.translation_cache_entries;
      result.bie_cartesian_full_fmm_cache_hits =
          full_stats.plan.translation_cache_hits;
      result.bie_cartesian_full_fmm_cache_bytes =
          full_stats.plan.translation_cache_bytes;
      result.bie_cartesian_full_fmm_plan_bytes =
          full_stats.plan.retained_plan_bytes;
      result.bie_cartesian_full_fmm_tree_seconds =
          full_stats.plan.tree_seconds;
      result.bie_cartesian_full_fmm_worklist_seconds =
          full_stats.plan.worklist_seconds;
      result.bie_cartesian_full_fmm_cache_seconds =
          full_stats.plan.cache_seconds;
      result.bie_cartesian_full_fmm_audit_seconds =
          full_stats.plan.audit_seconds;
      result.bie_cartesian_full_fmm_m2m_parity_error =
          full_stats.plan.m2m_parity_relative_error;
      result.bie_cartesian_full_fmm_l2l_parity_error =
          full_stats.plan.l2l_parity_relative_error;
      result.fmm_local_gmres_active = true;
      result.fmm_local_prepare_seconds = full_stats.plan.setup_seconds;
      result.fmm_local_plan_build_count = 1;
      result.fmm_local_cache_build_count =
          full_stats.plan.translation_cache_entries > 0 ? 1 : 0;
      result.fmm_local_operator_finite = full_stats.plan.finite_outputs;
      result.memory_bytes += full_stats.plan.retained_plan_bytes;
      jcp_tree_stats.matvec_mode =
          options.bie_fmm_cartesian_order == 1
              ? "jcp2013_cartesian_full_fmm_order1"
              : "jcp2013_cartesian_full_fmm_order4";
      jcp_tree_stats.taylor_order = options.bie_fmm_cartesian_order;
      jcp_tree_stats.fmm_m2l_mode = "cartesian-full-experimental";
      jcp_tree_stats.jcp_direct_kernel = full_stats.direct_kernel;
      jcp_tree_stats.fmm_m2l_expansion_order =
          options.bie_fmm_cartesian_order;
      jcp_tree_stats.fmm_m2l_near_pairs =
          full_stats.plan.direct_panel_pairs;
      jcp_tree_stats.fmm_m2l_far_pairs = full_stats.plan.far_pairs;
      jcp_tree_stats.fmm_m2l_setup_ms =
          1000.0 * full_stats.plan.setup_seconds;
      jcp_tree_stats.fmm_m2l_finite_outputs =
          full_stats.plan.finite_outputs;
    }
    std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext>
        gmres_p4_context;
    // The p4 input split is performed for every Krylov matvec. Retain these
    // panel-sized buffers for the solve instead of allocating two vectors per
    // apply; the context consumes them synchronously.
    std::vector<double> gmres_p4_phi_workspace;
    std::vector<double> gmres_p4_phin_workspace;
    if (gmres_p4) {
      gmres_p4_context = ensure_jcp_fmm_m2l_context(
          panels, eps_ratio, kappa, jcp_tree.leaf_size, jcp_tree.opening_theta,
          jcp_tree.taylor_order, executor, &jcp_tree.fmm_m2l_context);
      titan::tabi::fmm_m2l::JcpLocalP4OperatorOptions p4_options;
      p4_options.near_integration = result.near_integration;
      p4_options.quad_buffer = result.quad_buffer;
      p4_options.refine_mode = options.fmm_local_refine;
      p4_options.refine_tolerance = options.fmm_local_refine_tol;
      p4_options.cache_mode = options.fmm_local_cache;
      p4_options.cache_budget_mb = static_cast<std::size_t>(
          std::max(0, options.fmm_local_cache_budget_mb));
      p4_options.fallback_mode = options.fmm_local_fallback;
      p4_options.plan_builder = options.fmm_local_plan_builder;
      p4_options.refine_probe_count = static_cast<std::size_t>(
          std::max(4, options.fmm_local_refine_probe_count));
      p4_options.refine_probe_mode = options.fmm_local_refine_probe_mode;
      p4_options.geometry_guard = options.fmm_local_geometry_guard;
      p4_options.geometry_theta = options.fmm_local_geometry_theta;
      const auto prepare_begin = std::chrono::steady_clock::now();
      const auto prepared = gmres_p4_context->prepare_local_p4_operator(
          panels, executor, p4_options);
      result.fmm_local_gmres_active = true;
      result.fmm_local_prepare_seconds =
          elapsed_ms(prepare_begin, std::chrono::steady_clock::now()) / 1000.0;
      result.fmm_local_plan_build_count = prepared.plan_build_count;
      result.fmm_local_cache_build_count = prepared.cache_build_count;
      result.fmm_local_operator_finite = prepared.finite_outputs;
      result.fmm_local_refine_probe_mode = prepared.refine_probe_mode;
      result.fmm_local_refine_probe_count = prepared.static_probe_count;
      result.fmm_local_solution_probe_used = prepared.solution_probe_used;
      result.fmm_local_linf_max_abs_tail = prepared.linf_max_abs_tail;
      result.fmm_local_linf_max_relative_tail = prepared.linf_max_relative_tail;
      result.fmm_local_linf_outlier_panels = prepared.linf_outlier_panels;
      result.fmm_local_geometry_guard = prepared.geometry_guard;
      result.fmm_local_geometry_theta = prepared.geometry_theta;
      result.fmm_local_geometry_guard_rejected_cells =
          prepared.geometry_guard_rejected_cells;
      result.fmm_local_geometry_guard_direct_fallback_pairs =
          prepared.geometry_guard_direct_fallback_pairs;
      result.fmm_local_geometry_gap_min = prepared.geometry_gap_min;
      result.fmm_local_geometry_ratio_max = prepared.geometry_ratio_max;
      result.memory_bytes += prepared.cache_bytes +
                             prepared.fallback_metadata_bytes +
                             prepared.plan_metric_cache_peak_bytes;
      jcp_tree_stats.matvec_mode = "jcp2013_fmm_m2l_gmres_p4";
      jcp_tree_stats.taylor_order = prepared.local_source_order;
      jcp_tree_stats.fmm_m2l_mode = "single-matvec-analytic";
      jcp_tree_stats.fmm_m2l_expansion_order = prepared.local_source_order;
      jcp_tree_stats.fmm_m2l_near_pairs =
          prepared.fallback_base_near_pairs + prepared.direct_fallback_pairs;
      jcp_tree_stats.fmm_m2l_far_pairs = prepared.far_pairs;
      jcp_tree_stats.fmm_m2l_setup_ms =
          1000.0 * result.fmm_local_prepare_seconds;
      jcp_tree_stats.fmm_m2l_finite_outputs = prepared.finite_outputs;
      if (!prepared.finite_outputs)
        throw std::runtime_error("gmres_p4_prepare_non_finite");
      gmres_p4_phi_workspace.assign(panels.size(), 0.0);
      gmres_p4_phin_workspace.assign(panels.size(), 0.0);
    }

    titan::tabi::fmm_kifmm::KifmmFullPanelPreparedOperator
        kifmm_prepared_operator;
    titan::tabi::fmm_kifmm::DualWorklistChannelHybridPreparedOperator
        channel_hybrid_prepared_operator;
    bool kifmm_memory_accounted = false;
    std::ofstream kifmm_iteration_invariants;
    std::ofstream kifmm_gmres_progress;
    std::ofstream kifmm_gmres_convergence_trace;
    double last_kifmm_apply_seconds = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t last_kifmm_worklist_hash{0};
    std::uint64_t last_kifmm_policy_hash{0};
    std::size_t last_kifmm_cache_entries{0};
    std::size_t last_kifmm_far_pairs{0};
    std::size_t last_kifmm_fallback_pairs{0};
    auto write_kifmm_gmres_phase = [&](const std::string &phase) {
      if (!kifmm_gmres_progress.is_open())
        return;
      kifmm_gmres_progress << phase << ','
                           << result.fmm_local_gmres_matvec_calls << ','
                           << matvec_calls << '\n';
      kifmm_gmres_progress.flush();
    };
    auto write_kifmm_convergence_trace = [&](const std::string &stage,
                                             int gmres_iter,
                                             double preconditioned_residual,
                                             double reported_true_residual) {
      if (!kifmm_gmres_convergence_trace.is_open())
        return;
      kifmm_gmres_convergence_trace
          << stage << ",unknown," << gmres_iter << ','
          << preconditioned_residual << ',' << reported_true_residual << ','
          << last_kifmm_apply_seconds << ',' << last_kifmm_worklist_hash << ','
          << last_kifmm_policy_hash << ',' << last_kifmm_cache_entries << ','
          << last_kifmm_far_pairs << ',' << last_kifmm_fallback_pairs << '\n';
      kifmm_gmres_convergence_trace.flush();
    };
    if (kifmm_gmres) {
      if (!coupled) {
        throw std::runtime_error(
            "kifmm_lite_experimental_requires_coupled_jcp");
      }
      titan::tabi::fmm_kifmm::KifmmRealPanelCloudConfig kifmm_config;
      kifmm_config.lite.order = options.kifmm_order;
      kifmm_config.lite.leaf_size =
          static_cast<int>(std::max(8, options.tree_leaf_size));
      kifmm_config.lite.opening_theta = options.tree_opening_theta;
      kifmm_config.lite.equiv_scale = options.kifmm_equiv_scale;
      kifmm_config.lite.check_scale = options.kifmm_check_scale;
      kifmm_config.lite.kappa = kappa;
      kifmm_config.lite.eps_ratio = eps_ratio;
      kifmm_config.lite.solver = options.kifmm_solver;
      kifmm_config.lite.condition_policy = "qr-rank-only";
      kifmm_config.lite.source_box_policy = "level-cube";
      kifmm_config.lite.max_condition_estimate = 1.0e12;
      kifmm_config.lite.max_equivalent_relative_residual = 1.0e-10;
      kifmm_config.lite.max_aggregate_relative_rms = 1.0e-6;
      kifmm_config.lite.max_channel_relative_rms = 5.0e-6;
      kifmm_config.lite.min_far_panel_fraction = 0.10;
      kifmm_config.lite.require_far_work = true;
      kifmm_config.lite.require_unit_normals = true;
      kifmm_config.near_integration = result.near_integration;
      kifmm_config.quad_buffer = result.quad_buffer;
      kifmm_config.full_apply_policy = options.kifmm_full_apply;
      const std::vector<std::size_t> sampled_targets =
          build_sampled_target_rows(panels.size(),
                                    options.matvec_audit_sample_count);
      kifmm_gmres_progress.open("pbe_tabi_kifmm_gmres_progress.csv",
                                std::ios::out | std::ios::trunc);
      if (kifmm_gmres_progress.is_open()) {
        kifmm_gmres_progress
            << "phase,fmm_local_gmres_matvec_calls,matvec_calls\n";
        kifmm_gmres_progress.flush();
      }
      kifmm_gmres_convergence_trace.open(
          "pbe_tabi_kifmm_gmres_convergence_trace.csv",
          std::ios::out | std::ios::trunc);
      if (kifmm_gmres_convergence_trace.is_open()) {
        kifmm_gmres_convergence_trace
            << "stage,fixture,iteration,preconditioned_residual,"
               "reported_true_residual,apply_seconds,worklist_hash,"
               "policy_hash,cache_entries,far_pairs,fallback_pairs\n";
        kifmm_gmres_convergence_trace.flush();
      }
      write_kifmm_gmres_phase("prepare_start");
      const auto prepare_begin = std::chrono::steady_clock::now();
      kifmm_prepared_operator =
          titan::tabi::fmm_kifmm::prepare_kifmm_lite_full_panel_operator(
              panels, sampled_targets, kifmm_config);
      result.fmm_local_gmres_active = true;
      result.fmm_local_prepare_seconds =
          elapsed_ms(prepare_begin, std::chrono::steady_clock::now()) / 1000.0;
      result.fmm_local_plan_build_count =
          kifmm_prepared_operator.prepare_build_count;
      result.fmm_local_cache_build_count =
          kifmm_prepared_operator.frozen_equivalent_system_cache ? 1u : 0u;
      result.fmm_local_operator_finite =
          kifmm_prepared_operator.frozen_worklist &&
          kifmm_prepared_operator.frozen_equivalent_system_cache;
      jcp_tree_stats.matvec_mode = "jcp2013_kifmm_lite_experimental_gmres";
      jcp_tree_stats.taylor_order = options.kifmm_order;
      jcp_tree_stats.fmm_m2l_mode = "kifmm-lite-experimental";
      jcp_tree_stats.fmm_m2l_expansion_order = options.kifmm_order;
      jcp_tree_stats.fmm_m2l_setup_ms =
          1000.0 * result.fmm_local_prepare_seconds;
      jcp_tree_stats.fmm_m2l_finite_outputs = result.fmm_local_operator_finite;
      if (!result.fmm_local_operator_finite) {
        throw std::runtime_error("kifmm_lite_gmres_prepare_incomplete");
      }
      kifmm_iteration_invariants.open("pbe_tabi_kifmm_iteration_invariants.csv",
                                      std::ios::out | std::ios::trunc);
      if (kifmm_iteration_invariants.is_open()) {
        kifmm_iteration_invariants
            << "matvec_call,worklist_hash,policy_hash,panel_geometry_hash,"
               "sampled_target_hash,cache_entries,far_panel_pairs,"
               "fallback_panel_pairs,sampled_invalid_pairs,"
               "nonproduction_fallback_pairs,apply_seconds,status\n";
        kifmm_iteration_invariants.flush();
      }
      write_kifmm_gmres_phase("prepare_done");
    }
    if (channel_hybrid_gmres) {
      if (!coupled) {
        throw std::runtime_error(
            "channel_hybrid_experimental_requires_coupled_jcp");
      }
      titan::tabi::fmm_kifmm::KifmmHierarchyJcpConfig hybrid_config;
      hybrid_config.hierarchy.order = options.kifmm_hierarchy_order;
      hybrid_config.hierarchy.max_depth = 6;
      hybrid_config.hierarchy.leaf_size =
          static_cast<int>(std::max(8, options.tree_leaf_size));
      hybrid_config.hierarchy.theta = options.kifmm_hierarchy_theta;
      hybrid_config.hierarchy.equiv_scale = 1.05;
      hybrid_config.hierarchy.upward_check_scale = 3.0;
      hybrid_config.hierarchy.local_equiv_scale = 3.0;
      hybrid_config.hierarchy.local_check_scale = 1.05;
      hybrid_config.hierarchy.condition_policy = "qr-rank-only";
      hybrid_config.hierarchy.source_box_policy = "strict-octree";
      hybrid_config.hierarchy.m2l_eval_policy =
          options.kifmm_hierarchy_m2l_eval_policy;
      hybrid_config.hierarchy.m2l_matrix_cache_policy =
          options.kifmm_hierarchy_matrix_cache_policy;
      hybrid_config.hierarchy.m2l_matrix_cache_limit_mb =
          options.kifmm_hierarchy_matrix_cache_limit_mb;
      hybrid_config.hierarchy.m2l_gemm_batch_policy =
          options.kifmm_hierarchy_gemm_batch_policy;
      hybrid_config.hierarchy.m2l_parallel_policy =
          options.kifmm_hierarchy_m2l_parallel_policy;
      hybrid_config.hierarchy.m2l_parallel_workers =
          options.kifmm_hierarchy_m2l_parallel_workers;
      hybrid_config.hierarchy.tree_stop_policy =
          options.kifmm_hierarchy_tree_stop_policy;
      hybrid_config.hierarchy.tree_panel_radius_factor =
          options.kifmm_hierarchy_tree_panel_radius_factor;
      hybrid_config.hierarchy.tree_occupancy_stop =
          options.kifmm_hierarchy_tree_occupancy_stop;
      hybrid_config.hierarchy.jcp_channel_policy = "k123-only";
      hybrid_config.kappa = kappa;
      hybrid_config.eps_ratio = eps_ratio;
      hybrid_config.real_panel_guard = options.kifmm_hierarchy_real_panel_guard;
      hybrid_config.real_panel_guard_margin_factor =
          options.kifmm_hierarchy_guard_margin_factor;
      hybrid_config.hier_debug_mode =
          "dual-worklist-channel-hybrid-k123-kifmm-k4-cartesian";
      hybrid_config.build_dense_oracle = false;
      hybrid_config.include_direct_fallback = true;
      hybrid_config.aggregate_relative_rms_gate = 1.0e-6;
      hybrid_config.max_channel_relative_rms_gate = 5.0e-6;
      const std::vector<std::size_t> sampled_targets =
          build_sampled_target_rows(panels.size(),
                                    options.matvec_audit_sample_count);
      kifmm_gmres_progress.open("pbe_tabi_channel_hybrid_gmres_progress.csv",
                                std::ios::out | std::ios::trunc);
      if (kifmm_gmres_progress.is_open()) {
        kifmm_gmres_progress
            << "phase,fmm_local_gmres_matvec_calls,matvec_calls\n";
        kifmm_gmres_progress.flush();
      }
      kifmm_gmres_convergence_trace.open(
          "pbe_tabi_channel_hybrid_gmres_convergence_trace.csv",
          std::ios::out | std::ios::trunc);
      if (kifmm_gmres_convergence_trace.is_open()) {
        kifmm_gmres_convergence_trace
            << "stage,fixture,iteration,preconditioned_residual,"
               "reported_true_residual,apply_seconds,worklist_hash,"
               "policy_hash,cache_entries,far_pairs,fallback_pairs\n";
        kifmm_gmres_convergence_trace.flush();
      }
      write_kifmm_gmres_phase("prepare_start");
      const auto prepare_begin = std::chrono::steady_clock::now();
      channel_hybrid_prepared_operator =
          titan::tabi::fmm_kifmm::prepare_dual_worklist_channel_hybrid_operator(
              panels, sampled_targets, hybrid_config);
      result.fmm_local_gmres_active = true;
      result.fmm_local_prepare_seconds =
          elapsed_ms(prepare_begin, std::chrono::steady_clock::now()) / 1000.0;
      result.fmm_local_plan_build_count =
          channel_hybrid_prepared_operator.prepare_build_count;
      result.fmm_local_cache_build_count =
          channel_hybrid_prepared_operator.frozen_equivalent_system_cache ? 1u
                                                                          : 0u;
      result.fmm_local_operator_finite =
          channel_hybrid_prepared_operator.source_panels == panels.size() &&
          channel_hybrid_prepared_operator.panel_geometry_hash != 0 &&
          channel_hybrid_prepared_operator.sampled_target_hash != 0 &&
          channel_hybrid_prepared_operator.frozen_worklist &&
          channel_hybrid_prepared_operator.frozen_equivalent_system_cache &&
          channel_hybrid_prepared_operator.prepared_rebuild_status ==
              "frozen_prepared_apply";
      jcp_tree_stats.matvec_mode =
          "jcp2013_kifmm_channel_hybrid_experimental_gmres";
      jcp_tree_stats.taylor_order = options.kifmm_hierarchy_order;
      jcp_tree_stats.fmm_m2l_mode = "kifmm-channel-hybrid-experimental";
      jcp_tree_stats.fmm_m2l_expansion_order = options.kifmm_hierarchy_order;
      jcp_tree_stats.fmm_m2l_setup_ms =
          1000.0 * result.fmm_local_prepare_seconds;
      jcp_tree_stats.fmm_m2l_finite_outputs = result.fmm_local_operator_finite;
      if (!result.fmm_local_operator_finite) {
        throw std::runtime_error("channel_hybrid_gmres_prepare_incomplete");
      }
      kifmm_iteration_invariants.open(
          "pbe_tabi_channel_hybrid_iteration_invariants.csv",
          std::ios::out | std::ios::trunc);
      if (kifmm_iteration_invariants.is_open()) {
        kifmm_iteration_invariants
            << "matvec_call,worklist_hash,policy_hash,panel_geometry_hash,"
               "sampled_target_hash,cache_entries,far_panel_pairs,"
               "fallback_panel_pairs,sampled_invalid_pairs,"
               "nonproduction_fallback_pairs,apply_seconds,status,"
               "preconditioner_graph_source,frozen_worklist,"
               "frozen_equivalent_system_cache,prepared_rebuild_status,"
               "apply_source_build_seconds,apply_hierarchy_seconds,"
               "apply_guard_audit_seconds,apply_production_fallback_seconds,"
               "apply_cartesian_k4_seconds,apply_dense_oracle_seconds,"
               "apply_finalize_seconds,apply_unattributed_seconds,"
               "apply_rss_start_bytes,apply_rss_after_source_build_bytes,"
               "apply_rss_after_hierarchy_bytes,"
               "apply_rss_after_guard_audit_bytes,"
               "apply_rss_after_fallback_bytes,"
               "apply_rss_after_cartesian_k4_bytes,"
               "apply_rss_after_dense_oracle_bytes,"
               "apply_rss_after_finalize_bytes,"
               "apply_peak_working_set_start_bytes,"
               "apply_peak_working_set_after_source_build_bytes,"
               "apply_peak_working_set_after_hierarchy_bytes,"
               "apply_peak_working_set_after_guard_audit_bytes,"
               "apply_peak_working_set_after_fallback_bytes,"
               "apply_peak_working_set_after_cartesian_k4_bytes,"
               "apply_peak_working_set_after_dense_oracle_bytes,"
               "apply_peak_working_set_after_finalize_bytes,"
               "apply_peak_working_set_delta_bytes,"
               "runtime_memory_attribution_status,"
               "hierarchy_p2m_seconds,hierarchy_p2m_rhs_seconds,"
               "hierarchy_p2m_solve_seconds,hierarchy_m2m_seconds,"
               "hierarchy_m2l_seconds,hierarchy_local_solve_seconds,"
               "hierarchy_l2l_seconds,hierarchy_l2p_seconds,"
               "hierarchy_direct_fallback_seconds,"
               "hierarchy_dense_oracle_seconds,"
               "hierarchy_rss_start_bytes,hierarchy_rss_after_p2m_bytes,"
               "hierarchy_rss_after_m2m_bytes,hierarchy_rss_after_m2l_bytes,"
               "hierarchy_rss_after_l2l_bytes,hierarchy_rss_after_l2p_bytes,"
               "hierarchy_rss_after_finalize_bytes,"
               "hierarchy_peak_start_bytes,hierarchy_peak_after_p2m_bytes,"
               "hierarchy_peak_after_m2m_bytes,"
               "hierarchy_peak_after_m2l_bytes,"
               "hierarchy_peak_after_l2l_bytes,"
               "hierarchy_peak_after_l2p_bytes,"
               "hierarchy_peak_after_finalize_bytes,"
               "hierarchy_peak_delta_bytes,"
               "k123_upward_equivalent_estimated_bytes,"
               "k123_local_rhs_estimated_bytes,"
               "k123_dense_local_rhs_estimated_bytes,"
               "k123_local_equivalent_estimated_bytes,"
               "k123_rows_estimated_bytes,"
               "k123_equivalent_system_cache_estimated_bytes,"
               "k123_m2l_matrix_cache_limit_bytes,"
               "k123_m2l_matrix_cache_evictions,"
               "k123_m2l_matrix_cache_estimated_bytes,"
               "k123_m2l_matrix_cache_peak_estimated_bytes,"
               "k123_m2l_matrix_cache_total_built_bytes,"
               "k123_total_estimated_bytes,"
               "hierarchy_runtime_memory_attribution_status\n";
        kifmm_iteration_invariants.flush();
      }
      write_kifmm_gmres_phase("prepare_done");
    }

    auto apply_cartesian_full_operator = [&](const std::vector<double> &x,
                                              std::vector<double> &out,
                                              bool count_gmres_matvec) {
      if (!cartesian_full_operator)
        throw std::logic_error("cartesian_full_context_not_prepared");
      cartesian_full_operator->apply(x, lhs_phi, lhs_phin, coupled, executor,
                                     out, count_gmres_matvec);
      const auto &full_stats = cartesian_full_operator->stats();
      if (full_stats.plan.tree_hash != cartesian_full_tree_hash ||
          full_stats.plan.worklist_hash != cartesian_full_worklist_hash) {
        throw std::runtime_error(
            "cartesian_full_fmm_lifecycle_hash_drift");
      }
      result.fmm_local_operator_finite =
          result.fmm_local_operator_finite && full_stats.finite_outputs;
      result.bie_cartesian_full_fmm_workspace_bytes =
          std::max(result.bie_cartesian_full_fmm_workspace_bytes,
                   full_stats.last_apply.retained_workspace_bytes);
      result.bie_cartesian_full_fmm_direct_panel_pairs_expected =
          full_stats.last_apply.expected_direct_panel_pairs;
      result.bie_cartesian_full_fmm_direct_panel_pairs_applied_last =
          full_stats.last_apply.direct_panel_pairs;
      result.bie_cartesian_full_fmm_direct_panel_pairs_applied_total =
          full_stats.total_apply.direct_panel_pairs;
      if (!cartesian_full_workspace_accounted) {
        result.memory_bytes += full_stats.last_apply.retained_workspace_bytes;
        cartesian_full_workspace_accounted = true;
      }
      if (count_gmres_matvec) {
        result.fmm_local_gmres_matvec_calls = full_stats.apply_calls;
        result.fmm_local_gmres_apply_seconds_total =
            full_stats.total_apply.apply_seconds;
        result.fmm_local_gmres_apply_seconds_mean =
            full_stats.apply_calls > 0
                ? full_stats.total_apply.apply_seconds /
                      static_cast<double>(full_stats.apply_calls)
                : 0.0;
        result.fmm_local_moment_seconds_total =
            full_stats.total_apply.p2m_seconds +
            full_stats.total_apply.m2m_seconds;
        result.fmm_local_m2l_seconds_total =
            full_stats.total_apply.m2l_seconds;
        result.fmm_local_fallback_seconds_total =
            full_stats.total_apply.direct_seconds;
        result.fmm_local_l2p_seconds_total =
            full_stats.total_apply.l2l_seconds +
            full_stats.total_apply.l2p_seconds;
        result.bie_cartesian_full_fmm_p2m_seconds =
            full_stats.total_apply.p2m_seconds;
        result.bie_cartesian_full_fmm_m2m_seconds =
            full_stats.total_apply.m2m_seconds;
        result.bie_cartesian_full_fmm_m2l_seconds =
            full_stats.total_apply.m2l_seconds;
        result.bie_cartesian_full_fmm_l2l_seconds =
            full_stats.total_apply.l2l_seconds;
        result.bie_cartesian_full_fmm_l2p_seconds =
            full_stats.total_apply.l2p_seconds;
        result.bie_cartesian_full_fmm_direct_seconds =
            full_stats.total_apply.direct_seconds;
        jcp_tree_stats.fmm_m2l_apply_ms =
            1000.0 * full_stats.total_apply.apply_seconds;
        jcp_tree_stats.fmm_m2l_finite_outputs = full_stats.finite_outputs;
      }
    };

    auto apply_p4_operator = [&](const std::vector<double> &x,
                                 std::vector<double> &out,
                                 bool count_gmres_matvec) {
      if (!gmres_p4_context)
        throw std::logic_error("gmres_p4_context_not_prepared");
      const std::size_t n = panels.size();
      if (x.size() < (coupled ? 2 * n : n))
        throw std::invalid_argument("gmres_p4_incomplete_unknown_vector");
      if (gmres_p4_phi_workspace.size() != n ||
          gmres_p4_phin_workspace.size() != n)
        throw std::logic_error("gmres_p4_workspace_shape_mismatch");
      std::vector<double> &phi = gmres_p4_phi_workspace;
      std::vector<double> &phin = gmres_p4_phin_workspace;
      for (std::size_t panel = 0; panel < n; ++panel) {
        phi[panel] = x[panel];
        phin[panel] = coupled ? x[n + panel] : 0.0;
      }
      const auto applied = gmres_p4_context->apply_prepared_local_p4_operator(
          phi, phin, executor, coupled);
      if (!applied.finite_outputs)
        throw std::runtime_error("gmres_p4_apply_non_finite");
      titan::tabi::fmm_m2l::assemble_jcp_local_p4_system_matvec(
          applied, x, lhs_phi, lhs_phin, coupled, out);
      result.fmm_local_operator_finite =
          result.fmm_local_operator_finite && applied.finite_outputs;
      result.fmm_local_plan_build_count = applied.plan_build_count;
      result.fmm_local_cache_build_count = applied.cache_build_count;
      result.fmm_local_refine_probe_mode = applied.refine_probe_mode;
      result.fmm_local_refine_probe_count = applied.static_probe_count;
      result.fmm_local_solution_probe_used = applied.solution_probe_used;
      result.fmm_local_linf_max_abs_tail = applied.linf_max_abs_tail;
      result.fmm_local_linf_max_relative_tail = applied.linf_max_relative_tail;
      result.fmm_local_linf_outlier_panels = applied.linf_outlier_panels;
      result.fmm_local_geometry_guard = applied.geometry_guard;
      result.fmm_local_geometry_theta = applied.geometry_theta;
      result.fmm_local_geometry_guard_rejected_cells =
          applied.geometry_guard_rejected_cells;
      result.fmm_local_geometry_guard_direct_fallback_pairs =
          applied.geometry_guard_direct_fallback_pairs;
      result.fmm_local_geometry_gap_min = applied.geometry_gap_min;
      result.fmm_local_geometry_ratio_max = applied.geometry_ratio_max;
      if (count_gmres_matvec) {
        ++result.fmm_local_gmres_matvec_calls;
        result.fmm_local_gmres_apply_seconds_total += applied.apply_seconds;
        result.fmm_local_moment_seconds_total += applied.moment_seconds;
        result.fmm_local_m2l_seconds_total += applied.m2l_seconds;
        result.fmm_local_fallback_seconds_total +=
            applied.direct_fallback_seconds;
        result.fmm_local_l2p_seconds_total += applied.l2p_seconds;
        result.fmm_local_gmres_apply_seconds_mean =
            result.fmm_local_gmres_apply_seconds_total /
            static_cast<double>(result.fmm_local_gmres_matvec_calls);
      }
    };

    auto apply_kifmm_operator = [&](const std::vector<double> &x,
                                    std::vector<double> &out,
                                    bool count_gmres_matvec) {
      if (!kifmm_prepared_operator.state) {
        throw std::logic_error("kifmm_lite_gmres_context_not_prepared");
      }
      const std::size_t n = panels.size();
      if (x.size() < 2 * n) {
        throw std::invalid_argument("kifmm_lite_incomplete_unknown_vector");
      }
      std::string matvec_phase;
      if (count_gmres_matvec) {
        matvec_phase =
            "matvec_" + std::to_string(result.fmm_local_gmres_matvec_calls + 1);
        write_kifmm_gmres_phase(matvec_phase + "_start");
      }
      const auto applied =
          titan::tabi::fmm_kifmm::apply_prepared_kifmm_lite_full_panel_operator(
              kifmm_prepared_operator, x, &executor, false);
      if (!applied.finite || applied.status != "pass") {
        throw std::runtime_error("kifmm_lite_gmres_apply_" + applied.status);
      }
      if (applied.panel_geometry_hash !=
              kifmm_prepared_operator.panel_geometry_hash ||
          applied.sampled_target_hash !=
              kifmm_prepared_operator.sampled_target_hash ||
          applied.policy_hash != kifmm_prepared_operator.policy_hash ||
          applied.lite.worklist_hash != kifmm_prepared_operator.worklist_hash) {
        throw std::runtime_error("kifmm_lite_gmres_lifecycle_hash_drift");
      }
      if (applied.equivalent_factor_cache_misses != 0) {
        throw std::runtime_error("kifmm_lite_gmres_cache_rebuild");
      }
      if (applied.sampled_invalid_pairs != 0 ||
          applied.nonproduction_fallback_pairs != 0 ||
          applied.used_nonproduction_fallback) {
        throw std::runtime_error("kifmm_lite_gmres_fallback_gap");
      }
      const std::size_t expected_pairs = n * n;
      if (applied.full_pair_coverage != expected_pairs) {
        throw std::runtime_error("kifmm_lite_gmres_full_coverage_gap");
      }
      titan::tabi::fmm_kifmm::assemble_kifmm_lite_tabi_system_matvec(
          applied, x, lhs_phi, lhs_phin, out);
      result.fmm_local_operator_finite =
          result.fmm_local_operator_finite && applied.finite;
      result.fmm_local_plan_build_count =
          kifmm_prepared_operator.prepare_build_count;
      result.fmm_local_cache_build_count =
          kifmm_prepared_operator.frozen_equivalent_system_cache ? 1u : 0u;
      jcp_tree_stats.fmm_m2l_near_pairs =
          applied.lite.direct_fallback_panel_pairs;
      jcp_tree_stats.fmm_m2l_far_pairs = applied.lite.accepted_far_panel_pairs;
      if (!kifmm_memory_accounted) {
        result.memory_bytes +=
            applied.source_tree_bytes + applied.target_tree_bytes +
            applied.worklist_bytes + applied.equivalent_charge_storage_bytes +
            applied.cached_channel_object_estimated_bytes;
        kifmm_memory_accounted = true;
      }
      if (count_gmres_matvec) {
        ++result.fmm_local_gmres_matvec_calls;
        last_kifmm_apply_seconds = applied.apply_seconds;
        last_kifmm_worklist_hash = applied.lite.worklist_hash;
        last_kifmm_policy_hash = applied.policy_hash;
        last_kifmm_cache_entries = applied.equivalent_system_cache_entries;
        last_kifmm_far_pairs = applied.lite.accepted_far_panel_pairs;
        last_kifmm_fallback_pairs = applied.lite.direct_fallback_panel_pairs;
        result.fmm_local_gmres_apply_seconds_total += applied.apply_seconds;
        result.fmm_local_m2l_seconds_total +=
            applied.equivalent_build_solve_seconds +
            applied.full_far_eval_seconds;
        result.fmm_local_fallback_seconds_total +=
            applied.full_fallback_eval_seconds;
        result.fmm_local_gmres_apply_seconds_mean =
            result.fmm_local_gmres_apply_seconds_total /
            static_cast<double>(result.fmm_local_gmres_matvec_calls);
        if (kifmm_iteration_invariants.is_open()) {
          kifmm_iteration_invariants
              << result.fmm_local_gmres_matvec_calls << ','
              << applied.lite.worklist_hash << ',' << applied.policy_hash << ','
              << applied.panel_geometry_hash << ','
              << applied.sampled_target_hash << ','
              << applied.equivalent_system_cache_entries << ','
              << applied.lite.accepted_far_panel_pairs << ','
              << applied.lite.direct_fallback_panel_pairs << ','
              << applied.sampled_invalid_pairs << ','
              << applied.nonproduction_fallback_pairs << ','
              << applied.apply_seconds << ',' << applied.status << '\n';
          kifmm_iteration_invariants.flush();
        }
        write_kifmm_gmres_phase(matvec_phase + "_done");
      }
    };

    auto apply_channel_hybrid_operator = [&](const std::vector<double> &x,
                                             std::vector<double> &out,
                                             bool count_gmres_matvec) {
      const std::size_t n = panels.size();
      if (x.size() < 2 * n) {
        throw std::invalid_argument("channel_hybrid_incomplete_unknown_vector");
      }
      std::string matvec_phase;
      if (count_gmres_matvec) {
        matvec_phase =
            "matvec_" + std::to_string(result.fmm_local_gmres_matvec_calls + 1);
        write_kifmm_gmres_phase(matvec_phase + "_start");
      }
      auto applied = titan::tabi::fmm_kifmm::
          apply_prepared_dual_worklist_channel_hybrid_operator(
              channel_hybrid_prepared_operator, x, false);
      if (applied.execution_status != "pass" || applied.status != "ok") {
        throw std::runtime_error("channel_hybrid_gmres_apply_" +
                                 applied.status);
      }
      if (applied.panel_geometry_hash !=
              channel_hybrid_prepared_operator.panel_geometry_hash ||
          applied.sampled_target_hash !=
              channel_hybrid_prepared_operator.sampled_target_hash ||
          applied.worklist_hash !=
              channel_hybrid_prepared_operator.worklist_hash ||
          applied.policy_hash != channel_hybrid_prepared_operator.policy_hash ||
          applied.tree_hash != channel_hybrid_prepared_operator.tree_hash) {
        throw std::runtime_error("channel_hybrid_gmres_lifecycle_hash_drift");
      }
      if (!channel_hybrid_prepared_operator.frozen_worklist ||
          !channel_hybrid_prepared_operator.frozen_equivalent_system_cache ||
          channel_hybrid_prepared_operator.prepared_rebuild_status !=
              "frozen_prepared_apply") {
        throw std::runtime_error("channel_hybrid_gmres_prepared_freeze_gap");
      }
      if (applied.production_invalid_pairs != 0 ||
          applied.nonproduction_fallback_pairs != 0) {
        throw std::runtime_error("channel_hybrid_gmres_fallback_gap");
      }
      const std::uint64_t expected_pairs =
          static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n);
      if (applied.exact_pair_coverage != expected_pairs) {
        throw std::runtime_error("channel_hybrid_gmres_full_coverage_gap");
      }
      titan::tabi::fmm_kifmm::
          assemble_dual_worklist_channel_hybrid_tabi_system_matvec(
              applied, x, lhs_phi, lhs_phin, out);
      result.fmm_local_operator_finite = result.fmm_local_operator_finite &&
                                         applied.execution_status == "pass";
      result.fmm_local_plan_build_count =
          channel_hybrid_prepared_operator.prepare_build_count;
      result.fmm_local_cache_build_count =
          channel_hybrid_prepared_operator.frozen_equivalent_system_cache ? 1u
                                                                          : 0u;
      jcp_tree_stats.fmm_m2l_near_pairs = applied.direct_panel_pairs;
      jcp_tree_stats.fmm_m2l_far_pairs = applied.far_panel_pairs;
      if (!kifmm_memory_accounted) {
        result.memory_bytes += applied.m2l_matrix_cache_estimated_bytes;
        kifmm_memory_accounted = true;
      }
      if (count_gmres_matvec) {
        ++result.fmm_local_gmres_matvec_calls;
        last_kifmm_apply_seconds = applied.total_seconds;
        last_kifmm_worklist_hash = applied.worklist_hash;
        last_kifmm_policy_hash = applied.policy_hash;
        last_kifmm_cache_entries =
            static_cast<std::size_t>(applied.m2l_matrix_cache_entries);
        last_kifmm_far_pairs =
            static_cast<std::size_t>(applied.far_panel_pairs);
        last_kifmm_fallback_pairs =
            static_cast<std::size_t>(applied.direct_panel_pairs);
        result.fmm_local_gmres_apply_seconds_total += applied.total_seconds;
        result.fmm_local_m2l_seconds_total += applied.hierarchy.m2l_seconds;
        result.fmm_local_fallback_seconds_total +=
            applied.production_fallback_seconds +
            applied.cartesian_k4_total_seconds;
        result.fmm_local_gmres_apply_seconds_mean =
            result.fmm_local_gmres_apply_seconds_total /
            static_cast<double>(result.fmm_local_gmres_matvec_calls);
        if (kifmm_iteration_invariants.is_open()) {
          kifmm_iteration_invariants
              << result.fmm_local_gmres_matvec_calls << ','
              << applied.worklist_hash << ',' << applied.policy_hash << ','
              << applied.panel_geometry_hash << ','
              << applied.sampled_target_hash << ','
              << applied.m2l_matrix_cache_entries << ','
              << applied.far_panel_pairs << ',' << applied.direct_panel_pairs
              << ',' << applied.production_invalid_pairs << ','
              << applied.nonproduction_fallback_pairs << ','
              << applied.total_seconds << ',' << applied.status << ','
              << kifmm_ilu0_graph_source << ','
              << (channel_hybrid_prepared_operator.frozen_worklist ? "true"
                                                                   : "false")
              << ','
              << (channel_hybrid_prepared_operator
                          .frozen_equivalent_system_cache
                      ? "true"
                      : "false")
              << ',' << channel_hybrid_prepared_operator.prepared_rebuild_status
              << ',' << applied.apply_source_build_seconds << ','
              << applied.apply_hierarchy_seconds << ','
              << applied.apply_guard_audit_seconds << ','
              << applied.apply_production_fallback_seconds << ','
              << applied.apply_cartesian_k4_seconds << ','
              << applied.apply_dense_oracle_seconds << ','
              << applied.apply_finalize_seconds << ','
              << applied.apply_unattributed_seconds << ','
              << applied.apply_rss_start_bytes << ','
              << applied.apply_rss_after_source_build_bytes << ','
              << applied.apply_rss_after_hierarchy_bytes << ','
              << applied.apply_rss_after_guard_audit_bytes << ','
              << applied.apply_rss_after_fallback_bytes << ','
              << applied.apply_rss_after_cartesian_k4_bytes << ','
              << applied.apply_rss_after_dense_oracle_bytes << ','
              << applied.apply_rss_after_finalize_bytes << ','
              << applied.apply_peak_working_set_start_bytes << ','
              << applied.apply_peak_working_set_after_source_build_bytes << ','
              << applied.apply_peak_working_set_after_hierarchy_bytes << ','
              << applied.apply_peak_working_set_after_guard_audit_bytes << ','
              << applied.apply_peak_working_set_after_fallback_bytes << ','
              << applied.apply_peak_working_set_after_cartesian_k4_bytes << ','
              << applied.apply_peak_working_set_after_dense_oracle_bytes << ','
              << applied.apply_peak_working_set_after_finalize_bytes << ','
              << applied.apply_peak_working_set_delta_bytes << ','
              << applied.runtime_memory_attribution_status << ','
              << applied.hierarchy.p2m_seconds << ','
              << applied.hierarchy.p2m_rhs_seconds << ','
              << applied.hierarchy.p2m_solve_seconds << ','
              << applied.hierarchy.m2m_seconds << ','
              << applied.hierarchy.m2l_seconds << ','
              << applied.hierarchy.local_solve_seconds << ','
              << applied.hierarchy.l2l_seconds << ','
              << applied.hierarchy.l2p_seconds << ','
              << applied.hierarchy.direct_fallback_seconds << ','
              << applied.hierarchy.dense_oracle_seconds << ','
              << applied.hierarchy.hierarchy_rss_start_bytes << ','
              << applied.hierarchy.hierarchy_rss_after_p2m_bytes << ','
              << applied.hierarchy.hierarchy_rss_after_m2m_bytes << ','
              << applied.hierarchy.hierarchy_rss_after_m2l_bytes << ','
              << applied.hierarchy.hierarchy_rss_after_l2l_bytes << ','
              << applied.hierarchy.hierarchy_rss_after_l2p_bytes << ','
              << applied.hierarchy.hierarchy_rss_after_finalize_bytes << ','
              << applied.hierarchy.hierarchy_peak_start_bytes << ','
              << applied.hierarchy.hierarchy_peak_after_p2m_bytes << ','
              << applied.hierarchy.hierarchy_peak_after_m2m_bytes << ','
              << applied.hierarchy.hierarchy_peak_after_m2l_bytes << ','
              << applied.hierarchy.hierarchy_peak_after_l2l_bytes << ','
              << applied.hierarchy.hierarchy_peak_after_l2p_bytes << ','
              << applied.hierarchy.hierarchy_peak_after_finalize_bytes << ','
              << applied.hierarchy.hierarchy_peak_delta_bytes << ','
              << applied.hierarchy.k123_upward_equivalent_estimated_bytes << ','
              << applied.hierarchy.k123_local_rhs_estimated_bytes << ','
              << applied.hierarchy.k123_dense_local_rhs_estimated_bytes << ','
              << applied.hierarchy.k123_local_equivalent_estimated_bytes << ','
              << applied.hierarchy.k123_rows_estimated_bytes << ','
              << applied.hierarchy.k123_equivalent_system_cache_estimated_bytes
              << ',' << applied.hierarchy.m2l_matrix_cache_limit_bytes << ','
              << applied.hierarchy.m2l_matrix_cache_evictions << ','
              << applied.hierarchy.m2l_matrix_cache_estimated_bytes << ','
              << applied.hierarchy.m2l_matrix_cache_peak_estimated_bytes << ','
              << applied.hierarchy.m2l_matrix_cache_total_built_bytes << ','
              << applied.hierarchy.k123_total_estimated_bytes << ','
              << applied.hierarchy.hierarchy_runtime_memory_attribution_status
              << '\n';
          kifmm_iteration_invariants.flush();
        }
        write_kifmm_gmres_phase(matvec_phase + "_done");
      }
    };

    auto matvec = [&](const std::vector<double> &x, std::vector<double> &out) {
      const auto begin = std::chrono::steady_clock::now();
      if (cartesian_full_gmres) {
        apply_cartesian_full_operator(x, out, true);
      } else if (channel_hybrid_gmres) {
        apply_channel_hybrid_operator(x, out, true);
      } else if (kifmm_gmres) {
        apply_kifmm_operator(x, out, true);
      } else if (gmres_p4) {
        apply_p4_operator(x, out, true);
      } else if (experimental_treecode) {
        apply_jcp_treecode_operator(panels, x, eps_ratio, kappa, lhs_phi,
                                    lhs_phin, coupled, jcp_tree,
                                    result.near_integration, result.quad_buffer,
                                    executor, out, &jcp_tree_stats);
      } else if (coupled) {
        apply_jcp_linear_pb_operator(panels, x, eps_ratio, kappa, lhs_phi,
                                     lhs_phin, result.near_integration,
                                     result.quad_buffer, executor, out,
                                     &jcp_tree_stats);
      } else {
        apply_jcp_poisson_operator(panels, x, eps_ratio, lhs_phi,
                                   result.near_integration, result.quad_buffer,
                                   executor, out, &jcp_tree_stats);
      }
      matvec_ms_total += elapsed_ms(begin, std::chrono::steady_clock::now());
      ++matvec_calls;
    };
    // Pre-GMRES diagnostics must exercise the prepared operator without
    // contaminating Krylov/apply counters or their timing totals.
    auto audit_apply = [&](const std::vector<double> &x,
                           std::vector<double> &out) {
      if (cartesian_full_gmres) {
        apply_cartesian_full_operator(x, out, false);
      } else if (channel_hybrid_gmres) {
        apply_channel_hybrid_operator(x, out, false);
      } else if (kifmm_gmres) {
        apply_kifmm_operator(x, out, false);
      } else if (gmres_p4) {
        apply_p4_operator(x, out, false);
      } else if (experimental_treecode) {
        apply_jcp_treecode_operator(panels, x, eps_ratio, kappa, lhs_phi,
                                    lhs_phin, coupled, jcp_tree,
                                    result.near_integration, result.quad_buffer,
                                    executor, out, nullptr);
      } else if (coupled) {
        apply_jcp_linear_pb_operator(
            panels, x, eps_ratio, kappa, lhs_phi, lhs_phin,
            result.near_integration, result.quad_buffer, executor, out,
            nullptr);
      } else {
        apply_jcp_poisson_operator(panels, x, eps_ratio, lhs_phi,
                                   result.near_integration, result.quad_buffer,
                                   executor, out, nullptr);
      }
    };
    std::vector<std::size_t> direct_order(panels.size(), 0);
    std::iota(direct_order.begin(), direct_order.end(), std::size_t{0});
    auto build_audit_rows = [](std::size_t row_count, std::string mode,
                               int requested) {
      std::vector<std::size_t> rows;
      if (row_count == 0)
        return rows;
      const bool full = mode == "full-small" && row_count <= 4096u;
      const std::size_t m =
          full ? row_count
               : std::min<std::size_t>(row_count,
                                       static_cast<std::size_t>(
                                           std::clamp(requested, 128, 4096)));
      rows.reserve(m);
      if (full) {
        for (std::size_t row = 0; row < row_count; ++row)
          rows.push_back(row);
        return rows;
      }
      std::size_t previous = std::numeric_limits<std::size_t>::max();
      for (std::size_t k = 0; k < m; ++k) {
        std::size_t row =
            ((2u * k + 1u) * row_count) / std::max<std::size_t>(1, 2u * m);
        row = std::min(row, row_count - 1u);
        if (row != previous) {
          rows.push_back(row);
          previous = row;
        }
      }
      if (rows.empty())
        rows.push_back(0);
      return rows;
    };
    auto build_spatial_audit_targets =
        [](const std::vector<Panel> &audit_panels,
           std::size_t requested) -> std::vector<std::size_t> {
      std::vector<std::size_t> targets;
      if (audit_panels.empty() || requested == 0)
        return targets;
      double min_x = audit_panels.front().x;
      double min_y = audit_panels.front().y;
      double min_z = audit_panels.front().z;
      double max_x = min_x;
      double max_y = min_y;
      double max_z = min_z;
      for (const Panel &panel : audit_panels) {
        min_x = std::min(min_x, panel.x);
        min_y = std::min(min_y, panel.y);
        min_z = std::min(min_z, panel.z);
        max_x = std::max(max_x, panel.x);
        max_y = std::max(max_y, panel.y);
        max_z = std::max(max_z, panel.z);
      }
      const double span_x = std::max(1.0e-12, max_x - min_x);
      const double span_y = std::max(1.0e-12, max_y - min_y);
      const double span_z = std::max(1.0e-12, max_z - min_z);
      auto quantize = [](double value, double lo, double span) {
        const double unit = std::clamp((value - lo) / span, 0.0, 1.0);
        return static_cast<std::uint32_t>(unit * 1023.0 + 0.5);
      };
      auto morton_key = [](std::uint32_t x, std::uint32_t y,
                           std::uint32_t z) {
        std::uint64_t key = 0;
        for (unsigned bit = 0; bit < 10; ++bit) {
          key |= static_cast<std::uint64_t>((x >> bit) & 1u) << (3u * bit);
          key |= static_cast<std::uint64_t>((y >> bit) & 1u)
                 << (3u * bit + 1u);
          key |= static_cast<std::uint64_t>((z >> bit) & 1u)
                 << (3u * bit + 2u);
        }
        return key;
      };
      std::vector<std::pair<std::uint64_t, std::size_t>> spatial_order;
      spatial_order.reserve(audit_panels.size());
      for (std::size_t i = 0; i < audit_panels.size(); ++i) {
        const Panel &panel = audit_panels[i];
        spatial_order.emplace_back(
            morton_key(quantize(panel.x, min_x, span_x),
                       quantize(panel.y, min_y, span_y),
                       quantize(panel.z, min_z, span_z)),
            i);
      }
      std::sort(spatial_order.begin(), spatial_order.end());
      const std::size_t count =
          std::min<std::size_t>(requested, spatial_order.size());
      targets.reserve(count);
      for (std::size_t k = 0; k < count; ++k) {
        const std::size_t position =
            ((2u * k + 1u) * spatial_order.size()) /
            std::max<std::size_t>(1, 2u * count);
        targets.push_back(
            spatial_order[std::min(position, spatial_order.size() - 1u)]
                .second);
      }
      return targets;
    };
    SampledTrueResidualCallback sampled_true_residual;
    if (!experimental_treecode || kifmm_any_gmres) {
      sampled_true_residual =
          [&](const std::vector<double> &x,
              const std::vector<std::size_t> &rows) -> SampledResidualResult {
        SampledResidualResult sample;
        sample.rows = rows.size();
        if (rows.empty())
          return sample;
        const std::size_t unknown_count = rhs.size();
        const std::size_t panel_count = panels.size();
        const std::size_t workers = solver_workers(executor, rows.size());
        std::vector<double> partial_residual_sq(workers, 0.0);
        std::vector<double> partial_b_sq(workers, 0.0);
        std::atomic_bool finite{true};
        executor.parallel_for(
            rows.size(), workers, 1,
            [&](std::size_t worker, std::size_t begin, std::size_t end) {
              double local_residual_sq = 0.0;
              double local_b_sq = 0.0;
              for (std::size_t pos = begin; pos < end; ++pos) {
                const std::size_t row = rows[pos];
                if (row >= unknown_count)
                  continue;
                const std::size_t target =
                    coupled && row >= panel_count ? row - panel_count : row;
                if (target >= panel_count)
                  continue;
                const JcpNearAccum sum = accumulate_jcp_near_ordered(
                    panels[target], target, panels, x, eps_ratio, kappa,
                    coupled, direct_order, 0, panel_count,
                    result.near_integration, result.quad_buffer, nullptr);
                double ax_row = 0.0;
                if (coupled) {
                  if (row < panel_count) {
                    ax_row = jcp_lhs_phi_for_panel(
                                 panels[target], eps_ratio, lhs_phi) *
                                 x[target] -
                             sum.phi;
                  } else {
                    ax_row = jcp_lhs_phin_for_panel(
                                 panels[target], eps_ratio, lhs_phin) *
                                 x[panel_count + target] -
                             sum.phin;
                  }
                } else {
                  ax_row = jcp_lhs_phi_for_panel(
                               panels[target], eps_ratio, lhs_phi) *
                               x[target] -
                           sum.phi;
                }
                if (!std::isfinite(ax_row)) {
                  finite.store(false, std::memory_order_relaxed);
                  continue;
                }
                const double residual_value = rhs[row] - ax_row;
                local_residual_sq += residual_value * residual_value;
                local_b_sq += rhs[row] * rhs[row];
              }
              partial_residual_sq[worker] = local_residual_sq;
              partial_b_sq[worker] = local_b_sq;
            });
        const double residual_sq = std::accumulate(
            partial_residual_sq.begin(), partial_residual_sq.end(), 0.0);
        const double b_sq =
            std::accumulate(partial_b_sq.begin(), partial_b_sq.end(), 0.0);
        sample.abs = std::sqrt(std::max(0.0, residual_sq));
        const double denom = std::max(1.0e-300, std::sqrt(std::max(0.0, b_sq)));
        sample.rel = sample.abs / denom;
        sample.finite =
            finite.load(std::memory_order_relaxed) && std::isfinite(sample.rel);
        return sample;
      };
    }

    const bool cartesian_channel_shadow_requested =
        options.fmm_cartesian_error_audit == "channel-shadow";
    std::vector<std::size_t> cartesian_channel_shadow_targets;
    std::uint64_t cartesian_channel_shadow_target_hash = 0;
    // Order 1 or 4: see the note beside the matching CLI guard.  The audit
    // is order agnostic and order 4 is needed as the matched control.
    if (cartesian_channel_shadow_requested &&
        !(cartesian_full_gmres &&
          (options.bie_fmm_cartesian_order == 1 ||
           options.bie_fmm_cartesian_order == 4) &&
          node_patch_discretization && coupled &&
          options.node_patch_operator_audit == "on")) {
      throw std::logic_error(
          "cartesian_full_fmm_channel_shadow_scope_mismatch");
    }

    // Node-patch is opt-in and uses a distinct source quadrature center.  Do
    // not spend hundreds of Krylov iterations on a mismatched near/far
    // operator: compare one deterministic FMM action against sampled direct
    // rows before constructing the preconditioner or entering GMRES.
    if (node_patch_discretization && experimental_treecode &&
        options.node_patch_operator_audit == "on") {
      const auto audit_begin = std::chrono::steady_clock::now();
      FmmCartesianPreAuditTelemetry &pre_audit =
          result.fmm_cartesian_pre_audit;
      pre_audit.mode = cartesian_channel_shadow_requested
                           ? "combined+channel-shadow"
                           : "combined";
      pre_audit.enabled = true;
      std::vector<double> probe(rhs.size(), 0.0);
      for (std::size_t i = 0; i < panels.size(); ++i) {
        const double t = static_cast<double>(i + 1);
        probe[i] = 0.5 * std::sin(0.013 * t) + 0.25;
        if (coupled)
          probe[panels.size() + i] = 0.5 * std::cos(0.017 * t) - 0.15;
      }
      pre_audit.probe_hash = fmm_audit_probe_hash(probe, 0x434f4d42494e4544ULL);
      std::vector<double> approximate;
      audit_apply(probe, approximate);
      struct AuditMetrics {
        double diff_sq{0.0};
        double direct_sq{0.0};
        double max_stratum_rel{0.0};
        bool finite{true};
      };
      auto audit_targets = [&](const std::vector<std::size_t> &targets) {
        AuditMetrics metrics;
        if (targets.empty())
          return metrics;
        std::vector<double> target_diff_sq(targets.size(), 0.0);
        std::vector<double> target_direct_sq(targets.size(), 0.0);
        std::atomic_bool finite{true};
        const std::size_t workers = solver_workers(executor, targets.size());
        executor.parallel_for(
            targets.size(), workers, 1,
            [&](std::size_t, std::size_t begin, std::size_t end) {
              for (std::size_t pos = begin; pos < end; ++pos) {
                const std::size_t target = targets[pos];
                const JcpNearAccum sum = accumulate_jcp_near_ordered(
                    panels[target], target, panels, probe, eps_ratio, kappa,
                    coupled, direct_order, 0, panels.size(),
                    result.near_integration, result.quad_buffer, nullptr);
                const double direct_phi =
                    jcp_lhs_phi_for_panel(panels[target], eps_ratio, lhs_phi) *
                        probe[target] -
                    sum.phi;
                double local_diff_sq = 0.0;
                double local_direct_sq = direct_phi * direct_phi;
                if (target >= approximate.size() ||
                    !std::isfinite(direct_phi) ||
                    !std::isfinite(approximate[target])) {
                  finite.store(false, std::memory_order_relaxed);
                  continue;
                }
                const double delta_phi = approximate[target] - direct_phi;
                local_diff_sq += delta_phi * delta_phi;
                if (coupled) {
                  const std::size_t row = panels.size() + target;
                  const double direct_phin =
                      jcp_lhs_phin_for_panel(panels[target], eps_ratio,
                                             lhs_phin) *
                              probe[row] -
                          sum.phin;
                  if (row >= approximate.size() ||
                      !std::isfinite(direct_phin) ||
                      !std::isfinite(approximate[row])) {
                    finite.store(false, std::memory_order_relaxed);
                    continue;
                  }
                  const double delta_phin = approximate[row] - direct_phin;
                  local_diff_sq += delta_phin * delta_phin;
                  local_direct_sq += direct_phin * direct_phin;
                }
                target_diff_sq[pos] = local_diff_sq;
                target_direct_sq[pos] = local_direct_sq;
              }
            });
        metrics.diff_sq =
            std::accumulate(target_diff_sq.begin(), target_diff_sq.end(), 0.0);
        metrics.direct_sq = std::accumulate(target_direct_sq.begin(),
                                            target_direct_sq.end(), 0.0);
        constexpr std::size_t max_strata = 16;
        constexpr std::size_t targets_per_stratum = 8;
        const std::size_t stratum_count = std::max<std::size_t>(
            1u, std::min<std::size_t>(
                    max_strata, targets.size() / targets_per_stratum));
        std::vector<double> stratum_diff_sq(stratum_count, 0.0);
        std::vector<double> stratum_direct_sq(stratum_count, 0.0);
        for (std::size_t pos = 0; pos < targets.size(); ++pos) {
          const std::size_t stratum =
              std::min(stratum_count - 1u,
                       pos * stratum_count / targets.size());
          stratum_diff_sq[stratum] += target_diff_sq[pos];
          stratum_direct_sq[stratum] += target_direct_sq[pos];
        }
        for (std::size_t stratum = 0; stratum < stratum_count; ++stratum) {
          const double rel =
              std::sqrt(std::max(0.0, stratum_diff_sq[stratum])) /
              std::max(1.0e-300,
                       std::sqrt(std::max(0.0, stratum_direct_sq[stratum])));
          metrics.max_stratum_rel = std::max(metrics.max_stratum_rel, rel);
        }
        metrics.finite = finite.load(std::memory_order_relaxed) &&
                         std::isfinite(metrics.diff_sq) &&
                         std::isfinite(metrics.direct_sq) &&
                         std::isfinite(metrics.max_stratum_rel);
        return metrics;
      };

      const std::vector<std::size_t> spatial_targets =
          build_spatial_audit_targets(panels, 512u);
      struct AuditInterval {
        std::size_t begin{0};
        std::size_t end{0};
      };
      std::vector<std::size_t> nested_positions;
      nested_positions.reserve(spatial_targets.size());
      std::vector<AuditInterval> intervals;
      intervals.reserve(2u * spatial_targets.size() + 1u);
      intervals.push_back({0u, spatial_targets.size()});
      for (std::size_t head = 0;
           head < intervals.size() &&
           nested_positions.size() < spatial_targets.size();
           ++head) {
        const AuditInterval interval = intervals[head];
        if (interval.begin >= interval.end)
          continue;
        const std::size_t midpoint =
            interval.begin + (interval.end - interval.begin) / 2u;
        nested_positions.push_back(midpoint);
        if (interval.begin < midpoint)
          intervals.push_back({interval.begin, midpoint});
        if (midpoint + 1u < interval.end)
          intervals.push_back({midpoint + 1u, interval.end});
      }
      auto build_nested_stage =
          [&](std::size_t begin, std::size_t end) {
            end = std::min(end, nested_positions.size());
            begin = std::min(begin, end);
            std::vector<std::size_t> positions(
                nested_positions.begin() + static_cast<std::ptrdiff_t>(begin),
                nested_positions.begin() + static_cast<std::ptrdiff_t>(end));
            std::sort(positions.begin(), positions.end());
            std::vector<std::size_t> targets;
            targets.reserve(positions.size());
            for (const std::size_t position : positions)
              targets.push_back(spatial_targets[position]);
            return targets;
          };
      const std::size_t stage_one_end =
          std::min<std::size_t>(64u, nested_positions.size());
      const std::size_t stage_two_end =
          std::min<std::size_t>(256u, nested_positions.size());
      const std::size_t stage_three_end = nested_positions.size();
      const std::vector<std::size_t> stage_one_targets =
          build_nested_stage(0u, stage_one_end);
      const std::vector<std::size_t> stage_two_targets =
          build_nested_stage(stage_one_end, stage_two_end);
      const std::vector<std::size_t> stage_three_targets =
          build_nested_stage(stage_two_end, stage_three_end);
      const AuditMetrics stage_one = audit_targets(stage_one_targets);
      double diff_sq = stage_one.diff_sq;
      double direct_sq = stage_one.direct_sq;
      bool finite = stage_one.finite;
      double max_stratum_rel = stage_one.max_stratum_rel;
      std::size_t audited_targets = stage_one_targets.size();
      auto cumulative_rel = [&]() {
        return std::sqrt(std::max(0.0, diff_sq)) /
          std::max(1.0e-300,
                   std::sqrt(std::max(0.0, direct_sq)));
      };
      auto needs_expansion = [&](double global_threshold,
                                 double stratum_threshold) {
        const double rel = cumulative_rel();
        return !finite || !std::isfinite(rel) ||
               rel > global_threshold ||
               max_stratum_rel > stratum_threshold;
      };
      constexpr double kNodePatchAudit64ExpandGlobal = 4.0e-3;
      constexpr double kNodePatchAudit64ExpandStratum = 8.0e-3;
      constexpr double kNodePatchAudit256ExpandGlobal = 5.0e-3;
      constexpr double kNodePatchAudit256ExpandStratum = 1.0e-2;
      const bool expand_to_256 =
          needs_expansion(kNodePatchAudit64ExpandGlobal,
                          kNodePatchAudit64ExpandStratum);
      int audit_stages = 1;
      if (expand_to_256 && !stage_two_targets.empty()) {
        const AuditMetrics stage_two = audit_targets(stage_two_targets);
        diff_sq += stage_two.diff_sq;
        direct_sq += stage_two.direct_sq;
        finite = finite && stage_two.finite;
        max_stratum_rel =
            std::max(max_stratum_rel, stage_two.max_stratum_rel);
        audited_targets += stage_two_targets.size();
        ++audit_stages;
      }
      const bool expand_to_512 =
          expand_to_256 &&
          needs_expansion(kNodePatchAudit256ExpandGlobal,
                          kNodePatchAudit256ExpandStratum);
      if (expand_to_512 && !stage_three_targets.empty()) {
        const AuditMetrics stage_three = audit_targets(stage_three_targets);
        diff_sq += stage_three.diff_sq;
        direct_sq += stage_three.direct_sq;
        finite = finite && stage_three.finite;
        max_stratum_rel =
            std::max(max_stratum_rel, stage_three.max_stratum_rel);
        audited_targets += stage_three_targets.size();
        ++audit_stages;
      }
      std::vector<std::size_t> audited_target_set = stage_one_targets;
      if (expand_to_256 && !stage_two_targets.empty())
        audited_target_set.insert(audited_target_set.end(),
                                  stage_two_targets.begin(),
                                  stage_two_targets.end());
      if (expand_to_512 && !stage_three_targets.empty())
        audited_target_set.insert(audited_target_set.end(),
                                  stage_three_targets.begin(),
                                  stage_three_targets.end());
      const double aggregate_rel = cumulative_rel();
      pre_audit.finite = finite && std::isfinite(aggregate_rel);
      pre_audit.aggregate_rel = aggregate_rel;
      pre_audit.max_stratum_rel = max_stratum_rel;
      pre_audit.audited_targets = audited_targets;
      pre_audit.audited_rows = audited_targets * (coupled ? 2u : 1u);
      pre_audit.stages = audit_stages;
      pre_audit.expanded_to_256 = expand_to_256;
      pre_audit.expanded_to_512 = expand_to_512;
      pre_audit.sampled_target_hash =
          fmm_audit_target_hash(audited_target_set);
      if (cartesian_channel_shadow_requested) {
        cartesian_channel_shadow_targets = audited_target_set;
        cartesian_channel_shadow_target_hash =
            pre_audit.sampled_target_hash;
      }
      pre_audit.seconds =
          elapsed_ms(audit_begin, std::chrono::steady_clock::now());
      // Preserve the established node-patch compatibility fields.  They are
      // the fixed pre-GMRES probe; the opt-in solution audit below has its own
      // explicitly named telemetry and never overwrites these values.
      result.tree_vs_direct_matvec_rel = aggregate_rel;
      result.matvec_audit_rows = pre_audit.audited_rows;
      result.matvec_audit_ms = pre_audit.seconds;
      constexpr double kNodePatchOperatorAuditGate = 2.0e-2;
      if (!finite ||
          !std::isfinite(aggregate_rel) ||
          aggregate_rel > kNodePatchOperatorAuditGate) {
        result.status = "failed";
        result.converged = false;
        result.convergence_reason = "node_patch_operator_audit_failed";
        result.reaction_energy_source =
            "not_computed_node_patch_operator_audit_failed";
        result.preconditioner_effective =
            "not_built_node_patch_operator_audit_failed";
        result.matvec_ms_total = matvec_ms_total;
        result.matvec_calls = matvec_calls;
        result.matvec_ms_avg =
            matvec_calls > 0
                ? matvec_ms_total / static_cast<double>(matvec_calls)
                : 0.0;
        return result;
      }

      if (cartesian_channel_shadow_requested) {
        // The two fixed probes isolate the four JCP blocks.  Worklist and
        // operator configuration have already been prepared above; neither
        // the probes nor the sampled targets changes during GMRES.
        const auto channel_begin = std::chrono::steady_clock::now();
        const std::size_t panel_count = panels.size();
        if (panel_count > std::numeric_limits<std::size_t>::max() / 2u) {
          throw std::overflow_error(
              "cartesian_full_fmm_channel_shadow_unknown_count_overflow");
        }
        const std::size_t required_unknowns = 2u * panel_count;
        if (panel_count == 0 || rhs.size() != required_unknowns ||
            audited_target_set.empty() || direct_order.size() != panel_count) {
          throw std::runtime_error(
              "cartesian_full_fmm_channel_shadow_input_shape_mismatch");
        }
        if (pre_audit.sampled_target_hash !=
            fmm_audit_target_hash(audited_target_set)) {
          throw std::runtime_error(
              "cartesian_full_fmm_channel_shadow_target_hash_mismatch");
        }
        for (const std::size_t source : direct_order) {
          if (source >= panel_count)
            throw std::runtime_error(
                "cartesian_full_fmm_channel_shadow_source_index_invalid");
        }
        if (audited_target_set.size() >
            std::numeric_limits<std::uint64_t>::max() / panel_count) {
          throw std::overflow_error(
              "cartesian_full_fmm_channel_shadow_pair_count_overflow");
        }
        pre_audit.channel_direct_pairs_expected =
            static_cast<std::uint64_t>(audited_target_set.size()) *
            static_cast<std::uint64_t>(panel_count);

        std::vector<double> phi_only(required_unknowns, 0.0);
        std::vector<double> phin_only(required_unknowns, 0.0);
        for (std::size_t panel = 0; panel < panel_count; ++panel) {
          const double t = static_cast<double>(panel + 1);
          phi_only[panel] = 0.5 * std::sin(0.013 * t) + 0.25;
          phin_only[panel_count + panel] =
              0.5 * std::cos(0.017 * t) - 0.15;
        }
        std::uint64_t channel_probe_hash =
            fmm_audit_probe_hash(phi_only, 0x5048495f4f4e4c59ULL);
        fmm_audit_hash_mix(channel_probe_hash,
                           fmm_audit_probe_hash(phin_only,
                                                0x5048494e5f4f4e4cULL));
        pre_audit.channel_probe_hash =
            channel_probe_hash == 0 ? 1 : channel_probe_hash;

        std::vector<double> phi_only_approximate;
        std::vector<double> phin_only_approximate;
        apply_cartesian_full_operator(phi_only, phi_only_approximate, false);
        apply_cartesian_full_operator(phin_only, phin_only_approximate,
                                      false);
        if (phi_only_approximate.size() != required_unknowns ||
            phin_only_approximate.size() != required_unknowns) {
          throw std::runtime_error(
              "cartesian_full_fmm_channel_shadow_output_shape_mismatch");
        }

        struct ChannelShadowAccum {
          std::array<double, kJcpExpansionChannelCount> diff_sq{};
          std::array<double, kJcpExpansionChannelCount> reference_sq{};
          std::uint64_t applied_pairs{0};
          bool finite{true};
        };
        std::vector<ChannelShadowAccum> per_target(
            audited_target_set.size());
        std::atomic_bool shadow_finite{true};
        std::atomic<int> shadow_failure{0};
        std::atomic<std::size_t> failure_target{panel_count};
        std::atomic<std::size_t> failure_source{panel_count};
        auto record_shadow_failure = [&](int code, std::size_t target,
                                         std::size_t source) {
          int expected = 0;
          if (shadow_failure.compare_exchange_strong(
                  expected, code, std::memory_order_relaxed)) {
            failure_target.store(target, std::memory_order_relaxed);
            failure_source.store(source, std::memory_order_relaxed);
          }
          shadow_finite.store(false, std::memory_order_relaxed);
        };
        const std::size_t shadow_workers =
            solver_workers(executor, audited_target_set.size());
        executor.parallel_for(
            audited_target_set.size(), shadow_workers, 1,
            [&](std::size_t, std::size_t begin, std::size_t end) {
              for (std::size_t pos = begin; pos < end; ++pos) {
                const std::size_t target = audited_target_set[pos];
                if (target >= panel_count) {
                  record_shadow_failure(1, target, panel_count);
                  per_target[pos].finite = false;
                  continue;
                }
                ChannelShadowAccum &local = per_target[pos];
                double direct_k1 = 0.0;
                double direct_k2 = 0.0;
                double direct_k3 = 0.0;
                double direct_k4 = 0.0;
                // Evaluate each semantic target-source pair once, use that
                // kernel for both fixed probes, and count it only after the
                // P0 invalid/intentional-zero contract has been established.
                for (std::size_t source_position = 0;
                     source_position < direct_order.size(); ++source_position) {
                  const std::size_t source = direct_order[source_position];
                  if (source >= panel_count) {
                    local.finite = false;
                    record_shadow_failure(2, target, source);
                    break;
                  }
                  const JcpNearPairKernel kernel =
                      evaluate_jcp_near_pair_kernel(
                          panels[target], target, panels[source], source,
                          eps_ratio, kappa, true, result.near_integration,
                          result.quad_buffer, nullptr);
                  const bool intentional_zero_self =
                      is_intentional_zero_self_kernel(
                          kernel, true, result.near_integration);
                  if (!kernel.valid && !intentional_zero_self) {
                    local.finite = false;
                    record_shadow_failure(3, target, source);
                    break;
                  }
                  ++local.applied_pairs;
                  if (intentional_zero_self)
                    continue;
                  if (!std::isfinite(kernel.k1) || !std::isfinite(kernel.k2) ||
                      !std::isfinite(kernel.k3) || !std::isfinite(kernel.k4)) {
                    local.finite = false;
                    record_shadow_failure(4, target, source);
                    break;
                  }
                  direct_k1 += kernel.k1 * phin_only[panel_count + source];
                  direct_k2 += kernel.k2 * phi_only[source];
                  direct_k3 += kernel.k3 * phin_only[panel_count + source];
                  direct_k4 += kernel.k4 * phi_only[source];
                }
                if (!local.finite)
                  continue;
                // For K2 and K3, compare interaction terms after explicitly
                // removing the exact diagonal contribution.
                const std::array<double, kJcpExpansionChannelCount>
                    approximate_interaction{
                        -phin_only_approximate[target],
                        jcp_lhs_phi_for_panel(panels[target], eps_ratio,
                                              lhs_phi) *
                                phi_only[target] -
                            phi_only_approximate[target],
                        jcp_lhs_phin_for_panel(panels[target], eps_ratio,
                                               lhs_phin) *
                                phin_only[panel_count + target] -
                            phin_only_approximate[panel_count + target],
                        -phi_only_approximate[panel_count + target]};
                const std::array<double, kJcpExpansionChannelCount>
                    direct_interaction{direct_k1, direct_k2, direct_k3,
                                       direct_k4};
                for (std::size_t channel = 0;
                     channel < kJcpExpansionChannelCount; ++channel) {
                  const double approximate_value =
                      approximate_interaction[channel];
                  const double direct_value = direct_interaction[channel];
                  if (!std::isfinite(approximate_value) ||
                      !std::isfinite(direct_value)) {
                    local.finite = false;
                    record_shadow_failure(5, target, panel_count);
                    break;
                  }
                  const double delta = approximate_value - direct_value;
                  local.diff_sq[channel] = delta * delta;
                  local.reference_sq[channel] = direct_value * direct_value;
                }
              }
            });
        if (!shadow_finite.load(std::memory_order_relaxed)) {
          const int failure = shadow_failure.load(std::memory_order_relaxed);
          const char *reason =
              failure == 1 ? "target_index_invalid"
              : failure == 2 ? "source_index_invalid"
              : failure == 3 ? "kernel_invalid"
              : failure == 4 ? "kernel_nonfinite"
                             : "output_nonfinite";
          std::ostringstream message;
          message << "cartesian_full_fmm_channel_shadow_fail_closed"
                  << " reason=" << reason
                  << " target_panel="
                  << failure_target.load(std::memory_order_relaxed)
                  << " source_panel="
                  << failure_source.load(std::memory_order_relaxed)
                  << " near_integration=" << result.near_integration
                  << " quad_buffer=" << result.quad_buffer;
          throw std::runtime_error(message.str());
        }
        std::array<double, kJcpExpansionChannelCount> diff_sq_by_channel{};
        std::array<double, kJcpExpansionChannelCount>
            reference_sq_by_channel{};
        std::uint64_t applied_pair_count = 0;
        for (const ChannelShadowAccum &local : per_target) {
          if (!local.finite) {
            throw std::runtime_error(
                "cartesian_full_fmm_channel_shadow_coverage_failure");
          }
          if (applied_pair_count >
              std::numeric_limits<std::uint64_t>::max() -
                  local.applied_pairs) {
            throw std::overflow_error(
                "cartesian_full_fmm_channel_shadow_applied_pair_overflow");
          }
          applied_pair_count += local.applied_pairs;
          for (std::size_t channel = 0;
               channel < kJcpExpansionChannelCount; ++channel) {
            diff_sq_by_channel[channel] += local.diff_sq[channel];
            reference_sq_by_channel[channel] += local.reference_sq[channel];
          }
        }
        pre_audit.channel_direct_pairs_applied = applied_pair_count;
        if (pre_audit.channel_direct_pairs_applied !=
            pre_audit.channel_direct_pairs_expected) {
          throw std::runtime_error(
              "cartesian_full_fmm_channel_shadow_coverage_mismatch");
        }
        constexpr double kChannelRelativeReferenceFloor = 1.0e-14;
        for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
             ++channel) {
          FmmCartesianChannelAudit &metric =
              result.fmm_cartesian_channel_audit[channel];
          metric.samples = audited_target_set.size();
          metric.finite = std::isfinite(diff_sq_by_channel[channel]) &&
                          std::isfinite(reference_sq_by_channel[channel]);
          if (!metric.finite) {
            metric.status = "nonfinite";
            throw std::runtime_error(
                "cartesian_full_fmm_channel_shadow_metric_nonfinite");
          }
          metric.abs_l2 =
              std::sqrt(std::max(0.0, diff_sq_by_channel[channel]));
          metric.rms = metric.abs_l2 /
                       std::sqrt(static_cast<double>(metric.samples));
          metric.reference_l2 =
              std::sqrt(std::max(0.0, reference_sq_by_channel[channel]));
          if (metric.reference_l2 <= kChannelRelativeReferenceFloor) {
            metric.available = false;
            metric.status = "absolute_only";
            metric.relative_error =
                std::numeric_limits<double>::quiet_NaN();
          } else {
            metric.available = true;
            metric.status = "relative";
            metric.relative_error = metric.abs_l2 / metric.reference_l2;
          }
        }
        pre_audit.channel_shadow_seconds =
            elapsed_ms(channel_begin, std::chrono::steady_clock::now());
      }
    }

    std::function<void(const std::vector<double> &, std::vector<double> &)>
        custom_preconditioner;
    std::shared_ptr<JcpBlockPreconditioner> block_preconditioner;
    std::shared_ptr<JcpSparseIluPreconditioner> sparse_ilu_preconditioner;
    std::shared_ptr<JcpCoarseGalerkinPreconditioner> coarse_preconditioner;
    double preconditioner_setup_ms = 0.0;
    constexpr std::size_t kBlockPanelPrebuildCap = 12000;
    const bool legacy_block = result.preconditioner_requested == "block";
    const bool budgeted_block =
        result.preconditioner_requested == "block-budgeted";
    const bool coarse_probe =
        result.preconditioner_requested == "coarse-galerkin-probe";
    const bool fmm_nearfield_ilu0_coarse =
        result.preconditioner_requested == "fmm-nearfield-ilu0-coarse";
    const bool fmm_block_jacobi =
        result.preconditioner_requested == "fmm-block-jacobi";
    const bool fmm_overlap_schwarz =
        result.preconditioner_requested == "fmm-overlap-schwarz" ||
        result.preconditioner_requested == "distance-knn-overlap-schwarz";
    // ILU(0) on the classic treecode's own near (singular) list.  It needs no
    // FMM context, so it serves the curved node-patch treecode, which cannot
    // run the FMM M2L path.
    const bool classic_ilu =
        result.preconditioner_requested == "classic-nearfield-ilu0";
    const bool fmm_nearfield_ilu0 =
        result.preconditioner_requested == "fmm-nearfield-ilu0" ||
        fmm_nearfield_ilu0_coarse ||
        result.preconditioner_requested == "distance-knn-ilu0" || classic_ilu;
    const bool distance_knn_graph =
        result.preconditioner_requested == "distance-knn-overlap-schwarz" ||
        result.preconditioner_requested == "distance-knn-ilu0";
    const std::string preconditioner_graph_source =
        distance_knn_graph ? "distance-knn" : "fmm-near";
    auto block_effective_label = [&](bool budgeted,
                                     bool partial) -> std::string {
      std::string base = budgeted ? "block_budgeted" : "block";
      if (result.preconditioner_block_near == "buffer-aware")
        base += "_buffer_aware";
      else if (result.preconditioner_block_near == "operator-near")
        base += "_operator_near";
      if (partial)
        base += "_partial";
      return base;
    };
    if (fmm_block_jacobi || fmm_overlap_schwarz || fmm_nearfield_ilu0) {
      if (kifmm_any_gmres) {
        write_kifmm_gmres_phase("preconditioner_start");
      }
      progress.write("precond_start", result, &jcp_tree_stats, matvec_calls, 0,
                     0, 0.0, "running", true);
      if (!experimental_treecode || (!classic_ilu && !fmm_m2l_single_matvec) ||
          !coupled) {
        throw std::invalid_argument(
            "PBE BIE FMM-native preconditioners require coupled "
            "JCP treecode with -pbe-bie-fmm-m2l=single-matvec-analytic");
      }
      result.preconditioner_block_near = "operator-near";
      const auto precond_begin = std::chrono::steady_clock::now();
      std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext> fmm_context;
      if (!classic_ilu && !(fmm_nearfield_ilu0 && kifmm_fallback_ilu_graph)) {
        fmm_context = ensure_jcp_fmm_m2l_context(
            panels, eps_ratio, kappa,
            static_cast<std::size_t>(std::max(8, effective_tree_leaf_size)),
            effective_tree_opening_theta, result.taylor_order, executor,
            &jcp_tree.fmm_m2l_context);
      }
      std::vector<std::vector<std::size_t>> kifmm_fallback_graph;
      bool ilu_budget_downshifted = false;
      std::size_t ilu_estimated_bytes = 0;
      double ilu_graph_build_ms = 0.0;
      std::shared_ptr<const std::vector<std::vector<std::size_t>>>
          normalized_ilu_graph;
      std::size_t coarse_estimated_bytes = 0;
      if (fmm_nearfield_ilu0_coarse && fmm_context != nullptr &&
          !jcp_tree.nodes.empty()) {
        coarse_preconditioner = make_jcp_coarse_structure(
            panels, jcp_tree, coupled, diagonal,
            static_cast<std::size_t>(
                result.preconditioner_coarse_aggregate_cap));
        result.preconditioner_coarse_base = "fmm-nearfield-ilu0";
        result.preconditioner_coarse_aggregates =
            coarse_preconditioner->aggregates.size();
        result.preconditioner_coarse_dimension =
            coarse_preconditioner->dimension();
        result.preconditioner_coarse_matrix_bytes =
            coarse_preconditioner->dimension() *
                coarse_preconditioner->dimension() * sizeof(double) +
            coarse_preconditioner->dimension() * sizeof(int);
        coarse_estimated_bytes = coarse_preconditioner->estimated_bytes(true);
        result.preconditioner_coarse_status = "pending";
      } else if (fmm_nearfield_ilu0_coarse) {
        result.preconditioner_coarse_status = "operator_unsupported";
      }
      if (fmm_nearfield_ilu0) {
        if (classic_ilu || kifmm_fallback_ilu_graph) {
          if (classic_ilu) {
            const TreeInteractionLists &list = jcp_tree.interactions;
            if (list.singular_offsets.size() != panels.size() + 1U ||
                list.singular_indices.empty())
              throw std::runtime_error(
                  "classic_ilu_requires_cached_interactions: use "
                  "-pbe-bie-interaction-mode=cached");
            // Keep the cap nearest near-list neighbours of each row by
            // collocation distance; ties break on index.  A bounded heap keeps
            // row capacity at the cap even for long near rows.
            const auto cap = static_cast<std::size_t>(
                result.preconditioner_ilu_neighbor_cap);
            kifmm_fallback_graph.resize(panels.size());
            for (std::size_t i = 0; i < panels.size(); ++i) {
              auto &row = kifmm_fallback_graph[i];
              row.reserve(cap);
              const auto distance = [&](std::size_t j) {
                const Panel &a = panels[i];
                const Panel &b = panels[j];
                return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
                       (a.z - b.z) * (a.z - b.z);
              };
              const auto closer = [&](std::size_t j, std::size_t k) {
                const double x = distance(j), y = distance(k);
                return x == y ? j < k : x < y;
              };
              for (std::size_t pos = list.singular_offsets[i];
                   pos < list.singular_offsets[i + 1U]; ++pos) {
                const std::size_t j = list.singular_indices[pos];
                if (j >= panels.size())
                  continue;
                if (row.size() < cap) {
                  row.push_back(j);
                  std::push_heap(row.begin(), row.end(), closer);
                } else if (closer(j, row.front())) {
                  std::pop_heap(row.begin(), row.end(), closer);
                  row.back() = j;
                  std::push_heap(row.begin(), row.end(), closer);
                }
              }
              std::sort(row.begin(), row.end());
            }
          } else {
            kifmm_fallback_graph =
                titan::tabi::fmm_kifmm::build_kifmm_lite_fallback_panel_graph(
                    kifmm_prepared_operator,
                    static_cast<std::size_t>(
                        result.preconditioner_ilu_neighbor_cap));
          }
          result.preconditioner_estimated_bytes =
              estimate_jcp_sparse_ilu0_graph_bytes(kifmm_fallback_graph,
                  diagonal.size(),
                  result.preconditioner_ilu_regularization == "adaptive");
          ilu_estimated_bytes = result.preconditioner_estimated_bytes;
        } else {
          std::size_t ilu_memory_budget =
              result.preconditioner_memory_budget_bytes;
          if (coarse_preconditioner != nullptr) {
            if (coarse_estimated_bytes >= ilu_memory_budget) {
              coarse_preconditioner.reset();
              coarse_estimated_bytes = 0;
              result.preconditioner_coarse_status = "budget_fallback";
            } else {
              ilu_memory_budget -= coarse_estimated_bytes;
            }
          }
          auto ilu_budget = select_fmm_m2l_nearfield_ilu0_budget(
              *fmm_context, diagonal.size(), coupled,
              static_cast<std::size_t>(
                  result.preconditioner_ilu_neighbor_cap_requested),
              ilu_memory_budget,
              result.preconditioner_ilu_regularization == "adaptive",
              preconditioner_graph_source, &executor);
          if (coarse_preconditioner != nullptr && !ilu_budget.fits_budget) {
            coarse_preconditioner.reset();
            coarse_estimated_bytes = 0;
            result.preconditioner_coarse_status = "budget_fallback";
            ilu_budget = select_fmm_m2l_nearfield_ilu0_budget(
                *fmm_context, diagonal.size(), coupled,
                static_cast<std::size_t>(
                    result.preconditioner_ilu_neighbor_cap_requested),
                result.preconditioner_memory_budget_bytes,
                result.preconditioner_ilu_regularization == "adaptive",
                preconditioner_graph_source, &executor);
          }
          normalized_ilu_graph = ilu_budget.normalized_graph;
          ilu_graph_build_ms = ilu_budget.graph_build_ms;
          result.preconditioner_ilu_neighbor_cap =
              static_cast<int>(ilu_budget.effective_neighbor_cap);
          ilu_estimated_bytes = ilu_budget.estimated_bytes;
          result.preconditioner_estimated_bytes =
              ilu_estimated_bytes + coarse_estimated_bytes;
          ilu_budget_downshifted =
              ilu_budget.effective_neighbor_cap <
              ilu_budget.requested_neighbor_cap;
        }
      } else {
        result.preconditioner_estimated_bytes =
            fmm_overlap_schwarz
                ? estimate_fmm_m2l_overlap_schwarz_bytes(
                      *fmm_context, diagonal.size(), coupled,
                      static_cast<std::size_t>(
                          result.preconditioner_block_panel_cap),
                      static_cast<std::size_t>(
                          result.preconditioner_overlap_panel_cap),
                      preconditioner_graph_source)
                : estimate_fmm_m2l_block_jacobi_bytes(
                      *fmm_context, diagonal.size(), coupled,
                      static_cast<std::size_t>(
                          result.preconditioner_block_panel_cap));
      }
      const bool block_budget_cap = result.preconditioner_estimated_bytes >
                                    result.preconditioner_memory_budget_bytes;
      if (!block_budget_cap) {
        if (fmm_nearfield_ilu0) {
          if (classic_ilu || kifmm_fallback_ilu_graph) {
            sparse_ilu_preconditioner =
                build_jcp_sparse_ilu0_preconditioner_from_graph(
                    panels, kifmm_fallback_graph, coupled, eps_ratio, kappa,
                    lhs_phi, lhs_phin, diagonal,
                    result.preconditioner_ilu_drop_tol, result.near_integration,
                    result.quad_buffer,
                    result.preconditioner_ilu_regularization,
                    result.preconditioner_ilu_regularization_max);
          } else {
            sparse_ilu_preconditioner =
                build_fmm_m2l_nearfield_ilu0_preconditioner(
                    panels, *fmm_context, coupled, eps_ratio, kappa, lhs_phi,
                    lhs_phin, diagonal,
                    static_cast<std::size_t>(
                        result.preconditioner_ilu_neighbor_cap),
                    result.preconditioner_ilu_drop_tol, executor,
                    result.near_integration, result.quad_buffer,
                    result.preconditioner_ilu_regularization,
                    result.preconditioner_ilu_regularization_max,
                    preconditioner_graph_source,
                    fmm_nearfield_ilu0_coarse
                        ? coarse_preconditioner.get()
                        : nullptr,
                    normalized_ilu_graph);
            normalized_ilu_graph.reset();
          }
        } else if (fmm_overlap_schwarz) {
          block_preconditioner = build_fmm_m2l_overlap_schwarz_preconditioner(
              panels, *fmm_context, coupled, eps_ratio, kappa, lhs_phi,
              lhs_phin, diagonal,
              static_cast<std::size_t>(result.preconditioner_block_panel_cap),
              static_cast<std::size_t>(result.preconditioner_overlap_panel_cap),
              executor, result.near_integration, result.quad_buffer,
              preconditioner_graph_source);
        } else {
          block_preconditioner = build_fmm_m2l_block_jacobi_preconditioner(
              panels, *fmm_context, coupled, eps_ratio, kappa, lhs_phi,
              lhs_phin, diagonal,
              static_cast<std::size_t>(result.preconditioner_block_panel_cap),
              executor, result.near_integration, result.quad_buffer);
        }
      }
      const bool ilu_candidate_ready =
          sparse_ilu_preconditioner != nullptr &&
          sparse_ilu_preconditioner->valid_panel_count == panels.size() &&
          sparse_ilu_preconditioner->ill_conditioned_blocks == 0 &&
          sparse_ilu_preconditioner->quality_rejected_blocks == 0 &&
          !block_budget_cap;
      if (fmm_nearfield_ilu0_coarse && coarse_preconditioner != nullptr &&
          ilu_candidate_ready) {
        const auto coarse_begin = std::chrono::steady_clock::now();
        TreeWorkspace setup_tree = jcp_tree;
        setup_tree.moment_audit = "off";
        setup_tree.convergence_audit = "off";
        setup_tree.expansion_audit = "off";
        setup_tree.expansion_audit_channel = "off";
        setup_tree.accepted_miss_diagnostics = "off";
        setup_tree.convergence_overreject_audit = "off";
        setup_tree.convergence_overreject_detail = "off";
        setup_tree.convergence_shape_audit = "off";
        setup_tree.convergence_kernel_response_audit = "off";
        TreecodeStats setup_stats;
        auto setup_matvec = [&](const std::vector<double> &x,
                                std::vector<double> &out) {
          apply_jcp_treecode_operator(
              panels, x, eps_ratio, kappa, lhs_phi, lhs_phin, coupled,
              setup_tree, result.near_integration, result.quad_buffer, executor,
              out, &setup_stats);
        };
        const bool verified =
            options.preconditioner_coarse_verify == "full-small";
        constexpr double kHybridCoarseAuditTolerance = 0.35;
        constexpr double kFullSmallHybridAuditTolerance = 1.0e-12;
        bool coarse_valid = complete_jcp_hybrid_galerkin_monopole(
            panels, *coarse_preconditioner,
            sparse_ilu_preconditioner->row_ptr,
            sparse_ilu_preconditioner->col_index, eps_ratio, kappa,
            setup_matvec, verified ? 2 : 1,
            verified ? kFullSmallHybridAuditTolerance
                     : kHybridCoarseAuditTolerance,
            executor);
        bool exact_fallback = false;
        if (!coarse_valid) {
          exact_fallback = true;
          coarse_valid = build_jcp_exact_galerkin(
              panels, *coarse_preconditioner, setup_matvec, verified, true);
        }
        result.preconditioner_coarse_pivot_ratio =
            coarse_preconditioner->pivot_ratio;
        result.preconditioner_coarse_setup_matvec_calls =
            coarse_preconditioner->setup_matvec_calls;
        result.preconditioner_coarse_rp_identity_error_max =
            coarse_preconditioner->rp_identity_error_max;
        result.preconditioner_coarse_verify_rel_error_max =
            coarse_preconditioner->verify_rel_error_max;
        result.preconditioner_coarse_setup_ms =
            elapsed_ms(coarse_begin, std::chrono::steady_clock::now());
        result.preconditioner_coarse_moments_setup_ms =
            coarse_preconditioner->moments_setup_ms;
        result.preconditioner_coarse_far_setup_ms =
            coarse_preconditioner->far_setup_ms;
        result.preconditioner_coarse_audit_setup_ms =
            coarse_preconditioner->audit_setup_ms;
        if (coarse_valid) {
          result.preconditioner_coarse_status =
              exact_fallback ? "active_balanced_exact_fallback"
                             : "active_balanced_hybrid_monopole";
        } else {
          coarse_preconditioner.reset();
          coarse_estimated_bytes = 0;
          result.preconditioner_estimated_bytes = ilu_estimated_bytes;
          result.preconditioner_coarse_status = "factor_fallback";
        }
      } else if (fmm_nearfield_ilu0_coarse &&
                 coarse_preconditioner != nullptr) {
        coarse_preconditioner.reset();
        coarse_estimated_bytes = 0;
        result.preconditioner_estimated_bytes = ilu_estimated_bytes;
        result.preconditioner_coarse_status = "base_fallback";
      }
      preconditioner_setup_ms =
          elapsed_ms(precond_begin, std::chrono::steady_clock::now());
      result.preconditioner_setup_ms = preconditioner_setup_ms;
      if (fmm_nearfield_ilu0) {
        result.preconditioner_graph_setup_ms =
            ilu_graph_build_ms +
            (sparse_ilu_preconditioner != nullptr
                 ? sparse_ilu_preconditioner->graph_setup_ms
                 : 0.0);
        result.preconditioner_graph_cache_reused =
            sparse_ilu_preconditioner != nullptr &&
            sparse_ilu_preconditioner->graph_cache_reused;
        result.preconditioner_near_assembly_setup_ms =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->near_assembly_setup_ms
                : 0.0;
        result.preconditioner_ilu_factor_setup_ms =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->factor_setup_ms
                : 0.0;
        result.preconditioner_valid_blocks =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->valid_blocks
                : 0;
        result.preconditioner_valid_panels =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->valid_panel_count
                : 0;
        result.preconditioner_ill_conditioned_blocks =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->ill_conditioned_blocks
                : 0;
        result.preconditioner_quality_rejected_blocks =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->quality_rejected_blocks
                : 0;
        result.preconditioner_quality_probe_max_gain =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->quality_probe_max_gain
                : 0.0;
        result.preconditioner_quality_probe_rel_residual =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->quality_probe_rel_residual
                : 0.0;
        result.preconditioner_pivot_ratio_min =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->pivot_ratio_min
                : 0.0;
        result.preconditioner_factor_growth_max =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->factor_growth_max
                : 0.0;
        result.preconditioner_pivot_ratio_min_row =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->pivot_ratio_min_row
                : -1;
        result.preconditioner_pivot_ratio_min_column =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->pivot_ratio_min_column
                : -1;
        result.preconditioner_factor_growth_row =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->factor_growth_row
                : -1;
        result.preconditioner_factor_growth_column =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->factor_growth_column
                : -1;
        result.preconditioner_quality_failure_stage =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->quality_failure_stage
                : std::string("off");
        result.preconditioner_quality_failure_row =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->quality_failure_row
                : -1;
        result.preconditioner_quality_failure_column =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->quality_failure_column
                : -1;
        result.preconditioner_quality_failure_probe =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->quality_failure_probe
                : -1;
        result.preconditioner_quality_failure_probe_gain =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->quality_failure_probe_gain
                : 0.0;
        result.preconditioner_quality_failure_probe_rel_residual =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->quality_failure_probe_rel_residual
                : 0.0;
        result.preconditioner_quality_status =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->quality_status
                : std::string("off");
        result.preconditioner_ilu_regularization_selected =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->regularization_selected_relative
                : 0.0;
        result.preconditioner_ilu_regularization_absolute_max =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner
                      ->regularization_selected_absolute_max
                : 0.0;
        result.preconditioner_ilu_regularization_attempts =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->regularization_attempts
                : 0;
        result.preconditioner_ilu_regularization_rejected_attempts =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->regularization_rejected_attempts
                : 0;
        result.preconditioner_ilu_regularization_shifted_rows =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->regularization_shifted_rows
                : 0;
        result.preconditioner_ilu_unshifted_failure_stage =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner
                      ->regularization_unshifted_failure_stage
                : std::string("off");
        result.preconditioner_sparse_nnz_blocks =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->nnz_blocks
                : 0;
        result.preconditioner_sparse_avg_degree =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->avg_degree
                : 0.0;
        result.preconditioner_sparse_max_degree =
            sparse_ilu_preconditioner != nullptr
                ? sparse_ilu_preconditioner->max_degree
                : 0;
      } else {
        result.preconditioner_valid_blocks =
            block_preconditioner != nullptr ? block_preconditioner->valid_blocks
                                            : 0;
        result.preconditioner_valid_panels =
            block_preconditioner != nullptr
                ? block_preconditioner->valid_panel_count
                : 0;
        result.preconditioner_ill_conditioned_blocks =
            block_preconditioner != nullptr
                ? block_preconditioner->ill_conditioned_blocks
                : 0;
        result.preconditioner_quality_rejected_blocks =
            block_preconditioner != nullptr
                ? block_preconditioner->quality_rejected_blocks
                : 0;
        result.preconditioner_quality_probe_max_gain =
            block_preconditioner != nullptr
                ? block_preconditioner->quality_probe_max_gain
                : 0.0;
        result.preconditioner_quality_status =
            block_preconditioner != nullptr
                ? block_preconditioner->quality_status
                : std::string("off");
      }
      const bool block_ill_conditioned =
          block_preconditioner != nullptr &&
          block_preconditioner->ill_conditioned_blocks > 0;
      const bool ilu_ill_conditioned =
          sparse_ilu_preconditioner != nullptr &&
          sparse_ilu_preconditioner->ill_conditioned_blocks > 0;
      const bool ilu_quality_rejected =
          sparse_ilu_preconditioner != nullptr &&
          sparse_ilu_preconditioner->quality_rejected_blocks > 0;
      const std::size_t block_panel_threshold = static_cast<std::size_t>(
          std::ceil(0.70 * static_cast<double>(
                               std::max<std::size_t>(1, panels.size()))));
      if (fmm_nearfield_ilu0 && sparse_ilu_preconditioner != nullptr &&
          sparse_ilu_preconditioner->valid_panel_count == panels.size() &&
          !ilu_ill_conditioned && !ilu_quality_rejected && !block_budget_cap) {
        if (classic_ilu) {
          result.preconditioner_effective = "classic_nearfield_ilu0";
        } else if (kifmm_fallback_ilu_graph) {
          result.preconditioner_effective =
              "fmm_nearfield_ilu0_kifmm_fallback_capped";
        } else if (fmm_nearfield_ilu0_coarse &&
                   coarse_preconditioner != nullptr) {
          result.preconditioner_effective =
              "fmm_nearfield_ilu0_coarse_galerkin_balanced_" +
              result.preconditioner_coarse_status;
        } else if (fmm_nearfield_ilu0_coarse) {
          result.preconditioner_effective =
              "fmm_nearfield_ilu0_coarse_" +
              result.preconditioner_coarse_status + "_to_ilu0";
        } else {
          result.preconditioner_effective =
              distance_knn_graph ? "distance_knn_ilu0"
                                 : "fmm_nearfield_ilu0";
        }
        if (sparse_ilu_preconditioner->regularization_selected_relative > 0.0)
          result.preconditioner_effective += "_regularized";
        if (ilu_budget_downshifted)
          result.preconditioner_effective += "_budget_downshift";
        result.memory_bytes += result.preconditioner_estimated_bytes;
        if (fmm_nearfield_ilu0_coarse && coarse_preconditioner != nullptr) {
          std::function<void(const std::vector<double> &,
                             std::vector<double> &)>
              balanced_base_apply =
                  [sparse_ilu_preconditioner](
                      const std::vector<double> &base_in,
                      std::vector<double> &base_out) {
                    sparse_ilu_preconditioner->apply(base_in, base_out);
                  };
          custom_preconditioner =
              [coarse_preconditioner, &panels, &result,
               balanced_base_apply](const std::vector<double> &in,
                                    std::vector<double> &out) {
                const auto begin = std::chrono::steady_clock::now();
                coarse_preconditioner->apply_balanced(panels, in, out,
                                                      balanced_base_apply);
                result.preconditioner_coarse_apply_ms +=
                    elapsed_ms(begin, std::chrono::steady_clock::now());
                ++result.preconditioner_coarse_apply_calls;
              };
        } else {
          custom_preconditioner =
              [sparse_ilu_preconditioner](const std::vector<double> &in,
                                          std::vector<double> &out) {
                sparse_ilu_preconditioner->apply(in, out);
              };
        }
      } else if (!fmm_nearfield_ilu0 && block_preconditioner != nullptr &&
                 block_preconditioner->valid_blocks > 0 &&
                 block_preconditioner->valid_panel_count >=
                     block_panel_threshold &&
                 !block_budget_cap) {
        const bool partial_block =
            block_ill_conditioned ||
            block_preconditioner->valid_panel_count < panels.size();
        if (fmm_overlap_schwarz) {
          result.preconditioner_effective =
              distance_knn_graph
                  ? (partial_block ? "distance_knn_overlap_schwarz_partial"
                                   : "distance_knn_overlap_schwarz")
                  : (partial_block ? "fmm_overlap_schwarz_partial"
                                   : "fmm_overlap_schwarz");
        } else {
          result.preconditioner_effective =
              partial_block ? "fmm_block_jacobi_partial" : "fmm_block_jacobi";
        }
        result.memory_bytes += result.preconditioner_estimated_bytes;
        custom_preconditioner = [block_preconditioner,
                                 &executor](const std::vector<double> &in,
                                            std::vector<double> &out) {
          block_preconditioner->apply_parallel(in, out, executor);
        };
      } else if (block_budget_cap) {
        block_preconditioner.reset();
        sparse_ilu_preconditioner.reset();
        if (fmm_nearfield_ilu0) {
          result.preconditioner_effective =
              kifmm_fallback_ilu_graph
                  ? "diagonal_kifmm_fallback_ilu_budget_cap"
              : distance_knn_graph ? "diagonal_knn_ilu_budget_cap"
                                   : "diagonal_fmm_ilu_budget_cap";
        } else {
          result.preconditioner_effective =
              fmm_overlap_schwarz
                  ? (distance_knn_graph ? "diagonal_knn_overlap_budget_cap"
                                        : "diagonal_fmm_overlap_budget_cap")
                  : "diagonal_fmm_block_budget_cap";
        }
      } else if (block_ill_conditioned || ilu_ill_conditioned ||
                 ilu_quality_rejected) {
        block_preconditioner.reset();
        sparse_ilu_preconditioner.reset();
        if (fmm_nearfield_ilu0) {
          result.preconditioner_effective =
              kifmm_fallback_ilu_graph
                  ? "diagonal_kifmm_fallback_ilu_factor_fallback"
              : distance_knn_graph
                  ? (ilu_quality_rejected ? "diagonal_knn_ilu_quality_fallback"
                                          : "diagonal_knn_ilu_factor_fallback")
                  : (ilu_quality_rejected ? "diagonal_fmm_ilu_quality_fallback"
                                          : "diagonal_fmm_ilu_factor_fallback");
        } else {
          result.preconditioner_effective =
              fmm_overlap_schwarz
                  ? (distance_knn_graph
                         ? "diagonal_knn_overlap_ill_conditioned"
                         : "diagonal_fmm_overlap_ill_conditioned")
                  : "diagonal_fmm_block_ill_conditioned";
        }
      } else {
        block_preconditioner.reset();
        sparse_ilu_preconditioner.reset();
        if (fmm_nearfield_ilu0) {
          result.preconditioner_effective =
              kifmm_fallback_ilu_graph ? "diagonal_kifmm_fallback_ilu_fallback"
              : distance_knn_graph     ? "diagonal_knn_ilu_fallback"
                                       : "diagonal_fmm_ilu_fallback";
        } else {
          result.preconditioner_effective =
              fmm_overlap_schwarz
                  ? (distance_knn_graph ? "diagonal_knn_overlap_fallback"
                                        : "diagonal_fmm_overlap_fallback")
                  : "diagonal_fmm_block_fallback";
        }
      }
      progress.write("precond_end", result, &jcp_tree_stats, matvec_calls, 0, 0,
                     0.0, "running", true);
      if (kifmm_any_gmres) {
        write_kifmm_gmres_phase("preconditioner_done");
      }
    } else if ((legacy_block || budgeted_block) && !jcp_tree.nodes.empty() &&
               (!legacy_block || panels.size() <= kBlockPanelPrebuildCap)) {
      progress.write("precond_start", result, &jcp_tree_stats, matvec_calls, 0,
                     0, 0.0, "running", true);
      const auto precond_begin = std::chrono::steady_clock::now();
      constexpr std::size_t kBlockMemorySoftCapBytes =
          static_cast<std::size_t>(384) * 1024u * 1024u;
      constexpr std::size_t kBlockWorkingSetSoftCapBytes =
          static_cast<std::size_t>(1024) * 1024u * 1024u;
      const std::size_t block_panel_cap =
          budgeted_block
              ? static_cast<std::size_t>(result.preconditioner_block_panel_cap)
              : static_cast<std::size_t>(32);
      const std::vector<JcpBlockRange> block_ranges =
          make_jcp_block_ranges(jcp_tree, panels.size(), block_panel_cap);
      result.preconditioner_estimated_bytes =
          estimate_jcp_block_preconditioner_bytes(block_ranges, diagonal.size(),
                                                  coupled);
      const bool block_budget_cap =
          budgeted_block && result.preconditioner_estimated_bytes >
                                result.preconditioner_memory_budget_bytes;
      const bool block_prebuild_working_set_cap =
          legacy_block &&
          current_working_set_bytes() > kBlockWorkingSetSoftCapBytes;
      if (!block_budget_cap && !block_prebuild_working_set_cap) {
        block_preconditioner = build_jcp_block_preconditioner(
            panels, jcp_tree, coupled, eps_ratio, kappa, lhs_phi, lhs_phin,
            diagonal, result.near_integration, result.quad_buffer,
            result.preconditioner_block_near, block_ranges, executor);
      }
      preconditioner_setup_ms =
          elapsed_ms(precond_begin, std::chrono::steady_clock::now());
      result.preconditioner_setup_ms = preconditioner_setup_ms;
      result.preconditioner_valid_blocks =
          block_preconditioner != nullptr ? block_preconditioner->valid_blocks
                                          : 0;
      result.preconditioner_valid_panels =
          block_preconditioner != nullptr
              ? block_preconditioner->valid_panel_count
              : 0;
      result.preconditioner_ill_conditioned_blocks =
          block_preconditioner != nullptr
              ? block_preconditioner->ill_conditioned_blocks
              : 0;
      result.preconditioner_quality_rejected_blocks =
          block_preconditioner != nullptr
              ? block_preconditioner->quality_rejected_blocks
              : 0;
      result.preconditioner_quality_probe_max_gain =
          block_preconditioner != nullptr
              ? block_preconditioner->quality_probe_max_gain
              : 0.0;
      result.preconditioner_quality_status =
          block_preconditioner != nullptr ? block_preconditioner->quality_status
                                          : std::string("off");
      const bool block_memory_cap =
          legacy_block &&
          (result.preconditioner_estimated_bytes > kBlockMemorySoftCapBytes ||
           block_prebuild_working_set_cap ||
           current_working_set_bytes() > kBlockWorkingSetSoftCapBytes);
      const bool block_ill_conditioned =
          block_preconditioner != nullptr &&
          block_preconditioner->ill_conditioned_blocks > 0;
      const std::size_t block_panel_threshold = static_cast<std::size_t>(
          std::ceil(0.70 * static_cast<double>(
                               std::max<std::size_t>(1, panels.size()))));
      if (block_preconditioner != nullptr &&
          block_preconditioner->valid_blocks > 0 &&
          block_preconditioner->valid_panel_count >= block_panel_threshold &&
          !block_budget_cap && !block_memory_cap) {
        const bool partial_block =
            block_ill_conditioned ||
            block_preconditioner->valid_panel_count < panels.size();
        result.preconditioner_effective =
            block_effective_label(budgeted_block, partial_block);
        result.memory_bytes += result.preconditioner_estimated_bytes;
        custom_preconditioner = [block_preconditioner,
                                 &executor](const std::vector<double> &in,
                                            std::vector<double> &out) {
          block_preconditioner->apply_parallel(in, out, executor);
        };
      } else if (block_budget_cap) {
        block_preconditioner.reset();
        result.preconditioner_effective = "diagonal_block_budget_cap";
      } else if (block_memory_cap) {
        block_preconditioner.reset();
        result.preconditioner_effective = "diagonal_block_memory_cap";
      } else if (block_ill_conditioned) {
        block_preconditioner.reset();
        result.preconditioner_effective = "diagonal_block_ill_conditioned";
      } else {
        result.preconditioner_effective = "diagonal_block_fallback";
      }
      progress.write("precond_end", result, &jcp_tree_stats, matvec_calls, 0, 0,
                     0.0, "running", true);
    } else if (legacy_block || budgeted_block) {
      progress.write("precond_start", result, &jcp_tree_stats, matvec_calls, 0,
                     0, 0.0, "running", true);
      result.preconditioner_effective =
          legacy_block && panels.size() > kBlockPanelPrebuildCap
              ? "diagonal_block_memory_cap"
              : "diagonal_block_fallback";
      progress.write("precond_end", result, &jcp_tree_stats, matvec_calls, 0, 0,
                     0.0, "running", true);
    } else if (coarse_probe && experimental_treecode &&
               !jcp_tree.nodes.empty()) {
      progress.write("precond_start", result, &jcp_tree_stats, matvec_calls, 0,
                     0, 0.0, "running", true);
      const auto precond_begin = std::chrono::steady_clock::now();
      coarse_preconditioner = make_jcp_coarse_structure(
          panels, jcp_tree, coupled, diagonal,
          static_cast<std::size_t>(result.preconditioner_coarse_aggregate_cap));
      result.preconditioner_coarse_aggregates =
          coarse_preconditioner->aggregates.size();
      result.preconditioner_coarse_dimension =
          coarse_preconditioner->dimension();
      result.preconditioner_coarse_matrix_bytes =
          coarse_preconditioner->dimension() *
              coarse_preconditioner->dimension() * sizeof(double) +
          coarse_preconditioner->dimension() * sizeof(int);

      std::vector<JcpBlockRange> coarse_base_ranges;
      std::size_t coarse_base_estimated_bytes = 0;
      const bool coarse_block_base =
          result.preconditioner_coarse_base == "block-budgeted";
      if (coarse_block_base) {
        coarse_base_ranges = make_jcp_block_ranges(
            jcp_tree, panels.size(),
            static_cast<std::size_t>(result.preconditioner_block_panel_cap));
        coarse_base_estimated_bytes = estimate_jcp_block_preconditioner_bytes(
            coarse_base_ranges, diagonal.size(), coupled);
      }
      result.preconditioner_estimated_bytes =
          coarse_preconditioner->estimated_bytes() +
          coarse_base_estimated_bytes;
      const bool coarse_budget_cap = result.preconditioner_estimated_bytes >
                                     result.preconditioner_memory_budget_bytes;
      if (coarse_budget_cap) {
        coarse_preconditioner.reset();
        result.preconditioner_coarse_status = "budget_fallback";
        result.preconditioner_effective = "diagonal_coarse_budget_cap";
      } else {
        bool block_base_active = false;
        if (coarse_block_base) {
          block_preconditioner = build_jcp_block_preconditioner(
              panels, jcp_tree, coupled, eps_ratio, kappa, lhs_phi, lhs_phin,
              diagonal, result.near_integration, result.quad_buffer,
              result.preconditioner_block_near, coarse_base_ranges, executor);
          const std::size_t threshold = static_cast<std::size_t>(std::ceil(
              0.70 *
              static_cast<double>(std::max<std::size_t>(1, panels.size()))));
          block_base_active =
              block_preconditioner != nullptr &&
              block_preconditioner->valid_blocks > 0 &&
              block_preconditioner->valid_panel_count >= threshold;
          if (block_base_active)
            coarse_preconditioner->block_base = block_preconditioner;
        }
        TreeWorkspace setup_tree = jcp_tree;
        setup_tree.moment_audit = "off";
        setup_tree.convergence_audit = "off";
        setup_tree.expansion_audit = "off";
        setup_tree.expansion_audit_channel = "off";
        setup_tree.accepted_miss_diagnostics = "off";
        setup_tree.convergence_overreject_audit = "off";
        setup_tree.convergence_overreject_detail = "off";
        setup_tree.convergence_shape_audit = "off";
        setup_tree.convergence_kernel_response_audit = "off";
        TreecodeStats setup_stats;
        auto setup_matvec = [&](const std::vector<double> &x,
                                std::vector<double> &out) {
          apply_jcp_treecode_operator(
              panels, x, eps_ratio, kappa, lhs_phi, lhs_phin, coupled,
              setup_tree, result.near_integration, result.quad_buffer, executor,
              out, &setup_stats);
        };
        const bool verified =
            options.preconditioner_coarse_verify == "full-small";
        const bool coarse_valid = build_jcp_exact_galerkin(
            panels, *coarse_preconditioner, setup_matvec, verified);
        result.preconditioner_coarse_pivot_ratio =
            coarse_preconditioner->pivot_ratio;
        result.preconditioner_coarse_setup_matvec_calls =
            coarse_preconditioner->setup_matvec_calls;
        result.preconditioner_coarse_rp_identity_error_max =
            coarse_preconditioner->rp_identity_error_max;
        result.preconditioner_coarse_verify_rel_error_max =
            coarse_preconditioner->verify_rel_error_max;
        if (coarse_valid) {
          result.preconditioner_coarse_status = "active";
          result.preconditioner_effective =
              block_base_active ? "coarse_galerkin_probe_block_budgeted"
                                : "coarse_galerkin_probe_diagonal";
          result.memory_bytes += result.preconditioner_estimated_bytes;
          custom_preconditioner = [coarse_preconditioner, &panels, &executor,
                                   &result](const std::vector<double> &in,
                                            std::vector<double> &out) {
            const auto begin = std::chrono::steady_clock::now();
            coarse_preconditioner->apply(panels, in, out, executor);
            result.preconditioner_coarse_apply_ms +=
                elapsed_ms(begin, std::chrono::steady_clock::now());
            ++result.preconditioner_coarse_apply_calls;
          };
        } else {
          coarse_preconditioner.reset();
          result.preconditioner_coarse_status = "factor_fallback";
          result.preconditioner_effective = "diagonal_coarse_factor_fallback";
        }
      }
      preconditioner_setup_ms =
          elapsed_ms(precond_begin, std::chrono::steady_clock::now());
      result.preconditioner_setup_ms = preconditioner_setup_ms;
      result.preconditioner_coarse_setup_ms = preconditioner_setup_ms;
      progress.write("precond_end", result, &jcp_tree_stats, matvec_calls, 0, 0,
                     0.0, "running", true);
    } else if (coarse_probe) {
      progress.write("precond_start", result, &jcp_tree_stats, matvec_calls, 0,
                     0, 0.0, "running", true);
      result.preconditioner_coarse_status = "operator_unsupported";
      result.preconditioner_effective = "diagonal_coarse_unsupported";
      progress.write("precond_end", result, &jcp_tree_stats, matvec_calls, 0, 0,
                     0.0, "running", true);
    }

    const auto gmres_begin = std::chrono::steady_clock::now();
    if (kifmm_any_gmres) {
      write_kifmm_gmres_phase("gmres_start");
    }
    progress.write("gmres_start", result, &jcp_tree_stats, matvec_calls, 0, 0,
                   0.0, "running", true);
    auto gmres_progress = [&](int gmres_iter, int restart_iter,
                              double true_residual_rel,
                              double preconditioned_residual_rel, bool force,
                              const std::string &phase,
                              const LinearSolveResult &state) {
      result.orthogonalization_ms = state.orthogonalization_ms;
      result.preconditioner_ms = state.preconditioner_ms;
      result.preconditioner_apply_ms = state.preconditioner_apply_ms;
      result.preconditioner_apply_calls = state.preconditioner_apply_calls;
      result.true_residual_probe_scope = state.true_residual_probe_scope;
      result.true_residual_sample_count = state.true_residual_sample_count;
      result.true_residual_probe_calls = state.true_residual_probe_calls;
      result.true_residual_probe_ms = state.true_residual_probe_ms;
      result.sampled_true_residual_rel = state.sampled_true_residual_rel;
      result.sampled_true_residual_rows = state.sampled_true_residual_rows;
      result.sampled_probe_calls = state.sampled_probe_calls;
      result.sampled_probe_ms = state.sampled_probe_ms;
      result.full_true_residual_confirm_calls =
          state.full_true_residual_confirm_calls;
      result.full_true_residual_confirm_ms =
          state.full_true_residual_confirm_ms;
      if (phase != "gmres_sampled_true_probe" &&
          std::isfinite(true_residual_rel)) {
        result.final_residual = true_residual_rel;
        result.true_residual_rel = true_residual_rel;
        if (result.best_residual <= 0.0 ||
            true_residual_rel < result.best_residual) {
          result.best_residual = true_residual_rel;
          result.best_residual_iter = gmres_iter;
        }
      }
      if (std::isfinite(preconditioned_residual_rel)) {
        result.final_preconditioned_residual = preconditioned_residual_rel;
        if (result.best_preconditioned_residual <= 0.0 ||
            preconditioned_residual_rel < result.best_preconditioned_residual) {
          result.best_preconditioned_residual = preconditioned_residual_rel;
        }
      }
      progress.write(phase, result, &jcp_tree_stats, matvec_calls, gmres_iter,
                     restart_iter, true_residual_rel, "running", force);
      if (kifmm_any_gmres) {
        write_kifmm_convergence_trace(
            phase, gmres_iter, preconditioned_residual_rel, true_residual_rel);
      }
    };
    LinearSolveResult linear = gmres_solve(
        matvec, rhs, diagonal, max_iter, tol, options.gmres_restart, executor,
        custom_preconditioner, gmres_progress, result.true_residual_probe_mode,
        result.true_residual_probe_interval, result.true_residual_probe_scope,
        result.true_residual_sample_count, sampled_true_residual);
    result.gmres_ms = elapsed_ms(gmres_begin, std::chrono::steady_clock::now());
    result.iterations = linear.iterations;
    result.residual = linear.residual;
    result.gmres_restart_requested = linear.restart_requested;
    result.gmres_restart_effective = linear.restart_effective;
    result.best_residual = linear.best_residual;
    result.best_residual_iter = linear.best_residual_iter;
    result.final_residual = linear.final_residual;
    result.best_preconditioned_residual = linear.best_preconditioned_residual;
    result.final_preconditioned_residual = linear.final_preconditioned_residual;
    result.true_residual_abs = linear.true_residual_abs;
    result.true_residual_rel = linear.true_residual_rel;
    result.true_residual_probe_mode = linear.true_residual_probe_mode;
    result.true_residual_probe_scope = linear.true_residual_probe_scope;
    result.true_residual_probe_interval = linear.true_residual_probe_interval;
    result.true_residual_sample_count = linear.true_residual_sample_count;
    result.true_residual_probe_calls = linear.true_residual_probe_calls;
    result.true_residual_probe_ms = linear.true_residual_probe_ms;
    result.sampled_true_residual_rel = linear.sampled_true_residual_rel;
    result.sampled_true_residual_rows = linear.sampled_true_residual_rows;
    result.sampled_probe_calls = linear.sampled_probe_calls;
    result.sampled_probe_ms = linear.sampled_probe_ms;
    result.full_true_residual_confirm_calls =
        linear.full_true_residual_confirm_calls;
    result.full_true_residual_confirm_ms = linear.full_true_residual_confirm_ms;
    result.converged = linear.converged;
    result.convergence_reason = linear.convergence_reason;
    result.residual_history = residual_history_string(linear.residual_history);
    result.matvec_ms_total = matvec_ms_total;
    result.matvec_calls = matvec_calls;
    result.matvec_ms_avg =
        matvec_calls > 0 ? matvec_ms_total / static_cast<double>(matvec_calls)
                         : 0.0;
    // Node-patch deliberately retains its historical legacy audit fields as
    // the fixed pre-GMRES probe.  The opt-in channel shadow adds a separate
    // sampled comparison of the final GMRES iterate; it never repurposes the
    // legacy field or affects the fixed operator used by GMRES.
    if (cartesian_channel_shadow_requested) {
      const auto solution_audit_begin = std::chrono::steady_clock::now();
      FmmCartesianSolutionAuditTelemetry &solution_audit =
          result.fmm_cartesian_solution_audit;
      solution_audit.enabled = true;
      const std::size_t panel_count = panels.size();
      if (panel_count == 0 ||
          panel_count > std::numeric_limits<std::size_t>::max() / 2u) {
        throw std::overflow_error(
            "cartesian_full_fmm_solution_audit_unknown_count_overflow");
      }
      const std::size_t required_unknowns = 2u * panel_count;
      const std::vector<double> &audit_x =
          !linear.best_x.empty() ? linear.best_x : linear.x;
      if (audit_x.size() != required_unknowns ||
          cartesian_channel_shadow_targets.empty() ||
          direct_order.size() != panel_count) {
        throw std::runtime_error(
            "cartesian_full_fmm_solution_audit_input_shape_mismatch");
      }
      const std::uint64_t target_hash =
          fmm_audit_target_hash(cartesian_channel_shadow_targets);
      if (target_hash == 0 || target_hash != cartesian_channel_shadow_target_hash ||
          target_hash != result.fmm_cartesian_pre_audit.sampled_target_hash) {
        throw std::runtime_error(
            "cartesian_full_fmm_solution_audit_target_hash_mismatch");
      }
      if (cartesian_channel_shadow_targets.size() >
          std::numeric_limits<std::uint64_t>::max() / panel_count) {
        throw std::overflow_error(
            "cartesian_full_fmm_solution_audit_pair_count_overflow");
      }
      solution_audit.targets = cartesian_channel_shadow_targets.size();
      if (solution_audit.targets >
          std::numeric_limits<std::size_t>::max() / 2u) {
        throw std::overflow_error(
            "cartesian_full_fmm_solution_audit_row_count_overflow");
      }
      solution_audit.rows = 2u * solution_audit.targets;
      solution_audit.target_hash = target_hash;
      solution_audit.direct_pairs_expected =
          static_cast<std::uint64_t>(solution_audit.targets) *
          static_cast<std::uint64_t>(panel_count);

      // Do not use matvec here: audit_apply intentionally omits both GMRES
      // timing and full-FMM apply counters.
      std::vector<double> approximate;
      audit_apply(audit_x, approximate);
      if (approximate.size() != required_unknowns) {
        throw std::runtime_error(
            "cartesian_full_fmm_solution_audit_output_shape_mismatch");
      }

      struct SolutionAuditAccum {
        double diff_sq{0.0};
        double direct_sq{0.0};
        std::uint64_t applied_pairs{0};
        bool finite{true};
      };
      std::vector<SolutionAuditAccum> per_target(solution_audit.targets);
      std::atomic_bool audit_finite{true};
      std::atomic<int> audit_failure{0};
      std::atomic<std::size_t> failure_target{panel_count};
      std::atomic<std::size_t> failure_source{panel_count};
      auto record_solution_audit_failure = [&](int code, std::size_t target,
                                               std::size_t source) {
        int expected = 0;
        if (audit_failure.compare_exchange_strong(
                expected, code, std::memory_order_relaxed)) {
          failure_target.store(target, std::memory_order_relaxed);
          failure_source.store(source, std::memory_order_relaxed);
        }
        audit_finite.store(false, std::memory_order_relaxed);
      };
      const std::size_t audit_workers =
          solver_workers(executor, solution_audit.targets);
      executor.parallel_for(
          solution_audit.targets, audit_workers, 1,
          [&](std::size_t, std::size_t begin, std::size_t end) {
            for (std::size_t pos = begin; pos < end; ++pos) {
              const std::size_t target = cartesian_channel_shadow_targets[pos];
              SolutionAuditAccum &local = per_target[pos];
              if (target >= panel_count) {
                local.finite = false;
                record_solution_audit_failure(1, target, panel_count);
                continue;
              }
              double sum_phi = 0.0;
              double sum_phin = 0.0;
              // This is intentionally a local explicit direct loop rather
              // than accumulate_jcp_near_ordered(): the audit must fail closed
              // for an unexpected invalid kernel and measure actual semantic
              // target-source coverage exactly once.
              for (std::size_t source_position = 0;
                   source_position < direct_order.size(); ++source_position) {
                const std::size_t source = direct_order[source_position];
                if (source >= panel_count) {
                  local.finite = false;
                  record_solution_audit_failure(2, target, source);
                  break;
                }
                const JcpNearPairKernel kernel = evaluate_jcp_near_pair_kernel(
                    panels[target], target, panels[source], source, eps_ratio,
                    kappa, true, result.near_integration, result.quad_buffer,
                    nullptr);
                const bool intentional_zero_self =
                    is_intentional_zero_self_kernel(
                        kernel, true, result.near_integration);
                if (!kernel.valid && !intentional_zero_self) {
                  local.finite = false;
                  record_solution_audit_failure(3, target, source);
                  break;
                }
                ++local.applied_pairs;
                if (intentional_zero_self)
                  continue;
                if (!std::isfinite(kernel.k1) || !std::isfinite(kernel.k2) ||
                    !std::isfinite(kernel.k3) || !std::isfinite(kernel.k4)) {
                  local.finite = false;
                  record_solution_audit_failure(4, target, source);
                  break;
                }
                sum_phi += kernel.k1 * audit_x[panel_count + source] +
                           kernel.k2 * audit_x[source];
                sum_phin += kernel.k3 * audit_x[panel_count + source] +
                            kernel.k4 * audit_x[source];
              }
              if (!local.finite)
                continue;
              const double direct_phi =
                  jcp_lhs_phi_for_panel(panels[target], eps_ratio, lhs_phi) *
                      audit_x[target] -
                  sum_phi;
              const double direct_phin =
                  jcp_lhs_phin_for_panel(panels[target], eps_ratio, lhs_phin) *
                      audit_x[panel_count + target] -
                  sum_phin;
              const double approximate_phi = approximate[target];
              const double approximate_phin = approximate[panel_count + target];
              if (!std::isfinite(direct_phi) || !std::isfinite(direct_phin) ||
                  !std::isfinite(approximate_phi) ||
                  !std::isfinite(approximate_phin)) {
                local.finite = false;
                record_solution_audit_failure(5, target, panel_count);
                continue;
              }
              const double phi_delta = approximate_phi - direct_phi;
              const double phin_delta = approximate_phin - direct_phin;
              local.diff_sq = phi_delta * phi_delta + phin_delta * phin_delta;
              local.direct_sq = direct_phi * direct_phi +
                                direct_phin * direct_phin;
              if (!std::isfinite(local.diff_sq) ||
                  !std::isfinite(local.direct_sq)) {
                local.finite = false;
                record_solution_audit_failure(6, target, panel_count);
              }
            }
          });
      if (!audit_finite.load(std::memory_order_relaxed)) {
        const int failure = audit_failure.load(std::memory_order_relaxed);
        const char *reason =
            failure == 1 ? "target_index_invalid"
            : failure == 2 ? "source_index_invalid"
            : failure == 3 ? "kernel_invalid"
            : failure == 4 ? "kernel_nonfinite"
            : failure == 5 ? "output_nonfinite"
                           : "accumulator_nonfinite";
        std::ostringstream message;
        message << "cartesian_full_fmm_solution_audit_fail_closed"
                << " reason=" << reason
                << " target_panel="
                << failure_target.load(std::memory_order_relaxed)
                << " source_panel="
                << failure_source.load(std::memory_order_relaxed)
                << " near_integration=" << result.near_integration
                << " quad_buffer=" << result.quad_buffer;
        throw std::runtime_error(message.str());
      }
      double diff_sq = 0.0;
      double direct_sq = 0.0;
      std::uint64_t applied_pair_count = 0;
      for (const SolutionAuditAccum &local : per_target) {
        if (!local.finite) {
          throw std::runtime_error(
              "cartesian_full_fmm_solution_audit_coverage_failure");
        }
        if (applied_pair_count >
            std::numeric_limits<std::uint64_t>::max() -
                local.applied_pairs) {
          throw std::overflow_error(
              "cartesian_full_fmm_solution_audit_applied_pair_overflow");
        }
        applied_pair_count += local.applied_pairs;
        diff_sq += local.diff_sq;
        direct_sq += local.direct_sq;
      }
      solution_audit.direct_pairs_applied = applied_pair_count;
      if (solution_audit.direct_pairs_applied !=
          solution_audit.direct_pairs_expected) {
        throw std::runtime_error(
            "cartesian_full_fmm_solution_audit_coverage_mismatch");
      }
      if (!std::isfinite(diff_sq) || !std::isfinite(direct_sq)) {
        throw std::runtime_error(
            "cartesian_full_fmm_solution_audit_accumulator_nonfinite");
      }
      solution_audit.absolute_l2 = std::sqrt(std::max(0.0, diff_sq));
      solution_audit.direct_l2 = std::sqrt(std::max(0.0, direct_sq));
      constexpr double kSolutionRelativeReferenceFloor = 1.0e-14;
      if (solution_audit.direct_l2 <= kSolutionRelativeReferenceFloor) {
        solution_audit.relative_available = false;
        solution_audit.status = "absolute_only";
        solution_audit.relative_error =
            std::numeric_limits<double>::quiet_NaN();
      } else {
        solution_audit.relative_available = true;
        solution_audit.status = "relative";
        solution_audit.relative_error =
            solution_audit.absolute_l2 / solution_audit.direct_l2;
      }
      solution_audit.finite =
          std::isfinite(solution_audit.absolute_l2) &&
          std::isfinite(solution_audit.direct_l2) &&
          (!solution_audit.relative_available ||
           std::isfinite(solution_audit.relative_error));
      if (!solution_audit.finite) {
        throw std::runtime_error(
            "cartesian_full_fmm_solution_audit_metric_nonfinite");
      }
      solution_audit.seconds =
          elapsed_ms(solution_audit_begin, std::chrono::steady_clock::now());
    }
    // "targets" is the manuscript's audit (section 2.6) on every surface and
    // cell, node patches included: m = min(sample count, N) target panels at
    // the indices floor((2k + 1) N / (2m)), both equations of each (2m rows),
    // of the solution, with the sampled direct residual.  A node-patch run
    // then reports it in place of its pre-GMRES probe, which stays the
    // fail-closed guard (fmm_cartesian_pre_audit_*).
    const bool target_audit = options.matvec_audit == "targets";
    if (experimental_treecode && options.matvec_audit != "off" &&
        (!node_patch_discretization || target_audit)) {
      const auto audit_begin = std::chrono::steady_clock::now();
      if (kifmm_any_gmres) {
        write_kifmm_gmres_phase("post_audit_start");
      }
      const std::vector<double> &audit_x =
          !linear.best_x.empty() ? linear.best_x : linear.x;
      std::vector<std::size_t> audit_rows;
      if (options.matvec_audit == "sampled-detail" || target_audit) {
        const std::vector<std::size_t> audit_targets = build_audit_rows(
            panels.size(), "sampled", options.matvec_audit_sample_count);
        audit_rows.reserve(audit_targets.size() * (coupled ? 2u : 1u));
        audit_rows.insert(audit_rows.end(), audit_targets.begin(),
                          audit_targets.end());
        if (coupled) {
          for (const std::size_t target : audit_targets)
            audit_rows.push_back(panels.size() + target);
        }
      } else {
        audit_rows = build_audit_rows(audit_x.size(), options.matvec_audit,
                                      options.matvec_audit_sample_count);
      }
      if (!audit_x.empty() && !audit_rows.empty()) {
        std::vector<double> tree_ax;
        if (cartesian_full_gmres) {
          apply_cartesian_full_operator(audit_x, tree_ax, false);
        } else if (channel_hybrid_gmres) {
          apply_channel_hybrid_operator(audit_x, tree_ax, false);
        } else if (kifmm_gmres) {
          apply_kifmm_operator(audit_x, tree_ax, false);
        } else if (gmres_p4) {
          apply_p4_operator(audit_x, tree_ax, false);
        } else {
          apply_jcp_treecode_operator(
              panels, audit_x, eps_ratio, kappa, lhs_phi, lhs_phin, coupled,
              jcp_tree, result.near_integration, result.quad_buffer, executor,
              tree_ax, nullptr);
        }
        const bool full = options.matvec_audit == "full-small" &&
                          audit_rows.size() == audit_x.size();
        double diff_sq = 0.0;
        double direct_sq = 0.0;
        double direct_residual_sq = 0.0;
        double rhs_sq = 0.0;
        if (full) {
          std::vector<double> direct_ax;
          if (coupled) {
            apply_jcp_linear_pb_operator(
                panels, audit_x, eps_ratio, kappa, lhs_phi, lhs_phin,
                result.near_integration, result.quad_buffer, executor,
                direct_ax, nullptr);
          } else {
            apply_jcp_poisson_operator(
                panels, audit_x, eps_ratio, lhs_phi, result.near_integration,
                result.quad_buffer, executor, direct_ax, nullptr);
          }
          for (std::size_t row : audit_rows) {
            if (row >= direct_ax.size() || row >= tree_ax.size())
              continue;
            const double delta = tree_ax[row] - direct_ax[row];
            diff_sq += delta * delta;
            direct_sq += direct_ax[row] * direct_ax[row];
            const double residual_value = rhs[row] - direct_ax[row];
            direct_residual_sq += residual_value * residual_value;
            rhs_sq += rhs[row] * rhs[row];
          }
        } else {
          const std::size_t unknown_count = audit_x.size();
          const std::size_t panel_count = panels.size();
          const std::size_t workers =
              solver_workers(executor, audit_rows.size());
          std::vector<double> partial_diff_sq(workers, 0.0);
          std::vector<double> partial_direct_sq(workers, 0.0);
          std::vector<double> partial_direct_residual_sq(workers, 0.0);
          std::vector<double> partial_rhs_sq(workers, 0.0);
          executor.parallel_for(
              audit_rows.size(), workers, 1,
              [&](std::size_t worker, std::size_t begin, std::size_t end) {
                double local_diff_sq = 0.0;
                double local_direct_sq = 0.0;
                double local_direct_residual_sq = 0.0;
                double local_rhs_sq = 0.0;
                for (std::size_t pos = begin; pos < end; ++pos) {
                  const std::size_t row = audit_rows[pos];
                  if (row >= unknown_count || row >= tree_ax.size())
                    continue;
                  const std::size_t target =
                      coupled && row >= panel_count ? row - panel_count : row;
                  if (target >= panel_count)
                    continue;
                  const JcpNearAccum sum = accumulate_jcp_near_ordered(
                      panels[target], target, panels, audit_x, eps_ratio, kappa,
                      coupled, direct_order, 0, panel_count,
                      result.near_integration, result.quad_buffer, nullptr);
                  double direct_row = 0.0;
                  if (coupled) {
                    direct_row =
                        row < panel_count
                            ? jcp_lhs_phi_for_panel(
                                  panels[target], eps_ratio, lhs_phi) *
                                  audit_x[target] - sum.phi
                            : jcp_lhs_phin_for_panel(
                                  panels[target], eps_ratio, lhs_phin) *
                                  audit_x[panel_count + target] - sum.phin;
                  } else {
                    direct_row = jcp_lhs_phi_for_panel(
                                     panels[target], eps_ratio, lhs_phi) *
                                     audit_x[target] -
                                 sum.phi;
                  }
                  const double delta = tree_ax[row] - direct_row;
                  local_diff_sq += delta * delta;
                  local_direct_sq += direct_row * direct_row;
                  const double residual_value = rhs[row] - direct_row;
                  local_direct_residual_sq += residual_value * residual_value;
                  local_rhs_sq += rhs[row] * rhs[row];
                }
                partial_diff_sq[worker] = local_diff_sq;
                partial_direct_sq[worker] = local_direct_sq;
                partial_direct_residual_sq[worker] = local_direct_residual_sq;
                partial_rhs_sq[worker] = local_rhs_sq;
              });
          diff_sq = std::accumulate(partial_diff_sq.begin(),
                                    partial_diff_sq.end(), 0.0);
          direct_sq = std::accumulate(partial_direct_sq.begin(),
                                      partial_direct_sq.end(), 0.0);
          direct_residual_sq =
              std::accumulate(partial_direct_residual_sq.begin(),
                              partial_direct_residual_sq.end(), 0.0);
          rhs_sq = std::accumulate(partial_rhs_sq.begin(), partial_rhs_sq.end(),
                                   0.0);
        }
        result.tree_vs_direct_matvec_rel =
            std::sqrt(std::max(0.0, diff_sq)) /
            std::max(1.0e-300, std::sqrt(std::max(0.0, direct_sq)));
        result.matvec_audit_rows = audit_rows.size();
        if (cartesian_full_gmres || gmres_p4 || kifmm_any_gmres ||
            target_audit) {
          result.fmm_local_sampled_direct_true_residual =
              std::sqrt(std::max(0.0, direct_residual_sq)) /
              std::max(1.0e-300, std::sqrt(std::max(0.0, rhs_sq)));
        }
        if (options.matvec_audit == "sampled-detail" &&
            !cartesian_full_gmres && !kifmm_any_gmres) {
          const auto context = ensure_jcp_fmm_m2l_context(
              panels, eps_ratio, kappa, jcp_tree.leaf_size,
              jcp_tree.opening_theta, jcp_tree.taylor_order, executor,
              &jcp_tree.fmm_m2l_context);
          std::vector<double> solution_phi(panels.size(), 0.0);
          std::vector<double> solution_phin(panels.size(), 0.0);
          std::vector<double> phi_probe(panels.size(), 0.0);
          std::vector<double> phin_probe(panels.size(), 0.0);
          for (std::size_t panel = 0; panel < panels.size(); ++panel) {
            solution_phi[panel] = audit_x[panel];
            solution_phin[panel] =
                coupled ? audit_x[panels.size() + panel] : 0.0;
            const double t = static_cast<double>(panel + 1);
            phi_probe[panel] = 0.5 * std::sin(0.013 * t) + 0.25;
            phin_probe[panel] = 0.5 * std::cos(0.017 * t) - 0.15;
          }
          const std::vector<double> zeros(panels.size(), 0.0);
          std::vector<titan::tabi::fmm_m2l::JcpOperatorAttributionResult>
              attribution;
          attribution.reserve(9);
          attribution.push_back(context->attribute(
              solution_phi, solution_phin,
              static_cast<std::size_t>(options.matvec_audit_sample_count),
              "solution", executor, coupled, result.near_integration,
              result.quad_buffer, &panels));
          attribution.push_back(context->attribute(
              phi_probe, zeros,
              static_cast<std::size_t>(options.matvec_audit_sample_count),
              "phi_only", executor, coupled, result.near_integration,
              result.quad_buffer, &panels));
          attribution.push_back(context->attribute(
              zeros, phin_probe,
              static_cast<std::size_t>(options.matvec_audit_sample_count),
              "phin_only", executor, coupled, result.near_integration,
              result.quad_buffer, &panels));
          if (options.fmm_cartesian_audit == "on") {
            attribution.push_back(context->attribute_cartesian(
                solution_phi, solution_phin,
                static_cast<std::size_t>(options.matvec_audit_sample_count),
                "solution", executor, coupled, result.near_integration,
                result.quad_buffer, &panels));
            attribution.push_back(context->attribute_cartesian(
                phi_probe, zeros,
                static_cast<std::size_t>(options.matvec_audit_sample_count),
                "phi_only", executor, coupled, result.near_integration,
                result.quad_buffer, &panels));
            attribution.push_back(context->attribute_cartesian(
                zeros, phin_probe,
                static_cast<std::size_t>(options.matvec_audit_sample_count),
                "phin_only", executor, coupled, result.near_integration,
                result.quad_buffer, &panels));
          }
          if (options.fmm_local == "audit-p4") {
            if (options.fmm_local_refine == "operator-tail-static" ||
                options.fmm_local_refine == "operator-tail-linf") {
              const std::size_t cache_budget_mb = static_cast<std::size_t>(
                  std::max(0, options.fmm_local_cache_budget_mb));
              attribution.push_back(context->attribute_local_p4_candidate(
                  solution_phi, solution_phin,
                  static_cast<std::size_t>(options.matvec_audit_sample_count),
                  "solution", executor, coupled, result.near_integration,
                  result.quad_buffer, &panels, options.fmm_local_refine,
                  options.fmm_local_refine_tol, options.fmm_local_cache,
                  cache_budget_mb, 1, options.fmm_local_fallback,
                  options.fmm_local_plan_builder,
                  static_cast<std::size_t>(
                      options.fmm_local_refine_probe_count),
                  options.fmm_local_refine_probe_mode,
                  options.fmm_local_geometry_guard,
                  options.fmm_local_geometry_theta));
              attribution.push_back(context->attribute_local_p4_candidate(
                  phi_probe, zeros,
                  static_cast<std::size_t>(options.matvec_audit_sample_count),
                  "phi_only", executor, coupled, result.near_integration,
                  result.quad_buffer, &panels, options.fmm_local_refine,
                  options.fmm_local_refine_tol, options.fmm_local_cache,
                  cache_budget_mb, 1, options.fmm_local_fallback,
                  options.fmm_local_plan_builder,
                  static_cast<std::size_t>(
                      options.fmm_local_refine_probe_count),
                  options.fmm_local_refine_probe_mode,
                  options.fmm_local_geometry_guard,
                  options.fmm_local_geometry_theta));
              attribution.push_back(context->attribute_local_p4_candidate(
                  zeros, phin_probe,
                  static_cast<std::size_t>(options.matvec_audit_sample_count),
                  "phin_only", executor, coupled, result.near_integration,
                  result.quad_buffer, &panels, options.fmm_local_refine,
                  options.fmm_local_refine_tol, options.fmm_local_cache,
                  cache_budget_mb, 1, options.fmm_local_fallback,
                  options.fmm_local_plan_builder,
                  static_cast<std::size_t>(
                      options.fmm_local_refine_probe_count),
                  options.fmm_local_refine_probe_mode,
                  options.fmm_local_geometry_guard,
                  options.fmm_local_geometry_theta));
            } else {
              attribution.push_back(context->attribute_local_p4(
                  solution_phi, solution_phin,
                  static_cast<std::size_t>(options.matvec_audit_sample_count),
                  "solution", executor, coupled, result.near_integration,
                  result.quad_buffer, &panels, options.fmm_local_refine,
                  options.fmm_local_refine_tol));
              attribution.push_back(context->attribute_local_p4(
                  phi_probe, zeros,
                  static_cast<std::size_t>(options.matvec_audit_sample_count),
                  "phi_only", executor, coupled, result.near_integration,
                  result.quad_buffer, &panels, options.fmm_local_refine,
                  options.fmm_local_refine_tol));
              attribution.push_back(context->attribute_local_p4(
                  zeros, phin_probe,
                  static_cast<std::size_t>(options.matvec_audit_sample_count),
                  "phin_only", executor, coupled, result.near_integration,
                  result.quad_buffer, &panels, options.fmm_local_refine,
                  options.fmm_local_refine_tol));
            }
          }
          if (options.fmm_local == "single-matvec-p4" ||
              options.fmm_local == "gmres-p4") {
            const std::size_t cache_budget_mb = static_cast<std::size_t>(
                std::max(0, options.fmm_local_cache_budget_mb));
            attribution.push_back(context->attribute_local_p4_candidate(
                solution_phi, solution_phin,
                static_cast<std::size_t>(options.matvec_audit_sample_count),
                "solution", executor, coupled, result.near_integration,
                result.quad_buffer, &panels, options.fmm_local_refine,
                options.fmm_local_refine_tol, options.fmm_local_cache,
                cache_budget_mb,
                options.fmm_local == "single-matvec-p4" ? 5u : 1u,
                options.fmm_local_fallback, options.fmm_local_plan_builder,
                static_cast<std::size_t>(options.fmm_local_refine_probe_count),
                options.fmm_local_refine_probe_mode,
                options.fmm_local_geometry_guard,
                options.fmm_local_geometry_theta));
            attribution.push_back(context->attribute_local_p4_candidate(
                phi_probe, zeros,
                static_cast<std::size_t>(options.matvec_audit_sample_count),
                "phi_only", executor, coupled, result.near_integration,
                result.quad_buffer, &panels, options.fmm_local_refine,
                options.fmm_local_refine_tol, options.fmm_local_cache,
                cache_budget_mb, 1, options.fmm_local_fallback,
                options.fmm_local_plan_builder,
                static_cast<std::size_t>(options.fmm_local_refine_probe_count),
                options.fmm_local_refine_probe_mode,
                options.fmm_local_geometry_guard,
                options.fmm_local_geometry_theta));
            attribution.push_back(context->attribute_local_p4_candidate(
                zeros, phin_probe,
                static_cast<std::size_t>(options.matvec_audit_sample_count),
                "phin_only", executor, coupled, result.near_integration,
                result.quad_buffer, &panels, options.fmm_local_refine,
                options.fmm_local_refine_tol, options.fmm_local_cache,
                cache_budget_mb, 1, options.fmm_local_fallback,
                options.fmm_local_plan_builder,
                static_cast<std::size_t>(options.fmm_local_refine_probe_count),
                options.fmm_local_refine_probe_mode,
                options.fmm_local_geometry_guard,
                options.fmm_local_geometry_theta));
          }
          write_operator_attribution_outputs(
              attribution, result.tree_vs_direct_matvec_rel, direct_sq);
        }
      }
      result.matvec_audit_ms =
          elapsed_ms(audit_begin, std::chrono::steady_clock::now());
      if (kifmm_any_gmres) {
        write_kifmm_gmres_phase("post_audit_done");
      }
    }
    if (cartesian_full_gmres || gmres_p4 || kifmm_any_gmres) {
      jcp_tree_stats.fmm_m2l_apply_ms =
          1000.0 * result.fmm_local_gmres_apply_seconds_total;
      jcp_tree_stats.fmm_m2l_finite_outputs = result.fmm_local_operator_finite;
    }
    if (experimental_treecode) {
      result.matvec_mode = jcp_tree_stats.matvec_mode;
      result.taylor_order = jcp_tree_stats.taylor_order;
      result.derivative_mode = jcp_tree_stats.derivative_mode;
      result.interaction_build_ms = jcp_tree_stats.interaction_build_ms;
      result.tree_eval_ms = jcp_tree_stats.tree_eval_ms;
      result.tree_moment_ms = jcp_tree_stats.tree_moment_ms;
      result.tree_traversal_ms = jcp_tree_stats.tree_traversal_ms;
      result.near_eval_ms = jcp_tree_stats.near_eval_ms;
      result.far_eval_ms = jcp_tree_stats.far_eval_ms;
      result.singular_near_eval_ms = jcp_tree_stats.singular_near_eval_ms;
      result.leaf_near_eval_ms = jcp_tree_stats.leaf_near_eval_ms;
      result.leaf_panel_eval_ms = jcp_tree_stats.leaf_panel_eval_ms;
      result.leaf_tile_eval_ms = jcp_tree_stats.leaf_tile_eval_ms;
      result.leaf_residual_eval_ms = jcp_tree_stats.leaf_residual_eval_ms;
      result.smooth_leaf_node_eval_ms = jcp_tree_stats.smooth_leaf_node_eval_ms;
      result.near_interactions = jcp_tree_stats.near_interactions;
      result.far_interactions = jcp_tree_stats.far_interactions;
      result.singular_near_interactions =
          jcp_tree_stats.singular_near_interactions;
      result.leaf_near_interactions = jcp_tree_stats.leaf_near_interactions;
      result.far_node_interactions = jcp_tree_stats.far_node_interactions;
      result.leaf_panel_interactions = jcp_tree_stats.leaf_panel_interactions;
      result.leaf_tile_interactions = jcp_tree_stats.leaf_tile_interactions;
      result.leaf_tile_panel_interactions =
          jcp_tree_stats.leaf_tile_panel_interactions;
      result.leaf_residual_panel_interactions =
          jcp_tree_stats.leaf_residual_panel_interactions;
      result.smooth_leaf_node_interactions =
          jcp_tree_stats.smooth_leaf_node_interactions;
      result.leaf_pruned_interactions = jcp_tree_stats.leaf_pruned_interactions;
      result.leaf_mac_candidates = jcp_tree_stats.leaf_mac_candidates;
      result.leaf_mac_accepted = jcp_tree_stats.leaf_mac_accepted;
      result.leaf_mac_rejected = jcp_tree_stats.leaf_mac_rejected;
      result.leaf_mac_far_panels = jcp_tree_stats.leaf_mac_far_panels;
      result.leaf_mac_direct_panels_saved_estimate =
          jcp_tree_stats.leaf_mac_direct_panels_saved_estimate;
      result.fmm_m2l_mode = jcp_tree_stats.fmm_m2l_mode;
      result.fmm_m2l_expansion_order = jcp_tree_stats.fmm_m2l_expansion_order;
      result.fmm_m2l_near_pairs = jcp_tree_stats.fmm_m2l_near_pairs;
      result.fmm_m2l_far_pairs = jcp_tree_stats.fmm_m2l_far_pairs;
      result.fmm_m2l_setup_ms = jcp_tree_stats.fmm_m2l_setup_ms;
      result.fmm_m2l_apply_ms = jcp_tree_stats.fmm_m2l_apply_ms;
      result.fmm_m2l_finite_outputs = jcp_tree_stats.fmm_m2l_finite_outputs;
      result.leaf_work_profile = jcp_tree_stats.leaf_work_profile;
      result.leaf_work_mac_terminal_panels =
          jcp_tree_stats.leaf_work_mac_terminal_panels;
      result.leaf_work_physical_gap_panels =
          jcp_tree_stats.leaf_work_physical_gap_panels;
      result.leaf_work_fragmentation_panels =
          jcp_tree_stats.leaf_work_fragmentation_panels;
      result.leaf_work_convergence_panels =
          jcp_tree_stats.leaf_work_convergence_panels;
      result.leaf_work_convergence_estimator_panels =
          jcp_tree_stats.leaf_work_convergence_estimator_panels;
      result.leaf_work_convergence_rho_panels =
          jcp_tree_stats.leaf_work_convergence_rho_panels;
      result.leaf_work_convergence_aspect_panels =
          jcp_tree_stats.leaf_work_convergence_aspect_panels;
      result.leaf_work_surface_margin_panels =
          jcp_tree_stats.leaf_work_surface_margin_panels;
      result.leaf_work_surface_margin_hard_panels =
          jcp_tree_stats.leaf_work_surface_margin_hard_panels;
      result.leaf_work_surface_margin_soft_panels =
          jcp_tree_stats.leaf_work_surface_margin_soft_panels;
      result.leaf_work_k3_rho_cap_panels =
          jcp_tree_stats.leaf_work_k3_rho_cap_panels;
      result.leaf_work_singular_fallback_panels =
          jcp_tree_stats.leaf_work_singular_fallback_panels;
      result.leaf_work_scalar_tail_panels =
          jcp_tree_stats.leaf_work_scalar_tail_panels;
      result.leaf_work_profile_accounted_panels =
          jcp_tree_stats.leaf_work_profile_accounted_panels;
      result.leaf_work_mac_terminal_frac =
          jcp_tree_stats.leaf_work_mac_terminal_frac;
      result.leaf_work_convergence_frac =
          jcp_tree_stats.leaf_work_convergence_frac;
      result.leaf_work_surface_margin_frac =
          jcp_tree_stats.leaf_work_surface_margin_frac;
      result.leaf_work_singular_fallback_frac =
          jcp_tree_stats.leaf_work_singular_fallback_frac;
      result.leaf_work_scalar_tail_frac =
          jcp_tree_stats.leaf_work_scalar_tail_frac;
      result.adaptive_smooth_leaf_accepted =
          jcp_tree_stats.adaptive_smooth_leaf_accepted;
      result.adaptive_smooth_leaf_rejected =
          jcp_tree_stats.adaptive_smooth_leaf_rejected;
      result.adaptive_span_tile_accepted =
          jcp_tree_stats.adaptive_span_tile_accepted;
      result.adaptive_span_tile_rejected =
          jcp_tree_stats.adaptive_span_tile_rejected;
      result.adaptive_exact_leaf_fallbacks =
          jcp_tree_stats.adaptive_exact_leaf_fallbacks;
      result.adaptive_reject_one_ring_overlap =
          jcp_tree_stats.adaptive_reject_one_ring_overlap;
      result.adaptive_reject_error_bound_failed =
          jcp_tree_stats.adaptive_reject_error_bound_failed;
      result.adaptive_reject_p4_check_failed =
          jcp_tree_stats.adaptive_reject_p4_check_failed;
      result.adaptive_reject_too_close =
          jcp_tree_stats.adaptive_reject_too_close;
      result.leaf_work_reduction_ratio =
          jcp_tree_stats.leaf_work_reduction_ratio;
      result.target_cluster_interactions =
          jcp_tree_stats.target_cluster_interactions;
      result.target_cluster_targets = jcp_tree_stats.target_cluster_targets;
      result.target_cluster_panels_pruned =
          jcp_tree_stats.target_cluster_panels_pruned;
      result.target_cluster_rejected = jcp_tree_stats.target_cluster_rejected;
      result.target_cluster_eval_ms = jcp_tree_stats.target_cluster_eval_ms;
      result.leaf_pair_batch_interactions =
          jcp_tree_stats.leaf_pair_batch_interactions;
      result.leaf_pair_batch_targets = jcp_tree_stats.leaf_pair_batch_targets;
      result.leaf_pair_batch_source_panels =
          jcp_tree_stats.leaf_pair_batch_source_panels;
      result.leaf_pair_batch_residual_panels =
          jcp_tree_stats.leaf_pair_batch_residual_panels;
      result.leaf_pair_batch_fallbacks =
          jcp_tree_stats.leaf_pair_batch_fallbacks;
      result.leaf_pair_batch_ms = jcp_tree_stats.leaf_pair_batch_ms;
      result.leaf_pair_batch_kernel = jcp_tree_stats.leaf_pair_batch_kernel;
      result.leaf_pair_batch_vectorized_sources =
          jcp_tree_stats.leaf_pair_batch_vectorized_sources;
      result.leaf_pair_batch_vector_lanes =
          jcp_tree_stats.leaf_pair_batch_vector_lanes;
      result.leaf_pair_batch_pairs_per_second =
          jcp_tree_stats.leaf_pair_batch_pairs_per_second;
      result.leaf_kernel_profile = jcp_tree_stats.leaf_kernel_profile;
      result.leaf_work_profile = jcp_tree_stats.leaf_work_profile;
      result.jcp_direct_kernel = jcp_tree_stats.jcp_direct_kernel;
      result.leaf_kernel_scalar_tasks = jcp_tree_stats.leaf_kernel_scalar_tasks;
      result.leaf_kernel_batch_tasks = jcp_tree_stats.leaf_kernel_batch_tasks;
      result.leaf_kernel_source_panel_evals =
          jcp_tree_stats.leaf_kernel_source_panel_evals;
      result.leaf_kernel_scalar_ms = jcp_tree_stats.leaf_kernel_scalar_ms;
      result.leaf_kernel_target_avx2_ms =
          jcp_tree_stats.leaf_kernel_target_avx2_ms;
      result.leaf_kernel_tail_ms = jcp_tree_stats.leaf_kernel_tail_ms;
      result.leaf_pair_batch_soa_sources =
          jcp_tree_stats.leaf_pair_batch_soa_sources;
      result.leaf_pair_batch_soa_ms = jcp_tree_stats.leaf_pair_batch_soa_ms;
      result.leaf_pair_batch_scalar_tail_ms =
          jcp_tree_stats.leaf_pair_batch_scalar_tail_ms;
      result.leaf_pair_batch_clean_source_panels_per_second =
          jcp_tree_stats.leaf_pair_batch_clean_source_panels_per_second;
      result.leaf_pair_selection_profile =
          jcp_tree_stats.leaf_pair_selection_profile;
      result.leaf_pair_probe_cache = jcp_tree_stats.leaf_pair_probe_cache;
      result.leaf_pair_target_group = jcp_tree_stats.leaf_pair_target_group;
      result.leaf_pair_directness_probe =
          jcp_tree_stats.leaf_pair_directness_probe;
      result.leaf_pair_directness_cache =
          jcp_tree_stats.leaf_pair_directness_cache;
      result.leaf_pair_directness_verify =
          jcp_tree_stats.leaf_pair_directness_verify;
      result.leaf_pair_max_lookahead = jcp_tree_stats.leaf_pair_max_lookahead;
      result.leaf_pair_selection_candidate_scans =
          jcp_tree_stats.leaf_pair_selection_candidate_scans;
      result.leaf_pair_selection_clean_rejects =
          jcp_tree_stats.leaf_pair_selection_clean_rejects;
      result.leaf_pair_selection_probe_calls =
          jcp_tree_stats.leaf_pair_selection_probe_calls;
      result.leaf_pair_selection_probe_node_visits =
          jcp_tree_stats.leaf_pair_selection_probe_node_visits;
      result.leaf_pair_selection_probe_ms =
          jcp_tree_stats.leaf_pair_selection_probe_ms;
      result.leaf_pair_selection_ms = jcp_tree_stats.leaf_pair_selection_ms;
      result.leaf_pair_selection_probe_cache_hits =
          jcp_tree_stats.leaf_pair_selection_probe_cache_hits;
      result.leaf_pair_selection_probe_cache_misses =
          jcp_tree_stats.leaf_pair_selection_probe_cache_misses;
      result.leaf_pair_selection_insufficient_targets =
          jcp_tree_stats.leaf_pair_selection_insufficient_targets;
      result.leaf_pair_selection_same_target_leaf_rejects =
          jcp_tree_stats.leaf_pair_selection_same_target_leaf_rejects;
      result.leaf_pair_directness_ancestor_probe_calls =
          jcp_tree_stats.leaf_pair_directness_ancestor_probe_calls;
      result.leaf_pair_directness_ancestor_probe_node_visits =
          jcp_tree_stats.leaf_pair_directness_ancestor_probe_node_visits;
      result.leaf_pair_directness_ancestor_probe_ms =
          jcp_tree_stats.leaf_pair_directness_ancestor_probe_ms;
      result.leaf_pair_directness_pair_cache_hits =
          jcp_tree_stats.leaf_pair_directness_pair_cache_hits;
      result.leaf_pair_directness_pair_cache_misses =
          jcp_tree_stats.leaf_pair_directness_pair_cache_misses;
      result.leaf_pair_directness_all_direct_proofs =
          jcp_tree_stats.leaf_pair_directness_all_direct_proofs;
      result.leaf_pair_directness_mixed_fallbacks =
          jcp_tree_stats.leaf_pair_directness_mixed_fallbacks;
      result.leaf_pair_directness_verify_checks =
          jcp_tree_stats.leaf_pair_directness_verify_checks;
      result.leaf_pair_directness_verify_mismatches =
          jcp_tree_stats.leaf_pair_directness_verify_mismatches;
      result.leaf_pair_scheduler = jcp_tree_stats.leaf_pair_scheduler;
      result.leaf_pair_worklist_max_active =
          jcp_tree_stats.leaf_pair_worklist_max_active;
      result.leaf_pair_worklist_verify =
          jcp_tree_stats.leaf_pair_worklist_verify;
      result.leaf_pair_worklist_tasks = jcp_tree_stats.leaf_pair_worklist_tasks;
      result.leaf_pair_worklist_groups =
          jcp_tree_stats.leaf_pair_worklist_groups;
      result.leaf_pair_worklist_batched_groups =
          jcp_tree_stats.leaf_pair_worklist_batched_groups;
      result.leaf_pair_worklist_scalar_flush_tasks =
          jcp_tree_stats.leaf_pair_worklist_scalar_flush_tasks;
      result.leaf_pair_worklist_forced_flushes =
          jcp_tree_stats.leaf_pair_worklist_forced_flushes;
      result.leaf_pair_worklist_active_groups_max =
          jcp_tree_stats.leaf_pair_worklist_active_groups_max;
      result.leaf_pair_worklist_ms = jcp_tree_stats.leaf_pair_worklist_ms;
      result.leaf_pair_worklist_verify_checks =
          jcp_tree_stats.leaf_pair_worklist_verify_checks;
      result.leaf_pair_worklist_verify_mismatches =
          jcp_tree_stats.leaf_pair_worklist_verify_mismatches;
      result.leaf_block_verify = jcp_tree_stats.leaf_block_verify;
      result.leaf_block_target_leaves = jcp_tree_stats.leaf_block_target_leaves;
      result.leaf_block_targets = jcp_tree_stats.leaf_block_targets;
      result.leaf_block_source_groups = jcp_tree_stats.leaf_block_source_groups;
      result.leaf_block_batched_groups =
          jcp_tree_stats.leaf_block_batched_groups;
      result.leaf_block_scalar_tail_tasks =
          jcp_tree_stats.leaf_block_scalar_tail_tasks;
      result.leaf_block_group_ms = jcp_tree_stats.leaf_block_group_ms;
      result.leaf_block_flush_ms = jcp_tree_stats.leaf_block_flush_ms;
      result.leaf_block_writeback_ms = jcp_tree_stats.leaf_block_writeback_ms;
      result.leaf_block_verify_checks = jcp_tree_stats.leaf_block_verify_checks;
      result.leaf_block_verify_mismatches =
          jcp_tree_stats.leaf_block_verify_mismatches;
      result.adaptive_level = jcp_tree_stats.adaptive_level;
      result.cluster_policy = jcp_tree_stats.cluster_policy;
      result.target_batch_size = jcp_tree_stats.target_batch_size;
      result.leaf_pair_min_source_panels =
          jcp_tree_stats.leaf_pair_min_source_panels;
      result.leaf_pair_min_targets = jcp_tree_stats.leaf_pair_min_targets;
      result.leaf_near_policy = jcp_tree_stats.leaf_near_policy;
      result.leaf_panel_kernel = jcp_tree_stats.leaf_panel_kernel;
      result.interaction_mode = jcp_tree_stats.interaction_mode;
      result.node_bounds_mode = jcp_tree_stats.node_bounds_mode;
      result.tree_builder = jcp_tree_stats.tree_builder;
      result.expansion_center = jcp_tree_stats.expansion_center;
      result.fragmentation_safety = jcp_tree_stats.fragmentation_safety;
      result.mac_safety = jcp_tree_stats.mac_safety;
      result.mac_gap_factor = jcp_tree_stats.mac_gap_factor;
      result.tree_depth_max = jcp_tree_stats.tree_depth_max;
      result.mac_reject_radius_limited =
          jcp_tree_stats.mac_reject_radius_limited;
      result.mac_accept_rate = jcp_tree_stats.mac_accept_rate;
      result.node_radius_max = jcp_tree_stats.node_radius_max;
      result.node_radius_mean = jcp_tree_stats.node_radius_mean;
      result.node_aspect_ratio_max = jcp_tree_stats.node_aspect_ratio_max;
      result.node_aspect_ratio_mean = jcp_tree_stats.node_aspect_ratio_mean;
      result.leaf_panel_work_per_matvec =
          jcp_tree_stats.leaf_panel_work_per_matvec;
      result.far_node_work_per_matvec = jcp_tree_stats.far_node_work_per_matvec;
      result.far_geometric_unsafe_count =
          jcp_tree_stats.far_geometric_unsafe_count;
      result.far_geometric_unsafe_panels =
          jcp_tree_stats.far_geometric_unsafe_panels;
      result.far_geometric_min_gap = jcp_tree_stats.far_geometric_min_gap;
      result.far_geometric_gap_ratio_min =
          jcp_tree_stats.far_geometric_gap_ratio_min;
      result.far_geometric_gap_ratio_mean =
          jcp_tree_stats.far_geometric_gap_ratio_mean;
      result.far_geometric_audit_samples =
          jcp_tree_stats.far_geometric_audit_samples;
      result.mac_physical_gap_rejects = jcp_tree_stats.mac_physical_gap_rejects;
      result.mac_convergence_rho_rejects =
          jcp_tree_stats.mac_convergence_rho_rejects;
      result.mac_convergence_aspect_rejects =
          jcp_tree_stats.mac_convergence_aspect_rejects;
      result.mac_convergence_rho_reject_panels =
          jcp_tree_stats.mac_convergence_rho_reject_panels;
      result.mac_convergence_aspect_reject_panels =
          jcp_tree_stats.mac_convergence_aspect_reject_panels;
      result.fragmented_far_count = jcp_tree_stats.fragmented_far_count;
      result.fragmented_far_panels = jcp_tree_stats.fragmented_far_panels;
      result.fragmented_split_rejects = jcp_tree_stats.fragmented_split_rejects;
      result.fragmentation_center_gap_ratio_max =
          jcp_tree_stats.fragmentation_center_gap_ratio_max;
      result.fragmentation_center_gap_ratio_mean =
          jcp_tree_stats.fragmentation_center_gap_ratio_mean;
      result.moment_audit = jcp_tree_stats.moment_audit;
      result.moment_construction = jcp_tree_stats.moment_construction;
      result.moment_sum = jcp_tree_stats.moment_sum;
      result.convergence_audit = jcp_tree_stats.convergence_audit;
      result.convergence_safety = jcp_tree_stats.convergence_safety;
      result.convergence_rho_cap = jcp_tree_stats.convergence_rho_cap;
      result.mac_aspect_cap = jcp_tree_stats.mac_aspect_cap;
      result.convergence_estimator = jcp_tree_stats.convergence_estimator;
      result.convergence_tail_threshold =
          jcp_tree_stats.convergence_tail_threshold;
      result.convergence_estimator_error_cap =
          jcp_tree_stats.convergence_estimator_error_cap;
      result.convergence_unstable_rho_cap =
          jcp_tree_stats.convergence_unstable_rho_cap;
      result.convergence_calibration = jcp_tree_stats.convergence_calibration;
      result.convergence_dielectric_amp =
          jcp_tree_stats.convergence_dielectric_amp;
      result.convergence_estimator_model =
          jcp_tree_stats.convergence_estimator_model;
      result.convergence_channel_calibration =
          jcp_tree_stats.convergence_channel_calibration;
      result.convergence_split_policy = jcp_tree_stats.convergence_split_policy;
      result.convergence_split_selectivity =
          jcp_tree_stats.convergence_split_selectivity;
      result.convergence_split_absolute_floor =
          jcp_tree_stats.convergence_split_absolute_floor;
      result.convergence_overreject_audit =
          jcp_tree_stats.convergence_overreject_audit;
      result.convergence_overreject_detail =
          jcp_tree_stats.convergence_overreject_detail;
      result.convergence_shape_audit = jcp_tree_stats.convergence_shape_audit;
      result.convergence_kernel_response_audit =
          jcp_tree_stats.convergence_kernel_response_audit;
      result.channel_missed_semantics = jcp_tree_stats.channel_missed_semantics;
      result.accepted_miss_diagnostics =
          jcp_tree_stats.accepted_miss_diagnostics;
      result.accepted_miss_absolute_floor =
          jcp_tree_stats.accepted_miss_absolute_floor;
      result.convergence_k3_accepted_rho_cap =
          jcp_tree_stats.convergence_k3_accepted_rho_cap;
      result.mac_distance_policy = jcp_tree_stats.mac_distance_policy;
      result.mac_surface_margin_factor =
          jcp_tree_stats.mac_surface_margin_factor;
      result.mac_surface_margin_policy =
          jcp_tree_stats.mac_surface_margin_policy;
      result.mac_surface_margin_aspect_threshold =
          jcp_tree_stats.mac_surface_margin_aspect_threshold;
      result.mac_surface_margin_rho_trigger =
          jcp_tree_stats.mac_surface_margin_rho_trigger;
      result.mac_surface_margin_soft_aspect_threshold =
          jcp_tree_stats.mac_surface_margin_soft_aspect_threshold;
      result.mac_surface_margin_soft_rho_trigger =
          jcp_tree_stats.mac_surface_margin_soft_rho_trigger;
      result.mac_surface_margin_soft_factor =
          jcp_tree_stats.mac_surface_margin_soft_factor;
      result.convergence_estimator_samples =
          jcp_tree_stats.convergence_estimator_samples;
      result.convergence_estimator_unstable_far_count =
          jcp_tree_stats.convergence_estimator_unstable_far_count;
      result.convergence_estimator_unstable_far_panels =
          jcp_tree_stats.convergence_estimator_unstable_far_panels;
      result.convergence_estimator_split_rejects =
          jcp_tree_stats.convergence_estimator_split_rejects;
      result.convergence_estimator_split_reject_panels =
          jcp_tree_stats.convergence_estimator_split_reject_panels;
      result.convergence_low_abs_reaccepted_far_count =
          jcp_tree_stats.convergence_low_abs_reaccepted_far_count;
      result.convergence_low_abs_reaccepted_far_panels =
          jcp_tree_stats.convergence_low_abs_reaccepted_far_panels;
      result.convergence_low_abs_reaccepted_far_count =
          jcp_tree_stats.convergence_low_abs_reaccepted_far_count;
      result.convergence_low_abs_reaccepted_far_panels =
          jcp_tree_stats.convergence_low_abs_reaccepted_far_panels;
      result.convergence_estimator_tail_ratio_max =
          jcp_tree_stats.convergence_estimator_tail_ratio_max;
      result.convergence_estimator_tail_ratio_mean =
          jcp_tree_stats.convergence_estimator_tail_ratio_mean;
      result.convergence_estimator_error_max =
          jcp_tree_stats.convergence_estimator_error_max;
      result.convergence_estimator_error_mean =
          jcp_tree_stats.convergence_estimator_error_mean;
      result.convergence_estimator_abs_error_max =
          jcp_tree_stats.convergence_estimator_abs_error_max;
      result.convergence_estimator_abs_error_mean =
          jcp_tree_stats.convergence_estimator_abs_error_mean;
      result.convergence_estimator_abs_error_max =
          jcp_tree_stats.convergence_estimator_abs_error_max;
      result.convergence_estimator_abs_error_mean =
          jcp_tree_stats.convergence_estimator_abs_error_mean;
      result.convergence_tail_norm_p_max =
          jcp_tree_stats.convergence_tail_norm_p_max;
      result.convergence_tail_norm_p_mean =
          jcp_tree_stats.convergence_tail_norm_p_mean;
      result.convergence_tail_norm_pm1_max =
          jcp_tree_stats.convergence_tail_norm_pm1_max;
      result.convergence_tail_norm_pm1_mean =
          jcp_tree_stats.convergence_tail_norm_pm1_mean;
      result.convergence_effective_jcp_amp_max =
          jcp_tree_stats.convergence_effective_jcp_amp_max;
      result.convergence_effective_jcp_amp_mean =
          jcp_tree_stats.convergence_effective_jcp_amp_mean;
      result.convergence_dielectric_amp_max =
          jcp_tree_stats.convergence_dielectric_amp_max;
      result.convergence_dielectric_amp_mean =
          jcp_tree_stats.convergence_dielectric_amp_mean;
      result.convergence_channel_estimated_error_max =
          jcp_tree_stats.convergence_channel_estimated_error_max;
      result.convergence_channel_estimated_error_mean =
          jcp_tree_stats.convergence_channel_estimated_error_mean;
      result.convergence_channel_effective_jcp_amp_max =
          jcp_tree_stats.convergence_channel_effective_jcp_amp_max;
      result.convergence_channel_effective_jcp_amp_mean =
          jcp_tree_stats.convergence_channel_effective_jcp_amp_mean;
      result.convergence_channel_unstable_far_count =
          jcp_tree_stats.convergence_channel_unstable_far_count;
      result.convergence_channel_unstable_far_panels =
          jcp_tree_stats.convergence_channel_unstable_far_panels;
      result.convergence_channel_split_rejects =
          jcp_tree_stats.convergence_channel_split_rejects;
      result.convergence_channel_split_reject_panels =
          jcp_tree_stats.convergence_channel_split_reject_panels;
      result.convergence_channel_low_abs_reaccepted_count =
          jcp_tree_stats.convergence_channel_low_abs_reaccepted_count;
      result.convergence_channel_low_abs_reaccepted_panels =
          jcp_tree_stats.convergence_channel_low_abs_reaccepted_panels;
      result.convergence_channel_high_error_caught =
          jcp_tree_stats.convergence_channel_high_error_caught;
      result.convergence_channel_high_error_missed =
          jcp_tree_stats.convergence_channel_high_error_missed;
      result.convergence_channel_high_error_caught_by_split =
          jcp_tree_stats.convergence_channel_high_error_caught_by_split;
      result.convergence_estimator_high_error_caught =
          jcp_tree_stats.convergence_estimator_high_error_caught;
      result.convergence_estimator_high_error_missed =
          jcp_tree_stats.convergence_estimator_high_error_missed;
      result.convergence_estimator_high_error_caught_by_split =
          jcp_tree_stats.convergence_estimator_high_error_caught_by_split;
      result.convergence_estimator_phantom_high_error_count =
          jcp_tree_stats.convergence_estimator_phantom_high_error_count;
      result.convergence_estimator_phantom_high_error_abs_max =
          jcp_tree_stats.convergence_estimator_phantom_high_error_abs_max;
      result.convergence_overreject_count =
          jcp_tree_stats.convergence_overreject_count;
      result.convergence_overreject_panels =
          jcp_tree_stats.convergence_overreject_panels;
      result.convergence_overreject_caught_count =
          jcp_tree_stats.convergence_overreject_caught_count;
      result.convergence_overreject_caught_panels =
          jcp_tree_stats.convergence_overreject_caught_panels;
      result.convergence_overreject_abs_error_max =
          jcp_tree_stats.convergence_overreject_abs_error_max;
      result.convergence_overreject_abs_error_mean =
          jcp_tree_stats.convergence_overreject_abs_error_mean;
      result.convergence_overreject_abs_error_p95 =
          jcp_tree_stats.convergence_overreject_abs_error_p95;
      result.convergence_overreject_abs_error_p99 =
          jcp_tree_stats.convergence_overreject_abs_error_p99;
      result.convergence_overreject_estimated_abs_error_max =
          jcp_tree_stats.convergence_overreject_estimated_abs_error_max;
      result.convergence_overreject_estimated_abs_error_mean =
          jcp_tree_stats.convergence_overreject_estimated_abs_error_mean;
      result.convergence_overreject_required_scale_max =
          jcp_tree_stats.convergence_overreject_required_scale_max;
      result.convergence_overreject_required_scale_p95 =
          jcp_tree_stats.convergence_overreject_required_scale_p95;
      result.convergence_overreject_required_scale_p99 =
          jcp_tree_stats.convergence_overreject_required_scale_p99;
      result.convergence_channel_overreject_count =
          jcp_tree_stats.convergence_channel_overreject_count;
      result.convergence_channel_overreject_caught_count =
          jcp_tree_stats.convergence_channel_overreject_caught_count;
      result.convergence_channel_overreject_panels =
          jcp_tree_stats.convergence_channel_overreject_panels;
      result.convergence_channel_overreject_abs_error_max =
          jcp_tree_stats.convergence_channel_overreject_abs_error_max;
      result.convergence_channel_overreject_abs_error_mean =
          jcp_tree_stats.convergence_channel_overreject_abs_error_mean;
      result.convergence_channel_overreject_required_scale_max =
          jcp_tree_stats.convergence_channel_overreject_required_scale_max;
      result.convergence_channel_overreject_required_scale_p99 =
          jcp_tree_stats.convergence_channel_overreject_required_scale_p99;
      result.phantom_high_error_channel_count =
          jcp_tree_stats.phantom_high_error_channel_count;
      result.phantom_high_error_channel_abs_max =
          jcp_tree_stats.phantom_high_error_channel_abs_max;
      result.accepted_miss_channel_count =
          jcp_tree_stats.accepted_miss_channel_count;
      result.accepted_miss_channel_panels =
          jcp_tree_stats.accepted_miss_channel_panels;
      result.accepted_miss_channel_rho_min =
          jcp_tree_stats.accepted_miss_channel_rho_min;
      result.accepted_miss_channel_rho_mean =
          jcp_tree_stats.accepted_miss_channel_rho_mean;
      result.accepted_miss_channel_rho_max =
          jcp_tree_stats.accepted_miss_channel_rho_max;
      result.accepted_miss_channel_tail_ratio_mean =
          jcp_tree_stats.accepted_miss_channel_tail_ratio_mean;
      result.accepted_miss_channel_tail_ratio_max =
          jcp_tree_stats.accepted_miss_channel_tail_ratio_max;
      result.accepted_miss_channel_estimated_error_mean =
          jcp_tree_stats.accepted_miss_channel_estimated_error_mean;
      result.accepted_miss_channel_estimated_error_max =
          jcp_tree_stats.accepted_miss_channel_estimated_error_max;
      result.accepted_miss_worst_channel =
          jcp_tree_stats.accepted_miss_worst_channel;
      result.convergence_k3_rho_cap_rejects =
          jcp_tree_stats.convergence_k3_rho_cap_rejects;
      result.convergence_k3_rho_cap_reject_panels =
          jcp_tree_stats.convergence_k3_rho_cap_reject_panels;
      result.mac_surface_margin_rejects =
          jcp_tree_stats.mac_surface_margin_rejects;
      result.mac_surface_margin_reject_panels =
          jcp_tree_stats.mac_surface_margin_reject_panels;
      result.mac_surface_margin_applied =
          jcp_tree_stats.mac_surface_margin_applied;
      result.mac_surface_margin_skipped =
          jcp_tree_stats.mac_surface_margin_skipped;
      result.mac_surface_margin_adaptive_rejects =
          jcp_tree_stats.mac_surface_margin_adaptive_rejects;
      result.mac_surface_margin_adaptive_reject_panels =
          jcp_tree_stats.mac_surface_margin_adaptive_reject_panels;
      result.mac_surface_margin_skipped_accepted_far =
          jcp_tree_stats.mac_surface_margin_skipped_accepted_far;
      result.mac_surface_margin_hard_applied =
          jcp_tree_stats.mac_surface_margin_hard_applied;
      result.mac_surface_margin_soft_applied =
          jcp_tree_stats.mac_surface_margin_soft_applied;
      result.mac_surface_margin_hard_rejects =
          jcp_tree_stats.mac_surface_margin_hard_rejects;
      result.mac_surface_margin_hard_reject_panels =
          jcp_tree_stats.mac_surface_margin_hard_reject_panels;
      result.mac_surface_margin_soft_rejects =
          jcp_tree_stats.mac_surface_margin_soft_rejects;
      result.mac_surface_margin_soft_reject_panels =
          jcp_tree_stats.mac_surface_margin_soft_reject_panels;
      result.moment_audit_nodes = jcp_tree_stats.moment_audit_nodes;
      result.moment_translation_rel_error_max =
          jcp_tree_stats.moment_translation_rel_error_max;
      result.moment_translation_rel_error_mean =
          jcp_tree_stats.moment_translation_rel_error_mean;
      result.moment_translation_offset_max =
          jcp_tree_stats.moment_translation_offset_max;
      result.moment_translation_offset_mean =
          jcp_tree_stats.moment_translation_offset_mean;
      result.moment_cancellation_ratio_max =
          jcp_tree_stats.moment_cancellation_ratio_max;
      result.moment_cancellation_ratio_mean =
          jcp_tree_stats.moment_cancellation_ratio_mean;
      result.moment_dynamic_range_max = jcp_tree_stats.moment_dynamic_range_max;
      result.expansion_audit = jcp_tree_stats.expansion_audit;
      result.expansion_audit_channel = jcp_tree_stats.expansion_audit_channel;
      result.far_expansion_audit_pairs =
          jcp_tree_stats.far_expansion_audit_pairs;
      result.far_expansion_audit_panels =
          jcp_tree_stats.far_expansion_audit_panels;
      result.far_expansion_direct_ms = jcp_tree_stats.far_expansion_direct_ms;
      result.far_expansion_rel_error_max =
          jcp_tree_stats.far_expansion_rel_error_max;
      result.far_expansion_rel_error_mean =
          jcp_tree_stats.far_expansion_rel_error_mean;
      result.far_expansion_rel_error_p95 =
          jcp_tree_stats.far_expansion_rel_error_p95;
      result.far_expansion_rel_error_p99 =
          jcp_tree_stats.far_expansion_rel_error_p99;
      result.far_expansion_channel_rel_error_max =
          jcp_tree_stats.far_expansion_channel_rel_error_max;
      result.far_expansion_channel_rel_error_mean =
          jcp_tree_stats.far_expansion_channel_rel_error_mean;
      result.far_expansion_channel_rel_error_p99 =
          jcp_tree_stats.far_expansion_channel_rel_error_p99;
      result.far_expansion_worst_channel =
          jcp_tree_stats.far_expansion_worst_channel;
      result.far_expansion_mean_gate_pass =
          jcp_tree_stats.far_expansion_mean_gate_pass;
      result.far_convergence_rho_max = jcp_tree_stats.far_convergence_rho_max;
      result.far_convergence_rho_mean = jcp_tree_stats.far_convergence_rho_mean;
      result.accepted_far_aspect_ratio_max =
          jcp_tree_stats.accepted_far_aspect_ratio_max;
      result.accepted_far_aspect_ratio_mean =
          jcp_tree_stats.accepted_far_aspect_ratio_mean;
      result.leaf_tile_status = jcp_tree_stats.leaf_tile_status;
      result.leaf_panel_pairs_per_second =
          jcp_tree_stats.leaf_panel_pairs_per_second;
    } else {
      result.near_eval_ms = jcp_tree_stats.near_eval_ms;
      result.near_interactions = jcp_tree_stats.near_interactions;
      result.leaf_near_policy = jcp_tree_stats.leaf_near_policy;
      result.leaf_panel_kernel = jcp_tree_stats.leaf_panel_kernel;
      result.interaction_mode = jcp_tree_stats.interaction_mode;
      result.node_bounds_mode = jcp_tree_stats.node_bounds_mode;
      result.tree_builder = jcp_tree_stats.tree_builder;
      result.expansion_center = jcp_tree_stats.expansion_center;
      result.fragmentation_safety = jcp_tree_stats.fragmentation_safety;
      result.mac_safety = jcp_tree_stats.mac_safety;
      result.mac_gap_factor = jcp_tree_stats.mac_gap_factor;
      result.tree_depth_max = jcp_tree_stats.tree_depth_max;
      result.mac_reject_radius_limited =
          jcp_tree_stats.mac_reject_radius_limited;
      result.mac_accept_rate = jcp_tree_stats.mac_accept_rate;
      result.node_radius_max = jcp_tree_stats.node_radius_max;
      result.node_radius_mean = jcp_tree_stats.node_radius_mean;
      result.node_aspect_ratio_max = jcp_tree_stats.node_aspect_ratio_max;
      result.node_aspect_ratio_mean = jcp_tree_stats.node_aspect_ratio_mean;
      result.leaf_panel_work_per_matvec =
          jcp_tree_stats.leaf_panel_work_per_matvec;
      result.far_node_work_per_matvec = jcp_tree_stats.far_node_work_per_matvec;
      result.far_geometric_unsafe_count =
          jcp_tree_stats.far_geometric_unsafe_count;
      result.far_geometric_unsafe_panels =
          jcp_tree_stats.far_geometric_unsafe_panels;
      result.far_geometric_min_gap = jcp_tree_stats.far_geometric_min_gap;
      result.far_geometric_gap_ratio_min =
          jcp_tree_stats.far_geometric_gap_ratio_min;
      result.far_geometric_gap_ratio_mean =
          jcp_tree_stats.far_geometric_gap_ratio_mean;
      result.far_geometric_audit_samples =
          jcp_tree_stats.far_geometric_audit_samples;
      result.mac_physical_gap_rejects = jcp_tree_stats.mac_physical_gap_rejects;
      result.mac_convergence_rho_rejects =
          jcp_tree_stats.mac_convergence_rho_rejects;
      result.mac_convergence_aspect_rejects =
          jcp_tree_stats.mac_convergence_aspect_rejects;
      result.mac_convergence_rho_reject_panels =
          jcp_tree_stats.mac_convergence_rho_reject_panels;
      result.mac_convergence_aspect_reject_panels =
          jcp_tree_stats.mac_convergence_aspect_reject_panels;
      result.fragmented_far_count = jcp_tree_stats.fragmented_far_count;
      result.fragmented_far_panels = jcp_tree_stats.fragmented_far_panels;
      result.fragmented_split_rejects = jcp_tree_stats.fragmented_split_rejects;
      result.fragmentation_center_gap_ratio_max =
          jcp_tree_stats.fragmentation_center_gap_ratio_max;
      result.fragmentation_center_gap_ratio_mean =
          jcp_tree_stats.fragmentation_center_gap_ratio_mean;
      result.moment_audit = jcp_tree_stats.moment_audit;
      result.moment_construction = jcp_tree_stats.moment_construction;
      result.moment_sum = jcp_tree_stats.moment_sum;
      result.convergence_audit = jcp_tree_stats.convergence_audit;
      result.convergence_safety = jcp_tree_stats.convergence_safety;
      result.convergence_rho_cap = jcp_tree_stats.convergence_rho_cap;
      result.mac_aspect_cap = jcp_tree_stats.mac_aspect_cap;
      result.convergence_estimator = jcp_tree_stats.convergence_estimator;
      result.convergence_tail_threshold =
          jcp_tree_stats.convergence_tail_threshold;
      result.convergence_estimator_error_cap =
          jcp_tree_stats.convergence_estimator_error_cap;
      result.convergence_unstable_rho_cap =
          jcp_tree_stats.convergence_unstable_rho_cap;
      result.convergence_calibration = jcp_tree_stats.convergence_calibration;
      result.convergence_estimator_samples =
          jcp_tree_stats.convergence_estimator_samples;
      result.convergence_estimator_unstable_far_count =
          jcp_tree_stats.convergence_estimator_unstable_far_count;
      result.convergence_estimator_unstable_far_panels =
          jcp_tree_stats.convergence_estimator_unstable_far_panels;
      result.convergence_estimator_split_rejects =
          jcp_tree_stats.convergence_estimator_split_rejects;
      result.convergence_estimator_split_reject_panels =
          jcp_tree_stats.convergence_estimator_split_reject_panels;
      result.convergence_estimator_tail_ratio_max =
          jcp_tree_stats.convergence_estimator_tail_ratio_max;
      result.convergence_estimator_tail_ratio_mean =
          jcp_tree_stats.convergence_estimator_tail_ratio_mean;
      result.convergence_estimator_error_max =
          jcp_tree_stats.convergence_estimator_error_max;
      result.convergence_estimator_error_mean =
          jcp_tree_stats.convergence_estimator_error_mean;
      result.convergence_tail_norm_p_max =
          jcp_tree_stats.convergence_tail_norm_p_max;
      result.convergence_tail_norm_p_mean =
          jcp_tree_stats.convergence_tail_norm_p_mean;
      result.convergence_tail_norm_pm1_max =
          jcp_tree_stats.convergence_tail_norm_pm1_max;
      result.convergence_tail_norm_pm1_mean =
          jcp_tree_stats.convergence_tail_norm_pm1_mean;
      result.convergence_effective_jcp_amp_max =
          jcp_tree_stats.convergence_effective_jcp_amp_max;
      result.convergence_effective_jcp_amp_mean =
          jcp_tree_stats.convergence_effective_jcp_amp_mean;
      result.convergence_estimator_high_error_caught =
          jcp_tree_stats.convergence_estimator_high_error_caught;
      result.convergence_estimator_high_error_missed =
          jcp_tree_stats.convergence_estimator_high_error_missed;
      result.convergence_estimator_high_error_caught_by_split =
          jcp_tree_stats.convergence_estimator_high_error_caught_by_split;
      result.moment_audit_nodes = jcp_tree_stats.moment_audit_nodes;
      result.moment_translation_rel_error_max =
          jcp_tree_stats.moment_translation_rel_error_max;
      result.moment_translation_rel_error_mean =
          jcp_tree_stats.moment_translation_rel_error_mean;
      result.moment_translation_offset_max =
          jcp_tree_stats.moment_translation_offset_max;
      result.moment_translation_offset_mean =
          jcp_tree_stats.moment_translation_offset_mean;
      result.moment_cancellation_ratio_max =
          jcp_tree_stats.moment_cancellation_ratio_max;
      result.moment_cancellation_ratio_mean =
          jcp_tree_stats.moment_cancellation_ratio_mean;
      result.moment_dynamic_range_max = jcp_tree_stats.moment_dynamic_range_max;
      result.expansion_audit = jcp_tree_stats.expansion_audit;
      result.far_expansion_audit_pairs =
          jcp_tree_stats.far_expansion_audit_pairs;
      result.far_expansion_audit_panels =
          jcp_tree_stats.far_expansion_audit_panels;
      result.far_expansion_direct_ms = jcp_tree_stats.far_expansion_direct_ms;
      result.far_expansion_rel_error_max =
          jcp_tree_stats.far_expansion_rel_error_max;
      result.far_expansion_rel_error_mean =
          jcp_tree_stats.far_expansion_rel_error_mean;
      result.far_expansion_mean_gate_pass =
          jcp_tree_stats.far_expansion_mean_gate_pass;
      result.far_convergence_rho_max = jcp_tree_stats.far_convergence_rho_max;
      result.far_convergence_rho_mean = jcp_tree_stats.far_convergence_rho_mean;
      result.accepted_far_aspect_ratio_max =
          jcp_tree_stats.accepted_far_aspect_ratio_max;
      result.accepted_far_aspect_ratio_mean =
          jcp_tree_stats.accepted_far_aspect_ratio_mean;
    }
    result.near_integration = jcp_tree_stats.near_integration;
    result.quad_buffer = jcp_tree_stats.quad_buffer;
    result.near_kernel_policy = jcp_tree_stats.near_kernel_policy;
    result.near_self_pairs = jcp_tree_stats.near_self_pairs;
    result.near_one_ring_pairs = jcp_tree_stats.near_one_ring_pairs;
    result.near_duffy_pairs = jcp_tree_stats.near_duffy_pairs;
    result.near_centroid_pairs = jcp_tree_stats.near_centroid_pairs;
    result.near_fallbacks = jcp_tree_stats.near_fallbacks;
    result.near_geometric_buffer_pairs =
        jcp_tree_stats.near_geometric_buffer_pairs;
    result.near_geometric_buffer_gauss_pairs =
        jcp_tree_stats.near_geometric_buffer_gauss_pairs;
    result.near_geometric_buffer_adaptive_pairs =
        jcp_tree_stats.near_geometric_buffer_adaptive_pairs;
    result.near_geometric_buffer_centroid_pairs =
        jcp_tree_stats.near_geometric_buffer_centroid_pairs;
    result.near_geometric_buffer_min_gap =
        jcp_tree_stats.near_geometric_buffer_min_gap;
    result.near_geometric_buffer_mean_gap_ratio =
        jcp_tree_stats.near_geometric_buffer_mean_gap_ratio;
    result.near_geometric_buffer_max_edge_mean =
        jcp_tree_stats.near_geometric_buffer_max_edge_mean;
    result.near_integration_ms = jcp_tree_stats.near_integration_ms;
    result.duffy_subdivisions_max = jcp_tree_stats.duffy_subdivisions_max;
    result.duffy_subdivisions_mean = jcp_tree_stats.duffy_subdivisions_mean;
    result.near_k1_accepted = jcp_tree_stats.near_k1_accepted;
    result.near_k1_rejected = jcp_tree_stats.near_k1_rejected;
    result.near_k2_accepted = jcp_tree_stats.near_k2_accepted;
    result.near_k2_rejected = jcp_tree_stats.near_k2_rejected;
    result.near_k3_accepted = jcp_tree_stats.near_k3_accepted;
    result.near_k3_rejected = jcp_tree_stats.near_k3_rejected;
    result.near_k4_accepted = jcp_tree_stats.near_k4_accepted;
    result.near_k4_rejected = jcp_tree_stats.near_k4_rejected;
    result.near_one_ring_k1_accepted = jcp_tree_stats.near_one_ring_k1_accepted;
    result.near_one_ring_k1_rejected = jcp_tree_stats.near_one_ring_k1_rejected;
    result.near_one_ring_k2_accepted = jcp_tree_stats.near_one_ring_k2_accepted;
    result.near_one_ring_k2_rejected = jcp_tree_stats.near_one_ring_k2_rejected;
    result.near_one_ring_k3_accepted = jcp_tree_stats.near_one_ring_k3_accepted;
    result.near_one_ring_k3_rejected = jcp_tree_stats.near_one_ring_k3_rejected;
    result.near_one_ring_k4_accepted = jcp_tree_stats.near_one_ring_k4_accepted;
    result.near_one_ring_k4_rejected = jcp_tree_stats.near_one_ring_k4_rejected;
    result.near_k1_policy = jcp_tree_stats.near_k1_policy;
    result.near_k2_policy = jcp_tree_stats.near_k2_policy;
    result.near_k3_policy = jcp_tree_stats.near_k3_policy;
    result.near_k4_policy = jcp_tree_stats.near_k4_policy;
    result.near_block_audit = jcp_tree_stats.near_block_audit;
    result.geometric_near_audit = jcp_tree_stats.geometric_near_audit;
    result.orthogonalization_ms = linear.orthogonalization_ms;
    result.preconditioner_ms =
        linear.preconditioner_ms + preconditioner_setup_ms;
    result.preconditioner_apply_ms = linear.preconditioner_apply_ms;
    result.preconditioner_apply_calls = linear.preconditioner_apply_calls;
    if (coupled) {
      result.phi_norm = rms_norm(std::vector<double>(
          linear.x.begin(), linear.x.begin() + panels.size()));
      result.phin_norm = rms_norm(std::vector<double>(
          linear.x.begin() + panels.size(), linear.x.end()));
    } else {
      result.phi_norm = rms_norm(linear.x);
      result.phin_norm = 0.0;
    }

    const double native_energy_scale = 4.0 * kPi;
    std::vector<double> atom_reaction_phi;
    const auto energy_projection_begin = std::chrono::steady_clock::now();
    const double boundary_energy =
        native_energy_scale *
        (coupled ? jcp_linear_pb_energy_kcal(atoms, panels, linear.x, eps_ratio,
                                             kappa, executor,
                                             &atom_reaction_phi,
                                             &result.energy_projection_kernel,
                                             options.energy_far_eta)
                 : jcp_poisson_energy_kcal(atoms, panels, linear.x, eps_ratio,
                                           executor, &atom_reaction_phi,
                                           &result.energy_projection_kernel,
                                           options.energy_far_eta));
    result.energy_projection_ms = elapsed_ms(
        energy_projection_begin, std::chrono::steady_clock::now());
    const double born =
        born_reference_kcal(atoms, options.eps_in, options.eps_out, kappa);
    double energy = boundary_energy;
    std::string source = "tabi_boundary_integral";
    double selected_scale = 1.0;
    if (formulation.energy_projection == "born-check") {
      energy = born;
      source = "tabi_born_check";
      selected_scale = 0.0;
    } else if (formulation.energy_projection == "charge-potential-debug") {
      source = "tabi_charge_potential_debug";
    }
    result.boundary_energy_kcal_mol = boundary_energy;
    result.reference_energy_kcal_mol = born;
    result.selected_energy_scale = selected_scale;
    result.fallback_used = false;
    result.reaction_energy_source = source;
    if (!linear.converged || !std::isfinite(energy)) {
      result.status = "failed";
      if (!std::isfinite(energy))
        result.convergence_reason = "nan_guard";
    } else {
      result.status = "ok";
    }
    result.energy_kcal_mol = energy;
    result.energy_kj_mol = energy * kKcalToKJ;
    result.energy_kt = result.energy_kj_mol /
                       (kKBKJPerMolK * std::max(1.0, options.temperature_k));
    result.atom_reaction_potential_kj_mol_e.resize(atom_reaction_phi.size());
    result.atom_reaction_energy_kj_mol.resize(atom_reaction_phi.size());
    double atom_closure_kj = 0.0;
    const double phi_scale =
        native_energy_scale * kCoulombKcalPerMol * kKcalToKJ;
    for (std::size_t atom_index = 0; atom_index < atom_reaction_phi.size();
         ++atom_index) {
      const double potential_kj_mol_e =
          phi_scale * atom_reaction_phi[atom_index];
      const double contribution_kj =
          0.5 * atoms[atom_index].q * potential_kj_mol_e;
      result.atom_reaction_potential_kj_mol_e[atom_index] = potential_kj_mol_e;
      result.atom_reaction_energy_kj_mol[atom_index] = contribution_kj;
      atom_closure_kj += contribution_kj;
    }
    result.atom_reaction_energy_closure_kj_mol = atom_closure_kj;
    result.atom_reaction_energy_closure_error_kj_mol =
        atom_closure_kj - result.boundary_energy_kcal_mol * kKcalToKJ;
    result.peak_memory_bytes = result.memory_bytes;
    refresh_matvec_cost_summary(result);
    result.node_patch_certificate =
        node_patch_certificate(panels) +
        node_patch_solution_shape(panels, linear.x,
                                  result.bie_system == "linear_pb_2x2");
    if (options.surface_output != "off") {
      result.output_panels = panels;
      result.output_solution = linear.x;
      result.output_rhs = rhs;
    }
#if defined(TITAN_CARTESIAN_FMM_TESTING)
    if (cartesian_full_operator)
      maybe_write_p3d0_direct_phase_attribution(*cartesian_full_operator,
                                                 result);
#endif
    progress.write("solve_done", result, &jcp_tree_stats, result.matvec_calls,
                   result.iterations, 0, result.residual, result.status, true);
    return result;
  }

  TreeWorkspace tree;
  TreecodeStats matvec_stats;
  matvec_stats.matvec_mode = mode;
  matvec_stats.taylor_order = mode == "treecode" ? result.taylor_order : 0;
  matvec_stats.derivative_mode = result.derivative_mode;
  matvec_stats.interaction_mode = result.interaction_mode;
  matvec_stats.node_bounds_mode = result.node_bounds_mode;
  matvec_stats.tree_builder = result.tree_builder;
  matvec_stats.expansion_center = result.expansion_center;
  matvec_stats.fragmentation_safety = result.fragmentation_safety;
  matvec_stats.mac_safety = result.mac_safety;
  matvec_stats.mac_gap_factor = result.mac_gap_factor;
  matvec_stats.moment_audit = result.moment_audit;
  matvec_stats.moment_construction = result.moment_construction;
  matvec_stats.moment_sum = result.moment_sum;
  matvec_stats.convergence_audit = result.convergence_audit;
  matvec_stats.convergence_safety = result.convergence_safety;
  matvec_stats.convergence_rho_cap = result.convergence_rho_cap;
  matvec_stats.mac_aspect_cap = result.mac_aspect_cap;
  matvec_stats.convergence_estimator = result.convergence_estimator;
  matvec_stats.convergence_tail_threshold = result.convergence_tail_threshold;
  matvec_stats.convergence_estimator_error_cap =
      result.convergence_estimator_error_cap;
  matvec_stats.convergence_unstable_rho_cap =
      result.convergence_unstable_rho_cap;
  matvec_stats.convergence_calibration = result.convergence_calibration;
  matvec_stats.convergence_dielectric_amp = result.convergence_dielectric_amp;
  matvec_stats.convergence_estimator_model = result.convergence_estimator_model;
  matvec_stats.convergence_channel_calibration =
      result.convergence_channel_calibration;
  matvec_stats.convergence_split_policy = result.convergence_split_policy;
  matvec_stats.convergence_split_selectivity =
      result.convergence_split_selectivity;
  matvec_stats.convergence_split_absolute_floor =
      result.convergence_split_absolute_floor;
  matvec_stats.convergence_overreject_audit =
      result.convergence_overreject_audit;
  matvec_stats.convergence_overreject_detail =
      result.convergence_overreject_detail;
  matvec_stats.convergence_shape_audit = result.convergence_shape_audit;
  matvec_stats.convergence_kernel_response_audit =
      result.convergence_kernel_response_audit;
  matvec_stats.channel_missed_semantics = result.channel_missed_semantics;
  matvec_stats.accepted_miss_diagnostics = result.accepted_miss_diagnostics;
  matvec_stats.accepted_miss_absolute_floor =
      result.accepted_miss_absolute_floor;
  matvec_stats.convergence_k3_accepted_rho_cap =
      result.convergence_k3_accepted_rho_cap;
  matvec_stats.mac_distance_policy = result.mac_distance_policy;
  matvec_stats.mac_surface_margin_factor = result.mac_surface_margin_factor;
  matvec_stats.mac_surface_margin_policy = result.mac_surface_margin_policy;
  matvec_stats.mac_surface_margin_aspect_threshold =
      result.mac_surface_margin_aspect_threshold;
  matvec_stats.mac_surface_margin_rho_trigger =
      result.mac_surface_margin_rho_trigger;
  matvec_stats.mac_surface_margin_soft_aspect_threshold =
      result.mac_surface_margin_soft_aspect_threshold;
  matvec_stats.mac_surface_margin_soft_rho_trigger =
      result.mac_surface_margin_soft_rho_trigger;
  matvec_stats.mac_surface_margin_soft_factor =
      result.mac_surface_margin_soft_factor;
  matvec_stats.cluster_policy = result.cluster_policy;
  matvec_stats.target_batch_size = result.target_batch_size;
  matvec_stats.leaf_pair_min_source_panels = result.leaf_pair_min_source_panels;
  matvec_stats.leaf_pair_min_targets = result.leaf_pair_min_targets;
  matvec_stats.leaf_kernel_profile = result.leaf_kernel_profile;
  matvec_stats.leaf_work_profile = result.leaf_work_profile;
  matvec_stats.matvec_cost_profile = result.matvec_cost_profile;
  matvec_stats.jcp_direct_kernel = result.jcp_direct_kernel;
  matvec_stats.leaf_pair_selection_profile = result.leaf_pair_selection_profile;
  matvec_stats.leaf_pair_probe_cache = result.leaf_pair_probe_cache;
  matvec_stats.leaf_pair_target_group = result.leaf_pair_target_group;
  matvec_stats.leaf_pair_directness_probe = result.leaf_pair_directness_probe;
  matvec_stats.leaf_pair_directness_cache = result.leaf_pair_directness_cache;
  matvec_stats.leaf_pair_directness_verify = result.leaf_pair_directness_verify;
  matvec_stats.leaf_pair_scheduler = result.leaf_pair_scheduler;
  matvec_stats.leaf_pair_worklist_max_active =
      result.leaf_pair_worklist_max_active;
  matvec_stats.leaf_pair_worklist_verify = result.leaf_pair_worklist_verify;
  matvec_stats.leaf_block_verify = result.leaf_block_verify;
  matvec_stats.leaf_pair_max_lookahead = result.leaf_pair_max_lookahead;
  matvec_stats.expansion_audit = result.expansion_audit;
  matvec_stats.expansion_audit_channel = result.expansion_audit_channel;
  if (mode == "treecode") {
    progress.write("tree_build_start", result, &matvec_stats, 0, 0, 0, 0.0,
                   "running", true);
    tree = build_tree_workspace(
        panels, executor,
        static_cast<std::size_t>(std::max(8, effective_tree_leaf_size)),
        effective_tree_opening_theta, options.leaf_near_policy,
        options.leaf_panel_kernel, options.adaptive_level, "off",
        options.interaction_mode, result.node_bounds_mode, result.tree_builder,
        options.mac_safety, options.mac_gap_factor, options.moment_audit,
        result.expansion_center, result.fragmentation_safety,
        options.expansion_audit, options.expansion_audit_sample_count,
        options.expansion_audit_max_panels, options.moment_construction,
        options.moment_sum, options.convergence_audit,
        options.convergence_safety, options.convergence_rho_cap,
        options.mac_aspect_cap, options.convergence_estimator,
        options.convergence_tail_threshold,
        options.convergence_estimator_error_cap,
        options.convergence_unstable_rho_cap, options.convergence_calibration,
        options.cluster_policy, options.convergence_dielectric_amp,
        options.target_batch_size, options.leaf_pair_min_source_panels,
        options.leaf_pair_min_targets, options.convergence_estimator_model,
        options.expansion_audit_channel,
        options.convergence_channel_calibration,
        options.channel_missed_semantics, options.convergence_split_policy,
        options.accepted_miss_diagnostics,
        options.convergence_k3_accepted_rho_cap, options.mac_distance_policy,
        options.mac_surface_margin_factor, options.mac_surface_margin_policy,
        options.mac_surface_margin_aspect_threshold,
        options.mac_surface_margin_rho_trigger,
        options.accepted_miss_absolute_floor, options.leaf_pair_kernel,
        options.leaf_kernel_profile, options.leaf_work_profile,
        options.jcp_direct_kernel, options.leaf_pair_selection_profile,
        options.leaf_pair_probe_cache, options.leaf_pair_target_group,
        options.leaf_pair_max_lookahead, options.leaf_pair_directness_probe,
        options.leaf_pair_directness_cache, options.leaf_pair_directness_verify,
        options.leaf_pair_scheduler, options.leaf_pair_worklist_max_active,
        options.leaf_pair_worklist_verify, options.leaf_block_verify,
        options.mac_surface_margin_soft_aspect_threshold,
        options.mac_surface_margin_soft_rho_trigger,
        options.mac_surface_margin_soft_factor,
        options.convergence_split_selectivity,
        options.convergence_split_absolute_floor,
        options.convergence_overreject_audit,
        options.convergence_overreject_detail, options.convergence_shape_audit,
        options.convergence_kernel_response_audit, result.leaf_mac,
        options.fmm_m2l);
    tree.leaf_mac = result.leaf_mac;
    tree.taylor_order = result.taylor_order;
    tree.derivative_mode = result.derivative_mode;
    result.interaction_mode = tree.interaction_mode;
    result.node_bounds_mode = tree.node_bounds_mode;
    result.tree_builder = tree.tree_builder;
    result.expansion_center = tree.expansion_center;
    result.fragmentation_safety = tree.fragmentation_safety;
    result.mac_safety = tree.mac_safety;
    result.mac_gap_factor = tree.mac_gap_factor;
    result.moment_audit = tree.moment_audit;
    result.moment_construction = tree.moment_construction;
    result.moment_sum = tree.moment_sum;
    result.convergence_audit = tree.convergence_audit;
    result.convergence_safety = tree.convergence_safety;
    result.convergence_rho_cap = tree.convergence_rho_cap;
    result.mac_aspect_cap = tree.mac_aspect_cap;
    result.convergence_estimator = tree.convergence_estimator;
    result.convergence_tail_threshold = tree.convergence_tail_threshold;
    result.convergence_estimator_error_cap =
        tree.convergence_estimator_error_cap;
    result.convergence_unstable_rho_cap = tree.convergence_unstable_rho_cap;
    result.convergence_calibration = tree.convergence_calibration;
    result.convergence_dielectric_amp = tree.convergence_dielectric_amp;
    result.convergence_estimator_model = tree.convergence_estimator_model;
    result.convergence_channel_calibration =
        tree.convergence_channel_calibration;
    result.convergence_split_policy = tree.convergence_split_policy;
    result.convergence_split_selectivity = tree.convergence_split_selectivity;
    result.convergence_split_absolute_floor =
        tree.convergence_split_absolute_floor;
    result.convergence_overreject_audit = tree.convergence_overreject_audit;
    result.convergence_overreject_detail = tree.convergence_overreject_detail;
    result.convergence_shape_audit = tree.convergence_shape_audit;
    result.convergence_kernel_response_audit =
        tree.convergence_kernel_response_audit;
    result.channel_missed_semantics = tree.channel_missed_semantics;
    result.accepted_miss_diagnostics = tree.accepted_miss_diagnostics;
    result.accepted_miss_absolute_floor = tree.accepted_miss_absolute_floor;
    result.convergence_k3_accepted_rho_cap =
        tree.convergence_k3_accepted_rho_cap;
    result.mac_distance_policy = tree.mac_distance_policy;
    result.mac_surface_margin_factor = tree.mac_surface_margin_factor;
    result.mac_surface_margin_policy = tree.mac_surface_margin_policy;
    result.mac_surface_margin_aspect_threshold =
        tree.mac_surface_margin_aspect_threshold;
    result.mac_surface_margin_rho_trigger = tree.mac_surface_margin_rho_trigger;
    result.mac_surface_margin_soft_aspect_threshold =
        tree.mac_surface_margin_soft_aspect_threshold;
    result.mac_surface_margin_soft_rho_trigger =
        tree.mac_surface_margin_soft_rho_trigger;
    result.mac_surface_margin_soft_factor = tree.mac_surface_margin_soft_factor;
    result.cluster_policy = tree.cluster_policy;
    result.target_batch_size = static_cast<int>(tree.target_batch_size);
    result.leaf_pair_min_source_panels =
        static_cast<int>(tree.leaf_pair_min_source_panels);
    result.leaf_pair_min_targets = static_cast<int>(tree.leaf_pair_min_targets);
    result.leaf_kernel_profile = tree.leaf_kernel_profile;
    result.leaf_work_profile = tree.leaf_work_profile;
    result.jcp_direct_kernel = tree.jcp_direct_kernel;
    result.leaf_pair_selection_profile = tree.leaf_pair_selection_profile;
    result.leaf_pair_probe_cache = tree.leaf_pair_probe_cache;
    result.leaf_pair_target_group = tree.leaf_pair_target_group;
    result.leaf_pair_directness_probe = tree.leaf_pair_directness_probe;
    result.leaf_pair_directness_cache = tree.leaf_pair_directness_cache;
    result.leaf_pair_directness_verify = tree.leaf_pair_directness_verify;
    result.leaf_pair_scheduler = tree.leaf_pair_scheduler;
    result.leaf_pair_worklist_max_active =
        static_cast<int>(tree.leaf_pair_worklist_max_active);
    result.leaf_pair_worklist_verify = tree.leaf_pair_worklist_verify;
    result.leaf_block_verify = tree.leaf_block_verify;
    result.leaf_pair_max_lookahead =
        static_cast<int>(tree.leaf_pair_max_lookahead);
    result.expansion_audit = tree.expansion_audit;
    result.expansion_audit_channel = tree.expansion_audit_channel;
    matvec_stats.node_bounds_mode = tree.node_bounds_mode;
    matvec_stats.interaction_mode = tree.interaction_mode;
    matvec_stats.tree_builder = tree.tree_builder;
    matvec_stats.expansion_center = tree.expansion_center;
    matvec_stats.fragmentation_safety = tree.fragmentation_safety;
    matvec_stats.mac_safety = tree.mac_safety;
    matvec_stats.mac_gap_factor = tree.mac_gap_factor;
    matvec_stats.moment_audit = tree.moment_audit;
    matvec_stats.moment_construction = tree.moment_construction;
    matvec_stats.moment_sum = tree.moment_sum;
    matvec_stats.convergence_audit = tree.convergence_audit;
    matvec_stats.convergence_safety = tree.convergence_safety;
    matvec_stats.convergence_rho_cap = tree.convergence_rho_cap;
    matvec_stats.mac_aspect_cap = tree.mac_aspect_cap;
    matvec_stats.convergence_estimator = tree.convergence_estimator;
    matvec_stats.convergence_tail_threshold = tree.convergence_tail_threshold;
    matvec_stats.convergence_estimator_error_cap =
        tree.convergence_estimator_error_cap;
    matvec_stats.convergence_unstable_rho_cap =
        tree.convergence_unstable_rho_cap;
    matvec_stats.convergence_calibration = tree.convergence_calibration;
    matvec_stats.convergence_dielectric_amp = tree.convergence_dielectric_amp;
    matvec_stats.convergence_estimator_model = tree.convergence_estimator_model;
    matvec_stats.convergence_channel_calibration =
        tree.convergence_channel_calibration;
    matvec_stats.convergence_split_policy = tree.convergence_split_policy;
    matvec_stats.convergence_split_selectivity =
        tree.convergence_split_selectivity;
    matvec_stats.convergence_split_absolute_floor =
        tree.convergence_split_absolute_floor;
    matvec_stats.convergence_overreject_audit =
        tree.convergence_overreject_audit;
    matvec_stats.convergence_overreject_detail =
        tree.convergence_overreject_detail;
    matvec_stats.convergence_shape_audit = tree.convergence_shape_audit;
    matvec_stats.convergence_kernel_response_audit =
        tree.convergence_kernel_response_audit;
    matvec_stats.channel_missed_semantics = tree.channel_missed_semantics;
    matvec_stats.accepted_miss_diagnostics = tree.accepted_miss_diagnostics;
    matvec_stats.accepted_miss_absolute_floor =
        tree.accepted_miss_absolute_floor;
    matvec_stats.convergence_k3_accepted_rho_cap =
        tree.convergence_k3_accepted_rho_cap;
    matvec_stats.mac_distance_policy = tree.mac_distance_policy;
    matvec_stats.mac_surface_margin_factor = tree.mac_surface_margin_factor;
    matvec_stats.mac_surface_margin_policy = tree.mac_surface_margin_policy;
    matvec_stats.mac_surface_margin_aspect_threshold =
        tree.mac_surface_margin_aspect_threshold;
    matvec_stats.mac_surface_margin_rho_trigger =
        tree.mac_surface_margin_rho_trigger;
    matvec_stats.mac_surface_margin_soft_aspect_threshold =
        tree.mac_surface_margin_soft_aspect_threshold;
    matvec_stats.mac_surface_margin_soft_rho_trigger =
        tree.mac_surface_margin_soft_rho_trigger;
    matvec_stats.mac_surface_margin_soft_factor =
        tree.mac_surface_margin_soft_factor;
    matvec_stats.cluster_policy = tree.cluster_policy;
    matvec_stats.target_batch_size = static_cast<int>(tree.target_batch_size);
    matvec_stats.leaf_pair_min_source_panels =
        static_cast<int>(tree.leaf_pair_min_source_panels);
    matvec_stats.leaf_pair_min_targets =
        static_cast<int>(tree.leaf_pair_min_targets);
    matvec_stats.leaf_kernel_profile = tree.leaf_kernel_profile;
    matvec_stats.leaf_work_profile = tree.leaf_work_profile;
    matvec_stats.matvec_cost_profile = result.matvec_cost_profile;
    matvec_stats.jcp_direct_kernel = tree.jcp_direct_kernel;
    matvec_stats.leaf_pair_selection_profile = tree.leaf_pair_selection_profile;
    matvec_stats.leaf_pair_probe_cache = tree.leaf_pair_probe_cache;
    matvec_stats.leaf_pair_target_group = tree.leaf_pair_target_group;
    matvec_stats.leaf_pair_directness_probe = tree.leaf_pair_directness_probe;
    matvec_stats.leaf_pair_directness_cache = tree.leaf_pair_directness_cache;
    matvec_stats.leaf_pair_directness_verify = tree.leaf_pair_directness_verify;
    matvec_stats.leaf_pair_scheduler = tree.leaf_pair_scheduler;
    matvec_stats.leaf_pair_worklist_max_active =
        static_cast<int>(tree.leaf_pair_worklist_max_active);
    matvec_stats.leaf_pair_worklist_verify = tree.leaf_pair_worklist_verify;
    matvec_stats.leaf_block_verify = tree.leaf_block_verify;
    matvec_stats.leaf_pair_max_lookahead =
        static_cast<int>(tree.leaf_pair_max_lookahead);
    matvec_stats.expansion_audit = tree.expansion_audit;
    matvec_stats.expansion_audit_channel = tree.expansion_audit_channel;
    matvec_stats.tree_depth_max = tree.depth_max;
    matvec_stats.node_radius_max = tree.node_radius_max;
    matvec_stats.node_radius_mean = tree.node_radius_mean;
    matvec_stats.node_aspect_ratio_max = tree.node_aspect_ratio_max;
    matvec_stats.node_aspect_ratio_mean = tree.node_aspect_ratio_mean;
    matvec_stats.tree_build_ms = tree.build_ms;
    matvec_stats.tree_nodes = tree.nodes.size();
    matvec_stats.tree_leaf_count = tree.leaf_count;
    result.tree_build_ms = tree.build_ms;
    result.tree_nodes = tree.nodes.size();
    result.tree_leaf_count = tree.leaf_count;
    result.tree_depth_max = tree.depth_max;
    result.memory_bytes += tree.nodes.size() * sizeof(TreeNode) +
                           tree.child_indices.size() * sizeof(std::size_t) +
                           tree.order.size() * sizeof(std::size_t) +
                           tree.morton.size() * sizeof(std::uint64_t) +
                           panels.size() * 14u * sizeof(double) +
                           panels.size() * sizeof(std::size_t);
    progress.write("tree_build_end", result, &matvec_stats, 0, 0, 0, 0.0,
                   "running", true);
  }

  const auto rhs_begin = std::chrono::steady_clock::now();
  const std::vector<double> rhs =
      build_rhs(atoms, panels, contrast, kappa, formulation.rhs_scale);
  result.rhs_ms = elapsed_ms(rhs_begin, std::chrono::steady_clock::now());
  const int max_iter =
      std::min(std::max(1, options.max_iter),
               std::max(32, static_cast<int>(panels.size()) * 4));
  const double tol = std::max(1.0e-12, options.residual_tol);
  double matvec_ms_total = 0.0;
  std::size_t matvec_calls = 0;
  auto matvec = [&](const std::vector<double> &x, std::vector<double> &out) {
    const auto begin = std::chrono::steady_clock::now();
    if (mode == "treecode")
      apply_treecode_operator(panels, x, operator_contrast, kappa, tree,
                              executor, out, &matvec_stats);
    else
      apply_direct_operator(panels, x, operator_contrast, kappa, executor, out,
                            &matvec_stats);
    matvec_ms_total += elapsed_ms(begin, std::chrono::steady_clock::now());
    ++matvec_calls;
  };
  const auto gmres_begin = std::chrono::steady_clock::now();
  progress.write("gmres_start", result, &matvec_stats, matvec_calls, 0, 0, 0.0,
                 "running", true);
  auto legacy_progress = [&](int gmres_iter, int restart_iter,
                             double true_residual_rel,
                             double preconditioned_residual_rel, bool force,
                             const std::string &phase,
                             const LinearSolveResult &state) {
    result.true_residual_probe_scope = state.true_residual_probe_scope;
    result.true_residual_sample_count = state.true_residual_sample_count;
    result.true_residual_probe_calls = state.true_residual_probe_calls;
    result.true_residual_probe_ms = state.true_residual_probe_ms;
    result.sampled_true_residual_rel = state.sampled_true_residual_rel;
    result.sampled_true_residual_rows = state.sampled_true_residual_rows;
    result.sampled_probe_calls = state.sampled_probe_calls;
    result.sampled_probe_ms = state.sampled_probe_ms;
    result.full_true_residual_confirm_calls =
        state.full_true_residual_confirm_calls;
    result.full_true_residual_confirm_ms = state.full_true_residual_confirm_ms;
    if (phase != "gmres_sampled_true_probe" &&
        std::isfinite(true_residual_rel)) {
      result.final_residual = true_residual_rel;
      result.true_residual_rel = true_residual_rel;
      if (result.best_residual <= 0.0 ||
          true_residual_rel < result.best_residual) {
        result.best_residual = true_residual_rel;
        result.best_residual_iter = gmres_iter;
      }
    }
    if (std::isfinite(preconditioned_residual_rel)) {
      result.final_preconditioned_residual = preconditioned_residual_rel;
      if (result.best_preconditioned_residual <= 0.0 ||
          preconditioned_residual_rel < result.best_preconditioned_residual) {
        result.best_preconditioned_residual = preconditioned_residual_rel;
      }
    }
    progress.write(phase, result, &matvec_stats, matvec_calls, gmres_iter,
                   restart_iter, true_residual_rel, "running", force);
  };
  std::vector<double> legacy_diagonal(panels.size(), 1.0);
  for (std::size_t i = 0; i < panels.size(); ++i)
    legacy_diagonal[i] = panels[i].diagonal;
  LinearSolveResult linear = gmres_solve(
      matvec, rhs, legacy_diagonal, max_iter, tol, options.gmres_restart,
      executor, {}, legacy_progress, result.true_residual_probe_mode,
      result.true_residual_probe_interval, result.true_residual_probe_scope,
      result.true_residual_sample_count);
  result.gmres_ms = elapsed_ms(gmres_begin, std::chrono::steady_clock::now());
  result.iterations = linear.iterations;
  result.residual = linear.residual;
  result.gmres_restart_requested = linear.restart_requested;
  result.gmres_restart_effective = linear.restart_effective;
  result.best_residual = linear.best_residual;
  result.best_residual_iter = linear.best_residual_iter;
  result.final_residual = linear.final_residual;
  result.best_preconditioned_residual = linear.best_preconditioned_residual;
  result.final_preconditioned_residual = linear.final_preconditioned_residual;
  result.true_residual_abs = linear.true_residual_abs;
  result.true_residual_rel = linear.true_residual_rel;
  result.true_residual_probe_mode = linear.true_residual_probe_mode;
  result.true_residual_probe_scope = linear.true_residual_probe_scope;
  result.true_residual_probe_interval = linear.true_residual_probe_interval;
  result.true_residual_sample_count = linear.true_residual_sample_count;
  result.true_residual_probe_calls = linear.true_residual_probe_calls;
  result.true_residual_probe_ms = linear.true_residual_probe_ms;
  result.sampled_true_residual_rel = linear.sampled_true_residual_rel;
  result.sampled_true_residual_rows = linear.sampled_true_residual_rows;
  result.sampled_probe_calls = linear.sampled_probe_calls;
  result.sampled_probe_ms = linear.sampled_probe_ms;
  result.full_true_residual_confirm_calls =
      linear.full_true_residual_confirm_calls;
  result.full_true_residual_confirm_ms = linear.full_true_residual_confirm_ms;
  result.converged = linear.converged;
  result.convergence_reason = linear.convergence_reason;
  result.residual_history = residual_history_string(linear.residual_history);
  result.matvec_ms_total = matvec_ms_total;
  result.matvec_calls = matvec_calls;
  result.matvec_ms_avg =
      matvec_calls > 0 ? matvec_ms_total / static_cast<double>(matvec_calls)
                       : 0.0;
  result.orthogonalization_ms = linear.orthogonalization_ms;
  result.preconditioner_ms = linear.preconditioner_ms;
  result.tree_build_ms = matvec_stats.tree_build_ms;
  result.tree_nodes = matvec_stats.tree_nodes;
  result.tree_leaf_count = matvec_stats.tree_leaf_count;
  result.near_interactions = matvec_stats.near_interactions;
  result.far_interactions = matvec_stats.far_interactions;
  result.singular_near_interactions = matvec_stats.singular_near_interactions;
  result.leaf_near_interactions = matvec_stats.leaf_near_interactions;
  result.far_node_interactions = matvec_stats.far_node_interactions;
  result.leaf_panel_interactions = matvec_stats.leaf_panel_interactions;
  result.leaf_tile_interactions = matvec_stats.leaf_tile_interactions;
  result.leaf_tile_panel_interactions =
      matvec_stats.leaf_tile_panel_interactions;
  result.leaf_residual_panel_interactions =
      matvec_stats.leaf_residual_panel_interactions;
  result.smooth_leaf_node_interactions =
      matvec_stats.smooth_leaf_node_interactions;
  result.leaf_pruned_interactions = matvec_stats.leaf_pruned_interactions;
  result.leaf_mac_candidates = matvec_stats.leaf_mac_candidates;
  result.leaf_mac_accepted = matvec_stats.leaf_mac_accepted;
  result.leaf_mac_rejected = matvec_stats.leaf_mac_rejected;
  result.leaf_mac_far_panels = matvec_stats.leaf_mac_far_panels;
  result.leaf_mac_direct_panels_saved_estimate =
      matvec_stats.leaf_mac_direct_panels_saved_estimate;
  result.leaf_work_profile = matvec_stats.leaf_work_profile;
  result.leaf_work_mac_terminal_panels =
      matvec_stats.leaf_work_mac_terminal_panels;
  result.leaf_work_physical_gap_panels =
      matvec_stats.leaf_work_physical_gap_panels;
  result.leaf_work_fragmentation_panels =
      matvec_stats.leaf_work_fragmentation_panels;
  result.leaf_work_convergence_panels =
      matvec_stats.leaf_work_convergence_panels;
  result.leaf_work_convergence_estimator_panels =
      matvec_stats.leaf_work_convergence_estimator_panels;
  result.leaf_work_convergence_rho_panels =
      matvec_stats.leaf_work_convergence_rho_panels;
  result.leaf_work_convergence_aspect_panels =
      matvec_stats.leaf_work_convergence_aspect_panels;
  result.leaf_work_surface_margin_panels =
      matvec_stats.leaf_work_surface_margin_panels;
  result.leaf_work_surface_margin_hard_panels =
      matvec_stats.leaf_work_surface_margin_hard_panels;
  result.leaf_work_surface_margin_soft_panels =
      matvec_stats.leaf_work_surface_margin_soft_panels;
  result.leaf_work_k3_rho_cap_panels = matvec_stats.leaf_work_k3_rho_cap_panels;
  result.leaf_work_singular_fallback_panels =
      matvec_stats.leaf_work_singular_fallback_panels;
  result.leaf_work_scalar_tail_panels =
      matvec_stats.leaf_work_scalar_tail_panels;
  result.leaf_work_profile_accounted_panels =
      matvec_stats.leaf_work_profile_accounted_panels;
  result.leaf_work_mac_terminal_frac = matvec_stats.leaf_work_mac_terminal_frac;
  result.leaf_work_convergence_frac = matvec_stats.leaf_work_convergence_frac;
  result.leaf_work_surface_margin_frac =
      matvec_stats.leaf_work_surface_margin_frac;
  result.leaf_work_singular_fallback_frac =
      matvec_stats.leaf_work_singular_fallback_frac;
  result.leaf_work_scalar_tail_frac = matvec_stats.leaf_work_scalar_tail_frac;
  result.adaptive_smooth_leaf_accepted =
      matvec_stats.adaptive_smooth_leaf_accepted;
  result.adaptive_smooth_leaf_rejected =
      matvec_stats.adaptive_smooth_leaf_rejected;
  result.adaptive_span_tile_accepted = matvec_stats.adaptive_span_tile_accepted;
  result.adaptive_span_tile_rejected = matvec_stats.adaptive_span_tile_rejected;
  result.adaptive_exact_leaf_fallbacks =
      matvec_stats.adaptive_exact_leaf_fallbacks;
  result.adaptive_reject_one_ring_overlap =
      matvec_stats.adaptive_reject_one_ring_overlap;
  result.adaptive_reject_error_bound_failed =
      matvec_stats.adaptive_reject_error_bound_failed;
  result.adaptive_reject_p4_check_failed =
      matvec_stats.adaptive_reject_p4_check_failed;
  result.adaptive_reject_too_close = matvec_stats.adaptive_reject_too_close;
  result.leaf_work_reduction_ratio = matvec_stats.leaf_work_reduction_ratio;
  result.target_cluster_interactions = matvec_stats.target_cluster_interactions;
  result.target_cluster_targets = matvec_stats.target_cluster_targets;
  result.target_cluster_panels_pruned =
      matvec_stats.target_cluster_panels_pruned;
  result.target_cluster_rejected = matvec_stats.target_cluster_rejected;
  result.target_cluster_eval_ms = matvec_stats.target_cluster_eval_ms;
  result.leaf_pair_batch_interactions =
      matvec_stats.leaf_pair_batch_interactions;
  result.leaf_pair_batch_targets = matvec_stats.leaf_pair_batch_targets;
  result.leaf_pair_batch_source_panels =
      matvec_stats.leaf_pair_batch_source_panels;
  result.leaf_pair_batch_residual_panels =
      matvec_stats.leaf_pair_batch_residual_panels;
  result.leaf_pair_batch_fallbacks = matvec_stats.leaf_pair_batch_fallbacks;
  result.leaf_pair_batch_ms = matvec_stats.leaf_pair_batch_ms;
  result.leaf_pair_batch_kernel = matvec_stats.leaf_pair_batch_kernel;
  result.leaf_pair_batch_vectorized_sources =
      matvec_stats.leaf_pair_batch_vectorized_sources;
  result.leaf_pair_batch_vector_lanes =
      matvec_stats.leaf_pair_batch_vector_lanes;
  result.leaf_pair_batch_pairs_per_second =
      matvec_stats.leaf_pair_batch_pairs_per_second;
  result.leaf_kernel_profile = matvec_stats.leaf_kernel_profile;
  result.leaf_work_profile = matvec_stats.leaf_work_profile;
  result.jcp_direct_kernel = matvec_stats.jcp_direct_kernel;
  result.leaf_kernel_scalar_tasks = matvec_stats.leaf_kernel_scalar_tasks;
  result.leaf_kernel_batch_tasks = matvec_stats.leaf_kernel_batch_tasks;
  result.leaf_kernel_source_panel_evals =
      matvec_stats.leaf_kernel_source_panel_evals;
  result.leaf_kernel_scalar_ms = matvec_stats.leaf_kernel_scalar_ms;
  result.leaf_kernel_target_avx2_ms = matvec_stats.leaf_kernel_target_avx2_ms;
  result.leaf_kernel_tail_ms = matvec_stats.leaf_kernel_tail_ms;
  result.leaf_pair_batch_soa_sources = matvec_stats.leaf_pair_batch_soa_sources;
  result.leaf_pair_batch_soa_ms = matvec_stats.leaf_pair_batch_soa_ms;
  result.leaf_pair_batch_scalar_tail_ms =
      matvec_stats.leaf_pair_batch_scalar_tail_ms;
  result.leaf_pair_batch_clean_source_panels_per_second =
      matvec_stats.leaf_pair_batch_clean_source_panels_per_second;
  result.leaf_pair_selection_profile = matvec_stats.leaf_pair_selection_profile;
  result.leaf_pair_probe_cache = matvec_stats.leaf_pair_probe_cache;
  result.leaf_pair_target_group = matvec_stats.leaf_pair_target_group;
  result.leaf_pair_directness_probe = matvec_stats.leaf_pair_directness_probe;
  result.leaf_pair_directness_cache = matvec_stats.leaf_pair_directness_cache;
  result.leaf_pair_directness_verify = matvec_stats.leaf_pair_directness_verify;
  result.leaf_pair_max_lookahead = matvec_stats.leaf_pair_max_lookahead;
  result.leaf_pair_selection_candidate_scans =
      matvec_stats.leaf_pair_selection_candidate_scans;
  result.leaf_pair_selection_clean_rejects =
      matvec_stats.leaf_pair_selection_clean_rejects;
  result.leaf_pair_selection_probe_calls =
      matvec_stats.leaf_pair_selection_probe_calls;
  result.leaf_pair_selection_probe_node_visits =
      matvec_stats.leaf_pair_selection_probe_node_visits;
  result.leaf_pair_selection_probe_ms =
      matvec_stats.leaf_pair_selection_probe_ms;
  result.leaf_pair_selection_ms = matvec_stats.leaf_pair_selection_ms;
  result.leaf_pair_selection_probe_cache_hits =
      matvec_stats.leaf_pair_selection_probe_cache_hits;
  result.leaf_pair_selection_probe_cache_misses =
      matvec_stats.leaf_pair_selection_probe_cache_misses;
  result.leaf_pair_selection_insufficient_targets =
      matvec_stats.leaf_pair_selection_insufficient_targets;
  result.leaf_pair_selection_same_target_leaf_rejects =
      matvec_stats.leaf_pair_selection_same_target_leaf_rejects;
  result.leaf_pair_directness_ancestor_probe_calls =
      matvec_stats.leaf_pair_directness_ancestor_probe_calls;
  result.leaf_pair_directness_ancestor_probe_node_visits =
      matvec_stats.leaf_pair_directness_ancestor_probe_node_visits;
  result.leaf_pair_directness_ancestor_probe_ms =
      matvec_stats.leaf_pair_directness_ancestor_probe_ms;
  result.leaf_pair_directness_pair_cache_hits =
      matvec_stats.leaf_pair_directness_pair_cache_hits;
  result.leaf_pair_directness_pair_cache_misses =
      matvec_stats.leaf_pair_directness_pair_cache_misses;
  result.leaf_pair_directness_all_direct_proofs =
      matvec_stats.leaf_pair_directness_all_direct_proofs;
  result.leaf_pair_directness_mixed_fallbacks =
      matvec_stats.leaf_pair_directness_mixed_fallbacks;
  result.leaf_pair_directness_verify_checks =
      matvec_stats.leaf_pair_directness_verify_checks;
  result.leaf_pair_directness_verify_mismatches =
      matvec_stats.leaf_pair_directness_verify_mismatches;
  result.leaf_pair_scheduler = matvec_stats.leaf_pair_scheduler;
  result.leaf_pair_worklist_max_active =
      matvec_stats.leaf_pair_worklist_max_active;
  result.leaf_pair_worklist_verify = matvec_stats.leaf_pair_worklist_verify;
  result.leaf_pair_worklist_tasks = matvec_stats.leaf_pair_worklist_tasks;
  result.leaf_pair_worklist_groups = matvec_stats.leaf_pair_worklist_groups;
  result.leaf_pair_worklist_batched_groups =
      matvec_stats.leaf_pair_worklist_batched_groups;
  result.leaf_pair_worklist_scalar_flush_tasks =
      matvec_stats.leaf_pair_worklist_scalar_flush_tasks;
  result.leaf_pair_worklist_forced_flushes =
      matvec_stats.leaf_pair_worklist_forced_flushes;
  result.leaf_pair_worklist_active_groups_max =
      matvec_stats.leaf_pair_worklist_active_groups_max;
  result.leaf_pair_worklist_ms = matvec_stats.leaf_pair_worklist_ms;
  result.leaf_pair_worklist_verify_checks =
      matvec_stats.leaf_pair_worklist_verify_checks;
  result.leaf_pair_worklist_verify_mismatches =
      matvec_stats.leaf_pair_worklist_verify_mismatches;
  result.leaf_block_verify = matvec_stats.leaf_block_verify;
  result.leaf_block_target_leaves = matvec_stats.leaf_block_target_leaves;
  result.leaf_block_targets = matvec_stats.leaf_block_targets;
  result.leaf_block_source_groups = matvec_stats.leaf_block_source_groups;
  result.leaf_block_batched_groups = matvec_stats.leaf_block_batched_groups;
  result.leaf_block_scalar_tail_tasks =
      matvec_stats.leaf_block_scalar_tail_tasks;
  result.leaf_block_group_ms = matvec_stats.leaf_block_group_ms;
  result.leaf_block_flush_ms = matvec_stats.leaf_block_flush_ms;
  result.leaf_block_writeback_ms = matvec_stats.leaf_block_writeback_ms;
  result.leaf_block_verify_checks = matvec_stats.leaf_block_verify_checks;
  result.leaf_block_verify_mismatches =
      matvec_stats.leaf_block_verify_mismatches;
  result.adaptive_level = matvec_stats.adaptive_level;
  result.cluster_policy = matvec_stats.cluster_policy;
  result.target_batch_size = matvec_stats.target_batch_size;
  result.leaf_pair_min_source_panels = matvec_stats.leaf_pair_min_source_panels;
  result.leaf_pair_min_targets = matvec_stats.leaf_pair_min_targets;
  result.leaf_near_policy = matvec_stats.leaf_near_policy;
  result.leaf_panel_kernel = matvec_stats.leaf_panel_kernel;
  result.interaction_mode = matvec_stats.interaction_mode;
  result.node_bounds_mode = matvec_stats.node_bounds_mode;
  result.tree_builder = matvec_stats.tree_builder;
  result.expansion_center = matvec_stats.expansion_center;
  result.fragmentation_safety = matvec_stats.fragmentation_safety;
  result.mac_safety = matvec_stats.mac_safety;
  result.mac_gap_factor = matvec_stats.mac_gap_factor;
  result.tree_depth_max = matvec_stats.tree_depth_max;
  result.mac_reject_radius_limited = matvec_stats.mac_reject_radius_limited;
  result.mac_accept_rate = matvec_stats.mac_accept_rate;
  result.node_radius_max = matvec_stats.node_radius_max;
  result.node_radius_mean = matvec_stats.node_radius_mean;
  result.node_aspect_ratio_max = matvec_stats.node_aspect_ratio_max;
  result.node_aspect_ratio_mean = matvec_stats.node_aspect_ratio_mean;
  result.leaf_panel_work_per_matvec = matvec_stats.leaf_panel_work_per_matvec;
  result.far_node_work_per_matvec = matvec_stats.far_node_work_per_matvec;
  result.far_geometric_unsafe_count = matvec_stats.far_geometric_unsafe_count;
  result.far_geometric_unsafe_panels = matvec_stats.far_geometric_unsafe_panels;
  result.far_geometric_min_gap = matvec_stats.far_geometric_min_gap;
  result.far_geometric_gap_ratio_min = matvec_stats.far_geometric_gap_ratio_min;
  result.far_geometric_gap_ratio_mean =
      matvec_stats.far_geometric_gap_ratio_mean;
  result.far_geometric_audit_samples = matvec_stats.far_geometric_audit_samples;
  result.mac_physical_gap_rejects = matvec_stats.mac_physical_gap_rejects;
  result.mac_convergence_rho_rejects = matvec_stats.mac_convergence_rho_rejects;
  result.mac_convergence_aspect_rejects =
      matvec_stats.mac_convergence_aspect_rejects;
  result.mac_convergence_rho_reject_panels =
      matvec_stats.mac_convergence_rho_reject_panels;
  result.mac_convergence_aspect_reject_panels =
      matvec_stats.mac_convergence_aspect_reject_panels;
  result.fragmented_far_count = matvec_stats.fragmented_far_count;
  result.fragmented_far_panels = matvec_stats.fragmented_far_panels;
  result.fragmented_split_rejects = matvec_stats.fragmented_split_rejects;
  result.fragmentation_center_gap_ratio_max =
      matvec_stats.fragmentation_center_gap_ratio_max;
  result.fragmentation_center_gap_ratio_mean =
      matvec_stats.fragmentation_center_gap_ratio_mean;
  result.moment_audit = matvec_stats.moment_audit;
  result.moment_construction = matvec_stats.moment_construction;
  result.moment_sum = matvec_stats.moment_sum;
  result.convergence_audit = matvec_stats.convergence_audit;
  result.convergence_safety = matvec_stats.convergence_safety;
  result.convergence_rho_cap = matvec_stats.convergence_rho_cap;
  result.mac_aspect_cap = matvec_stats.mac_aspect_cap;
  result.convergence_estimator = matvec_stats.convergence_estimator;
  result.convergence_tail_threshold = matvec_stats.convergence_tail_threshold;
  result.convergence_estimator_error_cap =
      matvec_stats.convergence_estimator_error_cap;
  result.convergence_unstable_rho_cap =
      matvec_stats.convergence_unstable_rho_cap;
  result.convergence_calibration = matvec_stats.convergence_calibration;
  result.convergence_dielectric_amp = matvec_stats.convergence_dielectric_amp;
  result.convergence_estimator_model = matvec_stats.convergence_estimator_model;
  result.convergence_channel_calibration =
      matvec_stats.convergence_channel_calibration;
  result.convergence_split_policy = matvec_stats.convergence_split_policy;
  result.convergence_split_selectivity =
      matvec_stats.convergence_split_selectivity;
  result.convergence_split_absolute_floor =
      matvec_stats.convergence_split_absolute_floor;
  result.convergence_overreject_audit =
      matvec_stats.convergence_overreject_audit;
  result.convergence_overreject_detail =
      matvec_stats.convergence_overreject_detail;
  result.convergence_shape_audit = matvec_stats.convergence_shape_audit;
  result.convergence_kernel_response_audit =
      matvec_stats.convergence_kernel_response_audit;
  result.channel_missed_semantics = matvec_stats.channel_missed_semantics;
  result.accepted_miss_diagnostics = matvec_stats.accepted_miss_diagnostics;
  result.accepted_miss_absolute_floor =
      matvec_stats.accepted_miss_absolute_floor;
  result.convergence_k3_accepted_rho_cap =
      matvec_stats.convergence_k3_accepted_rho_cap;
  result.mac_distance_policy = matvec_stats.mac_distance_policy;
  result.mac_surface_margin_factor = matvec_stats.mac_surface_margin_factor;
  result.mac_surface_margin_policy = matvec_stats.mac_surface_margin_policy;
  result.mac_surface_margin_aspect_threshold =
      matvec_stats.mac_surface_margin_aspect_threshold;
  result.mac_surface_margin_rho_trigger =
      matvec_stats.mac_surface_margin_rho_trigger;
  result.mac_surface_margin_soft_aspect_threshold =
      matvec_stats.mac_surface_margin_soft_aspect_threshold;
  result.mac_surface_margin_soft_rho_trigger =
      matvec_stats.mac_surface_margin_soft_rho_trigger;
  result.mac_surface_margin_soft_factor =
      matvec_stats.mac_surface_margin_soft_factor;
  result.convergence_estimator_samples =
      matvec_stats.convergence_estimator_samples;
  result.convergence_estimator_unstable_far_count =
      matvec_stats.convergence_estimator_unstable_far_count;
  result.convergence_estimator_unstable_far_panels =
      matvec_stats.convergence_estimator_unstable_far_panels;
  result.convergence_estimator_split_rejects =
      matvec_stats.convergence_estimator_split_rejects;
  result.convergence_estimator_split_reject_panels =
      matvec_stats.convergence_estimator_split_reject_panels;
  result.convergence_low_abs_reaccepted_far_count =
      matvec_stats.convergence_low_abs_reaccepted_far_count;
  result.convergence_low_abs_reaccepted_far_panels =
      matvec_stats.convergence_low_abs_reaccepted_far_panels;
  result.convergence_estimator_tail_ratio_max =
      matvec_stats.convergence_estimator_tail_ratio_max;
  result.convergence_estimator_tail_ratio_mean =
      matvec_stats.convergence_estimator_tail_ratio_mean;
  result.convergence_estimator_error_max =
      matvec_stats.convergence_estimator_error_max;
  result.convergence_estimator_error_mean =
      matvec_stats.convergence_estimator_error_mean;
  result.convergence_estimator_abs_error_max =
      matvec_stats.convergence_estimator_abs_error_max;
  result.convergence_estimator_abs_error_mean =
      matvec_stats.convergence_estimator_abs_error_mean;
  result.convergence_tail_norm_p_max = matvec_stats.convergence_tail_norm_p_max;
  result.convergence_tail_norm_p_mean =
      matvec_stats.convergence_tail_norm_p_mean;
  result.convergence_tail_norm_pm1_max =
      matvec_stats.convergence_tail_norm_pm1_max;
  result.convergence_tail_norm_pm1_mean =
      matvec_stats.convergence_tail_norm_pm1_mean;
  result.convergence_effective_jcp_amp_max =
      matvec_stats.convergence_effective_jcp_amp_max;
  result.convergence_effective_jcp_amp_mean =
      matvec_stats.convergence_effective_jcp_amp_mean;
  result.convergence_dielectric_amp_max =
      matvec_stats.convergence_dielectric_amp_max;
  result.convergence_dielectric_amp_mean =
      matvec_stats.convergence_dielectric_amp_mean;
  result.convergence_channel_estimated_error_max =
      matvec_stats.convergence_channel_estimated_error_max;
  result.convergence_channel_estimated_error_mean =
      matvec_stats.convergence_channel_estimated_error_mean;
  result.convergence_channel_effective_jcp_amp_max =
      matvec_stats.convergence_channel_effective_jcp_amp_max;
  result.convergence_channel_effective_jcp_amp_mean =
      matvec_stats.convergence_channel_effective_jcp_amp_mean;
  result.convergence_channel_unstable_far_count =
      matvec_stats.convergence_channel_unstable_far_count;
  result.convergence_channel_unstable_far_panels =
      matvec_stats.convergence_channel_unstable_far_panels;
  result.convergence_channel_split_rejects =
      matvec_stats.convergence_channel_split_rejects;
  result.convergence_channel_split_reject_panels =
      matvec_stats.convergence_channel_split_reject_panels;
  result.convergence_channel_low_abs_reaccepted_count =
      matvec_stats.convergence_channel_low_abs_reaccepted_count;
  result.convergence_channel_low_abs_reaccepted_panels =
      matvec_stats.convergence_channel_low_abs_reaccepted_panels;
  result.convergence_channel_high_error_caught =
      matvec_stats.convergence_channel_high_error_caught;
  result.convergence_channel_high_error_missed =
      matvec_stats.convergence_channel_high_error_missed;
  result.convergence_channel_high_error_caught_by_split =
      matvec_stats.convergence_channel_high_error_caught_by_split;
  result.convergence_estimator_high_error_caught =
      matvec_stats.convergence_estimator_high_error_caught;
  result.convergence_estimator_high_error_missed =
      matvec_stats.convergence_estimator_high_error_missed;
  result.convergence_estimator_high_error_caught_by_split =
      matvec_stats.convergence_estimator_high_error_caught_by_split;
  result.convergence_estimator_phantom_high_error_count =
      matvec_stats.convergence_estimator_phantom_high_error_count;
  result.convergence_estimator_phantom_high_error_abs_max =
      matvec_stats.convergence_estimator_phantom_high_error_abs_max;
  result.convergence_overreject_count =
      matvec_stats.convergence_overreject_count;
  result.convergence_overreject_panels =
      matvec_stats.convergence_overreject_panels;
  result.convergence_overreject_caught_count =
      matvec_stats.convergence_overreject_caught_count;
  result.convergence_overreject_caught_panels =
      matvec_stats.convergence_overreject_caught_panels;
  result.convergence_overreject_abs_error_max =
      matvec_stats.convergence_overreject_abs_error_max;
  result.convergence_overreject_abs_error_mean =
      matvec_stats.convergence_overreject_abs_error_mean;
  result.convergence_overreject_abs_error_p95 =
      matvec_stats.convergence_overreject_abs_error_p95;
  result.convergence_overreject_abs_error_p99 =
      matvec_stats.convergence_overreject_abs_error_p99;
  result.convergence_overreject_estimated_abs_error_max =
      matvec_stats.convergence_overreject_estimated_abs_error_max;
  result.convergence_overreject_estimated_abs_error_mean =
      matvec_stats.convergence_overreject_estimated_abs_error_mean;
  result.convergence_overreject_required_scale_max =
      matvec_stats.convergence_overreject_required_scale_max;
  result.convergence_overreject_required_scale_p95 =
      matvec_stats.convergence_overreject_required_scale_p95;
  result.convergence_overreject_required_scale_p99 =
      matvec_stats.convergence_overreject_required_scale_p99;
  result.convergence_channel_overreject_count =
      matvec_stats.convergence_channel_overreject_count;
  result.convergence_channel_overreject_caught_count =
      matvec_stats.convergence_channel_overreject_caught_count;
  result.convergence_channel_overreject_panels =
      matvec_stats.convergence_channel_overreject_panels;
  result.convergence_channel_overreject_abs_error_max =
      matvec_stats.convergence_channel_overreject_abs_error_max;
  result.convergence_channel_overreject_abs_error_mean =
      matvec_stats.convergence_channel_overreject_abs_error_mean;
  result.convergence_channel_overreject_required_scale_max =
      matvec_stats.convergence_channel_overreject_required_scale_max;
  result.convergence_channel_overreject_required_scale_p99 =
      matvec_stats.convergence_channel_overreject_required_scale_p99;
  result.phantom_high_error_channel_count =
      matvec_stats.phantom_high_error_channel_count;
  result.phantom_high_error_channel_abs_max =
      matvec_stats.phantom_high_error_channel_abs_max;
  result.accepted_miss_channel_count = matvec_stats.accepted_miss_channel_count;
  result.accepted_miss_channel_panels =
      matvec_stats.accepted_miss_channel_panels;
  result.accepted_miss_channel_rho_min =
      matvec_stats.accepted_miss_channel_rho_min;
  result.accepted_miss_channel_rho_mean =
      matvec_stats.accepted_miss_channel_rho_mean;
  result.accepted_miss_channel_rho_max =
      matvec_stats.accepted_miss_channel_rho_max;
  result.accepted_miss_channel_tail_ratio_mean =
      matvec_stats.accepted_miss_channel_tail_ratio_mean;
  result.accepted_miss_channel_tail_ratio_max =
      matvec_stats.accepted_miss_channel_tail_ratio_max;
  result.accepted_miss_channel_estimated_error_mean =
      matvec_stats.accepted_miss_channel_estimated_error_mean;
  result.accepted_miss_channel_estimated_error_max =
      matvec_stats.accepted_miss_channel_estimated_error_max;
  result.accepted_miss_worst_channel = matvec_stats.accepted_miss_worst_channel;
  result.convergence_k3_rho_cap_rejects =
      matvec_stats.convergence_k3_rho_cap_rejects;
  result.convergence_k3_rho_cap_reject_panels =
      matvec_stats.convergence_k3_rho_cap_reject_panels;
  result.mac_surface_margin_rejects = matvec_stats.mac_surface_margin_rejects;
  result.mac_surface_margin_reject_panels =
      matvec_stats.mac_surface_margin_reject_panels;
  result.mac_surface_margin_applied = matvec_stats.mac_surface_margin_applied;
  result.mac_surface_margin_skipped = matvec_stats.mac_surface_margin_skipped;
  result.mac_surface_margin_adaptive_rejects =
      matvec_stats.mac_surface_margin_adaptive_rejects;
  result.mac_surface_margin_adaptive_reject_panels =
      matvec_stats.mac_surface_margin_adaptive_reject_panels;
  result.mac_surface_margin_skipped_accepted_far =
      matvec_stats.mac_surface_margin_skipped_accepted_far;
  result.mac_surface_margin_hard_applied =
      matvec_stats.mac_surface_margin_hard_applied;
  result.mac_surface_margin_soft_applied =
      matvec_stats.mac_surface_margin_soft_applied;
  result.mac_surface_margin_hard_rejects =
      matvec_stats.mac_surface_margin_hard_rejects;
  result.mac_surface_margin_hard_reject_panels =
      matvec_stats.mac_surface_margin_hard_reject_panels;
  result.mac_surface_margin_soft_rejects =
      matvec_stats.mac_surface_margin_soft_rejects;
  result.mac_surface_margin_soft_reject_panels =
      matvec_stats.mac_surface_margin_soft_reject_panels;
  result.moment_audit_nodes = matvec_stats.moment_audit_nodes;
  result.moment_translation_rel_error_max =
      matvec_stats.moment_translation_rel_error_max;
  result.moment_translation_rel_error_mean =
      matvec_stats.moment_translation_rel_error_mean;
  result.moment_translation_offset_max =
      matvec_stats.moment_translation_offset_max;
  result.moment_translation_offset_mean =
      matvec_stats.moment_translation_offset_mean;
  result.moment_cancellation_ratio_max =
      matvec_stats.moment_cancellation_ratio_max;
  result.moment_cancellation_ratio_mean =
      matvec_stats.moment_cancellation_ratio_mean;
  result.moment_dynamic_range_max = matvec_stats.moment_dynamic_range_max;
  result.expansion_audit = matvec_stats.expansion_audit;
  result.expansion_audit_channel = matvec_stats.expansion_audit_channel;
  result.far_expansion_audit_pairs = matvec_stats.far_expansion_audit_pairs;
  result.far_expansion_audit_panels = matvec_stats.far_expansion_audit_panels;
  result.far_expansion_direct_ms = matvec_stats.far_expansion_direct_ms;
  result.far_expansion_rel_error_max = matvec_stats.far_expansion_rel_error_max;
  result.far_expansion_rel_error_mean =
      matvec_stats.far_expansion_rel_error_mean;
  result.far_expansion_rel_error_p95 = matvec_stats.far_expansion_rel_error_p95;
  result.far_expansion_rel_error_p99 = matvec_stats.far_expansion_rel_error_p99;
  result.far_expansion_channel_rel_error_max =
      matvec_stats.far_expansion_channel_rel_error_max;
  result.far_expansion_channel_rel_error_mean =
      matvec_stats.far_expansion_channel_rel_error_mean;
  result.far_expansion_channel_rel_error_p99 =
      matvec_stats.far_expansion_channel_rel_error_p99;
  result.far_expansion_worst_channel = matvec_stats.far_expansion_worst_channel;
  result.far_expansion_mean_gate_pass =
      matvec_stats.far_expansion_mean_gate_pass;
  result.far_convergence_rho_max = matvec_stats.far_convergence_rho_max;
  result.far_convergence_rho_mean = matvec_stats.far_convergence_rho_mean;
  result.accepted_far_aspect_ratio_max =
      matvec_stats.accepted_far_aspect_ratio_max;
  result.accepted_far_aspect_ratio_mean =
      matvec_stats.accepted_far_aspect_ratio_mean;
  result.leaf_tile_status = matvec_stats.leaf_tile_status;
  result.leaf_panel_pairs_per_second = matvec_stats.leaf_panel_pairs_per_second;
  result.matvec_mode = matvec_stats.matvec_mode;
  result.taylor_order = mode == "treecode"
                            ? matvec_stats.taylor_order
                            : std::clamp(options.taylor_order, 2, 5);
  result.derivative_mode =
      mode == "treecode" ? matvec_stats.derivative_mode : std::string("none");
  result.interaction_build_ms = matvec_stats.interaction_build_ms;
  result.tree_eval_ms = matvec_stats.tree_eval_ms;
  result.tree_moment_ms = matvec_stats.tree_moment_ms;
  result.tree_traversal_ms = matvec_stats.tree_traversal_ms;
  result.near_eval_ms = matvec_stats.near_eval_ms;
  result.far_eval_ms = matvec_stats.far_eval_ms;
  result.leaf_panel_eval_ms = matvec_stats.leaf_panel_eval_ms;
  result.leaf_tile_eval_ms = matvec_stats.leaf_tile_eval_ms;
  result.leaf_residual_eval_ms = matvec_stats.leaf_residual_eval_ms;
  result.smooth_leaf_node_eval_ms = matvec_stats.smooth_leaf_node_eval_ms;

  const double boundary_energy =
      boundary_energy_kcal(atoms, panels, linear.x, kappa) *
      formulation.boundary_energy_scale;
  const double born =
      born_reference_kcal(atoms, options.eps_in, options.eps_out, kappa);
  double energy = boundary_energy;
  double selected_scale = formulation.boundary_energy_scale;
  bool used_fallback = false;
  std::string source = "tabi_boundary_integral";
  if (formulation.energy_projection == "born-check") {
    energy = born;
    source = "tabi_born_check";
    selected_scale = 0.0;
  } else if (formulation.formulation == "calibrated-debug" &&
             std::isfinite(boundary_energy) &&
             std::abs(boundary_energy) > 1.0e-12 && std::isfinite(born) &&
             std::abs(born) > 1.0e-12) {
    selected_scale = born / boundary_energy;
    energy = boundary_energy * selected_scale;
    source = "tabi_calibrated_debug";
  } else if (formulation.energy_projection == "charge-potential-debug") {
    source = "tabi_charge_potential_debug";
  } else if (!std::isfinite(energy) || std::abs(energy) <= 1.0e-12 ||
             (std::abs(born) > 1.0e-12 && energy * born < 0.0)) {
    energy = born;
    used_fallback = true;
    source = "tabi_born_fallback";
  }
  result.boundary_energy_kcal_mol = boundary_energy;
  result.reference_energy_kcal_mol = born;
  result.selected_energy_scale = selected_scale;
  result.fallback_used = used_fallback;
  result.reaction_energy_source = source;
  if (!linear.converged && result.convergence_reason.empty())
    result.convergence_reason = "max_iter";
  if (!linear.converged || !std::isfinite(energy)) {
    result.status = "failed";
    if (!std::isfinite(energy))
      result.convergence_reason = "nan_guard";
  } else if (used_fallback) {
    result.status = "fallback";
  } else {
    result.status = "ok";
  }
  result.energy_kcal_mol = energy;
  result.energy_kj_mol = energy * kKcalToKJ;
  result.energy_kt = result.energy_kj_mol /
                     (kKBKJPerMolK * std::max(1.0, options.temperature_k));
  result.peak_memory_bytes = result.memory_bytes;
  refresh_matvec_cost_summary(result);
  result.node_patch_certificate =
      node_patch_certificate(panels) +
      node_patch_solution_shape(panels, linear.x,
                                result.bie_system == "linear_pb_2x2");
  if (options.surface_output != "off") {
    result.output_panels = panels;
    result.output_solution = linear.x;
    result.output_rhs = rhs;
  }
  progress.write("solve_done", result, &matvec_stats, result.matvec_calls,
                 result.iterations, 0, result.residual, result.status, true);
  return result;
}

namespace {

constexpr std::string_view kCartesianRetryAuditMode =
    "channel-shadow-retry-global-theta015";

std::string cartesian_retry_eligibility_reason(const Options &options) {
  if (options.sum_mode != "treecode")
    return "sum_mode_not_treecode";
  if (options.formulation != "jcp2013-treecode")
    return "formulation_not_jcp2013_treecode";
  if (options.fmm_m2l != "single-matvec-analytic")
    return "fmm_m2l_not_analytic";
  if (options.fmm_engine != "cartesian-full-experimental")
    return "fmm_engine_not_cartesian_full";
  if (options.bie_fmm_cartesian_order != 1)
    return "cartesian_order_not_1";
  if (options.fmm_local != "off")
    return "local_p4_or_refinement_not_off";
  if (options.surface_geometry != "triangle")
    return "surface_geometry_not_triangle";
  if (options.discretization != "node-patch")
    return "discretization_not_node_patch";
  if (!std::isfinite(options.salt_m) || options.salt_m <= 0.0)
    return "salt_not_positive";
  if (options.node_patch_operator_audit != "on")
    return "node_patch_operator_audit_not_on";
  if (options.surface_only)
    return "surface_only_not_allowed";
  if (options.preconditioner != "fmm-nearfield-ilu0")
    return "requested_preconditioner_not_plain_ilu0";
  const int effective_leaf =
      options.tree_leaf_size_explicit ? options.tree_leaf_size : 64;
  if (effective_leaf != 64)
    return "effective_leaf_not_64";
  const double effective_theta =
      options.tree_opening_theta_explicit ? options.tree_opening_theta : 0.20;
  if (!fmm_cartesian_retry_theta_matches(effective_theta, 0.20))
    return "effective_theta_not_020";
  return {};
}

const char *cartesian_retry_disposition_name(
    FmmCartesianRetryAttemptDisposition disposition) noexcept {
  switch (disposition) {
  case FmmCartesianRetryAttemptDisposition::accept:
    return "accept";
  case FmmCartesianRetryAttemptDisposition::retry_trigger:
    return "retry_trigger";
  case FmmCartesianRetryAttemptDisposition::fail_closed:
    return "fail_closed";
  }
  return "fail_closed";
}

bool finite_nonnegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

FmmCartesianRetryAttemptTelemetry make_retry_attempt_snapshot(
    const Result &result, const Options &options, std::size_t attempt_number,
    double wall_ms, std::size_t rss_before_bytes,
    std::size_t rss_after_bytes) {
  FmmCartesianRetryAttemptTelemetry attempt;
  attempt.attempt = attempt_number;
  attempt.started = true;
  attempt.completed = true;
  attempt.status = result.status;
  attempt.convergence_reason = result.convergence_reason;
  attempt.converged = result.converged;
  attempt.theta = result.bie_cartesian_full_fmm_opening_theta;
  attempt.salt_molar = options.salt_m;
  attempt.coupled = result.bie_system == "linear_pb_2x2";
  attempt.discretization = result.discretization;
  attempt.node_patch_operator_audit_enabled =
      options.node_patch_operator_audit == "on" &&
      result.fmm_cartesian_pre_audit.enabled;
  attempt.true_residual_rel = result.true_residual_rel;

  attempt.cartesian_full_active = result.bie_cartesian_full_fmm_active;
  attempt.cartesian_full_finite = result.fmm_local_operator_finite &&
                                  result.fmm_m2l_finite_outputs;
  attempt.cartesian_full_requested = result.bie_cartesian_full_fmm_requested;
  attempt.cartesian_full_effective = result.bie_cartesian_full_fmm_effective;
  attempt.cartesian_full_order = result.bie_cartesian_full_fmm_order;
  attempt.cartesian_full_leaf_size = result.bie_cartesian_full_fmm_leaf_size;
  attempt.panel_count = result.panel_count;
  attempt.geometry_hash = result.fmm_cartesian_retry_geometry_valid
                              ? result.fmm_cartesian_retry_geometry_hash
                              : 0;
  attempt.tree_hash = result.bie_cartesian_full_fmm_tree_hash;
  attempt.worklist_hash = result.bie_cartesian_full_fmm_worklist_hash;
  attempt.policy_hash = result.bie_cartesian_full_fmm_policy_hash;
  attempt.direct_pairs_expected =
      result.bie_cartesian_full_fmm_direct_panel_pairs_expected;
  attempt.direct_pairs_applied_last =
      result.bie_cartesian_full_fmm_direct_panel_pairs_applied_last;
  attempt.direct_pairs_applied_total =
      result.bie_cartesian_full_fmm_direct_panel_pairs_applied_total;
  attempt.pre_audit_sampled_target_hash =
      result.fmm_cartesian_pre_audit.sampled_target_hash;
  attempt.solution_audit = result.fmm_cartesian_solution_audit;

  attempt.fallback_used = result.fallback_used;
  attempt.preconditioner_requested = result.preconditioner_requested;
  attempt.preconditioner_effective = result.preconditioner_effective;
  attempt.preconditioner_quality_status = result.preconditioner_quality_status;
  attempt.preconditioner_coarse_status = result.preconditioner_coarse_status;
  attempt.preconditioner_ilu_regularization =
      result.preconditioner_ilu_regularization;
  attempt.preconditioner_ilu_regularization_selected =
      result.preconditioner_ilu_regularization_selected;
  attempt.preconditioner_ilu_regularization_shifted_rows =
      result.preconditioner_ilu_regularization_shifted_rows;
  attempt.preconditioner_ilu_unshifted_failure_stage =
      result.preconditioner_ilu_unshifted_failure_stage;
  attempt.preconditioner_valid_panels = result.preconditioner_valid_panels;
  attempt.preconditioner_ill_conditioned_blocks =
      result.preconditioner_ill_conditioned_blocks;
  attempt.preconditioner_quality_rejected_blocks =
      result.preconditioner_quality_rejected_blocks;
  attempt.preconditioner_setup_ms = result.preconditioner_setup_ms;
  attempt.preconditioner_ilu_factor_setup_ms =
      result.preconditioner_ilu_factor_setup_ms;

  attempt.surface_build_ms = result.surface_build_ms;
  attempt.fmm_prepare_ms = 1000.0 * result.fmm_local_prepare_seconds;
  attempt.gmres_ms = result.gmres_ms;
  attempt.wall_ms = wall_ms;
  attempt.accounted_memory_bytes = result.memory_bytes;
  attempt.rss_before_bytes = rss_before_bytes;
  attempt.rss_after_bytes = rss_after_bytes;
  attempt.finite =
      std::isfinite(result.energy_kcal_mol) &&
      std::isfinite(result.energy_kj_mol) && std::isfinite(result.energy_kt) &&
      finite_nonnegative(result.true_residual_rel) &&
      finite_nonnegative(result.surface_build_ms) &&
      finite_nonnegative(result.fmm_local_prepare_seconds) &&
      finite_nonnegative(result.gmres_ms) && finite_nonnegative(wall_ms);
  return attempt;
}

FmmCartesianRetryAttemptClassification
classify_retry_attempt(FmmCartesianRetryAttemptTelemetry &attempt,
                       FmmCartesianRetryAttemptRole role) {
  const FmmCartesianRetryAttemptClassification classification =
      classify_fmm_cartesian_retry_attempt(attempt, role);
  attempt.classification =
      cartesian_retry_disposition_name(classification.disposition);
  attempt.classification_reason = std::string(classification.reason);
  attempt.technically_valid =
      classification.disposition != FmmCartesianRetryAttemptDisposition::fail_closed;
  attempt.accepted =
      classification.disposition == FmmCartesianRetryAttemptDisposition::accept;
  return classification;
}

FmmCartesianRetryAttemptTelemetry make_retry_inflight_attempt(
    std::size_t attempt_number, const Options &options, double theta,
    std::size_t rss_before_bytes) {
  FmmCartesianRetryAttemptTelemetry attempt;
  attempt.attempt = attempt_number;
  attempt.started = true;
  attempt.status = "running";
  attempt.convergence_reason = "running";
  attempt.theta = theta;
  attempt.salt_molar = options.salt_m;
  attempt.discretization = options.discretization;
  attempt.rss_before_bytes = rss_before_bytes;
  attempt.classification = "running";
  attempt.classification_reason = "not_evaluated";
  return attempt;
}

class CartesianRetryTraceWriter {
public:
  CartesianRetryTraceWriter() {
    out_.open("pbe_tabi_fmm_cartesian_retry.csv",
              std::ios::out | std::ios::trunc);
    if (!out_.is_open()) {
      throw std::runtime_error("cartesian_full_fmm_retry_trace_open_failed");
    }
    // The retry decision gate is 1e-4.  Keep every persisted scalar at
    // round-trip precision so trace replay cannot move a value across it.
    out_ << std::setprecision(std::numeric_limits<double>::max_digits10);
    out_ << "event,attempt,theta,status,converged,classification,"
            "classification_reason,reason,"
            "solution_audit_status,solution_audit_rel,true_residual_rel,"
            "panels,geometry_hash,tree_hash,worklist_hash,policy_hash,"
            "identity_reason\n";
    flush_or_throw("header");
  }

  void write(std::string_view event,
             const FmmCartesianRetryAttemptTelemetry &attempt,
             std::string_view reason,
             std::string_view identity_reason = "off") {
    out_ << csv_escape_progress(std::string(event)) << ',' << attempt.attempt
         << ',' << attempt.theta << ',' << csv_escape_progress(attempt.status)
         << ',' << (attempt.converged ? "true" : "false") << ','
         << csv_escape_progress(attempt.classification) << ','
         << csv_escape_progress(attempt.classification_reason) << ','
         << csv_escape_progress(std::string(reason)) << ','
         << csv_escape_progress(attempt.solution_audit.status) << ','
         << attempt.solution_audit.relative_error << ','
         << attempt.true_residual_rel << ',' << attempt.panel_count << ','
         << attempt.geometry_hash << ',' << attempt.tree_hash << ','
         << attempt.worklist_hash << ',' << attempt.policy_hash << ','
         << csv_escape_progress(std::string(identity_reason)) << '\n';
    flush_or_throw("row");
  }

private:
  void flush_or_throw(std::string_view stage) {
    out_.flush();
    if (!out_) {
      throw std::runtime_error("cartesian_full_fmm_retry_trace_write_failed:" +
                               std::string(stage));
    }
  }

  std::ofstream out_;
};

void write_cartesian_retry_trace_best_effort(
    CartesianRetryTraceWriter &trace, std::string_view event,
    const FmmCartesianRetryAttemptTelemetry &attempt,
    std::string_view reason,
    std::string_view identity_reason = "off") noexcept {
  try {
    trace.write(event, attempt, reason, identity_reason);
  } catch (...) {
    // The primary solver exception is more important than a diagnostic sink.
  }
}

std::string_view
cartesian_retry_exception_reason(const std::exception_ptr &original) noexcept {
  try {
    std::rethrow_exception(original);
  } catch (const std::exception &error) {
    return error.what();
  } catch (...) {
    return "unknown_exception_rethrown";
  }
}

bool preserve_cartesian_retry_progress(const Options &options,
                                       std::size_t attempt_number) {
  if (options.progress == "off")
    return false;
  const std::string source = "pbe_tabi_longrun_progress.csv";
  const std::string destination =
      "pbe_tabi_longrun_progress_retry_attempt" +
      std::to_string(attempt_number) + ".csv";
  std::ifstream input(source, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error(
        "cartesian_full_fmm_retry_progress_source_open_failed:" + source);
  }
  std::ofstream output(destination,
                       std::ios::out | std::ios::trunc | std::ios::binary);
  if (!output.is_open()) {
    throw std::runtime_error(
        "cartesian_full_fmm_retry_progress_destination_open_failed:" +
        destination);
  }
  output << input.rdbuf();
  output.flush();
  if (input.bad() || !output) {
    throw std::runtime_error(
        "cartesian_full_fmm_retry_progress_copy_failed:" + source + "->" +
        destination);
  }
  return true;
}

void record_retry_attempt(FmmCartesianRetryTelemetry &retry,
                          const FmmCartesianRetryAttemptTelemetry &attempt) {
  if (attempt.attempt == 0 || attempt.attempt > retry.attempts.size()) {
    throw std::logic_error("cartesian_full_fmm_retry_attempt_index_invalid");
  }
  retry.attempts[attempt.attempt - 1u] = attempt;
  ++retry.attempts_started;
  if (attempt.completed)
    ++retry.attempts_completed;
  if (!std::isfinite(attempt.wall_ms) || !std::isfinite(attempt.gmres_ms)) {
    throw std::runtime_error("cartesian_full_fmm_retry_timing_nonfinite");
  }
  retry.cumulative_wall_ms += attempt.wall_ms;
  retry.cumulative_gmres_ms += attempt.gmres_ms;
  if (retry.cumulative_accounted_memory_bytes >
      std::numeric_limits<std::size_t>::max() -
          attempt.accounted_memory_bytes) {
    throw std::overflow_error(
        "cartesian_full_fmm_retry_accounted_memory_overflow");
  }
  retry.cumulative_accounted_memory_bytes += attempt.accounted_memory_bytes;
  retry.max_boundary_rss_bytes = std::max(
      retry.max_boundary_rss_bytes,
      std::max(attempt.rss_before_bytes, attempt.rss_after_bytes));
}

struct CompletedCartesianRetryAttempt {
  Result result;
  FmmCartesianRetryAttemptTelemetry telemetry;
  FmmCartesianRetryAttemptClassification classification;
};

} // namespace

Result solve_linear_pb(const std::vector<Atom> &atoms, const Options &options,
                       TitanExecutor &executor) {
  if (options.fmm_cartesian_error_audit != kCartesianRetryAuditMode) {
    return solve_linear_pb_single_attempt(atoms, options, executor);
  }

  const std::string eligibility_reason =
      cartesian_retry_eligibility_reason(options);
  if (!eligibility_reason.empty()) {
    throw std::invalid_argument(
        "cartesian_full_fmm_retry_scope_rejected:" + eligibility_reason);
  }

  CartesianRetryTraceWriter trace;
  FmmCartesianRetryTelemetry retry;
  retry.mode = std::string(kCartesianRetryAuditMode);
  retry.enabled = true;
  FmmCartesianRetryAttemptTelemetry trace_context;

  const auto run_attempt =
      [&](std::size_t attempt_number, const Options &attempt_options,
          double requested_theta, FmmCartesianRetryAttemptRole role) {
        const auto begin = std::chrono::steady_clock::now();
        const std::size_t rss_before = current_working_set_bytes();
        FmmCartesianRetryAttemptTelemetry inflight =
            make_retry_inflight_attempt(attempt_number, attempt_options,
                                        requested_theta, rss_before);
        FmmCartesianRetryAttemptTelemetry trace_attempt = inflight;
        trace.write("attempt_started", inflight, "started");
        try {
          Result attempt_result =
              solve_linear_pb_single_attempt(atoms, attempt_options, executor);
          FmmCartesianRetryAttemptTelemetry attempt =
              make_retry_attempt_snapshot(
                  attempt_result, attempt_options, attempt_number,
                  elapsed_ms(begin, std::chrono::steady_clock::now()),
                  rss_before, current_working_set_bytes());
          const FmmCartesianRetryAttemptClassification classification =
              classify_retry_attempt(attempt, role);
          trace_attempt = attempt;
          trace.write("attempt_completed", attempt,
                      attempt.classification_reason);
          if (preserve_cartesian_retry_progress(attempt_options,
                                                attempt_number)) {
            trace.write("progress_preserved", attempt,
                        "canonical_progress_preserved");
          } else {
            trace.write("progress_disabled", attempt,
                        "canonical_progress_disabled");
          }
          return CompletedCartesianRetryAttempt{std::move(attempt_result),
                                                std::move(attempt),
                                                classification};
        } catch (...) {
          const std::exception_ptr original = std::current_exception();
          // All exception-path diagnostics are deliberately best-effort: no
          // trace/progress I/O error may replace the original solver failure.
          try {
            if (!trace_attempt.completed) {
              trace_attempt.status = "exception";
              trace_attempt.convergence_reason = "exception";
              trace_attempt.classification = "exception";
              trace_attempt.classification_reason = "exception_rethrown";
              trace_attempt.wall_ms =
                  elapsed_ms(begin, std::chrono::steady_clock::now());
              trace_attempt.rss_after_bytes = current_working_set_bytes();
            }
            write_cartesian_retry_trace_best_effort(
                trace, "attempt_exception", trace_attempt,
                cartesian_retry_exception_reason(original));
            // A solver exception can occur after LongRunProgressWriter has
            // emitted a useful partial or completed canonical file. Preserve
            // it best-effort without replacing the original scientific
            // exception when the copy or trace cannot be completed.
            if (attempt_options.progress != "off") {
              try {
                if (preserve_cartesian_retry_progress(attempt_options,
                                                      attempt_number)) {
                  write_cartesian_retry_trace_best_effort(
                      trace,
                      trace_attempt.completed
                          ? "progress_preserved_after_exception"
                          : "progress_preserved_partial",
                      trace_attempt,
                      "canonical_progress_preserved_after_exception");
                }
              } catch (...) {
                const std::exception_ptr copy_error =
                    std::current_exception();
                write_cartesian_retry_trace_best_effort(
                    trace, "progress_preserve_failed", trace_attempt,
                    cartesian_retry_exception_reason(copy_error));
              }
            }
          } catch (...) {
            // Even allocation failure while composing telemetry is secondary.
          }
          std::rethrow_exception(original);
        }
      };

  try {
    Options first_options = options;
    first_options.fmm_cartesian_error_audit = "channel-shadow";
    first_options.fmm_cartesian_retry_runtime = true;
    CompletedCartesianRetryAttempt first = run_attempt(
        1u, first_options, 0.20,
        FmmCartesianRetryAttemptRole::initial_theta_020);
    trace_context = first.telemetry;
    record_retry_attempt(retry, first.telemetry);

    const FmmCartesianRetryWrapperDecision first_decision =
        decide_fmm_cartesian_retry_after_attempt1(first.classification);
    if (first_decision.action == FmmCartesianRetryWrapperAction::accept_attempt1) {
      retry.accepted_attempt = first_decision.accepted_attempt;
      retry.trigger_reason = "not_needed";
      retry.final_reason = "accepted_attempt1";
      first.result.fmm_cartesian_retry = std::move(retry);
      trace.write("attempt_accepted", trace_context, "accepted_attempt1");
      return std::move(first.result);
    }
    if (first_decision.action != FmmCartesianRetryWrapperAction::run_attempt2) {
      retry.trigger_reason = first.telemetry.classification_reason;
      retry.final_reason = std::string(first_decision.reason);
      trace.write("retry_fail_closed", trace_context, retry.final_reason);
      return sanitize_cartesian_retry_failure(std::move(first.result),
                                              std::move(retry));
    }

    retry.trigger_reason = first.telemetry.classification_reason;
    trace.write("retry_triggered", trace_context, retry.trigger_reason);
    // Release the first surface/operator/preconditioner result before the
    // second complete solve constructs its own independent state.
    first.result = Result{};

    Options second_options = first_options;
    second_options.tree_opening_theta = 0.15;
    second_options.tree_opening_theta_explicit = true;
    CompletedCartesianRetryAttempt second = run_attempt(
        2u, second_options, 0.15,
        FmmCartesianRetryAttemptRole::tightened_theta_015);
    trace_context = second.telemetry;
    record_retry_attempt(retry, second.telemetry);

    const FmmCartesianRetryIdentityClassification identity =
        classify_fmm_cartesian_retry_identity(retry.attempts[0],
                                              retry.attempts[1]);
    retry.geometry_identity_valid =
        retry.attempts[0].geometry_hash != 0 &&
        retry.attempts[0].geometry_hash == retry.attempts[1].geometry_hash;
    retry.tree_identity_valid = retry.attempts[0].tree_hash != 0 &&
                                retry.attempts[0].tree_hash ==
                                    retry.attempts[1].tree_hash;
    retry.policy_transition_valid = retry.attempts[0].policy_hash != 0 &&
                                    retry.attempts[1].policy_hash != 0 &&
                                    retry.attempts[0].policy_hash !=
                                        retry.attempts[1].policy_hash;
    trace.write("cross_attempt_identity", second.telemetry,
                identity.reason, identity.reason);

    const FmmCartesianRetryWrapperDecision second_decision =
        decide_fmm_cartesian_retry_after_attempt2(second.classification,
                                                  identity);
    if (second_decision.action == FmmCartesianRetryWrapperAction::accept_attempt2) {
      retry.accepted_attempt = second_decision.accepted_attempt;
      retry.final_reason = "accepted_attempt2";
      second.result.fmm_cartesian_retry = std::move(retry);
      trace.write("attempt_accepted", trace_context, "accepted_attempt2",
                  identity.reason);
      return std::move(second.result);
    }

    second.telemetry.accepted = false;
    retry.attempts[1].accepted = false;
    retry.final_reason = std::string(second_decision.reason);
    trace.write("retry_fail_closed", trace_context, retry.final_reason,
                identity.reason);
    return sanitize_cartesian_retry_failure(std::move(second.result),
                                            std::move(retry));
  } catch (...) {
    const std::exception_ptr original = std::current_exception();
    write_cartesian_retry_trace_best_effort(
        trace, "wrapper_exception", trace_context,
        cartesian_retry_exception_reason(original));
    std::rethrow_exception(original);
  }
}

} // namespace TitanPBE::Tabi
