#ifndef TITAN_PBE_TYPES_H
#define TITAN_PBE_TYPES_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace TitanPBE {

struct ChargeResolutionDiagnostic {
  // min over charged atoms and retained curved panels of distance/max-edge.
  // Negative numeric sentinels mean unavailable; no_flag is not certification.
  std::string status{"not_evaluated"};
  double min_ratio{-1.0};
  double distance_a{-1.0};
  double local_edge_a{-1.0};
  double min_distance_over_atom_radius{-1.0};
  std::int64_t charge_index{-1};
  std::int64_t panel_index{-1};
  std::size_t charged_atoms{0};
  std::size_t surface_inside_atom_count{0};
  std::size_t patch_evaluations{0};
  double milliseconds{0.0};
  bool warning{false};
};

inline constexpr std::size_t kNearBlockCount = 4;
inline constexpr std::size_t kNearPairTypeCount = 4;
inline constexpr std::size_t kNearOrientationCount = 3;
inline constexpr std::size_t kNearBlockAuditBucketCount =
    kNearBlockCount * kNearPairTypeCount * kNearOrientationCount;

struct NearBlockAuditBucket {
  std::size_t accepted{0};
  std::size_t rejected{0};
  std::size_t candidate_worse_than_centroid{0};
  std::size_t reference_nonconverged{0};
  std::size_t orientation_sign_mismatch{0};
  std::size_t strict_ref_reject{0};
  double max_candidate_ref_rel{0.0};
  double sum_candidate_ref_rel{0.0};
  double max_centroid_ref_rel{0.0};
  std::size_t k2_laplace_sign_ok{0};
  double max_k2_yukawa_smooth_ref_rel{0.0};
  double max_k2_orientation_pair_ref_rel{0.0};
  std::size_t sample_count{0};
};

using NearBlockAuditBuckets =
    std::array<NearBlockAuditBucket, kNearBlockAuditBucketCount>;

inline constexpr std::size_t kGeometricGapBucketCount = 4;
inline constexpr std::size_t kGeometricCenterBucketCount = 3;
inline constexpr std::size_t kGeometricKernelCount = 3;
inline constexpr std::size_t kGeometricNearAuditBucketCount =
    kGeometricGapBucketCount * kGeometricCenterBucketCount *
    kGeometricKernelCount;
inline constexpr std::size_t kJcpExpansionChannelCount = 4;

// A sampled, pre-GMRES measurement for the Cartesian full-FMM operator.
// This is deliberately distinct from the post-solve matvec audit: the two
// measurements use different probes and answer different questions.
struct FmmCartesianPreAuditTelemetry {
  std::string mode{"off"};
  bool enabled{false};
  bool finite{true};
  double aggregate_rel{0.0};
  double max_stratum_rel{0.0};
  std::size_t audited_targets{0};
  std::size_t audited_rows{0};
  int stages{0};
  bool expanded_to_256{false};
  bool expanded_to_512{false};
  double seconds{0.0};
  double channel_shadow_seconds{0.0};
  std::uint64_t probe_hash{0};
  std::uint64_t channel_probe_hash{0};
  std::uint64_t sampled_target_hash{0};
  std::uint64_t channel_direct_pairs_expected{0};
  std::uint64_t channel_direct_pairs_applied{0};
};

// Opt-in sampled direct comparison of the converged solution.  It is kept
// separate from both the fixed pre-GMRES probe and the legacy audit fields.
struct FmmCartesianSolutionAuditTelemetry {
  bool enabled{false};
  bool finite{true};
  bool relative_available{false};
  std::string status{"off"};
  double relative_error{0.0};
  double absolute_l2{0.0};
  double direct_l2{0.0};
  std::size_t targets{0};
  std::size_t rows{0};
  double seconds{0.0};
  std::uint64_t target_hash{0};
  std::uint64_t direct_pairs_expected{0};
  std::uint64_t direct_pairs_applied{0};
};

// The four JCP interaction blocks are reported independently by the opt-in
// Cartesian channel shadow.  ``available=false`` with ``absolute_only``
// means the direct norm was too small for a meaningful relative metric.
struct FmmCartesianChannelAudit {
  bool available{false};
  bool finite{true};
  std::string status{"off"};
  std::size_t samples{0};
  double abs_l2{0.0};
  double rms{0.0};
  double reference_l2{0.0};
  double relative_error{0.0};
};

// P2B2 records retry state separately from the established pre- and
// post-solve audit telemetry.  The runtime remains opt-in; these default
// values deliberately make a non-retry solve self-describing in profiles.
inline constexpr double kFmmCartesianRetrySolutionRelativeGate = 1.0e-4;
inline constexpr double kFmmCartesianRetryThetaTolerance = 1.0e-12;

enum class FmmCartesianRetryAttemptRole {
  initial_theta_020,
  tightened_theta_015,
};

enum class FmmCartesianRetryAttemptDisposition {
  accept,
  retry_trigger,
  fail_closed,
};

struct FmmCartesianRetryAttemptClassification {
  FmmCartesianRetryAttemptDisposition disposition{
      FmmCartesianRetryAttemptDisposition::fail_closed};
  std::string_view reason{"not_evaluated"};
};

struct FmmCartesianRetryIdentityClassification {
  bool valid{false};
  std::string_view reason{"not_evaluated"};
};

// The outer retry wrapper has exactly two possible solve attempts.  Keeping
// this state machine pure makes the no-third-attempt contract testable without
// manufacturing another expensive molecular solve.
enum class FmmCartesianRetryWrapperAction {
  accept_attempt1,
  run_attempt2,
  accept_attempt2,
  fail_closed,
};

struct FmmCartesianRetryWrapperDecision {
  FmmCartesianRetryWrapperAction action{
      FmmCartesianRetryWrapperAction::fail_closed};
  int accepted_attempt{0};
  bool sanitize_result{true};
  std::string_view reason{"not_evaluated"};
};

inline FmmCartesianRetryWrapperDecision
decide_fmm_cartesian_retry_after_attempt1(
    const FmmCartesianRetryAttemptClassification &first) noexcept {
  switch (first.disposition) {
  case FmmCartesianRetryAttemptDisposition::accept:
    return {FmmCartesianRetryWrapperAction::accept_attempt1, 1, false,
            first.reason};
  case FmmCartesianRetryAttemptDisposition::retry_trigger:
    return {FmmCartesianRetryWrapperAction::run_attempt2, 0, false,
            first.reason};
  case FmmCartesianRetryAttemptDisposition::fail_closed:
    return {FmmCartesianRetryWrapperAction::fail_closed, 0, true,
            first.reason};
  }
  return {FmmCartesianRetryWrapperAction::fail_closed, 0, true,
          "invalid_attempt1_disposition"};
}

inline FmmCartesianRetryWrapperDecision
decide_fmm_cartesian_retry_after_attempt2(
    const FmmCartesianRetryAttemptClassification &second,
    const FmmCartesianRetryIdentityClassification &identity) noexcept {
  if (second.disposition != FmmCartesianRetryAttemptDisposition::accept) {
    return {FmmCartesianRetryWrapperAction::fail_closed, 0, true,
            second.reason};
  }
  if (!identity.valid) {
    return {FmmCartesianRetryWrapperAction::fail_closed, 0, true,
            identity.reason};
  }
  return {FmmCartesianRetryWrapperAction::accept_attempt2, 2, false,
          "accepted_attempt2"};
}

// A compact, self-contained record of one complete solve.  P2B2C will fill
// these records around two independent solve_linear_pb attempts; P2B2B keeps
// them default-off so the existing paths remain numerically unchanged.
struct FmmCartesianRetryAttemptTelemetry {
  std::size_t attempt{0};
  bool started{false};
  bool completed{false};
  bool accepted{false};
  bool technically_valid{false};
  std::string classification{"off"};
  std::string classification_reason{"off"};
  std::string status{"off"};
  std::string convergence_reason{"off"};
  bool converged{false};
  bool finite{false};
  double theta{0.0};
  double salt_molar{0.0};
  bool coupled{false};
  std::string discretization{"off"};
  bool node_patch_operator_audit_enabled{false};
  double true_residual_rel{0.0};

  bool cartesian_full_active{false};
  bool cartesian_full_finite{false};
  std::string cartesian_full_requested{"off"};
  std::string cartesian_full_effective{"off"};
  int cartesian_full_order{0};
  int cartesian_full_leaf_size{0};
  std::size_t panel_count{0};
  std::uint64_t geometry_hash{0};
  std::uint64_t tree_hash{0};
  std::uint64_t worklist_hash{0};
  std::uint64_t policy_hash{0};
  std::uint64_t direct_pairs_expected{0};
  std::uint64_t direct_pairs_applied_last{0};
  std::uint64_t direct_pairs_applied_total{0};
  std::uint64_t pre_audit_sampled_target_hash{0};
  FmmCartesianSolutionAuditTelemetry solution_audit{};

  bool fallback_used{false};
  std::string preconditioner_requested{"off"};
  std::string preconditioner_effective{"off"};
  std::string preconditioner_quality_status{"off"};
  std::string preconditioner_coarse_status{"off"};
  std::string preconditioner_ilu_regularization{"off"};
  double preconditioner_ilu_regularization_selected{0.0};
  std::size_t preconditioner_ilu_regularization_shifted_rows{0};
  std::string preconditioner_ilu_unshifted_failure_stage{"off"};
  std::size_t preconditioner_valid_panels{0};
  std::size_t preconditioner_ill_conditioned_blocks{0};
  std::size_t preconditioner_quality_rejected_blocks{0};
  double preconditioner_setup_ms{0.0};
  double preconditioner_ilu_factor_setup_ms{0.0};

  double surface_build_ms{0.0};
  double fmm_prepare_ms{0.0};
  double gmres_ms{0.0};
  double wall_ms{0.0};
  std::size_t accounted_memory_bytes{0};
  std::size_t rss_before_bytes{0};
  std::size_t rss_after_bytes{0};
};

struct FmmCartesianRetryTelemetry {
  std::string mode{"off"};
  bool enabled{false};
  std::size_t attempts_started{0};
  std::size_t attempts_completed{0};
  int accepted_attempt{0};
  std::string trigger_reason{"off"};
  std::string final_reason{"off"};
  bool geometry_identity_valid{false};
  bool tree_identity_valid{false};
  bool policy_transition_valid{false};
  double cumulative_wall_ms{0.0};
  double cumulative_gmres_ms{0.0};
  std::size_t cumulative_accounted_memory_bytes{0};
  std::size_t max_boundary_rss_bytes{0};
  std::array<FmmCartesianRetryAttemptTelemetry, 2> attempts{};
};

inline bool fmm_cartesian_retry_theta_matches(double actual,
                                               double expected) noexcept {
  if (!std::isfinite(actual) || !std::isfinite(expected))
    return false;
  const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
  return std::abs(actual - expected) <=
         kFmmCartesianRetryThetaTolerance * scale;
}

inline bool fmm_cartesian_retry_plain_ilu0_valid(
    const FmmCartesianRetryAttemptTelemetry &attempt) noexcept {
  const bool effective_plain =
      attempt.preconditioner_effective == "fmm_nearfield_ilu0";
  return attempt.preconditioner_requested == "fmm-nearfield-ilu0" &&
         effective_plain && attempt.preconditioner_coarse_status == "off" &&
         std::isfinite(attempt.preconditioner_ilu_regularization_selected) &&
         attempt.preconditioner_ilu_regularization_selected == 0.0 &&
         attempt.preconditioner_ilu_regularization_shifted_rows == 0 &&
         attempt.preconditioner_ilu_unshifted_failure_stage == "off" &&
         attempt.preconditioner_quality_status == "pass" &&
         attempt.preconditioner_valid_panels == attempt.panel_count &&
         attempt.preconditioner_ill_conditioned_blocks == 0 &&
         attempt.preconditioner_quality_rejected_blocks == 0;
}

inline FmmCartesianRetryAttemptClassification
classify_fmm_cartesian_retry_attempt(
    const FmmCartesianRetryAttemptTelemetry &attempt,
    FmmCartesianRetryAttemptRole role) noexcept {
  const double expected_theta =
      role == FmmCartesianRetryAttemptRole::initial_theta_020 ? 0.20 : 0.15;
  const auto fail = [](std::string_view reason) {
    return FmmCartesianRetryAttemptClassification{
        FmmCartesianRetryAttemptDisposition::fail_closed, reason};
  };
  if (!attempt.started || !attempt.completed)
    return fail("attempt_incomplete");
  if (attempt.status != "ok")
    return fail("solver_status_not_ok");
  if (!attempt.converged)
    return fail("not_converged");
  if (!attempt.finite)
    return fail("solver_nonfinite");
  if (!std::isfinite(attempt.true_residual_rel) ||
      attempt.true_residual_rel < 0.0 ||
      attempt.true_residual_rel > kFmmCartesianRetrySolutionRelativeGate)
    return fail("true_residual_gate_failed");
  if (!attempt.coupled || attempt.discretization != "node_patch" ||
      !attempt.node_patch_operator_audit_enabled ||
      !std::isfinite(attempt.salt_molar) || attempt.salt_molar <= 0.0)
    return fail("scientific_scope_mismatch");
  if (!attempt.cartesian_full_active || !attempt.cartesian_full_finite ||
      attempt.cartesian_full_requested != "cartesian-full-experimental" ||
      attempt.cartesian_full_effective != "cartesian-full-experimental" ||
      attempt.cartesian_full_order != 1 || attempt.cartesian_full_leaf_size != 64)
    return fail("cartesian_full_fmm_scope_mismatch");
  if (!fmm_cartesian_retry_theta_matches(attempt.theta, expected_theta))
    return fail("theta_mismatch");
  if (attempt.panel_count == 0 || attempt.geometry_hash == 0 ||
      attempt.tree_hash == 0 || attempt.worklist_hash == 0 ||
      attempt.policy_hash == 0)
    return fail("cartesian_full_fmm_hash_missing");
  if (attempt.direct_pairs_expected == 0 ||
      attempt.direct_pairs_applied_last != attempt.direct_pairs_expected ||
      attempt.direct_pairs_applied_total < attempt.direct_pairs_expected ||
      attempt.direct_pairs_applied_total % attempt.direct_pairs_expected != 0)
    return fail("cartesian_full_fmm_direct_coverage_mismatch");
  const FmmCartesianSolutionAuditTelemetry &audit = attempt.solution_audit;
  const bool audit_rows_valid =
      audit.targets <= std::numeric_limits<std::size_t>::max() / 2u &&
      audit.rows == 2u * audit.targets;
  if (!audit.enabled || !audit.finite || !audit.relative_available ||
      audit.status != "relative" || !std::isfinite(audit.relative_error) ||
      audit.relative_error < 0.0 || !std::isfinite(audit.absolute_l2) ||
      audit.absolute_l2 < 0.0 || !std::isfinite(audit.direct_l2) ||
      audit.direct_l2 <= 0.0 || !std::isfinite(audit.seconds) ||
      audit.seconds < 0.0 ||
      audit.target_hash == 0 || audit.targets == 0 || audit.rows == 0 ||
      !audit_rows_valid ||
      audit.target_hash != attempt.pre_audit_sampled_target_hash ||
      audit.direct_pairs_expected == 0 ||
      audit.direct_pairs_applied != audit.direct_pairs_expected)
    return fail("solution_audit_unavailable_or_invalid");
  if (attempt.fallback_used)
    return fail("fallback_used");
  if (!fmm_cartesian_retry_plain_ilu0_valid(attempt))
    return fail("plain_fmm_nearfield_ilu0_required");
  if (audit.relative_error <= kFmmCartesianRetrySolutionRelativeGate) {
    return {FmmCartesianRetryAttemptDisposition::accept,
            "solution_audit_within_gate"};
  }
  if (role == FmmCartesianRetryAttemptRole::initial_theta_020) {
    return {FmmCartesianRetryAttemptDisposition::retry_trigger,
            "solution_audit_above_gate"};
  }
  return fail("solution_audit_above_gate_after_tighten");
}

inline FmmCartesianRetryIdentityClassification
classify_fmm_cartesian_retry_identity(
    const FmmCartesianRetryAttemptTelemetry &first,
    const FmmCartesianRetryAttemptTelemetry &second) noexcept {
  const auto fail = [](std::string_view reason) {
    return FmmCartesianRetryIdentityClassification{false, reason};
  };
  if (!first.started || !first.completed || !second.started ||
      !second.completed)
    return fail("attempt_incomplete");
  if (first.panel_count == 0 || first.panel_count != second.panel_count)
    return fail("panel_count_mismatch");
  if (first.geometry_hash == 0 || first.geometry_hash != second.geometry_hash)
    return fail("geometry_hash_mismatch");
  if (first.tree_hash == 0 || first.tree_hash != second.tree_hash)
    return fail("tree_hash_mismatch");
  if (first.cartesian_full_order != 1 || second.cartesian_full_order != 1 ||
      first.cartesian_full_leaf_size != 64 ||
      second.cartesian_full_leaf_size != 64)
    return fail("cartesian_full_fmm_order_or_leaf_mismatch");
  if (!fmm_cartesian_retry_theta_matches(first.theta, 0.20) ||
      !fmm_cartesian_retry_theta_matches(second.theta, 0.15))
    return fail("retry_theta_mismatch");
  if (first.worklist_hash == 0 || second.worklist_hash == 0)
    return fail("worklist_hash_missing");
  if (first.policy_hash == 0 || second.policy_hash == 0 ||
      first.policy_hash == second.policy_hash)
    return fail("policy_hash_transition_invalid");
  return {true, "identity_match"};
}

struct GeometricNearAuditBucket {
  std::size_t pairs{0};
  double min_gap{0.0};
  double max_gap_ratio{0.0};
  double sum_gap_ratio{0.0};
  double sum_center_ratio{0.0};
};

using GeometricNearAuditBuckets =
    std::array<GeometricNearAuditBucket, kGeometricNearAuditBucketCount>;

struct FrameResult {
  std::size_t trajectory_id{0};
  std::size_t frame_index{0};
  std::size_t global_index{0};
  double time_ps{0.0};
  std::size_t atom_count{0};
  double total_charge{0.0};
  std::size_t nx{0};
  std::size_t ny{0};
  std::size_t nz{0};
  std::size_t grid_points{0};
  std::size_t memory_bytes{0};
  std::size_t memory_estimated_surface_bytes{0};
  std::size_t memory_estimated_solver_bytes{0};
  std::size_t memory_estimated_operator_bytes{0};
  std::size_t memory_estimated_tree_bytes{0};
  std::size_t memory_rss_before_bytes{0};
  std::size_t memory_rss_after_bytes{0};
  std::size_t memory_rss_delta_bytes{0};
  std::int64_t memory_rss_delta_signed_bytes{0};
  std::size_t memory_peak_working_set_bytes{0};
  double grid_origin_x_A{0.0};
  double grid_origin_y_A{0.0};
  double grid_origin_z_A{0.0};
  double grid_center_x_A{0.0};
  double grid_center_y_A{0.0};
  double grid_center_z_A{0.0};
  double grid_spacing_A{0.0};
  std::size_t solute_points{0};
  std::size_t solvent_points{0};
  bool focus_enabled{false};
  int focus_levels{1};
  double focus_initial_scale{0.0};
  double focus_initial_margin{0.0};
  std::string solver_mode{"sor"};
  int mg_levels{1};
  int mg_cycles{0};
  int mg_pre{0};
  int mg_post{0};
  int mg_coarse_iter{0};
  int solvent_iterations{0};
  int vacuum_iterations{0};
  int solvent_iterations_final{0};
  int vacuum_iterations_final{0};
  double solvent_rms{0.0};
  double vacuum_rms{0.0};
  double solvent_max_dphi{0.0};
  double vacuum_max_dphi{0.0};
  bool solvent_converged{false};
  bool vacuum_converged{false};
  std::string surface_effective{"vdw"};
  std::string ion_access{"dielectric"};
  double ion_radius_A{2.0};
  std::size_t ion_accessible_points{0};
  std::size_t stern_excluded_points{0};
  bool face_refine{false};
  std::string pbe_backend{"grid"};
  double tabi_surface_build_ms{0.0};
  std::size_t tabi_surface_panels{0};
  int tabi_surface_point_count_requested{0};
  double tabi_panel_density_per_a2{0.0};
  std::string tabi_panel_budget_status{"off"};
  ChargeResolutionDiagnostic tabi_charge_resolution;
  std::size_t tabi_surface_exposed_points{0};
  std::size_t tabi_surface_buried_points{0};
  double tabi_tree_build_ms{0.0};
  std::size_t tabi_tree_nodes{0};
  std::size_t tabi_tree_leaf_count{0};
  std::size_t tabi_near_interactions{0};
  std::size_t tabi_far_interactions{0};
  std::string tabi_matvec_mode{"off"};
  std::string tabi_fmm_m2l_mode{"off"};
  int tabi_fmm_m2l_expansion_order{0};
  std::size_t tabi_fmm_m2l_near_pairs{0};
  std::size_t tabi_fmm_m2l_far_pairs{0};
  double tabi_fmm_m2l_setup_ms{0.0};
  double tabi_fmm_m2l_apply_ms{0.0};
  bool tabi_fmm_m2l_finite_outputs{true};
  bool tabi_bie_cartesian_full_fmm_active{false};
  std::string tabi_bie_cartesian_full_fmm_requested{"off"};
  std::string tabi_bie_cartesian_full_fmm_effective{"off"};
  int tabi_bie_cartesian_full_fmm_order{0};
  int tabi_bie_cartesian_full_fmm_leaf_size{0};
  double tabi_bie_cartesian_full_fmm_opening_theta{0.0};
  std::uint64_t tabi_bie_cartesian_full_fmm_tree_hash{0};
  std::uint64_t tabi_bie_cartesian_full_fmm_worklist_hash{0};
  std::uint64_t tabi_bie_cartesian_full_fmm_policy_hash{0};
  std::size_t tabi_bie_cartesian_full_fmm_nodes{0};
  std::size_t tabi_bie_cartesian_full_fmm_far_pairs{0};
  std::size_t tabi_bie_cartesian_full_fmm_direct_pairs{0};
  std::uint64_t tabi_bie_cartesian_full_fmm_direct_panel_pairs_expected{0};
  std::uint64_t tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_last{0};
  std::uint64_t tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_total{0};
  std::size_t tabi_bie_cartesian_full_fmm_cache_entries{0};
  std::size_t tabi_bie_cartesian_full_fmm_cache_hits{0};
  std::size_t tabi_bie_cartesian_full_fmm_cache_bytes{0};
  std::size_t tabi_bie_cartesian_full_fmm_plan_bytes{0};
  std::size_t tabi_bie_cartesian_full_fmm_workspace_bytes{0};
  double tabi_bie_cartesian_full_fmm_tree_seconds{0.0};
  double tabi_bie_cartesian_full_fmm_worklist_seconds{0.0};
  double tabi_bie_cartesian_full_fmm_cache_seconds{0.0};
  double tabi_bie_cartesian_full_fmm_audit_seconds{0.0};
  double tabi_bie_cartesian_full_fmm_p2m_seconds{0.0};
  double tabi_bie_cartesian_full_fmm_m2m_seconds{0.0};
  double tabi_bie_cartesian_full_fmm_m2l_seconds{0.0};
  double tabi_bie_cartesian_full_fmm_l2l_seconds{0.0};
  double tabi_bie_cartesian_full_fmm_l2p_seconds{0.0};
  double tabi_bie_cartesian_full_fmm_direct_seconds{0.0};
  double tabi_bie_cartesian_full_fmm_m2m_parity_error{0.0};
  double tabi_bie_cartesian_full_fmm_l2l_parity_error{0.0};
  bool tabi_fmm_local_gmres_active{false};
  double tabi_fmm_local_prepare_seconds{0.0};
  std::size_t tabi_fmm_local_plan_build_count{0};
  std::size_t tabi_fmm_local_cache_build_count{0};
  std::size_t tabi_fmm_local_gmres_matvec_calls{0};
  double tabi_fmm_local_gmres_apply_seconds_total{0.0};
  double tabi_fmm_local_gmres_apply_seconds_mean{0.0};
  double tabi_fmm_local_moment_seconds_total{0.0};
  double tabi_fmm_local_m2l_seconds_total{0.0};
  double tabi_fmm_local_fallback_seconds_total{0.0};
  double tabi_fmm_local_l2p_seconds_total{0.0};
  double tabi_fmm_local_sampled_direct_true_residual{0.0};
  bool tabi_fmm_local_operator_finite{true};
  std::string tabi_fmm_local_refine_probe_mode{"rademacher"};
  std::size_t tabi_fmm_local_refine_probe_count{4};
  bool tabi_fmm_local_solution_probe_used{false};
  double tabi_fmm_local_linf_max_abs_tail{0.0};
  double tabi_fmm_local_linf_max_relative_tail{0.0};
  std::size_t tabi_fmm_local_linf_outlier_panels{0};
  std::string tabi_fmm_local_geometry_guard{"off"};
  double tabi_fmm_local_geometry_theta{0.0};
  std::size_t tabi_fmm_local_geometry_guard_rejected_cells{0};
  std::size_t tabi_fmm_local_geometry_guard_direct_fallback_pairs{0};
  double tabi_fmm_local_geometry_gap_min{0.0};
  double tabi_fmm_local_geometry_ratio_max{0.0};
  std::string tabi_status{"off"};
  std::string tabi_convergence_reason{"off"};
  std::string tabi_residual_history{};
  int tabi_gmres_restart_requested{32};
  int tabi_gmres_restart_effective{32};
  double tabi_best_residual{0.0};
  int tabi_best_residual_iter{0};
  double tabi_final_residual{0.0};
  double tabi_best_preconditioned_residual{0.0};
  double tabi_final_preconditioned_residual{0.0};
  double tabi_true_residual_abs{0.0};
  double tabi_true_residual_rel{0.0};
  std::string tabi_true_residual_probe_mode{"off"};
  std::string tabi_true_residual_probe_scope{"full"};
  int tabi_true_residual_probe_interval{0};
  int tabi_true_residual_sample_count{0};
  std::size_t tabi_true_residual_probe_calls{0};
  double tabi_true_residual_probe_ms{0.0};
  double tabi_sampled_true_residual_rel{0.0};
  std::size_t tabi_sampled_true_residual_rows{0};
  std::size_t tabi_sampled_probe_calls{0};
  double tabi_sampled_probe_ms{0.0};
  std::size_t tabi_full_true_residual_confirm_calls{0};
  double tabi_full_probe_ms{0.0};
  std::string tabi_reaction_energy_source{"off"};
  double tabi_rhs_ms{0.0};
  double tabi_gmres_ms{0.0};
  double tabi_matvec_ms_total{0.0};
  double tabi_matvec_ms_avg{0.0};
  std::size_t tabi_matvec_calls{0};
  double tabi_tree_vs_direct_matvec_rel{0.0};
  std::size_t tabi_matvec_audit_rows{0};
  double tabi_matvec_audit_ms{0.0};
  FmmCartesianPreAuditTelemetry tabi_fmm_cartesian_pre_audit{};
  FmmCartesianSolutionAuditTelemetry tabi_fmm_cartesian_solution_audit{};
  std::array<FmmCartesianChannelAudit, kJcpExpansionChannelCount>
      tabi_fmm_cartesian_channel_audit{};
  FmmCartesianRetryTelemetry tabi_fmm_cartesian_retry{};
  double tabi_orthogonalization_ms{0.0};
  double tabi_preconditioner_ms{0.0};
  double tabi_surface_neighbor_ms{0.0};
  double tabi_surface_mesh_ms{0.0};
  double tabi_solid_angle_ms{0.0};
  double tabi_tree_moment_ms{0.0};
  double tabi_tree_traversal_ms{0.0};
  std::string tabi_matvec_cost_profile{"off"};
  double tabi_leaf_panels_per_second{0.0};
  double tabi_clean_leaf_panels_per_second{0.0};
  double tabi_tree_traversal_fraction{0.0};
  double tabi_near_eval_ms{0.0};
  double tabi_far_eval_ms{0.0};
  double tabi_singular_near_eval_ms{0.0};
  double tabi_leaf_near_eval_ms{0.0};
  double tabi_leaf_panel_eval_ms{0.0};
  double tabi_leaf_tile_eval_ms{0.0};
  double tabi_leaf_residual_eval_ms{0.0};
  double tabi_smooth_leaf_node_eval_ms{0.0};
  std::size_t tabi_singular_near_interactions{0};
  std::size_t tabi_leaf_near_interactions{0};
  std::size_t tabi_far_node_interactions{0};
  std::size_t tabi_leaf_panel_interactions{0};
  std::size_t tabi_leaf_tile_interactions{0};
  std::size_t tabi_leaf_tile_panel_interactions{0};
  std::size_t tabi_leaf_residual_panel_interactions{0};
  std::size_t tabi_smooth_leaf_node_interactions{0};
  std::size_t tabi_leaf_pruned_interactions{0};
  std::size_t tabi_leaf_mac_candidates{0};
  std::size_t tabi_leaf_mac_accepted{0};
  std::size_t tabi_leaf_mac_rejected{0};
  std::size_t tabi_leaf_mac_far_panels{0};
  std::size_t tabi_leaf_mac_direct_panels_saved_estimate{0};
  std::string tabi_leaf_work_profile{"off"};
  std::size_t tabi_leaf_work_mac_terminal_panels{0};
  std::size_t tabi_leaf_work_physical_gap_panels{0};
  std::size_t tabi_leaf_work_fragmentation_panels{0};
  std::size_t tabi_leaf_work_convergence_panels{0};
  std::size_t tabi_leaf_work_convergence_estimator_panels{0};
  std::size_t tabi_leaf_work_convergence_rho_panels{0};
  std::size_t tabi_leaf_work_convergence_aspect_panels{0};
  std::size_t tabi_leaf_work_surface_margin_panels{0};
  std::size_t tabi_leaf_work_surface_margin_hard_panels{0};
  std::size_t tabi_leaf_work_surface_margin_soft_panels{0};
  std::size_t tabi_leaf_work_k3_rho_cap_panels{0};
  std::size_t tabi_leaf_work_singular_fallback_panels{0};
  std::size_t tabi_leaf_work_scalar_tail_panels{0};
  std::size_t tabi_leaf_work_profile_accounted_panels{0};
  double tabi_leaf_work_mac_terminal_frac{0.0};
  double tabi_leaf_work_convergence_frac{0.0};
  double tabi_leaf_work_surface_margin_frac{0.0};
  double tabi_leaf_work_singular_fallback_frac{0.0};
  double tabi_leaf_work_scalar_tail_frac{0.0};
  std::size_t tabi_adaptive_smooth_leaf_accepted{0};
  std::size_t tabi_adaptive_smooth_leaf_rejected{0};
  std::size_t tabi_adaptive_span_tile_accepted{0};
  std::size_t tabi_adaptive_span_tile_rejected{0};
  std::size_t tabi_adaptive_exact_leaf_fallbacks{0};
  std::size_t tabi_adaptive_reject_one_ring_overlap{0};
  std::size_t tabi_adaptive_reject_error_bound_failed{0};
  std::size_t tabi_adaptive_reject_p4_check_failed{0};
  std::size_t tabi_adaptive_reject_too_close{0};
  double tabi_leaf_work_reduction_ratio{0.0};
  std::size_t tabi_target_cluster_interactions{0};
  std::size_t tabi_target_cluster_targets{0};
  std::size_t tabi_target_cluster_panels_pruned{0};
  std::size_t tabi_target_cluster_rejected{0};
  double tabi_target_cluster_eval_ms{0.0};
  std::size_t tabi_leaf_pair_batch_interactions{0};
  std::size_t tabi_leaf_pair_batch_targets{0};
  std::size_t tabi_leaf_pair_batch_source_panels{0};
  std::size_t tabi_leaf_pair_batch_residual_panels{0};
  std::size_t tabi_leaf_pair_batch_fallbacks{0};
  double tabi_leaf_pair_batch_ms{0.0};
  std::string tabi_leaf_pair_batch_kernel{"off"};
  std::size_t tabi_leaf_pair_batch_vectorized_sources{0};
  std::size_t tabi_leaf_pair_batch_vector_lanes{0};
  double tabi_leaf_pair_batch_pairs_per_second{0.0};
  std::string tabi_leaf_kernel_profile{"off"};
  std::string tabi_jcp_direct_kernel{"baseline"};
  std::size_t tabi_leaf_kernel_scalar_tasks{0};
  std::size_t tabi_leaf_kernel_batch_tasks{0};
  std::size_t tabi_leaf_kernel_source_panel_evals{0};
  double tabi_leaf_kernel_scalar_ms{0.0};
  double tabi_leaf_kernel_target_avx2_ms{0.0};
  double tabi_leaf_kernel_tail_ms{0.0};
  std::size_t tabi_leaf_pair_batch_soa_sources{0};
  double tabi_leaf_pair_batch_soa_ms{0.0};
  double tabi_leaf_pair_batch_scalar_tail_ms{0.0};
  double tabi_leaf_pair_batch_clean_source_panels_per_second{0.0};
  std::string tabi_leaf_pair_selection_profile{"off"};
  std::string tabi_leaf_pair_probe_cache{"off"};
  std::string tabi_leaf_pair_target_group{"scan"};
  int tabi_leaf_pair_max_lookahead{32};
  std::size_t tabi_leaf_pair_selection_candidate_scans{0};
  std::size_t tabi_leaf_pair_selection_clean_rejects{0};
  std::size_t tabi_leaf_pair_selection_probe_calls{0};
  std::size_t tabi_leaf_pair_selection_probe_node_visits{0};
  double tabi_leaf_pair_selection_probe_ms{0.0};
  double tabi_leaf_pair_selection_ms{0.0};
  std::size_t tabi_leaf_pair_selection_probe_cache_hits{0};
  std::size_t tabi_leaf_pair_selection_probe_cache_misses{0};
  std::size_t tabi_leaf_pair_selection_insufficient_targets{0};
  std::size_t tabi_leaf_pair_selection_same_target_leaf_rejects{0};
  std::string tabi_leaf_pair_directness_probe{"full-traversal"};
  std::string tabi_leaf_pair_directness_cache{"off"};
  std::string tabi_leaf_pair_directness_verify{"off"};
  std::size_t tabi_leaf_pair_directness_ancestor_probe_calls{0};
  std::size_t tabi_leaf_pair_directness_ancestor_probe_node_visits{0};
  double tabi_leaf_pair_directness_ancestor_probe_ms{0.0};
  std::size_t tabi_leaf_pair_directness_pair_cache_hits{0};
  std::size_t tabi_leaf_pair_directness_pair_cache_misses{0};
  std::size_t tabi_leaf_pair_directness_all_direct_proofs{0};
  std::size_t tabi_leaf_pair_directness_mixed_fallbacks{0};
  std::size_t tabi_leaf_pair_directness_verify_checks{0};
  std::size_t tabi_leaf_pair_directness_verify_mismatches{0};
  std::string tabi_leaf_pair_scheduler{"inline-select"};
  int tabi_leaf_pair_worklist_max_active{2048};
  std::string tabi_leaf_pair_worklist_verify{"off"};
  std::size_t tabi_leaf_pair_worklist_tasks{0};
  std::size_t tabi_leaf_pair_worklist_groups{0};
  std::size_t tabi_leaf_pair_worklist_batched_groups{0};
  std::size_t tabi_leaf_pair_worklist_scalar_flush_tasks{0};
  std::size_t tabi_leaf_pair_worklist_forced_flushes{0};
  std::size_t tabi_leaf_pair_worklist_active_groups_max{0};
  double tabi_leaf_pair_worklist_ms{0.0};
  std::size_t tabi_leaf_pair_worklist_verify_checks{0};
  std::size_t tabi_leaf_pair_worklist_verify_mismatches{0};
  std::string tabi_leaf_block_verify{"off"};
  std::size_t tabi_leaf_block_target_leaves{0};
  std::size_t tabi_leaf_block_targets{0};
  std::size_t tabi_leaf_block_source_groups{0};
  std::size_t tabi_leaf_block_batched_groups{0};
  std::size_t tabi_leaf_block_scalar_tail_tasks{0};
  double tabi_leaf_block_group_ms{0.0};
  double tabi_leaf_block_flush_ms{0.0};
  double tabi_leaf_block_writeback_ms{0.0};
  std::size_t tabi_leaf_block_verify_checks{0};
  std::size_t tabi_leaf_block_verify_mismatches{0};
  std::string tabi_tree_preset{"off"};
  std::string tabi_adaptive_level{"off"};
  std::string tabi_cluster_policy{"off"};
  int tabi_target_batch_size{1};
  int tabi_leaf_pair_min_source_panels{64};
  int tabi_leaf_pair_min_targets{4};
  std::string tabi_leaf_near_policy{"off"};
  std::string tabi_leaf_panel_kernel{"off"};
  std::string tabi_interaction_mode{"off"};
  std::string tabi_node_bounds_mode{"off"};
  std::string tabi_tree_builder{"off"};
  std::string tabi_expansion_center{"off"};
  std::string tabi_fragmentation_safety{"off"};
  std::string tabi_mac_safety{"off"};
  double tabi_mac_gap_factor{1.5};
  std::size_t tabi_tree_depth_max{0};
  std::size_t tabi_mac_reject_radius_limited{0};
  double tabi_mac_accept_rate{0.0};
  double tabi_node_radius_max{0.0};
  double tabi_node_radius_mean{0.0};
  double tabi_node_aspect_ratio_max{0.0};
  double tabi_node_aspect_ratio_mean{0.0};
  double tabi_leaf_panel_work_per_matvec{0.0};
  double tabi_far_node_work_per_matvec{0.0};
  std::size_t tabi_far_geometric_unsafe_count{0};
  std::size_t tabi_far_geometric_unsafe_panels{0};
  double tabi_far_geometric_min_gap{0.0};
  double tabi_far_geometric_gap_ratio_min{0.0};
  double tabi_far_geometric_gap_ratio_mean{0.0};
  std::size_t tabi_far_geometric_audit_samples{0};
  std::size_t tabi_mac_physical_gap_rejects{0};
  std::size_t tabi_mac_convergence_rho_rejects{0};
  std::size_t tabi_mac_convergence_aspect_rejects{0};
  std::size_t tabi_mac_convergence_rho_reject_panels{0};
  std::size_t tabi_mac_convergence_aspect_reject_panels{0};
  std::size_t tabi_fragmented_far_count{0};
  std::size_t tabi_fragmented_far_panels{0};
  std::size_t tabi_fragmented_split_rejects{0};
  double tabi_fragmentation_center_gap_ratio_max{0.0};
  double tabi_fragmentation_center_gap_ratio_mean{0.0};
  std::string tabi_moment_audit{"off"};
  std::string tabi_moment_construction{"off"};
  std::string tabi_moment_sum{"off"};
  std::string tabi_convergence_audit{"off"};
  std::string tabi_convergence_safety{"off"};
  double tabi_convergence_rho_cap{0.0};
  double tabi_mac_aspect_cap{0.0};
  std::string tabi_convergence_estimator{"off"};
  double tabi_convergence_tail_threshold{1.0e4};
  double tabi_convergence_estimator_error_cap{1.0e-3};
  double tabi_convergence_unstable_rho_cap{0.25};
  std::string tabi_convergence_calibration{"stage3j"};
  std::size_t tabi_convergence_estimator_samples{0};
  std::size_t tabi_convergence_estimator_unstable_far_count{0};
  std::size_t tabi_convergence_estimator_unstable_far_panels{0};
  std::size_t tabi_convergence_estimator_split_rejects{0};
  std::size_t tabi_convergence_estimator_split_reject_panels{0};
  std::size_t tabi_convergence_low_abs_reaccepted_far_count{0};
  std::size_t tabi_convergence_low_abs_reaccepted_far_panels{0};
  double tabi_convergence_estimator_tail_ratio_max{0.0};
  double tabi_convergence_estimator_tail_ratio_mean{0.0};
  double tabi_convergence_estimator_error_max{0.0};
  double tabi_convergence_estimator_error_mean{0.0};
  double tabi_convergence_estimator_abs_error_max{0.0};
  double tabi_convergence_estimator_abs_error_mean{0.0};
  double tabi_convergence_tail_norm_p_max{0.0};
  double tabi_convergence_tail_norm_p_mean{0.0};
  double tabi_convergence_tail_norm_pm1_max{0.0};
  double tabi_convergence_tail_norm_pm1_mean{0.0};
  double tabi_convergence_effective_jcp_amp_max{0.0};
  double tabi_convergence_effective_jcp_amp_mean{0.0};
  std::string tabi_convergence_dielectric_amp{"off"};
  std::string tabi_convergence_estimator_model{"stage3l"};
  std::string tabi_convergence_channel_calibration{"stage3m"};
  std::string tabi_convergence_split_policy{"global"};
  std::string tabi_convergence_split_selectivity{"legacy"};
  double tabi_convergence_split_absolute_floor{0.0};
  std::string tabi_convergence_overreject_audit{"off"};
  std::string tabi_convergence_overreject_detail{"off"};
  std::string tabi_convergence_shape_audit{"off"};
  std::string tabi_convergence_kernel_response_audit{"off"};
  std::string tabi_channel_missed_semantics{"legacy"};
  std::string tabi_accepted_miss_diagnostics{"off"};
  double tabi_accepted_miss_absolute_floor{1.0e-7};
  double tabi_convergence_k3_accepted_rho_cap{0.0};
  std::string tabi_mac_distance_policy{"center"};
  double tabi_mac_surface_margin_factor{0.25};
  std::string tabi_mac_surface_margin_policy{"all"};
  double tabi_mac_surface_margin_aspect_threshold{3.0};
  double tabi_mac_surface_margin_rho_trigger{0.55};
  double tabi_mac_surface_margin_soft_aspect_threshold{0.0};
  double tabi_mac_surface_margin_soft_rho_trigger{0.0};
  double tabi_mac_surface_margin_soft_factor{0.10};
  double tabi_convergence_dielectric_amp_max{0.0};
  double tabi_convergence_dielectric_amp_mean{0.0};
  std::array<double, kJcpExpansionChannelCount>
      tabi_convergence_channel_estimated_error_max{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_convergence_channel_estimated_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_convergence_channel_effective_jcp_amp_max{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_convergence_channel_effective_jcp_amp_mean{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_unstable_far_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_unstable_far_panels{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_split_rejects{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_split_reject_panels{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_low_abs_reaccepted_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_low_abs_reaccepted_panels{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_high_error_caught{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_high_error_missed{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_high_error_caught_by_split{};
  std::size_t tabi_convergence_estimator_high_error_caught{0};
  std::size_t tabi_convergence_estimator_high_error_missed{0};
  std::size_t tabi_convergence_estimator_high_error_caught_by_split{0};
  std::size_t tabi_convergence_estimator_phantom_high_error_count{0};
  double tabi_convergence_estimator_phantom_high_error_abs_max{0.0};
  std::size_t tabi_convergence_overreject_count{0};
  std::size_t tabi_convergence_overreject_panels{0};
  std::size_t tabi_convergence_overreject_caught_count{0};
  std::size_t tabi_convergence_overreject_caught_panels{0};
  double tabi_convergence_overreject_abs_error_max{0.0};
  double tabi_convergence_overreject_abs_error_mean{0.0};
  double tabi_convergence_overreject_abs_error_p95{0.0};
  double tabi_convergence_overreject_abs_error_p99{0.0};
  double tabi_convergence_overreject_estimated_abs_error_max{0.0};
  double tabi_convergence_overreject_estimated_abs_error_mean{0.0};
  double tabi_convergence_overreject_required_scale_max{0.0};
  double tabi_convergence_overreject_required_scale_p95{0.0};
  double tabi_convergence_overreject_required_scale_p99{0.0};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_overreject_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_overreject_caught_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_convergence_channel_overreject_panels{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_convergence_channel_overreject_abs_error_max{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_convergence_channel_overreject_abs_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_convergence_channel_overreject_required_scale_max{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_convergence_channel_overreject_required_scale_p99{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_phantom_high_error_channel_count{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_phantom_high_error_channel_abs_max{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_accepted_miss_channel_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      tabi_accepted_miss_channel_panels{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_accepted_miss_channel_rho_min{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_accepted_miss_channel_rho_mean{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_accepted_miss_channel_rho_max{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_accepted_miss_channel_tail_ratio_mean{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_accepted_miss_channel_tail_ratio_max{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_accepted_miss_channel_estimated_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_accepted_miss_channel_estimated_error_max{};
  std::string tabi_accepted_miss_worst_channel{"none"};
  std::size_t tabi_convergence_k3_rho_cap_rejects{0};
  std::size_t tabi_convergence_k3_rho_cap_reject_panels{0};
  std::size_t tabi_mac_surface_margin_rejects{0};
  std::size_t tabi_mac_surface_margin_reject_panels{0};
  std::size_t tabi_mac_surface_margin_applied{0};
  std::size_t tabi_mac_surface_margin_skipped{0};
  std::size_t tabi_mac_surface_margin_adaptive_rejects{0};
  std::size_t tabi_mac_surface_margin_adaptive_reject_panels{0};
  std::size_t tabi_mac_surface_margin_skipped_accepted_far{0};
  std::size_t tabi_mac_surface_margin_hard_applied{0};
  std::size_t tabi_mac_surface_margin_soft_applied{0};
  std::size_t tabi_mac_surface_margin_hard_rejects{0};
  std::size_t tabi_mac_surface_margin_hard_reject_panels{0};
  std::size_t tabi_mac_surface_margin_soft_rejects{0};
  std::size_t tabi_mac_surface_margin_soft_reject_panels{0};
  std::size_t tabi_moment_audit_nodes{0};
  double tabi_moment_translation_rel_error_max{0.0};
  double tabi_moment_translation_rel_error_mean{0.0};
  double tabi_moment_translation_offset_max{0.0};
  double tabi_moment_translation_offset_mean{0.0};
  double tabi_moment_cancellation_ratio_max{0.0};
  double tabi_moment_cancellation_ratio_mean{0.0};
  double tabi_moment_dynamic_range_max{0.0};
  std::string tabi_expansion_audit{"off"};
  std::string tabi_expansion_audit_channel{"off"};
  std::size_t tabi_far_expansion_audit_pairs{0};
  std::size_t tabi_far_expansion_audit_panels{0};
  double tabi_far_expansion_direct_ms{0.0};
  double tabi_far_expansion_rel_error_max{0.0};
  double tabi_far_expansion_rel_error_mean{0.0};
  double tabi_far_expansion_rel_error_p95{0.0};
  double tabi_far_expansion_rel_error_p99{0.0};
  std::array<double, kJcpExpansionChannelCount>
      tabi_far_expansion_channel_rel_error_max{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_far_expansion_channel_rel_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      tabi_far_expansion_channel_rel_error_p99{};
  std::string tabi_far_expansion_worst_channel{"none"};
  bool tabi_far_expansion_mean_gate_pass{false};
  double tabi_far_convergence_rho_max{0.0};
  double tabi_far_convergence_rho_mean{0.0};
  double tabi_accepted_far_aspect_ratio_max{0.0};
  double tabi_accepted_far_aspect_ratio_mean{0.0};
  std::string tabi_quad_buffer_auto_candidate{"off"};
  std::string tabi_leaf_tile_status{"off"};
  double tabi_leaf_panel_pairs_per_second{0.0};
  std::string tabi_near_integration{"off"};
  std::string tabi_quad_buffer{"off"};
  std::size_t tabi_near_self_pairs{0};
  std::size_t tabi_near_one_ring_pairs{0};
  std::size_t tabi_near_duffy_pairs{0};
  std::size_t tabi_near_centroid_pairs{0};
  std::size_t tabi_near_fallbacks{0};
  std::size_t tabi_near_geometric_buffer_pairs{0};
  std::size_t tabi_near_geometric_buffer_gauss_pairs{0};
  std::size_t tabi_near_geometric_buffer_adaptive_pairs{0};
  std::size_t tabi_near_geometric_buffer_centroid_pairs{0};
  double tabi_near_geometric_buffer_min_gap{0.0};
  double tabi_near_geometric_buffer_mean_gap_ratio{0.0};
  double tabi_near_geometric_buffer_max_edge_mean{0.0};
  double tabi_near_integration_ms{0.0};
  std::size_t tabi_duffy_subdivisions_max{0};
  double tabi_duffy_subdivisions_mean{0.0};
  std::string tabi_near_kernel_policy{"off"};
  std::size_t tabi_near_k1_accepted{0};
  std::size_t tabi_near_k1_rejected{0};
  std::size_t tabi_near_k2_accepted{0};
  std::size_t tabi_near_k2_rejected{0};
  std::size_t tabi_near_k3_accepted{0};
  std::size_t tabi_near_k3_rejected{0};
  std::size_t tabi_near_k4_accepted{0};
  std::size_t tabi_near_k4_rejected{0};
  std::size_t tabi_near_one_ring_k1_accepted{0};
  std::size_t tabi_near_one_ring_k1_rejected{0};
  std::size_t tabi_near_one_ring_k2_accepted{0};
  std::size_t tabi_near_one_ring_k2_rejected{0};
  std::size_t tabi_near_one_ring_k3_accepted{0};
  std::size_t tabi_near_one_ring_k3_rejected{0};
  std::size_t tabi_near_one_ring_k4_accepted{0};
  std::size_t tabi_near_one_ring_k4_rejected{0};
  std::string tabi_near_k1_policy{"off"};
  std::string tabi_near_k2_policy{"off"};
  std::string tabi_near_k3_policy{"off"};
  std::string tabi_near_k4_policy{"off"};
  NearBlockAuditBuckets tabi_near_block_audit{};
  GeometricNearAuditBuckets tabi_geometric_near_audit{};
  std::string tabi_surface_geometry{"off"};
  std::string tabi_sphere_mesh_mode{"off"};
  double tabi_surface_quality_total_area_a2{0.0};
  double tabi_surface_quality_area_rel_error_vs_4piR2{0.0};
  double tabi_surface_quality_triangle_area_min{0.0};
  double tabi_surface_quality_triangle_area_mean{0.0};
  double tabi_surface_quality_triangle_area_max{0.0};
  double tabi_surface_quality_triangle_aspect_max{0.0};
  double tabi_surface_quality_normal_radial_dot_min{0.0};
  double tabi_surface_quality_normal_radial_dot_mean{0.0};
  double tabi_surface_quality_centroid_radial_error_max{0.0};
  double tabi_surface_quality_planar_area_a2{0.0};
  double tabi_surface_quality_planar_area_rel_error_vs_4piR2{0.0};
  double tabi_surface_quality_enclosed_volume_a3{0.0};
  double tabi_surface_quality_enclosed_volume_rel_error_vs_4piR3_over3{0.0};
  double tabi_surface_quality_triangle_area_cv{0.0};
  std::string tabi_surface_topology_source{"not_built"};
  std::string tabi_surface_analytic_ses_engine{"not_applicable"};
  std::string tabi_surface_status_detail{"not_built"};
  std::string tabi_surface_cavity_policy{"not_applicable"};
  std::size_t tabi_surface_odd_ray_rows{0};
  std::size_t tabi_surface_patch_ray_unresolved_uncertified_rows{0};
  std::size_t tabi_surface_morphology_fallback_vertices{0};
  std::string tabi_surface_analytic_orientation_status{"not_run"};
  std::size_t tabi_surface_analytic_inverted_faces{0};
  std::size_t tabi_surface_analytic_orientation_uncertified_faces{0};
  std::size_t tabi_surface_analytic_self_intersections{0};
  std::size_t tabi_surface_wide_repair_attempts{0};
  std::size_t tabi_surface_topology_certificate_accepts{0};
  std::size_t tabi_surface_topology_fallback_audits{0};
  double tabi_surface_initial_collapse_ms{0.0};
  double tabi_surface_stable_sliver_ms{0.0};
  double tabi_surface_full_projection_ms{0.0};
  double tabi_surface_orientation_repair_ms{0.0};
  double tabi_surface_wide_repair_ms{0.0};
  double tabi_surface_self_intersection_audit_ms{0.0};
  std::size_t tabi_surface_filled_cavity_regions{0};
  std::size_t tabi_surface_filled_cavity_vertices{0};
  std::size_t tabi_surface_poor_triangles_before{0};
  std::size_t tabi_surface_poor_triangles_after{0};
  double tabi_surface_triangle_quality_min_before{0.0};
  double tabi_surface_triangle_quality_min_after{0.0};
  double tabi_surface_max_edge_ratio_before{0.0};
  double tabi_surface_max_edge_ratio_after{0.0};
  std::string tabi_surface_analytic_audit_status{"not_run"};
  std::size_t tabi_surface_analytic_audit_samples{0};
  std::size_t tabi_surface_analytic_audit_hits{0};
  std::size_t tabi_surface_analytic_audit_misses{0};
  double tabi_surface_analytic_projection_distance_rms{0.0};
  double tabi_surface_analytic_projection_distance_max{0.0};
  double tabi_surface_analytic_patch_residual_rms{0.0};
  double tabi_surface_analytic_patch_residual_max{0.0};
  double tabi_surface_analytic_normal_angle_rms_deg{0.0};
  double tabi_surface_analytic_normal_angle_max_deg{0.0};
  double tabi_surface_sampled_hausdorff_proxy{0.0};
  std::string tabi_surface_analytic_normal_quality_status{"not_run"};
  std::size_t tabi_surface_analytic_orientation_smooth_faces{0};
  std::size_t tabi_surface_analytic_orientation_seam_faces{0};
  std::size_t tabi_surface_analytic_normal_cosine_below_025_faces{0};
  std::size_t tabi_surface_analytic_normal_cosine_below_050_faces{0};
  double tabi_surface_analytic_normal_area_weighted_rms_deg{0.0};
  double tabi_surface_analytic_normal_p99_deg{0.0};
  double tabi_surface_analytic_smooth_normal_p99_deg{0.0};
  double tabi_surface_analytic_seam_normal_p99_deg{0.0};
  double tabi_surface_analytic_normal_area_below_cosine_025_fraction{0.0};
  double tabi_surface_analytic_normal_area_below_cosine_050_fraction{0.0};
  std::string tabi_calderon_trace_space_status{"off"};
  std::size_t tabi_calderon_trace_vertices{0};
  std::size_t tabi_calderon_trace_edges{0};
  std::size_t tabi_calderon_trace_boundary_edges{0};
  std::size_t tabi_calderon_trace_orientation_mismatches{0};
  std::size_t tabi_calderon_trace_nonmanifold_edges{0};
  std::size_t tabi_calderon_trace_unreferenced_vertices{0};
  std::size_t tabi_calderon_trace_components{0};
  std::int64_t tabi_calderon_trace_euler_characteristic{0};
  std::size_t tabi_calderon_trace_genus_sum{0};
  bool tabi_calderon_trace_topology_invariants_valid{false};
  std::string tabi_calderon_aux_surface_mode{"off"};
  std::string tabi_calderon_aux_trace_status{"off"};
  std::size_t tabi_calderon_aux_panels{0};
  std::size_t tabi_calderon_aux_vertices{0};
  std::size_t tabi_calderon_aux_edges{0};
  std::size_t tabi_calderon_aux_boundary_edges{0};
  std::size_t tabi_calderon_aux_orientation_mismatches{0};
  std::size_t tabi_calderon_aux_nonmanifold_edges{0};
  std::size_t tabi_calderon_aux_unreferenced_vertices{0};
  std::size_t tabi_calderon_aux_grid_vertices{0};
  double tabi_calderon_aux_area_a2{0.0};
  double tabi_calderon_aux_build_ms{0.0};
  std::size_t tabi_mesh_vertices{0};
  std::size_t tabi_mesh_triangles{0};
  std::size_t tabi_boundary_nodes{0};
  double tabi_avg_valence{0.0};
  std::size_t tabi_nonmanifold_edges{0};
  std::size_t tabi_solid_angle_fallbacks{0};
  std::size_t tabi_solid_angle_clamps{0};
  double tabi_diag_min{1.0};
  double tabi_diag_max{1.0};
  int tabi_taylor_order{0};
  std::string tabi_derivative_mode{"off"};
  double tabi_interaction_build_ms{0.0};
  double tabi_tree_eval_ms{0.0};
  std::string tabi_formulation{"off"};
  std::string tabi_energy_projection{"off"};
  double tabi_energy_projection_ms{0.0};
  std::string tabi_energy_projection_kernel{"off"};
  std::string tabi_preconditioner_requested{"off"};
  std::string tabi_preconditioner_effective{"off"};
  std::string tabi_preconditioner_block_near{"off"};
  int tabi_preconditioner_block_panel_cap{32};
  int tabi_preconditioner_overlap_panel_cap{64};
  int tabi_preconditioner_ilu_neighbor_cap_requested{192};
  int tabi_preconditioner_ilu_neighbor_cap{192};
  double tabi_preconditioner_ilu_drop_tol{0.0};
  std::string tabi_preconditioner_ilu_regularization{"off"};
  double tabi_preconditioner_ilu_regularization_max{1.0e-2};
  double tabi_preconditioner_ilu_regularization_selected{0.0};
  double tabi_preconditioner_ilu_regularization_absolute_max{0.0};
  std::size_t tabi_preconditioner_ilu_regularization_attempts{0};
  std::size_t tabi_preconditioner_ilu_regularization_rejected_attempts{0};
  std::size_t tabi_preconditioner_ilu_regularization_shifted_rows{0};
  std::string tabi_preconditioner_ilu_unshifted_failure_stage{"off"};
  std::size_t tabi_preconditioner_memory_budget_bytes{0};
  std::size_t tabi_preconditioner_estimated_bytes{0};
  std::size_t tabi_preconditioner_valid_blocks{0};
  std::size_t tabi_preconditioner_valid_panels{0};
  std::size_t tabi_preconditioner_ill_conditioned_blocks{0};
  std::size_t tabi_preconditioner_quality_rejected_blocks{0};
  double tabi_preconditioner_quality_probe_max_gain{0.0};
  double tabi_preconditioner_quality_probe_rel_residual{0.0};
  double tabi_preconditioner_pivot_ratio_min{0.0};
  double tabi_preconditioner_factor_growth_max{0.0};
  std::int64_t tabi_preconditioner_pivot_ratio_min_row{-1};
  std::int64_t tabi_preconditioner_pivot_ratio_min_column{-1};
  std::int64_t tabi_preconditioner_factor_growth_row{-1};
  std::int64_t tabi_preconditioner_factor_growth_column{-1};
  std::string tabi_preconditioner_quality_failure_stage{"off"};
  std::int64_t tabi_preconditioner_quality_failure_row{-1};
  std::int64_t tabi_preconditioner_quality_failure_column{-1};
  std::int64_t tabi_preconditioner_quality_failure_probe{-1};
  double tabi_preconditioner_quality_failure_probe_gain{0.0};
  double tabi_preconditioner_quality_failure_probe_rel_residual{0.0};
  std::string tabi_preconditioner_quality_status{"off"};
  double tabi_preconditioner_setup_ms{0.0};
  double tabi_preconditioner_graph_setup_ms{0.0};
  bool tabi_preconditioner_graph_cache_reused{false};
  double tabi_preconditioner_near_assembly_setup_ms{0.0};
  double tabi_preconditioner_ilu_factor_setup_ms{0.0};
  double tabi_preconditioner_apply_ms{0.0};
  std::size_t tabi_preconditioner_apply_calls{0};
  std::size_t tabi_preconditioner_sparse_nnz_blocks{0};
  double tabi_preconditioner_sparse_avg_degree{0.0};
  std::size_t tabi_preconditioner_sparse_max_degree{0};
  int tabi_preconditioner_coarse_aggregate_cap{16};
  std::size_t tabi_preconditioner_coarse_aggregates{0};
  std::size_t tabi_preconditioner_coarse_dimension{0};
  std::string tabi_preconditioner_coarse_base{"diagonal"};
  std::size_t tabi_preconditioner_coarse_matrix_bytes{0};
  double tabi_preconditioner_coarse_pivot_ratio{0.0};
  std::string tabi_preconditioner_coarse_status{"off"};
  std::size_t tabi_preconditioner_coarse_setup_matvec_calls{0};
  double tabi_preconditioner_coarse_setup_ms{0.0};
  double tabi_preconditioner_coarse_moments_setup_ms{0.0};
  double tabi_preconditioner_coarse_far_setup_ms{0.0};
  double tabi_preconditioner_coarse_audit_setup_ms{0.0};
  double tabi_preconditioner_coarse_apply_ms{0.0};
  std::size_t tabi_preconditioner_coarse_apply_calls{0};
  double tabi_preconditioner_coarse_rp_identity_error_max{0.0};
  double tabi_preconditioner_coarse_verify_rel_error_max{0.0};
  std::string tabi_surface_generator{"off"};
  std::string tabi_discretization{"off"};
  std::string tabi_node_patch_representation{"n/a"};
  std::string tabi_node_patch_normal_source{"n/a"};
  std::string tabi_node_patch_certificate{"n/a"};
  std::string tabi_receptor_node_patch_certificate{"n/a"};
  std::string tabi_ligand_node_patch_certificate{"n/a"};
  double tabi_rhs_scale{1.0};
  double tabi_dielectric_contrast{0.0};
  double tabi_operator_contrast{0.0};
  std::size_t tabi_unknown_count{0};
  std::string tabi_bie_system{"off"};
  double tabi_lhs_diag_phi{0.0};
  double tabi_lhs_diag_phin{0.0};
  std::string tabi_energy_equation{"off"};
  std::string tabi_kernel_convention{"off"};
  double tabi_source_norm_s1{0.0};
  double tabi_source_norm_s2{0.0};
  double tabi_phi_norm{0.0};
  double tabi_phin_norm{0.0};
  double tabi_kappa_inv_a{0.0};
  double tabi_coulomb_factor_kcal_mol{0.0};
  double tabi_boundary_energy_kj_mol{0.0};
  double tabi_reference_energy_kj_mol{0.0};
  double tabi_selected_energy_scale{1.0};
  bool tabi_fallback_used{false};
  std::string pbe_model{"linear"};
  std::string pbe_model_effective{"linear"};
  std::string linear_seed_solver{"none"};
  std::string pbe_biomodel{"pbe"};
  std::string dielectric_model{"two-dielectric"};
  std::string nl_solver{"picard"};
  std::string nl_inner_solver{"sor"};
  std::string nl_mg_mode{"off"};
  double gaussian_exp{2.0};
  double gaussian_sigma{1.0};
  double density_cutoff{0.759};
  double surface_cutoff{0.0};
  double gap_diel{22.0};
  std::string radius_convention{"native"};
  std::string focus_parent_grid{};
  std::string focus_child_grid{};
  std::string probe_status{"off"};
  int newton_inner_iterations{0};
  double newton_step_norm{0.0};
  double newton_residual_before{0.0};
  double newton_residual_after{0.0};
  std::string newton_line_search_status{"off"};
  double newton_correction_residual{0.0};
  double newton_correction_max_residual{0.0};
  double newton_accepted_lambda{0.0};
  int newton_rejected_steps{0};
  double newton_jac_diag_min{0.0};
  double newton_jac_diag_max{0.0};
  std::size_t newton_screen_points{0};
  std::size_t newton_active_nonlinear_points{0};
  std::string newton_stop_reason{"off"};
  int newton_inner_levels{1};
  int newton_inner_cycles{0};
  double newton_inner_residual_before{0.0};
  double newton_inner_residual_after{0.0};
  std::string newton_inner_effective{"off"};
  double gaussian_density_sum{0.0};
  std::size_t gaussian_cutoff_points{0};
  double gaussian_eps_min{0.0};
  double gaussian_eps_max{0.0};
  double gaussian_eps_mean{0.0};
  double gaussian_charge_norm_error{0.0};
  double rpbe_coulomb_norm{0.0};
  double rpbe_grad_coulomb_norm{0.0};
  double rpbe_grad_epsilon_norm{0.0};
  double rpbe_source_rms{0.0};
  double rpbe_source_max{0.0};
  std::string rpbe_source_status{"off"};
  std::string boundary_mode{"coulombic"};
  std::string test_boundary_convention{"native-real"};
  std::string test_field_units{"native"};
  std::string test_operator_convention{"native"};
  std::string charge_assignment{"trilinear"};
  double rhs_scale{0.0};
  double charge_assigned{0.0};
  double charge_moment_error{0.0};
  double boundary_rms{0.0};
  double boundary_max{0.0};
  double linear_residual{0.0};
  int nl_iterations{0};
  int nl_blocks{0};
  double nl_relax{0.5};
  double nl_max_dphi{0.0};
  double nl_residual{0.0};
  std::size_t nl_clipped_points{0};
  int nl_continuation_steps{0};
  int nl_backtracks{0};
  std::string nl_status{"linear"};
  double energy_kcal_mol{0.0};
  double energy_kj_mol{0.0};
  double energy_kt{0.0};
  double solvent_phase_kcal_mol{0.0};
  double solvent_phase_kj_mol{0.0};
  double solvent_phase_kt{0.0};
  double vacuum_phase_kcal_mol{0.0};
  double vacuum_phase_kj_mol{0.0};
  double vacuum_phase_kt{0.0};
  double atom_corrected_kcal_mol{0.0};
  double atom_corrected_kj_mol{0.0};
  double atom_corrected_kt{0.0};
  double grid_corrected_kcal_mol{0.0};
  double grid_corrected_kj_mol{0.0};
  double grid_corrected_kt{0.0};
  double iscm_water_phase_kcal_mol{0.0};
  double iscm_water_phase_kj_mol{0.0};
  double iscm_water_phase_kt{0.0};
  double iscm_vacuum_phase_kcal_mol{0.0};
  double iscm_vacuum_phase_kj_mol{0.0};
  double iscm_vacuum_phase_kt{0.0};
  double iscm_corrected_kcal_mol{0.0};
  double iscm_corrected_kj_mol{0.0};
  double iscm_corrected_kt{0.0};
  double iscm_water_minus_vacuum_kcal_mol{0.0};
  double iscm_water_minus_vacuum_kj_mol{0.0};
  double iscm_water_minus_vacuum_kt{0.0};
  double nonlinear_stress_kt{0.0};
  double nonlinear_stress_kj_mol{0.0};
  double nonlinear_stress_kcal_mol{0.0};
  double nonlinear_osmotic_kt{0.0};
  double nonlinear_osmotic_kj_mol{0.0};
  double nonlinear_osmotic_kcal_mol{0.0};
  double nonlinear_energy_correction_kt{0.0};
  double nonlinear_energy_correction_kj_mol{0.0};
  double nonlinear_energy_correction_kcal_mol{0.0};
  std::string nonlinear_energy_status{"off"};
  std::string iscm_finalization_mode{"off"};
  std::string iscm_position_source{"off"};
  double iscm_position_rms_vs_oracle{0.0};
  std::size_t iscm_position_oracle_matches{0};
  bool iscm_native_independent{true};
  std::size_t iscm_boundary_points{0};
  std::size_t iscm_charged_boundary_points{0};
  double iscm_induced_charge_sum{0.0};
  double iscm_induced_charge_l1{0.0};
  double iscm_induced_charge_l2{0.0};
  double iscm_induced_charge_max{0.0};
  std::string iscm_status{"off"};
  std::size_t delphi_face_crossings{0};
  std::size_t delphi_eps_face_matches{0};
  std::size_t delphi_boundary_grid_points{0};
  std::size_t delphi_charged_boundary_points{0};
  std::string delphi_iscm_status{"off"};
  std::string vdwms_enabled{"off"};
  std::size_t vdwms_boundary_initial{0};
  std::size_t vdwms_boundary_final{0};
  std::size_t vdwms_added{0};
  std::size_t vdwms_removed{0};
  std::size_t vdwms_exposed_points{0};
  double vdwms_midpoint_rms{0.0};
  double coulomb_energy_kcal_mol{0.0};
  double coulomb_energy_kj_mol{0.0};
  double coulomb_energy_kt{0.0};
  std::string reaction_energy_source{"atom"};
  bool dgsolv_enabled{false};
  std::string dgsolv_status{"off"};
  double complex_energy_kcal_mol{0.0};
  double complex_energy_kj_mol{0.0};
  double mol1_energy_kcal_mol{0.0};
  double mol1_energy_kj_mol{0.0};
  double mol2_energy_kcal_mol{0.0};
  double mol2_energy_kj_mol{0.0};
  double deltaG_solv_kcal_mol{0.0};
  double deltaG_solv_kj_mol{0.0};
  double deltaG_solv_kt{0.0};
  bool pbsa_enabled{false};
  std::string pbsa_status{"off"};
  double pbsa_elec_kj_mol{0.0};
  double pbsa_vdw_kj_mol{0.0};
  double pbsa_mm_kj_mol{0.0};
  double pbsa_delta_sasa_a2{0.0};
  double pbsa_sasa_complex_a2{0.0};
  double pbsa_sasa_mol1_a2{0.0};
  double pbsa_sasa_mol2_a2{0.0};
  double pbsa_nonpolar_kj_mol{0.0};
  double pbsa_delta_solv_kj_mol{0.0};
  double pbsa_delta_solv_kcal_mol{0.0};
  double pbsa_no_entropy_kj_mol{0.0};
  double pbsa_no_entropy_kcal_mol{0.0};
  std::size_t pbsa_receptor_atom_count{0};
  std::size_t pbsa_ligand_atom_count{0};
  std::string pbsa_complex_status{"off"};
  std::string pbsa_receptor_status{"off"};
  std::string pbsa_ligand_status{"off"};
  int pbsa_receptor_solvent_iterations{0};
  int pbsa_receptor_vacuum_iterations{0};
  int pbsa_ligand_solvent_iterations{0};
  int pbsa_ligand_vacuum_iterations{0};
  int pbsa_receptor_mg_cycles{0};
  int pbsa_ligand_mg_cycles{0};
  double pbsa_receptor_solvent_rms{0.0};
  double pbsa_receptor_vacuum_rms{0.0};
  double pbsa_ligand_solvent_rms{0.0};
  double pbsa_ligand_vacuum_rms{0.0};
  double pbsa_receptor_solvent_max_dphi{0.0};
  double pbsa_receptor_vacuum_max_dphi{0.0};
  double pbsa_ligand_solvent_max_dphi{0.0};
  double pbsa_ligand_vacuum_max_dphi{0.0};
  double pbsa_time_complex_ms{0.0};
  double pbsa_time_receptor_ms{0.0};
  double pbsa_time_ligand_ms{0.0};
  double pbsa_time_components_ms{0.0};
  double pbsa_time_residue_ms{0.0};
  double pbsa_time_frame_ms{0.0};
  std::string pbsa_grid_reuse_effective{"off"};
  double pbsa_complex_reuse_delta_kj_mol{0.0};
  double pbsa_receptor_reuse_delta_kj_mol{0.0};
  double pbsa_ligand_reuse_delta_kj_mol{0.0};
  std::size_t pbsa_complex_reuse_surface_hits{0};
  std::size_t pbsa_complex_reuse_operator_hits{0};
  std::size_t pbsa_receptor_reuse_surface_hits{0};
  std::size_t pbsa_receptor_reuse_operator_hits{0};
  std::size_t pbsa_ligand_reuse_surface_hits{0};
  std::size_t pbsa_ligand_reuse_operator_hits{0};
  std::string status{"ok"};
};

} // namespace TitanPBE

#endif
