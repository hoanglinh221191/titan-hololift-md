#ifndef TITAN_PBCTOPO_REPORT_H
#define TITAN_PBCTOPO_REPORT_H

#include <cstddef>
#include <iosfwd>
#include <limits>
#include <string>

#include "types.h"

namespace titan_pbctopo {

enum class PbctopoFrameStatus {
  Certified,
  LocalUnwrap,
  Rescued,
  Fallback,
  WeakObservation,
  Contradicted,
};

enum class PbctopoCertificateScope {
  None,
  LocalUnwrap,
  ImageIntercomponentSteric,
  ImageIntercomponentStericInterface,
};

enum class PbctopoCertificateGraphSource {
  None,
  GromacsTopology,
  MetadataChainResidue,
  GeometryCutoff,
  Mixed,
};

enum class PbctopoHardGraphSource {
  None,
  ExplicitTopology,
  ValidatedMetadata,
  MixedHard,
};

enum class PbctopoSearchEvidenceSource {
  None,
  GeometryCutoff,
  ScoredContacts,
  ExplicitContacts,
  Mixed,
};

enum class PbctopoFrameMode {
  None,
  Virtual,
  EarlyProjection,
  ComponentGraph,
  ComponentSearchExact,
  ComponentSearchBeam,
  Warm,
  HullRescue,
  DistanceRescue,
  Pbci,
  LlpsLocal,
};

enum class PbctopoFailureReason {
  None,
  NoSeed,
  ProjectionInvalid,
  FullVerifyClash,
  ContinuityReject,
  AnchorReject,
  SplitOwnerReject,
  ComponentGraphFailed,
  BeamEmpty,
  NotBetterThanIncumbent,
  CertificationReject,
  InvalidLattice,
  LatticeQueryFailed,
  FallbackPbci,
  EvidenceDisconnected,
  EvidenceContradicted,
  EvidenceHypothesisUnavailable,
  EvidenceUnderdetermined,
};

constexpr bool pbctopo_frame_status_has_selected_assignment(
    PbctopoFrameStatus status) noexcept {
  return status == PbctopoFrameStatus::Certified ||
         status == PbctopoFrameStatus::Rescued ||
         status == PbctopoFrameStatus::WeakObservation ||
         status == PbctopoFrameStatus::Contradicted;
}

constexpr PbctopoFailureReason pbctopo_resolve_fallback_reason(
    bool lattice_query_failed, PbctopoFailureReason recorded_reason) noexcept {
  if (lattice_query_failed)
    return PbctopoFailureReason::LatticeQueryFailed;
  if (recorded_reason != PbctopoFailureReason::None)
    return recorded_reason;
  return PbctopoFailureReason::FallbackPbci;
}

const char *pbctopo_geometry_class_name(PbctopoGeometryClass value);

struct PbctopoScoreVector {
  bool valid = false;
  bool ultra_clash = false;
  std::size_t broken_edges = 0;
  double continuity_max_d2 = std::numeric_limits<double>::quiet_NaN();
  double continuity_path = std::numeric_limits<double>::quiet_NaN();
  double anchor_max_d2 = std::numeric_limits<double>::quiet_NaN();
  double anchor_mst2 = std::numeric_limits<double>::quiet_NaN();
  PbctopoAnchorGraphScore anchor_graph;
  double principal_rg2_max = std::numeric_limits<double>::quiet_NaN();
  double principal_rg2_mid = std::numeric_limits<double>::quiet_NaN();
  double principal_rg2_min = std::numeric_limits<double>::quiet_NaN();
  double shape_kappa2 = std::numeric_limits<double>::quiet_NaN();
  double shape_asphericity = std::numeric_limits<double>::quiet_NaN();
  double shape_acylindricity = std::numeric_limits<double>::quiet_NaN();
  double edge_stretch = std::numeric_limits<double>::quiet_NaN();
  double shift_norm2 = std::numeric_limits<double>::quiet_NaN();
  double gap_score = std::numeric_limits<double>::quiet_NaN();
};

struct PbctopoFrameReport {
  std::size_t frame = 0;
  double time_ps = 0.0;
  std::size_t trajectory_frame_index = 0;
  std::array<double, 9> box_matrix{};
  PbctopoHash128 box_hash;
  PbctopoHash128 atom_selection_hash;
  PbctopoHash128 canonical_atom_universe_hash;
  PbctopoHash128 wrapped_coordinate_hash;
  bool frame_binding_valid = false;
  std::uint32_t spatial_lift_policy_version = 0;
  PbctopoHash128 spatial_lift_identity;
  std::size_t spatial_lift_ambiguous_hard_edges = 0;
  std::size_t spatial_lift_cycle_residuals = 0;
  bool spatial_lift_valid = false;
  PbctopoFrameStatus status = PbctopoFrameStatus::Fallback;
  PbctopoCertificateScope certificate_scope = PbctopoCertificateScope::None;
  PbctopoCertificateGraphSource certificate_graph_source =
      PbctopoCertificateGraphSource::None;
  PbctopoHardGraphSource hard_graph_source = PbctopoHardGraphSource::None;
  PbctopoSearchEvidenceSource search_evidence_source =
      PbctopoSearchEvidenceSource::None;
  PbctopoFrameMode mode = PbctopoFrameMode::None;
  std::size_t component_count = 0;
  std::size_t soft_valid_candidate_count = 0;
  std::size_t nonclashing_candidate_count = 0;
  std::size_t unique_certified_assignment_count = 0;
  std::size_t committed_candidate_count = 0;
  PbctopoScoreVector selected_score;
  PbctopoScoreVector pool_best_score;
  PbctopoScoreVector pool_second_score;
  // Legacy pool aliases retained for downstream CSV readers.
  PbctopoScoreVector best_score;
  PbctopoScoreVector second_score;
  std::string score_gap_metric = "na";
  double score_gap = std::numeric_limits<double>::quiet_NaN();
  double clash_margin_A = std::numeric_limits<double>::quiet_NaN();
  double continuity_margin_A = std::numeric_limits<double>::quiet_NaN();
  std::size_t continuity_margin_atom_a =
      std::numeric_limits<std::size_t>::max();
  std::size_t continuity_margin_atom_b =
      std::numeric_limits<std::size_t>::max();
  double anchor_max_edge_A = std::numeric_limits<double>::quiet_NaN();
  double split_owner_gap_scaled = std::numeric_limits<double>::quiet_NaN();
  std::size_t split_owner_pairs = 0;
  PbctopoGeometryClassification geometry_classification;
  bool current_basis_fundamental_parallelepiped_certified = false;
  bool compact_unit_box_certified = false;
  PbctopoAnchorGraphScore anchor_graph;
  PbctopoHardFeasibility hard_feasibility;
  PbctopoFixedCertificateAudit fixed_certificate_audit;
  PbctopoIdentifiability identifiability;
  bool hard_feasible = false;
  PbctopoEvidenceState evidence_state = PbctopoEvidenceState::NotEvaluated;
  bool evidence_consistency_evaluated = false;
  bool evidence_consistent = false;
  bool evidence_graph_connected = false;
  PbctopoHash128 soft_observed_contact_pair_hash;
  PbctopoHash128 soft_observed_lost_pair_hash;
  PbctopoHash128 soft_observed_all_hypotheses_hash;
  PbctopoHash128 soft_observed_selected_hypothesis_hash;
  PbctopoHash128 soft_observed_selected_compatible_pair_hash;
  std::size_t soft_observed_hypothesis_count = 0;
  std::size_t soft_observed_component_relation_count = 0;
  std::size_t soft_observed_ambiguous_relation_count = 0;
  std::size_t soft_observed_compatible_hypothesis_count = 0;
  std::size_t soft_observed_selected_compatible_contact_count = 0;
  std::size_t soft_observed_alternative_hypothesis_contact_count = 0;
  std::size_t soft_observed_selected_compatible_contact_loss_count = 0;
  std::size_t soft_observed_no_support_relation_count = 0;
  bool soft_observed_selected_relation_has_support = false;
  bool soft_observed_hypothesis_construction_attempted = false;
  bool soft_observed_hypothesis_construction_complete = false;
  std::size_t soft_observed_hypothesis_mic_ambiguity_failures = 0;
  std::size_t soft_observed_hypothesis_component_mapping_failures = 0;
  std::size_t soft_observed_hypothesis_internal_edges_ignored = 0;
  std::size_t soft_observed_contacts_checked = 0;
  std::size_t soft_observed_contacts_lost = 0;
  double soft_observed_contact_cutoff_A =
      std::numeric_limits<double>::quiet_NaN();
  bool soft_observed_contact_construction_attempted = false;
  bool soft_observed_contact_construction_complete = false;
  std::size_t soft_observed_contact_cell_setup_failures = 0;
  std::size_t soft_observed_contact_atom_index_failures = 0;
  std::size_t soft_observed_contact_mic_query_failures = 0;
  std::uint64_t topology_epoch_id = 0;
  std::uint64_t component_layout_signature = 0;
  std::uint64_t assignment_signature = 0;
  std::size_t assignment_component_count = 0;
  bool temporal_selected = false;
  bool temporal_changed_from_greedy = false;
  bool temporal_inference_experimental = false;
  std::uint64_t temporal_assignment_signature = 0;
  // End-to-end virtual search is exhaustive only when every proof layer is.
  bool virtual_search_exhaustive = false;
  bool virtual_seam_enumeration_exhaustive = false;
  bool virtual_projection_scoring_exhaustive = false;
  bool virtual_full_atom_verification_exhaustive = false;
  bool output_certificate_search_exhaustive = false;
  bool virtual_branch_and_bound_used = false;
  std::uint64_t virtual_state_space = 0;
  std::size_t virtual_states_evaluated = 0;
  double virtual_omitted_seam_lower_bound =
      std::numeric_limits<double>::quiet_NaN();
  double virtual_lower_bound_gap =
      std::numeric_limits<double>::quiet_NaN();
  bool component_search_attempted = false;
  bool component_search_completed_under_admissible_bounds = false;
  bool component_search_incumbent_bound_pruning_used = false;
  bool component_search_best_objective_proven = false;
  bool component_search_equivalent_optima_complete = false;
  bool component_feasible_assignment_set_enumerated = false;
  std::size_t component_feasible_assignments_within_domain = 0;
  std::size_t component_soft_score_valid_assignments_within_domain = 0;
  bool component_bounded_domain_feasibility_known = false;
  bool component_bounded_domain_feasible = false;
  std::size_t component_search_shell_radius = 0;
  bool selected_assignment_boundary_known = false;
  bool selected_assignment_touches_shell_boundary = false;
  bool domain_expansion_attempted = false;
  bool domain_expansion_exhausted = false;
  bool bounded_domain_only = false;
  PbctopoDomainCompleteness domain_completeness =
      PbctopoDomainCompleteness::Unknown;
  // Legacy component_search_exhaustive aliases offset enumeration only.
  bool component_search_exhaustive = false;
  bool component_offset_enumeration_exhaustive = false;
  bool component_all_atom_feasibility_exhaustive = false;
  bool component_certificate_search_exhaustive = false;
  bool component_audit_exact_requested = false;
  bool component_audit_exact_completed = false;
  bool component_audit_exact_incomplete = false;
  std::string component_audit_exact_incomplete_reason{"none"};
  std::size_t component_audit_exact_states_visited = 0;
  double component_audit_exact_wall_seconds = 0.0;
  std::size_t component_audit_exact_memory_bytes_estimate = 0;
  std::string component_audit_exact_memory_budget_semantics{
      "estimated_workspace"};
  bool component_audit_exact_reference_comparison_available = false;
  bool component_audit_exact_reference_assignment_match = false;
  std::size_t beam_survivors_exactly_rescored = 0;
  bool component_search_optimality_gap_known = false;
  double component_search_optimality_gap =
      std::numeric_limits<double>::quiet_NaN();
  PbctopoFailureReason fallback_reason = PbctopoFailureReason::None;
  double time_total_ms = 0.0;
  double time_graph_ms = 0.0;
  double time_virtual_ms = 0.0;
  double time_component_ms = 0.0;
  std::size_t soft_pair_cycle_support = 0;
  double soft_pair_cycle_mismatch = 0.0;
  std::size_t soft_pair_inconsistent_edges = 0;
  std::size_t soft_pair_fundamental_cycle_count = 0;
  double soft_pair_fundamental_cycle_residual_rms = 0.0;
  double soft_pair_fundamental_cycle_residual_max = 0.0;
};

const char *pbctopo_frame_status_name(PbctopoFrameStatus status);
const char *pbctopo_evidence_state_name(PbctopoEvidenceState state);
const char *pbctopo_certificate_scope_name(PbctopoCertificateScope scope);
const char *pbctopo_certificate_graph_source_name(
    PbctopoCertificateGraphSource source);
const char *pbctopo_hard_graph_source_name(PbctopoHardGraphSource source);
const char *pbctopo_search_evidence_source_name(
    PbctopoSearchEvidenceSource source);
const char *pbctopo_frame_mode_name(PbctopoFrameMode mode);
const char *pbctopo_failure_reason_name(PbctopoFailureReason reason);
std::string format_pbctopo_report_double(double value);
std::string format_pbctopo_report_size(std::size_t value);
std::string format_pbctopo_score_vector(const PbctopoScoreVector &score);
void write_pbctopo_frame_report_header(std::ostream &out);
void write_pbctopo_frame_report_row(std::ostream &out,
                                    const PbctopoFrameReport &report);

} // namespace titan_pbctopo

#endif
