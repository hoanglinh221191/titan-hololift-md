#include "pbctopo/report.h"

#include "pbctopo/evidence_audit.h"
#include "pbctopo/certification.h"
#include "pbctopo/observation_binding.h"
#include "pbctopo/observation_schema.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>

namespace titan_pbctopo {

const char *pbctopo_geometry_class_name(PbctopoGeometryClass value) {
  switch (value) {
  case PbctopoGeometryClass::CompactUnitBox:
    return "compact_current_basis_parallelepiped";
  case PbctopoGeometryClass::Extended:
    return "extended";
  case PbctopoGeometryClass::Winding:
    return "winding";
  case PbctopoGeometryClass::ExtendedAndWinding:
    return "extended_and_winding";
  case PbctopoGeometryClass::Unknown:
  default:
    return "unknown";
  }
}
namespace {

std::string csv_escape(std::string value) {
  const bool needs_quotes = value.find_first_of(",\"\r\n") != std::string::npos;
  if (!needs_quotes)
    return value;
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"')
      escaped += "\"\"";
    else
      escaped += ch;
  }
  escaped += '"';
  return escaped;
}

PbctopoCertificateGraphSource effective_certificate_graph_source(
    const PbctopoFrameReport &report) {
  switch (report.hard_graph_source) {
  case PbctopoHardGraphSource::ExplicitTopology:
    return PbctopoCertificateGraphSource::GromacsTopology;
  case PbctopoHardGraphSource::ValidatedMetadata:
    return PbctopoCertificateGraphSource::MetadataChainResidue;
  case PbctopoHardGraphSource::MixedHard:
    return PbctopoCertificateGraphSource::Mixed;
  case PbctopoHardGraphSource::None:
  default:
    return PbctopoCertificateGraphSource::None;
  }
}

} // namespace

const char *pbctopo_frame_status_name(PbctopoFrameStatus status) {
  switch (status) {
  case PbctopoFrameStatus::Certified:
    return "certified";
  case PbctopoFrameStatus::WeakObservation:
    return "weak_observation";
  case PbctopoFrameStatus::LocalUnwrap:
    return "local_unwrap";
  case PbctopoFrameStatus::Rescued:
    return "rescued";
  case PbctopoFrameStatus::Contradicted:
    return "contradicted";
  case PbctopoFrameStatus::Fallback:
  default:
    return "fallback";
  }
}

const char *pbctopo_evidence_state_name(PbctopoEvidenceState state) {
  switch (state) {
  case PbctopoEvidenceState::NotEvaluated:
    return "not_evaluated";
  case PbctopoEvidenceState::NotApplicable:
    return "not_applicable";
  case PbctopoEvidenceState::SupportedConnected:
    return "supported_connected";
  case PbctopoEvidenceState::WeakDisconnected:
    return "weak_disconnected";
  case PbctopoEvidenceState::Contradicted:
    return "contradicted";
  case PbctopoEvidenceState::WeakAmbiguous:
    return "weak_ambiguous";
  }
  return "not_evaluated";
}

const char *pbctopo_certificate_scope_name(PbctopoCertificateScope scope) {
  switch (scope) {
  case PbctopoCertificateScope::LocalUnwrap:
    return "local_unwrap";
  case PbctopoCertificateScope::ImageIntercomponentSteric:
    return "image_intercomponent_steric";
  case PbctopoCertificateScope::ImageIntercomponentStericInterface:
    return "image_intercomponent_steric_interface";
  case PbctopoCertificateScope::None:
  default:
    return "none";
  }
}

const char *pbctopo_certificate_graph_source_name(
    PbctopoCertificateGraphSource source) {
  switch (source) {
  case PbctopoCertificateGraphSource::GromacsTopology:
    return "gromacs_topology";
  case PbctopoCertificateGraphSource::MetadataChainResidue:
    return "metadata_chain_residue";
  case PbctopoCertificateGraphSource::GeometryCutoff:
    return "geometry_cutoff";
  case PbctopoCertificateGraphSource::Mixed:
    return "mixed";
  case PbctopoCertificateGraphSource::None:
  default:
    return "none";
  }
}

const char *pbctopo_hard_graph_source_name(PbctopoHardGraphSource source) {
  switch (source) {
  case PbctopoHardGraphSource::ExplicitTopology:
    return "explicit_topology";
  case PbctopoHardGraphSource::ValidatedMetadata:
    return "validated_metadata";
  case PbctopoHardGraphSource::MixedHard:
    return "mixed_hard";
  case PbctopoHardGraphSource::None:
  default:
    return "none";
  }
}

const char *pbctopo_search_evidence_source_name(
    PbctopoSearchEvidenceSource source) {
  switch (source) {
  case PbctopoSearchEvidenceSource::GeometryCutoff:
    return "geometry_cutoff";
  case PbctopoSearchEvidenceSource::ScoredContacts:
    return "scored_contacts";
  case PbctopoSearchEvidenceSource::ExplicitContacts:
    return "explicit_contacts";
  case PbctopoSearchEvidenceSource::Mixed:
    return "mixed";
  case PbctopoSearchEvidenceSource::None:
  default:
    return "none";
  }
}

const char *pbctopo_frame_mode_name(PbctopoFrameMode mode) {
  switch (mode) {
  case PbctopoFrameMode::None:
    return "none";
  case PbctopoFrameMode::Virtual:
    return "virtual";
  case PbctopoFrameMode::EarlyProjection:
    return "early_projection";
  case PbctopoFrameMode::ComponentGraph:
    return "component_graph";
  case PbctopoFrameMode::ComponentSearchExact:
    return "component_exact";
  case PbctopoFrameMode::ComponentSearchBeam:
    return "beam";
  case PbctopoFrameMode::Warm:
    return "warm";
  case PbctopoFrameMode::HullRescue:
    return "hull_rescue";
  case PbctopoFrameMode::DistanceRescue:
    return "distance_rescue";
  case PbctopoFrameMode::Pbci:
    return "pbci";
  case PbctopoFrameMode::LlpsLocal:
    return "llps-local";
  default:
    return "pbci";
  }
}

const char *pbctopo_failure_reason_name(PbctopoFailureReason reason) {
  switch (reason) {
  case PbctopoFailureReason::None:
    return "none";
  case PbctopoFailureReason::NoSeed:
    return "no_seed";
  case PbctopoFailureReason::ProjectionInvalid:
    return "projection_invalid";
  case PbctopoFailureReason::FullVerifyClash:
    return "full_verify_clash";
  case PbctopoFailureReason::ContinuityReject:
    return "continuity_reject";
  case PbctopoFailureReason::AnchorReject:
    return "anchor_reject";
  case PbctopoFailureReason::SplitOwnerReject:
    return "split_owner_reject";
  case PbctopoFailureReason::ComponentGraphFailed:
    return "component_graph_failed";
  case PbctopoFailureReason::BeamEmpty:
    return "beam_empty";
  case PbctopoFailureReason::NotBetterThanIncumbent:
    return "not_better_than_incumbent";
  case PbctopoFailureReason::EvidenceDisconnected:
    return "evidence_disconnected";
  case PbctopoFailureReason::EvidenceContradicted:
    return "evidence_contradicted";
  case PbctopoFailureReason::EvidenceHypothesisUnavailable:
    return "evidence_hypothesis_unavailable";
  case PbctopoFailureReason::EvidenceUnderdetermined:
    return "evidence_underdetermined";
  case PbctopoFailureReason::CertificationReject:
    return "certification_reject";
  case PbctopoFailureReason::InvalidLattice:
    return "invalid_lattice";
  case PbctopoFailureReason::LatticeQueryFailed:
    return "lattice_query_failed";
  case PbctopoFailureReason::FallbackPbci:
  default:
    return "fallback_pbci";
  }
}

std::string format_pbctopo_report_double(double value) {
  if (!std::isfinite(value))
    return "na";
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10)
      << (value == 0.0 ? 0.0 : value);
  return out.str();
}

namespace {

std::string format_pbctopo_policy_double(double value) {
  return format_pbctopo_report_double(value);
}

} // namespace

std::string format_pbctopo_report_size(std::size_t value) {
  return value == std::numeric_limits<std::size_t>::max()
             ? std::string("na")
             : std::to_string(value);
}

std::string format_pbctopo_score_vector(const PbctopoScoreVector &score) {
  return "valid=" + std::string(score.valid ? "1" : "0") +
         ";ultra=" + std::string(score.ultra_clash ? "1" : "0") +
         ";broken=" + format_pbctopo_report_size(score.broken_edges) +
         ";cont_max=" +
         format_pbctopo_report_double(score.continuity_max_d2) +
         ";cont_path=" + format_pbctopo_report_double(score.continuity_path) +
         ";anchor_max=" + format_pbctopo_report_double(score.anchor_max_d2) +
         ";anchor_mst=" + format_pbctopo_report_double(score.anchor_mst2) +
         ";anchor_order_max=" +
         format_pbctopo_report_double(score.anchor_graph.order_max_d2) +
         ";anchor_order_path=" +
         format_pbctopo_report_double(score.anchor_graph.order_path_d2) +
         ";anchor_endpoint_gap=" +
         format_pbctopo_report_double(score.anchor_graph.endpoint_image_gap) +
         ";anchor_residue_violations=" +
         format_pbctopo_report_size(
             score.anchor_graph.residue_order_violations) +
         ";anchor_contact_lost=" +
         format_pbctopo_report_size(score.anchor_graph.contact_map_lost) +
         ";anchor_contact_max=" +
         format_pbctopo_report_double(score.anchor_graph.contact_map_max_d2) +
         ";anchor_ambiguity=" +
         format_pbctopo_report_size(score.anchor_graph.ambiguity_count) +
         ";rg2=" + format_pbctopo_report_double(score.principal_rg2_max) +
         "/" + format_pbctopo_report_double(score.principal_rg2_mid) + "/" +
         format_pbctopo_report_double(score.principal_rg2_min) +
         ";shape=" + format_pbctopo_report_double(score.shape_kappa2) +
         "/" + format_pbctopo_report_double(score.shape_asphericity) + "/" +
         format_pbctopo_report_double(score.shape_acylindricity) +
         ";edge=" + format_pbctopo_report_double(score.edge_stretch) +
         ";shift=" + format_pbctopo_report_double(score.shift_norm2) +
         ";gap=" + format_pbctopo_report_double(score.gap_score);
}

void write_pbctopo_frame_report_header(std::ostream &out) {
  out << "schema_version,producer_version,coordinate_unit,time_unit,"
         "endianness,floating_format,identity_hash_algorithm,"
         "binding_hash_algorithm,payload_checksum_algorithm,binding_mode,"
         "negative_zero_policy,"
         "nonfinite_policy,observation_scope,temporal_policy_status,"
         "frame,time_ps,trajectory_frame_index,frame_binding_valid,"
         "spatial_lift_policy_version,spatial_lift_identity,"
         "spatial_lift_ambiguous_hard_edges,spatial_lift_cycle_residuals,"
         "spatial_lift_valid,"
         "box_b00,box_b01,box_b02,box_b10,box_b11,box_b12,"
         "box_b20,box_b21,box_b22,box_hash,atom_selection_hash,"
         "canonical_atom_universe_hash,"
         "wrapped_coordinate_hash,status,certificate_scope,"
         "certificate_graph_source,"
         "hard_graph_source,search_evidence_source,"
         "hard_feasible,evidence_state,evidence_consistency_evaluated,"
         "evidence_consistent,evidence_graph_connected,"
         "soft_observed_contact_pair_hash,soft_observed_lost_pair_hash,"
         "soft_observed_all_hypotheses_hash,"
         "soft_observed_selected_hypothesis_hash,"
         "soft_observed_hypothesis_hash_algorithm,"
         "soft_observed_hypothesis_hash_domain,"
         "soft_observed_selected_compatible_pair_hash,"
         "soft_observed_hypothesis_count,"
         "soft_observed_component_relation_count,"
         "soft_observed_ambiguous_relation_count,"
         "soft_observed_compatible_hypothesis_count,"
         "soft_observed_selected_compatible_contact_count,"
         "soft_observed_alternative_hypothesis_contact_count,"
         "soft_observed_selected_compatible_contact_loss_count,"
         "soft_observed_no_support_relation_count,"
         "soft_observed_selected_relation_has_support,"
         "soft_observed_hypothesis_construction_attempted,"
         "soft_observed_hypothesis_construction_complete,"
         "soft_observed_hypothesis_mic_ambiguity_failures,"
         "soft_observed_hypothesis_component_mapping_failures,"
         "soft_observed_hypothesis_internal_edges_ignored,"
         "soft_observed_contact_hash_algorithm,"
         "soft_observed_contact_hash_domain,"
         "soft_observed_contact_cutoff_policy,"
         "soft_observed_contact_audit_policy_version,"
         "soft_observed_contact_max_pairs_per_unit_pair,"
         "soft_observed_contact_selection_tie_tolerance_d2_A2,"
         "soft_observed_contact_loss_tolerance_d2_A2,"
         "soft_observed_contact_cutoff_A,"
         "soft_observed_contact_construction_attempted,"
         "soft_observed_contact_construction_complete,"
         "soft_observed_contact_cell_setup_failures,"
         "soft_observed_contact_atom_index_failures,"
         "soft_observed_contact_mic_query_failures,identified,"
         "unique_within_search_domain,"
         "uniqueness_search_exhaustive,objective_uniqueness_known,"
         "objective_unique_within_search_domain,"
         "evidence_uniqueness_known,"
         "evidence_unique_within_search_domain,"
         "feasible_assignment_set_enumerated,"
         "feasible_assignment_set_semantics,"
         "feasible_assignments_within_domain,evidence_component_count,"
         "evidence_edge_count,explicit_contact_evidence_edges,"
         "scored_contact_evidence_edges,"
         "evidence_connected_component_count,"
         "unconstrained_relative_components,"
         "relative_lattice_degrees_of_freedom,"
         "equivalent_best_assignments_within_domain,"
         "certificate_atoms_total,hard_graph_atoms_covered,"
         "hard_graph_atom_fraction,topology_bond_atoms_covered,"
         "topology_bond_atom_fraction,metadata_only_atoms,"
         "metadata_only_atom_fraction,"
         "mode,component_count,"
         "soft_valid_candidate_count,nonclashing_candidate_count,"
         "unique_certified_assignment_count,committed_candidate_count,"
         "selected_score_vector,pool_best_score_vector,"
         "pool_second_score_vector,"
         "best_score_vector,second_score_vector,score_gap_metric,score_gap,"
         "clash_margin_A,continuity_margin_A,continuity_margin_atom_a,"
         "continuity_margin_atom_b,anchor_max_edge_A,"
         "split_owner_gap_scaled,split_owner_pairs,geometry_class,"
         "current_basis_fundamental_parallelepiped_certified,"
         "compact_unit_box_certified,scaled_span_x,scaled_span_y,"
         "scaled_span_z,fits_current_basis_fundamental_parallelepiped,"
         "fits_one_fundamental_cell,"
         "cartesian_extent_x_A,cartesian_extent_y_A,"
         "cartesian_extent_z_A,topological_winding_rank,"
         "topological_winding_b0_x,topological_winding_b0_y,"
         "topological_winding_b0_z,topological_winding_b1_x,"
         "topological_winding_b1_y,topological_winding_b1_z,"
         "topological_winding_b2_x,topological_winding_b2_y,"
         "topological_winding_b2_z,"
         "anchor_order_max_d2,anchor_order_path_d2,"
         "anchor_order_path_mean_d2,anchor_mst_mean_d2,"
         "anchor_endpoint_image_gap,anchor_residue_order_violations,"
         "anchor_contact_map_lost,anchor_contact_map_lost_fraction,"
         "anchor_contact_map_max_d2,soft_observed_contacts_checked,"
         "soft_observed_contacts_lost,soft_observed_contact_lost_fraction,"
         "anchor_ambiguity_count,hard_evaluated,hard_certified,"
         "hard_connectivity_edges_available,topology_edges_available,"
         "metadata_edges_available,"
         "hard_topology_edges_checked,hard_topology_edge_fraction,"
         "hard_ambiguous_topology_edges,"
         "hard_metadata_edges_checked,hard_metadata_edge_fraction,"
         "hard_ambiguous_metadata_edges,"
         "hard_metadata_edge_image_residuals,"
         "hard_metadata_edge_distance_residuals,"
         "hard_metadata_edge_max_distance_error_A,"
         "hard_edge_residuals,"
         "hard_atom_image_residuals,hard_atom_image_max_error,"
         "hard_assignment_point_residuals,hard_assignment_point_max_error,"
         "hard_steric_audit_available,hard_steric_decision_complete,"
         "hard_steric_pair_enumeration_complete,"
         "hard_steric_termination_reason,hard_steric_scan_complete,"
         "hard_steric_pairs_checked,"
         "steric_radius_force_field_atoms,steric_radius_explicit_zero_lj_atoms,"
         "steric_zero_lj_policy,"
         "steric_radius_inferred_atoms,"
         "steric_radius_generic_fallback_atoms,"
         "steric_radius_unavailable_atoms,"
         "hard_min_steric_margin_A,"
         "hard_topology_cycle_residuals,hard_unsupported_winding_rank,"
         "hard_clashes,hard_metadata_internal_pairs_checked,"
         "hard_metadata_internal_clashes,hard_required_contacts_checked,"
         "hard_required_contacts_lost,"
         "hard_explicit_interface_contacts_checked,"
         "hard_explicit_interface_contacts_lost,"
         "hard_explicit_interface_weight_checked,"
         "hard_explicit_interface_weight_lost,topology_epoch_id,"
         "component_layout_signature,"
         "assignment_signature,"
         "assignment_component_count,temporal_selected,"
         "temporal_changed_from_greedy,temporal_assignment_signature,"
         "virtual_search_exhaustive,"
         "virtual_seam_enumeration_exhaustive,"
         "virtual_projection_scoring_exhaustive,"
         "virtual_full_atom_verification_exhaustive,"
         "output_certificate_search_exhaustive,"
         "virtual_branch_and_bound_used,virtual_state_space,"
         "virtual_states_evaluated,virtual_omitted_seam_lower_bound,"
         "virtual_lower_bound_gap,component_search_attempted,"
         "component_search_completed_under_admissible_bounds,"
         "component_search_incumbent_bound_pruning_used,"
         "component_search_best_objective_proven,"
         "component_search_equivalent_optima_complete,"
         "component_feasible_assignment_set_enumerated,"
         "component_feasible_assignments_within_domain,"
         "component_soft_score_valid_assignments_within_domain,"
         "component_bounded_domain_feasibility_known,"
         "component_bounded_domain_feasible,"
         "component_search_shell_radius,"
         "selected_assignment_boundary_known,"
         "selected_assignment_touches_shell_boundary,"
         "domain_expansion_attempted,domain_expansion_exhausted,"
         "bounded_domain_only,domain_completeness,"
         "component_search_exhaustive,beam_survivors_exactly_rescored,"
         "component_offset_enumeration_exhaustive,"
         "component_all_atom_feasibility_exhaustive,"
         "component_certificate_search_exhaustive,"
         "component_fixed_certificate_audit_evaluated,"
         "component_fixed_certificate_input_valid,"
         "component_fixed_certificate_radius_model_valid,"
         "component_fixed_certificate_internal_steric_scan_complete,"
         "component_fixed_certificate_internal_pairs_checked,"
         "component_fixed_certificate_internal_clashes,"
         "component_fixed_certificate_required_contact_scan_complete,"
         "component_fixed_certificate_required_contacts_checked,"
         "component_fixed_certificate_required_contacts_lost,"
         "component_fixed_certificate_passed,"
         "component_audit_exact_requested,component_audit_exact_completed,"
         "component_audit_exact_incomplete,"
         "component_audit_exact_incomplete_reason,"
         "component_audit_exact_states_visited,"
         "component_audit_exact_wall_seconds,"
         "component_audit_exact_memory_bytes_estimate,"
         "component_audit_exact_memory_budget_semantics,"
         "component_audit_exact_reference_comparison_available,"
         "component_audit_exact_reference_assignment_match,"
         "component_search_optimality_gap_known,"
         "component_search_optimality_gap,fallback_reason,"
         "time_total_ms,time_graph_ms,time_virtual_ms,time_component_ms,"
         "soft_pair_cycle_support,soft_pair_cycle_mismatch,"
         "soft_pair_inconsistent_edges,"
         "soft_pair_fundamental_cycle_count,"
         "soft_pair_fundamental_cycle_residual_rms,"
         "soft_pair_fundamental_cycle_residual_max\n";
}

void write_pbctopo_frame_report_row(std::ostream &out,
                                    const PbctopoFrameReport &report) {
  const auto &hard = report.hard_feasibility;
  const auto &fixed = report.fixed_certificate_audit;
  const double hard_graph_atom_fraction =
      hard.certificate_atoms_total > 0
          ? static_cast<double>(hard.hard_graph_atoms_covered) /
                static_cast<double>(hard.certificate_atoms_total)
          : std::numeric_limits<double>::quiet_NaN();
  const double topology_bond_atom_fraction =
      hard.certificate_atoms_total > 0
          ? static_cast<double>(hard.topology_bond_atoms_covered) /
                static_cast<double>(hard.certificate_atoms_total)
          : std::numeric_limits<double>::quiet_NaN();
  const double metadata_only_atom_fraction =
      hard.certificate_atoms_total > 0
          ? static_cast<double>(hard.metadata_only_atoms) /
                static_cast<double>(hard.certificate_atoms_total)
          : std::numeric_limits<double>::quiet_NaN();
  const double hard_topology_edge_fraction =
      hard.topology_edges_available > 0
          ? static_cast<double>(
                hard.topology_edges_checked) /
                static_cast<double>(hard.topology_edges_available)
          : std::numeric_limits<double>::quiet_NaN();
  const double hard_metadata_edge_fraction =
      hard.metadata_edges_available > 0
          ? static_cast<double>(hard.metadata_edges_checked) /
                static_cast<double>(hard.metadata_edges_available)
          : std::numeric_limits<double>::quiet_NaN();
  out << VIBE_OBSERVATION_SCHEMA_VERSION << ','
      << csv_escape(vibe_observation_producer_version()) << ','
      << VIBE_OBSERVATION_COORDINATE_UNIT << ','
      << VIBE_OBSERVATION_TIME_UNIT << ','
      << VIBE_OBSERVATION_ENDIANNESS << ','
      << VIBE_OBSERVATION_FLOATING_FORMAT << ','
      << VIBE_OBSERVATION_IDENTITY_HASH_ALGORITHM << ','
      << VIBE_OBSERVATION_BINDING_HASH_ALGORITHM << ','
      << VIBE_OBSERVATION_PAYLOAD_CHECKSUM_ALGORITHM << ','
      << VIBE_OBSERVATION_BINDING_MODE << ','
      << VIBE_OBSERVATION_NEGATIVE_ZERO_POLICY << ','
      << VIBE_OBSERVATION_NONFINITE_POLICY << ','
      << VIBE_OBSERVATION_SCOPE << ','
       << (report.temporal_inference_experimental
              ? "experimental_viterbi_diagnostic"
              : "production_framewise_greedy")
       << ',' << report.frame << ','
       << format_pbctopo_report_double(report.time_ps)
       << ',' << report.trajectory_frame_index << ','
       << (report.frame_binding_valid ? 1 : 0) << ','
       << report.spatial_lift_policy_version << ','
       << format_pbctopo_hash128(report.spatial_lift_identity) << ','
       << report.spatial_lift_ambiguous_hard_edges << ','
       << report.spatial_lift_cycle_residuals << ','
       << (report.spatial_lift_valid ? 1 : 0);
  for (double value : report.box_matrix)
    out << ',' << format_pbctopo_policy_double(value);
  out << ',' << format_pbctopo_hash128(report.box_hash) << ','
      << format_pbctopo_hash128(report.atom_selection_hash) << ','
      << format_pbctopo_hash128(report.canonical_atom_universe_hash) << ','
      << format_pbctopo_hash128(report.wrapped_coordinate_hash) << ','
      << pbctopo_frame_status_name(report.status) << ','
       << pbctopo_certificate_scope_name(report.certificate_scope) << ','
      << pbctopo_certificate_graph_source_name(
             effective_certificate_graph_source(report))
      << ',' << pbctopo_hard_graph_source_name(report.hard_graph_source)
      << ','
      << pbctopo_search_evidence_source_name(report.search_evidence_source)
      << ',' << (report.hard_feasible ? 1 : 0)
      << ',' << pbctopo_evidence_state_name(report.evidence_state)
      << ',' << (report.evidence_consistency_evaluated ? 1 : 0)
      << ',' << (report.evidence_consistent ? 1 : 0)
      << ',' << (report.evidence_graph_connected ? 1 : 0)
      << ',' << format_pbctopo_hash128(report.soft_observed_contact_pair_hash)
      << ',' << format_pbctopo_hash128(report.soft_observed_lost_pair_hash)
      << ',' << format_pbctopo_hash128(
                    report.soft_observed_all_hypotheses_hash)
      << ',' << format_pbctopo_hash128(
                    report.soft_observed_selected_hypothesis_hash)
      << ',' << PBCTOPO_EOBS_PAIR_HASH_ALGORITHM
      << ',' << PBCTOPO_EOBS_HYPOTHESIS_HASH_DOMAIN
      << ',' << format_pbctopo_hash128(
                    report.soft_observed_selected_compatible_pair_hash)
      << ',' << report.soft_observed_hypothesis_count
      << ',' << report.soft_observed_component_relation_count
      << ',' << report.soft_observed_ambiguous_relation_count
      << ',' << report.soft_observed_compatible_hypothesis_count
      << ',' << report.soft_observed_selected_compatible_contact_count
      << ',' << report.soft_observed_alternative_hypothesis_contact_count
      << ',' << report.soft_observed_selected_compatible_contact_loss_count
      << ',' << report.soft_observed_no_support_relation_count
      << ',' << (report.soft_observed_selected_relation_has_support ? 1 : 0)
      << ',' << (report.soft_observed_hypothesis_construction_attempted ? 1 : 0)
      << ',' << (report.soft_observed_hypothesis_construction_complete ? 1 : 0)
      << ',' << report.soft_observed_hypothesis_mic_ambiguity_failures
      << ',' << report.soft_observed_hypothesis_component_mapping_failures
      << ',' << report.soft_observed_hypothesis_internal_edges_ignored
      << ',' << PBCTOPO_EOBS_PAIR_HASH_ALGORITHM
      << ',' << PBCTOPO_EOBS_PAIR_HASH_DOMAIN
      << ',' << PBCTOPO_EOBS_CUTOFF_POLICY
      << ',' << PBCTOPO_EOBS_AUDIT_POLICY_VERSION
      << ',' << PBCTOPO_EOBS_MAX_PAIRS_PER_UNIT_PAIR
      << ',' << format_pbctopo_policy_double(
                    PBCTOPO_EOBS_SELECTION_TIE_TOLERANCE_D2_A2)
      << ',' << format_pbctopo_policy_double(
                    PBCTOPO_EOBS_LOSS_TOLERANCE_D2_A2)
      << ',' << format_pbctopo_report_double(
                    report.soft_observed_contact_cutoff_A)
      << ',' << (report.soft_observed_contact_construction_attempted ? 1 : 0)
      << ',' << (report.soft_observed_contact_construction_complete ? 1 : 0)
      << ',' << report.soft_observed_contact_cell_setup_failures
      << ',' << report.soft_observed_contact_atom_index_failures
      << ',' << report.soft_observed_contact_mic_query_failures
      << ',' << (report.identifiability.identified ? 1 : 0)
      << ',' << (report.identifiability.unique_within_search_domain ? 1 : 0)
      << ',' << (report.identifiability.uniqueness_search_exhaustive ? 1 : 0)
      << ',' << (report.identifiability.objective_uniqueness_known ? 1 : 0)
      << ','
      << (report.identifiability.objective_unique_within_search_domain ? 1 : 0)
      << ',' << (report.identifiability.evidence_uniqueness_known ? 1 : 0)
      << ','
      << (report.identifiability.evidence_unique_within_search_domain ? 1 : 0)
      << ','
      << (report.identifiability.feasible_assignment_set_enumerated ? 1 : 0)
      << ','
      << (report.identifiability.feasible_assignment_set_enumerated
              ? "hard_constraints_within_search_domain"
              : "not_enumerated")
      << ',' << report.identifiability.feasible_assignments_within_domain
      << ',' << report.identifiability.evidence_component_count
      << ',' << report.identifiability.evidence_edge_count
      << ',' << report.identifiability.explicit_contact_evidence_edges
      << ',' << report.identifiability.scored_contact_evidence_edges
      << ',' << report.identifiability.evidence_connected_component_count
      << ',' << report.identifiability.unconstrained_relative_components
      << ',' << report.identifiability.relative_lattice_degrees_of_freedom
      << ','
      << report.identifiability.equivalent_best_assignments_within_domain
      << ',' << format_pbctopo_report_size(hard.certificate_atoms_total)
      << ',' << format_pbctopo_report_size(hard.hard_graph_atoms_covered)
      << ',' << format_pbctopo_report_double(hard_graph_atom_fraction)
      << ',' << format_pbctopo_report_size(hard.topology_bond_atoms_covered)
      << ',' << format_pbctopo_report_double(topology_bond_atom_fraction)
      << ',' << format_pbctopo_report_size(hard.metadata_only_atoms)
      << ',' << format_pbctopo_report_double(metadata_only_atom_fraction)
      << ','
      << pbctopo_frame_mode_name(report.mode) << ',' << report.component_count
      << ',' << report.soft_valid_candidate_count << ','
      << report.nonclashing_candidate_count << ','
      << report.unique_certified_assignment_count << ','
      << report.committed_candidate_count << ','
      << csv_escape(format_pbctopo_score_vector(report.selected_score)) << ','
      << csv_escape(format_pbctopo_score_vector(report.pool_best_score)) << ','
      << csv_escape(format_pbctopo_score_vector(report.pool_second_score)) << ','
      << csv_escape(format_pbctopo_score_vector(report.best_score)) << ','
      << csv_escape(format_pbctopo_score_vector(report.second_score)) << ','
      << csv_escape(report.score_gap_metric) << ','
      << format_pbctopo_report_double(report.score_gap) << ','
      << format_pbctopo_report_double(report.clash_margin_A) << ','
      << format_pbctopo_report_double(report.continuity_margin_A) << ','
      << format_pbctopo_report_size(report.continuity_margin_atom_a) << ','
      << format_pbctopo_report_size(report.continuity_margin_atom_b) << ','
      << format_pbctopo_report_double(report.anchor_max_edge_A) << ','
      << format_pbctopo_report_double(report.split_owner_gap_scaled) << ','
      << report.split_owner_pairs << ','
      << pbctopo_geometry_class_name(
             report.geometry_classification.geometry_class)
      << ','
      << (report.current_basis_fundamental_parallelepiped_certified ? 1 : 0)
      << ',' << (report.compact_unit_box_certified ? 1 : 0) << ','
      << format_pbctopo_report_double(
             report.geometry_classification.scaled_span[0])
      << ','
      << format_pbctopo_report_double(
             report.geometry_classification.scaled_span[1])
      << ','
      << format_pbctopo_report_double(
             report.geometry_classification.scaled_span[2])
      << ','
      << (report.geometry_classification
                  .fits_current_basis_fundamental_parallelepiped
              ? 1
              : 0)
      << ','
      << (report.geometry_classification.fits_one_fundamental_cell ? 1 : 0)
      << ','
      << format_pbctopo_report_double(
             report.geometry_classification.cartesian_extent_A[0])
      << ','
      << format_pbctopo_report_double(
             report.geometry_classification.cartesian_extent_A[1])
      << ','
      << format_pbctopo_report_double(
             report.geometry_classification.cartesian_extent_A[2])
      << ',' << report.geometry_classification.topological_winding_rank
      << ','
      << report.geometry_classification.topological_winding_basis[0].x
      << ','
      << report.geometry_classification.topological_winding_basis[0].y
      << ','
      << report.geometry_classification.topological_winding_basis[0].z
      << ','
      << report.geometry_classification.topological_winding_basis[1].x
      << ','
      << report.geometry_classification.topological_winding_basis[1].y
      << ','
      << report.geometry_classification.topological_winding_basis[1].z
      << ','
      << report.geometry_classification.topological_winding_basis[2].x
      << ','
      << report.geometry_classification.topological_winding_basis[2].y
      << ','
      << report.geometry_classification.topological_winding_basis[2].z
      << ','
      << format_pbctopo_report_double(report.anchor_graph.order_max_d2) << ','
      << format_pbctopo_report_double(report.anchor_graph.order_path_d2)
      << ','
      << format_pbctopo_report_double(report.anchor_graph.order_path_mean_d2)
      << ',' << format_pbctopo_report_double(report.anchor_graph.mst_mean_d2)
      << ','
      << format_pbctopo_report_double(report.anchor_graph.endpoint_image_gap)
      << ','
      << format_pbctopo_report_size(
             report.anchor_graph.residue_order_violations)
      << ',' << format_pbctopo_report_size(report.anchor_graph.contact_map_lost)
      << ','
      << format_pbctopo_report_double(
             report.anchor_graph.contact_map_lost_fraction)
      << ','
      << format_pbctopo_report_double(report.anchor_graph.contact_map_max_d2)
      << ','
      << format_pbctopo_report_size(report.soft_observed_contacts_checked)
      << ','
      << format_pbctopo_report_size(report.soft_observed_contacts_lost)
      << ','
      << format_pbctopo_report_double(
             report.soft_observed_selected_compatible_contact_count > 0
                 ? static_cast<double>(
                       report.soft_observed_selected_compatible_contact_loss_count) /
                       static_cast<double>(
                           report.soft_observed_selected_compatible_contact_count)
                 : std::numeric_limits<double>::quiet_NaN())
      << ',' << format_pbctopo_report_size(report.anchor_graph.ambiguity_count)
      << ',' << (report.hard_feasibility.evaluated ? 1 : 0) << ','
      << (report.hard_feasibility.certified() ? 1 : 0) << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.hard_connectivity_edges_available)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.topology_edges_available)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.metadata_edges_available)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.topology_edges_checked)
      << ',' << format_pbctopo_report_double(hard_topology_edge_fraction)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.ambiguous_topology_edges)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.metadata_edges_checked)
      << ','
      << format_pbctopo_report_double(hard_metadata_edge_fraction)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.ambiguous_metadata_edges)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.metadata_edge_image_residuals)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.metadata_edge_distance_residuals)
      << ','
      << format_pbctopo_report_double(
             report.hard_feasibility.max_metadata_edge_distance_error_A)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.hard_edge_residuals)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.atom_image_residuals)
      << ','
      << format_pbctopo_report_double(
             report.hard_feasibility.max_atom_image_error)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.assignment_point_residuals)
      << ','
      << format_pbctopo_report_double(
             report.hard_feasibility.max_assignment_point_error)
      << ',' << (report.hard_feasibility.steric_audit_available ? 1 : 0)
      << ',' << (report.hard_feasibility.steric_decision_complete ? 1 : 0)
      << ','
      << (report.hard_feasibility.steric_pair_enumeration_complete ? 1 : 0)
      << ',' << pbctopo_steric_termination_reason_name(
                        static_cast<PbctopoStericTerminationReason>(
                            report.hard_feasibility.steric_termination_reason))
      << ',' << (report.hard_feasibility.steric_scan_complete ? 1 : 0)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.steric_pairs_checked)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.steric_force_field_atoms)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.steric_explicit_zero_lj_atoms)
      << ",force_field_no_hard_core"
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.steric_inferred_atoms)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.steric_generic_fallback_atoms)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.steric_unavailable_atoms)
      << ','
      << format_pbctopo_report_double(
             report.hard_feasibility.min_steric_margin_A)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.topology_cycle_residuals)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.unsupported_winding_rank)
      << ','
      << format_pbctopo_report_size(report.hard_feasibility.hard_clashes)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.metadata_internal_pairs_checked)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.metadata_internal_clashes)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.required_contacts_checked)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.required_contacts_lost)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.explicit_interface_contacts_checked)
      << ','
      << format_pbctopo_report_size(
             report.hard_feasibility.explicit_interface_contacts_lost)
      << ','
      << format_pbctopo_report_double(
             report.hard_feasibility.explicit_interface_weight_checked)
      << ','
      << format_pbctopo_report_double(
             report.hard_feasibility.explicit_interface_weight_lost)
      << ',' << report.topology_epoch_id
      << ',' << report.component_layout_signature
      << ',' << report.assignment_signature << ','
      << report.assignment_component_count << ','
      << (report.temporal_selected ? 1 : 0) << ','
      << (report.temporal_changed_from_greedy ? 1 : 0) << ','
      << report.temporal_assignment_signature << ','
      << (report.virtual_search_exhaustive ? 1 : 0) << ','
      << (report.virtual_seam_enumeration_exhaustive ? 1 : 0) << ','
      << (report.virtual_projection_scoring_exhaustive ? 1 : 0) << ','
      << (report.virtual_full_atom_verification_exhaustive ? 1 : 0) << ','
      << (report.output_certificate_search_exhaustive ? 1 : 0) << ','
      << (report.virtual_branch_and_bound_used ? 1 : 0) << ','
      << report.virtual_state_space << ',' << report.virtual_states_evaluated
      << ','
      << format_pbctopo_report_double(
             report.virtual_omitted_seam_lower_bound)
      << ',' << format_pbctopo_report_double(report.virtual_lower_bound_gap)
      << ',' << (report.component_search_attempted ? 1 : 0) << ','
      << (report.component_search_completed_under_admissible_bounds ? 1 : 0)
      << ','
      << (report.component_search_incumbent_bound_pruning_used ? 1 : 0) << ','
      << (report.component_search_best_objective_proven ? 1 : 0) << ','
      << (report.component_search_equivalent_optima_complete ? 1 : 0)
      << ','
      << (report.component_feasible_assignment_set_enumerated ? 1 : 0)
      << ','
      << report.component_feasible_assignments_within_domain << ','
      << report.component_soft_score_valid_assignments_within_domain << ','
      << (report.component_bounded_domain_feasibility_known ? 1 : 0)
      << ','
      << (report.component_bounded_domain_feasible ? 1 : 0) << ','
      << report.component_search_shell_radius << ','
      << (report.selected_assignment_boundary_known ? 1 : 0) << ',';
  if (report.selected_assignment_boundary_known)
    out << (report.selected_assignment_touches_shell_boundary ? 1 : 0);
  else
    out << "na";
  out << ',' << (report.domain_expansion_attempted ? 1 : 0) << ','
      << (report.domain_expansion_exhausted ? 1 : 0) << ','
      << (report.bounded_domain_only ? 1 : 0) << ','
      << pbctopo_domain_completeness_name(report.domain_completeness) << ','
      << (report.component_search_exhaustive ? 1 : 0) << ','
      << report.beam_survivors_exactly_rescored << ','
      << (report.component_offset_enumeration_exhaustive ? 1 : 0) << ','
      << (report.component_all_atom_feasibility_exhaustive ? 1 : 0) << ','
      << (report.component_certificate_search_exhaustive ? 1 : 0) << ','
      << (fixed.evaluated ? 1 : 0) << ','
      << (fixed.input_valid ? 1 : 0) << ','
      << (fixed.radius_model_valid ? 1 : 0) << ','
      << (fixed.internal_steric_scan_complete ? 1 : 0) << ','
      << fixed.metadata_internal_pairs_checked << ','
      << fixed.metadata_internal_clashes << ','
      << (fixed.required_contact_scan_complete ? 1 : 0) << ','
      << fixed.required_contacts_checked << ','
      << fixed.required_contacts_lost << ','
      << (fixed.passed() ? 1 : 0) << ','
      << (report.component_audit_exact_requested ? 1 : 0) << ','
      << (report.component_audit_exact_completed ? 1 : 0) << ','
      << (report.component_audit_exact_incomplete ? 1 : 0) << ','
      << report.component_audit_exact_incomplete_reason << ','
      << report.component_audit_exact_states_visited << ','
      << format_pbctopo_report_double(
             report.component_audit_exact_wall_seconds)
      << ',' << report.component_audit_exact_memory_bytes_estimate << ','
      << csv_escape(report.component_audit_exact_memory_budget_semantics)
      << ','
      << (report.component_audit_exact_reference_comparison_available ? 1 : 0)
      << ','
      << (report.component_audit_exact_reference_assignment_match ? 1 : 0)
      << ','
      << (report.component_search_optimality_gap_known ? 1 : 0) << ','
      << format_pbctopo_report_double(report.component_search_optimality_gap)
      << ','
      << pbctopo_failure_reason_name(report.fallback_reason) << ','
      << format_pbctopo_report_double(report.time_total_ms) << ','
      << format_pbctopo_report_double(report.time_graph_ms) << ','
      << format_pbctopo_report_double(report.time_virtual_ms) << ','
      << format_pbctopo_report_double(report.time_component_ms) << ','
      << report.soft_pair_cycle_support << ','
      << format_pbctopo_report_double(report.soft_pair_cycle_mismatch) << ','
      << report.soft_pair_inconsistent_edges << ','
      << report.soft_pair_fundamental_cycle_count
      << ','
      << format_pbctopo_report_double(
             report.soft_pair_fundamental_cycle_residual_rms)
      << ','
      << format_pbctopo_report_double(
             report.soft_pair_fundamental_cycle_residual_max)
      << '\n';
}

} // namespace titan_pbctopo
