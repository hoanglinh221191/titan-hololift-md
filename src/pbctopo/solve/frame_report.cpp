#include "pbctopo/solve/solver.h"

#include "Multi_MOL.h"
#include "titan_executor.h"

using namespace std;

namespace pbctopo_solve {

bool PbctopoFrameSolver::record_output_candidate(PbctopoCutCandidate candidate,
                                                 const vector<Vec3> &points,
                                                 PbctopoFrameMode mode,
                                                 const vector<Int3> *known_offsets) {
  if (!candidate.valid || candidate.ultra_clash ||
      points.size() != base_atoms.size()) {
    return false;
  }
  ensure_component_context();
  vector<Int3> offsets;
  if (known_offsets != nullptr &&
      known_offsets->size() == components.size()) {
    offsets = *known_offsets;
  } else {
    offsets = extract_offsets_and_time(component_atoms, points);
  }
  if (!align_component_offsets_to_points(component_atoms, components,
                                         (*certification_lattice), points,
                                         offsets)) {
    ++timing.pbctopo_output_certification_band_precheck_rejects;
    return false;
  }
  candidate.assignment =
      make_pbctopo_image_assignment(offsets, components,
                                     component_atoms);
  if (!candidate.assignment.valid) {
    ++timing.pbctopo_output_certification_band_precheck_rejects;
    return false;
  }
  const auto source_point_consistency =
      titan_pbctopo::evaluate_pbctopo_assignment_point_consistency(
          component_atoms, points, atom_component_index,
          candidate.assignment, (*certification_lattice));
  if (!source_point_consistency.input_valid ||
      source_point_consistency.point_residuals != 0) {
    ++timing.pbctopo_output_certification_band_precheck_rejects;
    return false;
  }
  if (!annotate_candidate_evidence(candidate, points))
    return false;
  const PbctopoImageAssignment assignment = candidate.assignment;
  auto existing = find_if(
      output_candidate_band.begin(), output_candidate_band.end(),
      [&](const PbctopoOutputCandidateSnapshot &snapshot) {
        return same_pbctopo_image_assignment(
            snapshot.candidate.assignment, candidate.assignment);
      });
  if (existing != output_candidate_band.end()) {
    const PbctopoCutCandidate prior = existing->candidate;
    const bool equivalence_known =
        existing->candidate.search_equivalence_known ||
        candidate.search_equivalence_known;
    const bool equivalent_to_search_best =
        (existing->candidate.search_equivalence_known &&
         existing->candidate.equivalent_to_search_best) ||
        (candidate.search_equivalence_known &&
         candidate.equivalent_to_search_best);
    if (better_cut_candidate(candidate, existing->candidate)) {
      existing->candidate = candidate;
      existing->offsets = std::move(offsets);
      existing->mode = mode;
    }
    existing->candidate.search_equivalence_known = equivalence_known;
    existing->candidate.equivalent_to_search_best =
        equivalent_to_search_best;
    merge_pbctopo_search_domain_provenance(existing->candidate,
                                            prior);
    merge_pbctopo_search_domain_provenance(existing->candidate,
                                            candidate);
  } else {
    output_candidate_band.push_back(
        {candidate, std::move(offsets), mode});
  }
  stable_sort(
      output_candidate_band.begin(), output_candidate_band.end(),
      [](const PbctopoOutputCandidateSnapshot &lhs,
         const PbctopoOutputCandidateSnapshot &rhs) {
        return better_cut_candidate(lhs.candidate, rhs.candidate);
      });
  if (output_candidate_band.size() > output_certification_band_limit)
    output_candidate_band.resize(output_certification_band_limit);
  size_t offset_bytes = 0;
  for (const auto &snapshot : output_candidate_band) {
    offset_bytes += snapshot.offsets.capacity() * sizeof(Int3);
  }
  timing.pbctopo_output_certification_band_offset_bytes_peak = max(
      timing.pbctopo_output_certification_band_offset_bytes_peak,
      offset_bytes);
  if (base_atoms.size() <=
      numeric_limits<size_t>::max() /
          (max<size_t>(1, output_candidate_band.size()) *
           sizeof(Vec3))) {
    const size_t avoided_bytes = output_candidate_band.size() *
                                 base_atoms.size() * sizeof(Vec3);
    timing.pbctopo_output_certification_band_point_bytes_avoided_peak =
        max(timing
                .pbctopo_output_certification_band_point_bytes_avoided_peak,
            avoided_bytes);
  }
  return any_of(
      output_candidate_band.begin(), output_candidate_band.end(),
      [&](const PbctopoOutputCandidateSnapshot &snapshot) {
        return same_pbctopo_image_assignment(
            snapshot.candidate.assignment, assignment);
      });
}

void PbctopoFrameSolver::record_frame_failure(PbctopoFailureReason reason) {
  if (frame_report.fallback_reason == PbctopoFailureReason::None)
    frame_report.fallback_reason = reason;
}

bool PbctopoFrameSolver::commit_certified_candidate_to_warm_state(const PbctopoCutCandidate &selected_candidate) {
  const PbctopoWarmState trial_warm_state = warm_state;
  if (!commit_pbctopo_certified_warm_state(
          warm_state, frame_start_warm_state, trial_warm_state,
          selected_candidate, component_layout)) {
    ++timing.pbctopo_warm_state_commit_rejects;
    return false;
  }
  if (selected_candidate.virtual_seed_source !=
      PbctopoVirtualSeedSource::None) {
    update_kinematic_seed_history(warm_state, selected_candidate,
                                  frame_idx, timing);
  } else {
    reset_kinematic_seed_history(warm_state, timing);
  }
  ++timing.pbctopo_warm_state_commits;
  return true;
}

void PbctopoFrameSolver::apply_periodic_contact_audit(
    const titan_pbctopo::PbctopoPeriodicContactAudit &audit,
    bool relations_valid) {
  frame_report.steric_filter = runtime_cutoffs.steric_filter;
  frame_report.input_overlap_evaluated = audit.overlap.evaluated;
  frame_report.input_overlap_complete = audit.overlap.complete;
  frame_report.input_overlap_vdw_scale = audit.overlap.vdw_scale;
  frame_report.input_overlap_fallback_cutoff_A = audit.overlap.fallback_cutoff_A;
  frame_report.input_overlap_pairs = audit.overlap.overlap_pairs;
  frame_report.input_overlap_unit_pairs = audit.overlap.overlap_unit_pairs;
  frame_report.input_overlap_min_distance_A = audit.overlap.min_distance_A;
  frame_report.input_overlap_min_ratio = audit.overlap.min_ratio;
  frame_report.contact_cycle = {};
  if (relations_valid && audit.complete && audit.cycle.evaluated) {
    auto &cycle = frame_report.contact_cycle;
    cycle.evaluated = true;
    cycle.rank = audit.cycle.span.rank;
    cycle.basis = audit.cycle.span.basis;
    cycle.graph_components = audit.cycle.graph_components;
    cycle.periodic_graph_components = audit.cycle.periodic_graph_components;
    cycle.max_component_rank = audit.cycle.max_component_rank;
    cycle.nonzero_cycle_relations = audit.cycle.nonzero_cycle_relations;
    cycle.self_image_units = audit.cycle.self_image_units;
    cycle.self_image_atom_pairs = audit.cycle.self_image_atom_pairs;
    cycle.self_image_min_distance_A = audit.cycle.self_image_min_distance_A;
  }
  if (!audit.error.empty())
    append_pbctopo_log_line("[pbctopo] periodic contact audit at frame " +
                            to_string(frame_report.frame) + ": " +
                            audit.error);
}

void PbctopoFrameSolver::apply_selected_evidence_to_frame_identifiability() {
  titan_pbctopo::apply_pbctopo_final_selected_evidence(
      frame_identifiability, selected_evidence_audit.evaluated,
      selected_evidence_audit.evidence_graph_connected,
      selected_evidence_audit.evidence_component_count,
      selected_evidence_audit.evidence_edge_count,
      selected_evidence_audit.explicit_contact_evidence_edges,
      selected_evidence_audit.scored_contact_evidence_edges,
      selected_evidence_audit.evidence_connected_component_count);
}

void PbctopoFrameSolver::audit_selected_evidence(const vector<Vec3> &selected_points,
                                                 const PbctopoImageAssignment &assignment) {
  const auto audit_started = clock::now();
  auto audit = titan_pbctopo::audit_pbctopo_selected_hypotheses(
      span<const PbctopoAtom>(base_atoms), span<const Vec3>(selected_points),
      base_anchor_context.contact_hypotheses, assignment, true,
      explicit_evidence_component_edges);
  timing.pbctopo_selected_evidence_audit_s +=
      chrono::duration<double>(clock::now() - audit_started).count();
  if (!audit) {
    selected_evidence_audit = {};
    append_pbctopo_log_line(
        "[pbctopo] selected evidence audit failed closed at frame " +
        to_string(frame_report.frame) + ": " + audit.error());
    return;
  }
  selected_evidence_audit = *audit;
  cache_candidate_evidence_audit(assignment, *audit);
  apply_selected_evidence_to_frame_identifiability();
  ++timing.pbctopo_selected_evidence_audits;
  timing.pbctopo_selected_evidence_contacts_checked +=
      audit->contacts_checked;
  timing.pbctopo_selected_evidence_contacts_lost += audit->contacts_lost;
}

// Status and the warm-state commit must follow the same selected-evidence
// audit that finalize_frame_report writes over the frame identifiability.
// The contact graph built before the search can still call a frame
// disconnected that the audit of the selected assignment finds supported.
bool PbctopoFrameSolver::selected_relative_images_identified() {
  PbctopoIdentifiability selected_identifiability = frame_identifiability;
  titan_pbctopo::apply_pbctopo_final_selected_evidence(
      selected_identifiability, selected_evidence_audit.evaluated,
      selected_evidence_audit.evidence_graph_connected,
      selected_evidence_audit.evidence_component_count,
      selected_evidence_audit.evidence_edge_count,
      selected_evidence_audit.explicit_contact_evidence_edges,
      selected_evidence_audit.scored_contact_evidence_edges,
      selected_evidence_audit.evidence_connected_component_count);
  return pbctopo_relative_images_identified(selected_identifiability);
}

void PbctopoFrameSolver::export_h9_partition_and_orbit(const PbctopoCutCandidate &base_candidate,
                                                       const vector<Int3> &base_offsets) {
  if (!h9_orbit_enabled || !selected_evidence_audit.evaluated)
    return;
  if (!base_candidate.assignment.valid ||
      selected_evidence_audit.evidence_component_labels.size() !=
          components.size() ||
      base_candidate.assignment.component_ids.size() !=
          components.size() ||
      base_offsets.size() != components.size()) {
    append_pbctopo_log_line(
        "[pbctopo] H9 orbit skipped: evidence partition or assignment "
        "layout mismatch at frame " +
        to_string(frame_report.frame));
    return;
  }

  size_t canonical_root = 0;
  for (size_t comp_idx = 1; comp_idx < components.size(); ++comp_idx) {
    if (components[comp_idx].size() >
            components[canonical_root].size() ||
        (components[comp_idx].size() ==
             components[canonical_root].size() &&
         base_candidate.assignment.component_ids[comp_idx] <
             base_candidate.assignment.component_ids[canonical_root])) {
      canonical_root = comp_idx;
    }
  }
  const size_t root_label =
      selected_evidence_audit
          .evidence_component_labels[canonical_root];
  if (h9_evidence_component_report.is_open()) {
    for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
      h9_evidence_component_report
          << "h9_evidence_partition_v1," << frame_report.frame << ','
          << self.m_mol[frame_idx].time_of_snap << ',' << comp_idx
          << ',' << base_candidate.assignment.component_ids[comp_idx]
          << ',' << components[comp_idx].size() << ','
          << selected_evidence_audit
                 .evidence_component_labels[comp_idx]
          << ',' << static_cast<int>(comp_idx == canonical_root) << ','
          << selected_evidence_audit
                 .evidence_connected_component_count
          << ',' << selected_evidence_audit
                        .relative_lattice_degrees_of_freedom
          << '\n';
    }
  }

  const bool orbit_applicable =
      selected_evidence_audit.evidence_connected_component_count == 2 &&
      selected_evidence_audit.relative_lattice_degrees_of_freedom == 3 &&
      root_label < 2;
  if (!orbit_applicable || !h9_orbit_report.is_open())
    return;
  const size_t shifted_label = root_label == 0 ? 1 : 0;
  vector<Vec3> orbit_points;
  orbit_points.reserve(component_atoms.size());
  for (int qx = -1; qx <= 1; ++qx) {
    for (int qy = -1; qy <= 1; ++qy) {
      for (int qz = -1; qz <= 1; ++qz) {
        vector<Int3> orbit_offsets = base_offsets;
        bool within_radius = true;
        for (size_t comp_idx = 0; comp_idx < components.size();
             ++comp_idx) {
          if (selected_evidence_audit
                  .evidence_component_labels[comp_idx] !=
              shifted_label) {
            continue;
          }
          Int3 shifted{};
          if (!titan_pbctopo::checked_int3_add(
                  orbit_offsets[comp_idx], Int3{qx, qy, qz}, shifted)) {
            within_radius = false;
            break;
          }
          orbit_offsets[comp_idx] = shifted;
        }
        PbctopoCutCandidate orbit_candidate = base_candidate;
        if (within_radius) {
          const PbctopoImageAssignment provisional_assignment =
              make_pbctopo_image_assignment(
                  orbit_offsets, components, component_atoms);
          within_radius = provisional_assignment.valid;
          for (const Int3 &offset :
               provisional_assignment.component_offsets) {
            if (offset.x < -configured_component_shell_radius ||
                offset.x > configured_component_shell_radius ||
                offset.y < -configured_component_shell_radius ||
                offset.y > configured_component_shell_radius ||
                offset.z < -configured_component_shell_radius ||
                offset.z > configured_component_shell_radius) {
              within_radius = false;
              break;
            }
          }
        }
        bool hard_certified = false;
        optional<PbctopoSelectedEvidenceAudit> orbit_audit;
        if (within_radius) {
          double orbit_shift_norm2 = 0.0;
          build_component_points_from_offsets(
              component_atoms, components,
              self.m_mol[frame_idx].Box, orbit_offsets, orbit_points,
              orbit_shift_norm2);
          hard_certified = certify_candidate_for_output(
              orbit_candidate, orbit_points, &orbit_offsets);
          if (hard_certified) {
            auto audited =
                titan_pbctopo::audit_pbctopo_selected_hypotheses(
                    span<const PbctopoAtom>(base_atoms),
                    span<const Vec3>(orbit_points),
                    base_anchor_context.contact_hypotheses,
                    orbit_candidate.assignment, true,
                    explicit_evidence_component_edges);
            if (audited)
              orbit_audit = *audited;
          }
        }
        const bool evidence_evaluated =
            orbit_audit.has_value() && orbit_audit->evaluated;
        const bool evidence_noncontradicted =
            evidence_evaluated &&
            orbit_audit->state != PbctopoEvidenceState::Contradicted;
        const bool accepted = hard_certified &&
                              evidence_noncontradicted;
        h9_orbit_report
            << "h9_local_orbit_v1," << frame_report.frame << ','
            << self.m_mol[frame_idx].time_of_snap << ',' << qx << ','
            << qy << ',' << qz << ',' << root_label << ','
            << shifted_label << ',' << static_cast<int>(within_radius)
            << ',' << static_cast<int>(hard_certified) << ','
            << static_cast<int>(evidence_evaluated) << ','
            << static_cast<int>(evidence_noncontradicted) << ','
            << static_cast<int>(accepted) << ','
            << (orbit_candidate.assignment.valid
                    ? orbit_candidate.assignment.signature
                    : 0)
            << ','
            << (orbit_audit.has_value()
                    ? titan_pbctopo::pbctopo_evidence_state_name(
                          orbit_audit->state)
                    : "not_evaluated")
            << ','
            << (orbit_audit.has_value()
                    ? orbit_audit->evidence_connected_component_count
                    : 0)
            << ','
            << (orbit_audit.has_value()
                    ? orbit_audit->relative_lattice_degrees_of_freedom
                    : 0)
            << ','
            << (orbit_audit.has_value()
                    ? orbit_audit->evidence_edge_count
                    : 0)
            << ','
            << (orbit_audit.has_value()
                    ? orbit_audit->compatible_hypothesis_count
                    : 0)
            << ','
            << (orbit_audit.has_value()
                    ? orbit_audit->selected_compatible_contact_count
                    : 0)
            << '\n';
      }
    }
  }
}

void PbctopoFrameSolver::finalize_frame_report(PbctopoFrameStatus status,
                                               PbctopoFrameMode mode,
                                               PbctopoFailureReason reason,
                                               const PbctopoCutCandidate *selected_candidate) {
  if (selected_candidate != nullptr && selected_candidate->valid &&
      !selected_evidence_audit.evaluated) {
    status = PbctopoFrameStatus::Fallback;
    reason = PbctopoFailureReason::EvidenceHypothesisUnavailable;
  }
  if (selected_candidate != nullptr && selected_candidate->valid) {
    greedy_selected_candidates[frame_idx] = *selected_candidate;
  }
  frame_report.identifiability = frame_identifiability;
  frame_report.evidence_state = selected_evidence_audit.state;
  frame_report.evidence_consistency_evaluated =
      selected_evidence_audit.evaluated;
  frame_report.evidence_consistent =
      selected_evidence_audit.consistent;
  frame_report.evidence_graph_connected =
      selected_evidence_audit.evaluated
          ? selected_evidence_audit.evidence_graph_connected
          : frame_identifiability.evidence_graph_connected;
  titan_pbctopo::apply_pbctopo_final_selected_evidence(
      frame_report.identifiability,
      selected_evidence_audit.evaluated,
      selected_evidence_audit.evidence_graph_connected,
      selected_evidence_audit.evidence_component_count,
      selected_evidence_audit.evidence_edge_count,
      selected_evidence_audit.explicit_contact_evidence_edges,
      selected_evidence_audit.scored_contact_evidence_edges,
      selected_evidence_audit.evidence_connected_component_count);
  if (selected_candidate != nullptr && selected_candidate->valid &&
      selected_evidence_audit.evaluated) {
    const bool final_output_identified =
        pbctopo_relative_images_identified(
            frame_report.identifiability);
    const bool rescue_provenance =
        status == PbctopoFrameStatus::Rescued ||
        mode == PbctopoFrameMode::HullRescue ||
        mode == PbctopoFrameMode::DistanceRescue;
    if (selected_evidence_audit.state ==
        PbctopoEvidenceState::Contradicted) {
      status = PbctopoFrameStatus::Contradicted;
      reason = PbctopoFailureReason::EvidenceContradicted;
    } else if (pbctopo_evidence_state_underdetermined(selected_evidence_audit.state)) {
      status = PbctopoFrameStatus::WeakObservation;
      reason = pbctopo_weak_evidence_reason(selected_evidence_audit.state);
    } else if (selected_evidence_audit.state ==
                   PbctopoEvidenceState::WeakDisconnected ||
               !final_output_identified) {
      status = PbctopoFrameStatus::WeakObservation;
      reason = PbctopoFailureReason::EvidenceDisconnected;
    } else {
      status = rescue_provenance ? PbctopoFrameStatus::Rescued
                                 : PbctopoFrameStatus::Certified;
      reason = PbctopoFailureReason::None;
    }
  }
  // A bonded cycle that runs through the boundary leaves no whole lift of
  // the molecule; say so, and report its winding rank, instead of the search
  // failure that follows from it.
  if (status == PbctopoFrameStatus::Fallback &&
      reason != PbctopoFailureReason::LatticeQueryFailed &&
      frame_report.bonded_winding_rank > 0) {
    reason = PbctopoFailureReason::BondedWinding;
    if (frame_report.geometry_classification.topological_winding_rank == 0) {
      frame_report.geometry_classification.topological_winding_rank =
          frame_report.bonded_winding_rank;
      frame_report.geometry_classification.topological_winding_basis =
          frame_report.bonded_winding_basis;
    }
  }
  if (selected_evidence_audit.evaluated) {
    frame_report.soft_observed_contact_pair_hash =
        selected_evidence_audit.observed_pair_hash;
    frame_report.soft_observed_lost_pair_hash =
        selected_evidence_audit.lost_pair_hash;
    frame_report.soft_observed_all_hypotheses_hash =
        selected_evidence_audit.all_hypotheses_hash;
    frame_report.soft_observed_selected_hypothesis_hash =
        selected_evidence_audit.selected_hypothesis_hash;
    frame_report.soft_observed_selected_compatible_pair_hash =
        selected_evidence_audit.selected_compatible_pair_hash;
    frame_report.soft_observed_hypothesis_count =
        selected_evidence_audit.hypothesis_count;
    frame_report.soft_observed_component_relation_count =
        selected_evidence_audit.component_relation_count;
    frame_report.soft_observed_ambiguous_relation_count =
        selected_evidence_audit.ambiguous_component_relation_count;
    frame_report.soft_observed_compatible_hypothesis_count =
        selected_evidence_audit.compatible_hypothesis_count;
    frame_report.soft_observed_selected_compatible_contact_count =
        selected_evidence_audit.selected_compatible_contact_count;
    frame_report.soft_observed_alternative_hypothesis_contact_count =
        selected_evidence_audit.alternative_hypothesis_contact_count;
    frame_report.soft_observed_selected_compatible_contact_loss_count =
        selected_evidence_audit.selected_compatible_contact_loss_count;
    frame_report.soft_observed_no_support_relation_count =
        selected_evidence_audit.no_supported_contact_relation_count;
    frame_report.soft_observed_selected_relation_has_support =
        selected_evidence_audit.selected_relation_has_support;
    frame_report.soft_observed_hypothesis_construction_attempted =
        base_anchor_context.contact_hypotheses.attempted;
    frame_report.soft_observed_hypothesis_construction_complete =
        base_anchor_context.contact_hypotheses.complete;
    frame_report.soft_observed_hypothesis_mic_ambiguity_failures =
        base_anchor_context.contact_hypotheses.mic_ambiguity_failures;
    frame_report.soft_observed_hypothesis_component_mapping_failures =
        base_anchor_context.contact_hypotheses.component_mapping_failures;
    frame_report.soft_observed_hypothesis_internal_edges_ignored =
        base_anchor_context.contact_hypotheses.internal_edges_ignored;
    frame_report.soft_observed_contacts_checked =
        selected_evidence_audit.contacts_checked;
    frame_report.soft_observed_contacts_lost =
        selected_evidence_audit.contacts_lost;
    frame_report.soft_observed_contact_cutoff_A =
        runtime_cutoffs.pair_contact_cutoff;
    frame_report.soft_observed_contact_construction_attempted =
        base_anchor_context.soft_observed_contact_construction_attempted;
    frame_report.soft_observed_contact_construction_complete =
        base_anchor_context.soft_observed_contact_construction_complete;
    frame_report.soft_observed_contact_cell_setup_failures =
        base_anchor_context.soft_observed_contact_cell_setup_failures;
    frame_report.soft_observed_contact_atom_index_failures =
        base_anchor_context.soft_observed_contact_atom_index_failures;
    frame_report.soft_observed_contact_mic_query_failures =
        base_anchor_context.soft_observed_contact_mic_query_failures;
  } else {
    frame_report.soft_observed_contact_pair_hash =
        base_anchor_context.contact_hypotheses.raw_observed_pair_hash;
    frame_report.soft_observed_lost_pair_hash = {};
    frame_report.soft_observed_all_hypotheses_hash =
        base_anchor_context.contact_hypotheses.all_hypotheses_hash;
    frame_report.soft_observed_selected_hypothesis_hash = {};
    frame_report.soft_observed_selected_compatible_pair_hash = {};
    frame_report.soft_observed_hypothesis_count = 0;
    frame_report.soft_observed_component_relation_count = 0;
    frame_report.soft_observed_ambiguous_relation_count = 0;
    frame_report.soft_observed_compatible_hypothesis_count = 0;
    frame_report.soft_observed_selected_compatible_contact_count = 0;
    frame_report.soft_observed_alternative_hypothesis_contact_count = 0;
    frame_report.soft_observed_selected_compatible_contact_loss_count = 0;
    frame_report.soft_observed_no_support_relation_count = 0;
    frame_report.soft_observed_selected_relation_has_support = false;
    frame_report.soft_observed_hypothesis_construction_attempted =
        base_anchor_context.contact_hypotheses.attempted;
    frame_report.soft_observed_hypothesis_construction_complete = false;
    frame_report.soft_observed_hypothesis_mic_ambiguity_failures =
        base_anchor_context.contact_hypotheses.mic_ambiguity_failures;
    frame_report.soft_observed_hypothesis_component_mapping_failures =
        base_anchor_context.contact_hypotheses.component_mapping_failures;
    frame_report.soft_observed_hypothesis_internal_edges_ignored =
        base_anchor_context.contact_hypotheses.internal_edges_ignored;
    frame_report.soft_observed_contacts_checked =
        base_anchor_context.contact_hypotheses.raw_observed_pair_count;
    frame_report.soft_observed_contacts_lost = 0;
    frame_report.soft_observed_contact_cutoff_A =
        base_anchor_context.soft_observed_contact_construction_complete
            ? runtime_cutoffs.pair_contact_cutoff
            : numeric_limits<double>::quiet_NaN();
    frame_report.soft_observed_contact_construction_attempted =
        base_anchor_context.soft_observed_contact_construction_attempted;
    frame_report.soft_observed_contact_construction_complete =
        base_anchor_context.soft_observed_contact_construction_complete;
    frame_report.soft_observed_contact_cell_setup_failures =
        base_anchor_context.soft_observed_contact_cell_setup_failures;
    frame_report.soft_observed_contact_atom_index_failures =
        base_anchor_context.soft_observed_contact_atom_index_failures;
    frame_report.soft_observed_contact_mic_query_failures =
        base_anchor_context.soft_observed_contact_mic_query_failures;
  }
  const bool geometry_evidence = soft_geometry_edge_count > 0;
  const bool scored_contact_evidence =
      frame_identifiability.scored_contact_evidence_edges > 0;
  const bool explicit_contact_evidence =
      frame_identifiability.explicit_contact_evidence_edges > 0;
  const size_t evidence_source_count =
      static_cast<size_t>(geometry_evidence) +
      static_cast<size_t>(scored_contact_evidence) +
      static_cast<size_t>(explicit_contact_evidence);
  if (evidence_source_count > 1) {
    frame_report.search_evidence_source =
        PbctopoSearchEvidenceSource::Mixed;
  } else if (explicit_contact_evidence) {
    frame_report.search_evidence_source =
        PbctopoSearchEvidenceSource::ExplicitContacts;
  } else if (scored_contact_evidence) {
    frame_report.search_evidence_source =
        PbctopoSearchEvidenceSource::ScoredContacts;
  } else if (geometry_evidence) {
    frame_report.search_evidence_source =
        PbctopoSearchEvidenceSource::GeometryCutoff;
  }
  frame_report.status = status;
  frame_report.certificate_scope =
      status == PbctopoFrameStatus::Certified ||
              status == PbctopoFrameStatus::WeakObservation ||
              status == PbctopoFrameStatus::Rescued ||
              status == PbctopoFrameStatus::Contradicted
          ? (explicit_required_contacts.empty()
                 ? PbctopoCertificateScope::ImageIntercomponentSteric
                 : PbctopoCertificateScope::
                       ImageIntercomponentStericInterface)
          : PbctopoCertificateScope::None;
  if (frame_report.mode == PbctopoFrameMode::None ||
      status != PbctopoFrameStatus::Certified) {
    frame_report.mode = mode;
  }
  if (reason != PbctopoFailureReason::None)
    frame_report.fallback_reason = reason;
  else if (status != PbctopoFrameStatus::Fallback)
    frame_report.fallback_reason = PbctopoFailureReason::None;
  frame_report.component_count = components.size();
  frame_report.soft_valid_candidate_count =
      report_soft_valid_candidates;
  frame_report.nonclashing_candidate_count =
      report_nonclashing_candidates;
  frame_report.unique_certified_assignment_count =
      report_certified_candidates.size();
  frame_report.committed_candidate_count =
      status == PbctopoFrameStatus::Certified ||
              status == PbctopoFrameStatus::WeakObservation ||
              status == PbctopoFrameStatus::Rescued
          ? 1
          : 0;

  const PbctopoCutCandidate *selected_for_report =
      pbctopo_frame_status_has_selected_assignment(status) &&
              selected_candidate != nullptr && selected_candidate->valid
          ? selected_candidate
          : nullptr;
  const PbctopoCutCandidate *best_for_report =
      report_certified_candidates.empty()
          ? nullptr
          : &report_certified_candidates.front();
  const PbctopoCutCandidate *second_for_report =
      report_certified_candidates.size() > 1
          ? &report_certified_candidates[1]
          : nullptr;
  if (best_for_report != nullptr && best_for_report->valid) {
    frame_report.pool_best_score =
        make_score_vector(*best_for_report);
    frame_report.best_score = frame_report.pool_best_score;
  }
  if (second_for_report != nullptr && second_for_report->valid) {
    frame_report.pool_second_score =
        make_score_vector(*second_for_report);
    frame_report.second_score = frame_report.pool_second_score;
  }
  if (temporal_mode == PbctopoTemporalMode::Viterbi) {
    for (auto &temporal_candidate :
         temporal_cut_candidates[frame_idx]) {
      if (pbctopo_candidate_anchor_ambiguous(
              temporal_candidate, report_certified_candidates)) {
        temporal_candidate.anchor_graph.ambiguity_count =
            max<size_t>(1,
                        temporal_candidate.anchor_graph.ambiguity_count);
      }
    }
  }
  if (selected_for_report != nullptr) {
    populate_pbctopo_selected_candidate_report(
        frame_report, *selected_for_report, base_atoms,
        runtime_cutoffs.clash_cutoff,
        runtime_cutoffs.continuity_cutoff,
        report_certified_candidates);
    if (frame_report.anchor_graph.ambiguity_count > 0 &&
        !frame_anchor_ambiguity_counted) {
      ++timing.pbctopo_anchor_ambiguous_frames;
      frame_anchor_ambiguity_counted = true;
    }
  } else {
    frame_report.hard_feasibility = {};
    frame_report.hard_feasible = false;
  }
  if (best_for_report != nullptr && best_for_report->valid &&
      second_for_report != nullptr && second_for_report->valid) {
    const PbctopoLexicographicGap score_gap =
        candidate_lexicographic_gap(*best_for_report,
                                    *second_for_report);
    frame_report.score_gap_metric = score_gap.metric;
    frame_report.score_gap = score_gap.gap;
  }
  if (evidence_relation_export_enabled &&
      !evidence_relation_frame_written) {
    evidence_relation_frame_written = true;
    if (!evidence_relation_cache_error.empty()) {
      ++evidence_relation_export_failures;
      append_pbctopo_log_line(
          "[pbctopo] candidate evidence relation export failed "
          "closed at frame " +
          to_string(frame_report.frame) + ": " +
          evidence_relation_cache_error);
    } else {
      const auto &export_candidates =
          temporal_mode == PbctopoTemporalMode::Viterbi
              ? temporal_cut_candidates[frame_idx]
              : report_certified_candidates;
      const auto exported = write_pbctopo_candidate_evidence_frame(
          pbctopo_candidate_evidence_summary_report,
          pbctopo_candidate_evidence_relation_report, frame_report,
          export_candidates, selected_for_report,
          temporal_mode == PbctopoTemporalMode::Viterbi,
          evidence_relation_audit_snapshots);
      if (!exported) {
        ++evidence_relation_export_failures;
        append_pbctopo_log_line(
            "[pbctopo] candidate evidence relation export failed "
            "closed at frame " +
            to_string(frame_report.frame) + ": " + exported.error());
      } else {
        evidence_relation_export_groups +=
            exported->candidate_groups;
        evidence_relation_export_rows += exported->relation_rows;
        evidence_relation_export_duplicates +=
            exported->duplicate_candidates_skipped;
      }
    }
  }
  // The certified band is exported only for frames that have one.
  // The rejection ledger records every evaluated rejection whatever
  // the frame's status: a frame at which every candidate failed and
  // fell back is where the ledger matters most, and an empty ledger
  // must mean "nothing was rejected", not "the frame was skipped".
  const bool band_status =
      status == PbctopoFrameStatus::Certified ||
      status == PbctopoFrameStatus::Rescued ||
      status == PbctopoFrameStatus::WeakObservation;
  if ((pbctopo_assignment_set_report.is_open() ||
       pbctopo_rejected_assignment_report.is_open()) &&
      temporal_mode != PbctopoTemporalMode::Viterbi &&
      (band_status || !report_rejected_candidates.empty())) {
    double v1[3]{}, v2[3]{}, v3[3]{};
    self.m_mol[frame_idx].Box.get_v1(v1);
    self.m_mol[frame_idx].Box.get_v2(v2);
    self.m_mol[frame_idx].Box.get_v3(v3);
    const auto normalized_gram =
        titan_pbctopo::pbctopo_temporal_normalized_gram(
            {v1[0], v1[1], v1[2], v2[0], v2[1], v2[2],
             v3[0], v3[1], v3[2]});
    vector<PbctopoCutCandidate> report_band =
        report_certified_candidates;
    report_band.insert(report_band.end(),
                       report_rejected_candidates.begin(),
                       report_rejected_candidates.end());
    if (pbctopo_rejected_assignment_report.is_open())
      titan_pbctopo::write_pbctopo_rejected_assignment_rows(
          pbctopo_rejected_assignment_report,
          frame_report_offset + frame_idx,
          self.m_mol[frame_idx].time_of_snap, report_band,
          normalized_gram, frame_report.topology_epoch_id);
    if (band_status && pbctopo_assignment_set_report.is_open())
      write_pbctopo_assignment_set_rows(
        pbctopo_assignment_set_report,
        frame_report_offset + frame_idx,
        self.m_mol[frame_idx].time_of_snap,
        report_band, selected_for_report,
        frame_report.identifiability, normalized_gram,
        better_cut_candidate, frame_report.topology_epoch_id,
        frame_report
            .component_soft_score_valid_assignments_within_domain,
        frame_report.spatial_lift_ambiguous_hard_edges,
        frame_report.spatial_lift_cycle_residuals,
        frame_report.status);
  }
  frame_report.time_graph_ms =
      max(0.0, (timing.graph_s + timing.component_graph_s -
                frame_graph_s_before) *
                   1000.0);
  frame_report.time_virtual_ms =
      max(0.0, (timing.event_planes_s + timing.coarse_s +
                timing.refine_s + timing.early_projection_s -
                frame_virtual_s_before) *
                   1000.0);
  frame_report.time_component_ms =
      max(0.0, (timing.component_prep_s + timing.component_graph_s +
                timing.component_search_s + timing.warm_component_s -
                frame_component_s_before) *
                   1000.0);
  frame_report.time_total_ms =
      chrono::duration<double, milli>(clock::now() - frame_report_started)
          .count();
  if (status == PbctopoFrameStatus::WeakObservation)
    ++timing.pbctopo_weak_observation_output_frames;
  if (status == PbctopoFrameStatus::Contradicted)
    ++timing.pbctopo_contradicted_output_frames;
  if ((status == PbctopoFrameStatus::Certified ||
       status == PbctopoFrameStatus::Rescued) &&
      frame_report.geometry_classification.evaluated) {
    if (frame_report.geometry_classification.geometry_class ==
        titan_pbctopo::PbctopoGeometryClass::CompactUnitBox) {
      ++timing.pbctopo_compact_unit_box_certified_frames;
    } else if (frame_report.geometry_classification.geometry_class ==
               titan_pbctopo::PbctopoGeometryClass::Extended) {
      ++timing.pbctopo_noncompact_certified_frames;
      ++timing.pbctopo_extended_certified_frames;
    }
  }
  if ((status == PbctopoFrameStatus::Certified ||
       status == PbctopoFrameStatus::WeakObservation ||
       status == PbctopoFrameStatus::Rescued) &&
      frame_report.split_owner_pairs > 0) {
    ++timing.pbctopo_owner_com_split_observations;
  }
  write_or_defer_frame_report(frame_idx, frame_report);
}

bool PbctopoFrameSolver::accept_rescue_candidate(const vector<Vec3> &candidate_points,
                                                 PbctopoCutCandidate candidate) {
if (!certify_candidate_for_output(candidate, candidate_points)) {
    ++timing.pbctopo_diagnostic_only_candidates;
    record_frame_failure(PbctopoFailureReason::CertificationReject);
return false;
}
  ensure_component_context();
  audit_selected_evidence(candidate_points, candidate.assignment);
  if (!selected_evidence_audit.evaluated) {
    warm_state = frame_start_warm_state;
    reset_kinematic_seed_history(warm_state, timing);
    temporal_cut_candidates[frame_idx].clear();
    report_certified_candidates.clear();
    report_rejected_candidates.clear();
    ++timing.pbctopo_diagnostic_only_candidates;
    record_frame_failure(
        PbctopoFailureReason::EvidenceHypothesisUnavailable);
    return false;
  }
  const bool evidence_contradicted =
      selected_evidence_audit.state ==
      PbctopoEvidenceState::Contradicted;
  const bool evidence_weak =
      selected_evidence_audit.state == PbctopoEvidenceState::WeakDisconnected ||
      pbctopo_evidence_state_underdetermined(selected_evidence_audit.state);
  if (!evidence_contradicted)
    record_temporal_certified_candidate(candidate, candidate_points);
  record_report_soft_candidate(candidate);
  best_safe_candidate = candidate;
  best_candidate = candidate;
  best_safe_points = candidate_points;
  safe_candidates = max<size_t>(safe_candidates, 1);
  projection_clash_skip_active =
      candidate_has_projection_clash_margin(best_safe_candidate,
                                            runtime_cutoffs.clash_cutoff);
  if (evidence_contradicted) {
    warm_state = frame_start_warm_state;
  } else if (!evidence_weak && selected_relative_images_identified()) {
    if (!commit_certified_candidate_to_warm_state(
            best_safe_candidate)) {
      ++timing.pbctopo_diagnostic_only_candidates;
      record_frame_failure(PbctopoFailureReason::CertificationReject);
      return false;
    }
  } else {
    warm_state = frame_start_warm_state;
    ++timing.pbctopo_weak_observation_warm_preserved;
  }
  apply_cut_candidate_timed(best_safe_points);
  return true;
}

} // namespace pbctopo_solve
