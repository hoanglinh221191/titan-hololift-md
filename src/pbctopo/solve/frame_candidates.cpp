#include "pbctopo/solve/solver.h"

#include "Multi_MOL.h"
#include "titan_executor.h"

using namespace std;

namespace pbctopo_solve {

void PbctopoFrameSolver::cache_candidate_evidence_audit(const PbctopoImageAssignment &assignment,
                                                        const PbctopoSelectedEvidenceAudit &audit) {
  if (!evidence_relation_export_enabled ||
      !evidence_relation_cache_error.empty()) {
    return;
  }
  const auto cached =
      upsert_pbctopo_candidate_evidence_audit_snapshot(
          evidence_relation_audit_snapshots, assignment, audit);
  if (!cached)
    evidence_relation_cache_error = cached.error();
}

void PbctopoFrameSolver::record_report_certified_candidate(const PbctopoCutCandidate &candidate) {
  if (!candidate.valid || !candidate.assignment.valid) {
    return;
  }
  if (!candidate.hard_feasibility.certified()) {
    // Export separately, with the failed hard predicates, so a
    // consumer can distinguish a rejection from an unevaluated image.
    const bool known = any_of(
        report_rejected_candidates.begin(),
        report_rejected_candidates.end(),
        [&](const PbctopoCutCandidate &other) {
          return same_pbctopo_image_assignment(other.assignment,
                                                candidate.assignment);
        });
    if (!known)
      report_rejected_candidates.push_back(candidate);
    return;
  }
  auto existing = find_if(
      report_certified_candidates.begin(),
      report_certified_candidates.end(),
      [&](const PbctopoCutCandidate &other) {
        return same_pbctopo_image_assignment(other.assignment,
                                              candidate.assignment);
      });
  if (existing == report_certified_candidates.end()) {
    report_certified_candidates.push_back(candidate);
  } else {
    const PbctopoCutCandidate prior = *existing;
    const bool equivalence_known =
        existing->search_equivalence_known ||
        candidate.search_equivalence_known;
    const bool equivalent_to_search_best =
        (existing->search_equivalence_known &&
         existing->equivalent_to_search_best) ||
        (candidate.search_equivalence_known &&
         candidate.equivalent_to_search_best);
    if (better_cut_candidate(candidate, *existing))
      *existing = candidate;
    existing->search_equivalence_known = equivalence_known;
    existing->equivalent_to_search_best =
        equivalent_to_search_best;
    merge_pbctopo_search_domain_provenance(*existing, prior);
    merge_pbctopo_search_domain_provenance(*existing, candidate);
  }
  sort(report_certified_candidates.begin(),
       report_certified_candidates.end(), better_cut_candidate);
}

auto PbctopoFrameSolver::compute_candidate_evidence(const PbctopoCutCandidate &candidate,
                                                    const vector<Vec3> &points) -> optional<CandidateEvidenceAudit> {
  if (!candidate.assignment.valid || points.size() != base_atoms.size())
    return nullopt;
  return titan_pbctopo::audit_pbctopo_selected_hypotheses(
      span<const PbctopoAtom>(base_atoms), span<const Vec3>(points),
      base_anchor_context.contact_hypotheses, candidate.assignment, true,
      explicit_evidence_component_edges);
}

auto PbctopoFrameSolver::apply_candidate_evidence(PbctopoCutCandidate &candidate,
                                                  const optional<CandidateEvidenceAudit> &computed) -> bool {
  if (!computed || !*computed || !(*computed)->evaluated)
    return false;
  const auto &audit = **computed;
  cache_candidate_evidence_audit(candidate.assignment, audit);
  candidate.evidence_relative_relation_signature =
      candidate.assignment.signature;
  candidate.evidence_compatible_hypothesis_hash =
      audit.selected_hypothesis_hash;
  candidate.evidence_compatible_pair_hash =
      audit.selected_compatible_pair_hash;
  candidate.evidence_compatible_hypothesis_count =
      audit.compatible_hypothesis_count;
  candidate.evidence_supported_relation_count =
      audit.supported_relation_count;
  candidate.evidence_compatible_contact_count =
      audit.selected_compatible_contact_count;
  candidate.evidence_no_support_relation_count =
      audit.no_supported_contact_relation_count;
  return audit.state != PbctopoEvidenceState::Contradicted;
}

auto PbctopoFrameSolver::annotate_candidate_evidence(PbctopoCutCandidate &candidate,
                                                     const vector<Vec3> &points) -> bool {
  return apply_candidate_evidence(
      candidate, compute_candidate_evidence(candidate, points));
}

void PbctopoFrameSolver::compute_candidate_certification(CandidateCertificationWork &work,
                                                         PbctopoCutCandidate &candidate,
                                                         const vector<Vec3> &points,
                                                         const vector<Int3> *known_offsets,
                                                         bool on_worker) {
  const auto lattice_before =
      titan_pbctopo::pbctopo_lattice_query_audit();
  auto finish = [&]() {
    if (!on_worker)
      return;
    work.ran_on_worker = true;
    work.worker_lattice_delta =
        titan_pbctopo::pbctopo_lattice_query_audit_delta(
            lattice_before,
            titan_pbctopo::pbctopo_lattice_query_audit());
  };
  // A worker runs only with the component context built and the
  // offsets given, so these two frame-state writers stay serial.
  if (!on_worker)
    ensure_component_context();
  vector<Int3> extracted_offsets;
  const vector<Int3> *offsets = known_offsets;
  if (!on_worker &&
      (offsets == nullptr || offsets->size() != components.size())) {
    extracted_offsets =
        extract_offsets_and_time(component_atoms, points);
    offsets = &extracted_offsets;
  }
  // Extracted and search offsets are rebased; certify the points in
  // the gauge that placed them, or every atom reads as a residual.
  vector<Int3> aligned_offsets = *offsets;
  if (align_component_offsets_to_points(component_atoms, components,
                                        (*certification_lattice), points,
                                        aligned_offsets)) {
    offsets = &aligned_offsets;
  }
  candidate.assignment = make_pbctopo_image_assignment(
      *offsets, components, component_atoms);
  candidate.hard_feasibility.certificate_atoms_total =
      base_atoms.size();
  candidate.hard_feasibility.hard_graph_atoms_covered =
      hard_graph_atoms_covered;
  candidate.hard_feasibility.topology_bond_atoms_covered =
      topology_bond_atoms_covered;
  candidate.hard_feasibility.metadata_only_atoms =
      metadata_only_atoms;
  candidate.hard_feasibility.hard_connectivity_edges_available =
      hard_connectivity_edge_count;
  candidate.hard_feasibility.topology_edges_available =
      topology_edge_count;
  candidate.hard_feasibility.metadata_edges_available =
      metadata_edge_count;
  if (!candidate.assignment.valid) {
    candidate.hard_feasibility = {};
    candidate.hard_feasibility.evaluated = candidate.valid;
    finish();
    return;
  }
  work.assignment_valid = true;
  // A worker reads the cache only if the frame's thread built it before.
  const titan_pbctopo::PbctopoEdgeImageCache *edge_cache =
      on_worker ? (edge_image_cache.has_value() ? &*edge_image_cache : nullptr)
                : ensure_edge_image_cache();
  work.image_equations = evaluate_pbctopo_image_equations(
      base_atoms, points, local_edges, (*certification_lattice), 1.0e-6,
      edge_cache);
  work.certification_lattice_delta =
      titan_pbctopo::pbctopo_lattice_query_audit_delta(
          lattice_before, titan_pbctopo::pbctopo_lattice_query_audit());
  if (work.certification_lattice_delta.failures > 0) {
    work.lattice_failure = true;
    candidate.hard_feasibility = {};
    candidate.hard_feasibility.evaluated = candidate.valid;
    finish();
    return;
  }
  work.assignment_points =
      titan_pbctopo::evaluate_pbctopo_assignment_point_consistency(
          component_atoms, points, atom_component_index,
          candidate.assignment, (*certification_lattice));
  PbctopoTimingStats steric_timing;
  work.steric_audit = titan_pbctopo::evaluate_pbctopo_final_sterics(
      base_atoms, points, atom_component_index, steric_exclusion_set,
      runtime_cutoffs.clash_cutoff, &steric_timing,
      &(*certification_lattice));
  work.vdw_pair_checks = steric_timing.pbctopo_vdw_pair_checks;
  work.vdw_radius_pairs = steric_timing.pbctopo_vdw_radius_pairs;
  work.vdw_radius_fallback_pairs =
      steric_timing.pbctopo_vdw_radius_fallback_pairs;
  work.vdw_explicit_zero_lj_exempt_pairs =
      steric_timing.pbctopo_vdw_explicit_zero_lj_exempt_pairs;
  const auto &steric_audit = work.steric_audit;
  candidate.ultra_clash = steric_audit.hard_clashes != 0;
  // The candidate's min_pair_d2 and clash atoms describe its closest
  // pair (analyze_cut_candidate_ultraclash sets them that way), so
  // they are refreshed from the audit's closest pair, not from the
  // pair with the smallest margin.  The margin itself is carried in
  // hard_feasibility.min_steric_margin_A.
  if (isfinite(steric_audit.min_pair_d2) &&
      steric_audit.min_pair_d2 < candidate.min_pair_d2) {
    candidate.min_pair_d2 = steric_audit.min_pair_d2;
    candidate.clash_atom_a = steric_audit.min_pair_atom_a;
    candidate.clash_atom_b = steric_audit.min_pair_atom_b;
  }
  work.required_contacts =
      titan_pbctopo::evaluate_pbctopo_required_contacts(
          base_atoms, points, local_edges, explicit_required_contacts);
  candidate.geometry_classification =
      titan_pbctopo::classify_pbctopo_final_geometry(
          points, (*certification_lattice), 1.0e-6,
          &work.image_equations);
  work.certified = certify_pbctopo_candidate(
      candidate, work.image_equations, work.assignment_points,
      work.steric_audit, work.required_contacts);
  if (work.certified)
    work.evidence = compute_candidate_evidence(candidate, points);
  finish();
}

bool PbctopoFrameSolver::apply_candidate_certification(const CandidateCertificationWork &work,
                                                       PbctopoCutCandidate &candidate) {
  if (work.ran_on_worker)
    titan_pbctopo::pbctopo_add_lattice_query_audit(
        work.worker_lattice_delta);
  ++timing.pbctopo_certification_attempts;
  if (!work.assignment_valid) {
    ++timing.pbctopo_certification_assignment_rejects;
    return false;
  }
  frame_fatal_lattice_query_failures +=
      work.certification_lattice_delta.failures;
  if (work.lattice_failure) {
    ++timing.pbctopo_certification_hard_rejects;
    return false;
  }
  timing.pbctopo_vdw_pair_checks += work.vdw_pair_checks;
  timing.pbctopo_vdw_radius_pairs += work.vdw_radius_pairs;
  timing.pbctopo_vdw_radius_fallback_pairs +=
      work.vdw_radius_fallback_pairs;
  timing.pbctopo_vdw_explicit_zero_lj_exempt_pairs +=
      work.vdw_explicit_zero_lj_exempt_pairs;
  ++timing.pbctopo_geometry_classifications;
  if (candidate.geometry_classification.topological_winding_rank > 0) {
    ++timing.pbctopo_winding_rejected_candidates;
    if (candidate.geometry_classification.geometry_class ==
        titan_pbctopo::PbctopoGeometryClass::ExtendedAndWinding) {
      ++timing.pbctopo_extended_winding_rejected_candidates;
    }
  }
  const auto &image_equations = work.image_equations;
  timing.pbctopo_certification_topology_edges_checked +=
      image_equations.topology_edges_checked;
  timing.pbctopo_certification_ambiguous_topology_edges +=
      image_equations.ambiguous_topology_edges;
  timing.pbctopo_certification_topology_edge_residuals +=
      image_equations.topology_edge_residuals;
  timing.pbctopo_certification_topology_cycle_residuals +=
      image_equations.topology_cycle_residuals;
  timing.pbctopo_certification_atom_image_residuals +=
      image_equations.atom_image_residuals;
  timing.pbctopo_certification_assignment_point_residuals +=
      work.assignment_points.point_residuals;
  timing.pbctopo_certification_steric_pair_checks +=
      work.steric_audit.pair_checks;
  timing.pbctopo_certification_steric_clashes +=
      work.steric_audit.hard_clashes;
  timing.pbctopo_certification_required_contacts_checked +=
      work.required_contacts.contacts_checked;
  timing.pbctopo_certification_required_contacts_lost +=
      work.required_contacts.contacts_lost;
  timing.pbctopo_certification_explicit_interface_contacts_checked +=
      work.required_contacts.explicit_contacts_checked;
  timing.pbctopo_certification_explicit_interface_contacts_lost +=
      work.required_contacts.explicit_contacts_lost;
  if (!work.certified) {
    ++timing.pbctopo_certification_hard_rejects;
    // Preserve the rejection in the separate audit ledger.
    record_report_certified_candidate(candidate);
    return false;
  }
  ++timing.pbctopo_certification_successes;
  if (apply_candidate_evidence(candidate, work.evidence))
    record_report_certified_candidate(candidate);
  return true;
}

bool PbctopoFrameSolver::certify_candidate_for_output(PbctopoCutCandidate &candidate,
                                                      const vector<Vec3> &points,
                                                      const vector<Int3> *known_offsets) {
  CandidateCertificationWork work;
  compute_candidate_certification(work, candidate, points,
                                  known_offsets, false);
  return apply_candidate_certification(work, candidate);
}

void PbctopoFrameSolver::record_temporal_certified_candidate(const PbctopoCutCandidate &source_candidate,
                                                             const vector<Vec3> &candidate_points,
                                                             const vector<Int3> *known_offsets) {
  if (temporal_mode != PbctopoTemporalMode::Viterbi ||
      !source_candidate.valid || source_candidate.ultra_clash ||
      candidate_points.size() != base_atoms.size()) {
    return;
  }
  PbctopoCutCandidate candidate = source_candidate;
  if (!candidate.hard_feasibility.certified() ||
      !candidate.assignment.valid) {
    if (!certify_candidate_for_output(candidate, candidate_points,
                                      known_offsets)) {
      return;
    }
  }
  if (!annotate_candidate_evidence(candidate, candidate_points))
    return;
  if (candidate.evidence_relative_relation_signature !=
      candidate.assignment.signature)
    return;
  record_report_certified_candidate(candidate);
  auto &frame_candidates = temporal_cut_candidates[frame_idx];
  const auto existing = find_if(
      frame_candidates.begin(), frame_candidates.end(),
      [&](const PbctopoCutCandidate &other) {
        return same_pbctopo_image_assignment(other.assignment,
                                              candidate.assignment);
      });
  if (existing != frame_candidates.end()) {
    // The first copy keeps its scores, which feed the Viterbi costs,
    // but the search provenance of a later copy of the same
    // assignment is merged as the band and report ledger do. The
    // exported search domain is frame-constant, so a virtual copy
    // recorded before the search's must not hide the search audit.
    const bool equivalence_known =
        existing->search_equivalence_known ||
        candidate.search_equivalence_known;
    const bool equivalent_to_search_best =
        (existing->search_equivalence_known &&
         existing->equivalent_to_search_best) ||
        (candidate.search_equivalence_known &&
         candidate.equivalent_to_search_best);
    existing->search_equivalence_known = equivalence_known;
    existing->equivalent_to_search_best = equivalent_to_search_best;
    merge_pbctopo_search_domain_provenance(*existing, candidate);
    ++timing.pbctopo_temporal_assignment_deduped;
    return;
  }
  frame_candidates.push_back(std::move(candidate));
  sort(frame_candidates.begin(), frame_candidates.end(),
       better_cut_candidate);
  const size_t keep =
      static_cast<size_t>(Arguments.pbc.pbctopo_temporal_k);
  if (frame_candidates.size() > keep)
    frame_candidates.resize(keep);
  ++timing.pbctopo_temporal_certified_candidates;
}

void PbctopoFrameSolver::apply_cut_candidate_timed(const vector<Vec3> &points) {
  const auto apply_started = clock::now();
  apply_cut_candidate(base_atoms, points, self, mol2, frame_idx);
  ++timing.apply_calls;
  timing.apply_s +=
      chrono::duration<double>(clock::now() - apply_started).count();
}

} // namespace pbctopo_solve
