#include "pbctopo/solve/solver.h"

#include "Multi_MOL.h"
#include "titan_executor.h"

using namespace std;

namespace pbctopo_solve {

PbctopoSolveRun::PbctopoSolveRun(Multi_MOL &self_, Multi_MOL &mol2_, arguments &Arguments_,
                                 TitanExecutor *executor_, size_t frame_report_offset_)
    : self(self_), mol2(mol2_), Arguments(Arguments_), executor(executor_),
      frame_report_offset(frame_report_offset_),
      emitted_component_layouts(pbctopo_component_layout_report_state().emitted_layouts),
      base_atoms(scratch.base_atoms),
      projection_atoms(scratch.projection_atoms),
      component_atoms(scratch.component_atoms),
      local_edges(scratch.local_edges),
      component_scoring_edges(scratch.component_scoring_edges),
      projection_edges(scratch.projection_edges),
      projection_continuity_pairs(scratch.projection_continuity_pairs),
      projection_anchor_indices(scratch.projection_anchor_indices),
      local_graph_atom_coverage(scratch.local_graph_atom_coverage),
      local_edge_set(scratch.local_edge_set),
      steric_exclusion_set(scratch.steric_exclusion_set),
      projection_edge_set(scratch.projection_edge_set),
      components(scratch.components),
      shifts(scratch.shifts),
      volume_seeds(scratch.volume_seeds),
      warm_points(scratch.warm_points),
      coarse_warm_points(scratch.coarse_warm_points),
      placed_points(scratch.placed_points),
      best_safe_points(scratch.best_safe_points),
      verify_points(scratch.verify_points),
      component_points(scratch.component_points),
      coarse_candidates(scratch.coarse_candidates),
      coarse_seed_options(scratch.coarse_seed_options),
      coarse_results(scratch.coarse_results),
      projection_index(scratch.projection_index),
      atom_component_index(scratch.atom_component_index),
      virtual_search_workspace(scratch.virtual_search_workspace),
      virtual_search_plan(scratch.virtual_search_plan) {}

void PbctopoSolveRun::reset_uncommitted_warm_state() {
  reset_kinematic_seed_history(warm_state, timing);
  reset_pbctopo_warm_state(warm_state);
  ++timing.pbctopo_warm_state_resets;
}

void PbctopoSolveRun::write_or_defer_frame_report(size_t frame_idx,
                                                  PbctopoFrameReport report) {
  if (frame_idx < self.pbctopo_frame_status_.size()) {
    self.pbctopo_frame_status_[frame_idx] =
        titan_pbctopo::pbctopo_frame_status_name(report.status);
    self.pbctopo_frame_certified_[frame_idx] =
        report.status == PbctopoFrameStatus::Certified ? 1 : 0;
  }
  if (report.component_audit_exact_requested &&
      !report.component_audit_exact_completed &&
      !report.component_audit_exact_incomplete) {
    report.component_audit_exact_incomplete = true;
    report.component_audit_exact_incomplete_reason = "not_attempted";
    ++timing.component_audit_exact_incomplete;
    append_pbctopo_log_line(
        "[pbctopo] audit-exact incomplete at frame " +
        to_string(report.frame) +
        ": reason=not_attempted, action=fail_closed");
  }
  if (temporal_mode == PbctopoTemporalMode::Viterbi) {
    deferred_frame_reports[frame_idx] = report;
    deferred_frame_report_valid[frame_idx] = 1;
  } else if (pbctopo_frame_report.is_open()) {
    write_pbctopo_frame_report_row(pbctopo_frame_report, report);
  }
}

void PbctopoSolveRun::fallback_pbci_frame(size_t frame_index) {
  ++timing.pbci_fallback_frames;
  int chain_ref = Arguments.pbc.chain_ref;
  const size_t chain_count = self.Index.empty() ? 0 : self.Index.size() - 1;
  if (chain_count == 0) {
    chain_ref = 0;
  } else if (chain_ref < 0 ||
             static_cast<size_t>(chain_ref) >= chain_count) {
    chain_ref = 0;
  }

  self.find_best_image_envelope(frame_index, chain_ref, Arguments);
  if (frame_index < mol2.m_mol.size() && !mol2.m_mol[frame_index].pos.x.empty())
    self.find_best_image_of_mol2(mol2, frame_index, chain_ref, Arguments);

  vector<PbctopoAtom> fallback_atoms;
  fallback_atoms.reserve(trajectory_context.total_atoms_hint);
  append_atoms_for_system(self.m_mol[frame_index], &self.info,
                          &trajectory_context.rec_static, 0, fallback_atoms);
  if (frame_index < mol2.m_mol.size() && !mol2.m_mol[frame_index].pos.x.empty()) {
    append_atoms_for_system(mol2.m_mol[frame_index], &mol2.info,
                            &trajectory_context.lig_static, 1, fallback_atoms);
  }
  prepare_pbctopo_steric_radii(fallback_atoms,
                               runtime_cutoffs.vdw_clash_scale);
  vector<pair<int, int>> fallback_contacting_owner_pairs =
      build_contacting_owner_pairs({}, fallback_atoms,
                                   self.m_mol[frame_index].Box,
                                   runtime_cutoffs.pair_contact_cutoff,
                                   false);
  vector<Vec3> fallback_points;
  fallback_points.reserve(fallback_atoms.size());
  for (const auto &atom : fallback_atoms)
    fallback_points.push_back({atom.x, atom.y, atom.z});
  if (!reimage_contacting_owner_points_minimum_image(
          fallback_points, fallback_atoms, self.m_mol[frame_index].Box,
          fallback_contacting_owner_pairs)) {
    return;
  }
  for (size_t atom_idx = 0; atom_idx < fallback_atoms.size(); ++atom_idx) {
    const auto &atom = fallback_atoms[atom_idx];
    const auto &point = fallback_points[atom_idx];
    if (atom.owner == 0) {
      self.m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
      self.m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
      self.m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
    } else if (frame_index < mol2.m_mol.size()) {
      mol2.m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
      mol2.m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
      mol2.m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
    }
  }
  append_pbctopo_log_line(
      "[pbctopo] fallback pbci=0 reimaged contacting owners at frame " +
      to_string(static_cast<long long>(self.m_mol[frame_index].time_of_snap)));
}

std::optional<int> PbctopoSolveRun::setup() {
  frame_count =
      mol2.m_mol.empty() ? self.m_mol.size()
                         : min(self.m_mol.size(), mol2.m_mol.size());
  if (frame_count == 0)
    return 0;

  self.pbctopo_frame_certified_.assign(frame_count, 0);
  self.pbctopo_frame_status_.assign(frame_count, "missing_status");

  self.cache_topology_vdw_radii_from_parameters();
  mol2.cache_topology_vdw_radii_from_parameters();

  reset_pbctopo_log_file();
  pbctopo_frame_report = open_pbctopo_frame_report();
  pbctopo_assignment_set_report =
      open_pbctopo_assignment_set_report();
  pbctopo_rejected_assignment_report =
      open_pbctopo_rejected_assignment_report();
  if (Arguments.pbc.pbctopo_evidence_export) {
    pbctopo_candidate_evidence_summary_report =
        open_pbctopo_candidate_evidence_summary_report();
    pbctopo_candidate_evidence_relation_report =
        open_pbctopo_candidate_evidence_relation_report();
  }
  pbctopo_component_layout_report =
      open_pbctopo_component_layout_report();
  pbctopo_hard_edge_report = open_pbctopo_hard_edge_report();
  h9_orbit_enabled = Arguments.pbc.pbctopo_h9_orbit;
  if (h9_orbit_enabled) {
    h9_component_representative_report =
        open_pbctopo_h9_component_representative_report();
    h9_evidence_component_report =
        open_pbctopo_h9_evidence_component_report();
    h9_orbit_report = open_pbctopo_h9_orbit_report();
  }
  if (!pbctopo_frame_report.is_open()) {
    append_pbctopo_log_line(
        "[pbctopo] observation export unavailable: "
        "artifact=pbctopo_frame_report.csv");
  }
  if (!pbctopo_assignment_set_report.is_open()) {
    append_pbctopo_log_line(
        "[pbctopo] observation export unavailable: "
        "artifact=pbctopo_assignment_candidates.csv");
  }
  if (!pbctopo_rejected_assignment_report.is_open()) {
    append_pbctopo_log_line(
        "[pbctopo] rejected-assignment audit unavailable: "
        "artifact=pbctopo_rejected_candidates.csv");
  }
  if (Arguments.pbc.pbctopo_evidence_export &&
      (!pbctopo_candidate_evidence_summary_report.is_open() ||
       !pbctopo_candidate_evidence_relation_report.is_open())) {
    append_pbctopo_log_line(
        "[pbctopo] observation export unavailable: "
        "artifacts=pbctopo_candidate_evidence_summary.csv|"
        "pbctopo_candidate_evidence_relations.csv");
  }
  if (!pbctopo_component_layout_report.is_open()) {
    append_pbctopo_log_line(
        "[pbctopo] observation export unavailable: "
        "artifact=pbctopo_component_layout.csv");
  }
  if (!pbctopo_hard_edge_report.is_open()) {
    append_pbctopo_log_line(
        "[pbctopo] observation export unavailable: "
        "artifact=pbctopo_hard_edges.csv");
  }
  if (h9_orbit_enabled &&
      (!h9_component_representative_report.is_open() ||
       !h9_evidence_component_report.is_open() ||
       !h9_orbit_report.is_open())) {
    append_pbctopo_log_line(
        "[pbctopo] H9 export unavailable: one or more artifacts could not be "
        "opened");
  }
  fallback_frames = 0;
  valid_candidates = 0;
  runtime_cutoffs =
      resolve_pbctopo_cutoffs(Arguments);
  solve_mode = resolve_pbctopo_solve_mode(Arguments);
  search_policy =
      titan_pbctopo::resolve_search_policy(Arguments);
  audit_exact_search =
      search_policy == PbctopoSearchPolicy::AuditExact;
  audit_memory_mib =
      static_cast<size_t>(Arguments.pbc.pbctopo_audit_max_memory_mb);
  audit_memory_bytes =
      audit_memory_mib > numeric_limits<size_t>::max() / kBytesPerMiB
          ? numeric_limits<size_t>::max()
          : audit_memory_mib * kBytesPerMiB;
  audit_exact_limits = PbctopoComponentSearchLimits{
      static_cast<size_t>(Arguments.pbc.pbctopo_audit_max_states),
      Arguments.pbc.pbctopo_audit_max_seconds, audit_memory_bytes, true};
  anchor_backend =
      resolve_pbctopo_anchor_backend(Arguments);
  anchor_metric_mode =
      titan_pbctopo::resolve_anchor_metric_mode(Arguments);
  temporal_mode =
      titan_pbctopo::resolve_temporal_mode(Arguments);
  requested_exact_component_limit =
      Arguments.pbc.pbctopo_exact_component_limit_set
          ? static_cast<size_t>(Arguments.pbc.pbctopo_exact_component_limit)
          : kPbctopoExactComponentLimit;
  exact_component_limit =
      min(requested_exact_component_limit, kPbctopoExactComponentLimitMax);
  requested_beam_width =
      Arguments.pbc.pbctopo_beam_width_set
          ? static_cast<size_t>(Arguments.pbc.pbctopo_beam_width)
          : kPbctopoBeamWidth;
  beam_width =
      max<size_t>(1, min(requested_beam_width, kPbctopoBeamWidthMax));
  component_shell_radius_explicit =
      Arguments.pbc.pbctopo_component_shell_radius_set;
  configured_component_shell_radius =
      component_shell_radius_explicit
          ? Arguments.pbc.pbctopo_component_shell_radius
          : 0;
  if (h9_orbit_enabled &&
      (!component_shell_radius_explicit ||
       configured_component_shell_radius != 2)) {
    TitanUI::print_error(
        "PBCTOPO H9 orbit requires -pbctopo-component-shell-radius=2.");
    append_pbctopo_log_line(
        "[pbctopo] H9 orbit rejected: component shell radius must be "
        "explicitly fixed at 2");
    return 1;
  }
  append_pbctopo_log_line("[pbctopo] begin run: frames=" +
                          to_string(frame_count) + ", pbctopo_mode=" +
                          pbctopo_solve_mode_name(solve_mode) +
                          ", search_policy=" +
                          titan_pbctopo::search_policy_name(search_policy) +
                          ", observation_schema_version=" +
                          to_string(titan_pbctopo::
                                        VIBE_OBSERVATION_SCHEMA_VERSION) +
                          ", observation_scope=" +
                          string(titan_pbctopo::VIBE_OBSERVATION_SCOPE) +
                          ", anchor_backend=" +
                          pbctopo_anchor_backend_name(anchor_backend) +
                          ", anchor_metrics=" +
                          titan_pbctopo::anchor_metric_mode_name(
                              anchor_metric_mode) +
                          ", temporal=" +
                          titan_pbctopo::temporal_mode_name(temporal_mode) +
                          ", temporal_status=" +
                          string(temporal_mode == PbctopoTemporalMode::Viterbi
                                     ? "experimental_diagnostic"
                                     : "production_framewise") +
                          ", vdw_clash_scale=" +
                          to_string(runtime_cutoffs.vdw_clash_scale) +
                          ", bond_cutoff=" +
                          to_string(runtime_cutoffs.bond_cutoff) +
                          ", clash_cutoff=" +
                          to_string(runtime_cutoffs.clash_cutoff) +
                          ", contact_cutoff=" +
                          to_string(runtime_cutoffs.pair_contact_cutoff) +
                          ", continuity_cutoff=" +
                          to_string(runtime_cutoffs.continuity_cutoff) +
                          ", exact_component_limit=" +
                          to_string(exact_component_limit) + ", beam_width=" +
                          to_string(beam_width) +
                          ", component_shell_radius=" +
                          (component_shell_radius_explicit
                               ? to_string(configured_component_shell_radius)
                               : string("auto")));
  trajectory_context =
      build_pbctopo_trajectory_context(self, mol2, frame_count);
  if (!Arguments.pbc.pbctopo_required_contacts_file.empty()) {
    const size_t selected_atom_count =
        trajectory_context.rec_static.atom_count +
        trajectory_context.lig_static.atom_count;
    auto loaded_contacts = titan_pbctopo::load_pbctopo_required_contacts(
        Arguments.pbc.pbctopo_required_contacts_file, selected_atom_count);
    if (!loaded_contacts) {
      append_pbctopo_log_line("[pbctopo] required interface contact error: " +
                              loaded_contacts.error());
      TitanUI::print_error(loaded_contacts.error());
      return 1;
    }
    explicit_required_contacts = std::move(*loaded_contacts);
    timing.pbctopo_required_interface_contacts_loaded =
        explicit_required_contacts.size();
    append_pbctopo_log_line(
        "[pbctopo] required interface contacts loaded: source=user_file" +
        string(", file=") + Arguments.pbc.pbctopo_required_contacts_file +
        ", contacts=" + to_string(explicit_required_contacts.size()) +
        ", atom_indexing=selected_1_based");
  } else {
    append_pbctopo_log_line(
        "[pbctopo] required interface contacts: source=none, "
        "certificate_scope=image_intercomponent_steric");
  }
  temporal_cut_candidates = vector<vector<PbctopoCutCandidate>>(frame_count);
  greedy_selected_candidates = vector<PbctopoCutCandidate>(frame_count);
  temporal_frame_layout = vector<size_t>(
      frame_count, numeric_limits<size_t>::max());
  deferred_frame_reports = vector<PbctopoFrameReport>(frame_count);
  deferred_frame_report_valid = vector<unsigned char>(frame_count, 0);
  temporal_contact_hypotheses = vector<PbctopoContactHypothesisSet>(frame_count);
  evidence_relation_export_groups = 0;
  evidence_relation_export_rows = 0;
  evidence_relation_export_duplicates = 0;
  evidence_relation_export_failures = 0;
  has_ligand_static = trajectory_context.lig_static.atom_count > 0;
  static_topology_ready =
      trajectory_context.rec_static.topology_graph_available &&
      (!has_ligand_static ||
       trajectory_context.lig_static.topology_graph_available);
  static_metadata_ready =
      !static_topology_ready &&
      trajectory_context.rec_static.metadata_graph_available &&
      (!has_ligand_static ||
       trajectory_context.lig_static.metadata_graph_available);
  static_graph_source =
      static_topology_ready
          ? "gromacs_topology"
          : (static_metadata_ready ? "metadata_chain_residue"
                                   : "geometry_cutoff");
  append_pbctopo_log_line(
      "[pbctopo] local graph prepared: source=" + static_graph_source +
      ", selected_atoms=" +
      to_string(trajectory_context.rec_static.atom_count +
                trajectory_context.lig_static.atom_count) +
      ", topology_bonds=" +
      to_string(trajectory_context.rec_static.topology_bonds_available +
                trajectory_context.lig_static.topology_bonds_available) +
      ", projected_topology_bonds=" +
      to_string(
          trajectory_context.rec_static.projected_topology_bonds.size() +
          trajectory_context.lig_static.projected_topology_bonds.size()) +
      ", metadata_edges=" +
      to_string(trajectory_context.rec_static.projected_metadata_edges.size() +
                trajectory_context.lig_static.projected_metadata_edges.size()) +
      ", vdw_topology_radii=" +
      to_string(trajectory_context.rec_static.vdw_topology_count +
                trajectory_context.lig_static.vdw_topology_count) +
      ", vdw_parameter_radii=" +
      to_string(trajectory_context.rec_static.vdw_parameter_count +
                trajectory_context.lig_static.vdw_parameter_count) +
      ", vdw_zero_lj_radii=" +
      to_string(trajectory_context.rec_static.vdw_zero_lj_count +
                trajectory_context.lig_static.vdw_zero_lj_count) +
      ", vdw_inferred_radii=" +
      to_string(trajectory_context.rec_static.vdw_inferred_count +
                trajectory_context.lig_static.vdw_inferred_count) +
      ", vdw_generic_fallback_radii=" +
      to_string(trajectory_context.rec_static.vdw_generic_fallback_count +
                trajectory_context.lig_static.vdw_generic_fallback_count));
  scratch.reserve_from_context(trajectory_context);
  hull_scratch.reserve(trajectory_context.total_atoms_hint);
  hull_owner_spans.reserve(8);
  hull_rescue_points.reserve(trajectory_context.total_atoms_hint);
  projection_clash_skip_active = false;
  llps_local_unit_source = "none";
  return std::nullopt;
}

int PbctopoSolveRun::run() {
  if (const std::optional<int> early = setup())
    return *early;
  for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
    PbctopoFrameSolver frame(*this, frame_idx);
    frame.solve();
  }
  run_temporal();
  return finish();
}

} // namespace pbctopo_solve
