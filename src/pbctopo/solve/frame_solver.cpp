#include "pbctopo/solve/solver.h"

#include "Multi_MOL.h"
#include "titan_executor.h"

using namespace std;

namespace pbctopo_solve {

PbctopoFrameSolver::PbctopoFrameSolver(PbctopoSolveRun &run_, size_t frame_idx_)
    : run(run_), self(run_.self), mol2(run_.mol2), Arguments(run_.Arguments),
      executor(run_.executor), frame_report_offset(run_.frame_report_offset),
      frame_count(run_.frame_count),
      pbctopo_assignment_set_report(run_.pbctopo_assignment_set_report),
      pbctopo_rejected_assignment_report(run_.pbctopo_rejected_assignment_report),
      pbctopo_candidate_evidence_summary_report(run_.pbctopo_candidate_evidence_summary_report),
      pbctopo_candidate_evidence_relation_report(run_.pbctopo_candidate_evidence_relation_report),
      pbctopo_component_layout_report(run_.pbctopo_component_layout_report),
      pbctopo_hard_edge_report(run_.pbctopo_hard_edge_report),
      h9_orbit_enabled(run_.h9_orbit_enabled),
      h9_component_representative_report(run_.h9_component_representative_report),
      h9_evidence_component_report(run_.h9_evidence_component_report),
      h9_orbit_report(run_.h9_orbit_report),
      emitted_component_layouts(run_.emitted_component_layouts),
      fallback_frames(run_.fallback_frames),
      valid_candidates(run_.valid_candidates),
      timing(run_.timing),
      warm_state(run_.warm_state),
      runtime_cutoffs(run_.runtime_cutoffs),
      solve_mode(run_.solve_mode),
      audit_exact_search(run_.audit_exact_search),
      audit_exact_limits(run_.audit_exact_limits),
      anchor_backend(run_.anchor_backend),
      anchor_metric_mode(run_.anchor_metric_mode),
      temporal_mode(run_.temporal_mode),
      exact_component_limit(run_.exact_component_limit),
      beam_width(run_.beam_width),
      component_shell_radius_explicit(run_.component_shell_radius_explicit),
      configured_component_shell_radius(run_.configured_component_shell_radius),
      trajectory_context(run_.trajectory_context),
      explicit_required_contacts(run_.explicit_required_contacts),
      temporal_cut_candidates(run_.temporal_cut_candidates),
      greedy_selected_candidates(run_.greedy_selected_candidates),
      temporal_component_layouts(run_.temporal_component_layouts),
      temporal_frame_layout(run_.temporal_frame_layout),
      temporal_contact_hypotheses(run_.temporal_contact_hypotheses),
      evidence_relation_export_groups(run_.evidence_relation_export_groups),
      evidence_relation_export_rows(run_.evidence_relation_export_rows),
      evidence_relation_export_duplicates(run_.evidence_relation_export_duplicates),
      evidence_relation_export_failures(run_.evidence_relation_export_failures),
      scratch(run_.scratch),
      hull_scratch(run_.hull_scratch),
      base_atoms(run_.base_atoms),
      projection_atoms(run_.projection_atoms),
      component_atoms(run_.component_atoms),
      local_edges(run_.local_edges),
      component_scoring_edges(run_.component_scoring_edges),
      projection_edges(run_.projection_edges),
      projection_continuity_pairs(run_.projection_continuity_pairs),
      projection_anchor_indices(run_.projection_anchor_indices),
      local_graph_atom_coverage(run_.local_graph_atom_coverage),
      local_edge_set(run_.local_edge_set),
      steric_exclusion_set(run_.steric_exclusion_set),
      projection_edge_set(run_.projection_edge_set),
      components(run_.components),
      shifts(run_.shifts),
      volume_seeds(run_.volume_seeds),
      warm_points(run_.warm_points),
      coarse_warm_points(run_.coarse_warm_points),
      placed_points(run_.placed_points),
      best_safe_points(run_.best_safe_points),
      verify_points(run_.verify_points),
      component_points(run_.component_points),
      coarse_candidates(run_.coarse_candidates),
      coarse_seed_options(run_.coarse_seed_options),
      coarse_results(run_.coarse_results),
      projection_index(run_.projection_index),
      atom_component_index(run_.atom_component_index),
      virtual_search_workspace(run_.virtual_search_workspace),
      virtual_search_plan(run_.virtual_search_plan),
      hull_owner_spans(run_.hull_owner_spans),
      hull_rescue_points(run_.hull_rescue_points),
      projection_clash_skip_active(run_.projection_clash_skip_active),
      llps_local_unit_source(run_.llps_local_unit_source),
      frame_idx(frame_idx_) {}

void PbctopoFrameSolver::solve() {
  if (!prepare_frame())
    return;
  classify_frame();
  prepare_component_context();
  collect_seeds();
  search_virtual();
  search_components();
  if (!certify_and_commit())
    return;
}

} // namespace pbctopo_solve
