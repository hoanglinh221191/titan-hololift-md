#include "pbctopo/solve/solver.h"

#include "pbctopo/anchor_mst.h"

#include "Multi_MOL.h"
#include "titan_executor.h"

using namespace std;

namespace pbctopo_solve {

int PbctopoSolveRun::finish() {

  append_pbctopo_log_line(
      "[pbctopo evidence relation export] requested=" +
      string(Arguments.pbc.pbctopo_evidence_export ? "1" : "0") +
      ", enabled=" +
      string(pbctopo_candidate_evidence_summary_report.is_open() &&
                     pbctopo_candidate_evidence_relation_report.is_open()
                 ? "1"
                 : "0") +
      ", candidate_groups=" +
      to_string(evidence_relation_export_groups) + ", relation_rows=" +
      to_string(evidence_relation_export_rows) +
      ", duplicate_candidates_skipped=" +
      to_string(evidence_relation_export_duplicates) + ", failures=" +
      to_string(evidence_relation_export_failures));

  timing.pbctopo_frame_point_buffer_peak_bytes_estimate =
      max(timing.pbctopo_frame_point_buffer_peak_bytes_estimate,
          scratch.point_capacity_bytes_estimate());
  if (fallback_frames > 0) {
    const string fallback_summary = "[pbctopo] Fell back to pbci=0 on " +
                                    to_string(fallback_frames) + " frame(s).";
    append_pbctopo_log_line(fallback_summary);
    TitanUI::print_warning(fallback_summary);
  }
  if (timing.pbctopo_lattice_invalid_frames > 0) {
    const string invalid_lattice_summary =
        "[pbctopo] Left " +
        to_string(timing.pbctopo_lattice_invalid_frames) +
        " frame(s) unchanged because their periodic lattice was invalid.";
    append_pbctopo_log_line(invalid_lattice_summary);
    TitanUI::print_warning(invalid_lattice_summary);
  }
  append_pbctopo_log_line(
      "[pbctopo timing] frames=" + to_string(timing.frames) +
      ", append=" + to_string(timing.append_s) + " s, heavy_subset=" +
      to_string(timing.heavy_subset_s) + " s, translation_cache=" +
      to_string(timing.translation_cache_s) + " s, graph=" +
      to_string(timing.graph_s) + " s, pbctopo_local_graph_geometry_s=" +
      to_string(timing.pbctopo_local_graph_geometry_s) +
      " s, pbctopo_topology_edge_append_s=" +
      to_string(timing.pbctopo_topology_edge_append_s) +
      " s, projection_edges=" +
      to_string(timing.projection_edge_s) + " s, component_split=" +
      to_string(timing.component_split_s) + " s, seeds=" +
      to_string(timing.seed_s) + " s, event_planes=" +
      to_string(timing.event_planes_s) +
      " s, pbctopo_density_cut_plane_build=" +
      to_string(timing.pbctopo_density_cut_plane_build_s) +
      " s, pbctopo_virtual_seam_sweep=" +
      to_string(timing.pbctopo_virtual_seam_sweep_s) + " s, coarse=" +
      to_string(timing.coarse_s) + " s, refine=" +
      to_string(timing.refine_s) + " s, early_projection=" +
      to_string(timing.early_projection_s) + " s, warm_component=" +
      to_string(timing.warm_component_s) + " s, component_prep=" +
      to_string(timing.component_prep_s) +
      " s, component_context=" + to_string(timing.component_context_s) +
      " s, component_graph=" + to_string(timing.component_graph_s) +
      " s, component_graph_precompute=" +
      to_string(timing.component_graph_precompute_s) +
      " s, component_graph_lazy_materialize=" +
      to_string(timing.component_graph_lazy_materialize_s) +
      " s, component_graph_spatial_shortlist=" +
      to_string(timing.component_graph_spatial_shortlist_s) +
      " s, component_graph_pair_score=" +
      to_string(timing.component_graph_pair_score_s) +
      " s, component_graph_tree=" +
      to_string(timing.component_graph_tree_s) +
      " s, component_graph_refine=" +
      to_string(timing.component_graph_refine_s) +
      " s, component_graph_refine_rebase=" +
      to_string(timing.component_graph_refine_rebase_s) +
      " s, component_graph_refine_coarse=" +
      to_string(timing.component_graph_refine_coarse_s) +
      " s, component_graph_refine_coarse_stage1=" +
      to_string(timing.component_graph_refine_coarse_stage1_s) +
      " s, component_graph_refine_coarse_stage1_build_points=" +
      to_string(timing.component_graph_refine_coarse_stage1_build_points_s) +
      " s, component_graph_refine_coarse_stage1_continuity=" +
      to_string(timing.component_graph_refine_coarse_stage1_continuity_s) +
      " s, component_graph_refine_coarse_stage1_anchor=" +
      to_string(timing.component_graph_refine_coarse_stage1_anchor_s) +
      " s, component_graph_refine_coarse_stage1_cut_score=" +
      to_string(timing.component_graph_refine_coarse_stage1_cut_score_s) +
      " s, component_graph_refine_coarse_stage1_cut_score_principal=" +
      to_string(timing.component_graph_refine_coarse_stage1_cut_score_principal_s) +
      " s, component_graph_refine_coarse_stage1_cut_score_edge_scan=" +
      to_string(timing.component_graph_refine_coarse_stage1_cut_score_edge_scan_s) +
      " s, component_graph_refine_coarse_stage2=" +
      to_string(timing.component_graph_refine_coarse_stage2_s) +
      " s, component_graph_refine_coarse_stage2_build_points=" +
      to_string(timing.component_graph_refine_coarse_stage2_build_points_s) +
      " s, component_graph_refine_coarse_stage2_continuity=" +
      to_string(timing.component_graph_refine_coarse_stage2_continuity_s) +
      " s, component_graph_refine_coarse_stage2_anchor=" +
      to_string(timing.component_graph_refine_coarse_stage2_anchor_s) +
      " s, component_graph_refine_coarse_stage2_cut_score=" +
      to_string(timing.component_graph_refine_coarse_stage2_cut_score_s) +
      " s, component_graph_refine_build_points=" +
      to_string(timing.component_graph_refine_build_points_s) +
      " s, component_graph_refine_continuity=" +
      to_string(timing.component_graph_refine_continuity_s) +
      " s, component_graph_refine_anchor=" +
      to_string(timing.component_graph_refine_anchor_s) +
      " s, component_graph_refine_anchor_spatial_build=" +
      to_string(timing.component_graph_refine_anchor_spatial_build_s) +
      " s, component_graph_refine_anchor_spatial_query=" +
      to_string(timing.component_graph_refine_anchor_spatial_query_s) +
      " s, component_graph_refine_anchor_dense_fallback=" +
      to_string(timing.component_graph_refine_anchor_dense_fallback_s) +
      " s, pbctopo_anchor_context_build=" +
      to_string(timing.pbctopo_anchor_context_build_s) +
      " s, component_graph_refine_anchor_order=" +
      to_string(timing.component_graph_refine_anchor_order_s) +
      " s, component_graph_refine_anchor_contact=" +
      to_string(timing.component_graph_refine_anchor_contact_s) +
      " s, component_graph_refine_cut_score=" +
      to_string(timing.component_graph_refine_cut_score_s) +
      " s, component_graph_refine_ultraclash=" +
      to_string(timing.component_graph_refine_ultraclash_s) +
      " s, component_graph_refine_stats=" +
      to_string(timing.component_graph_refine_stats_s) +
      " s, component_cycle_sync=" +
      to_string(timing.component_cycle_sync_s) +
      " s, component_search=" + to_string(timing.component_search_s) +
      " s, offset_extract=" + to_string(timing.component_offset_extract_s) +
      " s, apply=" + to_string(timing.apply_s) + " s");
  append_pbctopo_log_line(
      "[pbctopo timing] coarse_candidates=" +
      to_string(timing.coarse_candidates) + ", refine_candidates=" +
      to_string(timing.refine_candidates) + ", projection_clash_skip_frames=" +
      to_string(timing.projection_clash_skip_frames) +
      ", projection_clash_skip_evals=" +
      to_string(timing.projection_clash_skip_evals) + ", component_frames=" +
      to_string(timing.component_frames) + ", component_shift_trials=" +
      to_string(timing.component_shift_trials) +
      ", hull_rescue_attempts=" + to_string(timing.hull_rescue_attempts) +
      ", hull_rescue_successes=" + to_string(timing.hull_rescue_successes) +
      ", distance_rescue_attempts=" +
      to_string(timing.distance_rescue_attempts) +
      ", distance_rescue_successes=" +
      to_string(timing.distance_rescue_successes) +
      ", pbci_fallback_frames=" + to_string(timing.pbci_fallback_frames) +
      ", pbctopo_llps_local_frames=" +
      to_string(timing.pbctopo_llps_local_frames) +
      ", pbctopo_llps_local_units=" +
      to_string(timing.pbctopo_llps_local_units) +
      ", pbctopo_llps_local_unit_source=" + llps_local_unit_source +
      ", pbctopo_llps_local_edges_used=" +
      to_string(timing.pbctopo_llps_local_edges_used) +
      ", pbctopo_llps_local_edges_skipped_cross_unit=" +
      to_string(timing.pbctopo_llps_local_edges_skipped_cross_unit) +
      ", pbctopo_llps_local_disconnected_units=" +
      to_string(timing.pbctopo_llps_local_disconnected_units) +
      ", early_projection_used=" + to_string(timing.early_projection_used) +
      ", previous_seed_used=" + to_string(timing.previous_seed_used) +
      ", pbctopo_kinematic_seed_candidates=" +
      to_string(timing.pbctopo_kinematic_seed_candidates) +
      ", pbctopo_kinematic_seed_evaluated=" +
      to_string(timing.pbctopo_kinematic_seed_evaluated) +
      ", pbctopo_kinematic_seed_deduped=" +
      to_string(timing.pbctopo_kinematic_seed_deduped) +
      ", pbctopo_kinematic_seed_best=" +
      to_string(timing.pbctopo_kinematic_seed_best) +
      ", pbctopo_kinematic_seed_history_resets=" +
      to_string(timing.pbctopo_kinematic_seed_history_resets) +
      ", pbctopo_density_cut_planes_frames=" +
      to_string(timing.pbctopo_density_cut_planes_frames) +
      ", pbctopo_density_cut_planes_x=" +
      to_string(timing.pbctopo_density_cut_planes_x) +
      ", pbctopo_density_cut_planes_y=" +
      to_string(timing.pbctopo_density_cut_planes_y) +
      ", pbctopo_density_cut_planes_z=" +
      to_string(timing.pbctopo_density_cut_planes_z) +
      ", pbctopo_density_cut_planes_dense_equiv_x=" +
      to_string(timing.pbctopo_density_cut_planes_dense_equiv_x) +
      ", pbctopo_density_cut_planes_dense_equiv_y=" +
      to_string(timing.pbctopo_density_cut_planes_dense_equiv_y) +
      ", pbctopo_density_cut_planes_dense_equiv_z=" +
      to_string(timing.pbctopo_density_cut_planes_dense_equiv_z) +
      ", pbctopo_density_cut_plane_collisions=" +
      to_string(timing.pbctopo_density_cut_plane_collisions) +
      ", pbctopo_density_cut_plane_dense_fallbacks=" +
      to_string(timing.pbctopo_density_cut_plane_dense_fallbacks) +
      ", pbctopo_virtual_discrete_frames=" +
      to_string(timing.pbctopo_virtual_discrete_frames) +
      ", pbctopo_virtual_axis_states_x=" +
      to_string(timing.pbctopo_virtual_axis_states_x) +
      ", pbctopo_virtual_axis_states_y=" +
      to_string(timing.pbctopo_virtual_axis_states_y) +
      ", pbctopo_virtual_axis_states_z=" +
      to_string(timing.pbctopo_virtual_axis_states_z) +
      ", pbctopo_virtual_state_space=" +
      to_string(timing.pbctopo_virtual_state_space) +
      ", pbctopo_virtual_states_evaluated=" +
      to_string(timing.pbctopo_virtual_states_evaluated) +
      ", pbctopo_virtual_search_exhaustive_frames=" +
      to_string(timing.pbctopo_virtual_search_exhaustive_frames) +
      ", pbctopo_virtual_seam_enumeration_exhaustive_frames=" +
      to_string(
          timing.pbctopo_virtual_seam_enumeration_exhaustive_frames) +
      ", pbctopo_virtual_projection_scoring_exhaustive_frames=" +
      to_string(
          timing.pbctopo_virtual_projection_scoring_exhaustive_frames) +
      ", pbctopo_virtual_full_atom_verification_exhaustive_frames=" +
      to_string(
          timing.pbctopo_virtual_full_atom_verification_exhaustive_frames) +
      ", pbctopo_virtual_search_shortlist_frames=" +
      to_string(timing.pbctopo_virtual_search_shortlist_frames) +
      ", pbctopo_virtual_branch_and_bound_frames=" +
      to_string(timing.pbctopo_virtual_branch_and_bound_frames) +
      ", pbctopo_virtual_sweep_workspace_reuse_frames=" +
      to_string(timing.pbctopo_virtual_sweep_workspace_reuse_frames) +
      ", pbctopo_virtual_sweep_events_x=" +
      to_string(timing.pbctopo_virtual_sweep_events_x) +
      ", pbctopo_virtual_sweep_events_y=" +
      to_string(timing.pbctopo_virtual_sweep_events_y) +
      ", pbctopo_virtual_sweep_events_z=" +
      to_string(timing.pbctopo_virtual_sweep_events_z) +
      ", pbctopo_virtual_sweep_workspace_peak_bytes=" +
      to_string(timing.pbctopo_virtual_sweep_workspace_peak_bytes) +
      ", pbctopo_verify_geometry_buffer_reuses=" +
      to_string(timing.pbctopo_verify_geometry_buffer_reuses) +
      ", pbctopo_frame_point_buffer_peak_bytes_estimate=" +
      to_string(timing.pbctopo_frame_point_buffer_peak_bytes_estimate) +
      ", pbctopo_virtual_omitted_seam_lower_bound_min=" +
      (isfinite(timing.pbctopo_virtual_omitted_seam_lower_bound_min)
           ? to_string(timing.pbctopo_virtual_omitted_seam_lower_bound_min)
           : string("na")) +
      ", pbctopo_virtual_lower_bound_gap_min=" +
      (isfinite(timing.pbctopo_virtual_lower_bound_gap_min)
           ? to_string(timing.pbctopo_virtual_lower_bound_gap_min)
           : string("na")) +
       ", component_graph_edges=" + to_string(timing.component_graph_edges) +
       ", component_graph_input_shift_domain_max=" +
       to_string(timing.component_graph_input_shift_domain_max) +
       ", component_graph_relative_shift_domain_max=" +
       to_string(timing.component_graph_relative_shift_domain_max) +
       ", component_graph_shift_descriptors=" +
       to_string(timing.component_graph_shift_descriptors) +
       ", component_graph_shift_materialized=" +
       to_string(timing.component_graph_shift_materialized) +
       ", component_graph_bbox_only_scores=" +
       to_string(timing.component_graph_bbox_only_scores) +
       ", component_graph_spatial_shortlist_builds=" +
       to_string(timing.component_graph_spatial_shortlist_builds) +
       ", component_graph_spatial_shortlist_fallbacks=" +
       to_string(timing.component_graph_spatial_shortlist_fallbacks) +
       ", component_graph_spatial_shortlist_pairs=" +
       to_string(timing.component_graph_spatial_shortlist_pairs) +
       ", component_graph_spatial_shortlist_shift_scores_avoided=" +
       to_string(
           timing.component_graph_spatial_shortlist_shift_scores_avoided) +
        ", component_graph_used=" + to_string(timing.component_graph_used) +
        ", component_graph_primary_attempts=" +
        to_string(timing.component_graph_primary_attempts) +
        ", component_graph_primary_certified=" +
        to_string(timing.component_graph_primary_certified) +
        ", component_graph_primary_virtual_skips=" +
        to_string(timing.component_graph_primary_virtual_skips) +
        ", component_graph_primary_fallbacks=" +
        to_string(timing.component_graph_primary_fallbacks) +
        ", component_graph_primary_unidentified=" +
        to_string(timing.component_graph_primary_unidentified) +
        ", component_graph_unidentified_output_rejects=" +
        to_string(timing.component_graph_unidentified_output_rejects) +
        ", component_image_shell_expansions=" +
        to_string(timing.component_image_shell_expansions) +
        ", component_image_shell_retry_successes=" +
        to_string(timing.component_image_shell_retry_successes) +
        ", component_image_shell_max_radius=" +
        to_string(timing.component_image_shell_max_radius) +
        ", component_image_shell_max_shift_count=" +
        to_string(timing.component_image_shell_max_shift_count) +
         ", component_warm_used=" + to_string(timing.component_warm_used) +
       ", component_warm_skip_search=" + to_string(timing.component_warm_skip_search) +
       ", component_search_frames=" + to_string(timing.component_search_frames) +
       ", component_graph_failed=" + to_string(timing.component_graph_failed) +
       ", component_graph_points_mismatch=" +
       to_string(timing.component_graph_points_mismatch) +
       ", component_graph_invalid=" + to_string(timing.component_graph_invalid) +
       ", component_graph_ultraclash=" +
       to_string(timing.component_graph_ultraclash) +
       ", component_graph_not_better_not_close=" +
       to_string(timing.component_graph_not_better_not_close) +
        ", component_refine_warm_shift_hits=" +
        to_string(timing.component_refine_warm_shift_hits) +
       ", component_refine_shortlist_hint_hits=" +
       to_string(timing.component_refine_shortlist_hint_hits) +
       ", component_pair_warm_shift_hits=" +
       to_string(timing.component_pair_warm_shift_hits) +
       ", component_refine_fanout_reduced=" +
       to_string(timing.component_refine_fanout_reduced) +
       ", component_graph_refine_score_calls=" +
       to_string(timing.component_graph_refine_score_calls) +
      ", component_graph_refine_coarse_score_calls=" +
      to_string(timing.component_graph_refine_coarse_score_calls) +
      ", component_graph_refine_coarse_stage1_score_calls=" +
      to_string(timing.component_graph_refine_coarse_stage1_score_calls) +
      ", component_graph_refine_coarse_stage2_score_calls=" +
      to_string(timing.component_graph_refine_coarse_stage2_score_calls) +
      ", component_graph_refine_exact_rescore_calls=" +
      to_string(timing.component_graph_refine_exact_rescore_calls) +
      ", component_graph_refine_preclash_prunes=" +
      to_string(timing.component_graph_refine_preclash_prunes) +
      ", component_graph_refine_trial_build_incremental=" +
      to_string(timing.component_graph_refine_trial_build_incremental) +
      ", component_graph_refine_trial_build_full=" +
      to_string(timing.component_graph_refine_trial_build_full) +
      ", component_graph_refine_trial_build_global_rebase=" +
      to_string(timing.component_graph_refine_trial_build_global_rebase) +
      ", component_graph_refine_stats_cached=" +
      to_string(timing.component_graph_refine_stats_cached) +
      ", component_graph_refine_stats_full=" +
      to_string(timing.component_graph_refine_stats_full) +
      ", component_graph_refine_anchor_spatial_calls=" +
      to_string(timing.component_graph_refine_anchor_spatial_calls) +
      ", component_graph_refine_anchor_spatial_fallbacks=" +
      to_string(timing.component_graph_refine_anchor_spatial_fallbacks) +
      ", component_graph_refine_anchor_spatial_disconnects=" +
      to_string(timing.component_graph_refine_anchor_spatial_disconnects) +
      ", component_graph_refine_anchor_spatial_expansions=" +
      to_string(timing.component_graph_refine_anchor_spatial_expansions) +
      ", component_graph_refine_anchor_spatial_exact_certified=" +
      to_string(timing.component_graph_refine_anchor_spatial_exact_certified) +
      ", component_graph_refine_anchor_spatial_uncertified=" +
      to_string(timing.component_graph_refine_anchor_spatial_uncertified) +
      ", component_graph_refine_anchor_spatial_max_k=" +
      to_string(timing.component_graph_refine_anchor_spatial_max_k) +
      ", component_graph_contact_epoch_scratch_resizes=" +
      to_string(timing.component_graph_contact_epoch_scratch_resizes) +
      ", component_graph_contact_epoch_wrap_clears=" +
      to_string(timing.component_graph_contact_epoch_wrap_clears) +
      ", component_graph_shift_descriptor_domain=" +
      to_string(timing.component_graph_shift_descriptor_domain) +
      ", component_graph_shift_cache_entries_peak=" +
      to_string(timing.component_graph_shift_cache_entries_peak) +
      ", component_graph_shift_cache_dense_entries_equivalent=" +
      to_string(
          timing.component_graph_shift_cache_dense_entries_equivalent) +
      ", component_cycle_sync_attempts=" +
      to_string(timing.component_cycle_sync_attempts) +
      ", component_cycle_sync_used=" +
      to_string(timing.component_cycle_sync_used) +
      ", component_cycle_sync_pair_candidates=" +
      to_string(timing.component_cycle_sync_pair_candidates) +
      ", component_cycle_sync_cycle_support=" +
      to_string(timing.component_cycle_sync_cycle_support) +
      ", component_cycle_sync_cycle_mismatch=" +
      to_string(timing.component_cycle_sync_cycle_mismatch) +
      ", component_cycle_sync_inconsistent_edges=" +
      to_string(timing.component_cycle_sync_inconsistent_edges) +
      ", pbctopo_topology_bonds_available=" +
      to_string(timing.pbctopo_topology_bonds_available) +
      ", pbctopo_topology_bond_edges_added=" +
      to_string(timing.pbctopo_topology_bond_edges_added) +
      ", pbctopo_topology_bond_edges_skipped=" +
      to_string(timing.pbctopo_topology_bond_edges_skipped) +
      ", pbctopo_geometry_edges_added=" +
      to_string(timing.pbctopo_geometry_edges_added) +
      ", pbctopo_hard_connectivity_edges=" +
      to_string(timing.pbctopo_hard_connectivity_edges) +
      ", pbctopo_soft_geometry_edges=" +
      to_string(timing.pbctopo_soft_geometry_edges) +
      ", pbctopo_steric_exclusion_pairs=" +
      to_string(timing.pbctopo_steric_exclusion_pairs) +
      ", pbctopo_component_logic_disabled_no_hard_graph=" +
      to_string(timing.pbctopo_component_logic_disabled_no_hard_graph) +
      ", pbctopo_topology_duplicate_edges=" +
      to_string(timing.pbctopo_topology_duplicate_edges) +
      ", pbctopo_topology_projected_bonds=" +
      to_string(timing.pbctopo_topology_projected_bonds) +
      ", pbctopo_topology_static_graph_frames=" +
      to_string(timing.pbctopo_topology_static_graph_frames) +
      ", pbctopo_metadata_graph_frames=" +
      to_string(timing.pbctopo_metadata_graph_frames) +
      ", pbctopo_metadata_residue_groups=" +
      to_string(timing.pbctopo_metadata_residue_groups) +
      ", pbctopo_metadata_intra_edges=" +
      to_string(timing.pbctopo_metadata_intra_edges) +
      ", pbctopo_metadata_inter_edges=" +
      to_string(timing.pbctopo_metadata_inter_edges) +
      ", pbctopo_metadata_inter_hard_edges=" +
      to_string(timing.pbctopo_metadata_inter_hard_edges) +
      ", pbctopo_metadata_inter_soft_edges=" +
      to_string(timing.pbctopo_metadata_inter_soft_edges) +
      ", pbctopo_geometry_fallback_frames=" +
      to_string(timing.pbctopo_geometry_fallback_frames) +
      ", pbctopo_topology_source=" +
      pbctopo_topology_source_label(timing) +
      ", pbctopo_topology_component_source=" +
      string((timing.pbctopo_topology_static_graph_frames > 0 &&
              timing.pbctopo_metadata_graph_frames > 0)
                 ? "topology+metadata"
                 : ((timing.frames > 0 &&
              timing.pbctopo_topology_static_graph_frames == timing.frames)
                        ? "topology"
                        : ((timing.frames > 0 &&
                            timing.pbctopo_metadata_graph_frames ==
                                timing.frames)
                               ? "metadata"
                               : ((timing.pbctopo_hard_connectivity_edges > 0)
                                      ? "trusted_mixed"
                                      : "none")))) +
      ", pbctopo_anchor_order_edges=" +
      to_string(timing.pbctopo_anchor_order_edges) +
      ", pbctopo_anchor_endpoint_edges=" +
      to_string(timing.pbctopo_anchor_endpoint_edges) +
      ", pbctopo_anchor_contact_pairs=" +
      to_string(timing.pbctopo_anchor_contact_pairs) +
      ", pbctopo_anchor_intra_owner_contact_pairs=" +
      to_string(timing.pbctopo_anchor_intra_owner_contact_pairs) +
      ", pbctopo_soft_observed_contact_pairs=" +
      to_string(timing.pbctopo_soft_observed_contact_pairs) +
      ", pbctopo_hard_required_owner_pairs=" +
      to_string(timing.pbctopo_hard_required_owner_pairs) +
      ", pbctopo_anchor_residue_order_violations=" +
      to_string(timing.pbctopo_anchor_residue_order_violations) +
      ", pbctopo_anchor_ambiguous_frames=" +
      to_string(timing.pbctopo_anchor_ambiguous_frames) +
      ", component_context_builds=" +
      to_string(timing.component_context_builds) +
      ", component_offset_extracts=" +
      to_string(timing.component_offset_extracts) + ", apply_calls=" +
      to_string(timing.apply_calls) +
      ", component_partial_checks=" +
      to_string(timing.component_partial_checks) +
      ", component_partial_prunes=" +
      to_string(timing.component_partial_prunes) +
      ", component_bbox_skips=" + to_string(timing.component_bbox_skips) +
      ", component_score_calls=" +
      to_string(timing.component_score_calls) + ", component_partial_time=" +
      to_string(timing.component_partial_s) + " s, component_score_time=" +
      to_string(timing.component_score_s) + " s");
  append_pbctopo_log_line(
      "[pbctopo timing] component_pair_table_builds=" +
      to_string(timing.component_pair_table_builds) +
      ", component_pair_table_time=" +
      to_string(timing.component_pair_table_s) +
      " s, component_pair_table_prunes=" +
      to_string(timing.component_pair_table_prunes) +
      ", component_pair_table_component_count=" +
      to_string(timing.component_pair_table_component_count) +
      ", component_pair_table_shift_count=" +
      to_string(timing.component_pair_table_shift_count) +
      ", component_pair_table_relative_deltas=" +
      to_string(timing.component_pair_table_relative_deltas) +
      ", component_pair_table_relative_reuses=" +
      to_string(timing.component_pair_table_relative_reuses) +
      ", component_relative_domain_analytic_builds=" +
      to_string(timing.component_relative_domain_analytic_builds) +
      ", component_relative_domain_generic_builds=" +
      to_string(timing.component_relative_domain_generic_builds) +
      ", component_pair_table_dense_entries_equivalent=" +
      to_string(timing.component_pair_table_dense_entries_equivalent) +
      ", component_pair_table_materialized_entries=" +
      to_string(timing.component_pair_table_materialized_entries) +
      ", component_pair_table_bbox_prefiltered=" +
      to_string(timing.component_pair_table_bbox_prefiltered) +
      ", component_pair_table_sparse_rows=" +
      to_string(timing.component_pair_table_sparse_rows) +
      ", component_pair_table_peak_bytes_estimate=" +
      to_string(timing.component_pair_table_peak_bytes_estimate) +
      ", component_search_zero_current_mask_prunes=" +
      to_string(timing.component_search_zero_current_mask_prunes) +
      ", component_search_zero_future_mask_prunes=" +
      to_string(timing.component_search_zero_future_mask_prunes) +
      ", component_search_fixed_minor_frames=" +
      to_string(timing.component_search_fixed_minor_frames) +
      ", component_search_fixed_minor_components=" +
      to_string(timing.component_search_fixed_minor_components) +
      ", component_search_shift_descriptors=" +
      to_string(timing.component_search_shift_descriptors) +
      ", component_search_shift_materialized=" +
      to_string(timing.component_search_shift_materialized) +
      ", component_search_shift_domain_max=" +
      to_string(timing.component_search_shift_domain_max) +
      ", component_search_lazy_materialize_time=" +
      to_string(timing.component_search_lazy_materialize_s) + " s" +
      ", component_search_seed_warm=" +
      to_string(timing.component_search_seed_warm) +
      ", component_search_seed_graph=" +
      to_string(timing.component_search_seed_graph) +
      ", component_search_seed_none=" +
      to_string(timing.component_search_seed_none) +
      ", component_search_prefix_continuity_checks=" +
      to_string(timing.component_search_prefix_continuity_checks) +
      ", component_search_prefix_continuity_prunes=" +
      to_string(timing.component_search_prefix_continuity_prunes) +
      ", component_search_prefix_continuity_hard_rejects=" +
      to_string(timing.component_search_prefix_continuity_hard_rejects) +
      ", component_search_exact_budget_skips=" +
      to_string(timing.component_search_exact_budget_skips) +
      ", component_audit_exact_frames=" +
      to_string(timing.component_audit_exact_frames) +
      ", component_audit_exact_completed=" +
      to_string(timing.component_audit_exact_completed) +
      ", component_audit_exact_incomplete=" +
      to_string(timing.component_audit_exact_incomplete) +
      ", component_audit_exact_state_limit=" +
      to_string(timing.component_audit_exact_state_limit) +
      ", component_audit_exact_wall_time_limit=" +
      to_string(timing.component_audit_exact_wall_time_limit) +
      ", component_audit_exact_memory_limit=" +
      to_string(timing.component_audit_exact_memory_limit) +
      ", component_audit_exact_reference_matches=" +
      to_string(timing.component_audit_exact_reference_matches) +
      ", component_audit_exact_reference_mismatches=" +
      to_string(timing.component_audit_exact_reference_mismatches) +
      ", component_pair_table_all_atom_relative_checks=" +
      to_string(timing.component_pair_table_all_atom_relative_checks) +
      ", component_pair_table_all_atom_relative_rejects=" +
      to_string(timing.component_pair_table_all_atom_relative_rejects) +
      ", component_pair_table_all_atom_materialized=" +
      to_string(timing.component_pair_table_all_atom_materialized) +
      ", component_search_beam_adaptive_retries=" +
      to_string(timing.component_search_beam_adaptive_retries) +
      ", component_search_beam_adaptive_retry_successes=" +
      to_string(timing.component_search_beam_adaptive_retry_successes) +
      ", component_search_beam_adaptive_max_width=" +
      to_string(timing.component_search_beam_adaptive_max_width) +
      ", component_search_exact_state_upper_bound=" +
      to_string(timing.component_search_exact_state_upper_bound) +
      ", component_search_filtered_state_upper_bound=" +
      to_string(timing.component_search_filtered_state_upper_bound) +
      ", component_search_arc_consistency_passes=" +
      to_string(timing.component_search_arc_consistency_passes) +
      ", component_search_arc_consistency_revisions=" +
      to_string(timing.component_search_arc_consistency_revisions) +
      ", component_search_arc_consistency_removals=" +
      to_string(timing.component_search_arc_consistency_removals) +
      ", component_search_arc_consistency_queue_pushes=" +
      to_string(timing.component_search_arc_consistency_queue_pushes) +
      ", component_search_arc_consistency_failures=" +
      to_string(timing.component_search_arc_consistency_failures) +
      ", component_search_exact_recoveries=" +
      to_string(timing.component_search_exact_recoveries) +
      ", component_search_arc_consistency_time=" +
      to_string(timing.component_search_arc_consistency_s) + " s" +
      ", component_search_prefix_continuity_time=" +
      to_string(timing.component_search_prefix_continuity_s) +
      " s" +
      ", component_large_mode_merge_frames=" +
      to_string(timing.component_large_mode_merge_frames) +
      ", component_large_mode_pair_only_frames=" +
      to_string(timing.component_large_mode_pair_only_frames) +
      ", component_large_mode_beam_frames=" +
      to_string(timing.component_large_mode_beam_frames) +
      ", component_large_mode_merge_count=" +
      to_string(timing.component_large_mode_merge_count) +
      ", component_large_mode_internal_offsets_preserved=" +
      to_string(timing.component_large_mode_internal_offsets_preserved) +
      ", component_large_mode_internal_offset_nonzero=" +
      to_string(timing.component_large_mode_internal_offset_nonzero) +
      ", component_large_mode_graph_seed_count=" +
      to_string(timing.component_large_mode_graph_seed_count) +
      ", component_large_mode_graph_seed_refined=" +
      to_string(timing.component_large_mode_graph_seed_refined) +
      ", component_large_mode_graph_seed_nonbaseline_win=" +
      to_string(timing.component_large_mode_graph_seed_nonbaseline_win) +
      ", component_large_mode_beam_expanded=" +
      to_string(timing.component_large_mode_beam_expanded) +
      ", component_large_mode_beam_kept=" +
      to_string(timing.component_large_mode_beam_kept) +
      ", component_large_mode_beam_complete_assignments=" +
      to_string(timing.component_large_mode_beam_complete_assignments) +
      ", component_large_mode_beam_final_rescored=" +
      to_string(timing.component_large_mode_beam_final_rescored) +
      ", component_large_mode_beam_final_rescore_all_frames=" +
      to_string(timing.component_large_mode_beam_final_rescore_all_frames) +
      ", component_search_beam_provisional_children=" +
      to_string(timing.component_search_beam_provisional_children) +
      ", component_search_beam_state_materializations=" +
      to_string(timing.component_search_beam_state_materializations) +
      ", component_search_beam_state_copies_avoided=" +
      to_string(timing.component_search_beam_state_copies_avoided) +
      ", component_search_beam_parent_state_moves=" +
      to_string(timing.component_search_beam_parent_state_moves) +
      ", component_search_certification_survivors=" +
      to_string(timing.component_search_certification_survivors) +
      ", component_search_certification_band_retained=" +
      to_string(timing.component_search_certification_band_retained) +
      ", component_large_mode_fallback_frames=" +
      to_string(timing.component_large_mode_fallback_frames) +
      ", component_search_shift_warm_hits=" +
      to_string(timing.component_search_shift_warm_hits) +
      ", pbctopo_certification_attempts=" +
      to_string(timing.pbctopo_certification_attempts) +
      ", pbctopo_certification_successes=" +
      to_string(timing.pbctopo_certification_successes) +
      ", pbctopo_certification_assignment_rejects=" +
      to_string(timing.pbctopo_certification_assignment_rejects) +
      ", pbctopo_certification_hard_rejects=" +
      to_string(timing.pbctopo_certification_hard_rejects) +
      ", pbctopo_output_certification_band_precheck_rejects=" +
      to_string(timing.pbctopo_output_certification_band_precheck_rejects) +
      ", pbctopo_output_certification_band_rebuilds=" +
      to_string(timing.pbctopo_output_certification_band_rebuilds) +
      ", pbctopo_output_certification_band_offset_bytes_peak=" +
      to_string(timing.pbctopo_output_certification_band_offset_bytes_peak) +
      ", pbctopo_output_certification_band_point_bytes_avoided_peak=" +
      to_string(
          timing.pbctopo_output_certification_band_point_bytes_avoided_peak) +
      ", pbctopo_heavy_search_frames=" +
      to_string(timing.pbctopo_heavy_search_frames) +
      ", pbctopo_heavy_search_all_atom_candidate_rejects=" +
      to_string(timing.pbctopo_heavy_search_all_atom_candidate_rejects) +
      ", pbctopo_heavy_search_top_candidate_rejects=" +
      to_string(timing.pbctopo_heavy_search_top_candidate_rejects) +
      ", pbctopo_heavy_search_all_atom_ranking_recoveries=" +
      to_string(timing.pbctopo_heavy_search_all_atom_ranking_recoveries) +
      ", pbctopo_heavy_search_all_atom_band_exhausted=" +
      to_string(timing.pbctopo_heavy_search_all_atom_band_exhausted) +
      ", pbctopo_runtime_simd_forced_scalar=" +
      to_string(titan_simd_dispatch::runtime_simd_forced_scalar() ? 1 : 0) +
      ", pbctopo_output_certification_band_candidates=" +
      to_string(timing.pbctopo_output_certification_band_candidates) +
      ", pbctopo_output_certification_band_peak=" +
      to_string(timing.pbctopo_output_certification_band_peak) +
      ", pbctopo_output_certification_band_rejects=" +
      to_string(timing.pbctopo_output_certification_band_rejects) +
      ", pbctopo_output_certification_band_recoveries=" +
      to_string(timing.pbctopo_output_certification_band_recoveries) +
      ", pbctopo_output_certification_band_expanded_frames=" +
      to_string(timing.pbctopo_output_certification_band_expanded_frames) +
      ", pbctopo_output_certification_band_expanded_recoveries=" +
      to_string(
          timing.pbctopo_output_certification_band_expanded_recoveries) +
      ", pbctopo_certification_topology_edges_checked=" +
      to_string(timing.pbctopo_certification_topology_edges_checked) +
      ", pbctopo_certification_ambiguous_topology_edges=" +
      to_string(timing.pbctopo_certification_ambiguous_topology_edges) +
      ", pbctopo_certification_topology_edge_residuals=" +
      to_string(timing.pbctopo_certification_topology_edge_residuals) +
      ", pbctopo_certification_topology_cycle_residuals=" +
      to_string(timing.pbctopo_certification_topology_cycle_residuals) +
      ", pbctopo_certification_atom_image_residuals=" +
      to_string(timing.pbctopo_certification_atom_image_residuals) +
      ", pbctopo_certification_assignment_point_residuals=" +
      to_string(timing.pbctopo_certification_assignment_point_residuals) +
      ", pbctopo_certification_steric_pair_checks=" +
      to_string(timing.pbctopo_certification_steric_pair_checks) +
      ", pbctopo_certification_steric_clashes=" +
      to_string(timing.pbctopo_certification_steric_clashes) +
      ", pbctopo_certification_required_contacts_checked=" +
      to_string(timing.pbctopo_certification_required_contacts_checked) +
      ", pbctopo_certification_required_contacts_lost=" +
      to_string(timing.pbctopo_certification_required_contacts_lost) +
      ", pbctopo_required_interface_contacts_loaded=" +
      to_string(timing.pbctopo_required_interface_contacts_loaded) +
      ", component_required_contact_constraint_builds=" +
      to_string(timing.component_required_contact_constraint_builds) +
      ", component_required_contact_constraint_pairs=" +
      to_string(timing.component_required_contact_constraint_pairs) +
      ", component_required_contact_relative_deltas_checked=" +
      to_string(timing.component_required_contact_relative_deltas_checked) +
      ", component_required_contact_relative_deltas_rejected=" +
      to_string(timing.component_required_contact_relative_deltas_rejected) +
      ", component_required_contact_infeasible=" +
      to_string(timing.component_required_contact_infeasible) +
      ", pbctopo_certification_explicit_interface_contacts_checked=" +
      to_string(
          timing.pbctopo_certification_explicit_interface_contacts_checked) +
      ", pbctopo_certification_explicit_interface_contacts_lost=" +
      to_string(
          timing.pbctopo_certification_explicit_interface_contacts_lost) +
      ", pbctopo_geometry_classifications=" +
      to_string(timing.pbctopo_geometry_classifications) +
      ", pbctopo_compact_unit_box_certified_frames=" +
      to_string(timing.pbctopo_compact_unit_box_certified_frames) +
      ", pbctopo_noncompact_certified_frames=" +
      to_string(timing.pbctopo_noncompact_certified_frames) +
      ", pbctopo_extended_certified_frames=" +
      to_string(timing.pbctopo_extended_certified_frames) +
      ", pbctopo_winding_rejected_candidates=" +
      to_string(timing.pbctopo_winding_rejected_candidates) +
      ", pbctopo_extended_winding_rejected_candidates=" +
      to_string(timing.pbctopo_extended_winding_rejected_candidates) +
      ", pbctopo_owner_com_split_observations=" +
      to_string(timing.pbctopo_owner_com_split_observations) +
      ", pbctopo_warm_state_commits=" +
      to_string(timing.pbctopo_warm_state_commits) +
      ", pbctopo_warm_state_resets=" +
      to_string(timing.pbctopo_warm_state_resets) +
      ", pbctopo_warm_state_commit_rejects=" +
      to_string(timing.pbctopo_warm_state_commit_rejects) +
      ", pbctopo_warm_projection_rejects=" +
      to_string(timing.pbctopo_warm_projection_rejects) +
      ", pbctopo_warm_component_rejects=" +
      to_string(timing.pbctopo_warm_component_rejects) +
      ", pbctopo_weak_observation_output_frames=" +
      to_string(timing.pbctopo_weak_observation_output_frames) +
      ", pbctopo_contradicted_output_frames=" +
      to_string(timing.pbctopo_contradicted_output_frames) +
      ", pbctopo_selected_evidence_audits=" +
      to_string(timing.pbctopo_selected_evidence_audits) +
      ", pbctopo_selected_evidence_contacts_checked=" +
      to_string(timing.pbctopo_selected_evidence_contacts_checked) +
      ", pbctopo_selected_evidence_contacts_lost=" +
      to_string(timing.pbctopo_selected_evidence_contacts_lost) +
      ", pbctopo_eobs_construction_attempts=" +
      to_string(timing.pbctopo_eobs_construction_attempts) +
      ", pbctopo_eobs_construction_completed=" +
      to_string(timing.pbctopo_eobs_construction_completed) +
      ", pbctopo_eobs_construction_failures=" +
      to_string(timing.pbctopo_eobs_construction_failures) +
      ", pbctopo_eobs_cell_setup_failures=" +
      to_string(timing.pbctopo_eobs_cell_setup_failures) +
      ", pbctopo_eobs_atom_index_failures=" +
      to_string(timing.pbctopo_eobs_atom_index_failures) +
      ", pbctopo_eobs_mic_query_failures=" +
      to_string(timing.pbctopo_eobs_mic_query_failures) +
      ", pbctopo_selected_evidence_audit_s=" +
      to_string(timing.pbctopo_selected_evidence_audit_s) +
      ", pbctopo_weak_observation_warm_preserved=" +
      to_string(timing.pbctopo_weak_observation_warm_preserved) +
      ", pbctopo_warm_layout_mismatches=" +
      to_string(timing.pbctopo_warm_layout_mismatches) +
      ", pbctopo_diagnostic_only_candidates=" +
      to_string(timing.pbctopo_diagnostic_only_candidates) +
      ", pbctopo_virtual_geometry_reference_evals=" +
      to_string(timing.pbctopo_virtual_geometry_reference_evals) +
      ", pbctopo_virtual_geometry_reference_safe=" +
      to_string(timing.pbctopo_virtual_geometry_reference_safe) +
      ", pbctopo_virtual_geometry_reference_clashing=" +
      to_string(timing.pbctopo_virtual_geometry_reference_clashing) +
      ", pbctopo_virtual_geometry_outlier_rejects=" +
      to_string(timing.pbctopo_virtual_geometry_outlier_rejects) +
      ", pbctopo_virtual_geometry_outlier_observations=" +
      to_string(timing.pbctopo_virtual_geometry_outlier_observations) +
      ", component_graph_outlier_rejects=" +
      to_string(timing.component_graph_outlier_rejects) +
      ", component_search_outlier_rejects=" +
      to_string(timing.component_search_outlier_rejects) +
      ", component_graph_outlier_observations=" +
      to_string(timing.component_graph_outlier_observations) +
      ", component_search_outlier_observations=" +
      to_string(timing.component_search_outlier_observations) +
      ", component_score_continuity_time=" +
      to_string(timing.component_score_continuity_s) +
      " s, component_score_anchor_time=" +
      to_string(timing.component_score_anchor_s) +
      " s, component_score_stats_principal_time=" +
      to_string(timing.component_score_stats_principal_s) +
      " s, component_score_component_metrics_time=" +
      to_string(timing.component_score_component_metrics_s) +
      " s, component_score_continuity_rejects=" +
      to_string(timing.component_score_continuity_rejects) +
      ", component_score_anchor_rejects=" +
      to_string(timing.component_score_anchor_rejects) +
      ", component_score_prefix_prunes_after_continuity=" +
      to_string(timing.component_score_prefix_prunes_after_continuity) +
      ", component_score_prefix_prunes_after_anchor=" +
      to_string(timing.component_score_prefix_prunes_after_anchor) +
      ", component_score_prefix_prunes_after_principal=" +
      to_string(timing.component_score_prefix_prunes_after_principal) +
      ", component_score_prefix_prunes_after_component_metrics=" +
      to_string(timing.component_score_prefix_prunes_after_component_metrics) +
      ", component_score_prefix_prunes_after_final=" +
      to_string(timing.component_score_prefix_prunes_after_final) +
      ", component_search_depth_visits=[d0:" +
      to_string(timing.component_search_depth_visits[0]) + ",d1:" +
      to_string(timing.component_search_depth_visits[1]) + ",d2:" +
      to_string(timing.component_search_depth_visits[2]) + ",d3:" +
      to_string(timing.component_search_depth_visits[3]) + ",d4:" +
      to_string(timing.component_search_depth_visits[4]) + ",d5:" +
      to_string(timing.component_search_depth_visits[5]) + ",d6:" +
      to_string(timing.component_search_depth_visits[6]) + ",d7:" +
      to_string(timing.component_search_depth_visits[7]) + "]");
  append_pbctopo_log_line(
      "[pbctopo math] pair_shift_score=normalized, shift_metric="
      "cartesian_component_mean_A2, gauge=closest_lattice_mean, cycle_residual="
      "fundamental_basis, anchor_metrics=" +
      string(titan_pbctopo::anchor_metric_mode_name(anchor_metric_mode)) +
      ", clash_model=vdw_radius, vdw_pair_checks=" +
      to_string(timing.pbctopo_vdw_pair_checks) +
      ", zero_lj_policy=" +
      string(titan_pbctopo::kPbctopoZeroLjStericPolicy) +
      ", zero_lj_exempt_pairs=" +
      to_string(timing.pbctopo_vdw_explicit_zero_lj_exempt_pairs) +
      ", vdw_radius_pairs=" + to_string(timing.pbctopo_vdw_radius_pairs) +
      ", vdw_fallback_pairs=" +
      to_string(timing.pbctopo_vdw_radius_fallback_pairs) +
      ", temporal_mode=" +
      string(titan_pbctopo::temporal_mode_name(temporal_mode)) +
      ", temporal_candidate_frames=" +
      to_string(timing.pbctopo_temporal_candidate_frames) +
      ", temporal_candidates=" +
      to_string(timing.pbctopo_temporal_candidates) +
      ", temporal_certified_candidates=" +
      to_string(timing.pbctopo_temporal_certified_candidates) +
      ", temporal_assignment_deduped=" +
      to_string(timing.pbctopo_temporal_assignment_deduped) +
      ", temporal_diagnostic_only_frames=" +
      to_string(timing.pbctopo_temporal_diagnostic_only_frames) +
      ", temporal_proposed_changes=" +
      to_string(timing.pbctopo_temporal_proposed_changes) +
      ", temporal_path_changes=" +
      to_string(timing.pbctopo_temporal_path_changes) +
      ", temporal_replay_frames=" +
      to_string(timing.pbctopo_temporal_replay_frames) +
      ", temporal_replay_applied=" +
      to_string(timing.pbctopo_temporal_replay_applied) +
      ", temporal_replay_skipped=" +
      to_string(timing.pbctopo_temporal_replay_skipped) +
      ", pbctopo_lattice_audits=" +
      to_string(timing.pbctopo_lattice_audits) +
      ", pbctopo_lattice_invalid_frames=" +
      to_string(timing.pbctopo_lattice_invalid_frames) +
      ", pbctopo_lattice_reduction_fallback_frames=" +
      to_string(timing.pbctopo_lattice_reduction_fallback_frames) +
      ", pbctopo_lattice_query_failures=" +
      to_string(timing.pbctopo_lattice_query_failures) +
      ", pbctopo_lattice_fatal_query_failures=" +
      to_string(timing.pbctopo_lattice_fatal_query_failures) +
      ", pbctopo_lattice_optional_query_failures=" +
      to_string(timing.pbctopo_lattice_optional_query_failures) +
      ", pbctopo_lattice_query_ambiguities=" +
      to_string(timing.pbctopo_lattice_query_ambiguities) +
      ", pbctopo_lattice_query_budget_failures=" +
      to_string(timing.pbctopo_lattice_query_budget_failures) +
      ", pbctopo_lattice_worker_queries=" +
      to_string(timing.pbctopo_lattice_worker_queries) +
      ", pbctopo_lattice_worker_query_failures=" +
      to_string(timing.pbctopo_lattice_worker_query_failures) +
      ", pbctopo_lattice_worker_query_ambiguities=" +
      to_string(timing.pbctopo_lattice_worker_query_ambiguities) +
      ", pbctopo_lattice_worker_query_budget_failures=" +
      to_string(timing.pbctopo_lattice_worker_query_budget_failures) +
      ", pbctopo_lattice_condition_max=" +
      to_string(timing.pbctopo_lattice_condition_max));
  const titan_pbctopo::PbctopoAnchorMstCounters anchor_mst =
      titan_pbctopo::pbctopo_anchor_mst_counters();
  append_pbctopo_log_line(
      "[pbctopo anchor-mst] dense_calls=" + to_string(anchor_mst.dense_calls) +
      ", fused_calls=" + to_string(anchor_mst.fused_calls) +
      ", fused_incumbent_exits=" +
      to_string(anchor_mst.fused_incumbent_exits));
  append_pbctopo_log_line("[pbctopo] Valid candidate evaluations: " +
                          to_string(valid_candidates));
  append_pbctopo_log_line("[pbctopo] end run");
  return 0;
}

} // namespace pbctopo_solve
