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

#include "analyzer_impl.h"

#include "file_sha256.h"

#include "tabi_types.h"
#include "titan_energy_decomposition.h"
#include "titan_executor.h"
#include "titan_interaction_entropy.h"
#include "titan_sasa.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "analyzer_impl_detail.h"

namespace TitanPBE {

using analyzer_impl_detail::csv_escape;
using analyzer_impl_detail::elapsed_ms;
using analyzer_impl_detail::kCoulombKcalPerMol;
using analyzer_impl_detail::kKBKcalPerMolK;

void AnalyzerImpl::write_tabi_audit_profile_header_if_needed() {
  if (args_.tabi_diagnostics == "off")
    return;
  if (!tabi_audit_profile_out_.is_open()) {
    tabi_audit_profile_out_.open("pbe_tabi_audit_profile.csv",
                                 std::ios::out | std::ios::trunc);
    if (!tabi_audit_profile_out_.is_open()) {
      throw std::runtime_error(
          "PBE: failed to open pbe_tabi_audit_profile.csv.");
    }
  }
  if (!tabi_audit_profile_header_written_) {
    tabi_audit_profile_out_
        << "trajectory_id,frame_index,global_index,fixture,atom_count,panels,"
           "energy_kJmol,iterations,internal_residual,sampled_direct_residual,"
           "audit_rows,wall_seconds,rss_mb,p_order,theta,guard,tail_tol,ilu_"
           "cap,far_pairs,fallback_pairs,status\n";
    tabi_audit_profile_header_written_ = true;
  }
}

void AnalyzerImpl::append_tabi_audit_profile_row(const FrameResult &result,
                                                 const TimingBuckets &timing) {
  if (args_.tabi_diagnostics == "off" || result.pbe_backend != "tabi")
    return;
  write_tabi_audit_profile_header_if_needed();

  const double sampled_direct_residual =
      result.tabi_fmm_local_sampled_direct_true_residual > 0.0
          ? result.tabi_fmm_local_sampled_direct_true_residual
          : (result.tabi_sampled_true_residual_rel > 0.0
                 ? result.tabi_sampled_true_residual_rel
                 : result.tabi_tree_vs_direct_matvec_rel);
  const std::size_t audit_rows = result.tabi_matvec_audit_rows > 0
                                     ? result.tabi_matvec_audit_rows
                                     : result.tabi_sampled_true_residual_rows;
  const double internal_residual = result.tabi_final_residual > 0.0
                                       ? result.tabi_final_residual
                                       : result.linear_residual;
  const std::size_t far_pairs = result.tabi_fmm_m2l_far_pairs > 0
                                    ? result.tabi_fmm_m2l_far_pairs
                                    : result.tabi_far_interactions;
  const std::size_t fallback_pairs =
      result.tabi_fmm_local_geometry_guard_direct_fallback_pairs;
  const double rss_mb =
      static_cast<double>(
          std::max(result.memory_peak_working_set_bytes, result.memory_bytes)) /
      (1024.0 * 1024.0);

  tabi_audit_profile_out_ << result.trajectory_id << ',' << result.frame_index
                          << ',' << result.global_index << ','
                          << csv_escape(current_source_label_) << ','
                          << result.atom_count << ','
                          << result.tabi_surface_panels << ','
                          << std::setprecision(12) << result.energy_kj_mol
                          << ',' << result.solvent_iterations << ','
                          << internal_residual << ',' << sampled_direct_residual
                          << ',' << audit_rows << ','
                          << timing.total_ms / 1000.0 << ',' << rss_mb << ','
                          << result.tabi_taylor_order << ','
                          << args_.tabi_tree_opening_theta << ','
                          << csv_escape(args_.tabi_fmm_local_geometry_guard)
                          << ',' << args_.tabi_fmm_local_refine_tol << ','
                          << args_.tabi_precond_ilu_neighbor_cap << ','
                          << far_pairs << ',' << fallback_pairs << ','
                          << csv_escape(result.tabi_status) << '\n';
}

void AnalyzerImpl::write_tabi_profile_header_if_needed() {
  if (args_.timing != "detail")
    return;
  if (!tabi_profile_out_.is_open()) {
    tabi_profile_out_.open("pbe_tabi_profile.csv",
                           std::ios::out | std::ios::trunc);
    if (!tabi_profile_out_.is_open())
      throw std::runtime_error("PBE: failed to open pbe_tabi_profile.csv.");
  }
  if (!tabi_profile_header_written_) {
    tabi_profile_out_
        << "trajectory_id,frame_index,global_index,fixture,atom_count,"
           "surface_geometry,panels,mesh_vertices,mesh_triangles,"
           "surface_point_count_requested,panel_density_per_A2,"
           "panel_budget_status,"
           "matvec_mode,fmm_m2l_mode,fmm_m2l_expansion_order,"
           "fmm_m2l_near_pairs,"
           "fmm_m2l_far_pairs,fmm_m2l_setup_seconds,"
           "fmm_m2l_apply_seconds,fmm_m2l_finite_outputs,"
           "bie_cartesian_full_fmm_active,"
           "bie_cartesian_full_fmm_requested,"
           "bie_cartesian_full_fmm_effective,"
           "bie_cartesian_full_fmm_order,"
           "bie_cartesian_full_fmm_leaf_size,"
           "bie_cartesian_full_fmm_opening_theta,"
           "bie_cartesian_full_fmm_nodes,"
           "bie_cartesian_full_fmm_far_pairs,"
           "bie_cartesian_full_fmm_direct_pairs,"
           "bie_cartesian_full_fmm_direct_panel_pairs_expected,"
           "bie_cartesian_full_fmm_direct_panel_pairs_applied_last,"
           "bie_cartesian_full_fmm_direct_panel_pairs_applied_total,"
           "bie_cartesian_full_fmm_cache_entries,"
           "bie_cartesian_full_fmm_cache_hits,"
           "bie_cartesian_full_fmm_cache_bytes,"
           "bie_cartesian_full_fmm_plan_bytes,"
           "bie_cartesian_full_fmm_workspace_bytes,"
           "bie_cartesian_full_fmm_tree_seconds,"
           "bie_cartesian_full_fmm_worklist_seconds,"
           "bie_cartesian_full_fmm_cache_seconds,"
           "bie_cartesian_full_fmm_audit_seconds,"
           "bie_cartesian_full_fmm_p2m_seconds,"
           "bie_cartesian_full_fmm_m2m_seconds,"
           "bie_cartesian_full_fmm_m2l_seconds,"
           "bie_cartesian_full_fmm_l2l_seconds,"
           "bie_cartesian_full_fmm_l2p_seconds,"
           "bie_cartesian_full_fmm_direct_seconds,"
           "bie_cartesian_full_fmm_m2m_parity_error,"
           "bie_cartesian_full_fmm_l2l_parity_error,"
           "fmm_local_gmres_active,fmm_local_prepare_seconds,"
           "fmm_local_plan_build_count,fmm_local_cache_build_count,"
           "fmm_local_gmres_matvec_calls,"
           "fmm_local_gmres_apply_seconds_total,"
           "fmm_local_gmres_apply_seconds_mean,"
           "fmm_local_moment_seconds_total,fmm_local_m2l_seconds_total,"
           "fmm_local_fallback_seconds_total,fmm_local_l2p_seconds_total,"
           "fmm_local_sampled_direct_true_residual,"
           "fmm_local_operator_finite,"
           "fmm_local_refine_probe_mode,fmm_local_refine_probe_count,"
           "fmm_local_solution_probe_used,"
           "fmm_local_linf_max_abs_tail,"
           "fmm_local_linf_max_relative_tail,"
           "fmm_local_linf_outlier_panels,"
           "fmm_local_geometry_guard,fmm_local_geometry_theta,"
           "fmm_local_geometry_guard_rejected_cells,"
           "fmm_local_geometry_guard_direct_fallback_pairs,"
           "fmm_local_geometry_gap_min,fmm_local_geometry_ratio_max,"
           "status,convergence_reason,reaction_energy_source,"
           "iterations,residual,residual_history,gmres_restart_requested,"
           "gmres_restart_effective,best_residual,best_residual_iter,"
           "final_residual,best_preconditioned_residual,"
           "final_preconditioned_residual,true_residual_abs,"
           "true_residual_rel,true_residual_probe_mode,"
           "true_residual_probe_scope,true_residual_probe_interval,"
           "true_residual_sample_count,true_residual_probe_calls,"
           "true_residual_probe_seconds,sampled_true_residual_rel,"
           "sampled_true_residual_rows,sampled_probe_calls,"
           "sampled_probe_seconds,full_true_residual_confirm_calls,"
           "full_probe_seconds,total_seconds,"
           "surface_build_seconds,surface_neighbor_seconds,"
           "surface_mesh_seconds,solid_angle_seconds,rhs_seconds,"
           "gmres_seconds,matvec_seconds_total,matvec_seconds_avg,"
           "matvec_calls,tree_vs_direct_matvec_rel,matvec_audit_rows,"
           "matvec_audit_seconds,orthogonalization_seconds,"
           "preconditioner_seconds,"
           "tree_build_seconds,tree_moment_seconds,tree_traversal_seconds,"
           "matvec_cost_profile,leaf_panels_per_second,"
           "clean_leaf_panels_per_second,tree_traversal_fraction,"
           "tree_eval_seconds,near_eval_seconds,far_eval_seconds,"
           "singular_near_eval_seconds,leaf_near_eval_seconds,"
           "leaf_panel_eval_seconds,leaf_tile_eval_seconds,"
           "leaf_residual_eval_seconds,smooth_leaf_node_eval_seconds,"
           "leaf_panel_kernel,leaf_tile_status,leaf_panel_pairs_per_second,"
           "leaf_near_policy,cluster_policy,target_batch_size,"
           "leaf_pair_min_source_panels,leaf_pair_min_targets,"
           "leaf_pair_selection_profile,leaf_pair_probe_cache,"
           "leaf_pair_target_group,leaf_pair_max_lookahead,"
           "leaf_pair_directness_probe,leaf_pair_directness_cache,"
           "leaf_pair_directness_verify,"
           "leaf_pair_scheduler,leaf_pair_worklist_max_active,"
           "leaf_pair_worklist_verify,leaf_block_verify,"
           "tree_preset,interaction_mode,node_bounds_mode,"
           "tree_builder,expansion_center,fragmentation_safety,"
           "mac_safety,mac_gap_factor,tree_depth_max,"
           "mac_reject_radius_limited,"
           "mac_accept_rate,node_radius_max,node_radius_mean,"
           "node_aspect_ratio_max,node_aspect_ratio_mean,"
           "leaf_panel_work_per_matvec,far_node_work_per_matvec,"
           "far_geometric_unsafe_count,far_geometric_unsafe_panels,"
           "far_geometric_min_gap,far_geometric_gap_ratio_min,"
           "far_geometric_gap_ratio_mean,far_geometric_audit_samples,"
           "mac_physical_gap_rejects,mac_convergence_rho_rejects,"
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
           "fragmented_far_count,"
           "fragmented_far_panels,fragmented_split_rejects,"
           "fragmentation_center_gap_ratio_max,"
           "fragmentation_center_gap_ratio_mean,"
           "moment_audit,moment_construction,moment_sum,"
           "convergence_audit,convergence_safety,convergence_rho_cap,"
           "mac_aspect_cap,convergence_estimator,"
           "convergence_tail_threshold,convergence_estimator_error_cap,"
           "convergence_unstable_rho_cap,convergence_calibration,"
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
           "convergence_tail_norm_p_max,convergence_tail_norm_p_mean,"
           "convergence_tail_norm_pm1_max,convergence_tail_norm_pm1_mean,"
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
           "moment_audit_nodes,"
           "moment_translation_rel_error_max,"
           "moment_translation_rel_error_mean,"
           "moment_translation_offset_max,moment_translation_offset_mean,"
           "moment_cancellation_ratio_max,moment_cancellation_ratio_mean,"
           "moment_dynamic_range_max,"
           "expansion_audit,expansion_audit_channel,"
           "far_expansion_audit_pairs,"
           "far_expansion_audit_panels,far_expansion_direct_seconds,"
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
           "accepted_far_aspect_ratio_max,accepted_far_aspect_ratio_mean,"
           "quad_buffer_auto_candidate,"
           "near_integration,quad_buffer,near_self_pairs,near_one_ring_pairs,"
           "near_duffy_pairs,near_centroid_pairs,near_fallbacks,"
           "near_geometric_buffer_pairs,near_geometric_buffer_gauss_pairs,"
           "near_geometric_buffer_adaptive_pairs,"
           "near_geometric_buffer_centroid_pairs,"
           "near_geometric_buffer_min_gap,"
           "near_geometric_buffer_mean_gap_ratio,"
           "near_geometric_buffer_max_edge_mean,"
           "near_integration_seconds,duffy_subdivisions_max,"
           "duffy_subdivisions_mean,near_kernel_policy,"
           "near_k1_accepted,near_k1_rejected,near_k2_accepted,"
           "near_k2_rejected,near_k3_accepted,near_k3_rejected,"
           "near_k4_accepted,near_k4_rejected,"
           "near_one_ring_k1_accepted,near_one_ring_k1_rejected,"
           "near_one_ring_k2_accepted,near_one_ring_k2_rejected,"
           "near_one_ring_k3_accepted,near_one_ring_k3_rejected,"
           "near_one_ring_k4_accepted,near_one_ring_k4_rejected,"
           "near_k1_policy,"
           "near_k2_policy,near_k3_policy,near_k4_policy,"
           "near_interactions,far_interactions,singular_near_interactions,"
           "leaf_near_interactions,far_node_interactions,"
           "leaf_panel_interactions,leaf_tile_interactions,"
           "leaf_tile_panel_interactions,leaf_residual_panel_interactions,"
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
           "smooth_leaf_node_interactions,"
           "leaf_pruned_interactions,leaf_mac_candidates,"
           "leaf_mac_accepted,leaf_mac_rejected,leaf_mac_far_panels,"
           "leaf_mac_direct_panels_saved_estimate,"
           "adaptive_smooth_leaf_accepted,"
           "adaptive_smooth_leaf_rejected,adaptive_span_tile_accepted,"
           "adaptive_span_tile_rejected,adaptive_exact_leaf_fallbacks,"
           "adaptive_reject_one_ring_overlap,"
           "adaptive_reject_error_bound_failed,"
           "adaptive_reject_p4_check_failed,adaptive_reject_too_close,"
           "leaf_work_reduction_ratio,target_cluster_interactions,"
           "target_cluster_targets,target_cluster_panels_pruned,"
           "target_cluster_rejected,target_cluster_eval_seconds,"
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
           "leaf_pair_worklist_tasks,leaf_pair_worklist_groups,"
           "leaf_pair_worklist_batched_groups,"
           "leaf_pair_worklist_scalar_flush_tasks,"
           "leaf_pair_worklist_forced_flushes,"
           "leaf_pair_worklist_active_groups_max,"
           "leaf_pair_worklist_seconds,"
           "leaf_pair_worklist_verify_checks,"
           "leaf_pair_worklist_verify_mismatches,"
           "leaf_block_target_leaves,leaf_block_targets,"
           "leaf_block_source_groups,leaf_block_batched_groups,"
           "leaf_block_scalar_tail_tasks,leaf_block_group_seconds,"
           "leaf_block_flush_seconds,leaf_block_writeback_seconds,"
           "leaf_block_verify_checks,leaf_block_verify_mismatches,"
           "adaptive_level,"
           "tree_nodes,tree_leaf_count,"
           "taylor_order,derivative_mode,diag_min,diag_max,solid_angle_"
           "fallbacks,"
           "nonmanifold_edges,formulation,energy_projection,"
           "energy_projection_seconds,energy_projection_kernel,"
           "fallback_used,"
           "preconditioner_requested,preconditioner_effective,"
           "preconditioner_block_near,"
           "preconditioner_block_panel_cap,preconditioner_overlap_panel_cap,"
           "preconditioner_ilu_neighbor_cap_requested,"
           "preconditioner_ilu_neighbor_cap,preconditioner_ilu_drop_tol,"
           "preconditioner_ilu_regularization,"
           "preconditioner_ilu_regularization_max,"
           "preconditioner_ilu_regularization_selected,"
           "preconditioner_ilu_regularization_absolute_max,"
           "preconditioner_ilu_regularization_attempts,"
           "preconditioner_ilu_regularization_rejected_attempts,"
           "preconditioner_ilu_regularization_shifted_rows,"
           "preconditioner_ilu_unshifted_failure_stage,"
           "preconditioner_memory_budget_bytes,"
           "preconditioner_estimated_bytes,preconditioner_valid_blocks,"
           "preconditioner_valid_panels,preconditioner_ill_conditioned_blocks,"
           "preconditioner_quality_rejected_blocks,"
           "preconditioner_quality_probe_max_gain,"
           "preconditioner_quality_probe_rel_residual,"
           "preconditioner_pivot_ratio_min,preconditioner_factor_growth_max,"
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
           "preconditioner_coarse_aggregate_cap,preconditioner_coarse_"
           "aggregates,"
           "preconditioner_coarse_dimension,preconditioner_coarse_base,"
           "preconditioner_coarse_matrix_bytes,preconditioner_coarse_pivot_"
           "ratio,"
           "preconditioner_coarse_status,preconditioner_coarse_setup_matvec_"
           "calls,"
           "preconditioner_coarse_setup_seconds,"
           "preconditioner_coarse_moments_setup_seconds,"
           "preconditioner_coarse_far_setup_seconds,"
           "preconditioner_coarse_audit_setup_seconds,"
           "preconditioner_coarse_apply_seconds,"
           "preconditioner_coarse_apply_calls,"
           "preconditioner_coarse_rp_identity_error_max,"
           "preconditioner_coarse_verify_rel_error_max,"
           "surface_generator,discretization,node_patch_representation,"
           "node_patch_normal_source,node_patch_certificate,"
           "node_patch_certificate_receptor,node_patch_certificate_ligand,"
           "rhs_scale,dielectric_contrast,operator_contrast,unknown_count,"
           "bie_system,lhs_diag_phi,lhs_diag_phin,energy_equation,"
           "kernel_convention,source_norm_s1,source_norm_s2,phi_norm,"
           "phin_norm,kappa_inv_A,"
           "boundary_energy_kJmol,reference_energy_kJmol,"
           "selected_energy_scale,memory_bytes,energy_kJmol,"
           "bie_cartesian_full_fmm_tree_hash,"
           "bie_cartesian_full_fmm_worklist_hash,"
           "bie_cartesian_full_fmm_policy_hash,"
           "fmm_cartesian_pre_audit_mode,fmm_cartesian_pre_audit_enabled,"
           "fmm_cartesian_pre_audit_finite,"
           "fmm_cartesian_pre_audit_aggregate_rel,"
           "fmm_cartesian_pre_audit_max_stratum_rel,"
           "fmm_cartesian_pre_audit_targets,"
           "fmm_cartesian_pre_audit_rows,"
           "fmm_cartesian_pre_audit_stages,"
           "fmm_cartesian_pre_audit_expanded_to_256,"
           "fmm_cartesian_pre_audit_expanded_to_512,"
           "fmm_cartesian_pre_audit_seconds,"
           "fmm_cartesian_channel_shadow_seconds,"
           "fmm_cartesian_pre_audit_probe_hash,"
           "fmm_cartesian_channel_probe_hash,"
           "fmm_cartesian_pre_audit_target_hash,"
           "fmm_cartesian_channel_direct_pairs_expected,"
           "fmm_cartesian_channel_direct_pairs_applied,"
           "fmm_cartesian_solution_audit_enabled,"
           "fmm_cartesian_solution_audit_finite,"
           "fmm_cartesian_solution_audit_relative_available,"
           "fmm_cartesian_solution_audit_status,"
           "fmm_cartesian_solution_audit_rel,"
           "fmm_cartesian_solution_audit_abs_l2,"
           "fmm_cartesian_solution_audit_direct_l2,"
           "fmm_cartesian_solution_audit_targets,"
           "fmm_cartesian_solution_audit_rows,"
           "fmm_cartesian_solution_audit_seconds,"
           "fmm_cartesian_solution_audit_target_hash,"
           "fmm_cartesian_solution_audit_direct_pairs_expected,"
           "fmm_cartesian_solution_audit_direct_pairs_applied,"
           "fmm_cartesian_channel_k1_available,"
           "fmm_cartesian_channel_k1_finite,"
           "fmm_cartesian_channel_k1_status,"
           "fmm_cartesian_channel_k1_samples,"
           "fmm_cartesian_channel_k1_abs_l2,"
           "fmm_cartesian_channel_k1_rms,"
           "fmm_cartesian_channel_k1_reference_l2,"
           "fmm_cartesian_channel_k1_rel,"
           "fmm_cartesian_channel_k2_available,"
           "fmm_cartesian_channel_k2_finite,"
           "fmm_cartesian_channel_k2_status,"
           "fmm_cartesian_channel_k2_samples,"
           "fmm_cartesian_channel_k2_abs_l2,"
           "fmm_cartesian_channel_k2_rms,"
           "fmm_cartesian_channel_k2_reference_l2,"
           "fmm_cartesian_channel_k2_rel,"
           "fmm_cartesian_channel_k3_available,"
           "fmm_cartesian_channel_k3_finite,"
           "fmm_cartesian_channel_k3_status,"
           "fmm_cartesian_channel_k3_samples,"
           "fmm_cartesian_channel_k3_abs_l2,"
           "fmm_cartesian_channel_k3_rms,"
           "fmm_cartesian_channel_k3_reference_l2,"
           "fmm_cartesian_channel_k3_rel,"
           "fmm_cartesian_channel_k4_available,"
           "fmm_cartesian_channel_k4_finite,"
           "fmm_cartesian_channel_k4_status,"
           "fmm_cartesian_channel_k4_samples,"
           "fmm_cartesian_channel_k4_abs_l2,"
           "fmm_cartesian_channel_k4_rms,"
           "fmm_cartesian_channel_k4_reference_l2,"
           "fmm_cartesian_channel_k4_rel,";
    tabi_profile_out_
        << "fmm_cartesian_retry_mode,fmm_cartesian_retry_enabled,"
           "fmm_cartesian_retry_attempts_started,"
           "fmm_cartesian_retry_attempts_completed,"
           "fmm_cartesian_retry_accepted_attempt,"
           "fmm_cartesian_retry_trigger_reason,"
           "fmm_cartesian_retry_final_reason,"
           "fmm_cartesian_retry_geometry_identity_valid,"
           "fmm_cartesian_retry_tree_identity_valid,"
           "fmm_cartesian_retry_policy_transition_valid,"
           "fmm_cartesian_retry_cumulative_wall_seconds,"
           "fmm_cartesian_retry_cumulative_gmres_seconds,"
           "fmm_cartesian_retry_cumulative_accounted_memory_bytes,"
           "fmm_cartesian_retry_max_boundary_rss_bytes,";
    for (int attempt_index = 1; attempt_index <= 2; ++attempt_index) {
      const std::string prefix =
          "fmm_cartesian_retry_attempt" + std::to_string(attempt_index) +
          "_";
      tabi_profile_out_
          << prefix << "number," << prefix << "started," << prefix
          << "completed," << prefix << "accepted," << prefix
          << "technically_valid," << prefix << "classification," << prefix
          << "classification_reason," << prefix << "status," << prefix
          << "convergence_reason," << prefix << "converged," << prefix
          << "finite," << prefix << "theta," << prefix << "salt_molar,"
          << prefix << "coupled," << prefix << "discretization," << prefix
          << "node_patch_operator_audit_enabled," << prefix
          << "true_residual_rel," << prefix << "cartesian_full_active,"
          << prefix << "cartesian_full_finite," << prefix
          << "cartesian_full_requested," << prefix
          << "cartesian_full_effective," << prefix << "cartesian_full_order,"
          << prefix << "cartesian_full_leaf_size," << prefix
          << "panel_count," << prefix
          << "geometry_hash," << prefix << "tree_hash," << prefix
          << "worklist_hash," << prefix << "policy_hash," << prefix
          << "direct_pairs_expected," << prefix
          << "direct_pairs_applied_last," << prefix
          << "direct_pairs_applied_total," << prefix
          << "pre_audit_target_hash," << prefix
          << "solution_audit_enabled," << prefix << "solution_audit_finite,"
          << prefix << "solution_audit_relative_available," << prefix
          << "solution_audit_status," << prefix << "solution_audit_rel,"
          << prefix << "solution_audit_abs_l2," << prefix
          << "solution_audit_direct_l2," << prefix
          << "solution_audit_seconds," << prefix << "solution_audit_targets,"
          << prefix
          << "solution_audit_rows," << prefix << "solution_audit_target_hash,"
          << prefix << "solution_audit_direct_pairs_expected," << prefix
          << "solution_audit_direct_pairs_applied," << prefix
          << "fallback_used," << prefix << "preconditioner_requested,"
          << prefix << "preconditioner_effective," << prefix
          << "preconditioner_quality_status," << prefix
          << "preconditioner_coarse_status," << prefix
          << "preconditioner_ilu_regularization," << prefix
          << "preconditioner_ilu_regularization_selected," << prefix
          << "preconditioner_ilu_regularization_shifted_rows," << prefix
          << "preconditioner_ilu_unshifted_failure_stage," << prefix
          << "preconditioner_valid_panels," << prefix
          << "preconditioner_ill_conditioned_blocks," << prefix
          << "preconditioner_quality_rejected_blocks," << prefix
          << "preconditioner_setup_seconds," << prefix
          << "preconditioner_ilu_factor_setup_seconds," << prefix
          << "surface_build_seconds," << prefix << "fmm_prepare_seconds,"
          << prefix << "gmres_seconds," << prefix << "wall_seconds,"
          << prefix << "accounted_memory_bytes," << prefix
          << "rss_before_bytes," << prefix << "rss_after_bytes";
      tabi_profile_out_ << (attempt_index == 1 ? ',' : '\n');
    }
    tabi_profile_header_written_ = true;
  }
}

void AnalyzerImpl::append_tabi_profile_row(const FrameResult &result,
                                           const TimingBuckets &timing) {
  if (args_.timing != "detail" || result.pbe_backend != "tabi")
    return;
  write_tabi_profile_header_if_needed();
  auto seconds = [](double ms) { return ms / 1000.0; };
  const FmmCartesianPreAuditTelemetry &pre_audit =
      result.tabi_fmm_cartesian_pre_audit;
  const FmmCartesianSolutionAuditTelemetry &solution_audit =
      result.tabi_fmm_cartesian_solution_audit;
  const auto &channel_audit = result.tabi_fmm_cartesian_channel_audit;
  const FmmCartesianRetryTelemetry &retry = result.tabi_fmm_cartesian_retry;
  const auto write_retry_attempt =
      [&](const FmmCartesianRetryAttemptTelemetry &attempt) {
        const FmmCartesianSolutionAuditTelemetry &attempt_audit =
            attempt.solution_audit;
        tabi_profile_out_
            << attempt.attempt << ',' << (attempt.started ? "true" : "false")
            << ',' << (attempt.completed ? "true" : "false") << ','
            << (attempt.accepted ? "true" : "false") << ','
            << (attempt.technically_valid ? "true" : "false") << ','
            << csv_escape(attempt.classification) << ','
            << csv_escape(attempt.classification_reason) << ','
            << csv_escape(attempt.status) << ','
            << csv_escape(attempt.convergence_reason) << ','
            << (attempt.converged ? "true" : "false") << ','
            << (attempt.finite ? "true" : "false") << ',' << attempt.theta
            << ',' << attempt.salt_molar << ','
            << (attempt.coupled ? "true" : "false") << ','
            << csv_escape(attempt.discretization) << ','
            << (attempt.node_patch_operator_audit_enabled ? "true" : "false")
            << ',' << attempt.true_residual_rel << ','
            << (attempt.cartesian_full_active ? "true" : "false") << ','
            << (attempt.cartesian_full_finite ? "true" : "false") << ','
            << csv_escape(attempt.cartesian_full_requested) << ','
            << csv_escape(attempt.cartesian_full_effective) << ','
            << attempt.cartesian_full_order << ','
            << attempt.cartesian_full_leaf_size << ',' << attempt.panel_count
            << ',' << attempt.geometry_hash << ',' << attempt.tree_hash << ','
            << attempt.worklist_hash << ',' << attempt.policy_hash << ','
            << attempt.direct_pairs_expected << ','
            << attempt.direct_pairs_applied_last << ','
            << attempt.direct_pairs_applied_total << ','
            << attempt.pre_audit_sampled_target_hash << ','
            << (attempt_audit.enabled ? "true" : "false") << ','
            << (attempt_audit.finite ? "true" : "false") << ','
            << (attempt_audit.relative_available ? "true" : "false") << ','
            << csv_escape(attempt_audit.status) << ','
            << attempt_audit.relative_error << ',' << attempt_audit.absolute_l2
            << ',' << attempt_audit.direct_l2 << ','
            << seconds(attempt_audit.seconds) << ',' << attempt_audit.targets
            << ',' << attempt_audit.rows << ',' << attempt_audit.target_hash << ','
            << attempt_audit.direct_pairs_expected << ','
            << attempt_audit.direct_pairs_applied << ','
            << (attempt.fallback_used ? "true" : "false") << ','
            << csv_escape(attempt.preconditioner_requested) << ','
            << csv_escape(attempt.preconditioner_effective) << ','
            << csv_escape(attempt.preconditioner_quality_status) << ','
            << csv_escape(attempt.preconditioner_coarse_status) << ','
            << csv_escape(attempt.preconditioner_ilu_regularization) << ','
            << attempt.preconditioner_ilu_regularization_selected << ','
            << attempt.preconditioner_ilu_regularization_shifted_rows << ','
            << csv_escape(attempt.preconditioner_ilu_unshifted_failure_stage)
            << ',' << attempt.preconditioner_valid_panels << ','
            << attempt.preconditioner_ill_conditioned_blocks << ','
            << attempt.preconditioner_quality_rejected_blocks << ','
            << seconds(attempt.preconditioner_setup_ms) << ','
            << seconds(attempt.preconditioner_ilu_factor_setup_ms) << ','
            << seconds(attempt.surface_build_ms) << ','
            << seconds(attempt.fmm_prepare_ms) << ','
            << seconds(attempt.gmres_ms) << ',' << seconds(attempt.wall_ms)
            << ',' << attempt.accounted_memory_bytes << ','
            << attempt.rss_before_bytes << ',' << attempt.rss_after_bytes;
      };
  tabi_profile_out_
      << result.trajectory_id << ',' << result.frame_index << ','
      << result.global_index << ',' << csv_escape(current_source_label_) << ','
      << result.atom_count << ',' << csv_escape(result.tabi_surface_geometry)
      << ',' << result.tabi_surface_panels << ',' << result.tabi_mesh_vertices
      << ',' << result.tabi_mesh_triangles << ','
      << result.tabi_surface_point_count_requested << ','
      << result.tabi_panel_density_per_a2 << ','
      << csv_escape(result.tabi_panel_budget_status) << ','
      << csv_escape(result.tabi_matvec_mode) << ','
      << csv_escape(result.tabi_fmm_m2l_mode) << ','
      << result.tabi_fmm_m2l_expansion_order << ','
      << result.tabi_fmm_m2l_near_pairs << ',' << result.tabi_fmm_m2l_far_pairs
      << ',' << seconds(result.tabi_fmm_m2l_setup_ms) << ','
      << seconds(result.tabi_fmm_m2l_apply_ms) << ','
      << (result.tabi_fmm_m2l_finite_outputs ? "true" : "false") << ','
      << (result.tabi_bie_cartesian_full_fmm_active ? "true" : "false")
      << ',' << csv_escape(result.tabi_bie_cartesian_full_fmm_requested) << ','
      << csv_escape(result.tabi_bie_cartesian_full_fmm_effective) << ','
      << result.tabi_bie_cartesian_full_fmm_order << ','
      << result.tabi_bie_cartesian_full_fmm_leaf_size << ','
      << result.tabi_bie_cartesian_full_fmm_opening_theta << ','
      << result.tabi_bie_cartesian_full_fmm_nodes << ','
      << result.tabi_bie_cartesian_full_fmm_far_pairs << ','
      << result.tabi_bie_cartesian_full_fmm_direct_pairs << ','
      << result.tabi_bie_cartesian_full_fmm_direct_panel_pairs_expected << ','
      << result.tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_last
      << ','
      << result.tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_total
      << ','
      << result.tabi_bie_cartesian_full_fmm_cache_entries << ','
      << result.tabi_bie_cartesian_full_fmm_cache_hits << ','
      << result.tabi_bie_cartesian_full_fmm_cache_bytes << ','
      << result.tabi_bie_cartesian_full_fmm_plan_bytes << ','
      << result.tabi_bie_cartesian_full_fmm_workspace_bytes << ','
      << result.tabi_bie_cartesian_full_fmm_tree_seconds << ','
      << result.tabi_bie_cartesian_full_fmm_worklist_seconds << ','
      << result.tabi_bie_cartesian_full_fmm_cache_seconds << ','
      << result.tabi_bie_cartesian_full_fmm_audit_seconds << ','
      << result.tabi_bie_cartesian_full_fmm_p2m_seconds << ','
      << result.tabi_bie_cartesian_full_fmm_m2m_seconds << ','
      << result.tabi_bie_cartesian_full_fmm_m2l_seconds << ','
      << result.tabi_bie_cartesian_full_fmm_l2l_seconds << ','
      << result.tabi_bie_cartesian_full_fmm_l2p_seconds << ','
      << result.tabi_bie_cartesian_full_fmm_direct_seconds << ','
      << result.tabi_bie_cartesian_full_fmm_m2m_parity_error << ','
      << result.tabi_bie_cartesian_full_fmm_l2l_parity_error << ','
      << (result.tabi_fmm_local_gmres_active ? "true" : "false") << ','
      << result.tabi_fmm_local_prepare_seconds << ','
      << result.tabi_fmm_local_plan_build_count << ','
      << result.tabi_fmm_local_cache_build_count << ','
      << result.tabi_fmm_local_gmres_matvec_calls << ','
      << result.tabi_fmm_local_gmres_apply_seconds_total << ','
      << result.tabi_fmm_local_gmres_apply_seconds_mean << ','
      << result.tabi_fmm_local_moment_seconds_total << ','
      << result.tabi_fmm_local_m2l_seconds_total << ','
      << result.tabi_fmm_local_fallback_seconds_total << ','
      << result.tabi_fmm_local_l2p_seconds_total << ','
      << result.tabi_fmm_local_sampled_direct_true_residual << ','
      << (result.tabi_fmm_local_operator_finite ? "true" : "false") << ','
      << csv_escape(result.tabi_fmm_local_refine_probe_mode) << ','
      << result.tabi_fmm_local_refine_probe_count << ','
      << (result.tabi_fmm_local_solution_probe_used ? "true" : "false") << ','
      << result.tabi_fmm_local_linf_max_abs_tail << ','
      << result.tabi_fmm_local_linf_max_relative_tail << ','
      << result.tabi_fmm_local_linf_outlier_panels << ','
      << csv_escape(result.tabi_fmm_local_geometry_guard) << ','
      << result.tabi_fmm_local_geometry_theta << ','
      << result.tabi_fmm_local_geometry_guard_rejected_cells << ','
      << result.tabi_fmm_local_geometry_guard_direct_fallback_pairs << ','
      << result.tabi_fmm_local_geometry_gap_min << ','
      << result.tabi_fmm_local_geometry_ratio_max << ','
      << csv_escape(result.tabi_status) << ','
      << csv_escape(result.tabi_convergence_reason) << ','
      << csv_escape(result.tabi_reaction_energy_source) << ','
      << result.solvent_iterations << ',' << result.linear_residual << ','
      << csv_escape(result.tabi_residual_history) << ','
      << result.tabi_gmres_restart_requested << ','
      << result.tabi_gmres_restart_effective << ',' << result.tabi_best_residual
      << ',' << result.tabi_best_residual_iter << ','
      << result.tabi_final_residual << ','
      << result.tabi_best_preconditioned_residual << ','
      << result.tabi_final_preconditioned_residual << ','
      << result.tabi_true_residual_abs << ',' << result.tabi_true_residual_rel
      << ',' << csv_escape(result.tabi_true_residual_probe_mode) << ','
      << csv_escape(result.tabi_true_residual_probe_scope) << ','
      << result.tabi_true_residual_probe_interval << ','
      << result.tabi_true_residual_sample_count << ','
      << result.tabi_true_residual_probe_calls << ','
      << seconds(result.tabi_true_residual_probe_ms) << ','
      << result.tabi_sampled_true_residual_rel << ','
      << result.tabi_sampled_true_residual_rows << ','
      << result.tabi_sampled_probe_calls << ','
      << seconds(result.tabi_sampled_probe_ms) << ','
      << result.tabi_full_true_residual_confirm_calls << ','
      << seconds(result.tabi_full_probe_ms) << ',' << seconds(timing.total_ms)
      << ',' << seconds(result.tabi_surface_build_ms) << ','
      << seconds(result.tabi_surface_neighbor_ms) << ','
      << seconds(result.tabi_surface_mesh_ms) << ','
      << seconds(result.tabi_solid_angle_ms) << ','
      << seconds(result.tabi_rhs_ms) << ',' << seconds(result.tabi_gmres_ms)
      << ',' << seconds(result.tabi_matvec_ms_total) << ','
      << seconds(result.tabi_matvec_ms_avg) << ',' << result.tabi_matvec_calls
      << ',' << result.tabi_tree_vs_direct_matvec_rel << ','
      << result.tabi_matvec_audit_rows << ','
      << seconds(result.tabi_matvec_audit_ms) << ','
      << seconds(result.tabi_orthogonalization_ms) << ','
      << seconds(result.tabi_preconditioner_ms) << ','
      << seconds(result.tabi_tree_build_ms) << ','
      << seconds(result.tabi_tree_moment_ms) << ','
      << seconds(result.tabi_tree_traversal_ms) << ','
      << csv_escape(result.tabi_matvec_cost_profile) << ','
      << result.tabi_leaf_panels_per_second << ','
      << result.tabi_clean_leaf_panels_per_second << ','
      << result.tabi_tree_traversal_fraction << ','
      << seconds(result.tabi_tree_eval_ms) << ','
      << seconds(result.tabi_near_eval_ms) << ','
      << seconds(result.tabi_far_eval_ms) << ','
      << seconds(result.tabi_singular_near_eval_ms) << ','
      << seconds(result.tabi_leaf_near_eval_ms) << ','
      << seconds(result.tabi_leaf_panel_eval_ms) << ','
      << seconds(result.tabi_leaf_tile_eval_ms) << ','
      << seconds(result.tabi_leaf_residual_eval_ms) << ','
      << seconds(result.tabi_smooth_leaf_node_eval_ms) << ','
      << csv_escape(result.tabi_leaf_panel_kernel) << ','
      << csv_escape(result.tabi_leaf_tile_status) << ','
      << result.tabi_leaf_panel_pairs_per_second << ','
      << csv_escape(result.tabi_leaf_near_policy) << ','
      << csv_escape(result.tabi_cluster_policy) << ','
      << result.tabi_target_batch_size << ','
      << result.tabi_leaf_pair_min_source_panels << ','
      << result.tabi_leaf_pair_min_targets << ','
      << csv_escape(result.tabi_leaf_pair_selection_profile) << ','
      << csv_escape(result.tabi_leaf_pair_probe_cache) << ','
      << csv_escape(result.tabi_leaf_pair_target_group) << ','
      << result.tabi_leaf_pair_max_lookahead << ','
      << csv_escape(result.tabi_leaf_pair_directness_probe) << ','
      << csv_escape(result.tabi_leaf_pair_directness_cache) << ','
      << csv_escape(result.tabi_leaf_pair_directness_verify) << ','
      << csv_escape(result.tabi_leaf_pair_scheduler) << ','
      << result.tabi_leaf_pair_worklist_max_active << ','
      << csv_escape(result.tabi_leaf_pair_worklist_verify) << ','
      << csv_escape(result.tabi_leaf_block_verify) << ','
      << csv_escape(result.tabi_tree_preset) << ','
      << csv_escape(result.tabi_interaction_mode) << ','
      << csv_escape(result.tabi_node_bounds_mode) << ','
      << csv_escape(result.tabi_tree_builder) << ','
      << csv_escape(result.tabi_expansion_center) << ','
      << csv_escape(result.tabi_fragmentation_safety) << ','
      << csv_escape(result.tabi_mac_safety) << ',' << result.tabi_mac_gap_factor
      << ',' << result.tabi_tree_depth_max << ','
      << result.tabi_mac_reject_radius_limited << ','
      << result.tabi_mac_accept_rate << ',' << result.tabi_node_radius_max
      << ',' << result.tabi_node_radius_mean << ','
      << result.tabi_node_aspect_ratio_max << ','
      << result.tabi_node_aspect_ratio_mean << ','
      << result.tabi_leaf_panel_work_per_matvec << ','
      << result.tabi_far_node_work_per_matvec << ','
      << result.tabi_far_geometric_unsafe_count << ','
      << result.tabi_far_geometric_unsafe_panels << ','
      << result.tabi_far_geometric_min_gap << ','
      << result.tabi_far_geometric_gap_ratio_min << ','
      << result.tabi_far_geometric_gap_ratio_mean << ','
      << result.tabi_far_geometric_audit_samples << ','
      << result.tabi_mac_physical_gap_rejects << ','
      << result.tabi_mac_convergence_rho_rejects << ','
      << result.tabi_mac_convergence_aspect_rejects << ','
      << result.tabi_mac_convergence_rho_reject_panels << ','
      << result.tabi_mac_convergence_aspect_reject_panels << ','
      << result.tabi_mac_surface_margin_rejects << ','
      << result.tabi_mac_surface_margin_reject_panels << ','
      << result.tabi_mac_surface_margin_applied << ','
      << result.tabi_mac_surface_margin_skipped << ','
      << result.tabi_mac_surface_margin_adaptive_rejects << ','
      << result.tabi_mac_surface_margin_adaptive_reject_panels << ','
      << result.tabi_mac_surface_margin_skipped_accepted_far << ','
      << result.tabi_mac_surface_margin_hard_applied << ','
      << result.tabi_mac_surface_margin_soft_applied << ','
      << result.tabi_mac_surface_margin_hard_rejects << ','
      << result.tabi_mac_surface_margin_hard_reject_panels << ','
      << result.tabi_mac_surface_margin_soft_rejects << ','
      << result.tabi_mac_surface_margin_soft_reject_panels << ','
      << result.tabi_fragmented_far_count << ','
      << result.tabi_fragmented_far_panels << ','
      << result.tabi_fragmented_split_rejects << ','
      << result.tabi_fragmentation_center_gap_ratio_max << ','
      << result.tabi_fragmentation_center_gap_ratio_mean << ','
      << csv_escape(result.tabi_moment_audit) << ','
      << csv_escape(result.tabi_moment_construction) << ','
      << csv_escape(result.tabi_moment_sum) << ','
      << csv_escape(result.tabi_convergence_audit) << ','
      << csv_escape(result.tabi_convergence_safety) << ','
      << result.tabi_convergence_rho_cap << ',' << result.tabi_mac_aspect_cap
      << ',' << csv_escape(result.tabi_convergence_estimator) << ','
      << result.tabi_convergence_tail_threshold << ','
      << result.tabi_convergence_estimator_error_cap << ','
      << result.tabi_convergence_unstable_rho_cap << ','
      << csv_escape(result.tabi_convergence_calibration) << ','
      << csv_escape(result.tabi_convergence_dielectric_amp) << ','
      << csv_escape(result.tabi_convergence_estimator_model) << ','
      << csv_escape(result.tabi_convergence_channel_calibration) << ','
      << csv_escape(result.tabi_convergence_split_policy) << ','
      << csv_escape(result.tabi_convergence_split_selectivity) << ','
      << result.tabi_convergence_split_absolute_floor << ','
      << csv_escape(result.tabi_convergence_overreject_audit) << ','
      << csv_escape(result.tabi_convergence_overreject_detail) << ','
      << csv_escape(result.tabi_convergence_shape_audit) << ','
      << csv_escape(result.tabi_convergence_kernel_response_audit) << ','
      << csv_escape(result.tabi_channel_missed_semantics) << ','
      << csv_escape(result.tabi_accepted_miss_diagnostics) << ','
      << result.tabi_accepted_miss_absolute_floor << ','
      << result.tabi_convergence_k3_accepted_rho_cap << ','
      << csv_escape(result.tabi_mac_distance_policy) << ','
      << result.tabi_mac_surface_margin_factor << ','
      << csv_escape(result.tabi_mac_surface_margin_policy) << ','
      << result.tabi_mac_surface_margin_aspect_threshold << ','
      << result.tabi_mac_surface_margin_rho_trigger << ','
      << result.tabi_mac_surface_margin_soft_aspect_threshold << ','
      << result.tabi_mac_surface_margin_soft_rho_trigger << ','
      << result.tabi_mac_surface_margin_soft_factor << ','
      << result.tabi_convergence_estimator_samples << ','
      << result.tabi_convergence_estimator_unstable_far_count << ','
      << result.tabi_convergence_estimator_unstable_far_panels << ','
      << result.tabi_convergence_estimator_split_rejects << ','
      << result.tabi_convergence_estimator_split_reject_panels << ','
      << result.tabi_convergence_low_abs_reaccepted_far_count << ','
      << result.tabi_convergence_low_abs_reaccepted_far_panels << ','
      << result.tabi_convergence_estimator_tail_ratio_max << ','
      << result.tabi_convergence_estimator_tail_ratio_mean << ','
      << result.tabi_convergence_estimator_error_max << ','
      << result.tabi_convergence_estimator_error_mean << ','
      << result.tabi_convergence_estimator_abs_error_max << ','
      << result.tabi_convergence_estimator_abs_error_mean << ','
      << result.tabi_convergence_tail_norm_p_max << ','
      << result.tabi_convergence_tail_norm_p_mean << ','
      << result.tabi_convergence_tail_norm_pm1_max << ','
      << result.tabi_convergence_tail_norm_pm1_mean << ','
      << result.tabi_convergence_effective_jcp_amp_max << ','
      << result.tabi_convergence_effective_jcp_amp_mean << ','
      << result.tabi_convergence_dielectric_amp_max << ','
      << result.tabi_convergence_dielectric_amp_mean << ','
      << result.tabi_convergence_channel_effective_jcp_amp_max[0] << ','
      << result.tabi_convergence_channel_effective_jcp_amp_max[1] << ','
      << result.tabi_convergence_channel_effective_jcp_amp_max[2] << ','
      << result.tabi_convergence_channel_effective_jcp_amp_max[3] << ','
      << result.tabi_convergence_channel_effective_jcp_amp_mean[0] << ','
      << result.tabi_convergence_channel_effective_jcp_amp_mean[1] << ','
      << result.tabi_convergence_channel_effective_jcp_amp_mean[2] << ','
      << result.tabi_convergence_channel_effective_jcp_amp_mean[3] << ','
      << result.tabi_convergence_channel_estimated_error_max[0] << ','
      << result.tabi_convergence_channel_estimated_error_max[1] << ','
      << result.tabi_convergence_channel_estimated_error_max[2] << ','
      << result.tabi_convergence_channel_estimated_error_max[3] << ','
      << result.tabi_convergence_channel_unstable_far_count[0] << ','
      << result.tabi_convergence_channel_unstable_far_count[1] << ','
      << result.tabi_convergence_channel_unstable_far_count[2] << ','
      << result.tabi_convergence_channel_unstable_far_count[3] << ','
      << result.tabi_convergence_channel_unstable_far_panels[0] << ','
      << result.tabi_convergence_channel_unstable_far_panels[1] << ','
      << result.tabi_convergence_channel_unstable_far_panels[2] << ','
      << result.tabi_convergence_channel_unstable_far_panels[3] << ','
      << result.tabi_convergence_channel_split_rejects[0] << ','
      << result.tabi_convergence_channel_split_rejects[1] << ','
      << result.tabi_convergence_channel_split_rejects[2] << ','
      << result.tabi_convergence_channel_split_rejects[3] << ','
      << result.tabi_convergence_channel_split_reject_panels[0] << ','
      << result.tabi_convergence_channel_split_reject_panels[1] << ','
      << result.tabi_convergence_channel_split_reject_panels[2] << ','
      << result.tabi_convergence_channel_split_reject_panels[3] << ','
      << result.tabi_convergence_channel_low_abs_reaccepted_count[0] << ','
      << result.tabi_convergence_channel_low_abs_reaccepted_count[1] << ','
      << result.tabi_convergence_channel_low_abs_reaccepted_count[2] << ','
      << result.tabi_convergence_channel_low_abs_reaccepted_count[3] << ','
      << result.tabi_convergence_channel_low_abs_reaccepted_panels[0] << ','
      << result.tabi_convergence_channel_low_abs_reaccepted_panels[1] << ','
      << result.tabi_convergence_channel_low_abs_reaccepted_panels[2] << ','
      << result.tabi_convergence_channel_low_abs_reaccepted_panels[3] << ','
      << result.tabi_convergence_channel_high_error_caught[0] << ','
      << result.tabi_convergence_channel_high_error_caught[1] << ','
      << result.tabi_convergence_channel_high_error_caught[2] << ','
      << result.tabi_convergence_channel_high_error_caught[3] << ','
      << result.tabi_convergence_channel_high_error_missed[0] << ','
      << result.tabi_convergence_channel_high_error_missed[1] << ','
      << result.tabi_convergence_channel_high_error_missed[2] << ','
      << result.tabi_convergence_channel_high_error_missed[3] << ','
      << result.tabi_convergence_channel_high_error_caught_by_split[0] << ','
      << result.tabi_convergence_channel_high_error_caught_by_split[1] << ','
      << result.tabi_convergence_channel_high_error_caught_by_split[2] << ','
      << result.tabi_convergence_channel_high_error_caught_by_split[3] << ','
      << result.tabi_convergence_estimator_high_error_caught << ','
      << result.tabi_convergence_estimator_high_error_missed << ','
      << result.tabi_convergence_estimator_high_error_caught_by_split << ','
      << result.tabi_convergence_estimator_phantom_high_error_count << ','
      << result.tabi_convergence_estimator_phantom_high_error_abs_max << ','
      << result.tabi_convergence_overreject_count << ','
      << result.tabi_convergence_overreject_panels << ','
      << result.tabi_convergence_overreject_caught_count << ','
      << result.tabi_convergence_overreject_caught_panels << ','
      << result.tabi_convergence_overreject_abs_error_max << ','
      << result.tabi_convergence_overreject_abs_error_mean << ','
      << result.tabi_convergence_overreject_abs_error_p95 << ','
      << result.tabi_convergence_overreject_abs_error_p99 << ','
      << result.tabi_convergence_overreject_estimated_abs_error_max << ','
      << result.tabi_convergence_overreject_estimated_abs_error_mean << ','
      << result.tabi_convergence_overreject_required_scale_max << ','
      << result.tabi_convergence_overreject_required_scale_p95 << ','
      << result.tabi_convergence_overreject_required_scale_p99 << ','
      << result.tabi_convergence_channel_overreject_count[0] << ','
      << result.tabi_convergence_channel_overreject_count[1] << ','
      << result.tabi_convergence_channel_overreject_count[2] << ','
      << result.tabi_convergence_channel_overreject_count[3] << ','
      << result.tabi_convergence_channel_overreject_caught_count[0] << ','
      << result.tabi_convergence_channel_overreject_caught_count[1] << ','
      << result.tabi_convergence_channel_overreject_caught_count[2] << ','
      << result.tabi_convergence_channel_overreject_caught_count[3] << ','
      << result.tabi_convergence_channel_overreject_panels[0] << ','
      << result.tabi_convergence_channel_overreject_panels[1] << ','
      << result.tabi_convergence_channel_overreject_panels[2] << ','
      << result.tabi_convergence_channel_overreject_panels[3] << ','
      << result.tabi_convergence_channel_overreject_abs_error_max[0] << ','
      << result.tabi_convergence_channel_overreject_abs_error_max[1] << ','
      << result.tabi_convergence_channel_overreject_abs_error_max[2] << ','
      << result.tabi_convergence_channel_overreject_abs_error_max[3] << ','
      << result.tabi_convergence_channel_overreject_abs_error_mean[0] << ','
      << result.tabi_convergence_channel_overreject_abs_error_mean[1] << ','
      << result.tabi_convergence_channel_overreject_abs_error_mean[2] << ','
      << result.tabi_convergence_channel_overreject_abs_error_mean[3] << ','
      << result.tabi_convergence_channel_overreject_required_scale_max[0] << ','
      << result.tabi_convergence_channel_overreject_required_scale_max[1] << ','
      << result.tabi_convergence_channel_overreject_required_scale_max[2] << ','
      << result.tabi_convergence_channel_overreject_required_scale_max[3] << ','
      << result.tabi_convergence_channel_overreject_required_scale_p99[0] << ','
      << result.tabi_convergence_channel_overreject_required_scale_p99[1] << ','
      << result.tabi_convergence_channel_overreject_required_scale_p99[2] << ','
      << result.tabi_convergence_channel_overreject_required_scale_p99[3] << ','
      << result.tabi_phantom_high_error_channel_count[0] << ','
      << result.tabi_phantom_high_error_channel_count[1] << ','
      << result.tabi_phantom_high_error_channel_count[2] << ','
      << result.tabi_phantom_high_error_channel_count[3] << ','
      << result.tabi_phantom_high_error_channel_abs_max[0] << ','
      << result.tabi_phantom_high_error_channel_abs_max[1] << ','
      << result.tabi_phantom_high_error_channel_abs_max[2] << ','
      << result.tabi_phantom_high_error_channel_abs_max[3] << ','
      << csv_escape(result.tabi_accepted_miss_worst_channel) << ','
      << result.tabi_accepted_miss_channel_count[0] << ','
      << result.tabi_accepted_miss_channel_count[1] << ','
      << result.tabi_accepted_miss_channel_count[2] << ','
      << result.tabi_accepted_miss_channel_count[3] << ','
      << result.tabi_accepted_miss_channel_panels[0] << ','
      << result.tabi_accepted_miss_channel_panels[1] << ','
      << result.tabi_accepted_miss_channel_panels[2] << ','
      << result.tabi_accepted_miss_channel_panels[3] << ','
      << result.tabi_accepted_miss_channel_rho_min[0] << ','
      << result.tabi_accepted_miss_channel_rho_min[1] << ','
      << result.tabi_accepted_miss_channel_rho_min[2] << ','
      << result.tabi_accepted_miss_channel_rho_min[3] << ','
      << result.tabi_accepted_miss_channel_rho_mean[0] << ','
      << result.tabi_accepted_miss_channel_rho_mean[1] << ','
      << result.tabi_accepted_miss_channel_rho_mean[2] << ','
      << result.tabi_accepted_miss_channel_rho_mean[3] << ','
      << result.tabi_accepted_miss_channel_rho_max[0] << ','
      << result.tabi_accepted_miss_channel_rho_max[1] << ','
      << result.tabi_accepted_miss_channel_rho_max[2] << ','
      << result.tabi_accepted_miss_channel_rho_max[3] << ','
      << result.tabi_accepted_miss_channel_tail_ratio_mean[0] << ','
      << result.tabi_accepted_miss_channel_tail_ratio_mean[1] << ','
      << result.tabi_accepted_miss_channel_tail_ratio_mean[2] << ','
      << result.tabi_accepted_miss_channel_tail_ratio_mean[3] << ','
      << result.tabi_accepted_miss_channel_tail_ratio_max[0] << ','
      << result.tabi_accepted_miss_channel_tail_ratio_max[1] << ','
      << result.tabi_accepted_miss_channel_tail_ratio_max[2] << ','
      << result.tabi_accepted_miss_channel_tail_ratio_max[3] << ','
      << result.tabi_accepted_miss_channel_estimated_error_mean[0] << ','
      << result.tabi_accepted_miss_channel_estimated_error_mean[1] << ','
      << result.tabi_accepted_miss_channel_estimated_error_mean[2] << ','
      << result.tabi_accepted_miss_channel_estimated_error_mean[3] << ','
      << result.tabi_accepted_miss_channel_estimated_error_max[0] << ','
      << result.tabi_accepted_miss_channel_estimated_error_max[1] << ','
      << result.tabi_accepted_miss_channel_estimated_error_max[2] << ','
      << result.tabi_accepted_miss_channel_estimated_error_max[3] << ','
      << result.tabi_convergence_k3_rho_cap_rejects << ','
      << result.tabi_convergence_k3_rho_cap_reject_panels << ','
      << result.tabi_moment_audit_nodes << ','
      << result.tabi_moment_translation_rel_error_max << ','
      << result.tabi_moment_translation_rel_error_mean << ','
      << result.tabi_moment_translation_offset_max << ','
      << result.tabi_moment_translation_offset_mean << ','
      << result.tabi_moment_cancellation_ratio_max << ','
      << result.tabi_moment_cancellation_ratio_mean << ','
      << result.tabi_moment_dynamic_range_max << ','
      << csv_escape(result.tabi_expansion_audit) << ','
      << csv_escape(result.tabi_expansion_audit_channel) << ','
      << result.tabi_far_expansion_audit_pairs << ','
      << result.tabi_far_expansion_audit_panels << ','
      << seconds(result.tabi_far_expansion_direct_ms) << ','
      << result.tabi_far_expansion_rel_error_max << ','
      << result.tabi_far_expansion_rel_error_mean << ','
      << result.tabi_far_expansion_rel_error_p95 << ','
      << result.tabi_far_expansion_rel_error_p99 << ','
      << csv_escape(result.tabi_far_expansion_worst_channel) << ','
      << result.tabi_far_expansion_channel_rel_error_max[0] << ','
      << result.tabi_far_expansion_channel_rel_error_max[1] << ','
      << result.tabi_far_expansion_channel_rel_error_max[2] << ','
      << result.tabi_far_expansion_channel_rel_error_max[3] << ','
      << result.tabi_far_expansion_channel_rel_error_mean[0] << ','
      << result.tabi_far_expansion_channel_rel_error_mean[1] << ','
      << result.tabi_far_expansion_channel_rel_error_mean[2] << ','
      << result.tabi_far_expansion_channel_rel_error_mean[3] << ','
      << result.tabi_far_expansion_channel_rel_error_p99[0] << ','
      << result.tabi_far_expansion_channel_rel_error_p99[1] << ','
      << result.tabi_far_expansion_channel_rel_error_p99[2] << ','
      << result.tabi_far_expansion_channel_rel_error_p99[3] << ','
      << (result.tabi_far_expansion_mean_gate_pass ? 1 : 0) << ','
      << result.tabi_far_convergence_rho_max << ','
      << result.tabi_far_convergence_rho_mean << ','
      << result.tabi_accepted_far_aspect_ratio_max << ','
      << result.tabi_accepted_far_aspect_ratio_mean << ','
      << csv_escape(result.tabi_quad_buffer_auto_candidate) << ','
      << csv_escape(result.tabi_near_integration) << ','
      << csv_escape(result.tabi_quad_buffer) << ','
      << result.tabi_near_self_pairs << ',' << result.tabi_near_one_ring_pairs
      << ',' << result.tabi_near_duffy_pairs << ','
      << result.tabi_near_centroid_pairs << ',' << result.tabi_near_fallbacks
      << ',' << result.tabi_near_geometric_buffer_pairs << ','
      << result.tabi_near_geometric_buffer_gauss_pairs << ','
      << result.tabi_near_geometric_buffer_adaptive_pairs << ','
      << result.tabi_near_geometric_buffer_centroid_pairs << ','
      << result.tabi_near_geometric_buffer_min_gap << ','
      << result.tabi_near_geometric_buffer_mean_gap_ratio << ','
      << result.tabi_near_geometric_buffer_max_edge_mean << ','
      << seconds(result.tabi_near_integration_ms) << ','
      << result.tabi_duffy_subdivisions_max << ','
      << result.tabi_duffy_subdivisions_mean << ','
      << csv_escape(result.tabi_near_kernel_policy) << ','
      << result.tabi_near_k1_accepted << ',' << result.tabi_near_k1_rejected
      << ',' << result.tabi_near_k2_accepted << ','
      << result.tabi_near_k2_rejected << ',' << result.tabi_near_k3_accepted
      << ',' << result.tabi_near_k3_rejected << ','
      << result.tabi_near_k4_accepted << ',' << result.tabi_near_k4_rejected
      << ',' << result.tabi_near_one_ring_k1_accepted << ','
      << result.tabi_near_one_ring_k1_rejected << ','
      << result.tabi_near_one_ring_k2_accepted << ','
      << result.tabi_near_one_ring_k2_rejected << ','
      << result.tabi_near_one_ring_k3_accepted << ','
      << result.tabi_near_one_ring_k3_rejected << ','
      << result.tabi_near_one_ring_k4_accepted << ','
      << result.tabi_near_one_ring_k4_rejected << ','
      << csv_escape(result.tabi_near_k1_policy) << ','
      << csv_escape(result.tabi_near_k2_policy) << ','
      << csv_escape(result.tabi_near_k3_policy) << ','
      << csv_escape(result.tabi_near_k4_policy) << ','
      << result.tabi_near_interactions << ',' << result.tabi_far_interactions
      << ',' << result.tabi_singular_near_interactions << ','
      << result.tabi_leaf_near_interactions << ','
      << result.tabi_far_node_interactions << ','
      << result.tabi_leaf_panel_interactions << ','
      << result.tabi_leaf_tile_interactions << ','
      << result.tabi_leaf_tile_panel_interactions << ','
      << result.tabi_leaf_residual_panel_interactions << ','
      << csv_escape(result.tabi_leaf_work_profile) << ','
      << result.tabi_leaf_work_mac_terminal_panels << ','
      << result.tabi_leaf_work_physical_gap_panels << ','
      << result.tabi_leaf_work_fragmentation_panels << ','
      << result.tabi_leaf_work_convergence_panels << ','
      << result.tabi_leaf_work_convergence_estimator_panels << ','
      << result.tabi_leaf_work_convergence_rho_panels << ','
      << result.tabi_leaf_work_convergence_aspect_panels << ','
      << result.tabi_leaf_work_surface_margin_panels << ','
      << result.tabi_leaf_work_surface_margin_hard_panels << ','
      << result.tabi_leaf_work_surface_margin_soft_panels << ','
      << result.tabi_leaf_work_k3_rho_cap_panels << ','
      << result.tabi_leaf_work_singular_fallback_panels << ','
      << result.tabi_leaf_work_scalar_tail_panels << ','
      << result.tabi_leaf_work_profile_accounted_panels << ','
      << result.tabi_leaf_work_mac_terminal_frac << ','
      << result.tabi_leaf_work_convergence_frac << ','
      << result.tabi_leaf_work_surface_margin_frac << ','
      << result.tabi_leaf_work_singular_fallback_frac << ','
      << result.tabi_leaf_work_scalar_tail_frac << ','
      << result.tabi_smooth_leaf_node_interactions << ','
      << result.tabi_leaf_pruned_interactions << ','
      << result.tabi_leaf_mac_candidates << ',' << result.tabi_leaf_mac_accepted
      << ',' << result.tabi_leaf_mac_rejected << ','
      << result.tabi_leaf_mac_far_panels << ','
      << result.tabi_leaf_mac_direct_panels_saved_estimate << ','
      << result.tabi_adaptive_smooth_leaf_accepted << ','
      << result.tabi_adaptive_smooth_leaf_rejected << ','
      << result.tabi_adaptive_span_tile_accepted << ','
      << result.tabi_adaptive_span_tile_rejected << ','
      << result.tabi_adaptive_exact_leaf_fallbacks << ','
      << result.tabi_adaptive_reject_one_ring_overlap << ','
      << result.tabi_adaptive_reject_error_bound_failed << ','
      << result.tabi_adaptive_reject_p4_check_failed << ','
      << result.tabi_adaptive_reject_too_close << ','
      << result.tabi_leaf_work_reduction_ratio << ','
      << result.tabi_target_cluster_interactions << ','
      << result.tabi_target_cluster_targets << ','
      << result.tabi_target_cluster_panels_pruned << ','
      << result.tabi_target_cluster_rejected << ','
      << seconds(result.tabi_target_cluster_eval_ms) << ','
      << result.tabi_leaf_pair_batch_interactions << ','
      << result.tabi_leaf_pair_batch_targets << ','
      << result.tabi_leaf_pair_batch_source_panels << ','
      << result.tabi_leaf_pair_batch_residual_panels << ','
      << result.tabi_leaf_pair_batch_fallbacks << ','
      << seconds(result.tabi_leaf_pair_batch_ms) << ','
      << csv_escape(result.tabi_leaf_pair_batch_kernel) << ','
      << result.tabi_leaf_pair_batch_vectorized_sources << ','
      << result.tabi_leaf_pair_batch_vector_lanes << ','
      << result.tabi_leaf_pair_batch_pairs_per_second << ','
      << csv_escape(result.tabi_leaf_kernel_profile) << ','
      << csv_escape(result.tabi_jcp_direct_kernel) << ','
      << result.tabi_leaf_kernel_scalar_tasks << ','
      << result.tabi_leaf_kernel_batch_tasks << ','
      << result.tabi_leaf_kernel_source_panel_evals << ','
      << seconds(result.tabi_leaf_kernel_scalar_ms) << ','
      << seconds(result.tabi_leaf_kernel_target_avx2_ms) << ','
      << seconds(result.tabi_leaf_kernel_tail_ms) << ','
      << result.tabi_leaf_pair_batch_soa_sources << ','
      << seconds(result.tabi_leaf_pair_batch_soa_ms) << ','
      << seconds(result.tabi_leaf_pair_batch_scalar_tail_ms) << ','
      << result.tabi_leaf_pair_batch_clean_source_panels_per_second << ','
      << result.tabi_leaf_pair_selection_candidate_scans << ','
      << result.tabi_leaf_pair_selection_clean_rejects << ','
      << result.tabi_leaf_pair_selection_probe_calls << ','
      << result.tabi_leaf_pair_selection_probe_node_visits << ','
      << seconds(result.tabi_leaf_pair_selection_probe_ms) << ','
      << seconds(result.tabi_leaf_pair_selection_ms) << ','
      << result.tabi_leaf_pair_selection_probe_cache_hits << ','
      << result.tabi_leaf_pair_selection_probe_cache_misses << ','
      << result.tabi_leaf_pair_selection_insufficient_targets << ','
      << result.tabi_leaf_pair_selection_same_target_leaf_rejects << ','
      << result.tabi_leaf_pair_directness_ancestor_probe_calls << ','
      << result.tabi_leaf_pair_directness_ancestor_probe_node_visits << ','
      << seconds(result.tabi_leaf_pair_directness_ancestor_probe_ms) << ','
      << result.tabi_leaf_pair_directness_pair_cache_hits << ','
      << result.tabi_leaf_pair_directness_pair_cache_misses << ','
      << result.tabi_leaf_pair_directness_all_direct_proofs << ','
      << result.tabi_leaf_pair_directness_mixed_fallbacks << ','
      << result.tabi_leaf_pair_directness_verify_checks << ','
      << result.tabi_leaf_pair_directness_verify_mismatches << ','
      << result.tabi_leaf_pair_worklist_tasks << ','
      << result.tabi_leaf_pair_worklist_groups << ','
      << result.tabi_leaf_pair_worklist_batched_groups << ','
      << result.tabi_leaf_pair_worklist_scalar_flush_tasks << ','
      << result.tabi_leaf_pair_worklist_forced_flushes << ','
      << result.tabi_leaf_pair_worklist_active_groups_max << ','
      << seconds(result.tabi_leaf_pair_worklist_ms) << ','
      << result.tabi_leaf_pair_worklist_verify_checks << ','
      << result.tabi_leaf_pair_worklist_verify_mismatches << ','
      << result.tabi_leaf_block_target_leaves << ','
      << result.tabi_leaf_block_targets << ','
      << result.tabi_leaf_block_source_groups << ','
      << result.tabi_leaf_block_batched_groups << ','
      << result.tabi_leaf_block_scalar_tail_tasks << ','
      << seconds(result.tabi_leaf_block_group_ms) << ','
      << seconds(result.tabi_leaf_block_flush_ms) << ','
      << seconds(result.tabi_leaf_block_writeback_ms) << ','
      << result.tabi_leaf_block_verify_checks << ','
      << result.tabi_leaf_block_verify_mismatches << ','
      << csv_escape(result.tabi_adaptive_level) << ',' << result.tabi_tree_nodes
      << ',' << result.tabi_tree_leaf_count << ',' << result.tabi_taylor_order
      << ',' << csv_escape(result.tabi_derivative_mode) << ','
      << result.tabi_diag_min << ',' << result.tabi_diag_max << ','
      << result.tabi_solid_angle_fallbacks << ','
      << result.tabi_nonmanifold_edges << ','
      << csv_escape(result.tabi_formulation) << ','
      << csv_escape(result.tabi_energy_projection) << ','
      << seconds(result.tabi_energy_projection_ms) << ','
      << csv_escape(result.tabi_energy_projection_kernel) << ','
      << (result.tabi_fallback_used ? "true" : "false") << ','
      << csv_escape(result.tabi_preconditioner_requested) << ','
      << csv_escape(result.tabi_preconditioner_effective) << ','
      << csv_escape(result.tabi_preconditioner_block_near) << ','
      << result.tabi_preconditioner_block_panel_cap << ','
      << result.tabi_preconditioner_overlap_panel_cap << ','
      << result.tabi_preconditioner_ilu_neighbor_cap_requested << ','
      << result.tabi_preconditioner_ilu_neighbor_cap << ','
      << result.tabi_preconditioner_ilu_drop_tol << ','
      << csv_escape(result.tabi_preconditioner_ilu_regularization) << ','
      << result.tabi_preconditioner_ilu_regularization_max << ','
      << result.tabi_preconditioner_ilu_regularization_selected << ','
      << result.tabi_preconditioner_ilu_regularization_absolute_max << ','
      << result.tabi_preconditioner_ilu_regularization_attempts << ','
      << result.tabi_preconditioner_ilu_regularization_rejected_attempts << ','
      << result.tabi_preconditioner_ilu_regularization_shifted_rows << ','
      << csv_escape(result.tabi_preconditioner_ilu_unshifted_failure_stage)
      << ',' << result.tabi_preconditioner_memory_budget_bytes << ','
      << result.tabi_preconditioner_estimated_bytes << ','
      << result.tabi_preconditioner_valid_blocks << ','
      << result.tabi_preconditioner_valid_panels << ','
      << result.tabi_preconditioner_ill_conditioned_blocks << ','
      << result.tabi_preconditioner_quality_rejected_blocks << ','
      << result.tabi_preconditioner_quality_probe_max_gain << ','
      << result.tabi_preconditioner_quality_probe_rel_residual << ','
      << result.tabi_preconditioner_pivot_ratio_min << ','
      << result.tabi_preconditioner_factor_growth_max << ','
      << result.tabi_preconditioner_pivot_ratio_min_row << ','
      << result.tabi_preconditioner_pivot_ratio_min_column << ','
      << result.tabi_preconditioner_factor_growth_row << ','
      << result.tabi_preconditioner_factor_growth_column << ','
      << csv_escape(result.tabi_preconditioner_quality_failure_stage) << ','
      << result.tabi_preconditioner_quality_failure_row << ','
      << result.tabi_preconditioner_quality_failure_column << ','
      << result.tabi_preconditioner_quality_failure_probe << ','
      << result.tabi_preconditioner_quality_failure_probe_gain << ','
      << result.tabi_preconditioner_quality_failure_probe_rel_residual << ','
      << csv_escape(result.tabi_preconditioner_quality_status) << ','
      << seconds(result.tabi_preconditioner_setup_ms) << ','
      << seconds(result.tabi_preconditioner_graph_setup_ms) << ','
      << (result.tabi_preconditioner_graph_cache_reused ? "true" : "false")
      << ','
      << seconds(result.tabi_preconditioner_near_assembly_setup_ms) << ','
      << seconds(result.tabi_preconditioner_ilu_factor_setup_ms) << ','
      << seconds(result.tabi_preconditioner_apply_ms) << ','
      << result.tabi_preconditioner_apply_calls << ','
      << result.tabi_preconditioner_sparse_nnz_blocks << ','
      << result.tabi_preconditioner_sparse_avg_degree << ','
      << result.tabi_preconditioner_sparse_max_degree << ','
      << result.tabi_preconditioner_coarse_aggregate_cap << ','
      << result.tabi_preconditioner_coarse_aggregates << ','
      << result.tabi_preconditioner_coarse_dimension << ','
      << csv_escape(result.tabi_preconditioner_coarse_base) << ','
      << result.tabi_preconditioner_coarse_matrix_bytes << ','
      << result.tabi_preconditioner_coarse_pivot_ratio << ','
      << csv_escape(result.tabi_preconditioner_coarse_status) << ','
      << result.tabi_preconditioner_coarse_setup_matvec_calls << ','
      << seconds(result.tabi_preconditioner_coarse_setup_ms) << ','
      << seconds(result.tabi_preconditioner_coarse_moments_setup_ms) << ','
      << seconds(result.tabi_preconditioner_coarse_far_setup_ms) << ','
      << seconds(result.tabi_preconditioner_coarse_audit_setup_ms) << ','
      << seconds(result.tabi_preconditioner_coarse_apply_ms) << ','
      << result.tabi_preconditioner_coarse_apply_calls << ','
      << result.tabi_preconditioner_coarse_rp_identity_error_max << ','
      << result.tabi_preconditioner_coarse_verify_rel_error_max << ','
      << csv_escape(result.tabi_surface_generator) << ','
      << csv_escape(result.tabi_discretization) << ','
      << csv_escape(result.tabi_node_patch_representation) << ','
      << csv_escape(result.tabi_node_patch_normal_source) << ','
      << csv_escape(result.tabi_node_patch_certificate) << ','
      << csv_escape(result.tabi_receptor_node_patch_certificate) << ','
      << csv_escape(result.tabi_ligand_node_patch_certificate) << ','
      << result.tabi_rhs_scale
      << ',' << result.tabi_dielectric_contrast << ','
      << result.tabi_operator_contrast << ',' << result.tabi_unknown_count
      << ',' << csv_escape(result.tabi_bie_system) << ','
      << result.tabi_lhs_diag_phi << ',' << result.tabi_lhs_diag_phin << ','
      << csv_escape(result.tabi_energy_equation) << ','
      << csv_escape(result.tabi_kernel_convention) << ','
      << result.tabi_source_norm_s1 << ',' << result.tabi_source_norm_s2 << ','
      << result.tabi_phi_norm << ',' << result.tabi_phin_norm << ','
      << result.tabi_kappa_inv_a << ',' << result.tabi_boundary_energy_kj_mol
      << ',' << result.tabi_reference_energy_kj_mol << ','
      << result.tabi_selected_energy_scale << ',' << result.memory_bytes << ','
      << result.energy_kj_mol << ','
      << result.tabi_bie_cartesian_full_fmm_tree_hash << ','
      << result.tabi_bie_cartesian_full_fmm_worklist_hash << ','
      << result.tabi_bie_cartesian_full_fmm_policy_hash << ','
      << csv_escape(pre_audit.mode) << ','
      << (pre_audit.enabled ? "true" : "false") << ','
      << (pre_audit.finite ? "true" : "false") << ','
      << pre_audit.aggregate_rel << ',' << pre_audit.max_stratum_rel << ','
      << pre_audit.audited_targets << ',' << pre_audit.audited_rows << ','
      << pre_audit.stages << ','
      << (pre_audit.expanded_to_256 ? "true" : "false") << ','
      << (pre_audit.expanded_to_512 ? "true" : "false") << ','
      << seconds(pre_audit.seconds) << ','
      << seconds(pre_audit.channel_shadow_seconds) << ','
      << pre_audit.probe_hash << ',' << pre_audit.channel_probe_hash << ','
      << pre_audit.sampled_target_hash << ','
      << pre_audit.channel_direct_pairs_expected << ','
      << pre_audit.channel_direct_pairs_applied << ','
      << (solution_audit.enabled ? "true" : "false") << ','
      << (solution_audit.finite ? "true" : "false") << ','
      << (solution_audit.relative_available ? "true" : "false") << ','
      << csv_escape(solution_audit.status) << ','
      << solution_audit.relative_error << ',' << solution_audit.absolute_l2
      << ',' << solution_audit.direct_l2 << ',' << solution_audit.targets
      << ',' << solution_audit.rows << ','
      << seconds(solution_audit.seconds) << ',' << solution_audit.target_hash
      << ',' << solution_audit.direct_pairs_expected << ','
      << solution_audit.direct_pairs_applied << ','
      << (channel_audit[0].available ? "true" : "false") << ','
      << (channel_audit[0].finite ? "true" : "false") << ','
      << csv_escape(channel_audit[0].status) << ','
      << channel_audit[0].samples << ',' << channel_audit[0].abs_l2 << ','
      << channel_audit[0].rms << ',' << channel_audit[0].reference_l2 << ','
      << channel_audit[0].relative_error << ','
      << (channel_audit[1].available ? "true" : "false") << ','
      << (channel_audit[1].finite ? "true" : "false") << ','
      << csv_escape(channel_audit[1].status) << ','
      << channel_audit[1].samples << ',' << channel_audit[1].abs_l2 << ','
      << channel_audit[1].rms << ',' << channel_audit[1].reference_l2 << ','
      << channel_audit[1].relative_error << ','
      << (channel_audit[2].available ? "true" : "false") << ','
      << (channel_audit[2].finite ? "true" : "false") << ','
      << csv_escape(channel_audit[2].status) << ','
      << channel_audit[2].samples << ',' << channel_audit[2].abs_l2 << ','
      << channel_audit[2].rms << ',' << channel_audit[2].reference_l2 << ','
      << channel_audit[2].relative_error << ','
      << (channel_audit[3].available ? "true" : "false") << ','
      << (channel_audit[3].finite ? "true" : "false") << ','
      << csv_escape(channel_audit[3].status) << ','
      << channel_audit[3].samples << ',' << channel_audit[3].abs_l2 << ','
      << channel_audit[3].rms << ',' << channel_audit[3].reference_l2 << ','
      << channel_audit[3].relative_error << ',';
  tabi_profile_out_
      << csv_escape(retry.mode) << ',' << (retry.enabled ? "true" : "false")
      << ',' << retry.attempts_started << ',' << retry.attempts_completed << ','
      << retry.accepted_attempt << ',' << csv_escape(retry.trigger_reason)
      << ',' << csv_escape(retry.final_reason) << ','
      << (retry.geometry_identity_valid ? "true" : "false") << ','
      << (retry.tree_identity_valid ? "true" : "false") << ','
      << (retry.policy_transition_valid ? "true" : "false") << ','
      << seconds(retry.cumulative_wall_ms) << ','
      << seconds(retry.cumulative_gmres_ms) << ','
      << retry.cumulative_accounted_memory_bytes << ','
      << retry.max_boundary_rss_bytes << ',';
  write_retry_attempt(retry.attempts[0]);
  tabi_profile_out_ << ',';
  write_retry_attempt(retry.attempts[1]);
  tabi_profile_out_ << '\n';
}

void AnalyzerImpl::write_tabi_near_block_audit_header_if_needed() {
  if (args_.timing != "detail")
    return;
  if (!tabi_near_block_audit_out_.is_open()) {
    tabi_near_block_audit_out_.open("pbe_tabi_near_block_audit.csv",
                                    std::ios::out | std::ios::trunc);
    if (!tabi_near_block_audit_out_.is_open()) {
      throw std::runtime_error(
          "PBE: failed to open pbe_tabi_near_block_audit.csv.");
    }
  }
  if (!tabi_near_block_audit_header_written_) {
    tabi_near_block_audit_out_
        << "trajectory_id,frame_index,global_index,fixture,block,pair_type,"
           "orientation,accepted,rejected,candidate_worse_than_centroid,"
           "reference_nonconverged,orientation_sign_mismatch,"
           "strict_ref_reject,"
           "max_candidate_ref_rel,mean_candidate_ref_rel,"
           "max_centroid_ref_rel,k2_laplace_sign_ok,"
           "k2_yukawa_smooth_ref_rel,k2_orientation_pair_ref_rel\n";
    tabi_near_block_audit_header_written_ = true;
  }
}

void AnalyzerImpl::append_tabi_near_block_audit_rows(
    const FrameResult &result) {
  if (args_.timing != "detail" || result.pbe_backend != "tabi")
    return;
  write_tabi_near_block_audit_header_if_needed();
  static constexpr std::array<const char *, kNearBlockCount> kBlockLabels = {
      "K1", "K2", "K3", "K4"};
  static constexpr std::array<const char *, kNearPairTypeCount>
      kPairTypeLabels = {"self", "edge", "vertex", "smooth_near"};
  static constexpr std::array<const char *, kNearOrientationCount>
      kOrientationLabels = {"same", "flipped", "mixed"};
  for (std::size_t block = 0; block < kNearBlockCount; ++block) {
    for (std::size_t pair_type = 0; pair_type < kNearPairTypeCount;
         ++pair_type) {
      for (std::size_t orientation = 0; orientation < kNearOrientationCount;
           ++orientation) {
        const std::size_t index =
            (block * kNearPairTypeCount + pair_type) * kNearOrientationCount +
            orientation;
        if (index >= result.tabi_near_block_audit.size())
          continue;
        const auto &bucket = result.tabi_near_block_audit[index];
        const double mean_candidate_ref_rel =
            bucket.sample_count > 0
                ? bucket.sum_candidate_ref_rel /
                      static_cast<double>(bucket.sample_count)
                : 0.0;
        tabi_near_block_audit_out_
            << result.trajectory_id << ',' << result.frame_index << ','
            << result.global_index << ',' << csv_escape(current_source_label_)
            << ',' << kBlockLabels[block] << ',' << kPairTypeLabels[pair_type]
            << ',' << kOrientationLabels[orientation] << ',' << bucket.accepted
            << ',' << bucket.rejected << ','
            << bucket.candidate_worse_than_centroid << ','
            << bucket.reference_nonconverged << ','
            << bucket.orientation_sign_mismatch << ','
            << bucket.strict_ref_reject << ',' << bucket.max_candidate_ref_rel
            << ',' << mean_candidate_ref_rel << ','
            << bucket.max_centroid_ref_rel << ',' << bucket.k2_laplace_sign_ok
            << ',' << bucket.max_k2_yukawa_smooth_ref_rel << ','
            << bucket.max_k2_orientation_pair_ref_rel << '\n';
      }
    }
  }
}

void AnalyzerImpl::write_tabi_geometric_near_audit_header_if_needed() {
  if (args_.timing != "detail")
    return;
  if (!tabi_geometric_near_audit_out_.is_open()) {
    tabi_geometric_near_audit_out_.open("pbe_tabi_geometric_near_audit.csv",
                                        std::ios::out | std::ios::trunc);
    if (!tabi_geometric_near_audit_out_.is_open()) {
      throw std::runtime_error(
          "PBE: failed to open pbe_tabi_geometric_near_audit.csv.");
    }
  }
  if (!tabi_geometric_near_audit_header_written_) {
    tabi_geometric_near_audit_out_
        << "trajectory_id,frame_index,global_index,fixture,gap_bucket,"
           "center_bucket,block,kernel,pairs,min_gap,max_gap_ratio,"
           "mean_gap_ratio,mean_center_ratio\n";
    tabi_geometric_near_audit_header_written_ = true;
  }
}

void AnalyzerImpl::append_tabi_geometric_near_audit_rows(
    const FrameResult &result) {
  if (args_.timing != "detail" || result.pbe_backend != "tabi")
    return;
  write_tabi_geometric_near_audit_header_if_needed();
  static constexpr std::array<const char *, kGeometricGapBucketCount>
      kGapLabels = {"lt_0p25", "0p25_0p50", "0p50_0p75", "gt_0p75"};
  static constexpr std::array<const char *, kGeometricCenterBucketCount>
      kCenterLabels = {"lt_1p0", "1p0_2p0", "ge_2p0"};
  static constexpr std::array<const char *, kGeometricKernelCount>
      kKernelLabels = {"centroid", "gauss7", "subdivided"};
  static constexpr std::array<const char *, kNearBlockCount> kBlockLabels = {
      "K1", "K2", "K3", "K4"};
  for (std::size_t gap = 0; gap < kGeometricGapBucketCount; ++gap) {
    for (std::size_t center = 0; center < kGeometricCenterBucketCount;
         ++center) {
      for (std::size_t kernel = 0; kernel < kGeometricKernelCount; ++kernel) {
        const std::size_t index = (gap * kGeometricCenterBucketCount + center) *
                                      kGeometricKernelCount +
                                  kernel;
        if (index >= result.tabi_geometric_near_audit.size())
          continue;
        const auto &bucket = result.tabi_geometric_near_audit[index];
        if (bucket.pairs == 0)
          continue;
        const double denom = static_cast<double>(bucket.pairs);
        const double mean_gap_ratio = bucket.sum_gap_ratio / denom;
        const double mean_center_ratio = bucket.sum_center_ratio / denom;
        for (const char *block_label : kBlockLabels) {
          tabi_geometric_near_audit_out_
              << result.trajectory_id << ',' << result.frame_index << ','
              << result.global_index << ',' << csv_escape(current_source_label_)
              << ',' << kGapLabels[gap] << ',' << kCenterLabels[center] << ','
              << block_label << ',' << kKernelLabels[kernel] << ','
              << bucket.pairs << ',' << bucket.min_gap << ','
              << bucket.max_gap_ratio << ',' << mean_gap_ratio << ','
              << mean_center_ratio << '\n';
        }
      }
    }
  }
}

void AnalyzerImpl::write_tabi_far_geometric_audit_header_if_needed() {
  if (args_.timing != "detail")
    return;
  if (!tabi_far_geometric_audit_out_.is_open()) {
    tabi_far_geometric_audit_out_.open("pbe_tabi_far_geometric_audit.csv",
                                       std::ios::out | std::ios::trunc);
    if (!tabi_far_geometric_audit_out_.is_open()) {
      throw std::runtime_error(
          "PBE: failed to open pbe_tabi_far_geometric_audit.csv.");
    }
  }
  if (!tabi_far_geometric_audit_header_written_) {
    tabi_far_geometric_audit_out_
        << "trajectory_id,frame_index,global_index,fixture,tree_builder,"
           "node_bounds,theta,leaf,node_depth,gap_ratio,node_panel_count,"
           "mac_safety,mac_gap_factor,unsafe_count,unsafe_panels,min_gap,"
           "min_gap_ratio,mean_gap_ratio,audit_samples,"
           "physical_gap_rejects,convergence_safety,convergence_rho_cap,"
           "mac_aspect_cap,convergence_estimator,"
           "convergence_tail_threshold,convergence_estimator_error_cap,"
           "convergence_unstable_rho_cap,convergence_calibration,"
           "convergence_estimator_samples,"
           "convergence_estimator_unstable_far_count,"
           "convergence_estimator_split_rejects,"
           "convergence_estimator_tail_ratio_max,"
           "convergence_estimator_error_max,convergence_tail_norm_p_max,"
           "convergence_tail_norm_pm1_max,"
           "convergence_effective_jcp_amp_max,"
           "far_expansion_mean_gate_pass,convergence_rho_rejects,"
           "convergence_aspect_rejects,convergence_rho_reject_panels,"
           "convergence_aspect_reject_panels\n";
    tabi_far_geometric_audit_header_written_ = true;
  }
}

void AnalyzerImpl::append_tabi_far_geometric_audit_row(
    const FrameResult &result) {
  if (args_.timing != "detail" || result.pbe_backend != "tabi")
    return;
  write_tabi_far_geometric_audit_header_if_needed();
  tabi_far_geometric_audit_out_
      << result.trajectory_id << ',' << result.frame_index << ','
      << result.global_index << ',' << csv_escape(current_source_label_) << ','
      << csv_escape(result.tabi_tree_builder) << ','
      << csv_escape(result.tabi_node_bounds_mode) << ','
      << args_.tabi_tree_opening_theta << ',' << args_.tabi_tree_leaf_size
      << ',' << "all,all,all," << csv_escape(result.tabi_mac_safety) << ','
      << result.tabi_mac_gap_factor << ','
      << result.tabi_far_geometric_unsafe_count << ','
      << result.tabi_far_geometric_unsafe_panels << ','
      << result.tabi_far_geometric_min_gap << ','
      << result.tabi_far_geometric_gap_ratio_min << ','
      << result.tabi_far_geometric_gap_ratio_mean << ','
      << result.tabi_far_geometric_audit_samples << ','
      << result.tabi_mac_physical_gap_rejects << ','
      << csv_escape(result.tabi_convergence_safety) << ','
      << result.tabi_convergence_rho_cap << ',' << result.tabi_mac_aspect_cap
      << ',' << csv_escape(result.tabi_convergence_estimator) << ','
      << result.tabi_convergence_tail_threshold << ','
      << result.tabi_convergence_estimator_error_cap << ','
      << result.tabi_convergence_unstable_rho_cap << ','
      << csv_escape(result.tabi_convergence_calibration) << ','
      << result.tabi_convergence_estimator_samples << ','
      << result.tabi_convergence_estimator_unstable_far_count << ','
      << result.tabi_convergence_estimator_split_rejects << ','
      << result.tabi_convergence_estimator_tail_ratio_max << ','
      << result.tabi_convergence_estimator_error_max << ','
      << result.tabi_convergence_tail_norm_p_max << ','
      << result.tabi_convergence_tail_norm_pm1_max << ','
      << result.tabi_convergence_effective_jcp_amp_max << ','
      << (result.tabi_far_expansion_mean_gate_pass ? 1 : 0) << ','
      << result.tabi_mac_convergence_rho_rejects << ','
      << result.tabi_mac_convergence_aspect_rejects << ','
      << result.tabi_mac_convergence_rho_reject_panels << ','
      << result.tabi_mac_convergence_aspect_reject_panels << '\n';
}

void AnalyzerImpl::write_tabi_moment_audit_header_if_needed() {
  if (args_.timing != "detail")
    return;
  if (!tabi_moment_audit_out_.is_open()) {
    tabi_moment_audit_out_.open("pbe_tabi_moment_audit.csv",
                                std::ios::out | std::ios::trunc);
    if (!tabi_moment_audit_out_.is_open()) {
      throw std::runtime_error(
          "PBE: failed to open pbe_tabi_moment_audit.csv.");
    }
  }
  if (!tabi_moment_audit_header_written_) {
    tabi_moment_audit_out_
        << "trajectory_id,frame_index,global_index,fixture,tree_builder,"
           "node_bounds,moment_audit,moment_construction,moment_sum,"
           "nodes,translation_rel_error_max,translation_rel_error_mean,"
           "translation_offset_max,translation_offset_mean,"
           "cancellation_ratio_max,cancellation_ratio_mean,"
           "dynamic_range_max,status\n";
    tabi_moment_audit_header_written_ = true;
  }
}

void AnalyzerImpl::append_tabi_moment_audit_row(const FrameResult &result) {
  if (args_.timing != "detail" || result.pbe_backend != "tabi" ||
      result.tabi_moment_audit == "off")
    return;
  write_tabi_moment_audit_header_if_needed();
  tabi_moment_audit_out_ << result.trajectory_id << ',' << result.frame_index
                         << ',' << result.global_index << ','
                         << csv_escape(current_source_label_) << ','
                         << csv_escape(result.tabi_tree_builder) << ','
                         << csv_escape(result.tabi_node_bounds_mode) << ','
                         << csv_escape(result.tabi_moment_audit) << ','
                         << csv_escape(result.tabi_moment_construction) << ','
                         << csv_escape(result.tabi_moment_sum) << ','
                         << result.tabi_moment_audit_nodes << ','
                         << result.tabi_moment_translation_rel_error_max << ','
                         << result.tabi_moment_translation_rel_error_mean << ','
                         << result.tabi_moment_translation_offset_max << ','
                         << result.tabi_moment_translation_offset_mean << ','
                         << result.tabi_moment_cancellation_ratio_max << ','
                         << result.tabi_moment_cancellation_ratio_mean << ','
                         << result.tabi_moment_dynamic_range_max << ','
                         << csv_escape(result.status) << '\n';
}

void AnalyzerImpl::write_tabi_surface_quality_header_if_needed() {
  if (args_.timing != "detail")
    return;
  if (!tabi_surface_quality_out_.is_open()) {
    tabi_surface_quality_out_.open("pbe_tabi_surface_quality.csv",
                                   std::ios::out | std::ios::trunc);
    if (!tabi_surface_quality_out_.is_open()) {
      throw std::runtime_error(
          "PBE: failed to open pbe_tabi_surface_quality.csv.");
    }
  }
  if (!tabi_surface_quality_header_written_) {
    tabi_surface_quality_out_
        << "trajectory_id,frame_index,global_index,fixture,pbe_backend,"
           "status,sphere_mesh_mode,panel_count,total_area,"
           "area_rel_error_vs_4piR2,triangle_area_min,triangle_area_mean,"
           "triangle_area_max,triangle_aspect_max,normal_radial_dot_min,"
           "normal_radial_dot_mean,centroid_radial_error_max,"
           "planar_area,planar_area_rel_error_vs_4piR2,enclosed_volume,"
           "enclosed_volume_rel_error_vs_4piR3_over3,triangle_area_cv,"
           "surface_topology_source,analytic_ses_engine,surface_status_detail,"
           "cavity_policy,odd_ray_rows,"
           "patch_ray_unresolved_uncertified_rows,"
           "morphology_fallback_vertices,analytic_orientation_status,"
           "analytic_inverted_faces,analytic_orientation_uncertified_faces,"
           "analytic_self_intersections,wide_repair_attempts,"
           "topology_certificate_accepts,topology_fallback_audits,"
           "initial_collapse_ms,stable_sliver_ms,full_projection_ms,"
           "orientation_repair_ms,wide_repair_ms,self_intersection_audit_ms,"
           "filled_cavity_regions,filled_cavity_vertices,"
           "poor_triangles_before,poor_triangles_after,"
           "triangle_quality_min_before,triangle_quality_min_after,"
           "max_edge_ratio_before,max_edge_ratio_after,analytic_audit_status,"
           "analytic_audit_samples,analytic_audit_hits,analytic_audit_misses,"
           "analytic_projection_distance_rms,analytic_projection_distance_max,"
           "analytic_patch_residual_rms,analytic_patch_residual_max,"
           "analytic_normal_angle_rms_deg,analytic_normal_angle_max_deg,"
           "sampled_hausdorff_proxy,analytic_normal_quality_status,"
           "analytic_orientation_smooth_faces,analytic_orientation_seam_faces,"
           "analytic_normal_cosine_below_025_faces,"
           "analytic_normal_cosine_below_050_faces,"
           "analytic_normal_area_weighted_rms_deg,analytic_normal_p99_deg,"
           "analytic_smooth_normal_p99_deg,analytic_seam_normal_p99_deg,"
           "analytic_normal_area_below_cosine_025_fraction,"
           "analytic_normal_area_below_cosine_050_fraction,"
           "calderon_trace_space_status,calderon_trace_vertices,"
           "calderon_trace_edges,calderon_trace_boundary_edges,"
           "calderon_trace_orientation_mismatches,"
           "calderon_trace_nonmanifold_edges,"
           "calderon_trace_unreferenced_vertices,"
           "calderon_trace_components,calderon_trace_euler_characteristic,"
           "calderon_trace_genus_sum,calderon_trace_topology_invariants_valid,"
           "calderon_aux_surface_mode,calderon_aux_trace_status,"
           "calderon_aux_panels,calderon_aux_vertices,calderon_aux_edges,"
           "calderon_aux_boundary_edges,calderon_aux_orientation_mismatches,"
           "calderon_aux_nonmanifold_edges,calderon_aux_unreferenced_vertices,"
           "calderon_aux_grid_vertices,calderon_aux_area_a2,calderon_aux_build_"
           "ms\n";
    tabi_surface_quality_header_written_ = true;
  }
}

void AnalyzerImpl::append_tabi_surface_quality_row(const FrameResult &result) {
  if (args_.timing != "detail" || result.pbe_backend != "tabi")
    return;
  write_tabi_surface_quality_header_if_needed();
  tabi_surface_quality_out_
      << result.trajectory_id << ',' << result.frame_index << ','
      << result.global_index << ',' << csv_escape(current_source_label_) << ','
      << csv_escape(result.pbe_backend) << ',' << csv_escape(result.status)
      << ',' << csv_escape(result.tabi_sphere_mesh_mode) << ','
      << result.tabi_surface_panels << ','
      << result.tabi_surface_quality_total_area_a2 << ','
      << result.tabi_surface_quality_area_rel_error_vs_4piR2 << ','
      << result.tabi_surface_quality_triangle_area_min << ','
      << result.tabi_surface_quality_triangle_area_mean << ','
      << result.tabi_surface_quality_triangle_area_max << ','
      << result.tabi_surface_quality_triangle_aspect_max << ','
      << result.tabi_surface_quality_normal_radial_dot_min << ','
      << result.tabi_surface_quality_normal_radial_dot_mean << ','
      << result.tabi_surface_quality_centroid_radial_error_max << ','
      << result.tabi_surface_quality_planar_area_a2 << ','
      << result.tabi_surface_quality_planar_area_rel_error_vs_4piR2 << ','
      << result.tabi_surface_quality_enclosed_volume_a3 << ','
      << result.tabi_surface_quality_enclosed_volume_rel_error_vs_4piR3_over3
      << ',' << result.tabi_surface_quality_triangle_area_cv << ','
      << csv_escape(result.tabi_surface_topology_source) << ','
      << csv_escape(result.tabi_surface_analytic_ses_engine) << ','
      << csv_escape(result.tabi_surface_status_detail) << ','
      << csv_escape(result.tabi_surface_cavity_policy) << ','
      << result.tabi_surface_odd_ray_rows << ','
      << result.tabi_surface_patch_ray_unresolved_uncertified_rows << ','
      << result.tabi_surface_morphology_fallback_vertices << ','
      << csv_escape(result.tabi_surface_analytic_orientation_status) << ','
      << result.tabi_surface_analytic_inverted_faces << ','
      << result.tabi_surface_analytic_orientation_uncertified_faces << ','
      << result.tabi_surface_analytic_self_intersections << ','
      << result.tabi_surface_wide_repair_attempts << ','
      << result.tabi_surface_topology_certificate_accepts << ','
      << result.tabi_surface_topology_fallback_audits << ','
      << result.tabi_surface_initial_collapse_ms << ','
      << result.tabi_surface_stable_sliver_ms << ','
      << result.tabi_surface_full_projection_ms << ','
      << result.tabi_surface_orientation_repair_ms << ','
      << result.tabi_surface_wide_repair_ms << ','
      << result.tabi_surface_self_intersection_audit_ms << ','
      << result.tabi_surface_filled_cavity_regions << ','
      << result.tabi_surface_filled_cavity_vertices << ','
      << result.tabi_surface_poor_triangles_before << ','
      << result.tabi_surface_poor_triangles_after << ','
      << result.tabi_surface_triangle_quality_min_before << ','
      << result.tabi_surface_triangle_quality_min_after << ','
      << result.tabi_surface_max_edge_ratio_before << ','
      << result.tabi_surface_max_edge_ratio_after << ','
      << csv_escape(result.tabi_surface_analytic_audit_status) << ','
      << result.tabi_surface_analytic_audit_samples << ','
      << result.tabi_surface_analytic_audit_hits << ','
      << result.tabi_surface_analytic_audit_misses << ','
      << result.tabi_surface_analytic_projection_distance_rms << ','
      << result.tabi_surface_analytic_projection_distance_max << ','
      << result.tabi_surface_analytic_patch_residual_rms << ','
      << result.tabi_surface_analytic_patch_residual_max << ','
      << result.tabi_surface_analytic_normal_angle_rms_deg << ','
      << result.tabi_surface_analytic_normal_angle_max_deg << ','
      << result.tabi_surface_sampled_hausdorff_proxy << ','
      << csv_escape(result.tabi_surface_analytic_normal_quality_status) << ','
      << result.tabi_surface_analytic_orientation_smooth_faces << ','
      << result.tabi_surface_analytic_orientation_seam_faces << ','
      << result.tabi_surface_analytic_normal_cosine_below_025_faces << ','
      << result.tabi_surface_analytic_normal_cosine_below_050_faces << ','
      << result.tabi_surface_analytic_normal_area_weighted_rms_deg << ','
      << result.tabi_surface_analytic_normal_p99_deg << ','
      << result.tabi_surface_analytic_smooth_normal_p99_deg << ','
      << result.tabi_surface_analytic_seam_normal_p99_deg << ','
      << result.tabi_surface_analytic_normal_area_below_cosine_025_fraction
      << ','
      << result.tabi_surface_analytic_normal_area_below_cosine_050_fraction
      << ','
      << csv_escape(result.tabi_calderon_trace_space_status) << ','
      << result.tabi_calderon_trace_vertices << ','
      << result.tabi_calderon_trace_edges << ','
      << result.tabi_calderon_trace_boundary_edges << ','
      << result.tabi_calderon_trace_orientation_mismatches << ','
      << result.tabi_calderon_trace_nonmanifold_edges << ','
      << result.tabi_calderon_trace_unreferenced_vertices << ','
      << result.tabi_calderon_trace_components << ','
      << result.tabi_calderon_trace_euler_characteristic << ','
      << result.tabi_calderon_trace_genus_sum << ','
      << static_cast<int>(result.tabi_calderon_trace_topology_invariants_valid)
      << ','
      << csv_escape(result.tabi_calderon_aux_surface_mode) << ','
      << csv_escape(result.tabi_calderon_aux_trace_status) << ','
      << result.tabi_calderon_aux_panels << ','
      << result.tabi_calderon_aux_vertices << ','
      << result.tabi_calderon_aux_edges << ','
      << result.tabi_calderon_aux_boundary_edges << ','
      << result.tabi_calderon_aux_orientation_mismatches << ','
      << result.tabi_calderon_aux_nonmanifold_edges << ','
      << result.tabi_calderon_aux_unreferenced_vertices << ','
      << result.tabi_calderon_aux_grid_vertices << ','
      << result.tabi_calderon_aux_area_a2 << ','
      << result.tabi_calderon_aux_build_ms << '\n';
}

void AnalyzerImpl::maybe_write_tabi_surface_solution(
    const FrameResult &result, const ReactionSolve &solve) const {
  if (args_.backend != "tabi" || args_.tabi_surface_output == "off")
    return;
  const std::size_t panel_count = solve.tabi_output_panels.size();
  if (panel_count == 0)
    throw std::runtime_error(
        "PBE: TABI surface output requested but no panels were retained.");
  if (solve.tabi_output_solution.size() < panel_count)
    throw std::runtime_error("PBE: TABI surface output requested but no "
                             "solution vector was retained.");
  const bool physical_trace_supported =
      solve.tabi_formulation.rfind("jcp2013", 0) == 0 ||
      solve.tabi_formulation == "tabi2-nodepatch";
  if (!physical_trace_supported)
    throw std::runtime_error(
        "PBE: physical TABI surface fields require a coupled JCP2013 "
        "boundary-trace formulation.");
  if (solve.tabi_output_rhs.size() < panel_count)
    throw std::runtime_error("PBE: TABI physical surface output requires the "
                             "retained Coulomb boundary trace.");
  const bool coupled = solve.tabi_output_solution.size() >= 2 * panel_count;
  const bool node_patch = solve.tabi_discretization == "node_patch";
  const Tabi::NodePatchGeometry *node_geometry = nullptr;
  if (node_patch) {
    node_geometry = solve.tabi_output_panels.front().node_patch_geometry.get();
    if (node_geometry == nullptr ||
        node_geometry->vertex_xyz.size() != panel_count) {
      throw std::runtime_error(
          "PBE: node-patch surface output requires retained connected mesh "
          "geometry matching the solution vector.");
    }
  }
  const bool write_csv =
      args_.tabi_surface_output == "csv" || args_.tabi_surface_output == "all";
  const bool write_vtk =
      args_.tabi_surface_output == "vtk" || args_.tabi_surface_output == "all";
  const auto begin = std::chrono::steady_clock::now();
  auto value_at = [&](std::size_t panel, bool phin) {
    if (phin)
      return coupled ? solve.tabi_output_solution[panel_count + panel] : 0.0;
    return solve.tabi_output_solution[panel];
  };
  const double coulomb_kcal_mol_e =
      solve.tabi_coulomb_factor_kcal_mol > 0.0
          ? solve.tabi_coulomb_factor_kcal_mol
          : kCoulombKcalPerMol;
  const double kt_kcal_mol =
      kKBKcalPerMolK * std::max(1.0, args_.temp);
  constexpr double four_pi =
      4.0 * 3.141592653589793238462643383279502884;
  auto total_potential_e_per_a_at = [&](std::size_t panel) {
    return four_pi * value_at(panel, false);
  };
  auto coulomb_potential_e_per_a_at = [&](std::size_t panel) {
    return four_pi * solve.tabi_output_rhs[panel];
  };
  auto reaction_potential_e_per_a_at = [&](std::size_t panel) {
    return total_potential_e_per_a_at(panel) -
           coulomb_potential_e_per_a_at(panel);
  };
  auto potential_kcal_mol_e_at = [&](std::size_t panel) {
    return coulomb_kcal_mol_e * total_potential_e_per_a_at(panel);
  };
  auto potential_kt_e_at = [&](std::size_t panel) {
    return potential_kcal_mol_e_at(panel) / kt_kcal_mol;
  };
  auto coulomb_potential_kcal_mol_e_at = [&](std::size_t panel) {
    return coulomb_kcal_mol_e * coulomb_potential_e_per_a_at(panel);
  };
  auto coulomb_potential_kt_e_at = [&](std::size_t panel) {
    return coulomb_potential_kcal_mol_e_at(panel) / kt_kcal_mol;
  };
  auto reaction_potential_kcal_mol_e_at = [&](std::size_t panel) {
    return coulomb_kcal_mol_e * reaction_potential_e_per_a_at(panel);
  };
  auto reaction_potential_kt_e_at = [&](std::size_t panel) {
    return reaction_potential_kcal_mol_e_at(panel) / kt_kcal_mol;
  };
  if (write_csv) {
    std::ofstream out("pbe_tabi_surface_solution.csv",
                      std::ios::out | std::ios::trunc);
    if (!out.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_tabi_surface_solution.csv.");
    out << std::setprecision(17);
    out << "trajectory_id,frame_index,global_index,fixture,surface_generator,"
           "discretization,data_association,panel_id,atom_index,"
           "x_A,y_A,z_A,nx,ny,nz,area_A2,diagonal,solid_angle,phi,phin,"
           "normal_flux_proxy,raw_boundary_unknown,"
           "total_potential_e_per_A,total_potential_kcal_mol_e,"
           "total_potential_kT_e,coulomb_potential_e_per_A,"
           "coulomb_potential_kcal_mol_e,coulomb_potential_kT_e,"
           "reaction_potential_e_per_A,reaction_potential_kcal_mol_e,"
           "reaction_potential_kT_e,potential_kcal_mol_e,potential_kT_e,"
           "v0x_A,v0y_A,v0z_A,v1x_A,v1y_A,v1z_A,v2x_A,v2y_A,v2z_A,"
           "has_triangle_geometry\n";
    for (std::size_t i = 0; i < panel_count; ++i) {
      const Tabi::Panel &panel = solve.tabi_output_panels[i];
      out << result.trajectory_id << ',' << result.frame_index << ','
          << result.global_index << ',' << csv_escape(current_source_label_)
          << ',' << csv_escape(solve.tabi_surface_generator) << ','
          << csv_escape(solve.tabi_discretization) << ','
          << (node_patch ? "point" : "cell") << ',' << i << ','
          << panel.atom_index << ',' << panel.x << ','
          << panel.y << ',' << panel.z << ',' << panel.nx << ',' << panel.ny
          << ',' << panel.nz << ',' << panel.area << ',' << panel.diagonal
          << ',' << panel.solid_angle << ',' << value_at(i, false) << ','
          << value_at(i, true) << ',' << value_at(i, true) << ','
          << value_at(i, false) << ',' << total_potential_e_per_a_at(i) << ','
          << potential_kcal_mol_e_at(i) << ',' << potential_kt_e_at(i) << ','
          << coulomb_potential_e_per_a_at(i) << ','
          << coulomb_potential_kcal_mol_e_at(i) << ','
          << coulomb_potential_kt_e_at(i) << ','
          << reaction_potential_e_per_a_at(i) << ','
          << reaction_potential_kcal_mol_e_at(i) << ','
          << reaction_potential_kt_e_at(i) << ','
          << potential_kcal_mol_e_at(i) << ',' << potential_kt_e_at(i) << ','
          << panel.v0x
          << ',' << panel.v0y << ',' << panel.v0z << ',' << panel.v1x << ','
          << panel.v1y << ',' << panel.v1z << ',' << panel.v2x << ','
          << panel.v2y << ',' << panel.v2z << ','
          << (panel.has_triangle_geometry ? "true" : "false") << '\n';
    }
  }
  if (write_vtk) {
    std::ofstream out("pbe_tabi_surface_solution.vtk",
                      std::ios::out | std::ios::trunc);
    if (!out.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_tabi_surface_solution.vtk.");
    out << std::setprecision(17);
    out << "# vtk DataFile Version 3.0\n";
    out << "TITAN TABI surface solution; surface="
        << solve.tabi_surface_generator << "; discretization="
        << solve.tabi_discretization
        << "; physical_field=boundary_total_4pi_phi; potential_unit=kT/e\n";
    out << "ASCII\n";
    out << "DATASET POLYDATA\n";
    if (node_patch) {
      out << "POINTS " << node_geometry->vertex_xyz.size() << " double\n";
      for (const auto &xyz : node_geometry->vertex_xyz)
        out << xyz[0] << ' ' << xyz[1] << ' ' << xyz[2] << '\n';
      out << "POLYGONS " << node_geometry->triangles.size() << ' '
          << (node_geometry->triangles.size() * 4u) << "\n";
      for (const Tabi::NodePatchTriangle &triangle : node_geometry->triangles)
        out << "3 " << triangle.vertices[0] << ' ' << triangle.vertices[1]
            << ' ' << triangle.vertices[2] << '\n';
      out << "POINT_DATA " << panel_count << "\n";
    } else {
      out << "POINTS " << (panel_count * 3u) << " double\n";
      for (const Tabi::Panel &panel : solve.tabi_output_panels) {
        out << panel.v0x << ' ' << panel.v0y << ' ' << panel.v0z << '\n';
        out << panel.v1x << ' ' << panel.v1y << ' ' << panel.v1z << '\n';
        out << panel.v2x << ' ' << panel.v2y << ' ' << panel.v2z << '\n';
      }
      out << "POLYGONS " << panel_count << ' ' << (panel_count * 4u)
          << "\n";
      for (std::size_t i = 0; i < panel_count; ++i) {
        const std::size_t base = 3u * i;
        out << "3 " << base << ' ' << (base + 1u) << ' ' << (base + 2u)
            << '\n';
      }
      out << "CELL_DATA " << panel_count << "\n";
    }
    out << "SCALARS phi double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << value_at(i, false) << '\n';
    out << "SCALARS raw_boundary_unknown double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << value_at(i, false) << '\n';
    out << "SCALARS phin double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << value_at(i, true) << '\n';
    out << "SCALARS total_potential_e_per_A double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << total_potential_e_per_a_at(i) << '\n';
    out << "SCALARS total_potential_kT_e double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << potential_kt_e_at(i) << '\n';
    out << "SCALARS coulomb_potential_kT_e double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << coulomb_potential_kt_e_at(i) << '\n';
    out << "SCALARS reaction_potential_kT_e double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << reaction_potential_kt_e_at(i) << '\n';
    out << "SCALARS potential_kcal_mol_e double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << potential_kcal_mol_e_at(i) << '\n';
    out << "SCALARS potential_kT_e double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << potential_kt_e_at(i) << '\n';
    out << "SCALARS normal_flux_proxy double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << value_at(i, true) << '\n';
    out << "SCALARS area_A2 double 1\nLOOKUP_TABLE default\n";
    for (const Tabi::Panel &panel : solve.tabi_output_panels)
      out << panel.area << '\n';
    out << "VECTORS normal double\n";
    for (const Tabi::Panel &panel : solve.tabi_output_panels)
      out << panel.nx << ' ' << panel.ny << ' ' << panel.nz << '\n';
  }
  add_timing_ms(TimingBucket::MapIo,
                elapsed_ms(begin, std::chrono::steady_clock::now()));
}

#if 0 // Duplicate produced by the cross-refactor merge; retained only as context.
void AnalyzerImpl::maybe_write_tabi_surface_solution(
    const FrameResult &result, const ReactionSolve &solve) const {
  if (args_.backend != "tabi" || args_.tabi_surface_output == "off")
    return;
  const std::size_t panel_count = solve.tabi_output_panels.size();
  if (panel_count == 0)
    throw std::runtime_error(
        "PBE: TABI surface output requested but no panels were retained.");
  if (solve.tabi_output_solution.size() < panel_count)
    throw std::runtime_error("PBE: TABI surface output requested but no "
                             "solution vector was retained.");
  const bool coupled = solve.tabi_output_solution.size() >= 2 * panel_count;
  const bool write_csv =
      args_.tabi_surface_output == "csv" || args_.tabi_surface_output == "all";
  const bool write_vtk =
      args_.tabi_surface_output == "vtk" || args_.tabi_surface_output == "all";
  const auto begin = std::chrono::steady_clock::now();
  auto value_at = [&](std::size_t panel, bool phin) {
    if (phin)
      return coupled ? solve.tabi_output_solution[panel_count + panel] : 0.0;
    return solve.tabi_output_solution[panel];
  };
  if (write_csv) {
    std::ofstream out("pbe_tabi_surface_solution.csv",
                      std::ios::out | std::ios::trunc);
    if (!out.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_tabi_surface_solution.csv.");
    out << std::setprecision(17);
    out << "trajectory_id,frame_index,global_index,fixture,panel_id,atom_index,"
           "x_A,y_A,z_A,nx,ny,nz,area_A2,diagonal,solid_angle,phi,phin,"
           "normal_flux_proxy,"
           "v0x_A,v0y_A,v0z_A,v1x_A,v1y_A,v1z_A,v2x_A,v2y_A,v2z_A,"
           "has_triangle_geometry\n";
    for (std::size_t i = 0; i < panel_count; ++i) {
      const Tabi::Panel &panel = solve.tabi_output_panels[i];
      out << result.trajectory_id << ',' << result.frame_index << ','
          << result.global_index << ',' << csv_escape(current_source_label_)
          << ',' << i << ',' << panel.atom_index << ',' << panel.x << ','
          << panel.y << ',' << panel.z << ',' << panel.nx << ',' << panel.ny
          << ',' << panel.nz << ',' << panel.area << ',' << panel.diagonal
          << ',' << panel.solid_angle << ',' << value_at(i, false) << ','
          << value_at(i, true) << ',' << value_at(i, true) << ',' << panel.v0x
          << ',' << panel.v0y << ',' << panel.v0z << ',' << panel.v1x << ','
          << panel.v1y << ',' << panel.v1z << ',' << panel.v2x << ','
          << panel.v2y << ',' << panel.v2z << ','
          << (panel.has_triangle_geometry ? "true" : "false") << '\n';
    }
  }
  if (write_vtk) {
    std::ofstream out("pbe_tabi_surface_solution.vtk",
                      std::ios::out | std::ios::trunc);
    if (!out.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_tabi_surface_solution.vtk.");
    out << std::setprecision(17);
    out << "# vtk DataFile Version 3.0\n";
    out << "TITAN TABI surface solution\n";
    out << "ASCII\n";
    out << "DATASET POLYDATA\n";
    out << "POINTS " << (panel_count * 3u) << " double\n";
    for (const Tabi::Panel &panel : solve.tabi_output_panels) {
      out << panel.v0x << ' ' << panel.v0y << ' ' << panel.v0z << '\n';
      out << panel.v1x << ' ' << panel.v1y << ' ' << panel.v1z << '\n';
      out << panel.v2x << ' ' << panel.v2y << ' ' << panel.v2z << '\n';
    }
    out << "POLYGONS " << panel_count << ' ' << (panel_count * 4u) << "\n";
    for (std::size_t i = 0; i < panel_count; ++i) {
      const std::size_t base = 3u * i;
      out << "3 " << base << ' ' << (base + 1u) << ' ' << (base + 2u) << '\n';
    }
    out << "CELL_DATA " << panel_count << "\n";
    out << "SCALARS phi double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << value_at(i, false) << '\n';
    out << "SCALARS phin double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << value_at(i, true) << '\n';
    out << "SCALARS normal_flux_proxy double 1\nLOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < panel_count; ++i)
      out << value_at(i, true) << '\n';
    out << "SCALARS area_A2 double 1\nLOOKUP_TABLE default\n";
    for (const Tabi::Panel &panel : solve.tabi_output_panels)
      out << panel.area << '\n';
    out << "VECTORS normal double\n";
    for (const Tabi::Panel &panel : solve.tabi_output_panels)
      out << panel.nx << ' ' << panel.ny << ' ' << panel.nz << '\n';
  }
  add_timing_ms(TimingBucket::MapIo,
                elapsed_ms(begin, std::chrono::steady_clock::now()));
}
#endif


} // namespace TitanPBE
