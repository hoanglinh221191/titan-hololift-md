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

std::vector<double>
AnalyzerImpl::compute_sasa_per_atom(const std::vector<Atom> &atoms,
                                    TitanExecutor &executor) const {
  std::vector<TitanSASA::Atom> sasa_atoms;
  sasa_atoms.reserve(atoms.size());
  for (const Atom &atom : atoms)
    sasa_atoms.push_back({atom.x, atom.y, atom.z, atom.radius});

  TitanSASA::Options options;
  options.probe_radius = args_.probe_radius;
  options.min_cell_size = 1.0;
  options.point_count = args_.pbsa_sasa_points;
  return TitanSASA::compute_per_atom_sasa(
      std::span<const TitanSASA::Atom>(sasa_atoms.data(), sasa_atoms.size()),
      options, {}, &executor);
}

double AnalyzerImpl::compute_sasa_total(const std::vector<Atom> &atoms,
                                        TitanExecutor &executor) const {
  const std::vector<double> area_by_atom =
      compute_sasa_per_atom(atoms, executor);
  return TitanSASA::sum_sasa(
      std::span<const double>(area_by_atom.data(), area_by_atom.size()));
}

void AnalyzerImpl::write_energy_header_if_needed() {
  if (!energy_out_.is_open()) {
    energy_out_.open("pbe_energy.csv", std::ios::out | std::ios::trunc);
    if (!energy_out_.is_open())
      throw std::runtime_error("PBE: failed to open pbe_energy.csv.");
  }
  if (!energy_header_written_) {
    energy_out_
        << "trajectory_id,frame_index,global_index,time_ps,atom_count,"
           "total_charge,grid_nx,grid_ny,grid_nz,grid_points,memory_bytes,"
           "surface,probe_A,solute_points,solvent_points,"
           "salt_M,iterations_solvent,iterations_vacuum,rms_solvent,"
           "rms_vacuum,max_dphi_solvent,max_dphi_vacuum,status,"
           "E_rxn_kT,E_rxn_kJmol,E_rxn_kcalmol,deltaG_solv_kT,"
           "deltaG_solv_kJmol,deltaG_solv_kcalmol,G_solv_complex_kJmol,"
           "G_solv_mol1_kJmol,G_solv_mol2_kJmol,G_solv_complex_kcalmol,"
           "G_solv_mol1_kcalmol,G_solv_mol2_kcalmol,dgsolv_status,"
           "focus_enabled,focus_levels,focus_initial_scale,"
           "focus_initial_margin,iterations_solvent_final,"
           "iterations_vacuum_final,solver_mode,mg_levels,mg_cycles,mg_pre,"
           "mg_post,mg_coarse_iter,surface_effective,ion_access,"
           "ion_radius_A,ion_accessible_points,stern_excluded_points,"
           "face_refine,pbe_backend,pbe_model,pbe_model_effective,linear_seed_"
           "solver,"
           "nl_iterations,nl_blocks,nl_relax,nl_max_dphi,"
           "nl_clipped_points,nl_status,boundary_mode,"
           "test_boundary_convention,test_field_units,charge_assignment,"
           "rhs_scale,charge_assigned,charge_moment_error,boundary_rms,"
           "linear_residual,nl_residual,nl_continuation_steps,nl_backtracks,"
           "grid_origin_x_A,grid_origin_y_A,grid_origin_z_A,grid_center_x_A,"
           "grid_center_y_A,grid_center_z_A,grid_spacing_A,boundary_max,"
           "reaction_energy_source,tabi_charge_resolution_status,"
           "tabi_charge_min_d_over_h,tabi_charge_distance_A,tabi_charge_local_max_edge_A,"
           "tabi_charge_index0,tabi_charge_panel_index0,tabi_charge_min_d_over_atom_radius,"
           "tabi_charge_surface_inside_atom_count,tabi_charged_atom_count,"
           "tabi_charge_patch_evaluations,tabi_charge_resolution_ms,solvent_phase_rf_kT,"
           "solvent_phase_rf_kJmol,solvent_phase_rf_kcalmol,"
           "vacuum_phase_rf_kT,vacuum_phase_rf_kJmol,"
           "vacuum_phase_rf_kcalmol,atom_corrected_rxn_kT,"
           "atom_corrected_rxn_kJmol,atom_corrected_rxn_kcalmol,"
           "grid_corrected_rxn_kT,grid_corrected_rxn_kJmol,"
           "grid_corrected_rxn_kcalmol,coulomb_energy_kT,"
           "coulomb_energy_kJmol,coulomb_energy_kcalmol,pbe_biomodel,"
           "dielectric_model,nl_solver,nl_inner_solver,nl_mg_mode,"
           "gaussian_exp,gaussian_sigma,density_cutoff,surface_cutoff,"
           "gap_diel,radius_convention,focus_parent_grid,focus_child_grid,"
           "probe_status,newton_inner_iterations,newton_step_norm,"
           "newton_residual_before,newton_residual_after,"
           "newton_line_search_status,newton_correction_residual,"
           "newton_correction_max_residual,newton_accepted_lambda,"
           "newton_rejected_steps,newton_jac_diag_min,newton_jac_diag_max,"
           "newton_screen_points,newton_active_nonlinear_points,"
           "newton_stop_reason,gaussian_density_sum,"
           "gaussian_cutoff_points,gaussian_eps_min,gaussian_eps_max,"
           "gaussian_eps_mean,gaussian_charge_norm_error,"
           "newton_inner_levels,newton_inner_cycles,"
           "newton_inner_residual_before,newton_inner_residual_after,"
           "newton_inner_effective,iscm_boundary_points,"
           "iscm_charged_boundary_points,iscm_induced_charge_sum,"
           "iscm_induced_charge_l1,iscm_induced_charge_l2,"
           "iscm_induced_charge_max,iscm_water_rf_kT,"
           "iscm_water_rf_kJmol,iscm_water_rf_kcalmol,iscm_vacuum_rf_kT,"
           "iscm_vacuum_rf_kJmol,iscm_vacuum_rf_kcalmol,"
           "iscm_corrected_rxn_kT,iscm_corrected_rxn_kJmol,"
           "iscm_corrected_rxn_kcalmol,iscm_status,delphi_face_crossings,"
           "delphi_eps_face_matches,delphi_boundary_grid_points,"
           "delphi_charged_boundary_points,delphi_iscm_status,"
           "vdwms_enabled,vdwms_boundary_initial,vdwms_boundary_final,"
           "vdwms_added,vdwms_removed,vdwms_exposed_points,"
           "vdwms_midpoint_rms,iscm_water_minus_vacuum_kT,"
           "iscm_water_minus_vacuum_kJmol,"
           "iscm_water_minus_vacuum_kcalmol,iscm_finalization_mode,"
           "iscm_position_source,iscm_position_rms_vs_oracle,"
           "iscm_position_oracle_matches,iscm_native_independent,"
           "nonlinear_stress_kT,nonlinear_stress_kJmol,"
           "nonlinear_stress_kcalmol,nonlinear_osmotic_kT,"
           "nonlinear_osmotic_kJmol,nonlinear_osmotic_kcalmol,"
           "nonlinear_energy_correction_kT,"
           "nonlinear_energy_correction_kJmol,"
           "nonlinear_energy_correction_kcalmol,"
           "nonlinear_energy_status,rpbe_coulomb_norm,"
           "rpbe_grad_coulomb_norm,rpbe_grad_epsilon_norm,"
           "rpbe_source_rms,rpbe_source_max,rpbe_source_status\n";
    energy_header_written_ = true;
  }
}

void AnalyzerImpl::append_energy_row(const FrameResult &result) {
  write_energy_header_if_needed();
  energy_out_ << result.trajectory_id << ',' << result.frame_index << ','
              << result.global_index << ',' << std::setprecision(10)
              << result.time_ps << ',' << result.atom_count << ','
              << result.total_charge << ',' << result.nx << ',' << result.ny
              << ',' << result.nz << ',' << result.grid_points << ','
              << result.memory_bytes << ',' << args_.surface << ','
              << args_.probe_radius << ',' << result.solute_points << ','
              << result.solvent_points << ',' << args_.salt << ','
              << result.solvent_iterations << ',' << result.vacuum_iterations
              << ',' << result.solvent_rms << ',' << result.vacuum_rms << ','
              << result.solvent_max_dphi << ',' << result.vacuum_max_dphi << ','
              << result.status << ',' << result.energy_kt << ','
              << result.energy_kj_mol << ',' << result.energy_kcal_mol << ',';
  if (result.dgsolv_enabled) {
    energy_out_ << result.deltaG_solv_kt << ',' << result.deltaG_solv_kj_mol
                << ',' << result.deltaG_solv_kcal_mol << ','
                << result.complex_energy_kj_mol << ','
                << result.mol1_energy_kj_mol << ',' << result.mol2_energy_kj_mol
                << ',' << result.complex_energy_kcal_mol << ','
                << result.mol1_energy_kcal_mol << ','
                << result.mol2_energy_kcal_mol << ',' << result.dgsolv_status
                << ',';
  } else {
    energy_out_ << ",,,,,,,,,off,";
  }
  energy_out_
      << (result.focus_enabled ? "true" : "false") << ',' << result.focus_levels
      << ',' << result.focus_initial_scale << ',' << result.focus_initial_margin
      << ',' << result.solvent_iterations_final << ','
      << result.vacuum_iterations_final << ',' << result.solver_mode << ','
      << result.mg_levels << ',' << result.mg_cycles << ',' << result.mg_pre
      << ',' << result.mg_post << ',' << result.mg_coarse_iter << ','
      << result.surface_effective << ',' << result.ion_access << ','
      << result.ion_radius_A << ',' << result.ion_accessible_points << ','
      << result.stern_excluded_points << ','
      << (result.face_refine ? "true" : "false") << ',' << result.pbe_backend
      << ',' << result.pbe_model << ',' << result.pbe_model_effective << ','
      << result.linear_seed_solver << ',' << result.nl_iterations << ','
      << result.nl_blocks << ',' << result.nl_relax << ',' << result.nl_max_dphi
      << ',' << result.nl_clipped_points << ',' << result.nl_status << ','
      << result.boundary_mode << ',' << result.test_boundary_convention << ','
      << result.test_field_units << ',' << result.charge_assignment << ','
      << result.rhs_scale << ',' << result.charge_assigned << ','
      << result.charge_moment_error << ',' << result.boundary_rms << ','
      << result.linear_residual << ',' << result.nl_residual << ','
      << result.nl_continuation_steps << ',' << result.nl_backtracks << ','
      << result.grid_origin_x_A << ',' << result.grid_origin_y_A << ','
      << result.grid_origin_z_A << ',' << result.grid_center_x_A << ','
      << result.grid_center_y_A << ',' << result.grid_center_z_A << ','
      << result.grid_spacing_A << ',' << result.boundary_max << ','
      << result.reaction_energy_source << ','
      << result.tabi_charge_resolution.status << ','
      << result.tabi_charge_resolution.min_ratio << ','
      << result.tabi_charge_resolution.distance_a << ','
      << result.tabi_charge_resolution.local_edge_a << ','
      << result.tabi_charge_resolution.charge_index << ','
      << result.tabi_charge_resolution.panel_index << ','
      << result.tabi_charge_resolution.min_distance_over_atom_radius << ','
      << result.tabi_charge_resolution.surface_inside_atom_count << ','
      << result.tabi_charge_resolution.charged_atoms << ','
      << result.tabi_charge_resolution.patch_evaluations << ','
      << result.tabi_charge_resolution.milliseconds << ','
      << result.solvent_phase_kt << ','
      << result.solvent_phase_kj_mol << ',' << result.solvent_phase_kcal_mol
      << ',' << result.vacuum_phase_kt << ',' << result.vacuum_phase_kj_mol
      << ',' << result.vacuum_phase_kcal_mol << ',' << result.atom_corrected_kt
      << ',' << result.atom_corrected_kj_mol << ','
      << result.atom_corrected_kcal_mol << ',' << result.grid_corrected_kt
      << ',' << result.grid_corrected_kj_mol << ','
      << result.grid_corrected_kcal_mol << ',' << result.coulomb_energy_kt
      << ',' << result.coulomb_energy_kj_mol << ','
      << result.coulomb_energy_kcal_mol << ',' << result.pbe_biomodel << ','
      << result.dielectric_model << ',' << result.nl_solver << ','
      << result.nl_inner_solver << ',' << result.nl_mg_mode << ','
      << result.gaussian_exp << ',' << result.gaussian_sigma << ','
      << result.density_cutoff << ',' << result.surface_cutoff << ','
      << result.gap_diel << ',' << result.radius_convention << ','
      << csv_escape(result.focus_parent_grid) << ','
      << csv_escape(result.focus_child_grid) << ','
      << csv_escape(result.probe_status) << ','
      << result.newton_inner_iterations << ',' << result.newton_step_norm << ','
      << result.newton_residual_before << ',' << result.newton_residual_after
      << ',' << csv_escape(result.newton_line_search_status) << ','
      << result.newton_correction_residual << ','
      << result.newton_correction_max_residual << ','
      << result.newton_accepted_lambda << ',' << result.newton_rejected_steps
      << ',' << result.newton_jac_diag_min << ',' << result.newton_jac_diag_max
      << ',' << result.newton_screen_points << ','
      << result.newton_active_nonlinear_points << ','
      << csv_escape(result.newton_stop_reason) << ','
      << result.gaussian_density_sum << ',' << result.gaussian_cutoff_points
      << ',' << result.gaussian_eps_min << ',' << result.gaussian_eps_max << ','
      << result.gaussian_eps_mean << ',' << result.gaussian_charge_norm_error
      << ',' << result.newton_inner_levels << ',' << result.newton_inner_cycles
      << ',' << result.newton_inner_residual_before << ','
      << result.newton_inner_residual_after << ','
      << csv_escape(result.newton_inner_effective) << ','
      << result.iscm_boundary_points << ','
      << result.iscm_charged_boundary_points << ','
      << result.iscm_induced_charge_sum << ',' << result.iscm_induced_charge_l1
      << ',' << result.iscm_induced_charge_l2 << ','
      << result.iscm_induced_charge_max << ',' << result.iscm_water_phase_kt
      << ',' << result.iscm_water_phase_kj_mol << ','
      << result.iscm_water_phase_kcal_mol << ',' << result.iscm_vacuum_phase_kt
      << ',' << result.iscm_vacuum_phase_kj_mol << ','
      << result.iscm_vacuum_phase_kcal_mol << ',' << result.iscm_corrected_kt
      << ',' << result.iscm_corrected_kj_mol << ','
      << result.iscm_corrected_kcal_mol << ',' << csv_escape(result.iscm_status)
      << ',' << result.delphi_face_crossings << ','
      << result.delphi_eps_face_matches << ','
      << result.delphi_boundary_grid_points << ','
      << result.delphi_charged_boundary_points << ','
      << csv_escape(result.delphi_iscm_status) << ','
      << csv_escape(result.vdwms_enabled) << ','
      << result.vdwms_boundary_initial << ',' << result.vdwms_boundary_final
      << ',' << result.vdwms_added << ',' << result.vdwms_removed << ','
      << result.vdwms_exposed_points << ',' << result.vdwms_midpoint_rms << ','
      << result.iscm_water_minus_vacuum_kt << ','
      << result.iscm_water_minus_vacuum_kj_mol << ','
      << result.iscm_water_minus_vacuum_kcal_mol << ','
      << csv_escape(result.iscm_finalization_mode) << ','
      << csv_escape(result.iscm_position_source) << ','
      << result.iscm_position_rms_vs_oracle << ','
      << result.iscm_position_oracle_matches << ','
      << (result.iscm_native_independent ? "true" : "false") << ','
      << result.nonlinear_stress_kt << ',' << result.nonlinear_stress_kj_mol
      << ',' << result.nonlinear_stress_kcal_mol << ','
      << result.nonlinear_osmotic_kt << ',' << result.nonlinear_osmotic_kj_mol
      << ',' << result.nonlinear_osmotic_kcal_mol << ','
      << result.nonlinear_energy_correction_kt << ','
      << result.nonlinear_energy_correction_kj_mol << ','
      << result.nonlinear_energy_correction_kcal_mol << ','
      << csv_escape(result.nonlinear_energy_status) << ','
      << result.rpbe_coulomb_norm << ',' << result.rpbe_grad_coulomb_norm << ','
      << result.rpbe_grad_epsilon_norm << ',' << result.rpbe_source_rms << ','
      << result.rpbe_source_max << ',' << csv_escape(result.rpbe_source_status)
      << '\n';
}

bool AnalyzerImpl::pbe_timing_enabled() const noexcept {
  return args_.timing == "summary" || args_.timing == "detail";
}

void AnalyzerImpl::add_timing_ms(TimingBucket bucket,
                                 double ms) const noexcept {
  if (active_timing_ == nullptr || ms <= 0.0 || !std::isfinite(ms))
    return;
  switch (bucket) {
  case TimingBucket::Setup:
    active_timing_->setup_ms += ms;
    break;
  case TimingBucket::Surface:
    active_timing_->surface_ms += ms;
    break;
  case TimingBucket::Epsilon:
    active_timing_->epsilon_ms += ms;
    break;
  case TimingBucket::Boundary:
    active_timing_->boundary_ms += ms;
    break;
  case TimingBucket::Solver:
    active_timing_->solver_ms += ms;
    break;
  case TimingBucket::Energy:
    active_timing_->energy_ms += ms;
    break;
  case TimingBucket::MapIo:
    active_timing_->map_io_ms += ms;
    break;
  case TimingBucket::RpbSource:
    active_timing_->rpb_source_ms += ms;
    break;
  case TimingBucket::FocusSetup:
    active_timing_->focus_setup_ms += ms;
    break;
  case TimingBucket::Probe:
    active_timing_->probe_ms += ms;
    break;
  case TimingBucket::FocusParentSolve:
    active_timing_->focus_parent_solve_ms += ms;
    break;
  case TimingBucket::FocusChildSetup:
    active_timing_->focus_child_setup_ms += ms;
    break;
  case TimingBucket::FocusBoundaryInterp:
    active_timing_->focus_boundary_interp_ms += ms;
    break;
  case TimingBucket::FocusChildSurface:
    active_timing_->focus_child_surface_ms += ms;
    break;
  case TimingBucket::FocusChildEpsilon:
    active_timing_->focus_child_epsilon_ms += ms;
    break;
  case TimingBucket::FocusChildSolver:
    active_timing_->focus_child_solver_ms += ms;
    break;
  case TimingBucket::ProbeEnergy:
    active_timing_->probe_energy_ms += ms;
    break;
  case TimingBucket::AuditOutput:
    active_timing_->audit_output_ms += ms;
    break;
  case TimingBucket::NwtHalfUpdate:
    active_timing_->nwt_half_update_ms += ms;
    break;
  case TimingBucket::NwtRmsReduce:
    active_timing_->nwt_rms_reduce_ms += ms;
    break;
  case TimingBucket::NwtOperatorResidual:
    active_timing_->nwt_operator_residual_ms += ms;
    break;
  case TimingBucket::NwtSourceUpdate:
    active_timing_->nwt_source_update_ms += ms;
    break;
  case TimingBucket::NwtProgressIo:
    active_timing_->nwt_progress_io_ms += ms;
    break;
  case TimingBucket::NwtCoeffPack:
    active_timing_->nwt_coeff_pack_ms += ms;
    break;
  case TimingBucket::NwtHomoUpdate:
    active_timing_->nwt_homo_update_ms += ms;
    break;
  case TimingBucket::NwtHeteroUpdate:
    active_timing_->nwt_hetero_update_ms += ms;
    break;
  case TimingBucket::NwtNonlinTerm:
    active_timing_->nwt_nonlin_term_ms += ms;
    break;
  case TimingBucket::NwtReduce:
    active_timing_->nwt_reduce_ms += ms;
    break;
  case TimingBucket::NwtHomoStencil:
    active_timing_->nwt_homo_stencil_ms += ms;
    break;
  case TimingBucket::NwtHomoSaltMath:
    active_timing_->nwt_homo_salt_math_ms += ms;
    break;
  case TimingBucket::NwtHomoWrite:
    active_timing_->nwt_homo_write_ms += ms;
    break;
  case TimingBucket::NwtThreadWait:
    active_timing_->nwt_thread_wait_ms += ms;
    break;
  case TimingBucket::NwtBlockOverhead:
    active_timing_->nwt_block_overhead_ms += ms;
    break;
  case TimingBucket::LinearHalfUpdate:
    active_timing_->linear_half_update_ms += ms;
    break;
  case TimingBucket::EpsilonCenterDensity:
    active_timing_->epsilon_center_density_ms += ms;
    break;
  case TimingBucket::EpsilonMidpointDensity:
    active_timing_->epsilon_midpoint_density_ms += ms;
    break;
  case TimingBucket::EpsilonIonExclusion:
    active_timing_->epsilon_ion_exclusion_ms += ms;
    break;
  case TimingBucket::EpsilonFinalize:
    active_timing_->epsilon_finalize_ms += ms;
    break;
  }
}

void AnalyzerImpl::write_timing_header_if_needed() {
  if (!pbe_timing_enabled())
    return;
  if (!timing_out_.is_open()) {
    timing_out_.open("pbe_timing.csv", std::ios::out | std::ios::trunc);
    if (!timing_out_.is_open())
      throw std::runtime_error("PBE: failed to open pbe_timing.csv.");
  }
  if (!timing_header_written_) {
    timing_out_
        << "trajectory_id,frame_index,global_index,time_ps,fixture,mode,"
           "pbe_backend,atom_count,focus_child_atom_count,num_threads,scope,"
           "tabi_surface_build_seconds,tabi_surface_panels,"
           "tabi_surface_point_count_requested,"
           "tabi_panel_density_per_A2,tabi_panel_budget_status,"
           "tabi_surface_exposed_points,tabi_surface_buried_points,"
           "tabi_tree_build_seconds,tabi_tree_nodes,tabi_tree_leaf_count,"
           "tabi_near_interactions,tabi_far_interactions,tabi_matvec_mode,"
           "tabi_status,tabi_convergence_reason,tabi_reaction_energy_source,"
           "tabi_gmres_restart_requested,tabi_gmres_restart_effective,"
           "tabi_best_residual,tabi_best_residual_iter,tabi_final_residual,"
           "tabi_best_preconditioned_residual,"
           "tabi_final_preconditioned_residual,"
           "tabi_true_residual_abs,tabi_true_residual_rel,"
           "tabi_true_residual_probe_mode,"
           "tabi_true_residual_probe_scope,"
           "tabi_true_residual_probe_interval,"
           "tabi_true_residual_sample_count,"
           "tabi_true_residual_probe_calls,"
           "tabi_true_residual_probe_seconds,"
           "tabi_sampled_true_residual_rel,"
           "tabi_sampled_true_residual_rows,"
           "tabi_sampled_probe_calls,"
           "tabi_sampled_probe_seconds,"
           "tabi_full_true_residual_confirm_calls,"
           "tabi_full_probe_seconds,"
           "tabi_rhs_seconds,tabi_gmres_seconds,"
           "tabi_matvec_seconds_total,tabi_matvec_seconds_avg,"
           "tabi_matvec_calls,tabi_tree_vs_direct_matvec_rel,"
           "tabi_matvec_audit_rows,tabi_matvec_audit_seconds,"
           "tabi_orthogonalization_seconds,"
           "tabi_preconditioner_seconds,tabi_surface_neighbor_seconds,"
           "tabi_surface_mesh_seconds,tabi_solid_angle_seconds,"
           "tabi_tree_moment_seconds,tabi_tree_traversal_seconds,"
           "tabi_matvec_cost_profile,tabi_leaf_panels_per_second,"
           "tabi_clean_leaf_panels_per_second,tabi_tree_traversal_fraction,"
           "tabi_near_eval_seconds,tabi_far_eval_seconds,"
           "tabi_singular_near_eval_seconds,tabi_leaf_near_eval_seconds,"
           "tabi_leaf_panel_eval_seconds,"
           "tabi_leaf_tile_eval_seconds,tabi_leaf_residual_eval_seconds,"
           "tabi_smooth_leaf_node_eval_seconds,"
           "tabi_leaf_panel_kernel,tabi_leaf_tile_status,"
           "tabi_leaf_panel_pairs_per_second,"
           "tabi_singular_near_interactions,tabi_leaf_near_interactions,"
           "tabi_far_node_interactions,"
           "tabi_leaf_panel_interactions,tabi_leaf_tile_interactions,"
           "tabi_leaf_tile_panel_interactions,"
           "tabi_leaf_residual_panel_interactions,"
           "tabi_leaf_work_profile,tabi_leaf_work_mac_terminal_panels,"
           "tabi_leaf_work_physical_gap_panels,"
           "tabi_leaf_work_fragmentation_panels,"
           "tabi_leaf_work_convergence_panels,"
           "tabi_leaf_work_convergence_estimator_panels,"
           "tabi_leaf_work_convergence_rho_panels,"
           "tabi_leaf_work_convergence_aspect_panels,"
           "tabi_leaf_work_surface_margin_panels,"
           "tabi_leaf_work_surface_margin_hard_panels,"
           "tabi_leaf_work_surface_margin_soft_panels,"
           "tabi_leaf_work_k3_rho_cap_panels,"
           "tabi_leaf_work_singular_fallback_panels,"
           "tabi_leaf_work_scalar_tail_panels,"
           "tabi_leaf_work_profile_accounted_panels,"
           "tabi_leaf_work_mac_terminal_frac,"
           "tabi_leaf_work_convergence_frac,"
           "tabi_leaf_work_surface_margin_frac,"
           "tabi_leaf_work_singular_fallback_frac,"
           "tabi_leaf_work_scalar_tail_frac,"
           "tabi_smooth_leaf_node_interactions,"
           "tabi_leaf_pruned_interactions,"
           "tabi_leaf_mac_candidates,"
           "tabi_leaf_mac_accepted,"
           "tabi_leaf_mac_rejected,"
           "tabi_leaf_mac_far_panels,"
           "tabi_leaf_mac_direct_panels_saved_estimate,"
           "tabi_adaptive_smooth_leaf_accepted,"
           "tabi_adaptive_smooth_leaf_rejected,"
           "tabi_adaptive_span_tile_accepted,"
           "tabi_adaptive_span_tile_rejected,"
           "tabi_adaptive_exact_leaf_fallbacks,"
           "tabi_adaptive_reject_one_ring_overlap,"
           "tabi_adaptive_reject_error_bound_failed,"
           "tabi_adaptive_reject_p4_check_failed,"
           "tabi_adaptive_reject_too_close,"
           "tabi_leaf_work_reduction_ratio,"
           "tabi_target_cluster_interactions,"
           "tabi_target_cluster_targets,"
           "tabi_target_cluster_panels_pruned,"
           "tabi_target_cluster_rejected,"
           "tabi_target_cluster_eval_seconds,"
           "tabi_leaf_pair_batch_interactions,"
           "tabi_leaf_pair_batch_targets,"
           "tabi_leaf_pair_batch_source_panels,"
           "tabi_leaf_pair_batch_residual_panels,"
           "tabi_leaf_pair_batch_fallbacks,"
           "tabi_leaf_pair_batch_seconds,"
           "tabi_leaf_pair_batch_kernel,"
           "tabi_leaf_pair_batch_vectorized_sources,"
           "tabi_leaf_pair_batch_vector_lanes,"
           "tabi_leaf_pair_batch_pairs_per_second,"
           "tabi_leaf_kernel_profile,"
           "tabi_jcp_direct_kernel,"
           "tabi_leaf_kernel_scalar_tasks,"
           "tabi_leaf_kernel_batch_tasks,"
           "tabi_leaf_kernel_source_panel_evals,"
           "tabi_leaf_kernel_scalar_seconds,"
           "tabi_leaf_kernel_target_avx2_seconds,"
           "tabi_leaf_kernel_tail_seconds,"
           "tabi_leaf_pair_batch_soa_sources,"
           "tabi_leaf_pair_batch_soa_seconds,"
           "tabi_leaf_pair_batch_scalar_tail_seconds,"
           "tabi_leaf_pair_batch_clean_source_panels_per_second,"
           "tabi_leaf_pair_selection_profile,"
           "tabi_leaf_pair_probe_cache,"
           "tabi_leaf_pair_target_group,"
           "tabi_leaf_pair_max_lookahead,"
           "tabi_leaf_pair_selection_candidate_scans,"
           "tabi_leaf_pair_selection_clean_rejects,"
           "tabi_leaf_pair_selection_probe_calls,"
           "tabi_leaf_pair_selection_probe_node_visits,"
           "tabi_leaf_pair_selection_probe_seconds,"
           "tabi_leaf_pair_selection_seconds,"
           "tabi_leaf_pair_selection_probe_cache_hits,"
           "tabi_leaf_pair_selection_probe_cache_misses,"
           "tabi_leaf_pair_selection_insufficient_targets,"
           "tabi_leaf_pair_selection_same_target_leaf_rejects,"
           "tabi_leaf_pair_directness_probe,"
           "tabi_leaf_pair_directness_cache,"
           "tabi_leaf_pair_directness_verify,"
           "tabi_leaf_pair_directness_ancestor_probe_calls,"
           "tabi_leaf_pair_directness_ancestor_probe_node_visits,"
           "tabi_leaf_pair_directness_ancestor_probe_seconds,"
           "tabi_leaf_pair_directness_pair_cache_hits,"
           "tabi_leaf_pair_directness_pair_cache_misses,"
           "tabi_leaf_pair_directness_all_direct_proofs,"
           "tabi_leaf_pair_directness_mixed_fallbacks,"
           "tabi_leaf_pair_directness_verify_checks,"
           "tabi_leaf_pair_directness_verify_mismatches,"
           "tabi_leaf_pair_scheduler,"
           "tabi_leaf_pair_worklist_max_active,"
           "tabi_leaf_pair_worklist_verify,"
           "tabi_leaf_pair_worklist_tasks,"
           "tabi_leaf_pair_worklist_groups,"
           "tabi_leaf_pair_worklist_batched_groups,"
           "tabi_leaf_pair_worklist_scalar_flush_tasks,"
           "tabi_leaf_pair_worklist_forced_flushes,"
           "tabi_leaf_pair_worklist_active_groups_max,"
           "tabi_leaf_pair_worklist_seconds,"
           "tabi_leaf_pair_worklist_verify_checks,"
           "tabi_leaf_pair_worklist_verify_mismatches,"
           "tabi_leaf_block_verify,"
           "tabi_leaf_block_target_leaves,"
           "tabi_leaf_block_targets,"
           "tabi_leaf_block_source_groups,"
           "tabi_leaf_block_batched_groups,"
           "tabi_leaf_block_scalar_tail_tasks,"
           "tabi_leaf_block_group_seconds,"
           "tabi_leaf_block_flush_seconds,"
           "tabi_leaf_block_writeback_seconds,"
           "tabi_leaf_block_verify_checks,"
           "tabi_leaf_block_verify_mismatches,"
           "tabi_tree_preset,tabi_adaptive_level,tabi_cluster_policy,"
           "tabi_target_batch_size,tabi_leaf_pair_min_source_panels,"
           "tabi_leaf_pair_min_targets,tabi_leaf_near_policy,"
           "tabi_interaction_mode,tabi_node_bounds_mode,"
           "tabi_tree_builder,tabi_expansion_center,"
           "tabi_fragmentation_safety,tabi_mac_safety,"
           "tabi_mac_gap_factor,"
           "tabi_tree_depth_max,"
           "tabi_mac_reject_radius_limited,"
           "tabi_mac_accept_rate,tabi_node_radius_max,"
           "tabi_node_radius_mean,tabi_node_aspect_ratio_max,"
           "tabi_node_aspect_ratio_mean,tabi_leaf_panel_work_per_matvec,"
           "tabi_far_node_work_per_matvec,"
           "tabi_far_geometric_unsafe_count,"
           "tabi_far_geometric_unsafe_panels,"
           "tabi_far_geometric_min_gap,"
           "tabi_far_geometric_gap_ratio_min,"
           "tabi_far_geometric_gap_ratio_mean,"
           "tabi_far_geometric_audit_samples,"
           "tabi_mac_physical_gap_rejects,"
           "tabi_mac_convergence_rho_rejects,"
           "tabi_mac_convergence_aspect_rejects,"
           "tabi_mac_convergence_rho_reject_panels,"
           "tabi_mac_convergence_aspect_reject_panels,"
           "tabi_mac_surface_margin_rejects,"
           "tabi_mac_surface_margin_reject_panels,"
           "tabi_mac_surface_margin_applied,"
           "tabi_mac_surface_margin_skipped,"
           "tabi_mac_surface_margin_adaptive_rejects,"
           "tabi_mac_surface_margin_adaptive_reject_panels,"
           "tabi_mac_surface_margin_skipped_accepted_far,"
           "tabi_mac_surface_margin_hard_applied,"
           "tabi_mac_surface_margin_soft_applied,"
           "tabi_mac_surface_margin_hard_rejects,"
           "tabi_mac_surface_margin_hard_reject_panels,"
           "tabi_mac_surface_margin_soft_rejects,"
           "tabi_mac_surface_margin_soft_reject_panels,"
           "tabi_fragmented_far_count,"
           "tabi_fragmented_far_panels,"
           "tabi_fragmented_split_rejects,"
           "tabi_fragmentation_center_gap_ratio_max,"
           "tabi_fragmentation_center_gap_ratio_mean,"
           "tabi_moment_audit,tabi_moment_construction,"
           "tabi_moment_sum,tabi_convergence_audit,"
           "tabi_convergence_safety,tabi_convergence_rho_cap,"
           "tabi_mac_aspect_cap,tabi_convergence_estimator,"
           "tabi_convergence_tail_threshold,"
           "tabi_convergence_estimator_error_cap,"
           "tabi_convergence_unstable_rho_cap,"
           "tabi_convergence_calibration,"
           "tabi_convergence_dielectric_amp,"
           "tabi_convergence_estimator_model,"
           "tabi_convergence_channel_calibration,"
           "tabi_convergence_split_policy,"
           "tabi_convergence_split_selectivity,"
           "tabi_convergence_split_absolute_floor,"
           "tabi_convergence_overreject_audit,"
           "tabi_convergence_overreject_detail,"
           "tabi_convergence_shape_audit,"
           "tabi_convergence_kernel_response_audit,"
           "tabi_channel_missed_semantics,"
           "tabi_accepted_miss_diagnostics,"
           "tabi_accepted_miss_absolute_floor,"
           "tabi_convergence_k3_accepted_rho_cap,"
           "tabi_mac_distance_policy,tabi_mac_surface_margin_factor,"
           "tabi_mac_surface_margin_policy,"
           "tabi_mac_surface_margin_aspect_threshold,"
           "tabi_mac_surface_margin_rho_trigger,"
           "tabi_mac_surface_margin_soft_aspect_threshold,"
           "tabi_mac_surface_margin_soft_rho_trigger,"
           "tabi_mac_surface_margin_soft_factor,"
           "tabi_convergence_estimator_samples,"
           "tabi_convergence_estimator_unstable_far_count,"
           "tabi_convergence_estimator_unstable_far_panels,"
           "tabi_convergence_estimator_split_rejects,"
           "tabi_convergence_estimator_split_reject_panels,"
           "tabi_convergence_low_abs_reaccepted_far_count,"
           "tabi_convergence_low_abs_reaccepted_far_panels,"
           "tabi_convergence_estimator_tail_ratio_max,"
           "tabi_convergence_estimator_tail_ratio_mean,"
           "tabi_convergence_estimator_error_max,"
           "tabi_convergence_estimator_error_mean,"
           "tabi_convergence_estimator_abs_error_max,"
           "tabi_convergence_estimator_abs_error_mean,"
           "tabi_convergence_tail_norm_p_max,"
           "tabi_convergence_tail_norm_p_mean,"
           "tabi_convergence_tail_norm_pm1_max,"
           "tabi_convergence_tail_norm_pm1_mean,"
           "tabi_convergence_effective_jcp_amp_max,"
           "tabi_convergence_effective_jcp_amp_mean,"
           "tabi_convergence_dielectric_amp_max,"
           "tabi_convergence_dielectric_amp_mean,"
           "tabi_convergence_k1_effective_jcp_amp_max,"
           "tabi_convergence_k2_effective_jcp_amp_max,"
           "tabi_convergence_k3_effective_jcp_amp_max,"
           "tabi_convergence_k4_effective_jcp_amp_max,"
           "tabi_convergence_k1_effective_jcp_amp_mean,"
           "tabi_convergence_k2_effective_jcp_amp_mean,"
           "tabi_convergence_k3_effective_jcp_amp_mean,"
           "tabi_convergence_k4_effective_jcp_amp_mean,"
           "tabi_convergence_k1_estimated_error_max,"
           "tabi_convergence_k2_estimated_error_max,"
           "tabi_convergence_k3_estimated_error_max,"
           "tabi_convergence_k4_estimated_error_max,"
           "tabi_convergence_k1_unstable_far_count,"
           "tabi_convergence_k2_unstable_far_count,"
           "tabi_convergence_k3_unstable_far_count,"
           "tabi_convergence_k4_unstable_far_count,"
           "tabi_convergence_k1_unstable_far_panels,"
           "tabi_convergence_k2_unstable_far_panels,"
           "tabi_convergence_k3_unstable_far_panels,"
           "tabi_convergence_k4_unstable_far_panels,"
           "tabi_convergence_k1_split_rejects,"
           "tabi_convergence_k2_split_rejects,"
           "tabi_convergence_k3_split_rejects,"
           "tabi_convergence_k4_split_rejects,"
           "tabi_convergence_k1_split_reject_panels,"
           "tabi_convergence_k2_split_reject_panels,"
           "tabi_convergence_k3_split_reject_panels,"
           "tabi_convergence_k4_split_reject_panels,"
           "tabi_convergence_k1_low_abs_reaccepted,"
           "tabi_convergence_k2_low_abs_reaccepted,"
           "tabi_convergence_k3_low_abs_reaccepted,"
           "tabi_convergence_k4_low_abs_reaccepted,"
           "tabi_convergence_k1_low_abs_reaccepted_panels,"
           "tabi_convergence_k2_low_abs_reaccepted_panels,"
           "tabi_convergence_k3_low_abs_reaccepted_panels,"
           "tabi_convergence_k4_low_abs_reaccepted_panels,"
           "tabi_convergence_k1_high_error_caught,"
           "tabi_convergence_k2_high_error_caught,"
           "tabi_convergence_k3_high_error_caught,"
           "tabi_convergence_k4_high_error_caught,"
           "tabi_convergence_k1_high_error_missed,"
           "tabi_convergence_k2_high_error_missed,"
           "tabi_convergence_k3_high_error_missed,"
           "tabi_convergence_k4_high_error_missed,"
           "tabi_convergence_k1_high_error_caught_by_split,"
           "tabi_convergence_k2_high_error_caught_by_split,"
           "tabi_convergence_k3_high_error_caught_by_split,"
           "tabi_convergence_k4_high_error_caught_by_split,"
           "tabi_convergence_estimator_high_error_caught,"
           "tabi_convergence_estimator_high_error_missed,"
           "tabi_convergence_estimator_high_error_caught_by_split,"
           "tabi_convergence_estimator_phantom_high_error_count,"
           "tabi_convergence_estimator_phantom_high_error_abs_max,"
           "tabi_convergence_overreject_count,"
           "tabi_convergence_overreject_panels,"
           "tabi_convergence_overreject_caught_count,"
           "tabi_convergence_overreject_caught_panels,"
           "tabi_convergence_overreject_abs_error_max,"
           "tabi_convergence_overreject_abs_error_mean,"
           "tabi_convergence_overreject_abs_error_p95,"
           "tabi_convergence_overreject_abs_error_p99,"
           "tabi_convergence_overreject_estimated_abs_error_max,"
           "tabi_convergence_overreject_estimated_abs_error_mean,"
           "tabi_convergence_overreject_required_scale_max,"
           "tabi_convergence_overreject_required_scale_p95,"
           "tabi_convergence_overreject_required_scale_p99,"
           "tabi_convergence_k1_overreject_count,"
           "tabi_convergence_k2_overreject_count,"
           "tabi_convergence_k3_overreject_count,"
           "tabi_convergence_k4_overreject_count,"
           "tabi_convergence_k1_overreject_caught_count,"
           "tabi_convergence_k2_overreject_caught_count,"
           "tabi_convergence_k3_overreject_caught_count,"
           "tabi_convergence_k4_overreject_caught_count,"
           "tabi_convergence_k1_overreject_panels,"
           "tabi_convergence_k2_overreject_panels,"
           "tabi_convergence_k3_overreject_panels,"
           "tabi_convergence_k4_overreject_panels,"
           "tabi_convergence_k1_overreject_abs_error_max,"
           "tabi_convergence_k2_overreject_abs_error_max,"
           "tabi_convergence_k3_overreject_abs_error_max,"
           "tabi_convergence_k4_overreject_abs_error_max,"
           "tabi_convergence_k1_overreject_abs_error_mean,"
           "tabi_convergence_k2_overreject_abs_error_mean,"
           "tabi_convergence_k3_overreject_abs_error_mean,"
           "tabi_convergence_k4_overreject_abs_error_mean,"
           "tabi_convergence_k1_overreject_required_scale_max,"
           "tabi_convergence_k2_overreject_required_scale_max,"
           "tabi_convergence_k3_overreject_required_scale_max,"
           "tabi_convergence_k4_overreject_required_scale_max,"
           "tabi_convergence_k1_overreject_required_scale_p99,"
           "tabi_convergence_k2_overreject_required_scale_p99,"
           "tabi_convergence_k3_overreject_required_scale_p99,"
           "tabi_convergence_k4_overreject_required_scale_p99,"
           "tabi_phantom_high_error_k1_count,"
           "tabi_phantom_high_error_k2_count,"
           "tabi_phantom_high_error_k3_count,"
           "tabi_phantom_high_error_k4_count,"
           "tabi_phantom_high_error_k1_abs_max,"
           "tabi_phantom_high_error_k2_abs_max,"
           "tabi_phantom_high_error_k3_abs_max,"
           "tabi_phantom_high_error_k4_abs_max,"
           "tabi_accepted_miss_worst_channel,"
           "tabi_accepted_miss_k1_count,tabi_accepted_miss_k2_count,"
           "tabi_accepted_miss_k3_count,tabi_accepted_miss_k4_count,"
           "tabi_accepted_miss_k1_panels,tabi_accepted_miss_k2_panels,"
           "tabi_accepted_miss_k3_panels,tabi_accepted_miss_k4_panels,"
           "tabi_accepted_miss_k1_rho_min,tabi_accepted_miss_k2_rho_min,"
           "tabi_accepted_miss_k3_rho_min,tabi_accepted_miss_k4_rho_min,"
           "tabi_accepted_miss_k1_rho_mean,"
           "tabi_accepted_miss_k2_rho_mean,"
           "tabi_accepted_miss_k3_rho_mean,"
           "tabi_accepted_miss_k4_rho_mean,"
           "tabi_accepted_miss_k1_rho_max,tabi_accepted_miss_k2_rho_max,"
           "tabi_accepted_miss_k3_rho_max,tabi_accepted_miss_k4_rho_max,"
           "tabi_accepted_miss_k1_tail_ratio_mean,"
           "tabi_accepted_miss_k2_tail_ratio_mean,"
           "tabi_accepted_miss_k3_tail_ratio_mean,"
           "tabi_accepted_miss_k4_tail_ratio_mean,"
           "tabi_accepted_miss_k1_tail_ratio_max,"
           "tabi_accepted_miss_k2_tail_ratio_max,"
           "tabi_accepted_miss_k3_tail_ratio_max,"
           "tabi_accepted_miss_k4_tail_ratio_max,"
           "tabi_accepted_miss_k1_estimated_error_mean,"
           "tabi_accepted_miss_k2_estimated_error_mean,"
           "tabi_accepted_miss_k3_estimated_error_mean,"
           "tabi_accepted_miss_k4_estimated_error_mean,"
           "tabi_accepted_miss_k1_estimated_error_max,"
           "tabi_accepted_miss_k2_estimated_error_max,"
           "tabi_accepted_miss_k3_estimated_error_max,"
           "tabi_accepted_miss_k4_estimated_error_max,"
           "tabi_convergence_k3_rho_cap_rejects,"
           "tabi_convergence_k3_rho_cap_reject_panels,"
           "tabi_moment_audit_nodes,"
           "tabi_moment_translation_rel_error_max,"
           "tabi_moment_translation_rel_error_mean,"
           "tabi_moment_translation_offset_max,"
           "tabi_moment_translation_offset_mean,"
           "tabi_moment_cancellation_ratio_max,"
           "tabi_moment_cancellation_ratio_mean,"
           "tabi_moment_dynamic_range_max,"
           "tabi_expansion_audit,tabi_expansion_audit_channel,"
           "tabi_far_expansion_audit_pairs,"
           "tabi_far_expansion_audit_panels,"
           "tabi_far_expansion_direct_seconds,"
           "tabi_far_expansion_rel_error_max,"
           "tabi_far_expansion_rel_error_mean,"
           "tabi_far_expansion_rel_error_p95,"
           "tabi_far_expansion_rel_error_p99,"
           "tabi_far_expansion_worst_channel,"
           "tabi_far_expansion_k1_rel_error_max,"
           "tabi_far_expansion_k2_rel_error_max,"
           "tabi_far_expansion_k3_rel_error_max,"
           "tabi_far_expansion_k4_rel_error_max,"
           "tabi_far_expansion_k1_rel_error_mean,"
           "tabi_far_expansion_k2_rel_error_mean,"
           "tabi_far_expansion_k3_rel_error_mean,"
           "tabi_far_expansion_k4_rel_error_mean,"
           "tabi_far_expansion_k1_rel_error_p99,"
           "tabi_far_expansion_k2_rel_error_p99,"
           "tabi_far_expansion_k3_rel_error_p99,"
           "tabi_far_expansion_k4_rel_error_p99,"
           "tabi_far_expansion_mean_gate_pass,"
           "tabi_far_convergence_rho_max,"
           "tabi_far_convergence_rho_mean,"
           "tabi_accepted_far_aspect_ratio_max,"
           "tabi_accepted_far_aspect_ratio_mean,"
           "tabi_quad_buffer_auto_candidate,"
           "tabi_near_integration,tabi_quad_buffer,"
           "tabi_near_self_pairs,"
           "tabi_near_one_ring_pairs,tabi_near_duffy_pairs,"
           "tabi_near_centroid_pairs,tabi_near_fallbacks,"
           "tabi_near_geometric_buffer_pairs,"
           "tabi_near_geometric_buffer_gauss_pairs,"
           "tabi_near_geometric_buffer_adaptive_pairs,"
           "tabi_near_geometric_buffer_centroid_pairs,"
           "tabi_near_geometric_buffer_min_gap,"
           "tabi_near_geometric_buffer_mean_gap_ratio,"
           "tabi_near_geometric_buffer_max_edge_mean,"
           "tabi_near_integration_seconds,tabi_duffy_subdivisions_max,"
           "tabi_duffy_subdivisions_mean,tabi_near_kernel_policy,"
           "tabi_near_k1_accepted,tabi_near_k1_rejected,"
           "tabi_near_k2_accepted,tabi_near_k2_rejected,"
           "tabi_near_k3_accepted,tabi_near_k3_rejected,"
           "tabi_near_k4_accepted,tabi_near_k4_rejected,"
           "tabi_near_one_ring_k1_accepted,"
           "tabi_near_one_ring_k1_rejected,"
           "tabi_near_one_ring_k2_accepted,"
           "tabi_near_one_ring_k2_rejected,"
           "tabi_near_one_ring_k3_accepted,"
           "tabi_near_one_ring_k3_rejected,"
           "tabi_near_one_ring_k4_accepted,"
           "tabi_near_one_ring_k4_rejected,"
           "tabi_near_k1_policy,tabi_near_k2_policy,"
           "tabi_near_k3_policy,tabi_near_k4_policy,"
           "tabi_surface_geometry,tabi_mesh_vertices,tabi_mesh_triangles,"
           "tabi_boundary_nodes,tabi_avg_valence,tabi_nonmanifold_edges,"
           "tabi_solid_angle_fallbacks,tabi_diag_min,tabi_diag_max,"
           "tabi_taylor_order,tabi_derivative_mode,"
           "tabi_interaction_build_seconds,"
           "tabi_tree_eval_seconds,"
           "tabi_formulation,tabi_energy_projection,"
           "tabi_energy_projection_seconds,tabi_energy_projection_kernel,"
           "tabi_preconditioner_requested,tabi_preconditioner_effective,"
           "tabi_preconditioner_block_near,"
           "tabi_preconditioner_block_panel_cap,"
           "tabi_preconditioner_overlap_panel_cap,"
           "tabi_preconditioner_ilu_neighbor_cap_requested,"
           "tabi_preconditioner_ilu_neighbor_cap,"
           "tabi_preconditioner_ilu_drop_tol,"
           "tabi_preconditioner_ilu_regularization,"
           "tabi_preconditioner_ilu_regularization_max,"
           "tabi_preconditioner_ilu_regularization_selected,"
           "tabi_preconditioner_ilu_regularization_absolute_max,"
           "tabi_preconditioner_ilu_regularization_attempts,"
           "tabi_preconditioner_ilu_regularization_rejected_attempts,"
           "tabi_preconditioner_ilu_regularization_shifted_rows,"
           "tabi_preconditioner_ilu_unshifted_failure_stage,"
           "tabi_preconditioner_memory_budget_bytes,"
           "tabi_preconditioner_estimated_bytes,"
           "tabi_preconditioner_valid_blocks,"
           "tabi_preconditioner_valid_panels,"
           "tabi_preconditioner_ill_conditioned_blocks,"
           "tabi_preconditioner_quality_rejected_blocks,"
           "tabi_preconditioner_quality_probe_max_gain,"
           "tabi_preconditioner_quality_probe_rel_residual,"
           "tabi_preconditioner_pivot_ratio_min,"
           "tabi_preconditioner_factor_growth_max,"
           "tabi_preconditioner_pivot_ratio_min_row,"
           "tabi_preconditioner_pivot_ratio_min_column,"
           "tabi_preconditioner_factor_growth_row,"
           "tabi_preconditioner_factor_growth_column,"
           "tabi_preconditioner_quality_failure_stage,"
           "tabi_preconditioner_quality_failure_row,"
           "tabi_preconditioner_quality_failure_column,"
           "tabi_preconditioner_quality_failure_probe,"
           "tabi_preconditioner_quality_failure_probe_gain,"
           "tabi_preconditioner_quality_failure_probe_rel_residual,"
           "tabi_preconditioner_quality_status,"
           "tabi_preconditioner_setup_seconds,"
           "tabi_preconditioner_graph_setup_seconds,"
           "tabi_preconditioner_graph_cache_reused,"
           "tabi_preconditioner_near_assembly_setup_seconds,"
           "tabi_preconditioner_ilu_factor_setup_seconds,"
           "tabi_preconditioner_apply_seconds,"
           "tabi_preconditioner_apply_calls,"
           "tabi_preconditioner_sparse_nnz_blocks,"
           "tabi_preconditioner_sparse_avg_degree,"
           "tabi_preconditioner_sparse_max_degree,"
           "tabi_preconditioner_coarse_aggregate_cap,"
           "tabi_preconditioner_coarse_aggregates,"
           "tabi_preconditioner_coarse_dimension,"
           "tabi_preconditioner_coarse_base,"
           "tabi_preconditioner_coarse_matrix_bytes,"
           "tabi_preconditioner_coarse_pivot_ratio,"
           "tabi_preconditioner_coarse_status,"
           "tabi_preconditioner_coarse_setup_matvec_calls,"
           "tabi_preconditioner_coarse_setup_seconds,"
           "tabi_preconditioner_coarse_moments_setup_seconds,"
           "tabi_preconditioner_coarse_far_setup_seconds,"
           "tabi_preconditioner_coarse_audit_setup_seconds,"
           "tabi_preconditioner_coarse_apply_seconds,"
           "tabi_preconditioner_coarse_apply_calls,"
           "tabi_preconditioner_coarse_rp_identity_error_max,"
           "tabi_preconditioner_coarse_verify_rel_error_max,"
           "tabi_surface_generator,tabi_discretization,"
           "tabi_fallback_used,"
           "tabi_rhs_scale,tabi_dielectric_contrast,tabi_operator_contrast,"
           "tabi_unknown_count,tabi_bie_system,tabi_lhs_diag_phi,"
           "tabi_lhs_diag_phin,tabi_energy_equation,"
           "tabi_kernel_convention,tabi_source_norm_s1,"
           "tabi_source_norm_s2,tabi_phi_norm,tabi_phin_norm,"
           "tabi_kappa_inv_A,tabi_coulomb_factor_kcal_mol,"
           "tabi_boundary_energy_kJmol,tabi_reference_energy_kJmol,"
           "tabi_selected_energy_scale,"
           "setup_seconds,surface_seconds,epsilon_seconds,boundary_seconds,"
           "solver_seconds,energy_seconds,map_io_seconds,rpb_source_seconds,"
           "focus_setup_seconds,probe_seconds,focus_parent_solve_seconds,"
           "focus_child_setup_seconds,focus_boundary_interp_seconds,"
           "focus_child_surface_seconds,focus_child_epsilon_seconds,"
           "focus_child_solver_seconds,probe_energy_seconds,"
           "audit_output_seconds,nwt_half_update_seconds,"
           "nwt_rms_reduce_seconds,nwt_operator_residual_seconds,"
           "nwt_source_update_seconds,nwt_progress_io_seconds,"
           "nwt_coeff_pack_seconds,nwt_homo_update_seconds,"
           "nwt_hetero_update_seconds,nwt_nonlin_term_seconds,"
           "nwt_reduce_seconds,nwt_homo_stencil_seconds,"
           "nwt_homo_salt_math_seconds,nwt_homo_write_seconds,"
           "nwt_thread_wait_seconds,nwt_block_overhead_seconds,"
           "linear_half_update_seconds,"
           "nwt_homo_cells,nwt_homo_linear_cells,nwt_homo_salt_cells,"
           "nwt_homo_updates,nwt_dispatch_calls,nwt_homo_ns_per_update,"
           "epsilon_center_density_seconds,epsilon_midpoint_density_seconds,"
           "epsilon_ion_exclusion_seconds,epsilon_finalize_seconds,"
           "total_seconds,iterations,blocks,"
           "status\n";
    timing_header_written_ = true;
  }
}

void AnalyzerImpl::append_timing_row(const FrameResult &result,
                                     const TimingBuckets &timing) {
  if (!pbe_timing_enabled())
    return;
  write_timing_header_if_needed();
  auto seconds = [](double ms) { return ms / 1000.0; };
  const std::size_t iterations =
      result.solvent_iterations + result.vacuum_iterations;
  const std::string mode =
      result.pbe_model_effective.empty()
          ? result.solver_mode
          : (result.pbe_model_effective + ":" + result.solver_mode);
  std::string scope = "single_grid";
  if (result.focus_enabled && result.focus_levels > 1)
    scope = result.probe_status == "off" ? "focus_full_workflow"
                                         : "focus_full_workflow_probe";
  else if (result.focus_enabled)
    scope = result.probe_status == "off" ? "focus_single_grid"
                                         : "focus_single_grid_probe";
  else if (result.pbe_biomodel == "rpbe")
    scope = "rpbe_full_workflow";
  else if (result.pbe_backend == "tabi")
    scope = "tabi_boundary_integral";
  timing_out_
      << result.trajectory_id << ',' << result.frame_index << ','
      << result.global_index << ',' << std::setprecision(10) << result.time_ps
      << ',' << csv_escape(current_source_label_) << ',' << csv_escape(mode)
      << ',' << csv_escape(result.pbe_backend) << ',' << result.atom_count
      << ',' << timing.focus_child_atom_count << ',' << configured_threads_
      << ',' << csv_escape(scope) << ','
      << seconds(result.tabi_surface_build_ms) << ','
      << result.tabi_surface_panels << ','
      << result.tabi_surface_point_count_requested << ','
      << result.tabi_panel_density_per_a2 << ','
      << csv_escape(result.tabi_panel_budget_status) << ','
      << result.tabi_surface_exposed_points << ','
      << result.tabi_surface_buried_points << ','
      << seconds(result.tabi_tree_build_ms) << ',' << result.tabi_tree_nodes
      << ',' << result.tabi_tree_leaf_count << ','
      << result.tabi_near_interactions << ',' << result.tabi_far_interactions
      << ',' << csv_escape(result.tabi_matvec_mode) << ','
      << csv_escape(result.tabi_status) << ','
      << csv_escape(result.tabi_convergence_reason) << ','
      << csv_escape(result.tabi_reaction_energy_source) << ','
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
      << seconds(result.tabi_full_probe_ms) << ','
      << seconds(result.tabi_rhs_ms) << ',' << seconds(result.tabi_gmres_ms)
      << ',' << seconds(result.tabi_matvec_ms_total) << ','
      << seconds(result.tabi_matvec_ms_avg) << ',' << result.tabi_matvec_calls
      << ',' << result.tabi_tree_vs_direct_matvec_rel << ','
      << result.tabi_matvec_audit_rows << ','
      << seconds(result.tabi_matvec_audit_ms) << ','
      << seconds(result.tabi_orthogonalization_ms) << ','
      << seconds(result.tabi_preconditioner_ms) << ','
      << seconds(result.tabi_surface_neighbor_ms) << ','
      << seconds(result.tabi_surface_mesh_ms) << ','
      << seconds(result.tabi_solid_angle_ms) << ','
      << seconds(result.tabi_tree_moment_ms) << ','
      << seconds(result.tabi_tree_traversal_ms) << ','
      << csv_escape(result.tabi_matvec_cost_profile) << ','
      << result.tabi_leaf_panels_per_second << ','
      << result.tabi_clean_leaf_panels_per_second << ','
      << result.tabi_tree_traversal_fraction << ','
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
      << result.tabi_singular_near_interactions << ','
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
      << csv_escape(result.tabi_leaf_pair_selection_profile) << ','
      << csv_escape(result.tabi_leaf_pair_probe_cache) << ','
      << csv_escape(result.tabi_leaf_pair_target_group) << ','
      << result.tabi_leaf_pair_max_lookahead << ','
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
      << csv_escape(result.tabi_leaf_pair_directness_probe) << ','
      << csv_escape(result.tabi_leaf_pair_directness_cache) << ','
      << csv_escape(result.tabi_leaf_pair_directness_verify) << ','
      << result.tabi_leaf_pair_directness_ancestor_probe_calls << ','
      << result.tabi_leaf_pair_directness_ancestor_probe_node_visits << ','
      << seconds(result.tabi_leaf_pair_directness_ancestor_probe_ms) << ','
      << result.tabi_leaf_pair_directness_pair_cache_hits << ','
      << result.tabi_leaf_pair_directness_pair_cache_misses << ','
      << result.tabi_leaf_pair_directness_all_direct_proofs << ','
      << result.tabi_leaf_pair_directness_mixed_fallbacks << ','
      << result.tabi_leaf_pair_directness_verify_checks << ','
      << result.tabi_leaf_pair_directness_verify_mismatches << ','
      << csv_escape(result.tabi_leaf_pair_scheduler) << ','
      << result.tabi_leaf_pair_worklist_max_active << ','
      << csv_escape(result.tabi_leaf_pair_worklist_verify) << ','
      << result.tabi_leaf_pair_worklist_tasks << ','
      << result.tabi_leaf_pair_worklist_groups << ','
      << result.tabi_leaf_pair_worklist_batched_groups << ','
      << result.tabi_leaf_pair_worklist_scalar_flush_tasks << ','
      << result.tabi_leaf_pair_worklist_forced_flushes << ','
      << result.tabi_leaf_pair_worklist_active_groups_max << ','
      << seconds(result.tabi_leaf_pair_worklist_ms) << ','
      << result.tabi_leaf_pair_worklist_verify_checks << ','
      << result.tabi_leaf_pair_worklist_verify_mismatches << ','
      << csv_escape(result.tabi_leaf_block_verify) << ','
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
      << csv_escape(result.tabi_tree_preset) << ','
      << csv_escape(result.tabi_adaptive_level) << ','
      << csv_escape(result.tabi_cluster_policy) << ','
      << result.tabi_target_batch_size << ','
      << result.tabi_leaf_pair_min_source_panels << ','
      << result.tabi_leaf_pair_min_targets << ','
      << csv_escape(result.tabi_leaf_near_policy) << ','
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
      << csv_escape(result.tabi_surface_geometry) << ','
      << result.tabi_mesh_vertices << ',' << result.tabi_mesh_triangles << ','
      << result.tabi_boundary_nodes << ',' << result.tabi_avg_valence << ','
      << result.tabi_nonmanifold_edges << ','
      << result.tabi_solid_angle_fallbacks << ',' << result.tabi_diag_min << ','
      << result.tabi_diag_max << ',' << result.tabi_taylor_order << ','
      << csv_escape(result.tabi_derivative_mode) << ','
      << seconds(result.tabi_interaction_build_ms) << ','
      << seconds(result.tabi_tree_eval_ms) << ','
      << csv_escape(result.tabi_formulation) << ','
      << csv_escape(result.tabi_energy_projection) << ','
      << seconds(result.tabi_energy_projection_ms) << ','
      << csv_escape(result.tabi_energy_projection_kernel) << ','
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
      << (result.tabi_fallback_used ? "true" : "false") << ','
      << result.tabi_rhs_scale << ',' << result.tabi_dielectric_contrast << ','
      << result.tabi_operator_contrast << ',' << result.tabi_unknown_count
      << ',' << csv_escape(result.tabi_bie_system) << ','
      << result.tabi_lhs_diag_phi << ',' << result.tabi_lhs_diag_phin << ','
      << csv_escape(result.tabi_energy_equation) << ','
      << csv_escape(result.tabi_kernel_convention) << ','
      << result.tabi_source_norm_s1 << ',' << result.tabi_source_norm_s2 << ','
      << result.tabi_phi_norm << ',' << result.tabi_phin_norm << ','
      << result.tabi_kappa_inv_a << ',' << result.tabi_coulomb_factor_kcal_mol
      << ',' << result.tabi_boundary_energy_kj_mol << ','
      << result.tabi_reference_energy_kj_mol << ','
      << result.tabi_selected_energy_scale << ',' << seconds(timing.setup_ms)
      << ',' << seconds(timing.surface_ms) << ',' << seconds(timing.epsilon_ms)
      << ',' << seconds(timing.boundary_ms) << ',' << seconds(timing.solver_ms)
      << ',' << seconds(timing.energy_ms) << ',' << seconds(timing.map_io_ms)
      << ',' << seconds(timing.rpb_source_ms) << ','
      << seconds(timing.focus_setup_ms) << ',' << seconds(timing.probe_ms)
      << ',' << seconds(timing.focus_parent_solve_ms) << ','
      << seconds(timing.focus_child_setup_ms) << ','
      << seconds(timing.focus_boundary_interp_ms) << ','
      << seconds(timing.focus_child_surface_ms) << ','
      << seconds(timing.focus_child_epsilon_ms) << ','
      << seconds(timing.focus_child_solver_ms) << ','
      << seconds(timing.probe_energy_ms) << ','
      << seconds(timing.audit_output_ms) << ','
      << seconds(timing.nwt_half_update_ms) << ','
      << seconds(timing.nwt_rms_reduce_ms) << ','
      << seconds(timing.nwt_operator_residual_ms) << ','
      << seconds(timing.nwt_source_update_ms) << ','
      << seconds(timing.nwt_progress_io_ms) << ','
      << seconds(timing.nwt_coeff_pack_ms) << ','
      << seconds(timing.nwt_homo_update_ms) << ','
      << seconds(timing.nwt_hetero_update_ms) << ','
      << seconds(timing.nwt_nonlin_term_ms) << ','
      << seconds(timing.nwt_reduce_ms) << ','
      << seconds(timing.nwt_homo_stencil_ms) << ','
      << seconds(timing.nwt_homo_salt_math_ms) << ','
      << seconds(timing.nwt_homo_write_ms) << ','
      << seconds(timing.nwt_thread_wait_ms) << ','
      << seconds(timing.nwt_block_overhead_ms) << ','
      << seconds(timing.linear_half_update_ms) << ',' << timing.nwt_homo_cells
      << ',' << timing.nwt_homo_linear_cells << ','
      << timing.nwt_homo_salt_cells << ',' << timing.nwt_homo_updates << ','
      << timing.nwt_dispatch_calls << ','
      << (timing.nwt_homo_updates > 0
              ? (timing.nwt_homo_update_ms * 1.0e6) /
                    static_cast<double>(timing.nwt_homo_updates)
              : 0.0)
      << ',' << seconds(timing.epsilon_center_density_ms) << ','
      << seconds(timing.epsilon_midpoint_density_ms) << ','
      << seconds(timing.epsilon_ion_exclusion_ms) << ','
      << seconds(timing.epsilon_finalize_ms) << ',' << seconds(timing.total_ms)
      << ',' << iterations << ',' << result.nl_blocks << ','
      << csv_escape(result.status) << '\n';
}


} // namespace TitanPBE
