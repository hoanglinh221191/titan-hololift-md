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

using analyzer_impl_detail::format_fixed;
using analyzer_impl_detail::kKcalToKJ;
using analyzer_impl_detail::surface_uses_face_refinement;

void AnalyzerImpl::finalize(std::fstream &log) {
  if (!enabled())
    return;
  if (energy_out_.is_open())
    energy_out_.flush();
  if (timing_out_.is_open())
    timing_out_.flush();
  if (tabi_profile_out_.is_open())
    tabi_profile_out_.flush();
  if (tabi_audit_profile_out_.is_open())
    tabi_audit_profile_out_.flush();
  if (memory_profile_out_.is_open())
    memory_profile_out_.flush();
  if (tabi_formulation_audit_out_.is_open())
    tabi_formulation_audit_out_.flush();
  if (tabi_near_block_audit_out_.is_open())
    tabi_near_block_audit_out_.flush();
  if (tabi_geometric_near_audit_out_.is_open())
    tabi_geometric_near_audit_out_.flush();
  if (tabi_surface_quality_out_.is_open())
    tabi_surface_quality_out_.flush();
  if (stability_audit_out_.is_open())
    stability_audit_out_.flush();
  if (stability_samples_out_.is_open())
    stability_samples_out_.flush();

  std::ofstream summary("pbe_summary.dat", std::ios::out | std::ios::trunc);
  if (summary.is_open()) {
    const double variance =
        frames_total_ > 1 ? energy_m2_ / static_cast<double>(frames_total_ - 1)
                          : 0.0;
    const double energy_kcal_mean = energy_mean_ / kKcalToKJ;
    const double energy_kcal_std = std::sqrt(variance) / kKcalToKJ;
    summary << "# TITAN PBE v1 summary\n";
    summary << "model "
            << (args_.model == "nonlinear" ? "nonlinear_pb" : "linear_pb")
            << "\n";
    summary << "pbe_backend " << args_.backend << "\n";
    summary << "pbe_model " << args_.model << "\n";
    summary << "pbe_model_effective "
            << (args_.model == "nonlinear" ? "nonlinear_or_linearized"
                                           : "linear")
            << "\n";
    summary << "pbe_biomodel " << args_.biomodel << "\n";
    summary << "dielectric_model " << args_.dielectric_model << "\n";
    summary << "surface " << args_.surface << "\n";
    summary << "probe_radius_A " << format_fixed(args_.probe_radius) << "\n";
    summary << "surface_effective " << args_.surface << "\n";
    summary << "face_refine "
            << (surface_uses_face_refinement(args_.surface) ? "true" : "false")
            << "\n";
    summary << "boundary_mode " << args_.boundary << "\n";
    summary << "charge_assignment " << args_.charge_assignment << "\n";
    summary << "radius_convention " << effective_radius_convention() << "\n";
    summary << "gaussian_exp " << format_fixed(args_.gaussian_exp, 6) << "\n";
    summary << "gaussian_sigma " << format_fixed(args_.gaussian_sigma, 6)
            << "\n";
    summary << "density_cutoff " << format_fixed(args_.density_cutoff, 6)
            << "\n";
    summary << "surface_cutoff "
            << (args_.surface_cutoff == "auto"
                    ? std::string("auto")
                    : format_fixed(args_.surface_cutoff_value, 6))
            << "\n";
    summary << "gap_diel " << format_fixed(args_.gap_diel, 6) << "\n";
    summary << "ion_access " << args_.ion_access << "\n";
    summary << "ion_radius_A " << format_fixed(args_.ion_radius) << "\n";
    summary << "precision cpu_double\n";
    summary << "scale_points_per_A " << format_fixed(args_.scale) << "\n";
    summary << "margin_A " << format_fixed(args_.margin) << "\n";
    summary << "gridbox " << args_.gridbox << "\n";
    summary << "int_diel " << format_fixed(args_.int_diel) << "\n";
    summary << "ext_diel " << format_fixed(args_.ext_diel) << "\n";
    summary << "salt_M " << format_fixed(args_.salt) << "\n";
    summary << "temperature_K " << format_fixed(args_.temp) << "\n";
    summary << "tabi_sum " << args_.tabi_sum << "\n";
    summary << "tabi_tree_preset " << args_.tabi_tree_preset << "\n";
    summary << "tabi_surface_geometry " << args_.tabi_geometry << "\n";
    summary << "tabi_sphere_mesh " << args_.tabi_sphere_mesh << "\n";
    summary << "tabi_surface_output " << args_.tabi_surface_output << "\n";
    summary << "tabi_surface_only "
            << (args_.tabi_surface_only ? "true" : "false") << "\n";
    summary << "tabi_ses_max_grid_vertices " << args_.tabi_ses_max_grid_vertices
            << "\n";
    summary << "tabi_surface_solution_csv "
            << ((args_.backend == "tabi" &&
                 (args_.tabi_surface_output == "csv" ||
                  args_.tabi_surface_output == "all"))
                    ? "pbe_tabi_surface_solution.csv"
                    : "off")
            << "\n";
    summary << "tabi_surface_solution_vtk "
            << ((args_.backend == "tabi" &&
                 (args_.tabi_surface_output == "vtk" ||
                  args_.tabi_surface_output == "all"))
                    ? "pbe_tabi_surface_solution.vtk"
                    : "off")
            << "\n";
    summary << "tabi_diagonal " << args_.tabi_diagonal << "\n";
    summary << "tabi_taylor_order " << args_.tabi_taylor_order << "\n";
    summary << "tabi_formulation " << args_.tabi_formulation << "\n";
    summary << "tabi_discretization " << args_.tabi_discretization << "\n";
    summary << "tabi_energy_projection " << args_.tabi_energy_projection
            << "\n";
    summary << "tabi_near_integration " << args_.tabi_near_integration << "\n";
    summary << "tabi_quad_buffer " << args_.tabi_quad_buffer << "\n";
    summary << "tabi_interaction_mode " << args_.tabi_interaction_mode << "\n";
    summary << "tabi_precond_block_near " << args_.tabi_precond_block_near
            << "\n";
    summary << "tabi_node_bounds " << args_.tabi_node_bounds << "\n";
    summary << "tabi_tree_builder " << args_.tabi_tree_builder << "\n";
    summary << "tabi_expansion_center " << args_.tabi_expansion_center << "\n";
    summary << "tabi_fragmentation_safety " << args_.tabi_fragmentation_safety
            << "\n";
    summary << "tabi_mac_safety " << args_.tabi_mac_safety << "\n";
    summary << "tabi_mac_gap_factor "
            << format_fixed(args_.tabi_mac_gap_factor, 6) << "\n";
    summary << "tabi_moment_audit " << args_.tabi_moment_audit << "\n";
    summary << "tabi_moment_construction " << args_.tabi_moment_construction
            << "\n";
    summary << "tabi_moment_sum " << args_.tabi_moment_sum << "\n";
    summary << "tabi_convergence_audit " << args_.tabi_convergence_audit
            << "\n";
    summary << "tabi_convergence_safety " << args_.tabi_convergence_safety
            << "\n";
    summary << "tabi_convergence_rho_cap "
            << format_fixed(args_.tabi_convergence_rho_cap, 6) << "\n";
    summary << "tabi_mac_aspect_cap "
            << format_fixed(args_.tabi_mac_aspect_cap, 6) << "\n";
    summary << "tabi_convergence_estimator " << args_.tabi_convergence_estimator
            << "\n";
    summary << "tabi_convergence_tail_threshold "
            << format_fixed(args_.tabi_convergence_tail_threshold, 6) << "\n";
    summary << "tabi_convergence_estimator_error_cap "
            << format_fixed(args_.tabi_convergence_estimator_error_cap, 8)
            << "\n";
    summary << "tabi_convergence_unstable_rho_cap "
            << format_fixed(args_.tabi_convergence_unstable_rho_cap, 6) << "\n";
    summary << "tabi_convergence_calibration "
            << args_.tabi_convergence_calibration << "\n";
    summary << "tabi_expansion_audit " << args_.tabi_expansion_audit << "\n";
    summary << "tabi_expansion_audit_sample_count "
            << args_.tabi_expansion_audit_sample_count << "\n";
    summary << "tabi_expansion_audit_max_panels "
            << args_.tabi_expansion_audit_max_panels << "\n";
    summary << "tabi_diagnostics " << args_.tabi_diagnostics << "\n";
    summary << "tabi_audit_profile_csv "
            << (args_.tabi_diagnostics == "off" ? "off"
                                                : "pbe_tabi_audit_profile.csv")
            << "\n";
    summary << "tabi_matvec_audit " << args_.tabi_matvec_audit << "\n";
    summary << "tabi_matvec_audit_sample_count "
            << args_.tabi_matvec_audit_sample_count << "\n";
    summary << "tabi_node_patch_operator_audit "
            << args_.tabi_node_patch_operator_audit << "\n";
    summary << "tabi_fmm_cartesian_error_audit "
            << args_.tabi_fmm_cartesian_error_audit << "\n";
    summary << "tabi_true_residual_probe " << args_.tabi_true_residual_probe
            << "\n";
    summary << "tabi_true_residual_probe_scope "
            << args_.tabi_true_residual_probe_scope << "\n";
    summary << "tabi_true_residual_probe_interval "
            << args_.tabi_true_residual_probe_interval << "\n";
    summary << "tabi_true_residual_sample_count "
            << args_.tabi_true_residual_sample_count << "\n";
    summary << "tabi_formulation_audit_csv "
            << "pbe_tabi_formulation_audit.csv\n";
    summary << "tabi_profile_csv "
            << (args_.timing == "detail" ? "pbe_tabi_profile.csv" : "off")
            << "\n";
    summary << "tabi_near_block_audit_csv "
            << (args_.timing == "detail" ? "pbe_tabi_near_block_audit.csv"
                                         : "off")
            << "\n";
    summary << "tabi_geometric_near_audit_csv "
            << (args_.timing == "detail" ? "pbe_tabi_geometric_near_audit.csv"
                                         : "off")
            << "\n";
    summary << "tabi_far_geometric_audit_csv "
            << (args_.timing == "detail" ? "pbe_tabi_far_geometric_audit.csv"
                                         : "off")
            << "\n";
    summary << "tabi_moment_audit_csv "
            << (args_.timing == "detail" ? "pbe_tabi_moment_audit.csv" : "off")
            << "\n";
    summary << "tabi_surface_quality_csv "
            << (args_.timing == "detail" ? "pbe_tabi_surface_quality.csv"
                                         : "off")
            << "\n";
    summary << "memory_profile_csv "
            << (pbe_timing_enabled() ? "pbe_memory_profile.csv" : "off")
            << "\n";
    summary << "tabi_surface_panels_max " << max_tabi_surface_panels_seen_
            << "\n";
    summary << "tabi_mesh_vertices_max " << max_tabi_mesh_vertices_seen_
            << "\n";
    summary << "tabi_mesh_triangles_max " << max_tabi_mesh_triangles_seen_
            << "\n";
    summary << "tabi_boundary_nodes_max " << max_tabi_boundary_nodes_seen_
            << "\n";
    summary << "tabi_diag_min " << format_fixed(min_tabi_diag_seen_, 6) << "\n";
    summary << "tabi_diag_max " << format_fixed(max_tabi_diag_seen_, 6) << "\n";
    summary << "tabi_nonmanifold_edges_max " << max_tabi_nonmanifold_edges_seen_
            << "\n";
    summary << "tabi_solid_angle_fallbacks_max "
            << max_tabi_solid_angle_fallbacks_seen_ << "\n";
    summary << "tabi_tree_nodes_max " << max_tabi_tree_nodes_seen_ << "\n";
    summary << "tabi_tree_leaf_count_max " << max_tabi_tree_leaf_count_seen_
            << "\n";
    summary << "tabi_near_interactions_max " << max_tabi_near_interactions_seen_
            << "\n";
    summary << "tabi_far_interactions_max " << max_tabi_far_interactions_seen_
            << "\n";
    summary << "kernel_requested " << args_.kernel << "\n";
    summary << "kernel_effective scalar_threaded\n";
    summary << "focus_mode " << args_.focus << "\n";
    summary << "multigrid_requested " << args_.multigrid << "\n";
    summary << "multigrid_effective " << effective_solver_mode() << "\n";
    summary << "focus_levels_effective " << effective_focus_levels() << "\n";
    summary << "focus_margin_extra_A " << format_fixed(args_.focus_margin)
            << "\n";
    summary << "mg_levels_config " << args_.mg_levels << "\n";
    summary << "mg_pre " << args_.mg_pre << "\n";
    summary << "mg_post " << args_.mg_post << "\n";
    summary << "mg_coarse_iter " << args_.mg_coarse_iter << "\n";
    summary << "nl_max_iter " << args_.nl_max_iter << "\n";
    summary << "nl_inner_iter " << args_.nl_inner_iter << "\n";
    summary << "nl_stagnation " << args_.nl_stagnation << "\n";
    summary << "nl_solver " << args_.nl_solver << "\n";
    summary << "nl_inner " << args_.nl_inner << "\n";
    summary << "nl_multigrid " << args_.nl_multigrid << "\n";
    summary << "nl_block " << args_.nl_block << "\n";
    summary << "nl_dphi_tol " << format_fixed(args_.nl_dphi, 10) << "\n";
    summary << "nl_relax " << format_fixed(effective_nl_relax(), 6) << "\n";
    summary << "nl_clip " << format_fixed(args_.nl_clip, 6) << "\n";
    summary << "nl_continuation " << args_.nl_continuation << "\n";
    summary << "nl_line_search " << args_.nl_line_search << "\n";
    summary << "nl_residual_tol " << format_fixed(args_.nl_residual, 10)
            << "\n";
    if (!args_.test_pqr.empty()) {
      summary << "test_pqr_adapter " << args_.test_pqr << "\n";
      summary << "test_grid_size " << args_.test_grid_size << "\n";
      summary << "test_center "
              << (args_.test_center_user
                      ? format_fixed(args_.test_center_x, 6) + "," +
                            format_fixed(args_.test_center_y, 6) + "," +
                            format_fixed(args_.test_center_z, 6)
                      : std::string("auto"))
              << "\n";
    }
    summary << "frames_total " << frames_total_ << "\n";
    summary << "frames_converged " << frames_converged_ << "\n";
    summary << "frames_failed " << frames_failed_ << "\n";
    summary << "energy_kJmol_mean " << format_fixed(energy_mean_) << "\n";
    summary << "energy_kJmol_std " << format_fixed(std::sqrt(variance)) << "\n";
    summary << "energy_kJmol_min " << format_fixed(energy_min_) << "\n";
    summary << "energy_kJmol_max " << format_fixed(energy_max_) << "\n";
    summary << "energy_kcalmol_mean " << format_fixed(energy_kcal_mean) << "\n";
    summary << "energy_kcalmol_std " << format_fixed(energy_kcal_std) << "\n";
    summary << "energy_kcalmol_min " << format_fixed(energy_min_ / kKcalToKJ)
            << "\n";
    summary << "energy_kcalmol_max " << format_fixed(energy_max_ / kKcalToKJ)
            << "\n";
    summary << "deltaG_solv_enabled " << (args_.dgsolv ? "true" : "false")
            << "\n";
    summary << "deltaG_solv_frames " << dgsolv_frames_ << "\n";
    if (dgsolv_frames_ > 0) {
      const double dg_variance =
          dgsolv_frames_ > 1
              ? dgsolv_m2_ / static_cast<double>(dgsolv_frames_ - 1)
              : 0.0;
      summary << "deltaG_solv_kJmol_mean " << format_fixed(dgsolv_mean_)
              << "\n";
      summary << "deltaG_solv_kJmol_std "
              << format_fixed(std::sqrt(dg_variance)) << "\n";
      summary << "deltaG_solv_kJmol_min " << format_fixed(dgsolv_min_) << "\n";
      summary << "deltaG_solv_kJmol_max " << format_fixed(dgsolv_max_) << "\n";
      summary << "deltaG_solv_kcalmol_mean "
              << format_fixed(dgsolv_mean_ / kKcalToKJ) << "\n";
      summary << "deltaG_solv_kcalmol_std "
              << format_fixed(std::sqrt(dg_variance) / kKcalToKJ) << "\n";
      summary << "deltaG_solv_kcalmol_min "
              << format_fixed(dgsolv_min_ / kKcalToKJ) << "\n";
      summary << "deltaG_solv_kcalmol_max "
              << format_fixed(dgsolv_max_ / kKcalToKJ) << "\n";
    }
    summary << "iterations_solvent_mean " << format_fixed(iter_mean_, 3)
            << "\n";
    summary << "max_grid_points " << max_grid_points_seen_ << "\n";
    summary << "max_memory_bytes " << max_memory_bytes_seen_ << "\n";
    summary << "max_solute_points " << max_solute_points_seen_ << "\n";
    summary << "max_solvent_points " << max_solvent_points_seen_ << "\n";
    summary << "max_ion_accessible_points " << max_ion_accessible_points_seen_
            << "\n";
    summary << "max_stern_excluded_points " << max_stern_excluded_points_seen_
            << "\n";
    summary << "max_focus_levels " << max_focus_levels_seen_ << "\n";
    summary << "max_mg_levels " << max_mg_levels_seen_ << "\n";
    summary << "mg_cycles_mean " << format_fixed(mg_cycle_mean_, 3) << "\n";
    summary << "max_mg_cycles " << max_mg_cycles_seen_ << "\n";
    summary << "max_boundary_rms " << format_fixed(max_boundary_rms_seen_, 10)
            << "\n";
    summary << "max_boundary_max " << format_fixed(max_boundary_max_seen_, 10)
            << "\n";
    summary << "max_linear_residual "
            << format_fixed(max_linear_residual_seen_, 10) << "\n";
    summary << "nonlinear_frames " << nonlinear_frames_ << "\n";
    summary << "nonlinear_converged " << nonlinear_converged_ << "\n";
    summary << "nonlinear_failed " << nonlinear_failed_ << "\n";
    summary << "max_nl_blocks " << max_nl_blocks_seen_ << "\n";
    summary << "max_nl_iterations " << max_nl_iterations_seen_ << "\n";
    summary << "max_nl_dphi " << format_fixed(max_nl_dphi_seen_, 10) << "\n";
    summary << "max_nl_residual " << format_fixed(max_nl_residual_seen_, 10)
            << "\n";
    summary << "max_nl_clipped_points " << max_nl_clipped_points_seen_ << "\n";
    summary << "max_newton_inner_iterations "
            << max_newton_inner_iterations_seen_ << "\n";
    summary << "max_newton_step_norm "
            << format_fixed(max_newton_step_norm_seen_, 10) << "\n";
    summary << "max_newton_residual_before "
            << format_fixed(max_newton_residual_before_seen_, 10) << "\n";
    summary << "max_newton_residual_after "
            << format_fixed(max_newton_residual_after_seen_, 10) << "\n";
    summary << "max_newton_correction_residual "
            << format_fixed(max_newton_correction_residual_seen_, 10) << "\n";
    summary << "max_newton_correction_max_residual "
            << format_fixed(max_newton_correction_max_residual_seen_, 10)
            << "\n";
    summary << "max_newton_rejected_steps " << max_newton_rejected_steps_seen_
            << "\n";
    summary << "max_newton_inner_levels " << max_newton_inner_levels_seen_
            << "\n";
    summary << "max_newton_inner_cycles " << max_newton_inner_cycles_seen_
            << "\n";
    summary << "max_newton_inner_residual_before "
            << format_fixed(max_newton_inner_residual_before_seen_, 10) << "\n";
    summary << "max_newton_inner_residual_after "
            << format_fixed(max_newton_inner_residual_after_seen_, 10) << "\n";
    summary << "energy_csv pbe_energy.csv\n";
    if (!args_.test_pqr.empty() && args_.parity_dump != "off") {
      summary << "parity_dump " << args_.parity_dump << "\n";
      summary << "test_energy_convention " << args_.test_energy_convention
              << "\n";
      summary << "test_midpoint_epsilon " << args_.test_midpoint_epsilon
              << "\n";
      summary << "test_boundary_convention " << args_.test_boundary_convention
              << "\n";
      summary << "test_field_units " << args_.test_field_units << "\n";
      summary << "test_operator_convention " << args_.test_operator_convention
              << "\n";
      summary << "test_rhs_map " << args_.test_rhs_map << "\n";
      summary << "test_screen_map " << args_.test_screen_map << "\n";
      summary << "test_denom_map " << args_.test_denom_map << "\n";
      if (args_.test_midpoint_diff != "off") {
        summary << "test_midpoint_diff " << args_.test_midpoint_diff << "\n";
        summary << "midpoint_face_diff_csv pbe_midpoint_face_diff.csv\n";
      }
      if (args_.test_charge_grid_diff != "off") {
        summary << "test_charge_grid_diff " << args_.test_charge_grid_diff
                << "\n";
        summary << "charge_grid_diff_csv pbe_charge_grid_diff.csv\n";
        summary << "charge_grid_diff_summary_csv "
                   "pbe_charge_grid_diff_summary.csv\n";
      }
      summary << "parity_stage_csv pbe_parity_stage.csv\n";
      if (args_.parity_dump == "points" || args_.parity_dump == "all")
        summary << "parity_points_csv pbe_parity_points.csv\n";
      if (args_.parity_dump == "all" &&
          (args_.test_energy_convention == "delphi-iscm" ||
           args_.test_energy_convention == "delphi-corrected"))
        summary << "iscm_surface_charge_samples_csv "
                   "pbe_iscm_surface_charge_samples.csv\n";
    }
    if (!args_.test_pqr.empty() && args_.parity_map_dump != "off") {
      summary << "parity_map_dump " << args_.parity_map_dump << "\n";
      summary << "test_charge_grid " << args_.test_charge_grid << "\n";
      summary << "test_charge_grid_diff " << args_.test_charge_grid_diff
              << "\n";
      summary << "test_surface_map " << args_.test_surface_map << "\n";
      summary << "test_boundary_map " << args_.test_boundary_map << "\n";
      summary << "test_boundary_convention " << args_.test_boundary_convention
              << "\n";
      summary << "test_field_units " << args_.test_field_units << "\n";
      summary << "test_operator_convention " << args_.test_operator_convention
              << "\n";
      summary << "test_rhs_map " << args_.test_rhs_map << "\n";
      summary << "test_screen_map " << args_.test_screen_map << "\n";
      summary << "test_denom_map " << args_.test_denom_map << "\n";
      summary << "field_samples_csv pbe_field_samples.csv\n";
      summary << "boundary_samples_csv pbe_boundary_samples.csv\n";
      summary << "operator_stage_csv pbe_operator_stage.csv\n";
      if (args_.parity_dump == "all")
        summary << "sor_trace_csv pbe_sor_trace.csv\n";
      summary << "charge_grid_audit_csv pbe_charge_grid_audit.csv\n";
    }
  } else {
    TitanUI::print_warning("PBE: failed to write pbe_summary.dat.");
  }

  write_pbsa_summary(log);

  if (log.is_open()) {
    TitanUI::log_section(log, "PBE OUTPUT AUDIT");
    TitanUI::log_kv(log, "Frames", std::to_string(frames_total_));
    TitanUI::log_kv(log, "Converged", std::to_string(frames_converged_));
    TitanUI::log_kv(log, "Failed/Not Converged",
                    std::to_string(frames_failed_));
    TitanUI::log_kv(log, "PBE Backend", args_.backend);
    if (args_.backend == "tabi") {
      TitanUI::log_kv(log, "TABI Geometry", args_.tabi_geometry);
      TitanUI::log_kv(log, "TABI Diagonal", args_.tabi_diagonal);
      TitanUI::log_kv(log, "TABI Sum", args_.tabi_sum);
      TitanUI::log_kv(log, "TABI Tree Preset", args_.tabi_tree_preset);
      TitanUI::log_kv(log, "TABI Formulation", args_.tabi_formulation);
      TitanUI::log_kv(log, "TABI Discretization",
                      args_.tabi_discretization);
      TitanUI::log_kv(log, "TABI Energy Projection",
                      args_.tabi_energy_projection);
      TitanUI::log_kv(log, "TABI Near Integration",
                      args_.tabi_near_integration);
      TitanUI::log_kv(log, "TABI Quad Buffer", args_.tabi_quad_buffer);
      TitanUI::log_kv(log, "TABI Interaction Mode",
                      args_.tabi_interaction_mode);
      TitanUI::log_kv(log, "TABI Preconditioner Block Near",
                      args_.tabi_precond_block_near);
      TitanUI::log_kv(log, "TABI Node Bounds", args_.tabi_node_bounds);
      TitanUI::log_kv(log, "TABI Tree Builder", args_.tabi_tree_builder);
      TitanUI::log_kv(log, "TABI Expansion Center",
                      args_.tabi_expansion_center);
      TitanUI::log_kv(log, "TABI Fragmentation Safety",
                      args_.tabi_fragmentation_safety);
      TitanUI::log_kv(log, "TABI MAC Safety", args_.tabi_mac_safety);
      TitanUI::log_kv(log, "TABI MAC Gap Factor",
                      format_fixed(args_.tabi_mac_gap_factor, 6));
      TitanUI::log_kv(log, "TABI Moment Audit", args_.tabi_moment_audit);
      TitanUI::log_kv(log, "TABI Moment Construction",
                      args_.tabi_moment_construction);
      TitanUI::log_kv(log, "TABI Moment Sum", args_.tabi_moment_sum);
      TitanUI::log_kv(log, "TABI Convergence Audit",
                      args_.tabi_convergence_audit);
      TitanUI::log_kv(log, "TABI Convergence Safety",
                      args_.tabi_convergence_safety);
      TitanUI::log_kv(log, "TABI Convergence Rho Cap",
                      format_fixed(args_.tabi_convergence_rho_cap, 6));
      TitanUI::log_kv(log, "TABI MAC Aspect Cap",
                      format_fixed(args_.tabi_mac_aspect_cap, 6));
      TitanUI::log_kv(log, "TABI Convergence Estimator",
                      args_.tabi_convergence_estimator);
      TitanUI::log_kv(log, "TABI Convergence Tail Threshold",
                      format_fixed(args_.tabi_convergence_tail_threshold, 6));
      TitanUI::log_kv(
          log, "TABI Convergence Estimator Error Cap",
          format_fixed(args_.tabi_convergence_estimator_error_cap, 8));
      TitanUI::log_kv(log, "TABI Convergence Unstable Rho Cap",
                      format_fixed(args_.tabi_convergence_unstable_rho_cap, 6));
      TitanUI::log_kv(log, "TABI Convergence Calibration",
                      args_.tabi_convergence_calibration);
      TitanUI::log_kv(log, "TABI Expansion Audit", args_.tabi_expansion_audit);
      TitanUI::log_kv(log, "TABI Diagnostics", args_.tabi_diagnostics);
      TitanUI::log_kv(log, "TABI Audit Profile CSV",
                      args_.tabi_diagnostics == "off"
                          ? "off"
                          : "pbe_tabi_audit_profile.csv");
      TitanUI::log_kv(log, "TABI True Residual Probe",
                      args_.tabi_true_residual_probe);
      TitanUI::log_kv(log, "TABI True Residual Probe Scope",
                      args_.tabi_true_residual_probe_scope);
      TitanUI::log_kv(log, "TABI True Residual Probe Interval",
                      std::to_string(args_.tabi_true_residual_probe_interval));
      TitanUI::log_kv(log, "TABI True Residual Sample Count",
                      std::to_string(args_.tabi_true_residual_sample_count));
      TitanUI::log_kv(log, "TABI Formulation Audit",
                      "pbe_tabi_formulation_audit.csv");
      TitanUI::log_kv(log, "TABI Profile CSV",
                      args_.timing == "detail" ? "pbe_tabi_profile.csv"
                                               : "off");
      TitanUI::log_kv(log, "TABI Near Block Audit",
                      args_.timing == "detail" ? "pbe_tabi_near_block_audit.csv"
                                               : "off");
      TitanUI::log_kv(log, "TABI Geometric Near Audit",
                      args_.timing == "detail"
                          ? "pbe_tabi_geometric_near_audit.csv"
                          : "off");
      TitanUI::log_kv(log, "TABI Far Geometric Audit",
                      args_.timing == "detail"
                          ? "pbe_tabi_far_geometric_audit.csv"
                          : "off");
      TitanUI::log_kv(log, "TABI Moment Audit",
                      args_.timing == "detail" ? "pbe_tabi_moment_audit.csv"
                                               : "off");
      TitanUI::log_kv(log, "TABI Surface Quality",
                      args_.timing == "detail" ? "pbe_tabi_surface_quality.csv"
                                               : "off");
    }
    TitanUI::log_kv(log, "PBE Memory Profile",
                    pbe_timing_enabled() ? "pbe_memory_profile.csv" : "off");
    TitanUI::log_kv(log, "PBE Model", args_.model);
    TitanUI::log_kv(log, "Bio Model", args_.biomodel);
    TitanUI::log_kv(log, "Dielectric Model", args_.dielectric_model);
    TitanUI::log_kv(log, "Radius Convention", effective_radius_convention());
    TitanUI::log_kv(log, "Boundary", args_.boundary);
    TitanUI::log_kv(log, "Charge Assignment", args_.charge_assignment);
    TitanUI::log_kv(log, "Nonlinear Solver", args_.nl_solver);
    TitanUI::log_kv(log, "Nonlinear Inner", args_.nl_inner);
    TitanUI::log_kv(log, "Nonlinear MG", args_.nl_multigrid);
    TitanUI::log_kv(log, "Newton Inner Iter Cap",
                    std::to_string(args_.nl_inner_iter));
    TitanUI::log_kv(log, "Newton Stagnation Window",
                    std::to_string(args_.nl_stagnation));
    TitanUI::log_kv(log, "Nonlinear Frames", std::to_string(nonlinear_frames_));
    TitanUI::log_kv(log, "Nonlinear Converged",
                    std::to_string(nonlinear_converged_));
    TitanUI::log_kv(log, "Nonlinear Failed", std::to_string(nonlinear_failed_));
    TitanUI::log_kv(log, "Max Nonlinear dPhi",
                    format_fixed(max_nl_dphi_seen_, 10));
    TitanUI::log_kv(log, "Max Nonlinear Residual",
                    format_fixed(max_nl_residual_seen_, 10));
    TitanUI::log_kv(log, "Max Newton Inner Iterations",
                    std::to_string(max_newton_inner_iterations_seen_));
    TitanUI::log_kv(log, "Max Newton Residual Before/After",
                    format_fixed(max_newton_residual_before_seen_, 10) + " / " +
                        format_fixed(max_newton_residual_after_seen_, 10));
    TitanUI::log_kv(log, "Max Newton Correction Residual",
                    format_fixed(max_newton_correction_residual_seen_, 10));
    TitanUI::log_kv(log, "Max Newton Correction Max Residual",
                    format_fixed(max_newton_correction_max_residual_seen_, 10));
    TitanUI::log_kv(log, "Max Newton Rejected Steps",
                    std::to_string(max_newton_rejected_steps_seen_));
    TitanUI::log_kv(log, "Max Newton Inner Levels/Cycles",
                    std::to_string(max_newton_inner_levels_seen_) + " / " +
                        std::to_string(max_newton_inner_cycles_seen_));
    TitanUI::log_kv(
        log, "Max Newton Inner Residual Before/After",
        format_fixed(max_newton_inner_residual_before_seen_, 10) + " / " +
            format_fixed(max_newton_inner_residual_after_seen_, 10));
    TitanUI::log_kv(log, "Max Linear Residual",
                    format_fixed(max_linear_residual_seen_, 10));
    TitanUI::log_kv(log, "Max Boundary Potential",
                    format_fixed(max_boundary_max_seen_, 10));
    TitanUI::log_kv(log, "Max Nonlinear Clipped Points",
                    std::to_string(max_nl_clipped_points_seen_));
    TitanUI::log_kv(log, "Surface", args_.surface);
    if (uses_gaussian_model()) {
      TitanUI::log_kv(log, "Gaussian Exp", format_fixed(args_.gaussian_exp));
      TitanUI::log_kv(log, "Gaussian Sigma",
                      format_fixed(args_.gaussian_sigma));
      TitanUI::log_kv(log, "Density Cutoff",
                      format_fixed(args_.density_cutoff));
      TitanUI::log_kv(log, "Gap Dielectric", format_fixed(args_.gap_diel));
    }
    TitanUI::log_kv(log, "Probe Radius",
                    format_fixed(args_.probe_radius) + " A");
    TitanUI::log_kv(log, "Face Refinement",
                    surface_uses_face_refinement(args_.surface) ? "enabled"
                                                                : "off");
    TitanUI::log_kv(log, "Ion Access", args_.ion_access);
    TitanUI::log_kv(log, "Ion Radius", format_fixed(args_.ion_radius) + " A");
    TitanUI::log_kv(log, "Focus Mode", args_.focus);
    TitanUI::log_kv(log, "Multigrid Requested", args_.multigrid);
    TitanUI::log_kv(log, "Multigrid Effective", effective_solver_mode());
    TitanUI::log_kv(log, "Focus Levels",
                    std::to_string(effective_focus_levels()));
    TitanUI::log_kv(log, "MG Cycles Mean", format_fixed(mg_cycle_mean_, 3));
    TitanUI::log_kv(log, "DeltaG Solv", args_.dgsolv ? "enabled" : "off");
    if (dgsolv_frames_ > 0) {
      TitanUI::log_kv(log, "DeltaG Solv Frames",
                      std::to_string(dgsolv_frames_));
      TitanUI::log_kv(log, "DeltaG Solv Mean",
                      format_fixed(dgsolv_mean_) + " kJ/mol / " +
                          format_fixed(dgsolv_mean_ / kKcalToKJ) + " kcal/mol");
    }
    if (!args_.test_pqr.empty()) {
      TitanUI::log_kv(log, "Test PQR Adapter", args_.test_pqr);
      TitanUI::log_kv(log, "Parity Dump", args_.parity_dump);
      TitanUI::log_kv(log, "Test Energy Convention",
                      args_.test_energy_convention);
      if (args_.parity_dump != "off") {
        TitanUI::log_kv(log, "Parity Stage CSV", "pbe_parity_stage.csv");
        if (args_.parity_dump == "points" || args_.parity_dump == "all")
          TitanUI::log_kv(log, "Parity Points CSV", "pbe_parity_points.csv");
        if (args_.parity_dump == "all" &&
            (args_.test_energy_convention == "delphi-iscm" ||
             args_.test_energy_convention == "delphi-corrected"))
          TitanUI::log_kv(log, "ISCM Surface Samples CSV",
                          "pbe_iscm_surface_charge_samples.csv");
      }
      TitanUI::log_kv(log, "Parity Map Dump", args_.parity_map_dump);
      TitanUI::log_kv(log, "Charge Grid Replay", args_.test_charge_grid);
      TitanUI::log_kv(log, "Charge Grid Diff", args_.test_charge_grid_diff);
      TitanUI::log_kv(log, "Surface Map Replay", args_.test_surface_map);
      TitanUI::log_kv(log, "Midpoint Epsilon Replay",
                      args_.test_midpoint_epsilon);
      TitanUI::log_kv(log, "Midpoint Face Diff", args_.test_midpoint_diff);
      TitanUI::log_kv(log, "Boundary Map Replay", args_.test_boundary_map);
      TitanUI::log_kv(log, "Boundary Convention",
                      args_.test_boundary_convention);
      TitanUI::log_kv(log, "Field Units", args_.test_field_units);
      TitanUI::log_kv(log, "Operator Convention",
                      args_.test_operator_convention);
      TitanUI::log_kv(log, "RHS Replay", args_.test_rhs_map);
      TitanUI::log_kv(log, "Screen Replay", args_.test_screen_map);
      TitanUI::log_kv(log, "Denom Replay", args_.test_denom_map);
      if (args_.test_midpoint_diff != "off")
        TitanUI::log_kv(log, "Midpoint Face Diff CSV",
                        "pbe_midpoint_face_diff.csv");
      if (args_.test_charge_grid_diff != "off") {
        TitanUI::log_kv(log, "Charge Grid Diff CSV",
                        "pbe_charge_grid_diff.csv");
        TitanUI::log_kv(log, "Charge Grid Diff Summary CSV",
                        "pbe_charge_grid_diff_summary.csv");
      }
      if (args_.parity_map_dump != "off") {
        TitanUI::log_kv(log, "Field Samples CSV", "pbe_field_samples.csv");
        TitanUI::log_kv(log, "Boundary Samples CSV",
                        "pbe_boundary_samples.csv");
        TitanUI::log_kv(log, "Operator Stage CSV", "pbe_operator_stage.csv");
        TitanUI::log_kv(log, "Charge Grid Audit CSV",
                        "pbe_charge_grid_audit.csv");
      }
    }
    TitanUI::log_kv(log, "Energy CSV", "pbe_energy.csv");
    TitanUI::log_kv(log, "Summary", "pbe_summary.dat");
    TitanUI::log_kv(log, "Max Grid Points",
                    std::to_string(max_grid_points_seen_));
    TitanUI::log_kv(log, "Max Memory Bytes",
                    std::to_string(max_memory_bytes_seen_));
    TitanUI::log_kv(log, "Max Solute Points",
                    std::to_string(max_solute_points_seen_));
    TitanUI::log_kv(log, "Max Solvent Points",
                    std::to_string(max_solvent_points_seen_));
    TitanUI::log_kv(log, "Max Ion Accessible Points",
                    std::to_string(max_ion_accessible_points_seen_));
    TitanUI::log_kv(log, "Max Stern Excluded Points",
                    std::to_string(max_stern_excluded_points_seen_));
  }
}


} // namespace TitanPBE
