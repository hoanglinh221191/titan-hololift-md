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

using analyzer_impl_detail::csv_double;
using analyzer_impl_detail::csv_escape;
using analyzer_impl_detail::csv_header_map;
using analyzer_impl_detail::csv_index;
using analyzer_impl_detail::idx3;
using analyzer_impl_detail::kKBKJPerMolK;
using analyzer_impl_detail::kKcalToKJ;
using analyzer_impl_detail::split_csv_simple;
using analyzer_impl_detail::trim_copy;

namespace {

template <typename Shape>
bool is_boundary(std::size_t i, std::size_t j, std::size_t k,
                 const Shape &shape) noexcept {
  return i == 0 || j == 0 || k == 0 || i + 1 == shape.nx || j + 1 == shape.ny ||
         k + 1 == shape.nz;
}

std::string read_oracle_fixture_filter() {
  std::ifstream in("pbe_oracle_fixture.txt");
  if (!in.is_open())
    return {};
  std::string line;
  if (!std::getline(in, line))
    return {};
  return trim_copy(line);
}

bool oracle_fixture_matches(
    const std::vector<std::string> &fields,
    const std::unordered_map<std::string, std::size_t> &header,
    const std::string &fixture_filter) {
  if (fixture_filter.empty())
    return true;
  const auto it = header.find("fixture");
  if (it == header.end() || it->second >= fields.size())
    return true;
  return trim_copy(fields[it->second]) == fixture_filter;
}

} // namespace

void AnalyzerImpl::write_memory_profile_header_if_needed() {
  if (!pbe_timing_enabled())
    return;
  if (!memory_profile_out_.is_open()) {
    memory_profile_out_.open("pbe_memory_profile.csv",
                             std::ios::out | std::ios::trunc);
    if (!memory_profile_out_.is_open())
      throw std::runtime_error("PBE: failed to open pbe_memory_profile.csv.");
  }
  if (!memory_profile_header_written_) {
    memory_profile_out_
        << "trajectory_id,frame_index,global_index,fixture,backend,"
           "atom_count,grid_points,panels,unknown_count,formulation,"
           "matvec_mode,preconditioner_effective,surface_generator,"
           "discretization,estimated_bytes_total,estimated_bytes_surface,"
           "estimated_bytes_solver,estimated_bytes_operator,"
           "estimated_bytes_tree,rss_before_bytes,rss_after_bytes,"
           "rss_delta_bytes,rss_delta_signed_bytes,peak_working_set_bytes,"
           "total_seconds,"
           "solver_seconds,status\n";
    memory_profile_header_written_ = true;
  }
}

void AnalyzerImpl::append_memory_profile_row(const FrameResult &result,
                                             const TimingBuckets &timing) {
  if (!pbe_timing_enabled())
    return;
  write_memory_profile_header_if_needed();
  auto seconds = [](double ms) { return ms / 1000.0; };
  const bool is_tabi = result.pbe_backend == "tabi";
  const std::size_t panels = is_tabi ? result.tabi_surface_panels : 0;
  const std::size_t unknowns = is_tabi ? result.tabi_unknown_count : 0;
  std::size_t surface_bytes = result.memory_estimated_surface_bytes;
  std::size_t solver_bytes = result.memory_estimated_solver_bytes;
  std::size_t operator_bytes = result.memory_estimated_operator_bytes;
  std::size_t tree_bytes = result.memory_estimated_tree_bytes;
  if (surface_bytes == 0 || solver_bytes == 0 || operator_bytes == 0) {
    if (is_tabi) {
      surface_bytes = panels * 96u + result.tabi_mesh_vertices * 56u +
                      result.tabi_mesh_triangles * 80u;
      const std::size_t basis_count = 40u;
      solver_bytes =
          std::max<std::size_t>(1, unknowns) * sizeof(double) * basis_count;
      operator_bytes = std::max<std::size_t>(1, unknowns) * sizeof(double) * 8u;
      tree_bytes =
          result.tabi_tree_nodes * 96u + result.tabi_tree_leaf_count * 64u;
      tree_bytes +=
          panels * 14u * sizeof(double) + panels * sizeof(std::size_t);
      operator_bytes += panels * 2u * sizeof(double);
      const std::size_t matvec_calls =
          std::max<std::size_t>(1, result.tabi_matvec_calls);
      const std::size_t interaction_entries =
          (result.tabi_singular_near_interactions +
           result.tabi_leaf_residual_panel_interactions +
           result.tabi_leaf_tile_interactions +
           result.tabi_far_node_interactions +
           result.tabi_smooth_leaf_node_interactions) /
          matvec_calls;
      tree_bytes += interaction_entries * sizeof(std::size_t) +
                    (panels + 1u) * 5u * sizeof(std::size_t);
    } else {
      const std::size_t n = std::max<std::size_t>(1, result.grid_points);
      surface_bytes =
          (result.solute_points + result.solvent_points +
           result.ion_accessible_points + result.stern_excluded_points) *
          sizeof(std::uint32_t);
      solver_bytes = n * sizeof(double) * 8u;
      operator_bytes = n * sizeof(double) * 6u;
      tree_bytes = 0;
    }
  }
  const std::size_t estimated_total =
      std::max(result.memory_bytes,
               surface_bytes + solver_bytes + operator_bytes + tree_bytes);
  memory_profile_out_
      << result.trajectory_id << ',' << result.frame_index << ','
      << result.global_index << ',' << csv_escape(current_source_label_) << ','
      << csv_escape(result.pbe_backend) << ',' << result.atom_count << ','
      << result.grid_points << ',' << panels << ',' << unknowns << ','
      << csv_escape(is_tabi ? result.tabi_formulation : std::string("grid"))
      << ','
      << csv_escape(is_tabi ? result.tabi_matvec_mode : result.solver_mode)
      << ','
      << csv_escape(is_tabi ? result.tabi_preconditioner_effective
                            : std::string("grid"))
      << ','
      << csv_escape(is_tabi ? result.tabi_surface_generator
                            : std::string("grid"))
      << ','
      << csv_escape(is_tabi ? result.tabi_discretization
                            : std::string("finite_difference"))
      << ',' << estimated_total << ',' << surface_bytes << ',' << solver_bytes
      << ',' << operator_bytes << ',' << tree_bytes << ','
      << result.memory_rss_before_bytes << ',' << result.memory_rss_after_bytes
      << ',' << result.memory_rss_delta_bytes << ','
      << result.memory_rss_delta_signed_bytes << ','
      << result.memory_peak_working_set_bytes << ',' << seconds(timing.total_ms)
      << ',' << seconds(timing.solver_ms) << ',' << csv_escape(result.status)
      << '\n';
}

void AnalyzerImpl::write_tabi_formulation_audit_header_if_needed() const {
  if (!tabi_formulation_audit_out_.is_open()) {
    tabi_formulation_audit_out_.open("pbe_tabi_formulation_audit.csv",
                                     std::ios::out | std::ios::trunc);
    if (!tabi_formulation_audit_out_.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_tabi_formulation_audit.csv.");
  }
  if (!tabi_formulation_audit_header_written_) {
    tabi_formulation_audit_out_
        << "trajectory_id,frame_index,global_index,fixture,pbe_backend,"
           "status,tabi_status,convergence_reason,reaction_energy_source,"
           "formulation,energy_projection,preconditioner_requested,"
           "preconditioner_effective,surface_generator,discretization,"
           "fallback_used,rhs_scale,"
           "dielectric_contrast,operator_contrast,unknown_count,bie_system,"
           "lhs_diag_phi,lhs_diag_phin,energy_equation,kernel_convention,"
           "derivative_mode,source_norm_s1,source_norm_s2,phi_norm,phin_norm,"
           "diagonal_mode,diag_min,"
           "diag_max,kappa_inv_A,coulomb_factor_kcal_mol,"
           "boundary_energy_kJmol,reference_energy_kJmol,"
           "selected_energy_scale,selected_energy_kJmol,residual,iterations,"
           "surface_geometry,panels,solid_angle_fallbacks,solid_angle_clamps,matvec_mode,"
           "near_integration,near_self_pairs,near_one_ring_pairs,"
           "near_duffy_pairs,near_centroid_pairs,near_fallbacks,"
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
           "near_k2_policy,near_k3_policy,near_k4_policy\n";
    tabi_formulation_audit_header_written_ = true;
  }
}

void AnalyzerImpl::append_tabi_formulation_audit_row(
    const FrameResult &result) const {
  if (result.pbe_backend != "tabi")
    return;
  write_tabi_formulation_audit_header_if_needed();
  auto seconds = [](double ms) { return ms / 1000.0; };
  tabi_formulation_audit_out_
      << result.trajectory_id << ',' << result.frame_index << ','
      << result.global_index << ',' << csv_escape(current_source_label_) << ','
      << csv_escape(result.pbe_backend) << ',' << csv_escape(result.status)
      << ',' << csv_escape(result.tabi_status) << ','
      << csv_escape(result.tabi_convergence_reason) << ','
      << csv_escape(result.tabi_reaction_energy_source) << ','
      << csv_escape(result.tabi_formulation) << ','
      << csv_escape(result.tabi_energy_projection) << ','
      << csv_escape(result.tabi_preconditioner_requested) << ','
      << csv_escape(result.tabi_preconditioner_effective) << ','
      << csv_escape(result.tabi_surface_generator) << ','
      << csv_escape(result.tabi_discretization) << ','
      << (result.tabi_fallback_used ? "true" : "false") << ','
      << result.tabi_rhs_scale << ',' << result.tabi_dielectric_contrast << ','
      << result.tabi_operator_contrast << ',' << result.tabi_unknown_count
      << ',' << csv_escape(result.tabi_bie_system) << ','
      << result.tabi_lhs_diag_phi << ',' << result.tabi_lhs_diag_phin << ','
      << csv_escape(result.tabi_energy_equation) << ','
      << csv_escape(result.tabi_kernel_convention) << ','
      << csv_escape(result.tabi_derivative_mode) << ','
      << result.tabi_source_norm_s1 << ',' << result.tabi_source_norm_s2 << ','
      << result.tabi_phi_norm << ',' << result.tabi_phin_norm << ','
      << csv_escape(args_.tabi_diagonal) << ',' << result.tabi_diag_min << ','
      << result.tabi_diag_max << ',' << result.tabi_kappa_inv_a << ','
      << result.tabi_coulomb_factor_kcal_mol << ','
      << result.tabi_boundary_energy_kj_mol << ','
      << result.tabi_reference_energy_kj_mol << ','
      << result.tabi_selected_energy_scale << ',' << result.energy_kj_mol << ','
      << result.linear_residual << ',' << result.solvent_iterations << ','
      << csv_escape(result.tabi_surface_geometry) << ','
      << result.tabi_surface_panels << ',' << result.tabi_solid_angle_fallbacks << ',' << result.tabi_solid_angle_clamps
      << ',' << csv_escape(result.tabi_matvec_mode) << ','
      << csv_escape(result.tabi_near_integration) << ','
      << result.tabi_near_self_pairs << ',' << result.tabi_near_one_ring_pairs
      << ',' << result.tabi_near_duffy_pairs << ','
      << result.tabi_near_centroid_pairs << ',' << result.tabi_near_fallbacks
      << ',' << seconds(result.tabi_near_integration_ms) << ','
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
      << csv_escape(result.tabi_near_k4_policy) << '\n';
}

void AnalyzerImpl::maybe_write_parity_stage(const FrameResult &result,
                                            const ReactionSolve &solve) const {
  if (args_.test_pqr.empty() || args_.parity_dump == "off")
    return;
  std::ofstream out;
  if (!parity_stage_header_written_) {
    out.open("pbe_parity_stage.csv", std::ios::out | std::ios::trunc);
    if (!out.is_open())
      throw std::runtime_error("PBE: failed to open pbe_parity_stage.csv.");
    out << "trajectory_id,frame_index,global_index,source_label,pbe_model,"
           "pbe_model_effective,solver_mode,nl_solver,nl_inner_solver,"
           "nl_inner_effective,nl_status,boundary_mode,"
           "test_boundary_convention,test_field_units,"
           "test_operator_convention,charge_assignment,"
           "surface_effective,ion_access,test_nl_source,"
           "test_energy_convention,grid_nx,grid_ny,grid_nz,grid_points,"
           "grid_origin_x_A,grid_origin_y_A,grid_origin_z_A,"
           "grid_center_x_A,grid_center_y_A,grid_center_z_A,grid_spacing_A,"
           "total_charge,charge_assigned,charge_moment_error,solute_points,"
           "solvent_points,ion_accessible_points,stern_excluded_points,"
           "eps_sum_min,eps_sum_max,eps_sum_mean,screen_points,screen_sum,"
           "screen_rms,screen_max,boundary_rms,boundary_max,rhs_scale,"
           "linear_residual,nl_residual,nl_continuation_steps,nl_backtracks,"
           "nl_clipped_points,source_sum,source_rms,source_max,"
           "nonlinear_residual_max_abs,newton_residual_before,"
           "newton_residual_after,newton_correction_residual,"
           "newton_correction_max_residual,newton_inner_residual_before,"
           "newton_inner_residual_after,newton_inner_iterations,"
           "newton_inner_levels,newton_inner_cycles,newton_jac_diag_min,"
           "newton_jac_diag_max,newton_screen_points,"
           "newton_active_nonlinear_points,newton_accepted_lambda,"
           "newton_rejected_steps,newton_line_search_status,"
           "newton_stop_reason,newton_derivative_samples,"
           "newton_derivative_max_abs,newton_derivative_max_rel,"
           "reaction_energy_source,solvent_phase_rf_kT,"
           "solvent_phase_rf_kJmol,vacuum_phase_rf_kT,"
           "vacuum_phase_rf_kJmol,atom_corrected_rxn_kT,"
           "atom_corrected_rxn_kJmol,grid_corrected_rxn_kT,"
           "grid_corrected_rxn_kJmol,coulomb_energy_kT,"
           "coulomb_energy_kJmol,final_E_rxn_kT,final_E_rxn_kJmol,"
           "iscm_boundary_points,iscm_charged_boundary_points,"
           "iscm_induced_charge_sum,iscm_induced_charge_l1,"
           "iscm_induced_charge_l2,iscm_induced_charge_max,"
           "iscm_water_rf_kT,iscm_water_rf_kJmol,iscm_vacuum_rf_kT,"
           "iscm_vacuum_rf_kJmol,iscm_water_minus_vacuum_kT,"
           "iscm_water_minus_vacuum_kJmol,iscm_corrected_rxn_kT,"
           "iscm_corrected_rxn_kJmol,iscm_finalization_mode,"
           "iscm_position_source,iscm_position_rms_vs_oracle,"
           "iscm_position_oracle_matches,iscm_native_independent,iscm_status,"
           "parity_map_dump,"
           "test_charge_grid,test_surface_map,test_boundary_map,"
           "status,test_midpoint_epsilon,delphi_face_crossings,"
           "delphi_eps_face_matches,delphi_boundary_grid_points,"
           "delphi_charged_boundary_points,delphi_iscm_status,"
           "test_vdwms_remap,vdwms_enabled,vdwms_boundary_initial,"
           "vdwms_boundary_final,vdwms_added,vdwms_removed,"
           "vdwms_exposed_points,vdwms_midpoint_rms,"
           "midpoint_eps_sum_min,midpoint_eps_sum_max,"
           "midpoint_eps_sum_mean,midpoint_eps_sum_interior_mean,"
           "nonlinear_stress_kT,nonlinear_stress_kJmol,"
           "nonlinear_osmotic_kT,nonlinear_osmotic_kJmol,"
           "nonlinear_energy_correction_kT,"
           "nonlinear_energy_correction_kJmol,"
           "nonlinear_energy_status\n";
    parity_stage_header_written_ = true;
  } else {
    out.open("pbe_parity_stage.csv", std::ios::out | std::ios::app);
    if (!out.is_open())
      throw std::runtime_error("PBE: failed to append pbe_parity_stage.csv.");
  }

  out << std::setprecision(12);
  const ParityAudit &audit = solve.parity;
  out << result.trajectory_id << ',' << result.frame_index << ','
      << result.global_index << ',' << csv_escape(current_source_label_) << ','
      << result.pbe_model << ',' << result.pbe_model_effective << ','
      << result.solver_mode << ',' << result.nl_solver << ','
      << result.nl_inner_solver << ',' << result.newton_inner_effective << ','
      << result.nl_status << ',' << result.boundary_mode << ','
      << result.test_boundary_convention << ',' << result.test_field_units
      << ',' << result.test_operator_convention << ','
      << result.charge_assignment << ',' << result.surface_effective << ','
      << result.ion_access << ',' << args_.test_nl_source << ','
      << args_.test_energy_convention << ',' << result.nx << ',' << result.ny
      << ',' << result.nz << ',' << result.grid_points << ','
      << result.grid_origin_x_A << ',' << result.grid_origin_y_A << ','
      << result.grid_origin_z_A << ',' << result.grid_center_x_A << ','
      << result.grid_center_y_A << ',' << result.grid_center_z_A << ','
      << result.grid_spacing_A << ',' << result.total_charge << ','
      << result.charge_assigned << ',' << result.charge_moment_error << ','
      << result.solute_points << ',' << result.solvent_points << ','
      << result.ion_accessible_points << ',' << result.stern_excluded_points
      << ',' << audit.eps_sum_min << ',' << audit.eps_sum_max << ','
      << audit.eps_sum_mean << ',' << audit.screen_points << ','
      << audit.screen_sum << ',' << audit.screen_rms << ',' << audit.screen_max
      << ',' << result.boundary_rms << ',' << result.boundary_max << ','
      << result.rhs_scale << ',' << result.linear_residual << ','
      << result.nl_residual << ',' << result.nl_continuation_steps << ','
      << result.nl_backtracks << ',' << result.nl_clipped_points << ','
      << audit.source_sum << ',' << audit.source_rms << ',' << audit.source_max
      << ',' << audit.residual_max_abs << ',' << result.newton_residual_before
      << ',' << result.newton_residual_after << ','
      << result.newton_correction_residual << ','
      << result.newton_correction_max_residual << ','
      << result.newton_inner_residual_before << ','
      << result.newton_inner_residual_after << ','
      << result.newton_inner_iterations << ',' << result.newton_inner_levels
      << ',' << result.newton_inner_cycles << ',' << result.newton_jac_diag_min
      << ',' << result.newton_jac_diag_max << ',' << result.newton_screen_points
      << ',' << result.newton_active_nonlinear_points << ','
      << result.newton_accepted_lambda << ',' << result.newton_rejected_steps
      << ',' << csv_escape(result.newton_line_search_status) << ','
      << csv_escape(result.newton_stop_reason) << ','
      << audit.derivative_samples << ',' << audit.derivative_max_abs << ','
      << audit.derivative_max_rel << ','
      << csv_escape(result.reaction_energy_source) << ','
      << result.solvent_phase_kt << ',' << result.solvent_phase_kj_mol << ','
      << result.vacuum_phase_kt << ',' << result.vacuum_phase_kj_mol << ','
      << result.atom_corrected_kt << ',' << result.atom_corrected_kj_mol << ','
      << result.grid_corrected_kt << ',' << result.grid_corrected_kj_mol << ','
      << result.coulomb_energy_kt << ',' << result.coulomb_energy_kj_mol << ','
      << result.energy_kt << ',' << result.energy_kj_mol << ','
      << result.iscm_boundary_points << ','
      << result.iscm_charged_boundary_points << ','
      << result.iscm_induced_charge_sum << ',' << result.iscm_induced_charge_l1
      << ',' << result.iscm_induced_charge_l2 << ','
      << result.iscm_induced_charge_max << ',' << result.iscm_water_phase_kt
      << ',' << result.iscm_water_phase_kj_mol << ','
      << result.iscm_vacuum_phase_kt << ',' << result.iscm_vacuum_phase_kj_mol
      << ',' << result.iscm_water_minus_vacuum_kt << ','
      << result.iscm_water_minus_vacuum_kj_mol << ','
      << result.iscm_corrected_kt << ',' << result.iscm_corrected_kj_mol << ','
      << csv_escape(result.iscm_finalization_mode) << ','
      << csv_escape(result.iscm_position_source) << ','
      << result.iscm_position_rms_vs_oracle << ','
      << result.iscm_position_oracle_matches << ','
      << (result.iscm_native_independent ? "true" : "false") << ','
      << csv_escape(result.iscm_status) << ','
      << csv_escape(args_.parity_map_dump) << ','
      << csv_escape(args_.test_charge_grid) << ','
      << csv_escape(args_.test_surface_map) << ','
      << csv_escape(args_.test_boundary_map) << ',' << result.status << ','
      << csv_escape(args_.test_midpoint_epsilon) << ','
      << result.delphi_face_crossings << ',' << result.delphi_eps_face_matches
      << ',' << result.delphi_boundary_grid_points << ','
      << result.delphi_charged_boundary_points << ','
      << csv_escape(result.delphi_iscm_status) << ','
      << csv_escape(args_.test_vdwms_remap) << ','
      << csv_escape(result.vdwms_enabled) << ','
      << result.vdwms_boundary_initial << ',' << result.vdwms_boundary_final
      << ',' << result.vdwms_added << ',' << result.vdwms_removed << ','
      << result.vdwms_exposed_points << ',' << result.vdwms_midpoint_rms << ','
      << audit.midpoint_eps_sum_min << ',' << audit.midpoint_eps_sum_max << ','
      << audit.midpoint_eps_sum_mean << ','
      << audit.midpoint_eps_sum_interior_mean << ','
      << result.nonlinear_stress_kt << ',' << result.nonlinear_stress_kj_mol
      << ',' << result.nonlinear_osmotic_kt << ','
      << result.nonlinear_osmotic_kj_mol << ','
      << result.nonlinear_energy_correction_kt << ','
      << result.nonlinear_energy_correction_kj_mol << ','
      << csv_escape(result.nonlinear_energy_status) << '\n';
}

bool AnalyzerImpl::barnase_forensic_enabled() const {
  if (args_.test_pqr.empty() || args_.parity_dump == "off")
    return false;
  std::string probe = args_.test_pqr;
  try {
    probe += " ";
    probe += std::filesystem::path(args_.test_pqr).filename().string();
  } catch (...) {
  }
  std::transform(
      probe.begin(), probe.end(), probe.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return probe.find("barnase") != std::string::npos;
}

void AnalyzerImpl::maybe_write_iscm_surface_charge_samples(
    const FrameResult &result, const IscmAudit &audit) const {
  if (args_.test_pqr.empty() || args_.parity_dump != "all" ||
      audit.samples.empty())
    return;

  std::ofstream out;
  if (!iscm_surface_samples_header_written_) {
    out.open("pbe_iscm_surface_charge_samples.csv",
             std::ios::out | std::ios::trunc);
    if (!out.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_iscm_surface_charge_samples.csv.");
    out << "trajectory_id,frame_index,global_index,kind,rank,i,j,k,x_A,y_A,"
           "z_A,phi_center,phi_neighbor_average,charge_adjustment,"
           "induced_charge,atom_potential_sum,energy_contribution_kT,"
           "energy_contribution_kJmol,iscm_status\n";
    iscm_surface_samples_header_written_ = true;
  } else {
    out.open("pbe_iscm_surface_charge_samples.csv",
             std::ios::out | std::ios::app);
    if (!out.is_open())
      throw std::runtime_error(
          "PBE: failed to append pbe_iscm_surface_charge_samples.csv.");
  }

  const double kt_kj = kKBKJPerMolK * std::max(1.0, args_.temp);
  out << std::setprecision(12);
  for (const IscmSample &sample : audit.samples) {
    out << result.trajectory_id << ',' << result.frame_index << ','
        << result.global_index << ',' << csv_escape(sample.kind) << ','
        << sample.rank << ',' << sample.i << ',' << sample.j << ',' << sample.k
        << ',' << sample.x << ',' << sample.y << ',' << sample.z << ','
        << sample.phi_center << ',' << sample.phi_neighbor_average << ','
        << sample.charge_adjustment << ',' << sample.induced_charge << ','
        << sample.atom_potential_sum << ',' << sample.energy_contribution_kt
        << ',' << sample.energy_contribution_kt * kt_kj << ','
        << csv_escape(audit.status) << '\n';
  }

  std::ofstream terms;
  if (!iscm_energy_terms_header_written_) {
    terms.open("pbe_iscm_energy_terms.csv", std::ios::out | std::ios::trunc);
    if (!terms.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_iscm_energy_terms.csv.");
    terms << "trajectory_id,frame_index,global_index,rank,i,j,k,"
             "surface_x_A,surface_y_A,surface_z_A,induced_charge,"
             "atom_potential_sum,energy_contribution_kT,"
             "energy_contribution_kJmol,iscm_status\n";
    iscm_energy_terms_header_written_ = true;
  } else {
    terms.open("pbe_iscm_energy_terms.csv", std::ios::out | std::ios::app);
    if (!terms.is_open())
      throw std::runtime_error(
          "PBE: failed to append pbe_iscm_energy_terms.csv.");
  }
  terms << std::setprecision(12);
  std::size_t rank = 0;
  for (const IscmSample &sample : audit.samples) {
    if (sample.kind != "top_energy")
      continue;
    ++rank;
    terms << result.trajectory_id << ',' << result.frame_index << ','
          << result.global_index << ',' << rank << ',' << sample.i << ','
          << sample.j << ',' << sample.k << ',' << sample.x << ',' << sample.y
          << ',' << sample.z << ',' << sample.induced_charge << ','
          << sample.atom_potential_sum << ',' << sample.energy_contribution_kt
          << ',' << sample.energy_contribution_kt * kt_kj << ','
          << csv_escape(audit.status) << '\n';
  }
}

void AnalyzerImpl::maybe_write_barnase_forensic_audits(
    const FrameResult &result, const std::vector<Atom> &atoms,
    const FieldSolve &solvent, const FieldSolve &vacuum,
    const ReactionSolve &solve, const ForensicAudit &solvent_audit) const {
  if (!barnase_forensic_enabled())
    return;

  struct GroupStats {
    std::size_t count{0};
    double charge_sum{0.0};
    double radius_sum{0.0};
    double radius_min{std::numeric_limits<double>::max()};
    double radius_max{0.0};
    double media_radius_sum{0.0};
    double media_radius_min{std::numeric_limits<double>::max()};
    double media_radius_max{0.0};
    double x_min{std::numeric_limits<double>::max()};
    double x_max{std::numeric_limits<double>::lowest()};
    double y_min{std::numeric_limits<double>::max()};
    double y_max{std::numeric_limits<double>::lowest()};
    double z_min{std::numeric_limits<double>::max()};
    double z_max{std::numeric_limits<double>::lowest()};
    std::size_t radius_floor_count{0};
    std::size_t radius_changed_count{0};
  };

  auto update_group = [&](GroupStats &stats, const Atom &atom) {
    ++stats.count;
    const double media_radius = effective_surface_radius(atom);
    stats.charge_sum += atom.q;
    stats.radius_sum += atom.radius;
    stats.radius_min = std::min(stats.radius_min, atom.radius);
    stats.radius_max = std::max(stats.radius_max, atom.radius);
    stats.media_radius_sum += media_radius;
    stats.media_radius_min = std::min(stats.media_radius_min, media_radius);
    stats.media_radius_max = std::max(stats.media_radius_max, media_radius);
    stats.x_min = std::min(stats.x_min, atom.x);
    stats.x_max = std::max(stats.x_max, atom.x);
    stats.y_min = std::min(stats.y_min, atom.y);
    stats.y_max = std::max(stats.y_max, atom.y);
    stats.z_min = std::min(stats.z_min, atom.z);
    stats.z_max = std::max(stats.z_max, atom.z);
    if (media_radius > atom.radius + 1.0e-12)
      ++stats.radius_floor_count;
    if (std::abs(media_radius - atom.radius) > 1.0e-12)
      ++stats.radius_changed_count;
  };
  auto finish_minmax = [](double value, std::size_t count) {
    return count > 0 ? value : 0.0;
  };
  auto atom_element = [&](const Atom &atom) {
    return inferred_element_symbol(atom);
  };
  auto residue_key = [](const Atom &atom) {
    std::ostringstream oss;
    oss << (atom.chain.empty() ? "_" : atom.chain) << ':'
        << (atom.resname.empty() ? "UNK" : atom.resname) << ':' << atom.resnr;
    return oss.str();
  };

  GroupStats all;
  std::map<std::string, GroupStats> by_element;
  std::map<std::string, GroupStats> by_residue;
  std::size_t h_floor = 0;
  std::size_t c_floor = 0;
  std::size_t other_floor = 0;
  for (const Atom &atom : atoms) {
    update_group(all, atom);
    update_group(by_element[atom_element(atom)], atom);
    update_group(by_residue[residue_key(atom)], atom);
    if (effective_surface_radius(atom) > atom.radius + 1.0e-12) {
      const std::string element = atom_element(atom);
      if (element == "H")
        ++h_floor;
      else if (element == "C")
        ++c_floor;
      else
        ++other_floor;
    }
  }

  const GridShape &shape = solvent.shape;
  auto write_input_header = [&](std::ofstream &out) {
    out << "trajectory_id,frame_index,global_index,section,key,atom_count,"
           "total_charge,radius_sum_A,radius_min_A,radius_max_A,x_min_A,"
           "x_max_A,y_min_A,y_max_A,z_min_A,z_max_A,grid_nx,grid_ny,"
           "grid_nz,grid_origin_x_A,grid_origin_y_A,grid_origin_z_A,"
           "grid_center_x_A,grid_center_y_A,grid_center_z_A,grid_spacing_A,"
           "pqr_source,radius_convention,radius_floor_count,"
           "media_radius_sum_A,media_radius_min_A,media_radius_max_A,"
           "radius_changed_count\n";
  };
  auto write_group_row = [&](std::ofstream &out, const std::string &section,
                             const std::string &key, const GroupStats &stats) {
    out << result.trajectory_id << ',' << result.frame_index << ','
        << result.global_index << ',' << csv_escape(section) << ','
        << csv_escape(key) << ',' << stats.count << ',' << stats.charge_sum
        << ',' << stats.radius_sum << ','
        << finish_minmax(stats.radius_min, stats.count) << ','
        << finish_minmax(stats.radius_max, stats.count) << ','
        << finish_minmax(stats.x_min, stats.count) << ','
        << finish_minmax(stats.x_max, stats.count) << ','
        << finish_minmax(stats.y_min, stats.count) << ','
        << finish_minmax(stats.y_max, stats.count) << ','
        << finish_minmax(stats.z_min, stats.count) << ','
        << finish_minmax(stats.z_max, stats.count) << ',' << shape.nx << ','
        << shape.ny << ',' << shape.nz << ',' << shape.ox << ',' << shape.oy
        << ',' << shape.oz << ','
        << shape.ox + 0.5 * static_cast<double>(shape.nx - 1) * shape.h << ','
        << shape.oy + 0.5 * static_cast<double>(shape.ny - 1) * shape.h << ','
        << shape.oz + 0.5 * static_cast<double>(shape.nz - 1) * shape.h << ','
        << shape.h << ',' << csv_escape(args_.test_pqr) << ','
        << csv_escape(effective_radius_convention()) << ','
        << stats.radius_floor_count << ',' << stats.media_radius_sum << ','
        << finish_minmax(stats.media_radius_min, stats.count) << ','
        << finish_minmax(stats.media_radius_max, stats.count) << ','
        << stats.radius_changed_count << '\n';
  };

  std::ofstream inputs;
  if (!forensic_inputs_header_written_) {
    inputs.open("pbe_barnase_forensic_inputs.csv",
                std::ios::out | std::ios::trunc);
    if (!inputs.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_barnase_forensic_inputs.csv.");
    write_input_header(inputs);
    forensic_inputs_header_written_ = true;
  } else {
    inputs.open("pbe_barnase_forensic_inputs.csv",
                std::ios::out | std::ios::app);
    if (!inputs.is_open())
      throw std::runtime_error(
          "PBE: failed to append pbe_barnase_forensic_inputs.csv.");
  }
  inputs << std::setprecision(12);
  write_group_row(inputs, "summary", "all", all);
  for (const auto &[key, stats] : by_element)
    write_group_row(inputs, "element", key, stats);
  for (const auto &[key, stats] : by_residue)
    write_group_row(inputs, "residue", key, stats);

  std::ofstream surface;
  if (!forensic_surface_header_written_) {
    surface.open("pbe_barnase_surface_audit.csv",
                 std::ios::out | std::ios::trunc);
    if (!surface.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_barnase_surface_audit.csv.");
    surface
        << "trajectory_id,frame_index,global_index,surface,"
           "charge_assignment,radius_convention,solute_points,solvent_points,"
           "ion_accessible_points,stern_excluded_points,face_cross_x,"
           "face_cross_y,face_cross_z,eps_face_min,eps_face_max,"
           "eps_face_mean,eps_sum_min,eps_sum_max,eps_sum_mean,"
           "vdw_volume_estimate_A3,h_radius_floor_count,c_radius_floor_count,"
           "other_radius_floor_count,eps_face_count_x,eps_face_count_y,"
           "eps_face_count_z,internal_face_count_x,internal_face_count_y,"
           "internal_face_count_z,eps_face_mean_x,eps_face_mean_y,"
           "eps_face_mean_z,vdwms_enabled,vdwms_boundary_initial,"
           "vdwms_boundary_final,vdwms_added,vdwms_removed,"
           "vdwms_exposed_points,vdwms_midpoint_rms,"
           "midpoint_eps_sum_min,midpoint_eps_sum_max,"
           "midpoint_eps_sum_mean,midpoint_eps_sum_interior_mean\n";
    forensic_surface_header_written_ = true;
  } else {
    surface.open("pbe_barnase_surface_audit.csv",
                 std::ios::out | std::ios::app);
    if (!surface.is_open())
      throw std::runtime_error(
          "PBE: failed to append pbe_barnase_surface_audit.csv.");
  }
  surface << std::setprecision(12);
  surface << result.trajectory_id << ',' << result.frame_index << ','
          << result.global_index << ',' << csv_escape(args_.surface) << ','
          << csv_escape(args_.charge_assignment) << ','
          << csv_escape(effective_radius_convention()) << ','
          << solve.solute_points << ',' << solve.solvent_points << ','
          << solve.ion_accessible_points << ',' << solve.stern_excluded_points
          << ',' << solvent_audit.face_cross_x << ','
          << solvent_audit.face_cross_y << ',' << solvent_audit.face_cross_z
          << ',' << solvent_audit.eps_face_min << ','
          << solvent_audit.eps_face_max << ',' << solvent_audit.eps_face_mean
          << ',' << solve.parity.eps_sum_min << ',' << solve.parity.eps_sum_max
          << ',' << solve.parity.eps_sum_mean << ','
          << solvent_audit.vdw_volume_estimate_A3 << ',' << h_floor << ','
          << c_floor << ',' << other_floor << ','
          << solvent_audit.eps_face_count_x << ','
          << solvent_audit.eps_face_count_y << ','
          << solvent_audit.eps_face_count_z << ','
          << solvent_audit.internal_face_count_x << ','
          << solvent_audit.internal_face_count_y << ','
          << solvent_audit.internal_face_count_z << ','
          << solvent_audit.eps_face_mean_x << ','
          << solvent_audit.eps_face_mean_y << ','
          << solvent_audit.eps_face_mean_z << ','
          << csv_escape(solve.vdwms_enabled) << ','
          << solve.vdwms_boundary_initial << ',' << solve.vdwms_boundary_final
          << ',' << solve.vdwms_added << ',' << solve.vdwms_removed << ','
          << solve.vdwms_exposed_points << ',' << solve.vdwms_midpoint_rms
          << ',' << solve.parity.midpoint_eps_sum_min << ','
          << solve.parity.midpoint_eps_sum_max << ','
          << solve.parity.midpoint_eps_sum_mean << ','
          << solve.parity.midpoint_eps_sum_interior_mean << '\n';

  std::ofstream radius_media;
  if (!radius_media_audit_header_written_) {
    radius_media.open("pbe_barnase_radius_media_audit.csv",
                      std::ios::out | std::ios::trunc);
    if (!radius_media.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_barnase_radius_media_audit.csv.");
    radius_media
        << "trajectory_id,frame_index,global_index,atom_index,atom_name,"
           "raw_element,inferred_element,resname,resnr,chain,q,raw_radius_A,"
           "media_radius_A,radius_source,radius_changed,radius_convention,"
           "pqr_source\n";
    radius_media_audit_header_written_ = true;
  } else {
    radius_media.open("pbe_barnase_radius_media_audit.csv",
                      std::ios::out | std::ios::app);
    if (!radius_media.is_open())
      throw std::runtime_error(
          "PBE: failed to append pbe_barnase_radius_media_audit.csv.");
  }
  radius_media << std::setprecision(12);
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const Atom &atom = atoms[i];
    const double media_radius = effective_surface_radius(atom);
    radius_media << result.trajectory_id << ',' << result.frame_index << ','
                 << result.global_index << ',' << (i + 1) << ','
                 << csv_escape(atom.atom_name) << ','
                 << csv_escape(atom.element.empty() ? "NA" : atom.element)
                 << ',' << csv_escape(inferred_element_symbol(atom)) << ','
                 << csv_escape(atom.resname) << ',' << atom.resnr << ','
                 << csv_escape(atom.chain) << ',' << atom.q << ','
                 << atom.radius << ',' << media_radius << ','
                 << csv_escape(radius_media_source(atom)) << ','
                 << (std::abs(media_radius - atom.radius) > 1.0e-12 ? 1 : 0)
                 << ',' << csv_escape(effective_radius_convention()) << ','
                 << csv_escape(args_.test_pqr) << '\n';
  }

  std::ofstream op;
  if (!forensic_operator_header_written_) {
    op.open("pbe_barnase_operator_audit.csv", std::ios::out | std::ios::trunc);
    if (!op.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_barnase_operator_audit.csv.");
    op << "trajectory_id,frame_index,global_index,rhs_scale,rho_sum,rho_l1,"
          "rho_l2,rho_mx,rho_my,rho_mz,boundary_rms,boundary_max,"
          "linear_residual,denom_min,denom_max,denom_mean,screen_sum,"
          "screen_max,solvent_phase_rf_kJmol,vacuum_phase_rf_kJmol,"
          "atom_corrected_rxn_kJmol,grid_corrected_rxn_kJmol,"
          "coulomb_energy_kJmol,final_E_rxn_kJmol,reaction_energy_source,"
          "iscm_boundary_points,iscm_charged_boundary_points,"
          "iscm_induced_charge_sum,iscm_induced_charge_l1,"
          "iscm_induced_charge_l2,iscm_induced_charge_max,"
          "iscm_water_rf_kJmol,iscm_vacuum_rf_kJmol,"
          "iscm_water_minus_vacuum_kJmol,iscm_corrected_rxn_kJmol,"
          "iscm_finalization_mode,iscm_position_source,"
          "iscm_position_rms_vs_oracle,iscm_position_oracle_matches,"
          "iscm_native_independent,iscm_status,"
          "vdwms_enabled,vdwms_boundary_initial,vdwms_boundary_final,"
          "vdwms_added,vdwms_removed,vdwms_exposed_points,"
          "vdwms_midpoint_rms,"
          "status\n";
    forensic_operator_header_written_ = true;
  } else {
    op.open("pbe_barnase_operator_audit.csv", std::ios::out | std::ios::app);
    if (!op.is_open())
      throw std::runtime_error(
          "PBE: failed to append pbe_barnase_operator_audit.csv.");
  }
  op << std::setprecision(12);
  op << result.trajectory_id << ',' << result.frame_index << ','
     << result.global_index << ',' << solve.rhs_scale << ','
     << solvent_audit.rho_sum << ',' << solvent_audit.rho_l1 << ','
     << solvent_audit.rho_l2 << ',' << solvent_audit.rho_mx << ','
     << solvent_audit.rho_my << ',' << solvent_audit.rho_mz << ','
     << solve.boundary_rms << ',' << solve.boundary_max << ','
     << solve.linear_residual << ',' << solvent_audit.denom_min << ','
     << solvent_audit.denom_max << ',' << solvent_audit.denom_mean << ','
     << solvent_audit.screen_sum << ',' << solvent_audit.screen_max << ','
     << solve.solvent_phase_kj_mol << ',' << solve.vacuum_phase_kj_mol << ','
     << solve.atom_corrected_kj_mol << ',' << solve.grid_corrected_kj_mol << ','
     << solve.coulomb_energy_kj_mol << ',' << solve.energy_kj_mol << ','
     << csv_escape(solve.reaction_energy_source) << ','
     << solve.iscm_boundary_points << ',' << solve.iscm_charged_boundary_points
     << ',' << solve.iscm_induced_charge_sum << ','
     << solve.iscm_induced_charge_l1 << ',' << solve.iscm_induced_charge_l2
     << ',' << solve.iscm_induced_charge_max << ','
     << solve.iscm_water_phase_kj_mol << ',' << solve.iscm_vacuum_phase_kj_mol
     << ',' << solve.iscm_water_minus_vacuum_kj_mol << ','
     << solve.iscm_corrected_kj_mol << ','
     << csv_escape(solve.iscm_finalization_mode) << ','
     << csv_escape(solve.iscm_position_source) << ','
     << solve.iscm_position_rms_vs_oracle << ','
     << solve.iscm_position_oracle_matches << ','
     << (solve.iscm_native_independent ? "true" : "false") << ','
     << csv_escape(solve.iscm_status) << ',' << csv_escape(solve.vdwms_enabled)
     << ',' << solve.vdwms_boundary_initial << ',' << solve.vdwms_boundary_final
     << ',' << solve.vdwms_added << ',' << solve.vdwms_removed << ','
     << solve.vdwms_exposed_points << ',' << solve.vdwms_midpoint_rms << ','
     << (solve.converged ? "ok" : "not_converged") << '\n';

  maybe_write_atom_phi_audit(result, atoms, solvent, vacuum, solve);
  maybe_write_linear_forensic_points(result, atoms, solvent, vacuum, solve);
}

void AnalyzerImpl::maybe_write_atom_phi_audit(
    const FrameResult &result, const std::vector<Atom> &atoms,
    const FieldSolve &solvent, const FieldSolve &vacuum,
    const ReactionSolve &solve) const {
  if (!barnase_forensic_enabled() || solvent.atom_phi.size() != atoms.size() ||
      vacuum.atom_phi.size() != atoms.size())
    return;
  const GridShape &shape = solvent.shape;
  auto grid_index = [&](double value, double origin, std::size_t n) {
    if (n == 0 || shape.h <= 0.0 || !std::isfinite(value))
      return std::size_t{0};
    const double scaled = std::floor((value - origin) / shape.h);
    const double clamped =
        std::clamp(scaled, 0.0, static_cast<double>(n > 0 ? n - 1 : 0));
    return static_cast<std::size_t>(clamped);
  };

  std::ofstream out;
  if (!atom_phi_audit_header_written_) {
    out.open("pbe_atom_phi_audit.csv", std::ios::out | std::ios::trunc);
    if (!out.is_open())
      throw std::runtime_error("PBE: failed to open pbe_atom_phi_audit.csv.");
    out << "trajectory_id,frame_index,global_index,atom_index,atom_name,"
           "element,resname,resnr,chain,x_A,y_A,z_A,q,radius_A,grid_i,"
           "grid_j,grid_k,solvent_phi,vacuum_phi,reaction_phi,"
           "contribution_kJmol\n";
    atom_phi_audit_header_written_ = true;
  } else {
    out.open("pbe_atom_phi_audit.csv", std::ios::out | std::ios::app);
    if (!out.is_open())
      throw std::runtime_error("PBE: failed to append pbe_atom_phi_audit.csv.");
  }
  out << std::setprecision(12);

  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const Atom &atom = atoms[i];
    const double solvent_phi = solvent.atom_phi[i];
    const double vacuum_phi = vacuum.atom_phi[i];
    const double contribution = i < solve.atom_reaction_kj_mol.size()
                                    ? solve.atom_reaction_kj_mol[i]
                                    : 0.0;
    out << result.trajectory_id << ',' << result.frame_index << ','
        << result.global_index << ',' << (i + 1) << ','
        << csv_escape(atom.atom_name) << ','
        << csv_escape(inferred_element_symbol(atom)) << ','
        << csv_escape(atom.resname) << ',' << atom.resnr << ','
        << csv_escape(atom.chain) << ',' << atom.x << ',' << atom.y << ','
        << atom.z << ',' << atom.q << ',' << atom.radius << ','
        << grid_index(atom.x, shape.ox, shape.nx) << ','
        << grid_index(atom.y, shape.oy, shape.ny) << ','
        << grid_index(atom.z, shape.oz, shape.nz) << ',' << solvent_phi << ','
        << vacuum_phi << ',' << (solvent_phi - vacuum_phi) << ','
        << contribution << '\n';
  }
}

void AnalyzerImpl::maybe_write_atom_reaction_potentials(
    const FrameResult &result, const std::vector<Atom> &atoms,
    const ReactionSolve &solve, const std::string &component) const {
  if (args_.atom_potentials != "csv")
    return;
  if (solve.atom_reaction_potential_kj_mol_e.size() != atoms.size() ||
      solve.atom_reaction_kj_mol.size() != atoms.size()) {
    throw std::runtime_error(
        "PBE: atom reaction-potential output is unavailable for the selected "
        "solver path.");
  }

  std::ofstream atom_out;
  const auto atom_mode = atom_reaction_potential_header_written_
                             ? (std::ios::out | std::ios::app)
                             : (std::ios::out | std::ios::trunc);
  atom_out.open("pbe_atom_reaction_potential.csv", atom_mode);
  if (!atom_out.is_open()) {
    throw std::runtime_error(
        "PBE: failed to open pbe_atom_reaction_potential.csv.");
  }
  if (!atom_reaction_potential_header_written_) {
    atom_out
        << "schema_version,trajectory_id,frame_index,global_index,component,"
           "pqr_sha256,backend,reaction_energy_source,atom_index,atom_serial,"
           "atom_name,element,"
           "resname,resnr,chain,x_A,y_A,z_A,q_e,radius_A,"
           "solvent_potential_kJmol_e,vacuum_potential_kJmol_e,"
           "reaction_potential_kJmol_e,reaction_energy_contribution_kJmol,"
           "closure_energy_kJmol,closure_error_kJmol,selected_energy_kJmol,"
           "salt_M,eps_in,eps_out,temperature_K,surface_model,"
           "surface_generator,surface_points_requested,surface_panels,"
           "probe_radius_A,grid_scale_pts_per_A,grid_margin_A,status\n";
    atom_reaction_potential_header_written_ = true;
  }
  atom_out << std::setprecision(12);
  const std::string pqr_sha256 = pqr_sha256_for_component(component);
  const std::string surface_generator = solve.pbe_backend == "tabi"
                                            ? solve.tabi_surface_generator
                                            : "grid-" + args_.surface;
  const int surface_points_requested =
      solve.pbe_backend == "tabi" ? solve.tabi_surface_point_count_requested
                                  : 0;

  struct ResidueAggregate {
    std::size_t atom_count{0};
    double charge{0.0};
    double reaction_energy_kj_mol{0.0};
  };
  using ResidueKey = std::tuple<std::string, int, std::string>;
  std::map<ResidueKey, ResidueAggregate> residues;

  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const Atom &atom = atoms[i];
    auto write_optional = [&](const std::vector<double> &values) {
      if (i < values.size() && std::isfinite(values[i]))
        atom_out << values[i];
    };
    atom_out << "2," << result.trajectory_id << ',' << result.frame_index << ','
             << result.global_index << ',' << csv_escape(component) << ','
             << csv_escape(pqr_sha256) << ',' << csv_escape(solve.pbe_backend)
             << ',' << csv_escape(solve.reaction_energy_source) << ','
             << (i + 1) << ',' << atom.serial << ','
             << csv_escape(atom.atom_name) << ','
             << csv_escape(inferred_element_symbol(atom)) << ','
             << csv_escape(atom.resname) << ',' << atom.resnr << ','
             << csv_escape(atom.chain) << ',' << atom.x << ',' << atom.y << ','
             << atom.z << ',' << atom.q << ',' << atom.radius << ',';
    write_optional(solve.atom_solvent_potential_kj_mol_e);
    atom_out << ',';
    write_optional(solve.atom_vacuum_potential_kj_mol_e);
    atom_out << ',' << solve.atom_reaction_potential_kj_mol_e[i] << ','
             << solve.atom_reaction_kj_mol[i] << ','
             << solve.atom_reaction_energy_closure_kj_mol << ','
             << solve.atom_reaction_energy_closure_error_kj_mol << ','
             << solve.energy_kj_mol << ',' << args_.salt << ','
             << args_.int_diel << ',' << args_.ext_diel << ',' << args_.temp
             << ',' << csv_escape(args_.surface) << ','
             << csv_escape(surface_generator) << ',' << surface_points_requested
             << ',' << solve.tabi_surface_panels << ',' << args_.probe_radius
             << ',' << args_.scale << ',' << args_.margin << ','
             << (solve.converged ? "ok" : "not_converged") << '\n';

    ResidueAggregate &aggregate =
        residues[{atom.chain, atom.resnr, atom.resname}];
    ++aggregate.atom_count;
    aggregate.charge += atom.q;
    aggregate.reaction_energy_kj_mol += solve.atom_reaction_kj_mol[i];
  }

  std::ofstream residue_out;
  const auto residue_mode = residue_electrostatics_header_written_
                                ? (std::ios::out | std::ios::app)
                                : (std::ios::out | std::ios::trunc);
  residue_out.open("pbe_residue_electrostatics.csv", residue_mode);
  if (!residue_out.is_open()) {
    throw std::runtime_error(
        "PBE: failed to open pbe_residue_electrostatics.csv.");
  }
  if (!residue_electrostatics_header_written_) {
    residue_out
        << "schema_version,trajectory_id,frame_index,global_index,component,"
           "pqr_sha256,backend,chain,resnr,resname,atom_count,total_charge_e,"
           "reaction_energy_kJmol,salt_M,eps_in,eps_out,temperature_K,"
           "surface_model,surface_generator,surface_points_requested,"
           "surface_panels,probe_radius_A,status\n";
    residue_electrostatics_header_written_ = true;
  }
  residue_out << std::setprecision(12);
  for (const auto &[key, aggregate] : residues) {
    residue_out << "2," << result.trajectory_id << ',' << result.frame_index
                << ',' << result.global_index << ',' << csv_escape(component)
                << ',' << csv_escape(pqr_sha256) << ','
                << csv_escape(solve.pbe_backend) << ','
                << csv_escape(std::get<0>(key)) << ',' << std::get<1>(key)
                << ',' << csv_escape(std::get<2>(key)) << ','
                << aggregate.atom_count << ',' << aggregate.charge << ','
                << aggregate.reaction_energy_kj_mol << ',' << args_.salt << ','
                << args_.int_diel << ',' << args_.ext_diel << ',' << args_.temp
                << ',' << csv_escape(args_.surface) << ','
                << csv_escape(surface_generator) << ','
                << surface_points_requested << ',' << solve.tabi_surface_panels
                << ',' << args_.probe_radius << ','
                << (solve.converged ? "ok" : "not_converged") << '\n';
  }
}

void AnalyzerImpl::maybe_write_binding_electrostatics(
    const FrameResult &result, const std::vector<Atom> &complex_atoms,
    const std::vector<Atom> &receptor_atoms,
    const std::vector<Atom> &ligand_atoms, const ReactionSolve &complex_solve,
    const ReactionSolve &receptor_solve,
    const ReactionSolve &ligand_solve) const {
  if (args_.binding_cycle != "pqr-components")
    return;

  std::ofstream component_out;
  const auto component_mode = component_electrostatics_header_written_
                                  ? (std::ios::out | std::ios::app)
                                  : (std::ios::out | std::ios::trunc);
  component_out.open("pbe_component_electrostatics.csv", component_mode);
  if (!component_out.is_open()) {
    throw std::runtime_error(
        "PBE: failed to open pbe_component_electrostatics.csv.");
  }
  if (!component_electrostatics_header_written_) {
    component_out
        << "schema_version,trajectory_id,frame_index,global_index,component,"
           "pqr_sha256,backend,atom_count,total_charge_e,reaction_energy_kJmol,"
           "closure_energy_kJmol,closure_error_kJmol,iterations,"
           "internal_residual,true_residual,sampled_direct_operator_rms,"
           "sampled_operator_rows,sampled_direct_true_residual,"
           "fmm_local_gmres_active,plan_build_count,cache_build_count,"
           "fmm_prepare_seconds,fmm_local_matvec_calls,"
           "fmm_local_apply_seconds,fmm_local_moment_seconds,"
           "fmm_local_m2l_seconds,fmm_local_fallback_seconds,"
           "fmm_local_l2p_seconds,surface_seconds,rhs_seconds,"
           "operator_audit_seconds,ilu_setup_seconds,gmres_seconds,"
           "gmres_orthogonalization_seconds,gmres_preconditioner_seconds,"
           "matvec_seconds_mean,energy_projection_seconds,"
           "energy_projection_kernel,"
           "peak_memory_bytes,reaction_energy_source,salt_M,eps_in,eps_out,"
           "temperature_K,surface_model,surface_generator,"
           "surface_points_requested,surface_panels,probe_radius_A,status\n";
    component_electrostatics_header_written_ = true;
  }
  component_out << std::setprecision(12);
  auto write_component = [&](const char *component,
                             const std::vector<Atom> &atoms,
                             const ReactionSolve &solve) {
    double charge = 0.0;
    for (const Atom &atom : atoms)
      charge += atom.q;
    const std::string component_name = component;
    const std::string component_hash = pqr_sha256_for_component(component_name);
    const std::string generator = solve.pbe_backend == "tabi"
                                      ? solve.tabi_surface_generator
                                      : "grid-" + args_.surface;
    const int points_requested = solve.pbe_backend == "tabi"
                                     ? solve.tabi_surface_point_count_requested
                                     : 0;
    component_out << "6," << result.trajectory_id << ',' << result.frame_index
                  << ',' << result.global_index << ',' << component << ','
                  << csv_escape(component_hash) << ','
                  << csv_escape(solve.pbe_backend) << ',' << atoms.size() << ','
                  << charge << ',' << solve.energy_kj_mol << ','
                  << solve.atom_reaction_energy_closure_kj_mol << ','
                  << solve.atom_reaction_energy_closure_error_kj_mol << ','
                  << solve.solvent_total_iterations << ','
                  << solve.linear_residual << ','
                  << solve.tabi_true_residual_rel << ','
                  << solve.tabi_tree_vs_direct_matvec_rel << ','
                  << solve.tabi_matvec_audit_rows << ','
                  << solve.tabi_fmm_local_sampled_direct_true_residual << ','
                  << (solve.tabi_fmm_local_gmres_active ? "true" : "false")
                  << ',' << solve.tabi_fmm_local_plan_build_count << ','
                  << solve.tabi_fmm_local_cache_build_count << ','
                  << solve.tabi_fmm_local_prepare_seconds << ','
                  << solve.tabi_fmm_local_gmres_matvec_calls << ','
                  << solve.tabi_fmm_local_gmres_apply_seconds_total << ','
                  << solve.tabi_fmm_local_moment_seconds_total << ','
                  << solve.tabi_fmm_local_m2l_seconds_total << ','
                  << solve.tabi_fmm_local_fallback_seconds_total << ','
                  << solve.tabi_fmm_local_l2p_seconds_total << ','
                  << (solve.tabi_surface_build_ms * 1.0e-3) << ','
                  << (solve.tabi_rhs_ms * 1.0e-3) << ','
                  << (solve.tabi_matvec_audit_ms * 1.0e-3) << ','
                  << (solve.tabi_preconditioner_setup_ms * 1.0e-3) << ','
                  << (solve.tabi_gmres_ms * 1.0e-3) << ','
                  << (solve.tabi_orthogonalization_ms * 1.0e-3) << ','
                  << (solve.tabi_preconditioner_ms * 1.0e-3) << ','
                  << (solve.tabi_matvec_ms_avg * 1.0e-3) << ','
                  << (solve.tabi_energy_projection_ms * 1.0e-3) << ','
                  << csv_escape(solve.tabi_energy_projection_kernel) << ','
                  << solve.peak_memory_bytes << ','
                  << csv_escape(solve.reaction_energy_source) << ','
                  << args_.salt << ',' << args_.int_diel << ','
                  << args_.ext_diel << ',' << args_.temp << ','
                  << csv_escape(args_.surface) << ',' << csv_escape(generator)
                  << ',' << points_requested << ',' << solve.tabi_surface_panels
                  << ',' << args_.probe_radius << ','
                  << (solve.converged ? "ok" : "not_converged") << '\n';
  };
  write_component("complex", complex_atoms, complex_solve);
  write_component("receptor", receptor_atoms, receptor_solve);
  write_component("ligand", ligand_atoms, ligand_solve);

  const double delta_g_pol_kj = complex_solve.energy_kj_mol -
                                receptor_solve.energy_kj_mol -
                                ligand_solve.energy_kj_mol;
  const double cross_coulomb_kcal = intercomponent_coulomb_energy_kcal(
      receptor_atoms, ligand_atoms, args_.int_diel);
  const double cross_coulomb_kj = cross_coulomb_kcal * kKcalToKJ;
  const double delta_g_elec_kj = delta_g_pol_kj + cross_coulomb_kj;
  const double closure_error_max = std::max(
      {std::abs(complex_solve.atom_reaction_energy_closure_error_kj_mol),
       std::abs(receptor_solve.atom_reaction_energy_closure_error_kj_mol),
       std::abs(ligand_solve.atom_reaction_energy_closure_error_kj_mol)});
  const bool converged = complex_solve.converged && receptor_solve.converged &&
                         ligand_solve.converged;
  const double sampled_audit_max =
      std::max({complex_solve.tabi_tree_vs_direct_matvec_rel,
                receptor_solve.tabi_tree_vs_direct_matvec_rel,
                ligand_solve.tabi_tree_vs_direct_matvec_rel});
  const double sampled_direct_true_residual_max =
      std::max({complex_solve.tabi_fmm_local_sampled_direct_true_residual,
                receptor_solve.tabi_fmm_local_sampled_direct_true_residual,
                ligand_solve.tabi_fmm_local_sampled_direct_true_residual});
  const std::size_t peak_memory_max_bytes = std::max(
      {complex_solve.peak_memory_bytes, receptor_solve.peak_memory_bytes,
       ligand_solve.peak_memory_bytes});

  std::ofstream binding_out;
  const auto binding_mode = binding_electrostatics_header_written_
                                ? (std::ios::out | std::ios::app)
                                : (std::ios::out | std::ios::trunc);
  binding_out.open("pbe_binding_electrostatics.csv", binding_mode);
  if (!binding_out.is_open()) {
    throw std::runtime_error(
        "PBE: failed to open pbe_binding_electrostatics.csv.");
  }
  if (!binding_electrostatics_header_written_) {
    binding_out
        << "schema_version,trajectory_id,frame_index,global_index,backend,"
           "complex_atoms,receptor_atoms,ligand_atoms,complex_pqr_sha256,"
           "receptor_pqr_sha256,ligand_pqr_sha256,Gpol_complex_kJmol,"
           "Gpol_receptor_kJmol,Gpol_ligand_kJmol,Delta_Gpol_bind_kJmol,"
           "E_coul_receptor_ligand_kJmol,Delta_Gelec_bind_kJmol,"
           "coulomb_dielectric,closure_error_max_kJmol,salt_M,eps_in,eps_out,"
           "temperature_K,surface_model,surface_generator,"
           "surface_points_requested,complex_surface_panels,"
           "receptor_surface_panels,ligand_surface_panels,probe_radius_A,"
           "grid_scale_pts_per_A,grid_margin_A,complex_status,"
           "receptor_status,ligand_status,sampled_audit_max,"
           "sampled_direct_true_residual_max,peak_memory_max_bytes,status\n";
    binding_electrostatics_header_written_ = true;
  }
  const std::string binding_surface_generator =
      complex_solve.pbe_backend == "tabi" ? complex_solve.tabi_surface_generator
                                          : "grid-" + args_.surface;
  const int binding_surface_points =
      complex_solve.pbe_backend == "tabi"
          ? complex_solve.tabi_surface_point_count_requested
          : 0;
  binding_out << std::setprecision(12) << "3," << result.trajectory_id << ','
              << result.frame_index << ',' << result.global_index << ','
              << csv_escape(complex_solve.pbe_backend) << ','
              << complex_atoms.size() << ',' << receptor_atoms.size() << ','
              << ligand_atoms.size() << ','
              << csv_escape(pqr_sha256_for_component("complex")) << ','
              << csv_escape(pqr_sha256_for_component("receptor")) << ','
              << csv_escape(pqr_sha256_for_component("ligand")) << ','
              << complex_solve.energy_kj_mol << ','
              << receptor_solve.energy_kj_mol << ','
              << ligand_solve.energy_kj_mol << ',' << delta_g_pol_kj << ','
              << cross_coulomb_kj << ',' << delta_g_elec_kj << ','
              << args_.int_diel << ',' << closure_error_max << ',' << args_.salt
              << ',' << args_.int_diel << ',' << args_.ext_diel << ','
              << args_.temp << ',' << csv_escape(args_.surface) << ','
              << csv_escape(binding_surface_generator) << ','
              << binding_surface_points << ','
              << complex_solve.tabi_surface_panels << ','
              << receptor_solve.tabi_surface_panels << ','
              << ligand_solve.tabi_surface_panels << ',' << args_.probe_radius
              << ',' << args_.scale << ',' << args_.margin << ','
              << (complex_solve.converged ? "ok" : "not_converged") << ','
              << (receptor_solve.converged ? "ok" : "not_converged") << ','
              << (ligand_solve.converged ? "ok" : "not_converged") << ','
              << sampled_audit_max << ',' << sampled_direct_true_residual_max
              << ',' << peak_memory_max_bytes << ','
              << (converged ? "ok" : "not_converged") << '\n';
}

void AnalyzerImpl::maybe_write_linear_forensic_points(
    const FrameResult &result, const std::vector<Atom> &atoms,
    const FieldSolve &solvent, const FieldSolve &vacuum,
    const ReactionSolve &solve) const {
  if (!barnase_forensic_enabled() ||
      (args_.parity_dump != "points" && args_.parity_dump != "all") ||
      solvent.shape.n == 0 || solvent.phi_grid.size() != solvent.shape.n ||
      vacuum.phi_grid.size() != solvent.shape.n)
    return;
  const GridShape &shape = solvent.shape;
  auto point_id = [&](double x, double y, double z) {
    auto to_index = [&](double value, double origin, std::size_t n) {
      if (n <= 2 || shape.h <= 0.0 || !std::isfinite(value))
        return std::size_t{1};
      const double scaled = std::floor((value - origin) / shape.h);
      return static_cast<std::size_t>(
          std::clamp(scaled, 1.0, static_cast<double>(n - 2)));
    };
    return idx3(to_index(x, shape.ox, shape.nx),
                to_index(y, shape.oy, shape.ny),
                to_index(z, shape.oz, shape.nz), shape.nx, shape.ny);
  };
  std::vector<std::pair<std::string, std::size_t>> rows;
  std::vector<std::pair<double, std::size_t>> ranked_atoms;
  ranked_atoms.reserve(atoms.size());
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const double contribution = i < solve.atom_reaction_kj_mol.size()
                                    ? std::abs(solve.atom_reaction_kj_mol[i])
                                    : std::abs(atoms[i].q);
    ranked_atoms.push_back({contribution, i});
  }
  std::sort(ranked_atoms.begin(), ranked_atoms.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
  for (std::size_t rank = 0;
       rank < std::min<std::size_t>(16, ranked_atoms.size()); ++rank) {
    const Atom &atom = atoms[ranked_atoms[rank].second];
    rows.push_back({"top_atom_energy_" + std::to_string(rank + 1),
                    point_id(atom.x, atom.y, atom.z)});
  }
  std::vector<std::pair<double, std::size_t>> top_rho;
  top_rho.reserve(16);
  auto consider_rho = [&](std::size_t id) {
    const double value =
        grid_.rho.size() == shape.n ? std::abs(grid_.rho[id]) : 0.0;
    if (value <= 0.0)
      return;
    if (top_rho.size() < 16) {
      top_rho.push_back({value, id});
      return;
    }
    auto min_it = std::min_element(top_rho.begin(), top_rho.end());
    if (min_it != top_rho.end() && value > min_it->first)
      *min_it = {value, id};
  };
  if (grid_.rho.size() == shape.n) {
    for (std::size_t id = 0; id < shape.n; ++id)
      consider_rho(id);
  }
  std::sort(top_rho.begin(), top_rho.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
  for (std::size_t rank = 0; rank < top_rho.size(); ++rank)
    rows.push_back(
        {"top_rho_" + std::to_string(rank + 1), top_rho[rank].second});

  auto add_surface = [&](const std::string &label, std::size_t id) {
    if (rows.size() >= 48)
      return;
    rows.push_back({label, id});
  };
  if (grid_.inside.size() == shape.n) {
    const std::size_t xy = shape.nx * shape.ny;
    std::size_t added = 0;
    for (std::size_t k = 1; k + 1 < shape.nz && added < 16; ++k) {
      for (std::size_t j = 1; j + 1 < shape.ny && added < 16; ++j) {
        for (std::size_t i = 1; i + 1 < shape.nx && added < 16; ++i) {
          const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
          const bool crossing =
              grid_.inside[id] != grid_.inside[id + 1] ||
              grid_.inside[id] != grid_.inside[id + shape.nx] ||
              grid_.inside[id] != grid_.inside[id + xy];
          if (crossing) {
            add_surface("surface_cross_" + std::to_string(++added), id);
          }
        }
      }
    }
  }
  if (rows.empty())
    return;

  std::ofstream out;
  if (!parity_points_header_written_) {
    out.open("pbe_parity_points.csv", std::ios::out | std::ios::trunc);
    if (!out.is_open())
      throw std::runtime_error("PBE: failed to open pbe_parity_points.csv.");
    out << "trajectory_id,frame_index,global_index,point_kind,rank,i,j,k,"
           "x_A,y_A,z_A,rho,eps_sum,screen,phi,source,rhs,F,Jdiag,Jdelta,"
           "residual\n";
    parity_points_header_written_ = true;
  } else {
    out.open("pbe_parity_points.csv", std::ios::out | std::ios::app);
    if (!out.is_open())
      throw std::runtime_error("PBE: failed to append pbe_parity_points.csv.");
  }
  out << std::setprecision(12);

  const double rhs = effective_rhs_scale(shape);
  const std::size_t xy = shape.nx * shape.ny;
  std::size_t rank = 0;
  for (const auto &[kind, id] : rows) {
    if (id >= shape.n)
      continue;
    const std::size_t k = id / xy;
    const std::size_t rem = id - k * xy;
    const std::size_t j = rem / shape.nx;
    const std::size_t i = rem - j * shape.nx;
    const double eps_sum =
        grid_.denom.size() == shape.n && grid_.screen.size() == shape.n
            ? std::max(1.0e-12, grid_.denom[id] - grid_.screen[id])
            : 0.0;
    const double screen =
        grid_.screen.size() == shape.n ? grid_.screen[id] : 0.0;
    const double rho = grid_.rho.size() == shape.n ? grid_.rho[id] : 0.0;
    const double solvent_phi = solvent.phi_grid[id];
    const double vacuum_phi = vacuum.phi_grid[id];
    out << result.trajectory_id << ',' << result.frame_index << ','
        << result.global_index << ',' << csv_escape(kind) << ',' << ++rank
        << ',' << i << ',' << j << ',' << k << ','
        << shape.ox + static_cast<double>(i) * shape.h << ','
        << shape.oy + static_cast<double>(j) * shape.h << ','
        << shape.oz + static_cast<double>(k) * shape.h << ',' << rho << ','
        << eps_sum << ',' << screen << ',' << solvent_phi << ',' << vacuum_phi
        << ',' << rhs * rho << ',' << (solvent_phi - vacuum_phi) << ','
        << eps_sum << ",0,0\n";
  }
}

bool AnalyzerImpl::parity_map_dump_enabled() const noexcept {
  return !args_.test_pqr.empty() && args_.parity_map_dump != "off";
}

std::vector<AnalyzerImpl::MapSamplePoint>
AnalyzerImpl::collect_map_sample_points(const GridShape &shape,
                                        const std::vector<Atom> &atoms,
                                        const ReactionSolve &solve) const {
  std::vector<MapSamplePoint> rows;
  rows.reserve(128);
  std::vector<unsigned char> used(shape.n, 0U);
  auto to_index = [&](double value, double origin, std::size_t n) {
    if (n <= 2 || shape.h <= 0.0 || !std::isfinite(value))
      return std::size_t{1};
    const double scaled = std::floor((value - origin) / shape.h);
    return static_cast<std::size_t>(
        std::clamp(scaled, 1.0, static_cast<double>(n - 2)));
  };
  auto add = [&](std::string kind, std::size_t id, double hint) {
    if (id >= shape.n || used[id])
      return;
    used[id] = 1U;
    rows.push_back({std::move(kind), rows.size() + 1, id, hint});
  };
  auto atom_id = [&](const Atom &atom) {
    return idx3(to_index(atom.x, shape.ox, shape.nx),
                to_index(atom.y, shape.oy, shape.ny),
                to_index(atom.z, shape.oz, shape.nz), shape.nx, shape.ny);
  };

  add("center",
      idx3(shape.nx / 2, shape.ny / 2, shape.nz / 2, shape.nx, shape.ny), 0.0);
  add("quartile_low",
      idx3(std::max<std::size_t>(1, shape.nx / 4),
           std::max<std::size_t>(1, shape.ny / 4),
           std::max<std::size_t>(1, shape.nz / 4), shape.nx, shape.ny),
      0.0);
  add("quartile_high",
      idx3(std::min<std::size_t>(shape.nx - 2, (shape.nx * 3) / 4),
           std::min<std::size_t>(shape.ny - 2, (shape.ny * 3) / 4),
           std::min<std::size_t>(shape.nz - 2, (shape.nz * 3) / 4), shape.nx,
           shape.ny),
      0.0);

  std::vector<std::pair<double, std::size_t>> ranked_atoms;
  ranked_atoms.reserve(atoms.size());
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const double hint = i < solve.atom_reaction_kj_mol.size()
                            ? std::abs(solve.atom_reaction_kj_mol[i])
                            : std::abs(atoms[i].q);
    ranked_atoms.push_back({hint, i});
  }
  std::sort(ranked_atoms.begin(), ranked_atoms.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
  for (std::size_t rank = 0;
       rank < std::min<std::size_t>(32, ranked_atoms.size()); ++rank) {
    const std::size_t atom_index = ranked_atoms[rank].second;
    add("charge_site_" + std::to_string(rank + 1), atom_id(atoms[atom_index]),
        ranked_atoms[rank].first);
  }

  std::vector<std::pair<double, std::size_t>> top_rho;
  auto consider_top = [&](std::vector<std::pair<double, std::size_t>> &top,
                          double value, std::size_t id, std::size_t limit) {
    if (value <= 0.0)
      return;
    if (top.size() < limit) {
      top.push_back({value, id});
      return;
    }
    auto min_it = std::min_element(top.begin(), top.end());
    if (min_it != top.end() && value > min_it->first)
      *min_it = {value, id};
  };
  if (grid_.rho.size() == shape.n) {
    for (std::size_t id = 0; id < shape.n; ++id)
      consider_top(top_rho, std::abs(grid_.rho[id]), id, 32);
  }
  std::sort(top_rho.begin(), top_rho.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });
  for (std::size_t rank = 0; rank < top_rho.size(); ++rank)
    add("top_rho_" + std::to_string(rank + 1), top_rho[rank].second,
        top_rho[rank].first);

  if (grid_.inside.size() == shape.n) {
    const std::size_t xy = shape.nx * shape.ny;
    std::size_t added = 0;
    for (std::size_t k = 1; k + 1 < shape.nz && added < 32; ++k) {
      for (std::size_t j = 1; j + 1 < shape.ny && added < 32; ++j) {
        for (std::size_t i = 1; i + 1 < shape.nx && added < 32; ++i) {
          const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
          const bool crossing =
              grid_.inside[id] != grid_.inside[id + 1] ||
              grid_.inside[id] != grid_.inside[id + shape.nx] ||
              grid_.inside[id] != grid_.inside[id + xy] ||
              grid_.inside[id] != grid_.inside[id - 1] ||
              grid_.inside[id] != grid_.inside[id - shape.nx] ||
              grid_.inside[id] != grid_.inside[id - xy];
          if (crossing)
            add("surface_cross_" + std::to_string(++added), id, 0.0);
        }
      }
    }
  }
  for (std::size_t i = 0; i < rows.size(); ++i)
    rows[i].rank = i + 1;
  return rows;
}

void AnalyzerImpl::maybe_write_parity_map_dumps(
    const FrameResult &result, const std::vector<Atom> &atoms,
    const FieldSolve &solvent, const FieldSolve &vacuum,
    const ReactionSolve &solve) const {
  if (!parity_map_dump_enabled() || solvent.shape.n == 0 ||
      solvent.phi_grid.size() != solvent.shape.n ||
      vacuum.phi_grid.size() != solvent.shape.n)
    return;
  const GridShape &shape = solvent.shape;
  const bool full = args_.parity_map_dump == "full";
  const auto samples = collect_map_sample_points(shape, atoms, solve);
  auto coords = [&](std::size_t id, std::size_t &i, std::size_t &j,
                    std::size_t &k) {
    const std::size_t xy = shape.nx * shape.ny;
    k = id / xy;
    const std::size_t rem = id - k * xy;
    j = rem / shape.nx;
    i = rem - j * shape.nx;
  };
  auto eps_sum_at = [&](std::size_t id) {
    if (solvent.eps_sum_grid.size() == shape.n)
      return solvent.eps_sum_grid[id];
    return grid_.denom.size() == shape.n && grid_.screen.size() == shape.n
               ? std::max(1.0e-12, grid_.denom[id] - grid_.screen[id])
               : 0.0;
  };
  auto screen_at = [&](std::size_t id) {
    if (solvent.screen_grid.size() == shape.n)
      return solvent.screen_grid[id];
    return grid_.screen.size() == shape.n ? grid_.screen[id] : 0.0;
  };
  auto rho_at = [&](std::size_t id) {
    if (solvent.rho_grid.size() == shape.n)
      return solvent.rho_grid[id];
    return grid_.rho.size() == shape.n ? grid_.rho[id] : 0.0;
  };
  auto inside_at = [&](std::size_t id) {
    if (solvent.inside_grid.size() == shape.n)
      return solvent.inside_grid[id] ? 1 : 0;
    return grid_.inside.size() == shape.n && grid_.inside[id] ? 1 : 0;
  };
  auto eps_x_at = [&](std::size_t id) {
    if (solvent.eps_x_grid.size() == shape.n)
      return solvent.eps_x_grid[id];
    return grid_.eps_x.size() == shape.n ? grid_.eps_x[id] : 0.0;
  };
  auto eps_y_at = [&](std::size_t id) {
    if (solvent.eps_y_grid.size() == shape.n)
      return solvent.eps_y_grid[id];
    return grid_.eps_y.size() == shape.n ? grid_.eps_y[id] : 0.0;
  };
  auto eps_z_at = [&](std::size_t id) {
    if (solvent.eps_z_grid.size() == shape.n)
      return solvent.eps_z_grid[id];
    return grid_.eps_z.size() == shape.n ? grid_.eps_z[id] : 0.0;
  };

  {
    std::ofstream out;
    if (!field_samples_header_written_) {
      out.open("pbe_field_samples.csv", std::ios::out | std::ios::trunc);
      if (!out.is_open())
        throw std::runtime_error("PBE: failed to open pbe_field_samples.csv.");
      out << "trajectory_id,frame_index,global_index,point_kind,rank,i,j,k,"
             "x_A,y_A,z_A,rho,inside,eps_sum,eps_x,eps_y,eps_z,screen,solvent_"
             "phi,"
             "vacuum_phi,reaction_phi,atom_energy_hint_kJmol,is_boundary,"
             "field_units,solvent_phi_raw,solvent_phi_native,"
             "vacuum_phi_raw,vacuum_phi_native,reaction_phi_native\n";
      field_samples_header_written_ = true;
    } else {
      out.open("pbe_field_samples.csv", std::ios::out | std::ios::app);
      if (!out.is_open())
        throw std::runtime_error(
            "PBE: failed to append pbe_field_samples.csv.");
    }
    out << std::setprecision(12);
    auto write_field = [&](const std::string &kind, std::size_t rank,
                           std::size_t id, double hint) {
      if (id >= shape.n)
        return;
      std::size_t i = 0, j = 0, k = 0;
      coords(id, i, j, k);
      const double to_native = field_to_native_factor();
      const double to_raw = delphi_field_units_active() ? 1.0 : delphi_epkt();
      const double solvent_raw = solvent.phi_grid[id] * to_raw;
      const double vacuum_raw = vacuum.phi_grid[id] * to_raw;
      const double solvent_native = solvent.phi_grid[id] * to_native;
      const double vacuum_native = vacuum.phi_grid[id] * to_native;
      out << result.trajectory_id << ',' << result.frame_index << ','
          << result.global_index << ',' << csv_escape(kind) << ',' << rank
          << ',' << i << ',' << j << ',' << k << ','
          << shape.ox + static_cast<double>(i) * shape.h << ','
          << shape.oy + static_cast<double>(j) * shape.h << ','
          << shape.oz + static_cast<double>(k) * shape.h << ',' << rho_at(id)
          << ',' << inside_at(id) << ',' << eps_sum_at(id) << ','
          << eps_x_at(id) << ',' << eps_y_at(id) << ',' << eps_z_at(id) << ','
          << screen_at(id) << ',' << solvent.phi_grid[id] << ','
          << vacuum.phi_grid[id] << ','
          << solvent.phi_grid[id] - vacuum.phi_grid[id] << ',' << hint << ','
          << (is_boundary(i, j, k, shape) ? 1 : 0) << ','
          << csv_escape(args_.test_field_units) << ',' << solvent_raw << ','
          << solvent_native << ',' << vacuum_raw << ',' << vacuum_native << ','
          << (solvent_native - vacuum_native) << '\n';
    };
    if (full) {
      for (std::size_t id = 0; id < shape.n; ++id)
        write_field("full_grid", id + 1, id, 0.0);
    } else {
      for (const MapSamplePoint &sample : samples)
        write_field(sample.kind, sample.rank, sample.id, sample.hint);
    }
  }

  {
    std::ofstream out;
    if (!boundary_samples_header_written_) {
      out.open("pbe_boundary_samples.csv", std::ios::out | std::ios::trunc);
      if (!out.is_open())
        throw std::runtime_error(
            "PBE: failed to open pbe_boundary_samples.csv.");
      out << "trajectory_id,frame_index,global_index,boundary_kind,rank,i,j,k,"
             "x_A,y_A,z_A,solvent_phi,vacuum_phi,rho,inside,eps_sum,"
             "eps_x,eps_y,eps_z,screen,field_units,solvent_phi_raw,"
             "solvent_phi_native,vacuum_phi_raw,vacuum_phi_native\n";
      boundary_samples_header_written_ = true;
    } else {
      out.open("pbe_boundary_samples.csv", std::ios::out | std::ios::app);
      if (!out.is_open())
        throw std::runtime_error(
            "PBE: failed to append pbe_boundary_samples.csv.");
    }
    out << std::setprecision(12);
    auto write_boundary = [&](const std::string &kind, std::size_t rank,
                              std::size_t id) {
      std::size_t i = 0, j = 0, k = 0;
      coords(id, i, j, k);
      if (!is_boundary(i, j, k, shape))
        return;
      const double to_native = field_to_native_factor();
      const double to_raw = delphi_field_units_active() ? 1.0 : delphi_epkt();
      out << result.trajectory_id << ',' << result.frame_index << ','
          << result.global_index << ',' << csv_escape(kind) << ',' << rank
          << ',' << i << ',' << j << ',' << k << ','
          << shape.ox + static_cast<double>(i) * shape.h << ','
          << shape.oy + static_cast<double>(j) * shape.h << ','
          << shape.oz + static_cast<double>(k) * shape.h << ','
          << solvent.phi_grid[id] << ',' << vacuum.phi_grid[id] << ','
          << rho_at(id) << ',' << inside_at(id) << ',' << eps_sum_at(id) << ','
          << eps_x_at(id) << ',' << eps_y_at(id) << ',' << eps_z_at(id) << ','
          << screen_at(id) << ',' << csv_escape(args_.test_field_units) << ','
          << solvent.phi_grid[id] * to_raw << ','
          << solvent.phi_grid[id] * to_native << ','
          << vacuum.phi_grid[id] * to_raw << ','
          << vacuum.phi_grid[id] * to_native << '\n';
    };
    if (full) {
      std::size_t rank = 0;
      for (std::size_t k = 0; k < shape.nz; ++k) {
        for (std::size_t j = 0; j < shape.ny; ++j) {
          for (std::size_t i = 0; i < shape.nx; ++i) {
            if (is_boundary(i, j, k, shape))
              write_boundary("full_boundary", ++rank,
                             idx3(i, j, k, shape.nx, shape.ny));
          }
        }
      }
    } else {
      std::vector<std::size_t> ids = {
          idx3(0, 0, 0, shape.nx, shape.ny),
          idx3(shape.nx - 1, 0, 0, shape.nx, shape.ny),
          idx3(0, shape.ny - 1, 0, shape.nx, shape.ny),
          idx3(0, 0, shape.nz - 1, shape.nx, shape.ny),
          idx3(shape.nx / 2, shape.ny / 2, 0, shape.nx, shape.ny),
          idx3(shape.nx / 2, shape.ny / 2, shape.nz - 1, shape.nx, shape.ny),
          idx3(shape.nx / 2, 0, shape.nz / 2, shape.nx, shape.ny),
          idx3(shape.nx / 2, shape.ny - 1, shape.nz / 2, shape.nx, shape.ny),
          idx3(0, shape.ny / 2, shape.nz / 2, shape.nx, shape.ny),
          idx3(shape.nx - 1, shape.ny / 2, shape.nz / 2, shape.nx, shape.ny)};
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      for (std::size_t rank = 0; rank < ids.size(); ++rank)
        write_boundary("sample_boundary", rank + 1, ids[rank]);
    }
  }

  {
    std::ofstream out;
    if (!operator_stage_header_written_) {
      out.open("pbe_operator_stage.csv", std::ios::out | std::ios::trunc);
      if (!out.is_open())
        throw std::runtime_error("PBE: failed to open pbe_operator_stage.csv.");
      out << "trajectory_id,frame_index,global_index,point_kind,rank,i,j,k,"
             "rhs_scale_effective,rho,denom,screen,eps_x_plus,eps_x_minus,"
             "eps_y_plus,eps_y_minus,eps_z_plus,eps_z_minus,neighbor_sum,"
             "rhs,F,residual_abs,phi,phi_native,field_units,"
             "rhs_delta_vs_oracle,screen_delta_vs_oracle,"
             "denom_delta_vs_oracle,fixed_boundary_used,"
             "operator_convention\n";
      operator_stage_header_written_ = true;
    } else {
      out.open("pbe_operator_stage.csv", std::ios::out | std::ios::app);
      if (!out.is_open())
        throw std::runtime_error(
            "PBE: failed to append pbe_operator_stage.csv.");
    }
    out << std::setprecision(12);
    const double rhs = effective_rhs_scale(shape);
    const double to_native = field_to_native_factor();
    struct OperatorOracleRow {
      bool rhs_present{false};
      bool screen_present{false};
      bool denom_present{false};
      double rhs{0.0};
      double screen{0.0};
      double denom{0.0};
    };
    std::unordered_map<std::size_t, OperatorOracleRow> operator_oracle;
    {
      std::ifstream oracle("pbe_oracle_operator_stage.csv");
      if (oracle.is_open()) {
        std::string line;
        if (std::getline(oracle, line)) {
          const auto header = csv_header_map(split_csv_simple(line));
          const std::string fixture_filter = read_oracle_fixture_filter();
          while (std::getline(oracle, line)) {
            if (trim_copy(line).empty())
              continue;
            const auto fields = split_csv_simple(line);
            if (!oracle_fixture_matches(fields, header, fixture_filter))
              continue;
            double vacuum_flag = 0.0;
            if (csv_double(fields, header, {"vacuum"}, vacuum_flag) &&
                std::llround(vacuum_flag) != 0)
              continue;
            std::size_t i = 0, j = 0, k = 0;
            if (!csv_index(fields, header, "i", i) ||
                !csv_index(fields, header, "j", j) ||
                !csv_index(fields, header, "k", k) || i >= shape.nx ||
                j >= shape.ny || k >= shape.nz)
              continue;
            OperatorOracleRow row;
            row.rhs_present = csv_double(
                fields, header,
                {"charge_map_scaled", "rhs", "rhs_value", "charge_density"},
                row.rhs);
            row.screen_present = csv_double(
                fields, header, {"screen", "screen_term"}, row.screen);
            row.denom_present = csv_double(
                fields, header,
                {"denom", "eps_sum_plus_screen", "epsilon_sum_plus_screen"},
                row.denom);
            operator_oracle[idx3(i, j, k, shape.nx, shape.ny)] = row;
          }
        }
      }
    }
    auto write_operator = [&](const std::string &kind, std::size_t rank,
                              std::size_t id) {
      if (id >= shape.n)
        return;
      std::size_t i = 0, j = 0, k = 0;
      coords(id, i, j, k);
      if (is_boundary(i, j, k, shape) || i == 0 || j == 0 || k == 0 ||
          i + 1 >= shape.nx || j + 1 >= shape.ny || k + 1 >= shape.nz)
        return;
      const std::size_t xy = shape.nx * shape.ny;
      const double eps_x_plus = eps_x_at(id);
      const double eps_x_minus = eps_x_at(id - 1);
      const double eps_y_plus = eps_y_at(id);
      const double eps_y_minus = eps_y_at(id - shape.nx);
      const double eps_z_plus = eps_z_at(id);
      const double eps_z_minus = eps_z_at(id - xy);
      const double denom = eps_x_plus + eps_x_minus + eps_y_plus + eps_y_minus +
                           eps_z_plus + eps_z_minus + screen_at(id);
      const double neighbor_sum =
          eps_x_plus * solvent.phi_grid[id + 1] +
          eps_x_minus * solvent.phi_grid[id - 1] +
          eps_y_plus * solvent.phi_grid[id + shape.nx] +
          eps_y_minus * solvent.phi_grid[id - shape.nx] +
          eps_z_plus * solvent.phi_grid[id + xy] +
          eps_z_minus * solvent.phi_grid[id - xy];
      const double rhs_value = rhs * rho_at(id);
      const double residual =
          denom * solvent.phi_grid[id] - neighbor_sum - rhs_value;
      const auto oracle_it = operator_oracle.find(id);
      const bool has_oracle = oracle_it != operator_oracle.end();
      const bool fixed_boundary_used =
          is_boundary(i + 1, j, k, shape) || is_boundary(i - 1, j, k, shape) ||
          is_boundary(i, j + 1, k, shape) || is_boundary(i, j - 1, k, shape) ||
          is_boundary(i, j, k + 1, shape) || is_boundary(i, j, k - 1, shape);
      out << result.trajectory_id << ',' << result.frame_index << ','
          << result.global_index << ',' << csv_escape(kind) << ',' << rank
          << ',' << i << ',' << j << ',' << k << ',' << rhs << ',' << rho_at(id)
          << ',' << denom << ',' << screen_at(id) << ',' << eps_x_plus << ','
          << eps_x_minus << ',' << eps_y_plus << ',' << eps_y_minus << ','
          << eps_z_plus << ',' << eps_z_minus << ',' << neighbor_sum << ','
          << rhs_value << ',' << residual << ',' << std::abs(residual) << ','
          << solvent.phi_grid[id] << ',' << solvent.phi_grid[id] * to_native
          << ',' << csv_escape(args_.test_field_units);
      if (has_oracle && oracle_it->second.rhs_present)
        out << ',' << (rhs_value - oracle_it->second.rhs);
      else
        out << ",NA";
      if (has_oracle && oracle_it->second.screen_present)
        out << ',' << (screen_at(id) - oracle_it->second.screen);
      else
        out << ",NA";
      if (has_oracle && oracle_it->second.denom_present)
        out << ',' << (denom - oracle_it->second.denom);
      else
        out << ",NA";
      out << ',' << (fixed_boundary_used ? 1 : 0) << ','
          << csv_escape(args_.test_operator_convention) << '\n';
    };
    if (full) {
      std::size_t rank = 0;
      for (std::size_t id = 0; id < shape.n; ++id)
        write_operator("full_operator", ++rank, id);
    } else {
      for (const MapSamplePoint &sample : samples)
        write_operator(sample.kind, sample.rank, sample.id);
    }
  }

  {
    std::ofstream out;
    if (!charge_grid_audit_header_written_) {
      out.open("pbe_charge_grid_audit.csv", std::ios::out | std::ios::trunc);
      if (!out.is_open())
        throw std::runtime_error(
            "PBE: failed to open pbe_charge_grid_audit.csv.");
      out << "trajectory_id,frame_index,global_index,charge_kind,rank,i,j,k,"
             "x_A,y_A,z_A,rho,abs_rho,inside,eps_sum,eps_x,eps_y,eps_z,"
             "screen\n";
      charge_grid_audit_header_written_ = true;
    } else {
      out.open("pbe_charge_grid_audit.csv", std::ios::out | std::ios::app);
      if (!out.is_open())
        throw std::runtime_error(
            "PBE: failed to append pbe_charge_grid_audit.csv.");
    }
    out << std::setprecision(12);
    auto write_charge = [&](const std::string &kind, std::size_t rank,
                            std::size_t id) {
      const double rho = rho_at(id);
      if (rho == 0.0 && full)
        return;
      std::size_t i = 0, j = 0, k = 0;
      coords(id, i, j, k);
      out << result.trajectory_id << ',' << result.frame_index << ','
          << result.global_index << ',' << csv_escape(kind) << ',' << rank
          << ',' << i << ',' << j << ',' << k << ','
          << shape.ox + static_cast<double>(i) * shape.h << ','
          << shape.oy + static_cast<double>(j) * shape.h << ','
          << shape.oz + static_cast<double>(k) * shape.h << ',' << rho << ','
          << std::abs(rho) << ',' << inside_at(id) << ',' << eps_sum_at(id)
          << ',' << eps_x_at(id) << ',' << eps_y_at(id) << ',' << eps_z_at(id)
          << ',' << screen_at(id) << '\n';
    };
    if (full) {
      std::size_t rank = 0;
      for (std::size_t id = 0; id < shape.n; ++id) {
        if (rho_at(id) != 0.0)
          write_charge("full_nonzero_rho", ++rank, id);
      }
    } else {
      std::size_t rank = 0;
      for (const MapSamplePoint &sample : samples) {
        if (std::abs(rho_at(sample.id)) > 0.0)
          write_charge("sample_rho", ++rank, sample.id);
      }
    }
  }
}

void AnalyzerImpl::maybe_write_parity_points(
    const FrameResult &result, const GridShape &shape,
    const AlignedDoubleVector &source, const AlignedDoubleVector &residual,
    const AlignedDoubleVector &delta, double alpha, double clip,
    double screen_scale) const {
  if (args_.test_pqr.empty() ||
      (args_.parity_dump != "points" && args_.parity_dump != "all") ||
      shape.n == 0 || source.size() != shape.n || residual.size() != shape.n ||
      delta.size() != shape.n || grid_.phi.size() != shape.n)
    return;

  std::vector<std::pair<double, std::size_t>> top;
  top.reserve(16);
  auto consider_top = [&](std::size_t id) {
    const double value = std::abs(residual[id]);
    if (top.size() < 16) {
      top.push_back({value, id});
      return;
    }
    auto min_it = std::min_element(top.begin(), top.end());
    if (min_it != top.end() && value > min_it->first)
      *min_it = {value, id};
  };
  for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
    for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
      for (std::size_t i = 1; i + 1 < shape.nx; ++i)
        consider_top(idx3(i, j, k, shape.nx, shape.ny));
    }
  }
  std::sort(top.begin(), top.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  std::vector<std::pair<std::string, std::size_t>> rows;
  for (std::size_t rank = 0; rank < top.size(); ++rank)
    rows.push_back(
        {"top_residual_" + std::to_string(rank + 1), top[rank].second});
  auto add_fixed = [&](const std::string &label, std::size_t i, std::size_t j,
                       std::size_t k) {
    i = std::clamp<std::size_t>(i, 1, shape.nx > 1 ? shape.nx - 2 : 1);
    j = std::clamp<std::size_t>(j, 1, shape.ny > 1 ? shape.ny - 2 : 1);
    k = std::clamp<std::size_t>(k, 1, shape.nz > 1 ? shape.nz - 2 : 1);
    const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
    const auto duplicate =
        std::find_if(rows.begin(), rows.end(), [&](const auto &row) {
          return row.first == label || row.second == id;
        });
    if (duplicate == rows.end())
      rows.push_back({label, id});
  };
  add_fixed("center", shape.nx / 2, shape.ny / 2, shape.nz / 2);
  add_fixed("quartile_low", shape.nx / 4, shape.ny / 4, shape.nz / 4);
  add_fixed("quartile_high", (shape.nx * 3) / 4, (shape.ny * 3) / 4,
            (shape.nz * 3) / 4);

  std::ofstream out;
  if (!parity_points_header_written_) {
    out.open("pbe_parity_points.csv", std::ios::out | std::ios::trunc);
    if (!out.is_open())
      throw std::runtime_error("PBE: failed to open pbe_parity_points.csv.");
    out << "trajectory_id,frame_index,global_index,point_kind,rank,i,j,k,"
           "x_A,y_A,z_A,rho,eps_sum,screen,phi,source,rhs,F,Jdiag,Jdelta,"
           "residual\n";
    parity_points_header_written_ = true;
  } else {
    out.open("pbe_parity_points.csv", std::ios::out | std::ios::app);
    if (!out.is_open())
      throw std::runtime_error("PBE: failed to append pbe_parity_points.csv.");
  }

  const double rhs = nonlinear_rhs_scale(shape, alpha);
  const std::size_t xy = shape.nx * shape.ny;
  std::size_t rank = 0;
  for (const auto &[kind, id] : rows) {
    const std::size_t k = id / xy;
    const std::size_t rem = id - k * xy;
    const std::size_t j = rem / shape.nx;
    const std::size_t i = rem - j * shape.nx;
    const double eps_sum =
        std::max(1.0e-12, grid_.denom[id] - grid_.screen[id]);
    const double neighbor_sum =
        grid_.eps_x[id] * grid_.phi[id + 1] +
        grid_.eps_x[id - 1] * grid_.phi[id - 1] +
        grid_.eps_y[id] * grid_.phi[id + shape.nx] +
        grid_.eps_y[id - shape.nx] * grid_.phi[id - shape.nx] +
        grid_.eps_z[id] * grid_.phi[id + xy] +
        grid_.eps_z[id - xy] * grid_.phi[id - xy];
    const double f_value = eps_sum * grid_.phi[id] - neighbor_sum +
                           (grid_.screen[id] * screen_scale) * source[id] -
                           rhs * grid_.rho[id];
    const double jdiag = std::max(
        1.0e-12,
        eps_sum + (grid_.screen[id] * screen_scale) *
                      nonlinear_source_derivative(grid_.phi[id], alpha, clip));
    const double neighbor_delta_sum =
        grid_.eps_x[id] * delta[id + 1] + grid_.eps_x[id - 1] * delta[id - 1] +
        grid_.eps_y[id] * delta[id + shape.nx] +
        grid_.eps_y[id - shape.nx] * delta[id - shape.nx] +
        grid_.eps_z[id] * delta[id + xy] +
        grid_.eps_z[id - xy] * delta[id - xy];
    const double jdelta = jdiag * delta[id] - neighbor_delta_sum;
    out << result.trajectory_id << ',' << result.frame_index << ','
        << result.global_index << ',' << csv_escape(kind) << ',' << ++rank
        << ',' << i << ',' << j << ',' << k << ','
        << shape.ox + static_cast<double>(i) * shape.h << ','
        << shape.oy + static_cast<double>(j) * shape.h << ','
        << shape.oz + static_cast<double>(k) * shape.h << ',' << grid_.rho[id]
        << ',' << eps_sum << ',' << grid_.screen[id] << ',' << grid_.phi[id]
        << ',' << source[id] << ',' << rhs * grid_.rho[id] << ',' << f_value
        << ',' << jdiag << ',' << jdelta << ',' << residual[id] << '\n';
  }
}


} // namespace TitanPBE
