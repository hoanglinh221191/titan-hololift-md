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

using analyzer_impl_detail::ProcessMemorySample;
using analyzer_impl_detail::elapsed_ms;
using analyzer_impl_detail::kKBKJPerMolK;
using analyzer_impl_detail::kKJToKcal;
using analyzer_impl_detail::kKcalToKJ;
using analyzer_impl_detail::sample_process_memory;
using analyzer_impl_detail::surface_uses_face_refinement;

void AnalyzerImpl::compute_stability_ensemble(const std::vector<Atom> &atoms,
                                              TitanExecutor &executor,
                                              const FrameResult &result) {
  last_stability_ensemble_ = StabilityEnsembleState{};
  if (args_.stability_mode != "ensemble" || atoms.empty() ||
      stability_probe_solve_active_)
    return;

  struct ProbeSolveScope {
    AnalyzerImpl &self;
    bool previous;
    explicit ProbeSolveScope(AnalyzerImpl &impl)
        : self(impl), previous(impl.stability_probe_solve_active_) {
      self.stability_probe_solve_active_ = true;
    }
    ~ProbeSolveScope() { self.stability_probe_solve_active_ = previous; }
  } probe_scope(*this);

  std::vector<double> energies;
  energies.reserve(7);
  energies.push_back(result.energy_kj_mol);
  constexpr double kJitterA = 1.0e-7;
  const std::array<std::array<double, 3>, 6> offsets{{
      {{kJitterA, 0.0, 0.0}},
      {{-kJitterA, 0.0, 0.0}},
      {{0.0, kJitterA, 0.0}},
      {{0.0, -kJitterA, 0.0}},
      {{0.0, 0.0, kJitterA}},
      {{0.0, 0.0, -kJitterA}},
  }};
  for (const auto &offset : offsets) {
    std::vector<Atom> shifted = atoms;
    for (Atom &atom : shifted) {
      atom.x += offset[0];
      atom.y += offset[1];
      atom.z += offset[2];
    }
    const ReactionSolve probe =
        solve_reaction_field(shifted, executor, nullptr);
    if (probe.converged && std::isfinite(probe.energy_kj_mol))
      energies.push_back(probe.energy_kj_mol);
  }
  if (energies.empty())
    return;
  const double sum = std::accumulate(energies.begin(), energies.end(), 0.0);
  const double mean = sum / static_cast<double>(energies.size());
  double m2 = 0.0;
  for (double value : energies) {
    const double delta = value - mean;
    m2 += delta * delta;
  }
  StabilityEnsembleState state;
  state.valid = true;
  state.samples = energies.size();
  state.mean_kj_mol = mean;
  state.stddev_kj_mol =
      energies.size() > 1
          ? std::sqrt(m2 / static_cast<double>(energies.size() - 1))
          : 0.0;
  const auto mm = std::minmax_element(energies.begin(), energies.end());
  state.min_kj_mol = *mm.first;
  state.max_kj_mol = *mm.second;
  last_stability_ensemble_ = state;
}

FrameResult AnalyzerImpl::consume_frame(
    const MOL &mol1, const Multi_MOL_parameters &params1, const MOL *mol2,
    const Multi_MOL_parameters *params2, std::size_t trajectory_id,
    std::size_t frame_index, std::size_t global_index, TitanExecutor &executor,
    std::fstream &log, const EnergyDecompositionProvider *energy_provider) {
  if (!enabled())
    return {};

  const auto frame_time_begin = std::chrono::steady_clock::now();
  const ProcessMemorySample frame_memory_before = sample_process_memory();
  TimingBuckets frame_timing;
  struct ActiveTimingScope {
    AnalyzerImpl &self;
    TimingBuckets *previous;
    ActiveTimingScope(AnalyzerImpl &impl, TimingBuckets *next)
        : self(impl), previous(impl.active_timing_) {
      self.active_timing_ = next;
    }
    ~ActiveTimingScope() { self.active_timing_ = previous; }
  } timing_scope(*this, pbe_timing_enabled() ? &frame_timing : nullptr);
  atoms_.clear();
  std::vector<Atom> mol1_atoms;
  std::vector<Atom> mol2_atoms;
  if (!args_.test_pqr.empty()) {
    collect_test_pqr_atoms(atoms_);
    if (args_.binding_cycle == "pqr-components") {
      collect_pqr_atoms(args_.binding_receptor_pqr, mol1_atoms);
      collect_pqr_atoms(args_.binding_ligand_pqr, mol2_atoms);
      validate_binding_component_union(atoms_, mol1_atoms, mol2_atoms);
    }
  } else {
    collect_atoms(mol1, params1, mol1_atoms, "mol1");
    atoms_ = mol1_atoms;
    if (mol2 != nullptr && params2 != nullptr && mol2->size() > 0) {
      collect_atoms(*mol2, *params2, mol2_atoms, "mol2");
      atoms_.insert(atoms_.end(), mol2_atoms.begin(), mol2_atoms.end());
    }
  }
  if (args_.dgsolv && mol2_atoms.empty() && args_.backend != "tabi" &&
      args_.test_pqr.empty()) {
    throw std::runtime_error(
        "PBE: -pbe-dgsolv requires a two-molecule input with mol2 atoms.");
  }

  FrameResult result;
  result.trajectory_id = trajectory_id;
  result.frame_index = frame_index + 1;
  result.global_index = global_index + 1;
  result.time_ps = mol1.time_of_snap;
  result.atom_count = atoms_.size();
  result.surface_effective = args_.surface;
  result.ion_access = args_.ion_access;
  result.ion_radius_A = args_.ion_radius;
  result.face_refine = surface_uses_face_refinement(args_.surface);
  result.pbe_backend = args_.backend;
  result.pbe_model = args_.model;
  result.pbe_biomodel = args_.biomodel;
  result.dielectric_model =
      uses_gaussian_model() ? std::string("gaussian") : args_.dielectric_model;
  result.nl_solver = args_.nl_solver;
  result.nl_inner_solver = args_.nl_inner;
  result.nl_mg_mode = args_.nl_multigrid;
  result.gaussian_exp = args_.gaussian_exp;
  result.gaussian_sigma = args_.gaussian_sigma;
  result.density_cutoff = args_.density_cutoff;
  result.surface_cutoff = args_.surface_cutoff == "auto"
                              ? args_.density_cutoff
                              : args_.surface_cutoff_value;
  result.gap_diel = args_.gap_diel;
  result.radius_convention = effective_radius_convention();
  result.boundary_mode = args_.boundary;
  result.test_boundary_convention = effective_boundary_convention();
  result.test_field_units = args_.test_field_units;
  result.test_operator_convention =
      (args_.test_operator_convention == "native" && args_.focus == "delphi")
          ? std::string("delphi")
          : args_.test_operator_convention;
  result.charge_assignment = args_.charge_assignment;
  for (const Atom &atom : atoms_)
    result.total_charge += atom.q;
  last_stability_ensemble_ = StabilityEnsembleState{};
  stability_hysteresis_cells_kept_frame_ = 0;
  stability_hysteresis_cells_released_frame_ = 0;
  add_timing_ms(TimingBucket::Setup,
                elapsed_ms(frame_time_begin, std::chrono::steady_clock::now()));

  if (args_.backend == "tabi") {
    const auto tabi_time_begin = std::chrono::steady_clock::now();
    const auto tabi_complex_time_begin = std::chrono::steady_clock::now();
    const ReactionSolve tabi_solve =
        solve_tabi_reaction_field(atoms_, executor);
    const auto tabi_complex_time_end = std::chrono::steady_clock::now();
    ReactionSolve tabi_receptor_solve;
    ReactionSolve tabi_ligand_solve;
    const bool binding_cycle = args_.binding_cycle == "pqr-components";
    const bool native_pbsa_cycle = args_.pbsa;
    const bool endpoint_cycle = binding_cycle || native_pbsa_cycle;
    if (endpoint_cycle) {
      const auto receptor_time_begin = std::chrono::steady_clock::now();
      tabi_receptor_solve = solve_tabi_reaction_field(mol1_atoms, executor);
      const auto receptor_time_end = std::chrono::steady_clock::now();
      const auto ligand_time_begin = std::chrono::steady_clock::now();
      tabi_ligand_solve = solve_tabi_reaction_field(mol2_atoms, executor);
      const auto ligand_time_end = std::chrono::steady_clock::now();
      result.pbsa_time_receptor_ms =
          elapsed_ms(receptor_time_begin, receptor_time_end);
      result.pbsa_time_ligand_ms = elapsed_ms(ligand_time_begin, ligand_time_end);
    }
    const auto tabi_time_end = std::chrono::steady_clock::now();
    const double tabi_total_ms = elapsed_ms(tabi_time_begin, tabi_time_end);
    add_timing_ms(TimingBucket::Surface, tabi_solve.tabi_surface_build_ms);
    add_timing_ms(
        TimingBucket::Solver,
        std::max(0.0, tabi_total_ms - tabi_solve.tabi_surface_build_ms));

    result.nx = tabi_solve.shape.nx;
    result.ny = tabi_solve.shape.ny;
    result.nz = tabi_solve.shape.nz;
    result.grid_points = tabi_solve.shape.n;
    result.memory_bytes =
        std::max(tabi_solve.shape.memory_bytes, tabi_solve.peak_memory_bytes);
    result.grid_origin_x_A = tabi_solve.shape.ox;
    result.grid_origin_y_A = tabi_solve.shape.oy;
    result.grid_origin_z_A = tabi_solve.shape.oz;
    result.grid_spacing_A = tabi_solve.shape.h;
    result.grid_center_x_A = tabi_solve.shape.ox;
    result.grid_center_y_A = tabi_solve.shape.oy;
    result.grid_center_z_A = tabi_solve.shape.oz;
    result.solute_points = tabi_solve.solute_points;
    result.solvent_points = tabi_solve.solvent_points;
    result.ion_accessible_points = tabi_solve.ion_accessible_points;
    result.stern_excluded_points = tabi_solve.stern_excluded_points;
    result.tabi_surface_build_ms = tabi_solve.tabi_surface_build_ms;
    result.tabi_surface_panels = tabi_solve.tabi_surface_panels;
    result.tabi_surface_point_count_requested =
        tabi_solve.tabi_surface_point_count_requested;
    result.tabi_panel_density_per_a2 = tabi_solve.tabi_panel_density_per_a2;
    result.tabi_panel_budget_status = tabi_solve.tabi_panel_budget_status;
    result.tabi_charge_resolution = tabi_solve.tabi_charge_resolution;
    result.tabi_surface_exposed_points = tabi_solve.tabi_surface_exposed_points;
    result.tabi_surface_buried_points = tabi_solve.tabi_surface_buried_points;
    result.tabi_tree_build_ms = tabi_solve.tabi_tree_build_ms;
    result.tabi_tree_nodes = tabi_solve.tabi_tree_nodes;
    result.tabi_tree_leaf_count = tabi_solve.tabi_tree_leaf_count;
    result.tabi_near_interactions = tabi_solve.tabi_near_interactions;
    result.tabi_far_interactions = tabi_solve.tabi_far_interactions;
    result.tabi_matvec_mode = tabi_solve.tabi_matvec_mode;
    result.tabi_fmm_m2l_mode = tabi_solve.tabi_fmm_m2l_mode;
    result.tabi_fmm_m2l_expansion_order =
        tabi_solve.tabi_fmm_m2l_expansion_order;
    result.tabi_fmm_m2l_near_pairs = tabi_solve.tabi_fmm_m2l_near_pairs;
    result.tabi_fmm_m2l_far_pairs = tabi_solve.tabi_fmm_m2l_far_pairs;
    result.tabi_fmm_m2l_setup_ms = tabi_solve.tabi_fmm_m2l_setup_ms;
    result.tabi_fmm_m2l_apply_ms = tabi_solve.tabi_fmm_m2l_apply_ms;
    result.tabi_fmm_m2l_finite_outputs = tabi_solve.tabi_fmm_m2l_finite_outputs;
    result.tabi_bie_cartesian_full_fmm_active =
        tabi_solve.tabi_bie_cartesian_full_fmm_active;
    result.tabi_bie_cartesian_full_fmm_requested =
        tabi_solve.tabi_bie_cartesian_full_fmm_requested;
    result.tabi_bie_cartesian_full_fmm_effective =
        tabi_solve.tabi_bie_cartesian_full_fmm_effective;
    result.tabi_bie_cartesian_full_fmm_order =
        tabi_solve.tabi_bie_cartesian_full_fmm_order;
    result.tabi_bie_cartesian_full_fmm_leaf_size =
        tabi_solve.tabi_bie_cartesian_full_fmm_leaf_size;
    result.tabi_bie_cartesian_full_fmm_opening_theta =
        tabi_solve.tabi_bie_cartesian_full_fmm_opening_theta;
    result.tabi_bie_cartesian_full_fmm_tree_hash =
        tabi_solve.tabi_bie_cartesian_full_fmm_tree_hash;
    result.tabi_bie_cartesian_full_fmm_worklist_hash =
        tabi_solve.tabi_bie_cartesian_full_fmm_worklist_hash;
    result.tabi_bie_cartesian_full_fmm_policy_hash =
        tabi_solve.tabi_bie_cartesian_full_fmm_policy_hash;
    result.tabi_bie_cartesian_full_fmm_nodes =
        tabi_solve.tabi_bie_cartesian_full_fmm_nodes;
    result.tabi_bie_cartesian_full_fmm_far_pairs =
        tabi_solve.tabi_bie_cartesian_full_fmm_far_pairs;
    result.tabi_bie_cartesian_full_fmm_direct_pairs =
        tabi_solve.tabi_bie_cartesian_full_fmm_direct_pairs;
    result.tabi_bie_cartesian_full_fmm_direct_panel_pairs_expected =
        tabi_solve.tabi_bie_cartesian_full_fmm_direct_panel_pairs_expected;
    result.tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_last =
        tabi_solve.tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_last;
    result.tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_total =
        tabi_solve.tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_total;
    result.tabi_bie_cartesian_full_fmm_cache_entries =
        tabi_solve.tabi_bie_cartesian_full_fmm_cache_entries;
    result.tabi_bie_cartesian_full_fmm_cache_hits =
        tabi_solve.tabi_bie_cartesian_full_fmm_cache_hits;
    result.tabi_bie_cartesian_full_fmm_cache_bytes =
        tabi_solve.tabi_bie_cartesian_full_fmm_cache_bytes;
    result.tabi_bie_cartesian_full_fmm_plan_bytes =
        tabi_solve.tabi_bie_cartesian_full_fmm_plan_bytes;
    result.tabi_bie_cartesian_full_fmm_workspace_bytes =
        tabi_solve.tabi_bie_cartesian_full_fmm_workspace_bytes;
    result.tabi_bie_cartesian_full_fmm_tree_seconds =
        tabi_solve.tabi_bie_cartesian_full_fmm_tree_seconds;
    result.tabi_bie_cartesian_full_fmm_worklist_seconds =
        tabi_solve.tabi_bie_cartesian_full_fmm_worklist_seconds;
    result.tabi_bie_cartesian_full_fmm_cache_seconds =
        tabi_solve.tabi_bie_cartesian_full_fmm_cache_seconds;
    result.tabi_bie_cartesian_full_fmm_audit_seconds =
        tabi_solve.tabi_bie_cartesian_full_fmm_audit_seconds;
    result.tabi_bie_cartesian_full_fmm_p2m_seconds =
        tabi_solve.tabi_bie_cartesian_full_fmm_p2m_seconds;
    result.tabi_bie_cartesian_full_fmm_m2m_seconds =
        tabi_solve.tabi_bie_cartesian_full_fmm_m2m_seconds;
    result.tabi_bie_cartesian_full_fmm_m2l_seconds =
        tabi_solve.tabi_bie_cartesian_full_fmm_m2l_seconds;
    result.tabi_bie_cartesian_full_fmm_l2l_seconds =
        tabi_solve.tabi_bie_cartesian_full_fmm_l2l_seconds;
    result.tabi_bie_cartesian_full_fmm_l2p_seconds =
        tabi_solve.tabi_bie_cartesian_full_fmm_l2p_seconds;
    result.tabi_bie_cartesian_full_fmm_direct_seconds =
        tabi_solve.tabi_bie_cartesian_full_fmm_direct_seconds;
    result.tabi_bie_cartesian_full_fmm_m2m_parity_error =
        tabi_solve.tabi_bie_cartesian_full_fmm_m2m_parity_error;
    result.tabi_bie_cartesian_full_fmm_l2l_parity_error =
        tabi_solve.tabi_bie_cartesian_full_fmm_l2l_parity_error;
    result.tabi_fmm_local_gmres_active = tabi_solve.tabi_fmm_local_gmres_active;
    result.tabi_fmm_local_prepare_seconds =
        tabi_solve.tabi_fmm_local_prepare_seconds;
    result.tabi_fmm_local_plan_build_count =
        tabi_solve.tabi_fmm_local_plan_build_count;
    result.tabi_fmm_local_cache_build_count =
        tabi_solve.tabi_fmm_local_cache_build_count;
    result.tabi_fmm_local_gmres_matvec_calls =
        tabi_solve.tabi_fmm_local_gmres_matvec_calls;
    result.tabi_fmm_local_gmres_apply_seconds_total =
        tabi_solve.tabi_fmm_local_gmres_apply_seconds_total;
    result.tabi_fmm_local_gmres_apply_seconds_mean =
        tabi_solve.tabi_fmm_local_gmres_apply_seconds_mean;
    result.tabi_fmm_local_moment_seconds_total =
        tabi_solve.tabi_fmm_local_moment_seconds_total;
    result.tabi_fmm_local_m2l_seconds_total =
        tabi_solve.tabi_fmm_local_m2l_seconds_total;
    result.tabi_fmm_local_fallback_seconds_total =
        tabi_solve.tabi_fmm_local_fallback_seconds_total;
    result.tabi_fmm_local_l2p_seconds_total =
        tabi_solve.tabi_fmm_local_l2p_seconds_total;
    result.tabi_fmm_local_sampled_direct_true_residual =
        tabi_solve.tabi_fmm_local_sampled_direct_true_residual;
    result.tabi_fmm_local_operator_finite =
        tabi_solve.tabi_fmm_local_operator_finite;
    result.tabi_fmm_local_refine_probe_mode =
        tabi_solve.tabi_fmm_local_refine_probe_mode;
    result.tabi_fmm_local_refine_probe_count =
        tabi_solve.tabi_fmm_local_refine_probe_count;
    result.tabi_fmm_local_solution_probe_used =
        tabi_solve.tabi_fmm_local_solution_probe_used;
    result.tabi_fmm_local_linf_max_abs_tail =
        tabi_solve.tabi_fmm_local_linf_max_abs_tail;
    result.tabi_fmm_local_linf_max_relative_tail =
        tabi_solve.tabi_fmm_local_linf_max_relative_tail;
    result.tabi_fmm_local_linf_outlier_panels =
        tabi_solve.tabi_fmm_local_linf_outlier_panels;
    result.tabi_fmm_local_geometry_guard =
        tabi_solve.tabi_fmm_local_geometry_guard;
    result.tabi_fmm_local_geometry_theta =
        tabi_solve.tabi_fmm_local_geometry_theta;
    result.tabi_fmm_local_geometry_guard_rejected_cells =
        tabi_solve.tabi_fmm_local_geometry_guard_rejected_cells;
    result.tabi_fmm_local_geometry_guard_direct_fallback_pairs =
        tabi_solve.tabi_fmm_local_geometry_guard_direct_fallback_pairs;
    result.tabi_fmm_local_geometry_gap_min =
        tabi_solve.tabi_fmm_local_geometry_gap_min;
    result.tabi_fmm_local_geometry_ratio_max =
        tabi_solve.tabi_fmm_local_geometry_ratio_max;
    result.tabi_status = tabi_solve.tabi_status;
    result.tabi_convergence_reason = tabi_solve.tabi_convergence_reason;
    result.tabi_residual_history = tabi_solve.tabi_residual_history;
    result.tabi_gmres_restart_requested =
        tabi_solve.tabi_gmres_restart_requested;
    result.tabi_gmres_restart_effective =
        tabi_solve.tabi_gmres_restart_effective;
    result.tabi_best_residual = tabi_solve.tabi_best_residual;
    result.tabi_best_residual_iter = tabi_solve.tabi_best_residual_iter;
    result.tabi_final_residual = tabi_solve.tabi_final_residual;
    result.tabi_best_preconditioned_residual =
        tabi_solve.tabi_best_preconditioned_residual;
    result.tabi_final_preconditioned_residual =
        tabi_solve.tabi_final_preconditioned_residual;
    result.tabi_true_residual_abs = tabi_solve.tabi_true_residual_abs;
    result.tabi_true_residual_rel = tabi_solve.tabi_true_residual_rel;
    result.tabi_true_residual_probe_mode =
        tabi_solve.tabi_true_residual_probe_mode;
    result.tabi_true_residual_probe_scope =
        tabi_solve.tabi_true_residual_probe_scope;
    result.tabi_true_residual_probe_interval =
        tabi_solve.tabi_true_residual_probe_interval;
    result.tabi_true_residual_sample_count =
        tabi_solve.tabi_true_residual_sample_count;
    result.tabi_true_residual_probe_calls =
        tabi_solve.tabi_true_residual_probe_calls;
    result.tabi_true_residual_probe_ms = tabi_solve.tabi_true_residual_probe_ms;
    result.tabi_sampled_true_residual_rel =
        tabi_solve.tabi_sampled_true_residual_rel;
    result.tabi_sampled_true_residual_rows =
        tabi_solve.tabi_sampled_true_residual_rows;
    result.tabi_sampled_probe_calls = tabi_solve.tabi_sampled_probe_calls;
    result.tabi_sampled_probe_ms = tabi_solve.tabi_sampled_probe_ms;
    result.tabi_full_true_residual_confirm_calls =
        tabi_solve.tabi_full_true_residual_confirm_calls;
    result.tabi_full_probe_ms = tabi_solve.tabi_full_probe_ms;
    result.tabi_reaction_energy_source = tabi_solve.tabi_reaction_energy_source;
    result.tabi_rhs_ms = tabi_solve.tabi_rhs_ms;
    result.tabi_gmres_ms = tabi_solve.tabi_gmres_ms;
    result.tabi_matvec_ms_total = tabi_solve.tabi_matvec_ms_total;
    result.tabi_matvec_ms_avg = tabi_solve.tabi_matvec_ms_avg;
    result.tabi_matvec_calls = tabi_solve.tabi_matvec_calls;
    result.tabi_tree_vs_direct_matvec_rel =
        tabi_solve.tabi_tree_vs_direct_matvec_rel;
    result.tabi_matvec_audit_rows = tabi_solve.tabi_matvec_audit_rows;
    result.tabi_matvec_audit_ms = tabi_solve.tabi_matvec_audit_ms;
    result.tabi_fmm_cartesian_pre_audit =
        tabi_solve.tabi_fmm_cartesian_pre_audit;
    result.tabi_fmm_cartesian_solution_audit =
        tabi_solve.tabi_fmm_cartesian_solution_audit;
    result.tabi_fmm_cartesian_channel_audit =
        tabi_solve.tabi_fmm_cartesian_channel_audit;
    result.tabi_fmm_cartesian_retry = tabi_solve.tabi_fmm_cartesian_retry;
    result.tabi_orthogonalization_ms = tabi_solve.tabi_orthogonalization_ms;
    result.tabi_preconditioner_ms = tabi_solve.tabi_preconditioner_ms;
    result.tabi_surface_neighbor_ms = tabi_solve.tabi_surface_neighbor_ms;
    result.tabi_surface_mesh_ms = tabi_solve.tabi_surface_mesh_ms;
    result.tabi_solid_angle_ms = tabi_solve.tabi_solid_angle_ms;
    result.tabi_tree_moment_ms = tabi_solve.tabi_tree_moment_ms;
    result.tabi_tree_traversal_ms = tabi_solve.tabi_tree_traversal_ms;
    result.tabi_matvec_cost_profile = tabi_solve.tabi_matvec_cost_profile;
    result.tabi_leaf_panels_per_second = tabi_solve.tabi_leaf_panels_per_second;
    result.tabi_clean_leaf_panels_per_second =
        tabi_solve.tabi_clean_leaf_panels_per_second;
    result.tabi_tree_traversal_fraction =
        tabi_solve.tabi_tree_traversal_fraction;
    result.tabi_near_eval_ms = tabi_solve.tabi_near_eval_ms;
    result.tabi_far_eval_ms = tabi_solve.tabi_far_eval_ms;
    result.tabi_singular_near_eval_ms = tabi_solve.tabi_singular_near_eval_ms;
    result.tabi_leaf_near_eval_ms = tabi_solve.tabi_leaf_near_eval_ms;
    result.tabi_leaf_panel_eval_ms = tabi_solve.tabi_leaf_panel_eval_ms;
    result.tabi_leaf_tile_eval_ms = tabi_solve.tabi_leaf_tile_eval_ms;
    result.tabi_leaf_residual_eval_ms = tabi_solve.tabi_leaf_residual_eval_ms;
    result.tabi_smooth_leaf_node_eval_ms =
        tabi_solve.tabi_smooth_leaf_node_eval_ms;
    result.tabi_singular_near_interactions =
        tabi_solve.tabi_singular_near_interactions;
    result.tabi_leaf_near_interactions = tabi_solve.tabi_leaf_near_interactions;
    result.tabi_far_node_interactions = tabi_solve.tabi_far_node_interactions;
    result.tabi_leaf_panel_interactions =
        tabi_solve.tabi_leaf_panel_interactions;
    result.tabi_leaf_tile_interactions = tabi_solve.tabi_leaf_tile_interactions;
    result.tabi_leaf_tile_panel_interactions =
        tabi_solve.tabi_leaf_tile_panel_interactions;
    result.tabi_leaf_residual_panel_interactions =
        tabi_solve.tabi_leaf_residual_panel_interactions;
    result.tabi_smooth_leaf_node_interactions =
        tabi_solve.tabi_smooth_leaf_node_interactions;
    result.tabi_leaf_pruned_interactions =
        tabi_solve.tabi_leaf_pruned_interactions;
    result.tabi_leaf_mac_candidates = tabi_solve.tabi_leaf_mac_candidates;
    result.tabi_leaf_mac_accepted = tabi_solve.tabi_leaf_mac_accepted;
    result.tabi_leaf_mac_rejected = tabi_solve.tabi_leaf_mac_rejected;
    result.tabi_leaf_mac_far_panels = tabi_solve.tabi_leaf_mac_far_panels;
    result.tabi_leaf_mac_direct_panels_saved_estimate =
        tabi_solve.tabi_leaf_mac_direct_panels_saved_estimate;
    result.tabi_leaf_work_profile = tabi_solve.tabi_leaf_work_profile;
    result.tabi_leaf_work_mac_terminal_panels =
        tabi_solve.tabi_leaf_work_mac_terminal_panels;
    result.tabi_leaf_work_physical_gap_panels =
        tabi_solve.tabi_leaf_work_physical_gap_panels;
    result.tabi_leaf_work_fragmentation_panels =
        tabi_solve.tabi_leaf_work_fragmentation_panels;
    result.tabi_leaf_work_convergence_panels =
        tabi_solve.tabi_leaf_work_convergence_panels;
    result.tabi_leaf_work_convergence_estimator_panels =
        tabi_solve.tabi_leaf_work_convergence_estimator_panels;
    result.tabi_leaf_work_convergence_rho_panels =
        tabi_solve.tabi_leaf_work_convergence_rho_panels;
    result.tabi_leaf_work_convergence_aspect_panels =
        tabi_solve.tabi_leaf_work_convergence_aspect_panels;
    result.tabi_leaf_work_surface_margin_panels =
        tabi_solve.tabi_leaf_work_surface_margin_panels;
    result.tabi_leaf_work_surface_margin_hard_panels =
        tabi_solve.tabi_leaf_work_surface_margin_hard_panels;
    result.tabi_leaf_work_surface_margin_soft_panels =
        tabi_solve.tabi_leaf_work_surface_margin_soft_panels;
    result.tabi_leaf_work_k3_rho_cap_panels =
        tabi_solve.tabi_leaf_work_k3_rho_cap_panels;
    result.tabi_leaf_work_singular_fallback_panels =
        tabi_solve.tabi_leaf_work_singular_fallback_panels;
    result.tabi_leaf_work_scalar_tail_panels =
        tabi_solve.tabi_leaf_work_scalar_tail_panels;
    result.tabi_leaf_work_profile_accounted_panels =
        tabi_solve.tabi_leaf_work_profile_accounted_panels;
    result.tabi_leaf_work_mac_terminal_frac =
        tabi_solve.tabi_leaf_work_mac_terminal_frac;
    result.tabi_leaf_work_convergence_frac =
        tabi_solve.tabi_leaf_work_convergence_frac;
    result.tabi_leaf_work_surface_margin_frac =
        tabi_solve.tabi_leaf_work_surface_margin_frac;
    result.tabi_leaf_work_singular_fallback_frac =
        tabi_solve.tabi_leaf_work_singular_fallback_frac;
    result.tabi_leaf_work_scalar_tail_frac =
        tabi_solve.tabi_leaf_work_scalar_tail_frac;
    result.tabi_adaptive_smooth_leaf_accepted =
        tabi_solve.tabi_adaptive_smooth_leaf_accepted;
    result.tabi_adaptive_smooth_leaf_rejected =
        tabi_solve.tabi_adaptive_smooth_leaf_rejected;
    result.tabi_adaptive_span_tile_accepted =
        tabi_solve.tabi_adaptive_span_tile_accepted;
    result.tabi_adaptive_span_tile_rejected =
        tabi_solve.tabi_adaptive_span_tile_rejected;
    result.tabi_adaptive_exact_leaf_fallbacks =
        tabi_solve.tabi_adaptive_exact_leaf_fallbacks;
    result.tabi_adaptive_reject_one_ring_overlap =
        tabi_solve.tabi_adaptive_reject_one_ring_overlap;
    result.tabi_adaptive_reject_error_bound_failed =
        tabi_solve.tabi_adaptive_reject_error_bound_failed;
    result.tabi_adaptive_reject_p4_check_failed =
        tabi_solve.tabi_adaptive_reject_p4_check_failed;
    result.tabi_adaptive_reject_too_close =
        tabi_solve.tabi_adaptive_reject_too_close;
    result.tabi_leaf_work_reduction_ratio =
        tabi_solve.tabi_leaf_work_reduction_ratio;
    result.tabi_target_cluster_interactions =
        tabi_solve.tabi_target_cluster_interactions;
    result.tabi_target_cluster_targets = tabi_solve.tabi_target_cluster_targets;
    result.tabi_target_cluster_panels_pruned =
        tabi_solve.tabi_target_cluster_panels_pruned;
    result.tabi_target_cluster_rejected =
        tabi_solve.tabi_target_cluster_rejected;
    result.tabi_target_cluster_eval_ms = tabi_solve.tabi_target_cluster_eval_ms;
    result.tabi_leaf_pair_batch_interactions =
        tabi_solve.tabi_leaf_pair_batch_interactions;
    result.tabi_leaf_pair_batch_targets =
        tabi_solve.tabi_leaf_pair_batch_targets;
    result.tabi_leaf_pair_batch_source_panels =
        tabi_solve.tabi_leaf_pair_batch_source_panels;
    result.tabi_leaf_pair_batch_residual_panels =
        tabi_solve.tabi_leaf_pair_batch_residual_panels;
    result.tabi_leaf_pair_batch_fallbacks =
        tabi_solve.tabi_leaf_pair_batch_fallbacks;
    result.tabi_leaf_pair_batch_ms = tabi_solve.tabi_leaf_pair_batch_ms;
    result.tabi_leaf_pair_batch_kernel = tabi_solve.tabi_leaf_pair_batch_kernel;
    result.tabi_leaf_pair_batch_vectorized_sources =
        tabi_solve.tabi_leaf_pair_batch_vectorized_sources;
    result.tabi_leaf_pair_batch_vector_lanes =
        tabi_solve.tabi_leaf_pair_batch_vector_lanes;
    result.tabi_leaf_pair_batch_pairs_per_second =
        tabi_solve.tabi_leaf_pair_batch_pairs_per_second;
    result.tabi_leaf_kernel_profile = tabi_solve.tabi_leaf_kernel_profile;
    result.tabi_leaf_work_profile = tabi_solve.tabi_leaf_work_profile;
    result.tabi_jcp_direct_kernel = tabi_solve.tabi_jcp_direct_kernel;
    result.tabi_leaf_kernel_scalar_tasks =
        tabi_solve.tabi_leaf_kernel_scalar_tasks;
    result.tabi_leaf_kernel_batch_tasks =
        tabi_solve.tabi_leaf_kernel_batch_tasks;
    result.tabi_leaf_kernel_source_panel_evals =
        tabi_solve.tabi_leaf_kernel_source_panel_evals;
    result.tabi_leaf_kernel_scalar_ms = tabi_solve.tabi_leaf_kernel_scalar_ms;
    result.tabi_leaf_kernel_target_avx2_ms =
        tabi_solve.tabi_leaf_kernel_target_avx2_ms;
    result.tabi_leaf_kernel_tail_ms = tabi_solve.tabi_leaf_kernel_tail_ms;
    result.tabi_leaf_pair_batch_soa_sources =
        tabi_solve.tabi_leaf_pair_batch_soa_sources;
    result.tabi_leaf_pair_batch_soa_ms = tabi_solve.tabi_leaf_pair_batch_soa_ms;
    result.tabi_leaf_pair_batch_scalar_tail_ms =
        tabi_solve.tabi_leaf_pair_batch_scalar_tail_ms;
    result.tabi_leaf_pair_batch_clean_source_panels_per_second =
        tabi_solve.tabi_leaf_pair_batch_clean_source_panels_per_second;
    result.tabi_leaf_pair_selection_profile =
        tabi_solve.tabi_leaf_pair_selection_profile;
    result.tabi_leaf_pair_probe_cache = tabi_solve.tabi_leaf_pair_probe_cache;
    result.tabi_leaf_pair_target_group = tabi_solve.tabi_leaf_pair_target_group;
    result.tabi_leaf_pair_directness_probe =
        tabi_solve.tabi_leaf_pair_directness_probe;
    result.tabi_leaf_pair_directness_cache =
        tabi_solve.tabi_leaf_pair_directness_cache;
    result.tabi_leaf_pair_directness_verify =
        tabi_solve.tabi_leaf_pair_directness_verify;
    result.tabi_leaf_pair_max_lookahead =
        tabi_solve.tabi_leaf_pair_max_lookahead;
    result.tabi_leaf_pair_selection_candidate_scans =
        tabi_solve.tabi_leaf_pair_selection_candidate_scans;
    result.tabi_leaf_pair_selection_clean_rejects =
        tabi_solve.tabi_leaf_pair_selection_clean_rejects;
    result.tabi_leaf_pair_selection_probe_calls =
        tabi_solve.tabi_leaf_pair_selection_probe_calls;
    result.tabi_leaf_pair_selection_probe_node_visits =
        tabi_solve.tabi_leaf_pair_selection_probe_node_visits;
    result.tabi_leaf_pair_selection_probe_ms =
        tabi_solve.tabi_leaf_pair_selection_probe_ms;
    result.tabi_leaf_pair_selection_ms = tabi_solve.tabi_leaf_pair_selection_ms;
    result.tabi_leaf_pair_selection_probe_cache_hits =
        tabi_solve.tabi_leaf_pair_selection_probe_cache_hits;
    result.tabi_leaf_pair_selection_probe_cache_misses =
        tabi_solve.tabi_leaf_pair_selection_probe_cache_misses;
    result.tabi_leaf_pair_selection_insufficient_targets =
        tabi_solve.tabi_leaf_pair_selection_insufficient_targets;
    result.tabi_leaf_pair_selection_same_target_leaf_rejects =
        tabi_solve.tabi_leaf_pair_selection_same_target_leaf_rejects;
    result.tabi_leaf_pair_directness_ancestor_probe_calls =
        tabi_solve.tabi_leaf_pair_directness_ancestor_probe_calls;
    result.tabi_leaf_pair_directness_ancestor_probe_node_visits =
        tabi_solve.tabi_leaf_pair_directness_ancestor_probe_node_visits;
    result.tabi_leaf_pair_directness_ancestor_probe_ms =
        tabi_solve.tabi_leaf_pair_directness_ancestor_probe_ms;
    result.tabi_leaf_pair_directness_pair_cache_hits =
        tabi_solve.tabi_leaf_pair_directness_pair_cache_hits;
    result.tabi_leaf_pair_directness_pair_cache_misses =
        tabi_solve.tabi_leaf_pair_directness_pair_cache_misses;
    result.tabi_leaf_pair_directness_all_direct_proofs =
        tabi_solve.tabi_leaf_pair_directness_all_direct_proofs;
    result.tabi_leaf_pair_directness_mixed_fallbacks =
        tabi_solve.tabi_leaf_pair_directness_mixed_fallbacks;
    result.tabi_leaf_pair_directness_verify_checks =
        tabi_solve.tabi_leaf_pair_directness_verify_checks;
    result.tabi_leaf_pair_directness_verify_mismatches =
        tabi_solve.tabi_leaf_pair_directness_verify_mismatches;
    result.tabi_leaf_pair_scheduler = tabi_solve.tabi_leaf_pair_scheduler;
    result.tabi_leaf_pair_worklist_max_active =
        tabi_solve.tabi_leaf_pair_worklist_max_active;
    result.tabi_leaf_pair_worklist_verify =
        tabi_solve.tabi_leaf_pair_worklist_verify;
    result.tabi_leaf_pair_worklist_tasks =
        tabi_solve.tabi_leaf_pair_worklist_tasks;
    result.tabi_leaf_pair_worklist_groups =
        tabi_solve.tabi_leaf_pair_worklist_groups;
    result.tabi_leaf_pair_worklist_batched_groups =
        tabi_solve.tabi_leaf_pair_worklist_batched_groups;
    result.tabi_leaf_pair_worklist_scalar_flush_tasks =
        tabi_solve.tabi_leaf_pair_worklist_scalar_flush_tasks;
    result.tabi_leaf_pair_worklist_forced_flushes =
        tabi_solve.tabi_leaf_pair_worklist_forced_flushes;
    result.tabi_leaf_pair_worklist_active_groups_max =
        tabi_solve.tabi_leaf_pair_worklist_active_groups_max;
    result.tabi_leaf_pair_worklist_ms = tabi_solve.tabi_leaf_pair_worklist_ms;
    result.tabi_leaf_pair_worklist_verify_checks =
        tabi_solve.tabi_leaf_pair_worklist_verify_checks;
    result.tabi_leaf_pair_worklist_verify_mismatches =
        tabi_solve.tabi_leaf_pair_worklist_verify_mismatches;
    result.tabi_leaf_block_verify = tabi_solve.tabi_leaf_block_verify;
    result.tabi_leaf_block_target_leaves =
        tabi_solve.tabi_leaf_block_target_leaves;
    result.tabi_leaf_block_targets = tabi_solve.tabi_leaf_block_targets;
    result.tabi_leaf_block_source_groups =
        tabi_solve.tabi_leaf_block_source_groups;
    result.tabi_leaf_block_batched_groups =
        tabi_solve.tabi_leaf_block_batched_groups;
    result.tabi_leaf_block_scalar_tail_tasks =
        tabi_solve.tabi_leaf_block_scalar_tail_tasks;
    result.tabi_leaf_block_group_ms = tabi_solve.tabi_leaf_block_group_ms;
    result.tabi_leaf_block_flush_ms = tabi_solve.tabi_leaf_block_flush_ms;
    result.tabi_leaf_block_writeback_ms =
        tabi_solve.tabi_leaf_block_writeback_ms;
    result.tabi_leaf_block_verify_checks =
        tabi_solve.tabi_leaf_block_verify_checks;
    result.tabi_leaf_block_verify_mismatches =
        tabi_solve.tabi_leaf_block_verify_mismatches;
    result.tabi_tree_preset = tabi_solve.tabi_tree_preset;
    result.tabi_adaptive_level = tabi_solve.tabi_adaptive_level;
    result.tabi_cluster_policy = tabi_solve.tabi_cluster_policy;
    result.tabi_target_batch_size = tabi_solve.tabi_target_batch_size;
    result.tabi_leaf_pair_min_source_panels =
        tabi_solve.tabi_leaf_pair_min_source_panels;
    result.tabi_leaf_pair_min_targets = tabi_solve.tabi_leaf_pair_min_targets;
    result.tabi_leaf_near_policy = tabi_solve.tabi_leaf_near_policy;
    result.tabi_leaf_panel_kernel = tabi_solve.tabi_leaf_panel_kernel;
    result.tabi_interaction_mode = tabi_solve.tabi_interaction_mode;
    result.tabi_node_bounds_mode = tabi_solve.tabi_node_bounds_mode;
    result.tabi_tree_builder = tabi_solve.tabi_tree_builder;
    result.tabi_expansion_center = tabi_solve.tabi_expansion_center;
    result.tabi_fragmentation_safety = tabi_solve.tabi_fragmentation_safety;
    result.tabi_mac_safety = tabi_solve.tabi_mac_safety;
    result.tabi_mac_gap_factor = tabi_solve.tabi_mac_gap_factor;
    result.tabi_tree_depth_max = tabi_solve.tabi_tree_depth_max;
    result.tabi_mac_reject_radius_limited =
        tabi_solve.tabi_mac_reject_radius_limited;
    result.tabi_mac_accept_rate = tabi_solve.tabi_mac_accept_rate;
    result.tabi_node_radius_max = tabi_solve.tabi_node_radius_max;
    result.tabi_node_radius_mean = tabi_solve.tabi_node_radius_mean;
    result.tabi_node_aspect_ratio_max = tabi_solve.tabi_node_aspect_ratio_max;
    result.tabi_node_aspect_ratio_mean = tabi_solve.tabi_node_aspect_ratio_mean;
    result.tabi_leaf_panel_work_per_matvec =
        tabi_solve.tabi_leaf_panel_work_per_matvec;
    result.tabi_far_node_work_per_matvec =
        tabi_solve.tabi_far_node_work_per_matvec;
    result.tabi_far_geometric_unsafe_count =
        tabi_solve.tabi_far_geometric_unsafe_count;
    result.tabi_far_geometric_unsafe_panels =
        tabi_solve.tabi_far_geometric_unsafe_panels;
    result.tabi_far_geometric_min_gap = tabi_solve.tabi_far_geometric_min_gap;
    result.tabi_far_geometric_gap_ratio_min =
        tabi_solve.tabi_far_geometric_gap_ratio_min;
    result.tabi_far_geometric_gap_ratio_mean =
        tabi_solve.tabi_far_geometric_gap_ratio_mean;
    result.tabi_far_geometric_audit_samples =
        tabi_solve.tabi_far_geometric_audit_samples;
    result.tabi_mac_physical_gap_rejects =
        tabi_solve.tabi_mac_physical_gap_rejects;
    result.tabi_mac_convergence_rho_rejects =
        tabi_solve.tabi_mac_convergence_rho_rejects;
    result.tabi_mac_convergence_aspect_rejects =
        tabi_solve.tabi_mac_convergence_aspect_rejects;
    result.tabi_mac_convergence_rho_reject_panels =
        tabi_solve.tabi_mac_convergence_rho_reject_panels;
    result.tabi_mac_convergence_aspect_reject_panels =
        tabi_solve.tabi_mac_convergence_aspect_reject_panels;
    result.tabi_mac_surface_margin_rejects =
        tabi_solve.tabi_mac_surface_margin_rejects;
    result.tabi_mac_surface_margin_reject_panels =
        tabi_solve.tabi_mac_surface_margin_reject_panels;
    result.tabi_mac_surface_margin_applied =
        tabi_solve.tabi_mac_surface_margin_applied;
    result.tabi_mac_surface_margin_skipped =
        tabi_solve.tabi_mac_surface_margin_skipped;
    result.tabi_mac_surface_margin_adaptive_rejects =
        tabi_solve.tabi_mac_surface_margin_adaptive_rejects;
    result.tabi_mac_surface_margin_adaptive_reject_panels =
        tabi_solve.tabi_mac_surface_margin_adaptive_reject_panels;
    result.tabi_mac_surface_margin_skipped_accepted_far =
        tabi_solve.tabi_mac_surface_margin_skipped_accepted_far;
    result.tabi_mac_surface_margin_hard_applied =
        tabi_solve.tabi_mac_surface_margin_hard_applied;
    result.tabi_mac_surface_margin_soft_applied =
        tabi_solve.tabi_mac_surface_margin_soft_applied;
    result.tabi_mac_surface_margin_hard_rejects =
        tabi_solve.tabi_mac_surface_margin_hard_rejects;
    result.tabi_mac_surface_margin_hard_reject_panels =
        tabi_solve.tabi_mac_surface_margin_hard_reject_panels;
    result.tabi_mac_surface_margin_soft_rejects =
        tabi_solve.tabi_mac_surface_margin_soft_rejects;
    result.tabi_mac_surface_margin_soft_reject_panels =
        tabi_solve.tabi_mac_surface_margin_soft_reject_panels;
    result.tabi_fragmented_far_count = tabi_solve.tabi_fragmented_far_count;
    result.tabi_fragmented_far_panels = tabi_solve.tabi_fragmented_far_panels;
    result.tabi_fragmented_split_rejects =
        tabi_solve.tabi_fragmented_split_rejects;
    result.tabi_fragmentation_center_gap_ratio_max =
        tabi_solve.tabi_fragmentation_center_gap_ratio_max;
    result.tabi_fragmentation_center_gap_ratio_mean =
        tabi_solve.tabi_fragmentation_center_gap_ratio_mean;
    result.tabi_moment_audit = tabi_solve.tabi_moment_audit;
    result.tabi_moment_construction = tabi_solve.tabi_moment_construction;
    result.tabi_moment_sum = tabi_solve.tabi_moment_sum;
    result.tabi_convergence_audit = tabi_solve.tabi_convergence_audit;
    result.tabi_convergence_safety = tabi_solve.tabi_convergence_safety;
    result.tabi_convergence_rho_cap = tabi_solve.tabi_convergence_rho_cap;
    result.tabi_mac_aspect_cap = tabi_solve.tabi_mac_aspect_cap;
    result.tabi_convergence_estimator = tabi_solve.tabi_convergence_estimator;
    result.tabi_convergence_tail_threshold =
        tabi_solve.tabi_convergence_tail_threshold;
    result.tabi_convergence_estimator_error_cap =
        tabi_solve.tabi_convergence_estimator_error_cap;
    result.tabi_convergence_unstable_rho_cap =
        tabi_solve.tabi_convergence_unstable_rho_cap;
    result.tabi_convergence_calibration =
        tabi_solve.tabi_convergence_calibration;
    result.tabi_convergence_dielectric_amp =
        tabi_solve.tabi_convergence_dielectric_amp;
    result.tabi_convergence_estimator_model =
        tabi_solve.tabi_convergence_estimator_model;
    result.tabi_convergence_channel_calibration =
        tabi_solve.tabi_convergence_channel_calibration;
    result.tabi_convergence_split_policy =
        tabi_solve.tabi_convergence_split_policy;
    result.tabi_convergence_split_selectivity =
        tabi_solve.tabi_convergence_split_selectivity;
    result.tabi_convergence_split_absolute_floor =
        tabi_solve.tabi_convergence_split_absolute_floor;
    result.tabi_convergence_overreject_audit =
        tabi_solve.tabi_convergence_overreject_audit;
    result.tabi_convergence_overreject_detail =
        tabi_solve.tabi_convergence_overreject_detail;
    result.tabi_convergence_shape_audit =
        tabi_solve.tabi_convergence_shape_audit;
    result.tabi_convergence_kernel_response_audit =
        tabi_solve.tabi_convergence_kernel_response_audit;
    result.tabi_channel_missed_semantics =
        tabi_solve.tabi_channel_missed_semantics;
    result.tabi_accepted_miss_diagnostics =
        tabi_solve.tabi_accepted_miss_diagnostics;
    result.tabi_accepted_miss_absolute_floor =
        tabi_solve.tabi_accepted_miss_absolute_floor;
    result.tabi_convergence_k3_accepted_rho_cap =
        tabi_solve.tabi_convergence_k3_accepted_rho_cap;
    result.tabi_mac_distance_policy = tabi_solve.tabi_mac_distance_policy;
    result.tabi_mac_surface_margin_factor =
        tabi_solve.tabi_mac_surface_margin_factor;
    result.tabi_mac_surface_margin_policy =
        tabi_solve.tabi_mac_surface_margin_policy;
    result.tabi_mac_surface_margin_aspect_threshold =
        tabi_solve.tabi_mac_surface_margin_aspect_threshold;
    result.tabi_mac_surface_margin_rho_trigger =
        tabi_solve.tabi_mac_surface_margin_rho_trigger;
    result.tabi_mac_surface_margin_soft_aspect_threshold =
        tabi_solve.tabi_mac_surface_margin_soft_aspect_threshold;
    result.tabi_mac_surface_margin_soft_rho_trigger =
        tabi_solve.tabi_mac_surface_margin_soft_rho_trigger;
    result.tabi_mac_surface_margin_soft_factor =
        tabi_solve.tabi_mac_surface_margin_soft_factor;
    result.tabi_convergence_estimator_samples =
        tabi_solve.tabi_convergence_estimator_samples;
    result.tabi_convergence_estimator_unstable_far_count =
        tabi_solve.tabi_convergence_estimator_unstable_far_count;
    result.tabi_convergence_estimator_unstable_far_panels =
        tabi_solve.tabi_convergence_estimator_unstable_far_panels;
    result.tabi_convergence_estimator_split_rejects =
        tabi_solve.tabi_convergence_estimator_split_rejects;
    result.tabi_convergence_estimator_split_reject_panels =
        tabi_solve.tabi_convergence_estimator_split_reject_panels;
    result.tabi_convergence_low_abs_reaccepted_far_count =
        tabi_solve.tabi_convergence_low_abs_reaccepted_far_count;
    result.tabi_convergence_low_abs_reaccepted_far_panels =
        tabi_solve.tabi_convergence_low_abs_reaccepted_far_panels;
    result.tabi_convergence_estimator_tail_ratio_max =
        tabi_solve.tabi_convergence_estimator_tail_ratio_max;
    result.tabi_convergence_estimator_tail_ratio_mean =
        tabi_solve.tabi_convergence_estimator_tail_ratio_mean;
    result.tabi_convergence_estimator_error_max =
        tabi_solve.tabi_convergence_estimator_error_max;
    result.tabi_convergence_estimator_error_mean =
        tabi_solve.tabi_convergence_estimator_error_mean;
    result.tabi_convergence_estimator_abs_error_max =
        tabi_solve.tabi_convergence_estimator_abs_error_max;
    result.tabi_convergence_estimator_abs_error_mean =
        tabi_solve.tabi_convergence_estimator_abs_error_mean;
    result.tabi_convergence_tail_norm_p_max =
        tabi_solve.tabi_convergence_tail_norm_p_max;
    result.tabi_convergence_tail_norm_p_mean =
        tabi_solve.tabi_convergence_tail_norm_p_mean;
    result.tabi_convergence_tail_norm_pm1_max =
        tabi_solve.tabi_convergence_tail_norm_pm1_max;
    result.tabi_convergence_tail_norm_pm1_mean =
        tabi_solve.tabi_convergence_tail_norm_pm1_mean;
    result.tabi_convergence_effective_jcp_amp_max =
        tabi_solve.tabi_convergence_effective_jcp_amp_max;
    result.tabi_convergence_effective_jcp_amp_mean =
        tabi_solve.tabi_convergence_effective_jcp_amp_mean;
    result.tabi_convergence_dielectric_amp_max =
        tabi_solve.tabi_convergence_dielectric_amp_max;
    result.tabi_convergence_dielectric_amp_mean =
        tabi_solve.tabi_convergence_dielectric_amp_mean;
    result.tabi_convergence_channel_estimated_error_max =
        tabi_solve.tabi_convergence_channel_estimated_error_max;
    result.tabi_convergence_channel_estimated_error_mean =
        tabi_solve.tabi_convergence_channel_estimated_error_mean;
    result.tabi_convergence_channel_effective_jcp_amp_max =
        tabi_solve.tabi_convergence_channel_effective_jcp_amp_max;
    result.tabi_convergence_channel_effective_jcp_amp_mean =
        tabi_solve.tabi_convergence_channel_effective_jcp_amp_mean;
    result.tabi_convergence_channel_unstable_far_count =
        tabi_solve.tabi_convergence_channel_unstable_far_count;
    result.tabi_convergence_channel_unstable_far_panels =
        tabi_solve.tabi_convergence_channel_unstable_far_panels;
    result.tabi_convergence_channel_split_rejects =
        tabi_solve.tabi_convergence_channel_split_rejects;
    result.tabi_convergence_channel_split_reject_panels =
        tabi_solve.tabi_convergence_channel_split_reject_panels;
    result.tabi_convergence_channel_low_abs_reaccepted_count =
        tabi_solve.tabi_convergence_channel_low_abs_reaccepted_count;
    result.tabi_convergence_channel_low_abs_reaccepted_panels =
        tabi_solve.tabi_convergence_channel_low_abs_reaccepted_panels;
    result.tabi_convergence_channel_high_error_caught =
        tabi_solve.tabi_convergence_channel_high_error_caught;
    result.tabi_convergence_channel_high_error_missed =
        tabi_solve.tabi_convergence_channel_high_error_missed;
    result.tabi_convergence_channel_high_error_caught_by_split =
        tabi_solve.tabi_convergence_channel_high_error_caught_by_split;
    result.tabi_convergence_estimator_high_error_caught =
        tabi_solve.tabi_convergence_estimator_high_error_caught;
    result.tabi_convergence_estimator_high_error_missed =
        tabi_solve.tabi_convergence_estimator_high_error_missed;
    result.tabi_convergence_estimator_high_error_caught_by_split =
        tabi_solve.tabi_convergence_estimator_high_error_caught_by_split;
    result.tabi_convergence_estimator_phantom_high_error_count =
        tabi_solve.tabi_convergence_estimator_phantom_high_error_count;
    result.tabi_convergence_estimator_phantom_high_error_abs_max =
        tabi_solve.tabi_convergence_estimator_phantom_high_error_abs_max;
    result.tabi_convergence_overreject_count =
        tabi_solve.tabi_convergence_overreject_count;
    result.tabi_convergence_overreject_panels =
        tabi_solve.tabi_convergence_overreject_panels;
    result.tabi_convergence_overreject_caught_count =
        tabi_solve.tabi_convergence_overreject_caught_count;
    result.tabi_convergence_overreject_caught_panels =
        tabi_solve.tabi_convergence_overreject_caught_panels;
    result.tabi_convergence_overreject_abs_error_max =
        tabi_solve.tabi_convergence_overreject_abs_error_max;
    result.tabi_convergence_overreject_abs_error_mean =
        tabi_solve.tabi_convergence_overreject_abs_error_mean;
    result.tabi_convergence_overreject_abs_error_p95 =
        tabi_solve.tabi_convergence_overreject_abs_error_p95;
    result.tabi_convergence_overreject_abs_error_p99 =
        tabi_solve.tabi_convergence_overreject_abs_error_p99;
    result.tabi_convergence_overreject_estimated_abs_error_max =
        tabi_solve.tabi_convergence_overreject_estimated_abs_error_max;
    result.tabi_convergence_overreject_estimated_abs_error_mean =
        tabi_solve.tabi_convergence_overreject_estimated_abs_error_mean;
    result.tabi_convergence_overreject_required_scale_max =
        tabi_solve.tabi_convergence_overreject_required_scale_max;
    result.tabi_convergence_overreject_required_scale_p95 =
        tabi_solve.tabi_convergence_overreject_required_scale_p95;
    result.tabi_convergence_overreject_required_scale_p99 =
        tabi_solve.tabi_convergence_overreject_required_scale_p99;
    result.tabi_convergence_channel_overreject_count =
        tabi_solve.tabi_convergence_channel_overreject_count;
    result.tabi_convergence_channel_overreject_caught_count =
        tabi_solve.tabi_convergence_channel_overreject_caught_count;
    result.tabi_convergence_channel_overreject_panels =
        tabi_solve.tabi_convergence_channel_overreject_panels;
    result.tabi_convergence_channel_overreject_abs_error_max =
        tabi_solve.tabi_convergence_channel_overreject_abs_error_max;
    result.tabi_convergence_channel_overreject_abs_error_mean =
        tabi_solve.tabi_convergence_channel_overreject_abs_error_mean;
    result.tabi_convergence_channel_overreject_required_scale_max =
        tabi_solve.tabi_convergence_channel_overreject_required_scale_max;
    result.tabi_convergence_channel_overreject_required_scale_p99 =
        tabi_solve.tabi_convergence_channel_overreject_required_scale_p99;
    result.tabi_phantom_high_error_channel_count =
        tabi_solve.tabi_phantom_high_error_channel_count;
    result.tabi_phantom_high_error_channel_abs_max =
        tabi_solve.tabi_phantom_high_error_channel_abs_max;
    result.tabi_accepted_miss_channel_count =
        tabi_solve.tabi_accepted_miss_channel_count;
    result.tabi_accepted_miss_channel_panels =
        tabi_solve.tabi_accepted_miss_channel_panels;
    result.tabi_accepted_miss_channel_rho_min =
        tabi_solve.tabi_accepted_miss_channel_rho_min;
    result.tabi_accepted_miss_channel_rho_mean =
        tabi_solve.tabi_accepted_miss_channel_rho_mean;
    result.tabi_accepted_miss_channel_rho_max =
        tabi_solve.tabi_accepted_miss_channel_rho_max;
    result.tabi_accepted_miss_channel_tail_ratio_mean =
        tabi_solve.tabi_accepted_miss_channel_tail_ratio_mean;
    result.tabi_accepted_miss_channel_tail_ratio_max =
        tabi_solve.tabi_accepted_miss_channel_tail_ratio_max;
    result.tabi_accepted_miss_channel_estimated_error_mean =
        tabi_solve.tabi_accepted_miss_channel_estimated_error_mean;
    result.tabi_accepted_miss_channel_estimated_error_max =
        tabi_solve.tabi_accepted_miss_channel_estimated_error_max;
    result.tabi_accepted_miss_worst_channel =
        tabi_solve.tabi_accepted_miss_worst_channel;
    result.tabi_convergence_k3_rho_cap_rejects =
        tabi_solve.tabi_convergence_k3_rho_cap_rejects;
    result.tabi_convergence_k3_rho_cap_reject_panels =
        tabi_solve.tabi_convergence_k3_rho_cap_reject_panels;
    result.tabi_mac_surface_margin_rejects =
        tabi_solve.tabi_mac_surface_margin_rejects;
    result.tabi_mac_surface_margin_reject_panels =
        tabi_solve.tabi_mac_surface_margin_reject_panels;
    result.tabi_mac_surface_margin_applied =
        tabi_solve.tabi_mac_surface_margin_applied;
    result.tabi_mac_surface_margin_skipped =
        tabi_solve.tabi_mac_surface_margin_skipped;
    result.tabi_mac_surface_margin_adaptive_rejects =
        tabi_solve.tabi_mac_surface_margin_adaptive_rejects;
    result.tabi_mac_surface_margin_adaptive_reject_panels =
        tabi_solve.tabi_mac_surface_margin_adaptive_reject_panels;
    result.tabi_mac_surface_margin_skipped_accepted_far =
        tabi_solve.tabi_mac_surface_margin_skipped_accepted_far;
    result.tabi_mac_surface_margin_hard_applied =
        tabi_solve.tabi_mac_surface_margin_hard_applied;
    result.tabi_mac_surface_margin_soft_applied =
        tabi_solve.tabi_mac_surface_margin_soft_applied;
    result.tabi_mac_surface_margin_hard_rejects =
        tabi_solve.tabi_mac_surface_margin_hard_rejects;
    result.tabi_mac_surface_margin_hard_reject_panels =
        tabi_solve.tabi_mac_surface_margin_hard_reject_panels;
    result.tabi_mac_surface_margin_soft_rejects =
        tabi_solve.tabi_mac_surface_margin_soft_rejects;
    result.tabi_mac_surface_margin_soft_reject_panels =
        tabi_solve.tabi_mac_surface_margin_soft_reject_panels;
    result.tabi_moment_audit_nodes = tabi_solve.tabi_moment_audit_nodes;
    result.tabi_moment_translation_rel_error_max =
        tabi_solve.tabi_moment_translation_rel_error_max;
    result.tabi_moment_translation_rel_error_mean =
        tabi_solve.tabi_moment_translation_rel_error_mean;
    result.tabi_moment_translation_offset_max =
        tabi_solve.tabi_moment_translation_offset_max;
    result.tabi_moment_translation_offset_mean =
        tabi_solve.tabi_moment_translation_offset_mean;
    result.tabi_moment_cancellation_ratio_max =
        tabi_solve.tabi_moment_cancellation_ratio_max;
    result.tabi_moment_cancellation_ratio_mean =
        tabi_solve.tabi_moment_cancellation_ratio_mean;
    result.tabi_moment_dynamic_range_max =
        tabi_solve.tabi_moment_dynamic_range_max;
    result.tabi_expansion_audit = tabi_solve.tabi_expansion_audit;
    result.tabi_expansion_audit_channel =
        tabi_solve.tabi_expansion_audit_channel;
    result.tabi_far_expansion_audit_pairs =
        tabi_solve.tabi_far_expansion_audit_pairs;
    result.tabi_far_expansion_audit_panels =
        tabi_solve.tabi_far_expansion_audit_panels;
    result.tabi_far_expansion_direct_ms =
        tabi_solve.tabi_far_expansion_direct_ms;
    result.tabi_far_expansion_rel_error_max =
        tabi_solve.tabi_far_expansion_rel_error_max;
    result.tabi_far_expansion_rel_error_mean =
        tabi_solve.tabi_far_expansion_rel_error_mean;
    result.tabi_far_expansion_rel_error_p95 =
        tabi_solve.tabi_far_expansion_rel_error_p95;
    result.tabi_far_expansion_rel_error_p99 =
        tabi_solve.tabi_far_expansion_rel_error_p99;
    result.tabi_far_expansion_channel_rel_error_max =
        tabi_solve.tabi_far_expansion_channel_rel_error_max;
    result.tabi_far_expansion_channel_rel_error_mean =
        tabi_solve.tabi_far_expansion_channel_rel_error_mean;
    result.tabi_far_expansion_channel_rel_error_p99 =
        tabi_solve.tabi_far_expansion_channel_rel_error_p99;
    result.tabi_far_expansion_worst_channel =
        tabi_solve.tabi_far_expansion_worst_channel;
    result.tabi_far_expansion_mean_gate_pass =
        tabi_solve.tabi_far_expansion_mean_gate_pass;
    result.tabi_far_convergence_rho_max =
        tabi_solve.tabi_far_convergence_rho_max;
    result.tabi_far_convergence_rho_mean =
        tabi_solve.tabi_far_convergence_rho_mean;
    result.tabi_accepted_far_aspect_ratio_max =
        tabi_solve.tabi_accepted_far_aspect_ratio_max;
    result.tabi_accepted_far_aspect_ratio_mean =
        tabi_solve.tabi_accepted_far_aspect_ratio_mean;
    result.tabi_quad_buffer_auto_candidate =
        tabi_solve.tabi_quad_buffer_auto_candidate;
    result.tabi_leaf_tile_status = tabi_solve.tabi_leaf_tile_status;
    result.tabi_leaf_panel_pairs_per_second =
        tabi_solve.tabi_leaf_panel_pairs_per_second;
    result.tabi_near_integration = tabi_solve.tabi_near_integration;
    result.tabi_quad_buffer = tabi_solve.tabi_quad_buffer;
    result.tabi_near_self_pairs = tabi_solve.tabi_near_self_pairs;
    result.tabi_near_one_ring_pairs = tabi_solve.tabi_near_one_ring_pairs;
    result.tabi_near_duffy_pairs = tabi_solve.tabi_near_duffy_pairs;
    result.tabi_near_centroid_pairs = tabi_solve.tabi_near_centroid_pairs;
    result.tabi_near_fallbacks = tabi_solve.tabi_near_fallbacks;
    result.tabi_near_geometric_buffer_pairs =
        tabi_solve.tabi_near_geometric_buffer_pairs;
    result.tabi_near_geometric_buffer_gauss_pairs =
        tabi_solve.tabi_near_geometric_buffer_gauss_pairs;
    result.tabi_near_geometric_buffer_adaptive_pairs =
        tabi_solve.tabi_near_geometric_buffer_adaptive_pairs;
    result.tabi_near_geometric_buffer_centroid_pairs =
        tabi_solve.tabi_near_geometric_buffer_centroid_pairs;
    result.tabi_near_geometric_buffer_min_gap =
        tabi_solve.tabi_near_geometric_buffer_min_gap;
    result.tabi_near_geometric_buffer_mean_gap_ratio =
        tabi_solve.tabi_near_geometric_buffer_mean_gap_ratio;
    result.tabi_near_geometric_buffer_max_edge_mean =
        tabi_solve.tabi_near_geometric_buffer_max_edge_mean;
    result.tabi_near_integration_ms = tabi_solve.tabi_near_integration_ms;
    result.tabi_duffy_subdivisions_max = tabi_solve.tabi_duffy_subdivisions_max;
    result.tabi_duffy_subdivisions_mean =
        tabi_solve.tabi_duffy_subdivisions_mean;
    result.tabi_near_kernel_policy = tabi_solve.tabi_near_kernel_policy;
    result.tabi_near_k1_accepted = tabi_solve.tabi_near_k1_accepted;
    result.tabi_near_k1_rejected = tabi_solve.tabi_near_k1_rejected;
    result.tabi_near_k2_accepted = tabi_solve.tabi_near_k2_accepted;
    result.tabi_near_k2_rejected = tabi_solve.tabi_near_k2_rejected;
    result.tabi_near_k3_accepted = tabi_solve.tabi_near_k3_accepted;
    result.tabi_near_k3_rejected = tabi_solve.tabi_near_k3_rejected;
    result.tabi_near_k4_accepted = tabi_solve.tabi_near_k4_accepted;
    result.tabi_near_k4_rejected = tabi_solve.tabi_near_k4_rejected;
    result.tabi_near_one_ring_k1_accepted =
        tabi_solve.tabi_near_one_ring_k1_accepted;
    result.tabi_near_one_ring_k1_rejected =
        tabi_solve.tabi_near_one_ring_k1_rejected;
    result.tabi_near_one_ring_k2_accepted =
        tabi_solve.tabi_near_one_ring_k2_accepted;
    result.tabi_near_one_ring_k2_rejected =
        tabi_solve.tabi_near_one_ring_k2_rejected;
    result.tabi_near_one_ring_k3_accepted =
        tabi_solve.tabi_near_one_ring_k3_accepted;
    result.tabi_near_one_ring_k3_rejected =
        tabi_solve.tabi_near_one_ring_k3_rejected;
    result.tabi_near_one_ring_k4_accepted =
        tabi_solve.tabi_near_one_ring_k4_accepted;
    result.tabi_near_one_ring_k4_rejected =
        tabi_solve.tabi_near_one_ring_k4_rejected;
    result.tabi_near_k1_policy = tabi_solve.tabi_near_k1_policy;
    result.tabi_near_k2_policy = tabi_solve.tabi_near_k2_policy;
    result.tabi_near_k3_policy = tabi_solve.tabi_near_k3_policy;
    result.tabi_near_k4_policy = tabi_solve.tabi_near_k4_policy;
    result.tabi_near_block_audit = tabi_solve.tabi_near_block_audit;
    result.tabi_geometric_near_audit = tabi_solve.tabi_geometric_near_audit;
    result.tabi_surface_geometry = tabi_solve.tabi_surface_geometry;
    result.tabi_sphere_mesh_mode = tabi_solve.tabi_sphere_mesh_mode;
    result.tabi_surface_quality_total_area_a2 =
        tabi_solve.tabi_surface_quality_total_area_a2;
    result.tabi_surface_quality_area_rel_error_vs_4piR2 =
        tabi_solve.tabi_surface_quality_area_rel_error_vs_4piR2;
    result.tabi_surface_quality_triangle_area_min =
        tabi_solve.tabi_surface_quality_triangle_area_min;
    result.tabi_surface_quality_triangle_area_mean =
        tabi_solve.tabi_surface_quality_triangle_area_mean;
    result.tabi_surface_quality_triangle_area_max =
        tabi_solve.tabi_surface_quality_triangle_area_max;
    result.tabi_surface_quality_triangle_aspect_max =
        tabi_solve.tabi_surface_quality_triangle_aspect_max;
    result.tabi_surface_quality_normal_radial_dot_min =
        tabi_solve.tabi_surface_quality_normal_radial_dot_min;
    result.tabi_surface_quality_normal_radial_dot_mean =
        tabi_solve.tabi_surface_quality_normal_radial_dot_mean;
    result.tabi_surface_quality_centroid_radial_error_max =
        tabi_solve.tabi_surface_quality_centroid_radial_error_max;
    result.tabi_surface_quality_planar_area_a2 =
        tabi_solve.tabi_surface_quality_planar_area_a2;
    result.tabi_surface_quality_planar_area_rel_error_vs_4piR2 =
        tabi_solve.tabi_surface_quality_planar_area_rel_error_vs_4piR2;
    result.tabi_surface_quality_enclosed_volume_a3 =
        tabi_solve.tabi_surface_quality_enclosed_volume_a3;
    result.tabi_surface_quality_enclosed_volume_rel_error_vs_4piR3_over3 =
        tabi_solve
            .tabi_surface_quality_enclosed_volume_rel_error_vs_4piR3_over3;
    result.tabi_surface_quality_triangle_area_cv =
        tabi_solve.tabi_surface_quality_triangle_area_cv;
    result.tabi_surface_topology_source =
        tabi_solve.tabi_surface_topology_source;
    result.tabi_surface_analytic_ses_engine =
        tabi_solve.tabi_surface_analytic_ses_engine;
    result.tabi_surface_status_detail =
        tabi_solve.tabi_surface_status_detail;
    result.tabi_surface_cavity_policy =
        tabi_solve.tabi_surface_cavity_policy;
    result.tabi_surface_odd_ray_rows =
        tabi_solve.tabi_surface_odd_ray_rows;
    result.tabi_surface_patch_ray_unresolved_uncertified_rows =
        tabi_solve.tabi_surface_patch_ray_unresolved_uncertified_rows;
    result.tabi_surface_morphology_fallback_vertices =
        tabi_solve.tabi_surface_morphology_fallback_vertices;
    result.tabi_surface_analytic_orientation_status =
        tabi_solve.tabi_surface_analytic_orientation_status;
    result.tabi_surface_analytic_inverted_faces =
        tabi_solve.tabi_surface_analytic_inverted_faces;
    result.tabi_surface_analytic_orientation_uncertified_faces =
        tabi_solve.tabi_surface_analytic_orientation_uncertified_faces;
    result.tabi_surface_analytic_self_intersections =
        tabi_solve.tabi_surface_analytic_self_intersections;
    result.tabi_surface_wide_repair_attempts =
        tabi_solve.tabi_surface_wide_repair_attempts;
    result.tabi_surface_topology_certificate_accepts =
        tabi_solve.tabi_surface_topology_certificate_accepts;
    result.tabi_surface_topology_fallback_audits =
        tabi_solve.tabi_surface_topology_fallback_audits;
    result.tabi_surface_initial_collapse_ms =
        tabi_solve.tabi_surface_initial_collapse_ms;
    result.tabi_surface_stable_sliver_ms =
        tabi_solve.tabi_surface_stable_sliver_ms;
    result.tabi_surface_full_projection_ms =
        tabi_solve.tabi_surface_full_projection_ms;
    result.tabi_surface_orientation_repair_ms =
        tabi_solve.tabi_surface_orientation_repair_ms;
    result.tabi_surface_wide_repair_ms =
        tabi_solve.tabi_surface_wide_repair_ms;
    result.tabi_surface_self_intersection_audit_ms =
        tabi_solve.tabi_surface_self_intersection_audit_ms;
    result.tabi_surface_filled_cavity_regions =
        tabi_solve.tabi_surface_filled_cavity_regions;
    result.tabi_surface_filled_cavity_vertices =
        tabi_solve.tabi_surface_filled_cavity_vertices;
    result.tabi_surface_poor_triangles_before =
        tabi_solve.tabi_surface_poor_triangles_before;
    result.tabi_surface_poor_triangles_after =
        tabi_solve.tabi_surface_poor_triangles_after;
    result.tabi_surface_triangle_quality_min_before =
        tabi_solve.tabi_surface_triangle_quality_min_before;
    result.tabi_surface_triangle_quality_min_after =
        tabi_solve.tabi_surface_triangle_quality_min_after;
    result.tabi_surface_max_edge_ratio_before =
        tabi_solve.tabi_surface_max_edge_ratio_before;
    result.tabi_surface_max_edge_ratio_after =
        tabi_solve.tabi_surface_max_edge_ratio_after;
    result.tabi_surface_analytic_audit_status =
        tabi_solve.tabi_surface_analytic_audit_status;
    result.tabi_surface_analytic_audit_samples =
        tabi_solve.tabi_surface_analytic_audit_samples;
    result.tabi_surface_analytic_audit_hits =
        tabi_solve.tabi_surface_analytic_audit_hits;
    result.tabi_surface_analytic_audit_misses =
        tabi_solve.tabi_surface_analytic_audit_misses;
    result.tabi_surface_analytic_projection_distance_rms =
        tabi_solve.tabi_surface_analytic_projection_distance_rms;
    result.tabi_surface_analytic_projection_distance_max =
        tabi_solve.tabi_surface_analytic_projection_distance_max;
    result.tabi_surface_analytic_patch_residual_rms =
        tabi_solve.tabi_surface_analytic_patch_residual_rms;
    result.tabi_surface_analytic_patch_residual_max =
        tabi_solve.tabi_surface_analytic_patch_residual_max;
    result.tabi_surface_analytic_normal_angle_rms_deg =
        tabi_solve.tabi_surface_analytic_normal_angle_rms_deg;
    result.tabi_surface_analytic_normal_angle_max_deg =
        tabi_solve.tabi_surface_analytic_normal_angle_max_deg;
    result.tabi_surface_sampled_hausdorff_proxy =
        tabi_solve.tabi_surface_sampled_hausdorff_proxy;
    result.tabi_surface_analytic_normal_quality_status =
        tabi_solve.tabi_surface_analytic_normal_quality_status;
    result.tabi_surface_analytic_orientation_smooth_faces =
        tabi_solve.tabi_surface_analytic_orientation_smooth_faces;
    result.tabi_surface_analytic_orientation_seam_faces =
        tabi_solve.tabi_surface_analytic_orientation_seam_faces;
    result.tabi_surface_analytic_normal_cosine_below_025_faces =
        tabi_solve.tabi_surface_analytic_normal_cosine_below_025_faces;
    result.tabi_surface_analytic_normal_cosine_below_050_faces =
        tabi_solve.tabi_surface_analytic_normal_cosine_below_050_faces;
    result.tabi_surface_analytic_normal_area_weighted_rms_deg =
        tabi_solve.tabi_surface_analytic_normal_area_weighted_rms_deg;
    result.tabi_surface_analytic_normal_p99_deg =
        tabi_solve.tabi_surface_analytic_normal_p99_deg;
    result.tabi_surface_analytic_smooth_normal_p99_deg =
        tabi_solve.tabi_surface_analytic_smooth_normal_p99_deg;
    result.tabi_surface_analytic_seam_normal_p99_deg =
        tabi_solve.tabi_surface_analytic_seam_normal_p99_deg;
    result.tabi_surface_analytic_normal_area_below_cosine_025_fraction =
        tabi_solve
            .tabi_surface_analytic_normal_area_below_cosine_025_fraction;
    result.tabi_surface_analytic_normal_area_below_cosine_050_fraction =
        tabi_solve
            .tabi_surface_analytic_normal_area_below_cosine_050_fraction;
    result.tabi_calderon_trace_space_status =
        tabi_solve.tabi_calderon_trace_space_status;
    result.tabi_calderon_trace_vertices =
        tabi_solve.tabi_calderon_trace_vertices;
    result.tabi_calderon_trace_edges = tabi_solve.tabi_calderon_trace_edges;
    result.tabi_calderon_trace_boundary_edges =
        tabi_solve.tabi_calderon_trace_boundary_edges;
    result.tabi_calderon_trace_orientation_mismatches =
        tabi_solve.tabi_calderon_trace_orientation_mismatches;
    result.tabi_calderon_trace_nonmanifold_edges =
        tabi_solve.tabi_calderon_trace_nonmanifold_edges;
    result.tabi_calderon_trace_unreferenced_vertices =
        tabi_solve.tabi_calderon_trace_unreferenced_vertices;
    result.tabi_calderon_trace_components =
        tabi_solve.tabi_calderon_trace_components;
    result.tabi_calderon_trace_euler_characteristic =
        tabi_solve.tabi_calderon_trace_euler_characteristic;
    result.tabi_calderon_trace_genus_sum =
        tabi_solve.tabi_calderon_trace_genus_sum;
    result.tabi_calderon_trace_topology_invariants_valid =
        tabi_solve.tabi_calderon_trace_topology_invariants_valid;
    result.tabi_calderon_aux_surface_mode =
        tabi_solve.tabi_calderon_aux_surface_mode;
    result.tabi_calderon_aux_trace_status =
        tabi_solve.tabi_calderon_aux_trace_status;
    result.tabi_calderon_aux_panels = tabi_solve.tabi_calderon_aux_panels;
    result.tabi_calderon_aux_vertices = tabi_solve.tabi_calderon_aux_vertices;
    result.tabi_calderon_aux_edges = tabi_solve.tabi_calderon_aux_edges;
    result.tabi_calderon_aux_boundary_edges =
        tabi_solve.tabi_calderon_aux_boundary_edges;
    result.tabi_calderon_aux_orientation_mismatches =
        tabi_solve.tabi_calderon_aux_orientation_mismatches;
    result.tabi_calderon_aux_nonmanifold_edges =
        tabi_solve.tabi_calderon_aux_nonmanifold_edges;
    result.tabi_calderon_aux_unreferenced_vertices =
        tabi_solve.tabi_calderon_aux_unreferenced_vertices;
    result.tabi_calderon_aux_grid_vertices =
        tabi_solve.tabi_calderon_aux_grid_vertices;
    result.tabi_calderon_aux_area_a2 = tabi_solve.tabi_calderon_aux_area_a2;
    result.tabi_calderon_aux_build_ms = tabi_solve.tabi_calderon_aux_build_ms;
    result.tabi_mesh_vertices = tabi_solve.tabi_mesh_vertices;
    result.tabi_mesh_triangles = tabi_solve.tabi_mesh_triangles;
    result.tabi_boundary_nodes = tabi_solve.tabi_boundary_nodes;
    result.tabi_avg_valence = tabi_solve.tabi_avg_valence;
    result.tabi_nonmanifold_edges = tabi_solve.tabi_nonmanifold_edges;
    result.tabi_solid_angle_fallbacks = tabi_solve.tabi_solid_angle_fallbacks;
    result.tabi_solid_angle_clamps = tabi_solve.tabi_solid_angle_clamps;
    result.tabi_diag_min = tabi_solve.tabi_diag_min;
    result.tabi_diag_max = tabi_solve.tabi_diag_max;
    result.tabi_taylor_order = tabi_solve.tabi_taylor_order;
    result.tabi_derivative_mode = tabi_solve.tabi_derivative_mode;
    result.tabi_interaction_build_ms = tabi_solve.tabi_interaction_build_ms;
    result.tabi_tree_eval_ms = tabi_solve.tabi_tree_eval_ms;
    result.tabi_formulation = tabi_solve.tabi_formulation;
    result.tabi_energy_projection = tabi_solve.tabi_energy_projection;
    result.tabi_energy_projection_ms = tabi_solve.tabi_energy_projection_ms;
    result.tabi_energy_projection_kernel =
        tabi_solve.tabi_energy_projection_kernel;
    result.tabi_preconditioner_requested =
        tabi_solve.tabi_preconditioner_requested;
    result.tabi_preconditioner_effective =
        tabi_solve.tabi_preconditioner_effective;
    result.tabi_preconditioner_block_near =
        tabi_solve.tabi_preconditioner_block_near;
    result.tabi_preconditioner_block_panel_cap =
        tabi_solve.tabi_preconditioner_block_panel_cap;
    result.tabi_preconditioner_overlap_panel_cap =
        tabi_solve.tabi_preconditioner_overlap_panel_cap;
    result.tabi_preconditioner_ilu_neighbor_cap_requested =
        tabi_solve.tabi_preconditioner_ilu_neighbor_cap_requested;
    result.tabi_preconditioner_ilu_neighbor_cap =
        tabi_solve.tabi_preconditioner_ilu_neighbor_cap;
    result.tabi_preconditioner_ilu_drop_tol =
        tabi_solve.tabi_preconditioner_ilu_drop_tol;
    result.tabi_preconditioner_ilu_regularization =
        tabi_solve.tabi_preconditioner_ilu_regularization;
    result.tabi_preconditioner_ilu_regularization_max =
        tabi_solve.tabi_preconditioner_ilu_regularization_max;
    result.tabi_preconditioner_ilu_regularization_selected =
        tabi_solve.tabi_preconditioner_ilu_regularization_selected;
    result.tabi_preconditioner_ilu_regularization_absolute_max =
        tabi_solve.tabi_preconditioner_ilu_regularization_absolute_max;
    result.tabi_preconditioner_ilu_regularization_attempts =
        tabi_solve.tabi_preconditioner_ilu_regularization_attempts;
    result.tabi_preconditioner_ilu_regularization_rejected_attempts =
        tabi_solve.tabi_preconditioner_ilu_regularization_rejected_attempts;
    result.tabi_preconditioner_ilu_regularization_shifted_rows =
        tabi_solve.tabi_preconditioner_ilu_regularization_shifted_rows;
    result.tabi_preconditioner_ilu_unshifted_failure_stage =
        tabi_solve.tabi_preconditioner_ilu_unshifted_failure_stage;
    result.tabi_preconditioner_memory_budget_bytes =
        tabi_solve.tabi_preconditioner_memory_budget_bytes;
    result.tabi_preconditioner_estimated_bytes =
        tabi_solve.tabi_preconditioner_estimated_bytes;
    result.tabi_preconditioner_valid_blocks =
        tabi_solve.tabi_preconditioner_valid_blocks;
    result.tabi_preconditioner_valid_panels =
        tabi_solve.tabi_preconditioner_valid_panels;
    result.tabi_preconditioner_ill_conditioned_blocks =
        tabi_solve.tabi_preconditioner_ill_conditioned_blocks;
    result.tabi_preconditioner_quality_rejected_blocks =
        tabi_solve.tabi_preconditioner_quality_rejected_blocks;
    result.tabi_preconditioner_quality_probe_max_gain =
        tabi_solve.tabi_preconditioner_quality_probe_max_gain;
    result.tabi_preconditioner_quality_probe_rel_residual =
        tabi_solve.tabi_preconditioner_quality_probe_rel_residual;
    result.tabi_preconditioner_pivot_ratio_min =
        tabi_solve.tabi_preconditioner_pivot_ratio_min;
    result.tabi_preconditioner_factor_growth_max =
        tabi_solve.tabi_preconditioner_factor_growth_max;
    result.tabi_preconditioner_pivot_ratio_min_row =
        tabi_solve.tabi_preconditioner_pivot_ratio_min_row;
    result.tabi_preconditioner_pivot_ratio_min_column =
        tabi_solve.tabi_preconditioner_pivot_ratio_min_column;
    result.tabi_preconditioner_factor_growth_row =
        tabi_solve.tabi_preconditioner_factor_growth_row;
    result.tabi_preconditioner_factor_growth_column =
        tabi_solve.tabi_preconditioner_factor_growth_column;
    result.tabi_preconditioner_quality_failure_stage =
        tabi_solve.tabi_preconditioner_quality_failure_stage;
    result.tabi_preconditioner_quality_failure_row =
        tabi_solve.tabi_preconditioner_quality_failure_row;
    result.tabi_preconditioner_quality_failure_column =
        tabi_solve.tabi_preconditioner_quality_failure_column;
    result.tabi_preconditioner_quality_failure_probe =
        tabi_solve.tabi_preconditioner_quality_failure_probe;
    result.tabi_preconditioner_quality_failure_probe_gain =
        tabi_solve.tabi_preconditioner_quality_failure_probe_gain;
    result.tabi_preconditioner_quality_failure_probe_rel_residual =
        tabi_solve.tabi_preconditioner_quality_failure_probe_rel_residual;
    result.tabi_preconditioner_quality_status =
        tabi_solve.tabi_preconditioner_quality_status;
    result.tabi_preconditioner_setup_ms =
        tabi_solve.tabi_preconditioner_setup_ms;
    result.tabi_preconditioner_graph_setup_ms =
        tabi_solve.tabi_preconditioner_graph_setup_ms;
    result.tabi_preconditioner_graph_cache_reused =
        tabi_solve.tabi_preconditioner_graph_cache_reused;
    result.tabi_preconditioner_near_assembly_setup_ms =
        tabi_solve.tabi_preconditioner_near_assembly_setup_ms;
    result.tabi_preconditioner_ilu_factor_setup_ms =
        tabi_solve.tabi_preconditioner_ilu_factor_setup_ms;
    result.tabi_preconditioner_apply_ms =
        tabi_solve.tabi_preconditioner_apply_ms;
    result.tabi_preconditioner_apply_calls =
        tabi_solve.tabi_preconditioner_apply_calls;
    result.tabi_preconditioner_sparse_nnz_blocks =
        tabi_solve.tabi_preconditioner_sparse_nnz_blocks;
    result.tabi_preconditioner_sparse_avg_degree =
        tabi_solve.tabi_preconditioner_sparse_avg_degree;
    result.tabi_preconditioner_sparse_max_degree =
        tabi_solve.tabi_preconditioner_sparse_max_degree;
    result.tabi_preconditioner_coarse_aggregate_cap =
        tabi_solve.tabi_preconditioner_coarse_aggregate_cap;
    result.tabi_preconditioner_coarse_aggregates =
        tabi_solve.tabi_preconditioner_coarse_aggregates;
    result.tabi_preconditioner_coarse_dimension =
        tabi_solve.tabi_preconditioner_coarse_dimension;
    result.tabi_preconditioner_coarse_base =
        tabi_solve.tabi_preconditioner_coarse_base;
    result.tabi_preconditioner_coarse_matrix_bytes =
        tabi_solve.tabi_preconditioner_coarse_matrix_bytes;
    result.tabi_preconditioner_coarse_pivot_ratio =
        tabi_solve.tabi_preconditioner_coarse_pivot_ratio;
    result.tabi_preconditioner_coarse_status =
        tabi_solve.tabi_preconditioner_coarse_status;
    result.tabi_preconditioner_coarse_setup_matvec_calls =
        tabi_solve.tabi_preconditioner_coarse_setup_matvec_calls;
    result.tabi_preconditioner_coarse_setup_ms =
        tabi_solve.tabi_preconditioner_coarse_setup_ms;
    result.tabi_preconditioner_coarse_moments_setup_ms =
        tabi_solve.tabi_preconditioner_coarse_moments_setup_ms;
    result.tabi_preconditioner_coarse_far_setup_ms =
        tabi_solve.tabi_preconditioner_coarse_far_setup_ms;
    result.tabi_preconditioner_coarse_audit_setup_ms =
        tabi_solve.tabi_preconditioner_coarse_audit_setup_ms;
    result.tabi_preconditioner_coarse_apply_ms =
        tabi_solve.tabi_preconditioner_coarse_apply_ms;
    result.tabi_preconditioner_coarse_apply_calls =
        tabi_solve.tabi_preconditioner_coarse_apply_calls;
    result.tabi_preconditioner_coarse_rp_identity_error_max =
        tabi_solve.tabi_preconditioner_coarse_rp_identity_error_max;
    result.tabi_preconditioner_coarse_verify_rel_error_max =
        tabi_solve.tabi_preconditioner_coarse_verify_rel_error_max;
    result.tabi_surface_generator = tabi_solve.tabi_surface_generator;
    result.tabi_discretization = tabi_solve.tabi_discretization;
    result.tabi_node_patch_representation =
        tabi_solve.tabi_node_patch_representation;
    result.tabi_node_patch_normal_source =
        tabi_solve.tabi_node_patch_normal_source;
    result.tabi_node_patch_certificate =
        tabi_solve.tabi_node_patch_certificate;
    if (endpoint_cycle) {
      result.tabi_receptor_node_patch_certificate =
          tabi_receptor_solve.tabi_node_patch_certificate;
      result.tabi_ligand_node_patch_certificate =
          tabi_ligand_solve.tabi_node_patch_certificate;
    }
    result.tabi_rhs_scale = tabi_solve.tabi_rhs_scale;
    result.tabi_dielectric_contrast = tabi_solve.tabi_dielectric_contrast;
    result.tabi_operator_contrast = tabi_solve.tabi_operator_contrast;
    result.tabi_unknown_count = tabi_solve.tabi_unknown_count;
    result.tabi_bie_system = tabi_solve.tabi_bie_system;
    result.tabi_lhs_diag_phi = tabi_solve.tabi_lhs_diag_phi;
    result.tabi_lhs_diag_phin = tabi_solve.tabi_lhs_diag_phin;
    result.tabi_energy_equation = tabi_solve.tabi_energy_equation;
    result.tabi_kernel_convention = tabi_solve.tabi_kernel_convention;
    result.tabi_source_norm_s1 = tabi_solve.tabi_source_norm_s1;
    result.tabi_source_norm_s2 = tabi_solve.tabi_source_norm_s2;
    result.tabi_phi_norm = tabi_solve.tabi_phi_norm;
    result.tabi_phin_norm = tabi_solve.tabi_phin_norm;
    result.tabi_kappa_inv_a = tabi_solve.tabi_kappa_inv_a;
    result.tabi_coulomb_factor_kcal_mol =
        tabi_solve.tabi_coulomb_factor_kcal_mol;
    result.tabi_boundary_energy_kj_mol = tabi_solve.tabi_boundary_energy_kj_mol;
    result.tabi_reference_energy_kj_mol =
        tabi_solve.tabi_reference_energy_kj_mol;
    result.tabi_selected_energy_scale = tabi_solve.tabi_selected_energy_scale;
    result.tabi_fallback_used = tabi_solve.tabi_fallback_used;
    result.pbe_model_effective = tabi_solve.pbe_model_effective;
    result.linear_seed_solver = tabi_solve.linear_seed_solver;
    result.focus_parent_grid = tabi_solve.focus_parent_grid;
    result.probe_status = "off";
    result.rhs_scale = tabi_solve.rhs_scale;
    result.charge_assigned = tabi_solve.charge_assigned;
    result.charge_moment_error = tabi_solve.charge_moment_error;
    result.boundary_rms = tabi_solve.boundary_rms;
    result.boundary_max = tabi_solve.boundary_max;
    result.linear_residual = tabi_solve.linear_residual;
    result.nl_status = "linear";
    result.focus_enabled = false;
    result.focus_levels = 1;
    result.solver_mode = tabi_solve.solver_mode;
    result.mg_levels = 1;
    result.solvent_iterations = tabi_solve.solvent_total_iterations;
    result.vacuum_iterations = tabi_solve.vacuum_total_iterations;
    result.solvent_iterations_final = tabi_solve.solvent.iterations;
    result.vacuum_iterations_final = tabi_solve.vacuum.iterations;
    result.solvent_rms = tabi_solve.solvent.rms;
    result.vacuum_rms = tabi_solve.vacuum.rms;
    result.solvent_max_dphi = tabi_solve.solvent.max_dphi;
    result.vacuum_max_dphi = tabi_solve.vacuum.max_dphi;
    result.solvent_converged = tabi_solve.solvent.converged;
    result.vacuum_converged = tabi_solve.vacuum.converged;
    result.energy_kcal_mol = tabi_solve.energy_kcal_mol;
    result.energy_kj_mol = tabi_solve.energy_kj_mol;
    result.energy_kt = tabi_solve.energy_kt;
    result.solvent_phase_kcal_mol = tabi_solve.solvent_phase_kcal_mol;
    result.solvent_phase_kj_mol = tabi_solve.solvent_phase_kj_mol;
    result.solvent_phase_kt = tabi_solve.solvent_phase_kt;
    result.atom_corrected_kcal_mol = tabi_solve.atom_corrected_kcal_mol;
    result.atom_corrected_kj_mol = tabi_solve.atom_corrected_kj_mol;
    result.atom_corrected_kt = tabi_solve.atom_corrected_kt;
    result.grid_corrected_kcal_mol = tabi_solve.grid_corrected_kcal_mol;
    result.grid_corrected_kj_mol = tabi_solve.grid_corrected_kj_mol;
    result.grid_corrected_kt = tabi_solve.grid_corrected_kt;
    result.coulomb_energy_kcal_mol = tabi_solve.coulomb_energy_kcal_mol;
    result.coulomb_energy_kj_mol = tabi_solve.coulomb_energy_kj_mol;
    result.coulomb_energy_kt = tabi_solve.coulomb_energy_kt;
    result.reaction_energy_source = tabi_solve.reaction_energy_source;
    result.complex_energy_kcal_mol = tabi_solve.energy_kcal_mol;
    result.complex_energy_kj_mol = tabi_solve.energy_kj_mol;
    if (args_.dgsolv && !binding_cycle && !native_pbsa_cycle) {
      // TABI solves directly for the reaction field. By the standard PB/BIE
      // definition, Delta G_solv = 1/2 sum_i q_i phi_reac(r_i), so no second
      // vacuum solve or solvent-minus-vacuum subtraction is required here.
      result.dgsolv_enabled = true;
      result.deltaG_solv_kcal_mol = tabi_solve.energy_kcal_mol;
      result.deltaG_solv_kj_mol = tabi_solve.energy_kj_mol;
      result.deltaG_solv_kt = tabi_solve.energy_kt;
      result.dgsolv_status = tabi_solve.converged ? "ok" : "not_converged";
    }
    if (endpoint_cycle) {
      result.dgsolv_enabled = true;
      result.mol1_energy_kcal_mol = tabi_receptor_solve.energy_kcal_mol;
      result.mol1_energy_kj_mol = tabi_receptor_solve.energy_kj_mol;
      result.mol2_energy_kcal_mol = tabi_ligand_solve.energy_kcal_mol;
      result.mol2_energy_kj_mol = tabi_ligand_solve.energy_kj_mol;
      result.deltaG_solv_kcal_mol = tabi_solve.energy_kcal_mol -
                                    tabi_receptor_solve.energy_kcal_mol -
                                    tabi_ligand_solve.energy_kcal_mol;
      result.deltaG_solv_kj_mol = result.deltaG_solv_kcal_mol * kKcalToKJ;
      result.deltaG_solv_kt = result.deltaG_solv_kj_mol /
                              (kKBKJPerMolK * std::max(1.0, args_.temp));
      result.dgsolv_status =
          (tabi_solve.converged && tabi_receptor_solve.converged &&
           tabi_ligand_solve.converged)
              ? "ok"
              : "not_converged";
      result.memory_bytes =
          std::max({result.memory_bytes, tabi_receptor_solve.peak_memory_bytes,
                    tabi_ligand_solve.peak_memory_bytes});
    }
    if (native_pbsa_cycle) {
      result.pbsa_time_complex_ms =
          elapsed_ms(tabi_complex_time_begin, tabi_complex_time_end);
      result.pbsa_grid_reuse_effective = "not_applicable";
      const auto components_time_begin = std::chrono::steady_clock::now();
      const PbsaComponents pbsa = compute_pbsa_components(
          mol1, params1, *mol2, *params2, mol1_atoms, mol2_atoms, atoms_,
          result.deltaG_solv_kj_mol, executor, energy_provider);
      const auto components_time_end = std::chrono::steady_clock::now();
      result.pbsa_time_components_ms =
          elapsed_ms(components_time_begin, components_time_end);
      result.pbsa_enabled = true;
      const bool endpoints_converged =
          tabi_solve.converged && tabi_receptor_solve.converged &&
          tabi_ligand_solve.converged;
      result.pbsa_status = !endpoints_converged
                               ? "not_converged"
                               : (pbsa.status == "ok" ? "ok" : pbsa.status);
      result.pbsa_elec_kj_mol = pbsa.elec_kj_mol;
      result.pbsa_vdw_kj_mol = pbsa.vdw_kj_mol;
      result.pbsa_mm_kj_mol = pbsa.mm_kj_mol;
      result.pbsa_sasa_complex_a2 = pbsa.sasa_complex_a2;
      result.pbsa_sasa_mol1_a2 = pbsa.sasa_mol1_a2;
      result.pbsa_sasa_mol2_a2 = pbsa.sasa_mol2_a2;
      result.pbsa_delta_sasa_a2 = pbsa.delta_sasa_a2;
      result.pbsa_nonpolar_kj_mol = pbsa.nonpolar_kj_mol;
      result.pbsa_delta_solv_kj_mol = pbsa.delta_solv_kj_mol;
      result.pbsa_delta_solv_kcal_mol = pbsa.delta_solv_kj_mol * kKJToKcal;
      result.pbsa_no_entropy_kj_mol = pbsa.no_entropy_kj_mol;
      result.pbsa_no_entropy_kcal_mol = pbsa.no_entropy_kj_mol * kKJToKcal;
      result.pbsa_receptor_atom_count = mol1_atoms.size();
      result.pbsa_ligand_atom_count = mol2_atoms.size();
      result.pbsa_complex_status =
          tabi_solve.converged ? "ok" : "not_converged";
      result.pbsa_receptor_status =
          tabi_receptor_solve.converged ? "ok" : "not_converged";
      result.pbsa_ligand_status =
          tabi_ligand_solve.converged ? "ok" : "not_converged";
      result.pbsa_receptor_solvent_iterations =
          tabi_receptor_solve.solvent_total_iterations;
      result.pbsa_receptor_vacuum_iterations = 0;
      result.pbsa_ligand_solvent_iterations =
          tabi_ligand_solve.solvent_total_iterations;
      result.pbsa_ligand_vacuum_iterations = 0;
      result.pbsa_receptor_solvent_rms = tabi_receptor_solve.solvent.rms;
      result.pbsa_receptor_vacuum_rms = 0.0;
      result.pbsa_ligand_solvent_rms = tabi_ligand_solve.solvent.rms;
      result.pbsa_ligand_vacuum_rms = 0.0;
      result.pbsa_receptor_solvent_max_dphi =
          tabi_receptor_solve.solvent.max_dphi;
      result.pbsa_receptor_vacuum_max_dphi = 0.0;
      result.pbsa_ligand_solvent_max_dphi =
          tabi_ligand_solve.solvent.max_dphi;
      result.pbsa_ligand_vacuum_max_dphi = 0.0;

      if (args_.pbsa_perf == "detail") {
        auto append_tabi_endpoint =
            [&](const std::string &endpoint, const ReactionSolve &solve,
                double elapsed, const std::string &status) {
              PbsaEndpointTimingRow row;
              row.trajectory_id = result.trajectory_id;
              row.frame_index = result.frame_index;
              row.global_index = result.global_index;
              row.time_ps = result.time_ps;
              row.endpoint = endpoint;
              row.phase = "reaction_field";
              row.variant = "baseline";
              row.timing.surface_ms = solve.tabi_surface_build_ms;
              row.timing.solver_ms = solve.tabi_gmres_ms;
              row.timing.setup_ms = std::max(
                  0.0, elapsed - row.timing.surface_ms - row.timing.solver_ms);
              row.reuse_surface_hits = 0;
              row.reuse_operator_hits = 0;
              row.status = status;
              pbsa_endpoint_timing_rows_.push_back(std::move(row));
            };
        append_tabi_endpoint("complex", tabi_solve,
                             result.pbsa_time_complex_ms,
                             result.pbsa_complex_status);
        append_tabi_endpoint("receptor", tabi_receptor_solve,
                             result.pbsa_time_receptor_ms,
                             result.pbsa_receptor_status);
        append_tabi_endpoint("ligand", tabi_ligand_solve,
                             result.pbsa_time_ligand_ms,
                             result.pbsa_ligand_status);
      }

      const std::string pbsa_decomp = effective_pbsa_decomp();
      const bool legacy_pbsa_decomp =
          pbsa_decomp == "residue" || pbsa_decomp == "hybrid";
      if ((legacy_pbsa_decomp || edecomp_sink_) &&
          result.pbsa_status == "ok") {
        const auto residue_time_begin = std::chrono::steady_clock::now();
        const std::vector<ResidueFrameRow> residue_rows =
            compute_pbsa_residue_rows(
                mol1, *mol2, mol1_atoms, mol2_atoms, atoms_, tabi_solve,
                tabi_receptor_solve, tabi_ligand_solve, result, executor,
                energy_provider);
        const auto residue_time_end = std::chrono::steady_clock::now();
        result.pbsa_time_residue_ms =
            elapsed_ms(residue_time_begin, residue_time_end);
        if (legacy_pbsa_decomp)
          accumulate_pbsa_residue_rows(result, residue_rows);
        if (edecomp_sink_) {
          std::vector<EDecompRow> sink_rows;
          sink_rows.reserve(residue_rows.size());
          for (const ResidueFrameRow &row : residue_rows) {
            EDecompRow sink_row;
            sink_row.molecule = row.key.molecule;
            sink_row.unit_type = row.key.unit_type;
            sink_row.chain = row.key.chain;
            sink_row.resnr = row.key.resnr;
            sink_row.respos = row.key.respos;
            sink_row.resname = row.key.resname;
            sink_row.atom_index = row.key.atom_index;
            sink_row.atom_name = row.key.atom_name;
            sink_row.elec_kj_mol = row.elec_kj_mol;
            sink_row.vdw_kj_mol = row.vdw_kj_mol;
            sink_row.polar_kj_mol = row.polar_kj_mol;
            sink_row.nonpolar_kj_mol = row.nonpolar_kj_mol;
            sink_rows.push_back(std::move(sink_row));
          }
          edecomp_sink_(sink_rows);
        }
      }
    }
    result.status =
        result.tabi_status == "off" || result.tabi_status.empty()
            ? ((result.solvent_converged && result.vacuum_converged) ? "ok"
                                                                     : "failed")
            : result.tabi_status;
    if (result.dgsolv_enabled && result.dgsolv_status != "ok")
      result.status = "not_converged";
    if (result.pbsa_enabled && result.pbsa_status != "ok")
      result.status = "not_converged";
    result.pbsa_time_frame_ms =
        elapsed_ms(frame_time_begin, std::chrono::steady_clock::now());
    const ProcessMemorySample frame_memory_after = sample_process_memory();
    result.memory_rss_before_bytes = frame_memory_before.working_set_bytes;
    result.memory_rss_after_bytes = frame_memory_after.working_set_bytes;
    result.memory_rss_delta_bytes =
        frame_memory_after.working_set_bytes >=
                frame_memory_before.working_set_bytes
            ? frame_memory_after.working_set_bytes -
                  frame_memory_before.working_set_bytes
            : 0;
    result.memory_rss_delta_signed_bytes =
        static_cast<std::int64_t>(frame_memory_after.working_set_bytes) -
        static_cast<std::int64_t>(frame_memory_before.working_set_bytes);
    result.memory_peak_working_set_bytes =
        frame_memory_after.peak_working_set_bytes;
    maybe_write_tabi_surface_solution(result, tabi_solve);
    maybe_write_atom_reaction_potentials(result, atoms_, tabi_solve, "complex");
    if (endpoint_cycle) {
      maybe_write_atom_reaction_potentials(result, mol1_atoms,
                                           tabi_receptor_solve, "receptor");
      maybe_write_atom_reaction_potentials(result, mol2_atoms,
                                           tabi_ligand_solve, "ligand");
    }
    if (binding_cycle) {
      maybe_write_binding_electrostatics(result, atoms_, mol1_atoms, mol2_atoms,
                                          tabi_solve, tabi_receptor_solve,
                                          tabi_ligand_solve);
    }
    append_energy_row(result);
    frame_timing.total_ms =
        elapsed_ms(frame_time_begin, std::chrono::steady_clock::now());
    append_timing_row(result, frame_timing);
    append_memory_profile_row(result, frame_timing);
    append_tabi_audit_profile_row(result, frame_timing);
    append_tabi_profile_row(result, frame_timing);
    append_tabi_formulation_audit_row(result);
    append_tabi_near_block_audit_rows(result);
    append_tabi_geometric_near_audit_rows(result);
    append_tabi_far_geometric_audit_row(result);
    append_tabi_moment_audit_row(result);
    append_tabi_surface_quality_row(result);

    ++frames_total_;
    if (result.status == "ok")
      ++frames_converged_;
    else
      ++frames_failed_;
    max_grid_points_seen_ = std::max(max_grid_points_seen_, result.grid_points);
    max_memory_bytes_seen_ =
        std::max(max_memory_bytes_seen_, result.memory_bytes);
    max_tabi_surface_panels_seen_ =
        std::max(max_tabi_surface_panels_seen_, result.tabi_surface_panels);
    max_tabi_tree_nodes_seen_ =
        std::max(max_tabi_tree_nodes_seen_, result.tabi_tree_nodes);
    max_tabi_tree_leaf_count_seen_ =
        std::max(max_tabi_tree_leaf_count_seen_, result.tabi_tree_leaf_count);
    max_tabi_near_interactions_seen_ = std::max(
        max_tabi_near_interactions_seen_, result.tabi_near_interactions);
    max_tabi_far_interactions_seen_ =
        std::max(max_tabi_far_interactions_seen_, result.tabi_far_interactions);
    max_tabi_mesh_vertices_seen_ =
        std::max(max_tabi_mesh_vertices_seen_, result.tabi_mesh_vertices);
    max_tabi_mesh_triangles_seen_ =
        std::max(max_tabi_mesh_triangles_seen_, result.tabi_mesh_triangles);
    max_tabi_boundary_nodes_seen_ =
        std::max(max_tabi_boundary_nodes_seen_, result.tabi_boundary_nodes);
    max_tabi_nonmanifold_edges_seen_ = std::max(
        max_tabi_nonmanifold_edges_seen_, result.tabi_nonmanifold_edges);
    max_tabi_solid_angle_fallbacks_seen_ =
        std::max(max_tabi_solid_angle_fallbacks_seen_,
                 result.tabi_solid_angle_fallbacks);
    min_tabi_diag_seen_ = std::min(min_tabi_diag_seen_, result.tabi_diag_min);
    max_tabi_diag_seen_ = std::max(max_tabi_diag_seen_, result.tabi_diag_max);
    max_solute_points_seen_ =
        std::max(max_solute_points_seen_, result.solute_points);
    max_solvent_points_seen_ =
        std::max(max_solvent_points_seen_, result.solvent_points);
    max_ion_accessible_points_seen_ =
        std::max(max_ion_accessible_points_seen_, result.ion_accessible_points);
    max_stern_excluded_points_seen_ =
        std::max(max_stern_excluded_points_seen_, result.stern_excluded_points);
    max_focus_levels_seen_ =
        std::max(max_focus_levels_seen_, result.focus_levels);
    max_mg_levels_seen_ = std::max(max_mg_levels_seen_, result.mg_levels);
    max_mg_cycles_seen_ = std::max(max_mg_cycles_seen_, result.mg_cycles);
    max_linear_residual_seen_ =
        std::max(max_linear_residual_seen_, result.linear_residual);
    max_boundary_rms_seen_ =
        std::max(max_boundary_rms_seen_, result.boundary_rms);
    max_boundary_max_seen_ =
        std::max(max_boundary_max_seen_, result.boundary_max);

    const double frame_n = static_cast<double>(frames_total_);
    // A failed BIE solve has a NaN energy (analyzer_reaction.cpp); the energy
    // statistics cover the frames that have one.
    if (std::isfinite(result.energy_kj_mol)) {
      ++energy_frames_;
      const double energy_n = static_cast<double>(energy_frames_);
      const double delta = result.energy_kj_mol - energy_mean_;
      energy_mean_ += delta / energy_n;
      energy_m2_ += delta * (result.energy_kj_mol - energy_mean_);
      if (energy_frames_ == 1) {
        energy_min_ = result.energy_kj_mol;
        energy_max_ = result.energy_kj_mol;
      } else {
        energy_min_ = std::min(energy_min_, result.energy_kj_mol);
        energy_max_ = std::max(energy_max_, result.energy_kj_mol);
      }
    }
    iter_mean_ +=
        (static_cast<double>(result.solvent_iterations) - iter_mean_) / frame_n;
    mg_cycle_mean_ +=
        (static_cast<double>(result.mg_cycles) - mg_cycle_mean_) / frame_n;
    if (result.dgsolv_enabled && std::isfinite(result.deltaG_solv_kj_mol)) {
      ++dgsolv_frames_;
      const double dg_n = static_cast<double>(dgsolv_frames_);
      const double dg_delta = result.deltaG_solv_kj_mol - dgsolv_mean_;
      dgsolv_mean_ += dg_delta / dg_n;
      dgsolv_m2_ += dg_delta * (result.deltaG_solv_kj_mol - dgsolv_mean_);
      if (dgsolv_frames_ == 1) {
        dgsolv_min_ = result.deltaG_solv_kj_mol;
        dgsolv_max_ = result.deltaG_solv_kj_mol;
      } else {
        dgsolv_min_ = std::min(dgsolv_min_, result.deltaG_solv_kj_mol);
        dgsolv_max_ = std::max(dgsolv_max_, result.deltaG_solv_kj_mol);
      }
    }
    accumulate_pbsa(result);
    if (log.is_open() && result.status != "ok") {
      TitanUI::log_kv(log, "PBE Warning",
                      "frame " + std::to_string(result.global_index) +
                          " did not meet convergence thresholds");
    }
    return result;
  }

  const auto complex_time_begin = std::chrono::steady_clock::now();
  const ReactionSolve complex_solve =
      solve_reaction_field(atoms_, executor, &result);
  const auto complex_time_end = std::chrono::steady_clock::now();
  result.nx = complex_solve.shape.nx;
  result.ny = complex_solve.shape.ny;
  result.nz = complex_solve.shape.nz;
  result.grid_points = complex_solve.shape.n;
  result.memory_bytes = std::max(complex_solve.shape.memory_bytes,
                                 complex_solve.peak_memory_bytes);
  result.grid_origin_x_A = complex_solve.shape.ox;
  result.grid_origin_y_A = complex_solve.shape.oy;
  result.grid_origin_z_A = complex_solve.shape.oz;
  result.grid_spacing_A = complex_solve.shape.h;
  result.grid_center_x_A = complex_solve.shape.ox +
                           0.5 *
                               static_cast<double>(complex_solve.shape.nx - 1) *
                               complex_solve.shape.h;
  result.grid_center_y_A = complex_solve.shape.oy +
                           0.5 *
                               static_cast<double>(complex_solve.shape.ny - 1) *
                               complex_solve.shape.h;
  result.grid_center_z_A = complex_solve.shape.oz +
                           0.5 *
                               static_cast<double>(complex_solve.shape.nz - 1) *
                               complex_solve.shape.h;
  result.solute_points = complex_solve.solute_points;
  result.solvent_points = complex_solve.solvent_points;
  result.ion_accessible_points = complex_solve.ion_accessible_points;
  result.stern_excluded_points = complex_solve.stern_excluded_points;
  result.pbe_model_effective = complex_solve.pbe_model_effective;
  result.linear_seed_solver = complex_solve.linear_seed_solver;
  result.nl_solver = complex_solve.nl_solver;
  result.nl_inner_solver = complex_solve.nl_inner_solver;
  result.nl_mg_mode = complex_solve.nl_mg_mode;
  result.focus_parent_grid = complex_solve.focus_parent_grid;
  result.focus_child_grid = complex_solve.focus_child_grid;
  result.probe_status = complex_solve.probe_status;
  result.newton_inner_iterations = complex_solve.newton_inner_iterations;
  result.newton_step_norm = complex_solve.newton_step_norm;
  result.newton_residual_before = complex_solve.newton_residual_before;
  result.newton_residual_after = complex_solve.newton_residual_after;
  result.newton_line_search_status = complex_solve.newton_line_search_status;
  result.newton_correction_residual = complex_solve.newton_correction_residual;
  result.newton_correction_max_residual =
      complex_solve.newton_correction_max_residual;
  result.newton_accepted_lambda = complex_solve.newton_accepted_lambda;
  result.newton_rejected_steps = complex_solve.newton_rejected_steps;
  result.newton_jac_diag_min = complex_solve.newton_jac_diag_min;
  result.newton_jac_diag_max = complex_solve.newton_jac_diag_max;
  result.newton_screen_points = complex_solve.newton_screen_points;
  result.newton_active_nonlinear_points =
      complex_solve.newton_active_nonlinear_points;
  result.newton_stop_reason = complex_solve.newton_stop_reason;
  result.newton_inner_levels = complex_solve.newton_inner_levels;
  result.newton_inner_cycles = complex_solve.newton_inner_cycles;
  result.newton_inner_residual_before =
      complex_solve.newton_inner_residual_before;
  result.newton_inner_residual_after =
      complex_solve.newton_inner_residual_after;
  result.newton_inner_effective = complex_solve.newton_inner_effective;
  result.gaussian_density_sum = complex_solve.gaussian_density_sum;
  result.gaussian_cutoff_points = complex_solve.gaussian_cutoff_points;
  result.gaussian_eps_min = complex_solve.gaussian_eps_min;
  result.gaussian_eps_max = complex_solve.gaussian_eps_max;
  result.gaussian_eps_mean = complex_solve.gaussian_eps_mean;
  result.gaussian_charge_norm_error = complex_solve.gaussian_charge_norm_error;
  result.rpbe_coulomb_norm = complex_solve.rpbe_coulomb_norm;
  result.rpbe_grad_coulomb_norm = complex_solve.rpbe_grad_coulomb_norm;
  result.rpbe_grad_epsilon_norm = complex_solve.rpbe_grad_epsilon_norm;
  result.rpbe_source_rms = complex_solve.rpbe_source_rms;
  result.rpbe_source_max = complex_solve.rpbe_source_max;
  result.rpbe_source_status = complex_solve.rpbe_source_status;
  result.rhs_scale = complex_solve.rhs_scale;
  result.charge_assigned = complex_solve.charge_assigned;
  result.charge_moment_error = complex_solve.charge_moment_error;
  result.boundary_rms = complex_solve.boundary_rms;
  result.boundary_max = complex_solve.boundary_max;
  result.linear_residual = complex_solve.linear_residual;
  result.nl_iterations = complex_solve.nl_iterations;
  result.nl_blocks = complex_solve.nl_blocks;
  result.nl_relax = complex_solve.nl_relax;
  result.nl_max_dphi = complex_solve.nl_max_dphi;
  result.nl_residual = complex_solve.nl_residual;
  result.nl_clipped_points = complex_solve.nl_clipped_points;
  result.nl_continuation_steps = complex_solve.nl_continuation_steps;
  result.nl_backtracks = complex_solve.nl_backtracks;
  result.nl_status = complex_solve.nl_status;
  result.vdwms_enabled = complex_solve.vdwms_enabled;
  result.vdwms_boundary_initial = complex_solve.vdwms_boundary_initial;
  result.vdwms_boundary_final = complex_solve.vdwms_boundary_final;
  result.vdwms_added = complex_solve.vdwms_added;
  result.vdwms_removed = complex_solve.vdwms_removed;
  result.vdwms_exposed_points = complex_solve.vdwms_exposed_points;
  result.vdwms_midpoint_rms = complex_solve.vdwms_midpoint_rms;
  result.focus_enabled = (complex_solve.solver_mode == "nested_focus" ||
                          complex_solve.solver_mode == "delphi_focus") &&
                         complex_solve.focus_levels > 1;
  result.focus_levels = complex_solve.focus_levels;
  result.focus_initial_scale = complex_solve.focus_initial_scale;
  result.focus_initial_margin = complex_solve.focus_initial_margin;
  result.solver_mode = complex_solve.solver_mode;
  result.mg_levels = complex_solve.mg_levels;
  result.mg_cycles = complex_solve.mg_cycles;
  result.mg_pre = complex_solve.mg_pre;
  result.mg_post = complex_solve.mg_post;
  result.mg_coarse_iter = complex_solve.mg_coarse_iter;
  result.solvent_iterations = complex_solve.solvent_total_iterations;
  result.vacuum_iterations = complex_solve.vacuum_total_iterations;
  result.solvent_iterations_final = complex_solve.solvent.iterations;
  result.vacuum_iterations_final = complex_solve.vacuum.iterations;
  result.solvent_rms = complex_solve.solvent.rms;
  result.vacuum_rms = complex_solve.vacuum.rms;
  result.solvent_max_dphi = complex_solve.solvent.max_dphi;
  result.vacuum_max_dphi = complex_solve.vacuum.max_dphi;
  result.solvent_converged = complex_solve.solvent.converged;
  result.vacuum_converged = complex_solve.vacuum.converged;
  result.energy_kcal_mol = complex_solve.energy_kcal_mol;
  result.energy_kj_mol = complex_solve.energy_kj_mol;
  result.energy_kt = complex_solve.energy_kt;
  result.solvent_phase_kcal_mol = complex_solve.solvent_phase_kcal_mol;
  result.solvent_phase_kj_mol = complex_solve.solvent_phase_kj_mol;
  result.solvent_phase_kt = complex_solve.solvent_phase_kt;
  result.vacuum_phase_kcal_mol = complex_solve.vacuum_phase_kcal_mol;
  result.vacuum_phase_kj_mol = complex_solve.vacuum_phase_kj_mol;
  result.vacuum_phase_kt = complex_solve.vacuum_phase_kt;
  result.atom_corrected_kcal_mol = complex_solve.atom_corrected_kcal_mol;
  result.atom_corrected_kj_mol = complex_solve.atom_corrected_kj_mol;
  result.atom_corrected_kt = complex_solve.atom_corrected_kt;
  result.grid_corrected_kcal_mol = complex_solve.grid_corrected_kcal_mol;
  result.grid_corrected_kj_mol = complex_solve.grid_corrected_kj_mol;
  result.grid_corrected_kt = complex_solve.grid_corrected_kt;
  result.iscm_water_phase_kcal_mol = complex_solve.iscm_water_phase_kcal_mol;
  result.iscm_water_phase_kj_mol = complex_solve.iscm_water_phase_kj_mol;
  result.iscm_water_phase_kt = complex_solve.iscm_water_phase_kt;
  result.iscm_vacuum_phase_kcal_mol = complex_solve.iscm_vacuum_phase_kcal_mol;
  result.iscm_vacuum_phase_kj_mol = complex_solve.iscm_vacuum_phase_kj_mol;
  result.iscm_vacuum_phase_kt = complex_solve.iscm_vacuum_phase_kt;
  result.iscm_corrected_kcal_mol = complex_solve.iscm_corrected_kcal_mol;
  result.iscm_corrected_kj_mol = complex_solve.iscm_corrected_kj_mol;
  result.iscm_corrected_kt = complex_solve.iscm_corrected_kt;
  result.iscm_water_minus_vacuum_kcal_mol =
      complex_solve.iscm_water_minus_vacuum_kcal_mol;
  result.iscm_water_minus_vacuum_kj_mol =
      complex_solve.iscm_water_minus_vacuum_kj_mol;
  result.iscm_water_minus_vacuum_kt = complex_solve.iscm_water_minus_vacuum_kt;
  result.nonlinear_stress_kt = complex_solve.nonlinear_stress_kt;
  result.nonlinear_stress_kj_mol = complex_solve.nonlinear_stress_kj_mol;
  result.nonlinear_stress_kcal_mol = complex_solve.nonlinear_stress_kcal_mol;
  result.nonlinear_osmotic_kt = complex_solve.nonlinear_osmotic_kt;
  result.nonlinear_osmotic_kj_mol = complex_solve.nonlinear_osmotic_kj_mol;
  result.nonlinear_osmotic_kcal_mol = complex_solve.nonlinear_osmotic_kcal_mol;
  result.nonlinear_energy_correction_kt =
      complex_solve.nonlinear_energy_correction_kt;
  result.nonlinear_energy_correction_kj_mol =
      complex_solve.nonlinear_energy_correction_kj_mol;
  result.nonlinear_energy_correction_kcal_mol =
      complex_solve.nonlinear_energy_correction_kcal_mol;
  result.nonlinear_energy_status = complex_solve.nonlinear_energy_status;
  result.iscm_finalization_mode = complex_solve.iscm_finalization_mode;
  result.iscm_position_source = complex_solve.iscm_position_source;
  result.iscm_position_rms_vs_oracle =
      complex_solve.iscm_position_rms_vs_oracle;
  result.iscm_position_oracle_matches =
      complex_solve.iscm_position_oracle_matches;
  result.iscm_native_independent = complex_solve.iscm_native_independent;
  result.iscm_boundary_points = complex_solve.iscm_boundary_points;
  result.iscm_charged_boundary_points =
      complex_solve.iscm_charged_boundary_points;
  result.iscm_induced_charge_sum = complex_solve.iscm_induced_charge_sum;
  result.iscm_induced_charge_l1 = complex_solve.iscm_induced_charge_l1;
  result.iscm_induced_charge_l2 = complex_solve.iscm_induced_charge_l2;
  result.iscm_induced_charge_max = complex_solve.iscm_induced_charge_max;
  result.iscm_status = complex_solve.iscm_status;
  result.delphi_face_crossings = complex_solve.delphi_face_crossings;
  result.delphi_eps_face_matches = complex_solve.delphi_eps_face_matches;
  result.delphi_boundary_grid_points =
      complex_solve.delphi_boundary_grid_points;
  result.delphi_charged_boundary_points =
      complex_solve.delphi_charged_boundary_points;
  result.delphi_iscm_status = complex_solve.delphi_iscm_status;
  result.vdwms_enabled = complex_solve.vdwms_enabled;
  result.vdwms_boundary_initial = complex_solve.vdwms_boundary_initial;
  result.vdwms_boundary_final = complex_solve.vdwms_boundary_final;
  result.vdwms_added = complex_solve.vdwms_added;
  result.vdwms_removed = complex_solve.vdwms_removed;
  result.vdwms_exposed_points = complex_solve.vdwms_exposed_points;
  result.vdwms_midpoint_rms = complex_solve.vdwms_midpoint_rms;
  result.coulomb_energy_kcal_mol = complex_solve.coulomb_energy_kcal_mol;
  result.coulomb_energy_kj_mol = complex_solve.coulomb_energy_kj_mol;
  result.coulomb_energy_kt = complex_solve.coulomb_energy_kt;
  result.reaction_energy_source = complex_solve.reaction_energy_source;
  result.complex_energy_kcal_mol = complex_solve.energy_kcal_mol;
  result.complex_energy_kj_mol = complex_solve.energy_kj_mol;
  compute_stability_ensemble(atoms_, executor, result);

  ReactionSolve mol1_solve;
  ReactionSolve mol2_solve;
  bool have_endpoint_solve = false;
  const bool binding_cycle = args_.binding_cycle == "pqr-components";
  const bool single_system_dgsolv =
      args_.dgsolv && !args_.test_pqr.empty() && mol2_atoms.empty();
  if (single_system_dgsolv) {
    // The grid reaction solve already evaluates 1/2 sum q*phi_reac from the
    // solvent and homogeneous-vacuum fields, which is Delta G_solv for one
    // molecular system. Endpoint subtraction is only for binding/PBSA cycles.
    result.dgsolv_enabled = true;
    result.deltaG_solv_kcal_mol = complex_solve.energy_kcal_mol;
    result.deltaG_solv_kj_mol = complex_solve.energy_kj_mol;
    result.deltaG_solv_kt = complex_solve.energy_kt;
    result.dgsolv_status = complex_solve.converged ? "ok" : "not_converged";
  }
  if ((args_.dgsolv && !single_system_dgsolv) || binding_cycle) {
    const auto mol1_time_begin = std::chrono::steady_clock::now();
    mol1_solve = solve_reaction_field(mol1_atoms, executor, nullptr);
    const auto mol1_time_end = std::chrono::steady_clock::now();
    const auto mol2_time_begin = std::chrono::steady_clock::now();
    mol2_solve = solve_reaction_field(mol2_atoms, executor, nullptr);
    const auto mol2_time_end = std::chrono::steady_clock::now();
    result.pbsa_time_receptor_ms = elapsed_ms(mol1_time_begin, mol1_time_end);
    result.pbsa_time_ligand_ms = elapsed_ms(mol2_time_begin, mol2_time_end);
    have_endpoint_solve = true;
    result.dgsolv_enabled = true;
    result.mol1_energy_kcal_mol = mol1_solve.energy_kcal_mol;
    result.mol1_energy_kj_mol = mol1_solve.energy_kj_mol;
    result.mol2_energy_kcal_mol = mol2_solve.energy_kcal_mol;
    result.mol2_energy_kj_mol = mol2_solve.energy_kj_mol;
    result.deltaG_solv_kcal_mol = complex_solve.energy_kcal_mol -
                                  mol1_solve.energy_kcal_mol -
                                  mol2_solve.energy_kcal_mol;
    result.deltaG_solv_kj_mol = complex_solve.energy_kj_mol -
                                mol1_solve.energy_kj_mol -
                                mol2_solve.energy_kj_mol;
    result.deltaG_solv_kt =
        result.deltaG_solv_kj_mol / (kKBKJPerMolK * std::max(1.0, args_.temp));
    result.dgsolv_status = (complex_solve.converged && mol1_solve.converged &&
                            mol2_solve.converged)
                               ? "ok"
                               : "not_converged";
  }

  if (args_.pbsa) {
    result.pbsa_time_complex_ms =
        elapsed_ms(complex_time_begin, complex_time_end);
    const auto components_time_begin = std::chrono::steady_clock::now();
    const PbsaComponents pbsa = compute_pbsa_components(
        mol1, params1, *mol2, *params2, mol1_atoms, mol2_atoms, atoms_,
        result.deltaG_solv_kj_mol, executor, energy_provider);
    const auto components_time_end = std::chrono::steady_clock::now();
    result.pbsa_time_components_ms =
        elapsed_ms(components_time_begin, components_time_end);
    result.pbsa_enabled = true;
    const bool endpoints_converged =
        have_endpoint_solve && complex_solve.converged && mol1_solve.converged &&
        mol2_solve.converged;
    result.pbsa_status = !endpoints_converged
                             ? "not_converged"
                             : (pbsa.status == "ok" ? "ok" : pbsa.status);
    result.pbsa_elec_kj_mol = pbsa.elec_kj_mol;
    result.pbsa_vdw_kj_mol = pbsa.vdw_kj_mol;
    result.pbsa_mm_kj_mol = pbsa.mm_kj_mol;
    result.pbsa_sasa_complex_a2 = pbsa.sasa_complex_a2;
    result.pbsa_sasa_mol1_a2 = pbsa.sasa_mol1_a2;
    result.pbsa_sasa_mol2_a2 = pbsa.sasa_mol2_a2;
    result.pbsa_delta_sasa_a2 = pbsa.delta_sasa_a2;
    result.pbsa_nonpolar_kj_mol = pbsa.nonpolar_kj_mol;
    result.pbsa_delta_solv_kj_mol = pbsa.delta_solv_kj_mol;
    result.pbsa_delta_solv_kcal_mol = pbsa.delta_solv_kj_mol * kKJToKcal;
    result.pbsa_no_entropy_kj_mol = pbsa.no_entropy_kj_mol;
    result.pbsa_no_entropy_kcal_mol = pbsa.no_entropy_kj_mol * kKJToKcal;
    result.pbsa_receptor_atom_count = mol1_atoms.size();
    result.pbsa_ligand_atom_count = mol2_atoms.size();
    result.pbsa_complex_status =
        complex_solve.converged ? "ok" : "not_converged";
    result.pbsa_receptor_status =
        have_endpoint_solve ? (mol1_solve.converged ? "ok" : "not_converged")
                            : "missing_endpoint";
    result.pbsa_ligand_status =
        have_endpoint_solve ? (mol2_solve.converged ? "ok" : "not_converged")
                            : "missing_endpoint";
    if (have_endpoint_solve) {
      result.pbsa_complex_reuse_delta_kj_mol =
          complex_solve.pbsa_reuse_delta_kj_mol;
      result.pbsa_receptor_reuse_delta_kj_mol =
          mol1_solve.pbsa_reuse_delta_kj_mol;
      result.pbsa_ligand_reuse_delta_kj_mol =
          mol2_solve.pbsa_reuse_delta_kj_mol;
      result.pbsa_complex_reuse_surface_hits =
          complex_solve.pbsa_reuse_surface_hits;
      result.pbsa_complex_reuse_operator_hits =
          complex_solve.pbsa_reuse_operator_hits;
      result.pbsa_receptor_reuse_surface_hits =
          mol1_solve.pbsa_reuse_surface_hits;
      result.pbsa_receptor_reuse_operator_hits =
          mol1_solve.pbsa_reuse_operator_hits;
      result.pbsa_ligand_reuse_surface_hits =
          mol2_solve.pbsa_reuse_surface_hits;
      result.pbsa_ligand_reuse_operator_hits =
          mol2_solve.pbsa_reuse_operator_hits;
      const std::size_t reuse_hits = result.pbsa_complex_reuse_surface_hits +
                                     result.pbsa_receptor_reuse_surface_hits +
                                     result.pbsa_ligand_reuse_surface_hits;
      if (args_.pbsa_grid_reuse == "validate")
        result.pbsa_grid_reuse_effective =
            reuse_hits == 3 ? "validate" : "ineligible";
      else if (args_.pbsa_grid_reuse == "auto")
        result.pbsa_grid_reuse_effective =
            reuse_hits == 3 ? "auto" : "ineligible";
      else
        result.pbsa_grid_reuse_effective = "off";
      if (args_.pbsa_perf == "detail") {
        auto append_timing =
            [&](const std::string &endpoint, const std::string &phase,
                const std::string &variant, const TimingBuckets &timing,
                std::size_t surface_hits, std::size_t operator_hits,
                const std::string &status) {
              PbsaEndpointTimingRow row;
              row.trajectory_id = result.trajectory_id;
              row.frame_index = result.frame_index;
              row.global_index = result.global_index;
              row.time_ps = result.time_ps;
              row.endpoint = endpoint;
              row.phase = phase;
              row.variant = variant;
              row.timing = timing;
              row.reuse_surface_hits = surface_hits;
              row.reuse_operator_hits = operator_hits;
              row.status = status;
              pbsa_endpoint_timing_rows_.push_back(std::move(row));
            };
        auto append_endpoint = [&](const std::string &endpoint,
                                   const ReactionSolve &endpoint_solve,
                                   const std::string &status) {
          append_timing(endpoint, "solvent", "baseline",
                        endpoint_solve.pbsa_solvent_timing, 0U, 0U, status);
          const bool reused = endpoint_solve.pbsa_reuse_surface_hits > 0U ||
                              endpoint_solve.pbsa_reuse_operator_hits > 0U;
          const bool production_reuse =
              args_.pbsa_grid_reuse == "auto" && reused;
          if (production_reuse) {
            append_timing(endpoint, "vacuum", "reuse",
                          endpoint_solve.pbsa_reuse_timing,
                          endpoint_solve.pbsa_reuse_surface_hits,
                          endpoint_solve.pbsa_reuse_operator_hits, status);
          } else {
            append_timing(endpoint, "vacuum", "baseline",
                          endpoint_solve.pbsa_vacuum_timing, 0U, 0U, status);
          }
          if (reused && !production_reuse) {
            append_timing(endpoint, "vacuum", "reuse",
                          endpoint_solve.pbsa_reuse_timing,
                          endpoint_solve.pbsa_reuse_surface_hits,
                          endpoint_solve.pbsa_reuse_operator_hits, status);
          }
          TimingBuckets energy_timing;
          energy_timing.energy_ms = endpoint_solve.pbsa_energy_ms;
          append_timing(endpoint, "reaction_energy", "baseline", energy_timing,
                        0U, 0U, status);
        };
        append_endpoint("complex", complex_solve, result.pbsa_complex_status);
        append_endpoint("receptor", mol1_solve, result.pbsa_receptor_status);
        append_endpoint("ligand", mol2_solve, result.pbsa_ligand_status);
      }
      result.pbsa_receptor_solvent_iterations =
          mol1_solve.solvent_total_iterations;
      result.pbsa_receptor_vacuum_iterations =
          mol1_solve.vacuum_total_iterations;
      result.pbsa_ligand_solvent_iterations =
          mol2_solve.solvent_total_iterations;
      result.pbsa_ligand_vacuum_iterations = mol2_solve.vacuum_total_iterations;
      result.pbsa_receptor_mg_cycles = mol1_solve.mg_cycles;
      result.pbsa_ligand_mg_cycles = mol2_solve.mg_cycles;
      result.pbsa_receptor_solvent_rms = mol1_solve.solvent.rms;
      result.pbsa_receptor_vacuum_rms = mol1_solve.vacuum.rms;
      result.pbsa_ligand_solvent_rms = mol2_solve.solvent.rms;
      result.pbsa_ligand_vacuum_rms = mol2_solve.vacuum.rms;
      result.pbsa_receptor_solvent_max_dphi = mol1_solve.solvent.max_dphi;
      result.pbsa_receptor_vacuum_max_dphi = mol1_solve.vacuum.max_dphi;
      result.pbsa_ligand_solvent_max_dphi = mol2_solve.solvent.max_dphi;
      result.pbsa_ligand_vacuum_max_dphi = mol2_solve.vacuum.max_dphi;
    }
    const std::string pbsa_decomp = effective_pbsa_decomp();
    const bool legacy_pbsa_decomp =
        pbsa_decomp == "residue" || pbsa_decomp == "hybrid";
    if ((legacy_pbsa_decomp || edecomp_sink_) &&
        result.pbsa_status == "ok") {
      const auto residue_time_begin = std::chrono::steady_clock::now();
      const std::vector<ResidueFrameRow> residue_rows =
          compute_pbsa_residue_rows(mol1, *mol2, mol1_atoms, mol2_atoms, atoms_,
                                    complex_solve, mol1_solve, mol2_solve,
                                    result, executor, energy_provider);
      const auto residue_time_end = std::chrono::steady_clock::now();
      result.pbsa_time_residue_ms =
          elapsed_ms(residue_time_begin, residue_time_end);
      if (legacy_pbsa_decomp)
        accumulate_pbsa_residue_rows(result, residue_rows);
      if (edecomp_sink_) {
        std::vector<EDecompRow> sink_rows;
        sink_rows.reserve(residue_rows.size());
        for (const ResidueFrameRow &row : residue_rows) {
          EDecompRow sink_row;
          sink_row.molecule = row.key.molecule;
          sink_row.unit_type = row.key.unit_type;
          sink_row.chain = row.key.chain;
          sink_row.resnr = row.key.resnr;
          sink_row.respos = row.key.respos;
          sink_row.resname = row.key.resname;
          sink_row.atom_index = row.key.atom_index;
          sink_row.atom_name = row.key.atom_name;
          sink_row.elec_kj_mol = row.elec_kj_mol;
          sink_row.vdw_kj_mol = row.vdw_kj_mol;
          sink_row.polar_kj_mol = row.polar_kj_mol;
          sink_row.nonpolar_kj_mol = row.nonpolar_kj_mol;
          sink_rows.push_back(std::move(sink_row));
        }
        edecomp_sink_(sink_rows);
      }
    }
  }

  result.status = (result.solvent_converged && result.vacuum_converged)
                      ? "ok"
                      : "not_converged";
  if (result.dgsolv_enabled && result.dgsolv_status != "ok")
    result.status = "not_converged";
  if (result.pbsa_enabled && result.pbsa_status != "ok")
    result.status = "not_converged";
  result.pbsa_time_frame_ms =
      elapsed_ms(frame_time_begin, std::chrono::steady_clock::now());
  const auto probe_time_begin = std::chrono::steady_clock::now();
  result.probe_status = maybe_write_probe_sites(result, atoms_, complex_solve);
  add_timing_ms(TimingBucket::Probe,
                elapsed_ms(probe_time_begin, std::chrono::steady_clock::now()));

  maybe_write_parity_stage(result, complex_solve);
  maybe_write_stability_audit(result, atoms_, complex_solve);
  const ProcessMemorySample frame_memory_after = sample_process_memory();
  result.memory_rss_before_bytes = frame_memory_before.working_set_bytes;
  result.memory_rss_after_bytes = frame_memory_after.working_set_bytes;
  result.memory_rss_delta_bytes =
      frame_memory_after.working_set_bytes >=
              frame_memory_before.working_set_bytes
          ? frame_memory_after.working_set_bytes -
                frame_memory_before.working_set_bytes
          : 0;
  result.memory_rss_delta_signed_bytes =
      static_cast<std::int64_t>(frame_memory_after.working_set_bytes) -
      static_cast<std::int64_t>(frame_memory_before.working_set_bytes);
  result.memory_peak_working_set_bytes =
      frame_memory_after.peak_working_set_bytes;
  maybe_write_atom_reaction_potentials(result, atoms_, complex_solve,
                                       "complex");
  if (binding_cycle && have_endpoint_solve) {
    maybe_write_atom_reaction_potentials(result, mol1_atoms, mol1_solve,
                                         "receptor");
    maybe_write_atom_reaction_potentials(result, mol2_atoms, mol2_solve,
                                         "ligand");
    maybe_write_binding_electrostatics(result, atoms_, mol1_atoms, mol2_atoms,
                                       complex_solve, mol1_solve, mol2_solve);
  }
  append_energy_row(result);
  frame_timing.total_ms =
      elapsed_ms(frame_time_begin, std::chrono::steady_clock::now());
  append_timing_row(result, frame_timing);
  append_memory_profile_row(result, frame_timing);

  ++frames_total_;
  if (result.status == "ok")
    ++frames_converged_;
  else
    ++frames_failed_;
  max_grid_points_seen_ = std::max(max_grid_points_seen_, result.grid_points);
  max_memory_bytes_seen_ =
      std::max(max_memory_bytes_seen_, result.memory_bytes);
  max_solute_points_seen_ =
      std::max(max_solute_points_seen_, result.solute_points);
  max_solvent_points_seen_ =
      std::max(max_solvent_points_seen_, result.solvent_points);
  max_ion_accessible_points_seen_ =
      std::max(max_ion_accessible_points_seen_, result.ion_accessible_points);
  max_stern_excluded_points_seen_ =
      std::max(max_stern_excluded_points_seen_, result.stern_excluded_points);
  max_focus_levels_seen_ =
      std::max(max_focus_levels_seen_, result.focus_levels);
  max_mg_levels_seen_ = std::max(max_mg_levels_seen_, result.mg_levels);
  max_mg_cycles_seen_ = std::max(max_mg_cycles_seen_, result.mg_cycles);
  if (result.pbe_model == "nonlinear") {
    ++nonlinear_frames_;
    if (result.nl_status == "ok" || result.nl_status == "linearized_no_salt")
      ++nonlinear_converged_;
    else
      ++nonlinear_failed_;
    max_nl_blocks_seen_ = std::max(max_nl_blocks_seen_, result.nl_blocks);
    max_nl_iterations_seen_ =
        std::max(max_nl_iterations_seen_, result.nl_iterations);
    max_nl_clipped_points_seen_ =
        std::max(max_nl_clipped_points_seen_, result.nl_clipped_points);
    max_nl_dphi_seen_ = std::max(max_nl_dphi_seen_, result.nl_max_dphi);
    max_nl_residual_seen_ = std::max(max_nl_residual_seen_, result.nl_residual);
    max_newton_inner_iterations_seen_ = std::max(
        max_newton_inner_iterations_seen_, result.newton_inner_iterations);
    max_newton_step_norm_seen_ =
        std::max(max_newton_step_norm_seen_, result.newton_step_norm);
    max_newton_residual_before_seen_ = std::max(
        max_newton_residual_before_seen_, result.newton_residual_before);
    max_newton_residual_after_seen_ =
        std::max(max_newton_residual_after_seen_, result.newton_residual_after);
    max_newton_correction_residual_seen_ =
        std::max(max_newton_correction_residual_seen_,
                 result.newton_correction_residual);
    max_newton_correction_max_residual_seen_ =
        std::max(max_newton_correction_max_residual_seen_,
                 result.newton_correction_max_residual);
    max_newton_rejected_steps_seen_ =
        std::max(max_newton_rejected_steps_seen_, result.newton_rejected_steps);
    max_newton_inner_levels_seen_ =
        std::max(max_newton_inner_levels_seen_, result.newton_inner_levels);
    max_newton_inner_cycles_seen_ =
        std::max(max_newton_inner_cycles_seen_, result.newton_inner_cycles);
    max_newton_inner_residual_before_seen_ =
        std::max(max_newton_inner_residual_before_seen_,
                 result.newton_inner_residual_before);
    max_newton_inner_residual_after_seen_ =
        std::max(max_newton_inner_residual_after_seen_,
                 result.newton_inner_residual_after);
  }
  max_linear_residual_seen_ =
      std::max(max_linear_residual_seen_, result.linear_residual);
  max_boundary_rms_seen_ =
      std::max(max_boundary_rms_seen_, result.boundary_rms);
  max_boundary_max_seen_ =
      std::max(max_boundary_max_seen_, result.boundary_max);

  const double frame_n = static_cast<double>(frames_total_);
  ++energy_frames_;
  const double delta = result.energy_kj_mol - energy_mean_;
  energy_mean_ += delta / frame_n;
  energy_m2_ += delta * (result.energy_kj_mol - energy_mean_);
  if (frames_total_ == 1) {
    energy_min_ = result.energy_kj_mol;
    energy_max_ = result.energy_kj_mol;
  } else {
    energy_min_ = std::min(energy_min_, result.energy_kj_mol);
    energy_max_ = std::max(energy_max_, result.energy_kj_mol);
  }
  iter_mean_ +=
      (static_cast<double>(result.solvent_iterations) - iter_mean_) / frame_n;
  mg_cycle_mean_ +=
      (static_cast<double>(result.mg_cycles) - mg_cycle_mean_) / frame_n;

  if (result.dgsolv_enabled) {
    ++dgsolv_frames_;
    const double dg_n = static_cast<double>(dgsolv_frames_);
    const double dg_delta = result.deltaG_solv_kj_mol - dgsolv_mean_;
    dgsolv_mean_ += dg_delta / dg_n;
    dgsolv_m2_ += dg_delta * (result.deltaG_solv_kj_mol - dgsolv_mean_);
    if (dgsolv_frames_ == 1) {
      dgsolv_min_ = result.deltaG_solv_kj_mol;
      dgsolv_max_ = result.deltaG_solv_kj_mol;
    } else {
      dgsolv_min_ = std::min(dgsolv_min_, result.deltaG_solv_kj_mol);
      dgsolv_max_ = std::max(dgsolv_max_, result.deltaG_solv_kj_mol);
    }
  }
  accumulate_pbsa(result);

  if (log.is_open() && result.status != "ok") {
    TitanUI::log_kv(log, "PBE Warning",
                    "frame " + std::to_string(result.global_index) +
                        " did not meet convergence thresholds");
  }
  return result;
}


} // namespace TitanPBE
