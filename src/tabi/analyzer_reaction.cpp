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
using analyzer_impl_detail::csv_escape;
using analyzer_impl_detail::elapsed_ms;
using analyzer_impl_detail::format_fixed;
using analyzer_impl_detail::idx3;
using analyzer_impl_detail::kCoulombKcalPerMol;
using analyzer_impl_detail::kKBKcalPerMolK;
using analyzer_impl_detail::kKBKJPerMolK;
using analyzer_impl_detail::kKcalToKJ;
using analyzer_impl_detail::sample_process_memory;

namespace {

// Second copy of the screening length, for the grid path.  Keep it in step
// with the boundary-integral copy in tabi_solver.cpp: kappa scales as
// 1/sqrt(eps_r*T) and the 0.3291 prefactor already contains eps_r = 78.5.
double debye_kappa_inv_a(double salt_m, double temp_k,
                         double eps_out) noexcept {
  if (salt_m <= 0.0)
    return 0.0;
  constexpr double kReferenceEpsOut = 78.5;
  const double temp_scale = std::sqrt(298.15 / std::max(1.0, temp_k));
  const double diel_scale =
      std::sqrt(kReferenceEpsOut / std::max(1.0e-6, eps_out));
  return 0.3291 * std::sqrt(salt_m) * temp_scale * diel_scale;
}

class TabiPrepareProgressWriter {
public:
  TabiPrepareProgressWriter(const PbeArgs &args, std::size_t atom_count,
                            std::size_t threads)
      : enabled_(args.tabi_pre_solver_progress == "on"),
        start_(std::chrono::steady_clock::now()), atom_count_(atom_count),
        surface_points_(args.tabi_surface_points), threads_(threads),
        pqr_path_(args.test_pqr), tabi_sum_(args.tabi_sum),
        formulation_(args.tabi_formulation),
        tree_builder_(args.tabi_tree_builder),
        node_bounds_(args.tabi_node_bounds), mac_safety_(args.tabi_mac_safety),
        expansion_center_(args.tabi_expansion_center),
        fragmentation_safety_(args.tabi_fragmentation_safety),
        convergence_estimator_(args.tabi_convergence_estimator),
        convergence_estimator_model_(args.tabi_convergence_estimator_model),
        convergence_channel_calibration_(
            args.tabi_convergence_channel_calibration),
        convergence_split_policy_(args.tabi_convergence_split_policy),
        channel_missed_semantics_(args.tabi_channel_missed_semantics),
        progress_mode_(args.tabi_progress) {
    if (!enabled_)
      return;
    out_.open("pbe_tabi_prepare_progress.csv", std::ios::out | std::ios::trunc);
    if (!out_.is_open()) {
      enabled_ = false;
      return;
    }
    out_ << "elapsed_seconds,phase,atom_count,pqr_path,pqr_status,"
            "surface_points_requested,tabi_sum,tabi_formulation,"
            "tabi_tree_builder,tabi_node_bounds,tabi_mac_safety,"
            "tabi_expansion_center,tabi_fragmentation_safety,"
            "tabi_convergence_estimator,tabi_convergence_estimator_model,"
            "tabi_convergence_channel_calibration,"
            "tabi_convergence_split_policy,"
            "tabi_channel_missed_semantics,tabi_progress,threads,"
            "rss_bytes,status\n";
    out_.flush();
  }

  void write(const std::string &phase, const std::string &status) {
    if (!enabled_)
      return;
    const auto now = std::chrono::steady_clock::now();
    const ProcessMemorySample memory = sample_process_memory();
    const std::string pqr_status = pqr_path_.empty() ? "none" : "configured";
    out_ << std::fixed << std::setprecision(6)
         << std::chrono::duration<double>(now - start_).count() << ','
         << csv_escape(phase) << ',' << atom_count_ << ','
         << csv_escape(pqr_path_) << ',' << csv_escape(pqr_status) << ','
         << surface_points_ << ',' << csv_escape(tabi_sum_) << ','
         << csv_escape(formulation_) << ',' << csv_escape(tree_builder_) << ','
         << csv_escape(node_bounds_) << ',' << csv_escape(mac_safety_) << ','
         << csv_escape(expansion_center_) << ','
         << csv_escape(fragmentation_safety_) << ','
         << csv_escape(convergence_estimator_) << ','
         << csv_escape(convergence_estimator_model_) << ','
         << csv_escape(convergence_channel_calibration_) << ','
         << csv_escape(convergence_split_policy_) << ','
         << csv_escape(channel_missed_semantics_) << ','
         << csv_escape(progress_mode_) << ',' << threads_ << ','
         << memory.working_set_bytes << ',' << csv_escape(status) << '\n';
    out_.flush();
  }

private:
  bool enabled_{false};
  std::chrono::steady_clock::time_point start_{};
  std::size_t atom_count_{0};
  int surface_points_{96};
  std::size_t threads_{1};
  std::string pqr_path_;
  std::string tabi_sum_;
  std::string formulation_;
  std::string tree_builder_;
  std::string node_bounds_;
  std::string mac_safety_;
  std::string expansion_center_;
  std::string fragmentation_safety_;
  std::string convergence_estimator_;
  std::string convergence_estimator_model_;
  std::string convergence_channel_calibration_;
  std::string convergence_split_policy_;
  std::string channel_missed_semantics_;
  std::string progress_mode_;
  std::ofstream out_;
};

std::string uppercase_letters(std::string value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    if (std::isalpha(static_cast<unsigned char>(ch)))
      out.push_back(
          static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
  }
  return out;
}

} // namespace

bool AnalyzerImpl::pbsa_grid_reuse_gate() const {
  return args_.pbsa && !stability_probe_solve_active_ &&
         args_.backend == "grid" && args_.model == "linear" &&
         args_.biomodel == "pbe" &&
         args_.dielectric_model == "two-dielectric" &&
         args_.charge_assignment == "trilinear" &&
         args_.test_energy_convention == "auto" && args_.focus == "off" &&
         effective_solver_mode() == "sor";
}

AnalyzerImpl::ReactionSolve
AnalyzerImpl::solve_reaction_field(const std::vector<Atom> &atoms,
                                   TitanExecutor &executor,
                                   const FrameResult *map_result) {
  ReactionSolve solve;
  TimingBuckets solvent_timing;
  TimingBuckets vacuum_timing;
  TimingBuckets reuse_timing;
  const bool capture_pbsa_timing = args_.pbsa && args_.pbsa_perf == "detail" &&
                                   !stability_probe_solve_active_;
  struct PbsaTimingScope {
    AnalyzerImpl &self;
    TimingBuckets *previous;
    bool capture;
    PbsaTimingScope(AnalyzerImpl &impl, TimingBuckets &timing, bool enabled)
        : self(impl), previous(impl.active_timing_), capture(enabled) {
      if (capture)
        self.active_timing_ = &timing;
    }
    ~PbsaTimingScope() {
      if (capture)
        self.active_timing_ = previous;
    }
  };
  const bool reuse_gate = pbsa_grid_reuse_gate();
  const bool validate_reuse = args_.pbsa_grid_reuse == "validate" && reuse_gate;
  const bool auto_reuse = args_.pbsa_grid_reuse == "auto" && reuse_gate;
  const double solvent_kappa = debye_kappa_inv_a(args_.salt, args_.temp, args_.ext_diel);
  FieldSolve solvent;
  {
    PbsaTimingScope timing_scope(*this, solvent_timing, capture_pbsa_timing);
    solvent = solve_grid_field(
        atoms,
        GridSolveRequest::physical(args_.int_diel, args_.ext_diel,
                                   solvent_kappa, map_result),
        executor);
  }
  const ForensicAudit solvent_forensic =
      barnase_forensic_enabled() ? compute_forensic_audit(solvent.shape)
                                 : ForensicAudit{};
  const bool wants_iscm = args_.test_energy_convention == "delphi-iscm" ||
                          args_.test_energy_convention == "delphi-corrected" ||
                          delphi_nwt_parity_active() ||
                          barnase_forensic_enabled();
  const AlignedDoubleVector solvent_screen_for_iscm =
      wants_iscm ? grid_.screen : AlignedDoubleVector{};
  struct ReuseScope {
    AnalyzerImpl &self;
    ReuseScope(AnalyzerImpl &impl, const GridShape &shape) : self(impl) {
      self.pbsa_reuse_shape_ = shape;
      self.pbsa_reuse_vacuum_active_ = true;
    }
    ~ReuseScope() { self.pbsa_reuse_vacuum_active_ = false; }
  };
  FieldSolve vacuum;
  if (auto_reuse) {
    ReuseScope reuse_scope(*this, solvent.shape);
    PbsaTimingScope timing_scope(*this, vacuum_timing, capture_pbsa_timing);
    vacuum = solve_linear_field(atoms, args_.int_diel, args_.int_diel, 0.0,
                                executor, nullptr);
  } else {
    PbsaTimingScope timing_scope(*this, vacuum_timing, capture_pbsa_timing);
    vacuum = solve_grid_field(
        atoms, GridSolveRequest::linear(args_.int_diel, args_.int_diel),
        executor);
  }
  FieldSolve reused_vacuum;
  if (validate_reuse) {
    ReuseScope reuse_scope(*this, solvent.shape);
    PbsaTimingScope timing_scope(*this, reuse_timing, capture_pbsa_timing);
    reused_vacuum = solve_grid_field(
        atoms, GridSolveRequest::linear(args_.int_diel, args_.int_diel),
        executor);
  }

  solve.shape = solvent.shape;
  solve.solvent = solvent.stats;
  solve.vacuum = vacuum.stats;
  solve.solvent_total_iterations = solvent.total_iterations;
  solve.vacuum_total_iterations = vacuum.total_iterations;
  solve.focus_levels = solvent.levels;
  solve.focus_initial_scale = solvent.initial_scale;
  solve.focus_initial_margin = solvent.initial_margin;
  solve.solver_mode = solvent.solver_mode;
  solve.mg_levels = solvent.mg_levels;
  solve.mg_cycles = solvent.mg_cycles;
  solve.mg_pre = solvent.mg_pre;
  solve.mg_post = solvent.mg_post;
  solve.mg_coarse_iter = solvent.mg_coarse_iter;
  solve.solute_points = solvent.solute_points;
  solve.solvent_points = solvent.solvent_points;
  solve.ion_accessible_points = solvent.ion_accessible_points;
  solve.stern_excluded_points = solvent.stern_excluded_points;
  solve.peak_memory_bytes =
      std::max(solvent.peak_memory_bytes, vacuum.peak_memory_bytes);
  solve.pbsa_solvent_timing = solvent_timing;
  solve.pbsa_vacuum_timing = vacuum_timing;
  solve.pbsa_reuse_timing = auto_reuse ? vacuum_timing : reuse_timing;
  solve.pbe_model_effective = solvent.pbe_model_effective;
  solve.linear_seed_solver = solvent.linear_seed_solver;
  solve.nl_solver = solvent.nl_solver;
  solve.nl_inner_solver = solvent.nl_inner_solver;
  solve.nl_mg_mode = solvent.nl_mg_mode;
  solve.focus_parent_grid = solvent.focus_parent_grid;
  solve.focus_child_grid = solvent.focus_child_grid;
  solve.probe_status = solvent.probe_status;
  solve.newton_inner_iterations = solvent.newton_inner_iterations;
  solve.newton_step_norm = solvent.newton_step_norm;
  solve.newton_residual_before = solvent.newton_residual_before;
  solve.newton_residual_after = solvent.newton_residual_after;
  solve.newton_line_search_status = solvent.newton_line_search_status;
  solve.newton_correction_residual = solvent.newton_correction_residual;
  solve.newton_correction_max_residual = solvent.newton_correction_max_residual;
  solve.newton_accepted_lambda = solvent.newton_accepted_lambda;
  solve.newton_rejected_steps = solvent.newton_rejected_steps;
  solve.newton_jac_diag_min = solvent.newton_jac_diag_min;
  solve.newton_jac_diag_max = solvent.newton_jac_diag_max;
  solve.newton_screen_points = solvent.newton_screen_points;
  solve.newton_active_nonlinear_points = solvent.newton_active_nonlinear_points;
  solve.newton_stop_reason = solvent.newton_stop_reason;
  solve.newton_inner_levels = solvent.newton_inner_levels;
  solve.newton_inner_cycles = solvent.newton_inner_cycles;
  solve.newton_inner_residual_before = solvent.newton_inner_residual_before;
  solve.newton_inner_residual_after = solvent.newton_inner_residual_after;
  solve.newton_inner_effective = solvent.newton_inner_effective;
  solve.gaussian_density_sum = solvent.gaussian_density_sum;
  solve.gaussian_cutoff_points = solvent.gaussian_cutoff_points;
  solve.gaussian_eps_min = solvent.gaussian_eps_min;
  solve.gaussian_eps_max = solvent.gaussian_eps_max;
  solve.gaussian_eps_mean = solvent.gaussian_eps_mean;
  solve.gaussian_charge_norm_error = solvent.gaussian_charge_norm_error;
  solve.rhs_scale = solvent.rhs_scale;
  solve.charge_assigned = solvent.charge_assigned;
  solve.charge_moment_error = solvent.charge_moment_error;
  solve.boundary_rms = solvent.boundary_rms;
  solve.boundary_max = solvent.boundary_max;
  solve.linear_residual = solvent.linear_residual;
  solve.nl_iterations = solvent.nl_iterations;
  solve.nl_blocks = solvent.nl_blocks;
  solve.nl_relax = solvent.nl_relax;
  solve.nl_max_dphi = solvent.nl_max_dphi;
  solve.nl_residual = solvent.nl_residual;
  solve.nl_clipped_points = solvent.nl_clipped_points;
  solve.nl_continuation_steps = solvent.nl_continuation_steps;
  solve.nl_backtracks = solvent.nl_backtracks;
  solve.nl_status = solvent.nl_status;
  solve.parity = solvent.parity;
  solve.vdwms_enabled = solvent.vdwms_enabled;
  solve.vdwms_boundary_initial = solvent.vdwms_boundary_initial;
  solve.vdwms_boundary_final = solvent.vdwms_boundary_final;
  solve.vdwms_added = solvent.vdwms_added;
  solve.vdwms_removed = solvent.vdwms_removed;
  solve.vdwms_exposed_points = solvent.vdwms_exposed_points;
  solve.vdwms_midpoint_rms = solvent.vdwms_midpoint_rms;
  const FieldSolve *reuse_vacuum_for_stats =
      auto_reuse ? &vacuum : (validate_reuse ? &reused_vacuum : nullptr);
  if (reuse_vacuum_for_stats != nullptr) {
    solve.pbsa_reuse_surface_hits =
        reuse_vacuum_for_stats->pbsa_reuse_surface_hit ? 1U : 0U;
    solve.pbsa_reuse_operator_hits =
        reuse_vacuum_for_stats->pbsa_reuse_operator_hit ? 1U : 0U;
  }

  const auto energy_time_begin = std::chrono::steady_clock::now();
  double rpb_source_time_ms = 0.0;
  double probe_time_ms = 0.0;
  solve.atom_reaction_kj_mol.assign(atoms.size(), 0.0);
  solve.atom_solvent_potential_kj_mol_e.assign(atoms.size(), 0.0);
  solve.atom_vacuum_potential_kj_mol_e.assign(atoms.size(), 0.0);
  solve.atom_reaction_potential_kj_mol_e.assign(atoms.size(), 0.0);
  const double kt_kcal = kKBKcalPerMolK * std::max(1.0, args_.temp);
  solve.solvent_phase_kcal_mol =
      atom_phase_energy_kcal(atoms, solvent.atom_phi);
  solve.vacuum_phase_kcal_mol = atom_phase_energy_kcal(atoms, vacuum.atom_phi);
  if (validate_reuse) {
    const double reused_vacuum_phase_kcal =
        atom_phase_energy_kcal(atoms, reused_vacuum.atom_phi);
    solve.pbsa_reuse_delta_kj_mol =
        (reused_vacuum_phase_kcal - solve.vacuum_phase_kcal_mol) * kKcalToKJ;
  }
  solve.atom_corrected_kcal_mol =
      solve.solvent_phase_kcal_mol - solve.vacuum_phase_kcal_mol;
  solve.coulomb_energy_kcal_mol =
      analytic_coulomb_energy_kcal(atoms, args_.int_diel);
  const double atom_solvent_phase_kcal = solve.solvent_phase_kcal_mol;
  const double atom_vacuum_phase_kcal = solve.vacuum_phase_kcal_mol;
  const bool has_grid_energy = solvent.phi_grid.size() == solvent.shape.n &&
                               vacuum.phi_grid.size() == solvent.shape.n;
  IscmAudit iscm_audit;
  NonlinearEnergyAudit nonlinear_energy_audit;
  RpbAudit rpb_audit;
  AlignedDoubleVector rpb_source_map;
  AlignedDoubleVector rpb_coulomb_map;
  const AlignedDoubleVector *rpb_source_top_map = &rpb_source_map;
  const AlignedDoubleVector *rpb_coulomb_top_map = &rpb_coulomb_map;
  if (has_grid_energy) {
    const AlignedDoubleVector *solvent_energy_rho = &solvent.rho_grid;
    if (solvent.rho_energy_grid.size() == solvent.shape.n)
      solvent_energy_rho = &solvent.rho_energy_grid;
    const AlignedDoubleVector *vacuum_energy_rho = &vacuum.rho_grid;
    if (vacuum.rho_energy_grid.size() == solvent.shape.n)
      vacuum_energy_rho = &vacuum.rho_energy_grid;
    if (solvent_energy_rho->size() != solvent.shape.n)
      solvent_energy_rho = &grid_.rho;
    if (vacuum_energy_rho->size() != solvent.shape.n)
      vacuum_energy_rho = solvent_energy_rho;
    const double grid_solvent_phase_kcal = grid_phase_energy_kcal(
        solvent.shape, solvent.phi_grid, *solvent_energy_rho);
    const double grid_vacuum_phase_kcal = grid_phase_energy_kcal(
        solvent.shape, vacuum.phi_grid, *vacuum_energy_rho);
    solve.grid_corrected_kcal_mol =
        grid_solvent_phase_kcal - grid_vacuum_phase_kcal;
    if (wants_iscm) {
      iscm_audit = compute_iscm_audit(
          solvent.shape, atoms, solvent.phi_grid, solvent_screen_for_iscm,
          solvent.inside_grid, solvent.eps_x_grid, solvent.eps_y_grid,
          solvent.eps_z_grid, grid_vacuum_phase_kcal);
      copy_iscm_audit_to_solve(iscm_audit, solve);
    }
    if (args_.model == "nonlinear") {
      nonlinear_energy_audit = compute_nonlinear_energy_audit(
          solvent.shape, solvent.phi_grid, solvent_screen_for_iscm);
      copy_nonlinear_energy_audit_to_solve(nonlinear_energy_audit, solve);
    }
    if (args_.biomodel == "rpbe") {
      const bool wants_rpb_source_top =
          map_result != nullptr && args_.parity_dump != "off";
      const bool has_solvent_rpb_audit =
          solvent.rpbe_source_status != "off" &&
          solvent.rpbe_source_status != "missing_maps" &&
          solvent.rpbe_source_status != "empty_interior";
      const bool has_solvent_rpb_top =
          solvent.rpbe_source_grid.size() == solvent.shape.n &&
          solvent.rpbe_coulomb_grid.size() == solvent.shape.n;
      if (has_solvent_rpb_audit &&
          (!wants_rpb_source_top || has_solvent_rpb_top)) {
        rpb_audit.coulomb_norm = solvent.rpbe_coulomb_norm;
        rpb_audit.grad_coulomb_norm = solvent.rpbe_grad_coulomb_norm;
        rpb_audit.grad_epsilon_norm = solvent.rpbe_grad_epsilon_norm;
        rpb_audit.source_rms = solvent.rpbe_source_rms;
        rpb_audit.source_max = solvent.rpbe_source_max;
        rpb_audit.sampled_points =
            solvent.shape.n > 0
                ? (solvent.shape.nx > 2 && solvent.shape.ny > 2 &&
                           solvent.shape.nz > 2
                       ? (solvent.shape.nx - 2) * (solvent.shape.ny - 2) *
                             (solvent.shape.nz - 2)
                       : 0)
                : 0;
        rpb_audit.status = solvent.rpbe_source_status;
        if (has_solvent_rpb_top) {
          rpb_source_top_map = &solvent.rpbe_source_grid;
          rpb_coulomb_top_map = &solvent.rpbe_coulomb_grid;
        }
      } else {
        const auto rpb_time_begin = std::chrono::steady_clock::now();
        rpb_audit = compute_rpb_source_audit(
            solvent.shape, atoms, executor, solvent.eps_x_grid,
            solvent.eps_y_grid, solvent.eps_z_grid,
            wants_rpb_source_top ? &rpb_source_map : nullptr,
            wants_rpb_source_top ? &rpb_coulomb_map : nullptr);
        const double rpb_ms =
            elapsed_ms(rpb_time_begin, std::chrono::steady_clock::now());
        rpb_source_time_ms += rpb_ms;
        add_timing_ms(TimingBucket::RpbSource, rpb_ms);
      }
      solve.rpbe_coulomb_norm = rpb_audit.coulomb_norm;
      solve.rpbe_grad_coulomb_norm = rpb_audit.grad_coulomb_norm;
      solve.rpbe_grad_epsilon_norm = rpb_audit.grad_epsilon_norm;
      solve.rpbe_source_rms = rpb_audit.source_rms;
      solve.rpbe_source_max = rpb_audit.source_max;
      solve.rpbe_source_status = rpb_audit.status;
    }
    const bool legacy_grid = args_.test_energy_convention == "auto" &&
                             (args_.charge_assignment == "delphi" ||
                              args_.charge_assignment == "gaussian");
    const bool forced_grid = args_.test_energy_convention == "grid";
    const bool forced_iscm =
        iscm_audit.status == "ok" &&
        (args_.test_energy_convention == "delphi-iscm" ||
         args_.test_energy_convention == "delphi-corrected" ||
         delphi_nwt_parity_active());
    if (forced_iscm) {
      solve.solvent_phase_kcal_mol = solve.iscm_water_phase_kcal_mol;
      solve.vacuum_phase_kcal_mol = solve.iscm_vacuum_phase_kcal_mol;
      solve.energy_kcal_mol = solve.iscm_corrected_kcal_mol;
      solve.reaction_energy_source =
          delphi_nwt_parity_active()
              ? "delphi-nonlinear-iscm"
              : (args_.test_energy_convention == "delphi-corrected"
                     ? "delphi-corrected-iscm"
                     : "delphi-iscm");
      if (args_.model == "nonlinear" && nonlinear_energy_audit.status == "ok") {
        solve.energy_kcal_mol +=
            nonlinear_energy_audit.total_correction_kcal_mol;
        solve.reaction_energy_source += "+nonlinear";
      }
    } else if (legacy_grid || forced_grid ||
               args_.test_energy_convention == "delphi-corrected" ||
               args_.test_energy_convention == "delphi-iscm") {
      solve.solvent_phase_kcal_mol = grid_solvent_phase_kcal;
      solve.vacuum_phase_kcal_mol = grid_vacuum_phase_kcal;
      solve.energy_kcal_mol = solve.grid_corrected_kcal_mol;
      if (args_.test_energy_convention == "delphi-corrected")
        solve.reaction_energy_source = "delphi-corrected-grid-fallback";
      else if (args_.test_energy_convention == "delphi-iscm")
        solve.reaction_energy_source = "delphi-iscm-grid-fallback";
      else
        solve.reaction_energy_source = "grid";
    } else {
      solve.solvent_phase_kcal_mol = atom_solvent_phase_kcal;
      solve.vacuum_phase_kcal_mol = atom_vacuum_phase_kcal;
      solve.energy_kcal_mol = solve.atom_corrected_kcal_mol;
      solve.reaction_energy_source = "atom";
    }
  } else {
    if (args_.test_energy_convention == "delphi-iscm" ||
        args_.test_energy_convention == "delphi-corrected")
      solve.iscm_status = "missing_grid_field";
    solve.grid_corrected_kcal_mol = solve.atom_corrected_kcal_mol;
    solve.energy_kcal_mol = solve.atom_corrected_kcal_mol;
    solve.reaction_energy_source = "atom";
  }
  if (args_.test_energy_convention == "atom") {
    solve.solvent_phase_kcal_mol = atom_solvent_phase_kcal;
    solve.vacuum_phase_kcal_mol = atom_vacuum_phase_kcal;
    solve.energy_kcal_mol = solve.atom_corrected_kcal_mol;
    solve.reaction_energy_source = "atom";
  }
  const double atom_potential_scale =
      kCoulombKcalPerMol * field_to_native_factor() * kKcalToKJ;
  double atom_closure_kj_mol = 0.0;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const double solvent_potential = atom_potential_scale * solvent.atom_phi[i];
    const double vacuum_potential = atom_potential_scale * vacuum.atom_phi[i];
    const double reaction_potential = solvent_potential - vacuum_potential;
    const double atom_kcal_mol = 0.5 * kCoulombKcalPerMol *
                                 field_to_native_factor() * atoms[i].q *
                                 (solvent.atom_phi[i] - vacuum.atom_phi[i]);
    solve.atom_solvent_potential_kj_mol_e[i] = solvent_potential;
    solve.atom_vacuum_potential_kj_mol_e[i] = vacuum_potential;
    solve.atom_reaction_potential_kj_mol_e[i] = reaction_potential;
    solve.atom_reaction_kj_mol[i] = atom_kcal_mol * kKcalToKJ;
    atom_closure_kj_mol += solve.atom_reaction_kj_mol[i];
  }
  solve.atom_reaction_energy_closure_kj_mol = atom_closure_kj_mol;
  solve.atom_reaction_energy_closure_error_kj_mol =
      atom_closure_kj_mol - solve.atom_corrected_kcal_mol * kKcalToKJ;
  if (!args_.probe_file.empty()) {
    const auto probe_time_begin = std::chrono::steady_clock::now();
    solve.probe_energy_status = "missing_field";
    if (solvent.phi_grid.size() == solvent.shape.n) {
      const std::vector<Atom> probe_atoms = load_probe_atoms(args_.probe_file);
      auto uppercase_alnum = [](const std::string &value) {
        std::string out;
        out.reserve(value.size());
        for (char ch : value) {
          if (std::isalnum(static_cast<unsigned char>(ch)))
            out.push_back(static_cast<char>(
                std::toupper(static_cast<unsigned char>(ch))));
        }
        return out;
      };
      double q_phi = 0.0;
      std::size_t matched = 0;
      std::string matched_ids;
      std::string matched_groups;
      auto cubic_interp = [](const double p[4], double x) {
        return x * x * (p[3] - 2.0 * p[2] + p[1]) / 2.0 +
               x * (4.0 * p[2] - 3.0 * p[1] - p[3]) / 2.0 + p[1];
      };
      auto interpolate_probe_phi = [&](double x, double y, double z,
                                       double &phi) {
        const double gx = (x - solvent.shape.ox) / solvent.shape.h;
        const double gy = (y - solvent.shape.oy) / solvent.shape.h;
        const double gz = (z - solvent.shape.oz) / solvent.shape.h;
        if (gx >= 1.0 && gy >= 1.0 && gz >= 1.0 &&
            gx < static_cast<double>(solvent.shape.nx - 2) &&
            gy < static_cast<double>(solvent.shape.ny - 2) &&
            gz < static_cast<double>(solvent.shape.nz - 2)) {
          const auto ix = static_cast<std::size_t>(std::floor(gx));
          const auto iy = static_cast<std::size_t>(std::floor(gy));
          const auto iz = static_cast<std::size_t>(std::floor(gz));
          if (std::abs(gx - static_cast<double>(ix)) <= 1.0e-6 &&
              std::abs(gy - static_cast<double>(iy)) <= 1.0e-6 &&
              std::abs(gz - static_cast<double>(iz)) <= 1.0e-6) {
            phi = solvent.phi_grid[idx3(ix, iy, iz, solvent.shape.nx,
                                        solvent.shape.ny)];
            return true;
          }
          const double fx = gx - static_cast<double>(ix);
          const double fy = gy - static_cast<double>(iy);
          const double fz = gz - static_cast<double>(iz);
          double xvals[4]{};
          for (int lx = 0; lx < 4; ++lx) {
            double yvals[4]{};
            for (int ly = 0; ly < 4; ++ly) {
              double zvals[4]{};
              for (int lz = 0; lz < 4; ++lz) {
                const std::size_t px =
                    static_cast<std::size_t>(static_cast<int>(ix) + lx - 1);
                const std::size_t py =
                    static_cast<std::size_t>(static_cast<int>(iy) + ly - 1);
                const std::size_t pz =
                    static_cast<std::size_t>(static_cast<int>(iz) + lz - 1);
                zvals[lz] = solvent.phi_grid[idx3(px, py, pz, solvent.shape.nx,
                                                  solvent.shape.ny)];
              }
              yvals[ly] = cubic_interp(zvals, fz);
            }
            xvals[lx] = cubic_interp(yvals, fy);
          }
          phi = cubic_interp(xvals, fx);
          return true;
        }
        return interpolate_field_value(solvent.shape, solvent.phi_grid, x, y, z,
                                       phi);
      };
      for (std::size_t i = 0; i < probe_atoms.size(); ++i) {
        const Atom &atom = probe_atoms[i];
        double phi = 0.0;
        if (!interpolate_probe_phi(atom.x, atom.y, atom.z, phi))
          continue;
        q_phi += atom.q * phi;
        if (!matched_ids.empty())
          matched_ids += ';';
        matched_ids += std::to_string(i + 1);
        if (!matched_groups.empty())
          matched_groups += ';';
        matched_groups += uppercase_letters(atom.resname) +
                          std::to_string(atom.resnr) + ':' +
                          uppercase_alnum(atom.atom_name);
        ++matched;
      }
      if (map_result != nullptr && args_.parity_dump != "off") {
        const std::string sample_filename = "pbe_probe_atom_samples.csv";
        const bool need_header =
            !std::filesystem::exists(sample_filename) ||
            std::filesystem::file_size(sample_filename) == 0;
        std::ofstream probe_out(sample_filename, std::ios::out | std::ios::app);
        if (probe_out.is_open()) {
          if (need_header) {
            probe_out
                << "trajectory_id,frame_index,global_index,time_ps,atom_index,"
                   "atom_name,resname,resnr,x_A,y_A,z_A,q,phi_native,"
                   "phi_kT_per_e,contribution_kT,status\n";
          }
          const double kt_kj = kKBKJPerMolK * std::max(1.0, args_.temp);
          const double native_to_kt_per_e =
              kt_kj > 0.0 ? (kCoulombKcalPerMol * kKcalToKJ / kt_kj) : 0.0;
          for (std::size_t i = 0; i < probe_atoms.size(); ++i) {
            const Atom &atom = probe_atoms[i];
            double phi = 0.0;
            const bool ok = interpolate_probe_phi(atom.x, atom.y, atom.z, phi);
            const double phi_native = ok ? field_to_native_factor() * phi : 0.0;
            const double phi_kt = phi_native * native_to_kt_per_e;
            probe_out << map_result->trajectory_id << ','
                      << map_result->frame_index << ','
                      << map_result->global_index << ','
                      << std::setprecision(10) << map_result->time_ps << ','
                      << (i + 1) << ',' << csv_escape(atom.atom_name) << ','
                      << csv_escape(atom.resname) << ',' << atom.resnr << ','
                      << atom.x << ',' << atom.y << ',' << atom.z << ','
                      << atom.q << ',' << phi_native << ',' << phi_kt << ','
                      << (ok ? 0.5 * atom.q * phi_kt : 0.0) << ','
                      << (ok ? "ok" : "outside_grid") << '\n';
          }
        }
      }
      solve.probe_matched_atoms = matched;
      solve.probe_matched_ids = matched_ids;
      solve.probe_matched_groups = matched_groups;
      solve.probe_matching_semantics = "delphi_frc_probe_file";
      if (matched > 0) {
        solve.probe_energy_kcal_mol =
            0.5 * kCoulombKcalPerMol * field_to_native_factor() * q_phi;
        solve.probe_energy_kj_mol = solve.probe_energy_kcal_mol * kKcalToKJ;
        const double kt_kj = kKBKJPerMolK * std::max(1.0, args_.temp);
        solve.probe_energy_kt =
            kt_kj > 0.0 ? solve.probe_energy_kj_mol / kt_kj : 0.0;
        solve.probe_energy_status = "ok";
      } else {
        solve.probe_energy_status = "no_matching_atoms";
      }
    }
    const double probe_ms =
        elapsed_ms(probe_time_begin, std::chrono::steady_clock::now());
    probe_time_ms += probe_ms;
    add_timing_ms(TimingBucket::Probe, probe_ms);
    add_timing_ms(TimingBucket::ProbeEnergy, probe_ms);
  }
  solve.energy_kj_mol = solve.energy_kcal_mol * kKcalToKJ;
  solve.energy_kt =
      solve.energy_kj_mol / (kKBKJPerMolK * std::max(1.0, args_.temp));
  solve.solvent_phase_kj_mol = solve.solvent_phase_kcal_mol * kKcalToKJ;
  solve.solvent_phase_kt =
      kt_kcal > 0.0 ? solve.solvent_phase_kcal_mol / kt_kcal : 0.0;
  solve.vacuum_phase_kj_mol = solve.vacuum_phase_kcal_mol * kKcalToKJ;
  solve.vacuum_phase_kt =
      kt_kcal > 0.0 ? solve.vacuum_phase_kcal_mol / kt_kcal : 0.0;
  solve.atom_corrected_kj_mol = solve.atom_corrected_kcal_mol * kKcalToKJ;
  solve.atom_corrected_kt =
      kt_kcal > 0.0 ? solve.atom_corrected_kcal_mol / kt_kcal : 0.0;
  solve.grid_corrected_kj_mol = solve.grid_corrected_kcal_mol * kKcalToKJ;
  solve.grid_corrected_kt =
      kt_kcal > 0.0 ? solve.grid_corrected_kcal_mol / kt_kcal : 0.0;
  solve.coulomb_energy_kj_mol = solve.coulomb_energy_kcal_mol * kKcalToKJ;
  solve.coulomb_energy_kt =
      kt_kcal > 0.0 ? solve.coulomb_energy_kcal_mol / kt_kcal : 0.0;
  solve.converged = solve.solvent.converged && solve.vacuum.converged;
  const double energy_ms =
      elapsed_ms(energy_time_begin, std::chrono::steady_clock::now()) -
      rpb_source_time_ms - probe_time_ms;
  add_timing_ms(TimingBucket::Energy, std::max(0.0, energy_ms));
  solve.pbsa_energy_ms = std::max(0.0, energy_ms);
  if (map_result != nullptr) {
    const auto audit_output_begin = std::chrono::steady_clock::now();
    maybe_write_iscm_surface_charge_samples(*map_result, iscm_audit);
    maybe_write_rpb_source_audit(*map_result, rpb_audit);
    maybe_write_rpb_source_top(*map_result, solve.shape, *rpb_source_top_map,
                               *rpb_coulomb_top_map);
    maybe_write_barnase_forensic_audits(*map_result, atoms, solvent, vacuum,
                                        solve, solvent_forensic);
    maybe_write_parity_map_dumps(*map_result, atoms, solvent, vacuum, solve);
    add_timing_ms(
        TimingBucket::AuditOutput,
        elapsed_ms(audit_output_begin, std::chrono::steady_clock::now()));
  }
  return solve;
}

AnalyzerImpl::ReactionSolve
AnalyzerImpl::solve_tabi_reaction_field(const std::vector<Atom> &atoms,
                                        TitanExecutor &executor) const {
  ReactionSolve solve;
  TabiPrepareProgressWriter prepare_progress(args_, atoms.size(),
                                             configured_threads_);
  prepare_progress.write("tabi_prepare_start", "running");
  std::vector<Tabi::Atom> tabi_atoms;
  tabi_atoms.reserve(atoms.size());
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  for (const Atom &atom : atoms) {
    Tabi::Atom tabi_atom;
    tabi_atom.x = atom.x;
    tabi_atom.y = atom.y;
    tabi_atom.z = atom.z;
    tabi_atom.q = atom.q;
    tabi_atom.radius = effective_surface_radius(atom);
    tabi_atoms.push_back(tabi_atom);
    const double surface_radius =
        tabi_atom.radius + (args_.surface == "sas" ? args_.probe_radius : 0.0);
    min_x = std::min(min_x, atom.x - surface_radius);
    min_y = std::min(min_y, atom.y - surface_radius);
    min_z = std::min(min_z, atom.z - surface_radius);
    max_x = std::max(max_x, atom.x + surface_radius);
    max_y = std::max(max_y, atom.y + surface_radius);
    max_z = std::max(max_z, atom.z + surface_radius);
  }
  prepare_progress.write("pqr_loaded", "running");

  Tabi::Options options;
  options.eps_in = args_.int_diel;
  options.eps_out = args_.ext_diel;
  options.salt_m = args_.salt;
  options.temperature_k = args_.temp;
  options.max_iter = args_.max_iter;
  options.residual_tol = args_.rms_tol;
  options.gmres_restart = args_.tabi_gmres_restart;
  options.tree_preset = args_.tabi_tree_preset;
  options.sum_mode = args_.tabi_sum;
  options.probe_radius = args_.probe_radius;
  options.surface_mode = args_.surface;
  options.surface_geometry = args_.tabi_geometry;
  options.diagonal_mode = args_.tabi_diagonal;
  options.surface_point_count = args_.tabi_surface_points;
  options.surface_output = args_.tabi_surface_output;
  options.surface_only = args_.tabi_surface_only;
  options.highres_surface = args_.tabi_highres_surface;
  options.calderon_trace_audit = args_.tabi_calderon_trace_audit;
  options.calderon_aux_surface = args_.tabi_calderon_aux_surface;
  options.calderon_aux_grid_spacing = args_.tabi_calderon_aux_grid_spacing;
  options.primary_surface = args_.tabi_primary_surface;
  options.analytic_ses_engine = args_.tabi_analytic_ses_engine;
  options.ses_grid_spacing = args_.tabi_ses_grid_spacing;
  options.ses_max_grid_vertices = args_.tabi_ses_max_grid_vertices;
  options.sphere_mesh_mode = args_.tabi_sphere_mesh;
  options.sphere_patch_clip = args_.tabi_sphere_patch_clip;
  options.tree_leaf_size = args_.tabi_tree_leaf_size;
  options.curved_pair_cache_mb = args_.tabi_curved_pair_cache_mb;
  options.tree_leaf_size_explicit = args_.tabi_tree_leaf_size_explicit;
  options.tree_opening_theta = args_.tabi_tree_opening_theta;
  options.energy_far_eta = args_.tabi_energy_far_eta;
  options.ses_remesh_edge = args_.tabi_ses_remesh_edge;
  options.ses_remesh_iterations = args_.tabi_ses_remesh_iterations;
  options.ses_charge_refine = args_.tabi_ses_charge_refine;
  options.ses_charge_refine_min_edge = args_.tabi_ses_charge_refine_min_edge;
  options.ses_charge_refine_min_charge = args_.tabi_ses_charge_refine_min_charge;
  options.curved_sample_order = args_.tabi_curved_sample_order;
  options.curved_near_zone_radii = args_.tabi_curved_near_zone_radii;
  options.tree_opening_theta_explicit =
      args_.tabi_tree_opening_theta_explicit;
  options.taylor_order = args_.tabi_taylor_order;
  options.derivative_mode = args_.tabi_derivatives;
  options.formulation = args_.tabi_formulation;
  options.discretization = args_.tabi_discretization;
  options.energy_projection = args_.tabi_energy_projection;
  options.preconditioner = args_.tabi_precond;
  options.preconditioner_block_near = args_.tabi_precond_block_near;
  options.preconditioner_block_panel_cap = args_.tabi_precond_block_panel_cap;
  options.preconditioner_overlap_panel_cap =
      args_.tabi_precond_overlap_panel_cap;
  options.preconditioner_ilu_neighbor_cap = args_.tabi_precond_ilu_neighbor_cap;
  options.preconditioner_ilu_drop_tol = args_.tabi_precond_ilu_drop_tol;
  options.preconditioner_ilu_regularization =
      args_.tabi_precond_ilu_regularization;
  options.preconditioner_ilu_regularization_max =
      args_.tabi_precond_ilu_regularization_max;
  options.preconditioner_memory_budget_mb = args_.tabi_precond_memory_budget_mb;
  options.preconditioner_coarse_aggregate_cap =
      args_.tabi_precond_coarse_aggregate_cap;
  options.preconditioner_coarse_base = args_.tabi_precond_coarse_base;
  options.preconditioner_coarse_verify = args_.tabi_precond_coarse_verify;
  options.near_integration = args_.tabi_near_integration;
  options.quad_buffer = args_.tabi_quad_buffer;
  options.leaf_near_policy = args_.tabi_leaf_near_policy;
  options.leaf_panel_kernel = args_.tabi_leaf_panel_kernel;
  options.adaptive_level = args_.tabi_adaptive_level;
  options.cluster_policy = args_.tabi_cluster_policy;
  options.leaf_pair_kernel = args_.tabi_leaf_pair_kernel;
  options.leaf_kernel_profile = args_.tabi_leaf_kernel_profile;
  options.leaf_work_profile = args_.tabi_leaf_work_profile;
  options.matvec_cost_profile = args_.tabi_matvec_cost_profile;
  options.jcp_direct_kernel = args_.tabi_jcp_direct_kernel;
  options.leaf_pair_selection_profile = args_.tabi_leaf_pair_selection_profile;
  options.leaf_pair_probe_cache = args_.tabi_leaf_pair_probe_cache;
  options.leaf_pair_target_group = args_.tabi_leaf_pair_target_group;
  options.leaf_pair_directness_probe = args_.tabi_leaf_pair_directness_probe;
  options.leaf_pair_directness_cache = args_.tabi_leaf_pair_directness_cache;
  options.leaf_pair_directness_verify = args_.tabi_leaf_pair_directness_verify;
  options.leaf_pair_scheduler = args_.tabi_leaf_pair_scheduler;
  options.leaf_pair_worklist_max_active =
      args_.tabi_leaf_pair_worklist_max_active;
  options.leaf_pair_worklist_verify = args_.tabi_leaf_pair_worklist_verify;
  options.leaf_block_verify = args_.tabi_leaf_block_verify;
  options.leaf_mac = args_.tabi_leaf_mac;
  options.fmm_m2l = args_.tabi_fmm_m2l;
  options.fmm_engine = args_.tabi_fmm_engine;
  options.bie_fmm_cartesian_order = args_.bie_fmm_cartesian_order;
  options.kifmm_order = args_.tabi_kifmm_order;
  options.kifmm_equiv_scale = args_.tabi_kifmm_equiv_scale;
  options.kifmm_check_scale = args_.tabi_kifmm_check_scale;
  options.kifmm_solver = args_.tabi_kifmm_solver;
  options.kifmm_profile = args_.tabi_kifmm_profile;
  options.kifmm_full_apply = args_.tabi_kifmm_full_apply;
  options.kifmm_ilu_graph = args_.tabi_kifmm_ilu_graph;
  options.kifmm_hierarchy_order = args_.tabi_kifmm_hierarchy_order;
  options.kifmm_hierarchy_theta = args_.tabi_kifmm_hierarchy_theta;
  options.kifmm_hierarchy_real_panel_guard =
      args_.tabi_kifmm_hierarchy_real_panel_guard;
  options.kifmm_hierarchy_guard_margin_factor =
      args_.tabi_kifmm_hierarchy_guard_margin_factor;
  options.kifmm_hierarchy_tree_stop_policy =
      args_.tabi_kifmm_hierarchy_tree_stop_policy;
  options.kifmm_hierarchy_tree_panel_radius_factor =
      args_.tabi_kifmm_hierarchy_tree_panel_radius_factor;
  options.kifmm_hierarchy_tree_occupancy_stop =
      args_.tabi_kifmm_hierarchy_tree_occupancy_stop;
  options.kifmm_hierarchy_m2l_eval_policy =
      args_.tabi_kifmm_hierarchy_m2l_eval_policy;
  options.kifmm_hierarchy_matrix_cache_policy =
      args_.tabi_kifmm_hierarchy_matrix_cache_policy;
  options.kifmm_hierarchy_matrix_cache_limit_mb =
      args_.tabi_kifmm_hierarchy_matrix_cache_limit_mb;
  options.kifmm_hierarchy_gemm_batch_policy =
      args_.tabi_kifmm_hierarchy_gemm_batch_policy;
  options.kifmm_hierarchy_m2l_parallel_policy =
      args_.tabi_kifmm_hierarchy_m2l_parallel_policy;
  options.kifmm_hierarchy_m2l_parallel_workers =
      args_.tabi_kifmm_hierarchy_m2l_parallel_workers;
  options.cartesian_k4_order = args_.tabi_cartesian_k4_order;
  options.cartesian_k4_theta = args_.tabi_cartesian_k4_theta;
  options.cartesian_k4_geometry_theta = args_.tabi_cartesian_k4_geometry_theta;
  options.cartesian_k4_tail_tol = args_.tabi_cartesian_k4_tail_tol;
  options.target_batch_size = args_.tabi_target_batch_size;
  options.leaf_pair_min_source_panels = args_.tabi_leaf_pair_min_source_panels;
  options.leaf_pair_min_targets = args_.tabi_leaf_pair_min_targets;
  options.leaf_pair_max_lookahead = args_.tabi_leaf_pair_max_lookahead;
  options.interaction_mode = args_.tabi_interaction_mode;
  options.node_bounds = args_.tabi_node_bounds;
  options.tree_builder = args_.tabi_tree_builder;
  options.expansion_center = args_.tabi_expansion_center;
  options.fragmentation_safety = args_.tabi_fragmentation_safety;
  options.mac_safety = args_.tabi_mac_safety;
  options.mac_gap_factor = args_.tabi_mac_gap_factor;
  options.moment_audit = args_.tabi_moment_audit;
  options.moment_construction = args_.tabi_moment_construction;
  options.moment_sum = args_.tabi_moment_sum;
  options.convergence_audit = args_.tabi_convergence_audit;
  options.convergence_safety = args_.tabi_convergence_safety;
  options.convergence_rho_cap = args_.tabi_convergence_rho_cap;
  options.mac_aspect_cap = args_.tabi_mac_aspect_cap;
  options.convergence_estimator = args_.tabi_convergence_estimator;
  options.convergence_tail_threshold = args_.tabi_convergence_tail_threshold;
  options.convergence_estimator_error_cap =
      args_.tabi_convergence_estimator_error_cap;
  options.convergence_unstable_rho_cap =
      args_.tabi_convergence_unstable_rho_cap;
  options.convergence_calibration = args_.tabi_convergence_calibration;
  options.convergence_dielectric_amp = args_.tabi_convergence_dielectric_amp;
  options.convergence_estimator_model = args_.tabi_convergence_estimator_model;
  options.convergence_channel_calibration =
      args_.tabi_convergence_channel_calibration;
  options.convergence_split_policy = args_.tabi_convergence_split_policy;
  options.convergence_split_selectivity =
      args_.tabi_convergence_split_selectivity;
  options.convergence_split_absolute_floor =
      args_.tabi_convergence_split_absolute_floor;
  options.convergence_overreject_audit =
      args_.tabi_convergence_overreject_audit;
  options.convergence_overreject_detail =
      args_.tabi_convergence_overreject_detail;
  options.convergence_shape_audit = args_.tabi_convergence_shape_audit;
  options.convergence_kernel_response_audit =
      args_.tabi_convergence_kernel_response_audit;
  options.channel_missed_semantics = args_.tabi_channel_missed_semantics;
  options.accepted_miss_diagnostics = args_.tabi_accepted_miss_diagnostics;
  options.accepted_miss_absolute_floor =
      args_.tabi_accepted_miss_absolute_floor;
  options.convergence_k3_accepted_rho_cap =
      args_.tabi_convergence_k3_accepted_rho_cap;
  options.mac_distance_policy = args_.tabi_mac_distance_policy;
  options.mac_surface_margin_factor = args_.tabi_mac_surface_margin_factor;
  options.mac_surface_margin_policy = args_.tabi_mac_surface_margin_policy;
  options.mac_surface_margin_aspect_threshold =
      args_.tabi_mac_surface_margin_aspect_threshold;
  options.mac_surface_margin_rho_trigger =
      args_.tabi_mac_surface_margin_rho_trigger;
  options.mac_surface_margin_soft_aspect_threshold =
      args_.tabi_mac_surface_margin_soft_aspect_threshold;
  options.mac_surface_margin_soft_rho_trigger =
      args_.tabi_mac_surface_margin_soft_rho_trigger;
  options.mac_surface_margin_soft_factor =
      args_.tabi_mac_surface_margin_soft_factor;
  options.expansion_audit = args_.tabi_expansion_audit;
  options.expansion_audit_channel = args_.tabi_expansion_audit_channel;
  options.expansion_audit_sample_count =
      args_.tabi_expansion_audit_sample_count;
  options.expansion_audit_max_panels = args_.tabi_expansion_audit_max_panels;
  options.matvec_audit = args_.tabi_matvec_audit;
  options.matvec_audit_sample_count = args_.tabi_matvec_audit_sample_count;
  options.node_patch_operator_audit =
      args_.tabi_node_patch_operator_audit;
  options.fmm_cartesian_audit = args_.tabi_fmm_cartesian_audit;
  options.fmm_cartesian_error_audit =
      args_.tabi_fmm_cartesian_error_audit;
  options.fmm_local = args_.tabi_fmm_local;
  options.fmm_local_refine = args_.tabi_fmm_local_refine;
  options.fmm_local_refine_tol = args_.tabi_fmm_local_refine_tol;
  options.fmm_local_cache = args_.tabi_fmm_local_cache;
  options.fmm_local_cache_budget_mb = args_.tabi_fmm_local_cache_budget_mb;
  options.fmm_local_fallback = args_.tabi_fmm_local_fallback;
  options.fmm_local_plan_builder = args_.tabi_fmm_local_plan_builder;
  options.fmm_local_refine_probe_count =
      args_.tabi_fmm_local_refine_probe_count;
  options.fmm_local_refine_probe_mode = args_.tabi_fmm_local_refine_probe_mode;
  options.fmm_local_geometry_guard = args_.tabi_fmm_local_geometry_guard;
  options.fmm_local_geometry_theta = args_.tabi_fmm_local_geometry_theta;
  options.fmm_local_geometry_sweep = args_.tabi_fmm_local_geometry_sweep;
  options.true_residual_probe = args_.tabi_true_residual_probe;
  options.true_residual_probe_scope = args_.tabi_true_residual_probe_scope;
  options.true_residual_probe_interval =
      args_.tabi_true_residual_probe_interval;
  options.true_residual_sample_count = args_.tabi_true_residual_sample_count;
  options.progress = args_.tabi_progress;
  if (options.progress == "auto" && args_.timing != "detail")
    options.progress = "off";
  options.progress_interval_seconds = args_.tabi_progress_interval;
  prepare_progress.write("tabi_options_built", "running");
  prepare_progress.write("solver_enter", "running");
  const Tabi::Result tabi =
      Tabi::solve_linear_pb(tabi_atoms, options, executor);

  solve.pbe_backend = "tabi";
  solve.tabi_surface_build_ms = tabi.surface_build_ms;
  solve.tabi_surface_panels = tabi.panel_count;
  solve.tabi_surface_point_count_requested = tabi.surface_point_count_requested;
  solve.tabi_panel_density_per_a2 = tabi.panel_density_per_a2;
  solve.tabi_panel_budget_status = tabi.panel_budget_status;
  solve.tabi_charge_resolution = tabi.charge_resolution;
  solve.tabi_surface_exposed_points = tabi.exposed_points;
  solve.tabi_surface_buried_points = tabi.buried_points;
  solve.tabi_tree_build_ms = tabi.tree_build_ms;
  solve.tabi_tree_nodes = tabi.tree_nodes;
  solve.tabi_tree_leaf_count = tabi.tree_leaf_count;
  solve.tabi_near_interactions = tabi.near_interactions;
  solve.tabi_far_interactions = tabi.far_interactions;
  solve.tabi_matvec_mode = tabi.matvec_mode;
  solve.tabi_fmm_m2l_mode = tabi.fmm_m2l_mode;
  solve.tabi_fmm_m2l_expansion_order = tabi.fmm_m2l_expansion_order;
  solve.tabi_fmm_m2l_near_pairs = tabi.fmm_m2l_near_pairs;
  solve.tabi_fmm_m2l_far_pairs = tabi.fmm_m2l_far_pairs;
  solve.tabi_fmm_m2l_setup_ms = tabi.fmm_m2l_setup_ms;
  solve.tabi_fmm_m2l_apply_ms = tabi.fmm_m2l_apply_ms;
  solve.tabi_fmm_m2l_finite_outputs = tabi.fmm_m2l_finite_outputs;
  solve.tabi_bie_cartesian_full_fmm_active =
      tabi.bie_cartesian_full_fmm_active;
  solve.tabi_bie_cartesian_full_fmm_requested =
      tabi.bie_cartesian_full_fmm_requested;
  solve.tabi_bie_cartesian_full_fmm_effective =
      tabi.bie_cartesian_full_fmm_effective;
  solve.tabi_bie_cartesian_full_fmm_order =
      tabi.bie_cartesian_full_fmm_order;
  solve.tabi_bie_cartesian_full_fmm_leaf_size =
      tabi.bie_cartesian_full_fmm_leaf_size;
  solve.tabi_bie_cartesian_full_fmm_opening_theta =
      tabi.bie_cartesian_full_fmm_opening_theta;
  solve.tabi_bie_cartesian_full_fmm_tree_hash =
      tabi.bie_cartesian_full_fmm_tree_hash;
  solve.tabi_bie_cartesian_full_fmm_worklist_hash =
      tabi.bie_cartesian_full_fmm_worklist_hash;
  solve.tabi_bie_cartesian_full_fmm_policy_hash =
      tabi.bie_cartesian_full_fmm_policy_hash;
  solve.tabi_bie_cartesian_full_fmm_nodes =
      tabi.bie_cartesian_full_fmm_nodes;
  solve.tabi_bie_cartesian_full_fmm_far_pairs =
      tabi.bie_cartesian_full_fmm_far_pairs;
  solve.tabi_bie_cartesian_full_fmm_direct_pairs =
      tabi.bie_cartesian_full_fmm_direct_pairs;
  solve.tabi_bie_cartesian_full_fmm_direct_panel_pairs_expected =
      tabi.bie_cartesian_full_fmm_direct_panel_pairs_expected;
  solve.tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_last =
      tabi.bie_cartesian_full_fmm_direct_panel_pairs_applied_last;
  solve.tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_total =
      tabi.bie_cartesian_full_fmm_direct_panel_pairs_applied_total;
  solve.tabi_bie_cartesian_full_fmm_cache_entries =
      tabi.bie_cartesian_full_fmm_cache_entries;
  solve.tabi_bie_cartesian_full_fmm_cache_hits =
      tabi.bie_cartesian_full_fmm_cache_hits;
  solve.tabi_bie_cartesian_full_fmm_cache_bytes =
      tabi.bie_cartesian_full_fmm_cache_bytes;
  solve.tabi_bie_cartesian_full_fmm_plan_bytes =
      tabi.bie_cartesian_full_fmm_plan_bytes;
  solve.tabi_bie_cartesian_full_fmm_workspace_bytes =
      tabi.bie_cartesian_full_fmm_workspace_bytes;
  solve.tabi_bie_cartesian_full_fmm_tree_seconds =
      tabi.bie_cartesian_full_fmm_tree_seconds;
  solve.tabi_bie_cartesian_full_fmm_worklist_seconds =
      tabi.bie_cartesian_full_fmm_worklist_seconds;
  solve.tabi_bie_cartesian_full_fmm_cache_seconds =
      tabi.bie_cartesian_full_fmm_cache_seconds;
  solve.tabi_bie_cartesian_full_fmm_audit_seconds =
      tabi.bie_cartesian_full_fmm_audit_seconds;
  solve.tabi_bie_cartesian_full_fmm_p2m_seconds =
      tabi.bie_cartesian_full_fmm_p2m_seconds;
  solve.tabi_bie_cartesian_full_fmm_m2m_seconds =
      tabi.bie_cartesian_full_fmm_m2m_seconds;
  solve.tabi_bie_cartesian_full_fmm_m2l_seconds =
      tabi.bie_cartesian_full_fmm_m2l_seconds;
  solve.tabi_bie_cartesian_full_fmm_l2l_seconds =
      tabi.bie_cartesian_full_fmm_l2l_seconds;
  solve.tabi_bie_cartesian_full_fmm_l2p_seconds =
      tabi.bie_cartesian_full_fmm_l2p_seconds;
  solve.tabi_bie_cartesian_full_fmm_direct_seconds =
      tabi.bie_cartesian_full_fmm_direct_seconds;
  solve.tabi_bie_cartesian_full_fmm_m2m_parity_error =
      tabi.bie_cartesian_full_fmm_m2m_parity_error;
  solve.tabi_bie_cartesian_full_fmm_l2l_parity_error =
      tabi.bie_cartesian_full_fmm_l2l_parity_error;
  solve.tabi_fmm_local_gmres_active = tabi.fmm_local_gmres_active;
  solve.tabi_fmm_local_prepare_seconds = tabi.fmm_local_prepare_seconds;
  solve.tabi_fmm_local_plan_build_count = tabi.fmm_local_plan_build_count;
  solve.tabi_fmm_local_cache_build_count = tabi.fmm_local_cache_build_count;
  solve.tabi_fmm_local_gmres_matvec_calls = tabi.fmm_local_gmres_matvec_calls;
  solve.tabi_fmm_local_gmres_apply_seconds_total =
      tabi.fmm_local_gmres_apply_seconds_total;
  solve.tabi_fmm_local_gmres_apply_seconds_mean =
      tabi.fmm_local_gmres_apply_seconds_mean;
  solve.tabi_fmm_local_moment_seconds_total =
      tabi.fmm_local_moment_seconds_total;
  solve.tabi_fmm_local_m2l_seconds_total = tabi.fmm_local_m2l_seconds_total;
  solve.tabi_fmm_local_fallback_seconds_total =
      tabi.fmm_local_fallback_seconds_total;
  solve.tabi_fmm_local_l2p_seconds_total = tabi.fmm_local_l2p_seconds_total;
  solve.tabi_fmm_local_sampled_direct_true_residual =
      tabi.fmm_local_sampled_direct_true_residual;
  solve.tabi_fmm_local_operator_finite = tabi.fmm_local_operator_finite;
  solve.tabi_fmm_local_refine_probe_mode = tabi.fmm_local_refine_probe_mode;
  solve.tabi_fmm_local_refine_probe_count = tabi.fmm_local_refine_probe_count;
  solve.tabi_fmm_local_solution_probe_used = tabi.fmm_local_solution_probe_used;
  solve.tabi_fmm_local_linf_max_abs_tail = tabi.fmm_local_linf_max_abs_tail;
  solve.tabi_fmm_local_linf_max_relative_tail =
      tabi.fmm_local_linf_max_relative_tail;
  solve.tabi_fmm_local_linf_outlier_panels = tabi.fmm_local_linf_outlier_panels;
  solve.tabi_fmm_local_geometry_guard = tabi.fmm_local_geometry_guard;
  solve.tabi_fmm_local_geometry_theta = tabi.fmm_local_geometry_theta;
  solve.tabi_fmm_local_geometry_guard_rejected_cells =
      tabi.fmm_local_geometry_guard_rejected_cells;
  solve.tabi_fmm_local_geometry_guard_direct_fallback_pairs =
      tabi.fmm_local_geometry_guard_direct_fallback_pairs;
  solve.tabi_fmm_local_geometry_gap_min = tabi.fmm_local_geometry_gap_min;
  solve.tabi_fmm_local_geometry_ratio_max = tabi.fmm_local_geometry_ratio_max;
  solve.tabi_status = tabi.status;
  solve.tabi_convergence_reason = tabi.convergence_reason;
  solve.tabi_residual_history = tabi.residual_history;
  solve.tabi_gmres_restart_requested = tabi.gmres_restart_requested;
  solve.tabi_gmres_restart_effective = tabi.gmres_restart_effective;
  solve.tabi_best_residual = tabi.best_residual;
  solve.tabi_best_residual_iter = tabi.best_residual_iter;
  solve.tabi_final_residual = tabi.final_residual;
  solve.tabi_best_preconditioned_residual = tabi.best_preconditioned_residual;
  solve.tabi_final_preconditioned_residual = tabi.final_preconditioned_residual;
  solve.tabi_true_residual_abs = tabi.true_residual_abs;
  solve.tabi_true_residual_rel = tabi.true_residual_rel;
  solve.tabi_true_residual_probe_mode = tabi.true_residual_probe_mode;
  solve.tabi_true_residual_probe_scope = tabi.true_residual_probe_scope;
  solve.tabi_true_residual_probe_interval = tabi.true_residual_probe_interval;
  solve.tabi_true_residual_sample_count = tabi.true_residual_sample_count;
  solve.tabi_true_residual_probe_calls = tabi.true_residual_probe_calls;
  solve.tabi_true_residual_probe_ms = tabi.true_residual_probe_ms;
  solve.tabi_sampled_true_residual_rel = tabi.sampled_true_residual_rel;
  solve.tabi_sampled_true_residual_rows = tabi.sampled_true_residual_rows;
  solve.tabi_sampled_probe_calls = tabi.sampled_probe_calls;
  solve.tabi_sampled_probe_ms = tabi.sampled_probe_ms;
  solve.tabi_full_true_residual_confirm_calls =
      tabi.full_true_residual_confirm_calls;
  solve.tabi_full_probe_ms = tabi.full_true_residual_confirm_ms;
  solve.tabi_reaction_energy_source = tabi.reaction_energy_source;
  solve.tabi_rhs_ms = tabi.rhs_ms;
  solve.tabi_gmres_ms = tabi.gmres_ms;
  solve.tabi_matvec_ms_total = tabi.matvec_ms_total;
  solve.tabi_matvec_ms_avg = tabi.matvec_ms_avg;
  solve.tabi_matvec_calls = tabi.matvec_calls;
  solve.tabi_tree_vs_direct_matvec_rel = tabi.tree_vs_direct_matvec_rel;
  solve.tabi_matvec_audit_rows = tabi.matvec_audit_rows;
  solve.tabi_matvec_audit_ms = tabi.matvec_audit_ms;
  solve.tabi_fmm_cartesian_pre_audit = tabi.fmm_cartesian_pre_audit;
  solve.tabi_fmm_cartesian_solution_audit =
      tabi.fmm_cartesian_solution_audit;
  solve.tabi_fmm_cartesian_channel_audit =
      tabi.fmm_cartesian_channel_audit;
  solve.tabi_fmm_cartesian_retry = tabi.fmm_cartesian_retry;
  solve.tabi_orthogonalization_ms = tabi.orthogonalization_ms;
  solve.tabi_preconditioner_ms = tabi.preconditioner_ms;
  solve.tabi_surface_neighbor_ms = tabi.surface_neighbor_ms;
  solve.tabi_surface_mesh_ms = tabi.surface_mesh_ms;
  solve.tabi_solid_angle_ms = tabi.solid_angle_ms;
  solve.tabi_tree_moment_ms = tabi.tree_moment_ms;
  solve.tabi_tree_traversal_ms = tabi.tree_traversal_ms;
  solve.tabi_matvec_cost_profile = tabi.matvec_cost_profile;
  solve.tabi_leaf_panels_per_second = tabi.leaf_panels_per_second;
  solve.tabi_clean_leaf_panels_per_second = tabi.clean_leaf_panels_per_second;
  solve.tabi_tree_traversal_fraction = tabi.tree_traversal_fraction;
  solve.tabi_near_eval_ms = tabi.near_eval_ms;
  solve.tabi_far_eval_ms = tabi.far_eval_ms;
  solve.tabi_singular_near_eval_ms = tabi.singular_near_eval_ms;
  solve.tabi_leaf_near_eval_ms = tabi.leaf_near_eval_ms;
  solve.tabi_leaf_panel_eval_ms = tabi.leaf_panel_eval_ms;
  solve.tabi_leaf_tile_eval_ms = tabi.leaf_tile_eval_ms;
  solve.tabi_leaf_residual_eval_ms = tabi.leaf_residual_eval_ms;
  solve.tabi_smooth_leaf_node_eval_ms = tabi.smooth_leaf_node_eval_ms;
  solve.tabi_singular_near_interactions = tabi.singular_near_interactions;
  solve.tabi_leaf_near_interactions = tabi.leaf_near_interactions;
  solve.tabi_far_node_interactions = tabi.far_node_interactions;
  solve.tabi_leaf_panel_interactions = tabi.leaf_panel_interactions;
  solve.tabi_leaf_tile_interactions = tabi.leaf_tile_interactions;
  solve.tabi_leaf_tile_panel_interactions = tabi.leaf_tile_panel_interactions;
  solve.tabi_leaf_residual_panel_interactions =
      tabi.leaf_residual_panel_interactions;
  solve.tabi_smooth_leaf_node_interactions = tabi.smooth_leaf_node_interactions;
  solve.tabi_leaf_pruned_interactions = tabi.leaf_pruned_interactions;
  solve.tabi_leaf_mac_candidates = tabi.leaf_mac_candidates;
  solve.tabi_leaf_mac_accepted = tabi.leaf_mac_accepted;
  solve.tabi_leaf_mac_rejected = tabi.leaf_mac_rejected;
  solve.tabi_leaf_mac_far_panels = tabi.leaf_mac_far_panels;
  solve.tabi_leaf_mac_direct_panels_saved_estimate =
      tabi.leaf_mac_direct_panels_saved_estimate;
  solve.tabi_leaf_work_profile = tabi.leaf_work_profile;
  solve.tabi_leaf_work_mac_terminal_panels = tabi.leaf_work_mac_terminal_panels;
  solve.tabi_leaf_work_physical_gap_panels = tabi.leaf_work_physical_gap_panels;
  solve.tabi_leaf_work_fragmentation_panels =
      tabi.leaf_work_fragmentation_panels;
  solve.tabi_leaf_work_convergence_panels = tabi.leaf_work_convergence_panels;
  solve.tabi_leaf_work_convergence_estimator_panels =
      tabi.leaf_work_convergence_estimator_panels;
  solve.tabi_leaf_work_convergence_rho_panels =
      tabi.leaf_work_convergence_rho_panels;
  solve.tabi_leaf_work_convergence_aspect_panels =
      tabi.leaf_work_convergence_aspect_panels;
  solve.tabi_leaf_work_surface_margin_panels =
      tabi.leaf_work_surface_margin_panels;
  solve.tabi_leaf_work_surface_margin_hard_panels =
      tabi.leaf_work_surface_margin_hard_panels;
  solve.tabi_leaf_work_surface_margin_soft_panels =
      tabi.leaf_work_surface_margin_soft_panels;
  solve.tabi_leaf_work_k3_rho_cap_panels = tabi.leaf_work_k3_rho_cap_panels;
  solve.tabi_leaf_work_singular_fallback_panels =
      tabi.leaf_work_singular_fallback_panels;
  solve.tabi_leaf_work_scalar_tail_panels = tabi.leaf_work_scalar_tail_panels;
  solve.tabi_leaf_work_profile_accounted_panels =
      tabi.leaf_work_profile_accounted_panels;
  solve.tabi_leaf_work_mac_terminal_frac = tabi.leaf_work_mac_terminal_frac;
  solve.tabi_leaf_work_convergence_frac = tabi.leaf_work_convergence_frac;
  solve.tabi_leaf_work_surface_margin_frac = tabi.leaf_work_surface_margin_frac;
  solve.tabi_leaf_work_singular_fallback_frac =
      tabi.leaf_work_singular_fallback_frac;
  solve.tabi_leaf_work_scalar_tail_frac = tabi.leaf_work_scalar_tail_frac;
  solve.tabi_adaptive_smooth_leaf_accepted = tabi.adaptive_smooth_leaf_accepted;
  solve.tabi_adaptive_smooth_leaf_rejected = tabi.adaptive_smooth_leaf_rejected;
  solve.tabi_adaptive_span_tile_accepted = tabi.adaptive_span_tile_accepted;
  solve.tabi_adaptive_span_tile_rejected = tabi.adaptive_span_tile_rejected;
  solve.tabi_adaptive_exact_leaf_fallbacks = tabi.adaptive_exact_leaf_fallbacks;
  solve.tabi_adaptive_reject_one_ring_overlap =
      tabi.adaptive_reject_one_ring_overlap;
  solve.tabi_adaptive_reject_error_bound_failed =
      tabi.adaptive_reject_error_bound_failed;
  solve.tabi_adaptive_reject_p4_check_failed =
      tabi.adaptive_reject_p4_check_failed;
  solve.tabi_adaptive_reject_too_close = tabi.adaptive_reject_too_close;
  solve.tabi_leaf_work_reduction_ratio = tabi.leaf_work_reduction_ratio;
  solve.tabi_target_cluster_interactions = tabi.target_cluster_interactions;
  solve.tabi_target_cluster_targets = tabi.target_cluster_targets;
  solve.tabi_target_cluster_panels_pruned = tabi.target_cluster_panels_pruned;
  solve.tabi_target_cluster_rejected = tabi.target_cluster_rejected;
  solve.tabi_target_cluster_eval_ms = tabi.target_cluster_eval_ms;
  solve.tabi_leaf_pair_batch_interactions = tabi.leaf_pair_batch_interactions;
  solve.tabi_leaf_pair_batch_targets = tabi.leaf_pair_batch_targets;
  solve.tabi_leaf_pair_batch_source_panels = tabi.leaf_pair_batch_source_panels;
  solve.tabi_leaf_pair_batch_residual_panels =
      tabi.leaf_pair_batch_residual_panels;
  solve.tabi_leaf_pair_batch_fallbacks = tabi.leaf_pair_batch_fallbacks;
  solve.tabi_leaf_pair_batch_ms = tabi.leaf_pair_batch_ms;
  solve.tabi_leaf_pair_batch_kernel = tabi.leaf_pair_batch_kernel;
  solve.tabi_leaf_pair_batch_vectorized_sources =
      tabi.leaf_pair_batch_vectorized_sources;
  solve.tabi_leaf_pair_batch_vector_lanes = tabi.leaf_pair_batch_vector_lanes;
  solve.tabi_leaf_pair_batch_pairs_per_second =
      tabi.leaf_pair_batch_pairs_per_second;
  solve.tabi_leaf_kernel_profile = tabi.leaf_kernel_profile;
  solve.tabi_leaf_work_profile = tabi.leaf_work_profile;
  solve.tabi_jcp_direct_kernel = tabi.jcp_direct_kernel;
  solve.tabi_leaf_kernel_scalar_tasks = tabi.leaf_kernel_scalar_tasks;
  solve.tabi_leaf_kernel_batch_tasks = tabi.leaf_kernel_batch_tasks;
  solve.tabi_leaf_kernel_source_panel_evals =
      tabi.leaf_kernel_source_panel_evals;
  solve.tabi_leaf_kernel_scalar_ms = tabi.leaf_kernel_scalar_ms;
  solve.tabi_leaf_kernel_target_avx2_ms = tabi.leaf_kernel_target_avx2_ms;
  solve.tabi_leaf_kernel_tail_ms = tabi.leaf_kernel_tail_ms;
  solve.tabi_leaf_pair_batch_soa_sources = tabi.leaf_pair_batch_soa_sources;
  solve.tabi_leaf_pair_batch_soa_ms = tabi.leaf_pair_batch_soa_ms;
  solve.tabi_leaf_pair_batch_scalar_tail_ms =
      tabi.leaf_pair_batch_scalar_tail_ms;
  solve.tabi_leaf_pair_batch_clean_source_panels_per_second =
      tabi.leaf_pair_batch_clean_source_panels_per_second;
  solve.tabi_leaf_pair_selection_profile = tabi.leaf_pair_selection_profile;
  solve.tabi_leaf_pair_probe_cache = tabi.leaf_pair_probe_cache;
  solve.tabi_leaf_pair_target_group = tabi.leaf_pair_target_group;
  solve.tabi_leaf_pair_directness_probe = tabi.leaf_pair_directness_probe;
  solve.tabi_leaf_pair_directness_cache = tabi.leaf_pair_directness_cache;
  solve.tabi_leaf_pair_directness_verify = tabi.leaf_pair_directness_verify;
  solve.tabi_leaf_pair_max_lookahead = tabi.leaf_pair_max_lookahead;
  solve.tabi_leaf_pair_selection_candidate_scans =
      tabi.leaf_pair_selection_candidate_scans;
  solve.tabi_leaf_pair_selection_clean_rejects =
      tabi.leaf_pair_selection_clean_rejects;
  solve.tabi_leaf_pair_selection_probe_calls =
      tabi.leaf_pair_selection_probe_calls;
  solve.tabi_leaf_pair_selection_probe_node_visits =
      tabi.leaf_pair_selection_probe_node_visits;
  solve.tabi_leaf_pair_selection_probe_ms = tabi.leaf_pair_selection_probe_ms;
  solve.tabi_leaf_pair_selection_ms = tabi.leaf_pair_selection_ms;
  solve.tabi_leaf_pair_selection_probe_cache_hits =
      tabi.leaf_pair_selection_probe_cache_hits;
  solve.tabi_leaf_pair_selection_probe_cache_misses =
      tabi.leaf_pair_selection_probe_cache_misses;
  solve.tabi_leaf_pair_selection_insufficient_targets =
      tabi.leaf_pair_selection_insufficient_targets;
  solve.tabi_leaf_pair_selection_same_target_leaf_rejects =
      tabi.leaf_pair_selection_same_target_leaf_rejects;
  solve.tabi_leaf_pair_directness_ancestor_probe_calls =
      tabi.leaf_pair_directness_ancestor_probe_calls;
  solve.tabi_leaf_pair_directness_ancestor_probe_node_visits =
      tabi.leaf_pair_directness_ancestor_probe_node_visits;
  solve.tabi_leaf_pair_directness_ancestor_probe_ms =
      tabi.leaf_pair_directness_ancestor_probe_ms;
  solve.tabi_leaf_pair_directness_pair_cache_hits =
      tabi.leaf_pair_directness_pair_cache_hits;
  solve.tabi_leaf_pair_directness_pair_cache_misses =
      tabi.leaf_pair_directness_pair_cache_misses;
  solve.tabi_leaf_pair_directness_all_direct_proofs =
      tabi.leaf_pair_directness_all_direct_proofs;
  solve.tabi_leaf_pair_directness_mixed_fallbacks =
      tabi.leaf_pair_directness_mixed_fallbacks;
  solve.tabi_leaf_pair_directness_verify_checks =
      tabi.leaf_pair_directness_verify_checks;
  solve.tabi_leaf_pair_directness_verify_mismatches =
      tabi.leaf_pair_directness_verify_mismatches;
  solve.tabi_leaf_pair_scheduler = tabi.leaf_pair_scheduler;
  solve.tabi_leaf_pair_worklist_max_active = tabi.leaf_pair_worklist_max_active;
  solve.tabi_leaf_pair_worklist_verify = tabi.leaf_pair_worklist_verify;
  solve.tabi_leaf_pair_worklist_tasks = tabi.leaf_pair_worklist_tasks;
  solve.tabi_leaf_pair_worklist_groups = tabi.leaf_pair_worklist_groups;
  solve.tabi_leaf_pair_worklist_batched_groups =
      tabi.leaf_pair_worklist_batched_groups;
  solve.tabi_leaf_pair_worklist_scalar_flush_tasks =
      tabi.leaf_pair_worklist_scalar_flush_tasks;
  solve.tabi_leaf_pair_worklist_forced_flushes =
      tabi.leaf_pair_worklist_forced_flushes;
  solve.tabi_leaf_pair_worklist_active_groups_max =
      tabi.leaf_pair_worklist_active_groups_max;
  solve.tabi_leaf_pair_worklist_ms = tabi.leaf_pair_worklist_ms;
  solve.tabi_leaf_pair_worklist_verify_checks =
      tabi.leaf_pair_worklist_verify_checks;
  solve.tabi_leaf_pair_worklist_verify_mismatches =
      tabi.leaf_pair_worklist_verify_mismatches;
  solve.tabi_leaf_block_verify = tabi.leaf_block_verify;
  solve.tabi_leaf_block_target_leaves = tabi.leaf_block_target_leaves;
  solve.tabi_leaf_block_targets = tabi.leaf_block_targets;
  solve.tabi_leaf_block_source_groups = tabi.leaf_block_source_groups;
  solve.tabi_leaf_block_batched_groups = tabi.leaf_block_batched_groups;
  solve.tabi_leaf_block_scalar_tail_tasks = tabi.leaf_block_scalar_tail_tasks;
  solve.tabi_leaf_block_group_ms = tabi.leaf_block_group_ms;
  solve.tabi_leaf_block_flush_ms = tabi.leaf_block_flush_ms;
  solve.tabi_leaf_block_writeback_ms = tabi.leaf_block_writeback_ms;
  solve.tabi_leaf_block_verify_checks = tabi.leaf_block_verify_checks;
  solve.tabi_leaf_block_verify_mismatches = tabi.leaf_block_verify_mismatches;
  solve.tabi_tree_preset = tabi.tree_preset;
  solve.tabi_adaptive_level = tabi.adaptive_level;
  solve.tabi_cluster_policy = tabi.cluster_policy;
  solve.tabi_target_batch_size = tabi.target_batch_size;
  solve.tabi_leaf_pair_min_source_panels = tabi.leaf_pair_min_source_panels;
  solve.tabi_leaf_pair_min_targets = tabi.leaf_pair_min_targets;
  solve.tabi_leaf_near_policy = tabi.leaf_near_policy;
  solve.tabi_leaf_panel_kernel = tabi.leaf_panel_kernel;
  solve.tabi_interaction_mode = tabi.interaction_mode;
  solve.tabi_node_bounds_mode = tabi.node_bounds_mode;
  solve.tabi_tree_builder = tabi.tree_builder;
  solve.tabi_expansion_center = tabi.expansion_center;
  solve.tabi_fragmentation_safety = tabi.fragmentation_safety;
  solve.tabi_mac_safety = tabi.mac_safety;
  solve.tabi_mac_gap_factor = tabi.mac_gap_factor;
  solve.tabi_tree_depth_max = tabi.tree_depth_max;
  solve.tabi_mac_reject_radius_limited = tabi.mac_reject_radius_limited;
  solve.tabi_mac_accept_rate = tabi.mac_accept_rate;
  solve.tabi_node_radius_max = tabi.node_radius_max;
  solve.tabi_node_radius_mean = tabi.node_radius_mean;
  solve.tabi_node_aspect_ratio_max = tabi.node_aspect_ratio_max;
  solve.tabi_node_aspect_ratio_mean = tabi.node_aspect_ratio_mean;
  solve.tabi_leaf_panel_work_per_matvec = tabi.leaf_panel_work_per_matvec;
  solve.tabi_far_node_work_per_matvec = tabi.far_node_work_per_matvec;
  solve.tabi_far_geometric_unsafe_count = tabi.far_geometric_unsafe_count;
  solve.tabi_far_geometric_unsafe_panels = tabi.far_geometric_unsafe_panels;
  solve.tabi_far_geometric_min_gap = tabi.far_geometric_min_gap;
  solve.tabi_far_geometric_gap_ratio_min = tabi.far_geometric_gap_ratio_min;
  solve.tabi_far_geometric_gap_ratio_mean = tabi.far_geometric_gap_ratio_mean;
  solve.tabi_far_geometric_audit_samples = tabi.far_geometric_audit_samples;
  solve.tabi_mac_physical_gap_rejects = tabi.mac_physical_gap_rejects;
  solve.tabi_mac_convergence_rho_rejects = tabi.mac_convergence_rho_rejects;
  solve.tabi_mac_convergence_aspect_rejects =
      tabi.mac_convergence_aspect_rejects;
  solve.tabi_mac_convergence_rho_reject_panels =
      tabi.mac_convergence_rho_reject_panels;
  solve.tabi_mac_convergence_aspect_reject_panels =
      tabi.mac_convergence_aspect_reject_panels;
  solve.tabi_mac_surface_margin_rejects = tabi.mac_surface_margin_rejects;
  solve.tabi_mac_surface_margin_reject_panels =
      tabi.mac_surface_margin_reject_panels;
  solve.tabi_mac_surface_margin_applied = tabi.mac_surface_margin_applied;
  solve.tabi_mac_surface_margin_skipped = tabi.mac_surface_margin_skipped;
  solve.tabi_mac_surface_margin_adaptive_rejects =
      tabi.mac_surface_margin_adaptive_rejects;
  solve.tabi_mac_surface_margin_adaptive_reject_panels =
      tabi.mac_surface_margin_adaptive_reject_panels;
  solve.tabi_mac_surface_margin_skipped_accepted_far =
      tabi.mac_surface_margin_skipped_accepted_far;
  solve.tabi_mac_surface_margin_hard_applied =
      tabi.mac_surface_margin_hard_applied;
  solve.tabi_mac_surface_margin_soft_applied =
      tabi.mac_surface_margin_soft_applied;
  solve.tabi_mac_surface_margin_hard_rejects =
      tabi.mac_surface_margin_hard_rejects;
  solve.tabi_mac_surface_margin_hard_reject_panels =
      tabi.mac_surface_margin_hard_reject_panels;
  solve.tabi_mac_surface_margin_soft_rejects =
      tabi.mac_surface_margin_soft_rejects;
  solve.tabi_mac_surface_margin_soft_reject_panels =
      tabi.mac_surface_margin_soft_reject_panels;
  solve.tabi_fragmented_far_count = tabi.fragmented_far_count;
  solve.tabi_fragmented_far_panels = tabi.fragmented_far_panels;
  solve.tabi_fragmented_split_rejects = tabi.fragmented_split_rejects;
  solve.tabi_fragmentation_center_gap_ratio_max =
      tabi.fragmentation_center_gap_ratio_max;
  solve.tabi_fragmentation_center_gap_ratio_mean =
      tabi.fragmentation_center_gap_ratio_mean;
  solve.tabi_moment_audit = tabi.moment_audit;
  solve.tabi_moment_construction = tabi.moment_construction;
  solve.tabi_moment_sum = tabi.moment_sum;
  solve.tabi_convergence_audit = tabi.convergence_audit;
  solve.tabi_convergence_safety = tabi.convergence_safety;
  solve.tabi_convergence_rho_cap = tabi.convergence_rho_cap;
  solve.tabi_mac_aspect_cap = tabi.mac_aspect_cap;
  solve.tabi_convergence_estimator = tabi.convergence_estimator;
  solve.tabi_convergence_tail_threshold = tabi.convergence_tail_threshold;
  solve.tabi_convergence_estimator_error_cap =
      tabi.convergence_estimator_error_cap;
  solve.tabi_convergence_unstable_rho_cap = tabi.convergence_unstable_rho_cap;
  solve.tabi_convergence_calibration = tabi.convergence_calibration;
  solve.tabi_convergence_dielectric_amp = tabi.convergence_dielectric_amp;
  solve.tabi_convergence_estimator_model = tabi.convergence_estimator_model;
  solve.tabi_convergence_channel_calibration =
      tabi.convergence_channel_calibration;
  solve.tabi_convergence_split_policy = tabi.convergence_split_policy;
  solve.tabi_convergence_split_selectivity = tabi.convergence_split_selectivity;
  solve.tabi_convergence_split_absolute_floor =
      tabi.convergence_split_absolute_floor;
  solve.tabi_convergence_overreject_audit = tabi.convergence_overreject_audit;
  solve.tabi_convergence_overreject_detail = tabi.convergence_overreject_detail;
  solve.tabi_convergence_shape_audit = tabi.convergence_shape_audit;
  solve.tabi_convergence_kernel_response_audit =
      tabi.convergence_kernel_response_audit;
  solve.tabi_channel_missed_semantics = tabi.channel_missed_semantics;
  solve.tabi_accepted_miss_diagnostics = tabi.accepted_miss_diagnostics;
  solve.tabi_accepted_miss_absolute_floor = tabi.accepted_miss_absolute_floor;
  solve.tabi_convergence_k3_accepted_rho_cap =
      tabi.convergence_k3_accepted_rho_cap;
  solve.tabi_mac_distance_policy = tabi.mac_distance_policy;
  solve.tabi_mac_surface_margin_factor = tabi.mac_surface_margin_factor;
  solve.tabi_mac_surface_margin_policy = tabi.mac_surface_margin_policy;
  solve.tabi_mac_surface_margin_aspect_threshold =
      tabi.mac_surface_margin_aspect_threshold;
  solve.tabi_mac_surface_margin_rho_trigger =
      tabi.mac_surface_margin_rho_trigger;
  solve.tabi_mac_surface_margin_soft_aspect_threshold =
      tabi.mac_surface_margin_soft_aspect_threshold;
  solve.tabi_mac_surface_margin_soft_rho_trigger =
      tabi.mac_surface_margin_soft_rho_trigger;
  solve.tabi_mac_surface_margin_soft_factor =
      tabi.mac_surface_margin_soft_factor;
  solve.tabi_convergence_estimator_samples = tabi.convergence_estimator_samples;
  solve.tabi_convergence_estimator_unstable_far_count =
      tabi.convergence_estimator_unstable_far_count;
  solve.tabi_convergence_estimator_unstable_far_panels =
      tabi.convergence_estimator_unstable_far_panels;
  solve.tabi_convergence_estimator_split_rejects =
      tabi.convergence_estimator_split_rejects;
  solve.tabi_convergence_estimator_split_reject_panels =
      tabi.convergence_estimator_split_reject_panels;
  solve.tabi_convergence_low_abs_reaccepted_far_count =
      tabi.convergence_low_abs_reaccepted_far_count;
  solve.tabi_convergence_low_abs_reaccepted_far_panels =
      tabi.convergence_low_abs_reaccepted_far_panels;
  solve.tabi_convergence_estimator_tail_ratio_max =
      tabi.convergence_estimator_tail_ratio_max;
  solve.tabi_convergence_estimator_tail_ratio_mean =
      tabi.convergence_estimator_tail_ratio_mean;
  solve.tabi_convergence_estimator_error_max =
      tabi.convergence_estimator_error_max;
  solve.tabi_convergence_estimator_error_mean =
      tabi.convergence_estimator_error_mean;
  solve.tabi_convergence_estimator_abs_error_max =
      tabi.convergence_estimator_abs_error_max;
  solve.tabi_convergence_estimator_abs_error_mean =
      tabi.convergence_estimator_abs_error_mean;
  solve.tabi_convergence_tail_norm_p_max = tabi.convergence_tail_norm_p_max;
  solve.tabi_convergence_tail_norm_p_mean = tabi.convergence_tail_norm_p_mean;
  solve.tabi_convergence_tail_norm_pm1_max = tabi.convergence_tail_norm_pm1_max;
  solve.tabi_convergence_tail_norm_pm1_mean =
      tabi.convergence_tail_norm_pm1_mean;
  solve.tabi_convergence_effective_jcp_amp_max =
      tabi.convergence_effective_jcp_amp_max;
  solve.tabi_convergence_effective_jcp_amp_mean =
      tabi.convergence_effective_jcp_amp_mean;
  solve.tabi_convergence_dielectric_amp_max =
      tabi.convergence_dielectric_amp_max;
  solve.tabi_convergence_dielectric_amp_mean =
      tabi.convergence_dielectric_amp_mean;
  solve.tabi_convergence_channel_estimated_error_max =
      tabi.convergence_channel_estimated_error_max;
  solve.tabi_convergence_channel_estimated_error_mean =
      tabi.convergence_channel_estimated_error_mean;
  solve.tabi_convergence_channel_effective_jcp_amp_max =
      tabi.convergence_channel_effective_jcp_amp_max;
  solve.tabi_convergence_channel_effective_jcp_amp_mean =
      tabi.convergence_channel_effective_jcp_amp_mean;
  solve.tabi_convergence_channel_unstable_far_count =
      tabi.convergence_channel_unstable_far_count;
  solve.tabi_convergence_channel_unstable_far_panels =
      tabi.convergence_channel_unstable_far_panels;
  solve.tabi_convergence_channel_split_rejects =
      tabi.convergence_channel_split_rejects;
  solve.tabi_convergence_channel_split_reject_panels =
      tabi.convergence_channel_split_reject_panels;
  solve.tabi_convergence_channel_low_abs_reaccepted_count =
      tabi.convergence_channel_low_abs_reaccepted_count;
  solve.tabi_convergence_channel_low_abs_reaccepted_panels =
      tabi.convergence_channel_low_abs_reaccepted_panels;
  solve.tabi_convergence_channel_high_error_caught =
      tabi.convergence_channel_high_error_caught;
  solve.tabi_convergence_channel_high_error_missed =
      tabi.convergence_channel_high_error_missed;
  solve.tabi_convergence_channel_high_error_caught_by_split =
      tabi.convergence_channel_high_error_caught_by_split;
  solve.tabi_convergence_estimator_high_error_caught =
      tabi.convergence_estimator_high_error_caught;
  solve.tabi_convergence_estimator_high_error_missed =
      tabi.convergence_estimator_high_error_missed;
  solve.tabi_convergence_estimator_high_error_caught_by_split =
      tabi.convergence_estimator_high_error_caught_by_split;
  solve.tabi_convergence_estimator_phantom_high_error_count =
      tabi.convergence_estimator_phantom_high_error_count;
  solve.tabi_convergence_estimator_phantom_high_error_abs_max =
      tabi.convergence_estimator_phantom_high_error_abs_max;
  solve.tabi_convergence_overreject_count = tabi.convergence_overreject_count;
  solve.tabi_convergence_overreject_panels = tabi.convergence_overreject_panels;
  solve.tabi_convergence_overreject_caught_count =
      tabi.convergence_overreject_caught_count;
  solve.tabi_convergence_overreject_caught_panels =
      tabi.convergence_overreject_caught_panels;
  solve.tabi_convergence_overreject_abs_error_max =
      tabi.convergence_overreject_abs_error_max;
  solve.tabi_convergence_overreject_abs_error_mean =
      tabi.convergence_overreject_abs_error_mean;
  solve.tabi_convergence_overreject_abs_error_p95 =
      tabi.convergence_overreject_abs_error_p95;
  solve.tabi_convergence_overreject_abs_error_p99 =
      tabi.convergence_overreject_abs_error_p99;
  solve.tabi_convergence_overreject_estimated_abs_error_max =
      tabi.convergence_overreject_estimated_abs_error_max;
  solve.tabi_convergence_overreject_estimated_abs_error_mean =
      tabi.convergence_overreject_estimated_abs_error_mean;
  solve.tabi_convergence_overreject_required_scale_max =
      tabi.convergence_overreject_required_scale_max;
  solve.tabi_convergence_overreject_required_scale_p95 =
      tabi.convergence_overreject_required_scale_p95;
  solve.tabi_convergence_overreject_required_scale_p99 =
      tabi.convergence_overreject_required_scale_p99;
  solve.tabi_convergence_channel_overreject_count =
      tabi.convergence_channel_overreject_count;
  solve.tabi_convergence_channel_overreject_caught_count =
      tabi.convergence_channel_overreject_caught_count;
  solve.tabi_convergence_channel_overreject_panels =
      tabi.convergence_channel_overreject_panels;
  solve.tabi_convergence_channel_overreject_abs_error_max =
      tabi.convergence_channel_overreject_abs_error_max;
  solve.tabi_convergence_channel_overreject_abs_error_mean =
      tabi.convergence_channel_overreject_abs_error_mean;
  solve.tabi_convergence_channel_overreject_required_scale_max =
      tabi.convergence_channel_overreject_required_scale_max;
  solve.tabi_convergence_channel_overreject_required_scale_p99 =
      tabi.convergence_channel_overreject_required_scale_p99;
  solve.tabi_phantom_high_error_channel_count =
      tabi.phantom_high_error_channel_count;
  solve.tabi_phantom_high_error_channel_abs_max =
      tabi.phantom_high_error_channel_abs_max;
  solve.tabi_accepted_miss_channel_count = tabi.accepted_miss_channel_count;
  solve.tabi_accepted_miss_channel_panels = tabi.accepted_miss_channel_panels;
  solve.tabi_accepted_miss_channel_rho_min = tabi.accepted_miss_channel_rho_min;
  solve.tabi_accepted_miss_channel_rho_mean =
      tabi.accepted_miss_channel_rho_mean;
  solve.tabi_accepted_miss_channel_rho_max = tabi.accepted_miss_channel_rho_max;
  solve.tabi_accepted_miss_channel_tail_ratio_mean =
      tabi.accepted_miss_channel_tail_ratio_mean;
  solve.tabi_accepted_miss_channel_tail_ratio_max =
      tabi.accepted_miss_channel_tail_ratio_max;
  solve.tabi_accepted_miss_channel_estimated_error_mean =
      tabi.accepted_miss_channel_estimated_error_mean;
  solve.tabi_accepted_miss_channel_estimated_error_max =
      tabi.accepted_miss_channel_estimated_error_max;
  solve.tabi_accepted_miss_worst_channel = tabi.accepted_miss_worst_channel;
  solve.tabi_convergence_k3_rho_cap_rejects =
      tabi.convergence_k3_rho_cap_rejects;
  solve.tabi_convergence_k3_rho_cap_reject_panels =
      tabi.convergence_k3_rho_cap_reject_panels;
  solve.tabi_mac_surface_margin_rejects = tabi.mac_surface_margin_rejects;
  solve.tabi_mac_surface_margin_reject_panels =
      tabi.mac_surface_margin_reject_panels;
  solve.tabi_moment_audit_nodes = tabi.moment_audit_nodes;
  solve.tabi_moment_translation_rel_error_max =
      tabi.moment_translation_rel_error_max;
  solve.tabi_moment_translation_rel_error_mean =
      tabi.moment_translation_rel_error_mean;
  solve.tabi_moment_translation_offset_max = tabi.moment_translation_offset_max;
  solve.tabi_moment_translation_offset_mean =
      tabi.moment_translation_offset_mean;
  solve.tabi_moment_cancellation_ratio_max = tabi.moment_cancellation_ratio_max;
  solve.tabi_moment_cancellation_ratio_mean =
      tabi.moment_cancellation_ratio_mean;
  solve.tabi_moment_dynamic_range_max = tabi.moment_dynamic_range_max;
  solve.tabi_expansion_audit = tabi.expansion_audit;
  solve.tabi_expansion_audit_channel = tabi.expansion_audit_channel;
  solve.tabi_far_expansion_audit_pairs = tabi.far_expansion_audit_pairs;
  solve.tabi_far_expansion_audit_panels = tabi.far_expansion_audit_panels;
  solve.tabi_far_expansion_direct_ms = tabi.far_expansion_direct_ms;
  solve.tabi_far_expansion_rel_error_max = tabi.far_expansion_rel_error_max;
  solve.tabi_far_expansion_rel_error_mean = tabi.far_expansion_rel_error_mean;
  solve.tabi_far_expansion_rel_error_p95 = tabi.far_expansion_rel_error_p95;
  solve.tabi_far_expansion_rel_error_p99 = tabi.far_expansion_rel_error_p99;
  solve.tabi_far_expansion_channel_rel_error_max =
      tabi.far_expansion_channel_rel_error_max;
  solve.tabi_far_expansion_channel_rel_error_mean =
      tabi.far_expansion_channel_rel_error_mean;
  solve.tabi_far_expansion_channel_rel_error_p99 =
      tabi.far_expansion_channel_rel_error_p99;
  solve.tabi_far_expansion_worst_channel = tabi.far_expansion_worst_channel;
  solve.tabi_far_expansion_mean_gate_pass = tabi.far_expansion_mean_gate_pass;
  solve.tabi_far_convergence_rho_max = tabi.far_convergence_rho_max;
  solve.tabi_far_convergence_rho_mean = tabi.far_convergence_rho_mean;
  solve.tabi_accepted_far_aspect_ratio_max = tabi.accepted_far_aspect_ratio_max;
  solve.tabi_accepted_far_aspect_ratio_mean =
      tabi.accepted_far_aspect_ratio_mean;
  solve.tabi_quad_buffer_auto_candidate = tabi.quad_buffer_auto_candidate;
  solve.tabi_leaf_tile_status = tabi.leaf_tile_status;
  solve.tabi_leaf_panel_pairs_per_second = tabi.leaf_panel_pairs_per_second;
  solve.tabi_near_integration = tabi.near_integration;
  solve.tabi_quad_buffer = tabi.quad_buffer;
  solve.tabi_near_self_pairs = tabi.near_self_pairs;
  solve.tabi_near_one_ring_pairs = tabi.near_one_ring_pairs;
  solve.tabi_near_duffy_pairs = tabi.near_duffy_pairs;
  solve.tabi_near_centroid_pairs = tabi.near_centroid_pairs;
  solve.tabi_near_fallbacks = tabi.near_fallbacks;
  solve.tabi_near_geometric_buffer_pairs = tabi.near_geometric_buffer_pairs;
  solve.tabi_near_geometric_buffer_gauss_pairs =
      tabi.near_geometric_buffer_gauss_pairs;
  solve.tabi_near_geometric_buffer_adaptive_pairs =
      tabi.near_geometric_buffer_adaptive_pairs;
  solve.tabi_near_geometric_buffer_centroid_pairs =
      tabi.near_geometric_buffer_centroid_pairs;
  solve.tabi_near_geometric_buffer_min_gap = tabi.near_geometric_buffer_min_gap;
  solve.tabi_near_geometric_buffer_mean_gap_ratio =
      tabi.near_geometric_buffer_mean_gap_ratio;
  solve.tabi_near_geometric_buffer_max_edge_mean =
      tabi.near_geometric_buffer_max_edge_mean;
  solve.tabi_near_integration_ms = tabi.near_integration_ms;
  solve.tabi_duffy_subdivisions_max = tabi.duffy_subdivisions_max;
  solve.tabi_duffy_subdivisions_mean = tabi.duffy_subdivisions_mean;
  solve.tabi_near_kernel_policy = tabi.near_kernel_policy;
  solve.tabi_near_k1_accepted = tabi.near_k1_accepted;
  solve.tabi_near_k1_rejected = tabi.near_k1_rejected;
  solve.tabi_near_k2_accepted = tabi.near_k2_accepted;
  solve.tabi_near_k2_rejected = tabi.near_k2_rejected;
  solve.tabi_near_k3_accepted = tabi.near_k3_accepted;
  solve.tabi_near_k3_rejected = tabi.near_k3_rejected;
  solve.tabi_near_k4_accepted = tabi.near_k4_accepted;
  solve.tabi_near_k4_rejected = tabi.near_k4_rejected;
  solve.tabi_near_one_ring_k1_accepted = tabi.near_one_ring_k1_accepted;
  solve.tabi_near_one_ring_k1_rejected = tabi.near_one_ring_k1_rejected;
  solve.tabi_near_one_ring_k2_accepted = tabi.near_one_ring_k2_accepted;
  solve.tabi_near_one_ring_k2_rejected = tabi.near_one_ring_k2_rejected;
  solve.tabi_near_one_ring_k3_accepted = tabi.near_one_ring_k3_accepted;
  solve.tabi_near_one_ring_k3_rejected = tabi.near_one_ring_k3_rejected;
  solve.tabi_near_one_ring_k4_accepted = tabi.near_one_ring_k4_accepted;
  solve.tabi_near_one_ring_k4_rejected = tabi.near_one_ring_k4_rejected;
  solve.tabi_near_k1_policy = tabi.near_k1_policy;
  solve.tabi_near_k2_policy = tabi.near_k2_policy;
  solve.tabi_near_k3_policy = tabi.near_k3_policy;
  solve.tabi_near_k4_policy = tabi.near_k4_policy;
  for (std::size_t i = 0; i < solve.tabi_near_block_audit.size(); ++i) {
    const auto &src = tabi.near_block_audit[i];
    auto &dst = solve.tabi_near_block_audit[i];
    dst.accepted = src.accepted;
    dst.rejected = src.rejected;
    dst.candidate_worse_than_centroid = src.candidate_worse_than_centroid;
    dst.reference_nonconverged = src.reference_nonconverged;
    dst.orientation_sign_mismatch = src.orientation_sign_mismatch;
    dst.strict_ref_reject = src.strict_ref_reject;
    dst.max_candidate_ref_rel = src.max_candidate_ref_rel;
    dst.sum_candidate_ref_rel = src.sum_candidate_ref_rel;
    dst.max_centroid_ref_rel = src.max_centroid_ref_rel;
    dst.k2_laplace_sign_ok = src.k2_laplace_sign_ok;
    dst.max_k2_yukawa_smooth_ref_rel = src.max_k2_yukawa_smooth_ref_rel;
    dst.max_k2_orientation_pair_ref_rel = src.max_k2_orientation_pair_ref_rel;
    dst.sample_count = src.sample_count;
  }
  for (std::size_t i = 0; i < solve.tabi_geometric_near_audit.size(); ++i) {
    const auto &src = tabi.geometric_near_audit[i];
    auto &dst = solve.tabi_geometric_near_audit[i];
    dst.pairs = src.pairs;
    dst.min_gap = src.min_gap;
    dst.max_gap_ratio = src.max_gap_ratio;
    dst.sum_gap_ratio = src.sum_gap_ratio;
    dst.sum_center_ratio = src.sum_center_ratio;
  }
  solve.tabi_surface_geometry = tabi.surface_geometry;
  solve.tabi_sphere_mesh_mode = tabi.sphere_mesh_mode;
  solve.tabi_surface_quality_total_area_a2 = tabi.surface_quality_total_area_a2;
  solve.tabi_surface_quality_area_rel_error_vs_4piR2 =
      tabi.surface_quality_area_rel_error_vs_4piR2;
  solve.tabi_surface_quality_triangle_area_min =
      tabi.surface_quality_triangle_area_min;
  solve.tabi_surface_quality_triangle_area_mean =
      tabi.surface_quality_triangle_area_mean;
  solve.tabi_surface_quality_triangle_area_max =
      tabi.surface_quality_triangle_area_max;
  solve.tabi_surface_quality_triangle_aspect_max =
      tabi.surface_quality_triangle_aspect_max;
  solve.tabi_surface_quality_normal_radial_dot_min =
      tabi.surface_quality_normal_radial_dot_min;
  solve.tabi_surface_quality_normal_radial_dot_mean =
      tabi.surface_quality_normal_radial_dot_mean;
  solve.tabi_surface_quality_centroid_radial_error_max =
      tabi.surface_quality_centroid_radial_error_max;
  solve.tabi_surface_quality_planar_area_a2 =
      tabi.surface_quality_planar_area_a2;
  solve.tabi_surface_quality_planar_area_rel_error_vs_4piR2 =
      tabi.surface_quality_planar_area_rel_error_vs_4piR2;
  solve.tabi_surface_quality_enclosed_volume_a3 =
      tabi.surface_quality_enclosed_volume_a3;
  solve.tabi_surface_quality_enclosed_volume_rel_error_vs_4piR3_over3 =
      tabi.surface_quality_enclosed_volume_rel_error_vs_4piR3_over3;
  solve.tabi_surface_quality_triangle_area_cv =
      tabi.surface_quality_triangle_area_cv;
  solve.tabi_surface_topology_source = tabi.surface_topology_source;
  solve.tabi_surface_analytic_ses_engine = tabi.surface_analytic_ses_engine;
  solve.tabi_surface_status_detail = tabi.surface_status_detail;
  solve.tabi_surface_cavity_policy = tabi.surface_cavity_policy;
  solve.tabi_surface_odd_ray_rows = tabi.surface_odd_ray_rows;
  solve.tabi_surface_patch_ray_unresolved_uncertified_rows =
      tabi.surface_patch_ray_unresolved_uncertified_rows;
  solve.tabi_surface_morphology_fallback_vertices =
      tabi.surface_morphology_fallback_vertices;
  solve.tabi_surface_analytic_orientation_status =
      tabi.surface_analytic_orientation_status;
  solve.tabi_surface_analytic_inverted_faces =
      tabi.surface_analytic_inverted_faces;
  solve.tabi_surface_analytic_orientation_uncertified_faces =
      tabi.surface_analytic_orientation_uncertified_faces;
  solve.tabi_surface_analytic_self_intersections =
      tabi.surface_analytic_self_intersections;
  solve.tabi_surface_wide_repair_attempts =
      tabi.surface_wide_repair_attempts;
  solve.tabi_surface_topology_certificate_accepts =
      tabi.surface_topology_certificate_accepts;
  solve.tabi_surface_topology_fallback_audits =
      tabi.surface_topology_fallback_audits;
  solve.tabi_surface_initial_collapse_ms = tabi.surface_initial_collapse_ms;
  solve.tabi_surface_stable_sliver_ms = tabi.surface_stable_sliver_ms;
  solve.tabi_surface_full_projection_ms = tabi.surface_full_projection_ms;
  solve.tabi_surface_orientation_repair_ms =
      tabi.surface_orientation_repair_ms;
  solve.tabi_surface_wide_repair_ms = tabi.surface_wide_repair_ms;
  solve.tabi_surface_self_intersection_audit_ms =
      tabi.surface_self_intersection_audit_ms;
  solve.tabi_surface_filled_cavity_regions =
      tabi.surface_filled_cavity_regions;
  solve.tabi_surface_filled_cavity_vertices =
      tabi.surface_filled_cavity_vertices;
  solve.tabi_surface_poor_triangles_before =
      tabi.surface_poor_triangles_before;
  solve.tabi_surface_poor_triangles_after = tabi.surface_poor_triangles_after;
  solve.tabi_surface_triangle_quality_min_before =
      tabi.surface_triangle_quality_min_before;
  solve.tabi_surface_triangle_quality_min_after =
      tabi.surface_triangle_quality_min_after;
  solve.tabi_surface_max_edge_ratio_before =
      tabi.surface_max_edge_ratio_before;
  solve.tabi_surface_max_edge_ratio_after = tabi.surface_max_edge_ratio_after;
  solve.tabi_surface_analytic_audit_status =
      tabi.surface_analytic_audit_status;
  solve.tabi_surface_analytic_audit_samples =
      tabi.surface_analytic_audit_samples;
  solve.tabi_surface_analytic_audit_hits = tabi.surface_analytic_audit_hits;
  solve.tabi_surface_analytic_audit_misses =
      tabi.surface_analytic_audit_misses;
  solve.tabi_surface_analytic_projection_distance_rms =
      tabi.surface_analytic_projection_distance_rms;
  solve.tabi_surface_analytic_projection_distance_max =
      tabi.surface_analytic_projection_distance_max;
  solve.tabi_surface_analytic_patch_residual_rms =
      tabi.surface_analytic_patch_residual_rms;
  solve.tabi_surface_analytic_patch_residual_max =
      tabi.surface_analytic_patch_residual_max;
  solve.tabi_surface_analytic_normal_angle_rms_deg =
      tabi.surface_analytic_normal_angle_rms_deg;
  solve.tabi_surface_analytic_normal_angle_max_deg =
      tabi.surface_analytic_normal_angle_max_deg;
  solve.tabi_surface_sampled_hausdorff_proxy =
      tabi.surface_sampled_hausdorff_proxy;
  solve.tabi_surface_analytic_normal_quality_status =
      tabi.surface_analytic_normal_quality_status;
  solve.tabi_surface_analytic_orientation_smooth_faces =
      tabi.surface_analytic_orientation_smooth_faces;
  solve.tabi_surface_analytic_orientation_seam_faces =
      tabi.surface_analytic_orientation_seam_faces;
  solve.tabi_surface_analytic_normal_cosine_below_025_faces =
      tabi.surface_analytic_normal_cosine_below_025_faces;
  solve.tabi_surface_analytic_normal_cosine_below_050_faces =
      tabi.surface_analytic_normal_cosine_below_050_faces;
  solve.tabi_surface_analytic_normal_area_weighted_rms_deg =
      tabi.surface_analytic_normal_area_weighted_rms_deg;
  solve.tabi_surface_analytic_normal_p99_deg =
      tabi.surface_analytic_normal_p99_deg;
  solve.tabi_surface_analytic_smooth_normal_p99_deg =
      tabi.surface_analytic_smooth_normal_p99_deg;
  solve.tabi_surface_analytic_seam_normal_p99_deg =
      tabi.surface_analytic_seam_normal_p99_deg;
  solve.tabi_surface_analytic_normal_area_below_cosine_025_fraction =
      tabi.surface_analytic_normal_area_below_cosine_025_fraction;
  solve.tabi_surface_analytic_normal_area_below_cosine_050_fraction =
      tabi.surface_analytic_normal_area_below_cosine_050_fraction;
  solve.tabi_calderon_trace_space_status = tabi.calderon_trace_space_status;
  solve.tabi_calderon_trace_vertices = tabi.calderon_trace_vertices;
  solve.tabi_calderon_trace_edges = tabi.calderon_trace_edges;
  solve.tabi_calderon_trace_boundary_edges = tabi.calderon_trace_boundary_edges;
  solve.tabi_calderon_trace_orientation_mismatches =
      tabi.calderon_trace_orientation_mismatches;
  solve.tabi_calderon_trace_nonmanifold_edges =
      tabi.calderon_trace_nonmanifold_edges;
  solve.tabi_calderon_trace_unreferenced_vertices =
      tabi.calderon_trace_unreferenced_vertices;
  solve.tabi_calderon_trace_components = tabi.calderon_trace_components;
  solve.tabi_calderon_trace_euler_characteristic =
      tabi.calderon_trace_euler_characteristic;
  solve.tabi_calderon_trace_genus_sum = tabi.calderon_trace_genus_sum;
  solve.tabi_calderon_trace_topology_invariants_valid =
      tabi.calderon_trace_topology_invariants_valid;
  solve.tabi_calderon_aux_surface_mode = tabi.calderon_aux_surface_mode;
  solve.tabi_calderon_aux_trace_status = tabi.calderon_aux_trace_status;
  solve.tabi_calderon_aux_panels = tabi.calderon_aux_panels;
  solve.tabi_calderon_aux_vertices = tabi.calderon_aux_vertices;
  solve.tabi_calderon_aux_edges = tabi.calderon_aux_edges;
  solve.tabi_calderon_aux_boundary_edges = tabi.calderon_aux_boundary_edges;
  solve.tabi_calderon_aux_orientation_mismatches =
      tabi.calderon_aux_orientation_mismatches;
  solve.tabi_calderon_aux_nonmanifold_edges =
      tabi.calderon_aux_nonmanifold_edges;
  solve.tabi_calderon_aux_unreferenced_vertices =
      tabi.calderon_aux_unreferenced_vertices;
  solve.tabi_calderon_aux_grid_vertices = tabi.calderon_aux_grid_vertices;
  solve.tabi_calderon_aux_area_a2 = tabi.calderon_aux_area_a2;
  solve.tabi_calderon_aux_build_ms = tabi.calderon_aux_build_ms;
  solve.tabi_mesh_vertices = tabi.mesh_vertices;
  solve.tabi_mesh_triangles = tabi.mesh_triangles;
  solve.tabi_boundary_nodes = tabi.boundary_nodes;
  solve.tabi_avg_valence = tabi.avg_valence;
  solve.tabi_nonmanifold_edges = tabi.nonmanifold_edges;
  solve.tabi_solid_angle_fallbacks = tabi.solid_angle_fallbacks;
  solve.tabi_solid_angle_clamps = tabi.solid_angle_clamps;
  solve.tabi_diag_min = tabi.diag_min;
  solve.tabi_diag_max = tabi.diag_max;
  solve.tabi_taylor_order = tabi.taylor_order;
  solve.tabi_derivative_mode = tabi.derivative_mode;
  solve.tabi_interaction_build_ms = tabi.interaction_build_ms;
  solve.tabi_tree_eval_ms = tabi.tree_eval_ms;
  solve.tabi_formulation = tabi.formulation;
  solve.tabi_energy_projection = tabi.energy_projection;
  solve.tabi_energy_projection_ms = tabi.energy_projection_ms;
  solve.tabi_energy_projection_kernel = tabi.energy_projection_kernel;
  solve.tabi_preconditioner_requested = tabi.preconditioner_requested;
  solve.tabi_preconditioner_effective = tabi.preconditioner_effective;
  solve.tabi_preconditioner_block_near = tabi.preconditioner_block_near;
  solve.tabi_preconditioner_block_panel_cap =
      tabi.preconditioner_block_panel_cap;
  solve.tabi_preconditioner_overlap_panel_cap =
      tabi.preconditioner_overlap_panel_cap;
  solve.tabi_preconditioner_ilu_neighbor_cap_requested =
      tabi.preconditioner_ilu_neighbor_cap_requested;
  solve.tabi_preconditioner_ilu_neighbor_cap =
      tabi.preconditioner_ilu_neighbor_cap;
  solve.tabi_preconditioner_ilu_drop_tol = tabi.preconditioner_ilu_drop_tol;
  solve.tabi_preconditioner_ilu_regularization =
      tabi.preconditioner_ilu_regularization;
  solve.tabi_preconditioner_ilu_regularization_max =
      tabi.preconditioner_ilu_regularization_max;
  solve.tabi_preconditioner_ilu_regularization_selected =
      tabi.preconditioner_ilu_regularization_selected;
  solve.tabi_preconditioner_ilu_regularization_absolute_max =
      tabi.preconditioner_ilu_regularization_absolute_max;
  solve.tabi_preconditioner_ilu_regularization_attempts =
      tabi.preconditioner_ilu_regularization_attempts;
  solve.tabi_preconditioner_ilu_regularization_rejected_attempts =
      tabi.preconditioner_ilu_regularization_rejected_attempts;
  solve.tabi_preconditioner_ilu_regularization_shifted_rows =
      tabi.preconditioner_ilu_regularization_shifted_rows;
  solve.tabi_preconditioner_ilu_unshifted_failure_stage =
      tabi.preconditioner_ilu_unshifted_failure_stage;
  solve.tabi_preconditioner_memory_budget_bytes =
      tabi.preconditioner_memory_budget_bytes;
  solve.tabi_preconditioner_estimated_bytes =
      tabi.preconditioner_estimated_bytes;
  solve.tabi_preconditioner_valid_blocks = tabi.preconditioner_valid_blocks;
  solve.tabi_preconditioner_valid_panels = tabi.preconditioner_valid_panels;
  solve.tabi_preconditioner_ill_conditioned_blocks =
      tabi.preconditioner_ill_conditioned_blocks;
  solve.tabi_preconditioner_quality_rejected_blocks =
      tabi.preconditioner_quality_rejected_blocks;
  solve.tabi_preconditioner_quality_probe_max_gain =
      tabi.preconditioner_quality_probe_max_gain;
  solve.tabi_preconditioner_quality_probe_rel_residual =
      tabi.preconditioner_quality_probe_rel_residual;
  solve.tabi_preconditioner_pivot_ratio_min =
      tabi.preconditioner_pivot_ratio_min;
  solve.tabi_preconditioner_factor_growth_max =
      tabi.preconditioner_factor_growth_max;
  solve.tabi_preconditioner_pivot_ratio_min_row =
      tabi.preconditioner_pivot_ratio_min_row;
  solve.tabi_preconditioner_pivot_ratio_min_column =
      tabi.preconditioner_pivot_ratio_min_column;
  solve.tabi_preconditioner_factor_growth_row =
      tabi.preconditioner_factor_growth_row;
  solve.tabi_preconditioner_factor_growth_column =
      tabi.preconditioner_factor_growth_column;
  solve.tabi_preconditioner_quality_failure_stage =
      tabi.preconditioner_quality_failure_stage;
  solve.tabi_preconditioner_quality_failure_row =
      tabi.preconditioner_quality_failure_row;
  solve.tabi_preconditioner_quality_failure_column =
      tabi.preconditioner_quality_failure_column;
  solve.tabi_preconditioner_quality_failure_probe =
      tabi.preconditioner_quality_failure_probe;
  solve.tabi_preconditioner_quality_failure_probe_gain =
      tabi.preconditioner_quality_failure_probe_gain;
  solve.tabi_preconditioner_quality_failure_probe_rel_residual =
      tabi.preconditioner_quality_failure_probe_rel_residual;
  solve.tabi_preconditioner_quality_status = tabi.preconditioner_quality_status;
  solve.tabi_preconditioner_setup_ms = tabi.preconditioner_setup_ms;
  solve.tabi_preconditioner_graph_setup_ms =
      tabi.preconditioner_graph_setup_ms;
  solve.tabi_preconditioner_graph_cache_reused =
      tabi.preconditioner_graph_cache_reused;
  solve.tabi_preconditioner_near_assembly_setup_ms =
      tabi.preconditioner_near_assembly_setup_ms;
  solve.tabi_preconditioner_ilu_factor_setup_ms =
      tabi.preconditioner_ilu_factor_setup_ms;
  solve.tabi_preconditioner_apply_ms = tabi.preconditioner_apply_ms;
  solve.tabi_preconditioner_apply_calls = tabi.preconditioner_apply_calls;
  solve.tabi_preconditioner_sparse_nnz_blocks =
      tabi.preconditioner_sparse_nnz_blocks;
  solve.tabi_preconditioner_sparse_avg_degree =
      tabi.preconditioner_sparse_avg_degree;
  solve.tabi_preconditioner_sparse_max_degree =
      tabi.preconditioner_sparse_max_degree;
  solve.tabi_preconditioner_coarse_aggregate_cap =
      tabi.preconditioner_coarse_aggregate_cap;
  solve.tabi_preconditioner_coarse_aggregates =
      tabi.preconditioner_coarse_aggregates;
  solve.tabi_preconditioner_coarse_dimension =
      tabi.preconditioner_coarse_dimension;
  solve.tabi_preconditioner_coarse_base = tabi.preconditioner_coarse_base;
  solve.tabi_preconditioner_coarse_matrix_bytes =
      tabi.preconditioner_coarse_matrix_bytes;
  solve.tabi_preconditioner_coarse_pivot_ratio =
      tabi.preconditioner_coarse_pivot_ratio;
  solve.tabi_preconditioner_coarse_status = tabi.preconditioner_coarse_status;
  solve.tabi_preconditioner_coarse_setup_matvec_calls =
      tabi.preconditioner_coarse_setup_matvec_calls;
  solve.tabi_preconditioner_coarse_setup_ms =
      tabi.preconditioner_coarse_setup_ms;
  solve.tabi_preconditioner_coarse_moments_setup_ms =
      tabi.preconditioner_coarse_moments_setup_ms;
  solve.tabi_preconditioner_coarse_far_setup_ms =
      tabi.preconditioner_coarse_far_setup_ms;
  solve.tabi_preconditioner_coarse_audit_setup_ms =
      tabi.preconditioner_coarse_audit_setup_ms;
  solve.tabi_preconditioner_coarse_apply_ms =
      tabi.preconditioner_coarse_apply_ms;
  solve.tabi_preconditioner_coarse_apply_calls =
      tabi.preconditioner_coarse_apply_calls;
  solve.tabi_preconditioner_coarse_rp_identity_error_max =
      tabi.preconditioner_coarse_rp_identity_error_max;
  solve.tabi_preconditioner_coarse_verify_rel_error_max =
      tabi.preconditioner_coarse_verify_rel_error_max;
  solve.tabi_surface_generator = tabi.surface_generator;
  solve.tabi_discretization = tabi.discretization;
  solve.tabi_node_patch_representation = tabi.node_patch_representation;
  solve.tabi_node_patch_normal_source = tabi.node_patch_normal_source;
  solve.tabi_node_patch_certificate = tabi.node_patch_certificate;
  solve.tabi_rhs_scale = tabi.rhs_scale;
  solve.tabi_dielectric_contrast = tabi.dielectric_contrast;
  solve.tabi_operator_contrast = tabi.operator_contrast;
  solve.tabi_unknown_count = tabi.unknown_count;
  solve.tabi_bie_system = tabi.bie_system;
  solve.tabi_lhs_diag_phi = tabi.lhs_diag_phi;
  solve.tabi_lhs_diag_phin = tabi.lhs_diag_phin;
  solve.tabi_energy_equation = tabi.energy_equation;
  solve.tabi_kernel_convention = tabi.kernel_convention;
  solve.tabi_source_norm_s1 = tabi.source_norm_s1;
  solve.tabi_source_norm_s2 = tabi.source_norm_s2;
  solve.tabi_phi_norm = tabi.phi_norm;
  solve.tabi_phin_norm = tabi.phin_norm;
  solve.tabi_kappa_inv_a = tabi.kappa_inv_a;
  solve.tabi_coulomb_factor_kcal_mol = tabi.coulomb_factor_kcal_mol;
  solve.tabi_boundary_energy_kj_mol = tabi.boundary_energy_kcal_mol * kKcalToKJ;
  solve.tabi_reference_energy_kj_mol =
      tabi.reference_energy_kcal_mol * kKcalToKJ;
  solve.tabi_selected_energy_scale = tabi.selected_energy_scale;
  solve.tabi_fallback_used = tabi.fallback_used;
  solve.atom_reaction_potential_kj_mol_e =
      tabi.atom_reaction_potential_kj_mol_e;
  solve.atom_reaction_kj_mol = tabi.atom_reaction_energy_kj_mol;
  solve.atom_reaction_energy_closure_kj_mol =
      tabi.atom_reaction_energy_closure_kj_mol;
  solve.atom_reaction_energy_closure_error_kj_mol =
      tabi.atom_reaction_energy_closure_error_kj_mol;
  if (args_.tabi_surface_output != "off") {
    solve.tabi_output_panels = tabi.output_panels;
    solve.tabi_output_solution = tabi.output_solution;
    solve.tabi_output_rhs = tabi.output_rhs;
  }
  solve.shape.n = tabi.panel_count;
  solve.shape.nx = std::max<std::size_t>(1, tabi.panel_count);
  solve.shape.ny = 1;
  solve.shape.nz = 1;
  solve.shape.h = 0.0;
  if (!atoms.empty()) {
    solve.shape.ox = min_x;
    solve.shape.oy = min_y;
    solve.shape.oz = min_z;
    solve.focus_parent_grid = format_fixed(0.5 * (min_x + max_x), 6) + "," +
                              format_fixed(0.5 * (min_y + max_y), 6) + "," +
                              format_fixed(0.5 * (min_z + max_z), 6);
  }
  solve.shape.memory_bytes =
      tabi.peak_memory_bytes > 0 ? tabi.peak_memory_bytes : tabi.memory_bytes;
  solve.peak_memory_bytes = solve.shape.memory_bytes;
  solve.solute_points = tabi.panel_count;
  solve.solvent_points = 0;
  solve.solvent_total_iterations = tabi.iterations;
  solve.vacuum_total_iterations = 0;
  solve.solvent.iterations = tabi.iterations;
  solve.solvent.rms = tabi.residual;
  solve.solvent.max_dphi = tabi.residual;
  solve.solvent.converged = tabi.converged;
  solve.vacuum.converged = true;
  solve.pbe_model_effective = "linear";
  solve.linear_seed_solver = "none";
  solve.solver_mode = "tabi_" + tabi.sum_mode_effective;
  solve.mg_levels = 1;
  solve.focus_levels = 1;
  solve.boundary_rms = tabi.residual;
  solve.boundary_max = tabi.residual;
  solve.linear_residual = tabi.residual;
  solve.charge_assigned = tabi.total_charge;
  // A solve that did not succeed has no reaction energy. Its value here is 0
  // when nothing was computed (a refused operator) or the projection of an
  // unconverged solution, and either one read downstream as a result. The
  // projection stays in tabi_boundary_energy for diagnosis. A Born fallback
  // keeps its value; its energy source says what it is.
  const bool energy_valid = tabi.status == "ok" || tabi.status == "fallback";
  const double no_energy = std::numeric_limits<double>::quiet_NaN();
  const double energy_kcal = energy_valid ? tabi.energy_kcal_mol : no_energy;
  const double energy_kj = energy_valid ? tabi.energy_kj_mol : no_energy;
  const double energy_kt = energy_valid ? tabi.energy_kt : no_energy;
  solve.energy_kcal_mol = energy_kcal;
  solve.energy_kj_mol = energy_kj;
  solve.energy_kt = energy_kt;
  solve.atom_corrected_kcal_mol = energy_kcal;
  solve.atom_corrected_kj_mol = energy_kj;
  solve.atom_corrected_kt = energy_kt;
  solve.grid_corrected_kcal_mol = energy_kcal;
  solve.grid_corrected_kj_mol = energy_kj;
  solve.grid_corrected_kt = energy_kt;
  solve.solvent_phase_kcal_mol = energy_kcal;
  solve.solvent_phase_kj_mol = energy_kj;
  solve.solvent_phase_kt = energy_kt;
  if (tabi.surface_only) {
    solve.reaction_energy_source = tabi.reaction_energy_source;
    solve.converged = tabi.status == "ok";
    return solve;
  }
  solve.coulomb_energy_kcal_mol =
      analytic_coulomb_energy_kcal(atoms, args_.int_diel);
  solve.coulomb_energy_kj_mol = solve.coulomb_energy_kcal_mol * kKcalToKJ;
  const double kt_kcal = kKBKcalPerMolK * std::max(1.0, args_.temp);
  solve.coulomb_energy_kt =
      kt_kcal > 0.0 ? solve.coulomb_energy_kcal_mol / kt_kcal : 0.0;
  solve.reaction_energy_source = tabi.reaction_energy_source;
  solve.converged = tabi.status == "ok";
  return solve;
}


} // namespace TitanPBE
