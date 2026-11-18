#include "tabi/analyzer_impl.h"

#include "pbe/grid/galerkin_mg.h"
#include "pbe/grid/grid_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace TitanPBE {

// GRID-03: -pbe-mg-levels=N builds N Galerkin levels where the grid allows
// it; auto (0) coarsens until the coarsest level is small enough for the
// dense solve.
int AnalyzerImpl::vcycle_level_cap() const {
  if (args_.mg_levels == "auto")
    return 0;
  return std::max(1, std::stoi(args_.mg_levels));
}

void AnalyzerImpl::prepare_multigrid_level(
    MultigridLevel &level, const GridShape &shape,
    const std::vector<Atom> &atoms, double eps_in, double eps_out, double kappa,
    bool physical_rhs, bool physical_boundary, TitanExecutor &executor) {
  const auto setup_begin = std::chrono::steady_clock::now();
  ensure_workspace(shape);
  add_timing_ms(
      TimingBucket::Setup,
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - setup_begin)
          .count());
  const auto surface_begin = std::chrono::steady_clock::now();
  classify_surface(shape, atoms, executor);
  apply_oracle_surface_map(shape);
  classify_ion_accessibility(shape, atoms, executor);
  add_timing_ms(
      TimingBucket::Surface,
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - surface_begin)
          .count());
  const auto epsilon_begin = std::chrono::steady_clock::now();
  const ChargeDiagnostics charge_diag =
      args_.test_charge_grid == "oracle"
          ? assign_oracle_charge_grid(shape, atoms)
          : assign_charges(shape, atoms);
  build_coefficients(shape, atoms, eps_in, eps_out, kappa, executor);
  add_timing_ms(
      TimingBucket::Epsilon,
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - epsilon_begin)
          .count());
  BoundaryDiagnostics boundary_diag;
  if (physical_boundary) {
    const auto boundary_begin = std::chrono::steady_clock::now();
    boundary_diag = set_boundary(shape, atoms, eps_out, kappa, executor);
    if (args_.test_boundary_map == "oracle" && oracle_replay_applies(shape))
      boundary_diag = apply_oracle_boundary_map(shape, eps_out, kappa);
    add_timing_ms(
        TimingBucket::Boundary,
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - boundary_begin)
            .count());
  } else {
    std::fill(grid_.phi.begin(), grid_.phi.end(), 0.0);
  }

  level.shape = shape;
  level.charge_diag = charge_diag;
  level.boundary_diag = boundary_diag;
  level.rhs.assign(shape.n, 0.0);
  if (physical_rhs) {
    const double rhs = effective_rhs_scale(shape);
    for (std::size_t id = 0; id < shape.n; ++id)
      level.rhs[id] = rhs * grid_.rho[id];
  }
}

AnalyzerImpl::FieldSolve AnalyzerImpl::solve_vcycle_field(
    const std::vector<Atom> &atoms, double eps_in, double eps_out, double kappa,
    TitanExecutor &executor, const FrameResult *map_result) {
  FieldSolve field;
  field.solver_mode = "vcycle";
  field.mg_pre = args_.mg_pre;
  field.mg_post = args_.mg_post;
  field.mg_coarse_iter = args_.mg_coarse_iter;

  const GridShape fine_shape = build_shape(atoms);
  field.initial_scale = args_.scale;
  field.initial_margin = args_.margin;

  // GRID-03: only the fine level is discretised (grid_ holds its operator);
  // every coarse operator is the Galerkin product P^T A P of the one above.
  MultigridLevel fine;
  prepare_multigrid_level(fine, fine_shape, atoms, eps_in, eps_out, kappa,
                          true, true, executor);
  maybe_write_boundary_diff(fine_shape, atoms, eps_out, kappa);
  if (map_result != nullptr)
    maybe_write_maps(fine_shape, *map_result, false);

  galerkin_mg::Options options;
  options.max_levels = vcycle_level_cap();
  options.pre_sweeps = args_.mg_pre;
  options.post_sweeps = args_.mg_post;
  options.coarse_sweeps = args_.mg_coarse_iter;
  options.workers = effective_workers(executor, fine_shape.nz);
  galerkin_mg::Hierarchy hierarchy;
  const auto hierarchy_begin = std::chrono::steady_clock::now();
  hierarchy.build({fine_shape.nx, fine_shape.ny, fine_shape.nz,
                   grid_.eps_x.data(), grid_.eps_y.data(), grid_.eps_z.data(),
                   grid_.denom.data()},
                  options, executor);
  add_timing_ms(
      TimingBucket::Setup,
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - hierarchy_begin)
          .count());
  field.levels = hierarchy.level_count();
  field.mg_levels = hierarchy.level_count();
  // workspace, fine right-hand side, coarse levels, four solver vectors
  field.peak_memory_bytes = grid_.memory_bytes() +
                            fine.rhs.size() * sizeof(double) +
                            hierarchy.memory_bytes() +
                            4 * fine_shape.n * sizeof(double);

  // -pbe-max-iter is a budget in fine sweeps; one iteration costs the pre-
  // and post-smoothing sweeps plus about one fine operator application.
  const int cycle_work = std::max(1, args_.mg_pre + args_.mg_post + 1);
  const int max_cycles = std::max(1, args_.max_iter / cycle_work);
  // CG needs a symmetric V-cycle: as many sweeps after the coarse
  // correction as before it.
  const bool use_cg = args_.mg_accel == "cg" && args_.mg_pre == args_.mg_post;
  SolveStats cycle_stats;
  const double residual_tolerance = grid_linear_residual_tolerance();
  const auto solver_begin = std::chrono::steady_clock::now();
  const galerkin_mg::SolveResult solved = galerkin_mg::solve(
      hierarchy, fine.rhs.data(), grid_.phi.data(), max_cycles, use_cg,
      [&](const galerkin_mg::IterationReport &report) {
        cycle_stats.iterations = report.iteration;
        cycle_stats.rms = report.update_rms;
        cycle_stats.max_dphi = report.update_max;
        if (report.update_rms > args_.rms_tol ||
            report.update_max > args_.dphi_tol)
          return false;
        cycle_stats.converged =
            galerkin_mg::residual_rms(hierarchy, fine.rhs.data(),
                                      grid_.phi.data(), executor) <=
            residual_tolerance;
        return cycle_stats.converged;
      },
      executor);
  field.mg_cycles = solved.iterations;
  const int cycle_iterations = field.mg_cycles * cycle_work;
  SolveStats final_stats;
  if (cycle_stats.converged) {
    final_stats = cycle_stats;
    final_stats.iterations = 0;
  } else {
    const int remaining_iterations =
        std::max(0, args_.max_iter - cycle_iterations);
    if (remaining_iterations > 0) {
      final_stats = solve_sor(fine_shape, executor, remaining_iterations);
    } else {
      final_stats = cycle_stats;
      final_stats.iterations = 0;
    }
  }
  add_timing_ms(
      TimingBucket::Solver,
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - solver_begin)
          .count());
  field.total_iterations = cycle_iterations + final_stats.iterations;
  field.stats = final_stats;
  field.boundary_convention = effective_boundary_convention();
  field.field_units = args_.test_field_units;
  field.rhs_scale = effective_rhs_scale(fine_shape);
  field.charge_assigned = fine.charge_diag.assigned_charge;
  field.charge_moment_error = fine.charge_diag.moment_error;
  field.gaussian_density_sum = last_gaussian_density_sum_;
  field.gaussian_cutoff_points = last_gaussian_cutoff_points_;
  field.gaussian_eps_min = last_gaussian_eps_min_;
  field.gaussian_eps_max = last_gaussian_eps_max_;
  field.gaussian_eps_mean = last_gaussian_eps_mean_;
  field.vdwms_enabled = last_vdwms_audit_.enabled;
  field.vdwms_boundary_initial = last_vdwms_audit_.boundary_initial;
  field.vdwms_boundary_final = last_vdwms_audit_.boundary_final;
  field.vdwms_added = last_vdwms_audit_.added;
  field.vdwms_removed = last_vdwms_audit_.removed;
  field.vdwms_exposed_points = last_vdwms_audit_.exposed_points;
  field.vdwms_midpoint_rms = last_vdwms_audit_.midpoint_rms;
  field.gaussian_charge_norm_error =
      std::abs(fine.charge_diag.assigned_charge -
               std::accumulate(atoms.begin(), atoms.end(), 0.0,
                               [](double sum, const Atom &atom) {
                                 return sum + atom.q;
                               }));
  field.boundary_rms = fine.boundary_diag.rms;
  field.boundary_max = fine.boundary_diag.max_abs;
  field.linear_residual = compute_linear_residual(fine_shape);
  field.stats.converged =
      field.stats.converged &&
      field.linear_residual <= grid_linear_residual_tolerance();
  field.parity = compute_parity_audit(fine_shape, nullptr, nullptr);
  if (map_result != nullptr)
    maybe_write_maps(fine_shape, *map_result, true);
  field.shape = fine_shape;
  field.solute_points = static_cast<std::size_t>(
      std::count(grid_.inside.begin(), grid_.inside.end(), 1U));
  field.solvent_points = fine_shape.n - field.solute_points;
  field.ion_accessible_points = static_cast<std::size_t>(
      std::count(grid_.ion_accessible.begin(), grid_.ion_accessible.end(), 1U));
  field.stern_excluded_points =
      args_.ion_access == "stern" ? fine_shape.n - field.ion_accessible_points
                                  : 0;
  field.atom_phi = interpolate_atoms(fine_shape, atoms);
  const bool keep_field_grids =
      parity_map_dump_enabled() ||
      args_.test_energy_convention == "delphi-iscm" ||
      args_.test_energy_convention == "delphi-corrected" ||
      args_.biomodel == "rpbe" ||
      barnase_forensic_enabled();
  if (args_.charge_assignment == "delphi" ||
      args_.charge_assignment == "gaussian" || keep_field_grids)
    field.phi_grid = grid_.phi;
  if (keep_field_grids) {
    field.rho_grid = grid_.rho;
    field.screen_grid = grid_.screen;
    field.inside_grid = grid_.inside;
    field.eps_x_grid = grid_.eps_x;
    field.eps_y_grid = grid_.eps_y;
    field.eps_z_grid = grid_.eps_z;
    field.eps_sum_grid.assign(fine_shape.n, 0.0);
    for (std::size_t id = 0; id < fine_shape.n; ++id) {
      field.eps_sum_grid[id] =
          grid_.denom.size() == fine_shape.n && grid_.screen.size() == fine_shape.n
              ? std::max(1.0e-12, grid_.denom[id] - grid_.screen[id])
              : 0.0;
    }
  }
  if (map_result != nullptr && (kappa > 0.0 || eps_out > eps_in + 1.0e-9))
    store_stability_medium_state(fine_shape);
  return field;
}

} // namespace TitanPBE
