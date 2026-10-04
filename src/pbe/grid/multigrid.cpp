#include "tabi/analyzer_impl.h"

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
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

} // namespace

AnalyzerImpl::GridShape AnalyzerImpl::coarsen_shape(const GridShape &fine) const {
  GridShape coarse;
  coarse.ox = fine.ox;
  coarse.oy = fine.oy;
  coarse.oz = fine.oz;
  coarse.h = fine.h * 2.0;
  coarse.nx = std::max<std::size_t>(3, ((fine.nx - 1) / 2) + 1);
  coarse.ny = std::max<std::size_t>(3, ((fine.ny - 1) / 2) + 1);
  coarse.nz = std::max<std::size_t>(3, ((fine.nz - 1) / 2) + 1);
  if (coarse.nx > std::numeric_limits<std::size_t>::max() / coarse.ny ||
      coarse.nx * coarse.ny >
          std::numeric_limits<std::size_t>::max() / coarse.nz) {
    throw std::runtime_error("PBE: multigrid dimensions overflow size_t.");
  }
  coarse.n = coarse.nx * coarse.ny * coarse.nz;
  coarse.memory_bytes =
      coarse.n * (7 * sizeof(double) + 2 * sizeof(unsigned char));
  if (coarse.n > args_.max_grid_points) {
    throw std::runtime_error(
        "PBE: multigrid coarse grid exceeds -pbe-max-grid.");
  }
  return coarse;
}

int AnalyzerImpl::effective_vcycle_levels(const GridShape &fine_shape) const {
  int requested = 4;
  if (args_.mg_levels != "auto") {
    requested = std::clamp(std::stoi(args_.mg_levels), 1, 4);
  }
  int levels = 1;
  GridShape current = fine_shape;
  while (levels < requested) {
    const std::size_t next_nx = ((current.nx - 1) / 2) + 1;
    const std::size_t next_ny = ((current.ny - 1) / 2) + 1;
    const std::size_t next_nz = ((current.nz - 1) / 2) + 1;
    if (next_nx < 3 || next_ny < 3 || next_nz < 3)
      break;
    ++levels;
    current = coarsen_shape(current);
    if (current.nx == 3 && current.ny == 3 && current.nz == 3)
      break;
  }
  return levels;
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
    boundary_diag = set_boundary(shape, atoms, eps_out, kappa);
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
  level.workspace = grid_;
  level.charge_diag = charge_diag;
  level.boundary_diag = boundary_diag;
  level.rhs.assign(shape.n, 0.0);
  if (physical_rhs) {
    const double rhs = effective_rhs_scale(shape);
    for (std::size_t id = 0; id < shape.n; ++id)
      level.rhs[id] = rhs * grid_.rho[id];
  }
}

void AnalyzerImpl::copy_level_to_workspace(const MultigridLevel &level) {
  grid_ = level.workspace;
}

AnalyzerImpl::SolveStats
AnalyzerImpl::smooth_multigrid_level(MultigridLevel &level, int iterations,
                                 bool stop_on_convergence) const {
  SolveStats stats;
  if (iterations <= 0)
    return stats;
  const GridShape &shape = level.shape;
  const double omega =
      args_.omega == "auto"
          ? std::clamp(2.0 / (1.0 + std::sin(kPi / static_cast<double>(
                                             std::max({shape.nx, shape.ny,
                                                       shape.nz})))),
                       1.0, 1.95)
          : args_.omega_value;
  for (int iter = 1; iter <= iterations; ++iter) {
    double sumsq = 0.0;
    double max_delta = 0.0;
    std::size_t count = 0;
    for (int color = 0; color < 2; ++color) {
      for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
        for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
          const std::size_t first_i =
              1 + (static_cast<int>((1 + j + k) & 1ULL) == color ? 0U : 1U);
          for (std::size_t i = first_i; i + 1 < shape.nx; i += 2) {
            const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
            const double neighbor_sum =
                level.workspace.eps_x[id] * level.workspace.phi[id + 1] +
                level.workspace.eps_x[id - 1] * level.workspace.phi[id - 1] +
                level.workspace.eps_y[id] *
                    level.workspace.phi[id + shape.nx] +
                level.workspace.eps_y[id - shape.nx] *
                    level.workspace.phi[id - shape.nx] +
                level.workspace.eps_z[id] *
                    level.workspace.phi[id + shape.nx * shape.ny] +
                level.workspace.eps_z[id - shape.nx * shape.ny] *
                    level.workspace.phi[id - shape.nx * shape.ny];
            const double target =
                (neighbor_sum + level.rhs[id]) / level.workspace.denom[id];
            const double updated =
                level.workspace.phi[id] +
                omega * (target - level.workspace.phi[id]);
            const double delta = updated - level.workspace.phi[id];
            level.workspace.phi[id] = updated;
            const double abs_delta = std::abs(delta);
            sumsq += delta * delta;
            max_delta = std::max(max_delta, abs_delta);
            ++count;
          }
        }
      }
    }
    stats.iterations = iter;
    stats.rms = count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
    stats.max_dphi = max_delta;
    if (stop_on_convergence && stats.rms <= args_.rms_tol &&
        stats.max_dphi <= args_.dphi_tol) {
      AlignedDoubleVector residual;
      if (compute_multigrid_residual(level, residual) <=
          grid_linear_residual_tolerance()) {
        stats.converged = true;
        break;
      }
    }
  }
  return stats;
}

double AnalyzerImpl::compute_multigrid_residual(
    const MultigridLevel &level, AlignedDoubleVector &residual) const {
  const GridShape &shape = level.shape;
  residual.assign(shape.n, 0.0);
  double sumsq = 0.0;
  std::size_t count = 0;
  for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
    for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
      for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        const double neighbor_sum =
            level.workspace.eps_x[id] * level.workspace.phi[id + 1] +
            level.workspace.eps_x[id - 1] * level.workspace.phi[id - 1] +
            level.workspace.eps_y[id] * level.workspace.phi[id + shape.nx] +
            level.workspace.eps_y[id - shape.nx] *
                level.workspace.phi[id - shape.nx] +
            level.workspace.eps_z[id] *
                level.workspace.phi[id + shape.nx * shape.ny] +
            level.workspace.eps_z[id - shape.nx * shape.ny] *
                level.workspace.phi[id - shape.nx * shape.ny];
        const double a_phi =
            level.workspace.denom[id] * level.workspace.phi[id] -
            neighbor_sum;
        const double r = level.rhs[id] - a_phi;
        residual[id] = r;
        sumsq += r * r;
        ++count;
      }
    }
  }
  return count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
}

void AnalyzerImpl::restrict_residual_to_coarse(
    const MultigridLevel &fine, const AlignedDoubleVector &fine_residual,
    MultigridLevel &coarse) const {
  std::fill(coarse.rhs.begin(), coarse.rhs.end(), 0.0);
  std::fill(coarse.workspace.phi.begin(), coarse.workspace.phi.end(), 0.0);
  const int weights[3] = {1, 2, 1};
  for (std::size_t kc = 1; kc + 1 < coarse.shape.nz; ++kc) {
    for (std::size_t jc = 1; jc + 1 < coarse.shape.ny; ++jc) {
      for (std::size_t ic = 1; ic + 1 < coarse.shape.nx; ++ic) {
        const std::size_t fi = std::min(fine.shape.nx - 2, 2 * ic);
        const std::size_t fj = std::min(fine.shape.ny - 2, 2 * jc);
        const std::size_t fk = std::min(fine.shape.nz - 2, 2 * kc);
        double sum = 0.0;
        int wsum = 0;
        for (int dz = -1; dz <= 1; ++dz) {
          const std::size_t iz = static_cast<std::size_t>(
              std::clamp<int>(static_cast<int>(fk) + dz, 1,
                              static_cast<int>(fine.shape.nz - 2)));
          for (int dy = -1; dy <= 1; ++dy) {
            const std::size_t iy = static_cast<std::size_t>(
                std::clamp<int>(static_cast<int>(fj) + dy, 1,
                                static_cast<int>(fine.shape.ny - 2)));
            for (int dx = -1; dx <= 1; ++dx) {
              const std::size_t ix = static_cast<std::size_t>(
                  std::clamp<int>(static_cast<int>(fi) + dx, 1,
                                  static_cast<int>(fine.shape.nx - 2)));
              const int w = weights[dx + 1] * weights[dy + 1] *
                            weights[dz + 1];
              sum += static_cast<double>(w) *
                     fine_residual[idx3(ix, iy, iz, fine.shape.nx,
                                        fine.shape.ny)];
              wsum += w;
            }
          }
        }
        coarse.rhs[idx3(ic, jc, kc, coarse.shape.nx, coarse.shape.ny)] =
            wsum > 0 ? sum / static_cast<double>(wsum) : 0.0;
      }
    }
  }
}

void AnalyzerImpl::prolongate_add_correction(const MultigridLevel &coarse,
                                         MultigridLevel &fine) const {
  for (std::size_t k = 1; k + 1 < fine.shape.nz; ++k) {
    const double z = fine.shape.oz + static_cast<double>(k) * fine.shape.h;
    for (std::size_t j = 1; j + 1 < fine.shape.ny; ++j) {
      const double y = fine.shape.oy + static_cast<double>(j) * fine.shape.h;
      for (std::size_t i = 1; i + 1 < fine.shape.nx; ++i) {
        const double x = fine.shape.ox + static_cast<double>(i) * fine.shape.h;
        double correction = 0.0;
        if (interpolate_field_value(coarse.shape, coarse.workspace.phi, x, y, z,
                                    correction)) {
          fine.workspace.phi[idx3(i, j, k, fine.shape.nx, fine.shape.ny)] +=
              correction;
        }
      }
    }
  }
}

void AnalyzerImpl::run_multigrid_vcycle(std::vector<MultigridLevel> &levels,
                                    std::size_t level_index) {
  MultigridLevel &level = levels[level_index];
  if (level_index + 1 >= levels.size()) {
    smooth_multigrid_level(level, args_.mg_coarse_iter, true);
    return;
  }

  smooth_multigrid_level(level, args_.mg_pre, false);
  AlignedDoubleVector residual;
  compute_multigrid_residual(level, residual);
  MultigridLevel &coarse = levels[level_index + 1];
  restrict_residual_to_coarse(level, residual, coarse);
  run_multigrid_vcycle(levels, level_index + 1);
  prolongate_add_correction(coarse, level);
  smooth_multigrid_level(level, args_.mg_post, false);
}

std::size_t AnalyzerImpl::multigrid_memory_bytes(
    const std::vector<MultigridLevel> &levels) const {
  std::size_t bytes = 0;
  for (const MultigridLevel &level : levels) {
    bytes += level.workspace.memory_bytes();
    bytes += level.rhs.size() * sizeof(double);
  }
  return bytes;
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
  const int levels_count = effective_vcycle_levels(fine_shape);
  field.levels = levels_count;
  field.mg_levels = levels_count;
  field.initial_scale = args_.scale;
  field.initial_margin = args_.margin;

  std::vector<MultigridLevel> levels(static_cast<std::size_t>(levels_count));
  GridShape shape = fine_shape;
  for (int level = 0; level < levels_count; ++level) {
    prepare_multigrid_level(levels[static_cast<std::size_t>(level)], shape,
                            atoms, eps_in, eps_out, kappa, level == 0,
                            level == 0, executor);
    if (level + 1 < levels_count)
      shape = coarsen_shape(shape);
  }
  field.peak_memory_bytes = multigrid_memory_bytes(levels);

  copy_level_to_workspace(levels.front());
  maybe_write_boundary_diff(fine_shape, atoms, eps_out, kappa);
  if (map_result != nullptr)
    maybe_write_maps(fine_shape, *map_result, false);

  const int smoothing_levels = std::max(0, levels_count - 1);
  const int cycle_work = std::max(
      1, smoothing_levels * (args_.mg_pre + args_.mg_post) +
             args_.mg_coarse_iter);
  const int max_cycles = std::max(1, args_.max_iter / cycle_work);
  SolveStats cycle_stats;
  const auto solver_begin = std::chrono::steady_clock::now();
  AlignedDoubleVector previous_phi(levels.front().workspace.phi.size());
  for (int cycle = 1; cycle <= max_cycles; ++cycle) {
    std::copy(levels.front().workspace.phi.begin(),
              levels.front().workspace.phi.end(), previous_phi.begin());
    run_multigrid_vcycle(levels, 0);
    double sumsq = 0.0;
    double max_delta = 0.0;
    std::size_t count = 0;
    for (std::size_t k = 1; k + 1 < fine_shape.nz; ++k) {
      for (std::size_t j = 1; j + 1 < fine_shape.ny; ++j) {
        for (std::size_t i = 1; i + 1 < fine_shape.nx; ++i) {
          const std::size_t id = idx3(i, j, k, fine_shape.nx, fine_shape.ny);
          const double delta =
              levels.front().workspace.phi[id] - previous_phi[id];
          sumsq += delta * delta;
          max_delta = std::max(max_delta, std::abs(delta));
          ++count;
        }
      }
    }
    cycle_stats.iterations = cycle;
    cycle_stats.rms =
        count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
    cycle_stats.max_dphi = max_delta;
    if (cycle_stats.rms <= args_.rms_tol &&
        cycle_stats.max_dphi <= args_.dphi_tol) {
      AlignedDoubleVector residual;
      if (compute_multigrid_residual(levels.front(), residual) <=
          grid_linear_residual_tolerance()) {
        cycle_stats.converged = true;
        break;
      }
    }
  }

  field.mg_cycles = cycle_stats.iterations;
  copy_level_to_workspace(levels.front());
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
  field.charge_assigned = levels.front().charge_diag.assigned_charge;
  field.charge_moment_error = levels.front().charge_diag.moment_error;
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
      std::abs(levels.front().charge_diag.assigned_charge -
               std::accumulate(atoms.begin(), atoms.end(), 0.0,
                               [](double sum, const Atom &atom) {
                                 return sum + atom.q;
                               }));
  field.boundary_rms = levels.front().boundary_diag.rms;
  field.boundary_max = levels.front().boundary_diag.max_abs;
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
