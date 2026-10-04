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
using analyzer_impl_detail::effective_workers;
using analyzer_impl_detail::idx3;
using analyzer_impl_detail::kCoulombKcalPerMol;
using analyzer_impl_detail::kKBKcalPerMolK;
using analyzer_impl_detail::kKBKJPerMolK;
using analyzer_impl_detail::kKcalToKJ;
using analyzer_impl_detail::kPi;

namespace {

struct NwtHomoOffsets {
  std::ptrdiff_t xp{0};
  std::ptrdiff_t xm{0};
  std::ptrdiff_t yp{0};
  std::ptrdiff_t ym{0};
  std::ptrdiff_t zp{0};
  std::ptrdiff_t zm{0};
};

NwtHomoOffsets nwt_homo_offsets(std::size_t nx_value, std::size_t ny_value,
                                int color) noexcept {
  const std::ptrdiff_t nx = static_cast<std::ptrdiff_t>(nx_value);
  const std::ptrdiff_t xy = static_cast<std::ptrdiff_t>(nx_value * ny_value);
  if (color == 0) {
    return {0, -1, nx / 2, -((nx + 1) / 2), xy / 2, -((xy + 1) / 2)};
  }
  return {1, 0, (nx + 1) / 2, -(nx / 2), (xy + 1) / 2, -(xy / 2)};
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
void update_nwt_homo_linear_span_avx2(double *write_phi, const double *read_phi,
                                      const double *rhs, const double *eps_sum,
                                      const double *eps, std::uint32_t begin,
                                      std::uint32_t count,
                                      const NwtHomoOffsets offsets,
                                      double omega, bool accumulate_rms,
                                      double &sumsq, double &max_delta,
                                      std::size_t &sample_count) noexcept {
  const __m256d omega_v = _mm256_set1_pd(omega);
  const __m256d min_diag = _mm256_set1_pd(1.0e-12);
  const __m256d sign_mask = _mm256_set1_pd(-0.0);
  __m256d sumsq_v = _mm256_setzero_pd();
  __m256d max_v = _mm256_setzero_pd();

  std::size_t h = begin;
  const std::size_t end = static_cast<std::size_t>(begin) + count;
  for (; h + 4 <= end; h += 4) {
    const auto base = static_cast<std::ptrdiff_t>(h);
    const __m256d old_phi = _mm256_loadu_pd(write_phi + h);
    __m256d phi_sum =
        _mm256_add_pd(_mm256_loadu_pd(read_phi + base + offsets.xp),
                      _mm256_loadu_pd(read_phi + base + offsets.xm));
    phi_sum =
        _mm256_add_pd(phi_sum, _mm256_loadu_pd(read_phi + base + offsets.yp));
    phi_sum =
        _mm256_add_pd(phi_sum, _mm256_loadu_pd(read_phi + base + offsets.ym));
    phi_sum =
        _mm256_add_pd(phi_sum, _mm256_loadu_pd(read_phi + base + offsets.zp));
    phi_sum =
        _mm256_add_pd(phi_sum, _mm256_loadu_pd(read_phi + base + offsets.zm));
    const __m256d numerator =
        _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(eps + h), phi_sum),
                      _mm256_loadu_pd(rhs + h));
    const __m256d diagonal =
        _mm256_max_pd(_mm256_loadu_pd(eps_sum + h), min_diag);
    const __m256d target = _mm256_div_pd(numerator, diagonal);
    const __m256d delta =
        _mm256_mul_pd(omega_v, _mm256_sub_pd(target, old_phi));
    const __m256d updated = _mm256_add_pd(old_phi, delta);
    _mm256_storeu_pd(write_phi + h, updated);
    if (accumulate_rms) {
      sumsq_v = _mm256_add_pd(sumsq_v, _mm256_mul_pd(delta, delta));
      const __m256d abs_delta = _mm256_andnot_pd(sign_mask, delta);
      max_v = _mm256_max_pd(max_v, abs_delta);
    }
  }

  if (accumulate_rms) {
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, sumsq_v);
    sumsq += lanes[0] + lanes[1] + lanes[2] + lanes[3];
    _mm256_store_pd(lanes, max_v);
    max_delta = std::max(max_delta, std::max(std::max(lanes[0], lanes[1]),
                                             std::max(lanes[2], lanes[3])));
    sample_count += h - static_cast<std::size_t>(begin);
  }

  for (; h < end; ++h) {
    const auto base = static_cast<std::ptrdiff_t>(h);
    const double old_phi = write_phi[h];
    const double neighbor_sum =
        eps[h] * (read_phi[base + offsets.xp] + read_phi[base + offsets.xm] +
                  read_phi[base + offsets.yp] + read_phi[base + offsets.ym] +
                  read_phi[base + offsets.zp] + read_phi[base + offsets.zm]);
    const double target =
        (neighbor_sum + rhs[h]) / std::max(1.0e-12, eps_sum[h]);
    const double updated = old_phi + omega * (target - old_phi);
    const double delta = updated - old_phi;
    write_phi[h] = updated;
    if (accumulate_rms) {
      sumsq += delta * delta;
      max_delta = std::max(max_delta, std::abs(delta));
      ++sample_count;
    }
  }
}
#endif

std::pair<double, double> delphi_sinh_cosh_tail(double phi,
                                                double clip) noexcept {
  const double effective_clip = std::max(1.0e-12, clip);
  const double sinh_clip = std::sinh(effective_clip);
  if (phi > effective_clip) {
    const double sinh_phi = sinh_clip + sinh_clip * (phi - effective_clip);
    return {sinh_phi, std::sqrt(1.0 + sinh_phi * sinh_phi)};
  }
  if (phi < -effective_clip) {
    const double sinh_phi = -sinh_clip + sinh_clip * (phi + effective_clip);
    return {sinh_phi, std::sqrt(1.0 + sinh_phi * sinh_phi)};
  }
  return {std::sinh(phi), std::cosh(phi)};
}

std::pair<double, double>
delphi_sinh_cosh_tail_precomputed(double phi, double effective_clip,
                                  double sinh_clip) noexcept {
  if (phi > effective_clip) {
    const double sinh_phi = sinh_clip + sinh_clip * (phi - effective_clip);
    return {sinh_phi, std::sqrt(1.0 + sinh_phi * sinh_phi)};
  }
  if (phi < -effective_clip) {
    const double sinh_phi = -sinh_clip + sinh_clip * (phi + effective_clip);
    return {sinh_phi, std::sqrt(1.0 + sinh_phi * sinh_phi)};
  }
  return {std::sinh(phi), std::cosh(phi)};
}

std::pair<double, double>
delphi_sinh_cosh_tail_single_sinh(double phi, double effective_clip,
                                  double sinh_clip) noexcept {
  if (phi > effective_clip) {
    const double sinh_phi = sinh_clip + sinh_clip * (phi - effective_clip);
    return {sinh_phi, std::sqrt(1.0 + sinh_phi * sinh_phi)};
  }
  if (phi < -effective_clip) {
    const double sinh_phi = -sinh_clip + sinh_clip * (phi + effective_clip);
    return {sinh_phi, std::sqrt(1.0 + sinh_phi * sinh_phi)};
  }
  const double abs_phi = std::abs(phi);
  if (abs_phi <= 1.5) {
    const double x2 = phi * phi;
    const double sinh_poly =
        phi *
        (1.0 + x2 * (1.0 / 6.0 +
                     x2 * (1.0 / 120.0 +
                           x2 * (1.0 / 5040.0 +
                                 x2 * (1.0 / 362880.0 +
                                       x2 * (1.0 / 39916800.0 +
                                             x2 * (1.0 / 6227020800.0)))))));
    const double cosh_poly =
        1.0 + x2 * (0.5 + x2 * (1.0 / 24.0 +
                                x2 * (1.0 / 720.0 +
                                      x2 * (1.0 / 40320.0 +
                                            x2 * (1.0 / 3628800.0 +
                                                  x2 * (1.0 / 479001600.0))))));
    return {sinh_poly, cosh_poly};
  }
  const double sinh_phi = std::sinh(phi);
  return {sinh_phi, std::sqrt(1.0 + sinh_phi * sinh_phi)};
}

} // namespace

double
AnalyzerImpl::compute_nonlinear_residual(const GridShape &shape,
                                         const AlignedDoubleVector &source,
                                         double screen_scale) const {
  return compute_nonlinear_residual_from_phi(shape, grid_.phi, source,
                                             screen_scale);
}

double AnalyzerImpl::compute_nonlinear_residual_from_phi(
    const GridShape &shape, const AlignedDoubleVector &field,
    const AlignedDoubleVector &source, double screen_scale) const {
  if (shape.n == 0 || field.size() != shape.n || grid_.rho.size() != shape.n ||
      source.size() != shape.n)
    return 0.0;
  const double rhs = nonlinear_rhs_scale(shape, nonlinear_alpha());
  double sumsq = 0.0;
  std::size_t count = 0;
  for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
    for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
      for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        const double neighbor_sum =
            grid_.eps_x[id] * field[id + 1] +
            grid_.eps_x[id - 1] * field[id - 1] +
            grid_.eps_y[id] * field[id + shape.nx] +
            grid_.eps_y[id - shape.nx] * field[id - shape.nx] +
            grid_.eps_z[id] * field[id + shape.nx * shape.ny] +
            grid_.eps_z[id - shape.nx * shape.ny] *
                field[id - shape.nx * shape.ny];
        const double eps_sum =
            std::max(1.0e-12, grid_.denom[id] - grid_.screen[id]);
        const double residual =
            rhs * grid_.rho[id] -
            (eps_sum * field[id] - neighbor_sum +
             (grid_.screen[id] * screen_scale) * source[id]);
        sumsq += residual * residual;
        ++count;
      }
    }
  }
  return count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
}

double AnalyzerImpl::compute_nonlinear_residual_vector(
    const GridShape &shape, const AlignedDoubleVector &field,
    const AlignedDoubleVector &source, double screen_scale,
    AlignedDoubleVector &residual, TitanExecutor &executor) const {
  residual.assign(shape.n, 0.0);
  if (shape.n == 0 || field.size() != shape.n || grid_.rho.size() != shape.n ||
      source.size() != shape.n)
    return 0.0;

  const std::size_t interior_z = shape.nz > 2 ? shape.nz - 2 : 0;
  const std::size_t workers = effective_workers(executor, interior_z);
  std::vector<double> local_sumsq(workers, 0.0);
  std::vector<std::size_t> local_count(workers, 0);
  const double rhs = nonlinear_rhs_scale(shape, nonlinear_alpha());
  const std::size_t xy = shape.nx * shape.ny;

  executor.parallel_for(
      interior_z, workers, 1,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        double sumsq = 0.0;
        std::size_t count = 0;
        for (std::size_t kk = begin; kk < end; ++kk) {
          const std::size_t k = kk + 1;
          for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
            for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
              const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
              const double neighbor_sum =
                  grid_.eps_x[id] * field[id + 1] +
                  grid_.eps_x[id - 1] * field[id - 1] +
                  grid_.eps_y[id] * field[id + shape.nx] +
                  grid_.eps_y[id - shape.nx] * field[id - shape.nx] +
                  grid_.eps_z[id] * field[id + xy] +
                  grid_.eps_z[id - xy] * field[id - xy];
              const double eps_sum =
                  std::max(1.0e-12, grid_.denom[id] - grid_.screen[id]);
              const double r = rhs * grid_.rho[id] -
                               (eps_sum * field[id] - neighbor_sum +
                                (grid_.screen[id] * screen_scale) * source[id]);
              residual[id] = r;
              sumsq += r * r;
              ++count;
            }
          }
        }
        local_sumsq[worker] = sumsq;
        local_count[worker] = count;
      });

  double sumsq = 0.0;
  std::size_t count = 0;
  for (std::size_t worker = 0; worker < workers; ++worker) {
    sumsq += local_sumsq[worker];
    count += local_count[worker];
  }
  return count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
}

AnalyzerImpl::NewtonCorrectionStats AnalyzerImpl::solve_newton_correction_sor(
    const GridShape &shape, const AlignedDoubleVector &residual,
    double residual_rms, double alpha, double clip, double screen_scale,
    AlignedDoubleVector &delta, int max_sweeps, TitanExecutor &executor) const {
  NewtonCorrectionStats stats;
  delta.assign(shape.n, 0.0);
  if (shape.n == 0 || residual.size() != shape.n)
    return stats;

  const int requested_sweeps =
      std::clamp(args_.nl_inner_iter, 1, std::max(1, args_.nl_inner_iter));
  const int sweeps = max_sweeps > 0 ? std::min(requested_sweeps, max_sweeps)
                                    : requested_sweeps;
  const double omega =
      args_.omega == "auto"
          ? std::clamp(2.0 / (1.0 + std::sin(kPi / static_cast<double>(std::max(
                                                       {shape.nx, shape.ny,
                                                        shape.nz})))),
                       1.0, 1.95)
          : args_.omega_value;
  const double max_correction = 20.0 * std::max(1.0e-12, shape.h);
  const std::size_t interior_z = shape.nz > 2 ? shape.nz - 2 : 0;
  const std::size_t workers = effective_workers(executor, interior_z);
  const std::size_t xy = shape.nx * shape.ny;
  std::vector<double> local_sumsq(workers, 0.0);
  std::vector<double> local_max(workers, 0.0);
  std::vector<std::size_t> local_count(workers, 0);
  std::vector<double> local_resid_sumsq(workers, 0.0);
  std::vector<double> local_resid_max(workers, 0.0);
  std::vector<std::size_t> local_resid_count(workers, 0);
  std::vector<double> local_jac_min(workers,
                                    std::numeric_limits<double>::max());
  std::vector<double> local_jac_max(workers, 0.0);
  std::vector<std::size_t> local_screen_points(workers, 0);
  std::vector<std::size_t> local_active_points(workers, 0);

  auto compute_correction_residual = [&]() {
    std::fill(local_resid_sumsq.begin(), local_resid_sumsq.end(), 0.0);
    std::fill(local_resid_max.begin(), local_resid_max.end(), 0.0);
    std::fill(local_resid_count.begin(), local_resid_count.end(), 0);
    std::fill(local_jac_min.begin(), local_jac_min.end(),
              std::numeric_limits<double>::max());
    std::fill(local_jac_max.begin(), local_jac_max.end(), 0.0);
    std::fill(local_screen_points.begin(), local_screen_points.end(), 0);
    std::fill(local_active_points.begin(), local_active_points.end(), 0);

    executor.parallel_for(
        interior_z, workers, 1,
        [&](std::size_t worker, std::size_t begin, std::size_t end) {
          double sumsq = 0.0;
          double max_abs = 0.0;
          double jac_min = std::numeric_limits<double>::max();
          double jac_max = 0.0;
          std::size_t count = 0;
          std::size_t screen_points = 0;
          std::size_t active_points = 0;
          for (std::size_t kk = begin; kk < end; ++kk) {
            const std::size_t k = kk + 1;
            for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
              for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
                const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
                const double eps_sum =
                    std::max(1.0e-12, grid_.denom[id] - grid_.screen[id]);
                const double diag = std::max(
                    1.0e-12, eps_sum + (grid_.screen[id] * screen_scale) *
                                           nonlinear_source_derivative(
                                               grid_.phi[id], alpha, clip));
                const double neighbor_delta_sum =
                    grid_.eps_x[id] * delta[id + 1] +
                    grid_.eps_x[id - 1] * delta[id - 1] +
                    grid_.eps_y[id] * delta[id + shape.nx] +
                    grid_.eps_y[id - shape.nx] * delta[id - shape.nx] +
                    grid_.eps_z[id] * delta[id + xy] +
                    grid_.eps_z[id - xy] * delta[id - xy];
                const double j_delta = diag * delta[id] - neighbor_delta_sum;
                const double r = residual[id] - j_delta;
                const double abs_r = std::abs(r);
                sumsq += r * r;
                max_abs = std::max(max_abs, abs_r);
                jac_min = std::min(jac_min, diag);
                jac_max = std::max(jac_max, diag);
                if (grid_.screen[id] > 0.0)
                  ++screen_points;
                if (grid_.screen[id] * screen_scale > 0.0)
                  ++active_points;
                ++count;
              }
            }
          }
          local_resid_sumsq[worker] = sumsq;
          local_resid_max[worker] = max_abs;
          local_resid_count[worker] = count;
          local_jac_min[worker] = jac_min;
          local_jac_max[worker] = jac_max;
          local_screen_points[worker] = screen_points;
          local_active_points[worker] = active_points;
        });

    double sumsq = 0.0;
    double max_abs = 0.0;
    double jac_min = std::numeric_limits<double>::max();
    double jac_max = 0.0;
    std::size_t count = 0;
    std::size_t screen_points = 0;
    std::size_t active_points = 0;
    for (std::size_t worker = 0; worker < workers; ++worker) {
      sumsq += local_resid_sumsq[worker];
      max_abs = std::max(max_abs, local_resid_max[worker]);
      count += local_resid_count[worker];
      jac_min = std::min(jac_min, local_jac_min[worker]);
      jac_max = std::max(jac_max, local_jac_max[worker]);
      screen_points += local_screen_points[worker];
      active_points += local_active_points[worker];
    }
    stats.residual_rms =
        count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
    stats.residual_max = max_abs;
    stats.jac_diag_min = count > 0 ? jac_min : 0.0;
    stats.jac_diag_max = count > 0 ? jac_max : 0.0;
    stats.screen_points = screen_points;
    stats.active_nonlinear_points = active_points;
  };

  compute_correction_residual();
  const double initial_correction_residual =
      std::isfinite(stats.residual_rms) ? stats.residual_rms : residual_rms;
  double best_residual = stats.residual_rms;
  int stale_iterations = 0;

  for (int iter = 1; iter <= sweeps; ++iter) {
    std::fill(local_sumsq.begin(), local_sumsq.end(), 0.0);
    std::fill(local_max.begin(), local_max.end(), 0.0);
    std::fill(local_count.begin(), local_count.end(), 0);

    for (int color = 0; color < 2; ++color) {
      executor.parallel_for(
          interior_z, workers, 1,
          [&](std::size_t worker, std::size_t begin, std::size_t end) {
            double sumsq = 0.0;
            double max_update = 0.0;
            std::size_t count = 0;
            for (std::size_t kk = begin; kk < end; ++kk) {
              const std::size_t k = kk + 1;
              for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
                const std::size_t first_i =
                    1 +
                    (static_cast<int>((1 + j + k) & 1ULL) == color ? 0U : 1U);
                for (std::size_t i = first_i; i + 1 < shape.nx; i += 2) {
                  const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
                  const double eps_sum =
                      std::max(1.0e-12, grid_.denom[id] - grid_.screen[id]);
                  const double diag = std::max(
                      1.0e-12, eps_sum + (grid_.screen[id] * screen_scale) *
                                             nonlinear_source_derivative(
                                                 grid_.phi[id], alpha, clip));
                  const double neighbor_delta_sum =
                      grid_.eps_x[id] * delta[id + 1] +
                      grid_.eps_x[id - 1] * delta[id - 1] +
                      grid_.eps_y[id] * delta[id + shape.nx] +
                      grid_.eps_y[id - shape.nx] * delta[id - shape.nx] +
                      grid_.eps_z[id] * delta[id + xy] +
                      grid_.eps_z[id - xy] * delta[id - xy];
                  const double target =
                      std::clamp((residual[id] + neighbor_delta_sum) / diag,
                                 -max_correction, max_correction);
                  const double updated =
                      delta[id] + omega * (target - delta[id]);
                  const double d = updated - delta[id];
                  delta[id] = updated;
                  const double abs_d = std::abs(d);
                  sumsq += d * d;
                  max_update = std::max(max_update, abs_d);
                  ++count;
                }
              }
            }
            local_sumsq[worker] += sumsq;
            local_max[worker] = std::max(local_max[worker], max_update);
            local_count[worker] += count;
          });
    }

    double sumsq = 0.0;
    double max_update = 0.0;
    std::size_t count = 0;
    for (std::size_t worker = 0; worker < workers; ++worker) {
      sumsq += local_sumsq[worker];
      max_update = std::max(max_update, local_max[worker]);
      count += local_count[worker];
    }
    stats.sweeps.iterations = iter;
    stats.sweeps.rms =
        count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
    stats.sweeps.max_dphi = max_update;

    compute_correction_residual();
    const double correction_target =
        std::max(args_.nl_residual, 0.05 * initial_correction_residual);
    if (stats.residual_rms <= correction_target) {
      stats.sweeps.converged = true;
      stats.residual_decreased = true;
      break;
    }
    if (std::isfinite(stats.residual_rms) &&
        stats.residual_rms < best_residual * 0.999) {
      best_residual = stats.residual_rms;
      stale_iterations = 0;
    } else if (++stale_iterations >= std::min(4, args_.nl_stagnation)) {
      break;
    }
  }
  stats.residual_decreased = std::isfinite(stats.residual_rms) &&
                             stats.residual_rms < initial_correction_residual;
  stats.inner_levels = 1;
  stats.inner_cycles = stats.sweeps.iterations;
  stats.inner_residual_before = initial_correction_residual;
  stats.inner_residual_after = stats.residual_rms;
  stats.inner_effective = "sor";
  return stats;
}

void AnalyzerImpl::prepare_newton_correction_hierarchy(
    const GridShape &shape, const AlignedDoubleVector &residual, double alpha,
    double clip, double screen_scale,
    std::vector<NewtonCorrectionLevel> &levels) const {
  const int level_count = effective_vcycle_levels(shape);
  levels.clear();
  levels.resize(static_cast<std::size_t>(level_count));

  auto allocate_level = [](NewtonCorrectionLevel &level,
                           const GridShape &level_shape,
                           double max_correction) {
    level.shape = level_shape;
    level.delta.assign(level_shape.n, 0.0);
    level.rhs.assign(level_shape.n, 0.0);
    level.eps_x.assign(level_shape.n, 0.0);
    level.eps_y.assign(level_shape.n, 0.0);
    level.eps_z.assign(level_shape.n, 0.0);
    level.diag.assign(level_shape.n, 1.0);
    level.screen.assign(level_shape.n, 0.0);
    level.base_phi.assign(level_shape.n, 0.0);
    level.max_correction = max_correction;
  };

  auto fill_diag = [&](NewtonCorrectionLevel &level) {
    const GridShape &s = level.shape;
    const std::size_t xy = s.nx * s.ny;
    std::fill(level.diag.begin(), level.diag.end(), 1.0);
    if (s.nx < 3 || s.ny < 3 || s.nz < 3)
      return;
    for (std::size_t k = 1; k + 1 < s.nz; ++k) {
      for (std::size_t j = 1; j + 1 < s.ny; ++j) {
        for (std::size_t i = 1; i + 1 < s.nx; ++i) {
          const std::size_t id = idx3(i, j, k, s.nx, s.ny);
          const double eps_sum = level.eps_x[id] + level.eps_x[id - 1] +
                                 level.eps_y[id] + level.eps_y[id - s.nx] +
                                 level.eps_z[id] + level.eps_z[id - xy];
          level.diag[id] = std::max(
              1.0e-12, eps_sum + (level.screen[id] * screen_scale) *
                                     nonlinear_source_derivative(
                                         level.base_phi[id], alpha, clip));
        }
      }
    }
  };

  const double correction_limit = 20.0 * std::max(1.0e-12, shape.h);
  allocate_level(levels.front(), shape, correction_limit);
  levels.front().rhs = residual;
  levels.front().eps_x = grid_.eps_x;
  levels.front().eps_y = grid_.eps_y;
  levels.front().eps_z = grid_.eps_z;
  levels.front().screen = grid_.screen;
  levels.front().base_phi = grid_.phi;
  for (std::size_t k = 0; k < shape.nz; ++k) {
    for (std::size_t j = 0; j < shape.ny; ++j) {
      for (std::size_t i = 0; i < shape.nx; ++i) {
        if (i == 0 || j == 0 || k == 0 || i + 1 == shape.nx ||
            j + 1 == shape.ny || k + 1 == shape.nz) {
          levels.front().rhs[idx3(i, j, k, shape.nx, shape.ny)] = 0.0;
        }
      }
    }
  }
  fill_diag(levels.front());

  for (std::size_t level_index = 1; level_index < levels.size();
       ++level_index) {
    const NewtonCorrectionLevel &fine = levels[level_index - 1];
    NewtonCorrectionLevel &coarse = levels[level_index];
    allocate_level(coarse, coarsen_shape(fine.shape), correction_limit);
    for (std::size_t k = 0; k < coarse.shape.nz; ++k) {
      const double z =
          coarse.shape.oz + static_cast<double>(k) * coarse.shape.h;
      for (std::size_t j = 0; j < coarse.shape.ny; ++j) {
        const double y =
            coarse.shape.oy + static_cast<double>(j) * coarse.shape.h;
        for (std::size_t i = 0; i < coarse.shape.nx; ++i) {
          const double x =
              coarse.shape.ox + static_cast<double>(i) * coarse.shape.h;
          const std::size_t id =
              idx3(i, j, k, coarse.shape.nx, coarse.shape.ny);
          double value = 0.0;
          if (interpolate_field_value(fine.shape, fine.eps_x, x, y, z, value))
            coarse.eps_x[id] = value;
          if (interpolate_field_value(fine.shape, fine.eps_y, x, y, z, value))
            coarse.eps_y[id] = value;
          if (interpolate_field_value(fine.shape, fine.eps_z, x, y, z, value))
            coarse.eps_z[id] = value;
          if (interpolate_field_value(fine.shape, fine.screen, x, y, z, value))
            coarse.screen[id] = 4.0 * value;
          if (interpolate_field_value(fine.shape, fine.base_phi, x, y, z,
                                      value))
            coarse.base_phi[id] = value;
        }
      }
    }
    fill_diag(coarse);
  }
}

AnalyzerImpl::SolveStats
AnalyzerImpl::smooth_newton_correction_level(NewtonCorrectionLevel &level,
                                             int iterations,
                                             bool stop_on_convergence) const {
  SolveStats stats;
  if (iterations <= 0)
    return stats;
  const GridShape &shape = level.shape;
  const double omega =
      args_.omega == "auto"
          ? std::clamp(2.0 / (1.0 + std::sin(kPi / static_cast<double>(std::max(
                                                       {shape.nx, shape.ny,
                                                        shape.nz})))),
                       1.0, 1.95)
          : args_.omega_value;
  const double max_correction = std::max(1.0e-12, level.max_correction);
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
                level.eps_x[id] * level.delta[id + 1] +
                level.eps_x[id - 1] * level.delta[id - 1] +
                level.eps_y[id] * level.delta[id + shape.nx] +
                level.eps_y[id - shape.nx] * level.delta[id - shape.nx] +
                level.eps_z[id] * level.delta[id + shape.nx * shape.ny] +
                level.eps_z[id - shape.nx * shape.ny] *
                    level.delta[id - shape.nx * shape.ny];
            const double target =
                std::clamp((level.rhs[id] + neighbor_sum) /
                               std::max(1.0e-12, level.diag[id]),
                           -max_correction, max_correction);
            const double updated =
                std::clamp(level.delta[id] + omega * (target - level.delta[id]),
                           -max_correction, max_correction);
            const double d = updated - level.delta[id];
            level.delta[id] = updated;
            sumsq += d * d;
            max_delta = std::max(max_delta, std::abs(d));
            ++count;
          }
        }
      }
    }
    stats.iterations = iter;
    stats.rms = count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
    stats.max_dphi = max_delta;
    if (stop_on_convergence &&
        (stats.rms <= args_.nl_residual || stats.max_dphi <= args_.nl_dphi)) {
      stats.converged = true;
      break;
    }
  }
  return stats;
}

double AnalyzerImpl::compute_newton_correction_level_residual(
    const NewtonCorrectionLevel &level, AlignedDoubleVector &residual) const {
  const GridShape &shape = level.shape;
  residual.assign(shape.n, 0.0);
  double sumsq = 0.0;
  std::size_t count = 0;
  for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
    for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
      for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        const double neighbor_sum =
            level.eps_x[id] * level.delta[id + 1] +
            level.eps_x[id - 1] * level.delta[id - 1] +
            level.eps_y[id] * level.delta[id + shape.nx] +
            level.eps_y[id - shape.nx] * level.delta[id - shape.nx] +
            level.eps_z[id] * level.delta[id + shape.nx * shape.ny] +
            level.eps_z[id - shape.nx * shape.ny] *
                level.delta[id - shape.nx * shape.ny];
        const double j_delta = level.diag[id] * level.delta[id] - neighbor_sum;
        const double r = level.rhs[id] - j_delta;
        residual[id] = r;
        sumsq += r * r;
        ++count;
      }
    }
  }
  return count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
}

void AnalyzerImpl::restrict_newton_correction_residual(
    const NewtonCorrectionLevel &fine, const AlignedDoubleVector &fine_residual,
    NewtonCorrectionLevel &coarse) const {
  std::fill(coarse.rhs.begin(), coarse.rhs.end(), 0.0);
  std::fill(coarse.delta.begin(), coarse.delta.end(), 0.0);
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
              const int w = weights[dx + 1] * weights[dy + 1] * weights[dz + 1];
              sum +=
                  static_cast<double>(w) *
                  fine_residual[idx3(ix, iy, iz, fine.shape.nx, fine.shape.ny)];
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

void AnalyzerImpl::prolongate_add_newton_correction(
    const NewtonCorrectionLevel &coarse, NewtonCorrectionLevel &fine) const {
  const double max_correction = std::max(1.0e-12, fine.max_correction);
  const double correction_damping = 0.25;
  for (std::size_t k = 1; k + 1 < fine.shape.nz; ++k) {
    const double z = fine.shape.oz + static_cast<double>(k) * fine.shape.h;
    for (std::size_t j = 1; j + 1 < fine.shape.ny; ++j) {
      const double y = fine.shape.oy + static_cast<double>(j) * fine.shape.h;
      for (std::size_t i = 1; i + 1 < fine.shape.nx; ++i) {
        const double x = fine.shape.ox + static_cast<double>(i) * fine.shape.h;
        double correction = 0.0;
        if (interpolate_field_value(coarse.shape, coarse.delta, x, y, z,
                                    correction)) {
          const std::size_t id = idx3(i, j, k, fine.shape.nx, fine.shape.ny);
          fine.delta[id] =
              std::clamp(fine.delta[id] + correction_damping * correction,
                         -max_correction, max_correction);
        }
      }
    }
  }
}

void AnalyzerImpl::run_newton_correction_vcycle(
    std::vector<NewtonCorrectionLevel> &levels, std::size_t level_index) const {
  NewtonCorrectionLevel &level = levels[level_index];
  if (level_index + 1 >= levels.size()) {
    smooth_newton_correction_level(level, args_.mg_coarse_iter, true);
    return;
  }

  smooth_newton_correction_level(level, args_.mg_pre, false);
  AlignedDoubleVector residual;
  compute_newton_correction_level_residual(level, residual);
  NewtonCorrectionLevel &coarse = levels[level_index + 1];
  restrict_newton_correction_residual(level, residual, coarse);
  run_newton_correction_vcycle(levels, level_index + 1);
  prolongate_add_newton_correction(coarse, level);
  smooth_newton_correction_level(level, args_.mg_post, false);
}

AnalyzerImpl::NewtonCorrectionStats
AnalyzerImpl::solve_newton_correction_vcycle(
    const GridShape &shape, const AlignedDoubleVector &residual,
    double residual_rms, double alpha, double clip, double screen_scale,
    AlignedDoubleVector &delta, int max_cycles) const {
  NewtonCorrectionStats stats;
  delta.assign(shape.n, 0.0);
  if (shape.n == 0 || residual.size() != shape.n)
    return stats;

  std::vector<NewtonCorrectionLevel> levels;
  prepare_newton_correction_hierarchy(shape, residual, alpha, clip,
                                      screen_scale, levels);
  stats.inner_effective = "vcycle";
  stats.inner_levels = static_cast<int>(levels.size());

  AlignedDoubleVector correction_residual;
  stats.residual_rms = compute_newton_correction_level_residual(
      levels.front(), correction_residual);
  stats.residual_max = 0.0;
  stats.jac_diag_min = std::numeric_limits<double>::max();
  stats.jac_diag_max = 0.0;
  stats.screen_points = 0;
  stats.active_nonlinear_points = 0;
  for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
    for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
      for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        stats.residual_max =
            std::max(stats.residual_max, std::abs(correction_residual[id]));
        stats.jac_diag_min =
            std::min(stats.jac_diag_min, levels.front().diag[id]);
        stats.jac_diag_max =
            std::max(stats.jac_diag_max, levels.front().diag[id]);
        if (levels.front().screen[id] > 0.0)
          ++stats.screen_points;
        if (levels.front().screen[id] * screen_scale > 0.0)
          ++stats.active_nonlinear_points;
      }
    }
  }
  if (stats.jac_diag_min == std::numeric_limits<double>::max())
    stats.jac_diag_min = 0.0;

  const double initial_correction_residual =
      std::isfinite(stats.residual_rms) ? stats.residual_rms : residual_rms;
  stats.inner_residual_before = initial_correction_residual;
  double best_residual = stats.residual_rms;
  int stale_cycles = 0;
  const int cycles =
      max_cycles > 0 ? max_cycles : std::max(1, args_.nl_inner_iter);

  for (int cycle = 1; cycle <= cycles; ++cycle) {
    const AlignedDoubleVector before = levels.front().delta;
    run_newton_correction_vcycle(levels, 0);
    double sumsq = 0.0;
    double max_update = 0.0;
    std::size_t count = 0;
    for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
      for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
        for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
          const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
          const double d = levels.front().delta[id] - before[id];
          sumsq += d * d;
          max_update = std::max(max_update, std::abs(d));
          ++count;
        }
      }
    }
    stats.sweeps.iterations = cycle;
    stats.sweeps.rms =
        count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
    stats.sweeps.max_dphi = max_update;
    stats.residual_rms = compute_newton_correction_level_residual(
        levels.front(), correction_residual);
    stats.residual_max = 0.0;
    for (double r : correction_residual)
      stats.residual_max = std::max(stats.residual_max, std::abs(r));
    const double correction_target =
        std::max(args_.nl_residual, 0.05 * initial_correction_residual);
    if (stats.residual_rms <= correction_target) {
      stats.sweeps.converged = true;
      break;
    }
    if (std::isfinite(stats.residual_rms) &&
        stats.residual_rms < best_residual * 0.999) {
      best_residual = stats.residual_rms;
      stale_cycles = 0;
    } else if (++stale_cycles >= std::min(4, args_.nl_stagnation)) {
      break;
    }
  }

  delta = levels.front().delta;
  stats.inner_cycles = stats.sweeps.iterations;
  stats.inner_residual_after = stats.residual_rms;
  stats.residual_decreased = std::isfinite(stats.residual_rms) &&
                             stats.residual_rms < initial_correction_residual;
  return stats;
}

AnalyzerImpl::NewtonCorrectionStats AnalyzerImpl::solve_newton_correction(
    const GridShape &shape, const AlignedDoubleVector &residual,
    double residual_rms, double alpha, double clip, double screen_scale,
    AlignedDoubleVector &delta, int max_iterations,
    const std::string &inner_solver, TitanExecutor &executor) const {
  if (inner_solver == "vcycle") {
    NewtonCorrectionStats vcycle = solve_newton_correction_vcycle(
        shape, residual, residual_rms, alpha, clip, screen_scale, delta,
        max_iterations);
    if (vcycle.residual_decreased)
      return vcycle;
    NewtonCorrectionStats fallback = solve_newton_correction_sor(
        shape, residual, residual_rms, alpha, clip, screen_scale, delta,
        max_iterations, executor);
    fallback.inner_effective = "vcycle_sor_fallback";
    fallback.inner_levels = vcycle.inner_levels;
    fallback.inner_cycles = vcycle.inner_cycles;
    fallback.inner_residual_before = vcycle.inner_residual_before;
    fallback.inner_residual_after = fallback.residual_rms;
    return fallback;
  }
  return solve_newton_correction_sor(shape, residual, residual_rms, alpha, clip,
                                     screen_scale, delta, max_iterations,
                                     executor);
}

std::size_t AnalyzerImpl::update_nonlinear_source(const GridShape &shape,
                                                  AlignedDoubleVector &source,
                                                  double alpha,
                                                  double clip) const {
  if (source.size() != shape.n)
    source.assign(shape.n, 0.0);
  std::size_t clipped = 0;
  for (std::size_t id = 0; id < shape.n; ++id) {
    if (grid_.screen[id] <= 0.0) {
      source[id] = 0.0;
      continue;
    }
    const double raw_arg = nonlinear_argument_value(grid_.phi[id], alpha);
    if (raw_arg > clip) {
      ++clipped;
    } else if (raw_arg < -clip) {
      ++clipped;
    }
    source[id] = nonlinear_source_value(grid_.phi[id], alpha, clip);
  }
  return clipped;
}

double AnalyzerImpl::grid_reaction_energy_kcal(
    const GridShape &shape, const AlignedDoubleVector &solvent_phi,
    const AlignedDoubleVector &vacuum_phi) const {
  if (shape.n == 0 || solvent_phi.size() != shape.n ||
      vacuum_phi.size() != shape.n || grid_.rho.size() != shape.n)
    return 0.0;
  double sum = 0.0;
  for (std::size_t id = 0; id < shape.n; ++id)
    sum += grid_.rho[id] * (solvent_phi[id] - vacuum_phi[id]);
  return 0.5 * kCoulombKcalPerMol * field_to_native_factor() * sum;
}

double
AnalyzerImpl::grid_phase_energy_kcal(const GridShape &shape,
                                     const AlignedDoubleVector &field) const {
  return grid_phase_energy_kcal(shape, field, grid_.rho);
}

double
AnalyzerImpl::grid_phase_energy_kcal(const GridShape &shape,
                                     const AlignedDoubleVector &field,
                                     const AlignedDoubleVector &rho) const {
  if (shape.n == 0 || field.size() != shape.n || rho.size() != shape.n)
    return 0.0;
  double sum = 0.0;
  for (std::size_t id = 0; id < shape.n; ++id)
    sum += rho[id] * field[id];
  return 0.5 * kCoulombKcalPerMol * field_to_native_factor() * sum;
}

double AnalyzerImpl::atom_phase_energy_kcal(
    const std::vector<Atom> &atoms, const std::vector<double> &atom_phi) const {
  if (atoms.size() != atom_phi.size())
    return 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < atoms.size(); ++i)
    sum += atoms[i].q * atom_phi[i];
  return 0.5 * kCoulombKcalPerMol * field_to_native_factor() * sum;
}

double
AnalyzerImpl::analytic_coulomb_energy_kcal(const std::vector<Atom> &atoms,
                                           double eps) const {
  const double dielectric = std::max(1.0e-12, eps);
  double energy = 0.0;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    for (std::size_t j = i + 1; j < atoms.size(); ++j) {
      const double dx = atoms[i].x - atoms[j].x;
      const double dy = atoms[i].y - atoms[j].y;
      const double dz = atoms[i].z - atoms[j].z;
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r > 1.0e-12)
        energy +=
            kCoulombKcalPerMol * atoms[i].q * atoms[j].q / (dielectric * r);
    }
  }
  return energy;
}

double AnalyzerImpl::effective_nl_relax() const noexcept {
  if (args_.nl_relax == "auto")
    return 0.5;
  return std::clamp(args_.nl_relax_value, 0.05, 1.0);
}

double AnalyzerImpl::nonlinear_alpha() const noexcept {
  return kCoulombKcalPerMol / (kKBKcalPerMolK * std::max(1.0, args_.temp));
}

double AnalyzerImpl::effective_nonlinear_clip() const noexcept {
  if (delphi_nwt_parity_active() && delphi_field_units_active())
    return 50.0;
  return std::max(1.0e-12, args_.nl_clip);
}

double AnalyzerImpl::rhs_scale(const GridShape &shape) const noexcept {
  return 4.0 * kPi / std::max(1.0e-12, shape.h);
}

bool AnalyzerImpl::nonlinear_dimensionless_mode() const noexcept {
  return args_.test_nl_source == "delphi-dimensionless";
}

double AnalyzerImpl::nonlinear_rhs_scale(const GridShape &shape,
                                         double alpha) const noexcept {
  const double base = effective_rhs_scale(shape);
  if (delphi_field_units_active())
    return base;
  return nonlinear_dimensionless_mode() ? alpha * base : base;
}

double AnalyzerImpl::nonlinear_argument_value(double field_value,
                                              double alpha) const noexcept {
  if (delphi_field_units_active())
    return field_value;
  return nonlinear_dimensionless_mode() ? field_value : alpha * field_value;
}

double AnalyzerImpl::nonlinear_source_value(double phi, double alpha,
                                            double clip) const noexcept {
  if (alpha <= 0.0)
    return 0.0;
  const double arg = nonlinear_argument_value(phi, alpha);
  const auto [sinh_arg, unused_cosh] = delphi_sinh_cosh_tail(arg, clip);
  (void)unused_cosh;
  if (delphi_field_units_active() ||
      args_.test_nl_source == "delphi-dimensionless")
    return sinh_arg;
  return sinh_arg / alpha;
}

double AnalyzerImpl::nonlinear_source_derivative(double phi, double alpha,
                                                 double clip) const noexcept {
  if (alpha <= 0.0)
    return 0.0;
  const double arg = nonlinear_argument_value(phi, alpha);
  const auto [unused_sinh, cosh_arg] = delphi_sinh_cosh_tail(arg, clip);
  (void)unused_sinh;
  return cosh_arg;
}

void AnalyzerImpl::scale_nonlinear_field_to_solve_units(
    AlignedDoubleVector &field, double alpha) const {
  if (!nonlinear_dimensionless_mode() || delphi_field_units_active() ||
      alpha <= 0.0)
    return;
  for (double &value : field)
    value *= alpha;
}

void AnalyzerImpl::scale_nonlinear_field_to_physical_units(
    AlignedDoubleVector &field, double alpha) const {
  if (!nonlinear_dimensionless_mode() || delphi_field_units_active() ||
      alpha <= 0.0)
    return;
  const double inv_alpha = 1.0 / alpha;
  for (double &value : field)
    value *= inv_alpha;
}


AnalyzerImpl::SolveStats AnalyzerImpl::solve_nonlinear_sor_block(
    const GridShape &shape, const AlignedDoubleVector &source, int sweeps,
    double relax, double screen_scale, TitanExecutor &executor,
    std::size_t &clipped_points) {
  SolveStats stats;
  const double omega =
      args_.omega == "auto"
          ? std::clamp(2.0 / (1.0 + std::sin(kPi / static_cast<double>(std::max(
                                                       {shape.nx, shape.ny,
                                                        shape.nz})))),
                       1.0, 1.95)
          : args_.omega_value;
  const double effective_omega = std::clamp(omega * relax, 0.05, 1.95);
  const std::size_t interior_z = shape.nz > 2 ? shape.nz - 2 : 0;
  const std::size_t workers = effective_workers(executor, interior_z);
  const double rhs = nonlinear_rhs_scale(shape, nonlinear_alpha());

  std::vector<double> local_sumsq(workers, 0.0);
  std::vector<double> local_max(workers, 0.0);
  std::vector<std::size_t> local_count(workers, 0);
  clipped_points = 0;

  for (int iter = 1; iter <= sweeps; ++iter) {
    std::fill(local_sumsq.begin(), local_sumsq.end(), 0.0);
    std::fill(local_max.begin(), local_max.end(), 0.0);
    std::fill(local_count.begin(), local_count.end(), 0);

    for (int color = 0; color < 2; ++color) {
      executor.parallel_for(
          interior_z, workers, 1,
          [&](std::size_t worker, std::size_t begin, std::size_t end) {
            double sumsq = 0.0;
            double max_delta = 0.0;
            std::size_t count = 0;
            for (std::size_t kk = begin; kk < end; ++kk) {
              const std::size_t k = kk + 1;
              for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
                const std::size_t first_i =
                    1 +
                    (static_cast<int>((1 + j + k) & 1ULL) == color ? 0U : 1U);
                for (std::size_t i = first_i; i + 1 < shape.nx; i += 2) {
                  const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
                  const double neighbor_sum =
                      grid_.eps_x[id] * grid_.phi[id + 1] +
                      grid_.eps_x[id - 1] * grid_.phi[id - 1] +
                      grid_.eps_y[id] * grid_.phi[id + shape.nx] +
                      grid_.eps_y[id - shape.nx] * grid_.phi[id - shape.nx] +
                      grid_.eps_z[id] * grid_.phi[id + shape.nx * shape.ny] +
                      grid_.eps_z[id - shape.nx * shape.ny] *
                          grid_.phi[id - shape.nx * shape.ny];
                  const double eps_sum =
                      std::max(1.0e-12, grid_.denom[id] - grid_.screen[id]);
                  const double target =
                      (neighbor_sum + rhs * grid_.rho[id] -
                       (grid_.screen[id] * screen_scale) * source[id]) /
                      eps_sum;
                  const double updated =
                      grid_.phi[id] +
                      effective_omega * (target - grid_.phi[id]);
                  const double delta = updated - grid_.phi[id];
                  grid_.phi[id] = updated;
                  const double abs_delta = std::abs(delta);
                  sumsq += delta * delta;
                  max_delta = std::max(max_delta, abs_delta);
                  ++count;
                }
              }
            }
            local_sumsq[worker] += sumsq;
            local_max[worker] = std::max(local_max[worker], max_delta);
            local_count[worker] += count;
          });
    }

    double sumsq = 0.0;
    double max_delta = 0.0;
    std::size_t count = 0;
    for (std::size_t worker = 0; worker < workers; ++worker) {
      sumsq += local_sumsq[worker];
      max_delta = std::max(max_delta, local_max[worker]);
      count += local_count[worker];
    }
    stats.iterations = iter;
    stats.rms = count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
    stats.max_dphi = max_delta;
  }
  stats.converged = stats.max_dphi <= args_.nl_dphi;
  return stats;
}

bool AnalyzerImpl::nwt_half_shape_matches(
    const GridShape &shape) const noexcept {
  const GridShape &cached = nwt_half_workspace_.shape;
  return nwt_half_workspace_.active && cached.n == shape.n &&
         cached.nx == shape.nx && cached.ny == shape.ny &&
         cached.nz == shape.nz;
}

void AnalyzerImpl::invalidate_nwt_half_workspace() noexcept {
  nwt_half_workspace_.active = false;
  nwt_half_workspace_.coeff_ready = false;
  nwt_half_workspace_.packed_ready = false;
  nwt_half_workspace_.row_span_ready = false;
  nwt_half_workspace_.salt_tile_ready = false;
  nwt_half_workspace_.rhs_scale = 0.0;
}

void AnalyzerImpl::prepare_nwt_half_workspace(const GridShape &shape,
                                              double rhs_scale) {
  const std::size_t even_count = (shape.n + 1) / 2;
  const std::size_t odd_count = shape.n / 2;
  const bool shape_match = nwt_half_workspace_.shape.n == shape.n &&
                           nwt_half_workspace_.shape.nx == shape.nx &&
                           nwt_half_workspace_.shape.ny == shape.ny &&
                           nwt_half_workspace_.shape.nz == shape.nz;
  if (!shape_match || nwt_half_workspace_.even_phi.size() != even_count ||
      nwt_half_workspace_.odd_phi.size() != odd_count) {
    nwt_half_workspace_.shape = shape;
    nwt_half_workspace_.even_phi.assign(even_count, 0.0);
    nwt_half_workspace_.odd_phi.assign(odd_count, 0.0);
    nwt_half_workspace_.rhs_even.assign(even_count, 0.0);
    nwt_half_workspace_.rhs_odd.assign(odd_count, 0.0);
    nwt_half_workspace_.eps_sum_even.assign(even_count, 1.0);
    nwt_half_workspace_.eps_sum_odd.assign(odd_count, 1.0);
    nwt_half_workspace_.screen_even.assign(even_count, 0.0);
    nwt_half_workspace_.screen_odd.assign(odd_count, 0.0);
    nwt_half_workspace_.homo_eps_even.assign(even_count, 0.0);
    nwt_half_workspace_.homo_eps_odd.assign(odd_count, 0.0);
    nwt_half_workspace_.coeff_ready = false;
    nwt_half_workspace_.packed_ready = false;
    nwt_half_workspace_.row_span_ready = false;
    nwt_half_workspace_.salt_tile_ready = false;
  }
  if (!nwt_half_workspace_.active) {
    for (std::size_t id = 0; id < shape.n; ++id) {
      if ((id & 1U) == 0U)
        nwt_half_workspace_.even_phi[id / 2] = grid_.phi[id];
      else
        nwt_half_workspace_.odd_phi[id / 2] = grid_.phi[id];
    }
    nwt_half_workspace_.active = true;
  }
  if (!nwt_half_workspace_.coeff_ready ||
      std::abs(nwt_half_workspace_.rhs_scale - rhs_scale) > 1.0e-30) {
    const auto pack_begin = std::chrono::steady_clock::now();
    for (std::size_t id = 0; id < shape.n; ++id) {
      const std::size_t h = id / 2;
      const double eps_sum =
          std::max(1.0e-12, grid_.denom[id] - grid_.screen[id]);
      if ((id & 1U) == 0U) {
        nwt_half_workspace_.rhs_even[h] = rhs_scale * grid_.rho[id];
        nwt_half_workspace_.eps_sum_even[h] = eps_sum;
        nwt_half_workspace_.screen_even[h] = grid_.screen[id];
      } else {
        nwt_half_workspace_.rhs_odd[h] = rhs_scale * grid_.rho[id];
        nwt_half_workspace_.eps_sum_odd[h] = eps_sum;
        nwt_half_workspace_.screen_odd[h] = grid_.screen[id];
      }
    }

    nwt_half_workspace_.even_homo_cells.clear();
    nwt_half_workspace_.even_hetero_cells.clear();
    nwt_half_workspace_.odd_homo_cells.clear();
    nwt_half_workspace_.odd_hetero_cells.clear();
    nwt_half_workspace_.even_homo_linear_spans.clear();
    nwt_half_workspace_.even_homo_salt_spans.clear();
    nwt_half_workspace_.odd_homo_linear_spans.clear();
    nwt_half_workspace_.odd_homo_salt_spans.clear();
    nwt_half_workspace_.even_homo_linear_tiles.clear();
    nwt_half_workspace_.even_homo_salt_tiles.clear();
    nwt_half_workspace_.odd_homo_linear_tiles.clear();
    nwt_half_workspace_.odd_homo_salt_tiles.clear();
    nwt_half_workspace_.homo_cells = 0;
    nwt_half_workspace_.homo_linear_cells = 0;
    nwt_half_workspace_.homo_salt_cells = 0;
    if (shape.n <=
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      const std::size_t xy = shape.nx * shape.ny;
      const std::size_t reserve_each =
          shape.n > 0 ? std::max<std::size_t>(1, shape.n / 4) : 0;
      nwt_half_workspace_.even_homo_cells.reserve(reserve_each);
      nwt_half_workspace_.even_hetero_cells.reserve(reserve_each);
      nwt_half_workspace_.odd_homo_cells.reserve(reserve_each);
      nwt_half_workspace_.odd_hetero_cells.reserve(reserve_each);
      const auto append_span =
          [](std::vector<NwtHalfWorkspace::HomoSpan> &spans, std::uint32_t h) {
            if (!spans.empty()) {
              auto &last = spans.back();
              if (last.begin + last.count == h) {
                ++last.count;
                return;
              }
            }
            spans.push_back({h, 1});
          };
      auto add_cell = [&](int color, std::size_t id) {
        const double ex_p = grid_.eps_x[id];
        const double ex_m = grid_.eps_x[id - 1];
        const double ey_p = grid_.eps_y[id];
        const double ey_m = grid_.eps_y[id - shape.nx];
        const double ez_p = grid_.eps_z[id];
        const double ez_m = grid_.eps_z[id - xy];
        NwtHalfWorkspace::PackedCell cell{};
        cell.h = static_cast<std::uint32_t>(id / 2);
        cell.xp = static_cast<std::uint32_t>((id + 1) / 2);
        cell.xm = static_cast<std::uint32_t>((id - 1) / 2);
        cell.yp = static_cast<std::uint32_t>((id + shape.nx) / 2);
        cell.ym = static_cast<std::uint32_t>((id - shape.nx) / 2);
        cell.zp = static_cast<std::uint32_t>((id + xy) / 2);
        cell.zm = static_cast<std::uint32_t>((id - xy) / 2);
        cell.rhs = rhs_scale * grid_.rho[id];
        cell.eps_sum = std::max(1.0e-12, grid_.denom[id] - grid_.screen[id]);
        cell.screen = grid_.screen[id];
        cell.ex_p = ex_p;
        cell.ex_m = ex_m;
        cell.ey_p = ey_p;
        cell.ey_m = ey_m;
        cell.ez_p = ez_p;
        cell.ez_m = ez_m;
        const bool homogeneous = ex_p == ex_m && ex_p == ey_p && ex_p == ey_m &&
                                 ex_p == ez_p && ex_p == ez_m;
        if (color == 0) {
          (homogeneous ? nwt_half_workspace_.even_homo_cells
                       : nwt_half_workspace_.even_hetero_cells)
              .push_back(cell);
          if (homogeneous) {
            nwt_half_workspace_.homo_eps_even[cell.h] = ex_p;
            if (cell.screen > 0.0) {
              append_span(nwt_half_workspace_.even_homo_salt_spans, cell.h);
              ++nwt_half_workspace_.homo_salt_cells;
            } else {
              append_span(nwt_half_workspace_.even_homo_linear_spans, cell.h);
              ++nwt_half_workspace_.homo_linear_cells;
            }
            ++nwt_half_workspace_.homo_cells;
          }
        } else {
          (homogeneous ? nwt_half_workspace_.odd_homo_cells
                       : nwt_half_workspace_.odd_hetero_cells)
              .push_back(cell);
          if (homogeneous) {
            nwt_half_workspace_.homo_eps_odd[cell.h] = ex_p;
            if (cell.screen > 0.0) {
              append_span(nwt_half_workspace_.odd_homo_salt_spans, cell.h);
              ++nwt_half_workspace_.homo_salt_cells;
            } else {
              append_span(nwt_half_workspace_.odd_homo_linear_spans, cell.h);
              ++nwt_half_workspace_.homo_linear_cells;
            }
            ++nwt_half_workspace_.homo_cells;
          }
        }
      };
      for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
        for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
          for (int color = 0; color < 2; ++color) {
            const std::size_t first_i =
                1 + (static_cast<int>((1 + j + k) & 1ULL) == color ? 0U : 1U);
            for (std::size_t i = first_i; i + 1 < shape.nx; i += 2) {
              add_cell(color, idx3(i, j, k, shape.nx, shape.ny));
            }
          }
        }
      }
      const auto build_tiles =
          [](const std::vector<NwtHalfWorkspace::HomoSpan> &spans,
             std::vector<NwtHalfWorkspace::HomoTile> &tiles) {
            constexpr std::uint32_t kTargetTileCells = 2048;
            tiles.clear();
            for (const auto &span : spans) {
              std::uint32_t begin = span.begin;
              std::uint32_t remaining = span.count;
              while (remaining > 0) {
                const std::uint32_t take =
                    std::min<std::uint32_t>(remaining, kTargetTileCells);
                if (!tiles.empty()) {
                  auto &last = tiles.back();
                  const bool contiguous = last.begin + last.count == begin;
                  const bool has_room = last.count < kTargetTileCells &&
                                        take <= kTargetTileCells - last.count;
                  if (contiguous && has_room) {
                    last.count += take;
                  } else {
                    tiles.push_back({begin, take});
                  }
                } else {
                  tiles.push_back({begin, take});
                }
                begin += take;
                remaining -= take;
              }
            }
          };
      build_tiles(nwt_half_workspace_.even_homo_linear_spans,
                  nwt_half_workspace_.even_homo_linear_tiles);
      build_tiles(nwt_half_workspace_.even_homo_salt_spans,
                  nwt_half_workspace_.even_homo_salt_tiles);
      build_tiles(nwt_half_workspace_.odd_homo_linear_spans,
                  nwt_half_workspace_.odd_homo_linear_tiles);
      build_tiles(nwt_half_workspace_.odd_homo_salt_spans,
                  nwt_half_workspace_.odd_homo_salt_tiles);
      nwt_half_workspace_.packed_ready = true;
      nwt_half_workspace_.row_span_ready = true;
      nwt_half_workspace_.salt_tile_ready = true;
    } else {
      nwt_half_workspace_.packed_ready = false;
      nwt_half_workspace_.row_span_ready = false;
      nwt_half_workspace_.salt_tile_ready = false;
    }
    nwt_half_workspace_.rhs_scale = rhs_scale;
    nwt_half_workspace_.coeff_ready = true;
    add_timing_ms(TimingBucket::NwtCoeffPack,
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - pack_begin)
                      .count());
  }
}

void AnalyzerImpl::flush_nwt_half_workspace(const GridShape &shape) {
  if (!nwt_half_shape_matches(shape))
    return;
  for (std::size_t id = 0; id < shape.n; ++id) {
    grid_.phi[id] = (id & 1U) == 0U ? nwt_half_workspace_.even_phi[id / 2]
                                    : nwt_half_workspace_.odd_phi[id / 2];
  }
  nwt_half_workspace_.active = false;
}

AnalyzerImpl::SolveStats AnalyzerImpl::solve_newton_nwt_sor_block(
    const GridShape &shape, int sweeps, double screen_scale,
    TitanExecutor &executor, std::size_t &clipped_points) {
  const auto update_time_begin = std::chrono::steady_clock::now();
  SolveStats stats;
  if (shape.n == 0 || sweeps <= 0)
    return stats;

  double omega =
      args_.omega == "auto"
          ? std::clamp(2.0 / (1.0 + std::sin(kPi / static_cast<double>(std::max(
                                                       {shape.nx, shape.ny,
                                                        shape.nz})))),
                       1.0, 1.95)
          : args_.omega_value;
  if (args_.omega == "auto" && args_.test_operator_convention == "delphi" &&
      !args_.test_pqr.empty()) {
    omega = delphi_relaxation_omega(shape);
  }
  if (args_.omega == "auto" && delphi_nwt_parity_active()) {
    // pyDelPhi NWT defaults to nwt_adaptive_omega=1.0; keep this parity path
    // separate from the linear DelPhi SOR relaxation convention.
    omega = 1.0;
  }

  const std::size_t interior_z = shape.nz > 2 ? shape.nz - 2 : 0;
  const std::size_t workers = effective_workers(executor, interior_z);
  const double rhs = nonlinear_rhs_scale(shape, nonlinear_alpha());
  const double alpha = nonlinear_alpha();
  const double clip = effective_nonlinear_clip();
  const std::size_t xy = shape.nx * shape.ny;
  std::vector<double> local_sumsq(workers, 0.0);
  std::vector<double> local_max(workers, 0.0);
  std::vector<std::size_t> local_count(workers, 0);
  std::vector<std::size_t> local_clipped(workers, 0);
  clipped_points = 0;
  const bool delphi_block_stop = delphi_nwt_parity_active();
  const std::size_t delphi_rms_denominator =
      delphi_block_stop ? (shape.n + 1) / 2 : 0;
  const bool write_trace = !args_.test_pqr.empty() &&
                           args_.parity_dump == "all" &&
                           delphi_nwt_parity_active();
  const bool use_auto_half_kernel = args_.nwt_kernel == "auto" &&
                                    args_.nwt_progress == "off" &&
                                    delphi_nwt_parity_active();
  const bool use_forced_half_kernel = args_.nwt_kernel == "delphi-half";
  if ((use_auto_half_kernel || use_forced_half_kernel) &&
      args_.nwt_progress == "off" && delphi_nwt_parity_active() &&
      !write_trace && (shape.nx & 1U) != 0U && (shape.ny & 1U) != 0U &&
      (shape.nz & 1U) != 0U) {
    return solve_newton_nwt_delphi_half_block(shape, sweeps, screen_scale,
                                              executor, clipped_points);
  }

  std::vector<std::size_t> trace_ids;
  if (write_trace && shape.nx > 2 && shape.ny > 2 && shape.nz > 2) {
    trace_ids = {
        idx3(shape.nx / 2, shape.ny / 2, shape.nz / 2, shape.nx, shape.ny),
        idx3(shape.nx / 2 - 1, shape.ny / 2, shape.nz / 2, shape.nx, shape.ny),
        idx3(shape.nx / 2, shape.ny / 2 - 1, shape.nz / 2, shape.nx, shape.ny),
        idx3(shape.nx / 2, shape.ny / 2, shape.nz / 2 - 1, shape.nx, shape.ny),
        idx3(1, 1, 1, shape.nx, shape.ny),
        idx3(shape.nx - 2, shape.ny - 2, shape.nz - 2, shape.nx, shape.ny),
        idx3(1, shape.ny / 2, shape.nz / 2, shape.nx, shape.ny),
        idx3(shape.nx - 2, shape.ny / 2, shape.nz / 2, shape.nx, shape.ny),
        idx3(shape.nx / 2, shape.ny / 2, 1, shape.nx, shape.ny),
        idx3(shape.nx / 2, shape.ny / 2, shape.nz - 2, shape.nx, shape.ny)};
    for (std::size_t id = 0; id < grid_.rho.size(); ++id) {
      if (std::abs(grid_.rho[id]) > 0.0) {
        trace_ids.push_back(id);
      }
    }
    std::sort(trace_ids.begin(), trace_ids.end());
    trace_ids.erase(std::unique(trace_ids.begin(), trace_ids.end()),
                    trace_ids.end());
  }
  std::ofstream trace_out;
  if (write_trace) {
    if (!nwt_trace_header_written_) {
      trace_out.open("pbe_nwt_trace.csv", std::ios::out | std::ios::trunc);
      if (!trace_out.is_open())
        throw std::runtime_error("PBE: failed to open pbe_nwt_trace.csv.");
      trace_out << "iteration,color,i,j,k,omega,old_phi,neighbor_sum,"
                   "charge_density,eps_sum,screen,sinh_phi,cosh_phi,"
                   "linearized_rhs,denom,target_phi,new_phi,dphi,rms,max_dphi,"
                   "field_units,operator_convention,charge_grid_source\n";
      nwt_trace_header_written_ = true;
    } else {
      trace_out.open("pbe_nwt_trace.csv", std::ios::out | std::ios::app);
      if (!trace_out.is_open())
        throw std::runtime_error("PBE: failed to append pbe_nwt_trace.csv.");
    }
    trace_out << std::setprecision(12);
  }

  for (int iter = 1; iter <= sweeps; ++iter) {
    std::fill(local_sumsq.begin(), local_sumsq.end(), 0.0);
    std::fill(local_max.begin(), local_max.end(), 0.0);
    std::fill(local_count.begin(), local_count.end(), 0);
    std::fill(local_clipped.begin(), local_clipped.end(), 0);

    for (int color = 0; color < 2; ++color) {
      std::vector<std::vector<std::string>> trace_lines;
      if (write_trace && iter <= 4)
        trace_lines.resize(workers);
      executor.parallel_for(
          interior_z, workers, 1,
          [&](std::size_t worker, std::size_t begin, std::size_t end) {
            double sumsq = 0.0;
            double max_delta = 0.0;
            std::size_t count = 0;
            std::size_t clipped = 0;
            for (std::size_t kk = begin; kk < end; ++kk) {
              const std::size_t k = kk + 1;
              for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
                const std::size_t first_i =
                    1 +
                    (static_cast<int>((1 + j + k) & 1ULL) == color ? 0U : 1U);
                for (std::size_t i = first_i; i + 1 < shape.nx; i += 2) {
                  const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
                  const double neighbor_sum =
                      grid_.eps_x[id] * grid_.phi[id + 1] +
                      grid_.eps_x[id - 1] * grid_.phi[id - 1] +
                      grid_.eps_y[id] * grid_.phi[id + shape.nx] +
                      grid_.eps_y[id - shape.nx] * grid_.phi[id - shape.nx] +
                      grid_.eps_z[id] * grid_.phi[id + xy] +
                      grid_.eps_z[id - xy] * grid_.phi[id - xy];
                  const double eps_sum =
                      std::max(1.0e-12, grid_.denom[id] - grid_.screen[id]);
                  const double active_screen = grid_.screen[id] * screen_scale;
                  double diagonal = eps_sum;
                  double nonlinear_rhs = 0.0;
                  double sinh_arg = 0.0;
                  double cosh_arg = 1.0;
                  if (active_screen > 0.0) {
                    const double raw_arg =
                        nonlinear_argument_value(grid_.phi[id], alpha);
                    if (raw_arg > clip || raw_arg < -clip)
                      ++clipped;
                    std::tie(sinh_arg, cosh_arg) =
                        delphi_sinh_cosh_tail(raw_arg, clip);
                    const double source =
                        (delphi_field_units_active() ||
                         args_.test_nl_source == "delphi-dimensionless")
                            ? sinh_arg
                            : sinh_arg / std::max(1.0e-30, alpha);
                    diagonal += active_screen * cosh_arg;
                    nonlinear_rhs =
                        active_screen * (cosh_arg * grid_.phi[id] - source);
                  }
                  const double target =
                      (neighbor_sum + rhs * grid_.rho[id] + nonlinear_rhs) /
                      std::max(1.0e-12, diagonal);
                  const double old_phi = grid_.phi[id];
                  const double updated = old_phi + omega * (target - old_phi);
                  const double delta = updated - old_phi;
                  grid_.phi[id] = updated;
                  if (write_trace && iter <= 4 &&
                      std::binary_search(trace_ids.begin(), trace_ids.end(),
                                         id)) {
                    const double charge_density = rhs * grid_.rho[id];
                    std::ostringstream row;
                    row << std::setprecision(12) << iter << ',' << color << ','
                        << i << ',' << j << ',' << k << ',' << omega << ','
                        << old_phi << ',' << neighbor_sum << ','
                        << charge_density << ',' << eps_sum << ','
                        << active_screen << ',' << sinh_arg << ',' << cosh_arg
                        << ',' << nonlinear_rhs << ','
                        << std::max(1.0e-12, diagonal) << ',' << target << ','
                        << updated << ',' << delta << ",0,0,"
                        << csv_escape(delphi_field_units_active()
                                          ? "delphi-dimensionless"
                                          : args_.test_field_units)
                        << ',' << csv_escape("delphi_nwt") << ','
                        << csv_escape(args_.test_charge_grid) << '\n';
                    trace_lines[worker].push_back(row.str());
                  }
                  if (!delphi_block_stop || (iter == sweeps && color == 1)) {
                    sumsq += delta * delta;
                    max_delta = std::max(max_delta, std::abs(delta));
                    ++count;
                  }
                }
              }
            }
            local_sumsq[worker] += sumsq;
            local_max[worker] = std::max(local_max[worker], max_delta);
            local_count[worker] += count;
            local_clipped[worker] += clipped;
          });
      if (write_trace && iter <= 4) {
        for (const auto &worker_lines : trace_lines) {
          for (const std::string &row : worker_lines)
            trace_out << row;
        }
      }
    }

    const auto reduce_time_begin = std::chrono::steady_clock::now();
    double sumsq = 0.0;
    double max_delta = 0.0;
    std::size_t count = 0;
    std::size_t clipped = 0;
    for (std::size_t worker = 0; worker < workers; ++worker) {
      sumsq += local_sumsq[worker];
      max_delta = std::max(max_delta, local_max[worker]);
      count += local_count[worker];
      clipped += local_clipped[worker];
    }
    clipped_points = std::max(clipped_points, clipped);
    stats.iterations = iter;
    if (delphi_block_stop) {
      stats.rms =
          delphi_rms_denominator > 0
              ? std::sqrt(sumsq / static_cast<double>(delphi_rms_denominator))
              : 0.0;
    } else {
      stats.rms =
          count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
    }
    stats.max_dphi = max_delta;
    if (write_trace && iter <= 4) {
      trace_out << iter << ",summary,0,0,0," << omega
                << ",0,0,0,0,0,0,1,0,0,0,0,0," << stats.rms << ','
                << stats.max_dphi << ','
                << csv_escape(delphi_field_units_active()
                                  ? "delphi-dimensionless"
                                  : args_.test_field_units)
                << ',' << csv_escape("delphi_nwt") << ','
                << csv_escape(args_.test_charge_grid) << '\n';
    }
    add_timing_ms(TimingBucket::NwtRmsReduce,
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - reduce_time_begin)
                      .count());
    if (!delphi_block_stop &&
        (stats.max_dphi <= args_.nl_dphi || stats.rms <= args_.nl_residual)) {
      stats.converged = true;
      break;
    }
  }
  if (delphi_block_stop &&
      (stats.max_dphi < args_.nl_dphi || stats.rms < args_.nl_residual))
    stats.converged = true;
  add_timing_ms(TimingBucket::NwtHalfUpdate,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - update_time_begin)
                    .count());
  return stats;
}

AnalyzerImpl::SolveStats AnalyzerImpl::solve_newton_nwt_delphi_half_block(
    const GridShape &shape, int sweeps, double screen_scale,
    TitanExecutor &executor, std::size_t &clipped_points) {
  const auto update_time_begin = std::chrono::steady_clock::now();
  SolveStats stats;
  if (shape.n == 0 || sweeps <= 0)
    return stats;

  double omega =
      args_.omega == "auto"
          ? std::clamp(2.0 / (1.0 + std::sin(kPi / static_cast<double>(std::max(
                                                       {shape.nx, shape.ny,
                                                        shape.nz})))),
                       1.0, 1.95)
          : args_.omega_value;
  if (args_.omega == "auto" && args_.test_operator_convention == "delphi" &&
      !args_.test_pqr.empty()) {
    omega = delphi_relaxation_omega(shape);
  }
  if (args_.omega == "auto" && delphi_nwt_parity_active())
    omega = 1.0;

  const std::size_t interior_z = shape.nz > 2 ? shape.nz - 2 : 0;
  const std::size_t workers = effective_workers(executor, interior_z);
  const double rhs = nonlinear_rhs_scale(shape, nonlinear_alpha());
  prepare_nwt_half_workspace(shape, rhs);
  const double alpha = nonlinear_alpha();
  const double clip = effective_nonlinear_clip();
  const std::size_t xy = shape.nx * shape.ny;
  const bool delphi_block_stop = delphi_nwt_parity_active();
  const std::size_t delphi_rms_denominator =
      delphi_block_stop ? (shape.n + 1) / 2 : 0;
  if (nwt_half_workspace_.local_sumsq.size() != workers) {
    nwt_half_workspace_.local_sumsq.assign(workers, 0.0);
    nwt_half_workspace_.local_max.assign(workers, 0.0);
    nwt_half_workspace_.local_count.assign(workers, 0);
    nwt_half_workspace_.local_clipped.assign(workers, 0);
  }
  auto &local_sumsq = nwt_half_workspace_.local_sumsq;
  auto &local_max = nwt_half_workspace_.local_max;
  auto &local_count = nwt_half_workspace_.local_count;
  auto &local_clipped = nwt_half_workspace_.local_clipped;
  auto &even_phi = nwt_half_workspace_.even_phi;
  auto &odd_phi = nwt_half_workspace_.odd_phi;
  const auto &rhs_even = nwt_half_workspace_.rhs_even;
  const auto &rhs_odd = nwt_half_workspace_.rhs_odd;
  const auto &eps_sum_even = nwt_half_workspace_.eps_sum_even;
  const auto &eps_sum_odd = nwt_half_workspace_.eps_sum_odd;
  const auto &screen_even = nwt_half_workspace_.screen_even;
  const auto &screen_odd = nwt_half_workspace_.screen_odd;
  const auto &homo_eps_even = nwt_half_workspace_.homo_eps_even;
  const auto &homo_eps_odd = nwt_half_workspace_.homo_eps_odd;
  clipped_points = 0;
  if (active_timing_) {
    active_timing_->nwt_homo_cells = nwt_half_workspace_.homo_cells;
    active_timing_->nwt_homo_linear_cells =
        nwt_half_workspace_.homo_linear_cells;
    active_timing_->nwt_homo_salt_cells = nwt_half_workspace_.homo_salt_cells;
  }
  const bool argument_is_dimensionless =
      delphi_field_units_active() || nonlinear_dimensionless_mode();
  const double argument_scale = argument_is_dimensionless ? 1.0 : alpha;
  const bool use_delphi_source_units =
      delphi_field_units_active() ||
      args_.test_nl_source == "delphi-dimensionless";
  const double source_scale =
      use_delphi_source_units ? 1.0 : 1.0 / std::max(1.0e-30, alpha);
  const double effective_clip = std::max(1.0e-12, clip);
  const double sinh_clip = std::sinh(effective_clip);

  if (nwt_half_workspace_.packed_ready) {
    const auto update_cells =
        [&](const std::vector<NwtHalfWorkspace::PackedCell> &cells,
            AlignedDoubleVector &write_phi, const AlignedDoubleVector &read_phi,
            bool homogeneous, bool accumulate_rms) {
          if (cells.empty())
            return;
          executor.parallel_for(
              cells.size(), workers, 256,
              [&](std::size_t worker, std::size_t begin, std::size_t end) {
                double sumsq = 0.0;
                double max_delta = 0.0;
                std::size_t count = 0;
                std::size_t clipped = 0;
                for (std::size_t cell_index = begin; cell_index < end;
                     ++cell_index) {
                  const auto &cell = cells[cell_index];
                  const double old_phi = write_phi[cell.h];
                  double neighbor_sum = 0.0;
                  if (homogeneous) {
                    neighbor_sum =
                        cell.ex_p * (read_phi[cell.xp] + read_phi[cell.xm] +
                                     read_phi[cell.yp] + read_phi[cell.ym] +
                                     read_phi[cell.zp] + read_phi[cell.zm]);
                  } else {
                    neighbor_sum = cell.ex_p * read_phi[cell.xp] +
                                   cell.ex_m * read_phi[cell.xm] +
                                   cell.ey_p * read_phi[cell.yp] +
                                   cell.ey_m * read_phi[cell.ym] +
                                   cell.ez_p * read_phi[cell.zp] +
                                   cell.ez_m * read_phi[cell.zm];
                  }
                  const double active_screen = cell.screen * screen_scale;
                  double diagonal = cell.eps_sum;
                  double nonlinear_rhs = 0.0;
                  if (active_screen > 0.0) {
                    const double raw_arg = old_phi * argument_scale;
                    if (raw_arg > effective_clip || raw_arg < -effective_clip)
                      ++clipped;
                    const auto sinh_cosh = delphi_sinh_cosh_tail_precomputed(
                        raw_arg, effective_clip, sinh_clip);
                    const double source = sinh_cosh.first * source_scale;
                    diagonal += active_screen * sinh_cosh.second;
                    nonlinear_rhs =
                        active_screen * (sinh_cosh.second * old_phi - source);
                  }
                  const double target =
                      (neighbor_sum + cell.rhs + nonlinear_rhs) /
                      std::max(1.0e-12, diagonal);
                  const double updated = old_phi + omega * (target - old_phi);
                  const double delta = updated - old_phi;
                  write_phi[cell.h] = updated;
                  if (accumulate_rms) {
                    sumsq += delta * delta;
                    max_delta = std::max(max_delta, std::abs(delta));
                    ++count;
                  }
                }
                local_sumsq[worker] += sumsq;
                local_max[worker] = std::max(local_max[worker], max_delta);
                local_count[worker] += count;
                local_clipped[worker] += clipped;
              });
        };

    auto run_timed_cells =
        [&](const std::vector<NwtHalfWorkspace::PackedCell> &cells,
            AlignedDoubleVector &write_phi, const AlignedDoubleVector &read_phi,
            bool homogeneous, bool accumulate_rms, TimingBucket bucket) {
          const auto begin = std::chrono::steady_clock::now();
          if (!cells.empty() && active_timing_) {
            ++active_timing_->nwt_dispatch_calls;
            if (bucket == TimingBucket::NwtHomoUpdate)
              active_timing_->nwt_homo_updates += cells.size();
          }
          update_cells(cells, write_phi, read_phi, homogeneous, accumulate_rms);
          add_timing_ms(bucket, std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - begin)
                                    .count());
        };

    const bool row_span_requested = args_.nwt_homo_kernel == "row-span" ||
                                    args_.nwt_homo_kernel == "avx2" ||
                                    args_.nwt_homo_kernel == "auto";
#if TITAN_SIMD_X86
    const bool row_span_avx2 = row_span_requested &&
                               args_.nwt_homo_kernel != "row-span" &&
                               titan_simd_dispatch::runtime_has_avx2();
#else
    const bool row_span_avx2 = false;
#endif
    const bool salt_tiled_requested = args_.nwt_homo_kernel == "salt-tiled" ||
                                      args_.nwt_homo_kernel == "auto";
#if TITAN_SIMD_X86
    const bool salt_tiled_avx2 =
        salt_tiled_requested && titan_simd_dispatch::runtime_has_avx2();
#else
    const bool salt_tiled_avx2 = false;
#endif
    if (salt_tiled_requested && nwt_half_workspace_.salt_tile_ready) {
      const auto tile_cell_count =
          [](const std::vector<NwtHalfWorkspace::HomoTile> &tiles) {
            std::uint64_t total = 0;
            for (const auto &tile : tiles)
              total += tile.count;
            return total;
          };
      const std::uint64_t even_homo_updates =
          tile_cell_count(nwt_half_workspace_.even_homo_linear_tiles) +
          tile_cell_count(nwt_half_workspace_.even_homo_salt_tiles);
      const std::uint64_t odd_homo_updates =
          tile_cell_count(nwt_half_workspace_.odd_homo_linear_tiles) +
          tile_cell_count(nwt_half_workspace_.odd_homo_salt_tiles);
      const bool collect_nwt_detail = args_.timing == "detail";
      std::vector<double> local_stencil_ms(workers, 0.0);
      std::vector<double> local_salt_ms(workers, 0.0);
      std::vector<double> local_write_ms(workers, 0.0);
      std::vector<double> local_worker_ms(workers, 0.0);

      const auto update_homo_tiles =
          [&](const std::vector<NwtHalfWorkspace::HomoTile> &linear_tiles,
              const std::vector<NwtHalfWorkspace::HomoTile> &salt_tiles,
              AlignedDoubleVector &write_phi,
              const AlignedDoubleVector &read_phi,
              const AlignedDoubleVector &rhs_vec,
              const AlignedDoubleVector &eps_sum_vec,
              const AlignedDoubleVector &screen_vec,
              const AlignedDoubleVector &eps_vec, int color,
              bool accumulate_rms, std::uint64_t update_count) {
            if (linear_tiles.empty() && salt_tiles.empty())
              return;
            if (collect_nwt_detail) {
              std::fill(local_stencil_ms.begin(), local_stencil_ms.end(), 0.0);
              std::fill(local_salt_ms.begin(), local_salt_ms.end(), 0.0);
              std::fill(local_write_ms.begin(), local_write_ms.end(), 0.0);
              std::fill(local_worker_ms.begin(), local_worker_ms.end(), 0.0);
            }
            const auto begin_time = std::chrono::steady_clock::now();
            const NwtHomoOffsets offsets =
                nwt_homo_offsets(shape.nx, shape.ny, color);
            if (active_timing_) {
              active_timing_->nwt_homo_updates += update_count;
              ++active_timing_->nwt_dispatch_calls;
            }
            executor.parallel_do(workers, [&](std::size_t worker) {
              const auto worker_begin =
                  collect_nwt_detail ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
              double sumsq = 0.0;
              double max_delta = 0.0;
              std::size_t count = 0;
              std::size_t clipped = 0;

              const auto process_linear_tile =
                  [&](const NwtHalfWorkspace::HomoTile &tile) {
                    const std::size_t end =
                        static_cast<std::size_t>(tile.begin) + tile.count;
                    for (std::size_t h = tile.begin; h < end; ++h) {
                      const auto base = static_cast<std::ptrdiff_t>(h);
                      const double old_phi = write_phi[h];
                      const double neighbor_sum =
                          eps_vec[h] * (read_phi[base + offsets.xp] +
                                        read_phi[base + offsets.xm] +
                                        read_phi[base + offsets.yp] +
                                        read_phi[base + offsets.ym] +
                                        read_phi[base + offsets.zp] +
                                        read_phi[base + offsets.zm]);
                      const double target = (neighbor_sum + rhs_vec[h]) /
                                            std::max(1.0e-12, eps_sum_vec[h]);
                      const double updated =
                          old_phi + omega * (target - old_phi);
                      const double delta = updated - old_phi;
                      write_phi[h] = updated;
                      if (accumulate_rms) {
                        sumsq += delta * delta;
                        max_delta = std::max(max_delta, std::abs(delta));
                        ++count;
                      }
                    }
                  };

              const auto process_salt_tile =
                  [&](const NwtHalfWorkspace::HomoTile &tile) {
                    const std::size_t end =
                        static_cast<std::size_t>(tile.begin) + tile.count;
                    if (screen_scale == 0.0) {
                      for (std::size_t h = tile.begin; h < end; ++h) {
                        const auto base = static_cast<std::ptrdiff_t>(h);
                        const double old_phi = write_phi[h];
                        const double neighbor_sum =
                            eps_vec[h] * (read_phi[base + offsets.xp] +
                                          read_phi[base + offsets.xm] +
                                          read_phi[base + offsets.yp] +
                                          read_phi[base + offsets.ym] +
                                          read_phi[base + offsets.zp] +
                                          read_phi[base + offsets.zm]);
                        const double target = (neighbor_sum + rhs_vec[h]) /
                                              std::max(1.0e-12, eps_sum_vec[h]);
                        const double updated =
                            old_phi + omega * (target - old_phi);
                        const double delta = updated - old_phi;
                        write_phi[h] = updated;
                        if (accumulate_rms) {
                          sumsq += delta * delta;
                          max_delta = std::max(max_delta, std::abs(delta));
                          ++count;
                        }
                      }
                      return;
                    }
                    for (std::size_t h = tile.begin; h < end; ++h) {
                      const auto base = static_cast<std::ptrdiff_t>(h);
                      const double old_phi = write_phi[h];
                      const double neighbor_sum =
                          eps_vec[h] * (read_phi[base + offsets.xp] +
                                        read_phi[base + offsets.xm] +
                                        read_phi[base + offsets.yp] +
                                        read_phi[base + offsets.ym] +
                                        read_phi[base + offsets.zp] +
                                        read_phi[base + offsets.zm]);
                      const double active_screen = screen_vec[h] * screen_scale;
                      const double raw_arg = old_phi * argument_scale;
                      if (raw_arg > effective_clip || raw_arg < -effective_clip)
                        ++clipped;
                      const auto sinh_cosh = delphi_sinh_cosh_tail_single_sinh(
                          raw_arg, effective_clip, sinh_clip);
                      const double source = sinh_cosh.first * source_scale;
                      const double diagonal =
                          eps_sum_vec[h] + active_screen * sinh_cosh.second;
                      const double nonlinear_rhs =
                          active_screen * (sinh_cosh.second * old_phi - source);
                      const double target =
                          (neighbor_sum + rhs_vec[h] + nonlinear_rhs) /
                          std::max(1.0e-12, diagonal);
                      const double updated =
                          old_phi + omega * (target - old_phi);
                      const double delta = updated - old_phi;
                      write_phi[h] = updated;
                      if (accumulate_rms) {
                        sumsq += delta * delta;
                        max_delta = std::max(max_delta, std::abs(delta));
                        ++count;
                      }
                    }
                  };

              const std::size_t linear_begin =
                  linear_tiles.size() * worker / workers;
              const std::size_t linear_end =
                  linear_tiles.size() * (worker + 1) / workers;
              const auto linear_begin_time =
                  collect_nwt_detail ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
              for (std::size_t tile_index = linear_begin;
                   tile_index < linear_end; ++tile_index) {
#if TITAN_SIMD_X86
                if (salt_tiled_avx2) {
                  update_nwt_homo_linear_span_avx2(
                      write_phi.data(), read_phi.data(), rhs_vec.data(),
                      eps_sum_vec.data(), eps_vec.data(),
                      linear_tiles[tile_index].begin,
                      linear_tiles[tile_index].count, offsets, omega,
                      accumulate_rms, sumsq, max_delta, count);
                } else
#endif
                {
                  process_linear_tile(linear_tiles[tile_index]);
                }
              }
              if (collect_nwt_detail) {
                local_stencil_ms[worker] = elapsed_ms(
                    linear_begin_time, std::chrono::steady_clock::now());
              }

              const std::size_t salt_begin =
                  salt_tiles.size() * worker / workers;
              const std::size_t salt_end =
                  salt_tiles.size() * (worker + 1) / workers;
              const auto salt_begin_time =
                  collect_nwt_detail ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
              for (std::size_t tile_index = salt_begin; tile_index < salt_end;
                   ++tile_index) {
                process_salt_tile(salt_tiles[tile_index]);
              }
              if (collect_nwt_detail) {
                local_salt_ms[worker] = elapsed_ms(
                    salt_begin_time, std::chrono::steady_clock::now());
                local_worker_ms[worker] =
                    elapsed_ms(worker_begin, std::chrono::steady_clock::now());
              }

              local_sumsq[worker] += sumsq;
              local_max[worker] = std::max(local_max[worker], max_delta);
              local_count[worker] += count;
              local_clipped[worker] += clipped;
            });
            const double wall_ms =
                elapsed_ms(begin_time, std::chrono::steady_clock::now());
            add_timing_ms(TimingBucket::NwtHomoUpdate, wall_ms);
            if (collect_nwt_detail) {
              double max_worker_ms = 0.0;
              double max_stencil_ms = 0.0;
              double max_salt_ms = 0.0;
              double max_write_ms = 0.0;
              for (std::size_t worker = 0; worker < workers; ++worker) {
                max_worker_ms =
                    std::max(max_worker_ms, local_worker_ms[worker]);
                max_stencil_ms =
                    std::max(max_stencil_ms, local_stencil_ms[worker]);
                max_salt_ms = std::max(max_salt_ms, local_salt_ms[worker]);
                max_write_ms = std::max(max_write_ms, local_write_ms[worker]);
              }
              add_timing_ms(TimingBucket::NwtHomoStencil, max_stencil_ms);
              add_timing_ms(TimingBucket::NwtHomoSaltMath, max_salt_ms);
              add_timing_ms(TimingBucket::NwtHomoWrite, max_write_ms);
              add_timing_ms(TimingBucket::NwtThreadWait,
                            std::max(0.0, wall_ms - max_worker_ms));
            }
          };

      for (int iter = 1; iter <= sweeps; ++iter) {
        std::fill(local_sumsq.begin(), local_sumsq.end(), 0.0);
        std::fill(local_max.begin(), local_max.end(), 0.0);
        std::fill(local_count.begin(), local_count.end(), 0);
        std::fill(local_clipped.begin(), local_clipped.end(), 0);

        update_homo_tiles(nwt_half_workspace_.even_homo_linear_tiles,
                          nwt_half_workspace_.even_homo_salt_tiles, even_phi,
                          odd_phi, rhs_even, eps_sum_even, screen_even,
                          homo_eps_even, 0, false, even_homo_updates);
        run_timed_cells(nwt_half_workspace_.even_hetero_cells, even_phi,
                        odd_phi, false, false, TimingBucket::NwtHeteroUpdate);
        const bool accumulate_odd = !delphi_block_stop || (iter == sweeps);
        update_homo_tiles(nwt_half_workspace_.odd_homo_linear_tiles,
                          nwt_half_workspace_.odd_homo_salt_tiles, odd_phi,
                          even_phi, rhs_odd, eps_sum_odd, screen_odd,
                          homo_eps_odd, 1, accumulate_odd, odd_homo_updates);
        run_timed_cells(nwt_half_workspace_.odd_hetero_cells, odd_phi, even_phi,
                        false, accumulate_odd, TimingBucket::NwtHeteroUpdate);

        const auto reduce_time_begin = std::chrono::steady_clock::now();
        double sumsq = 0.0;
        double max_delta = 0.0;
        std::size_t count = 0;
        std::size_t clipped = 0;
        for (std::size_t worker = 0; worker < workers; ++worker) {
          sumsq += local_sumsq[worker];
          max_delta = std::max(max_delta, local_max[worker]);
          count += local_count[worker];
          clipped += local_clipped[worker];
        }
        clipped_points = std::max(clipped_points, clipped);
        stats.iterations = iter;
        if (delphi_block_stop) {
          stats.rms = delphi_rms_denominator > 0
                          ? std::sqrt(sumsq / static_cast<double>(
                                                  delphi_rms_denominator))
                          : 0.0;
        } else {
          stats.rms =
              count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
        }
        stats.max_dphi = max_delta;
        const double reduce_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - reduce_time_begin)
                .count();
        add_timing_ms(TimingBucket::NwtRmsReduce, reduce_ms);
        add_timing_ms(TimingBucket::NwtReduce, reduce_ms);
        add_timing_ms(TimingBucket::NwtBlockOverhead, reduce_ms);
        if (!delphi_block_stop && (stats.max_dphi <= args_.nl_dphi ||
                                   stats.rms <= args_.nl_residual)) {
          stats.converged = true;
          break;
        }
      }

      if (delphi_block_stop &&
          (stats.max_dphi < args_.nl_dphi || stats.rms < args_.nl_residual))
        stats.converged = true;
      add_timing_ms(TimingBucket::NwtHalfUpdate,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - update_time_begin)
                        .count());
      return stats;
    }
    if (row_span_requested && nwt_half_workspace_.row_span_ready) {
      const auto span_cell_count =
          [](const std::vector<NwtHalfWorkspace::HomoSpan> &spans) {
            std::uint64_t total = 0;
            for (const auto &span : spans)
              total += span.count;
            return total;
          };
      const std::uint64_t even_homo_updates =
          span_cell_count(nwt_half_workspace_.even_homo_linear_spans) +
          span_cell_count(nwt_half_workspace_.even_homo_salt_spans);
      const std::uint64_t odd_homo_updates =
          span_cell_count(nwt_half_workspace_.odd_homo_linear_spans) +
          span_cell_count(nwt_half_workspace_.odd_homo_salt_spans);

      const auto update_homo_spans =
          [&](const std::vector<NwtHalfWorkspace::HomoSpan> &linear_spans,
              const std::vector<NwtHalfWorkspace::HomoSpan> &salt_spans,
              AlignedDoubleVector &write_phi,
              const AlignedDoubleVector &read_phi,
              const AlignedDoubleVector &rhs_vec,
              const AlignedDoubleVector &eps_sum_vec,
              const AlignedDoubleVector &screen_vec,
              const AlignedDoubleVector &eps_vec, int color,
              bool accumulate_rms, std::uint64_t update_count) {
            if (linear_spans.empty() && salt_spans.empty())
              return;
            const auto begin_time = std::chrono::steady_clock::now();
            const NwtHomoOffsets offsets =
                nwt_homo_offsets(shape.nx, shape.ny, color);
            if (active_timing_) {
              active_timing_->nwt_homo_updates += update_count;
              ++active_timing_->nwt_dispatch_calls;
            }
            executor.parallel_do(workers, [&](std::size_t worker) {
              double sumsq = 0.0;
              double max_delta = 0.0;
              std::size_t count = 0;
              std::size_t clipped = 0;

              const auto process_linear_span_scalar =
                  [&](const NwtHalfWorkspace::HomoSpan &span) {
                    const std::size_t end =
                        static_cast<std::size_t>(span.begin) + span.count;
                    for (std::size_t h = span.begin; h < end; ++h) {
                      const auto base = static_cast<std::ptrdiff_t>(h);
                      const double old_phi = write_phi[h];
                      const double neighbor_sum =
                          eps_vec[h] * (read_phi[base + offsets.xp] +
                                        read_phi[base + offsets.xm] +
                                        read_phi[base + offsets.yp] +
                                        read_phi[base + offsets.ym] +
                                        read_phi[base + offsets.zp] +
                                        read_phi[base + offsets.zm]);
                      const double target = (neighbor_sum + rhs_vec[h]) /
                                            std::max(1.0e-12, eps_sum_vec[h]);
                      const double updated =
                          old_phi + omega * (target - old_phi);
                      const double delta = updated - old_phi;
                      write_phi[h] = updated;
                      if (accumulate_rms) {
                        sumsq += delta * delta;
                        max_delta = std::max(max_delta, std::abs(delta));
                        ++count;
                      }
                    }
                  };

              const auto process_salt_span =
                  [&](const NwtHalfWorkspace::HomoSpan &span) {
                    const std::size_t end =
                        static_cast<std::size_t>(span.begin) + span.count;
                    for (std::size_t h = span.begin; h < end; ++h) {
                      const auto base = static_cast<std::ptrdiff_t>(h);
                      const double old_phi = write_phi[h];
                      const double neighbor_sum =
                          eps_vec[h] * (read_phi[base + offsets.xp] +
                                        read_phi[base + offsets.xm] +
                                        read_phi[base + offsets.yp] +
                                        read_phi[base + offsets.ym] +
                                        read_phi[base + offsets.zp] +
                                        read_phi[base + offsets.zm]);
                      const double active_screen = screen_vec[h] * screen_scale;
                      double diagonal = eps_sum_vec[h];
                      double nonlinear_rhs = 0.0;
                      if (active_screen > 0.0) {
                        const double raw_arg = old_phi * argument_scale;
                        if (raw_arg > effective_clip ||
                            raw_arg < -effective_clip)
                          ++clipped;
                        const auto sinh_cosh =
                            delphi_sinh_cosh_tail_precomputed(
                                raw_arg, effective_clip, sinh_clip);
                        const double source = sinh_cosh.first * source_scale;
                        diagonal += active_screen * sinh_cosh.second;
                        nonlinear_rhs = active_screen *
                                        (sinh_cosh.second * old_phi - source);
                      }
                      const double target =
                          (neighbor_sum + rhs_vec[h] + nonlinear_rhs) /
                          std::max(1.0e-12, diagonal);
                      const double updated =
                          old_phi + omega * (target - old_phi);
                      const double delta = updated - old_phi;
                      write_phi[h] = updated;
                      if (accumulate_rms) {
                        sumsq += delta * delta;
                        max_delta = std::max(max_delta, std::abs(delta));
                        ++count;
                      }
                    }
                  };

              const auto process_linear_range = [&](std::size_t begin,
                                                    std::size_t end) {
                for (std::size_t span_index = begin; span_index < end;
                     ++span_index) {
#if TITAN_SIMD_X86
                  if (row_span_avx2) {
                    update_nwt_homo_linear_span_avx2(
                        write_phi.data(), read_phi.data(), rhs_vec.data(),
                        eps_sum_vec.data(), eps_vec.data(),
                        linear_spans[span_index].begin,
                        linear_spans[span_index].count, offsets, omega,
                        accumulate_rms, sumsq, max_delta, count);
                  } else
#endif
                  {
                    process_linear_span_scalar(linear_spans[span_index]);
                  }
                }
              };

              const std::size_t linear_begin =
                  linear_spans.size() * worker / workers;
              const std::size_t linear_end =
                  linear_spans.size() * (worker + 1) / workers;
              process_linear_range(linear_begin, linear_end);

              const std::size_t salt_begin =
                  salt_spans.size() * worker / workers;
              const std::size_t salt_end =
                  salt_spans.size() * (worker + 1) / workers;
              for (std::size_t span_index = salt_begin; span_index < salt_end;
                   ++span_index) {
                process_salt_span(salt_spans[span_index]);
              }

              local_sumsq[worker] += sumsq;
              local_max[worker] = std::max(local_max[worker], max_delta);
              local_count[worker] += count;
              local_clipped[worker] += clipped;
            });
            add_timing_ms(TimingBucket::NwtHomoUpdate,
                          std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - begin_time)
                              .count());
          };

      for (int iter = 1; iter <= sweeps; ++iter) {
        std::fill(local_sumsq.begin(), local_sumsq.end(), 0.0);
        std::fill(local_max.begin(), local_max.end(), 0.0);
        std::fill(local_count.begin(), local_count.end(), 0);
        std::fill(local_clipped.begin(), local_clipped.end(), 0);

        update_homo_spans(nwt_half_workspace_.even_homo_linear_spans,
                          nwt_half_workspace_.even_homo_salt_spans, even_phi,
                          odd_phi, rhs_even, eps_sum_even, screen_even,
                          homo_eps_even, 0, false, even_homo_updates);
        run_timed_cells(nwt_half_workspace_.even_hetero_cells, even_phi,
                        odd_phi, false, false, TimingBucket::NwtHeteroUpdate);
        const bool accumulate_odd = !delphi_block_stop || (iter == sweeps);
        update_homo_spans(nwt_half_workspace_.odd_homo_linear_spans,
                          nwt_half_workspace_.odd_homo_salt_spans, odd_phi,
                          even_phi, rhs_odd, eps_sum_odd, screen_odd,
                          homo_eps_odd, 1, accumulate_odd, odd_homo_updates);
        run_timed_cells(nwt_half_workspace_.odd_hetero_cells, odd_phi, even_phi,
                        false, accumulate_odd, TimingBucket::NwtHeteroUpdate);

        const auto reduce_time_begin = std::chrono::steady_clock::now();
        double sumsq = 0.0;
        double max_delta = 0.0;
        std::size_t count = 0;
        std::size_t clipped = 0;
        for (std::size_t worker = 0; worker < workers; ++worker) {
          sumsq += local_sumsq[worker];
          max_delta = std::max(max_delta, local_max[worker]);
          count += local_count[worker];
          clipped += local_clipped[worker];
        }
        clipped_points = std::max(clipped_points, clipped);
        stats.iterations = iter;
        if (delphi_block_stop) {
          stats.rms = delphi_rms_denominator > 0
                          ? std::sqrt(sumsq / static_cast<double>(
                                                  delphi_rms_denominator))
                          : 0.0;
        } else {
          stats.rms =
              count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
        }
        stats.max_dphi = max_delta;
        const double reduce_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - reduce_time_begin)
                .count();
        add_timing_ms(TimingBucket::NwtRmsReduce, reduce_ms);
        add_timing_ms(TimingBucket::NwtReduce, reduce_ms);
        if (!delphi_block_stop && (stats.max_dphi <= args_.nl_dphi ||
                                   stats.rms <= args_.nl_residual)) {
          stats.converged = true;
          break;
        }
      }

      if (delphi_block_stop &&
          (stats.max_dphi < args_.nl_dphi || stats.rms < args_.nl_residual))
        stats.converged = true;
      add_timing_ms(TimingBucket::NwtHalfUpdate,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - update_time_begin)
                        .count());
      return stats;
    }

    for (int iter = 1; iter <= sweeps; ++iter) {
      std::fill(local_sumsq.begin(), local_sumsq.end(), 0.0);
      std::fill(local_max.begin(), local_max.end(), 0.0);
      std::fill(local_count.begin(), local_count.end(), 0);
      std::fill(local_clipped.begin(), local_clipped.end(), 0);

      run_timed_cells(nwt_half_workspace_.even_homo_cells, even_phi, odd_phi,
                      true, false, TimingBucket::NwtHomoUpdate);
      run_timed_cells(nwt_half_workspace_.even_hetero_cells, even_phi, odd_phi,
                      false, false, TimingBucket::NwtHeteroUpdate);
      const bool accumulate_odd = !delphi_block_stop || (iter == sweeps);
      run_timed_cells(nwt_half_workspace_.odd_homo_cells, odd_phi, even_phi,
                      true, accumulate_odd, TimingBucket::NwtHomoUpdate);
      run_timed_cells(nwt_half_workspace_.odd_hetero_cells, odd_phi, even_phi,
                      false, accumulate_odd, TimingBucket::NwtHeteroUpdate);

      const auto reduce_time_begin = std::chrono::steady_clock::now();
      double sumsq = 0.0;
      double max_delta = 0.0;
      std::size_t count = 0;
      std::size_t clipped = 0;
      for (std::size_t worker = 0; worker < workers; ++worker) {
        sumsq += local_sumsq[worker];
        max_delta = std::max(max_delta, local_max[worker]);
        count += local_count[worker];
        clipped += local_clipped[worker];
      }
      clipped_points = std::max(clipped_points, clipped);
      stats.iterations = iter;
      if (delphi_block_stop) {
        stats.rms =
            delphi_rms_denominator > 0
                ? std::sqrt(sumsq / static_cast<double>(delphi_rms_denominator))
                : 0.0;
      } else {
        stats.rms =
            count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
      }
      stats.max_dphi = max_delta;
      const double reduce_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - reduce_time_begin)
              .count();
      add_timing_ms(TimingBucket::NwtRmsReduce, reduce_ms);
      add_timing_ms(TimingBucket::NwtReduce, reduce_ms);
      if (!delphi_block_stop &&
          (stats.max_dphi <= args_.nl_dphi || stats.rms <= args_.nl_residual)) {
        stats.converged = true;
        break;
      }
    }

    if (delphi_block_stop &&
        (stats.max_dphi < args_.nl_dphi || stats.rms < args_.nl_residual))
      stats.converged = true;
    add_timing_ms(TimingBucket::NwtHalfUpdate,
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - update_time_begin)
                      .count());
    return stats;
  }

  for (int iter = 1; iter <= sweeps; ++iter) {
    std::fill(local_sumsq.begin(), local_sumsq.end(), 0.0);
    std::fill(local_max.begin(), local_max.end(), 0.0);
    std::fill(local_count.begin(), local_count.end(), 0);
    std::fill(local_clipped.begin(), local_clipped.end(), 0);

    for (int color = 0; color < 2; ++color) {
      executor.parallel_for(
          interior_z, workers, 1,
          [&](std::size_t worker, std::size_t begin, std::size_t end) {
            double sumsq = 0.0;
            double max_delta = 0.0;
            std::size_t count = 0;
            std::size_t clipped = 0;
            for (std::size_t kk = begin; kk < end; ++kk) {
              const std::size_t k = kk + 1;
              for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
                const std::size_t first_i =
                    1 +
                    (static_cast<int>((1 + j + k) & 1ULL) == color ? 0U : 1U);
                for (std::size_t i = first_i; i + 1 < shape.nx; i += 2) {
                  const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
                  const std::size_t h = id / 2;
                  double old_phi = 0.0;
                  double neighbor_sum = 0.0;
                  double eps_sum = 1.0;
                  double active_screen = 0.0;
                  double rhs_rho = 0.0;
                  if (color == 0) {
                    old_phi = even_phi[h];
                    neighbor_sum =
                        grid_.eps_x[id] * odd_phi[(id + 1) / 2] +
                        grid_.eps_x[id - 1] * odd_phi[(id - 1) / 2] +
                        grid_.eps_y[id] * odd_phi[(id + shape.nx) / 2] +
                        grid_.eps_y[id - shape.nx] *
                            odd_phi[(id - shape.nx) / 2] +
                        grid_.eps_z[id] * odd_phi[(id + xy) / 2] +
                        grid_.eps_z[id - xy] * odd_phi[(id - xy) / 2];
                    eps_sum = eps_sum_even[h];
                    active_screen = screen_even[h] * screen_scale;
                    rhs_rho = rhs_even[h];
                  } else {
                    old_phi = odd_phi[h];
                    neighbor_sum =
                        grid_.eps_x[id] * even_phi[(id + 1) / 2] +
                        grid_.eps_x[id - 1] * even_phi[(id - 1) / 2] +
                        grid_.eps_y[id] * even_phi[(id + shape.nx) / 2] +
                        grid_.eps_y[id - shape.nx] *
                            even_phi[(id - shape.nx) / 2] +
                        grid_.eps_z[id] * even_phi[(id + xy) / 2] +
                        grid_.eps_z[id - xy] * even_phi[(id - xy) / 2];
                    eps_sum = eps_sum_odd[h];
                    active_screen = screen_odd[h] * screen_scale;
                    rhs_rho = rhs_odd[h];
                  }
                  double diagonal = eps_sum;
                  double nonlinear_rhs = 0.0;
                  if (active_screen > 0.0) {
                    const double raw_arg =
                        nonlinear_argument_value(old_phi, alpha);
                    if (raw_arg > clip || raw_arg < -clip)
                      ++clipped;
                    const auto sinh_cosh = delphi_sinh_cosh_tail(raw_arg, clip);
                    const double source =
                        (delphi_field_units_active() ||
                         args_.test_nl_source == "delphi-dimensionless")
                            ? sinh_cosh.first
                            : sinh_cosh.first / std::max(1.0e-30, alpha);
                    diagonal += active_screen * sinh_cosh.second;
                    nonlinear_rhs =
                        active_screen * (sinh_cosh.second * old_phi - source);
                  }
                  const double target =
                      (neighbor_sum + rhs_rho + nonlinear_rhs) /
                      std::max(1.0e-12, diagonal);
                  const double updated = old_phi + omega * (target - old_phi);
                  const double delta = updated - old_phi;
                  if (color == 0)
                    even_phi[h] = updated;
                  else
                    odd_phi[h] = updated;
                  if (!delphi_block_stop || (iter == sweeps && color == 1)) {
                    sumsq += delta * delta;
                    max_delta = std::max(max_delta, std::abs(delta));
                    ++count;
                  }
                }
              }
            }
            local_sumsq[worker] += sumsq;
            local_max[worker] = std::max(local_max[worker], max_delta);
            local_count[worker] += count;
            local_clipped[worker] += clipped;
          });
    }

    const auto reduce_time_begin = std::chrono::steady_clock::now();
    double sumsq = 0.0;
    double max_delta = 0.0;
    std::size_t count = 0;
    std::size_t clipped = 0;
    for (std::size_t worker = 0; worker < workers; ++worker) {
      sumsq += local_sumsq[worker];
      max_delta = std::max(max_delta, local_max[worker]);
      count += local_count[worker];
      clipped += local_clipped[worker];
    }
    clipped_points = std::max(clipped_points, clipped);
    stats.iterations = iter;
    if (delphi_block_stop) {
      stats.rms =
          delphi_rms_denominator > 0
              ? std::sqrt(sumsq / static_cast<double>(delphi_rms_denominator))
              : 0.0;
    } else {
      stats.rms =
          count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
    }
    stats.max_dphi = max_delta;
    add_timing_ms(TimingBucket::NwtRmsReduce,
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - reduce_time_begin)
                      .count());
    if (!delphi_block_stop &&
        (stats.max_dphi <= args_.nl_dphi || stats.rms <= args_.nl_residual)) {
      stats.converged = true;
      break;
    }
  }

  if (delphi_block_stop &&
      (stats.max_dphi < args_.nl_dphi || stats.rms < args_.nl_residual))
    stats.converged = true;
  add_timing_ms(TimingBucket::NwtHalfUpdate,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - update_time_begin)
                    .count());
  return stats;
}

AnalyzerImpl::FieldSolve AnalyzerImpl::solve_newton_nonlinear_field(
    const std::vector<Atom> &atoms, double eps_in, double eps_out, double kappa,
    TitanExecutor &executor, const FrameResult *map_result) {
  FieldSolve field =
      solve_linear_field(atoms, eps_in, eps_out, kappa, executor, nullptr);
  field.linear_seed_solver = field.solver_mode;
  field.pbe_model_effective = "nonlinear";
  field.nl_solver = "newton";
  if (args_.nl_inner == "auto") {
    const bool delphi_nwt_fixture =
        !args_.test_pqr.empty() && args_.surface == "delphi-vdw";
    field.nl_inner_solver =
        delphi_nwt_fixture ? std::string("sor")
                           : (field.shape.n > 250000 ? std::string("vcycle")
                                                     : std::string("sor"));
  } else {
    field.nl_inner_solver = args_.nl_inner;
  }
  field.nl_mg_mode = args_.nl_multigrid;
  field.nl_relax = args_.nl_relax == "auto" ? 1.0 : effective_nl_relax();

  if (kappa <= 0.0 || args_.salt <= 0.0) {
    field.pbe_model_effective = "linear";
    field.nl_status = "linearized_no_salt";
    if (map_result != nullptr) {
      maybe_write_maps(field.shape, *map_result, false);
      maybe_write_maps(field.shape, *map_result, true);
    }
    return field;
  }

  if (map_result != nullptr)
    maybe_write_maps(field.shape, *map_result, false);
  const auto nonlinear_time_begin = std::chrono::steady_clock::now();
  if (delphi_nwt_parity_active()) {
    for (std::size_t k = 1; k + 1 < field.shape.nz; ++k) {
      for (std::size_t j = 1; j + 1 < field.shape.ny; ++j) {
        for (std::size_t i = 1; i + 1 < field.shape.nx; ++i) {
          grid_.phi[idx3(i, j, k, field.shape.nx, field.shape.ny)] = 0.0;
        }
      }
    }
    field.linear_seed_solver = "delphi_boundary_zero";
  }

  const double alpha = nonlinear_alpha();
  const double clip = effective_nonlinear_clip();
  const bool dimensionless_mode = nonlinear_dimensionless_mode();
  scale_nonlinear_field_to_solve_units(grid_.phi, alpha);
  AlignedDoubleVector source(field.shape.n, 0.0);
  AlignedDoubleVector residual_vec(field.shape.n, 0.0);
  AlignedDoubleVector delta(field.shape.n, 0.0);
  SolveStats last_stats;
  int total_iterations = 0;
  int blocks = 0;
  int backtracks = 0;
  int continuation_steps = 0;
  int inner_iterations = 0;
  int rejected_steps = 0;
  std::size_t max_clipped = 0;
  double max_step = 0.0;
  double last_step_norm = 0.0;
  double last_residual_before = 0.0;
  double last_residual_after = 0.0;
  double last_correction_residual = 0.0;
  double last_correction_max_residual = 0.0;
  double last_accepted_lambda = 0.0;
  double last_jac_diag_min = 0.0;
  double last_jac_diag_max = 0.0;
  std::size_t last_screen_points = 0;
  std::size_t last_active_nonlinear_points = 0;
  int last_inner_levels = 1;
  int last_inner_cycles = 0;
  double last_inner_residual_before = 0.0;
  double last_inner_residual_after = 0.0;
  std::string last_inner_effective = field.nl_inner_solver;
  double residual = std::numeric_limits<double>::infinity();
  double final_screen_scale = 1.0;
  double trust_psi = 0.5;
  bool finite = true;
  bool converged = false;
  std::string stop_reason = "work_budget";
  std::string line_search_status =
      args_.nl_multigrid == "fas" ? "fas_sor_correction" : "not_started";

  std::vector<double> screen_scales;
  if (delphi_nwt_parity_active())
    screen_scales = {1.0};
  else if (args_.nl_continuation == "auto" && args_.salt > 0.5)
    screen_scales = {0.1, 0.25, 0.5, 0.75, 1.0};
  else if (args_.nl_continuation == "auto")
    screen_scales = {0.25, 0.5, 0.75, 1.0};
  else
    screen_scales = {1.0};

  AlignedDoubleVector nonlinear_before;
  AlignedDoubleVector nwt_before;
  std::ofstream nwt_progress;
  const bool nwt_progress_strict =
      delphi_nwt_parity_active() &&
      (args_.nwt_progress == "strict" ||
       (args_.nwt_progress == "auto" && args_.parity_dump != "off"));
  if (nwt_progress_strict) {
    nwt_progress.open("pbe_nwt_solver_progress.csv",
                      std::ios::out | std::ios::trunc);
    if (!nwt_progress.is_open())
      throw std::runtime_error(
          "PBE: failed to open pbe_nwt_solver_progress.csv.");
    nwt_progress << "block,total_iterations,screen_scale,sweeps,rms,max_dphi,"
                    "operator_residual,operator_residual_max,clipped_points,"
                    "stop_reason,field_units,operator_convention\n";
    nwt_progress << std::setprecision(12);
  }

  for (double screen_scale : screen_scales) {
    final_screen_scale = screen_scale;
    ++continuation_steps;
    std::size_t clipped = 0;
    const bool fast_delphi_nwt = delphi_nwt_parity_active() &&
                                 field.nl_inner_solver == "sor" &&
                                 !nwt_progress_strict;
    if (fast_delphi_nwt) {
      residual = std::numeric_limits<double>::infinity();
      last_residual_after = residual;
    } else {
      const auto source_time_begin = std::chrono::steady_clock::now();
      clipped = update_nonlinear_source(field.shape, source, alpha, clip);
      add_timing_ms(TimingBucket::NwtSourceUpdate,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - source_time_begin)
                        .count());
      max_clipped = std::max(max_clipped, clipped);
      const auto residual_time_begin = std::chrono::steady_clock::now();
      residual = compute_nonlinear_residual_vector(
          field.shape, grid_.phi, source, screen_scale, residual_vec, executor);
      add_timing_ms(TimingBucket::NwtOperatorResidual,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - residual_time_begin)
                        .count());
      last_residual_after = residual;
    }
    const double stage_initial_residual =
        std::isfinite(residual) ? residual : std::numeric_limits<double>::max();
    const double stage_residual_target =
        screen_scale < 1.0 ? std::max({args_.nl_residual, 1.0e-4,
                                       0.20 * stage_initial_residual})
                           : args_.nl_residual;
    bool step_done = false;
    int stagnant_steps = 0;
    double best_stage_residual = residual;
    if (delphi_nwt_parity_active() && field.nl_inner_solver == "sor") {
      const int block_size = std::max(1, args_.nl_block);
      while (total_iterations < args_.nl_max_iter && !step_done) {
        last_residual_before = residual;
        std::size_t nwt_clipped = 0;
        const int sweeps = std::max(
            1, std::min(block_size, args_.nl_max_iter - total_iterations));
        SolveStats nwt_stats = solve_newton_nwt_sor_block(
            field.shape, sweeps, screen_scale, executor, nwt_clipped);
        std::size_t nwt_source_clipped = 0;
        double operator_residual = nwt_stats.rms;
        if (nwt_progress_strict) {
          const auto source_time_begin = std::chrono::steady_clock::now();
          nwt_source_clipped =
              update_nonlinear_source(field.shape, source, alpha, clip);
          add_timing_ms(
              TimingBucket::NwtSourceUpdate,
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - source_time_begin)
                  .count());
          const auto residual_time_begin = std::chrono::steady_clock::now();
          operator_residual = compute_nonlinear_residual_vector(
              field.shape, grid_.phi, source, screen_scale, residual_vec,
              executor);
          add_timing_ms(
              TimingBucket::NwtOperatorResidual,
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - residual_time_begin)
                  .count());
        }
        double operator_residual_max = 0.0;
        if (nwt_progress_strict) {
          for (double value : residual_vec)
            operator_residual_max =
                std::max(operator_residual_max, std::abs(value));
        } else {
          operator_residual_max = nwt_stats.max_dphi;
        }
        finite = std::isfinite(operator_residual) &&
                 std::isfinite(nwt_stats.rms) &&
                 std::isfinite(nwt_stats.max_dphi) &&
                 std::all_of(grid_.phi.begin(), grid_.phi.end(),
                             [](double v) { return std::isfinite(v); });
        const int used_inner = std::max(1, nwt_stats.iterations);
        inner_iterations += used_inner;
        total_iterations += used_inner;
        ++blocks;
        residual =
            std::isfinite(operator_residual) ? operator_residual : residual;
        last_residual_after = residual;
        last_step_norm = nwt_stats.rms;
        max_step = nwt_stats.max_dphi;
        max_clipped =
            std::max(max_clipped, std::max(nwt_clipped, nwt_source_clipped));
        last_stats = nwt_stats;
        last_inner_effective = "nwt_sor_block";
        last_inner_levels = 1;
        last_inner_cycles = used_inner;
        last_inner_residual_before = last_residual_before;
        last_inner_residual_after = residual;
        last_correction_residual = residual;
        last_correction_max_residual = operator_residual_max;
        line_search_status = "nwt_block_stop";
        last_accepted_lambda = 1.0;
        if (!finite) {
          stop_reason = "nonfinite";
          step_done = true;
        } else if (nwt_stats.max_dphi < args_.nl_dphi ||
                   nwt_stats.rms < args_.nl_residual) {
          last_stats.converged = true;
          converged = true;
          stop_reason =
              nwt_stats.max_dphi < args_.nl_dphi ? "dphi_tol" : "rms_tol";
          step_done = true;
        }
        if (nwt_progress.is_open()) {
          const auto progress_io_begin = std::chrono::steady_clock::now();
          nwt_progress << blocks << ',' << total_iterations << ','
                       << screen_scale << ',' << used_inner << ','
                       << nwt_stats.rms << ',' << nwt_stats.max_dphi << ','
                       << operator_residual << ',' << operator_residual_max
                       << ',' << max_clipped << ',' << csv_escape(stop_reason)
                       << ','
                       << csv_escape(delphi_field_units_active()
                                         ? "delphi-dimensionless"
                                         : args_.test_field_units)
                       << ',' << csv_escape("delphi_nwt") << '\n';
          add_timing_ms(
              TimingBucket::NwtProgressIo,
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - progress_io_begin)
                  .count());
        }
        if (step_done)
          break;
      }
      if (!converged && finite && total_iterations >= args_.nl_max_iter)
        stop_reason = "work_budget";
      break;
    }
    while (total_iterations < args_.nl_max_iter && !step_done) {
      nonlinear_before = grid_.phi;
      last_residual_before = residual;
      if (screen_scale >= 1.0 && residual <= args_.nl_residual) {
        converged = true;
        last_stats.converged = true;
        stop_reason = "residual_tol";
        step_done = true;
        break;
      }

      if (field.nl_inner_solver == "sor") {
        nwt_before = grid_.phi;
        std::size_t nwt_clipped = 0;
        SolveStats nwt_stats = solve_newton_nwt_sor_block(
            field.shape,
            std::max(1, std::min(args_.nl_inner_iter,
                                 args_.nl_max_iter - total_iterations)),
            screen_scale, executor, nwt_clipped);
        const std::size_t nwt_source_clipped =
            update_nonlinear_source(field.shape, source, alpha, clip);
        const double nwt_residual = compute_nonlinear_residual_vector(
            field.shape, grid_.phi, source, screen_scale, residual_vec,
            executor);
        double nwt_residual_max = 0.0;
        for (double value : residual_vec)
          nwt_residual_max = std::max(nwt_residual_max, std::abs(value));
        const bool nwt_finite =
            std::isfinite(nwt_residual) &&
            std::all_of(grid_.phi.begin(), grid_.phi.end(),
                        [](double v) { return std::isfinite(v); });
        if (nwt_finite &&
            (nwt_residual < residual || nwt_residual <= stage_residual_target ||
             nwt_stats.max_dphi <= args_.nl_dphi)) {
          const int used_inner = std::max(1, nwt_stats.iterations);
          inner_iterations += used_inner;
          total_iterations += used_inner;
          ++blocks;
          residual = nwt_residual;
          last_residual_after = residual;
          last_step_norm = nwt_stats.rms;
          max_step = std::max(max_step, nwt_stats.max_dphi);
          max_clipped =
              std::max(max_clipped, std::max(nwt_clipped, nwt_source_clipped));
          last_stats = nwt_stats;
          last_stats.rms = residual;
          last_inner_effective = "nwt_sor";
          last_inner_levels = 1;
          last_inner_cycles = used_inner;
          last_inner_residual_before = last_residual_before;
          last_inner_residual_after = residual;
          last_correction_residual = residual;
          last_correction_max_residual = nwt_residual_max;
          line_search_status = "nwt_sor";
          last_accepted_lambda = 1.0;
          if (residual < best_stage_residual * 0.999) {
            best_stage_residual = residual;
            stagnant_steps = 0;
          } else if (++stagnant_steps >= args_.nl_stagnation) {
            stop_reason = "stagnation";
            step_done = true;
          }
          if (screen_scale >= 1.0 && residual <= args_.nl_residual) {
            last_stats.converged = true;
            converged = true;
            stop_reason = "residual_tol";
            step_done = true;
          } else if (screen_scale < 1.0 &&
                     (residual <= stage_residual_target ||
                      nwt_stats.max_dphi <= args_.nl_dphi)) {
            stop_reason = "stage_target";
            step_done = true;
          }
          continue;
        }
        grid_.phi = nwt_before;
        update_nonlinear_source(field.shape, source, alpha, clip);
        compute_nonlinear_residual_vector(field.shape, grid_.phi, source,
                                          screen_scale, residual_vec, executor);
      }

      const NewtonCorrectionStats correction_stats = solve_newton_correction(
          field.shape, residual_vec, residual, alpha, clip, screen_scale, delta,
          args_.nl_inner_iter, field.nl_inner_solver, executor);
      last_stats = correction_stats.sweeps;
      const int used_inner = std::max(1, correction_stats.sweeps.iterations);
      inner_iterations += used_inner;
      ++total_iterations;
      ++blocks;
      last_correction_residual = correction_stats.residual_rms;
      last_correction_max_residual = correction_stats.residual_max;
      last_jac_diag_min = correction_stats.jac_diag_min;
      last_jac_diag_max = correction_stats.jac_diag_max;
      last_screen_points = correction_stats.screen_points;
      last_active_nonlinear_points = correction_stats.active_nonlinear_points;
      last_inner_levels = correction_stats.inner_levels;
      last_inner_cycles = correction_stats.inner_cycles;
      last_inner_residual_before = correction_stats.inner_residual_before;
      last_inner_residual_after = correction_stats.inner_residual_after;
      last_inner_effective = correction_stats.inner_effective;

      bool accepted = false;
      double accepted_residual = std::numeric_limits<double>::infinity();
      double accepted_step_norm = 0.0;
      double accepted_max_step = 0.0;
      std::size_t accepted_clipped = clipped;
      double trial_lambda = field.nl_relax;
      while (trial_lambda >= 1.0e-4) {
        grid_.phi = nonlinear_before;
        double sumsq = 0.0;
        double local_max_step = 0.0;
        std::size_t count = 0;
        const double max_field_step =
            args_.nl_line_search == "auto"
                ? trust_psi /
                      (dimensionless_mode ? 1.0 : std::max(1.0e-12, alpha))
                : std::numeric_limits<double>::infinity();
        for (std::size_t k = 1; k + 1 < field.shape.nz; ++k) {
          for (std::size_t j = 1; j + 1 < field.shape.ny; ++j) {
            for (std::size_t i = 1; i + 1 < field.shape.nx; ++i) {
              const std::size_t id =
                  idx3(i, j, k, field.shape.nx, field.shape.ny);
              const double raw_step = trial_lambda * delta[id];
              const double step =
                  std::clamp(raw_step, -max_field_step, max_field_step);
              grid_.phi[id] = nonlinear_before[id] + step;
              sumsq += step * step;
              local_max_step = std::max(local_max_step, std::abs(step));
              ++count;
            }
          }
        }
        finite = std::all_of(grid_.phi.begin(), grid_.phi.end(),
                             [](double v) { return std::isfinite(v); });
        if (finite) {
          const std::size_t trial_clipped =
              update_nonlinear_source(field.shape, source, alpha, clip);
          const double trial_residual = compute_nonlinear_residual_vector(
              field.shape, grid_.phi, source, screen_scale, residual_vec,
              executor);
          const bool residual_reduced =
              std::isfinite(trial_residual) &&
              trial_residual <= residual * (1.0 - 1.0e-4 * trial_lambda);
          if (args_.nl_line_search != "auto" || residual_reduced ||
              trial_residual <= args_.nl_residual) {
            accepted = true;
            accepted_residual = trial_residual;
            accepted_step_norm =
                count > 0 ? std::sqrt(sumsq / static_cast<double>(count)) : 0.0;
            accepted_max_step = local_max_step;
            accepted_clipped = trial_clipped;
            line_search_status = trial_lambda < field.nl_relax
                                     ? "accepted_backtracked"
                                     : "accepted_full";
            last_accepted_lambda = trial_lambda;
            break;
          }
        }
        if (args_.nl_line_search != "auto")
          break;
        trial_lambda *= 0.5;
        ++backtracks;
      }

      if (!finite) {
        trust_psi = std::max(0.05, trust_psi * 0.5);
        line_search_status = "nonfinite";
        stop_reason = "nonfinite";
        break;
      }
      if (!accepted) {
        trust_psi = std::max(0.05, trust_psi * 0.5);
        grid_.phi = nonlinear_before;
        update_nonlinear_source(field.shape, source, alpha, clip);
        std::size_t fallback_clipped = 0;
        SolveStats fallback_stats = solve_newton_nwt_sor_block(
            field.shape, std::max(1, std::min(args_.nl_inner_iter, 50)),
            screen_scale, executor, fallback_clipped);
        const std::size_t fallback_source_clipped =
            update_nonlinear_source(field.shape, source, alpha, clip);
        const double fallback_residual = compute_nonlinear_residual_vector(
            field.shape, grid_.phi, source, screen_scale, residual_vec,
            executor);
        if (std::isfinite(fallback_residual) && fallback_residual < residual) {
          accepted = true;
          residual = fallback_residual;
          last_residual_after = residual;
          last_step_norm = fallback_stats.rms;
          max_step = std::max(max_step, fallback_stats.max_dphi);
          max_clipped = std::max(
              max_clipped, std::max(fallback_clipped, fallback_source_clipped));
          line_search_status = "fallback_nwt_sor";
          last_inner_effective = "nwt_sor_fallback";
          last_inner_cycles = fallback_stats.iterations;
          inner_iterations += fallback_stats.iterations;
          last_stats = fallback_stats;
          last_stats.rms = residual;
          ++total_iterations;
          ++blocks;
          if (residual < best_stage_residual * 0.999) {
            best_stage_residual = residual;
            stagnant_steps = 0;
          } else if (++stagnant_steps >= args_.nl_stagnation) {
            stop_reason = "stagnation";
            step_done = true;
          }
          if (screen_scale >= 1.0 && residual <= args_.nl_residual) {
            converged = true;
            stop_reason = "residual_tol";
            step_done = true;
          } else if (screen_scale < 1.0 &&
                     (residual <= stage_residual_target ||
                      fallback_stats.max_dphi <= args_.nl_dphi)) {
            stop_reason = "stage_target";
            step_done = true;
          }
          continue;
        }
        grid_.phi = nonlinear_before;
        update_nonlinear_source(field.shape, source, alpha, clip);
        compute_nonlinear_residual_vector(field.shape, grid_.phi, source,
                                          screen_scale, residual_vec, executor);
        if (screen_scale < 1.0 && (residual <= stage_residual_target ||
                                   residual < stage_initial_residual)) {
          line_search_status = "stage_target_line_search_floor";
          stop_reason = "stage_target";
          step_done = true;
        } else {
          line_search_status = "line_search_floor";
          stop_reason = "line_search_floor";
        }
        ++rejected_steps;
        ++backtracks;
        break;
      }

      if (args_.nl_line_search == "auto" &&
          last_accepted_lambda >= field.nl_relax &&
          accepted_residual < 0.95 * last_residual_before) {
        field.nl_relax = std::min(1.0, field.nl_relax * 1.25);
        trust_psi = std::min(2.0, trust_psi * 1.25);
      }
      residual = accepted_residual;
      last_residual_after = residual;
      last_step_norm = accepted_step_norm;
      max_step = std::max(max_step, accepted_max_step);
      max_clipped = std::max(max_clipped, accepted_clipped);
      last_stats.rms = residual;
      last_stats.max_dphi = accepted_max_step;
      last_stats.converged = false;
      if (accepted_residual < best_stage_residual * 0.999) {
        best_stage_residual = accepted_residual;
        stagnant_steps = 0;
      } else if (++stagnant_steps >= args_.nl_stagnation) {
        stop_reason = "stagnation";
        step_done = true;
      }
      if (screen_scale >= 1.0 && residual <= args_.nl_residual) {
        last_stats.converged = true;
        converged = true;
        stop_reason = "residual_tol";
        step_done = true;
      } else if (screen_scale < 1.0 && (residual <= stage_residual_target ||
                                        accepted_max_step <= args_.nl_dphi)) {
        stop_reason = "stage_target";
        step_done = true;
      }
    }
    if (!finite || total_iterations >= args_.nl_max_iter ||
        stop_reason == "line_search_floor" ||
        (screen_scale >= 1.0 && (converged || stop_reason == "stagnation")))
      break;
  }
  if (!converged && total_iterations >= args_.nl_max_iter &&
      stop_reason != "line_search_floor" && stop_reason != "nonfinite" &&
      stop_reason != "stagnation")
    stop_reason = "work_budget";

  if (nwt_half_shape_matches(field.shape)) {
    flush_nwt_half_workspace(field.shape);
    const auto source_time_begin = std::chrono::steady_clock::now();
    const std::size_t final_source_clipped =
        update_nonlinear_source(field.shape, source, alpha, clip);
    add_timing_ms(TimingBucket::NwtSourceUpdate,
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - source_time_begin)
                      .count());
    max_clipped = std::max(max_clipped, final_source_clipped);
    const auto residual_time_begin = std::chrono::steady_clock::now();
    residual = compute_nonlinear_residual_vector(field.shape, grid_.phi, source,
                                                 final_screen_scale,
                                                 residual_vec, executor);
    add_timing_ms(TimingBucket::NwtOperatorResidual,
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - residual_time_begin)
                      .count());
    last_residual_after = residual;
  }

  const double reported_nl_residual =
      delphi_nwt_parity_active() ? last_stats.rms : residual;
  field.nl_iterations = total_iterations;
  field.nl_blocks = blocks;
  field.nl_max_dphi = max_step;
  field.nl_residual =
      std::isfinite(reported_nl_residual) ? reported_nl_residual : 0.0;
  field.nl_clipped_points = max_clipped;
  field.nl_continuation_steps = continuation_steps;
  field.nl_backtracks = backtracks;
  field.newton_inner_iterations = inner_iterations;
  field.newton_step_norm = last_step_norm;
  field.newton_residual_before =
      std::isfinite(last_residual_before) ? last_residual_before : 0.0;
  field.newton_residual_after =
      std::isfinite(last_residual_after) ? last_residual_after : 0.0;
  field.newton_line_search_status = line_search_status;
  field.newton_correction_residual =
      std::isfinite(last_correction_residual) ? last_correction_residual : 0.0;
  field.newton_correction_max_residual =
      std::isfinite(last_correction_max_residual) ? last_correction_max_residual
                                                  : 0.0;
  field.newton_accepted_lambda = last_accepted_lambda;
  field.newton_rejected_steps = rejected_steps;
  field.newton_jac_diag_min =
      std::isfinite(last_jac_diag_min) ? last_jac_diag_min : 0.0;
  field.newton_jac_diag_max =
      std::isfinite(last_jac_diag_max) ? last_jac_diag_max : 0.0;
  field.newton_screen_points = last_screen_points;
  field.newton_active_nonlinear_points = last_active_nonlinear_points;
  field.newton_stop_reason = stop_reason;
  field.newton_inner_levels = last_inner_levels;
  field.newton_inner_cycles = last_inner_cycles;
  field.newton_inner_residual_before = std::isfinite(last_inner_residual_before)
                                           ? last_inner_residual_before
                                           : 0.0;
  field.newton_inner_residual_after = std::isfinite(last_inner_residual_after)
                                          ? last_inner_residual_after
                                          : 0.0;
  field.newton_inner_effective = last_inner_effective;
  field.nl_status =
      finite ? (converged
                    ? "ok"
                    : (stop_reason == "stagnation"
                           ? "nonlinear_not_converged_stagnated"
                           : (stop_reason == "line_search_floor"
                                  ? "nonlinear_not_converged_line_search_floor"
                                  : "nonlinear_not_converged")))
             : "nonlinear_nonfinite";
  field.stats = last_stats;
  field.stats.iterations = total_iterations;
  field.stats.converged = finite && converged;
  field.total_iterations += total_iterations;
  field.parity = compute_parity_audit(field.shape, &source, &residual_vec);
  add_timing_ms(
      TimingBucket::Solver,
      elapsed_ms(nonlinear_time_begin, std::chrono::steady_clock::now()));
  update_newton_derivative_audit(field.shape, source, delta, alpha, clip,
                                 final_screen_scale, field.parity);
  if (map_result != nullptr &&
      (args_.parity_dump == "points" || args_.parity_dump == "all")) {
    maybe_write_parity_points(*map_result, field.shape, source, residual_vec,
                              delta, alpha, clip, final_screen_scale);
  }
  scale_nonlinear_field_to_physical_units(grid_.phi, alpha);
  field.atom_phi = interpolate_atoms(field.shape, atoms);
  if (args_.charge_assignment == "delphi" ||
      args_.charge_assignment == "gaussian")
    field.phi_grid = grid_.phi;

  if (map_result != nullptr)
    maybe_write_maps(field.shape, *map_result, true);
  return field;
}

AnalyzerImpl::FieldSolve AnalyzerImpl::solve_nonlinear_field(
    const std::vector<Atom> &atoms, double eps_in, double eps_out, double kappa,
    TitanExecutor &executor, const FrameResult *map_result) {
  if (args_.nl_solver == "newton" || args_.nl_multigrid == "fas")
    return solve_newton_nonlinear_field(atoms, eps_in, eps_out, kappa, executor,
                                        map_result);

  FieldSolve field =
      solve_linear_field(atoms, eps_in, eps_out, kappa, executor, nullptr);
  field.linear_seed_solver = field.solver_mode;
  field.pbe_model_effective = "nonlinear";
  field.nl_solver = "picard";
  field.nl_inner_solver = "sor";
  field.nl_mg_mode = "off";
  field.nl_relax = effective_nl_relax();

  if (kappa <= 0.0 || args_.salt <= 0.0) {
    field.pbe_model_effective = "linear";
    field.nl_status = "linearized_no_salt";
    if (map_result != nullptr) {
      maybe_write_maps(field.shape, *map_result, false);
      maybe_write_maps(field.shape, *map_result, true);
    }
    return field;
  }

  if (map_result != nullptr)
    maybe_write_maps(field.shape, *map_result, false);

  const auto nonlinear_time_begin = std::chrono::steady_clock::now();
  const double alpha = nonlinear_alpha();
  const double clip = effective_nonlinear_clip();
  scale_nonlinear_field_to_solve_units(grid_.phi, alpha);
  AlignedDoubleVector source(field.shape.n, 0.0);
  SolveStats last_stats;
  int total_iterations = 0;
  int blocks = 0;
  double max_block_dphi = 0.0;
  double residual = std::numeric_limits<double>::infinity();
  bool finite = true;
  std::size_t max_clipped = 0;
  int continuation_steps = 0;
  int backtracks = 0;
  std::vector<double> screen_scales;
  if (args_.nl_continuation == "auto" && args_.salt > 0.5)
    screen_scales = {0.25, 0.5, 1.0};
  else if (args_.nl_continuation == "auto")
    screen_scales = {0.5, 1.0};
  else
    screen_scales = {1.0};

  AlignedDoubleVector picard_before;
  for (double screen_scale : screen_scales) {
    ++continuation_steps;
    double previous_residual = std::numeric_limits<double>::infinity();
    bool step_done = false;
    while (total_iterations < args_.nl_max_iter && !step_done) {
      const std::size_t clipped_this =
          update_nonlinear_source(field.shape, source, alpha, clip);
      max_clipped = std::max(max_clipped, clipped_this);

      const int sweeps =
          std::min(args_.nl_block, args_.nl_max_iter - total_iterations);
      if (args_.nl_line_search == "auto")
        picard_before = grid_.phi;
      else
        picard_before.clear();
      std::size_t unused_clipped = 0;
      last_stats =
          solve_nonlinear_sor_block(field.shape, source, sweeps, field.nl_relax,
                                    screen_scale, executor, unused_clipped);
      total_iterations += sweeps;
      ++blocks;
      max_block_dphi = std::max(max_block_dphi, last_stats.max_dphi);

      finite = std::all_of(grid_.phi.begin(), grid_.phi.end(),
                           [](double v) { return std::isfinite(v); });
      if (!finite)
        break;

      update_nonlinear_source(field.shape, source, alpha, clip);
      residual = compute_nonlinear_residual(field.shape, source, screen_scale);
      if (args_.nl_line_search == "auto" && std::isfinite(previous_residual) &&
          residual > previous_residual * 1.05 && field.nl_relax > 0.051 &&
          picard_before.size() == grid_.phi.size()) {
        grid_.phi = picard_before;
        field.nl_relax = std::max(0.05, 0.5 * field.nl_relax);
        ++backtracks;
        residual = previous_residual;
        continue;
      }
      previous_residual = residual;

      if (last_stats.max_dphi <= args_.nl_dphi ||
          residual <= args_.nl_residual) {
        if (screen_scale >= 1.0) {
          last_stats.converged = true;
        }
        step_done = true;
      }
    }
    if (!finite || total_iterations >= args_.nl_max_iter)
      break;
  }

  field.nl_iterations = total_iterations;
  field.nl_blocks = blocks;
  field.nl_max_dphi = max_block_dphi;
  field.nl_residual = std::isfinite(residual) ? residual : 0.0;
  field.nl_clipped_points = max_clipped;
  field.nl_continuation_steps = continuation_steps;
  field.nl_backtracks = backtracks;
  field.nl_status =
      finite ? (last_stats.converged ? "ok" : "nonlinear_not_converged")
             : "nonlinear_nonfinite";
  field.stats = last_stats;
  field.stats.iterations = total_iterations;
  field.stats.converged = finite && last_stats.converged;
  field.total_iterations += total_iterations;
  scale_nonlinear_field_to_physical_units(grid_.phi, alpha);
  field.atom_phi = interpolate_atoms(field.shape, atoms);
  if (args_.charge_assignment == "delphi" ||
      args_.charge_assignment == "gaussian")
    field.phi_grid = grid_.phi;
  add_timing_ms(
      TimingBucket::Solver,
      elapsed_ms(nonlinear_time_begin, std::chrono::steady_clock::now()));

  if (map_result != nullptr)
    maybe_write_maps(field.shape, *map_result, true);
  return field;
}

AnalyzerImpl::FieldSolve AnalyzerImpl::solve_physical_field(
    const std::vector<Atom> &atoms, double eps_in, double eps_out, double kappa,
    TitanExecutor &executor, const FrameResult *map_result) {
  if (args_.model == "nonlinear")
    return solve_nonlinear_field(atoms, eps_in, eps_out, kappa, executor,
                                 map_result);
  return solve_linear_field(atoms, eps_in, eps_out, kappa, executor,
                            map_result);
}


} // namespace TitanPBE
