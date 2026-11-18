#include "tabi_treecode_internal.h"
#include "tabi_near.h"
#include "tabi_radial_taylor.h"
#include "tabi_node_patch_curved.h"

#include "../titan_executor.h"
#include "../titan_simd_dispatch.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TitanPBE::Tabi::detail {
double factorial(int value) noexcept {
  double out = 1.0;
  for (int i = 2; i <= value; ++i)
    out *= static_cast<double>(i);
  return out;
}

const std::vector<MomentIndex> &moment_indices(int order) {
  static const std::vector<MomentIndex> indices = [] {
    std::vector<MomentIndex> out;
    out.reserve(kTaylorMomentMax);
    for (int degree = 0; degree <= kTaylorOrderMax; ++degree) {
      for (int ax = degree; ax >= 0; --ax) {
        for (int ay = degree - ax; ay >= 0; --ay) {
          const int az = degree - ax - ay;
          out.push_back({ax, ay, az,
                         1.0 / (factorial(ax) * factorial(ay) *
                                factorial(az))});
        }
      }
    }
    return out;
  }();
  // One list per order, built once. The far field asks for it on every
  // (target, node) pair, so it must not be rebuilt per call.
  static const auto by_order = [] {
    std::array<std::vector<MomentIndex>, kTaylorOrderMax + 1> out;
    for (int p = 0; p <= kTaylorOrderMax; ++p)
      for (const MomentIndex &index : indices)
        if (index.ax + index.ay + index.az <= p)
          out[static_cast<std::size_t>(p)].push_back(index);
    return out;
  }();
  return by_order[static_cast<std::size_t>(std::clamp(order, 0, kTaylorOrderMax))];
}

double pow_int(double value, int power) noexcept {
  double out = 1.0;
  for (int i = 0; i < power; ++i)
    out *= value;
  return out;
}

double binomial(int n, int k) noexcept {
  if (k < 0 || k > n)
    return 0.0;
  if (k == 0 || k == n)
    return 1.0;
  double out = 1.0;
  for (int i = 1; i <= k; ++i)
    out = out * static_cast<double>(n - k + i) / static_cast<double>(i);
  return out;
}

int moment_lookup_index(int ax, int ay, int az) noexcept {
  constexpr int side = kTaylorOrderMax + 1;
  return (ax * side + ay) * side + az;
}

template <typename Fn> void for_each_multi_index(int order, Fn &&fn) {
  const int clamped = std::clamp(order, 0, kTaylorOrderMax);
  for (int degree = 0; degree <= clamped; ++degree) {
    for (int ax = degree; ax >= 0; --ax) {
      for (int ay = degree - ax; ay >= 0; --ay) {
        const int az = degree - ax - ay;
        fn(ax, ay, az);
      }
    }
  }
}

TaylorSeries series_constant(double value) noexcept {
  TaylorSeries out;
  out.coeff[static_cast<std::size_t>(moment_lookup_index(0, 0, 0))] = value;
  return out;
}

TaylorSeries series_variable(double value, int axis) noexcept {
  TaylorSeries out = series_constant(value);
  const int ax = axis == 0 ? 1 : 0;
  const int ay = axis == 1 ? 1 : 0;
  const int az = axis == 2 ? 1 : 0;
  out.coeff[static_cast<std::size_t>(moment_lookup_index(ax, ay, az))] = 1.0;
  return out;
}

TaylorSeries series_add(const TaylorSeries &a, const TaylorSeries &b,
                        int order) noexcept {
  TaylorSeries out;
  for_each_multi_index(order, [&](int ax, int ay, int az) {
    const std::size_t idx =
        static_cast<std::size_t>(moment_lookup_index(ax, ay, az));
    out.coeff[idx] = a.coeff[idx] + b.coeff[idx];
  });
  return out;
}

TaylorSeries series_sub(const TaylorSeries &a, const TaylorSeries &b,
                        int order) noexcept {
  TaylorSeries out;
  for_each_multi_index(order, [&](int ax, int ay, int az) {
    const std::size_t idx =
        static_cast<std::size_t>(moment_lookup_index(ax, ay, az));
    out.coeff[idx] = a.coeff[idx] - b.coeff[idx];
  });
  return out;
}

TaylorSeries series_scale(const TaylorSeries &a, double scale,
                          int order) noexcept {
  TaylorSeries out;
  for_each_multi_index(order, [&](int ax, int ay, int az) {
    const std::size_t idx =
        static_cast<std::size_t>(moment_lookup_index(ax, ay, az));
    out.coeff[idx] = scale * a.coeff[idx];
  });
  return out;
}

TaylorSeries series_mul(const TaylorSeries &a, const TaylorSeries &b,
                        int order) noexcept {
  TaylorSeries out;
  for_each_multi_index(order, [&](int ax, int ay, int az) {
    double value = 0.0;
    for (int bx = 0; bx <= ax; ++bx) {
      for (int by = 0; by <= ay; ++by) {
        for (int bz = 0; bz <= az; ++bz) {
          const std::size_t lhs =
              static_cast<std::size_t>(moment_lookup_index(bx, by, bz));
          const std::size_t rhs = static_cast<std::size_t>(
              moment_lookup_index(ax - bx, ay - by, az - bz));
          value += a.coeff[lhs] * b.coeff[rhs];
        }
      }
    }
    out.coeff[static_cast<std::size_t>(moment_lookup_index(ax, ay, az))] =
        value;
  });
  return out;
}

TaylorSeries series_reciprocal(const TaylorSeries &a, int order) noexcept {
  TaylorSeries out;
  const std::size_t zero =
      static_cast<std::size_t>(moment_lookup_index(0, 0, 0));
  const double a0 = a.coeff[zero];
  if (!std::isfinite(a0) || std::abs(a0) <= 1.0e-30)
    return out;
  out.coeff[zero] = 1.0 / a0;
  for_each_multi_index(order, [&](int ax, int ay, int az) {
    if (ax == 0 && ay == 0 && az == 0)
      return;
    double sum = 0.0;
    for (int bx = 0; bx <= ax; ++bx) {
      for (int by = 0; by <= ay; ++by) {
        for (int bz = 0; bz <= az; ++bz) {
          if (bx == 0 && by == 0 && bz == 0)
            continue;
          const std::size_t lhs =
              static_cast<std::size_t>(moment_lookup_index(bx, by, bz));
          const std::size_t rhs = static_cast<std::size_t>(
              moment_lookup_index(ax - bx, ay - by, az - bz));
          sum += a.coeff[lhs] * out.coeff[rhs];
        }
      }
    }
    out.coeff[static_cast<std::size_t>(moment_lookup_index(ax, ay, az))] =
        -sum / a0;
  });
  return out;
}

TaylorSeries series_sqrt(const TaylorSeries &a, int order) noexcept {
  TaylorSeries out;
  const std::size_t zero =
      static_cast<std::size_t>(moment_lookup_index(0, 0, 0));
  const double a0 = a.coeff[zero];
  if (!std::isfinite(a0) || a0 <= 0.0)
    return out;
  const double root = std::sqrt(a0);
  out.coeff[zero] = root;
  for_each_multi_index(order, [&](int ax, int ay, int az) {
    if (ax == 0 && ay == 0 && az == 0)
      return;
    double sum = 0.0;
    for (int bx = 0; bx <= ax; ++bx) {
      for (int by = 0; by <= ay; ++by) {
        for (int bz = 0; bz <= az; ++bz) {
          if ((bx == 0 && by == 0 && bz == 0) ||
              (bx == ax && by == ay && bz == az))
            continue;
          const std::size_t lhs =
              static_cast<std::size_t>(moment_lookup_index(bx, by, bz));
          const std::size_t rhs = static_cast<std::size_t>(
              moment_lookup_index(ax - bx, ay - by, az - bz));
          sum += out.coeff[lhs] * out.coeff[rhs];
        }
      }
    }
    const std::size_t idx =
        static_cast<std::size_t>(moment_lookup_index(ax, ay, az));
    out.coeff[idx] = (a.coeff[idx] - sum) / (2.0 * root);
  });
  return out;
}

TaylorSeries series_exp(const TaylorSeries &a, int order) noexcept {
  TaylorSeries out;
  const std::size_t zero =
      static_cast<std::size_t>(moment_lookup_index(0, 0, 0));
  out.coeff[zero] = std::exp(a.coeff[zero]);
  for_each_multi_index(order, [&](int ax, int ay, int az) {
    if (ax == 0 && ay == 0 && az == 0)
      return;
    int axis = 0;
    if (ax == 0)
      axis = ay > 0 ? 1 : 2;
    const int gx = ax - (axis == 0 ? 1 : 0);
    const int gy = ay - (axis == 1 ? 1 : 0);
    const int gz = az - (axis == 2 ? 1 : 0);
    double rhs = 0.0;
    for (int bx = 0; bx <= gx; ++bx) {
      for (int by = 0; by <= gy; ++by) {
        for (int bz = 0; bz <= gz; ++bz) {
          const int dx = gx - bx + (axis == 0 ? 1 : 0);
          const int dy = gy - by + (axis == 1 ? 1 : 0);
          const int dz = gz - bz + (axis == 2 ? 1 : 0);
          const int axis_degree =
              axis == 0 ? dx : (axis == 1 ? dy : dz);
          const std::size_t lhs =
              static_cast<std::size_t>(moment_lookup_index(bx, by, bz));
          const std::size_t deriv =
              static_cast<std::size_t>(moment_lookup_index(dx, dy, dz));
          rhs += out.coeff[lhs] * static_cast<double>(axis_degree) *
                 a.coeff[deriv];
        }
      }
    }
    const int alpha_axis = axis == 0 ? ax : (axis == 1 ? ay : az);
    out.coeff[static_cast<std::size_t>(moment_lookup_index(ax, ay, az))] =
        rhs / static_cast<double>(alpha_axis);
  });
  return out;
}

const std::array<int, (kTaylorOrderMax + 1) * (kTaylorOrderMax + 1) *
                         (kTaylorOrderMax + 1)> &
moment_lookup() {
  static const auto lookup = [] {
    std::array<int, (kTaylorOrderMax + 1) * (kTaylorOrderMax + 1) *
                        (kTaylorOrderMax + 1)>
        out{};
    out.fill(-1);
    const auto &indices = moment_indices(kTaylorOrderMax);
    for (std::size_t i = 0; i < indices.size(); ++i) {
      const MomentIndex &index = indices[i];
      out[static_cast<std::size_t>(
          moment_lookup_index(index.ax, index.ay, index.az))] =
          static_cast<int>(i);
    }
    return out;
  }();
  return lookup;
}

double point_kernel(const Panel &target, double sx, double sy, double sz,
                    double kappa) noexcept {
  const double dx = target.x - sx;
  const double dy = target.y - sy;
  const double dz = target.z - sz;
  const double r2 = dx * dx + dy * dy + dz * dz;
  if (r2 <= 1.0e-16)
    return 0.0;
  const double r = std::sqrt(r2);
  const double normal_projection =
      (target.nx * dx + target.ny * dy + target.nz * dz) / r;
  const double screened = std::exp(-std::max(0.0, kappa) * r);
  return screened * normal_projection /
         (4.0 * kPi * std::max(1.0e-12, r2));
}

double panel_kernel(const Panel &target, const Panel &source,
                    double kappa) noexcept {
  return source.area * point_kernel(target, source.x, source.y, source.z,
                                    kappa);
}

KernelSeriesBase make_kernel_series_base(double tx, double ty, double tz,
                                         double sx, double sy, double sz,
                                         double kappa, int order) noexcept {
  KernelSeriesBase base;
  const TaylorSeries source_x = series_variable(sx, 0);
  const TaylorSeries source_y = series_variable(sy, 1);
  const TaylorSeries source_z = series_variable(sz, 2);
  base.ux = series_sub(series_constant(tx), source_x, order);
  base.uy = series_sub(series_constant(ty), source_y, order);
  base.uz = series_sub(series_constant(tz), source_z, order);
  const TaylorSeries r2 = series_add(
      series_add(series_mul(base.ux, base.ux, order),
                 series_mul(base.uy, base.uy, order), order),
      series_mul(base.uz, base.uz, order), order);
  const double r2_zero =
      r2.coeff[static_cast<std::size_t>(moment_lookup_index(0, 0, 0))];
  if (!std::isfinite(r2_zero) || r2_zero <= 1.0e-28)
    return base;
  base.r = series_sqrt(r2, order);
  base.inv_r = series_reciprocal(base.r, order);
  base.inv_r2 = series_mul(base.inv_r, base.inv_r, order);
  base.inv_r3 = series_mul(base.inv_r2, base.inv_r, order);
  base.screened = series_exp(series_scale(base.r, -std::max(0.0, kappa),
                                          order),
                             order);
  return base;
}

TaylorSeries green_series(double tx, double ty, double tz, double sx,
                          double sy, double sz, double kappa,
                          int order) noexcept {
  const KernelSeriesBase base =
      make_kernel_series_base(tx, ty, tz, sx, sy, sz, kappa, order);
  return series_scale(series_mul(base.screened, base.inv_r, order),
                      1.0 / (4.0 * kPi), order);
}

TaylorSeries source_normal_component_green_series(double tx, double ty,
                                                  double tz, double sx,
                                                  double sy, double sz,
                                                  int component, double kappa,
                                                  int order) noexcept {
  const double kk = std::max(0.0, kappa);
  const KernelSeriesBase base =
      make_kernel_series_base(tx, ty, tz, sx, sy, sz, kk, order);
  const TaylorSeries one_plus_kr =
      series_add(series_constant(1.0), series_scale(base.r, kk, order), order);
  const TaylorSeries &uc =
      component == 0 ? base.ux : (component == 1 ? base.uy : base.uz);
  TaylorSeries value =
      series_mul(series_mul(base.screened, one_plus_kr, order),
                 series_mul(uc, base.inv_r3, order), order);
  return series_scale(value, 1.0 / (4.0 * kPi), order);
}

TaylorSeries target_normal_green_series(const Panel &target, double sx,
                                        double sy, double sz, double kappa,
                                        int order) noexcept {
  const double kk = std::max(0.0, kappa);
  const KernelSeriesBase base = make_kernel_series_base(
      target.x, target.y, target.z, sx, sy, sz, kk, order);
  const TaylorSeries normal_dot_u = series_add(
      series_add(series_scale(base.ux, target.nx, order),
                 series_scale(base.uy, target.ny, order), order),
      series_scale(base.uz, target.nz, order), order);
  const TaylorSeries one_plus_kr =
      series_add(series_constant(1.0), series_scale(base.r, kk, order), order);
  TaylorSeries value =
      series_mul(series_mul(base.screened, one_plus_kr, order),
                 series_mul(normal_dot_u, base.inv_r3, order), order);
  return series_scale(value, -1.0 / (4.0 * kPi), order);
}

TaylorSeries mixed_normal_component_green_series(const Panel &target,
                                                 double sx, double sy,
                                                 double sz, int component,
                                                 double kappa,
                                                 int order) noexcept {
  const double kk = std::max(0.0, kappa);
  const KernelSeriesBase base = make_kernel_series_base(
      target.x, target.y, target.z, sx, sy, sz, kk, order);
  const TaylorSeries normal_dot_u = series_add(
      series_add(series_scale(base.ux, target.nx, order),
                 series_scale(base.uy, target.ny, order), order),
      series_scale(base.uz, target.nz, order), order);
  const TaylorSeries nxu = series_mul(normal_dot_u, base.inv_r, order);
  const TaylorSeries &uc =
      component == 0 ? base.ux : (component == 1 ? base.uy : base.uz);
  const TaylorSeries nyu = series_mul(uc, base.inv_r, order);
  const double nxny =
      component == 0 ? target.nx : (component == 1 ? target.ny : target.nz);
  const TaylorSeries one_plus_kr =
      series_add(series_constant(1.0), series_scale(base.r, kk, order), order);
  const TaylorSeries fprime = series_scale(
      series_mul(series_mul(base.screened, one_plus_kr, order), base.inv_r2,
                 order),
      -1.0 / (4.0 * kPi), order);
  TaylorSeries fsecond_radial = series_add(
      series_add(series_scale(base.inv_r, kk * kk, order),
                 series_scale(base.inv_r2, 2.0 * kk, order), order),
      series_scale(base.inv_r3, 2.0, order), order);
  const TaylorSeries fsecond = series_scale(
      series_mul(base.screened, fsecond_radial, order), 1.0 / (4.0 * kPi),
      order);
  const TaylorSeries nxu_nyu = series_mul(nxu, nyu, order);
  const TaylorSeries metric =
      series_sub(series_constant(nxny), nxu_nyu, order);
  // Source differentiation supplies the minus sign for every Taylor term.
  return series_scale(
      series_add(series_mul(fsecond, nxu_nyu, order),
                 series_mul(series_mul(fprime, base.inv_r, order), metric,
                            order),
                 order),
      -1.0, order);
}

TaylorSeries point_kernel_series(const Panel &target, double sx, double sy,
                                 double sz, double kappa,
                                 int order) noexcept {
  const KernelSeriesBase base = make_kernel_series_base(
      target.x, target.y, target.z, sx, sy, sz, kappa, order);
  const TaylorSeries normal_dot_u = series_add(
      series_add(series_scale(base.ux, target.nx, order),
                 series_scale(base.uy, target.ny, order), order),
      series_scale(base.uz, target.nz, order), order);
  return series_scale(series_mul(base.screened,
                                 series_mul(normal_dot_u, base.inv_r3, order),
                                 order),
                      1.0 / (4.0 * kPi), order);
}

double taylor_sum_series(const NodeMoments &moments, int order,
                         const TaylorSeries &series) noexcept {
  const auto &indices = moment_indices(order);
  double value = 0.0;
  for (std::size_t m = 0; m < indices.size(); ++m) {
    const MomentIndex &index = indices[m];
    const std::size_t coeff =
        static_cast<std::size_t>(moment_lookup_index(index.ax, index.ay,
                                                     index.az));
    value += moments.values[m] * series.coeff[coeff];
  }
  return std::isfinite(value) ? value : 0.0;
}

double finite_difference_derivative(const Panel &target, double cx, double cy,
                                    double cz, int ax, int ay, int az,
                                    double h, double kappa) noexcept {
  if (ax == 0 && ay == 0 && az == 0)
    return point_kernel(target, cx, cy, cz, kappa);
  if (ax > 0) {
    return (finite_difference_derivative(target, cx + h, cy, cz, ax - 1, ay,
                                         az, h, kappa) -
            finite_difference_derivative(target, cx - h, cy, cz, ax - 1, ay,
                                         az, h, kappa)) /
           (2.0 * h);
  }
  if (ay > 0) {
    return (finite_difference_derivative(target, cx, cy + h, cz, ax, ay - 1,
                                         az, h, kappa) -
            finite_difference_derivative(target, cx, cy - h, cz, ax, ay - 1,
                                         az, h, kappa)) /
           (2.0 * h);
  }
  return (finite_difference_derivative(target, cx, cy, cz + h, ax, ay, az - 1,
                                       h, kappa) -
          finite_difference_derivative(target, cx, cy, cz - h, ax, ay, az - 1,
                                       h, kappa)) /
         (2.0 * h);
}

double node_kernel_taylor(const Panel &target, const TreeNode &node,
                          const NodeMoments &moments, int order,
                          double kappa,
                          const std::string &derivative_mode) noexcept {
  if (derivative_mode != "finite-diff-debug") {
    return taylor_sum_series(
        moments, order,
        point_kernel_series(target, node.cx, node.cy, node.cz, kappa, order));
  }
  const double dx = target.x - node.cx;
  const double dy = target.y - node.cy;
  const double dz = target.z - node.cz;
  const double distance =
      std::sqrt(std::max(1.0e-16, dx * dx + dy * dy + dz * dz));
  const double h = std::clamp(0.05 * distance, 1.0e-4,
                              std::max(1.0e-3, 0.20 * node.radius));
  const auto &indices = moment_indices(order);
  double value = 0.0;
  for (std::size_t m = 0; m < indices.size(); ++m) {
    const MomentIndex &index = indices[m];
    const double derivative = finite_difference_derivative(
        target, node.cx, node.cy, node.cz, index.ax, index.ay, index.az, h,
        kappa);
    value += moments.values[m] * derivative * index.inv_factorial;
  }
  return std::isfinite(value) ? value : 0.0;
}

double direct_sum_scalar(const Panel &target, std::size_t target_index,
                         const std::vector<Panel> &panels,
                         const std::vector<double> &density, double kappa,
                         std::size_t begin, std::size_t end) noexcept {
  double coupled = 0.0;
  for (std::size_t j = begin; j < end; ++j) {
    if (j == target_index)
      continue;
    coupled += panel_kernel(target, panels[j], kappa) * density[j];
  }
  return coupled;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
double direct_sum_avx2(const Panel &target, std::size_t target_index,
                       const std::vector<Panel> &panels,
                       const std::vector<double> &density, double kappa,
                       std::size_t begin, std::size_t end) noexcept {
  double coupled = 0.0;
  const __m256d tx = _mm256_set1_pd(target.x);
  const __m256d ty = _mm256_set1_pd(target.y);
  const __m256d tz = _mm256_set1_pd(target.z);
  const __m256d tnx = _mm256_set1_pd(target.nx);
  const __m256d tny = _mm256_set1_pd(target.ny);
  const __m256d tnz = _mm256_set1_pd(target.nz);
  const __m256d four_pi = _mm256_set1_pd(4.0 * kPi);
  const __m256d tiny_r2 = _mm256_set1_pd(1.0e-16);
  const __m256d min_denom = _mm256_set1_pd(1.0e-12);
  const double kk = std::max(0.0, kappa);

  std::size_t j = begin;
  alignas(32) double base_lane[4];
  alignas(32) double r_lane[4];
  for (; j + 4 <= end; j += 4) {
    const __m256d sx =
        _mm256_set_pd(panels[j + 3].x, panels[j + 2].x, panels[j + 1].x,
                      panels[j].x);
    const __m256d sy =
        _mm256_set_pd(panels[j + 3].y, panels[j + 2].y, panels[j + 1].y,
                      panels[j].y);
    const __m256d sz =
        _mm256_set_pd(panels[j + 3].z, panels[j + 2].z, panels[j + 1].z,
                      panels[j].z);
    const __m256d area =
        _mm256_set_pd(panels[j + 3].area, panels[j + 2].area,
                      panels[j + 1].area, panels[j].area);
    const __m256d dx = _mm256_sub_pd(tx, sx);
    const __m256d dy = _mm256_sub_pd(ty, sy);
    const __m256d dz = _mm256_sub_pd(tz, sz);
    __m256d r2 = _mm256_add_pd(_mm256_mul_pd(dx, dx),
                               _mm256_add_pd(_mm256_mul_pd(dy, dy),
                                             _mm256_mul_pd(dz, dz)));
    const __m256d valid = _mm256_cmp_pd(r2, tiny_r2, _CMP_GT_OQ);
    const __m256d r = _mm256_sqrt_pd(r2);
    const __m256d dot =
        _mm256_add_pd(_mm256_mul_pd(tnx, dx),
                      _mm256_add_pd(_mm256_mul_pd(tny, dy),
                                    _mm256_mul_pd(tnz, dz)));
    const __m256d normal = _mm256_div_pd(dot, r);
    const __m256d denom = _mm256_mul_pd(four_pi, _mm256_max_pd(r2, min_denom));
    __m256d base = _mm256_div_pd(_mm256_mul_pd(area, normal), denom);
    base = _mm256_and_pd(base, valid);
    _mm256_store_pd(base_lane, base);
    _mm256_store_pd(r_lane, r);
    for (int lane = 0; lane < 4; ++lane) {
      const std::size_t index = j + static_cast<std::size_t>(lane);
      if (index == target_index)
        continue;
      coupled += base_lane[lane] * std::exp(-kk * r_lane[lane]) *
                 density[index];
    }
  }
  for (; j < end; ++j) {
    if (j == target_index)
      continue;
    coupled += panel_kernel(target, panels[j], kappa) * density[j];
  }
  return coupled;
}
#endif

double direct_sum(const Panel &target, std::size_t target_index,
                  const std::vector<Panel> &panels,
                  const std::vector<double> &density, double kappa,
                  std::size_t begin, std::size_t end) noexcept {
  using Kernel = double (*)(const Panel &, std::size_t,
                            const std::vector<Panel> &,
                            const std::vector<double> &, double, std::size_t,
                            std::size_t) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return titan_simd_dispatch::runtime_has_avx2() ? &direct_sum_avx2
                                                   : &direct_sum_scalar;
#else
    return &direct_sum_scalar;
#endif
  }();
  return kernel(target, target_index, panels, density, kappa, begin, end);
}

double direct_ordered_sum(const Panel &target, std::size_t target_index,
                          const std::vector<Panel> &panels,
                          const std::vector<double> &density, double kappa,
                          const std::vector<std::size_t> &order,
                          std::size_t begin, std::size_t count,
                          std::size_t &near_count) noexcept {
  double coupled = 0.0;
  const std::size_t end = begin + count;
  for (std::size_t pos = begin; pos < end; ++pos) {
    const std::size_t j = order[pos];
    if (j == target_index)
      continue;
    coupled += panel_kernel(target, panels[j], kappa) * density[j];
    ++near_count;
  }
  return coupled;
}

double green_value(double tx, double ty, double tz, double sx, double sy,
                   double sz, double kappa) noexcept {
  const double dx = tx - sx;
  const double dy = ty - sy;
  const double dz = tz - sz;
  const double r2 = dx * dx + dy * dy + dz * dz;
  if (r2 <= 1.0e-28)
    return 0.0;
  const double r = std::sqrt(r2);
  return std::exp(-std::max(0.0, kappa) * r) / (4.0 * kPi * r);
}

double source_normal_component_green(double tx, double ty, double tz,
                                     double sx, double sy, double sz,
                                     int component,
                                     double kappa) noexcept {
  const double ux = tx - sx;
  const double uy = ty - sy;
  const double uz = tz - sz;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (r2 <= 1.0e-28)
    return 0.0;
  const double r = std::sqrt(r2);
  const double n_dot_u = component == 0 ? ux : (component == 1 ? uy : uz);
  return std::exp(-std::max(0.0, kappa) * r) *
         (1.0 + std::max(0.0, kappa) * r) * n_dot_u /
         (4.0 * kPi * r2 * r);
}

double target_normal_green(const Panel &target, double sx, double sy,
                           double sz, double kappa) noexcept {
  const double ux = target.x - sx;
  const double uy = target.y - sy;
  const double uz = target.z - sz;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (r2 <= 1.0e-28)
    return 0.0;
  const double r = std::sqrt(r2);
  const double n_dot_u = target.nx * ux + target.ny * uy + target.nz * uz;
  const double kk = std::max(0.0, kappa);
  return -std::exp(-kk * r) * (1.0 + kk * r) * n_dot_u /
         (4.0 * kPi * r2 * r);
}

double mixed_normal_component_green(const Panel &target, double sx, double sy,
                                    double sz, int component,
                                    double kappa) noexcept {
  const double ux = target.x - sx;
  const double uy = target.y - sy;
  const double uz = target.z - sz;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (r2 <= 1.0e-28)
    return 0.0;
  const double r = std::sqrt(r2);
  const double inv_r = 1.0 / r;
  const double nxu =
      (target.nx * ux + target.ny * uy + target.nz * uz) * inv_r;
  const double nyu =
      (component == 0 ? ux : (component == 1 ? uy : uz)) * inv_r;
  const double nxny =
      component == 0 ? target.nx : (component == 1 ? target.ny : target.nz);
  const double kk = std::max(0.0, kappa);
  const double expkr = std::exp(-kk * r);
  const double inv_r2 = inv_r * inv_r;
  const double fprime =
      -expkr * (1.0 + kk * r) * inv_r2 / (4.0 * kPi);
  const double fsecond =
      expkr *
      (kk * kk * inv_r + 2.0 * kk * inv_r2 + 2.0 * inv_r2 * inv_r) /
      (4.0 * kPi);
  return -fsecond * nxu * nyu -
         (fprime * inv_r) * (nxny - nxu * nyu);
}

template <typename Kernel>
double finite_difference_source_derivative(const Kernel &kernel, double sx,
                                           double sy, double sz, int ax,
                                           int ay, int az, double h) noexcept {
  if (ax == 0 && ay == 0 && az == 0)
    return kernel(sx, sy, sz);
  if (ax > 0) {
    return (finite_difference_source_derivative(kernel, sx + h, sy, sz,
                                                ax - 1, ay, az, h) -
            finite_difference_source_derivative(kernel, sx - h, sy, sz,
                                                ax - 1, ay, az, h)) /
           (2.0 * h);
  }
  if (ay > 0) {
    return (finite_difference_source_derivative(kernel, sx, sy + h, sz, ax,
                                                ay - 1, az, h) -
            finite_difference_source_derivative(kernel, sx, sy - h, sz, ax,
                                                ay - 1, az, h)) /
           (2.0 * h);
  }
  return (finite_difference_source_derivative(kernel, sx, sy, sz + h, ax, ay,
                                              az - 1, h) -
          finite_difference_source_derivative(kernel, sx, sy, sz - h, ax, ay,
                                              az - 1, h)) /
         (2.0 * h);
}

template <typename Kernel>
double taylor_sum_kernel(const TreeNode &node, const NodeMoments &moments,
                         int order, double h, const Kernel &kernel) noexcept {
  const auto &indices = moment_indices(order);
  double value = 0.0;
  for (std::size_t m = 0; m < indices.size(); ++m) {
    const MomentIndex &index = indices[m];
    const double derivative = finite_difference_source_derivative(
        kernel, node.cx, node.cy, node.cz, index.ax, index.ay, index.az, h);
    value += moments.values[m] * derivative * index.inv_factorial;
  }
  return std::isfinite(value) ? value : 0.0;
}

JcpAccum sum_jcp_channels(const JcpChannelAccum &channels) noexcept {
  JcpAccum total;
  for (const JcpAccum &entry : channels.channel) {
    total.phi += entry.phi;
    total.phin += entry.phin;
  }
  return total;
}



void kahan_add(double &sum, double &compensation, double value) noexcept {
  const double y = value - compensation;
  const double t = sum + y;
  compensation = (t - sum) - y;
  sum = t;
}

void accumulate_leaf_moment(
    NodeMoments &moment, const Panel &panel, const TreeNode &node,
    double weight, const std::vector<MomentIndex> &indices,
    std::array<double, kTaylorMomentMax> *compensation = nullptr) {
  const double dx = panel.x - node.cx;
  const double dy = panel.y - node.cy;
  const double dz = panel.z - node.cz;
  for (std::size_t m = 0; m < indices.size(); ++m) {
    const MomentIndex &index = indices[m];
    const double value = weight * pow_int(dx, index.ax) *
                         pow_int(dy, index.ay) * pow_int(dz, index.az);
    if (compensation != nullptr)
      kahan_add(moment.values[m], (*compensation)[m], value);
    else
      moment.values[m] += value;
  }
}

void translate_child_moment(NodeMoments &parent, const NodeMoments &child,
                            const TreeNode &parent_node,
                            const TreeNode &child_node,
                            const std::vector<MomentIndex> &indices,
                            const std::array<int, kSeriesSize> &lookup,
                            std::array<double, kTaylorMomentMax>
                                *compensation = nullptr) {
  const double ox = child_node.cx - parent_node.cx;
  const double oy = child_node.cy - parent_node.cy;
  const double oz = child_node.cz - parent_node.cz;
  for (std::size_t alpha_index = 0; alpha_index < indices.size();
       ++alpha_index) {
    const MomentIndex &alpha = indices[alpha_index];
    double translated = 0.0;
    for (int bx = 0; bx <= alpha.ax; ++bx) {
      for (int by = 0; by <= alpha.ay; ++by) {
        for (int bz = 0; bz <= alpha.az; ++bz) {
          const int beta_index =
              lookup[static_cast<std::size_t>(moment_lookup_index(bx, by, bz))];
          if (beta_index < 0)
            continue;
          const double coeff =
              binomial(alpha.ax, bx) * binomial(alpha.ay, by) *
              binomial(alpha.az, bz) * pow_int(ox, alpha.ax - bx) *
              pow_int(oy, alpha.ay - by) * pow_int(oz, alpha.az - bz);
          translated +=
              coeff * child.values[static_cast<std::size_t>(beta_index)];
        }
      }
    }
    if (compensation != nullptr)
      kahan_add(parent.values[alpha_index], (*compensation)[alpha_index],
                translated);
    else
      parent.values[alpha_index] += translated;
  }
}

std::vector<NodeMoments>
compute_node_moments(const std::vector<Panel> &panels,
                     const std::vector<double> &density,
                     const TreeWorkspace &tree, int taylor_order,
                     TitanExecutor &executor) {
  std::vector<NodeMoments> moments(tree.nodes.size());
  const auto &indices = moment_indices(taylor_order);
  const auto &lookup = moment_lookup();
  const bool use_kahan = tree.moment_sum == "kahan";
  if (tree.moment_levels.empty()) {
    for (std::size_t node_index = 0; node_index < tree.nodes.size();
         ++node_index) {
      const TreeNode &node = tree.nodes[node_index];
      if (node.child_count == 0) {
        std::array<double, kTaylorMomentMax> compensation{};
        for (std::size_t pos = node.first_panel;
             pos < node.first_panel + node.panel_count; ++pos) {
          const std::size_t panel_index = tree.order[pos];
          const Panel &panel = panels[panel_index];
          accumulate_leaf_moment(moments[node_index], panel, node,
                                 panel.area * density[panel_index], indices,
                                 use_kahan ? &compensation : nullptr);
        }
      } else {
        std::array<double, kTaylorMomentMax> compensation{};
        for (std::size_t child = 0; child < node.child_count; ++child) {
          const std::size_t child_index = child_node_index(tree, node, child);
          if (child_index >= tree.nodes.size())
            continue;
          translate_child_moment(moments[node_index], moments[child_index],
                                 node, tree.nodes[child_index], indices,
                                 lookup,
                                 use_kahan ? &compensation : nullptr);
        }
      }
    }
    return moments;
  }

  const auto leaf_level = tree.moment_levels.front();
  const std::size_t leaf_count = leaf_level.second - leaf_level.first;
  const std::size_t leaf_workers = effective_workers(executor, leaf_count);
  executor.parallel_for(
      leaf_count, leaf_workers, 16,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t offset = begin; offset < end; ++offset) {
          const std::size_t node_index = leaf_level.first + offset;
          const TreeNode &node = tree.nodes[node_index];
          std::array<double, kTaylorMomentMax> compensation{};
          for (std::size_t pos = node.first_panel;
               pos < node.first_panel + node.panel_count; ++pos) {
            const std::size_t panel_index = tree.order[pos];
            const Panel &panel = panels[panel_index];
            accumulate_leaf_moment(moments[node_index], panel, node,
                                   panel.area * density[panel_index], indices,
                                   use_kahan ? &compensation : nullptr);
          }
        }
      });

  for (std::size_t level = 1; level < tree.moment_levels.size(); ++level) {
    const auto range = tree.moment_levels[level];
    const std::size_t count = range.second - range.first;
    const std::size_t workers = effective_workers(executor, count);
    executor.parallel_for(
        count, workers, 16,
        [&](std::size_t, std::size_t begin, std::size_t end) {
          for (std::size_t offset = begin; offset < end; ++offset) {
            const std::size_t node_index = range.first + offset;
            const TreeNode &node = tree.nodes[node_index];
            std::array<double, kTaylorMomentMax> compensation{};
            for (std::size_t child = 0; child < node.child_count; ++child) {
              const std::size_t child_index =
                  child_node_index(tree, node, child);
              if (child_index >= tree.nodes.size())
                continue;
              translate_child_moment(moments[node_index],
                                     moments[child_index], node,
                                     tree.nodes[child_index], indices, lookup,
                                     use_kahan ? &compensation : nullptr);
            }
          }
        });
  }
  return moments;
}

void accumulate_jcp_panel_moments(
    PackedJcpMoments &packed, std::size_t node_index, const Panel &panel,
    std::size_t panel_index, std::size_t panel_count,
    const std::vector<double> &unknowns, bool coupled, const TreeNode &node,
    const std::vector<MomentIndex> &indices,
    std::array<double, kTaylorMomentMax> *phin_comp,
    std::array<std::array<double, kTaylorMomentMax>, 3> *normal_comp) {
  if (const auto*cell=curved_dual_cell(panel)) {
    // The curved cell builds its moments only off the FMM route.
    if(cell->moments.size()!=4)
      throw std::runtime_error("curved_dual_cell_moments_not_built");
    const double phi=panel_index<unknowns.size()?unknowns[panel_index]:0.0;
    const double phin=coupled&&panel_count+panel_index<unknowns.size()?unknowns[panel_count+panel_index]:0.0;
    const double ox=cell->center[0]-node.cx,oy=cell->center[1]-node.cy,oz=cell->center[2]-node.cz;
    for(std::size_t m=0;m<indices.size();++m){const auto&a=indices[m];std::array<double,4> translated{};
      for(int bx=0;bx<=a.ax;++bx)for(int by=0;by<=a.ay;++by)for(int bz=0;bz<=a.az;++bz){
        const double coefficient=binomial(a.ax,bx)*binomial(a.ay,by)*binomial(a.az,bz)*pow_int(ox,a.ax-bx)*pow_int(oy,a.ay-by)*pow_int(oz,a.az-bz);
        for(int c=0;c<4;++c)translated[c]+=coefficient*cell->moments[c][kMomentSlot[bx*36+by*6+bz]];}
      for(int c=0;c<3;++c){auto&v=packed.phi_normal[c][node_index].values[m];double value=phi*translated[c+1];
        if(normal_comp)kahan_add(v,(*normal_comp)[c][m],value);else v+=value;}
      if(coupled){auto&v=packed.phin[node_index].values[m];double value=phin*translated[0];if(phin_comp)kahan_add(v,(*phin_comp)[m],value);else v+=value;}
    }
    return;
  }
  const double phi =
      panel_index < unknowns.size() ? unknowns[panel_index] : 0.0;
  const double area_phi = panel.area * phi;
  accumulate_leaf_moment(
      packed.phi_normal[0][node_index], panel, node, area_phi * panel.nx,
      indices, normal_comp != nullptr ? &(*normal_comp)[0] : nullptr);
  accumulate_leaf_moment(
      packed.phi_normal[1][node_index], panel, node, area_phi * panel.ny,
      indices, normal_comp != nullptr ? &(*normal_comp)[1] : nullptr);
  accumulate_leaf_moment(
      packed.phi_normal[2][node_index], panel, node, area_phi * panel.nz,
      indices, normal_comp != nullptr ? &(*normal_comp)[2] : nullptr);
  if (coupled && panel_count + panel_index < unknowns.size()) {
    accumulate_leaf_moment(packed.phin[node_index], panel, node,
                           panel.area * unknowns[panel_count + panel_index],
                           indices, phin_comp);
  }
}

double convergence_norm_by_degree(const NodeMoments &moments,
                                  const std::vector<MomentIndex> &indices,
                                  int degree, double radius) noexcept {
  double out = 0.0;
  const double safe_radius = std::max(1.0e-12, radius);
  const double scale = pow_int(safe_radius, degree);
  for (std::size_t m = 0; m < indices.size(); ++m) {
    const MomentIndex &index = indices[m];
    if (index.ax + index.ay + index.az != degree)
      continue;
    out = std::max(out, std::abs(moments.values[m]) /
                            std::max(1.0e-300, scale));
  }
  return out;
}

void observe_convergence_norms(const NodeMoments &moments,
                               const std::vector<MomentIndex> &indices,
                               int taylor_order, double radius,
                               double &source_norm, double &tail_norm_p,
                               double &tail_norm_pm1) noexcept {
  source_norm =
      std::max(source_norm, convergence_norm_by_degree(moments, indices, 0,
                                                       radius));
  source_norm =
      std::max(source_norm, convergence_norm_by_degree(moments, indices, 1,
                                                       radius));
  tail_norm_p =
      std::max(tail_norm_p, convergence_norm_by_degree(
                                moments, indices, taylor_order, radius));
  tail_norm_pm1 =
      std::max(tail_norm_pm1,
               convergence_norm_by_degree(moments, indices, taylor_order - 1,
                                          radius));
}

void precompute_jcp_convergence_norms(PackedJcpMoments &packed,
                                      const TreeWorkspace &tree,
                                      int taylor_order,
                                      TitanExecutor &executor) {
  packed.convergence_norms.assign(tree.nodes.size(), JcpConvergenceNorms{});
  if (tree.nodes.empty() || taylor_order <= 0)
    return;
  const auto &indices = moment_indices(taylor_order);
  if (indices.empty())
    return;
  const std::size_t node_count = tree.nodes.size();
  const std::size_t workers = effective_workers(executor, node_count);
  executor.parallel_for(
      node_count, workers, 16,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t node_index = begin; node_index < end; ++node_index) {
          const TreeNode &node = tree.nodes[node_index];
          JcpConvergenceNorms norms;
          norms.valid = true;
          norms.taylor_order = taylor_order;
          if (node_index < packed.phin.size()) {
            observe_convergence_norms(
                packed.phin[node_index], indices, taylor_order, node.radius,
                norms.scalar_source_norm, norms.scalar_tail_norm_p,
                norms.scalar_tail_norm_pm1);
            observe_convergence_norms(
                packed.phin[node_index], indices, taylor_order, node.radius,
                norms.channel_source_norm[kJcpK1PhiChannel],
                norms.channel_tail_norm_p[kJcpK1PhiChannel],
                norms.channel_tail_norm_pm1[kJcpK1PhiChannel]);
            observe_convergence_norms(
                packed.phin[node_index], indices, taylor_order, node.radius,
                norms.channel_source_norm[kJcpK3PhinChannel],
                norms.channel_tail_norm_p[kJcpK3PhinChannel],
                norms.channel_tail_norm_pm1[kJcpK3PhinChannel]);
          }
          if (node_index < packed.phi_normal[0].size()) {
            for (int component = 0; component < 3; ++component) {
              const NodeMoments &moments =
                  packed.phi_normal[static_cast<std::size_t>(component)]
                                    [node_index];
              observe_convergence_norms(
                  moments, indices, taylor_order, node.radius,
                  norms.normal_source_norm, norms.normal_tail_norm_p,
                  norms.normal_tail_norm_pm1);
              observe_convergence_norms(
                  moments, indices, taylor_order, node.radius,
                  norms.channel_source_norm[kJcpK2PhiNormalChannel],
                  norms.channel_tail_norm_p[kJcpK2PhiNormalChannel],
                  norms.channel_tail_norm_pm1[kJcpK2PhiNormalChannel]);
              observe_convergence_norms(
                  moments, indices, taylor_order, node.radius,
                  norms.channel_source_norm[kJcpK4MixedChannel],
                  norms.channel_tail_norm_p[kJcpK4MixedChannel],
                  norms.channel_tail_norm_pm1[kJcpK4MixedChannel]);
            }
          }
          packed.convergence_norms[node_index] = norms;
        }
      });
}

PackedJcpMoments compute_jcp_node_moments_packed(
    const std::vector<Panel> &panels, const std::vector<double> &unknowns,
    bool coupled, const TreeWorkspace &tree, int taylor_order,
    TitanExecutor &executor) {
  PackedJcpMoments packed;
  packed.phin.assign(tree.nodes.size(), NodeMoments{});
  for (auto &channel : packed.phi_normal)
    channel.assign(tree.nodes.size(), NodeMoments{});
  packed.convergence_norms.assign(tree.nodes.size(), JcpConvergenceNorms{});
  const auto &indices = moment_indices(taylor_order);
  const auto &lookup = moment_lookup();
  if (tree.moment_levels.empty())
    return packed;
  const bool use_kahan = tree.moment_sum == "kahan";
  if (tree.moment_construction == "direct-node") {
    const std::size_t node_count = tree.nodes.size();
    const std::size_t workers = effective_workers(executor, node_count);
    const std::size_t n = panels.size();
    executor.parallel_for(
        node_count, workers, 8,
        [&](std::size_t, std::size_t begin, std::size_t end) {
          for (std::size_t node_index = begin; node_index < end; ++node_index) {
            const TreeNode &node = tree.nodes[node_index];
            std::array<double, kTaylorMomentMax> phin_comp{};
            std::array<std::array<double, kTaylorMomentMax>, 3> normal_comp{};
            const std::size_t last =
                std::min(tree.order.size(), node.first_panel + node.panel_count);
            for (std::size_t pos = node.first_panel; pos < last; ++pos) {
              const std::size_t panel_index = tree.order[pos];
              if (panel_index >= panels.size())
                continue;
              accumulate_jcp_panel_moments(
                  packed, node_index, panels[panel_index], panel_index, n,
                  unknowns, coupled, node, indices,
                  use_kahan ? &phin_comp : nullptr,
                  use_kahan ? &normal_comp : nullptr);
            }
          }
        });
    precompute_jcp_convergence_norms(packed, tree, taylor_order, executor);
    return packed;
  }

  const auto leaf_level = tree.moment_levels.front();
  const std::size_t n = panels.size();
  const std::size_t leaf_count = leaf_level.second - leaf_level.first;
  const std::size_t leaf_workers = effective_workers(executor, leaf_count);
  executor.parallel_for(
      leaf_count, leaf_workers, 16,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (std::size_t offset = begin; offset < end; ++offset) {
          const std::size_t node_index = leaf_level.first + offset;
          const TreeNode &node = tree.nodes[node_index];
          std::array<double, kTaylorMomentMax> phin_comp{};
          std::array<std::array<double, kTaylorMomentMax>, 3> normal_comp{};
          for (std::size_t pos = node.first_panel;
               pos < node.first_panel + node.panel_count; ++pos) {
            const std::size_t panel_index = tree.order[pos];
            const Panel &panel = panels[panel_index];
            accumulate_jcp_panel_moments(
                packed, node_index, panel, panel_index, n, unknowns, coupled,
                node, indices, use_kahan ? &phin_comp : nullptr,
                use_kahan ? &normal_comp : nullptr);
          }
        }
      });

  for (std::size_t level = 1; level < tree.moment_levels.size(); ++level) {
    const auto range = tree.moment_levels[level];
    const std::size_t count = range.second - range.first;
    const std::size_t workers = effective_workers(executor, count);
    executor.parallel_for(
        count, workers, 16,
        [&](std::size_t, std::size_t begin, std::size_t end) {
          for (std::size_t offset = begin; offset < end; ++offset) {
            const std::size_t node_index = range.first + offset;
            const TreeNode &node = tree.nodes[node_index];
            std::array<double, kTaylorMomentMax> phin_comp{};
            std::array<std::array<double, kTaylorMomentMax>, 3> normal_comp{};
            for (std::size_t child = 0; child < node.child_count; ++child) {
              const std::size_t child_index =
                  child_node_index(tree, node, child);
              if (child_index >= tree.nodes.size())
                continue;
              const TreeNode &child_node = tree.nodes[child_index];
              translate_child_moment(packed.phin[node_index],
                                     packed.phin[child_index], node,
                                     child_node, indices, lookup,
                                     use_kahan ? &phin_comp : nullptr);
              for (int c = 0; c < 3; ++c) {
                translate_child_moment(
                    packed.phi_normal[static_cast<std::size_t>(c)]
                                      [node_index],
                    packed.phi_normal[static_cast<std::size_t>(c)]
                                      [child_index],
                    node, child_node, indices, lookup,
                    use_kahan ? &normal_comp[static_cast<std::size_t>(c)]
                              : nullptr);
              }
            }
          }
        });
  }
  precompute_jcp_convergence_norms(packed, tree, taylor_order, executor);
  return packed;
}



double moment_channel_rel_error(const NodeMoments &direct,
                                const NodeMoments &translated,
                                std::size_t count) noexcept {
  double diff = 0.0;
  double denom = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    diff = std::max(diff, std::abs(direct.values[i] - translated.values[i]));
    denom = std::max(denom, std::abs(direct.values[i]));
  }
  return diff / std::max(denom, 1.0e-30);
}

void accumulate_leaf_moment_audit(
    NodeMoments &moment, std::array<double, kTaylorMomentMax> &abs_sum,
    std::array<double, kTaylorMomentMax> &max_abs,
    std::array<double, kTaylorMomentMax> &min_abs, const Panel &panel,
    const TreeNode &node, double weight,
    const std::vector<MomentIndex> &indices) {
  const double dx = panel.x - node.cx;
  const double dy = panel.y - node.cy;
  const double dz = panel.z - node.cz;
  for (std::size_t m = 0; m < indices.size(); ++m) {
    const MomentIndex &index = indices[m];
    const double value = weight * pow_int(dx, index.ax) *
                         pow_int(dy, index.ay) * pow_int(dz, index.az);
    moment.values[m] += value;
    const double av = std::abs(value);
    abs_sum[m] += av;
    max_abs[m] = std::max(max_abs[m], av);
    if (av > 0.0 && (min_abs[m] == 0.0 || av < min_abs[m]))
      min_abs[m] = av;
  }
}

void observe_moment_cancellation(
    const NodeMoments &moment, const std::array<double, kTaylorMomentMax> &abs_sum,
    const std::array<double, kTaylorMomentMax> &max_abs,
    const std::array<double, kTaylorMomentMax> &min_abs,
    std::size_t moment_count, double &ratio_sum, std::size_t &ratio_count,
    double &ratio_max, double &dynamic_range_max) noexcept {
  for (std::size_t m = 0; m < moment_count; ++m) {
    if (abs_sum[m] <= 0.0)
      continue;
    const double ratio =
        abs_sum[m] / std::max(std::abs(moment.values[m]), 1.0e-300);
    ratio_sum += ratio;
    ++ratio_count;
    ratio_max = std::max(ratio_max, ratio);
    if (min_abs[m] > 0.0)
      dynamic_range_max =
          std::max(dynamic_range_max, max_abs[m] / min_abs[m]);
  }
}

MomentAuditSummary audit_jcp_moment_translation(
    const std::vector<Panel> &panels, const std::vector<double> &unknowns,
    bool coupled, const TreeWorkspace &tree, int taylor_order,
    const PackedJcpMoments &packed) {
  MomentAuditSummary summary;
  if (tree.moment_audit == "off" || tree.nodes.empty())
    return summary;
  const auto &indices = moment_indices(taylor_order);
  if (indices.empty())
    return summary;
  const std::size_t moment_count = indices.size();
  const std::size_t n = panels.size();
  double offset_sum = 0.0;
  std::size_t offset_count = 0;
  for (const TreeNode &node : tree.nodes) {
    for (std::size_t child = 0; child < node.child_count; ++child) {
      const std::size_t child_index = child_node_index(tree, node, child);
      if (child_index >= tree.nodes.size())
        continue;
      const TreeNode &child_node = tree.nodes[child_index];
      const double offset =
          point_distance(child_node.cx, child_node.cy, child_node.cz, node.cx,
                         node.cy, node.cz);
      summary.translation_offset_max =
          std::max(summary.translation_offset_max, offset);
      offset_sum += offset;
      ++offset_count;
    }
  }
  if (offset_count > 0)
    summary.translation_offset_mean =
        offset_sum / static_cast<double>(offset_count);
  const bool full =
      tree.moment_audit == "full-small" && tree.nodes.size() <= 4096;
  const std::size_t sample_cap = full ? tree.nodes.size() : 64;
  const std::size_t stride =
      full ? 1 : std::max<std::size_t>(1, tree.nodes.size() / sample_cap);
  double rel_sum = 0.0;
  double cancellation_sum = 0.0;
  std::size_t cancellation_count = 0;
  for (std::size_t node_index = 0; node_index < tree.nodes.size();
       node_index += stride) {
    if (!full && summary.nodes >= sample_cap)
      break;
    const TreeNode &node = tree.nodes[node_index];
    if (node.panel_count == 0)
      continue;
    NodeMoments direct_phin;
    std::array<NodeMoments, 3> direct_phi_normal{};
    std::array<double, kTaylorMomentMax> phin_abs{};
    std::array<double, kTaylorMomentMax> phin_max{};
    std::array<double, kTaylorMomentMax> phin_min{};
    std::array<std::array<double, kTaylorMomentMax>, 3> normal_abs{};
    std::array<std::array<double, kTaylorMomentMax>, 3> normal_max{};
    std::array<std::array<double, kTaylorMomentMax>, 3> normal_min{};
    const std::size_t end =
        std::min(tree.order.size(), node.first_panel + node.panel_count);
    for (std::size_t pos = node.first_panel; pos < end; ++pos) {
      const std::size_t panel_index = tree.order[pos];
      if (panel_index >= panels.size())
        continue;
      const Panel &panel = panels[panel_index];
      const double phi = panel_index < unknowns.size() ? unknowns[panel_index]
                                                       : 0.0;
      if(const auto*cell=curved_dual_cell(panel)) {
        for(const auto&q:cell->samples){Panel sample;
          sample.x=q.xyz[0];sample.y=q.xyz[1];sample.z=q.xyz[2];
          for(int c=0;c<3;++c)accumulate_leaf_moment_audit(direct_phi_normal[c],normal_abs[c],normal_max[c],normal_min[c],sample,node,q.weight*phi*q.normal[c],indices);
          if(coupled&&n+panel_index<unknowns.size())accumulate_leaf_moment_audit(direct_phin,phin_abs,phin_max,phin_min,sample,node,q.weight*unknowns[n+panel_index],indices);
        }
        continue;
      }
      const double area_phi = panel.area * phi;
      accumulate_leaf_moment_audit(direct_phi_normal[0], normal_abs[0],
                                   normal_max[0], normal_min[0], panel, node,
                                   area_phi * panel.nx, indices);
      accumulate_leaf_moment_audit(direct_phi_normal[1], normal_abs[1],
                                   normal_max[1], normal_min[1], panel, node,
                                   area_phi * panel.ny, indices);
      accumulate_leaf_moment_audit(direct_phi_normal[2], normal_abs[2],
                                   normal_max[2], normal_min[2], panel, node,
                                   area_phi * panel.nz, indices);
      if (coupled && n + panel_index < unknowns.size()) {
        accumulate_leaf_moment_audit(
            direct_phin, phin_abs, phin_max, phin_min, panel, node,
            panel.area * unknowns[n + panel_index], indices);
      }
    }
    observe_moment_cancellation(
        direct_phin, phin_abs, phin_max, phin_min, moment_count,
        cancellation_sum, cancellation_count, summary.cancellation_ratio_max,
        summary.dynamic_range_max);
    for (int c = 0; c < 3; ++c) {
      observe_moment_cancellation(
          direct_phi_normal[static_cast<std::size_t>(c)],
          normal_abs[static_cast<std::size_t>(c)],
          normal_max[static_cast<std::size_t>(c)],
          normal_min[static_cast<std::size_t>(c)], moment_count,
          cancellation_sum, cancellation_count, summary.cancellation_ratio_max,
          summary.dynamic_range_max);
    }
    double rel = 0.0;
    if (node_index < packed.phin.size())
      rel = std::max(rel, moment_channel_rel_error(
                              direct_phin, packed.phin[node_index],
                              moment_count));
    for (int c = 0; c < 3; ++c) {
      const std::size_t channel = static_cast<std::size_t>(c);
      if (node_index < packed.phi_normal[channel].size()) {
        rel = std::max(rel, moment_channel_rel_error(
                                direct_phi_normal[channel],
                                packed.phi_normal[channel][node_index],
                                moment_count));
      }
    }
    summary.max_rel_error = std::max(summary.max_rel_error, rel);
    rel_sum += rel;
    ++summary.nodes;
  }
  if (summary.nodes > 0)
    summary.mean_rel_error = rel_sum / static_cast<double>(summary.nodes);
  if (cancellation_count > 0)
    summary.cancellation_ratio_mean =
        cancellation_sum / static_cast<double>(cancellation_count);
  return summary;
}



// Orders 2-5 with analytic derivatives take the radial-derivative coefficients
// of tabi_radial_taylor.h: one pass per (target, node) instead of building a
// truncated power series of every kernel (sixteen per coupled pair). The
// series functions below remain the reference, the finite-difference debug
// mode and the path for other orders.
static bool use_radial_taylor(int order,
                              const std::string &derivative_mode) noexcept {
  return derivative_mode != "finite-diff-debug" && order >= 2 &&
         order <= kTaylorOrderMax;
}

JcpFarEvaluator make_jcp_far_evaluator(
    int order, bool coupled, const std::string &derivative_mode) noexcept {
  JcpFarEvaluator out;
  out.order = order;
  out.coupled = coupled;
  out.derivative_mode = &derivative_mode;
  if (use_radial_taylor(order, derivative_mode) &&
      !titan_simd_dispatch::runtime_simd_forced_scalar()) {
    out.fast = radial_jcp_far_kernel(order, coupled);
    out.fast4 = radial_jcp_far_kernel4(order, coupled);
  }
  return out;
}

JcpAccum taylor_jcp_node_sum(const Panel &target, const TreeNode &node,
                             const NodeMoments &phin_scalar,
                             const std::array<NodeMoments, 3> &phi_normal,
                             double eps_ratio, double kappa, bool coupled,
                             int order,
                             const std::string &derivative_mode) noexcept {
  if (use_radial_taylor(order, derivative_mode))
    return radial_jcp_node_sum(target, node, phin_scalar, phi_normal,
                               eps_ratio, kappa, coupled, order);
  return taylor_jcp_node_sum_series(target, node, phin_scalar, phi_normal,
                                    eps_ratio, kappa, coupled, order,
                                    derivative_mode);
}

JcpAccum taylor_jcp_node_sum_series(
    const Panel &target, const TreeNode &node, const NodeMoments &phin_scalar,
    const std::array<NodeMoments, 3> &phi_normal, double eps_ratio,
    double kappa, bool coupled, int order,
    const std::string &derivative_mode) noexcept {
  JcpAccum acc;
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double poisson_scale = eps_ratio - 1.0;
  if (derivative_mode != "finite-diff-debug") {
    if (coupled) {
      acc.phi += taylor_sum_series(
          phin_scalar, order,
          series_sub(green_series(target.x, target.y, target.z, node.cx,
                                  node.cy, node.cz, 0.0, order),
                     green_series(target.x, target.y, target.z, node.cx,
                                  node.cy, node.cz, kappa, order),
                     order));
      acc.phin += taylor_sum_series(
          phin_scalar, order,
          series_sub(target_normal_green_series(target, node.cx, node.cy,
                                                node.cz, 0.0, order),
                     series_scale(target_normal_green_series(
                                      target, node.cx, node.cy, node.cz, kappa,
                                      order),
                                  inv_eps_ratio, order),
                     order));
    }
    for (int c = 0; c < 3; ++c) {
      const TaylorSeries d0 = source_normal_component_green_series(
          target.x, target.y, target.z, node.cx, node.cy, node.cz, c, 0.0,
          order);
      if (!coupled) {
        acc.phi += taylor_sum_series(
            phi_normal[static_cast<std::size_t>(c)], order,
            series_scale(d0, poisson_scale, order));
      } else {
        const TaylorSeries dk = source_normal_component_green_series(
            target.x, target.y, target.z, node.cx, node.cy, node.cz, c, kappa,
            order);
        acc.phi += taylor_sum_series(
            phi_normal[static_cast<std::size_t>(c)], order,
            series_sub(series_scale(dk, eps_ratio, order), d0, order));
        acc.phin += taylor_sum_series(
            phi_normal[static_cast<std::size_t>(c)], order,
            series_sub(mixed_normal_component_green_series(
                           target, node.cx, node.cy, node.cz, c, kappa,
                           order),
                       mixed_normal_component_green_series(
                           target, node.cx, node.cy, node.cz, c, 0.0, order),
                       order));
      }
    }
    return acc;
  }

  const double dx = target.x - node.cx;
  const double dy = target.y - node.cy;
  const double dz = target.z - node.cz;
  const double distance =
      std::sqrt(std::max(1.0e-16, dx * dx + dy * dy + dz * dz));
  const double h = std::clamp(0.05 * distance, 1.0e-4,
                              std::max(1.0e-3, 0.20 * node.radius));
  if (coupled) {
    acc.phi += taylor_sum_kernel(node, phin_scalar, order, h,
                                 [&](double sx, double sy, double sz) {
                                   return green_value(target.x, target.y,
                                                      target.z, sx, sy, sz,
                                                      0.0) -
                                          green_value(target.x, target.y,
                                                      target.z, sx, sy, sz,
                                                      kappa);
                                 });
    acc.phin += taylor_sum_kernel(node, phin_scalar, order, h,
                                  [&](double sx, double sy, double sz) {
                                    return target_normal_green(target, sx, sy,
                                                               sz, 0.0) -
                                           inv_eps_ratio *
                                               target_normal_green(
                                                   target, sx, sy, sz, kappa);
                                  });
  }
  for (int c = 0; c < 3; ++c) {
    acc.phi += taylor_sum_kernel(
        node, phi_normal[static_cast<std::size_t>(c)], order, h,
        [&](double sx, double sy, double sz) {
          const double d0 = source_normal_component_green(
              target.x, target.y, target.z, sx, sy, sz, c, 0.0);
          if (!coupled)
            return poisson_scale * d0;
          const double dk = source_normal_component_green(
              target.x, target.y, target.z, sx, sy, sz, c, kappa);
          return eps_ratio * dk - d0;
        });
    if (coupled) {
      acc.phin += taylor_sum_kernel(
          node, phi_normal[static_cast<std::size_t>(c)], order, h,
          [&](double sx, double sy, double sz) {
            return mixed_normal_component_green(target, sx, sy, sz, c, kappa) -
                   mixed_normal_component_green(target, sx, sy, sz, c, 0.0);
          });
    }
  }
  return acc;
}

JcpChannelAccum taylor_jcp_node_channel_sum(
    const Panel &target, const TreeNode &node, const NodeMoments &phin_scalar,
    const std::array<NodeMoments, 3> &phi_normal, double eps_ratio,
    double kappa, bool coupled, int order,
    const std::string &derivative_mode) noexcept {
  if (use_radial_taylor(order, derivative_mode))
    return radial_jcp_node_channel_sum(target, node, phin_scalar, phi_normal,
                                       eps_ratio, kappa, coupled, order);
  return taylor_jcp_node_channel_sum_series(target, node, phin_scalar,
                                            phi_normal, eps_ratio, kappa,
                                            coupled, order, derivative_mode);
}

JcpChannelAccum taylor_jcp_node_channel_sum_series(
    const Panel &target, const TreeNode &node, const NodeMoments &phin_scalar,
    const std::array<NodeMoments, 3> &phi_normal, double eps_ratio,
    double kappa, bool coupled, int order,
    const std::string &derivative_mode) noexcept {
  JcpChannelAccum acc;
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double poisson_scale = eps_ratio - 1.0;
  if (derivative_mode != "finite-diff-debug") {
    if (coupled) {
      acc.channel[kJcpK1PhiChannel].phi += taylor_sum_series(
          phin_scalar, order,
          series_sub(green_series(target.x, target.y, target.z, node.cx,
                                  node.cy, node.cz, 0.0, order),
                     green_series(target.x, target.y, target.z, node.cx,
                                  node.cy, node.cz, kappa, order),
                     order));
      acc.channel[kJcpK3PhinChannel].phin += taylor_sum_series(
          phin_scalar, order,
          series_sub(target_normal_green_series(target, node.cx, node.cy,
                                                node.cz, 0.0, order),
                     series_scale(target_normal_green_series(
                                      target, node.cx, node.cy, node.cz, kappa,
                                      order),
                                  inv_eps_ratio, order),
                     order));
    }
    for (int c = 0; c < 3; ++c) {
      const TaylorSeries d0 = source_normal_component_green_series(
          target.x, target.y, target.z, node.cx, node.cy, node.cz, c, 0.0,
          order);
      if (!coupled) {
        acc.channel[kJcpK2PhiNormalChannel].phi += taylor_sum_series(
            phi_normal[static_cast<std::size_t>(c)], order,
            series_scale(d0, poisson_scale, order));
      } else {
        const TaylorSeries dk = source_normal_component_green_series(
            target.x, target.y, target.z, node.cx, node.cy, node.cz, c, kappa,
            order);
        acc.channel[kJcpK2PhiNormalChannel].phi += taylor_sum_series(
            phi_normal[static_cast<std::size_t>(c)], order,
            series_sub(series_scale(dk, eps_ratio, order), d0, order));
        acc.channel[kJcpK4MixedChannel].phin += taylor_sum_series(
            phi_normal[static_cast<std::size_t>(c)], order,
            series_sub(mixed_normal_component_green_series(
                           target, node.cx, node.cy, node.cz, c, kappa,
                           order),
                       mixed_normal_component_green_series(
                           target, node.cx, node.cy, node.cz, c, 0.0, order),
                       order));
      }
    }
    return acc;
  }

  const double dx = target.x - node.cx;
  const double dy = target.y - node.cy;
  const double dz = target.z - node.cz;
  const double distance =
      std::sqrt(std::max(1.0e-16, dx * dx + dy * dy + dz * dz));
  const double h = std::clamp(0.05 * distance, 1.0e-4,
                              std::max(1.0e-3, 0.20 * node.radius));
  if (coupled) {
    acc.channel[kJcpK1PhiChannel].phi += taylor_sum_kernel(
        node, phin_scalar, order, h, [&](double sx, double sy, double sz) {
          return green_value(target.x, target.y, target.z, sx, sy, sz, 0.0) -
                 green_value(target.x, target.y, target.z, sx, sy, sz, kappa);
        });
    acc.channel[kJcpK3PhinChannel].phin += taylor_sum_kernel(
        node, phin_scalar, order, h, [&](double sx, double sy, double sz) {
          return target_normal_green(target, sx, sy, sz, 0.0) -
                 inv_eps_ratio *
                     target_normal_green(target, sx, sy, sz, kappa);
        });
  }
  for (int c = 0; c < 3; ++c) {
    acc.channel[kJcpK2PhiNormalChannel].phi += taylor_sum_kernel(
        node, phi_normal[static_cast<std::size_t>(c)], order, h,
        [&](double sx, double sy, double sz) {
          const double d0 = source_normal_component_green(
              target.x, target.y, target.z, sx, sy, sz, c, 0.0);
          if (!coupled)
            return poisson_scale * d0;
          const double dk = source_normal_component_green(
              target.x, target.y, target.z, sx, sy, sz, c, kappa);
          return eps_ratio * dk - d0;
        });
    if (coupled) {
      acc.channel[kJcpK4MixedChannel].phin += taylor_sum_kernel(
          node, phi_normal[static_cast<std::size_t>(c)], order, h,
          [&](double sx, double sy, double sz) {
            return mixed_normal_component_green(target, sx, sy, sz, c, kappa) -
                   mixed_normal_component_green(target, sx, sy, sz, c, 0.0);
          });
    }
  }
  return acc;
}

JcpChannelAccum taylor_jcp_node_channel_degree_sum(
    const Panel &target, const TreeNode &node, const NodeMoments &phin_scalar,
    const std::array<NodeMoments, 3> &phi_normal, double eps_ratio,
    double kappa, bool coupled, int order, const std::string &derivative_mode,
    int moment_degree) noexcept {
  NodeMoments masked_phin;
  std::array<NodeMoments, 3> masked_phi_normal{};
  const auto &indices = moment_indices(order);
  for (std::size_t m = 0; m < indices.size(); ++m) {
    const MomentIndex &index = indices[m];
    if (index.ax + index.ay + index.az != moment_degree)
      continue;
    masked_phin.values[m] = phin_scalar.values[m];
    for (std::size_t component = 0; component < masked_phi_normal.size();
         ++component)
      masked_phi_normal[component].values[m] =
          phi_normal[component].values[m];
  }
  return taylor_jcp_node_channel_sum(
      target, node, masked_phin, masked_phi_normal, eps_ratio, kappa, coupled,
      order, derivative_mode);
}




} // namespace TitanPBE::Tabi::detail
