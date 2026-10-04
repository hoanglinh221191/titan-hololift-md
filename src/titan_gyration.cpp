#include "titan_gyration.h"

#include "titan_simd_dispatch.h"

#include <algorithm>
#include <cmath>

namespace titan_gyration {

namespace {

inline void accumulate_point(PointStats &stats, double x, double y,
                             double z) noexcept {
  stats.sum_x += x;
  stats.sum_y += y;
  stats.sum_z += z;
  stats.sum_sq += x * x + y * y + z * z;
  stats.sum_x2 += x * x;
  stats.sum_y2 += y * y;
  stats.sum_z2 += z * z;
  stats.sum_xy += x * y;
  stats.sum_xz += x * z;
  stats.sum_yz += y * z;
  stats.min_x = std::min(stats.min_x, x);
  stats.min_y = std::min(stats.min_y, y);
  stats.min_z = std::min(stats.min_z, z);
  stats.max_x = std::max(stats.max_x, x);
  stats.max_y = std::max(stats.max_y, y);
  stats.max_z = std::max(stats.max_z, z);
}

PointStats build_point_stats_scalar(const std::vector<Vec3> &points) {
  PointStats stats;
  for (const auto &point : points)
    stats.add(point.x, point.y, point.z);
  return stats;
}

PointStats build_point_stats_soa_scalar_impl(const double *xs, const double *ys,
                                             const double *zs,
                                             std::size_t count) noexcept {
  PointStats stats;
  stats.count = count;
  if (count == 0)
    return stats;
  for (std::size_t i = 0; i < count; ++i)
    accumulate_point(stats, xs[i], ys[i], zs[i]);
  return stats;
}

void sync_soa(const std::vector<Vec3> &points, GyrationScratch &scratch) {
  scratch.reserve(points.size());
  scratch.xs.resize(points.size());
  scratch.ys.resize(points.size());
  scratch.zs.resize(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    scratch.xs[i] = points[i].x;
    scratch.ys[i] = points[i].y;
    scratch.zs[i] = points[i].z;
  }
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
PointStats build_point_stats_soa_avx2_impl(const double *xs, const double *ys,
                                           const double *zs,
                                           std::size_t count) noexcept {
  PointStats stats;
  stats.count = count;
  if (count == 0)
    return stats;

  const __m256d inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
  const __m256d neg_inf =
      _mm256_set1_pd(-std::numeric_limits<double>::infinity());
  __m256d sum_x = _mm256_setzero_pd();
  __m256d sum_y = _mm256_setzero_pd();
  __m256d sum_z = _mm256_setzero_pd();
  __m256d sum_sq = _mm256_setzero_pd();
  __m256d sum_x2 = _mm256_setzero_pd();
  __m256d sum_y2 = _mm256_setzero_pd();
  __m256d sum_z2 = _mm256_setzero_pd();
  __m256d sum_xy = _mm256_setzero_pd();
  __m256d sum_xz = _mm256_setzero_pd();
  __m256d sum_yz = _mm256_setzero_pd();
  __m256d min_x = inf;
  __m256d min_y = inf;
  __m256d min_z = inf;
  __m256d max_x = neg_inf;
  __m256d max_y = neg_inf;
  __m256d max_z = neg_inf;

  std::size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const __m256d x = _mm256_loadu_pd(xs + i);
    const __m256d y = _mm256_loadu_pd(ys + i);
    const __m256d z = _mm256_loadu_pd(zs + i);
    const __m256d x2 = _mm256_mul_pd(x, x);
    const __m256d y2 = _mm256_mul_pd(y, y);
    const __m256d z2 = _mm256_mul_pd(z, z);
    sum_x = _mm256_add_pd(sum_x, x);
    sum_y = _mm256_add_pd(sum_y, y);
    sum_z = _mm256_add_pd(sum_z, z);
    sum_x2 = _mm256_add_pd(sum_x2, x2);
    sum_y2 = _mm256_add_pd(sum_y2, y2);
    sum_z2 = _mm256_add_pd(sum_z2, z2);
    sum_sq = _mm256_add_pd(sum_sq, _mm256_add_pd(_mm256_add_pd(x2, y2), z2));
    sum_xy = _mm256_add_pd(sum_xy, _mm256_mul_pd(x, y));
    sum_xz = _mm256_add_pd(sum_xz, _mm256_mul_pd(x, z));
    sum_yz = _mm256_add_pd(sum_yz, _mm256_mul_pd(y, z));
    min_x = _mm256_min_pd(min_x, x);
    min_y = _mm256_min_pd(min_y, y);
    min_z = _mm256_min_pd(min_z, z);
    max_x = _mm256_max_pd(max_x, x);
    max_y = _mm256_max_pd(max_y, y);
    max_z = _mm256_max_pd(max_z, z);
  }

  alignas(32) double lane_x[4];
  alignas(32) double lane_y[4];
  alignas(32) double lane_z[4];
  alignas(32) double lane_sq[4];
  alignas(32) double lane_x2[4];
  alignas(32) double lane_y2[4];
  alignas(32) double lane_z2[4];
  alignas(32) double lane_xy[4];
  alignas(32) double lane_xz[4];
  alignas(32) double lane_yz[4];
  alignas(32) double lane_min_x[4];
  alignas(32) double lane_min_y[4];
  alignas(32) double lane_min_z[4];
  alignas(32) double lane_max_x[4];
  alignas(32) double lane_max_y[4];
  alignas(32) double lane_max_z[4];

  _mm256_store_pd(lane_x, sum_x);
  _mm256_store_pd(lane_y, sum_y);
  _mm256_store_pd(lane_z, sum_z);
  _mm256_store_pd(lane_sq, sum_sq);
  _mm256_store_pd(lane_x2, sum_x2);
  _mm256_store_pd(lane_y2, sum_y2);
  _mm256_store_pd(lane_z2, sum_z2);
  _mm256_store_pd(lane_xy, sum_xy);
  _mm256_store_pd(lane_xz, sum_xz);
  _mm256_store_pd(lane_yz, sum_yz);
  _mm256_store_pd(lane_min_x, min_x);
  _mm256_store_pd(lane_min_y, min_y);
  _mm256_store_pd(lane_min_z, min_z);
  _mm256_store_pd(lane_max_x, max_x);
  _mm256_store_pd(lane_max_y, max_y);
  _mm256_store_pd(lane_max_z, max_z);

  for (int lane = 0; lane < 4; ++lane) {
    stats.sum_x += lane_x[lane];
    stats.sum_y += lane_y[lane];
    stats.sum_z += lane_z[lane];
    stats.sum_sq += lane_sq[lane];
    stats.sum_x2 += lane_x2[lane];
    stats.sum_y2 += lane_y2[lane];
    stats.sum_z2 += lane_z2[lane];
    stats.sum_xy += lane_xy[lane];
    stats.sum_xz += lane_xz[lane];
    stats.sum_yz += lane_yz[lane];
    stats.min_x = std::min(stats.min_x, lane_min_x[lane]);
    stats.min_y = std::min(stats.min_y, lane_min_y[lane]);
    stats.min_z = std::min(stats.min_z, lane_min_z[lane]);
    stats.max_x = std::max(stats.max_x, lane_max_x[lane]);
    stats.max_y = std::max(stats.max_y, lane_max_y[lane]);
    stats.max_z = std::max(stats.max_z, lane_max_z[lane]);
  }

  for (; i < count; ++i)
    accumulate_point(stats, xs[i], ys[i], zs[i]);
  return stats;
}
#endif

} // namespace

void PointStats::add(double x, double y, double z) noexcept {
  ++count;
  accumulate_point(*this, x, y, z);
}

void GyrationScratch::reserve(std::size_t point_count) {
  xs.reserve(point_count);
  ys.reserve(point_count);
  zs.reserve(point_count);
}

bool avx2_available() noexcept { return titan_simd_dispatch::runtime_has_avx2(); }

PointStats build_point_stats(const std::vector<Vec3> &points,
                             GyrationScratch *scratch) {
  if (points.empty())
    return {};

  GyrationScratch local_scratch;
  GyrationScratch *work = scratch;
  if (work == nullptr && avx2_available())
    work = &local_scratch;

  if (work != nullptr && avx2_available()) {
    sync_soa(points, *work);
    return build_point_stats_soa(work->xs.data(), work->ys.data(),
                                 work->zs.data(), points.size());
  }
  return build_point_stats_scalar(points);
}

PointStats build_point_stats_soa(const double *xs, const double *ys,
                                 const double *zs, std::size_t count) noexcept {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  if (avx2_available())
    return build_point_stats_soa_avx2_impl(xs, ys, zs, count);
#endif
  return build_point_stats_soa_scalar_impl(xs, ys, zs, count);
}

double compute_bbox_volume(const PointStats &stats) noexcept {
  if (stats.count == 0)
    return std::numeric_limits<double>::infinity();
  return std::max(0.0, stats.max_x - stats.min_x) *
         std::max(0.0, stats.max_y - stats.min_y) *
         std::max(0.0, stats.max_z - stats.min_z);
}

double compute_rg2_from_stats(const PointStats &stats) noexcept {
  if (stats.count == 0)
    return std::numeric_limits<double>::infinity();
  const double inv_count = 1.0 / static_cast<double>(stats.count);
  const double mean_x = stats.sum_x * inv_count;
  const double mean_y = stats.sum_y * inv_count;
  const double mean_z = stats.sum_z * inv_count;
  return std::max(
      0.0, stats.sum_sq * inv_count -
               (mean_x * mean_x + mean_y * mean_y + mean_z * mean_z));
}

double compute_rg_from_stats(const PointStats &stats) noexcept {
  const double rg2 = compute_rg2_from_stats(stats);
  if (!std::isfinite(rg2))
    return std::numeric_limits<double>::infinity();
  return std::sqrt(std::max(0.0, rg2));
}

double compute_rg2(const std::vector<Vec3> &points, GyrationScratch *scratch) {
  return compute_rg2_from_stats(build_point_stats(points, scratch));
}

std::array<double, 3> compute_principal_rg2(const PointStats &stats) {
  if (stats.count == 0)
    return {std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
  const double inv_n = 1.0 / static_cast<double>(stats.count);
  const double mx = stats.sum_x * inv_n;
  const double my = stats.sum_y * inv_n;
  const double mz = stats.sum_z * inv_n;
  double a[3][3] = {
      {std::max(0.0, stats.sum_x2 * inv_n - mx * mx),
       stats.sum_xy * inv_n - mx * my, stats.sum_xz * inv_n - mx * mz},
      {stats.sum_xy * inv_n - mx * my,
       std::max(0.0, stats.sum_y2 * inv_n - my * my),
       stats.sum_yz * inv_n - my * mz},
      {stats.sum_xz * inv_n - mx * mz, stats.sum_yz * inv_n - my * mz,
       std::max(0.0, stats.sum_z2 * inv_n - mz * mz)}};

  for (int iter = 0; iter < 12; ++iter) {
    int p = 0;
    int q = 1;
    double max_off = std::fabs(a[0][1]);
    if (std::fabs(a[0][2]) > max_off) {
      p = 0;
      q = 2;
      max_off = std::fabs(a[0][2]);
    }
    if (std::fabs(a[1][2]) > max_off) {
      p = 1;
      q = 2;
      max_off = std::fabs(a[1][2]);
    }
    if (max_off <= 1.0e-12)
      break;
    const double app = a[p][p];
    const double aqq = a[q][q];
    const double apq = a[p][q];
    const double tau = (aqq - app) / (2.0 * apq);
    const double t = ((tau >= 0.0) ? 1.0 : -1.0) /
                     (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
    const double c = 1.0 / std::sqrt(1.0 + t * t);
    const double s = t * c;
    a[p][p] = app - t * apq;
    a[q][q] = aqq + t * apq;
    a[p][q] = 0.0;
    a[q][p] = 0.0;
    for (int r = 0; r < 3; ++r) {
      if (r == p || r == q)
        continue;
      const double arp = a[r][p];
      const double arq = a[r][q];
      a[r][p] = c * arp - s * arq;
      a[p][r] = a[r][p];
      a[r][q] = c * arq + s * arp;
      a[q][r] = a[r][q];
    }
  }

  std::array<double, 3> eig{std::max(0.0, a[0][0]), std::max(0.0, a[1][1]),
                            std::max(0.0, a[2][2])};
  std::sort(eig.begin(), eig.end(), std::greater<double>());
  return eig;
}

GyrationShapeMetrics
compute_gyration_shape_metrics(const std::array<double, 3> &principal_rg2) {
  GyrationShapeMetrics metrics;
  if (!std::isfinite(principal_rg2[0]) || !std::isfinite(principal_rg2[1]) ||
      !std::isfinite(principal_rg2[2])) {
    return metrics;
  }
  const double lambda_max = std::max(0.0, principal_rg2[0]);
  const double lambda_mid = std::max(0.0, principal_rg2[1]);
  const double lambda_min = std::max(0.0, principal_rg2[2]);
  const double rg2_sum = lambda_max + lambda_mid + lambda_min;
  if (rg2_sum <= 1.0e-12) {
    metrics.kappa2 = 0.0;
    metrics.asphericity = 0.0;
    metrics.acylindricity = 0.0;
    return metrics;
  }
  const double pair_sum =
      lambda_max * lambda_mid + lambda_mid * lambda_min +
      lambda_max * lambda_min;
  metrics.kappa2 =
      1.0 - 3.0 * pair_sum / std::max(1.0e-24, rg2_sum * rg2_sum);
  metrics.kappa2 = std::max(0.0, std::min(1.0, metrics.kappa2));
  metrics.asphericity =
      std::max(0.0, lambda_max - 0.5 * (lambda_mid + lambda_min));
  metrics.acylindricity = std::max(0.0, lambda_mid - lambda_min);
  return metrics;
}

} // namespace titan_gyration
