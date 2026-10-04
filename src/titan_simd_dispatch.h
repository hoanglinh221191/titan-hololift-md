#ifndef TITAN_SIMD_DISPATCH_H
#define TITAN_SIMD_DISPATCH_H

#include "titan_build_info.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) ||           \
    defined(_M_X64)
#define TITAN_SIMD_X86 1
#include <immintrin.h>
#else
#define TITAN_SIMD_X86 0
#endif

#if TITAN_SIMD_X86 && (defined(__GNUC__) || defined(__clang__))
#define TITAN_TARGET_AVX2 __attribute__((target("avx2")))
#define TITAN_TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#define TITAN_LAMBDA_TARGET_AVX2 [[gnu::target("avx2")]]
#define TITAN_LAMBDA_TARGET_AVX2_FMA [[gnu::target("avx2,fma")]]
#define TITAN_RUNTIME_TARGET_ATTRS 1
#else
#define TITAN_TARGET_AVX2
#define TITAN_TARGET_AVX2_FMA
#define TITAN_LAMBDA_TARGET_AVX2
#define TITAN_LAMBDA_TARGET_AVX2_FMA
#define TITAN_RUNTIME_TARGET_ATTRS 0
#endif

namespace titan_simd_dispatch {

inline bool runtime_simd_forced_scalar() noexcept {
  static const bool forced = [] {
    const char *value = std::getenv("TITAN_FORCE_SCALAR");
    if (value == nullptr)
      return false;
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T';
  }();
  return forced;
}

inline bool runtime_has_avx2() noexcept {
  return !runtime_simd_forced_scalar() &&
         titan_build::runtime_cpu_features_cached().avx2;
}

inline bool runtime_has_fma() noexcept {
  return !runtime_simd_forced_scalar() &&
         titan_build::runtime_cpu_features_cached().fma;
}

inline bool runtime_has_avx2_fma() noexcept {
  if (runtime_simd_forced_scalar())
    return false;
  const auto &features = titan_build::runtime_cpu_features_cached();
  return features.avx2 && features.fma;
}

inline bool is_aligned_32(const void *ptr) noexcept {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 31u) == 0u;
}

inline double sum_doubles_scalar(const double *data, std::size_t n) noexcept {
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    total += data[i];
  return total;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline double sum_doubles_avx2(const double *data, std::size_t n) noexcept {
  std::size_t i = 0;
  __m256d acc = _mm256_setzero_pd();
  for (; i + 4 <= n; i += 4)
    acc = _mm256_add_pd(acc, _mm256_loadu_pd(data + i));

  alignas(32) double lane[4];
  _mm256_store_pd(lane, acc);
  double total = lane[0] + lane[1] + lane[2] + lane[3];
  for (; i < n; ++i)
    total += data[i];
  return total;
}
#endif

inline double sum_doubles(const double *data, std::size_t n) noexcept {
  using Kernel = double (*)(const double *, std::size_t) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2() ? &sum_doubles_avx2 : &sum_doubles_scalar;
#else
    return &sum_doubles_scalar;
#endif
  }();
  return kernel(data, n);
}

inline double squared_diff_sum_scalar(const double *lhs, const double *rhs,
                                      std::size_t n) noexcept {
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = lhs[i] - rhs[i];
    total += d * d;
  }
  return total;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline double squared_diff_sum_avx2(const double *lhs, const double *rhs,
                                    std::size_t n) noexcept {
  std::size_t i = 0;
  __m256d acc = _mm256_setzero_pd();
  for (; i + 4 <= n; i += 4) {
    const __m256d a = _mm256_loadu_pd(lhs + i);
    const __m256d b = _mm256_loadu_pd(rhs + i);
    const __m256d d = _mm256_sub_pd(a, b);
    acc = _mm256_add_pd(acc, _mm256_mul_pd(d, d));
  }
  alignas(32) double lane[4];
  _mm256_store_pd(lane, acc);
  double total = lane[0] + lane[1] + lane[2] + lane[3];
  for (; i < n; ++i) {
    const double d = lhs[i] - rhs[i];
    total += d * d;
  }
  return total;
}
#endif

inline double squared_diff_sum(const double *lhs, const double *rhs,
                               std::size_t n) noexcept {
  using Kernel =
      double (*)(const double *, const double *, std::size_t) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2() ? &squared_diff_sum_avx2
                              : &squared_diff_sum_scalar;
#else
    return &squared_diff_sum_scalar;
#endif
  }();
  return kernel(lhs, rhs, n);
}

inline double squared_distance_rowmajor_scalar(const double *lhs,
                                               const double *rhs,
                                               int dim) noexcept {
  double total = 0.0;
  for (int c = 0; c < dim; ++c) {
    const double diff = lhs[c] - rhs[c];
    total += diff * diff;
  }
  return total;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline double squared_distance_rowmajor_avx2(const double *lhs,
                                             const double *rhs,
                                             int dim) noexcept {
  __m256d acc = _mm256_setzero_pd();
  int c = 0;
  for (; c + 4 <= dim; c += 4) {
    const __m256d a = _mm256_loadu_pd(lhs + c);
    const __m256d b = _mm256_loadu_pd(rhs + c);
    const __m256d diff = _mm256_sub_pd(a, b);
    acc = _mm256_add_pd(acc, _mm256_mul_pd(diff, diff));
  }
  alignas(32) double lane[4];
  _mm256_store_pd(lane, acc);
  double total = lane[0] + lane[1] + lane[2] + lane[3];
  for (; c < dim; ++c) {
    const double diff = lhs[c] - rhs[c];
    total += diff * diff;
  }
  return total;
}
#endif

inline double squared_distance_rowmajor(const double *lhs, const double *rhs,
                                        int dim) noexcept {
  using Kernel = double (*)(const double *, const double *, int) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2() ? &squared_distance_rowmajor_avx2
                              : &squared_distance_rowmajor_scalar;
#else
    return &squared_distance_rowmajor_scalar;
#endif
  }();
  return kernel(lhs, rhs, dim);
}

inline void add_row_inplace_scalar(double *dst, const double *src,
                                   int dim) noexcept {
  for (int c = 0; c < dim; ++c)
    dst[c] += src[c];
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline void add_row_inplace_avx2(double *dst, const double *src,
                                 int dim) noexcept {
  int c = 0;
  for (; c + 4 <= dim; c += 4) {
    const __m256d a = _mm256_loadu_pd(dst + c);
    const __m256d b = _mm256_loadu_pd(src + c);
    _mm256_storeu_pd(dst + c, _mm256_add_pd(a, b));
  }
  for (; c < dim; ++c)
    dst[c] += src[c];
}
#endif

inline void add_row_inplace(double *dst, const double *src, int dim) noexcept {
  using Kernel = void (*)(double *, const double *, int) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2() ? &add_row_inplace_avx2 : &add_row_inplace_scalar;
#else
    return &add_row_inplace_scalar;
#endif
  }();
  kernel(dst, src, dim);
}

inline void translate_xyz_scalar(double *dst_x, double *dst_y, double *dst_z,
                                 const double *src_x, const double *src_y,
                                 const double *src_z, std::size_t n, double dx,
                                 double dy, double dz) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    dst_x[i] = src_x[i] + dx;
    dst_y[i] = src_y[i] + dy;
    dst_z[i] = src_z[i] + dz;
  }
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline void translate_xyz_avx2(double *dst_x, double *dst_y, double *dst_z,
                               const double *src_x, const double *src_y,
                               const double *src_z, std::size_t n, double dx,
                               double dy, double dz) noexcept {
  const __m256d shift_x = _mm256_set1_pd(dx);
  const __m256d shift_y = _mm256_set1_pd(dy);
  const __m256d shift_z = _mm256_set1_pd(dz);
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    _mm256_storeu_pd(dst_x + i,
                     _mm256_add_pd(_mm256_loadu_pd(src_x + i), shift_x));
    _mm256_storeu_pd(dst_y + i,
                     _mm256_add_pd(_mm256_loadu_pd(src_y + i), shift_y));
    _mm256_storeu_pd(dst_z + i,
                     _mm256_add_pd(_mm256_loadu_pd(src_z + i), shift_z));
  }
  for (; i < n; ++i) {
    dst_x[i] = src_x[i] + dx;
    dst_y[i] = src_y[i] + dy;
    dst_z[i] = src_z[i] + dz;
  }
}
#endif

inline void translate_xyz(double *dst_x, double *dst_y, double *dst_z,
                          const double *src_x, const double *src_y,
                          const double *src_z, std::size_t n, double dx,
                          double dy, double dz) noexcept {
  using Kernel = void (*)(double *, double *, double *, const double *,
                          const double *, const double *, std::size_t, double,
                          double, double) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2() ? &translate_xyz_avx2 : &translate_xyz_scalar;
#else
    return &translate_xyz_scalar;
#endif
  }();
  kernel(dst_x, dst_y, dst_z, src_x, src_y, src_z, n, dx, dy, dz);
}

inline double dot_product_scalar(const double *a, const double *b, std::size_t n,
                                 bool) noexcept {
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    total += a[i] * b[i];
  return total;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline double dot_product_avx2(const double *a, const double *b, std::size_t n,
                               bool assume_aligned) noexcept {
  double total = 0.0;
  std::size_t i = 0;
  __m256d sum_vec = _mm256_setzero_pd();
  const bool aligned = assume_aligned && is_aligned_32(a) && is_aligned_32(b);
  for (; i + 4 <= n; i += 4) {
    const __m256d va = aligned ? _mm256_load_pd(a + i) : _mm256_loadu_pd(a + i);
    const __m256d vb = aligned ? _mm256_load_pd(b + i) : _mm256_loadu_pd(b + i);
    sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(va, vb));
  }
  alignas(32) double buffer[4];
  _mm256_store_pd(buffer, sum_vec);
  total = buffer[0] + buffer[1] + buffer[2] + buffer[3];
  for (; i < n; ++i)
    total += a[i] * b[i];
  return total;
}

TITAN_TARGET_AVX2_FMA
inline double dot_product_avx2_fma(const double *a, const double *b,
                                   std::size_t n,
                                   bool assume_aligned) noexcept {
  double total = 0.0;
  std::size_t i = 0;
  __m256d sum_vec = _mm256_setzero_pd();
  const bool aligned = assume_aligned && is_aligned_32(a) && is_aligned_32(b);
  for (; i + 4 <= n; i += 4) {
    const __m256d va = aligned ? _mm256_load_pd(a + i) : _mm256_loadu_pd(a + i);
    const __m256d vb = aligned ? _mm256_load_pd(b + i) : _mm256_loadu_pd(b + i);
    sum_vec = _mm256_fmadd_pd(va, vb, sum_vec);
  }
  alignas(32) double buffer[4];
  _mm256_store_pd(buffer, sum_vec);
  total = buffer[0] + buffer[1] + buffer[2] + buffer[3];
  for (; i < n; ++i)
    total += a[i] * b[i];
  return total;
}
#endif

inline double dot_product(const double *a, const double *b, std::size_t n,
                          bool assume_aligned = false) noexcept {
  using Kernel = double (*)(const double *, const double *, std::size_t,
                            bool) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    if (runtime_has_avx2_fma())
      return &dot_product_avx2_fma;
    if (runtime_has_avx2())
      return &dot_product_avx2;
    return &dot_product_scalar;
#else
    return &dot_product_scalar;
#endif
  }();
  return kernel(a, b, n, assume_aligned);
}

inline double centered_sq_sum_xyz_scalar(const double *x, const double *y,
                                         const double *z, std::size_t n,
                                         double cx, double cy,
                                         double cz) noexcept {
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = x[i] - cx;
    const double dy = y[i] - cy;
    const double dz = z[i] - cz;
    total += dx * dx + dy * dy + dz * dz;
  }
  return total;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline double centered_sq_sum_xyz_avx2(const double *x, const double *y,
                                       const double *z, std::size_t n,
                                       double cx, double cy,
                                       double cz) noexcept {
  std::size_t i = 0;
  __m256d acc = _mm256_setzero_pd();
  const __m256d vcx = _mm256_set1_pd(cx);
  const __m256d vcy = _mm256_set1_pd(cy);
  const __m256d vcz = _mm256_set1_pd(cz);
  for (; i + 4 <= n; i += 4) {
    const __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(x + i), vcx);
    const __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(y + i), vcy);
    const __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(z + i), vcz);
    const __m256d dxy2 =
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy));
    acc = _mm256_add_pd(acc, _mm256_add_pd(dxy2, _mm256_mul_pd(dz, dz)));
  }
  alignas(32) double lane[4];
  _mm256_store_pd(lane, acc);
  double total = lane[0] + lane[1] + lane[2] + lane[3];
  for (; i < n; ++i) {
    const double dx = x[i] - cx;
    const double dy = y[i] - cy;
    const double dz = z[i] - cz;
    total += dx * dx + dy * dy + dz * dz;
  }
  return total;
}

TITAN_TARGET_AVX2_FMA
inline double centered_sq_sum_xyz_avx2_fma(const double *x, const double *y,
                                           const double *z, std::size_t n,
                                           double cx, double cy,
                                           double cz) noexcept {
  std::size_t i = 0;
  __m256d acc = _mm256_setzero_pd();
  const __m256d vcx = _mm256_set1_pd(cx);
  const __m256d vcy = _mm256_set1_pd(cy);
  const __m256d vcz = _mm256_set1_pd(cz);
  for (; i + 4 <= n; i += 4) {
    const __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(x + i), vcx);
    const __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(y + i), vcy);
    const __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(z + i), vcz);
    const __m256d r2 =
        _mm256_fmadd_pd(dz, dz, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx)));
    acc = _mm256_add_pd(acc, r2);
  }
  alignas(32) double lane[4];
  _mm256_store_pd(lane, acc);
  double total = lane[0] + lane[1] + lane[2] + lane[3];
  for (; i < n; ++i) {
    const double dx = x[i] - cx;
    const double dy = y[i] - cy;
    const double dz = z[i] - cz;
    total += dx * dx + dy * dy + dz * dz;
  }
  return total;
}
#endif

inline double centered_sq_sum_xyz(const double *x, const double *y,
                                  const double *z, std::size_t n, double cx,
                                  double cy, double cz) noexcept {
  using Kernel = double (*)(const double *, const double *, const double *,
                            std::size_t, double, double, double) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    if (runtime_has_avx2_fma())
      return &centered_sq_sum_xyz_avx2_fma;
    if (runtime_has_avx2())
      return &centered_sq_sum_xyz_avx2;
    return &centered_sq_sum_xyz_scalar;
#else
    return &centered_sq_sum_xyz_scalar;
#endif
  }();
  return kernel(x, y, z, n, cx, cy, cz);
}

inline double squared_displacement_sum_xyz_scalar(const double *dx,
                                                  const double *dy,
                                                  const double *dz,
                                                  std::size_t n) noexcept {
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    total += dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i];
  return total;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline double squared_displacement_sum_xyz_avx2(const double *dx,
                                                const double *dy,
                                                const double *dz,
                                                std::size_t n) noexcept {
  std::size_t i = 0;
  __m256d acc = _mm256_setzero_pd();
  for (; i + 4 <= n; i += 4) {
    const __m256d vdx = _mm256_loadu_pd(dx + i);
    const __m256d vdy = _mm256_loadu_pd(dy + i);
    const __m256d vdz = _mm256_loadu_pd(dz + i);
    const __m256d dxy2 =
        _mm256_add_pd(_mm256_mul_pd(vdx, vdx), _mm256_mul_pd(vdy, vdy));
    acc = _mm256_add_pd(acc, _mm256_add_pd(dxy2, _mm256_mul_pd(vdz, vdz)));
  }
  alignas(32) double lane[4];
  _mm256_store_pd(lane, acc);
  double total = lane[0] + lane[1] + lane[2] + lane[3];
  for (; i < n; ++i)
    total += dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i];
  return total;
}

TITAN_TARGET_AVX2_FMA
inline double squared_displacement_sum_xyz_avx2_fma(const double *dx,
                                                    const double *dy,
                                                    const double *dz,
                                                    std::size_t n) noexcept {
  std::size_t i = 0;
  __m256d acc = _mm256_setzero_pd();
  for (; i + 4 <= n; i += 4) {
    const __m256d vdx = _mm256_loadu_pd(dx + i);
    const __m256d vdy = _mm256_loadu_pd(dy + i);
    const __m256d vdz = _mm256_loadu_pd(dz + i);
    const __m256d r2 = _mm256_fmadd_pd(
        vdz, vdz, _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdx, vdx)));
    acc = _mm256_add_pd(acc, r2);
  }
  alignas(32) double lane[4];
  _mm256_store_pd(lane, acc);
  double total = lane[0] + lane[1] + lane[2] + lane[3];
  for (; i < n; ++i)
    total += dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i];
  return total;
}
#endif

inline double squared_displacement_sum_xyz(const double *dx, const double *dy,
                                           const double *dz,
                                           std::size_t n) noexcept {
  using Kernel = double (*)(const double *, const double *, const double *,
                            std::size_t) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    if (runtime_has_avx2_fma())
      return &squared_displacement_sum_xyz_avx2_fma;
    if (runtime_has_avx2())
      return &squared_displacement_sum_xyz_avx2;
    return &squared_displacement_sum_xyz_scalar;
#else
    return &squared_displacement_sum_xyz_scalar;
#endif
  }();
  return kernel(dx, dy, dz, n);
}

inline unsigned point_vs_four_cutoff_mask_scalar(
    double px, double py, double pz, const double *xs, const double *ys,
    const double *zs, double cutoff_sq, double min_sq,
    bool inclusive_cutoff) noexcept {
  unsigned mask = 0u;
  for (unsigned lane = 0; lane < 4; ++lane) {
    const double dx = px - xs[lane];
    const double dy = py - ys[lane];
    const double dz = pz - zs[lane];
    const double dist_sq = dx * dx + dy * dy + dz * dz;
    const bool cutoff_ok =
        inclusive_cutoff ? (dist_sq <= cutoff_sq) : (dist_sq < cutoff_sq);
    if (dist_sq > min_sq && cutoff_ok)
      mask |= (1u << lane);
  }
  return mask;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline unsigned point_vs_four_cutoff_mask_avx2(
    double px, double py, double pz, const double *xs, const double *ys,
    const double *zs, double cutoff_sq, double min_sq,
    bool inclusive_cutoff) noexcept {
  const __m256d vpx = _mm256_set1_pd(px);
  const __m256d vpy = _mm256_set1_pd(py);
  const __m256d vpz = _mm256_set1_pd(pz);
  const __m256d vx = _mm256_loadu_pd(xs);
  const __m256d vy = _mm256_loadu_pd(ys);
  const __m256d vz = _mm256_loadu_pd(zs);
  const __m256d dx = _mm256_sub_pd(vpx, vx);
  const __m256d dy = _mm256_sub_pd(vpy, vy);
  const __m256d dz = _mm256_sub_pd(vpz, vz);
  const __m256d dist_sq =
      _mm256_add_pd(_mm256_mul_pd(dx, dx),
                    _mm256_add_pd(_mm256_mul_pd(dy, dy), _mm256_mul_pd(dz, dz)));
  const __m256d cutoff_vec = _mm256_set1_pd(cutoff_sq);
  const __m256d min_vec = _mm256_set1_pd(min_sq);
  const __m256d cutoff_mask =
      inclusive_cutoff ? _mm256_cmp_pd(dist_sq, cutoff_vec, _CMP_LE_OQ)
                       : _mm256_cmp_pd(dist_sq, cutoff_vec, _CMP_LT_OQ);
  const __m256d min_mask = _mm256_cmp_pd(dist_sq, min_vec, _CMP_GT_OQ);
  return static_cast<unsigned>(
      _mm256_movemask_pd(_mm256_and_pd(cutoff_mask, min_mask)));
}
#endif

inline unsigned point_vs_four_cutoff_mask(double px, double py, double pz,
                                          const double *xs, const double *ys,
                                          const double *zs, double cutoff_sq,
                                          double min_sq = 0.0,
                                          bool inclusive_cutoff = false) noexcept {
  using Kernel = unsigned (*)(double, double, double, const double *,
                              const double *, const double *, double, double,
                              bool) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2() ? &point_vs_four_cutoff_mask_avx2
                              : &point_vs_four_cutoff_mask_scalar;
#else
    return &point_vs_four_cutoff_mask_scalar;
#endif
  }();
  return kernel(px, py, pz, xs, ys, zs, cutoff_sq, min_sq, inclusive_cutoff);
}

inline unsigned point_vs_gathered4_cutoff_mask_scalar(
    double px, double py, double pz, const double *xs, const double *ys,
    const double *zs, const std::int64_t *indices, double cutoff_sq,
    double min_sq, bool inclusive_cutoff) noexcept {
  unsigned mask = 0u;
  for (unsigned lane = 0; lane < 4; ++lane) {
    const std::size_t idx = static_cast<std::size_t>(indices[lane]);
    const double dx = px - xs[idx];
    const double dy = py - ys[idx];
    const double dz = pz - zs[idx];
    const double dist_sq = dx * dx + dy * dy + dz * dz;
    const bool cutoff_ok =
        inclusive_cutoff ? (dist_sq <= cutoff_sq) : (dist_sq < cutoff_sq);
    if (dist_sq > min_sq && cutoff_ok)
      mask |= (1u << lane);
  }
  return mask;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline unsigned point_vs_gathered4_cutoff_mask_avx2(
    double px, double py, double pz, const double *xs, const double *ys,
    const double *zs, const std::int64_t *indices, double cutoff_sq,
    double min_sq, bool inclusive_cutoff) noexcept {
  const __m256i idx = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(indices));
  const __m256d vpx = _mm256_set1_pd(px);
  const __m256d vpy = _mm256_set1_pd(py);
  const __m256d vpz = _mm256_set1_pd(pz);
  const __m256d vx = _mm256_i64gather_pd(xs, idx, 8);
  const __m256d vy = _mm256_i64gather_pd(ys, idx, 8);
  const __m256d vz = _mm256_i64gather_pd(zs, idx, 8);
  const __m256d dx = _mm256_sub_pd(vpx, vx);
  const __m256d dy = _mm256_sub_pd(vpy, vy);
  const __m256d dz = _mm256_sub_pd(vpz, vz);
  const __m256d dist_sq =
      _mm256_add_pd(_mm256_mul_pd(dx, dx),
                    _mm256_add_pd(_mm256_mul_pd(dy, dy), _mm256_mul_pd(dz, dz)));
  const __m256d cutoff_vec = _mm256_set1_pd(cutoff_sq);
  const __m256d min_vec = _mm256_set1_pd(min_sq);
  const __m256d cutoff_mask =
      inclusive_cutoff ? _mm256_cmp_pd(dist_sq, cutoff_vec, _CMP_LE_OQ)
                       : _mm256_cmp_pd(dist_sq, cutoff_vec, _CMP_LT_OQ);
  const __m256d min_mask = _mm256_cmp_pd(dist_sq, min_vec, _CMP_GT_OQ);
  return static_cast<unsigned>(
      _mm256_movemask_pd(_mm256_and_pd(cutoff_mask, min_mask)));
}
#endif

inline unsigned point_vs_gathered4_cutoff_mask(
    double px, double py, double pz, const double *xs, const double *ys,
    const double *zs, const std::int64_t *indices, double cutoff_sq,
    double min_sq = 0.0, bool inclusive_cutoff = false) noexcept {
  using Kernel =
      unsigned (*)(double, double, double, const double *, const double *,
                   const double *, const std::int64_t *, double, double,
                   bool) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2() ? &point_vs_gathered4_cutoff_mask_avx2
                              : &point_vs_gathered4_cutoff_mask_scalar;
#else
    return &point_vs_gathered4_cutoff_mask_scalar;
#endif
  }();
  return kernel(px, py, pz, xs, ys, zs, indices, cutoff_sq, min_sq,
                inclusive_cutoff);
}

inline unsigned point_vs_gathered4_variable_cutoff_mask_scalar(
    double px, double py, double pz, const double *xs, const double *ys,
    const double *zs, const std::int64_t *indices, const double *cutoff_sq,
    double min_sq, bool inclusive_cutoff) noexcept {
  unsigned mask = 0u;
  for (unsigned lane = 0; lane < 4; ++lane) {
    const std::size_t idx = static_cast<std::size_t>(indices[lane]);
    const double dx = px - xs[idx];
    const double dy = py - ys[idx];
    const double dz = pz - zs[idx];
    const double dist_sq = dx * dx + dy * dy + dz * dz;
    const bool cutoff_ok = inclusive_cutoff ? (dist_sq <= cutoff_sq[lane])
                                            : (dist_sq < cutoff_sq[lane]);
    if (dist_sq > min_sq && cutoff_ok)
      mask |= (1u << lane);
  }
  return mask;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline unsigned point_vs_gathered4_variable_cutoff_mask_avx2(
    double px, double py, double pz, const double *xs, const double *ys,
    const double *zs, const std::int64_t *indices, const double *cutoff_sq,
    double min_sq, bool inclusive_cutoff) noexcept {
  const __m256i idx = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(indices));
  const __m256d vpx = _mm256_set1_pd(px);
  const __m256d vpy = _mm256_set1_pd(py);
  const __m256d vpz = _mm256_set1_pd(pz);
  const __m256d vx = _mm256_i64gather_pd(xs, idx, 8);
  const __m256d vy = _mm256_i64gather_pd(ys, idx, 8);
  const __m256d vz = _mm256_i64gather_pd(zs, idx, 8);
  const __m256d dx = _mm256_sub_pd(vpx, vx);
  const __m256d dy = _mm256_sub_pd(vpy, vy);
  const __m256d dz = _mm256_sub_pd(vpz, vz);
  const __m256d dist_sq =
      _mm256_add_pd(_mm256_mul_pd(dx, dx),
                    _mm256_add_pd(_mm256_mul_pd(dy, dy), _mm256_mul_pd(dz, dz)));
  const __m256d cutoff_vec = _mm256_loadu_pd(cutoff_sq);
  const __m256d min_vec = _mm256_set1_pd(min_sq);
  const __m256d cutoff_mask =
      inclusive_cutoff ? _mm256_cmp_pd(dist_sq, cutoff_vec, _CMP_LE_OQ)
                       : _mm256_cmp_pd(dist_sq, cutoff_vec, _CMP_LT_OQ);
  const __m256d min_mask = _mm256_cmp_pd(dist_sq, min_vec, _CMP_GT_OQ);
  return static_cast<unsigned>(
      _mm256_movemask_pd(_mm256_and_pd(cutoff_mask, min_mask)));
}
#endif

inline unsigned point_vs_gathered4_variable_cutoff_mask(
    double px, double py, double pz, const double *xs, const double *ys,
    const double *zs, const std::int64_t *indices, const double *cutoff_sq,
    double min_sq = 0.0, bool inclusive_cutoff = false) noexcept {
  using Kernel =
      unsigned (*)(double, double, double, const double *, const double *,
                   const double *, const std::int64_t *, const double *,
                   double, bool) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2()
               ? &point_vs_gathered4_variable_cutoff_mask_avx2
               : &point_vs_gathered4_variable_cutoff_mask_scalar;
#else
    return &point_vs_gathered4_variable_cutoff_mask_scalar;
#endif
  }();
  return kernel(px, py, pz, xs, ys, zs, indices, cutoff_sq, min_sq,
                inclusive_cutoff);
}

inline double gbsa_inverse_r4_volume_sum_scalar(
    const double *x, const double *y, const double *z, const double *rho,
    std::size_t n, double cx, double cy, double cz, double min_sq = 1e-6,
    bool = false) noexcept {
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = x[i] - cx;
    const double dy = y[i] - cy;
    const double dz = z[i] - cz;
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > min_sq) {
      const double rho3 = rho[i] * rho[i] * rho[i];
      const double vol = rho3 / 3.0;
      total += vol / (r2 * r2);
    }
  }
  return total;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline double gbsa_inverse_r4_volume_sum_avx2(
    const double *x, const double *y, const double *z, const double *rho,
    std::size_t n, double cx, double cy, double cz, double min_sq = 1e-6,
    bool assume_aligned = false) noexcept {
  std::size_t i = 0;
  __m256d acc = _mm256_setzero_pd();
  const __m256d vcx = _mm256_set1_pd(cx);
  const __m256d vcy = _mm256_set1_pd(cy);
  const __m256d vcz = _mm256_set1_pd(cz);
  const __m256d vone = _mm256_set1_pd(1.0);
  const __m256d vthree = _mm256_set1_pd(3.0);
  const __m256d vmin = _mm256_set1_pd(min_sq);
  const bool aligned = assume_aligned && is_aligned_32(x) && is_aligned_32(y) &&
                       is_aligned_32(z) && is_aligned_32(rho);
  for (; i + 4 <= n; i += 4) {
    const __m256d vx = aligned ? _mm256_load_pd(x + i) : _mm256_loadu_pd(x + i);
    const __m256d vy = aligned ? _mm256_load_pd(y + i) : _mm256_loadu_pd(y + i);
    const __m256d vz = aligned ? _mm256_load_pd(z + i) : _mm256_loadu_pd(z + i);
    const __m256d vr =
        aligned ? _mm256_load_pd(rho + i) : _mm256_loadu_pd(rho + i);
    const __m256d dx = _mm256_sub_pd(vx, vcx);
    const __m256d dy = _mm256_sub_pd(vy, vcy);
    const __m256d dz = _mm256_sub_pd(vz, vcz);
    const __m256d r2 =
        _mm256_add_pd(_mm256_mul_pd(dx, dx),
                      _mm256_add_pd(_mm256_mul_pd(dy, dy), _mm256_mul_pd(dz, dz)));
    const __m256d mask = _mm256_cmp_pd(r2, vmin, _CMP_GT_OQ);
    const __m256d rho3 = _mm256_mul_pd(vr, _mm256_mul_pd(vr, vr));
    const __m256d vol = _mm256_div_pd(rho3, vthree);
    const __m256d inv_r2 = _mm256_div_pd(vone, r2);
    const __m256d term = _mm256_mul_pd(vol, _mm256_mul_pd(inv_r2, inv_r2));
    acc = _mm256_add_pd(acc, _mm256_and_pd(term, mask));
  }
  alignas(32) double lane[4];
  _mm256_store_pd(lane, acc);
  double total = lane[0] + lane[1] + lane[2] + lane[3];
  for (; i < n; ++i) {
    const double dx = x[i] - cx;
    const double dy = y[i] - cy;
    const double dz = z[i] - cz;
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > min_sq) {
      const double rho3 = rho[i] * rho[i] * rho[i];
      const double vol = rho3 / 3.0;
      total += vol / (r2 * r2);
    }
  }
  return total;
}

TITAN_TARGET_AVX2_FMA
inline double gbsa_inverse_r4_volume_sum_avx2_fma(
    const double *x, const double *y, const double *z, const double *rho,
    std::size_t n, double cx, double cy, double cz, double min_sq = 1e-6,
    bool assume_aligned = false) noexcept {
  std::size_t i = 0;
  __m256d acc = _mm256_setzero_pd();
  const __m256d vcx = _mm256_set1_pd(cx);
  const __m256d vcy = _mm256_set1_pd(cy);
  const __m256d vcz = _mm256_set1_pd(cz);
  const __m256d vone = _mm256_set1_pd(1.0);
  const __m256d vthree = _mm256_set1_pd(3.0);
  const __m256d vmin = _mm256_set1_pd(min_sq);
  const bool aligned = assume_aligned && is_aligned_32(x) && is_aligned_32(y) &&
                       is_aligned_32(z) && is_aligned_32(rho);
  for (; i + 4 <= n; i += 4) {
    const __m256d vx = aligned ? _mm256_load_pd(x + i) : _mm256_loadu_pd(x + i);
    const __m256d vy = aligned ? _mm256_load_pd(y + i) : _mm256_loadu_pd(y + i);
    const __m256d vz = aligned ? _mm256_load_pd(z + i) : _mm256_loadu_pd(z + i);
    const __m256d vr =
        aligned ? _mm256_load_pd(rho + i) : _mm256_loadu_pd(rho + i);
    const __m256d dx = _mm256_sub_pd(vx, vcx);
    const __m256d dy = _mm256_sub_pd(vy, vcy);
    const __m256d dz = _mm256_sub_pd(vz, vcz);
    const __m256d r2 =
        _mm256_fmadd_pd(dz, dz, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx)));
    const __m256d mask = _mm256_cmp_pd(r2, vmin, _CMP_GT_OQ);
    const __m256d rho3 = _mm256_mul_pd(vr, _mm256_mul_pd(vr, vr));
    const __m256d vol = _mm256_div_pd(rho3, vthree);
    const __m256d inv_r2 = _mm256_div_pd(vone, r2);
    const __m256d term = _mm256_mul_pd(vol, _mm256_mul_pd(inv_r2, inv_r2));
    acc = _mm256_add_pd(acc, _mm256_and_pd(term, mask));
  }
  alignas(32) double lane[4];
  _mm256_store_pd(lane, acc);
  double total = lane[0] + lane[1] + lane[2] + lane[3];
  for (; i < n; ++i) {
    const double dx = x[i] - cx;
    const double dy = y[i] - cy;
    const double dz = z[i] - cz;
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > min_sq) {
      const double rho3 = rho[i] * rho[i] * rho[i];
      const double vol = rho3 / 3.0;
      total += vol / (r2 * r2);
    }
  }
  return total;
}
#endif

inline double gbsa_inverse_r4_volume_sum(
    const double *x, const double *y, const double *z, const double *rho,
    std::size_t n, double cx, double cy, double cz, double min_sq = 1e-6,
    bool assume_aligned = false) noexcept {
  using Kernel = double (*)(const double *, const double *, const double *,
                            const double *, std::size_t, double, double, double,
                            double, bool) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    if (runtime_has_avx2_fma())
      return &gbsa_inverse_r4_volume_sum_avx2_fma;
    if (runtime_has_avx2())
      return &gbsa_inverse_r4_volume_sum_avx2;
    return &gbsa_inverse_r4_volume_sum_scalar;
#else
    return &gbsa_inverse_r4_volume_sum_scalar;
#endif
  }();
  return kernel(x, y, z, rho, n, cx, cy, cz, min_sq, assume_aligned);
}

inline void gbsa_lj_mixed_sigma_epsilon_r6_fourlanes_scalar(
    const double *sigma, const double *epsilon, const double *r6_nm,
    double *energy_out) noexcept {
  for (std::size_t lane = 0; lane < 4; ++lane) {
    if (!(sigma[lane] > 0.0) || epsilon[lane] == 0.0) {
      energy_out[lane] = 0.0;
      continue;
    }
    const double sig2 = sigma[lane] * sigma[lane];
    const double sig6 = sig2 * sig2 * sig2;
    const double sr6 = sig6 / r6_nm[lane];
    energy_out[lane] = 4.0 * epsilon[lane] * (sr6 * sr6 - sr6);
  }
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline void gbsa_lj_mixed_sigma_epsilon_r6_fourlanes_avx2(
    const double *sigma, const double *epsilon, const double *r6_nm,
    double *energy_out) noexcept {
  const __m256d sigma_vec = _mm256_loadu_pd(sigma);
  const __m256d epsilon_vec = _mm256_loadu_pd(epsilon);
  const __m256d r6_vec = _mm256_loadu_pd(r6_nm);
  const __m256d zero = _mm256_setzero_pd();
  const __m256d four = _mm256_set1_pd(4.0);
  const __m256d sigma_valid = _mm256_cmp_pd(sigma_vec, zero, _CMP_GT_OQ);
  const __m256d epsilon_valid = _mm256_cmp_pd(epsilon_vec, zero, _CMP_NEQ_OQ);
  const __m256d valid_mask = _mm256_and_pd(sigma_valid, epsilon_valid);
  const __m256d sig2 = _mm256_mul_pd(sigma_vec, sigma_vec);
  const __m256d sig6 = _mm256_mul_pd(_mm256_mul_pd(sig2, sig2), sig2);
  const __m256d sr6 = _mm256_div_pd(sig6, r6_vec);
  const __m256d sr12 = _mm256_mul_pd(sr6, sr6);
  const __m256d energy =
      _mm256_mul_pd(_mm256_mul_pd(four, epsilon_vec), _mm256_sub_pd(sr12, sr6));
  _mm256_storeu_pd(energy_out, _mm256_and_pd(energy, valid_mask));
}
#endif

inline void gbsa_lj_mixed_sigma_epsilon_r6_fourlanes(
    const double *sigma, const double *epsilon, const double *r6_nm,
    double *energy_out) noexcept {
  using Kernel =
      void (*)(const double *, const double *, const double *, double *) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2() ? &gbsa_lj_mixed_sigma_epsilon_r6_fourlanes_avx2
                              : &gbsa_lj_mixed_sigma_epsilon_r6_fourlanes_scalar;
#else
    return &gbsa_lj_mixed_sigma_epsilon_r6_fourlanes_scalar;
#endif
  }();
  kernel(sigma, epsilon, r6_nm, energy_out);
}

inline void lj_mixed_sigma_epsilon_distance_fourlanes_scalar(
    const double *sigma, const double *epsilon, const double *distance,
    double *energy_out) noexcept {
  for (std::size_t lane = 0; lane < 4; ++lane) {
    const double ratio = sigma[lane] / distance[lane];
    const double ratio2 = ratio * ratio;
    const double ratio4 = ratio2 * ratio2;
    const double c6 = ratio4 * ratio2;
    energy_out[lane] = epsilon[lane] * (c6 * c6 - c6);
  }
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline void lj_mixed_sigma_epsilon_distance_fourlanes_avx2(
    const double *sigma, const double *epsilon, const double *distance,
    double *energy_out) noexcept {
  const __m256d sigma_vec = _mm256_loadu_pd(sigma);
  const __m256d epsilon_vec = _mm256_loadu_pd(epsilon);
  const __m256d distance_vec = _mm256_loadu_pd(distance);
  const __m256d ratio = _mm256_div_pd(sigma_vec, distance_vec);
  const __m256d ratio2 = _mm256_mul_pd(ratio, ratio);
  const __m256d ratio4 = _mm256_mul_pd(ratio2, ratio2);
  const __m256d c6 = _mm256_mul_pd(ratio4, ratio2);
  const __m256d energy =
      _mm256_mul_pd(epsilon_vec, _mm256_sub_pd(_mm256_mul_pd(c6, c6), c6));
  _mm256_storeu_pd(energy_out, energy);
}
#endif

inline void lj_mixed_sigma_epsilon_distance_fourlanes(
    const double *sigma, const double *epsilon, const double *distance,
    double *energy_out) noexcept {
  using Kernel =
      void (*)(const double *, const double *, const double *, double *) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2() ? &lj_mixed_sigma_epsilon_distance_fourlanes_avx2
                              : &lj_mixed_sigma_epsilon_distance_fourlanes_scalar;
#else
    return &lj_mixed_sigma_epsilon_distance_fourlanes_scalar;
#endif
  }();
  kernel(sigma, epsilon, distance, energy_out);
}

inline void lj_mixed_sigma_epsilon_distance_c6_fourlanes_scalar(
    const double *sigma, const double *epsilon, const double *distance,
    double *c6_out, double *energy_out) noexcept {
  for (std::size_t lane = 0; lane < 4; ++lane) {
    const double ratio = sigma[lane] / distance[lane];
    const double ratio2 = ratio * ratio;
    const double ratio4 = ratio2 * ratio2;
    const double c6 = ratio4 * ratio2;
    c6_out[lane] = c6;
    energy_out[lane] = epsilon[lane] * (c6 * c6 - c6);
  }
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline void lj_mixed_sigma_epsilon_distance_c6_fourlanes_avx2(
    const double *sigma, const double *epsilon, const double *distance,
    double *c6_out, double *energy_out) noexcept {
  const __m256d sigma_vec = _mm256_loadu_pd(sigma);
  const __m256d epsilon_vec = _mm256_loadu_pd(epsilon);
  const __m256d distance_vec = _mm256_loadu_pd(distance);
  const __m256d ratio = _mm256_div_pd(sigma_vec, distance_vec);
  const __m256d ratio2 = _mm256_mul_pd(ratio, ratio);
  const __m256d ratio4 = _mm256_mul_pd(ratio2, ratio2);
  const __m256d c6 = _mm256_mul_pd(ratio4, ratio2);
  const __m256d energy =
      _mm256_mul_pd(epsilon_vec, _mm256_sub_pd(_mm256_mul_pd(c6, c6), c6));
  _mm256_storeu_pd(c6_out, c6);
  _mm256_storeu_pd(energy_out, energy);
}
#endif

inline void lj_mixed_sigma_epsilon_distance_c6_fourlanes(
    const double *sigma, const double *epsilon, const double *distance,
    double *c6_out, double *energy_out) noexcept {
  using Kernel = void (*)(const double *, const double *, const double *, double *,
                          double *) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2()
               ? &lj_mixed_sigma_epsilon_distance_c6_fourlanes_avx2
               : &lj_mixed_sigma_epsilon_distance_c6_fourlanes_scalar;
#else
    return &lj_mixed_sigma_epsilon_distance_c6_fourlanes_scalar;
#endif
  }();
  kernel(sigma, epsilon, distance, c6_out, energy_out);
}

inline void lj_sigma_product_distance_c6_fourlanes_scalar(
    const double *sigma_product, const double *epsilon, const double *distance,
    double *c6_out, double *energy_out) noexcept {
  for (std::size_t lane = 0; lane < 4; ++lane) {
    const double ratio = sigma_product[lane] / distance[lane];
    const double rooted_ratio = std::sqrt(ratio);
    const double restored_ratio = rooted_ratio * rooted_ratio;
    const double c6 =
        restored_ratio * restored_ratio * restored_ratio;
    c6_out[lane] = c6;
    energy_out[lane] = epsilon[lane] * (c6 * c6 - c6);
  }
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline void lj_sigma_product_distance_c6_fourlanes_avx2(
    const double *sigma_product, const double *epsilon, const double *distance,
    double *c6_out, double *energy_out) noexcept {
  const __m256d sigma_product_vec = _mm256_loadu_pd(sigma_product);
  const __m256d epsilon_vec = _mm256_loadu_pd(epsilon);
  const __m256d distance_vec = _mm256_loadu_pd(distance);
  const __m256d ratio = _mm256_div_pd(sigma_product_vec, distance_vec);
  const __m256d rooted_ratio = _mm256_sqrt_pd(ratio);
  const __m256d restored_ratio = _mm256_mul_pd(rooted_ratio, rooted_ratio);
  const __m256d c6 = _mm256_mul_pd(
      _mm256_mul_pd(restored_ratio, restored_ratio), restored_ratio);
  const __m256d energy =
      _mm256_mul_pd(epsilon_vec, _mm256_sub_pd(_mm256_mul_pd(c6, c6), c6));
  _mm256_storeu_pd(c6_out, c6);
  _mm256_storeu_pd(energy_out, energy);
}
#endif

inline void lj_sigma_product_distance_c6_fourlanes(
    const double *sigma_product, const double *epsilon, const double *distance,
    double *c6_out, double *energy_out) noexcept {
  using Kernel = void (*)(const double *, const double *, const double *, double *,
                          double *) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2()
               ? &lj_sigma_product_distance_c6_fourlanes_avx2
               : &lj_sigma_product_distance_c6_fourlanes_scalar;
#else
    return &lj_sigma_product_distance_c6_fourlanes_scalar;
#endif
  }();
  kernel(sigma_product, epsilon, distance, c6_out, energy_out);
}

struct XorOverlapCounts {
  std::uint64_t overlap = 0;
  std::uint64_t reference = 0;
};

inline XorOverlapCounts xor_overlap_popcount_scalar(
    const std::uint64_t *lhs, const std::uint64_t *rhs,
    const std::uint64_t *mean_bits, std::size_t count_words) noexcept {
  XorOverlapCounts counts{};
  for (std::size_t i = 0; i < count_words; ++i) {
    const std::uint64_t dl = lhs[i] ^ mean_bits[i];
    const std::uint64_t dr = rhs[i] ^ mean_bits[i];
    counts.overlap += static_cast<std::uint64_t>(std::popcount(dl & dr));
    counts.reference += static_cast<std::uint64_t>(std::popcount(dl));
  }
  return counts;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline XorOverlapCounts xor_overlap_popcount_avx2(
    const std::uint64_t *lhs, const std::uint64_t *rhs,
    const std::uint64_t *mean_bits, std::size_t count_words) noexcept {
  XorOverlapCounts counts{};
  const __m256i lookup = _mm256_setr_epi8(
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3,
      1, 2, 2, 3, 2, 3, 3, 4);
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  __m256i acc_num = _mm256_setzero_si256();
  __m256i acc_den = _mm256_setzero_si256();

  const std::size_t n4 = count_words & ~std::size_t{3};
  for (std::size_t i = 0; i < n4; i += 4) {
    const __m256i vl =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lhs + i));
    const __m256i vr =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rhs + i));
    const __m256i vm =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(mean_bits + i));
    const __m256i dl = _mm256_xor_si256(vl, vm);
    const __m256i dr = _mm256_xor_si256(vr, vm);
    const __m256i overlap = _mm256_and_si256(dl, dr);

    __m256i lo = _mm256_and_si256(overlap, low_mask);
    __m256i hi =
        _mm256_and_si256(_mm256_srli_epi16(overlap, 4), low_mask);
    __m256i cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo),
                                  _mm256_shuffle_epi8(lookup, hi));
    acc_num = _mm256_add_epi64(
        acc_num, _mm256_sad_epu8(cnt, _mm256_setzero_si256()));

    lo = _mm256_and_si256(dl, low_mask);
    hi = _mm256_and_si256(_mm256_srli_epi16(dl, 4), low_mask);
    cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo),
                          _mm256_shuffle_epi8(lookup, hi));
    acc_den = _mm256_add_epi64(
        acc_den, _mm256_sad_epu8(cnt, _mm256_setzero_si256()));
  }

  alignas(32) std::uint64_t buf_num[4];
  alignas(32) std::uint64_t buf_den[4];
  _mm256_store_si256(reinterpret_cast<__m256i *>(buf_num), acc_num);
  _mm256_store_si256(reinterpret_cast<__m256i *>(buf_den), acc_den);
  counts.overlap = buf_num[0] + buf_num[1] + buf_num[2] + buf_num[3];
  counts.reference = buf_den[0] + buf_den[1] + buf_den[2] + buf_den[3];

  for (std::size_t i = n4; i < count_words; ++i) {
    const std::uint64_t dl = lhs[i] ^ mean_bits[i];
    const std::uint64_t dr = rhs[i] ^ mean_bits[i];
    counts.overlap += static_cast<std::uint64_t>(std::popcount(dl & dr));
    counts.reference += static_cast<std::uint64_t>(std::popcount(dl));
  }
  return counts;
}
#endif

inline XorOverlapCounts xor_overlap_popcount(
    const std::uint64_t *lhs, const std::uint64_t *rhs,
    const std::uint64_t *mean_bits, std::size_t count_words) noexcept {
  using Kernel = XorOverlapCounts (*)(const std::uint64_t *, const std::uint64_t *,
                                      const std::uint64_t *, std::size_t) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    return runtime_has_avx2() ? &xor_overlap_popcount_avx2
                              : &xor_overlap_popcount_scalar;
#else
    return &xor_overlap_popcount_scalar;
#endif
  }();
  return kernel(lhs, rhs, mean_bits, count_words);
}

inline void compute_4dist_scalar(double cax_i, double cay_i, double caz_i,
                                 double cbx_i, double cby_i, double cbz_i,
                                 double cax_j, double cay_j, double caz_j,
                                 double cbx_j, double cby_j, double cbz_j,
                                 double &d1, double &d2, double &d3,
                                 double &d4) noexcept {
  double dx, dy, dz;
  dx = cax_i - cax_j;
  dy = cay_i - cay_j;
  dz = caz_i - caz_j;
  d1 = std::sqrt(dx * dx + dy * dy + dz * dz);
  dx = cax_i - cbx_j;
  dy = cay_i - cby_j;
  dz = caz_i - cbz_j;
  d2 = std::sqrt(dx * dx + dy * dy + dz * dz);
  dx = cbx_i - cax_j;
  dy = cby_i - cay_j;
  dz = cbz_i - caz_j;
  d3 = std::sqrt(dx * dx + dy * dy + dz * dz);
  dx = cbx_i - cbx_j;
  dy = cby_i - cby_j;
  dz = cbz_i - cbz_j;
  d4 = std::sqrt(dx * dx + dy * dy + dz * dz);
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline void compute_4dist_avx2(double cax_i, double cay_i, double caz_i,
                               double cbx_i, double cby_i, double cbz_i,
                               double cax_j, double cay_j, double caz_j,
                               double cbx_j, double cby_j, double cbz_j,
                               double &d1, double &d2, double &d3,
                               double &d4) noexcept {
  const __m256d dx =
      _mm256_set_pd(cbx_i - cbx_j, cbx_i - cax_j, cax_i - cbx_j, cax_i - cax_j);
  const __m256d dy =
      _mm256_set_pd(cby_i - cby_j, cby_i - cay_j, cay_i - cby_j, cay_i - cay_j);
  const __m256d dz =
      _mm256_set_pd(cbz_i - cbz_j, cbz_i - caz_j, caz_i - cbz_j, caz_i - caz_j);
  const __m256d r2 =
      _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                    _mm256_mul_pd(dz, dz));
  const __m256d r = _mm256_sqrt_pd(r2);
  alignas(32) double buf[4];
  _mm256_store_pd(buf, r);
  d1 = buf[0];
  d2 = buf[1];
  d3 = buf[2];
  d4 = buf[3];
}

TITAN_TARGET_AVX2_FMA
inline void compute_4dist_avx2_fma(double cax_i, double cay_i, double caz_i,
                                   double cbx_i, double cby_i, double cbz_i,
                                   double cax_j, double cay_j, double caz_j,
                                   double cbx_j, double cby_j, double cbz_j,
                                   double &d1, double &d2, double &d3,
                                   double &d4) noexcept {
  const __m256d dx =
      _mm256_set_pd(cbx_i - cbx_j, cbx_i - cax_j, cax_i - cbx_j, cax_i - cax_j);
  const __m256d dy =
      _mm256_set_pd(cby_i - cby_j, cby_i - cay_j, cay_i - cby_j, cay_i - cay_j);
  const __m256d dz =
      _mm256_set_pd(cbz_i - cbz_j, cbz_i - caz_j, caz_i - cbz_j, caz_i - caz_j);
  const __m256d r2 =
      _mm256_fmadd_pd(dz, dz, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx)));
  const __m256d r = _mm256_sqrt_pd(r2);
  alignas(32) double buf[4];
  _mm256_store_pd(buf, r);
  d1 = buf[0];
  d2 = buf[1];
  d3 = buf[2];
  d4 = buf[3];
}
#endif

inline void compute_4dist(double cax_i, double cay_i, double caz_i,
                          double cbx_i, double cby_i, double cbz_i,
                          double cax_j, double cay_j, double caz_j,
                          double cbx_j, double cby_j, double cbz_j, double &d1,
                          double &d2, double &d3, double &d4) noexcept {
  using Kernel =
      void (*)(double, double, double, double, double, double, double, double,
               double, double, double, double, double &, double &, double &,
               double &) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    if (runtime_has_avx2_fma())
      return &compute_4dist_avx2_fma;
    if (runtime_has_avx2())
      return &compute_4dist_avx2;
    return &compute_4dist_scalar;
#else
    return &compute_4dist_scalar;
#endif
  }();
  kernel(cax_i, cay_i, caz_i, cbx_i, cby_i, cbz_i, cax_j, cay_j, caz_j, cbx_j,
         cby_j, cbz_j, d1, d2, d3, d4);
}

} // namespace titan_simd_dispatch

#endif
