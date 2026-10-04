#include "titan_s2_kernels.h"

#include "BoxMatrix.h"
#include "titan_executor.h"
#include "titan_simd_dispatch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace titan_s2_kernel {
namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr std::size_t kLogScanPoints = 64;

struct CorrelationAccum {
  double sum{0.0};
  std::uint64_t samples{0};
};

bool consistent(const BondVectorsSoA &vectors) {
  return vectors.y.size() == vectors.size() &&
         vectors.z.size() == vectors.size() &&
         vectors.valid.size() == vectors.size();
}

void validate_vectors(const BondVectorsSoA &vectors) {
  if (!consistent(vectors))
    throw std::invalid_argument("S2: malformed bond-vector SoA.");
}

bool valid_time_axis(std::span<const double> time_ps) {
  for (std::size_t i = 0; i < time_ps.size(); ++i) {
    if (!std::isfinite(time_ps[i]) ||
        (i > 0 && !(time_ps[i] > time_ps[i - 1])))
      return false;
  }
  return true;
}

void validate_time_axis(const BondVectorsSoA &vectors,
                        std::span<const double> time_ps) {
  if (time_ps.size() != vectors.size() || !valid_time_axis(time_ps))
    throw std::invalid_argument(
        "S2: time axis must match vectors and increase strictly.");
}

void validate_lag(const BondVectorsSoA &vectors, std::size_t lag) {
  if (lag == 0 || lag >= vectors.size())
    throw std::out_of_range("S2: lag must be in [1, vector_count).");
}

double p2(double dot) { return 0.5 * (3.0 * dot * dot - 1.0); }

void add_scalar_sample(const BondVectorsSoA &vectors, std::size_t origin,
                       std::size_t lag, CorrelationAccum &accum) {
  if (!vectors.valid[origin] || !vectors.valid[origin + lag])
    return;
  const double dot = vectors.x[origin] * vectors.x[origin + lag] +
                     vectors.y[origin] * vectors.y[origin + lag] +
                     vectors.z[origin] * vectors.z[origin + lag];
  accum.sum += p2(std::clamp(dot, -1.0, 1.0));
  ++accum.samples;
}

CorrelationAccum reduce_scalar_range(const BondVectorsSoA &vectors,
                                     std::size_t lag, std::size_t begin,
                                     std::size_t end) {
  CorrelationAccum accum;
  for (std::size_t origin = begin; origin < end; ++origin)
    add_scalar_sample(vectors, origin, lag, accum);
  return accum;
}

CorrelationAccum reduce_scalar(const BondVectorsSoA &vectors,
                               std::size_t lag) noexcept {
  if (!consistent(vectors) || lag == 0 || lag >= vectors.size())
    return {};
  return reduce_scalar_range(vectors, lag, 0, vectors.size() - lag);
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
CorrelationAccum reduce_avx2(const BondVectorsSoA &vectors,
                             std::size_t lag) noexcept {
  if (!consistent(vectors) || lag == 0 || lag >= vectors.size())
    return {};
  const std::size_t origins = vectors.size() - lag;
  std::size_t origin = 0;
  CorrelationAccum accum;
  __m256d sum = _mm256_setzero_pd();
  const __m256d three = _mm256_set1_pd(3.0);
  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d minus_one = _mm256_set1_pd(-1.0);
  for (; origin + 4 <= origins; origin += 4) {
    bool all_valid = true;
    for (std::size_t lane = 0; lane < 4; ++lane)
      all_valid = all_valid && vectors.valid[origin + lane] &&
                  vectors.valid[origin + lane + lag];
    if (!all_valid) {
      for (std::size_t lane = 0; lane < 4; ++lane)
        add_scalar_sample(vectors, origin + lane, lag, accum);
      continue;
    }
    const __m256d x0 = _mm256_loadu_pd(vectors.x.data() + origin);
    const __m256d y0 = _mm256_loadu_pd(vectors.y.data() + origin);
    const __m256d z0 = _mm256_loadu_pd(vectors.z.data() + origin);
    const __m256d x1 = _mm256_loadu_pd(vectors.x.data() + origin + lag);
    const __m256d y1 = _mm256_loadu_pd(vectors.y.data() + origin + lag);
    const __m256d z1 = _mm256_loadu_pd(vectors.z.data() + origin + lag);
    __m256d dot =
        _mm256_add_pd(_mm256_mul_pd(x0, x1),
                      _mm256_add_pd(_mm256_mul_pd(y0, y1),
                                    _mm256_mul_pd(z0, z1)));
    dot = _mm256_max_pd(minus_one, _mm256_min_pd(one, dot));
    const __m256d dot_sq = _mm256_mul_pd(dot, dot);
    const __m256d values = _mm256_mul_pd(
        half, _mm256_sub_pd(_mm256_mul_pd(three, dot_sq), one));
    sum = _mm256_add_pd(sum, values);
    accum.samples += 4;
  }
  alignas(32) std::array<double, 4> lanes{};
  _mm256_store_pd(lanes.data(), sum);
  accum.sum += lanes[0] + lanes[1] + lanes[2] + lanes[3];
  const CorrelationAccum tail =
      reduce_scalar_range(vectors, lag, origin, origins);
  accum.sum += tail.sum;
  accum.samples += tail.samples;
  return accum;
}

TITAN_TARGET_AVX2_FMA
CorrelationAccum reduce_avx2_fma(const BondVectorsSoA &vectors,
                                 std::size_t lag) noexcept {
  if (!consistent(vectors) || lag == 0 || lag >= vectors.size())
    return {};
  const std::size_t origins = vectors.size() - lag;
  std::size_t origin = 0;
  CorrelationAccum accum;
  __m256d sum = _mm256_setzero_pd();
  const __m256d three = _mm256_set1_pd(3.0);
  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d minus_one = _mm256_set1_pd(-1.0);
  for (; origin + 4 <= origins; origin += 4) {
    bool all_valid = true;
    for (std::size_t lane = 0; lane < 4; ++lane)
      all_valid = all_valid && vectors.valid[origin + lane] &&
                  vectors.valid[origin + lane + lag];
    if (!all_valid) {
      for (std::size_t lane = 0; lane < 4; ++lane)
        add_scalar_sample(vectors, origin + lane, lag, accum);
      continue;
    }
    const __m256d x0 = _mm256_loadu_pd(vectors.x.data() + origin);
    const __m256d y0 = _mm256_loadu_pd(vectors.y.data() + origin);
    const __m256d z0 = _mm256_loadu_pd(vectors.z.data() + origin);
    const __m256d x1 = _mm256_loadu_pd(vectors.x.data() + origin + lag);
    const __m256d y1 = _mm256_loadu_pd(vectors.y.data() + origin + lag);
    const __m256d z1 = _mm256_loadu_pd(vectors.z.data() + origin + lag);
    __m256d dot = _mm256_fmadd_pd(
        z0, z1, _mm256_fmadd_pd(y0, y1, _mm256_mul_pd(x0, x1)));
    dot = _mm256_max_pd(minus_one, _mm256_min_pd(one, dot));
    const __m256d dot_sq = _mm256_mul_pd(dot, dot);
    const __m256d values = _mm256_mul_pd(
        half, _mm256_sub_pd(_mm256_mul_pd(three, dot_sq), one));
    sum = _mm256_add_pd(sum, values);
    accum.samples += 4;
  }
  alignas(32) std::array<double, 4> lanes{};
  _mm256_store_pd(lanes.data(), sum);
  accum.sum += lanes[0] + lanes[1] + lanes[2] + lanes[3];
  const CorrelationAccum tail =
      reduce_scalar_range(vectors, lag, origin, origins);
  accum.sum += tail.sum;
  accum.samples += tail.samples;
  return accum;
}
#endif

CorrelationAccum reduce_optimized(const BondVectorsSoA &vectors,
                                  std::size_t lag) noexcept {
  using Kernel = CorrelationAccum (*)(const BondVectorsSoA &,
                                      std::size_t) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    if (titan_simd_dispatch::runtime_has_avx2_fma())
      return &reduce_avx2_fma;
    if (titan_simd_dispatch::runtime_has_avx2())
      return &reduce_avx2;
#endif
    return &reduce_scalar;
  }();
  return kernel(vectors, lag);
}

CorrelationRow make_row(std::size_t lag, CorrelationAccum accum) {
  CorrelationRow row;
  row.lag_index = lag;
  row.samples = accum.samples;
  if (row.samples > 0)
    row.correlation = accum.sum / static_cast<double>(row.samples);
  else
    row.correlation = std::numeric_limits<double>::quiet_NaN();
  return row;
}

void set_lag_time(CorrelationRow &row, const BondVectorsSoA &vectors,
                  std::span<const double> time_ps) {
  double total = 0.0;
  std::uint64_t samples = 0;
  for (std::size_t origin = 0; origin + row.lag_index < vectors.size();
       ++origin) {
    if (!vectors.valid[origin] || !vectors.valid[origin + row.lag_index])
      continue;
    total += time_ps[origin + row.lag_index] - time_ps[origin];
    ++samples;
  }
  if (samples > 0)
    row.lag_time_ps = total / static_cast<double>(samples);
}

std::vector<CorrelationRow>
usable_rows(std::span<const CorrelationRow> rows) {
  std::vector<CorrelationRow> usable;
  usable.reserve(rows.size());
  for (const CorrelationRow &row : rows)
    if (row.samples > 0)
      usable.push_back(row);
  return usable;
}

bool valid_fit_rows(std::span<const CorrelationRow> rows) {
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (!std::isfinite(rows[i].lag_time_ps) || !(rows[i].lag_time_ps > 0.0) ||
        !std::isfinite(rows[i].correlation) ||
        (i > 0 && !(rows[i].lag_time_ps > rows[i - 1].lag_time_ps)))
      return false;
  }
  return true;
}

FitResult tail_fallback(std::span<const CorrelationRow> rows) {
  FitResult fit;
  fit.used_tail_fallback = true;
  const std::vector<CorrelationRow> usable = usable_rows(rows);
  if (usable.size() < 3 || !valid_fit_rows(usable))
    return fit;
  const std::size_t wanted = std::max<std::size_t>(
      3, static_cast<std::size_t>(std::ceil(0.2 * usable.size())));
  fit.tail_rows = std::min(wanted, usable.size());
  const std::size_t begin = usable.size() - fit.tail_rows;
  double weighted_sum = 0.0;
  double weight_sum = 0.0;
  for (std::size_t i = begin; i < usable.size(); ++i) {
    const double weight =
        static_cast<double>(usable[i].samples);
    weighted_sum += weight * usable[i].correlation;
    weight_sum += weight;
  }
  if (!(weight_sum > 0.0))
    return fit;
  fit.raw_plateau = weighted_sum / weight_sum;
  fit.s2 = std::clamp(fit.raw_plateau, 0.0, 1.0);
  const double last_time = usable.back().lag_time_ps;
  fit.tau_ps =
      std::isfinite(last_time) && last_time > 0.0 ? last_time : 1.0;
  fit.ok = true;
  return fit;
}

struct FitCandidate {
  bool ok{false};
  double tau_ps{0.0};
  double raw_plateau{0.0};
  double residual{std::numeric_limits<double>::infinity()};
};

FitCandidate evaluate_tau(std::span<const CorrelationRow> rows,
                          double tau_ps) {
  FitCandidate candidate;
  candidate.tau_ps = tau_ps;
  if (!(tau_ps > 0.0) || !std::isfinite(tau_ps))
    return candidate;
  double numerator = 0.0;
  double denominator = 0.0;
  for (const CorrelationRow &row : rows) {
    const double decay = std::exp(-row.lag_time_ps / tau_ps);
    const double x = 1.0 - decay;
    const double y = row.correlation - decay;
    const double weight =
        static_cast<double>(std::max<std::uint64_t>(1, row.samples));
    numerator += weight * x * y;
    denominator += weight * x * x;
  }
  if (!(denominator > kEpsilon) || !std::isfinite(denominator))
    return candidate;
  candidate.raw_plateau = numerator / denominator;
  candidate.residual = 0.0;
  for (const CorrelationRow &row : rows) {
    const double decay = std::exp(-row.lag_time_ps / tau_ps);
    const double model =
        candidate.raw_plateau + (1.0 - candidate.raw_plateau) * decay;
    const double delta = model - row.correlation;
    const double weight =
        static_cast<double>(std::max<std::uint64_t>(1, row.samples));
    candidate.residual += weight * delta * delta;
  }
  candidate.ok = std::isfinite(candidate.raw_plateau) &&
                 std::isfinite(candidate.residual);
  return candidate;
}

std::size_t validate_coordinate_spans(std::span<const double> from_x,
                                      std::span<const double> from_y,
                                      std::span<const double> from_z,
                                      std::span<const double> to_x,
                                      std::span<const double> to_y,
                                      std::span<const double> to_z) {
  const std::size_t count = from_x.size();
  if (from_y.size() != count || from_z.size() != count ||
      to_x.size() != count || to_y.size() != count || to_z.size() != count)
    throw std::invalid_argument(
        "S2: bond-vector coordinate spans must have equal sizes.");
  return count;
}

void resize_vectors(BondVectorsSoA &out, std::size_t count) {
  out.x.resize(count);
  out.y.resize(count);
  out.z.resize(count);
  out.valid.resize(count);
}

std::optional<BoxMatrix> valid_matrix(const box *frame_box) {
  return frame_box != nullptr && BoxMatrix::valid_box(*frame_box)
             ? std::optional<BoxMatrix>{*frame_box}
             : std::nullopt;
}

void store_normalized(double dx, double dy, double dz, BondVectorsSoA &out,
                      std::size_t i) {
  const double norm = std::sqrt(dx * dx + dy * dy + dz * dz);
  const bool valid = std::isfinite(norm) && norm > kEpsilon;
  if (valid) {
    dx /= norm;
    dy /= norm;
    dz /= norm;
  } else {
    dx = 0.0;
    dy = 0.0;
    dz = 0.0;
  }
  out.x[i] = dx;
  out.y[i] = dy;
  out.z[i] = dz;
  out.valid[i] = valid ? 1 : 0;
}

void build_vectors_scalar_range(std::span<const double> from_x,
                                std::span<const double> from_y,
                                std::span<const double> from_z,
                                std::span<const double> to_x,
                                std::span<const double> to_y,
                                std::span<const double> to_z,
                                const BoxMatrix *matrix, std::size_t begin,
                                std::size_t end, BondVectorsSoA &out) {
  for (std::size_t i = begin; i < end; ++i) {
    double dx = to_x[i] - from_x[i];
    double dy = to_y[i] - from_y[i];
    double dz = to_z[i] - from_z[i];
    if (matrix != nullptr)
      matrix->apply_mic(dx, dy, dz);
    store_normalized(dx, dy, dz, out, i);
  }
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
void apply_mic_avx2(const BoxMatrix &matrix, __m256d &dx, __m256d &dy,
                    __m256d &dz) {
#if TITAN_COMPILED_WITH_AVX
  matrix.apply_mic_avx(dx, dy, dz);
#else
  alignas(32) std::array<double, 4> x{};
  alignas(32) std::array<double, 4> y{};
  alignas(32) std::array<double, 4> z{};
  _mm256_store_pd(x.data(), dx);
  _mm256_store_pd(y.data(), dy);
  _mm256_store_pd(z.data(), dz);
  for (std::size_t lane = 0; lane < 4; ++lane)
    matrix.apply_mic(x[lane], y[lane], z[lane]);
  dx = _mm256_load_pd(x.data());
  dy = _mm256_load_pd(y.data());
  dz = _mm256_load_pd(z.data());
#endif
}

TITAN_TARGET_AVX2
void store_normalized_avx2(__m256d dx, __m256d dy, __m256d dz,
                           std::size_t i, BondVectorsSoA &out) {
  const __m256d zero = _mm256_setzero_pd();
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d min_sq = _mm256_set1_pd(kEpsilon * kEpsilon);
  const __m256d infinity =
      _mm256_set1_pd(std::numeric_limits<double>::infinity());
  const __m256d norm_sq =
      _mm256_add_pd(_mm256_mul_pd(dx, dx),
                    _mm256_add_pd(_mm256_mul_pd(dy, dy),
                                  _mm256_mul_pd(dz, dz)));
  const __m256d valid =
      _mm256_and_pd(_mm256_cmp_pd(norm_sq, min_sq, _CMP_GT_OQ),
                    _mm256_cmp_pd(norm_sq, infinity, _CMP_LT_OQ));
  const __m256d inv_norm = _mm256_div_pd(one, _mm256_sqrt_pd(norm_sq));
  _mm256_storeu_pd(out.x.data() + i,
                   _mm256_and_pd(valid, _mm256_mul_pd(dx, inv_norm)));
  _mm256_storeu_pd(out.y.data() + i,
                   _mm256_and_pd(valid, _mm256_mul_pd(dy, inv_norm)));
  _mm256_storeu_pd(out.z.data() + i,
                   _mm256_and_pd(valid, _mm256_mul_pd(dz, inv_norm)));
  const unsigned mask =
      static_cast<unsigned>(_mm256_movemask_pd(_mm256_or_pd(valid, zero)));
  for (std::size_t lane = 0; lane < 4; ++lane)
    out.valid[i + lane] = (mask & (1u << lane)) != 0u ? 1 : 0;
}

TITAN_TARGET_AVX2_FMA
void store_normalized_avx2_fma(__m256d dx, __m256d dy, __m256d dz,
                               std::size_t i, BondVectorsSoA &out) {
  const __m256d zero = _mm256_setzero_pd();
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d min_sq = _mm256_set1_pd(kEpsilon * kEpsilon);
  const __m256d infinity =
      _mm256_set1_pd(std::numeric_limits<double>::infinity());
  const __m256d norm_sq = _mm256_fmadd_pd(
      dz, dz, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx)));
  const __m256d valid =
      _mm256_and_pd(_mm256_cmp_pd(norm_sq, min_sq, _CMP_GT_OQ),
                    _mm256_cmp_pd(norm_sq, infinity, _CMP_LT_OQ));
  const __m256d inv_norm = _mm256_div_pd(one, _mm256_sqrt_pd(norm_sq));
  _mm256_storeu_pd(out.x.data() + i,
                   _mm256_and_pd(valid, _mm256_mul_pd(dx, inv_norm)));
  _mm256_storeu_pd(out.y.data() + i,
                   _mm256_and_pd(valid, _mm256_mul_pd(dy, inv_norm)));
  _mm256_storeu_pd(out.z.data() + i,
                   _mm256_and_pd(valid, _mm256_mul_pd(dz, inv_norm)));
  const unsigned mask =
      static_cast<unsigned>(_mm256_movemask_pd(_mm256_or_pd(valid, zero)));
  for (std::size_t lane = 0; lane < 4; ++lane)
    out.valid[i + lane] = (mask & (1u << lane)) != 0u ? 1 : 0;
}

TITAN_TARGET_AVX2
void build_vectors_avx2(std::span<const double> from_x,
                        std::span<const double> from_y,
                        std::span<const double> from_z,
                        std::span<const double> to_x,
                        std::span<const double> to_y,
                        std::span<const double> to_z, const BoxMatrix *matrix,
                        BondVectorsSoA &out) {
  std::size_t i = 0;
  for (; i + 4 <= out.size(); i += 4) {
    __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(to_x.data() + i),
                               _mm256_loadu_pd(from_x.data() + i));
    __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(to_y.data() + i),
                               _mm256_loadu_pd(from_y.data() + i));
    __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(to_z.data() + i),
                               _mm256_loadu_pd(from_z.data() + i));
    if (matrix != nullptr)
      apply_mic_avx2(*matrix, dx, dy, dz);
    store_normalized_avx2(dx, dy, dz, i, out);
  }
  build_vectors_scalar_range(from_x, from_y, from_z, to_x, to_y, to_z, matrix,
                             i, out.size(), out);
}

TITAN_TARGET_AVX2_FMA
void build_vectors_avx2_fma(std::span<const double> from_x,
                            std::span<const double> from_y,
                            std::span<const double> from_z,
                            std::span<const double> to_x,
                            std::span<const double> to_y,
                            std::span<const double> to_z,
                            const BoxMatrix *matrix, BondVectorsSoA &out) {
  std::size_t i = 0;
  for (; i + 4 <= out.size(); i += 4) {
    __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(to_x.data() + i),
                               _mm256_loadu_pd(from_x.data() + i));
    __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(to_y.data() + i),
                               _mm256_loadu_pd(from_y.data() + i));
    __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(to_z.data() + i),
                               _mm256_loadu_pd(from_z.data() + i));
    if (matrix != nullptr)
      apply_mic_avx2(*matrix, dx, dy, dz);
    store_normalized_avx2_fma(dx, dy, dz, i, out);
  }
  build_vectors_scalar_range(from_x, from_y, from_z, to_x, to_y, to_z, matrix,
                             i, out.size(), out);
}

TITAN_TARGET_AVX2
void rotate_vectors_avx2(BondVectorsSoA &vectors,
                         const std::array<double, 9> &r) {
  const __m256d r0 = _mm256_set1_pd(r[0]);
  const __m256d r1 = _mm256_set1_pd(r[1]);
  const __m256d r2 = _mm256_set1_pd(r[2]);
  const __m256d r3 = _mm256_set1_pd(r[3]);
  const __m256d r4 = _mm256_set1_pd(r[4]);
  const __m256d r5 = _mm256_set1_pd(r[5]);
  const __m256d r6 = _mm256_set1_pd(r[6]);
  const __m256d r7 = _mm256_set1_pd(r[7]);
  const __m256d r8 = _mm256_set1_pd(r[8]);
  std::size_t i = 0;
  for (; i + 4 <= vectors.size(); i += 4) {
    const __m256d x = _mm256_loadu_pd(vectors.x.data() + i);
    const __m256d y = _mm256_loadu_pd(vectors.y.data() + i);
    const __m256d z = _mm256_loadu_pd(vectors.z.data() + i);
    _mm256_storeu_pd(vectors.x.data() + i,
                     _mm256_add_pd(_mm256_mul_pd(r0, x),
                                   _mm256_add_pd(_mm256_mul_pd(r1, y),
                                                 _mm256_mul_pd(r2, z))));
    _mm256_storeu_pd(vectors.y.data() + i,
                     _mm256_add_pd(_mm256_mul_pd(r3, x),
                                   _mm256_add_pd(_mm256_mul_pd(r4, y),
                                                 _mm256_mul_pd(r5, z))));
    _mm256_storeu_pd(vectors.z.data() + i,
                     _mm256_add_pd(_mm256_mul_pd(r6, x),
                                   _mm256_add_pd(_mm256_mul_pd(r7, y),
                                                 _mm256_mul_pd(r8, z))));
  }
  for (; i < vectors.size(); ++i) {
    const double x = vectors.x[i];
    const double y = vectors.y[i];
    const double z = vectors.z[i];
    vectors.x[i] = r[0] * x + r[1] * y + r[2] * z;
    vectors.y[i] = r[3] * x + r[4] * y + r[5] * z;
    vectors.z[i] = r[6] * x + r[7] * y + r[8] * z;
  }
}

TITAN_TARGET_AVX2_FMA
void rotate_vectors_avx2_fma(BondVectorsSoA &vectors,
                             const std::array<double, 9> &r) {
  const __m256d r0 = _mm256_set1_pd(r[0]);
  const __m256d r1 = _mm256_set1_pd(r[1]);
  const __m256d r2 = _mm256_set1_pd(r[2]);
  const __m256d r3 = _mm256_set1_pd(r[3]);
  const __m256d r4 = _mm256_set1_pd(r[4]);
  const __m256d r5 = _mm256_set1_pd(r[5]);
  const __m256d r6 = _mm256_set1_pd(r[6]);
  const __m256d r7 = _mm256_set1_pd(r[7]);
  const __m256d r8 = _mm256_set1_pd(r[8]);
  std::size_t i = 0;
  for (; i + 4 <= vectors.size(); i += 4) {
    const __m256d x = _mm256_loadu_pd(vectors.x.data() + i);
    const __m256d y = _mm256_loadu_pd(vectors.y.data() + i);
    const __m256d z = _mm256_loadu_pd(vectors.z.data() + i);
    _mm256_storeu_pd(vectors.x.data() + i,
                     _mm256_fmadd_pd(r2, z,
                                     _mm256_fmadd_pd(r1, y,
                                                     _mm256_mul_pd(r0, x))));
    _mm256_storeu_pd(vectors.y.data() + i,
                     _mm256_fmadd_pd(r5, z,
                                     _mm256_fmadd_pd(r4, y,
                                                     _mm256_mul_pd(r3, x))));
    _mm256_storeu_pd(vectors.z.data() + i,
                     _mm256_fmadd_pd(r8, z,
                                     _mm256_fmadd_pd(r7, y,
                                                     _mm256_mul_pd(r6, x))));
  }
  for (; i < vectors.size(); ++i) {
    const double x = vectors.x[i];
    const double y = vectors.y[i];
    const double z = vectors.z[i];
    vectors.x[i] = r[0] * x + r[1] * y + r[2] * z;
    vectors.y[i] = r[3] * x + r[4] * y + r[5] * z;
    vectors.z[i] = r[6] * x + r[7] * y + r[8] * z;
  }
}
#endif

} // namespace

void BondVectorsSoA::clear() {
  x.clear();
  y.clear();
  z.clear();
  valid.clear();
}

void BondVectorsSoA::reserve(std::size_t n) {
  x.reserve(n);
  y.reserve(n);
  z.reserve(n);
  valid.reserve(n);
}

void BondVectorsSoA::push(double vx, double vy, double vz, bool is_valid) {
  x.push_back(vx);
  y.push_back(vy);
  z.push_back(vz);
  valid.push_back(is_valid ? 1 : 0);
}

void build_bond_vectors_scalar(std::span<const double> from_x,
                               std::span<const double> from_y,
                               std::span<const double> from_z,
                               std::span<const double> to_x,
                               std::span<const double> to_y,
                               std::span<const double> to_z,
                               const box *frame_box, BondVectorsSoA &out) {
  const std::size_t count =
      validate_coordinate_spans(from_x, from_y, from_z, to_x, to_y, to_z);
  resize_vectors(out, count);
  const std::optional<BoxMatrix> matrix = valid_matrix(frame_box);
  build_vectors_scalar_range(from_x, from_y, from_z, to_x, to_y, to_z,
                             matrix ? &*matrix : nullptr, 0, count, out);
}

void build_bond_vectors_optimized(std::span<const double> from_x,
                                  std::span<const double> from_y,
                                  std::span<const double> from_z,
                                  std::span<const double> to_x,
                                  std::span<const double> to_y,
                                  std::span<const double> to_z,
                                  const box *frame_box, BondVectorsSoA &out) {
  const std::size_t count =
      validate_coordinate_spans(from_x, from_y, from_z, to_x, to_y, to_z);
  resize_vectors(out, count);
  const std::optional<BoxMatrix> matrix = valid_matrix(frame_box);
  const BoxMatrix *matrix_ptr = matrix ? &*matrix : nullptr;
  using Kernel = void (*)(std::span<const double>, std::span<const double>,
                          std::span<const double>, std::span<const double>,
                          std::span<const double>, std::span<const double>,
                          const BoxMatrix *, BondVectorsSoA &);
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    if (titan_simd_dispatch::runtime_has_avx2_fma())
      return &build_vectors_avx2_fma;
    if (titan_simd_dispatch::runtime_has_avx2())
      return &build_vectors_avx2;
#endif
    return [](std::span<const double> fx, std::span<const double> fy,
              std::span<const double> fz, std::span<const double> tx,
              std::span<const double> ty, std::span<const double> tz,
              const BoxMatrix *box_matrix, BondVectorsSoA &vectors) {
      build_vectors_scalar_range(fx, fy, fz, tx, ty, tz, box_matrix, 0,
                                 vectors.size(), vectors);
    };
  }();
  kernel(from_x, from_y, from_z, to_x, to_y, to_z, matrix_ptr, out);
}

void build_bond_vectors(std::span<const double> from_x,
                        std::span<const double> from_y,
                        std::span<const double> from_z,
                        std::span<const double> to_x,
                        std::span<const double> to_y,
                        std::span<const double> to_z, const box *frame_box,
                        BondVectorsSoA &out) {
  build_bond_vectors_optimized(from_x, from_y, from_z, to_x, to_y, to_z,
                               frame_box, out);
}

void rotate_bond_vectors_inplace_scalar(
    BondVectorsSoA &vectors, const std::array<double, 9> &rotation) {
  validate_vectors(vectors);
  for (std::size_t i = 0; i < vectors.size(); ++i) {
    const double x = vectors.x[i];
    const double y = vectors.y[i];
    const double z = vectors.z[i];
    vectors.x[i] = rotation[0] * x + rotation[1] * y + rotation[2] * z;
    vectors.y[i] = rotation[3] * x + rotation[4] * y + rotation[5] * z;
    vectors.z[i] = rotation[6] * x + rotation[7] * y + rotation[8] * z;
  }
}

void rotate_bond_vectors_inplace_optimized(
    BondVectorsSoA &vectors, const std::array<double, 9> &rotation) {
  validate_vectors(vectors);
  using Kernel = void (*)(BondVectorsSoA &, const std::array<double, 9> &);
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    if (titan_simd_dispatch::runtime_has_avx2_fma())
      return &rotate_vectors_avx2_fma;
    if (titan_simd_dispatch::runtime_has_avx2())
      return &rotate_vectors_avx2;
#endif
    return [](BondVectorsSoA &values, const std::array<double, 9> &r) {
      for (std::size_t i = 0; i < values.size(); ++i) {
        const double x = values.x[i];
        const double y = values.y[i];
        const double z = values.z[i];
        values.x[i] = r[0] * x + r[1] * y + r[2] * z;
        values.y[i] = r[3] * x + r[4] * y + r[5] * z;
        values.z[i] = r[6] * x + r[7] * y + r[8] * z;
      }
    };
  }();
  kernel(vectors, rotation);
}

void rotate_bond_vectors_inplace(
    BondVectorsSoA &vectors, const std::array<double, 9> &rotation) {
  rotate_bond_vectors_inplace_optimized(vectors, rotation);
}

CorrelationRow compute_correlation_lag_scalar(const BondVectorsSoA &vectors,
                                              std::size_t lag) {
  validate_vectors(vectors);
  validate_lag(vectors, lag);
  return make_row(lag, reduce_scalar(vectors, lag));
}

CorrelationRow compute_correlation_lag_optimized(const BondVectorsSoA &vectors,
                                                 std::size_t lag) {
  validate_vectors(vectors);
  validate_lag(vectors, lag);
  return make_row(lag, reduce_optimized(vectors, lag));
}

std::vector<CorrelationRow>
compute_correlation_scalar(const BondVectorsSoA &vectors,
                           std::span<const double> time_ps,
                           std::size_t maxlag) {
  validate_vectors(vectors);
  validate_time_axis(vectors, time_ps);
  if (maxlag == 0)
    return {};
  if (vectors.size() < 2)
    throw std::out_of_range("S2: positive maxlag requires at least two vectors.");
  maxlag = std::min(maxlag, vectors.size() - 1);
  std::vector<CorrelationRow> rows(maxlag);
  for (std::size_t lag0 = 0; lag0 < maxlag; ++lag0) {
    rows[lag0] = compute_correlation_lag_scalar(vectors, lag0 + 1);
    set_lag_time(rows[lag0], vectors, time_ps);
  }
  return rows;
}

std::vector<CorrelationRow>
compute_correlation(const BondVectorsSoA &vectors,
                    std::span<const double> time_ps, std::size_t maxlag,
                    TitanExecutor &executor) {
  validate_vectors(vectors);
  validate_time_axis(vectors, time_ps);
  if (maxlag == 0)
    return {};
  if (vectors.size() < 2)
    throw std::out_of_range("S2: positive maxlag requires at least two vectors.");
  maxlag = std::min(maxlag, vectors.size() - 1);
  std::vector<CorrelationRow> rows(maxlag);
  executor.parallel_for(maxlag, std::max<std::size_t>(1, executor.capacity()),
                        1, [&](std::size_t, std::size_t begin,
                               std::size_t end) {
    for (std::size_t work = begin; work < end; ++work) {
      const std::size_t lag0 =
          work % 2 == 0 ? work / 2 : maxlag - 1 - work / 2;
      rows[lag0] = compute_correlation_lag_optimized(vectors, lag0 + 1);
      set_lag_time(rows[lag0], vectors, time_ps);
    }
  });
  return rows;
}

FitResult fit_single_exp_plateau(std::span<const CorrelationRow> rows) {
  const std::vector<CorrelationRow> usable = usable_rows(rows);
  if (usable.size() < 3 || !valid_fit_rows(usable))
    return {};
  if (usable.size() < 5)
    return tail_fallback(usable);
  double min_time = std::numeric_limits<double>::infinity();
  double max_time = 0.0;
  double min_correlation = std::numeric_limits<double>::infinity();
  double max_correlation = -std::numeric_limits<double>::infinity();
  for (const CorrelationRow &row : usable) {
    min_time = std::min(min_time, row.lag_time_ps);
    max_time = std::max(max_time, row.lag_time_ps);
    min_correlation = std::min(min_correlation, row.correlation);
    max_correlation = std::max(max_correlation, row.correlation);
  }
  if (!(max_correlation - min_correlation > kEpsilon))
    return tail_fallback(usable);

  const double log_min = std::log(min_time * 1.0e-3);
  const double log_max = std::log(max_time * 1.0e3);
  std::array<double, kLogScanPoints> logs{};
  FitCandidate best;
  std::size_t best_index = 0;
  for (std::size_t i = 0; i < logs.size(); ++i) {
    logs[i] = log_min + (log_max - log_min) * static_cast<double>(i) /
                            static_cast<double>(logs.size() - 1);
    const FitCandidate candidate = evaluate_tau(usable, std::exp(logs[i]));
    if (candidate.ok && candidate.residual < best.residual) {
      best = candidate;
      best_index = i;
    }
  }
  if (!best.ok)
    return tail_fallback(usable);

  double left = logs[best_index == 0 ? 0 : best_index - 1];
  double right =
      logs[best_index + 1 < logs.size() ? best_index + 1 : logs.size() - 1];
  constexpr double ratio = 0.6180339887498948482;
  double x1 = right - ratio * (right - left);
  double x2 = left + ratio * (right - left);
  FitCandidate f1 = evaluate_tau(usable, std::exp(x1));
  FitCandidate f2 = evaluate_tau(usable, std::exp(x2));
  for (int iteration = 0; iteration < 96; ++iteration) {
    if (f1.residual > f2.residual) {
      left = x1;
      x1 = x2;
      f1 = f2;
      x2 = left + ratio * (right - left);
      f2 = evaluate_tau(usable, std::exp(x2));
    } else {
      right = x2;
      x2 = x1;
      f2 = f1;
      x1 = right - ratio * (right - left);
      f1 = evaluate_tau(usable, std::exp(x1));
    }
  }
  if (f1.ok && f1.residual < best.residual)
    best = f1;
  if (f2.ok && f2.residual < best.residual)
    best = f2;
  if (!best.ok || !(best.tau_ps > 0.0))
    return tail_fallback(usable);

  FitResult fit;
  fit.ok = true;
  fit.raw_plateau = best.raw_plateau;
  fit.s2 = std::clamp(best.raw_plateau, 0.0, 1.0);
  fit.tau_ps = best.tau_ps;
  fit.weighted_residual = best.residual;
  return fit;
}

} // namespace titan_s2_kernel
