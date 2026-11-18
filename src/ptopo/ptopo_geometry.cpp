#include "ptopo_geometry.h"

#include "../titan_executor.h"
#include "../titan_simd_dispatch.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace titan_ptopo {
namespace {

using L3 = std::array<long double, 3>;
using M3 = std::array<L3, 3>;

struct LongMetrics {
  M3 B{};
  M3 inverse{};
  long double determinant = 0;
  long double norm_B = 0;
  long double norm_inverse = 0;
};

long double norm3(const L3 &x) { return std::hypot(x[0], x[1], x[2]); }

Result<void> check_policy(const GeometryPolicy &p) {
  if (!(p.absolute_tolerance > 0) || !std::isfinite(p.absolute_tolerance) ||
      !(p.relative_tolerance >= 0) || !std::isfinite(p.relative_tolerance) ||
      !(p.maximum_condition_bound >= 3) ||
      !std::isfinite(p.maximum_condition_bound) || p.maximum_candidates == 0)
    return make_error(ErrorCode::InvalidInput, "invalid numerical policy");
  return {};
}

Result<LongMetrics> long_metrics(const Cell &c, const GeometryPolicy &p) {
  if (auto ok = check_policy(p); !ok)
    return std::unexpected(ok.error());
  LongMetrics m;
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      if (!std::isfinite(c.B[i][j]))
        return make_error(ErrorCode::InvalidCell, "nonfinite cell entry");
      m.B[i][j] = c.B[i][j];
      m.norm_B = std::hypot(m.norm_B, m.B[i][j]);
    }
  const M3 &a = m.B;
  m.determinant = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                  a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                  a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  if (m.determinant == 0 || !std::isfinite(m.determinant))
    return make_error(ErrorCode::InvalidCell,
                      "singular or nonfinite cell determinant");
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      std::array<std::size_t, 2> r{}, q{};
      std::size_t nr = 0, nq = 0;
      for (std::size_t k = 0; k < 3; ++k) {
        if (k != j)
          r[nr++] = k;
        if (k != i)
          q[nq++] = k;
      }
      const long double x =
          a[r[0]][q[0]] * a[r[1]][q[1]] - a[r[0]][q[1]] * a[r[1]][q[0]];
      m.inverse[i][j] = ((i + j) % 2 == 0 ? x : -x) / m.determinant;
      m.norm_inverse = std::hypot(m.norm_inverse, m.inverse[i][j]);
    }
  if (!std::isfinite(m.norm_B * m.norm_inverse) ||
      m.norm_B * m.norm_inverse > p.maximum_condition_bound)
    return make_error(ErrorCode::InvalidCell,
                      "cell conditioning exceeds declared geometry policy");
  return m;
}

long double tolerance(const LongMetrics &m, long double radius,
                      const GeometryPolicy &p) {
  return p.absolute_tolerance +
         p.relative_tolerance * std::max(radius, m.norm_B);
}

// Every integer image a with |B (df + a)| <= R. Finite box from
// |df_i + a_i| <= ||row_i(B^-1)|| R, padded by one integer per side against
// endpoint rounding (not interval arithmetic).
Result<std::vector<Image>> enumerate(const LongMetrics &m, const Vec3 &df,
                                     long double R, const GeometryPolicy &p) {
  if (!std::isfinite(R) || R < 0)
    return make_error(ErrorCode::InvalidInput, "invalid search radius");
  std::array<std::int64_t, 3> lo{}, hi{};
  std::size_t count = 1;
  for (std::size_t i = 0; i < 3; ++i) {
    const long double bound = norm3(m.inverse[i]) * R;
    const long double l =
        std::floor(-static_cast<long double>(df[i]) - bound) - 1;
    const long double h =
        std::ceil(-static_cast<long double>(df[i]) + bound) + 1;
    constexpr long double safe =
        static_cast<long double>(std::numeric_limits<std::int64_t>::max()) / 2;
    if (!std::isfinite(df[i]) || !std::isfinite(l) || !std::isfinite(h) ||
        l < -safe || h > safe)
      return make_error(ErrorCode::ResourceLimit,
                        "image bounds exceed safe integer range");
    lo[i] = static_cast<std::int64_t>(l);
    hi[i] = static_cast<std::int64_t>(h);
    const auto width = static_cast<std::uint64_t>(hi[i] - lo[i] + 1);
    if (width > p.maximum_candidates / count)
      return make_error(ErrorCode::ResourceLimit,
                        "image-enumeration budget exceeded; no truncated MIC "
                        "accepted");
    count *= static_cast<std::size_t>(width);
  }
  std::vector<Image> out;
  for (auto i = lo[0]; i <= hi[0]; ++i)
    for (auto j = lo[1]; j <= hi[1]; ++j)
      for (auto k = lo[2]; k <= hi[2]; ++k) {
        const L3 f{static_cast<long double>(df[0]) + i,
                   static_cast<long double>(df[1]) + j,
                   static_cast<long double>(df[2]) + k};
        L3 x{};
        for (std::size_t a = 0; a < 3; ++a)
          for (std::size_t b = 0; b < 3; ++b)
            x[a] += m.B[a][b] * f[b];
        const long double d = norm3(x);
        if (!std::isfinite(d) || d > std::numeric_limits<double>::max())
          return make_error(ErrorCode::NumericalFailure,
                            "nonfinite image distance");
        if (d <= R)
          out.push_back({{i, j, k},
                         {static_cast<double>(x[0]), static_cast<double>(x[1]),
                          static_cast<double>(x[2])},
                         static_cast<double>(d)});
      }
  std::sort(out.begin(), out.end(), [](const Image &a, const Image &b) {
    return a.distance != b.distance ? a.distance < b.distance
                                    : a.shift < b.shift;
  });
  return out;
}

Int floor_div(Int a, Int b) noexcept { // b > 0
  Int q = a / b;
  if (a % b < 0)
    --q;
  return q;
}

// ---------------------------------------------------------------------------
// Distance kernel. Cluster points are y = x_q + t (one image of a neighbour
// bin, added once in scalar code) and the test point is x_i:
//   d = y - x_i per axis,  d2 = fma(dz, dz, fma(dy, dy, dx * dx)).
// The AVX2 lanes perform exactly these operations, and std::fma is correctly
// rounded (hardware FMA in both TITAN profiles), so both paths give the same
// bits and the same contact set. No contraction can merge `dx * dx` into the
// explicit fma, and no expression here has a mul feeding an add.
// ---------------------------------------------------------------------------
inline double pair_d2(double yx, double yy, double yz, double xi, double yi,
                      double zi) noexcept {
  const double dx = yx - xi;
  const double dy = yy - yi;
  const double dz = yz - zi;
  return std::fma(dz, dz, std::fma(dy, dy, dx * dx));
}

std::size_t cluster_hits_scalar(const double *yx, const double *yy,
                                const double *yz, std::size_t count, double xi,
                                double yi, double zi, double limit2,
                                std::uint32_t *hit, double *hit_d2) noexcept {
  std::size_t found = 0;
  for (std::size_t q = 0; q < count; ++q) {
    const double d2 = pair_d2(yx[q], yy[q], yz[q], xi, yi, zi);
    if (d2 <= limit2) {
      hit[found] = static_cast<std::uint32_t>(q);
      hit_d2[found] = d2;
      ++found;
    }
  }
  return found;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2_FMA
std::size_t cluster_hits_avx2(const double *yx, const double *yy,
                              const double *yz, std::size_t count, double xi,
                              double yi, double zi, double limit2,
                              std::uint32_t *hit, double *hit_d2) noexcept {
  const __m256d xv = _mm256_set1_pd(xi);
  const __m256d yv = _mm256_set1_pd(yi);
  const __m256d zv = _mm256_set1_pd(zi);
  const __m256d limit = _mm256_set1_pd(limit2);
  std::size_t found = 0;
  std::size_t q = 0;
  for (; q + 4 <= count; q += 4) {
    const __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(yx + q), xv);
    const __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(yy + q), yv);
    const __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(yz + q), zv);
    const __m256d d2 = _mm256_fmadd_pd(
        dz, dz, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx)));
    // Ordered compare: a NaN lane is never a hit, as in the scalar test.
    unsigned mask = static_cast<unsigned>(
        _mm256_movemask_pd(_mm256_cmp_pd(d2, limit, _CMP_LE_OQ)));
    if (mask == 0)
      continue;
    alignas(32) double lane[4];
    _mm256_store_pd(lane, d2);
    while (mask != 0) {
      const unsigned b = static_cast<unsigned>(std::countr_zero(mask));
      hit[found] = static_cast<std::uint32_t>(q + b);
      hit_d2[found] = lane[b];
      ++found;
      mask &= mask - 1;
    }
  }
  for (; q < count; ++q) {
    const double dx = yx[q] - xi;
    const double dy = yy[q] - yi;
    const double dz = yz[q] - zi;
    const double d2 = std::fma(dz, dz, std::fma(dy, dy, dx * dx));
    if (d2 <= limit2) {
      hit[found] = static_cast<std::uint32_t>(q);
      hit_d2[found] = d2;
      ++found;
    }
  }
  return found;
}
#endif

bool contact_less(const PeriodicContact &a, const PeriodicContact &b) noexcept {
  if (a.i != b.i)
    return a.i < b.i;
  if (a.j != b.j)
    return a.j < b.j;
  return a.shift < b.shift;
}

struct WorkerResult {
  std::vector<PeriodicContact> contacts;
  std::size_t evaluations = 0;
  std::size_t ambiguous = 0;
  bool has_first_ambiguous = false;
  PeriodicContact first_ambiguous;
  bool overflow = false;
  bool budget_exceeded = false;
};

} // namespace

Cell cell_from_box_rows(const std::array<Vec3, 3> &rows) noexcept {
  Cell c;
  for (std::size_t r = 0; r < 3; ++r)
    for (std::size_t k = 0; k < 3; ++k)
      c.B[r][k] = rows[k][r];
  return c;
}

Result<CellMetrics> analyze_cell(const Cell &cell, const GeometryPolicy &p) {
  auto lm = long_metrics(cell, p);
  if (!lm)
    return std::unexpected(lm.error());
  CellMetrics m;
  m.cell = cell;
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      m.inverse[i][j] = static_cast<double>(lm->inverse[i][j]);
  m.determinant = static_cast<double>(lm->determinant);
  m.norm_B = static_cast<double>(lm->norm_B);
  m.norm_inverse = static_cast<double>(lm->norm_inverse);
  long double widest_row = 0;
  for (std::size_t k = 0; k < 3; ++k) {
    const long double row = norm3(lm->inverse[k]);
    m.inverse_row_norm[k] = static_cast<double>(row);
    widest_row = std::max(widest_row, row);
  }
  m.minimum_width = static_cast<double>(1.0L / widest_row);
  if (!std::isfinite(m.determinant) || m.determinant == 0.0 ||
      !std::isfinite(m.minimum_width))
    return make_error(ErrorCode::NumericalFailure,
                      "cell metrics outside double range");
  return m;
}

Result<double> cell_volume(const Cell &c) {
  auto m = long_metrics(c, GeometryPolicy{});
  if (!m)
    return std::unexpected(m.error());
  const long double v = std::fabs(m->determinant);
  if (v < std::numeric_limits<double>::min() ||
      v > std::numeric_limits<double>::max())
    return make_error(ErrorCode::NumericalFailure,
                      "cell volume outside double range");
  return static_cast<double>(v);
}

Vec3 cartesian(const Cell &c, const Vec3 &f) noexcept {
  Vec3 x{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      x[i] += c.B[i][j] * f[j];
  return x;
}

Vec3 to_fractional(const CellMetrics &m, const Vec3 &x) noexcept {
  Vec3 s{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      s[i] += m.inverse[i][j] * x[j];
  return s;
}

Result<NearestImages> nearest_images(const Cell &c, const Vec3 &d,
                                     const GeometryPolicy &p) {
  auto m = long_metrics(c, p);
  if (!m)
    return std::unexpected(m.error());
  L3 trial{};
  for (std::size_t i = 0; i < 3; ++i) {
    if (!std::isfinite(d[i]))
      return make_error(ErrorCode::InvalidInput, "nonfinite coordinate");
    trial[i] = static_cast<long double>(d[i]) -
               std::round(static_cast<long double>(d[i]));
  }
  L3 x{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      x[i] += m->B[i][j] * trial[j];
  // The rounded image is a feasible upper bound R0 on the minimum distance.
  const long double R = norm3(x);
  const long double tol = tolerance(*m, R, p);
  auto all = enumerate(*m, d, R + 4 * tol, p);
  if (!all)
    return std::unexpected(all.error());
  if (all->empty())
    return make_error(ErrorCode::NumericalFailure,
                      "initial feasible image lost numerically");
  NearestImages out;
  out.separation_to_next = std::numeric_limits<double>::infinity();
  const double best = all->front().distance;
  for (const Image &image : *all) {
    if (image.distance - best <= tol) {
      out.minimizers.push_back(image);
    } else {
      out.separation_to_next = image.distance - best;
      break;
    }
  }
  out.unique_under_tolerance = out.minimizers.size() == 1;
  return out;
}

Result<std::vector<Image>> images_within(const Cell &c, const Vec3 &d, double r,
                                         const GeometryPolicy &p) {
  if (!std::isfinite(r) || r < 0)
    return make_error(ErrorCode::InvalidInput, "invalid cutoff radius");
  auto m = long_metrics(c, p);
  if (!m)
    return std::unexpected(m.error());
  const long double tol = tolerance(*m, r, p);
  auto all = enumerate(*m, d, static_cast<long double>(r) + 2 * tol, p);
  if (!all)
    return std::unexpected(all.error());
  std::vector<Image> out;
  for (const Image &image : *all) {
    if (std::fabs(static_cast<long double>(image.distance) - r) <= tol)
      return make_error(ErrorCode::AmbiguousGeometry,
                        "contact lies within numerical cutoff tolerance");
    if (image.distance < r)
      out.push_back(image);
  }
  return out;
}

Result<Z3> unique_nearest_shift(const CellMetrics &m, const Vec3 &delta,
                                const GeometryPolicy &p) {
  constexpr double kMaxFractional = 4503599627370496.0; // 2^52
  Vec3 r{};
  Z3 shift{};
  for (std::size_t k = 0; k < 3; ++k) {
    if (!std::isfinite(delta[k]) || std::fabs(delta[k]) >= kMaxFractional)
      return make_error(ErrorCode::InvalidInput,
                        "nonfinite or out-of-range fractional delta");
    const double rounded = std::round(delta[k]);
    r[k] = delta[k] - rounded;
    shift[k] = -static_cast<Int>(rounded);
  }
  const Vec3 x = cartesian(m.cell, r);
  const double d = std::hypot(x[0], x[1], x[2]);
  const double tol =
      p.absolute_tolerance + p.relative_tolerance * std::max(d, m.norm_B);
  if (2.0 * d + tol < m.minimum_width)
    return shift;
  auto nearest = nearest_images(m.cell, delta, p);
  if (!nearest)
    return std::unexpected(nearest.error());
  if (!nearest->unique_under_tolerance)
    return make_error(ErrorCode::AmbiguousGeometry,
                      "nearest image is tied within numerical tolerance");
  const auto &a = nearest->minimizers.front().shift;
  return Z3{a[0], a[1], a[2]};
}

Result<std::vector<PeriodicContact>>
find_periodic_contacts(const Cell &cell, std::span<const Vec3> s,
                       const ContactSearchOptions &opt, TitanExecutor *executor,
                       ContactSearchStats *stats) {
  auto metrics = analyze_cell(cell, opt.policy);
  if (!metrics)
    return std::unexpected(metrics.error());
  const CellMetrics &m = *metrics;
  if (!(opt.cutoff > 0.0) || !std::isfinite(opt.cutoff))
    return make_error(ErrorCode::InvalidInput,
                      "contact cutoff must be positive and finite");
  const std::size_t n = s.size();
  if (n >= std::numeric_limits<std::uint32_t>::max())
    return make_error(ErrorCode::ResourceLimit,
                      "point count exceeds 32-bit indexing");
  const double tol = opt.policy.absolute_tolerance +
                     opt.policy.relative_tolerance *
                         std::max(opt.cutoff, m.norm_B);
  if (!(tol < opt.cutoff))
    return make_error(ErrorCode::InvalidInput,
                      "tolerance band swallows the contact cutoff");
  const double lo = opt.cutoff - tol;
  const double hi = opt.cutoff + tol;
  const double lo2 = lo * lo;
  const double hi2 = hi * hi;

  // 1. Wrap into [0,1)^3: s = r + w with integer w, recorded so shifts can be
  //    reported against the caller's representation.
  constexpr double kMaxFractional = 4503599627370496.0; // 2^52
  std::vector<Vec3> wrapped(n);
  std::vector<Z3> wrap(n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < 3; ++k) {
      const double v = s[i][k];
      if (!std::isfinite(v) || std::fabs(v) >= kMaxFractional)
        return make_error(ErrorCode::InvalidInput,
                          "nonfinite or out-of-range fractional coordinate");
      const double f = std::floor(v);
      double r = v - f;
      Int w = static_cast<Int>(f);
      if (r >= 1.0) { // v just below an integer rounded up
        r -= 1.0;
        ++w;
      }
      wrapped[i][k] = r;
      wrap[i][k] = w;
    }

  // 2. Bins. A pair within `hi` has |Delta s_k| <= h_k = ||row_k(B^-1)|| hi.
  //    With n_k bins, bin offsets |D_k| <= reach_k = floor((h_k + 2 eps) n_k)
  //    + 1 cover every such pair, eps bounding the bin-assignment rounding.
  //    h_k carries a relative and absolute safety margin for the rounding of
  //    B^-1. Several offsets may land in one bin with different images.
  constexpr double kEps = 1e-9;
  constexpr std::size_t kMaxAxisBins = std::size_t{1} << 20;
  std::array<std::size_t, 3> nbin{}, reach{};
  std::array<double, 3> half{};
  const auto reach_for = [&](std::size_t k) {
    return static_cast<std::size_t>(
               std::floor((half[k] + 2 * kEps) *
                          static_cast<double>(nbin[k]))) +
           1;
  };
  for (std::size_t k = 0; k < 3; ++k) {
    half[k] = m.inverse_row_norm[k] * hi * (1.0 + 1e-6) + kEps;
    const double fit = std::floor(1.0 / (half[k] + 2 * kEps));
    nbin[k] = fit < 1.0 ? 1
                        : static_cast<std::size_t>(
                              std::min(fit, static_cast<double>(kMaxAxisBins)));
  }
  const std::size_t bin_budget = std::max<std::size_t>(
      1, std::min(opt.maximum_bins, std::max<std::size_t>(n, 1) * 8));
  while (nbin[0] * nbin[1] * nbin[2] > bin_budget) {
    const std::size_t k = static_cast<std::size_t>(
        std::max_element(nbin.begin(), nbin.end()) - nbin.begin());
    nbin[k] = std::max<std::size_t>(1, nbin[k] / 2);
  }
  for (std::size_t k = 0; k < 3; ++k)
    reach[k] = reach_for(k);
  const std::size_t total_bins = nbin[0] * nbin[1] * nbin[2];

  // 3. Counting sort by bin; SoA Cartesian coordinates of the wrapped points.
  std::vector<std::uint32_t> point_bin(n);
  std::vector<std::uint32_t> start(total_bins + 1, 0);
  for (std::size_t i = 0; i < n; ++i) {
    std::array<std::size_t, 3> b{};
    for (std::size_t k = 0; k < 3; ++k)
      b[k] = std::min(nbin[k] - 1,
                      static_cast<std::size_t>(wrapped[i][k] *
                                               static_cast<double>(nbin[k])));
    point_bin[i] =
        static_cast<std::uint32_t>((b[0] * nbin[1] + b[1]) * nbin[2] + b[2]);
    ++start[point_bin[i] + 1];
  }
  for (std::size_t b = 1; b <= total_bins; ++b)
    start[b] += start[b - 1];
  std::vector<std::uint32_t> cursor(start.begin(), start.end() - 1);
  std::vector<double> sx(n), sy(n), sz(n);
  std::vector<std::uint32_t> sid(n), sbin(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    const std::uint32_t slot = cursor[point_bin[i]]++;
    const Vec3 x = cartesian(cell, wrapped[i]);
    sx[slot] = x[0];
    sy[slot] = x[1];
    sz[slot] = x[2];
    sid[slot] = i;
    sbin[slot] = point_bin[i];
  }

  // 4. Lattice translations t = B m for every image a neighbour offset can
  //    reach, computed once and shared by both kernels.
  std::array<Int, 3> mmax{};
  std::array<std::size_t, 3> span{};
  for (std::size_t k = 0; k < 3; ++k) {
    mmax[k] = static_cast<Int>(reach[k] / nbin[k]) + 2;
    span[k] = static_cast<std::size_t>(2 * mmax[k] + 1);
  }
  std::vector<Vec3> translation(span[0] * span[1] * span[2]);
  const auto translation_index = [&](const Z3 &mm) {
    return (static_cast<std::size_t>(mm[0] + mmax[0]) * span[1] +
            static_cast<std::size_t>(mm[1] + mmax[1])) *
               span[2] +
           static_cast<std::size_t>(mm[2] + mmax[2]);
  };
  for (Int a = -mmax[0]; a <= mmax[0]; ++a)
    for (Int b = -mmax[1]; b <= mmax[1]; ++b)
      for (Int c = -mmax[2]; c <= mmax[2]; ++c)
        translation[translation_index({a, b, c})] = cartesian(
            cell, {static_cast<double>(a), static_cast<double>(b),
                   static_cast<double>(c)});
  std::vector<std::array<Int, 3>> offsets;
  for (Int a = -static_cast<Int>(reach[0]); a <= static_cast<Int>(reach[0]); ++a)
    for (Int b = -static_cast<Int>(reach[1]); b <= static_cast<Int>(reach[1]);
         ++b)
      for (Int c = -static_cast<Int>(reach[2]);
           c <= static_cast<Int>(reach[2]); ++c)
        offsets.push_back({a, b, c});

  bool use_avx2 = false;
#if TITAN_SIMD_X86
  use_avx2 = opt.allow_avx2 && titan_simd_dispatch::runtime_has_avx2_fma();
#endif

  // 5. Search. Items are points in bin order; a worker builds the cluster
  //    of a home bin once (every point of every reachable bin, translated by
  //    its image) and tests each of its home points against it. Each pair is
  //    kept only from its i < j side, a self-image pair only for the
  //    lexicographically positive image, so no decision is taken twice.
  const std::size_t worker_slots =
      executor != nullptr ? std::max<std::size_t>(1, executor->capacity()) : 1;
  std::vector<WorkerResult> results(worker_slots);
  const auto body = [&](std::size_t worker, std::size_t begin,
                        std::size_t end) {
    WorkerResult &out = results[worker];
    std::vector<double> cx, cy, cz, hit_d2;
    std::vector<std::uint32_t> cj, hit;
    std::vector<Z3> cm;
    std::size_t p = begin;
    while (p < end) {
      const std::uint32_t bin = sbin[p];
      const std::size_t home_end = std::min<std::size_t>(end, start[bin + 1]);
      const std::array<Int, 3> b{
          static_cast<Int>(bin / (nbin[1] * nbin[2])),
          static_cast<Int>((bin / nbin[2]) % nbin[1]),
          static_cast<Int>(bin % nbin[2])};
      cx.clear();
      cy.clear();
      cz.clear();
      cj.clear();
      cm.clear();
      for (const auto &D : offsets) {
        Z3 image{};
        std::array<std::size_t, 3> w{};
        for (std::size_t k = 0; k < 3; ++k) {
          const Int virtual_bin = b[k] + D[k];
          const Int nk = static_cast<Int>(nbin[k]);
          image[k] = floor_div(virtual_bin, nk);
          w[k] = static_cast<std::size_t>(virtual_bin - image[k] * nk);
        }
        const std::size_t neighbour = (w[0] * nbin[1] + w[1]) * nbin[2] + w[2];
        const std::uint32_t q0 = start[neighbour], q1 = start[neighbour + 1];
        if (q0 == q1)
          continue;
        const Vec3 &t = translation[translation_index(image)];
        for (std::uint32_t q = q0; q < q1; ++q) {
          cx.push_back(sx[q] + t[0]);
          cy.push_back(sy[q] + t[1]);
          cz.push_back(sz[q] + t[2]);
          cj.push_back(sid[q]);
          cm.push_back(image);
        }
      }
      const std::size_t cluster = cx.size();
      hit.resize(cluster);
      hit_d2.resize(cluster);
      for (std::size_t h = p; h < home_end; ++h) {
        const std::uint32_t i = sid[h];
        std::size_t found = 0;
#if TITAN_SIMD_X86
        if (use_avx2)
          found = cluster_hits_avx2(cx.data(), cy.data(), cz.data(), cluster,
                                    sx[h], sy[h], sz[h], hi2, hit.data(),
                                    hit_d2.data());
        else
#endif
          found = cluster_hits_scalar(cx.data(), cy.data(), cz.data(), cluster,
                                      sx[h], sy[h], sz[h], hi2, hit.data(),
                                      hit_d2.data());
        out.evaluations += cluster;
        for (std::size_t k = 0; k < found; ++k) {
          const std::uint32_t q = hit[k];
          const std::uint32_t j = cj[q];
          if (j < i || (j == i && lexicographic_sign(cm[q]) <= 0))
            continue;
          PeriodicContact contact;
          contact.i = i;
          contact.j = j;
          Z3 wd{};
          if (!checked::sub3(wrap[i], wrap[j], wd) ||
              !checked::add3(cm[q], wd, contact.shift)) {
            out.overflow = true;
            return;
          }
          contact.distance = std::sqrt(hit_d2[k]);
          if (hit_d2[k] < lo2) {
            out.contacts.push_back(contact);
            if (out.contacts.size() > opt.maximum_contacts) {
              out.budget_exceeded = true;
              return;
            }
          } else {
            ++out.ambiguous;
            if (!out.has_first_ambiguous ||
                contact_less(contact, out.first_ambiguous)) {
              out.first_ambiguous = contact;
              out.has_first_ambiguous = true;
            }
          }
        }
      }
      p = home_end;
    }
  };
  if (executor != nullptr && opt.workers > 1 && n > 0)
    executor->parallel_for(n, opt.workers, std::max<std::size_t>(1, opt.min_grain),
                           body);
  else if (n > 0)
    body(0, 0, n);

  std::size_t total = 0, evaluations = 0, ambiguous = 0;
  const WorkerResult *first_ambiguous = nullptr;
  for (const WorkerResult &r : results) {
    if (r.overflow)
      return overflow_error("periodic contact shift");
    if (r.budget_exceeded)
      return make_error(ErrorCode::ResourceLimit,
                        "periodic contact budget exceeded; no partial "
                        "contact set returned");
    total += r.contacts.size();
    evaluations += r.evaluations;
    ambiguous += r.ambiguous;
    if (r.has_first_ambiguous &&
        (first_ambiguous == nullptr ||
         contact_less(r.first_ambiguous, first_ambiguous->first_ambiguous)))
      first_ambiguous = &r;
  }
  if (ambiguous > 0) {
    const PeriodicContact &c = first_ambiguous->first_ambiguous;
    return make_error(
        ErrorCode::AmbiguousGeometry,
        std::to_string(ambiguous) +
            " pair distance(s) inside the cutoff tolerance band; first: " +
            std::to_string(c.i) + "-" + std::to_string(c.j) + " shift (" +
            std::to_string(c.shift[0]) + "," + std::to_string(c.shift[1]) +
            "," + std::to_string(c.shift[2]) + ") distance " +
            std::to_string(c.distance));
  }
  if (total > opt.maximum_contacts)
    return make_error(ErrorCode::ResourceLimit,
                      "periodic contact budget exceeded; no partial contact "
                      "set returned");
  std::vector<PeriodicContact> contacts;
  contacts.reserve(total);
  for (WorkerResult &r : results)
    contacts.insert(contacts.end(), r.contacts.begin(), r.contacts.end());
  std::sort(contacts.begin(), contacts.end(), contact_less);
  if (stats != nullptr) {
    stats->bins = nbin;
    stats->reach = reach;
    stats->distance_evaluations = evaluations;
    stats->contacts = contacts.size();
    stats->used_avx2 = use_avx2;
  }
  return contacts;
}

Result<std::vector<PeriodicContact>>
find_periodic_contacts_reference(const Cell &cell, std::span<const Vec3> s,
                                 double cutoff, const GeometryPolicy &policy) {
  if (s.size() >= std::numeric_limits<std::uint32_t>::max())
    return make_error(ErrorCode::ResourceLimit,
                      "point count exceeds 32-bit indexing");
  std::vector<PeriodicContact> out;
  for (std::uint32_t i = 0; i < s.size(); ++i)
    for (std::uint32_t j = i; j < s.size(); ++j) {
      const Vec3 delta{s[j][0] - s[i][0], s[j][1] - s[i][1],
                       s[j][2] - s[i][2]};
      auto images = images_within(cell, delta, cutoff, policy);
      if (!images)
        return std::unexpected(images.error());
      for (const Image &image : *images) {
        const Z3 shift{image.shift[0], image.shift[1], image.shift[2]};
        if (i == j && lexicographic_sign(shift) <= 0)
          continue;
        out.push_back({i, j, shift, image.distance});
      }
    }
  std::sort(out.begin(), out.end(), contact_less);
  return out;
}

} // namespace titan_ptopo
