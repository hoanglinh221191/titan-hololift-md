#include "anchor_mst.h"

#include "../titan_simd_dispatch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

using namespace std;

namespace titan_pbctopo {

namespace {

// The relaxation and selection kernels of the dense Prim, moved here unchanged
// from anchor_graph.cpp.

inline void relax_anchor_best_scalar(double *best, const double *px,
                                     const double *py, const double *pz,
                                     size_t n, double sx, double sy,
                                     double sz) noexcept {
  for (size_t i = 0; i < n; ++i) {
    const double dx = sx - px[i];
    const double dy = sy - py[i];
    const double dz = sz - pz[i];
    const double d2 = dx * dx + dy * dy + dz * dz;
    best[i] = min(best[i], d2);
  }
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline void relax_anchor_best_avx2(double *best, const double *px,
                                   const double *py, const double *pz,
                                   size_t n, double sx, double sy,
                                   double sz) noexcept {
  const __m256d vsx = _mm256_set1_pd(sx);
  const __m256d vsy = _mm256_set1_pd(sy);
  const __m256d vsz = _mm256_set1_pd(sz);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const __m256d dx = _mm256_sub_pd(vsx, _mm256_loadu_pd(px + i));
    const __m256d dy = _mm256_sub_pd(vsy, _mm256_loadu_pd(py + i));
    const __m256d dz = _mm256_sub_pd(vsz, _mm256_loadu_pd(pz + i));
    const __m256d d2 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
        _mm256_mul_pd(dz, dz));
    const __m256d best_vec = _mm256_loadu_pd(best + i);
    _mm256_storeu_pd(best + i, _mm256_min_pd(best_vec, d2));
  }
  for (; i < n; ++i) {
    const double dx = sx - px[i];
    const double dy = sy - py[i];
    const double dz = sz - pz[i];
    const double d2 = dx * dx + dy * dy + dz * dz;
    best[i] = min(best[i], d2);
  }
}

TITAN_TARGET_AVX2_FMA
inline void relax_anchor_best_avx2_fma(double *best, const double *px,
                                       const double *py, const double *pz,
                                       size_t n, double sx, double sy,
                                       double sz) noexcept {
  const __m256d vsx = _mm256_set1_pd(sx);
  const __m256d vsy = _mm256_set1_pd(sy);
  const __m256d vsz = _mm256_set1_pd(sz);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const __m256d dx = _mm256_sub_pd(vsx, _mm256_loadu_pd(px + i));
    const __m256d dy = _mm256_sub_pd(vsy, _mm256_loadu_pd(py + i));
    const __m256d dz = _mm256_sub_pd(vsz, _mm256_loadu_pd(pz + i));
    const __m256d d2 =
        _mm256_fmadd_pd(dz, dz,
                        _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx)));
    const __m256d best_vec = _mm256_loadu_pd(best + i);
    _mm256_storeu_pd(best + i, _mm256_min_pd(best_vec, d2));
  }
  for (; i < n; ++i) {
    const double dx = sx - px[i];
    const double dy = sy - py[i];
    const double dz = sz - pz[i];
    const double d2 = dx * dx + dy * dy + dz * dz;
    best[i] = min(best[i], d2);
  }
}
#endif

inline void relax_anchor_best(double *best, const double *px, const double *py,
                              const double *pz, size_t n, double sx,
                              double sy, double sz) noexcept {
  using Kernel = void (*)(double *, const double *, const double *,
                          const double *, size_t, double, double,
                          double) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    if (titan_simd_dispatch::runtime_has_avx2_fma())
      return &relax_anchor_best_avx2_fma;
    if (titan_simd_dispatch::runtime_has_avx2())
      return &relax_anchor_best_avx2;
#endif
    return &relax_anchor_best_scalar;
  }();
  kernel(best, px, py, pz, n, sx, sy, sz);
}

inline size_t select_anchor_next_scalar(const double *best,
                                        const unsigned char *used, size_t n,
                                        double &next_cost) noexcept {
  size_t next = numeric_limits<size_t>::max();
  next_cost = numeric_limits<double>::infinity();
  for (size_t i = 0; i < n; ++i) {
    if (used[i] != 0)
      continue;
    if (best[i] < next_cost) {
      next = i;
      next_cost = best[i];
    }
  }
  return next;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline size_t select_anchor_next_avx2(const double *best,
                                      const unsigned char *used, size_t n,
                                      double &next_cost) noexcept {
  const double inf = numeric_limits<double>::infinity();
  size_t next = numeric_limits<size_t>::max();
  next_cost = inf;
  alignas(32) double lane_best[4];
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const __m256d best_vec = _mm256_loadu_pd(best + i);
    _mm256_store_pd(lane_best, best_vec);
    for (size_t lane = 0; lane < 4; ++lane) {
      if (used[i + lane] != 0)
        continue;
      if (lane_best[lane] < next_cost) {
        next = i + lane;
        next_cost = lane_best[lane];
      }
    }
  }
  for (; i < n; ++i) {
    if (used[i] != 0)
      continue;
    if (best[i] < next_cost) {
      next = i;
      next_cost = best[i];
    }
  }
  return next;
}
#endif

inline size_t select_anchor_next(const double *best, const unsigned char *used,
                                 size_t n, double &next_cost) noexcept {
  using Kernel =
      size_t (*)(const double *, const unsigned char *, size_t, double &) noexcept;
  static const Kernel kernel = []() noexcept -> Kernel {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    if (titan_simd_dispatch::runtime_has_avx2())
      return &select_anchor_next_avx2;
#endif
    return &select_anchor_next_scalar;
  }();
  return kernel(best, used, n, next_cost);
}

} // namespace

bool anchor_graph_metrics_dense_prim(const vector<Vec3> &placed_points,
                                     const vector<size_t> &anchor_indices,
                                     double &mst2, double &max_d2,
                                     const PbctopoCutCandidate *incumbent) {
  mst2 = 0.0;
  max_d2 = 0.0;
  if (anchor_indices.size() < 2)
    return true;

  const size_t n = anchor_indices.size();
  thread_local vector<double> best_storage;
  thread_local vector<unsigned char> used_storage;
  thread_local vector<double> px_storage;
  thread_local vector<double> py_storage;
  thread_local vector<double> pz_storage;

  if (best_storage.size() < n)
    best_storage.resize(n);
  if (used_storage.size() < n)
    used_storage.resize(n);
  if (px_storage.size() < n)
    px_storage.resize(n);
  if (py_storage.size() < n)
    py_storage.resize(n);
  if (pz_storage.size() < n)
    pz_storage.resize(n);

  const double inf = numeric_limits<double>::infinity();
  fill(best_storage.begin(), best_storage.begin() + n, inf);
  fill(used_storage.begin(), used_storage.begin() + n, static_cast<unsigned char>(0));
  for (size_t i = 0; i < n; ++i) {
    const Vec3 &point = placed_points[anchor_indices[i]];
    px_storage[i] = point.x;
    py_storage[i] = point.y;
    pz_storage[i] = point.z;
  }

  double *best = best_storage.data();
  unsigned char *used = used_storage.data();
  const double *px = px_storage.data();
  const double *py = py_storage.data();
  const double *pz = pz_storage.data();
  best[0] = 0.0;
  const bool has_incumbent =
      (incumbent != nullptr && incumbent->valid &&
       isfinite(incumbent->anchor_mst2) && isfinite(incumbent->anchor_max_d2));
  const double incumbent_anchor_max_d2 =
      has_incumbent ? incumbent->anchor_max_d2 : inf;

  for (size_t step = 0; step < n; ++step) {
    double next_cost = inf;
    const size_t next = select_anchor_next(best, used, n, next_cost);
    if (next == numeric_limits<size_t>::max())
      return false;
    used[next] = 1;
    mst2 += next_cost;
    max_d2 = max(max_d2, next_cost);
    if (has_incumbent && max_d2 > incumbent_anchor_max_d2 + 1.0e-6) {
      return false;
    }
    const double sx = px[next];
    const double sy = py[next];
    const double sz = pz[next];
    relax_anchor_best(best, px, py, pz, n, sx, sy, sz);
  }
  return true;
}

namespace {

std::atomic<std::uint64_t> g_dense_calls{0};
std::atomic<std::uint64_t> g_fused_calls{0};
std::atomic<std::uint64_t> g_fused_incumbent_exits{0};

// Which kernel relax_anchor_best dispatches to.  The fused scan reproduces
// the two AVX2 kernels; with the scalar kernel the dense scan runs.
enum class AnchorKernelKind { Avx2Fma, Avx2, Scalar };

AnchorKernelKind anchor_kernel_kind() noexcept {
  static const AnchorKernelKind kind = []() noexcept {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    if (titan_simd_dispatch::runtime_has_avx2_fma())
      return AnchorKernelKind::Avx2Fma;
    if (titan_simd_dispatch::runtime_has_avx2())
      return AnchorKernelKind::Avx2;
#endif
    return AnchorKernelKind::Scalar;
  }();
  return kind;
}

// The anchors not yet in the tree, compacted, with their keys and original
// indices, padded to a multiple of sixteen with entries never selected
// (NaN coordinates make their keys NaN).
struct FusedPrimWorkspace {
  std::vector<double> px, py, pz;
  std::vector<double> ax, ay, az, key, index;
  // Position in the compacted arrays of each anchor below n - n % 4.
  std::vector<std::size_t> position;
};

FusedPrimWorkspace &fused_prim_workspace() {
  thread_local FusedPrimWorkspace workspace;
  return workspace;
}

#if TITAN_SIMD_X86
// The smallest key of four independent running minima of four lanes each,
// ties to the smallest original index.  Each running minimum holds the
// smallest (key, index) pair of its entries, and that order is associative,
// so the result is the smallest pair over all entries: the dense scan's
// first index among the smallest keys.
TITAN_TARGET_AVX2
inline void reduce_fused_minima(const __m256d (&vmin)[4],
                                const __m256d (&vidx)[4], double &key,
                                double &index) noexcept {
  alignas(32) double m[16];
  alignas(32) double i[16];
  for (int a = 0; a < 4; ++a) {
    _mm256_store_pd(m + 4 * a, vmin[a]);
    _mm256_store_pd(i + 4 * a, vidx[a]);
  }
  key = m[0];
  index = i[0];
  for (int lane = 1; lane < 16; ++lane) {
    if (m[lane] < key || (m[lane] == key && i[lane] < index)) {
      key = m[lane];
      index = i[lane];
    }
  }
}

// One step over the active anchors: relax their keys with anchor s (the
// arithmetic of relax_anchor_best_avx2_fma, lane for lane) and find the
// smallest key, ties to the smallest original index.  Four independent
// running minima keep the compare-and-blend chains short.
TITAN_TARGET_AVX2_FMA
void fused_prim_pass_avx2_fma(const double *ax, const double *ay,
                              const double *az, double *key,
                              const double *index, std::size_t padded16,
                              double sx, double sy, double sz,
                              double &min_key, double &min_index) noexcept {
  const double inf = std::numeric_limits<double>::infinity();
  const __m256d vsx = _mm256_set1_pd(sx);
  const __m256d vsy = _mm256_set1_pd(sy);
  const __m256d vsz = _mm256_set1_pd(sz);
  __m256d vmin[4], vidx[4];
  for (int a = 0; a < 4; ++a) {
    vmin[a] = _mm256_set1_pd(inf);
    vidx[a] = _mm256_set1_pd(inf);
  }
  for (std::size_t j = 0; j < padded16; j += 16) {
    for (int a = 0; a < 4; ++a) {
      const std::size_t o = j + 4 * static_cast<std::size_t>(a);
      const __m256d dx = _mm256_sub_pd(vsx, _mm256_loadu_pd(ax + o));
      const __m256d dy = _mm256_sub_pd(vsy, _mm256_loadu_pd(ay + o));
      const __m256d dz = _mm256_sub_pd(vsz, _mm256_loadu_pd(az + o));
      const __m256d d2 =
          _mm256_fmadd_pd(dz, dz,
                          _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx)));
      const __m256d k = _mm256_min_pd(_mm256_loadu_pd(key + o), d2);
      _mm256_storeu_pd(key + o, k);
      const __m256d idx = _mm256_loadu_pd(index + o);
      const __m256d take = _mm256_or_pd(
          _mm256_cmp_pd(k, vmin[a], _CMP_LT_OQ),
          _mm256_and_pd(_mm256_cmp_pd(k, vmin[a], _CMP_EQ_OQ),
                        _mm256_cmp_pd(idx, vidx[a], _CMP_LT_OQ)));
      vmin[a] = _mm256_blendv_pd(vmin[a], k, take);
      vidx[a] = _mm256_blendv_pd(vidx[a], idx, take);
    }
  }
  reduce_fused_minima(vmin, vidx, min_key, min_index);
}

// The same with the arithmetic of relax_anchor_best_avx2 (no FMA).
TITAN_TARGET_AVX2
void fused_prim_pass_avx2(const double *ax, const double *ay, const double *az,
                          double *key, const double *index,
                          std::size_t padded16, double sx, double sy,
                          double sz, double &min_key,
                          double &min_index) noexcept {
  const double inf = std::numeric_limits<double>::infinity();
  const __m256d vsx = _mm256_set1_pd(sx);
  const __m256d vsy = _mm256_set1_pd(sy);
  const __m256d vsz = _mm256_set1_pd(sz);
  __m256d vmin[4], vidx[4];
  for (int a = 0; a < 4; ++a) {
    vmin[a] = _mm256_set1_pd(inf);
    vidx[a] = _mm256_set1_pd(inf);
  }
  for (std::size_t j = 0; j < padded16; j += 16) {
    for (int a = 0; a < 4; ++a) {
      const std::size_t o = j + 4 * static_cast<std::size_t>(a);
      const __m256d dx = _mm256_sub_pd(vsx, _mm256_loadu_pd(ax + o));
      const __m256d dy = _mm256_sub_pd(vsy, _mm256_loadu_pd(ay + o));
      const __m256d dz = _mm256_sub_pd(vsz, _mm256_loadu_pd(az + o));
      const __m256d d2 = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
          _mm256_mul_pd(dz, dz));
      const __m256d k = _mm256_min_pd(_mm256_loadu_pd(key + o), d2);
      _mm256_storeu_pd(key + o, k);
      const __m256d idx = _mm256_loadu_pd(index + o);
      const __m256d take = _mm256_or_pd(
          _mm256_cmp_pd(k, vmin[a], _CMP_LT_OQ),
          _mm256_and_pd(_mm256_cmp_pd(k, vmin[a], _CMP_EQ_OQ),
                        _mm256_cmp_pd(idx, vidx[a], _CMP_LT_OQ)));
      vmin[a] = _mm256_blendv_pd(vmin[a], k, take);
      vidx[a] = _mm256_blendv_pd(vidx[a], idx, take);
    }
  }
  reduce_fused_minima(vmin, vidx, min_key, min_index);
}
#endif

} // namespace

// The dense scan, step for step, with two changes that leave every value as
// it was.  (1) Selection and relaxation share one pass: the keys relaxed by
// the vertex added in this step are scanned for the next minimum right away,
// in AVX2 lanes, ties to the smallest original index as the dense scan's
// first-index rule.  (2) Only anchors not yet in the tree are scanned: they
// are kept compacted, with their original index.  Each key is still
// min(key, d2) with d2 computed by the arithmetic of the dense kernel at the
// anchor's original position: the vector formula for anchors below
// n - n % 4, and for the last n % 4 anchors a call of the kernel itself on
// that position (its scalar tail).
bool anchor_graph_metrics_ok(const vector<Vec3> &placed_points,
                             const vector<size_t> &anchor_indices,
                             double &mst2, double &max_d2,
                             const PbctopoCutCandidate *incumbent) {
  const size_t n = anchor_indices.size();
  const AnchorKernelKind kind = anchor_kernel_kind();
#if TITAN_SIMD_X86
  const bool fused_available = kind != AnchorKernelKind::Scalar;
#else
  const bool fused_available = false;
#endif
  if (n < kAnchorMstFusedMinAnchors || !fused_available ||
      n > (size_t{1} << 52)) {
    g_dense_calls.fetch_add(1, memory_order_relaxed);
    return anchor_graph_metrics_dense_prim(placed_points, anchor_indices, mst2,
                                           max_d2, incumbent);
  }
#if TITAN_SIMD_X86
  FusedPrimWorkspace &w = fused_prim_workspace();
  w.px.resize(n);
  w.py.resize(n);
  w.pz.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const Vec3 &point = placed_points[anchor_indices[i]];
    if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z)) {
      g_dense_calls.fetch_add(1, memory_order_relaxed);
      return anchor_graph_metrics_dense_prim(placed_points, anchor_indices,
                                             mst2, max_d2, incumbent);
    }
    w.px[i] = point.x;
    w.py[i] = point.y;
    w.pz[i] = point.z;
  }
  g_fused_calls.fetch_add(1, memory_order_relaxed);

  const double inf = numeric_limits<double>::infinity();
  const double nan = numeric_limits<double>::quiet_NaN();
  // Anchors [0, vector_end) are relaxed with the vector formula; the tail
  // (at most three anchors) with the kernel's own scalar code.
  const size_t vector_end = n - n % 4;
  size_t active = vector_end;
  const size_t capacity = vector_end + 16;
  w.ax.assign(capacity, nan);
  w.ay.assign(capacity, nan);
  w.az.assign(capacity, nan);
  w.key.assign(capacity, inf);
  w.index.assign(capacity, inf);
  w.position.resize(vector_end);
  for (size_t i = 0; i < vector_end; ++i) {
    w.ax[i] = w.px[i];
    w.ay[i] = w.py[i];
    w.az[i] = w.pz[i];
    w.index[i] = static_cast<double>(i);
    w.position[i] = i;
  }
  array<double, 4> tail_key{inf, inf, inf, inf};
  array<unsigned char, 4> tail_used{0, 0, 0, 0};
  const size_t tail_count = n - vector_end;

  mst2 = 0.0;
  max_d2 = 0.0;
  const bool has_incumbent =
      (incumbent != nullptr && incumbent->valid &&
       isfinite(incumbent->anchor_mst2) && isfinite(incumbent->anchor_max_d2));
  const double incumbent_anchor_max_d2 =
      has_incumbent ? incumbent->anchor_max_d2 : inf;

  // Step 0 of the dense scan: anchor 0 has key 0, every other key is +inf.
  size_t next = 0;
  double next_cost = 0.0;
  if (vector_end > 0) {
    w.key[0] = 0.0;
  } else {
    tail_key[0] = 0.0;
  }
  for (size_t step = 0; step < n; ++step) {
    // Leave the tree's new vertex out of the active set.
    if (next < vector_end) {
      const size_t p = w.position[next];
      const size_t last = active - 1;
      const size_t moved = static_cast<size_t>(w.index[last]);
      w.position[moved] = p;
      w.ax[p] = w.ax[last];
      w.ay[p] = w.ay[last];
      w.az[p] = w.az[last];
      w.key[p] = w.key[last];
      w.index[p] = w.index[last];
      w.ax[last] = nan;
      w.ay[last] = nan;
      w.az[last] = nan;
      w.key[last] = inf;
      w.index[last] = inf;
      active = last;
    } else {
      tail_used[next - vector_end] = 1;
    }
    mst2 += next_cost;
    max_d2 = max(max_d2, next_cost);
    if (has_incumbent && max_d2 > incumbent_anchor_max_d2 + 1.0e-6) {
      g_fused_incumbent_exits.fetch_add(1, memory_order_relaxed);
      return false;
    }
    if (step + 1 == n)
      break;
    const double sx = w.px[next];
    const double sy = w.py[next];
    const double sz = w.pz[next];
    double min_key = inf, min_index = inf;
    const size_t padded16 = (active + 15) / 16 * 16;
    if (kind == AnchorKernelKind::Avx2Fma) {
      fused_prim_pass_avx2_fma(w.ax.data(), w.ay.data(), w.az.data(),
                               w.key.data(), w.index.data(), padded16, sx,
                               sy, sz, min_key, min_index);
    } else {
      fused_prim_pass_avx2(w.ax.data(), w.ay.data(), w.az.data(),
                           w.key.data(), w.index.data(), padded16, sx, sy,
                           sz, min_key, min_index);
    }
    for (size_t t = 0; t < tail_count; ++t) {
      if (tail_used[t] != 0)
        continue;
      const size_t i = vector_end + t;
      relax_anchor_best(&tail_key[t], w.px.data() + i, w.py.data() + i,
                        w.pz.data() + i, 1, sx, sy, sz);
      const double index = static_cast<double>(i);
      if (tail_key[t] < min_key ||
          (tail_key[t] == min_key && index < min_index)) {
        min_key = tail_key[t];
        min_index = index;
      }
    }
    // The dense scan selects only keys below +inf.
    if (!(min_key < inf))
      return false;
    next = static_cast<size_t>(min_index);
    next_cost = min_key;
  }
  return true;
#else
  return anchor_graph_metrics_dense_prim(placed_points, anchor_indices, mst2,
                                         max_d2, incumbent);
#endif
}

PbctopoAnchorMstCounters pbctopo_anchor_mst_counters() {
  PbctopoAnchorMstCounters out;
  out.dense_calls = g_dense_calls.load(memory_order_relaxed);
  out.fused_calls = g_fused_calls.load(memory_order_relaxed);
  out.fused_incumbent_exits = g_fused_incumbent_exits.load(memory_order_relaxed);
  return out;
}

} // namespace titan_pbctopo
