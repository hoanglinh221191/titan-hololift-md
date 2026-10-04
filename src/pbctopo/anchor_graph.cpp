#include "anchor_graph.h"

#include "evidence_audit.h"

#include "../BoxMatrix.h"
#include "../titan_simd_dispatch.h"
#include "candidate_order.h"
#include "lattice_math.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

namespace titan_pbctopo {

using pbctopo_hull::norm2;

namespace {

constexpr double kDistanceEps = 1.0e-12;
constexpr size_t kPbctopoSpatialAnchorThreshold = 512;
constexpr size_t kPbctopoSpatialAnchorK = 12;
constexpr size_t kPbctopoSpatialAnchorMaxK = 64;

struct AnchorContactCellKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const AnchorContactCellKey &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }
};

struct AnchorContactCellKeyHash {
  size_t operator()(const AnchorContactCellKey &key) const noexcept {
    return ClashCellKeyHash{}(
        {static_cast<int64_t>(key.x), static_cast<int64_t>(key.y),
         static_cast<int64_t>(key.z)});
  }
};

struct PbctopoSparseAnchorEdge {
  size_t a = 0;
  size_t b = 0;
  double d2 = numeric_limits<double>::infinity();
};

struct PbctopoAnchorKdNode {
  size_t point_idx = 0;
  int axis = 0;
  int left = -1;
  int right = -1;
  double min_x = 0.0;
  double max_x = 0.0;
  double min_y = 0.0;
  double max_y = 0.0;
  double min_z = 0.0;
  double max_z = 0.0;
};

struct PbctopoAnchorUnitKey {
  int owner = 0;
  size_t fragment_id = 0;

  bool operator==(const PbctopoAnchorUnitKey &) const = default;
};

bool anchor_unit_less(const PbctopoAnchorUnitKey &lhs,
                      const PbctopoAnchorUnitKey &rhs) {
  if (lhs.owner != rhs.owner)
    return lhs.owner < rhs.owner;
  return lhs.fragment_id < rhs.fragment_id;
}

struct PbctopoAnchorUnitPairKey {
  PbctopoAnchorUnitKey lo;
  PbctopoAnchorUnitKey hi;

  bool operator==(const PbctopoAnchorUnitPairKey &) const = default;
};

struct PbctopoAnchorUnitPairKeyHash {
  size_t operator()(const PbctopoAnchorUnitPairKey &key) const noexcept {
    size_t value = hash<int>{}(key.lo.owner);
    auto combine = [&](size_t part) {
      value ^= part + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    };
    combine(hash<size_t>{}(key.lo.fragment_id));
    combine(hash<int>{}(key.hi.owner));
    combine(hash<size_t>{}(key.hi.fragment_id));
    return value;
  }
};

PbctopoAnchorUnitPairKey pbctopo_anchor_unit_pair_key(
    const PbctopoAtom &lhs, const PbctopoAtom &rhs) {
  PbctopoAnchorUnitKey lhs_key{lhs.owner, lhs.fragment_id};
  PbctopoAnchorUnitKey rhs_key{rhs.owner, rhs.fragment_id};
  if (anchor_unit_less(rhs_key, lhs_key))
    swap(lhs_key, rhs_key);
  return {lhs_key, rhs_key};
}

int compare_pbctopo_size_metric(size_t lhs, size_t rhs) {
  if (lhs == rhs)
    return 0;
  return lhs < rhs ? -1 : 1;
}

int compare_pbctopo_double_metric(double lhs, double rhs,
                                  double eps = 1.0e-6) {
  return compare_pbctopo_metric(lhs, rhs, {eps, 0.0});
}

} // namespace

PbctopoAnchorGraphContext build_anchor_graph_context(
    const vector<PbctopoAtom> &atoms, box Box,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, double contact_cutoff,
    PbctopoTimingStats *timing) {
  const auto build_started = chrono::steady_clock::now();
  PbctopoAnchorGraphContext context;
  context.anchor_count = anchor_indices.size();
  const BoxMatrix box_matrix(Box);
  const PbctopoLatticeMetric lattice(Box);
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      context.box_m[row][col] = box_matrix.m[row][col];
      context.box_inv[row][col] = box_matrix.inv[row][col];
    }
  }
  context.has_box = true;

  if (atoms.empty()) {
    if (timing != nullptr) {
      timing->pbctopo_anchor_context_build_s +=
          chrono::duration<double>(chrono::steady_clock::now() - build_started)
              .count();
    }
    return context;
  }

  vector<char> is_anchor(atoms.size(), 0);
  for (size_t anchor_idx : anchor_indices) {
    if (anchor_idx < atoms.size())
      is_anchor[anchor_idx] = 1;
  }

  context.ordered_edges.reserve(continuity_pairs.size());
  for (const auto &pair : continuity_pairs) {
    if (pair.a < atoms.size() && pair.b < atoms.size() &&
        is_anchor[pair.a] != 0 && is_anchor[pair.b] != 0 &&
        atoms[pair.a].owner == atoms[pair.b].owner &&
        atoms[pair.a].fragment_id == atoms[pair.b].fragment_id &&
        atoms[pair.a].chain_key == atoms[pair.b].chain_key) {
      context.ordered_edges.push_back(pair);
    }
  }

  size_t chain_begin = 0;
  while (chain_begin < anchor_indices.size()) {
    const size_t first_anchor = anchor_indices[chain_begin];
    const bool first_valid = first_anchor < atoms.size();
    const int owner = first_valid ? atoms[first_anchor].owner : 0;
    const size_t fragment_id =
        first_valid ? atoms[first_anchor].fragment_id : 0;
    const uint64_t chain_key = first_valid ? atoms[first_anchor].chain_key : 0;
    size_t chain_end = chain_begin + 1;
    while (chain_end < anchor_indices.size()) {
      const size_t anchor_idx = anchor_indices[chain_end];
      if (anchor_idx >= atoms.size() || atoms[anchor_idx].owner != owner ||
          atoms[anchor_idx].fragment_id != fragment_id ||
          atoms[anchor_idx].chain_key != chain_key) {
        break;
      }
      ++chain_end;
    }

    bool chain_has_metadata = true;
    int previous_residue = numeric_limits<int>::min();
    bool has_previous = false;
    for (size_t idx = chain_begin; idx < chain_end; ++idx) {
      const size_t anchor_idx = anchor_indices[idx];
      if (anchor_idx >= atoms.size() || !atoms[anchor_idx].has_topology_key) {
        chain_has_metadata = false;
        continue;
      }
      const int residue_id = atoms[anchor_idx].residue_id;
      if (has_previous && residue_id <= previous_residue)
        ++context.residue_order_violations;
      previous_residue = residue_id;
      has_previous = true;
    }

    if (chain_has_metadata && chain_end > chain_begin + 1) {
      context.endpoint_edges.push_back(
          {anchor_indices[chain_begin], anchor_indices[chain_end - 1]});
    }
    chain_begin = chain_end;
  }

  if (atoms.size() >= 2 && contact_cutoff > kDistanceEps) {
    context.soft_observed_contact_construction_attempted = true;
    if (timing != nullptr)
      ++timing->pbctopo_eobs_construction_attempts;
    struct AnchorContactCandidate {
      size_t a = 0;
      size_t b = 0;
      double d2 = numeric_limits<double>::infinity();
      Int3 nearest_image{};
      size_t equivalent_nearest_images = 0;
    };
    unordered_map<PbctopoAnchorUnitPairKey, vector<AnchorContactCandidate>,
                  PbctopoAnchorUnitPairKeyHash>
        contacts_by_unit_pair;
    contacts_by_unit_pair.reserve(16);
    unordered_map<AnchorContactCellKey, vector<size_t>,
                  AnchorContactCellKeyHash>
        contact_grid;
    contact_grid.reserve(max<size_t>(64, atoms.size() / 2));

    const double cutoff2 = contact_cutoff * contact_cutoff;
    const double sx_bound = min(
        0.5, contact_cutoff *
                 sqrt(box_matrix.inv[0][0] * box_matrix.inv[0][0] +
                      box_matrix.inv[1][0] * box_matrix.inv[1][0] +
                      box_matrix.inv[2][0] * box_matrix.inv[2][0]) +
                 1.0e-9);
    const double sy_bound = min(
        0.5, contact_cutoff *
                 sqrt(box_matrix.inv[0][1] * box_matrix.inv[0][1] +
                      box_matrix.inv[1][1] * box_matrix.inv[1][1] +
                      box_matrix.inv[2][1] * box_matrix.inv[2][1]) +
                 1.0e-9);
    const double sz_bound = min(
        0.5, contact_cutoff *
                 sqrt(box_matrix.inv[0][2] * box_matrix.inv[0][2] +
                      box_matrix.inv[1][2] * box_matrix.inv[1][2] +
                      box_matrix.inv[2][2] * box_matrix.inv[2][2]) +
                 1.0e-9);
    const auto checked_cell_count = [](double bound, int &count) {
      if (!isfinite(bound))
        return false;
      const double raw = floor(1.0 / max(bound, 1.0e-6));
      if (!isfinite(raw) || raw < 1.0 ||
          raw > static_cast<double>(numeric_limits<int>::max())) {
        return false;
      }
      count = static_cast<int>(raw);
      return true;
    };
    int nx = 0;
    int ny = 0;
    int nz = 0;
    const bool contact_cell_setup_valid =
        lattice.valid() && checked_cell_count(sx_bound, nx) &&
        checked_cell_count(sy_bound, ny) && checked_cell_count(sz_bound, nz);
    if (contact_cell_setup_valid) {
      auto wrap_index = [](int value, int mod) {
        int wrapped = value % mod;
        if (wrapped < 0)
          wrapped += mod;
        return wrapped;
      };
      const auto checked_cell_index = [](double scaled, int cell_count,
                                         int &index) {
        if (!isfinite(scaled) || cell_count <= 0)
          return false;
        const double wrapped = scaled - floor(scaled);
        const double raw = floor(wrapped * static_cast<double>(cell_count));
        if (!isfinite(raw))
          return false;
        index = clamp(static_cast<int>(raw), 0, cell_count - 1);
        return true;
      };
      auto insert_contact = [&](const PbctopoAnchorUnitPairKey &unit_pair,
                                size_t a, size_t b, double d2,
                                const PbctopoNearestImage &mic) {
        auto &contacts = contacts_by_unit_pair[unit_pair];
        const size_t pair_lo = min(a, b);
        const size_t pair_hi = max(a, b);
        for (const auto &existing : contacts) {
          if (min(existing.a, existing.b) == pair_lo &&
              max(existing.a, existing.b) == pair_hi) {
            return;
          }
        }
        Int3 reversed_image{};
        if (!checked_int3_negate(
                {mic.image_x, mic.image_y, mic.image_z}, reversed_image)) {
          ++context.soft_observed_contact_mic_query_failures;
          return;
        }
        // The stored orientation is (a=other,b=atom), while the MIC query was
        // other-atom. Store n_ab for the declared H(s_b-s_a+n_ab) convention.
        contacts.push_back({a, b, d2, reversed_image,
                            mic.equivalent_minima});
        sort(contacts.begin(), contacts.end(),
             [](const AnchorContactCandidate &lhs,
                const AnchorContactCandidate &rhs) {
               const int metric_cmp =
                   compare_pbctopo_metric(
                       lhs.d2, rhs.d2,
                       {PBCTOPO_EOBS_SELECTION_TIE_TOLERANCE_D2_A2, 0.0});
               if (metric_cmp != 0)
                 return metric_cmp < 0;
               if (lhs.a != rhs.a)
                 return lhs.a < rhs.a;
               return lhs.b < rhs.b;
             });
        if (contacts.size() > PBCTOPO_EOBS_MAX_PAIRS_PER_UNIT_PAIR)
          contacts.resize(PBCTOPO_EOBS_MAX_PAIRS_PER_UNIT_PAIR);
      };

      for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
        const auto &atom = atoms[atom_idx];
        if (!atom.is_heavy)
          continue;
        int cx = 0;
        int cy = 0;
        int cz = 0;
        if (!checked_cell_index(atom.sx, nx, cx) ||
            !checked_cell_index(atom.sy, ny, cy) ||
            !checked_cell_index(atom.sz, nz, cz)) {
          ++context.soft_observed_contact_atom_index_failures;
          continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
              const AnchorContactCellKey key{wrap_index(cx + dx, nx),
                                             wrap_index(cy + dy, ny),
                                             wrap_index(cz + dz, nz)};
              const auto it = contact_grid.find(key);
              if (it == contact_grid.end())
                continue;
              for (size_t other_idx : it->second) {
                const auto &other = atoms[other_idx];
                if (atom.owner == other.owner &&
                    atom.fragment_id == other.fragment_id)
                  continue;
                const PbctopoNearestImage mic = lattice.nearest_image_cartesian(
                    other.x - atom.x, other.y - atom.y, other.z - atom.z);
                if (!mic.valid) {
                  ++context.soft_observed_contact_mic_query_failures;
                  continue;
                }
                const double d2 = mic.d2;
                if (d2 > cutoff2)
                  continue;
                insert_contact(pbctopo_anchor_unit_pair_key(atom, other),
                               other_idx, atom_idx, d2, mic);
              }
            }
          }
        }
        contact_grid[{cx, cy, cz}].push_back(atom_idx);
      }

      size_t intra_owner_contact_pairs = 0;
      for (const auto &[unit_pair, contacts] : contacts_by_unit_pair) {
        for (const auto &contact : contacts) {
          context.soft_observed_contact_edges.push_back(
              {contact.a, contact.b, cutoff2, contact.nearest_image,
               contact.equivalent_nearest_images, true});
          if (unit_pair.lo.owner == unit_pair.hi.owner)
            ++intra_owner_contact_pairs;
        }
      }
      if (timing != nullptr)
        timing->pbctopo_anchor_intra_owner_contact_pairs +=
            intra_owner_contact_pairs;
    } else {
      ++context.soft_observed_contact_cell_setup_failures;
    }
    context.soft_observed_contact_construction_complete =
        context.soft_observed_contact_cell_setup_failures == 0 &&
        context.soft_observed_contact_atom_index_failures == 0 &&
        context.soft_observed_contact_mic_query_failures == 0;
    if (!context.soft_observed_contact_construction_complete)
      context.soft_observed_contact_edges.clear();
    if (timing != nullptr) {
      timing->pbctopo_eobs_construction_completed +=
          context.soft_observed_contact_construction_complete ? 1 : 0;
      timing->pbctopo_eobs_construction_failures +=
          context.soft_observed_contact_construction_complete ? 0 : 1;
      timing->pbctopo_eobs_cell_setup_failures +=
          context.soft_observed_contact_cell_setup_failures;
      timing->pbctopo_eobs_atom_index_failures +=
          context.soft_observed_contact_atom_index_failures;
      timing->pbctopo_eobs_mic_query_failures +=
          context.soft_observed_contact_mic_query_failures;
    }
  }

  if (timing != nullptr) {
    timing->pbctopo_anchor_order_edges += context.ordered_edges.size();
    timing->pbctopo_anchor_endpoint_edges += context.endpoint_edges.size();
    timing->pbctopo_anchor_contact_pairs +=
        context.soft_observed_contact_edges.size();
    timing->pbctopo_soft_observed_contact_pairs +=
        context.soft_observed_contact_edges.size();
    timing->pbctopo_anchor_residue_order_violations +=
        context.residue_order_violations;
    timing->pbctopo_anchor_context_build_s +=
        chrono::duration<double>(chrono::steady_clock::now() - build_started)
            .count();
  }
  return context;
}

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

bool anchor_graph_metrics_ok(const vector<Vec3> &placed_points,
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

double anchor_axis_value(const Vec3 &point, int axis) {
  switch (axis) {
  case 1:
    return point.y;
  case 2:
    return point.z;
  case 0:
  default:
    return point.x;
  }
}

double anchor_kd_bbox_distance2(const PbctopoAnchorKdNode &node,
                                const Vec3 &point) {
  const double dx = (point.x < node.min_x)
                        ? (node.min_x - point.x)
                        : ((point.x > node.max_x) ? (point.x - node.max_x) : 0.0);
  const double dy = (point.y < node.min_y)
                        ? (node.min_y - point.y)
                        : ((point.y > node.max_y) ? (point.y - node.max_y) : 0.0);
  const double dz = (point.z < node.min_z)
                        ? (node.min_z - point.z)
                        : ((point.z > node.max_z) ? (point.z - node.max_z) : 0.0);
  return dx * dx + dy * dy + dz * dz;
}

int build_anchor_kdtree_recursive(vector<size_t> &order, size_t begin, size_t end,
                                  const vector<Vec3> &points,
                                  vector<PbctopoAnchorKdNode> &nodes) {
  if (begin >= end)
    return -1;

  double min_x = numeric_limits<double>::infinity();
  double max_x = -numeric_limits<double>::infinity();
  double min_y = numeric_limits<double>::infinity();
  double max_y = -numeric_limits<double>::infinity();
  double min_z = numeric_limits<double>::infinity();
  double max_z = -numeric_limits<double>::infinity();
  for (size_t idx = begin; idx < end; ++idx) {
    const Vec3 &point = points[order[idx]];
    min_x = min(min_x, point.x);
    max_x = max(max_x, point.x);
    min_y = min(min_y, point.y);
    max_y = max(max_y, point.y);
    min_z = min(min_z, point.z);
    max_z = max(max_z, point.z);
  }

  int axis = 0;
  const double span_x = max_x - min_x;
  const double span_y = max_y - min_y;
  const double span_z = max_z - min_z;
  if (span_y > span_x && span_y >= span_z) {
    axis = 1;
  } else if (span_z > span_x && span_z > span_y) {
    axis = 2;
  }

  const size_t mid = begin + (end - begin) / 2;
  nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
              [&](size_t lhs, size_t rhs) {
                return anchor_axis_value(points[lhs], axis) <
                       anchor_axis_value(points[rhs], axis);
              });

  const int node_idx = static_cast<int>(nodes.size());
  nodes.push_back({});
  const int left = build_anchor_kdtree_recursive(order, begin, mid, points, nodes);
  const int right =
      build_anchor_kdtree_recursive(order, mid + 1, end, points, nodes);

  PbctopoAnchorKdNode &node = nodes[node_idx];
  node.point_idx = order[mid];
  node.axis = axis;
  node.left = left;
  node.right = right;
  node.min_x = min_x;
  node.max_x = max_x;
  node.min_y = min_y;
  node.max_y = max_y;
  node.min_z = min_z;
  node.max_z = max_z;
  return node_idx;
}

void query_anchor_kdtree_knn(
    const vector<PbctopoAnchorKdNode> &nodes, int node_idx,
    const vector<Vec3> &points, const Vec3 &query_point, size_t query_idx,
    size_t knn_k, priority_queue<pair<double, size_t>> &neighbors) {
  if (node_idx < 0)
    return;

  const PbctopoAnchorKdNode &node = nodes[static_cast<size_t>(node_idx)];
  const Vec3 &point = points[node.point_idx];
  if (node.point_idx != query_idx) {
    const double dx = query_point.x - point.x;
    const double dy = query_point.y - point.y;
    const double dz = query_point.z - point.z;
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (neighbors.size() < knn_k) {
      neighbors.push({d2, node.point_idx});
    } else if (d2 < neighbors.top().first) {
      neighbors.pop();
      neighbors.push({d2, node.point_idx});
    }
  }

  const double query_axis = anchor_axis_value(query_point, node.axis);
  const double split_axis = anchor_axis_value(point, node.axis);
  const int near_child = (query_axis <= split_axis) ? node.left : node.right;
  const int far_child = (query_axis <= split_axis) ? node.right : node.left;

  query_anchor_kdtree_knn(nodes, near_child, points, query_point, query_idx,
                          knn_k, neighbors);
  if (far_child < 0)
    return;

  const double worst_d2 =
      (neighbors.size() < knn_k) ? numeric_limits<double>::infinity()
                                 : neighbors.top().first;
  if (anchor_kd_bbox_distance2(nodes[static_cast<size_t>(far_child)], query_point) <=
      worst_d2 + kDistanceEps) {
    query_anchor_kdtree_knn(nodes, far_child, points, query_point, query_idx,
                            knn_k, neighbors);
  }
}

bool build_anchor_knn_edges_kdtree(
    const vector<Vec3> &placed_points, const vector<size_t> &anchor_indices,
    const vector<PbctopoContinuityPair> &continuity_pairs, size_t knn_k,
    vector<PbctopoSparseAnchorEdge> &edges,
    double &omitted_edge_lower_bound_d2, double &build_s, double &query_s) {
  edges.clear();
  omitted_edge_lower_bound_d2 = numeric_limits<double>::infinity();
  build_s = 0.0;
  query_s = 0.0;
  const size_t n = anchor_indices.size();
  if (n < 2)
    return true;

  vector<Vec3> anchor_points(n);
  unordered_map<size_t, size_t> anchor_to_local;
  anchor_to_local.reserve(n * 2 + 1);
  for (size_t idx = 0; idx < n; ++idx) {
    anchor_points[idx] = placed_points[anchor_indices[idx]];
    anchor_to_local.emplace(anchor_indices[idx], idx);
  }

  const auto build_started = chrono::steady_clock::now();
  vector<size_t> order(n);
  iota(order.begin(), order.end(), 0);
  vector<PbctopoAnchorKdNode> nodes;
  nodes.reserve(n);
  const int root = build_anchor_kdtree_recursive(order, 0, n, anchor_points, nodes);
  build_s = chrono::duration<double>(chrono::steady_clock::now() - build_started)
                .count();
  if (root < 0)
    return false;

  unordered_set<uint64_t> seen_edges;
  seen_edges.reserve(n * (knn_k + 2));
  edges.reserve(n * (knn_k + 2));
  auto append_unique_edge = [&](size_t a, size_t b, double d2) {
    if (a == b)
      return;
    const size_t lo = min(a, b);
    const size_t hi = max(a, b);
    const uint64_t key =
        (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    if (!seen_edges.insert(key).second)
      return;
    edges.push_back({lo, hi, d2});
  };

  for (const auto &pair : continuity_pairs) {
    const auto it_a = anchor_to_local.find(pair.a);
    if (it_a == anchor_to_local.end())
      continue;
    const auto it_b = anchor_to_local.find(pair.b);
    if (it_b == anchor_to_local.end())
      continue;
    const Vec3 &lhs_point = placed_points[pair.a];
    const Vec3 &rhs_point = placed_points[pair.b];
    const double dx = lhs_point.x - rhs_point.x;
    const double dy = lhs_point.y - rhs_point.y;
    const double dz = lhs_point.z - rhs_point.z;
    append_unique_edge(it_a->second, it_b->second, dx * dx + dy * dy + dz * dz);
  }

  const size_t k_eff = min(knn_k, n - 1);
  const auto query_started = chrono::steady_clock::now();
  for (size_t query_idx = 0; query_idx < n; ++query_idx) {
    priority_queue<pair<double, size_t>> neighbors;
    query_anchor_kdtree_knn(nodes, root, anchor_points, anchor_points[query_idx],
                            query_idx, k_eff, neighbors);
    if (k_eff < n - 1 && neighbors.size() == k_eff) {
      omitted_edge_lower_bound_d2 =
          min(omitted_edge_lower_bound_d2, neighbors.top().first);
    }
    while (!neighbors.empty()) {
      append_unique_edge(query_idx, neighbors.top().second,
                         neighbors.top().first);
      neighbors.pop();
    }
  }
  query_s = chrono::duration<double>(chrono::steady_clock::now() - query_started)
                .count();
  return true;
}

bool compute_sparse_anchor_mst(const vector<PbctopoSparseAnchorEdge> &edges,
                               size_t node_count, double &mst2,
                               double &max_d2) {
  mst2 = 0.0;
  max_d2 = 0.0;
  if (node_count < 2)
    return true;
  if (edges.size() + 1 < node_count)
    return false;

  vector<PbctopoSparseAnchorEdge> ranked_edges = edges;
  sort(ranked_edges.begin(), ranked_edges.end(),
       [](const PbctopoSparseAnchorEdge &lhs,
          const PbctopoSparseAnchorEdge &rhs) { return lhs.d2 < rhs.d2; });

  vector<size_t> parent(node_count);
  vector<unsigned char> rank(node_count, 0);
  for (size_t idx = 0; idx < node_count; ++idx)
    parent[idx] = idx;

  const auto find_root = [&](size_t node, auto &&self) -> size_t {
    if (parent[node] == node)
      return node;
    parent[node] = self(parent[node], self);
    return parent[node];
  };
  auto unite = [&](size_t lhs, size_t rhs) {
    lhs = find_root(lhs, find_root);
    rhs = find_root(rhs, find_root);
    if (lhs == rhs)
      return false;
    if (rank[lhs] < rank[rhs])
      swap(lhs, rhs);
    parent[rhs] = lhs;
    if (rank[lhs] == rank[rhs])
      ++rank[lhs];
    return true;
  };

  size_t used_edges = 0;
  for (const auto &edge : ranked_edges) {
    if (!unite(edge.a, edge.b))
      continue;
    mst2 += edge.d2;
    max_d2 = max(max_d2, edge.d2);
    ++used_edges;
    if (used_edges + 1 == node_count)
      return true;
  }
  return false;
}

bool anchor_graph_metrics_spatial_or_dense(
    const vector<Vec3> &placed_points, const vector<size_t> &anchor_indices,
    const vector<PbctopoContinuityPair> &continuity_pairs, double &mst2,
    double &max_d2, const PbctopoCutCandidate *incumbent,
    PbctopoRefineScorePhase phase, AnchorBackend backend,
    PbctopoTimingStats *timing) {
  const bool allow_spatial =
      backend == AnchorBackend::SpatialAll &&
      anchor_indices.size() >= kPbctopoSpatialAnchorThreshold;
  if (!allow_spatial) {
    return anchor_graph_metrics_ok(placed_points, anchor_indices, mst2, max_d2,
                                   incumbent);
  }

  if (timing != nullptr)
    ++timing->component_graph_refine_anchor_spatial_calls;
  vector<PbctopoSparseAnchorEdge> sparse_edges;
  const size_t max_adaptive_k =
      min(anchor_indices.size() - 1, kPbctopoSpatialAnchorMaxK);
  size_t knn_k = min(kPbctopoSpatialAnchorK, max_adaptive_k);
  bool last_sparse_ok = false;
  while (knn_k > 0) {
    double omitted_edge_lower_bound_d2 =
        numeric_limits<double>::infinity();
    double build_s = 0.0;
    double query_s = 0.0;
    const bool built = build_anchor_knn_edges_kdtree(
        placed_points, anchor_indices, continuity_pairs, knn_k, sparse_edges,
        omitted_edge_lower_bound_d2, build_s, query_s);
    if (timing != nullptr) {
      timing->component_graph_refine_anchor_spatial_build_s += build_s;
      timing->component_graph_refine_anchor_spatial_query_s += query_s;
      timing->component_graph_refine_anchor_spatial_max_k =
          max(timing->component_graph_refine_anchor_spatial_max_k, knn_k);
    }
    last_sparse_ok =
        built && compute_sparse_anchor_mst(sparse_edges, anchor_indices.size(),
                                           mst2, max_d2);
    if (!last_sparse_ok && timing != nullptr)
      ++timing->component_graph_refine_anchor_spatial_disconnects;

    const double certificate_eps =
        1.0e-9 * max({1.0, abs(max_d2), abs(omitted_edge_lower_bound_d2)});
    const bool complete_graph = knn_k + 1 >= anchor_indices.size();
    const bool exact_certificate =
        last_sparse_ok &&
        (complete_graph ||
         max_d2 <= omitted_edge_lower_bound_d2 + certificate_eps);
    if (exact_certificate) {
      if (timing != nullptr)
        ++timing->component_graph_refine_anchor_spatial_exact_certified;
      const bool has_incumbent =
          incumbent != nullptr && incumbent->valid &&
          isfinite(incumbent->anchor_max_d2);
      return !has_incumbent ||
             max_d2 <= incumbent->anchor_max_d2 + 1.0e-6;
    }

    if (timing != nullptr)
      ++timing->component_graph_refine_anchor_spatial_uncertified;
    if (knn_k >= max_adaptive_k)
      break;
    const size_t next_k = min(max_adaptive_k, knn_k * 2);
    if (next_k == knn_k)
      break;
    knn_k = next_k;
    if (timing != nullptr)
      ++timing->component_graph_refine_anchor_spatial_expansions;
  }

  if (timing != nullptr) {
    ++timing->component_graph_refine_anchor_spatial_fallbacks;
  }
  const auto dense_started = chrono::steady_clock::now();
  const bool dense_ok = anchor_graph_metrics_ok(placed_points, anchor_indices, mst2,
                                                max_d2, incumbent);
  if (timing != nullptr) {
    timing->component_graph_refine_anchor_dense_fallback_s +=
        chrono::duration<double>(chrono::steady_clock::now() - dense_started)
            .count();
  }
  return dense_ok;
}

double anchor_context_distance2(const vector<Vec3> &placed_points, size_t a,
                                size_t b) {
  if (a >= placed_points.size() || b >= placed_points.size())
    return numeric_limits<double>::infinity();
  const Vec3 delta = placed_points[a] - placed_points[b];
  return norm2(delta);
}

double anchor_endpoint_image_gap(const vector<Vec3> &placed_points,
                                 const PbctopoAnchorGraphContext &context,
                                 size_t a, size_t b) {
  if (!context.has_box || a >= placed_points.size() || b >= placed_points.size())
    return numeric_limits<double>::infinity();
  const Vec3 delta = placed_points[b] - placed_points[a];
  const PbctopoLatticeMetric lattice(context.box_m, context.box_inv);
  const PbctopoNearestImage mic =
      lattice.nearest_image_cartesian(delta.x, delta.y, delta.z);
  if (!mic.valid)
    return numeric_limits<double>::infinity();
  const double raw_norm = sqrt(norm2(delta));
  const double mic_norm = sqrt(max(0.0, mic.d2));
  return max(0.0, raw_norm - mic_norm);
}

bool anchor_graph_metrics_enhanced(
    const vector<Vec3> &placed_points, const vector<size_t> &anchor_indices,
    const vector<PbctopoContinuityPair> &continuity_pairs, double &mst2,
    double &max_d2, PbctopoAnchorGraphScore &anchor_graph,
    const PbctopoCutCandidate *incumbent, PbctopoRefineScorePhase phase,
    AnchorBackend backend, PbctopoTimingStats *timing,
    const PbctopoAnchorGraphContext *anchor_context) {
  anchor_graph = {};
  if (!anchor_graph_metrics_spatial_or_dense(
          placed_points, anchor_indices, continuity_pairs, mst2, max_d2,
          incumbent, phase, backend, timing)) {
    return false;
  }
  anchor_graph.mst_mean_d2 =
      anchor_indices.size() > 1
          ? mst2 / static_cast<double>(anchor_indices.size() - 1)
          : 0.0;

  if (anchor_context == nullptr)
    return true;

  anchor_graph.residue_order_violations =
      anchor_context->use_order_metrics
          ? anchor_context->residue_order_violations
          : 0;
  const auto order_started = chrono::steady_clock::now();
  if (anchor_context->use_order_metrics &&
      !anchor_context->ordered_edges.empty()) {
    anchor_graph.order_max_d2 = 0.0;
    anchor_graph.order_path_d2 = 0.0;
    for (const auto &edge : anchor_context->ordered_edges) {
      const double d2 = anchor_context_distance2(placed_points, edge.a, edge.b);
      if (!isfinite(d2))
        continue;
      anchor_graph.order_max_d2 = max(anchor_graph.order_max_d2, d2);
      anchor_graph.order_path_d2 += d2;
      ++anchor_graph.order_edge_count;
    }
    anchor_graph.order_path_mean_d2 =
        anchor_graph.order_edge_count > 0
            ? anchor_graph.order_path_d2 /
                  static_cast<double>(anchor_graph.order_edge_count)
            : 0.0;
  }
  if (anchor_context->use_order_metrics &&
      !anchor_context->endpoint_edges.empty()) {
    anchor_graph.endpoint_image_gap = 0.0;
    for (const auto &edge : anchor_context->endpoint_edges) {
      const double gap =
          anchor_endpoint_image_gap(placed_points, *anchor_context, edge.a, edge.b);
      if (isfinite(gap))
        anchor_graph.endpoint_image_gap =
            max(anchor_graph.endpoint_image_gap, gap);
    }
  }
  if (timing != nullptr) {
    timing->component_graph_refine_anchor_order_s +=
        chrono::duration<double>(chrono::steady_clock::now() - order_started)
            .count();
  }

  const auto contact_started = chrono::steady_clock::now();
  // Optional shift-labelled contacts are audited against the final assignment.
  // They are deliberately neutral here: ranking all retained hard-feasible
  // relations by a global contact conjunction would penalize H0 and violate the
  // alternative-hypothesis semantics.
  anchor_graph.contact_map_lost = 0;
  anchor_graph.contact_map_lost_fraction =
      numeric_limits<double>::infinity();
  anchor_graph.contact_map_max_d2 = numeric_limits<double>::infinity();
  anchor_graph.contact_pair_count = 0;
  if (timing != nullptr) {
    timing->component_graph_refine_anchor_contact_s +=
        chrono::duration<double>(chrono::steady_clock::now() - contact_started)
            .count();
  }
  return true;
}

int compare_anchor_graph_score(const PbctopoAnchorGraphScore &lhs,
                               const PbctopoAnchorGraphScore &rhs) {
  int cmp = compare_pbctopo_size_metric(lhs.residue_order_violations,
                                        rhs.residue_order_violations);
  if (cmp != 0)
    return cmp;
  cmp = compare_pbctopo_double_metric(lhs.endpoint_image_gap,
                                      rhs.endpoint_image_gap);
  if (cmp != 0)
    return cmp;
  cmp = compare_pbctopo_double_metric(lhs.order_max_d2, rhs.order_max_d2);
  if (cmp != 0)
    return cmp;
  cmp = compare_pbctopo_double_metric(lhs.order_path_mean_d2,
                                      rhs.order_path_mean_d2);
  if (cmp != 0)
    return cmp;
  return compare_pbctopo_double_metric(lhs.mst_mean_d2, rhs.mst_mean_d2);
}

} // namespace titan_pbctopo
