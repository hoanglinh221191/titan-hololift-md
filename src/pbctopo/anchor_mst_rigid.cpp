#include "anchor_mst_rigid.h"

#include "anchor_mst.h"
#include "../titan_simd_dispatch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace titan_pbctopo {

namespace {

// ---------------------------------------------------------------------------
// Tolerances.  All lengths are in the coordinate unit of the points.
//
// A call is answered only if every point is its reference plus one
// translation per group to within kRigidTolerance per axis, and a cross table
// is reused only for a relative translation within kDeltaTolerance of the one
// it was built for.  With coordinates below kMaxCoordinate the rounding of
// the checks themselves is below 1e-10, so the difference vector of any two
// anchors deviates from its reference value by less than
// kDifferenceDeviation per axis.  For a squared length w that changes the
// value by at most 2*sqrt(3)*eps*sqrt(w) + 3*eps^2 plus the rounding of the
// two evaluations (a few 1e-16 w), which is below margin(w) = 1e-8 (w + 1)
// because sqrt(w) <= (w + 1) / 2.  An edge is dropped only when a path whose
// reference lengths are all below it by more than both margins exists, so
// the path is strictly shorter for the actual points as well.
// ---------------------------------------------------------------------------
constexpr double kMaxCoordinate = 1.0e5;
constexpr double kRigidTolerance = 5.0e-10;
constexpr double kDeltaTolerance = 1.0e-9;
[[maybe_unused]] constexpr double kDifferenceDeviation = 4.0e-9;
constexpr double kMarginScale = 1.0e-8;
// Box lower bounds are taken this much short per axis, and the result a
// little short again, so that they stay below every pair's computed length.
constexpr double kBoxSlack = 1.0e-9;
constexpr double kBoundDeflate = 1.0 - 1.0e-12;
// Keys of the cross-table map: the relative translation on this grid.  Only
// a hash; a hit is checked against kDeltaTolerance.
constexpr double kDeltaGrid = 1.0e-3;

constexpr size_t kLeafSize = 16;
constexpr size_t kMaxAnchorSets = 8;
// More kept pairs than this per anchor mean the sparse graph is not sparse:
// the dense scan is used instead.
constexpr size_t kMaxInternalPairsPerAnchor = 64;
constexpr size_t kMaxCrossPairsPerAnchor = 32;
constexpr size_t kMaxCrossEdgesPerSet = size_t{1} << 25;

inline double upper(double w) noexcept { return w + kMarginScale * (w + 1.0); }
inline double lower(double w) noexcept { return w - kMarginScale * (w + 1.0); }
// The largest reference length whose lower() does not exceed bound, rounded
// up: lower(w) <= bound  =>  w <= keep_limit(bound).
inline double keep_limit(double bound) noexcept {
  return (bound + kMarginScale) / (1.0 - kMarginScale) * (1.0 + 1.0e-12) +
         1.0e-300;
}

atomic<uint64_t> g_calls{0};
atomic<uint64_t> g_not_rigid{0};
atomic<uint64_t> g_not_covered{0};
atomic<uint64_t> g_anchor_sets{0};
atomic<uint64_t> g_cross_tables{0};
atomic<uint64_t> g_cross_edges{0};
atomic<uint64_t> g_checked{0};

bool env_flag(const char *name, bool fallback) noexcept {
  const char *value = getenv(name);
  if (value == nullptr || value[0] == '\0')
    return fallback;
  return !(value[0] == '0' || value[0] == 'n' || value[0] == 'N' ||
           value[0] == 'f' || value[0] == 'F');
}

// TITAN_PBCTOPO_RIGID_MST=0 turns the sparse scan off;
// TITAN_PBCTOPO_RIGID_MST_CHECK=1 runs the dense scan after every sparse one
// and stops the program if any output differs.
bool rigid_mst_enabled() noexcept {
  static const bool enabled = env_flag("TITAN_PBCTOPO_RIGID_MST", true);
  return enabled;
}
bool rigid_mst_check() noexcept {
  static const bool check = env_flag("TITAN_PBCTOPO_RIGID_MST_CHECK", false);
  return check;
}

// ---------------------------------------------------------------------------
// A k-d tree over the reference points of one group, points stored leaf by
// leaf.
// ---------------------------------------------------------------------------
struct KdNode {
  double lo[3];
  double hi[3];
  uint32_t begin = 0;
  uint32_t end = 0;
  int32_t left = -1;
  int32_t right = -1;
};

struct Group {
  // In tree order: reference coordinates and the anchor position (index into
  // the anchor list) of each point.
  vector<double> x, y, z;
  vector<uint32_t> anchor;
  vector<KdNode> nodes;
  // upper() of the longest edge of the group's reference spanning tree: any
  // two of its points are joined by a path of edges no longer than this for
  // the actual points.
  double bottleneck_upper = 0.0;
  bool tail_singleton = false;
};

int32_t build_kd_node(Group &group, vector<uint32_t> &order,
                      const vector<double> &rx, const vector<double> &ry,
                      const vector<double> &rz, uint32_t begin, uint32_t end) {
  KdNode node;
  node.begin = begin;
  node.end = end;
  for (int k = 0; k < 3; ++k) {
    node.lo[k] = numeric_limits<double>::infinity();
    node.hi[k] = -numeric_limits<double>::infinity();
  }
  for (uint32_t i = begin; i < end; ++i) {
    const double p[3] = {rx[order[i]], ry[order[i]], rz[order[i]]};
    for (int k = 0; k < 3; ++k) {
      node.lo[k] = min(node.lo[k], p[k]);
      node.hi[k] = max(node.hi[k], p[k]);
    }
  }
  const int32_t index = static_cast<int32_t>(group.nodes.size());
  group.nodes.push_back(node);
  if (end - begin > kLeafSize) {
    int axis = 0;
    for (int k = 1; k < 3; ++k) {
      if (node.hi[k] - node.lo[k] > node.hi[axis] - node.lo[axis])
        axis = k;
    }
    const vector<double> &coordinate = axis == 0 ? rx : (axis == 1 ? ry : rz);
    const uint32_t middle = begin + (end - begin) / 2;
    nth_element(order.begin() + begin, order.begin() + middle,
                order.begin() + end, [&](uint32_t a, uint32_t b) {
                  if (coordinate[a] != coordinate[b])
                    return coordinate[a] < coordinate[b];
                  return a < b;
                });
    const int32_t left = build_kd_node(group, order, rx, ry, rz, begin, middle);
    const int32_t right = build_kd_node(group, order, rx, ry, rz, middle, end);
    group.nodes[index].left = left;
    group.nodes[index].right = right;
  }
  return index;
}

// A lower bound of the squared length between a point of box a and a point
// of box b translated by d.
inline double box_lower_bound(const KdNode &a, const KdNode &b,
                              const double d[3]) noexcept {
  double sum = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double gap =
        max(max(b.lo[k] + d[k] - a.hi[k], a.lo[k] - (b.hi[k] + d[k])), 0.0);
    const double g = max(gap - kBoxSlack, 0.0);
    sum += g * g;
  }
  return sum * kBoundDeflate;
}

inline double reference_length2(const Group &a, uint32_t i, const Group &b,
                                uint32_t j, const double d[3]) noexcept {
  const double dx = (b.x[j] + d[0]) - a.x[i];
  const double dy = (b.y[j] + d[1]) - a.y[i];
  const double dz = (b.z[j] + d[2]) - a.z[i];
  return dx * dx + dy * dy + dz * dz;
}

inline bool is_leaf(const KdNode &node) noexcept { return node.left < 0; }

// The smallest reference length between a point of a and a point of b + d.
void closest_pair(const Group &a, int32_t na, const Group &b, int32_t nb,
                  const double d[3], double &best) {
  const KdNode &A = a.nodes[na];
  const KdNode &B = b.nodes[nb];
  if (box_lower_bound(A, B, d) > best)
    return;
  if (is_leaf(A) && is_leaf(B)) {
    for (uint32_t i = A.begin; i < A.end; ++i) {
      for (uint32_t j = B.begin; j < B.end; ++j)
        best = min(best, reference_length2(a, i, b, j, d));
    }
    return;
  }
  const bool split_a =
      !is_leaf(A) && (is_leaf(B) || A.end - A.begin >= B.end - B.begin);
  int32_t first_a = na, first_b = nb, second_a = na, second_b = nb;
  if (split_a) {
    first_a = A.left;
    second_a = A.right;
  } else {
    first_b = B.left;
    second_b = B.right;
  }
  const double first_bound =
      box_lower_bound(a.nodes[first_a], b.nodes[first_b], d);
  const double second_bound =
      box_lower_bound(a.nodes[second_a], b.nodes[second_b], d);
  if (second_bound < first_bound) {
    swap(first_a, second_a);
    swap(first_b, second_b);
  }
  closest_pair(a, first_a, b, first_b, d, best);
  closest_pair(a, second_a, b, second_b, d, best);
}

struct TreePair {
  double w;
  uint32_t i; // tree-order index in the first group
  uint32_t j; // tree-order index in the second group
};

// Every pair (point of a, point of b + d) whose reference length is at most
// limit.  Stops, returning false, once more than cap pairs were found.
bool pairs_within(const Group &a, int32_t na, const Group &b, int32_t nb,
                  const double d[3], double limit, size_t cap,
                  vector<TreePair> &out) {
  const KdNode &A = a.nodes[na];
  const KdNode &B = b.nodes[nb];
  if (box_lower_bound(A, B, d) > limit)
    return true;
  if (is_leaf(A) && is_leaf(B)) {
    for (uint32_t i = A.begin; i < A.end; ++i) {
      for (uint32_t j = B.begin; j < B.end; ++j) {
        const double w = reference_length2(a, i, b, j, d);
        if (w <= limit)
          out.push_back({w, i, j});
      }
    }
    return out.size() <= cap;
  }
  const bool split_a =
      !is_leaf(A) && (is_leaf(B) || A.end - A.begin >= B.end - B.begin);
  if (split_a) {
    return pairs_within(a, A.left, b, nb, d, limit, cap, out) &&
           pairs_within(a, A.right, b, nb, d, limit, cap, out);
  }
  return pairs_within(a, na, b, B.left, d, limit, cap, out) &&
         pairs_within(a, na, b, B.right, d, limit, cap, out);
}

// The same inside one group: every unordered pair once.
bool self_pairs_within(const Group &g, int32_t na, int32_t nb, double limit,
                       size_t cap, vector<TreePair> &out) {
  static const double zero[3] = {0.0, 0.0, 0.0};
  const KdNode &A = g.nodes[na];
  const KdNode &B = g.nodes[nb];
  if (na != nb) {
    if (box_lower_bound(A, B, zero) > limit)
      return true;
    if (is_leaf(A) && is_leaf(B)) {
      for (uint32_t i = A.begin; i < A.end; ++i) {
        for (uint32_t j = B.begin; j < B.end; ++j) {
          const double w = reference_length2(g, i, g, j, zero);
          if (w <= limit)
            out.push_back({w, i, j});
        }
      }
      return out.size() <= cap;
    }
    const bool split_a =
        !is_leaf(A) && (is_leaf(B) || A.end - A.begin >= B.end - B.begin);
    if (split_a) {
      return self_pairs_within(g, A.left, nb, limit, cap, out) &&
             self_pairs_within(g, A.right, nb, limit, cap, out);
    }
    return self_pairs_within(g, na, B.left, limit, cap, out) &&
           self_pairs_within(g, na, B.right, limit, cap, out);
  }
  if (is_leaf(A)) {
    for (uint32_t i = A.begin; i < A.end; ++i) {
      for (uint32_t j = i + 1; j < A.end; ++j) {
        const double w = reference_length2(g, i, g, j, zero);
        if (w <= limit)
          out.push_back({w, i, j});
      }
    }
    return out.size() <= cap;
  }
  return self_pairs_within(g, A.left, A.left, limit, cap, out) &&
         self_pairs_within(g, A.left, A.right, limit, cap, out) &&
         self_pairs_within(g, A.right, A.right, limit, cap, out);
}

struct UnionFind {
  vector<uint32_t> parent;
  explicit UnionFind(size_t n) : parent(n) {
    iota(parent.begin(), parent.end(), uint32_t{0});
  }
  uint32_t find(uint32_t v) noexcept {
    while (parent[v] != v) {
      parent[v] = parent[parent[v]];
      v = parent[v];
    }
    return v;
  }
  bool unite(uint32_t a, uint32_t b) noexcept {
    a = find(a);
    b = find(b);
    if (a == b)
      return false;
    parent[b] = a;
    return true;
  }
};

// ---------------------------------------------------------------------------
// Cross tables.
// ---------------------------------------------------------------------------
struct DeltaKey {
  int64_t x = 0, y = 0, z = 0;
  bool operator==(const DeltaKey &) const = default;
};
struct DeltaKeyHash {
  size_t operator()(const DeltaKey &key) const noexcept {
    auto mix = [](uint64_t value) noexcept {
      value += 0x9e3779b97f4a7c15ull;
      value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
      value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
      return value ^ (value >> 31);
    };
    return static_cast<size_t>(mix(static_cast<uint64_t>(key.x)) ^
                               (mix(static_cast<uint64_t>(key.y)) << 1) ^
                               (mix(static_cast<uint64_t>(key.z)) >> 1));
  }
};

struct CrossTable {
  double delta[3] = {0.0, 0.0, 0.0};
  // Anchor positions: first in the pair's first group, second in its second.
  vector<pair<uint32_t, uint32_t>> edges;
  bool over_cap = false;
};

struct PairSlot {
  mutex guard;
  unordered_map<DeltaKey, unique_ptr<CrossTable>, DeltaKeyHash> tables;
};

struct AnchorSet {
  vector<size_t> anchor_indices;
  size_t n = 0;
  size_t vector_end = 0;
  bool usable = false;
  vector<double> rx, ry, rz;      // reference coordinates per anchor
  vector<uint32_t> group_of;      // per anchor
  vector<uint32_t> group_first;   // per group: its first anchor
  vector<Group> groups;
  // Edges inside the groups, both directions.
  vector<uint32_t> adjacency_begin;
  vector<uint32_t> adjacency;
  unique_ptr<PairSlot[]> pair_slots;
  mutable atomic<size_t> cross_edges{0};

  size_t pair_index(size_t i, size_t j) const noexcept {
    // i < j
    return j * (j - 1) / 2 + i;
  }
};

} // namespace

class PbctopoRigidAnchorMst {
public:
  vector<double> ref_x, ref_y, ref_z; // per atom
  vector<uint32_t> group_of_atom;     // per atom, npos if in no group
  size_t group_count = 0;
  mutable shared_mutex sets_guard;
  mutable vector<unique_ptr<AnchorSet>> sets;
};

namespace {

constexpr uint32_t kNoGroup = numeric_limits<uint32_t>::max();

unique_ptr<AnchorSet> build_anchor_set(const PbctopoRigidAnchorMst &cache,
                                       const vector<size_t> &anchor_indices) {
  auto set = make_unique<AnchorSet>();
  set->anchor_indices = anchor_indices;
  const size_t n = anchor_indices.size();
  set->n = n;
  set->vector_end = n - n % 4;
  if (n >= (size_t{1} << 31))
    return set;
  set->rx.resize(n);
  set->ry.resize(n);
  set->rz.resize(n);
  set->group_of.assign(n, kNoGroup);

  // Groups: the given ones restricted to the anchors the dense kernel relaxes
  // with its vector formula, then one group per tail anchor.
  vector<uint32_t> group_of_source(cache.group_count, kNoGroup);
  vector<vector<uint32_t>> members;
  for (size_t i = 0; i < n; ++i) {
    const size_t atom = anchor_indices[i];
    if (atom >= cache.group_of_atom.size() ||
        cache.group_of_atom[atom] == kNoGroup) {
      return set;
    }
    set->rx[i] = cache.ref_x[atom];
    set->ry[i] = cache.ref_y[atom];
    set->rz[i] = cache.ref_z[atom];
    if (!(abs(set->rx[i]) <= kMaxCoordinate) ||
        !(abs(set->ry[i]) <= kMaxCoordinate) ||
        !(abs(set->rz[i]) <= kMaxCoordinate)) {
      return set;
    }
    uint32_t group = kNoGroup;
    if (i < set->vector_end) {
      uint32_t &mapped = group_of_source[cache.group_of_atom[atom]];
      if (mapped == kNoGroup) {
        mapped = static_cast<uint32_t>(members.size());
        members.emplace_back();
      }
      group = mapped;
    } else {
      group = static_cast<uint32_t>(members.size());
      members.emplace_back();
    }
    members[group].push_back(static_cast<uint32_t>(i));
    set->group_of[i] = group;
  }
  const size_t group_count = members.size();
  set->groups.resize(group_count);
  set->group_first.resize(group_count);

  vector<vector<uint32_t>> neighbours(n);
  vector<TreePair> pairs;
  vector<Vec3> prim_points;
  vector<size_t> prim_indices;
  for (size_t g = 0; g < group_count; ++g) {
    Group &group = set->groups[g];
    vector<uint32_t> order = members[g];
    set->group_first[g] = order.front();
    group.tail_singleton = order.front() >= set->vector_end;
    const size_t m = order.size();
    group.nodes.reserve(2 * m / kLeafSize + 2);
    build_kd_node(group, order, set->rx, set->ry, set->rz, 0,
                  static_cast<uint32_t>(m));
    group.anchor = order;
    group.x.resize(m);
    group.y.resize(m);
    group.z.resize(m);
    for (size_t i = 0; i < m; ++i) {
      group.x[i] = set->rx[order[i]];
      group.y[i] = set->ry[order[i]];
      group.z[i] = set->rz[order[i]];
    }
    if (m < 2) {
      group.bottleneck_upper = upper(0.0);
      continue;
    }
    // The longest edge of the group's reference spanning tree.
    prim_points.resize(m);
    prim_indices.resize(m);
    for (size_t i = 0; i < m; ++i) {
      prim_points[i] = {group.x[i], group.y[i], group.z[i]};
      prim_indices[i] = i;
    }
    double mst2 = 0.0, bottleneck = 0.0;
    if (!anchor_graph_metrics_ok(prim_points, prim_indices, mst2, bottleneck,
                                 nullptr) ||
        !isfinite(bottleneck)) {
      return set;
    }
    group.bottleneck_upper = upper(bottleneck * (1.0 + 1.0e-12));

    // Pairs that can be tree edges: no path of strictly shorter edges joins
    // their ends.  Longer than the bottleneck they cannot be.
    pairs.clear();
    const size_t cap = kMaxInternalPairsPerAnchor * m;
    if (!self_pairs_within(group, 0, 0, keep_limit(group.bottleneck_upper),
                           cap, pairs)) {
      return set;
    }
    sort(pairs.begin(), pairs.end(), [](const TreePair &a, const TreePair &b) {
      if (a.w != b.w)
        return a.w < b.w;
      if (a.i != b.i)
        return a.i < b.i;
      return a.j < b.j;
    });
    UnionFind joined(m);
    UnionFind kept(m);
    size_t merged = 0;
    size_t kept_components = m;
    for (size_t k = 0; k < pairs.size(); ++k) {
      const double lower_k = lower(pairs[k].w);
      while (merged < pairs.size() && upper(pairs[merged].w) < lower_k) {
        joined.unite(pairs[merged].i, pairs[merged].j);
        ++merged;
      }
      if (joined.find(pairs[k].i) == joined.find(pairs[k].j))
        continue;
      const uint32_t a = group.anchor[pairs[k].i];
      const uint32_t b = group.anchor[pairs[k].j];
      neighbours[a].push_back(b);
      neighbours[b].push_back(a);
      if (kept.unite(pairs[k].i, pairs[k].j))
        --kept_components;
    }
    if (kept_components != 1)
      return set;
  }

  set->adjacency_begin.resize(n + 1);
  size_t total = 0;
  for (size_t i = 0; i < n; ++i) {
    set->adjacency_begin[i] = static_cast<uint32_t>(total);
    total += neighbours[i].size();
  }
  set->adjacency_begin[n] = static_cast<uint32_t>(total);
  set->adjacency.reserve(total);
  for (size_t i = 0; i < n; ++i) {
    sort(neighbours[i].begin(), neighbours[i].end());
    set->adjacency.insert(set->adjacency.end(), neighbours[i].begin(),
                          neighbours[i].end());
  }
  set->pair_slots =
      make_unique<PairSlot[]>(max<size_t>(1, group_count * (group_count - 1) / 2));
  set->usable = true;
  return set;
}

// The cross edges of groups gi < gj with group gj translated by delta
// relative to group gi (both from their reference positions).
unique_ptr<CrossTable> build_cross_table(const AnchorSet &set, size_t gi,
                                         size_t gj, const double delta[3]) {
  auto table = make_unique<CrossTable>();
  memcpy(table->delta, delta, sizeof(table->delta));
  const Group &a = set.groups[gi];
  const Group &b = set.groups[gj];
  double closest = numeric_limits<double>::infinity();
  closest_pair(a, 0, b, 0, delta, closest);
  const double bound = max(
      {a.bottleneck_upper, b.bottleneck_upper, upper(closest * (1.0 + 1.0e-12))});
  vector<TreePair> pairs;
  const size_t cap =
      kMaxCrossPairsPerAnchor * (a.anchor.size() + b.anchor.size()) + 1024;
  if (!isfinite(bound) ||
      !pairs_within(a, 0, b, 0, delta, keep_limit(bound), cap, pairs)) {
    table->over_cap = true;
    return table;
  }
  table->edges.reserve(pairs.size());
  for (const TreePair &pair : pairs)
    table->edges.emplace_back(a.anchor[pair.i], b.anchor[pair.j]);
  sort(table->edges.begin(), table->edges.end());
  return table;
}

const CrossTable *find_cross_table(const AnchorSet &set, size_t gi, size_t gj,
                                   const double delta[3]) {
  DeltaKey key;
  key.x = llround(delta[0] / kDeltaGrid);
  key.y = llround(delta[1] / kDeltaGrid);
  key.z = llround(delta[2] / kDeltaGrid);
  PairSlot &slot = set.pair_slots[set.pair_index(gi, gj)];
  lock_guard lock(slot.guard);
  auto found = slot.tables.find(key);
  if (found == slot.tables.end()) {
    if (set.cross_edges.load(memory_order_relaxed) > kMaxCrossEdgesPerSet)
      return nullptr;
    unique_ptr<CrossTable> table = build_cross_table(set, gi, gj, delta);
    g_cross_tables.fetch_add(1, memory_order_relaxed);
    g_cross_edges.fetch_add(table->edges.size(), memory_order_relaxed);
    set.cross_edges.fetch_add(table->edges.size(), memory_order_relaxed);
    found = slot.tables.emplace(key, std::move(table)).first;
  }
  const CrossTable *table = found->second.get();
  for (int k = 0; k < 3; ++k) {
    if (!(abs(delta[k] - table->delta[k]) <= kDeltaTolerance))
      return nullptr; // another translation in the same grid cell
  }
  return table;
}

struct PrimWorkspace {
  vector<double> px, py, pz, key;
  vector<unsigned char> in_tree;
  vector<double> tx, ty, tz;
  vector<const CrossTable *> tables;
  // Cross edges of this call as linked half-edges per anchor.
  vector<int32_t> head;
  vector<uint32_t> touched;
  vector<uint32_t> half_to;
  vector<int32_t> half_next;
  vector<pair<double, uint32_t>> heap;
};

PrimWorkspace &prim_workspace() {
  thread_local PrimWorkspace workspace;
  return workspace;
}

#if TITAN_SIMD_X86
// The key of relax_anchor_best_avx2_fma for an anchor below n - n % 4:
// fma(dz, dz, fma(dy, dy, dx * dx)) with d = source - anchor.
TITAN_TARGET_AVX2_FMA
inline double vector_kernel_length2(double sx, double sy, double sz, double x,
                                    double y, double z) noexcept {
  const __m128d dx = _mm_set_sd(sx - x);
  const __m128d dy = _mm_set_sd(sy - y);
  const __m128d dz = _mm_set_sd(sz - z);
  return _mm_cvtsd_f64(
      _mm_fmadd_sd(dz, dz, _mm_fmadd_sd(dy, dy, _mm_mul_sd(dx, dx))));
}

struct HeapOrder {
  bool operator()(const pair<double, uint32_t> &a,
                  const pair<double, uint32_t> &b) const noexcept {
    // std::push_heap keeps the largest on top: order by "greater" to get the
    // smallest (key, index).
    if (a.first != b.first)
      return a.first > b.first;
    return a.second > b.second;
  }
};

// Prim's rule on the sparse graph with the dense scan's keys and its
// selection: smallest key, ties to the smallest anchor.  false: the scan ran
// out of reachable anchors, which a complete edge set excludes; the caller
// then runs the dense scan.
TITAN_TARGET_AVX2_FMA
bool sparse_prim_avx2_fma(const AnchorSet &set, PrimWorkspace &w, double &mst2,
                          double &max_d2, bool has_incumbent,
                          double incumbent_anchor_max_d2, bool &ok) {
  const size_t n = set.n;
  const size_t vector_end = set.vector_end;
  const size_t tail_count = n - vector_end;
  const double inf = numeric_limits<double>::infinity();
  const double *px = w.px.data();
  const double *py = w.py.data();
  const double *pz = w.pz.data();
  double *key = w.key.data();
  unsigned char *in_tree = w.in_tree.data();
  const uint32_t *adjacency_begin = set.adjacency_begin.data();
  const uint32_t *adjacency = set.adjacency.data();
  array<double, 4> tail_key{inf, inf, inf, inf};
  array<unsigned char, 4> tail_used{0, 0, 0, 0};
  auto &heap = w.heap;
  heap.clear();
  const HeapOrder order;

  mst2 = 0.0;
  max_d2 = 0.0;
  size_t next = 0;
  double next_cost = 0.0;
  for (size_t step = 0; step < n; ++step) {
    if (next < vector_end)
      in_tree[next] = 1;
    else
      tail_used[next - vector_end] = 1;
    mst2 += next_cost;
    max_d2 = max(max_d2, next_cost);
    if (has_incumbent && max_d2 > incumbent_anchor_max_d2 + 1.0e-6) {
      ok = false;
      return true;
    }
    if (step + 1 == n)
      break;
    const double sx = px[next];
    const double sy = py[next];
    const double sz = pz[next];
    for (uint32_t e = adjacency_begin[next]; e < adjacency_begin[next + 1];
         ++e) {
      const uint32_t v = adjacency[e];
      if (in_tree[v] != 0)
        continue;
      const double d2 = vector_kernel_length2(sx, sy, sz, px[v], py[v], pz[v]);
      if (d2 < key[v]) {
        key[v] = d2;
        heap.emplace_back(d2, v);
        push_heap(heap.begin(), heap.end(), order);
      }
    }
    for (int32_t h = w.head[next]; h >= 0; h = w.half_next[h]) {
      const uint32_t v = w.half_to[h];
      if (v >= vector_end || in_tree[v] != 0)
        continue;
      const double d2 = vector_kernel_length2(sx, sy, sz, px[v], py[v], pz[v]);
      if (d2 < key[v]) {
        key[v] = d2;
        heap.emplace_back(d2, v);
        push_heap(heap.begin(), heap.end(), order);
      }
    }
    while (!heap.empty() && (in_tree[heap.front().second] != 0 ||
                             heap.front().first != key[heap.front().second])) {
      pop_heap(heap.begin(), heap.end(), order);
      heap.pop_back();
    }
    double min_key = inf;
    size_t min_index = numeric_limits<size_t>::max();
    if (!heap.empty()) {
      min_key = heap.front().first;
      min_index = heap.front().second;
    }
    // The last n % 4 anchors: the dense kernel's own code for them.
    for (size_t t = 0; t < tail_count; ++t) {
      if (tail_used[t] != 0)
        continue;
      const size_t i = vector_end + t;
      pbctopo_anchor_relax_keys(&tail_key[t], px + i, py + i, pz + i, 1, sx,
                                sy, sz);
      if (tail_key[t] < min_key || (tail_key[t] == min_key && i < min_index)) {
        min_key = tail_key[t];
        min_index = i;
      }
    }
    if (!(min_key < inf))
      return false;
    if (min_index < vector_end) {
      pop_heap(heap.begin(), heap.end(), order);
      heap.pop_back();
    }
    next = min_index;
    next_cost = min_key;
  }
  ok = true;
  return true;
}
#endif

const AnchorSet *find_anchor_set(const PbctopoRigidAnchorMst &cache,
                                 const vector<size_t> &anchor_indices) {
  {
    shared_lock lock(cache.sets_guard);
    for (const auto &set : cache.sets) {
      if (set->anchor_indices == anchor_indices)
        return set.get();
    }
    if (cache.sets.size() >= kMaxAnchorSets)
      return nullptr;
  }
  unique_lock lock(cache.sets_guard);
  for (const auto &set : cache.sets) {
    if (set->anchor_indices == anchor_indices)
      return set.get();
  }
  if (cache.sets.size() >= kMaxAnchorSets)
    return nullptr;
  cache.sets.push_back(build_anchor_set(cache, anchor_indices));
  g_anchor_sets.fetch_add(1, memory_order_relaxed);
  return cache.sets.back().get();
}

} // namespace

shared_ptr<PbctopoRigidAnchorMst> make_pbctopo_rigid_anchor_mst(
    const vector<PbctopoAtom> &reference,
    const vector<vector<size_t>> &groups) {
  if (groups.empty() || groups.size() >= kNoGroup)
    return {};
  auto cache = make_shared<PbctopoRigidAnchorMst>();
  const size_t atom_count = reference.size();
  cache->ref_x.resize(atom_count);
  cache->ref_y.resize(atom_count);
  cache->ref_z.resize(atom_count);
  for (size_t i = 0; i < atom_count; ++i) {
    cache->ref_x[i] = reference[i].x;
    cache->ref_y[i] = reference[i].y;
    cache->ref_z[i] = reference[i].z;
  }
  cache->group_of_atom.assign(atom_count, kNoGroup);
  cache->group_count = groups.size();
  for (size_t g = 0; g < groups.size(); ++g) {
    for (size_t atom : groups[g]) {
      if (atom >= atom_count || cache->group_of_atom[atom] != kNoGroup)
        return {};
      cache->group_of_atom[atom] = static_cast<uint32_t>(g);
    }
  }
  return cache;
}

bool pbctopo_rigid_anchor_mst_metrics(const PbctopoRigidAnchorMst &cache,
                                      const vector<Vec3> &placed_points,
                                      const vector<size_t> &anchor_indices,
                                      double &mst2, double &max_d2,
                                      const PbctopoCutCandidate *incumbent,
                                      bool &ok) {
#if TITAN_SIMD_X86
  const size_t n = anchor_indices.size();
  if (!rigid_mst_enabled() || n < kAnchorMstFusedMinAnchors ||
      !pbctopo_anchor_kernel_is_avx2_fma()) {
    return false;
  }
  const AnchorSet *set_ptr = find_anchor_set(cache, anchor_indices);
  if (set_ptr == nullptr || !set_ptr->usable) {
    g_not_covered.fetch_add(1, memory_order_relaxed);
    return false;
  }
  const AnchorSet &set = *set_ptr;
  PrimWorkspace &w = prim_workspace();
  w.px.resize(n);
  w.py.resize(n);
  w.pz.resize(n);
  const size_t point_count = placed_points.size();
  for (size_t i = 0; i < n; ++i) {
    const size_t atom = anchor_indices[i];
    if (atom >= point_count) {
      g_not_covered.fetch_add(1, memory_order_relaxed);
      return false;
    }
    const Vec3 &point = placed_points[atom];
    w.px[i] = point.x;
    w.py[i] = point.y;
    w.pz[i] = point.z;
  }

  // Is this the reference, translated group by group?
  const size_t group_count = set.groups.size();
  w.tx.resize(group_count);
  w.ty.resize(group_count);
  w.tz.resize(group_count);
  for (size_t g = 0; g < group_count; ++g) {
    const uint32_t first = set.group_first[g];
    w.tx[g] = w.px[first] - set.rx[first];
    w.ty[g] = w.py[first] - set.ry[first];
    w.tz[g] = w.pz[first] - set.rz[first];
  }
  double worst = 0.0;
  double largest = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const uint32_t g = set.group_of[i];
    const double ex = abs((w.px[i] - set.rx[i]) - w.tx[g]);
    const double ey = abs((w.py[i] - set.ry[i]) - w.ty[g]);
    const double ez = abs((w.pz[i] - set.rz[i]) - w.tz[g]);
    worst = max(worst, max(ex, max(ey, ez)));
    largest = max(largest, max(abs(w.px[i]), max(abs(w.py[i]), abs(w.pz[i]))));
  }
  // NaN and infinite coordinates fail both comparisons.
  if (!(worst <= kRigidTolerance) || !(largest <= kMaxCoordinate)) {
    g_not_rigid.fetch_add(1, memory_order_relaxed);
    return false;
  }

  // The cross tables of this placement.
  w.tables.clear();
  for (size_t gj = 1; gj < group_count; ++gj) {
    for (size_t gi = 0; gi < gj; ++gi) {
      if (set.groups[gi].tail_singleton && set.groups[gj].tail_singleton)
        continue;
      const double delta[3] = {w.tx[gj] - w.tx[gi], w.ty[gj] - w.ty[gi],
                               w.tz[gj] - w.tz[gi]};
      const CrossTable *table = find_cross_table(set, gi, gj, delta);
      if (table == nullptr || table->over_cap) {
        g_not_covered.fetch_add(1, memory_order_relaxed);
        return false;
      }
      w.tables.push_back(table);
    }
  }
  if (w.head.size() < n)
    w.head.resize(n, -1);
  w.half_to.clear();
  w.half_next.clear();
  w.touched.clear();
  auto add_half = [&](uint32_t from, uint32_t to) {
    if (w.head[from] < 0)
      w.touched.push_back(from);
    w.half_to.push_back(to);
    w.half_next.push_back(w.head[from]);
    w.head[from] = static_cast<int32_t>(w.half_to.size() - 1);
  };
  for (const CrossTable *table : w.tables) {
    for (const auto &edge : table->edges) {
      add_half(edge.first, edge.second);
      add_half(edge.second, edge.first);
    }
  }

  const double inf = numeric_limits<double>::infinity();
  w.key.assign(n, inf);
  w.in_tree.assign(n, static_cast<unsigned char>(0));
  const bool has_incumbent =
      (incumbent != nullptr && incumbent->valid &&
       isfinite(incumbent->anchor_mst2) && isfinite(incumbent->anchor_max_d2));
  const double incumbent_anchor_max_d2 =
      has_incumbent ? incumbent->anchor_max_d2 : inf;
  const bool completed = sparse_prim_avx2_fma(
      set, w, mst2, max_d2, has_incumbent, incumbent_anchor_max_d2, ok);
  for (uint32_t anchor : w.touched)
    w.head[anchor] = -1;
  if (!completed) {
    g_not_covered.fetch_add(1, memory_order_relaxed);
    return false;
  }
  g_calls.fetch_add(1, memory_order_relaxed);

  if (rigid_mst_check()) {
    double dense_mst2 = 0.0, dense_max_d2 = 0.0;
    const bool dense_ok = anchor_graph_metrics_ok(
        placed_points, anchor_indices, dense_mst2, dense_max_d2, incumbent);
    g_checked.fetch_add(1, memory_order_relaxed);
    if (dense_ok != ok || memcmp(&dense_mst2, &mst2, sizeof(double)) != 0 ||
        memcmp(&dense_max_d2, &max_d2, sizeof(double)) != 0) {
      fprintf(stderr,
              "[pbctopo] rigid anchor MST differs from the dense scan: "
              "anchors=%zu ok=%d/%d mst2=%.17g/%.17g max_d2=%.17g/%.17g\n",
              n, static_cast<int>(ok), static_cast<int>(dense_ok), mst2,
              dense_mst2, max_d2, dense_max_d2);
      fflush(stderr);
      abort();
    }
  }
  return true;
#else
  (void)cache;
  (void)placed_points;
  (void)anchor_indices;
  (void)mst2;
  (void)max_d2;
  (void)incumbent;
  (void)ok;
  return false;
#endif
}

PbctopoRigidAnchorMstCounters pbctopo_rigid_anchor_mst_counters() {
  PbctopoRigidAnchorMstCounters out;
  out.calls = g_calls.load(memory_order_relaxed);
  out.not_rigid = g_not_rigid.load(memory_order_relaxed);
  out.not_covered = g_not_covered.load(memory_order_relaxed);
  out.anchor_sets = g_anchor_sets.load(memory_order_relaxed);
  out.cross_tables = g_cross_tables.load(memory_order_relaxed);
  out.cross_edges = g_cross_edges.load(memory_order_relaxed);
  out.checked = g_checked.load(memory_order_relaxed);
  return out;
}

} // namespace titan_pbctopo
