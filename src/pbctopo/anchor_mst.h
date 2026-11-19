#ifndef TITAN_PBCTOPO_ANCHOR_MST_H
#define TITAN_PBCTOPO_ANCHOR_MST_H

#include "types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace titan_pbctopo {

// Minimum spanning tree of the anchors under squared Euclidean distance, grown
// by Prim's rule from anchor 0: mst2 is the sum of the tree's squared edge
// lengths, added in the order Prim adds the vertices, and max_d2 the longest.
// With a valid incumbent, false is returned at the first step whose running
// max_d2 exceeds incumbent->anchor_max_d2 + 1e-6; mst2 and max_d2 then hold
// the values of that step.
//
// For at least kAnchorMstFusedMinAnchors anchors with finite coordinates and
// an AVX2 kernel this runs the dense scan with selection and relaxation fused
// into one AVX2 pass over the anchors not yet in the tree (anchor_mst.cpp);
// every key, every selection and the order of the sums are those of
// anchor_graph_metrics_dense_prim, so mst2 and max_d2 are the same bit for
// bit.
bool anchor_graph_metrics_ok(const std::vector<Vec3> &placed_points,
                             const std::vector<std::size_t> &anchor_indices,
                             double &mst2, double &max_d2,
                             const PbctopoCutCandidate *incumbent);

// The O(n^2) Prim over all anchor pairs as it was written first: the
// reference for, and the fallback of, anchor_graph_metrics_ok.
bool anchor_graph_metrics_dense_prim(
    const std::vector<Vec3> &placed_points,
    const std::vector<std::size_t> &anchor_indices, double &mst2,
    double &max_d2, const PbctopoCutCandidate *incumbent);

inline constexpr std::size_t kAnchorMstFusedMinAnchors = 256;

// The relaxation kernel of the dense scan, best[i] = min(best[i], |s - p_i|^2)
// for i < n, as the scan dispatches it; for code that must reproduce the
// scan's keys (anchor_mst_rigid.cpp).
void pbctopo_anchor_relax_keys(double *best, const double *px,
                               const double *py, const double *pz,
                               std::size_t n, double sx, double sy,
                               double sz) noexcept;

// Whether that kernel is the AVX2+FMA one.
bool pbctopo_anchor_kernel_is_avx2_fma() noexcept;

// Process-wide counters of the two paths, for the run log.
struct PbctopoAnchorMstCounters {
  std::uint64_t dense_calls = 0;
  std::uint64_t fused_calls = 0;
  std::uint64_t fused_incumbent_exits = 0;
};
PbctopoAnchorMstCounters pbctopo_anchor_mst_counters();

} // namespace titan_pbctopo

#endif
