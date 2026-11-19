#ifndef TITAN_PBCTOPO_ANCHOR_MST_RIGID_H
#define TITAN_PBCTOPO_ANCHOR_MST_RIGID_H

#include "types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace titan_pbctopo {

// The anchor minimum spanning tree of anchor_graph_metrics_ok for points that
// move as rigid groups.
//
// A component search scores many placements of one frame; each placement
// translates whole components and leaves their shape as it is.  The dense
// Prim scan of anchor_mst.cpp pays O(n^2) for every placement.  Here the
// part of the answer that does not depend on the placement is kept:
//
//   * inside a group, the edges that can be in a minimum spanning tree of
//     the group (all others have a path of strictly shorter edges between
//     their ends);
//   * between two groups at a given relative translation, the cross edges
//     no longer than the larger of the two groups' longest tree edge and the
//     groups' closest pair (any longer cross edge has a strictly shorter
//     path through the closest pair).
//
// Every edge that the dense scan can select is one of these (cut property),
// so Prim's rule on this sparse graph, with the dense scan's keys and its
// first-index rule for ties, adds the same vertices in the same order with
// the same keys: mst2, max_d2 and the incumbent exit are those of
// anchor_graph_metrics_ok bit for bit.  Edge sets are built from the
// reference coordinates with a margin that covers the rounding of
// "reference + translation"; each call first checks that the points are the
// reference translated group by group, and otherwise reports that it does
// not apply (the caller then runs the dense scan).
class PbctopoRigidAnchorMst;

// reference: the atoms as the placements translate them; groups: the atom
// indices of each rigid group (the components).  Cheap; the edge sets are
// built at the first call for an anchor list.
std::shared_ptr<PbctopoRigidAnchorMst> make_pbctopo_rigid_anchor_mst(
    const std::vector<PbctopoAtom> &reference,
    const std::vector<std::vector<std::size_t>> &groups);

// true: mst2, max_d2 and ok hold what anchor_graph_metrics_ok would return
// for these arguments.  false: not applicable here (anchor list too short or
// not covered by the groups, points not a rigid placement of the reference,
// no AVX2+FMA kernel, or a pair of groups with too many close cross pairs);
// the outputs are then unspecified.
bool pbctopo_rigid_anchor_mst_metrics(
    const PbctopoRigidAnchorMst &cache,
    const std::vector<Vec3> &placed_points,
    const std::vector<std::size_t> &anchor_indices, double &mst2,
    double &max_d2, const PbctopoCutCandidate *incumbent, bool &ok);

// Process-wide counters, for the run log.
struct PbctopoRigidAnchorMstCounters {
  std::uint64_t calls = 0;            // answered by the sparse scan
  std::uint64_t not_rigid = 0;        // points were not a rigid placement
  std::uint64_t not_covered = 0;      // list short, uncovered, or over a cap
  std::uint64_t anchor_sets = 0;      // anchor lists prepared
  std::uint64_t cross_tables = 0;     // (group pair, translation) tables built
  std::uint64_t cross_edges = 0;      // edges in those tables
  std::uint64_t checked = 0;          // calls compared with the dense scan
};
PbctopoRigidAnchorMstCounters pbctopo_rigid_anchor_mst_counters();

} // namespace titan_pbctopo

#endif
