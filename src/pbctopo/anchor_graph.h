#ifndef TITAN_PBCTOPO_ANCHOR_GRAPH_H
#define TITAN_PBCTOPO_ANCHOR_GRAPH_H

#include "config.h"
#include "types.h"
#include "../box_type.h"

#include <cstddef>
#include <vector>

namespace titan_pbctopo {

PbctopoAnchorGraphContext build_anchor_graph_context(
    const std::vector<PbctopoAtom> &atoms, box Box,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &anchor_indices, double contact_cutoff,
    PbctopoTimingStats *timing = nullptr);

bool anchor_graph_metrics_enhanced(
    const std::vector<Vec3> &placed_points,
    const std::vector<std::size_t> &anchor_indices,
    const std::vector<PbctopoContinuityPair> &continuity_pairs, double &mst2,
    double &max_d2, PbctopoAnchorGraphScore &anchor_graph,
    const PbctopoCutCandidate *incumbent, PbctopoRefineScorePhase phase,
    AnchorBackend backend, PbctopoTimingStats *timing = nullptr,
    const PbctopoAnchorGraphContext *anchor_context = nullptr);

int compare_anchor_graph_score(const PbctopoAnchorGraphScore &lhs,
                               const PbctopoAnchorGraphScore &rhs);

} // namespace titan_pbctopo

#endif
