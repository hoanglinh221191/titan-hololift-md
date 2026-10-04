#ifndef TITAN_PBCTOPO_COMPONENT_GRAPH_H
#define TITAN_PBCTOPO_COMPONENT_GRAPH_H

#include "component_refine.h"
#include "config.h"
#include "types.h"
#include "../box_type.h"

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace titan_pbctopo {

bool component_bbox_may_clash(const PbctopoShiftedComponent &lhs,
                              const PbctopoShiftedComponent &rhs,
                              double cutoff);
std::vector<std::vector<PbctopoShiftedComponent>>
precompute_shifted_components(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::vector<std::size_t>> &components,
    const std::vector<PbctopoShift> &shifts, double clash_cutoff,
    double contact_cutoff, bool build_clash_grid);
double shifted_component_bbox_distance2(const PbctopoShiftedComponent &lhs,
                                        const PbctopoShiftedComponent &rhs);

bool better_pair_shift_score(const PbctopoPairShiftScore &lhs,
                             const PbctopoPairShiftScore &rhs);

bool better_component_graph_edge(const PbctopoGraphEdge &lhs,
                                 const PbctopoGraphEdge &rhs);
bool graph_edge_has_zero_shift(const PbctopoGraphEdge &edge);
bool graph_edge_matches_delta(const PbctopoGraphEdge &edge, const Int3 &delta);

std::vector<PbctopoGraphEdge> build_component_graph_edge_candidates(
    const std::vector<PbctopoAtom> &atoms, box Box,
    const std::vector<std::vector<std::size_t>> &components,
    const std::vector<PbctopoShift> &shifts,
    const RuntimeCutoffs &cutoffs, PbctopoTimingStats *timing,
    const std::vector<Int3> *warm_component_offsets,
    std::vector<PbctopoPairShiftCandidateSet> *pair_candidates_out = nullptr,
    const std::vector<PbctopoRequiredContact> *required_contacts = nullptr);

bool solve_component_offsets_from_ranked_edges(
    std::size_t component_count, box Box,
    const std::vector<std::vector<std::size_t>> &components,
    const std::vector<PbctopoGraphEdge> &ranked_edges,
    std::vector<Int3> &offsets);

bool solve_component_offsets_via_pair_graph(
    const std::vector<PbctopoAtom> &atoms, box Box,
    const std::vector<std::vector<std::size_t>> &components,
    const std::vector<PbctopoShift> &shifts,
    const std::vector<PbctopoLocalEdge> &local_edges,
    const std::unordered_set<std::uint64_t> &edge_set,
    std::vector<Vec3> &placed_points,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &anchor_indices,
    const RuntimeCutoffs &cutoffs, PbctopoCutCandidate &best_cut,
    double &shift_norm2, std::size_t component_beam_width,
    PbctopoTimingStats *timing,
    const PbctopoRefineScoringCallbacks &refine_callbacks,
    AnchorBackend anchor_backend = AnchorBackend::Auto,
    const std::vector<Int3> *warm_component_offsets = nullptr,
    PbctopoWarmState *warm_state = nullptr,
    const PbctopoCutCandidate *warm_reference_candidate = nullptr,
    std::vector<Int3> *tree_offsets_out = nullptr,
    std::vector<Int3> *refined_offsets_out = nullptr,
    std::vector<PbctopoGraphEdge> *tree_edges_out = nullptr,
    PbctopoCycleSyncMetrics *cycle_metrics_out = nullptr,
    const PbctopoAnchorGraphContext *anchor_context = nullptr,
    const std::vector<PbctopoRequiredContact> *required_contacts = nullptr,
    PbctopoIdentifiability *identifiability_out = nullptr);

} // namespace titan_pbctopo

#endif
