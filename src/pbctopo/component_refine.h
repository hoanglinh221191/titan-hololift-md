#ifndef TITAN_PBCTOPO_COMPONENT_REFINE_H
#define TITAN_PBCTOPO_COMPONENT_REFINE_H

#include "anchor_graph.h"
#include "config.h"
#include "types.h"
#include "../box_type.h"
#include "../titan_gyration.h"

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace titan_pbctopo {

struct PbctopoRefineScoringCallbacks {
  using ScoreCutCandidateFn = PbctopoCutCandidate (*)(
      const std::vector<PbctopoLocalEdge> &, const std::vector<Vec3> &, double,
      double, double, double, const titan_gyration::PointStats &, double,
      const PbctopoStage1EdgeContext *, const PbctopoEdgeScanSoA *,
      const PbctopoCutCandidate *, const PbctopoCutCandidate *,
      PbctopoRefineScorePhase);
  using ScorePrincipalOnlyFn = PbctopoCutCandidate (*)(
      const std::vector<PbctopoLocalEdge> &, const std::vector<Vec3> &, double,
      double, double, double, const titan_gyration::PointStats &, double,
      PbctopoCutScoreBreakdown *, const PbctopoStage1EdgeContext *,
      const PbctopoEdgeScanSoA *, const PbctopoCutCandidate *,
      const PbctopoCutCandidate *);
  using ContinuityMetricsFn = bool (*)(
      const std::vector<Vec3> &, const std::vector<PbctopoContinuityPair> &,
      double, double &, double &, double &, std::size_t &, std::size_t &,
      const PbctopoCutCandidate *);
  using AnalyzeUltraclashFn = void (*)(const std::vector<Vec3> &,
                                       const std::unordered_set<std::uint64_t> &,
                                       PbctopoCutCandidate &, double,
                                       const std::vector<PbctopoAtom> *,
                                       const std::vector<std::size_t> *);
  using BetterCandidateFn = bool (*)(const PbctopoCutCandidate &,
                                     const PbctopoCutCandidate &);
  using CandidateThresholdFn = bool (*)(const PbctopoCutCandidate &,
                                        const PbctopoCutCandidate *);
  using EdgeThresholdFn = bool (*)(const PbctopoCutCandidate &,
                                   const PbctopoCutCandidate *,
                                   PbctopoRefineScorePhase);
  using CandidateCloseEnoughFn = bool (*)(const PbctopoCutCandidate &,
                                          const PbctopoCutCandidate &);

  ScoreCutCandidateFn score_cut_candidate = nullptr;
  ScorePrincipalOnlyFn score_cut_candidate_principal_only = nullptr;
  ContinuityMetricsFn continuity_metrics_ok = nullptr;
  AnalyzeUltraclashFn analyze_cut_candidate_ultraclash = nullptr;
  BetterCandidateFn better_cut_candidate = nullptr;
  BetterCandidateFn better_preclash_cut_candidate = nullptr;
  BetterCandidateFn better_stage1_cut_candidate = nullptr;
  CandidateThresholdFn cut_candidate_cannot_beat_incumbent = nullptr;
  EdgeThresholdFn edge_scan_cannot_beat_threshold = nullptr;
  CandidateCloseEnoughFn candidate_close_enough_to_skip_search = nullptr;
};

void clear_component_trial_hints(PbctopoWarmState &warm_state);
bool component_trial_hints_compatible(const PbctopoWarmState &warm_state,
                                      std::size_t component_count);

std::vector<std::size_t>
build_coarse_anchor_subset(const std::vector<PbctopoAtom> &atoms,
                           const std::vector<std::size_t> &anchor_indices);
std::vector<std::size_t>
build_third_anchor_subset(const std::vector<PbctopoAtom> &atoms,
                          const std::vector<std::size_t> &anchor_indices);
std::vector<std::size_t>
build_tapered_stage1_anchor_subset(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::size_t> &anchor_indices);

std::vector<std::size_t> build_atom_component_index(
    std::size_t atom_count,
    const std::vector<std::vector<std::size_t>> &components);
PbctopoStage1EdgeContext build_stage1_edge_context(
    const std::vector<PbctopoLocalEdge> &local_edges,
    const std::vector<std::size_t> &atom_component_index,
    std::size_t moving_component_idx, const std::vector<Vec3> &placed_points);
PbctopoEdgeScanSoA
build_edge_scan_soa(const std::vector<PbctopoLocalEdge> &local_edges);

void build_component_points_from_offsets(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::vector<std::size_t>> &components, box Box,
    const std::vector<Int3> &offsets, std::vector<Vec3> &placed_points,
    double &shift_norm2);

PbctopoCutCandidate score_component_placed_points_solution(
    const std::vector<Vec3> &placed_points, double shift_norm2,
    const std::vector<PbctopoLocalEdge> &local_edges,
    const std::unordered_set<std::uint64_t> &edge_set, double clash_cutoff,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &anchor_indices, double continuity_cutoff,
    const PbctopoRefineScoringCallbacks &callbacks,
    const PbctopoCutCandidate *incumbent = nullptr,
    PbctopoTimingStats *timing = nullptr, bool run_ultraclash = true,
    PbctopoRefineScorePhase phase = PbctopoRefineScorePhase::Exact,
    const PbctopoStage1EdgeContext *stage1_edge_context = nullptr,
    const PbctopoEdgeScanSoA *edge_scan_soa = nullptr,
    const PbctopoCutCandidate *edge_prune_threshold = nullptr,
    AnchorBackend anchor_backend = AnchorBackend::Auto,
    const titan_gyration::PointStats *precomputed_stats = nullptr,
    const PbctopoAnchorGraphContext *anchor_context = nullptr,
    const std::vector<PbctopoAtom> *clash_atoms = nullptr,
    const std::vector<std::size_t> *atom_component_index = nullptr);

PbctopoCutCandidate score_component_offset_solution(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::vector<std::size_t>> &components, box Box,
    const std::vector<Int3> &offsets,
    const std::vector<PbctopoLocalEdge> &local_edges,
    const std::unordered_set<std::uint64_t> &edge_set, double clash_cutoff,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &anchor_indices, double continuity_cutoff,
    std::vector<Vec3> &placed_points,
    const PbctopoRefineScoringCallbacks &callbacks,
    const PbctopoCutCandidate *incumbent = nullptr,
    PbctopoTimingStats *timing = nullptr, bool run_ultraclash = true,
    PbctopoRefineScorePhase phase = PbctopoRefineScorePhase::Exact,
    const PbctopoStage1EdgeContext *stage1_edge_context = nullptr,
    const PbctopoEdgeScanSoA *edge_scan_soa = nullptr,
    const PbctopoCutCandidate *edge_prune_threshold = nullptr,
    AnchorBackend anchor_backend = AnchorBackend::Auto,
    const PbctopoAnchorGraphContext *anchor_context = nullptr,
    const std::vector<std::size_t> *atom_component_index = nullptr);

PbctopoCutCandidate improve_component_offsets_locally(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::vector<std::size_t>> &components, box Box,
    const std::vector<PbctopoShift> &shifts,
    const std::vector<PbctopoLocalEdge> &local_edges,
    const std::unordered_set<std::uint64_t> &edge_set,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &anchor_indices, std::vector<Int3> &offsets,
    double clash_cutoff, double continuity_cutoff, std::vector<Vec3> &placed_points,
    const PbctopoRefineScoringCallbacks &callbacks,
    AnchorBackend anchor_backend = AnchorBackend::Auto,
    PbctopoTimingStats *timing = nullptr, PbctopoWarmState *warm_state = nullptr,
    const PbctopoCutCandidate *warm_reference_candidate = nullptr,
    const PbctopoAnchorGraphContext *anchor_context = nullptr);

} // namespace titan_pbctopo

#endif
