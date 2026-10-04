#ifndef TITAN_PBCTOPO_CANDIDATE_SCORING_H
#define TITAN_PBCTOPO_CANDIDATE_SCORING_H

#include "report.h"
#include "types.h"

#include <string>

namespace titan_pbctopo {

enum class PbctopoComponentCandidatePrefixStage {
  Continuity,
  Anchor,
  Principal,
  ComponentMetrics,
  Final
};

struct PbctopoLexicographicGap {
  std::string metric = "equivalent";
  double gap = 0.0;
};

int compare_pbctopo_double_metric(double lhs, double rhs,
                                  double eps = 1.0e-6);
int compare_principal_rg2(double lhs, double rhs);
int compare_shape_kappa(double lhs, double rhs);
int compare_shape_moment(double lhs, double rhs);

PbctopoScoreVector make_score_vector(const PbctopoCutCandidate &candidate);
PbctopoCutCandidate component_candidate_to_cut_candidate(
    const PbctopoComponentCandidate &candidate);

bool candidate_anchor_prefix_equivalent(const PbctopoCutCandidate &lhs,
                                        const PbctopoCutCandidate &rhs);
bool component_candidate_is_outlier(const PbctopoCutCandidate &candidate,
                                    const PbctopoCutCandidate *reference);
const char *candidate_skip_search_reject_reason(
    const PbctopoCutCandidate &candidate,
    const PbctopoCutCandidate &reference);
bool candidate_close_enough_to_skip_search(
    const PbctopoCutCandidate &candidate,
    const PbctopoCutCandidate &reference);
bool candidate_has_projection_clash_margin(
    const PbctopoCutCandidate &candidate, double clash_cutoff);

bool better_stage1_cut_candidate(const PbctopoCutCandidate &lhs,
                                 const PbctopoCutCandidate &rhs);
bool better_preclash_cut_candidate(const PbctopoCutCandidate &lhs,
                                   const PbctopoCutCandidate &rhs);
bool better_cut_candidate(const PbctopoCutCandidate &lhs,
                          const PbctopoCutCandidate &rhs);
bool better_component_candidate(const PbctopoComponentCandidate &lhs,
                                const PbctopoComponentCandidate &rhs);

PbctopoLexicographicGap candidate_lexicographic_gap(
    const PbctopoCutCandidate &best, const PbctopoCutCandidate &second);
double candidate_temporal_emission_cost(
    const PbctopoCutCandidate &frame_best,
    const PbctopoCutCandidate &candidate);
bool preclash_cut_candidate_cannot_beat_incumbent(
    const PbctopoCutCandidate &candidate,
    const PbctopoCutCandidate *incumbent);
bool cut_candidate_cannot_beat_incumbent(
    const PbctopoCutCandidate &partial,
    const PbctopoCutCandidate *incumbent);
int compare_candidate_prefix_before_edge_stretch(
    const PbctopoCutCandidate &lhs, const PbctopoCutCandidate &rhs,
    PbctopoRefineScorePhase phase);
bool component_candidate_cannot_beat_incumbent_prefix(
    const PbctopoComponentCandidate &candidate,
    const PbctopoComponentCandidate *incumbent,
    PbctopoComponentCandidatePrefixStage stage);

} // namespace titan_pbctopo

#endif
