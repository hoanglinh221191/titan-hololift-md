#ifndef PATHWAY_DE_VARIANTS_H
#define PATHWAY_DE_VARIANTS_H

#include "pull_de_variants.h"
#include <functional>
#include <random>
#include <span>
#include <vector>

enum class PathwayMode { Advance, Recovery };

struct PathwayCandidateState {
  double progress{0.0};
  double requested_delta_progress{0.0};
  double delta_progress{0.0};
  double applied_progress{0.0};
  double applied_path_distance{std::numeric_limits<double>::infinity()};
  double requested_pull_distance{0.0};
  double pull_distance{0.0};
  double probe_length{0.0};
  bool fixed_probe{false};
  bool navigation_enabled{false};
  bool guided_seed{false};
  std::size_t guided_seed_index{0};
  double applied_corridor_radius{0.0};
  double navigation_tube_excess{0.0};
  double navigation_backtrack_distance{0.0};
  double navigation_tube_cost{0.0};
  double navigation_backtrack_cost{0.0};
  double navigation_cost{0.0};
  force requested_vector{};
  double requested_angle_deg{0.0};
  bool turn_capped{false};
  double corridor_offset_u{0.0};
  double corridor_offset_v{0.0};
  double corridor_radius{0.0};
  double minimum_clearance{0.0};
  double endpoint_minimum_clearance{0.0};
  double swept_minimum_clearance{0.0};
  double swept_minimum_fraction{1.0};
  std::size_t swept_minimum_receptor_index{0};
  std::size_t swept_minimum_ligand_index{0};
  bool swept_minimum_complete{false};
  cache_energy energy{};
  double raw_score{std::numeric_limits<double>::infinity()};
  double energy_key{std::numeric_limits<double>::infinity()};
  double angle_deg{180.0};
  force vector{};
  bool valid{false};
};

struct PathwayPatternStats {
  bool enabled{false};
  bool improved{false};
  bool stopped_by_eval_budget{false};
  std::size_t sweeps{0};
  std::size_t accepted_moves{0};
  std::size_t evaluations_used{0};
  std::size_t seeds_attempted{0};
  std::size_t distinct_seed_basins{0};
  double initial_score{std::numeric_limits<double>::infinity()};
  double final_score{std::numeric_limits<double>::infinity()};
  double wall_seconds{0.0};
};

struct PathwayLegacyDeStats {
  bool enabled{false};
  bool stopped_by_eval_budget{false};
  bool stopped_by_stagnation{false};
  std::size_t generations_completed{0};
  std::size_t best_generation{0};
};

using PathwayBatchEvaluator = std::function<PullDeBatchEvaluation(
    std::span<const double> proposals,
    std::span<PathwayCandidateState> evaluated_candidates)>;

struct PathwayProgressDomain {
  double min_progress{0.0};
  double max_progress{0.0};
  std::function<double(std::mt19937 &)> sample{};
  std::function<double(double)> sanitize{};
};

bool run_pathway_legacy_de(const de_arg &arguments,
                           const PathwayProgressDomain &domain,
                           PathwayMode mode,
                           const PathwayBatchEvaluator &evaluator,
                           std::vector<PathwayCandidateState> &population,
                           PathwayCandidateState &best_candidate,
                           PathwayLegacyDeStats &stats);

bool run_pathway_de_variant(const PullDeVariantOptions &options,
                            const PathwayProgressDomain &domain,
                            PathwayMode mode,
                            const PathwayBatchEvaluator &evaluator,
                            std::vector<PathwayCandidateState> &population,
                            PathwayCandidateState &best_candidate,
                            PullDeVariantStats &stats);

double pathway_variant_pattern_initial_step(const PullDeVariantStats &stats,
                                            double progress_window);

std::vector<std::size_t> select_pathway_pattern_seed_indices(
    const std::vector<PathwayCandidateState> &population, std::size_t max_seeds,
    double min_progress_gap, PathwayMode mode);

double pathway_energy_key(double raw_score);
bool pathway_candidate_better(const PathwayCandidateState &lhs,
                              std::size_t lhs_index,
                              const PathwayCandidateState &rhs,
                              std::size_t rhs_index, PathwayMode mode);
// Exact lexicographic priority plus stable identity for std::sort/stable_sort.
// Keep tolerance-based acceptance/equivalence in the decision functions above
// and below; pairwise epsilon ties do not define a strict weak ordering.
bool pathway_candidate_strict_less(const PathwayCandidateState &lhs,
                                   std::size_t lhs_index,
                                   const PathwayCandidateState &rhs,
                                   std::size_t rhs_index, PathwayMode mode);
bool pathway_candidate_equivalent(const PathwayCandidateState &lhs,
                                  const PathwayCandidateState &rhs,
                                  PathwayMode mode);

#endif
