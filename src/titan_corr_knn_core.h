#ifndef TITAN_CORR_KNN_CORE_H
#define TITAN_CORR_KNN_CORE_H

#include "titan_corr_knn_shared.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

class TitanExecutor;

namespace titan_corr_knn {

struct ExactEstimatorConfig {
  int estimator_k{4};
  int tau_lag{5};
  bool compute_te{true};
};

struct ExactComputationResult {
  std::size_t total_frames{0};
  std::size_t effective_te_samples{0};
  std::size_t node_count{0};
  std::size_t dense_pair_count{0};
  std::size_t candidate_pair_count{0};
  std::size_t degenerate_node_count{0};
  std::size_t degenerate_pair_count{0};
  bool mi_available{false};
  bool te_available{false};
  bool te_requested{true};
  int estimator_k{4};
  int tau_lag{5};
  std::vector<std::string> node_labels;
  std::vector<unsigned char> degenerate_nodes;
  std::vector<std::pair<std::size_t, std::size_t>> candidate_pairs;
  std::vector<double> mi_values;
  std::vector<double> gc_values;
  std::vector<double> te_forward_values;
  std::vector<double> te_reverse_values;
};

ExactComputationResult compute_exact_scores(
    const TrajectoryRows &traj_x, const TrajectoryRows &traj_y,
    const TrajectoryRows &traj_z, const std::vector<std::string> &node_labels,
    const std::vector<std::pair<std::size_t, std::size_t>> &candidate_pairs,
    std::size_t dense_pair_count, const ExactEstimatorConfig &cfg,
    TitanExecutor &executor);

std::string describe_pair_filter(const PairPruningConfig &cfg);

void write_summary(const std::string &path, const ExactComputationResult &result,
                   const PairPruningConfig &pair_cfg,
                   const std::string &context_label);

void write_raw_mi_csv(const std::string &path,
                      const ExactComputationResult &result);
void write_raw_gc_csv(const std::string &path,
                      const ExactComputationResult &result);
void write_raw_te_csv(const std::string &path,
                      const ExactComputationResult &result);

void write_legacy_allostery_exports(const std::string &mi_path,
                                    const std::string &gc_path,
                                    const std::string &te_path,
                                    const ExactComputationResult &result);

} // namespace titan_corr_knn

#endif // TITAN_CORR_KNN_CORE_H
