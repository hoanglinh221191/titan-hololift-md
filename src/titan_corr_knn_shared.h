#ifndef TITAN_CORR_KNN_SHARED_H
#define TITAN_CORR_KNN_SHARED_H

#include "aligned_allocator.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace titan_corr_knn {

using TrajectoryRow = AlignedDoubleVector;
using TrajectoryRows = std::vector<TrajectoryRow>;

struct PairPruningConfig {
  int pair_topk{0};
  double pair_cutoff_ang{0.0};
  bool use_avg_cutoff{true};
  bool pair_use_contact_prior{false};
  double pair_contact_cutoff_ang{8.0};
};

std::string build_node_label(const std::vector<std::string> &resnames,
                             const std::vector<int> &resnrs,
                             const std::vector<std::string> &chains,
                             std::size_t index);

std::vector<std::string> build_node_labels(
    const std::vector<std::string> &resnames, const std::vector<int> &resnrs,
    const std::vector<std::string> &chains);

std::vector<std::pair<std::size_t, std::size_t>> build_candidate_pairs(
    const TrajectoryRows &traj_x, const TrajectoryRows &traj_y,
    const TrajectoryRows &traj_z, const TrajectoryRows &anchor_x,
    const TrajectoryRows &anchor_y, const TrajectoryRows &anchor_z,
    std::size_t node_count, std::size_t total_frames,
    const PairPruningConfig &cfg, std::size_t *dense_pair_count_out = nullptr);

} // namespace titan_corr_knn

#endif // TITAN_CORR_KNN_SHARED_H
