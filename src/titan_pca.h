#ifndef TITAN_PCA_H
#define TITAN_PCA_H

#include "aligned_allocator.h"
#include "arguments.h"
#include "info.h"
#include "titan_kinetics.h"
#include "titan_ui.h"
#include <chrono>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>

class TitanExecutor;

namespace TitanPCA {

using CoordinateRow = AlignedDoubleVector;
using CoordinateRows = std::vector<CoordinateRow>;

// Structure representing a single grid bin on the 2D Free Energy Surface
struct FES_Grid {
  double pc1{0.0};
  double pc2{0.0};
  int count{0};
  double count_w{0.0};
  double free_energy{20.0};
  int sink_id{-1}; // -1 means unassigned, else indexes a local minimum
};

struct FES2DResult {
  int grid_size{0};
  double min_x{0.0};
  double min_y{0.0};
  double step_x{1.0};
  double step_y{1.0};
  double total_weight{0.0};
  double max_bin_weight{0.0};
  AlignedVector<FES_Grid> bins{};

  FES_Grid &bin(int ix, int iy) {
    return bins[static_cast<size_t>(ix) * static_cast<size_t>(grid_size) +
                static_cast<size_t>(iy)];
  }

  const FES_Grid &bin(int ix, int iy) const {
    return bins[static_cast<size_t>(ix) * static_cast<size_t>(grid_size) +
                static_cast<size_t>(iy)];
  }
};

// Structure representing a macrostate cluster defined by a prominent density
// sink
struct ClusterNode {
  int id;
  double min_pc1;
  double min_pc2;
  double min_energy;
  int representative_frame_idx;
  double min_euclidean_distance;
  int population;
  double population_w;
  double percentage;
  double percentage_w;
};

struct FesPostprocMetrics {
  bool zts_ran{false};
  std::chrono::milliseconds zts_time{0};
};

struct WeightContext {
  std::string source_name{"equilibrium"};
  std::string mode_label{"uniform"};
  double target_temp{300.0};
};

bool export_projection_fes_analysis(
    const std::string &label, const std::string &space_name,
    const arguments &args,
    const std::vector<std::pair<double, double>> &projections,
    const AlignedDoubleVector &analysis_weights, bool use_weights,
    double target_temp, const std::string &weighted_mode_label,
    TitanExecutor &executor, std::fstream &log, const std::string &fes_filename,
    const std::string &cluster_filename,
    const std::vector<double> &traj_times = {},
    const std::vector<int> &traj_replicas = {}, FES2DResult *out_fes = nullptr,
    std::vector<ClusterNode> *out_clusters = nullptr);

// Main API isolated from impl_engine.cpp handling Covariance, FES Histograms &
// Peak Clustering
void execute_pca_fes_analysis(
    const arguments &args, const CoordinateRows &traj_x,
    const CoordinateRows &traj_y, const CoordinateRows &traj_z,
    const std::vector<std::string> &ca_resname,
    const std::vector<int> &ca_resnrs, const std::vector<std::string> &ca_chain,
    const std::vector<int> &ca_is_lig, TitanExecutor &executor, info &info_out,
    std::fstream &log,
    const std::vector<double> &frame_weights = {},
    const WeightContext &weight_context = WeightContext{},
    const std::vector<double> &traj_times = {},
    const std::vector<int> &traj_replicas = {},
    const std::vector<std::string> &replica_traj_files = {},
    FesPostprocMetrics *metrics = nullptr);

void execute_tica_fes_analysis(
    const arguments &args, const MatrixXdAnyConstRef &tica_proj_mat,
    int64_t n_actual_frames, const std::vector<int64_t> &frame_global_indices,
    TitanExecutor &executor, info &info_out, std::fstream &log,
    const std::vector<double> &frame_weights = {},
    const WeightContext &weight_context = WeightContext{},
    FesPostprocMetrics *metrics = nullptr);

} // namespace TitanPCA

#endif
