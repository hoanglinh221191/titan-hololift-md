#ifndef TITAN_LJ_H
#define TITAN_LJ_H

#include "Multi_MOL.h"
#include "arguments.h"
#include "titan_neighbor_grid.h"
#include "titan_sparse_graph.h"

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

class TitanExecutor;

class TitanLJAnalyzer {
public:
  static constexpr std::size_t kFeatureShellCount = 3;
  static constexpr int kTicaFeatureDim = 20;

  struct FrameResult {
    int frame_index{0};
    double time_ps{0.0};
    double volume_a3{0.0};
    double rho_bulk_a3{0.0};
    double rho_bulk_nm3{0.0};
    int n_clusters{0};
    int largest_cluster_size{0};
    double largest_cluster_fraction{0.0};
    double largest_cluster_rg_a{0.0};
    double largest_cluster_rmax_a{0.0};
    double mean_dense_q4{0.0};
    double mean_dense_q6{0.0};
    double mean_dense_q4_bar{0.0};
    double mean_dense_q6_bar{0.0};
    double solid_like_fraction{0.0};
    int largest_solid_cluster_size{0};
    double largest_solid_cluster_fraction{0.0};
    double fcc_like_fraction{0.0};
    double hcp_like_fraction{0.0};
    double bcc_like_fraction{0.0};
    double ico_like_fraction{0.0};
    double other_solid_fraction{0.0};
  };

  TitanLJAnalyzer() = default;

  void configure(const LjArgs &args);
  void init(const Multi_MOL &mol1, const arguments &args, std::fstream &log);
  void process_frame(const Multi_MOL &mol1, std::size_t snap_idx,
                     TitanExecutor &executor);
  void finalize_trajectory(const std::string &timeseries_filename,
                           const std::string &rho_filename,
                           const std::string &feature_filename,
                           const std::string &summary_filename,
                           std::fstream &log);

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] std::size_t atom_count() const noexcept { return atom_count_; }
  [[nodiscard]] std::size_t current_frame_count() const noexcept {
    return frames_.size();
  }
  [[nodiscard]] bool phase2_ready() const noexcept {
    return initialized_ && atom_count_ > 0;
  }
  [[nodiscard]] int feature_dim() const noexcept {
    return phase2_ready() ? kTicaFeatureDim : 0;
  }
  [[nodiscard]] bool has_feature_vector() const noexcept {
    return latest_feature_vector_.size() == kTicaFeatureDim;
  }
  [[nodiscard]] const Eigen::VectorXd &latest_feature_vector() const noexcept {
    return latest_feature_vector_;
  }

private:
  struct ClusterGeometry {
    double centroid_x{0.0};
    double centroid_y{0.0};
    double centroid_z{0.0};
    double rg_a{0.0};
    double rmax_a{0.0};
  };

  struct LocalOrderSummary {
    double mean_dense_q4{0.0};
    double mean_dense_q6{0.0};
    double mean_dense_q4_bar{0.0};
    double mean_dense_q6_bar{0.0};
    double solid_like_fraction{0.0};
    int largest_solid_cluster_size{0};
    double fcc_like_fraction{0.0};
    double hcp_like_fraction{0.0};
    double bcc_like_fraction{0.0};
    double ico_like_fraction{0.0};
    double other_solid_fraction{0.0};
  };

  static bool has_periodic_box(const box &box_value) noexcept;

  void prepare_atom_buffers(std::size_t atom_count);
  void reset_trajectory_state();
  void ensure_profile_layout(const MOL &frame);
  void reset_union_find(std::size_t atom_count);
  int find_root(int atom_index);
  void unite_atoms(int lhs, int rhs);
  void build_contact_graph(const MOL &frame, std::size_t atom_count,
                           bool periodic);
  int find_largest_cluster_root(std::size_t atom_count,
                                int &n_clusters_out);
  void build_largest_cluster_members(std::size_t atom_count, int largest_root);
  void unwrap_largest_cluster(const MOL &frame, bool periodic);
  ClusterGeometry compute_cluster_geometry(TitanExecutor &executor) const;
  LocalOrderSummary compute_local_order_metrics(const MOL &frame,
                                                std::size_t atom_count,
                                                bool periodic,
                                                TitanExecutor &executor);
  void accumulate_radial_density(const MOL &frame, bool periodic,
                                 const ClusterGeometry &geometry,
                                 std::array<double, kFeatureShellCount>
                                     &shell_all_nm3,
                                 TitanExecutor &executor);
  void export_timeseries(const std::string &filename, std::fstream &log) const;
  void export_rho_profile(const std::string &filename, std::fstream &log) const;
  void export_feature_timeseries(const std::string &filename,
                                 std::fstream &log) const;
  void export_summary(const std::string &filename, std::fstream &log) const;

  bool requested_{false};
  bool enabled_{false};
  bool initialized_{false};

  double cutoff_a_{5.0};
  double cutoff_sq_a_{25.0};
  int min_cluster_size_{2};
  double rho_bin_a_{1.0};
  double rho_max_input_a_{0.0};
  double fixed_rho_max_a_{0.0};
  double solid_q6_threshold_{0.40};
  int solid_min_cn_{6};
  double crystal_tol_{0.08};
  bool profile_layout_ready_{false};

  std::size_t atom_count_{0};
  std::size_t radial_bin_count_{0};

  std::vector<int> parent_;
  std::vector<int> rank_;
  std::vector<int> size_;
  std::vector<int> atom_degree_;
  std::vector<int> root_seen_;
  std::vector<int> component_root_;
  std::vector<int> cluster_members_;
  std::vector<int> atom_to_local_;
  std::vector<int> bfs_queue_;
  std::vector<unsigned char> visited_;
  std::vector<std::pair<int, int>> edges_;
  titan_sparse::CSRTopology cluster_graph_;
  titan_neighbor::HalfShellGrid neighbor_grid_;

  std::vector<double> unwrap_x_;
  std::vector<double> unwrap_y_;
  std::vector<double> unwrap_z_;
  std::vector<double> q4_values_;
  std::vector<double> q6_values_;
  std::vector<unsigned char> solid_like_;

  std::vector<FrameResult> frames_;
  std::vector<std::array<double, kTicaFeatureDim>> feature_history_;
  Eigen::VectorXd latest_feature_vector_;

  std::vector<double> rho_cluster_mean_nm3_;
  std::vector<double> rho_cluster_m2_nm3_;
  std::vector<std::uint64_t> rho_cluster_count_;
  std::vector<double> rho_all_mean_nm3_;
  std::vector<double> rho_all_m2_nm3_;
  std::vector<std::uint64_t> rho_all_count_;
};

#endif // TITAN_LJ_H
