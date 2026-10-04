#ifndef DCCM_NETWORK_H
#define DCCM_NETWORK_H

#include "Multi_MOL.h"
#include "aligned_allocator.h"
#include "arguments.h"
#include "titan_sparse_graph.h"
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

class TitanExecutor;

class DCCMNetwork {
public:
  using TrajectoryRow = AlignedDoubleVector;
  using TrajectoryRows = std::vector<TrajectoryRow>;

  DCCMNetwork() = default;
  ~DCCMNetwork() = default;

  void init(const Multi_MOL &pro_in, const Multi_MOL &lig,
            const arguments &args);

  void process_frame(const Multi_MOL &pro_in, const Multi_MOL &lig,
                     size_t chunk_snap_idx, size_t global_frame_idx,
                     double time_ps = 0.0, int replica_id = 0,
                     double box_volume_nm3 =
                         std::numeric_limits<double>::quiet_NaN());

  void calculate_and_export(TitanExecutor &executor, const info &info_output);

  struct CommunityConfig {
    bool enabled{false};
    std::string method{};
  };

  struct StoredEdge {
    int source{-1};
    int target{-1};
    double raw_score{0.0};
    double weight{0.0};
  };

  struct CommunityResult {
    std::vector<int> membership;
    std::vector<int> community_sizes;
    double modularity{0.0};
    bool has_result{false};
  };

  struct AllosteryConfig {
    int tau_lag = 5;
    int num_bins = -1;
    int knn = 0;
    double cutoff_ang = 0.0;
    bool use_avg_cutoff = true;
    bool knn_use_contact_prior = false;
    double knn_contact_cutoff = 8.0;
    bool compute_te = true;
    std::string estimator{"hist"};
    double kde_bandwidth = 0.0;
    bool force_sparse_candidates = false;
  };

  void calculate_allostery_information(TitanExecutor &executor, int tau_lag,
                                       int num_bins,
                                       const std::string &gc_filename =
                                           "network_gc_edges.csv",
                                       const std::string &mi_filename =
                                           "raw_mi_edges.csv",
                                       const std::string &te_filename =
                                           "network_te_edges.csv");
  void calculate_allostery_information(TitanExecutor &executor,
                                       const AllosteryConfig &cfg,
                                       const std::string &gc_filename =
                                           "network_gc_edges.csv",
                                       const std::string &mi_filename =
                                           "raw_mi_edges.csv",
                                       const std::string &te_filename =
                                           "network_te_edges.csv");
  void calculate_allostery_information_optimized(TitanExecutor &executor,
                                                  int tau_lag, int num_bins,
                                                  const std::string &gc_filename =
                                                      "network_gc_edges.csv",
                                                  const std::string &mi_filename =
                                                      "raw_mi_edges.csv",
                                                  const std::string &te_filename =
                                                      "network_te_edges.csv");
  void calculate_allostery_information_optimized(TitanExecutor &executor,
                                                  const AllosteryConfig &cfg,
                                                  const std::string &gc_filename =
                                                      "network_gc_edges.csv",
                                                  const std::string &mi_filename =
                                                      "raw_mi_edges.csv",
                                                  const std::string &te_filename =
                                                      "network_te_edges.csv");
  size_t get_last_allostery_candidate_pairs() const {
    return last_allostery_candidate_pairs_;
  }
  size_t get_last_allostery_dense_pairs() const {
    return last_allostery_dense_pairs_;
  }

  struct LSPMDConfig {
    double cutoff_ang = 12.0;
    bool use_avg_cutoff = true;
  };

  struct LSPMDResult {
    std::vector<double> dc;
    std::vector<double> bc;
    std::vector<std::tuple<int, int, double>> edges;
  };

  void calculate_lspmd(TitanExecutor &executor, const LSPMDConfig &cfg,
                       LSPMDResult &result);
  void export_lspmd(const LSPMDResult &result, double weight_threshold = 0.0);

  const std::vector<std::string> &get_resnames() const { return ca_resname; }
  const std::vector<int> &get_resnrs() const { return ca_resnrs; }
  const std::vector<std::string> &get_chains() const { return ca_chain; }
  const std::vector<int> &get_is_lig() const { return ca_is_lig; }
  const std::vector<int> &get_ca_indices() const { return ca_indices; }

  size_t get_total_frames() const { return total_frames; }
  const std::vector<int64_t> &get_global_frame_indices() const {
    return frame_global_indices_;
  }
  const std::vector<double> &get_traj_times() const { return traj_times_; }
  const std::vector<int> &get_traj_replicas() const { return traj_replicas_; }
  const std::vector<double> &get_traj_box_volumes_nm3() const {
    return traj_box_volumes_nm3_;
  }
  const TrajectoryRows &get_traj_x() const { return traj_x; }
  const TrajectoryRows &get_traj_y() const { return traj_y; }
  const TrajectoryRows &get_traj_z() const { return traj_z; }
  const TrajectoryRow &get_latest_traj_x() const {
    return store_full_traj_ ? traj_x.back() : latest_traj_x_;
  }
  const TrajectoryRow &get_latest_traj_y() const {
    return store_full_traj_ ? traj_y.back() : latest_traj_y_;
  }
  const TrajectoryRow &get_latest_traj_z() const {
    return store_full_traj_ ? traj_z.back() : latest_traj_z_;
  }
  bool stores_full_trajectory() const { return store_full_traj_; }

  const std::vector<int> &get_cb_indices() const { return cb_indices; }
  const TrajectoryRows &get_traj_cbx() const { return traj_cbx; }
  const TrajectoryRows &get_traj_cby() const { return traj_cby; }
  const TrajectoryRows &get_traj_cbz() const { return traj_cbz; }
  const std::vector<StoredEdge> &get_last_network_graph_edges() const {
    return last_network_graph_edges_;
  }
  const std::vector<StoredEdge> &get_last_allostery_gc_edges() const {
    return last_allostery_gc_edges_;
  }
  const std::vector<StoredEdge> &get_last_allostery_te_edges() const {
    return last_allostery_te_edges_;
  }
  void set_last_allostery_graph_edges(std::vector<StoredEdge> gc_edges,
                                      std::vector<StoredEdge> te_edges) {
    last_allostery_gc_edges_ = std::move(gc_edges);
    last_allostery_te_edges_ = std::move(te_edges);
  }
  const CommunityResult &get_last_community_result() const {
    return last_community_result_;
  }
  void export_community_outputs_from_graph(
      const titan_sparse::WeightedCSRGraph<double> &graph,
      const std::vector<int> &membership, const info &info_output);
  bool dense_postproc_skipped_due_memory() const {
    return dense_postproc_skipped_due_memory_;
  }
  bool network_sparse_mode_active() const { return network_sparse_mode_active_; }

  struct NetworkMemoryConfig {
    std::string mode{"auto"};
    double mem_budget_gb{-1.0};
    bool low_memory{false};
    double sparse_cut_ang{30.0};
    int sparse_k{24};
    std::string cache_precision{"auto"};
  };

private:
  std::vector<int> ca_indices;
  std::vector<int> ca_resnrs;
  std::vector<int> ca_is_lig;
  std::vector<std::string> ca_chain;
  std::vector<std::string> ca_resname;
  std::vector<int> cb_indices;
  bool store_full_traj_ = true;
  bool store_cb_traj_ = false;
  bool has_first_snapshot_frame_ = false;

  TrajectoryRows traj_x;
  TrajectoryRows traj_y;
  TrajectoryRows traj_z;
  TrajectoryRows traj_cbx;
  TrajectoryRows traj_cby;
  TrajectoryRows traj_cbz;

  size_t total_frames = 0;
  std::vector<int64_t> frame_global_indices_;
  std::vector<double> traj_times_;
  std::vector<int> traj_replicas_;
  std::vector<double> traj_box_volumes_nm3_;
  TrajectoryRow latest_traj_x_;
  TrajectoryRow latest_traj_y_;
  TrajectoryRow latest_traj_z_;
  bool kabsch_aligned_ = false;
  bool dense_postproc_skipped_due_memory_ = false;
  bool network_sparse_mode_active_ = false;
  size_t last_allostery_candidate_pairs_ = 0;
  size_t last_allostery_dense_pairs_ = 0;
  std::vector<StoredEdge> last_network_graph_edges_;
  std::vector<StoredEdge> last_allostery_gc_edges_;
  std::vector<StoredEdge> last_allostery_te_edges_;
  NetworkMemoryConfig network_memory_cfg_;
  CommunityConfig community_cfg_;
  UncertaintyArgs uncertainty_cfg_;
  CommunityResult last_community_result_;
  MOL first_pro_frame_;
  MOL first_lig_frame_;
  std::vector<mol_info> first_pro_info_;
  std::vector<mol_info> first_lig_info_;

  static int find_cb_for_residue(const Multi_MOL &mol, int resnr,
                                 const std::string &chain, bool is_gly);
};

#endif // DCCM_NETWORK_H
