#pragma once

#include "titan_sparse_graph.h"
#include "titan_ui.h"

#include <Eigen/Dense>

#include <fstream>
#include <string>
#include <vector>

class TitanExecutor;

class TitanTPT {
public:
  struct EdgeFlux {
    int from{-1};
    int to{-1};
    double gross_flux{0.0};
    double net_flux{0.0};
  };

  struct Pathway {
    std::vector<int> nodes;
    double flux{0.0};
    double fraction{0.0};
  };

  struct Summary {
    int n_states{0};
    int n_source{0};
    int n_sink{0};
    int n_intermediate{0};
    double source_population{0.0};
    double sink_population{0.0};
    double intermediate_population{0.0};
    double total_reactive_flux{0.0};
    double max_net_flux{0.0};
    double rate_source_to_sink_per_lag{0.0};
    double mfpt_source_to_sink_lags{
        std::numeric_limits<double>::infinity()};
  };

  TitanTPT(const Eigen::MatrixXd &tpm, const Eigen::VectorXd &stationary_dist);

  void compute(const std::vector<int> &source_states,
               const std::vector<int> &sink_states, int top_paths = 5,
               double flux_cut = 0.0, TitanExecutor *executor = nullptr,
               int n_threads = 1);

  bool computed() const { return computed_; }
  int n_states() const { return n_states_; }

  const Eigen::VectorXd &forward_committor() const { return q_forward_; }
  const Eigen::VectorXd &backward_committor() const { return q_backward_; }
  const Eigen::VectorXd &reactive_density() const { return reactive_density_; }
  const Eigen::VectorXd &stationary_dist() const { return stationary_dist_; }
  const Eigen::VectorXd &gross_out_flux() const { return gross_out_flux_; }
  const Eigen::VectorXd &gross_in_flux() const { return gross_in_flux_; }
  const Eigen::VectorXd &net_out_flux() const { return net_out_flux_; }
  const Eigen::VectorXd &net_in_flux() const { return net_in_flux_; }
  const std::vector<EdgeFlux> &edges() const { return edges_; }
  const std::vector<Pathway> &paths() const { return paths_; }
  const Summary &summary() const { return summary_; }
  const std::vector<int> &source_states() const { return source_states_; }
  const std::vector<int> &sink_states() const { return sink_states_; }

  void write_results(const std::string &summary_file,
                     const std::string &committor_file,
                     const std::string &flux_edges_file,
                     const std::string &paths_file, std::fstream &log,
                     const std::string &mode_label,
                     bool dham_requested,
                     const std::string &source_label = {},
                     const std::string &sink_label = {},
                     const std::string &model_source_label = {}) const;

  static bool parse_state_spec(const std::string &spec, int n_states,
                               std::vector<int> &states_out,
                               std::string *error_message = nullptr);
  static std::string format_state_list(const std::vector<int> &states);

private:
  void build_sparse_transition_backend(const Eigen::MatrixXd &transition_matrix);
  void validate_basins(const std::vector<int> &source_states,
                       const std::vector<int> &sink_states) const;
  void build_membership_masks();
  void solve_committor_system(const titan_sparse::WeightedCSRGraph<double> &graph,
                              const std::vector<unsigned char> &one_boundary,
                              const std::vector<unsigned char> &zero_boundary,
                              Eigen::VectorXd &out_committor) const;
  void compute_reactive_densities();
  void compute_flux_matrices(TitanExecutor *executor, int n_threads,
                             double flux_cut);
  void extract_dominant_paths(int top_paths, double flux_cut);

  static std::string trim_copy(const std::string &value);

  int n_states_{0};
  Eigen::VectorXd stationary_dist_;
  titan_sparse::WeightedCSRGraph<double> tpm_graph_;
  titan_sparse::WeightedCSRGraph<double> reverse_graph_;
  std::vector<int> reverse_edge_lookup_;

  std::vector<int> source_states_;
  std::vector<int> sink_states_;
  std::vector<unsigned char> is_source_;
  std::vector<unsigned char> is_sink_;

  Eigen::VectorXd q_forward_;
  Eigen::VectorXd q_backward_;
  Eigen::VectorXd reactive_density_;
  Eigen::VectorXd gross_out_flux_;
  Eigen::VectorXd gross_in_flux_;
  Eigen::VectorXd net_out_flux_;
  Eigen::VectorXd net_in_flux_;

  std::vector<EdgeFlux> edges_;
  std::vector<Pathway> paths_;
  Summary summary_;
  bool computed_{false};
};
