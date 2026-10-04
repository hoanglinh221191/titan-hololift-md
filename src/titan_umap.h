#ifndef TITAN_UMAP_H
#define TITAN_UMAP_H

#include "titan_umap_knn.h"
#include "titan_kinetics.h"
#include "titan_ui.h"

#include <Eigen/Dense>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

class TitanExecutor;

class TitanUMAP {
public:
  using InputMatrixRM = TitanUMAPKNN::MatrixRM;
  using EmbeddingMatrixRM = InputMatrixRM;
  using KnnMode = TitanUMAPKNN::Mode;

  struct Config {
    int n_neighbors = 15;
    int n_components = 2;
    int n_epochs = 200;
    double min_dist = 0.1;
    double spread = 1.0;
    unsigned int seed = 42;
    KnnMode knn_mode = KnnMode::Exact;
    bool benchmark_knn = false;
    int knn_auto_exact_threshold = 5000;
    int knn_approx_iterations = 8;
    int knn_approx_seed_neighbors = 24;
    int knn_approx_extra_random_candidates = 8;
    int n_threads = 1;
    TitanExecutor *executor = nullptr;
    bool verbose = false;
  };

  explicit TitanUMAP(const Config &cfg);

  void fit_transform(const MatrixXdAnyConstRef &proj_mat, int64_t n_actual_frames);

  void write_embedding(const std::string &out_file, std::fstream &log) const;

  bool is_fitted() const { return fitted_; }
  const EmbeddingMatrixRM &embedding() const { return embedding_; }

private:
  using Neighbor = TitanUMAPKNN::Neighbor;

  struct Edge {
    int head = -1;
    int tail = -1;
    double weight = 0.0;
  };

  Config cfg_;
  bool fitted_ = false;
  EmbeddingMatrixRM embedding_;

  static double compute_membership_sum(const std::vector<Neighbor> &neighbors,
                                       double rho, double sigma);

  void compute_smooth_knn_memberships(std::vector<std::vector<Neighbor>> &knn) const;
  std::vector<Edge>
  build_fuzzy_simplicial_edges(const std::vector<std::vector<Neighbor>> &knn) const;
  std::pair<double, double> fit_ab_params() const;
  void initialize_embedding(const MatrixXdAnyConstRef &input, std::mt19937 &rng);
  void optimize_layout(const std::vector<Edge> &edges, double a, double b,
                       std::mt19937 &rng);
};

#endif // TITAN_UMAP_H
