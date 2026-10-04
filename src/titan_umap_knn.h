#ifndef TITAN_UMAP_KNN_H
#define TITAN_UMAP_KNN_H

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

class TitanExecutor;

namespace TitanUMAPKNN {

using MatrixRM =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixRMConstRef = Eigen::Ref<const MatrixRM>;

struct Neighbor {
  int index = -1;
  double distance = 0.0;
  double membership = 0.0;
};

enum class Mode {
  Exact,
  Approx,
  Auto,
};

struct BuildConfig {
  Mode mode = Mode::Exact;
  unsigned int seed = 42;
  bool benchmark = false;
  int auto_exact_threshold = 5000;
  int approx_iterations = 8;
  int approx_seed_neighbors = 24;
  int approx_extra_random_candidates = 8;
  int n_threads = 1;
  TitanExecutor *executor = nullptr;
};

struct BuildStats {
  Mode requested_mode = Mode::Exact;
  Mode selected_mode = Mode::Exact;
  double selected_elapsed_ms = 0.0;
  double exact_elapsed_ms = 0.0;
  double approx_elapsed_ms = 0.0;
  double benchmark_recall = 0.0;
  std::uint64_t selected_distance_evals = 0;
  std::uint64_t exact_distance_evals = 0;
  std::uint64_t approx_distance_evals = 0;
  int approx_iterations_completed = 0;
  bool benchmark_has_baseline = false;
};

struct BuildResult {
  std::vector<std::vector<Neighbor>> knn;
  BuildStats stats;
};

BuildResult build_knn_graph(const MatrixRMConstRef &input, int neighbor_count,
                            const BuildConfig &cfg);

const char *mode_name(Mode mode);

} // namespace TitanUMAPKNN

#endif // TITAN_UMAP_KNN_H
