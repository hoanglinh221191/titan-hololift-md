#ifndef TITAN_KINETICS_H
#define TITAN_KINETICS_H

#include "titan_ui.h"
#include <Eigen/Dense>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

using MatrixXdAnyConstRef =
    Eigen::Ref<const Eigen::MatrixXd, 0,
               Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

enum class TicaBackend {
  Auto,
  ExactDense,
  ExactDual,
  Randomized,
  ExactStreaming
};

std::string tica_backend_to_string(TicaBackend backend);
bool tica_backend_from_string(const std::string &text, TicaBackend &backend);

// ============================================================================
// TitanKinetics — Time-lagged Independent Component Analysis (TICA)
//
// Backends:
//   - exact_dense : original D×D accumulator for small systems
//   - exact_dual  : in-memory feature cache + exact thin-SVD reduction
//   - randomized  : file-backed feature spool + randomized PCA + reduced TICA
// ============================================================================
class TitanKinetics {
public:
  using FrameMatrixRM =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ProjectionSink =
      std::function<void(int64_t, const Eigen::Ref<const Eigen::VectorXd> &)>;

  struct TimingStats {
    double feature_build_seconds = 0.0;
    double cache_write_seconds = 0.0;
    double cache_read_seconds = 0.0;
    double spool_write_seconds = 0.0;
    double spool_read_seconds = 0.0;
    double cache_promotion_seconds = 0.0;
    double exact_dual_center_seconds = 0.0;
    double exact_dual_svd_seconds = 0.0;
    double randomized_sketch_seconds = 0.0;
    double randomized_qr_seconds = 0.0;
    double randomized_compression_seconds = 0.0;
    double randomized_svd_seconds = 0.0;
    double reduced_solve_replay_seconds = 0.0;
    double reduced_eigensolve_seconds = 0.0;
    double projection_replay_seconds = 0.0;

    int64_t cache_write_frames = 0;
    int64_t spool_write_frames = 0;
    int64_t promoted_cache_frames = 0;
    int64_t randomized_sketch_read_frames = 0;
    int64_t randomized_compression_read_frames = 0;
    int64_t reduced_solve_replay_frames = 0;
    int64_t projection_replay_frames = 0;
    int64_t raw_feature_cache_released_frames = 0;

    uint64_t cache_write_bytes = 0;
    uint64_t spool_write_bytes = 0;
    uint64_t spool_read_bytes = 0;

    int exact_dual_svd_calls = 0;
    int randomized_svd_calls = 0;
    int reduced_solve_replay_calls = 0;
    int projection_replay_calls = 0;
  };

  // ---- Construction --------------------------------------------------------
  // feature_dim : dimensionality D of the feature vector (e.g. 3*N_atoms)
  // lag_time    : tau in frames for the time-lagged correlation
  // n_components: number of slow TICA components to retain
  // dt_ns       : physical timestep between frames in nanoseconds
  TitanKinetics(int feature_dim, int lag_time, int n_components, double dt_ns,
                TicaBackend backend = TicaBackend::Auto,
                int randomized_rank = 24,
                size_t hard_memory_cap_bytes = 0,
                bool enforce_hard_memory_cap = false);
  ~TitanKinetics();

  // ---- Hot-loop methods (called once per frame) ----------------------------
  // Push a single feature vector and update all accumulators.
  void accumulate_frame(const Eigen::VectorXd &x);

  // Flush the lag buffer at trajectory boundaries to prevent cross-trajectory
  // lag pairs. The global accumulators are NOT reset. Large-system backends
  // also record the boundary so pair construction matches the trajectory.
  void reset_lag_buffer();

  // ---- Post-loop methods ---------------------------------------------------
  // Build covariance matrices, solve generalised eigenvalue problem,
  // sort eigenvectors by descending eigenvalue, compute implied timescales.
  void solve_tica();

  // Project a single frame onto the TICA subspace.
  // Returns an n_components-dimensional vector.
  Eigen::VectorXd project_frame(const Eigen::VectorXd &x) const;

  // Write eigenvalues + implied timescales and eigenvectors to disk.
  void write_results(const std::string &eval_file, const std::string &evec_file,
                     std::fstream &log,
                     bool kinetics_interpretable = true) const;

  // Write TICA projections from a PRE-PROJECTED matrix (rows = frames, cols =
  // PCs). Use with msm_analyzer->proj_mat_ref() which already holds TICA-space
  // coordinates. n_actual_frames: real row count (proj_mat may be
  // over-allocated with doubling). Each row [0..n_actual_frames) is written
  // as-is — no re-projection applied.
  void write_projections(const MatrixXdAnyConstRef &proj_mat,
                         int64_t n_actual_frames, const std::string &proj_file,
                         std::fstream &log) const;

  // Materialize all cached frames in solved TICA space. For randomized
  // backends, this streams from the spool file instead of keeping the raw
  // feature matrix in RAM.
  FrameMatrixRM project_all_frames() const;

  // Stream all cached frames in solved TICA space to a caller-provided sink.
  // This avoids allocating a second n_frames x n_components matrix when the
  // caller already owns the destination projection cache.
  void project_all_frames_to_sink(const ProjectionSink &sink) const;

  // Engine-side feature construction happens before accumulate_frame(), so the
  // trajectory loop reports that timing explicitly through this hook.
  void add_feature_build_seconds(double seconds);

  // Release exact_dual raw feature rows after a reduced projection cache has
  // been built. Solved eigenvectors and project_frame() remain valid.
  void release_raw_feature_cache();

  // ---- Accessors -----------------------------------------------------------
  int feature_dim() const { return dim_; }
  int lag_time() const { return lag_; }
  int n_components() const { return n_comp_; }
  int64_t num_frames() const { return n_frames_; }
  int64_t num_pairs() const { return n_pairs_; }
  bool is_solved() const { return solved_; }
  int retained_components() const { return retained_components_; }
  int randomized_rank() const { return randomized_rank_; }
  TicaBackend backend() const { return backend_; }
  TicaBackend requested_backend() const { return requested_backend_; }
  int64_t raw_feature_cache_rows() const { return frame_cache_.rows(); }
  const TimingStats &timing_stats() const { return timing_stats_; }
  const std::filesystem::path &spool_path() const { return spool_path_; }

  const Eigen::VectorXd &eigenvalues() const { return eigenvalues_; }
  const Eigen::MatrixXd &eigenvectors() const { return eigenvectors_; }
  const Eigen::VectorXd &implied_timescales() const { return timescales_; }
  const Eigen::VectorXd &mean() const { return mean_; }

private:
  // Dimensions
  int dim_;
  int lag_;
  int n_comp_;
  double dt_ns_;
  TicaBackend requested_backend_;
  TicaBackend backend_;
  int randomized_rank_;
  int retained_components_ = 0;
  size_t hard_memory_cap_bytes_ = 0;
  bool enforce_hard_memory_cap_ = false;

  // Online accumulators shared by all backends
  Eigen::VectorXd sum_x_;        // (D×1) sum of features at time t
  Eigen::VectorXd sum_x_lagged_; // (D×1) sum of features at time t-tau
  Eigen::VectorXd sum_x_paired_; // (D×1) sum of x_t for paired frames only
  Eigen::MatrixXd sum_xx_;       // dense-only: sum of x_t * x_t^T
  Eigen::MatrixXd sum_xtau_;     // dense-only: sum of x_t * x_{t-tau}^T
  int64_t n_frames_;
  int64_t n_pairs_;

  // Lag buffer — row-major so lagged-row access stays contiguous.
  FrameMatrixRM circ_buf_;
  int circ_head_ = 0;
  int circ_count_ = 0;

  // Large-system backends
  FrameMatrixRM frame_cache_;              // exact_dual feature cache
  std::vector<int64_t> traj_break_indices_;
  std::filesystem::path spool_path_;
  std::ofstream spool_writer_;
  mutable TimingStats timing_stats_;

  // Results (populated by solve_tica)
  bool solved_;
  Eigen::VectorXd mean_;
  Eigen::VectorXd eigenvalues_;
  Eigen::MatrixXd eigenvectors_; // (D × requested_components), zero-padded
  Eigen::VectorXd timescales_;

  void append_cached_frame_(const Eigen::VectorXd &x);
  void maybe_promote_cache_to_randomized_();
  void open_spool_for_write_();
  void close_spool_writer_();
  void remove_spool_file_();

  std::vector<int64_t> build_segment_boundaries_() const;

  void solve_exact_dense_();
  void solve_exact_dual_();
  void solve_randomized_();
  void solve_reduced_from_basis_(const Eigen::MatrixXd &basis,
                                 const std::string &label);

  Eigen::MatrixXd build_exact_dual_basis_();
  Eigen::MatrixXd build_randomized_basis_();
};

#endif // TITAN_KINETICS_H
