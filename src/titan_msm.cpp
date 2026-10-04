// ============================================================================
// TitanMSM — Markov State Model (Phase 2) Implementation
// ============================================================================
#include "titan_msm.h"
#include "titan_executor.h"
#include "titan_simd_dispatch.h"

#include "titan_kinetics.h"
#include "titan_markov_estimator.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int kMSMFrameBatch = 16;
constexpr int kMSMCentroidBlock = 8;

inline double squared_distance_rowmajor_avx2(const double *lhs,
                                             const double *rhs,
                                             int dim) noexcept {
  return titan_simd_dispatch::squared_distance_rowmajor(lhs, rhs, dim);
}

inline void prefetch_row_read(const double *ptr) noexcept {
#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__)
  _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
#else
  (void)ptr;
#endif
}

inline void add_row_inplace(double *dst, const double *src, int dim) noexcept {
  titan_simd_dispatch::add_row_inplace(dst, src, dim);
}

inline std::vector<int> build_segment_boundaries(
    const std::vector<int> &traj_break_indices, int n_frames) {
  std::vector<int> boundaries;
  boundaries.reserve(traj_break_indices.size() + 2);
  boundaries.push_back(0);
  for (int b : traj_break_indices) {
    if (b > 0 && b < n_frames)
      boundaries.push_back(b);
  }
  if (boundaries.back() != n_frames)
    boundaries.push_back(n_frames);
  return boundaries;
}

inline bool build_tpm_for_lag(const std::vector<int> &discrete_traj,
                              const std::vector<int> &boundaries, int n_k,
                              int lag, Eigen::MatrixXd &count_tmp,
                              Eigen::MatrixXd &c_sym_tmp,
                              Eigen::MatrixXd &tpm_tmp) {
  count_tmp.setZero();
  for (std::size_t s = 0; s + 1 < boundaries.size(); ++s) {
    const int seg_start = boundaries[s];
    const int seg_end = boundaries[s + 1];
    for (int i = seg_start; i < seg_end - lag; ++i) {
      count_tmp(discrete_traj[static_cast<size_t>(i)],
                discrete_traj[static_cast<size_t>(i + lag)]) += 1.0;
    }
  }

  (void)c_sym_tmp;
  Eigen::VectorXd stationary;
  std::string error;
  return titan_markov::reversible_mle(count_tmp, tpm_tmp, stationary, error);
}

inline std::vector<double> compute_its_values_from_eigenvalues(
    const Eigen::VectorXcd &evals, int n_k, int lag, double dt_ns,
    int n_timescales) {
  std::vector<int> idx(static_cast<size_t>(n_k));
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return std::abs(evals[a]) > std::abs(evals[b]);
  });

  std::vector<double> values;
  values.reserve(static_cast<size_t>(std::max(0, n_timescales)));
  int written = 0;
  for (int rank = 1; rank < n_k && written < n_timescales; ++rank) {
    const int ei = idx[rank];
    const double abs_l = std::abs(evals[ei]);
    const double im_l = std::abs(evals[ei].imag());
    if (abs_l >= 1.0 || im_l > 1e-6 || abs_l < 1e-15) {
      values.push_back(std::numeric_limits<double>::quiet_NaN());
    } else {
      values.push_back(-static_cast<double>(lag) * dt_ns / std::log(abs_l));
    }
    ++written;
  }
  while (written < n_timescales) {
    values.push_back(std::numeric_limits<double>::quiet_NaN());
    ++written;
  }
  return values;
}

inline std::string format_its_value(double v) {
  if (!std::isfinite(v))
    return "  nan";
  std::ostringstream oss;
  oss << std::fixed << "  " << v;
  return oss.str();
}

inline bool solve_its_eigenvalues(
    const Eigen::MatrixXd &tpm, TitanMSMItsSolverMode mode, int n_timescales,
    int its_topk, int its_max_k_full, Eigen::VectorXcd &out_evals,
    std::string &effective_solver_mode) {
  const int n_k = static_cast<int>(tpm.rows());
  TitanMSMItsSolverMode chosen_mode = mode;
  if (chosen_mode == TitanMSMItsSolverMode::Full && n_k > its_max_k_full) {
    chosen_mode = TitanMSMItsSolverMode::Schur;
  }

  auto try_full = [&]() -> bool {
    Eigen::EigenSolver<Eigen::MatrixXd> es(tpm, /*computeEigenvectors=*/false);
    if (es.info() != Eigen::Success)
      return false;
    out_evals = es.eigenvalues();
    effective_solver_mode = "full";
    return true;
  };

  auto try_schur = [&]() -> bool {
    Eigen::RealSchur<Eigen::MatrixXd> schur;
    schur.compute(tpm, /*computeU=*/false);
    if (schur.info() != Eigen::Success)
      return false;
    const Eigen::MatrixXd T = schur.matrixT();
    out_evals.resize(n_k);
    int i = 0;
    while (i < n_k) {
      if (i + 1 < n_k && std::abs(T(i + 1, i)) > 1e-12) {
        const double a = T(i, i);
        const double b = T(i, i + 1);
        const double c = T(i + 1, i);
        const double d = T(i + 1, i + 1);
        const double tr = a + d;
        const double det = a * d - b * c;
        const double disc = tr * tr - 4.0 * det;
        if (disc >= 0.0) {
          const double root = std::sqrt(disc);
          out_evals[i] = std::complex<double>((tr + root) * 0.5, 0.0);
          out_evals[i + 1] = std::complex<double>((tr - root) * 0.5, 0.0);
        } else {
          const double im = std::sqrt(-disc) * 0.5;
          out_evals[i] = std::complex<double>(tr * 0.5, im);
          out_evals[i + 1] = std::complex<double>(tr * 0.5, -im);
        }
        i += 2;
      } else {
        out_evals[i] = std::complex<double>(T(i, i), 0.0);
        ++i;
      }
    }
    effective_solver_mode = "schur";
    return true;
  };

  auto try_partial = [&]() -> bool {
    int topk = its_topk;
    if (topk <= 0)
      topk = std::max(1, n_timescales + 4);
    topk = std::min(topk, n_k);
    if (topk <= 1) {
      return try_full();
    }

    constexpr int max_subspace_iter = 64;
    constexpr double subspace_tol = 1e-8;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Random(n_k, topk);
    Q.col(0).setConstant(1.0);

    auto orthonormalize = [&](Eigen::MatrixXd &M) {
      Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
      M = qr.householderQ() * Eigen::MatrixXd::Identity(M.rows(), M.cols());
    };

    orthonormalize(Q);

    Eigen::MatrixXd Z(n_k, topk);
    Eigen::MatrixXd B(topk, topk);
    Eigen::MatrixXd B_prev = Eigen::MatrixXd::Zero(topk, topk);

    for (int it = 0; it < max_subspace_iter; ++it) {
      Z.noalias() = tpm * Q;
      orthonormalize(Z);
      Q.swap(Z);
      B.noalias() = Q.transpose() * tpm * Q;

      if (it > 0) {
        const double diff = (B - B_prev).norm();
        const double scale = std::max(1.0, B.norm());
        if (diff <= subspace_tol * scale)
          break;
      }
      B_prev = B;
    }

    Eigen::EigenSolver<Eigen::MatrixXd> es_small(B, /*computeEigenvectors=*/false);
    if (es_small.info() != Eigen::Success)
      return false;

    const auto evals_small = es_small.eigenvalues();
    std::vector<int> idx_local(static_cast<size_t>(topk));
    std::iota(idx_local.begin(), idx_local.end(), 0);
    std::sort(idx_local.begin(), idx_local.end(),
              [&](int a, int b) {
                return std::abs(evals_small[a]) > std::abs(evals_small[b]);
              });

    out_evals.resize(topk);
    for (int i = 0; i < topk; ++i)
      out_evals[i] = evals_small[idx_local[static_cast<size_t>(i)]];

    effective_solver_mode = "partial(subspace)";
    return true;
  };

  bool ok = false;
  switch (chosen_mode) {
  case TitanMSMItsSolverMode::Partial:
    ok = try_partial();
    break;
  case TitanMSMItsSolverMode::Schur:
    ok = try_schur();
    break;
  case TitanMSMItsSolverMode::Full:
  default:
    ok = try_full();
    break;
  }

  if (!ok && chosen_mode != TitanMSMItsSolverMode::Full) {
    ok = try_full();
    if (ok) {
      effective_solver_mode = "full(fallback)";
    }
  }
  return ok;
}

template <typename RowConsumer>
inline void scan_best_centroids_blocked(
    const TitanMSM::ProjMatrixRM &frames, int64_t row_start, int64_t row_end,
    const TitanMSM::DenseMatrixRM &centroids, int centroid_count, int dim,
    RowConsumer &&consume_row) {
  if (row_start >= row_end || centroid_count <= 0) {
    return;
  }

  const double *frame_data = frames.data();
  const double *centroid_data = centroids.data();
  std::array<const double *, kMSMFrameBatch> frame_ptrs{};
  std::array<double, kMSMFrameBatch> best_d2{};
  std::array<int, kMSMFrameBatch> best_idx{};

  for (int64_t batch_start = row_start; batch_start < row_end;
       batch_start += kMSMFrameBatch) {
    const int batch_size = static_cast<int>(
        std::min<int64_t>(kMSMFrameBatch, row_end - batch_start));
    for (int b = 0; b < batch_size; ++b) {
      frame_ptrs[b] =
          frame_data + static_cast<ptrdiff_t>(batch_start + b) * dim;
      best_d2[b] = std::numeric_limits<double>::infinity();
      best_idx[b] = 0;
    }

    for (int centroid_block = 0; centroid_block < centroid_count;
         centroid_block += kMSMCentroidBlock) {
      const int centroid_end =
          std::min(centroid_count, centroid_block + kMSMCentroidBlock);
      if (centroid_end < centroid_count) {
        prefetch_row_read(centroid_data +
                          static_cast<ptrdiff_t>(centroid_end) * dim);
      }
      for (int c = centroid_block; c < centroid_end; ++c) {
        const double *centroid =
            centroid_data + static_cast<ptrdiff_t>(c) * dim;
        for (int b = 0; b < batch_size; ++b) {
          const double d2 =
              squared_distance_rowmajor_avx2(frame_ptrs[b], centroid, dim);
          if (d2 < best_d2[b]) {
            best_d2[b] = d2;
            best_idx[b] = c;
          }
        }
      }
    }

    for (int b = 0; b < batch_size; ++b) {
      consume_row(batch_start + b, best_idx[b], best_d2[b]);
    }
  }
}

} // namespace

// ============================================================================
// Constructor
// ============================================================================
TitanMSM::TitanMSM(int n_clusters, int n_tica_components, double dt_ns)
    : n_k_(n_clusters), n_comp_(n_tica_components), dt_ns_(dt_ns),
      msm_lag_time_(0),
      centroids_(DenseMatrixRM::Zero(n_clusters, n_tica_components)),
      count_matrix_(Eigen::MatrixXd::Zero(n_clusters, n_clusters)),
      tpm_(Eigen::MatrixXd::Zero(n_clusters, n_clusters)),
      stationary_dist_(
          Eigen::VectorXd::Constant(n_clusters, 1.0 / n_clusters)) {}

// ============================================================================
// Trajectory accumulation
// ============================================================================
void TitanMSM::add_projected_frame(const Eigen::VectorXd &proj_x,
                                   int replica_id,
                                   int64_t global_frame_index,
                                   double box_volume_nm3) {
  // Fix 3: grow proj_mat_ with doubling strategy (amortized O(1), contiguous)
  if (n_proj_frames_ >= proj_mat_.rows()) {
    int64_t new_rows = std::max(int64_t(1024), proj_mat_.rows() * 2);
    proj_mat_.conservativeResize(new_rows, n_comp_);
    // Reserve replica tracking vectors to match
    frame_replica_ids_.reserve(static_cast<size_t>(new_rows));
    frame_timestamps_.reserve(static_cast<size_t>(new_rows));
    frame_box_volumes_nm3_.reserve(static_cast<size_t>(new_rows));
    frame_global_indices_.reserve(static_cast<size_t>(new_rows));
  }
  proj_mat_.row(n_proj_frames_++) = proj_x.transpose();
  frame_replica_ids_.push_back(replica_id);
  frame_timestamps_.push_back(0.0);  // overwritten by record_frame_time
  frame_box_volumes_nm3_.push_back(box_volume_nm3);
  frame_global_indices_.push_back(global_frame_index);
  last_added_was_raw_ = false;
}

void TitanMSM::add_feature_frame(const Eigen::VectorXd &feature_x,
                                 int replica_id,
                                 int64_t global_frame_index,
                                 double box_volume_nm3) {
  if (feature_x.size() <= 0) {
    TitanUI::print_error(
        "TitanMSM::add_feature_frame: feature vector is empty.");
    return;
  }

  if (raw_feat_mat_.cols() == 0) {
    raw_feat_mat_.resize(0, feature_x.size());
  } else if (raw_feat_mat_.cols() != feature_x.size()) {
    TitanUI::print_error(
        "TitanMSM::add_feature_frame: inconsistent feature dimensionality (" +
        std::to_string(feature_x.size()) + " vs " +
        std::to_string(raw_feat_mat_.cols()) + ").");
    return;
  }

  if (n_raw_frames_ >= raw_feat_mat_.rows()) {
    int64_t new_rows = std::max(int64_t(1024), raw_feat_mat_.rows() * 2);
    raw_feat_mat_.conservativeResize(new_rows, feature_x.size());
    raw_frame_replica_ids_.reserve(static_cast<size_t>(new_rows));
    raw_frame_timestamps_.reserve(static_cast<size_t>(new_rows));
    raw_frame_box_volumes_nm3_.reserve(static_cast<size_t>(new_rows));
    raw_frame_global_indices_.reserve(static_cast<size_t>(new_rows));
  }

  raw_feat_mat_.row(n_raw_frames_++) = feature_x.transpose();
  raw_frame_replica_ids_.push_back(replica_id);
  raw_frame_timestamps_.push_back(0.0); // overwritten by record_frame_time
  raw_frame_box_volumes_nm3_.push_back(box_volume_nm3);
  raw_frame_global_indices_.push_back(global_frame_index);
  last_added_was_raw_ = true;
}

void TitanMSM::add_pending_frame_metadata(double time_ps, int replica_id,
                                          int64_t global_frame_index,
                                          double box_volume_nm3) {
  ++n_raw_frames_;
  raw_frame_replica_ids_.push_back(replica_id);
  raw_frame_timestamps_.push_back(time_ps);
  raw_frame_box_volumes_nm3_.push_back(box_volume_nm3);
  raw_frame_global_indices_.push_back(global_frame_index);
  last_added_was_raw_ = true;
}

void TitanMSM::add_trajectory_break() {
  if (n_raw_frames_ > 0 && n_proj_frames_ == 0) {
    raw_traj_break_indices_.push_back(static_cast<int>(n_raw_frames_));
    return;
  }
  traj_break_indices_.push_back(static_cast<int>(n_proj_frames_));
}

void TitanMSM::clear_projected() {
  n_proj_frames_ = 0;
  n_raw_frames_ = 0;
  // proj_mat_ capacity is kept to avoid re-allocation on reuse
  traj_break_indices_.clear();
  raw_traj_break_indices_.clear();
  discrete_traj_.clear();
  clustered_ = false;
  msm_built_ = false;
  eigen_solved_ = false;
  last_added_was_raw_ = false;
  frame_replica_ids_.clear();
  frame_timestamps_.clear();
  frame_box_volumes_nm3_.clear();
  frame_global_indices_.clear();
  raw_frame_replica_ids_.clear();
  raw_frame_timestamps_.clear();
  raw_frame_box_volumes_nm3_.clear();
  raw_frame_global_indices_.clear();
}

void TitanMSM::record_frame_time(double time_ps, int /*replica_id*/) {
  if (last_added_was_raw_) {
    if (!raw_frame_timestamps_.empty())
      raw_frame_timestamps_.back() = time_ps;
    return;
  }
  if (!frame_timestamps_.empty())
    frame_timestamps_.back() = time_ps;
}

void TitanMSM::project_pending_features(TitanKinetics &tica_analyzer) {
  if (!tica_analyzer.is_solved()) {
    TitanUI::print_error(
        "TitanMSM::project_pending_features: TICA model is not solved.");
    return;
  }
  if (n_raw_frames_ == 0) {
    return;
  }

  n_proj_frames_ = 0;
  traj_break_indices_.clear();
  discrete_traj_.clear();
  clustered_ = false;
  msm_built_ = false;
  eigen_solved_ = false;
  frame_replica_ids_.clear();
  frame_timestamps_.clear();
  frame_box_volumes_nm3_.clear();
  frame_global_indices_.clear();

  if (raw_feat_mat_.cols() == 0) {
    if (tica_analyzer.num_frames() != n_raw_frames_) {
      TitanUI::print_error(
          "TitanMSM::project_pending_features: TICA frame count (" +
          std::to_string(tica_analyzer.num_frames()) +
          ") does not match pending metadata frame count (" +
          std::to_string(n_raw_frames_) + ").");
      return;
    }

    proj_mat_.resize(n_raw_frames_, n_comp_);
    int64_t projected_rows = 0;
    try {
      tica_analyzer.project_all_frames_to_sink(
          [&](int64_t frame_index,
              const Eigen::Ref<const Eigen::VectorXd> &projected_row) {
            if (frame_index < 0 || frame_index >= n_raw_frames_) {
              throw std::runtime_error(
                  "TitanMSM::project_pending_features: projection frame index out of range.");
            }
            if (projected_row.size() != n_comp_) {
              throw std::runtime_error(
                  "TitanMSM::project_pending_features: projected row dimension mismatch.");
            }
            proj_mat_.row(frame_index) = projected_row.transpose();
            ++projected_rows;
          });
    } catch (const std::exception &e) {
      TitanUI::print_warning(e.what());
      return;
    }
    if (projected_rows != n_raw_frames_) {
      TitanUI::print_error(
          "TitanMSM::project_pending_features: projected frame count (" +
          std::to_string(projected_rows) +
          ") does not match pending metadata frame count (" +
          std::to_string(n_raw_frames_) + ").");
      return;
    }

    n_proj_frames_ = n_raw_frames_;
    frame_replica_ids_ = raw_frame_replica_ids_;
    frame_timestamps_ = raw_frame_timestamps_;
    frame_box_volumes_nm3_ = raw_frame_box_volumes_nm3_;
    frame_global_indices_ = raw_frame_global_indices_;
    traj_break_indices_ = raw_traj_break_indices_;

    n_raw_frames_ = 0;
    raw_feat_mat_.resize(0, 0);
    raw_traj_break_indices_.clear();
    raw_frame_replica_ids_.clear();
    raw_frame_timestamps_.clear();
    raw_frame_box_volumes_nm3_.clear();
    raw_frame_global_indices_.clear();
    last_added_was_raw_ = false;
    tica_analyzer.release_raw_feature_cache();
    return;
  }

  if (raw_feat_mat_.cols() != tica_analyzer.feature_dim()) {
    TitanUI::print_error(
        "TitanMSM::project_pending_features: raw feature dimension (" +
        std::to_string(raw_feat_mat_.cols()) + ") != TICA feature dimension (" +
        std::to_string(tica_analyzer.feature_dim()) + ").");
    return;
  }

  proj_mat_.resize(n_raw_frames_, n_comp_);
  frame_replica_ids_ = raw_frame_replica_ids_;
  frame_timestamps_ = raw_frame_timestamps_;
  frame_box_volumes_nm3_ = raw_frame_box_volumes_nm3_;
  frame_global_indices_ = raw_frame_global_indices_;
  traj_break_indices_ = raw_traj_break_indices_;

  for (int64_t i = 0; i < n_raw_frames_; ++i) {
    Eigen::Map<const Eigen::VectorXd> raw_frame(raw_feat_mat_.row(i).data(),
                                                raw_feat_mat_.cols());
    proj_mat_.row(i) = tica_analyzer.project_frame(raw_frame).transpose();
  }
  n_proj_frames_ = n_raw_frames_;

  n_raw_frames_ = 0;
  raw_feat_mat_.resize(0, 0);
  raw_traj_break_indices_.clear();
  raw_frame_replica_ids_.clear();
  raw_frame_timestamps_.clear();
  raw_frame_box_volumes_nm3_.clear();
  raw_frame_global_indices_.clear();
  last_added_was_raw_ = false;
  tica_analyzer.release_raw_feature_cache();
}

void TitanMSM::reconfigure_clusters(int n_clusters) {
  const int new_k = std::max(2, n_clusters);
  if (new_k == n_k_) {
    return;
  }

  n_k_ = new_k;
  centroids_ = DenseMatrixRM::Zero(n_k_, n_comp_);
  count_matrix_ = Eigen::MatrixXd::Zero(n_k_, n_k_);
  tpm_ = Eigen::MatrixXd::Zero(n_k_, n_k_);
  stationary_dist_ = Eigen::VectorXd::Constant(n_k_, 1.0 / n_k_);
  discrete_traj_.clear();
  clustered_ = false;
  msm_built_ = false;
  eigen_solved_ = false;
  cached_eigenvalues_.resize(0);
  cached_eigenvectors_.resize(0, 0);
}

std::unordered_map<int, std::vector<int>>
TitanMSM::get_per_replica_dtrajs() const {
  std::unordered_map<int, std::vector<int>> result;
  for (size_t i = 0; i < discrete_traj_.size(); ++i) {
    int rid = (i < frame_replica_ids_.size())
              ? frame_replica_ids_[i] : -1;
    result[rid].push_back(discrete_traj_[i]);
  }
  return result;
}

std::unordered_map<int, std::vector<double>>
TitanMSM::get_per_replica_timestamps() const {
  std::unordered_map<int, std::vector<double>> result;
  for (size_t i = 0; i < frame_timestamps_.size(); ++i) {
    int rid = (i < frame_replica_ids_.size())
              ? frame_replica_ids_[i] : -1;
    result[rid].push_back(frame_timestamps_[i]);
  }
  return result;
}

std::unordered_map<int, std::vector<double>>
TitanMSM::get_per_replica_box_volumes_nm3() const {
  std::unordered_map<int, std::vector<double>> result;
  for (size_t i = 0; i < frame_box_volumes_nm3_.size(); ++i) {
    int rid = (i < frame_replica_ids_.size()) ? frame_replica_ids_[i] : -1;
    result[rid].push_back(frame_box_volumes_nm3_[i]);
  }
  return result;
}

// ============================================================================
// train_kmeans — K-Means++ init + Lloyd iteration  (Fix 3: RowMajor proj_mat_)
// n_threads: number of std::threads for the assignment step
// ============================================================================
void TitanMSM::train_kmeans(TitanExecutor &executor, int max_iter,
                            unsigned int seed) {
  n_threads_ = std::max(1, static_cast<int>(executor.capacity()));
  // --- Preconditions -------------------------------------------------------
  if (n_proj_frames_ == 0) {
    TitanUI::print_error(
        "TitanMSM::train_kmeans: no projected frames collected. "
        "Call add_projected_frame() first.");
    return;
  }
  if (n_proj_frames_ < static_cast<int64_t>(n_k_)) {
    TitanUI::print_error(
        "TitanMSM::train_kmeans: fewer frames than clusters (" +
        std::to_string(n_proj_frames_) + " < " + std::to_string(n_k_) +
        "). Reduce msm_clusters.");
    return;
  }

  TitanUI::print_module_header("K-Means Clustering (MSM Phase 2A)");

  const int64_t N = n_proj_frames_;
  std::mt19937 rng(seed);

  // -------------------------------------------------------------------------
  // K-Means++ Initialization  (single-threaded, touches whole proj_mat_)
  // -------------------------------------------------------------------------
  {
    std::uniform_int_distribution<int64_t> uni(0, N - 1);
    centroids_.row(0) = proj_mat_.row(uni(rng));
  }

  // -------------------------------------------------------------------------
  // K-Means++ Initialization — parallel weight computation per centroid step
  // -------------------------------------------------------------------------
  std::vector<double> weights(N, 0.0);

  auto compute_weights_parallel = [&](int up_to_c) {
    const int64_t chunk = (N + n_threads_ - 1) / n_threads_;
    executor.parallel_do(static_cast<size_t>(n_threads_), [&](size_t t_sz) {
      const int t = static_cast<int>(t_sz);
      const int64_t start = t * chunk;
      const int64_t end = std::min(start + chunk, N);
      if (start >= end)
        return;
      scan_best_centroids_blocked(
          proj_mat_, start, end, centroids_, up_to_c, n_comp_,
          [&](int64_t i, int /*best_c*/, double min_d2) {
            weights[static_cast<size_t>(i)] = min_d2;
          });
    });
  };

  for (int c = 1; c < n_k_; ++c) {
    compute_weights_parallel(c);
    std::discrete_distribution<int64_t> weighted(weights.begin(),
                                                 weights.end());
    centroids_.row(c) = proj_mat_.row(weighted(rng));
  }

  TitanUI::print_info("  K-Means++ initialization complete (" +
                      std::to_string(n_k_) + " centroids).");

  // -------------------------------------------------------------------------
  // K-Means Lloyd Iteration  (multi-threaded assignment via shared executor)
  // -------------------------------------------------------------------------
  discrete_traj_.assign(static_cast<size_t>(N), 0);
  DenseMatrixRM new_centroids(n_k_, n_comp_);
  std::vector<int> counts(n_k_);

  // Per-thread partial accumulators (to avoid mutex contention in update step)
  std::vector<DenseMatrixRM> thread_sums(
      n_threads_, DenseMatrixRM::Zero(n_k_, n_comp_));
  std::vector<std::vector<int>> thread_counts(n_threads_,
                                              std::vector<int>(n_k_, 0));

  double convergence = std::numeric_limits<double>::max();
  int iter = 0;

  for (iter = 0; iter < max_iter && convergence > 1e-6; ++iter) {

    // ----- Assignment step (parallel over frames) -------------------------
    auto assign_worker = [&](int64_t row_start, int64_t row_end) {
      scan_best_centroids_blocked(
          proj_mat_, row_start, row_end, centroids_, n_k_, n_comp_,
          [&](int64_t i, int best_c, double /*best_d2*/) {
            discrete_traj_[static_cast<size_t>(i)] = best_c;
          });
    };

    {
      const int64_t chunk = (N + n_threads_ - 1) / n_threads_;
      executor.parallel_do(static_cast<size_t>(n_threads_), [&](size_t t_sz) {
        const int64_t start = static_cast<int64_t>(t_sz) * chunk;
        const int64_t end = std::min(start + chunk, N);
        if (start < end)
          assign_worker(start, end);
      });
    }

    // ----- Update step (parallel partial sums, serial merge) --------------
    for (int t = 0; t < n_threads_; ++t) {
      thread_sums[t].setZero();
      std::fill(thread_counts[t].begin(), thread_counts[t].end(), 0);
    }

    auto update_worker = [&](int tid, int64_t row_start, int64_t row_end) {
      for (int64_t i = row_start; i < row_end; ++i) {
        int c = discrete_traj_[static_cast<size_t>(i)];
        double *dst =
            thread_sums[tid].data() + static_cast<ptrdiff_t>(c) * n_comp_;
        const double *src =
            proj_mat_.data() + static_cast<ptrdiff_t>(i) * n_comp_;
        add_row_inplace(dst, src, n_comp_);
        thread_counts[tid][c]++;
      }
    };
    {
      const int64_t chunk = (N + n_threads_ - 1) / n_threads_;
      executor.parallel_do(static_cast<size_t>(n_threads_), [&](size_t t_sz) {
        const int t = static_cast<int>(t_sz);
        const int64_t start = static_cast<int64_t>(t) * chunk;
        const int64_t end = std::min(start + chunk, N);
        if (start < end)
          update_worker(t, start, end);
      });
    }

    // Merge partial sums
    new_centroids.setZero();
    std::fill(counts.begin(), counts.end(), 0);
    for (int t = 0; t < n_threads_; ++t) {
      new_centroids += thread_sums[t];
      for (int c = 0; c < n_k_; ++c)
        counts[c] += thread_counts[t][c];
    }

    for (int c = 0; c < n_k_; ++c) {
      if (counts[c] > 0) {
        new_centroids.row(c) /= static_cast<double>(counts[c]);
      } else {
        // Empty cluster: reinit to farthest point (single-thread scan)
        TitanUI::print_info("  [WARNING] K-Means: cluster " +
                            std::to_string(c) +
                            " is empty. Reinitializing to farthest point.");
        std::vector<double> local_max_d2(static_cast<size_t>(n_threads_), -1.0);
        std::vector<int64_t> local_far_idx(static_cast<size_t>(n_threads_), 0);
        {
          const int64_t chunk = (N + n_threads_ - 1) / n_threads_;
          executor.parallel_do(static_cast<size_t>(n_threads_), [&](size_t t_sz) {
            const int t = static_cast<int>(t_sz);
            const int64_t start = static_cast<int64_t>(t) * chunk;
            const int64_t end = std::min(start + chunk, N);
            if (start >= end)
              return;
              double best_d2 = -1.0;
              int64_t best_idx = start;
              for (int64_t i = start; i < end; ++i) {
                const int near_c = discrete_traj_[static_cast<size_t>(i)];
                const double *frame =
                    proj_mat_.data() + static_cast<ptrdiff_t>(i) * n_comp_;
                const double *centroid =
                    centroids_.data() + static_cast<ptrdiff_t>(near_c) * n_comp_;
                const double d2 =
                    squared_distance_rowmajor_avx2(frame, centroid, n_comp_);
                if (d2 > best_d2) {
                  best_d2 = d2;
                  best_idx = i;
                }
              }
              local_max_d2[static_cast<size_t>(t)] = best_d2;
              local_far_idx[static_cast<size_t>(t)] = best_idx;
          });
        }

        double max_d2 = -1.0;
        int far_idx = 0;
        for (int t = 0; t < n_threads_; ++t) {
          if (local_max_d2[static_cast<size_t>(t)] > max_d2) {
            max_d2 = local_max_d2[static_cast<size_t>(t)];
            far_idx = static_cast<int>(local_far_idx[static_cast<size_t>(t)]);
          }
        }
        new_centroids.row(c) = proj_mat_.row(far_idx);
        counts[c] = 1;
      }
    }

    // Convergence check
    convergence = 0.0;
    for (int c = 0; c < n_k_; ++c) {
      const double move = std::sqrt(squared_distance_rowmajor_avx2(
          new_centroids.data() + static_cast<ptrdiff_t>(c) * n_comp_,
          centroids_.data() + static_cast<ptrdiff_t>(c) * n_comp_, n_comp_));
      if (move > convergence)
        convergence = move;
    }
    centroids_ = new_centroids;
  }

  // Final assignment
  {
    auto assign_worker = [&](int64_t row_start, int64_t row_end) {
      scan_best_centroids_blocked(
          proj_mat_, row_start, row_end, centroids_, n_k_, n_comp_,
          [&](int64_t i, int best_c, double /*best_d2*/) {
            discrete_traj_[static_cast<size_t>(i)] = best_c;
          });
    };
    const int64_t chunk = (N + n_threads_ - 1) / n_threads_;
    executor.parallel_do(static_cast<size_t>(n_threads_), [&](size_t t_sz) {
      const int64_t start = static_cast<int64_t>(t_sz) * chunk;
      const int64_t end = std::min(start + chunk, N);
      if (start < end)
        assign_worker(start, end);
    });
  }

  // Inertia
  double inertia = 0.0;
  std::vector<double> local_inertia(static_cast<size_t>(n_threads_), 0.0);
  {
    const int64_t chunk = (N + n_threads_ - 1) / n_threads_;
    executor.parallel_do(static_cast<size_t>(n_threads_), [&](size_t t_sz) {
      const int t = static_cast<int>(t_sz);
      const int64_t start = static_cast<int64_t>(t) * chunk;
      const int64_t end = std::min(start + chunk, N);
      if (start >= end)
        return;
        double partial = 0.0;
        for (int64_t i = start; i < end; ++i) {
          const double *frame =
              proj_mat_.data() + static_cast<ptrdiff_t>(i) * n_comp_;
          const int cluster = discrete_traj_[static_cast<size_t>(i)];
          const double *centroid =
              centroids_.data() + static_cast<ptrdiff_t>(cluster) * n_comp_;
          partial += squared_distance_rowmajor_avx2(frame, centroid, n_comp_);
        }
        local_inertia[static_cast<size_t>(t)] = partial;
    });
  }
  for (double partial : local_inertia) {
    inertia += partial;
  }

  clustered_ = true;
  TitanUI::print_info("  Converged in " + std::to_string(iter) +
                      " iterations (max centroid shift = " +
                      std::to_string(convergence) + ").");
  TitanUI::print_info("  K-Means inertia: " + std::to_string(inertia));
}

// ============================================================================
// build_markov_model — boundary-aware count matrix → TPM → stationary dist
// ============================================================================
void TitanMSM::build_markov_model(int msm_lag_time) {
  // --- Preconditions -------------------------------------------------------
  if (!clustered_) {
    TitanUI::print_error(
        "TitanMSM::build_markov_model: train_kmeans() must run first.");
    return;
  }
  if (msm_lag_time <= 0) {
    TitanUI::print_error(
        "TitanMSM::build_markov_model: msm_lag_time must be > 0.");
    return;
  }
  const int N_frames = static_cast<int>(n_proj_frames_);
  if (N_frames < msm_lag_time * 10) {
    TitanUI::print_info(
        "[WARNING] TitanMSM: trajectory length (" + std::to_string(N_frames) +
        " frames) < 10 * lag_time (" + std::to_string(msm_lag_time * 10) +
        "). MSM statistics may be unreliable.");
  }

  msm_lag_time_ = msm_lag_time;

  // -------------------------------------------------------------------------
  // Step 1 — Count matrix (boundary-aware)
  // Build segments from traj_break_indices_. The last segment ends at N_frames.
  // -------------------------------------------------------------------------
  count_matrix_ = Eigen::MatrixXd::Zero(n_k_, n_k_);

  // Build segment boundary list: [0, b0, b1, ..., N_frames]
  std::vector<int> boundaries;
  boundaries.push_back(0);
  for (int b : traj_break_indices_) {
    if (b > 0 && b < N_frames)
      boundaries.push_back(b);
  }
  // If no breaks we declared, or all frames form one segment,
  // the last boundary is always N_frames.
  if (boundaries.back() != N_frames)
    boundaries.push_back(N_frames);

  for (std::size_t s = 0; s + 1 < boundaries.size(); ++s) {
    const int seg_start = boundaries[s];
    const int seg_end = boundaries[s + 1]; // exclusive
    for (int i = seg_start; i < seg_end - msm_lag_time; ++i) {
      count_matrix_(discrete_traj_[i], discrete_traj_[i + msm_lag_time]) += 1.0;
    }
  }

  std::string estimator_error;
  if (!titan_markov::reversible_mle(count_matrix_, tpm_, stationary_dist_, estimator_error)) {
    msm_built_ = false;
    eigen_solved_ = false;
    TitanUI::print_error("MSM: " + estimator_error);
    return;
  }
  msm_built_ = true;

  // Fix 4: cache eigendecomposition NOW (once, used by ITS + PCCA)
  {
    Eigen::EigenSolver<Eigen::MatrixXd> es(tpm_, /*computeEigenvectors=*/true);
    if (es.info() == Eigen::Success) {
      cached_eigenvalues_ = es.eigenvalues();
      cached_eigenvectors_ = es.eigenvectors();
      eigen_solved_ = true;
    } else {
      TitanUI::print_info(
          "[WARNING] TitanMSM: EigenSolver failed to converge — "
          "ITS and PCCA will be unavailable.");
      eigen_solved_ = false;
    }
  }
}

// ============================================================================
// compute_implied_timescales — eigendecomposition of TPM
// ============================================================================
std::vector<double>
TitanMSM::compute_implied_timescales(int n_timescales) const {
  if (!msm_built_) {
    TitanUI::print_error("TitanMSM::compute_implied_timescales: "
                         "build_markov_model() must run first.");
    return {};
  }
  if (!eigen_solved_)
    return {};

  // Fix 4: use cached eigenvalues from build_markov_model
  const auto &evals = cached_eigenvalues_;

  // Sort indices by descending |eigenvalue|
  const int k = n_k_;
  std::vector<int> idx(k);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return std::abs(evals[a]) > std::abs(evals[b]);
  });

  std::vector<double> timescales;
  timescales.reserve(n_timescales);

  for (int rank = 1;
       rank < k && static_cast<int>(timescales.size()) < n_timescales; ++rank) {
    int ei = idx[rank];
    double abs_lambda = std::abs(evals[ei]);
    double imag_part = std::abs(evals[ei].imag());

    if (abs_lambda >= 1.0) {
      continue;
    }
    if (imag_part > 1e-6) {
      continue;
    }
    if (abs_lambda < 1e-15) {
      continue;
    }

    double t_i =
        -static_cast<double>(msm_lag_time_) * dt_ns_ / std::log(abs_lambda);
    timescales.push_back(t_i);
  }

  return timescales;
}

// ============================================================================
// generate_its_lags — geometric sequence for ITS scan
// ============================================================================
std::vector<int> TitanMSM::generate_its_lags() const {
  const int max_usable = static_cast<int>(n_proj_frames_) / 10;
  if (max_usable < 5) {
    TitanUI::print_info(
        "[WARNING] TitanMSM: trajectory too short for reliable ITS scan "
        "(max_usable_lag=" +
        std::to_string(max_usable) +
        " < 5). "
        "Skipping ITS scan.");
    return {};
  }

  // Geometric sequence: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, ...
  static const int geo_steps[] = {1,    2,    5,     10,    20,
                                  50,   100,  200,   500,   1000,
                                  2000, 5000, 10000, 20000, 50000};
  std::vector<int> lags;
  for (int lag : geo_steps) {
    if (lag > max_usable)
      break;
    lags.push_back(lag);
  }
  return lags;
}

// ============================================================================
// run_pcca_crisp — weighted spectral clustering on slow MSM eigenmodes
// ============================================================================
std::vector<int> TitanMSM::run_pcca_crisp(int n_macro) const {
  if (!eigen_solved_) {
    TitanUI::print_info("[WARNING] PCCA: no cached eigenvalues. Skipping.");
    return std::vector<int>(n_k_, 0);
  }

  const int k = n_k_;
  if (n_macro >= k) {
    std::vector<int> identity(k, 0);
    std::iota(identity.begin(), identity.end(), 0);
    return identity;
  }

  std::vector<int> idx(k);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    const auto lambda_a = cached_eigenvalues_[a];
    const auto lambda_b = cached_eigenvalues_[b];
    if (lambda_a.real() != lambda_b.real()) {
      return lambda_a.real() > lambda_b.real();
    }
    if (std::abs(lambda_a.imag()) != std::abs(lambda_b.imag())) {
      return std::abs(lambda_a.imag()) < std::abs(lambda_b.imag());
    }
    return std::abs(lambda_a) > std::abs(lambda_b);
  });

  const int embedding_dim =
      std::min(std::max(1, n_macro - 1), std::max(1, k - 1));
  std::vector<int> selected_modes;
  selected_modes.reserve(static_cast<size_t>(embedding_dim));
  std::vector<unsigned char> mode_taken(static_cast<size_t>(k), 0u);

  for (int sorted_pos = 0; sorted_pos < k &&
                           static_cast<int>(selected_modes.size()) < embedding_dim;
       ++sorted_pos) {
    const int eig_idx = idx[sorted_pos];
    const auto lambda = cached_eigenvalues_[eig_idx];
    const bool stationary_like =
        std::abs(lambda.real() - 1.0) < 1e-8 && std::abs(lambda.imag()) < 1e-8;
    if (stationary_like && k > 1) {
      continue;
    }
    selected_modes.push_back(eig_idx);
    mode_taken[static_cast<size_t>(eig_idx)] = 1u;
  }
  for (int sorted_pos = 0; sorted_pos < k &&
                           static_cast<int>(selected_modes.size()) < embedding_dim;
       ++sorted_pos) {
    const int eig_idx = idx[sorted_pos];
    if (mode_taken[static_cast<size_t>(eig_idx)] != 0u) {
      continue;
    }
    selected_modes.push_back(eig_idx);
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      embedding(k, embedding_dim);
  embedding.setZero();
  for (int col = 0; col < static_cast<int>(selected_modes.size()); ++col) {
    const int eig_idx = selected_modes[static_cast<size_t>(col)];
    embedding.col(col) = cached_eigenvectors_.col(eig_idx).real();
    Eigen::Index pivot_row = 0;
    embedding.col(col).cwiseAbs().maxCoeff(&pivot_row);
    if (embedding(pivot_row, col) < 0.0) {
      embedding.col(col) *= -1.0;
    }
  }

  for (int i = 0; i < k; ++i) {
    const double row_norm = embedding.row(i).norm();
    if (row_norm > 1e-12) {
      embedding.row(i) /= row_norm;
    }
  }

  std::vector<double> sample_weights(static_cast<size_t>(k), 1.0);
  if (stationary_dist_.size() == k) {
    for (int i = 0; i < k; ++i) {
      const double w = stationary_dist_[i];
      if (std::isfinite(w) && w > 0.0) {
        sample_weights[static_cast<size_t>(i)] = w;
      }
    }
  }

  Eigen::MatrixXd centroids = Eigen::MatrixXd::Zero(n_macro, embedding_dim);
  std::vector<int> macrostates(static_cast<size_t>(k), 0);
  std::vector<unsigned char> chosen(static_cast<size_t>(k), 0u);

  int first_centroid = 0;
  double first_weight = -1.0;
  for (int i = 0; i < k; ++i) {
    if (sample_weights[static_cast<size_t>(i)] > first_weight) {
      first_weight = sample_weights[static_cast<size_t>(i)];
      first_centroid = i;
    }
  }
  centroids.row(0) = embedding.row(first_centroid);
  chosen[static_cast<size_t>(first_centroid)] = 1u;

  std::mt19937 rng(1337u + static_cast<unsigned int>(n_macro) +
                   static_cast<unsigned int>(k));
  std::vector<double> scores(static_cast<size_t>(k), 0.0);
  for (int c = 1; c < n_macro; ++c) {
    double total_score = 0.0;
    int fallback_idx = -1;
    double fallback_weight = -1.0;
    for (int i = 0; i < k; ++i) {
      if (chosen[static_cast<size_t>(i)] != 0u) {
        scores[static_cast<size_t>(i)] = 0.0;
        continue;
      }
      double min_d2 = std::numeric_limits<double>::max();
      for (int prev = 0; prev < c; ++prev) {
        const double d2 =
            (embedding.row(i) - centroids.row(prev)).squaredNorm();
        if (d2 < min_d2) {
          min_d2 = d2;
        }
      }
      const double score = sample_weights[static_cast<size_t>(i)] * min_d2;
      scores[static_cast<size_t>(i)] = score;
      total_score += score;
      if (sample_weights[static_cast<size_t>(i)] > fallback_weight) {
        fallback_weight = sample_weights[static_cast<size_t>(i)];
        fallback_idx = i;
      }
    }

    int picked = fallback_idx;
    if (total_score > 0.0) {
      std::discrete_distribution<int> weighted(scores.begin(), scores.end());
      picked = weighted(rng);
      if (picked < 0 || picked >= k ||
          chosen[static_cast<size_t>(picked)] != 0u) {
        picked = fallback_idx;
      }
    }
    if (picked < 0) {
      picked = c % k;
    }

    centroids.row(c) = embedding.row(picked);
    chosen[static_cast<size_t>(picked)] = 1u;
  }

  constexpr int max_iter = 100;
  for (int iter = 0; iter < max_iter; ++iter) {
    bool changed = false;
    for (int i = 0; i < k; ++i) {
      double best_d2 = std::numeric_limits<double>::max();
      int best_c = 0;
      for (int c = 0; c < n_macro; ++c) {
        const double d2 = (embedding.row(i) - centroids.row(c)).squaredNorm();
        if (d2 < best_d2) {
          best_d2 = d2;
          best_c = c;
        }
      }
      if (macrostates[static_cast<size_t>(i)] != best_c) {
        macrostates[static_cast<size_t>(i)] = best_c;
        changed = true;
      }
    }

    Eigen::MatrixXd next_centroids = Eigen::MatrixXd::Zero(n_macro, embedding_dim);
    std::vector<double> cluster_weights(static_cast<size_t>(n_macro), 0.0);
    for (int i = 0; i < k; ++i) {
      const int c = macrostates[static_cast<size_t>(i)];
      const double w = sample_weights[static_cast<size_t>(i)];
      next_centroids.row(c).noalias() += w * embedding.row(i);
      cluster_weights[static_cast<size_t>(c)] += w;
    }

    for (int c = 0; c < n_macro; ++c) {
      if (cluster_weights[static_cast<size_t>(c)] > 0.0) {
        next_centroids.row(c) /= cluster_weights[static_cast<size_t>(c)];
        continue;
      }

      double farthest_score = -1.0;
      int farthest_idx = 0;
      for (int i = 0; i < k; ++i) {
        const int assigned = macrostates[static_cast<size_t>(i)];
        const double d2 =
            (embedding.row(i) - centroids.row(assigned)).squaredNorm();
        const double score = sample_weights[static_cast<size_t>(i)] * d2;
        if (score > farthest_score) {
          farthest_score = score;
          farthest_idx = i;
        }
      }
      next_centroids.row(c) = embedding.row(farthest_idx);
    }

    centroids.swap(next_centroids);
    if (!changed) {
      break;
    }
  }

  return macrostates;
}

int TitanMSM::effective_pcca_macrostates(int msm_macrostates) const {
  if (msm_macrostates == -1) {
    const int n_timescales = std::max(1, n_k_ - 1);
    const auto ref_timescales = compute_implied_timescales(n_timescales);
    return auto_detect_macrostates(ref_timescales);
  }
  return std::clamp(msm_macrostates, 2, n_k_);
}

std::vector<int> TitanMSM::compute_pcca_macrostates(int msm_macrostates) const {
  if (!msm_built_) {
    TitanUI::print_warning("PCCA: MSM is not built yet. Returning empty macrostate map.");
    return {};
  }
  const int n_macro = effective_pcca_macrostates(msm_macrostates);
  return run_pcca_crisp(n_macro);
}

// ============================================================================
// auto_detect_macrostates
// ============================================================================
int TitanMSM::auto_detect_macrostates(
    const std::vector<double> &timescales) const {
  double threshold = 2.0 * static_cast<double>(msm_lag_time_) * dt_ns_;
  int n_slow = 0;
  for (double t : timescales) {
    if (t > threshold)
      ++n_slow;
  }
  // n_macro = n_slow_processes + 1 (includes ground state)
  int n_macro = n_slow + 1;
  // Clamp: at least 2, at most k/2
  n_macro = std::clamp(n_macro, 2, std::max(2, n_k_ / 2));
  return n_macro;
}

// ============================================================================
// export_results — write all output files + run ITS scan + PCCA
// ============================================================================
void TitanMSM::export_results(
    TitanExecutor &executor, const info &info_output, std::fstream &log,
    int n_timescales, bool msm_run_its, bool msm_run_pcca, int msm_macrostates,
    const std::vector<int> &its_lags, TitanMSMItsSolverMode its_solver_mode,
    int its_topk, int its_max_k_full, bool kinetics_interpretable) const {
  if (!msm_built_) {
    TitanUI::print_error(
        "TitanMSM::export_results: build_markov_model() must run first.");
    return;
  }

  TitanUI::print_module_header("MSM Export Results (Phase 2D)");

  // =========================================================================
  // 1. Compute reference implied timescales (at the stored lag)
  // =========================================================================
  const auto ref_timescales = kinetics_interpretable
                                  ? compute_implied_timescales(n_timescales)
                                  : std::vector<double>{};
  TitanUI::log_kv(log, "MSM Kinetic Interpretation",
                  kinetics_interpretable ? "physical continuity available"
                                         : "disabled (REMD state streams)");

  // =========================================================================
  // 2. Write TPM
  // =========================================================================
  {
    std::ofstream f(info_output.get_msm_tpm_filename());
    if (!f.is_open()) {
      TitanUI::print_error("Cannot open " + info_output.get_msm_tpm_filename());
    } else {
      f << "# k=" << n_k_ << "  lag=" << msm_lag_time_ << "  dt=" << dt_ns_
        << " ns\n";
      f << std::fixed;
      for (int i = 0; i < n_k_; ++i) {
        for (int j = 0; j < n_k_; ++j) {
          if (j > 0)
            f << ' ';
          f << tpm_(i, j);
        }
        f << '\n';
      }
      TitanUI::print_info("  Written: " + info_output.get_msm_tpm_filename());
    }
  }

  // =========================================================================
  // 2b. Write raw count matrix (before symmetrization, for PyEMMA/msmtools MLE)
  // =========================================================================
  {
    std::ofstream f(info_output.get_msm_count_matrix_filename());
    if (!f.is_open()) {
      TitanUI::print_error("Cannot open " +
                           info_output.get_msm_count_matrix_filename());
    } else {
      f << "# Raw count matrix k=" << n_k_ << " lag=" << msm_lag_time_ << "\n";
      f << std::fixed << std::setprecision(1);
      for (int i = 0; i < n_k_; ++i) {
        for (int j = 0; j < n_k_; ++j) {
          if (j > 0)
            f << ' ';
          f << count_matrix_(i, j);
        }
        f << '\n';
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_msm_count_matrix_filename());
    }
  }

  // =========================================================================
  // 2c. Write stationary distribution π (for deeptime/msmtools weighted PCCA+)
  // =========================================================================
  {
    std::ofstream f(info_output.get_msm_stationary_filename());
    if (!f.is_open()) {
      TitanUI::print_error("Cannot open " +
                           info_output.get_msm_stationary_filename());
    } else {
      f << "# microstate  stationary_prob\n";
      f << std::scientific << std::setprecision(8);
      for (int i = 0; i < n_k_; ++i)
        f << i << "  " << stationary_dist_[i] << "\n";
      TitanUI::print_info("  Written: " +
                          info_output.get_msm_stationary_filename());
    }
  }

  // =========================================================================
  // 3. Write discrete trajectory
  // =========================================================================
  {
    std::ofstream f(info_output.get_msm_discrete_traj_filename());
    if (!f.is_open()) {
      TitanUI::print_error("Cannot open " +
                           info_output.get_msm_discrete_traj_filename());
    } else {
      for (int s : discrete_traj_)
        f << s << '\n';
      TitanUI::print_info("  Written: " +
                          info_output.get_msm_discrete_traj_filename());
    }
  }

  // =========================================================================
  // 4. Write cluster centroids
  // =========================================================================
  {
    std::ofstream f(info_output.get_msm_centroids_filename());
    if (!f.is_open()) {
      TitanUI::print_error("Cannot open " +
                           info_output.get_msm_centroids_filename());
    } else {
      f << "# k=" << n_k_ << "  n_tica_components=" << n_comp_ << '\n';
      f << std::fixed;
      for (int i = 0; i < n_k_; ++i) {
        for (int j = 0; j < n_comp_; ++j) {
          if (j > 0)
            f << ' ';
          f << centroids_(i, j);
        }
        f << '\n';
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_msm_centroids_filename());
    }
  }

  // =========================================================================
  // 5. Write implied timescales (reference lag)
  // =========================================================================
  {
    std::ofstream f(info_output.get_msm_timescales_filename());
    if (!f.is_open()) {
      TitanUI::print_error("Cannot open " +
                           info_output.get_msm_timescales_filename());
    } else {
      if (!kinetics_interpretable) {
        f << "# unavailable: REMD thermodynamic-state trajectories contain "
             "accepted configuration exchanges\n";
      }
      // Fix 4: use cached eigenvalues from build_markov_model for the timescale
      // table
      if (!kinetics_interpretable) {
        // State populations and count matrices remain available above.
      } else if (!eigen_solved_) {
        TitanUI::print_info(
            "[WARNING] Eigenvalues not cached; skipping table.");
      } else {
        const auto &evals = cached_eigenvalues_;
        std::vector<int> idx(n_k_);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
          return std::abs(evals[a]) > std::abs(evals[b]);
        });

        int written = 0;
        for (int rank = 1; rank < n_k_ && written < n_timescales; ++rank) {
          int ei = idx[rank];
          double abs_lambda = std::abs(evals[ei]);
          double imag_part = std::abs(evals[ei].imag());
          if (abs_lambda >= 1.0 || imag_part > 1e-6 || abs_lambda < 1e-15)
            continue;
          double t_i = -static_cast<double>(msm_lag_time_) * dt_ns_ /
                       std::log(abs_lambda);
          f << std::fixed << written << "  " << t_i << "  " << abs_lambda
            << '\n';
          ++written;
        }
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_msm_timescales_filename());
    }
  }

  // =========================================================================
  // =========================================================================
  // 6. ITS scan (multi-lag)
  // =========================================================================
  if (msm_run_its && kinetics_interpretable) {
    std::vector<int> lags_to_use =
        its_lags.empty() ? generate_its_lags() : its_lags;

    if (!lags_to_use.empty()) {
      std::ofstream f(info_output.get_msm_its_filename());
      if (!f.is_open()) {
        TitanUI::print_error("Cannot open " +
                             info_output.get_msm_its_filename());
      } else {
        f << "# lag_frames  lag_ns";
        for (int t = 0; t < n_timescales; ++t)
          f << "  t" << (t + 1) << "_ns";
        f << '\n';

        struct LagResult {
          int lag = 0;
          bool ok = false;
          std::vector<double> its_values;
          std::string solver_mode;
        };

        const int n_lags = static_cast<int>(lags_to_use.size());
        std::vector<LagResult> results(static_cast<size_t>(n_lags));

        const int n_frames = static_cast<int>(n_proj_frames_);
        const std::vector<int> boundaries =
            build_segment_boundaries(traj_break_indices_, n_frames);
        const int n_workers = std::max(1, std::min(n_lags, n_threads_));

        const auto its_start = std::chrono::steady_clock::now();
        const auto matrix_start = std::chrono::steady_clock::now();
        executor.parallel_do(static_cast<size_t>(n_workers), [&](size_t w_sz) {
          const int w = static_cast<int>(w_sz);
          const int shard_begin = n_lags * w / n_workers;
          const int shard_end = n_lags * (w + 1) / n_workers;
          if (shard_begin >= shard_end)
            return;

          Eigen::MatrixXd c_tmp(n_k_, n_k_);
          Eigen::MatrixXd c_sym_tmp(n_k_, n_k_);
          Eigen::MatrixXd t_tmp(n_k_, n_k_);
          Eigen::VectorXcd evals_tmp;

          for (int li = shard_begin; li < shard_end; ++li) {
            const int lag = lags_to_use[static_cast<size_t>(li)];
            LagResult &res = results[static_cast<size_t>(li)];
            res.lag = lag;

            if (!build_tpm_for_lag(discrete_traj_, boundaries, n_k_, lag, c_tmp,
                                  c_sym_tmp, t_tmp)) {
              res.its_values.assign(static_cast<size_t>(std::max(0, n_timescales)),
                                    std::numeric_limits<double>::quiet_NaN());
              res.solver_mode = "invalid_counts_or_mle_failure";
              res.ok = true;
              continue;
            }

            std::string effective_mode;
            if (!solve_its_eigenvalues(t_tmp, its_solver_mode, n_timescales,
                                       its_topk, its_max_k_full, evals_tmp,
                                       effective_mode)) {
              res.its_values.assign(static_cast<size_t>(std::max(0, n_timescales)),
                                    std::numeric_limits<double>::quiet_NaN());
              res.solver_mode = "failed";
              res.ok = true;
              continue;
            }

            res.its_values = compute_its_values_from_eigenvalues(
                evals_tmp, static_cast<int>(evals_tmp.size()), lag, dt_ns_,
                n_timescales);
            res.solver_mode = effective_mode;
            res.ok = true;
          }
        });
        const auto matrix_end = std::chrono::steady_clock::now();

        std::filesystem::create_directories(info_output.get_msm_its_tpm_dir());

        Eigen::MatrixXd c_tmp_write(n_k_, n_k_);
        Eigen::MatrixXd c_sym_tmp_write(n_k_, n_k_);
        Eigen::MatrixXd t_tmp_write(n_k_, n_k_);
        std::unordered_map<std::string, int> solver_hist;
        for (int li = 0; li < n_lags; ++li) {
          const LagResult &res = results[static_cast<size_t>(li)];
          if (!res.ok)
            continue;

          const int lag = res.lag;
          solver_hist[res.solver_mode] += 1;

          const bool have_tpm = build_tpm_for_lag(discrete_traj_, boundaries, n_k_, lag, c_tmp_write,
                                                 c_sym_tmp_write, t_tmp_write);
          if (have_tpm) {
            const std::string tpm_path = info_output.get_msm_its_tpm_dir() +
                                         "/tpm_lag" + std::to_string(lag) + ".dat";
            std::ofstream ft(tpm_path);
            if (ft.is_open()) {
              ft << "# TPM lag=" << lag << "  dt=" << dt_ns_
                 << " ns  k=" << n_k_ << "\n";
              ft << std::fixed;
              for (int i = 0; i < n_k_; ++i) {
                for (int j = 0; j < n_k_; ++j) {
                  if (j > 0)
                    ft << ' ';
                  ft << t_tmp_write(i, j);
                }
                ft << '\n';
              }
            }
          }

          const double lag_ns = static_cast<double>(lag) * dt_ns_;
          f << std::fixed << lag << "  " << lag_ns;
          for (double v : res.its_values)
            f << format_its_value(v);
          f << '\n';
        }
        const auto its_end = std::chrono::steady_clock::now();

        TitanUI::print_info("  Written: " + info_output.get_msm_its_filename());
        TitanUI::log_kv(log, "ITS Scan Lags", std::to_string(lags_to_use.size()));
        const double matrix_sec =
            std::chrono::duration<double>(matrix_end - matrix_start).count();
        const double total_sec =
            std::chrono::duration<double>(its_end - its_start).count();
        TitanUI::log_time(log, "ITS Matrix+Spectral", matrix_sec);
        TitanUI::log_time(log, "ITS Total", total_sec);
        std::ostringstream solver_oss;
        bool first_mode = true;
        for (const auto &kv : solver_hist) {
          if (!first_mode)
            solver_oss << ", ";
          first_mode = false;
          solver_oss << kv.first << "=" << kv.second;
        }
        TitanUI::log_kv(log, "ITS Solver Usage",
                        solver_hist.empty() ? std::string("none")
                                            : solver_oss.str());
      }
    }
  } else {
    TitanUI::log_kv(log, "ITS Scan",
                    kinetics_interpretable
                        ? "disabled"
                        : "disabled (REMD state trajectories)");
  }
  // 7. PCCA crisp macrostate lumping
  // =========================================================================
  int n_macro_used = 0;
  if (msm_run_pcca && kinetics_interpretable) {
    TitanUI::log_kv(log, "PCCA Method",
                    "weighted spectral k-means on slow eigenmodes");
    // Determine n_macro
    if (msm_macrostates == -1) {
      n_macro_used = auto_detect_macrostates(ref_timescales);
      TitanUI::log_kv(log, "Auto-detected macrostates",
                      std::to_string(n_macro_used));
    } else {
      n_macro_used = std::clamp(msm_macrostates, 2, n_k_);
    }

    auto macrostates = run_pcca_crisp(n_macro_used);

    // Write macrostate assignments
    std::ofstream f(info_output.get_msm_macrostates_filename());
    if (!f.is_open()) {
      TitanUI::print_error("Cannot open " +
                           info_output.get_msm_macrostates_filename());
    } else {
      f << "# microstate_id  macrostate_id\n";
      for (int i = 0; i < n_k_; ++i) {
        f << i << "  " << macrostates[i] << '\n';
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_msm_macrostates_filename());
    }

    // Write macrostate trajectory (frame → macrostate, for Python scatter
    // plots)
    {
      std::ofstream fm(info_output.get_msm_macrostate_traj_filename());
      if (!fm.is_open()) {
        TitanUI::print_error("Cannot open " +
                             info_output.get_msm_macrostate_traj_filename());
      } else {
        fm << "# frame  microstate  macrostate\n";
        for (size_t i = 0; i < discrete_traj_.size(); ++i) {
          int micro = discrete_traj_[i];
          int macro = (micro < static_cast<int>(macrostates.size()))
                          ? macrostates[micro]
                          : -1;
          fm << i << "  " << micro << "  " << macro << "\n";
        }
        TitanUI::print_info("  Written: " +
                            info_output.get_msm_macrostate_traj_filename());
      }
    }

    TitanUI::log_kv(log, "PCCA Macrostates", std::to_string(n_macro_used));
  } else if (msm_run_pcca) {
    TitanUI::log_kv(log, "PCCA",
                    "disabled (requires interpretable kinetic eigenmodes)");
  } else {
    TitanUI::log_kv(log, "PCCA Macrostates", "disabled");
  }

  // =========================================================================
  // 8. Log summary
  // =========================================================================
  TitanUI::log_kv(log, "MSM States", std::to_string(n_k_));
  TitanUI::log_kv(log, "MSM Lag", std::to_string(msm_lag_time_));

  if (ref_timescales.size() >= 1)
    TitanUI::log_kv(log, "Slowest Timescale (ns)",
                    std::to_string(ref_timescales[0]));
  if (ref_timescales.size() >= 2)
    TitanUI::log_kv(log, "Second Timescale (ns)",
                    std::to_string(ref_timescales[1]));
}
