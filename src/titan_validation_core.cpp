#include "titan_markov_estimator.h"
#include "titan_validation_core.h"

#include "titan_ui.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <tuple>
#include <unordered_set>

namespace TitanValidation {

namespace {

Eigen::MatrixXd matrix_power(Eigen::MatrixXd base, int exponent) {
  if (exponent <= 0)
    return Eigen::MatrixXd::Identity(base.rows(), base.cols());
  Eigen::MatrixXd result =
      Eigen::MatrixXd::Identity(base.rows(), base.cols());
  for (int i = 0; i < exponent; ++i)
    result = result * base;
  return result;
}

ChapmanKolmogorovMetric
compute_ck_metric(const Eigen::MatrixXd &predicted_tpm,
                  const Eigen::MatrixXd &direct_counts, int multiple) {
  ChapmanKolmogorovMetric metric;
  metric.multiple = multiple;
  metric.direct_transition_count =
      static_cast<std::int64_t>(std::llround(direct_counts.sum()));

  Eigen::MatrixXd direct_tpm =
      Eigen::MatrixXd::Zero(direct_counts.rows(), direct_counts.cols());
  double weighted_abs = 0.0;
  double weighted_sq = 0.0;
  double total_weight = 0.0;

  for (int i = 0; i < direct_counts.rows(); ++i) {
    const double row_weight = direct_counts.row(i).sum();
    if (row_weight <= 1e-12)
      continue;
    direct_tpm.row(i) = direct_counts.row(i) / row_weight;
    const Eigen::RowVectorXd diff = direct_tpm.row(i) - predicted_tpm.row(i);
    ChapmanKolmogorovRowError row;
    row.multiple = multiple;
    row.state = i;
    row.row_weight = row_weight;
    row.row_mae = diff.cwiseAbs().sum() / static_cast<double>(diff.size());
    row.row_frobenius = std::sqrt(std::max(0.0, diff.squaredNorm()));
    row.row_max_abs = diff.cwiseAbs().maxCoeff();
    metric.row_errors.push_back(row);
    weighted_abs += row.row_mae * row_weight;
    weighted_sq += diff.squaredNorm() * row_weight;
    total_weight += row_weight;
    metric.max_abs = std::max(metric.max_abs, row.row_max_abs);
  }

  metric.rows_compared = static_cast<int>(metric.row_errors.size());
  if (metric.rows_compared > 0 && total_weight > 0.0) {
    metric.weighted_mae = weighted_abs / total_weight;
    metric.weighted_frobenius = std::sqrt(weighted_sq / total_weight);
  }
  return metric;
}

bool better_candidate(const VampCandidateResult &lhs,
                      const VampCandidateResult &rhs) {
  const bool lhs_valid = std::isfinite(lhs.score_mean);
  const bool rhs_valid = std::isfinite(rhs.score_mean);
  if (lhs_valid != rhs_valid)
    return lhs_valid;
  if (!lhs_valid && !rhs_valid)
    return false;
  if (lhs.score_mean != rhs.score_mean)
    return lhs.score_mean > rhs.score_mean;
  if (lhs.score_std != rhs.score_std)
    return lhs.score_std < rhs.score_std;
  if (lhs.config.n_clusters != rhs.config.n_clusters)
    return lhs.config.n_clusters < rhs.config.n_clusters;
  if (lhs.config.lag_frames != rhs.config.lag_frames)
    return lhs.config.lag_frames < rhs.config.lag_frames;
  return lhs.config.components < rhs.config.components;
}

unsigned int fold_seed(unsigned int base_seed, const ModelConfig &config) {
  unsigned int seed = base_seed;
  seed ^= static_cast<unsigned int>(config.components * 73856093u);
  seed ^= static_cast<unsigned int>(config.lag_frames * 19349663u);
  seed ^= static_cast<unsigned int>(config.n_clusters * 83492791u);
  return seed;
}

std::unique_ptr<TitanMSM>
build_candidate_model(const TitanMSM &base_msm, const ModelConfig &config,
                      int kmeans_iter, unsigned int seed,
                      TitanExecutor &executor) {
  if (config.components <= 0 || config.n_clusters <= 1 || config.lag_frames <= 0)
    return nullptr;
  if (config.components > base_msm.n_components())
    return nullptr;
  if (base_msm.n_frames() <= 0 ||
      base_msm.n_frames() < static_cast<int64_t>(config.n_clusters))
    return nullptr;

  auto candidate = std::make_unique<TitanMSM>(config.n_clusters,
                                              config.components,
                                              base_msm.dt_ns());
  const auto &proj = base_msm.proj_mat_ref();
  const auto &replica_ids = base_msm.get_frame_replica_ids();
  const auto &timestamps = base_msm.get_frame_timestamps();
  const auto &volumes = base_msm.get_frame_box_volumes_nm3();
  const auto &global_indices = base_msm.get_frame_global_indices();
  const auto &breaks = base_msm.traj_break_indices();
  std::size_t break_cursor = 0;

  for (int64_t frame = 0; frame < base_msm.n_frames(); ++frame) {
    const Eigen::VectorXd prefix =
        proj.row(frame).leftCols(config.components).transpose();
    const int replica_id =
        frame < static_cast<int64_t>(replica_ids.size()) ? replica_ids[frame] : -1;
    const int64_t global_index =
        frame < static_cast<int64_t>(global_indices.size())
            ? global_indices[frame]
            : frame;
    const double volume =
        frame < static_cast<int64_t>(volumes.size())
            ? volumes[frame]
            : std::numeric_limits<double>::quiet_NaN();
    candidate->add_projected_frame(prefix, replica_id, global_index, volume);
    if (frame < static_cast<int64_t>(timestamps.size()))
      candidate->record_frame_time(timestamps[frame], replica_id);
    while (break_cursor < breaks.size() &&
           breaks[break_cursor] == static_cast<int>(frame + 1)) {
      candidate->add_trajectory_break();
      ++break_cursor;
    }
  }

  candidate->train_kmeans(executor, kmeans_iter, seed);
  if (!candidate->is_clustered())
    return nullptr;
  candidate->build_markov_model(config.lag_frames);
  if (!candidate->is_built())
    return nullptr;
  return candidate;
}

struct FrameRange {
  int64_t begin{0};
  int64_t end{0};
};

struct FoldDiscreteData {
  bool valid{false};
  StateSegments train_segments;
  StateSegments test_segments;
};

std::vector<FrameRange> build_frame_ranges(const TitanMSM &msm) {
  std::vector<FrameRange> ranges;
  const int64_t n_frames = msm.n_frames();
  if (n_frames <= 0)
    return ranges;

  int64_t begin = 0;
  for (int raw_boundary : msm.traj_break_indices()) {
    const int64_t boundary = std::clamp<int64_t>(raw_boundary, 0, n_frames);
    if (boundary > begin)
      ranges.push_back({begin, boundary});
    begin = std::max(begin, boundary);
  }
  if (begin < n_frames)
    ranges.push_back({begin, n_frames});
  return ranges;
}

int nearest_centroid(const TitanMSM::ProjMatrixRM &projections, int64_t frame,
                     int components,
                     const TitanMSM::DenseMatrixRM &centroids) {
  int best_cluster = -1;
  double best_distance = std::numeric_limits<double>::infinity();
  for (int cluster = 0; cluster < centroids.rows(); ++cluster) {
    double distance = 0.0;
    for (int component = 0; component < components; ++component) {
      const double delta = projections(frame, component) -
                           centroids(cluster, component);
      distance += delta * delta;
    }
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }
  return best_cluster;
}

FoldDiscreteData build_fold_discrete_data(
    const TitanMSM &base_msm, const ModelConfig &config,
    const std::vector<FrameRange> &ranges, int held_out_fold, int n_folds,
    int kmeans_iter, unsigned int seed, TitanExecutor &executor) {
  FoldDiscreteData fold;
  if (ranges.size() < 2 || n_folds < 2)
    return fold;

  int64_t train_frames = 0;
  for (std::size_t segment = 0; segment < ranges.size(); ++segment) {
    if (static_cast<int>(segment % static_cast<std::size_t>(n_folds)) !=
        held_out_fold) {
      train_frames += ranges[segment].end - ranges[segment].begin;
    }
  }
  if (train_frames < config.n_clusters)
    return fold;

  TitanMSM train_model(config.n_clusters, config.components, base_msm.dt_ns());
  const auto &projections = base_msm.proj_mat_ref();
  for (std::size_t segment = 0; segment < ranges.size(); ++segment) {
    if (static_cast<int>(segment % static_cast<std::size_t>(n_folds)) ==
        held_out_fold) {
      continue;
    }
    for (int64_t frame = ranges[segment].begin; frame < ranges[segment].end;
         ++frame) {
      train_model.add_projected_frame(
          projections.row(frame).leftCols(config.components).transpose());
    }
    train_model.add_trajectory_break();
  }
  train_model.train_kmeans(executor, kmeans_iter,
                           seed ^ static_cast<unsigned int>(held_out_fold + 1) *
                                      0x9e3779b9u);
  if (!train_model.is_clustered())
    return fold;

  const auto &centroids = train_model.centroids();
  for (std::size_t segment = 0; segment < ranges.size(); ++segment) {
    std::vector<int> assignments;
    assignments.reserve(static_cast<std::size_t>(ranges[segment].end -
                                                  ranges[segment].begin));
    for (int64_t frame = ranges[segment].begin; frame < ranges[segment].end;
         ++frame) {
      assignments.push_back(nearest_centroid(
          projections, frame, config.components, centroids));
    }
    if (static_cast<int>(segment % static_cast<std::size_t>(n_folds)) ==
        held_out_fold) {
      fold.test_segments.push_back(std::move(assignments));
    } else {
      fold.train_segments.push_back(std::move(assignments));
    }
  }
  fold.valid = !fold.train_segments.empty() && !fold.test_segments.empty();
  return fold;
}

Eigen::MatrixXd symmetric_inverse_sqrt(const Eigen::MatrixXd &matrix) {
  if (matrix.rows() == 0 || matrix.rows() != matrix.cols())
    return {};
  const Eigen::MatrixXd sym = 0.5 * (matrix + matrix.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sym);
  if (solver.info() != Eigen::Success)
    return Eigen::MatrixXd::Zero(matrix.rows(), matrix.cols());
  const double max_eval = std::max(0.0, solver.eigenvalues().maxCoeff());
  const double cutoff = std::max(1e-12, max_eval * 1e-10);
  Eigen::VectorXd inv = Eigen::VectorXd::Zero(matrix.rows());
  for (int i = 0; i < inv.size(); ++i) {
    if (solver.eigenvalues()[i] > cutoff)
      inv[i] = 1.0 / std::sqrt(solver.eigenvalues()[i]);
  }
  return solver.eigenvectors() * inv.asDiagonal() *
         solver.eigenvectors().transpose();
}

double compute_held_out_vamp2_score(const StateSegments &train_segments,
                                    const StateSegments &test_segments,
                                    int n_states, int lag_frames,
                                    int score_rank) {
  Eigen::MatrixXd train_counts;
  Eigen::VectorXd train_occupancy;
  std::int64_t train_transitions = 0;
  std::string error;
  if (!build_segment_count_matrix(n_states, lag_frames, train_segments,
                                  train_counts, train_occupancy,
                                  train_transitions, error) ||
      train_transitions <= 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  Eigen::MatrixXd test_counts;
  Eigen::VectorXd test_occupancy;
  std::int64_t test_transitions = 0;
  if (!build_segment_count_matrix(n_states, lag_frames, test_segments,
                                  test_counts, test_occupancy, test_transitions,
                                  error) ||
      test_transitions <= 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double inv_train = 1.0 / static_cast<double>(train_transitions);
  const Eigen::MatrixXd train_c01 = train_counts * inv_train;
  const Eigen::MatrixXd train_c00 =
      (train_counts.rowwise().sum() * inv_train).asDiagonal();
  const Eigen::MatrixXd train_c11 =
      (train_counts.colwise().sum().transpose() * inv_train).asDiagonal();
  const Eigen::MatrixXd train_w0 = symmetric_inverse_sqrt(train_c00);
  const Eigen::MatrixXd train_w1 = symmetric_inverse_sqrt(train_c11);
  const Eigen::MatrixXd train_koopman = train_w0 * train_c01 * train_w1;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(train_koopman,
                                        Eigen::ComputeThinU |
                                            Eigen::ComputeThinV);
  if (svd.info() != Eigen::Success || svd.singularValues().size() == 0)
    return std::numeric_limits<double>::quiet_NaN();
  const int rank = std::max(
      1, std::min({score_rank, static_cast<int>(svd.singularValues().size()),
                   n_states}));
  const Eigen::MatrixXd left_basis =
      train_w0 * svd.matrixU().leftCols(rank);
  const Eigen::MatrixXd right_basis =
      train_w1 * svd.matrixV().leftCols(rank);

  const double inv_test = 1.0 / static_cast<double>(test_transitions);
  const Eigen::MatrixXd test_c01 = test_counts * inv_test;
  const Eigen::MatrixXd test_c00 =
      (test_counts.rowwise().sum() * inv_test).asDiagonal();
  const Eigen::MatrixXd test_c11 =
      (test_counts.colwise().sum().transpose() * inv_test).asDiagonal();
  const Eigen::MatrixXd projected_c00 =
      left_basis.transpose() * test_c00 * left_basis;
  const Eigen::MatrixXd projected_c11 =
      right_basis.transpose() * test_c11 * right_basis;
  const Eigen::MatrixXd projected_c01 =
      left_basis.transpose() * test_c01 * right_basis;
  const Eigen::MatrixXd normalized_test =
      symmetric_inverse_sqrt(projected_c00) * projected_c01 *
      symmetric_inverse_sqrt(projected_c11);
  const double score = normalized_test.squaredNorm();
  return std::isfinite(score) ? score
                              : std::numeric_limits<double>::quiet_NaN();
}

void append_if_finite(std::vector<double> &out, double value) {
  if (std::isfinite(value))
    out.push_back(value);
}

} // namespace

StateSegments build_msm_state_segments(const TitanMSM &msm_analyzer) {
  StateSegments segments;
  const auto &dtraj = msm_analyzer.discrete_traj();
  const int n_frames = static_cast<int>(dtraj.size());
  if (n_frames <= 0)
    return segments;

  int start = 0;
  for (int stop : msm_analyzer.traj_break_indices()) {
    if (stop <= start || stop > n_frames)
      continue;
    segments.emplace_back(dtraj.begin() + start, dtraj.begin() + stop);
    start = stop;
  }
  if (start < n_frames)
    segments.emplace_back(dtraj.begin() + start, dtraj.end());
  if (segments.empty())
    segments.emplace_back(dtraj.begin(), dtraj.end());
  return segments;
}

bool build_segment_count_matrix(int n_states, int lag_frames,
                                const StateSegments &segments,
                                Eigen::MatrixXd &out_counts,
                                Eigen::VectorXd &out_occupancy,
                                std::int64_t &out_transition_count,
                                std::string &error_message) {
  out_counts = Eigen::MatrixXd::Zero(n_states, n_states);
  out_occupancy = Eigen::VectorXd::Zero(n_states);
  out_transition_count = 0;
  if (n_states <= 0) {
    error_message = "at least one state is required";
    return false;
  }
  if (lag_frames <= 0) {
    error_message = "lag_frames must be > 0";
    return false;
  }

  for (const auto &segment : segments) {
    for (int state : segment) {
      if (state >= 0 && state < n_states)
        out_occupancy[state] += 1.0;
    }
    if (segment.size() <= static_cast<std::size_t>(lag_frames))
      continue;
    for (std::size_t i = 0;
         i + static_cast<std::size_t>(lag_frames) < segment.size(); ++i) {
      const int state_i = segment[i];
      const int state_j = segment[i + static_cast<std::size_t>(lag_frames)];
      if (state_i < 0 || state_i >= n_states || state_j < 0 ||
          state_j >= n_states) {
        continue;
      }
      out_counts(state_i, state_j) += 1.0;
      ++out_transition_count;
    }
  }

  if (out_transition_count <= 0) {
    error_message = "segments contain no lag-valid transitions";
    return false;
  }
  return true;
}

bool build_markov_model_from_segments(int n_states, int lag_frames,
                                      const StateSegments &segments,
                                      bool symmetrize_counts,
                                      Eigen::MatrixXd &out_tpm,
                                      Eigen::VectorXd &out_pi,
                                      std::int64_t &out_transition_count,
                                      std::string &error_message) {
  out_tpm = Eigen::MatrixXd::Zero(n_states, n_states);
  out_pi = Eigen::VectorXd::Zero(n_states);

  Eigen::MatrixXd counts;
  Eigen::VectorXd occupancy;
  if (!build_segment_count_matrix(n_states, lag_frames, segments, counts,
                                  occupancy, out_transition_count,
                                  error_message)) {
    return false;
  }

  if (symmetrize_counts)
    return titan_markov::reversible_mle(counts, out_tpm, out_pi, error_message);

  for (int i=0;i<n_states;++i) {
    const double row_sum=counts.row(i).sum();
    if (row_sum<=0) {
      error_message="unvisited state in nonreversible transition counts";
      return false;
    }
    out_tpm.row(i)=counts.row(i)/row_sum;
  }
  return titan_markov::stationary(out_tpm,out_pi,error_message);
}

bool build_dham_model_from_context(const DhamRebuildContext &context,
                                   int lag_frames, Eigen::MatrixXd &out_tpm,
                                   Eigen::VectorXd &out_pi,
                                   std::int64_t &out_transition_count,
                                   std::string &error_message,
                                   Eigen::MatrixXd *out_counts) {
  out_tpm.resize(0, 0);
  out_pi.resize(0);
  out_transition_count = 0;
  if (out_counts)
    out_counts->resize(0, 0);

  if (!context.available) {
    error_message = "DHAM rebuild context is unavailable";
    return false;
  }
  if (lag_frames <= 0) {
    error_message = "DHAM lag_frames must be > 0";
    return false;
  }

  const auto state_observables = TitanDHAM::compute_state_energies(
      context.per_replica_dtrajs, context.per_replica_observables,
      context.n_states, context.n_replicas, 1);
  TitanDHAM dham(context.n_states, context.n_replicas, context.temps_k,
                 context.target_temp_k);
  std::vector<std::vector<unsigned char>> valid_observable(
      static_cast<std::size_t>(context.n_replicas),
      std::vector<unsigned char>(static_cast<std::size_t>(context.n_states), 0));

  for (int replica = 0; replica < context.n_replicas; ++replica) {
    for (int state = 0; state < context.n_states; ++state) {
      const double value = state_observables(state, replica);
      if (!std::isfinite(value))
        continue;
      valid_observable[static_cast<std::size_t>(replica)]
                      [static_cast<std::size_t>(state)] = 1;
      dham.set_state_energy(replica, state, value);
    }
  }

  Eigen::MatrixXd counts = Eigen::MatrixXd::Zero(context.n_states, context.n_states);
  for (int replica = 0; replica < context.n_replicas; ++replica) {
    auto it_dtraj = context.per_replica_dtrajs.find(replica);
    auto it_time = context.per_replica_timestamps_ps.find(replica);
    if (it_dtraj == context.per_replica_dtrajs.end() ||
        it_time == context.per_replica_timestamps_ps.end()) {
      continue;
    }
    const auto &dtraj = it_dtraj->second;
    const auto &times = it_time->second;
    if (dtraj.size() <= static_cast<std::size_t>(lag_frames) ||
        times.size() != dtraj.size()) {
      continue;
    }

    double expected_gap = static_cast<double>(lag_frames);
    if (times.size() > 1) {
      const double dt = times[1] - times[0];
      expected_gap = dt * lag_frames;
    }

    for (std::size_t i = 0; i + static_cast<std::size_t>(lag_frames) < dtraj.size();
         ++i) {
      const double actual_gap = times[i + static_cast<std::size_t>(lag_frames)] -
                                times[i];
      if (std::abs(actual_gap - expected_gap) >
          0.1 * std::max(1e-12, std::abs(expected_gap))) {
        continue;
      }
      const int state_i = dtraj[i];
      const int state_j = dtraj[i + static_cast<std::size_t>(lag_frames)];
      if (state_i < 0 || state_i >= context.n_states || state_j < 0 ||
          state_j >= context.n_states) {
        continue;
      }
      if (!valid_observable[static_cast<std::size_t>(replica)]
                           [static_cast<std::size_t>(state_i)] ||
          !valid_observable[static_cast<std::size_t>(replica)]
                           [static_cast<std::size_t>(state_j)]) {
        continue;
      }
      dham.accumulate_counts(replica, state_i, state_j);
      counts(state_i, state_j) += 1.0;
      ++out_transition_count;
    }
  }

  if (out_transition_count <= 0) {
    error_message = "DHAM rebuild context contains no lag-valid transitions";
    return false;
  }
  if (!dham.solve(context.max_iter, context.tolerance)) {
    error_message = "DHAM solve() did not converge during validation rebuild";
    return false;
  }
  out_tpm = dham.get_unbiased_transition_matrix();
  out_pi = dham.get_stationary_distribution();
  if (out_counts)
    *out_counts = counts;
  return true;
}

ChapmanKolmogorovSummary
compute_ck_for_segments(int n_states, int lag_frames, const StateSegments &segments,
                        bool symmetrize_counts,
                        const std::vector<int> &multiples,
                        const std::string &model_source) {
  ChapmanKolmogorovSummary summary;
  summary.lag_frames = lag_frames;
  summary.model_source = model_source;

  Eigen::MatrixXd p_tau;
  Eigen::VectorXd pi_tau;
  std::int64_t transition_count = 0;
  if (!build_markov_model_from_segments(n_states, lag_frames, segments,
                                        symmetrize_counts, p_tau, pi_tau,
                                        transition_count,
                                        summary.error_message)) {
    return summary;
  }

  summary = compute_ck_for_transition_model(p_tau, lag_frames, n_states, segments,
                                            multiples, model_source);
  return summary;
}

ChapmanKolmogorovSummary
compute_ck_for_transition_model(const Eigen::MatrixXd &p_tau,
                                int lag_frames, int n_states,
                                const StateSegments &segments,
                                const std::vector<int> &multiples,
                                const std::string &model_source) {
  ChapmanKolmogorovSummary summary;
  summary.lag_frames = lag_frames;
  summary.model_source = model_source;
  if (p_tau.rows() != n_states || p_tau.cols() != n_states) {
    summary.error_message = "invalid base transition matrix dimensions";
    return summary;
  }

  for (int multiple : multiples) {
    if (multiple <= 1)
      continue;
    Eigen::MatrixXd direct_counts;
    Eigen::VectorXd occupancy;
    std::int64_t direct_transitions = 0;
    std::string error;
    if (!build_segment_count_matrix(n_states, lag_frames * multiple, segments,
                                    direct_counts, occupancy,
                                    direct_transitions, error)) {
      continue;
    }
    const Eigen::MatrixXd predicted = matrix_power(p_tau, multiple);
    auto metric = compute_ck_metric(predicted, direct_counts, multiple);
    metric.direct_transition_count = direct_transitions;
    summary.metrics.push_back(std::move(metric));
  }

  summary.valid = !summary.metrics.empty();
  if (!summary.valid && summary.error_message.empty())
    summary.error_message = "no CK multiples produced valid direct transitions";
  return summary;
}

ChapmanKolmogorovSummary
compute_ck_for_dham(const DhamRebuildContext &context,
                    const Eigen::MatrixXd &p_tau, int lag_frames,
                    const std::vector<int> &multiples,
                    const std::string &model_source) {
  ChapmanKolmogorovSummary summary;
  summary.lag_frames = lag_frames;
  summary.model_source = model_source;
  for (int multiple : multiples) {
    if (multiple <= 1)
      continue;
    Eigen::MatrixXd direct_tpm;
    Eigen::VectorXd direct_pi;
    Eigen::MatrixXd direct_counts;
    std::int64_t direct_transitions = 0;
    std::string error;
    if (!build_dham_model_from_context(context, lag_frames * multiple, direct_tpm,
                                       direct_pi, direct_transitions, error,
                                       &direct_counts)) {
      continue;
    }
    const Eigen::MatrixXd predicted = matrix_power(p_tau, multiple);
    auto metric = compute_ck_metric(predicted, direct_counts, multiple);
    metric.direct_transition_count = direct_transitions;
    summary.metrics.push_back(std::move(metric));
  }
  summary.valid = !summary.metrics.empty();
  if (!summary.valid)
    summary.error_message = "no DHAM CK multiples converged";
  return summary;
}

double compute_discrete_vamp2_score(const StateSegments &segments, int n_states,
                                    int lag_frames) {
  Eigen::MatrixXd counts;
  Eigen::VectorXd occupancy;
  std::int64_t transition_count = 0;
  std::string error;
  if (!build_segment_count_matrix(n_states, lag_frames, segments, counts,
                                  occupancy, transition_count, error) ||
      transition_count <= 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  Eigen::VectorXd row_sum = counts.rowwise().sum();
  Eigen::RowVectorXd col_sum = counts.colwise().sum();
  double score = 0.0;
  for (int i = 0; i < n_states; ++i) {
    if (row_sum[i] <= 1e-12)
      continue;
    for (int j = 0; j < n_states; ++j) {
      if (col_sum[j] <= 1e-12)
        continue;
      const double cij = counts(i, j);
      if (cij <= 0.0)
        continue;
      score += (cij * cij) / (row_sum[i] * col_sum[j]);
    }
  }
  return score;
}

VampSweepResult evaluate_vamp_candidates(
    const TitanMSM &base_msm, int base_components,
    const std::vector<ModelConfig> &candidates, int requested_folds,
    int kmeans_iter, unsigned int base_seed, TitanExecutor &executor) {
  VampSweepResult sweep;
  VampCandidateResult best_result;
  bool have_best = false;
  const auto ranges = build_frame_ranges(base_msm);
  using FoldKey = std::tuple<int, int, int, int>;
  std::map<FoldKey, FoldDiscreteData> fold_cache;

  for (const auto &candidate_config : candidates) {
    VampCandidateResult candidate_result;
    candidate_result.config = candidate_config;
    candidate_result.folds_requested = requested_folds;
    if (candidate_config.components <= 0 ||
        candidate_config.components > base_components ||
        candidate_config.n_clusters <= 1 || candidate_config.lag_frames <= 0 ||
        ranges.size() < 2) {
      candidate_result.status = "build_failed";
      sweep.candidates.push_back(candidate_result);
      continue;
    }

    candidate_result.folds_effective =
        std::max(1, std::min(requested_folds, static_cast<int>(ranges.size())));

    std::vector<double> fold_scores;
    fold_scores.reserve(
        static_cast<std::size_t>(candidate_result.folds_effective));
    for (int fold_index = 0; fold_index < candidate_result.folds_effective;
         ++fold_index) {
      const FoldKey key{candidate_config.components,
                        candidate_config.n_clusters,
                        candidate_result.folds_effective, fold_index};
      auto [it, inserted] = fold_cache.try_emplace(key);
      if (inserted) {
        ModelConfig clustering_config = candidate_config;
        clustering_config.lag_frames = 1;
        it->second = build_fold_discrete_data(
            base_msm, clustering_config, ranges, fold_index,
            candidate_result.folds_effective, kmeans_iter,
            fold_seed(base_seed, clustering_config), executor);
      }
      if (!it->second.valid)
        continue;
      const double score = compute_held_out_vamp2_score(
          it->second.train_segments, it->second.test_segments,
          candidate_config.n_clusters, candidate_config.lag_frames,
          candidate_config.components);
      if (std::isfinite(score))
        fold_scores.push_back(score);
    }

    candidate_result.folds_scored = static_cast<int>(fold_scores.size());
    if (!fold_scores.empty()) {
      candidate_result.score_mean =
          std::accumulate(fold_scores.begin(), fold_scores.end(), 0.0) /
          static_cast<double>(fold_scores.size());
      if (fold_scores.size() > 1) {
        double var = 0.0;
        for (double score : fold_scores)
          var += (score - candidate_result.score_mean) *
                 (score - candidate_result.score_mean);
        var /= static_cast<double>(fold_scores.size() - 1);
        candidate_result.score_std = std::sqrt(std::max(0.0, var));
      } else {
        candidate_result.score_std = 0.0;
      }
    }

    if (candidate_result.folds_scored <= 0) {
      candidate_result.status = "insufficient";
    } else if (candidate_result.folds_effective < requested_folds ||
               candidate_result.folds_effective < 2) {
      candidate_result.status = "exploratory";
    } else {
      candidate_result.status = "ok";
    }

    if (candidate_result.folds_scored > 0 &&
        (!have_best || better_candidate(candidate_result, best_result))) {
      have_best = true;
      best_result = candidate_result;
    }
    sweep.candidates.push_back(candidate_result);
  }

  if (have_best) {
    sweep.selected = best_result.config;
    sweep.winning_model = build_candidate_model(
        base_msm, sweep.selected, kmeans_iter,
        fold_seed(base_seed, sweep.selected), executor);
  } else {
    sweep.warning_message = "no VAMP candidate produced a valid score";
  }

  sweep.valid = have_best && static_cast<bool>(sweep.winning_model);
  if (!sweep.valid)
    sweep.winning_model.reset();
  return sweep;
}

TptKineticDiagnostics compute_tpt_kinetic_diagnostics_from_segments(
    const StateSegments &segments, int n_states, int lag_frames,
    double lag_time_ns, const TitanTPT &tpt_analyzer) {
  TptKineticDiagnostics diag;
  if (segments.empty())
    return diag;

  std::vector<unsigned char> basin_map(
      static_cast<std::size_t>(n_states),
      static_cast<unsigned char>(0));
  for (int state : tpt_analyzer.source_states()) {
    if (state >= 0 && state < n_states)
      basin_map[static_cast<std::size_t>(state)] = 1;
  }
  for (int state : tpt_analyzer.sink_states()) {
    if (state >= 0 && state < n_states)
      basin_map[static_cast<std::size_t>(state)] = 2;
  }

  diag.n_segments = static_cast<int>(segments.size());
  if (!segments.empty())
    diag.min_segment_length_frames = static_cast<int>(segments.front().size());

  for (const auto &segment : segments) {
    const int seg_len = static_cast<int>(segment.size());
    diag.total_frames += seg_len;
    diag.min_segment_length_frames =
        std::min(diag.min_segment_length_frames, seg_len);
    diag.max_segment_length_frames =
        std::max(diag.max_segment_length_frames, seg_len);

    bool seen_source = false;
    bool seen_sink = false;
    int last_committed = -1;
    int previous_label = -1;

    for (int state : segment) {
      const int label =
          (state >= 0 && state < n_states)
              ? static_cast<int>(basin_map[static_cast<std::size_t>(state)])
              : 0;
      if (label == 1) {
        ++diag.frames_source;
        seen_source = true;
      } else if (label == 2) {
        ++diag.frames_sink;
        seen_sink = true;
      } else {
        ++diag.frames_intermediate;
      }

      if ((label == 1 || label == 2) && label != previous_label) {
        if (label == 1 && last_committed == 2) {
          ++diag.observed_b_to_a_events;
        } else if (label == 2 && last_committed == 1) {
          ++diag.observed_a_to_b_events;
        }
        last_committed = label;
      }
      previous_label = label;
    }

    if (seen_source)
      ++diag.segments_with_source;
    if (seen_sink)
      ++diag.segments_with_sink;
    if (seen_source && seen_sink)
      ++diag.segments_with_both;
  }

  if (diag.n_segments > 0) {
    diag.mean_segment_length_frames =
        static_cast<double>(diag.total_frames) / diag.n_segments;
  }
  if (diag.total_frames > 0) {
    diag.observed_source_fraction =
        static_cast<double>(diag.frames_source) / diag.total_frames;
    diag.observed_sink_fraction =
        static_cast<double>(diag.frames_sink) / diag.total_frames;
    diag.observed_intermediate_fraction =
        static_cast<double>(diag.frames_intermediate) / diag.total_frames;
  }

  diag.observed_round_trips =
      std::min(diag.observed_a_to_b_events, diag.observed_b_to_a_events);
  diag.lag_time_ns = lag_time_ns;
  diag.model_rate_ab_per_lag =
      tpt_analyzer.summary().rate_source_to_sink_per_lag;
  diag.model_mfpt_ab_lags = tpt_analyzer.summary().mfpt_source_to_sink_lags;
  if (diag.lag_time_ns > 0.0) {
    diag.model_rate_ab_per_ns =
        diag.model_rate_ab_per_lag / diag.lag_time_ns;
    if (std::isfinite(diag.model_mfpt_ab_lags))
      diag.model_mfpt_ab_ns = diag.model_mfpt_ab_lags * diag.lag_time_ns;
  }

  const auto ck_summary =
      compute_ck_for_segments(n_states, lag_frames, segments, false, {2},
                              "kinetic_segments");
  if (ck_summary.valid && !ck_summary.metrics.empty()) {
    const auto &metric = ck_summary.metrics.front();
    diag.ck_rows_compared = metric.rows_compared;
    diag.ck_mean_abs_error = metric.weighted_mae;
    diag.ck_max_abs_error = metric.max_abs;
    diag.ck_weighted_frobenius = metric.weighted_frobenius;
    if (diag.ck_weighted_frobenius > 0.25 || diag.ck_max_abs_error > 0.50) {
      diag.markov_sanity = "fail";
    } else if (diag.ck_weighted_frobenius > 0.10 ||
               diag.ck_max_abs_error > 0.25) {
      diag.markov_sanity = "weak";
    } else {
      diag.markov_sanity = "pass";
    }
  }

  const bool too_few_frames =
      diag.total_frames < std::max<int64_t>(10, 10LL * lag_frames);
  const bool no_forward = diag.observed_a_to_b_events == 0;
  const bool no_backward = diag.observed_b_to_a_events == 0;
  const bool low_source_occ = diag.observed_source_fraction < 0.01;
  const bool low_sink_occ = diag.observed_sink_fraction < 0.01;
  const bool ck_fail = diag.markov_sanity == "fail";
  const bool ck_weak = diag.markov_sanity == "weak";

  if (too_few_frames) {
    diag.suitability = "insufficient: too few frames for kinetic diagnostics";
  } else if (no_forward) {
    diag.suitability = "insufficient: no observed A->B events";
  } else if (no_backward) {
    diag.suitability = "insufficient: no observed B->A back-transitions";
  } else if (ck_fail) {
    diag.suitability =
        "weak: CK / Markov consistency check indicates poor kinetic stability";
  } else if (low_source_occ || low_sink_occ) {
    diag.suitability = "weak: basin occupancy is very imbalanced";
  } else if (ck_weak) {
    diag.suitability =
        "exploratory: CK / Markov consistency is only moderate";
  } else if (diag.observed_round_trips < 3) {
    diag.suitability = "exploratory: limited round-trips between basins";
  } else {
    diag.suitability = "acceptable: exploratory kinetic-TPT conditions met";
  }

  return diag;
}

bool run_tpt_bootstrap(const StateSegments &base_segments, int n_states,
                       int lag_frames, double lag_time_ns,
                       bool symmetrize_counts,
                       const std::vector<int> &source_states,
                       const std::vector<int> &sink_states, int top_paths,
                       double flux_cut, int n_threads, int n_samples,
                       unsigned int seed, TitanExecutor &executor,
                       TptBootstrapAggregate &out_bootstrap,
                       std::string &error_message) {
  (void)lag_time_ns;
  out_bootstrap = {};
  out_bootstrap.requested_samples = n_samples;
  if (n_samples <= 0)
    return true;
  if (base_segments.empty()) {
    error_message = "no trajectory segments are available for TPT bootstrap";
    return false;
  }
  if (base_segments.size() < 2) {
    error_message =
        "at least two trajectory segments are recommended for bootstrap";
    return false;
  }

  out_bootstrap.q_forward_sum = Eigen::VectorXd::Zero(n_states);
  out_bootstrap.q_forward_sq_sum = Eigen::VectorXd::Zero(n_states);
  out_bootstrap.reactive_density_sum = Eigen::VectorXd::Zero(n_states);
  out_bootstrap.reactive_density_sq_sum = Eigen::VectorXd::Zero(n_states);

  struct LocalAccum {
    int successful_samples{0};
    std::vector<double> total_reactive_flux;
    std::vector<double> rate_source_to_sink_per_lag;
    std::vector<double> mfpt_source_to_sink_lags;
    std::vector<double> source_population;
    std::vector<double> sink_population;
    Eigen::VectorXd q_forward_sum;
    Eigen::VectorXd q_forward_sq_sum;
    Eigen::VectorXd reactive_density_sum;
    Eigen::VectorXd reactive_density_sq_sum;

    explicit LocalAccum(int n_states_in)
        : q_forward_sum(Eigen::VectorXd::Zero(n_states_in)),
          q_forward_sq_sum(Eigen::VectorXd::Zero(n_states_in)),
          reactive_density_sum(Eigen::VectorXd::Zero(n_states_in)),
          reactive_density_sq_sum(Eigen::VectorXd::Zero(n_states_in)) {}
  };

  std::mutex merge_mutex;
  const int worker_count = std::max(1, std::min(n_threads, n_samples));
  const int chunk = (n_samples + worker_count - 1) / worker_count;
  executor.parallel_do(static_cast<size_t>(worker_count), [&](size_t worker_sz) {
    const int worker = static_cast<int>(worker_sz);
    const int begin = worker * chunk;
    const int end = std::min(n_samples, begin + chunk);
    if (begin >= end)
      return;

    LocalAccum local(n_states);
    std::mt19937 rng(seed + static_cast<unsigned int>(worker * 7919 + 17));
    std::uniform_int_distribution<int> seg_pick(
        0, static_cast<int>(base_segments.size()) - 1);
    StateSegments sampled_segments;
    sampled_segments.reserve(base_segments.size());

    for (int sample_idx = begin; sample_idx < end; ++sample_idx) {
      (void)sample_idx;
      sampled_segments.clear();
      for (std::size_t seg = 0; seg < base_segments.size(); ++seg) {
        sampled_segments.push_back(
            base_segments[static_cast<std::size_t>(seg_pick(rng))]);
      }

      Eigen::MatrixXd sample_tpm;
      Eigen::VectorXd sample_pi;
      std::int64_t sample_transitions = 0;
      std::string sample_error;
      if (!build_markov_model_from_segments(n_states, lag_frames,
                                            sampled_segments,
                                            symmetrize_counts, sample_tpm,
                                            sample_pi, sample_transitions,
                                            sample_error)) {
        continue;
      }

      try {
        TitanTPT sample_tpt(sample_tpm, sample_pi);
        sample_tpt.compute(source_states, sink_states, top_paths, flux_cut,
                           nullptr, 1);
        const auto &summary = sample_tpt.summary();
        append_if_finite(local.total_reactive_flux,
                         summary.total_reactive_flux);
        append_if_finite(local.rate_source_to_sink_per_lag,
                         summary.rate_source_to_sink_per_lag);
        append_if_finite(local.mfpt_source_to_sink_lags,
                         summary.mfpt_source_to_sink_lags);
        append_if_finite(local.source_population, summary.source_population);
        append_if_finite(local.sink_population, summary.sink_population);
        local.q_forward_sum += sample_tpt.forward_committor();
        local.q_forward_sq_sum +=
            sample_tpt.forward_committor().array().square().matrix();
        local.reactive_density_sum += sample_tpt.reactive_density();
        local.reactive_density_sq_sum +=
            sample_tpt.reactive_density().array().square().matrix();
        ++local.successful_samples;
      } catch (const std::exception &) {
        continue;
      }
    }

    std::lock_guard<std::mutex> lock(merge_mutex);
    out_bootstrap.successful_samples += local.successful_samples;
    out_bootstrap.total_reactive_flux.insert(
        out_bootstrap.total_reactive_flux.end(),
        local.total_reactive_flux.begin(), local.total_reactive_flux.end());
    out_bootstrap.rate_source_to_sink_per_lag.insert(
        out_bootstrap.rate_source_to_sink_per_lag.end(),
        local.rate_source_to_sink_per_lag.begin(),
        local.rate_source_to_sink_per_lag.end());
    out_bootstrap.mfpt_source_to_sink_lags.insert(
        out_bootstrap.mfpt_source_to_sink_lags.end(),
        local.mfpt_source_to_sink_lags.begin(),
        local.mfpt_source_to_sink_lags.end());
    out_bootstrap.source_population.insert(
        out_bootstrap.source_population.end(), local.source_population.begin(),
        local.source_population.end());
    out_bootstrap.sink_population.insert(out_bootstrap.sink_population.end(),
                                         local.sink_population.begin(),
                                         local.sink_population.end());
    out_bootstrap.q_forward_sum += local.q_forward_sum;
    out_bootstrap.q_forward_sq_sum += local.q_forward_sq_sum;
    out_bootstrap.reactive_density_sum += local.reactive_density_sum;
    out_bootstrap.reactive_density_sq_sum += local.reactive_density_sq_sum;
  });

  if (out_bootstrap.successful_samples <= 0) {
    error_message = "all bootstrap replicates failed";
    return false;
  }
  return true;
}

bool run_dham_tpt_bootstrap(const DhamRebuildContext &context, int top_paths,
                            double flux_cut,
                            const std::vector<int> &source_states,
                            const std::vector<int> &sink_states, int n_samples,
                            unsigned int seed, int n_threads,
                            TitanExecutor &executor,
                            TptBootstrapAggregate &out_bootstrap,
                            std::string &error_message) {
  out_bootstrap = {};
  out_bootstrap.requested_samples = n_samples;
  if (!context.available) {
    error_message = "DHAM validation context is unavailable";
    return false;
  }
  if (n_samples <= 0)
    return true;
  if (context.n_replicas < 2) {
    error_message =
        "at least two replica series are recommended for DHAM TPT bootstrap";
    return false;
  }

  struct ReplicaSeries {
    int source_replica{-1};
    std::vector<int> dtraj;
    std::vector<double> times;
    std::vector<double> observables;
  };
  std::vector<ReplicaSeries> series;
  series.reserve(static_cast<std::size_t>(context.n_replicas));
  for (int replica = 0; replica < context.n_replicas; ++replica) {
    auto it_dtraj = context.per_replica_dtrajs.find(replica);
    auto it_time = context.per_replica_timestamps_ps.find(replica);
    auto it_obs = context.per_replica_observables.find(replica);
    if (it_dtraj == context.per_replica_dtrajs.end() ||
        it_time == context.per_replica_timestamps_ps.end() ||
        it_obs == context.per_replica_observables.end()) {
      continue;
    }
    series.push_back({replica, it_dtraj->second, it_time->second,
                      it_obs->second});
  }
  if (series.size() < 2) {
    error_message =
        "DHAM validation context does not contain enough complete replica series";
    return false;
  }

  out_bootstrap.q_forward_sum = Eigen::VectorXd::Zero(context.n_states);
  out_bootstrap.q_forward_sq_sum = Eigen::VectorXd::Zero(context.n_states);
  out_bootstrap.reactive_density_sum = Eigen::VectorXd::Zero(context.n_states);
  out_bootstrap.reactive_density_sq_sum = Eigen::VectorXd::Zero(context.n_states);

  struct LocalAccum {
    int successful_samples{0};
    std::vector<double> total_reactive_flux;
    std::vector<double> rate_source_to_sink_per_lag;
    std::vector<double> mfpt_source_to_sink_lags;
    std::vector<double> source_population;
    std::vector<double> sink_population;
    Eigen::VectorXd q_forward_sum;
    Eigen::VectorXd q_forward_sq_sum;
    Eigen::VectorXd reactive_density_sum;
    Eigen::VectorXd reactive_density_sq_sum;

    explicit LocalAccum(int n_states_in)
        : q_forward_sum(Eigen::VectorXd::Zero(n_states_in)),
          q_forward_sq_sum(Eigen::VectorXd::Zero(n_states_in)),
          reactive_density_sum(Eigen::VectorXd::Zero(n_states_in)),
          reactive_density_sq_sum(Eigen::VectorXd::Zero(n_states_in)) {}
  };

  std::mutex merge_mutex;
  const int worker_count = std::max(1, std::min(n_threads, n_samples));
  const int chunk = (n_samples + worker_count - 1) / worker_count;
  executor.parallel_do(static_cast<size_t>(worker_count), [&](size_t worker_sz) {
    const int worker = static_cast<int>(worker_sz);
    const int begin = worker * chunk;
    const int end = std::min(n_samples, begin + chunk);
    if (begin >= end)
      return;

    LocalAccum local(context.n_states);
    std::mt19937 rng(seed + static_cast<unsigned int>(worker * 17011 + 23));
    std::uniform_int_distribution<int> pick(0,
                                            static_cast<int>(series.size()) - 1);

    for (int sample_idx = begin; sample_idx < end; ++sample_idx) {
      (void)sample_idx;
      DhamRebuildContext sampled = context;
      sampled.per_replica_dtrajs.clear();
      sampled.per_replica_timestamps_ps.clear();
      sampled.per_replica_observables.clear();
      sampled.temps_k.clear();
      sampled.n_replicas = static_cast<int>(series.size());

      for (int replica_slot = 0; replica_slot < sampled.n_replicas; ++replica_slot) {
        const ReplicaSeries &picked = series[static_cast<std::size_t>(pick(rng))];
        sampled.per_replica_dtrajs[replica_slot] = picked.dtraj;
        sampled.per_replica_timestamps_ps[replica_slot] = picked.times;
        sampled.per_replica_observables[replica_slot] = picked.observables;
        const int temp_index =
            std::clamp(picked.source_replica, 0,
                       static_cast<int>(context.temps_k.size()) - 1);
        sampled.temps_k.push_back(context.temps_k[static_cast<std::size_t>(temp_index)]);
      }

      Eigen::MatrixXd sample_tpm;
      Eigen::VectorXd sample_pi;
      std::int64_t transitions = 0;
      std::string sample_error;
      if (!build_dham_model_from_context(sampled, context.lag_frames, sample_tpm,
                                         sample_pi, transitions, sample_error)) {
        continue;
      }

      try {
        TitanTPT sample_tpt(sample_tpm, sample_pi);
        sample_tpt.compute(source_states, sink_states, top_paths, flux_cut,
                           nullptr, 1);
        const auto &summary = sample_tpt.summary();
        append_if_finite(local.total_reactive_flux,
                         summary.total_reactive_flux);
        append_if_finite(local.rate_source_to_sink_per_lag,
                         summary.rate_source_to_sink_per_lag);
        append_if_finite(local.mfpt_source_to_sink_lags,
                         summary.mfpt_source_to_sink_lags);
        append_if_finite(local.source_population, summary.source_population);
        append_if_finite(local.sink_population, summary.sink_population);
        local.q_forward_sum += sample_tpt.forward_committor();
        local.q_forward_sq_sum +=
            sample_tpt.forward_committor().array().square().matrix();
        local.reactive_density_sum += sample_tpt.reactive_density();
        local.reactive_density_sq_sum +=
            sample_tpt.reactive_density().array().square().matrix();
        ++local.successful_samples;
      } catch (const std::exception &) {
        continue;
      }
    }

    std::lock_guard<std::mutex> lock(merge_mutex);
    out_bootstrap.successful_samples += local.successful_samples;
    out_bootstrap.total_reactive_flux.insert(
        out_bootstrap.total_reactive_flux.end(),
        local.total_reactive_flux.begin(), local.total_reactive_flux.end());
    out_bootstrap.rate_source_to_sink_per_lag.insert(
        out_bootstrap.rate_source_to_sink_per_lag.end(),
        local.rate_source_to_sink_per_lag.begin(),
        local.rate_source_to_sink_per_lag.end());
    out_bootstrap.mfpt_source_to_sink_lags.insert(
        out_bootstrap.mfpt_source_to_sink_lags.end(),
        local.mfpt_source_to_sink_lags.begin(),
        local.mfpt_source_to_sink_lags.end());
    out_bootstrap.source_population.insert(
        out_bootstrap.source_population.end(), local.source_population.begin(),
        local.source_population.end());
    out_bootstrap.sink_population.insert(out_bootstrap.sink_population.end(),
                                         local.sink_population.begin(),
                                         local.sink_population.end());
    out_bootstrap.q_forward_sum += local.q_forward_sum;
    out_bootstrap.q_forward_sq_sum += local.q_forward_sq_sum;
    out_bootstrap.reactive_density_sum += local.reactive_density_sum;
    out_bootstrap.reactive_density_sq_sum += local.reactive_density_sq_sum;
  });

  if (out_bootstrap.successful_samples <= 0) {
    error_message = "all DHAM-backed bootstrap replicates failed";
    return false;
  }
  return true;
}

std::vector<double> compute_implied_timescales_from_tpm(const Eigen::MatrixXd &tpm,
                                                        double lag_time_ns,
                                                        int n_timescales) {
  std::vector<double> timescales;
  if (tpm.rows() == 0 || lag_time_ns <= 0.0 || n_timescales <= 0)
    return timescales;

  Eigen::EigenSolver<Eigen::MatrixXd> solver(tpm, false);
  if (solver.info() != Eigen::Success)
    return timescales;

  std::vector<double> lambdas;
  lambdas.reserve(static_cast<std::size_t>(tpm.rows()));
  for (int i = 0; i < solver.eigenvalues().size(); ++i) {
    const auto lambda = solver.eigenvalues()[i];
    if (std::abs(lambda.imag()) > 1e-8)
      continue;
    lambdas.push_back(lambda.real());
  }
  std::sort(lambdas.begin(), lambdas.end(), std::greater<double>());
  for (double lambda : lambdas) {
    if (lambda >= 1.0 - 1e-12 || lambda <= 0.0)
      continue;
    const double its = -lag_time_ns / std::log(lambda);
    if (std::isfinite(its) && its > 0.0)
      timescales.push_back(its);
    if (static_cast<int>(timescales.size()) >= n_timescales)
      break;
  }
  while (static_cast<int>(timescales.size()) < n_timescales) {
    timescales.push_back(std::numeric_limits<double>::quiet_NaN());
  }
  return timescales;
}

bool run_msm_bootstrap(const StateSegments &base_segments, int n_states,
                       int lag_frames, double lag_time_ns,
                       const std::vector<int> &ck_multiples,
                       int n_timescales, int n_samples, unsigned int seed,
                       int n_threads, TitanExecutor &executor,
                       MsmBootstrapAggregate &out_bootstrap,
                       std::string &error_message) {
  out_bootstrap = {};
  out_bootstrap.requested_samples = n_samples;
  if (n_samples <= 0)
    return true;
  if (base_segments.empty()) {
    error_message = "no trajectory segments are available for MSM bootstrap";
    return false;
  }
  if (base_segments.size() < 2) {
    error_message =
        "at least two trajectory segments are recommended for bootstrap";
    return false;
  }

  out_bootstrap.stationary_sum = Eigen::VectorXd::Zero(n_states);
  out_bootstrap.stationary_sq_sum = Eigen::VectorXd::Zero(n_states);
  out_bootstrap.implied_timescale_samples.resize(
      static_cast<std::size_t>(std::max(0, n_timescales)));
  out_bootstrap.ck_weighted_frobenius_samples.resize(ck_multiples.size());
  out_bootstrap.ck_weighted_mae_samples.resize(ck_multiples.size());

  struct LocalAccum {
    int successful_samples{0};
    Eigen::VectorXd stationary_sum;
    Eigen::VectorXd stationary_sq_sum;
    std::vector<std::vector<double>> implied_timescale_samples;
    std::vector<std::vector<double>> ck_weighted_frobenius_samples;
    std::vector<std::vector<double>> ck_weighted_mae_samples;

    LocalAccum(int n_states_in, int n_timescales_in, std::size_t n_ck)
        : stationary_sum(Eigen::VectorXd::Zero(n_states_in)),
          stationary_sq_sum(Eigen::VectorXd::Zero(n_states_in)),
          implied_timescale_samples(
              static_cast<std::size_t>(std::max(0, n_timescales_in))),
          ck_weighted_frobenius_samples(n_ck),
          ck_weighted_mae_samples(n_ck) {}
  };

  std::mutex merge_mutex;
  const int worker_count = std::max(1, std::min(n_threads, n_samples));
  const int chunk = (n_samples + worker_count - 1) / worker_count;
  executor.parallel_do(static_cast<size_t>(worker_count), [&](size_t worker_sz) {
    const int worker = static_cast<int>(worker_sz);
    const int begin = worker * chunk;
    const int end = std::min(n_samples, begin + chunk);
    if (begin >= end)
      return;

    LocalAccum local(n_states, n_timescales, ck_multiples.size());
    std::mt19937 rng(seed + static_cast<unsigned int>(worker * 6151 + 29));
    std::uniform_int_distribution<int> seg_pick(
        0, static_cast<int>(base_segments.size()) - 1);
    StateSegments sampled_segments;
    sampled_segments.reserve(base_segments.size());

    for (int sample_idx = begin; sample_idx < end; ++sample_idx) {
      (void)sample_idx;
      sampled_segments.clear();
      for (std::size_t seg = 0; seg < base_segments.size(); ++seg) {
        sampled_segments.push_back(
            base_segments[static_cast<std::size_t>(seg_pick(rng))]);
      }

      Eigen::MatrixXd sample_tpm;
      Eigen::VectorXd sample_pi;
      std::int64_t transitions = 0;
      std::string sample_error;
      if (!build_markov_model_from_segments(n_states, lag_frames,
                                            sampled_segments, true,
                                            sample_tpm, sample_pi,
                                            transitions, sample_error)) {
        continue;
      }

      local.stationary_sum += sample_pi;
      local.stationary_sq_sum += sample_pi.array().square().matrix();
      const auto its = compute_implied_timescales_from_tpm(sample_tpm,
                                                           lag_time_ns,
                                                           n_timescales);
      for (std::size_t idx = 0; idx < its.size(); ++idx) {
        append_if_finite(local.implied_timescale_samples[idx], its[idx]);
      }

      const auto ck_summary = compute_ck_for_transition_model(
          sample_tpm, lag_frames, n_states, sampled_segments, ck_multiples,
          "msm_bootstrap_sample");
      for (std::size_t idx = 0; idx < ck_multiples.size(); ++idx) {
        auto metric_it = std::find_if(
            ck_summary.metrics.begin(), ck_summary.metrics.end(),
            [&](const ChapmanKolmogorovMetric &metric) {
              return metric.multiple == ck_multiples[idx];
            });
        if (metric_it == ck_summary.metrics.end())
          continue;
        append_if_finite(local.ck_weighted_frobenius_samples[idx],
                         metric_it->weighted_frobenius);
        append_if_finite(local.ck_weighted_mae_samples[idx],
                         metric_it->weighted_mae);
      }
      ++local.successful_samples;
    }

    std::lock_guard<std::mutex> lock(merge_mutex);
    out_bootstrap.successful_samples += local.successful_samples;
    out_bootstrap.stationary_sum += local.stationary_sum;
    out_bootstrap.stationary_sq_sum += local.stationary_sq_sum;
    for (std::size_t idx = 0; idx < out_bootstrap.implied_timescale_samples.size();
         ++idx) {
      auto &dst = out_bootstrap.implied_timescale_samples[idx];
      const auto &src = local.implied_timescale_samples[idx];
      dst.insert(dst.end(), src.begin(), src.end());
    }
    for (std::size_t idx = 0; idx < out_bootstrap.ck_weighted_frobenius_samples.size();
         ++idx) {
      auto &dst_frob = out_bootstrap.ck_weighted_frobenius_samples[idx];
      auto &dst_mae = out_bootstrap.ck_weighted_mae_samples[idx];
      const auto &src_frob = local.ck_weighted_frobenius_samples[idx];
      const auto &src_mae = local.ck_weighted_mae_samples[idx];
      dst_frob.insert(dst_frob.end(), src_frob.begin(), src_frob.end());
      dst_mae.insert(dst_mae.end(), src_mae.begin(), src_mae.end());
    }
  });

  if (out_bootstrap.successful_samples <= 0) {
    error_message = "all MSM bootstrap replicates failed";
    return false;
  }
  return true;
}

} // namespace TitanValidation
