#include "titan_corr_knn_core.h"

#include "titan_executor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

namespace titan_corr_knn {
namespace {

constexpr double kEulerGamma = 0.5772156649015328606;
constexpr double kDegenerateTol = 1e-12;

struct CenteredTraceData {
  std::vector<AlignedDoubleVector> x;
  std::vector<AlignedDoubleVector> y;
  std::vector<AlignedDoubleVector> z;
  std::vector<unsigned char> degenerate;
  std::size_t degenerate_count{0};
};

inline double chebyshev3(const AlignedDoubleVector &ax,
                         const AlignedDoubleVector &ay,
                         const AlignedDoubleVector &az, std::size_t lhs,
                         std::size_t rhs) {
  const double dx = std::abs(ax[lhs] - ax[rhs]);
  const double dy = std::abs(ay[lhs] - ay[rhs]);
  const double dz = std::abs(az[lhs] - az[rhs]);
  return std::max(dx, std::max(dy, dz));
}

inline double chebyshev6(const AlignedDoubleVector &ax1,
                         const AlignedDoubleVector &ay1,
                         const AlignedDoubleVector &az1,
                         const AlignedDoubleVector &ax2,
                         const AlignedDoubleVector &ay2,
                         const AlignedDoubleVector &az2, std::size_t lhs,
                         std::size_t rhs) {
  return std::max(chebyshev3(ax1, ay1, az1, lhs, rhs),
                  chebyshev3(ax2, ay2, az2, lhs, rhs));
}

CenteredTraceData build_centered_trace_data(const TrajectoryRows &traj_x,
                                            const TrajectoryRows &traj_y,
                                            const TrajectoryRows &traj_z,
                                            TitanExecutor &executor) {
  CenteredTraceData centered;
  const std::size_t total_frames = traj_x.size();
  if (total_frames == 0)
    return centered;
  const std::size_t node_count = traj_x.front().size();
  centered.x.assign(node_count, AlignedDoubleVector(total_frames, 0.0));
  centered.y.assign(node_count, AlignedDoubleVector(total_frames, 0.0));
  centered.z.assign(node_count, AlignedDoubleVector(total_frames, 0.0));
  centered.degenerate.assign(node_count, 0);

  const std::size_t worker_count = std::max<std::size_t>(1, executor.capacity());
  const std::size_t atoms_per_worker =
      (node_count + worker_count - 1) / worker_count;
  std::vector<std::size_t> local_degenerate_counts(worker_count, 0);
  executor.parallel_do(worker_count, [&](std::size_t worker_id) {
    const std::size_t begin = worker_id * atoms_per_worker;
    const std::size_t end = std::min(begin + atoms_per_worker, node_count);
    if (begin >= end)
      return;
    std::size_t local_count = 0;
    for (std::size_t node = begin; node < end; ++node) {
      double mean_x = 0.0;
      double mean_y = 0.0;
      double mean_z = 0.0;
      for (std::size_t frame = 0; frame < total_frames; ++frame) {
        mean_x += traj_x[frame][node];
        mean_y += traj_y[frame][node];
        mean_z += traj_z[frame][node];
      }
      const double inv_frames = 1.0 / static_cast<double>(total_frames);
      mean_x *= inv_frames;
      mean_y *= inv_frames;
      mean_z *= inv_frames;

      double max_abs_dev = 0.0;
      for (std::size_t frame = 0; frame < total_frames; ++frame) {
        const double dx = traj_x[frame][node] - mean_x;
        const double dy = traj_y[frame][node] - mean_y;
        const double dz = traj_z[frame][node] - mean_z;
        centered.x[node][frame] = dx;
        centered.y[node][frame] = dy;
        centered.z[node][frame] = dz;
        max_abs_dev = std::max(max_abs_dev, std::abs(dx));
        max_abs_dev = std::max(max_abs_dev, std::abs(dy));
        max_abs_dev = std::max(max_abs_dev, std::abs(dz));
      }
      if (max_abs_dev <= kDegenerateTol) {
        centered.degenerate[node] = 1;
        ++local_count;
      }
    }
    local_degenerate_counts[worker_id] = local_count;
  });

  centered.degenerate_count = std::accumulate(local_degenerate_counts.begin(),
                                              local_degenerate_counts.end(),
                                              static_cast<std::size_t>(0));
  return centered;
}

std::vector<double> build_digamma_lookup(std::size_t max_n) {
  std::vector<double> lookup(max_n + 2, -kEulerGamma);
  double harmonic = 0.0;
  for (std::size_t n = 2; n < lookup.size(); ++n) {
    harmonic += 1.0 / static_cast<double>(n - 1);
    lookup[n] = harmonic - kEulerGamma;
  }
  return lookup;
}

double estimate_mi_ksg(const CenteredTraceData &trace, std::size_t lhs_node,
                       std::size_t rhs_node, int k,
                       const std::vector<double> &digamma) {
  const auto &x1 = trace.x[lhs_node];
  const auto &y1 = trace.y[lhs_node];
  const auto &z1 = trace.z[lhs_node];
  const auto &x2 = trace.x[rhs_node];
  const auto &y2 = trace.y[rhs_node];
  const auto &z2 = trace.z[rhs_node];
  const std::size_t sample_count = x1.size();
  if (sample_count <= static_cast<std::size_t>(k) || sample_count < 2)
    return 0.0;

  std::vector<double> joint_distances(sample_count - 1, 0.0);
  double digamma_sum = 0.0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    std::size_t offset = 0;
    for (std::size_t other = 0; other < sample_count; ++other) {
      if (other == sample)
        continue;
      joint_distances[offset++] =
          chebyshev6(x1, y1, z1, x2, y2, z2, sample, other);
    }
    std::nth_element(joint_distances.begin(),
                     joint_distances.begin() + static_cast<std::ptrdiff_t>(k - 1),
                     joint_distances.end());
    const double epsilon = joint_distances[static_cast<std::size_t>(k - 1)];
    const double marginal_eps =
        (epsilon > 0.0) ? std::nextafter(epsilon, 0.0) : 0.0;

    int count_x = 1;
    int count_y = 1;
    if (epsilon > 0.0) {
      for (std::size_t other = 0; other < sample_count; ++other) {
        if (other == sample)
          continue;
        if (chebyshev3(x1, y1, z1, sample, other) <= marginal_eps)
          ++count_x;
        if (chebyshev3(x2, y2, z2, sample, other) <= marginal_eps)
          ++count_y;
      }
    }
    digamma_sum += digamma[static_cast<std::size_t>(count_x)];
    digamma_sum += digamma[static_cast<std::size_t>(count_y)];
  }

  const double mi =
      digamma[static_cast<std::size_t>(k)] +
      digamma[static_cast<std::size_t>(sample_count)] -
      digamma_sum / static_cast<double>(sample_count);
  if (!std::isfinite(mi) || mi < 0.0)
    return 0.0;
  return mi;
}

double estimate_te_frenzel_pompe(const CenteredTraceData &trace,
                                 std::size_t source_node,
                                 std::size_t target_node, int tau,
                                 int k, const std::vector<double> &digamma) {
  const auto &sx = trace.x[source_node];
  const auto &sy = trace.y[source_node];
  const auto &sz = trace.z[source_node];
  const auto &tx = trace.x[target_node];
  const auto &ty = trace.y[target_node];
  const auto &tz = trace.z[target_node];
  const std::size_t total_frames = sx.size();
  if (tau <= 0 || total_frames <= static_cast<std::size_t>(tau))
    return 0.0;
  const std::size_t sample_count = total_frames - static_cast<std::size_t>(tau);
  if (sample_count <= static_cast<std::size_t>(k) || sample_count < 2)
    return 0.0;

  std::vector<double> joint_distances(sample_count - 1, 0.0);
  double digamma_sum = 0.0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const std::size_t now_frame = sample + static_cast<std::size_t>(tau);
    std::size_t offset = 0;
    for (std::size_t other = 0; other < sample_count; ++other) {
      if (other == sample)
        continue;
      const std::size_t other_now_frame =
          other + static_cast<std::size_t>(tau);
      const double d_src_past =
          chebyshev3(sx, sy, sz, sample, other);
      const double d_tgt_now =
          chebyshev3(tx, ty, tz, now_frame, other_now_frame);
      const double d_tgt_past =
          chebyshev3(tx, ty, tz, sample, other);
      joint_distances[offset++] =
          std::max(d_src_past, std::max(d_tgt_now, d_tgt_past));
    }
    std::nth_element(joint_distances.begin(),
                     joint_distances.begin() + static_cast<std::ptrdiff_t>(k - 1),
                     joint_distances.end());
    const double epsilon = joint_distances[static_cast<std::size_t>(k - 1)];
    const double marginal_eps =
        (epsilon > 0.0) ? std::nextafter(epsilon, 0.0) : 0.0;

    int count_z = 1;
    int count_xz = 1;
    int count_yz = 1;
    if (epsilon > 0.0) {
      for (std::size_t other = 0; other < sample_count; ++other) {
        if (other == sample)
          continue;
        const std::size_t other_now_frame =
            other + static_cast<std::size_t>(tau);
        const double d_src_past =
            chebyshev3(sx, sy, sz, sample, other);
        const double d_tgt_now =
            chebyshev3(tx, ty, tz, now_frame, other_now_frame);
        const double d_tgt_past =
            chebyshev3(tx, ty, tz, sample, other);
        if (d_tgt_past <= marginal_eps)
          ++count_z;
        if (std::max(d_src_past, d_tgt_past) <= marginal_eps)
          ++count_xz;
        if (std::max(d_tgt_now, d_tgt_past) <= marginal_eps)
          ++count_yz;
      }
    }
    digamma_sum += digamma[static_cast<std::size_t>(count_z)];
    digamma_sum -= digamma[static_cast<std::size_t>(count_xz)];
    digamma_sum -= digamma[static_cast<std::size_t>(count_yz)];
  }

  const double te = digamma[static_cast<std::size_t>(k)] +
                    digamma_sum / static_cast<double>(sample_count);
  if (!std::isfinite(te) || te < 0.0)
    return 0.0;
  return te;
}

void write_undirected_csv(const std::string &path, const std::string &header,
                          const ExactComputationResult &result,
                          const std::vector<double> &values,
                          bool enabled) {
  std::ofstream out(path);
  out << header << "\n";
  if (!enabled)
    return;
  out << std::setprecision(15);
  for (std::size_t p = 0; p < result.candidate_pairs.size(); ++p) {
    const auto [i, j] = result.candidate_pairs[p];
    out << result.node_labels[i] << "," << result.node_labels[j] << ","
        << values[p] << "\n";
  }
}

} // namespace

ExactComputationResult compute_exact_scores(
    const TrajectoryRows &traj_x, const TrajectoryRows &traj_y,
    const TrajectoryRows &traj_z, const std::vector<std::string> &node_labels,
    const std::vector<std::pair<std::size_t, std::size_t>> &candidate_pairs,
    std::size_t dense_pair_count, const ExactEstimatorConfig &cfg,
    TitanExecutor &executor) {
  ExactComputationResult result;
  result.total_frames = traj_x.size();
  result.node_count = node_labels.size();
  result.dense_pair_count = dense_pair_count;
  result.candidate_pair_count = candidate_pairs.size();
  result.estimator_k = cfg.estimator_k;
  result.tau_lag = cfg.tau_lag;
  result.te_requested = cfg.compute_te;
  result.node_labels = node_labels;
  result.candidate_pairs = candidate_pairs;

  if (traj_x.empty() || node_labels.empty())
    return result;

  CenteredTraceData trace =
      build_centered_trace_data(traj_x, traj_y, traj_z, executor);
  result.degenerate_nodes = trace.degenerate;
  result.degenerate_node_count = trace.degenerate_count;

  const std::size_t total_frames = result.total_frames;
  result.mi_available =
      (cfg.estimator_k > 0 &&
       total_frames > static_cast<std::size_t>(cfg.estimator_k));
  result.effective_te_samples =
      (cfg.tau_lag > 0 && total_frames > static_cast<std::size_t>(cfg.tau_lag))
          ? total_frames - static_cast<std::size_t>(cfg.tau_lag)
          : 0;
  result.te_available =
      cfg.compute_te &&
      result.effective_te_samples > static_cast<std::size_t>(cfg.estimator_k);

  if (!result.mi_available)
    return result;

  result.mi_values.assign(candidate_pairs.size(), 0.0);
  result.gc_values.assign(candidate_pairs.size(), 0.0);
  if (cfg.compute_te) {
    result.te_forward_values.assign(candidate_pairs.size(), 0.0);
    result.te_reverse_values.assign(candidate_pairs.size(), 0.0);
  }

  const std::size_t max_lookup =
      std::max(total_frames, result.effective_te_samples) + 1;
  const std::vector<double> digamma = build_digamma_lookup(max_lookup + 1);
  const std::size_t worker_count = std::max<std::size_t>(1, executor.capacity());
  const std::size_t chunk =
      (candidate_pairs.size() + worker_count - 1) / worker_count;
  std::vector<std::size_t> local_degenerate_pairs(worker_count, 0);
  executor.parallel_do(worker_count, [&](std::size_t worker_id) {
    const std::size_t begin = worker_id * chunk;
    const std::size_t end = std::min(begin + chunk, candidate_pairs.size());
    if (begin >= end)
      return;
    std::size_t local_deg_pairs = 0;
    for (std::size_t p = begin; p < end; ++p) {
      const auto [i, j] = candidate_pairs[p];
      if (trace.degenerate[i] != 0 || trace.degenerate[j] != 0) {
        ++local_deg_pairs;
        continue;
      }

      const double mi =
          estimate_mi_ksg(trace, i, j, cfg.estimator_k, digamma);
      result.mi_values[p] = mi;
      const double gc = std::sqrt(std::max(
          0.0, 1.0 - std::exp((-2.0 * mi) / 3.0)));
      result.gc_values[p] = std::isfinite(gc) ? gc : 0.0;

      if (!result.te_available)
        continue;
      result.te_forward_values[p] = estimate_te_frenzel_pompe(
          trace, i, j, cfg.tau_lag, cfg.estimator_k, digamma);
      result.te_reverse_values[p] = estimate_te_frenzel_pompe(
          trace, j, i, cfg.tau_lag, cfg.estimator_k, digamma);
    }
    local_degenerate_pairs[worker_id] = local_deg_pairs;
  });
  result.degenerate_pair_count = std::accumulate(
      local_degenerate_pairs.begin(), local_degenerate_pairs.end(),
      static_cast<std::size_t>(0));
  return result;
}

std::string describe_pair_filter(const PairPruningConfig &cfg) {
  std::ostringstream out;
  bool has_filter = false;
  if (cfg.pair_cutoff_ang > 0.0) {
    out << cfg.pair_cutoff_ang << " A cutoff";
    has_filter = true;
  }
  if (cfg.pair_topk > 0) {
    if (has_filter)
      out << " + ";
    out << "top-" << cfg.pair_topk << " "
        << (cfg.pair_use_contact_prior ? "contact-frequency kNN @ "
                                       : "frame-voted kNN");
    if (cfg.pair_use_contact_prior)
      out << cfg.pair_contact_cutoff_ang << " A";
    has_filter = true;
  }
  if (!has_filter)
    out << "dense";
  out << " (" << (cfg.use_avg_cutoff ? "union across frames" : "frame-0 only")
      << ")";
  return out.str();
}

void write_summary(const std::string &path, const ExactComputationResult &result,
                   const PairPruningConfig &pair_cfg,
                   const std::string &context_label) {
  std::ofstream out(path);
  out << "# TITAN corr-kNN summary\n";
  out << "# context " << context_label << "\n";
  out << "# method exact_ksg_mi + exact_frenzel_pompe_te\n";
  out << "# metric_units MI=nats GC=unitless TE=nats\n";
  out << "total_frames " << result.total_frames << "\n";
  out << "effective_te_samples " << result.effective_te_samples << "\n";
  out << "node_count " << result.node_count << "\n";
  out << "dense_pair_count " << result.dense_pair_count << "\n";
  out << "candidate_pair_count " << result.candidate_pair_count << "\n";
  out << "estimator_k " << result.estimator_k << "\n";
  out << "tau_lag " << result.tau_lag << "\n";
  out << "mi_available " << (result.mi_available ? "yes" : "no") << "\n";
  out << "te_requested " << (result.te_requested ? "yes" : "no") << "\n";
  out << "te_available " << (result.te_available ? "yes" : "no") << "\n";
  out << "degenerate_node_count " << result.degenerate_node_count << "\n";
  out << "degenerate_pair_count " << result.degenerate_pair_count << "\n";
  out << "pair_filter " << describe_pair_filter(pair_cfg) << "\n";
}

void write_raw_mi_csv(const std::string &path,
                      const ExactComputationResult &result) {
  write_undirected_csv(path, "Source,Target,MI", result, result.mi_values,
                       result.mi_available);
}

void write_raw_gc_csv(const std::string &path,
                      const ExactComputationResult &result) {
  write_undirected_csv(path, "Source,Target,GC", result, result.gc_values,
                       result.mi_available);
}

void write_raw_te_csv(const std::string &path,
                      const ExactComputationResult &result) {
  std::ofstream out(path);
  out << "Source,Target,TE\n";
  if (!result.te_available)
    return;
  out << std::setprecision(15);
  for (std::size_t p = 0; p < result.candidate_pairs.size(); ++p) {
    const auto [i, j] = result.candidate_pairs[p];
    out << result.node_labels[i] << "," << result.node_labels[j] << ","
        << result.te_forward_values[p] << "\n";
    out << result.node_labels[j] << "," << result.node_labels[i] << ","
        << result.te_reverse_values[p] << "\n";
  }
}

void write_legacy_allostery_exports(const std::string &mi_path,
                                    const std::string &gc_path,
                                    const std::string &te_path,
                                    const ExactComputationResult &result) {
  write_raw_mi_csv(mi_path, result);

  {
    std::ofstream out(gc_path);
    out << "Source,Target,GC\n";
    if (result.mi_available) {
      out << std::setprecision(15);
      for (std::size_t p = 0; p < result.candidate_pairs.size(); ++p) {
        if (result.gc_values[p] <= 0.35)
          continue;
        const auto [i, j] = result.candidate_pairs[p];
        out << result.node_labels[i] << "," << result.node_labels[j] << ","
            << result.gc_values[p] << "\n";
      }
    }
  }

  {
    std::ofstream out(te_path);
    out << "Source,Target,Net_TE\n";
    if (result.te_available) {
      out << std::setprecision(15);
      for (std::size_t p = 0; p < result.candidate_pairs.size(); ++p) {
        const double net_te =
            result.te_forward_values[p] - result.te_reverse_values[p];
        if (std::abs(net_te) <= 0.05)
          continue;
        const auto [i, j] = result.candidate_pairs[p];
        const std::size_t source = (net_te >= 0.0) ? i : j;
        const std::size_t target = (net_te >= 0.0) ? j : i;
        out << result.node_labels[source] << "," << result.node_labels[target]
            << "," << std::abs(net_te) << "\n";
      }
    }
  }
}

} // namespace titan_corr_knn
