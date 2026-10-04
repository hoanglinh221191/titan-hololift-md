#include "titan_umap.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace {

constexpr double kDistanceEps = 1e-6;
constexpr double kGraphWeightEps = 1e-8;
constexpr double kSigmaFloor = 1e-3;
constexpr double kGradClip = 4.0;
constexpr int kNegativeSampleRate = 5;
constexpr double kRepulsionGamma = 0.35;

double clamp_grad(double value) {
  return std::clamp(value, -kGradClip, kGradClip);
}

double safe_pow(double value, double power) {
  return std::pow(std::max(value, kDistanceEps), power);
}

} // namespace

TitanUMAP::TitanUMAP(const Config &cfg) : cfg_(cfg) {}

double TitanUMAP::compute_membership_sum(const std::vector<Neighbor> &neighbors,
                                         double rho, double sigma) {
  double total = 0.0;
  for (const Neighbor &neighbor : neighbors) {
    if (neighbor.index < 0)
      continue;
    if (neighbor.distance <= rho) {
      total += 1.0;
    } else if (sigma > 0.0) {
      total += std::exp(-(neighbor.distance - rho) / sigma);
    }
  }
  return total;
}

void TitanUMAP::compute_smooth_knn_memberships(
    std::vector<std::vector<Neighbor>> &knn) const {
  for (std::vector<Neighbor> &neighbors : knn) {
    if (neighbors.empty())
      continue;

    double rho = 0.0;
    // neighbors are stored in ascending distance order by
    // build_exact_knn_graph(), so the first strictly positive distance is the
    // nearest non-self neighbour used as rho.
    for (const Neighbor &neighbor : neighbors) {
      if (neighbor.index >= 0 && neighbor.distance > 0.0) {
        rho = neighbor.distance;
        break;
      }
    }

    const double target =
        std::log2(static_cast<double>(std::max<size_t>(2, neighbors.size())));
    double lo = 0.0;
    double hi = 1.0;
    while (compute_membership_sum(neighbors, rho, hi) < target && hi < 1e6) {
      hi *= 2.0;
    }

    for (int iter = 0; iter < 64; ++iter) {
      const double mid = 0.5 * (lo + hi);
      const double value = compute_membership_sum(neighbors, rho, mid);
      if (value < target) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    const double sigma = std::max(hi, kSigmaFloor);
    for (Neighbor &neighbor : neighbors) {
      if (neighbor.index < 0) {
        neighbor.membership = 0.0;
      } else if (neighbor.distance <= rho) {
        neighbor.membership = 1.0;
      } else {
        neighbor.membership = std::exp(-(neighbor.distance - rho) / sigma);
      }
    }
  }
}

std::vector<TitanUMAP::Edge> TitanUMAP::build_fuzzy_simplicial_edges(
    const std::vector<std::vector<Neighbor>> &knn) const {
  const auto lookup_membership = [&knn](int src, int dst) {
    for (const Neighbor &neighbor : knn[static_cast<size_t>(src)]) {
      if (neighbor.index == dst)
        return neighbor.membership;
    }
    return 0.0;
  };

  std::vector<Edge> edges;
  std::unordered_set<uint64_t> seen;
  size_t total_directed = 0;
  for (const auto &row : knn)
    total_directed += row.size();
  seen.reserve(total_directed);
  edges.reserve(total_directed);

  for (size_t i = 0; i < knn.size(); ++i) {
    for (const Neighbor &neighbor : knn[i]) {
      if (neighbor.index < 0)
        continue;

      const int j = neighbor.index;
      const int a = std::min(static_cast<int>(i), j);
      const int b = std::max(static_cast<int>(i), j);
      if (a == b)
        continue;

      const uint64_t key =
          (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
          static_cast<uint32_t>(b);
      if (!seen.insert(key).second)
        continue;

      const double p_ab = lookup_membership(a, b);
      const double p_ba = lookup_membership(b, a);
      const double weight = p_ab + p_ba - p_ab * p_ba;
      if (weight <= kGraphWeightEps)
        continue;

      edges.push_back(Edge{a, b, weight});
    }
  }

  return edges;
}

std::pair<double, double> TitanUMAP::fit_ab_params() const {
  const int n_samples = 300;
  const double max_x = std::max(cfg_.spread * 3.0, cfg_.min_dist * 3.0);
  double best_a = 1.0;
  double best_b = 1.0;
  double best_error = std::numeric_limits<double>::infinity();

  for (double b = 0.5; b <= 4.0; b += 0.05) {
    double a_sum = 0.0;
    int a_count = 0;

    for (int i = 1; i < n_samples; ++i) {
      const double x = max_x * static_cast<double>(i) / (n_samples - 1);
      const double target =
          (x <= cfg_.min_dist)
              ? 1.0
              : std::exp(-(x - cfg_.min_dist) / std::max(cfg_.spread, 1e-6));
      if (target >= 0.999999)
        continue;

      const double x_term = safe_pow(x * x, b);
      const double a_i = (1.0 / target - 1.0) / x_term;
      if (std::isfinite(a_i) && a_i > 0.0) {
        a_sum += a_i;
        ++a_count;
      }
    }

    if (a_count == 0)
      continue;

    const double a = a_sum / static_cast<double>(a_count);
    double error = 0.0;
    for (int i = 0; i < n_samples; ++i) {
      const double x = max_x * static_cast<double>(i) / (n_samples - 1);
      const double target =
          (x <= cfg_.min_dist)
              ? 1.0
              : std::exp(-(x - cfg_.min_dist) / std::max(cfg_.spread, 1e-6));
      const double fit = 1.0 / (1.0 + a * safe_pow(x * x, b));
      const double diff = fit - target;
      error += diff * diff;
    }

    if (error < best_error) {
      best_error = error;
      best_a = a;
      best_b = b;
    }
  }

  return {best_a, best_b};
}

void TitanUMAP::initialize_embedding(const MatrixXdAnyConstRef &input,
                                     std::mt19937 &rng) {
  const int n_samples = static_cast<int>(input.rows());
  const int in_dim = static_cast<int>(input.cols());
  embedding_.setZero(n_samples, cfg_.n_components);

  for (int c = 0; c < cfg_.n_components; ++c) {
    if (c < in_dim) {
      embedding_.col(c) = input.col(c);
    }
  }

  for (int c = 0; c < cfg_.n_components; ++c) {
    const double mean = embedding_.col(c).mean();
    embedding_.col(c).array() -= mean;
  }

  const double max_abs = std::max(embedding_.cwiseAbs().maxCoeff(), 1e-6);
  embedding_ /= max_abs;

  std::normal_distribution<double> jitter(0.0, 1e-3);
  for (int i = 0; i < n_samples; ++i) {
    for (int c = 0; c < cfg_.n_components; ++c) {
      if (c >= in_dim)
        embedding_(i, c) = jitter(rng);
      embedding_(i, c) += jitter(rng);
    }
  }
}

void TitanUMAP::optimize_layout(const std::vector<Edge> &edges, double a, double b,
                                std::mt19937 &rng) {
  if (edges.empty())
    return;

  std::vector<size_t> order(edges.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::uniform_int_distribution<int> node_dist(0, embedding_.rows() - 1);

  for (int epoch = 0; epoch < cfg_.n_epochs; ++epoch) {
    const double alpha =
        (cfg_.n_epochs <= 1)
            ? 1.0
            : std::max(0.01,
                       1.0 -
                           static_cast<double>(epoch) /
                               static_cast<double>(cfg_.n_epochs - 1));
    std::shuffle(order.begin(), order.end(), rng);

    for (size_t edge_idx : order) {
      const Edge &edge = edges[edge_idx];
      std::array<double, 3> delta{0.0, 0.0, 0.0};
      double dist_sq = kDistanceEps;
      for (int c = 0; c < cfg_.n_components; ++c) {
        delta[c] = embedding_(edge.head, c) - embedding_(edge.tail, c);
        dist_sq += delta[c] * delta[c];
      }

      const double dist_sq_b = safe_pow(dist_sq, b);
      const double attr_coeff =
          -2.0 * a * b * (dist_sq_b / dist_sq) /
          (1.0 + a * dist_sq_b);

      for (int c = 0; c < cfg_.n_components; ++c) {
        const double grad = clamp_grad(attr_coeff * delta[c] * edge.weight);
        embedding_(edge.head, c) += alpha * grad;
        embedding_(edge.tail, c) -= alpha * grad;
      }

      for (int neg = 0; neg < kNegativeSampleRate; ++neg) {
        int other = node_dist(rng);
        if (other == edge.head || other == edge.tail)
          continue;

        std::array<double, 3> neg_delta{0.0, 0.0, 0.0};
        double neg_dist_sq = kDistanceEps;
        for (int c = 0; c < cfg_.n_components; ++c) {
          neg_delta[c] = embedding_(edge.head, c) - embedding_(other, c);
          neg_dist_sq += neg_delta[c] * neg_delta[c];
        }

        const double neg_dist_sq_b = safe_pow(neg_dist_sq, b);
        const double rep_coeff =
            kRepulsionGamma * 2.0 * b /
            ((0.001 + neg_dist_sq) * (1.0 + a * neg_dist_sq_b));

        for (int c = 0; c < cfg_.n_components; ++c) {
          const double grad = clamp_grad(rep_coeff * neg_delta[c]);
          embedding_(edge.head, c) += alpha * grad;
        }
      }
    }

    for (int c = 0; c < cfg_.n_components; ++c) {
      embedding_.col(c).array() -= embedding_.col(c).mean();
    }

    if (cfg_.verbose && ((epoch + 1) % 50 == 0 || epoch + 1 == cfg_.n_epochs)) {
      TitanUI::print_info("UMAP: completed epoch " + std::to_string(epoch + 1) +
                          "/" + std::to_string(cfg_.n_epochs));
    }
  }
}

void TitanUMAP::fit_transform(const MatrixXdAnyConstRef &proj_mat,
                              int64_t n_actual_frames) {
  fitted_ = false;
  embedding_.resize(0, 0);

  if (n_actual_frames <= 1) {
    TitanUI::print_error("UMAP: need at least 2 projected frames.");
    return;
  }
  if (cfg_.n_components != 2 && cfg_.n_components != 3) {
    TitanUI::print_error("UMAP: only 2D or 3D embeddings are supported.");
    return;
  }
  if (cfg_.n_neighbors < 2 || cfg_.n_epochs <= 0) {
    TitanUI::print_error("UMAP: invalid configuration.");
    return;
  }
  if (proj_mat.cols() <= 0) {
    TitanUI::print_error("UMAP: empty TICA projection matrix.");
    return;
  }

  const int effective_neighbors =
      std::min(cfg_.n_neighbors, static_cast<int>(n_actual_frames) - 1);
  if (effective_neighbors < 2) {
    TitanUI::print_error("UMAP: trajectory is too short for nearest-neighbour graph.");
    return;
  }
  if (effective_neighbors != cfg_.n_neighbors) {
    TitanUI::print_warning("UMAP: clamping n_neighbors to " +
                           std::to_string(effective_neighbors) +
                           " because the trajectory is short.");
  }

  const auto run_fit = [&](const auto &input) {
    if (cfg_.verbose) {
      TitanUI::print_info("UMAP: building kNN graph for " +
                          std::to_string(n_actual_frames) + " frames in " +
                          std::to_string(input.cols()) + " TICA dimensions.");
    }

    TitanUMAPKNN::BuildConfig knn_cfg;
    knn_cfg.mode = cfg_.knn_mode;
    knn_cfg.seed = cfg_.seed;
    knn_cfg.benchmark = cfg_.benchmark_knn;
    knn_cfg.auto_exact_threshold = cfg_.knn_auto_exact_threshold;
    knn_cfg.approx_iterations = cfg_.knn_approx_iterations;
    knn_cfg.approx_seed_neighbors = cfg_.knn_approx_seed_neighbors;
    knn_cfg.approx_extra_random_candidates = cfg_.knn_approx_extra_random_candidates;
    knn_cfg.n_threads = std::max(1, cfg_.n_threads);
    knn_cfg.executor = cfg_.executor;

    auto knn_result =
        TitanUMAPKNN::build_knn_graph(input, effective_neighbors, knn_cfg);
    if (cfg_.verbose || cfg_.benchmark_knn) {
      TitanUI::print_info(
          "UMAP: kNN mode requested=" +
          std::string(TitanUMAPKNN::mode_name(knn_result.stats.requested_mode)) +
          ", selected=" +
          std::string(TitanUMAPKNN::mode_name(knn_result.stats.selected_mode)) +
          ", time=" + std::to_string(knn_result.stats.selected_elapsed_ms) +
          " ms.");
      if (knn_result.stats.selected_mode == KnnMode::Approx) {
        TitanUI::print_info("UMAP: approx kNN iterations=" +
                            std::to_string(knn_result.stats.approx_iterations_completed) +
                            ", distance evals=" +
                            std::to_string(knn_result.stats.selected_distance_evals) + ".");
      }
      if (knn_result.stats.benchmark_has_baseline) {
        TitanUI::print_info("UMAP: kNN benchmark exact=" +
                            std::to_string(knn_result.stats.exact_elapsed_ms) +
                            " ms, approx=" +
                            std::to_string(knn_result.stats.approx_elapsed_ms) +
                            " ms, recall=" +
                            std::to_string(knn_result.stats.benchmark_recall) + ".");
      }
    }

    auto knn = std::move(knn_result.knn);
    compute_smooth_knn_memberships(knn);
    const auto edges = build_fuzzy_simplicial_edges(knn);
    if (edges.empty()) {
      TitanUI::print_error("UMAP: fuzzy simplicial graph is empty.");
      return false;
    }

    auto [a, b] = fit_ab_params();
    std::mt19937 rng(cfg_.seed);
    initialize_embedding(input, rng);
    optimize_layout(edges, a, b, rng);
    if (cfg_.verbose) {
      TitanUI::print_info("UMAP: fitted embedding with " +
                          std::to_string(edges.size()) + " fuzzy edges.");
    }
    return true;
  };

  const Eigen::Index n_rows = static_cast<Eigen::Index>(n_actual_frames);
  const Eigen::Index n_cols = proj_mat.cols();
  const bool row_major_contiguous =
      proj_mat.innerStride() == 1 && proj_mat.outerStride() == n_cols;

  if (row_major_contiguous) {
    if (cfg_.verbose) {
      TitanUI::print_info("UMAP: reusing row-major TICA projection buffer without copy.");
    }
    Eigen::Map<const InputMatrixRM> input_view(proj_mat.data(), n_rows, n_cols);
    fitted_ = run_fit(input_view);
  } else {
    InputMatrixRM input_copy = proj_mat.topRows(n_rows);
    if (input_copy.rows() != n_actual_frames) {
      TitanUI::print_error("UMAP: failed to materialize projected coordinates.");
      return;
    }
    fitted_ = run_fit(input_copy);
  }
}

void TitanUMAP::write_embedding(const std::string &out_file,
                                std::fstream &log) const {
  if (!fitted_) {
    TitanUI::print_warning("UMAP: fit_transform() did not finish successfully.");
    return;
  }

  std::ofstream out(out_file);
  if (!out.is_open()) {
    TitanUI::print_error("UMAP: cannot open " + out_file + " for writing.");
    return;
  }

  out << "frame";
  for (int c = 0; c < embedding_.cols(); ++c)
    out << ",UMAP" << (c + 1);
  out << "\n";

  out << std::fixed << std::setprecision(6);
  for (Eigen::Index i = 0; i < embedding_.rows(); ++i) {
    out << i;
    for (Eigen::Index c = 0; c < embedding_.cols(); ++c)
      out << "," << embedding_(i, c);
    out << "\n";
  }

  TitanUI::print_info("UMAP: embedding written to " + out_file);
  TitanUI::log_kv(log, "UMAP Embedding File", out_file);
}
