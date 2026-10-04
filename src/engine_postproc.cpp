#include "titan_dham_segments.h"
// engine_postproc.cpp — Post-loop analysis orchestration
// Extracted from impl_engine.cpp for faster incremental builds.
// Each function manages its own timing via std::chrono and logs via fstream.
#include "engine_postproc.h"
#include "titan_alignment.h"
#include "titan_corr_knn.h"
#include "titan_executor.h"
#include "titan_mbar_weights.h"
#include "titan_sparse_graph_module.h"
#include "titan_ui.h"
#include <Eigen/Dense>
#include <algorithm>
#include <mutex>
#include <thread>

// Clang 21 COMDAT workaround: force emission of Matrix destructor
namespace {
void _eigen_force_dtor() {
  Eigen::MatrixXd m;
  (void)m;
  Eigen::VectorXd v;
  (void)v;
}
} // namespace

#include <cmath>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <cstdint>
#include <unordered_map>
#include <type_traits>
#include <array>

namespace {

constexpr double kBarNm3ToKJPerMol = 0.06022140857;
constexpr double kEnergyTimeFilterTolerancePs = 1e-3;
constexpr int kLjFeatureColumns = 20;

void log_tica_timing_stats(std::fstream &log,
                           const TitanKinetics::TimingStats &stats) {
  TitanUI::log_time(log, "TICA Feature Build", stats.feature_build_seconds);
  TitanUI::log_time(log, "TICA Cache Write", stats.cache_write_seconds);
  TitanUI::log_time(log, "TICA Cache Read", stats.cache_read_seconds);
  TitanUI::log_time(log, "TICA Spool Write", stats.spool_write_seconds);
  TitanUI::log_time(log, "TICA Spool Read", stats.spool_read_seconds);
  TitanUI::log_time(log, "TICA Cache Promotion", stats.cache_promotion_seconds);
  TitanUI::log_time(log, "TICA exact_dual Center",
                    stats.exact_dual_center_seconds);
  TitanUI::log_time(log, "TICA exact_dual SVD", stats.exact_dual_svd_seconds);
  TitanUI::log_time(log, "TICA randomized Sketch",
                    stats.randomized_sketch_seconds);
  TitanUI::log_time(log, "TICA randomized QR", stats.randomized_qr_seconds);
  TitanUI::log_time(log, "TICA randomized Compress",
                    stats.randomized_compression_seconds);
  TitanUI::log_time(log, "TICA randomized SVD", stats.randomized_svd_seconds);
  TitanUI::log_time(log, "TICA Reduced Solve Replay",
                    stats.reduced_solve_replay_seconds);
  TitanUI::log_time(log, "TICA Reduced Eigensolve",
                    stats.reduced_eigensolve_seconds);
  TitanUI::log_time(log, "TICA Projection Replay",
                    stats.projection_replay_seconds);

  TitanUI::log_kv(log, "TICA Cache Write Frames",
                  std::to_string(stats.cache_write_frames));
  TitanUI::log_kv(log, "TICA Spool Write Frames",
                  std::to_string(stats.spool_write_frames));
  TitanUI::log_kv(log, "TICA Promoted Cache Frames",
                  std::to_string(stats.promoted_cache_frames));
  TitanUI::log_kv(log, "TICA Randomized Sketch Frames",
                  std::to_string(stats.randomized_sketch_read_frames));
  TitanUI::log_kv(log, "TICA Randomized Compress Frames",
                  std::to_string(stats.randomized_compression_read_frames));
  TitanUI::log_kv(log, "TICA Reduced Replay Frames",
                  std::to_string(stats.reduced_solve_replay_frames));
  TitanUI::log_kv(log, "TICA Projection Replay Frames",
                  std::to_string(stats.projection_replay_frames));
  TitanUI::log_kv(log, "TICA Raw Cache Released Frames",
                  std::to_string(stats.raw_feature_cache_released_frames));
  TitanUI::log_kv(log, "TICA Cache Write Bytes",
                  std::to_string(stats.cache_write_bytes));
  TitanUI::log_kv(log, "TICA Spool Write Bytes",
                  std::to_string(stats.spool_write_bytes));
  TitanUI::log_kv(log, "TICA Spool Read Bytes",
                  std::to_string(stats.spool_read_bytes));
}

struct LjMicrostateSummary {
  bool valid{false};
  std::vector<double> rho_bulk_nm3;
  std::vector<double> cluster_frac;
  std::vector<double> dense_frac;
  std::vector<double> q6_bar;
  std::vector<double> solid_like_frac;
  std::vector<double> fcc_frac;
  std::vector<double> hcp_frac;
};

struct ProteinMicrostateSummary {
  bool valid_rmsd{false};
  bool valid_dssp{false};
  std::vector<double> rmsd_mean;
  std::vector<double> helix_frac;
  std::vector<double> strand_frac;
  std::vector<double> coil_frac;
  std::vector<double> turn_frac;
  std::vector<double> bend_frac;
};

struct QNativeMicrostateSummary {
  bool valid{false};
  std::vector<double> mean_q;
};

enum class TptBasinLabel : unsigned char { Intermediate = 0, Source = 1, Sink = 2 };

using TptKineticDiagnostics = TitanValidation::TptKineticDiagnostics;
using TptBootstrapAggregate = TitanValidation::TptBootstrapAggregate;

struct DhamWindowAuditRow {
  int replica_id{-1};
  bool traj_present{false};
  bool energy_present{false};
  bool overlap_found{false};
  bool truncated_start{false};
  bool truncated_end{false};
  bool irregular_energy_dt{false};
  bool sparse_energy{false};
  std::size_t traj_frames{0};
  std::size_t energy_points{0};
  std::size_t kept_frames{0};
  std::size_t dropped_outside_overlap{0};
  std::size_t unmatched_in_overlap{0};
  double traj_min_ps{std::numeric_limits<double>::quiet_NaN()};
  double traj_max_ps{std::numeric_limits<double>::quiet_NaN()};
  double energy_min_ps{std::numeric_limits<double>::quiet_NaN()};
  double energy_max_ps{std::numeric_limits<double>::quiet_NaN()};
  double overlap_start_ps{std::numeric_limits<double>::quiet_NaN()};
  double overlap_end_ps{std::numeric_limits<double>::quiet_NaN()};
  double inferred_dt_energy_ps{std::numeric_limits<double>::quiet_NaN()};
  std::string status{"uninitialized"};
};

struct MappedEquilibriumFrameWeights {
  AlignedDoubleVector local_weights{};
  bool use_weights{false};
  std::size_t matched_frames{0};
  std::size_t invalid_frames{0};
  double raw_weight_sum{0.0};
  double ess{0.0};
  std::vector<std::size_t> ranked_local_indices{};
};

struct PairScoreRow {
  std::string source{};
  std::string target{};
  double score{0.0};
};

struct TptPathCluster {
  int representative_rank{0};
  double total_flux{0.0};
  double total_fraction{0.0};
  std::vector<int> member_ranks;
  std::vector<int> representative_nodes;
};

struct TptPathSupport {
  std::vector<double> edge_flux;
  std::vector<double> edge_fraction;
  std::vector<int> edge_count;
  std::vector<double> state_flux;
  std::vector<double> state_fraction;
  std::vector<int> state_count;

  TptPathSupport() = default;
  TptPathSupport(std::size_t n_edges, std::size_t n_states)
      : edge_flux(n_edges, 0.0), edge_fraction(n_edges, 0.0),
        edge_count(n_edges, 0), state_flux(n_states, 0.0),
        state_fraction(n_states, 0.0), state_count(n_states, 0) {}
};

struct TptBottleneckEdgeRow {
  int from{-1};
  int to{-1};
  int from_macro{-1};
  int to_macro{-1};
  double gross_flux{0.0};
  double net_flux{0.0};
  double relative_net_flux{0.0};
  double from_q{std::numeric_limits<double>::quiet_NaN()};
  double to_q{std::numeric_limits<double>::quiet_NaN()};
  double midpoint_q{std::numeric_limits<double>::quiet_NaN()};
  double transition_proximity{0.0};
  double bottleneck_score{0.0};
  double dominant_path_support_flux{0.0};
  double dominant_path_support_fraction{0.0};
  int dominant_path_count{0};
};

struct TptBottleneckStateRow {
  int state{-1};
  char label{'I'};
  int macro{-1};
  double pi{0.0};
  double q_plus{std::numeric_limits<double>::quiet_NaN()};
  double q_minus{std::numeric_limits<double>::quiet_NaN()};
  double reactive_density{0.0};
  double net_in_flux{0.0};
  double net_out_flux{0.0};
  double through_flux{0.0};
  double transition_proximity{0.0};
  double bottleneck_score{0.0};
  double dominant_path_support_flux{0.0};
  double dominant_path_support_fraction{0.0};
  int dominant_path_count{0};
};

titan_sparse_graph_module::TptCapturedEdgeRow
to_graph_tpt_edge_row(const TptBottleneckEdgeRow &row) {
  titan_sparse_graph_module::TptCapturedEdgeRow out;
  out.from = row.from;
  out.to = row.to;
  out.from_macro = row.from_macro;
  out.to_macro = row.to_macro;
  out.gross_flux = row.gross_flux;
  out.net_flux = row.net_flux;
  out.relative_net_flux = row.relative_net_flux;
  out.from_q = row.from_q;
  out.to_q = row.to_q;
  out.midpoint_q = row.midpoint_q;
  out.transition_proximity = row.transition_proximity;
  out.bottleneck_score = row.bottleneck_score;
  out.dominant_path_support_flux = row.dominant_path_support_flux;
  out.dominant_path_support_fraction = row.dominant_path_support_fraction;
  out.dominant_path_count =
      static_cast<std::size_t>(std::max(row.dominant_path_count, 0));
  return out;
}

titan_sparse_graph_module::TptCapturedNodeRow
to_graph_tpt_node_row(const TptBottleneckStateRow &row) {
  titan_sparse_graph_module::TptCapturedNodeRow out;
  out.state = row.state;
  out.label = row.label;
  out.macro = row.macro;
  out.pi = row.pi;
  out.q_plus = row.q_plus;
  out.q_minus = row.q_minus;
  out.reactive_density = row.reactive_density;
  out.net_in_flux = row.net_in_flux;
  out.net_out_flux = row.net_out_flux;
  out.through_flux = row.through_flux;
  out.transition_proximity = row.transition_proximity;
  out.bottleneck_score = row.bottleneck_score;
  out.dominant_path_support_flux = row.dominant_path_support_flux;
  out.dominant_path_support_fraction = row.dominant_path_support_fraction;
  out.dominant_path_count =
      static_cast<std::size_t>(std::max(row.dominant_path_count, 0));
  return out;
}

void store_allostery_graph_edges_from_corr_knn(
    DCCMNetwork &dccm_analyzer, const TitanCorrKnnAnalyzer &analyzer) {
  const auto &result = analyzer.result();
  std::vector<DCCMNetwork::StoredEdge> gc_edges;
  std::vector<DCCMNetwork::StoredEdge> te_edges;
  gc_edges.reserve(result.candidate_pairs.size());
  te_edges.reserve(result.candidate_pairs.size());

  for (std::size_t pair_idx = 0; pair_idx < result.candidate_pairs.size();
       ++pair_idx) {
    const auto [i, j] = result.candidate_pairs[pair_idx];
    const int source = static_cast<int>(i);
    const int target = static_cast<int>(j);

    if (pair_idx < result.gc_values.size()) {
      const double gc = result.gc_values[pair_idx];
      if (gc > 0.35) {
        gc_edges.push_back({source, target, gc, gc});
      }
    }

    if (result.te_available && pair_idx < result.te_forward_values.size() &&
        pair_idx < result.te_reverse_values.size()) {
      const double net_te =
          result.te_forward_values[pair_idx] - result.te_reverse_values[pair_idx];
      if (std::abs(net_te) > 0.05) {
        const int driver = (net_te >= 0.0) ? source : target;
        const int sink = (net_te >= 0.0) ? target : source;
        te_edges.push_back({driver, sink, std::abs(net_te), std::abs(net_te)});
      }
    }
  }

  dccm_analyzer.set_last_allostery_graph_edges(std::move(gc_edges),
                                               std::move(te_edges));
}

std::vector<std::vector<int>>
build_msm_state_segments(const TitanMSM &msm_analyzer);

double tpt_transition_proximity(double q_value) {
  if (!std::isfinite(q_value))
    return 0.0;
  return std::clamp(1.0 - 2.0 * std::abs(q_value - 0.5), 0.0, 1.0);
}

std::uint64_t pack_directed_edge_key(int from, int to) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32) |
         static_cast<std::uint32_t>(to);
}

double percentile_sorted(std::vector<double> values, double frac) {
  if (values.empty())
    return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  if (values.size() == 1)
    return values.front();
  const double pos = std::clamp(frac, 0.0, 1.0) * (values.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  const double alpha = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - alpha) + values[hi] * alpha;
}

[[nodiscard]] bool keep_postproc_energy_time(const arguments &args,
                                             double time_ps) noexcept {
  if (args.flag_begin_set &&
      time_ps < args.begin_time - kEnergyTimeFilterTolerancePs) {
    return false;
  }
  if (args.flag_end_set &&
      time_ps > args.end_time + kEnergyTimeFilterTolerancePs) {
    return false;
  }
  if (args.dt > 0.0) {
    const double ref_time = args.flag_begin_set ? args.begin_time : 0.0;
    const double diff = time_ps - ref_time;
    if (diff < -kEnergyTimeFilterTolerancePs)
      return false;
    const long long steps = std::lround(diff / args.dt);
    if (std::abs(diff - static_cast<double>(steps) * args.dt) >
        kEnergyTimeFilterTolerancePs) {
      return false;
    }
  }
  return true;
}

std::vector<std::pair<double, double>>
filter_postproc_energy_pairs(const std::vector<std::pair<double, double>> &rows,
                             const arguments &args) {
  std::vector<std::pair<double, double>> filtered;
  filtered.reserve(rows.size());
  for (const auto &row : rows) {
    if (!keep_postproc_energy_time(args, row.first))
      continue;
    filtered.push_back(row);
  }
  return filtered;
}

void append_if_finite(std::vector<double> &dst, double value) {
  if (std::isfinite(value))
    dst.push_back(value);
}

std::string trim_copy(std::string value) {
  const auto first =
      std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
      });
  if (first == value.end())
    return {};
  const auto last =
      std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
      }).base();
  return std::string(first, last);
}

std::string to_lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

bool parse_scalar_condition(const std::string &expr, std::string &feature_name,
                            std::string &op, double &threshold,
                            std::string &error_message) {
  static const std::array<std::string, 4> kOps{{">=", "<=", ">", "<"}};
  const std::string trimmed = trim_copy(expr);
  for (const auto &candidate : kOps) {
    const auto pos = trimmed.find(candidate);
    if (pos == std::string::npos)
      continue;
    feature_name = to_lower_copy(trim_copy(trimmed.substr(0, pos)));
    op = candidate;
    const std::string rhs = trim_copy(trimmed.substr(pos + candidate.size()));
    if (feature_name.empty() || rhs.empty()) {
      error_message = "invalid LJ threshold expression '" + expr + "'";
      return false;
    }
    char *end_ptr = nullptr;
    const double value = std::strtod(rhs.c_str(), &end_ptr);
    if (end_ptr == rhs.c_str() || *end_ptr != '\0' || !std::isfinite(value)) {
      error_message = "invalid threshold value in '" + expr + "'";
      return false;
    }
    threshold = value;
    return true;
  }
  error_message = "missing comparison operator in '" + expr + "'";
  return false;
}

bool evaluate_scalar_condition(double value, const std::string &op,
                               double threshold) {
  if (!std::isfinite(value))
    return false;
  if (op == ">=")
    return value >= threshold;
  if (op == "<=")
    return value <= threshold;
  if (op == ">")
    return value > threshold;
  if (op == "<")
    return value < threshold;
  return false;
}

const std::vector<double> *
lookup_lj_feature_vector(const LjMicrostateSummary &summary,
                         const std::string &feature_name) {
  if (feature_name == "rho" || feature_name == "rho_bulk" ||
      feature_name == "rho_bulk_nm3") {
    return &summary.rho_bulk_nm3;
  }
  if (feature_name == "cluster_frac" ||
      feature_name == "largest_cluster_fraction") {
    return &summary.cluster_frac;
  }
  if (feature_name == "dense_frac" || feature_name == "dense_fraction") {
    return &summary.dense_frac;
  }
  if (feature_name == "q6_bar" || feature_name == "mean_dense_q6_bar" ||
      feature_name == "q6bar") {
    return &summary.q6_bar;
  }
  if (feature_name == "solid_like_frac" ||
      feature_name == "solid_like_fraction") {
    return &summary.solid_like_frac;
  }
  if (feature_name == "fcc_frac" || feature_name == "fcc_like_fraction") {
    return &summary.fcc_frac;
  }
  if (feature_name == "hcp_frac" || feature_name == "hcp_like_fraction") {
    return &summary.hcp_frac;
  }
  return nullptr;
}

double quantile_of_finite(const std::vector<double> &values, double frac) {
  std::vector<double> finite;
  finite.reserve(values.size());
  for (double v : values) {
    if (std::isfinite(v))
      finite.push_back(v);
  }
  return percentile_sorted(std::move(finite), frac);
}

std::vector<int> fallback_extreme_state_indices(const std::vector<double> &values,
                                                bool select_max) {
  std::vector<int> states;
  if (values.empty())
    return states;
  double best = select_max ? -std::numeric_limits<double>::infinity()
                           : std::numeric_limits<double>::infinity();
  for (double v : values) {
    if (!std::isfinite(v))
      continue;
    best = select_max ? std::max(best, v) : std::min(best, v);
  }
  if (!std::isfinite(best))
    return states;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double v = values[i];
    if (!std::isfinite(v))
      continue;
    if (std::abs(v - best) <= 1e-12)
      states.push_back(static_cast<int>(i));
  }
  return states;
}

bool resolve_lj_preset(const std::string &preset_name,
                       const LjMicrostateSummary &summary,
                       std::vector<int> &states_out,
                       std::string &resolved_label,
                       std::string &error_message) {
  states_out.clear();
  if (!summary.valid) {
    error_message = "LJ feature summaries are unavailable";
    return false;
  }
  const std::string name = to_lower_copy(trim_copy(preset_name));
  if (name.empty()) {
    error_message = "empty LJ preset name";
    return false;
  }

  const double cluster_q25 = quantile_of_finite(summary.cluster_frac, 0.25);
  const double cluster_q75 = quantile_of_finite(summary.cluster_frac, 0.75);
  const double dense_q25 = quantile_of_finite(summary.dense_frac, 0.25);
  const double dense_q75 = quantile_of_finite(summary.dense_frac, 0.75);
  const double q6_q75 = quantile_of_finite(summary.q6_bar, 0.75);
  const double solid_q75 = quantile_of_finite(summary.solid_like_frac, 0.75);

  auto push_if = [&](auto predicate) {
    for (std::size_t idx = 0; idx < summary.cluster_frac.size(); ++idx) {
      if (predicate(idx))
        states_out.push_back(static_cast<int>(idx));
    }
  };

  if (name == "gas" || name == "dilute") {
    push_if([&](std::size_t idx) {
      return std::isfinite(summary.cluster_frac[idx]) &&
             std::isfinite(summary.dense_frac[idx]) &&
             summary.cluster_frac[idx] <= cluster_q25 &&
             summary.dense_frac[idx] <= dense_q25;
    });
    if (states_out.empty())
      states_out = fallback_extreme_state_indices(summary.cluster_frac, false);
  } else if (name == "dense" || name == "condensed") {
    push_if([&](std::size_t idx) {
      return (std::isfinite(summary.cluster_frac[idx]) &&
              summary.cluster_frac[idx] >= cluster_q75) ||
             (std::isfinite(summary.dense_frac[idx]) &&
              summary.dense_frac[idx] >= dense_q75);
    });
    if (states_out.empty())
      states_out = fallback_extreme_state_indices(summary.cluster_frac, true);
  } else if (name == "liquid" || name == "dense_liquid") {
    push_if([&](std::size_t idx) {
      const bool dense =
          (std::isfinite(summary.cluster_frac[idx]) &&
           summary.cluster_frac[idx] >= cluster_q75) ||
          (std::isfinite(summary.dense_frac[idx]) &&
           summary.dense_frac[idx] >= dense_q75);
      const bool solid =
          (std::isfinite(summary.solid_like_frac[idx]) &&
           summary.solid_like_frac[idx] >= solid_q75) ||
          (std::isfinite(summary.q6_bar[idx]) &&
           summary.q6_bar[idx] >= q6_q75);
      return dense && !solid;
    });
    if (states_out.empty()) {
      for (std::size_t idx = 0; idx < summary.dense_frac.size(); ++idx) {
        const bool solid =
            (std::isfinite(summary.solid_like_frac[idx]) &&
             summary.solid_like_frac[idx] >= solid_q75) ||
            (std::isfinite(summary.q6_bar[idx]) &&
             summary.q6_bar[idx] >= q6_q75);
        if (!solid && std::isfinite(summary.dense_frac[idx]))
          states_out.push_back(static_cast<int>(idx));
      }
      if (states_out.empty())
        states_out = fallback_extreme_state_indices(summary.dense_frac, true);
    }
  } else if (name == "solid" || name == "ordered") {
    push_if([&](std::size_t idx) {
      return (std::isfinite(summary.solid_like_frac[idx]) &&
              summary.solid_like_frac[idx] >= solid_q75) ||
             (std::isfinite(summary.q6_bar[idx]) &&
              summary.q6_bar[idx] >= q6_q75);
    });
    if (states_out.empty())
      states_out = fallback_extreme_state_indices(summary.solid_like_frac, true);
  } else {
    error_message = "unsupported LJ preset '" + preset_name + "'";
    return false;
  }

  std::sort(states_out.begin(), states_out.end());
  states_out.erase(std::unique(states_out.begin(), states_out.end()),
                   states_out.end());
  if (states_out.empty()) {
    error_message = "LJ preset '" + preset_name +
                    "' resolved to zero microstates";
    return false;
  }
  resolved_label = "ljpreset:" + name;
  return true;
}

bool resolve_protein_preset(const std::string &preset_name,
                            const TitanMSM &msm_analyzer,
                            const std::vector<int> &macro_assignments,
                            const ProteinMicrostateSummary &protein_summary,
                            const Eigen::VectorXd &state_pi,
                            std::vector<int> &states_out,
                            std::string &resolved_label,
                            std::string &error_message) {
  states_out.clear();
  const std::string name = to_lower_copy(trim_copy(preset_name));
  if (name.empty()) {
    error_message = "empty protein preset name";
    return false;
  }

  auto centroid_quantile_select = [&](int dim, bool high_tail) {
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(msm_analyzer.n_clusters()));
    if (msm_analyzer.centroids().cols() <= dim)
      return false;
    for (int state = 0; state < msm_analyzer.n_clusters(); ++state)
      values.push_back(msm_analyzer.centroids()(state, dim));
    const double cutoff = quantile_of_finite(values, high_tail ? 0.75 : 0.25);
    for (int state = 0; state < msm_analyzer.n_clusters(); ++state) {
      const double value = msm_analyzer.centroids()(state, dim);
      if (!std::isfinite(value))
        continue;
      if ((!high_tail && value <= cutoff) || (high_tail && value >= cutoff))
        states_out.push_back(state);
    }
    if (states_out.empty())
      states_out = fallback_extreme_state_indices(values, high_tail);
    return !states_out.empty();
  };
  auto metric_quantile_select = [&](const std::vector<double> &values,
                                    bool high_tail,
                                    const std::string &metric_name) {
    if (values.size() != static_cast<std::size_t>(msm_analyzer.n_clusters())) {
      error_message = "protein preset '" + preset_name +
                      "' requires protein microstate " + metric_name +
                      " summary";
      return false;
    }
    const double cutoff = quantile_of_finite(values, high_tail ? 0.75 : 0.25);
    for (int state = 0; state < msm_analyzer.n_clusters(); ++state) {
      const double value = values[static_cast<std::size_t>(state)];
      if (!std::isfinite(value))
        continue;
      if ((!high_tail && value <= cutoff) || (high_tail && value >= cutoff))
        states_out.push_back(state);
    }
    if (states_out.empty())
      states_out = fallback_extreme_state_indices(values, high_tail);
    return !states_out.empty();
  };

  if (name == "left" || name == "low_pc1") {
    if (!centroid_quantile_select(0, false)) {
      error_message = "protein preset '" + preset_name +
                      "' requires at least one TICA component";
      return false;
    }
  } else if (name == "right" || name == "high_pc1") {
    if (!centroid_quantile_select(0, true)) {
      error_message = "protein preset '" + preset_name +
                      "' requires at least one TICA component";
      return false;
    }
  } else if (name == "low_pc2") {
    if (!centroid_quantile_select(1, false)) {
      error_message = "protein preset '" + preset_name +
                      "' requires at least two TICA components";
      return false;
    }
  } else if (name == "high_pc2") {
    if (!centroid_quantile_select(1, true)) {
      error_message = "protein preset '" + preset_name +
                      "' requires at least two TICA components";
      return false;
    }
  } else if (name == "rmsd_low") {
    if (!protein_summary.valid_rmsd ||
        !metric_quantile_select(protein_summary.rmsd_mean, false, "RMSD")) {
      if (error_message.empty())
        error_message = "protein preset '" + preset_name + "' requires -rms";
      return false;
    }
  } else if (name == "rmsd_high") {
    if (!protein_summary.valid_rmsd ||
        !metric_quantile_select(protein_summary.rmsd_mean, true, "RMSD")) {
      if (error_message.empty())
        error_message = "protein preset '" + preset_name + "' requires -rms";
      return false;
    }
  } else if (name == "helical" || name == "helix") {
    if (!protein_summary.valid_dssp ||
        !metric_quantile_select(protein_summary.helix_frac, true,
                                "DSSP helix fraction")) {
      if (error_message.empty())
        error_message = "protein preset '" + preset_name + "' requires -dssp";
      return false;
    }
  } else if (name == "strand" || name == "beta") {
    if (!protein_summary.valid_dssp ||
        !metric_quantile_select(protein_summary.strand_frac, true,
                                "DSSP strand fraction")) {
      if (error_message.empty())
        error_message = "protein preset '" + preset_name + "' requires -dssp";
      return false;
    }
  } else if (name == "coil" || name == "disordered_like") {
    if (!protein_summary.valid_dssp ||
        !metric_quantile_select(protein_summary.coil_frac, true,
                                "DSSP coil fraction")) {
      if (error_message.empty())
        error_message = "protein preset '" + preset_name + "' requires -dssp";
      return false;
    }
  } else if (name == "turn") {
    if (!protein_summary.valid_dssp ||
        !metric_quantile_select(protein_summary.turn_frac, true,
                                "DSSP turn fraction")) {
      if (error_message.empty())
        error_message = "protein preset '" + preset_name + "' requires -dssp";
      return false;
    }
  } else if (name == "bend") {
    if (!protein_summary.valid_dssp ||
        !metric_quantile_select(protein_summary.bend_frac, true,
                                "DSSP bend fraction")) {
      if (error_message.empty())
        error_message = "protein preset '" + preset_name + "' requires -dssp";
      return false;
    }
  } else if (name == "major" || name == "major_macro" ||
             name == "native_like" || name == "minor" ||
             name == "minor_macro" || name == "rare_macro") {
    if (macro_assignments.empty() ||
        macro_assignments.size() != static_cast<std::size_t>(msm_analyzer.n_clusters())) {
      error_message = "protein preset '" + preset_name +
                      "' requires PCCA macrostate assignments";
      return false;
    }
    const int n_macro =
        *std::max_element(macro_assignments.begin(), macro_assignments.end()) + 1;
    std::vector<double> macro_pop(static_cast<std::size_t>(n_macro), 0.0);
    for (int state = 0; state < msm_analyzer.n_clusters(); ++state) {
      const int macro = macro_assignments[static_cast<std::size_t>(state)];
      const double pi =
          (state < state_pi.size() && std::isfinite(state_pi[state]) &&
           state_pi[state] > 0.0)
              ? state_pi[state]
              : 0.0;
      if (macro >= 0 && macro < n_macro)
        macro_pop[static_cast<std::size_t>(macro)] += pi;
    }
    int target_macro = -1;
    double target_value = (name == "major" || name == "major_macro" ||
                           name == "native_like")
                              ? -std::numeric_limits<double>::infinity()
                              : std::numeric_limits<double>::infinity();
    for (int macro = 0; macro < n_macro; ++macro) {
      const double pop = macro_pop[static_cast<std::size_t>(macro)];
      if ((name == "major" || name == "major_macro" || name == "native_like")) {
        if (pop > target_value) {
          target_value = pop;
          target_macro = macro;
        }
      } else if (pop > 0.0 && pop < target_value) {
        target_value = pop;
        target_macro = macro;
      }
    }
    if (target_macro < 0) {
      error_message = "protein preset '" + preset_name +
                      "' could not identify a target macrostate";
      return false;
    }
    for (int state = 0; state < msm_analyzer.n_clusters(); ++state) {
      if (macro_assignments[static_cast<std::size_t>(state)] == target_macro)
        states_out.push_back(state);
    }
  } else {
    error_message = "unsupported protein preset '" + preset_name + "'";
    return false;
  }

  std::sort(states_out.begin(), states_out.end());
  states_out.erase(std::unique(states_out.begin(), states_out.end()),
                   states_out.end());
  if (states_out.empty()) {
    error_message = "protein preset '" + preset_name +
                    "' resolved to zero microstates";
    return false;
  }
  resolved_label = "protein:" + name;
  return true;
}

bool load_lj_feature_rows(const std::vector<std::string> &feature_files,
                          size_t expected_frames,
                          std::vector<std::array<double, kLjFeatureColumns>>
                              &rows_out,
                          std::fstream &log) {
  rows_out.clear();
  rows_out.reserve(expected_frames);
  for (const auto &path : feature_files) {
    std::ifstream in(path);
    if (!in.is_open()) {
      TitanUI::print_warning("TPT: cannot open LJ feature file " + path +
                             "; LJ-threshold basins will be unavailable.");
      return false;
    }
    std::string line;
    while (std::getline(in, line)) {
      line = trim_copy(line);
      if (line.empty() || line[0] == '#')
        continue;
      std::stringstream ss(line);
      int frame_index = 0;
      double time_ps = 0.0;
      if (!(ss >> frame_index >> time_ps))
        continue;
      std::array<double, kLjFeatureColumns> row{};
      bool ok = true;
      for (int col = 0; col < kLjFeatureColumns; ++col) {
        if (!(ss >> row[static_cast<size_t>(col)])) {
          ok = false;
          break;
        }
      }
      if (ok)
        rows_out.push_back(row);
    }
  }

  if (rows_out.empty()) {
    TitanUI::print_warning(
        "TPT: no LJ feature rows were read; LJ-threshold basins will be unavailable.");
    return false;
  }

  if (rows_out.size() != expected_frames) {
    TitanUI::print_warning(
        "TPT: LJ feature frame count (" + std::to_string(rows_out.size()) +
        ") does not match MSM discrete trajectory length (" +
        std::to_string(expected_frames) + ").");
    TitanUI::log_kv(log, "TPT LJ Feature Frames",
                    std::to_string(rows_out.size()));
    TitanUI::log_kv(log, "TPT MSM Frames",
                    std::to_string(expected_frames));
    if (rows_out.size() < expected_frames)
      return false;
    rows_out.resize(expected_frames);
  }
  return true;
}

LjMicrostateSummary compute_lj_microstate_summary(
    const TitanMSM &msm_analyzer,
    const std::vector<std::string> &feature_files,
    std::fstream &log) {
  LjMicrostateSummary result;
  const auto &dtraj = msm_analyzer.discrete_traj();
  if (feature_files.empty() || dtraj.empty())
    return result;

  std::vector<std::array<double, kLjFeatureColumns>> rows;
  if (!load_lj_feature_rows(feature_files, dtraj.size(), rows, log))
    return result;

  const int n_states = msm_analyzer.n_clusters();
  auto make_nan_vector = [n_states]() {
    return std::vector<double>(static_cast<size_t>(n_states),
                               std::numeric_limits<double>::quiet_NaN());
  };
  result.rho_bulk_nm3 = make_nan_vector();
  result.cluster_frac = make_nan_vector();
  result.dense_frac = make_nan_vector();
  result.q6_bar = make_nan_vector();
  result.solid_like_frac = make_nan_vector();
  result.fcc_frac = make_nan_vector();
  result.hcp_frac = make_nan_vector();

  std::vector<double> sum_rho(static_cast<size_t>(n_states), 0.0);
  std::vector<double> sum_cluster(static_cast<size_t>(n_states), 0.0);
  std::vector<double> sum_dense(static_cast<size_t>(n_states), 0.0);
  std::vector<double> sum_q6_bar(static_cast<size_t>(n_states), 0.0);
  std::vector<double> sum_solid(static_cast<size_t>(n_states), 0.0);
  std::vector<double> sum_fcc(static_cast<size_t>(n_states), 0.0);
  std::vector<double> sum_hcp(static_cast<size_t>(n_states), 0.0);
  std::vector<std::uint64_t> count(static_cast<size_t>(n_states), 0);

  const size_t n_frames = std::min(rows.size(), dtraj.size());
  for (size_t frame = 0; frame < n_frames; ++frame) {
    const int state = dtraj[frame];
    if (state < 0 || state >= n_states)
      continue;
    const size_t idx = static_cast<size_t>(state);
    const auto &row = rows[frame];
    sum_rho[idx] += row[0];
    sum_cluster[idx] += row[1];
    sum_dense[idx] += row[8];
    sum_q6_bar[idx] += row[12];
    sum_solid[idx] += row[13];
    sum_fcc[idx] += row[15];
    sum_hcp[idx] += row[16];
    ++count[idx];
  }

  for (int state = 0; state < n_states; ++state) {
    const size_t idx = static_cast<size_t>(state);
    if (count[idx] == 0)
      continue;
    const double inv_count = 1.0 / static_cast<double>(count[idx]);
    result.rho_bulk_nm3[idx] = sum_rho[idx] * inv_count;
    result.cluster_frac[idx] = sum_cluster[idx] * inv_count;
    result.dense_frac[idx] = sum_dense[idx] * inv_count;
    result.q6_bar[idx] = sum_q6_bar[idx] * inv_count;
    result.solid_like_frac[idx] = sum_solid[idx] * inv_count;
    result.fcc_frac[idx] = sum_fcc[idx] * inv_count;
    result.hcp_frac[idx] = sum_hcp[idx] * inv_count;
  }

  result.valid = true;
  return result;
}

ProteinMicrostateSummary compute_protein_microstate_summary(
    const TitanMSM &msm_analyzer, const TitanRMSAnalyzer *rms_analyzer,
    const TitanDSSPAnalyzer *dssp_analyzer, std::fstream &log) {
  ProteinMicrostateSummary result;
  const auto segments = build_msm_state_segments(msm_analyzer);
  if (segments.empty())
    return result;

  const int n_states = msm_analyzer.n_clusters();
  auto make_nan_vector = [n_states]() {
    return std::vector<double>(static_cast<std::size_t>(n_states),
                               std::numeric_limits<double>::quiet_NaN());
  };
  result.rmsd_mean = make_nan_vector();
  result.helix_frac = make_nan_vector();
  result.strand_frac = make_nan_vector();
  result.coil_frac = make_nan_vector();
  result.turn_frac = make_nan_vector();
  result.bend_frac = make_nan_vector();

  if (rms_analyzer && !rms_analyzer->trajectory_series().empty()) {
    const auto &rms_trajs = rms_analyzer->trajectory_series();
    std::vector<double> sum_rmsd(static_cast<std::size_t>(n_states), 0.0);
    std::vector<std::uint64_t> count_rmsd(static_cast<std::size_t>(n_states), 0);
    const std::size_t n_traj = std::min(segments.size(), rms_trajs.size());
    bool warned_mismatch = false;
    for (std::size_t traj = 0; traj < n_traj; ++traj) {
      const auto &seg = segments[traj];
      const auto &rmsd_series = rms_trajs[traj].rmsd_series;
      const std::size_t n_frames = std::min(seg.size(), rmsd_series.size());
      if (!warned_mismatch && seg.size() != rmsd_series.size()) {
        TitanUI::print_warning(
            "TPT protein presets: RMSD trajectory length does not exactly "
            "match the MSM segment length. TITAN will use the shared prefix.");
        warned_mismatch = true;
      }
      for (std::size_t frame = 0; frame < n_frames; ++frame) {
        const int state = seg[frame];
        if (state < 0 || state >= n_states)
          continue;
        const double value = rmsd_series[frame];
        if (!std::isfinite(value))
          continue;
        const std::size_t idx = static_cast<std::size_t>(state);
        sum_rmsd[idx] += value;
        ++count_rmsd[idx];
      }
    }
    for (int state = 0; state < n_states; ++state) {
      const std::size_t idx = static_cast<std::size_t>(state);
      if (count_rmsd[idx] == 0)
        continue;
      result.rmsd_mean[idx] =
          sum_rmsd[idx] / static_cast<double>(count_rmsd[idx]);
    }
    result.valid_rmsd =
        std::any_of(count_rmsd.begin(), count_rmsd.end(),
                    [](std::uint64_t c) { return c > 0; });
    if (result.valid_rmsd) {
      TitanUI::log_kv(log, "TPT Protein RMSD Summary", "available");
    }
  }

  if (dssp_analyzer && !dssp_analyzer->trajectory_frame_profiles().empty()) {
    const auto &dssp_trajs = dssp_analyzer->trajectory_frame_profiles();
    std::vector<double> sum_helix(static_cast<std::size_t>(n_states), 0.0);
    std::vector<double> sum_strand(static_cast<std::size_t>(n_states), 0.0);
    std::vector<double> sum_coil(static_cast<std::size_t>(n_states), 0.0);
    std::vector<double> sum_turn(static_cast<std::size_t>(n_states), 0.0);
    std::vector<double> sum_bend(static_cast<std::size_t>(n_states), 0.0);
    std::vector<std::uint64_t> count_dssp(static_cast<std::size_t>(n_states), 0);
    const std::size_t n_traj = std::min(segments.size(), dssp_trajs.size());
    bool warned_mismatch = false;
    for (std::size_t traj = 0; traj < n_traj; ++traj) {
      const auto &seg = segments[traj];
      const auto &profiles = dssp_trajs[traj];
      const std::size_t n_frames = std::min(seg.size(), profiles.size());
      if (!warned_mismatch && seg.size() != profiles.size()) {
        TitanUI::print_warning(
            "TPT protein presets: DSSP trajectory length does not exactly "
            "match the MSM segment length. TITAN will use the shared prefix.");
        warned_mismatch = true;
      }
      for (std::size_t frame = 0; frame < n_frames; ++frame) {
        const int state = seg[frame];
        if (state < 0 || state >= n_states)
          continue;
        const auto &profile = profiles[frame];
        const std::size_t idx = static_cast<std::size_t>(state);
        sum_helix[idx] += profile[0] + profile[3] + profile[4];
        sum_strand[idx] += profile[1] + profile[2];
        sum_turn[idx] += profile[5];
        sum_bend[idx] += profile[6];
        sum_coil[idx] += profile[7];
        ++count_dssp[idx];
      }
    }
    for (int state = 0; state < n_states; ++state) {
      const std::size_t idx = static_cast<std::size_t>(state);
      if (count_dssp[idx] == 0)
        continue;
      const double inv_count = 1.0 / static_cast<double>(count_dssp[idx]);
      result.helix_frac[idx] = sum_helix[idx] * inv_count;
      result.strand_frac[idx] = sum_strand[idx] * inv_count;
      result.coil_frac[idx] = sum_coil[idx] * inv_count;
      result.turn_frac[idx] = sum_turn[idx] * inv_count;
      result.bend_frac[idx] = sum_bend[idx] * inv_count;
    }
    result.valid_dssp =
        std::any_of(count_dssp.begin(), count_dssp.end(),
                    [](std::uint64_t c) { return c > 0; });
    if (result.valid_dssp) {
      TitanUI::log_kv(log, "TPT Protein DSSP Summary", "available");
    }
  }

  return result;
}

QNativeMicrostateSummary compute_qnative_microstate_summary(
    const TitanMSM &msm_analyzer, const TitanQNativeAnalyzer *qnative_analyzer,
    std::fstream &log) {
  QNativeMicrostateSummary result;
  const int n_states = msm_analyzer.n_clusters();
  result.mean_q.assign(static_cast<std::size_t>(n_states),
                       std::numeric_limits<double>::quiet_NaN());
  if (!qnative_analyzer)
    return result;
  const auto &q_values = qnative_analyzer->all_q_values();
  const auto &states = msm_analyzer.discrete_traj();
  if (q_values.empty() || states.empty())
    return result;
  const std::size_t n_frames = std::min(q_values.size(), states.size());
  if (q_values.size() != states.size()) {
    TitanUI::print_warning(
        "TPT qnative selectors: qnative trajectory length does not exactly "
        "match the MSM discrete trajectory. TITAN will use the shared prefix.");
  }
  std::vector<double> sum(static_cast<std::size_t>(n_states), 0.0);
  std::vector<std::uint64_t> count(static_cast<std::size_t>(n_states), 0);
  for (std::size_t frame = 0; frame < n_frames; ++frame) {
    const int state = states[frame];
    if (state < 0 || state >= n_states)
      continue;
    const double q = q_values[frame];
    if (!std::isfinite(q))
      continue;
    const std::size_t idx = static_cast<std::size_t>(state);
    sum[idx] += q;
    ++count[idx];
  }
  for (int state = 0; state < n_states; ++state) {
    const std::size_t idx = static_cast<std::size_t>(state);
    if (count[idx] == 0)
      continue;
    result.mean_q[idx] = sum[idx] / static_cast<double>(count[idx]);
  }
  result.valid = std::any_of(count.begin(), count.end(),
                             [](std::uint64_t c) { return c > 0; });
  if (result.valid)
    TitanUI::log_kv(log, "TPT QNative Summary", "available");
  return result;
}

bool resolve_tpt_basin_spec(const std::string &spec, const TitanMSM &msm_analyzer,
                            const std::vector<int> &macro_assignments,
                            const LjMicrostateSummary &lj_summary,
                            const ProteinMicrostateSummary &protein_summary,
                            const QNativeMicrostateSummary &qnative_summary,
                            const Eigen::VectorXd &state_pi,
                            std::vector<int> &states_out,
                            std::string &resolved_label,
                            std::string &error_message) {
  states_out.clear();
  resolved_label = trim_copy(spec);
  error_message.clear();

  const int n_states = msm_analyzer.n_clusters();
  std::string trimmed = trim_copy(spec);
  const auto colon_pos = trimmed.find(':');
  std::string prefix =
      (colon_pos == std::string::npos)
          ? std::string("micro")
          : to_lower_copy(trim_copy(trimmed.substr(0, colon_pos)));

  if (prefix == "micro" || prefix == "state" || prefix == "states") {
    return TitanTPT::parse_state_spec(trimmed, n_states, states_out,
                                      &error_message);
  }

  const std::string body =
      (colon_pos == std::string::npos) ? trimmed
                                       : trim_copy(trimmed.substr(colon_pos + 1));
  if (prefix == "macro") {
    std::vector<int> macros;
    if (!TitanTPT::parse_state_spec("micro:" + body, -1, macros,
                                    &error_message)) {
      error_message = "invalid macro basin: " + error_message;
      return false;
    }
    if (macro_assignments.size() != static_cast<size_t>(n_states)) {
      error_message = "macrostate assignments are unavailable";
      return false;
    }
    for (int state = 0; state < n_states; ++state) {
      const int macro = macro_assignments[static_cast<size_t>(state)];
      if (std::find(macros.begin(), macros.end(), macro) != macros.end()) {
        states_out.push_back(state);
      }
    }
    if (states_out.empty()) {
      error_message = "macro basin " + spec + " resolved to zero microstates";
      return false;
    }
    return true;
  }

  if (prefix == "tica_rect") {
    std::stringstream ss(body);
    std::string token;
    std::vector<double> values;
    while (std::getline(ss, token, ',')) {
      token = trim_copy(token);
      if (token.empty()) {
        error_message = "invalid empty coordinate in " + spec;
        return false;
      }
      char *end_ptr = nullptr;
      const double value = std::strtod(token.c_str(), &end_ptr);
      if (end_ptr == token.c_str() || *end_ptr != '\0' || !std::isfinite(value)) {
        error_message = "invalid rectangle coordinate in " + spec;
        return false;
      }
      values.push_back(value);
    }
    if (values.size() != 4) {
      error_message = "tica_rect expects four comma-separated values";
      return false;
    }
    const double x_min = std::min(values[0], values[2]);
    const double x_max = std::max(values[0], values[2]);
    const double y_min = std::min(values[1], values[3]);
    const double y_max = std::max(values[1], values[3]);
    if (msm_analyzer.centroids().cols() < 2) {
      error_message = "TICA rectangle basins require at least 2 TICA components";
      return false;
    }
    for (int state = 0; state < n_states; ++state) {
      const double pc1 = msm_analyzer.centroids()(state, 0);
      const double pc2 = msm_analyzer.centroids()(state, 1);
      if (pc1 >= x_min && pc1 <= x_max && pc2 >= y_min && pc2 <= y_max) {
        states_out.push_back(state);
      }
    }
    if (states_out.empty()) {
      error_message = "tica_rect basin " + spec + " resolved to zero microstates";
      return false;
    }
    return true;
  }

  if (prefix == "lj") {
    if (!lj_summary.valid) {
      error_message =
          "LJ feature summaries are unavailable for lj: basin specifications";
      return false;
    }
    std::string feature_name;
    std::string op;
    double threshold = 0.0;
    std::string scalar_error;
    if (!parse_scalar_condition(body, feature_name, op, threshold,
                                scalar_error)) {
      if (resolve_lj_preset(body, lj_summary, states_out, resolved_label,
                            error_message)) {
        return true;
      }
      error_message = "invalid lj: basin: " + scalar_error + "; " + error_message;
      return false;
    }
    const auto *feature_values =
        lookup_lj_feature_vector(lj_summary, feature_name);
    if (!feature_values) {
      error_message = "unsupported LJ feature '" + feature_name +
                      "' in basin specification";
      return false;
    }
    for (int state = 0; state < n_states; ++state) {
      if (evaluate_scalar_condition((*feature_values)[static_cast<size_t>(state)],
                                    op, threshold)) {
        states_out.push_back(state);
      }
    }
    if (states_out.empty()) {
      error_message = "LJ basin " + spec + " resolved to zero microstates";
      return false;
    }
    return true;
  }

  if (prefix == "ljpreset" || prefix == "lj_preset") {
    return resolve_lj_preset(body, lj_summary, states_out, resolved_label,
                             error_message);
  }

  if (prefix == "protein" || prefix == "prot" || prefix == "protein_preset") {
    return resolve_protein_preset(body, msm_analyzer, macro_assignments,
                                  protein_summary, state_pi, states_out,
                                  resolved_label, error_message);
  }

  if (prefix == "qnative_range" || prefix == "qnative_lt" ||
      prefix == "qnative_gt") {
    if (!qnative_summary.valid) {
      error_message =
          "qnative summaries are unavailable; enable -qnative or use a qnative selector so TITAN can collect Q";
      return false;
    }
    auto parse_unit = [&](const std::string &raw, double &value_out,
                          const char *name) {
      char *end_ptr = nullptr;
      value_out = std::strtod(raw.c_str(), &end_ptr);
      if (end_ptr == raw.c_str() || *end_ptr != '\0' ||
          !std::isfinite(value_out) || value_out < 0.0 ||
          value_out > 1.0) {
        error_message = std::string("invalid qnative ") + name +
                        " in basin specification";
        return false;
      }
      return true;
    };
    double q_min = 0.0;
    double q_max = 1.0;
    if (prefix == "qnative_range") {
      const auto comma = body.find(',');
      if (comma == std::string::npos ||
          body.find(',', comma + 1) != std::string::npos) {
        error_message = "qnative_range expects min,max";
        return false;
      }
      if (!parse_unit(trim_copy(body.substr(0, comma)), q_min, "range minimum") ||
          !parse_unit(trim_copy(body.substr(comma + 1)), q_max,
                      "range maximum"))
        return false;
      if (q_min > q_max) {
        error_message = "qnative_range minimum must be <= maximum";
        return false;
      }
    } else {
      double threshold = 0.0;
      if (!parse_unit(trim_copy(body), threshold, "threshold"))
        return false;
      if (prefix == "qnative_lt") {
        q_min = -std::numeric_limits<double>::infinity();
        q_max = threshold;
      } else {
        q_min = threshold;
        q_max = std::numeric_limits<double>::infinity();
      }
    }
    for (int state = 0; state < n_states; ++state) {
      const double q = qnative_summary.mean_q[static_cast<std::size_t>(state)];
      if (!std::isfinite(q))
        continue;
      if (prefix == "qnative_lt") {
        if (q < q_max)
          states_out.push_back(state);
      } else if (prefix == "qnative_gt") {
        if (q > q_min)
          states_out.push_back(state);
      } else if (q >= q_min && q <= q_max) {
        states_out.push_back(state);
      }
    }
    if (states_out.empty()) {
      error_message = "qnative basin " + spec + " resolved to zero microstates";
      return false;
    }
    return true;
  }

  error_message = "unsupported basin prefix '" + prefix + "'";
  return false;
}

void write_tpt_state_labels(const std::string &filename,
                            const TitanTPT &tpt_analyzer,
                            const TitanMSM &msm_analyzer,
                            const std::vector<int> &macro_assignments,
                            const LjMicrostateSummary &lj_summary,
                            const ProteinMicrostateSummary &protein_summary,
                            const QNativeMicrostateSummary &qnative_summary,
                            const Eigen::VectorXd &state_pi) {
  std::ofstream out(filename);
  const size_t n_states = static_cast<size_t>(tpt_analyzer.n_states());
  const bool include_lj_metrics =
      lj_summary.valid && lj_summary.rho_bulk_nm3.size() == n_states &&
      lj_summary.cluster_frac.size() == n_states &&
      lj_summary.dense_frac.size() == n_states &&
      lj_summary.q6_bar.size() == n_states &&
      lj_summary.solid_like_frac.size() == n_states &&
      lj_summary.fcc_frac.size() == n_states &&
      lj_summary.hcp_frac.size() == n_states;
  const bool include_protein_metrics =
      (protein_summary.valid_rmsd || protein_summary.valid_dssp) &&
      protein_summary.rmsd_mean.size() == n_states &&
      protein_summary.helix_frac.size() == n_states &&
      protein_summary.strand_frac.size() == n_states &&
      protein_summary.coil_frac.size() == n_states &&
      protein_summary.turn_frac.size() == n_states &&
      protein_summary.bend_frac.size() == n_states;
  const bool include_qnative_metrics =
      qnative_summary.valid && qnative_summary.mean_q.size() == n_states;
  out << "# state set macrostate pi q_plus q_minus reactive_density tica_c1 "
         "tica_c2";
  if (include_lj_metrics) {
    out << " lj_rho lj_cluster_frac lj_dense_frac lj_q6_bar "
           "lj_solid_like_frac lj_fcc_frac lj_hcp_frac";
  }
  if (include_protein_metrics) {
    out << " protein_rmsd protein_helix_frac protein_strand_frac "
           "protein_coil_frac protein_turn_frac protein_bend_frac";
  }
  if (include_qnative_metrics) {
    out << " qnative_mean_Q";
  }
  out << '\n';
  out << std::setprecision(15);
  const auto &source = tpt_analyzer.source_states();
  const auto &sink = tpt_analyzer.sink_states();
  const auto &pi = tpt_analyzer.summary();
  (void)pi;
  for (int state = 0; state < tpt_analyzer.n_states(); ++state) {
    char label = 'I';
    if (std::find(source.begin(), source.end(), state) != source.end()) {
      label = 'A';
    } else if (std::find(sink.begin(), sink.end(), state) != sink.end()) {
      label = 'B';
    }
    const int macro =
        (macro_assignments.size() == static_cast<size_t>(tpt_analyzer.n_states()))
            ? macro_assignments[static_cast<size_t>(state)]
            : -1;
    const double pc1 =
        (msm_analyzer.centroids().cols() > 0)
            ? msm_analyzer.centroids()(state, 0)
            : std::numeric_limits<double>::quiet_NaN();
    const double pc2 =
        (msm_analyzer.centroids().cols() > 1)
            ? msm_analyzer.centroids()(state, 1)
            : std::numeric_limits<double>::quiet_NaN();
    out << state << ' ' << label << ' ' << macro << ' '
        << state_pi[state] << ' '
        << tpt_analyzer.forward_committor()[state] << ' '
        << tpt_analyzer.backward_committor()[state] << ' '
        << tpt_analyzer.reactive_density()[state] << ' ' << pc1 << ' ' << pc2;
    if (include_lj_metrics) {
      const size_t idx = static_cast<size_t>(state);
      out << ' ' << lj_summary.rho_bulk_nm3[idx] << ' '
          << lj_summary.cluster_frac[idx] << ' '
          << lj_summary.dense_frac[idx] << ' '
          << lj_summary.q6_bar[idx] << ' '
          << lj_summary.solid_like_frac[idx] << ' '
          << lj_summary.fcc_frac[idx] << ' '
          << lj_summary.hcp_frac[idx];
    }
    if (include_protein_metrics) {
      const size_t idx = static_cast<size_t>(state);
      out << ' ' << protein_summary.rmsd_mean[idx] << ' '
          << protein_summary.helix_frac[idx] << ' '
          << protein_summary.strand_frac[idx] << ' '
          << protein_summary.coil_frac[idx] << ' '
          << protein_summary.turn_frac[idx] << ' '
          << protein_summary.bend_frac[idx];
    }
    if (include_qnative_metrics) {
      out << ' ' << qnative_summary.mean_q[static_cast<size_t>(state)];
    }
    out << '\n';
  }
}

double estimate_tpt_projection_bandwidth(const TitanMSM &msm_analyzer) {
  if (msm_analyzer.centroids().cols() < 2 || msm_analyzer.n_clusters() <= 1) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  std::vector<double> nearest;
  nearest.reserve(static_cast<size_t>(msm_analyzer.n_clusters()));
  for (int i = 0; i < msm_analyzer.n_clusters(); ++i) {
    const double xi = msm_analyzer.centroids()(i, 0);
    const double yi = msm_analyzer.centroids()(i, 1);
    if (!std::isfinite(xi) || !std::isfinite(yi))
      continue;

    double best = std::numeric_limits<double>::infinity();
    for (int j = 0; j < msm_analyzer.n_clusters(); ++j) {
      if (i == j)
        continue;
      const double xj = msm_analyzer.centroids()(j, 0);
      const double yj = msm_analyzer.centroids()(j, 1);
      if (!std::isfinite(xj) || !std::isfinite(yj))
        continue;
      const double dx = xi - xj;
      const double dy = yi - yj;
      const double dist = std::sqrt(dx * dx + dy * dy);
      if (dist < best)
        best = dist;
    }
    if (std::isfinite(best) && best > 0.0)
      nearest.push_back(best);
  }

  if (nearest.empty())
    return std::numeric_limits<double>::quiet_NaN();
  const size_t mid = nearest.size() / 2;
  std::nth_element(nearest.begin(), nearest.begin() + mid, nearest.end());
  return nearest[mid];
}

bool write_tpt_reactive_fes_2d(const std::string &filename,
                               const TitanTPT &tpt_analyzer,
                               const TitanMSM &msm_analyzer,
                               std::fstream &log) {
  constexpr int kGrid = 80;
  if (msm_analyzer.centroids().cols() < 2 || tpt_analyzer.n_states() <= 0) {
    TitanUI::print_warning(
        "TPT: skipping reactive FES export because fewer than 2 TICA coordinates are available.");
    return false;
  }

  double x_min = std::numeric_limits<double>::infinity();
  double x_max = -std::numeric_limits<double>::infinity();
  double y_min = std::numeric_limits<double>::infinity();
  double y_max = -std::numeric_limits<double>::infinity();
  double total_state_weight = 0.0;
  bool have_centroids = false;

  for (int state = 0; state < tpt_analyzer.n_states(); ++state) {
    const double x = msm_analyzer.centroids()(state, 0);
    const double y = msm_analyzer.centroids()(state, 1);
    const double w = std::max(0.0, tpt_analyzer.reactive_density()[state]);
    if (!std::isfinite(x) || !std::isfinite(y))
      continue;
    x_min = std::min(x_min, x);
    x_max = std::max(x_max, x);
    y_min = std::min(y_min, y);
    y_max = std::max(y_max, y);
    total_state_weight += w;
    have_centroids = true;
  }

  if (!have_centroids || total_state_weight <= 0.0) {
    TitanUI::print_warning(
        "TPT: skipping reactive FES export because no positive reactive state weights are available.");
    return false;
  }

  double span_x = x_max - x_min;
  double span_y = y_max - y_min;
  const double span_ref = std::max(span_x, span_y);
  const double pad = std::max(0.05 * span_ref, 1e-3);
  x_min -= pad;
  x_max += pad;
  y_min -= pad;
  y_max += pad;
  span_x = std::max(x_max - x_min, 1e-6);
  span_y = std::max(y_max - y_min, 1e-6);

  const double dx = span_x / static_cast<double>(kGrid - 1);
  const double dy = span_y / static_cast<double>(kGrid - 1);
  const double nearest_bandwidth =
      estimate_tpt_projection_bandwidth(msm_analyzer);
  const double sigma = std::max(
      std::max(dx, dy) * 1.5,
      std::isfinite(nearest_bandwidth) ? 0.5 * nearest_bandwidth
                                       : 0.1 * std::max(span_x, span_y));
  const double sigma2 = sigma * sigma;

  std::vector<double> rho_grid(static_cast<size_t>(kGrid * kGrid), 0.0);
  double max_rho = 0.0;
  double sum_rho = 0.0;

  for (int ix = 0; ix < kGrid; ++ix) {
    const double x = x_min + dx * ix;
    for (int iy = 0; iy < kGrid; ++iy) {
      const double y = y_min + dy * iy;
      double rho = 0.0;
      for (int state = 0; state < tpt_analyzer.n_states(); ++state) {
        const double w = std::max(0.0, tpt_analyzer.reactive_density()[state]);
        if (w <= 0.0)
          continue;
        const double cx = msm_analyzer.centroids()(state, 0);
        const double cy = msm_analyzer.centroids()(state, 1);
        if (!std::isfinite(cx) || !std::isfinite(cy))
          continue;
        const double ddx = x - cx;
        const double ddy = y - cy;
        rho += w * std::exp(-0.5 * (ddx * ddx + ddy * ddy) / sigma2);
      }
      rho_grid[static_cast<size_t>(ix * kGrid + iy)] = rho;
      max_rho = std::max(max_rho, rho);
      sum_rho += rho;
    }
  }

  if (!(max_rho > 0.0) || !(sum_rho > 0.0)) {
    TitanUI::print_warning(
        "TPT: skipping reactive FES export because the projected reactive density grid is empty.");
    return false;
  }

  std::ofstream out(filename);
  out << "# x y reactive_fes_kbt reactive_prob reactive_density\n";
  out << "# grid " << kGrid << " " << kGrid << "\n";
  out << "# sigma " << std::setprecision(15) << sigma << "\n";
  out << std::setprecision(15);
  for (int ix = 0; ix < kGrid; ++ix) {
    const double x = x_min + dx * ix;
    for (int iy = 0; iy < kGrid; ++iy) {
      const double y = y_min + dy * iy;
      const double rho = rho_grid[static_cast<size_t>(ix * kGrid + iy)];
      const double prob = rho / sum_rho;
      const double fes =
          (rho > 0.0) ? -std::log(rho / max_rho)
                      : std::numeric_limits<double>::infinity();
      out << x << ' ' << y << ' ' << fes << ' ' << prob << ' ' << rho
          << '\n';
    }
  }
  TitanUI::log_kv(log, "TPT Reactive FES Grid",
                  std::to_string(kGrid) + " x " + std::to_string(kGrid));
  TitanUI::log_kv(log, "TPT Reactive FES Sigma", std::to_string(sigma));
  return true;
}

bool write_tpt_flux_map_2d(const std::string &filename,
                           const TitanTPT &tpt_analyzer,
                           const TitanMSM &msm_analyzer) {
  if (msm_analyzer.centroids().cols() < 2 || tpt_analyzer.edges().empty()) {
    return false;
  }

  std::ofstream out(filename);
  out << "# from to x_from y_from x_to y_to gross_flux net_flux "
         "net_flux_fraction q_plus_from q_plus_to reactive_density_from "
         "reactive_density_to\n";
  out << std::setprecision(15);
  for (const auto &edge : tpt_analyzer.edges()) {
    if (edge.from < 0 || edge.to < 0 || edge.from >= tpt_analyzer.n_states() ||
        edge.to >= tpt_analyzer.n_states()) {
      continue;
    }
    const double x_from = msm_analyzer.centroids()(edge.from, 0);
    const double y_from = msm_analyzer.centroids()(edge.from, 1);
    const double x_to = msm_analyzer.centroids()(edge.to, 0);
    const double y_to = msm_analyzer.centroids()(edge.to, 1);
    if (!std::isfinite(x_from) || !std::isfinite(y_from) ||
        !std::isfinite(x_to) || !std::isfinite(y_to)) {
      continue;
    }
    const double fraction =
        (tpt_analyzer.summary().total_reactive_flux > 0.0)
            ? edge.net_flux / tpt_analyzer.summary().total_reactive_flux
            : 0.0;
    out << edge.from << ' ' << edge.to << ' ' << x_from << ' ' << y_from
        << ' ' << x_to << ' ' << y_to << ' ' << edge.gross_flux << ' '
        << edge.net_flux << ' ' << fraction << ' '
        << tpt_analyzer.forward_committor()[edge.from] << ' '
        << tpt_analyzer.forward_committor()[edge.to] << ' '
        << tpt_analyzer.reactive_density()[edge.from] << ' '
        << tpt_analyzer.reactive_density()[edge.to] << '\n';
  }
  return true;
}

bool is_trajectory_boundary(const std::vector<int> &replicas,
                            const std::vector<double> &times_ps, size_t prev,
                            size_t curr) {
  if (replicas.size() > curr && replicas[prev] != replicas[curr])
    return true;
  if (times_ps.size() > curr && times_ps[curr] <= times_ps[prev])
    return true;
  return false;
}

std::vector<std::vector<int>>
build_msm_state_segments(const TitanMSM &msm_analyzer) {
  std::vector<std::vector<int>> segments;
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

bool read_tpt_demux_map(const std::string &path,
                        std::vector<std::vector<int>> &rows_out,
                        std::string &error_message) {
  rows_out.clear();
  std::ifstream in(path);
  if (!in.is_open()) {
    error_message = "cannot open demux map file: " + path;
    return false;
  }

  std::string line;
  std::size_t expected_cols = 0;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const auto hash_pos = line.find('#');
    if (hash_pos != std::string::npos)
      line.erase(hash_pos);
    std::replace(line.begin(), line.end(), ',', ' ');
    line = trim_copy(line);
    if (line.empty())
      continue;

    std::istringstream iss(line);
    std::vector<int> row;
    int value = 0;
    while (iss >> value)
      row.push_back(value);
    if (row.empty()) {
      error_message = "no integer replica ids found on demux-map line " +
                      std::to_string(line_no);
      return false;
    }
    if (expected_cols == 0) {
      expected_cols = row.size();
    } else if (row.size() != expected_cols) {
      error_message =
          "inconsistent demux-map column count at line " +
          std::to_string(line_no) + " (" + std::to_string(row.size()) +
          " vs expected " + std::to_string(expected_cols) + ")";
      return false;
    }
    rows_out.push_back(std::move(row));
  }

  if (rows_out.empty()) {
    error_message = "demux map is empty";
    return false;
  }
  return true;
}

bool build_demux_state_segments(
    const TitanMSM &msm_analyzer, const std::vector<std::vector<int>> &demux_rows,
    std::vector<std::vector<int>> &segments_out, std::string &error_message) {
  segments_out.clear();
  const auto per_replica_dtrajs = msm_analyzer.get_per_replica_dtrajs();
  if (per_replica_dtrajs.empty()) {
    error_message = "no per-replica discrete trajectories are available";
    return false;
  }

  const std::size_t n_cols = demux_rows.front().size();
  for (std::size_t col = 0; col < n_cols; ++col) {
    std::vector<int> current_segment;
    current_segment.reserve(demux_rows.size());
    for (std::size_t row = 0; row < demux_rows.size(); ++row) {
      const int replica_id = demux_rows[row][col];
      if (replica_id < 0) {
        if (!current_segment.empty()) {
          segments_out.push_back(std::move(current_segment));
          current_segment.clear();
          current_segment.reserve(demux_rows.size() - row);
        }
        continue;
      }

      const auto it = per_replica_dtrajs.find(replica_id);
      if (it == per_replica_dtrajs.end()) {
        error_message = "demux map references missing replica id " +
                        std::to_string(replica_id);
        return false;
      }
      if (row >= it->second.size()) {
        error_message =
            "demux map row " + std::to_string(row) + " exceeds available "
            "frames for replica " + std::to_string(replica_id) + " (" +
            std::to_string(it->second.size()) + ")";
        return false;
      }
      current_segment.push_back(it->second[row]);
    }
    if (!current_segment.empty())
      segments_out.push_back(std::move(current_segment));
  }

  segments_out.erase(
      std::remove_if(segments_out.begin(), segments_out.end(),
                     [](const std::vector<int> &segment) {
                       return segment.empty();
                     }),
      segments_out.end());
  if (segments_out.empty()) {
    error_message = "demux map did not produce any non-empty segments";
    return false;
  }
  return true;
}

bool build_segment_count_matrix(
    int n_states, int lag_frames, const std::vector<std::vector<int>> &segments,
    Eigen::MatrixXd &out_counts, Eigen::VectorXd &out_occupancy,
    std::int64_t &out_transition_count, std::string &error_message) {
  out_counts = Eigen::MatrixXd::Zero(n_states, n_states);
  out_occupancy = Eigen::VectorXd::Zero(n_states);
  out_transition_count = 0;
  if (n_states <= 0) {
    error_message = "kinetic demux model requires at least one microstate";
    return false;
  }
  if (lag_frames <= 0) {
    error_message = "kinetic demux model requires lag_frames > 0";
    return false;
  }

  for (const auto &segment : segments) {
    for (int state : segment) {
      if (state >= 0 && state < n_states)
        out_occupancy[state] += 1.0;
    }
    if (segment.size() <= static_cast<std::size_t>(lag_frames))
      continue;
    for (std::size_t i = 0; i + static_cast<std::size_t>(lag_frames) < segment.size();
         ++i) {
      const int state_i = segment[i];
      const int state_j = segment[i + static_cast<std::size_t>(lag_frames)];
      if (state_i < 0 || state_i >= n_states || state_j < 0 || state_j >= n_states)
        continue;
      out_counts(state_i, state_j) += 1.0;
      ++out_transition_count;
    }
  }
  if (out_transition_count <= 0) {
    error_message = "demux segments contain no lag-valid transitions";
    return false;
  }
  return true;
}

bool build_tpt_model_from_segments(
    int n_states, int lag_frames, const std::vector<std::vector<int>> &segments,
    bool symmetrize_counts, Eigen::MatrixXd &out_tpm, Eigen::VectorXd &out_pi,
    std::int64_t &out_transition_count, std::string &error_message) {
  out_tpm = Eigen::MatrixXd::Zero(n_states, n_states);
  out_pi = Eigen::VectorXd::Zero(n_states);

  Eigen::MatrixXd counts;
  Eigen::VectorXd occupancy;
  if (!build_segment_count_matrix(n_states, lag_frames, segments, counts,
                                  occupancy, out_transition_count,
                                  error_message)) {
    return false;
  }

  if (symmetrize_counts) {
    counts = 0.5 * (counts + counts.transpose());
  }

  for (int i = 0; i < n_states; ++i) {
    const double row_sum = counts.row(i).sum();
    if (row_sum <= 1e-12) {
      out_tpm.row(i).setZero();
      out_tpm(i, i) = 1.0;
    } else {
      out_tpm.row(i) = counts.row(i) / row_sum;
    }
  }

  Eigen::RowVectorXd pi(n_states);
  if (occupancy.sum() > 0.0) {
    pi = (occupancy / occupancy.sum()).transpose();
  } else {
    pi = Eigen::RowVectorXd::Constant(n_states, 1.0 / n_states);
  }
  Eigen::RowVectorXd pi_new(n_states);
  for (int it = 0; it < 1000; ++it) {
    pi_new.noalias() = pi * out_tpm;
    const double delta = (pi_new - pi).cwiseAbs().maxCoeff();
    pi = pi_new;
    if (delta < 1e-10)
      break;
  }
  const double pi_sum = pi.sum();
  if (pi_sum > 1e-30) {
    pi /= pi_sum;
  } else {
    pi = Eigen::RowVectorXd::Constant(n_states, 1.0 / n_states);
  }
  out_pi = pi.transpose();
  return true;
}

[[maybe_unused]] TptKineticDiagnostics compute_tpt_kinetic_diagnostics_from_segments(
    const std::vector<std::vector<int>> &segments, int n_states,
    int lag_frames, double lag_time_ns, const TitanTPT &tpt_analyzer) {
  TptKineticDiagnostics diag;
  if (segments.empty())
    return diag;

  std::vector<unsigned char> basin_map(static_cast<size_t>(n_states),
                                       static_cast<unsigned char>(
                                           TptBasinLabel::Intermediate));
  for (int state : tpt_analyzer.source_states()) {
    if (state >= 0 && state < n_states) {
      basin_map[static_cast<size_t>(state)] =
          static_cast<unsigned char>(TptBasinLabel::Source);
    }
  }
  for (int state : tpt_analyzer.sink_states()) {
    if (state >= 0 && state < n_states) {
      basin_map[static_cast<size_t>(state)] =
          static_cast<unsigned char>(TptBasinLabel::Sink);
    }
  }

  diag.n_segments = static_cast<int>(segments.size());
  if (!segments.empty()) {
    diag.min_segment_length_frames = static_cast<int>(segments.front().size());
  }

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
      const int label = (state >= 0 && state < n_states)
                            ? static_cast<int>(
                                  basin_map[static_cast<size_t>(state)])
                            : static_cast<int>(TptBasinLabel::Intermediate);

      if (label == static_cast<int>(TptBasinLabel::Source)) {
        ++diag.frames_source;
        seen_source = true;
      } else if (label == static_cast<int>(TptBasinLabel::Sink)) {
        ++diag.frames_sink;
        seen_sink = true;
      } else {
        ++diag.frames_intermediate;
      }

      if ((label == static_cast<int>(TptBasinLabel::Source) ||
           label == static_cast<int>(TptBasinLabel::Sink)) &&
          label != previous_label) {
        if (label == static_cast<int>(TptBasinLabel::Source) &&
            last_committed == static_cast<int>(TptBasinLabel::Sink)) {
          ++diag.observed_b_to_a_events;
        } else if (label == static_cast<int>(TptBasinLabel::Sink) &&
                   last_committed == static_cast<int>(TptBasinLabel::Source)) {
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
  diag.model_mfpt_ab_lags =
      tpt_analyzer.summary().mfpt_source_to_sink_lags;
  if (diag.lag_time_ns > 0.0) {
    diag.model_rate_ab_per_ns =
        diag.model_rate_ab_per_lag / diag.lag_time_ns;
    if (std::isfinite(diag.model_mfpt_ab_lags)) {
      diag.model_mfpt_ab_ns = diag.model_mfpt_ab_lags * diag.lag_time_ns;
    }
  }

  Eigen::MatrixXd counts_lag;
  Eigen::VectorXd occ_lag;
  std::int64_t transitions_lag = 0;
  std::string counts_error;
  if (build_segment_count_matrix(n_states, lag_frames, segments, counts_lag,
                                 occ_lag, transitions_lag, counts_error)) {
    Eigen::MatrixXd counts_2lag;
    Eigen::VectorXd occ_2lag;
    std::int64_t transitions_2lag = 0;
    if (build_segment_count_matrix(n_states, 2 * lag_frames, segments,
                                   counts_2lag, occ_2lag, transitions_2lag,
                                   counts_error)) {
      Eigen::MatrixXd p1 = Eigen::MatrixXd::Zero(n_states, n_states);
      Eigen::MatrixXd p2 = Eigen::MatrixXd::Zero(n_states, n_states);
      std::vector<double> row_weights;
      row_weights.reserve(static_cast<std::size_t>(n_states));
      double sum_abs = 0.0;
      double max_abs = 0.0;
      double weighted_sq = 0.0;
      double total_weight = 0.0;
      for (int i = 0; i < n_states; ++i) {
        const double row_sum1 = counts_lag.row(i).sum();
        const double row_sum2 = counts_2lag.row(i).sum();
        if (row_sum1 > 1e-12)
          p1.row(i) = counts_lag.row(i) / row_sum1;
        if (row_sum2 > 1e-12)
          p2.row(i) = counts_2lag.row(i) / row_sum2;
      }
      const Eigen::MatrixXd p2_pred = p1 * p1;
      for (int i = 0; i < n_states; ++i) {
        const double row_sum2 = counts_2lag.row(i).sum();
        if (row_sum2 <= 1e-12)
          continue;
        const Eigen::RowVectorXd diff = p2.row(i) - p2_pred.row(i);
        const double row_mae =
            diff.cwiseAbs().sum() / static_cast<double>(n_states);
        const double row_max = diff.cwiseAbs().maxCoeff();
        const double row_weight = row_sum2;
        sum_abs += row_mae * row_weight;
        max_abs = std::max(max_abs, row_max);
        weighted_sq += row_weight * diff.squaredNorm();
        total_weight += row_weight;
        ++diag.ck_rows_compared;
      }
      if (diag.ck_rows_compared > 0 && total_weight > 0.0) {
        diag.ck_mean_abs_error = sum_abs / total_weight;
        diag.ck_max_abs_error = max_abs;
        diag.ck_weighted_frobenius = std::sqrt(weighted_sq / total_weight);
        if (diag.ck_weighted_frobenius > 0.25 ||
            diag.ck_max_abs_error > 0.50) {
          diag.markov_sanity = "fail";
        } else if (diag.ck_weighted_frobenius > 0.10 ||
                   diag.ck_max_abs_error > 0.25) {
          diag.markov_sanity = "weak";
        } else {
          diag.markov_sanity = "pass";
        }
      }
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

[[maybe_unused]] bool run_tpt_bootstrap(
    const std::vector<std::vector<int>> &base_segments, int n_states,
    int lag_frames, double lag_time_ns, bool symmetrize_counts,
    const std::vector<int> &source_states, const std::vector<int> &sink_states,
    int top_paths, double flux_cut, int n_threads, int n_samples,
    unsigned int seed, TitanExecutor &executor, TptBootstrapAggregate &out_bootstrap,
    std::string &error_message) {
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
      std::vector<std::vector<int>> sampled_segments;
      sampled_segments.reserve(base_segments.size());

      for (int sample_idx = begin; sample_idx < end; ++sample_idx) {
        sampled_segments.clear();
        for (std::size_t seg = 0; seg < base_segments.size(); ++seg) {
          sampled_segments.push_back(
              base_segments[static_cast<std::size_t>(seg_pick(rng))]);
        }

        Eigen::MatrixXd sample_tpm;
        Eigen::VectorXd sample_pi;
        std::int64_t sample_transitions = 0;
        std::string sample_error;
        if (!build_tpt_model_from_segments(
                n_states, lag_frames, sampled_segments, symmetrize_counts,
                sample_tpm, sample_pi, sample_transitions, sample_error)) {
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
          out_bootstrap.source_population.end(),
          local.source_population.begin(), local.source_population.end());
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

void write_tpt_bootstrap_files(const std::string &summary_filename,
                               const std::string &states_filename,
                               const TptBootstrapAggregate &bootstrap,
                               const TitanTPT &reference_tpt,
                               const std::string &model_source) {
  auto mean_std = [](const std::vector<double> &values) {
    std::pair<double, double> out{std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN()};
    if (values.empty())
      return out;
    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double var = 0.0;
    for (double v : values)
      var += (v - mean) * (v - mean);
    var /= std::max<std::size_t>(1, values.size() - 1);
    out.first = mean;
    out.second = std::sqrt(std::max(0.0, var));
    return out;
  };

  std::ofstream out(summary_filename);
  out << "# TITAN TPT bootstrap summary\n";
  out << "model_source " << model_source << "\n";
  out << "requested_samples " << bootstrap.requested_samples << "\n";
  out << "successful_samples " << bootstrap.successful_samples << "\n";
  out << "# metric mean std p2.5 p97.5\n";

  const auto flux_stats = mean_std(bootstrap.total_reactive_flux);
  out << "total_reactive_flux " << flux_stats.first << ' ' << flux_stats.second
      << ' ' << percentile_sorted(bootstrap.total_reactive_flux, 0.025) << ' '
      << percentile_sorted(bootstrap.total_reactive_flux, 0.975) << "\n";
  const auto rate_stats = mean_std(bootstrap.rate_source_to_sink_per_lag);
  out << "rate_source_to_sink_per_lag " << rate_stats.first << ' '
      << rate_stats.second << ' '
      << percentile_sorted(bootstrap.rate_source_to_sink_per_lag, 0.025) << ' '
      << percentile_sorted(bootstrap.rate_source_to_sink_per_lag, 0.975) << "\n";
  const auto mfpt_stats = mean_std(bootstrap.mfpt_source_to_sink_lags);
  out << "mfpt_source_to_sink_lags " << mfpt_stats.first << ' '
      << mfpt_stats.second << ' '
      << percentile_sorted(bootstrap.mfpt_source_to_sink_lags, 0.025) << ' '
      << percentile_sorted(bootstrap.mfpt_source_to_sink_lags, 0.975) << "\n";
  const auto src_stats = mean_std(bootstrap.source_population);
  out << "source_population " << src_stats.first << ' ' << src_stats.second
      << ' ' << percentile_sorted(bootstrap.source_population, 0.025) << ' '
      << percentile_sorted(bootstrap.source_population, 0.975) << "\n";
  const auto sink_stats = mean_std(bootstrap.sink_population);
  out << "sink_population " << sink_stats.first << ' ' << sink_stats.second
      << ' ' << percentile_sorted(bootstrap.sink_population, 0.025) << ' '
      << percentile_sorted(bootstrap.sink_population, 0.975) << "\n";

  std::ofstream states_out(states_filename);
  states_out << "# state q_plus_ref q_plus_mean q_plus_std reactive_density_ref "
                "reactive_density_mean reactive_density_std\n";
  const double n =
      static_cast<double>(std::max(1, bootstrap.successful_samples));
  for (int state = 0; state < reference_tpt.n_states(); ++state) {
    const double q_mean = bootstrap.q_forward_sum[state] / n;
    const double q_var =
        std::max(0.0, bootstrap.q_forward_sq_sum[state] / n - q_mean * q_mean);
    const double rho_mean = bootstrap.reactive_density_sum[state] / n;
    const double rho_var = std::max(
        0.0, bootstrap.reactive_density_sq_sum[state] / n - rho_mean * rho_mean);
    states_out << state << ' ' << reference_tpt.forward_committor()[state] << ' '
               << q_mean << ' ' << std::sqrt(q_var) << ' '
               << reference_tpt.reactive_density()[state] << ' ' << rho_mean
               << ' ' << std::sqrt(rho_var) << "\n";
  }
}

void write_tpt_kinetics_file(const std::string &filename,
                             const TptKineticDiagnostics &diag,
                             const std::string &model_source) {
  std::ofstream out(filename);
  out << "# TITAN TPT kinetic diagnostics\n";
  out << "model_source " << model_source << "\n";
  out << "n_segments " << diag.n_segments << "\n";
  out << "segments_with_source " << diag.segments_with_source << "\n";
  out << "segments_with_sink " << diag.segments_with_sink << "\n";
  out << "segments_with_both " << diag.segments_with_both << "\n";
  out << "min_segment_length_frames " << diag.min_segment_length_frames
      << "\n";
  out << "max_segment_length_frames " << diag.max_segment_length_frames
      << "\n";
  out << std::setprecision(15);
  out << "mean_segment_length_frames " << diag.mean_segment_length_frames
      << "\n";
  out << "total_frames " << diag.total_frames << "\n";
  out << "frames_source " << diag.frames_source << "\n";
  out << "frames_sink " << diag.frames_sink << "\n";
  out << "frames_intermediate " << diag.frames_intermediate << "\n";
  out << "observed_source_fraction " << diag.observed_source_fraction << "\n";
  out << "observed_sink_fraction " << diag.observed_sink_fraction << "\n";
  out << "observed_intermediate_fraction "
      << diag.observed_intermediate_fraction << "\n";
  out << "observed_a_to_b_events " << diag.observed_a_to_b_events << "\n";
  out << "observed_b_to_a_events " << diag.observed_b_to_a_events << "\n";
  out << "observed_round_trips " << diag.observed_round_trips << "\n";
  out << "lag_time_ns " << diag.lag_time_ns << "\n";
  out << "model_rate_ab_per_lag " << diag.model_rate_ab_per_lag << "\n";
  out << "model_rate_ab_per_ns " << diag.model_rate_ab_per_ns << "\n";
  out << "model_mfpt_ab_lags " << diag.model_mfpt_ab_lags << "\n";
  out << "model_mfpt_ab_ns " << diag.model_mfpt_ab_ns << "\n";
  out << "ck_rows_compared " << diag.ck_rows_compared << "\n";
  out << "ck_mean_abs_error " << diag.ck_mean_abs_error << "\n";
  out << "ck_max_abs_error " << diag.ck_max_abs_error << "\n";
  out << "ck_weighted_frobenius " << diag.ck_weighted_frobenius << "\n";
  out << "markov_sanity " << diag.markov_sanity << "\n";
  out << "suitability " << diag.suitability << "\n";
}

std::string join_int_list(const std::vector<int> &values) {
  if (values.empty())
    return "-";
  std::ostringstream oss;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0)
      oss << ',';
    oss << values[i];
  }
  return oss.str();
}

double path_edge_similarity(const std::vector<int> &lhs,
                            const std::vector<int> &rhs) {
  if (lhs.size() < 2 || rhs.size() < 2)
    return 0.0;
  int intersection = 0;
  int union_count = 0;
  std::vector<std::pair<int, int>> lhs_edges;
  std::vector<std::pair<int, int>> rhs_edges;
  lhs_edges.reserve(lhs.size() - 1);
  rhs_edges.reserve(rhs.size() - 1);
  for (std::size_t i = 0; i + 1 < lhs.size(); ++i)
    lhs_edges.emplace_back(lhs[i], lhs[i + 1]);
  for (std::size_t i = 0; i + 1 < rhs.size(); ++i)
    rhs_edges.emplace_back(rhs[i], rhs[i + 1]);
  union_count = static_cast<int>(lhs_edges.size());
  for (const auto &edge_rhs : rhs_edges) {
    bool found = false;
    for (const auto &edge_lhs : lhs_edges) {
      if (edge_lhs == edge_rhs) {
        found = true;
        break;
      }
    }
    if (found) {
      ++intersection;
    } else {
      ++union_count;
    }
  }
  if (union_count <= 0)
    return 0.0;
  return static_cast<double>(intersection) / union_count;
}

std::vector<TptPathCluster>
cluster_tpt_paths(const TitanTPT &tpt_analyzer, double similarity_cut = 0.6) {
  std::vector<TptPathCluster> clusters;
  const auto &paths = tpt_analyzer.paths();
  for (std::size_t rank = 0; rank < paths.size(); ++rank) {
    const auto &path = paths[rank];
    int best_cluster = -1;
    double best_similarity = 0.0;
    for (std::size_t c = 0; c < clusters.size(); ++c) {
      const double sim = path_edge_similarity(path.nodes,
                                              clusters[c].representative_nodes);
      if (sim >= similarity_cut && sim > best_similarity) {
        best_similarity = sim;
        best_cluster = static_cast<int>(c);
      }
    }
    if (best_cluster < 0) {
      TptPathCluster cluster;
      cluster.representative_rank = static_cast<int>(rank + 1);
      cluster.total_flux = path.flux;
      cluster.total_fraction = path.fraction;
      cluster.member_ranks.push_back(static_cast<int>(rank + 1));
      cluster.representative_nodes = path.nodes;
      clusters.push_back(std::move(cluster));
    } else {
      auto &cluster = clusters[static_cast<std::size_t>(best_cluster)];
      cluster.total_flux += path.flux;
      cluster.total_fraction += path.fraction;
      cluster.member_ranks.push_back(static_cast<int>(rank + 1));
    }
  }
  std::sort(clusters.begin(), clusters.end(),
            [](const TptPathCluster &lhs, const TptPathCluster &rhs) {
              if (lhs.total_flux != rhs.total_flux)
                return lhs.total_flux > rhs.total_flux;
              return lhs.representative_rank < rhs.representative_rank;
            });
  return clusters;
}

std::size_t write_tpt_path_clusters_file(const std::string &filename,
                                         const TitanTPT &tpt_analyzer) {
  const auto clusters = cluster_tpt_paths(tpt_analyzer);
  std::ofstream out(filename);
  out << "# cluster_id n_paths total_flux total_fraction representative_rank "
         "representative_path member_ranks\n";
  out << std::setprecision(15);
  for (std::size_t i = 0; i < clusters.size(); ++i) {
    const auto &cluster = clusters[i];
    out << (i + 1) << ' ' << cluster.member_ranks.size() << ' '
        << cluster.total_flux << ' ' << cluster.total_fraction << ' '
        << cluster.representative_rank << ' '
        << TitanTPT::format_state_list(cluster.representative_nodes) << ' '
        << join_int_list(cluster.member_ranks) << '\n';
  }
  return clusters.size();
}

TptPathSupport accumulate_tpt_path_support(const TitanTPT &tpt_analyzer,
                                           TitanExecutor &executor) {
  const auto &paths = tpt_analyzer.paths();
  const auto &edges = tpt_analyzer.edges();
  const std::size_t n_edges = edges.size();
  const std::size_t n_states =
      static_cast<std::size_t>(std::max(0, tpt_analyzer.n_states()));
  TptPathSupport total(n_edges, n_states);
  if (paths.empty() || n_edges == 0 || n_states == 0)
    return total;

  std::unordered_map<std::uint64_t, std::size_t> edge_index;
  edge_index.reserve(n_edges * 2 + 1);
  for (std::size_t i = 0; i < n_edges; ++i) {
    edge_index.emplace(pack_directed_edge_key(edges[i].from, edges[i].to), i);
  }

  std::size_t total_work = 0;
  for (const auto &path : paths) {
    if (path.nodes.size() >= 2)
      total_work += path.nodes.size() - 1;
  }

  const std::size_t worker_count =
      (executor.capacity() > 1 && paths.size() >= 8 && total_work >= 256)
          ? std::min<std::size_t>(executor.capacity(), paths.size())
          : 1;

  auto accumulate_range = [&](std::size_t begin, std::size_t end,
                              TptPathSupport &accum) {
    for (std::size_t rank = begin; rank < end; ++rank) {
      const auto &path = paths[rank];
      if (path.nodes.size() < 2)
        continue;
      for (std::size_t i = 0; i + 1 < path.nodes.size(); ++i) {
        const auto key =
            pack_directed_edge_key(path.nodes[i], path.nodes[i + 1]);
        const auto edge_it = edge_index.find(key);
        if (edge_it != edge_index.end()) {
          const std::size_t edge_idx = edge_it->second;
          accum.edge_flux[edge_idx] += path.flux;
          accum.edge_fraction[edge_idx] += path.fraction;
          accum.edge_count[edge_idx] += 1;
        }
      }
      for (std::size_t i = 1; i + 1 < path.nodes.size(); ++i) {
        const int state = path.nodes[i];
        if (state < 0 || static_cast<std::size_t>(state) >= n_states)
          continue;
        const std::size_t state_idx = static_cast<std::size_t>(state);
        accum.state_flux[state_idx] += path.flux;
        accum.state_fraction[state_idx] += path.fraction;
        accum.state_count[state_idx] += 1;
      }
    }
  };

  if (worker_count == 1) {
    accumulate_range(0, paths.size(), total);
    return total;
  }

  std::vector<TptPathSupport> locals;
  locals.reserve(worker_count);
  for (std::size_t i = 0; i < worker_count; ++i)
    locals.emplace_back(n_edges, n_states);

  const std::size_t chunk = (paths.size() + worker_count - 1) / worker_count;
  executor.parallel_do(worker_count, [&](size_t worker) {
    const std::size_t begin = worker * chunk;
    const std::size_t end = std::min(paths.size(), begin + chunk);
    if (begin >= end)
      return;
    accumulate_range(begin, end, locals[worker]);
  });

  for (const auto &local : locals) {
    for (std::size_t i = 0; i < n_edges; ++i) {
      total.edge_flux[i] += local.edge_flux[i];
      total.edge_fraction[i] += local.edge_fraction[i];
      total.edge_count[i] += local.edge_count[i];
    }
    for (std::size_t i = 0; i < n_states; ++i) {
      total.state_flux[i] += local.state_flux[i];
      total.state_fraction[i] += local.state_fraction[i];
      total.state_count[i] += local.state_count[i];
    }
  }
  return total;
}

std::vector<TptBottleneckEdgeRow>
compute_tpt_bottleneck_edges(const TitanTPT &tpt_analyzer,
                             const std::vector<int> &macro_assignments,
                             const TptPathSupport &support) {
  std::vector<TptBottleneckEdgeRow> rows;
  rows.reserve(tpt_analyzer.edges().size());
  const double total_flux =
      std::max(0.0, tpt_analyzer.summary().total_reactive_flux);
  const bool have_macros =
      macro_assignments.size() == static_cast<std::size_t>(tpt_analyzer.n_states());
  for (std::size_t idx = 0; idx < tpt_analyzer.edges().size(); ++idx) {
    const auto &edge = tpt_analyzer.edges()[idx];
    TptBottleneckEdgeRow row;
    row.from = edge.from;
    row.to = edge.to;
    row.from_macro =
        (have_macros && edge.from >= 0 &&
         static_cast<std::size_t>(edge.from) < macro_assignments.size())
            ? macro_assignments[static_cast<std::size_t>(edge.from)]
            : -1;
    row.to_macro =
        (have_macros && edge.to >= 0 &&
         static_cast<std::size_t>(edge.to) < macro_assignments.size())
            ? macro_assignments[static_cast<std::size_t>(edge.to)]
            : -1;
    row.gross_flux = edge.gross_flux;
    row.net_flux = edge.net_flux;
    row.relative_net_flux =
        (total_flux > 0.0) ? edge.net_flux / total_flux : 0.0;
    row.from_q = tpt_analyzer.forward_committor()[edge.from];
    row.to_q = tpt_analyzer.forward_committor()[edge.to];
    row.midpoint_q = 0.5 * (row.from_q + row.to_q);
    row.transition_proximity = tpt_transition_proximity(row.midpoint_q);
    row.bottleneck_score = row.net_flux * row.transition_proximity;
    if (idx < support.edge_flux.size()) {
      row.dominant_path_support_flux = support.edge_flux[idx];
      row.dominant_path_support_fraction = support.edge_fraction[idx];
      row.dominant_path_count = support.edge_count[idx];
    }
    rows.push_back(row);
  }
  std::sort(rows.begin(), rows.end(),
            [](const TptBottleneckEdgeRow &lhs,
               const TptBottleneckEdgeRow &rhs) {
              if (lhs.bottleneck_score != rhs.bottleneck_score)
                return lhs.bottleneck_score > rhs.bottleneck_score;
              if (lhs.dominant_path_support_flux !=
                  rhs.dominant_path_support_flux)
                return lhs.dominant_path_support_flux >
                       rhs.dominant_path_support_flux;
              if (lhs.net_flux != rhs.net_flux)
                return lhs.net_flux > rhs.net_flux;
              if (lhs.from != rhs.from)
                return lhs.from < rhs.from;
              return lhs.to < rhs.to;
            });
  return rows;
}

std::vector<TptBottleneckStateRow>
compute_tpt_bottleneck_states(const TitanTPT &tpt_analyzer,
                              const std::vector<int> &macro_assignments,
                              const TptPathSupport &support) {
  std::vector<unsigned char> is_source(
      static_cast<std::size_t>(std::max(0, tpt_analyzer.n_states())), 0);
  std::vector<unsigned char> is_sink(
      static_cast<std::size_t>(std::max(0, tpt_analyzer.n_states())), 0);
  for (int state : tpt_analyzer.source_states()) {
    if (state >= 0 && static_cast<std::size_t>(state) < is_source.size())
      is_source[static_cast<std::size_t>(state)] = 1;
  }
  for (int state : tpt_analyzer.sink_states()) {
    if (state >= 0 && static_cast<std::size_t>(state) < is_sink.size())
      is_sink[static_cast<std::size_t>(state)] = 1;
  }

  std::vector<TptBottleneckStateRow> rows;
  rows.reserve(static_cast<std::size_t>(tpt_analyzer.n_states()));
  const bool have_macros =
      macro_assignments.size() == static_cast<std::size_t>(tpt_analyzer.n_states());
  for (int state = 0; state < tpt_analyzer.n_states(); ++state) {
    TptBottleneckStateRow row;
    row.state = state;
    if (is_source[static_cast<std::size_t>(state)])
      row.label = 'A';
    else if (is_sink[static_cast<std::size_t>(state)])
      row.label = 'B';
    row.macro =
        have_macros ? macro_assignments[static_cast<std::size_t>(state)] : -1;
    row.pi = tpt_analyzer.stationary_dist()[state];
    row.q_plus = tpt_analyzer.forward_committor()[state];
    row.q_minus = tpt_analyzer.backward_committor()[state];
    row.reactive_density = tpt_analyzer.reactive_density()[state];
    row.net_in_flux = tpt_analyzer.net_in_flux()[state];
    row.net_out_flux = tpt_analyzer.net_out_flux()[state];
    row.through_flux = std::min(row.net_in_flux, row.net_out_flux);
    row.transition_proximity = tpt_transition_proximity(row.q_plus);
    row.bottleneck_score = row.through_flux * row.transition_proximity;
    if (static_cast<std::size_t>(state) < support.state_flux.size()) {
      row.dominant_path_support_flux =
          support.state_flux[static_cast<std::size_t>(state)];
      row.dominant_path_support_fraction =
          support.state_fraction[static_cast<std::size_t>(state)];
      row.dominant_path_count =
          support.state_count[static_cast<std::size_t>(state)];
    }
    rows.push_back(row);
  }
  std::sort(rows.begin(), rows.end(),
            [](const TptBottleneckStateRow &lhs,
               const TptBottleneckStateRow &rhs) {
              if (lhs.bottleneck_score != rhs.bottleneck_score)
                return lhs.bottleneck_score > rhs.bottleneck_score;
              if (lhs.dominant_path_support_flux !=
                  rhs.dominant_path_support_flux)
                return lhs.dominant_path_support_flux >
                       rhs.dominant_path_support_flux;
              if (lhs.through_flux != rhs.through_flux)
                return lhs.through_flux > rhs.through_flux;
              return lhs.state < rhs.state;
            });
  return rows;
}

std::size_t write_tpt_bottleneck_edges_file(
    const std::string &filename,
    const std::vector<TptBottleneckEdgeRow> &rows) {
  std::ofstream out(filename);
  out << "# rank from to from_macro to_macro gross_flux net_flux "
         "relative_net_flux from_q to_q midpoint_q transition_proximity "
         "bottleneck_score dominant_path_support_flux "
         "dominant_path_support_fraction dominant_path_count\n";
  out << std::setprecision(15);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto &row = rows[i];
    out << (i + 1) << ' ' << row.from << ' ' << row.to << ' '
        << row.from_macro << ' ' << row.to_macro << ' ' << row.gross_flux
        << ' ' << row.net_flux << ' ' << row.relative_net_flux << ' '
        << row.from_q << ' ' << row.to_q << ' ' << row.midpoint_q << ' '
        << row.transition_proximity << ' ' << row.bottleneck_score << ' '
        << row.dominant_path_support_flux << ' '
        << row.dominant_path_support_fraction << ' '
        << row.dominant_path_count << '\n';
  }
  return rows.size();
}

std::size_t write_tpt_bottleneck_states_file(
    const std::string &filename,
    const std::vector<TptBottleneckStateRow> &rows) {
  std::ofstream out(filename);
  out << "# rank state set macrostate pi q_plus q_minus reactive_density "
         "net_in_flux net_out_flux through_flux transition_proximity "
         "bottleneck_score dominant_path_support_flux "
         "dominant_path_support_fraction dominant_path_count\n";
  out << std::setprecision(15);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto &row = rows[i];
    out << (i + 1) << ' ' << row.state << ' ' << row.label << ' '
        << row.macro << ' ' << row.pi << ' ' << row.q_plus << ' '
        << row.q_minus << ' ' << row.reactive_density << ' '
        << row.net_in_flux << ' ' << row.net_out_flux << ' '
        << row.through_flux << ' ' << row.transition_proximity << ' '
        << row.bottleneck_score << ' ' << row.dominant_path_support_flux
        << ' ' << row.dominant_path_support_fraction << ' '
        << row.dominant_path_count << '\n';
  }
  return rows.size();
}

void append_tpt_bottleneck_summary(
    const std::string &summary_filename,
    const std::vector<TptBottleneckEdgeRow> &edge_rows,
    const std::vector<TptBottleneckStateRow> &state_rows,
    std::size_t top_n = 5) {
  std::ofstream out(summary_filename, std::ios::app);
  if (!out)
    return;

  out << "\n# bottleneck summary\n";
  out << "bottleneck_top_n " << top_n << "\n";

  out << "# bottleneck_edges_top\n";
  out << "# rank from to score net_flux midpoint_q dominant_path_support_flux "
         "dominant_path_count\n";
  std::size_t emitted_edges = 0;
  for (const auto &row : edge_rows) {
    if (!(row.bottleneck_score > 0.0 || row.dominant_path_support_flux > 0.0 ||
          row.net_flux > 0.0)) {
      continue;
    }
    out << "edge_rank " << (emitted_edges + 1) << ' ' << row.from << ' '
        << row.to << ' ' << row.bottleneck_score << ' ' << row.net_flux << ' '
        << row.midpoint_q << ' ' << row.dominant_path_support_flux << ' '
        << row.dominant_path_count << '\n';
    ++emitted_edges;
    if (emitted_edges >= top_n)
      break;
  }
  if (emitted_edges == 0)
    out << "edge_rank none\n";

  out << "# bottleneck_states_top\n";
  out << "# rank state set macrostate score through_flux q_plus "
         "reactive_density dominant_path_support_flux dominant_path_count\n";
  std::size_t emitted_states = 0;
  for (const auto &row : state_rows) {
    if (!(row.bottleneck_score > 0.0 || row.dominant_path_support_flux > 0.0 ||
          row.through_flux > 0.0)) {
      continue;
    }
    out << "state_rank " << (emitted_states + 1) << ' ' << row.state << ' '
        << row.label << ' ' << row.macro << ' ' << row.bottleneck_score << ' '
        << row.through_flux << ' ' << row.q_plus << ' ' << row.reactive_density
        << ' ' << row.dominant_path_support_flux << ' '
        << row.dominant_path_count << '\n';
    ++emitted_states;
    if (emitted_states >= top_n)
      break;
  }
  if (emitted_states == 0)
    out << "state_rank none\n";
}

bool build_macro_tpt_model(const Eigen::MatrixXd &micro_tpm,
                           const Eigen::VectorXd &micro_pi,
                           const std::vector<int> &macro_assignments,
                           Eigen::MatrixXd &macro_tpm,
                           Eigen::VectorXd &macro_pi,
                           std::string &error_message) {
  if (micro_tpm.rows() == 0 || micro_tpm.rows() != micro_tpm.cols()) {
    error_message = "microstate TPM is unavailable";
    return false;
  }
  if (micro_pi.size() != micro_tpm.rows()) {
    error_message = "microstate stationary distribution does not match TPM";
    return false;
  }
  if (macro_assignments.size() != static_cast<std::size_t>(micro_tpm.rows())) {
    error_message = "macrostate assignments do not match microstate count";
    return false;
  }
  const int n_macro =
      *std::max_element(macro_assignments.begin(), macro_assignments.end()) + 1;
  if (n_macro < 2) {
    error_message = "need at least two macrostates for all-pairs kinetics";
    return false;
  }

  macro_tpm = Eigen::MatrixXd::Zero(n_macro, n_macro);
  macro_pi = Eigen::VectorXd::Zero(n_macro);
  for (int i = 0; i < micro_pi.size(); ++i) {
    const int macro_i = macro_assignments[static_cast<std::size_t>(i)];
    if (macro_i >= 0 && macro_i < n_macro &&
        std::isfinite(micro_pi[i]) && micro_pi[i] > 0.0) {
      macro_pi[macro_i] += micro_pi[i];
    }
  }

  Eigen::MatrixXd macro_flow = Eigen::MatrixXd::Zero(n_macro, n_macro);
  for (int i = 0; i < micro_tpm.rows(); ++i) {
    const int macro_i = macro_assignments[static_cast<std::size_t>(i)];
    if (macro_i < 0 || macro_i >= n_macro)
      continue;
    const double pi_i =
        (std::isfinite(micro_pi[i]) && micro_pi[i] > 0.0) ? micro_pi[i] : 0.0;
    for (int j = 0; j < micro_tpm.cols(); ++j) {
      const int macro_j = macro_assignments[static_cast<std::size_t>(j)];
      if (macro_j < 0 || macro_j >= n_macro)
        continue;
      const double p_ij = micro_tpm(i, j);
      if (std::isfinite(p_ij) && p_ij > 0.0)
        macro_flow(macro_i, macro_j) += pi_i * p_ij;
    }
  }

  for (int macro = 0; macro < n_macro; ++macro) {
    if (macro_pi[macro] <= 1e-12) {
      macro_tpm.row(macro).setZero();
      macro_tpm(macro, macro) = 1.0;
    } else {
      macro_tpm.row(macro) = macro_flow.row(macro) / macro_pi[macro];
    }
  }
  const double pi_sum = macro_pi.sum();
  if (pi_sum <= 1e-12) {
    error_message = "macrostate populations collapsed to zero";
    return false;
  }
  macro_pi /= pi_sum;
  return true;
}

std::size_t write_tpt_all_pairs_kinetics_file(const std::string &filename,
                                              const Eigen::MatrixXd &macro_tpm,
                                              const Eigen::VectorXd &macro_pi,
                                              double lag_time_ns) {
  std::ofstream out(filename);
  out << "# source_macro sink_macro source_population sink_population "
         "total_reactive_flux rate_per_lag rate_per_ns mfpt_lags mfpt_ns "
         "n_positive_flux_edges n_paths\n";
  out << std::setprecision(15);
  std::size_t n_pairs = 0;
  for (int source = 0; source < macro_tpm.rows(); ++source) {
    for (int sink = 0; sink < macro_tpm.rows(); ++sink) {
      if (source == sink)
        continue;
      TitanTPT macro_tpt(macro_tpm, macro_pi);
      macro_tpt.compute({source}, {sink}, 3, 0.0, nullptr, 1);
      const auto &summary = macro_tpt.summary();
      const double rate_ns =
          (lag_time_ns > 0.0) ? summary.rate_source_to_sink_per_lag / lag_time_ns
                              : std::numeric_limits<double>::quiet_NaN();
      const double mfpt_ns =
          (lag_time_ns > 0.0 && std::isfinite(summary.mfpt_source_to_sink_lags))
              ? summary.mfpt_source_to_sink_lags * lag_time_ns
              : std::numeric_limits<double>::quiet_NaN();
      out << source << ' ' << sink << ' ' << summary.source_population << ' '
          << summary.sink_population << ' ' << summary.total_reactive_flux
          << ' ' << summary.rate_source_to_sink_per_lag << ' ' << rate_ns
          << ' ' << summary.mfpt_source_to_sink_lags << ' ' << mfpt_ns << ' '
          << macro_tpt.edges().size() << ' ' << macro_tpt.paths().size() << '\n';
      ++n_pairs;
    }
  }
  return n_pairs;
}

bool detect_irregular_energy_dt(
    const std::vector<std::pair<double, double>> &xvg_data, double inferred_dt_ps) {
  if (xvg_data.size() < 4 || !std::isfinite(inferred_dt_ps) || inferred_dt_ps <= 0.0)
    return false;
  int positive_deltas = 0;
  int irregular = 0;
  const double tol = std::max(1e-6, 0.25 * inferred_dt_ps);
  for (std::size_t i = 1; i < xvg_data.size(); ++i) {
    const double dt = xvg_data[i].first - xvg_data[i - 1].first;
    if (!(dt > 0.0))
      continue;
    ++positive_deltas;
    if (std::abs(dt - inferred_dt_ps) > tol)
      ++irregular;
  }
  return positive_deltas > 0 &&
         static_cast<double>(irregular) / positive_deltas > 0.20;
}

void write_dham_window_audit_file(const std::string &filename,
                                  const std::vector<DhamWindowAuditRow> &rows) {
  std::ofstream out(filename);
  out << "# replica status traj_present energy_present overlap_found "
         "truncated_start truncated_end irregular_energy_dt sparse_energy "
         "traj_frames energy_points kept_frames dropped_outside unmatched_in_overlap "
         "traj_min_ps traj_max_ps energy_min_ps energy_max_ps overlap_start_ps "
         "overlap_end_ps inferred_dt_energy_ps\n";
  out << std::setprecision(15);
  for (const auto &row : rows) {
    out << row.replica_id << ' ' << row.status << ' '
        << (row.traj_present ? 1 : 0) << ' ' << (row.energy_present ? 1 : 0)
        << ' ' << (row.overlap_found ? 1 : 0) << ' '
        << (row.truncated_start ? 1 : 0) << ' '
        << (row.truncated_end ? 1 : 0) << ' '
        << (row.irregular_energy_dt ? 1 : 0) << ' '
        << (row.sparse_energy ? 1 : 0) << ' ' << row.traj_frames << ' '
        << row.energy_points << ' ' << row.kept_frames << ' '
        << row.dropped_outside_overlap << ' ' << row.unmatched_in_overlap << ' '
        << row.traj_min_ps << ' ' << row.traj_max_ps << ' '
        << row.energy_min_ps << ' ' << row.energy_max_ps << ' '
        << row.overlap_start_ps << ' ' << row.overlap_end_ps << ' '
        << row.inferred_dt_energy_ps << '\n';
  }
}

bool is_mbar_equilibrium_weight_source(
    const PostprocEquilibriumWeights &weight_selection) {
  return weight_selection.resolved_source == "mbar" &&
         weight_selection.global_frame_weights &&
         !weight_selection.global_frame_weights->empty();
}

const std::vector<double> *
context_global_frame_weights(const PostprocEquilibriumWeights &weight_selection,
                             bool pca_context) {
  if (!weight_selection.use_weights)
    return nullptr;
  if (weight_selection.resolved_source == "mbar") {
    return pca_context ? weight_selection.pca_global_frame_weights
                       : weight_selection.tica_global_frame_weights;
  }
  return weight_selection.global_frame_weights;
}

MappedEquilibriumFrameWeights map_global_weights_to_local_frame_context(
    const std::string &label, const std::vector<int64_t> &local_global_indices,
    const std::vector<double> &global_frame_weights) {
  MappedEquilibriumFrameWeights mapped;
  mapped.local_weights.assign(local_global_indices.size(), 0.0);
  if (local_global_indices.empty() || global_frame_weights.empty())
    return mapped;

  double sum_sq = 0.0;
  std::size_t weighted_frames = 0;
  for (std::size_t idx = 0; idx < local_global_indices.size(); ++idx) {
    const int64_t global_idx = local_global_indices[idx];
    if (global_idx < 0 ||
        static_cast<std::size_t>(global_idx) >= global_frame_weights.size()) {
      ++mapped.invalid_frames;
      continue;
    }
    const double weight =
        global_frame_weights[static_cast<std::size_t>(global_idx)];
    if (!std::isfinite(weight) || weight < 0.0) {
      ++mapped.invalid_frames;
      continue;
    }
    mapped.local_weights[idx] = weight;
    mapped.raw_weight_sum += weight;
    sum_sq += weight * weight;
    ++mapped.matched_frames;
    if (weight > 0.0)
      ++weighted_frames;
    mapped.ranked_local_indices.push_back(idx);
  }

  if (mapped.raw_weight_sum <= 0.0) {
    TitanUI::print_warning(label + ": mapped equilibrium weights sum to zero. "
                                   "Skipping weighted sidecar export.");
    return mapped;
  }

  mapped.use_weights = true;
  mapped.ess =
      (sum_sq > 0.0) ? (mapped.raw_weight_sum * mapped.raw_weight_sum / sum_sq)
                     : 0.0;
  std::stable_sort(mapped.ranked_local_indices.begin(),
                   mapped.ranked_local_indices.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     const double wl = mapped.local_weights[lhs];
                     const double wr = mapped.local_weights[rhs];
                     if (wl != wr)
                       return wl > wr;
                     return lhs < rhs;
                   });

  TitanUI::print_info(label + ": Mapped equilibrium weights onto " +
                      std::to_string(weighted_frames) + "/" +
                      std::to_string(local_global_indices.size()) +
                      " frames (matched indices: " +
                      std::to_string(mapped.matched_frames) +
                      ", invalid frames: " +
                      std::to_string(mapped.invalid_frames) +
                      ", weight sum = " +
                      std::to_string(mapped.raw_weight_sum) + ").");
  return mapped;
}

std::string residue_label(const std::string &resname, int resnr,
                          const std::string &chain) {
  return resname + std::to_string(resnr) + "_" + chain;
}

void ensure_csv_header_if_missing(const std::string &filename,
                                  const std::string &header) {
  bool has_content = false;
  {
    std::ifstream in(filename);
    if (in.is_open()) {
      std::string first_line;
      if (std::getline(in, first_line) && !first_line.empty())
        has_content = true;
    }
  }
  if (has_content)
    return;

  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("Cannot open " + filename +
                           " to create a header-only placeholder.");
    return;
  }
  out << header << "\n";
}

void write_equilibrium_provenance(std::ofstream &out,
                                  const PostprocEquilibriumWeights &selection,
                                  const MappedEquilibriumFrameWeights &mapped,
                                  const std::string &note,
                                  int embedding_dims = -1) {
  out << "# source " << selection.resolved_source << "\n";
  out << "# target_temp_k " << std::setprecision(15) << selection.target_temp_k
      << "\n";
  out << "# matched_frames " << mapped.matched_frames << "\n";
  out << "# invalid_frames " << mapped.invalid_frames << "\n";
  out << "# raw_weight_sum " << mapped.raw_weight_sum << "\n";
  out << "# ess " << mapped.ess << "\n";
  if (embedding_dims >= 0)
    out << "# embedding_dims " << embedding_dims << "\n";
  out << "# note " << note << "\n";
}

void write_umap_equilibrium_summary_file(
    const std::string &filename, const TitanUMAP &umap_analyzer,
    const TitanMSM &msm_analyzer, const MappedEquilibriumFrameWeights &mapped,
    const PostprocEquilibriumWeights &weight_selection) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("UMAP: cannot open " + filename +
                           " for equilibrium summary export.");
    return;
  }

  const auto &embedding = umap_analyzer.embedding();
  write_equilibrium_provenance(
      out, weight_selection, mapped,
      "contextual_only raw UMAP embedding; MBAR weights apply only to this "
      "summary and optional UMAP FES sidecars.",
      static_cast<int>(embedding.cols()));
  out << "Rank LocalFrame GlobalFrame ReplicaID Time_ps Weight NormalizedWeight "
         "UMAP1 UMAP2 UMAP3\n";
  out << std::setprecision(15);

  const auto &global_indices = msm_analyzer.get_frame_global_indices();
  const auto &replica_ids = msm_analyzer.get_frame_replica_ids();
  const auto &timestamps = msm_analyzer.get_frame_timestamps();
  const double weight_norm =
      (mapped.raw_weight_sum > 0.0) ? mapped.raw_weight_sum : 1.0;

  for (std::size_t rank = 0; rank < mapped.ranked_local_indices.size(); ++rank) {
    const std::size_t local_idx = mapped.ranked_local_indices[rank];
    const double weight = mapped.local_weights[local_idx];
    const int64_t global_idx =
        (local_idx < global_indices.size()) ? global_indices[local_idx] : -1;
    const int replica_id =
        (local_idx < replica_ids.size()) ? replica_ids[local_idx] : -1;
    const double time_ps =
        (local_idx < timestamps.size())
            ? timestamps[local_idx]
            : std::numeric_limits<double>::quiet_NaN();
    const double umap1 =
        (local_idx < static_cast<std::size_t>(embedding.rows()) &&
         embedding.cols() > 0)
            ? embedding(static_cast<Eigen::Index>(local_idx), 0)
            : std::numeric_limits<double>::quiet_NaN();
    const double umap2 =
        (local_idx < static_cast<std::size_t>(embedding.rows()) &&
         embedding.cols() > 1)
            ? embedding(static_cast<Eigen::Index>(local_idx), 1)
            : std::numeric_limits<double>::quiet_NaN();
    const double umap3 =
        (local_idx < static_cast<std::size_t>(embedding.rows()) &&
         embedding.cols() > 2)
            ? embedding(static_cast<Eigen::Index>(local_idx), 2)
            : std::numeric_limits<double>::quiet_NaN();
    out << (rank + 1) << ' ' << local_idx << ' ' << global_idx << ' '
        << replica_id << ' ' << time_ps << ' ' << weight << ' '
        << (weight / weight_norm) << ' ' << umap1 << ' ' << umap2 << ' '
        << umap3 << '\n';
  }
}

void write_dccm_equilibrium_context_file(
    const std::string &filename, const DCCMNetwork &dccm_analyzer,
    const MappedEquilibriumFrameWeights &mapped,
    const PostprocEquilibriumWeights &weight_selection) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("DCCM: cannot open " + filename +
                           " for equilibrium context export.");
    return;
  }

  write_equilibrium_provenance(
      out, weight_selection, mapped,
      "contextual_only ranked DCCM/LSP-MD frames under MBAR equilibrium "
      "weights; no network or allostery scores are reweighted.");
  out << "Rank LocalFrame GlobalFrame ReplicaID Time_ps Weight NormalizedWeight\n";
  out << std::setprecision(15);

  const auto &global_indices = dccm_analyzer.get_global_frame_indices();
  const auto &replica_ids = dccm_analyzer.get_traj_replicas();
  const auto &timestamps = dccm_analyzer.get_traj_times();
  const double weight_norm =
      (mapped.raw_weight_sum > 0.0) ? mapped.raw_weight_sum : 1.0;

  for (std::size_t rank = 0; rank < mapped.ranked_local_indices.size(); ++rank) {
    const std::size_t local_idx = mapped.ranked_local_indices[rank];
    const double weight = mapped.local_weights[local_idx];
    const int64_t global_idx =
        (local_idx < global_indices.size()) ? global_indices[local_idx] : -1;
    const int replica_id =
        (local_idx < replica_ids.size()) ? replica_ids[local_idx] : -1;
    const double time_ps =
        (local_idx < timestamps.size())
            ? timestamps[local_idx]
            : std::numeric_limits<double>::quiet_NaN();
    out << (rank + 1) << ' ' << local_idx << ' ' << global_idx << ' '
        << replica_id << ' ' << time_ps << ' ' << weight << ' '
        << (weight / weight_norm) << '\n';
  }
}

std::vector<PairScoreRow> read_pair_score_rows(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open())
    return {};

  std::vector<PairScoreRow> rows;
  std::string line;
  bool header_skipped = false;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    if (!header_skipped) {
      header_skipped = true;
      if (line.find("Source") != std::string::npos &&
          line.find("Target") != std::string::npos) {
        continue;
      }
    }
    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> cols;
    while (std::getline(ss, token, ','))
      cols.push_back(token);
    if (cols.size() < 3)
      continue;
    try {
      PairScoreRow row;
      row.source = cols[0];
      row.target = cols[1];
      row.score = std::stod(cols[2]);
      rows.push_back(std::move(row));
    } catch (...) {
      continue;
    }
  }

  return rows;
}

void append_ranked_pair_scores(std::ofstream &out, const std::string &metric,
                               std::vector<PairScoreRow> rows) {
  std::stable_sort(rows.begin(), rows.end(),
                   [](const PairScoreRow &lhs, const PairScoreRow &rhs) {
                     if (lhs.score != rhs.score)
                       return lhs.score > rhs.score;
                     if (lhs.source != rhs.source)
                       return lhs.source < rhs.source;
                     return lhs.target < rhs.target;
                   });
  for (std::size_t idx = 0; idx < rows.size(); ++idx) {
    out << metric << ',' << (idx + 1) << ',' << rows[idx].source << ','
        << rows[idx].target << ',' << std::setprecision(15) << rows[idx].score
        << "\n";
  }
}

void write_allostery_equilibrium_ranking_file(
    const std::string &filename, const info &info_output,
    const MappedEquilibriumFrameWeights &mapped,
    const PostprocEquilibriumWeights &weight_selection) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("Allostery: cannot open " + filename +
                           " for equilibrium ranking export.");
    return;
  }

  write_equilibrium_provenance(
      out, weight_selection, mapped,
      "contextual_only raw MI/GC/TE exports are ranked under MBAR frame "
      "context; scores below are not reweighted.");
  out << "Metric,Rank,Source,Target,Score\n";
  append_ranked_pair_scores(
      out, "MI",
      read_pair_score_rows(info_output.get_network_mi_edges_filename()));
  append_ranked_pair_scores(
      out, "GC",
      read_pair_score_rows(info_output.get_network_gc_edges_filename()));
  append_ranked_pair_scores(
      out, "TE",
      read_pair_score_rows(info_output.get_network_te_edges_filename()));
}

void write_corr_knn_equilibrium_ranking_file(
    const std::string &filename, const info &info_output,
    const MappedEquilibriumFrameWeights &mapped,
    const PostprocEquilibriumWeights &weight_selection) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("Corr-kNN: cannot open " + filename +
                           " for equilibrium ranking export.");
    return;
  }

  write_equilibrium_provenance(
      out, weight_selection, mapped,
      "contextual_only raw corr-kNN MI/GC/TE exports are ranked under MBAR "
      "frame context; scores below are not reweighted.");
  out << "Metric,Rank,Source,Target,Score\n";
  append_ranked_pair_scores(
      out, "MI",
      read_pair_score_rows(info_output.get_corr_knn_mi_pairs_filename()));
  append_ranked_pair_scores(
      out, "GC",
      read_pair_score_rows(info_output.get_corr_knn_gc_pairs_filename()));
  append_ranked_pair_scores(
      out, "TE",
      read_pair_score_rows(info_output.get_corr_knn_te_pairs_filename()));
}

void write_lspmd_equilibrium_ranking_file(
    const std::string &filename, const DCCMNetwork &dccm_analyzer,
    const DCCMNetwork::LSPMDResult &result,
    const MappedEquilibriumFrameWeights &mapped,
    const PostprocEquilibriumWeights &weight_selection) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("LSP-MD: cannot open " + filename +
                           " for equilibrium ranking export.");
    return;
  }

  write_equilibrium_provenance(
      out, weight_selection, mapped,
      "contextual_only raw LSP-MD centrality and edge weights are ranked under "
      "MBAR frame context; scores below are not reweighted.");
  out << "EntityKind,Metric,Rank,Label,Source,Target,Score,ScoreNorm\n";
  out << std::setprecision(15);

  struct ResidueMetricRow {
    std::string label{};
    double score{0.0};
    double score_norm{0.0};
  };

  const auto &resnames = dccm_analyzer.get_resnames();
  const auto &resnrs = dccm_analyzer.get_resnrs();
  const auto &chains = dccm_analyzer.get_chains();

  const auto build_residue_rows = [&](const std::vector<double> &values) {
    std::vector<ResidueMetricRow> rows;
    rows.reserve(values.size());
    double max_value = 0.0;
    for (double value : values) {
      if (std::isfinite(value))
        max_value = std::max(max_value, value);
    }
    if (max_value <= 0.0)
      max_value = 1.0;
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
      const double value = values[idx];
      ResidueMetricRow row;
      row.label =
          residue_label(idx < resnames.size() ? resnames[idx] : "UNK",
                        idx < resnrs.size() ? resnrs[idx] : -1,
                        idx < chains.size() ? chains[idx] : "?");
      row.score = value;
      row.score_norm = value / max_value;
      rows.push_back(std::move(row));
    }
    std::stable_sort(rows.begin(), rows.end(),
                     [](const ResidueMetricRow &lhs,
                        const ResidueMetricRow &rhs) {
                       if (lhs.score_norm != rhs.score_norm)
                         return lhs.score_norm > rhs.score_norm;
                       if (lhs.score != rhs.score)
                         return lhs.score > rhs.score;
                       return lhs.label < rhs.label;
                     });
    return rows;
  };

  const auto dc_rows = build_residue_rows(result.dc);
  for (std::size_t idx = 0; idx < dc_rows.size(); ++idx) {
    out << "residue,DC_norm," << (idx + 1) << ',' << dc_rows[idx].label
        << ",,," << dc_rows[idx].score << ',' << dc_rows[idx].score_norm
        << "\n";
  }
  const auto bc_rows = build_residue_rows(result.bc);
  for (std::size_t idx = 0; idx < bc_rows.size(); ++idx) {
    out << "residue,BC_norm," << (idx + 1) << ',' << bc_rows[idx].label
        << ",,," << bc_rows[idx].score << ',' << bc_rows[idx].score_norm
        << "\n";
  }

  struct EdgeMetricRow {
    std::string source{};
    std::string target{};
    double score{0.0};
    double score_norm{0.0};
  };

  std::vector<EdgeMetricRow> edge_rows;
  edge_rows.reserve(result.edges.size());
  double max_weight = 0.0;
  for (const auto &[i, j, weight] : result.edges) {
    if (std::isfinite(weight))
      max_weight = std::max(max_weight, weight);
    EdgeMetricRow row;
    row.source =
        residue_label((i >= 0 && static_cast<std::size_t>(i) < resnames.size())
                          ? resnames[static_cast<std::size_t>(i)]
                          : "UNK",
                      (i >= 0 && static_cast<std::size_t>(i) < resnrs.size())
                          ? resnrs[static_cast<std::size_t>(i)]
                          : -1,
                      (i >= 0 && static_cast<std::size_t>(i) < chains.size())
                          ? chains[static_cast<std::size_t>(i)]
                          : "?");
    row.target =
        residue_label((j >= 0 && static_cast<std::size_t>(j) < resnames.size())
                          ? resnames[static_cast<std::size_t>(j)]
                          : "UNK",
                      (j >= 0 && static_cast<std::size_t>(j) < resnrs.size())
                          ? resnrs[static_cast<std::size_t>(j)]
                          : -1,
                      (j >= 0 && static_cast<std::size_t>(j) < chains.size())
                          ? chains[static_cast<std::size_t>(j)]
                          : "?");
    row.score = weight;
    edge_rows.push_back(std::move(row));
  }
  if (max_weight <= 0.0)
    max_weight = 1.0;
  for (auto &row : edge_rows)
    row.score_norm = row.score / max_weight;
  std::stable_sort(edge_rows.begin(), edge_rows.end(),
                   [](const EdgeMetricRow &lhs, const EdgeMetricRow &rhs) {
                     if (lhs.score != rhs.score)
                       return lhs.score > rhs.score;
                     if (lhs.source != rhs.source)
                       return lhs.source < rhs.source;
                     return lhs.target < rhs.target;
                   });
  for (std::size_t idx = 0; idx < edge_rows.size(); ++idx) {
    out << "edge,Weight," << (idx + 1) << ",," << edge_rows[idx].source << ','
        << edge_rows[idx].target << ',' << edge_rows[idx].score << ','
        << edge_rows[idx].score_norm << "\n";
  }
}

AlignedDoubleVector map_global_weights_to_local_indices(
    const std::string &label, const std::vector<int64_t> &local_global_indices,
    const std::vector<double> &global_frame_weights, bool &use_weights) {
  const auto mapped_context = map_global_weights_to_local_frame_context(
      label, local_global_indices, global_frame_weights);
  use_weights = mapped_context.use_weights;
  return mapped_context.local_weights;
}

struct EquilibriumPopulationSummary {
  std::vector<double> microstate_populations{};
  std::vector<double> macrostate_populations{};
  std::vector<int64_t> microstate_frame_counts{};
  std::vector<int64_t> macrostate_frame_counts{};
  std::size_t matched_frames{0};
  std::size_t invalid_frames{0};
  double raw_weight_sum{0.0};
};

TitanPCA::WeightContext
make_pca_weight_context(const PostprocEquilibriumWeights &weight_selection) {
  TitanPCA::WeightContext context;
  context.source_name = weight_selection.source_name;
  context.mode_label = weight_selection.mode_label;
  context.target_temp = weight_selection.target_temp_k;
  return context;
}

bool compute_equilibrium_populations(
    const TitanMSM &msm_analyzer, const std::vector<double> &global_frame_weights,
    const std::vector<int> &macro_assignments,
    EquilibriumPopulationSummary &summary) {
  summary = {};
  if (!msm_analyzer.is_built() || global_frame_weights.empty())
    return false;

  const int n_states = msm_analyzer.n_clusters();
  if (n_states <= 0)
    return false;

  const auto &discrete_traj = msm_analyzer.discrete_traj();
  const auto &frame_global_indices = msm_analyzer.get_frame_global_indices();
  const std::size_t tracked_frames =
      std::min(discrete_traj.size(), frame_global_indices.size());
  if (tracked_frames == 0)
    return false;

  summary.microstate_populations.assign(static_cast<std::size_t>(n_states), 0.0);
  summary.microstate_frame_counts.assign(static_cast<std::size_t>(n_states), 0);
  if (macro_assignments.size() == static_cast<std::size_t>(n_states)) {
    const int n_macro =
        1 + *std::max_element(macro_assignments.begin(), macro_assignments.end());
    if (n_macro > 0) {
      summary.macrostate_populations.assign(static_cast<std::size_t>(n_macro),
                                            0.0);
      summary.macrostate_frame_counts.assign(static_cast<std::size_t>(n_macro), 0);
    }
  }

  for (std::size_t frame = 0; frame < tracked_frames; ++frame) {
    const int state = discrete_traj[frame];
    const int64_t global_index = frame_global_indices[frame];
    if (state < 0 || state >= n_states || global_index < 0 ||
        static_cast<std::size_t>(global_index) >= global_frame_weights.size()) {
      ++summary.invalid_frames;
      continue;
    }
    const double weight = global_frame_weights[static_cast<std::size_t>(global_index)];
    if (!std::isfinite(weight) || weight < 0.0) {
      ++summary.invalid_frames;
      continue;
    }
    const std::size_t state_index = static_cast<std::size_t>(state);
    summary.microstate_populations[state_index] += weight;
    summary.microstate_frame_counts[state_index] += 1;
    summary.raw_weight_sum += weight;
    ++summary.matched_frames;

    if (!summary.macrostate_populations.empty()) {
      const int macro = macro_assignments[state_index];
      if (macro >= 0 &&
          static_cast<std::size_t>(macro) < summary.macrostate_populations.size()) {
        const std::size_t macro_index = static_cast<std::size_t>(macro);
        summary.macrostate_populations[macro_index] += weight;
        summary.macrostate_frame_counts[macro_index] += 1;
      }
    }
  }

  if (summary.raw_weight_sum <= 0.0)
    return false;

  for (double &value : summary.microstate_populations)
    value /= summary.raw_weight_sum;
  for (double &value : summary.macrostate_populations)
    value /= summary.raw_weight_sum;
  return true;
}

void write_msm_equilibrium_summary_file(
    const std::string &filename, const TitanMSM &msm_analyzer,
    const std::vector<int> &macro_assignments,
    const EquilibriumPopulationSummary &summary,
    const PostprocEquilibriumWeights &weight_selection) {
  std::ofstream out(filename);
  out << "# source " << weight_selection.resolved_source << "\n";
  out << "# target_temp_k " << weight_selection.target_temp_k << "\n";
  out << "# matched_frames " << summary.matched_frames << "\n";
  out << "# invalid_frames " << summary.invalid_frames << "\n";
  out << "# raw_weight_sum " << std::setprecision(15) << summary.raw_weight_sum
      << "\n";
  out << "# microstate frame_count equilibrium_population msm_pi pi_ratio "
         "tica_c1 tica_c2\n";
  out << std::setprecision(15);
  for (int state = 0; state < msm_analyzer.n_clusters(); ++state) {
    const std::size_t idx = static_cast<std::size_t>(state);
    const double eq_pi =
        (idx < summary.microstate_populations.size())
            ? summary.microstate_populations[idx]
            : 0.0;
    const double msm_pi =
        (state < msm_analyzer.stationary_dist().size())
            ? msm_analyzer.stationary_dist()[state]
            : std::numeric_limits<double>::quiet_NaN();
    const double ratio =
        (std::isfinite(msm_pi) && msm_pi > 0.0)
            ? (eq_pi / msm_pi)
            : std::numeric_limits<double>::quiet_NaN();
    const double pc1 =
        (msm_analyzer.centroids().cols() > 0)
            ? msm_analyzer.centroids()(state, 0)
            : std::numeric_limits<double>::quiet_NaN();
    const double pc2 =
        (msm_analyzer.centroids().cols() > 1)
            ? msm_analyzer.centroids()(state, 1)
            : std::numeric_limits<double>::quiet_NaN();
    out << state << ' '
        << ((idx < summary.microstate_frame_counts.size())
                ? summary.microstate_frame_counts[idx]
                : 0)
        << ' ' << eq_pi << ' ' << msm_pi << ' ' << ratio << ' ' << pc1 << ' '
        << pc2 << '\n';
  }

  if (!summary.macrostate_populations.empty() &&
      macro_assignments.size() == static_cast<std::size_t>(msm_analyzer.n_clusters())) {
    std::vector<int64_t> macro_microstate_counts(summary.macrostate_populations.size(),
                                                 0);
    for (int macro : macro_assignments) {
      if (macro >= 0 &&
          static_cast<std::size_t>(macro) < macro_microstate_counts.size()) {
        macro_microstate_counts[static_cast<std::size_t>(macro)] += 1;
      }
    }
    out << "\n# macrostate microstate_count frame_count equilibrium_population\n";
    for (std::size_t macro = 0; macro < summary.macrostate_populations.size();
         ++macro) {
      out << macro << ' ' << macro_microstate_counts[macro] << ' '
          << ((macro < summary.macrostate_frame_counts.size())
                  ? summary.macrostate_frame_counts[macro]
                  : 0)
          << ' ' << summary.macrostate_populations[macro] << '\n';
    }
  }
}

void write_tpt_equilibrium_overlay_file(
    const std::string &filename, const TitanTPT &tpt_analyzer,
    const std::vector<int> &macro_assignments,
    const EquilibriumPopulationSummary &summary,
    const PostprocEquilibriumWeights &weight_selection) {
  std::ofstream out(filename);
  out << "# source " << weight_selection.resolved_source << "\n";
  out << "# target_temp_k " << weight_selection.target_temp_k << "\n";
  out << "# matched_frames " << summary.matched_frames << "\n";
  out << "# invalid_frames " << summary.invalid_frames << "\n";
  out << "# raw_weight_sum " << std::setprecision(15) << summary.raw_weight_sum
      << "\n";
  out << std::setprecision(15);

  const auto accumulate_population = [&](const std::vector<int> &states) {
    double total = 0.0;
    for (int state : states) {
      if (state >= 0 &&
          static_cast<std::size_t>(state) < summary.microstate_populations.size()) {
        total += summary.microstate_populations[static_cast<std::size_t>(state)];
      }
    }
    return total;
  };

  const double source_eq = accumulate_population(tpt_analyzer.source_states());
  const double sink_eq = accumulate_population(tpt_analyzer.sink_states());
  const double intermediate_eq = std::max(0.0, 1.0 - source_eq - sink_eq);
  out << "source_population_eq " << source_eq << "\n";
  out << "sink_population_eq " << sink_eq << "\n";
  out << "intermediate_population_eq " << intermediate_eq << "\n";
  out << "source_population_model " << tpt_analyzer.summary().source_population
      << "\n";
  out << "sink_population_model " << tpt_analyzer.summary().sink_population
      << "\n";
  out << "intermediate_population_model "
      << tpt_analyzer.summary().intermediate_population << "\n\n";

  out << "# state set macrostate model_pi equilibrium_pi eq_over_model_pi\n";
  const auto &source_states = tpt_analyzer.source_states();
  const auto &sink_states = tpt_analyzer.sink_states();
  for (int state = 0; state < tpt_analyzer.n_states(); ++state) {
    char label = 'I';
    if (std::find(source_states.begin(), source_states.end(), state) !=
        source_states.end()) {
      label = 'A';
    } else if (std::find(sink_states.begin(), sink_states.end(), state) !=
               sink_states.end()) {
      label = 'B';
    }
    const int macro =
        (macro_assignments.size() == static_cast<std::size_t>(tpt_analyzer.n_states()))
            ? macro_assignments[static_cast<std::size_t>(state)]
            : -1;
    const double model_pi = tpt_analyzer.stationary_dist()[state];
    const double eq_pi =
        (static_cast<std::size_t>(state) < summary.microstate_populations.size())
            ? summary.microstate_populations[static_cast<std::size_t>(state)]
            : 0.0;
    const double ratio =
        (std::isfinite(model_pi) && model_pi > 0.0)
            ? (eq_pi / model_pi)
            : std::numeric_limits<double>::quiet_NaN();
    out << state << ' ' << label << ' ' << macro << ' ' << model_pi << ' '
        << eq_pi << ' ' << ratio << '\n';
  }
}

std::vector<std::pair<double, double>>
read_projection_csv_pairs(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open())
    return {};

  std::vector<std::pair<double, double>> projections;
  std::string line;
  bool first_line = true;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    if (first_line) {
      first_line = false;
      if (line.find("Frame") != std::string::npos)
        continue;
    }
    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> cols;
    while (std::getline(ss, token, ','))
      cols.push_back(token);
    if (cols.size() < 3)
      continue;
    try {
      projections.emplace_back(std::stod(cols[1]), std::stod(cols[2]));
    } catch (...) {
      continue;
    }
  }
  return projections;
}

int select_effective_msm_clusters(const arguments &args,
                                  const TitanKinetics &tica_analyzer) {
  const int requested_k = std::max(2, args.msm.msm_clusters);
  if (args.msm.msm_clusters_user_set) {
    return requested_k;
  }

  // Auto mode is intentionally conservative for DHAM-facing workflows.
  // We only reduce the default cluster count when the TICA dataset is too
  // short to support a stable microstate model.
  const int64_t n_frames = tica_analyzer.num_frames();
  const int64_t n_pairs = tica_analyzer.num_pairs();
  const int64_t by_frames = std::max<int64_t>(2, n_frames / 5);
  const int64_t by_pairs =
      (n_pairs > 0) ? std::max<int64_t>(2, n_pairs / 3) : by_frames;
  const int64_t auto_cap =
      std::min<int64_t>(requested_k, std::min(by_frames, by_pairs));
  return std::clamp(static_cast<int>(auto_cap), 2, requested_k);
}

std::string describe_msm_cluster_selection(const arguments &args,
                                           int effective_k,
                                           const TitanKinetics &tica_analyzer) {
  const std::string mode =
      args.msm.msm_clusters_user_set ? "user" : "auto";
  return mode + " (requested=" + std::to_string(args.msm.msm_clusters) +
         ", effective=" + std::to_string(effective_k) +
         ", frames=" + std::to_string(tica_analyzer.num_frames()) +
         ", lagged_pairs=" + std::to_string(tica_analyzer.num_pairs()) + ")";
}

void rebuild_msm_projections_from_dccm(const arguments &args,
                                       const DCCMNetwork &dccm_analyzer,
                                       const TitanKinetics &tica_analyzer,
                                       TitanMSM &msm_analyzer,
                                       std::fstream &log) {
  const size_t n_frames = dccm_analyzer.get_total_frames();
  const int64_t expected_frames = tica_analyzer.num_frames();
  const int n_ca = static_cast<int>(dccm_analyzer.get_resnrs().size());
  const int feat_dim = tica_analyzer.feature_dim();

  if (n_frames == 0 || n_ca <= 0 || feat_dim <= 0) {
    TitanUI::print_warning(
        "MSM: Cannot rebuild TICA projections from cached DCCM trajectory.");
    return;
  }
  if (expected_frames > 0 && static_cast<int64_t>(n_frames) != expected_frames) {
    TitanUI::print_warning(
        "MSM: DCCM cache holds " + std::to_string(n_frames) +
        " frames, but TICA accumulated " + std::to_string(expected_frames) +
        ". Refusing partial projection rebuild.");
    return;
  }

  const auto &traj_x = dccm_analyzer.get_traj_x();
  const auto &traj_y = dccm_analyzer.get_traj_y();
  const auto &traj_z = dccm_analyzer.get_traj_z();
  const auto &traj_times = dccm_analyzer.get_traj_times();
  const auto &traj_replicas = dccm_analyzer.get_traj_replicas();
  const auto &global_indices = dccm_analyzer.get_global_frame_indices();

  if (traj_x.size() != n_frames || traj_y.size() != n_frames ||
      traj_z.size() != n_frames) {
    TitanUI::print_warning(
        "MSM: DCCM trajectory cache is inconsistent. Skipping projection rebuild.");
    return;
  }

  msm_analyzer.clear_projected();

  Eigen::VectorXd features(feat_dim);
  Eigen::MatrixXd mobile(n_ca, 3);
  Eigen::MatrixXd ref_centered(n_ca, 3);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  bool ref_initialized = false;
  size_t n_breaks = 0;

  for (size_t f = 0; f < n_frames; ++f) {
    if (traj_x[f].size() < static_cast<size_t>(n_ca) ||
        traj_y[f].size() < static_cast<size_t>(n_ca) ||
        traj_z[f].size() < static_cast<size_t>(n_ca)) {
      TitanUI::print_warning(
          "MSM: Encountered truncated cached CA coordinates. Aborting projection rebuild.");
      msm_analyzer.clear_projected();
      return;
    }

    for (int a = 0; a < n_ca; ++a) {
      const size_t ai = static_cast<size_t>(a);
      mobile(a, 0) = traj_x[f][ai];
      mobile(a, 1) = traj_y[f][ai];
      mobile(a, 2) = traj_z[f][ai];
    }

    if (!ref_initialized) {
      ref_centered = mobile;
      ref_centered.rowwise() -= ref_centered.colwise().mean().eval();
      mobile = ref_centered;
      ref_initialized = true;
    } else {
      TitanAlignment::align_frame_inplace(ref_centered, mobile, R);
    }

    int feat_idx = 0;
    for (int a = 0; a < n_ca; ++a) {
      for (int b = a + 1; b < n_ca; ++b) {
        const double dx = mobile(a, 0) - mobile(b, 0);
        const double dy = mobile(a, 1) - mobile(b, 1);
        const double dz = mobile(a, 2) - mobile(b, 2);
        features[feat_idx++] = std::sqrt(dx * dx + dy * dy + dz * dz);
      }
    }

    const int replica_id =
        (traj_replicas.size() > f) ? traj_replicas[f] : -1;
    const int64_t global_idx =
        (global_indices.size() > f) ? global_indices[f]
                                    : static_cast<int64_t>(f);

    msm_analyzer.add_projected_frame(
        tica_analyzer.project_frame(features), replica_id, global_idx);
    if (traj_times.size() > f)
      msm_analyzer.record_frame_time(traj_times[f], replica_id);

    if (args.msm.msm_auto_boundary && f + 1 < n_frames &&
        is_trajectory_boundary(traj_replicas, traj_times, f, f + 1)) {
      msm_analyzer.add_trajectory_break();
      ++n_breaks;
    }
  }

  TitanUI::log_kv(log, "MSM Projection Rebuild", "from cached DCCM trajectory");
  TitanUI::log_kv(log, "MSM Projected Frames",
                  std::to_string(msm_analyzer.n_frames()));
  if (args.msm.msm_auto_boundary) {
    TitanUI::log_kv(log, "MSM Rebuilt Trajectory Breaks",
                    std::to_string(n_breaks));
  }
}

} // namespace

PostprocEquilibriumWeights resolve_postproc_equilibrium_weights(
    const arguments &args, const std::vector<double> *dham_frame_weights,
    const std::vector<double> *mbar_tica_frame_weights,
    const std::vector<double> *mbar_pca_frame_weights, std::fstream &log) {
  PostprocEquilibriumWeights selection;
  selection.requested_source = args.eq_weight_source;

  const auto use_uniform = [&]() {
    selection.resolved_source = "uniform";
    selection.source_name = "equilibrium";
    selection.mode_label = "uniform";
    selection.target_temp_k = 300.0;
    selection.global_frame_weights = nullptr;
    selection.tica_global_frame_weights = nullptr;
    selection.pca_global_frame_weights = nullptr;
    selection.use_weights = false;
  };
  const auto try_source = [&](const std::string &source_name,
                              const std::vector<double> *weights,
                              double target_temp_k) {
    if (weights && !weights->empty()) {
      selection.resolved_source = source_name;
      selection.source_name = source_name;
      selection.mode_label =
          (source_name == "mbar") ? "MBAR-weighted" : "DHAM-weighted";
      selection.target_temp_k = target_temp_k;
      selection.global_frame_weights = weights;
      selection.tica_global_frame_weights = weights;
      selection.pca_global_frame_weights = weights;
      selection.use_weights = true;
      return true;
    }
    return false;
  };
  const auto try_mbar_source = [&]() {
    const bool have_tica =
        mbar_tica_frame_weights && !mbar_tica_frame_weights->empty();
    const bool have_pca =
        mbar_pca_frame_weights && !mbar_pca_frame_weights->empty();
    if (!have_tica && !have_pca)
      return false;
    selection.resolved_source = "mbar";
    selection.source_name = "mbar";
    selection.mode_label = "MBAR-weighted";
    selection.target_temp_k = args.mbar.target_temp;
    selection.tica_global_frame_weights =
        have_tica ? mbar_tica_frame_weights : nullptr;
    selection.pca_global_frame_weights =
        have_pca ? mbar_pca_frame_weights : nullptr;
    selection.global_frame_weights =
        have_tica ? mbar_tica_frame_weights : mbar_pca_frame_weights;
    selection.use_weights = true;
    return true;
  };

  use_uniform();
  if (selection.requested_source == "auto") {
    try_source("dham", dham_frame_weights, args.dham.target_temp);
  } else if (selection.requested_source == "uniform") {
    use_uniform();
  } else if (selection.requested_source == "dham") {
    if (!args.dham.flag_dham) {
      TitanUI::print_warning(
          "Equilibrium weights: -eq-weight-source dham requested without "
          "-dham. Falling back to uniform.");
    } else if (!try_source("dham", dham_frame_weights, args.dham.target_temp)) {
      TitanUI::print_warning(
          "Equilibrium weights: requested DHAM, but no converged DHAM frame "
          "weights are available. Falling back to uniform.");
    }
  } else if (selection.requested_source == "mbar") {
    if (!args.mbar.flag_mbar) {
      TitanUI::print_warning(
          "Equilibrium weights: -eq-weight-source mbar requested without "
          "-mbar. Falling back to uniform.");
    } else if (!try_mbar_source()) {
      TitanUI::print_warning(
          "Equilibrium weights: requested MBAR, but no converged MBAR frame "
          "weights are available. Falling back to uniform.");
    }
  }

  TitanUI::log_kv(log, "Equilibrium Weight Source", selection.resolved_source);
  if (selection.use_weights) {
    TitanUI::log_kv(log, "Equilibrium Weight Target Temp",
                    std::to_string(selection.target_temp_k));
  }
  return selection;
}

void run_msm_equilibrium_postproc(
    const arguments &args, std::unique_ptr<TitanMSM> &msm_analyzer,
    info &info_output, std::fstream &log,
    const PostprocEquilibriumWeights &weight_selection) {
  (void)args;
  const auto *weights = context_global_frame_weights(weight_selection, false);
  if (!(msm_analyzer && msm_analyzer->is_built() && weight_selection.use_weights &&
        weight_selection.resolved_source == "mbar" &&
        weights)) {
    return;
  }

  const auto macro_assignments =
      msm_analyzer->compute_pcca_macrostates(args.msm.msm_macrostates);
  EquilibriumPopulationSummary summary;
  if (!compute_equilibrium_populations(*msm_analyzer, *weights,
                                       macro_assignments, summary)) {
    TitanUI::print_warning(
        "MSM equilibrium summary: unable to map selected frame weights onto "
        "MSM microstates.");
    return;
  }

  write_msm_equilibrium_summary_file(
      info_output.get_msm_equilibrium_summary_filename(), *msm_analyzer,
      macro_assignments, summary, weight_selection);
  TitanUI::log_kv(log, "MSM Equilibrium Summary",
                  info_output.get_msm_equilibrium_summary_filename());
  TitanUI::print_info("MSM: Exported equilibrium occupancy summary to " +
                      info_output.get_msm_equilibrium_summary_filename());
}

// ============================================================================
// 1. GBSA + ML + CAS + Nucleic Acid post-processing
// ============================================================================
void run_gbsa_postproc(const arguments &args, size_t tot_snap_allfiles,
                       MLFeatureExtractor &ml_extractor,
                       GBSA_Aggregator &global_gbsa_agg,
                       CASAnalyzer &cas_analyzer, NucleicProfiler &na_profiler,
                       bool global_has_nucleic_acid, info &info_output,
                       std::fstream &log) {
  if (args.flag_gbsa) {
    if (args.flag_ml) {
      if (tot_snap_allfiles > 0) {
        TitanUI::print_module_header("Machine Learning Feature Extraction");
        std::string ml_out_file = info_output.get_ml_features_filename();
        ml_extractor.export_features(ml_out_file);
        log << "Exported ML features to " << ml_out_file << "\n";
        TitanUI::print_info("Successfully exported 5 ML features to " +
                            ml_out_file);

        // KIN-10: the bundled random forest is closed. Its training set,
        // feature units, cross-validation score and applicability domain are
        // unknown, and feature 0 is an absolute interaction energy, so the
        // prediction is not transferable between systems, force fields or
        // cutoff conventions. Feature export stays available because the five
        // descriptors are reproducible on their own; only the pKd number is
        // withheld. Re-open it by restoring this branch once a model card
        // exists (see KIN-10 in docs/TITAN_CODE_REVIEW_2026-09-08.md).
        const std::string ml_closed_notice =
            "ML features exported. Affinity prediction is closed: the bundled "
            "model has no training/calibration provenance and its leading "
            "feature is an absolute energy, so the pKd is not transferable. No "
            "prediction file is written.";
        TitanUI::print_info(ml_closed_notice);
        log << "[Titan ML] " << ml_closed_notice << "\n";
        if (args.ml_predict_experimental) {
          const std::string ml_optin_notice =
              "-ml-predict-experimental no longer enables the bundled model; "
              "the option is retained only to report that it has no effect.";
          TitanUI::print_info(ml_optin_notice);
          log << "[Titan ML] " << ml_optin_notice << "\n";
        }
      }
    }
    try {
      if (tot_snap_allfiles > 0) {
        log << "Printing GBSA statistics & decompositions...\n";
        TitanUI::print_module_header("GBSA Statistics & Decompositions");
        std::string out_dir = ".";
        global_gbsa_agg.print_all_stats(out_dir, &log);
        TitanUI::print_info(
            "GBSA Decomposition files successfully written to " + out_dir);
      }
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] Exception in GBSA: " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] Unknown exception in GBSA!\n";
    }

    if (args.flag_cas) {
      try {
        if (tot_snap_allfiles > 0) {
          cas_analyzer.export_results(info_output.get_cas_receptor_filename(),
                                      info_output.get_cas_ligand_filename(),
                                      args.eu_par);
          cas_analyzer.export_hotspot_pdb(
              info_output.get_cas_hotspot_pdb_filename(), args.eu_par);
          cas_analyzer.export_summary(info_output.get_cas_summary_filename(),
                                      args);
          cas_analyzer.export_performance(
              info_output.get_cas_performance_filename());
        }
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] Exception in CAS: " << e.what() << "\n";
      } catch (...) {
        std::cerr << "[ERROR] Unknown exception in CAS!\n";
      }
    }
  }

  if (global_has_nucleic_acid) {
    try {
      if (tot_snap_allfiles > 0) {
        na_profiler.export_results(
            info_output.get_nucleic_acid_interactions_filename());
      }
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] Exception in NA Profiler: " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] Unknown exception in NA Profiler!\n";
    }
  }
}

void run_dssp_postproc(const arguments &args, TitanDSSPAnalyzer &mol1_dssp,
                       TitanDSSPAnalyzer &mol2_dssp, info &info_output,
                       std::fstream &log) {
  if (!args.dssp.flag_dssp)
    return;

  bool exported = false;
  if (mol1_dssp.has_results()) {
    mol1_dssp.export_results(info_output.get_dssp_mol1_filename(), log);
    mol1_dssp.export_content_profile(
        info_output.get_dssp_mol1_content_filename(), log);
    mol1_dssp.export_segment_content(
        info_output.get_dssp_mol1_segments_filename(), log);
    mol1_dssp.export_bridge_registry(
        info_output.get_dssp_mol1_bridges_filename(), log);
    mol1_dssp.export_ladder_distribution(
        info_output.get_dssp_mol1_ladders_filename(), log);
    TitanUI::print_info("[DSSP] Exported mol1 summary to " +
                        info_output.get_dssp_mol1_filename());
    TitanUI::print_info("[DSSP] Exported mol1 content to " +
                        info_output.get_dssp_mol1_content_filename());
    exported = true;
  }
  if (mol2_dssp.has_results()) {
    mol2_dssp.export_results(info_output.get_dssp_mol2_filename(), log);
    mol2_dssp.export_content_profile(
        info_output.get_dssp_mol2_content_filename(), log);
    mol2_dssp.export_segment_content(
        info_output.get_dssp_mol2_segments_filename(), log);
    mol2_dssp.export_bridge_registry(
        info_output.get_dssp_mol2_bridges_filename(), log);
    mol2_dssp.export_ladder_distribution(
        info_output.get_dssp_mol2_ladders_filename(), log);
    TitanUI::print_info("[DSSP] Exported mol2 summary to " +
                        info_output.get_dssp_mol2_filename());
    TitanUI::print_info("[DSSP] Exported mol2 content to " +
                        info_output.get_dssp_mol2_content_filename());
    exported = true;
  }

  if (!exported && log.is_open()) {
    TitanUI::log_kv(log, "DSSP",
                    "No protein/protein-user molecule produced output");
  }
}

// ============================================================================
// 2. TICA + MSM post-processing
// ============================================================================
void run_tica_msm_postproc(const arguments &args,
                           DCCMNetwork &dccm_analyzer,
                           std::unique_ptr<TitanKinetics> &tica_analyzer,
                           std::unique_ptr<TitanMSM> &msm_analyzer,
                           TitanExecutor &executor, info &info_output,
                           std::fstream &log,
                           TitanValidation::RuntimeCache *validation_cache) {
  if (!(args.msm.flag_tica && tica_analyzer))
    return;

  try {
    auto start_tica = std::chrono::steady_clock::now();
    TitanUI::print_module_header("TICA Analysis");
    TitanUI::log_kv(log, "TICA Total Frames",
                    std::to_string(tica_analyzer->num_frames()));
    TitanUI::log_kv(log, "TICA Lagged Pairs",
                    std::to_string(tica_analyzer->num_pairs()));
    TitanUI::log_kv(log, "TICA Backend",
                    tica_backend_to_string(tica_analyzer->backend()));
    TitanUI::log_kv(log, "TICA Rank",
                    std::to_string(tica_analyzer->randomized_rank()));

    tica_analyzer->solve_tica();
    TitanUI::log_kv(log, "TICA Retained Components",
                    std::to_string(tica_analyzer->retained_components()));
    tica_analyzer->write_results(info_output.get_tica_eigenvalues_filename(),
                                  info_output.get_tica_eigenvectors_filename(),
                                  log,
                                  args.trajectory_semantics == "continuous");

    if (msm_analyzer) {
      TitanUI::print_module_header("TICA Projection Cache");
      if (msm_analyzer->n_frames() == 0 &&
          msm_analyzer->pending_feature_frames() > 0) {
        const int64_t pending_frames = msm_analyzer->pending_feature_frames();
        msm_analyzer->project_pending_features(*tica_analyzer);
        TitanUI::log_kv(log, "TICA Projection Rebuild",
                        "from TitanKinetics feature cache");
        TitanUI::log_kv(log, "TICA Pending Feature Frames",
                        std::to_string(pending_frames));
        TitanUI::log_kv(log, "TICA Projected Frames",
                        std::to_string(msm_analyzer->n_frames()));
      }

      if (msm_analyzer->n_frames() == 0) {
        rebuild_msm_projections_from_dccm(args, dccm_analyzer, *tica_analyzer,
                                          *msm_analyzer, log);
      }

      if (args.msm.flag_msm && validation_cache && args.validation.auto_pick &&
          !validation_cache->have_vamp_sweep && msm_analyzer->n_frames() > 0) {
        const int projection_effective_k =
            select_effective_msm_clusters(args, *tica_analyzer);
        TitanValidation::maybe_run_preselect(args, *tica_analyzer,
                                             projection_effective_k,
                                             msm_analyzer, executor, log,
                                             *validation_cache);
      }

      if (msm_analyzer->n_frames() > 0) {
        tica_analyzer->write_projections(
            msm_analyzer->proj_mat_ref(), msm_analyzer->n_frames(),
            info_output.get_tica_projections_filename(), log);
      } else {
        TitanUI::log_kv(log, "TICA Projections",
                        "skipped (no projected frames available)");
      }
    } else {
      TitanUI::log_kv(log, "TICA Projections",
                      "skipped (projection cache not enabled)");
    }

    // --- MSM Phase 2 ---
    if (args.msm.flag_msm && msm_analyzer) {
      const int effective_k =
          select_effective_msm_clusters(args, *tica_analyzer);
      if (validation_cache) {
        TitanValidation::maybe_run_preselect(args, *tica_analyzer, effective_k,
                                             msm_analyzer, executor, log,
                                             *validation_cache);
      }
      int msm_k_for_build = effective_k;
      int msm_lag_for_build = args.msm.msm_lag;
      if (validation_cache && validation_cache->auto_pick_applied &&
          validation_cache->have_vamp_sweep) {
        msm_k_for_build = validation_cache->vamp_sweep.selected.n_clusters;
        msm_lag_for_build = validation_cache->vamp_sweep.selected.lag_frames;
      }
      msm_analyzer->reconfigure_clusters(msm_k_for_build);
      TitanUI::log_kv(log, "MSM Cluster Selection",
                      describe_msm_cluster_selection(args, msm_k_for_build,
                                                     *tica_analyzer));
      TitanUI::print_module_header("MSM Phase 2 — Collecting TICA Projections");
      if (msm_analyzer->n_frames() == 0 &&
          msm_analyzer->pending_feature_frames() > 0) {
        const int64_t pending_frames = msm_analyzer->pending_feature_frames();
        msm_analyzer->project_pending_features(*tica_analyzer);
        TitanUI::log_kv(log, "MSM Projection Rebuild",
                        "from TitanKinetics feature cache");
        TitanUI::log_kv(log, "MSM Pending Feature Frames",
                        std::to_string(pending_frames));
        TitanUI::log_kv(log, "MSM Projected Frames",
                        std::to_string(msm_analyzer->n_frames()));
      }

      if (msm_analyzer->n_frames() == 0) {
        rebuild_msm_projections_from_dccm(args, dccm_analyzer, *tica_analyzer,
                                          *msm_analyzer, log);
      }

      if (msm_analyzer->n_frames() > 0) {
        const bool already_selected_and_built =
            msm_analyzer->is_built() &&
            msm_analyzer->n_clusters() == msm_k_for_build &&
            msm_analyzer->msm_lag_time() == msm_lag_for_build;
        if (!already_selected_and_built) {
          msm_analyzer->train_kmeans(executor, args.msm.msm_kmeans_iter,
                                     static_cast<unsigned>(args.msm.msm_seed));

          if (msm_analyzer->is_clustered())
            msm_analyzer->build_markov_model(msm_lag_for_build);
        } else {
          TitanUI::log_kv(log, "Validation Auto-Pick Reuse",
                          "reusing preselected MSM model");
        }

        if (msm_analyzer->is_built()) {
          TitanMSMItsSolverMode its_solver_mode = TitanMSMItsSolverMode::Full;
          switch (args.msm.msm_its_solver) {
          case MsmArgs::ItsSolverMode::Schur:
            its_solver_mode = TitanMSMItsSolverMode::Schur;
            break;
          case MsmArgs::ItsSolverMode::Partial:
            its_solver_mode = TitanMSMItsSolverMode::Partial;
            break;
          case MsmArgs::ItsSolverMode::Full:
          default:
            its_solver_mode = TitanMSMItsSolverMode::Full;
            break;
          }
          msm_analyzer->export_results(
              executor, info_output, log, args.msm.msm_nts, args.msm.msm_run_its,
              args.msm.msm_run_pcca, args.msm.msm_macrostates,
              args.msm.msm_its_lags, its_solver_mode, args.msm.msm_its_topk,
              args.msm.msm_its_max_k_full,
              args.trajectory_semantics == "continuous");
        }
      } else {
        TitanUI::log_kv(log, "TICA Projections",
                        "skipped (no projected frames available)");
      }
    } else {
      TitanUI::log_kv(log, "MSM Analysis", "skipped (module not enabled)");
    }

    log_tica_timing_stats(log, tica_analyzer->timing_stats());

    auto time_tica = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_tica);
    TitanUI::log_time(log, args.msm.flag_msm ? "TICA + MSM Analysis"
                                             : "TICA Analysis",
                      time_tica.count() / 1000.0);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in TICA/MSM: " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] Unknown exception in TICA/MSM!\n";
  }
}

// ============================================================================
// 3. TICA 2D FES post-processing
// ============================================================================
void run_tica_fes_postproc(const arguments &args,
                           std::unique_ptr<TitanKinetics> &tica_analyzer,
                           std::unique_ptr<TitanMSM> &msm_analyzer,
                           TitanExecutor &executor, info &info_output,
                           std::fstream &log,
                           const PostprocEquilibriumWeights &weight_selection,
                           TitanPCA::FesPostprocMetrics *metrics) {
  (void)tica_analyzer;
  if (!(args.msm.flag_tica && msm_analyzer))
    return;
  if (msm_analyzer->n_frames() <= 0)
    return;

  try {
    auto start_tica_fes = std::chrono::steady_clock::now();
    TitanUI::print_module_header("TICA 2D Free Energy Surface");
    const auto *weights = context_global_frame_weights(weight_selection, false);
    TitanPCA::execute_tica_fes_analysis(
        args, msm_analyzer->proj_mat_ref(), msm_analyzer->n_frames(),
        msm_analyzer->get_frame_global_indices(), executor, info_output, log,
        weights ? *weights : std::vector<double>{},
        make_pca_weight_context(weight_selection), metrics);
    auto time_tica_fes = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_tica_fes);
    TitanUI::log_time(log, "TICA 2D FES", time_tica_fes.count() / 1000.0);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in TICA 2D FES: " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] Unknown exception in TICA 2D FES!\n";
  }
}

void run_umap_postproc(const arguments &args,
                       std::unique_ptr<TitanMSM> &msm_analyzer,
                       std::unique_ptr<TitanUMAP> &umap_analyzer,
                       TitanExecutor &executor,
                       const info &info_output, std::fstream &log,
                       const PostprocEquilibriumWeights &weight_selection) {
  if (!(args.umap.flag_umap && msm_analyzer && msm_analyzer->n_frames() > 0))
    return;

  try {
    auto start_umap = std::chrono::steady_clock::now();
    TitanUI::print_module_header("UMAP Analysis");

    TitanUMAP::Config cfg;
    cfg.n_neighbors = args.umap.umap_neighbors;
    cfg.n_components = args.umap.umap_components;
    cfg.n_epochs = args.umap.umap_epochs;
    cfg.min_dist = args.umap.umap_min_dist;
    cfg.spread = args.umap.umap_spread;
    cfg.seed = args.umap.umap_seed;
    cfg.n_threads = static_cast<int>(std::max<size_t>(1, executor.capacity()));
    cfg.executor = &executor;
    if (args.umap.umap_knn_mode == "approx") {
      cfg.knn_mode = TitanUMAP::KnnMode::Approx;
    } else if (args.umap.umap_knn_mode == "auto") {
      cfg.knn_mode = TitanUMAP::KnnMode::Auto;
    } else {
      cfg.knn_mode = TitanUMAP::KnnMode::Exact;
    }
    cfg.benchmark_knn = args.umap.umap_knn_benchmark;
    cfg.verbose = args.umap.umap_verbose;

    TitanUI::log_kv(log, "UMAP Input Frames",
                    std::to_string(msm_analyzer->n_frames()));
    TitanUI::log_kv(log, "UMAP Input Dimensions",
                    std::to_string(msm_analyzer->proj_mat_ref().cols()));
    TitanUI::log_kv(log, "UMAP Components",
                    std::to_string(cfg.n_components));
    TitanUI::log_kv(log, "UMAP Neighbors",
                    std::to_string(cfg.n_neighbors));
    TitanUI::log_kv(log, "UMAP Epochs", std::to_string(cfg.n_epochs));
    TitanUI::log_kv(log, "UMAP Min Dist", std::to_string(cfg.min_dist));
    TitanUI::log_kv(log, "UMAP Spread", std::to_string(cfg.spread));
    TitanUI::log_kv(log, "UMAP Seed", std::to_string(cfg.seed));
    TitanUI::log_kv(log, "UMAP Threads", std::to_string(cfg.n_threads));
    TitanUI::log_kv(log, "UMAP kNN Mode", args.umap.umap_knn_mode);
    TitanUI::log_kv(log, "UMAP kNN Benchmark",
                    cfg.benchmark_knn ? "on" : "off");

    umap_analyzer = std::make_unique<TitanUMAP>(cfg);
    umap_analyzer->fit_transform(msm_analyzer->proj_mat_ref(),
                                 msm_analyzer->n_frames());
    if (umap_analyzer->is_fitted()) {
      umap_analyzer->write_embedding(
          info_output.get_umap_embedding_filename(), log);
      const auto *weights = context_global_frame_weights(weight_selection, false);
      if (is_mbar_equilibrium_weight_source(weight_selection) && weights) {
        const auto &embedding = umap_analyzer->embedding();
        const auto &global_indices = msm_analyzer->get_frame_global_indices();
        if (static_cast<std::size_t>(embedding.rows()) != global_indices.size()) {
          TitanUI::print_warning(
              "UMAP: frame/global-index count mismatch. Skipping MBAR sidecars.");
        } else {
          const auto mapped_context = map_global_weights_to_local_frame_context(
              "UMAP", global_indices, *weights);
          if (mapped_context.use_weights) {
            write_umap_equilibrium_summary_file(
                info_output.get_umap_equilibrium_summary_filename(),
                *umap_analyzer, *msm_analyzer, mapped_context, weight_selection);
            TitanUI::log_kv(log, "UMAP Equilibrium Summary",
                            info_output.get_umap_equilibrium_summary_filename());
            TitanUI::print_info(
                "UMAP: Exported MBAR equilibrium summary to " +
                info_output.get_umap_equilibrium_summary_filename());

            if (embedding.cols() >= 2) {
              if (embedding.cols() > 2) {
                TitanUI::print_info(
                    "UMAP: MBAR-weighted FES uses UMAP1/UMAP2 while preserving "
                    "the full 3D embedding export.");
              }
              std::vector<std::pair<double, double>> projections(
                  static_cast<std::size_t>(embedding.rows()));
              for (std::size_t frame = 0; frame < projections.size(); ++frame) {
                projections[frame] = {
                    embedding(static_cast<Eigen::Index>(frame), 0),
                    embedding(static_cast<Eigen::Index>(frame), 1)};
              }
              TitanPCA::export_projection_fes_analysis(
                  "UMAP", "UMAP", args, projections, mapped_context.local_weights,
                  true, weight_selection.target_temp_k,
                  weight_selection.mode_label, executor, log,
                  info_output.get_umap_fes_2d_filename(),
                  info_output.get_umap_cluster_representatives_filename(),
                  msm_analyzer->get_frame_timestamps(),
                  msm_analyzer->get_frame_replica_ids());
            } else {
              TitanUI::print_warning(
                  "UMAP: fewer than 2 embedding components are available. "
                  "Skipping MBAR UMAP FES sidecars.");
            }
          }
        }
      }
    }

    auto time_umap = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_umap);
    TitanUI::log_time(log, "UMAP Analysis", time_umap.count() / 1000.0);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in UMAP: " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] Unknown exception in UMAP!\n";
  }
}

// ============================================================================
// 4. LLPS post-processing
// ============================================================================
void run_llps_postproc(const arguments &args,
                       std::unique_ptr<TitanLLPS> &llps_analyzer,
                       TitanExecutor &executor, info &info_output,
                       std::fstream &log) {
  if (!(args.llps.flag_llps && llps_analyzer &&
        llps_analyzer->n_frames_processed() > 0))
    return;

  try {
    auto start_llps = std::chrono::steady_clock::now();
    TitanUI::print_module_header("LLPS Analysis");
    llps_analyzer->finalize(executor, args.llps.llps_maxlag);
    llps_analyzer->export_results(info_output, log);
    auto time_llps = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_llps);
    TitanUI::log_time(log, "LLPS Analysis", time_llps.count() / 1000.0);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in LLPS: " << e.what() << "\n";
  }
}

// ============================================================================
// 5. DCCM + PCA post-processing
// ============================================================================
void run_dccm_pca_postproc(const arguments &args, size_t tot_snap_allfiles,
                           DCCMNetwork &dccm_analyzer, TitanExecutor &executor,
                           std::chrono::milliseconds &time_network,
                           info &info_output, std::fstream &log,
                           const PostprocEquilibriumWeights &weight_selection,
                           const std::vector<std::string> &replica_traj_files,
                           TitanPCA::FesPostprocMetrics *metrics,
                           titan_sparse_graph_module::TitanUnifiedGraphRuntime
                               *graph_runtime) {
  (void)graph_runtime;
  if (!(args.flag_network_explicit || args.flag_pca || args.flag_allostery ||
        args.corr_knn.flag_corr_knn || args.community.flag_community))
    return;

  try {
    if (tot_snap_allfiles > 0) {
      const auto *pca_context_weights =
          context_global_frame_weights(weight_selection, true);
      const bool want_mbar_sidecars =
          is_mbar_equilibrium_weight_source(weight_selection) &&
          pca_context_weights;
      MappedEquilibriumFrameWeights dccm_weight_context;
      auto start_dccm = std::chrono::steady_clock::now();
      TitanUI::print_module_header("DCCM Network Analysis");

      dccm_analyzer.calculate_and_export(executor, info_output);
      if (dccm_analyzer.dense_postproc_skipped_due_memory()) {
        TitanUI::print_warning(
            "DCCM/PCA downstream modules skipped because dense DCCM network "
            "post-processing exceeded the safe memory budget.");
        ensure_csv_header_if_missing(info_output.get_network_mi_edges_filename(),
                                     "Source,Target,MI");
        ensure_csv_header_if_missing(info_output.get_network_gc_edges_filename(),
                                     "Source,Target,GC");
        ensure_csv_header_if_missing(info_output.get_network_te_edges_filename(),
                                     "Source,Target,Net_TE");
        time_network += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_dccm);
        TitanUI::log_time(log, "Network Output", time_network.count() / 1000.0);
        TitanUI::log_kv(log, "DCCM Network Analysis",
                        "skipped by memory guard");
        return;
      }

      if (want_mbar_sidecars) {
        dccm_weight_context = map_global_weights_to_local_frame_context(
            "DCCM", dccm_analyzer.get_global_frame_indices(),
            *pca_context_weights);
      }

      if (dccm_weight_context.use_weights &&
          (args.flag_network_explicit || args.flag_allostery ||
           args.corr_knn.flag_corr_knn ||
           args.community.flag_community)) {
        write_dccm_equilibrium_context_file(
            info_output.get_dccm_equilibrium_context_filename(), dccm_analyzer,
            dccm_weight_context, weight_selection);
        TitanUI::log_kv(log, "DCCM Equilibrium Context",
                        info_output.get_dccm_equilibrium_context_filename());
      }

      time_network += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_dccm);
      TitanUI::log_time(log, "Network Output", time_network.count() / 1000.0);

      TitanCorrKnnAnalyzer corr_knn_cached_analyzer;
      bool corr_knn_cached = false;
      auto corr_knn_cfg =
          TitanCorrKnnAnalyzer::from_corr_knn_args(args.corr_knn);
      auto allostery_knn_cfg =
          TitanCorrKnnAnalyzer::from_allostery_args(args);
      if (dccm_analyzer.network_sparse_mode_active()) {
        auto apply_auto_sparse_pruning = [&](TitanCorrKnnAnalyzer::Config &cfg) {
          if (cfg.pair_pruning.pair_cutoff_ang <= 0.0)
            cfg.pair_pruning.pair_cutoff_ang = args.network_sparse_cut;
          if (cfg.pair_pruning.pair_topk <= 0)
            cfg.pair_pruning.pair_topk = args.network_sparse_k;
          cfg.pair_pruning.use_avg_cutoff = false;
        };
        apply_auto_sparse_pruning(corr_knn_cfg);
        apply_auto_sparse_pruning(allostery_knn_cfg);
        TitanUI::print_warning(
            "Information-theory modules are using sparse pair pruning inherited "
            "from DCCM auto-sparse mode.");
      }
      auto log_exact_corr_knn_result =
          [&](const std::string &label_prefix,
              const TitanCorrKnnAnalyzer &analyzer,
              const std::string &summary_path, const std::string &mi_path,
              const std::string &gc_path, const std::string &te_path) {
            const auto &result = analyzer.result();
            if (!summary_path.empty()) {
              TitanUI::log_kv(log, label_prefix + " Summary File", summary_path);
            }
            TitanUI::log_kv(log, label_prefix + " MI File", mi_path);
            TitanUI::log_kv(log, label_prefix + " GC File", gc_path);
            TitanUI::log_kv(log, label_prefix + " TE File", te_path);
            TitanUI::log_kv(
                log, label_prefix + " TE Stage",
                analyzer.config().compute_te ? "enabled" : "disabled");
            TitanUI::log_kv(
                log, label_prefix + " Estimator",
                "knn (exact continuous 3D, k=" +
                    std::to_string(analyzer.config().estimator_k) + ")");
            TitanUI::log_kv(log, label_prefix + " Tau Lag",
                            std::to_string(analyzer.config().tau_lag));
            TitanUI::log_kv(
                log, label_prefix + " Pair Filter",
                titan_corr_knn::describe_pair_filter(
                    analyzer.config().pair_pruning));
            TitanUI::log_kv(
                log, label_prefix + " Candidate Pairs",
                std::to_string(result.candidate_pair_count) + " / " +
                    std::to_string(result.dense_pair_count));
            if (!result.mi_available) {
              TitanUI::print_warning(
                  label_prefix + ": insufficient effective frames for exact MI "
                                 "with k=" +
                  std::to_string(analyzer.config().estimator_k) +
                  "; CSV outputs contain headers only.");
            } else if (analyzer.config().compute_te && !result.te_available) {
              TitanUI::print_warning(
                  label_prefix +
                  ": insufficient effective lagged samples for exact TE; TE "
                  "CSV contains headers only.");
            }
            if (result.degenerate_node_count > 0 ||
                result.degenerate_pair_count > 0) {
              TitanUI::log_kv(
                  log, label_prefix + " Degenerate Inputs",
                  std::to_string(result.degenerate_node_count) + " nodes, " +
                      std::to_string(result.degenerate_pair_count) + " pairs");
            }
            if (result.candidate_pair_count == 0 &&
                result.dense_pair_count > 0) {
              TitanUI::print_warning(
                  label_prefix +
                  ": pair pruning removed all candidate pairs; outputs contain "
                  "headers only.");
            }
          };

      if (args.corr_knn.flag_corr_knn) {
        auto start_corr_knn = std::chrono::steady_clock::now();
        TitanUI::print_module_header("Corr-kNN Information Analysis");
        corr_knn_cached_analyzer.configure(corr_knn_cfg);
        corr_knn_cached_analyzer.run(dccm_analyzer, executor, log);
        corr_knn_cached_analyzer.export_module_outputs(info_output);
        corr_knn_cached = true;
        log_exact_corr_knn_result(
            "Corr-kNN", corr_knn_cached_analyzer,
            info_output.get_corr_knn_summary_filename(),
            info_output.get_corr_knn_mi_pairs_filename(),
            info_output.get_corr_knn_gc_pairs_filename(),
            info_output.get_corr_knn_te_pairs_filename());
        if (dccm_weight_context.use_weights) {
          write_corr_knn_equilibrium_ranking_file(
              info_output.get_corr_knn_equilibrium_ranking_filename(),
              info_output, dccm_weight_context, weight_selection);
          TitanUI::log_kv(log, "Corr-kNN Equilibrium Ranking",
                          info_output.get_corr_knn_equilibrium_ranking_filename());
        }
        const auto time_corr_knn =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_corr_knn);
        TitanUI::log_time(log, "Corr-kNN", time_corr_knn.count() / 1000.0);
      }

      // Allosteric Information Theory (MI -> GC + Transfer Entropy)
      if (args.flag_allostery) {
        auto start_allost = std::chrono::steady_clock::now();
        TitanUI::print_module_header(
            "Allosteric Information Theory (MI, GC + TE)");
        if (args.allostery_estimator == "knn") {
          TitanUI::log_kv(log, "Allostery Backend Route",
                          "knn -> titan_corr_knn_core");
          TitanUI::log_kv(log, "Allostery Histogram Bins",
                          "ignored in knn mode");
          TitanUI::log_kv(log, "Allostery KDE Bandwidth",
                          "ignored in knn mode");
          if (args.allostery_bins > 0) {
            TitanUI::print_warning(
                "Allostery: -allostery-bins is ignored when "
                "-allostery-estimator=knn.");
          }
          if (args.allostery_kde_bandwidth > 0.0) {
            TitanUI::print_warning(
                "Allostery: -allostery-kde-bandwidth is ignored when "
                "-allostery-estimator=knn.");
          }

          const bool reuse_corr_knn =
              corr_knn_cached &&
              TitanCorrKnnAnalyzer::equivalent_config(corr_knn_cfg,
                                                      allostery_knn_cfg);
          TitanCorrKnnAnalyzer allostery_knn_analyzer;
          const TitanCorrKnnAnalyzer *active_analyzer = nullptr;
          if (reuse_corr_knn) {
            corr_knn_cached_analyzer.export_legacy_allostery_outputs(
                info_output.get_network_mi_edges_filename(),
                info_output.get_network_gc_edges_filename(),
                info_output.get_network_te_edges_filename());
            store_allostery_graph_edges_from_corr_knn(dccm_analyzer,
                                                     corr_knn_cached_analyzer);
            TitanUI::log_kv(log, "Allostery kNN Reuse",
                            "reused corr-kNN result");
            active_analyzer = &corr_knn_cached_analyzer;
          } else {
            if (corr_knn_cached) {
              TitanUI::log_kv(log, "Allostery kNN Reuse",
                              "disabled (effective config differs from "
                              "-corr-knn)");
            }
            allostery_knn_analyzer.configure(allostery_knn_cfg);
            allostery_knn_analyzer.run(dccm_analyzer, executor, log);
            allostery_knn_analyzer.export_legacy_allostery_outputs(
                info_output.get_network_mi_edges_filename(),
                info_output.get_network_gc_edges_filename(),
                info_output.get_network_te_edges_filename());
            store_allostery_graph_edges_from_corr_knn(dccm_analyzer,
                                                     allostery_knn_analyzer);
            active_analyzer = &allostery_knn_analyzer;
          }
          log_exact_corr_knn_result(
              "Allostery", *active_analyzer, "",
              info_output.get_network_mi_edges_filename(),
              info_output.get_network_gc_edges_filename(),
              info_output.get_network_te_edges_filename());
        } else {
          DCCMNetwork::AllosteryConfig cfg;
          cfg.tau_lag = args.allostery_tau;
          cfg.num_bins = args.allostery_bins;
          cfg.knn = args.allostery_knn;
          cfg.cutoff_ang = args.allostery_cutoff;
          cfg.use_avg_cutoff = args.allostery_use_avg_cutoff;
          cfg.knn_use_contact_prior = args.allostery_knn_use_contact_prior;
          cfg.knn_contact_cutoff = args.allostery_knn_contact_cutoff;
          cfg.estimator = args.allostery_estimator;
          cfg.kde_bandwidth = args.allostery_kde_bandwidth;
          cfg.compute_te = args.allostery_compute_te;
          if (dccm_analyzer.network_sparse_mode_active()) {
            cfg.force_sparse_candidates = true;
            if (cfg.cutoff_ang <= 0.0)
              cfg.cutoff_ang = args.network_sparse_cut;
            if (cfg.knn <= 0)
              cfg.knn = args.network_sparse_k;
            cfg.use_avg_cutoff = false;
          }
          dccm_analyzer.calculate_allostery_information_optimized(
              executor, cfg, info_output.get_network_gc_edges_filename(),
              info_output.get_network_mi_edges_filename(),
              info_output.get_network_te_edges_filename());
          ensure_csv_header_if_missing(info_output.get_network_mi_edges_filename(),
                                       "Source,Target,MI");
          ensure_csv_header_if_missing(info_output.get_network_gc_edges_filename(),
                                       "Source,Target,GC");
          ensure_csv_header_if_missing(
              info_output.get_network_te_edges_filename(),
              "Source,Target,Net_TE");
          TitanUI::log_kv(log, "Allostery MI File",
                          info_output.get_network_mi_edges_filename());
          TitanUI::log_kv(log, "Allostery GC File",
                          info_output.get_network_gc_edges_filename());
          TitanUI::log_kv(log, "Allostery TE File",
                          info_output.get_network_te_edges_filename());
          TitanUI::log_kv(log, "Allostery TE Stage",
                          cfg.compute_te ? "enabled" : "disabled (GC only)");
          TitanUI::log_kv(log, "Allostery Estimator",
                          cfg.estimator == "kde"
                              ? (std::string("kde (bandwidth=") +
                                 std::to_string(cfg.kde_bandwidth) + ")")
                              : cfg.estimator);
          TitanUI::log_kv(log, "Allostery Prune Window",
                          cfg.use_avg_cutoff ? "union across frames"
                                             : "frame-0 only");
          if (cfg.knn > 0) {
            std::ostringstream knn_prior;
            if (cfg.knn_use_contact_prior) {
              knn_prior << "contact-frequency (Cbeta/N anchors, "
                        << cfg.knn_contact_cutoff << " A)";
            } else {
              knn_prior << "frame-voted nearest";
            }
            TitanUI::log_kv(log, "Allostery kNN Prior", knn_prior.str());
          }
          if (cfg.cutoff_ang > 0.0 || cfg.knn > 0) {
            std::ostringstream pair_filter;
            bool has_filter = false;
            if (cfg.cutoff_ang > 0.0) {
              pair_filter << cfg.cutoff_ang << " A cutoff";
              has_filter = true;
            }
            if (cfg.knn > 0) {
              if (has_filter)
                pair_filter << " + ";
              pair_filter << "top-" << cfg.knn << " "
                          << (cfg.knn_use_contact_prior
                                  ? "contact-frequency kNN @ "
                                  : "frame-voted kNN");
              if (cfg.knn_use_contact_prior)
                pair_filter << cfg.knn_contact_cutoff << " A";
            }
            pair_filter << " ("
                        << (cfg.use_avg_cutoff ? "union across frames"
                                               : "frame-0 only")
                        << ")";
            TitanUI::log_kv(log, "Allostery Pair Filter", pair_filter.str());
            TitanUI::log_kv(
                log, "Allostery Candidate Pairs",
                std::to_string(dccm_analyzer.get_last_allostery_candidate_pairs()) +
                    " / " +
                    std::to_string(dccm_analyzer.get_last_allostery_dense_pairs()));
            if (dccm_analyzer.get_last_allostery_candidate_pairs() == 0 &&
                dccm_analyzer.get_last_allostery_dense_pairs() > 0) {
              TitanUI::print_warning(
                  "Allostery pruning removed all candidate pairs; GC/TE outputs "
                  "contain headers only.");
            }
          }
        }
        auto time_allost =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_allost);
        TitanUI::log_time(log, "Allostery MI+GC+TE",
                          time_allost.count() / 1000.0);
        if (dccm_weight_context.use_weights) {
          write_allostery_equilibrium_ranking_file(
              info_output.get_allostery_equilibrium_ranking_filename(),
              info_output, dccm_weight_context, weight_selection);
          TitanUI::log_kv(
              log, "Allostery Equilibrium Ranking",
              info_output.get_allostery_equilibrium_ranking_filename());
        }
      }

      if (args.flag_pca) {
        const auto &pca_traj_x = dccm_analyzer.get_traj_x();
        const auto &pca_traj_y = dccm_analyzer.get_traj_y();
        const auto &pca_traj_z = dccm_analyzer.get_traj_z();
        const size_t pca_frame_count = pca_traj_x.size();
        std::vector<double> pca_frame_weights;
        if (pca_context_weights && !pca_context_weights->empty()) {
          const auto &global_weights = *pca_context_weights;
          const auto &dccm_global_indices = dccm_analyzer.get_global_frame_indices();
          if (dccm_global_indices.size() != pca_frame_count) {
            TitanUI::print_warning(
                "PCA: DCCM global frame index count (" +
                std::to_string(dccm_global_indices.size()) +
                ") != PCA frame count (" + std::to_string(pca_frame_count) +
                "). Falling back to uniform weights.");
          } else {
            pca_frame_weights.assign(pca_frame_count, 0.0);
            size_t matched_frames = 0;
            size_t weighted_frames = 0;
            double mapped_weight_sum = 0.0;
            for (size_t f = 0; f < dccm_global_indices.size(); ++f) {
              const int64_t global_idx = dccm_global_indices[f];
              if (global_idx < 0 ||
                  static_cast<size_t>(global_idx) >= global_weights.size()) {
                continue;
              }
              const double w = global_weights[static_cast<size_t>(global_idx)];
              pca_frame_weights[f] = w;
              ++matched_frames;
              if (w > 0.0)
                ++weighted_frames;
              mapped_weight_sum += w;
            }
            TitanUI::print_info(
                "PCA: Mapped " + weight_selection.source_name + " weights onto " +
                std::to_string(weighted_frames) + "/" +
                std::to_string(pca_frame_count) +
                " frames (matched indices: " + std::to_string(matched_frames) +
                ", weight sum = " + std::to_string(mapped_weight_sum) + ").");
          }
        }
        TitanPCA::execute_pca_fes_analysis(
            args, pca_traj_x, pca_traj_y, pca_traj_z,
            dccm_analyzer.get_resnames(), dccm_analyzer.get_resnrs(),
            dccm_analyzer.get_chains(), dccm_analyzer.get_is_lig(), executor,
            info_output, log, pca_frame_weights,
            make_pca_weight_context(weight_selection),
            dccm_analyzer.get_traj_times(), dccm_analyzer.get_traj_replicas(),
            replica_traj_files, metrics);
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in DCCM/PCA: " << e.what() << "\n";
  }
}

// ============================================================================
// 5. LSP-MD post-processing (Kornev, JCTC 2025)
// ============================================================================
void run_lspmd_postproc(const arguments &args, DCCMNetwork &dccm_analyzer,
                        TitanExecutor &executor, info &info_output,
                        std::fstream &log,
                        const PostprocEquilibriumWeights &weight_selection,
                        titan_sparse_graph_module::TitanUnifiedGraphRuntime
                            *graph_runtime) {
  if (!args.lspmd.flag_lspmd)
    return;
  if (dccm_analyzer.get_total_frames() == 0) {
    TitanUI::print_warning("LSP-MD: No frames collected, skipping.");
    return;
  }

  try {
    auto start_lspmd = std::chrono::steady_clock::now();
    TitanUI::print_module_header(
        "LSP-MD: Local Spatial Pattern Network Analysis");
    TitanUI::log_kv(log, "LSP-MD Frames",
                    std::to_string(dccm_analyzer.get_total_frames()));
    TitanUI::log_kv(
        log, "LSP-MD Cutoff",
        std::to_string(args.lspmd.lspmd_cutoff) + " A");

    DCCMNetwork::LSPMDConfig cfg;
    cfg.cutoff_ang = args.lspmd.lspmd_cutoff;
    cfg.use_avg_cutoff = args.lspmd.lspmd_use_avg;

    DCCMNetwork::LSPMDResult result;
    dccm_analyzer.calculate_lspmd(executor, cfg, result);
    dccm_analyzer.export_lspmd(result);
    if (graph_runtime) {
      graph_runtime->capture_lspmd(dccm_analyzer, result);
    }
    const auto *pca_context_weights =
        context_global_frame_weights(weight_selection, true);
    if (is_mbar_equilibrium_weight_source(weight_selection) &&
        pca_context_weights) {
      const auto mapped_context = map_global_weights_to_local_frame_context(
          "LSP-MD", dccm_analyzer.get_global_frame_indices(),
          *pca_context_weights);
      if (mapped_context.use_weights) {
        const bool dccm_context_already_written =
            args.flag_network_explicit || args.flag_allostery ||
            args.corr_knn.flag_corr_knn ||
            args.community.flag_community;
        if (!dccm_context_already_written) {
          write_dccm_equilibrium_context_file(
              info_output.get_dccm_equilibrium_context_filename(),
              dccm_analyzer, mapped_context, weight_selection);
          TitanUI::log_kv(log, "DCCM Equilibrium Context",
                          info_output.get_dccm_equilibrium_context_filename());
        }
        write_lspmd_equilibrium_ranking_file(
            info_output.get_lspmd_equilibrium_ranking_filename(),
            dccm_analyzer, result, mapped_context, weight_selection);
        TitanUI::log_kv(log, "LSP-MD Equilibrium Ranking",
                        info_output.get_lspmd_equilibrium_ranking_filename());
      }
    }

    TitanUI::log_kv(log, "LSP-MD Edges",
                    std::to_string(result.edges.size()));
    auto time_lspmd = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_lspmd);
    TitanUI::log_time(log, "LSP-MD Analysis", time_lspmd.count() / 1000.0);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in LSP-MD: " << e.what() << "\n";
  }
}

// ============================================================================
// 6. DHAM post-processing
// ============================================================================
void run_dham_postproc(const arguments &args,
                       std::unique_ptr<TitanMSM> &msm_analyzer,
                       TitanExecutor &executor, info &info_output,
                       std::fstream &log,
                       std::vector<double> &out_frame_weights,
                       Eigen::MatrixXd *out_reweighted_tpm,
                       Eigen::VectorXd *out_reweighted_pi,
                       bool *out_converged,
                       TitanValidation::DhamRebuildContext
                           *out_validation_dham_context) {
  out_frame_weights.clear();
  if (out_reweighted_tpm)
    out_reweighted_tpm->resize(0, 0);
  if (out_reweighted_pi)
    out_reweighted_pi->resize(0);
  if (out_converged)
    *out_converged = false;
  if (out_validation_dham_context)
    *out_validation_dham_context = {};
  if (!args.dham.flag_dham || !msm_analyzer)
    return;

  auto start_dham = std::chrono::steady_clock::now();
  TitanUI::print_module_header("DHAM Post-Processing");
  const bool use_npt_observable = (args.dham.ensemble == "npt");
  if (args.lj.flag_lj) {
    TitanUI::log_kv(log, "DHAM Feature Domain", "LJ phase-2 feature space");
  } else {
    TitanUI::log_kv(log, "DHAM Feature Domain",
                    "protein network -> TICA -> MSM feature space");
  }
  TitanUI::log_kv(log, "DHAM Ensemble", args.dham.ensemble);
  if (use_npt_observable) {
    TitanUI::log_kv(log, "DHAM Thermodynamic Observable",
                    "enthalpy-like reweighting H = column1 + P*V(box)");
    TitanUI::log_kv(log, "DHAM Target Pressure (bar)",
                    std::to_string(args.dham.target_pressure_bar));
    TitanUI::print_info(
        "DHAM: NPT mode enabled. Using H = E_col1 + P*V with P = " +
        std::to_string(args.dham.target_pressure_bar) +
        " bar and V from trajectory box metadata.");
    TitanUI::print_warning(
        "DHAM NPT: for best thermodynamic consistency, the first numeric "
        "column in each .xvg should be configurational/potential energy, not "
        "a mixed observable.");
  } else {
    TitanUI::log_kv(log, "DHAM Thermodynamic Observable",
                    "energy-based NVT/REMD reweighting");
  }

  // 1. Read temperatures
  if (args.dham.temps_file.empty()) {
    TitanUI::print_error("DHAM: -dham-temps not set. Skipping DHAM.");
    return;
  }
  auto temps = TitanDHAM::read_temps(args.dham.temps_file);
  if (temps.empty()) {
    TitanUI::print_error("DHAM: No temperatures read. Skipping.");
    return;
  }

  // 2. Read energy file list
  const bool segment_audit = !args.dham.segment_map_file.empty();
  if (args.dham.ener_list_file.empty() && !segment_audit) {
    TitanUI::print_error("DHAM: -dham-ener not set. Skipping DHAM.");
    return;
  }
  auto ener_paths = segment_audit ? std::vector<std::string>{} : TitanDHAM::read_file_list(args.dham.ener_list_file);
  if (ener_paths.empty() && !segment_audit) {
    TitanUI::print_error("DHAM: No energy files read. Skipping.");
    return;
  }

  const int n_replicas = static_cast<int>(temps.size());
  if (!segment_audit && static_cast<int>(ener_paths.size()) != n_replicas) {
    TitanUI::print_error(
        "DHAM: #temps (" + std::to_string(n_replicas) +
        ") != #energy_files (" + std::to_string(ener_paths.size()) + ").");
    return;
  }

  // Validate n_replicas if explicitly set
  if (args.dham.n_replicas > 0 && args.dham.n_replicas != n_replicas) {
    TitanUI::print_warning(
        "DHAM: --dham-n-replicas=" + std::to_string(args.dham.n_replicas) +
        " != temps file count (" + std::to_string(n_replicas) +
        "). Using temps file count.");
  }

  TitanUI::print_info("DHAM: " + std::to_string(n_replicas) + " replicas, " +
                       "target T = " + std::to_string(args.dham.target_temp) + " K");
  // Fix C: Document the 1-file-per-replica assumption
  TitanUI::print_warning(
      "DHAM: Each input trajectory file is treated as one replica. "
      "Concatenation does not establish physical continuity; REMD inputs "
      "need audited segment/exchange metadata.");

  // 3. Extract per-replica data from MSM
  if (!msm_analyzer->is_clustered()) {
    TitanUI::print_error("DHAM: MSM K-means has not run. "
        "Ensure -msm flag is set and enough frames are collected.");
    return;
  }
  if (!segment_audit && !msm_analyzer->is_built()) {
    TitanUI::print_error(
        "DHAM: MSM transition model has not been built successfully.");
    return;
  }

  const int dham_lag = msm_analyzer->msm_lag_time();
  if (dham_lag <= 0) {
    TitanUI::print_error("DHAM: invalid MSM lag time. Aborting DHAM.");
    return;
  }

  auto per_replica_dtrajs     = msm_analyzer->get_per_replica_dtrajs();
  auto per_replica_timestamps = msm_analyzer->get_per_replica_timestamps();
  auto per_replica_box_volumes = use_npt_observable
      ? msm_analyzer->get_per_replica_box_volumes_nm3()
      : std::unordered_map<int, std::vector<double>>{};

  if (per_replica_dtrajs.empty()) {
    TitanUI::print_error("DHAM: No per-replica discrete trajectories. "
                          "Was replica_id passed to add_projected_frame?");
    return;
  }

  if (segment_audit) {
    try {
      const auto metadata = titan_dham_segments::Metadata::read(
          args.dham.segment_map_file, n_replicas);
      std::vector<titan_dham_segments::CountAudit> audits;
      for (int k = 0; k < n_replicas; ++k) {
        const auto &rep = metadata.replicas[k];
        if (std::abs(rep.temperature - temps[k]) > 1e-6 ||
            (use_npt_observable && std::abs(rep.pressure - args.dham.target_pressure_bar) > 1e-6))
          throw std::invalid_argument("DHAM segment metadata ensemble mismatch");
        const auto &states = per_replica_dtrajs.at(k);
        const auto &times = per_replica_timestamps.at(k);
        if (use_npt_observable) {
          const auto &volumes = per_replica_box_volumes.at(k);
          if (volumes.size() != times.size())
            throw std::invalid_argument("DHAM segment volume dimensions differ");
          for (size_t i = 0; i < times.size(); ++i) {
            const double expected = metadata.match(k, times[i]).volume;
            if (!std::isfinite(volumes[i]) ||
                std::abs(volumes[i] - expected) > std::max(1e-4, 2e-6 * expected))
              throw std::invalid_argument("DHAM segment trajectory volume mismatch");
          }
        }
        audits.push_back(titan_dham_segments::audit_counts(
            metadata, k, states, times, dham_lag, msm_analyzer->n_clusters()));
      }
      std::ofstream counts("dham_segment_counts.csv"), decisions("dham_segment_decisions.csv");
      if (!counts || !decisions) throw std::runtime_error("Cannot write DHAM segment audit");
      counts << "replica,state_from,state_to,candidate_count,physical_rates_validated\n";
      decisions << "replica,decision,windows\n";
      size_t eligible = 0;
      for (int k = 0; k < n_replicas; ++k) {
        for (const auto &[pair, n] : audits[k].counts)
          counts << k << ',' << pair.first << ',' << pair.second << ',' << n << ",false\n";
        for (const auto &[decision, n] : audits[k].decisions) {
          decisions << k << ',' << titan_dham_segments::name(decision) << ',' << n << '\n';
          if (decision == titan_dham_segments::Decision::eligible) eligible += n;
        }
      }
      counts.flush();decisions.flush();
      if (!counts || !decisions) throw std::runtime_error("Incomplete DHAM segment audit output");
      TitanUI::log_kv(log, "DHAM Segment Audit Eligible", std::to_string(eligible));
      TitanUI::log_kv(log, "DHAM Mode", "segment/exchange count audit; estimator not run");
      TitanUI::print_info("DHAM segment/exchange candidate counts written. Exact endpoint energies required; no nearest substitution.");
      TitanUI::print_warning("DHAM segment audit does not correct exchange censoring. No FES, rates, weights or bootstrap validation are produced by this mode.");
    } catch (const std::exception &error) {
      TitanUI::print_error(std::string("DHAM segment audit rejected: ") + error.what());
    }
    return;
  }

  // 4. Parse + align energy .xvg files with optional per-replica
  // window matching against the common trajectory/energy overlap.
  std::unordered_map<int, std::vector<double>> per_replica_energies;
  std::unordered_map<int, std::vector<std::pair<double, double>>> parsed_xvg;
  std::mutex parsed_xvg_mutex;
  const int n_io_threads =
      std::min(static_cast<int>(executor.capacity()), n_replicas);
  const int chunk = (n_replicas + n_io_threads - 1) / n_io_threads;
  executor.parallel_do(static_cast<size_t>(n_io_threads), [&](size_t t_sz) {
    const int t = static_cast<int>(t_sz);
    const int ks = t * chunk;
    const int ke = std::min(ks + chunk, n_replicas);
    if (ks >= ke)
      return;
    for (int k = ks; k < ke; ++k) {
      auto xvg_data = TitanDHAM::parse_xvg(ener_paths[static_cast<size_t>(k)]);
      if (!xvg_data.empty())
        xvg_data = filter_postproc_energy_pairs(xvg_data, args);
      std::lock_guard<std::mutex> lock(parsed_xvg_mutex);
      parsed_xvg[k] = std::move(xvg_data);
    }
  });
  TitanUI::log_kv(log, "DHAM Window Matching", args.dham.window_match);
  if (args.dham.dt_energy_ps > 0.0) {
    TitanUI::print_info("DHAM: Using user-specified dt_energy = " +
                        std::to_string(args.dham.dt_energy_ps) + " ps.");
  } else {
    TitanUI::print_info(
        "DHAM: Auto-detecting dt_energy independently for each energy file.");
  }
  if (args.dham.window_match == "off") {
    TitanUI::print_info(
        "DHAM: Window matching disabled. TITAN will align against the full "
        "trajectory time axis.");
  } else if (args.dham.window_match == "strict") {
    TitanUI::print_info(
        "DHAM: Strict window matching enabled. TITAN will abort if any "
        "replica has a trajectory/energy mismatch.");
  } else {
    TitanUI::print_info(
        "DHAM: Auto window matching enabled. TITAN will trim each replica to "
        "the common trajectory/energy overlap before DHAM.");
  }

  auto subset_by_indices = [](const auto &src,
                              const std::vector<std::size_t> &indices) {
    using value_type = typename std::decay_t<decltype(src)>::value_type;
    std::vector<value_type> out;
    out.reserve(indices.size());
    for (std::size_t idx : indices) {
      if (idx < src.size())
        out.push_back(src[idx]);
    }
    return out;
  };
  auto format_ps = [](double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << value;
    return oss.str();
  };

  std::size_t window_total_original_frames = 0;
  std::size_t window_total_kept_frames = 0;
  std::size_t window_total_dropped_frames = 0;
  std::size_t window_total_unmatched_frames = 0;
  std::size_t window_trimmed_replicas = 0;
  std::size_t window_replicas_without_overlap = 0;
  std::size_t window_missing_energy_files = 0;
  std::size_t window_truncated_start = 0;
  std::size_t window_truncated_end = 0;
  std::size_t window_irregular_dt = 0;
  std::size_t window_sparse_energy = 0;
  bool strict_window_violation = false;
  std::vector<DhamWindowAuditRow> window_audit_rows;
  window_audit_rows.reserve(static_cast<std::size_t>(n_replicas));

  for (int k = 0; k < n_replicas; ++k) {
    auto it_xvg = parsed_xvg.find(k);
    auto it_dtraj = per_replica_dtrajs.find(k);
    auto it_ts = per_replica_timestamps.find(k);
    DhamWindowAuditRow audit;
    audit.replica_id = k;
    audit.traj_present =
        (it_dtraj != per_replica_dtrajs.end() && it_ts != per_replica_timestamps.end());
    audit.energy_present =
        (it_xvg != parsed_xvg.end() && !it_xvg->second.empty());
    if (audit.traj_present) {
      audit.traj_frames = it_dtraj->second.size();
      if (!it_ts->second.empty()) {
        audit.traj_min_ps = it_ts->second.front();
        audit.traj_max_ps = it_ts->second.back();
      }
    }
    if (it_xvg != parsed_xvg.end()) {
      audit.energy_points = it_xvg->second.size();
      if (!it_xvg->second.empty()) {
        audit.energy_min_ps = it_xvg->second.front().first;
        audit.energy_max_ps = it_xvg->second.back().first;
      }
    }
    if (it_xvg == parsed_xvg.end() || it_xvg->second.empty() ||
        it_dtraj == per_replica_dtrajs.end() ||
        it_ts == per_replica_timestamps.end()) {
      if (!audit.energy_present)
        ++window_missing_energy_files;
      audit.status = !audit.traj_present ? "missing_trajectory" :
                     (!audit.energy_present ? "missing_or_empty_energy" :
                                              "incomplete_inputs");
      window_audit_rows.push_back(std::move(audit));
      continue;
    }

    const double dt_energy_ps =
        (args.dham.dt_energy_ps > 0.0) ? args.dham.dt_energy_ps : 0.0;
    if (args.dham.window_match == "off") {
      const auto matched = TitanDHAM::window_match_energy_to_dtraj(
          it_ts->second, it_xvg->second, dt_energy_ps);
      audit.inferred_dt_energy_ps = matched.inferred_dt_energy_ps;
      audit.overlap_found = matched.overlap_found;
      audit.overlap_start_ps = matched.overlap_start_ps;
      audit.overlap_end_ps = matched.overlap_end_ps;
      audit.kept_frames = matched.kept_frames;
      audit.dropped_outside_overlap = matched.dropped_outside_overlap;
      audit.unmatched_in_overlap = matched.unmatched_in_overlap;
      audit.sparse_energy = (audit.energy_points < 2);
      audit.irregular_energy_dt =
          detect_irregular_energy_dt(it_xvg->second, matched.inferred_dt_energy_ps);
      per_replica_energies[k] = TitanDHAM::align_energy_to_dtraj(
          it_ts->second, it_xvg->second, dt_energy_ps);
      audit.status = "legacy_off";
      if (audit.irregular_energy_dt)
        ++window_irregular_dt;
      if (audit.sparse_energy)
        ++window_sparse_energy;
      window_audit_rows.push_back(std::move(audit));
      continue;
    }

    const auto matched = TitanDHAM::window_match_energy_to_dtraj(
        it_ts->second, it_xvg->second, dt_energy_ps);
    audit.inferred_dt_energy_ps = matched.inferred_dt_energy_ps;
    audit.overlap_found = matched.overlap_found;
    audit.overlap_start_ps = matched.overlap_start_ps;
    audit.overlap_end_ps = matched.overlap_end_ps;
    audit.kept_frames = matched.kept_frames;
    audit.dropped_outside_overlap = matched.dropped_outside_overlap;
    audit.unmatched_in_overlap = matched.unmatched_in_overlap;
    const double coverage_tol =
        std::max(1e-6, 2.0 * std::max(1e-6, matched.inferred_dt_energy_ps));
    audit.truncated_start =
        std::isfinite(matched.traj_min_ps) && std::isfinite(matched.energy_min_ps) &&
        matched.energy_min_ps > matched.traj_min_ps + coverage_tol;
    audit.truncated_end =
        std::isfinite(matched.traj_max_ps) && std::isfinite(matched.energy_max_ps) &&
        matched.energy_max_ps < matched.traj_max_ps - coverage_tol;
    audit.sparse_energy = (audit.energy_points < 2);
    audit.irregular_energy_dt =
        detect_irregular_energy_dt(it_xvg->second, matched.inferred_dt_energy_ps);
    window_total_original_frames += matched.original_frames;
    window_total_kept_frames += matched.kept_frames;
    window_total_dropped_frames += matched.dropped_outside_overlap;
    window_total_unmatched_frames += matched.unmatched_in_overlap;
    if (audit.truncated_start)
      ++window_truncated_start;
    if (audit.truncated_end)
      ++window_truncated_end;
    if (audit.irregular_energy_dt)
      ++window_irregular_dt;
    if (audit.sparse_energy)
      ++window_sparse_energy;

    const bool no_usable_overlap =
        !matched.overlap_found || matched.kept_frames == 0;
    const bool has_window_loss =
        matched.dropped_outside_overlap > 0 || matched.unmatched_in_overlap > 0;

    if (no_usable_overlap) {
      ++window_replicas_without_overlap;
      const std::string msg =
          "DHAM: Replica " + std::to_string(k) +
          " has no usable overlap between trajectory [" +
          format_ps(matched.traj_min_ps) + ", " + format_ps(matched.traj_max_ps) +
          "] ps and energy [" + format_ps(matched.energy_min_ps) + ", " +
          format_ps(matched.energy_max_ps) + "] ps.";
      if (args.dham.window_match == "strict") {
        TitanUI::print_warning(msg);
        strict_window_violation = true;
      } else {
        TitanUI::print_warning(msg + " Replica will be ignored by DHAM.");
      }
      audit.status = "no_overlap";
      window_audit_rows.push_back(std::move(audit));
      per_replica_dtrajs.erase(k);
      per_replica_timestamps.erase(k);
      per_replica_energies.erase(k);
      if (use_npt_observable)
        per_replica_box_volumes.erase(k);
      continue;
    }

    if (matched.dropped_outside_overlap > 0)
      ++window_trimmed_replicas;
    if (args.dham.window_match == "strict" && has_window_loss) {
      TitanUI::print_warning(
          "DHAM: Replica " + std::to_string(k) +
          " violated strict window matching. kept=" +
          std::to_string(matched.kept_frames) + "/" +
          std::to_string(matched.original_frames) + ", dropped_outside=" +
          std::to_string(matched.dropped_outside_overlap) +
          ", unmatched_inside=" +
          std::to_string(matched.unmatched_in_overlap) + ".");
      strict_window_violation = true;
      audit.status = "strict_rejected";
      window_audit_rows.push_back(std::move(audit));
      continue;
    }

    if (has_window_loss) {
      TitanUI::print_info(
          "DHAM: Replica " + std::to_string(k) + " window-matched to [" +
          format_ps(matched.overlap_start_ps) + ", " +
          format_ps(matched.overlap_end_ps) + "] ps; kept " +
          std::to_string(matched.kept_frames) + "/" +
          std::to_string(matched.original_frames) + " frames, dropped " +
          std::to_string(matched.dropped_outside_overlap) + ", unmatched " +
          std::to_string(matched.unmatched_in_overlap) + ".");
    }

    if (audit.truncated_start && audit.truncated_end) {
      audit.status = "truncated_both";
    } else if (audit.truncated_start) {
      audit.status = "truncated_start";
    } else if (audit.truncated_end) {
      audit.status = "truncated_end";
    } else if (audit.irregular_energy_dt) {
      audit.status = "irregular_dt";
    } else if (audit.sparse_energy) {
      audit.status = "sparse_energy";
    } else if (has_window_loss) {
      audit.status = "trimmed_to_overlap";
    } else {
      audit.status = "ok";
    }

    per_replica_dtrajs[k] =
        subset_by_indices(it_dtraj->second, matched.kept_indices);
    per_replica_timestamps[k] = matched.kept_dtraj_times_ps;
    per_replica_energies[k] = matched.aligned_values;
    if (use_npt_observable) {
      auto it_volume = per_replica_box_volumes.find(k);
      if (it_volume != per_replica_box_volumes.end()) {
        per_replica_box_volumes[k] =
            subset_by_indices(it_volume->second, matched.kept_indices);
      }
    }
    window_audit_rows.push_back(std::move(audit));
  }

  if (args.dham.window_match != "off") {
    TitanUI::log_kv(log, "DHAM Window Frames Kept",
                    std::to_string(window_total_kept_frames) + "/" +
                        std::to_string(window_total_original_frames));
    TitanUI::log_kv(log, "DHAM Window Frames Dropped",
                    std::to_string(window_total_dropped_frames));
    TitanUI::log_kv(log, "DHAM Window Frames Unmatched",
                    std::to_string(window_total_unmatched_frames));
    TitanUI::log_kv(log, "DHAM Window Replicas Trimmed",
                    std::to_string(window_trimmed_replicas));
    TitanUI::log_kv(log, "DHAM Window Replicas No Overlap",
                    std::to_string(window_replicas_without_overlap));
    if (window_total_dropped_frames > 0 || window_total_unmatched_frames > 0) {
      TitanUI::print_info(
        "DHAM: Window matching kept " +
          std::to_string(window_total_kept_frames) + "/" +
          std::to_string(window_total_original_frames) +
          " trajectory frames across replicas.");
    }
  }
  TitanUI::log_kv(log, "DHAM Window Energy Missing",
                  std::to_string(window_missing_energy_files));
  TitanUI::log_kv(log, "DHAM Window Truncated Start",
                  std::to_string(window_truncated_start));
  TitanUI::log_kv(log, "DHAM Window Truncated End",
                  std::to_string(window_truncated_end));
  TitanUI::log_kv(log, "DHAM Window Irregular dt",
                  std::to_string(window_irregular_dt));
  TitanUI::log_kv(log, "DHAM Window Sparse Energy",
                  std::to_string(window_sparse_energy));
  write_dham_window_audit_file(info_output.get_dham_window_audit_filename(),
                               window_audit_rows);
  TitanUI::print_info("DHAM: Exported window audit to " +
                      info_output.get_dham_window_audit_filename());
  if (strict_window_violation) {
    TitanUI::log_kv(log, "DHAM Converged", "no");
    TitanUI::log_kv(log, "DHAM Abort Reason",
                    "strict window matching rejected one or more replicas");
    TitanUI::print_warning(
        "DHAM: strict window matching failed. Skipping DHAM reweighting.");
    return;
  }
  std::size_t post_match_candidate_transitions = 0;
  for (const auto &[replica_id, dtraj] : per_replica_dtrajs) {
    (void)replica_id;
    if (dtraj.size() > static_cast<std::size_t>(dham_lag)) {
      post_match_candidate_transitions +=
          dtraj.size() - static_cast<std::size_t>(dham_lag);
    }
  }
  TitanUI::log_kv(log, "DHAM Window Candidate Transitions",
                  std::to_string(post_match_candidate_transitions));
  if (post_match_candidate_transitions == 0) {
    TitanUI::log_kv(log, "DHAM Converged", "no");
    TitanUI::log_kv(log, "DHAM Abort Reason",
                    "zero lag-valid transitions after window matching");
    TitanUI::print_warning(
        "DHAM: Window-matched data contain no lag-valid transitions for the "
        "current MSM lag. Merge matching trajectory/energy windows, reduce "
        "-msm-lag, or disable automatic trimming with -dham-window-match off.");
    return;
  }

  std::unordered_map<int, std::vector<double>> per_replica_observables;
  if (use_npt_observable) {
    const double pv_factor =
        args.dham.target_pressure_bar * kBarNm3ToKJPerMol;
    size_t replicas_missing_volume = 0;
    size_t volume_size_mismatches = 0;
    size_t frames_missing_valid_volume = 0;
    size_t frames_combined = 0;

    for (int k = 0; k < n_replicas; ++k) {
      auto it_energy = per_replica_energies.find(k);
      if (it_energy == per_replica_energies.end())
        continue;

      auto it_volume = per_replica_box_volumes.find(k);
      if (it_volume == per_replica_box_volumes.end()) {
        ++replicas_missing_volume;
        continue;
      }

      const auto &energies = it_energy->second;
      const auto &volumes_nm3 = it_volume->second;
      if (volumes_nm3.size() != energies.size()) {
        ++volume_size_mismatches;
      }

      std::vector<double> observable(
          energies.size(), std::numeric_limits<double>::quiet_NaN());
      const size_t n_frames = std::min(energies.size(), volumes_nm3.size());
      for (size_t t = 0; t < n_frames; ++t) {
        const double e = energies[t];
        const double v = volumes_nm3[t];
        if (!std::isfinite(e) || !std::isfinite(v) || v < 0.0) {
          ++frames_missing_valid_volume;
          continue;
        }
        observable[t] = e + pv_factor * v;
        ++frames_combined;
      }
      per_replica_observables[k] = std::move(observable);
    }

    if (replicas_missing_volume > 0) {
      TitanUI::print_error(
          "DHAM NPT: Missing per-frame box volumes for " +
          std::to_string(replicas_missing_volume) + "/" +
          std::to_string(n_replicas) +
          " replicas. Re-run with the current TITAN build so TICA/MSM caches "
          "trajectory volume metadata.");
      return;
    }
    if (frames_combined == 0) {
      TitanUI::print_error(
          "DHAM NPT: No valid E(t) + P*V(t) samples were constructed from the "
          "aligned energy traces and trajectory box metadata.");
      return;
    }
    if (volume_size_mismatches > 0) {
      TitanUI::print_warning(
          "DHAM NPT: " + std::to_string(volume_size_mismatches) +
          " replica volume series had a different length than the aligned "
          "energy series; truncation to the common prefix was applied.");
    }
    if (frames_missing_valid_volume > 0) {
      TitanUI::print_warning(
          "DHAM NPT: " + std::to_string(frames_missing_valid_volume) +
          " aligned frames had invalid energy/volume data and were excluded "
          "from the enthalpy-like observable.");
    }
    TitanUI::print_info(
        "DHAM NPT: Constructed " + std::to_string(frames_combined) +
        " frame-level H = E_col1 + P*V samples.");
  } else {
    per_replica_observables = per_replica_energies;
  }
  if (per_replica_observables.empty()) {
    TitanUI::log_kv(log, "DHAM Converged", "no");
    TitanUI::log_kv(log, "DHAM Abort Reason",
                    "no usable aligned observables remained after window matching");
    TitanUI::print_warning(
        "DHAM: No usable aligned thermodynamic observables remain after "
        "window matching.");
    return;
  }

  if (out_validation_dham_context) {
    out_validation_dham_context->available = true;
    out_validation_dham_context->n_states = msm_analyzer->n_clusters();
    out_validation_dham_context->lag_frames = dham_lag;
    out_validation_dham_context->n_replicas = n_replicas;
    out_validation_dham_context->temps_k = temps;
    out_validation_dham_context->target_temp_k = args.dham.target_temp;
    out_validation_dham_context->max_iter = args.dham.max_iter;
    out_validation_dham_context->tolerance = args.dham.tolerance;
    out_validation_dham_context->ensemble = args.dham.ensemble;
    out_validation_dham_context->per_replica_dtrajs = per_replica_dtrajs;
    out_validation_dham_context->per_replica_timestamps_ps =
        per_replica_timestamps;
    out_validation_dham_context->per_replica_observables =
        per_replica_observables;
  }

  // 5. Determine number of microstates from MSM
  const int n_states = msm_analyzer->n_clusters();

  // 6. Compute per-state mean thermodynamic observables (Opt 4: multi-threaded)
  auto state_energies = TitanDHAM::compute_state_energies(
      per_replica_dtrajs, per_replica_observables, n_states, n_replicas,
      static_cast<int>(executor.capacity()));

  // 7. Create TitanDHAM and load data
  TitanDHAM dham(n_states, n_replicas, temps, args.dham.target_temp);
  std::vector<std::vector<unsigned char>> valid_energy(
      static_cast<size_t>(n_replicas),
      std::vector<unsigned char>(static_cast<size_t>(n_states), 0));
  size_t valid_state_energy_pairs = 0;

  // Load state energies (skip NaN = unvisited state/replica pairs)
  for (int k = 0; k < n_replicas; ++k) {
    for (int i = 0; i < n_states; ++i) {
      double e = state_energies(i, k);
      if (std::isfinite(e)) {
        valid_energy[static_cast<size_t>(k)][static_cast<size_t>(i)] = 1;
        ++valid_state_energy_pairs;
        dham.set_state_energy(k, i, e);
      }
    }
  }
  const size_t total_state_energy_pairs =
      static_cast<size_t>(n_states) * static_cast<size_t>(n_replicas);
  const size_t missing_state_energy_pairs =
      total_state_energy_pairs - valid_state_energy_pairs;
  if (missing_state_energy_pairs > 0) {
    TitanUI::print_warning(
        "DHAM: Missing aligned thermodynamic observables for " +
        std::to_string(missing_state_energy_pairs) + "/" +
        std::to_string(total_state_energy_pairs) +
        " state/replica pairs. Transitions touching those states will be skipped.");
  }

  // Build count matrices from per-replica dtrajs (use same lag as MSM)
  // Skip transitions that cross trajectory breaks or states with missing energy.
  TitanUI::print_info("DHAM: Using MSM lag = " + std::to_string(dham_lag) +
                        " frames for count matrix construction.");
  size_t total_counts = 0;
  size_t skipped_breaks = 0;
  size_t skipped_missing_energy = 0;
  size_t skipped_invalid_states = 0;
  for (int k = 0; k < n_replicas; ++k) {
    auto it = per_replica_dtrajs.find(k);
    if (it == per_replica_dtrajs.end()) continue;
    const auto &dtraj = it->second;
    auto it_ts = per_replica_timestamps.find(k);
    const bool have_times = (it_ts != per_replica_timestamps.end() &&
                              it_ts->second.size() == dtraj.size());
    // Estimate expected gap = lag * dt_frame
    double dt_frame = 1.0, expected_gap = dham_lag;
    if (have_times && it_ts->second.size() > 1) {
      dt_frame = it_ts->second[1] - it_ts->second[0];
      expected_gap = dham_lag * dt_frame;
    }
    for (size_t t = 0; t + static_cast<size_t>(dham_lag) < dtraj.size(); ++t) {
      if (have_times) {
        double actual_gap = it_ts->second[t + static_cast<size_t>(dham_lag)]
                          - it_ts->second[t];
        if (std::abs(actual_gap - expected_gap) > 0.1 * std::abs(expected_gap)) {
          ++skipped_breaks;
          continue; // trajectory break — skip phantom transition
        }
      }
      const int state_i = dtraj[t];
      const int state_j = dtraj[t + static_cast<size_t>(dham_lag)];
      if (state_i < 0 || state_i >= n_states || state_j < 0 || state_j >= n_states) {
        ++skipped_invalid_states;
        continue;
      }
      if (!valid_energy[static_cast<size_t>(k)][static_cast<size_t>(state_i)] ||
          !valid_energy[static_cast<size_t>(k)][static_cast<size_t>(state_j)]) {
        ++skipped_missing_energy;
        continue;
      }
      dham.accumulate_counts(k, state_i, state_j);
      ++total_counts;
    }
  }
  if (skipped_breaks > 0) {
    TitanUI::print_warning("DHAM: Skipped " + std::to_string(skipped_breaks) +
        " transitions across trajectory breaks (kept " +
        std::to_string(total_counts) + ").");
  }
  if (skipped_missing_energy > 0) {
    TitanUI::print_warning("DHAM: Skipped " +
        std::to_string(skipped_missing_energy) +
        " transitions involving states without aligned thermodynamic observables.");
  }
  if (skipped_invalid_states > 0) {
    TitanUI::print_warning("DHAM: Skipped " +
        std::to_string(skipped_invalid_states) +
        " transitions with invalid microstate indices.");
  }

  // 8. Solve DHAM
  bool converged = dham.solve(args.dham.max_iter, args.dham.tolerance);
  TitanUI::log_kv(log, "DHAM Converged", converged ? "yes" : "no");
  if (out_converged)
    *out_converged = converged;
  if (converged) {
    if (out_reweighted_pi)
      *out_reweighted_pi = dham.get_stationary_distribution();
    if (out_reweighted_tpm)
      *out_reweighted_tpm = dham.get_unbiased_transition_matrix();
    auto per_replica_weights = dham.get_frame_weights(per_replica_dtrajs);
    const auto &frame_replica_ids = msm_analyzer->get_frame_replica_ids();
    const auto &frame_global_indices = msm_analyzer->get_frame_global_indices();

    if (frame_replica_ids.size() != frame_global_indices.size()) {
      TitanUI::print_warning(
          "DHAM: projected frame metadata mismatch: replica_ids=" +
          std::to_string(frame_replica_ids.size()) + ", global_indices=" +
          std::to_string(frame_global_indices.size()) +
          ". Some frame weights may be dropped.");
    }

    int64_t max_global_index = -1;
    for (int64_t global_idx : frame_global_indices) {
      if (global_idx > max_global_index)
        max_global_index = global_idx;
    }
    if (max_global_index >= 0) {
      out_frame_weights.assign(static_cast<size_t>(max_global_index) + 1, 0.0);
    }

    std::unordered_map<int, size_t> per_replica_offsets;
    per_replica_offsets.reserve(per_replica_weights.size());
    std::vector<unsigned char> assigned(out_frame_weights.size(), 0);
    const size_t tracked_projected_frames =
        std::min(frame_replica_ids.size(), frame_global_indices.size());
    size_t mapped_frames = 0;
    size_t dropped_missing_replica_weights = 0;
    size_t dropped_invalid_global_index = 0;
    size_t duplicate_global_indices = 0;

    for (size_t frame_idx = 0; frame_idx < tracked_projected_frames; ++frame_idx) {
      const int replica_id = frame_replica_ids[frame_idx];
      auto it_weights = per_replica_weights.find(replica_id);
      size_t &replica_offset = per_replica_offsets[replica_id];
      if (it_weights == per_replica_weights.end() ||
          replica_offset >= it_weights->second.size()) {
        ++dropped_missing_replica_weights;
        continue;
      }

      const int64_t global_idx = frame_global_indices[frame_idx];
      if (global_idx < 0 ||
          static_cast<size_t>(global_idx) >= out_frame_weights.size()) {
        ++dropped_invalid_global_index;
        ++replica_offset;
        continue;
      }

      const double w = it_weights->second[replica_offset++];
      const size_t global_slot = static_cast<size_t>(global_idx);
      if (assigned[global_slot]) {
        ++duplicate_global_indices;
      }
      out_frame_weights[global_slot] = w;
      assigned[global_slot] = 1;
      ++mapped_frames;
    }

    double weight_sum = 0.0;
    for (double w : out_frame_weights)
      weight_sum += w;
    TitanUI::print_info(
        "DHAM: Computed " + std::to_string(mapped_frames) +
        " global frame weights for reweighted FES.");
    TitanUI::print_info("DHAM: Weight sum = " + std::to_string(weight_sum) +
                        " (expected ~1.0)");
    if (dropped_missing_replica_weights > 0) {
      TitanUI::print_warning(
          "DHAM: Dropped " +
          std::to_string(dropped_missing_replica_weights) +
          " projected frames while rebuilding global weights.");
    }
    if (dropped_invalid_global_index > 0) {
      TitanUI::print_warning(
          "DHAM: Dropped " + std::to_string(dropped_invalid_global_index) +
          " projected frames with invalid global indices.");
    }
    if (duplicate_global_indices > 0) {
      TitanUI::print_warning(
          "DHAM: Encountered " + std::to_string(duplicate_global_indices) +
          " duplicate global frame indices while assigning weights.");
    }

    // 9. Export results
    dham.write_results(info_output.get_dham_pi_filename(),
                       info_output.get_dham_fes_filename(),
                       info_output.get_dham_tpm_filename(), log,
                       args.eu_par);
  } else {
    TitanUI::print_warning(
        "DHAM: solve() did not converge - results were not written and "
        "PCA/TICA FES will use uniform weights. Increase --dham-max-iter "
        "or check the aligned energy data.");
  }

  auto time_dham = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_dham);
  TitanUI::log_time(log, "DHAM Analysis", time_dham.count() / 1000.0);
}

TitanMBAR::CoreResult
run_mbar_postproc(const arguments &args,
                  std::unique_ptr<TitanMSM> &msm_analyzer,
                  DCCMNetwork *dccm_analyzer, TitanExecutor &executor,
                  info &info_output, std::fstream &log,
                  std::vector<double> *out_frame_weights,
                  std::vector<double> *out_pca_frame_weights) {
  TitanMBAR::CoreResult mbar_result;
  if (out_frame_weights)
    out_frame_weights->clear();
  if (out_pca_frame_weights)
    out_pca_frame_weights->clear();
  if (!args.mbar.flag_mbar)
    return mbar_result;

  auto start_mbar = std::chrono::steady_clock::now();
  TitanUI::print_module_header("MBAR Reweighting");

  const bool want_tica =
      (args.mbar.domain == "tica" || args.mbar.domain == "both");
  const bool want_pca =
      (args.mbar.domain == "pca" || args.mbar.domain == "both");

  auto make_tica_metadata = [&]() {
    TitanMBAR::FrameMetadata metadata;
    if (msm_analyzer && msm_analyzer->n_frames() > 0) {
      metadata.replica_ids = msm_analyzer->get_frame_replica_ids();
      metadata.timestamps_ps = msm_analyzer->get_frame_timestamps();
      metadata.global_indices = msm_analyzer->get_frame_global_indices();
      metadata.box_volumes_nm3 = msm_analyzer->get_frame_box_volumes_nm3();
    }
    return metadata;
  };
  auto make_pca_metadata = [&]() {
    TitanMBAR::FrameMetadata metadata;
    if (dccm_analyzer && dccm_analyzer->get_total_frames() > 0) {
      metadata.replica_ids = dccm_analyzer->get_traj_replicas();
      metadata.timestamps_ps = dccm_analyzer->get_traj_times();
      metadata.global_indices = dccm_analyzer->get_global_frame_indices();
      metadata.box_volumes_nm3 = dccm_analyzer->get_traj_box_volumes_nm3();
    }
    return metadata;
  };

  const TitanMBAR::FrameMetadata tica_metadata = make_tica_metadata();
  const TitanMBAR::FrameMetadata pca_metadata = make_pca_metadata();
  const bool have_tica_metadata = !tica_metadata.empty();
  const bool have_pca_metadata = !pca_metadata.empty();

  TitanMBAR::FrameMetadata primary_metadata;
  std::string primary_metadata_context{"energy-only"};
  if (want_tica && have_tica_metadata) {
    primary_metadata = tica_metadata;
    primary_metadata_context = "tica";
  } else if (want_pca && have_pca_metadata) {
    primary_metadata = pca_metadata;
    primary_metadata_context = "pca";
    if (want_tica) {
      TitanUI::print_warning(
          "MBAR: mbar-domain=both requested but no TICA/MSM metadata is "
          "available. Solving with PCA/DCCM metadata; weighted TICA exports "
          "will be unavailable unless TICA projections are present.");
    }
  }
  TitanUI::log_kv(log, "MBAR Primary Metadata", primary_metadata_context);

  mbar_result =
      TitanMBAR::run_temperature_ladder_reweighting(args, primary_metadata,
                                                    executor, log);

  if (!mbar_result.global_frame_weights.empty()) {
    if (primary_metadata_context == "tica") {
      mbar_result.tica_global_frame_weights = mbar_result.global_frame_weights;
      mbar_result.mapped_tica_global_frames = mbar_result.mapped_global_frames;
      if (want_pca && have_pca_metadata) {
        const auto remap = TitanMBAR::remap_frame_weights_to_metadata(
            mbar_result.frame_weights, pca_metadata);
        mbar_result.pca_global_frame_weights = remap.global_frame_weights;
        mbar_result.mapped_pca_global_frames = remap.mapped_global_frames;
      }
    } else if (primary_metadata_context == "pca") {
      mbar_result.pca_global_frame_weights = mbar_result.global_frame_weights;
      mbar_result.mapped_pca_global_frames = mbar_result.mapped_global_frames;
      if (want_tica && have_tica_metadata) {
        const auto remap = TitanMBAR::remap_frame_weights_to_metadata(
            mbar_result.frame_weights, tica_metadata);
        mbar_result.tica_global_frame_weights = remap.global_frame_weights;
        mbar_result.mapped_tica_global_frames = remap.mapped_global_frames;
      }
    }
  }

  if (mbar_result.mapped_tica_global_frames > 0) {
    TitanUI::log_kv(log, "MBAR Mapped TICA Frames",
                    std::to_string(mbar_result.mapped_tica_global_frames));
  }
  if (mbar_result.mapped_pca_global_frames > 0) {
    TitanUI::log_kv(log, "MBAR Mapped PCA Frames",
                    std::to_string(mbar_result.mapped_pca_global_frames));
  }

  TitanMBAR::write_window_audit_file(info_output.get_mbar_window_audit_filename(),
                                     mbar_result.window_audit);
  TitanMBAR::write_frame_weights_file(
      info_output.get_mbar_frame_weights_filename(), mbar_result.frame_weights);
  TitanMBAR::write_summary_file(info_output.get_mbar_summary_filename(),
                                mbar_result, args, log);
  if (args.mbar.scan_points > 1 && !mbar_result.cv_scan.empty()) {
    TitanMBAR::write_cv_file(info_output.get_mbar_cv_filename(),
                             mbar_result.cv_scan);
    TitanUI::log_kv(log, "MBAR Cv File", info_output.get_mbar_cv_filename());
  }
  TitanUI::log_kv(log, "MBAR Weight File",
                  info_output.get_mbar_frame_weights_filename());
  TitanUI::log_kv(log, "MBAR Window Audit File",
                  info_output.get_mbar_window_audit_filename());

  if (!mbar_result.solved) {
    auto time_mbar = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_mbar);
    TitanUI::log_time(log, "MBAR Analysis", time_mbar.count() / 1000.0);
    return mbar_result;
  }
  if (!mbar_result.converged) {
    TitanUI::print_warning(
        "MBAR: solver did not converge. Skipping MBAR-weighted downstream "
        "FES exports.");
    auto time_mbar = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_mbar);
    TitanUI::log_time(log, "MBAR Analysis", time_mbar.count() / 1000.0);
    return mbar_result;
  }
  if (out_frame_weights) {
    if (!mbar_result.tica_global_frame_weights.empty()) {
      *out_frame_weights = mbar_result.tica_global_frame_weights;
    } else {
      *out_frame_weights = mbar_result.global_frame_weights;
    }
  }
  if (out_pca_frame_weights) {
    *out_pca_frame_weights = mbar_result.pca_global_frame_weights;
  }

  auto time_mbar = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_mbar);
  TitanUI::log_time(log, "MBAR Analysis", time_mbar.count() / 1000.0);
  return mbar_result;
}

void run_mbar_projection_exports_postproc(
    const arguments &args, std::unique_ptr<TitanMSM> &msm_analyzer,
    DCCMNetwork *dccm_analyzer, TitanExecutor &executor, info &info_output,
    std::fstream &log, const TitanMBAR::CoreResult &mbar_result) {
  if (!args.mbar.flag_mbar || !mbar_result.solved || !mbar_result.converged)
    return;

  const bool want_tica =
      (args.mbar.domain == "tica" || args.mbar.domain == "both");
  const bool want_pca =
      (args.mbar.domain == "pca" || args.mbar.domain == "both");
  const std::vector<double> *tica_weights =
      !mbar_result.tica_global_frame_weights.empty()
          ? &mbar_result.tica_global_frame_weights
          : (args.mbar.domain == "tica" ? &mbar_result.global_frame_weights
                                         : nullptr);
  const std::vector<double> *pca_weights =
      !mbar_result.pca_global_frame_weights.empty()
          ? &mbar_result.pca_global_frame_weights
          : (args.mbar.domain == "pca" ? &mbar_result.global_frame_weights
                                        : nullptr);

  if (want_tica) {
    if (!(msm_analyzer && msm_analyzer->n_frames() > 0 &&
          msm_analyzer->proj_mat_ref().cols() >= 2)) {
      TitanUI::print_warning(
          "MBAR/TICA: no valid TICA projection metadata available. "
          "Skipping weighted TICA FES.");
    } else if (!tica_weights || tica_weights->empty()) {
      TitanUI::print_warning(
          "MBAR/TICA: no mapped global frame weights available. "
          "Skipping weighted TICA FES.");
    } else {
      bool use_weights = false;
      const auto local_weights = map_global_weights_to_local_indices(
          "MBAR/TICA", msm_analyzer->get_frame_global_indices(),
          *tica_weights, use_weights);
      if (use_weights) {
        std::vector<std::pair<double, double>> projections(
            static_cast<std::size_t>(msm_analyzer->n_frames()));
        for (std::size_t frame = 0; frame < projections.size(); ++frame) {
          projections[frame] = {
              msm_analyzer->proj_mat_ref()(static_cast<Eigen::Index>(frame), 0),
              msm_analyzer->proj_mat_ref()(static_cast<Eigen::Index>(frame), 1)};
        }
        TitanPCA::export_projection_fes_analysis(
            "MBAR TICA", "TICA", args, projections, local_weights, true,
            args.mbar.target_temp, "MBAR-weighted", executor, log,
            info_output.get_mbar_tica_fes_2d_filename(),
            info_output.get_mbar_tica_cluster_representatives_filename(),
            msm_analyzer->get_frame_timestamps(),
            msm_analyzer->get_frame_replica_ids());
      }
    }
  }

  if (want_pca) {
    if (!(dccm_analyzer && dccm_analyzer->get_total_frames() > 0)) {
      TitanUI::print_warning(
          "MBAR/PCA: no PCA trajectory metadata available. "
          "Skipping weighted PCA FES.");
    } else if (!pca_weights || pca_weights->empty()) {
      TitanUI::print_warning(
          "MBAR/PCA: no mapped global frame weights available. "
          "Skipping weighted PCA FES.");
    } else {
      const auto projections =
          read_projection_csv_pairs(info_output.get_pca_fel_filename());
      if (projections.size() != dccm_analyzer->get_total_frames()) {
        TitanUI::print_warning(
            "MBAR/PCA: projection count in " +
            info_output.get_pca_fel_filename() + " (" +
            std::to_string(projections.size()) +
            ") does not match PCA frame count (" +
            std::to_string(dccm_analyzer->get_total_frames()) +
            "). Skipping weighted PCA FES.");
      } else {
        bool use_weights = false;
        const auto local_weights = map_global_weights_to_local_indices(
            "MBAR/PCA", dccm_analyzer->get_global_frame_indices(),
            *pca_weights, use_weights);
        if (use_weights) {
          TitanPCA::export_projection_fes_analysis(
              "MBAR PCA", "PCA", args, projections, local_weights, true,
              args.mbar.target_temp, "MBAR-weighted", executor, log,
              info_output.get_mbar_pca_fes_2d_filename(),
              info_output.get_mbar_pca_cluster_representatives_filename(),
              dccm_analyzer->get_traj_times(),
              dccm_analyzer->get_traj_replicas());
        }
      }
    }
  }
}

void run_tpt_postproc(const arguments &args,
                      std::unique_ptr<TitanMSM> &msm_analyzer,
                      TitanExecutor &executor, info &info_output,
                      std::fstream &log,
                      const std::vector<std::string> &lj_feature_files,
                      const TitanRMSAnalyzer *rms_analyzer,
                      const TitanDSSPAnalyzer *protein_dssp_analyzer,
                      const TitanQNativeAnalyzer *qnative_analyzer,
                      const Eigen::MatrixXd *dham_tpm,
                      const Eigen::VectorXd *dham_pi,
                      bool dham_converged,
                      const PostprocEquilibriumWeights *weight_selection,
                      TitanValidation::RuntimeCache *validation_cache,
                      titan_sparse_graph_module::TitanUnifiedGraphRuntime
                          *graph_runtime) {
  if (!args.tpt.flag_tpt || !msm_analyzer)
    return;

  if (!msm_analyzer->is_built()) {
    TitanUI::print_warning(
        "TPT: MSM has not been built successfully. Skipping TPT.");
    return;
  }

  auto start_tpt = std::chrono::steady_clock::now();
  TitanUI::print_module_header("Transition Path Theory");
  TitanUI::log_kv(log, "TPT Mode", args.tpt.mode);
  if (args.tpt.mode == "kinetic") {
    TitanUI::print_warning(
        "TPT kinetic mode assumes the MSM came from continuous trajectories "
        "at the target ensemble. TITAN will report kinetic diagnostics, "
        "but does not certify kinetic validity automatically.");
  }
  if (args.dham.flag_dham && !args.tpt.use_dham) {
    TitanUI::print_warning(
        "TPT: DHAM is available but -tpt-use-dham is off, so TITAN will use the built MSM model.");
  }

  std::vector<int> macro_assignments;
  macro_assignments =
      msm_analyzer->compute_pcca_macrostates(args.msm.msm_macrostates);
  if (!macro_assignments.empty()) {
    TitanUI::log_kv(
        log, "TPT Macrostate Count",
        std::to_string(msm_analyzer->effective_pcca_macrostates(
            args.msm.msm_macrostates)));
  }

  const LjMicrostateSummary lj_summary =
      compute_lj_microstate_summary(*msm_analyzer, lj_feature_files, log);
  const ProteinMicrostateSummary protein_summary =
      compute_protein_microstate_summary(*msm_analyzer, rms_analyzer,
                                         protein_dssp_analyzer, log);
  const QNativeMicrostateSummary qnative_summary =
      compute_qnative_microstate_summary(*msm_analyzer, qnative_analyzer, log);

  try {
    const bool kinetic_mode = (args.tpt.mode == "kinetic");
    if (kinetic_mode && args.tpt.use_dham && args.dham.flag_dham) {
      TitanUI::print_warning(
          "TPT kinetic mode ignores DHAM-reweighted TPM/pi and uses the built MSM model from the continuous trajectories.");
    }
    if (!kinetic_mode && !args.tpt.demux_map_file.empty()) {
      TitanUI::print_warning(
          "TPT: -tpt-demux-map is only used in kinetic mode. Ensemble TPT will ignore it.");
    }

    const bool can_use_dham =
        !kinetic_mode && args.tpt.use_dham && args.dham.flag_dham &&
        dham_converged &&
        dham_tpm && dham_pi && dham_tpm->rows() == msm_analyzer->n_clusters() &&
        dham_tpm->cols() == msm_analyzer->n_clusters() &&
        dham_pi->size() == msm_analyzer->n_clusters();

    Eigen::MatrixXd demux_tpm;
    Eigen::VectorXd demux_pi;
    std::vector<std::vector<int>> kinetic_segments;
    std::int64_t demux_transition_count = 0;
    std::string demux_error;
    const Eigen::MatrixXd *model_tpm_ptr = &msm_analyzer->tpm();
    const Eigen::VectorXd *model_pi_ptr = &msm_analyzer->stationary_dist();
    std::string model_source = "msm_built_model";

    if (kinetic_mode) {
      model_source = "msm_built_model_kinetic";
      if (!args.tpt.demux_map_file.empty()) {
        std::vector<std::vector<int>> demux_rows;
        if (!read_tpt_demux_map(args.tpt.demux_map_file, demux_rows,
                                demux_error)) {
          TitanUI::print_error("TPT kinetic demux: " + demux_error);
          return;
        }
        if (!build_demux_state_segments(*msm_analyzer, demux_rows,
                                        kinetic_segments, demux_error)) {
          TitanUI::print_error("TPT kinetic demux: " + demux_error);
          return;
        }
        if (!build_tpt_model_from_segments(
                msm_analyzer->n_clusters(), msm_analyzer->msm_lag_time(),
                kinetic_segments, false, demux_tpm, demux_pi,
                demux_transition_count, demux_error)) {
          TitanUI::print_error("TPT kinetic demux: " + demux_error);
          return;
        }
        model_tpm_ptr = &demux_tpm;
        model_pi_ptr = &demux_pi;
        model_source = "msm_demux_kinetic";
        TitanUI::print_info("TPT kinetic demux: rebuilt kinetic MSM from " +
                            std::to_string(kinetic_segments.size()) +
                            " demuxed segment(s) using " +
                            std::to_string(demux_transition_count) +
                            " lag-valid transitions.");
        TitanUI::log_kv(log, "TPT Demux Map", args.tpt.demux_map_file);
        TitanUI::log_kv(log, "TPT Demux Segments",
                        std::to_string(kinetic_segments.size()));
        TitanUI::log_kv(log, "TPT Demux Transitions",
                        std::to_string(demux_transition_count));
      } else {
        kinetic_segments =
            TitanValidation::build_msm_state_segments(*msm_analyzer);
      }
    } else if (can_use_dham) {
      model_tpm_ptr = dham_tpm;
      model_pi_ptr = dham_pi;
      model_source = "dham_reweighted_" + args.dham.ensemble + "_target";
    }

    if (!kinetic_mode && args.tpt.use_dham && args.dham.flag_dham &&
        !can_use_dham) {
      TitanUI::print_warning(
          "TPT: requested DHAM-backed model, but no converged DHAM TPM/pi is available. Falling back to MSM.");
    }
    TitanUI::log_kv(log, "TPT Model Source", model_source);

    std::vector<int> source_states;
    std::vector<int> sink_states;
    std::string parse_error;
    std::string source_label;
    std::string sink_label;
    if (!resolve_tpt_basin_spec(args.tpt.source_spec, *msm_analyzer,
                                macro_assignments, lj_summary, protein_summary,
                                qnative_summary, *model_pi_ptr, source_states,
                                source_label, parse_error)) {
      TitanUI::print_error("TPT: invalid -tpt-source: " + parse_error);
      return;
    }
    if (!resolve_tpt_basin_spec(args.tpt.sink_spec, *msm_analyzer,
                                macro_assignments, lj_summary, protein_summary,
                                qnative_summary, *model_pi_ptr, sink_states,
                                sink_label, parse_error)) {
      TitanUI::print_error("TPT: invalid -tpt-sink: " + parse_error);
      return;
    }

    TitanTPT analyzer(*model_tpm_ptr, *model_pi_ptr);
    analyzer.compute(source_states, sink_states, args.tpt.top_paths,
                     args.tpt.flux_cut, &executor,
                     static_cast<int>(executor.capacity()));
    analyzer.write_results(info_output.get_tpt_summary_filename(),
                           info_output.get_tpt_committor_filename(),
                           info_output.get_tpt_flux_edges_filename(),
                           info_output.get_tpt_paths_filename(), log,
                           args.tpt.mode,
                           (!kinetic_mode && args.dham.flag_dham),
                           source_label,
                           sink_label, model_source);
    write_tpt_state_labels(info_output.get_tpt_state_labels_filename(), analyzer,
                           *msm_analyzer, macro_assignments, lj_summary,
                           protein_summary, qnative_summary,
                           analyzer.stationary_dist());
    const auto *tica_context_weights =
        weight_selection ? context_global_frame_weights(*weight_selection, false)
                         : nullptr;
    if (weight_selection && weight_selection->use_weights &&
        weight_selection->resolved_source == "mbar" && tica_context_weights) {
      EquilibriumPopulationSummary equilibrium_summary;
      if (compute_equilibrium_populations(*msm_analyzer,
                                          *tica_context_weights,
                                          macro_assignments,
                                          equilibrium_summary)) {
        write_tpt_equilibrium_overlay_file(
            info_output.get_tpt_equilibrium_overlay_filename(), analyzer,
            macro_assignments, equilibrium_summary, *weight_selection);
        TitanUI::log_kv(log, "TPT Equilibrium Overlay",
                        info_output.get_tpt_equilibrium_overlay_filename());
        TitanUI::print_info("TPT: Exported equilibrium overlay to " +
                            info_output.get_tpt_equilibrium_overlay_filename());
      } else {
        TitanUI::print_warning(
            "TPT: unable to map selected equilibrium frame weights onto "
            "microstates. Skipping equilibrium overlay.");
      }
    }
    const bool wrote_reactive_fes = write_tpt_reactive_fes_2d(
        info_output.get_tpt_reactive_fes_2d_filename(), analyzer,
        *msm_analyzer, log);
    const bool wrote_flux_map = write_tpt_flux_map_2d(
        info_output.get_tpt_flux_map_2d_filename(), analyzer, *msm_analyzer);
    const std::size_t n_path_clusters = write_tpt_path_clusters_file(
        info_output.get_tpt_path_clusters_filename(), analyzer);
    const TptPathSupport bottleneck_support =
        accumulate_tpt_path_support(analyzer, executor);
    const auto bottleneck_edge_rows =
        compute_tpt_bottleneck_edges(analyzer, macro_assignments,
                                     bottleneck_support);
    const auto bottleneck_state_rows =
        compute_tpt_bottleneck_states(analyzer, macro_assignments,
                                      bottleneck_support);
    const std::size_t n_bottleneck_edges = write_tpt_bottleneck_edges_file(
        info_output.get_tpt_bottleneck_edges_filename(),
        bottleneck_edge_rows);
    const std::size_t n_bottleneck_states = write_tpt_bottleneck_states_file(
        info_output.get_tpt_bottleneck_states_filename(),
        bottleneck_state_rows);
    append_tpt_bottleneck_summary(
        info_output.get_tpt_summary_filename(), bottleneck_edge_rows,
        bottleneck_state_rows);
    TitanUI::log_kv(log, "TPT Path Clusters",
                    std::to_string(n_path_clusters));
    TitanUI::log_kv(log, "TPT Bottleneck Edges",
                    std::to_string(n_bottleneck_edges));
    TitanUI::log_kv(log, "TPT Bottleneck States",
                    std::to_string(n_bottleneck_states));
    TitanUI::print_info("TPT: Exported path clusters to " +
                        info_output.get_tpt_path_clusters_filename());
    TitanUI::print_info("TPT: Exported bottleneck edges to " +
                        info_output.get_tpt_bottleneck_edges_filename());
    TitanUI::print_info("TPT: Exported bottleneck states to " +
                        info_output.get_tpt_bottleneck_states_filename());
    if (graph_runtime) {
      std::vector<titan_sparse_graph_module::TptCapturedEdgeRow>
          graph_edge_rows;
      graph_edge_rows.reserve(bottleneck_edge_rows.size());
      for (const auto &row : bottleneck_edge_rows) {
        graph_edge_rows.push_back(to_graph_tpt_edge_row(row));
      }
      std::vector<titan_sparse_graph_module::TptCapturedNodeRow>
          graph_node_rows;
      graph_node_rows.reserve(bottleneck_state_rows.size());
      for (const auto &row : bottleneck_state_rows) {
        graph_node_rows.push_back(to_graph_tpt_node_row(row));
      }
      graph_runtime->capture_tpt(analyzer, macro_assignments, graph_edge_rows,
                                 graph_node_rows);
    }

    if (!macro_assignments.empty()) {
      Eigen::MatrixXd macro_tpm;
      Eigen::VectorXd macro_pi;
      std::string macro_error;
      if (build_macro_tpt_model(*model_tpm_ptr, *model_pi_ptr, macro_assignments,
                                macro_tpm, macro_pi, macro_error)) {
        const std::size_t n_pairs = write_tpt_all_pairs_kinetics_file(
            info_output.get_tpt_all_pairs_kinetics_filename(), macro_tpm,
            macro_pi, msm_analyzer->lag_time_ns());
        TitanUI::log_kv(log, "TPT All-Pairs Macro Kinetics",
                        std::to_string(n_pairs));
        TitanUI::print_info("TPT: Exported all-pairs macro kinetics to " +
                            info_output.get_tpt_all_pairs_kinetics_filename());
      } else {
        TitanUI::print_warning("TPT all-pairs kinetics: " + macro_error);
      }
    }
    if (kinetic_mode) {
      if (kinetic_segments.empty()) {
        kinetic_segments =
            TitanValidation::build_msm_state_segments(*msm_analyzer);
      }
      const TitanValidation::TptKineticDiagnostics kinetic_diag =
          TitanValidation::compute_tpt_kinetic_diagnostics_from_segments(
              kinetic_segments, msm_analyzer->n_clusters(),
              msm_analyzer->msm_lag_time(), msm_analyzer->lag_time_ns(),
              analyzer);
      write_tpt_kinetics_file(info_output.get_tpt_kinetics_filename(),
                              kinetic_diag, model_source);
      TitanUI::log_kv(log, "TPT Kinetic Suitability",
                      kinetic_diag.suitability);
      TitanUI::log_kv(log, "TPT Observed A->B Events",
                      std::to_string(kinetic_diag.observed_a_to_b_events));
      TitanUI::log_kv(log, "TPT Observed B->A Events",
                      std::to_string(kinetic_diag.observed_b_to_a_events));
      TitanUI::log_kv(log, "TPT Observed Round Trips",
                      std::to_string(kinetic_diag.observed_round_trips));
      TitanUI::log_kv(log, "TPT Model MFPT A->B (ns)",
                      std::to_string(kinetic_diag.model_mfpt_ab_ns));
      TitanUI::log_kv(log, "TPT CK / Markov Sanity",
                      kinetic_diag.markov_sanity);
      TitanUI::print_info("TPT: Exported kinetic diagnostics to " +
                          info_output.get_tpt_kinetics_filename());
      if (kinetic_diag.observed_b_to_a_events == 0) {
        TitanUI::print_warning(
            "TPT kinetic diagnostics: no observed B->A back-transitions. Treat rates as exploratory only.");
      } else if (kinetic_diag.observed_round_trips < 3) {
        TitanUI::print_warning(
            "TPT kinetic diagnostics: fewer than 3 observed round-trips. Treat MFPT/rates as exploratory.");
      }
      if (kinetic_diag.markov_sanity == "fail") {
        TitanUI::print_warning(
            "TPT kinetic diagnostics: CK / Markov consistency is poor. Treat kinetic rates as weak.");
      } else if (kinetic_diag.markov_sanity == "weak") {
        TitanUI::print_warning(
            "TPT kinetic diagnostics: CK / Markov consistency is only moderate.");
      }
    }
    if (validation_cache) {
      validation_cache->tpt_context.available = true;
      validation_cache->tpt_context.kinetic_mode = kinetic_mode;
      validation_cache->tpt_context.symmetrize_bootstrap = !kinetic_mode;
      validation_cache->tpt_context.n_states = msm_analyzer->n_clusters();
      validation_cache->tpt_context.lag_frames = msm_analyzer->msm_lag_time();
      validation_cache->tpt_context.top_paths = args.tpt.top_paths;
      validation_cache->tpt_context.flux_cut = args.tpt.flux_cut;
      validation_cache->tpt_context.lag_time_ns = msm_analyzer->lag_time_ns();
      validation_cache->tpt_context.model_source = model_source;
      validation_cache->tpt_context.source_states = source_states;
      validation_cache->tpt_context.sink_states = sink_states;
      validation_cache->tpt_context.model_tpm = *model_tpm_ptr;
      validation_cache->tpt_context.model_pi = *model_pi_ptr;
      validation_cache->tpt_context.bootstrap_segments =
          kinetic_mode
              ? kinetic_segments
              : TitanValidation::build_msm_state_segments(*msm_analyzer);
    }
    const bool defer_legacy_bootstrap =
        validation_cache &&
        TitanValidation::validation_bootstrap_matches_tpt_legacy(args);
    if (args.tpt.bootstrap_samples > 0 && !defer_legacy_bootstrap) {
      if (model_source.rfind("dham_reweighted_", 0) == 0) {
        TitanValidation::TptBootstrapAggregate bootstrap;
        std::string bootstrap_error;
        if (TitanValidation::run_dham_tpt_bootstrap(
                validation_cache ? validation_cache->dham_context
                                 : TitanValidation::DhamRebuildContext{},
                args.tpt.top_paths, args.tpt.flux_cut, source_states,
                sink_states, args.tpt.bootstrap_samples, args.tpt.bootstrap_seed,
                static_cast<int>(executor.capacity()), executor, bootstrap,
                bootstrap_error)) {
          write_tpt_bootstrap_files(
              info_output.get_tpt_bootstrap_summary_filename(),
              info_output.get_tpt_bootstrap_states_filename(), bootstrap,
              analyzer, model_source);
          TitanUI::log_kv(log, "TPT Bootstrap Requested",
                          std::to_string(args.tpt.bootstrap_samples));
          TitanUI::log_kv(log, "TPT Bootstrap Successful",
                          std::to_string(bootstrap.successful_samples));
          TitanUI::print_info(
              "TPT: Exported bootstrap summary to " +
              info_output.get_tpt_bootstrap_summary_filename());
          TitanUI::print_info(
              "TPT: Exported bootstrap state stats to " +
              info_output.get_tpt_bootstrap_states_filename());
        } else {
          TitanUI::print_warning("TPT bootstrap: " + bootstrap_error);
        }
      } else {
        std::vector<std::vector<int>> bootstrap_segments =
            kinetic_mode ? kinetic_segments
                         : TitanValidation::build_msm_state_segments(
                               *msm_analyzer);
        const bool symmetrize_bootstrap = !kinetic_mode;
        TitanValidation::TptBootstrapAggregate bootstrap;
        std::string bootstrap_error;
        if (TitanValidation::run_tpt_bootstrap(
                bootstrap_segments, msm_analyzer->n_clusters(),
                msm_analyzer->msm_lag_time(), msm_analyzer->lag_time_ns(),
                symmetrize_bootstrap, source_states, sink_states,
                args.tpt.top_paths, args.tpt.flux_cut,
                static_cast<int>(executor.capacity()),
                args.tpt.bootstrap_samples, args.tpt.bootstrap_seed, executor,
                bootstrap, bootstrap_error)) {
          write_tpt_bootstrap_files(
              info_output.get_tpt_bootstrap_summary_filename(),
              info_output.get_tpt_bootstrap_states_filename(), bootstrap,
              analyzer, model_source);
          TitanUI::log_kv(log, "TPT Bootstrap Requested",
                          std::to_string(args.tpt.bootstrap_samples));
          TitanUI::log_kv(log, "TPT Bootstrap Successful",
                          std::to_string(bootstrap.successful_samples));
          TitanUI::print_info(
              "TPT: Exported bootstrap summary to " +
              info_output.get_tpt_bootstrap_summary_filename());
          TitanUI::print_info(
              "TPT: Exported bootstrap state stats to " +
              info_output.get_tpt_bootstrap_states_filename());
        } else {
          TitanUI::print_warning("TPT bootstrap: " + bootstrap_error);
        }
      }
    } else if (defer_legacy_bootstrap) {
      TitanUI::log_kv(log, "TPT Bootstrap Reuse",
                      "validation core will write legacy bootstrap files");
    }
    TitanUI::log_kv(log, "TPT Source Basin", source_label);
    TitanUI::log_kv(log, "TPT Sink Basin", sink_label);
    TitanUI::print_info("TPT: Exported summary to " +
                        info_output.get_tpt_summary_filename());
    TitanUI::print_info("TPT: Exported committors to " +
                        info_output.get_tpt_committor_filename());
    TitanUI::print_info("TPT: Exported flux edges to " +
                        info_output.get_tpt_flux_edges_filename());
    TitanUI::print_info("TPT: Exported dominant paths to " +
                        info_output.get_tpt_paths_filename());
    TitanUI::print_info("TPT: Exported state labels to " +
                        info_output.get_tpt_state_labels_filename());
    if (wrote_reactive_fes) {
      TitanUI::print_info("TPT: Exported reactive FES to " +
                          info_output.get_tpt_reactive_fes_2d_filename());
    }
    if (wrote_flux_map) {
      TitanUI::print_info("TPT: Exported flux map to " +
                          info_output.get_tpt_flux_map_2d_filename());
    }
  } catch (const std::exception &e) {
    TitanUI::print_error(std::string("TPT: ") + e.what());
    return;
  }

  auto time_tpt = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_tpt);
  TitanUI::log_time(log, "TPT Analysis", time_tpt.count() / 1000.0);
}

void run_validation_postproc(const arguments &args,
                             std::unique_ptr<TitanKinetics> &tica_analyzer,
                             std::unique_ptr<TitanMSM> &msm_analyzer,
                             TitanExecutor &executor, info &info_output,
                             std::fstream &log,
                             TitanValidation::RuntimeCache &validation_cache) {
  if (!args.validation.flag_validation || !tica_analyzer || !msm_analyzer)
    return;
  TitanValidation::run_postproc(args, *tica_analyzer, msm_analyzer, executor,
                                info_output, log, validation_cache);
}

namespace {

struct AwhFrameAssignmentRow {
  std::size_t bin_id{0};
  std::int64_t global_frame_index{-1};
  double rc_left{0.0};
  double rc_center{0.0};
  double rc_right{0.0};
  std::string overlap_status;
  std::string stability_status;
};

struct TptCommittorRow {
  int state{-1};
  char label{'I'};
  double q_plus{std::numeric_limits<double>::quiet_NaN()};
  double q_minus{std::numeric_limits<double>::quiet_NaN()};
  double reactive_density{std::numeric_limits<double>::quiet_NaN()};
  double net_out{std::numeric_limits<double>::quiet_NaN()};
};

std::vector<AwhFrameAssignmentRow>
read_awh_frame_assignments(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open())
    return {};

  std::vector<AwhFrameAssignmentRow> rows;
  std::string line;
  bool first_line = true;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    if (first_line) {
      first_line = false;
      if (line.find("bin_id") != std::string::npos)
        continue;
    }
    std::stringstream ss(line);
    AwhFrameAssignmentRow row;
    int replica_id = -1;
    if (!(ss >> row.bin_id >> replica_id >> row.global_frame_index))
      continue;
    double time_ps = 0.0;
    double rc_value = 0.0;
    if (!(ss >> time_ps >> rc_value >> row.rc_left >> row.rc_center >> row.rc_right >> row.overlap_status >> row.stability_status))
      continue;
    rows.push_back(row);
  }
  return rows;
}

std::unordered_map<int, TptCommittorRow>
read_tpt_committor_rows(const std::string &path) {
  std::ifstream in(path);
  if (!in.is_open())
    return {};

  std::unordered_map<int, TptCommittorRow> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::stringstream ss(line);
    TptCommittorRow row;
    if (!(ss >> row.state >> row.label))
      continue;
    double pi = 0.0;
    double gross_out = 0.0;
    double gross_in = 0.0;
    double net_in = 0.0;
    if (!(ss >> pi >> row.q_plus >> row.q_minus >> row.reactive_density >> gross_out >> gross_in >> row.net_out >> net_in))
      continue;
    rows[row.state] = row;
  }
  return rows;
}

} // namespace

void run_awh_exploratory_postproc(const arguments &args,
                                  std::unique_ptr<TitanMSM> &msm_analyzer,
                                  std::unique_ptr<TitanUMAP> &umap_analyzer,
                                  DCCMNetwork *dccm_analyzer,
                                  info &info_output, std::fstream &log) {
  if (!args.awh.flag_awh_ifp)
    return;

  const auto awh_rows =
      read_awh_frame_assignments(info_output.get_awh_frame_assignments_filename());
  if (awh_rows.empty()) {
    TitanUI::print_warning(
        "AWH exploratory context: no frame assignments found. Skipping overlay export.");
    return;
  }

  std::unordered_map<std::size_t, std::vector<std::int64_t>> bin_to_globals;
  std::unordered_map<std::size_t, AwhFrameAssignmentRow> bin_meta;
  for (const auto &row : awh_rows) {
    if (row.global_frame_index < 0)
      continue;
    bin_to_globals[row.bin_id].push_back(row.global_frame_index);
    bin_meta[row.bin_id] = row;
  }

  const auto pca_pairs = read_projection_csv_pairs(info_output.get_pca_fel_filename());
  const auto tpt_rows = read_tpt_committor_rows(info_output.get_tpt_committor_filename());
  const auto &umap_matrix = (umap_analyzer && umap_analyzer->is_fitted())
                                ? umap_analyzer->embedding()
                                : TitanUMAP::EmbeddingMatrixRM{};

  std::ofstream out(info_output.get_awh_exploratory_context_filename());
  if (!out.is_open()) {
    TitanUI::print_warning(
        "AWH exploratory context: cannot open output file for writing.");
    return;
  }

  out << "# exploratory_only biased_ensemble_context\n";
  out << "# Do not interpret these overlays as unbiased kinetics or equilibrium populations.\n";
  out << "bin_id\trc_left\trc_center\trc_right\tmatched_frames\tpca_frames\t"
         "pca_pc1_mean\tpca_pc2_mean\tumap_frames\tumap1_mean\tumap2_mean\t"
         "microstate_frames\tdominant_microstate\tdominant_microstate_fraction\t"
         "macrostate_frames\tdominant_macrostate\tdominant_macrostate_fraction\t"
         "tpt_frames\tmean_q_plus\tmean_q_minus\tmean_reactive_density\t"
         "mean_net_out\thas_source_like\thas_sink_like\thas_intermediate_like\t"
         "overlap_status\tstability_status\tnote\n";
  out << std::fixed << std::setprecision(6);

  std::vector<int> macro_assignments;
  if (msm_analyzer && msm_analyzer->is_built()) {
    macro_assignments =
        msm_analyzer->compute_pcca_macrostates(args.msm.msm_macrostates);
  }

  for (const auto &[bin_id, globals] : bin_to_globals) {
    const auto meta_it = bin_meta.find(bin_id);
    if (meta_it == bin_meta.end())
      continue;
    const auto &meta = meta_it->second;

    double pca_pc1_sum = 0.0, pca_pc2_sum = 0.0;
    std::size_t pca_count = 0;
    double umap1_sum = 0.0, umap2_sum = 0.0;
    std::size_t umap_count = 0;
    std::unordered_map<int, std::size_t> micro_counts;
    std::unordered_map<int, std::size_t> macro_counts;
    double q_plus_sum = 0.0, q_minus_sum = 0.0, reactive_sum = 0.0, net_out_sum = 0.0;
    std::size_t tpt_count = 0;
    bool has_source = false, has_sink = false, has_intermediate = false;

    for (std::int64_t global_idx : globals) {
      if (global_idx >= 0 && static_cast<std::size_t>(global_idx) < pca_pairs.size()) {
        pca_pc1_sum += pca_pairs[static_cast<std::size_t>(global_idx)].first;
        pca_pc2_sum += pca_pairs[static_cast<std::size_t>(global_idx)].second;
        ++pca_count;
      }
      if (msm_analyzer && msm_analyzer->n_frames() > 0) {
        const auto &frame_globals = msm_analyzer->get_frame_global_indices();
        auto found = std::find(frame_globals.begin(), frame_globals.end(), global_idx);
        if (found != frame_globals.end()) {
          const auto local_idx = static_cast<std::size_t>(std::distance(frame_globals.begin(), found));
          if (umap_analyzer && umap_analyzer->is_fitted() && local_idx < static_cast<std::size_t>(umap_matrix.rows()) && umap_matrix.cols() >= 2) {
            umap1_sum += umap_matrix(static_cast<Eigen::Index>(local_idx), 0);
            umap2_sum += umap_matrix(static_cast<Eigen::Index>(local_idx), 1);
            ++umap_count;
          }
          if (local_idx < msm_analyzer->discrete_traj().size()) {
            const int micro = msm_analyzer->discrete_traj()[local_idx];
            ++micro_counts[micro];
            if (!macro_assignments.empty() && micro >= 0 && micro < static_cast<int>(macro_assignments.size())) {
              ++macro_counts[macro_assignments[static_cast<std::size_t>(micro)]];
            }
            auto tpt_it = tpt_rows.find(micro);
            if (tpt_it != tpt_rows.end()) {
              q_plus_sum += tpt_it->second.q_plus;
              q_minus_sum += tpt_it->second.q_minus;
              reactive_sum += tpt_it->second.reactive_density;
              net_out_sum += tpt_it->second.net_out;
              ++tpt_count;
              has_source = has_source || tpt_it->second.label == 'A';
              has_sink = has_sink || tpt_it->second.label == 'B';
              has_intermediate = has_intermediate || tpt_it->second.label == 'I';
            }
          }
        }
      }
    }

    auto dominant_state = [](const auto &counts) {
      int state = -1;
      std::size_t n = 0;
      for (const auto &[key, value] : counts) {
        if (value > n) {
          state = key;
          n = value;
        }
      }
      return std::pair<int, std::size_t>{state, n};
    };

    const auto [dominant_micro, dominant_micro_n] = dominant_state(micro_counts);
    const auto [dominant_macro, dominant_macro_n] = dominant_state(macro_counts);

    out << bin_id << '\t' << meta.rc_left << '\t' << meta.rc_center << '\t'
        << meta.rc_right << '\t' << globals.size() << '\t' << pca_count << '\t'
        << (pca_count > 0 ? pca_pc1_sum / static_cast<double>(pca_count) : 0.0) << '\t'
        << (pca_count > 0 ? pca_pc2_sum / static_cast<double>(pca_count) : 0.0) << '\t'
        << umap_count << '\t'
        << (umap_count > 0 ? umap1_sum / static_cast<double>(umap_count) : 0.0) << '\t'
        << (umap_count > 0 ? umap2_sum / static_cast<double>(umap_count) : 0.0) << '\t'
        << micro_counts.size() << '\t' << dominant_micro << '\t'
        << (globals.empty() ? 0.0 : static_cast<double>(dominant_micro_n) /
                                   static_cast<double>(globals.size()))
        << '\t' << macro_counts.size() << '\t' << dominant_macro << '\t'
        << (globals.empty() ? 0.0 : static_cast<double>(dominant_macro_n) /
                                   static_cast<double>(globals.size()))
        << '\t' << tpt_count << '\t'
        << (tpt_count > 0 ? q_plus_sum / static_cast<double>(tpt_count) : 0.0)
        << '\t'
        << (tpt_count > 0 ? q_minus_sum / static_cast<double>(tpt_count) : 0.0)
        << '\t'
        << (tpt_count > 0 ? reactive_sum / static_cast<double>(tpt_count) : 0.0)
        << '\t'
        << (tpt_count > 0 ? net_out_sum / static_cast<double>(tpt_count) : 0.0)
        << '\t' << (has_source ? 1 : 0) << '\t' << (has_sink ? 1 : 0) << '\t'
        << (has_intermediate ? 1 : 0) << '\t' << meta.overlap_status << '\t'
        << meta.stability_status << '\t'
        << "exploratory_context_only"
        << '\n';
  }

  TitanUI::log_kv(log, "AWH Exploratory Bins",
                  std::to_string(bin_to_globals.size()));
  TitanUI::log_kv(log, "AWH Exploratory Output",
                  info_output.get_awh_exploratory_context_filename());
  TitanUI::print_info(
      "AWH exploratory context: exported biased-ensemble overlay to " +
      info_output.get_awh_exploratory_context_filename());
  TitanUI::print_warning(
      "AWH exploratory context remains exploratory only; do not interpret it as unbiased kinetics or equilibrium populations.");
}

