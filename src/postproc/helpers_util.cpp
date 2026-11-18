#include "postproc/postproc_common.h"

namespace postproc_detail {

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
  TitanUI::log_time(log, "TICA Dense Eigensolve",
                    stats.dense_eigensolve_seconds);
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

} // namespace postproc_detail
