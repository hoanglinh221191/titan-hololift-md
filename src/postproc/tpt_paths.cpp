#include "postproc/postproc_common.h"

namespace postproc_detail {

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
cluster_tpt_paths(const TitanTPT &tpt_analyzer, double similarity_cut) {
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
    std::size_t top_n) {
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

} // namespace postproc_detail
