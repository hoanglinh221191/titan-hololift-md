#include "postproc/postproc_common.h"

namespace postproc_detail {

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
  // KIN-23: every token must be a whole integer replica id; the old
  // `iss >> int` loop stopped silently at the first other token, so a
  // GROMACS replica_index.xvg row "0.000 0 1 2 3" became the single id 0
  const auto where = [&path](int at) {
    return "demux map " + path + ":" + std::to_string(at);
  };
  while (std::getline(in, line)) {
    ++line_no;
    const auto hash_pos = line.find('#');
    if (hash_pos != std::string::npos)
      line.erase(hash_pos);
    std::replace(line.begin(), line.end(), ',', ' ');
    line = trim_copy(line);
    if (line.empty() || line.front() == '@')
      continue; // blank, comment or xvg directive

    std::istringstream iss(line);
    std::vector<int> row;
    std::string token;
    while (iss >> token) {
      long value = 0;
      std::size_t used = 0;
      bool ok = true;
      try {
        value = std::stol(token, &used, 10);
      } catch (const std::exception &) {
        ok = false;
      }
      if (!ok || used != token.size() || value < -1 ||
          value > std::numeric_limits<int>::max()) {
        error_message =
            where(line_no) + ": '" + token +
            "' is not a replica id (an integer >= -1). -tpt-demux-map rows "
            "hold one replica id per segment, without a time column";
        return false;
      }
      row.push_back(static_cast<int>(value));
    }
    if (expected_cols == 0) {
      expected_cols = row.size();
    } else if (row.size() != expected_cols) {
      error_message = where(line_no) + ": " + std::to_string(row.size()) +
                      " columns, the first data row has " +
                      std::to_string(expected_cols);
      return false;
    }
    rows_out.push_back(std::move(row));
  }

  if (rows_out.empty()) {
    error_message = "demux map " + path + " has no data rows";
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

} // namespace postproc_detail
