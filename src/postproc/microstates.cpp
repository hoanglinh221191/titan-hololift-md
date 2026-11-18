#include "postproc/postproc_common.h"

namespace postproc_detail {

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

} // namespace postproc_detail
