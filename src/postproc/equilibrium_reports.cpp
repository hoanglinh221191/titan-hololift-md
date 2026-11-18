#include "postproc/postproc_common.h"

namespace postproc_detail {

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
         "overlap_end_ps inferred_dt_energy_ps energy_coarser_than_traj "
         "frames_without_own_energy_row inferred_dt_traj_ps\n";
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
        << row.inferred_dt_energy_ps << ' '
        << (row.energy_coarser_than_traj ? 1 : 0) << ' '
        << row.duplicate_energy_matches << ' ' << row.inferred_dt_traj_ps
        << '\n';
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
                                  int embedding_dims) {
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

} // namespace postproc_detail
