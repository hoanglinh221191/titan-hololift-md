#include "postproc/postproc_common.h"

using namespace postproc_detail;

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
