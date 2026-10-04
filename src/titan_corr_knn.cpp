#include "titan_corr_knn.h"

#include "titan_corr_knn_shared.h"
#include "titan_executor.h"
#include "titan_ui.h"

#include <algorithm>
#include <string>

namespace {

bool has_complete_rows(const titan_corr_knn::TrajectoryRows &x_rows,
                       const titan_corr_knn::TrajectoryRows &y_rows,
                       const titan_corr_knn::TrajectoryRows &z_rows,
                       std::size_t node_count, std::size_t frame_count) {
  if (x_rows.size() < frame_count || y_rows.size() < frame_count ||
      z_rows.size() < frame_count) {
    return false;
  }
  for (std::size_t frame = 0; frame < frame_count; ++frame) {
    if (x_rows[frame].size() < node_count || y_rows[frame].size() < node_count ||
        z_rows[frame].size() < node_count) {
      return false;
    }
  }
  return true;
}

} // namespace

bool TitanCorrKnnAnalyzer::run(const DCCMNetwork &dccm_analyzer,
                               TitanExecutor &executor, std::fstream &log) {
  (void)log;
  result_ = {};
  const auto &traj_x = dccm_analyzer.get_traj_x();
  const auto &traj_y = dccm_analyzer.get_traj_y();
  const auto &traj_z = dccm_analyzer.get_traj_z();
  const std::size_t total_frames = dccm_analyzer.get_total_frames();
  const std::size_t node_count = dccm_analyzer.get_resnrs().size();

  const auto node_labels = titan_corr_knn::build_node_labels(
      dccm_analyzer.get_resnames(), dccm_analyzer.get_resnrs(),
      dccm_analyzer.get_chains());

  bool use_contact_anchor_cache = false;
  if (config_.pair_pruning.pair_use_contact_prior) {
    use_contact_anchor_cache = has_complete_rows(
        dccm_analyzer.get_traj_cbx(), dccm_analyzer.get_traj_cby(),
        dccm_analyzer.get_traj_cbz(), node_count, total_frames);
    if (!use_contact_anchor_cache) {
      TitanUI::print_warning(
          "Corr-kNN: contact-frequency pair prior is falling back to Calpha "
          "anchors because Cbeta/N trajectory cache is unavailable.");
      TitanUI::log_message(
          log,
          "Corr-kNN: contact-frequency pair prior fell back to Calpha anchors.");
    }
  }

  const auto &anchor_x =
      use_contact_anchor_cache ? dccm_analyzer.get_traj_cbx() : traj_x;
  const auto &anchor_y =
      use_contact_anchor_cache ? dccm_analyzer.get_traj_cby() : traj_y;
  const auto &anchor_z =
      use_contact_anchor_cache ? dccm_analyzer.get_traj_cbz() : traj_z;

  std::size_t dense_pair_count = 0;
  const auto candidate_pairs = titan_corr_knn::build_candidate_pairs(
      traj_x, traj_y, traj_z, anchor_x, anchor_y, anchor_z, node_count,
      total_frames, config_.pair_pruning, &dense_pair_count);

  titan_corr_knn::ExactEstimatorConfig estimator_cfg;
  estimator_cfg.estimator_k = config_.estimator_k;
  estimator_cfg.tau_lag = config_.tau_lag;
  estimator_cfg.compute_te = config_.compute_te;
  result_ = titan_corr_knn::compute_exact_scores(
      traj_x, traj_y, traj_z, node_labels, candidate_pairs, dense_pair_count,
      estimator_cfg, executor);
  return true;
}

void TitanCorrKnnAnalyzer::export_module_outputs(const info &info_output) const {
  titan_corr_knn::write_summary(info_output.get_corr_knn_summary_filename(),
                                result_, config_.pair_pruning, "corr_knn");
  titan_corr_knn::write_raw_mi_csv(info_output.get_corr_knn_mi_pairs_filename(),
                                   result_);
  titan_corr_knn::write_raw_gc_csv(info_output.get_corr_knn_gc_pairs_filename(),
                                   result_);
  titan_corr_knn::write_raw_te_csv(info_output.get_corr_knn_te_pairs_filename(),
                                   result_);
}

void TitanCorrKnnAnalyzer::export_legacy_allostery_outputs(
    const std::string &mi_path, const std::string &gc_path,
    const std::string &te_path) const {
  titan_corr_knn::write_legacy_allostery_exports(mi_path, gc_path, te_path,
                                                 result_);
}

TitanCorrKnnAnalyzer::Config
TitanCorrKnnAnalyzer::from_corr_knn_args(const CorrKnnArgs &args) {
  Config cfg;
  cfg.estimator_k = args.estimator_k;
  cfg.tau_lag = args.tau_lag;
  cfg.compute_te = args.compute_te;
  cfg.pair_pruning.pair_topk = args.pair_topk;
  cfg.pair_pruning.pair_cutoff_ang = args.pair_cutoff_ang;
  cfg.pair_pruning.use_avg_cutoff = args.use_avg_cutoff;
  cfg.pair_pruning.pair_use_contact_prior = args.pair_use_contact_prior;
  cfg.pair_pruning.pair_contact_cutoff_ang = args.pair_contact_cutoff_ang;
  return cfg;
}

TitanCorrKnnAnalyzer::Config
TitanCorrKnnAnalyzer::from_allostery_args(const arguments &args) {
  Config cfg;
  cfg.estimator_k = args.allostery_estimator_k;
  cfg.tau_lag = args.allostery_tau;
  cfg.compute_te = args.allostery_compute_te;
  cfg.pair_pruning.pair_topk = args.allostery_knn;
  cfg.pair_pruning.pair_cutoff_ang = args.allostery_cutoff;
  cfg.pair_pruning.use_avg_cutoff = args.allostery_use_avg_cutoff;
  cfg.pair_pruning.pair_use_contact_prior =
      args.allostery_knn_use_contact_prior;
  cfg.pair_pruning.pair_contact_cutoff_ang =
      args.allostery_knn_contact_cutoff;
  return cfg;
}

bool TitanCorrKnnAnalyzer::equivalent_config(const Config &lhs,
                                             const Config &rhs) {
  return lhs.estimator_k == rhs.estimator_k &&
         lhs.tau_lag == rhs.tau_lag &&
         lhs.compute_te == rhs.compute_te &&
         lhs.pair_pruning.pair_topk == rhs.pair_pruning.pair_topk &&
         lhs.pair_pruning.pair_cutoff_ang == rhs.pair_pruning.pair_cutoff_ang &&
         lhs.pair_pruning.use_avg_cutoff == rhs.pair_pruning.use_avg_cutoff &&
         lhs.pair_pruning.pair_use_contact_prior ==
             rhs.pair_pruning.pair_use_contact_prior &&
         lhs.pair_pruning.pair_contact_cutoff_ang ==
             rhs.pair_pruning.pair_contact_cutoff_ang;
}
