#include "postproc/postproc_common.h"

using namespace postproc_detail;

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
