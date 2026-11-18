#include "postproc/postproc_common.h"

using namespace postproc_detail;

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
