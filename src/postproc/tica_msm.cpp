#include "postproc/postproc_common.h"

using namespace postproc_detail;

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
