#include "postproc/postproc_common.h"

using namespace postproc_detail;

// ============================================================================
// 5. DCCM + PCA post-processing
// ============================================================================
void run_dccm_pca_postproc(const arguments &args, size_t tot_snap_allfiles,
                           DCCMNetwork &dccm_analyzer, TitanExecutor &executor,
                           std::chrono::milliseconds &time_network,
                           info &info_output, std::fstream &log,
                           const PostprocEquilibriumWeights &weight_selection,
                           const std::vector<std::string> &replica_traj_files,
                           TitanPCA::FesPostprocMetrics *metrics,
                           titan_sparse_graph_module::TitanUnifiedGraphRuntime
                               *graph_runtime) {
  (void)graph_runtime;
  if (!(args.flag_network_explicit || args.flag_pca || args.flag_allostery ||
        args.corr_knn.flag_corr_knn || args.community.flag_community))
    return;

  try {
    if (tot_snap_allfiles > 0) {
      const auto *pca_context_weights =
          context_global_frame_weights(weight_selection, true);
      const bool want_mbar_sidecars =
          is_mbar_equilibrium_weight_source(weight_selection) &&
          pca_context_weights;
      MappedEquilibriumFrameWeights dccm_weight_context;
      auto start_dccm = std::chrono::steady_clock::now();
      TitanUI::print_module_header("DCCM Network Analysis");

      dccm_analyzer.calculate_and_export(executor, info_output);
      if (dccm_analyzer.dense_postproc_skipped_due_memory()) {
        TitanUI::print_warning(
            "DCCM/PCA downstream modules skipped because dense DCCM network "
            "post-processing exceeded the safe memory budget.");
        ensure_csv_header_if_missing(info_output.get_network_mi_edges_filename(),
                                     "Source,Target,MI");
        ensure_csv_header_if_missing(info_output.get_network_gc_edges_filename(),
                                     "Source,Target,GC");
        ensure_csv_header_if_missing(info_output.get_network_te_edges_filename(),
                                     "Source,Target,Net_TE");
        time_network += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_dccm);
        TitanUI::log_time(log, "Network Output", time_network.count() / 1000.0);
        TitanUI::log_kv(log, "DCCM Network Analysis",
                        "skipped by memory guard");
        return;
      }

      if (want_mbar_sidecars) {
        dccm_weight_context = map_global_weights_to_local_frame_context(
            "DCCM", dccm_analyzer.get_global_frame_indices(),
            *pca_context_weights);
      }

      if (dccm_weight_context.use_weights &&
          (args.flag_network_explicit || args.flag_allostery ||
           args.corr_knn.flag_corr_knn ||
           args.community.flag_community)) {
        write_dccm_equilibrium_context_file(
            info_output.get_dccm_equilibrium_context_filename(), dccm_analyzer,
            dccm_weight_context, weight_selection);
        TitanUI::log_kv(log, "DCCM Equilibrium Context",
                        info_output.get_dccm_equilibrium_context_filename());
      }

      time_network += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_dccm);
      TitanUI::log_time(log, "Network Output", time_network.count() / 1000.0);

      TitanCorrKnnAnalyzer corr_knn_cached_analyzer;
      bool corr_knn_cached = false;
      auto corr_knn_cfg =
          TitanCorrKnnAnalyzer::from_corr_knn_args(args.corr_knn);
      auto allostery_knn_cfg =
          TitanCorrKnnAnalyzer::from_allostery_args(args);
      if (dccm_analyzer.network_sparse_mode_active()) {
        auto apply_auto_sparse_pruning = [&](TitanCorrKnnAnalyzer::Config &cfg) {
          if (cfg.pair_pruning.pair_cutoff_ang <= 0.0)
            cfg.pair_pruning.pair_cutoff_ang = args.network_sparse_cut;
          if (cfg.pair_pruning.pair_topk <= 0)
            cfg.pair_pruning.pair_topk = args.network_sparse_k;
          cfg.pair_pruning.use_avg_cutoff = false;
        };
        apply_auto_sparse_pruning(corr_knn_cfg);
        apply_auto_sparse_pruning(allostery_knn_cfg);
        TitanUI::print_warning(
            "Information-theory modules are using sparse pair pruning inherited "
            "from DCCM auto-sparse mode.");
      }
      auto log_exact_corr_knn_result =
          [&](const std::string &label_prefix,
              const TitanCorrKnnAnalyzer &analyzer,
              const std::string &summary_path, const std::string &mi_path,
              const std::string &gc_path, const std::string &te_path) {
            const auto &result = analyzer.result();
            if (!summary_path.empty()) {
              TitanUI::log_kv(log, label_prefix + " Summary File", summary_path);
            }
            TitanUI::log_kv(log, label_prefix + " MI File", mi_path);
            TitanUI::log_kv(log, label_prefix + " GC File", gc_path);
            TitanUI::log_kv(log, label_prefix + " TE File", te_path);
            TitanUI::log_kv(
                log, label_prefix + " TE Stage",
                analyzer.config().compute_te ? "enabled" : "disabled");
            TitanUI::log_kv(
                log, label_prefix + " Estimator",
                "knn (exact continuous 3D, k=" +
                    std::to_string(analyzer.config().estimator_k) + ")");
            TitanUI::log_kv(log, label_prefix + " Tau Lag",
                            std::to_string(analyzer.config().tau_lag));
            TitanUI::log_kv(
                log, label_prefix + " Pair Filter",
                titan_corr_knn::describe_pair_filter(
                    analyzer.config().pair_pruning));
            TitanUI::log_kv(
                log, label_prefix + " Candidate Pairs",
                std::to_string(result.candidate_pair_count) + " / " +
                    std::to_string(result.dense_pair_count));
            if (!result.mi_available) {
              TitanUI::print_warning(
                  label_prefix + ": insufficient effective frames for exact MI "
                                 "with k=" +
                  std::to_string(analyzer.config().estimator_k) +
                  "; CSV outputs contain headers only.");
            } else if (analyzer.config().compute_te && !result.te_available) {
              TitanUI::print_warning(
                  label_prefix +
                  ": insufficient effective lagged samples for exact TE; TE "
                  "CSV contains headers only.");
            }
            if (result.degenerate_node_count > 0 ||
                result.degenerate_pair_count > 0) {
              TitanUI::log_kv(
                  log, label_prefix + " Degenerate Inputs",
                  std::to_string(result.degenerate_node_count) + " nodes, " +
                      std::to_string(result.degenerate_pair_count) + " pairs");
            }
            if (result.candidate_pair_count == 0 &&
                result.dense_pair_count > 0) {
              TitanUI::print_warning(
                  label_prefix +
                  ": pair pruning removed all candidate pairs; outputs contain "
                  "headers only.");
            }
          };

      if (args.corr_knn.flag_corr_knn) {
        auto start_corr_knn = std::chrono::steady_clock::now();
        TitanUI::print_module_header("Corr-kNN Information Analysis");
        corr_knn_cached_analyzer.configure(corr_knn_cfg);
        corr_knn_cached_analyzer.run(dccm_analyzer, executor, log);
        corr_knn_cached_analyzer.export_module_outputs(info_output);
        corr_knn_cached = true;
        log_exact_corr_knn_result(
            "Corr-kNN", corr_knn_cached_analyzer,
            info_output.get_corr_knn_summary_filename(),
            info_output.get_corr_knn_mi_pairs_filename(),
            info_output.get_corr_knn_gc_pairs_filename(),
            info_output.get_corr_knn_te_pairs_filename());
        if (dccm_weight_context.use_weights) {
          write_corr_knn_equilibrium_ranking_file(
              info_output.get_corr_knn_equilibrium_ranking_filename(),
              info_output, dccm_weight_context, weight_selection);
          TitanUI::log_kv(log, "Corr-kNN Equilibrium Ranking",
                          info_output.get_corr_knn_equilibrium_ranking_filename());
        }
        const auto time_corr_knn =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_corr_knn);
        TitanUI::log_time(log, "Corr-kNN", time_corr_knn.count() / 1000.0);
      }

      // Allosteric Information Theory (MI -> GC + Transfer Entropy)
      if (args.flag_allostery) {
        auto start_allost = std::chrono::steady_clock::now();
        TitanUI::print_module_header(
            "Allosteric Information Theory (MI, GC + TE)");
        if (args.allostery_estimator == "knn") {
          TitanUI::log_kv(log, "Allostery Backend Route",
                          "knn -> titan_corr_knn_core");
          TitanUI::log_kv(log, "Allostery Histogram Bins",
                          "ignored in knn mode");
          TitanUI::log_kv(log, "Allostery KDE Bandwidth",
                          "ignored in knn mode");
          if (args.allostery_bins > 0) {
            TitanUI::print_warning(
                "Allostery: -allostery-bins is ignored when "
                "-allostery-estimator=knn.");
          }
          if (args.allostery_kde_bandwidth > 0.0) {
            TitanUI::print_warning(
                "Allostery: -allostery-kde-bandwidth is ignored when "
                "-allostery-estimator=knn.");
          }

          const bool reuse_corr_knn =
              corr_knn_cached &&
              TitanCorrKnnAnalyzer::equivalent_config(corr_knn_cfg,
                                                      allostery_knn_cfg);
          TitanCorrKnnAnalyzer allostery_knn_analyzer;
          const TitanCorrKnnAnalyzer *active_analyzer = nullptr;
          if (reuse_corr_knn) {
            corr_knn_cached_analyzer.export_legacy_allostery_outputs(
                info_output.get_network_mi_edges_filename(),
                info_output.get_network_gc_edges_filename(),
                info_output.get_network_te_edges_filename());
            store_allostery_graph_edges_from_corr_knn(dccm_analyzer,
                                                     corr_knn_cached_analyzer);
            TitanUI::log_kv(log, "Allostery kNN Reuse",
                            "reused corr-kNN result");
            active_analyzer = &corr_knn_cached_analyzer;
          } else {
            if (corr_knn_cached) {
              TitanUI::log_kv(log, "Allostery kNN Reuse",
                              "disabled (effective config differs from "
                              "-corr-knn)");
            }
            allostery_knn_analyzer.configure(allostery_knn_cfg);
            allostery_knn_analyzer.run(dccm_analyzer, executor, log);
            allostery_knn_analyzer.export_legacy_allostery_outputs(
                info_output.get_network_mi_edges_filename(),
                info_output.get_network_gc_edges_filename(),
                info_output.get_network_te_edges_filename());
            store_allostery_graph_edges_from_corr_knn(dccm_analyzer,
                                                     allostery_knn_analyzer);
            active_analyzer = &allostery_knn_analyzer;
          }
          log_exact_corr_knn_result(
              "Allostery", *active_analyzer, "",
              info_output.get_network_mi_edges_filename(),
              info_output.get_network_gc_edges_filename(),
              info_output.get_network_te_edges_filename());
        } else {
          DCCMNetwork::AllosteryConfig cfg;
          cfg.tau_lag = args.allostery_tau;
          cfg.num_bins = args.allostery_bins;
          cfg.knn = args.allostery_knn;
          cfg.cutoff_ang = args.allostery_cutoff;
          cfg.use_avg_cutoff = args.allostery_use_avg_cutoff;
          cfg.knn_use_contact_prior = args.allostery_knn_use_contact_prior;
          cfg.knn_contact_cutoff = args.allostery_knn_contact_cutoff;
          cfg.estimator = args.allostery_estimator;
          cfg.kde_bandwidth = args.allostery_kde_bandwidth;
          cfg.compute_te = args.allostery_compute_te;
          if (dccm_analyzer.network_sparse_mode_active()) {
            cfg.force_sparse_candidates = true;
            if (cfg.cutoff_ang <= 0.0)
              cfg.cutoff_ang = args.network_sparse_cut;
            if (cfg.knn <= 0)
              cfg.knn = args.network_sparse_k;
            cfg.use_avg_cutoff = false;
          }
          dccm_analyzer.calculate_allostery_information_optimized(
              executor, cfg, info_output.get_network_gc_edges_filename(),
              info_output.get_network_mi_edges_filename(),
              info_output.get_network_te_edges_filename());
          ensure_csv_header_if_missing(info_output.get_network_mi_edges_filename(),
                                       "Source,Target,MI");
          ensure_csv_header_if_missing(info_output.get_network_gc_edges_filename(),
                                       "Source,Target,GC");
          ensure_csv_header_if_missing(
              info_output.get_network_te_edges_filename(),
              "Source,Target,Net_TE");
          TitanUI::log_kv(log, "Allostery MI File",
                          info_output.get_network_mi_edges_filename());
          TitanUI::log_kv(log, "Allostery GC File",
                          info_output.get_network_gc_edges_filename());
          TitanUI::log_kv(log, "Allostery TE File",
                          info_output.get_network_te_edges_filename());
          TitanUI::log_kv(log, "Allostery TE Stage",
                          cfg.compute_te ? "enabled" : "disabled (GC only)");
          TitanUI::log_kv(log, "Allostery Estimator",
                          cfg.estimator == "kde"
                              ? (std::string("kde (bandwidth=") +
                                 std::to_string(cfg.kde_bandwidth) + ")")
                              : cfg.estimator);
          TitanUI::log_kv(log, "Allostery Prune Window",
                          cfg.use_avg_cutoff ? "union across frames"
                                             : "frame-0 only");
          if (cfg.knn > 0) {
            std::ostringstream knn_prior;
            if (cfg.knn_use_contact_prior) {
              knn_prior << "contact-frequency (Cbeta/N anchors, "
                        << cfg.knn_contact_cutoff << " A)";
            } else {
              knn_prior << "frame-voted nearest";
            }
            TitanUI::log_kv(log, "Allostery kNN Prior", knn_prior.str());
          }
          if (cfg.cutoff_ang > 0.0 || cfg.knn > 0) {
            std::ostringstream pair_filter;
            bool has_filter = false;
            if (cfg.cutoff_ang > 0.0) {
              pair_filter << cfg.cutoff_ang << " A cutoff";
              has_filter = true;
            }
            if (cfg.knn > 0) {
              if (has_filter)
                pair_filter << " + ";
              pair_filter << "top-" << cfg.knn << " "
                          << (cfg.knn_use_contact_prior
                                  ? "contact-frequency kNN @ "
                                  : "frame-voted kNN");
              if (cfg.knn_use_contact_prior)
                pair_filter << cfg.knn_contact_cutoff << " A";
            }
            pair_filter << " ("
                        << (cfg.use_avg_cutoff ? "union across frames"
                                               : "frame-0 only")
                        << ")";
            TitanUI::log_kv(log, "Allostery Pair Filter", pair_filter.str());
            TitanUI::log_kv(
                log, "Allostery Candidate Pairs",
                std::to_string(dccm_analyzer.get_last_allostery_candidate_pairs()) +
                    " / " +
                    std::to_string(dccm_analyzer.get_last_allostery_dense_pairs()));
            if (dccm_analyzer.get_last_allostery_candidate_pairs() == 0 &&
                dccm_analyzer.get_last_allostery_dense_pairs() > 0) {
              TitanUI::print_warning(
                  "Allostery pruning removed all candidate pairs; GC/TE outputs "
                  "contain headers only.");
            }
          }
        }
        auto time_allost =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_allost);
        TitanUI::log_time(log, "Allostery MI+GC+TE",
                          time_allost.count() / 1000.0);
        if (dccm_weight_context.use_weights) {
          write_allostery_equilibrium_ranking_file(
              info_output.get_allostery_equilibrium_ranking_filename(),
              info_output, dccm_weight_context, weight_selection);
          TitanUI::log_kv(
              log, "Allostery Equilibrium Ranking",
              info_output.get_allostery_equilibrium_ranking_filename());
        }
      }

      if (args.flag_pca) {
        const auto &pca_traj_x = dccm_analyzer.get_traj_x();
        const auto &pca_traj_y = dccm_analyzer.get_traj_y();
        const auto &pca_traj_z = dccm_analyzer.get_traj_z();
        const size_t pca_frame_count = pca_traj_x.size();
        std::vector<double> pca_frame_weights;
        if (pca_context_weights && !pca_context_weights->empty()) {
          const auto &global_weights = *pca_context_weights;
          const auto &dccm_global_indices = dccm_analyzer.get_global_frame_indices();
          if (dccm_global_indices.size() != pca_frame_count) {
            TitanUI::print_warning(
                "PCA: DCCM global frame index count (" +
                std::to_string(dccm_global_indices.size()) +
                ") != PCA frame count (" + std::to_string(pca_frame_count) +
                "). Falling back to uniform weights.");
          } else {
            pca_frame_weights.assign(pca_frame_count, 0.0);
            size_t matched_frames = 0;
            size_t weighted_frames = 0;
            double mapped_weight_sum = 0.0;
            for (size_t f = 0; f < dccm_global_indices.size(); ++f) {
              const int64_t global_idx = dccm_global_indices[f];
              if (global_idx < 0 ||
                  static_cast<size_t>(global_idx) >= global_weights.size()) {
                continue;
              }
              const double w = global_weights[static_cast<size_t>(global_idx)];
              pca_frame_weights[f] = w;
              ++matched_frames;
              if (w > 0.0)
                ++weighted_frames;
              mapped_weight_sum += w;
            }
            TitanUI::print_info(
                "PCA: Mapped " + weight_selection.source_name + " weights onto " +
                std::to_string(weighted_frames) + "/" +
                std::to_string(pca_frame_count) +
                " frames (matched indices: " + std::to_string(matched_frames) +
                ", weight sum = " + std::to_string(mapped_weight_sum) + ").");
          }
        }
        TitanPCA::execute_pca_fes_analysis(
            args, pca_traj_x, pca_traj_y, pca_traj_z,
            dccm_analyzer.get_resnames(), dccm_analyzer.get_resnrs(),
            dccm_analyzer.get_chains(), dccm_analyzer.get_is_lig(), executor,
            info_output, log, pca_frame_weights,
            make_pca_weight_context(weight_selection),
            dccm_analyzer.get_traj_times(), dccm_analyzer.get_traj_replicas(),
            replica_traj_files, metrics);
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in DCCM/PCA: " << e.what() << "\n";
  }
}

// ============================================================================
// 5. LSP-MD post-processing (Kornev, JCTC 2025)
// ============================================================================
void run_lspmd_postproc(const arguments &args, DCCMNetwork &dccm_analyzer,
                        TitanExecutor &executor, info &info_output,
                        std::fstream &log,
                        const PostprocEquilibriumWeights &weight_selection,
                        titan_sparse_graph_module::TitanUnifiedGraphRuntime
                            *graph_runtime) {
  if (!args.lspmd.flag_lspmd)
    return;
  if (dccm_analyzer.get_total_frames() == 0) {
    TitanUI::print_warning("LSP-MD: No frames collected, skipping.");
    return;
  }

  try {
    auto start_lspmd = std::chrono::steady_clock::now();
    TitanUI::print_module_header(
        "LSP-MD: Local Spatial Pattern Network Analysis");
    TitanUI::log_kv(log, "LSP-MD Frames",
                    std::to_string(dccm_analyzer.get_total_frames()));
    TitanUI::log_kv(
        log, "LSP-MD Cutoff",
        std::to_string(args.lspmd.lspmd_cutoff) + " A");

    DCCMNetwork::LSPMDConfig cfg;
    cfg.cutoff_ang = args.lspmd.lspmd_cutoff;
    cfg.use_avg_cutoff = args.lspmd.lspmd_use_avg;

    DCCMNetwork::LSPMDResult result;
    dccm_analyzer.calculate_lspmd(executor, cfg, result);
    dccm_analyzer.export_lspmd(result);
    if (graph_runtime) {
      graph_runtime->capture_lspmd(dccm_analyzer, result);
    }
    const auto *pca_context_weights =
        context_global_frame_weights(weight_selection, true);
    if (is_mbar_equilibrium_weight_source(weight_selection) &&
        pca_context_weights) {
      const auto mapped_context = map_global_weights_to_local_frame_context(
          "LSP-MD", dccm_analyzer.get_global_frame_indices(),
          *pca_context_weights);
      if (mapped_context.use_weights) {
        const bool dccm_context_already_written =
            args.flag_network_explicit || args.flag_allostery ||
            args.corr_knn.flag_corr_knn ||
            args.community.flag_community;
        if (!dccm_context_already_written) {
          write_dccm_equilibrium_context_file(
              info_output.get_dccm_equilibrium_context_filename(),
              dccm_analyzer, mapped_context, weight_selection);
          TitanUI::log_kv(log, "DCCM Equilibrium Context",
                          info_output.get_dccm_equilibrium_context_filename());
        }
        write_lspmd_equilibrium_ranking_file(
            info_output.get_lspmd_equilibrium_ranking_filename(),
            dccm_analyzer, result, mapped_context, weight_selection);
        TitanUI::log_kv(log, "LSP-MD Equilibrium Ranking",
                        info_output.get_lspmd_equilibrium_ranking_filename());
      }
    }

    TitanUI::log_kv(log, "LSP-MD Edges",
                    std::to_string(result.edges.size()));
    auto time_lspmd = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_lspmd);
    TitanUI::log_time(log, "LSP-MD Analysis", time_lspmd.count() / 1000.0);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Exception in LSP-MD: " << e.what() << "\n";
  }
}
