#include "postproc/postproc_common.h"

using namespace postproc_detail;

void run_tpt_postproc(const arguments &args,
                      std::unique_ptr<TitanMSM> &msm_analyzer,
                      TitanExecutor &executor, info &info_output,
                      std::fstream &log,
                      const std::vector<std::string> &lj_feature_files,
                      const TitanRMSAnalyzer *rms_analyzer,
                      const TitanDSSPAnalyzer *protein_dssp_analyzer,
                      const TitanQNativeAnalyzer *qnative_analyzer,
                      const Eigen::MatrixXd *dham_tpm,
                      const Eigen::VectorXd *dham_pi,
                      bool dham_converged,
                      const PostprocEquilibriumWeights *weight_selection,
                      TitanValidation::RuntimeCache *validation_cache,
                      titan_sparse_graph_module::TitanUnifiedGraphRuntime
                          *graph_runtime) {
  if (!args.tpt.flag_tpt || !msm_analyzer)
    return;

  if (!msm_analyzer->is_built()) {
    TitanUI::print_warning(
        "TPT: MSM has not been built successfully. Skipping TPT.");
    return;
  }

  auto start_tpt = std::chrono::steady_clock::now();
  TitanUI::print_module_header("Transition Path Theory");
  TitanUI::log_kv(log, "TPT Mode", args.tpt.mode);
  if (args.tpt.mode == "kinetic") {
    TitanUI::print_warning(
        "TPT kinetic mode assumes the MSM came from continuous trajectories "
        "at the target ensemble. TITAN will report kinetic diagnostics, "
        "but does not certify kinetic validity automatically.");
  }
  if (args.dham.flag_dham && !args.tpt.use_dham) {
    TitanUI::print_warning(
        "TPT: DHAM is available but -tpt-use-dham is off, so TITAN will use the built MSM model.");
  }

  std::vector<int> macro_assignments;
  macro_assignments =
      msm_analyzer->compute_pcca_macrostates(args.msm.msm_macrostates);
  if (!macro_assignments.empty()) {
    TitanUI::log_kv(
        log, "TPT Macrostate Count",
        std::to_string(msm_analyzer->effective_pcca_macrostates(
            args.msm.msm_macrostates)));
  }

  const LjMicrostateSummary lj_summary =
      compute_lj_microstate_summary(*msm_analyzer, lj_feature_files, log);
  const ProteinMicrostateSummary protein_summary =
      compute_protein_microstate_summary(*msm_analyzer, rms_analyzer,
                                         protein_dssp_analyzer, log);
  const QNativeMicrostateSummary qnative_summary =
      compute_qnative_microstate_summary(*msm_analyzer, qnative_analyzer, log);

  try {
    const bool kinetic_mode = (args.tpt.mode == "kinetic");
    if (kinetic_mode && args.tpt.use_dham && args.dham.flag_dham) {
      TitanUI::print_warning(
          "TPT kinetic mode ignores DHAM-reweighted TPM/pi and uses the built MSM model from the continuous trajectories.");
    }
    if (!kinetic_mode && !args.tpt.demux_map_file.empty()) {
      TitanUI::print_warning(
          "TPT: -tpt-demux-map is only used in kinetic mode. Ensemble TPT will ignore it.");
    }

    const bool can_use_dham =
        !kinetic_mode && args.tpt.use_dham && args.dham.flag_dham &&
        dham_converged &&
        dham_tpm && dham_pi && dham_tpm->rows() == msm_analyzer->n_clusters() &&
        dham_tpm->cols() == msm_analyzer->n_clusters() &&
        dham_pi->size() == msm_analyzer->n_clusters();

    Eigen::MatrixXd demux_tpm;
    Eigen::VectorXd demux_pi;
    std::vector<std::vector<int>> kinetic_segments;
    std::int64_t demux_transition_count = 0;
    std::string demux_error;
    const Eigen::MatrixXd *model_tpm_ptr = &msm_analyzer->tpm();
    const Eigen::VectorXd *model_pi_ptr = &msm_analyzer->stationary_dist();
    std::string model_source = "msm_built_model";

    if (kinetic_mode) {
      model_source = "msm_built_model_kinetic";
      if (!args.tpt.demux_map_file.empty()) {
        std::vector<std::vector<int>> demux_rows;
        if (!read_tpt_demux_map(args.tpt.demux_map_file, demux_rows,
                                demux_error)) {
          TitanUI::print_error("TPT kinetic demux: " + demux_error);
          return;
        }
        if (!build_demux_state_segments(*msm_analyzer, demux_rows,
                                        kinetic_segments, demux_error)) {
          TitanUI::print_error("TPT kinetic demux: " + demux_error);
          return;
        }
        if (!build_tpt_model_from_segments(
                msm_analyzer->n_clusters(), msm_analyzer->msm_lag_time(),
                kinetic_segments, false, demux_tpm, demux_pi,
                demux_transition_count, demux_error)) {
          TitanUI::print_error("TPT kinetic demux: " + demux_error);
          return;
        }
        model_tpm_ptr = &demux_tpm;
        model_pi_ptr = &demux_pi;
        model_source = "msm_demux_kinetic";
        TitanUI::print_info("TPT kinetic demux: rebuilt kinetic MSM from " +
                            std::to_string(kinetic_segments.size()) +
                            " demuxed segment(s) using " +
                            std::to_string(demux_transition_count) +
                            " lag-valid transitions.");
        TitanUI::log_kv(log, "TPT Demux Map", args.tpt.demux_map_file);
        TitanUI::log_kv(log, "TPT Demux Segments",
                        std::to_string(kinetic_segments.size()));
        TitanUI::log_kv(log, "TPT Demux Transitions",
                        std::to_string(demux_transition_count));
      } else {
        kinetic_segments =
            TitanValidation::build_msm_state_segments(*msm_analyzer);
      }
    } else if (can_use_dham) {
      model_tpm_ptr = dham_tpm;
      model_pi_ptr = dham_pi;
      model_source = "dham_reweighted_" + args.dham.ensemble + "_target";
    }

    if (!kinetic_mode && args.tpt.use_dham && args.dham.flag_dham &&
        !can_use_dham) {
      TitanUI::print_warning(
          "TPT: requested DHAM-backed model, but no converged DHAM TPM/pi is available. Falling back to MSM.");
    }
    TitanUI::log_kv(log, "TPT Model Source", model_source);

    std::vector<int> source_states;
    std::vector<int> sink_states;
    std::string parse_error;
    std::string source_label;
    std::string sink_label;
    if (!resolve_tpt_basin_spec(args.tpt.source_spec, *msm_analyzer,
                                macro_assignments, lj_summary, protein_summary,
                                qnative_summary, *model_pi_ptr, source_states,
                                source_label, parse_error)) {
      TitanUI::print_error("TPT: invalid -tpt-source: " + parse_error);
      return;
    }
    if (!resolve_tpt_basin_spec(args.tpt.sink_spec, *msm_analyzer,
                                macro_assignments, lj_summary, protein_summary,
                                qnative_summary, *model_pi_ptr, sink_states,
                                sink_label, parse_error)) {
      TitanUI::print_error("TPT: invalid -tpt-sink: " + parse_error);
      return;
    }

    TitanTPT analyzer(*model_tpm_ptr, *model_pi_ptr);
    analyzer.compute(source_states, sink_states, args.tpt.top_paths,
                     args.tpt.flux_cut, &executor,
                     static_cast<int>(executor.capacity()));
    analyzer.write_results(info_output.get_tpt_summary_filename(),
                           info_output.get_tpt_committor_filename(),
                           info_output.get_tpt_flux_edges_filename(),
                           info_output.get_tpt_paths_filename(), log,
                           args.tpt.mode,
                           (!kinetic_mode && args.dham.flag_dham),
                           source_label,
                           sink_label, model_source);
    write_tpt_state_labels(info_output.get_tpt_state_labels_filename(), analyzer,
                           *msm_analyzer, macro_assignments, lj_summary,
                           protein_summary, qnative_summary,
                           analyzer.stationary_dist());
    const auto *tica_context_weights =
        weight_selection ? context_global_frame_weights(*weight_selection, false)
                         : nullptr;
    if (weight_selection && weight_selection->use_weights &&
        weight_selection->resolved_source == "mbar" && tica_context_weights) {
      EquilibriumPopulationSummary equilibrium_summary;
      if (compute_equilibrium_populations(*msm_analyzer,
                                          *tica_context_weights,
                                          macro_assignments,
                                          equilibrium_summary)) {
        write_tpt_equilibrium_overlay_file(
            info_output.get_tpt_equilibrium_overlay_filename(), analyzer,
            macro_assignments, equilibrium_summary, *weight_selection);
        TitanUI::log_kv(log, "TPT Equilibrium Overlay",
                        info_output.get_tpt_equilibrium_overlay_filename());
        TitanUI::print_info("TPT: Exported equilibrium overlay to " +
                            info_output.get_tpt_equilibrium_overlay_filename());
      } else {
        TitanUI::print_warning(
            "TPT: unable to map selected equilibrium frame weights onto "
            "microstates. Skipping equilibrium overlay.");
      }
    }
    const bool wrote_reactive_fes = write_tpt_reactive_fes_2d(
        info_output.get_tpt_reactive_fes_2d_filename(), analyzer,
        *msm_analyzer, log);
    const bool wrote_flux_map = write_tpt_flux_map_2d(
        info_output.get_tpt_flux_map_2d_filename(), analyzer, *msm_analyzer);
    const std::size_t n_path_clusters = write_tpt_path_clusters_file(
        info_output.get_tpt_path_clusters_filename(), analyzer);
    const TptPathSupport bottleneck_support =
        accumulate_tpt_path_support(analyzer, executor);
    const auto bottleneck_edge_rows =
        compute_tpt_bottleneck_edges(analyzer, macro_assignments,
                                     bottleneck_support);
    const auto bottleneck_state_rows =
        compute_tpt_bottleneck_states(analyzer, macro_assignments,
                                      bottleneck_support);
    const std::size_t n_bottleneck_edges = write_tpt_bottleneck_edges_file(
        info_output.get_tpt_bottleneck_edges_filename(),
        bottleneck_edge_rows);
    const std::size_t n_bottleneck_states = write_tpt_bottleneck_states_file(
        info_output.get_tpt_bottleneck_states_filename(),
        bottleneck_state_rows);
    append_tpt_bottleneck_summary(
        info_output.get_tpt_summary_filename(), bottleneck_edge_rows,
        bottleneck_state_rows);
    TitanUI::log_kv(log, "TPT Path Clusters",
                    std::to_string(n_path_clusters));
    TitanUI::log_kv(log, "TPT Bottleneck Edges",
                    std::to_string(n_bottleneck_edges));
    TitanUI::log_kv(log, "TPT Bottleneck States",
                    std::to_string(n_bottleneck_states));
    TitanUI::print_info("TPT: Exported path clusters to " +
                        info_output.get_tpt_path_clusters_filename());
    TitanUI::print_info("TPT: Exported bottleneck edges to " +
                        info_output.get_tpt_bottleneck_edges_filename());
    TitanUI::print_info("TPT: Exported bottleneck states to " +
                        info_output.get_tpt_bottleneck_states_filename());
    if (graph_runtime) {
      std::vector<titan_sparse_graph_module::TptCapturedEdgeRow>
          graph_edge_rows;
      graph_edge_rows.reserve(bottleneck_edge_rows.size());
      for (const auto &row : bottleneck_edge_rows) {
        graph_edge_rows.push_back(to_graph_tpt_edge_row(row));
      }
      std::vector<titan_sparse_graph_module::TptCapturedNodeRow>
          graph_node_rows;
      graph_node_rows.reserve(bottleneck_state_rows.size());
      for (const auto &row : bottleneck_state_rows) {
        graph_node_rows.push_back(to_graph_tpt_node_row(row));
      }
      graph_runtime->capture_tpt(analyzer, macro_assignments, graph_edge_rows,
                                 graph_node_rows);
    }

    if (!macro_assignments.empty()) {
      Eigen::MatrixXd macro_tpm;
      Eigen::VectorXd macro_pi;
      std::string macro_error;
      if (build_macro_tpt_model(*model_tpm_ptr, *model_pi_ptr, macro_assignments,
                                macro_tpm, macro_pi, macro_error)) {
        const std::size_t n_pairs = write_tpt_all_pairs_kinetics_file(
            info_output.get_tpt_all_pairs_kinetics_filename(), macro_tpm,
            macro_pi, msm_analyzer->lag_time_ns());
        TitanUI::log_kv(log, "TPT All-Pairs Macro Kinetics",
                        std::to_string(n_pairs));
        TitanUI::print_info("TPT: Exported all-pairs macro kinetics to " +
                            info_output.get_tpt_all_pairs_kinetics_filename());
      } else {
        TitanUI::print_warning("TPT all-pairs kinetics: " + macro_error);
      }
    }
    if (kinetic_mode) {
      if (kinetic_segments.empty()) {
        kinetic_segments =
            TitanValidation::build_msm_state_segments(*msm_analyzer);
      }
      const TitanValidation::TptKineticDiagnostics kinetic_diag =
          TitanValidation::compute_tpt_kinetic_diagnostics_from_segments(
              kinetic_segments, msm_analyzer->n_clusters(),
              msm_analyzer->msm_lag_time(), msm_analyzer->lag_time_ns(),
              analyzer);
      write_tpt_kinetics_file(info_output.get_tpt_kinetics_filename(),
                              kinetic_diag, model_source);
      TitanUI::log_kv(log, "TPT Kinetic Suitability",
                      kinetic_diag.suitability);
      TitanUI::log_kv(log, "TPT Observed A->B Events",
                      std::to_string(kinetic_diag.observed_a_to_b_events));
      TitanUI::log_kv(log, "TPT Observed B->A Events",
                      std::to_string(kinetic_diag.observed_b_to_a_events));
      TitanUI::log_kv(log, "TPT Observed Round Trips",
                      std::to_string(kinetic_diag.observed_round_trips));
      TitanUI::log_kv(log, "TPT Model MFPT A->B (ns)",
                      std::to_string(kinetic_diag.model_mfpt_ab_ns));
      TitanUI::log_kv(log, "TPT CK / Markov Sanity",
                      kinetic_diag.markov_sanity);
      TitanUI::print_info("TPT: Exported kinetic diagnostics to " +
                          info_output.get_tpt_kinetics_filename());
      if (kinetic_diag.observed_b_to_a_events == 0) {
        TitanUI::print_warning(
            "TPT kinetic diagnostics: no observed B->A back-transitions. Treat rates as exploratory only.");
      } else if (kinetic_diag.observed_round_trips < 3) {
        TitanUI::print_warning(
            "TPT kinetic diagnostics: fewer than 3 observed round-trips. Treat MFPT/rates as exploratory.");
      }
      if (kinetic_diag.markov_sanity == "fail") {
        TitanUI::print_warning(
            "TPT kinetic diagnostics: CK / Markov consistency is poor. Treat kinetic rates as weak.");
      } else if (kinetic_diag.markov_sanity == "weak") {
        TitanUI::print_warning(
            "TPT kinetic diagnostics: CK / Markov consistency is only moderate.");
      }
    }
    if (validation_cache) {
      validation_cache->tpt_context.available = true;
      validation_cache->tpt_context.kinetic_mode = kinetic_mode;
      validation_cache->tpt_context.symmetrize_bootstrap = !kinetic_mode;
      validation_cache->tpt_context.n_states = msm_analyzer->n_clusters();
      validation_cache->tpt_context.lag_frames = msm_analyzer->msm_lag_time();
      validation_cache->tpt_context.top_paths = args.tpt.top_paths;
      validation_cache->tpt_context.flux_cut = args.tpt.flux_cut;
      validation_cache->tpt_context.lag_time_ns = msm_analyzer->lag_time_ns();
      validation_cache->tpt_context.model_source = model_source;
      validation_cache->tpt_context.source_states = source_states;
      validation_cache->tpt_context.sink_states = sink_states;
      validation_cache->tpt_context.model_tpm = *model_tpm_ptr;
      validation_cache->tpt_context.model_pi = *model_pi_ptr;
      validation_cache->tpt_context.bootstrap_segments =
          kinetic_mode
              ? kinetic_segments
              : TitanValidation::build_msm_state_segments(*msm_analyzer);
    }
    const bool defer_legacy_bootstrap =
        validation_cache &&
        TitanValidation::validation_bootstrap_matches_tpt_legacy(args);
    if (args.tpt.bootstrap_samples > 0 && !defer_legacy_bootstrap) {
      if (model_source.rfind("dham_reweighted_", 0) == 0) {
        TitanValidation::TptBootstrapAggregate bootstrap;
        std::string bootstrap_error;
        if (TitanValidation::run_dham_tpt_bootstrap(
                validation_cache ? validation_cache->dham_context
                                 : TitanValidation::DhamRebuildContext{},
                args.tpt.top_paths, args.tpt.flux_cut, source_states,
                sink_states, args.tpt.bootstrap_samples, args.tpt.bootstrap_seed,
                static_cast<int>(executor.capacity()), executor, bootstrap,
                bootstrap_error)) {
          write_tpt_bootstrap_files(
              info_output.get_tpt_bootstrap_summary_filename(),
              info_output.get_tpt_bootstrap_states_filename(), bootstrap,
              analyzer, model_source);
          TitanUI::log_kv(log, "TPT Bootstrap Requested",
                          std::to_string(args.tpt.bootstrap_samples));
          TitanUI::log_kv(log, "TPT Bootstrap Successful",
                          std::to_string(bootstrap.successful_samples));
          TitanUI::print_info(
              "TPT: Exported bootstrap summary to " +
              info_output.get_tpt_bootstrap_summary_filename());
          TitanUI::print_info(
              "TPT: Exported bootstrap state stats to " +
              info_output.get_tpt_bootstrap_states_filename());
        } else {
          TitanUI::print_warning("TPT bootstrap: " + bootstrap_error);
        }
      } else {
        std::vector<std::vector<int>> bootstrap_segments =
            kinetic_mode ? kinetic_segments
                         : TitanValidation::build_msm_state_segments(
                               *msm_analyzer);
        const bool symmetrize_bootstrap = !kinetic_mode;
        TitanValidation::TptBootstrapAggregate bootstrap;
        std::string bootstrap_error;
        if (TitanValidation::run_tpt_bootstrap(
                bootstrap_segments, msm_analyzer->n_clusters(),
                msm_analyzer->msm_lag_time(), msm_analyzer->lag_time_ns(),
                symmetrize_bootstrap, source_states, sink_states,
                args.tpt.top_paths, args.tpt.flux_cut,
                static_cast<int>(executor.capacity()),
                args.tpt.bootstrap_samples, args.tpt.bootstrap_seed, executor,
                bootstrap, bootstrap_error)) {
          write_tpt_bootstrap_files(
              info_output.get_tpt_bootstrap_summary_filename(),
              info_output.get_tpt_bootstrap_states_filename(), bootstrap,
              analyzer, model_source);
          TitanUI::log_kv(log, "TPT Bootstrap Requested",
                          std::to_string(args.tpt.bootstrap_samples));
          TitanUI::log_kv(log, "TPT Bootstrap Successful",
                          std::to_string(bootstrap.successful_samples));
          TitanUI::print_info(
              "TPT: Exported bootstrap summary to " +
              info_output.get_tpt_bootstrap_summary_filename());
          TitanUI::print_info(
              "TPT: Exported bootstrap state stats to " +
              info_output.get_tpt_bootstrap_states_filename());
        } else {
          TitanUI::print_warning("TPT bootstrap: " + bootstrap_error);
        }
      }
    } else if (defer_legacy_bootstrap) {
      TitanUI::log_kv(log, "TPT Bootstrap Reuse",
                      "validation core will write legacy bootstrap files");
    }
    TitanUI::log_kv(log, "TPT Source Basin", source_label);
    TitanUI::log_kv(log, "TPT Sink Basin", sink_label);
    TitanUI::print_info("TPT: Exported summary to " +
                        info_output.get_tpt_summary_filename());
    TitanUI::print_info("TPT: Exported committors to " +
                        info_output.get_tpt_committor_filename());
    TitanUI::print_info("TPT: Exported flux edges to " +
                        info_output.get_tpt_flux_edges_filename());
    TitanUI::print_info("TPT: Exported dominant paths to " +
                        info_output.get_tpt_paths_filename());
    TitanUI::print_info("TPT: Exported state labels to " +
                        info_output.get_tpt_state_labels_filename());
    if (wrote_reactive_fes) {
      TitanUI::print_info("TPT: Exported reactive FES to " +
                          info_output.get_tpt_reactive_fes_2d_filename());
    }
    if (wrote_flux_map) {
      TitanUI::print_info("TPT: Exported flux map to " +
                          info_output.get_tpt_flux_map_2d_filename());
    }
  } catch (const std::exception &e) {
    TitanUI::print_error(std::string("TPT: ") + e.what());
    return;
  }

  auto time_tpt = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_tpt);
  TitanUI::log_time(log, "TPT Analysis", time_tpt.count() / 1000.0);
}
