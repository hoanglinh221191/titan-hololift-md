#include "engine/engine_run.h"

namespace engine_detail {

std::optional<int> EngineRun::finish_outputs() {

  TitanUI::print_info(
      "\033[3mFinalizing module calculations and aggregating data...\033[0m");

  if (pbe_analyzer.enabled()) {
    pbe_analyzer.finalize(log);
  }

  // --- IFP: Aggregate occupancy and export CSV ---
  if (Arguments.ifp.flag_ifp && ifp_analyzer) {
    if (Arguments.ifp.pbctopo_certified_only) {
      ifp_pbctopo_filter_report.flush();
      const size_t processed_frames = ifp_analyzer->processed_frame_count();
      std::ofstream summary(
          (*info_output).get_ifp_pbctopo_filter_summary_filename());
      if (!summary.is_open()) {
        TitanUI::print_error(
            "[IFP] Cannot open PBCTOPO frame-filter summary output.");
        return 1;
      }
      const auto status_count = [&](const char *status) {
        const auto found = ifp_pbctopo_status_counts.find(status);
        return found == ifp_pbctopo_status_counts.end() ? size_t{0}
                                                        : found->second;
      };
      summary
          << "criterion,total_frames,included_frames,excluded_frames,"
             "ifp_processed_frames,certified_frames,rescued_frames,"
             "weak_observation_frames,local_unwrap_frames,fallback_frames,"
             "missing_status_frames\n"
          << "status_equals_certified," << ifp_pbctopo_total_frames << ','
          << ifp_pbctopo_included_frames << ','
          << (ifp_pbctopo_total_frames - ifp_pbctopo_included_frames) << ','
          << processed_frames << ',' << status_count("certified") << ','
          << status_count("rescued") << ','
          << status_count("weak_observation") << ','
          << status_count("local_unwrap") << ','
          << status_count("fallback") << ','
          << ifp_pbctopo_missing_status_frames << '\n';
      TitanUI::print_info(
          "[IFP] PBCTOPO certified-frame filter included " +
          std::to_string(ifp_pbctopo_included_frames) + " of " +
          std::to_string(ifp_pbctopo_total_frames) + " frames.");
      if (ifp_pbctopo_missing_status_frames != 0 ||
          processed_frames != ifp_pbctopo_included_frames) {
        TitanUI::print_error(
            "[IFP] PBCTOPO frame-filter accounting failed closed.");
        return 1;
      }
    }
    ifp_analyzer->aggregate_occupancy();
    std::string ifp_outfile = Arguments.ifp.output_filename.empty()
                                  ? (*info_output).get_ifp_filename()
                                  : Arguments.ifp.output_filename;
    ifp_analyzer->export_csv(ifp_outfile);
    if (ifp_analyzer->has_rna_annotations()) {
      ifp_analyzer->export_rna_annotations_csv(
          (*info_output).get_ifp_rna_annotations_filename());
    }
  }
  if (Arguments.awh.flag_awh_ifp && awh_ifp_analyzer) {
    awh_ifp_analyzer->export_results(Arguments, (*info_output), pro_in.info,
                                     lig.info);
  }
  if (rg_analyzer.enabled() && rg_analyzer.has_multiple_trajectories()) {
    rg_analyzer.finalize_all((*info_output).get_rg2_all_filename(),
                             (*info_output).get_rg2_summary_all_filename(),
                             (*info_output).get_rg2_hist_all_filename(), log);
  }
  if (cmap_analyzer.enabled() &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    cmap_analyzer.finalize_all(".", (*info_output), log);
  }
  if (interface_analyzer.enabled() &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input)) {
    try {
      interface_analyzer.finalize_all(".", (*info_output), Arguments, log);
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Interface analysis aggregate: ") +
                           e.what());
      return 1;
    }
  }
  if (qnative_analyzer.enabled() &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    qnative_analyzer.finalize_all(".", (*info_output), log);
  }
  if (s2_analyzer.enabled() &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    s2_analyzer.finalize_all(".", (*info_output), log);
  }
  if (edecomp_mm_requested(Arguments) && edecomp_engine &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    try {
      titan_energy_decomp::export_all_results(edecomp_engine->finalize_all(),
                                              ".", "_average",
                                              edecomp_output_unit(Arguments));
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Energy decomposition aggregate: ") +
                           e.what());
      return 1;
    }
  }
  if (edecomp_gbsa_requested(Arguments) && edecomp_gbsa_engine &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    try {
      titan_energy_decomp::export_endpoint_all_results(
          edecomp_gbsa_engine->finalize_all("mmgbsa"),
          edecomp_gbsa_engine->finalize_pooled_frames("mmgbsa"), ".",
          "mmgbsa", "_average", edecomp_output_unit(Arguments));
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Energy decomposition GBSA aggregate: ") +
                           e.what());
      return 1;
    }
  }
  if (edecomp_pbsa_requested(Arguments) && edecomp_pbsa_engine &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    try {
      titan_energy_decomp::export_endpoint_all_results(
          edecomp_pbsa_engine->finalize_all("mmpbsa"),
          edecomp_pbsa_engine->finalize_pooled_frames("mmpbsa"), ".",
          "mmpbsa", "_average", edecomp_output_unit(Arguments));
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Energy decomposition PBSA aggregate: ") +
                           e.what());
      return 1;
    }
  }

  if (status == false && input_name == Arguments.b_input) {
    TitanUI::print_error("Error occurred during processing!");
    return 1;
  }

  TitanUI::print_module_header("Final Results Analysis & Summary");

  try {
    out(E_e, E_v, time_of_snap, cal_out, Arguments, input_name, E_e_vector,
        E_v_vector, size_snap);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] out(): " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] out() crashed!\n";
  }

  try {
    paths.out_to_folders(E_e, E_v, time_of_snap, E_e_vector, E_v_vector,
                         Arguments);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] paths.out_to_folders(): " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] paths.out_to_folders() crashed!\n";
  }

  if (Arguments.flag_cal_force == true) {
    try {
      out_force(Force_e, Force_v, time_of_snap, Arguments, input_name,
                Force_e_vector, Force_v_vector, size_snap);
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] out_force(): " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] out_force() crashed!\n";
    }

    try {
      paths.out_force_to_folders(Force_e, Force_v, time_of_snap, Force_e_vector,
                                 Force_v_vector, Arguments);
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] paths.out_force_to_folders(): " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] paths.out_force_to_folders() crashed!\n";
    }
  }

  if (Arguments.flag_maxforce == true) {
    try {
      stat_maxforce_res(Force_res_stat_mol1, (*info_output).get_group_1_name(),
                        (*info_output), Arguments);
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] stat_maxforce_res() mol1: " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] stat_maxforce_res() mol1 crashed!\n";
    }

    try {
      stat_maxforce_res(Force_res_stat_mol2, (*info_output).get_group_2_name(),
                        (*info_output), Arguments);
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] stat_maxforce_res() mol2: " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] stat_maxforce_res() mol2 crashed!\n";
    }
  }

  try {
    impl_entropy_duan_zhang(paths, E_e_vector, E_v_vector, time_of_snap,
                            Arguments, E_e, E_v);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] impl_entropy_duan_zhang(): " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] impl_entropy_duan_zhang() crashed!\n";
  }

  try {
    print_cscd(Arguments, cscd_out_dis, log);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] print_cscd(): " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] print_cscd() crashed!\n";
  }

  try {
    nocal_analyses.finish_run(Arguments);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] nocal_analyses.finish_run(): " << e.what() << "\n";
  }

  if (Arguments.flag_sub_groups == true) {
    if (sg_streaming_analyzer.active()) {
      try {
        sg_streaming_analyzer.finalize(Arguments);
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] sg_streaming_analyzer.finalize(): " << e.what()
                  << "\n";
      } catch (...) {
        std::cerr << "[ERROR] sg_streaming_analyzer.finalize() crashed!\n";
      }
    } else {
      try {
        post_process_Energy_force_subgroups(time_of_snap, EF_subgroups, paths,
                                            Arguments, (*info_output));
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] post_process_Energy_force_subgroups(): "
                  << e.what() << "\n";
      } catch (...) {
        std::cerr << "[ERROR] post_process_Energy_force_subgroups() crashed!\n";
      }
    }

    if (Arguments.flag_cal_mindis == true) {
      try {
        post_proc_mindis(Groups_from_index_file, Arguments, (*info_output),
                         MIN_DIS);
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] post_proc_mindis(): " << e.what() << "\n";
      } catch (...) {
        std::cerr << "[ERROR] post_proc_mindis() crashed!\n";
      }
    }
  }

  try {
    stat_hbonds(Hbonds_vector, (*info_output), Arguments, time_of_snap,
                input_line);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] stat_hbonds(): " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] stat_hbonds() crashed!\n";
  }

  if (Arguments.flag_h_bond == 1) {
    if (num_threads > 1) {
      (*executor).parallel_do(2, [&](size_t worker_id) {
        if (worker_id == 0) {
          stat_hb_res(res_hb_rec, (*info_output).get_res_list_hb_rec_filename());
        } else {
          stat_hb_res(res_hb_lg, (*info_output).get_res_list_hb_lg_filename());
        }
      });
    } else {
      try {
        stat_hb_res(res_hb_rec, (*info_output).get_res_list_hb_rec_filename());
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] stat_hb_res() rec: " << e.what() << "\n";
      } catch (...) {
        std::cerr << "[ERROR] stat_hb_res() rec crashed!\n";
      }

      try {
        stat_hb_res(res_hb_lg, (*info_output).get_res_list_hb_lg_filename());
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] stat_hb_res() lg: " << e.what() << "\n";
      } catch (...) {
        std::cerr << "[ERROR] stat_hb_res() lg crashed!\n";
      }
    }
  }
  return std::nullopt;
}

std::optional<int> EngineRun::finish_postproc() {

  // --- Post-loop analysis (src/postproc/) -----------------------------------
  start_postproc = std::chrono::steady_clock::now();

  start_pp_gbsa = std::chrono::steady_clock::now();
  run_gbsa_postproc(Arguments, tot_snap_allfiles, ml_extractor, (*global_gbsa_agg),
                    cas_analyzer, na_profiler, global_has_nucleic_acid,
                    (*info_output), log);
  time_pp_gbsa = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_gbsa);

  if (Arguments.hotspot.flag_hotspot_report) {
    try {
      TitanUI::print_module_header("CAS + EDecomp Hotspot Report");
      titan_hotspot::export_report(
          ".", titan_hotspot::make_config_from_arguments(Arguments,
                                                         (*info_output)));
      TitanUI::log_kv(log, "Hotspot Report",
                      (*info_output).get_hotspot_residue_ranking_filename());
      TitanUI::print_info("Hotspot report exported to " +
                          (*info_output).get_hotspot_residue_ranking_filename());
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Hotspot report: ") + e.what());
      return 1;
    }
  }

  start_pp_dssp = std::chrono::steady_clock::now();
  run_dssp_postproc(Arguments, dssp_mol1_analyzer, dssp_mol2_analyzer,
                    (*info_output), log);
  time_pp_dssp = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_dssp);

  start_pp_tica = std::chrono::steady_clock::now();
  graph_runtime.configure(Arguments);
  run_tica_msm_postproc(Arguments, dccm_analyzer, tica_analyzer, msm_analyzer,
                        (*executor), (*info_output), log, &validation_runtime);
  time_pp_tica = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_tica);

  dham_tpt_converged = false;
  start_pp_dham = std::chrono::steady_clock::now();
  run_dham_postproc(Arguments, msm_analyzer, (*executor), (*info_output), log,
                    dham_frame_weights, &dham_tpt_tpm, &dham_tpt_pi,
                    &dham_tpt_converged, &validation_runtime.dham_context);
  if (Arguments.dham.flag_dham &&
      Arguments.eq_weight_source == "dham" && !dham_tpt_converged) {
    TitanUI::print_error(
        "DHAM was selected as the required equilibrium weight source but did "
        "not converge. No successful analysis marker should be written.");
    return 1;
  }
  time_pp_dham = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_dham);

  start_pp_mbar = std::chrono::steady_clock::now();
  mbar_result =
      run_mbar_postproc(Arguments, msm_analyzer, &dccm_analyzer, (*executor),
                        (*info_output), log, &mbar_frame_weights,
                        &mbar_pca_frame_weights);
  if (Arguments.mbar.flag_mbar &&
      Arguments.eq_weight_source == "mbar" &&
      (!mbar_result.solved || !mbar_result.converged ||
       mbar_result.global_frame_weights.empty())) {
    TitanUI::print_error(
        "MBAR was selected as the required equilibrium weight source but did "
        "not produce converged frame weights. No successful analysis marker "
        "should be written.");
    return 1;
  }
  time_pp_mbar = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_mbar);

  equilibrium_weights =
      resolve_postproc_equilibrium_weights(Arguments, &dham_frame_weights,
                                           &mbar_frame_weights,
                                           &mbar_pca_frame_weights, log);
  run_msm_equilibrium_postproc(Arguments, msm_analyzer, (*info_output), log,
                               equilibrium_weights);

  start_pp_tpt = std::chrono::steady_clock::now();
  run_tpt_postproc(Arguments, msm_analyzer, (*executor), (*info_output), log,
                   lj_feature_files, &rms_analyzer, &dssp_mol1_analyzer,
                   qnative_analyzer.enabled() ? &qnative_analyzer : nullptr,
                   &dham_tpt_tpm, &dham_tpt_pi,
                   dham_tpt_converged, &equilibrium_weights,
                   &validation_runtime, &graph_runtime);
  time_pp_tpt = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_tpt);

  run_validation_postproc(Arguments, tica_analyzer, msm_analyzer, (*executor),
                          (*info_output), log, validation_runtime);

  start_pp_tica_fes = std::chrono::steady_clock::now();
  run_tica_fes_postproc(Arguments, tica_analyzer, msm_analyzer, (*executor),
                        (*info_output), log, equilibrium_weights,
                        &tica_fes_metrics);
  time_pp_tica_fes = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_tica_fes);

  run_umap_postproc(Arguments, msm_analyzer, umap_analyzer, (*executor),
                    (*info_output), log, equilibrium_weights);

  // Drain LLPS worker queue before finalize
  start_pp_llps = std::chrono::steady_clock::now();
  if (Arguments.llps.flag_llps && llps_analyzer)
    llps_analyzer->flush_async();
  run_llps_postproc(Arguments, llps_analyzer, (*executor), (*info_output), log);
  time_pp_llps = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_llps);

  if (!input_line.empty()) {
    if (Arguments.flag_xdr) {
      replica_traj_files = input_line;
    } else {
      replica_traj_files.reserve(input_line.size());
      for (const auto &stem : input_line)
        replica_traj_files.push_back(stem + ".xtc");
    }
  }

  start_pp_dccm = std::chrono::steady_clock::now();
  run_dccm_pca_postproc(Arguments, tot_snap_allfiles, dccm_analyzer,
                        (*executor), time_network, (*info_output), log,
                        equilibrium_weights, replica_traj_files,
                        &pca_fes_metrics, &graph_runtime);
  time_pp_dccm = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_dccm);

  run_mbar_projection_exports_postproc(Arguments, msm_analyzer, &dccm_analyzer,
                                       (*executor), (*info_output), log,
                                       mbar_result);
  run_awh_exploratory_postproc(Arguments, msm_analyzer, umap_analyzer,
                               &dccm_analyzer, (*info_output), log);
  {
    std::vector<std::string> fes_outputs;
    if (Arguments.msm.flag_tica && msm_analyzer && msm_analyzer->n_frames() > 0 &&
        msm_analyzer->n_components() >= 2) {
      fes_outputs.push_back((*info_output).get_tica_fes_2d_filename());
      fes_outputs.push_back(
          (*info_output).get_tica_cluster_representatives_filename());
      if (tica_fes_metrics.zts_ran)
        fes_outputs.push_back((*info_output).get_tica_zts_path_filename());
    }
    if (Arguments.flag_pca) {
      fes_outputs.push_back((*info_output).get_pca_fes_2d_filename());
      if (pca_fes_metrics.zts_ran)
        fes_outputs.push_back((*info_output).get_pca_zts_path_filename());
    }
    if (!fes_outputs.empty()) {
      std::string joined = fes_outputs.front();
      for (size_t idx = 1; idx < fes_outputs.size(); ++idx)
        joined += "; " + fes_outputs[idx];
      TitanUI::log_kv(log, "2D FES Outputs", joined);
    }
  }

  start_pp_lspmd = std::chrono::steady_clock::now();
  run_lspmd_postproc(Arguments, dccm_analyzer, (*executor), (*info_output), log,
                     equilibrium_weights, &graph_runtime);
  time_pp_lspmd = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_lspmd);

  start_pp_graph = std::chrono::steady_clock::now();
  graph_runtime.export_all(Arguments, dccm_analyzer, (*executor), (*info_output), log);
  time_pp_graph = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_graph);

  start_pp_water = std::chrono::steady_clock::now();
  if (Arguments.n_wat >= 0) {
    TitanUI::print_module_header(
        "Multi-Trajectory Global Water Bridges Analysis");
    water_analyzer.export_global_summary(
        (*info_output).get_global_water_bridges_summary(),
        (*info_output).get_global_water_survival());
  }
  time_pp_water = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_water);

  start_pp_gist = std::chrono::steady_clock::now();
  if (Arguments.gist.flag_gist && (!gist_analyzer || gist_analyzer->processed_frames()==0)) {
    TitanUI::print_error("GIST processed no water frames; refusing a successful empty result.");
    return 1;
  }
  if (Arguments.gist.flag_gist && gist_analyzer) {
    TitanUI::print_module_header("GIST Water Thermodynamics");
    gist_analyzer->export_dx_maps((*info_output).get_gist_prefix(),
                                  static_cast<int>(tot_snap_allfiles),
                                  Arguments.gist.gist_min_count);
  }
  time_pp_gist = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_gist);

  // --- LLPS DBSCAN export ---
  if (dbscan_llps) {
    TitanUI::print_module_header("LLPS DBSCAN Cluster Analysis");
    dbscan_llps->export_timeseries(
        (*info_output).get_llps_dbscan_timeseries_filename());
    TitanUI::log_section(log, "LLPS DBSCAN Output Audit");
    TitanUI::log_kv(log, (*info_output).get_llps_dbscan_timeseries_filename(),
                    "emitted");
    TitanUI::log_kv(log, (*info_output).get_llps_dbscan_droplets_filename(),
                    "emitted");
    TitanUI::log_kv(log, (*info_output).get_llps_droplet_shape_timeseries_filename(),
                    "emitted");
    TitanUI::log_kv(log, "LLPS Dynamics Outputs",
                    Arguments.llps.llps_dynamics ? "enabled" : "skipped");
    if (Arguments.llps.llps_dynamics) {
      dbscan_llps->finalize_dynamics((*executor));
      dbscan_llps->export_dynamics(
          (*info_output), log,
          llps_analyzer ? &llps_analyzer->contact_dynamics() : nullptr);
    }
    if (Arguments.llps.llps_paper)
      dbscan_llps->export_paper_outputs((*info_output), log);
    dbscan_llps->export_performance_report((*info_output), (*executor));
    TitanUI::log_time(log, "LLPS DBSCAN", time_llps_dbscan.count() / 1000.0);
  }

  total_postproc = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_postproc);
  time_pp_zts = tica_fes_metrics.zts_time + pca_fes_metrics.zts_time;

  // --- Post-processing timing report ---
  TitanUI::log_section(log, "POST-PROCESSING TIMING");
  TitanUI::log_time(log, "GBSA/CAS/ML Postproc", time_pp_gbsa.count() / 1000.0);
  TitanUI::log_time(log, "DSSP Postproc", time_pp_dssp.count() / 1000.0);
  TitanUI::log_time(log, "TICA/MSM Solve+Export",
                    time_pp_tica.count() / 1000.0);
  TitanUI::log_time(log, "DHAM Reweighting", time_pp_dham.count() / 1000.0);
  TitanUI::log_time(log, "TPT Postproc", time_pp_tpt.count() / 1000.0);
  TitanUI::log_time(log, "TICA 2D FES Export",
                    time_pp_tica_fes.count() / 1000.0);
  if (tica_fes_metrics.zts_ran || pca_fes_metrics.zts_ran) {
    TitanUI::log_time(log, "ZTS MEP", time_pp_zts.count() / 1000.0);
  }
  TitanUI::log_time(log, "LLPS Flush+Postproc", time_pp_llps.count() / 1000.0);
  TitanUI::log_time(log, "DCCM/PCA Postproc", time_pp_dccm.count() / 1000.0);
  TitanUI::log_time(log, "MBAR Postproc", time_pp_mbar.count() / 1000.0);
  TitanUI::log_time(log, "LSP-MD Postproc", time_pp_lspmd.count() / 1000.0);
  TitanUI::log_time(log, "Graph Analytics", time_pp_graph.count() / 1000.0);
  TitanUI::log_time(log, "Water Bridges Export",
                    time_pp_water.count() / 1000.0);
  TitanUI::log_time(log, "GIST Thermodynamics", time_pp_gist.count() / 1000.0);
  log << "--------------------------------------------------------------------"
         "----------\n";
  TitanUI::log_time(log, "TOTAL POST-PROCESSING",
                    total_postproc.count() / 1000.0);

  titan_uncertainty::export_uncertainty_outputs(Arguments, (*info_output), log);

  perf_observer.export_outputs(
      (*info_output).get_titan_performance_summary_filename(),
      (*info_output).get_titan_performance_modules_filename(),
      (*info_output).get_titan_performance_frames_filename(), log);

  // Every module has written its outputs. A BIE frame that did not succeed
  // makes the run as a whole a failure, so it must not exit 0.
  pbe_failure =
      pbe_analyzer.enabled() ? pbe_analyzer.failure_summary() : std::string();
  if (!pbe_failure.empty()) {
    TitanUI::log_section(log, "TITAN ENGINE FINISHED WITH FAILED PBE FRAMES");
    log << pbe_failure << "\n";
    log.close();
    TitanUI::print_error(pbe_failure);
    return 1;
  }

  TitanUI::log_section(log, "TITAN ENGINE TERMINATED SUCCESSFULLY");
  TitanUI::print_info(
      "\n\033[1;32mTITAN ENGINE TERMINATED SUCCESSFULLY\033[0m");
  totalruntime = chrono::duration_cast<milliseconds>(
                            chrono::steady_clock::now() - total_run_time)
                            .count() /
                        1000.0;
  TitanUI::log_time(log, "GLOBAL TOTAL RUN TIME", totalruntime);
  TitanUI::log_message(
      log,
      "[TITAN] Historical note: TITAN is based on Universal Calculator and "
      "has used the TITAN name since 2026.");
  log << "\nContact: nguyenhoanglinh9@duytan.edu.vn\n";
  log.close();
  return 0;
  return std::nullopt;
}

} // namespace engine_detail
