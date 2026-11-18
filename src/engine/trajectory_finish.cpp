#include "engine/engine_run.h"

namespace engine_detail {

std::optional<int> EngineTrajectory::finish_trajectory() {

  consumed_reference_structural_frame = false;
  if (global_frame_counter == 0 && !Arguments.flag_xdr &&
      !saved_pro_frame0.empty()) {
    pro_in.m_mol = saved_pro_frame0;
    lig.m_mol = saved_lig_frame0;
    membrane_traj.m_mol = saved_membrane_frame0;

    if (dssp_mol1_analyzer.enabled() ||
        dssp_mol2_analyzer.enabled()) {
      auto start_dssp = std::chrono::steady_clock::now();
      if (dssp_mol1_analyzer.enabled())
        dssp_mol1_analyzer.process_frame(pro_in, 0, executor);
      if (dssp_mol2_analyzer.enabled() &&
          !lig.m_mol.empty())
        dssp_mol2_analyzer.process_frame(lig, 0, executor);
      time_dssp += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_dssp);
      consumed_reference_structural_frame = true;
    }

    const bool rms_reference_ready =
        Arguments.flag_nocal || rms_analyzer.mol2_trace_count() == 0 ||
        !lig.m_mol.empty();
    if (rms_analyzer.enabled() && rms_reference_ready) {
      auto start_rms = std::chrono::steady_clock::now();
      rms_analyzer.process_frame(pro_in, lig, 0, executor);
      time_rms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_rms);
      consumed_reference_structural_frame = true;
    }

    if (cmap_analyzer.enabled()) {
      auto start_cmap = std::chrono::steady_clock::now();
      const Multi_MOL *cmap_mol2 =
          Arguments.needs_mol2_for_cmap_interface() ? &lig : nullptr;
      cmap_analyzer.process_frame(pro_in, cmap_mol2, 0, executor);
      time_cmap += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_cmap);
      consumed_reference_structural_frame = true;
    }

    if (interface_analyzer.enabled()) {
      const auto start_interface = std::chrono::steady_clock::now();
      const Multi_MOL *interface_mol2 =
          Arguments.needs_mol2_for_interface_analysis() ? &lig : nullptr;
      try {
        interface_analyzer.process_frame(pro_in, interface_mol2, 0, 0,
                                         executor);
      } catch (const std::exception &e) {
        TitanUI::print_error(std::string("Interface analysis: ") +
                             e.what());
        return 1;
      }
      time_interface +=
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_interface);
      consumed_reference_structural_frame = true;
    }

    if (qnative_analyzer.enabled()) {
      auto start_qnative = std::chrono::steady_clock::now();
      const Multi_MOL *qnative_mol2 =
          Arguments.needs_mol2_for_qnative_interface() ? &lig : nullptr;
      qnative_analyzer.process_frame(pro_in, qnative_mol2, 0, executor);
      time_qnative += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_qnative);
      consumed_reference_structural_frame = true;
    }

    if (msd_analyzer.enabled()) {
      auto start_msd = std::chrono::steady_clock::now();
      const Multi_MOL *msd_mol2 =
          Arguments.needs_mol2_for_msd() ? &lig : nullptr;
      msd_analyzer.process_frame(pro_in, msd_mol2, 0, executor);
      time_msd += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_msd);
      consumed_reference_structural_frame = true;
    }

    if (dihedral_analyzer.enabled()) {
      auto start_dihedral = std::chrono::steady_clock::now();
      dihedral_analyzer.process_frame(pro_in, 0, executor);
      time_dihedral += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_dihedral);
      consumed_reference_structural_frame = true;
    }

    if (membrane_analyzer.enabled() && !membrane_traj.m_mol.empty()) {
      auto start_membrane = std::chrono::steady_clock::now();
      membrane_analyzer.process_frame(membrane_traj, &pro_in, 0, executor);
      time_membrane +=
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_membrane);
      consumed_reference_structural_frame = true;
    }

    if (s2_analyzer.enabled()) {
      auto start_s2 = std::chrono::steady_clock::now();
      s2_analyzer.process_frame(pro_in, 0, executor);
      time_s2 += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_s2);
      consumed_reference_structural_frame = true;
    }
  }

  if (pbe_analyzer.enabled() && global_frame_counter == 0 &&
      !Arguments.flag_xdr && !saved_pro_frame0.empty()) {
    auto start_pbe = std::chrono::steady_clock::now();
    const MOL *pbe_lig_frame =
        saved_lig_frame0.empty() ? nullptr : &saved_lig_frame0.front();
    const Multi_MOL_parameters *pbe_lig_params =
        pbe_lig_frame ? &lig.parameters : nullptr;
    pbe_analyzer.consume_frame(
        saved_pro_frame0.front(), pro_in.parameters, pbe_lig_frame,
        pbe_lig_params,
        static_cast<std::size_t>(input_name - Arguments.b_input + 1), 0,
        tot_snap_allfiles, executor, log,
        Arguments.pbe.pbsa ? &pbe_energy_provider : nullptr);
    time_pbe += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_pbe);
  }

  if (consumed_reference_structural_frame) {
    global_frame_counter = 1;
    TitanUI::log_kv(log, "Reference Frame Analysis",
                    "Processed frame 0 for single-frame structural modules");
  }

  tot_snap_allfiles += global_frame_counter;

  // --- IFP post-trajectory export ---
  // (aggregate across all files at the very end, handled after all-files loop)

  // Reset TICA lag buffer at trajectory boundary
  if (Arguments.msm.flag_tica && tica_analyzer) {
    tica_analyzer->reset_lag_buffer();
  }
  // Insert MSM trajectory break at each file boundary
  if (need_tica_projection_cache && msm_analyzer &&
      Arguments.msm.msm_auto_boundary) {
    msm_analyzer->add_trajectory_break();
    const int64_t msm_frame_count =
        (msm_analyzer->n_frames() > 0) ? msm_analyzer->n_frames()
                                       : msm_analyzer->pending_feature_frames();
    TitanUI::log_kv(log, "MSM trajectory break at frame",
                    std::to_string(msm_frame_count));
  }

  if (Arguments.n_wat >= 0 && !water_traj.m_mol.empty()) {
    water_analyzer.finalize_trajectory();
  }
  if (pbe_analyzer.enabled()) {
    pbe_analyzer.end_trajectory(log);
  }
  if (edecomp_mm_requested(Arguments) && edecomp_engine) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    try {
      const auto result = edecomp_engine->finalize_trajectory();
      titan_energy_decomp::export_result(
          result, base_dir,
          edecomp_output_suffix(Arguments, input_line, input_name),
          edecomp_output_unit(Arguments));
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Energy decomposition finalize: ") +
                           e.what());
      return 1;
    }
  }
  if (edecomp_gbsa_requested(Arguments) && edecomp_gbsa_engine) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    try {
      const auto result =
          edecomp_gbsa_engine->finalize_trajectory("mmgbsa");
      titan_energy_decomp::export_endpoint_result(
          result, base_dir, "mmgbsa",
          edecomp_output_suffix(Arguments, input_line, input_name),
          edecomp_output_unit(Arguments));
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Energy decomposition GBSA finalize: ") +
                           e.what());
      return 1;
    }
  }
  if (edecomp_pbsa_requested(Arguments) && edecomp_pbsa_engine) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    try {
      const auto result =
          edecomp_pbsa_engine->finalize_trajectory("mmpbsa");
      titan_energy_decomp::export_endpoint_result(
          result, base_dir, "mmpbsa",
          edecomp_output_suffix(Arguments, input_line, input_name),
          edecomp_output_unit(Arguments));
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Energy decomposition PBSA finalize: ") +
                           e.what());
      return 1;
    }
  }
  dssp_mol1_analyzer.finalize_trajectory();
  dssp_mol2_analyzer.finalize_trajectory();
  if (lj_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    const std::string lj_timeseries_out =
        base_dir + "/" +
        info_output.get_lj_timeseries_filename(Arguments, input_line,
                                               input_name);
    const std::string lj_rho_out =
        base_dir + "/" +
        info_output.get_lj_rho_r_filename(Arguments, input_line, input_name);
    const std::string lj_feature_out =
        base_dir + "/" +
        info_output.get_lj_features_filename(Arguments, input_line,
                                             input_name);
    const std::string lj_summary_out =
        base_dir + "/" +
        info_output.get_lj_summary_filename(Arguments, input_line,
                                            input_name);
    lj_analyzer.finalize_trajectory(lj_timeseries_out, lj_rho_out,
                                    lj_feature_out,
                                    lj_summary_out, log);
    lj_feature_files.push_back(lj_feature_out);
  }
  if (sq_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    sq_analyzer.finalize_trajectory(base_dir, info_output, Arguments, input_line,
                                    input_name, log);
  }
  if (cmap_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    const auto start_cmap_export = std::chrono::steady_clock::now();
    cmap_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                      input_line, input_name, log);
    perf_observer.add_export_elapsed(
        "CMap", elapsed_ms_since(start_cmap_export));
  }
  if (interface_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    const auto start_interface_export = std::chrono::steady_clock::now();
    try {
      interface_analyzer.finalize_trajectory(
          base_dir, info_output, Arguments, input_line, input_name, log);
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Interface analysis export: ") +
                           e.what());
      return 1;
    }
    perf_observer.add_export_elapsed(
        "Interface", elapsed_ms_since(start_interface_export));
  }
  if (qnative_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    const auto start_qnative_export = std::chrono::steady_clock::now();
    qnative_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                         input_line, input_name, log);
    perf_observer.add_export_elapsed(
        "QNative", elapsed_ms_since(start_qnative_export));
  }
  if (msd_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    const auto start_msd_export = std::chrono::steady_clock::now();
    try {
      msd_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                       input_line, input_name, executor, log);
    } catch (const std::exception &e) {
      TitanUI::print_error(e.what());
      return 1;
    }
    perf_observer.add_export_elapsed("MSD",
                                     elapsed_ms_since(start_msd_export));
  }
  if (dihedral_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    const auto start_dihedral_export = std::chrono::steady_clock::now();
    dihedral_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                          input_line, input_name, log);
    perf_observer.add_export_elapsed(
        "Dihedral", elapsed_ms_since(start_dihedral_export));
  }
  if (feature_router.active()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    feature_router.export_outputs(
        base_dir + "/" +
            info_output.get_features_by_frame_filename(Arguments, input_line,
                                                       input_name),
        base_dir + "/" +
            info_output.get_features_metadata_filename(Arguments, input_line,
                                                       input_name),
        base_dir + "/" +
            info_output.get_feature_export_summary_filename(
                Arguments, input_line, input_name),
        log);
  }
  if (membrane_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    const auto start_membrane_export = std::chrono::steady_clock::now();
    membrane_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                          input_line, input_name, log);
    perf_observer.add_export_elapsed(
        "Membrane", elapsed_ms_since(start_membrane_export));
  }
  if (s2_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    const auto start_s2_export = std::chrono::steady_clock::now();
    s2_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                    input_line, input_name, executor, log);
    perf_observer.add_export_elapsed("S2",
                                     elapsed_ms_since(start_s2_export));
  }
  if (hbk_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    hbk_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                     input_line, input_name, log);
  }
  if (rg_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    const std::string rg2_out =
        base_dir + "/" +
        info_output.get_rg2_filename(Arguments, input_line, input_name);
    const std::string rg2_summary_out =
        base_dir + "/" +
        info_output.get_rg2_summary_filename(Arguments, input_line, input_name);
    const std::string rg2_hist_out =
        base_dir + "/" +
        info_output.get_rg2_hist_filename(Arguments, input_line, input_name);
    const std::string traj_label =
        (!input_line.empty() && traj_index < input_line.size())
            ? input_line[traj_index]
            : ("traj_" + std::to_string(traj_index));
    rg_analyzer.finalize_trajectory(rg2_out, rg2_summary_out, rg2_hist_out,
                                    traj_index, traj_label, log);
  }

  if (rms_analyzer.enabled()) {
    const std::size_t traj_index =
        input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
    const std::size_t path_index =
        (Arguments.flag_separation_results && traj_index < paths.get_size())
            ? traj_index
            : 0;
    const std::string base_dir = paths.get_path(path_index);
    const std::string rmsd_out =
        base_dir + "/" +
        info_output.get_rmsd_filename(Arguments, input_line, input_name);
    const std::string rmsf_out =
        base_dir + "/" +
        info_output.get_rmsf_filename(Arguments, input_line, input_name);
    rms_analyzer.finalize_trajectory(rmsd_out, rmsf_out, log);
  }

  if (end_cscd_trajectory(Arguments, cscd_out_dis, paths, input_name) != 0)
    return 1;
  try {
    nocal_analyses.end_trajectory(Arguments);
  } catch (const std::exception &e) {
    TitanUI::print_error(e.what());
    return 1;
  }
  return std::nullopt;
}

std::optional<int> EngineTrajectory::report_trajectory() {

  start_print_force = std::chrono::steady_clock::now();
  if (Arguments.flag_cal_force == true && Arguments.print_force_res == true) {
    print_force_res(Force_res_mol1, info_output.get_group_1_name(), paths,
                    input_name - Arguments.b_input, Arguments);
    print_force_res(Force_res_mol2, info_output.get_group_2_name(), paths,
                    input_name - Arguments.b_input, Arguments);
  }
  time_print_force += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_print_force);

  start_maxforce = std::chrono::steady_clock::now();
  if (Arguments.flag_maxforce == true) {
    find_maxforce_res(Force_res_mol1, info_output.get_group_1_name(), paths,
                      input_name - Arguments.b_input, Arguments,
                      Force_res_stat_mol1);
    find_maxforce_res(Force_res_mol2, info_output.get_group_2_name(), paths,
                      input_name - Arguments.b_input, Arguments,
                      Force_res_stat_mol2);
  }
  time_maxforce += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_maxforce);

  if (Arguments.flag_h_bond == 1) {
    hb_lg_1traj.snap_num = global_frame_counter;
    hb_rec_1traj.snap_num = global_frame_counter;
    res_hb_lg.emplace_back(hb_lg_1traj);
    res_hb_rec.emplace_back(hb_rec_1traj);
  }

  if (Arguments.flag_h_bond == 1) {
    auto start_hb_flush = std::chrono::steady_clock::now();
    finalize_hbond_output_files();
    time_hb += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_hb_flush);
  }

  if (Arguments.opdb) {
    auto start_pdb_flush = std::chrono::steady_clock::now();
    finalize_pdb_output_stream();
    time_pdb += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_pdb_flush);
  }

  /**************************************************************************/
  TitanUI::log_section(log, "PERFORMANCE & TIMING REPORT");
  total_traj_time = std::chrono::duration_cast<milliseconds>(
                                 std::chrono::steady_clock::now() -
                                 trajectory_timing_start)
                                 .count() /
                             1000.0;
  TitanUI::log_kv(log, "Timing Scope", "current trajectory/input");
  input_init_time =
        (time_input_init - time_input_init_base).count() / 1000.0;
  input_stream_time =
        (time_input_stream - time_input_stream_base).count() / 1000.0;
  input_init_label =
        Arguments.flag_xdr ? "Reference Parse + XDR Init" : "Input Init (PDB)";
  input_stream_label =
        Arguments.flag_xdr ? "Trajectory Streaming (XDR)"
                           : "Trajectory Streaming (PDB)";
  TitanUI::log_time(log, input_init_label, input_init_time);
  TitanUI::log_time(log, input_stream_label, input_stream_time);
  TitanUI::log_time(log, "Solving PBC",
                    (time_solve_pbc - time_solve_pbc_base).count() / 1000.0);
  TitanUI::log_time(log, "GBSA Calculation",
                    (time_gbsa - time_gbsa_base).count() / 1000.0);
  TitanUI::log_time(log, "PBE Calculation",
                    (time_pbe - time_pbe_base).count() / 1000.0);
  TitanUI::log_time(log, "CAS Calculation",
                    (time_cas - time_cas_base).count() / 1000.0);
  TitanUI::log_time(log, "DSSP Analysis",
                    (time_dssp - time_dssp_base).count() / 1000.0);
  TitanUI::log_time(log, "HBond Kinetics Analysis",
                    (time_hbk - time_hbk_base).count() / 1000.0);
  TitanUI::log_time(log, "LJ Analysis",
                    (time_lj - time_lj_base).count() / 1000.0);
  TitanUI::log_time(log, "SQ Analysis",
                    (time_sq - time_sq_base).count() / 1000.0);
  TitanUI::log_time(log, "CMap Analysis",
                    (time_cmap - time_cmap_base).count() / 1000.0);
  TitanUI::log_time(
      log, "Interface Analysis",
      (time_interface - time_interface_base).count() / 1000.0);
  TitanUI::log_time(log, "Dihedral Analysis",
                    (time_dihedral - time_dihedral_base).count() / 1000.0);
  TitanUI::log_time(log, "QNative Analysis",
                    (time_qnative - time_qnative_base).count() / 1000.0);
  TitanUI::log_time(log, "MSD Analysis",
                    (time_msd - time_msd_base).count() / 1000.0);
  TitanUI::log_time(log, "Membrane Analysis",
                    (time_membrane - time_membrane_base).count() / 1000.0);
  TitanUI::log_time(log, "S2 Analysis",
                    (time_s2 - time_s2_base).count() / 1000.0);
  TitanUI::log_time(log, "RG2 Analysis",
                    (time_rg - time_rg_base).count() / 1000.0);
  TitanUI::log_time(log, "RMSD/RMSF Analysis",
                    (time_rms - time_rms_base).count() / 1000.0);
  TitanUI::log_time(log, "Network Data Collection",
                    (time_network - time_network_base).count() / 1000.0);
  TitanUI::log_time(log, "Standard Elec+vdW Calc",
                    (time_cal - time_cal_base).count() / 1000.0);
  TitanUI::log_time(log, "Sidechain Contact",
                    (time_sc - time_sc_base).count() / 1000.0);
  TitanUI::log_time(log, "Sidechain CSCD",
                    (time_cscd - time_cscd_base).count() / 1000.0);
  TitanUI::log_time(log, "Hydrogen Bonds",
                    (time_hb - time_hb_base).count() / 1000.0);
  TitanUI::log_time(log, "PDB Output",
                    (time_pdb - time_pdb_base).count() / 1000.0);
  TitanUI::log_time(log, "Force Map (FMAP)",
                    (time_fmap - time_fmap_base).count() / 1000.0);
  TitanUI::log_time(log, "Contact Map & Dis2Mol",
                    (time_other - time_other_base).count() / 1000.0);
  TitanUI::log_time(log, "Water Bridges Analysis",
                    (time_water - time_water_base).count() / 1000.0);
  TitanUI::log_time(log, "Subgroups Calc",
                    (time_subgroups_cal - time_subgroups_cal_base).count() /
                        1000.0);
  TitanUI::log_time(log, "Nucleic Acid Profiler",
                    (time_na - time_na_base).count() / 1000.0);
  TitanUI::log_time(log, "LLPS Analysis",
                    (time_llps - time_llps_base).count() / 1000.0);
  TitanUI::log_time(log, "TICA/MSM Accumulation",
                    (time_tica - time_tica_base).count() / 1000.0);
  TitanUI::log_time(log, "IFP Analysis",
                    (time_ifp - time_ifp_base).count() / 1000.0);
  accounted_traj_time =
        input_init_time + input_stream_time +
        (time_solve_pbc - time_solve_pbc_base).count() / 1000.0 +
        (time_gbsa - time_gbsa_base).count() / 1000.0 +
        (time_pbe - time_pbe_base).count() / 1000.0 +
        (time_cas - time_cas_base).count() / 1000.0 +
        (time_dssp - time_dssp_base).count() / 1000.0 +
        (time_hbk - time_hbk_base).count() / 1000.0 +
        (time_lj - time_lj_base).count() / 1000.0 +
        (time_sq - time_sq_base).count() / 1000.0 +
        (time_cmap - time_cmap_base).count() / 1000.0 +
        (time_interface - time_interface_base).count() / 1000.0 +
        (time_dihedral - time_dihedral_base).count() / 1000.0 +
        (time_qnative - time_qnative_base).count() / 1000.0 +
        (time_msd - time_msd_base).count() / 1000.0 +
        (time_membrane - time_membrane_base).count() / 1000.0 +
        (time_s2 - time_s2_base).count() / 1000.0 +
        (time_rg - time_rg_base).count() / 1000.0 +
        (time_rms - time_rms_base).count() / 1000.0 +
        (time_network - time_network_base).count() / 1000.0 +
        (time_cal - time_cal_base).count() / 1000.0 +
        (time_sc - time_sc_base).count() / 1000.0 +
        (time_cscd - time_cscd_base).count() / 1000.0 +
        (time_hb - time_hb_base).count() / 1000.0 +
        (time_pdb - time_pdb_base).count() / 1000.0 +
        (time_fmap - time_fmap_base).count() / 1000.0 +
        (time_other - time_other_base).count() / 1000.0 +
        (time_water - time_water_base).count() / 1000.0 +
        (time_subgroups_cal - time_subgroups_cal_base).count() / 1000.0 +
        (time_na - time_na_base).count() / 1000.0 +
        (time_llps - time_llps_base).count() / 1000.0 +
        (time_tica - time_tica_base).count() / 1000.0 +
        (time_ifp - time_ifp_base).count() / 1000.0;
  unaccounted_traj_time =
        std::max(0.0, total_traj_time - accounted_traj_time);
  TitanUI::log_time(log, "Unaccounted / Orchestration",
                    unaccounted_traj_time);

  log << "-----------------------------------------------------------------"
         "--"
         "-------------\n";
  TitanUI::log_time(log, "TOTAL TRAJECTORY TIME", total_traj_time);
  std::cout << std::endl;
  //<< std::endl;
  //<< std::endl;
  if (check_end_cal(Arguments, input_line, input_name) == true) {
    return stop_trajectories();
  }

  std::cout << std::endl;
  return std::nullopt;
}

} // namespace engine_detail
