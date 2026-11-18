#include "engine/engine_run.h"
#include "titan_seeded_random.h"

namespace engine_detail {

std::optional<int> EngineTrajectory::prepare_analyzers() {
  if (Arguments.flag_xdr) {
    current_input_file = "PDB=" + Arguments.pdb_name;
    if (!Arguments.traj_file.empty())
      current_input_file += "; TRAJ=" + Arguments.traj_file;
  } else if (Arguments.flag_1_pdb) {
    current_input_file = Arguments.pdb_name;
  } else {
    current_input_file = "REC=" + name_pro + "; LIG=" + name_lig;
  }
  TitanUI::log_kv(log, "Current Input File", current_input_file);
  receptor_group_label =
        info_output.get_group_1_name() + " [" +
        std::to_string(Arguments.group_1) + "] (" +
        std::to_string(pro_in.info.size()) + " atoms)";
  ligand_group_label =
        info_output.get_group_2_name() + " [" +
        std::to_string(Arguments.group_2) + "] (" +
        std::to_string(lig.info.size()) + " atoms)";
  TitanUI::log_kv(log, "Receptor Group",
                  receptor_group_label);
  TitanUI::log_kv(log, "Ligand Group",
                  ligand_group_label);
  TitanUI::log_kv(log, "Receptor Charge", format_double(chg_pro) + " e");
  TitanUI::log_kv(log, "Ligand Charge", format_double(chg_lig) + " e");
  if (force_f == "MARTINI") {
    TitanUI::log_kv(log, "Martini LJ",
                    "pair-specific matrix ([ nonbond_params ])");
    TitanUI::log_kv(log, "Martini Electrostatics",
                    martini::electrostatics_summary(Arguments));
  }

  if (Arguments.flag_sub_groups && !sg_streaming_initialized) {
    sg_streaming_analyzer.init(Groups_from_index_file, pro_in, lig,
                               CachedData, Arguments, available_ram, &log);
    sg_streaming_initialized = true;
  }

  if (Arguments.de.flag_find_pulling_direct) {
    auto snapshot_status = prepare_pull_direct_snapshot(
        Arguments, DE_arguments, force_f, name_pro, name_lig, pro_in, lig,
        water_traj, Groups_from_index_file, amine, info_output, executor,
        num_threads, log);
    if (!snapshot_status) {
      TitanUI::print_error(snapshot_status.error());
      TitanUI::log_message(log, "[Pull-direct] " + snapshot_status.error());
      return 1;
    }
    return find_pull_direct(pro_in, lig, CachedData, DE_arguments, Arguments,
                            info_output, executor, Groups_from_index_file);
  }

  if (Arguments.flag_cas)
    cas_analyzer.init(pro_in, lig, Arguments.mol_2_type);
  dssp_allow_mol2 =
        !Arguments.flag_nocal ||
        Arguments.needs_mol2_for_interface_analysis() ||
        Arguments.needs_mol2_for_cmap_interface() ||
        Arguments.needs_mol2_for_qnative_interface() ||
        Arguments.needs_mol2_for_msd();
  dssp_mol1_requested =
        Arguments.dssp.flag_dssp &&
        TitanDSSPAnalyzer::supports_moltype(Arguments.mol_1_type);
  dssp_mol2_requested =
        Arguments.dssp.flag_dssp && dssp_allow_mol2 &&
        TitanDSSPAnalyzer::supports_moltype(Arguments.mol_2_type);
  dssp_mol1_compatible_residues =
        TitanDSSPAnalyzer::compatible_backbone_residue_count(pro_in.info);
  dssp_mol2_compatible_residues =
        TitanDSSPAnalyzer::compatible_backbone_residue_count(lig.info);
  dssp_mol1_has_backbone =
        dssp_mol1_compatible_residues >= 3;
  dssp_mol2_has_backbone =
        dssp_mol2_compatible_residues >= 3;

  if (dssp_mol1_requested && !dssp_mol1_has_backbone) {
    const std::string message =
        Arguments.lj.flag_lj
            ? "[DSSP] Auto-skipping mol1: no DSSP-compatible protein "
              "backbone detected (typical for LJ/monoatomic jobs)."
            : "[DSSP] Auto-skipping mol1: no DSSP-compatible protein "
              "backbone detected in topology.";
    TitanUI::print_info(message);
    TitanUI::log_message(log, message);
    TitanUI::log_kv(log, "DSSP mol1",
                    "auto-skipped (compatible backbone residues=" +
                        std::to_string(dssp_mol1_compatible_residues) +
                        ", selected atoms=" +
                        std::to_string(pro_in.info.size()) + ")");
  }
  if (dssp_mol1_requested && dssp_mol1_has_backbone) {
    try {
      dssp_mol1_analyzer.init(pro_in, "mol1");
    } catch (const std::exception &e) {
      TitanUI::print_error(e.what());
      return 1;
    }
    TitanUI::log_kv(log, "DSSP mol1 residues",
                    std::to_string(dssp_mol1_analyzer.residue_count()));
  }
  if (dssp_mol2_requested && !dssp_mol2_has_backbone) {
    const std::string message =
        "[DSSP] Auto-skipping mol2: no DSSP-compatible protein backbone "
        "detected in topology.";
    TitanUI::print_info(message);
    TitanUI::log_message(log, message);
    TitanUI::log_kv(log, "DSSP mol2",
                    "auto-skipped (compatible backbone residues=" +
                        std::to_string(dssp_mol2_compatible_residues) +
                        ", selected atoms=" +
                        std::to_string(lig.info.size()) + ")");
  }
  if (dssp_mol2_requested && dssp_mol2_has_backbone) {
    try {
      dssp_mol2_analyzer.init(lig, "mol2");
    } catch (const std::exception &e) {
      TitanUI::print_error(e.what());
      return 1;
    }
    TitanUI::log_kv(log, "DSSP mol2 residues",
                    std::to_string(dssp_mol2_analyzer.residue_count()));
  }
  if (Arguments.lj.flag_lj) {
    lj_analyzer.init(pro_in, Arguments, log);
  }
  if (Arguments.hbk.flag_hbk) {
    const bool water_stream_available =
        Arguments.n_wat >= 0 && !water_traj.m_mol.empty();
    const Multi_MOL *lig_ptr = lig.m_mol.empty() ? nullptr : &lig;
    hbk_analyzer.init(pro_in, lig_ptr, water_stream_available, Arguments, log);
  }
  if (Arguments.sq.flag_sq) {
    sq_analyzer.init(pro_in, Arguments, log);
  }
  if (Arguments.cmap.flag_cmap) {
    const Multi_MOL *cmap_mol2 =
        Arguments.needs_mol2_for_cmap_interface() ? &lig : nullptr;
    cmap_analyzer.init(pro_in, cmap_mol2, Arguments, log);
  }
  if (Arguments.interface_analysis.flag_interface_analysis) {
    try {
      const Multi_MOL *interface_mol2 =
          Arguments.needs_mol2_for_interface_analysis() ? &lig : nullptr;
      interface_analyzer.init(pro_in, interface_mol2, Arguments, log);
    } catch (const std::exception &e) {
      TitanUI::print_error(e.what());
      return 1;
    }
  }
  if (Arguments.dihedral.flag_dihedral) {
    try {
      dihedral_analyzer.init(pro_in, Arguments, log);
    } catch (const std::exception &e) {
      TitanUI::print_error(e.what());
      return 1;
    }
  }
  if (Arguments.qnative.flag_qnative) {
    Multi_MOL qnative_ref_pro(force_f), qnative_ref_lig(force_f),
        qnative_ref_water(force_f), qnative_ref_membrane(force_f);
    const Multi_MOL *qnative_ref_pro_ptr = &pro_in;
    const Multi_MOL *qnative_mol2 =
        Arguments.needs_mol2_for_qnative_interface() ? &lig : nullptr;
    if (Arguments.qnative.ref_explicit) {
      arguments qnative_ref_args = Arguments;
      qnative_ref_args.pdb_name = Arguments.qnative.ref_file;
      qnative_ref_args.flag_1_pdb = true;
      qnative_ref_args.flag_xdr = false;
      qnative_ref_args.traj_file.clear();
      const int ref_status = pre_proc_pdb(
          force_f, qnative_ref_pro, qnative_ref_lig, qnative_ref_water,
          qnative_ref_membrane, Groups_from_index_file, 0, qnative_ref_args,
          amine, info_output);
      if (ref_status != 0 || qnative_ref_pro.m_mol.empty()) {
        TitanUI::print_error("QNative: failed to load -qnative-ref " +
                             Arguments.qnative.ref_file);
        return 1;
      }
      if (qnative_ref_pro.info.empty() ||
          qnative_ref_pro.parameters.size() < qnative_ref_pro.info.size()) {
        TitanUI::print_error(
            "QNative: -qnative-ref molecule 1 topology/parameters are incomplete.");
        return 1;
      }
      if (Arguments.needs_mol2_for_qnative_interface()) {
        if (qnative_ref_lig.m_mol.empty()) {
          TitanUI::print_error(
              "QNative: -qnative-scope=interface requires molecule 2 in -qnative-ref.");
          return 1;
        }
        if (qnative_ref_lig.info.empty() ||
            qnative_ref_lig.parameters.size() < qnative_ref_lig.info.size()) {
          TitanUI::print_error(
              "QNative: -qnative-ref molecule 2 topology/parameters are incomplete.");
          return 1;
        }
        qnative_mol2 = &qnative_ref_lig;
      }
      qnative_ref_pro_ptr = &qnative_ref_pro;
    }
    try {
      qnative_analyzer.init(
          *qnative_ref_pro_ptr, qnative_mol2, Arguments, log,
          Arguments.qnative.ref_explicit ? &pro_in : nullptr,
          (Arguments.qnative.ref_explicit &&
           Arguments.needs_mol2_for_qnative_interface())
              ? &lig
              : nullptr);
    } catch (const std::exception &e) {
      TitanUI::print_error(e.what());
      return 1;
    }
  }
  if (Arguments.msd.flag_msd) {
    try {
      msd_analyzer.init(pro_in,
                        Arguments.needs_mol2_for_msd() ? &lig : nullptr,
                        Arguments, log);
    } catch (const std::exception &e) {
      TitanUI::print_error(e.what());
      return 1;
    }
  }
  if (Arguments.membrane.flag_membrane) {
    try {
      membrane_analyzer.init(membrane_traj, log);
    } catch (const std::exception &e) {
      TitanUI::print_error(e.what());
      return 1;
    }
  }
  if (Arguments.s2.flag_s2) {
    try {
      s2_analyzer.init(pro_in, Arguments, log);
    } catch (const std::exception &e) {
      TitanUI::print_error(e.what());
      return 1;
    }
  }
  if (Arguments.rg.flag_rg2) {
    rg_analyzer.init(pro_in, lig, Arguments, log);
  }
  if (Arguments.rms.flag_rms) {
    rms_analyzer.init(pro_in, lig, Arguments, log);
  }

  if (Arguments.flag_network || Arguments.flag_pca ||
      Arguments.corr_knn.flag_corr_knn) {
    if (!dccm_analyzer_initialized) {
      dccm_analyzer.init(pro_in, lig, Arguments);
      dccm_analyzer_initialized = true;
    }
    const bool needs_side_trace_cache =
        Arguments.flag_allostery ||
        (Arguments.corr_knn.flag_corr_knn &&
         Arguments.corr_knn.pair_use_contact_prior);
    TitanUI::log_kv(log, "Network/PCA Nodes",
                    std::to_string(dccm_analyzer.get_resnrs().size()));
    TitanUI::log_kv(
        log, "DCCM Trace Scope",
        Arguments.flag_nocal
            ? "mol1 only (-nocal; mol2 trace nodes disabled)"
            : "mol1 + mol2 (compatible trace nodes only)");
    TitanUI::log_kv(log, "DCCM Trace Cache",
                    dccm_analyzer.stores_full_trajectory()
                        ? (needs_side_trace_cache
                               ? "full CA + Cbeta/N trajectory"
                               : "full CA trajectory")
                        : "latest CA frame only (TICA streaming)");
  }
  // LSP-MD standalone: init DCCM data collector without DCCM/PCA
  if (Arguments.lspmd.flag_lspmd && !Arguments.flag_network &&
      !Arguments.flag_pca) {
    if (!dccm_analyzer_initialized) {
      dccm_analyzer.init(pro_in, lig, Arguments);
      dccm_analyzer_initialized = true;
    }
    const bool needs_side_trace_cache =
        Arguments.flag_allostery ||
        (Arguments.corr_knn.flag_corr_knn &&
         Arguments.corr_knn.pair_use_contact_prior);
    TitanUI::log_kv(log, "LSP-MD Nodes",
                    std::to_string(dccm_analyzer.get_resnrs().size()));
    TitanUI::log_kv(
        log, "DCCM Trace Scope",
        Arguments.flag_nocal
            ? "mol1 only (-nocal; mol2 trace nodes disabled)"
            : "mol1 + mol2 (compatible trace nodes only)");
    TitanUI::log_kv(log, "DCCM Trace Cache",
                    dccm_analyzer.stores_full_trajectory()
                        ? (needs_side_trace_cache
                               ? "full CA + Cbeta/N trajectory"
                               : "full CA trajectory")
                        : "latest CA frame only (TICA streaming)");
  }
  return std::nullopt;
}

std::optional<int> EngineTrajectory::init_streaming() {

  // --- STREAMING ARCHITECTURE INITIALIZATION ---
  streamer.emplace(Arguments, info_output, Groups_from_index_file,
                           force_f, executor, num_threads);

  stream_ok =
        (*streamer).open_streams(name_pro, name_lig, pro_in, lig, membrane_traj);
  if (!stream_ok) {
    TitanUI::print_error(stream_ok.error());
    return 1;
  }

  available_ram_bytes = get_available_ram();
  CHUNK_SIZE = 100;

  if (total_traj_bytes > 0 && available_ram_bytes > 0 &&
      total_traj_bytes <= available_ram_bytes * 0.5 && !Arguments.flag_xdr) {
    // File fits in memory AND is PDB — load everything at once
    CHUNK_SIZE = std::numeric_limits<size_t>::max();
    TitanUI::log_kv(log, "Memory Strategy",
                    "In-Memory Execution (File < 50% RAM)");
  } else if (Arguments.flag_xdr) {
    // XDR: stream conservatively. XTC/TRR frames are expanded into
    // double-precision coordinate buffers plus per-frame topology copies,
    // which makes GBSA especially memory-hungry on large systems.
    const size_t selected_atoms_per_frame =
        Groups_from_index_file.groups_atom_number(
            Groups_from_index_file.group_1_index()) +
        (!Arguments.flag_nocal
             ? Groups_from_index_file.groups_atom_number(
                   Groups_from_index_file.group_2_index())
             : 0) +
        (Arguments.n_wat >= 0
             ? Groups_from_index_file.groups_atom_number(Arguments.n_wat)
             : 0) +
        (Arguments.needs_membrane_coordinates() &&
                 Arguments.membrane.resolved_group >= 0
             ? Groups_from_index_file.groups_atom_number(
                   static_cast<size_t>(Arguments.membrane.resolved_group))
             : 0);

    // 3 coordinates * 8 bytes, multiplied by a safety factor to account for
    // frame objects, box/time metadata, copied topology fields, and transient
    // GBSA work buffers.
    const size_t coord_bytes =
        selected_atoms_per_frame * 3ULL * sizeof(double);
    const size_t frame_safety_factor = Arguments.flag_gbsa ? 8ULL : 4ULL;
    const size_t bytes_per_frame =
        std::max(coord_bytes * frame_safety_factor, size_t(1));

    const size_t target_chunk_bytes =
        Arguments.flag_gbsa ? (16ULL * 1024ULL * 1024ULL)
                            : (64ULL * 1024ULL * 1024ULL);
    const size_t ram_limited_chunk_bytes =
        available_ram_bytes > 0
            ? std::max<size_t>(8ULL * 1024ULL * 1024ULL,
                               available_ram_bytes /
                                   (Arguments.flag_gbsa ? 16ULL : 8ULL))
            : target_chunk_bytes;
    const size_t effective_chunk_budget =
        std::min(target_chunk_bytes, ram_limited_chunk_bytes);

    const size_t max_xdr_chunk = Arguments.flag_gbsa ? 4ULL : 64ULL;
    CHUNK_SIZE = std::max(
        size_t(1),
        std::min(max_xdr_chunk, effective_chunk_budget / bytes_per_frame));

    TitanUI::log_kv(log, "XDR Selected Atoms",
                    std::to_string(selected_atoms_per_frame));
    TitanUI::log_kv(log, "XDR Frame Memory Est",
                    std::to_string(bytes_per_frame / (1024ULL * 1024ULL)) +
                        " MB");
    TitanUI::log_kv(log, "Memory Strategy",
                    "XDR Streaming (Adaptive Chunk = " +
                        std::to_string(CHUNK_SIZE) + " frame(s), budget " +
                        std::to_string(effective_chunk_budget /
                                       (1024ULL * 1024ULL)) +
                        " MB)");
  } else {
    // Adaptive chunk: target ~256 MB per chunk, clamped [10, 5000] frames
    size_t atoms_per_frame = pro_in.info.size() + lig.info.size();
    size_t bytes_per_frame = std::max(atoms_per_frame * 80, size_t(1));
    constexpr size_t TARGET_CHUNK_BYTES = 256ULL * 1024 * 1024; // 256 MB
    CHUNK_SIZE =
        std::max(size_t(10), std::min(size_t(5000),
                                      TARGET_CHUNK_BYTES / bytes_per_frame));
    TitanUI::log_kv(log, "Memory Strategy",
                    "Out-of-Core Streaming (Adaptive Chunk = " +
                        std::to_string(CHUNK_SIZE) + " frames, " +
                        std::to_string(bytes_per_frame / 1024) +
                        " KB/frame)");
  }

  time_input_init += std::chrono::duration_cast<milliseconds>(
      std::chrono::steady_clock::now() - start_input_init);

  // --- TICA INITIALIZATION (after DCCM init, uses Cα count) ---
  if (Arguments.msm.flag_tica && !tica_analyzer) {
    int n_ca = static_cast<int>(dccm_analyzer.get_resnrs().size());
    const bool use_lj_tica_features =
        Arguments.feature_export.source == "lj" ||
        (!Arguments.feature_export.source_explicit && Arguments.lj.flag_lj &&
         lj_analyzer.phase2_ready() && n_ca == 0);
    if (Arguments.feature_export.source == "lj" &&
        !lj_analyzer.phase2_ready()) {
      TitanUI::print_error(
          "Feature export: -feature-source=lj requires active LJ phase-2 features.");
      return 1;
    }
    const bool use_observable_tica_features =
        feature_source_is_observable_tica(Arguments);
    std::string tica_feature_source =
        use_observable_tica_features
            ? ("observable feature source: " + Arguments.feature_export.source)
            : (use_lj_tica_features
                   ? "LJ condensation phase-2 features (density, cluster, CN, shell)"
                   : "pairwise C-alpha distances");
    // Science fix: pairwise inter-Cα distances (rotation+translation
    // invariant) D = N_ca*(N_ca-1)/2 — eliminates 6 spurious DOF from raw xyz
    std::vector<FeatureColumn> feature_columns;
    if (Arguments.feature_export.export_features ||
        use_observable_tica_features) {
      feature_columns = build_observable_feature_columns(
          Arguments, n_ca, qnative_analyzer, dihedral_analyzer,
          cmap_analyzer, lj_analyzer, use_lj_tica_features);
    }
    int tica_feature_dim = use_observable_tica_features
                               ? static_cast<int>(feature_columns.size())
                               : (use_lj_tica_features
                                      ? lj_analyzer.feature_dim()
                                      : n_ca * (n_ca - 1) / 2);
    if (use_observable_tica_features && tica_feature_dim <= 0) {
      TitanUI::print_error(
          "Feature export: selected -feature-source produced zero TICA features.");
      return 1;
    }
    const double selected_dt_ps =
        Arguments.dt > 0 ? static_cast<double>(Arguments.dt)
                         : info_output.get_current_dt();
    const double tica_dt_ns =
        selected_dt_ps > 0.0 ? selected_dt_ps / 1000.0 : 0.001;
    TicaBackend requested_backend = TicaBackend::Auto;
    if (!tica_backend_from_string(Arguments.msm.tica_backend,
                                  requested_backend)) {
      requested_backend = TicaBackend::Auto;
    }

    auto bytes_to_mb_string = [](long double bytes) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(1)
          << (bytes / (1024.0L * 1024.0L)) << " MB";
      return oss.str();
    };

    const size_t atoms_per_frame =
        std::max<size_t>(1, pro_in.info.size() + lig.info.size());
    long double bytes_per_frame_estimate =
        static_cast<long double>(atoms_per_frame) *
        static_cast<long double>(Arguments.flag_xdr ? 12.0 : 80.0);
    if (bytes_per_frame_estimate < 1.0L) {
      bytes_per_frame_estimate = 1.0L;
    }
    long double estimated_total_traj_bytes =
        static_cast<long double>(std::max<size_t>(1, total_traj_bytes)) *
        static_cast<long double>(std::max<size_t>(1, Arguments.number_of_files));
    TitanTicaFrameEstimateConfig frame_estimate_config;
    frame_estimate_config.begin_time_ps = Arguments.begin_time;
    frame_estimate_config.end_time_ps = Arguments.end_time;
    frame_estimate_config.dt_ps = static_cast<double>(Arguments.dt);
    frame_estimate_config.begin_set = Arguments.flag_begin_set;
    frame_estimate_config.end_set = Arguments.flag_end_set;
    frame_estimate_config.trajectory_count =
        input_line.empty() ? 0 : input_line.size();
    frame_estimate_config.number_of_files = Arguments.number_of_files;
    frame_estimate_config.fallback_estimated_frames =
        estimated_total_traj_bytes / bytes_per_frame_estimate;
    if (Arguments.flag_xdr && Arguments.xdr_type == "xtc")
      frame_estimate_config.xtc_paths = input_line;
    const TitanTicaFrameEstimate frame_estimate =
        estimate_tica_selected_frames(frame_estimate_config);
    long double estimated_frames =
        static_cast<long double>(std::max<std::size_t>(1, frame_estimate.frames));
    const int sketch_rank =
        std::max(Arguments.msm.tica_components, Arguments.msm.tica_rank);
    const long double dense_memory_estimate =
        6.0L * static_cast<long double>(tica_feature_dim) *
        static_cast<long double>(tica_feature_dim) *
        static_cast<long double>(sizeof(double));
    const long double dual_memory_estimate =
        3.0L * estimated_frames * static_cast<long double>(tica_feature_dim) *
        static_cast<long double>(sizeof(double));
    const long double randomized_memory_estimate =
        (estimated_frames * static_cast<long double>(sketch_rank + 8) +
         static_cast<long double>(tica_feature_dim) *
             static_cast<long double>(2 * (sketch_rank + 8))) *
        static_cast<long double>(sizeof(double));
    TitanTicaBackendPolicyConfig backend_policy_config;
    backend_policy_config.requested_backend = requested_backend;
    backend_policy_config.available_ram_bytes =
        static_cast<long double>(available_ram_bytes);
    backend_policy_config.ram_budget_fraction = 0.30L;
    backend_policy_config.dense_memory_estimate = dense_memory_estimate;
    backend_policy_config.dual_memory_estimate = dual_memory_estimate;
    backend_policy_config.estimated_frames = estimated_frames;
    backend_policy_config.feature_dim = tica_feature_dim;
    backend_policy_config.need_projection_cache = need_tica_projection_cache;
    const TitanTicaBackendPolicyDecision backend_decision =
        select_tica_backend_for_policy(backend_policy_config);
    const TicaBackend selected_backend = backend_decision.selected_backend;

    long double hard_memory_cap_bytes = backend_decision.ram_budget;
    if (hard_memory_cap_bytes <= 0.0L) {
      constexpr long double kExactDualPeakFactor = 3.0L;
      hard_memory_cap_bytes = backend_decision.large_matrix_value_limit *
                              sizeof(double) * kExactDualPeakFactor;
    }
    const size_t hard_memory_cap = static_cast<size_t>(std::min<long double>(
        hard_memory_cap_bytes,
        static_cast<long double>(std::numeric_limits<size_t>::max())));

    tica_analyzer = std::make_unique<TitanKinetics>(
        tica_feature_dim, Arguments.msm.tica_lag,
        Arguments.msm.tica_components, tica_dt_ns, selected_backend,
        Arguments.msm.tica_rank, hard_memory_cap,
        selected_backend == TicaBackend::ExactDual);
    tica_analyzer->set_random_seed(
        titan_random::solver_seed(Arguments.msm.msm_seed));
    tica_use_lj_features_global = use_lj_tica_features;
    tica_use_observable_features_global = use_observable_tica_features;
    if (Arguments.feature_export.export_features ||
        use_observable_tica_features) {
      if (feature_columns.empty()) {
        feature_columns = build_observable_feature_columns(
            Arguments, n_ca, qnative_analyzer, dihedral_analyzer,
            cmap_analyzer, lj_analyzer, use_lj_tica_features);
      }
      feature_router.configure(Arguments.feature_export.export_features,
                               use_observable_tica_features,
                               Arguments.feature_export.source);
      feature_router.begin_trajectory(std::move(feature_columns),
                                      Arguments.feature_export.source);
    }
    TitanUI::log_kv(log, "TICA Feature Source", tica_feature_source);
    TitanUI::log_kv(
        log, "TICA Trace Scope",
        use_observable_tica_features
            ? "observable feature router"
            : use_lj_tica_features
            ? "mol1 only (LJ feature backend)"
            : (Arguments.flag_nocal
                   ? "mol1 only (-nocal; mol2 ignored for DCCM/TICA)"
                   : "mol1 + mol2 trace nodes (if compatible)"));
    if (use_lj_tica_features) {
      TitanUI::log_kv(log, "TICA LJ Atoms",
                      std::to_string(lj_analyzer.atom_count()));
      TitanUI::log_kv(log, "TICA Feature Dim (LJ phase-2)",
                      std::to_string(tica_feature_dim));
    }
    if (!use_lj_tica_features && !use_observable_tica_features) {
    TitanUI::log_kv(log, "TICA Cα Atoms", std::to_string(n_ca));
      TitanUI::log_kv(log, "TICA Feature Dim (pairwise dist)",
                      std::to_string(tica_feature_dim));
    }
    if (use_observable_tica_features) {
      TitanUI::log_kv(log, "TICA Feature Dim (observable)",
                      std::to_string(tica_feature_dim));
    }
    TitanUI::log_kv(log, "TICA Lag Time",
                    std::to_string(Arguments.msm.tica_lag) + " frames");
    TitanUI::log_kv(log, "TICA Components",
                    std::to_string(Arguments.msm.tica_components));
    TitanUI::log_kv(log, "TICA Backend Requested",
                    tica_backend_to_string(requested_backend));
    TitanUI::log_kv(log, "TICA Backend Selected",
                    tica_backend_to_string(selected_backend));
    if (backend_decision.large_matrix_limit_exceeded) {
      TitanUI::log_kv(log, "TICA Large-Matrix Policy",
                      "exact covariance/replay selected by memory budget; approximation requires explicit opt-in");
    }
    TitanUI::log_kv(log, "TICA Rank",
                    std::to_string(Arguments.msm.tica_rank));
    TitanUI::log_kv(log, "TICA Estimated Frames",
                    std::to_string(static_cast<int64_t>(estimated_frames)));
    TitanUI::log_kv(log, "TICA Frame Estimate Source",
                    frame_estimate.source);
    if (!frame_estimate.diagnostic.empty()) {
      TitanUI::log_kv(log, "TICA Frame Scan Fallback",
                      frame_estimate.diagnostic);
      TitanUI::print_warning("TICA: " + frame_estimate.diagnostic +
                             "; using " + frame_estimate.source +
                             " estimate.");
    }
    TitanUI::log_kv(
        log, "TICA Raw Feature Values",
        std::to_string(static_cast<int64_t>(
            std::ceil(backend_decision.raw_feature_values))));
    TitanUI::log_kv(
        log, "TICA Large-Matrix Limit",
        std::to_string(static_cast<int64_t>(
            std::ceil(backend_decision.large_matrix_value_limit))));
    TitanUI::log_kv(log, "TICA Dense Memory Est",
                    bytes_to_mb_string(dense_memory_estimate));
    TitanUI::log_kv(log, "TICA Dual Cache Est",
                    bytes_to_mb_string(dual_memory_estimate));
    TitanUI::log_kv(log, "TICA Randomized Est",
                    bytes_to_mb_string(randomized_memory_estimate));
    TitanUI::log_kv(log, "TICA Hard Memory Cap",
                    bytes_to_mb_string(hard_memory_cap_bytes));
  }

  if (Arguments.msm.flag_tica && tica_analyzer &&
      (Arguments.feature_export.export_features ||
       tica_use_observable_features_global) &&
      input_name > Arguments.b_input) {
    const int n_ca = static_cast<int>(dccm_analyzer.get_resnrs().size());
    auto feature_columns = build_observable_feature_columns(
        Arguments, n_ca, qnative_analyzer, dihedral_analyzer, cmap_analyzer,
        lj_analyzer, tica_use_lj_features_global);
    feature_router.configure(Arguments.feature_export.export_features,
                             tica_use_observable_features_global,
                             Arguments.feature_export.source);
    feature_router.begin_trajectory(std::move(feature_columns),
                                    Arguments.feature_export.source);
  }

  if (!Arguments.msm.flag_tica && Arguments.feature_export.export_features) {
    const int n_ca = static_cast<int>(dccm_analyzer.get_resnrs().size());
    auto feature_columns = build_observable_feature_columns(
        Arguments, n_ca, qnative_analyzer, dihedral_analyzer, cmap_analyzer,
        lj_analyzer, false);
    if (feature_columns.empty()) {
      TitanUI::print_warning(
          "Feature export: no active frame feature producers were resolved.");
    } else {
      feature_router.configure(true, false, Arguments.feature_export.source);
      feature_router.begin_trajectory(std::move(feature_columns),
                                      Arguments.feature_export.source);
      TitanUI::log_kv(log, "Feature Export Dim",
                      std::to_string(feature_router.feature_dim()));
    }
  }
  return std::nullopt;
}

std::string EngineTrajectory::format_double(double v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

} // namespace engine_detail
