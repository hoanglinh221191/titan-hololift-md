#include "engine/engine_run.h"

namespace engine_detail {

std::optional<int> EngineTrajectory::init_modules() {

  // --- LLPS INITIALIZATION (after Martini topo parsed, chainnum set) ---
  if (Arguments.llps.flag_llps && !llps_analyzer) {
    // Build chain_start / chain_len from pro_in.info (chainnum assigned by
    // impl_martini)
    const auto &mol_info_vec = pro_in.info;
    const int n_info = static_cast<int>(mol_info_vec.size());
    int n_chains_llps = 0;
    for (int i = 0; i < n_info; ++i)
      n_chains_llps =
          std::max(n_chains_llps,
                   static_cast<int>(mol_info_vec[i].get_chainnum()) + 1);
    std::vector<int> llps_chain_start(n_chains_llps, -1);
    std::vector<int> llps_chain_len(n_chains_llps, 0);
    for (int i = 0; i < n_info; ++i) {
      int c = static_cast<int>(mol_info_vec[i].get_chainnum());
      if (c < 0 || c >= n_chains_llps)
        continue;
      if (llps_chain_start[c] < 0)
        llps_chain_start[c] = i;
      llps_chain_len[c]++;
    }
    llps_analyzer = std::make_unique<TitanLLPS>(
        n_chains_llps, llps_chain_start, llps_chain_len,
        Arguments.llps.llps_rcut, Arguments.llps.llps_perc_thr,
        Arguments.llps.llps_dimer_size, Arguments.llps.llps_contacts,
        Arguments.llps.llps_voxel_size, Arguments.llps.llps_dynamics,
        Arguments.llps.llps_dynamics_detailed,
        Arguments.llps.llps_prna_rcut, Arguments.llps.llps_kernel,
        Arguments.llps.llps_density);
    llps_analyzer->configure_phase({Arguments.llps.llps_phase_geometry,
        Arguments.llps.llps_phase_axis, Arguments.llps.llps_phase_dense,
        Arguments.llps.llps_phase_dilute_inner, Arguments.llps.llps_phase_dilute_outer});
    llps_analyzer->set_sticker_beads(mol_info_vec,
                                     Arguments.llps.llps_sticker_beads);
    if ((Arguments.mol_2_type.find("rna") != std::string::npos ||
         Arguments.mol_2_type.find("RNA") != std::string::npos) &&
        !lig.info.empty()) {
      llps_analyzer->set_rna_metadata(lig.info);
    }
    llps_analyzer->configure_trace_outputs(info_output);
    TitanUI::log_section(log, "LLPS Run Context");
    TitanUI::log_kv(log, "LLPS Enabled", "Yes");
    TitanUI::log_kv(log, "LLPS Chains", std::to_string(n_chains_llps));
    TitanUI::log_kv(log, "LLPS Macro-Nodes",
                    std::to_string(llps_analyzer->n_macronodes()));
    TitanUI::log_kv(log, "LLPS Dimer Size",
                    std::to_string(Arguments.llps.llps_dimer_size));
    TitanUI::log_kv(log, "LLPS Contact Rcut (nm)",
                    std::to_string(Arguments.llps.llps_rcut));
    TitanUI::log_kv(log, "LLPS Percolation Threshold",
                    std::to_string(Arguments.llps.llps_perc_thr));
    TitanUI::log_kv(log, "LLPS Density Mode",
                    Arguments.llps.llps_density);
    TitanUI::log_kv(log, "LLPS Kernel Policy",
                    Arguments.llps.llps_kernel);
    TitanUI::log_kv(log, "LLPS Dynamics",
                    Arguments.llps.llps_dynamics ? "enabled" : "disabled");
    TitanUI::log_kv(log, "LLPS Detailed Dynamics",
                    Arguments.llps.llps_dynamics_detailed ? "enabled"
                                                          : "disabled");
    TitanUI::log_kv(log, "LLPS Stickers",
                    Arguments.llps.llps_sticker_beads);
    // Start dedicated LLPS worker thread (pipeline mode)
    llps_analyzer->start_async();
  }

  // --- LLPS DBSCAN (bead-level morphology) ---
  if (Arguments.llps.flag_llps && Arguments.llps.llps_dbscan && !dbscan_llps) {
    dbscan_llps = std::make_unique<TitanLLPSCluster>();
    const std::string pdb_out =
        info_output.get_llps_dbscan_droplets_filename();
    const std::string csv_out =
        info_output.get_llps_dbscan_timeseries_filename();
    // [BUG 1 fix] Use user-supplied eps (Å) and min_pts; default 15 Å / 5
    // beads. Pass csv_out so data is flushed per-frame (crash-safe, ISSUE 4
    // fix).
    TitanUI::log_kv(log, "LLPS-DBSCAN eps",
                    std::to_string(Arguments.llps.llps_dbscan_eps) + " A");
    TitanUI::log_kv(log, "LLPS-DBSCAN min_pts",
                    std::to_string(Arguments.llps.llps_dbscan_minpts));
    dbscan_llps->init(Arguments.llps.llps_dbscan_eps,
                      Arguments.llps.llps_dbscan_minpts, pdb_out, csv_out);
    dbscan_llps->configure_dynamics(
        Arguments.llps.llps_dynamics, Arguments.llps.llps_dynamics_detailed,
        Arguments.llps.llps_maxlag, Arguments.llps.llps_lineage_overlap,
        std::max(1, Arguments.llps.llps_dimer_size),
        Arguments.llps.llps_msd_fit_start_ns,
        Arguments.llps.llps_msd_fit_end_ns);
    dbscan_llps->configure_liquidity(
        Arguments.llps.llps_rna_segment_beads,
        Arguments.llps.llps_msd_drift_correct,
        Arguments.llps.llps_mobile_displacement_nm,
        Arguments.llps.llps_mobile_lag_ns);
    std::string llps_mol2_type = Arguments.mol_2_type;
    std::transform(llps_mol2_type.begin(), llps_mol2_type.end(),
                   llps_mol2_type.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    const bool llps_has_rna =
        llps_mol2_type.find("rna") != std::string::npos && !lig.info.empty();
    const float prna_rcut =
        (Arguments.llps.llps_prna_rcut > 0.0f)
            ? Arguments.llps.llps_prna_rcut
            : Arguments.llps.llps_rcut;
    const float intra_rcut =
        (Arguments.llps.llps_rna_intra_rcut > 0.0f)
            ? Arguments.llps.llps_rna_intra_rcut
            : Arguments.llps.llps_rcut;
    TitanUI::log_kv(log, "LLPS-DBSCAN Output PDB", pdb_out);
    TitanUI::log_kv(log, "LLPS-DBSCAN Output CSV", csv_out);
    TitanUI::log_kv(log, "LLPS-Fuzzy",
                    Arguments.llps.llps_fuzzy ? "enabled" : "disabled");
    TitanUI::log_kv(log, "LLPS RNA Detected",
                    llps_has_rna ? "yes" : "no");
    TitanUI::log_kv(log, "LLPS Protein-RNA Rcut (nm)",
                    std::to_string(prna_rcut));
    TitanUI::log_kv(log, "LLPS RNA Intra Rcut (nm)",
                    std::to_string(intra_rcut));
    TitanUI::log_kv(log, "LLPS RNA Compaction Mode",
                    Arguments.llps.llps_rna_compaction);
    TitanUI::log_kv(log, "LLPS RNA Information Mode",
                    Arguments.llps.llps_rna_information);
    TitanUI::log_kv(log, "LLPS RNA Topology Mode",
                    Arguments.llps.llps_rna_topology);
    TitanUI::log_kv(log, "LLPS RNA Topology Detailed",
                    Arguments.llps.llps_rna_topology_detailed ? "enabled"
                                                              : "disabled");
    TitanUI::log_kv(log, "LLPS Energy Correlation",
                    Arguments.llps.llps_energy_correlation);
    TitanUI::log_kv(log, "LLPS Paper Bundle",
                    Arguments.llps.llps_paper ? "enabled" : "disabled");
    const bool llps_fractal_enabled =
        Arguments.llps.llps_paper_fractal == "auto"
            ? Arguments.llps.llps_paper
            : (Arguments.llps.llps_paper_fractal == "true" ||
               Arguments.llps.llps_paper_fractal == "yes" ||
               Arguments.llps.llps_paper_fractal == "on" ||
               Arguments.llps.llps_paper_fractal == "1");
    TitanUI::log_kv(log, "LLPS Paper Fractal",
                    llps_fractal_enabled ? "enabled" : "disabled");
    if (Arguments.llps.llps_paper) {
      TitanUI::log_kv(log, "LLPS Paper Profile Bin (nm)",
                      std::to_string(
                          Arguments.llps.llps_paper_profile_bin_nm));
      TitanUI::log_kv(log, "LLPS Paper Blocks",
                      Arguments.llps.llps_paper_blocks > 0
                          ? std::to_string(Arguments.llps.llps_paper_blocks)
                          : "auto");
      TitanUI::log_kv(log, "LLPS Paper Geometry",
                      Arguments.llps.llps_paper_geometry);
    }
    dbscan_llps->configure_rna_compaction(
        Arguments.llps.llps_rna_compaction, Arguments.llps.llps_dynamics,
        Arguments.llps.llps_dynamics_detailed, llps_has_rna,
        Arguments.llps.llps_sticker_beads, prna_rcut, intra_rcut,
        Arguments.llps.llps_rna_longrange_min);
    dbscan_llps->configure_rna_information(
        Arguments.llps.llps_rna_information, Arguments.llps.llps_dynamics,
        Arguments.llps.llps_dynamics_detailed, llps_has_rna,
        Arguments.llps.llps_rna_info_rg_bins,
        Arguments.llps.llps_rna_info_longrange_bins);
    dbscan_llps->configure_rna_topology(
        Arguments.llps.llps_rna_topology, Arguments.llps.llps_dynamics,
        Arguments.llps.llps_rna_topology_detailed, llps_has_rna,
        Arguments.llps.llps_rcut, prna_rcut,
        Arguments.llps.llps_rna_longrange_min,
        Arguments.llps.llps_rna_bridge_geom_min,
        Arguments.llps.llps_rna_patch_gap_max,
        Arguments.llps.llps_rna_clamp_require_loop, info_output);
    llps_energy_provider.configure(pro_in, lig, CachedData, Arguments,
                                   static_cast<double>(prna_rcut),
                                   prna_rcut > 0.0f);
    dbscan_llps->configure_energy_correlation(
        Arguments.llps.llps_energy_correlation, Arguments.llps.llps_dynamics,
        llps_has_rna, Arguments.flag_nocal, &llps_energy_provider,
        Arguments.llps.llps_energy_detailed, info_output);
    dbscan_llps->set_kernel_mode(Arguments.llps.llps_kernel);
    dbscan_llps->configure_paper(
        {Arguments.llps.llps_paper,
         Arguments.llps.llps_paper_profile_bin_nm,
         Arguments.llps.llps_paper_blocks,
         Arguments.llps.llps_paper_geometry,
         llps_fractal_enabled, Arguments.llps.llps_phase_axis});
    dbscan_llps->configure_trace_outputs(info_output);
    if (Arguments.llps.llps_fuzzy) {
      dbscan_llps->enable_fuzzy(true);
      TitanUI::print_info("[LLPS-Fuzzy] Fuzzy EM refinement enabled.");
    }
  }

  if (Arguments.pbe.pbsa) {
    const double pbsa_cutoff_nm = titan_resolve_sg_cutoff_nm(Arguments);
    pbe_energy_provider.configure(pro_in, lig, CachedData, Arguments,
                                  pbsa_cutoff_nm, pbsa_cutoff_nm > 0.0);
    TitanUI::log_kv(log, "MM/PBSA", "enabled");
    TitanUI::log_kv(log, "MM/PBSA Energy Source",
                    "PBE deltaG_solv + force-field elec/vdw + SASA IE");
    TitanUI::log_kv(
        log, "MM/PBSA Pair Cutoff",
        pbsa_cutoff_nm > 0.0 ? std::to_string(pbsa_cutoff_nm) + " nm"
                             : "none");
  }

  if (edecomp_mm_requested(Arguments)) {
    const double edecomp_cutoff_nm = titan_resolve_sg_cutoff_nm(Arguments);
    edecomp_energy_provider.configure(pro_in, lig, CachedData, Arguments,
                                      edecomp_cutoff_nm,
                                      edecomp_cutoff_nm > 0.0);
    if (!edecomp_engine) {
      edecomp_engine =
          std::make_unique<titan_energy_decomp::EnergyDecompositionEngine>(
              titan_energy_decomp::make_axes(pro_in, "mol1"),
              titan_energy_decomp::make_axes(lig, "mol2"),
              make_edecomp_config(Arguments));
    }
    TitanUI::log_kv(log, "Energy Decomposition",
                    "MM elec+vdW enabled");
    TitanUI::log_kv(
        log, "Energy Decomposition Pair Cutoff",
        edecomp_cutoff_nm > 0.0
            ? std::to_string(edecomp_cutoff_nm) + " nm fallback cutoff"
            : "none");
  }

  if (edecomp_gbsa_requested(Arguments) && !edecomp_gbsa_engine) {
    edecomp_gbsa_engine =
        std::make_unique<titan_energy_decomp::EndpointEntityDecompositionEngine>(
            titan_energy_decomp::make_axes(pro_in, "mol1"),
            titan_energy_decomp::make_axes(lig, "mol2"),
            make_edecomp_config(Arguments));
    TitanUI::log_kv(log, "Energy Decomposition",
                    "MM/GBSA endpoint maps enabled");
  }

  if (edecomp_pbsa_requested(Arguments) && !edecomp_pbsa_engine) {
    edecomp_pbsa_engine =
        std::make_unique<titan_energy_decomp::EndpointEntityDecompositionEngine>(
            titan_energy_decomp::make_axes(pro_in, "mol1"),
            titan_energy_decomp::make_axes(lig, "mol2"),
            make_edecomp_config(Arguments));
    pbe_analyzer.set_edecomp_sink(
        [&](const std::vector<TitanPBE::EDecompRow> &rows) {
          if (!edecomp_pbsa_engine)
            return;
          edecomp_pbsa_engine->process_frame(
              make_pbsa_endpoint_rows(rows, *edecomp_pbsa_engine));
        });
    TitanUI::log_kv(log, "Energy Decomposition",
                    "MM/PBSA endpoint maps enabled");
  }

  // --- GIST Water Thermodynamics Initialization ---
  if (Arguments.gist.flag_gist) {
    // The group index must be resolved before topology/coordinate loading.
    if (Arguments.n_wat < 0 || water_traj.info.empty() || water_traj.info.size()%3) {
      TitanUI::print_error("GIST has no complete water topology; select a full O,H,H water group with -n_wat.");
      return 1;
    }
    for (size_t i=0;i<water_traj.info.size();++i) {
      const auto name=water_traj.info[i].get_atom();
      const bool oxygen=(name=="O" || name=="OW" || name=="OH2");
      const bool hydrogen=!name.empty() && name[0]=='H';
      if ((i%3==0 && !oxygen) || (i%3!=0 && !hydrogen)) {
        TitanUI::print_error("GIST water group must be ordered oxygen, hydrogen, hydrogen for each molecule.");
        return 1;
      }
    }

    if (!gist_analyzer && !lig.m_mol.empty() &&
        lig.m_mol[0].get_pos_x().size() > 0) {
      gist_analyzer = std::make_unique<TitanGIST>();
      gist_analyzer->init(
          lig.m_mol[0], Arguments.gist.gist_padding,
          Arguments.gist.gist_resolution, Arguments.gist.gist_temperature,
          Arguments.gist.gist_cutoff, Arguments.nt > 0 ? Arguments.nt : 1,
          Arguments.gist.gist_water_model, pro_in.parameters, pro_in.info,
          lig.parameters, lig.info);
      // BUG2: pass DCCM's exact Cα indices for TICA centroid
      if (!dccm_analyzer.get_ca_indices().empty())
        gist_analyzer->set_tica_ca_indices(dccm_analyzer.get_ca_indices());
      TitanUI::log_kv(log, "GIST Water Model",
                      Arguments.gist.gist_water_model);
      TitanUI::log_kv(log, "GIST Resolution",
                      std::to_string(Arguments.gist.gist_resolution) + " A");
      TitanUI::log_kv(log, "GIST Voxels",
                      std::to_string(gist_analyzer->n_voxels()));
      TitanUI::log_kv(log, "GIST Cutoff",
                      std::to_string(Arguments.gist.gist_cutoff) + " A");
    }
  }

  // --- MSM INITIALIZATION (after TICA init, shares n_components) ---
  if (need_tica_projection_cache && tica_analyzer && !msm_analyzer) {
    const double selected_dt_ps =
        Arguments.dt > 0 ? static_cast<double>(Arguments.dt)
                         : info_output.get_current_dt();
    const double msm_dt_ns =
        selected_dt_ps > 0.0 ? selected_dt_ps / 1000.0 : 0.001;
    msm_analyzer = std::make_unique<TitanMSM>(
        Arguments.msm.msm_clusters, tica_analyzer->n_components(), msm_dt_ns);
    if (Arguments.msm.flag_msm) {
      const std::string msm_cluster_cfg =
          Arguments.msm.msm_clusters_user_set
              ? std::to_string(Arguments.msm.msm_clusters)
              : "auto (default " +
                    std::to_string(Arguments.msm.msm_clusters) + ")";
      TitanUI::log_kv(log, "MSM Clusters", msm_cluster_cfg);
      TitanUI::log_kv(log, "MSM Lag",
                      std::to_string(Arguments.msm.msm_lag) + " frames");
      TitanUI::log_kv(log, "MSM PCCA",
                      Arguments.msm.msm_run_pcca ? "enabled" : "disabled");
    } else {
      TitanUI::log_kv(log, "TICA Projection Cache",
                      "enabled for downstream modules");
    }
    TitanUI::log_kv(log, "TICA Projection Strategy",
                    "rebuild projected frames from TitanKinetics feature "
                    "cache after solve_tica()");
  }

  // Fix 5: hoist TICA workspace OUTSIDE streaming loops
  // (pairwise distance vector + Kabsch mobile/reference matrices — zero heap
  // per frame)
  // Fix 1: these are aliases to global variables hoisted outside while(true)
  // so all replicas share a single Kabsch reference frame.
  if (Arguments.msm.flag_tica && tica_analyzer) {
    tica_feat_dim_global = tica_analyzer->feature_dim();
    if (tica_feat_dim_global > 0 &&
        tica_features_global.size() != tica_feat_dim_global) {
      tica_features_global.resize(tica_feat_dim_global);
    }
    if (!tica_use_lj_features_global && tica_n_ca_global == 0) {
      tica_n_ca_global = static_cast<int>(dccm_analyzer.get_resnrs().size());
      if (tica_n_ca_global > 0) {
        tica_mobile_global.resize(tica_n_ca_global, 3);
        tica_ref_centered_global.resize(tica_n_ca_global, 3);
      }
    }
  }
  use_lj_tica_features = tica_use_lj_features_global;
  use_observable_tica_features =
        tica_use_observable_features_global;
  tica_or_export_needs_dccm =
        (Arguments.msm.flag_tica && !use_lj_tica_features &&
         feature_source_needs_dccm(Arguments,
                                   !use_observable_tica_features)) ||
        (feature_router.active() && !feature_router.consume_tica() &&
         feature_source_needs_dccm(Arguments, false));

  if (pbe_analyzer.enabled())
    append_module_label(resolved_modules, "PBE");
  if (Arguments.pbe.pbsa)
    append_module_label(resolved_modules, "MM/PBSA");
  if (Arguments.flag_cas)
    append_module_label(resolved_modules, "CAS");
  dssp_runtime_enabled =
        dssp_mol1_analyzer.enabled() || dssp_mol2_analyzer.enabled();
if (dssp_runtime_enabled) {
  append_module_label(resolved_modules, "DSSP");
} else if (dssp_mol1_requested || dssp_mol2_requested) {
  append_module_label(skipped_modules,
                      "DSSP(auto-skipped: no compatible protein backbone)");
}
if (lj_analyzer.enabled())
  append_module_label(resolved_modules, "LJ");
if (cmap_analyzer.enabled())
  append_module_label(resolved_modules, "CMap");
if (interface_analyzer.enabled())
  append_module_label(resolved_modules, "Interface");
if (dihedral_analyzer.enabled())
  append_module_label(resolved_modules, "Dihedral");
if (qnative_analyzer.enabled())
  append_module_label(resolved_modules, "QNative");
if (msd_analyzer.enabled())
  append_module_label(resolved_modules, "MSD");
if (membrane_analyzer.enabled())
  append_module_label(resolved_modules, "Membrane");
if (s2_analyzer.enabled())
  append_module_label(resolved_modules, "S2");
if (rg_analyzer.enabled())
  append_module_label(resolved_modules, "Rg2");
if (rms_analyzer.enabled())
  append_module_label(resolved_modules, "RMSD/RMSF");
if (Arguments.flag_network || Arguments.community.flag_community)
  append_module_label(resolved_modules, "DCCM/Network");
  if (Arguments.corr_knn.flag_corr_knn)
    append_module_label(resolved_modules, "Corr-kNN");
if (Arguments.community.flag_community)
  append_module_label(resolved_modules, "Community");
if (Arguments.flag_pca)
  append_module_label(resolved_modules, "PCA_FEL");
  if (Arguments.flag_tda)
    append_module_label(resolved_modules, "TDA");
  if (Arguments.msm.flag_tica && tica_analyzer)
    append_module_label(resolved_modules, "TICA");
  if (Arguments.msm.flag_msm && msm_analyzer)
    append_module_label(resolved_modules, "MSM");
  if (Arguments.umap.flag_umap)
    append_module_label(resolved_modules, "UMAP");
  if (Arguments.dham.flag_dham && msm_analyzer)
    append_module_label(resolved_modules, "DHAM");
  if (Arguments.tpt.flag_tpt && msm_analyzer)
    append_module_label(resolved_modules, "TPT");
  if (Arguments.flag_h_bond)
    append_module_label(resolved_modules, "H-Bonds");
  if (Arguments.n_wat >= 0)
    append_module_label(resolved_modules, "Water");
  if (Arguments.flag_allostery)
    append_module_label(resolved_modules, "Allostery");
  if (Arguments.lspmd.flag_lspmd)
    append_module_label(resolved_modules, "LSP-MD");

  if (!resolved_modules.empty() || !skipped_modules.empty()) {
    TitanUI::log_section(log, "RESOLVED MODULES");
    TitanUI::log_kv(log, "Active Features",
                    resolved_modules.empty() ? "None" : resolved_modules);
  }

  if (pbe_analyzer.enabled())
    append_module_label(streaming_modules, "PBE");
  if (Arguments.pbe.pbsa)
    append_module_label(streaming_modules, "MM/PBSA");
  if (Arguments.flag_gbsa)
    append_module_label(streaming_modules, "GBSA");
  if (Arguments.flag_cas)
    append_module_label(streaming_modules, "CAS");
  if (global_has_nucleic_acid)
    append_module_label(streaming_modules, "RNA-Profile");
  if (dssp_runtime_enabled)
    append_module_label(streaming_modules, "DSSP");
  if (lj_analyzer.enabled())
    append_module_label(streaming_modules, "LJ");
  if (sq_analyzer.enabled())
    append_module_label(streaming_modules, "SQ");
  if (cmap_analyzer.enabled())
    append_module_label(streaming_modules, "CMap");
  if (dihedral_analyzer.enabled())
    append_module_label(streaming_modules, "Dihedral");
  if (qnative_analyzer.enabled())
    append_module_label(streaming_modules, "QNative");
  if (msd_analyzer.enabled())
    append_module_label(streaming_modules, "MSD");
  if (membrane_analyzer.enabled())
    append_module_label(streaming_modules, "Membrane");
  if (s2_analyzer.enabled())
    append_module_label(streaming_modules, "S2");
  if (rg_analyzer.enabled())
    append_module_label(streaming_modules, "Rg2");
  if (rms_analyzer.enabled())
    append_module_label(streaming_modules, "RMSD/RMSF");
  if (Arguments.flag_network || Arguments.community.flag_community ||
      Arguments.corr_knn.flag_corr_knn ||
      Arguments.flag_pca || Arguments.lspmd.flag_lspmd) {
    append_module_label(streaming_modules, "DCCM/PCA");
  }
  if (Arguments.msm.flag_tica && tica_analyzer)
    append_module_label(streaming_modules, "TICA");
  if (Arguments.ifp.flag_ifp && ifp_analyzer)
    append_module_label(streaming_modules, "IFP");
  if (Arguments.awh.flag_awh_ifp && awh_ifp_analyzer)
    append_module_label(streaming_modules, "AWH-IFP");
  if (Arguments.llps.flag_llps)
    append_module_label(streaming_modules, "LLPS");
  if (edecomp_mm_requested(Arguments))
    append_module_label(streaming_modules, "EDecomp");
  if (edecomp_gbsa_requested(Arguments))
    append_module_label(streaming_modules, "EDecomp-GBSA");
  if (edecomp_pbsa_requested(Arguments))
    append_module_label(streaming_modules, "EDecomp-PBSA");
  if (streaming_modules.empty())
    streaming_modules = "I/O";
  if (!resolved_modules.empty() || !skipped_modules.empty()) {
    TitanUI::log_kv(log, "Streaming Features", streaming_modules);
    if (!skipped_modules.empty())
      TitanUI::log_kv(log, "Skipped Features", skipped_modules);
  }
  TitanUI::print_runtime_module_summary(
      active_modules, resolved_modules.empty() ? "None" : resolved_modules,
      skipped_modules, streaming_modules);

  if (pbe_analyzer.enabled()) {
    pbe_analyzer.begin_trajectory(
        static_cast<std::size_t>(input_name - Arguments.b_input + 1),
        progress_source_label(Arguments, name_pro));
  }
  if (edecomp_mm_requested(Arguments) && edecomp_engine) {
    edecomp_engine->begin_trajectory(
        progress_source_label(Arguments, name_pro));
  }
  if (edecomp_gbsa_requested(Arguments) && edecomp_gbsa_engine) {
    edecomp_gbsa_engine->begin_trajectory(
        progress_source_label(Arguments, name_pro));
  }
  if (edecomp_pbsa_requested(Arguments) && edecomp_pbsa_engine) {
    edecomp_pbsa_engine->begin_trajectory(
        progress_source_label(Arguments, name_pro));
  }

  {
    titan_nocal::Trajectory trajectory;
    trajectory.index = input_name >= Arguments.b_input
                           ? static_cast<size_t>(input_name - Arguments.b_input)
                           : 0;
    const bool own_folder = Arguments.flag_separation_results &&
                            trajectory.index < paths.get_size();
    trajectory.folder = paths.get_path(own_folder ? trajectory.index : 0);
    trajectory.first_in_folder = own_folder || trajectory.index == 0;
    trajectory.stem =
        std::filesystem::path(Arguments.flag_xdr && !Arguments.traj_file.empty()
                                  ? Arguments.traj_file
                                  : Arguments.pdb_name)
            .stem()
            .string();
    try {
      nocal_analyses.begin_trajectory(Arguments, info_output, pro_in,
                                      trajectory);
    } catch (const std::exception &e) {
      TitanUI::print_error(e.what());
      return 1;
    }
  }

  global_frame_counter = 0;
  is_streaming = true;
  NUM = 0;

  // ========================================================================
  // Save Frame 0 topology+coords for post-loop functions that need m_mol
  // (e.g. print_energy_pdbtype which writes B-factor PDB).
  // The streaming loop clears pro_in/lig.m_mol each iteration.
  // ========================================================================
  saved_pro_frame0 = pro_in.m_mol;
  saved_lig_frame0 = lig.m_mol;
  saved_membrane_frame0 = membrane_traj.m_mol;
  hb_rec_1traj.clear_frame();
  hb_lg_1traj.clear_frame();

  need_com_distance =
        (Arguments.com_dis_energy_2mol != 4) || (Arguments.com_dis_2mol != 0);
  need_distance_scan =
        Arguments.ifp.flag_ifp || Arguments.flag_cas || Arguments.flag_ml ||
        Arguments.cal_dis_sc_contact ||
        (Arguments.flag_tda && Arguments.flag_sidechain) ||
        (Arguments.flag_cal_force &&
         (Arguments.force_map == 1 || Arguments.force_map == 3)) ||
        (Arguments.flag_sub_groups && Arguments.flag_cal_mindis) ||
        need_com_distance;

  // ========================================================================
  // CHUNK PREFETCH (XDR only)
  // ========================================================================
  // XTC/TRR decode (decompress + float->double scatter) used to run on the
  // main thread with the worker pool idle, then analysis ran with the reader
  // idle. The reader now fills back buffers for chunk N+1 while chunk N is
  // analysed, and only the frame vectors are swapped, so the shared topology
  // in pro_in/lig/water_traj/membrane_traj is never touched or duplicated.
  //
  // Safety: read_next_chunk() is the sole streamer entry point inside the
  // loop, start_chunk_prefetch() only runs after the previous read has been
  // consumed by take_next_chunk(), and take_next_chunk() joins before any
  // further read, so exactly one thread uses the streamer at a time. The
  // worker is declared after the buffers it writes, so its destructor joins
  // before those buffers die on any early return.
  //
  // The PDB path keeps the synchronous read because its loader also
  // populates topology, not just frames.
  pro_next = Multi_MOL(force_f);
  lig_next = Multi_MOL(force_f);
  water_next = Multi_MOL(force_f);
  membrane_next = Multi_MOL(force_f);
  // TITAN_DISABLE_CHUNK_PREFETCH forces the previous serial reader. It exists
  // so the overlapped and serial paths can be diffed on the same binary, and
  // as an operator kill switch if a host ever misbehaves under the extra
  // reader thread. Both paths must produce identical output.
  chunk_prefetch_enabled =
        Arguments.flag_xdr && CHUNK_SIZE != std::numeric_limits<size_t>::max() &&
        std::getenv("TITAN_DISABLE_CHUNK_PREFETCH") == nullptr;
  return std::nullopt;
}

} // namespace engine_detail
