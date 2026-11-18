#include "engine/engine_run.h"

namespace engine_detail {

std::optional<int> EngineRun::setup() {
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_TEXT);
  _setmode(_fileno(stderr), _O_TEXT);
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);

  // KHÔNG bật ENABLE_VIRTUAL_TERMINAL_PROCESSING ở đây nữa
  // Windows Terminal tự bật VT cho mọi app hiện đại
  // Việc bật thủ công kích hoạt DA response từ ConPTY
#endif
  TitanUI::print_logo();
  Arguments.post_process();
  /************************************/
  if (Arguments.de.flag_find_pulling_direct == true)
    DE_arguments.parse_input(Arguments);
  paths.add_paths(Arguments, list_paths);
  if (paths.create_folders(Arguments) != 0)
    return 1;

  sc_flag = int{Arguments.flag_all_side_chain_ct};
  total_run_time = std::chrono::steady_clock::time_point{chrono::steady_clock::now()};
  force_f = string{Arguments.force_f};

  if (Arguments.flag_cal_force && Arguments.force_map > 0) {
    fstream lg_force_inter("force_lg.txt", ios::out);
    if (Arguments.mol_2_type.find("protein") != string::npos)
      lg_force_inter << "#Res" << setw(11) << "Electro." << setw(12)
                     << "Van der W." << '\n';
    else
      lg_force_inter << "#Atom/Res" << setw(11) << "Electro." << setw(12)
                     << "Van der W." << '\n';
    fstream rec_force_inter("force_rec.txt", ios::out);
    rec_force_inter << "#Res" << setw(12) << "Electro." << setw(12)
                    << "Van der W." << '\n';
    lg_force_inter.close();
    rec_force_inter.close();
  }

  /*************************************/
  d_standard = double{Arguments.d_standard};
  num_amine = int{};
  num_threads = size_t{Arguments.nt};
  tot_snap_allfiles = size_t{};
  executor.emplace(std::max<size_t>(1, num_threads));
  name_pro = string{};
  name_lig = string{};

  sg_streaming_initialized = bool{false};
  topo_2_is_open = bool{Arguments.flag_topo_2};
  topo_is_open = bool(Arguments.flag_topo_1);
  status = bool{true};

  info_output.emplace(Arguments);

  /************************************/
  /*************************************/
  log.open("log.txt", ios::out);
  log.close();
  log.open("log.txt", ios::out | ios::app);
  TitanUI::log_logo(log);
  /*************************************/
  /******************************************/

  if (Arguments.output_manifest.output_manifest ||
      Arguments.output_manifest.schema_export == "json") {
    auto manifest_rows =
        titan_output_manifest::build_output_manifest(Arguments, (*info_output));
    titan_output_manifest::export_output_manifest(
        manifest_rows, (*info_output).get_titan_outputs_manifest_filename(), &log);
    titan_output_manifest::export_run_config_summary(
        Arguments, manifest_rows,
        (*info_output).get_titan_run_config_summary_filename(), &log);
    if (Arguments.output_manifest.schema_export == "json") {
      titan_config_schema::export_schema_json(
          Arguments, (*info_output).get_titan_schema_filename(), &log);
    }
    if (Arguments.output_manifest.dry_run) {
      TitanUI::print_info(
          "Dry-run complete: wrote manifest/config/schema outputs and skipped trajectory processing.");
      TitanUI::log_message(
          log,
          "[DryRun] Exiting before coordinate loading and trajectory processing.");
      return 0;
    }
  }

  cal_out.open("cal.txt", ios::out);

  if (Arguments.com_dis_energy_2mol != 4) {
    for (size_t i{}; i < paths.get_size(); ++i) {
      fstream output_tot_dis(
          paths.get_path(i) + "/" + (*info_output).getDisCom2Mol(), ios::out);
      fstream output_elec_dis(paths.get_path(i) + "/electrostatic_" +
                                  (*info_output).getDisCom2Mol(),
                              ios::out);
      fstream output_vdw_dis(paths.get_path(i) + "/vanderWaals_" +
                                 (*info_output).getDisCom2Mol(),
                             ios::out);

      output_elec_dis.close();
      output_tot_dis.close();
      output_vdw_dis.close();
    }
  }

  if (Arguments.flag_h_bond == 1) {
    init_hbond_output_files(Arguments);
  }

  TitanUI::log_section(log, "TITAN ENGINE INITIALIZATION");
  TitanUI::log_kv(log, "Software Build", (*info_output).getBuild());
  TitanUI::log_kv(log, "Software Version", titan_build::product_version());
  TitanUI::log_kv(log, "Build Number",
                  titan_build::build_number_string());
  TitanUI::log_kv(log, "Git Commit", titan_build::git_commit());
  TitanUI::log_kv(log, "Git State", titan_build::git_state_name());
  TitanUI::log_kv(log, "Release Build",
                  titan_build::release_build() ? "yes" : "no");
  TitanUI::log_kv(log, "Source Date", titan_build::source_date_utc());
  if (std::string_view(titan_build::ci_build_number()).empty() == false) {
    TitanUI::log_kv(log, "CI Provider", titan_build::ci_provider());
    TitanUI::log_kv(log, "CI Build", titan_build::ci_build_number());
    TitanUI::log_kv(log, "CI Run ID", titan_build::ci_run_id());
  }
  TitanUI::log_kv(log, "Build Profile", titan_build::build_profile_name());
  TitanUI::log_kv(log, "Compiler", titan_build::compiler_name());
  TitanUI::log_kv(log, "Target OS", titan_build::target_os_name());
  TitanUI::log_kv(log, "Target Architecture",
                  titan_build::target_arch_name());
  TitanUI::log_kv(log, "C++23 Floor", titan_build::cxx23_floor_name());
  TitanUI::log_kv(log, "Compiled SIMD",
                  titan_build::simd_level_name(
                      titan_build::compiled_simd_level()));
  TitanUI::log_kv(log, "Runtime SIMD",
                  titan_build::simd_level_name(
                      titan_build::runtime_simd_level()));
  TitanUI::log_kv(log, "Runtime Supports Build SIMD",
                  titan_build::runtime_supports_compiled_simd() ? "yes"
                                                                 : "no");
  TitanUI::log_kv(log, "Runtime Dispatch",
                  titan_build::runtime_dispatch_mode_name());
  TitanUI::log_kv(log, "std::jthread",
                  TITAN_HAS_STD_JTHREAD ? "available" : "missing");

  timeRun = time_t{time(nullptr)};
  strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S",
           localtime(&timeRun));
  TitanUI::log_kv(log, "Execution Time", std::string(time_str));
  TitanUI::log_kv(log, "Force Field", force_f);
  TitanUI::log_kv(log, "Number of Threads", std::to_string(num_threads));

  TitanUI::log_section(log, "SYSTEM CONFIGURATION");
  TitanUI::log_kv(log, "Input File (-f)", Arguments.pdb_name);
  if (Arguments.flag_topo_1)
    TitanUI::log_kv(log, "Topology File (-topo)", Arguments.topo_name);
  if (Arguments.flag_topo_2)
    TitanUI::log_kv(log, "Topology File 2 (-topo2)", Arguments.topo_name_2);

  TitanUI::log_kv(log, "Molecule 1 Type", Arguments.mol_1_type);
  TitanUI::log_kv(log, "Molecule 2 Type", Arguments.mol_2_type);
  TitanUI::log_kv(log, "Trajectory Semantics",
                  Arguments.trajectory_semantics == "remd_state"
                      ? "remd-state (equilibrium/descriptive only)"
                      : "continuous");

  group_str = std::to_string(Arguments.group_1) + " - " +
                          std::to_string(Arguments.group_2);
  TitanUI::log_kv(log, "Interaction Groups (-g)", group_str);

  available_ram = get_available_ram();
  if (available_ram > 0) {
    TitanUI::log_kv(log, "Available RAM",
                    std::to_string(available_ram / (1024 * 1024)) + " MB");
  }

  TitanUI::log_kv(log, "Distance Cutoff (-dsc)",
                  format_double(Arguments.d_standard) + " A");

  if (Arguments.flag_gbsa) {
    TitanUI::log_section(log, "GBSA CONFIGURATION");
    TitanUI::log_kv(log, "IGB Model", std::to_string(Arguments.igb));
    TitanUI::log_kv(log, "Internal Dielectric",
                    format_double(Arguments.int_diel));
    TitanUI::log_kv(log, "External Dielectric",
                    format_double(Arguments.ext_diel));
    TitanUI::log_kv(log, "Salt Concentration",
                    format_double(Arguments.saltcon) + " M");
    TitanUI::log_kv(log, "GBSA Temperature",
                    format_double(Arguments.gbsa_temperature) + " K");
    TitanUI::log_kv(log, "GBSA SASA Points",
                    std::to_string(Arguments.gbsa_sasa_points));
    TitanUI::log_kv(log, "Born Model",
                    "Amber HCT/OBC analytical pair descreening");
    TitanUI::log_kv(log, "Salt Screening",
                    "Amber Debye kappa with 0.73 ion-exclusion factor");
    TitanUI::log_kv(log, "IE Minimum Samples",
                    std::to_string(Arguments.gbsa_ie_min_samples));
    TitanUI::log_kv(log, "IE Filter Sigma",
                    format_double(Arguments.gbsa_ie_filter_sigma));
  }
  if (Arguments.pbe.flag_pbe) {
    TitanUI::log_section(log, "PBE CONFIGURATION");
    TitanUI::log_kv(log, "Model", "linear PB / two-dielectric");
    TitanUI::log_kv(log, "Surface", Arguments.pbe.surface);
    TitanUI::log_kv(log, "Probe Radius",
                    format_double(Arguments.pbe.probe_radius) + " A");
    if (!Arguments.pbe.test_pqr.empty())
      TitanUI::log_kv(log, "Test PQR Adapter", Arguments.pbe.test_pqr);
    TitanUI::log_kv(log, "Scale", format_double(Arguments.pbe.scale) + " pt/A");
    TitanUI::log_kv(log, "Margin", format_double(Arguments.pbe.margin) + " A");
    TitanUI::log_kv(log, "Gridbox", Arguments.pbe.gridbox);
    TitanUI::log_kv(log, "Internal Dielectric",
                    format_double(Arguments.pbe.int_diel));
    TitanUI::log_kv(log, "External Dielectric",
                    format_double(Arguments.pbe.ext_diel));
    TitanUI::log_kv(log, "Salt Concentration",
                    format_double(Arguments.pbe.salt) + " M");
    TitanUI::log_kv(log, "DeltaG Solv",
                    Arguments.pbe.dgsolv ? "enabled (complex/mol1/mol2)"
                                         : "off");
    TitanUI::log_kv(log, "MM/PBSA",
                    Arguments.pbe.pbsa ? "enabled" : "off");
    TitanUI::log_kv(log, "Kernel Effective",
                    "scalar-threaded v1 (AVX2 reserved for later hot paths)");
    TitanUI::log_kv(log, "Map Output",
                    Arguments.pbe.map + " / " + Arguments.pbe.map_frames);
  }

  need_tica_projection_cache =
      Arguments.msm.flag_msm || Arguments.dham.flag_dham ||
      Arguments.umap.flag_umap ||
      (Arguments.zts.flag_zts && Arguments.zts.zts_domain == "tica");

  if (Arguments.pbe.flag_pbe)
    append_module_label(active_modules, "PBE");
  if (Arguments.pbe.pbsa)
    append_module_label(active_modules, "MM/PBSA");
  if (Arguments.flag_cas)
    append_module_label(active_modules, "CAS");
  if (Arguments.dssp.flag_dssp &&
      (TitanDSSPAnalyzer::supports_moltype(Arguments.mol_1_type) ||
       TitanDSSPAnalyzer::supports_moltype(Arguments.mol_2_type))) {
    append_module_label(active_modules, "DSSP");
  }
  if (Arguments.lj.flag_lj)
    append_module_label(active_modules, "LJ");
  if (Arguments.sq.flag_sq)
    append_module_label(active_modules, "SQ");
  if (Arguments.cmap.flag_cmap)
    append_module_label(active_modules, "CMap");
  if (Arguments.dihedral.flag_dihedral)
    append_module_label(active_modules, "Dihedral");
  if (Arguments.qnative.flag_qnative)
    append_module_label(active_modules, "QNative");
  if (Arguments.msd.flag_msd)
    append_module_label(active_modules, "MSD");
  if (Arguments.membrane.flag_membrane)
    append_module_label(active_modules, "Membrane");
  if (Arguments.s2.flag_s2)
    append_module_label(active_modules, "S2");
  if (Arguments.rg.flag_rg2)
    append_module_label(active_modules, "Rg2");
  if (Arguments.rms.flag_rms)
    append_module_label(active_modules, "RMSD/RMSF");
  if (Arguments.flag_network || Arguments.community.flag_community)
    append_module_label(active_modules, "DCCM/Network");
  if (Arguments.corr_knn.flag_corr_knn)
    append_module_label(active_modules, "Corr-kNN");
  if (Arguments.community.flag_community)
    append_module_label(active_modules, "Community");
  if (Arguments.flag_pca)
    append_module_label(active_modules, "PCA_FEL");
  if (Arguments.flag_tda)
    append_module_label(active_modules, "TDA");
  if (Arguments.msm.flag_tica)
    append_module_label(active_modules, "TICA");
  if (Arguments.msm.flag_msm)
    append_module_label(active_modules, "MSM");
  if (Arguments.zts.flag_zts)
    append_module_label(active_modules, "ZTS");
  if (Arguments.umap.flag_umap)
    append_module_label(active_modules, "UMAP");
  if (Arguments.dham.flag_dham)
    append_module_label(active_modules, "DHAM");
  if (Arguments.tpt.flag_tpt)
    append_module_label(active_modules, "TPT");
  if (Arguments.flag_h_bond)
    append_module_label(active_modules, "H-Bonds");
  if (Arguments.n_wat >= 0)
    append_module_label(active_modules, "Water");
  if (Arguments.edecomp.flag_edecomp)
    append_module_label(active_modules, "EDecomp");
  if (edecomp_gbsa_requested(Arguments))
    append_module_label(active_modules, "EDecomp-GBSA");
  if (Arguments.hotspot.flag_hotspot_report)
    append_module_label(active_modules, "Hotspot");
  if (Arguments.flag_allostery)
    append_module_label(active_modules, "Allostery");
  if (Arguments.lspmd.flag_lspmd)
    append_module_label(active_modules, "LSP-MD");

  if (!active_modules.empty()) {
    TitanUI::log_section(log, "REQUESTED MODULES");
    TitanUI::log_kv(log, "Requested Features", active_modules);
    // PBC algorithm in use
    if (Arguments.pbc.solv_pbctopo)
      TitanUI::log_kv(log, "PBC Algorithm",
                      Arguments.pbc.pbctopo_mode == "llps-local"
                          ? "pbctopo (LLPS local molecule unwrap)"
                          : "pbctopo (VIBE: Virtual Intact Block Extraction)");
    else if (Arguments.pbc.solv_pbcn)
      TitanUI::log_kv(log, "PBC Algorithm",
                      Arguments.pbc.auto_enabled_pbcn
                          ? "FixBox-style staged envelope (auto-enabled)"
                          : "FixBox-style staged envelope");
    else if (Arguments.pbc.solv_pbc_int == 1)
      TitanUI::log_kv(log, "PBC Algorithm",
                      std::string("Internal-reference envelope (chain ref=") +
                          std::to_string(Arguments.pbc.chain_ref) +
                          (Arguments.pbc.auto_enabled_pbci ? ", auto-enabled)"
                                                          : ")"));
    else if (Arguments.pbc.solv_pbc == 1)
      TitanUI::log_kv(log, "PBC Algorithm", "Prob (legacy)");
    else
      TitanUI::log_kv(log, "PBC Algorithm", "None");
    if (!Arguments.flag_nocal || Arguments.flag_gbsa ||
        Arguments.pbe.flag_pbe)
      TitanUI::log_kv(log, "Energy Distance Policy",
                      "preprocessed_raw (no implicit MIC in energy kernels)");
    if (Arguments.pbc.auto_enabled_pbcn || Arguments.pbc.auto_enabled_pbci) {
      std::vector<std::string> auto_pbc_modules;
      if (Arguments.flag_nocal)
        auto_pbc_modules.push_back("Nocal");
      if (Arguments.rms.flag_rms)
        auto_pbc_modules.push_back("RMSD/RMSF");
      if (Arguments.rg.flag_rg2)
        auto_pbc_modules.push_back("Rg2");
      if (Arguments.msd.flag_msd)
        auto_pbc_modules.push_back("MSD");
      if (Arguments.lj.flag_lj)
        auto_pbc_modules.push_back("LJ");
      if (Arguments.sq.flag_sq)
        auto_pbc_modules.push_back("SQ");
      std::string auto_pbc_message;
      for (std::size_t i = 0; i < auto_pbc_modules.size(); ++i) {
        if (i > 0)
          auto_pbc_message +=
              (i + 1 == auto_pbc_modules.size()) ? " and " : ", ";
        auto_pbc_message += auto_pbc_modules[i];
      }
      if (auto_pbc_modules.size() == 1 &&
          (auto_pbc_modules.front() == "LJ" ||
           auto_pbc_modules.front() == "SQ"))
        auto_pbc_message += " analysis";
      auto_pbc_message +=
          " requested without explicit PBC mode; enabled -pbci 0";
      TitanUI::log_kv(log, "PBC Auto-Enable", auto_pbc_message);
    }
  }

  take_para(parameter, force_f, Arguments);
  take_amino_acids(amine, num_amine, force_f, Arguments);
  take_ff_para(amine, parameter);

  input_name = int{};
  gen_input_list(Arguments, input_line, paths, log, input_name);

  if (edecomp_mm_requested(Arguments) && Arguments.flag_nocal) {
    TitanUI::print_error(
        "Energy decomposition: -edecomp-source=mm requires energy "
        "calculation; remove -nocal or select a future non-MM adapter.");
    return 1;
  }

  // ── Auto-detect XDR trajectory from -i file list when -s was used ──
  // If -s set flag_xdr but no -traj was given, detect from input_line
  if (Arguments.flag_xdr && Arguments.traj_file.empty() &&
      !input_line.empty()) {
    bool detected = false;
    for (const auto &file : input_line) {
      if (file.length() <= 4)
        continue;
      std::string ext = file.substr(file.length() - 4);
      if (ext == ".xtc" || ext == ".trr") {
        const std::string detected_type = (ext == ".xtc") ? "xtc" : "trr";
        if (!detected) {
          Arguments.traj_file = file;
          Arguments.xdr_type = detected_type;
          detected = true;
        }
        TitanUI::log_kv(log, "Auto-detect XDR",
                        detected_type + " from " + file);
      }
    }
  }

  global_gbsa_agg.emplace(Arguments, (*info_output));
  global_has_nucleic_acid = false;

  // Network Analysis (DCCM) Variables
  dccm_analyzer_initialized = false;
  time_gbsa = std::chrono::milliseconds{0};
  time_cas = std::chrono::milliseconds{0};
  time_dssp = std::chrono::milliseconds{0};
  time_hbk = std::chrono::milliseconds{0};
  time_lj = std::chrono::milliseconds{0};
  time_sq = std::chrono::milliseconds{0};
  time_cmap = std::chrono::milliseconds{0};
  time_interface = std::chrono::milliseconds{0};
  time_dihedral = std::chrono::milliseconds{0};
  time_qnative = std::chrono::milliseconds{0};
  time_msd = std::chrono::milliseconds{0};
  time_rg = std::chrono::milliseconds{0};
  time_rms = std::chrono::milliseconds{0};
  time_membrane = std::chrono::milliseconds{0};
  time_s2 = std::chrono::milliseconds{0};
  time_edecomp = std::chrono::milliseconds{0};
  time_network = std::chrono::milliseconds{0};
  time_tda = std::chrono::milliseconds{0};
  time_llps = std::chrono::milliseconds{0};
  time_tica = std::chrono::milliseconds{0};
  time_llps_dbscan = std::chrono::milliseconds{0};

  dssp_mol1_analyzer.configure(Arguments.dssp);
  dssp_mol2_analyzer.configure(Arguments.dssp);
  dssp_mol1_analyzer.configure_segments(Arguments.segments, 1);
  dssp_mol2_analyzer.configure_segments(Arguments.segments, 2);
  hbk_analyzer.configure(Arguments.hbk);
  lj_analyzer.configure(Arguments.lj);
  sq_analyzer.configure(Arguments.sq);
  cmap_analyzer.configure(Arguments.cmap);
  interface_analyzer.configure(Arguments.interface_analysis,
                               Arguments.segments);
  dihedral_analyzer.configure(Arguments.dihedral);
  qnative_analyzer.configure(Arguments.qnative);
  msd_analyzer.configure(Arguments.msd);
  membrane_analyzer.configure(Arguments.membrane);
  s2_analyzer.configure(Arguments.s2);
  rg_analyzer.configure(Arguments.rg);
  rms_analyzer.configure(Arguments.rms);
  pbe_analyzer.configure(Arguments);
  perf_observer.configure(Arguments.perf_report.mode);

  time_cal = std::chrono::milliseconds{};
  time_fmap = std::chrono::milliseconds{};
  time_subgroups_cal = std::chrono::milliseconds{};
  time_solve_pbc = std::chrono::milliseconds{};
  time_sc = std::chrono::milliseconds{};
  time_hb = std::chrono::milliseconds{};
  time_pdb = std::chrono::milliseconds{};
  time_other = std::chrono::milliseconds{};
  time_print_force = std::chrono::milliseconds{};
  time_maxforce = std::chrono::milliseconds{};
  time_contact = std::chrono::milliseconds{};
  time_input_init = std::chrono::milliseconds{};
  time_input_stream = std::chrono::milliseconds{};
  time_cscd = std::chrono::milliseconds{0};
  time_water = std::chrono::milliseconds{0};
  time_na = std::chrono::milliseconds{0};
  time_pbe = std::chrono::milliseconds{0};
  pro_in = Multi_MOL(force_f);
  lig = Multi_MOL(force_f);
  water_traj = Multi_MOL(force_f);
  membrane_traj = Multi_MOL(force_f);
  topo_initialized = false;
  charge_initialized = false;

  // --- IFP (Interaction Fingerprint) Initialization ---
  if (Arguments.ifp.flag_ifp) {
    ifp_analyzer = std::make_unique<TitanIFP>();
    TitanUI::print_info("[IFP] Interaction Fingerprint module enabled.");
  }
  ifp_pbctopo_total_frames = 0;
  ifp_pbctopo_included_frames = 0;
  ifp_pbctopo_missing_status_frames = 0;
  if (Arguments.ifp.pbctopo_certified_only) {
    ifp_pbctopo_filter_report.open(
        (*info_output).get_ifp_pbctopo_frame_filter_filename());
    if (!ifp_pbctopo_filter_report.is_open()) {
      TitanUI::print_error(
          "[IFP] Cannot open PBCTOPO frame-filter audit output.");
      return 1;
    }
    ifp_pbctopo_filter_report
        << "trajectory_id,trajectory_frame_index,global_frame_index,time_ps,"
           "pbctopo_status,ifp_included\n";
  }
  if (Arguments.awh.flag_awh_ifp) {
    awh_ifp_analyzer = std::make_unique<TitanAwhIfp>();
    const size_t expected_awh_walkers = input_line.empty() ? 1u : input_line.size();
    if (!awh_ifp_analyzer->initialize(Arguments, (*info_output),
                                      expected_awh_walkers)) {
      TitanUI::print_error(
          "AWH: failed to initialize RC-binned IFP workflow.");
      return 1;
    }
    TitanUI::print_info(
        "[AWH] RC-binned IFP workflow enabled for biased AWH analysis.");
  }
  time_ifp = std::chrono::milliseconds{0};

  // === TICA workspace — hoisted outside outer loop (Fix 1: single reference
  // frame shared across all replicas for H-REMD cross-replica consistency) ===
  tica_n_ca_global = 0;
  tica_feat_dim_global = 0;
  tica_R_global = Eigen::Matrix3d::Identity();
  tica_ref_initialized_global = false;
  tica_use_lj_features_global = false;
  tica_use_observable_features_global = false;
  return std::nullopt;
}

// Strip trailing zeros for doubles using a lambda or just stringstream
std::string EngineRun::format_double(double v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

double EngineRun::elapsed_ms_since(const std::chrono::steady_clock::time_point &start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

bool EngineRun::include_ifp_frame(const Multi_MOL &source,
                                  size_t chunk_frame_index,
                                  size_t trajectory_id,
                                  size_t trajectory_frame_index,
                                  size_t global_frame_index) {
  if (!Arguments.ifp.pbctopo_certified_only)
    return true;

  ++ifp_pbctopo_total_frames;
  const bool have_status =
      source.has_pbctopo_frame_status(chunk_frame_index);
  const std::string status =
      have_status
          ? std::string(source.pbctopo_frame_status(chunk_frame_index))
          : "missing_status";
  const bool included =
      have_status && source.pbctopo_frame_certified(chunk_frame_index);
  if (!have_status)
    ++ifp_pbctopo_missing_status_frames;
  if (included)
    ++ifp_pbctopo_included_frames;
  ++ifp_pbctopo_status_counts[status];

  const double time_ps =
      chunk_frame_index < source.m_mol.size()
          ? source.m_mol[chunk_frame_index].time_of_snap
          : std::numeric_limits<double>::quiet_NaN();
  ifp_pbctopo_filter_report
      << trajectory_id << ',' << trajectory_frame_index << ','
      << global_frame_index << ',' << std::setprecision(12) << time_ps
      << ',' << status << ',' << (included ? 1 : 0) << '\n';
  return included;
}

} // namespace engine_detail
