#include "MOL.h"
#include "arguments.h"
#include "cli/option_validation.h"
#include "functions.h"
#include "titan_build_info.h"
#include "titan_ui.h"
#include "titan_version_info.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

// Options of arguments::parse_arguments(): see parse_arguments.cpp for the order.

arguments::OptionStep arguments::parse_options_llps_topology(int argc,
                                                             char **argv,
                                                             int &i,
                                                             const std::string &arg,
                                                             std::string &common_value) {
  std::string llps_value;
  if (consume_option_value(arg, {"-llps-rcut"}, llps_value, i)) {
    llps.llps_rcut = stof(llps_value);
  }
  if (consume_option_value(arg, {"-llps-perc"}, llps_value, i)) {
    llps.llps_perc_thr = stof(llps_value);
  }
  if (consume_option_value(arg, {"-llps-maxlag"}, llps_value, i)) {
    llps.llps_maxlag = stoi(llps_value);
  }
  if (consume_option_value(arg, {"-llps-dimer-size"}, llps_value, i)) {
    llps.llps_dimer_size = stoi(llps_value);
  }
  if (consume_bool_option(arg, {"-llps-contacts"}, llps.llps_contacts, i)) {
  }
  if (consume_option_value(arg, {"-llps-sticker"}, llps_value, i)) {
    llps.llps_sticker_beads = llps_value;
  }
  if (consume_option_value(arg, {"-llps-voxel"}, llps_value, i)) {
    llps.llps_voxel_size = stof(llps_value);
  }
  if (consume_option_value(arg, {"-llps-density", "--llps-density"},
                           llps_value, i)) {
    llps.llps_density = require_choice(
        llps_value, {"auto", "full", "projection", "off"}, "-llps-density");
  }
  if (consume_bool_option(arg, {"-llps-dbscan"}, llps.llps_dbscan, i)) {}
  if (consume_option_value(arg, {"-llps-phase-geometry"}, llps_value, i))
    llps.llps_phase_geometry=require_choice(llps_value,{"none","droplet","slab"},"-llps-phase-geometry");
  if (consume_option_value(arg, {"-llps-phase-axis"}, llps_value, i)) {
    auto axis=require_choice(llps_value,{"a","b","c"},"-llps-phase-axis");
    llps.llps_phase_axis=axis=="a"?0:axis=="b"?1:2;
  }
  if (consume_option_value(arg, {"-llps-phase-dense-nm"}, llps_value, i)) llps.llps_phase_dense=stod(llps_value);
  if (consume_option_value(arg, {"-llps-phase-dilute-inner-nm"}, llps_value, i)) llps.llps_phase_dilute_inner=stod(llps_value);
  if (consume_option_value(arg, {"-llps-phase-dilute-outer-nm"}, llps_value, i)) llps.llps_phase_dilute_outer=stod(llps_value);
  if (consume_option_value(arg, {"-llps-kernel", "--llps-kernel"}, llps_value,
                           i)) {
    llps.llps_kernel = require_choice(llps_value, {"auto", "scalar", "avx2"},
                                      "-llps-kernel");
  }
  if (consume_bool_option(arg, {"-llps-dynamics", "--llps-dynamics"},
                          llps.llps_dynamics, i)) {
  }
  bool llps_detailed_flag = llps.llps_dynamics_detailed;
  if (consume_bool_option(
          arg, {"-llps-dynamics-detailed", "--llps-dynamics-detailed"},
          llps_detailed_flag, i)) {
    llps.llps_dynamics_detailed = llps_detailed_flag;
    if (llps_detailed_flag)
      llps.llps_dynamics = true;
  }
  if (consume_option_value(
          arg, {"-llps-lineage-overlap", "--llps-lineage-overlap"},
          llps_value, i)) {
    llps.llps_dynamics = true;
    llps.llps_lineage_overlap = stod(llps_value);
  }
  if (consume_option_value(arg, {"-llps-prna-rcut", "--llps-prna-rcut"},
                           llps_value, i)) {
    llps.llps_dynamics = true;
    llps.llps_prna_rcut = stof(llps_value);
  }
  if (consume_option_value(
          arg, {"-llps-msd-fit-start-ns", "--llps-msd-fit-start-ns"},
          llps_value, i)) {
    llps.llps_dynamics = true;
    if (llps_value == "auto" || llps_value == "AUTO" || llps_value == "Auto")
      llps.llps_msd_fit_start_ns = -1.0;
    else
      llps.llps_msd_fit_start_ns = stod(llps_value);
  }
  if (consume_option_value(arg,
                           {"-llps-msd-fit-end-ns", "--llps-msd-fit-end-ns"},
                           llps_value, i)) {
    llps.llps_dynamics = true;
    if (llps_value == "auto" || llps_value == "AUTO" || llps_value == "Auto")
      llps.llps_msd_fit_end_ns = -1.0;
    else
      llps.llps_msd_fit_end_ns = stod(llps_value);
  }
  if (consume_option_value(
          arg, {"-llps-rna-segment-beads", "--llps-rna-segment-beads"},
          llps_value, i)) {
    llps.llps_dynamics = true;
    if (llps_value == "none" || llps_value == "off")
      llps.llps_rna_segment_beads = 0;
    else
      llps.llps_rna_segment_beads = stoi(llps_value);
  }
  if (consume_bool_option(
          arg, {"-llps-msd-drift-correct", "--llps-msd-drift-correct"},
          llps.llps_msd_drift_correct, i)) {
    llps.llps_dynamics = true;
  }
  if (consume_option_value(
          arg,
          {"-llps-mobile-displacement-nm", "--llps-mobile-displacement-nm"},
          llps_value, i)) {
    llps.llps_dynamics = true;
    llps.llps_mobile_displacement_nm = stod(llps_value);
  }
  if (consume_option_value(arg,
                           {"-llps-mobile-lag-ns", "--llps-mobile-lag-ns"},
                           llps_value, i)) {
    llps.llps_dynamics = true;
    if (llps_value == "auto" || llps_value == "AUTO" || llps_value == "Auto")
      llps.llps_mobile_lag_ns = -1.0;
    else
      llps.llps_mobile_lag_ns = stod(llps_value);
  }
  if (consume_option_value(arg,
                           {"-llps-rna-compaction", "--llps-rna-compaction"},
                           llps_value, i)) {
    llps.llps_rna_compaction = require_choice(
        llps_value,
        {"auto", "true", "false", "yes", "no", "on", "off", "1", "0"},
        "-llps-rna-compaction");
    if (llps.llps_rna_compaction == "true" ||
        llps.llps_rna_compaction == "yes" ||
        llps.llps_rna_compaction == "on" || llps.llps_rna_compaction == "1")
      llps.llps_dynamics = true;
  }
  if (consume_option_value(
          arg, {"-llps-rna-information", "--llps-rna-information"},
          llps_value, i)) {
    llps.llps_rna_information = require_choice(
        llps_value,
        {"auto", "true", "false", "yes", "no", "on", "off", "1", "0"},
        "-llps-rna-information");
  }
  if (consume_option_value(arg, {"-llps-rna-topology", "--llps-rna-topology"},
                           llps_value, i)) {
    llps.llps_rna_topology = require_choice(
        llps_value,
        {"auto", "true", "false", "yes", "no", "on", "off", "1", "0"},
        "-llps-rna-topology");
    if (llps.llps_rna_topology == "true" || llps.llps_rna_topology == "yes" ||
        llps.llps_rna_topology == "on" || llps.llps_rna_topology == "1")
      llps.llps_dynamics = true;
  }
  bool llps_rna_topology_detailed_flag = llps.llps_rna_topology_detailed;
  if (consume_bool_option(
          arg,
          {"-llps-rna-topology-detailed", "--llps-rna-topology-detailed"},
          llps_rna_topology_detailed_flag, i)) {
    llps.llps_rna_topology_detailed = llps_rna_topology_detailed_flag;
    if (llps_rna_topology_detailed_flag) {
      llps.llps_dynamics = true;
      if (llps.llps_rna_topology == "auto")
        llps.llps_rna_topology = "true";
    }
  }
  if (consume_option_value(
          arg, {"-llps-energy-correlation", "--llps-energy-correlation"},
          llps_value, i)) {
    llps.llps_energy_correlation = require_choice(
        llps_value,
        {"auto", "true", "false", "yes", "no", "on", "off", "1", "0"},
        "-llps-energy-correlation");
    if (llps.llps_energy_correlation == "true" ||
        llps.llps_energy_correlation == "yes" ||
        llps.llps_energy_correlation == "on" ||
        llps.llps_energy_correlation == "1")
      llps.llps_dynamics = true;
  }
  bool llps_energy_detailed_flag = llps.llps_energy_detailed;
  if (consume_bool_option(arg,
                          {"-llps-energy-detailed", "--llps-energy-detailed"},
                          llps_energy_detailed_flag, i)) {
    llps.llps_energy_detailed = llps_energy_detailed_flag;
    if (llps_energy_detailed_flag) {
      llps.llps_dynamics = true;
      if (llps.llps_energy_correlation == "auto")
        llps.llps_energy_correlation = "true";
    }
  }
  bool llps_paper_flag = llps.llps_paper;
  if (consume_bool_option(arg, {"-llps-paper", "--llps-paper"},
                          llps_paper_flag, i)) {
    llps.llps_paper = llps_paper_flag;
    if (llps_paper_flag) {
      llps.flag_llps = true;
      llps.llps_contacts = true;
      llps.llps_dynamics = true;
    }
  }
  if (consume_option_value(
          arg, {"-llps-paper-profile-bin", "--llps-paper-profile-bin"},
          llps_value, i)) {
    llps.llps_paper_profile_bin_nm = std::max(0.01, stod(llps_value));
  }
  if (consume_option_value(arg, {"-llps-paper-blocks", "--llps-paper-blocks"},
                           llps_value, i)) {
    if (llps_value == "auto" || llps_value == "AUTO" ||
        llps_value == "Auto") {
      llps.llps_paper_blocks = 0;
    } else {
      llps.llps_paper_blocks = std::max(1, stoi(llps_value));
    }
  }
  if (consume_option_value(arg,
                           {"-llps-paper-geometry", "--llps-paper-geometry"},
                           llps_value, i)) {
    llps.llps_paper_geometry = require_choice(
        llps_value, {"auto", "droplet", "slab"}, "-llps-paper-geometry");
  }
  if (consume_option_value(arg,
                           {"-llps-paper-fractal", "--llps-paper-fractal"},
                           llps_value, i)) {
    llps.llps_paper_fractal = require_choice(
        llps_value,
        {"auto", "true", "false", "yes", "no", "on", "off", "1", "0"},
        "-llps-paper-fractal");
    if (llps.llps_paper_fractal == "true" ||
        llps.llps_paper_fractal == "yes" || llps.llps_paper_fractal == "on" ||
        llps.llps_paper_fractal == "1") {
      llps.flag_llps = true;
      llps.llps_paper = true;
      llps.llps_contacts = true;
      llps.llps_dynamics = true;
    }
  }
  if (consume_option_value(arg,
                           {"-llps-rna-intra-rcut", "--llps-rna-intra-rcut"},
                           llps_value, i)) {
    llps.llps_rna_intra_rcut = stof(llps_value);
  }
  if (consume_option_value(
          arg, {"-llps-rna-longrange-min", "--llps-rna-longrange-min"},
          llps_value, i)) {
    llps.llps_rna_longrange_min = std::max(0, stoi(llps_value));
  }
  if (consume_option_value(
          arg, {"-llps-rna-bridge-geom-min", "--llps-rna-bridge-geom-min"},
          llps_value, i)) {
    if (llps_value == "auto" || llps_value == "AUTO" || llps_value == "Auto")
      llps.llps_rna_bridge_geom_min = -1.0f;
    else
      llps.llps_rna_bridge_geom_min = stof(llps_value);
  }
  if (consume_option_value(
          arg, {"-llps-rna-patch-gap-max", "--llps-rna-patch-gap-max"},
          llps_value, i)) {
    llps.llps_rna_patch_gap_max = std::max(0, stoi(llps_value));
  }
  bool llps_rna_clamp_require_loop_flag = llps.llps_rna_clamp_require_loop;
  if (consume_bool_option(
          arg,
          {"-llps-rna-clamp-require-loop", "--llps-rna-clamp-require-loop"},
          llps_rna_clamp_require_loop_flag, i)) {
    llps.llps_rna_clamp_require_loop = llps_rna_clamp_require_loop_flag;
  }
  if (consume_option_value(
          arg, {"-llps-rna-info-rg-bins", "--llps-rna-info-rg-bins"},
          llps_value, i)) {
    llps.llps_rna_info_rg_bins = std::max(1, stoi(llps_value));
  }
  if (consume_option_value(
          arg,
          {"-llps-rna-info-longrange-bins", "--llps-rna-info-longrange-bins"},
          llps_value, i)) {
    llps.llps_rna_info_longrange_bins = std::max(1, stoi(llps_value));
  }
  if (consume_option_value(arg, {"-llps-dbscan-eps"}, llps_value, i)) {
    llps.llps_dbscan_eps = stod(llps_value);
  }
  if (consume_option_value(arg, {"-llps-dbscan-minpts"}, llps_value, i)) {
    llps.llps_dbscan_minpts = stoi(llps_value);
  }
  if (consume_bool_option(arg, {"-llps-fuzzy"}, llps.llps_fuzzy, i)) {
  }
  // --- GIST: use the same normalized split-value parser as other modules ---
  if (consume_bool_option(arg, {"-gist", "--gist"}, gist.flag_gist, i)) {
    return option_consumed();
  }
  std::string gist_value;
  if (consume_required_option_value(arg, {"-gist-resolution", "--gist-resolution"}, gist_value, i)) {
    gist.gist_resolution=parse_numeric_option(gist_value, "-gist-resolution");
    if (gist.gist_resolution<=0) { TitanUI::print_error("-gist-resolution must be positive."); exit(EXIT_FAILURE); }
    return option_consumed();
  }
  if (consume_required_option_value(arg, {"-gist-padding", "--gist-padding"}, gist_value, i)) {
    gist.gist_padding=parse_numeric_option(gist_value, "-gist-padding");
    if (gist.gist_padding<=0) { TitanUI::print_error("-gist-padding must be positive."); exit(EXIT_FAILURE); }
    return option_consumed();
  }
  if (consume_required_option_value(arg, {"-gist-temperature", "--gist-temperature"}, gist_value, i)) {
    gist.gist_temperature=parse_numeric_option(gist_value, "-gist-temperature");
    if (gist.gist_temperature<=0) { TitanUI::print_error("-gist-temperature must be positive."); exit(EXIT_FAILURE); }
    return option_consumed();
  }
  if (consume_required_option_value(arg, {"-gist-cutoff", "--gist-cutoff"}, gist_value, i)) {
    gist.gist_cutoff=parse_numeric_option(gist_value, "-gist-cutoff");
    if (gist.gist_cutoff<=0) { TitanUI::print_error("-gist-cutoff must be positive."); exit(EXIT_FAILURE); }
    return option_consumed();
  }
  if (consume_required_option_value(arg, {"-gist-min-count", "--gist-min-count"}, gist_value, i)) {
    gist.gist_min_count=parse_integer_option(gist_value, "-gist-min-count");
    if (gist.gist_min_count<1) { TitanUI::print_error("-gist-min-count must be positive."); exit(EXIT_FAILURE); }
    return option_consumed();
  }
  if (consume_required_option_value(arg, {"-gist-water-model", "--gist-water-model"}, gist_value, i)) {
    gist.gist_water_model=gist_value;
    return option_consumed();
  }
  // --- IFP (Interaction Fingerprint) flags ---
  if (string(argv[i]) == "-ifp") {
    ifp.flag_ifp = true;
  }
  if (consume_bool_option(
          arg, {"-ifp-pbctopo-certified-only"},
          ifp.pbctopo_certified_only, i)) {
    return option_consumed();
  }
  if (string(argv[i]) == "-ifp-hbond" && i + 1 < argc) {
    ifp.cutoff_hbond = stod(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]) == "-ifp-salt" && i + 1 < argc) {
    ifp.cutoff_salt_bridge = stod(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]) == "-ifp-hphob" && i + 1 < argc) {
    ifp.cutoff_hydrophobic = stod(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]) == "-ifp-pipi" && i + 1 < argc) {
    ifp.cutoff_pi_pi = stod(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]) == "-ifp-cpi" && i + 1 < argc) {
    ifp.cutoff_cation_pi = stod(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]) == "-ifp-hal" && i + 1 < argc) {
    ifp.cutoff_halogen = stod(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]) == "-ifp-out" && i + 1 < argc) {
    ifp.output_filename = string(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]) == "-awh-ifp") {
    awh.flag_awh_ifp = true;
    ifp.flag_ifp = true;
  }
  if ((string(argv[i]) == "-awh-xvg" || string(argv[i]) == "--awh-xvg") &&
      i + 1 < argc) {
    awh.xvg_list_file = string(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]).find("-awh-xvg-col=") == 0 ||
      string(argv[i]).find("--awh-xvg-col=") == 0) {
    awh.xvg_value_column =
        stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
  }
  if ((string(argv[i]) == "-awh-xvg-col" ||
       string(argv[i]) == "--awh-xvg-col") &&
      i + 1 < argc) {
    awh.xvg_value_column = stoi(string(argv[i + 1]));
    ++i;
  }
  if (string(argv[i]).find("-awh-bins=") == 0 ||
      string(argv[i]).find("--awh-bins=") == 0) {
    awh.n_bins = stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
  }
  if ((string(argv[i]) == "-awh-bins" || string(argv[i]) == "--awh-bins") &&
      i + 1 < argc) {
    awh.n_bins = stoi(string(argv[i + 1]));
    ++i;
  }
  if ((string(argv[i]) == "-awh-out-prefix" ||
       string(argv[i]) == "--awh-out-prefix") &&
      i + 1 < argc) {
    awh.output_prefix = string(argv[i + 1]);
    ++i;
  }
  // --- XDR trajectory (XTC/TRR) ---
  if (string(argv[i]) == "-traj" && i + 1 < argc) {
    traj_file = string(argv[i + 1]);
    flag_xdr = true;
    flag_1_pdb = true; // XDR always uses single-topology reference
    if (traj_file.length() > 4) {
      string ext = traj_file.substr(traj_file.length() - 4);
      if (ext == ".xtc")
        xdr_type = "xtc";
      else if (ext == ".trr")
        xdr_type = "trr";
    }
    ++i;
  }
  if (arg == "-gbnsr6" || arg == "--gbnsr6") {
    flag_gbsa = true;
    igb = 66;
    return option_consumed();
  }
  if (consume_required_option_value(arg, {"-gbnsr6-spacing"}, common_value, i)) {
    gbnsr6_spacing = parse_numeric_option(common_value, "-gbnsr6-spacing");
    return option_consumed();
  }
  if (string(argv[i]) == "-igb" && i + 1 < argc) {
    igb = stoi(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]) == "-ext_diel" && i + 1 < argc) {
    ext_diel = stod(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]) == "-int_diel" && i + 1 < argc) {
    int_diel = stod(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]) == "-saltcon" && i + 1 < argc) {
    saltcon = stod(argv[i + 1]);
    salt_concentration = saltcon;
    pbe.salt = saltcon;
    ++i;
  }
  // ENE-17(b): one analysis temperature (K) for Debye screening
  // (-elec-screening), GB salt screening and both interaction-entropy paths.
  if (consume_option_value(arg, {"-temperature", "--temperature", "-gbsa-temp", "--gbsa-temp"},
                           common_value, i)) {
    gbsa_temperature = stod(common_value);
  }
  if (consume_required_option_value(arg, {"-n_wat"}, common_value, i)) {
    n_wat = parse_integer_option(common_value, "-n_wat");
    if(n_wat < -1){TitanUI::print_error("-n_wat must be -1 or a zero-based water group index.");exit(EXIT_FAILURE);}
    return option_consumed();
  }
  if (string(argv[i]) == "-cscd") {
    cal_dis_sc_contact = true;
  }
  if (string(argv[i]) == "-opdb" || string(argv[i]) == "-opdbsep" ||
      string(argv[i]).rfind("-opdb=", 0) == 0 ||
      string(argv[i]).rfind("-opdbsep=", 0) == 0) {
    opdb = true;
    const string arg(argv[i]);
    if (arg == "-opdbsep" || arg.rfind("-opdbsep=", 0) == 0) {
      flag_separation_results = true;
      flag_sep_opdb = true;
    }

    if (arg.rfind("-opdb=", 0) == 0 || arg.rfind("-opdbsep=", 0) == 0) {
      const size_t eq_pos = arg.find('=');
      if (eq_pos != string::npos && eq_pos + 1 < arg.size()) {
        out_pdb_name = arg.substr(eq_pos + 1);
        opdb_name_explicit = true;
      }
    } else if (i + 1 < argc) {
      if (string(argv[i + 1]).find(".pdb")) {
        out_pdb_name = string(argv[i + 1]);
        opdb_name_explicit = true;
      }
    }
  }
  if (string(argv[i]) == "-pqr") {
    flag_pqr = 1;
  }
  if (string(argv[i]) == "-interchain-energy") {
    flag_interchain_energy = true;
  }
  if (string(argv[i]) == "-dipole-moment") {
    flag_cal_dipole_moment = true;
  }
  if (string(argv[i]) == "-act") {
    flag_all_side_chain_ct = 0;
  }
  if (string(argv[i]) == "-vibe-help" || string(argv[i]) == "--vibe-help") {
    vibe_help_print();
    return option_exit(2);
  }
  if (string(argv[i]) == "-hololift-help" || string(argv[i]) == "--hololift-help") {
    hololift_help_print();
    return option_exit(2);
  }
  if (string(argv[i]) == "-h" || string(argv[i]) == "--help") {
    help = true;
    if (help_print() == 0)
      return option_exit(2);
  }
  if (string(argv[i]) == "-v" || string(argv[i]) == "--version") {
    cout << "TITAN " << titan_build::full_version() << '\n'
         << "Version: " << titan_build::product_version() << '\n'
         << "Build number: " << titan_build::build_number_string() << '\n'
         << "Commit: " << titan_build::git_commit_short() << " ("
         << titan_build::git_state_name() << ")\n"
         << "Profile: " << titan_build::build_profile_name() << '\n'
         << "Compiler: " << titan_build::compiler_name() << '\n'
         << "Target: " << titan_build::target_os_name() << '-'
         << titan_build::target_arch_name() << '\n'
         << "Source date: " << titan_build::source_date_utc() << '\n';
    if (std::string_view(titan_build::ci_build_number()).empty() == false) {
      cout << "CI: " << titan_build::ci_provider()
           << " build " << titan_build::ci_build_number();
      if (std::string_view(titan_build::ci_run_id()).empty() == false)
        cout << " (run " << titan_build::ci_run_id() << ')';
      cout << '\n';
    }
    return option_exit(2);
  }
  if (string(argv[i]) == "-pbc") {
    pbc.solv_pbc = 1;
  }
  if (string(argv[i]) == "-cat-result") {
    flag_cat_result_multi_infile = true;
  }
  if (string(argv[i]) == "-pbci") {
    pbc.solv_pbc_int = 1;
    pbc.solv_pbc = 1;
    if (i + 1 < argc && isdigit(argv[i + 1][0])) {
      pbc.solv_pbc = 0;
      pbc.chain_ref = parse_integer_option(argv[i + 1], "-pbci");
    } else {
      cout << "WARNING!! The option -pbci lacks reference chain!\n";
      pbc.solv_pbc_int = 0;
      pbc.solv_pbc = 0;
    }
  }
  if (string(argv[i]) == "-pbcn") {
    pbc.solv_pbcn = true;
  }
  if (string(argv[i]) == "-pbctopo") {
    pbc.solv_pbctopo = true;
  }
  if (string(argv[i]) == "-pbctopo-llps-local" ||
      string(argv[i]) == "--pbctopo-llps-local") {
    pbc.solv_pbctopo = true;
    pbc.pbctopo_mode = "llps-local";
  }
  if (consume_required_option_value(arg, {"-pbctopo-mode", "--pbctopo-mode"},
                                    common_value, i)) {
    pbc.solv_pbctopo = true;
    pbc.pbctopo_mode = common_value;
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-search-policy", "--pbctopo-search-policy"},
          common_value, i)) {
    pbc.solv_pbctopo = true;
    pbc.pbctopo_search_policy = common_value;
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-audit-max-states",
                "--pbctopo-audit-max-states"},
          common_value, i)) {
    pbc.solv_pbctopo = true;
    pbc.pbctopo_audit_max_states =
        parse_integer_option(common_value, "-pbctopo-audit-max-states");
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-audit-max-seconds",
                "--pbctopo-audit-max-seconds"},
          common_value, i)) {
    pbc.solv_pbctopo = true;
    pbc.pbctopo_audit_max_seconds =
        parse_numeric_option(common_value, "-pbctopo-audit-max-seconds");
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-audit-max-memory-mb",
                "--pbctopo-audit-max-memory-mb"},
          common_value, i)) {
    pbc.solv_pbctopo = true;
    pbc.pbctopo_audit_max_memory_mb = parse_integer_option(
        common_value, "-pbctopo-audit-max-memory-mb");
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-anchor-backend", "--pbctopo-anchor-backend"},
          common_value, i)) {
    pbc.pbctopo_anchor_backend = common_value;
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-anchor-metrics", "--pbctopo-anchor-metrics"},
          common_value, i)) {
    pbc.pbctopo_anchor_metrics = common_value;
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-temporal", "--pbctopo-temporal"}, common_value,
          i)) {
    pbc.pbctopo_temporal = common_value;
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-temporal-k", "--pbctopo-temporal-k"},
          common_value, i)) {
    pbc.pbctopo_temporal_k =
        parse_integer_option(common_value, "-pbctopo-temporal-k");
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-temporal-penalty", "--pbctopo-temporal-penalty"},
          common_value, i)) {
    pbc.pbctopo_temporal_penalty =
        parse_numeric_option(common_value, "-pbctopo-temporal-penalty");
  }
  if (consume_bool_option(
          arg, {"-pbctopo-evidence-export", "--pbctopo-evidence-export"},
          pbc.pbctopo_evidence_export, i)) {
    pbc.solv_pbctopo = true;
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-steric", "--pbctopo-steric"}, common_value, i)) {
    pbc.pbctopo_steric = common_value;
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-vdw-clash-scale", "--pbctopo-vdw-clash-scale"},
          common_value, i)) {
    pbc.pbctopo_vdw_clash_scale =
        parse_numeric_option(common_value, "-pbctopo-vdw-clash-scale");
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-bond-cutoff", "--pbctopo-bond-cutoff"},
          common_value, i)) {
    pbc.pbctopo_bond_cutoff =
        parse_numeric_option(common_value, "-pbctopo-bond-cutoff");
    pbc.pbctopo_bond_cutoff_set = true;
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-clash-cutoff", "--pbctopo-clash-cutoff"},
          common_value, i)) {
    pbc.pbctopo_clash_cutoff =
        parse_numeric_option(common_value, "-pbctopo-clash-cutoff");
    pbc.pbctopo_clash_cutoff_set = true;
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-contact-cutoff", "--pbctopo-contact-cutoff"},
          common_value, i)) {
    pbc.pbctopo_contact_cutoff =
        parse_numeric_option(common_value, "-pbctopo-contact-cutoff");
    pbc.pbctopo_contact_cutoff_set = true;
  }
  if (consume_required_option_value(
          arg,
          {"-pbctopo-required-contacts", "--pbctopo-required-contacts"},
          common_value, i)) {
    pbc.solv_pbctopo = true;
    pbc.pbctopo_required_contacts_file = common_value;
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-continuity-cutoff", "--pbctopo-continuity-cutoff"},
          common_value, i)) {
    pbc.pbctopo_continuity_cutoff =
        parse_numeric_option(common_value, "-pbctopo-continuity-cutoff");
    pbc.pbctopo_continuity_cutoff_set = true;
  }
  if (consume_required_option_value(arg,
                                    {"-pbctopo-exact-component-limit",
                                     "--pbctopo-exact-component-limit"},
                                    common_value, i)) {
    pbc.pbctopo_exact_component_limit =
        parse_integer_option(common_value, "-pbctopo-exact-component-limit");
    pbc.pbctopo_exact_component_limit_set = true;
  }
  if (consume_required_option_value(
          arg, {"-pbctopo-beam-width", "--pbctopo-beam-width"}, common_value,
          i)) {
    pbc.pbctopo_beam_width =
        parse_integer_option(common_value, "-pbctopo-beam-width");
    pbc.pbctopo_beam_width_set = true;
  }
  if (consume_required_option_value(
          arg,
          {"-pbctopo-component-shell-radius",
           "--pbctopo-component-shell-radius"},
          common_value, i)) {
    pbc.pbctopo_component_shell_radius = parse_integer_option(
        common_value, "-pbctopo-component-shell-radius");
    pbc.pbctopo_component_shell_radius_set = true;
  }
  if (consume_bool_option(
          arg, {"-pbctopo-h9-orbit", "--pbctopo-h9-orbit"},
          pbc.pbctopo_h9_orbit, i)) {
    pbc.solv_pbctopo = true;
  }
  if (string(argv[i]) == "-pbci2") {
    pbc.solv_pbc_int = 1;
    pbc.solv_pbc_int2 = 1;
    pbc.solv_pbc = 1;
    if (i + 1 < argc && isdigit(argv[i + 1][0])) {
      pbc.solv_pbc = 0;
      pbc.chain_ref = parse_integer_option(argv[i + 1], "-pbci2");
    } else {
      cout << "WARNING!! The option -pbci2 lacks reference chain!\n";
      pbc.solv_pbc_int = 0;
      pbc.solv_pbc_int2 = 0;
      pbc.solv_pbc = 0;
    }
  }
  if (string(argv[i]) == "-pbci3") {
    pbc.solv_pbc_int = 1;
    pbc.solv_pbc_int2 = 1;
    pbc.solv_pbc = 1;
    pbc.solv_pbc_int3 = 1;
    if (i + 1 < argc && isdigit(argv[i + 1][0])) {
      pbc.solv_pbc = 0;
      pbc.chain_ref = parse_integer_option(argv[i + 1], "-pbci3");
    } else {
      cout << "WARNING!! The option -pbci3 lacks reference chain!\n";
      pbc.solv_pbc_int = 0;
      pbc.solv_pbc_int2 = 0;
      pbc.solv_pbc_int3 = 0;
      pbc.solv_pbc = 0;
    }
  }
  if (string(argv[i]) == "-pbca") {
    pbc.solv_pbc_int = 1;
    pbc.solv_pbc = 1;
    pbc.pbc_aggressive = true;
    if (i + 1 < argc && isdigit(argv[i + 1][0])) {
      pbc.solv_pbc = 0;
      pbc.chain_ref = parse_integer_option(argv[i + 1], "-pbca");
      if (pbc.chain_ref < 0)
        pbc.chain_ref = 0;
    } else {
      pbc.solv_pbc_int = 0;
      pbc.solv_pbc = 0;
    }
  }
  if (string(argv[i]) == "-pbcclust") {
    pbc.solv_pbc_clust = true;
    if (i + 1 < argc && string(argv[i + 1]).find("-") == string::npos) {
      string temp{string(argv[i + 1])};
      for (size_t i{}; i < temp.size(); ++i)
        pbc.pbc_clust_axis.emplace_back(temp[i]);
    } else
      pbc.solv_pbc_clust = false;
  }
  if (string(argv[i]) == "-1pdb") {
    flag_1_pdb = true;
  }
  if (string(argv[i]).find("-1pdb=") == 0) {
    flag_1_pdb = parse_bool_equals(
        string(argv[i]).substr(string("-1pdb=").size()), "-1pdb");
  }
  if (string(argv[i]) == "-hb") {
    flag_h_bond = 1;
    if (argc > i + 1 && string(argv[i + 1]) == "side-chain")
      flag_sidechain_hbond = true;
  }
  if (string(argv[i]) == "-entropy") {
    entropy = true;
  }
  // KIN-22: optional trimming of the -entropy sample (default 0 = none)
  if (consume_option_value(
          arg, {"-entropy-filter-sigma", "--entropy-filter-sigma"},
          common_value, i)) {
    entropy_filter_sigma =
        parse_numeric_option(common_value, "-entropy-filter-sigma");
    return option_consumed();
  }
  if (string(argv[i]) == "-nocal") {
    flag_nocal = true;
    flag_1_pdb = true;
  }
  if (string(argv[i]).find("-nocal=") == 0) {
    flag_nocal = parse_bool_equals(
        string(argv[i]).substr(string("-nocal=").size()), "-nocal");
    if (flag_nocal)
      flag_1_pdb = true;
  }
  if (string(argv[i]) == "-all-subfolders") {
    flag_all_folders = true;
    flag_separation_results = true;
  }
  if (string(argv[i]) == "-split") {
    flag_separation_results = true;
  }
  if (string(argv[i]) == "-cal-force") {
    flag_cal_force = true;
  }
  if (string(argv[i]) == "-cal-volume") {
    flag_cal_vol = true;
  }
  return {};
}
