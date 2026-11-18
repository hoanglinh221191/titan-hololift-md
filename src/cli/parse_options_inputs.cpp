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

arguments::OptionStep arguments::parse_options_inputs(int argc, char **argv,
                                                      int &i,
                                                      const std::string &arg,
                                                      std::string &common_value) {
  bool cache_enabled = true;
  if (consume_bool_option(arg, {"-cache", "--cache"}, cache_enabled, i)) {
    flag_cache = cache_enabled ? 1 : 2;
    return option_consumed();
  }
  if (string(argv[i]) == "-max-force") {
    flag_maxforce = true;
  }
  if (string(argv[i]) == "-force-res") {
    print_force_res = true;
    force_map = 1;
  }
  if (string(argv[i]) == "-use-CI") {
    flag_use_CI = true;
  }
  if (string(argv[i]) == "-elec-screening") {
    flag_elec_screening = true;
  }
  if (string(argv[i]) == "-charged_atom_dis") {
    flag_charged_atom_distance_dis = true;
    if (i + 1 < argc && argv[i + 1][0] != '-' && isdigit(argv[i + 1][0]) != 0)
      number_charged_distances_bin = parse_integer_option(argv[i + 1], "-charged_atom_dis");
  }
  if (string(argv[i]) == "-charged_distances_min") {
    flag_charged_atom_distance_dis = true;
    if (i + 1 < argc && argv[i + 1][0] != '-' && isdigit(argv[i + 1][0]) != 0)
      charged_distances_min = parse_numeric_option(argv[i + 1], "-charged_distances_min");
  }
  if (string(argv[i]) == "-charged_distances_max") {
    flag_charged_atom_distance_dis = true;
    if (i + 1 < argc && argv[i + 1][0] != '-' && isdigit(argv[i + 1][0]) != 0)
      charged_distances_max = parse_numeric_option(argv[i + 1], "-charged_distances_max");
  }
  if (string(argv[i]) == "-resdis2com") {
    flag_cal_dis_res_2com = true;
  }
  if (string(argv[i]) == "-angle_dipole_efield") {
    flag_angle_dipole_efield = true;
    if (i + 1 < argc && argv[i + 1][0] != '-' && isdigit(argv[i + 1][0]) != 0)
      number_bin_angle_dipole_efield = parse_integer_option(argv[i + 1], "-angle_dipole_efield");
  }
  if (string(argv[i]) == "-side_chain_contact" && i + 1 < argc) {
    if (string(argv[i + 1]) == "full-res")
      flag_sidechain = false;
    else
      flag_sidechain = true;
  }
  if (string(argv[i]) == "-noprint-side_chain_contact" && i + 1 < argc) {
    flag_print_sidechain = false;
  }
  if (arg == "-mol-1-type" && i + 1 < argc) {
    assign_mol_type(string(argv[i + 1]), true);
  }
  if (arg == "-mol-2-type" && i + 1 < argc) {
    assign_mol_type(string(argv[i + 1]), false);
  }
  if (string(argv[i]) == "-com_energy" && i + 1 < argc) {
    if (string(argv[i + 1]) == "x")
      com_dis_energy_2mol = 1;
    else if (string(argv[i + 1]) == "y")
      com_dis_energy_2mol = 2;
    else if (string(argv[i + 1]) == "z")
      com_dis_energy_2mol = 3;
    else if (string(argv[i + 1]) == "all")
      com_dis_energy_2mol = 0;
    else
      com_dis_energy_2mol = 4;
  }
  if (string(argv[i]) == "-b" && i + 1 < argc) {
    begin_time = parse_numeric_option(argv[i + 1], "-b");
    flag_begin_set = true;
  }
  if (string(argv[i]) == "-e" && i + 1 < argc) {
    end_time = parse_numeric_option(argv[i + 1], "-e");
    flag_end_set = true;
  }
  if (string(argv[i]) == "-dt" && i + 1 < argc) {
    dt = parse_integer_option(argv[i + 1], "-dt");
  }
  if (string(argv[i]) == "-mol2" && i + 1 < argc) {
    begin_mol_2 = parse_integer_option(argv[i + 1], "-mol2");
  }
  if (string(argv[i]) == "-topo" && i + 1 < argc) {
    topo_name = argv[i + 1];
    if (check_topo(topo_name, 1) == true)
      flag_topo_1 = true;
  }
  if (string(argv[i]) == "-topo2" && i + 1 < argc) {
    topo_name_2 = argv[i + 1];
    if (check_topo(topo_name_2, 2) == true)
      flag_topo_2 = true;
  }
  if (string(argv[i]) == "-f" && i + 1 < argc) {
    pdb_name = argv[i + 1];
    flag_1_pdb = true;
  }
  // -s = structure file (topology reference for XDR trajectory)
  // Sets pdb_name as reference and enables XDR mode.
  // XDR type is auto-detected from -traj or -i file list.
  if (string(argv[i]) == "-s" && i + 1 < argc) {
    pdb_name = argv[i + 1];
    flag_1_pdb = true;
    flag_xdr = true;
  }
  if (string(argv[i]) == "-n" && i + 1 < argc) {
    index_name = argv[i + 1];
    flag_index = true;
  }
  if (string(argv[i]) == "-eu" && i + 1 < argc) {
    eu = argv[i + 1];
  }
  if (string(argv[i]) == "-fu" && i + 1 < argc) {
    fu = argv[i + 1];
  }
  if (consume_required_option_value(arg, {"-tu"}, common_value, i)) {
    std::transform(
        common_value.begin(), common_value.end(), common_value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (common_value != "ps" && common_value != "ns") {
      TitanUI::print_error("-tu must be ps or ns.");
      exit(EXIT_FAILURE);
    }
    tu = common_value;
    return option_consumed();
  }

  if (string(argv[i]) == "-ff" && i + 1 < argc) {
    force_f = argv[i + 1];
    transform(force_f.begin(), force_f.end(), force_f.begin(), ::toupper);
  }
  if (string(argv[i]) == "-epsilon_in" && i + 1 < argc) {
    epsilon_in = parse_numeric_option(argv[i + 1], "-epsilon_in");
    ++i;
  }
  if (string(argv[i]) == "-g" && i + 1 < argc) {
    auto found = string(argv[i + 1]).find("&");
    auto found_2 = string(argv[i + 1]).find("-");
    if (found != std::string::npos) {
      group_1 = stoi(string(argv[i + 1]).substr(0, found));
      group_2 = stoi(string(argv[i + 1]).substr(found + 1));
      flag_groups_explicit = true;
    } else if (found_2 != std::string::npos) {
      group_1 = stoi(string(argv[i + 1]).substr(0, found_2));
      group_2 = stoi(string(argv[i + 1]).substr(found_2 + 1));
      flag_groups_explicit = true;
    } else {
      cout << "Your argument -g is invalid!\n";
    }
    if (group_1 == group_2)
      self_int = true;
  }
  if (consume_compound_option_value(arg, {"-sg", "--sg"}, common_value, i)) {
    titan_cli::SubgroupSelection selection;
    std::string error;
    if (!titan_cli::parse_subgroup_expression(common_value, selection,
                                              error)) {
      TitanUI::print_error(
          "Invalid -sg expression: " + error +
          ". Terminal: '-sg 1 2-3 4'; arg.in: '-sg= 1 2-3 4'.");
      return option_exit(titan_cli::kInvalidArguments);
    }
    sub_group_1 = std::move(selection.molecule_1);
    sub_group_2 = std::move(selection.molecule_2);
    flag_sub_groups = true;
    return option_consumed();
  }
  if (arg == "-fmap" &&
      (i + 1 >= argc || std::string(argv[i + 1]).rfind("-", 0) == 0)) {
    force_map = 1;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-fmap"}, common_value, i)) {
    force_map = 1;
    if (common_value == "residue-atom" || common_value == "residue-residue" ||
        common_value == "atom-atom" || common_value == "atom-residue")
      fmap_type = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-fmap-level"}, common_value, i)) {
    if (common_value == "all" || common_value == "per-res")
      fmap_level = common_value;
    return option_consumed();
  }
  if (string(argv[i]) == "-nt" && i + 1 < argc) {
    if (stoi(string(argv[i + 1])) < 0)
      nt = 0;
    else
      nt = stoi(string(argv[i + 1]));
  }
  if (string(argv[i]) == "-dsc" && i + 1 < argc) {
    d_standard = parse_numeric_option(argv[i + 1], "-dsc");
  }
  if (string(argv[i]) == "-denfo" && i + 1 < argc) {
    distance_energy_cutoff = parse_numeric_option(argv[i + 1], "-denfo");
    distance_energy_cutoff *= 0.1;
  }
  if ((string(argv[i]) == "-path-a" && i + 1 < argc) ||
      string(argv[i]).find("-path-a=") == 0) {
    if (string(argv[i]).find("-path-a=") == 0)
      path_amino_acid = string(argv[i]).substr(string("-path-a=").size());
    else
      path_amino_acid = argv[i + 1];
    if (path_amino_acid.empty()) {
      TitanUI::print_error("-path-a must not be empty.");
      exit(EXIT_FAILURE);
    }
    if (path_amino_acid[path_amino_acid.length() - 1] != '/')
      path_amino_acid += '/';
  }
  if ((string(argv[i]) == "-path-n" && i + 1 < argc) ||
      string(argv[i]).find("-path-n=") == 0) {
    if (string(argv[i]).find("-path-n=") == 0)
      path_nonbonded = string(argv[i]).substr(string("-path-n=").size());
    else
      path_nonbonded = argv[i + 1];
    if (path_nonbonded.empty()) {
      TitanUI::print_error("-path-n must not be empty.");
      exit(EXIT_FAILURE);
    }
    if (path_nonbonded[path_nonbonded.length() - 1] != '/')
      path_nonbonded += '/';
  }
  if ((string(argv[i]) == "-fflib-auto" ||
       string(argv[i]).find("-fflib-auto=") == 0)) {
    string value;
    if (string(argv[i]).find("-fflib-auto=") == 0) {
      value = string(argv[i]).substr(string("-fflib-auto=").size());
    } else if (i + 1 < argc) {
      value = string(argv[i + 1]);
      ++i;
    }
    transform(value.begin(), value.end(), value.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    if (value == "on" || value == "yes" || value == "true" || value == "1")
      fflib_auto = true;
    else if (value == "off" || value == "no" || value == "false" ||
             value == "0")
      fflib_auto = false;
    else
      cout << "WARNING!!! -fflib-auto expects on/off.\n";
  }
  if ((string(argv[i]) == "-fflib-missing" ||
       string(argv[i]).find("-fflib-missing=") == 0)) {
    string value;
    if (string(argv[i]).find("-fflib-missing=") == 0) {
      value = string(argv[i]).substr(string("-fflib-missing=").size());
    } else if (i + 1 < argc) {
      value = string(argv[i + 1]);
      ++i;
    }
    transform(value.begin(), value.end(), value.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    if (value == "warn" || value == "strict")
      fflib_missing = value;
    else
      cout << "WARNING!!! -fflib-missing expects warn/strict.\n";
  }
  if ((string(argv[i]) == "-fflib-hydrogens" ||
       string(argv[i]).find("-fflib-hydrogens=") == 0)) {
    string value;
    if (string(argv[i]).find("-fflib-hydrogens=") == 0) {
      value = string(argv[i]).substr(string("-fflib-hydrogens=").size());
    } else if (i + 1 < argc) {
      value = string(argv[i + 1]);
      ++i;
    }
    transform(value.begin(), value.end(), value.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    if (value == "reference" || value == "none")
      fflib_hydrogens = value;
    else
      cout << "WARNING!!! -fflib-hydrogens expects reference/none.\n";
  }
  if (string(argv[i]) == "-pattern-input" && i + 1 < argc) {
    flag_pattern_input = true;
    flag_1_pdb = true;
    if (string(argv[i + 1]).find("*") != string::npos) {
      input_pattern = string(argv[i + 1]);
      if (i + 2 < argc && argv[i + 2][0] != '-')
        b_input = parse_integer_option(argv[i + 2], "-pattern-input");
      if (i + 3 < argc && argv[i + 3][0] != '-')
        e_input = parse_integer_option(argv[i + 3], "-pattern-input");
      if (i + 4 < argc && argv[i + 4][0] != '-')
        step_input = parse_integer_option(argv[i + 4], "-pattern-input");
      if (e_input != 10000000)
        flag_range_input = true;
    } else
      cout << "Can not parse -pattern-input option, it must contain * "
              "character!\n";
  }
  if (string(argv[i]) == "-ahb" && i + 1 < argc) {
    angle_h = parse_numeric_option(argv[i + 1], "-ahb");
  }
  if (string(argv[i]) == "-dhb" && i + 1 < argc) {
    dis_h = parse_numeric_option(argv[i + 1], "-dhb");
  }
  if (consume_bool_option(arg, {"-ml-predict-experimental", "--ml-predict-experimental"},
                             ml_predict_experimental, i)) {
    if (ml_predict_experimental) flag_ml = true;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-qha-temp", "--qha-temp"}, common_value,
                           i)) {
    qha_temperature = parse_numeric_option(common_value, "-qha-temp");
    if (!std::isfinite(qha_temperature) || qha_temperature <= 0.0)
      throw std::invalid_argument("-qha-temp must be finite and positive");
    return option_consumed();
  }
  // -entropy-quasi=<com>,<lig>,<rec> (arg.in) or -entropy-quasi
  // <com>,<lig>,<rec>: one value, so the option validators accept it (the
  // three-token form below is rejected by the terminal validator, CLI010).
  if (consume_option_value(arg, {"-entropy-quasi", "--entropy-quasi"},
                           common_value, i)) {
    std::vector<int> group_ids;
    std::stringstream fields(common_value);
    std::string field;
    while (std::getline(fields, field, ','))
      group_ids.push_back(parse_integer_option(field, "-entropy-quasi"));
    if (group_ids.size() != 3)
      throw std::invalid_argument(
          "-entropy-quasi takes three group indices: <com>,<lig>,<rec>");
    entropy_quasi = true;
    group_entropy_com = group_ids[0];
    group_entropy_lg = group_ids[1];
    group_entropy_rec = group_ids[2];
    return option_consumed();
  }
  if (string(argv[i]) == "-entropy-quasi" && i + 3 < argc) {
    entropy_quasi = true;
    if (isdigit(argv[i + 1][0]) && isdigit(argv[i + 2][0]) &&
        isdigit(argv[i + 3][0])) {
      group_entropy_com = parse_integer_option(argv[i + 1], "-entropy-quasi");
      group_entropy_lg = parse_integer_option(argv[i + 2], "-entropy-quasi");
      group_entropy_rec = parse_integer_option(argv[i + 3], "-entropy-quasi");
    } else {
      cout << "-entropy-quasi option is invalid\n";
      entropy_quasi = false;
    }
  }
  if (string(argv[i]) == "-a" && string(argv[i]) != "-alpb" && i + 1 < argc) {
    arg_filename = string(argv[i + 1]);
    const int arg_file_status = parse_arg_file();
    if (arg_file_status != titan_cli::kParseSuccess)
      return option_exit(arg_file_status);
    ++i;
    return option_consumed();
  }
  if (string(argv[i]) == "-i" && i + 1 < argc) {
    list_filename = string(argv[i + 1]);
    fstream test_listFile(list_filename, ios::in);
    if (test_listFile.is_open())
      flag_listfile = true;
    else {
      flag_listfile = false;
      cout << "Cannot open " << list_filename << " file!\n";
      exit(EXIT_FAILURE);
    }
    test_listFile.close();
  }
  if (string(argv[i]) == "-water-type" && i + 1 < argc) {
    if (string(argv[i + 1]) == "surface" || string(argv[i + 1]) == "sphere")
      water_type = string(argv[i + 1]);
    else {
      cout << "WARNING!!! -water-type option is incorrect!\n";
    }
  }
  if (string(argv[i]) == "-dis-water" && i + 1 < argc) {
    d_crit_water = parse_numeric_option(argv[i + 1], "-dis-water");
  }
  if (string(argv[i]) == "-all-respair" && i + 1 < argc) {
    flag_all_respair = true;
  }
  if (string(argv[i]) == "-pull-direct") {
    de.flag_find_pulling_direct = true;
  }
  if (arg.rfind("-pull-opt=", 0) == 0) {
    de.pull_opt = arg.substr(10);
  }
  if (string(argv[i]) == "-pull-opt" && i + 1 < argc) {
    de.pull_opt = string(argv[i + 1]);
    ++i;
  }
  if (arg.rfind("-pull-max-evals=", 0) == 0) {
    de.pull_max_evals = static_cast<size_t>(titan_cli::parse_unsigned_option(arg.substr(16), "-pull-max-evals"));
  }
  if (string(argv[i]) == "-pull-max-evals" && i + 1 < argc) {
    de.pull_max_evals = static_cast<size_t>(titan_cli::parse_unsigned_option(argv[i + 1], "-pull-max-evals"));
    ++i;
  }
  if (arg.rfind("-pull-seed=", 0) == 0) {
    de.pull_seed = static_cast<unsigned int>(
        titan_cli::parse_unsigned_option(arg.substr(11), "-pull-seed", std::numeric_limits<unsigned int>::max()));
  }
  if (string(argv[i]) == "-pull-seed" && i + 1 < argc) {
    de.pull_seed = static_cast<unsigned int>(
        titan_cli::parse_unsigned_option(argv[i + 1], "-pull-seed", std::numeric_limits<unsigned int>::max()));
    ++i;
  }
  if (arg.rfind("-de-diff-weight=", 0) == 0) {
    de.de_diff_weight = parse_numeric_option(arg.substr(16), "-de-diff-weight");
    de.user_set_de_diff_weight = true;
  }
  if (string(argv[i]) == "-de-diff-weight" && i + 1 < argc) {
    de.de_diff_weight = parse_numeric_option(argv[i + 1], "-de-diff-weight");
    de.user_set_de_diff_weight = true;
  }
  if (arg.rfind("-de-cross-prob=", 0) == 0) {
    de.de_cross_prob = parse_numeric_option(arg.substr(15), "-de-cross-prob");
    de.user_set_de_cross_prob = true;
  }
  if (string(argv[i]) == "-de-cross-prob" && i + 1 < argc) {
    de.de_cross_prob = parse_numeric_option(argv[i + 1], "-de-cross-prob");
    de.user_set_de_cross_prob = true;
  }
  if (arg.rfind("-de-F=", 0) == 0) {
    de.de_F = parse_numeric_option(arg.substr(6), "-de-F");
    de.user_set_de_F = true;
  }
  if (string(argv[i]) == "-de-F" && i + 1 < argc) {
    de.de_F = parse_numeric_option(argv[i + 1], "-de-F");
    de.user_set_de_F = true;
  }
  if (string(argv[i]) == "-de-scan-radius" && i + 1 < argc) {
    de.de_scan_radius = parse_numeric_option(argv[i + 1], "-de-scan-radius");
  }
  if (string(argv[i]) == "-de-min-scan-radius" && i + 1 < argc) {
    de.de_min_scan_radius = parse_numeric_option(argv[i + 1], "-de-min-scan-radius");
  }
  if (string(argv[i]) == "-de-score-function" && i + 1 < argc) {
    de.de_flag_score_function = parse_integer_option(argv[i + 1], "-de-score-function");
  }
  if (string(argv[i]) == "-de-pop-num" && i + 1 < argc) {
    de.de_pop_num = parse_integer_option(argv[i + 1], "-de-pop-num");
  }
  if (string(argv[i]) == "-de-ite-num" && i + 1 < argc) {
    de.de_ite = parse_integer_option(argv[i + 1], "-de-ite-num");
  }
  if (arg.rfind("-de-algorithm=", 0) == 0) {
    de.de_algorithm = parse_integer_option(arg.substr(14), "-de-algorithm");
    de.user_set_de_algorithm = true;
  }
  if (string(argv[i]) == "-de-algorithm" && i + 1 < argc) {
    de.de_algorithm = parse_integer_option(argv[i + 1], "-de-algorithm");
    de.user_set_de_algorithm = true;
  }
  if (string(argv[i]) == "-de-angle" && i + 1 < argc) {
    de.de_restrain_angle = parse_numeric_option(argv[i + 1], "-de-angle");
  }
  if (string(argv[i]) == "-de-auto-term") {
    de.flag_de_auto_term = true;
  }
  if (string(argv[i]) == "-de-adaptive-F") {
    de.de_adaptive_F = true;
    de.user_set_de_adaptive_F = true;
  }
  if (string(argv[i]) == "-de-stagnation-limit" && i + 1 < argc) {
    de.de_stagnation_limit = parse_integer_option(argv[i + 1], "-de-stagnation-limit");
  }
  if (arg.rfind("-de-memory-size=", 0) == 0) {
    de.de_memory_size = static_cast<size_t>(titan_cli::parse_unsigned_option(arg.substr(16), "-de-memory-size"));
    de.user_set_de_memory_size = true;
  }
  if (string(argv[i]) == "-de-memory-size" && i + 1 < argc) {
    de.de_memory_size = static_cast<size_t>(titan_cli::parse_unsigned_option(argv[i + 1], "-de-memory-size"));
    de.user_set_de_memory_size = true;
  }
  if (arg.rfind("-de-p-best-rate=", 0) == 0) {
    de.de_p_best_rate = parse_numeric_option(arg.substr(16), "-de-p-best-rate");
    de.user_set_de_p_best_rate = true;
  }
  if (string(argv[i]) == "-de-p-best-rate" && i + 1 < argc) {
    de.de_p_best_rate = parse_numeric_option(argv[i + 1], "-de-p-best-rate");
    de.user_set_de_p_best_rate = true;
  }
  if (arg.rfind("-de-archive-scale=", 0) == 0) {
    de.de_archive_scale = parse_numeric_option(arg.substr(18), "-de-archive-scale");
    de.user_set_de_archive_scale = true;
  }
  if (string(argv[i]) == "-de-archive-scale" && i + 1 < argc) {
    de.de_archive_scale = parse_numeric_option(argv[i + 1], "-de-archive-scale");
    de.user_set_de_archive_scale = true;
  }
  if (arg.rfind("-de-min-pop-num=", 0) == 0) {
    de.de_min_pop_num = static_cast<size_t>(titan_cli::parse_unsigned_option(arg.substr(16), "-de-min-pop-num"));
    de.user_set_de_min_pop_num = true;
  }
  if (string(argv[i]) == "-de-min-pop-num" && i + 1 < argc) {
    de.de_min_pop_num = static_cast<size_t>(titan_cli::parse_unsigned_option(argv[i + 1], "-de-min-pop-num"));
    de.user_set_de_min_pop_num = true;
  }
  if (string(argv[i]) == "-de-pathway-group" && i + 1 < argc) {
    de.group_pathway = static_cast<size_t>(titan_cli::parse_unsigned_option(argv[i + 1], "-de-pathway-group"));
    de.flag_pathway = true;
    de.user_set_pathway_group = true;
  }
  if (string(argv[i]) == "-de-pathway-radius" && i + 1 < argc) {
    de.dis_along_path = parse_numeric_option(argv[i + 1], "-de-pathway-radius");
    // PATH-01: resolved in post_process, after every explicit option.
    de.path_radius_alias = true;
  }
  if (string(argv[i]) == "-pathway-file" && i + 1 < argc) {
    de.pathway_file = string(argv[i + 1]);
    de.flag_pathway = true;
  }
  if (string(argv[i]) == "-pathway-route-id" && i + 1 < argc) {
    de.pathway_route_id = string(argv[i + 1]);
  }
  if (string(argv[i]) == "-pathway-previous-status" && i + 1 < argc) {
    de.path_previous_status = string(argv[i + 1]);
    de.flag_pathway = true;
  }
  if (string(argv[i]) == "-pathway-continuity-window" && i + 1 < argc) {
    de.path_continuity_window = parse_numeric_option(
        string(argv[i + 1]), "-pathway-continuity-window");
    de.flag_pathway = true;
  }
  if (string(argv[i]) == "-pathway-graph-edge-cutoff" && i + 1 < argc) {
    de.path_graph_edge_cutoff = parse_numeric_option(argv[i + 1], "-pathway-graph-edge-cutoff");
    de.path_edge_cutoff_explicit = true;
  }
  if (string(argv[i]) == "-pathway-recovery-enter" && i + 1 < argc) {
    de.path_recovery_enter_radius = parse_numeric_option(argv[i + 1], "-pathway-recovery-enter");
    de.path_recovery_enter_explicit = true;
  }
  if (string(argv[i]) == "-pathway-recovery-exit" && i + 1 < argc) {
    de.path_recovery_exit_radius = parse_numeric_option(argv[i + 1], "-pathway-recovery-exit");
    de.path_recovery_exit_explicit = true;
  }
  if (string(argv[i]) == "-pathway-recovery-max-turn-angle" &&
      i + 1 < argc) {
    de.path_recovery_max_turn_angle = parse_numeric_option(argv[i + 1], "-pathway-recovery-max-turn-angle");
  }
  if (string(argv[i]) == "-pathway-corridor-spacing" && i + 1 < argc) {
    de.path_corridor_spacing = parse_numeric_option(argv[i + 1], "-pathway-corridor-spacing");
  }
  if (string(argv[i]) == "-pathway-corridor-radial-samples" &&
      i + 1 < argc) {
    de.path_corridor_radial_samples =
        static_cast<size_t>(titan_cli::parse_unsigned_option(argv[i + 1], "-pathway-corridor-radial-samples"));
  }
  if (string(argv[i]) == "-pathway-registration-max-rmsd" && i + 1 < argc) {
    de.path_registration_max_rmsd = parse_numeric_option(argv[i + 1], "-pathway-registration-max-rmsd");
  }
  if (string(argv[i]) == "-pathway-completion-tolerance" && i + 1 < argc) {
    de.path_completion_tolerance = parse_numeric_option(argv[i + 1], "-pathway-completion-tolerance");
  }
  if (string(argv[i]) == "-pathway-completion-min-separation" &&
      i + 1 < argc) {
    de.path_completion_min_separation = parse_numeric_option(argv[i + 1], "-pathway-completion-min-separation");
  }
  if (string(argv[i]) == "-pathway-max-turn-angle" && i + 1 < argc) {
    de.path_max_turn_angle = parse_numeric_option(argv[i + 1], "-pathway-max-turn-angle");
  }
  if (string(argv[i]) == "-pathway-energy-key-epsilon" && i + 1 < argc) {
    de.path_energy_key_epsilon = parse_numeric_option(argv[i + 1], "-pathway-energy-key-epsilon");
  }
  if (string(argv[i]) == "-pathway-min-clearance" && i + 1 < argc) {
    de.path_min_clearance = parse_numeric_option(argv[i + 1], "-pathway-min-clearance");
  }
  if (string(argv[i]) == "-pathway-probe-length" && i + 1 < argc) {
    de.path_probe_length = parse_numeric_option(argv[i + 1], "-pathway-probe-length");
    if (de.path_probe_length > 0.0)
      de.path_probe_lengths = {de.path_probe_length};
    else
      de.path_probe_lengths.clear();
  }
  if (arg.rfind("-pathway-probe-lengths=", 0) == 0) {
    parse_path_probe_lengths(arg.substr(23));
  }
  if (string(argv[i]) == "-pathway-probe-lengths" && i + 1 < argc) {
    parse_path_probe_lengths(string(argv[i + 1]));
  }
  if (string(argv[i]) == "-pathway-navigation") {
    de.path_navigation_enabled = true;
  }
  if (arg.rfind("-pathway-navigation-tube-weight=", 0) == 0) {
    de.path_navigation_tube_weight = parse_numeric_option(
        arg.substr(32), "-pathway-navigation-tube-weight");
  }
  if (string(argv[i]) == "-pathway-navigation-tube-weight" &&
      i + 1 < argc) {
    de.path_navigation_tube_weight = parse_numeric_option(
        string(argv[i + 1]), "-pathway-navigation-tube-weight");
  }
  if (arg.rfind("-pathway-navigation-backtrack-weight=", 0) == 0) {
    de.path_navigation_backtrack_weight = parse_numeric_option(
        arg.substr(37), "-pathway-navigation-backtrack-weight");
  }
  if (string(argv[i]) == "-pathway-navigation-backtrack-weight" &&
      i + 1 < argc) {
    de.path_navigation_backtrack_weight = parse_numeric_option(
        string(argv[i + 1]), "-pathway-navigation-backtrack-weight");
  }
  if (arg.rfind("-pathway-navigation-cost-epsilon=", 0) == 0) {
    de.path_navigation_cost_epsilon = parse_numeric_option(
        arg.substr(33), "-pathway-navigation-cost-epsilon");
  }
  if (string(argv[i]) == "-pathway-navigation-cost-epsilon" &&
      i + 1 < argc) {
    de.path_navigation_cost_epsilon = parse_numeric_option(
        string(argv[i + 1]), "-pathway-navigation-cost-epsilon");
  }
  if (arg.rfind("-pathway-navigation-seed-radial-samples=", 0) == 0) {
    de.path_navigation_seed_radial_samples = static_cast<size_t>(
        parse_integer_option(arg.substr(40),
                             "-pathway-navigation-seed-radial-samples"));
  }
  if (string(argv[i]) == "-pathway-navigation-seed-radial-samples" &&
      i + 1 < argc) {
    de.path_navigation_seed_radial_samples = static_cast<size_t>(
        parse_integer_option(
            string(argv[i + 1]),
            "-pathway-navigation-seed-radial-samples"));
  }
  if (arg.rfind("-pathway-navigation-seed-amplitude=", 0) == 0) {
    de.path_navigation_seed_amplitude = parse_numeric_option(
        arg.substr(35), "-pathway-navigation-seed-amplitude");
  }
  if (string(argv[i]) == "-pathway-navigation-seed-amplitude" &&
      i + 1 < argc) {
    de.path_navigation_seed_amplitude = parse_numeric_option(
        string(argv[i + 1]), "-pathway-navigation-seed-amplitude");
  }
  if (arg.rfind("-pathway-clearance-mode=", 0) == 0) {
    de.path_clearance_mode = require_choice(
        arg.substr(24), {"endpoint-center", "swept-linear-center"},
        "-pathway-clearance-mode");
  }
  if (string(argv[i]) == "-pathway-clearance-mode" && i + 1 < argc) {
    de.path_clearance_mode = require_choice(
        string(argv[i + 1]),
        {"endpoint-center", "swept-linear-center"},
        "-pathway-clearance-mode");
  }
  if (string(argv[i]) == "-pathway-geometry-only") {
    de.path_geometry_only = true;
  }
  // --- Flags previously only in arg-file parser, now unified ---
  if (string(argv[i]) == "-new_al" || string(argv[i]) == "-new-al") {
    new_al_1pdb = true;
  }
  if (string(argv[i]) == "-refpbc" && i + 1 < argc) {
    pbc.solv_pbc_int = 1;
    pbc.chain_ref = 0;
    pbc.ref_pdb_name = string(argv[i + 1]);
    if (pbc.ref_pdb_name.length() <= 4 ||
        pbc.ref_pdb_name.substr(pbc.ref_pdb_name.length() - 4) != ".pdb")
      pbc.ref_pdb_name += ".pdb";
    pbc.flag_pbc_ref = true;
    ++i;
  }
  if (string(argv[i]) == "-pbcm") {
    pbc.solv_pbc_int = 1;
    pbc.solv_pbc = 1;
    pbc.pbc_mem = true;
    if (i + 1 < argc && isdigit(argv[i + 1][0])) {
      pbc.solv_pbc = 0;
      pbc.chain_ref = parse_integer_option(argv[i + 1], "-pbcm");
      ++i;
    }
  }
  if (string(argv[i]) == "-pbc-envi") {
    while (i + 1 < argc && argv[i + 1][0] != '-') {
      string t = string(argv[++i]);
      if (t == "@")
        t = " ";
      pbc.pbc_envi_chain.emplace_back(t);
    }
    if (!pbc.pbc_envi_chain.empty())
      pbc.pbc_envi = true;
  }
  if (string(argv[i]) == "-com-dis" && i + 1 < argc) {
    string axis = string(argv[i + 1]);
    if (axis == "x")
      com_dis_2mol = 1;
    else if (axis == "y")
      com_dis_2mol = 2;
    else if (axis == "z")
      com_dis_2mol = 3;
    else
      com_dis_2mol = 4;
    ++i;
  }
  if (string(argv[i]) == "-cal-mindis") {
    flag_cal_mindis = true;
  }
  if ((string(argv[i]) == "-common_topo" ||
       string(argv[i]) == "-common-topo") &&
      i + 1 < argc) {
    flag_use_common_topo = true;
    if (flag_topo_1 == false && flag_topo_2 == false) {
      topo_name = string(argv[i + 1]);
      topo_name_2 = string(argv[i + 1]);
      flag_topo_1 = true;
      flag_topo_2 = true;
      mol_1_type = "protein-user";
      mol_2_type = "protein-user";
    }
    ++i;
  }
  if (string(argv[i]).find("-contact-map=") == 0) {
    string val = string(argv[i]).substr(string("-contact-map=").size());
    cmap.legacy_alias_seen = true;
    if (val == "no")
      cmap.legacy_alias_enabled = false;
    else if (val == "yes")
      cmap.legacy_alias_enabled = true;
  }
  if (string(argv[i]) == "-contact-map" && i + 1 < argc) {
    string val = string(argv[i + 1]);
    cmap.legacy_alias_seen = true;
    if (val == "no")
      cmap.legacy_alias_enabled = false;
    else if (val == "yes")
      cmap.legacy_alias_enabled = true;
    ++i;
  }
  if (string(argv[i]).find("-eq-weight-source=") == 0 ||
      string(argv[i]).find("--eq-weight-source=") == 0) {
    eq_weight_source = string(argv[i]).substr(string(argv[i]).find('=') + 1);
  }
  if ((string(argv[i]) == "-eq-weight-source" ||
       string(argv[i]) == "--eq-weight-source") &&
      i + 1 < argc) {
    eq_weight_source = string(argv[i + 1]);
    ++i;
  }
  if (string(argv[i]).find("-trajectory-semantics=") == 0 ||
      string(argv[i]).find("--trajectory-semantics=") == 0) {
    trajectory_semantics =
        string(argv[i]).substr(string(argv[i]).find('=') + 1);
  }
  if ((string(argv[i]) == "-trajectory-semantics" ||
       string(argv[i]) == "--trajectory-semantics") &&
      i + 1 < argc) {
    trajectory_semantics = string(argv[i + 1]);
    ++i;
  }
  return {};
}
