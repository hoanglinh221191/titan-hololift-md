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
int arguments::parse_arguments(int argc, char **argv) {
  if (!parsing_arg_file_tokens) {
    const titan_cli::ValidationResult validation =
        titan_cli::validate_terminal_tokens(argc, argv);
    titan_cli::emit_diagnostics(validation);
    if (!validation.ok())
      return titan_cli::kInvalidArguments;
  }
  auto normalize_mol_type = [](std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "ligand")
      value = "lig";
    return value;
  };

  auto assign_mol_type = [&](const std::string &value, bool mol1) {
    const std::string t = normalize_mol_type(value);
    if (t != "protein" && t != "protein-user" && t != "complex" && t != "lig" &&
        t != "rna" && t != "dna")
      return false;

    if (mol1) {
      if (t == "protein-user")
        return false;
      mol_1_type = t;
      mol_1_type_explicit = true;
    } else {
      mol_2_type = t;
      mol_2_type_explicit = true;
    }
    return true;
  };
  auto parse_allostery_prune_window = [&](std::string raw_mode) {
    std::transform(
        raw_mode.begin(), raw_mode.end(), raw_mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (raw_mode == "avg" || raw_mode == "all" || raw_mode == "mean") {
      allostery_use_avg_cutoff = true;
      return;
    }
    if (raw_mode == "first" || raw_mode == "frame0" || raw_mode == "instant") {
      allostery_use_avg_cutoff = false;
      return;
    }
    TitanUI::print_error("Allostery: prune window must be 'avg' or 'first'.");
    exit(EXIT_FAILURE);
  };
  auto parse_allostery_knn_prior = [&](std::string raw_mode) {
    std::transform(
        raw_mode.begin(), raw_mode.end(), raw_mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (raw_mode == "frame" || raw_mode == "nearest" || raw_mode == "vote") {
      allostery_knn_use_contact_prior = false;
      return;
    }
    if (raw_mode == "contact" || raw_mode == "contacts" ||
        raw_mode == "contact-frequency") {
      allostery_knn_use_contact_prior = true;
      return;
    }
    TitanUI::print_error(
        "Allostery: -allostery-knn-prior must be 'frame' or 'contact'.");
    exit(EXIT_FAILURE);
  };
  auto parse_corr_knn_prune_window = [&](std::string raw_mode) {
    std::transform(
        raw_mode.begin(), raw_mode.end(), raw_mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (raw_mode == "avg" || raw_mode == "all" || raw_mode == "mean") {
      corr_knn.use_avg_cutoff = true;
      return;
    }
    if (raw_mode == "first" || raw_mode == "frame0" || raw_mode == "instant") {
      corr_knn.use_avg_cutoff = false;
      return;
    }
    TitanUI::print_error("Corr-kNN: prune window must be 'avg' or 'first'.");
    exit(EXIT_FAILURE);
  };
  auto parse_corr_knn_pair_prior = [&](std::string raw_mode) {
    std::transform(
        raw_mode.begin(), raw_mode.end(), raw_mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (raw_mode == "frame" || raw_mode == "nearest" || raw_mode == "vote") {
      corr_knn.pair_use_contact_prior = false;
      return;
    }
    if (raw_mode == "contact" || raw_mode == "contacts" ||
        raw_mode == "contact-frequency") {
      corr_knn.pair_use_contact_prior = true;
      return;
    }
    TitanUI::print_error(
        "Corr-kNN: -corr-knn-pair-prior must be 'frame' or 'contact'.");
    exit(EXIT_FAILURE);
  };
  auto parse_validation_int_list = [&](const std::string &raw_list,
                                       std::vector<int> &out_values,
                                       const char *flag_name) {
    out_values.clear();
    std::string normalized = raw_list;
    std::replace(normalized.begin(), normalized.end(), ';', ',');
    std::istringstream iss(normalized);
    for (std::string token; std::getline(iss, token, ',');) {
      token.erase(
          std::remove_if(token.begin(), token.end(),
                         [](unsigned char c) { return std::isspace(c) != 0; }),
          token.end());
      if (token.empty())
        continue;
      out_values.push_back(std::stoi(token));
    }
    if (out_values.empty()) {
      TitanUI::print_error(std::string(flag_name) +
                           " expects a comma-separated integer list.");
      exit(EXIT_FAILURE);
    }
  };
  auto enable_validation = [&]() {
    validation.flag_validation = true;
    msm.flag_msm = true;
    msm.flag_tica = true;
    flag_network = true;
  };
  auto parse_allostery_estimator = [&](std::string raw_mode) {
    std::transform(
        raw_mode.begin(), raw_mode.end(), raw_mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (raw_mode == "hist" || raw_mode == "histogram") {
      allostery_estimator = "hist";
      return;
    }
    if (raw_mode == "knn") {
      allostery_estimator = "knn";
      return;
    }
    if (raw_mode == "kde") {
      allostery_estimator = "kde";
      return;
    }
    TitanUI::print_error(
        "Allostery: -allostery-estimator must be one of hist, knn, kde.");
    exit(EXIT_FAILURE);
  };
  auto parse_msm_its_solver = [&](std::string raw_mode) {
    std::transform(
        raw_mode.begin(), raw_mode.end(), raw_mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (raw_mode == "full") {
      msm.msm_its_solver = MsmArgs::ItsSolverMode::Full;
      return;
    }
    if (raw_mode == "schur") {
      msm.msm_its_solver = MsmArgs::ItsSolverMode::Schur;
      return;
    }
    if (raw_mode == "partial") {
      msm.msm_its_solver = MsmArgs::ItsSolverMode::Partial;
      return;
    }
    TitanUI::print_error(
        "MSM: -msm-its-solver must be one of full, schur, partial.");
    exit(EXIT_FAILURE);
  };

  auto parse_pca_solver = [&](std::string raw_mode) {
    std::transform(
        raw_mode.begin(), raw_mode.end(), raw_mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (raw_mode == "full") {
      msm.pca_solver = MsmArgs::PcaSolverMode::Full;
      return;
    }
    if (raw_mode == "partial") {
      msm.pca_solver = MsmArgs::PcaSolverMode::Partial;
      return;
    }
    TitanUI::print_error("PCA: -pca-solver must be one of full, partial.");
    exit(EXIT_FAILURE);
  };
  auto parse_sq_auto_or_value = [&](const std::string &raw_value, double &out,
                                    bool &auto_flag, const char *label) {
    std::string value = raw_value;
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "auto") {
      out = 0.0;
      auto_flag = true;
      return;
    }
    try {
      out = stod(raw_value);
      auto_flag = false;
    } catch (...) {
      TitanUI::print_error(std::string("SQ: invalid value for ") + label +
                           ". Use a positive number or 'auto'.");
      exit(EXIT_FAILURE);
    }
  };
  auto parse_hbk_domain = [&](std::string raw_mode) {
    std::transform(
        raw_mode.begin(), raw_mode.end(), raw_mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (raw_mode == "inter" || raw_mode == "intra" || raw_mode == "both") {
      hbk.hbk_domain = raw_mode;
      return;
    }
    TitanUI::print_error(
        "HBond kinetics: -hbk-domain must be one of inter, intra, both.");
    exit(EXIT_FAILURE);
  };
  auto parse_bool_equals = [&](std::string raw_value, const char *flag_name) {
    std::transform(
        raw_value.begin(), raw_value.end(), raw_value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (raw_value.empty() || raw_value == "1" || raw_value == "true" ||
        raw_value == "yes" || raw_value == "on")
      return true;
    if (raw_value == "0" || raw_value == "false" || raw_value == "no" ||
        raw_value == "off")
      return false;
    TitanUI::print_error(std::string(flag_name) +
                         " must be true/false (or 1/0, yes/no, on/off).");
    exit(EXIT_FAILURE);
  };
  auto consume_option_value = [&](const std::string &arg,
                                  std::initializer_list<const char *> flags,
                                  std::string &value_out, int &arg_index) {
    for (const char *flag : flags) {
      const std::string flag_text(flag);
      const std::string prefix = flag_text + "=";
      if (arg.rfind(prefix, 0) == 0) {
        value_out = arg.substr(prefix.size());
        return true;
      }
      if (arg == flag_text && arg_index + 1 < argc) {
        value_out = string(argv[arg_index + 1]);
        ++arg_index;
        return true;
      }
    }
    return false;
  };
  auto consume_required_option_value =
      [&](const std::string &arg, std::initializer_list<const char *> flags,
          std::string &value_out, int &arg_index) {
        const bool consumed =
            consume_option_value(arg, flags, value_out, arg_index);
        if (consumed)
          return true;
        for (const char *flag : flags) {
          if (arg == std::string(flag)) {
            TitanUI::print_error(std::string(flag) + " requires a value.");
            exit(EXIT_FAILURE);
          }
        }
        return false;
      };
  auto consume_bool_option = [&](const std::string &arg,
                                 std::initializer_list<const char *> flags,
                                 bool &value_out, int &arg_index) {
    for (const char *flag : flags) {
      const std::string flag_text(flag);
      const std::string prefix = flag_text + "=";
      if (arg == flag_text) {
        if (arg_index + 1 < argc) {
          std::string next = string(argv[arg_index + 1]);
          std::transform(next.begin(), next.end(), next.begin(),
                         [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                         });
          if (next == "1" || next == "0" || next == "true" || next == "false" ||
              next == "yes" || next == "no" || next == "on" || next == "off") {
            value_out = parse_bool_equals(next, flag);
            ++arg_index;
            return true;
          }
        }
        value_out = true;
        return true;
      }
      if (arg.rfind(prefix, 0) == 0) {
        value_out = parse_bool_equals(arg.substr(prefix.size()), flag);
        return true;
      }
    }
    return false;
  };
  auto parse_path_probe_lengths = [&](const std::string &raw_list) {
    de.path_probe_lengths.clear();
    std::string normalized = raw_list;
    std::replace(normalized.begin(), normalized.end(), ';', ',');
    std::istringstream iss(normalized);
    for (std::string token; std::getline(iss, token, ',');) {
      token.erase(
          std::remove_if(token.begin(), token.end(),
                         [](unsigned char c) { return std::isspace(c) != 0; }),
          token.end());
      if (token.empty())
        continue;
      std::size_t parsed_chars = 0;
      double value = 0.0;
      try {
        value = std::stod(token, &parsed_chars);
      } catch (const std::exception &) {
        TitanUI::print_error(
            "-pathway-probe-lengths expects positive finite distances.");
        exit(EXIT_FAILURE);
      }
      if (parsed_chars != token.size() || !std::isfinite(value) ||
          value <= 0.0) {
        TitanUI::print_error(
            "-pathway-probe-lengths expects positive finite distances.");
        exit(EXIT_FAILURE);
      }
      de.path_probe_lengths.push_back(value);
    }
    if (de.path_probe_lengths.empty()) {
      TitanUI::print_error(
          "-pathway-probe-lengths expects a comma-separated distance list.");
      exit(EXIT_FAILURE);
    }
    de.path_probe_length = de.path_probe_lengths.front();
  };
  auto consume_compound_option_value =
      [&](const std::string &arg, std::initializer_list<const char *> flags,
          std::string &value_out, int &arg_index) {
        for (const char *flag : flags) {
          const std::string flag_text(flag);
          const std::string prefix = flag_text + "=";
          if (arg.rfind(prefix, 0) == 0) {
            value_out = arg.substr(prefix.size());
          } else if (arg == flag_text && arg_index + 1 < argc) {
            value_out = string(argv[++arg_index]);
          } else {
            continue;
          }
          while (arg_index + 1 < argc &&
                 !titan_cli::looks_like_option(argv[arg_index + 1])) {
            if (!value_out.empty())
              value_out.push_back(' ');
            value_out += argv[++arg_index];
          }
          return true;
        }
        return false;
      };
  auto lowercase_copy = [](std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
  };
  auto require_choice = [&](std::string value,
                            std::initializer_list<const char *> choices,
                            const char *flag_name) {
    value = lowercase_copy(value);
    for (const char *choice : choices) {
      if (value == choice)
        return value;
    }
    std::ostringstream msg;
    msg << flag_name << " must be one of ";
    bool first = true;
    for (const char *choice : choices) {
      if (!first)
        msg << ", ";
      msg << choice;
      first = false;
    }
    TitanUI::print_error(msg.str());
    exit(EXIT_FAILURE);
  };
  auto parse_double_range = [&](const std::string &value, const char *flag_name,
                                double min_value, double max_value) {
    double parsed = 0.0;
    try {
      parsed = std::stod(value);
    } catch (...) {
      std::ostringstream msg;
      msg << flag_name << " must be numeric.";
      TitanUI::print_error(msg.str());
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(parsed) || parsed < min_value || parsed > max_value) {
      std::ostringstream msg;
      msg << flag_name << " must be in [" << min_value << ", " << max_value
          << "].";
      TitanUI::print_error(msg.str());
      exit(EXIT_FAILURE);
    }
    return parsed;
  };
  auto parse_numeric_option = [&](const std::string &value,
                                  const char *flag_name) {
    double parsed = 0.0;
    std::size_t parsed_chars = 0;
    try {
      parsed = std::stod(value, &parsed_chars);
    } catch (...) {
      TitanUI::print_error(std::string(flag_name) + " must be numeric.");
      exit(EXIT_FAILURE);
    }
    if (parsed_chars != value.size() || !std::isfinite(parsed)) {
      TitanUI::print_error(std::string(flag_name) + " must be numeric.");
      exit(EXIT_FAILURE);
    }
    return parsed;
  };
  auto parse_integer_option = [&](const std::string &value,
                                  const char *flag_name) {
    int parsed = 0;
    std::size_t parsed_chars = 0;
    try {
      parsed = std::stoi(value, &parsed_chars);
    } catch (...) {
      TitanUI::print_error(std::string(flag_name) + " must be an integer.");
      exit(EXIT_FAILURE);
    }
    if (parsed_chars != value.size()) {
      TitanUI::print_error(std::string(flag_name) + " must be an integer.");
      exit(EXIT_FAILURE);
    }
    return parsed;
  };
  auto normalize_pbe_cli_alias = [](std::string arg) {
    constexpr const char *legacy_short = "-pbe-tabi-";
    constexpr const char *legacy_long = "--pbe-tabi-";
    constexpr const char *public_short = "-pbe-bie-";
    constexpr const char *public_long = "--pbe-bie-";
    if (arg.rfind(legacy_short, 0) == 0) {
      arg.replace(0, std::char_traits<char>::length(legacy_short), public_short);
    } else if (arg.rfind(legacy_long, 0) == 0) {
      arg.replace(0, std::char_traits<char>::length(legacy_long), public_long);
    }
    return arg;
  };
  auto apply_tabi_tree_preset = [&](const std::string &preset) {
    if (preset == "off")
      return;
    if (preset == "stage3u-safe" || preset == "stage3u-perf" ||
        preset == "stage3u-perf2") {
      const bool perf = preset == "stage3u-perf" || preset == "stage3u-perf2";
      const bool perf2 = preset == "stage3u-perf2";
      pbe.tabi_sum = "treecode";
      pbe.tabi_formulation = "jcp2013-treecode";
      pbe.tabi_tree_leaf_size = 128;
      pbe.tabi_tree_opening_theta = 0.60;
      pbe.tabi_taylor_order = 4;
      pbe.tabi_precond = "diagonal";
      pbe.tabi_near_integration = "duffy_adaptive";
      pbe.tabi_quad_buffer = "adaptive";
      pbe.tabi_interaction_mode = "onthefly";
      pbe.tabi_node_bounds = "tight-aabb";
      pbe.tabi_tree_builder = "topdown-aabb";
      pbe.tabi_expansion_center = "area-centroid";
      pbe.tabi_fragmentation_safety = "split";
      pbe.tabi_mac_safety = "physical-gap";
      pbe.tabi_moment_audit = perf ? "off" : "sampled";
      pbe.tabi_moment_construction = "translated";
      pbe.tabi_moment_sum = "naive";
      pbe.tabi_convergence_audit = perf ? "off" : "sampled";
      pbe.tabi_convergence_safety = "split";
      pbe.tabi_convergence_rho_cap = 0.0;
      pbe.tabi_mac_aspect_cap = 0.0;
      pbe.tabi_convergence_estimator = "hybrid";
      pbe.tabi_convergence_tail_threshold = 1.0e4;
      pbe.tabi_convergence_estimator_error_cap = 1.0e-3;
      pbe.tabi_convergence_unstable_rho_cap = 0.25;
      pbe.tabi_convergence_calibration = "consecutive-aspect";
      pbe.tabi_convergence_dielectric_amp = "on";
      pbe.tabi_convergence_estimator_model = "channel-tail-eps";
      pbe.tabi_convergence_channel_calibration = "polynomial-eps";
      pbe.tabi_convergence_split_policy = "global";
      pbe.tabi_channel_missed_semantics = "accepted-only";
      pbe.tabi_accepted_miss_diagnostics = "summary";
      pbe.tabi_accepted_miss_absolute_floor = 1.0e-7;
      pbe.tabi_convergence_k3_accepted_rho_cap = 0.0;
      pbe.tabi_mac_distance_policy = "surface-margin";
      pbe.tabi_mac_surface_margin_factor = 0.25;
      pbe.tabi_mac_surface_margin_policy = "all";
      pbe.tabi_mac_surface_margin_aspect_threshold = 3.0;
      pbe.tabi_mac_surface_margin_rho_trigger = 0.55;
      pbe.tabi_leaf_panel_kernel = perf ? "tile-avx2" : "auto";
      pbe.tabi_cluster_policy = perf ? "leaf-pair-balanced" : "off";
      pbe.tabi_leaf_pair_kernel = perf ? "target-avx2" : "auto";
      pbe.tabi_leaf_kernel_profile = "off";
      pbe.tabi_leaf_work_profile = "off";
      pbe.tabi_matvec_cost_profile = "off";
      pbe.tabi_jcp_direct_kernel = "baseline";
      pbe.tabi_target_batch_size = perf ? 4 : 1;
      pbe.tabi_leaf_pair_min_source_panels = 64;
      pbe.tabi_leaf_pair_min_targets = 4;
      pbe.tabi_leaf_pair_selection_profile = perf2 ? "summary" : "off";
      pbe.tabi_leaf_pair_probe_cache = perf2 ? "per-matvec" : "off";
      pbe.tabi_leaf_pair_target_group = perf2 ? "same-target-leaf" : "scan";
      pbe.tabi_leaf_pair_directness_probe = "full-traversal";
      pbe.tabi_leaf_pair_directness_cache = "off";
      pbe.tabi_leaf_pair_directness_verify = "off";
      pbe.tabi_leaf_pair_scheduler = "inline-select";
      pbe.tabi_leaf_pair_worklist_max_active = 2048;
      pbe.tabi_leaf_pair_worklist_verify = "off";
      pbe.tabi_leaf_block_verify = "off";
      pbe.tabi_leaf_pair_max_lookahead = 32;
      pbe.tabi_expansion_audit = perf ? "off" : "sampled";
      pbe.tabi_expansion_audit_channel = perf ? "off" : "summary";
      pbe.tabi_matvec_audit = "off";
      return;
    }
    TitanUI::print_error("PBE: -pbe-bie-tree-preset must be off, "
                         "stage3u-safe, stage3u-perf, or stage3u-perf2.");
    exit(EXIT_FAILURE);
  };
  auto parse_pbe_test_center = [&](std::string value) {
    value = lowercase_copy(value);
    if (value == "auto") {
      pbe.test_center_user = false;
      return;
    }
    std::replace(value.begin(), value.end(), ',', ' ');
    std::istringstream iss(value);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(iss >> x >> y >> z)) {
      TitanUI::print_error("PBE: -pbe-test-center must be auto or x,y,z.");
      exit(EXIT_FAILURE);
    }
    std::string trailing;
    if (iss >> trailing) {
      TitanUI::print_error(
          "PBE: -pbe-test-center must contain exactly three coordinates.");
      exit(EXIT_FAILURE);
    }
    pbe.test_center_user = true;
    pbe.test_center_x = x;
    pbe.test_center_y = y;
    pbe.test_center_z = z;
  };
  auto parse_pbe_focus_center = [&](std::string value) {
    value = lowercase_copy(value);
    if (value == "auto") {
      pbe.focus_center_user = false;
      return;
    }
    std::replace(value.begin(), value.end(), ',', ' ');
    std::istringstream iss(value);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(iss >> x >> y >> z)) {
      TitanUI::print_error("PBE: -pbe-focus-center must be auto or x,y,z.");
      exit(EXIT_FAILURE);
    }
    std::string trailing;
    if (iss >> trailing) {
      TitanUI::print_error(
          "PBE: -pbe-focus-center must contain exactly three coordinates.");
      exit(EXIT_FAILURE);
    }
    pbe.focus_center_user = true;
    pbe.focus_center_x = x;
    pbe.focus_center_y = y;
    pbe.focus_center_z = z;
  };
  auto parse_memory_budget = [&](std::string value) {
    mem_budget_gb_raw = lowercase_copy(value);
    if (mem_budget_gb_raw == "auto") {
      mem_budget_gb = -1.0;
      return;
    }
    try {
      mem_budget_gb = stod(value);
      if (mem_budget_gb <= 0.0)
        throw std::invalid_argument("non-positive");
    } catch (...) {
      TitanUI::print_error(
          "-mem-budget-gb expects 'auto' or a positive GB value.");
      exit(EXIT_FAILURE);
    }
  };
  auto parse_sg_memory_budget = [&](std::string value) {
    sg_memory_gb_raw = lowercase_copy(value);
    if (sg_memory_gb_raw == "auto") {
      sg_memory_gb = -1.0;
      return;
    }
    try {
      sg_memory_gb = stod(value);
      if (sg_memory_gb <= 0.0)
        throw std::invalid_argument("non-positive");
    } catch (...) {
      TitanUI::print_error(
          "-sg-memory-gb expects 'auto' or a positive GB value.");
      exit(EXIT_FAILURE);
    }
  };
  auto parse_sg_cutoff = [&](std::string value) {
    sg_cutoff = lowercase_copy(value);
    if (sg_cutoff == "auto" || sg_cutoff == "none") {
      sg_cutoff_value_nm = -1.0;
      return;
    }
    try {
      sg_cutoff_value_nm = stod(value);
      if (sg_cutoff_value_nm <= 0.0)
        throw std::invalid_argument("non-positive");
      sg_cutoff = "value";
    } catch (...) {
      TitanUI::print_error(
          "-sg-cutoff expects auto, none, or a positive cutoff in nm.");
      exit(EXIT_FAILURE);
    }
  };
  for (int i = 1; i < argc; ++i) {
    // Remember what is being parsed so a std::sto* failure can name the option
    // instead of surfacing a bare "stoi"/"stod".
    titan_cli::active_option_token() = argv[i];
    const std::string arg = normalize_pbe_cli_alias(string(argv[i]));
    std::string common_value;

    if (arg == "-useForceOptionforEnergy" ||
        arg.rfind("-useForceOptionforEnergy=", 0) == 0) {
      TitanUI::print_error(
          "Energy decomposition: legacy option " + arg +
          " is no longer supported; use -edecomp=true for energy "
          "decomposition and keep -fmap for force maps.");
      exit(EXIT_FAILURE);
    }
    if (arg == "-emap" || arg.rfind("-emap=", 0) == 0 || arg == "-emap-level" ||
        arg.rfind("-emap-level=", 0) == 0 || arg == "-energy-res" ||
        arg.rfind("-energy-res=", 0) == 0 || arg == "-new-emap" ||
        arg.rfind("-new-emap=", 0) == 0) {
      TitanUI::print_error(
          "Energy decomposition: legacy option " + arg +
          " is no longer supported; use -edecomp=true and the -edecomp-* "
          "options.");
      exit(EXIT_FAILURE);
    }
    if (consume_option_value(arg, {"-f"}, common_value, i)) {
      pdb_name = common_value;
      flag_1_pdb = true;
      continue;
    }
    if (consume_option_value(arg, {"-s"}, common_value, i)) {
      pdb_name = common_value;
      flag_1_pdb = true;
      flag_xdr = true;
      continue;
    }
    if (consume_option_value(arg, {"-n"}, common_value, i)) {
      index_name = common_value;
      flag_index = true;
      continue;
    }
    if (consume_option_value(arg, {"-g"}, common_value, i)) {
      auto found = common_value.find("&");
      auto found_2 = common_value.find("-");
      if (found != std::string::npos) {
        group_1 = stoi(common_value.substr(0, found));
        group_2 = stoi(common_value.substr(found + 1));
        flag_groups_explicit = true;
      } else if (found_2 != std::string::npos) {
        group_1 = stoi(common_value.substr(0, found_2));
        group_2 = stoi(common_value.substr(found_2 + 1));
        flag_groups_explicit = true;
      } else {
        TitanUI::print_error("Your argument -g is invalid!");
        exit(EXIT_FAILURE);
      }
      if (group_1 == group_2)
        self_int = true;
      continue;
    }
    if (consume_option_value(arg, {"-ff"}, common_value, i)) {
      force_f = common_value;
      transform(force_f.begin(), force_f.end(), force_f.begin(), ::toupper);
      continue;
    }
    if (consume_bool_option(arg, {"-gbsa"}, flag_gbsa, i)) {
      continue;
    }
    if (consume_bool_option(arg, {"-cas"}, flag_cas, i)) {
      continue;
    }
    if (consume_bool_option(arg, {"-edecomp", "--edecomp"},
                            edecomp.flag_edecomp, i)) {
      continue;
    }
    if (consume_option_value(arg, {"-edecomp-source", "--edecomp-source"},
                             common_value, i)) {
      edecomp.flag_edecomp = true;
      edecomp.source = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-edecomp-level", "--edecomp-level"},
                             common_value, i)) {
      edecomp.flag_edecomp = true;
      edecomp.level = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-edecomp-edge-top", "--edecomp-edge-top"},
                             common_value, i)) {
      edecomp.flag_edecomp = true;
      edecomp.edge_top = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-edecomp-rank-by", "--edecomp-rank-by"},
                             common_value, i)) {
      edecomp.flag_edecomp = true;
      edecomp.rank_by = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-edecomp-frame-output", "--edecomp-frame-output"},
            common_value, i)) {
      edecomp.flag_edecomp = true;
      edecomp.frame_output = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-edecomp-perf", "--edecomp-perf"},
                             common_value, i)) {
      edecomp.flag_edecomp = true;
      edecomp.perf = lowercase_copy(common_value);
      continue;
    }
    if (consume_bool_option(arg, {"-hotspot-report", "--hotspot-report"},
                            hotspot.flag_hotspot_report, i)) {
      continue;
    }
    if (consume_option_value(arg, {"-hotspot-source", "--hotspot-source"},
                             common_value, i)) {
      hotspot.flag_hotspot_report = true;
      hotspot.source = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-hotspot-rank-by", "--hotspot-rank-by"},
                             common_value, i)) {
      hotspot.flag_hotspot_report = true;
      hotspot.rank_by = lowercase_copy(common_value);
      hotspot.rank_by_explicit = true;
      continue;
    }
    if (consume_option_value(arg, {"-perf-report", "--perf-report"},
                             common_value, i)) {
      perf_report.mode = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-uncertainty", "--uncertainty"},
                             common_value, i)) {
      uncertainty.mode = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-uncertainty-blocks", "--uncertainty-blocks"},
                             common_value, i)) {
      uncertainty.blocks = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-uncertainty-bootstrap", "--uncertainty-bootstrap"},
            common_value, i)) {
      uncertainty.bootstrap_samples = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-uncertainty-seed", "--uncertainty-seed"},
                             common_value, i)) {
      uncertainty.seed = static_cast<unsigned int>(stoul(common_value));
      continue;
    }
    if (consume_bool_option(arg, {"-dry-run", "--dry-run"},
                            output_manifest.dry_run, i)) {
      if (output_manifest.dry_run)
        output_manifest.output_manifest = true;
      continue;
    }
    if (consume_bool_option(arg, {"-output-manifest", "--output-manifest"},
                            output_manifest.output_manifest, i)) {
      continue;
    }
    if (consume_option_value(arg, {"-schema-export", "--schema-export"},
                             common_value, i)) {
      output_manifest.schema_export = lowercase_copy(common_value);
      continue;
    }

    if (arg.rfind("-mol-1-type=", 0) == 0) {
      assign_mol_type(arg.substr(12), true);
      continue;
    }
    if (arg.rfind("-mol-2-type=", 0) == 0) {
      assign_mol_type(arg.substr(12), false);
      continue;
    }

    if (consume_required_option_value(arg, {"-community", "--community"},
                                      common_value, i)) {
      community.flag_community = true;
      community.method = lowercase_copy(common_value);
      flag_network = true;
      continue;
    }

    if (consume_bool_option(arg, {"-graph", "--graph"}, graph.flag_graph, i)) {
      continue;
    }
    if (consume_bool_option(arg, {"-feature-export", "--feature-export"},
                            feature_export.export_features, i)) {
      continue;
    }
    if (consume_option_value(arg, {"-feature-source", "--feature-source"},
                             common_value, i)) {
      feature_export.source = common_value;
      feature_export.source_explicit = true;
      feature_export.export_features = true;
      continue;
    }
    if (consume_option_value(arg, {"-segment-config", "--segment-config"},
                             common_value, i)) {
      segments.config_file = common_value;
      continue;
    }
    if (consume_bool_option(
            arg, {"-interface-analysis", "--interface-analysis"},
            interface_analysis.flag_interface_analysis, i)) {
      continue;
    }
    if (consume_option_value(
            arg, {"-interface-config", "--interface-config"}, common_value,
            i)) {
      segments.config_file = common_value;
      interface_analysis.flag_interface_analysis = true;
      continue;
    }
    if (consume_option_value(
            arg, {"-interface-contact-mode", "--interface-contact-mode"},
            common_value, i)) {
      interface_analysis.flag_interface_analysis = true;
      interface_analysis.contact_mode = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-interface-cut", "--interface-cut"},
                             common_value, i)) {
      interface_analysis.flag_interface_analysis = true;
      interface_analysis.cutoff_ang = stod(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-interface-state-min-contacts",
             "--interface-state-min-contacts"},
            common_value, i)) {
      interface_analysis.flag_interface_analysis = true;
      interface_analysis.state_min_contacts = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-interface-sasa-stride", "--interface-sasa-stride"},
            common_value, i)) {
      interface_analysis.flag_interface_analysis = true;
      interface_analysis.sasa_stride = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-interface-sasa-points", "--interface-sasa-points"},
            common_value, i)) {
      interface_analysis.flag_interface_analysis = true;
      interface_analysis.sasa_points = stoi(common_value);
      continue;
    }
    if (consume_bool_option(arg, {"-cmap", "--cmap"}, cmap.flag_cmap, i)) {
      cmap.new_option_explicit = true;
      continue;
    }
    if (consume_option_value(arg, {"-cmap-mode", "--cmap-mode"}, common_value,
                             i)) {
      cmap.flag_cmap = true;
      cmap.new_option_explicit = true;
      cmap.mode = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-cmap-cut", "--cmap-cut"}, common_value,
                             i)) {
      cmap.flag_cmap = true;
      cmap.new_option_explicit = true;
      cmap.cutoff_explicit = true;
      cmap.cutoff_ang = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-cmap-scope", "--cmap-scope"}, common_value,
                             i)) {
      cmap.flag_cmap = true;
      cmap.new_option_explicit = true;
      cmap.scope = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-cmap-domain", "--cmap-domain"},
                             common_value, i)) {
      cmap.flag_cmap = true;
      cmap.new_option_explicit = true;
      cmap.domain_explicit = true;
      cmap.domain = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-cmap-min-seqsep", "--cmap-min-seqsep"},
                             common_value, i)) {
      cmap.flag_cmap = true;
      cmap.new_option_explicit = true;
      cmap.min_seqsep_explicit = true;
      cmap.min_seqsep = stoi(common_value);
      continue;
    }
    if (consume_bool_option(arg, {"-qnative", "--qnative"},
                            qnative.flag_qnative, i)) {
      continue;
    }
    if (consume_option_value(arg, {"-qnative-ref", "--qnative-ref"},
                             common_value, i)) {
      qnative.flag_qnative = true;
      qnative.ref_explicit = true;
      qnative.ref_file = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-qnative-mode", "--qnative-mode"},
                             common_value, i)) {
      qnative.flag_qnative = true;
      qnative.mode = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-qnative-cut", "--qnative-cut"},
                             common_value, i)) {
      qnative.flag_qnative = true;
      qnative.cutoff_explicit = true;
      qnative.cutoff_ang = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-qnative-scope", "--qnative-scope"},
                             common_value, i)) {
      qnative.flag_qnative = true;
      qnative.scope = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-qnative-domain", "--qnative-domain"},
                             common_value, i)) {
      qnative.flag_qnative = true;
      qnative.domain_explicit = true;
      qnative.domain = common_value;
      continue;
    }
    if (consume_option_value(arg,
                             {"-qnative-min-seqsep", "--qnative-min-seqsep"},
                             common_value, i)) {
      qnative.flag_qnative = true;
      qnative.min_seqsep_explicit = true;
      qnative.min_seqsep = stoi(common_value);
      continue;
    }
    if (consume_bool_option(arg, {"-qnative-soft", "--qnative-soft"},
                            qnative.soft, i)) {
      qnative.flag_qnative = true;
      qnative.soft_explicit = true;
      continue;
    }
    if (consume_option_value(arg, {"-qnative-beta", "--qnative-beta"},
                             common_value, i)) {
      qnative.flag_qnative = true;
      qnative.beta_explicit = true;
      qnative.beta = stod(common_value);
      continue;
    }
    if (consume_bool_option(arg, {"-dihedral", "--dihedral"},
                            dihedral.flag_dihedral, i)) {
      continue;
    }
    if (consume_option_value(arg, {"-dihedral-kind", "--dihedral-kind"},
                             common_value, i)) {
      dihedral.flag_dihedral = true;
      dihedral.kind = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-dihedral-residues", "--dihedral-residues"},
                             common_value, i)) {
      dihedral.flag_dihedral = true;
      dihedral.residues = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-dihedral-stride", "--dihedral-stride"},
                             common_value, i)) {
      dihedral.flag_dihedral = true;
      dihedral.stride = stoi(common_value);
      continue;
    }
    if (consume_bool_option(arg,
                            {"-dihedral-export-tica", "--dihedral-export-tica"},
                            dihedral.export_tica, i)) {
      dihedral.flag_dihedral = true;
      continue;
    }
    if (arg.rfind("-s2-plateau-window", 0) == 0 ||
        arg.rfind("--s2-plateau-window", 0) == 0) {
      TitanUI::print_error(
          "S2: -s2-plateau-window is no longer supported; use -s2-fit-window.");
      exit(EXIT_FAILURE);
    }
    if (consume_bool_option(arg, {"-s2", "--s2"}, s2.flag_s2, i)) {
      continue;
    }
    if (consume_option_value(arg, {"-s2-vector", "--s2-vector"}, common_value,
                             i)) {
      s2.flag_s2 = true;
      s2.vector = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-s2-align", "--s2-align"}, common_value,
                             i)) {
      s2.flag_s2 = true;
      s2.align = lowercase_copy(common_value);
      if (s2.align == "provided") {
        TitanUI::print_error(
            "S2: -s2-align=provided is not supported in v1; use rms for "
            "first-frame alignment or none.");
        exit(EXIT_FAILURE);
      }
      continue;
    }
    if (consume_option_value(arg, {"-s2-maxlag", "--s2-maxlag"}, common_value,
                             i)) {
      s2.flag_s2 = true;
      s2.maxlag = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-s2-fit-window", "--s2-fit-window"},
                             common_value, i)) {
      s2.flag_s2 = true;
      s2.fit_window = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-s2-residues", "--s2-residues"},
                             common_value, i)) {
      s2.flag_s2 = true;
      s2.residues = common_value;
      continue;
    }
    if (consume_bool_option(arg, {"-s2-bfactor-pdb", "--s2-bfactor-pdb"},
                            s2.bfactor_pdb, i)) {
      s2.flag_s2 = s2.flag_s2 || s2.bfactor_pdb;
      continue;
    }
    if (consume_bool_option(arg, {"-msd", "--msd"}, msd.flag_msd, i)) {
      continue;
    }
    if (consume_option_value(arg, {"-msd-selection", "--msd-selection"},
                             common_value, i)) {
      msd.flag_msd = true;
      msd.selection = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-msd-mode", "--msd-mode"}, common_value,
                             i)) {
      msd.flag_msd = true;
      msd.mode = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-msd-maxlag", "--msd-maxlag"}, common_value,
                             i)) {
      msd.flag_msd = true;
      msd.maxlag = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-msd-fit-window", "--msd-fit-window"},
                             common_value, i)) {
      msd.flag_msd = true;
      msd.fit_window = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-msd-pbc", "--msd-pbc"}, common_value, i)) {
      msd.flag_msd = true;
      msd.pbc = common_value;
      continue;
    }
    if (consume_bool_option(arg, {"-msd-rot", "--msd-rot"}, msd.rot, i)) {
      msd.flag_msd = true;
      continue;
    }
    if (consume_option_value(arg,
                             {"-msd-rot-fit-window", "--msd-rot-fit-window"},
                             common_value, i)) {
      msd.flag_msd = true;
      msd.rot = true;
      msd.rot_fit_window = common_value;
      continue;
    }
    if (consume_bool_option(arg, {"-membrane", "--membrane"},
                            membrane.flag_membrane, i)) {
      continue;
    }
    if (consume_option_value(arg, {"-membrane-group", "--membrane-group"},
                             common_value, i)) {
      membrane.flag_membrane = true;
      membrane.group = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-membrane-lipids", "--membrane-lipids"},
                             common_value, i)) {
      membrane.flag_membrane = true;
      membrane.lipids = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-membrane-head", "--membrane-head"},
                             common_value, i)) {
      membrane.flag_membrane = true;
      membrane.head = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-membrane-tail", "--membrane-tail"},
                             common_value, i)) {
      membrane.flag_membrane = true;
      membrane.tail = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-membrane-normal", "--membrane-normal"},
                             common_value, i)) {
      membrane.flag_membrane = true;
      membrane.normal = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-membrane-grid", "--membrane-grid"},
                             common_value, i)) {
      membrane.flag_membrane = true;
      membrane.grid = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-membrane-leaflet", "--membrane-leaflet"},
                             common_value, i)) {
      membrane.flag_membrane = true;
      membrane.leaflet = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-membrane-leaflet-file", "--membrane-leaflet-file"},
            common_value, i)) {
      membrane.flag_membrane = true;
      membrane.leaflet_file = common_value;
      continue;
    }
    if (consume_option_value(arg,
                             {"-membrane-flipflop-persistence",
                              "--membrane-flipflop-persistence"},
                             common_value, i)) {
      membrane.flag_membrane = true;
      try {
        size_t parsed_chars = 0;
        membrane.flipflop_persistence = stoi(common_value, &parsed_chars);
        if (parsed_chars != common_value.size())
          throw std::invalid_argument("trailing characters");
      } catch (...) {
        TitanUI::print_error("Membrane: -membrane-flipflop-persistence must be "
                             "an integer >= 1.");
        exit(EXIT_FAILURE);
      }
      continue;
    }
    if (consume_bool_option(
            arg, {"-membrane-protein-contacts", "--membrane-protein-contacts"},
            membrane.protein_contacts, i)) {
      if (membrane.protein_contacts)
        membrane.flag_membrane = true;
      continue;
    }
    if (consume_option_value(
            arg, {"-membrane-contact-cut", "--membrane-contact-cut"},
            common_value, i)) {
      membrane.flag_membrane = true;
      try {
        size_t parsed_chars = 0;
        membrane.contact_cut = stod(common_value, &parsed_chars);
        if (parsed_chars != common_value.size())
          throw std::invalid_argument("trailing characters");
      } catch (...) {
        TitanUI::print_error(
            "Membrane: -membrane-contact-cut must be a positive number.");
        exit(EXIT_FAILURE);
      }
      continue;
    }
    if (consume_bool_option(arg,
                            {"-membrane-enrichment", "--membrane-enrichment"},
                            membrane.enrichment, i)) {
      if (membrane.enrichment) {
        membrane.flag_membrane = true;
        membrane.protein_contacts = true;
      }
      continue;
    }
    if (consume_option_value(arg,
                             {"-membrane-curvature", "--membrane-curvature"},
                             common_value, i)) {
      membrane.curvature = lowercase_copy(common_value);
      if (membrane.curvature == "grid")
        membrane.flag_membrane = true;
      continue;
    }
    if (consume_option_value(arg, {"-graph-source", "--graph-source"},
                             common_value, i)) {
      graph.flag_graph = true;
      graph.sources.push_back(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-graph-community", "--graph-community"},
                             common_value, i)) {
      graph.flag_graph = true;
      graph.community_method = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-graph-filter-topk", "--graph-filter-topk"},
                             common_value, i)) {
      graph.flag_graph = true;
      graph.filter_topk = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-graph-filter-component", "--graph-filter-component"},
            common_value, i)) {
      graph.flag_graph = true;
      graph.filter_component = common_value;
      continue;
    }

    if (consume_option_value(arg, {"-mem-budget-gb", "--mem-budget-gb"},
                             common_value, i)) {
      parse_memory_budget(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-sg-mode", "--sg-mode"}, common_value, i)) {
      sg_mode =
          require_choice(common_value, {"auto", "stream", "cache"}, "-sg-mode");
      continue;
    }
    if (consume_option_value(arg, {"-sg-cache", "--sg-cache"}, common_value,
                             i)) {
      sg_cache =
          require_choice(common_value, {"auto", "off", "on"}, "-sg-cache");
      continue;
    }
    if (consume_option_value(arg, {"-sg-memory-gb", "--sg-memory-gb"},
                             common_value, i)) {
      parse_sg_memory_budget(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-sg-timeseries", "--sg-timeseries"},
                             common_value, i)) {
      sg_timeseries = require_choice(
          common_value, {"auto", "full", "summary-only"}, "-sg-timeseries");
      continue;
    }
    if (consume_option_value(arg, {"-sg-kernel", "--sg-kernel"}, common_value,
                             i)) {
      sg_kernel = require_choice(common_value, {"auto", "scalar", "avx2"},
                                 "-sg-kernel");
      continue;
    }
    if (consume_option_value(arg, {"-sg-cutoff", "--sg-cutoff"}, common_value,
                             i)) {
      parse_sg_cutoff(common_value);
      continue;
    }
    if ((arg == "-low-memory" || arg == "--low-memory") && i + 1 < argc) {
      const std::string next = string(argv[i + 1]);
      const std::string next_lc = lowercase_copy(next);
      if (next_lc == "true" || next_lc == "false" || next_lc == "1" ||
          next_lc == "0" || next_lc == "yes" || next_lc == "no" ||
          next_lc == "on" || next_lc == "off") {
        low_memory = parse_bool_equals(next, "-low-memory");
        ++i;
        continue;
      }
    }
    if (consume_bool_option(arg, {"-low-memory", "--low-memory"}, low_memory,
                            i)) {
      continue;
    }
    if (consume_option_value(arg, {"-network-mode", "--network-mode"},
                             common_value, i)) {
      network_mode = require_choice(common_value, {"auto", "dense", "sparse"},
                                    "-network-mode");
      continue;
    }
    if (consume_option_value(arg,
                             {"-network-sparse-cut", "--network-sparse-cut"},
                             common_value, i)) {
      network_sparse_cut = stod(common_value);
      if (network_sparse_cut <= 0.0) {
        TitanUI::print_error("-network-sparse-cut must be positive.");
        exit(EXIT_FAILURE);
      }
      continue;
    }
    if (consume_option_value(arg, {"-network-sparse-k", "--network-sparse-k"},
                             common_value, i)) {
      network_sparse_k = stoi(common_value);
      if (network_sparse_k < 0) {
        TitanUI::print_error("-network-sparse-k must be >= 0.");
        exit(EXIT_FAILURE);
      }
      continue;
    }
    if (consume_option_value(
            arg, {"-network-cache-precision", "--network-cache-precision"},
            common_value, i)) {
      network_cache_precision = require_choice(
          common_value, {"auto", "fp32", "fp64"}, "-network-cache-precision");
      continue;
    }
    if (consume_option_value(arg, {"-tda-max-points", "--tda-max-points"},
                             common_value, i)) {
      tda_max_points = static_cast<size_t>(stoull(common_value));
      continue;
    }
    if (arg.rfind("-pbe=", 0) == 0 || arg.rfind("--pbe=", 0) == 0) {
      std::string backend = lowercase_copy(arg.substr(arg.find('=') + 1));
      if (backend == "delphi" || backend == "pydelphi" || backend == "fd")
        backend = "grid";
      if (backend == "bie")
        backend = "tabi";
      if (backend != "grid" && backend != "tabi") {
        TitanUI::print_error(
            "PBE: -pbe backend must be grid or bie (legacy alias: tabi; "
            "grid aliases: delphi, pydelphi, fd).");
        exit(EXIT_FAILURE);
      }
      pbe.backend = backend;
      pbe.flag_pbe = true;
      continue;
    }
    if (arg == "-pbe" || arg == "--pbe") {
      if (i + 1 < argc) {
        std::string backend = lowercase_copy(string(argv[i + 1]));
        if (backend == "delphi" || backend == "pydelphi" || backend == "fd")
          backend = "grid";
        if (backend == "bie")
          backend = "tabi";
        if (backend == "grid" || backend == "tabi") {
          pbe.backend = backend;
          pbe.flag_pbe = true;
          ++i;
          continue;
        }
        if (!backend.empty() && backend.front() != '-') {
          TitanUI::print_error(
              "PBE: -pbe backend must be grid or bie (legacy alias: tabi; "
              "grid aliases: delphi, pydelphi, fd).");
          exit(EXIT_FAILURE);
        }
      }
      pbe.flag_pbe = true;
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-fast-operator", "--pbe-bie-fast-operator"},
            common_value, i)) {
      const std::string mode = require_choice(
          common_value, {"auto", "direct", "p1"},
          "-pbe-bie-fast-operator");
      pbe.backend = "tabi";
      pbe.flag_pbe = true;
      pbe.tabi_fmm_local = "off";
      if (mode == "auto") {
        pbe.tabi_sum = "auto";
        pbe.tabi_fmm_m2l = "auto";
        pbe.tabi_fmm_m2l_explicit = false;
      } else if (mode == "direct") {
        pbe.tabi_sum = "direct";
        pbe.tabi_formulation = "jcp2013";
        pbe.tabi_formulation_explicit = true;
        pbe.tabi_fmm_m2l = "off";
        pbe.tabi_fmm_m2l_explicit = true;
      } else {
        pbe.tabi_sum = "treecode";
        pbe.tabi_formulation = "jcp2013-treecode";
        pbe.tabi_formulation_explicit = true;
        pbe.tabi_fmm_m2l = "single-matvec-analytic";
        pbe.tabi_fmm_m2l_explicit = true;
        pbe.tabi_taylor_order = 4;
      }
      continue;
    }
    if (arg == "-pbsa" || arg == "--pbsa") {
      if (i + 1 < argc) {
        std::string next_value = lowercase_copy(string(argv[i + 1]));
        if (next_value == "1" || next_value == "true" || next_value == "yes" ||
            next_value == "on" || next_value == "0" || next_value == "false" ||
            next_value == "no" || next_value == "off") {
          pbe.pbsa = parse_bool_equals(next_value, "-pbsa");
          if (pbe.pbsa) {
            pbe.flag_pbe = true;
            pbe.dgsolv = true;
          }
          ++i;
          continue;
        }
      }
      pbe.pbsa = true;
      pbe.flag_pbe = true;
      pbe.dgsolv = true;
      continue;
    }
    if (arg.rfind("-pbsa=", 0) == 0 || arg.rfind("--pbsa=", 0) == 0) {
      pbe.pbsa = parse_bool_equals(arg.substr(arg.find('=') + 1), "-pbsa");
      if (pbe.pbsa) {
        pbe.flag_pbe = true;
        pbe.dgsolv = true;
      }
      continue;
    }
    if (consume_compound_option_value(
            arg, {"-pbsa-components", "--pbsa-components"}, common_value, i)) {
      std::string error;
      if (!titan_cli::normalize_pbsa_component_selection(
              common_value, pbe.pbsa_components, error)) {
        TitanUI::print_error("PBSA: invalid -pbsa-components value: " + error +
                             ". Use complex,ligand,receptor or "
                             "ligand-receptor-complex.");
        return titan_cli::kInvalidArguments;
      }
      continue;
    }
    if (consume_option_value(arg, {"-pbsa-decomp", "--pbsa-decomp"},
                             common_value, i)) {
      pbe.pbsa_decomp = require_choice(
          common_value, {"off", "molecule", "residue", "hybrid", "auto"},
          "-pbsa-decomp");
      continue;
    }
    if (consume_bool_option(arg, {"-pbsa-ie", "--pbsa-ie"}, pbe.pbsa_ie, i)) {
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbsa-residue-frames", "--pbsa-residue-frames"},
                             common_value, i)) {
      pbe.pbsa_residue_frames =
          require_choice(common_value, {"off", "all"}, "-pbsa-residue-frames");
      continue;
    }
    if (consume_option_value(arg, {"-pbsa-ie-block", "--pbsa-ie-block"},
                             common_value, i)) {
      pbe.pbsa_ie_block = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbsa-ie-min-samples", "--pbsa-ie-min-samples"},
                             common_value, i)) {
      pbe.pbsa_ie_min_samples = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbsa-ie-filter-sigma", "--pbsa-ie-filter-sigma"},
            common_value, i)) {
      pbe.pbsa_ie_filter_sigma = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbsa-perf", "--pbsa-perf"}, common_value,
                             i)) {
      pbe.pbsa_perf = require_choice(common_value, {"off", "summary", "detail"},
                                     "-pbsa-perf");
      continue;
    }
    if (consume_option_value(arg, {"-pbsa-grid-reuse", "--pbsa-grid-reuse"},
                             common_value, i)) {
      pbe.pbsa_grid_reuse = require_choice(
          common_value, {"off", "validate", "auto"}, "-pbsa-grid-reuse");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbsa-sasa-points", "--pbsa-sasa-points"},
            common_value, i)) {
      pbe.pbsa_sasa_points = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-timing", "--pbe-timing"}, common_value,
                             i)) {
      pbe.timing = require_choice(common_value, {"off", "summary", "detail"},
                                  "-pbe-timing");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-tree-preset", "--pbe-bie-tree-preset"},
            common_value, i)) {
      pbe.tabi_tree_preset = require_choice(
          common_value,
          {"off", "stage3u-safe", "stage3u-perf", "stage3u-perf2"},
          "-pbe-bie-tree-preset");
      apply_tabi_tree_preset(pbe.tabi_tree_preset);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-bie-sum", "--pbe-bie-sum"},
                             common_value, i)) {
      pbe.tabi_sum = require_choice(
          common_value, {"auto", "direct", "treecode"}, "-pbe-bie-sum");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-bie-geometry", "--pbe-bie-geometry"},
                             common_value, i)) {
      pbe.tabi_geometry = require_choice(
          common_value, {"triangle", "point-panel"}, "-pbe-bie-geometry");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-bie-diagonal", "--pbe-bie-diagonal"},
                             common_value, i)) {
      pbe.tabi_diagonal = require_choice(
          common_value, {"solid-angle", "flat", "off"}, "-pbe-bie-diagonal");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-taylor-order", "--pbe-bie-taylor-order"},
            common_value, i)) {
      pbe.tabi_taylor_order = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-derivatives", "--pbe-bie-derivatives"},
            common_value, i)) {
      pbe.tabi_derivatives =
          require_choice(common_value, {"analytic", "finite-diff-debug"},
                         "-pbe-bie-derivatives");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-surface-points", "--pbe-bie-surface-points"},
            common_value, i)) {
      pbe.tabi_surface_points = stoi(common_value);
      continue;
    }
    if (consume_bool_option(
            arg, {"-pbe-bie-surface-only", "--pbe-bie-surface-only"},
            pbe.tabi_surface_only, i)) {
      continue;
    }
    if (consume_bool_option(
            arg, {"-pbe-bie-highres-surface", "--pbe-bie-highres-surface"},
            pbe.tabi_highres_surface, i)) {
      continue;
    }
    if (consume_bool_option(arg,
                            {"-pbe-bie-calderon-trace-audit",
                             "--pbe-bie-calderon-trace-audit"},
                            pbe.tabi_calderon_trace_audit, i)) {
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-calderon-aux-surface",
                              "--pbe-bie-calderon-aux-surface"},
                             common_value, i)) {
      pbe.tabi_calderon_aux_surface =
          require_choice(common_value, {"off", "implicit-vdw"},
                         "-pbe-bie-calderon-aux-surface");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-calderon-aux-grid-spacing",
                              "--pbe-bie-calderon-aux-grid-spacing"},
                             common_value, i)) {
      pbe.tabi_calderon_aux_grid_spacing = stod(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-primary-surface", "--pbe-bie-primary-surface"},
            common_value, i)) {
      pbe.tabi_primary_surface = require_choice(
          common_value, {"sphere-patch", "grid-ses", "analytic-ses"},
          "-pbe-bie-primary-surface");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-analytic-ses-engine",
                              "--pbe-bie-analytic-ses-engine"},
                             common_value, i)) {
      pbe.tabi_analytic_ses_engine = require_choice(
          common_value,
          {"current", "current-dense", "patch-ray-experimental"},
          "-pbe-bie-analytic-ses-engine");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-ses-grid-spacing", "--pbe-bie-ses-grid-spacing"},
            common_value, i)) {
      pbe.tabi_ses_grid_spacing = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-ses-max-grid-vertices",
                              "--pbe-bie-ses-max-grid-vertices"},
                             common_value, i)) {
      pbe.tabi_ses_max_grid_vertices = stoull(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-sphere-mesh", "--pbe-bie-sphere-mesh"},
            common_value, i)) {
      pbe.tabi_sphere_mesh =
          require_choice(common_value, {"auto", "latlong", "icosphere"},
                         "-pbe-bie-sphere-mesh");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-curved-pair-cache-mb",
                  "--pbe-bie-curved-pair-cache-mb"},
            common_value, i)) {
      pbe.tabi_curved_pair_cache_mb = parse_integer_option(
          common_value, "-pbe-bie-curved-pair-cache-mb");
      if (pbe.tabi_curved_pair_cache_mb < 0 ||
          pbe.tabi_curved_pair_cache_mb > 262144) {
        TitanUI::print_error("-pbe-bie-curved-pair-cache-mb must be between "
                             "0 and 262144 MiB.");
        exit(EXIT_FAILURE);
      }
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-tree-leaf-size", "--pbe-bie-tree-leaf-size"},
            common_value, i)) {
      pbe.tabi_tree_leaf_size = stoi(common_value);
      pbe.tabi_tree_leaf_size_explicit = true;
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-opening-theta", "--pbe-bie-opening-theta"},
            common_value, i)) {
      pbe.tabi_tree_opening_theta = stod(common_value);
      pbe.tabi_tree_opening_theta_explicit = true;
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-formulation", "--pbe-bie-formulation"},
            common_value, i)) {
      pbe.tabi_formulation = require_choice(
          common_value,
          {"titan-v1", "jcp2013", "jcp2013-centroid", "jcp2013-treecode",
           "tabi2-nodepatch", "calibrated-debug"},
          "-pbe-bie-formulation");
      pbe.tabi_formulation_explicit = true;
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-discretization", "--pbe-bie-discretization"},
            common_value, i)) {
      pbe.tabi_discretization =
          require_choice(common_value, {"centroid", "node-patch"},
                         "-pbe-bie-discretization");
      pbe.tabi_discretization_explicit = true;
      continue;
    }
    if (consume_option_value(arg, {"-pbe-bie-precond", "--pbe-bie-precond"},
                             common_value, i)) {
      pbe.tabi_precond_requested = require_choice(
          common_value,
          {"auto", "none", "diagonal", "block", "block-budgeted",
           "coarse-galerkin-probe", "fmm-block-jacobi",
           "fmm-overlap-schwarz", "fmm-nearfield-ilu0",
           "fmm-nearfield-ilu0-coarse", "distance-knn-overlap-schwarz",
           "distance-knn-ilu0"},
          "-pbe-bie-precond");
      pbe.tabi_precond = pbe.tabi_precond_requested;
      pbe.tabi_precond_explicit = true;
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-precond-block-near", "--pbe-bie-precond-block-near"},
            common_value, i)) {
      pbe.tabi_precond_block_near = require_choice(
          common_value, {"singular", "buffer-aware", "operator-near"},
          "-pbe-bie-precond-block-near");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-precond-block-panel-cap",
                              "--pbe-bie-precond-block-panel-cap"},
                             common_value, i)) {
      pbe.tabi_precond_block_panel_cap = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-precond-overlap-panel-cap",
                              "--pbe-bie-precond-overlap-panel-cap"},
                             common_value, i)) {
      pbe.tabi_precond_overlap_panel_cap = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-precond-ilu-neighbor-cap",
                              "--pbe-bie-precond-ilu-neighbor-cap"},
                             common_value, i)) {
      pbe.tabi_precond_ilu_neighbor_cap = stoi(common_value);
      pbe.tabi_precond_ilu_neighbor_cap_explicit = true;
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-precond-ilu-drop-tol",
                              "--pbe-bie-precond-ilu-drop-tol"},
                             common_value, i)) {
      pbe.tabi_precond_ilu_drop_tol = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-precond-ilu-regularization",
                              "--pbe-bie-precond-ilu-regularization"},
                             common_value, i)) {
      pbe.tabi_precond_ilu_regularization =
          require_choice(common_value, {"off", "adaptive"},
                         "-pbe-bie-precond-ilu-regularization");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-precond-ilu-regularization-max",
                              "--pbe-bie-precond-ilu-regularization-max"},
                             common_value, i)) {
      pbe.tabi_precond_ilu_regularization_max = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-precond-memory-budget-mb",
                              "--pbe-bie-precond-memory-budget-mb"},
                             common_value, i)) {
      pbe.tabi_precond_memory_budget_mb = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-precond-coarse-aggregate-cap",
                              "--pbe-bie-precond-coarse-aggregate-cap"},
                             common_value, i)) {
      pbe.tabi_precond_coarse_aggregate_cap = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-precond-coarse-base", "--pbe-bie-precond-coarse-base"},
            common_value, i)) {
      pbe.tabi_precond_coarse_base =
          require_choice(common_value, {"diagonal", "block-budgeted"},
                         "-pbe-bie-precond-coarse-base");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-precond-coarse-verify",
                              "--pbe-bie-precond-coarse-verify"},
                             common_value, i)) {
      pbe.tabi_precond_coarse_verify =
          require_choice(common_value, {"off", "full-small"},
                         "-pbe-bie-precond-coarse-verify");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-near-integration", "--pbe-bie-near-integration"},
            common_value, i)) {
      pbe.tabi_near_integration = require_choice(
          common_value,
          {"auto", "centroid", "analytic_singular", "duffy_adaptive"},
          "-pbe-bie-near-integration");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-quad-buffer", "--pbe-bie-quad-buffer"},
            common_value, i)) {
      pbe.tabi_quad_buffer =
          require_choice(common_value, {"auto", "off", "gauss7", "adaptive"},
                         "-pbe-bie-quad-buffer");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-leaf-near-policy", "--pbe-bie-leaf-near-policy"},
            common_value, i)) {
      pbe.tabi_leaf_near_policy = require_choice(
          common_value,
          {"auto", "exact", "smooth-node", "large-protein", "adaptive-large"},
          "-pbe-bie-leaf-near-policy");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-leaf-panel-kernel", "--pbe-bie-leaf-panel-kernel"},
            common_value, i)) {
      pbe.tabi_leaf_panel_kernel = require_choice(
          common_value,
          {"auto", "scalar", "avx2-gather", "tile", "tile-scalar", "tile-avx2"},
          "-pbe-bie-leaf-panel-kernel");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-adaptive-level", "--pbe-bie-adaptive-level"},
            common_value, i)) {
      pbe.tabi_adaptive_level =
          require_choice(common_value, {"strict", "balanced", "aggressive"},
                         "-pbe-bie-adaptive-level");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-cluster-policy", "--pbe-bie-cluster-policy"},
            common_value, i)) {
      pbe.tabi_cluster_policy = require_choice(
          common_value, {"off", "leaf-pair-strict", "leaf-pair-balanced"},
          "-pbe-bie-cluster-policy");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-leaf-pair-kernel", "--pbe-bie-leaf-pair-kernel"},
            common_value, i)) {
      pbe.tabi_leaf_pair_kernel =
          require_choice(common_value,
                         {"auto", "scalar", "target-avx2", "target-avx2-v2",
                          "target-avx2-4x4"},
                         "-pbe-bie-leaf-pair-kernel");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-leaf-kernel-profile", "--pbe-bie-leaf-kernel-profile"},
            common_value, i)) {
      pbe.tabi_leaf_kernel_profile = require_choice(
          common_value, {"off", "summary"}, "-pbe-bie-leaf-kernel-profile");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-leaf-work-profile", "--pbe-bie-leaf-work-profile"},
            common_value, i)) {
      pbe.tabi_leaf_work_profile = require_choice(
          common_value, {"off", "summary"}, "-pbe-bie-leaf-work-profile");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-matvec-cost-profile", "--pbe-bie-matvec-cost-profile"},
            common_value, i)) {
      pbe.tabi_matvec_cost_profile = require_choice(
          common_value, {"off", "summary"}, "-pbe-bie-matvec-cost-profile");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-jcp-direct-kernel", "--pbe-bie-jcp-direct-kernel"},
            common_value, i)) {
      pbe.tabi_jcp_direct_kernel = require_choice(
          common_value, {"baseline", "soa-avx2", "fused-soa-avx2"},
          "-pbe-bie-jcp-direct-kernel");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-leaf-pair-selection-profile",
                              "--pbe-bie-leaf-pair-selection-profile"},
                             common_value, i)) {
      pbe.tabi_leaf_pair_selection_profile =
          require_choice(common_value, {"off", "summary"},
                         "-pbe-bie-leaf-pair-selection-profile");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-leaf-pair-probe-cache",
                              "--pbe-bie-leaf-pair-probe-cache"},
                             common_value, i)) {
      pbe.tabi_leaf_pair_probe_cache =
          require_choice(common_value, {"off", "per-matvec"},
                         "-pbe-bie-leaf-pair-probe-cache");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-leaf-pair-target-group",
                              "--pbe-bie-leaf-pair-target-group"},
                             common_value, i)) {
      pbe.tabi_leaf_pair_target_group =
          require_choice(common_value, {"scan", "same-target-leaf"},
                         "-pbe-bie-leaf-pair-target-group");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-leaf-pair-directness-probe",
                              "--pbe-bie-leaf-pair-directness-probe"},
                             common_value, i)) {
      pbe.tabi_leaf_pair_directness_probe =
          require_choice(common_value, {"full-traversal", "ancestor-path"},
                         "-pbe-bie-leaf-pair-directness-probe");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-leaf-pair-directness-cache",
                              "--pbe-bie-leaf-pair-directness-cache"},
                             common_value, i)) {
      pbe.tabi_leaf_pair_directness_cache =
          require_choice(common_value, {"off", "target-leaf-source-leaf"},
                         "-pbe-bie-leaf-pair-directness-cache");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-leaf-pair-directness-verify",
                              "--pbe-bie-leaf-pair-directness-verify"},
                             common_value, i)) {
      pbe.tabi_leaf_pair_directness_verify =
          require_choice(common_value, {"off", "sampled", "full-small"},
                         "-pbe-bie-leaf-pair-directness-verify");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-leaf-pair-scheduler", "--pbe-bie-leaf-pair-scheduler"},
            common_value, i)) {
      pbe.tabi_leaf_pair_scheduler =
          require_choice(common_value,
                         {"inline-select", "deferred-worklist",
                          "target-leaf-block", "cost-tile-worklist"},
                         "-pbe-bie-leaf-pair-scheduler");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-leaf-pair-worklist-max-active",
                              "--pbe-bie-leaf-pair-worklist-max-active"},
                             common_value, i)) {
      pbe.tabi_leaf_pair_worklist_max_active = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-leaf-pair-worklist-verify",
                              "--pbe-bie-leaf-pair-worklist-verify"},
                             common_value, i)) {
      pbe.tabi_leaf_pair_worklist_verify =
          require_choice(common_value, {"off", "sampled", "full-small"},
                         "-pbe-bie-leaf-pair-worklist-verify");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-leaf-block-verify", "--pbe-bie-leaf-block-verify"},
            common_value, i)) {
      pbe.tabi_leaf_block_verify =
          require_choice(common_value, {"off", "sampled", "full-small"},
                         "-pbe-bie-leaf-block-verify");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-bie-leaf-mac", "--pbe-bie-leaf-mac"},
                             common_value, i)) {
      pbe.tabi_leaf_mac = require_choice(common_value, {"off", "audit", "on"},
                                         "-pbe-bie-leaf-mac");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-bie-fmm-m2l", "--pbe-bie-fmm-m2l"},
                             common_value, i)) {
      pbe.tabi_fmm_m2l = require_choice(
          common_value, {"auto", "off", "single-matvec-analytic"},
          "-pbe-bie-fmm-m2l");
      pbe.tabi_fmm_m2l_explicit = true;
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-fmm-engine", "--pbe-bie-fmm-engine"},
                             common_value, i)) {
      pbe.tabi_fmm_engine = require_choice(
          common_value,
          {"cartesian", "cartesian-full-experimental", "kifmm-lite",
           "kifmm-local", "kifmm-lite-experimental",
           "kifmm-channel-hybrid-experimental"},
          "-pbe-bie-fmm-engine");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-fmm-cartesian-order",
             "--pbe-bie-fmm-cartesian-order"},
            common_value, i)) {
      pbe.bie_fmm_cartesian_order = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-kifmm-order", "--pbe-bie-kifmm-order"},
            common_value, i)) {
      pbe.tabi_kifmm_order = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-kifmm-equiv-scale", "--pbe-bie-kifmm-equiv-scale"},
            common_value, i)) {
      pbe.tabi_kifmm_equiv_scale = stod(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-kifmm-check-scale", "--pbe-bie-kifmm-check-scale"},
            common_value, i)) {
      pbe.tabi_kifmm_check_scale = stod(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-kifmm-solver", "--pbe-bie-kifmm-solver"},
            common_value, i)) {
      pbe.tabi_kifmm_solver =
          require_choice(common_value, {"qr", "svd", "normal-tikhonov"},
                         "-pbe-bie-kifmm-solver");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-kifmm-profile", "--pbe-bie-kifmm-profile"},
            common_value, i)) {
      pbe.tabi_kifmm_profile = require_choice(
          common_value, {"off", "summary", "full"}, "-pbe-bie-kifmm-profile");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-kifmm-full-apply", "--pbe-bie-kifmm-full-apply"},
            common_value, i)) {
      pbe.tabi_kifmm_full_apply =
          require_choice(common_value, {"serial", "target-node-parallel"},
                         "-pbe-bie-kifmm-full-apply");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-kifmm-ilu-graph", "--pbe-bie-kifmm-ilu-graph"},
            common_value, i)) {
      pbe.tabi_kifmm_ilu_graph = require_choice(
          common_value, {"cartesian-near", "kifmm-fallback-capped"},
          "-pbe-bie-kifmm-ilu-graph");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-kifmm-hierarchy-order",
                              "--pbe-bie-kifmm-hierarchy-order"},
                             common_value, i)) {
      pbe.tabi_kifmm_hierarchy_order = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-kifmm-hierarchy-theta",
                              "--pbe-bie-kifmm-hierarchy-theta"},
                             common_value, i)) {
      pbe.tabi_kifmm_hierarchy_theta = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-kifmm-hierarchy-real-panel-guard",
                              "--pbe-bie-kifmm-hierarchy-real-panel-guard"},
                             common_value, i)) {
      pbe.tabi_kifmm_hierarchy_real_panel_guard =
          require_choice(common_value, {"strict-local-check"},
                         "-pbe-bie-kifmm-hierarchy-real-panel-guard");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-kifmm-hierarchy-guard-margin-factor",
                              "--pbe-bie-kifmm-hierarchy-guard-margin-factor"},
                             common_value, i)) {
      pbe.tabi_kifmm_hierarchy_guard_margin_factor = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-kifmm-hierarchy-tree-stop-policy",
                              "--pbe-bie-kifmm-hierarchy-tree-stop-policy"},
                             common_value, i)) {
      pbe.tabi_kifmm_hierarchy_tree_stop_policy =
          require_choice(common_value, {"panel-radius-occupancy"},
                         "-pbe-bie-kifmm-hierarchy-tree-stop-policy");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-kifmm-hierarchy-tree-panel-radius-factor",
             "--pbe-bie-kifmm-hierarchy-tree-panel-radius-factor"},
            common_value, i)) {
      pbe.tabi_kifmm_hierarchy_tree_panel_radius_factor = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-kifmm-hierarchy-tree-occupancy-stop",
                              "--pbe-bie-kifmm-hierarchy-tree-occupancy-stop"},
                             common_value, i)) {
      pbe.tabi_kifmm_hierarchy_tree_occupancy_stop = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-kifmm-hierarchy-m2l-eval-policy",
                              "--pbe-bie-kifmm-hierarchy-m2l-eval-policy"},
                             common_value, i)) {
      pbe.tabi_kifmm_hierarchy_m2l_eval_policy =
          require_choice(common_value, {"target-check-batched"},
                         "-pbe-bie-kifmm-hierarchy-m2l-eval-policy");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-kifmm-hierarchy-matrix-cache-policy",
                              "--pbe-bie-kifmm-hierarchy-matrix-cache-policy"},
                             common_value, i)) {
      pbe.tabi_kifmm_hierarchy_matrix_cache_policy = require_choice(
          common_value,
          {"canonical-offset-cache", "static-translation-table-cache",
           "target-node-streaming-cache", "bounded-lru-cache"},
          "-pbe-bie-kifmm-hierarchy-matrix-cache-policy");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-kifmm-hierarchy-matrix-cache-limit-mb",
             "--pbe-bie-kifmm-hierarchy-matrix-cache-limit-mb"},
            common_value, i)) {
      pbe.tabi_kifmm_hierarchy_matrix_cache_limit_mb = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-kifmm-hierarchy-gemm-batch-policy",
                              "--pbe-bie-kifmm-hierarchy-gemm-batch-policy"},
                             common_value, i)) {
      pbe.tabi_kifmm_hierarchy_gemm_batch_policy =
          require_choice(common_value, {"per-pair"},
                         "-pbe-bie-kifmm-hierarchy-gemm-batch-policy");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-kifmm-hierarchy-m2l-parallel-policy",
                              "--pbe-bie-kifmm-hierarchy-m2l-parallel-policy"},
                             common_value, i)) {
      pbe.tabi_kifmm_hierarchy_m2l_parallel_policy =
          require_choice(common_value, {"serial", "target-node-key-group"},
                         "-pbe-bie-kifmm-hierarchy-m2l-parallel-policy");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-kifmm-hierarchy-m2l-parallel-workers",
             "--pbe-bie-kifmm-hierarchy-m2l-parallel-workers"},
            common_value, i)) {
      pbe.tabi_kifmm_hierarchy_m2l_parallel_workers = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-cartesian-k4-order", "--pbe-bie-cartesian-k4-order"},
            common_value, i)) {
      pbe.tabi_cartesian_k4_order = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-cartesian-k4-theta", "--pbe-bie-cartesian-k4-theta"},
            common_value, i)) {
      pbe.tabi_cartesian_k4_theta = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-cartesian-k4-geometry-theta",
                              "--pbe-bie-cartesian-k4-geometry-theta"},
                             common_value, i)) {
      pbe.tabi_cartesian_k4_geometry_theta = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-cartesian-k4-tail-tol",
                              "--pbe-bie-cartesian-k4-tail-tol"},
                             common_value, i)) {
      pbe.tabi_cartesian_k4_tail_tol = stod(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-fmm-cartesian-audit", "--pbe-bie-fmm-cartesian-audit"},
            common_value, i)) {
      pbe.tabi_fmm_cartesian_audit = require_choice(
          common_value, {"off", "on"}, "-pbe-bie-fmm-cartesian-audit");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-fmm-cartesian-error-audit",
             "--pbe-bie-fmm-cartesian-error-audit"},
            common_value, i)) {
      pbe.tabi_fmm_cartesian_error_audit = require_choice(
          common_value,
          {"off", "channel-shadow",
           "channel-shadow-retry-global-theta015"},
          "-pbe-bie-fmm-cartesian-error-audit");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-fmm-local", "--pbe-bie-fmm-local"},
                             common_value, i)) {
      pbe.tabi_fmm_local = require_choice(
          common_value, {"off", "audit-p4", "single-matvec-p4", "gmres-p4"},
          "-pbe-bie-fmm-local");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-fmm-local-refine", "--pbe-bie-fmm-local-refine"},
            common_value, i)) {
      pbe.tabi_fmm_local_refine = require_choice(
          common_value,
          {"off", "p3-p4-tail", "operator-tail-static", "operator-tail-linf"},
          "-pbe-bie-fmm-local-refine");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-fmm-local-refine-tol",
                              "--pbe-bie-fmm-local-refine-tol"},
                             common_value, i)) {
      pbe.tabi_fmm_local_refine_tol = stod(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-fmm-local-cache", "--pbe-bie-fmm-local-cache"},
            common_value, i)) {
      pbe.tabi_fmm_local_cache = require_choice(
          common_value,
          {"off", "kernel-jet", "green-jet-compressed", "green-jet-tiled"},
          "-pbe-bie-fmm-local-cache");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-fmm-local-cache-budget-mb",
                              "--pbe-bie-fmm-local-cache-budget-mb"},
                             common_value, i)) {
      pbe.tabi_fmm_local_cache_budget_mb = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-fmm-local-fallback", "--pbe-bie-fmm-local-fallback"},
            common_value, i)) {
      pbe.tabi_fmm_local_fallback =
          require_choice(common_value, {"scalar", "packed-avx2"},
                         "-pbe-bie-fmm-local-fallback");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-fmm-local-plan-builder",
                              "--pbe-bie-fmm-local-plan-builder"},
                             common_value, i)) {
      pbe.tabi_fmm_local_plan_builder = require_choice(
          common_value,
          {"legacy-rescan", "metric-cache", "rescan", "cached-metrics"},
          "-pbe-bie-fmm-local-plan-builder");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-fmm-local-refine-probe-count",
                              "--pbe-bie-fmm-local-refine-probe-count"},
                             common_value, i)) {
      pbe.tabi_fmm_local_refine_probe_count = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-fmm-local-refine-probe-mode",
                              "--pbe-bie-fmm-local-refine-probe-mode"},
                             common_value, i)) {
      pbe.tabi_fmm_local_refine_probe_mode =
          require_choice(common_value, {"rademacher", "solution", "mixed"},
                         "-pbe-bie-fmm-local-refine-probe-mode");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-fmm-local-geometry-guard",
                              "--pbe-bie-fmm-local-geometry-guard"},
                             common_value, i)) {
      pbe.tabi_fmm_local_geometry_guard =
          require_choice(common_value, {"off", "strict-geometry-mac"},
                         "-pbe-bie-fmm-local-geometry-guard");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-fmm-local-geometry-theta",
                              "--pbe-bie-fmm-local-geometry-theta"},
                             common_value, i)) {
      pbe.tabi_fmm_local_geometry_theta = stod(common_value);
      continue;
    }
    if (consume_bool_option(arg,
                            {"-pbe-bie-fmm-local-ilu-cap-sweep",
                             "--pbe-bie-fmm-local-ilu-cap-sweep"},
                            pbe.tabi_fmm_local_ilu_cap_sweep, i)) {
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-interaction-mode", "--pbe-bie-interaction-mode"},
            common_value, i)) {
      pbe.tabi_interaction_mode =
          require_choice(common_value, {"auto", "cached", "onthefly"},
                         "-pbe-bie-interaction-mode");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-node-bounds", "--pbe-bie-node-bounds"},
            common_value, i)) {
      pbe.tabi_node_bounds = require_choice(
          common_value, {"moment", "tight-aabb"}, "-pbe-bie-node-bounds");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-tree-builder", "--pbe-bie-tree-builder"},
            common_value, i)) {
      pbe.tabi_tree_builder =
          require_choice(common_value, {"grouped-morton", "topdown-aabb"},
                         "-pbe-bie-tree-builder");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-expansion-center", "--pbe-bie-expansion-center"},
            common_value, i)) {
      pbe.tabi_expansion_center = require_choice(
          common_value, {"legacy", "area-centroid", "aabb-center"},
          "-pbe-bie-expansion-center");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-fragmentation-safety",
                              "--pbe-bie-fragmentation-safety"},
                             common_value, i)) {
      pbe.tabi_fragmentation_safety =
          require_choice(common_value, {"off", "audit", "split"},
                         "-pbe-bie-fragmentation-safety");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-mac-safety", "--pbe-bie-mac-safety"},
                             common_value, i)) {
      pbe.tabi_mac_safety =
          require_choice(common_value, {"off", "audit", "physical-gap"},
                         "-pbe-bie-mac-safety");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-mac-gap-factor", "--pbe-bie-mac-gap-factor"},
            common_value, i)) {
      pbe.tabi_mac_gap_factor = stod(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-moment-audit", "--pbe-bie-moment-audit"},
            common_value, i)) {
      pbe.tabi_moment_audit =
          require_choice(common_value, {"off", "sampled", "full-small"},
                         "-pbe-bie-moment-audit");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-moment-construction", "--pbe-bie-moment-construction"},
            common_value, i)) {
      pbe.tabi_moment_construction =
          require_choice(common_value, {"translated", "direct-node"},
                         "-pbe-bie-moment-construction");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-moment-sum", "--pbe-bie-moment-sum"},
                             common_value, i)) {
      pbe.tabi_moment_sum = require_choice(common_value, {"naive", "kahan"},
                                           "-pbe-bie-moment-sum");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-convergence-audit", "--pbe-bie-convergence-audit"},
            common_value, i)) {
      pbe.tabi_convergence_audit =
          require_choice(common_value, {"off", "sampled", "full-small"},
                         "-pbe-bie-convergence-audit");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-convergence-safety", "--pbe-bie-convergence-safety"},
            common_value, i)) {
      pbe.tabi_convergence_safety =
          require_choice(common_value, {"off", "audit", "split"},
                         "-pbe-bie-convergence-safety");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-convergence-rho-cap", "--pbe-bie-convergence-rho-cap"},
            common_value, i)) {
      pbe.tabi_convergence_rho_cap = stod(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-mac-aspect-cap", "--pbe-bie-mac-aspect-cap"},
            common_value, i)) {
      pbe.tabi_mac_aspect_cap = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-estimator",
                              "--pbe-bie-convergence-estimator"},
                             common_value, i)) {
      pbe.tabi_convergence_estimator =
          require_choice(common_value, {"off", "moment-tail", "hybrid"},
                         "-pbe-bie-convergence-estimator");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-tail-threshold",
                              "--pbe-bie-convergence-tail-threshold"},
                             common_value, i)) {
      pbe.tabi_convergence_tail_threshold = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-estimator-error-cap",
                              "--pbe-bie-convergence-estimator-error-cap"},
                             common_value, i)) {
      pbe.tabi_convergence_estimator_error_cap = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-unstable-rho-cap",
                              "--pbe-bie-convergence-unstable-rho-cap"},
                             common_value, i)) {
      pbe.tabi_convergence_unstable_rho_cap = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-calibration",
                              "--pbe-bie-convergence-calibration"},
                             common_value, i)) {
      pbe.tabi_convergence_calibration =
          require_choice(common_value,
                         {"stage3j", "calibrated-tail-v1", "consecutive",
                          "aspect", "consecutive-aspect"},
                         "-pbe-bie-convergence-calibration");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-dielectric-amp",
                              "--pbe-bie-convergence-dielectric-amp"},
                             common_value, i)) {
      pbe.tabi_convergence_dielectric_amp = require_choice(
          common_value, {"off", "on"}, "-pbe-bie-convergence-dielectric-amp");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-estimator-model",
                              "--pbe-bie-convergence-estimator-model"},
                             common_value, i)) {
      pbe.tabi_convergence_estimator_model =
          require_choice(common_value,
                         {"stage3l", "calibrated-estimator-v1", "channel-tail",
                          "channel-tail-eps"},
                         "-pbe-bie-convergence-estimator-model");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-channel-calibration",
                              "--pbe-bie-convergence-channel-calibration"},
                             common_value, i)) {
      pbe.tabi_convergence_channel_calibration = require_choice(
          common_value,
          {"stage3m", "k3-sqrt-eps", "k3-eps", "k4-sqrt-eps", "k4-eps",
           "all-derivative-sqrt-eps", "polynomial-eps"},
          "-pbe-bie-convergence-channel-calibration");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-split-policy",
                              "--pbe-bie-convergence-split-policy"},
                             common_value, i)) {
      pbe.tabi_convergence_split_policy =
          require_choice(common_value, {"global", "k3-hot"},
                         "-pbe-bie-convergence-split-policy");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-split-selectivity",
                              "--pbe-bie-convergence-split-selectivity"},
                             common_value, i)) {
      pbe.tabi_convergence_split_selectivity = require_choice(
          common_value, {"legacy", "absolute-aware", "channel-absolute-aware"},
          "-pbe-bie-convergence-split-selectivity");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-split-absolute-floor",
                              "--pbe-bie-convergence-split-absolute-floor"},
                             common_value, i)) {
      pbe.tabi_convergence_split_absolute_floor = parse_double_range(
          common_value, "-pbe-bie-convergence-split-absolute-floor", 0.0,
          1.0e-4);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-overreject-audit",
                              "--pbe-bie-convergence-overreject-audit"},
                             common_value, i)) {
      pbe.tabi_convergence_overreject_audit =
          require_choice(common_value, {"off", "sampled", "full-small"},
                         "-pbe-bie-convergence-overreject-audit");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-overreject-detail",
                              "--pbe-bie-convergence-overreject-detail"},
                             common_value, i)) {
      pbe.tabi_convergence_overreject_detail =
          require_choice(common_value, {"off", "summary", "sampled-detail"},
                         "-pbe-bie-convergence-overreject-detail");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-shape-audit",
                              "--pbe-bie-convergence-shape-audit"},
                             common_value, i)) {
      pbe.tabi_convergence_shape_audit =
          require_choice(common_value, {"off", "summary", "sampled-detail"},
                         "-pbe-bie-convergence-shape-audit");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-kernel-response-audit",
                              "--pbe-bie-convergence-kernel-response-audit"},
                             common_value, i)) {
      pbe.tabi_convergence_kernel_response_audit =
          require_choice(common_value, {"off", "summary", "sampled-detail"},
                         "-pbe-bie-convergence-kernel-response-audit");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-channel-missed-semantics",
                              "--pbe-bie-channel-missed-semantics"},
                             common_value, i)) {
      pbe.tabi_channel_missed_semantics =
          require_choice(common_value, {"legacy", "accepted-only"},
                         "-pbe-bie-channel-missed-semantics");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-accepted-miss-diagnostics",
                              "--pbe-bie-accepted-miss-diagnostics"},
                             common_value, i)) {
      pbe.tabi_accepted_miss_diagnostics =
          require_choice(common_value, {"off", "summary", "sampled-detail"},
                         "-pbe-bie-accepted-miss-diagnostics");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-accepted-miss-absolute-floor",
                              "--pbe-bie-accepted-miss-absolute-floor"},
                             common_value, i)) {
      pbe.tabi_accepted_miss_absolute_floor = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-convergence-k3-accepted-rho-cap",
                              "--pbe-bie-convergence-k3-accepted-rho-cap"},
                             common_value, i)) {
      pbe.tabi_convergence_k3_accepted_rho_cap = stod(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-mac-distance-policy", "--pbe-bie-mac-distance-policy"},
            common_value, i)) {
      pbe.tabi_mac_distance_policy =
          require_choice(common_value, {"center", "surface-margin"},
                         "-pbe-bie-mac-distance-policy");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-mac-surface-margin-factor",
                              "--pbe-bie-mac-surface-margin-factor"},
                             common_value, i)) {
      pbe.tabi_mac_surface_margin_factor = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-mac-surface-margin-policy",
                              "--pbe-bie-mac-surface-margin-policy"},
                             common_value, i)) {
      pbe.tabi_mac_surface_margin_policy =
          require_choice(common_value, {"all", "adaptive", "risk-scaled"},
                         "-pbe-bie-mac-surface-margin-policy");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-mac-surface-margin-aspect-threshold",
                              "--pbe-bie-mac-surface-margin-aspect-threshold"},
                             common_value, i)) {
      pbe.tabi_mac_surface_margin_aspect_threshold = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-mac-surface-margin-rho-trigger",
                              "--pbe-bie-mac-surface-margin-rho-trigger"},
                             common_value, i)) {
      pbe.tabi_mac_surface_margin_rho_trigger = stod(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-mac-surface-margin-soft-aspect-threshold",
             "--pbe-bie-mac-surface-margin-soft-aspect-threshold"},
            common_value, i)) {
      pbe.tabi_mac_surface_margin_soft_aspect_threshold = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-mac-surface-margin-soft-rho-trigger",
                              "--pbe-bie-mac-surface-margin-soft-rho-trigger"},
                             common_value, i)) {
      pbe.tabi_mac_surface_margin_soft_rho_trigger = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-mac-surface-margin-soft-factor",
                              "--pbe-bie-mac-surface-margin-soft-factor"},
                             common_value, i)) {
      pbe.tabi_mac_surface_margin_soft_factor = stod(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-target-batch-size", "--pbe-bie-target-batch-size"},
            common_value, i)) {
      pbe.tabi_target_batch_size = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-leaf-pair-min-source-panels",
                              "--pbe-bie-leaf-pair-min-source-panels"},
                             common_value, i)) {
      pbe.tabi_leaf_pair_min_source_panels = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-leaf-pair-min-targets",
                              "--pbe-bie-leaf-pair-min-targets"},
                             common_value, i)) {
      pbe.tabi_leaf_pair_min_targets = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-leaf-pair-max-lookahead",
                              "--pbe-bie-leaf-pair-max-lookahead"},
                             common_value, i)) {
      pbe.tabi_leaf_pair_max_lookahead = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-expansion-audit", "--pbe-bie-expansion-audit"},
            common_value, i)) {
      pbe.tabi_expansion_audit =
          require_choice(common_value, {"off", "sampled", "full-small"},
                         "-pbe-bie-expansion-audit");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-expansion-audit-channel",
                              "--pbe-bie-expansion-audit-channel"},
                             common_value, i)) {
      pbe.tabi_expansion_audit_channel =
          require_choice(common_value, {"off", "summary", "sampled-detail"},
                         "-pbe-bie-expansion-audit-channel");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-pre-solver-progress", "--pbe-bie-pre-solver-progress"},
            common_value, i)) {
      pbe.tabi_pre_solver_progress = require_choice(
          common_value, {"off", "on"}, "-pbe-bie-pre-solver-progress");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-expansion-audit-sample-count",
                              "--pbe-bie-expansion-audit-sample-count"},
                             common_value, i)) {
      pbe.tabi_expansion_audit_sample_count = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-expansion-audit-max-panels",
                              "--pbe-bie-expansion-audit-max-panels"},
                             common_value, i)) {
      pbe.tabi_expansion_audit_max_panels = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-matvec-audit", "--pbe-bie-matvec-audit"},
            common_value, i)) {
      pbe.tabi_matvec_audit = require_choice(
          common_value, {"off", "sampled", "sampled-detail", "full-small"},
          "-pbe-bie-matvec-audit");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-matvec-audit-sample-count",
                              "--pbe-bie-matvec-audit-sample-count"},
                             common_value, i)) {
      pbe.tabi_matvec_audit_sample_count = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-node-patch-operator-audit",
             "--pbe-bie-node-patch-operator-audit"},
            common_value, i)) {
      pbe.tabi_node_patch_operator_audit = require_choice(
          common_value, {"on", "off"},
          "-pbe-bie-node-patch-operator-audit");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-diagnostics", "--pbe-bie-diagnostics",
                              "--diagnostics"},
                             common_value, i)) {
      pbe.tabi_diagnostics = require_choice(
          common_value, {"off", "audit", "full"}, "-pbe-bie-diagnostics");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-bie-progress", "--pbe-bie-progress"},
                             common_value, i)) {
      pbe.tabi_progress = require_choice(
          common_value, {"off", "auto", "detail"}, "-pbe-bie-progress");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-progress-interval", "--pbe-bie-progress-interval"},
            common_value, i)) {
      pbe.tabi_progress_interval = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-gmres-restart", "--pbe-bie-gmres-restart"},
            common_value, i)) {
      pbe.tabi_gmres_restart = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-true-residual-probe", "--pbe-bie-true-residual-probe"},
            common_value, i)) {
      pbe.tabi_true_residual_probe =
          require_choice(common_value, {"restart", "interval", "off"},
                         "-pbe-bie-true-residual-probe");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-true-residual-probe-scope",
                              "--pbe-bie-true-residual-probe-scope"},
                             common_value, i)) {
      pbe.tabi_true_residual_probe_scope =
          require_choice(common_value, {"full", "sampled", "adaptive"},
                         "-pbe-bie-true-residual-probe-scope");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-true-residual-probe-interval",
                              "--pbe-bie-true-residual-probe-interval"},
                             common_value, i)) {
      pbe.tabi_true_residual_probe_interval = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-bie-true-residual-sample-count",
                              "--pbe-bie-true-residual-sample-count"},
                             common_value, i)) {
      pbe.tabi_true_residual_sample_count = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-bie-energy-projection", "--pbe-bie-energy-projection"},
            common_value, i)) {
      pbe.tabi_energy_projection = require_choice(
          common_value, {"boundary", "born-check", "charge-potential-debug"},
          "-pbe-bie-energy-projection");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-stability-audit", "--pbe-stability-audit"},
                             common_value, i)) {
      pbe.stability_audit = require_choice(
          common_value, {"off", "summary", "detail"}, "-pbe-stability-audit");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-stability-mode", "--pbe-stability-mode"},
                             common_value, i)) {
      pbe.stability_mode = require_choice(
          common_value, {"off", "diagnose", "anchor", "hysteresis", "ensemble"},
          "-pbe-stability-mode");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-grid-anchor", "--pbe-grid-anchor"},
                             common_value, i)) {
      pbe.grid_anchor = require_choice(
          common_value, {"auto", "per-frame", "first-frame", "fixed-center"},
          "-pbe-grid-anchor");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-scale", "--pbe-scale"}, common_value,
                             i)) {
      pbe.scale = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-margin", "--pbe-margin"}, common_value,
                             i)) {
      pbe.margin = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-gridbox", "--pbe-gridbox"},
                             common_value, i)) {
      pbe.gridbox =
          require_choice(common_value, {"cuboidal", "cubic"}, "-pbe-gridbox");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-int-diel", "--pbe-int-diel"},
                             common_value, i)) {
      pbe.int_diel = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-ext-diel", "--pbe-ext-diel"},
                             common_value, i)) {
      pbe.ext_diel = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-salt", "--pbe-salt"}, common_value,
                             i)) {
      pbe.salt = stod(common_value);
      saltcon = pbe.salt;
      salt_concentration = pbe.salt;
      continue;
    }
    if (consume_option_value(arg, {"-pbe-temp", "--pbe-temp"}, common_value,
                             i)) {
      pbe.temp = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-model", "--pbe-model"}, common_value,
                             i)) {
      pbe.model =
          require_choice(common_value, {"linear", "nonlinear"}, "-pbe-model");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-biomodel", "--pbe-biomodel"},
                             common_value, i)) {
      pbe.biomodel =
          require_choice(common_value, {"pbe", "rpbe"}, "-pbe-biomodel");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-dielectric", "--pbe-dielectric"},
                             common_value, i)) {
      pbe.dielectric_model = require_choice(
          common_value, {"two-dielectric", "gaussian"}, "-pbe-dielectric");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-boundary", "--pbe-boundary"},
                             common_value, i)) {
      pbe.boundary = require_choice(
          common_value, {"coulombic", "dipolar", "zero"}, "-pbe-boundary");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-charge-assignment", "--pbe-charge-assignment"},
            common_value, i)) {
      pbe.charge_assignment =
          require_choice(common_value, {"trilinear", "delphi", "gaussian"},
                         "-pbe-charge-assignment");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-surface", "--pbe-surface"},
                             common_value, i)) {
      pbe.surface =
          require_choice(common_value,
                         {"vdw", "sas", "ses", "delphi-ses", "delphi-vdw",
                          "gaussian", "gaussian-cutoff"},
                         "-pbe-surface");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-radius-convention", "--pbe-radius-convention"},
            common_value, i)) {
      pbe.radius_convention =
          require_choice(common_value, {"auto", "native", "delphi-bondi"},
                         "-pbe-radius-convention");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-probe", "--pbe-probe"}, common_value,
                             i)) {
      pbe.probe_radius = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-ion-access", "--pbe-ion-access"},
                             common_value, i)) {
      pbe.ion_access = require_choice(common_value, {"dielectric", "stern"},
                                      "-pbe-ion-access");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-ion-radius", "--pbe-ion-radius"},
                             common_value, i)) {
      pbe.ion_radius = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-max-iter", "--pbe-max-iter"},
                             common_value, i)) {
      pbe.max_iter = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-rms", "--pbe-rms"}, common_value, i)) {
      pbe.rms_tol = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-dphi", "--pbe-dphi"}, common_value,
                             i)) {
      pbe.dphi_tol = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-omega", "--pbe-omega"}, common_value,
                             i)) {
      pbe.omega = lowercase_copy(common_value);
      if (pbe.omega != "auto")
        pbe.omega_value = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-max-grid", "--pbe-max-grid"},
                             common_value, i)) {
      pbe.max_grid_points = static_cast<std::size_t>(stoull(common_value));
      continue;
    }
    if (consume_option_value(arg, {"-pbe-map", "--pbe-map"}, common_value, i)) {
      pbe.map = require_choice(
          common_value, {"off", "phi", "epsilon", "ion", "all"}, "-pbe-map");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-map-frames", "--pbe-map-frames"},
                             common_value, i)) {
      pbe.map_frames = require_choice(
          common_value, {"none", "first", "last", "all"}, "-pbe-map-frames");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-atom-potentials", "--pbe-atom-potentials"},
                             common_value, i)) {
      pbe.atom_potentials =
          require_choice(common_value, {"off", "csv"}, "-pbe-atom-potentials");
      if (pbe.atom_potentials != "off")
        pbe.flag_pbe = true;
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-bie-surface-output", "--pbe-bie-surface-output"},
            common_value, i)) {
      pbe.tabi_surface_output =
          require_choice(common_value, {"off", "csv", "vtk", "all"},
                         "-pbe-bie-surface-output");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-kernel", "--pbe-kernel"}, common_value,
                             i)) {
      pbe.kernel = require_choice(common_value, {"auto", "scalar", "avx2"},
                                  "-pbe-kernel");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-focus", "--pbe-focus"}, common_value,
                             i)) {
      pbe.focus =
          require_choice(common_value, {"off", "auto", "delphi"}, "-pbe-focus");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-multigrid", "--pbe-multigrid"},
                             common_value, i)) {
      const std::string multigrid = require_choice(
          common_value, {"off", "focus", "auto", "vcycle"}, "-pbe-multigrid");
      pbe.multigrid = multigrid;
      if (multigrid == "focus" || multigrid == "auto")
        pbe.focus = "auto";
      continue;
    }
    if (consume_option_value(arg, {"-pbe-focus-levels", "--pbe-focus-levels"},
                             common_value, i)) {
      pbe.focus_levels = stoi(common_value);
      if (pbe.focus_levels > 1 && pbe.focus == "off")
        pbe.focus = "auto";
      continue;
    }
    if (consume_option_value(arg, {"-pbe-focus-margin", "--pbe-focus-margin"},
                             common_value, i)) {
      pbe.focus_margin = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-mg-levels", "--pbe-mg-levels"},
                             common_value, i)) {
      pbe.mg_levels = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-mg-pre", "--pbe-mg-pre"}, common_value,
                             i)) {
      pbe.mg_pre = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-mg-post", "--pbe-mg-post"},
                             common_value, i)) {
      pbe.mg_post = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-mg-coarse-iter", "--pbe-mg-coarse-iter"},
                             common_value, i)) {
      pbe.mg_coarse_iter = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nl-max-iter", "--pbe-nl-max-iter"},
                             common_value, i)) {
      pbe.nl_max_iter = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nl-inner-iter", "--pbe-nl-inner-iter"},
                             common_value, i)) {
      pbe.nl_inner_iter = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nl-stagnation", "--pbe-nl-stagnation"},
                             common_value, i)) {
      pbe.nl_stagnation = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nl-block", "--pbe-nl-block"},
                             common_value, i)) {
      pbe.nl_block = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nl-dphi", "--pbe-nl-dphi"},
                             common_value, i)) {
      pbe.nl_dphi = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nl-relax", "--pbe-nl-relax"},
                             common_value, i)) {
      pbe.nl_relax = lowercase_copy(common_value);
      if (pbe.nl_relax != "auto")
        pbe.nl_relax_value = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nl-clip", "--pbe-nl-clip"},
                             common_value, i)) {
      pbe.nl_clip = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-nl-continuation", "--pbe-nl-continuation"},
                             common_value, i)) {
      pbe.nl_continuation =
          require_choice(common_value, {"off", "auto"}, "-pbe-nl-continuation");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-nl-line-search", "--pbe-nl-line-search"},
                             common_value, i)) {
      pbe.nl_line_search =
          require_choice(common_value, {"off", "auto"}, "-pbe-nl-line-search");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nl-residual", "--pbe-nl-residual"},
                             common_value, i)) {
      pbe.nl_residual = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nl-solver", "--pbe-nl-solver"},
                             common_value, i)) {
      pbe.nl_solver =
          require_choice(common_value, {"picard", "newton"}, "-pbe-nl-solver");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nl-inner", "--pbe-nl-inner"},
                             common_value, i)) {
      pbe.nl_inner = require_choice(common_value, {"auto", "sor", "vcycle"},
                                    "-pbe-nl-inner");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nl-multigrid", "--pbe-nl-multigrid"},
                             common_value, i)) {
      pbe.nl_multigrid =
          require_choice(common_value, {"off", "fas"}, "-pbe-nl-multigrid");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nwt-kernel", "--pbe-nwt-kernel"},
                             common_value, i)) {
      pbe.nwt_kernel = require_choice(
          common_value, {"auto", "legacy", "delphi-half"}, "-pbe-nwt-kernel");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-nwt-homo-kernel", "--pbe-nwt-homo-kernel"},
                             common_value, i)) {
      pbe.nwt_homo_kernel = require_choice(
          common_value,
          {"auto", "packed-list", "row-span", "avx2", "salt-tiled"},
          "-pbe-nwt-homo-kernel");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-nwt-progress", "--pbe-nwt-progress"},
                             common_value, i)) {
      pbe.nwt_progress = require_choice(common_value, {"auto", "off", "strict"},
                                        "-pbe-nwt-progress");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-linear-kernel", "--pbe-linear-kernel"},
                             common_value, i)) {
      pbe.linear_kernel =
          require_choice(common_value, {"auto", "legacy", "half-packed"},
                         "-pbe-linear-kernel");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-gaussian-exp", "--pbe-gaussian-exp"},
                             common_value, i)) {
      pbe.gaussian_exp = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-gaussian-sigma", "--pbe-gaussian-sigma"},
                             common_value, i)) {
      pbe.gaussian_sigma = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-density-cutoff", "--pbe-density-cutoff"},
                             common_value, i)) {
      pbe.density_cutoff = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-surface-cutoff", "--pbe-surface-cutoff"},
                             common_value, i)) {
      pbe.surface_cutoff = lowercase_copy(common_value);
      if (pbe.surface_cutoff != "auto")
        pbe.surface_cutoff_value = stod(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-gap-diel", "--pbe-gap-diel"},
                             common_value, i)) {
      pbe.gap_diel = stod(common_value);
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-probe-file", "--pbe-probe-file",
                              "-pbe-test-probe-file", "--pbe-test-probe-file"},
                             common_value, i)) {
      pbe.probe_file = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-pbe-probe-residue", "--pbe-probe-residue"},
                             common_value, i)) {
      pbe.probe_residue = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-pbe-probe-site", "--pbe-probe-site"},
                             common_value, i)) {
      pbe.probe_site = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-pbe-focus-center", "--pbe-focus-center"},
                             common_value, i)) {
      parse_pbe_focus_center(common_value);
      continue;
    }
    if (arg == "-pbe-dgsolv" || arg == "--pbe-dgsolv") {
      if (i + 1 < argc) {
        std::string next_value = lowercase_copy(string(argv[i + 1]));
        if (next_value == "1" || next_value == "true" || next_value == "yes" ||
            next_value == "on" || next_value == "0" || next_value == "false" ||
            next_value == "no" || next_value == "off") {
          pbe.dgsolv = parse_bool_equals(next_value, "-pbe-dgsolv");
          if (pbe.dgsolv)
            pbe.flag_pbe = true;
          ++i;
          continue;
        }
      }
      pbe.dgsolv = true;
      pbe.flag_pbe = true;
      continue;
    }
    if (arg.rfind("-pbe-dgsolv=", 0) == 0 ||
        arg.rfind("--pbe-dgsolv=", 0) == 0) {
      pbe.dgsolv =
          parse_bool_equals(arg.substr(arg.find('=') + 1), "-pbe-dgsolv");
      if (pbe.dgsolv)
        pbe.flag_pbe = true;
      continue;
    }
    if (consume_option_value(arg, {"-pbe-test-pqr", "--pbe-test-pqr"},
                             common_value, i)) {
      pbe.flag_pbe = true;
      pbe.test_pqr = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-pbe-binding-cycle", "--pbe-binding-cycle"},
                             common_value, i)) {
      pbe.binding_cycle = require_choice(
          common_value, {"off", "pqr-components"}, "-pbe-binding-cycle");
      if (pbe.binding_cycle != "off")
        pbe.flag_pbe = true;
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-binding-receptor-pqr", "--pbe-binding-receptor-pqr"},
            common_value, i)) {
      pbe.binding_receptor_pqr = common_value;
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-binding-ligand-pqr", "--pbe-binding-ligand-pqr"},
            common_value, i)) {
      pbe.binding_ligand_pqr = common_value;
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-test-grid-size", "--pbe-test-grid-size"},
                             common_value, i)) {
      pbe.test_grid_size = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-test-center", "--pbe-test-center"},
                             common_value, i)) {
      parse_pbe_test_center(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-pbe-parity-dump", "--pbe-parity-dump"},
                             common_value, i)) {
      pbe.parity_dump =
          require_choice(common_value, {"off", "summary", "points", "all"},
                         "-pbe-parity-dump");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-parity-map-dump", "--pbe-parity-map-dump"},
                             common_value, i)) {
      pbe.parity_map_dump = require_choice(
          common_value, {"off", "samples", "full"}, "-pbe-parity-map-dump");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-test-charge-grid", "--pbe-test-charge-grid"},
            common_value, i)) {
      pbe.test_charge_grid = require_choice(common_value, {"native", "oracle"},
                                            "-pbe-test-charge-grid");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-test-charge-grid-diff", "--pbe-test-charge-grid-diff"},
            common_value, i)) {
      pbe.test_charge_grid_diff =
          require_choice(common_value, {"off", "samples", "full"},
                         "-pbe-test-charge-grid-diff");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-test-surface-map", "--pbe-test-surface-map"},
            common_value, i)) {
      pbe.test_surface_map = require_choice(common_value, {"native", "oracle"},
                                            "-pbe-test-surface-map");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-test-midpoint-epsilon", "--pbe-test-midpoint-epsilon"},
            common_value, i)) {
      pbe.test_midpoint_epsilon = require_choice(
          common_value, {"native", "oracle"}, "-pbe-test-midpoint-epsilon");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-test-midpoint-diff", "--pbe-test-midpoint-diff"},
            common_value, i)) {
      pbe.test_midpoint_diff = require_choice(
          common_value, {"off", "samples", "full"}, "-pbe-test-midpoint-diff");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-test-vdwms-remap", "--pbe-test-vdwms-remap"},
            common_value, i)) {
      pbe.test_vdwms_remap = require_choice(common_value, {"auto", "off"},
                                            "-pbe-test-vdwms-remap");
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-test-boundary-map", "--pbe-test-boundary-map"},
            common_value, i)) {
      pbe.test_boundary_map = require_choice(common_value, {"native", "oracle"},
                                             "-pbe-test-boundary-map");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-test-boundary-convention", "--pbe-test-boundary-convention"},
            common_value, i)) {
      pbe.test_boundary_convention =
          require_choice(common_value, {"native-real", "delphi-grid", "oracle"},
                         "-pbe-test-boundary-convention");
      if (pbe.test_boundary_convention == "oracle")
        pbe.test_boundary_map = "oracle";
      continue;
    }
    if (consume_option_value(
            arg, {"-pbe-test-field-units", "--pbe-test-field-units"},
            common_value, i)) {
      pbe.test_field_units =
          require_choice(common_value, {"native", "delphi-dimensionless"},
                         "-pbe-test-field-units");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-test-operator-convention", "--pbe-test-operator-convention"},
            common_value, i)) {
      pbe.test_operator_convention = require_choice(
          common_value, {"native", "delphi"}, "-pbe-test-operator-convention");
      continue;
    }
    if (consume_option_value(arg, {"-pbe-test-rhs-map", "--pbe-test-rhs-map"},
                             common_value, i)) {
      pbe.test_rhs_map = require_choice(common_value, {"native", "oracle"},
                                        "-pbe-test-rhs-map");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-test-screen-map", "--pbe-test-screen-map"},
                             common_value, i)) {
      pbe.test_screen_map = require_choice(common_value, {"native", "oracle"},
                                           "-pbe-test-screen-map");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-test-denom-map", "--pbe-test-denom-map"},
                             common_value, i)) {
      pbe.test_denom_map = require_choice(common_value, {"native", "oracle"},
                                          "-pbe-test-denom-map");
      continue;
    }
    if (consume_option_value(arg,
                             {"-pbe-test-nl-source", "--pbe-test-nl-source"},
                             common_value, i)) {
      pbe.test_nl_source =
          require_choice(common_value, {"titan", "delphi-dimensionless"},
                         "-pbe-test-nl-source");
      continue;
    }
    if (arg == "-pbe-test-newton-derivative" ||
        arg == "--pbe-test-newton-derivative") {
      if (i + 1 < argc) {
        std::string next_value = lowercase_copy(string(argv[i + 1]));
        if (next_value == "1" || next_value == "true" || next_value == "yes" ||
            next_value == "on" || next_value == "0" || next_value == "false" ||
            next_value == "no" || next_value == "off") {
          pbe.test_newton_derivative =
              parse_bool_equals(next_value, "-pbe-test-newton-derivative");
          ++i;
          continue;
        }
      }
      pbe.test_newton_derivative = true;
      continue;
    }
    if (arg.rfind("-pbe-test-newton-derivative=", 0) == 0 ||
        arg.rfind("--pbe-test-newton-derivative=", 0) == 0) {
      pbe.test_newton_derivative = parse_bool_equals(
          arg.substr(arg.find('=') + 1), "-pbe-test-newton-derivative");
      continue;
    }
    if (consume_option_value(
            arg,
            {"-pbe-test-energy-convention", "--pbe-test-energy-convention"},
            common_value, i)) {
      pbe.test_energy_convention = require_choice(
          common_value,
          {"auto", "atom", "grid", "delphi-corrected", "delphi-iscm"},
          "-pbe-test-energy-convention");
      continue;
    }

    if (consume_option_value(arg, {"-gbsa-perf", "--gbsa-perf"}, common_value,
                             i)) {
      gbsa_perf = require_choice(common_value, {"off", "summary", "detail"},
                                 "-gbsa-perf");
      continue;
    }
    if (consume_option_value(
            arg, {"-gbsa-sasa-points", "--gbsa-sasa-points"},
            common_value, i)) {
      gbsa_sasa_points = stoi(common_value);
      continue;
    }
    if (consume_option_value(arg, {"-gbsa-surften", "--gbsa-surften"},
                             common_value, i)) {
      gbsa_surften = parse_numeric_option(common_value, "-gbsa-surften");
      if (gbsa_surften < 0.0) {
        TitanUI::print_error(
            "-gbsa-surften must be a non-negative surface tension in "
            "kcal/mol/A^2.");
        exit(EXIT_FAILURE);
      }
      continue;
    }
    if (consume_option_value(arg, {"-gbsa-rgbmax", "--gbsa-rgbmax"},
                             common_value, i)) {
      // Cutoff on the HCT descreening integral for the effective Born radii.
      // "off" keeps the uncut sum (TITAN default, reproduces earlier results);
      // 25 is Amber's egb default. Any finite value changes DeltaG_polar, so
      // the chosen value is recorded in gbsa_summary.dat.
      const std::string choice = lowercase_copy(common_value);
      if (choice == "off" || choice == "none" || choice == "inf" ||
          choice == "infinite") {
        gbsa_rgbmax = std::numeric_limits<double>::infinity();
      } else {
        gbsa_rgbmax = parse_numeric_option(common_value, "-gbsa-rgbmax");
        if (!(gbsa_rgbmax > 0.0)) {
          TitanUI::print_error(
              "-gbsa-rgbmax must be a positive distance in Angstrom, or "
              "'off' for no cutoff.");
          exit(EXIT_FAILURE);
        }
      }
      continue;
    }
    if (consume_option_value(arg, {"-gbsa-ie-block", "--gbsa-ie-block"},
                             common_value, i)) {
      gbsa_ie_block = lowercase_copy(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-gbsa-ie-min-samples", "--gbsa-ie-min-samples"},
            common_value, i)) {
      gbsa_ie_min_samples = stoi(common_value);
      continue;
    }
    if (consume_option_value(
            arg, {"-gbsa-ie-filter-sigma", "--gbsa-ie-filter-sigma"},
            common_value, i)) {
      gbsa_ie_filter_sigma = stod(common_value);
      continue;
    }
    if (arg == "-gbsa") {
      flag_gbsa = true;
    }
    if (arg == "-alpb") {
      flag_alpb = true;
      flag_gbsa = true;
    }
    if (arg == "-dssp") {
      dssp.flag_dssp = true;
    }
    if (arg == "-no-dssp") {
      dssp.flag_dssp = false;
    }
    if (arg == "-rms" || arg == "-rmsd" || arg == "-rmsf") {
      rms.flag_rms = true;
    }
    if (arg == "-rg2") {
      rg.flag_rg2 = true;
    }
    if (arg.rfind("-rg2-hist-bins=", 0) == 0 ||
        arg.rfind("--rg2-hist-bins=", 0) == 0) {
      rg.hist_bins = stoi(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-rg2-hist-bins" || arg == "--rg2-hist-bins") && i + 1 < argc) {
      rg.hist_bins = stoi(string(argv[i + 1]));
      ++i;
    }
    if (arg == "-lj") {
      lj.flag_lj = true;
    }
    if (arg == "-hbk" || arg == "--hbk") {
      hbk.flag_hbk = true;
    }
    if (arg.rfind("-hbk-kind=", 0) == 0 || arg.rfind("--hbk-kind=", 0) == 0) {
      hbk.flag_hbk = true;
      hbk.hbk_kinds.push_back(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-hbk-kind" || arg == "--hbk-kind") && i + 1 < argc) {
      hbk.flag_hbk = true;
      hbk.hbk_kinds.push_back(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-hbk-domain=", 0) == 0 ||
        arg.rfind("--hbk-domain=", 0) == 0) {
      hbk.flag_hbk = true;
      parse_hbk_domain(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-hbk-domain" || arg == "--hbk-domain") && i + 1 < argc) {
      hbk.flag_hbk = true;
      parse_hbk_domain(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-hbk-maxlag=", 0) == 0 ||
        arg.rfind("--hbk-maxlag=", 0) == 0) {
      hbk.flag_hbk = true;
      hbk.hbk_maxlag = stoi(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-hbk-maxlag" || arg == "--hbk-maxlag") && i + 1 < argc) {
      hbk.flag_hbk = true;
      hbk.hbk_maxlag = stoi(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-hbk-hbond-cutoff=", 0) == 0 ||
        arg.rfind("--hbk-hbond-cutoff=", 0) == 0) {
      hbk.flag_hbk = true;
      hbk.hbk_hbond_cutoff = stod(arg.substr(arg.find('=') + 1));
      hbk.hbk_hbond_cutoff_explicit = true;
    }
    if ((arg == "-hbk-hbond-cutoff" || arg == "--hbk-hbond-cutoff") &&
        i + 1 < argc) {
      hbk.flag_hbk = true;
      hbk.hbk_hbond_cutoff = stod(string(argv[i + 1]));
      hbk.hbk_hbond_cutoff_explicit = true;
      ++i;
    }
    if (arg.rfind("-hbk-angle=", 0) == 0 || arg.rfind("--hbk-angle=", 0) == 0) {
      hbk.flag_hbk = true;
      hbk.hbk_angle_deg = stod(arg.substr(arg.find('=') + 1));
      hbk.hbk_angle_explicit = true;
    }
    if ((arg == "-hbk-angle" || arg == "--hbk-angle") && i + 1 < argc) {
      hbk.flag_hbk = true;
      hbk.hbk_angle_deg = stod(string(argv[i + 1]));
      hbk.hbk_angle_explicit = true;
      ++i;
    }
    if (arg.rfind("-hbk-salt-cutoff=", 0) == 0 ||
        arg.rfind("--hbk-salt-cutoff=", 0) == 0) {
      hbk.flag_hbk = true;
      hbk.hbk_salt_cutoff = stod(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-hbk-salt-cutoff" || arg == "--hbk-salt-cutoff") &&
        i + 1 < argc) {
      hbk.flag_hbk = true;
      hbk.hbk_salt_cutoff = stod(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-hbk-water-cutoff=", 0) == 0 ||
        arg.rfind("--hbk-water-cutoff=", 0) == 0) {
      hbk.flag_hbk = true;
      hbk.hbk_water_cutoff = stod(arg.substr(arg.find('=') + 1));
      hbk.hbk_water_cutoff_explicit = true;
    }
    if ((arg == "-hbk-water-cutoff" || arg == "--hbk-water-cutoff") &&
        i + 1 < argc) {
      hbk.flag_hbk = true;
      hbk.hbk_water_cutoff = stod(string(argv[i + 1]));
      hbk.hbk_water_cutoff_explicit = true;
      ++i;
    }
    if (arg.rfind("-lj-cut=", 0) == 0 || arg.rfind("--lj-cut=", 0) == 0) {
      lj.lj_cluster_cutoff = stod(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-lj-cut" || arg == "--lj-cut") && i + 1 < argc) {
      lj.lj_cluster_cutoff = stod(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-lj-min-cluster=", 0) == 0 ||
        arg.rfind("--lj-min-cluster=", 0) == 0) {
      lj.lj_min_cluster_size = stoi(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-lj-min-cluster" || arg == "--lj-min-cluster") &&
        i + 1 < argc) {
      lj.lj_min_cluster_size = stoi(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-lj-rho-bin=", 0) == 0 ||
        arg.rfind("--lj-rho-bin=", 0) == 0) {
      lj.lj_rho_bin = stod(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-lj-rho-bin" || arg == "--lj-rho-bin") && i + 1 < argc) {
      lj.lj_rho_bin = stod(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-lj-rho-max=", 0) == 0 ||
        arg.rfind("--lj-rho-max=", 0) == 0) {
      lj.lj_rho_max = stod(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-lj-rho-max" || arg == "--lj-rho-max") && i + 1 < argc) {
      lj.lj_rho_max = stod(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-lj-solid-q6=", 0) == 0 ||
        arg.rfind("--lj-solid-q6=", 0) == 0) {
      lj.lj_solid_q6 = stod(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-lj-solid-q6" || arg == "--lj-solid-q6") && i + 1 < argc) {
      lj.lj_solid_q6 = stod(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-lj-solid-min-cn=", 0) == 0 ||
        arg.rfind("--lj-solid-min-cn=", 0) == 0) {
      lj.lj_solid_min_cn = stoi(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-lj-solid-min-cn" || arg == "--lj-solid-min-cn") &&
        i + 1 < argc) {
      lj.lj_solid_min_cn = stoi(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-lj-crystal-tol=", 0) == 0 ||
        arg.rfind("--lj-crystal-tol=", 0) == 0) {
      lj.lj_crystal_tol = stod(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-lj-crystal-tol" || arg == "--lj-crystal-tol") &&
        i + 1 < argc) {
      lj.lj_crystal_tol = stod(string(argv[i + 1]));
      ++i;
    }
    if (arg == "-sq" || arg == "--sq") {
      sq.flag_sq = true;
    }
    if (arg == "-corr-knn" || arg == "--corr-knn") {
      corr_knn.flag_corr_knn = true;
      flag_network = true;
    }
    if (arg.rfind("-sq-channel=", 0) == 0 ||
        arg.rfind("--sq-channel=", 0) == 0) {
      sq.sq_channels.push_back(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-sq-channel" || arg == "--sq-channel") && i + 1 < argc) {
      sq.sq_channels.push_back(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-sq-rdf-bin=", 0) == 0 ||
        arg.rfind("--sq-rdf-bin=", 0) == 0) {
      sq.sq_rdf_bin = stod(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-sq-rdf-bin" || arg == "--sq-rdf-bin") && i + 1 < argc) {
      sq.sq_rdf_bin = stod(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-sq-rdf-max=", 0) == 0 ||
        arg.rfind("--sq-rdf-max=", 0) == 0) {
      parse_sq_auto_or_value(arg.substr(arg.find('=') + 1), sq.sq_rdf_max,
                             sq.sq_rdf_max_auto, "-sq-rdf-max");
    }
    if ((arg == "-sq-rdf-max" || arg == "--sq-rdf-max") && i + 1 < argc) {
      parse_sq_auto_or_value(string(argv[i + 1]), sq.sq_rdf_max,
                             sq.sq_rdf_max_auto, "-sq-rdf-max");
      ++i;
    }
    if (arg.rfind("-sq-q-bin=", 0) == 0 || arg.rfind("--sq-q-bin=", 0) == 0) {
      sq.sq_q_bin = stod(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-sq-q-bin" || arg == "--sq-q-bin") && i + 1 < argc) {
      sq.sq_q_bin = stod(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-sq-q-max=", 0) == 0 || arg.rfind("--sq-q-max=", 0) == 0) {
      parse_sq_auto_or_value(arg.substr(arg.find('=') + 1), sq.sq_q_max,
                             sq.sq_q_max_auto, "-sq-q-max");
    }
    if ((arg == "-sq-q-max" || arg == "--sq-q-max") && i + 1 < argc) {
      parse_sq_auto_or_value(string(argv[i + 1]), sq.sq_q_max, sq.sq_q_max_auto,
                             "-sq-q-max");
      ++i;
    }
    if (arg.rfind("-corr-knn-k=", 0) == 0 ||
        arg.rfind("--corr-knn-k=", 0) == 0) {
      corr_knn.estimator_k = stoi(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-corr-knn-k" || arg == "--corr-knn-k") && i + 1 < argc) {
      corr_knn.estimator_k = stoi(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-corr-knn-tau=", 0) == 0 ||
        arg.rfind("--corr-knn-tau=", 0) == 0) {
      corr_knn.tau_lag = stoi(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-corr-knn-tau" || arg == "--corr-knn-tau") && i + 1 < argc) {
      corr_knn.tau_lag = stoi(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-corr-knn-pair-cut=", 0) == 0 ||
        arg.rfind("--corr-knn-pair-cut=", 0) == 0) {
      corr_knn.pair_cutoff_ang = stod(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-corr-knn-pair-cut" || arg == "--corr-knn-pair-cut") &&
        i + 1 < argc) {
      corr_knn.pair_cutoff_ang = stod(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-corr-knn-pair-topk=", 0) == 0 ||
        arg.rfind("--corr-knn-pair-topk=", 0) == 0) {
      corr_knn.pair_topk = stoi(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-corr-knn-pair-topk" || arg == "--corr-knn-pair-topk") &&
        i + 1 < argc) {
      corr_knn.pair_topk = stoi(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-corr-knn-pair-prior=", 0) == 0 ||
        arg.rfind("--corr-knn-pair-prior=", 0) == 0) {
      parse_corr_knn_pair_prior(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-corr-knn-pair-prior" || arg == "--corr-knn-pair-prior") &&
        i + 1 < argc) {
      parse_corr_knn_pair_prior(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-corr-knn-pair-contact-cut=", 0) == 0 ||
        arg.rfind("--corr-knn-pair-contact-cut=", 0) == 0) {
      corr_knn.pair_contact_cutoff_ang = stod(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-corr-knn-pair-contact-cut" ||
         arg == "--corr-knn-pair-contact-cut") &&
        i + 1 < argc) {
      corr_knn.pair_contact_cutoff_ang = stod(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-corr-knn-prune-window=", 0) == 0 ||
        arg.rfind("--corr-knn-prune-window=", 0) == 0) {
      parse_corr_knn_prune_window(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-corr-knn-prune-window" || arg == "--corr-knn-prune-window") &&
        i + 1 < argc) {
      parse_corr_knn_prune_window(string(argv[i + 1]));
      ++i;
    }
    if (arg == "-corr-knn-no-te" || arg == "--corr-knn-no-te") {
      corr_knn.compute_te = false;
    }
    if (arg == "-validation" || arg == "--validation") {
      enable_validation();
    }
    if (arg.rfind("-validation-ck-multiples=", 0) == 0 ||
        arg.rfind("--validation-ck-multiples=", 0) == 0) {
      enable_validation();
      parse_validation_int_list(arg.substr(arg.find('=') + 1),
                                validation.ck_multiples,
                                "-validation-ck-multiples");
    }
    if ((arg == "-validation-ck-multiples" ||
         arg == "--validation-ck-multiples") &&
        i + 1 < argc) {
      enable_validation();
      parse_validation_int_list(string(argv[i + 1]), validation.ck_multiples,
                                "-validation-ck-multiples");
      ++i;
    }
    if (arg.rfind("-validation-vamp-components=", 0) == 0 ||
        arg.rfind("--validation-vamp-components=", 0) == 0) {
      enable_validation();
      parse_validation_int_list(arg.substr(arg.find('=') + 1),
                                validation.vamp_components,
                                "-validation-vamp-components");
    }
    if ((arg == "-validation-vamp-components" ||
         arg == "--validation-vamp-components") &&
        i + 1 < argc) {
      enable_validation();
      parse_validation_int_list(string(argv[i + 1]), validation.vamp_components,
                                "-validation-vamp-components");
      ++i;
    }
    if (arg.rfind("-validation-vamp-lags=", 0) == 0 ||
        arg.rfind("--validation-vamp-lags=", 0) == 0) {
      enable_validation();
      parse_validation_int_list(arg.substr(arg.find('=') + 1),
                                validation.vamp_lags, "-validation-vamp-lags");
    }
    if ((arg == "-validation-vamp-lags" || arg == "--validation-vamp-lags") &&
        i + 1 < argc) {
      enable_validation();
      parse_validation_int_list(string(argv[i + 1]), validation.vamp_lags,
                                "-validation-vamp-lags");
      ++i;
    }
    if (arg.rfind("-validation-vamp-clusters=", 0) == 0 ||
        arg.rfind("--validation-vamp-clusters=", 0) == 0) {
      enable_validation();
      parse_validation_int_list(arg.substr(arg.find('=') + 1),
                                validation.vamp_clusters,
                                "-validation-vamp-clusters");
    }
    if ((arg == "-validation-vamp-clusters" ||
         arg == "--validation-vamp-clusters") &&
        i + 1 < argc) {
      enable_validation();
      parse_validation_int_list(string(argv[i + 1]), validation.vamp_clusters,
                                "-validation-vamp-clusters");
      ++i;
    }
    if (arg.rfind("-validation-vamp-folds=", 0) == 0 ||
        arg.rfind("--validation-vamp-folds=", 0) == 0) {
      enable_validation();
      validation.vamp_folds = stoi(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-validation-vamp-folds" || arg == "--validation-vamp-folds") &&
        i + 1 < argc) {
      enable_validation();
      validation.vamp_folds = stoi(string(argv[i + 1]));
      ++i;
    }
    if (arg == "-validation-auto-pick" || arg == "--validation-auto-pick") {
      enable_validation();
      validation.auto_pick = true;
    }
    if (arg.rfind("-validation-bootstrap=", 0) == 0 ||
        arg.rfind("--validation-bootstrap=", 0) == 0) {
      enable_validation();
      validation.bootstrap_samples = stoi(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-validation-bootstrap" || arg == "--validation-bootstrap") &&
        i + 1 < argc) {
      enable_validation();
      validation.bootstrap_samples = stoi(string(argv[i + 1]));
      ++i;
    }
    if (arg.rfind("-validation-bootstrap-seed=", 0) == 0 ||
        arg.rfind("--validation-bootstrap-seed=", 0) == 0) {
      enable_validation();
      validation.bootstrap_seed =
          static_cast<unsigned int>(std::stoul(arg.substr(arg.find('=') + 1)));
    }
    if ((arg == "-validation-bootstrap-seed" ||
         arg == "--validation-bootstrap-seed") &&
        i + 1 < argc) {
      enable_validation();
      validation.bootstrap_seed =
          static_cast<unsigned int>(std::stoul(string(argv[i + 1])));
      ++i;
    }
    if (arg.rfind("-dssp-hbond-cutoff=", 0) == 0 ||
        arg.rfind("--dssp-hbond-cutoff=", 0) == 0) {
      dssp.hbond_energy_cutoff = stod(arg.substr(arg.find('=') + 1));
    }
    if ((arg == "-dssp-hbond-cutoff" || arg == "--dssp-hbond-cutoff") &&
        i + 1 < argc) {
      dssp.hbond_energy_cutoff = stod(string(argv[i + 1]));
      ++i;
    }
    if (arg == "-network") {
      flag_network = true;
      flag_network_explicit = true;
    }
    if (string(argv[i]) == "-pca") {
      flag_pca = true;
      flag_network = true; // PCA depends on CA data from network module
    }
    if (string(argv[i]) == "-ml") {
      flag_ml = true;
    }
    if (string(argv[i]).find("--fes-grid=") == 0 ||
        string(argv[i]).find("-fes-grid=") == 0) {
      fes_grid = stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]) == "-allostery") {
      flag_allostery = true;
      flag_network = true; // Allostery depends on DCCM trajectory data
    }
    if (string(argv[i]) == "-lspmd") {
      lspmd.flag_lspmd = true;
      // NOTE: does NOT auto-enable flag_network — LSP-MD runs standalone
      // without triggering DCCM matrix calculation.
    }
    if (string(argv[i]).find("-lspmd_cut=") == 0) {
      lspmd.lspmd_cutoff = stod(string(argv[i]).substr(11));
    }
    if (string(argv[i]).find("--allostery-tau=") == 0 ||
        string(argv[i]).find("-allostery-tau=") == 0) {
      allostery_tau =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--allostery-tau" ||
         string(argv[i]) == "-allostery-tau") &&
        i + 1 < argc) {
      allostery_tau = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--allostery-bins=") == 0 ||
        string(argv[i]).find("-allostery-bins=") == 0) {
      allostery_bins =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--allostery-bins" ||
         string(argv[i]) == "-allostery-bins") &&
        i + 1 < argc) {
      allostery_bins = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--allostery-knn=") == 0 ||
        string(argv[i]).find("-allostery-knn=") == 0) {
      allostery_knn =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--allostery-knn" ||
         string(argv[i]) == "-allostery-knn") &&
        i + 1 < argc) {
      allostery_knn = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--allostery-cut=") == 0 ||
        string(argv[i]).find("-allostery-cut=") == 0) {
      allostery_cutoff =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--allostery-cut" ||
         string(argv[i]) == "-allostery-cut") &&
        i + 1 < argc) {
      allostery_cutoff = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--allostery-cut-mode=") == 0 ||
        string(argv[i]).find("-allostery-cut-mode=") == 0) {
      parse_allostery_prune_window(
          string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--allostery-cut-mode" ||
         string(argv[i]) == "-allostery-cut-mode") &&
        i + 1 < argc) {
      parse_allostery_prune_window(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--allostery-prune-window=") == 0 ||
        string(argv[i]).find("-allostery-prune-window=") == 0) {
      parse_allostery_prune_window(
          string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--allostery-prune-window" ||
         string(argv[i]) == "-allostery-prune-window") &&
        i + 1 < argc) {
      parse_allostery_prune_window(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--allostery-knn-prior=") == 0 ||
        string(argv[i]).find("-allostery-knn-prior=") == 0) {
      parse_allostery_knn_prior(
          string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--allostery-knn-prior" ||
         string(argv[i]) == "-allostery-knn-prior") &&
        i + 1 < argc) {
      parse_allostery_knn_prior(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--allostery-knn-contact-cut=") == 0 ||
        string(argv[i]).find("-allostery-knn-contact-cut=") == 0) {
      allostery_knn_contact_cutoff =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--allostery-knn-contact-cut" ||
         string(argv[i]) == "-allostery-knn-contact-cut") &&
        i + 1 < argc) {
      allostery_knn_contact_cutoff = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]) == "--allostery-no-te" ||
        string(argv[i]) == "-allostery-no-te") {
      allostery_compute_te = false;
    }
    if (string(argv[i]).find("--allostery-estimator=") == 0 ||
        string(argv[i]).find("-allostery-estimator=") == 0) {
      parse_allostery_estimator(
          string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--allostery-estimator" ||
         string(argv[i]) == "-allostery-estimator") &&
        i + 1 < argc) {
      parse_allostery_estimator(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--allostery-estimator-k=") == 0 ||
        string(argv[i]).find("-allostery-estimator-k=") == 0) {
      allostery_estimator_k =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--allostery-estimator-k" ||
         string(argv[i]) == "-allostery-estimator-k") &&
        i + 1 < argc) {
      allostery_estimator_k = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--allostery-kde-bandwidth=") == 0 ||
        string(argv[i]).find("-allostery-kde-bandwidth=") == 0) {
      allostery_kde_bandwidth =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--allostery-kde-bandwidth" ||
         string(argv[i]) == "-allostery-kde-bandwidth") &&
        i + 1 < argc) {
      allostery_kde_bandwidth = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]) == "-tda") {
      flag_tda = true;
    }
    if (string(argv[i]) == "-tica") {
      msm.flag_tica = true;
      flag_network = true; // TICA depends on CA data from network module
    }
    if (string(argv[i]) == "-zts" || string(argv[i]) == "--zts") {
      zts.flag_zts = true;
    }
    if (string(argv[i]).find("--zts-images=") == 0 ||
        string(argv[i]).find("-zts-images=") == 0) {
      zts.zts_images =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--zts-images" ||
         string(argv[i]) == "-zts-images") &&
        i + 1 < argc) {
      zts.zts_images = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--zts-iter=") == 0 ||
        string(argv[i]).find("-zts-iter=") == 0) {
      zts.zts_max_iter =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--zts-iter" || string(argv[i]) == "-zts-iter") &&
        i + 1 < argc) {
      zts.zts_max_iter = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--zts-step=") == 0 ||
        string(argv[i]).find("-zts-step=") == 0) {
      zts.zts_step =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--zts-step" || string(argv[i]) == "-zts-step") &&
        i + 1 < argc) {
      zts.zts_step = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--zts-tol=") == 0 ||
        string(argv[i]).find("-zts-tol=") == 0) {
      zts.zts_tol = stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--zts-tol" || string(argv[i]) == "-zts-tol") &&
        i + 1 < argc) {
      zts.zts_tol = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--zts-source=") == 0 ||
        string(argv[i]).find("-zts-source=") == 0) {
      zts.zts_source =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--zts-source" ||
         string(argv[i]) == "-zts-source") &&
        i + 1 < argc) {
      zts.zts_source = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--zts-sink=") == 0 ||
        string(argv[i]).find("-zts-sink=") == 0) {
      zts.zts_sink =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--zts-sink" || string(argv[i]) == "-zts-sink") &&
        i + 1 < argc) {
      zts.zts_sink = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--zts-sigma=") == 0 ||
        string(argv[i]).find("-zts-sigma=") == 0) {
      zts.zts_fes_sigma =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--zts-sigma" || string(argv[i]) == "-zts-sigma") &&
        i + 1 < argc) {
      zts.zts_fes_sigma = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]) == "--zts-no-smooth" ||
        string(argv[i]) == "-zts-no-smooth") {
      zts.zts_smooth_fes = false;
    }
    if (string(argv[i]).find("--zts-init=") == 0 ||
        string(argv[i]).find("-zts-init=") == 0) {
      zts.zts_init_mode = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if ((string(argv[i]) == "--zts-init" || string(argv[i]) == "-zts-init") &&
        i + 1 < argc) {
      zts.zts_init_mode = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("--zts-domain=") == 0 ||
        string(argv[i]).find("-zts-domain=") == 0) {
      zts.zts_domain = string(argv[i]).substr(string(argv[i]).find('=') + 1);
      zts.zts_domain_explicit = true;
    }
    if ((string(argv[i]) == "--zts-domain" ||
         string(argv[i]) == "-zts-domain") &&
        i + 1 < argc) {
      zts.zts_domain = string(argv[i + 1]);
      zts.zts_domain_explicit = true;
      ++i;
    }
    if (string(argv[i]).find("--tica-lag=") == 0 ||
        string(argv[i]).find("-tica-lag=") == 0) {
      msm.tica_lag =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--tica-lag" || string(argv[i]) == "-tica-lag") &&
        i + 1 < argc) {
      msm.tica_lag = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--tica-components=") == 0 ||
        string(argv[i]).find("-tica-components=") == 0) {
      msm.tica_components =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--tica-components" ||
         string(argv[i]) == "-tica-components") &&
        i + 1 < argc) {
      msm.tica_components = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--tica-backend=") == 0 ||
        string(argv[i]).find("-tica-backend=") == 0) {
      msm.tica_backend = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if ((string(argv[i]) == "--tica-backend" ||
         string(argv[i]) == "-tica-backend") &&
        i + 1 < argc) {
      msm.tica_backend = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("--tica-rank=") == 0 ||
        string(argv[i]).find("-tica-rank=") == 0) {
      msm.tica_rank =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--tica-rank" || string(argv[i]) == "-tica-rank") &&
        i + 1 < argc) {
      msm.tica_rank = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]) == "-umap") {
      umap.flag_umap = true;
      msm.flag_tica = true;
      flag_network = true;
    }
    if (string(argv[i]).find("--umap-neighbors=") == 0 ||
        string(argv[i]).find("-umap-neighbors=") == 0) {
      umap.umap_neighbors =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--umap-neighbors" ||
         string(argv[i]) == "-umap-neighbors") &&
        i + 1 < argc) {
      umap.umap_neighbors = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--umap-components=") == 0 ||
        string(argv[i]).find("-umap-components=") == 0) {
      umap.umap_components =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--umap-components" ||
         string(argv[i]) == "-umap-components") &&
        i + 1 < argc) {
      umap.umap_components = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--umap-epochs=") == 0 ||
        string(argv[i]).find("-umap-epochs=") == 0) {
      umap.umap_epochs =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--umap-epochs" ||
         string(argv[i]) == "-umap-epochs") &&
        i + 1 < argc) {
      umap.umap_epochs = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--umap-min-dist=") == 0 ||
        string(argv[i]).find("-umap-min-dist=") == 0) {
      umap.umap_min_dist =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--umap-min-dist" ||
         string(argv[i]) == "-umap-min-dist") &&
        i + 1 < argc) {
      umap.umap_min_dist = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--umap-spread=") == 0 ||
        string(argv[i]).find("-umap-spread=") == 0) {
      umap.umap_spread =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--umap-spread" ||
         string(argv[i]) == "-umap-spread") &&
        i + 1 < argc) {
      umap.umap_spread = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--umap-seed=") == 0 ||
        string(argv[i]).find("-umap-seed=") == 0) {
      umap.umap_seed = static_cast<unsigned int>(
          stoul(string(argv[i]).substr(string(argv[i]).find('=') + 1)));
    }
    if ((string(argv[i]) == "--umap-seed" || string(argv[i]) == "-umap-seed") &&
        i + 1 < argc) {
      umap.umap_seed = static_cast<unsigned int>(stoul(string(argv[i + 1])));
      ++i;
    }
    if (string(argv[i]).find("--umap-knn-mode=") == 0 ||
        string(argv[i]).find("-umap-knn-mode=") == 0) {
      umap.umap_knn_mode =
          string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if ((string(argv[i]) == "--umap-knn-mode" ||
         string(argv[i]) == "-umap-knn-mode") &&
        i + 1 < argc) {
      umap.umap_knn_mode = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]) == "-umap-knn-benchmark" ||
        string(argv[i]) == "--umap-knn-benchmark") {
      umap.umap_knn_benchmark = true;
    }
    if (string(argv[i]) == "-umap-verbose" ||
        string(argv[i]) == "--umap-verbose") {
      umap.umap_verbose = true;
    }
    // --- MSM Phase 2 flags ---
    if (string(argv[i]) == "-msm") {
      msm.flag_msm = true;
      msm.flag_tica = true; // MSM depends on TICA projection
      flag_network = true;  // TICA depends on CA data from network module
    }
    if (string(argv[i]) == "-msm-clusters" && i + 1 < argc) {
      msm.msm_clusters = stoi(string(argv[i + 1]));
      msm.msm_clusters_user_set = true;
      ++i;
    }
    if (string(argv[i]) == "-msm-lag" && i + 1 < argc) {
      msm.msm_lag = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]) == "-msm-nts" && i + 1 < argc) {
      msm.msm_nts = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]) == "-msm-kmeans-iter" && i + 1 < argc) {
      msm.msm_kmeans_iter = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]) == "-msm-seed" && i + 1 < argc) {
      msm.msm_seed = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]) == "-msm-macrostates" && i + 1 < argc) {
      msm.msm_macrostates = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]) == "-msm-no-boundary-split") {
      msm.msm_auto_boundary = false;
    }
    if (string(argv[i]) == "-msm-no-pcca") {
      msm.msm_run_pcca = false;
    }
    if (string(argv[i]) == "-msm-no-its") {
      msm.msm_run_its = false;
    }
    if (string(argv[i]) == "-msm-its-lags" && i + 1 < argc) {
      msm.msm_its_lags.clear();
      istringstream iss{string(argv[i + 1])};
      for (string tok; getline(iss, tok, ',');)
        if (!tok.empty())
          msm.msm_its_lags.push_back(stoi(tok));
      ++i;
    }
    if (string(argv[i]).find("--msm-its-solver=") == 0 ||
        string(argv[i]).find("-msm-its-solver=") == 0) {
      parse_msm_its_solver(
          string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--msm-its-solver" ||
         string(argv[i]) == "-msm-its-solver") &&
        i + 1 < argc) {
      parse_msm_its_solver(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--msm-its-topk=") == 0 ||
        string(argv[i]).find("-msm-its-topk=") == 0) {
      msm.msm_its_topk =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--msm-its-topk" ||
         string(argv[i]) == "-msm-its-topk") &&
        i + 1 < argc) {
      msm.msm_its_topk = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--msm-its-max-k-full=") == 0 ||
        string(argv[i]).find("-msm-its-max-k-full=") == 0) {
      msm.msm_its_max_k_full =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--msm-its-max-k-full" ||
         string(argv[i]) == "-msm-its-max-k-full") &&
        i + 1 < argc) {
      msm.msm_its_max_k_full = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--pca-solver=") == 0 ||
        string(argv[i]).find("-pca-solver=") == 0) {
      parse_pca_solver(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--pca-solver" ||
         string(argv[i]) == "-pca-solver") &&
        i + 1 < argc) {
      parse_pca_solver(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--pca-topk=") == 0 ||
        string(argv[i]).find("-pca-topk=") == 0) {
      msm.pca_topk =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--pca-topk" || string(argv[i]) == "-pca-topk") &&
        i + 1 < argc) {
      msm.pca_topk = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--pca-max-k-full=") == 0 ||
        string(argv[i]).find("-pca-max-k-full=") == 0) {
      msm.pca_max_k_full =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--pca-max-k-full" ||
         string(argv[i]) == "-pca-max-k-full") &&
        i + 1 < argc) {
      msm.pca_max_k_full = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]) == "--pca-symmetry-aware" ||
        string(argv[i]) == "-pca-symmetry-aware") {
      msm.pca_symmetry_aware = true;
    }
    if (string(argv[i]) == "--no-pca-symmetry-aware" ||
        string(argv[i]) == "-no-pca-symmetry-aware") {
      msm.pca_symmetry_aware = false;
    }

    // --- LLPS flags ---
    if (consume_bool_option(arg, {"-llps"}, llps.flag_llps, i)) {
    }
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
      continue;
    }
    std::string gist_value;
    if (consume_required_option_value(arg, {"-gist-resolution", "--gist-resolution"}, gist_value, i)) {
      gist.gist_resolution=parse_numeric_option(gist_value, "-gist-resolution");
      if (gist.gist_resolution<=0) { TitanUI::print_error("-gist-resolution must be positive."); exit(EXIT_FAILURE); }
      continue;
    }
    if (consume_required_option_value(arg, {"-gist-padding", "--gist-padding"}, gist_value, i)) {
      gist.gist_padding=parse_numeric_option(gist_value, "-gist-padding");
      if (gist.gist_padding<=0) { TitanUI::print_error("-gist-padding must be positive."); exit(EXIT_FAILURE); }
      continue;
    }
    if (consume_required_option_value(arg, {"-gist-temperature", "--gist-temperature"}, gist_value, i)) {
      gist.gist_temperature=parse_numeric_option(gist_value, "-gist-temperature");
      if (gist.gist_temperature<=0) { TitanUI::print_error("-gist-temperature must be positive."); exit(EXIT_FAILURE); }
      continue;
    }
    if (consume_required_option_value(arg, {"-gist-cutoff", "--gist-cutoff"}, gist_value, i)) {
      gist.gist_cutoff=parse_numeric_option(gist_value, "-gist-cutoff");
      if (gist.gist_cutoff<=0) { TitanUI::print_error("-gist-cutoff must be positive."); exit(EXIT_FAILURE); }
      continue;
    }
    if (consume_required_option_value(arg, {"-gist-min-count", "--gist-min-count"}, gist_value, i)) {
      gist.gist_min_count=parse_integer_option(gist_value, "-gist-min-count");
      if (gist.gist_min_count<1) { TitanUI::print_error("-gist-min-count must be positive."); exit(EXIT_FAILURE); }
      continue;
    }
    if (consume_required_option_value(arg, {"-gist-water-model", "--gist-water-model"}, gist_value, i)) {
      gist.gist_water_model=gist_value;
      continue;
    }
    // --- IFP (Interaction Fingerprint) flags ---
    if (string(argv[i]) == "-ifp") {
      ifp.flag_ifp = true;
    }
    if (consume_bool_option(
            arg, {"-ifp-pbctopo-certified-only"},
            ifp.pbctopo_certified_only, i)) {
      continue;
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
      continue;
    }
    if (consume_required_option_value(arg, {"-gbnsr6-spacing"}, common_value, i)) {
      gbnsr6_spacing = parse_numeric_option(common_value, "-gbnsr6-spacing");
      continue;
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
    if (consume_option_value(arg, {"-gbsa-temp", "--gbsa-temp"},
                             common_value, i)) {
      gbsa_temperature = stod(common_value);
    }
    if (consume_required_option_value(arg, {"-n_wat"}, common_value, i)) {
      n_wat = parse_integer_option(common_value, "-n_wat");
      if(n_wat < -1){TitanUI::print_error("-n_wat must be -1 or a zero-based water group index.");exit(EXIT_FAILURE);}
      continue;
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
    if (string(argv[i]) == "-h" || string(argv[i]) == "--help") {
      help = true;
      if (help_print() == 0)
        return 2;
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
      return 2;
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
    bool cache_enabled = true;
    if (consume_bool_option(arg, {"-cache", "--cache"}, cache_enabled, i)) {
      flag_cache = cache_enabled ? 1 : 2;
      continue;
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
        return titan_cli::kInvalidArguments;
      }
      sub_group_1 = std::move(selection.molecule_1);
      sub_group_2 = std::move(selection.molecule_2);
      flag_sub_groups = true;
      continue;
    }
    if (arg == "-fmap" &&
        (i + 1 >= argc || std::string(argv[i + 1]).rfind("-", 0) == 0)) {
      force_map = 1;
      continue;
    }
    if (consume_option_value(arg, {"-fmap"}, common_value, i)) {
      force_map = 1;
      if (common_value == "residue-atom" || common_value == "residue-residue" ||
          common_value == "atom-atom" || common_value == "atom-residue")
        fmap_type = common_value;
      continue;
    }
    if (consume_option_value(arg, {"-fmap-level"}, common_value, i)) {
      if (common_value == "all" || common_value == "per-res")
        fmap_level = common_value;
      continue;
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
      continue;
    }
    if (consume_option_value(arg, {"-qha-temp", "--qha-temp"}, common_value,
                             i)) {
      qha_temperature = parse_numeric_option(common_value, "-qha-temp");
      if (!std::isfinite(qha_temperature) || qha_temperature <= 0.0)
        throw std::invalid_argument("-qha-temp must be finite and positive");
      continue;
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
        return arg_file_status;
      ++i;
      continue;
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
      if (string(argv[i + 1]) == "surface" && string(argv[i + 1]) == "sphere")
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
      de.pull_max_evals = static_cast<size_t>(stoull(arg.substr(16)));
    }
    if (string(argv[i]) == "-pull-max-evals" && i + 1 < argc) {
      de.pull_max_evals = static_cast<size_t>(stoull(string(argv[i + 1])));
      ++i;
    }
    if (arg.rfind("-pull-seed=", 0) == 0) {
      de.pull_seed = static_cast<unsigned int>(stoul(arg.substr(11)));
    }
    if (string(argv[i]) == "-pull-seed" && i + 1 < argc) {
      de.pull_seed = static_cast<unsigned int>(stoul(string(argv[i + 1])));
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
      de.de_pop_num = stoi(string(argv[i + 1]));
    }
    if (string(argv[i]) == "-de-ite-num" && i + 1 < argc) {
      de.de_ite = stoi(string(argv[i + 1]));
    }
    if (arg.rfind("-de-algorithm=", 0) == 0) {
      de.de_algorithm = stoi(arg.substr(14));
      de.user_set_de_algorithm = true;
    }
    if (string(argv[i]) == "-de-algorithm" && i + 1 < argc) {
      de.de_algorithm = stoi(string(argv[i + 1]));
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
      de.de_stagnation_limit = stoi(string(argv[i + 1]));
    }
    if (arg.rfind("-de-memory-size=", 0) == 0) {
      de.de_memory_size = static_cast<size_t>(stoull(arg.substr(16)));
      de.user_set_de_memory_size = true;
    }
    if (string(argv[i]) == "-de-memory-size" && i + 1 < argc) {
      de.de_memory_size = static_cast<size_t>(stoull(string(argv[i + 1])));
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
      de.de_min_pop_num = static_cast<size_t>(stoull(arg.substr(16)));
      de.user_set_de_min_pop_num = true;
    }
    if (string(argv[i]) == "-de-min-pop-num" && i + 1 < argc) {
      de.de_min_pop_num = static_cast<size_t>(stoull(string(argv[i + 1])));
      de.user_set_de_min_pop_num = true;
    }
    if (string(argv[i]) == "-de-pathway-group" && i + 1 < argc) {
      de.group_pathway = static_cast<size_t>(stoull(string(argv[i + 1])));
      de.flag_pathway = true;
      de.user_set_pathway_group = true;
    }
    if (string(argv[i]) == "-de-pathway-radius" && i + 1 < argc) {
      de.dis_along_path = parse_numeric_option(argv[i + 1], "-de-pathway-radius");
      de.path_graph_edge_cutoff = de.dis_along_path;
      de.path_recovery_enter_radius = de.dis_along_path;
      de.path_recovery_exit_radius = de.dis_along_path;
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
    }
    if (string(argv[i]) == "-pathway-recovery-enter" && i + 1 < argc) {
      de.path_recovery_enter_radius = parse_numeric_option(argv[i + 1], "-pathway-recovery-enter");
    }
    if (string(argv[i]) == "-pathway-recovery-exit" && i + 1 < argc) {
      de.path_recovery_exit_radius = parse_numeric_option(argv[i + 1], "-pathway-recovery-exit");
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
          static_cast<size_t>(stoull(string(argv[i + 1])));
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
    // --- DHAM (Dynamic Histogram Analysis Method) flags ---
    if (string(argv[i]) == "-dham") {
      dham.flag_dham = true;
      msm.flag_msm = true;  // DHAM depends on MSM (K-means + dtrajs)
      msm.flag_tica = true; // MSM depends on TICA projection
      flag_network = true;  // TICA depends on CA data from network module
    }
    if (string(argv[i]).find("-dham-temps=") == 0 ||
        string(argv[i]).find("--dham-temps=") == 0) {
      dham.temps_file = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if (string(argv[i]) == "-dham-temps" && i + 1 < argc) {
      dham.temps_file = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("-dham-segment-map=") == 0 ||
        string(argv[i]).find("--dham-segment-map=") == 0) {
      dham.segment_map_file = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if ((string(argv[i]) == "-dham-segment-map" ||
         string(argv[i]) == "--dham-segment-map") && i + 1 < argc) {
      dham.segment_map_file = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("-dham-ener=") == 0 ||
        string(argv[i]).find("--dham-ener=") == 0) {
      dham.ener_list_file =
          string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if (string(argv[i]) == "-dham-ener" && i + 1 < argc) {
      dham.ener_list_file = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("-dham-target-temp=") == 0 ||
        string(argv[i]).find("--dham-target-temp=") == 0) {
      dham.target_temp =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]) == "-dham-target-temp" && i + 1 < argc) {
      dham.target_temp = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("-dham-ensemble=") == 0 ||
        string(argv[i]).find("--dham-ensemble=") == 0) {
      dham.ensemble = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if (string(argv[i]) == "-dham-ensemble" && i + 1 < argc) {
      dham.ensemble = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("-dham-window-match=") == 0 ||
        string(argv[i]).find("--dham-window-match=") == 0) {
      dham.window_match = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if (string(argv[i]) == "-dham-window-match" && i + 1 < argc) {
      dham.window_match = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("-dham-pressure-bar=") == 0 ||
        string(argv[i]).find("--dham-pressure-bar=") == 0) {
      dham.target_pressure_bar =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]) == "-dham-pressure-bar" && i + 1 < argc) {
      dham.target_pressure_bar = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--dham-max-iter=") == 0 ||
        string(argv[i]).find("-dham-max-iter=") == 0) {
      dham.max_iter =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]).find("--dham-tol=") == 0 ||
        string(argv[i]).find("-dham-tol=") == 0) {
      dham.tolerance =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]).find("--dham-n-replicas=") == 0 ||
        string(argv[i]).find("-dham-n-replicas=") == 0) {
      dham.n_replicas =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]).find("-dham-dt-energy=") == 0 ||
        string(argv[i]).find("--dham-dt-energy=") == 0) {
      dham.dt_energy_ps =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]) == "-dham-dt-energy" && i + 1 < argc) {
      dham.dt_energy_ps = stod(string(argv[i + 1]));
      ++i;
    }
    // --- MBAR (temperature-ladder REMD/HREMD reweighting) flags ---
    if (string(argv[i]) == "-mbar") {
      mbar.flag_mbar = true;
      flag_network = true;
    }
    if (string(argv[i]).find("-mbar-temps=") == 0 ||
        string(argv[i]).find("--mbar-temps=") == 0) {
      mbar.temps_file = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if (string(argv[i]) == "-mbar-temps" && i + 1 < argc) {
      mbar.temps_file = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("-mbar-ener=") == 0 ||
        string(argv[i]).find("--mbar-ener=") == 0) {
      mbar.ener_list_file =
          string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if (string(argv[i]) == "-mbar-ener" && i + 1 < argc) {
      mbar.ener_list_file = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("-mbar-target-temp=") == 0 ||
        string(argv[i]).find("--mbar-target-temp=") == 0) {
      mbar.target_temp =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]) == "-mbar-target-temp" && i + 1 < argc) {
      mbar.target_temp = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("-mbar-domain=") == 0 ||
        string(argv[i]).find("--mbar-domain=") == 0) {
      mbar.domain = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if (string(argv[i]) == "-mbar-domain" && i + 1 < argc) {
      mbar.domain = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("-mbar-ensemble=") == 0 ||
        string(argv[i]).find("--mbar-ensemble=") == 0) {
      mbar.ensemble = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if (string(argv[i]) == "-mbar-ensemble" && i + 1 < argc) {
      mbar.ensemble = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("-mbar-window-match=") == 0 ||
        string(argv[i]).find("--mbar-window-match=") == 0) {
      mbar.window_match = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if (string(argv[i]) == "-mbar-window-match" && i + 1 < argc) {
      mbar.window_match = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("-mbar-dt-energy=") == 0 ||
        string(argv[i]).find("--mbar-dt-energy=") == 0) {
      mbar.dt_energy_ps =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]) == "-mbar-dt-energy" && i + 1 < argc) {
      mbar.dt_energy_ps = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("-mbar-temp-mode=") == 0 ||
        string(argv[i]).find("--mbar-temp-mode=") == 0) {
      mbar.temp_mode = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if (string(argv[i]) == "-mbar-temp-mode" && i + 1 < argc) {
      mbar.temp_mode = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("-mbar-temp-col=") == 0 ||
        string(argv[i]).find("--mbar-temp-col=") == 0) {
      mbar.temp_column =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]) == "-mbar-temp-col" && i + 1 < argc) {
      mbar.temp_column = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("-mbar-pressure-bar=") == 0 ||
        string(argv[i]).find("--mbar-pressure-bar=") == 0) {
      mbar.target_pressure_bar =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]) == "-mbar-pressure-bar" && i + 1 < argc) {
      mbar.target_pressure_bar = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("-mbar-bootstrap=") == 0 ||
        string(argv[i]).find("--mbar-bootstrap=") == 0) {
      mbar.bootstrap_samples =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]) == "-mbar-bootstrap" && i + 1 < argc) {
      mbar.bootstrap_samples = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("-mbar-scan-points=") == 0 ||
        string(argv[i]).find("--mbar-scan-points=") == 0) {
      mbar.scan_points =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if (string(argv[i]) == "-mbar-scan-points" && i + 1 < argc) {
      mbar.scan_points = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("-mbar-demux-map=") == 0 ||
        string(argv[i]).find("--mbar-demux-map=") == 0) {
      mbar.demux_map_file =
          string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if ((string(argv[i]) == "-mbar-demux-map" ||
         string(argv[i]) == "--mbar-demux-map") &&
        i + 1 < argc) {
      mbar.demux_map_file = string(argv[i + 1]);
      ++i;
    }
    // --- TPT (Transition Path Theory) flags ---
    if (string(argv[i]) == "-tpt") {
      tpt.flag_tpt = true;
      msm.flag_msm = true;  // TPT depends on MSM microstates / TPM
      msm.flag_tica = true; // MSM depends on TICA projection
      flag_network = true; // protein workflows still use network -> TICA -> MSM
    }
    if (string(argv[i]).find("--tpt-mode=") == 0 ||
        string(argv[i]).find("-tpt-mode=") == 0) {
      tpt.mode = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if ((string(argv[i]) == "--tpt-mode" || string(argv[i]) == "-tpt-mode") &&
        i + 1 < argc) {
      tpt.mode = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("--tpt-source=") == 0 ||
        string(argv[i]).find("-tpt-source=") == 0) {
      tpt.source_spec = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if ((string(argv[i]) == "--tpt-source" ||
         string(argv[i]) == "-tpt-source") &&
        i + 1 < argc) {
      tpt.source_spec = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("--tpt-sink=") == 0 ||
        string(argv[i]).find("-tpt-sink=") == 0) {
      tpt.sink_spec = string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if ((string(argv[i]) == "--tpt-sink" || string(argv[i]) == "-tpt-sink") &&
        i + 1 < argc) {
      tpt.sink_spec = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("--tpt-top-paths=") == 0 ||
        string(argv[i]).find("-tpt-top-paths=") == 0) {
      tpt.top_paths =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--tpt-top-paths" ||
         string(argv[i]) == "-tpt-top-paths") &&
        i + 1 < argc) {
      tpt.top_paths = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--tpt-flux-cut=") == 0 ||
        string(argv[i]).find("-tpt-flux-cut=") == 0) {
      tpt.flux_cut =
          stod(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--tpt-flux-cut" ||
         string(argv[i]) == "-tpt-flux-cut") &&
        i + 1 < argc) {
      tpt.flux_cut = stod(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--tpt-use-dham=") == 0 ||
        string(argv[i]).find("-tpt-use-dham=") == 0) {
      std::string value = string(argv[i]).substr(string(argv[i]).find('=') + 1);
      std::transform(
          value.begin(), value.end(), value.begin(),
          [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      tpt.use_dham = !(value == "no" || value == "false" || value == "0" ||
                       value == "off");
    }
    if (string(argv[i]).find("--tpt-demux-map=") == 0 ||
        string(argv[i]).find("-tpt-demux-map=") == 0) {
      tpt.demux_map_file =
          string(argv[i]).substr(string(argv[i]).find('=') + 1);
    }
    if ((string(argv[i]) == "--tpt-demux-map" ||
         string(argv[i]) == "-tpt-demux-map") &&
        i + 1 < argc) {
      tpt.demux_map_file = string(argv[i + 1]);
      ++i;
    }
    if (string(argv[i]).find("--tpt-bootstrap=") == 0 ||
        string(argv[i]).find("-tpt-bootstrap=") == 0) {
      tpt.bootstrap_samples =
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1));
    }
    if ((string(argv[i]) == "--tpt-bootstrap" ||
         string(argv[i]) == "-tpt-bootstrap") &&
        i + 1 < argc) {
      tpt.bootstrap_samples = stoi(string(argv[i + 1]));
      ++i;
    }
    if (string(argv[i]).find("--tpt-bootstrap-seed=") == 0 ||
        string(argv[i]).find("-tpt-bootstrap-seed=") == 0) {
      tpt.bootstrap_seed = static_cast<unsigned int>(
          stoi(string(argv[i]).substr(string(argv[i]).find('=') + 1)));
    }
    if ((string(argv[i]) == "--tpt-bootstrap-seed" ||
         string(argv[i]) == "-tpt-bootstrap-seed") &&
        i + 1 < argc) {
      tpt.bootstrap_seed = static_cast<unsigned int>(stoi(string(argv[i + 1])));
      ++i;
    }
    if ((string(argv[i]) == "--tpt-use-dham" ||
         string(argv[i]) == "-tpt-use-dham") &&
        i + 1 < argc) {
      std::string value = string(argv[i + 1]);
      std::transform(
          value.begin(), value.end(), value.begin(),
          [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      tpt.use_dham = !(value == "no" || value == "false" || value == "0" ||
                       value == "off");
      ++i;
    }
    if (string(argv[i]) == "-no-center") {
      pbc.no_center = true;
    }
    command += argv[i];
    command += " ";
  }
  // Parsing finished without throwing, so a later exception from the engine
  // must not be attributed to the last option inspected.
  titan_cli::active_option_token().clear();

  if (flag_topo_1 && flag_topo_2 && topo_name == topo_name_2) {
    flag_use_common_topo = true;
  }

  if (hotspot.flag_hotspot_report) {
    hotspot.source = lowercase_copy(hotspot.source);
    hotspot.rank_by = lowercase_copy(hotspot.rank_by);
    if (!hotspot.rank_by_explicit) {
      if (hotspot.source == "cas")
        hotspot.rank_by = "cas-ddg";
      else if (hotspot.source == "edecomp")
        hotspot.rank_by = "edecomp-abs-mean";
      else
        hotspot.rank_by = "combined";
    }
    if (const std::string validation_error = hotspot.validation_error();
        !validation_error.empty()) {
      TitanUI::print_error("Hotspot report: " + validation_error);
      exit(EXIT_FAILURE);
    }
    if (hotspot.source == "cas" || hotspot.source == "both")
      flag_cas = true;
    if (hotspot.source == "edecomp" || hotspot.source == "both") {
      edecomp.flag_edecomp = true;
      if (edecomp.level == "atom") {
        TitanUI::print_error("Hotspot report: EDecomp hotspot ranking requires "
                             "-edecomp-level=residue or -edecomp-level=both.");
        exit(EXIT_FAILURE);
      }
    }
  }

  if (edecomp.flag_edecomp) {
    const std::string validation_error = edecomp.validation_error();
    if (!validation_error.empty()) {
      TitanUI::print_error("Energy decomposition: " + validation_error);
      exit(EXIT_FAILURE);
    }
    if (edecomp.source == "gbsa" && !flag_gbsa) {
      TitanUI::print_error(
          "Energy decomposition: -edecomp-source=gbsa requires -gbsa.");
      exit(EXIT_FAILURE);
    }
    if (edecomp.source == "pbsa" && !pbe.pbsa) {
      TitanUI::print_error(
          "Energy decomposition: -edecomp-source=pbsa requires -pbsa=true.");
      exit(EXIT_FAILURE);
    }
  }

  if (pbc.solv_pbctopo) {
    const bool conflict_with_reference =
        pbc.solv_pbc == 1 || pbc.solv_pbc_int == 1 || pbc.solv_pbcn ||
        pbc.pbc_aggressive || pbc.pbc_mem || pbc.flag_pbc_ref || pbc.pbc_envi ||
        pbc.solv_pbc_clust;
    if (conflict_with_reference) {
      TitanUI::print_error(
          "The option -pbctopo cannot be combined with reference/legacy PBC "
          "modes.");
      return titan_cli::kInvalidArguments;
    }
  }

  return 0;
}
int arguments::parse_arguments_interactive(int argc_in) {
  int flag;
  string force_f;
  do {
    cout << "Please choose the force field that you want to use:" << '\n';
    cout << "1. AMBER99SB" << '\n';
    cout << "2. AMBER99SB-ILDN" << '\n';
    cout << "3. AMBER99SB-BSC1" << '\n';
    cout << "4. AMBER99SB-DISP" << '\n';
    cout << "5. AMBER14SB" << '\n';
    cout << "6. AMBER19SB" << '\n';
    cout << "7. GROMOS43A1" << '\n';
    cout << "8. GROMOS53A6" << '\n';
    cout << "9. GROMOS54A7" << '\n';
    cout << "10. CHARMM27" << '\n';
    cout << "11. CHARMM36" << '\n';
    cout << "12. OPLSAA" << '\n';
    getline(cin, force_f);
    istringstream st(force_f);
    st >> flag;
  } while (flag > 12 || flag < 1);
  if (flag == 1) {
    force_f = "AMBER99SB";
  } else if (flag == 2) {
    force_f = "AMBER99SB-ILDN";
  } else if (flag == 3) {
    force_f = "AMBER99SB-BSC1";
  } else if (flag == 4) {
    force_f = "AMBER99SB-DISP";
  } else if (flag == 5) {
    force_f = "AMBER14SB";
  } else if (flag == 6) {
    force_f = "AMBER19SB";
  } else if (flag == 7) {
    force_f = "GROMOS43A1";
  } else if (flag == 8) {
    force_f = "GROMOS53A6";
  } else if (flag == 9) {
    force_f = "GROMOS54A7";
  } else if (flag == 10) {
    force_f = "CHARMM27";
  } else if (flag == 11) {
    force_f = "CHARMM36";
  } else
    force_f = "OPLSAA";

  do {
    cout << "Do you want to take only minimun sc-ct distance? (Y/N=1/2)";
    if (!(cin >> flag_all_side_chain_ct)) {
      cin.clear();
      cin.ignore(10000, '\n');
    }
  } while (flag_all_side_chain_ct != 1 && flag_all_side_chain_ct != 2);
  do {
    cout << "Do you want to creat .PQR file? (Y/N=1/2)";
    cin >> flag_pqr;
  } while (flag_pqr != 1 && flag_pqr != 2);

  cout << "Do you want to use topo file (no if you don't want to use topo): ";
  cin >> topo_name;
  if (check_topo(topo_name, 1) == true)
    flag_topo_1 = true;

  cout << "Number of threads: ";
  cin >> nt;
  return 0;
}
bool arguments::check_topo(string NameofTopoFile, int Topo) {
  if (NameofTopoFile == "no")
    return false;
  else {
    auto assign_topo_name = [&](const std::string &name) {
      if (Topo == 2)
        topo_name_2 = name;
      else
        topo_name = name;
    };

    fstream input_topo_protein(NameofTopoFile, ios::in);
    bool topo_is_open{input_topo_protein.is_open()};
    input_topo_protein.close();
    if (topo_is_open == true)
      return true;
    else {
      int attempts = 0;
      constexpr int MAX_ATTEMPTS = 5;
      while (!topo_is_open) {
        cout << "Cannot open " << NameofTopoFile
             << ". Please type your lig topo file (no if you don't want to use "
                "topo): ";
        cin >> NameofTopoFile;
        cout << NameofTopoFile << "\n";
        assign_topo_name(NameofTopoFile);
        if (NameofTopoFile == "no")
          return false;
        else {
          input_topo_protein.open(NameofTopoFile, ios::in);
          if (input_topo_protein.is_open()) {
            input_topo_protein.close();
            return true;
          } else {
            input_topo_protein.close();
            ++attempts;
            if (attempts >= MAX_ATTEMPTS) {
              TitanUI::print_error("Too many invalid attempts (" +
                                   std::to_string(MAX_ATTEMPTS) +
                                   "). Exiting.");
              return false;
            }
            cout << "File not found. Attempts remaining: "
                 << (MAX_ATTEMPTS - attempts) << '\n';
            continue;
          }
        }
      }
      return false;
    }
  }
}
