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

// Helpers of arguments::parse_arguments() (formerly its local lambdas), shared
// by the option families in src/cli/parse_options_*.cpp.

std::string arguments::normalize_mol_type(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "ligand")
    value = "lig";
  return value;
}

bool arguments::assign_mol_type(const std::string &value, bool mol1) {
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
}

void arguments::parse_allostery_prune_window(std::string raw_mode) {
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
}

void arguments::parse_allostery_knn_prior(std::string raw_mode) {
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
}

void arguments::parse_corr_knn_prune_window(std::string raw_mode) {
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
}

void arguments::parse_corr_knn_pair_prior(std::string raw_mode) {
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
}

void arguments::parse_validation_int_list(const std::string &raw_list,
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
}

void arguments::enable_validation() {
  validation.flag_validation = true;
  msm.flag_msm = true;
  msm.flag_tica = true;
  flag_network = true;
}

void arguments::parse_allostery_estimator(std::string raw_mode) {
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
}

void arguments::parse_msm_its_solver(std::string raw_mode) {
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
}

void arguments::parse_pca_solver(std::string raw_mode) {
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
}

void arguments::parse_sq_auto_or_value(const std::string &raw_value,
                                       double &out, bool &auto_flag,
                                       const char *label) {
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
}

void arguments::parse_hbk_domain(std::string raw_mode) {
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
}

bool arguments::parse_bool_equals(std::string raw_value, const char *flag_name) {
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
}

bool arguments::consume_option_value(const std::string &arg,
                                     std::initializer_list<const char *> flags,
                                     std::string &value_out, int &arg_index) {
  // the tokens of the parse_arguments() call in progress
  const int argc = cli_argc_;
  char **const argv = cli_argv_;
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
}

bool arguments::consume_required_option_value(const std::string &arg,
                                              std::initializer_list<const char *> flags,
                                              std::string &value_out,
                                              int &arg_index) {
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
}

bool arguments::consume_bool_option(const std::string &arg,
                                    std::initializer_list<const char *> flags,
                                    bool &value_out, int &arg_index) {
  // the tokens of the parse_arguments() call in progress
  const int argc = cli_argc_;
  char **const argv = cli_argv_;
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
}

void arguments::parse_path_probe_lengths(const std::string &raw_list) {
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
}

bool arguments::consume_compound_option_value(const std::string &arg,
                                              std::initializer_list<const char *> flags,
                                              std::string &value_out,
                                              int &arg_index) {
  // the tokens of the parse_arguments() call in progress
  const int argc = cli_argc_;
  char **const argv = cli_argv_;
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
}

std::string arguments::lowercase_copy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string arguments::require_choice(std::string value,
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
}

double arguments::parse_double_range(const std::string &value,
                                     const char *flag_name, double min_value,
                                     double max_value) {
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
}

double arguments::parse_numeric_option(const std::string &value,
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
}

int arguments::parse_integer_option(const std::string &value,
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
}

std::string arguments::normalize_pbe_cli_alias(std::string arg) {
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
}

void arguments::apply_tabi_tree_preset(const std::string &preset) {
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
}

void arguments::parse_pbe_test_center(std::string value) {
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
}

void arguments::parse_pbe_focus_center(std::string value) {
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
}

void arguments::parse_memory_budget(std::string value) {
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
}

void arguments::parse_sg_memory_budget(std::string value) {
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
}

void arguments::parse_sg_cutoff(std::string value) {
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
}
