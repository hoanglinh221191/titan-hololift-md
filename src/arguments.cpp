#include "arguments.h"
#include "pbctopo/config.h"
#include "tabi/types.h"
#include "titan_segment_config.h"
#include "titan_tpt.h"
#include "titan_ui.h"
#include "titan_version_info.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

namespace fs = std::filesystem;

bool arguments::parse_input_file_is_open() {
  bool check_file_open{false};
  fstream test_file(pdb_name, ios::in);
  if (test_file.is_open() || flag_all_folders == true)
    check_file_open = true;
  test_file.close();
  return check_file_open;
}

int arguments::recheck_and_change_pdb_input() {
  add_input_file_type();

  if (flag_1_pdb == true && parse_input_file_is_open() == false &&
      flag_pattern_input == false && flag_listfile == false) {
    bool check_file_open{false};
    int attempts = 0;
    constexpr int MAX_ATTEMPTS = 5;
    do {
      pdb_name = string();
      cout << "Please type your input " << get_input_file_type() << " file:\n";
      cin >> pdb_name;
      add_input_file_type();
      cout << "Your input: " << pdb_name << '\n';
      if (pdb_name.length() < 1)
        return 1;
      flag_1_pdb = true;
      check_file_open = parse_input_file_is_open();
      if (!check_file_open) {
        ++attempts;
        if (attempts >= MAX_ATTEMPTS) {
          TitanUI::print_error("Too many invalid attempts (" +
                               std::to_string(MAX_ATTEMPTS) + "). Exiting.");
          return 1;
        }
        cout << "File not found. Attempts remaining: "
             << (MAX_ATTEMPTS - attempts) << '\n';
      }
    } while (check_file_open == false);
  } else
    return 0;
  return 0;
}

string arguments::get_build_number() const {
  return std::string(titan_build::full_version());
}

string arguments::get_pdbOutput() const {
  if (opdb_name_explicit && !out_pdb_name.empty())
    return out_pdb_name;
  fs::path p_temp = pdb_name;
  return "out_" + p_temp.stem().string() + ".pdb";
}

int arguments::post_process() {
  recheck_and_change_pdb_input();

  if (pdb_name.length() > 4 &&
      pdb_name.substr(pdb_name.length() - 4) != get_input_file_type()) {
    cout << "Your input file " << pdb_name << " is invalid, input file must be "
         << get_input_file_type() << " file!\n";
    exit(EXIT_FAILURE);
  }
  if (force_f != "AMBER99SB" && force_f != "AMBER99SB-ILDN" &&
      force_f != "AMBER99SB-BSC1" && force_f != "AMBER99SB-DISP" &&
      force_f != "AMBER14SB" && force_f != "AMBER19SB" &&
      force_f != "CHARMM27" && force_f != "CHARMM36" &&
      force_f != "GROMOS43A1" && force_f != "GROMOS53A6" &&
      force_f != "GROMOS54A7" && force_f != "OPLSAA" && force_f != "MARTINI") {
    cout << "Force field " << force_f << " is not supported!\n"
         << "This tool supports AMBER99SB, AMBER99SB-ILDN, AMBER99SB-BSC1, "
            "AMBER99SB-DISP, AMBER14SB, AMBER19SB, GROMOS43A1, GROMOS53A6, "
            "GROMOS54A7, CHARMM27, CHARMM36, OPLSAA and MARTINI\n"
            "Please check your option.\n";
    exit(EXIT_FAILURE);
  }
  // Martini CG uses relative permittivity epsilon_r = 15
  if (force_f == "MARTINI" && epsilon_in == 1.0) {
    epsilon_in = 15.0;
    TitanUI::print_info(
        "[MARTINI] Auto-set epsilon_in = 15.0 (Martini relative "
        "permittivity). Override with -epsilon_in if needed.");
  }
  if (!std::isfinite(epsilon_in) || epsilon_in <= 0.0) {
    TitanUI::print_error("Electrostatics: -epsilon_in must be finite and > 0.");
    exit(EXIT_FAILURE);
  }
  if (flag_elec_screening) {
    const bool supports_screening =
        force_f.find("AMBER") != string::npos ||
        force_f.find("CHARMM") != string::npos;
    if (!supports_screening) {
      TitanUI::print_error(
          "Electrostatic screening is currently supported only for AMBER "
          "and CHARMM force fields; refusing to run an unscreened " +
          force_f + " calculation.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(salt_concentration) || salt_concentration <= 0.0) {
      TitanUI::print_error(
          "Electrostatic screening: salt concentration must be finite and > 0.");
      exit(EXIT_FAILURE);
    }
  }
  if (begin_time < 0. || end_time < 0.) {
    cout << "Your begin time and end time must not be negative!\n";
    exit(EXIT_FAILURE);
  }
  if (dt < 0) {
    cout << "Time interval must not be negative!\n";
    exit(EXIT_FAILURE);
  }
  if (const std::string membrane_error = membrane.validation_error();
      !membrane_error.empty()) {
    TitanUI::print_error("Membrane: " + membrane_error);
    exit(EXIT_FAILURE);
  }
  if (const std::string s2_error = s2.validation_error(); !s2_error.empty()) {
    TitanUI::print_error("S2: " + s2_error);
    exit(EXIT_FAILURE);
  }
  std::transform(eq_weight_source.begin(), eq_weight_source.end(),
                 eq_weight_source.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (eq_weight_source != "auto" && eq_weight_source != "uniform" &&
      eq_weight_source != "dham" && eq_weight_source != "mbar") {
    TitanUI::print_error(
        "-eq-weight-source must be auto, uniform, dham, or mbar.");
    exit(EXIT_FAILURE);
  }
  std::transform(trajectory_semantics.begin(), trajectory_semantics.end(),
                 trajectory_semantics.begin(), [](unsigned char c) {
                   if (c == '-')
                     return '_';
                   return static_cast<char>(std::tolower(c));
                 });
  if (trajectory_semantics != "continuous" &&
      trajectory_semantics != "remd_state") {
    TitanUI::print_error(
        "-trajectory-semantics must be continuous or remd-state.");
    exit(EXIT_FAILURE);
  }
  if (mbar.flag_mbar) {
    std::transform(mbar.domain.begin(), mbar.domain.end(), mbar.domain.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::transform(mbar.ensemble.begin(), mbar.ensemble.end(),
                   mbar.ensemble.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::transform(mbar.window_match.begin(), mbar.window_match.end(),
                   mbar.window_match.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::transform(mbar.temp_mode.begin(), mbar.temp_mode.end(),
                   mbar.temp_mode.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    flag_network = true;
    if (mbar.domain == "tica" || mbar.domain == "both") {
      msm.flag_tica = true;
      msm.flag_msm = true;
    }
    if (mbar.domain == "pca" || mbar.domain == "both") {
      flag_pca = true;
    }
  }
  if ((msm.flag_msm || dham.flag_dham ||
       (mbar.flag_mbar && (mbar.domain == "tica" || mbar.domain == "both"))) &&
      msm.msm_lag <= 0) {
    TitanUI::print_error("MSM/DHAM/MBAR: -msm-lag must be > 0.");
    exit(EXIT_FAILURE);
  }
  if (dham.flag_dham) {
    if (!dham.segment_map_file.empty() && !dham.ener_list_file.empty()) {
      TitanUI::print_error("DHAM: -dham-segment-map and -dham-ener are mutually exclusive.");
      exit(EXIT_FAILURE);
    }
    std::transform(dham.ensemble.begin(), dham.ensemble.end(),
                   dham.ensemble.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::transform(dham.window_match.begin(), dham.window_match.end(),
                   dham.window_match.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (dham.ensemble != "nvt" && dham.ensemble != "npt") {
      TitanUI::print_error("DHAM: -dham-ensemble must be either nvt or npt.");
      exit(EXIT_FAILURE);
    }
    if (dham.window_match != "auto" && dham.window_match != "off" &&
        dham.window_match != "strict") {
      TitanUI::print_error(
          "DHAM: -dham-window-match must be auto, off, or strict.");
      exit(EXIT_FAILURE);
    }
    if (dham.target_temp <= 0.0) {
      TitanUI::print_error("DHAM: -dham-target-temp must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (dham.target_pressure_bar < 0.0) {
      TitanUI::print_error("DHAM: -dham-pressure-bar must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (dham.max_iter <= 0) {
      TitanUI::print_error("DHAM: --dham-max-iter must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (dham.tolerance <= 0.0) {
      TitanUI::print_error("DHAM: --dham-tol must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (dham.dt_energy_ps < 0.0) {
      TitanUI::print_error("DHAM: -dham-dt-energy must be >= 0.");
      exit(EXIT_FAILURE);
    }
  }
  if (mbar.flag_mbar) {
    if (mbar.domain != "tica" && mbar.domain != "pca" &&
        mbar.domain != "both") {
      TitanUI::print_error("MBAR: -mbar-domain must be tica, pca, or both.");
      exit(EXIT_FAILURE);
    }
    if (mbar.ensemble != "nvt" && mbar.ensemble != "npt") {
      TitanUI::print_error("MBAR: -mbar-ensemble must be either nvt or npt.");
      exit(EXIT_FAILURE);
    }
    if (mbar.window_match != "auto" && mbar.window_match != "off" &&
        mbar.window_match != "strict") {
      TitanUI::print_error(
          "MBAR: -mbar-window-match must be auto, off, or strict.");
      exit(EXIT_FAILURE);
    }
    if (mbar.temp_mode != "fixed" && mbar.temp_mode != "per_frame") {
      TitanUI::print_error(
          "MBAR: -mbar-temp-mode must be fixed or per_frame.");
      exit(EXIT_FAILURE);
    }
    if (mbar.temps_file.empty() && mbar.temp_mode != "per_frame") {
      TitanUI::print_error(
          "MBAR: omitting -mbar-temps is only supported with "
          "-mbar-temp-mode per_frame.");
      exit(EXIT_FAILURE);
    }
    if (mbar.target_temp <= 0.0) {
      TitanUI::print_error("MBAR: -mbar-target-temp must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (mbar.target_pressure_bar < 0.0) {
      TitanUI::print_error("MBAR: -mbar-pressure-bar must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (mbar.dt_energy_ps < 0.0) {
      TitanUI::print_error("MBAR: -mbar-dt-energy must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (mbar.temp_column < 1) {
      TitanUI::print_error("MBAR: -mbar-temp-col must be >= 1.");
      exit(EXIT_FAILURE);
    }
    if (mbar.temp_mode == "per_frame" && mbar.demux_map_file.empty() &&
        mbar.temp_column < 2) {
      TitanUI::print_error(
          "MBAR: raw per-frame mode expects a 3-column XVG layout "
          "(time, observable, temperature), so -mbar-temp-col must be >= 2.");
      exit(EXIT_FAILURE);
    }
    if (mbar.bootstrap_samples < 0) {
      TitanUI::print_error("MBAR: -mbar-bootstrap must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (mbar.scan_points < 0) {
      TitanUI::print_error("MBAR: -mbar-scan-points must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (!mbar.demux_map_file.empty() && mbar.temp_mode != "per_frame") {
      TitanUI::print_info(
          "MBAR: -mbar-demux-map is only used with -mbar-temp-mode per_frame. "
          "Fixed-temperature mode will ignore it.");
    }
  }
  if (tpt.flag_tpt) {
    std::transform(tpt.mode.begin(), tpt.mode.end(), tpt.mode.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (tpt.mode != "ensemble" && tpt.mode != "kinetic") {
      TitanUI::print_error(
          "TPT: -tpt-mode must be either ensemble or kinetic.");
      exit(EXIT_FAILURE);
    }
    if (tpt.source_spec.empty()) {
      TitanUI::print_error(
          "TPT: -tpt-source is required when -tpt is enabled.");
      exit(EXIT_FAILURE);
    }
    if (tpt.sink_spec.empty()) {
      TitanUI::print_error(
          "TPT: -tpt-sink is required when -tpt is enabled.");
      exit(EXIT_FAILURE);
    }
    if (tpt.top_paths <= 0) {
      TitanUI::print_error("TPT: -tpt-top-paths must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (tpt.flux_cut < 0.0) {
      TitanUI::print_error("TPT: -tpt-flux-cut must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (tpt.bootstrap_samples < 0) {
      TitanUI::print_error("TPT: -tpt-bootstrap must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (tpt.use_dham && !dham.flag_dham) {
      TitanUI::print_info(
          "TPT: -tpt-use-dham requested without -dham. TPT will fall back to MSM.");
    }
    if (!tpt.demux_map_file.empty() && tpt.mode != "kinetic") {
      TitanUI::print_info(
          "TPT: -tpt-demux-map is only used in kinetic mode. Ensemble TPT will ignore it.");
    }

    auto basin_prefix = [](const std::string &spec) {
      const auto colon_pos = spec.find(':');
      if (colon_pos == std::string::npos)
        return std::string("micro");
      std::string prefix = spec.substr(0, colon_pos);
      std::transform(prefix.begin(), prefix.end(), prefix.begin(),
                     [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                     });
      return prefix;
    };
    const std::string source_prefix = basin_prefix(tpt.source_spec);
    const std::string sink_prefix = basin_prefix(tpt.sink_spec);
    const auto is_qnative_prefix = [](const std::string &prefix) {
      return prefix == "qnative_range" || prefix == "qnative_lt" ||
             prefix == "qnative_gt";
    };
    if (is_qnative_prefix(source_prefix) || is_qnative_prefix(sink_prefix))
      qnative.flag_qnative = true;
    const auto prefix_supported = [](const std::string &prefix) {
      return prefix == "micro" || prefix == "state" || prefix == "states" ||
             prefix == "macro" || prefix == "tica_rect" || prefix == "lj" ||
             prefix == "ljpreset" || prefix == "lj_preset" ||
             prefix == "protein" || prefix == "prot" ||
             prefix == "protein_preset" || prefix == "qnative_range" ||
             prefix == "qnative_lt" || prefix == "qnative_gt";
    };
    if (!prefix_supported(source_prefix)) {
      TitanUI::print_error("TPT: unsupported -tpt-source prefix '" +
                           source_prefix + "'.");
      exit(EXIT_FAILURE);
    }
    if (!prefix_supported(sink_prefix)) {
      TitanUI::print_error("TPT: unsupported -tpt-sink prefix '" +
                           sink_prefix + "'.");
      exit(EXIT_FAILURE);
    }
    if (source_prefix == "micro" || source_prefix == "state" ||
        source_prefix == "states") {
      std::vector<int> source_states;
      std::string parse_error;
      if (!TitanTPT::parse_state_spec(tpt.source_spec, -1, source_states,
                                      &parse_error)) {
        TitanUI::print_error("TPT: invalid -tpt-source: " + parse_error);
        exit(EXIT_FAILURE);
      }
    }
    if (sink_prefix == "micro" || sink_prefix == "state" ||
        sink_prefix == "states") {
      std::vector<int> sink_states;
      std::string parse_error;
      if (!TitanTPT::parse_state_spec(tpt.sink_spec, -1, sink_states,
                                      &parse_error)) {
        TitanUI::print_error("TPT: invalid -tpt-sink: " + parse_error);
        exit(EXIT_FAILURE);
      }
    }
    auto validate_qnative_basin = [&](const std::string &spec,
                                      const std::string &prefix,
                                      const char *flag_name) {
      if (!is_qnative_prefix(prefix))
        return;
      const auto colon_pos = spec.find(':');
      if (colon_pos == std::string::npos) {
        TitanUI::print_error(std::string("TPT: ") + flag_name +
                             " qnative selector requires ':' payload.");
        exit(EXIT_FAILURE);
      }
      const std::string body = spec.substr(colon_pos + 1);
      auto parse_unit = [&](const std::string &raw, const char *what) {
        char *end_ptr = nullptr;
        const double value = std::strtod(raw.c_str(), &end_ptr);
        if (end_ptr == raw.c_str() || *end_ptr != '\0' ||
            !std::isfinite(value) || value < 0.0 || value > 1.0) {
          TitanUI::print_error(std::string("TPT: ") + flag_name + " " + what +
                               " must be a finite value in [0,1].");
          exit(EXIT_FAILURE);
        }
        return value;
      };
      if (prefix == "qnative_range") {
        const auto comma = body.find(',');
        if (comma == std::string::npos || body.find(',', comma + 1) != std::string::npos) {
          TitanUI::print_error(std::string("TPT: ") + flag_name +
                               " qnative_range expects min,max.");
          exit(EXIT_FAILURE);
        }
        const double q_min = parse_unit(body.substr(0, comma), "range minimum");
        const double q_max = parse_unit(body.substr(comma + 1), "range maximum");
        if (q_min > q_max) {
          TitanUI::print_error(std::string("TPT: ") + flag_name +
                               " qnative_range minimum must be <= maximum.");
          exit(EXIT_FAILURE);
        }
      } else {
        parse_unit(body, "threshold");
      }
    };
    validate_qnative_basin(tpt.source_spec, source_prefix, "-tpt-source");
    validate_qnative_basin(tpt.sink_spec, sink_prefix, "-tpt-sink");
    if ((source_prefix == "micro" || source_prefix == "state" ||
         source_prefix == "states") &&
        (sink_prefix == "micro" || sink_prefix == "state" ||
         sink_prefix == "states")) {
      std::vector<int> source_states;
      std::vector<int> sink_states;
      std::string parse_error;
      TitanTPT::parse_state_spec(tpt.source_spec, -1, source_states,
                                 &parse_error);
      TitanTPT::parse_state_spec(tpt.sink_spec, -1, sink_states,
                                 &parse_error);
      std::vector<unsigned char> basin_mask;
      int max_state = -1;
      for (int state : source_states)
        max_state = std::max(max_state, state);
      for (int state : sink_states)
        max_state = std::max(max_state, state);
      basin_mask.assign(static_cast<size_t>(max_state + 1), 0);
      for (int state : source_states)
        basin_mask[static_cast<size_t>(state)] = 1;
      for (int state : sink_states) {
        if (basin_mask[static_cast<size_t>(state)] != 0) {
          TitanUI::print_error(
              "TPT: source and sink basins must be disjoint.");
          exit(EXIT_FAILURE);
        }
      }
    }
  }
  if (ifp.pbctopo_certified_only) {
    if (!ifp.flag_ifp) {
      TitanUI::print_error(
          "IFP: -ifp-pbctopo-certified-only requires -ifp.");
      exit(EXIT_FAILURE);
    }
    if (!pbc.solv_pbctopo) {
      TitanUI::print_error(
          "IFP: -ifp-pbctopo-certified-only requires -pbctopo=true.");
      exit(EXIT_FAILURE);
    }
  }
  if (awh.flag_awh_ifp) {
    if (!ifp.flag_ifp) {
      TitanUI::print_error(
          "AWH: -awh-ifp requires IFP to be enabled.");
      exit(EXIT_FAILURE);
    }
    if (awh.xvg_list_file.empty()) {
      TitanUI::print_error(
          "AWH: -awh-xvg <filelist> is required when -awh-ifp is enabled.");
      exit(EXIT_FAILURE);
    }
    if (awh.xvg_value_column < 1) {
      TitanUI::print_error("AWH: -awh-xvg-col must be >= 1.");
      exit(EXIT_FAILURE);
    }
    if (awh.n_bins <= 0) {
      TitanUI::print_error("AWH: -awh-bins must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (flag_nocal) {
      TitanUI::print_error(
          "AWH: -awh-ifp currently requires two-group analysis and does not support -nocal.");
      exit(EXIT_FAILURE);
    }
    if (awh.output_prefix.empty()) {
      awh.output_prefix = "awh";
    }
  }

  if (community.flag_community) {
    std::transform(community.method.begin(), community.method.end(),
                   community.method.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                    });
    if (community.method != "louvain") {
      TitanUI::print_error("Community: unsupported method '" +
                           community.method +
                           "'. V1 supports only -community=louvain.");
      exit(EXIT_FAILURE);
    }
  }
  if (graph.flag_graph || community.flag_community) {
    graph.flag_graph = true;
    if (graph.filter_topk < 0) {
      TitanUI::print_error("Graph: -graph-filter-topk must be >= 0.");
      exit(EXIT_FAILURE);
    }

    std::transform(graph.community_method.begin(), graph.community_method.end(),
                   graph.community_method.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (graph.community_method.empty())
      graph.community_method = "auto";
    if (graph.community_method != "auto" && graph.community_method != "off" &&
        graph.community_method != "louvain") {
      TitanUI::print_error(
          "Graph: -graph-community must be one of auto, off, louvain.");
      exit(EXIT_FAILURE);
    }
    if (community.flag_community && graph.community_method == "auto")
      graph.community_method = community.method;

    std::transform(graph.filter_component.begin(), graph.filter_component.end(),
                   graph.filter_component.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (graph.filter_component.empty())
      graph.filter_component = "all";
    if (graph.filter_component != "all" &&
        graph.filter_component != "largest") {
      TitanUI::print_error(
          "Graph: -graph-filter-component must be one of all or largest.");
      exit(EXIT_FAILURE);
    }

    std::vector<std::string> normalized_sources;
    normalized_sources.reserve(graph.sources.size());
    bool use_auto_sources = graph.sources.empty();
    for (const std::string &raw_source : graph.sources) {
      std::string source = raw_source;
      std::transform(source.begin(), source.end(), source.begin(),
                     [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                     });
      if (source.empty())
        continue;
      if (source == "auto") {
        use_auto_sources = true;
        continue;
      }
      if (source == "network")
        source = "dccm";
      else if (source == "allostery_gc" || source == "allostery-gc-graph")
        source = "allostery-gc";
      else if (source == "allostery_te" || source == "allostery-te-graph")
        source = "allostery-te";
      if (source != "dccm" && source != "lspmd" && source != "allostery-gc" &&
          source != "allostery-te" && source != "tpt") {
        TitanUI::print_error(
            "Graph: -graph-source must be one of dccm, lspmd, allostery-gc, allostery-te, tpt.");
        exit(EXIT_FAILURE);
      }
      if (std::find(normalized_sources.begin(), normalized_sources.end(),
                    source) == normalized_sources.end()) {
        normalized_sources.push_back(source);
      }
    }

    if (!use_auto_sources && community.flag_community &&
        std::find(normalized_sources.begin(), normalized_sources.end(),
                  "dccm") == normalized_sources.end()) {
      normalized_sources.insert(normalized_sources.begin(), "dccm");
    }

    if (use_auto_sources || normalized_sources.empty()) {
      graph.sources = {"auto"};
    } else {
      graph.sources = std::move(normalized_sources);
    }

    const auto graph_has_source = [&](const char *name) {
      return std::find(graph.sources.begin(), graph.sources.end(), name) !=
             graph.sources.end();
    };
    const bool graph_auto_sources =
        graph.sources.size() == 1 && graph.sources.front() == "auto";

    if (!graph_auto_sources) {
      if (graph_has_source("dccm")) {
        flag_network = true;
        flag_network_explicit = true;
      }
      if (graph_has_source("allostery-gc") || graph_has_source("allostery-te")) {
        flag_allostery = true;
        flag_network = true;
      }
      if (graph_has_source("lspmd")) {
        lspmd.flag_lspmd = true;
      }
      if (graph_has_source("tpt")) {
        tpt.flag_tpt = true;
      }
    }

    const bool dccm_source_active =
        flag_network_explicit || flag_pca || flag_allostery ||
        corr_knn.flag_corr_knn || community.flag_community;
    const bool allostery_source_active = flag_allostery;
    const bool lspmd_source_active = lspmd.flag_lspmd;
    const bool tpt_source_active = tpt.flag_tpt;
    if (graph_auto_sources && !dccm_source_active && !allostery_source_active &&
        !lspmd_source_active && !tpt_source_active) {
      TitanUI::print_error(
          "Graph: -graph with auto sources requires at least one active source producer (dccm/network, allostery, lspmd, or tpt).");
      exit(EXIT_FAILURE);
    }
  }
  if (cmap.legacy_alias_seen && cmap.new_option_explicit) {
    TitanUI::print_error(
        "CMap: do not mix deprecated -contact-map with -cmap options; migrate to -cmap -cmap-scope=interface -cmap-mode=sidechain-com.");
    exit(EXIT_FAILURE);
  }
  std::transform(feature_export.source.begin(), feature_export.source.end(),
                 feature_export.source.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (feature_export.source != "auto" && feature_export.source != "dccm" &&
      feature_export.source != "lj" && feature_export.source != "dihedral" &&
      feature_export.source != "qnative" && feature_export.source != "cmap" &&
      feature_export.source != "mixed") {
    TitanUI::print_error(
        "Feature export: -feature-source must be one of auto, dccm, lj, dihedral, qnative, cmap, mixed.");
    exit(EXIT_FAILURE);
  }
  if (feature_export.source_explicit)
    feature_export.export_features = true;
  if (feature_export.source == "dihedral" || feature_export.source == "mixed")
    dihedral.flag_dihedral = true;
  if (feature_export.source == "qnative" || feature_export.source == "mixed")
    qnative.flag_qnative = true;
  if (feature_export.source == "cmap")
    cmap.flag_cmap = true;
  std::transform(perf_report.mode.begin(), perf_report.mode.end(),
                 perf_report.mode.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (perf_report.mode != "off" && perf_report.mode != "summary" &&
      perf_report.mode != "detail") {
    TitanUI::print_error(
        "Performance report: -perf-report must be one of off, summary, detail.");
    exit(EXIT_FAILURE);
  }
  std::transform(uncertainty.mode.begin(), uncertainty.mode.end(),
                 uncertainty.mode.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (uncertainty.mode != "off" && uncertainty.mode != "block" &&
      uncertainty.mode != "bootstrap") {
    TitanUI::print_error(
        "Uncertainty: -uncertainty must be one of off, block, bootstrap.");
    exit(EXIT_FAILURE);
  }
  std::transform(uncertainty.blocks.begin(), uncertainty.blocks.end(),
                 uncertainty.blocks.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (uncertainty.blocks != "auto") {
    try {
      const int n_blocks = std::stoi(uncertainty.blocks);
      if (n_blocks < 1) {
        TitanUI::print_error(
            "Uncertainty: -uncertainty-blocks must be auto or a positive integer.");
        exit(EXIT_FAILURE);
      }
    } catch (...) {
      TitanUI::print_error(
          "Uncertainty: -uncertainty-blocks must be auto or a positive integer.");
      exit(EXIT_FAILURE);
    }
  }
  if (uncertainty.bootstrap_samples < 1) {
    TitanUI::print_error(
        "Uncertainty: -uncertainty-bootstrap must be a positive integer.");
    exit(EXIT_FAILURE);
  }
  std::transform(output_manifest.schema_export.begin(),
                 output_manifest.schema_export.end(),
                 output_manifest.schema_export.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (output_manifest.schema_export != "off" &&
      output_manifest.schema_export != "json") {
    TitanUI::print_error(
        "Output manifest: -schema-export must be one of off, json.");
    exit(EXIT_FAILURE);
  }
  if (output_manifest.dry_run)
    output_manifest.output_manifest = true;
  if (!segments.config_file.empty()) {
    try {
      const titan_segments::Config segment_config =
          titan_segments::load_config(segments.config_file);
      interface_analysis.uses_mol2 = false;
      if (interface_analysis.flag_interface_analysis) {
        const titan_segments::Entity *core_a =
            segment_config.entity_for_role("core_a");
        const titan_segments::Entity *core_b =
            segment_config.entity_for_role("core_b");
        if (core_a == nullptr || core_b == nullptr) {
          TitanUI::print_error(
              "Interface analysis: the segment config must define roles "
              "core_a and core_b.");
          exit(EXIT_FAILURE);
        }
        const titan_segments::Entity *partner =
            segment_config.entity_for_role("partner");
        interface_analysis.uses_mol2 =
            core_a->molecule == 2 || core_b->molecule == 2 ||
            (partner != nullptr && partner->molecule == 2);
      }
    } catch (const std::exception &e) {
      TitanUI::print_error(e.what());
      exit(EXIT_FAILURE);
    }
  } else if (interface_analysis.flag_interface_analysis) {
    TitanUI::print_error(
        "Interface analysis: -interface-analysis requires "
        "-segment-config or -interface-config.");
    exit(EXIT_FAILURE);
  }
  if (interface_analysis.flag_interface_analysis) {
    std::transform(interface_analysis.contact_mode.begin(),
                   interface_analysis.contact_mode.end(),
                   interface_analysis.contact_mode.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::replace(interface_analysis.contact_mode.begin(),
                 interface_analysis.contact_mode.end(), '_', '-');
    if (interface_analysis.contact_mode == "sidechain")
      interface_analysis.contact_mode = "sidechain-com";
    if (interface_analysis.contact_mode == "backbone")
      interface_analysis.contact_mode = "bb";
    if (interface_analysis.contact_mode != "sidechain-com" &&
        interface_analysis.contact_mode != "sidechain-heavy" &&
        interface_analysis.contact_mode != "cb" &&
        interface_analysis.contact_mode != "ca" &&
        interface_analysis.contact_mode != "bb") {
      TitanUI::print_error(
          "Interface analysis: -interface-contact-mode must be one of "
          "sidechain-com, sidechain-heavy, cb, ca, bb.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(interface_analysis.cutoff_ang) ||
        interface_analysis.cutoff_ang <= 0.0) {
      TitanUI::print_error("Interface analysis: -interface-cut must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (interface_analysis.state_min_contacts < 1) {
      TitanUI::print_error(
          "Interface analysis: -interface-state-min-contacts must be >= 1.");
      exit(EXIT_FAILURE);
    }
    if (interface_analysis.sasa_stride < 0) {
      TitanUI::print_error(
          "Interface analysis: -interface-sasa-stride must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (interface_analysis.sasa_points < 12) {
      TitanUI::print_error(
          "Interface analysis: -interface-sasa-points must be >= 12.");
      exit(EXIT_FAILURE);
    }
  }
  if (cmap.legacy_alias_enabled) {
    cmap.flag_cmap = true;
    cmap.mode = "sidechain-com";
    cmap.scope = "interface";
    if (!cmap.cutoff_explicit) {
      cmap.cutoff_ang = d_standard;
      cmap.cutoff_explicit = true;
    }
    TitanUI::print_warning(
        "Deprecated: -contact-map=yes now maps to -cmap -cmap-scope=interface -cmap-mode=sidechain-com; use -dsc for alias cutoff, or migrate fully to -cmap-cut.");
  }
  if (cmap.flag_cmap) {
    std::transform(cmap.mode.begin(), cmap.mode.end(), cmap.mode.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::replace(cmap.mode.begin(), cmap.mode.end(), '_', '-');
    if (cmap.mode == "sidechain")
      cmap.mode = "sidechain-com";
    if (cmap.mode == "backbone")
      cmap.mode = "bb";
    if (cmap.mode != "sidechain-com" && cmap.mode != "sidechain-heavy" &&
        cmap.mode != "cb" &&
        cmap.mode != "ca" && cmap.mode != "bb") {
      TitanUI::print_error(
          "CMap: -cmap-mode must be one of sidechain-com, sidechain-heavy, cb, ca, bb.");
      exit(EXIT_FAILURE);
    }
    if (!cmap.cutoff_explicit)
      cmap.cutoff_ang = cmap.mode == "sidechain-com" ? 6.5 : 4.5;
    if (!std::isfinite(cmap.cutoff_ang) || cmap.cutoff_ang <= 0.0) {
      TitanUI::print_error("CMap: -cmap-cut must be > 0.");
      exit(EXIT_FAILURE);
    }
    std::transform(cmap.domain.begin(), cmap.domain.end(), cmap.domain.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (cmap.domain != "intra" && cmap.domain != "inter" &&
        cmap.domain != "both") {
      TitanUI::print_error(
          "CMap: -cmap-domain must be one of intra, inter, both.");
      exit(EXIT_FAILURE);
    }
    if (cmap.min_seqsep < 0) {
      TitanUI::print_error("CMap: -cmap-min-seqsep must be >= 0.");
      exit(EXIT_FAILURE);
    }
    std::transform(cmap.scope.begin(), cmap.scope.end(), cmap.scope.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (cmap.scope != "mol1" && cmap.scope != "interface") {
      TitanUI::print_error(
          "CMap: -cmap-scope must be one of mol1, interface.");
      exit(EXIT_FAILURE);
    }
    if (cmap.scope == "interface" &&
        (cmap.domain_explicit || cmap.min_seqsep_explicit)) {
      TitanUI::print_error(
          "CMap: -cmap-domain and -cmap-min-seqsep apply only to -cmap-scope=mol1.");
      exit(EXIT_FAILURE);
    }
  }
  if (qnative.flag_qnative) {
    std::transform(qnative.mode.begin(), qnative.mode.end(),
                   qnative.mode.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::replace(qnative.mode.begin(), qnative.mode.end(), '_', '-');
    if (qnative.mode == "sidechain")
      qnative.mode = "sidechain-com";
    if (qnative.mode == "backbone")
      qnative.mode = "bb";
    if (qnative.mode != "sidechain-com" &&
        qnative.mode != "sidechain-heavy" && qnative.mode != "cb" &&
        qnative.mode != "ca" && qnative.mode != "bb") {
      TitanUI::print_error(
          "QNative: -qnative-mode must be one of sidechain-com, sidechain-heavy, cb, ca, bb.");
      exit(EXIT_FAILURE);
    }
    if (!qnative.cutoff_explicit)
      qnative.cutoff_ang = qnative.mode == "sidechain-com" ? 6.5 : 4.5;
    if (!std::isfinite(qnative.cutoff_ang) || qnative.cutoff_ang <= 0.0) {
      TitanUI::print_error("QNative: -qnative-cut must be > 0.");
      exit(EXIT_FAILURE);
    }
    std::transform(qnative.domain.begin(), qnative.domain.end(),
                   qnative.domain.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (qnative.domain != "intra" && qnative.domain != "inter" &&
        qnative.domain != "both") {
      TitanUI::print_error(
          "QNative: -qnative-domain must be one of intra, inter, both.");
      exit(EXIT_FAILURE);
    }
    if (qnative.min_seqsep < 0) {
      TitanUI::print_error("QNative: -qnative-min-seqsep must be >= 0.");
      exit(EXIT_FAILURE);
    }
    std::transform(qnative.scope.begin(), qnative.scope.end(),
                   qnative.scope.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (qnative.scope != "mol1" && qnative.scope != "interface") {
      TitanUI::print_error(
          "QNative: -qnative-scope must be one of mol1, interface.");
      exit(EXIT_FAILURE);
    }
    if (qnative.scope == "interface" &&
        (qnative.domain_explicit || qnative.min_seqsep_explicit)) {
      TitanUI::print_error(
          "QNative: -qnative-domain and -qnative-min-seqsep apply only to -qnative-scope=mol1.");
      exit(EXIT_FAILURE);
    }
    if (qnative.beta_explicit && !qnative.soft) {
      TitanUI::print_error(
          "QNative: -qnative-beta is valid only with -qnative-soft=true.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(qnative.beta) || qnative.beta <= 0.0) {
      TitanUI::print_error("QNative: -qnative-beta must be > 0.");
      exit(EXIT_FAILURE);
    }
  }
  if (dihedral.flag_dihedral) {
    std::transform(dihedral.kind.begin(), dihedral.kind.end(),
                   dihedral.kind.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::replace(dihedral.kind.begin(), dihedral.kind.end(), '_', '-');
    if (dihedral.kind != "backbone" && dihedral.kind != "sidechain" &&
        dihedral.kind != "both") {
      TitanUI::print_error(
          "Dihedral: -dihedral-kind must be backbone, sidechain, or both.");
      exit(EXIT_FAILURE);
    }
    if (dihedral.stride < 1) {
      TitanUI::print_error("Dihedral: -dihedral-stride must be >= 1.");
      exit(EXIT_FAILURE);
    }
    if (dihedral.residues.empty()) {
      TitanUI::print_error(
          "Dihedral: -dihedral-residues must be 'all' or a residue list.");
      exit(EXIT_FAILURE);
    }
  }
  if (msd.flag_msd) {
    std::transform(msd.selection.begin(), msd.selection.end(),
                   msd.selection.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::transform(msd.mode.begin(), msd.mode.end(), msd.mode.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::transform(msd.maxlag.begin(), msd.maxlag.end(), msd.maxlag.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::transform(msd.fit_window.begin(), msd.fit_window.end(),
                   msd.fit_window.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::transform(msd.pbc.begin(), msd.pbc.end(), msd.pbc.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::transform(msd.rot_fit_window.begin(), msd.rot_fit_window.end(),
                   msd.rot_fit_window.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (msd.selection != "mol1" && msd.selection != "mol2" &&
        msd.selection != "trace" && msd.selection != "backbone" &&
        msd.selection.rfind("atom:", 0) != 0 &&
        msd.selection.rfind("residue:", 0) != 0) {
      TitanUI::print_error(
          "MSD: -msd-selection must be mol1, mol2, trace, backbone, atom:<names>, or residue:<list>.");
      exit(EXIT_FAILURE);
    }
    if (msd.mode != "com" && msd.mode != "atom") {
      TitanUI::print_error("MSD: -msd-mode must be com or atom.");
      exit(EXIT_FAILURE);
    }
    if (msd.maxlag != "auto") {
      try {
        if (std::stoi(msd.maxlag) < 1)
          throw std::runtime_error("bad");
      } catch (...) {
        TitanUI::print_error("MSD: -msd-maxlag must be auto or an integer >= 1.");
        exit(EXIT_FAILURE);
      }
    }
    if (msd.pbc != "auto" && msd.pbc != "unwrap" && msd.pbc != "mic") {
      TitanUI::print_error("MSD: -msd-pbc must be auto, unwrap, or mic.");
      exit(EXIT_FAILURE);
    }
    if (msd.rot_fit_window != "auto") {
      const auto comma = msd.rot_fit_window.find(',');
      try {
        if (comma == std::string::npos ||
            !(std::stod(msd.rot_fit_window.substr(comma + 1)) >
              std::stod(msd.rot_fit_window.substr(0, comma))))
          throw std::runtime_error("bad");
      } catch (...) {
        TitanUI::print_error(
            "MSD: -msd-rot-fit-window must be auto or tmin,tmax with tmax > tmin.");
        exit(EXIT_FAILURE);
      }
    }
  }
  if (msm.flag_msm || dham.flag_dham) {
    if (msm.msm_nts <= 0) {
      TitanUI::print_error("MSM: -msm-nts must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (msm.msm_its_topk < 0) {
      TitanUI::print_error("MSM: -msm-its-topk must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (msm.msm_its_max_k_full <= 0) {
      TitanUI::print_error("MSM: -msm-its-max-k-full must be > 0.");
      exit(EXIT_FAILURE);
    }
  }
  if (flag_pca) {
    if (msm.pca_topk < 0) {
      TitanUI::print_error("PCA: -pca-topk must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (msm.pca_max_k_full <= 0) {
      TitanUI::print_error("PCA: -pca-max-k-full must be > 0.");
      exit(EXIT_FAILURE);
    }
  }
  if (msm.flag_tica) {
    if (msm.tica_lag <= 0) {
      TitanUI::print_error("TICA: -tica-lag must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (msm.tica_components <= 0) {
      TitanUI::print_error("TICA: -tica-components must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (msm.tica_rank <= 0) {
      TitanUI::print_error("TICA: -tica-rank must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (msm.tica_rank < msm.tica_components) {
      TitanUI::print_error("TICA: -tica-rank must be >= -tica-components.");
      exit(EXIT_FAILURE);
    }
    std::string backend = msm.tica_backend;
    std::transform(backend.begin(), backend.end(), backend.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (backend != "auto" && backend != "exact_dense" && backend != "dense" &&
        backend != "exact_dual" && backend != "dual" &&
        backend != "randomized" && backend != "rand") {
      TitanUI::print_error(
          "TICA: -tica-backend must be one of auto, exact_dense, exact_dual, randomized.");
      exit(EXIT_FAILURE);
    }
    if (backend == "dense") {
      backend = "exact_dense";
    } else if (backend == "dual") {
      backend = "exact_dual";
    } else if (backend == "rand") {
      backend = "randomized";
    }
    const bool need_tica_projection_cache =
        msm.flag_msm || dham.flag_dham || umap.flag_umap ||
        (zts.flag_zts && zts.zts_domain == "tica");
    if (need_tica_projection_cache && backend == "exact_dense") {
      TitanUI::print_warning(
          "TICA: exact_dense does not retain per-frame projection history "
          "needed by MSM/DHAM/UMAP/ZTS. Switching backend to exact_dual.");
      backend = "exact_dual";
    }
    msm.tica_backend = backend;
  }
  if (zts.flag_zts) {
    std::transform(zts.zts_domain.begin(), zts.zts_domain.end(),
                   zts.zts_domain.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    std::transform(zts.zts_init_mode.begin(), zts.zts_init_mode.end(),
                   zts.zts_init_mode.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (flag_pca && msm.flag_tica && !zts.zts_domain_explicit) {
      TitanUI::print_error(
          "ZTS: when both -pca and -tica are enabled, -zts-domain is required.");
      exit(EXIT_FAILURE);
    }
    if (zts.zts_domain != "pca" && zts.zts_domain != "tica") {
      TitanUI::print_error("ZTS: -zts-domain must be either pca or tica.");
      exit(EXIT_FAILURE);
    }
    if (zts.zts_domain == "pca" && !flag_pca) {
      TitanUI::print_error("ZTS: -zts-domain pca requires -pca.");
      exit(EXIT_FAILURE);
    }
    if (zts.zts_domain == "tica" && !msm.flag_tica) {
      TitanUI::print_error("ZTS: -zts-domain tica requires -tica.");
      exit(EXIT_FAILURE);
    }
    if (zts.zts_images < 3) {
      TitanUI::print_error("ZTS: -zts-images must be >= 3.");
      exit(EXIT_FAILURE);
    }
    if (zts.zts_step <= 0.0) {
      TitanUI::print_error("ZTS: -zts-step must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (zts.zts_tol <= 0.0) {
      TitanUI::print_error("ZTS: -zts-tol must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (zts.zts_max_iter <= 0) {
      TitanUI::print_error("ZTS: -zts-iter must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (zts.zts_fes_sigma <= 0.0) {
      TitanUI::print_error("ZTS: -zts-sigma must be > 0.");
      exit(EXIT_FAILURE);
    }
    const bool source_explicit = zts.zts_source >= 0;
    const bool sink_explicit = zts.zts_sink >= 0;
    if (source_explicit != sink_explicit) {
      TitanUI::print_error(
          "ZTS: -zts-source and -zts-sink must be set together, or both left at auto.");
      exit(EXIT_FAILURE);
    }
    if (source_explicit && zts.zts_source == zts.zts_sink) {
      TitanUI::print_error(
          "ZTS: -zts-source and -zts-sink must refer to different basins.");
      exit(EXIT_FAILURE);
    }
    if (zts.zts_init_mode != "linear") {
      TitanUI::print_error(
          "ZTS: -zts-init currently supports only 'linear' in this build.");
      exit(EXIT_FAILURE);
    }
  }
  if (umap.flag_umap) {
    std::transform(umap.umap_knn_mode.begin(), umap.umap_knn_mode.end(),
                   umap.umap_knn_mode.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (umap.umap_neighbors < 2) {
      TitanUI::print_error("UMAP: -umap-neighbors must be >= 2.");
      exit(EXIT_FAILURE);
    }
    if (umap.umap_components != 2 && umap.umap_components != 3) {
      TitanUI::print_error("UMAP: -umap-components must be 2 or 3.");
      exit(EXIT_FAILURE);
    }
    if (umap.umap_epochs <= 0) {
      TitanUI::print_error("UMAP: -umap-epochs must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (umap.umap_min_dist <= 0.0) {
      TitanUI::print_error("UMAP: -umap-min-dist must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (umap.umap_spread <= 0.0) {
      TitanUI::print_error("UMAP: -umap-spread must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (umap.umap_knn_mode != "exact" && umap.umap_knn_mode != "approx" &&
        umap.umap_knn_mode != "auto") {
      TitanUI::print_error(
          "UMAP: -umap-knn-mode must be one of exact, approx, auto.");
      exit(EXIT_FAILURE);
    }
  }
  if (group_1 < 0 || group_2 < 0) {
    cout << "Groups for calculating energy are invalid! ";
    cout << "Your index group(s) must not be negative!\n";
    exit(EXIT_FAILURE);
  }
  if (de.flag_find_pulling_direct) {
    std::transform(de.pull_opt.begin(), de.pull_opt.end(), de.pull_opt.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (de.pull_opt.empty())
      de.pull_opt = "de";
    if (de.pull_opt == "de-pattern" || de.pull_opt == "de_pattern")
      de.pull_opt = "de+pattern";
    if (de.pull_opt == "de-jso-pattern" || de.pull_opt == "de_jso_pattern")
      de.pull_opt = "de-jso+pattern";
    if (de.pull_opt == "de-nl-shade-lbc-pattern" ||
        de.pull_opt == "de_nl_shade_lbc_pattern")
      de.pull_opt = "de-nl-shade-lbc+pattern";
    if (de.pull_opt == "direct-pattern" || de.pull_opt == "direct_pattern")
      de.pull_opt = "direct+pattern";
    if (de.pull_opt != "de" && de.pull_opt != "de+pattern" &&
        de.pull_opt != "de-shade" && de.pull_opt != "de-l-shade" &&
        de.pull_opt != "de-jso" && de.pull_opt != "de-jso+pattern" &&
        de.pull_opt != "de-nl-shade-lbc" &&
        de.pull_opt != "de-nl-shade-lbc+pattern" && de.pull_opt != "direct" &&
        de.pull_opt != "direct+pattern" && de.pull_opt != "corridor-grid") {
      TitanUI::print_error(
          "Pull-direction optimizer: supported values are de, de+pattern, de-shade, de-l-shade, de-jso, de-jso+pattern, de-nl-shade-lbc, de-nl-shade-lbc+pattern, direct, direct+pattern, and corridor-grid.");
      exit(EXIT_FAILURE);
    }
    if (de.pull_opt == "corridor-grid" &&
        (!de.flag_pathway || de.pathway_file.empty())) {
      TitanUI::print_error(
          "Pull-direction corridor-grid requires pathway mode with -pathway-file.");
      exit(EXIT_FAILURE);
    }
    const bool use_legacy_de =
        (de.pull_opt == "de" || de.pull_opt == "de+pattern");
    const bool use_shade_family =
        (de.pull_opt == "de-shade" || de.pull_opt == "de-l-shade" ||
         de.pull_opt == "de-jso" || de.pull_opt == "de-jso+pattern" ||
         de.pull_opt == "de-nl-shade-lbc" ||
         de.pull_opt == "de-nl-shade-lbc+pattern");
    const bool use_de_family = use_legacy_de || use_shade_family;
    const bool use_direct =
        (de.pull_opt == "direct" || de.pull_opt == "direct+pattern");
    if ((de.pull_opt == "de-jso" || de.pull_opt == "de-jso+pattern") &&
        !de.user_set_de_p_best_rate)
      de.de_p_best_rate = 0.11;
    if (use_de_family && de.de_pop_num < 4) {
      TitanUI::print_error(
          "Pull-direction DE: -de-pop-num must be >= 4.");
      exit(EXIT_FAILURE);
    }
    if (use_de_family && de.de_ite == 0) {
      TitanUI::print_error(
          "Pull-direction DE: -de-ite-num must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (de.de_cross_prob < 0.0 || de.de_cross_prob > 1.0) {
      TitanUI::print_error(
          "Pull-direction DE: -de-cross-prob must be in [0, 1].");
      exit(EXIT_FAILURE);
    }
    if (de.de_scan_radius <= 0.0 || de.de_min_scan_radius <= 0.0) {
      TitanUI::print_error(
          "Pull-direction DE: scan radii must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (de.de_scan_radius < de.de_min_scan_radius) {
      TitanUI::print_error(
          "Pull-direction DE: -de-scan-radius must be >= -de-min-scan-radius.");
      exit(EXIT_FAILURE);
    }
    if (use_de_family && de.de_stagnation_limit < 1) {
      TitanUI::print_error(
          "Pull-direction DE: -de-stagnation-limit must be >= 1.");
      exit(EXIT_FAILURE);
    }
    if (use_shade_family && de.de_memory_size == 0) {
      TitanUI::print_error(
          "Pull-direction SHADE-family: -de-memory-size must be >= 1.");
      exit(EXIT_FAILURE);
    }
    if (use_shade_family &&
        (de.de_p_best_rate <= 0.0 || de.de_p_best_rate > 1.0)) {
      TitanUI::print_error(
          "Pull-direction SHADE-family: -de-p-best-rate must be in (0, 1].");
      exit(EXIT_FAILURE);
    }
    if (use_shade_family && de.de_archive_scale < 0.0) {
      TitanUI::print_error(
          "Pull-direction SHADE-family: -de-archive-scale must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if ((de.pull_opt == "de-l-shade" || de.pull_opt == "de-jso" ||
         de.pull_opt == "de-jso+pattern" ||
         de.pull_opt == "de-nl-shade-lbc" ||
         de.pull_opt == "de-nl-shade-lbc+pattern") &&
        de.de_min_pop_num < 4) {
      TitanUI::print_error(
          "Pull-direction SHADE-family: -de-min-pop-num must be >= 4.");
      exit(EXIT_FAILURE);
    }
    if ((de.pull_opt == "de-l-shade" || de.pull_opt == "de-jso" ||
         de.pull_opt == "de-jso+pattern" ||
         de.pull_opt == "de-nl-shade-lbc" ||
         de.pull_opt == "de-nl-shade-lbc+pattern") &&
        de.de_min_pop_num > de.de_pop_num) {
      TitanUI::print_error(
          "Pull-direction SHADE-family: -de-min-pop-num must be <= -de-pop-num.");
      exit(EXIT_FAILURE);
    }
    if (use_de_family && de.pull_max_evals != 0 &&
        de.pull_max_evals < de.de_pop_num) {
      TitanUI::print_error(
          "Pull-direction DE: -pull-max-evals must be 0 or >= -de-pop-num.");
      exit(EXIT_FAILURE);
    }
    if (use_direct && de.pull_max_evals == 0 && de.de_ite == 0) {
      TitanUI::print_error(
          "Pull-direction DIRECT: provide -pull-max-evals or keep -de-ite-num > 0 for fallback budget.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway && de.dis_along_path <= 0.0) {
      TitanUI::print_error(
          "Pull-direction DE pathway: -de-pathway-radius must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway && !de.pathway_file.empty() &&
        de.user_set_pathway_group) {
      TitanUI::print_error(
          "TITAN-Path: use either -pathway-file or -de-pathway-group, not both.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway && !de.pathway_route_id.empty() &&
        de.pathway_file.empty()) {
      TitanUI::print_error(
          "TITAN-Path: -pathway-route-id requires -pathway-file.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway &&
        (!de.path_previous_status.empty() || de.path_continuity_window != 0.0) &&
        (de.path_previous_status.empty() || de.pathway_file.empty() ||
         !std::isfinite(de.path_continuity_window) || de.path_continuity_window <= 0.0)) {
      TitanUI::print_error(
          "TITAN-Path: continuity requires -pathway-file, -pathway-previous-status and a positive finite -pathway-continuity-window.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway &&
        (de.path_graph_edge_cutoff <= 0.0 ||
          de.path_recovery_enter_radius <= 0.0 ||
          de.path_recovery_exit_radius <= 0.0 ||
          de.path_recovery_max_turn_angle <= 0.0 ||
          de.path_recovery_max_turn_angle > 180.0 ||
         de.path_corridor_spacing <= 0.0 ||
         de.path_registration_max_rmsd <= 0.0 ||
         de.path_completion_tolerance <= 0.0 ||
          de.path_completion_min_separation < 0.0 ||
          de.path_max_turn_angle <= 0.0 || de.path_max_turn_angle > 180.0 ||
          de.path_energy_key_epsilon < 0.0 || de.path_min_clearance <= 0.0 ||
          de.path_probe_length < 0.0 ||
          !std::isfinite(de.path_navigation_tube_weight) ||
          de.path_navigation_tube_weight < 0.0 ||
          !std::isfinite(de.path_navigation_backtrack_weight) ||
          de.path_navigation_backtrack_weight < 0.0 ||
          !std::isfinite(de.path_navigation_cost_epsilon) ||
          de.path_navigation_cost_epsilon < 0.0 ||
          !std::isfinite(de.path_navigation_seed_amplitude) ||
          de.path_navigation_seed_amplitude < 0.0)) {
      TitanUI::print_error(
          "TITAN-Path: pathway cutoffs, radii, spacing, registration RMSD, completion tolerance, and minimum clearance must be positive; completion separation, probe length, and energy epsilon must be non-negative; advance and recovery max turn angles must be in (0,180].");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway && de.path_navigation_seed_radial_samples > 64) {
      TitanUI::print_error(
          "TITAN-Path: -pathway-navigation-seed-radial-samples must be <= 64.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway && de.path_navigation_enabled &&
        (de.pathway_file.empty() || de.path_probe_length <= 0.0)) {
      TitanUI::print_error(
          "TITAN-Path: navigation requires an explicit -pathway-file and a positive fixed probe length or ladder.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway && de.path_clearance_mode != "endpoint-center" &&
        de.path_clearance_mode != "swept-linear-center") {
      TitanUI::print_error(
          "TITAN-Path: -pathway-clearance-mode must be endpoint-center or swept-linear-center.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway &&
        de.path_clearance_mode == "swept-linear-center" &&
        de.path_probe_length <= 0.0) {
      TitanUI::print_error(
          "TITAN-Path: swept-linear-center clearance requires a positive fixed probe length or ladder.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway &&
        de.path_clearance_mode == "swept-linear-center" &&
        !pbc.solv_pbcn && !pbc.solv_pbctopo) {
      TitanUI::print_error(
          "TITAN-Path: swept-linear-center clearance requires -pbcn or -pbctopo so receptor and ligand coordinates are reconstructed into one upstream PBC-resolved Cartesian frame.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway && de.path_corridor_radial_samples > 4096) {
      TitanUI::print_error(
          "TITAN-Path: -pathway-corridor-radial-samples must be <= 4096.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway && de.path_probe_lengths.size() > 16) {
      TitanUI::print_error(
          "TITAN-Path: -pathway-probe-lengths supports at most 16 distances.");
      exit(EXIT_FAILURE);
    }
    if (de.flag_pathway && !de.path_probe_lengths.empty()) {
      for (std::size_t index = 0; index < de.path_probe_lengths.size(); ++index) {
        const double value = de.path_probe_lengths[index];
        if (!std::isfinite(value) || value <= 0.0 ||
            (index > 0 &&
             value >= de.path_probe_lengths[index - 1] - 1.0e-12)) {
          TitanUI::print_error(
              "TITAN-Path: -pathway-probe-lengths must be a strictly decreasing list of positive finite distances.");
          exit(EXIT_FAILURE);
        }
      }
    }
    if (de.flag_pathway &&
        de.path_recovery_exit_radius > de.path_recovery_enter_radius) {
      TitanUI::print_error(
          "TITAN-Path: -pathway-recovery-exit must be <= -pathway-recovery-enter.");
      exit(EXIT_FAILURE);
    }
    if (use_shade_family &&
        (de.user_set_de_diff_weight || de.user_set_de_cross_prob ||
         de.user_set_de_F || de.user_set_de_algorithm ||
         de.user_set_de_adaptive_F)) {
      TitanUI::print_error(
          "Pull-direction SHADE-family: legacy flags -de-diff-weight, -de-cross-prob, -de-F, -de-algorithm, and -de-adaptive-F are not supported.");
      exit(EXIT_FAILURE);
    }
    if ((use_legacy_de || use_direct) &&
        (de.user_set_de_memory_size || de.user_set_de_p_best_rate ||
         de.user_set_de_archive_scale || de.user_set_de_min_pop_num)) {
      TitanUI::print_error(
          "Pull-direction optimizer: SHADE-family flags -de-memory-size, -de-p-best-rate, -de-archive-scale, and -de-min-pop-num are only valid with de-shade, de-l-shade, de-jso, de-jso+pattern, de-nl-shade-lbc, or de-nl-shade-lbc+pattern.");
      exit(EXIT_FAILURE);
    }
  }
  if (entropy_quasi == true &&
      (group_entropy_com < 0 || group_entropy_rec < 0 ||
       group_entropy_lg < 0)) {
    cout << "Entropy-quasi option is invalid! ";
    cout << "Your entropy-quasi group(s) must not be negative!\n";
    exit(EXIT_FAILURE);
  }
  if (b_input < 0 || e_input < 0) {
    cout << "Begin/end input must be positive!\n";
    exit(EXIT_FAILURE);
  }
  if (b_input > e_input) {
    cout << "Begin input must be smaller than end!\n";
    exit(EXIT_FAILURE);
  }
  if (step_input <= 0) {
    cout << "Step input must be positive!\n";
    exit(EXIT_FAILURE);
  }
  if (nt == 0 || nt > std::thread::hardware_concurrency()) {
    nt = std::thread::hardware_concurrency();
    cout << "The number of threads is invalid. Will use " << nt
         << " threads!\n";
  }

  if (flag_cas && flag_nocal) {
    TitanUI::print_error(
        "CAS requires GBSA energy evaluation; -nocal=true cannot be combined "
        "with -cas=true.");
    exit(EXIT_FAILURE);
  }
  if (gist.flag_gist && flag_nocal) {
    TitanUI::print_error("GIST requires the standard analysis path; -gist cannot be combined with -nocal.");
    exit(EXIT_FAILURE);
  }
  if (flag_cas && !flag_gbsa) {
    flag_gbsa = true;
    TitanUI::print_info(
        "CAS auto-enabled GBSA because CAS uses GBSA alanine-scan energetics.");
  }

  if (flag_nocal && flag_gbsa) {
    TitanUI::print_warning(
        "Nocal mode is structural-only; ignoring -gbsa/-alpb. Remove -nocal "
        "to compute implicit-solvent free energies.");
    flag_gbsa = false;
    flag_alpb = false;
  }

  if (flag_gbsa) {
    if (igb == 66) {
      flag_alpb = true;
      if (flag_cas || std::isfinite(gbsa_rgbmax) ||
          !std::isfinite(gbnsr6_spacing) || gbnsr6_spacing < .1 || gbnsr6_spacing > 1.) {
        TitanUI::print_error("GBNSR6 requires spacing in [0.1,1.0] A, rgbmax=off, and no CAS.");
        exit(EXIT_FAILURE);
      }
    }
    if (gbsa_perf != "off" && gbsa_perf != "summary" &&
        gbsa_perf != "detail") {
      cout << "GBSA Error: -gbsa-perf must be off, summary, or detail.\n";
      exit(EXIT_FAILURE);
    }
    if (gbsa_sasa_points < 24 || gbsa_sasa_points > 8192) {
      cout << "GBSA Error: -gbsa-sasa-points must be in [24, 8192].\n";
      exit(EXIT_FAILURE);
    }
    if (entropy == false) {
      TitanUI::print_info(
          "Note: -gbsa option is on, will automatically activate -entropy "
          "option for Free Energy completeness!");
      entropy = true;
    }
    if (igb != 1 && igb != 2 && igb != 5 && igb != 8 && igb != 66) {
      cout << "GBSA Error: Invalid igb model! Only igb=1 (HCT), igb=2 (OBC1), "
              "igb=5 (OBC2), igb=8 (GBn2), and igb=66 (GBNSR6) are supported.\n";
      exit(EXIT_FAILURE);
    }
    if (ext_diel <= 0.0 || int_diel <= 0.0) {
      cout << "GBSA Error: Dielectric constants (ext_diel and int_diel) must "
              "be strictly positive!\n";
      exit(EXIT_FAILURE);
    }
    if (saltcon < 0.0) {
      cout << "GBSA Error: Salt concentration (saltcon) cannot be negative!\n";
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(gbsa_temperature) || gbsa_temperature <= 0.0) {
      cout << "GBSA Error: -gbsa-temp must be a positive finite temperature "
              "in kelvin.\n";
      exit(EXIT_FAILURE);
    }
    if (gbsa_ie_block != "auto") {
      try {
        if (std::stoi(gbsa_ie_block) <= 0)
          throw std::invalid_argument("non-positive");
      } catch (...) {
        cout << "GBSA Error: -gbsa-ie-block must be auto or a positive "
                "integer.\n";
        exit(EXIT_FAILURE);
      }
    }
    if (gbsa_ie_min_samples <= 0) {
      cout << "GBSA Error: -gbsa-ie-min-samples must be > 0.\n";
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(gbsa_ie_filter_sigma) ||
        gbsa_ie_filter_sigma < 0.0) {
      cout << "GBSA Error: -gbsa-ie-filter-sigma must be finite and >= 0 "
              "(0 disables filtering).\n";
      exit(EXIT_FAILURE);
    }
  }

  if (pbe.flag_pbe) {
    if (!(pbe.tabi_diagnostics == "off" || pbe.tabi_diagnostics == "audit" || pbe.tabi_diagnostics == "full")) {
      TitanUI::print_error(
          "PBE: -pbe-bie-diagnostics must be off, audit, or full.");
      exit(EXIT_FAILURE);
    }
    if (pbe.pbsa) {
      pbe.dgsolv = true;
      pbe.flag_pbe = true;
    }
    if (pbe.backend == "delphi" || pbe.backend == "pydelphi" ||
        pbe.backend == "fd") {
      pbe.backend = "grid";
    }
    if (pbe.backend != "grid" && pbe.backend != "tabi") {
      TitanUI::print_error(
          "PBE: -pbe backend must be grid or bie (legacy alias: tabi; grid aliases: delphi, pydelphi, fd).");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_sum != "auto" && pbe.tabi_sum != "direct" &&
        pbe.tabi_sum != "treecode") {
      TitanUI::print_error(
          "PBE: -pbe-bie-sum must be auto, direct, or treecode.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_tree_preset != "off" &&
        pbe.tabi_tree_preset != "stage3u-safe" &&
        pbe.tabi_tree_preset != "stage3u-perf" &&
        pbe.tabi_tree_preset != "stage3u-perf2") {
      TitanUI::print_error(
          "PBE: -pbe-bie-tree-preset must be off, stage3u-safe, stage3u-perf, or stage3u-perf2.");
      exit(EXIT_FAILURE);
    }
    if (pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
        !pbe.tabi_formulation_explicit &&
        pbe.tabi_formulation == "jcp2013") {
      pbe.tabi_formulation = "jcp2013-treecode";
    }
    const bool scope_y_treecode_default =
        pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
        pbe.tabi_formulation == "jcp2013-treecode" &&
        pbe.tabi_tree_preset == "off";
    if (pbe.tabi_fmm_m2l == "auto") {
      pbe.tabi_fmm_m2l = scope_y_treecode_default
                             ? "single-matvec-analytic"
                             : "off";
    }
    if (pbe.tabi_fmm_m2l == "single-matvec-analytic" &&
        pbe.tabi_precond == "auto") {
      pbe.tabi_precond = "fmm-nearfield-ilu0";
      if (!pbe.tabi_precond_ilu_neighbor_cap_explicit) {
        pbe.tabi_precond_ilu_neighbor_cap = 64;
      }
    } else if ((pbe.tabi_precond == "fmm-nearfield-ilu0" ||
                pbe.tabi_precond == "fmm-nearfield-ilu0-coarse") &&
               !pbe.tabi_precond_ilu_neighbor_cap_explicit) {
      pbe.tabi_precond_ilu_neighbor_cap = 64;
    }
    if (pbe.tabi_geometry != "triangle" &&
        pbe.tabi_geometry != "point-panel") {
      TitanUI::print_error(
          "PBE: -pbe-bie-geometry must be triangle or point-panel.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_diagonal != "solid-angle" && pbe.tabi_diagonal != "flat" &&
        pbe.tabi_diagonal != "off") {
      TitanUI::print_error(
          "PBE: -pbe-bie-diagonal must be solid-angle, flat, or off.");
      exit(EXIT_FAILURE);
    }
    const bool guarded_local_order_path =
        pbe.tabi_fmm_local == "audit-p4" ||
        pbe.tabi_fmm_local == "single-matvec-p4" ||
        pbe.tabi_fmm_local == "gmres-p4";
    if (pbe.tabi_taylor_order < 2 || pbe.tabi_taylor_order > 5) {
      TitanUI::print_error("PBE: -pbe-bie-taylor-order must be 2, 3, 4, or 5.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_taylor_order == 2 && !guarded_local_order_path) {
      TitanUI::print_error(
          "PBE: -pbe-bie-taylor-order=2 is only supported for guarded local FMM/M2L diagnostics.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_taylor_order == 5 && !guarded_local_order_path) {
      TitanUI::print_error(
          "PBE: -pbe-bie-taylor-order=5 is only supported for guarded local FMM/M2L diagnostics.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_derivatives != "analytic" &&
        pbe.tabi_derivatives != "finite-diff-debug") {
      TitanUI::print_error(
          "PBE: -pbe-bie-derivatives must be analytic or finite-diff-debug.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_highres_surface) {
      const bool direct_test_pqr =
          pbe.backend == "tabi" && pbe.tabi_sum == "direct" &&
          !pbe.test_pqr.empty() && pbe.tabi_fmm_m2l == "off";
      if (!direct_test_pqr) {
        TitanUI::print_error(
            "PBE: -pbe-bie-highres-surface=true is restricted to direct BIE -pbe-test-pqr benchmark runs with FMM/M2L off.");
        exit(EXIT_FAILURE);
      }
    }
    if (pbe.tabi_surface_only && pbe.backend != "tabi") {
      TitanUI::print_error(
          "PBE: -pbe-bie-surface-only=true requires -pbe=bie.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_calderon_trace_audit && pbe.backend != "tabi") {
      TitanUI::print_error(
          "PBE: -pbe-bie-calderon-trace-audit=true requires -pbe=bie.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_calderon_aux_surface != "off" &&
        pbe.tabi_calderon_aux_surface != "implicit-vdw") {
      TitanUI::print_error(
          "PBE: -pbe-bie-calderon-aux-surface must be off or implicit-vdw.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_calderon_aux_grid_spacing) ||
        pbe.tabi_calderon_aux_grid_spacing < 0.5 ||
        pbe.tabi_calderon_aux_grid_spacing > 2.5) {
      TitanUI::print_error(
          "PBE: -pbe-bie-calderon-aux-grid-spacing must be between 0.5 and 2.5 Angstrom.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_calderon_aux_surface != "off" &&
        (pbe.backend != "tabi" || !pbe.tabi_calderon_trace_audit)) {
      TitanUI::print_error(
          "PBE: -pbe-bie-calderon-aux-surface=implicit-vdw requires -pbe=bie and -pbe-bie-calderon-trace-audit=true.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_primary_surface != "sphere-patch" &&
        pbe.tabi_primary_surface != "grid-ses" &&
        pbe.tabi_primary_surface != "analytic-ses") {
      TitanUI::print_error(
          "PBE: -pbe-bie-primary-surface must be sphere-patch, grid-ses, or analytic-ses.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_analytic_ses_engine != "current" &&
        pbe.tabi_analytic_ses_engine != "current-dense" &&
        pbe.tabi_analytic_ses_engine != "patch-ray-experimental") {
      TitanUI::print_error(
          "PBE: -pbe-bie-analytic-ses-engine must be current, current-dense, or patch-ray-experimental.");
      exit(EXIT_FAILURE);
    }
    // Only a non-default engine is worth rejecting here.  The value is inert
    // unless the analytic SES builder runs, so holding the default must not
    // fail a sphere-patch or grid-ses run.
    if (pbe.tabi_analytic_ses_engine != kTabiDefaultAnalyticSesEngine &&
        (pbe.backend != "tabi" ||
         pbe.tabi_primary_surface != "analytic-ses")) {
      TitanUI::print_error(
          "PBE: -pbe-bie-analytic-ses-engine only applies to -pbe=bie with -pbe-bie-primary-surface=analytic-ses.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_ses_grid_spacing) ||
        pbe.tabi_ses_grid_spacing < 0.25 ||
        pbe.tabi_ses_grid_spacing > 4.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-ses-grid-spacing must be between 0.25 and 4.0 Angstrom.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_ses_max_grid_vertices < 4000000u ||
        pbe.tabi_ses_max_grid_vertices > 100000000u) {
      TitanUI::print_error(
          "PBE: -pbe-bie-ses-max-grid-vertices must be between 4000000 and 100000000.");
      exit(EXIT_FAILURE);
    }
    if ((pbe.tabi_primary_surface == "grid-ses" ||
         pbe.tabi_primary_surface == "analytic-ses") &&
        (pbe.backend != "tabi" || pbe.tabi_geometry != "triangle")) {
      TitanUI::print_error(
          "PBE: -pbe-bie-primary-surface=" + pbe.tabi_primary_surface +
          " requires the triangle BIE geometry.");
      exit(EXIT_FAILURE);
    }
    if ((pbe.tabi_primary_surface == "grid-ses" ||
         pbe.tabi_primary_surface == "analytic-ses") &&
        (!(pbe.probe_radius > 0.0) || !std::isfinite(pbe.probe_radius))) {
      TitanUI::print_error(
          "PBE: -pbe-bie-primary-surface=" + pbe.tabi_primary_surface +
          " requires a positive -pbe-probe radius.");
      exit(EXIT_FAILURE);
    }
    const int tabi_surface_point_max =
        pbe.tabi_highres_surface ? 65536 : 4096;
    if (pbe.tabi_surface_points < 8 ||
        pbe.tabi_surface_points > tabi_surface_point_max) {
      TitanUI::print_error(
          pbe.tabi_highres_surface
              ? "PBE: -pbe-bie-surface-points must be between 8 and 65536 when -pbe-bie-highres-surface=true."
              : "PBE: -pbe-bie-surface-points must be between 8 and 4096.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_sphere_mesh != "auto" &&
        pbe.tabi_sphere_mesh != "latlong" &&
        pbe.tabi_sphere_mesh != "icosphere") {
      TitanUI::print_error(
          "PBE: -pbe-bie-sphere-mesh must be auto, latlong, or icosphere.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_tree_leaf_size < 8 || pbe.tabi_tree_leaf_size > 512) {
      TitanUI::print_error(
          "PBE: -pbe-bie-tree-leaf-size must be between 8 and 512.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_tree_opening_theta < 0.05 ||
        pbe.tabi_tree_opening_theta > 2.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-opening-theta must be between 0.05 and 2.0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_formulation != "titan-v1" &&
        pbe.tabi_formulation != "jcp2013" &&
        pbe.tabi_formulation != "jcp2013-centroid" &&
        pbe.tabi_formulation != "jcp2013-treecode" &&
        pbe.tabi_formulation != "tabi2-nodepatch" &&
        pbe.tabi_formulation != "calibrated-debug") {
      TitanUI::print_error(
          "PBE: -pbe-bie-formulation must be titan-v1, jcp2013, jcp2013-centroid, jcp2013-treecode, tabi2-nodepatch, or calibrated-debug.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_discretization != "centroid" &&
        pbe.tabi_discretization != "node-patch") {
      TitanUI::print_error(
          "PBE: -pbe-bie-discretization must be centroid or node-patch.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_formulation == "tabi2-nodepatch") {
      if (pbe.tabi_discretization_explicit &&
          pbe.tabi_discretization != "node-patch") {
        TitanUI::print_error(
            "PBE: -pbe-bie-formulation=tabi2-nodepatch conflicts with "
            "-pbe-bie-discretization=centroid.");
        exit(EXIT_FAILURE);
      }
      pbe.tabi_discretization = "node-patch";
    }
    if (pbe.tabi_precond != "auto" && pbe.tabi_precond != "none" &&
        pbe.tabi_precond != "diagonal" && pbe.tabi_precond != "block" &&
        pbe.tabi_precond != "block-budgeted" &&
        pbe.tabi_precond != "coarse-galerkin-probe" &&
        pbe.tabi_precond != "fmm-block-jacobi" &&
        pbe.tabi_precond != "fmm-overlap-schwarz" &&
        pbe.tabi_precond != "fmm-nearfield-ilu0" &&
        pbe.tabi_precond != "fmm-nearfield-ilu0-coarse" &&
        pbe.tabi_precond != "distance-knn-overlap-schwarz" &&
        pbe.tabi_precond != "distance-knn-ilu0") {
      TitanUI::print_error(
          "PBE: -pbe-bie-precond must be auto, none, diagonal, block, block-budgeted, coarse-galerkin-probe, fmm-block-jacobi, fmm-overlap-schwarz, fmm-nearfield-ilu0, fmm-nearfield-ilu0-coarse, distance-knn-overlap-schwarz, or distance-knn-ilu0.");
      exit(EXIT_FAILURE);
    }
    const bool fmm_native_preconditioner =
        pbe.tabi_precond == "fmm-block-jacobi" ||
        pbe.tabi_precond == "fmm-overlap-schwarz" ||
        pbe.tabi_precond == "fmm-nearfield-ilu0" ||
        pbe.tabi_precond == "fmm-nearfield-ilu0-coarse" ||
        pbe.tabi_precond == "distance-knn-overlap-schwarz" ||
        pbe.tabi_precond == "distance-knn-ilu0";
    if (fmm_native_preconditioner &&
        pbe.tabi_fmm_m2l != "single-matvec-analytic") {
      TitanUI::print_error(
          "PBE: FMM-native preconditioners require -pbe-bie-fmm-m2l=single-matvec-analytic.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_precond_block_near != "singular" &&
        pbe.tabi_precond_block_near != "buffer-aware" &&
        pbe.tabi_precond_block_near != "operator-near") {
      TitanUI::print_error(
          "PBE: -pbe-bie-precond-block-near must be singular, buffer-aware, or operator-near.");
      exit(EXIT_FAILURE);
    }
    const int max_block_panel_cap =
        pbe.tabi_precond == "fmm-block-jacobi" ? 128 : 32;
    if (pbe.tabi_precond_block_panel_cap < 8 ||
        pbe.tabi_precond_block_panel_cap > max_block_panel_cap) {
      TitanUI::print_error(
          pbe.tabi_precond == "fmm-block-jacobi"
              ? "PBE: -pbe-bie-precond-block-panel-cap must be between 8 and 128 for fmm-block-jacobi."
              : "PBE: -pbe-bie-precond-block-panel-cap must be between 8 and 32 for this preconditioner.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_precond_overlap_panel_cap < 0 ||
        pbe.tabi_precond_overlap_panel_cap > 128) {
      TitanUI::print_error(
          "PBE: -pbe-bie-precond-overlap-panel-cap must be between 0 and 128.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_precond_ilu_neighbor_cap < 32 ||
        pbe.tabi_precond_ilu_neighbor_cap > 1024) {
      TitanUI::print_error(
          "PBE: -pbe-bie-precond-ilu-neighbor-cap must be between 32 and 1024.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_precond_ilu_drop_tol) ||
        pbe.tabi_precond_ilu_drop_tol < 0.0 ||
        pbe.tabi_precond_ilu_drop_tol > 1.0e-2) {
      TitanUI::print_error(
          "PBE: -pbe-bie-precond-ilu-drop-tol must be finite and between 0 and 1e-2.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_precond_memory_budget_mb < 16 ||
        pbe.tabi_precond_memory_budget_mb > 8192) {
      TitanUI::print_error(
          "PBE: -pbe-bie-precond-memory-budget-mb must be between 16 and 8192.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_precond_coarse_aggregate_cap < 4 ||
        pbe.tabi_precond_coarse_aggregate_cap > 64) {
      TitanUI::print_error(
          "PBE: -pbe-bie-precond-coarse-aggregate-cap must be between 4 and 64.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_precond_coarse_base != "diagonal" &&
        pbe.tabi_precond_coarse_base != "block-budgeted") {
      TitanUI::print_error(
          "PBE: -pbe-bie-precond-coarse-base must be diagonal or block-budgeted.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_precond_coarse_verify != "off" &&
        pbe.tabi_precond_coarse_verify != "full-small") {
      TitanUI::print_error(
          "PBE: -pbe-bie-precond-coarse-verify must be off or full-small.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_near_integration != "auto" &&
        pbe.tabi_near_integration != "centroid" &&
        pbe.tabi_near_integration != "analytic_singular" &&
        pbe.tabi_near_integration != "duffy_adaptive") {
      TitanUI::print_error(
          "PBE: -pbe-bie-near-integration must be auto, centroid, analytic_singular, or duffy_adaptive.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_quad_buffer != "auto" &&
        pbe.tabi_quad_buffer != "off" &&
        pbe.tabi_quad_buffer != "gauss7" &&
        pbe.tabi_quad_buffer != "adaptive") {
      TitanUI::print_error(
          "PBE: -pbe-bie-quad-buffer must be auto, off, gauss7, or adaptive.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_near_policy != "auto" &&
        pbe.tabi_leaf_near_policy != "exact" &&
        pbe.tabi_leaf_near_policy != "smooth-node" &&
        pbe.tabi_leaf_near_policy != "large-protein" &&
        pbe.tabi_leaf_near_policy != "adaptive-large") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-near-policy must be auto, exact, smooth-node, large-protein, or adaptive-large.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_panel_kernel != "auto" &&
        pbe.tabi_leaf_panel_kernel != "scalar" &&
        pbe.tabi_leaf_panel_kernel != "avx2-gather" &&
        pbe.tabi_leaf_panel_kernel != "tile" &&
        pbe.tabi_leaf_panel_kernel != "tile-scalar" &&
        pbe.tabi_leaf_panel_kernel != "tile-avx2") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-panel-kernel must be auto, scalar, avx2-gather, tile, tile-scalar, or tile-avx2.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_adaptive_level != "strict" &&
        pbe.tabi_adaptive_level != "balanced" &&
        pbe.tabi_adaptive_level != "aggressive") {
      TitanUI::print_error(
          "PBE: -pbe-bie-adaptive-level must be strict, balanced, or aggressive.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_cluster_policy != "off" &&
        pbe.tabi_cluster_policy != "leaf-pair-strict" &&
        pbe.tabi_cluster_policy != "leaf-pair-balanced") {
      TitanUI::print_error(
          "PBE: -pbe-bie-cluster-policy must be off, leaf-pair-strict, or leaf-pair-balanced.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_kernel != "auto" &&
        pbe.tabi_leaf_pair_kernel != "scalar" &&
        pbe.tabi_leaf_pair_kernel != "target-avx2" &&
        pbe.tabi_leaf_pair_kernel != "target-avx2-v2" &&
        pbe.tabi_leaf_pair_kernel != "target-avx2-4x4") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-kernel must be auto, scalar, target-avx2, target-avx2-v2, or target-avx2-4x4.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_kernel_profile != "off" &&
        pbe.tabi_leaf_kernel_profile != "summary") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-kernel-profile must be off or summary.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_work_profile != "off" &&
        pbe.tabi_leaf_work_profile != "summary") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-work-profile must be off or summary.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_matvec_cost_profile != "off" &&
        pbe.tabi_matvec_cost_profile != "summary") {
      TitanUI::print_error(
          "PBE: -pbe-bie-matvec-cost-profile must be off or summary.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_jcp_direct_kernel != "baseline" &&
        pbe.tabi_jcp_direct_kernel != "soa-avx2" &&
        pbe.tabi_jcp_direct_kernel != "fused-soa-avx2") {
      TitanUI::print_error(
          "PBE: -pbe-bie-jcp-direct-kernel must be baseline, soa-avx2, or "
          "fused-soa-avx2.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_selection_profile != "off" &&
        pbe.tabi_leaf_pair_selection_profile != "summary") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-selection-profile must be off or summary.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_probe_cache != "off" &&
        pbe.tabi_leaf_pair_probe_cache != "per-matvec") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-probe-cache must be off or per-matvec.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_target_group != "scan" &&
        pbe.tabi_leaf_pair_target_group != "same-target-leaf") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-target-group must be scan or same-target-leaf.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_directness_probe != "full-traversal" &&
        pbe.tabi_leaf_pair_directness_probe != "ancestor-path") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-directness-probe must be full-traversal or ancestor-path.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_directness_cache != "off" &&
        pbe.tabi_leaf_pair_directness_cache != "target-leaf-source-leaf") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-directness-cache must be off or target-leaf-source-leaf.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_directness_verify != "off" &&
        pbe.tabi_leaf_pair_directness_verify != "sampled" &&
        pbe.tabi_leaf_pair_directness_verify != "full-small") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-directness-verify must be off, sampled, or full-small.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_scheduler != "inline-select" &&
        pbe.tabi_leaf_pair_scheduler != "deferred-worklist" &&
        pbe.tabi_leaf_pair_scheduler != "target-leaf-block" &&
        pbe.tabi_leaf_pair_scheduler != "cost-tile-worklist") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-scheduler must be inline-select, deferred-worklist, target-leaf-block, or cost-tile-worklist.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_worklist_max_active < 64 ||
        pbe.tabi_leaf_pair_worklist_max_active > 8192) {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-worklist-max-active must be between 64 and 8192.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_worklist_verify != "off" &&
        pbe.tabi_leaf_pair_worklist_verify != "sampled" &&
        pbe.tabi_leaf_pair_worklist_verify != "full-small") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-worklist-verify must be off, sampled, or full-small.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_block_verify != "off" &&
        pbe.tabi_leaf_block_verify != "sampled" &&
        pbe.tabi_leaf_block_verify != "full-small") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-block-verify must be off, sampled, or full-small.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_mac != "off" && pbe.tabi_leaf_mac != "audit" &&
        pbe.tabi_leaf_mac != "on") {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-mac must be off, audit, or on.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_m2l != "auto" &&
        pbe.tabi_fmm_m2l != "off" &&
        pbe.tabi_fmm_m2l != "single-matvec-analytic") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-m2l must be auto, off, or single-matvec-analytic.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_engine != "cartesian" &&
        pbe.tabi_fmm_engine != "cartesian-full-experimental" &&
        pbe.tabi_fmm_engine != "kifmm-lite" &&
        pbe.tabi_fmm_engine != "kifmm-local" &&
        pbe.tabi_fmm_engine != "kifmm-lite-experimental" &&
        pbe.tabi_fmm_engine != "kifmm-channel-hybrid-experimental") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-engine must be cartesian, cartesian-full-experimental, kifmm-lite, kifmm-local, kifmm-lite-experimental, or kifmm-channel-hybrid-experimental.");
      exit(EXIT_FAILURE);
    }
    const bool cartesian_full_experimental =
        pbe.tabi_fmm_engine == "cartesian-full-experimental";
    if (cartesian_full_experimental &&
        pbe.bie_fmm_cartesian_order != 1 &&
        pbe.bie_fmm_cartesian_order != 4) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-engine=cartesian-full-experimental requires -pbe-bie-fmm-cartesian-order=1 or 4.");
      exit(EXIT_FAILURE);
    }
    if (!cartesian_full_experimental && pbe.bie_fmm_cartesian_order != 0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-cartesian-order is only valid with -pbe-bie-fmm-engine=cartesian-full-experimental.");
      exit(EXIT_FAILURE);
    }
    if (cartesian_full_experimental && pbe.tabi_fmm_local != "off") {
      TitanUI::print_error(
          "PBE: Cartesian full FMM is a separate engine and requires -pbe-bie-fmm-local=off.");
      exit(EXIT_FAILURE);
    }
    if (cartesian_full_experimental &&
        !(pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
          pbe.tabi_formulation == "jcp2013-treecode" &&
          pbe.tabi_derivatives == "analytic")) {
      TitanUI::print_error(
          "PBE: Cartesian full FMM requires the P1 JCP treecode workflow with analytic derivatives; select -pbe=bie -pbe-bie-fast-operator=p1.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_order < 2 || pbe.tabi_kifmm_order > 16 ||
        (pbe.tabi_kifmm_order % 2) != 0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-order must be an even integer between 2 and 16.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_kifmm_equiv_scale) ||
        !std::isfinite(pbe.tabi_kifmm_check_scale) ||
        pbe.tabi_kifmm_equiv_scale <= 1.0 ||
        pbe.tabi_kifmm_check_scale <= pbe.tabi_kifmm_equiv_scale) {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-equiv-scale and -pbe-bie-kifmm-check-scale require 1.0 < equiv-scale < check-scale.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_solver != "qr" && pbe.tabi_kifmm_solver != "svd" &&
        pbe.tabi_kifmm_solver != "normal-tikhonov") {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-solver must be qr, svd, or normal-tikhonov.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_profile != "off" &&
        pbe.tabi_kifmm_profile != "summary" &&
        pbe.tabi_kifmm_profile != "full") {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-profile must be off, summary, or full.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_full_apply != "serial" &&
        pbe.tabi_kifmm_full_apply != "target-node-parallel") {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-full-apply must be serial or target-node-parallel.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_ilu_graph != "cartesian-near" &&
        pbe.tabi_kifmm_ilu_graph != "kifmm-fallback-capped") {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-ilu-graph must be cartesian-near or kifmm-fallback-capped.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_hierarchy_order < 2 ||
        pbe.tabi_kifmm_hierarchy_order > 16 ||
        (pbe.tabi_kifmm_hierarchy_order % 2) != 0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-order must be an even integer between 2 and 16.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_kifmm_hierarchy_theta) ||
        pbe.tabi_kifmm_hierarchy_theta < 0.05 ||
        pbe.tabi_kifmm_hierarchy_theta > 2.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-theta must be between 0.05 and 2.0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_hierarchy_real_panel_guard !=
        "strict-local-check") {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-real-panel-guard must be strict-local-check.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_kifmm_hierarchy_guard_margin_factor) ||
        pbe.tabi_kifmm_hierarchy_guard_margin_factor < 0.0 ||
        pbe.tabi_kifmm_hierarchy_guard_margin_factor > 1.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-guard-margin-factor must be between 0.0 and 1.0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_hierarchy_tree_stop_policy !=
        "panel-radius-occupancy") {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-tree-stop-policy must be panel-radius-occupancy.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(
            pbe.tabi_kifmm_hierarchy_tree_panel_radius_factor) ||
        pbe.tabi_kifmm_hierarchy_tree_panel_radius_factor <= 0.0 ||
        pbe.tabi_kifmm_hierarchy_tree_panel_radius_factor > 8.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-tree-panel-radius-factor must be in (0, 8].");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_hierarchy_tree_occupancy_stop < 1 ||
        pbe.tabi_kifmm_hierarchy_tree_occupancy_stop > 512) {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-tree-occupancy-stop must be between 1 and 512.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_hierarchy_m2l_eval_policy !=
        "target-check-batched") {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-m2l-eval-policy must be target-check-batched.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_hierarchy_matrix_cache_policy !=
            "canonical-offset-cache" &&
        pbe.tabi_kifmm_hierarchy_matrix_cache_policy !=
            "static-translation-table-cache" &&
        pbe.tabi_kifmm_hierarchy_matrix_cache_policy !=
            "target-node-streaming-cache" &&
        pbe.tabi_kifmm_hierarchy_matrix_cache_policy !=
            "bounded-lru-cache") {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-matrix-cache-policy must be canonical-offset-cache, static-translation-table-cache, target-node-streaming-cache, or bounded-lru-cache.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(
            pbe.tabi_kifmm_hierarchy_matrix_cache_limit_mb) ||
        pbe.tabi_kifmm_hierarchy_matrix_cache_limit_mb < 0.0 ||
        (pbe.tabi_kifmm_hierarchy_matrix_cache_policy ==
             "bounded-lru-cache" &&
         pbe.tabi_kifmm_hierarchy_matrix_cache_limit_mb <= 0.0)) {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-matrix-cache-limit-mb must be positive for bounded-lru-cache and non-negative otherwise.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_hierarchy_gemm_batch_policy != "per-pair") {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-gemm-batch-policy must be per-pair.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_hierarchy_m2l_parallel_policy != "serial" &&
        pbe.tabi_kifmm_hierarchy_m2l_parallel_policy !=
            "target-node-key-group") {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-m2l-parallel-policy must be serial or target-node-key-group.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_kifmm_hierarchy_m2l_parallel_workers < 1 ||
        pbe.tabi_kifmm_hierarchy_m2l_parallel_workers > 64) {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-hierarchy-m2l-parallel-workers must be between 1 and 64.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_cartesian_k4_order < 2 ||
        pbe.tabi_cartesian_k4_order > 8) {
      TitanUI::print_error(
          "PBE: -pbe-bie-cartesian-k4-order must be between 2 and 8.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_cartesian_k4_theta) ||
        pbe.tabi_cartesian_k4_theta < 0.05 ||
        pbe.tabi_cartesian_k4_theta > 2.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-cartesian-k4-theta must be between 0.05 and 2.0.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_cartesian_k4_geometry_theta) ||
        pbe.tabi_cartesian_k4_geometry_theta <= 0.0 ||
        pbe.tabi_cartesian_k4_geometry_theta > 2.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-cartesian-k4-geometry-theta must be in (0, 2].");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_cartesian_k4_tail_tol) ||
        pbe.tabi_cartesian_k4_tail_tol < 0.0 ||
        pbe.tabi_cartesian_k4_tail_tol > 1.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-cartesian-k4-tail-tol must be between 0.0 and 1.0.");
      exit(EXIT_FAILURE);
    }
    const bool kifmm_experimental =
        pbe.tabi_fmm_engine == "kifmm-lite-experimental";
    const bool kifmm_channel_hybrid_experimental =
        pbe.tabi_fmm_engine == "kifmm-channel-hybrid-experimental";
    if (pbe.flag_pbe && pbe.tabi_kifmm_full_apply != "serial" &&
        !kifmm_experimental) {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-full-apply=target-node-parallel requires -pbe-bie-fmm-engine=kifmm-lite-experimental.");
      exit(EXIT_FAILURE);
    }
    if (pbe.flag_pbe && pbe.tabi_kifmm_ilu_graph != "cartesian-near" &&
        !kifmm_experimental) {
      TitanUI::print_error(
          "PBE: -pbe-bie-kifmm-ilu-graph=kifmm-fallback-capped requires -pbe-bie-fmm-engine=kifmm-lite-experimental.");
      exit(EXIT_FAILURE);
    }
    if (pbe.flag_pbe &&
        (pbe.tabi_fmm_engine == "kifmm-lite" ||
         pbe.tabi_fmm_engine == "kifmm-local")) {
      TitanUI::print_error(
          "PBE: KIFMM engine skeleton only; matvec not implemented in FMM-E1.");
      exit(EXIT_FAILURE);
    }
    if (pbe.flag_pbe && kifmm_experimental) {
      const bool kifmm_theta_allowed =
          std::abs(pbe.tabi_tree_opening_theta - 0.6) <= 1.0e-12 ||
          std::abs(pbe.tabi_tree_opening_theta - 0.7) <= 1.0e-12 ||
          std::abs(pbe.tabi_tree_opening_theta - 0.75) <= 1.0e-12 ||
          std::abs(pbe.tabi_tree_opening_theta - 0.8) <= 1.0e-12;
      if (!(pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
            pbe.tabi_fmm_m2l == "single-matvec-analytic" &&
            pbe.tabi_matvec_audit == "sampled-detail" &&
            pbe.tabi_kifmm_order == 6 &&
            pbe.tabi_kifmm_solver == "qr" &&
            pbe.tabi_kifmm_full_apply == "target-node-parallel" &&
            kifmm_theta_allowed &&
            std::abs(pbe.tabi_kifmm_equiv_scale - 1.05) <= 1.0e-12 &&
            std::abs(pbe.tabi_kifmm_check_scale - 2.40) <= 1.0e-12 &&
            pbe.tabi_near_integration == "analytic_singular" &&
            pbe.tabi_quad_buffer == "off" &&
            pbe.tabi_precond == "fmm-nearfield-ilu0" &&
            pbe.tabi_precond_ilu_neighbor_cap == 64 &&
            pbe.tabi_precond_ilu_drop_tol == 0.0)) {
        TitanUI::print_error(
            "PBE: -pbe-bie-fmm-engine=kifmm-lite-experimental requires TABI treecode with analytical FMM/M2L, sampled-detail audit, KIFMM order 6, theta 0.6/0.7/0.75/0.8, equiv-scale 1.05, check-scale 2.40, QR solver, target-node-parallel full apply, analytic_singular near integration, quad-buffer=off, and fmm-nearfield-ilu0 with neighbor cap 64 and drop tolerance 0.");
        exit(EXIT_FAILURE);
      }
    }
    if (pbe.flag_pbe && kifmm_channel_hybrid_experimental) {
      if (!(pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
            pbe.tabi_fmm_m2l == "single-matvec-analytic" &&
            pbe.tabi_matvec_audit == "sampled-detail" &&
            pbe.tabi_near_integration == "analytic_singular" &&
            pbe.tabi_quad_buffer == "off" &&
            pbe.tabi_precond == "fmm-nearfield-ilu0" &&
            pbe.tabi_precond_ilu_neighbor_cap == 64 &&
            pbe.tabi_precond_ilu_drop_tol == 0.0 &&
            pbe.tabi_kifmm_hierarchy_order == 8 &&
            std::abs(pbe.tabi_kifmm_hierarchy_theta - 0.6) <=
                1.0e-12 &&
            pbe.tabi_kifmm_hierarchy_real_panel_guard ==
                "strict-local-check" &&
            pbe.tabi_kifmm_hierarchy_guard_margin_factor == 0.0 &&
            pbe.tabi_kifmm_hierarchy_tree_stop_policy ==
                "panel-radius-occupancy" &&
            std::abs(
                pbe.tabi_kifmm_hierarchy_tree_panel_radius_factor -
                1.5) <= 1.0e-12 &&
            pbe.tabi_kifmm_hierarchy_tree_occupancy_stop == 32 &&
            pbe.tabi_kifmm_hierarchy_m2l_eval_policy ==
                "target-check-batched" &&
            (pbe.tabi_kifmm_hierarchy_matrix_cache_policy ==
                 "canonical-offset-cache" ||
             pbe.tabi_kifmm_hierarchy_matrix_cache_policy ==
                 "static-translation-table-cache" ||
             pbe.tabi_kifmm_hierarchy_matrix_cache_policy ==
                 "target-node-streaming-cache" ||
             pbe.tabi_kifmm_hierarchy_matrix_cache_policy ==
                 "bounded-lru-cache") &&
            (pbe.tabi_kifmm_hierarchy_matrix_cache_policy !=
                 "bounded-lru-cache" ||
             pbe.tabi_kifmm_hierarchy_matrix_cache_limit_mb > 0.0) &&
            pbe.tabi_kifmm_hierarchy_gemm_batch_policy == "per-pair" &&
            (pbe.tabi_kifmm_hierarchy_m2l_parallel_policy == "serial" ||
             pbe.tabi_kifmm_hierarchy_m2l_parallel_policy ==
                 "target-node-key-group") &&
            pbe.tabi_kifmm_hierarchy_m2l_parallel_workers >= 1 &&
            pbe.tabi_cartesian_k4_order == 5 &&
            std::abs(pbe.tabi_cartesian_k4_theta - 0.6) <= 1.0e-12 &&
            std::abs(pbe.tabi_cartesian_k4_geometry_theta - 0.4) <=
                1.0e-12 &&
            std::abs(pbe.tabi_cartesian_k4_tail_tol - 0.002) <=
                1.0e-12)) {
        TitanUI::print_error(
            "PBE: -pbe-bie-fmm-engine=kifmm-channel-hybrid-experimental requires TABI treecode with analytical FMM/M2L, sampled-detail audit, analytic_singular near integration, quad-buffer=off, fmm-nearfield-ilu0 with neighbor cap 64 and drop tolerance 0, hierarchy order 8, hierarchy theta 0.6, strict-local-check real-panel guard, guard margin 0, panel-radius-occupancy tree stop with radius factor 1.5 and occupancy stop 32, target-check-batched M2L, canonical-offset-cache, static-translation-table-cache, target-node-streaming-cache, or bounded-lru-cache with a positive cache limit, per-pair GEMM, serial or target-node-key-group M2L parallel policy, Cartesian K4 order 5, theta 0.6, geometry theta 0.4, and tail tolerance 0.002.");
        exit(EXIT_FAILURE);
      }
    }
    if (pbe.tabi_matvec_audit == "sampled-detail" &&
        !(pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
          pbe.tabi_fmm_m2l == "single-matvec-analytic")) {
      TitanUI::print_error(
          "PBE: -pbe-bie-matvec-audit=sampled-detail requires analytical FMM/M2L TABI treecode.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_cartesian_audit != "off" &&
        pbe.tabi_fmm_cartesian_audit != "on") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-cartesian-audit must be off or on.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_cartesian_audit == "on" &&
        !(pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
          pbe.tabi_fmm_m2l == "single-matvec-analytic" &&
          pbe.tabi_matvec_audit == "sampled-detail" &&
          (pbe.tabi_taylor_order >= 2 && pbe.tabi_taylor_order <= 5))) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-cartesian-audit=on requires analytical FMM/M2L TABI treecode with -pbe-bie-matvec-audit=sampled-detail and Taylor order 2, 3, 4, or 5.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_cartesian_error_audit != "off" &&
        pbe.tabi_fmm_cartesian_error_audit != "channel-shadow" &&
        pbe.tabi_fmm_cartesian_error_audit !=
            "channel-shadow-retry-global-theta015") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-cartesian-error-audit must be off, channel-shadow, or channel-shadow-retry-global-theta015.");
      exit(EXIT_FAILURE);
    }
    // Order 4 is admitted alongside order 1 so that a matched per-channel
    // O1/O4 comparison can be run at all.  The audit itself is order
    // agnostic -- it reuses the pre-audit's sampled target set and compares
    // each channel's direct sum against whatever the Cartesian operator
    // produced -- and the previous restriction to order 1 was the scope of
    // the study that qualified it, not a property of the measurement.  The
    // P4 closeout records the consequence as its first open item: "P2A
    // channel controls unavailable".  Admitting order 4 supplies the
    // control; it does not certify order 4 for anything else.
    if (pbe.tabi_fmm_cartesian_error_audit == "channel-shadow" &&
        !(pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
          pbe.tabi_fmm_engine == "cartesian-full-experimental" &&
          (pbe.bie_fmm_cartesian_order == 1 ||
           pbe.bie_fmm_cartesian_order == 4) &&
          pbe.tabi_discretization == "node-patch" && pbe.salt > 0.0 &&
          pbe.tabi_node_patch_operator_audit == "on")) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-cartesian-error-audit=channel-shadow requires coupled node-patch Cartesian full FMM order 1 or 4 with -pbe-bie-node-patch-operator-audit=on.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_cartesian_error_audit ==
        "channel-shadow-retry-global-theta015") {
      const int retry_effective_leaf =
          pbe.tabi_tree_leaf_size_explicit ? pbe.tabi_tree_leaf_size : 64;
      const double retry_effective_theta =
          pbe.tabi_tree_opening_theta_explicit
              ? pbe.tabi_tree_opening_theta
              : 0.20;
      const bool retry_scope =
          pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
          pbe.tabi_formulation == "jcp2013-treecode" &&
          pbe.tabi_fmm_m2l == "single-matvec-analytic" &&
          pbe.tabi_fmm_engine == "cartesian-full-experimental" &&
          pbe.bie_fmm_cartesian_order == 1 &&
          pbe.tabi_geometry == "triangle" &&
          pbe.tabi_discretization == "node-patch" && pbe.salt > 0.0 &&
          pbe.tabi_node_patch_operator_audit == "on" &&
          !pbe.tabi_surface_only && pbe.tabi_precond_explicit &&
          pbe.tabi_precond_requested == "fmm-nearfield-ilu0" &&
          pbe.tabi_precond == "fmm-nearfield-ilu0" &&
          retry_effective_leaf == 64 &&
          TitanPBE::fmm_cartesian_retry_theta_matches(retry_effective_theta,
                                                       0.20);
      if (!retry_scope) {
        TitanUI::print_error(
            "PBE: -pbe-bie-fmm-cartesian-error-audit=channel-shadow-retry-global-theta015 requires TABI JCP treecode coupled node-patch Cartesian full FMM order 1, salt>0, node-patch audit=on, surface-only=false, explicit -pbe-bie-precond=fmm-nearfield-ilu0, and effective leaf=64/theta=0.20.");
        exit(EXIT_FAILURE);
      }
    }
    if (pbe.tabi_precond_ilu_regularization != "off" &&
        pbe.tabi_precond_ilu_regularization != "adaptive") {
      TitanUI::print_error(
          "PBE: -pbe-bie-precond-ilu-regularization must be off or adaptive.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_precond_ilu_regularization_max) ||
        pbe.tabi_precond_ilu_regularization_max < 0.0 ||
        pbe.tabi_precond_ilu_regularization_max > 1.0e-2) {
      TitanUI::print_error(
          "PBE: -pbe-bie-precond-ilu-regularization-max must be finite and between 0 and 1e-2.");
      exit(EXIT_FAILURE);
    }
    const bool sparse_ilu_preconditioner =
        pbe.tabi_precond == "fmm-nearfield-ilu0" ||
        pbe.tabi_precond == "fmm-nearfield-ilu0-coarse" ||
        pbe.tabi_precond == "distance-knn-ilu0";
    // Adaptive is the default for sparse ILU. Other preconditioners do not use
    // this setting; keep their audit state unambiguous instead of rejecting a
    // default that is irrelevant to them.
    if (!sparse_ilu_preconditioner)
      pbe.tabi_precond_ilu_regularization = "off";
    if (pbe.tabi_fmm_local != "off" &&
        pbe.tabi_fmm_local != "audit-p4" &&
        pbe.tabi_fmm_local != "single-matvec-p4" &&
        pbe.tabi_fmm_local != "gmres-p4") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local must be off, audit-p4, single-matvec-p4, or gmres-p4.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_ilu_cap_sweep &&
        pbe.tabi_fmm_local != "gmres-p4") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-ilu-cap-sweep=true is restricted to explicit gmres-p4 diagnostic rows.");
      exit(EXIT_FAILURE);
    }
    const bool tabi_fmm_local_static_refine =
        pbe.tabi_fmm_local_refine == "operator-tail-static" ||
        pbe.tabi_fmm_local_refine == "operator-tail-linf";
    if (pbe.tabi_fmm_local == "single-matvec-p4" &&
        !(pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
          pbe.tabi_fmm_m2l == "single-matvec-analytic" &&
          pbe.tabi_matvec_audit == "sampled-detail" &&
          pbe.tabi_taylor_order >= 2 && pbe.tabi_taylor_order <= 5)) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local=single-matvec-p4 requires analytical FMM/M2L TABI treecode with -pbe-bie-matvec-audit=sampled-detail and Taylor order 2, 3, 4, or 5.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local == "audit-p4" &&
        !(pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
          pbe.tabi_fmm_m2l == "single-matvec-analytic" &&
          pbe.tabi_matvec_audit == "sampled-detail" &&
          pbe.tabi_taylor_order >= 2 && pbe.tabi_taylor_order <= 5)) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local=audit-p4 requires analytical FMM/M2L TABI treecode with -pbe-bie-matvec-audit=sampled-detail and Taylor order 2, 3, 4, or 5.");
      exit(EXIT_FAILURE);
    }
    const bool tabi_gmres_p4_refine_tolerance_ok =
        (pbe.tabi_fmm_local_refine == "operator-tail-static" &&
         std::abs(pbe.tabi_fmm_local_refine_tol - 2.0e-3) <= 1.0e-15) ||
        pbe.tabi_fmm_local_refine == "operator-tail-linf";
    const bool tabi_gmres_p4_ilu_cap_sweep_value =
        pbe.tabi_precond_ilu_neighbor_cap == 32 ||
        pbe.tabi_precond_ilu_neighbor_cap == 64 ||
        pbe.tabi_precond_ilu_neighbor_cap == 128;
    const bool tabi_gmres_p4_ilu_cap_ok =
        pbe.tabi_precond_ilu_neighbor_cap == 64 ||
        (pbe.tabi_fmm_local_ilu_cap_sweep &&
         tabi_gmres_p4_ilu_cap_sweep_value);
    const bool tabi_fmm_local_refine_probe_mode_valid =
        pbe.tabi_fmm_local_refine_probe_mode == "rademacher" ||
        pbe.tabi_fmm_local_refine_probe_mode == "solution" ||
        pbe.tabi_fmm_local_refine_probe_mode == "mixed";
    if (!tabi_fmm_local_refine_probe_mode_valid) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-refine-probe-mode must be rademacher, solution, or mixed.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local == "gmres-p4" &&
        pbe.tabi_fmm_local_refine_probe_mode != "rademacher") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local=gmres-p4 uses a prepared static operator and requires rademacher refinement probes; solution or mixed probes are restricted to non-GMRES diagnostics.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local == "gmres-p4" &&
        !(pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
          pbe.tabi_fmm_m2l == "single-matvec-analytic" &&
          pbe.tabi_matvec_audit == "sampled-detail" &&
          pbe.tabi_taylor_order >= 2 && pbe.tabi_taylor_order <= 5 &&
          tabi_fmm_local_static_refine &&
          tabi_gmres_p4_refine_tolerance_ok &&
          pbe.tabi_fmm_local_plan_builder == "metric-cache" &&
          (pbe.tabi_fmm_local_cache == "green-jet-compressed" ||
           pbe.tabi_fmm_local_cache == "green-jet-tiled") &&
          pbe.tabi_fmm_local_fallback == "packed-avx2" &&
          pbe.tabi_near_integration == "analytic_singular" &&
           pbe.tabi_quad_buffer == "off" &&
           pbe.tabi_precond == "fmm-nearfield-ilu0" &&
           tabi_gmres_p4_ilu_cap_ok &&
           pbe.tabi_fmm_local_refine_probe_mode == "rademacher" &&
           pbe.tabi_precond_ilu_drop_tol == 0.0)) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local=gmres-p4 requires analytical FMM/M2L TABI treecode, sampled-detail audit, Taylor order 2, 3, 4, or 5, static local refinement (operator-tail-static at tolerance 0.002, or operator-tail-linf with a valid diagnostic tolerance), rademacher refinement probes, metric-cache plan building, green-jet-compressed or green-jet-tiled cache, packed-avx2 fallback, analytic_singular near integration with quad-buffer=off, and fmm-nearfield-ilu0 with neighbor cap 64 and drop tolerance 0. Diagnostic ILU cap sweeps may use neighbor cap 32, 64, or 128 only when -pbe-bie-fmm-local-ilu-cap-sweep=true.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_refine != "off" &&
        pbe.tabi_fmm_local_refine != "p3-p4-tail" &&
        pbe.tabi_fmm_local_refine != "operator-tail-static" &&
        pbe.tabi_fmm_local_refine != "operator-tail-linf") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-refine must be off, p3-p4-tail, operator-tail-static, or operator-tail-linf.");
      exit(EXIT_FAILURE);
    }
    if (tabi_fmm_local_static_refine &&
        pbe.tabi_fmm_local != "audit-p4" &&
        pbe.tabi_fmm_local != "single-matvec-p4" &&
        pbe.tabi_fmm_local != "gmres-p4") {
      TitanUI::print_error(
          "PBE: static FMM local refinement requires -pbe-bie-fmm-local=audit-p4, single-matvec-p4, or gmres-p4.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_refine == "p3-p4-tail" &&
        pbe.tabi_fmm_local != "audit-p4") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-refine=p3-p4-tail requires -pbe-bie-fmm-local=audit-p4.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_refine == "p3-p4-tail" &&
        pbe.tabi_taylor_order != 4) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-refine=p3-p4-tail requires Taylor order 4.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_cache != "off" &&
        pbe.tabi_fmm_local_cache != "kernel-jet" &&
        pbe.tabi_fmm_local_cache != "green-jet-compressed" &&
        pbe.tabi_fmm_local_cache != "green-jet-tiled") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-cache must be off, kernel-jet, green-jet-compressed, or green-jet-tiled.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_cache == "kernel-jet" &&
        !tabi_fmm_local_static_refine) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-cache=kernel-jet requires static local refinement.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_cache == "green-jet-compressed" &&
        !tabi_fmm_local_static_refine) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-cache=green-jet-compressed requires static local refinement.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_cache == "green-jet-tiled" &&
        !tabi_fmm_local_static_refine) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-cache=green-jet-tiled requires static local refinement.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_cache == "green-jet-tiled" &&
        pbe.tabi_fmm_local != "single-matvec-p4" &&
        pbe.tabi_fmm_local != "gmres-p4") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-cache=green-jet-tiled requires -pbe-bie-fmm-local=single-matvec-p4 or gmres-p4.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_cache == "green-jet-tiled" &&
        pbe.tabi_matvec_audit != "sampled-detail") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-cache=green-jet-tiled requires -pbe-bie-matvec-audit=sampled-detail.");
      exit(EXIT_FAILURE);
    }
    if ((pbe.tabi_fmm_local == "single-matvec-p4" ||
         pbe.tabi_fmm_local == "gmres-p4") &&
        !tabi_fmm_local_static_refine) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local=single-matvec-p4 or gmres-p4 requires static local refinement.");
      exit(EXIT_FAILURE);
    }
    if ((pbe.tabi_fmm_local == "single-matvec-p4" ||
         pbe.tabi_fmm_local == "gmres-p4") &&
        pbe.tabi_fmm_local_cache != "kernel-jet" &&
        pbe.tabi_fmm_local_cache != "green-jet-compressed" &&
        pbe.tabi_fmm_local_cache != "green-jet-tiled") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local=single-matvec-p4 or gmres-p4 requires kernel-jet, green-jet-compressed, or green-jet-tiled cache.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_cache_budget_mb < 64 ||
        pbe.tabi_fmm_local_cache_budget_mb > 8192) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-cache-budget-mb must be between 64 and 8192.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_fallback != "scalar" &&
        pbe.tabi_fmm_local_fallback != "packed-avx2") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-fallback must be scalar or packed-avx2.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_fallback == "packed-avx2" &&
        pbe.tabi_fmm_local != "single-matvec-p4" &&
        pbe.tabi_fmm_local != "gmres-p4") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-fallback=packed-avx2 requires -pbe-bie-fmm-local=single-matvec-p4 or gmres-p4.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_fallback == "packed-avx2" &&
        !tabi_fmm_local_static_refine) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-fallback=packed-avx2 requires static local refinement.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_fallback == "packed-avx2" &&
        pbe.tabi_fmm_local_cache != "kernel-jet" &&
        pbe.tabi_fmm_local_cache != "green-jet-compressed" &&
        pbe.tabi_fmm_local_cache != "green-jet-tiled") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-fallback=packed-avx2 requires kernel-jet, green-jet-compressed, or green-jet-tiled cache.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_fallback == "packed-avx2" &&
        pbe.tabi_near_integration != "analytic_singular") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-fallback=packed-avx2 requires -pbe-bie-near-integration=analytic_singular.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_fallback == "packed-avx2" &&
        pbe.tabi_quad_buffer != "off") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-fallback=packed-avx2 requires -pbe-bie-quad-buffer=off.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_plan_builder == "rescan") {
      pbe.tabi_fmm_local_plan_builder = "legacy-rescan";
    } else if (pbe.tabi_fmm_local_plan_builder == "cached-metrics") {
      pbe.tabi_fmm_local_plan_builder = "metric-cache";
    }
    if (pbe.tabi_convergence_calibration == "calibrated-tail-v1") {
      pbe.tabi_convergence_calibration = "stage3j";
    }
    if (pbe.tabi_convergence_estimator_model ==
        "calibrated-estimator-v1") {
      pbe.tabi_convergence_estimator_model = "stage3l";
    }
    if (pbe.tabi_fmm_local_plan_builder != "legacy-rescan" &&
        pbe.tabi_fmm_local_plan_builder != "metric-cache") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-plan-builder must be rescan, cached-metrics, legacy-rescan, or metric-cache.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_plan_builder == "metric-cache" &&
        !tabi_fmm_local_static_refine) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-plan-builder=metric-cache requires static local refinement.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_plan_builder == "metric-cache" &&
        pbe.tabi_fmm_local != "audit-p4" &&
        pbe.tabi_fmm_local != "single-matvec-p4" &&
        pbe.tabi_fmm_local != "gmres-p4") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-plan-builder=metric-cache requires audit-p4, single-matvec-p4, or gmres-p4.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_plan_builder == "metric-cache" &&
        pbe.tabi_matvec_audit != "sampled-detail") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-plan-builder=metric-cache requires sampled-detail audit.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_plan_builder == "metric-cache" &&
        (pbe.tabi_taylor_order < 2 || pbe.tabi_taylor_order > 5)) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-plan-builder=metric-cache requires Taylor order 2, 3, 4, or 5.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_refine_probe_count != 4 &&
        pbe.tabi_fmm_local_refine_probe_count != 8 &&
        pbe.tabi_fmm_local_refine_probe_count != 16) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-refine-probe-count must be 4, 8, or 16.");
      exit(EXIT_FAILURE);
    }
    if ((pbe.tabi_fmm_local_refine_probe_count != 4 ||
         pbe.tabi_fmm_local_refine_probe_mode != "rademacher") &&
        !tabi_fmm_local_static_refine) {
      TitanUI::print_error(
          "PBE: FMM local refine probe controls require static local refinement.");
      exit(EXIT_FAILURE);
    }
    if ((pbe.tabi_fmm_local_refine_probe_count != 4 ||
         pbe.tabi_fmm_local_refine_probe_mode != "rademacher") &&
        pbe.tabi_matvec_audit != "sampled-detail") {
      TitanUI::print_error(
          "PBE: FMM local refine probe controls require sampled-detail audit.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_geometry_guard != "off" &&
        pbe.tabi_fmm_local_geometry_guard != "strict-geometry-mac") {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-geometry-guard must be off or strict-geometry-mac.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_geometry_guard != "off" &&
        !(tabi_fmm_local_static_refine &&
          (pbe.tabi_fmm_local == "audit-p4" ||
           pbe.tabi_fmm_local == "single-matvec-p4" ||
           pbe.tabi_fmm_local == "gmres-p4") &&
          pbe.tabi_matvec_audit == "sampled-detail" &&
          pbe.tabi_taylor_order >= 2 && pbe.tabi_taylor_order <= 5)) {
      TitanUI::print_error(
          "PBE: strict FMM local geometry guard requires static local diagnostics with sampled-detail audit and Taylor order 2, 3, 4, or 5.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_geometry_guard != "off" &&
        (!std::isfinite(pbe.tabi_fmm_local_geometry_theta) ||
         pbe.tabi_fmm_local_geometry_theta <= 0.0)) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-geometry-theta must be positive when strict geometry guard is enabled.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local_geometry_guard == "off" &&
        pbe.tabi_fmm_local_geometry_theta != 0.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-geometry-theta requires -pbe-bie-fmm-local-geometry-guard=strict-geometry-mac.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_fmm_local == "gmres-p4" &&
        (pbe.tabi_fmm_local_geometry_guard != "strict-geometry-mac" ||
         std::abs(pbe.tabi_fmm_local_geometry_theta - 0.5) > 1.0e-15)) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local=gmres-p4 requires -pbe-bie-fmm-local-geometry-guard=strict-geometry-mac and -pbe-bie-fmm-local-geometry-theta=0.5.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_fmm_local_refine_tol) ||
        pbe.tabi_fmm_local_refine_tol < 1.0e-4 ||
        pbe.tabi_fmm_local_refine_tol > 5.0e-2) {
      TitanUI::print_error(
          "PBE: -pbe-bie-fmm-local-refine-tol must be between 0.0001 and 0.05.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_matvec_audit_sample_count < 128 ||
        pbe.tabi_matvec_audit_sample_count > 4096) {
      TitanUI::print_error(
          "PBE: -pbe-bie-matvec-audit-sample-count must be between 128 and 4096.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_node_patch_operator_audit != "on" &&
        pbe.tabi_node_patch_operator_audit != "off") {
      TitanUI::print_error(
          "PBE: -pbe-bie-node-patch-operator-audit must be on or off.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_target_batch_size != 1 &&
        pbe.tabi_target_batch_size != 4 &&
        pbe.tabi_target_batch_size != 8) {
      TitanUI::print_error(
          "PBE: -pbe-bie-target-batch-size must be 1, 4, or 8.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_min_source_panels < 16 ||
        pbe.tabi_leaf_pair_min_source_panels > 512) {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-min-source-panels must be between 16 and 512.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_min_targets < 2 ||
        pbe.tabi_leaf_pair_min_targets > 16) {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-min-targets must be between 2 and 16.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_leaf_pair_max_lookahead < 4 ||
        pbe.tabi_leaf_pair_max_lookahead > 128) {
      TitanUI::print_error(
          "PBE: -pbe-bie-leaf-pair-max-lookahead must be between 4 and 128.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_interaction_mode != "auto" &&
        pbe.tabi_interaction_mode != "cached" &&
        pbe.tabi_interaction_mode != "onthefly") {
      TitanUI::print_error(
          "PBE: -pbe-bie-interaction-mode must be auto, cached, or onthefly.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_node_bounds != "moment" &&
        pbe.tabi_node_bounds != "tight-aabb") {
      TitanUI::print_error(
          "PBE: -pbe-bie-node-bounds must be moment or tight-aabb.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_tree_builder != "grouped-morton" &&
        pbe.tabi_tree_builder != "topdown-aabb") {
      TitanUI::print_error(
          "PBE: -pbe-bie-tree-builder must be grouped-morton or topdown-aabb.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_mac_safety != "off" && pbe.tabi_mac_safety != "audit" &&
        pbe.tabi_mac_safety != "physical-gap") {
      TitanUI::print_error(
          "PBE: -pbe-bie-mac-safety must be off, audit, or physical-gap.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_mac_gap_factor) ||
        pbe.tabi_mac_gap_factor < 0.5 || pbe.tabi_mac_gap_factor > 4.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-mac-gap-factor must be between 0.5 and 4.0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_moment_audit != "off" &&
        pbe.tabi_moment_audit != "sampled" &&
        pbe.tabi_moment_audit != "full-small") {
      TitanUI::print_error(
          "PBE: -pbe-bie-moment-audit must be off, sampled, or full-small.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_progress != "off" && pbe.tabi_progress != "auto" &&
        pbe.tabi_progress != "detail") {
      TitanUI::print_error(
          "PBE: -pbe-bie-progress must be off, auto, or detail.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_convergence_dielectric_amp != "off" &&
        pbe.tabi_convergence_dielectric_amp != "on") {
      TitanUI::print_error(
          "PBE: -pbe-bie-convergence-dielectric-amp must be off or on.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_convergence_estimator_model != "stage3l" &&
        pbe.tabi_convergence_estimator_model != "channel-tail" &&
        pbe.tabi_convergence_estimator_model != "channel-tail-eps") {
      TitanUI::print_error(
          "PBE: -pbe-bie-convergence-estimator-model must be calibrated-estimator-v1, stage3l, channel-tail, or channel-tail-eps.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_convergence_channel_calibration != "stage3m" &&
        pbe.tabi_convergence_channel_calibration != "k3-sqrt-eps" &&
        pbe.tabi_convergence_channel_calibration != "k3-eps" &&
        pbe.tabi_convergence_channel_calibration != "k4-sqrt-eps" &&
        pbe.tabi_convergence_channel_calibration != "k4-eps" &&
        pbe.tabi_convergence_channel_calibration !=
            "all-derivative-sqrt-eps" &&
        pbe.tabi_convergence_channel_calibration != "polynomial-eps") {
      TitanUI::print_error(
          "PBE: -pbe-bie-convergence-channel-calibration must be stage3m, k3-sqrt-eps, k3-eps, k4-sqrt-eps, k4-eps, all-derivative-sqrt-eps, or polynomial-eps.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_convergence_split_policy != "global" &&
        pbe.tabi_convergence_split_policy != "k3-hot") {
      TitanUI::print_error(
          "PBE: -pbe-bie-convergence-split-policy must be global or k3-hot.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_convergence_split_selectivity != "legacy" &&
        pbe.tabi_convergence_split_selectivity != "absolute-aware" &&
        pbe.tabi_convergence_split_selectivity !=
            "channel-absolute-aware") {
      TitanUI::print_error(
          "PBE: -pbe-bie-convergence-split-selectivity must be legacy, absolute-aware, or channel-absolute-aware.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_convergence_split_absolute_floor) ||
        pbe.tabi_convergence_split_absolute_floor < 0.0 ||
        pbe.tabi_convergence_split_absolute_floor > 1.0e-4) {
      TitanUI::print_error(
          "PBE: -pbe-bie-convergence-split-absolute-floor must be in [0, 1e-4].");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_convergence_overreject_audit != "off" &&
        pbe.tabi_convergence_overreject_audit != "sampled" &&
        pbe.tabi_convergence_overreject_audit != "full-small") {
      TitanUI::print_error(
          "PBE: -pbe-bie-convergence-overreject-audit must be off, sampled, or full-small.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_convergence_overreject_detail != "off" &&
        pbe.tabi_convergence_overreject_detail != "summary" &&
        pbe.tabi_convergence_overreject_detail != "sampled-detail") {
      TitanUI::print_error(
          "PBE: -pbe-bie-convergence-overreject-detail must be off, summary, or sampled-detail.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_convergence_shape_audit != "off" &&
        pbe.tabi_convergence_shape_audit != "summary" &&
        pbe.tabi_convergence_shape_audit != "sampled-detail") {
      TitanUI::print_error(
          "PBE: -pbe-bie-convergence-shape-audit must be off, summary, or sampled-detail.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_convergence_kernel_response_audit != "off" &&
        pbe.tabi_convergence_kernel_response_audit != "summary" &&
        pbe.tabi_convergence_kernel_response_audit != "sampled-detail") {
      TitanUI::print_error(
          "PBE: -pbe-bie-convergence-kernel-response-audit must be off, summary, or sampled-detail.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_channel_missed_semantics != "legacy" &&
        pbe.tabi_channel_missed_semantics != "accepted-only") {
      TitanUI::print_error(
          "PBE: -pbe-bie-channel-missed-semantics must be legacy or accepted-only.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_accepted_miss_diagnostics != "off" &&
        pbe.tabi_accepted_miss_diagnostics != "summary" &&
        pbe.tabi_accepted_miss_diagnostics != "sampled-detail") {
      TitanUI::print_error(
          "PBE: -pbe-bie-accepted-miss-diagnostics must be off, summary, or sampled-detail.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_accepted_miss_absolute_floor) ||
        pbe.tabi_accepted_miss_absolute_floor < 0.0 ||
        pbe.tabi_accepted_miss_absolute_floor > 1.0e-4) {
      TitanUI::print_error(
          "PBE: -pbe-bie-accepted-miss-absolute-floor must be between 0.0 and 1.0e-4.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_convergence_k3_accepted_rho_cap) ||
        pbe.tabi_convergence_k3_accepted_rho_cap < 0.0 ||
        pbe.tabi_convergence_k3_accepted_rho_cap > 2.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-convergence-k3-accepted-rho-cap must be between 0.0 and 2.0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_mac_distance_policy != "center" &&
        pbe.tabi_mac_distance_policy != "surface-margin") {
      TitanUI::print_error(
          "PBE: -pbe-bie-mac-distance-policy must be center or surface-margin.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_mac_surface_margin_factor) ||
        pbe.tabi_mac_surface_margin_factor < 0.0 ||
        pbe.tabi_mac_surface_margin_factor > 0.75) {
      TitanUI::print_error(
          "PBE: -pbe-bie-mac-surface-margin-factor must be between 0.0 and 0.75.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_mac_surface_margin_policy != "all" &&
        pbe.tabi_mac_surface_margin_policy != "adaptive" &&
        pbe.tabi_mac_surface_margin_policy != "risk-scaled") {
      TitanUI::print_error(
          "PBE: -pbe-bie-mac-surface-margin-policy must be all, adaptive, or risk-scaled.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_mac_surface_margin_aspect_threshold) ||
        pbe.tabi_mac_surface_margin_aspect_threshold < 1.0 ||
        pbe.tabi_mac_surface_margin_aspect_threshold > 50.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-mac-surface-margin-aspect-threshold must be between 1.0 and 50.0.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_mac_surface_margin_rho_trigger) ||
        pbe.tabi_mac_surface_margin_rho_trigger < 0.0 ||
        pbe.tabi_mac_surface_margin_rho_trigger > 2.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-mac-surface-margin-rho-trigger must be between 0.0 and 2.0.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_mac_surface_margin_soft_aspect_threshold) ||
        pbe.tabi_mac_surface_margin_soft_aspect_threshold < 0.0 ||
        pbe.tabi_mac_surface_margin_soft_aspect_threshold > 50.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-mac-surface-margin-soft-aspect-threshold must be between 0.0 and 50.0.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_mac_surface_margin_soft_rho_trigger) ||
        pbe.tabi_mac_surface_margin_soft_rho_trigger < 0.0 ||
        pbe.tabi_mac_surface_margin_soft_rho_trigger > 2.0) {
      TitanUI::print_error(
          "PBE: -pbe-bie-mac-surface-margin-soft-rho-trigger must be between 0.0 and 2.0.");
      exit(EXIT_FAILURE);
    }
    if (!std::isfinite(pbe.tabi_mac_surface_margin_soft_factor) ||
        pbe.tabi_mac_surface_margin_soft_factor < 0.0 ||
        pbe.tabi_mac_surface_margin_soft_factor > 0.75) {
      TitanUI::print_error(
          "PBE: -pbe-bie-mac-surface-margin-soft-factor must be between 0.0 and 0.75.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_expansion_audit_channel != "off" &&
        pbe.tabi_expansion_audit_channel != "summary" &&
        pbe.tabi_expansion_audit_channel != "sampled-detail") {
      TitanUI::print_error(
          "PBE: -pbe-bie-expansion-audit-channel must be off, summary, or sampled-detail.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_pre_solver_progress != "off" &&
        pbe.tabi_pre_solver_progress != "on") {
      TitanUI::print_error(
          "PBE: -pbe-bie-pre-solver-progress must be off or on.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_progress_interval < 1 ||
        pbe.tabi_progress_interval > 300) {
      TitanUI::print_error(
          "PBE: -pbe-bie-progress-interval must be between 1 and 300 seconds.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_gmres_restart < 8 || pbe.tabi_gmres_restart > 128) {
      TitanUI::print_error(
          "PBE: -pbe-bie-gmres-restart must be between 8 and 128.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_true_residual_probe != "restart" &&
        pbe.tabi_true_residual_probe != "interval" &&
        pbe.tabi_true_residual_probe != "off") {
      TitanUI::print_error(
          "PBE: -pbe-bie-true-residual-probe must be restart, interval, or off.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_true_residual_probe_scope != "full" &&
        pbe.tabi_true_residual_probe_scope != "sampled" &&
        pbe.tabi_true_residual_probe_scope != "adaptive") {
      TitanUI::print_error(
          "PBE: -pbe-bie-true-residual-probe-scope must be full, sampled, or adaptive.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_true_residual_probe_interval < 1 ||
        pbe.tabi_true_residual_probe_interval > 128) {
      TitanUI::print_error(
          "PBE: -pbe-bie-true-residual-probe-interval must be between 1 and 128.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_true_residual_sample_count < 128 ||
        pbe.tabi_true_residual_sample_count > 4096) {
      TitanUI::print_error(
          "PBE: -pbe-bie-true-residual-sample-count must be between 128 and 4096.");
      exit(EXIT_FAILURE);
    }
    if (pbe.tabi_energy_projection != "boundary" &&
        pbe.tabi_energy_projection != "born-check" &&
        pbe.tabi_energy_projection != "charge-potential-debug") {
      TitanUI::print_error(
          "PBE: -pbe-bie-energy-projection must be boundary, born-check, or charge-potential-debug.");
      exit(EXIT_FAILURE);
    }
    if (flag_nocal) {
      TitanUI::print_error(
          "PBE: -pbe requires charge/radius initialization; remove -nocal for "
          "the v1 frame-based solver.");
      exit(EXIT_FAILURE);
    }
    if (pbe.scale <= 0.0) {
      TitanUI::print_error("PBE: -pbe-scale must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.margin < 0.0) {
      TitanUI::print_error("PBE: -pbe-margin must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.int_diel <= 0.0 || pbe.ext_diel <= 0.0) {
      TitanUI::print_error(
          "PBE: -pbe-int-diel and -pbe-ext-diel must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.salt < 0.0) {
      TitanUI::print_error("PBE: -pbe-salt must be >= 0.");
      exit(EXIT_FAILURE);
    }
    const bool fmm_native_coupled_preconditioner =
        pbe.backend == "tabi" && pbe.tabi_sum == "treecode" &&
        pbe.tabi_fmm_m2l == "single-matvec-analytic" &&
        (pbe.tabi_precond == "fmm-block-jacobi" ||
         pbe.tabi_precond == "fmm-overlap-schwarz" ||
         pbe.tabi_precond == "fmm-nearfield-ilu0" ||
         pbe.tabi_precond == "fmm-nearfield-ilu0-coarse" ||
         pbe.tabi_precond == "distance-knn-overlap-schwarz" ||
         pbe.tabi_precond == "distance-knn-ilu0");
    if (fmm_native_coupled_preconditioner && pbe.salt <= 0.0) {
      TitanUI::print_error(
          "PBE: FMM-native BIE preconditioners require positive ionic strength because they operate on the coupled JCP system; use -pbe-salt>0 or select a Poisson-compatible non-FMM preconditioner.");
      exit(EXIT_FAILURE);
    }
    if (pbe.temp <= 0.0) {
      TitanUI::print_error("PBE: -pbe-temp must be > 0 K.");
      exit(EXIT_FAILURE);
    }
    if (pbe.model != "linear" && pbe.model != "nonlinear") {
      TitanUI::print_error("PBE: -pbe-model must be linear or nonlinear.");
      exit(EXIT_FAILURE);
    }
    if (pbe.biomodel != "pbe" && pbe.biomodel != "rpbe") {
      TitanUI::print_error("PBE: -pbe-biomodel must be pbe or rpbe.");
      exit(EXIT_FAILURE);
    }
    if (pbe.dielectric_model != "two-dielectric" &&
        pbe.dielectric_model != "gaussian") {
      TitanUI::print_error(
          "PBE: -pbe-dielectric must be two-dielectric or gaussian.");
      exit(EXIT_FAILURE);
    }
    if (pbe.boundary != "coulombic" && pbe.boundary != "dipolar" &&
        pbe.boundary != "zero") {
      TitanUI::print_error(
          "PBE: -pbe-boundary must be coulombic, dipolar, or zero.");
      exit(EXIT_FAILURE);
    }
    if (pbe.charge_assignment != "trilinear" &&
        pbe.charge_assignment != "delphi" &&
        pbe.charge_assignment != "gaussian") {
      TitanUI::print_error(
          "PBE: -pbe-charge-assignment must be trilinear, delphi, or gaussian.");
      exit(EXIT_FAILURE);
    }
    if (pbe.surface != "vdw" && pbe.surface != "sas" &&
        pbe.surface != "ses" && pbe.surface != "delphi-ses" &&
        pbe.surface != "delphi-vdw" && pbe.surface != "gaussian" &&
        pbe.surface != "gaussian-cutoff") {
      TitanUI::print_error(
          "PBE: -pbe-surface must be vdw, sas, ses, delphi-ses, delphi-vdw, gaussian, or gaussian-cutoff.");
      exit(EXIT_FAILURE);
    }
    if (pbe.radius_convention != "auto" &&
        pbe.radius_convention != "native" &&
        pbe.radius_convention != "delphi-bondi") {
      TitanUI::print_error(
          "PBE: -pbe-radius-convention must be auto, native, or delphi-bondi.");
      exit(EXIT_FAILURE);
    }
    if (pbe.probe_radius < 0.0) {
      TitanUI::print_error("PBE: -pbe-probe must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.ion_access != "dielectric" && pbe.ion_access != "stern") {
      TitanUI::print_error(
          "PBE: -pbe-ion-access must be dielectric or stern.");
      exit(EXIT_FAILURE);
    }
    if (pbe.ion_radius < 0.0) {
      TitanUI::print_error("PBE: -pbe-ion-radius must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_max_iter <= 0) {
      TitanUI::print_error("PBE: -pbe-nl-max-iter must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_inner_iter <= 0) {
      TitanUI::print_error("PBE: -pbe-nl-inner-iter must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_stagnation <= 0) {
      TitanUI::print_error("PBE: -pbe-nl-stagnation must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_block <= 0) {
      TitanUI::print_error("PBE: -pbe-nl-block must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_dphi <= 0.0) {
      TitanUI::print_error("PBE: -pbe-nl-dphi must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_relax != "auto") {
      if (pbe.nl_relax_value < 0.05 || pbe.nl_relax_value > 1.0) {
        TitanUI::print_error(
            "PBE: -pbe-nl-relax must be auto or a value in [0.05, 1.0].");
        exit(EXIT_FAILURE);
      }
    }
    if (pbe.nl_clip <= 0.0) {
      TitanUI::print_error("PBE: -pbe-nl-clip must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_continuation != "off" && pbe.nl_continuation != "auto") {
      TitanUI::print_error(
          "PBE: -pbe-nl-continuation must be off or auto.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_line_search != "off" && pbe.nl_line_search != "auto") {
      TitanUI::print_error(
          "PBE: -pbe-nl-line-search must be off or auto.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_residual <= 0.0) {
      TitanUI::print_error("PBE: -pbe-nl-residual must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_solver != "picard" && pbe.nl_solver != "newton") {
      TitanUI::print_error("PBE: -pbe-nl-solver must be picard or newton.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_inner != "auto" && pbe.nl_inner != "sor" &&
        pbe.nl_inner != "vcycle") {
      TitanUI::print_error("PBE: -pbe-nl-inner must be auto, sor, or vcycle.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nl_multigrid != "off" && pbe.nl_multigrid != "fas") {
      TitanUI::print_error("PBE: -pbe-nl-multigrid must be off or fas.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nwt_kernel != "auto" && pbe.nwt_kernel != "legacy" &&
        pbe.nwt_kernel != "delphi-half") {
      TitanUI::print_error(
          "PBE: -pbe-nwt-kernel must be auto, legacy, or delphi-half.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nwt_homo_kernel != "auto" &&
        pbe.nwt_homo_kernel != "packed-list" &&
        pbe.nwt_homo_kernel != "row-span" &&
        pbe.nwt_homo_kernel != "avx2" &&
        pbe.nwt_homo_kernel != "salt-tiled") {
      TitanUI::print_error("PBE: -pbe-nwt-homo-kernel must be auto, "
                           "packed-list, row-span, avx2, or salt-tiled.");
      exit(EXIT_FAILURE);
    }
    if (pbe.nwt_progress != "auto" && pbe.nwt_progress != "off" &&
        pbe.nwt_progress != "strict") {
      TitanUI::print_error(
          "PBE: -pbe-nwt-progress must be auto, off, or strict.");
      exit(EXIT_FAILURE);
    }
    if (pbe.linear_kernel != "auto" && pbe.linear_kernel != "legacy" &&
        pbe.linear_kernel != "half-packed") {
      TitanUI::print_error(
          "PBE: -pbe-linear-kernel must be auto, legacy, or half-packed.");
      exit(EXIT_FAILURE);
    }
    if (pbe.stability_audit != "off" && pbe.stability_audit != "summary" &&
        pbe.stability_audit != "detail") {
      TitanUI::print_error(
          "PBE: -pbe-stability-audit must be off, summary, or detail.");
      exit(EXIT_FAILURE);
    }
    if (pbe.stability_mode != "off" && pbe.stability_mode != "diagnose" &&
        pbe.stability_mode != "anchor" &&
        pbe.stability_mode != "hysteresis" &&
        pbe.stability_mode != "ensemble") {
      TitanUI::print_error("PBE: -pbe-stability-mode must be off, diagnose, "
                           "anchor, hysteresis, or ensemble.");
      exit(EXIT_FAILURE);
    }
    if (pbe.grid_anchor != "auto" && pbe.grid_anchor != "per-frame" &&
        pbe.grid_anchor != "first-frame" &&
        pbe.grid_anchor != "fixed-center") {
      TitanUI::print_error(
          "PBE: -pbe-grid-anchor must be auto, per-frame, first-frame, or fixed-center.");
      exit(EXIT_FAILURE);
    }
    if (pbe.grid_anchor == "fixed-center" && !pbe.test_center_user &&
        !pbe.focus_center_user) {
      TitanUI::print_error(
          "PBE: -pbe-grid-anchor=fixed-center requires -pbe-test-center or -pbe-focus-center.");
      exit(EXIT_FAILURE);
    }
    if (pbe.gaussian_exp <= 0.0 || pbe.gaussian_sigma <= 0.0) {
      TitanUI::print_error(
          "PBE: -pbe-gaussian-exp and -pbe-gaussian-sigma must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.density_cutoff < 0.0 || pbe.density_cutoff > 1.0) {
      TitanUI::print_error("PBE: -pbe-density-cutoff must be in [0, 1].");
      exit(EXIT_FAILURE);
    }
    if (pbe.surface_cutoff != "auto" &&
        (pbe.surface_cutoff_value < 0.0 || pbe.surface_cutoff_value > 1.0)) {
      TitanUI::print_error(
          "PBE: -pbe-surface-cutoff must be auto or a value in [0, 1].");
      exit(EXIT_FAILURE);
    }
    if (pbe.gap_diel <= 0.0) {
      TitanUI::print_error("PBE: -pbe-gap-diel must be > 0.");
      exit(EXIT_FAILURE);
    }
    if ((pbe.test_grid_size > 0 || pbe.test_center_user) &&
        pbe.test_pqr.empty()) {
      TitanUI::print_error(
          "PBE: -pbe-test-grid-size and -pbe-test-center require "
          "-pbe-test-pqr.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_grid_size != 0 && pbe.test_grid_size < 3) {
      TitanUI::print_error("PBE: -pbe-test-grid-size must be >= 3.");
      exit(EXIT_FAILURE);
    }
    if (pbe.parity_dump != "off" && pbe.parity_dump != "summary" &&
        pbe.parity_dump != "points" && pbe.parity_dump != "all") {
      TitanUI::print_error(
          "PBE: -pbe-parity-dump must be off, summary, points, or all.");
      exit(EXIT_FAILURE);
    }
    if (pbe.parity_map_dump != "off" && pbe.parity_map_dump != "samples" &&
        pbe.parity_map_dump != "full") {
      TitanUI::print_error(
          "PBE: -pbe-parity-map-dump must be off, samples, or full.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_charge_grid != "native" &&
        pbe.test_charge_grid != "oracle") {
      TitanUI::print_error(
          "PBE: -pbe-test-charge-grid must be native or oracle.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_charge_grid_diff != "off" &&
        pbe.test_charge_grid_diff != "samples" &&
        pbe.test_charge_grid_diff != "full") {
      TitanUI::print_error(
          "PBE: -pbe-test-charge-grid-diff must be off, samples, or full.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_surface_map != "native" &&
        pbe.test_surface_map != "oracle") {
      TitanUI::print_error(
          "PBE: -pbe-test-surface-map must be native or oracle.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_midpoint_epsilon != "native" &&
        pbe.test_midpoint_epsilon != "oracle") {
      TitanUI::print_error(
          "PBE: -pbe-test-midpoint-epsilon must be native or oracle.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_midpoint_diff != "off" &&
        pbe.test_midpoint_diff != "samples" &&
        pbe.test_midpoint_diff != "full") {
      TitanUI::print_error(
          "PBE: -pbe-test-midpoint-diff must be off, samples, or full.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_boundary_map != "native" &&
        pbe.test_boundary_map != "oracle") {
      TitanUI::print_error(
          "PBE: -pbe-test-boundary-map must be native or oracle.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_boundary_convention != "native-real" &&
        pbe.test_boundary_convention != "delphi-grid" &&
        pbe.test_boundary_convention != "oracle") {
      TitanUI::print_error(
          "PBE: -pbe-test-boundary-convention must be native-real, delphi-grid, or oracle.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_field_units != "native" &&
        pbe.test_field_units != "delphi-dimensionless") {
      TitanUI::print_error(
          "PBE: -pbe-test-field-units must be native or delphi-dimensionless.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_operator_convention != "native" &&
        pbe.test_operator_convention != "delphi") {
      TitanUI::print_error(
          "PBE: -pbe-test-operator-convention must be native or delphi.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_rhs_map != "native" && pbe.test_rhs_map != "oracle") {
      TitanUI::print_error(
          "PBE: -pbe-test-rhs-map must be native or oracle.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_screen_map != "native" && pbe.test_screen_map != "oracle") {
      TitanUI::print_error(
          "PBE: -pbe-test-screen-map must be native or oracle.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_denom_map != "native" && pbe.test_denom_map != "oracle") {
      TitanUI::print_error(
          "PBE: -pbe-test-denom-map must be native or oracle.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_boundary_convention == "oracle")
      pbe.test_boundary_map = "oracle";
    if (pbe.test_nl_source != "titan" &&
        pbe.test_nl_source != "delphi-dimensionless") {
      TitanUI::print_error(
          "PBE: -pbe-test-nl-source must be titan or delphi-dimensionless.");
      exit(EXIT_FAILURE);
    }
    if (pbe.test_energy_convention != "auto" &&
        pbe.test_energy_convention != "atom" &&
        pbe.test_energy_convention != "grid" &&
        pbe.test_energy_convention != "delphi-corrected" &&
        pbe.test_energy_convention != "delphi-iscm") {
      TitanUI::print_error(
          "PBE: -pbe-test-energy-convention must be auto, atom, grid, delphi-corrected, or delphi-iscm.");
      exit(EXIT_FAILURE);
    }
    if ((pbe.parity_dump != "off" || pbe.parity_map_dump != "off" ||
         pbe.test_charge_grid != "native" ||
         pbe.test_charge_grid_diff != "off" ||
         pbe.test_surface_map != "native" ||
         pbe.test_midpoint_epsilon != "native" ||
         pbe.test_midpoint_diff != "off" ||
         pbe.test_boundary_map != "native" ||
         pbe.test_boundary_convention != "native-real" ||
         pbe.test_field_units != "native" ||
         pbe.test_operator_convention != "native" ||
         pbe.test_rhs_map != "native" || pbe.test_screen_map != "native" ||
         pbe.test_denom_map != "native" ||
         pbe.nwt_kernel != "auto" || pbe.nwt_homo_kernel != "auto" ||
         pbe.nwt_progress != "auto" || pbe.linear_kernel != "auto" ||
         pbe.test_nl_source != "titan" || pbe.test_newton_derivative ||
         pbe.test_energy_convention != "auto") &&
        pbe.test_pqr.empty()) {
      TitanUI::print_error(
          "PBE: hidden parity diagnostics require -pbe-test-pqr.");
      exit(EXIT_FAILURE);
    }
    if (pbe.pbsa && !pbe.test_pqr.empty()) {
      TitanUI::print_error(
          "PBSA: -pbsa is not compatible with the hidden PQR fixture adapter; "
          "run it on a native two-molecule TITAN input.");
      exit(EXIT_FAILURE);
    }
    if (pbe.binding_cycle == "pqr-components") {
      if (pbe.test_pqr.empty() || pbe.binding_receptor_pqr.empty() ||
          pbe.binding_ligand_pqr.empty()) {
        TitanUI::print_error(
            "PBE: -pbe-binding-cycle=pqr-components requires "
            "-pbe-test-pqr, -pbe-binding-receptor-pqr, and "
            "-pbe-binding-ligand-pqr.");
        exit(EXIT_FAILURE);
      }
      if (pbe.model != "linear" || pbe.biomodel != "pbe" ||
          pbe.dielectric_model != "two-dielectric") {
        TitanUI::print_error(
            "PBE: pqr-components binding cycle requires linear PBE with the "
            "two-dielectric model.");
        exit(EXIT_FAILURE);
      }
      if (pbe.dgsolv || pbe.pbsa) {
        TitanUI::print_error(
            "PBE: pqr-components binding cycle cannot be combined with "
            "native -pbe-dgsolv or -pbsa endpoints.");
        exit(EXIT_FAILURE);
      }
    } else if (!pbe.binding_receptor_pqr.empty() ||
               !pbe.binding_ligand_pqr.empty()) {
      TitanUI::print_error(
          "PBE: binding component PQR options require "
          "-pbe-binding-cycle=pqr-components.");
      exit(EXIT_FAILURE);
    }
    if (pbe.pbsa_components != "auto" &&
        pbe.pbsa_components != "complex,receptor,ligand") {
      TitanUI::print_error(
          "PBSA: -pbsa-components must be auto or complex,receptor,ligand.");
      exit(EXIT_FAILURE);
    }
    if (pbe.pbsa_decomp != "off" && pbe.pbsa_decomp != "molecule" &&
        pbe.pbsa_decomp != "residue" && pbe.pbsa_decomp != "hybrid" &&
        pbe.pbsa_decomp != "auto") {
      TitanUI::print_error(
          "PBSA: -pbsa-decomp must be off, molecule, residue, hybrid, or auto.");
      exit(EXIT_FAILURE);
    }
    const bool pbsa_mol1_lig =
        (mol_1_type == "ligand" || mol_1_type == "lig");
    const bool pbsa_mol2_lig =
        (mol_2_type == "ligand" || mol_2_type == "lig");
    if (pbe.pbsa && pbe.pbsa_decomp == "hybrid" &&
        pbsa_mol1_lig == pbsa_mol2_lig) {
      TitanUI::print_error(
          "PBSA: -pbsa-decomp=hybrid requires exactly one ligand molecule "
          "(mol type ligand or lig).");
      exit(EXIT_FAILURE);
    }
    if (pbe.pbsa_residue_frames != "off" &&
        pbe.pbsa_residue_frames != "all") {
      TitanUI::print_error(
          "PBSA: -pbsa-residue-frames must be off or all.");
      exit(EXIT_FAILURE);
    }
    if (pbe.pbsa_ie_block != "auto") {
      try {
        const int block_size = std::stoi(pbe.pbsa_ie_block);
        if (block_size <= 0) {
          TitanUI::print_error(
              "PBSA: -pbsa-ie-block must be auto or a positive integer.");
          exit(EXIT_FAILURE);
        }
      } catch (...) {
        TitanUI::print_error(
            "PBSA: -pbsa-ie-block must be auto or a positive integer.");
        exit(EXIT_FAILURE);
      }
    }
    if (pbe.pbsa_ie_min_samples <= 0) {
      TitanUI::print_error("PBSA: -pbsa-ie-min-samples must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.pbsa_ie_filter_sigma < 0.0) {
      TitanUI::print_error("PBSA: -pbsa-ie-filter-sigma must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.pbsa_perf != "off" && pbe.pbsa_perf != "summary" &&
        pbe.pbsa_perf != "detail") {
      TitanUI::print_error("PBSA: -pbsa-perf must be off, summary, or detail.");
      exit(EXIT_FAILURE);
    }
    if (pbe.pbsa_grid_reuse != "off" && pbe.pbsa_grid_reuse != "validate" &&
        pbe.pbsa_grid_reuse != "auto") {
      TitanUI::print_error(
          "PBSA: -pbsa-grid-reuse must be off, validate, or auto.");
      exit(EXIT_FAILURE);
    }
    if (pbe.pbsa_sasa_points < 24 || pbe.pbsa_sasa_points > 8192) {
      TitanUI::print_error(
          "PBSA: -pbsa-sasa-points must be in [24, 8192].");
      exit(EXIT_FAILURE);
    }
    if (pbe.timing != "off" && pbe.timing != "summary" &&
        pbe.timing != "detail") {
      TitanUI::print_error(
          "PBE: -pbe-timing must be off, summary, or detail.");
      exit(EXIT_FAILURE);
    }
    if (pbe.max_iter <= 0) {
      TitanUI::print_error("PBE: -pbe-max-iter must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.rms_tol <= 0.0 || pbe.dphi_tol <= 0.0) {
      TitanUI::print_error("PBE: convergence tolerances must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.max_grid_points < 27) {
      TitanUI::print_error("PBE: -pbe-max-grid must allow at least 27 points.");
      exit(EXIT_FAILURE);
    }
    if (pbe.omega != "auto" &&
        (pbe.omega_value <= 0.0 || pbe.omega_value >= 2.0)) {
      TitanUI::print_error("PBE: -pbe-omega must be auto or in (0, 2).");
      exit(EXIT_FAILURE);
    }
    if (pbe.focus != "off" && pbe.focus != "auto" &&
        pbe.focus != "delphi") {
      TitanUI::print_error("PBE: -pbe-focus must be off, auto, or delphi.");
      exit(EXIT_FAILURE);
    }
    if (pbe.multigrid != "off" && pbe.multigrid != "focus" &&
        pbe.multigrid != "auto" && pbe.multigrid != "vcycle") {
      TitanUI::print_error(
          "PBE: -pbe-multigrid must be off, focus, auto, or vcycle.");
      exit(EXIT_FAILURE);
    }
    if (pbe.focus_levels < 1 || pbe.focus_levels > 4) {
      TitanUI::print_error("PBE: -pbe-focus-levels must be in [1, 4].");
      exit(EXIT_FAILURE);
    }
    if (pbe.focus_margin < 0.0) {
      TitanUI::print_error("PBE: -pbe-focus-margin must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.mg_levels != "auto") {
      try {
        const int parsed_mg_levels = std::stoi(pbe.mg_levels);
        if (parsed_mg_levels < 1 || parsed_mg_levels > 4) {
          TitanUI::print_error("PBE: -pbe-mg-levels must be auto or in [1, 4].");
          exit(EXIT_FAILURE);
        }
      } catch (...) {
        TitanUI::print_error("PBE: -pbe-mg-levels must be auto or an integer.");
        exit(EXIT_FAILURE);
      }
    }
    if (pbe.mg_pre < 0 || pbe.mg_post < 0 || pbe.mg_coarse_iter <= 0) {
      TitanUI::print_error(
          "PBE: -pbe-mg-pre/post must be >= 0 and -pbe-mg-coarse-iter "
          "must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (pbe.backend != "tabi" && pbe.tabi_surface_output != "off") {
      TitanUI::print_error(
          "PBE: -pbe-bie-surface-output requires -pbe=bie.");
      exit(EXIT_FAILURE);
    }
    if (pbe.backend == "tabi") {
      if (pbe.model != "linear") {
        TitanUI::print_error(
            "PBE: -pbe=bie v1 supports only -pbe-model=linear.");
        exit(EXIT_FAILURE);
      }
      if (pbe.biomodel != "pbe") {
        TitanUI::print_error(
            "PBE: -pbe=bie v1 supports only -pbe-biomodel=pbe.");
        exit(EXIT_FAILURE);
      }
      if (pbe.dielectric_model != "two-dielectric") {
        TitanUI::print_error(
            "PBE: -pbe=bie v1 supports only -pbe-dielectric=two-dielectric.");
        exit(EXIT_FAILURE);
      }
      if (pbe.surface != "vdw" && pbe.surface != "sas") {
        TitanUI::print_error(
            "PBE: -pbe=bie v1 supports only -pbe-surface=vdw or sas.");
        exit(EXIT_FAILURE);
      }
      if (pbe.charge_assignment == "gaussian") {
        TitanUI::print_error(
            "PBE: -pbe=bie v1 does not support Gaussian charge assignment.");
        exit(EXIT_FAILURE);
      }
      if (pbe.focus != "off" || pbe.multigrid != "off") {
        TitanUI::print_error(
            "PBE: -pbe=bie v1 does not support focus or multigrid options.");
        exit(EXIT_FAILURE);
      }
      if (pbe.pbsa && pbe.pbsa_grid_reuse == "validate") {
        TitanUI::print_error(
            "PBSA: -pbsa-grid-reuse=validate is available only for the grid "
            "backend; BIE computes independent boundary-integral endpoint "
            "solves.");
        exit(EXIT_FAILURE);
      }
      if (pbe.map != "off") {
        TitanUI::print_error(
            "PBE: -pbe=bie v1 does not emit grid maps; use -pbe-map=off.");
        exit(EXIT_FAILURE);
      }
      if (pbe.parity_dump != "off" || pbe.parity_map_dump != "off" ||
          pbe.test_charge_grid != "native" ||
          pbe.test_charge_grid_diff != "off" ||
          pbe.test_surface_map != "native" ||
          pbe.test_midpoint_epsilon != "native" ||
          pbe.test_midpoint_diff != "off" ||
          pbe.test_boundary_map != "native" ||
          pbe.test_boundary_convention != "native-real" ||
          pbe.test_field_units != "native" ||
          pbe.test_operator_convention != "native" ||
          pbe.test_rhs_map != "native" ||
          pbe.test_screen_map != "native" ||
          pbe.test_denom_map != "native") {
        TitanUI::print_error(
            "PBE: -pbe=bie v1 does not support grid parity diagnostics.");
        exit(EXIT_FAILURE);
      }
    }
  }

  if (flag_listfile == true) {
    fstream list(list_filename, ios::in);
    if (!list.is_open()) {
      cout << "Cannot open " << list_filename << "!\n";
      flag_listfile = false;
      return 1;
    }
    list.close();
  }

  if (lj.flag_lj) {
    if (lj.lj_cluster_cutoff <= 0.0) {
      TitanUI::print_error("LJ: -lj-cut must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (lj.lj_min_cluster_size <= 0) {
      TitanUI::print_error("LJ: -lj-min-cluster must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (lj.lj_rho_bin <= 0.0) {
      TitanUI::print_error("LJ: -lj-rho-bin must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (lj.lj_rho_max < 0.0) {
      TitanUI::print_error("LJ: -lj-rho-max must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (lj.lj_solid_q6 < 0.0 || lj.lj_solid_q6 > 1.0) {
      TitanUI::print_error("LJ: -lj-solid-q6 must be in [0, 1].");
      exit(EXIT_FAILURE);
    }
    if (lj.lj_solid_min_cn <= 0) {
      TitanUI::print_error("LJ: -lj-solid-min-cn must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (lj.lj_crystal_tol <= 0.0) {
      TitanUI::print_error("LJ: -lj-crystal-tol must be > 0.");
      exit(EXIT_FAILURE);
      }
  }
  if (hbk.flag_hbk) {
    if (hbk.hbk_maxlag <= 0) {
      TitanUI::print_error("HBond kinetics: -hbk-maxlag must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (!hbk.hbk_hbond_cutoff_explicit)
      hbk.hbk_hbond_cutoff = dis_h;
    if (!hbk.hbk_angle_explicit)
      hbk.hbk_angle_deg = angle_h;
    if (!hbk.hbk_water_cutoff_explicit)
      hbk.hbk_water_cutoff = d_crit_water;
    if (hbk.hbk_hbond_cutoff <= 0.0) {
      TitanUI::print_error("HBond kinetics: -hbk-hbond-cutoff must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (hbk.hbk_angle_deg <= 0.0 || hbk.hbk_angle_deg >= 180.0) {
      TitanUI::print_error("HBond kinetics: -hbk-angle must be in (0, 180).");
      exit(EXIT_FAILURE);
    }
    if (hbk.hbk_salt_cutoff <= 0.0) {
      TitanUI::print_error("HBond kinetics: -hbk-salt-cutoff must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (hbk.hbk_water_cutoff <= 0.0) {
      TitanUI::print_error("HBond kinetics: -hbk-water-cutoff must be > 0.");
      exit(EXIT_FAILURE);
    }

    std::string domain = hbk.hbk_domain;
    std::transform(domain.begin(), domain.end(), domain.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (domain != "inter" && domain != "intra" && domain != "both") {
      TitanUI::print_error(
          "HBond kinetics: -hbk-domain must be one of inter, intra, both.");
      exit(EXIT_FAILURE);
    }
    hbk.hbk_domain = domain;

    std::vector<std::string> normalized_kinds;
    normalized_kinds.reserve(hbk.hbk_kinds.size());
    bool use_all = hbk.hbk_kinds.empty();
    for (const std::string &raw_kind : hbk.hbk_kinds) {
      std::string kind = raw_kind;
      std::transform(kind.begin(), kind.end(), kind.begin(),
                     [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                     });
      if (kind.empty())
        continue;
      if (kind == "all") {
        use_all = true;
        break;
      }
      if (kind != "hbond" && kind != "salt" && kind != "water") {
        TitanUI::print_error(
            "HBond kinetics: -hbk-kind must be one of all, hbond, salt, water.");
        exit(EXIT_FAILURE);
      }
      if (std::find(normalized_kinds.begin(), normalized_kinds.end(), kind) ==
          normalized_kinds.end()) {
        normalized_kinds.push_back(kind);
      }
    }
    if (use_all)
      normalized_kinds = {"hbond", "salt", "water"};
    hbk.hbk_kinds = std::move(normalized_kinds);
  }
  if (sq.flag_sq) {
    if (sq.sq_rdf_bin <= 0.0) {
      TitanUI::print_error("SQ: -sq-rdf-bin must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (!sq.sq_rdf_max_auto && sq.sq_rdf_max <= 0.0) {
      TitanUI::print_error("SQ: -sq-rdf-max must be > 0 or auto.");
      exit(EXIT_FAILURE);
    }
    if (!sq.sq_rdf_max_auto && sq.sq_rdf_max < sq.sq_rdf_bin) {
      TitanUI::print_error("SQ: -sq-rdf-max must be >= -sq-rdf-bin.");
      exit(EXIT_FAILURE);
    }
    if (sq.sq_q_bin <= 0.0) {
      TitanUI::print_error("SQ: -sq-q-bin must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (!sq.sq_q_max_auto && sq.sq_q_max <= 0.0) {
      TitanUI::print_error("SQ: -sq-q-max must be > 0 or auto.");
      exit(EXIT_FAILURE);
    }
    if (!sq.sq_q_max_auto && sq.sq_q_max < sq.sq_q_bin) {
      TitanUI::print_error("SQ: -sq-q-max must be >= -sq-q-bin.");
      exit(EXIT_FAILURE);
    }

    std::vector<std::string> normalized_channels;
    normalized_channels.reserve(sq.sq_channels.size());
    for (const std::string &raw_channel : sq.sq_channels) {
      std::string channel = raw_channel;
      std::transform(channel.begin(), channel.end(), channel.begin(),
                     [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                     });
      if (channel.empty())
        continue;
      if (channel != "all" && channel != "trace" && channel != "backbone" &&
          channel != "sticker") {
        TitanUI::print_error(
            "SQ: -sq-channel must be one of all, trace, backbone, sticker.");
        exit(EXIT_FAILURE);
      }
      if (std::find(normalized_channels.begin(), normalized_channels.end(),
                    channel) == normalized_channels.end()) {
        normalized_channels.push_back(channel);
      }
    }
    if (normalized_channels.empty())
      normalized_channels.push_back("all");
    sq.sq_channels = std::move(normalized_channels);
  }

  if (corr_knn.flag_corr_knn) {
    flag_network = true;
    if (corr_knn.estimator_k <= 0) {
      TitanUI::print_error("Corr-kNN: -corr-knn-k must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (corr_knn.tau_lag <= 0) {
      TitanUI::print_error("Corr-kNN: -corr-knn-tau must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (corr_knn.pair_cutoff_ang < 0.0) {
      TitanUI::print_error("Corr-kNN: -corr-knn-pair-cut must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (corr_knn.pair_topk < 0) {
      TitanUI::print_error("Corr-kNN: -corr-knn-pair-topk must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (corr_knn.pair_use_contact_prior &&
        corr_knn.pair_contact_cutoff_ang <= 0.0) {
      TitanUI::print_error(
          "Corr-kNN: -corr-knn-pair-contact-cut must be > 0.");
        exit(EXIT_FAILURE);
    }
  }

  if (validation.flag_validation) {
    flag_network = true;
    msm.flag_tica = true;
    msm.flag_msm = true;

    auto normalize_positive_int_list =
        [](std::vector<int> &values, const char *flag_name) {
          std::vector<int> normalized;
          normalized.reserve(values.size());
          for (int value : values) {
            if (value <= 0) {
              TitanUI::print_error(std::string("Validation: ") + flag_name +
                                   " entries must be > 0.");
              exit(EXIT_FAILURE);
            }
            if (std::find(normalized.begin(), normalized.end(), value) ==
                normalized.end()) {
              normalized.push_back(value);
            }
          }
          values = std::move(normalized);
        };

    normalize_positive_int_list(validation.ck_multiples,
                                "-validation-ck-multiples");
    if (!validation.vamp_components.empty()) {
      normalize_positive_int_list(validation.vamp_components,
                                  "-validation-vamp-components");
    }
    if (!validation.vamp_lags.empty()) {
      normalize_positive_int_list(validation.vamp_lags,
                                  "-validation-vamp-lags");
    }
    if (!validation.vamp_clusters.empty()) {
      normalize_positive_int_list(validation.vamp_clusters,
                                  "-validation-vamp-clusters");
    }
    if (validation.vamp_folds <= 0) {
      TitanUI::print_error("Validation: -validation-vamp-folds must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (validation.bootstrap_samples < 0) {
      TitanUI::print_error(
          "Validation: -validation-bootstrap must be >= 0.");
      exit(EXIT_FAILURE);
    }
  }

  // Run this guard after all module implication rules above. For example,
  // -validation and MBAR/TICA domains can enable MSM after initial parsing.
  if (trajectory_semantics == "remd_state") {
    if (mbar.flag_mbar && mbar.temp_mode != "fixed") {
      TitanUI::print_error(
          "MBAR: fixed-temperature REMD state streams require "
          "-mbar-temp-mode=fixed. Instantaneous kinetic temperature is not "
          "the thermodynamic-state label.");
      exit(EXIT_FAILURE);
    }
    if (dham.flag_dham) {
      TitanUI::print_error(
          "DHAM/dTRAM transition counts require contiguous trajectory "
          "segments within each thermodynamic state. Fixed-temperature REMD "
          "streams cross accepted exchanges; use MBAR for equilibrium "
          "reweighting or provide exchange-segmented continuous input.");
      exit(EXIT_FAILURE);
    }
    if (validation.flag_validation) {
      TitanUI::print_error(
          "Validation: VAMP/CK validation requires continuous trajectories. "
          "Fixed-temperature REMD state streams contain configuration swaps; "
          "remove -validation or provide continuous trajectories.");
      exit(EXIT_FAILURE);
    }
    if (hbk.flag_hbk || flag_allostery || corr_knn.flag_corr_knn) {
      TitanUI::print_error(
          "Kinetic correlation modules (-hbk, -allostery, -corr-knn) require "
          "continuous trajectories and cannot run with "
          "-trajectory-semantics=remd-state.");
      exit(EXIT_FAILURE);
    }
    if (tpt.flag_tpt && tpt.mode == "kinetic" &&
        tpt.demux_map_file.empty()) {
      TitanUI::print_error(
          "Kinetic TPT on REMD state trajectories requires "
          "-tpt-demux-map to reconstruct continuous walker trajectories.");
      exit(EXIT_FAILURE);
    }
    if (msm.flag_msm) {
      msm.msm_run_its = false;
      TitanUI::print_warning(
          "REMD state trajectories: MSM/TICA are descriptive state-space "
          "models only. Absolute timescales, ITS, CK, and kinetic claims are "
          "disabled because accepted exchanges break physical continuity.");
    }
  }

  if (flag_allostery) {
    if (allostery_tau <= 0) {
      TitanUI::print_error("Allostery: -allostery-tau must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (allostery_bins == 0) {
      TitanUI::print_error(
          "Allostery: -allostery-bins must be > 0 or -1 for auto.");
      exit(EXIT_FAILURE);
    }
    if (allostery_bins < -1) {
      TitanUI::print_error(
          "Allostery: -allostery-bins must be >= 1 or -1 for auto.");
      exit(EXIT_FAILURE);
    }
    if (allostery_cutoff < 0.0) {
      TitanUI::print_error("Allostery: -allostery-cut must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (allostery_knn < 0) {
      TitanUI::print_error("Allostery: -allostery-knn must be >= 0.");
      exit(EXIT_FAILURE);
    }
    if (allostery_knn_use_contact_prior && allostery_knn_contact_cutoff <= 0.0) {
      TitanUI::print_error(
          "Allostery: -allostery-knn-contact-cut must be > 0.");
      exit(EXIT_FAILURE);
    }
    std::string estimator = allostery_estimator;
    std::transform(estimator.begin(), estimator.end(), estimator.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (estimator == "histogram")
      estimator = "hist";
    if (estimator != "hist" && estimator != "knn" && estimator != "kde") {
      TitanUI::print_error(
          "Allostery: -allostery-estimator must be one of hist, knn, kde.");
      exit(EXIT_FAILURE);
    }
    allostery_estimator = estimator;
    if (allostery_estimator == "knn" && allostery_estimator_k <= 0) {
      TitanUI::print_error("Allostery: -allostery-estimator-k must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (allostery_kde_bandwidth < 0.0) {
      TitanUI::print_error("Allostery: -allostery-kde-bandwidth must be >= 0.");
      exit(EXIT_FAILURE);
    }
  }

  const bool lj_trace_free_kinetics =
      lj.flag_lj && (msm.flag_tica || msm.flag_msm || dham.flag_dham ||
                     tpt.flag_tpt ||
                     umap.flag_umap) &&
      !flag_network_explicit && !flag_pca && !flag_allostery &&
      !corr_knn.flag_corr_knn &&
      !community.flag_community && !lspmd.flag_lspmd;
  if (lj_trace_free_kinetics && flag_network) {
    flag_network = false;
    TitanUI::print_info(
        "[LJ] Using permutation-invariant LJ phase-2 features for "
        "TICA/MSM/UMAP/DHAM; skipping auto-enabled DCCM network setup.");
  }

  const bool has_explicit_pbc_mode =
      pbc.solv_pbcn || pbc.solv_pbctopo || pbc.solv_pbc == 1 ||
      pbc.solv_pbc_int == 1 ||
      pbc.solv_pbc_clust || pbc.flag_pbc_ref || pbc.pbc_mem ||
      pbc.pbc_envi;
  std::transform(pbc.pbctopo_anchor_backend.begin(),
                 pbc.pbctopo_anchor_backend.end(),
                 pbc.pbctopo_anchor_backend.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  std::transform(pbc.pbctopo_mode.begin(), pbc.pbctopo_mode.end(),
                 pbc.pbctopo_mode.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  std::transform(pbc.pbctopo_search_policy.begin(),
                 pbc.pbctopo_search_policy.end(),
                 pbc.pbctopo_search_policy.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  std::transform(pbc.pbctopo_anchor_metrics.begin(),
                 pbc.pbctopo_anchor_metrics.end(),
                 pbc.pbctopo_anchor_metrics.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  std::transform(pbc.pbctopo_temporal.begin(), pbc.pbctopo_temporal.end(),
                 pbc.pbctopo_temporal.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (pbc.pbctopo_mode != "vibe" && pbc.pbctopo_mode != "llps-local") {
    TitanUI::print_error(
        "PBCTOPO: -pbctopo-mode must be one of vibe, llps-local.");
    exit(EXIT_FAILURE);
  }
  if (pbc.pbctopo_search_policy != "fast-certified" &&
      pbc.pbctopo_search_policy != "audit-exact") {
    TitanUI::print_error(
        "PBCTOPO: -pbctopo-search-policy must be one of fast-certified, "
        "audit-exact.");
    exit(EXIT_FAILURE);
  }
  if (pbc.pbctopo_mode == "llps-local" &&
      pbc.pbctopo_search_policy != "fast-certified") {
    TitanUI::print_error(
        "PBCTOPO: audit-exact search policy is only valid in vibe mode.");
    exit(EXIT_FAILURE);
  }
  if (pbc.pbctopo_audit_max_states <= 0 ||
      pbc.pbctopo_audit_max_states > 1000000000) {
    TitanUI::print_error(
        "PBCTOPO: -pbctopo-audit-max-states must be in 1..1000000000.");
    exit(EXIT_FAILURE);
  }
  if (!std::isfinite(pbc.pbctopo_audit_max_seconds) ||
      pbc.pbctopo_audit_max_seconds <= 0.0 ||
      pbc.pbctopo_audit_max_seconds > 86400.0) {
    TitanUI::print_error(
        "PBCTOPO: -pbctopo-audit-max-seconds must be in (0, 86400].");
    exit(EXIT_FAILURE);
  }
  if (pbc.pbctopo_audit_max_memory_mb <= 0 ||
      pbc.pbctopo_audit_max_memory_mb > 1048576) {
    TitanUI::print_error(
        "PBCTOPO: -pbctopo-audit-max-memory-mb must be in 1..1048576.");
    exit(EXIT_FAILURE);
  }
  if (!pbc.pbctopo_required_contacts_file.empty() &&
      pbc.pbctopo_mode == "llps-local") {
    TitanUI::print_error(
        "PBCTOPO: -pbctopo-required-contacts is only valid in vibe mode.");
    exit(EXIT_FAILURE);
  }
  if (pbc.pbctopo_anchor_backend != "auto" &&
      pbc.pbctopo_anchor_backend != "prim" &&
      pbc.pbctopo_anchor_backend != "spatial-all") {
    TitanUI::print_error(
        "PBCTOPO: -pbctopo-anchor-backend must be one of auto, prim, spatial-all.");
    exit(EXIT_FAILURE);
  }
  if (pbc.pbctopo_anchor_metrics != "full" &&
      pbc.pbctopo_anchor_metrics != "mst-only" &&
      pbc.pbctopo_anchor_metrics != "no-order" &&
      pbc.pbctopo_anchor_metrics != "no-contact") {
    TitanUI::print_error(
        "PBCTOPO: -pbctopo-anchor-metrics must be one of full, mst-only, "
        "no-order, no-contact.");
    exit(EXIT_FAILURE);
  }
  if (pbc.pbctopo_temporal != "greedy" &&
      pbc.pbctopo_temporal != "viterbi" &&
      pbc.pbctopo_temporal != "hololift") {
    TitanUI::print_error(
        "PBCTOPO: -pbctopo-temporal must be one of greedy, viterbi, hololift.");
    exit(EXIT_FAILURE);
  }
  // hololift runs HoloLift-MD on the package VIBE writes, against the XTC
  // trajectory VIBE read (src/Main.cpp, src/hololift/temporal_runner.cpp).
  if (pbc.pbctopo_temporal == "hololift") {
    if (pbc.pbctopo_mode != "vibe") {
      TitanUI::print_error(
          "PBCTOPO: -pbctopo-temporal=hololift requires -pbctopo-mode=vibe.");
      exit(EXIT_FAILURE);
    }
    if (traj_file.empty() || xdr_type != "xtc") {
      TitanUI::print_error("PBCTOPO: -pbctopo-temporal=hololift reads the "
                           "trajectory again and requires -traj <file.xtc>.");
      exit(EXIT_FAILURE);
    }
  }
  if (pbc.pbctopo_temporal_k <= 0 || pbc.pbctopo_temporal_k > 32) {
    TitanUI::print_error("PBCTOPO: -pbctopo-temporal-k must be in 1..32.");
    exit(EXIT_FAILURE);
  }
  if (!std::isfinite(pbc.pbctopo_temporal_penalty) ||
      pbc.pbctopo_temporal_penalty < 0.0) {
    TitanUI::print_error(
        "PBCTOPO: -pbctopo-temporal-penalty must be finite and >= 0.");
    exit(EXIT_FAILURE);
  }
  if (!std::isfinite(pbc.pbctopo_vdw_clash_scale) ||
      pbc.pbctopo_vdw_clash_scale <= 0.0 ||
      pbc.pbctopo_vdw_clash_scale > 1.0) {
    TitanUI::print_error(
        "PBCTOPO: -pbctopo-vdw-clash-scale must be in (0, 1].");
    exit(EXIT_FAILURE);
  }
  auto require_pbctopo_positive_double = [&](bool is_set, double value,
                                             const char *flag_name) {
    if (is_set && (!std::isfinite(value) || value <= 0.0)) {
      TitanUI::print_error(std::string("PBCTOPO: ") + flag_name +
                           " must be > 0.");
      exit(EXIT_FAILURE);
    }
  };
  auto require_pbctopo_positive_int = [&](bool is_set, int value,
                                          const char *flag_name,
                                          int max_value) {
    if (is_set && value <= 0) {
      TitanUI::print_error(std::string("PBCTOPO: ") + flag_name +
                           " must be > 0.");
      exit(EXIT_FAILURE);
    }
    if (is_set && value > max_value) {
      TitanUI::print_error(std::string("PBCTOPO: ") + flag_name +
                           " must be <= " + std::to_string(max_value) + ".");
      exit(EXIT_FAILURE);
    }
  };
  require_pbctopo_positive_double(pbc.pbctopo_bond_cutoff_set,
                                  pbc.pbctopo_bond_cutoff,
                                  "-pbctopo-bond-cutoff");
  require_pbctopo_positive_double(pbc.pbctopo_clash_cutoff_set,
                                  pbc.pbctopo_clash_cutoff,
                                  "-pbctopo-clash-cutoff");
  require_pbctopo_positive_double(pbc.pbctopo_contact_cutoff_set,
                                  pbc.pbctopo_contact_cutoff,
                                  "-pbctopo-contact-cutoff");
  require_pbctopo_positive_double(pbc.pbctopo_continuity_cutoff_set,
                                  pbc.pbctopo_continuity_cutoff,
                                  "-pbctopo-continuity-cutoff");
  require_pbctopo_positive_int(pbc.pbctopo_exact_component_limit_set,
                               pbc.pbctopo_exact_component_limit,
                               "-pbctopo-exact-component-limit",
                               titan_pbctopo::kMaxExactComponentLimit);
  require_pbctopo_positive_int(pbc.pbctopo_beam_width_set,
                               pbc.pbctopo_beam_width,
                               "-pbctopo-beam-width",
                               titan_pbctopo::kMaxBeamWidth);
  require_pbctopo_positive_int(pbc.pbctopo_component_shell_radius_set,
                               pbc.pbctopo_component_shell_radius,
                               "-pbctopo-component-shell-radius", 2);
  if (rg.flag_rg2 && rg.hist_bins <= 0) {
    TitanUI::print_error("RG2: -rg2-hist-bins must be a positive integer.");
    exit(EXIT_FAILURE);
  }
  if ((flag_nocal || rms.flag_rms || rg.flag_rg2 || lj.flag_lj || sq.flag_sq ||
       interface_analysis.flag_interface_analysis) &&
      !has_explicit_pbc_mode) {
    pbc.solv_pbc_int = 1;
    pbc.solv_pbc = 0;
    pbc.chain_ref = 0;
    pbc.auto_enabled_pbci = true;
    vector<string> modules;
    if (flag_nocal)
      modules.push_back("Nocal");
    if (rms.flag_rms)
      modules.push_back("RMSD/RMSF");
    if (rg.flag_rg2)
      modules.push_back("Rg2");
    if (lj.flag_lj)
      modules.push_back("LJ");
    if (sq.flag_sq)
      modules.push_back("SQ");
    if (interface_analysis.flag_interface_analysis)
      modules.push_back("Interface");
    std::string reason;
    for (size_t i = 0; i < modules.size(); ++i) {
      if (i > 0)
        reason += (i + 1 == modules.size()) ? " and " : ", ";
      reason += modules[i];
    }
    if (modules.size() == 1 &&
        (modules.front() == "LJ" || modules.front() == "SQ"))
      reason += " analysis";
    reason += " requested without an explicit PBC mode. Auto-enabling -pbci 0.";
    TitanUI::print_warning(reason);
  }

  if (pbc.solv_pbc_int == 1 && pbc.solv_pbc_clust == true) {
    cout << "WARNING!! Both intensive pbc solver and clustered pbc solver are "
            "turned on, will turn off clustered pbc solver!\n";
    pbc.solv_pbc_clust = false;
  }
  if (flag_maxforce == true && flag_cal_force == false) {
    cout << "Note: -max-force option is on, will activate -cal-force option!\n";
    flag_cal_force = true;
  }

  if (flag_topo_1 == false && !mol_2_type_explicit)
    mol_2_type = "protein";

  if (group_1 == group_2)
    self_int = true;

  transform(eu.begin(), eu.end(), eu.begin(), ::toupper);
  if (eu != "KJ/MOL" && eu != "KCAL/MOL") {
    cout << "Energy unit is invalid, only kcal/mol and kJ/mol are supported. "
            "Will use the default unit (kJ/mol)\n";
    eu = "kJ/mol";
  } else if (eu == "KJ/MOL")
    eu = "kJ/mol";
  else {
    eu = "kcal/mol";
    eu_par = 4.184;
  }

  transform(fu.begin(), fu.end(), fu.begin(), ::toupper);
  if (fu != "PN" && fu != "KJ/MOL/NM") {
    cout << "Force unit is invalid, only pN and kJ/mol/nm are supported. Will "
            "use the default unit (pN)\n";
    fu = "pN";
  } else if (fu == "PN") {
    fu = "pN";
  } else {
    fu = "kJ/mol/nm";
    fu_par = 1.;
  }

  transform(tu.begin(), tu.end(), tu.begin(), ::toupper);
  if (tu != "PS" && tu != "NS") {
    cout << "Time unit is invalid, only ps and ns are supported. Will use the "
            "default unit (ps)\n";
  } else if (tu == "PS")
    tu = "ps";
  else
    tu = "ns";

  if (tu == "ns") {
    begin_time *= 1000.;
    end_time *= 1000.;
    dt *= 1000;
  }

  if (flag_index == false) {
    fstream index_file("index.ndx", ios::in);
    if (index_file.is_open())
      flag_index = true;
  }

  return 0;
}
