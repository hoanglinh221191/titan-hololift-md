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

arguments::OptionStep arguments::parse_options_general(int argc, char **argv,
                                                       int &i,
                                                       const std::string &arg,
                                                       std::string &common_value) {
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
    return option_consumed();
  }
  if (consume_option_value(arg, {"-s"}, common_value, i)) {
    pdb_name = common_value;
    flag_1_pdb = true;
    flag_xdr = true;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-n"}, common_value, i)) {
    index_name = common_value;
    flag_index = true;
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(arg, {"-ff"}, common_value, i)) {
    force_f = common_value;
    transform(force_f.begin(), force_f.end(), force_f.begin(), ::toupper);
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-gbsa"}, flag_gbsa, i)) {
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-cas"}, flag_cas, i)) {
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-edecomp", "--edecomp"},
                          edecomp.flag_edecomp, i)) {
    return option_consumed();
  }
  if (consume_option_value(arg, {"-edecomp-source", "--edecomp-source"},
                           common_value, i)) {
    edecomp.flag_edecomp = true;
    edecomp.source = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-edecomp-level", "--edecomp-level"},
                           common_value, i)) {
    edecomp.flag_edecomp = true;
    edecomp.level = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-edecomp-edge-top", "--edecomp-edge-top"},
                           common_value, i)) {
    edecomp.flag_edecomp = true;
    edecomp.edge_top = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-edecomp-rank-by", "--edecomp-rank-by"},
                           common_value, i)) {
    edecomp.flag_edecomp = true;
    edecomp.rank_by = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-edecomp-frame-output", "--edecomp-frame-output"},
          common_value, i)) {
    edecomp.flag_edecomp = true;
    edecomp.frame_output = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-edecomp-perf", "--edecomp-perf"},
                           common_value, i)) {
    edecomp.flag_edecomp = true;
    edecomp.perf = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-hotspot-report", "--hotspot-report"},
                          hotspot.flag_hotspot_report, i)) {
    return option_consumed();
  }
  if (consume_option_value(arg, {"-hotspot-source", "--hotspot-source"},
                           common_value, i)) {
    hotspot.flag_hotspot_report = true;
    hotspot.source = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-hotspot-rank-by", "--hotspot-rank-by"},
                           common_value, i)) {
    hotspot.flag_hotspot_report = true;
    hotspot.rank_by = lowercase_copy(common_value);
    hotspot.rank_by_explicit = true;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-perf-report", "--perf-report"},
                           common_value, i)) {
    perf_report.mode = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-uncertainty", "--uncertainty"},
                           common_value, i)) {
    uncertainty.mode = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-uncertainty-blocks", "--uncertainty-blocks"},
                           common_value, i)) {
    uncertainty.blocks = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-uncertainty-bootstrap", "--uncertainty-bootstrap"},
          common_value, i)) {
    uncertainty.bootstrap_samples = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-uncertainty-seed", "--uncertainty-seed"},
                           common_value, i)) {
    uncertainty.seed = static_cast<unsigned int>(stoul(common_value));
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-dry-run", "--dry-run"},
                          output_manifest.dry_run, i)) {
    if (output_manifest.dry_run)
      output_manifest.output_manifest = true;
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-output-manifest", "--output-manifest"},
                          output_manifest.output_manifest, i)) {
    return option_consumed();
  }
  if (consume_option_value(arg, {"-schema-export", "--schema-export"},
                           common_value, i)) {
    output_manifest.schema_export = lowercase_copy(common_value);
    return option_consumed();
  }

  if (arg.rfind("-mol-1-type=", 0) == 0) {
    assign_mol_type(arg.substr(12), true);
    return option_consumed();
  }
  if (arg.rfind("-mol-2-type=", 0) == 0) {
    assign_mol_type(arg.substr(12), false);
    return option_consumed();
  }

  if (consume_required_option_value(arg, {"-community", "--community"},
                                    common_value, i)) {
    community.flag_community = true;
    community.method = lowercase_copy(common_value);
    flag_network = true;
    return option_consumed();
  }

  if (consume_bool_option(arg, {"-graph", "--graph"}, graph.flag_graph, i)) {
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-feature-export", "--feature-export"},
                          feature_export.export_features, i)) {
    return option_consumed();
  }
  if (consume_option_value(arg, {"-feature-source", "--feature-source"},
                           common_value, i)) {
    feature_export.source = common_value;
    feature_export.source_explicit = true;
    feature_export.export_features = true;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-segment-config", "--segment-config"},
                           common_value, i)) {
    segments.config_file = common_value;
    return option_consumed();
  }
  if (consume_bool_option(
          arg, {"-interface-analysis", "--interface-analysis"},
          interface_analysis.flag_interface_analysis, i)) {
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-interface-config", "--interface-config"}, common_value,
          i)) {
    segments.config_file = common_value;
    interface_analysis.flag_interface_analysis = true;
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-interface-contact-mode", "--interface-contact-mode"},
          common_value, i)) {
    interface_analysis.flag_interface_analysis = true;
    interface_analysis.contact_mode = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-interface-cut", "--interface-cut"},
                           common_value, i)) {
    interface_analysis.flag_interface_analysis = true;
    interface_analysis.cutoff_ang = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-interface-state-min-contacts",
           "--interface-state-min-contacts"},
          common_value, i)) {
    interface_analysis.flag_interface_analysis = true;
    interface_analysis.state_min_contacts = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-interface-sasa-stride", "--interface-sasa-stride"},
          common_value, i)) {
    interface_analysis.flag_interface_analysis = true;
    interface_analysis.sasa_stride = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-interface-sasa-points", "--interface-sasa-points"},
          common_value, i)) {
    interface_analysis.flag_interface_analysis = true;
    interface_analysis.sasa_points = stoi(common_value);
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-cmap", "--cmap"}, cmap.flag_cmap, i)) {
    cmap.new_option_explicit = true;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-cmap-mode", "--cmap-mode"}, common_value,
                           i)) {
    cmap.flag_cmap = true;
    cmap.new_option_explicit = true;
    cmap.mode = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-cmap-cut", "--cmap-cut"}, common_value,
                           i)) {
    cmap.flag_cmap = true;
    cmap.new_option_explicit = true;
    cmap.cutoff_explicit = true;
    cmap.cutoff_ang = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-cmap-scope", "--cmap-scope"}, common_value,
                           i)) {
    cmap.flag_cmap = true;
    cmap.new_option_explicit = true;
    cmap.scope = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-cmap-domain", "--cmap-domain"},
                           common_value, i)) {
    cmap.flag_cmap = true;
    cmap.new_option_explicit = true;
    cmap.domain_explicit = true;
    cmap.domain = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-cmap-min-seqsep", "--cmap-min-seqsep"},
                           common_value, i)) {
    cmap.flag_cmap = true;
    cmap.new_option_explicit = true;
    cmap.min_seqsep_explicit = true;
    cmap.min_seqsep = stoi(common_value);
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-qnative", "--qnative"},
                          qnative.flag_qnative, i)) {
    return option_consumed();
  }
  if (consume_option_value(arg, {"-qnative-ref", "--qnative-ref"},
                           common_value, i)) {
    qnative.flag_qnative = true;
    qnative.ref_explicit = true;
    qnative.ref_file = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-qnative-mode", "--qnative-mode"},
                           common_value, i)) {
    qnative.flag_qnative = true;
    qnative.mode = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-qnative-cut", "--qnative-cut"},
                           common_value, i)) {
    qnative.flag_qnative = true;
    qnative.cutoff_explicit = true;
    qnative.cutoff_ang = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-qnative-scope", "--qnative-scope"},
                           common_value, i)) {
    qnative.flag_qnative = true;
    qnative.scope = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-qnative-domain", "--qnative-domain"},
                           common_value, i)) {
    qnative.flag_qnative = true;
    qnative.domain_explicit = true;
    qnative.domain = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-qnative-min-seqsep", "--qnative-min-seqsep"},
                           common_value, i)) {
    qnative.flag_qnative = true;
    qnative.min_seqsep_explicit = true;
    qnative.min_seqsep = stoi(common_value);
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-qnative-soft", "--qnative-soft"},
                          qnative.soft, i)) {
    qnative.flag_qnative = true;
    qnative.soft_explicit = true;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-qnative-beta", "--qnative-beta"},
                           common_value, i)) {
    qnative.flag_qnative = true;
    qnative.beta_explicit = true;
    qnative.beta = stod(common_value);
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-dihedral", "--dihedral"},
                          dihedral.flag_dihedral, i)) {
    return option_consumed();
  }
  if (consume_option_value(arg, {"-dihedral-kind", "--dihedral-kind"},
                           common_value, i)) {
    dihedral.flag_dihedral = true;
    dihedral.kind = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-dihedral-residues", "--dihedral-residues"},
                           common_value, i)) {
    dihedral.flag_dihedral = true;
    dihedral.residues = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-dihedral-stride", "--dihedral-stride"},
                           common_value, i)) {
    dihedral.flag_dihedral = true;
    dihedral.stride = stoi(common_value);
    return option_consumed();
  }
  if (consume_bool_option(arg,
                          {"-dihedral-export-tica", "--dihedral-export-tica"},
                          dihedral.export_tica, i)) {
    dihedral.flag_dihedral = true;
    return option_consumed();
  }
  if (arg.rfind("-s2-plateau-window", 0) == 0 ||
      arg.rfind("--s2-plateau-window", 0) == 0) {
    TitanUI::print_error(
        "S2: -s2-plateau-window is no longer supported; use -s2-fit-window.");
    exit(EXIT_FAILURE);
  }
  if (consume_bool_option(arg, {"-s2", "--s2"}, s2.flag_s2, i)) {
    return option_consumed();
  }
  if (consume_option_value(arg, {"-s2-vector", "--s2-vector"}, common_value,
                           i)) {
    s2.flag_s2 = true;
    s2.vector = lowercase_copy(common_value);
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(arg, {"-s2-maxlag", "--s2-maxlag"}, common_value,
                           i)) {
    s2.flag_s2 = true;
    s2.maxlag = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-s2-fit-window", "--s2-fit-window"},
                           common_value, i)) {
    s2.flag_s2 = true;
    s2.fit_window = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-s2-residues", "--s2-residues"},
                           common_value, i)) {
    s2.flag_s2 = true;
    s2.residues = common_value;
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-s2-bfactor-pdb", "--s2-bfactor-pdb"},
                          s2.bfactor_pdb, i)) {
    s2.flag_s2 = s2.flag_s2 || s2.bfactor_pdb;
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-msd", "--msd"}, msd.flag_msd, i)) {
    return option_consumed();
  }
  if (consume_option_value(arg, {"-msd-selection", "--msd-selection"},
                           common_value, i)) {
    msd.flag_msd = true;
    msd.selection = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-msd-mode", "--msd-mode"}, common_value,
                           i)) {
    msd.flag_msd = true;
    msd.mode = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-msd-maxlag", "--msd-maxlag"}, common_value,
                           i)) {
    msd.flag_msd = true;
    msd.maxlag = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-msd-fit-window", "--msd-fit-window"},
                           common_value, i)) {
    msd.flag_msd = true;
    msd.fit_window = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-msd-pbc", "--msd-pbc"}, common_value, i)) {
    msd.flag_msd = true;
    msd.pbc = common_value;
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-msd-rot", "--msd-rot"}, msd.rot, i)) {
    msd.flag_msd = true;
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-msd-rot-fit-window", "--msd-rot-fit-window"},
                           common_value, i)) {
    msd.flag_msd = true;
    msd.rot = true;
    msd.rot_fit_window = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-msd-memory-limit", "--msd-memory-limit"},
                           common_value, i)) {
    msd.flag_msd = true;
    msd.memory_limit = common_value;
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-membrane", "--membrane"},
                          membrane.flag_membrane, i)) {
    return option_consumed();
  }
  if (consume_option_value(arg, {"-membrane-group", "--membrane-group"},
                           common_value, i)) {
    membrane.flag_membrane = true;
    membrane.group = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-membrane-lipids", "--membrane-lipids"},
                           common_value, i)) {
    membrane.flag_membrane = true;
    membrane.lipids = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-membrane-head", "--membrane-head"},
                           common_value, i)) {
    membrane.flag_membrane = true;
    membrane.head = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-membrane-tail", "--membrane-tail"},
                           common_value, i)) {
    membrane.flag_membrane = true;
    membrane.tail = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-membrane-normal", "--membrane-normal"},
                           common_value, i)) {
    membrane.flag_membrane = true;
    membrane.normal = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-membrane-grid", "--membrane-grid"},
                           common_value, i)) {
    membrane.flag_membrane = true;
    membrane.grid = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-membrane-leaflet", "--membrane-leaflet"},
                           common_value, i)) {
    membrane.flag_membrane = true;
    membrane.leaflet = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-membrane-leaflet-file", "--membrane-leaflet-file"},
          common_value, i)) {
    membrane.flag_membrane = true;
    membrane.leaflet_file = common_value;
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_bool_option(
          arg, {"-membrane-protein-contacts", "--membrane-protein-contacts"},
          membrane.protein_contacts, i)) {
    if (membrane.protein_contacts)
      membrane.flag_membrane = true;
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_bool_option(arg,
                          {"-membrane-enrichment", "--membrane-enrichment"},
                          membrane.enrichment, i)) {
    if (membrane.enrichment) {
      membrane.flag_membrane = true;
      membrane.protein_contacts = true;
    }
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-membrane-curvature", "--membrane-curvature"},
                           common_value, i)) {
    membrane.curvature = lowercase_copy(common_value);
    if (membrane.curvature == "grid")
      membrane.flag_membrane = true;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-graph-source", "--graph-source"},
                           common_value, i)) {
    graph.flag_graph = true;
    graph.sources.push_back(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-graph-community", "--graph-community"},
                           common_value, i)) {
    graph.flag_graph = true;
    graph.community_method = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-graph-filter-topk", "--graph-filter-topk"},
                           common_value, i)) {
    graph.flag_graph = true;
    graph.filter_topk = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-graph-filter-component", "--graph-filter-component"},
          common_value, i)) {
    graph.flag_graph = true;
    graph.filter_component = common_value;
    return option_consumed();
  }

  if (consume_option_value(arg, {"-mem-budget-gb", "--mem-budget-gb"},
                           common_value, i)) {
    parse_memory_budget(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-sg-mode", "--sg-mode"}, common_value, i)) {
    sg_mode =
        require_choice(common_value, {"auto", "stream", "cache"}, "-sg-mode");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-sg-cache", "--sg-cache"}, common_value,
                           i)) {
    sg_cache =
        require_choice(common_value, {"auto", "off", "on"}, "-sg-cache");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-sg-memory-gb", "--sg-memory-gb"},
                           common_value, i)) {
    parse_sg_memory_budget(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-sg-timeseries", "--sg-timeseries"},
                           common_value, i)) {
    sg_timeseries = require_choice(
        common_value, {"auto", "full", "summary-only"}, "-sg-timeseries");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-sg-kernel", "--sg-kernel"}, common_value,
                           i)) {
    sg_kernel = require_choice(common_value, {"auto", "scalar", "avx2"},
                               "-sg-kernel");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-sg-cutoff", "--sg-cutoff"}, common_value,
                           i)) {
    parse_sg_cutoff(common_value);
    return option_consumed();
  }
  if ((arg == "-low-memory" || arg == "--low-memory") && i + 1 < argc) {
    const std::string next = string(argv[i + 1]);
    const std::string next_lc = lowercase_copy(next);
    if (next_lc == "true" || next_lc == "false" || next_lc == "1" ||
        next_lc == "0" || next_lc == "yes" || next_lc == "no" ||
        next_lc == "on" || next_lc == "off") {
      low_memory = parse_bool_equals(next, "-low-memory");
      ++i;
      return option_consumed();
    }
  }
  if (consume_bool_option(arg, {"-low-memory", "--low-memory"}, low_memory,
                          i)) {
    return option_consumed();
  }
  if (consume_option_value(arg, {"-network-mode", "--network-mode"},
                           common_value, i)) {
    network_mode = require_choice(common_value, {"auto", "dense", "sparse"},
                                  "-network-mode");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-network-sparse-cut", "--network-sparse-cut"},
                           common_value, i)) {
    network_sparse_cut = stod(common_value);
    if (network_sparse_cut <= 0.0) {
      TitanUI::print_error("-network-sparse-cut must be positive.");
      exit(EXIT_FAILURE);
    }
    return option_consumed();
  }
  if (consume_option_value(arg, {"-network-sparse-k", "--network-sparse-k"},
                           common_value, i)) {
    network_sparse_k = stoi(common_value);
    if (network_sparse_k < 0) {
      TitanUI::print_error("-network-sparse-k must be >= 0.");
      exit(EXIT_FAILURE);
    }
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-network-cache-precision", "--network-cache-precision"},
          common_value, i)) {
    network_cache_precision = require_choice(
        common_value, {"auto", "fp32", "fp64"}, "-network-cache-precision");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-tda-max-points", "--tda-max-points"},
                           common_value, i)) {
    tda_max_points = static_cast<size_t>(stoull(common_value));
    return option_consumed();
  }
  return {};
}
