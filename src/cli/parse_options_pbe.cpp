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

arguments::OptionStep arguments::parse_options_pbe(int argc, char **argv,
                                                   int &i,
                                                   const std::string &arg,
                                                   std::string &common_value) {
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
    return option_consumed();
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
        return option_consumed();
      }
      if (!backend.empty() && backend.front() != '-') {
        TitanUI::print_error(
            "PBE: -pbe backend must be grid or bie (legacy alias: tabi; "
            "grid aliases: delphi, pydelphi, fd).");
        exit(EXIT_FAILURE);
      }
    }
    pbe.flag_pbe = true;
    return option_consumed();
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
    return option_consumed();
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
        return option_consumed();
      }
    }
    pbe.pbsa = true;
    pbe.flag_pbe = true;
    pbe.dgsolv = true;
    return option_consumed();
  }
  if (arg.rfind("-pbsa=", 0) == 0 || arg.rfind("--pbsa=", 0) == 0) {
    pbe.pbsa = parse_bool_equals(arg.substr(arg.find('=') + 1), "-pbsa");
    if (pbe.pbsa) {
      pbe.flag_pbe = true;
      pbe.dgsolv = true;
    }
    return option_consumed();
  }
  if (consume_compound_option_value(
          arg, {"-pbsa-components", "--pbsa-components"}, common_value, i)) {
    std::string error;
    if (!titan_cli::normalize_pbsa_component_selection(
            common_value, pbe.pbsa_components, error)) {
      TitanUI::print_error("PBSA: invalid -pbsa-components value: " + error +
                           ". Use complex,ligand,receptor or "
                           "ligand-receptor-complex.");
      return option_exit(titan_cli::kInvalidArguments);
    }
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbsa-decomp", "--pbsa-decomp"},
                           common_value, i)) {
    pbe.pbsa_decomp = require_choice(
        common_value, {"off", "molecule", "residue", "hybrid", "auto"},
        "-pbsa-decomp");
    return option_consumed();
  }
  if (consume_bool_option(arg, {"-pbsa-ie", "--pbsa-ie"}, pbe.pbsa_ie, i)) {
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbsa-residue-frames", "--pbsa-residue-frames"},
                           common_value, i)) {
    pbe.pbsa_residue_frames =
        require_choice(common_value, {"off", "all"}, "-pbsa-residue-frames");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbsa-ie-block", "--pbsa-ie-block"},
                           common_value, i)) {
    pbe.pbsa_ie_block = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbsa-ie-min-samples", "--pbsa-ie-min-samples"},
                           common_value, i)) {
    pbe.pbsa_ie_min_samples = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbsa-ie-filter-sigma", "--pbsa-ie-filter-sigma"},
          common_value, i)) {
    pbe.pbsa_ie_filter_sigma = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbsa-perf", "--pbsa-perf"}, common_value,
                           i)) {
    pbe.pbsa_perf = require_choice(common_value, {"off", "summary", "detail"},
                                   "-pbsa-perf");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbsa-grid-reuse", "--pbsa-grid-reuse"},
                           common_value, i)) {
    pbe.pbsa_grid_reuse = require_choice(
        common_value, {"off", "validate", "auto"}, "-pbsa-grid-reuse");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbsa-sasa-points", "--pbsa-sasa-points"},
          common_value, i)) {
    pbe.pbsa_sasa_points = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-timing", "--pbe-timing"}, common_value,
                           i)) {
    pbe.timing = require_choice(common_value, {"off", "summary", "detail"},
                                "-pbe-timing");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-tree-preset", "--pbe-bie-tree-preset"},
          common_value, i)) {
    pbe.tabi_tree_preset = require_choice(
        common_value,
        {"off", "stage3u-safe", "stage3u-perf", "stage3u-perf2"},
        "-pbe-bie-tree-preset");
    apply_tabi_tree_preset(pbe.tabi_tree_preset);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-bie-sum", "--pbe-bie-sum"},
                           common_value, i)) {
    pbe.tabi_sum = require_choice(
        common_value, {"auto", "direct", "treecode"}, "-pbe-bie-sum");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-bie-geometry", "--pbe-bie-geometry"},
                           common_value, i)) {
    pbe.tabi_geometry = require_choice(
        common_value, {"triangle", "point-panel"}, "-pbe-bie-geometry");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-bie-diagonal", "--pbe-bie-diagonal"},
                           common_value, i)) {
    pbe.tabi_diagonal = require_choice(
        common_value, {"solid-angle", "flat", "off"}, "-pbe-bie-diagonal");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-taylor-order", "--pbe-bie-taylor-order"},
          common_value, i)) {
    pbe.tabi_taylor_order = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-derivatives", "--pbe-bie-derivatives"},
          common_value, i)) {
    pbe.tabi_derivatives =
        require_choice(common_value, {"analytic", "finite-diff-debug"},
                       "-pbe-bie-derivatives");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-surface-points", "--pbe-bie-surface-points"},
          common_value, i)) {
    pbe.tabi_surface_points = stoi(common_value);
    return option_consumed();
  }
  if (consume_bool_option(
          arg, {"-pbe-bie-surface-only", "--pbe-bie-surface-only"},
          pbe.tabi_surface_only, i)) {
    return option_consumed();
  }
  if (consume_bool_option(
          arg, {"-pbe-bie-highres-surface", "--pbe-bie-highres-surface"},
          pbe.tabi_highres_surface, i)) {
    return option_consumed();
  }
  if (consume_bool_option(arg,
                          {"-pbe-bie-calderon-trace-audit",
                           "--pbe-bie-calderon-trace-audit"},
                          pbe.tabi_calderon_trace_audit, i)) {
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-calderon-aux-surface",
                            "--pbe-bie-calderon-aux-surface"},
                           common_value, i)) {
    pbe.tabi_calderon_aux_surface =
        require_choice(common_value, {"off", "implicit-vdw"},
                       "-pbe-bie-calderon-aux-surface");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-calderon-aux-grid-spacing",
                            "--pbe-bie-calderon-aux-grid-spacing"},
                           common_value, i)) {
    pbe.tabi_calderon_aux_grid_spacing = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-primary-surface", "--pbe-bie-primary-surface"},
          common_value, i)) {
    pbe.tabi_primary_surface = require_choice(
        common_value, {"sphere-patch", "grid-ses", "analytic-ses"},
        "-pbe-bie-primary-surface");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-analytic-ses-engine",
                            "--pbe-bie-analytic-ses-engine"},
                           common_value, i)) {
    pbe.tabi_analytic_ses_engine = require_choice(
        common_value,
        {"current", "current-dense", "patch-ray-experimental",
         "patch-ray-resolution"},
        "-pbe-bie-analytic-ses-engine");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-ses-grid-spacing", "--pbe-bie-ses-grid-spacing"},
          common_value, i)) {
    pbe.tabi_ses_grid_spacing = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-ses-max-grid-vertices",
                            "--pbe-bie-ses-max-grid-vertices"},
                           common_value, i)) {
    pbe.tabi_ses_max_grid_vertices = stoull(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-sphere-mesh", "--pbe-bie-sphere-mesh"},
          common_value, i)) {
    pbe.tabi_sphere_mesh =
        require_choice(common_value, {"auto", "latlong", "icosphere"},
                       "-pbe-bie-sphere-mesh");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-sphere-patch-clip", "--pbe-bie-sphere-patch-clip"},
          common_value, i)) {
    pbe.tabi_sphere_patch_clip =
        require_choice(common_value, {"on", "off"},
                       "-pbe-bie-sphere-patch-clip") == "on";
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-tree-leaf-size", "--pbe-bie-tree-leaf-size"},
          common_value, i)) {
    pbe.tabi_tree_leaf_size = stoi(common_value);
    pbe.tabi_tree_leaf_size_explicit = true;
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-opening-theta", "--pbe-bie-opening-theta"},
          common_value, i)) {
    pbe.tabi_tree_opening_theta = stod(common_value);
    pbe.tabi_tree_opening_theta_explicit = true;
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-energy-far-eta", "--pbe-bie-energy-far-eta"},
          common_value, i)) {
    pbe.tabi_energy_far_eta = stod(common_value);
    if (!std::isfinite(pbe.tabi_energy_far_eta) ||
        (pbe.tabi_energy_far_eta != 0.0 &&
         (pbe.tabi_energy_far_eta < 2.0 ||
          pbe.tabi_energy_far_eta > 1000.0))) {
      TitanUI::print_error(
          "-pbe-bie-energy-far-eta must be 0 (exact) or between 2 and 1000.");
      exit(EXIT_FAILURE);
    }
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-ses-remesh-edge",
                            "--pbe-bie-ses-remesh-edge"},
                           common_value, i)) {
    if (common_value == "off") {
      pbe.tabi_ses_remesh_edge = -1.0;
    } else {
      pbe.tabi_ses_remesh_edge = stod(common_value);
      if (!std::isfinite(pbe.tabi_ses_remesh_edge) ||
          (pbe.tabi_ses_remesh_edge != 0.0 &&
           (pbe.tabi_ses_remesh_edge < 0.25 ||
            pbe.tabi_ses_remesh_edge > 4.0))) {
        TitanUI::print_error(
            "-pbe-bie-ses-remesh-edge must be off, 0 (grid spacing) or between 0.25 and 4.0 A.");
        exit(EXIT_FAILURE);
      }
    }
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-ses-remesh-iterations",
                            "--pbe-bie-ses-remesh-iterations"},
                           common_value, i)) {
    pbe.tabi_ses_remesh_iterations = stoi(common_value);
    if (pbe.tabi_ses_remesh_iterations < 1 ||
        pbe.tabi_ses_remesh_iterations > 12) {
      TitanUI::print_error(
          "-pbe-bie-ses-remesh-iterations must be between 1 and 12.");
      exit(EXIT_FAILURE);
    }
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-ses-charge-refine",
                            "--pbe-bie-ses-charge-refine"},
                           common_value, i)) {
    pbe.tabi_ses_charge_refine =
        common_value == "off" ? 0.0 : stod(common_value);
    if (!std::isfinite(pbe.tabi_ses_charge_refine) ||
        pbe.tabi_ses_charge_refine < 0.0 || pbe.tabi_ses_charge_refine > 4.0) {
      TitanUI::print_error(
          "-pbe-bie-ses-charge-refine must be off or an edge-to-distance ratio in (0, 4].");
      exit(EXIT_FAILURE);
    }
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-ses-charge-refine-min-edge",
                            "--pbe-bie-ses-charge-refine-min-edge"},
                           common_value, i)) {
    pbe.tabi_ses_charge_refine_min_edge = stod(common_value);
    if (!std::isfinite(pbe.tabi_ses_charge_refine_min_edge) ||
        pbe.tabi_ses_charge_refine_min_edge < 0.05 ||
        pbe.tabi_ses_charge_refine_min_edge > 2.0) {
      TitanUI::print_error(
          "-pbe-bie-ses-charge-refine-min-edge must be between 0.05 and 2.0 A.");
      exit(EXIT_FAILURE);
    }
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-ses-charge-refine-min-charge",
                            "--pbe-bie-ses-charge-refine-min-charge"},
                           common_value, i)) {
    pbe.tabi_ses_charge_refine_min_charge = stod(common_value);
    if (!std::isfinite(pbe.tabi_ses_charge_refine_min_charge) ||
        pbe.tabi_ses_charge_refine_min_charge < 0.0 ||
        pbe.tabi_ses_charge_refine_min_charge > 2.0) {
      TitanUI::print_error(
          "-pbe-bie-ses-charge-refine-min-charge must be between 0 and 2 e.");
      exit(EXIT_FAILURE);
    }
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-curved-sample-order",
                            "--pbe-bie-curved-sample-order"},
                           common_value, i)) {
    pbe.tabi_curved_sample_order = std::stoi(common_value);
    if (pbe.tabi_curved_sample_order < 2 ||
        pbe.tabi_curved_sample_order > 4) {
      TitanUI::print_error("-pbe-bie-curved-sample-order must be 2, 3 or 4.");
      exit(EXIT_FAILURE);
    }
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-curved-near-zone",
                            "--pbe-bie-curved-near-zone"},
                           common_value, i)) {
    pbe.tabi_curved_near_zone_radii = std::stod(common_value);
    if (!std::isfinite(pbe.tabi_curved_near_zone_radii) ||
        pbe.tabi_curved_near_zone_radii < 1.0 ||
        pbe.tabi_curved_near_zone_radii > 16.0) {
      TitanUI::print_error(
          "-pbe-bie-curved-near-zone must be between 1 and 16 cell radii.");
      exit(EXIT_FAILURE);
    }
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-discretization", "--pbe-bie-discretization"},
          common_value, i)) {
    pbe.tabi_discretization =
        require_choice(common_value, {"centroid", "node-patch"},
                       "-pbe-bie-discretization");
    pbe.tabi_discretization_explicit = true;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-bie-precond", "--pbe-bie-precond"},
                           common_value, i)) {
    pbe.tabi_precond_requested = require_choice(
        common_value,
        {"auto", "none", "diagonal", "block", "block-budgeted",
         "coarse-galerkin-probe", "fmm-block-jacobi",
         "fmm-overlap-schwarz", "fmm-nearfield-ilu0",
         "fmm-nearfield-ilu0-coarse", "distance-knn-overlap-schwarz",
         "distance-knn-ilu0", "classic-nearfield-ilu0"},
        "-pbe-bie-precond");
    pbe.tabi_precond = pbe.tabi_precond_requested;
    pbe.tabi_precond_explicit = true;
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-precond-block-near", "--pbe-bie-precond-block-near"},
          common_value, i)) {
    pbe.tabi_precond_block_near = require_choice(
        common_value, {"singular", "buffer-aware", "operator-near"},
        "-pbe-bie-precond-block-near");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-precond-block-panel-cap",
                            "--pbe-bie-precond-block-panel-cap"},
                           common_value, i)) {
    pbe.tabi_precond_block_panel_cap = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-precond-overlap-panel-cap",
                            "--pbe-bie-precond-overlap-panel-cap"},
                           common_value, i)) {
    pbe.tabi_precond_overlap_panel_cap = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-precond-ilu-neighbor-cap",
                            "--pbe-bie-precond-ilu-neighbor-cap"},
                           common_value, i)) {
    pbe.tabi_precond_ilu_neighbor_cap = stoi(common_value);
    pbe.tabi_precond_ilu_neighbor_cap_explicit = true;
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-precond-ilu-drop-tol",
                            "--pbe-bie-precond-ilu-drop-tol"},
                           common_value, i)) {
    pbe.tabi_precond_ilu_drop_tol = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-precond-ilu-regularization",
                            "--pbe-bie-precond-ilu-regularization"},
                           common_value, i)) {
    pbe.tabi_precond_ilu_regularization =
        require_choice(common_value, {"off", "adaptive"},
                       "-pbe-bie-precond-ilu-regularization");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-precond-ilu-regularization-max",
                            "--pbe-bie-precond-ilu-regularization-max"},
                           common_value, i)) {
    pbe.tabi_precond_ilu_regularization_max = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-precond-memory-budget-mb",
                            "--pbe-bie-precond-memory-budget-mb"},
                           common_value, i)) {
    pbe.tabi_precond_memory_budget_mb = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-precond-coarse-aggregate-cap",
                            "--pbe-bie-precond-coarse-aggregate-cap"},
                           common_value, i)) {
    pbe.tabi_precond_coarse_aggregate_cap = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-precond-coarse-base", "--pbe-bie-precond-coarse-base"},
          common_value, i)) {
    pbe.tabi_precond_coarse_base =
        require_choice(common_value, {"diagonal", "block-budgeted"},
                       "-pbe-bie-precond-coarse-base");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-precond-coarse-verify",
                            "--pbe-bie-precond-coarse-verify"},
                           common_value, i)) {
    pbe.tabi_precond_coarse_verify =
        require_choice(common_value, {"off", "full-small"},
                       "-pbe-bie-precond-coarse-verify");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-near-integration", "--pbe-bie-near-integration"},
          common_value, i)) {
    pbe.tabi_near_integration = require_choice(
        common_value,
        {"auto", "centroid", "analytic_singular", "duffy_adaptive"},
        "-pbe-bie-near-integration");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-quad-buffer", "--pbe-bie-quad-buffer"},
          common_value, i)) {
    pbe.tabi_quad_buffer =
        require_choice(common_value, {"auto", "off", "gauss7", "adaptive"},
                       "-pbe-bie-quad-buffer");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-leaf-near-policy", "--pbe-bie-leaf-near-policy"},
          common_value, i)) {
    pbe.tabi_leaf_near_policy = require_choice(
        common_value,
        {"auto", "exact", "smooth-node", "large-protein", "adaptive-large"},
        "-pbe-bie-leaf-near-policy");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-leaf-panel-kernel", "--pbe-bie-leaf-panel-kernel"},
          common_value, i)) {
    pbe.tabi_leaf_panel_kernel = require_choice(
        common_value,
        {"auto", "scalar", "avx2-gather", "tile", "tile-scalar", "tile-avx2"},
        "-pbe-bie-leaf-panel-kernel");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-adaptive-level", "--pbe-bie-adaptive-level"},
          common_value, i)) {
    pbe.tabi_adaptive_level =
        require_choice(common_value, {"strict", "balanced", "aggressive"},
                       "-pbe-bie-adaptive-level");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-cluster-policy", "--pbe-bie-cluster-policy"},
          common_value, i)) {
    pbe.tabi_cluster_policy = require_choice(
        common_value, {"off", "leaf-pair-strict", "leaf-pair-balanced"},
        "-pbe-bie-cluster-policy");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-leaf-pair-kernel", "--pbe-bie-leaf-pair-kernel"},
          common_value, i)) {
    pbe.tabi_leaf_pair_kernel =
        require_choice(common_value,
                       {"auto", "scalar", "target-avx2", "target-avx2-v2",
                        "target-avx2-4x4"},
                       "-pbe-bie-leaf-pair-kernel");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-leaf-kernel-profile", "--pbe-bie-leaf-kernel-profile"},
          common_value, i)) {
    pbe.tabi_leaf_kernel_profile = require_choice(
        common_value, {"off", "summary"}, "-pbe-bie-leaf-kernel-profile");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-leaf-work-profile", "--pbe-bie-leaf-work-profile"},
          common_value, i)) {
    pbe.tabi_leaf_work_profile = require_choice(
        common_value, {"off", "summary"}, "-pbe-bie-leaf-work-profile");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-matvec-cost-profile", "--pbe-bie-matvec-cost-profile"},
          common_value, i)) {
    pbe.tabi_matvec_cost_profile = require_choice(
        common_value, {"off", "summary"}, "-pbe-bie-matvec-cost-profile");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-jcp-direct-kernel", "--pbe-bie-jcp-direct-kernel"},
          common_value, i)) {
    pbe.tabi_jcp_direct_kernel = require_choice(
        common_value, {"baseline", "soa-avx2", "fused-soa-avx2"},
        "-pbe-bie-jcp-direct-kernel");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-leaf-pair-selection-profile",
                            "--pbe-bie-leaf-pair-selection-profile"},
                           common_value, i)) {
    pbe.tabi_leaf_pair_selection_profile =
        require_choice(common_value, {"off", "summary"},
                       "-pbe-bie-leaf-pair-selection-profile");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-leaf-pair-probe-cache",
                            "--pbe-bie-leaf-pair-probe-cache"},
                           common_value, i)) {
    pbe.tabi_leaf_pair_probe_cache =
        require_choice(common_value, {"off", "per-matvec"},
                       "-pbe-bie-leaf-pair-probe-cache");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-leaf-pair-target-group",
                            "--pbe-bie-leaf-pair-target-group"},
                           common_value, i)) {
    pbe.tabi_leaf_pair_target_group =
        require_choice(common_value, {"scan", "same-target-leaf"},
                       "-pbe-bie-leaf-pair-target-group");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-leaf-pair-directness-probe",
                            "--pbe-bie-leaf-pair-directness-probe"},
                           common_value, i)) {
    pbe.tabi_leaf_pair_directness_probe =
        require_choice(common_value, {"full-traversal", "ancestor-path"},
                       "-pbe-bie-leaf-pair-directness-probe");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-leaf-pair-directness-cache",
                            "--pbe-bie-leaf-pair-directness-cache"},
                           common_value, i)) {
    pbe.tabi_leaf_pair_directness_cache =
        require_choice(common_value, {"off", "target-leaf-source-leaf"},
                       "-pbe-bie-leaf-pair-directness-cache");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-leaf-pair-directness-verify",
                            "--pbe-bie-leaf-pair-directness-verify"},
                           common_value, i)) {
    pbe.tabi_leaf_pair_directness_verify =
        require_choice(common_value, {"off", "sampled", "full-small"},
                       "-pbe-bie-leaf-pair-directness-verify");
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-leaf-pair-worklist-max-active",
                            "--pbe-bie-leaf-pair-worklist-max-active"},
                           common_value, i)) {
    pbe.tabi_leaf_pair_worklist_max_active = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-leaf-pair-worklist-verify",
                            "--pbe-bie-leaf-pair-worklist-verify"},
                           common_value, i)) {
    pbe.tabi_leaf_pair_worklist_verify =
        require_choice(common_value, {"off", "sampled", "full-small"},
                       "-pbe-bie-leaf-pair-worklist-verify");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-leaf-block-verify", "--pbe-bie-leaf-block-verify"},
          common_value, i)) {
    pbe.tabi_leaf_block_verify =
        require_choice(common_value, {"off", "sampled", "full-small"},
                       "-pbe-bie-leaf-block-verify");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-bie-leaf-mac", "--pbe-bie-leaf-mac"},
                           common_value, i)) {
    pbe.tabi_leaf_mac = require_choice(common_value, {"off", "audit", "on"},
                                       "-pbe-bie-leaf-mac");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-bie-fmm-m2l", "--pbe-bie-fmm-m2l"},
                           common_value, i)) {
    pbe.tabi_fmm_m2l = require_choice(
        common_value, {"auto", "off", "single-matvec-analytic"},
        "-pbe-bie-fmm-m2l");
    pbe.tabi_fmm_m2l_explicit = true;
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-fmm-cartesian-order",
           "--pbe-bie-fmm-cartesian-order"},
          common_value, i)) {
    pbe.bie_fmm_cartesian_order = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-kifmm-order", "--pbe-bie-kifmm-order"},
          common_value, i)) {
    pbe.tabi_kifmm_order = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-kifmm-equiv-scale", "--pbe-bie-kifmm-equiv-scale"},
          common_value, i)) {
    pbe.tabi_kifmm_equiv_scale = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-kifmm-check-scale", "--pbe-bie-kifmm-check-scale"},
          common_value, i)) {
    pbe.tabi_kifmm_check_scale = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-kifmm-solver", "--pbe-bie-kifmm-solver"},
          common_value, i)) {
    pbe.tabi_kifmm_solver =
        require_choice(common_value, {"qr", "svd", "normal-tikhonov"},
                       "-pbe-bie-kifmm-solver");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-kifmm-profile", "--pbe-bie-kifmm-profile"},
          common_value, i)) {
    pbe.tabi_kifmm_profile = require_choice(
        common_value, {"off", "summary", "full"}, "-pbe-bie-kifmm-profile");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-kifmm-full-apply", "--pbe-bie-kifmm-full-apply"},
          common_value, i)) {
    pbe.tabi_kifmm_full_apply =
        require_choice(common_value, {"serial", "target-node-parallel"},
                       "-pbe-bie-kifmm-full-apply");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-kifmm-ilu-graph", "--pbe-bie-kifmm-ilu-graph"},
          common_value, i)) {
    pbe.tabi_kifmm_ilu_graph = require_choice(
        common_value, {"cartesian-near", "kifmm-fallback-capped"},
        "-pbe-bie-kifmm-ilu-graph");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-kifmm-hierarchy-order",
                            "--pbe-bie-kifmm-hierarchy-order"},
                           common_value, i)) {
    pbe.tabi_kifmm_hierarchy_order = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-kifmm-hierarchy-theta",
                            "--pbe-bie-kifmm-hierarchy-theta"},
                           common_value, i)) {
    pbe.tabi_kifmm_hierarchy_theta = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-kifmm-hierarchy-real-panel-guard",
                            "--pbe-bie-kifmm-hierarchy-real-panel-guard"},
                           common_value, i)) {
    pbe.tabi_kifmm_hierarchy_real_panel_guard =
        require_choice(common_value, {"strict-local-check"},
                       "-pbe-bie-kifmm-hierarchy-real-panel-guard");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-kifmm-hierarchy-guard-margin-factor",
                            "--pbe-bie-kifmm-hierarchy-guard-margin-factor"},
                           common_value, i)) {
    pbe.tabi_kifmm_hierarchy_guard_margin_factor = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-kifmm-hierarchy-tree-stop-policy",
                            "--pbe-bie-kifmm-hierarchy-tree-stop-policy"},
                           common_value, i)) {
    pbe.tabi_kifmm_hierarchy_tree_stop_policy =
        require_choice(common_value, {"panel-radius-occupancy"},
                       "-pbe-bie-kifmm-hierarchy-tree-stop-policy");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-kifmm-hierarchy-tree-panel-radius-factor",
           "--pbe-bie-kifmm-hierarchy-tree-panel-radius-factor"},
          common_value, i)) {
    pbe.tabi_kifmm_hierarchy_tree_panel_radius_factor = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-kifmm-hierarchy-tree-occupancy-stop",
                            "--pbe-bie-kifmm-hierarchy-tree-occupancy-stop"},
                           common_value, i)) {
    pbe.tabi_kifmm_hierarchy_tree_occupancy_stop = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-kifmm-hierarchy-m2l-eval-policy",
                            "--pbe-bie-kifmm-hierarchy-m2l-eval-policy"},
                           common_value, i)) {
    pbe.tabi_kifmm_hierarchy_m2l_eval_policy =
        require_choice(common_value, {"target-check-batched"},
                       "-pbe-bie-kifmm-hierarchy-m2l-eval-policy");
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-kifmm-hierarchy-matrix-cache-limit-mb",
           "--pbe-bie-kifmm-hierarchy-matrix-cache-limit-mb"},
          common_value, i)) {
    pbe.tabi_kifmm_hierarchy_matrix_cache_limit_mb = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-kifmm-hierarchy-gemm-batch-policy",
                            "--pbe-bie-kifmm-hierarchy-gemm-batch-policy"},
                           common_value, i)) {
    pbe.tabi_kifmm_hierarchy_gemm_batch_policy =
        require_choice(common_value, {"per-pair"},
                       "-pbe-bie-kifmm-hierarchy-gemm-batch-policy");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-kifmm-hierarchy-m2l-parallel-policy",
                            "--pbe-bie-kifmm-hierarchy-m2l-parallel-policy"},
                           common_value, i)) {
    pbe.tabi_kifmm_hierarchy_m2l_parallel_policy =
        require_choice(common_value, {"serial", "target-node-key-group"},
                       "-pbe-bie-kifmm-hierarchy-m2l-parallel-policy");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-kifmm-hierarchy-m2l-parallel-workers",
           "--pbe-bie-kifmm-hierarchy-m2l-parallel-workers"},
          common_value, i)) {
    pbe.tabi_kifmm_hierarchy_m2l_parallel_workers = stoi(common_value);
    return option_consumed();
  }
  return {};
}
