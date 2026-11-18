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

arguments::OptionStep arguments::parse_options_pbe_solver(int argc, char **argv,
                                                          int &i,
                                                          const std::string &arg,
                                                          std::string &common_value) {
  if (consume_option_value(arg,
                           {"-pbe-stability-audit", "--pbe-stability-audit"},
                           common_value, i)) {
    pbe.stability_audit = require_choice(
        common_value, {"off", "summary", "detail"}, "-pbe-stability-audit");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-stability-mode", "--pbe-stability-mode"},
                           common_value, i)) {
    pbe.stability_mode = require_choice(
        common_value, {"off", "diagnose", "anchor", "hysteresis", "ensemble"},
        "-pbe-stability-mode");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-grid-anchor", "--pbe-grid-anchor"},
                           common_value, i)) {
    pbe.grid_anchor = require_choice(
        common_value, {"auto", "per-frame", "first-frame", "fixed-center"},
        "-pbe-grid-anchor");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-scale", "--pbe-scale"}, common_value,
                           i)) {
    pbe.scale = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-margin", "--pbe-margin"}, common_value,
                           i)) {
    pbe.margin = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-gridbox", "--pbe-gridbox"},
                           common_value, i)) {
    pbe.gridbox =
        require_choice(common_value, {"cuboidal", "cubic"}, "-pbe-gridbox");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-int-diel", "--pbe-int-diel"},
                           common_value, i)) {
    pbe.int_diel = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-ext-diel", "--pbe-ext-diel"},
                           common_value, i)) {
    pbe.ext_diel = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-salt", "--pbe-salt"}, common_value,
                           i)) {
    pbe.salt = stod(common_value);
    saltcon = pbe.salt;
    salt_concentration = pbe.salt;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-temp", "--pbe-temp"}, common_value,
                           i)) {
    pbe.temp = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-model", "--pbe-model"}, common_value,
                           i)) {
    pbe.model =
        require_choice(common_value, {"linear", "nonlinear"}, "-pbe-model");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-biomodel", "--pbe-biomodel"},
                           common_value, i)) {
    pbe.biomodel =
        require_choice(common_value, {"pbe", "rpbe"}, "-pbe-biomodel");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-dielectric", "--pbe-dielectric"},
                           common_value, i)) {
    pbe.dielectric_model = require_choice(
        common_value, {"two-dielectric", "gaussian"}, "-pbe-dielectric");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-boundary", "--pbe-boundary"},
                           common_value, i)) {
    pbe.boundary = require_choice(
        common_value, {"coulombic", "dipolar", "zero"}, "-pbe-boundary");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-charge-assignment", "--pbe-charge-assignment"},
          common_value, i)) {
    pbe.charge_assignment =
        require_choice(common_value, {"trilinear", "delphi", "gaussian"},
                       "-pbe-charge-assignment");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-surface", "--pbe-surface"},
                           common_value, i)) {
    pbe.surface =
        require_choice(common_value,
                       {"vdw", "sas", "ses", "delphi-ses", "delphi-vdw",
                        "gaussian", "gaussian-cutoff"},
                       "-pbe-surface");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-radius-convention", "--pbe-radius-convention"},
          common_value, i)) {
    pbe.radius_convention =
        require_choice(common_value, {"auto", "native", "delphi-bondi"},
                       "-pbe-radius-convention");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-probe", "--pbe-probe"}, common_value,
                           i)) {
    pbe.probe_radius = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-ion-access", "--pbe-ion-access"},
                           common_value, i)) {
    pbe.ion_access = require_choice(common_value, {"dielectric", "stern"},
                                    "-pbe-ion-access");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-ion-radius", "--pbe-ion-radius"},
                           common_value, i)) {
    pbe.ion_radius = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-max-iter", "--pbe-max-iter"},
                           common_value, i)) {
    pbe.max_iter = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-rms", "--pbe-rms"}, common_value, i)) {
    pbe.rms_tol = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-dphi", "--pbe-dphi"}, common_value,
                           i)) {
    pbe.dphi_tol = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-omega", "--pbe-omega"}, common_value,
                           i)) {
    pbe.omega = lowercase_copy(common_value);
    if (pbe.omega != "auto")
      pbe.omega_value = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-max-grid", "--pbe-max-grid"},
                           common_value, i)) {
    pbe.max_grid_points = static_cast<std::size_t>(stoull(common_value));
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-map", "--pbe-map"}, common_value, i)) {
    pbe.map = require_choice(
        common_value, {"off", "phi", "epsilon", "ion", "all"}, "-pbe-map");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-map-frames", "--pbe-map-frames"},
                           common_value, i)) {
    pbe.map_frames = require_choice(
        common_value, {"none", "first", "last", "all"}, "-pbe-map-frames");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-atom-potentials", "--pbe-atom-potentials"},
                           common_value, i)) {
    pbe.atom_potentials =
        require_choice(common_value, {"off", "csv"}, "-pbe-atom-potentials");
    if (pbe.atom_potentials != "off")
      pbe.flag_pbe = true;
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-surface-output", "--pbe-bie-surface-output"},
          common_value, i)) {
    pbe.tabi_surface_output =
        require_choice(common_value, {"off", "csv", "vtk", "all"},
                       "-pbe-bie-surface-output");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-kernel", "--pbe-kernel"}, common_value,
                           i)) {
    pbe.kernel = require_choice(common_value, {"auto", "scalar", "avx2"},
                                "-pbe-kernel");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-focus", "--pbe-focus"}, common_value,
                           i)) {
    pbe.focus =
        require_choice(common_value, {"off", "auto", "delphi"}, "-pbe-focus");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-multigrid", "--pbe-multigrid"},
                           common_value, i)) {
    const std::string multigrid = require_choice(
        common_value, {"off", "focus", "auto", "vcycle"}, "-pbe-multigrid");
    pbe.multigrid = multigrid;
    if (multigrid == "focus" || multigrid == "auto")
      pbe.focus = "auto";
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-focus-levels", "--pbe-focus-levels"},
                           common_value, i)) {
    pbe.focus_levels = stoi(common_value);
    if (pbe.focus_levels > 1 && pbe.focus == "off")
      pbe.focus = "auto";
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-focus-margin", "--pbe-focus-margin"},
                           common_value, i)) {
    pbe.focus_margin = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-mg-levels", "--pbe-mg-levels"},
                           common_value, i)) {
    pbe.mg_levels = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-mg-pre", "--pbe-mg-pre"}, common_value,
                           i)) {
    pbe.mg_pre = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-mg-post", "--pbe-mg-post"},
                           common_value, i)) {
    pbe.mg_post = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-mg-coarse-iter", "--pbe-mg-coarse-iter"},
                           common_value, i)) {
    pbe.mg_coarse_iter = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-mg-accel", "--pbe-mg-accel"},
                           common_value, i)) {
    pbe.mg_accel =
        require_choice(lowercase_copy(common_value), {"cg", "none"},
                       "-pbe-mg-accel");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nl-max-iter", "--pbe-nl-max-iter"},
                           common_value, i)) {
    pbe.nl_max_iter = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nl-inner-iter", "--pbe-nl-inner-iter"},
                           common_value, i)) {
    pbe.nl_inner_iter = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nl-stagnation", "--pbe-nl-stagnation"},
                           common_value, i)) {
    pbe.nl_stagnation = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nl-block", "--pbe-nl-block"},
                           common_value, i)) {
    pbe.nl_block = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nl-dphi", "--pbe-nl-dphi"},
                           common_value, i)) {
    pbe.nl_dphi = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nl-relax", "--pbe-nl-relax"},
                           common_value, i)) {
    pbe.nl_relax = lowercase_copy(common_value);
    if (pbe.nl_relax != "auto")
      pbe.nl_relax_value = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nl-clip", "--pbe-nl-clip"},
                           common_value, i)) {
    pbe.nl_clip = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-nl-continuation", "--pbe-nl-continuation"},
                           common_value, i)) {
    pbe.nl_continuation =
        require_choice(common_value, {"off", "auto"}, "-pbe-nl-continuation");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-nl-line-search", "--pbe-nl-line-search"},
                           common_value, i)) {
    pbe.nl_line_search =
        require_choice(common_value, {"off", "auto"}, "-pbe-nl-line-search");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nl-residual", "--pbe-nl-residual"},
                           common_value, i)) {
    pbe.nl_residual = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nl-solver", "--pbe-nl-solver"},
                           common_value, i)) {
    pbe.nl_solver =
        require_choice(common_value, {"auto", "picard", "newton"},
                       "-pbe-nl-solver");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nl-inner", "--pbe-nl-inner"},
                           common_value, i)) {
    pbe.nl_inner = require_choice(common_value, {"auto", "sor", "vcycle"},
                                  "-pbe-nl-inner");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nl-multigrid", "--pbe-nl-multigrid"},
                           common_value, i)) {
    pbe.nl_multigrid =
        require_choice(common_value, {"off", "fas"}, "-pbe-nl-multigrid");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nwt-kernel", "--pbe-nwt-kernel"},
                           common_value, i)) {
    pbe.nwt_kernel = require_choice(
        common_value, {"auto", "legacy", "delphi-half"}, "-pbe-nwt-kernel");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-nwt-homo-kernel", "--pbe-nwt-homo-kernel"},
                           common_value, i)) {
    pbe.nwt_homo_kernel = require_choice(
        common_value,
        {"auto", "packed-list", "row-span", "avx2", "salt-tiled"},
        "-pbe-nwt-homo-kernel");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-nwt-progress", "--pbe-nwt-progress"},
                           common_value, i)) {
    pbe.nwt_progress = require_choice(common_value, {"auto", "off", "strict"},
                                      "-pbe-nwt-progress");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-linear-kernel", "--pbe-linear-kernel"},
                           common_value, i)) {
    pbe.linear_kernel =
        require_choice(common_value, {"auto", "legacy", "half-packed"},
                       "-pbe-linear-kernel");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-gaussian-exp", "--pbe-gaussian-exp"},
                           common_value, i)) {
    pbe.gaussian_exp = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-gaussian-sigma", "--pbe-gaussian-sigma"},
                           common_value, i)) {
    pbe.gaussian_sigma = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-density-cutoff", "--pbe-density-cutoff"},
                           common_value, i)) {
    pbe.density_cutoff = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-surface-cutoff", "--pbe-surface-cutoff"},
                           common_value, i)) {
    pbe.surface_cutoff = lowercase_copy(common_value);
    if (pbe.surface_cutoff != "auto")
      pbe.surface_cutoff_value = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-gap-diel", "--pbe-gap-diel"},
                           common_value, i)) {
    pbe.gap_diel = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-probe-file", "--pbe-probe-file",
                            "-pbe-test-probe-file", "--pbe-test-probe-file"},
                           common_value, i)) {
    pbe.probe_file = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-probe-residue", "--pbe-probe-residue"},
                           common_value, i)) {
    pbe.probe_residue = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-probe-site", "--pbe-probe-site"},
                           common_value, i)) {
    pbe.probe_site = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-focus-center", "--pbe-focus-center"},
                           common_value, i)) {
    parse_pbe_focus_center(common_value);
    return option_consumed();
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
        return option_consumed();
      }
    }
    pbe.dgsolv = true;
    pbe.flag_pbe = true;
    return option_consumed();
  }
  if (arg.rfind("-pbe-dgsolv=", 0) == 0 ||
      arg.rfind("--pbe-dgsolv=", 0) == 0) {
    pbe.dgsolv =
        parse_bool_equals(arg.substr(arg.find('=') + 1), "-pbe-dgsolv");
    if (pbe.dgsolv)
      pbe.flag_pbe = true;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-test-pqr", "--pbe-test-pqr"},
                           common_value, i)) {
    pbe.flag_pbe = true;
    pbe.test_pqr = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-binding-cycle", "--pbe-binding-cycle"},
                           common_value, i)) {
    pbe.binding_cycle = require_choice(
        common_value, {"off", "pqr-components"}, "-pbe-binding-cycle");
    if (pbe.binding_cycle != "off")
      pbe.flag_pbe = true;
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-binding-receptor-pqr", "--pbe-binding-receptor-pqr"},
          common_value, i)) {
    pbe.binding_receptor_pqr = common_value;
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-binding-ligand-pqr", "--pbe-binding-ligand-pqr"},
          common_value, i)) {
    pbe.binding_ligand_pqr = common_value;
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-test-grid-size", "--pbe-test-grid-size"},
                           common_value, i)) {
    pbe.test_grid_size = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-test-center", "--pbe-test-center"},
                           common_value, i)) {
    parse_pbe_test_center(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-parity-dump", "--pbe-parity-dump"},
                           common_value, i)) {
    pbe.parity_dump =
        require_choice(common_value, {"off", "summary", "points", "all"},
                       "-pbe-parity-dump");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-parity-map-dump", "--pbe-parity-map-dump"},
                           common_value, i)) {
    pbe.parity_map_dump = require_choice(
        common_value, {"off", "samples", "full"}, "-pbe-parity-map-dump");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-test-charge-grid", "--pbe-test-charge-grid"},
          common_value, i)) {
    pbe.test_charge_grid = require_choice(common_value, {"native", "oracle"},
                                          "-pbe-test-charge-grid");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-test-charge-grid-diff", "--pbe-test-charge-grid-diff"},
          common_value, i)) {
    pbe.test_charge_grid_diff =
        require_choice(common_value, {"off", "samples", "full"},
                       "-pbe-test-charge-grid-diff");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-test-surface-map", "--pbe-test-surface-map"},
          common_value, i)) {
    pbe.test_surface_map = require_choice(common_value, {"native", "oracle"},
                                          "-pbe-test-surface-map");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-test-midpoint-epsilon", "--pbe-test-midpoint-epsilon"},
          common_value, i)) {
    pbe.test_midpoint_epsilon = require_choice(
        common_value, {"native", "oracle"}, "-pbe-test-midpoint-epsilon");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-test-midpoint-diff", "--pbe-test-midpoint-diff"},
          common_value, i)) {
    pbe.test_midpoint_diff = require_choice(
        common_value, {"off", "samples", "full"}, "-pbe-test-midpoint-diff");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-test-vdwms-remap", "--pbe-test-vdwms-remap"},
          common_value, i)) {
    pbe.test_vdwms_remap = require_choice(common_value, {"auto", "off"},
                                          "-pbe-test-vdwms-remap");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-test-boundary-map", "--pbe-test-boundary-map"},
          common_value, i)) {
    pbe.test_boundary_map = require_choice(common_value, {"native", "oracle"},
                                           "-pbe-test-boundary-map");
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-test-field-units", "--pbe-test-field-units"},
          common_value, i)) {
    pbe.test_field_units =
        require_choice(common_value, {"native", "delphi-dimensionless"},
                       "-pbe-test-field-units");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-test-operator-convention", "--pbe-test-operator-convention"},
          common_value, i)) {
    pbe.test_operator_convention = require_choice(
        common_value, {"native", "delphi"}, "-pbe-test-operator-convention");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-test-rhs-map", "--pbe-test-rhs-map"},
                           common_value, i)) {
    pbe.test_rhs_map = require_choice(common_value, {"native", "oracle"},
                                      "-pbe-test-rhs-map");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-test-screen-map", "--pbe-test-screen-map"},
                           common_value, i)) {
    pbe.test_screen_map = require_choice(common_value, {"native", "oracle"},
                                         "-pbe-test-screen-map");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-test-denom-map", "--pbe-test-denom-map"},
                           common_value, i)) {
    pbe.test_denom_map = require_choice(common_value, {"native", "oracle"},
                                        "-pbe-test-denom-map");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-test-nl-source", "--pbe-test-nl-source"},
                           common_value, i)) {
    pbe.test_nl_source =
        require_choice(common_value, {"titan", "delphi-dimensionless"},
                       "-pbe-test-nl-source");
    return option_consumed();
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
        return option_consumed();
      }
    }
    pbe.test_newton_derivative = true;
    return option_consumed();
  }
  if (arg.rfind("-pbe-test-newton-derivative=", 0) == 0 ||
      arg.rfind("--pbe-test-newton-derivative=", 0) == 0) {
    pbe.test_newton_derivative = parse_bool_equals(
        arg.substr(arg.find('=') + 1), "-pbe-test-newton-derivative");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-test-energy-convention", "--pbe-test-energy-convention"},
          common_value, i)) {
    pbe.test_energy_convention = require_choice(
        common_value,
        {"auto", "atom", "grid", "delphi-corrected", "delphi-iscm"},
        "-pbe-test-energy-convention");
    return option_consumed();
  }
  return {};
}
