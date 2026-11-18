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

arguments::OptionStep arguments::parse_options_bie(int argc, char **argv,
                                                   int &i,
                                                   const std::string &arg,
                                                   std::string &common_value) {
  if (consume_option_value(
          arg,
          {"-pbe-bie-cartesian-k4-order", "--pbe-bie-cartesian-k4-order"},
          common_value, i)) {
    pbe.tabi_cartesian_k4_order = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-cartesian-k4-theta", "--pbe-bie-cartesian-k4-theta"},
          common_value, i)) {
    pbe.tabi_cartesian_k4_theta = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-cartesian-k4-geometry-theta",
                            "--pbe-bie-cartesian-k4-geometry-theta"},
                           common_value, i)) {
    pbe.tabi_cartesian_k4_geometry_theta = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-cartesian-k4-tail-tol",
                            "--pbe-bie-cartesian-k4-tail-tol"},
                           common_value, i)) {
    pbe.tabi_cartesian_k4_tail_tol = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-fmm-cartesian-audit", "--pbe-bie-fmm-cartesian-audit"},
          common_value, i)) {
    pbe.tabi_fmm_cartesian_audit = require_choice(
        common_value, {"off", "on"}, "-pbe-bie-fmm-cartesian-audit");
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-fmm-local", "--pbe-bie-fmm-local"},
                           common_value, i)) {
    pbe.tabi_fmm_local = require_choice(
        common_value, {"off", "audit-p4", "single-matvec-p4", "gmres-p4"},
        "-pbe-bie-fmm-local");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-fmm-local-refine", "--pbe-bie-fmm-local-refine"},
          common_value, i)) {
    pbe.tabi_fmm_local_refine = require_choice(
        common_value,
        {"off", "p3-p4-tail", "operator-tail-static", "operator-tail-linf"},
        "-pbe-bie-fmm-local-refine");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-fmm-local-refine-tol",
                            "--pbe-bie-fmm-local-refine-tol"},
                           common_value, i)) {
    pbe.tabi_fmm_local_refine_tol = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-fmm-local-cache", "--pbe-bie-fmm-local-cache"},
          common_value, i)) {
    pbe.tabi_fmm_local_cache = require_choice(
        common_value,
        {"off", "kernel-jet", "green-jet-compressed", "green-jet-tiled"},
        "-pbe-bie-fmm-local-cache");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-fmm-local-cache-budget-mb",
                            "--pbe-bie-fmm-local-cache-budget-mb"},
                           common_value, i)) {
    pbe.tabi_fmm_local_cache_budget_mb = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-fmm-local-fallback", "--pbe-bie-fmm-local-fallback"},
          common_value, i)) {
    pbe.tabi_fmm_local_fallback =
        require_choice(common_value, {"scalar", "packed-avx2"},
                       "-pbe-bie-fmm-local-fallback");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-fmm-local-plan-builder",
                            "--pbe-bie-fmm-local-plan-builder"},
                           common_value, i)) {
    pbe.tabi_fmm_local_plan_builder = require_choice(
        common_value,
        {"legacy-rescan", "metric-cache", "rescan", "cached-metrics"},
        "-pbe-bie-fmm-local-plan-builder");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-fmm-local-refine-probe-count",
                            "--pbe-bie-fmm-local-refine-probe-count"},
                           common_value, i)) {
    pbe.tabi_fmm_local_refine_probe_count = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-fmm-local-refine-probe-mode",
                            "--pbe-bie-fmm-local-refine-probe-mode"},
                           common_value, i)) {
    pbe.tabi_fmm_local_refine_probe_mode =
        require_choice(common_value, {"rademacher", "solution", "mixed"},
                       "-pbe-bie-fmm-local-refine-probe-mode");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-fmm-local-geometry-guard",
                            "--pbe-bie-fmm-local-geometry-guard"},
                           common_value, i)) {
    pbe.tabi_fmm_local_geometry_guard =
        require_choice(common_value, {"off", "strict-geometry-mac"},
                       "-pbe-bie-fmm-local-geometry-guard");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-fmm-local-geometry-theta",
                            "--pbe-bie-fmm-local-geometry-theta"},
                           common_value, i)) {
    pbe.tabi_fmm_local_geometry_theta = stod(common_value);
    return option_consumed();
  }
  if (consume_bool_option(arg,
                          {"-pbe-bie-fmm-local-ilu-cap-sweep",
                           "--pbe-bie-fmm-local-ilu-cap-sweep"},
                          pbe.tabi_fmm_local_ilu_cap_sweep, i)) {
    return option_consumed();
  }
  if (consume_bool_option(arg,
                          {"-pbe-bie-fmm-local-geometry-sweep",
                           "--pbe-bie-fmm-local-geometry-sweep"},
                          pbe.tabi_fmm_local_geometry_sweep, i)) {
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-interaction-mode", "--pbe-bie-interaction-mode"},
          common_value, i)) {
    pbe.tabi_interaction_mode =
        require_choice(common_value, {"auto", "cached", "onthefly"},
                       "-pbe-bie-interaction-mode");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-node-bounds", "--pbe-bie-node-bounds"},
          common_value, i)) {
    pbe.tabi_node_bounds = require_choice(
        common_value, {"moment", "tight-aabb"}, "-pbe-bie-node-bounds");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-tree-builder", "--pbe-bie-tree-builder"},
          common_value, i)) {
    pbe.tabi_tree_builder =
        require_choice(common_value, {"grouped-morton", "topdown-aabb"},
                       "-pbe-bie-tree-builder");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-expansion-center", "--pbe-bie-expansion-center"},
          common_value, i)) {
    pbe.tabi_expansion_center = require_choice(
        common_value, {"legacy", "area-centroid", "aabb-center"},
        "-pbe-bie-expansion-center");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-fragmentation-safety",
                            "--pbe-bie-fragmentation-safety"},
                           common_value, i)) {
    pbe.tabi_fragmentation_safety =
        require_choice(common_value, {"off", "audit", "split"},
                       "-pbe-bie-fragmentation-safety");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-mac-safety", "--pbe-bie-mac-safety"},
                           common_value, i)) {
    pbe.tabi_mac_safety =
        require_choice(common_value, {"off", "audit", "physical-gap"},
                       "-pbe-bie-mac-safety");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-mac-gap-factor", "--pbe-bie-mac-gap-factor"},
          common_value, i)) {
    pbe.tabi_mac_gap_factor = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-moment-audit", "--pbe-bie-moment-audit"},
          common_value, i)) {
    pbe.tabi_moment_audit =
        require_choice(common_value, {"off", "sampled", "full-small"},
                       "-pbe-bie-moment-audit");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-moment-construction", "--pbe-bie-moment-construction"},
          common_value, i)) {
    pbe.tabi_moment_construction =
        require_choice(common_value, {"translated", "direct-node"},
                       "-pbe-bie-moment-construction");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-moment-sum", "--pbe-bie-moment-sum"},
                           common_value, i)) {
    pbe.tabi_moment_sum = require_choice(common_value, {"naive", "kahan"},
                                         "-pbe-bie-moment-sum");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-convergence-audit", "--pbe-bie-convergence-audit"},
          common_value, i)) {
    pbe.tabi_convergence_audit =
        require_choice(common_value, {"off", "sampled", "full-small"},
                       "-pbe-bie-convergence-audit");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-convergence-safety", "--pbe-bie-convergence-safety"},
          common_value, i)) {
    pbe.tabi_convergence_safety =
        require_choice(common_value, {"off", "audit", "split"},
                       "-pbe-bie-convergence-safety");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-convergence-rho-cap", "--pbe-bie-convergence-rho-cap"},
          common_value, i)) {
    pbe.tabi_convergence_rho_cap = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-mac-aspect-cap", "--pbe-bie-mac-aspect-cap"},
          common_value, i)) {
    pbe.tabi_mac_aspect_cap = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-estimator",
                            "--pbe-bie-convergence-estimator"},
                           common_value, i)) {
    pbe.tabi_convergence_estimator =
        require_choice(common_value, {"off", "moment-tail", "hybrid"},
                       "-pbe-bie-convergence-estimator");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-tail-threshold",
                            "--pbe-bie-convergence-tail-threshold"},
                           common_value, i)) {
    pbe.tabi_convergence_tail_threshold = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-estimator-error-cap",
                            "--pbe-bie-convergence-estimator-error-cap"},
                           common_value, i)) {
    pbe.tabi_convergence_estimator_error_cap = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-unstable-rho-cap",
                            "--pbe-bie-convergence-unstable-rho-cap"},
                           common_value, i)) {
    pbe.tabi_convergence_unstable_rho_cap = stod(common_value);
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-dielectric-amp",
                            "--pbe-bie-convergence-dielectric-amp"},
                           common_value, i)) {
    pbe.tabi_convergence_dielectric_amp = require_choice(
        common_value, {"off", "on"}, "-pbe-bie-convergence-dielectric-amp");
    return option_consumed();
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
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-split-policy",
                            "--pbe-bie-convergence-split-policy"},
                           common_value, i)) {
    pbe.tabi_convergence_split_policy =
        require_choice(common_value, {"global", "k3-hot"},
                       "-pbe-bie-convergence-split-policy");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-split-selectivity",
                            "--pbe-bie-convergence-split-selectivity"},
                           common_value, i)) {
    pbe.tabi_convergence_split_selectivity = require_choice(
        common_value, {"legacy", "absolute-aware", "channel-absolute-aware"},
        "-pbe-bie-convergence-split-selectivity");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-split-absolute-floor",
                            "--pbe-bie-convergence-split-absolute-floor"},
                           common_value, i)) {
    pbe.tabi_convergence_split_absolute_floor = parse_double_range(
        common_value, "-pbe-bie-convergence-split-absolute-floor", 0.0,
        1.0e-4);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-overreject-audit",
                            "--pbe-bie-convergence-overreject-audit"},
                           common_value, i)) {
    pbe.tabi_convergence_overreject_audit =
        require_choice(common_value, {"off", "sampled", "full-small"},
                       "-pbe-bie-convergence-overreject-audit");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-overreject-detail",
                            "--pbe-bie-convergence-overreject-detail"},
                           common_value, i)) {
    pbe.tabi_convergence_overreject_detail =
        require_choice(common_value, {"off", "summary", "sampled-detail"},
                       "-pbe-bie-convergence-overreject-detail");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-shape-audit",
                            "--pbe-bie-convergence-shape-audit"},
                           common_value, i)) {
    pbe.tabi_convergence_shape_audit =
        require_choice(common_value, {"off", "summary", "sampled-detail"},
                       "-pbe-bie-convergence-shape-audit");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-kernel-response-audit",
                            "--pbe-bie-convergence-kernel-response-audit"},
                           common_value, i)) {
    pbe.tabi_convergence_kernel_response_audit =
        require_choice(common_value, {"off", "summary", "sampled-detail"},
                       "-pbe-bie-convergence-kernel-response-audit");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-channel-missed-semantics",
                            "--pbe-bie-channel-missed-semantics"},
                           common_value, i)) {
    pbe.tabi_channel_missed_semantics =
        require_choice(common_value, {"legacy", "accepted-only"},
                       "-pbe-bie-channel-missed-semantics");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-accepted-miss-diagnostics",
                            "--pbe-bie-accepted-miss-diagnostics"},
                           common_value, i)) {
    pbe.tabi_accepted_miss_diagnostics =
        require_choice(common_value, {"off", "summary", "sampled-detail"},
                       "-pbe-bie-accepted-miss-diagnostics");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-accepted-miss-absolute-floor",
                            "--pbe-bie-accepted-miss-absolute-floor"},
                           common_value, i)) {
    pbe.tabi_accepted_miss_absolute_floor = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-convergence-k3-accepted-rho-cap",
                            "--pbe-bie-convergence-k3-accepted-rho-cap"},
                           common_value, i)) {
    pbe.tabi_convergence_k3_accepted_rho_cap = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-mac-distance-policy", "--pbe-bie-mac-distance-policy"},
          common_value, i)) {
    pbe.tabi_mac_distance_policy =
        require_choice(common_value, {"center", "surface-margin"},
                       "-pbe-bie-mac-distance-policy");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-mac-surface-margin-factor",
                            "--pbe-bie-mac-surface-margin-factor"},
                           common_value, i)) {
    pbe.tabi_mac_surface_margin_factor = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-mac-surface-margin-policy",
                            "--pbe-bie-mac-surface-margin-policy"},
                           common_value, i)) {
    pbe.tabi_mac_surface_margin_policy =
        require_choice(common_value, {"all", "adaptive", "risk-scaled"},
                       "-pbe-bie-mac-surface-margin-policy");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-mac-surface-margin-aspect-threshold",
                            "--pbe-bie-mac-surface-margin-aspect-threshold"},
                           common_value, i)) {
    pbe.tabi_mac_surface_margin_aspect_threshold = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-mac-surface-margin-rho-trigger",
                            "--pbe-bie-mac-surface-margin-rho-trigger"},
                           common_value, i)) {
    pbe.tabi_mac_surface_margin_rho_trigger = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-mac-surface-margin-soft-aspect-threshold",
           "--pbe-bie-mac-surface-margin-soft-aspect-threshold"},
          common_value, i)) {
    pbe.tabi_mac_surface_margin_soft_aspect_threshold = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-mac-surface-margin-soft-rho-trigger",
                            "--pbe-bie-mac-surface-margin-soft-rho-trigger"},
                           common_value, i)) {
    pbe.tabi_mac_surface_margin_soft_rho_trigger = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-mac-surface-margin-soft-factor",
                            "--pbe-bie-mac-surface-margin-soft-factor"},
                           common_value, i)) {
    pbe.tabi_mac_surface_margin_soft_factor = stod(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-target-batch-size", "--pbe-bie-target-batch-size"},
          common_value, i)) {
    pbe.tabi_target_batch_size = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-leaf-pair-min-source-panels",
                            "--pbe-bie-leaf-pair-min-source-panels"},
                           common_value, i)) {
    pbe.tabi_leaf_pair_min_source_panels = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-leaf-pair-min-targets",
                            "--pbe-bie-leaf-pair-min-targets"},
                           common_value, i)) {
    pbe.tabi_leaf_pair_min_targets = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-leaf-pair-max-lookahead",
                            "--pbe-bie-leaf-pair-max-lookahead"},
                           common_value, i)) {
    pbe.tabi_leaf_pair_max_lookahead = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-expansion-audit", "--pbe-bie-expansion-audit"},
          common_value, i)) {
    pbe.tabi_expansion_audit =
        require_choice(common_value, {"off", "sampled", "full-small"},
                       "-pbe-bie-expansion-audit");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-expansion-audit-channel",
                            "--pbe-bie-expansion-audit-channel"},
                           common_value, i)) {
    pbe.tabi_expansion_audit_channel =
        require_choice(common_value, {"off", "summary", "sampled-detail"},
                       "-pbe-bie-expansion-audit-channel");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-pre-solver-progress", "--pbe-bie-pre-solver-progress"},
          common_value, i)) {
    pbe.tabi_pre_solver_progress = require_choice(
        common_value, {"off", "on"}, "-pbe-bie-pre-solver-progress");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-expansion-audit-sample-count",
                            "--pbe-bie-expansion-audit-sample-count"},
                           common_value, i)) {
    pbe.tabi_expansion_audit_sample_count = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-expansion-audit-max-panels",
                            "--pbe-bie-expansion-audit-max-panels"},
                           common_value, i)) {
    pbe.tabi_expansion_audit_max_panels = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-matvec-audit", "--pbe-bie-matvec-audit"},
          common_value, i)) {
    pbe.tabi_matvec_audit = require_choice(
        common_value,
        {"off", "sampled", "sampled-detail", "targets", "full-small"},
        "-pbe-bie-matvec-audit");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-matvec-audit-sample-count",
                            "--pbe-bie-matvec-audit-sample-count"},
                           common_value, i)) {
    pbe.tabi_matvec_audit_sample_count = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-node-patch-operator-audit",
           "--pbe-bie-node-patch-operator-audit"},
          common_value, i)) {
    pbe.tabi_node_patch_operator_audit = require_choice(
        common_value, {"on", "off"},
        "-pbe-bie-node-patch-operator-audit");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-diagnostics", "--pbe-bie-diagnostics",
                            "--diagnostics"},
                           common_value, i)) {
    pbe.tabi_diagnostics = require_choice(
        common_value, {"off", "audit", "full"}, "-pbe-bie-diagnostics");
    return option_consumed();
  }
  if (consume_option_value(arg, {"-pbe-bie-progress", "--pbe-bie-progress"},
                           common_value, i)) {
    pbe.tabi_progress = require_choice(
        common_value, {"off", "auto", "detail"}, "-pbe-bie-progress");
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-progress-interval", "--pbe-bie-progress-interval"},
          common_value, i)) {
    pbe.tabi_progress_interval = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-pbe-bie-gmres-restart", "--pbe-bie-gmres-restart"},
          common_value, i)) {
    pbe.tabi_gmres_restart = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-true-residual-probe", "--pbe-bie-true-residual-probe"},
          common_value, i)) {
    pbe.tabi_true_residual_probe =
        require_choice(common_value, {"restart", "interval", "off"},
                       "-pbe-bie-true-residual-probe");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-true-residual-probe-scope",
                            "--pbe-bie-true-residual-probe-scope"},
                           common_value, i)) {
    pbe.tabi_true_residual_probe_scope =
        require_choice(common_value, {"full", "sampled", "adaptive"},
                       "-pbe-bie-true-residual-probe-scope");
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-true-residual-probe-interval",
                            "--pbe-bie-true-residual-probe-interval"},
                           common_value, i)) {
    pbe.tabi_true_residual_probe_interval = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(arg,
                           {"-pbe-bie-true-residual-sample-count",
                            "--pbe-bie-true-residual-sample-count"},
                           common_value, i)) {
    pbe.tabi_true_residual_sample_count = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg,
          {"-pbe-bie-energy-projection", "--pbe-bie-energy-projection"},
          common_value, i)) {
    pbe.tabi_energy_projection = require_choice(
        common_value, {"boundary", "born-check", "charge-potential-debug"},
        "-pbe-bie-energy-projection");
    return option_consumed();
  }
  return {};
}
