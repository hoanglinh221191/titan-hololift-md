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

arguments::OptionStep arguments::parse_options_analysis(int argc, char **argv,
                                                        int &i,
                                                        const std::string &arg,
                                                        std::string &common_value) {
  if (consume_option_value(arg, {"-gbsa-perf", "--gbsa-perf"}, common_value,
                           i)) {
    gbsa_perf = require_choice(common_value, {"off", "summary", "detail"},
                               "-gbsa-perf");
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-gbsa-sasa-points", "--gbsa-sasa-points"},
          common_value, i)) {
    gbsa_sasa_points = stoi(common_value);
    return option_consumed();
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
    return option_consumed();
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
    return option_consumed();
  }
  if (consume_option_value(arg, {"-gbsa-ie-block", "--gbsa-ie-block"},
                           common_value, i)) {
    gbsa_ie_block = lowercase_copy(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-gbsa-ie-min-samples", "--gbsa-ie-min-samples"},
          common_value, i)) {
    gbsa_ie_min_samples = stoi(common_value);
    return option_consumed();
  }
  if (consume_option_value(
          arg, {"-gbsa-ie-filter-sigma", "--gbsa-ie-filter-sigma"},
          common_value, i)) {
    gbsa_ie_filter_sigma = stod(common_value);
    return option_consumed();
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
  return {};
}
