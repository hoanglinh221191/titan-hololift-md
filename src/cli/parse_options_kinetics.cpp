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

arguments::OptionStep arguments::parse_options_kinetics(int argc, char **argv,
                                                        int &i,
                                                        const std::string &arg,
                                                        std::string &common_value) {
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
  return {};
}
