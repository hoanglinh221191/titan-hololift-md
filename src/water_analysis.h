#ifndef WATER_ANALYSIS_H
#define WATER_ANALYSIS_H

#include "MOL.h"
#include "info.h"
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Custom hash for std::tuple<size_t, size_t, size_t>
struct BridgeHash {
  size_t operator()(const std::tuple<size_t, size_t, size_t> &t) const {
    size_t h1 = std::hash<size_t>{}(std::get<0>(t));
    size_t h2 = std::hash<size_t>{}(std::get<1>(t));
    size_t h3 = std::hash<size_t>{}(std::get<2>(t));
    // Combine hashes (boost-style)
    h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
    h1 ^= h3 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
    return h1;
  }
};

using BridgeTuple = std::tuple<size_t, size_t, size_t>;
using BridgeSet = std::unordered_set<BridgeTuple, BridgeHash>;
using BridgeCountMap = std::unordered_map<BridgeTuple, size_t, BridgeHash>;
using BridgeDurMap = std::unordered_map<BridgeTuple, int, BridgeHash>;

class WaterAnalyzer {
private:
  size_t global_total_frames;
  BridgeCountMap global_bridge_counts;        // B3: unordered_map
  std::vector<int> global_all_survival_times;

  BridgeDurMap bridge_duration;               // B3: unordered_map
  BridgeSet previous_bridges;                 // B3: unordered_set

  std::unordered_map<size_t, std::string> pro_id_to_name;  // B3
  std::unordered_map<size_t, std::string> lig_id_to_name;  // B3

  // B1: reusable buffer — avoid per-water allocation
  std::vector<std::pair<size_t, double>> valid_pro_buf_;

  // B4: persistent timeseries output stream
  std::ofstream timeseries_out_;
  std::string timeseries_path_;

  // B2: write protein/ligand PDB only once
  bool pdb_solute_written_{false};

public:
  WaterAnalyzer();

  void reset_trajectory_state();

  void process_frame(const MOL &pro_frame, const MOL &lig_frame,
                     const MOL &water_frame,
                     const std::vector<size_t> &rec_heavy_atoms,
                     const std::vector<size_t> &lig_heavy_atoms,
                     const std::vector<mol_info> &pro_info,
                     const std::vector<mol_info> &lig_info, size_t input_name,
                     size_t num_snap, const std::string &timeseries_file_path,
                     const std::string &water_pdb_filename);

  void finalize_trajectory();

  void export_global_summary(const std::string &summary_file_path,
                             const std::string &survival_file_path);
};

#endif // WATER_ANALYSIS_H
