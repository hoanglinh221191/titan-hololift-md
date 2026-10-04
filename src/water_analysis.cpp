#include "water_analysis.h"
#include <algorithm>
#include <cmath>
#include <iostream>

WaterAnalyzer::WaterAnalyzer() : global_total_frames(0) {}

void WaterAnalyzer::reset_trajectory_state() {
  bridge_duration.clear();
  previous_bridges.clear();
  // B4: close previous trajectory's timeseries file
  if (timeseries_out_.is_open())
    timeseries_out_.close();
  timeseries_path_.clear();
}

void WaterAnalyzer::process_frame(const MOL &pro_frame, const MOL &lig_frame,
                                  const MOL &w_frame,
                                  const std::vector<size_t> &rec_heavy_atoms,
                                  const std::vector<size_t> &lig_heavy_atoms,
                                  const std::vector<mol_info> &pro_info,
                                  const std::vector<mol_info> &lig_info,
                                  size_t input_name, size_t num_snap,
                                  const std::string &timeseries_file_path,
                                  const std::string &water_pdb_filename) {

  BridgeSet current_bridges;
  std::unordered_set<size_t> filtered_waters; // B3: unordered_set
  size_t n_water_atoms = w_frame.pos.x.size();

  // B5: use ostringstream instead of string concatenation
  std::ostringstream frame_bridge_buffer;

  // B4: open timeseries file persistently (lazy open / path change)
  if (!timeseries_out_.is_open() || timeseries_path_ != timeseries_file_path) {
    if (timeseries_out_.is_open())
      timeseries_out_.close();
    timeseries_out_.open(timeseries_file_path, std::ios::app);
    timeseries_path_ = timeseries_file_path;
  }

  double min_d2_pro = 1e10;
  double min_d2_lig = 1e10;

  for (size_t w_idx = 0; w_idx + 2 < n_water_atoms; w_idx += 3) {
    size_t ox = w_idx;

    // B1: reuse member buffer instead of per-water allocation
    valid_pro_buf_.clear();
    for (size_t p_idx : rec_heavy_atoms) {
      double dx = w_frame.pos.x[ox] - pro_frame.pos.x[p_idx];
      double dy = w_frame.pos.y[ox] - pro_frame.pos.y[p_idx];
      double dz = w_frame.pos.z[ox] - pro_frame.pos.z[p_idx];
      double d2 = dx * dx + dy * dy + dz * dz;

      if (d2 < min_d2_pro)
        min_d2_pro = d2;

      if (d2 <= 12.25) { // 3.5^2
        valid_pro_buf_.emplace_back(p_idx, d2);
      }
    }

    // Check against ligand heavy atoms
    for (size_t l_idx : lig_heavy_atoms) {
      double dx = w_frame.pos.x[ox] - lig_frame.pos.x[l_idx];
      double dy = w_frame.pos.y[ox] - lig_frame.pos.y[l_idx];
      double dz = w_frame.pos.z[ox] - lig_frame.pos.z[l_idx];
      double d2 = dx * dx + dy * dy + dz * dz;

      if (d2 < min_d2_lig)
        min_d2_lig = d2;

      if (!valid_pro_buf_.empty() && d2 <= 12.25) {
        filtered_waters.insert(ox);
        for (auto &p_pair : valid_pro_buf_) {
          size_t p_idx = p_pair.first;
          double d2_pro = p_pair.second;

          current_bridges.emplace(std::make_tuple(w_idx / 3, p_idx, l_idx));

          // B6: guard pro_id_to_name rebuild — only insert if not present
          if (pro_id_to_name.find(p_idx) == pro_id_to_name.end()) {
            pro_id_to_name[p_idx] = pro_info[p_idx].get_res_name() +
                                    std::to_string(pro_info[p_idx].get_resnr()) +
                                    "@" + pro_info[p_idx].get_atom();
          }
          if (lig_id_to_name.find(l_idx) == lig_id_to_name.end()) {
            lig_id_to_name[l_idx] = lig_info[l_idx].get_res_name() +
                                    std::to_string(lig_info[l_idx].get_resnr()) +
                                    "@" + lig_info[l_idx].get_atom();
          }

          // B5: ostringstream instead of string +=
          frame_bridge_buffer
              << input_name << "," << (num_snap + 1) << ","
              << pro_frame.time_of_snap << "," << (w_idx / 3) << ","
              << pro_id_to_name[p_idx] << "," << lig_id_to_name[l_idx] << ","
              << ((std::sqrt(d2_pro) + std::sqrt(d2)) / 2.0) << ",0.0\n";
        }
      }
    }
  }

  // B4: write to persistent stream (no open/close per frame)
  std::string buf_str = frame_bridge_buffer.str();
  if (!buf_str.empty() && timeseries_out_.is_open()) {
    timeseries_out_ << buf_str;
    timeseries_out_.flush();
  }

  // Track global residence mapping
  for (const auto &wb_tuple : current_bridges) {
    bridge_duration[wb_tuple]++;
    global_bridge_counts[wb_tuple]++;
  }

  global_total_frames++;

  // Write out broken bridges to global survival accumulator
  for (const auto &old_wb : previous_bridges) {
    if (current_bridges.find(old_wb) == current_bridges.end()) {
      global_all_survival_times.emplace_back(bridge_duration[old_wb]);
      bridge_duration.erase(old_wb);
    }
  }
  previous_bridges = std::move(current_bridges);

  // --- PDB output ---
  std::ofstream out_pdb;
  if (num_snap == 0 && !pdb_solute_written_) {
    out_pdb.open(water_pdb_filename, std::ios::out);
  } else {
    out_pdb.open(water_pdb_filename, std::ios::app);
  }

  if (out_pdb.is_open()) {
    out_pdb << "MODEL " << (num_snap + 1) << "\n";
    int atom_idx = 1;

    // B2: write protein/ligand atoms only on first frame
    if (!pdb_solute_written_) {
      // Protein
      for (size_t i = 0; i < pro_frame.pos.x.size(); ++i) {
        std::string chain_str = pro_info[i].get_chain();
        char chain = chain_str.empty() ? ' ' : chain_str[0];
        if (chain == '\0')
          chain = ' ';

        char buf[256];
        std::string a_name = pro_info[i].get_atom();
        if (a_name.length() < 4)
          a_name = " " + a_name;
        std::string chain_str_single(1, chain);
        snprintf(buf, sizeof(buf),
                 "%-6s%5d %-4s %-4s%1s%4d    %8.3f%8.3f%8.3f  1.00  0.00\n",
                 "ATOM  ", atom_idx % 100000, a_name.c_str(),
                 pro_info[i].get_res_name().c_str(), chain_str_single.c_str(),
                 (int)(pro_info[i].get_resnr() % 10000), pro_frame.pos.x[i],
                 pro_frame.pos.y[i], pro_frame.pos.z[i]);
        out_pdb << buf;
        atom_idx++;
      }

      // Ligand
      int last_resnr = 0;
      if (!pro_info.empty())
        last_resnr = pro_info.back().get_resnr();

      for (size_t i = 0; i < lig_frame.pos.x.size(); ++i) {
        std::string chain_str = lig_info[i].get_chain();
        char chain = chain_str.empty() ? ' ' : chain_str[0];
        if (chain == '\0')
          chain = ' ';

        char buf[256];
        std::string a_name = lig_info[i].get_atom();
        if (a_name.length() < 4)
          a_name = " " + a_name;
        std::string chain_str_single(1, chain);
        snprintf(buf, sizeof(buf),
                 "%-6s%5d %-4s %-4s%1s%4d    %8.3f%8.3f%8.3f  1.00  0.00\n",
                 "HETATM", atom_idx % 100000, a_name.c_str(),
                 lig_info[i].get_res_name().c_str(), chain_str_single.c_str(),
                 (int)(lig_info[i].get_resnr() % 10000), lig_frame.pos.x[i],
                 lig_frame.pos.y[i], lig_frame.pos.z[i]);
        out_pdb << buf;
        atom_idx++;
      }
      pdb_solute_written_ = true;
    }

    // Structural Waters (always written, every frame)
    int last_resnr_w = 0;
    if (!lig_info.empty())
      last_resnr_w = lig_info.back().get_resnr();
    else if (!pro_info.empty())
      last_resnr_w = pro_info.back().get_resnr();
    int water_resnr = last_resnr_w + 1;

    for (size_t w_idx : filtered_waters) {
      int r_num = water_resnr % 10000;
      char buf[256];

      // O atom
      snprintf(buf, sizeof(buf),
               "%-6s%5d %-4s %-4s%1s%4d    %8.3f%8.3f%8.3f  1.00  0.00\n",
               "ATOM  ", atom_idx % 100000, " O  ", "WAT ", " ", r_num,
               w_frame.pos.x[w_idx], w_frame.pos.y[w_idx], w_frame.pos.z[w_idx]);
      out_pdb << buf;
      atom_idx++;

      // H1 atom
      if (w_idx + 1 < w_frame.pos.x.size()) {
        snprintf(buf, sizeof(buf),
                 "%-6s%5d %-4s %-4s%1s%4d    %8.3f%8.3f%8.3f  1.00  0.00\n",
                 "ATOM  ", atom_idx % 100000, " H1 ", "WAT ", " ", r_num,
                 w_frame.pos.x[w_idx + 1], w_frame.pos.y[w_idx + 1],
                 w_frame.pos.z[w_idx + 1]);
        out_pdb << buf;
        atom_idx++;
      }

      // H2 atom
      if (w_idx + 2 < w_frame.pos.x.size()) {
        snprintf(buf, sizeof(buf),
                 "%-6s%5d %-4s %-4s%1s%4d    %8.3f%8.3f%8.3f  1.00  0.00\n",
                 "ATOM  ", atom_idx % 100000, " H2 ", "WAT ", " ", r_num,
                 w_frame.pos.x[w_idx + 2], w_frame.pos.y[w_idx + 2],
                 w_frame.pos.z[w_idx + 2]);
        out_pdb << buf;
        atom_idx++;
      }

      water_resnr++;
    }
    out_pdb << "ENDMDL\n";
    out_pdb.close();
  }
}

void WaterAnalyzer::finalize_trajectory() {
  for (const auto &active_wb : previous_bridges) {
    global_all_survival_times.emplace_back(bridge_duration[active_wb]);
  }
  // B4: close timeseries file at end of trajectory
  if (timeseries_out_.is_open())
    timeseries_out_.close();
}

void WaterAnalyzer::export_global_summary(
    const std::string &summary_file_path,
    const std::string &survival_file_path) {
  if (global_total_frames > 0) {
    // 1. Export Global Summary Map
    std::vector<std::pair<BridgeTuple, size_t>>
        sorted_wbs(global_bridge_counts.begin(), global_bridge_counts.end());

    std::sort(sorted_wbs.begin(), sorted_wbs.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    std::ofstream sum_out(summary_file_path);
    if (sum_out.is_open()) {
      sum_out << "Water_ID,Protein_Atom,Ligand_Atom,Global_Count,Total_"
                 "Occupancy_%\n";
      for (const auto &pair : sorted_wbs) {
        double occupancy = (double)pair.second / global_total_frames * 100.0;
        // B6: safe access — use find to avoid inserting empty entries
        auto p_it = pro_id_to_name.find(std::get<1>(pair.first));
        auto l_it = lig_id_to_name.find(std::get<2>(pair.first));
        sum_out << std::get<0>(pair.first) << ","
                << (p_it != pro_id_to_name.end() ? p_it->second : "?") << ","
                << (l_it != lig_id_to_name.end() ? l_it->second : "?") << ","
                << pair.second << "," << occupancy << "\n";
      }
      sum_out.close();
    }

    // 2. Export Global Survival Probability S(t)
    if (!global_all_survival_times.empty()) {
      std::unordered_map<int, int> length_distribution; // B3: unordered_map
      for (int len : global_all_survival_times) {
        length_distribution[len]++;
      }

      int total_bridge_events = global_all_survival_times.size();
      int surviving = total_bridge_events;

      // Find max length
      int max_len = 0;
      for (const auto &kv : length_distribution) {
        if (kv.first > max_len)
          max_len = kv.first;
      }

      std::ofstream surv_out(survival_file_path);
      if (surv_out.is_open()) {
        surv_out << "Lifetime_Frames,S(t)_Fraction\n";

        for (int t = 1; t <= max_len; ++t) {
          double s_t = (double)surviving / total_bridge_events;
          surv_out << t << "," << s_t << "\n";

          auto it = length_distribution.find(t);
          if (it != length_distribution.end()) {
            surviving -= it->second;
          }
        }
        surv_out.close();
      }
    }
  }
}
