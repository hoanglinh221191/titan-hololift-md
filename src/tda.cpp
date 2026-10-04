#include "tda.h"
#include <algorithm>
#include <fstream>
#include <random>
#include <unordered_set>

TDAAnalyzer::~TDAAnalyzer() { flush_TDA_pointcloud(); }

void TDAAnalyzer::flush_TDA_pointcloud() {
  if (reservoir_output_csv_.empty() || reservoir_max_points_ == 0)
    return;
  std::ofstream out(reservoir_output_csv_, std::ios::out | std::ios::trunc);
  if (!out.is_open())
    return;
  out << "Traj_ID,Frame,X,Y,Z\n";
  for (const auto &row : reservoir_rows_) {
    out << row.traj_id << "," << row.frame << "," << row.x << "," << row.y
        << "," << row.z << "\n";
  }
}

void TDAAnalyzer::add_point_row(const PointRow &row, std::size_t max_points,
                                const std::string &output_csv) {
  if (max_points == 0)
    return;
  if (!reservoir_output_csv_.empty() && reservoir_output_csv_ != output_csv) {
    flush_TDA_pointcloud();
    reservoir_rows_.clear();
    reservoir_seen_ = 0;
  }
  reservoir_output_csv_ = output_csv;
  reservoir_max_points_ = max_points;
  ++reservoir_seen_;
  if (reservoir_rows_.size() < max_points) {
    reservoir_rows_.push_back(row);
    return;
  }
  std::uniform_int_distribution<std::size_t> dist(0, reservoir_seen_ - 1);
  const std::size_t pick = dist(reservoir_rng_);
  if (pick < max_points)
    reservoir_rows_[pick] = row;
}

void TDAAnalyzer::export_TDA_pointcloud(
    const Multi_MOL &pro_in, const Multi_MOL &lig, int traj_id, int num_snap,
    const std::vector<int> &target_idx_1, const std::vector<int> &target_idx_2,
    const arguments &Arguments, const std::string &output_csv) {
  std::unordered_set<int> interface_1(target_idx_1.begin(), target_idx_1.end());
  std::unordered_set<int> interface_2(target_idx_2.begin(), target_idx_2.end());
  bool filter_1 = !interface_1.empty();
  bool filter_2 = !interface_2.empty();

  const std::size_t max_points = Arguments.tda_max_points;
  if (max_points > 0 && num_snap == 0) {
    reservoir_rows_.clear();
    reservoir_seen_ = 0;
    reservoir_max_points_ = max_points;
    reservoir_output_csv_ = output_csv;
    const bool new_report =
        !static_cast<bool>(std::ifstream("memory_report.dat"));
    std::ofstream report("memory_report.dat", std::ios::app);
    if (report.is_open()) {
      if (new_report) {
        report << "Module,Mode,Nodes,Frames,EstimatedExtraGiB,BudgetGiB,Pairs,"
                  "Reason\n";
      }
      report << "TDA,reservoir,0,0,0,0," << max_points
             << ",point-cloud cap\n";
    }
  }

  std::fstream out;
  if (max_points == 0) {
    std::ios_base::openmode mode =
        (num_snap == 0) ? (std::ios::out | std::ios::trunc) : std::ios::app;
    out.open(output_csv, mode);
    if (!out.is_open())
      return;
  }

  // Output header if file is empty or it's the first frame
  if (max_points == 0 && num_snap == 0) {
    out << "Traj_ID,Frame,X,Y,Z\n";
  }

  // Determine type for receptor (mol 1)
  bool is_rec_lig =
      (Arguments.mol_1_type == "ligand" || Arguments.mol_1_type == "lig");

  // Process Receptor
  if (!pro_in.m_mol.empty() && num_snap < pro_in.m_mol.size()) {
    const auto &mol = pro_in.m_mol[num_snap];
    for (size_t i = 0; i < pro_in.info.size(); ++i) {
      // Check heavy atom
      if (pro_in.info[i].get_atom().empty() ||
          pro_in.info[i].get_atom()[0] == 'H')
        continue;

      int idx_to_check =
          is_rec_lig ? static_cast<int>(i) : pro_in.info[i].get_resnr();

      if (!filter_1 || interface_1.find(idx_to_check) != interface_1.end()) {
        if (max_points == 0) {
          out << traj_id << "," << num_snap << "," << mol.pos.x[i] << ","
              << mol.pos.y[i] << "," << mol.pos.z[i] << "\n";
        } else {
          add_point_row(PointRow{traj_id, num_snap, mol.pos.x[i], mol.pos.y[i],
                                 mol.pos.z[i]},
                        max_points, output_csv);
        }
      }
    }
  }

  // Process Ligand
  bool is_lig_lig =
      (Arguments.mol_2_type == "ligand" || Arguments.mol_2_type == "lig");
  if (!lig.m_mol.empty()) {
    size_t l_snap = (lig.m_mol.size() > 1) ? num_snap : 0;
    if (l_snap < lig.m_mol.size()) {
      const auto &l_mol = lig.m_mol[l_snap];
      for (size_t i = 0; i < lig.info.size(); ++i) {
        // Check heavy atom
        if (lig.info[i].get_atom().empty() || lig.info[i].get_atom()[0] == 'H')
          continue;

        int idx_to_check =
            is_lig_lig ? static_cast<int>(i) : lig.info[i].get_resnr();

        if (!filter_2 || interface_2.find(idx_to_check) != interface_2.end()) {
          if (max_points == 0) {
            out << traj_id << "," << num_snap << "," << l_mol.pos.x[i] << ","
                << l_mol.pos.y[i] << "," << l_mol.pos.z[i] << "\n";
          } else {
            add_point_row(PointRow{traj_id, num_snap, l_mol.pos.x[i],
                                   l_mol.pos.y[i], l_mol.pos.z[i]},
                          max_points, output_csv);
          }
        }
      }
    }
  }
}
