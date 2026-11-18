#include "Multi_MOL.h"
#include "distance_v2.h"
#include "folder.h"
#include "force.h"
#include "info.h"
#include "miscellaneous.h"
#include "titan_ui.h"
#include "uti_func.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

using namespace std;

namespace {
std::string snap_count_source_label(const arguments &Arguments) {
  if (Arguments.flag_xdr && !Arguments.traj_file.empty())
    return TitanUI::compact_source_label(Arguments.traj_file, 64);
  return TitanUI::compact_source_label(Arguments.pdb_name, 64);
}

bool &abnormal_snap_warning_emitted() {
  static bool emitted = false;
  return emitted;
}
} // namespace

int Multi_MOL::tran_multi_mol(std::span<double> TranVec) {
  for (size_t i{}; i < m_mol.size(); ++i)
    for (size_t j{}; j < m_mol[i].size(); ++j) {
      m_mol[i].pos.x[j] += TranVec[0];
      m_mol[i].pos.y[j] += TranVec[1];
      m_mol[i].pos.z[j] += TranVec[2];
    }
  return 0;
}
int Multi_MOL::align_to_box() {
  for (size_t i{}; i < m_mol.size(); ++i) {
    if (m_mol[i].size() == 0)
      continue;
    double v1[3]{}, v2[3]{}, v3[3]{};
    m_mol[i].Box.get_v1(v1);
    m_mol[i].Box.get_v2(v2);
    m_mol[i].Box.get_v3(v3);

    double com_box[3]{v1[0] / 2., v2[1] / 2., v3[2] / 2.}, com_mol[3]{};

    for (size_t j{}; j < m_mol[i].size(); ++j) {
      com_mol[0] += m_mol[i].pos.x[j];
      com_mol[1] += m_mol[i].pos.y[j];
      com_mol[2] += m_mol[i].pos.z[j];
    }
    for (int j{}; j < 3; ++j)
      com_mol[j] /= m_mol[i].size();
    for (size_t k{}; k < m_mol[i].size(); ++k) {
      m_mol[i].pos.x[k] += com_box[0] - com_mol[0];
      m_mol[i].pos.y[k] += com_box[1] - com_mol[1];
      m_mol[i].pos.z[k] += com_box[2] - com_mol[2];
    }
  }
  return 0;
}
int Multi_MOL::find_box_from_min_max() {
  for (size_t i{}; i < m_mol.size(); ++i) {
    double x[2]{1000., -1000.}, y[2]{1000., -1000.}, z[2]{1000., -1000.};
    for (size_t j{}; j < m_mol[i].size(); ++j) {
      if (m_mol[i].pos.x[j] < x[0])
        x[0] = m_mol[i].pos.x[j];
      if (m_mol[i].pos.x[j] > x[1])
        x[1] = m_mol[i].pos.x[j];
      if (m_mol[i].pos.y[j] < y[0])
        y[0] = m_mol[i].pos.y[j];
      if (m_mol[i].pos.y[j] > y[1])
        y[1] = m_mol[i].pos.y[j];
      if (m_mol[i].pos.z[j] < z[0])
        z[0] = m_mol[i].pos.z[j];
      if (m_mol[i].pos.z[j] > z[1])
        z[1] = m_mol[i].pos.z[j];
    }
    m_mol[i].Box.set_a(x[1] - x[0]);
    m_mol[i].Box.set_b(y[1] - y[0]);
    m_mol[i].Box.set_c(z[1] - z[0]);
  }
  return 0;
}

int align_complex(Multi_MOL &rec, Multi_MOL &lig, int begin, int end) {
  if (rec.m_mol.empty())
    return 0;
  if (lig.m_mol.size() < rec.m_mol.size())
    return 1;

  vector<double> X, Y, Z;
  for (size_t i{}; i < rec.m_mol.size(); ++i) {
    const size_t atom_count = rec.m_mol[i].size() + lig.m_mol[i].size();
    if (atom_count == 0)
      return 1;
    double x{}, y{}, z{};
    for (size_t j{}; j < rec.m_mol[i].size(); ++j) {
      x += rec.m_mol[i].pos.x[j];
      y += rec.m_mol[i].pos.y[j];
      z += rec.m_mol[i].pos.z[j];
    }

    for (size_t j{}; j < lig.m_mol[i].size(); ++j) {
      x += lig.m_mol[i].pos.x[j];
      y += lig.m_mol[i].pos.y[j];
      z += lig.m_mol[i].pos.z[j];
    }
    X.emplace_back(x / atom_count);
    Y.emplace_back(y / atom_count);
    Z.emplace_back(z / atom_count);
  }
  double X_mean{}, Y_mean{}, Z_mean{};
  for (size_t i{}; i < X.size(); ++i) {
    X_mean += X[i];
    Y_mean += Y[i];
    Z_mean += Z[i];
  }
  X_mean /= X.size();
  Y_mean /= Y.size();
  Z_mean /= Z.size();
  for (size_t i{}; i < rec.m_mol.size(); ++i) {
    double delta_x{X_mean - X[i]}, delta_y{Y_mean - Y[i]},
        delta_z{Z_mean - Z[i]};
    for (size_t j{}; j < rec.m_mol[i].size(); ++j) {
      rec.m_mol[i].pos.x[j] += delta_x;
      rec.m_mol[i].pos.y[j] += delta_y;
      rec.m_mol[i].pos.z[j] += delta_z;
    }
    for (size_t j{}; j < lig.m_mol[i].size(); ++j) {
      lig.m_mol[i].pos.x[j] += delta_x;
      lig.m_mol[i].pos.y[j] += delta_y;
      lig.m_mol[i].pos.z[j] += delta_z;
    }
  }
  return 0;
}
int impl_add_coms_2_mols(Multi_MOL &mol1, Multi_MOL &mol2, arguments &Arguments,
                         size_t snap_index) {
  if (Arguments.nt > 1) {
    thread t1(&Multi_MOL::add_coms, &mol1,
              Arguments.flag_topo_1 && !Arguments.flag_use_common_topo,
              snap_index);
    thread t2(&Multi_MOL::add_coms, &mol2,
              Arguments.flag_topo_2 && !Arguments.flag_use_common_topo,
              snap_index);
    t1.join();
    t2.join();
    return 0;
  } else {
    mol1.add_coms(Arguments.flag_topo_1 && !Arguments.flag_use_common_topo,
                  snap_index);
    mol2.add_coms(Arguments.flag_topo_2 && !Arguments.flag_use_common_topo,
                  snap_index);
    return 0;
  }
}

int filt_duplicate_e(vector<vector<data_out>> &data,
                     vector<data_out> &data_processed) {
  // unordered_set<pair<string, size_t>,pair_hash> list_data;

  for (size_t i{}; i < data.size(); ++i)
    for (size_t j{}; j < data[i].size(); ++j) {
      // if(list_data.emplace(data[i][j].chain, data[i][j].resnr).second==true)
      data_processed.emplace_back(data[i][j]);
    }
  return 0;
}

bool check_in_energy_list(vector<data_out> &lg_ener_inter, string chain,
                          size_t resnr) {
  for (size_t i{}; i < lg_ener_inter.size(); ++i) {
    if (chain == lg_ener_inter[i].chain && resnr == lg_ener_inter[i].resnr)
      return true;
  }
  return false;
}

bool sort_my_list_e(const data_out &a, const data_out &b) {
  if (a.chain != b.chain)
    return a.chain < b.chain;
  else
    return a.resnr < b.resnr;
}

template <typename T>
size_t check_in_energy_res(vector<energy_res> &Energy_res, size_t _resnr,
                           string _chain, T &&_e, T &&_v) {
  for (size_t i{}; i < Energy_res.size(); ++i) {
    if (Energy_res[i].resnr == _resnr && Energy_res[i].chain == _chain)
      return i;
  }
  return Energy_res.size();
}

template <typename T, typename T2>
bool add_energy_res(
    unordered_map<pair<string, size_t>, size_t, pair_hash> &map_energy_res,
    vector<energy_res> &Energy_res, size_t _resnr, string _chain,
    string _resname, T &&_e, T &&_v, force &_f_e, force &_f_v, T2 &&_time) {
  if (false ==
      map_energy_res
          .try_emplace(pair<string, size_t>(_chain, _resnr), Energy_res.size())
          .second)
    Energy_res[map_energy_res[pair<string, size_t>(_chain, _resnr)]]
        .emplace_back(_e, _v, _f_e, _f_v, _time);
  else
    Energy_res.emplace_back(_resnr, _chain, _resname, _e, _v, _f_e, _f_v,
                            _time);

  /*size_t pos{check_in_energy_res(Energy_res,_resnr,_chain,_e,_v)};
  if (Energy_res.empty() || pos >= Energy_res.size())
          Energy_res.emplace_back(_resnr, _chain, _resname, _e, _v, _f_e, _f_v,
  _time); else Energy_res[pos].emplace_back(_e, _v, _f_e, _f_v, _time);*/
  return true;
}

int print_force_res(vector<energy_res> &Energy_res, string name, folders &paths,
                    int file_num, const arguments &Arguments) {
  if (Energy_res.empty())
    return 0;
  fstream out(paths.get_path(file_num) + "/" + name + "_out_force_res.dat",
              ios::out);

  out << left << setw(12) << "#Time(ps)";
  for (size_t i{}; i < Energy_res.size(); ++i) {
    string res_prefix = Energy_res[i].chain + to_string(Energy_res[i].resnr);
    out << " " << setw(14) << (res_prefix + "_Tot") << " " << setw(14)
        << (res_prefix + "_Elec") << " " << setw(14) << (res_prefix + "_VdW");
  }
  out << '\n';

  size_t num_snaps{};
  for (size_t i{}; i < Energy_res.size(); ++i) {
    if (Energy_res[i].base.size() > num_snaps)
      num_snaps = Energy_res[i].base.size();
  }

  for (size_t j{}; j < num_snaps; ++j) {
    double frame_time = 0.0;
    for (size_t i{}; i < Energy_res.size(); ++i) {
      if (j < Energy_res[i].base.size()) {
        frame_time = Energy_res[i].base[j].time;
        break;
      }
    }
    out << left << setw(12) << frame_time;
    for (size_t i{}; i < Energy_res.size(); ++i) {
      if (j < Energy_res[i].base.size()) {
        double tot =
            Arguments.fu_par * sqrt(square(Energy_res[i].base[j].F_elec.Fx +
                                           Energy_res[i].base[j].F_vdw.Fx) +
                                    square(Energy_res[i].base[j].F_elec.Fy +
                                           Energy_res[i].base[j].F_vdw.Fy) +
                                    square(Energy_res[i].base[j].F_elec.Fz +
                                           Energy_res[i].base[j].F_vdw.Fz));
        double elec =
            Arguments.fu_par * sqrt(square(Energy_res[i].base[j].F_elec.Fx) +
                                    square(Energy_res[i].base[j].F_elec.Fy) +
                                    square(Energy_res[i].base[j].F_elec.Fz));
        double vdw =
            Arguments.fu_par * sqrt(square(Energy_res[i].base[j].F_vdw.Fx) +
                                    square(Energy_res[i].base[j].F_vdw.Fy) +
                                    square(Energy_res[i].base[j].F_vdw.Fz));
        out << " " << setw(14) << setprecision(6) << tot << " " << setw(14)
            << setprecision(6) << elec << " " << setw(14) << setprecision(6)
            << vdw;
      } else {
        out << " " << setw(14) << "0.0" << " " << setw(14) << "0.0" << " "
            << setw(14) << "0.0";
      }
    }
    out << '\n';
  }
  out.close();
  return 0;
}

int impl_print_fmap(
    fstream &force_file, const arguments &Arguments,
    vector<data_out> &data_processed, vector<energy_res> &Energy_res,
    size_t num_snap, size_t j, info &info_output,
    unordered_map<pair<string, size_t>, size_t, pair_hash> &map_energy_res) {
  if (Arguments.print_force_res == true) {
    add_energy_res(map_energy_res, Energy_res, data_processed[j].resnr,
                   data_processed[j].chain, data_processed[j].resname,
                   data_processed[j].E_E, data_processed[j].E_V,
                   data_processed[j].F_elec, data_processed[j].F_vdw,
                   1. * num_snap * info_output.get_current_dt() +
                       info_output.get_current_begin_time());
  }
  force_file << data_processed[j].chain << left << setw(3)
             << data_processed[j].resnr << right << setw(12) << setprecision(5)
             << Arguments.fu_par * sqrt(square(data_processed[j].F_elec.Fx) +
                                        square(data_processed[j].F_elec.Fy) +
                                        square(data_processed[j].F_elec.Fz))
             << setw(15) << setprecision(5)
             << Arguments.fu_par * sqrt(square(data_processed[j].F_vdw.Fx) +
                                        square(data_processed[j].F_vdw.Fy) +
                                        square(data_processed[j].F_vdw.Fz))
             << '\n';
  return 0;
}

void print_force_map(size_t nt, vector<vector<data_out>> &data,
                     fstream &force_file, bool flag_all_mode,
                     const arguments &Arguments, vector<energy_res> &Energy_res,
                     size_t num_snap, info &info_output) {
  vector<data_out> data_processed(nt);
  filt_duplicate_e(data, data_processed);

  unordered_map<pair<string, size_t>, size_t, pair_hash> map_energy_res;
  for (size_t i{}; i < Energy_res.size(); ++i)
    map_energy_res.emplace(
        pair<string, size_t>(Energy_res[i].chain, Energy_res[i].resnr), i);

  sort(data_processed.begin(), data_processed.end(), sort_my_list_e);
  for (size_t j{}; j < data_processed.size(); ++j) {
    impl_print_fmap(force_file, Arguments, data_processed, Energy_res, num_snap,
                    j, info_output, map_energy_res);
  }
  force_file.close();
}

std::istream &safeGetline(std::istream &is, std::string &t) {
  t.clear();
  std::istream::sentry se(is, true);
  std::streambuf *sb = is.rdbuf();

  for (;;) {
    int c{sb->sbumpc()};
    switch (c) {
    case '\n':
      return is;
    case '\r':
      if (sb->sgetc() == '\n')
        sb->sbumpc();
      return is;
    case EOF:
      // Also handle the case when the last line has no line ending
      if (t.empty())
        is.setstate(std::ios::eofbit);
      return is;
    default:
      t += (char)c;
    }
  }
}

bool check_end_cal(arguments &Arguments, vector<string> &input_line,
                   int &input_name) {
  if (!input_line.empty()) {
    if (input_line.size() > (input_name + 1 - Arguments.b_input)) {
      ++input_name;
    } else {
      return true;
    }
  } else {
    if (Arguments.flag_listfile == false &&
        Arguments.flag_pattern_input == false) {
      return true;
    }
    //<< std::endl;
    ++input_name;
  }
  return false;
}

int find_maxforce_res(vector<energy_res> &Energy_res, string name,
                      folders &paths, int file_num, const arguments &Arguments,
                      vector<energy_res> &Energy_res_out) {
  fstream out(paths.get_path(file_num) + "/" + name + "_out_max_force_res" +
                  ".dat",
              ios::out);
  out << left << setw(10) << "#res" << "Total (" << Arguments.fu << ")   Elec ("
      << Arguments.fu << ")   Van der Waals (" << Arguments.fu << ") Time ("
      << Arguments.tu << ")\n";
  for (size_t i{}; i < Energy_res.size(); ++i) {
    double maxforce{-10e9};
    size_t max_pos{};
    for (size_t j{}; j < Energy_res[i].base.size(); ++j) {
      double total_force{Arguments.fu_par *
                         sqrt(square(Energy_res[i].base[j].F_elec.Fx +
                                     Energy_res[i].base[j].F_vdw.Fx) +
                              square(Energy_res[i].base[j].F_elec.Fy +
                                     Energy_res[i].base[j].F_vdw.Fy) +
                              square(Energy_res[i].base[j].F_elec.Fz +
                                     Energy_res[i].base[j].F_vdw.Fz))};
      if (total_force > maxforce) {
        maxforce = total_force;
        max_pos = j;
      }
    }
    out << left << setw(6)
        << Energy_res[i].chain + to_string(Energy_res[i].resnr) << right
        << setw(12) << setprecision(6) << maxforce << setw(12)
        << setprecision(6)
        << Arguments.fu_par *
               sqrt(square(Energy_res[i].base[max_pos].F_elec.Fx) +
                    square(Energy_res[i].base[max_pos].F_elec.Fy) +
                    square(Energy_res[i].base[max_pos].F_elec.Fz))
        << setw(12) << setprecision(6)
        << Arguments.fu_par *
               sqrt(square(Energy_res[i].base[max_pos].F_vdw.Fx) +
                    square(Energy_res[i].base[max_pos].F_vdw.Fy) +
                    square(Energy_res[i].base[max_pos].F_vdw.Fz))
        << setw(10) << Energy_res[i].base[max_pos].time << '\n';

    if (i >= Energy_res_out.size())
      Energy_res_out.emplace_back(Energy_res[i].resnr, Energy_res[i].chain,
                                  Energy_res[i].base[max_pos]);
    else
      Energy_res_out[i].emplace(Energy_res[i].base[max_pos]);
  }
  return 0;
}

int stat_maxforce_res(vector<energy_res> &Force_res_stat, string name,
                      const info &infos, arguments &Arguments) {
  fstream out(name + infos.get_stat_maxforce_name(), ios::out);
  string error{Arguments.flag_use_CI == true ? "CI (95%)" : "SD"};
  out << left << setw(10) << "#res" << "Total (" << Arguments.fu << ") "
      << error << " (" << Arguments.fu << ") \n";
  for (size_t i{}; i < Force_res_stat.size(); ++i) {
    if (Force_res_stat[i].base.empty())
      continue;
    double sd_force{}, ave_total_force{};
    for (size_t j{}; j < Force_res_stat[i].base.size(); ++j) {
      ave_total_force +=
          Arguments.fu_par * sqrt(square(Force_res_stat[i].base[j].F_elec.Fx +
                                         Force_res_stat[i].base[j].F_vdw.Fx) +
                                  square(Force_res_stat[i].base[j].F_elec.Fy +
                                         Force_res_stat[i].base[j].F_vdw.Fy) +
                                  square(Force_res_stat[i].base[j].F_elec.Fz +
                                         Force_res_stat[i].base[j].F_vdw.Fz));
    }
    ave_total_force /= Force_res_stat[i].base.size();
    for (size_t j{}; j < Force_res_stat[i].base.size(); ++j) {
      sd_force += square(Arguments.fu_par *
                             sqrt(square(Force_res_stat[i].base[j].F_elec.Fx +
                                         Force_res_stat[i].base[j].F_vdw.Fx) +
                                  square(Force_res_stat[i].base[j].F_elec.Fy +
                                         Force_res_stat[i].base[j].F_vdw.Fy) +
                                  square(Force_res_stat[i].base[j].F_elec.Fz +
                                         Force_res_stat[i].base[j].F_vdw.Fz)) -
                         ave_total_force);
    }
    sd_force = sqrt(sd_force / Force_res_stat[i].base.size());
    if (Arguments.flag_use_CI == true)
      sd_force = 1.96 * sd_force / sqrt(Force_res_stat[i].base.size());
    out << left << setw(6)
        << Force_res_stat[i].chain + to_string(Force_res_stat[i].resnr) << right
        << setw(12) << setprecision(6) << ave_total_force << setw(12)
        << setprecision(6) << sd_force << '\n';
  }
  return 0;
}

bool remove_double_counting(cache_sigma_epsilon &CachedData,
                            vector<cache_energy> &Energy) {
  if (!CachedData.elec_ener.empty()) {
    for (size_t i{}; i < CachedData.elec_ener.size(); ++i) {
      CachedData.elec_ener[i] /= 2.;
      CachedData.vdw_ener[i] /= 2.;
    }
  }
  for (size_t i{}; i < Energy.size(); ++i) {
    Energy[i].elec_ener /= 2.;
    Energy[i].vdw_ener /= 2.;
  }
  return true;
}

string add_input_file_type(arguments &Arguments) {
  if (Arguments.pdb_name.length() <= 4 ||
      Arguments.pdb_name.substr(Arguments.pdb_name.length() - 4) !=
          Arguments.get_input_file_type())
    return Arguments.pdb_name + Arguments.get_input_file_type();
  else
    return Arguments.pdb_name;
}

// Time metadata of a newly opened input. The per-row times (time_of_snap)
// are not set here: when an input is opened pro_in holds only its reference
// frame (the -s structure, or the first model), not the analysed frames, so
// the engine records one time per row as the frames are analysed.
int process_time_of_infiles(Multi_MOL &pro_in, arguments &Arguments,
                            info &info_output) {
  if (pro_in.m_mol.size() >= 2 && Arguments.dt == 0 &&
      pro_in.m_mol[1].time_of_snap - pro_in.m_mol[0].time_of_snap > 0) {
    info_output.set_current_dt(pro_in.m_mol[1].time_of_snap -
                               pro_in.m_mol[0].time_of_snap);
  }

  info_output.set_current_begin_time(pro_in.m_mol[0].time_of_snap);

  if (Arguments.flag_end_set == true)
    info_output.set_current_end_time(
        pro_in.m_mol[pro_in.m_mol.size() - 1].get_time());
  return 0;
}

int process_adding_energies_forces(
    arguments &Arguments, int input_name, vector<double> &E_v,
    vector<double> &E_e, double &E_E_, double &E_V_, force &f_e_, force &f_v_,
    vector<vector<double>> &E_v_vector, vector<vector<double>> &E_e_vector,
    vector<force> &Force_e, vector<force> &Force_v,
    vector<vector<force>> &Force_e_vector,
    vector<vector<force>> &Force_v_vector, vector<size_t> &size_snap,
    size_t num_snap) {
  if ((Arguments.flag_pattern_input == false &&
       Arguments.flag_listfile == false &&
       Arguments.flag_all_folders == false) ||
      input_name == Arguments.b_input ||
      Arguments.flag_cat_result_multi_infile == true) {
    E_v.emplace_back(E_V_);
    E_e.emplace_back(E_E_);
    E_e_vector.emplace_back(vector<double>{E_E_});
    E_v_vector.emplace_back(vector<double>{E_V_});
    Force_e.emplace_back(f_e_);
    Force_v.emplace_back(f_v_);
    Force_e_vector.emplace_back(vector<force>{f_e_});
    Force_v_vector.emplace_back(vector<force>{f_v_});
    size_snap.emplace_back(1);
  } else {
    if (num_snap < E_v.size()) {
      if (E_e_vector[num_snap].size() < input_name - Arguments.b_input + 1) {
        size_t num{static_cast<size_t>(input_name - Arguments.b_input) -
                   E_e_vector[num_snap].size()};
        for (size_t nu{}; nu < num; ++nu) {
          E_e_vector[num_snap].emplace_back(10e-11);
          E_v_vector[num_snap].emplace_back(10e-11);
          Force_e_vector[num_snap].emplace_back(force());
          Force_v_vector[num_snap].emplace_back(force());
        }
      }
      E_v[num_snap] += E_V_;
      E_e[num_snap] += E_E_;
      E_e_vector[num_snap].emplace_back(E_E_);
      E_v_vector[num_snap].emplace_back(E_V_);
      Force_e[num_snap] += f_e_;
      Force_v[num_snap] += f_v_;
      Force_e_vector[num_snap].emplace_back(f_e_);
      Force_v_vector[num_snap].emplace_back(f_v_);
      ++size_snap[num_snap];
    } else {
      if (!abnormal_snap_warning_emitted()) {
        abnormal_snap_warning_emitted() = true;
        TitanUI::print_warning("Number of snaps of " +
                               snap_count_source_label(Arguments) +
                               " is abnormal! Further repeats suppressed.");
      }
      E_v.emplace_back(E_V_);
      E_e.emplace_back(E_E_);
      Force_e.emplace_back(f_e_);
      Force_v.emplace_back(f_v_);

      vector<double> ee_v_temp(input_name - Arguments.b_input, 10e-12),
          ev_v_temp(input_name - Arguments.b_input, 10e-13);
      vector<force> fe_v_temp(input_name - Arguments.b_input, force()),
          fv_v_temp(input_name - Arguments.b_input, force());
      ee_v_temp.emplace_back(E_E_);
      ev_v_temp.emplace_back(E_V_);
      fe_v_temp.emplace_back(f_e_);
      fv_v_temp.emplace_back(f_v_);

      E_e_vector.emplace_back(ee_v_temp);
      E_v_vector.emplace_back(ev_v_temp);
      Force_e_vector.emplace_back(fe_v_temp);
      Force_v_vector.emplace_back(fv_v_temp);
      size_snap.emplace_back(1);
    }
  }
  return 0;
}

