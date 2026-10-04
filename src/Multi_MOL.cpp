#include "Multi_MOL.h"
#include "info.h"
#include "titan_executor.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string_view>
using std::string_view;
#include <thread>

int sup_cal_dis_res_com(vector<multi_points> &com_res,
                        vector<double> &distances, vector<string> &name_res,
                        bool flag_o);

bool Multi_MOL::frame_coordinates_consistent(size_t frame_index) const
    noexcept {
  return frame_index < m_mol.size() &&
         m_mol[frame_index].coordinates_consistent();
}

bool Multi_MOL::topology_matches_frame(size_t frame_index) const noexcept {
  return frame_index < m_mol.size() &&
         m_mol[frame_index].topology_consistent(info.size());
}

bool Multi_MOL::parameters_match_info() const noexcept {
  return parameters.size() == info.size();
}

bool Multi_MOL::parameters_match_frame(size_t frame_index) const noexcept {
  return frame_index < m_mol.size() && parameters.size() >= m_mol[frame_index].size();
}

bool Multi_MOL::analysis_frame_ready(size_t frame_index) const noexcept {
  return topology_matches_frame(frame_index) && parameters_match_frame(frame_index);
}

void Multi_MOL::post_process_coordinates_user_define(string_view type) {
  mol_type = type;
  generate_index();
}

string Multi_MOL::check_head_tail_res(class index &in, size_t pos_res) {
  if (info.empty() || pos_res >= in.get_size_v())
    return "normal";
  const size_t residue_start = in.get_res_pos_e(pos_res);
  if (residue_start == 0 || residue_start > info.size())
    return "normal";
  const size_t first_atom = residue_start - 1;
  if (in.get_type_base(pos_res) == "protein") {
    if (first_atom + 3 < info.size() &&
        info[first_atom].get_atom().find("N") !=
            string::npos &&
        info[first_atom + 1].get_atom().find("H") != string::npos &&
        info[first_atom + 2].get_atom().find("H") != string::npos &&
        info[first_atom + 3].get_atom().find("H") != string::npos)
      return "nterminal";
    size_t remain{in.get_res_pos_e(pos_res) + 160 >= info.size()
                      ? info.size()
                      : in.get_res_pos_e(pos_res) + 160},
        pos{};
    for (size_t i = in.get_res_pos_e(pos_res) - 1; i < remain; ++i) {
      if (info[in.get_res_pos_e(pos_res) - 1].get_resnr() !=
          info[i].get_resnr()) {
        pos = i;
        break;
      }
    }
    if (pos == 0)
      pos = info.size();
    if (pos >= 3 &&
        info[pos - 3].get_atom().find("C") != string::npos &&
        info[pos - 1].get_atom().find("O") != string::npos &&
        info[pos - 2].get_atom().find("O") != string::npos)
      return "cterminal";
    return "normal";
  }
  if (in.get_type_base(pos_res) == "dna" ||
      in.get_type_base(pos_res) == "rna") {
    int flag_poo{}, flag_h3t{}, flag_h5t{};
    size_t remain{in.get_res_pos_e(pos_res) + 50 >= info.size()
                      ? info.size()
                      : in.get_res_pos_e(pos_res) + 50};
    for (size_t i = in.get_res_pos_e(pos_res) - 1;
         (i < remain) && (info[in.get_res_pos_e(pos_res) - 1].get_resnr() ==
                          info[i].get_resnr());
         ++i) {
      if (info[i].get_atom() == "P" || info[i].get_atom() == "O1P" ||
          info[i].get_atom() == "O2P")
        ++flag_poo;
      if (info[i].get_atom() == "H3T")
        ++flag_h3t;
      if (info[i].get_atom() == "H5T")
        ++flag_h5t;
    }
    if (flag_h3t == 0 && 1 == flag_h5t)
      return "5terminal";
    if (1 == flag_h3t && 1 == flag_h5t)
      return "nsingle";
    if (3 == flag_poo && 0 == flag_h3t)
      return "normal";
    if (3 == flag_poo && 1 == flag_h3t)
      return "3terminal";
  }
  return "ligand";
}
int Multi_MOL::emplace(size_t i, box &box_type, double time,
                       string_view force_f, mol_info &&info_in, MOL &&mol_in) {
  this->force_f = force_f;
  if (i < m_mol.size()) {
    m_mol[i].pos.x.emplace_back(mol_in.pos.x[0]);
    m_mol[i].pos.y.emplace_back(mol_in.pos.y[0]);
    m_mol[i].pos.z.emplace_back(mol_in.pos.z[0]);
  } else {
    m_mol.emplace_back(mol_in);
    m_mol.back().Box = box_type;
    m_mol.back().set_time(time);
    m_mol.back().force_f = force_f;
  }

  if (m_mol.size() <= 1)
    info.emplace_back(std::move(info_in));
  return 0;
}

void Multi_MOL::begin_frame(size_t frame_index, const box &box_type, double time,
                            string_view force_field, size_t reserve_atoms,
                            string_view frame_mol_type) {
  this->force_f = force_field;
  if (frame_index >= m_mol.size())
    m_mol.resize(frame_index + 1);

  MOL &frame = m_mol[frame_index];
  frame.pos.clear();
  if (frame.force_vec.has_value())
    frame.force_vec.reset();
  frame.thread_local_forces.clear();
  frame.Com.clear();
  frame.list_h_bonds.dat.clear();
  frame.Box = box_type;
  frame.set_time(time);
  frame.force_f = std::string(force_field);
  frame.flag_set_time = true;
  frame.compute_forces = false;
  if (!frame_mol_type.empty())
    frame.mol_type = std::string(frame_mol_type);
  else if (!this->mol_type.empty())
    frame.mol_type = this->mol_type;

  this->reserve_atoms(frame_index, reserve_atoms);
}

void Multi_MOL::reserve_atoms(size_t frame_index, size_t atom_count) {
  if (frame_index >= m_mol.size())
    m_mol.resize(frame_index + 1);

  if (atom_count == 0)
    return;

  MOL &frame = m_mol[frame_index];
  frame.pos.x.reserve(atom_count);
  frame.pos.y.reserve(atom_count);
  frame.pos.z.reserve(atom_count);
}

void Multi_MOL::append_xyz(size_t frame_index, double x, double y, double z) {
  if (frame_index >= m_mol.size())
    m_mol.resize(frame_index + 1);

  MOL &frame = m_mol[frame_index];
  frame.pos.x.emplace_back(x);
  frame.pos.y.emplace_back(y);
  frame.pos.z.emplace_back(z);
}

void Multi_MOL::append_topology_if_frame0(const mol_info &info_in) {
  if (m_mol.empty())
    return;

  const size_t next_atom_index = m_mol[0].pos.x.size() + 1;
  if (info.size() < next_atom_index)
    info.emplace_back(info_in);
}

void Multi_MOL::discard_pending_frame(size_t frame_index) {
  if (frame_index < m_mol.size())
    m_mol.resize(frame_index);
}

int Multi_MOL::emplace(size_t i, box &box_type, double time,
                       string_view force_f, const mol_info &info_in,
                       const MOL &mol_in, string_view _mol_type) {
  this->force_f = force_f;
  if (i < m_mol.size()) {
    m_mol[i].pos.x.emplace_back(mol_in.pos.x[0]);
    m_mol[i].pos.y.emplace_back(mol_in.pos.y[0]);
    m_mol[i].pos.z.emplace_back(mol_in.pos.z[0]);
  } else {
    m_mol.emplace_back(mol_in);
    m_mol.back().Box = box_type;
    m_mol.back().set_time(time);
    m_mol.back().force_f = force_f;
    m_mol.back().mol_type = _mol_type;
  }

  if (m_mol.size() <= 1)
    info.emplace_back(info_in);
  return 0;
}

int Multi_MOL::emplace(size_t i, box &box_type, double time,
                       string_view force_f, const mol_info &info_in,
                       MOL &&mol_in, string_view _mol_type) {
  this->force_f = force_f;
  if (i < m_mol.size()) {
    m_mol[i].pos.x.emplace_back(mol_in.pos.x[0]);
    m_mol[i].pos.y.emplace_back(mol_in.pos.y[0]);
    m_mol[i].pos.z.emplace_back(mol_in.pos.z[0]);
  } else {
    m_mol.emplace_back(mol_in);
    m_mol.back().Box = box_type;
    m_mol.back().set_time(time);
    m_mol.back().force_f = force_f;
    m_mol.back().mol_type = _mol_type;
  }

  if (m_mol.size() <= 1)
    info.emplace_back(info_in);
  return 0;
}

void Multi_MOL::pbc_prob(Multi_MOL &lig, size_t i, size_t num_threads) {
  if (num_threads == 0)
    num_threads = 1;
  for (size_t j{i * m_mol.size() / num_threads};
       j < (i + 1) * m_mol.size() / num_threads; ++j)
    solve_pbc(m_mol[j], lig.m_mol[j], m_mol[j].Box);
}

void Multi_MOL::imp_post_proc(int begin, int end, string_view type,
                              const amino_acid &amine) {
  if (type == "AMBER") {
    post_process_coordinates(amine);
  } else if (type == "GROMOS") {
    post_process_coordinates_gromos(amine);
  } else if (type == "lig") {
    post_process_coordinates_lig();
  } else if (type == "CHARMM") {
    post_process_coordinates_charmm(amine);
  } else if (type == "protein-user" || type == "complex") {
    post_process_coordinates_user_define(type);
  } else // (type == "OPLSAA")
  {
    post_process_coordinates_oplsaa(amine);
  }
}
int Multi_MOL::wrapper_impl_post_proc(arguments &Arguments, string_view type,
                                      const amino_acid &amine) {
  if (m_mol.empty())
    return 1;
  this->imp_post_proc(0, 1, type, amine);

  if (Arguments.flag_sidechain == true)
    this->check_side_chain(Arguments);
  return 0;
}
int Multi_MOL::add_coms(bool flag, size_t snap_index) {
  if (snap_index >= m_mol.size() || Index.empty())
    return 1;
  if (flag == false) {
    const size_t chain_count = Index.size() > 1 ? Index.size() - 1 : 1;
    for (size_t i{}; i < chain_count; ++i)
      for (size_t j{}; j < Index[i].get_size_v(); ++j) {
        m_mol[snap_index].Com.emplace_back(
            Index[i].get_res_x_e(j), Index[i].get_res_y_e(j),
            Index[i].get_res_z_e(j), Index[i].get_chain(),
            Index[i].get_res_name_e(j), Index[i].get_res_num_e(j),
            Index[i].get_res_pos_e(j), Index[i].get_chainid(),
            Index[i].get_res_pos_e(j));
      }
    return 0;
  } else {
    for (size_t j{}; j < Index[0].get_size_v(); ++j) {
      m_mol[snap_index].Com.emplace_back(
          Index[0].get_res_x_e(j), Index[0].get_res_y_e(j),
          Index[0].get_res_z_e(j), Index[0].get_chain(),
          Index[0].get_res_name_e(j), Index[0].get_res_num_e(j),
          Index[0].get_res_pos_e(j), Index[0].get_chainid(),
          Index[0].get_res_pos_e(j));
    }
    return 0;
  }
}
int Multi_MOL::gen_vector_parameters() {
  if (m_mol.empty() || !parameters.empty())
    return 1;
  else {
    this->parameters.resize(info.size());
    return 0;
  }
}

void Multi_MOL::cache_topology_vdw_radii_from_parameters() {
  const std::span<const std::int32_t> type_ids =
      parameters.has_type_ids()
          ? std::span<const std::int32_t>(parameters.type_id.data(),
                                          parameters.type_id.size())
          : std::span<const std::int32_t>{};
  topology.cache_missing_vdw_radii(
      std::span<const double>(parameters.radius.data(),
                              parameters.radius.size()),
      std::span<const double>(parameters.sigma.data(), parameters.sigma.size()),
      std::span<const double>(parameters.epsilon.data(),
                              parameters.epsilon.size()),
      type_ids);
}

int Multi_MOL::com_mol(const size_t num_snap, std::span<double> com) const {
  if (com.size() < 3 || !analysis_frame_ready(num_snap))
    return 1;

  double mol_mass{};
  double x{};
  double y{};
  double z{};
  double unit_x{};
  double unit_y{};
  double unit_z{};
  const size_t atom_count = m_mol[num_snap].size();
  if (atom_count == 0)
    return 1;

  for (size_t i{}; i < atom_count; ++i) {
    unit_x += m_mol[num_snap].pos.x[i];
    unit_y += m_mol[num_snap].pos.y[i];
    unit_z += m_mol[num_snap].pos.z[i];
    const double mass = parameters.mass[i];
    if (mass > 0.0) {
      x += m_mol[num_snap].pos.x[i] * mass;
      y += m_mol[num_snap].pos.y[i] * mass;
      z += m_mol[num_snap].pos.z[i] * mass;
      mol_mass += mass;
    }
  }

  if (mol_mass > 0.0) {
    com[0] = x / mol_mass;
    com[1] = y / mol_mass;
    com[2] = z / mol_mass;
  } else {
    const double denom = static_cast<double>(atom_count);
    com[0] = unit_x / denom;
    com[1] = unit_y / denom;
    com[2] = unit_z / denom;
  }
  return 0;
}

int Multi_MOL::check_side_chain(const arguments &Arguments) {
  if (Arguments.nt < 2 || Index.size() < 5)
    this->helper_check_side_chain(Arguments, 0, Index.size() - 1);
  else {
    vector<thread> threads(Arguments.nt);
    for (size_t i{}; i < Arguments.nt; ++i)
      threads[i] = thread(&Multi_MOL::helper_check_side_chain, &(*this),
                          ref(Arguments), i * (Index.size() - 1) / Arguments.nt,
                          (i + 1) * (Index.size() - 1) / Arguments.nt);
    for (auto &t : threads)
      t.join();
  }
  return 0;
}

int Multi_MOL::helper_check_side_chain(const arguments &Arguments, size_t begin,
                                       size_t end) {
  for (size_t i{begin}; i < end; ++i)
    for (size_t j{}; j < Index[i].get_size_v(); ++j)
      if (Arguments.flag_sidechain == true &&
          Index[i].get_type_base(j) == "protein") {
        size_t end_pos{j + 1 >= Index[i].get_size_v()
                           ? Index[i + 1].get_res_pos_e(0) - 1
                           : Index[i].get_res_pos_e(j + 1) - 1};
        for (size_t k{Index[i].get_res_pos_e(j) - 1}; k < end_pos; ++k)
          info[k].check_side_chain();
      }
  return 0;
}

int Multi_MOL::set_chainnum() {
  if (this->Index.empty())
    return 0;

  if (!this->info.empty()) {
    this->info[0].set_internal_resnr(0);
  }
  size_t internal_cnt{};
  for (size_t k = 1; k < this->info.size(); ++k) {
    if (this->info[k].get_resnr() != this->info[k - 1].get_resnr() ||
        this->info[k].get_res_name() != this->info[k - 1].get_res_name() ||
        this->info[k].get_chain() != this->info[k - 1].get_chain())
      internal_cnt++;
    this->info[k].set_internal_resnr(internal_cnt);
  }

  size_t chainid{}, anum{};
  for (size_t i{}; i < this->Index.size() - 1; ++i) {
    for (size_t j{}; j < this->Index[i].get_size_v(); ++j) {
      size_t max_pos{j != this->Index[i].get_size_v() - 1
                         ? this->Index[i].get_res_pos_e(j + 1) - 1
                         : this->Index[i + 1].get_res_pos_e(0) - 1};
      for (size_t k = this->Index[i].get_res_pos_e(j) - 1; k < max_pos; ++k) {
        if (anum >= this->info.size()) {
          cout << "overflow info! atom num: " << anum
               << "; info size: " << this->info.size() << "\n";
          break;
        }
        this->info[anum].set_chainnum(chainid);
        this->info[anum].set_respos(this->Index[i].get_res_pos_e(j));
        this->Index[i].set_chainid(chainid);
        ++anum;
      }
    }
    ++chainid;
  }
  // cout << "anum: " << anum << " info size: " << info.size() << '\n';
  return 0;
}

int Multi_MOL::com_dis_2mol(Multi_MOL &rhs, arguments &Arguments,
                            const ::info &info_output, fstream &output_dis) {
  cout << "calculating COM distance......";
  for (size_t i{}; i < m_mol.size(); ++i) {
    if (i % 200 == 0)
      cout << i << " ";
    if (impl_com_dis_2mol(rhs, Arguments, info_output, output_dis, i) == 1)
      break;
  }
  return 0;
}

int Multi_MOL::impl_com_dis_2mol(Multi_MOL &rhs, const arguments &Arguments,
                                 const ::info &info_output, fstream &output_dis,
                                 size_t i) {
  double com_mol_1[3]{}, com_mol_2[3]{};

  com_mol(i, com_mol_1);

  if (i >= rhs.m_mol.size())
    return 1;
  else
    rhs.com_mol(i, com_mol_2);

  double dis{sqrt(square(com_mol_1[0] - com_mol_2[0]) +
                  square(com_mol_1[1] - com_mol_2[1]) +
                  square(com_mol_1[2] - com_mol_2[2]))};
  if (Arguments.com_dis_2mol == 1)
    dis = fabs(com_mol_1[0] - com_mol_2[0]);
  else if (Arguments.com_dis_2mol == 2)
    dis = fabs(com_mol_1[1] - com_mol_2[1]);
  else if (Arguments.com_dis_2mol == 3)
    dis = fabs(com_mol_1[2] - com_mol_2[2]);
  else {
  }

  output_dis << setw(10) << left
             << i * info_output.get_current_dt() +
                    info_output.get_current_begin_time()
             << setw(12) << right << dis;

  double min_dist[3]{10e10, 10e10, 10e10};

  for (size_t j{}; j < rhs.m_mol[i].pos.x.size(); ++j) {
    if (min_dist[0] > fabs(rhs.m_mol[i].pos.x[j] - com_mol_1[0]))
      min_dist[0] = fabs(rhs.m_mol[i].pos.x[j] - com_mol_1[0]);
    if (min_dist[1] > fabs(rhs.m_mol[i].pos.y[j] - com_mol_1[1]))
      min_dist[1] = fabs(rhs.m_mol[i].pos.y[j] - com_mol_1[1]);
    if (min_dist[2] > fabs(rhs.m_mol[i].pos.z[j] - com_mol_1[2]))
      min_dist[2] = fabs(rhs.m_mol[i].pos.z[j] - com_mol_1[2]);
  }

  output_dis << setw(12) << min_dist[0] << setw(12) << min_dist[1] << setw(12)
             << min_dist[2] << '\n';
  return 0;
}

int Multi_MOL::find_water(vector<size_t> &list_atoms_water,
                          vector<size_t> &list_heavy_atoms) {
  for (size_t i{}; i < Index.size() - 1; ++i) {
    for (size_t j{}; j < Index[i].get_size_v(); ++j) {
      size_t remain_index{j + 1 == Index[i].get_size_v()
                              ? Index[i + 1].get_res_pos_e(0) - 1
                              : Index[i].get_res_pos_e(j + 1) - 1};
      if (Index[i].get_type_base(j) == "water") {
        for (size_t k{Index[i].get_res_pos_e(j) - 1}; k < remain_index; ++k)
          if ((this->force_f.find("OPLSAA") != string::npos &&
               this->info[k].get_atom_type() != "opls_114") ||
              this->info[k].get_atom_type()[0] != 'H')
            list_atoms_water.emplace_back(k);
      }

      if (Index[i].get_type_base(j) != "water" &&
          Index[i].get_type_base(j) != "ion") {
        for (size_t k{Index[i].get_res_pos_e(j) - 1}; k < remain_index; ++k)
          if ((this->force_f.find("OPLSAA") != string::npos &&
               this->info[k].get_atom_type() != "opls_114") ||
              this->info[k].get_atom_type()[0] != 'H')
            list_heavy_atoms.emplace_back(k);
      }
    }
  }

  if (list_atoms_water.empty()) {
    cout << "WARNING!! No water!\n";
  }

  return 0;
}

int Multi_MOL::cal_angle_dipole_water_elecField(const arguments &Arguments,
                                                ::info &output_info) {
  vector<size_t> list_waters, list_heavy_atoms;

  this->find_water(list_waters, list_heavy_atoms);
  vector<double> cos_angle_dipoles_efield;

  for (size_t i{}; i < m_mol.size(); ++i) {
    vector<size_t> list_surface_waters;

    if (Arguments.water_type == "surface")
      m_mol[i].find_surface_water(list_waters, list_heavy_atoms,
                                  list_surface_waters);
    else
      m_mol[i].find_sphere_water(list_waters, list_heavy_atoms,
                                 list_surface_waters, Arguments.d_crit_water);

    vector<force> elec_fields(list_surface_waters.size());
    // cout <<"water: " << list_waters.size()<< " atoms. water sphere: " <<
    // list_surface_waters.size() << " atoms. Heavy: " <<
    // list_heavy_atoms.size() << " atoms\n"; cout <<"crit dis: "<<
    // Arguments.d_crit_water << '\n';

    m_mol[i].cal_electric_field(info, Arguments, parameters, list_heavy_atoms,
                                list_surface_waters, elec_fields);

    m_mol[i].cal_angle_dipoles_elecfield(
        info, Arguments, elec_fields, parameters, list_heavy_atoms,
        list_surface_waters, cos_angle_dipoles_efield);
  }

  double max{-1000.}, min{1000.};
  for (size_t i{}; i < cos_angle_dipoles_efield.size(); ++i) {
    if (cos_angle_dipoles_efield[i] > max)
      max = cos_angle_dipoles_efield[i];
    if (cos_angle_dipoles_efield[i] < min)
      min = cos_angle_dipoles_efield[i];
  }
  double delta{(max - min) / Arguments.number_bin_angle_dipole_efield};
  vector<size_t> histo(Arguments.number_bin_angle_dipole_efield);

  for (size_t j{}; j < cos_angle_dipoles_efield.size(); ++j) {
    size_t pos{
        static_cast<size_t>((cos_angle_dipoles_efield[j] - min) / delta)};
    if (pos >= histo.size())
      pos = histo.size() - 1;
    ++histo[pos];
  }

  fstream out_angle_dipole_efield(
      Arguments.pdb_name.substr(0, Arguments.pdb_name.length() - 4) +
          output_info.getDistributionAngleDipoleEfield(),
      ios::out);
  out_angle_dipole_efield << "#Cosine        Population (%)\n";
  for (size_t i{}; i < histo.size(); ++i)
    out_angle_dipole_efield
        << setw(15) << left << i * delta + min << " " << right << setw(15)
        << histo[i] * 100. / cos_angle_dipoles_efield.size() << '\n';
  return 0;
}

bool Multi_MOL::check_and_pop_size_2last_snaps(size_t &I, string_view name) {
  if (this->m_mol.size() >= 2 &&
      this->m_mol[this->m_mol.size() - 1].pos.x.size() !=
          this->m_mol[this->m_mol.size() - 2].pos.x.size()) {
    cout << "WARNING!! Snap number" << name << I
         << " has different number of atoms ("
         << this->m_mol[this->m_mol.size() - 1].pos.x.size()
         << " atoms) compared to snap " << I - 1 << " ("
         << this->m_mol[this->m_mol.size() - 2].pos.x.size()
         << " atoms)! Removed.\n";

    this->m_mol.pop_back();
    I = this->m_mol.size() - 1;
    return false;
  }
  return true;
}

double Multi_MOL::total_charge() const {
  double charge{};
  for (size_t i{}; i < parameters.size(); ++i)
    charge += parameters[i].get_charge();
  if (fabs(charge) < 10e-8)
    charge = 0.0000000;
  return charge;
}

int Multi_MOL::cal_dis_res_2_com(const arguments &Arguments, fstream &output) {
  if (Index.size() < 2)
    return 1;
  const size_t num_threads =
      Arguments.nt < 1 ? size_t{1} : static_cast<size_t>(Arguments.nt);
  TitanExecutor executor(num_threads);
  vector<vector<double>> distances;
  vector<string> name_res;
  for (size_t snap_num{}; snap_num < this->m_mol.size(); ++snap_num) {
    vector<double> dis_temp;
    const size_t chain_count = this->Index.size() - 1;
    vector<multi_points> pos_res(chain_count);
    executor.parallel_for(chain_count, num_threads, 1,
                          [&](size_t, size_t begin, size_t end) {
      for (size_t chain_idx = begin; chain_idx < end; ++chain_idx) {
        this->m_mol[snap_num].take_com_all_atoms(
            this->parameters, chain_idx,
            this->Index[chain_idx + 1].get_res_pos_e(0), this->Index,
            this->info, pos_res[chain_idx]);
      }
    });
    bool flag_o{snap_num > 0 ? false : true};
    sup_cal_dis_res_com(pos_res, dis_temp, name_res, flag_o);
    distances.emplace_back(dis_temp);
  }

  double delta{0.2}, min_dis{}, max_dis{100.};
  if (distances.empty())
    return 1;
  vector<pair<double, size_t>> tot_dis_res(distances[0].size()),
      sd_dis_res(distances[0].size());

  for (size_t i{}; i < distances[0].size(); ++i) {
    for (size_t j{}; j < distances.size(); ++j) {
      if (i >= tot_dis_res.size())
        continue;
      tot_dis_res[i].first += distances[j][i];
      ++tot_dis_res[i].second;
    }
    tot_dis_res[i].first /= tot_dis_res[i].second;
    for (size_t j{}; j < distances.size(); ++j) {
      if (i >= tot_dis_res.size())
        continue;
      sd_dis_res[i].first += square(distances[j][i] - tot_dis_res[i].first);
    }
    output << setw(10) << left << name_res[i] << right << setw(12)
           << tot_dis_res[i].first << setw(12)
           << sqrt(sd_dis_res[i].first / distances.size()) << '\n';
  }
  return 0;
}

int Multi_MOL::cal_volumes(vector<double> &vol_out) const {
  for (auto &m : this->m_mol)
    vol_out.emplace_back(m.Box.volume());
  return 0;
}
