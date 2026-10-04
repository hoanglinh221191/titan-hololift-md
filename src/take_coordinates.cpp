#include "arguments.h"
#include "functions.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
using std::string_view;
#include <vector>

using namespace std;

int Multi_MOL::take_coordinate(std::string_view input_name,
                               std::string_view force_f, arguments &Arguments,
                               const amino_acid &amine, ::info &output_info,
                               int flag_raw, size_t max_frames) {
  this->force_f = force_f;

  if (parse_pdb(input_name, force_f, Arguments, output_info, max_frames) == 1) {
    return 1;
  }

  if (flag_raw == 0) {
    string type{"AMBER"};
    if (force_f.find("GROMOS") != string::npos)
      type = "GROMOS";
    else if (force_f.find("CHARMM") != string::npos)
      type = "CHARMM";
    else if (force_f.find("OPLSAA") != string::npos) {
      type = "OPLSAA";
    } else {
    }

    wrapper_impl_post_proc(Arguments, type, amine);
  }

  gen_vector_parameters();

  return 0;
}

int Multi_MOL::take_coordinate_from_group(
    std::string_view input_name, std::string_view _force_f,
    const Multi_MOL &base_mol, const amino_acid &amine, ::info &output_info,
    const std::vector<size_t> &group_indices, arguments &Arguments) {
  this->force_f = _force_f;

  if (base_mol.m_mol.empty())
    return 1;

  size_t num_snaps = base_mol.m_mol.size();
  for (size_t snap = 0; snap < num_snaps; ++snap) {
    if (!base_mol.frame_coordinates_consistent(snap))
      return 1;
  }
  for (size_t idx : group_indices) {
    if (idx >= base_mol.info.size())
      return 1;
    for (size_t snap = 0; snap < num_snaps; ++snap) {
      if (idx >= base_mol.m_mol[snap].size())
        return 1;
    }
  }

  this->m_mol.resize(num_snaps);

  // Copy topology and info for the atoms in the group
  this->info.clear();
  this->info.reserve(group_indices.size());
  for (size_t idx : group_indices) {
    this->info.emplace_back(base_mol.info[idx]);
  }

  // Copy trajectory frame by frame
  for (size_t snap = 0; snap < num_snaps; ++snap) {
    this->m_mol[snap].force_f = string(_force_f);
    this->m_mol[snap].Box = base_mol.m_mol[snap].Box;
    this->m_mol[snap].set_time(base_mol.m_mol[snap].get_time());
    this->m_mol[snap].pos.x.reserve(group_indices.size());
    this->m_mol[snap].pos.y.reserve(group_indices.size());
    this->m_mol[snap].pos.z.reserve(group_indices.size());

    for (size_t idx : group_indices) {
      this->m_mol[snap].pos.x.emplace_back(base_mol.m_mol[snap].pos.x[idx]);
      this->m_mol[snap].pos.y.emplace_back(base_mol.m_mol[snap].pos.y[idx]);
      this->m_mol[snap].pos.z.emplace_back(base_mol.m_mol[snap].pos.z[idx]);
    }
  }

  // Set other properties
  string type = "water";
  this->wrapper_impl_post_proc(Arguments, type,
                               amine); // Using generic or skip
  this->gen_vector_parameters();

  return 0;
}
int Multi_MOL::post_process_coordinates(const amino_acid &amine) {
  for (size_t i{}; i < info.size(); ++i) {
    if (info[i].get_res_name() == "HIS") {
      int check1{}, check2{};
      size_t pos{};
      size_t remain_num{i + 24 > info.size() ? info.size() - i : 24};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HD2" || info[ii].get_atom() == "HD1")
            check1++;
          else if (info[ii].get_atom() == "HE2" || info[ii].get_atom() == "HE1")
            check2++;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check1 == 2 && check2 == 2)
        for (size_t ii = i; ii < pos; ii++)
          info[ii].set_res_name("HIP");
      if (check1 == 2 && check2 == 1)
        for (size_t ii = i; ii < pos; ii++)
          info[ii].set_res_name("HID");
      if (check1 == 1 && check2 == 2)
        for (size_t ii = i; ii < pos; ii++)
          info[ii].set_res_name("HIE");
      i = pos - 1;
      continue;
    }
    if (info[i].get_res_name() == "CYS") {
      int check{};
      size_t pos{};
      size_t remain_num{i + 16 > info.size() ? info.size() - i : 16};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HG")
            check++;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("CYX");
      i = pos - 1;
      continue;
    }
  }
  /*******************************************************************************************/

  this->generate_index();
  this->parse_type(amine);
  return 0;
}

void Multi_MOL::post_process_coordinates_gromos(const amino_acid &amine) {
  for (size_t i{}; i < info.size(); ++i) {
    if (info[i].get_res_name() == "ARG") {
      int check{};
      size_t pos{};
      size_t remain_num{i + 20 > info.size() ? info.size() - i : 20};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "2HH1" || info[ii].get_atom() == "HH12")
            check = 1;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check == 0)
        for (size_t ii = i; ii < pos; ++ii)
          if (info[ii].get_resnr() == info[i].get_resnr() &&
              info[ii].get_res_name() == info[i].get_res_name())
            info[ii].set_res_name("ARGN");
      i = pos - 1;
      continue;
    }
    /**********************************************************************************/
    if (info[i].get_res_name() == "HIS" && force_f == "GROMOS43A1") {
      int check1{}, check2{};
      size_t pos{};
      size_t remain_num{i + 19 > info.size() ? info.size() - i : 19};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HD2" || info[ii].get_atom() == "HD1")
            check1++;
          else if (info[ii].get_atom() == "HE2" || info[ii].get_atom() == "HE1")
            check2++;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();

      if (check1 == 1 && check2 == 1)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HISH");
      if (check1 == 1 && check2 == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HISA");
      if (check1 == 0 && check2 == 1)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HISB");
      i = pos - 1;
      continue;
    } else if (info[i].get_res_name() == "HIS" &&
               (force_f == "GROMOS53A6" || force_f == "GROMOS54A7")) {
      int check1{}, check2{};
      size_t pos{};
      size_t remain_num{i + 17 > info.size() ? info.size() - i : 17};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HD1")
            check1++;
          else if (info[ii].get_atom() == "HE2")
            check2++;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check1 == 1 && check2 == 1)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HISH");
      if (check1 == 1 && check2 == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HISA");
      if (check1 == 0 && check2 == 1)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HISB");
      i = pos - 1;
      continue;
    }
    /*******************************************************************************************************/
    if (info[i].get_res_name() == "CYS") {
      int check{};
      size_t pos{};
      size_t remain_num{i + 12 > info.size() ? info.size() - i : 12};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "H")
            check++;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check == 2) {
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("CYSH");
      }
      if (check == 1) {
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("CYS2");
      }
      i = pos - 1;
      continue;
    }
    /*************************************************/
    if (info[i].get_res_name() == "FMN") {
      int check{};
      size_t pos{};
      size_t remain_num{i + 40 > info.size() ? info.size() - i : 40};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom().find("H") != string::npos)
            ++check;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check == 6) {
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("FMNS");
      }
      if (check == 5) {
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("FMNO");
      }
      if (check == 7) {
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("FMNR");
      }
      i = pos - 1;
      continue;
    }
    size_t ind_t{check_res_H(info, "LYS", "HZ3", i)};
    i = ind_t;
    if (i >= info.size())
      break;
    ind_t = check_res_H(info, "GLU", "HE2", i);
    i = ind_t;
    if (i >= info.size())
      break;
    ind_t = check_res_H(info, "ASP", "HD2", i);
    i = ind_t;
  }
  this->generate_index();
}

void Multi_MOL::post_process_coordinates_charmm(const amino_acid &amine) {
  for (size_t i{}; i < info.size(); ++i) {
    if (info[i].get_res_name() == "BGAL") {
      size_t remain_num{i + 30 > info.size() ? info.size() - i : 30};
      size_t count{}, pos{};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr())
          ++count;
        else {
          pos = ii;
          break;
        }
      }
      if (count >= 26) {
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("BGAN");
      }
      if (count == 22) {
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("BGAM");
      }
    }

    if (info[i].get_res_name() == "HIS") {
      int check1{}, check2{};
      size_t pos{};
      size_t remain_num{i + 24 > info.size() ? info.size() - i : 24};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HD1")
            ++check1;
          if (info[ii].get_atom() == "HE2")
            ++check2;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check1 == 1 && check2 == 1)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HSP");
      if (check1 == 1 && check2 == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HSD");
      if (check2 == 1 && check1 == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HSE");

      i = pos - 1;
      continue;
    }
    if (info[i].get_res_name() == "CYS") {
      int check = 0;
      size_t pos{};
      size_t remain_num{i + 15 > info.size() ? info.size() - i : 15};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HG1")
            check++;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("CYS2");
      i = pos - 1;
      continue;
    }

    if (info[i].get_res_name() == "LYS") {
      int check = 0;
      size_t pos{};
      size_t remain_num{i + 26 > info.size() ? info.size() - i : 26};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HZ3")
            ++check;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("LSN");
      i = pos - 1;
      continue;
    }

    if (info[i].get_res_name() == "GLU") {
      int check = 0;
      size_t pos{};
      size_t remain_num{i + 17 > info.size() ? info.size() - i : 17};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HE2")
            check++;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check == 1)
        for (auto ii = i; ii < pos; ++ii)
          info[ii].set_res_name("GLUP");
      i = pos - 1;
      continue;
    }

    if (info[i].get_res_name() == "ASP") {
      int check{};
      size_t pos{};
      size_t remain_num{i + 17 > info.size() ? info.size() - i : 17};
      for (size_t ii{i}; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr()) {
          if (info[ii].get_atom() == "HD2" &&
              info[ii].get_res_name() == info[i].get_res_name())
            check++;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check == 1)
        for (auto ii = i; ii < pos; ++ii)
          info[ii].set_res_name("ASPP");
      i = pos - 1;
      continue;
    }
  }

  this->generate_index();
  this->parse_type(amine);
}

void Multi_MOL::post_process_coordinates_oplsaa(const amino_acid &amine) {
  for (size_t i = 0; i < info.size(); ++i) {
    if (info[i].get_res_name() == "HIS") {
      int check1{}, check2{};
      size_t pos{};
      size_t remain_num{i + 24 > info.size() ? info.size() - i : 24};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HD1")
            check1 = 1;
          if (info[ii].get_atom() == "HE2")
            check2 = 1;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check1 == 1 && check2 == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HISD");
      if (check2 == 1 && check1 == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HISE");
      if (check2 == 1 && check1 == 1)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("HISH");

      i = pos - 1;
      continue;
    }

    if (info[i].get_res_name() == "GLU") {
      int check1{}, check2{};
      size_t pos{};
      size_t remain_num{i + 17 > info.size() ? info.size() - i : 17};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "OE1" || info[ii].get_atom() == "OE2")
            ++check1;
          if (info[ii].get_atom() == "HE2")
            check2 = 1;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check1 == 2 && check2 == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("GLU");
      if (check1 == 2 && check2 == 1)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("GLUH");
      if (check2 == 0 && check1 == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("PGLU");

      i = pos - 1;
      continue;
    }

    if (info[i].get_res_name() == "CYS") {
      int check{};
      size_t pos{};
      size_t remain_num{i + 12 > info.size() ? info.size() - i : 12};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HG")
            check = 1;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check == 1)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("CYSH");
      else
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("CYS2");
      i = pos - 1;
      continue;
    }

    if (info[i].get_res_name() == "ARG") {
      int check{};
      size_t pos{};
      size_t remain_num{i + 26 > info.size() ? info.size() - i : 26};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HH11" || info[ii].get_atom() == "HH12" ||
              info[ii].get_atom() == "1HH1" || info[ii].get_atom() == "2HH1")
            ++check;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check == 0)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("ARGN");
      i = pos - 1;
      continue;
    }

    if (info[i].get_res_name() == "GLN") {
      int check{};
      size_t pos{};
      size_t remain_num{i + 19 > info.size() ? info.size() - i : 19};
      for (size_t ii = i; ii < i + remain_num; ++ii) {
        if (info[ii].get_resnr() == info[i].get_resnr() &&
            info[ii].get_res_name() == info[i].get_res_name()) {
          if (info[ii].get_atom() == "HE1")
            check = 1;
        } else {
          pos = ii;
          break;
        }
      }
      if (pos == 0)
        pos = info.size();
      if (check == 1)
        for (size_t ii = i; ii < pos; ++ii)
          info[ii].set_res_name("QLN");
      i = pos - 1;
      continue;
    }

    size_t check_asp{check_res_H(info, "ASP", "HD2", i)};
    if (check_asp >= info.size())
      break;
    size_t check_lys{check_res_H(info, "LYS", "HZ3", i)};
    if (check_lys >= info.size())
      break;
  }
  this->generate_index();
  this->parse_type(amine);
}

size_t check_res_H(vector<mol_info> &protein_in, string_view res_name,
                   string_view atom_name, size_t i) {
  if (protein_in[i].get_res_name() == res_name) {
    int check{};
    size_t pos{0};
    size_t remain_num{i + 26 > protein_in.size() ? protein_in.size() - i : 26};
    for (size_t ii = i; ii < i + remain_num; ++ii) {
      if (protein_in[ii].get_resnr() == protein_in[i].get_resnr() &&
          protein_in[ii].get_res_name() == protein_in[i].get_res_name()) {
        if (protein_in[ii].get_atom() == atom_name)
          check++;
      } else {
        pos = ii;
        break;
      }
    }
    if (pos == 0)
      pos = protein_in.size();
    if (check == 1)
      for (size_t ii = i; ii < pos; ++ii)
        protein_in[ii].set_res_name(std::string(res_name) + "H");
    return pos - 1;
  } else
    return i;
}
