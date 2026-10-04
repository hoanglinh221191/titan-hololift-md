#include "Multi_MOL.h"
#include <map>
#include <set>
using namespace std;
using index_type = class index;

bool find_lipid(string res_name) {
  set<string> list_lipid{"DOPA", "DOPC", "DOPE", "DOPG", "DOPS", "DMPA", "DMPC",
                         "DMPG", "DMPI", "DMPR", "DMPS", "DMPU", "DDPC", "DPPA",
                         "DPPC", "DPPE", "DPPG", "DPPS", "DPRO", "CER1", "BGAL",
                         "BGAM", "BGAN", "BGLC", "ANE5", "POPC", "CHL1", "PSM"};
  return (list_lipid.find(res_name) != list_lipid.end());
}

int Multi_MOL::generate_index() {
  if (m_mol.empty())
    return 1;

  index_type index_temp{};
  size_t index_i{}, index_e{};
  index_temp.set_chain(info[0].get_chain());
  Index.emplace_back(index_temp);
  Index[0].add_base(1u, info[0].get_resnr(), 0.0, 0.0, 0.0,
                    info[0].get_res_name());

  for (size_t i = 1; i < info.size(); ++i)
    if (Index[index_i].get_chain() == info[i].get_chain()) {
      if (Index[index_i].get_res_num_e(index_e) != info[i].get_resnr() ||
          (Index[index_i].get_res_num_e(index_e) == info[i].get_resnr() &&
           Index[index_i].get_res_name_e(index_e) != info[i].get_res_name())) {
        if (Index[index_i].get_size_v() < 5) {
        }
        Index[index_i].add_base(i + 1u, info[i].get_resnr(), 0.0, 0.0, 0.0,
                                info[i].get_res_name());
        ++index_e;
      } else {
      }
    } else {
      index_temp.set_chain(info[i].get_chain());
      Index.emplace_back(index_temp);
      ++index_i;
      Index[index_i].add_base(i + 1u, info[i].get_resnr(), 0.0, 0.0, 0.0,
                              info[i].get_res_name());
      index_e = 0u;
    }
  Index.emplace_back(index_type(std::string("NULL_S"), info.size() + 1));
  return 0;
}
int Multi_MOL::check_dna_or_rna(const amino_acid &amine) {
  map<int, string> type_mol;
  for (size_t i{}; i < Index.size() - 1; ++i) {
    map<int, string> type_res;
    for (size_t j{}; j < Index[i].get_size_v(); ++j) {
      string res_name{info[Index[i].get_res_pos_e(j) - 1].get_res_name()};
      if (res_name == "U" || res_name == "A" || res_name == "C" ||
          res_name == "G" || res_name == "T" || res_name == "DA" ||
          res_name == "DC" || res_name == "DG" || res_name == "DT" ||
          res_name == "RA" || res_name == "RC" || res_name == "RG" ||
          res_name == "RU" || res_name == "ADE" || res_name == "CYT" ||
          res_name == "GUA" || res_name == "URA" || res_name == "THY") {
        bool flag_base{false};
        size_t remain_index{j + 1 == Index[i].get_size_v()
                                ? Index[i + 1].get_res_pos_e(0) - 1
                                : Index[i].get_res_pos_e(j + 1) - 1};
        for (size_t k{Index[i].get_res_pos_e(j) - 1}; k < remain_index; ++k) {
          if (info[k].get_atom() == "O2'" ||
              info[k].get_original_res_name() == "U") {
            Index[i].set_type_base(j, "rna");
            type_res[1] = "rna";
            flag_base = true;
          }
        }
        if (flag_base == false) {
          Index[i].set_type_base(j, "dna");
          type_res[0] = "dna";
        }
      } else if (amine.check_res_name(res_name)) {
        if (res_name == "SOL" || res_name == "WAT") {
          Index[i].set_type_base(j, "water");
          type_res[4] = "water";
        }
        if (res_name == "NA" || res_name == "CL" || res_name == "MG" ||
            res_name == "K" || res_name == "CU" || res_name == "FE" ||
            res_name == "CA" || res_name == "ZN") {
          Index[i].set_type_base(j, "ion");
          type_res[5] = "ion";
        }
        if (res_name == "CER1" || res_name == "ANE5" || res_name == "DPPC" ||
            res_name == "POPC" || res_name == "CHL1" || res_name == "DDPC" ||
            find_lipid(res_name)) {
          Index[i].set_type_base(j, "lipid");
          type_res[6] = "lipid";
        } else {
          Index[i].set_type_base(j, "protein");
          type_res[2] = "protein";
        }
      } else {
        Index[i].set_type_base(j, "ligand");
        type_res[3] = "ligand";
      }
    }
    if (type_res.size() > 1)
      Index[i].set_type("complex");
    else
      Index[i].set_type(type_res.begin()->second);

    for (auto &type : type_res)
      type_mol.insert(type);
  }

  if (type_mol.size() > 1)
    this->set_mol_type("complex");
  if (type_mol.size() == 1)
    this->set_mol_type(type_mol.begin()->second);
  else
    this->set_mol_type("unknown");
  return 0;
}
int Multi_MOL::post_process_nucleotide() {
  for (size_t i{}; i < Index.size() - 1; ++i) {
    if (Index[i].get_type() != "protein" &&
        force_f.find("AMBER") != string::npos) {
      string prefix{};
      if (Index[i].get_type() == "dna")
        prefix = "D";
      else if (Index[i].get_type() == "rna")
        prefix = "R";
      else {
      }

      if (Index[i].get_size_v() == 1) {
        for (size_t j{Index[i].get_res_pos_e(0) - 1};
             j < Index[i + 1].get_res_pos_e(0) - 1; ++j)
          info[j].set_res_name(prefix + info[j].get_res_name() + "N");
      } else {
        for (size_t j = Index[i].get_res_pos_e(0) - 1;
             j < Index[i].get_res_pos_e(1) - 1; ++j) {
          if (info[j].get_res_name()[0] == 'D' ||
              info[j].get_res_name()[0] == 'R')
            prefix = {};
          info[j].set_res_name(prefix + info[j].get_res_name() + "5");
        }
        for (size_t j = Index[i].get_res_pos_e(Index[i].get_size_v() - 1) - 1;
             j < Index[i + 1].get_res_pos_e(0) - 1; ++j) {
          if (info[j].get_res_name()[0] == 'D' ||
              info[j].get_res_name()[0] == 'R')
            prefix = {};
          info[j].set_res_name(prefix + info[j].get_res_name() + "3");
        }
        for (size_t j = Index[i].get_res_pos_e(1) - 1;
             j < Index[i].get_res_pos_e(Index[i].get_size_v() - 1) - 1; ++j) {
          if (info[j].get_res_name()[0] == 'D' ||
              info[j].get_res_name()[0] == 'R')
            prefix = {};
          info[j].set_res_name(prefix + info[j].get_res_name());
        }
      }
    }
  }
  return 0;
}
int Multi_MOL::post_process_protein() {
  for (size_t i = 0; i < Index.size() - 1; ++i) {
    if (Index[i].get_type() == "protein" && Index[i].get_size_v() > 1) {
      for (auto j = Index[i].get_res_pos_e(0) - 1;
           j < Index[i].get_res_pos_e(1) - 1; ++j)
        info[j].set_res_name("N" + info[j].get_res_name());
      for (auto j = Index[i].get_res_pos_e(Index[i].get_size_v() - 1) - 1;
           j < Index[i + 1].get_res_pos_e(0) - 1; ++j)
        info[j].set_res_name("C" + info[j].get_res_name());
    }
  }
  return 0;
}
int Multi_MOL::post_process_type_res(class index &in, size_t pos_res) {
  string type_of_res{this->check_head_tail_res(in, pos_res)};
  if (this->force_f.find("AMBER") != string::npos &&
      in.get_type_base(pos_res) == "protein") {
    string prefix{};
    if (type_of_res == "cterminal")
      prefix = "C";
    else if (type_of_res == "nterminal")
      prefix = "N";
    else
      return 0;
    size_t remain{in.get_res_pos_e(pos_res) + 50 < m_mol[0].size()
                      ? in.get_res_pos_e(pos_res) + 50
                      : m_mol[0].size()};
    for (size_t i{in.get_res_pos_e(pos_res) - 1};
         (i < remain) && (info[i].get_resnr() ==
                          info[in.get_res_pos_e(pos_res) - 1].get_resnr());
         ++i) {
      if ((prefix == "C" || prefix == "N")) {
        info[i].set_res_name(prefix + info[i].get_res_name());
      } else
        break;
    }
  } else if ((this->force_f.find("AMBER") != string::npos) &&
             (in.get_type_base(pos_res) == "dna" ||
              in.get_type_base(pos_res) == "rna")) {
    string prefix{}, postfix{};
    if (in.get_type_base(pos_res) == "dna")
      prefix = "D";
    else
      prefix = "R";

    if (type_of_res == "nsingle")
      postfix = "N";
    else if (type_of_res == "3terminal")
      postfix = "3";
    else if (type_of_res == "5terminal")
      postfix = "5";
    else {
    }

    size_t remain{in.get_res_pos_e(pos_res) + 50 < m_mol[0].size()
                      ? in.get_res_pos_e(pos_res) + 50
                      : m_mol[0].size()};
    for (size_t i{in.get_res_pos_e(pos_res) - 1};
         (i < remain) && (info[i].get_resnr() ==
                          info[in.get_res_pos_e(pos_res) - 1].get_resnr());
         ++i) {
      if (info[i].get_res_name()[0] != 'D' && info[i].get_res_name()[0] != 'R')
        info[i].set_res_name(prefix + info[i].get_res_name());

      if (postfix == "3" || postfix == "N" || postfix == "5")
        info[i].set_res_name(info[i].get_res_name() + postfix);
    }
  } else
    return 0;
  return 0;
}
int Multi_MOL::parse_type(const amino_acid &amine) {
  this->check_dna_or_rna(amine);
  for (size_t i{}; i < Index.size() - 1; ++i) {
    for (size_t j{}; j < Index[i].get_size_v(); ++j)
      this->post_process_type_res(Index[i], j);
  }
  return 0;
}
