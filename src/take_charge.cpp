#include "Multi_MOL.h"
#include "titan_fflib_assigner.h"
#include <iostream>
#include <string_view>
using std::string_view;


int Multi_MOL::take_charge(amino_acid &amine, const para &parameter, size_t end,
                           size_t begin, string_view force_f, int flag,
                           const arguments *ArgumentsPtr,
                           TitanExecutor *executor,
                           string_view molecule_label) {
  auto assign_type_ids = [&]() {
    for (size_t i = begin; i < end && i < info.size() && i < parameters.size();
         ++i) {
      const auto it = parameter.atom_type_to_id.find(info[i].get_atom_type());
      if (it != parameter.atom_type_to_id.end())
        parameters[i].set_type_id(static_cast<int>(it->second));
    }
  };

  if (flag == 0 && ArgumentsPtr != nullptr && ArgumentsPtr->fflib_auto &&
      begin == 0 && end >= info.size()) {
    const string label =
        molecule_label.empty() ? string(this->get_mol_type()) : string(molecule_label);
    return TitanFfLib::assign_from_library(*this, amine, parameter,
                                           *ArgumentsPtr, force_f, executor,
                                           label);
  }

  if (begin >= info.size() || end > info.size()) {
  }
  if (begin >= parameters.size() || end > parameters.size()) {
  }
  if (flag == 0) {
    if (begin == 0) {
      for (size_t i{}; i < Index.size() - 1; ++i)
        for (size_t j{}; j < Index[i].get_size_v(); ++j) {
          if (amine.amino_acids.find(
                  info[Index[i].get_res_pos_e(j) - 1].get_res_name()) ==
              amine.amino_acids.end()) {
            cout << "WARNING (" << mol_type << ")!! Residue "
                 << Index[i].get_chain() << Index[i].get_res_num_e(j) << " ("
                 << info[Index[i].get_res_pos_e(j) - 1].get_res_name()
                 << ") is not contained in force field!\n";
            continue;
          }
        }
    }

    for (size_t i{begin}; i < end; ++i) {
      string aa_name{string(info[i].get_res_name())};

      for (size_t j{}; j < amine.amino_acids[aa_name].size(); ++j) {
        if (info[i].get_atom() == amine.amino_acids[aa_name][j].atom) {
          info[i].set_atom_type(amine.amino_acids[aa_name][j].atom_type);
          parameters[i].set_data(amine.amino_acids[aa_name][j].charge,
                                 amine.amino_acids[aa_name][j].radius,
                                 amine.amino_acids[aa_name][j].sigma,
                                 amine.amino_acids[aa_name][j].epsilon,
                                 amine.amino_acids[aa_name][j].atom_mass);
          break;
        }
      }
    }

    // find strange residues :D
    if ((force_f == "GROMOS43A1" || force_f == "GROMOS53A6" ||
         force_f == "GROMOS54A7" || force_f == "OPLSAA") &&
        end == m_mol[0].size()) {
      for (size_t i{}; i < m_mol[0].size() - 1; ++i) {
        amino_acid temp_acid;
        vector<::mol_info> temp_pro;
        ::mol_info temp_p;
        vector<size_t> ite;
        if (info[i].get_atom_type() == "NULL") {
          ite.emplace_back(i);
          for (size_t j = i; j < m_mol[0].size(); ++j)
            if (info[j].get_atom_type() == "NULL" &&
                info[j].get_resnr() == info[i].get_resnr()) {
              temp_p.set_atom(info[j].get_atom());
              temp_pro.emplace_back(temp_p);
            } else {
              i = j;
              break;
            }
          if (!temp_pro.empty()) {
            for (auto &amino_acid : amine.amino_acids) {
              for (size_t j{}; j < amino_acid.second.size(); ++j) {
                if (j + temp_pro.size() - 1 < amino_acid.second.size() &&
                    amino_acid.second[j].atom.substr(0, 1) ==
                        temp_pro[0].get_atom().substr(0, 1) &&
                    amino_acid.second[j].atom.substr(
                        amino_acid.second[j].atom.length() - 1, 1) ==
                        temp_pro[0].get_atom().substr(
                            temp_pro[0].get_atom().length() - 1, 1)) {
                  if (temp_pro[temp_pro.size() - 1].get_atom().substr(0, 1) ==
                          amino_acid.second[j + temp_pro.size() - 1]
                              .atom.substr(0, 1) &&
                      amino_acid.second[j + temp_pro.size() - 1].atom.substr(
                          amino_acid.second[j + temp_pro.size() - 1]
                                  .atom.length() -
                              1,
                          1) ==
                          temp_pro[temp_pro.size() - 1].get_atom().substr(
                              temp_pro[temp_pro.size() - 1]
                                      .get_atom()
                                      .length() -
                                  1,
                              1)) {
                    for (size_t k = 0; k < amino_acid.second.size(); ++k)
                      if (amino_acid.second[k].charge_group ==
                          amino_acid.second[j].charge_group) {
                        temp_acid.amino_acids[amino_acid.first].emplace_back(
                            amino_acid.second[k].charge_group,
                            amino_acid.second[k].charge,
                            amino_acid.second[k].atom_mass,
                            amino_acid.second[k].epsilon,
                            amino_acid.second[k].sigma,
                            amino_acid.second[k].radius,
                            amino_acid.second[k].atom,
                            amino_acid.second[k].atom_type);
                      }
                  }
                }
              }
            }
          }
        }
        if (!temp_acid.amino_acids.empty()) {
          for (size_t i = 0; i < ite.size(); ++i) {
            auto amino_acid = temp_acid.amino_acids.begin();
            for (size_t k = 0; k < amino_acid->second.size(); ++k)
              for (size_t j{}; j < amino_acid->second.size(); ++j)
                if (ite[i] + k - 1 >= 0 && ite[i] + k - 1 < m_mol[0].size() &&
                    info[ite[i] + k - 1].get_atom().substr(0, 1) ==
                        amino_acid->second[j].atom.substr(0, 1)) {
                  info[ite[i] + k - 1].set_atom_type(
                      amino_acid->second[j].atom_type);
                  parameters[ite[i] + k - 1].set_data(
                      amino_acid->second[j].charge,
                      amino_acid->second[j].radius, amino_acid->second[j].sigma,
                      amino_acid->second[j].epsilon,
                      amino_acid->second[j].atom_mass);
                }
          }
          temp_acid.amino_acids.clear();
        }
        temp_pro.clear();
        ite.clear();
      }
      if (force_f == "OPLSAA") {
        vector<size_t> heads;
        for (size_t i{}; i < Index.size() - 1; ++i) {
          if (Index[i].get_res_pos_e(0) - 1 >= begin &&
              Index[i].get_res_pos_e(0) - 1 < end)
            heads.emplace_back(i);
        }
        for (size_t i{}; i < heads.size(); ++i) {
          vector<::mol_info> temp_pro;
          ::mol_info temp_p;
          for (size_t j = Index[i].get_res_pos_e(0) - 1;
               j < Index[i].get_res_pos_e(0) + 3; ++j) {
            temp_p.set_atom(info[j].get_atom());
            temp_pro.emplace_back(temp_p);
          }
          size_t pos{};
          string pos_amino{};
          for (auto &amino : amine.amino_acids) {
            bool check_stop_outer{false};
            size_t pos_temp{};
            for (size_t k{}; k < amino.second.size(); ++k) {
              bool check_stop{false};
              if (amino.second[k].atom.substr(0, 1) ==
                  temp_pro[0].get_atom().substr(0, 1)) {
                for (size_t ite_atom{1}; ite_atom < temp_pro.size(); ++ite_atom)
                  if (k + ite_atom >= amino.second.size() ||
                      amino.second[k].charge_group !=
                          amino.second[k + ite_atom].charge_group ||
                      amino.second[k + ite_atom].atom.substr(0, 1) !=
                          temp_pro[ite_atom].get_atom().substr(0, 1)) {
                    check_stop = true;
                    break;
                  } else
                    check_stop = false;
                pos_temp = k;
              } else
                check_stop = true;
              check_stop_outer = check_stop;
              if (check_stop == false) {
                break;
              }
            }
            if (check_stop_outer == false) {
              pos = pos_temp;
              pos_amino = amino.first;
              break;
            }
          }
          for (size_t k{}; k < 4; ++k) {
            info[Index[i].get_res_pos_e(0) - 1 + k].set_atom_type(
                amine.amino_acids[pos_amino][pos + k].atom_type);
            parameters[Index[i].get_res_pos_e(0) - 1 + k].set_data(
                amine.amino_acids[pos_amino][pos + k].charge,
                amine.amino_acids[pos_amino][pos + k].radius,
                amine.amino_acids[pos_amino][pos + k].sigma,
                amine.amino_acids[pos_amino][pos + k].epsilon,
                amine.amino_acids[pos_amino][pos + k].atom_mass);
          }

          double charge_trial{20.}, charge_original{};
          for (size_t k{};
               k < amine
                       .amino_acids[info[Index[i].get_res_pos_e(0) - 1]
                                        .get_res_name()]
                       .size();
               ++k)
            charge_original +=
                amine
                    .amino_acids[info[Index[i].get_res_pos_e(0) - 1]
                                     .get_res_name()][k]
                    .charge;
          charge_original += 1.;
          double charge_total{};
          for (size_t k{};
               k < Index[i].get_res_pos_e(1) - Index[i].get_res_pos_e(0); ++k)
            if (k != 4) {
              charge_total +=
                  parameters[Index[i].get_res_pos_e(0) - 1 + k].get_charge();
            }
          size_t poss{};
          for (size_t j{}; j < parameter.base.size(); ++j) {
            if (parameter.base[j].code_name.substr(0, 1) ==
                info[Index[i].get_res_pos_e(0) - 1 + 4].get_atom().substr(0,
                                                                          1)) {
              if (fabs(charge_original - charge_total -
                       parameter.base[j].charge) <
                  fabs(charge_original - charge_trial)) {
                charge_trial = charge_total;
                poss = j;
              }
            }
          }
          info[Index[i].get_res_pos_e(0) - 1 + 4].set_atom_type(
              parameter.base[poss].atom_type);
          parameters[Index[i].get_res_pos_e(0) - 1 + 4].set_data(
              parameter.base[poss].charge, parameter.base[poss].radius,
              parameter.base[poss].c6_sigma, parameter.base[poss].c12_epsilon,
              parameter.base[poss].mass);
        }
      }
    } else if (force_f.find("CHARMM") != string::npos) {
      vector<size_t> heads, tails;
      for (size_t i{}; i < Index.size() - 1; ++i) {
        if (Index[i].get_type() != "protein")
          continue;
        if (Index[i].get_res_pos_e(0) - 1 >= begin &&
            Index[i].get_res_pos_e(0) - 1 < end)
          heads.emplace_back(i);
        if (Index[i].get_res_pos_e(Index[i].get_size_v() - 1) - 1 < end &&
            Index[i].get_res_pos_e(Index[i].get_size_v() - 1) - 1 >= begin)
          tails.emplace_back(i);
      }

      for (size_t i{}; i < heads.size(); ++i) {
        double tot_charge_r1{};
        for (size_t j{Index[heads[i]].get_res_pos_e(0) - 1};
             j < Index[heads[i]].get_res_pos_e(0) + 3; ++j) {
          if (info[j].get_atom().find("N") != string::npos)
            info[j].set_atom_type("NH3");
          if (info[j].get_atom().find("H") != string::npos)
            info[j].set_atom_type("HC");
          for (auto &amino_acid : amine.amino_acids) {
            for (size_t jj{}; jj < amino_acid.second.size(); ++jj)
              if (info[j].get_atom_type() == amino_acid.second[jj].atom_type &&
                  jj + Index[heads[i]].get_res_pos_e(0) >= 1 + j &&
                  amino_acid
                          .second[jj + Index[heads[i]].get_res_pos_e(0) - 1 - j]
                          .atom_type == "NH3") {
                parameters[j].set_data(
                    amino_acid.second[jj].charge, amino_acid.second[jj].radius,
                    amino_acid.second[jj].sigma, amino_acid.second[jj].epsilon,
                    amino_acid.second[jj].atom_mass);
                break;
              }
          }
        }

        size_t end_pos{Index[heads[i]].get_size_v() > 1
                           ? Index[heads[i]].get_res_pos_e(1) - 1
                           : Index[heads[i] + 1].get_res_pos_e(0) - 1};
        for (size_t j{Index[heads[i]].get_res_pos_e(0) - 1}; j < end_pos; ++j) {
          if (j != Index[heads[i]].get_res_pos_e(0) + 4 &&
              j != Index[heads[i]].get_res_pos_e(0) + 3)
            tot_charge_r1 += parameters[j].get_charge();
        }

        parameters[Index[heads[i]].get_res_pos_e(0) + 4].set_charge(0.1);
        if (tot_charge_r1 + 0.1 > 0.5)
          parameters[Index[heads[i]].get_res_pos_e(0) + 3].set_charge(
              0.9 - tot_charge_r1);
        else if (tot_charge_r1 + 0.1 < -0.5)
          parameters[Index[heads[i]].get_res_pos_e(0) + 3].set_charge(
              -1.1 - tot_charge_r1);
        else
          parameters[Index[heads[i]].get_res_pos_e(0) + 3].set_charge(
              -0.1 - tot_charge_r1);
      }

      for (size_t i{}; i < tails.size(); ++i) {
        for (size_t j{
                 static_cast<size_t>(Index[tails[i] + 1].get_res_pos_e(0) - 4)};
             j < Index[tails[i] + 1].get_res_pos_e(0) - 1; ++j) {
          if (info[j].get_atom().find("C") != string::npos) {
            parameters[j].set_data(0.34, 5.0 * 0.356359487256, 0.356359487256,
                                   0.29288, 12.011);
            info[j].set_atom_type("CC");
          }
          if (info[j].get_atom().find("O") != string::npos) {
            parameters[j].set_data(-0.67, 5.0 * 0.302905564168, 0.302905564168,
                                   0.50208, 15.9994);
            info[j].set_atom_type("OC");
          }
        }
      }
    }
  } else {
    for (size_t i = begin; i < end && i < info.size() && i < parameters.size();
         ++i)
      TitanFfLib::fill_vdw_from_atom_type(parameters, i, parameter,
                                          info[i].get_atom_type());
  }
  assign_type_ids();
  return 0;
}
