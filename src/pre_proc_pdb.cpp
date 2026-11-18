#include "arguments.h"
#include "functions.h"
#include "groups.h"
#include <fstream>
#include <iostream>

using namespace std;

int parse_begin_end_atoms_2mols(groups_cal &groups, arguments &Arguments,
                                Multi_MOL &PDB, Multi_MOL &protein_1,
                                Multi_MOL &protein_2, size_t begin_mol_1,
                                size_t begin_mol_2, size_t end_mol_1,
                                size_t end_mol_2, const amino_acid &amine,
                                string type_mol) {
  if (groups.empty()) {
    if (!PDB.m_mol.empty()) {
      PDB.wrapper_impl_post_proc(Arguments, type_mol, amine);
      if (groups.create_auto_index_file(Arguments, PDB, amine) != 0)
        return 1;
      if (groups.finalize_index_groups(Arguments) != 0)
        return 1;

      if (groups.groups_atom_number(groups.group_1_index()) == 0 ||
          (Arguments.needs_mol2_coordinates() &&
           groups.groups_atom_number(groups.group_2_index()) == 0)) {
        cout << "Selected index group is empty!\n";
        return 1;
      }

      begin_mol_1 = groups.atom_index(groups.group_1_index(), 0);
      begin_mol_2 = groups.atom_index(groups.group_2_index(), 0);
      end_mol_1 = groups.atom_index(
          groups.group_1_index(),
          groups.groups_atom_number(groups.group_1_index()) - 1);
      end_mol_2 = groups.atom_index(
          groups.group_2_index(),
          groups.groups_atom_number(groups.group_2_index()) - 1);
    }

    else if (Arguments.new_al_1pdb == true && !(Arguments.begin_mol_2 == 0)) {
      groups.add_group_from_range("molecule 1", 1ull,
                                  protein_1.m_mol[0].size());
      groups.add_group_from_range("molecule 2", protein_1.m_mol[0].size() + 1,
                                  protein_1.m_mol[0].size() +
                                      protein_2.m_mol[0].size());
    } else if (Arguments.flag_nocal == true &&
               !Arguments.needs_mol2_coordinates()) {
      end_mol_1 = PDB.m_mol[0].size();
      groups.add_group_from_range("molecule 1", 1ull, end_mol_1);
      end_mol_2 = 0;
      // -nocal analyzes a single molecule/reference only. Keep a placeholder
      // second group so legacy callers can still address group_2 safely
      // without duplicating the full system in XDR/PDB streaming paths.
      groups.add_empty_group("molecule 2");
    } else if (Arguments.begin_mol_2 == 0) {
      cout << "Please type the first atom number of molecule 2:\n";
      cin >> begin_mol_2;
      cout << "The molecule 2 begins from atom " << begin_mol_2 << '\n';

      end_mol_1 = begin_mol_2 - 1;
      groups.add_group_from_range("molecule 1", 1ull, end_mol_1);
      end_mol_2 = PDB.m_mol[0].size();
      groups.add_group_from_range("molecule 2", begin_mol_2, end_mol_2);
    } else {
      begin_mol_2 = Arguments.begin_mol_2;
      end_mol_1 = begin_mol_2 - 1;
      groups.add_group_from_range("molecule 1", 1ull, end_mol_1);
      end_mol_2 = PDB.m_mol[0].size();
      groups.add_group_from_range("molecule 2", begin_mol_2, end_mol_2);
    }
  } else {
    begin_mol_1 = groups.atom_index(groups.group_1_index(), 0);
    begin_mol_2 = groups.atom_index(groups.group_2_index(), 0);
    end_mol_1 = groups.atom_index(
        groups.group_1_index(),
        groups.groups_atom_number(groups.group_1_index()) - 1);
    end_mol_2 = groups.atom_index(
        groups.group_2_index(),
        groups.groups_atom_number(groups.group_2_index()) - 1);
  }

  if ((end_mol_1 > PDB.m_mol[0].size() || end_mol_2 > PDB.m_mol[0].size()) &&
      Arguments.new_al_1pdb == false) {
    cout << end_mol_1 << " " << end_mol_2 << "  " << PDB.m_mol[0].size()
         << '\n';
    cout << "Your chosen groups is not contained in pdb file!\n";
    if (PDB.m_mol[0].size() < 1)
      cout << "Please check your " << Arguments.get_input_file_type()
           << " file!\n";
    return 1;
  }
  return 0;
}

int pre_proc_pdb(string force_field, Multi_MOL &protein_1, Multi_MOL &protein_2,
                 Multi_MOL &water_traj, Multi_MOL &membrane_traj,
                 groups_cal &groups, int flag, arguments &Arguments,
                 const amino_acid &amine, info &output_info) {

  Multi_MOL PDB{};
  int flag_add_atom{0};

  const size_t reference_max_frames = 1;
  const bool can_subset_reference_directly =
      !Arguments.flag_xdr && !Arguments.needs_membrane_coordinates() &&
      !groups.empty() && Arguments.new_al_1pdb == false;

  if (can_subset_reference_directly) {
    bool saved_begin_set = Arguments.flag_begin_set;
    bool saved_end_set = Arguments.flag_end_set;
    Arguments.flag_begin_set = false;
    Arguments.flag_end_set = false;

    take_coordinate(protein_1, protein_2, water_traj, groups,
                    Arguments.pdb_name, force_field, Arguments, amine,
                    output_info, 1, reference_max_frames);

    Arguments.flag_begin_set = saved_begin_set;
    Arguments.flag_end_set = saved_end_set;
  }

  if (can_subset_reference_directly &&
      (!Arguments.needs_mol2_coordinates() ||
       (!protein_1.m_mol.empty() && !protein_2.m_mol.empty()))) {
    if (Arguments.n_wat >= 0)
      water_traj.set_mol_type("water");
  } else {
    protein_1.m_mol.clear();
    protein_1.info.clear();
    protein_1.parameters.clear();
    protein_1.Index.clear();
    protein_2.m_mol.clear();
    protein_2.info.clear();
    protein_2.parameters.clear();
    protein_2.Index.clear();
    water_traj.m_mol.clear();
    water_traj.info.clear();
    water_traj.parameters.clear();
    water_traj.Index.clear();
    membrane_traj.m_mol.clear();
    membrane_traj.info.clear();
    membrane_traj.parameters.clear();
    membrane_traj.Index.clear();

    if (Arguments.new_al_1pdb == false) {
    // Temporarily disable begin/end time filters for reference PDB loading.
    // The reference PDB provides topology only (typically time=0);
    // applying -b/-e filters here would reject it and cause a segfault.
      bool saved_begin_set = Arguments.flag_begin_set;
      bool saved_end_set = Arguments.flag_end_set;
      Arguments.flag_begin_set = false;
      Arguments.flag_end_set = false;

      if (PDB.take_coordinate(Arguments.pdb_name, force_field, Arguments, amine,
                              output_info, 1, reference_max_frames) == 1) {
        Arguments.flag_begin_set = saved_begin_set;
        Arguments.flag_end_set = saved_end_set;
        return 2;
      }
      Arguments.flag_begin_set = saved_begin_set;
      Arguments.flag_end_set = saved_end_set;
    } else {
      if (groups.empty() && Arguments.begin_mol_2 == 0) {
        PDB.take_coordinate(Arguments.pdb_name, force_field, Arguments, amine,
                            output_info, 1, reference_max_frames);
      } else {
        take_coordinate(protein_1, protein_2, water_traj, groups,
                        Arguments.pdb_name, force_field, Arguments, amine,
                        output_info, 1, reference_max_frames);
        flag_add_atom = 1;
      }
    }
  }

  string type{flag == 1 ? "lig" : "GROMOS"}, type_2{"GROMOS"};
  if (flag == 2) {
    type = "lig";
    type_2 = "lig";
  } else if (flag == 3) {
    type = "protein-user";
    type_2 = "protein-user";
  } else if (Arguments.force_f.find("GROMOS") != string::npos) {
  } else if (Arguments.force_f.find("AMBER") != string::npos) {
    type_2 = "AMBER";
    if (flag == 0)
      type = "AMBER";
  } else if (Arguments.force_f.find("CHARMM") != string::npos) {
    type_2 = "CHARMM";
    if (flag == 0)
      type = "CHARMM";
  } else {
    type_2 = "OPLSAA";
    if (flag == 0)
      type = "OPLSAA";
  }

  // Receptor-only -topo2: preserve its topology atom naming, while mol2
  // still follows the force-field library preprocessing used without -topo.
  if (flag == 4) {
    type = type_2;
    type_2 = "lig";
  }

  if (Arguments.flag_set_type_1 == true)
    type_2 = Arguments.mol_1_type;
  if (Arguments.flag_set_type_2 == true) {
    type = Arguments.mol_2_type;
    if (Arguments.mol_2_type == "protein" &&
        Arguments.force_f.find("AMBER") != string::npos)
      type = "AMBER";
    if (Arguments.mol_2_type == "protein" &&
        Arguments.force_f.find("CHARMM") != string::npos)
      type = "CHARMM";
    if (Arguments.mol_2_type == "protein" &&
        Arguments.force_f.find("OPLSAA") != string::npos)
      type = "OPLSAA";
  }

  if (can_subset_reference_directly &&
      (!Arguments.needs_mol2_coordinates() ||
       (!protein_1.m_mol.empty() && !protein_2.m_mol.empty()))) {
    if (flag != -1) {
      protein_1.wrapper_impl_post_proc(Arguments, type_2, amine);
      protein_2.wrapper_impl_post_proc(Arguments, type, amine);
    } else if (Arguments.flag_sidechain) {
      // A reload rebuilds info with default marks; classify side chains as
      // the first file did, or later files get whole-residue centres.
      protein_1.check_side_chain(Arguments);
      protein_2.check_side_chain(Arguments);
    }

    protein_1.set_mol_type(Arguments.mol_1_type);
    protein_2.set_mol_type(Arguments.mol_2_type);
    return 0;
  }

  size_t begin_mol_1{}, begin_mol_2{}, end_mol_1{}, end_mol_2{};

  if (parse_begin_end_atoms_2mols(groups, Arguments, PDB, protein_1, protein_2,
                                  begin_mol_1, begin_mol_2, end_mol_1,
                                  end_mol_2, amine, type_2) == 1)
    return 1;

  if (Arguments.n_wat >= 0) {
    add_atom_to_MOL(PDB, water_traj, groups, Arguments.n_wat, Arguments,
                    "water", amine, 0);
    water_traj.set_mol_type("water");
  }

  if (Arguments.needs_membrane_coordinates()) {
    int membrane_group = Arguments.membrane.resolved_group;
    if (membrane_group < 0) {
      if (Arguments.membrane.group == "auto") {
        const auto resolved = groups.find_group_index("lipid");
        if (!resolved) {
          cout << "Membrane analysis requires an index group named lipid or "
                  "-membrane-group=<N>.\n";
          return 1;
        }
        membrane_group = static_cast<int>(*resolved);
      } else {
        membrane_group = std::stoi(Arguments.membrane.group);
      }
      if (membrane_group < 0 ||
          static_cast<size_t>(membrane_group) >= groups.group_num() ||
          groups.groups_atom_number(static_cast<size_t>(membrane_group)) == 0) {
        cout << "Membrane analysis selected an invalid or empty lipid group.\n";
        return 1;
      }
      Arguments.membrane.resolved_group = membrane_group;
    }
    add_atom_to_MOL(PDB, membrane_traj, groups, membrane_group, Arguments,
                    "lipid", amine, 0);
    membrane_traj.set_mol_type("lipid");
  }

  trans_atom_to_MOL(PDB, protein_1, protein_2, groups, groups.group_1_index(),
                    groups.group_2_index(), Arguments, amine, flag_add_atom);

  if (Arguments.needs_mol2_coordinates() &&
      (protein_1.m_mol.empty() || protein_2.m_mol.empty())) {
    cout << "Cannot extract selected groups from reference structure "
         << Arguments.pdb_name << '\n';
    return 2;
  }

  if (flag != -1) {
    protein_1.wrapper_impl_post_proc(Arguments, type_2, amine);
    protein_2.wrapper_impl_post_proc(Arguments, type, amine);
  } else if (Arguments.flag_sidechain) {
    // See above: a reload must classify side chains again.
    protein_1.check_side_chain(Arguments);
    protein_2.check_side_chain(Arguments);
  }

  protein_1.set_mol_type(Arguments.mol_1_type);
  protein_2.set_mol_type(Arguments.mol_2_type);

  return 0;
}
int add_atom_to_MOL(Multi_MOL &MOL_source, Multi_MOL &protein_des,
                    groups_cal &groups, int group_index,
                    const arguments &Arguments, string type,
                    const amino_acid &amine, int flag = 0) {
  if (flag == 0) {
    for (size_t i{}; i < MOL_source.m_mol.size(); ++i) {
      for (size_t j{}; j < groups.groups_atom_number(group_index); ++j) {
        if (i == 0) {
          if (groups.atom_index(group_index, j) - 1 < 0 ||
              groups.atom_index(group_index, j) - 1 >=
                  MOL_source.m_mol[0].size()) {
            cout << "Your chosen group " << groups.group_name(group_index)
                 << " is not exist in pdb file!\n";
            exit(EXIT_FAILURE);
          }
        }
        auto mutact_info =
            MOL_source.info[groups.atom_index(group_index, j) - 1];
        mutact_info.set_original_atom_id(groups.atom_index(group_index, j) - 1);

        protein_des.emplace(
            i, MOL_source.m_mol[i].Box, MOL_source.m_mol[i].get_time(),
            MOL_source.m_mol[i].force_f, mutact_info,
            [&] {
              MOL m;
              m.pos.x.emplace_back(
                  MOL_source.m_mol[i]
                      .pos.x[groups.atom_index(group_index, j) - 1]);
              m.pos.y.emplace_back(
                  MOL_source.m_mol[i]
                      .pos.y[groups.atom_index(group_index, j) - 1]);
              m.pos.z.emplace_back(
                  MOL_source.m_mol[i]
                      .pos.z[groups.atom_index(group_index, j) - 1]);
              return m;
            }(),
            MOL_source.get_mol_type());
      }
    }
  }
  protein_des.gen_vector_parameters();
  return 0;
}

int trans_atom_to_MOL(Multi_MOL &MOL_source, Multi_MOL &protein_des,
                      Multi_MOL &protein_des_2, groups_cal &groups,
                      size_t group_index, size_t group_index_2,
                      const arguments &Arguments, const amino_acid &amine,
                      int flag = 0) {
  if (flag == 0) {
    if (Arguments.flag_nocal == true &&
        !Arguments.needs_mol2_coordinates() && groups.empty()) {
      protein_des = MOL_source;
      protein_des.gen_vector_parameters();
      return 0;
    }

    for (size_t i{}; i < MOL_source.m_mol.size(); ++i) {
      for (size_t j{}; j < groups.groups_atom_number(group_index); ++j) {
        size_t atom_idx_val = groups.atom_index(group_index, j);
        if (atom_idx_val == 0) {
          std::cout << "ERROR: Invalid atom_index 0 returned for group "
                    << group_index << " j=" << j << "\n";
          exit(EXIT_FAILURE);
        }
        size_t info_idx = atom_idx_val - 1;
        if (info_idx >= MOL_source.info.size()) {
          std::cout << "ERROR: info index " << info_idx
                    << " out of bounds (size=" << MOL_source.info.size()
                    << ")\n";
          exit(EXIT_FAILURE);
        }
        if (i == 0) {
          if (info_idx >= MOL_source.m_mol[0].size()) {
            std::cout << "Your chosen group " << groups.group_name(group_index)
                      << " is not exist in pdb file!\n";
            exit(EXIT_FAILURE);
          }
        }
        auto mut_info = MOL_source.info[info_idx];
        mut_info.set_original_atom_id(info_idx);

        protein_des.emplace(
            i, MOL_source.m_mol[i].Box, MOL_source.m_mol[i].get_time(),
            MOL_source.m_mol[i].force_f, mut_info,
            [&] {
              MOL m;
              m.pos.x.emplace_back(MOL_source.m_mol[i].pos.x[info_idx]);
              m.pos.y.emplace_back(MOL_source.m_mol[i].pos.y[info_idx]);
              m.pos.z.emplace_back(MOL_source.m_mol[i].pos.z[info_idx]);
              return m;
            }(),
            MOL_source.get_mol_type());
      }
      if (Arguments.flag_nocal == true &&
          !Arguments.needs_mol2_coordinates())
        continue;

      for (size_t j{}; j < groups.groups_atom_number(group_index_2); ++j) {
        if (i == 0) {
          if (groups.atom_index(group_index_2, j) - 1 < 0 ||
              groups.atom_index(group_index_2, j) - 1 >=
                  MOL_source.m_mol[0].size()) {
            cout << "Your chosen group " << groups.group_name(group_index_2)
                 << " is not exist in pdb file!\n";
            exit(EXIT_FAILURE);
          }
        }
        auto mut_info2 =
            MOL_source.info[groups.atom_index(group_index_2, j) - 1];
        mut_info2.set_original_atom_id(groups.atom_index(group_index_2, j) - 1);

        protein_des_2.emplace(
            i, MOL_source.m_mol[i].Box, MOL_source.m_mol[i].get_time(),
            MOL_source.m_mol[i].force_f, mut_info2,
            [&] {
              MOL m;
              m.pos.x.emplace_back(
                  MOL_source.m_mol[i]
                      .pos.x[groups.atom_index(group_index_2, j) - 1]);
              m.pos.y.emplace_back(
                  MOL_source.m_mol[i]
                      .pos.y[groups.atom_index(group_index_2, j) - 1]);
              m.pos.z.emplace_back(
                  MOL_source.m_mol[i]
                      .pos.z[groups.atom_index(group_index_2, j) - 1]);
              return m;
            }(),
            MOL_source.get_mol_type());
      }

      MOL_source.m_mol[i] = MOL();
    }
  }
  protein_des.gen_vector_parameters();
  protein_des_2.gen_vector_parameters();
  return 0;
}
