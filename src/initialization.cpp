#include "MOL.h"
#include "arguments.h"
#include "functions.h"
#include "groups.h"
#include "topology/lj_topology.h"
#include "titan_ui.h"
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int helper_parse_groups(groups_cal &Groups_from_index_file,
                        arguments &Arguments) {
  return Groups_from_index_file.get_parse_index_flag() == false
             ? Groups_from_index_file.index(Arguments)
             : 0;
}

std::expected<void, std::string> initialize_topology_streaming(
    arguments &Arguments, fstream &log, string name_pro, string name_lig,
    string force_f, int input_name, vector<string> &input_line,
    bool topo_2_is_open, bool topo_is_open, Multi_MOL &pro_in, Multi_MOL &lig,
    Multi_MOL &water_traj, Multi_MOL &membrane_traj,
    groups_cal &Groups_from_index_file, para &parameter, amino_acid &amine,
    info &output_info) {

  auto show_init_pdb_progress = [](const std::string &label, bool completed) {
    TitanUI::show_progress_exact(label, completed ? 1U : 0U, 1U, "Loading PDB");
  };

  auto file_label = [](const std::string &path_text) {
    const fs::path path(path_text);
    const std::string filename = path.filename().string();
    return filename.empty() ? path_text : filename;
  };

  int flag_groups{helper_parse_groups(
      Groups_from_index_file,
      Arguments)}; // ---------------------------------------------------------------------------
  if (Arguments.gist.flag_gist && Arguments.flag_index && flag_groups!=0)
    return std::unexpected("GIST index/water group selection failed.");
  // SCENARIO 1: SINGLE PDB FILE (-f, -i list, -pattern)
  // ---------------------------------------------------------------------------
  if (Arguments.flag_1_pdb) {

    // 1. Resolve the correct filename
    if ((Arguments.flag_listfile || Arguments.flag_all_folders) &&
        !input_line.empty() && input_name < input_line.size() &&
        !Arguments.flag_xdr) {
      Arguments.pdb_name = input_line[input_name];
      if (Arguments.pdb_name.length() <= 4 ||
          !Arguments.pdb_name.ends_with(".pdb")) {
        Arguments.pdb_name += ".pdb";
      }
    } else if (Arguments.flag_pattern_input) {
      auto asterisk_pos = Arguments.input_pattern.find("*");
      std::string prefix = Arguments.input_pattern.substr(0, asterisk_pos);
      std::string suffix = Arguments.input_pattern.substr(asterisk_pos + 1);

      if (Arguments.flag_range_input) {
        Arguments.pdb_name =
            prefix + input_line[input_name - Arguments.b_input] + suffix;
      } else {
        Arguments.pdb_name = prefix + std::to_string(input_name) + suffix;
      }
      Arguments.add_input_file_type();
    }

    // 2. LJ single-molecule mode does not need the protein topology pipeline.
    // Use the reference structure directly and only parse the common topology
    // for lightweight atom-count validation.
    if (Arguments.flag_nocal && Arguments.lj.flag_lj) {
      show_init_pdb_progress(file_label(Arguments.pdb_name), false);
      const int r_value =
          pre_proc_pdb(force_f, pro_in, lig, water_traj, membrane_traj,
                       Groups_from_index_file, 0, Arguments, amine, output_info);
      show_init_pdb_progress(file_label(Arguments.pdb_name), true);
      if (r_value != 0)
        return std::unexpected("pre_proc_pdb failed.");

      if (Arguments.topo_name != "no" && (!Arguments.file_par || topo_is_open)) {
        auto lj_topology = parse_lj_topology_summary(Arguments.topo_name);
        if (!lj_topology)
          return std::unexpected(lj_topology.error());

        TitanUI::log_kv(log, "LJ Topology", Arguments.topo_name);
        TitanUI::log_kv(log, "LJ Molecule",
                        lj_topology->primary_molecule.empty()
                            ? std::string("unknown")
                            : lj_topology->primary_molecule);
        TitanUI::log_kv(log, "LJ Topology Atoms",
                        std::to_string(lj_topology->total_atoms));

        if (!pro_in.m_mol.empty()) {
          const size_t reference_atoms = pro_in.m_mol[0].size();
          if (lj_topology->total_atoms != reference_atoms) {
            const std::string warning =
                "LJ topology atom count (" +
                std::to_string(lj_topology->total_atoms) +
                ") differs from reference atoms (" +
                std::to_string(reference_atoms) + ")";
            TitanUI::print_warning(warning);
            TitanUI::log_kv(log, "LJ Topology Warning", warning);
          }
        }
      }

      lig.m_mol.clear();
      lig.info.clear();
      lig.Index.clear();
      return {};
    }

    // 3. Build Topology using pre_proc_pdb (reads Frame 0 internally)
    int flag_pre_proc = 1;
    if (topo_2_is_open)
      flag_pre_proc = topo_is_open ? 2 : 4; // 4: receptor topology only
    if (Arguments.flag_use_common_topo)
      flag_pre_proc = 3;

    if ((Arguments.topo_name != "no" && (!Arguments.file_par || topo_is_open)) ||
        (Arguments.topo_name_2 != "no" && (!Arguments.file_par || topo_2_is_open))) {
      show_init_pdb_progress(file_label(Arguments.pdb_name), false);
      const int r_value =
          pre_proc_pdb(force_f, pro_in, lig, water_traj, membrane_traj,
                       Groups_from_index_file, flag_pre_proc, Arguments, amine,
                       output_info);
      show_init_pdb_progress(file_label(Arguments.pdb_name), true);
      if (r_value != 0)
        return std::unexpected("pre_proc_pdb failed.");

      if (flag_pre_proc != 3) {
        if (flag_pre_proc != 4)
          lig.topo_lg(Arguments.topo_name, force_f, parameter);
        if (flag_pre_proc == 2 || flag_pre_proc == 4)
          pro_in.topo_lg(Arguments.topo_name_2, force_f, parameter);
      } else {
        vector<topo_info_temp> info_topo;
        const bool pro_failed =
            pro_in.topo_protein(Arguments.topo_name_2, force_f, parameter,
                                info_topo) == 1;
        const bool need_lig_topology =
            Arguments.needs_mol2_coordinates() && !lig.info.empty();
        const bool lig_failed =
            need_lig_topology &&
            lig.topo_protein(Arguments.topo_name, force_f, parameter,
                             info_topo) == 1;
        if (pro_failed || lig_failed) {
          return std::unexpected("Cannot parse topology files");
        }
      }
    } else {
      show_init_pdb_progress(file_label(Arguments.pdb_name), false);
      const int r_value =
          pre_proc_pdb(force_f, pro_in, lig, water_traj, membrane_traj,
                       Groups_from_index_file, 0, Arguments, amine, output_info);
      show_init_pdb_progress(file_label(Arguments.pdb_name), true);
      if (r_value != 0)
        return std::unexpected("pre_proc_pdb failed.");
    }

    return {};
  }
  // ---------------------------------------------------------------------------
  // SCENARIO 2: TWO SEPARATE PDB FILES (Receptor and Ligand)
  // ---------------------------------------------------------------------------
  else {
    show_init_pdb_progress("rec+lig", false);
    // 1. Build Topology using legacy take_coordinate (reads Frame 0)
    if (topo_is_open && !topo_2_is_open) {
      pro_in.take_coordinate(name_pro, force_f, Arguments, amine, output_info,
                             0, 1);
      lig.take_topo_lg(name_lig, Arguments.topo_name, Arguments, force_f,
                       parameter, amine, output_info, Arguments.mol_2_type);
    } else if (topo_is_open && topo_2_is_open) {
      lig.take_topo_lg(name_lig, Arguments.topo_name, Arguments, force_f,
                       parameter, amine, output_info, Arguments.mol_2_type);
      pro_in.take_topo_lg(name_pro, Arguments.topo_name_2, Arguments, force_f,
                          parameter, amine, output_info, Arguments.mol_1_type);
    } else if (!topo_is_open && topo_2_is_open) {
      pro_in.take_topo_lg(name_pro, Arguments.topo_name_2, Arguments, force_f,
                          parameter, amine, output_info, Arguments.mol_1_type);
      lig.take_coordinate(name_lig, force_f, Arguments, amine, output_info, 0,
                          1);
    } else {
      pro_in.take_coordinate(name_pro, force_f, Arguments, amine, output_info,
                             0, 1);
      lig.take_coordinate(name_lig, force_f, Arguments, amine, output_info, 0,
                          1);
    }
    show_init_pdb_progress("rec+lig", true);

    if (pro_in.m_mol.empty() && lig.m_mol.empty())
      return std::unexpected(
          std::string("Both pro_in and lig are empty after parsing Frame 0."));

    Arguments.setPdbname(name_lig.substr(name_lig.find("_")));
    output_info.setPdbName(Arguments);

    return {};
  }
}

int parse_input_pdb(arguments &Arguments, fstream &log, string name_pro,
                    string name_lig, string force_f, int input_name,
                    vector<string> &input_line, bool topo_2_is_open,
                    bool topo_is_open, Multi_MOL &pro_in, Multi_MOL &lig,
                    Multi_MOL &water_traj, groups_cal &Groups_from_index_file,
                    para &parameter, amino_acid &amine, info &output_info) {
  Multi_MOL membrane_traj;
  int flag_groups{helper_parse_groups(Groups_from_index_file, Arguments)};

  fstream test(name_pro, ios::in);
  if (!test.is_open()) {
    if ((Arguments.flag_1_pdb == true || Arguments.argc == 1) &&
        input_name > 0 && Arguments.flag_listfile == false &&
        Arguments.flag_pattern_input == false &&
        Arguments.flag_all_folders == false)
      return 1;

    if (Arguments.flag_1_pdb == false) {
      std::cout << "\033[1;31m[ERROR]\033[0m Cannot open input file: "
                << name_pro << "\n";
      TitanUI::log_kv(log, "FATAL ERROR",
                      "Cannot open input file: " + name_pro);
      return 1;
    } else if ((Arguments.flag_listfile == true ||
                Arguments.flag_all_folders == true) &&
               Arguments.flag_1_pdb == true && !input_line.empty() &&
               input_name < input_line.size()) {
      if (input_name == 0) {
        if (Arguments.flag_listfile == true) {
          std::cout << "\033[1;36m[TITAN]\033[0m \033[1;32m>> Batch "
                       "Mode:\033[0m Loading PDBs from list: "
                    << Arguments.list_filename << "\n";
          TitanUI::log_kv(log, "Batch Mode",
                          "List file: " + Arguments.list_filename);
        } else if (Arguments.flag_all_folders == true) {
          std::cout << "\033[1;36m[TITAN]\033[0m \033[1;32m>> Batch "
                       "Mode:\033[0m Loading PDBs from folder: "
                    << fs::current_path().string() << "\n";
          TitanUI::log_kv(log, "Batch Mode",
                          "Folder: " + fs::current_path().string());
        }
      }
      Arguments.pdb_name = input_line[input_name];
      if ((Arguments.pdb_name.length() > 4 &&
           Arguments.pdb_name.substr(Arguments.pdb_name.length() - 4) !=
               ".pdb") ||
          Arguments.pdb_name.length() <= 4)
        Arguments.pdb_name += ".pdb";

      // Silenced "Arguments.pdb_name..." console print
    } else if (Arguments.flag_pattern_input == true &&
               Arguments.flag_1_pdb == true) {
      if (Arguments.flag_range_input == true)
        Arguments.pdb_name = Arguments.input_pattern.substr(
                                 0, Arguments.input_pattern.find("*")) +
                             input_line[input_name - Arguments.b_input] +
                             Arguments.input_pattern.substr(
                                 Arguments.input_pattern.find("*") + 1);
      else
        Arguments.pdb_name = Arguments.input_pattern.substr(
                                 0, Arguments.input_pattern.find("*")) +
                             to_string(input_name) +
                             Arguments.input_pattern.substr(
                                 Arguments.input_pattern.find("*") + 1);

      Arguments.add_input_file_type();
    }

    if (Arguments.flag_1_pdb == true) {
      int flag_pre_proc{1};
      if (topo_2_is_open == true)
        flag_pre_proc = topo_is_open ? 2 : 4;
      if (Arguments.flag_use_common_topo == true)
        flag_pre_proc = 3;

      if ((Arguments.topo_name != "no" &&
           (Arguments.file_par == false || topo_is_open)) ||
          (Arguments.topo_name_2 != "no" &&
           (Arguments.file_par == false || topo_2_is_open))) {
        const int r_value_pdb{pre_proc_pdb(
            force_f, pro_in, lig, water_traj, membrane_traj,
            Groups_from_index_file,
            flag_pre_proc, Arguments, amine, output_info)};

        if (r_value_pdb == 1) {
          std::cout << "\033[1;31m[FATAL ERROR]\033[0m Selected index group is "
                       "not contained in the PDB file!\n";
          TitanUI::log_kv(log, "FATAL ERROR", "Index group missing in PDB");
          exit(EXIT_FAILURE);
        } else if (r_value_pdb == 2)
          return 2;
        else {
        }

        if (flag_pre_proc != 3) {
          if (flag_pre_proc != 4) {
            std::cout << "\033[1;36m[TITAN]\033[0m \033[1;32m>> Loading "
                         "Topology:\033[0m "
                      << Arguments.topo_name << "\n";
            TitanUI::log_kv(log, "Topology Loaded", Arguments.topo_name);
            lig.topo_lg(Arguments.topo_name, force_f, parameter);
          }

          if (flag_pre_proc == 2 || flag_pre_proc == 4) {
            std::cout << "\033[1;36m[TITAN]\033[0m \033[1;32m>> Loading "
                         "Topology 2:\033[0m "
                      << Arguments.topo_name_2 << "\n";
            TitanUI::log_kv(log, "Topology 2 Loaded", Arguments.topo_name_2);
            pro_in.topo_lg(Arguments.topo_name_2, force_f, parameter);
          }
        } else {
          vector<topo_info_temp> info_topo;
          std::cout << "\033[1;36m[TITAN]\033[0m \033[1;35m>> Loading Common "
                       "Topology:\033[0m "
                    << Arguments.topo_name << "\n";
          TitanUI::log_kv(log, "Common Topo Loaded", Arguments.topo_name);
          int topo_parse_result_1 = pro_in.topo_protein(
              Arguments.topo_name_2, force_f, parameter, info_topo);

          int topo_parse_result_2 = 0;
          const bool need_lig_topology =
              Arguments.needs_mol2_coordinates() && !lig.info.empty();
          if (need_lig_topology) {
            std::cout
                << "\033[1;36m[TITAN]\033[0m \033[1;35m>> Loading Common "
                   "Topology 2:\033[0m "
                << Arguments.topo_name << "\n";
            TitanUI::log_kv(log, "Common Topo 2 Loaded", Arguments.topo_name);
            topo_parse_result_2 = lig.topo_protein(
                Arguments.topo_name, force_f, parameter, info_topo);
          }

          if (topo_parse_result_1 == 1 ||
              (need_lig_topology && topo_parse_result_2 == 1)) {
            std::cout << "\033[1;31m[FATAL ERROR]\033[0m Cannot parse topology "
                         "files ("
                      << Arguments.topo_name << " and " << Arguments.topo_name_2
                      << ")\n";
            TitanUI::log_kv(log, "FATAL ERROR", "Cannot parse topology files");
            return 1;
          }
        }
      } else {
        const int r_value_pdb{pre_proc_pdb(
            force_f, pro_in, lig, water_traj, membrane_traj,
            Groups_from_index_file, 0, Arguments, amine, output_info)};

        if (r_value_pdb == 1) {
          std::cout << "\033[1;31m[FATAL ERROR]\033[0m Selected index group is "
                       "not contained in the PDB file!\n";
          TitanUI::log_kv(log, "FATAL ERROR", "Index group missing in PDB");
          exit(EXIT_FAILURE);
        } else if (r_value_pdb == 2)
          return 1;
        else {
        }
      }

    } else
      return 1;
  } else {
    fstream test_2(name_lig, ios::in);
    if (!test_2.is_open()) {
      cout << "Cannot open " << name_lig << " file!\n";
    }
    test_2.close();
    test.close();
    if (topo_is_open == true && topo_2_is_open == false &&
        pro_in.m_mol.empty() && lig.m_mol.empty()) {
      pro_in.take_coordinate(name_pro, force_f, Arguments, amine, output_info,
                             0);
      lig.take_topo_lg(name_lig, Arguments.topo_name, Arguments, force_f,
                       parameter, amine, output_info, Arguments.mol_2_type);
    } else if (topo_is_open == true && topo_2_is_open == true &&
               pro_in.m_mol.empty() && lig.m_mol.empty()) {
      lig.take_topo_lg(name_lig, Arguments.topo_name, Arguments, force_f,
                       parameter, amine, output_info, Arguments.mol_2_type);
      pro_in.take_topo_lg(name_pro, Arguments.topo_name_2, Arguments, force_f,
                          parameter, amine, output_info, Arguments.mol_1_type);
    } else if (topo_is_open == false && topo_2_is_open == true &&
               pro_in.m_mol.empty() && lig.m_mol.empty()) {
      pro_in.take_topo_lg(name_pro, Arguments.topo_name_2, Arguments, force_f,
                          parameter, amine, output_info, Arguments.mol_1_type);
      lig.take_coordinate(name_lig, force_f, Arguments, amine, output_info, 0);
    } else {
      if (pro_in.m_mol.empty() && lig.m_mol.empty()) {
        pro_in.take_coordinate(name_pro, force_f, Arguments, amine, output_info,
                               0);
        lig.take_coordinate(name_lig, force_f, Arguments, amine, output_info,
                            0);
      }
    }
    if (pro_in.m_mol.empty() && lig.m_mol.empty())
      return 1;
    Arguments.setPdbname(name_lig.substr(name_lig.find("_")));
    output_info.setPdbName(Arguments);
  }
  return {};
}
