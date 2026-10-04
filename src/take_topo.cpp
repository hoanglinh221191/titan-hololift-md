#include "Multi_MOL.h"
#include "arguments.h"
#include "functions.h"
#include "info.h"
#include "miscellaneous.h"
#include "topology_include_utils.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
using std::string_view;
#include <vector>

using namespace std;
using index_type = class index;

namespace {

bool open_topology_include(std::string_view current_file,
                           std::string_view include_target,
                           std::fstream &stream,
                           std::string &resolved_path) {
  for (const auto &candidate :
       TitanTopo::build_include_candidates(current_file, include_target)) {
    stream.open(candidate.string(), std::ios::in);
    if (stream.is_open()) {
      resolved_path = candidate.string();
      return true;
    }
    stream.clear();
  }
  return false;
}

} // namespace

void Multi_MOL::take_topo_lg(std::string &input_name, std::string &input_topo,
                             arguments &Arguments, std::string_view force_f,
                             para &parameter, const amino_acid &amine,
                             ::info &output_info, std::string_view type) {
  parse_pdb(input_name, force_f, Arguments, output_info);
  wrapper_impl_post_proc(Arguments, type, amine);
  gen_vector_parameters();
  topo_lg(input_topo, force_f, parameter);
}

void Multi_MOL::post_process_coordinates_lig() {
  mol_type = "ligand";
  index_type index_temp;
  size_t index_i{};
  index_temp.set_chain(info[0].get_chain());
  Index.emplace_back(index_temp);
  for (size_t i{}; i < m_mol[0].size(); ++i) {
    Index[index_i].add_base(i + 1u, info[i].get_resnr(), m_mol[0].pos.x[i],
                            m_mol[0].pos.y[i], m_mol[0].pos.z[i],
                            info[i].get_res_name());
  }

  Index.emplace_back(index_type(std::string("NULL_S"), m_mol[0].size() + 1));
}

void Multi_MOL::topo_lg(std::string_view input_topo, std::string_view force_f,
                        para &parameter) {
  string lig_name{"LIG"};
  if (input_topo.length() >= 5) {
    fstream input_topo_protein(std::string(input_topo), ios::in);

    if (!input_topo_protein.is_open()) {
      //<< endl;
      // return; // Or continue to let it fail? Better to warn.
    } else {
      //<< endl;
    }

    if ((input_topo.substr(input_topo.length() - 4) == ".top" ||
         input_topo.substr(input_topo.length() - 4) == ".itp")) {
      vector<topo_info_temp> info_topo;

      ofstream debug_log("debug_topo_lg.txt", ios::out);
      //<< input_topo << endl;

      helper_topo_lg_gromacs_format(input_topo, force_f, parameter,
                                    input_topo_protein, lig_name, info_topo);
      //<< info_topo.size() << ". Calling post_process.\n";

      this->post_process_atom_from_topo(info_topo);
      attach_gromacs_mol_topology_graph(*this, input_topo);
    } else if (input_topo.substr(input_topo.length() - 6) == ".parm7") {
      helper_topo_lg_parm_format(input_topo, force_f, parameter,
                                 input_topo_protein, lig_name);
    } else {
      cout << "The file type of ligand topo file (" << input_topo
           << ") is not supported, this software supports only .top, .itp and "
              ".parm7!\n";
      return;
    }
  } else {
  }
}

int Multi_MOL::helper_topo_lg_gromacs_format(
    std::string_view input_topo, std::string_view force_f, para &parameter,
    std::fstream &input_topo_f, std::string &lig_name,
    std::vector<topo_info_temp> &info_topo) {
  ofstream debug_log("debug_helper.txt", ios::app);
  //<< input_topo << endl;
  int i{};
  // Info_topo passed by reference now

  std::vector<std::pair<std::string, int>> system_molecules;

  // First pass: extract [ molecules ] and all lines for second pass processing
  string content{};
  for (string protein_1{}; getline(input_topo_f, protein_1);) {
    if (protein_1.find("[ molecules ]") != string::npos) {
      content = "molecules";
    } else if (content == "molecules" && protein_1.length() > 2 &&
               protein_1[0] != ';' && protein_1[0] != '#') {
      std::istringstream iss(protein_1);
      std::string m_name;
      int m_count;
      if (iss >> m_name >> m_count) {
        system_molecules.emplace_back(m_name, m_count);
      }
    }
  }

  // Rewind stream for second pass
  input_topo_f.clear();
  input_topo_f.seekg(0);

  // Second pass: actual parsing
  for (string protein_1{}; getline(input_topo_f, protein_1);) {
    if (const auto include_target =
            TitanTopo::extract_include_target(protein_1)) {
      fstream included_stream;
      string resolved_include_path;
      if (open_topology_include(input_topo, *include_target, included_stream,
                                resolved_include_path)) {
        helper_topo_lg_gromacs_format(resolved_include_path, force_f, parameter,
                                      included_stream, lig_name, info_topo);
        included_stream.close();
      }
      continue;
    }

    process_atomtypes(input_topo_f, protein_1, parameter, std::string(force_f),
                      lig_name);

    if (protein_1.find("moleculetype") != string::npos) {
      process_moleculetype(input_topo_f, lig_name, protein_1);
      // If we found a moleculetype, lig_name is updated.
    }

    size_t prev_size = info_topo.size();
    this->process_atom_from_topo(input_topo_f, protein_1, lig_name, i,
                                 info_topo);
    size_t curr_size = info_topo.size();

    // If process_atom_from_topo actually found [ atoms ] and returned, meaning
    // we parsed a moleculetype block
    int num_copies = 1;
    if (curr_size > prev_size) {
      for (const auto &smol : system_molecules) {
        if (smol.first == lig_name) {
          num_copies = smol.second;
          break;
        }
      }
    }

    if (num_copies > 1) {
      size_t atoms_per_mol = curr_size - prev_size;
      for (int c = 1; c < num_copies; ++c) {
        for (size_t a = 0; a < atoms_per_mol; ++a) {
          info_topo.emplace_back(info_topo[prev_size + a]);
        }
      }
    }
  }

  // Post process is called by the caller (topo_lg) to avoid multiple calls
  // during recursion
  //<< ". Atoms read so far: " << info_topo.size() << endl;
  return 0;
}

int Multi_MOL::helper_topo_lg_parm_format(std::string_view input_topo,
                                          std::string_view force_f,
                                          para &parameter,
                                          std::fstream &input_topo_f,
                                          std::string &lig_name) {
  vector<string> topol_ligand_datafile, atom_names, atom_types;
  vector<double> atom_charges, atom_masses;

  size_t line_begincharge{}, line_endcharge{}, line_beginatoms{},
      line_endatoms{}, line_begintypes{}, line_endtypes{}, line_beginmasses{},
      line_endmasses{};

  for (string temp{}; getline(input_topo_f, temp);) {
    topol_ligand_datafile.emplace_back(temp);
  }

  for (size_t i{}; i < topol_ligand_datafile.size(); ++i) {
    if (topol_ligand_datafile[i].length() >= 11 &&
        topol_ligand_datafile[i].substr(0, 11) == "%FLAG TITLE" &&
        i < topol_ligand_datafile.size() - 2) {
      size_t find_space{topol_ligand_datafile[i + 2].find(" ")};
      lig_name = topol_ligand_datafile[i + 2].substr(0, find_space);
    }

    if (topol_ligand_datafile[i].length() >= 15 &&
        topol_ligand_datafile[i].substr(0, 15) == "%FLAG ATOM_NAME" &&
        i < topol_ligand_datafile.size() - 2) {
      line_beginatoms = i + 2;
    }

    if (topol_ligand_datafile[i].length() >= 12 &&
        topol_ligand_datafile[i].substr(0, 12) == "%FLAG CHARGE" &&
        i < topol_ligand_datafile.size() - 2) {
      line_begincharge = i + 2;
      line_endatoms = i;
    }

    if (topol_ligand_datafile[i].length() >= 19 &&
        topol_ligand_datafile[i].substr(0, 19) == "%FLAG ATOMIC_NUMBER") {
      line_endcharge = i;
    }

    if (topol_ligand_datafile[i].length() >= 10 &&
        topol_ligand_datafile[i].substr(0, 10) == "%FLAG MASS" &&
        i < topol_ligand_datafile.size() - 2) {
      line_beginmasses = i + 2;
    }

    if (topol_ligand_datafile[i].length() >= 21 &&
        topol_ligand_datafile[i].substr(0, 21) == "%FLAG ATOM_TYPE_INDEX") {
      line_endmasses = i;
    }

    if (topol_ligand_datafile[i].length() >= 21 &&
        topol_ligand_datafile[i].substr(0, 21) == "%FLAG AMBER_ATOM_TYPE" &&
        i < topol_ligand_datafile.size() - 2) {
      line_begintypes = i + 2;
    }

    if (topol_ligand_datafile[i].length() >= 31 &&
        topol_ligand_datafile[i].substr(0, 31) ==
            "%FLAG TREE_CHAIN_CLASSIFICATION") {
      line_endtypes = i;
    }
  }

  for (size_t i{line_beginatoms}; i < line_endatoms; ++i) {
    size_t length{topol_ligand_datafile[i].size()}, begin{};
    while (length > 0 && begin <= topol_ligand_datafile[i].length() - 4) {
      string name{topol_ligand_datafile[i].substr(begin, 4)};
      begin += 4;
      length -= 4;
      name.erase(remove(name.begin(), name.end(), ' '), name.end());
      if (name.length() > 0)
        atom_names.emplace_back(name);
    }
  }

  for (size_t i{line_begincharge}; i < line_endcharge; ++i) {
    size_t length{topol_ligand_datafile[i].size()}, begin{};
    while (length > 0 && begin <= topol_ligand_datafile[i].length() - 16) {
      atom_charges.emplace_back(
          stod(topol_ligand_datafile[i].substr(begin, 16)) /
          18.223); // *4.184 / (1389.35457839));
      begin += 16;
      length -= 16;
    }
  }

  for (size_t i{line_beginmasses}; i < line_endmasses; ++i) {
    size_t length{topol_ligand_datafile[i].size()}, begin{};
    while (length > 0 && begin <= topol_ligand_datafile[i].length() - 16) {
      atom_masses.emplace_back(
          stod(topol_ligand_datafile[i].substr(begin, 16)));
      begin += 16;
      length -= 16;
    }
  }

  for (size_t i{line_begintypes}; i < line_endtypes; ++i) {
    size_t length{topol_ligand_datafile[i].size()}, begin{};
    while (length > 0 && begin <= topol_ligand_datafile[i].length() - 4) {
      string type{topol_ligand_datafile[i].substr(begin, 4)};
      begin += 4;
      length -= 4;
      type.erase(remove(type.begin(), type.end(), ' '), type.end());
      if (!type.empty())
        atom_types.emplace_back(type);
    }
  }

  size_t min_size{atom_charges.size()};
  if (atom_charges.size() != atom_names.size() ||
      atom_names.size() != atom_types.size() ||
      atom_types.size() != atom_charges.size()) {

    size_t min{atom_charges.size() > atom_types.size() ? atom_types.size()
                                                       : atom_charges.size()};
    cout << "min " << min << '\n';
    min_size = atom_names.size() > min ? min : atom_names.size();
  }

  // cout << min_size <<" "<<atom_charges.size()<<"  "<<atom_types.size()<<"
  // "<<atom_names.size()  << '\n';

  double tot_charge{};
  for (size_t i{}; i < min_size; ++i) {
    // cout << "type: " << atom_types[i] << " charge: " << atom_charges[i] <<
    // '\n';
    tot_charge += atom_charges[i];
    info[i].set_atom_type(atom_types[i]);
    info[i].set_res_name(lig_name);
    info[i].set_atom(atom_names[i]);
    this->parameters[i].set_charge(atom_charges[i]);
    this->parameters[i].set_mass(atom_masses[i]);
  }
  cout << "\ntotal charge:" << tot_charge << '\n';
  return 0;
}

int process_moleculetype(fstream &input_topo_protein, string &lig_name,
                         string &protein_1) {
  if (protein_1.find("moleculetype") != string::npos) {
    while (getline(input_topo_protein, protein_1)) {
      if (protein_1.length() > 1 && protein_1.find("[ ") == string::npos) {
        if (protein_1[0] != ';') {
          istringstream is(protein_1);
          int nrexcl{};
          is >> lig_name >> nrexcl;
          return 0;
        } else
          continue;
      } else
        return 1;
    }
    return 0;
  } else
    return 1;
}
int process_atomtypes(fstream &input_topo_protein, string &protein_1,
                      para &parameters, string force_f, string &lig_name) {
  if (protein_1.find("atomtypes") != string::npos) {
    while (getline(input_topo_protein, protein_1)) {
      if (protein_1.length() <= 1 || protein_1.find(";") == 0)
        continue;
      if (protein_1.find(";") != string::npos) {
        protein_1 = protein_1.substr(0, protein_1.find(";"));
      }

      if (protein_1.length() > 10 && protein_1.find("[ ") == string::npos &&
          protein_1.find(" ]") == string::npos) {
        string s_t1{}, s_t2{}, s_t3{}, s_t4{}, s_t5{}, s_t6{}, s_t7{};
        istringstream i_temp(protein_1);
        i_temp >> s_t1 >> s_t2 >> s_t3 >> s_t4 >> s_t5 >> s_t6 >> s_t7;

        if (s_t3.empty() || s_t6.empty() || s_t7.empty())
          continue;

        try {
          double mass{1.0};
          if (stod(s_t3) > 0.1)
            mass = stod(s_t3);
          if (force_f.find("AMBER") != string::npos ||
              force_f.find("CHARMM") != string::npos) {
            if (std::fabs(stod(s_t7)) <= 0.0001 &&
                std::fabs(stod(s_t6)) <= 0.0001)
              parameters.base.emplace_back(stod(s_t6), stod(s_t7), mass, 0.0,
                                           s_t1);
            else
              parameters.base.emplace_back(
                  stod(s_t6), stod(s_t7), mass,
                  5.0 * pow(2.0, 1.0 / 6.0) * stod(s_t6), s_t1);
            parameters.register_atom_type(s_t1);
          } else {
            if (std::fabs(stod(s_t7)) <= 0.0001 &&
                std::fabs(stod(s_t6)) <= 0.0001)
              parameters.base.emplace_back(stod(s_t6), stod(s_t7), mass, 0.0,
                                           s_t1);
            else
              parameters.base.emplace_back(
                  stod(s_t6), stod(s_t7), mass,
                  10 * 0.5 *
                      pow((stod(s_t7) * 2 * pow(stod(s_t6), -1.0)),
                          (1.0 / 6.0)),
                  s_t1);
            parameters.register_atom_type(s_t1);
          }
        } catch (const std::exception &) {
          continue;
        }
      } else if (protein_1.find("moleculetype") != string::npos) {
        process_moleculetype(input_topo_protein, lig_name, protein_1);
        break;
      } else
        break;
    }
    return 0;
  }
  return 1;
}
int Multi_MOL::process_atom_from_topo(fstream &input_topo_protein,
                                      string &protein_1, string &lig_name,
                                      int &i,
                                      vector<topo_info_temp> &info_topo) {
  if (protein_1.find("[ atoms ]") != string::npos) {
    while (getline(input_topo_protein, protein_1)) {
      string line{protein_1};
      const size_t first_non_ws{line.find_first_not_of(" \t\r")};
      if (first_non_ws == string::npos)
        continue;

      line.erase(0, first_non_ws);

      if (line[0] == '[')
        break;

      if (line[0] == ';' || line[0] == '#')
        continue;

      if (const size_t comment_pos = line.find(';'); comment_pos != string::npos)
        line = line.substr(0, comment_pos);

      if (line.find_first_not_of(" \t\r") == string::npos)
        continue;

      // Every well-formed row inside [ atoms ] belongs to the active
      // moleculetype. Matching against lig_name breaks RNA/DNA because those
      // rows contain residue names (GUA/URA/...) instead of the parent name.
      istringstream topo_is(line);
      int i_temp{}, chrg{}, resnr{};
      string a_type{}, a_name{}, res_name{};
      double charge{}, mass{};
      if (!(topo_is >> i_temp >> a_type >> resnr >> res_name >> a_name >>
            chrg >> charge >> mass))
        continue;

      info_topo.emplace_back(a_name, a_type, res_name, charge, mass, resnr);
      ++i;
    }
  } else
    return 1;
  return 0;
}

int Multi_MOL::post_process_atom_from_topo(vector<topo_info_temp> &info_topo) {
  if (info_topo.empty()) {
    std::cerr << "[INFO] Standard topology parser: 0 atoms parsed from "
              << "topo file. If using MARTINI, this is expected.\n";
    return 1;
  }

  if (this->parameters.size() < info.size()) {
    this->parameters.resize(info.size());
  }

  // ASSIGN INTERNAL RESNUM to PDB atoms to allow unique identification
  // This separates "Display ResNr" (which can be duplicate/confusing) from
  // "Calculation ID". We do this purely based on PDB connectivity/changes.
  int internal_cnt = 0;
  if (!info.empty()) {
    info[0].set_internal_resnr(0);
  }
  for (size_t k = 1; k < info.size(); ++k) {
    if (info[k].get_resnr() != info[k - 1].get_resnr() ||
        info[k].get_chain() != info[k - 1].get_chain()) {
      internal_cnt++;
    }
    info[k].set_internal_resnr(internal_cnt);
  }

  // AUTO-SUBSET FOR COMMON TOPOLOGY SYSTEMS
  // If the parsed topology is larger than the extracted PDB group, we use the
  // original PDB atom IDs to pluck the exact corresponding topology atom,
  // bypassing string-matching.
  if (info_topo.size() > info.size() && !info.empty()) {
    const bool ids_were_set =
        std::all_of(info.begin(), info.end(), [](const mol_info &atom_info) {
          return atom_info.has_original_atom_id();
        });

    if (ids_were_set) {
      bool can_subset = true;
      for (size_t i = 0; i < info.size(); ++i) {
        if (info[i].get_original_atom_id() >= info_topo.size()) {
          can_subset = false;
          break;
        }
      }
      if (can_subset) {
        vector<topo_info_temp> subset_topo;
        subset_topo.reserve(info.size());
        for (size_t i = 0; i < info.size(); ++i) {
          subset_topo.push_back(info_topo[info[i].get_original_atom_id()]);
        }
        info_topo =
            std::move(subset_topo); // Now info.size() == info_topo.size()
      }
    }
  }

  // 1-1 MAPPING LOGIC (Common Topo)
  if (info.size() == info_topo.size()) {
    //<< "). Using sequential 1-1 parameter assignment.\n";
    int mismatch_count = 0;
    for (size_t i = 0; i < info.size(); ++i) {
      // Warning for sanity check
      if (info[i].get_res_name() != info_topo[i].res_name ||
          info[i].get_atom() != info_topo[i].atom_name) {
        if (mismatch_count < 10) { // Limit spam
          //<< " PDB: " << info[i].get_res_name() << ":"
          //<< info[i].get_atom() << " Topo: " << info_topo[i].res_name
          //<< ":" << info_topo[i].atom_name << endl;
        }
        mismatch_count++;
      }

      // Assign Parameters sequentially
      info[i].set_atom_type(info_topo[i].atom_type);
      this->parameters[i].set_charge(info_topo[i].atom_charge);
      this->parameters[i].set_mass(info_topo[i].atom_mass);
      // Do NOT overwrite info[i].resnr - keep PDB numbering for output.
    }
    if (mismatch_count > 0) {
      //<< endl;
    }
  } else {
    //<< info_topo.size() << "). Falling back to Strict Match Search.\n";
    // Fallback Search Logic
    int matched_count = 0;
    vector<bool> consumed(info_topo.size(), false);
    for (size_t i{}; i < info.size(); ++i) {
      int best_match_index = -1;
      for (size_t j = 0; j < info_topo.size(); ++j) {
        if (consumed[j])
          continue;

        bool names_match = (info[i].get_res_name() == info_topo[j].res_name);
        // Simple alias check
        if (!names_match) {
          string pdb_name = info[i].get_res_name();
          string topo_name = info_topo[j].res_name;
          if (pdb_name.length() == topo_name.length() + 1 &&
              (pdb_name[0] == 'N' || pdb_name[0] == 'C')) {
            if (pdb_name.substr(1) == topo_name)
              names_match = true;
          }
        }

        if (names_match && info[i].get_atom() == info_topo[j].atom_name) {
          best_match_index = j;
          break;
        }
      }

      if (best_match_index != -1) {
        info[i].set_atom_type(info_topo[best_match_index].atom_type);
        this->parameters[i].set_charge(info_topo[best_match_index].atom_charge);
        this->parameters[i].set_mass(info_topo[best_match_index].atom_mass);
        consumed[best_match_index] = true;
        matched_count++;
      } else {
        //<< ":" << info[i].get_atom() << ") not found in Topo!\n";
        this->parameters[i].set_charge(0.0); // Reset to 0
      }
    }
    //<< " atoms.\n";
  }

  return 0;
}
