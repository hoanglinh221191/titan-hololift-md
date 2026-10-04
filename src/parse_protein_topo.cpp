#include "Multi_MOL.h"
#include "functions.h"
#include "topology_include_utils.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
using std::string_view;
#include <vector>

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

void collect_topology_file_list(const std::string &current_file,
                                std::vector<std::string> &out_files,
                                std::vector<std::string> &include_stack) {
  namespace fs = std::filesystem;

  const std::string normalized =
      fs::path(current_file).lexically_normal().string();
  if (std::find(include_stack.begin(), include_stack.end(), normalized) !=
      include_stack.end()) {
    return;
  }

  include_stack.emplace_back(normalized);
  out_files.emplace_back(normalized);

  std::fstream input(normalized, std::ios::in);
  if (input.is_open()) {
    for (std::string line{}; std::getline(input, line);) {
      const auto include_target = TitanTopo::extract_include_target(line);
      if (!include_target)
        continue;

      std::fstream included_stream;
      std::string resolved_path;
      if (open_topology_include(normalized, *include_target, included_stream,
                                resolved_path)) {
        included_stream.close();
        collect_topology_file_list(resolved_path, out_files, include_stack);
      }
    }
  }

  include_stack.pop_back();
}

std::string trim_topology_line(std::string line) {
  const auto comment_pos = line.find(';');
  if (comment_pos != std::string::npos)
    line = line.substr(0, comment_pos);

  const auto first = line.find_first_not_of(" \t\r");
  if (first == std::string::npos)
    return {};

  const auto last = line.find_last_not_of(" \t\r");
  return line.substr(first, last - first + 1);
}

void collect_nonbond_params(const std::string &filename, para &parameters) {
  std::fstream input(filename, std::ios::in);
  if (!input.is_open())
    return;

  bool in_nonbond_params = false;
  for (std::string line{}; std::getline(input, line);) {
    line = trim_topology_line(std::move(line));
    if (line.empty() || line[0] == '#')
      continue;

    if (line.front() == '[') {
      in_nonbond_params = (line.find("nonbond_params") != std::string::npos);
      continue;
    }

    if (!in_nonbond_params)
      continue;

    std::istringstream iss(line);
    std::string atom_i, atom_j;
    int funct = 0;
    double sigma = 0.0, epsilon = 0.0;
    if (iss >> atom_i >> atom_j >> funct >> sigma >> epsilon) {
      parameters.add_nonbond_override(std::move(atom_i), std::move(atom_j),
                                      sigma, epsilon);
    }
  }
}

} // namespace

int Multi_MOL::topo_protein(std::string_view input_topo,
                            std::string_view force_f, para &parameters,
                            const std::vector<topo_info_temp> &extra_bonds) {
  if (input_topo.length() > 5 &&
      input_topo.substr(input_topo.length() - 4) != ".top" &&
      input_topo.substr(input_topo.length() - 4) != ".itp") {
    cout << "The file type of topo " << input_topo
         << " is not supported. This software supports only .top and .itp\n";
    return 1;
  }
  //<< info.size() << "\n";

  fstream input(std::string(input_topo), ios::in);
  if (!input.is_open()) {
    cout << "Cannot open topo file " << input_topo << "\n";
    return 1;
  } else {
    cout << "Parsing topo from file " << input_topo << "\n";
  }

  vector<string> content_input;
  std::vector<std::pair<std::string, int>> system_molecules;
  std::vector<std::string> include_stack;
  collect_topology_file_list(std::string(input_topo), content_input,
                             include_stack);
  string main_content{};

  for (string temp{}; getline(input, temp);) {
    if (temp.find("[ molecules ]") != string::npos) {
      main_content = "molecules";
    } else if (main_content == "molecules" && temp.length() > 2 &&
               temp[0] != ';' && temp[0] != '#') {
      std::istringstream iss(temp);
      std::string m_name;
      int m_count;
      if (iss >> m_name >> m_count) {
        system_molecules.emplace_back(m_name, m_count);
      }
    }

    if (temp.find(";") != string::npos)
      temp = temp.substr(0, temp.find(";"));
    if (temp.length() < 1)
      continue;
  }

  bool flag_in{false};
  vector<topo_info_temp> info_topo_total_temp;
  vector<topo_info_temp> info_topo;
  size_t topology_atom_cursor = 0;

  std::unordered_map<size_t, size_t> selected_atom_pos;
  selected_atom_pos.reserve(info.size());
  const bool any_source_ids =
      std::any_of(info.begin(), info.end(), [](const mol_info &atom_info) {
        return atom_info.has_original_atom_id();
      });
  bool use_direct_subset = !info.empty() &&
                           std::all_of(info.begin(), info.end(),
                                       [](const mol_info &atom_info) {
                                         return atom_info.has_original_atom_id();
                                       });
  if (any_source_ids && !use_direct_subset) {
    std::cerr << "[WARNING] Selection has partially populated source atom IDs "
              << "for " << input_topo << ".\n";
  }
  for (size_t idx = 0; idx < info.size() && use_direct_subset; ++idx) {
    if (!selected_atom_pos.emplace(info[idx].get_original_atom_id(), idx)
             .second) {
      std::cerr << "[WARNING] Duplicate source atom ID in selection for "
                << input_topo << ".\n";
      use_direct_subset = false;
      selected_atom_pos.clear();
    }
  }
  vector<topo_info_temp> subset_topo;
  vector<unsigned char> subset_filled;
  if (use_direct_subset) {
    subset_topo.resize(info.size());
    subset_filled.assign(info.size(), 0);
  }

  auto append_selected_copies = [&](const vector<topo_info_temp> &atoms,
                                    int num_copies) {
    for (int c = 0; c < num_copies; ++c) {
      for (size_t a = 0; a < atoms.size(); ++a, ++topology_atom_cursor) {
        if (!use_direct_subset)
          continue;
        const auto it = selected_atom_pos.find(topology_atom_cursor);
        if (it == selected_atom_pos.end())
          continue;
        subset_topo[it->second] = atoms[a];
        subset_filled[it->second] = 1;
      }
    }
  };

  for (auto &filename : content_input) {
    // cout << filename << "\n";
    string content{}, molname{};
    fstream input(filename, ios::in);
    vector<string> data;
    for (string temp{}; getline(input, temp);) {
      /*if (temp.find(";") != string::npos)
              temp = temp.substr(0, temp.find(";"));
      if (temp.length() < 1)
              continue;*/
      data.emplace_back(temp);

      if (temp.find("[ moleculetype ]") != string::npos) {
        content = "moleculetype";
      }

      if (content == "moleculetype" && temp.find("[ atoms ]") != string::npos) {
        for (int j = (int)data.size() - 1; j >= 0; --j) {
          if (data[j].find("; name	nrexcl") != string::npos &&
              j + 1 < (int)data.size()) {
            std::istringstream iss(data[j + 1]);
            iss >> molname;
            break;
          }
        }
      }

      if (temp.find("[ atomtypes ]") != string::npos) {
        content = "atomtypes";
      }

      if (temp.find("[ molecules ]") != string::npos) {
        content = "molecules";
      } else if (content == "molecules" && temp.length() > 2 &&
                 temp[0] != ';' && temp[0] != '#') {
        // BUG 4 FIX: Removed 'if (filename == input_topo)' since
        // system_molecules was already parsed correctly in the first pass and
        // this guard was always false.
      }
    }

    input.close();
    input.open(filename, ios::in);

    for (string temp{}; getline(input, temp);) {
      process_atomtypes(input, temp, parameters, std::string(force_f), molname);
    }
    input.close();

    if (std::string(force_f).find("CHARMM") != std::string::npos ||
        std::string(force_f).find("GROMOS") != std::string::npos) {
      collect_nonbond_params(filename, parameters);
    }

    int atomnum{};
    input.open(filename, ios::in);
    // BUG 3 FIX: Removed hardcoded molname = "Protein";
    // process_atom_from_topo needs actual names. We let process_moleculetype
    // set it later. If we passed molname as it was found in the first include
    // pass, we're good.
    vector<topo_info_temp> info_topo_temp;
    for (string temp{}; getline(input, temp);) {
      if (temp.find("moleculetype") != string::npos) {
        process_moleculetype(input, molname, temp);
      }
      this->process_atom_from_topo(input, temp, molname, atomnum,
                                   info_topo_temp);
    }
    // Check if this molecule is in system_molecules and requires multiple
    // copies
    int num_copies = 1;
    for (const auto &smol : system_molecules) {
      if (smol.first == molname) {
        num_copies = smol.second;
        break;
      }
    }

    if (use_direct_subset) {
      append_selected_copies(info_topo_temp, num_copies);
    } else if (info_topo_temp.size() == info.size()) {
      info_topo = info_topo_temp;
      flag_in = true;
      for (size_t a = 0; a < info.size(); ++a) {
        info[a].set_original_atom_id(a);
      }
    } else {
      for (auto &i : info_topo_temp)
        info_topo_total_temp.emplace_back(i);
      if (num_copies > 1 && !info_topo_temp.empty()) {
        for (int c = 1; c < num_copies; ++c) {
          for (auto &i : info_topo_temp) {
            info_topo_total_temp.emplace_back(i);
          }
        }
      }
    }
  }

  if (use_direct_subset) {
    const bool all_subset_atoms_found =
        std::all_of(subset_filled.begin(), subset_filled.end(),
                    [](unsigned char filled) { return filled != 0; });
    if (!all_subset_atoms_found) {
      std::cerr << "[WARNING] Direct topology subset parse missed some atoms for "
                << input_topo << ". Falling back to legacy expanded topology "
                << "matching may be required.\n";
    }
    info_topo = std::move(subset_topo);
    flag_in = true;
  } else if (flag_in == false) {
    info_topo = info_topo_total_temp;
  }

  // A full unselected system may legitimately have no source provenance. Only
  // synthesize sequential IDs in that exact-size case; never overwrite IDs
  // carried from an index-selected reference structure.
  if (!any_source_ids && info_topo.size() == info.size()) {
    for (size_t a = 0; a < info.size(); ++a) {
      info[a].set_original_atom_id(a);
    }
  }

  attach_gromacs_mol_topology_graph(*this, input_topo);

  this->post_process_atom_from_topo(info_topo);
  // cout << "info topo size: " << info_topo.size() << '\n';
  // cout << "parameter size: " << parameters.base.size() << '\n';
  return 0;
}
