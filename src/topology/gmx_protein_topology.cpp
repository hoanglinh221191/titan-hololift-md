#include "Multi_MOL.h"
#include "functions.h"
#include "topology/include_utils.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

// One [ moleculetype ] block: its name and the rows of its [ atoms ].
struct MoleculetypeBlock {
  std::string name;
  std::vector<topo_info_temp> atoms;
};

std::string section_name(const std::string &line) {
  std::string name;
  for (const char c : line) {
    if (c == '[' || c == ']' || c == ' ' || c == '\t' || c == '\r')
      continue;
    name.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return name;
}

// Every [ moleculetype ] of every file, in file order, with the rows of its
// [ atoms ]. A row is "nr type resnr residue atom cgnr charge [mass ...]"; as in
// GROMACS a missing mass is the mass of the atom type ([ atomtypes ] of any
// file, columns counted from the end: ... mass charge ptype v w). Comment text
// after ';' and preprocessor lines are ignored; a file may hold any number of
// blocks and the name is the first data row after the header.
std::vector<MoleculetypeBlock>
read_moleculetype_blocks(const std::vector<std::string> &files) {
  std::unordered_map<std::string, double> type_mass;
  for (const auto &filename : files) {
    std::fstream input(filename, std::ios::in);
    std::string section;
    for (std::string line{}; std::getline(input, line);) {
      line = trim_topology_line(std::move(line));
      if (line.empty() || line[0] == '#')
        continue;
      if (line.front() == '[') {
        section = section_name(line);
        continue;
      }
      if (section != "atomtypes")
        continue;
      std::istringstream iss(line);
      std::vector<std::string> cols;
      for (std::string token; iss >> token;)
        cols.push_back(token);
      const std::size_t n = cols.size();
      if (n < 6 || (cols[n - 3] != "A" && cols[n - 3] != "V" &&
                    cols[n - 3] != "D" && cols[n - 3] != "S"))
        continue;
      try {
        type_mass.emplace(cols[0], std::stod(cols[n - 5]));
      } catch (const std::exception &) {
      }
    }
  }

  std::vector<MoleculetypeBlock> blocks;
  for (const auto &filename : files) {
    std::fstream input(filename, std::ios::in);
    std::string section;
    bool awaiting_name = false;
    MoleculetypeBlock *current = nullptr;
    for (std::string line{}; std::getline(input, line);) {
      line = trim_topology_line(std::move(line));
      if (line.empty() || line[0] == '#')
        continue;
      if (line.front() == '[') {
        section = section_name(line);
        if (section == "moleculetype") {
          awaiting_name = true;
          current = nullptr;
        }
        continue;
      }
      if (section == "moleculetype" && awaiting_name) {
        std::istringstream iss(line);
        MoleculetypeBlock block;
        iss >> block.name;
        blocks.push_back(std::move(block));
        current = &blocks.back();
        awaiting_name = false;
        continue;
      }
      if (section != "atoms" || current == nullptr)
        continue;
      std::istringstream iss(line);
      int nr{}, resnr{};
      std::string type, residue, atom, cgnr;
      double charge{};
      if (!(iss >> nr >> type >> resnr >> residue >> atom >> cgnr >> charge))
        continue;
      double mass{};
      if (!(iss >> mass)) {
        const auto found = type_mass.find(type);
        mass = found != type_mass.end() ? found->second : 0.0;
      }
      current->atoms.emplace_back(atom, type, residue, charge, mass, resnr);
    }
  }
  return blocks;
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
    const std::string stripped = trim_topology_line(temp);
    if (!stripped.empty() && stripped.front() == '[') {
      main_content = section_name(stripped) == "molecules" ? "molecules" : "";
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

  for (auto &filename : content_input) {
    string molname{};
    fstream input(filename, ios::in);
    for (string temp{}; getline(input, temp);) {
      process_atomtypes(input, temp, parameters, std::string(force_f), molname);
    }
    input.close();

    // ENE-11 (21/10/2026): pair-specific LJ parameters are read for every
    // force field (AMBER/OPLS NBFIX and CUFIX included), not only CHARMM and
    // GROMOS. MARTINI builds its own pair matrix in parse_martini_topology.
    if (std::string(force_f).find("MARTINI") == std::string::npos) {
      collect_nonbond_params(filename, parameters);
    }

  }

  // The system's atoms in [ molecules ] order: every moleculetype block of
  // every included file, copies and repeated names expanded. (Until Nov 2026
  // this read one block per file, in include order, with the copy count of the
  // first [ molecules ] entry of that name, and only [ atoms ] rows with a mass
  // column; MARTINI and multi-block topologies came out in the wrong order.)
  // The system is a list of runs (block, copies); a source atom ID is located
  // by its run, so an index selection never expands the whole system (solvent
  // included) in memory.
  const auto blocks = read_moleculetype_blocks(content_input);
  struct MoleculeRun {
    std::size_t begin;
    const MoleculetypeBlock *block;
    std::size_t copies;
  };
  std::vector<MoleculeRun> runs;
  std::size_t system_atoms = 0;
  auto add_run = [&](const MoleculetypeBlock &block, std::size_t copies) {
    if (block.atoms.empty() || copies == 0)
      return;
    runs.push_back({system_atoms, &block, copies});
    system_atoms += block.atoms.size() * copies;
  };
  if (system_molecules.empty()) {
    for (const auto &block : blocks)
      add_run(block, 1);
  } else {
    for (const auto &[name, count] : system_molecules) {
      const auto block = std::find_if(
          blocks.begin(), blocks.end(),
          [&](const MoleculetypeBlock &b) { return b.name == name; });
      if (block == blocks.end()) {
        std::cerr << "[WARNING] [ molecules ] lists '" << name
                  << "', which no included [ moleculetype ] of " << input_topo
                  << " defines.\n";
        continue;
      }
      add_run(*block, count > 0 ? static_cast<std::size_t>(count) : 0);
    }
  }
  if (use_direct_subset) {
    for (size_t idx = 0; idx < info.size(); ++idx) {
      const size_t source = info[idx].get_original_atom_id();
      if (source >= system_atoms)
        continue;
      const auto run = std::prev(std::upper_bound(
          runs.begin(), runs.end(), source,
          [](std::size_t id, const MoleculeRun &r) { return id < r.begin; }));
      const auto &atoms = run->block->atoms;
      subset_topo[idx] = atoms[(source - run->begin) % atoms.size()];
      subset_filled[idx] = 1;
    }
  } else {
    info_topo_total_temp.reserve(system_atoms);
    for (const auto &run : runs)
      for (std::size_t c = 0; c < run.copies; ++c)
        info_topo_total_temp.insert(info_topo_total_temp.end(),
                                    run.block->atoms.begin(),
                                    run.block->atoms.end());
  }
  if (!use_direct_subset && info_topo_total_temp.size() != info.size()) {
    // A structure of one moleculetype (one copy) without source atom IDs.
    for (const auto &block : blocks) {
      if (!block.atoms.empty() && block.atoms.size() == info.size()) {
        info_topo = block.atoms;
        flag_in = true;
        for (size_t a = 0; a < info.size(); ++a)
          info[a].set_original_atom_id(a);
        break;
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

  // ENE-12: 2 = names disagree beyond the tolerance; the caller stops.
  // MARTINI topologies are assigned again by parse_martini_topology, so a
  // mismatch of this generic pass is only reported there.
  const bool martini = std::string(force_f).find("MARTINI") != std::string::npos;
  if (this->post_process_atom_from_topo(info_topo, !martini) == 2)
    return 1;
  return 0;
}
