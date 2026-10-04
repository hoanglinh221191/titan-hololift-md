#include "lj_topology.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

std::string trim_copy(std::string text) {
  size_t begin = 0;
  while (begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[begin])))
    ++begin;

  size_t end = text.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(text[end - 1])))
    --end;

  return text.substr(begin, end - begin);
}

std::string strip_comment(std::string text) {
  const size_t semi_pos = text.find(';');
  if (semi_pos != std::string::npos)
    text.erase(semi_pos);
  return trim_copy(std::move(text));
}

std::string parse_section_name(const std::string &line) {
  const size_t lbracket = line.find('[');
  const size_t rbracket = line.find(']');
  if (lbracket == std::string::npos || rbracket == std::string::npos ||
      rbracket <= lbracket + 1)
    return {};
  return trim_copy(line.substr(lbracket + 1, rbracket - lbracket - 1));
}

void flush_moleculetype(const std::string &name, size_t atom_count,
                        std::unordered_map<std::string, size_t> &atoms_per_mol,
                        std::vector<std::string> &mol_order) {
  if (name.empty() || atom_count == 0)
    return;
  atoms_per_mol[name] = atom_count;
  if (std::find(mol_order.begin(), mol_order.end(), name) == mol_order.end())
    mol_order.emplace_back(name);
}

} // namespace

std::expected<LjTopologySummary, std::string>
parse_lj_topology_summary(const std::string &topo_path) {
  std::ifstream input(topo_path);
  if (!input.is_open())
    return std::unexpected("LJ topology parser cannot open file: " + topo_path);

  enum class Section { None, Moleculetype, Atoms, Molecules };

  Section section = Section::None;
  std::unordered_map<std::string, size_t> atoms_per_molecule;
  std::vector<std::pair<std::string, size_t>> molecule_counts;
  std::vector<std::string> molecule_order;

  std::string current_moleculetype;
  size_t current_atom_count = 0;

  for (std::string raw_line; std::getline(input, raw_line);) {
    std::string line = strip_comment(std::move(raw_line));
    if (line.empty())
      continue;

    if (!line.empty() && line.front() == '[') {
      flush_moleculetype(current_moleculetype, current_atom_count,
                         atoms_per_molecule, molecule_order);

      const std::string section_name = parse_section_name(line);
      if (section_name == "moleculetype") {
        section = Section::Moleculetype;
        current_moleculetype.clear();
        current_atom_count = 0;
      } else if (section_name == "atoms") {
        section = Section::Atoms;
      } else if (section_name == "molecules") {
        section = Section::Molecules;
      } else {
        section = Section::None;
      }
      continue;
    }

    std::istringstream iss(line);
    if (section == Section::Moleculetype) {
      if (current_moleculetype.empty())
        iss >> current_moleculetype;
      continue;
    }

    if (section == Section::Atoms) {
      std::string atom_nr_token;
      if (iss >> atom_nr_token)
        ++current_atom_count;
      continue;
    }

    if (section == Section::Molecules) {
      std::string molecule_name;
      size_t molecule_count = 0;
      if (iss >> molecule_name >> molecule_count)
        molecule_counts.emplace_back(molecule_name, molecule_count);
    }
  }

  flush_moleculetype(current_moleculetype, current_atom_count,
                     atoms_per_molecule, molecule_order);

  LjTopologySummary summary;
  if (!molecule_counts.empty()) {
    summary.primary_molecule = molecule_counts.front().first;
    summary.molecule_count = molecule_counts.front().second;

    for (const auto &[molecule_name, molecule_count] : molecule_counts) {
      const auto it = atoms_per_molecule.find(molecule_name);
      if (it == atoms_per_molecule.end()) {
        return std::unexpected("LJ topology parser cannot resolve molecule '" +
                               molecule_name + "' in [ molecules ] from " +
                               topo_path);
      }
      if (summary.primary_molecule == molecule_name)
        summary.atoms_per_molecule = it->second;
      summary.total_atoms += it->second * molecule_count;
    }
  } else if (!molecule_order.empty()) {
    summary.primary_molecule = molecule_order.front();
    summary.atoms_per_molecule = atoms_per_molecule[molecule_order.front()];
    summary.molecule_count = 1;
    summary.total_atoms = summary.atoms_per_molecule;
  }

  if (summary.total_atoms == 0)
    return std::unexpected("LJ topology parser found no atoms in topology: " +
                           topo_path);

  return summary;
}
