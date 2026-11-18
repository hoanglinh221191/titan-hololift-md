#include "topology/mol_topology.h"

#include "Multi_MOL.h"
#include "topology/include_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

struct GromacsMoleculeType {
  std::string name;
  std::uint32_t atom_count = 0;
  std::vector<std::string> atom_types;
  std::vector<double> explicit_masses;
  std::vector<MolTopologyBond> bonds;
};

struct GromacsLennardJonesType {
  double first = 0.0;
  double second = 0.0;
};

struct GromacsMoleculeInstance {
  std::string name;
  std::uint32_t count = 0;
};

struct ExpandedGromacsTopology {
  std::uint32_t atom_count = 0;
  std::vector<MolTopologyBond> bonds;
  std::vector<MolTopologyMoleculeRange> molecule_ranges;
  std::vector<double> vdw_radii;
  std::vector<double> atomic_masses;
};

double gromacs_vdw_radius_angstrom(const GromacsLennardJonesType &lj,
                                   int combination_rule) {
  if (!std::isfinite(lj.first) || !std::isfinite(lj.second) ||
      lj.first < 0.0 || lj.second < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  // A zero self Lennard-Jones term is explicit force-field data, notably for
  // polar hydrogens in CHARMM. Preserve it as a zero steric radius instead of
  // treating it as missing and replacing it with an element-based radius.
  if (combination_rule == 1) {
    if (lj.first == 0.0 && lj.second == 0.0)
      return 0.0;
    if (lj.first == 0.0 || lj.second == 0.0)
      return std::numeric_limits<double>::quiet_NaN();
  } else if (lj.first == 0.0 || lj.second == 0.0) {
    return 0.0;
  }

  double minimum_distance_nm = 0.0;
  if (combination_rule == 1) {
    minimum_distance_nm = std::pow(2.0 * lj.second / lj.first, 1.0 / 6.0);
  } else {
    minimum_distance_nm = std::pow(2.0, 1.0 / 6.0) * lj.first;
  }
  const double radius_angstrom = 5.0 * minimum_distance_nm;
  return std::isfinite(radius_angstrom) && radius_angstrom > 0.0
             ? radius_angstrom
             : std::numeric_limits<double>::quiet_NaN();
}

std::string trim_topology_line(std::string line) {
  const size_t comment_pos = line.find(';');
  if (comment_pos != std::string::npos)
    line = line.substr(0, comment_pos);

  const size_t first = line.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};

  const size_t last = line.find_last_not_of(" \t\r\n");
  return line.substr(first, last - first + 1);
}

std::string lowercase_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string topology_section_name(const std::string &line) {
  const size_t open = line.find('[');
  const size_t close = line.find(']', open == std::string::npos ? 0 : open + 1);
  if (open == std::string::npos || close == std::string::npos || close <= open)
    return {};
  return lowercase_ascii(TitanTopo::trim_copy(
      std::string_view(line).substr(open + 1, close - open - 1)));
}

void collect_topology_file_list(const std::filesystem::path &current_file,
                                std::vector<std::filesystem::path> &out_files,
                                std::unordered_set<std::string> &visited,
                                bool &includes_complete) {
  namespace fs = std::filesystem;

  const fs::path normalized = current_file.lexically_normal();
  const std::string key = normalized.string();
  if (!visited.insert(key).second)
    return;

  out_files.emplace_back(normalized);
  std::fstream input(normalized, std::ios::in);
  if (!input.is_open()) {
    includes_complete = false;
    return;
  }

  for (std::string raw_line; std::getline(input, raw_line);) {
    const auto include_target = TitanTopo::extract_include_target(raw_line);
    if (!include_target)
      continue;
    bool resolved = false;
    for (const auto &candidate :
         TitanTopo::build_include_candidates(key, *include_target)) {
      std::fstream included(candidate, std::ios::in);
      if (!included.is_open())
        continue;
      included.close();
      resolved = true;
      collect_topology_file_list(candidate, out_files, visited, includes_complete);
      break;
    }
    if (!resolved) includes_complete = false;
  }
}

std::expected<ExpandedGromacsTopology, std::string>
parse_and_expand_gromacs_topology(std::string_view topology_path) {
  namespace fs = std::filesystem;

  const fs::path root_path(TitanTopo::trim_copy(topology_path));
  if (root_path.empty())
    return std::unexpected("empty topology path");

  std::fstream root(root_path, std::ios::in);
  if (!root.is_open())
    return std::unexpected("cannot open topology file: " + root_path.string());
  root.close();

  std::vector<fs::path> files;
  std::unordered_set<std::string> visited;
  bool includes_complete = true;
  collect_topology_file_list(root_path, files, visited, includes_complete);
  if (files.empty())
    return std::unexpected("no topology files collected");

  std::unordered_map<std::string, GromacsMoleculeType> molecule_types;
  std::unordered_map<std::string, GromacsLennardJonesType> atom_types;
  std::unordered_map<std::string, double> atom_type_masses;
  bool mass_binding_supported = includes_complete;
  std::unordered_map<std::string, GromacsLennardJonesType>
      self_nonbond_parameters;
  std::vector<std::string> molecule_type_order;
  std::vector<GromacsMoleculeInstance> molecules;
  int combination_rule = 2;

  for (const fs::path &filename : files) {
    std::fstream input(filename, std::ios::in);
    if (!input.is_open())
      continue;

    std::string section;
    GromacsMoleculeType *current_molecule = nullptr;
    bool expect_moleculetype_name = false;
    unsigned conditional_depth = 0;

    for (std::string raw_line; std::getline(input, raw_line);) {
      std::string line = trim_topology_line(std::move(raw_line));
      if (line.empty())
        continue;
      if (line[0] == '#') {
        if (line.starts_with("#if")) ++conditional_depth;
        else if (line.starts_with("#endif") && conditional_depth) --conditional_depth;
        continue;
      }

      if (line.front() == '[') {
        section = topology_section_name(line);
        if (conditional_depth && (section == "atoms" || section == "atomtypes" ||
                                  section == "moleculetype" || section == "molecules"))
          mass_binding_supported = false;
        expect_moleculetype_name = (section == "moleculetype");
        continue;
      }

      if (expect_moleculetype_name) {
        std::istringstream iss(line);
        std::string name;
        int nrexcl = 0;
        if (iss >> name >> nrexcl) {
          auto [it, inserted] = molecule_types.try_emplace(name);
          if (!inserted) mass_binding_supported = false;
          if (inserted) {
            it->second.name = name;
            molecule_type_order.emplace_back(name);
          }
          current_molecule = &it->second;
        }
        expect_moleculetype_name = false;
        continue;
      }

      if (section == "defaults") {
        std::istringstream iss(line);
        int nonbond_function = 0;
        int parsed_combination_rule = 0;
        if (iss >> nonbond_function >> parsed_combination_rule &&
            parsed_combination_rule >= 1 && parsed_combination_rule <= 3) {
          combination_rule = parsed_combination_rule;
        }
        continue;
      }

      if (section == "atomtypes") {
        if (conditional_depth) mass_binding_supported = false;
        std::istringstream iss(line);
        std::vector<std::string> fields;
        for (std::string field; iss >> field;)
          fields.push_back(std::move(field));
        if (fields.size() < 3)
          continue;
        try {
          const double first = std::stod(fields[fields.size() - 2]);
          const double second = std::stod(fields.back());
          atom_types[fields.front()] = {first, second};
          // The optional bond-type/atomic-number fields precede mass; the
          // trailing fields are always mass, charge, ptype, sigma/C6, epsilon/C12.
          if (fields.size() >= 6) {
            const double mass = std::stod(fields[fields.size() - 5]);
            const auto [it, inserted] = atom_type_masses.emplace(fields.front(), mass);
            if (!inserted && it->second != mass) mass_binding_supported = false;
          }
        } catch (const std::exception &) {
        }
        continue;
      }

      if (section == "nonbond_params") {
        std::istringstream iss(line);
        std::string lhs;
        std::string rhs;
        int function = 0;
        double first = 0.0;
        double second = 0.0;
        if (iss >> lhs >> rhs >> function >> first >> second && lhs == rhs)
          self_nonbond_parameters[lhs] = {first, second};
        continue;
      }

      if (section == "atoms") {
        if (conditional_depth) mass_binding_supported = false;
        if (current_molecule == nullptr)
          continue;
        std::istringstream iss(line);
        std::uint32_t atom_index = 0;
        std::string atom_type;
        if (iss >> atom_index >> atom_type && atom_index > 0) {
          current_molecule->atom_count =
              std::max(current_molecule->atom_count, atom_index);
          if (current_molecule->atom_types.size() < atom_index)
            current_molecule->atom_types.resize(atom_index);
          if (!current_molecule->atom_types[atom_index - 1].empty())
            mass_binding_supported = false;
          current_molecule->atom_types[atom_index - 1] = std::move(atom_type);
          current_molecule->explicit_masses.resize(current_molecule->atom_types.size(),
              std::numeric_limits<double>::quiet_NaN());
          std::string resnr, residue, atomname, cgnr, charge, mass_text;
          if (!(iss >> resnr >> residue >> atomname >> cgnr >> charge)) {
            mass_binding_supported = false;
          } else if (iss >> mass_text) {
            try {
              std::size_t consumed = 0;
              const double mass = std::stod(mass_text, &consumed);
              if (consumed != mass_text.size()) mass_binding_supported = false;
              current_molecule->explicit_masses[atom_index - 1] = mass;
              std::string type_b, charge_b, mass_b;
              if (iss >> type_b) {
                if (!(iss >> charge_b >> mass_b) || std::stod(mass_b) != mass)
                  mass_binding_supported = false;
              }
            } catch (const std::exception &) { mass_binding_supported = false; }
          }
        }
        continue;
      }

      // GROMACS [ constraints ] entries are covalent/topological links whose
      // distances are enforced rather than integrated as bond potentials.  For
      // periodic reconstruction they carry the same hard-connectivity meaning
      // as [ bonds ].  This is essential for Martini, where constrained side-
      // chain and nucleic-acid beads may otherwise appear as false singleton
      // components even though they belong to one molecule.
      if (section == "bonds" || section == "constraints") {
        if (current_molecule == nullptr)
          continue;
        std::istringstream iss(line);
        std::uint32_t a = 0;
        std::uint32_t b = 0;
        if (iss >> a >> b && a > 0 && b > 0 && a != b) {
          current_molecule->bonds.push_back({a - 1, b - 1});
          current_molecule->atom_count =
              std::max(current_molecule->atom_count, std::max(a, b));
        }
        continue;
      }

      if (section == "molecules") {
        if (conditional_depth) mass_binding_supported = false;
        std::istringstream iss(line);
        std::string name;
        std::uint32_t count = 0;
        if (iss >> name >> count && count > 0)
          molecules.push_back({std::move(name), count});
      }
    }
  }

  if (molecules.empty()) {
    for (const std::string &name : molecule_type_order)
      molecules.push_back({name, 1});
  }
  if (molecules.empty())
    return std::unexpected("no [ molecules ] or [ moleculetype ] entries");

  ExpandedGromacsTopology expanded;
  for (const auto &instance : molecules) {
    const auto type_it = molecule_types.find(instance.name);
    if (type_it == molecule_types.end())
      return std::unexpected("molecule type not defined: " + instance.name);

    const GromacsMoleculeType &type = type_it->second;
    for (std::uint32_t copy = 0; copy < instance.count; ++copy) {
      const std::uint32_t begin = expanded.atom_count;
      const std::uint32_t end = begin + type.atom_count;
      expanded.molecule_ranges.push_back({begin, end, instance.name, copy});
      expanded.vdw_radii.resize(
          end, std::numeric_limits<double>::quiet_NaN());
      expanded.atomic_masses.resize(end, std::numeric_limits<double>::quiet_NaN());
      for (std::uint32_t atom = 0; atom < type.atom_count; ++atom) {
        if (atom >= type.atom_types.size() || type.atom_types[atom].empty())
          continue;
        const std::string &atom_type = type.atom_types[atom];
        if (mass_binding_supported) {
          if (atom < type.explicit_masses.size() &&
              !std::isnan(type.explicit_masses[atom]))
            expanded.atomic_masses[begin + atom] = type.explicit_masses[atom];
          else if (const auto mass_it = atom_type_masses.find(atom_type);
                   mass_it != atom_type_masses.end())
            expanded.atomic_masses[begin + atom] = mass_it->second;
        }
        const GromacsLennardJonesType *lj = nullptr;
        if (const auto self_it = self_nonbond_parameters.find(atom_type);
            self_it != self_nonbond_parameters.end()) {
          lj = &self_it->second;
        } else if (const auto type_it = atom_types.find(atom_type);
                   type_it != atom_types.end()) {
          lj = &type_it->second;
        }
        if (lj == nullptr)
          continue;
        expanded.vdw_radii[begin + atom] =
            gromacs_vdw_radius_angstrom(*lj, combination_rule);
      }
      for (MolTopologyBond bond : type.bonds) {
        if (bond.a >= type.atom_count || bond.b >= type.atom_count)
          continue;
        expanded.bonds.push_back({begin + bond.a, begin + bond.b});
      }
      expanded.atom_count = end;
    }
  }

  return expanded;
}

std::uint64_t bond_key(std::uint32_t a, std::uint32_t b) {
  if (a > b)
    std::swap(a, b);
  return (static_cast<std::uint64_t>(a) << 32) | b;
}

bool source_ids_are_complete(std::span<const mol_info> selection_info) {
  return !selection_info.empty() &&
         std::all_of(selection_info.begin(), selection_info.end(),
                     [](const mol_info &info) {
                       return info.has_original_atom_id();
                     });
}

void fill_sequential_source_map(
    std::span<const mol_info> selection_info,
    std::unordered_map<std::uint32_t, std::uint32_t> &source_to_local,
    std::vector<std::uint32_t> &source_atom_ids) {
  source_to_local.clear();
  source_atom_ids.clear();
  source_to_local.reserve(selection_info.size() * 2 + 1);
  source_atom_ids.reserve(selection_info.size());
  for (std::uint32_t local = 0;
       local < static_cast<std::uint32_t>(selection_info.size()); ++local) {
    source_to_local.emplace(local, local);
    source_atom_ids.push_back(local);
  }
}

std::expected<std::unordered_map<std::uint32_t, std::uint32_t>, std::string>
build_source_to_local_map(const ExpandedGromacsTopology &expanded,
                          std::span<const mol_info> selection_info,
                          std::vector<std::uint32_t> &source_atom_ids) {
  std::unordered_map<std::uint32_t, std::uint32_t> source_to_local;
  source_to_local.reserve(selection_info.size() * 2 + 1);
  source_atom_ids.clear();
  source_atom_ids.reserve(selection_info.size());

  const bool any_source_ids =
      std::any_of(selection_info.begin(), selection_info.end(),
                  [](const mol_info &info) {
                    return info.has_original_atom_id();
                  });
  const bool use_direct_ids = source_ids_are_complete(selection_info);
  if (any_source_ids && !use_direct_ids) {
    return std::unexpected("selection has partially populated source atom IDs");
  }
  if (!use_direct_ids) {
    if (expanded.atom_count != selection_info.size()) {
      return std::unexpected(
          "topology atom count does not match selection and original atom IDs "
          "are unavailable");
    }
    fill_sequential_source_map(selection_info, source_to_local,
                               source_atom_ids);
    return source_to_local;
  }

  for (std::uint32_t local = 0;
       local < static_cast<std::uint32_t>(selection_info.size()); ++local) {
    const size_t source_id_size = selection_info[local].get_original_atom_id();
    if (source_id_size > std::numeric_limits<std::uint32_t>::max()) {
      return std::unexpected("source atom ID exceeds uint32_t range");
    }
    const auto source_id = static_cast<std::uint32_t>(source_id_size);
    if (source_id >= expanded.atom_count) {
      if (expanded.atom_count == selection_info.size()) {
        fill_sequential_source_map(selection_info, source_to_local,
                                   source_atom_ids);
        return source_to_local;
      }
      return std::unexpected("source atom ID exceeds topology atom count");
    }
    if (!source_to_local.emplace(source_id, local).second) {
      return std::unexpected("duplicate source atom ID in selection");
    }
    source_atom_ids.push_back(source_id);
  }
  return source_to_local;
}

std::vector<MolTopologyMoleculeRange>
project_molecule_ranges_to_selection(
    const ExpandedGromacsTopology &expanded,
    const std::unordered_map<std::uint32_t, std::uint32_t> &source_to_local,
    std::uint32_t local_atom_count) {
  std::vector<MolTopologyMoleculeRange> projected_ranges;
  projected_ranges.reserve(expanded.molecule_ranges.size());

  std::vector<std::uint32_t> local_atoms;
  for (const MolTopologyMoleculeRange &range : expanded.molecule_ranges) {
    if (range.begin >= range.end || range.end > expanded.atom_count)
      continue;

    local_atoms.clear();
    local_atoms.reserve(range.end - range.begin);
    bool complete_selection = true;
    for (std::uint32_t source = range.begin; source < range.end; ++source) {
      const auto it = source_to_local.find(source);
      if (it == source_to_local.end()) {
        complete_selection = false;
        break;
      }
      if (it->second >= local_atom_count) {
        complete_selection = false;
        break;
      }
      local_atoms.push_back(it->second);
    }
    if (!complete_selection || local_atoms.empty())
      continue;

    std::sort(local_atoms.begin(), local_atoms.end());
    if (std::adjacent_find(local_atoms.begin(), local_atoms.end()) !=
        local_atoms.end()) {
      return {};
    }
    const std::uint32_t local_begin = local_atoms.front();
    const std::uint32_t local_end = local_atoms.back() + 1;
    if (local_end - local_begin != local_atoms.size())
      continue;

    projected_ranges.push_back(
        {local_begin, local_end, range.name, range.copy_index});
  }
  return projected_ranges;
}

} // namespace

std::span<const std::uint32_t>
MolTopology::neighbors(std::uint32_t atom) const noexcept {
  if (atom >= atom_count_ || row_offsets_.size() != atom_count_ + 1)
    return {};
  const std::uint32_t begin = row_offsets_[atom];
  const std::uint32_t end = row_offsets_[atom + 1];
  if (begin == end)
    return {};
  return std::span<const std::uint32_t>(col_indices_.data() + begin,
                                        end - begin);
}

std::optional<double>
MolTopology::vdw_radius(std::uint32_t atom) const noexcept {
  if (atom >= vdw_radii_.size() || atom >= vdw_radius_sources_.size() ||
      vdw_radius_sources_[atom] == MolTopologyVdwRadiusSource::None ||
      !std::isfinite(vdw_radii_[atom]) || vdw_radii_[atom] < 0.0) {
    return std::nullopt;
  }
  return vdw_radii_[atom];
}

std::optional<double>
MolTopology::atomic_mass(std::uint32_t atom) const noexcept {
  if (atom >= atomic_masses_.size() || !std::isfinite(atomic_masses_[atom]) ||
      atomic_masses_[atom] <= 0.0) return std::nullopt;
  return atomic_masses_[atom];
}

void MolTopology::set_atomic_masses(std::vector<double> masses) {
  atomic_masses_.clear();
  if (masses.size() == atom_count_) atomic_masses_ = std::move(masses);
}

MolTopologyVdwRadiusSource
MolTopology::vdw_radius_source(std::uint32_t atom) const noexcept {
  if (atom >= vdw_radius_sources_.size())
    return MolTopologyVdwRadiusSource::None;
  return vdw_radius_sources_[atom];
}

void MolTopology::clear() noexcept {
  atom_count_ = 0;
  bonds_.clear();
  molecule_ranges_.clear();
  row_offsets_.clear();
  col_indices_.clear();
  source_atom_ids_.clear();
  vdw_radii_.clear();
  atomic_masses_.clear();
  vdw_radius_sources_.clear();
  vdw_radius_count_ = 0;
}

void MolTopology::set_bonds(
    std::uint32_t atom_count, std::vector<MolTopologyBond> bonds,
    std::vector<MolTopologyMoleculeRange> molecule_ranges,
    std::vector<std::uint32_t> source_atom_ids,
    std::vector<double> vdw_radii) {
  atom_count_ = atom_count;
  atomic_masses_.clear();
  bonds_.clear();
  bonds_.reserve(bonds.size());

  std::unordered_set<std::uint64_t> seen;
  seen.reserve(bonds.size() * 2 + 1);
  for (MolTopologyBond bond : bonds) {
    if (bond.a == bond.b || bond.a >= atom_count_ || bond.b >= atom_count_)
      continue;
    if (bond.a > bond.b)
      std::swap(bond.a, bond.b);
    if (!seen.insert(bond_key(bond.a, bond.b)).second)
      continue;
    bonds_.push_back(bond);
  }

  std::sort(bonds_.begin(), bonds_.end(),
            [](const MolTopologyBond &lhs, const MolTopologyBond &rhs) {
              if (lhs.a != rhs.a)
                return lhs.a < rhs.a;
              return lhs.b < rhs.b;
            });
  molecule_ranges_ = std::move(molecule_ranges);
  source_atom_ids_ = std::move(source_atom_ids);
  vdw_radii_.assign(atom_count_, 0.0);
  vdw_radius_sources_.assign(atom_count_, MolTopologyVdwRadiusSource::None);
  vdw_radius_count_ = 0;
  const std::size_t radius_count =
      std::min<std::size_t>(atom_count_, vdw_radii.size());
  for (std::size_t atom = 0; atom < radius_count; ++atom) {
    if (!std::isfinite(vdw_radii[atom]) || vdw_radii[atom] < 0.0)
      continue;
    vdw_radii_[atom] = vdw_radii[atom];
    vdw_radius_sources_[atom] =
        vdw_radii[atom] == 0.0
            ? MolTopologyVdwRadiusSource::GromacsTopologyExplicitZeroLJ
            : MolTopologyVdwRadiusSource::GromacsTopology;
    ++vdw_radius_count_;
  }
  build_adjacency();
}

void MolTopology::cache_missing_vdw_radii(
    std::span<const double> radii, std::span<const double> sigma,
    std::span<const double> epsilon,
    std::span<const std::int32_t> type_ids) {
  if (radii.empty())
    return;
  if (vdw_radii_.size() < radii.size()) {
    vdw_radii_.resize(radii.size(), 0.0);
    vdw_radius_sources_.resize(radii.size(),
                               MolTopologyVdwRadiusSource::None);
  }
  for (std::size_t atom = 0; atom < radii.size(); ++atom) {
    if (vdw_radius_sources_[atom] != MolTopologyVdwRadiusSource::None ||
        !std::isfinite(radii[atom]) || radii[atom] < 0.0)
      continue;
    const bool assigned_parameter = atom < type_ids.size() &&
                                    type_ids[atom] >= 0;
    const bool explicit_zero_lj =
        radii[atom] == 0.0 && assigned_parameter && atom < sigma.size() &&
        atom < epsilon.size() && std::isfinite(sigma[atom]) &&
        std::isfinite(epsilon[atom]) && sigma[atom] >= 0.0 &&
        epsilon[atom] >= 0.0 &&
        (sigma[atom] == 0.0 || epsilon[atom] == 0.0);
    if (radii[atom] == 0.0 && !explicit_zero_lj)
      continue;
    vdw_radii_[atom] = radii[atom];
    vdw_radius_sources_[atom] =
        explicit_zero_lj
            ? MolTopologyVdwRadiusSource::MultiMolParametersExplicitZeroLJ
            : MolTopologyVdwRadiusSource::MultiMolParameters;
    ++vdw_radius_count_;
  }
}

void MolTopology::build_adjacency() {
  row_offsets_.assign(static_cast<size_t>(atom_count_) + 1, 0);
  col_indices_.clear();

  for (const MolTopologyBond &bond : bonds_) {
    if (bond.a >= atom_count_ || bond.b >= atom_count_)
      continue;
    ++row_offsets_[bond.a + 1];
    ++row_offsets_[bond.b + 1];
  }

  for (size_t i = 1; i < row_offsets_.size(); ++i)
    row_offsets_[i] += row_offsets_[i - 1];

  col_indices_.assign(row_offsets_.empty() ? 0 : row_offsets_.back(), 0);
  std::vector<std::uint32_t> cursor = row_offsets_;
  for (const MolTopologyBond &bond : bonds_) {
    if (bond.a >= atom_count_ || bond.b >= atom_count_)
      continue;
    col_indices_[cursor[bond.a]++] = bond.b;
    col_indices_[cursor[bond.b]++] = bond.a;
  }

  for (std::uint32_t atom = 0; atom < atom_count_; ++atom) {
    auto begin = col_indices_.begin() + row_offsets_[atom];
    auto end = col_indices_.begin() + row_offsets_[atom + 1];
    std::sort(begin, end);
  }
}

std::expected<MolTopology, std::string>
build_gromacs_mol_topology(std::string_view topology_path,
                           std::span<const mol_info> selection_info) {
  auto expanded_result = parse_and_expand_gromacs_topology(topology_path);
  if (!expanded_result)
    return std::unexpected(expanded_result.error());

  const ExpandedGromacsTopology &expanded = *expanded_result;
  std::vector<std::uint32_t> source_atom_ids;
  auto map_result =
      build_source_to_local_map(expanded, selection_info, source_atom_ids);
  if (!map_result)
    return std::unexpected(map_result.error());

  const auto &source_to_local = *map_result;
  std::vector<MolTopologyBond> projected_bonds;
  projected_bonds.reserve(expanded.bonds.size());
  for (const MolTopologyBond &bond : expanded.bonds) {
    const auto a_it = source_to_local.find(bond.a);
    if (a_it == source_to_local.end())
      continue;
    const auto b_it = source_to_local.find(bond.b);
    if (b_it == source_to_local.end())
      continue;
    projected_bonds.push_back({a_it->second, b_it->second});
  }

  std::vector<double> projected_vdw_radii(
      selection_info.size(), std::numeric_limits<double>::quiet_NaN());
  std::vector<double> projected_masses(selection_info.size(),
      std::numeric_limits<double>::quiet_NaN());
  // Do not accept the legacy sequential fallback when explicit IDs disagree.
  const bool exact_mass_identity = std::all_of(source_to_local.begin(), source_to_local.end(),
      [&](const auto &entry) {
        const auto [source, local] = entry;
        return local < selection_info.size() &&
            selection_info[local].has_original_atom_id() &&
            selection_info[local].get_original_atom_id() == source;
      });
  for (const auto &[source, local] : source_to_local) {
    if (exact_mass_identity && source < expanded.atomic_masses.size() &&
        local < projected_masses.size()) projected_masses[local] = expanded.atomic_masses[source];
    if (source < expanded.vdw_radii.size() &&
        local < projected_vdw_radii.size())
      projected_vdw_radii[local] = expanded.vdw_radii[source];
  }

  MolTopology topology;
  auto molecule_ranges = project_molecule_ranges_to_selection(
      expanded, source_to_local,
      static_cast<std::uint32_t>(selection_info.size()));
  topology.set_bonds(static_cast<std::uint32_t>(selection_info.size()),
                     std::move(projected_bonds), std::move(molecule_ranges),
                     std::move(source_atom_ids),
                     std::move(projected_vdw_radii));
  topology.set_atomic_masses(std::move(projected_masses));
  return topology;
}

bool attach_gromacs_mol_topology_graph(Multi_MOL &mol,
                                       std::string_view topology_path) {
  if (mol.info.empty()) {
    mol.topology.clear();
    return false;
  }
  auto topology_result = build_gromacs_mol_topology(topology_path, mol.info);
  if (!topology_result) {
    mol.topology.clear();
    std::cerr << "[WARNING] MolTopology graph unavailable for "
              << topology_path << ": " << topology_result.error() << "\n";
    return false;
  }

  mol.topology = std::move(*topology_result);
  std::cout << "[INFO] MolTopology graph attached from " << topology_path
            << ": selection_atoms=" << mol.info.size()
            << ", projected_bonds=" << mol.topology.bonds().size()
            << ", molecule_ranges=" << mol.topology.molecule_ranges().size()
            << ", vdw_radii=" << mol.topology.vdw_radius_count()
            << "\n";
  return mol.topology.has_bonds();
}
