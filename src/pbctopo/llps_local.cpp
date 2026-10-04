#include "llps_local.h"
#include "../Multi_MOL.h"
#include "lattice_math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace titan_pbctopo {
namespace {

void scaled_to_cartesian(box Box, double sx, double sy, double sz, double &x,
                         double &y, double &z) {
  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  x = sx * v1[0] + sy * v2[0] + sz * v3[0];
  y = sx * v1[1] + sy * v2[1] + sz * v3[1];
  z = sx * v1[2] + sy * v2[2] + sz * v3[2];
}

void append_pbctopo_llps_unit(PbctopoLlpsLocalUnits &units,
                              std::vector<std::size_t> atoms) {
  if (atoms.empty())
    return;
  std::sort(atoms.begin(), atoms.end());
  const int unit_id = static_cast<int>(units.units.size());
  for (std::size_t atom : atoms)
    units.atom_unit[atom] = unit_id;
  units.units.push_back(std::move(atoms));
}

bool append_pbctopo_llps_topology_units_for_owner(
    const Multi_MOL &system, int owner, const std::vector<PbctopoAtom> &atoms,
    PbctopoLlpsLocalUnits &units) {
  const auto &ranges = system.topology.molecule_ranges();
  if (ranges.empty() || system.info.empty())
    return false;

  std::vector<std::size_t> local_to_atom(
      system.info.size(), std::numeric_limits<std::size_t>::max());
  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    const auto &atom = atoms[atom_idx];
    if (atom.owner != owner || atom.local_atom_index >= local_to_atom.size())
      continue;
    local_to_atom[atom.local_atom_index] = atom_idx;
  }

  bool added = false;
  for (const MolTopologyMoleculeRange &range : ranges) {
    if (range.begin >= range.end || range.end > local_to_atom.size())
      continue;

    std::unordered_map<std::uint64_t, std::size_t> chain_to_slot;
    std::vector<std::vector<std::size_t>> range_units;
    for (std::uint32_t local = range.begin; local < range.end; ++local) {
      const std::size_t atom_idx = local_to_atom[local];
      if (atom_idx == std::numeric_limits<std::size_t>::max())
        continue;
      const auto &atom = atoms[atom_idx];
      const std::uint64_t chain_key = atom.has_topology_key ? atom.chain_key : 0;
      auto [it, inserted] = chain_to_slot.emplace(chain_key, range_units.size());
      if (inserted)
        range_units.emplace_back();
      range_units[it->second].push_back(atom_idx);
    }

    for (auto &unit_atoms : range_units) {
      if (unit_atoms.empty())
        continue;
      append_pbctopo_llps_unit(units, std::move(unit_atoms));
      added = true;
    }
  }
  return added;
}

bool append_pbctopo_llps_chain_units_for_owner(
    int owner, const std::vector<PbctopoAtom> &atoms,
    PbctopoLlpsLocalUnits &units) {
  std::unordered_map<std::uint64_t, std::size_t> chain_to_slot;
  std::vector<std::vector<std::size_t>> chain_units;
  bool has_chain_metadata = false;
  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    const auto &atom = atoms[atom_idx];
    if (atom.owner != owner || !atom.has_topology_key)
      continue;
    has_chain_metadata = true;
    auto [it, inserted] = chain_to_slot.emplace(atom.chain_key,
                                                chain_units.size());
    if (inserted)
      chain_units.emplace_back();
    chain_units[it->second].push_back(atom_idx);
  }
  if (!has_chain_metadata)
    return false;
  for (auto &unit_atoms : chain_units)
    append_pbctopo_llps_unit(units, std::move(unit_atoms));
  return !chain_units.empty();
}

bool append_pbctopo_llps_geometry_units_for_owner(
    int owner, const std::vector<PbctopoAtom> &atoms,
    const std::vector<PbctopoLocalEdge> &edges,
    PbctopoLlpsLocalUnits &units) {
  std::vector<std::vector<std::size_t>> adjacency(atoms.size());
  std::vector<char> owner_atom(atoms.size(), 0);
  bool has_owner_atom = false;
  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    if (atoms[atom_idx].owner != owner)
      continue;
    owner_atom[atom_idx] = 1;
    has_owner_atom = true;
  }
  if (!has_owner_atom)
    return false;

  for (const auto &edge : edges) {
    if (edge.a >= atoms.size() || edge.b >= atoms.size())
      continue;
    if (owner_atom[edge.a] == 0 || owner_atom[edge.b] == 0)
      continue;
    adjacency[edge.a].push_back(edge.b);
    adjacency[edge.b].push_back(edge.a);
  }

  std::vector<char> seen(atoms.size(), 0);
  std::vector<std::size_t> stack;
  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    if (owner_atom[atom_idx] == 0 || seen[atom_idx] != 0)
      continue;
    seen[atom_idx] = 1;
    stack.clear();
    stack.push_back(atom_idx);
    std::vector<std::size_t> unit_atoms;
    while (!stack.empty()) {
      const std::size_t current = stack.back();
      stack.pop_back();
      unit_atoms.push_back(current);
      for (std::size_t nb : adjacency[current]) {
        if (seen[nb] != 0)
          continue;
        seen[nb] = 1;
        stack.push_back(nb);
      }
    }
    append_pbctopo_llps_unit(units, std::move(unit_atoms));
  }
  return true;
}

} // namespace

std::string merge_pbctopo_llps_unit_source(const std::string &current,
                                           const std::string &next) {
  if (next.empty() || next == "none")
    return current;
  if (current.empty() || current == "none")
    return next;
  if (current == next)
    return current;
  return "mixed";
}

PbctopoLlpsLocalUnits build_pbctopo_llps_local_units(
    const Multi_MOL &mol1_system, const Multi_MOL &mol2_system,
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<PbctopoLocalEdge> &edges) {
  PbctopoLlpsLocalUnits units;
  units.atom_unit.assign(atoms.size(), -1);

  auto append_owner_units = [&](const Multi_MOL &system, int owner) {
    std::string owner_source = "none";
    if (append_pbctopo_llps_topology_units_for_owner(system, owner, atoms,
                                                     units)) {
      owner_source = "topology_range";
    } else if (append_pbctopo_llps_chain_units_for_owner(owner, atoms, units)) {
      owner_source = "chain";
    } else if (append_pbctopo_llps_geometry_units_for_owner(owner, atoms, edges,
                                                            units)) {
      owner_source = "geometry_component";
    }
    units.source = merge_pbctopo_llps_unit_source(units.source, owner_source);
  };

  append_owner_units(mol1_system, 0);
  if (!mol2_system.m_mol.empty())
    append_owner_units(mol2_system, 1);

  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    if (units.atom_unit[atom_idx] >= 0)
      continue;
    append_pbctopo_llps_unit(units, std::vector<std::size_t>{atom_idx});
    units.source = merge_pbctopo_llps_unit_source(units.source,
                                                  "geometry_component");
  }
  return units;
}

std::vector<Vec3> unwrap_pbctopo_llps_local_points(
    const std::vector<PbctopoAtom> &wrapped_atoms,
    const std::vector<PbctopoLocalEdge> &edges, box Box,
    const PbctopoLlpsLocalUnits &units, PbctopoTimingStats &timing) {
  std::vector<PbctopoAtom> unwrapped = wrapped_atoms;
  std::vector<Vec3> points(wrapped_atoms.size());
  if (wrapped_atoms.empty())
    return points;
  const PbctopoLatticeMetric lattice(Box);
  if (!lattice.valid()) {
    for (std::size_t atom_idx = 0; atom_idx < wrapped_atoms.size(); ++atom_idx) {
      points[atom_idx] = {wrapped_atoms[atom_idx].x, wrapped_atoms[atom_idx].y,
                          wrapped_atoms[atom_idx].z};
    }
    return points;
  }

  std::vector<std::vector<std::size_t>> adjacency(wrapped_atoms.size());
  for (const auto &edge : edges) {
    if (edge.a >= wrapped_atoms.size() || edge.b >= wrapped_atoms.size())
      continue;
    const int unit_a = units.atom_unit[edge.a];
    const int unit_b = units.atom_unit[edge.b];
    if (unit_a >= 0 && unit_a == unit_b) {
      adjacency[edge.a].push_back(edge.b);
      adjacency[edge.b].push_back(edge.a);
      ++timing.pbctopo_llps_local_edges_used;
    } else {
      ++timing.pbctopo_llps_local_edges_skipped_cross_unit;
    }
  }

  std::vector<char> seen(wrapped_atoms.size(), 0);
  std::vector<std::size_t> stack;
  for (const auto &unit_atoms : units.units) {
    std::size_t unit_components = 0;
    for (std::size_t seed : unit_atoms) {
      if (seed >= wrapped_atoms.size() || seen[seed] != 0)
        continue;
      ++unit_components;
      seen[seed] = 1;
      stack.clear();
      stack.push_back(seed);
      unwrapped[seed].sx = wrapped_atoms[seed].sx;
      unwrapped[seed].sy = wrapped_atoms[seed].sy;
      unwrapped[seed].sz = wrapped_atoms[seed].sz;

      while (!stack.empty()) {
        const std::size_t current = stack.back();
        stack.pop_back();
        for (std::size_t nb : adjacency[current]) {
          if (seen[nb] != 0)
            continue;
          const PbctopoNearestImage mic = lattice.nearest_image_scaled(
              wrapped_atoms[nb].sx - wrapped_atoms[current].sx,
              wrapped_atoms[nb].sy - wrapped_atoms[current].sy,
              wrapped_atoms[nb].sz - wrapped_atoms[current].sz);
          if (!mic.valid)
            continue;
          unwrapped[nb].sx = unwrapped[current].sx + mic.sx;
          unwrapped[nb].sy = unwrapped[current].sy + mic.sy;
          unwrapped[nb].sz = unwrapped[current].sz + mic.sz;
          seen[nb] = 1;
          stack.push_back(nb);
        }
      }
    }
    if (unit_components > 1)
      ++timing.pbctopo_llps_local_disconnected_units;
  }

  for (std::size_t atom_idx = 0; atom_idx < unwrapped.size(); ++atom_idx) {
    auto &atom = unwrapped[atom_idx];
    scaled_to_cartesian(Box, atom.sx, atom.sy, atom.sz, atom.x, atom.y, atom.z);
    points[atom_idx] = {atom.x, atom.y, atom.z};
  }
  return points;
}

} // namespace titan_pbctopo
