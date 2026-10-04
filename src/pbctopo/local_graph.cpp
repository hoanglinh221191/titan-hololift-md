#include "local_graph.h"

#include "../BoxMatrix.h"
#include "../Multi_MOL.h"
#include "lattice_math.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>

namespace titan_pbctopo {
namespace {

double wrap_unit(double value) {
  value -= std::floor(value);
  if (value >= 1.0)
    value -= 1.0;
  if (value < 0.0)
    value += 1.0;
  return value;
}

std::uint64_t atom_pair_key(std::size_t a, std::size_t b) {
  if (a > b)
    std::swap(a, b);
  return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
}

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

std::vector<PbctopoRange>
build_ranges_from_index(const std::vector<::index> &Index,
                        std::size_t atom_count) {
  std::vector<PbctopoRange> ranges;
  ranges.reserve(Index.size());
  for (std::size_t k = 0; k + 1 < Index.size(); ++k) {
    const std::size_t begin = Index[k].get_res_pos_e(0) - 1;
    const std::size_t end = Index[k + 1].get_res_pos_e(0) - 1;
    if (begin >= atom_count || end > atom_count || begin >= end)
      continue;
    ranges.push_back({begin, end, ranges.size(), true});
  }
  return ranges;
}

std::vector<PbctopoRange>
build_ranges_from_info(const std::vector<mol_info> &info,
                       std::size_t atom_count) {
  std::vector<PbctopoRange> ranges;
  if (info.size() != atom_count || atom_count == 0)
    return ranges;

  std::size_t begin = 0;
  std::size_t source_index = 0;
  for (std::size_t i = 1; i < atom_count; ++i) {
    if (info[i].get_chainnum() != info[i - 1].get_chainnum()) {
      if (begin < i)
        ranges.push_back({begin, i, source_index++, true});
      begin = i;
    }
  }
  if (begin < atom_count)
    ranges.push_back({begin, atom_count, source_index, true});
  return ranges;
}

std::vector<PbctopoRange>
build_ranges_from_topology(const MolTopology &topology,
                           std::size_t atom_count) {
  std::vector<PbctopoRange> ranges;
  const auto &molecule_ranges = topology.molecule_ranges();
  ranges.reserve(molecule_ranges.size());
  for (const MolTopologyMoleculeRange &range : molecule_ranges) {
    if (range.begin >= range.end || range.end > atom_count)
      continue;
    ranges.push_back({range.begin, range.end, ranges.size(), true});
  }
  return ranges;
}

char first_alpha_upper(const std::string &text) {
  for (unsigned char ch : text) {
    if (std::isalpha(ch))
      return static_cast<char>(std::toupper(ch));
  }
  return '\0';
}

bool detect_heavy_atom(const std::vector<mol_info> *info,
                       std::size_t local_atom_index) {
  if (info == nullptr || local_atom_index >= info->size())
    return true;
  const auto &atom_info = (*info)[local_atom_index];
  const std::string atom_name = atom_info.get_atom();
  const std::string &atom_type = atom_info.get_atom_type();
  const char atom_symbol = first_alpha_upper(atom_name);
  const char type_symbol = first_alpha_upper(atom_type);
  return atom_symbol != 'H' && type_symbol != 'H';
}

struct PbctopoInferredRadius {
  double radius = 1.70;
  bool element_supported = false;
};

PbctopoInferredRadius
infer_bondi_vdw_radius(const std::vector<mol_info> &info,
                       std::size_t local_atom_index) {
  if (local_atom_index >= info.size())
    return {};
  std::string atom_name = info[local_atom_index].get_atom();
  std::string atom_type = info[local_atom_index].get_atom_type();
  for (char &ch : atom_name)
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  for (char &ch : atom_type)
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  if (atom_name == "CL" || atom_type.starts_with("CL"))
    return {1.75, true};
  if (atom_name == "BR" || atom_type.starts_with("BR"))
    return {1.85, true};
  const char element = first_alpha_upper(atom_name);
  switch (element) {
  case 'H':
    return {1.20, true};
  case 'N':
    return {1.55, true};
  case 'O':
    return {1.52, true};
  case 'F':
    return {1.47, true};
  case 'P':
    return {1.80, true};
  case 'S':
    return {1.80, true};
  case 'I':
    return {1.98, true};
  case 'C':
    return {1.70, true};
  default:
    return {};
  }
}

std::string uppercase_atom_name(std::string atom_name) {
  for (char &ch : atom_name) {
    ch = static_cast<char>(
        std::toupper(static_cast<unsigned char>(ch)));
  }
  return atom_name;
}

struct PbctopoMetadataResidueGroup {
  std::size_t begin = 0;
  std::size_t end = 0;
  std::size_t chainnum = 0;
  std::size_t internal_resnr = 0;
  std::size_t anchor = std::numeric_limits<std::size_t>::max();
};

int metadata_anchor_priority(const Multi_MOL &system,
                             const PbctopoSystemStaticData &static_data,
                             std::size_t atom_idx) {
  const std::string atom_name =
      uppercase_atom_name(system.info[atom_idx].get_atom());
  if (atom_name == "CA")
    return 0;
  if (atom_name == "C4'" || atom_name == "C4*")
    return 1;
  if (atom_name == "BB")
    return 2;
  const bool is_heavy = atom_idx >= static_data.heavy_mask.size() ||
                        static_data.heavy_mask[atom_idx] != 0;
  return is_heavy ? 3 : 4;
}

std::size_t choose_metadata_residue_anchor(
    const Multi_MOL &system, const PbctopoSystemStaticData &static_data,
    std::size_t begin, std::size_t end) {
  std::size_t anchor = begin;
  int best_priority = std::numeric_limits<int>::max();
  for (std::size_t atom_idx = begin; atom_idx < end; ++atom_idx) {
    const int priority =
        metadata_anchor_priority(system, static_data, atom_idx);
    if (priority < best_priority) {
      best_priority = priority;
      anchor = atom_idx;
    }
  }
  return anchor;
}

bool add_projected_metadata_edge(
    std::size_t a, std::size_t b, bool inter_residue,
    bool hard_connectivity,
    std::unordered_set<std::uint64_t> &edge_set,
    std::vector<PbctopoProjectedMetadataEdge> &edges) {
  if (a == b)
    return false;
  if (!edge_set.insert(atom_pair_key(a, b)).second)
    return false;
  edges.push_back({a, b, inter_residue, hard_connectivity});
  return true;
}

double metadata_anchor_continuity_cutoff(
    const Multi_MOL &system, const PbctopoSystemStaticData &data,
    std::size_t lhs_anchor, std::size_t rhs_anchor) {
  const int lhs_priority =
      metadata_anchor_priority(system, data, lhs_anchor);
  const int rhs_priority =
      metadata_anchor_priority(system, data, rhs_anchor);
  if (lhs_priority == 1 || rhs_priority == 1)
    return 8.0;
  if (lhs_priority <= 2 || rhs_priority <= 2)
    return 5.0;
  return 4.5;
}

bool metadata_inter_residue_edge_is_hard(
    const Multi_MOL &system, const MOL &mol,
    const PbctopoSystemStaticData &data,
    const PbctopoMetadataResidueGroup &prev,
    const PbctopoMetadataResidueGroup &curr,
    const PbctopoLatticeMetric &lattice) {
  if (prev.internal_resnr == std::numeric_limits<std::size_t>::max() ||
      curr.internal_resnr != prev.internal_resnr + 1 || !lattice.valid() ||
      prev.anchor >= mol.pos.x.size() || curr.anchor >= mol.pos.x.size()) {
    return false;
  }
  const PbctopoNearestImage nearest = lattice.nearest_image_cartesian(
      mol.pos.x[curr.anchor] - mol.pos.x[prev.anchor],
      mol.pos.y[curr.anchor] - mol.pos.y[prev.anchor],
      mol.pos.z[curr.anchor] - mol.pos.z[prev.anchor]);
  if (!nearest.valid)
    return false;
  const double cutoff = metadata_anchor_continuity_cutoff(
      system, data, prev.anchor, curr.anchor);
  return nearest.d2 <= cutoff * cutoff;
}

void build_projected_metadata_graph(const Multi_MOL &system, const MOL &mol,
                                    PbctopoSystemStaticData &data) {
  data.metadata_graph_available = false;
  data.metadata_residue_groups = 0;
  data.metadata_intra_edges = 0;
  data.metadata_inter_edges = 0;
  data.metadata_inter_hard_edges = 0;
  data.metadata_inter_soft_edges = 0;
  data.projected_metadata_edges.clear();
  if (system.info.size() != data.atom_count || data.atom_count == 0)
    return;

  std::vector<PbctopoMetadataResidueGroup> groups;
  groups.reserve(data.atom_count);
  std::size_t begin = 0;
  while (begin < data.atom_count) {
    const std::size_t chainnum = system.info[begin].get_chainnum();
    const std::size_t internal_resnr = system.info[begin].get_internal_resnr();
    std::size_t end = begin + 1;
    while (end < data.atom_count &&
           system.info[end].get_chainnum() == chainnum &&
           system.info[end].get_internal_resnr() == internal_resnr) {
      ++end;
    }
    groups.push_back({begin, end, chainnum, internal_resnr,
                      choose_metadata_residue_anchor(system, data, begin,
                                                     end)});
    begin = end;
  }
  data.metadata_residue_groups = groups.size();
  data.continuity_anchor_local.clear();
  data.continuity_anchor_local.reserve(groups.size());
  std::unordered_set<std::uint64_t> metadata_edge_set;
  metadata_edge_set.reserve(data.atom_count * 2 + 1);
  data.projected_metadata_edges.reserve(data.atom_count * 2);

  for (const auto &group : groups) {
    if (group.anchor == std::numeric_limits<std::size_t>::max())
      continue;
    data.continuity_anchor_local.push_back(group.anchor);
    for (std::size_t atom_idx = group.begin; atom_idx < group.end;
         ++atom_idx) {
      if (atom_idx == group.anchor)
        continue;
      const bool atom_is_heavy =
          atom_idx >= data.heavy_mask.size() || data.heavy_mask[atom_idx] != 0;
      if (!atom_is_heavy)
        continue;
      if (add_projected_metadata_edge(group.anchor, atom_idx, false, true,
                                      metadata_edge_set,
                                      data.projected_metadata_edges)) {
        ++data.metadata_intra_edges;
      }
    }
  }

  const PbctopoLatticeMetric lattice(mol.Box);
  for (std::size_t group_idx = 1; group_idx < groups.size(); ++group_idx) {
    const auto &prev = groups[group_idx - 1];
    const auto &curr = groups[group_idx];
    if (prev.chainnum != curr.chainnum)
      continue;
    if (prev.anchor == std::numeric_limits<std::size_t>::max() ||
        curr.anchor == std::numeric_limits<std::size_t>::max())
      continue;
    const bool hard_connectivity = metadata_inter_residue_edge_is_hard(
        system, mol, data, prev, curr, lattice);
    if (add_projected_metadata_edge(prev.anchor, curr.anchor, true,
                                    hard_connectivity,
                                    metadata_edge_set,
                                    data.projected_metadata_edges)) {
      ++data.metadata_inter_edges;
      if (hard_connectivity)
        ++data.metadata_inter_hard_edges;
      else
        ++data.metadata_inter_soft_edges;
    }
  }

  data.metadata_graph_available = true;
}

std::vector<PbctopoContinuityPair> build_local_continuity_pairs(
    const Multi_MOL &system, const PbctopoSystemStaticData &static_data) {
  std::vector<PbctopoContinuityPair> continuity_pairs;
  if (static_data.atom_count == 0)
    return continuity_pairs;

  if (!system.topology.has_bonds()) {
    if (!static_data.metadata_graph_available)
      return continuity_pairs;
    continuity_pairs.reserve(static_data.projected_metadata_edges.size());
    for (const auto &edge : static_data.projected_metadata_edges) {
      continuity_pairs.push_back(
          {edge.a, edge.b, 1.0, 1.0,
           edge.hard_connectivity ? PbctopoContinuityKind::MetadataHard
                                  : PbctopoContinuityKind::MetadataSoft});
    }
    return continuity_pairs;
  }

  continuity_pairs.reserve(system.topology.bonds().size());
  for (const MolTopologyBond &bond : system.topology.bonds()) {
    if (bond.a >= static_data.atom_count || bond.b >= static_data.atom_count ||
        bond.a == bond.b) {
      continue;
    }
    const bool a_heavy = bond.a >= static_data.heavy_mask.size() ||
                         static_data.heavy_mask[bond.a] != 0;
    const bool b_heavy = bond.b >= static_data.heavy_mask.size() ||
                         static_data.heavy_mask[bond.b] != 0;
    if (!a_heavy || !b_heavy)
      continue;
    continuity_pairs.push_back({bond.a, bond.b, 1.0, 1.0,
                                PbctopoContinuityKind::TopologyHard});
  }
  return continuity_pairs;
}

void initialize_continuity_reference_lengths(
    const MOL &mol, std::vector<PbctopoContinuityPair> &continuity_pairs) {
  const PbctopoLatticeMetric lattice(mol.Box);
  const std::size_t atom_count = mol.pos.x.size();
  for (auto &pair : continuity_pairs) {
    if (pair.a >= atom_count || pair.b >= atom_count)
      continue;
    const double dx = mol.pos.x[pair.b] - mol.pos.x[pair.a];
    const double dy = mol.pos.y[pair.b] - mol.pos.y[pair.a];
    const double dz = mol.pos.z[pair.b] - mol.pos.z[pair.a];
    const PbctopoNearestImage mic =
        lattice.nearest_image_cartesian(dx, dy, dz);
    const double raw_d2 = dx * dx + dy * dy + dz * dz;
    const double reference_d2 =
        mic.valid && std::isfinite(mic.d2) ? mic.d2 : raw_d2;
    const double ref_d2 = std::max(reference_d2, 1.0e-6);
    pair.ref_d2 = ref_d2;
    pair.inv_ref_d2 = 1.0 / ref_d2;
  }
}

void build_projected_topology_bonds(
    const MolTopology &topology, std::size_t atom_count,
    std::vector<PbctopoProjectedTopologyBond> &projected_bonds) {
  projected_bonds.clear();
  if (!topology.has_bonds() || atom_count == 0)
    return;

  projected_bonds.reserve(topology.bonds().size());
  for (const MolTopologyBond &bond : topology.bonds()) {
    if (bond.a >= atom_count || bond.b >= atom_count || bond.a == bond.b)
      continue;
    projected_bonds.push_back({bond.a, bond.b});
  }
}

void append_global_continuity_pairs(
    const std::vector<PbctopoContinuityPair> &local_pairs, std::size_t offset,
    std::vector<PbctopoContinuityPair> &global_pairs) {
  global_pairs.reserve(global_pairs.size() + local_pairs.size());
  for (const auto &pair : local_pairs)
    global_pairs.push_back({pair.a + offset, pair.b + offset, pair.ref_d2,
                            pair.inv_ref_d2, pair.kind, pair.weight,
                            pair.hard_cutoff_d2});
}

void build_anchor_index_list(
    std::size_t atom_count,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    std::vector<std::size_t> &anchor_indices) {
  anchor_indices.clear();
  if (atom_count == 0 || continuity_pairs.empty())
    return;
  std::vector<char> seen(atom_count, 0);
  for (const auto &pair : continuity_pairs) {
    if (pair.a < atom_count && seen[pair.a] == 0) {
      seen[pair.a] = 1;
      anchor_indices.push_back(pair.a);
    }
    if (pair.b < atom_count && seen[pair.b] == 0) {
      seen[pair.b] = 1;
      anchor_indices.push_back(pair.b);
    }
  }
}

void build_metadata_anchor_index_list(
    const Multi_MOL &system, const PbctopoSystemStaticData &static_data,
    std::vector<std::size_t> &anchor_indices) {
  anchor_indices.clear();
  if (system.info.size() != static_data.atom_count ||
      static_data.atom_count == 0) {
    return;
  }

  std::vector<char> seen(static_data.atom_count, 0);
  auto append_range_anchors = [&](std::size_t begin, std::size_t end) {
    if (begin >= end || begin >= static_data.atom_count)
      return;
    end = std::min(end, static_data.atom_count);
    std::size_t i = begin;
    while (i < end) {
      const auto &start_info = system.info[i];
      const std::size_t chainnum = start_info.get_chainnum();
      const std::size_t internal_resnr = start_info.get_internal_resnr();
      std::size_t j = i + 1;
      while (j < end && system.info[j].get_chainnum() == chainnum &&
             system.info[j].get_internal_resnr() == internal_resnr) {
        ++j;
      }

      const std::size_t anchor =
          choose_metadata_residue_anchor(system, static_data, i, j);
      if (anchor != std::numeric_limits<std::size_t>::max() &&
          seen[anchor] == 0) {
        seen[anchor] = 1;
        anchor_indices.push_back(anchor);
      }
      i = j;
    }
  };

  if (static_data.ranges.empty()) {
    append_range_anchors(0, static_data.atom_count);
  } else {
    for (const auto &range : static_data.ranges)
      append_range_anchors(range.begin, range.end);
  }
}

void append_global_anchor_indices(
    const std::vector<std::size_t> &local_indices, std::size_t offset,
    std::vector<std::size_t> &global_indices) {
  global_indices.reserve(global_indices.size() + local_indices.size());
  for (std::size_t local_idx : local_indices)
    global_indices.push_back(local_idx + offset);
}

} // namespace

std::vector<PbctopoRange> build_ranges_for_system(const Multi_MOL &system,
                                                  const MOL &mol) {
  std::vector<PbctopoRange> ranges =
      build_ranges_from_index(system.Index, mol.pos.x.size());
  if (!ranges.empty())
    return ranges;

  ranges = build_ranges_from_info(system.info, mol.pos.x.size());
  if (!ranges.empty())
    return ranges;

  if (!mol.pos.x.empty())
    ranges.push_back({0, mol.pos.x.size(), 0, false});
  return ranges;
}

void build_heavy_atom_subset(const std::vector<PbctopoAtom> &atoms,
                             std::vector<PbctopoAtom> &heavy_atoms) {
  heavy_atoms.clear();
  heavy_atoms.reserve(atoms.size());
  for (const auto &atom : atoms) {
    if (atom.is_heavy)
      heavy_atoms.push_back(atom);
  }
  if (heavy_atoms.empty())
    heavy_atoms = atoms;
}

PbctopoSystemStaticData
build_pbctopo_system_static_data(const Multi_MOL &system, const MOL &mol) {
  PbctopoSystemStaticData data;
  data.atom_count = mol.pos.x.size();
  data.topology_graph_available = system.topology.available();
  data.topology_bonds_available =
      data.topology_graph_available ? system.topology.bonds().size() : 0;
  build_projected_topology_bonds(system.topology, data.atom_count,
                                 data.projected_topology_bonds);
  data.ranges = build_ranges_for_system(system, mol);
  data.anchor_unit_ranges =
      build_ranges_from_topology(system.topology, mol.pos.x.size());
  if (data.anchor_unit_ranges.empty())
    data.anchor_unit_ranges =
        build_ranges_from_info(system.info, mol.pos.x.size());
  if (data.anchor_unit_ranges.empty())
    data.anchor_unit_ranges = data.ranges;
  data.heavy_mask.assign(data.atom_count, 1);
  data.vdw_radii.assign(data.atom_count, 0.0);
  data.vdw_radius_available.assign(data.atom_count, 0);
  data.vdw_radius_sources.assign(
      data.atom_count, PbctopoStericRadiusSource::Unavailable);
  data.atomic_masses.assign(data.atom_count, 0.0);
  data.atomic_mass_available.assign(data.atom_count, 0);
  data.heavy_count = 0;
  for (std::size_t atom_idx = 0; atom_idx < data.atom_count; ++atom_idx) {
    const bool is_heavy = detect_heavy_atom(&system.info, atom_idx);
    data.heavy_mask[atom_idx] = is_heavy ? 1 : 0;
    data.heavy_count += is_heavy ? 1u : 0u;
    if (const auto mass = system.topology.atomic_mass(static_cast<std::uint32_t>(atom_idx));
        mass.has_value()) {
      data.atomic_masses[atom_idx] = *mass;
      data.atomic_mass_available[atom_idx] = 1;
      ++data.atomic_mass_count;
    }
    if (const auto radius = system.topology.vdw_radius(
            static_cast<std::uint32_t>(atom_idx));
        radius.has_value()) {
      data.vdw_radii[atom_idx] = *radius;
      data.vdw_radius_available[atom_idx] = 1;
      const auto radius_source = system.topology.vdw_radius_source(
          static_cast<std::uint32_t>(atom_idx));
      const bool topology_source =
          radius_source == MolTopologyVdwRadiusSource::GromacsTopology ||
          radius_source ==
              MolTopologyVdwRadiusSource::GromacsTopologyExplicitZeroLJ;
      const bool explicit_zero_lj =
          radius_source ==
              MolTopologyVdwRadiusSource::GromacsTopologyExplicitZeroLJ ||
          radius_source ==
              MolTopologyVdwRadiusSource::MultiMolParametersExplicitZeroLJ;
      data.vdw_radius_sources[atom_idx] =
          explicit_zero_lj
              ? PbctopoStericRadiusSource::ExplicitZeroLennardJones
              : topology_source
                    ? PbctopoStericRadiusSource::ForceFieldTopology
                    : PbctopoStericRadiusSource::ForceFieldParameter;
      if (topology_source) {
        ++data.vdw_topology_count;
      } else {
        ++data.vdw_parameter_count;
      }
      if (*radius == 0.0)
        ++data.vdw_zero_lj_count;
    }
    if (data.vdw_radius_available[atom_idx] == 0 &&
        atom_idx < system.parameters.size()) {
      const double radius = system.parameters[atom_idx].get_radius();
      if (std::isfinite(radius) && radius > 0.0) {
        data.vdw_radii[atom_idx] = radius;
        data.vdw_radius_available[atom_idx] = 1;
        data.vdw_radius_sources[atom_idx] =
            PbctopoStericRadiusSource::ForceFieldParameter;
        ++data.vdw_parameter_count;
      }
    }
    if (data.vdw_radius_available[atom_idx] == 0 &&
        atom_idx < system.info.size()) {
      const PbctopoInferredRadius inferred =
          infer_bondi_vdw_radius(system.info, atom_idx);
      data.vdw_radii[atom_idx] = inferred.radius;
      data.vdw_radius_available[atom_idx] = 1;
      data.vdw_radius_sources[atom_idx] =
          inferred.element_supported
              ? PbctopoStericRadiusSource::InferredElement
              : PbctopoStericRadiusSource::GenericFallback;
      if (inferred.element_supported)
        ++data.vdw_inferred_count;
      else
        ++data.vdw_generic_fallback_count;
    }
  }
  if (data.heavy_count == 0 && data.atom_count > 0) {
    std::fill(data.heavy_mask.begin(), data.heavy_mask.end(), 1);
    data.heavy_count = data.atom_count;
  }
  if (!data.topology_graph_available)
    build_projected_metadata_graph(system, mol, data);
  PbctopoSystemStaticData continuity_data = data;
  continuity_data.ranges = data.anchor_unit_ranges;
  data.continuity_pairs_local =
      build_local_continuity_pairs(system, continuity_data);
  initialize_continuity_reference_lengths(mol,
                                          data.continuity_pairs_local);
  if (data.continuity_anchor_local.empty()) {
    build_metadata_anchor_index_list(system, continuity_data,
                                     data.continuity_anchor_local);
  }
  if (data.continuity_anchor_local.empty()) {
    build_anchor_index_list(data.atom_count, data.continuity_pairs_local,
                            data.continuity_anchor_local);
  }
  return data;
}

PbctopoTrajectoryContext
build_pbctopo_trajectory_context(const Multi_MOL &mol1_system,
                                 const Multi_MOL &mol2_system,
                                 std::size_t frame_count) {
  PbctopoTrajectoryContext context;
  if (frame_count == 0 || mol1_system.m_mol.empty())
    return context;

  const MOL &rec = mol1_system.m_mol.front();
  context.rec_static = build_pbctopo_system_static_data(mol1_system, rec);
  std::size_t lig_atoms = 0;
  std::size_t lig_heavy = 0;
  if (!mol2_system.m_mol.empty() && !mol2_system.m_mol.front().pos.x.empty()) {
    const MOL &lig = mol2_system.m_mol.front();
    context.lig_static = build_pbctopo_system_static_data(mol2_system, lig);
    lig_atoms = context.lig_static.atom_count;
    lig_heavy = context.lig_static.heavy_count;
  }

  context.total_atoms_hint =
      std::max<std::size_t>(1, context.rec_static.atom_count + lig_atoms);
  context.heavy_atoms_hint =
      std::max<std::size_t>(1, context.rec_static.heavy_count + lig_heavy);
  context.local_edges_hint =
      std::max<std::size_t>(1024, context.total_atoms_hint * 8);
  context.projection_edges_hint =
      std::max<std::size_t>(1024, context.heavy_atoms_hint * 8);
  append_global_continuity_pairs(context.rec_static.continuity_pairs_local, 0,
                                 context.continuity_pairs);
  append_global_continuity_pairs(context.lig_static.continuity_pairs_local,
                                 context.rec_static.atom_count,
                                 context.continuity_pairs);
  append_global_anchor_indices(context.rec_static.continuity_anchor_local, 0,
                               context.continuity_anchor_indices);
  append_global_anchor_indices(context.lig_static.continuity_anchor_local,
                               context.rec_static.atom_count,
                               context.continuity_anchor_indices);
  context.continuity_pair_hint =
      std::max<std::size_t>(64, context.continuity_pairs.size() + 1);
  context.component_hint =
      std::max<std::size_t>(8, mol1_system.Index.size() + mol2_system.Index.size());
  context.plane_hint = std::max<std::size_t>(32, context.heavy_atoms_hint * 2 + 4);
  context.virtual_state_hint = 512;
  context.coarse_seed_hint = 32;
  context.coarse_candidate_hint = 16;
  return context;
}

void build_projection_edge_subset(
    const std::vector<PbctopoAtom> &base_atoms,
    const std::vector<PbctopoAtom> &projection_atoms,
    const std::vector<PbctopoLocalEdge> &local_edges,
    std::vector<PbctopoLocalEdge> &projection_edges,
    std::unordered_set<std::uint64_t> &projection_edge_set,
    std::vector<std::size_t> &projection_index) {
  projection_edges.clear();
  projection_edge_set.clear();
  if (projection_atoms.size() == base_atoms.size()) {
    projection_edges = local_edges;
    projection_edge_set.reserve(local_edges.size() * 2 + 1);
    for (const auto &edge : projection_edges)
      projection_edge_set.insert(atom_pair_key(edge.a, edge.b));
    return;
  }

  const std::size_t invalid_index = std::numeric_limits<std::size_t>::max();
  projection_index.assign(base_atoms.size(), invalid_index);
  for (std::size_t idx = 0; idx < projection_atoms.size(); ++idx)
    projection_index[projection_atoms[idx].global_atom_id] = idx;

  projection_edges.reserve(local_edges.size());
  projection_edge_set.reserve(local_edges.size() + 1);
  for (const auto &edge : local_edges) {
    const std::size_t mapped_a =
        projection_index[base_atoms[edge.a].global_atom_id];
    const std::size_t mapped_b =
        projection_index[base_atoms[edge.b].global_atom_id];
    if (mapped_a == invalid_index || mapped_b == invalid_index)
      continue;
    if (!projection_edge_set.insert(atom_pair_key(mapped_a, mapped_b)).second)
      continue;
    projection_edges.push_back({mapped_a, mapped_b, edge.ref_d2,
                                edge.inv_ref_d2, edge.break_cutoff_d2,
                                edge.kind});
  }
}

void build_projection_continuity_subset(
    const std::vector<PbctopoAtom> &base_atoms,
    const std::vector<PbctopoAtom> &projection_atoms,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    std::vector<PbctopoContinuityPair> &projection_continuity_pairs,
    std::vector<std::size_t> &projection_index) {
  projection_continuity_pairs.clear();
  if (projection_atoms.size() == base_atoms.size()) {
    projection_continuity_pairs = continuity_pairs;
    return;
  }
  const std::size_t invalid_index = std::numeric_limits<std::size_t>::max();
  if (projection_index.size() != base_atoms.size())
    projection_index.assign(base_atoms.size(), invalid_index);
  for (const auto &pair : continuity_pairs) {
    const std::size_t mapped_a =
        projection_index[base_atoms[pair.a].global_atom_id];
    const std::size_t mapped_b =
        projection_index[base_atoms[pair.b].global_atom_id];
    if (mapped_a == invalid_index || mapped_b == invalid_index)
      continue;
    projection_continuity_pairs.push_back(
        {mapped_a, mapped_b, pair.ref_d2, pair.inv_ref_d2, pair.kind,
         pair.weight, pair.hard_cutoff_d2});
  }
}

void build_projection_anchor_subset(
    const std::vector<PbctopoAtom> &base_atoms,
    const std::vector<std::size_t> &anchor_indices,
    std::vector<std::size_t> &projection_anchor_indices,
    const std::vector<std::size_t> &projection_index) {
  projection_anchor_indices.clear();
  if (anchor_indices.empty())
    return;
  if (projection_index.size() != base_atoms.size()) {
    projection_anchor_indices = anchor_indices;
    return;
  }
  const std::size_t invalid_index = std::numeric_limits<std::size_t>::max();
  projection_anchor_indices.reserve(anchor_indices.size());
  for (std::size_t anchor_idx : anchor_indices) {
    if (anchor_idx >= base_atoms.size())
      continue;
    const std::size_t mapped =
        projection_index[base_atoms[anchor_idx].global_atom_id];
    if (mapped == invalid_index)
      continue;
    projection_anchor_indices.push_back(mapped);
  }
}

void append_atoms_for_system(const MOL &mol,
                             const std::vector<mol_info> *info,
                             const PbctopoSystemStaticData *static_data,
                             int owner,
                             std::vector<PbctopoAtom> &atoms) {
  std::size_t unit_range_index = 0;
  for (std::size_t local = 0; local < mol.pos.x.size(); ++local) {
    double r[3]{mol.pos.x[local], mol.pos.y[local], mol.pos.z[local]};
    double s[3]{};
    mol.Box.to_scaled(r, s);
    s[0] = wrap_unit(s[0]);
    s[1] = wrap_unit(s[1]);
    s[2] = wrap_unit(s[2]);
    scaled_to_cartesian(mol.Box, s[0], s[1], s[2], r[0], r[1], r[2]);

    PbctopoAtom atom;
    atom.global_atom_id = atoms.size();
    bool matched_anchor_unit = false;
    if (static_data != nullptr) {
      const auto &unit_ranges = static_data->anchor_unit_ranges;
      atom.fragment_id = unit_ranges.size();
      while (unit_range_index < unit_ranges.size() &&
             local >= unit_ranges[unit_range_index].end) {
        ++unit_range_index;
      }
      if (unit_range_index < unit_ranges.size() &&
          local >= unit_ranges[unit_range_index].begin &&
          local < unit_ranges[unit_range_index].end) {
        atom.fragment_id = unit_ranges[unit_range_index].source_index;
        matched_anchor_unit = true;
      }
    }
    atom.owner = owner;
    atom.local_atom_index = local;
    atom.source_atom_id = local;
    if (info != nullptr && local < info->size()) {
      const auto &atom_info = (*info)[local];
      if (atom_info.has_original_atom_id())
        atom.source_atom_id = atom_info.get_original_atom_id();
      const std::size_t chainnum = atom_info.get_chainnum();
      if (static_data != nullptr && !matched_anchor_unit)
        atom.fragment_id = static_data->anchor_unit_ranges.size() + chainnum;
      atom.chain_key = static_cast<std::uint64_t>(chainnum);
      atom.residue_id = static_cast<int>(atom_info.get_internal_resnr());
      atom.has_topology_key = true;
    }
    atom.has_topology_graph =
        static_data != nullptr && static_data->topology_graph_available;
    atom.is_heavy =
        (static_data != nullptr && local < static_data->heavy_mask.size())
            ? (static_data->heavy_mask[local] != 0)
            : detect_heavy_atom(info, local);
    if (static_data != nullptr && local < static_data->vdw_radii.size() &&
        local < static_data->vdw_radius_available.size() &&
        static_data->vdw_radius_available[local] != 0) {
      atom.vdw_radius = static_data->vdw_radii[local];
      atom.has_vdw_radius = true;
      if (local < static_data->vdw_radius_sources.size())
        atom.steric_radius_source = static_data->vdw_radius_sources[local];
    }
    if (static_data != nullptr &&
        local < static_data->atomic_masses.size() &&
        local < static_data->atomic_mass_available.size() &&
        static_data->atomic_mass_available[local] != 0) {
      atom.atomic_mass = static_data->atomic_masses[local];
      atom.has_atomic_mass = true;
    }
    atom.x = r[0];
    atom.y = r[1];
    atom.z = r[2];
    atom.sx = s[0];
    atom.sy = s[1];
    atom.sz = s[2];
    atoms.push_back(atom);
  }
}

void build_local_neighborhood_graph(
    const std::vector<PbctopoAtom> &atoms, box Box,
    std::vector<PbctopoLocalEdge> &edges,
    std::unordered_set<std::uint64_t> &edge_set, double bond_cutoff,
    bool append, std::span<const char> eligible_atoms) {
  if (!append) {
    edges.clear();
    edge_set.clear();
  }
  if (atoms.size() < 2)
    return;
  const PbctopoLatticeMetric basis(Box);
  if (!basis.valid())
    return;

  const double kBondCutoff = bond_cutoff;
  const double kBondCutoff2 = bond_cutoff * bond_cutoff;
  const auto is_eligible = [&](std::size_t atom_idx) {
    return eligible_atoms.empty() ||
           (atom_idx < eligible_atoms.size() && eligible_atoms[atom_idx] != 0);
  };
  edges.reserve(edges.size() + atoms.size() * 2);
  edge_set.reserve(edge_set.size() + atoms.size() * 2);
  std::vector<std::size_t> degree(atoms.size(), 0);
  if (append) {
    for (const auto &edge : edges) {
      if (edge.a >= atoms.size() || edge.b >= atoms.size() || edge.a == edge.b)
        continue;
      ++degree[edge.a];
      ++degree[edge.b];
    }
  }
  auto same_local_unit = [](const PbctopoAtom &lhs, const PbctopoAtom &rhs) {
    if (lhs.owner != rhs.owner)
      return false;
    if (lhs.has_topology_key && rhs.has_topology_key)
      return lhs.chain_key == rhs.chain_key;
    return true;
  };
  auto add_edge = [&](std::size_t a, std::size_t b, double d2) {
    const std::uint64_t pair_key = atom_pair_key(a, b);
    if (!edge_set.insert(pair_key).second)
      return false;
    const double ref_d2 = std::max(d2, 1.0e-6);
    edges.push_back({a, b, ref_d2, 1.0 / ref_d2,
                     std::max(9.0, ref_d2 * 4.0),
                     PbctopoLocalEdgeKind::GeometryHeuristic});
    ++degree[a];
    ++degree[b];
    return true;
  };
  const BoxMatrix box_matrix(Box);
  const double sx_bound = std::min(
      0.5, kBondCutoff *
                    std::sqrt(box_matrix.inv[0][0] * box_matrix.inv[0][0] +
                              box_matrix.inv[1][0] * box_matrix.inv[1][0] +
                              box_matrix.inv[2][0] * box_matrix.inv[2][0]) +
               1.0e-9);
  const double sy_bound = std::min(
      0.5, kBondCutoff *
                    std::sqrt(box_matrix.inv[0][1] * box_matrix.inv[0][1] +
                              box_matrix.inv[1][1] * box_matrix.inv[1][1] +
                              box_matrix.inv[2][1] * box_matrix.inv[2][1]) +
               1.0e-9);
  const double sz_bound = std::min(
      0.5, kBondCutoff *
                    std::sqrt(box_matrix.inv[0][2] * box_matrix.inv[0][2] +
                              box_matrix.inv[1][2] * box_matrix.inv[1][2] +
                              box_matrix.inv[2][2] * box_matrix.inv[2][2]) +
               1.0e-9);
  const int nx =
      std::max(1, static_cast<int>(std::floor(1.0 / std::max(sx_bound, 1.0e-6))));
  const int ny =
      std::max(1, static_cast<int>(std::floor(1.0 / std::max(sy_bound, 1.0e-6))));
  const int nz =
      std::max(1, static_cast<int>(std::floor(1.0 / std::max(sz_bound, 1.0e-6))));
  const double cell_sx = 1.0 / static_cast<double>(nx);
  const double cell_sy = 1.0 / static_cast<double>(ny);
  const double cell_sz = 1.0 / static_cast<double>(nz);
  auto wrap_index = [](int value, int mod) {
    int wrapped = value % mod;
    if (wrapped < 0)
      wrapped += mod;
    return wrapped;
  };

  std::unordered_map<ClashCellKey, std::vector<std::size_t>, ClashCellKeyHash>
      bond_grid;
  bond_grid.reserve(std::max<std::size_t>(64, atoms.size() / 2));
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    if (!is_eligible(i))
      continue;
    const auto &ai = atoms[i];
    const int cx = std::min(nx - 1, static_cast<int>(std::floor(ai.sx / cell_sx)));
    const int cy = std::min(ny - 1, static_cast<int>(std::floor(ai.sy / cell_sy)));
    const int cz = std::min(nz - 1, static_cast<int>(std::floor(ai.sz / cell_sz)));
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          const ClashCellKey key{wrap_index(cx + dx, nx),
                                 wrap_index(cy + dy, ny),
                                 wrap_index(cz + dz, nz)};
          const auto it = bond_grid.find(key);
          if (it == bond_grid.end())
            continue;
          for (std::size_t j : it->second) {
            if (!is_eligible(j))
              continue;
            const auto &aj = atoms[j];
            if (!same_local_unit(ai, aj))
                continue;
            double dsx = aj.sx - ai.sx;
            double dsy = aj.sy - ai.sy;
            double dsz = aj.sz - ai.sz;
            const PbctopoNearestImage mic =
                basis.nearest_image_scaled(dsx, dsy, dsz);
            if (!mic.valid)
              continue;
            const double d2 = mic.d2;
            if (d2 > kBondCutoff2)
              continue;
            add_edge(j, i, d2);
          }
        }
      }
    }
    bond_grid[{cx, cy, cz}].push_back(i);
  }

}

void append_topology_bond_edges_for_system(
    const PbctopoSystemStaticData &static_data, std::size_t atom_offset,
    const std::vector<PbctopoAtom> &atoms, box Box,
    std::vector<PbctopoLocalEdge> &edges,
    std::unordered_set<std::uint64_t> &edge_set, PbctopoTimingStats &timing) {
  if (!static_data.topology_graph_available &&
      static_data.topology_bonds_available == 0) {
    return;
  }

  timing.pbctopo_topology_bonds_available +=
      static_data.topology_bonds_available;
  timing.pbctopo_topology_projected_bonds +=
      static_data.projected_topology_bonds.size();
  if (static_data.topology_bonds_available >
      static_data.projected_topology_bonds.size()) {
    timing.pbctopo_topology_bond_edges_skipped +=
        static_data.topology_bonds_available -
        static_data.projected_topology_bonds.size();
  }
  if (static_data.projected_topology_bonds.empty())
    return;

  const PbctopoLatticeMetric basis(Box);
  for (const PbctopoProjectedTopologyBond &bond :
       static_data.projected_topology_bonds) {
    const std::size_t a = atom_offset + bond.a;
    const std::size_t b = atom_offset + bond.b;
    if (a >= atoms.size() || b >= atoms.size() || a == b) {
      ++timing.pbctopo_topology_bond_edges_skipped;
      continue;
    }

    double dsx = atoms[b].sx - atoms[a].sx;
    double dsy = atoms[b].sy - atoms[a].sy;
    double dsz = atoms[b].sz - atoms[a].sz;
    const PbctopoNearestImage mic =
        basis.nearest_image_scaled(dsx, dsy, dsz);
    if (!mic.valid || !std::isfinite(mic.d2)) {
      ++timing.pbctopo_topology_bond_edges_skipped;
      continue;
    }
    const std::uint64_t pair_key = atom_pair_key(a, b);
    if (!edge_set.insert(pair_key).second) {
      ++timing.pbctopo_topology_bond_edges_skipped;
      ++timing.pbctopo_topology_duplicate_edges;
      continue;
    }
    const double d2 = mic.d2;
    const double ref_d2 = std::max(d2, 1.0e-6);
    edges.push_back({a, b, ref_d2, 1.0 / ref_d2,
                     std::max(9.0, ref_d2 * 4.0),
                     PbctopoLocalEdgeKind::TopologyBond});
    ++timing.pbctopo_topology_bond_edges_added;
  }
}

void append_metadata_graph_edges_for_system(
    const PbctopoSystemStaticData &static_data, std::size_t atom_offset,
    const std::vector<PbctopoAtom> &atoms,
    std::vector<PbctopoLocalEdge> &edges,
    std::unordered_set<std::uint64_t> &edge_set, PbctopoTimingStats &timing) {
  if (!static_data.metadata_graph_available)
    return;

  timing.pbctopo_metadata_residue_groups +=
      static_data.metadata_residue_groups;
  timing.pbctopo_metadata_intra_edges += static_data.metadata_intra_edges;
  timing.pbctopo_metadata_inter_edges += static_data.metadata_inter_edges;
  timing.pbctopo_metadata_inter_hard_edges +=
      static_data.metadata_inter_hard_edges;
  timing.pbctopo_metadata_inter_soft_edges +=
      static_data.metadata_inter_soft_edges;
  if (static_data.projected_metadata_edges.empty())
    return;

  constexpr double neutral_ref = 0.0;
  const double neutral_break = std::numeric_limits<double>::infinity();
  for (const PbctopoProjectedMetadataEdge &edge :
       static_data.projected_metadata_edges) {
    if (!edge.hard_connectivity)
      continue;
    const std::size_t a = atom_offset + edge.a;
    const std::size_t b = atom_offset + edge.b;
    if (a >= atoms.size() || b >= atoms.size() || a == b)
      continue;
    if (!edge_set.insert(atom_pair_key(a, b)).second)
      continue;
    edges.push_back({a, b, neutral_ref, 0.0, neutral_break,
                     PbctopoLocalEdgeKind::MetadataSkeleton});
  }
}

std::vector<std::vector<std::size_t>>
build_hard_connected_components_from_local_edges(
    std::size_t atom_count, const std::vector<PbctopoLocalEdge> &edges) {
  std::vector<std::vector<std::size_t>> adjacency(atom_count);
  for (const auto &edge : edges) {
    if (!pbctopo_edge_is_hard_connectivity(edge.kind) ||
        edge.a >= atom_count || edge.b >= atom_count || edge.a == edge.b) {
      continue;
    }
    adjacency[edge.a].push_back(edge.b);
    adjacency[edge.b].push_back(edge.a);
  }

  std::vector<std::vector<std::size_t>> components;
  std::vector<char> seen(atom_count, 0);
  std::vector<std::size_t> stack;
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    if (seen[atom] != 0)
      continue;
    seen[atom] = 1;
    stack.clear();
    stack.push_back(atom);
    std::vector<std::size_t> component;
    while (!stack.empty()) {
      const std::size_t current = stack.back();
      stack.pop_back();
      component.push_back(current);
      for (std::size_t neighbor : adjacency[current]) {
        if (seen[neighbor] != 0)
          continue;
        seen[neighbor] = 1;
        stack.push_back(neighbor);
      }
    }
    std::sort(component.begin(), component.end());
    components.push_back(std::move(component));
  }

  std::sort(components.begin(), components.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.size() != rhs.size())
                return lhs.size() > rhs.size();
              return lhs.front() < rhs.front();
            });
  return components;
}

std::string pbctopo_topology_source_label(const PbctopoTimingStats &timing) {
  if (timing.pbctopo_metadata_graph_frames > 0 &&
      timing.pbctopo_topology_static_graph_frames > 0)
    return "mixed";
  if (timing.frames > 0 &&
      timing.pbctopo_topology_static_graph_frames == timing.frames)
    return "gromacs_topology";
  if (timing.frames > 0 &&
      timing.pbctopo_metadata_graph_frames == timing.frames)
    return "metadata_chain_residue";
  if (timing.pbctopo_metadata_graph_frames > 0 &&
      timing.pbctopo_topology_bonds_available > 0)
    return "mixed";
  if (timing.pbctopo_metadata_graph_frames > 0 &&
      timing.pbctopo_geometry_fallback_frames > 0)
    return "mixed";
  if (timing.pbctopo_metadata_graph_frames > 0)
    return "metadata_chain_residue";
  if (timing.pbctopo_topology_bonds_available == 0)
    return "geometry_cutoff";
  if (timing.pbctopo_geometry_fallback_frames > 0)
    return "mixed";
  return "gromacs_topology";
}

} // namespace titan_pbctopo
