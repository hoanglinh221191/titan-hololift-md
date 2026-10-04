#ifndef TITAN_PBCTOPO_LOCAL_GRAPH_H
#define TITAN_PBCTOPO_LOCAL_GRAPH_H

#include "types.h"
#include "../box_type.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

class MOL;
class Multi_MOL;
class mol_info;

namespace titan_pbctopo {

struct PbctopoRange {
  std::size_t begin = 0;
  std::size_t end = 0;
  std::size_t source_index = 0;
  bool topo_backed = false;
};

struct PbctopoProjectedTopologyBond {
  std::size_t a = 0;
  std::size_t b = 0;
};

struct PbctopoProjectedMetadataEdge {
  std::size_t a = 0;
  std::size_t b = 0;
  bool inter_residue = false;
  bool hard_connectivity = true;
};

struct PbctopoSystemStaticData {
  std::size_t atom_count = 0;
  std::size_t heavy_count = 0;
  std::vector<char> heavy_mask;
  std::vector<double> vdw_radii;
  std::vector<char> vdw_radius_available;
  std::vector<PbctopoStericRadiusSource> vdw_radius_sources;
  std::vector<double> atomic_masses;
  std::vector<char> atomic_mass_available;
  std::size_t atomic_mass_count = 0;
  std::size_t vdw_topology_count = 0;
  std::size_t vdw_parameter_count = 0;
  std::size_t vdw_zero_lj_count = 0;
  std::size_t vdw_inferred_count = 0;
  std::size_t vdw_generic_fallback_count = 0;
  std::vector<PbctopoRange> ranges;
  std::vector<PbctopoRange> anchor_unit_ranges;
  bool topology_graph_available = false;
  std::size_t topology_bonds_available = 0;
  std::vector<PbctopoProjectedTopologyBond> projected_topology_bonds;
  bool metadata_graph_available = false;
  std::size_t metadata_residue_groups = 0;
  std::size_t metadata_intra_edges = 0;
  std::size_t metadata_inter_edges = 0;
  std::size_t metadata_inter_hard_edges = 0;
  std::size_t metadata_inter_soft_edges = 0;
  std::vector<PbctopoProjectedMetadataEdge> projected_metadata_edges;
  std::vector<PbctopoContinuityPair> continuity_pairs_local;
  std::vector<std::size_t> continuity_anchor_local;
};

struct PbctopoTrajectoryContext {
  std::size_t total_atoms_hint = 0;
  std::size_t heavy_atoms_hint = 0;
  std::size_t local_edges_hint = 0;
  std::size_t projection_edges_hint = 0;
  std::size_t component_hint = 0;
  std::size_t plane_hint = 0;
  std::size_t virtual_state_hint = 512;
  std::size_t coarse_seed_hint = 32;
  std::size_t coarse_candidate_hint = 16;
  std::size_t continuity_pair_hint = 0;
  PbctopoSystemStaticData rec_static;
  PbctopoSystemStaticData lig_static;
  std::vector<PbctopoContinuityPair> continuity_pairs;
  std::vector<std::size_t> continuity_anchor_indices;
};

std::vector<PbctopoRange> build_ranges_for_system(const Multi_MOL &system,
                                                  const MOL &mol);

void build_heavy_atom_subset(const std::vector<PbctopoAtom> &atoms,
                             std::vector<PbctopoAtom> &heavy_atoms);

PbctopoSystemStaticData
build_pbctopo_system_static_data(const Multi_MOL &system, const MOL &mol);

PbctopoTrajectoryContext
build_pbctopo_trajectory_context(const Multi_MOL &mol1_system,
                                 const Multi_MOL &mol2_system,
                                 std::size_t frame_count);

void build_projection_edge_subset(
    const std::vector<PbctopoAtom> &base_atoms,
    const std::vector<PbctopoAtom> &projection_atoms,
    const std::vector<PbctopoLocalEdge> &local_edges,
    std::vector<PbctopoLocalEdge> &projection_edges,
    std::unordered_set<std::uint64_t> &projection_edge_set,
    std::vector<std::size_t> &projection_index);

void build_projection_continuity_subset(
    const std::vector<PbctopoAtom> &base_atoms,
    const std::vector<PbctopoAtom> &projection_atoms,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    std::vector<PbctopoContinuityPair> &projection_continuity_pairs,
    std::vector<std::size_t> &projection_index);

void build_projection_anchor_subset(
    const std::vector<PbctopoAtom> &base_atoms,
    const std::vector<std::size_t> &anchor_indices,
    std::vector<std::size_t> &projection_anchor_indices,
    const std::vector<std::size_t> &projection_index);

void append_atoms_for_system(const MOL &mol,
                             const std::vector<mol_info> *info,
                             const PbctopoSystemStaticData *static_data,
                             int owner,
                             std::vector<PbctopoAtom> &atoms);

void build_local_neighborhood_graph(
    const std::vector<PbctopoAtom> &atoms, box Box,
    std::vector<PbctopoLocalEdge> &edges,
    std::unordered_set<std::uint64_t> &edge_set, double bond_cutoff,
    bool append = false, std::span<const char> eligible_atoms = {});

std::vector<std::vector<std::size_t>>
build_hard_connected_components_from_local_edges(
    std::size_t atom_count, const std::vector<PbctopoLocalEdge> &edges);

void append_topology_bond_edges_for_system(
    const PbctopoSystemStaticData &static_data, std::size_t atom_offset,
    const std::vector<PbctopoAtom> &atoms,
    box Box, std::vector<PbctopoLocalEdge> &edges,
    std::unordered_set<std::uint64_t> &edge_set, PbctopoTimingStats &timing);

void append_metadata_graph_edges_for_system(
    const PbctopoSystemStaticData &static_data, std::size_t atom_offset,
    const std::vector<PbctopoAtom> &atoms,
    std::vector<PbctopoLocalEdge> &edges,
    std::unordered_set<std::uint64_t> &edge_set, PbctopoTimingStats &timing);

std::string pbctopo_topology_source_label(const PbctopoTimingStats &timing);

} // namespace titan_pbctopo

#endif
