#ifndef TITAN_GBSA_GB_TOPOLOGY_PAIRS_H
#define TITAN_GBSA_GB_TOPOLOGY_PAIRS_H

// ENE-22: topology exclusions and listed (1-4) pairs between the two MM/GBSA
// groups.
//
// The fused GBSA kernel sums Coulomb and Lennard-Jones over every
// receptor x ligand pair. When the two groups are joined by covalent bonds,
// the topology excludes the 1-2/1-3 pairs across the junction and lists the
// 1-4 pairs with their own parameters. This module collects those few pairs
// once per run, by local index, so the kernel can replace their plain terms:
// excluded pairs lose the plain Coulomb/LJ term and listed pairs gain their
// listed term (plain Coulomb with the listed charge product, as GROMACS
// evaluates them). The GB polar term keeps every pair, bonded or not, as in
// Amber's GB energy.

#include "MOL_info.h"
#include "topology/nonbonded_topology.h"

#include <cstddef>
#include <vector>

namespace titan_gbsa {

struct TopologyPair {
  std::size_t rec{};   // index in the receptor (mol1) selection
  std::size_t lig{};   // index in the ligand (mol2) selection
  bool excluded{};     // the plain Coulomb/LJ term is removed
  double listed_c6{};  // listed pair, kJ mol^-1 nm^6
  double listed_c12{}; // listed pair, kJ mol^-1 nm^12
  double listed_qq{};  // listed charge product, e^2
};

struct TopologyPairs {
  std::size_t rec_size{};
  std::size_t lig_size{};
  std::size_t excluded{};
  std::size_t listed{};
  std::vector<TopologyPair> pairs; // sorted by (rec, lig)
};

// Pairs (receptor atom, ligand atom) that the topology excludes or lists.
// Atoms are matched through their source atom identity (original_atom_id).
[[nodiscard]] TopologyPairs
build_topology_pairs(const titan_nb::Topology &topology,
                     const std::vector<mol_info> &rec_info,
                     const std::vector<mol_info> &lig_info);

} // namespace titan_gbsa

#endif
