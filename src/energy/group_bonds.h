#ifndef TITAN_ENERGY_GROUP_BONDS_H
#define TITAN_ENERGY_GROUP_BONDS_H

// ENE-22: covalent bonds between the two energy groups.
//
// The lib_ff route and the MM/GBSA route compute inter-group Coulomb and
// Lennard-Jones terms over every mol1 x mol2 pair, with no bonded exclusions
// and no 1-4 scaling. When the two groups are joined by a covalent bond
// (a glycan or a residue peptide-bonded to the protein, a disulfide between
// the groups) the bonded pairs enter the sums at 1.3-1.5 A and E_vdW reaches
// 1e5 kJ/mol without any warning. This scan finds such bonds on one frame so
// the run can be refused instead. With -common_topo the topology energies and
// MM/GBSA (gbsa/gb_topology_pairs.h) apply the topology's exclusions, and the
// scan is not needed.

#include "MOL.h"
#include "MOL_info.h"
#include "titan_executor.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace titan_group_bonds {

inline constexpr std::string_view kBondedGroupsError = "invalid_bonded_groups";

struct BondedPair {
  std::size_t mol1_atom{};
  std::size_t mol2_atom{};
  double distance_angstrom{};
};

struct BondScan {
  std::size_t bonds{0};
  std::vector<BondedPair> examples; // the first few, by mol1 atom
};

// Covalent radius (angstrom) of an atom from its PDB name, or 0 when the
// element is not one of H, C, N, O, S, P (metals, halogens, CG beads and
// unknown names are skipped: their contacts are not covalent bonds the
// nonbonded kernels need to exclude).
[[nodiscard]] double covalent_radius_from_name(std::string_view atom_name) noexcept;

// Pairs (mol1 atom, mol2 atom) closer than 1.2 x the sum of the covalent
// radii, on raw coordinates (the coordinates the energy kernels use).
// Atoms of single-atom residues (ions) are skipped.
[[nodiscard]] BondScan find_covalent_bonds(const MOL &mol1,
                                           const std::vector<mol_info> &info1,
                                           const MOL &mol2,
                                           const std::vector<mol_info> &info2,
                                           TitanExecutor &executor,
                                           std::size_t max_examples = 4);

// "mol1 A:342:PHE C - mol2 C:343:NLN N 1.337 A; ..." for the examples.
[[nodiscard]] std::string describe_examples(const BondScan &scan,
                                            const std::vector<mol_info> &info1,
                                            const std::vector<mol_info> &info2);

// The invalid_bonded_groups message when the groups are bonded, else "".
[[nodiscard]] std::string bonded_groups_error(const MOL &mol1,
                                              const std::vector<mol_info> &info1,
                                              const MOL &mol2,
                                              const std::vector<mol_info> &info2,
                                              TitanExecutor &executor);

} // namespace titan_group_bonds

#endif
