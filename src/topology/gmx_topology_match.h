#ifndef TITAN_TOPOLOGY_GMX_TOPOLOGY_MATCH_H
#define TITAN_TOPOLOGY_GMX_TOPOLOGY_MATCH_H

// ENE-12 (22/10/2026). How the atoms of a structure are paired with the
// [ atoms ] rows of a GROMACS topology (Multi_MOL::post_process_atom_from_topo):
//
//  - same atom count: by position, after checking that the names agree; a
//    structure whose atom names differ from the topology at more than
//    kMaxNameMismatchFraction of the atoms (another atom order, appended
//    hydrogens, another protein of the same size) is refused instead of
//    receiving the charges of other atoms;
//  - different counts: by residue name and atom name, first unused topology
//    atom in file order (the former O(N_pdb x N_topo) search, now hashed);
//    more than kMaxNameMismatchFraction unmatched atoms is refused, fewer are
//    reported (they keep charge 0).

#include "miscellaneous.h"
#include "MOL_info.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace TitanTopo {

inline constexpr double kMaxNameMismatchFraction = 0.05;

// Comparison key of an atom name: trimmed, upper case, '*' as '\'', a
// leading digit moved to the end (PDB "1HD1" = GROMACS "HD11").
std::string atom_name_key(std::string_view name);

// The same residue under PDB and topology naming: equal names, an N/C
// terminal prefix (NALA/ALA), or protonation variants of one residue (the
// histidine names of every family, ASP/ASH, GLU/GLH, LYS/LYN, CYS/CYX/CYM).
bool residue_names_compatible(std::string_view pdb, std::string_view topology);

struct NameCheck {
  std::size_t atoms{};
  std::size_t atom_mismatches{};
  std::size_t residue_mismatches{};
  std::vector<std::string> examples;   // first ten atom-name mismatches
};

// Position-by-position comparison of equally long lists.
NameCheck compare_positional_names(const std::vector<mol_info> &structure,
                                   const std::vector<topo_info_temp> &topology);

// Index of the topology atom for every structure atom (-1 when none), by the
// rule of the different-count branch above.
std::vector<long long> match_by_names(const std::vector<mol_info> &structure,
                                      const std::vector<topo_info_temp> &topology);

} // namespace TitanTopo

#endif // TITAN_TOPOLOGY_GMX_TOPOLOGY_MATCH_H
