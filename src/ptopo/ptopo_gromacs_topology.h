#ifndef TITAN_PTOPO_GROMACS_TOPOLOGY_H
#define TITAN_PTOPO_GROMACS_TOPOLOGY_H

// GROMACS .top/.itp reader for the hard topology of the periodic graph.
//
// Reads [moleculetype], [atoms], [bonds], [constraints], [settles],
// [virtual_sites1/2/3/4/n] and [molecules] through a preprocessor with
// GROMACS semantics: #include,
// #define, #undef, #ifdef, #ifndef, #else, #endif. Only the active branch
// is read. (mol_topology.cpp counts #if depth but reads both branches, which
// for a Martini itp merges the FLEXIBLE bonds with the constraints and any
// RUBBER_BANDS block whether or not it is defined.) #if/#elif and a missing
// include fail closed. Parameters and force-field tables are not parsed.

#include "ptopo_types.h"

#include <filesystem>

namespace titan_ptopo {

struct TopologyAtom {
  std::string type;
  std::string residue_name;
  std::string atom_name;
  std::int64_t residue_number = 0;
  double charge = 0.0;
  // NaN when the [atoms] line has no mass column (type masses not resolved).
  double mass = std::numeric_limits<double>::quiet_NaN();
};

enum class TopologyLinkKind : std::uint8_t {
  Bond,       // [bonds]
  Constraint, // [constraints]
  Settle,     // [settles]: O-H1, O-H2
  // [virtual_sites*]: site to each constructing atom. A virtual site is
  // rigidly attached; without this link it would be its own component.
  VirtualSite,
};

struct TopologyBond {
  std::uint32_t a = 0; // 0-based, local to the molecule type
  std::uint32_t b = 0;
  int function = 0; // funct column; 0 for [settles]
  TopologyLinkKind kind = TopologyLinkKind::Bond;
};

struct TopologyMoleculeType {
  std::string name;
  int nrexcl = 0;
  std::vector<TopologyAtom> atoms;
  std::vector<TopologyBond> bonds;
  std::string source; // file that defined it
};

struct TopologyMoleculeBlock {
  std::string name;
  std::uint64_t count = 0;
};

struct GromacsTopology {
  std::vector<TopologyMoleculeType> molecule_types;
  std::vector<TopologyMoleculeBlock> molecules; // [molecules], in order
  std::vector<std::string> files;               // read order
  // Constructs read but deliberately not used (reported, never silent).
  std::vector<std::string> notes;
};

struct TopologyReadOptions {
  // "NAME", "NAME=VALUE" or "-DNAME[=VALUE]", as the mdp `define` line.
  std::vector<std::string> defines;
  // Searched after the including file's directory and GMXLIB.
  std::vector<std::filesystem::path> include_directories;
  std::size_t max_include_depth = 64;
};

Result<GromacsTopology> read_gromacs_topology(
    const std::filesystem::path &topology, const TopologyReadOptions &options = {});

} // namespace titan_ptopo

#endif // TITAN_PTOPO_GROMACS_TOPOLOGY_H
