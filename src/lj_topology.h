#ifndef LJ_TOPOLOGY_H
#define LJ_TOPOLOGY_H

#include <expected>
#include <string>

struct LjTopologySummary {
  std::string primary_molecule;
  size_t atoms_per_molecule = 0;
  size_t molecule_count = 0;
  size_t total_atoms = 0;
};

std::expected<LjTopologySummary, std::string>
parse_lj_topology_summary(const std::string &topo_path);

#endif
