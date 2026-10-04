#ifndef TITAN_PBCTOPO_LLPS_LOCAL_H
#define TITAN_PBCTOPO_LLPS_LOCAL_H

#include "types.h"
#include "../box_type.h"

#include <cstddef>
#include <string>
#include <vector>

class Multi_MOL;

namespace titan_pbctopo {

struct PbctopoLlpsLocalUnits {
  std::vector<std::vector<std::size_t>> units;
  std::vector<int> atom_unit;
  std::string source = "none";
};

std::string merge_pbctopo_llps_unit_source(const std::string &current,
                                           const std::string &next);

PbctopoLlpsLocalUnits build_pbctopo_llps_local_units(
    const Multi_MOL &mol1_system, const Multi_MOL &mol2_system,
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<PbctopoLocalEdge> &edges);

std::vector<Vec3> unwrap_pbctopo_llps_local_points(
    const std::vector<PbctopoAtom> &wrapped_atoms,
    const std::vector<PbctopoLocalEdge> &edges, box Box,
    const PbctopoLlpsLocalUnits &units, PbctopoTimingStats &timing);

} // namespace titan_pbctopo

#endif
