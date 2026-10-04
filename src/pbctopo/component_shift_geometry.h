#ifndef TITAN_PBCTOPO_COMPONENT_SHIFT_GEOMETRY_H
#define TITAN_PBCTOPO_COMPONENT_SHIFT_GEOMETRY_H

#include "types.h"

#include <cstddef>
#include <vector>

namespace titan_pbctopo {

PbctopoComponentBaseGeometry pbctopo_build_component_base_geometry(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::size_t> &component);

PbctopoComponentBaseGeometry pbctopo_build_component_all_atom_base_geometry(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::size_t> &component);

PbctopoShiftedComponentDescriptor pbctopo_describe_shifted_component(
    const PbctopoComponentBaseGeometry &base, const PbctopoShift &shift);

void pbctopo_materialize_shifted_component(
    const PbctopoComponentBaseGeometry &base, const PbctopoShift &shift,
    double clash_cutoff, double contact_cutoff, bool build_clash_grid,
    bool build_contact_grid,
    PbctopoShiftedComponent &shifted);

double pbctopo_shifted_bbox_distance2(
    const PbctopoShiftedComponentDescriptor &lhs,
    const PbctopoShiftedComponentDescriptor &rhs);

} // namespace titan_pbctopo

#endif
