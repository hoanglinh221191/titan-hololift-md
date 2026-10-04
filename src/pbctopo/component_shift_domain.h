#ifndef TITAN_PBCTOPO_COMPONENT_SHIFT_DOMAIN_H
#define TITAN_PBCTOPO_COMPONENT_SHIFT_DOMAIN_H

#include "../box_type.h"
#include "types.h"

#include <cstddef>
#include <vector>

namespace titan_pbctopo {

struct PbctopoRelativeShiftMultiplicity {
  PbctopoShift shift;
  std::size_t multiplicity = 0;
};

std::vector<PbctopoShift> build_component_image_shifts(
    box Box, int shell_radius = 1);

std::vector<PbctopoRelativeShiftMultiplicity>
build_component_relative_shift_domain(
    const std::vector<PbctopoShift> &shifts,
    bool *used_analytic_construction = nullptr);

} // namespace titan_pbctopo

#endif
