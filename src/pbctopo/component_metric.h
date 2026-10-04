#ifndef TITAN_PBCTOPO_COMPONENT_METRIC_H
#define TITAN_PBCTOPO_COMPONENT_METRIC_H

#include "types.h"
#include "../box_type.h"

#include <vector>

namespace titan_pbctopo {

double pbctopo_translation_norm2(const PbctopoShift &shift) noexcept;
double pbctopo_translation_norm2(box Box, const Int3 &offset);

// Chooses the integer global gauge minimizing the mean Cartesian displacement.
Int3 pbctopo_component_offset_gauge(box Box,
                                    const std::vector<Int3> &offsets);
bool rebase_component_offsets(std::vector<Int3> &offsets, box Box);

// Gauge-invariant, component-balanced mean squared Cartesian displacement.
double pbctopo_component_shift_metric(box Box,
                                      const std::vector<Int3> &offsets);

} // namespace titan_pbctopo

#endif
