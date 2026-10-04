#ifndef TITAN_PBCTOPO_TEMPORAL_REPLAY_H
#define TITAN_PBCTOPO_TEMPORAL_REPLAY_H

#include "types.h"
#include "../box_type.h"

#include <cstdint>
#include <span>
#include <vector>

namespace titan_pbctopo {

bool replay_pbctopo_assignment_delta(
    std::span<const Vec3> applied_points,
    std::span<const std::uint32_t> atom_component_index,
    const PbctopoImageAssignment &applied_assignment,
    const PbctopoImageAssignment &selected_assignment, box Box,
    std::vector<Vec3> &replayed_points);

} // namespace titan_pbctopo

#endif
