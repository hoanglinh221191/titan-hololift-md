#ifndef TITAN_PBCTOPO_IMAGE_ASSIGNMENT_H
#define TITAN_PBCTOPO_IMAGE_ASSIGNMENT_H

#include "types.h"

#include <span>
#include <vector>

namespace titan_pbctopo {

PbctopoImageAssignment make_pbctopo_image_assignment(
    std::span<const Int3> component_offsets,
    std::span<const std::vector<std::size_t>> components);

PbctopoImageAssignment make_pbctopo_image_assignment(
    std::span<const Int3> component_offsets,
    std::span<const std::vector<std::size_t>> components,
    std::span<const PbctopoAtom> atoms);

bool same_pbctopo_image_assignment(const PbctopoImageAssignment &lhs,
                                    const PbctopoImageAssignment &rhs);

} // namespace titan_pbctopo

#endif
