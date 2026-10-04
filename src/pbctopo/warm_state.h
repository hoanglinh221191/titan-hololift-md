#ifndef TITAN_PBCTOPO_WARM_STATE_H
#define TITAN_PBCTOPO_WARM_STATE_H

#include "types.h"

#include <cstddef>
#include <vector>

namespace titan_pbctopo {

PbctopoComponentLayoutSignature build_component_layout_signature(
    const std::vector<std::vector<std::size_t>> &components);

bool same_component_layout(
    const std::vector<std::vector<std::size_t>> &components,
    const PbctopoComponentLayoutSignature &signature);

bool pbctopo_warm_state_has_data(const PbctopoWarmState &warm_state);

void reset_pbctopo_warm_state(PbctopoWarmState &warm_state);

bool commit_pbctopo_certified_warm_state(
    PbctopoWarmState &warm_state,
    const PbctopoWarmState &frame_start_state,
    const PbctopoWarmState &trial_state,
    const PbctopoCutCandidate &selected_candidate,
    const PbctopoComponentLayoutSignature &component_layout);

} // namespace titan_pbctopo

#endif
