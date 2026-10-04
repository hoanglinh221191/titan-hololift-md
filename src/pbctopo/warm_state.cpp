#include "warm_state.h"

#include <cstdint>
#include <limits>
#include <utility>

namespace titan_pbctopo {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_word(std::uint64_t value, std::uint64_t &hash) {
  for (unsigned int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8U)) & 0xffU;
    hash *= kFnvPrime;
  }
}

bool same_offsets(const std::vector<Int3> &lhs,
                  const std::vector<Int3> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (std::size_t idx = 0; idx < lhs.size(); ++idx) {
    if (lhs[idx].x != rhs[idx].x || lhs[idx].y != rhs[idx].y ||
        lhs[idx].z != rhs[idx].z) {
      return false;
    }
  }
  return true;
}

} // namespace

PbctopoComponentLayoutSignature build_component_layout_signature(
    const std::vector<std::vector<std::size_t>> &components) {
  PbctopoComponentLayoutSignature signature;
  if (components.empty())
    return signature;

  signature.component_hashes.reserve(components.size());
  std::size_t atom_count = 0;
  for (std::size_t component_idx = 0; component_idx < components.size();
       ++component_idx) {
    const auto &component = components[component_idx];
    if (component.empty() ||
        component.size() > std::numeric_limits<std::size_t>::max() -
                               atom_count) {
      return {};
    }
    atom_count += component.size();
    std::uint64_t hash = kFnvOffset;
    hash_word(static_cast<std::uint64_t>(component.size()), hash);
    for (std::size_t atom_idx : component)
      hash_word(static_cast<std::uint64_t>(atom_idx), hash);
    signature.component_hashes.push_back(hash);
  }
  signature.atom_count = atom_count;
  signature.valid = true;
  return signature;
}

bool same_component_layout(
    const std::vector<std::vector<std::size_t>> &components,
    const PbctopoComponentLayoutSignature &signature) {
  return signature.valid &&
         build_component_layout_signature(components) == signature;
}

bool pbctopo_warm_state_has_data(const PbctopoWarmState &warm_state) {
  return warm_state.has_projection_candidate ||
         warm_state.has_kinematic_previous_virtual_candidate ||
         warm_state.has_kinematic_latest_virtual_candidate ||
         warm_state.has_component_offsets ||
         warm_state.has_component_trial_hints;
}

void reset_pbctopo_warm_state(PbctopoWarmState &warm_state) {
  warm_state = {};
}

bool commit_pbctopo_certified_warm_state(
    PbctopoWarmState &warm_state,
    const PbctopoWarmState &frame_start_state,
    const PbctopoWarmState &trial_state,
    const PbctopoCutCandidate &selected_candidate,
    const PbctopoComponentLayoutSignature &component_layout) {
  if (!selected_candidate.valid || !selected_candidate.assignment.valid ||
      !selected_candidate.hard_feasibility.certified() ||
      !component_layout.valid ||
      selected_candidate.assignment.component_offsets.size() !=
          component_layout.component_hashes.size()) {
    return false;
  }

  PbctopoWarmState committed;
  committed.previous_projection_candidate = selected_candidate;
  committed.has_projection_candidate = true;
  committed.previous_component_layout = component_layout;
  committed.previous_component_offsets =
      selected_candidate.assignment.component_offsets;
  committed.has_component_offsets = true;

  committed.kinematic_previous_virtual_candidate =
      frame_start_state.kinematic_previous_virtual_candidate;
  committed.kinematic_latest_virtual_candidate =
      frame_start_state.kinematic_latest_virtual_candidate;
  committed.kinematic_previous_virtual_frame =
      frame_start_state.kinematic_previous_virtual_frame;
  committed.kinematic_latest_virtual_frame =
      frame_start_state.kinematic_latest_virtual_frame;
  committed.has_kinematic_previous_virtual_candidate =
      frame_start_state.has_kinematic_previous_virtual_candidate;
  committed.has_kinematic_latest_virtual_candidate =
      frame_start_state.has_kinematic_latest_virtual_candidate;

  const bool trial_hints_match_selected =
      trial_state.has_component_trial_hints &&
      trial_state.previous_component_layout == component_layout &&
      same_offsets(trial_state.previous_component_offsets,
                   selected_candidate.assignment.component_offsets);
  if (trial_hints_match_selected) {
    committed.previous_stage1_delta_hints_by_component =
        trial_state.previous_stage1_delta_hints_by_component;
    committed.previous_stage2_delta_hints_by_component =
        trial_state.previous_stage2_delta_hints_by_component;
    committed.has_component_trial_hints = true;
  }

  warm_state = std::move(committed);
  return true;
}

} // namespace titan_pbctopo
