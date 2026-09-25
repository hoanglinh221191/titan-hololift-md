#ifndef TITAN_PBCTOPO_OBSERVATION_BINDING_H
#define TITAN_PBCTOPO_OBSERVATION_BINDING_H

#include "types.h"

#include "../box_type.h"

#include <array>
#include <cstddef>
#include <span>
#include <string>

namespace titan_pbctopo {

struct PbctopoFrameBindingObservation {
  std::size_t trajectory_frame_index = 0;
  std::array<double, 9> box_matrix{};
  PbctopoHash128 box_hash;
  PbctopoHash128 atom_selection_hash;
  PbctopoHash128 canonical_atom_universe_hash;
  PbctopoHash128 wrapped_coordinate_hash;
  bool valid = false;
};

PbctopoFrameBindingObservation build_pbctopo_frame_binding_observation(
    std::size_t trajectory_frame_index, box Box,
    std::span<const PbctopoAtom> wrapped_atoms);

PbctopoHash128
hash_pbctopo_box_matrix(std::span<const double, 9> box_matrix) noexcept;

PbctopoHash128 hash_pbctopo_canonical_atom_universe(
    std::span<const PbctopoAtom> atoms);

std::string format_pbctopo_hash128(PbctopoHash128 hash);

} // namespace titan_pbctopo

#endif
