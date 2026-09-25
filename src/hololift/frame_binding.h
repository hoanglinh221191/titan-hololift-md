#ifndef TITAN_HOLOLIFT_FRAME_BINDING_H
#define TITAN_HOLOLIFT_FRAME_BINDING_H

#include "types.h"

#include <array>
#include <expected>
#include <span>
#include <string>
#include <string_view>

namespace titan_hololift {

HoloLiftHash128
hash_hololift_box_matrix(std::span<const double, 9> box_matrix) noexcept;

HoloLiftHash128 hash_hololift_atom_selection(
    std::span<const HoloLiftSourceAtomKey> atom_order) noexcept;

std::expected<HoloLiftHash128, std::string>
hash_hololift_canonical_atom_universe(
    std::span<const HoloLiftSourceAtomKey> atom_order);

HoloLiftHash128 hash_hololift_wrapped_coordinates(
    std::span<const std::array<double, 3>> coordinates) noexcept;

// Recompute the external-trajectory binding before a HoloLift consumer uses
// an observation. The coordinate and atom-order spans must describe the same
// selected atoms in the exact order used by VIBE, and their canonical key set
// must match the topology epoch.
std::expected<void, std::string> validate_hololift_frame_binding(
    const HoloLiftTopologyEpochRecord &epoch,
    const HoloLiftFrameRecord &observation,
    std::size_t trajectory_frame_index,
    std::span<const double, 9> box_matrix,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates);

std::string format_hololift_hash128(HoloLiftHash128 hash);

std::expected<HoloLiftHash128, std::string>
parse_hololift_hash128(std::string_view value);

} // namespace titan_hololift

#endif
