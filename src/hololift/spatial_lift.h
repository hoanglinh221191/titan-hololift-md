#ifndef TITAN_HOLOLIFT_SPATIAL_LIFT_H
#define TITAN_HOLOLIFT_SPATIAL_LIFT_H

#include "types.h"

#include <array>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace titan_hololift {

struct HoloLiftSpatialLiftReplay {
  std::vector<HoloLiftLatticeImage> atom_images;
  HoloLiftHash128 identity;
  std::size_t ambiguous_hard_edges = 0;
  std::size_t cycle_residuals = 0;
};

std::expected<HoloLiftSpatialLiftReplay, std::string>
replay_hololift_spatial_lift(
    const HoloLiftTopologyEpochRecord &epoch,
    const HoloLiftFrameRecord &frame,
    std::span<const HoloLiftHardEdgeRecord> hard_edges,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates);

std::expected<void, std::string> validate_hololift_spatial_lift_replay(
    const HoloLiftTopologyEpochRecord &epoch,
    const HoloLiftFrameRecord &frame,
    std::span<const HoloLiftHardEdgeRecord> hard_edges,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates);

} // namespace titan_hololift

#endif
