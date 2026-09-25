#include "spatial_lift.h"

#include "../pbctopo/spatial_lift.h"
#include "frame_binding.h"

namespace titan_hololift {

std::expected<HoloLiftSpatialLiftReplay, std::string>
replay_hololift_spatial_lift(
    const HoloLiftTopologyEpochRecord &epoch,
    const HoloLiftFrameRecord &frame,
    std::span<const HoloLiftHardEdgeRecord> hard_edges,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates) {
  if (!frame.spatial_lift_valid)
    return std::unexpected("HoloLift frame has no replayable spatial lift");
  if (frame.spatial_lift_policy_version !=
      titan_pbctopo::PBCTOPO_SPATIAL_LIFT_POLICY_VERSION) {
    return std::unexpected("unsupported HoloLift spatial-lift policy version");
  }
  if (atom_order.size() != wrapped_coordinates.size()) {
    return std::unexpected(
        "HoloLift spatial-lift atom and coordinate counts differ");
  }
  if (atom_order.size() != epoch.atom_count ||
      hard_edges.size() != epoch.hard_edges.count) {
    return std::unexpected(
        "HoloLift spatial-lift inputs do not match topology-epoch sizes");
  }
  if (frame.topology_epoch_id != epoch.topology_epoch_id ||
      frame.topology_epoch_identity != epoch.topology_epoch_identity) {
    return std::unexpected(
        "HoloLift spatial-lift frame does not reference this topology epoch");
  }
  const auto atom_universe_hash =
      hash_hololift_canonical_atom_universe(atom_order);
  if (!atom_universe_hash)
    return std::unexpected(atom_universe_hash.error());
  if (*atom_universe_hash != epoch.canonical_atom_universe_hash ||
      *atom_universe_hash != frame.canonical_atom_universe_hash) {
    return std::unexpected(
        "HoloLift spatial-lift atom universe does not match topology epoch");
  }

  std::vector<titan_pbctopo::PbctopoSpatialLiftAtom> atoms;
  atoms.reserve(atom_order.size());
  for (std::size_t idx = 0; idx < atom_order.size(); ++idx) {
    atoms.push_back({atom_order[idx].owner, atom_order[idx].source_atom_id,
                     wrapped_coordinates[idx]});
  }
  std::vector<titan_pbctopo::PbctopoSpatialLiftEdge> edges;
  edges.reserve(hard_edges.size());
  for (const auto &edge : hard_edges)
    edges.push_back({edge.atom_a, edge.atom_b});

  const auto replay = titan_pbctopo::build_pbctopo_spatial_lift(
      frame.box_matrix, atoms, edges);
  if (!replay.valid)
    return std::unexpected(replay.error);

  HoloLiftSpatialLiftReplay result;
  result.atom_images.reserve(replay.atom_images.size());
  for (const auto &image : replay.atom_images)
    result.atom_images.push_back({image.x, image.y, image.z});
  result.identity = {replay.identity.lo, replay.identity.hi};
  result.ambiguous_hard_edges = replay.ambiguous_hard_edges;
  result.cycle_residuals = replay.cycle_residuals;
  return result;
}

std::expected<void, std::string> validate_hololift_spatial_lift_replay(
    const HoloLiftTopologyEpochRecord &epoch,
    const HoloLiftFrameRecord &frame,
    std::span<const HoloLiftHardEdgeRecord> hard_edges,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates) {
  const auto replay = replay_hololift_spatial_lift(
      epoch, frame, hard_edges, atom_order, wrapped_coordinates);
  if (!replay)
    return std::unexpected(replay.error());
  if (replay->identity != frame.spatial_lift_identity)
    return std::unexpected("HoloLift spatial-lift identity mismatch");
  if (replay->ambiguous_hard_edges !=
      frame.spatial_lift_ambiguous_hard_edges) {
    return std::unexpected("HoloLift spatial-lift ambiguity count mismatch");
  }
  if (replay->cycle_residuals != frame.spatial_lift_cycle_residuals)
    return std::unexpected("HoloLift spatial-lift cycle audit mismatch");
  return {};
}

} // namespace titan_hololift
