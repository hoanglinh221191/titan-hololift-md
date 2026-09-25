#ifndef TITAN_HOLOLIFT_VIBE_ADAPTER_H
#define TITAN_HOLOLIFT_VIBE_ADAPTER_H

#include "types.h"

#include "../pbctopo/observation_layout.h"
#include "../pbctopo/report.h"

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace titan_hololift {

std::expected<std::uint32_t, std::string>
append_hololift_vibe_topology_epoch(
    HoloLiftObservationStoreBuilder &builder,
    const titan_pbctopo::PbctopoComponentLayoutObservation &layout);

std::expected<HoloLiftFrameProvenance, std::string>
hololift_vibe_frame_provenance(
    const titan_pbctopo::PbctopoFrameReport &report,
    std::uint64_t framewise_assignment_signature);

inline std::expected<std::uint32_t, std::string>
append_hololift_vibe_v1_topology_epoch(
    HoloLiftObservationStoreBuilder &builder,
    const titan_pbctopo::PbctopoComponentLayoutObservation &layout) {
  return append_hololift_vibe_topology_epoch(builder, layout);
}

inline std::expected<HoloLiftFrameProvenance, std::string>
hololift_vibe_v1_frame_provenance(
    const titan_pbctopo::PbctopoFrameReport &report,
    std::uint64_t framewise_assignment_signature) {
  return hololift_vibe_frame_provenance(
      report, framewise_assignment_signature);
}

struct HoloLiftVibeCandidateAppendInput {
  HoloLiftCandidateRecord record;
  std::vector<HoloLiftComponentImage> raw_component_images;
};

struct HoloLiftVibeFrameAppendInput {
  HoloLiftFrameRecord frame;
  std::vector<HoloLiftVibeCandidateAppendInput> candidates;
};

// Canonicalizes and validates a complete frame in temporary storage. The
// destination builder is changed only after every candidate and cross-field
// invariant has passed.
std::expected<std::size_t, std::string> append_hololift_vibe_frame(
    HoloLiftObservationStoreBuilder &builder,
    HoloLiftVibeFrameAppendInput input);

} // namespace titan_hololift

#endif
