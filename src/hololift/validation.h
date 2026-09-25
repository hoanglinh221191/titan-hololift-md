#ifndef TITAN_HOLOLIFT_VALIDATION_H
#define TITAN_HOLOLIFT_VALIDATION_H

#include "types.h"

#include <expected>
#include <string>
#include <string_view>

namespace titan_hololift {

enum class HoloLiftValidationCode : std::uint8_t {
  UnsupportedApiVersion,
  InvalidSourceContract,
  EmptyObservationStore,
  InvalidRange,
  InvalidTopologyEpoch,
  DuplicateTopologyEpoch,
  InvalidComponentMembership,
  DuplicateAtomMembership,
  ComponentIdentityMismatch,
  ComponentIdentityCollision,
  LayoutIdentityMismatch,
  LayoutIdentityCollision,
  TopologyEpochIdentityMismatch,
  InvalidFrameOrder,
  InvalidFrameTime,
  InvalidFrameProvenance,
  InvalidFrameBinding,
  InvalidSpatialLift,
  MissingTopologyEpoch,
  InvalidGramMatrix,
  InvalidCandidateSet,
  AssignmentLayoutMismatch,
  AssignmentComponentMismatch,
  AssignmentGaugeMismatch,
  AssignmentIdentityMismatch,
  ObservationClassMismatch,
};

struct HoloLiftValidationError {
  HoloLiftValidationCode code =
      HoloLiftValidationCode::InvalidSourceContract;
  std::size_t epoch_index = HOLOLIFT_NO_INDEX;
  std::size_t frame_index = HOLOLIFT_NO_INDEX;
  std::size_t candidate_index = HOLOLIFT_NO_INDEX;
  std::string message;
};

std::string_view
hololift_validation_code_name(HoloLiftValidationCode code) noexcept;

std::expected<void, HoloLiftValidationError>
validate_hololift_source_contract(
    const HoloLiftObservationStoreBuilder &builder);

std::expected<void, HoloLiftValidationError>
validate_hololift_topology_epoch(
    const HoloLiftObservationStoreBuilder &builder,
    std::size_t epoch_index);

std::expected<void, HoloLiftValidationError>
validate_hololift_frame(const HoloLiftObservationStoreBuilder &builder,
                        std::size_t frame_index);

std::expected<void, HoloLiftValidationError>
validate_hololift_observation_store(
    const HoloLiftObservationStoreBuilder &builder);

std::expected<HoloLiftObservationStore, HoloLiftValidationError>
finalize_hololift_observation_store(
    HoloLiftObservationStoreBuilder &&builder);

} // namespace titan_hololift

#endif
