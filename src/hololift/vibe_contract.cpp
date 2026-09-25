#include "vibe_contract.h"

#include "../pbctopo/observation_schema.h"

#include <utility>

namespace titan_hololift {

HoloLiftSourceContract
make_hololift_vibe_contract(std::string producer_version,
                            std::string package_id,
                            std::string trajectory_id) {
  HoloLiftSourceContract contract;
  contract.schema_version =
      titan_pbctopo::VIBE_OBSERVATION_SCHEMA_VERSION;
  contract.producer_version = std::move(producer_version);
  contract.package_id = std::move(package_id);
  contract.trajectory_id = std::move(trajectory_id);
  contract.coordinate_unit =
      titan_pbctopo::VIBE_OBSERVATION_COORDINATE_UNIT;
  contract.time_unit = titan_pbctopo::VIBE_OBSERVATION_TIME_UNIT;
  contract.endianness = titan_pbctopo::VIBE_OBSERVATION_ENDIANNESS;
  contract.floating_format =
      titan_pbctopo::VIBE_OBSERVATION_FLOATING_FORMAT;
  contract.identity_hash_algorithm =
      titan_pbctopo::VIBE_OBSERVATION_IDENTITY_HASH_ALGORITHM;
  contract.binding_hash_algorithm =
      titan_pbctopo::VIBE_OBSERVATION_BINDING_HASH_ALGORITHM;
  contract.payload_checksum_algorithm =
      titan_pbctopo::VIBE_OBSERVATION_PAYLOAD_CHECKSUM_ALGORITHM;
  contract.binding_mode = titan_pbctopo::VIBE_OBSERVATION_BINDING_MODE;
  contract.negative_zero_policy =
      titan_pbctopo::VIBE_OBSERVATION_NEGATIVE_ZERO_POLICY;
  contract.nonfinite_policy =
      titan_pbctopo::VIBE_OBSERVATION_NONFINITE_POLICY;
  contract.observation_scope =
      titan_pbctopo::VIBE_OBSERVATION_SCOPE;
  return contract;
}

std::expected<void, std::string>
validate_hololift_vibe_contract(
    const HoloLiftSourceContract &contract) {
  const auto schema =
      titan_pbctopo::validate_vibe_observation_schema_version(
          contract.schema_version);
  if (!schema)
    return std::unexpected(schema.error());
  if (contract.producer_version.empty())
    return std::unexpected("VIBE observation producer version is empty");
  if (contract.package_id.empty())
    return std::unexpected("VIBE observation package id is empty");
  if (contract.trajectory_id.empty())
    return std::unexpected("VIBE observation trajectory id is empty");
  if (contract.coordinate_unit !=
      titan_pbctopo::VIBE_OBSERVATION_COORDINATE_UNIT) {
    return std::unexpected(
        "VIBE observation coordinate unit must be angstrom");
  }
  if (contract.time_unit != titan_pbctopo::VIBE_OBSERVATION_TIME_UNIT)
    return std::unexpected("VIBE observation time unit must be ps");
  if (contract.endianness != titan_pbctopo::VIBE_OBSERVATION_ENDIANNESS) {
    return std::unexpected(
        "VIBE text observation endianness marker is invalid");
  }
  if (contract.floating_format !=
          titan_pbctopo::VIBE_OBSERVATION_FLOATING_FORMAT ||
      contract.identity_hash_algorithm !=
          titan_pbctopo::VIBE_OBSERVATION_IDENTITY_HASH_ALGORITHM ||
      contract.binding_hash_algorithm !=
          titan_pbctopo::VIBE_OBSERVATION_BINDING_HASH_ALGORITHM ||
      contract.payload_checksum_algorithm !=
          titan_pbctopo::VIBE_OBSERVATION_PAYLOAD_CHECKSUM_ALGORITHM ||
      contract.binding_mode != titan_pbctopo::VIBE_OBSERVATION_BINDING_MODE ||
      contract.negative_zero_policy !=
          titan_pbctopo::VIBE_OBSERVATION_NEGATIVE_ZERO_POLICY ||
      contract.nonfinite_policy !=
          titan_pbctopo::VIBE_OBSERVATION_NONFINITE_POLICY) {
    return std::unexpected(
        "HoloLift requires IEEE-754 binary64 bit-exact frame binding");
  }
  if (contract.observation_scope !=
      titan_pbctopo::VIBE_OBSERVATION_SCOPE) {
    return std::unexpected(
        "HoloLift Phase 0 accepts framewise_bounded_domain observations only");
  }
  return {};
}

std::expected<HoloLiftObservationClass, std::string>
parse_hololift_vibe_observation_class(std::string_view value) {
  if (value == "not_hard_feasible")
    return HoloLiftObservationClass::NotHardFeasible;
  if (value == "weak_observation")
    return HoloLiftObservationClass::WeakObservation;
  if (value == "ambiguous_anchor")
    return HoloLiftObservationClass::AmbiguousAnchor;
  if (value == "strong_bounded_anchor")
    return HoloLiftObservationClass::StrongBoundedAnchor;
  if (value == "strong_global_anchor") {
    return std::unexpected(
        "strong_global_anchor is unsupported without an outside-domain "
        "infeasibility proof");
  }
  return std::unexpected("unknown VIBE observation class: " +
                         std::string(value));
}

std::string_view hololift_vibe_observation_class_name(
    HoloLiftObservationClass value) noexcept {
  switch (value) {
  case HoloLiftObservationClass::NotHardFeasible:
    return "not_hard_feasible";
  case HoloLiftObservationClass::WeakObservation:
    return "weak_observation";
  case HoloLiftObservationClass::AmbiguousAnchor:
    return "ambiguous_anchor";
  case HoloLiftObservationClass::StrongBoundedAnchor:
    return "strong_bounded_anchor";
  case HoloLiftObservationClass::StrongGlobalAnchor:
    return "strong_global_anchor";
  }
  return "not_hard_feasible";
}

} // namespace titan_hololift
