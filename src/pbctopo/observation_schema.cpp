#include "observation_schema.h"

namespace titan_pbctopo {

std::expected<void, std::string>
validate_vibe_observation_schema_version(std::uint32_t version) {
  // Version 8 only appends columns and states to version 7, so packages
  // written by either can be read.
  if (version >= VIBE_OBSERVATION_MIN_READABLE_SCHEMA_VERSION &&
      version <= VIBE_OBSERVATION_SCHEMA_VERSION)
    return {};
  return std::unexpected(
      "unsupported VIBE observation schema version " +
      std::to_string(version) + "; supported versions are " +
      std::to_string(VIBE_OBSERVATION_MIN_READABLE_SCHEMA_VERSION) + " to " +
      std::to_string(VIBE_OBSERVATION_SCHEMA_VERSION));
}

} // namespace titan_pbctopo
