#include "observation_schema.h"

namespace titan_pbctopo {

std::expected<void, std::string>
validate_vibe_observation_schema_version(std::uint32_t version) {
  if (version == VIBE_OBSERVATION_SCHEMA_VERSION)
    return {};
  return std::unexpected(
      "unsupported VIBE observation schema version " +
      std::to_string(version) + "; supported version is " +
      std::to_string(VIBE_OBSERVATION_SCHEMA_VERSION));
}

} // namespace titan_pbctopo
