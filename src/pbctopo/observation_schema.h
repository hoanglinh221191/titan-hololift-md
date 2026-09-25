#ifndef TITAN_PBCTOPO_OBSERVATION_SCHEMA_H
#define TITAN_PBCTOPO_OBSERVATION_SCHEMA_H

#include "../titan_version_info.h"

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

namespace titan_pbctopo {

inline constexpr std::uint32_t VIBE_OBSERVATION_SCHEMA_VERSION = 7;
inline constexpr std::string_view VIBE_OBSERVATION_COORDINATE_UNIT =
    "angstrom";
inline constexpr std::string_view VIBE_OBSERVATION_TIME_UNIT = "ps";
inline constexpr std::string_view VIBE_OBSERVATION_ENDIANNESS =
    "not_applicable_text";
inline constexpr std::string_view VIBE_OBSERVATION_FLOATING_FORMAT =
    "IEEE754-binary64";
inline constexpr std::string_view VIBE_OBSERVATION_IDENTITY_HASH_ALGORITHM =
    "sha256-128";
inline constexpr std::string_view VIBE_OBSERVATION_BINDING_HASH_ALGORITHM =
    "sha256-128";
inline constexpr std::string_view VIBE_OBSERVATION_PAYLOAD_CHECKSUM_ALGORITHM =
    "sha256-256";
inline constexpr std::string_view VIBE_OBSERVATION_BINDING_MODE =
    "bit_exact";
inline constexpr std::string_view VIBE_OBSERVATION_NEGATIVE_ZERO_POLICY =
    "canonicalized_to_positive_zero";
inline constexpr std::string_view VIBE_OBSERVATION_NONFINITE_POLICY =
    "forbidden";
inline constexpr std::string_view VIBE_OBSERVATION_SCOPE =
    "framewise_bounded_domain";
inline constexpr std::uint64_t
    VIBE_OBSERVATION_ATOM_UNIVERSE_HASH_DOMAIN = 0x504243554e493031ULL;

inline const char *vibe_observation_producer_version() noexcept {
  return titan_build::full_version();
}

std::expected<void, std::string>
validate_vibe_observation_schema_version(std::uint32_t version);

} // namespace titan_pbctopo

#endif
