#ifndef TITAN_HOLOLIFT_SERIALIZATION_H
#define TITAN_HOLOLIFT_SERIALIZATION_H

#include "artifact_write.h"
#include "types.h"

#include <expected>
#include <filesystem>
#include <string>

namespace titan_hololift {

inline constexpr std::uint32_t HOLOLIFT_BINARY_FORMAT_VERSION = 10;

struct HoloLiftBinaryReadLimits {
  std::uint64_t max_artifact_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
  std::size_t max_string_bytes = 1024ULL * 1024ULL;
  std::size_t max_topology_epochs = 1000000ULL;
  std::size_t max_components = 10000000ULL;
  std::size_t max_atom_memberships = 200000000ULL;
  std::size_t max_hard_edges = 400000000ULL;
  std::size_t max_frames = 10000000ULL;
  std::size_t max_candidates = 100000000ULL;
  std::size_t max_component_images = 500000000ULL;
};

// On failure, inspect error().commit_state before retrying: the final path may
// already contain the new artifact with durability still unconfirmed.
[[nodiscard]]
HoloLiftArtifactWriteResult write_hololift_observation_store_binary(
    const HoloLiftObservationStore &store,
    const std::filesystem::path &path);

// SHA-256 of the exact canonical payload bytes used by the binary artifact.
std::expected<std::string, std::string>
hololift_observation_store_payload_sha256(
    const HoloLiftObservationStore &store);

std::expected<HoloLiftObservationStore, std::string>
read_hololift_observation_store_binary(const std::filesystem::path &path);

std::expected<HoloLiftObservationStore, std::string>
read_hololift_observation_store_binary(
    const std::filesystem::path &path,
    const HoloLiftBinaryReadLimits &limits);

} // namespace titan_hololift

#endif
