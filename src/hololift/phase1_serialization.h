#ifndef TITAN_HOLOLIFT_PHASE1_SERIALIZATION_H
#define TITAN_HOLOLIFT_PHASE1_SERIALIZATION_H

#include "artifact_write.h"
#include "phase1.h"

#include <expected>
#include <filesystem>
#include <string>

namespace titan_hololift {

inline constexpr std::uint32_t HOLOLIFT_PHASE1_RESULT_FORMAT_VERSION = 7;

// Writes a deterministic, atomically replaced JSON audit artifact. Only a
// sealed all-atom result is accepted; its SHA-256 digest binds trajectory-frame
// hashes, selected assignment identities, and the complete audit hierarchy.
// On failure, inspect error().commit_state before deciding whether to retry.
[[nodiscard]]
HoloLiftArtifactWriteResult write_hololift_phase1_result_json(
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    const HoloLiftPhase1AuditedResult &audited_result,
    const std::filesystem::path &path);

} // namespace titan_hololift

#endif
