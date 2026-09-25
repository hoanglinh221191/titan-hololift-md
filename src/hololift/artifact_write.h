#ifndef TITAN_HOLOLIFT_ARTIFACT_WRITE_H
#define TITAN_HOLOLIFT_ARTIFACT_WRITE_H

#include <cstdint>
#include <expected>
#include <string>

namespace titan_hololift {

enum class HoloLiftArtifactCommitState : std::uint8_t {
  NotCommitted,
  CommittedDurabilityConfirmed,
  CommittedDurabilityUnconfirmed,
};

struct HoloLiftArtifactWriteIssue {
  HoloLiftArtifactCommitState commit_state =
      HoloLiftArtifactCommitState::NotCommitted;
  std::string message;

  [[nodiscard]] bool committed() const noexcept {
    return commit_state != HoloLiftArtifactCommitState::NotCommitted;
  }
};

// A value is CommittedDurabilityConfirmed. Failures before the atomic replace
// carry NotCommitted; failures after it carry CommittedDurabilityUnconfirmed.
using HoloLiftArtifactWriteResult =
    std::expected<HoloLiftArtifactCommitState, HoloLiftArtifactWriteIssue>;

[[nodiscard]] constexpr bool hololift_artifact_was_committed(
    HoloLiftArtifactCommitState state) noexcept {
  return state != HoloLiftArtifactCommitState::NotCommitted;
}

} // namespace titan_hololift

#endif
