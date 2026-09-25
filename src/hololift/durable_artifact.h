#ifndef TITAN_HOLOLIFT_DURABLE_ARTIFACT_H
#define TITAN_HOLOLIFT_DURABLE_ARTIFACT_H

#include "artifact_write.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace titan_hololift {

namespace detail {

inline HoloLiftArtifactWriteResult artifact_write_failure(
    HoloLiftArtifactCommitState commit_state, std::string message) {
  return std::unexpected(
      HoloLiftArtifactWriteIssue{commit_state, std::move(message)});
}

#ifdef TITAN_HOLOLIFT_DURABLE_ARTIFACT_TEST_HOOKS
enum class HoloLiftDurableArtifactTestFault : std::uint8_t {
  None,
  BeforeCommit,
  AfterCommitBeforeDurabilityConfirmation,
};

inline HoloLiftDurableArtifactTestFault &durable_artifact_test_fault() {
  static thread_local HoloLiftDurableArtifactTestFault fault =
      HoloLiftDurableArtifactTestFault::None;
  return fault;
}

inline void set_durable_artifact_test_fault(
    HoloLiftDurableArtifactTestFault fault) noexcept {
  durable_artifact_test_fault() = fault;
}
#endif

inline HoloLiftArtifactWriteResult durably_replace_file(
    const std::filesystem::path &temporary,
    const std::filesystem::path &path,
    std::string_view artifact_name) {
#ifdef _WIN32
#ifdef TITAN_HOLOLIFT_DURABLE_ARTIFACT_TEST_HOOKS
  if (durable_artifact_test_fault() ==
      HoloLiftDurableArtifactTestFault::BeforeCommit) {
    return artifact_write_failure(
        HoloLiftArtifactCommitState::NotCommitted,
        "injected failure before committing " + std::string(artifact_name));
  }
#endif
  if (!MoveFileExW(temporary.c_str(), path.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return artifact_write_failure(
        HoloLiftArtifactCommitState::NotCommitted,
        "failed to atomically replace " + std::string(artifact_name) +
        ": " +
        std::to_string(static_cast<unsigned long>(GetLastError())));
  }
#ifdef TITAN_HOLOLIFT_DURABLE_ARTIFACT_TEST_HOOKS
  if (durable_artifact_test_fault() ==
      HoloLiftDurableArtifactTestFault::
          AfterCommitBeforeDurabilityConfirmation) {
    return artifact_write_failure(
        HoloLiftArtifactCommitState::CommittedDurabilityUnconfirmed,
        "injected durability-confirmation failure after committing " +
            std::string(artifact_name));
  }
#endif
#else
  const int file_descriptor = ::open(temporary.c_str(), O_RDONLY);
  if (file_descriptor < 0) {
    const int open_error = errno;
    return artifact_write_failure(
        HoloLiftArtifactCommitState::NotCommitted,
        "failed to open temporary " + std::string(artifact_name) +
            " for fsync: " + std::string(std::strerror(open_error)));
  }
  const int file_sync_status = ::fsync(file_descriptor);
  const int file_sync_error = errno;
  const int file_close_status = ::close(file_descriptor);
  const int file_close_error = errno;
  if (file_sync_status != 0) {
    return artifact_write_failure(
        HoloLiftArtifactCommitState::NotCommitted,
        "failed to fsync temporary " + std::string(artifact_name) + ": " +
            std::string(std::strerror(file_sync_error)));
  }
  if (file_close_status != 0) {
    return artifact_write_failure(
        HoloLiftArtifactCommitState::NotCommitted,
        "failed to close temporary " + std::string(artifact_name) +
            " after fsync: " +
            std::string(std::strerror(file_close_error)));
  }

#ifdef TITAN_HOLOLIFT_DURABLE_ARTIFACT_TEST_HOOKS
  if (durable_artifact_test_fault() ==
      HoloLiftDurableArtifactTestFault::BeforeCommit) {
    return artifact_write_failure(
        HoloLiftArtifactCommitState::NotCommitted,
        "injected failure before committing " + std::string(artifact_name));
  }
#endif
  std::error_code rename_error;
  std::filesystem::rename(temporary, path, rename_error);
  if (rename_error) {
    return artifact_write_failure(
        HoloLiftArtifactCommitState::NotCommitted,
        "failed to atomically replace " + std::string(artifact_name) +
            ": " + rename_error.message());
  }

#ifdef TITAN_HOLOLIFT_DURABLE_ARTIFACT_TEST_HOOKS
  if (durable_artifact_test_fault() ==
      HoloLiftDurableArtifactTestFault::
          AfterCommitBeforeDurabilityConfirmation) {
    return artifact_write_failure(
        HoloLiftArtifactCommitState::CommittedDurabilityUnconfirmed,
        "injected durability-confirmation failure after committing " +
            std::string(artifact_name));
  }
#endif
  const auto parent = path.has_parent_path() ? path.parent_path()
                                             : std::filesystem::path{"."};
  const int directory_descriptor =
      ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
  if (directory_descriptor < 0) {
    const int open_error = errno;
    return artifact_write_failure(
        HoloLiftArtifactCommitState::CommittedDurabilityUnconfirmed,
        "failed to open " + std::string(artifact_name) +
            " directory for fsync after commit: " +
            std::string(std::strerror(open_error)));
  }
  const int directory_sync_status = ::fsync(directory_descriptor);
  const int directory_sync_error = errno;
  const int directory_close_status = ::close(directory_descriptor);
  const int directory_close_error = errno;
  if (directory_sync_status != 0) {
    return artifact_write_failure(
        HoloLiftArtifactCommitState::CommittedDurabilityUnconfirmed,
        "failed to fsync " + std::string(artifact_name) +
            " directory after commit: " +
            std::string(std::strerror(directory_sync_error)));
  }
  // A close error can report delayed directory I/O failure even after fsync;
  // conservatively retain the committed file but do not confirm durability.
  if (directory_close_status != 0) {
    return artifact_write_failure(
        HoloLiftArtifactCommitState::CommittedDurabilityUnconfirmed,
        "failed to close " + std::string(artifact_name) +
            " directory after commit/fsync: " +
            std::string(std::strerror(directory_close_error)));
  }
#endif
  return HoloLiftArtifactCommitState::CommittedDurabilityConfirmed;
}

} // namespace detail
} // namespace titan_hololift

#endif
