#ifndef TITAN_HOLOLIFT_VIBE_IMPORTER_H
#define TITAN_HOLOLIFT_VIBE_IMPORTER_H

#include "types.h"

#include <cstddef>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>

namespace titan_hololift {

inline constexpr std::uint32_t HOLOLIFT_VIBE_IMPORTER_API_VERSION = 6;

enum class HoloLiftVibeImportCode {
  InvalidOptions,
  MissingArtifact,
  IoError,
  CsvSyntax,
  MissingColumn,
  DuplicateColumn,
  InvalidValue,
  ContractMismatch,
  ReferentialIntegrity,
  UnsupportedObservation,
  StoreValidationFailed,
};

struct HoloLiftVibeImportError {
  HoloLiftVibeImportCode code = HoloLiftVibeImportCode::InvalidValue;
  std::filesystem::path artifact;
  std::size_t line = 0;
  std::string message;
};

std::string_view
hololift_vibe_import_code_name(HoloLiftVibeImportCode code) noexcept;

struct HoloLiftVibeImportOptions {
  std::filesystem::path package_directory;
  std::string package_id;
  std::string trajectory_id;
  bool accept_experimental_temporal_diagnostics = true;
};

struct HoloLiftVibeImportStats {
  std::size_t topology_epochs = 0;
  std::size_t source_frame_rows = 0;
  std::size_t imported_frames = 0;
  std::size_t imported_weak_observation_frames = 0;
  std::size_t skipped_local_unwrap_frames = 0;
  // Retained for source compatibility; current-schema weak bands are imported.
  std::size_t skipped_weak_observation_frames = 0;
  std::size_t skipped_fallback_frames = 0;
  std::size_t candidate_rows = 0;
  std::size_t experimental_temporal_frames = 0;
  std::size_t observation_segments = 0;
};

struct HoloLiftVibeImportResult {
  HoloLiftObservationStore store;
  HoloLiftVibeImportStats stats;
};

// Imports the current stable VIBE observation package without retaining
// duplicate CSV rows in memory. Weak observations retain their hard-certified
// candidate bands but are never promoted to temporal anchors. Frames without a
// hard-certified band are omitted as observation gaps, while their counts are
// retained in the immutable store's source-coverage manifest.
std::expected<HoloLiftVibeImportResult, HoloLiftVibeImportError>
import_hololift_vibe_package(const HoloLiftVibeImportOptions &options);

inline std::expected<HoloLiftVibeImportResult, HoloLiftVibeImportError>
import_hololift_vibe_v1_package(const HoloLiftVibeImportOptions &options) {
  return import_hololift_vibe_package(options);
}

} // namespace titan_hololift

#endif
