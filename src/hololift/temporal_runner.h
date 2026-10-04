#ifndef TITAN_HOLOLIFT_TEMPORAL_RUNNER_H
#define TITAN_HOLOLIFT_TEMPORAL_RUNNER_H

// HoloLift temporal selection on a VIBE package and the trajectory it was made from: import, Phase 1 optimum,
// all-atom audit, the per-frame and per-segment records, and, on request, the selected path as coordinates.
// Used by TITAN's -pbctopo-temporal=hololift and by tools/hololift_temporal_value_runner.cpp; both produce the
// same records for the same settings.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace titan_hololift {

enum class HoloLiftCoordinateOutput : std::uint8_t {
  None,     // records only
  Lattice,  // every component on its selected image plus the cumulative lattice gauge (lattice-preserving)
  Toroidal, // the same relative images; the anchor component follows displacement-preserving (toroidal) unwrapping
};

enum class HoloLiftPdbOutput : std::uint8_t { None, First, All };

struct HoloLiftRunOptions {
  std::string system_label = "hololift";
  std::filesystem::path package_directory;
  std::filesystem::path trajectory_path;
  std::filesystem::path output_directory;
  // HoloLift's temporal transport band and its synthetic controls (H2-H4); 0 uses the package bands.
  std::size_t transport_radius = 0;
  bool synthetic_rank_inversion_v1 = false;
  bool synthetic_domain_miss_v1 = false;
  bool synthetic_domain_miss_v2 = false;
  // "strict" (default), "carry" or "restore"; "hard" is the former name of strict.
  std::string evidence_absent = "strict";
  // Unset values keep the Phase 1 defaults, so a run without overrides is bit-identical to one made before the
  // options existed.
  std::optional<double> rank_weight;
  std::optional<double> transition_weight;
  std::string rank_cost = "linear"; // "linear" or "saturated"
  std::optional<double> unique_evidence_weight;
  std::optional<double> supported_evidence_weight;
  std::optional<std::size_t> dp_workers;
  // The trajectory is streamed in blocks of trajectory_block_frames frames (0: TITAN's XDR chunk budget, 64 MB and
  // at most 64 frames, but at least one frame per worker); its frames are bound, prepared, audited and written on
  // trajectory_workers threads.  Neither number changes a result.
  std::size_t trajectory_workers = 1;
  std::size_t trajectory_block_frames = 0;
  // Coordinates of the selected path, reconstructed by the all-atom audit.
  HoloLiftCoordinateOutput coordinates = HoloLiftCoordinateOutput::None;
  HoloLiftPdbOutput pdb = HoloLiftPdbOutput::None;
  // Atom records for the PDB (names, residues, elements), indexed by source atom id; empty writes generic names.
  std::filesystem::path reference_pdb;
  // Wall time of each stage on stderr, as TIMING lines.
  bool report_stage_times = false;
};

// Returns 0 on success and the runner's error codes otherwise (2 import or configuration, 3 trajectory binding,
// 4-5 preparation, 6 solve, 7-9 audit, 10 output, 12 transport band, 13 evidence policy, 14 coordinates); errors
// are reported on stderr and a one-line PASS summary on stdout.
int run_hololift_temporal_value(const HoloLiftRunOptions &options);

[[nodiscard]] std::optional<HoloLiftCoordinateOutput> parse_hololift_coordinate_output(const std::string &name);
[[nodiscard]] std::optional<HoloLiftPdbOutput> parse_hololift_pdb_output(const std::string &name);

} // namespace titan_hololift

#endif
