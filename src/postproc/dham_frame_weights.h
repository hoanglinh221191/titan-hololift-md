// DHAM frame weights -> global trajectory frames (KIN-16).
//
// After window matching, each replica's discrete trajectory is the subset of
// its projected frames that has an energy sample. DHAM returns one weight per
// frame of that subset; this maps every weight back to the global frame it
// belongs to through the kept positions, so a frame always carries the weight
// of its own microstate. Frames removed by window matching get weight 0.
#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace titan_dham_weights {

struct MapStats {
  std::size_t mapped_frames{0};
  std::size_t dropped_by_window{0};          // no energy sample: weight 0
  std::size_t dropped_missing_replica{0};    // replica without DHAM weights
  std::size_t dropped_invalid_global_index{0};
  std::size_t duplicate_global_indices{0};
  std::size_t inconsistent_positions{0};     // kept position beyond the weights
};

// per_replica_weights[r][q]   weight of the q-th kept frame of replica r.
// per_replica_kept_positions[r][q]
//                             position of that frame among the projected
//                             frames of replica r, counted in the order of
//                             frame_replica_ids. A replica without an entry
//                             kept all of its frames (identity).
// Returns one weight per global frame index (size max index + 1).
std::vector<double> map_to_global_frames(
    const std::unordered_map<int, std::vector<double>> &per_replica_weights,
    const std::unordered_map<int, std::vector<std::size_t>>
        &per_replica_kept_positions,
    const std::vector<int> &frame_replica_ids,
    const std::vector<int64_t> &frame_global_indices, MapStats &stats);

} // namespace titan_dham_weights
