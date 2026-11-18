#include "postproc/dham_frame_weights.h"

#include <algorithm>

namespace titan_dham_weights {

std::vector<double> map_to_global_frames(
    const std::unordered_map<int, std::vector<double>> &per_replica_weights,
    const std::unordered_map<int, std::vector<std::size_t>>
        &per_replica_kept_positions,
    const std::vector<int> &frame_replica_ids,
    const std::vector<int64_t> &frame_global_indices, MapStats &stats) {
  stats = {};
  std::vector<double> out;
  int64_t max_global_index = -1;
  for (int64_t global_idx : frame_global_indices)
    max_global_index = std::max(max_global_index, global_idx);
  if (max_global_index < 0)
    return out;
  out.assign(static_cast<std::size_t>(max_global_index) + 1, 0.0);
  std::vector<unsigned char> assigned(out.size(), 0);

  // position among the replica's projected frames -> position in its kept
  // subset (-1: removed by window matching)
  std::unordered_map<int, std::vector<int64_t>> kept_slot;
  kept_slot.reserve(per_replica_kept_positions.size());
  for (const auto &[replica_id, positions] : per_replica_kept_positions) {
    std::size_t size = 0;
    for (std::size_t p : positions)
      size = std::max(size, p + 1);
    auto &slots = kept_slot[replica_id];
    slots.assign(size, -1);
    for (std::size_t q = 0; q < positions.size(); ++q)
      slots[positions[q]] = static_cast<int64_t>(q);
  }

  std::unordered_map<int, std::size_t> next_position;
  const std::size_t n_frames =
      std::min(frame_replica_ids.size(), frame_global_indices.size());
  for (std::size_t frame = 0; frame < n_frames; ++frame) {
    const int replica_id = frame_replica_ids[frame];
    const std::size_t position = next_position[replica_id]++;
    const auto it_weights = per_replica_weights.find(replica_id);
    if (it_weights == per_replica_weights.end()) {
      ++stats.dropped_missing_replica;
      continue;
    }
    std::size_t slot = position;
    const auto it_slots = kept_slot.find(replica_id);
    if (it_slots != kept_slot.end()) {
      const auto &slots = it_slots->second;
      if (position >= slots.size() || slots[position] < 0) {
        ++stats.dropped_by_window;
        continue;
      }
      slot = static_cast<std::size_t>(slots[position]);
    }
    if (slot >= it_weights->second.size()) {
      ++stats.inconsistent_positions;
      continue;
    }
    const int64_t global_idx = frame_global_indices[frame];
    if (global_idx < 0 || static_cast<std::size_t>(global_idx) >= out.size()) {
      ++stats.dropped_invalid_global_index;
      continue;
    }
    const std::size_t global_slot = static_cast<std::size_t>(global_idx);
    if (assigned[global_slot])
      ++stats.duplicate_global_indices;
    out[global_slot] = it_weights->second[slot];
    assigned[global_slot] = 1;
    ++stats.mapped_frames;
  }
  return out;
}

} // namespace titan_dham_weights
