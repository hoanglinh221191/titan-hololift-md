#include "titan_mbar_weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace TitanMBAR {
namespace {

struct MetadataFrame {
  double time_ps{0.0};
  std::int64_t global_index{-1};
};

using FramesByReplica = std::unordered_map<int, std::vector<MetadataFrame>>;

bool metadata_sizes_match(const FrameMetadata &metadata) {
  return !metadata.replica_ids.empty() &&
         metadata.replica_ids.size() == metadata.timestamps_ps.size() &&
         metadata.replica_ids.size() == metadata.global_indices.size();
}

FramesByReplica build_frames_by_replica(const FrameMetadata &metadata) {
  FramesByReplica by_replica;
  by_replica.reserve(metadata.replica_ids.size());
  for (std::size_t idx = 0; idx < metadata.replica_ids.size(); ++idx) {
    by_replica[metadata.replica_ids[idx]].push_back(
        {metadata.timestamps_ps[idx], metadata.global_indices[idx]});
  }
  for (auto &entry : by_replica) {
    auto &frames = entry.second;
    std::sort(frames.begin(), frames.end(),
              [](const MetadataFrame &lhs, const MetadataFrame &rhs) {
                return lhs.time_ps < rhs.time_ps;
              });
  }
  return by_replica;
}

std::int64_t max_valid_global_index(const FrameMetadata &metadata) {
  std::int64_t max_index = -1;
  for (std::int64_t value : metadata.global_indices) {
    if (value >= 0)
      max_index = std::max(max_index, value);
  }
  return max_index;
}

const MetadataFrame *find_matching_metadata_frame(
    const std::vector<MetadataFrame> &frames, double time_ps,
    double tolerance_ps) {
  if (frames.empty() || !std::isfinite(time_ps))
    return nullptr;

  const auto lower = std::lower_bound(
      frames.begin(), frames.end(), time_ps,
      [](const MetadataFrame &frame, double target_time) {
        return frame.time_ps < target_time;
      });

  const MetadataFrame *best = nullptr;
  double best_delta = std::numeric_limits<double>::infinity();
  const auto consider = [&](std::vector<MetadataFrame>::const_iterator it) {
    if (it == frames.end())
      return;
    const double delta = std::fabs(it->time_ps - time_ps);
    if (delta <= tolerance_ps && delta < best_delta) {
      best = &(*it);
      best_delta = delta;
    }
  };

  consider(lower);
  if (lower != frames.begin())
    consider(std::prev(lower));
  return best;
}

} // namespace

WeightRemapResult
remap_frame_weights_to_metadata(const std::vector<FrameWeightRow> &frame_weights,
                                const FrameMetadata &metadata,
                                double time_tolerance_ps) {
  WeightRemapResult result;
  if (frame_weights.empty())
    return result;
  if (!metadata_sizes_match(metadata)) {
    result.skipped_missing_metadata = frame_weights.size();
    return result;
  }

  const std::int64_t max_index = max_valid_global_index(metadata);
  if (max_index >= 0) {
    result.global_frame_weights.assign(static_cast<std::size_t>(max_index) + 1,
                                       0.0);
  }

  const FramesByReplica frames_by_replica = build_frames_by_replica(metadata);
  const double tolerance =
      std::isfinite(time_tolerance_ps) && time_tolerance_ps >= 0.0
          ? time_tolerance_ps
          : 1e-3;

  for (const FrameWeightRow &row : frame_weights) {
    if (!std::isfinite(row.weight) || row.weight < 0.0) {
      ++result.skipped_nonfinite_weight;
      continue;
    }

    const auto replica_it = frames_by_replica.find(row.replica_id);
    if (replica_it == frames_by_replica.end()) {
      ++result.skipped_missing_metadata;
      continue;
    }

    const MetadataFrame *match =
        find_matching_metadata_frame(replica_it->second, row.time_ps, tolerance);
    if (!match) {
      ++result.skipped_missing_metadata;
      continue;
    }

    if (match->global_index < 0 ||
        static_cast<std::size_t>(match->global_index) >=
            result.global_frame_weights.size()) {
      ++result.skipped_invalid_global_index;
      continue;
    }

    result.global_frame_weights[static_cast<std::size_t>(match->global_index)] +=
        row.weight;
    ++result.mapped_global_frames;
    result.mapped_weight_sum += row.weight;
  }

  return result;
}

} // namespace TitanMBAR
