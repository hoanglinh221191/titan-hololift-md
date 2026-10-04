#ifndef TITAN_MBAR_WEIGHTS_H
#define TITAN_MBAR_WEIGHTS_H

#include "titan_mbar.h"

#include <cstddef>
#include <vector>

namespace TitanMBAR {

struct WeightRemapResult {
  std::vector<double> global_frame_weights{};
  std::size_t mapped_global_frames{0};
  std::size_t skipped_missing_metadata{0};
  std::size_t skipped_invalid_global_index{0};
  std::size_t skipped_nonfinite_weight{0};
  double mapped_weight_sum{0.0};
};

WeightRemapResult
remap_frame_weights_to_metadata(const std::vector<FrameWeightRow> &frame_weights,
                                const FrameMetadata &metadata,
                                double time_tolerance_ps = 1e-3);

} // namespace TitanMBAR

#endif // TITAN_MBAR_WEIGHTS_H
