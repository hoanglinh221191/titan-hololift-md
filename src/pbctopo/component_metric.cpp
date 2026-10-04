#include "component_metric.h"

#include "lattice_math.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace titan_pbctopo {

double pbctopo_translation_norm2(const PbctopoShift &shift) noexcept {
  return shift.tx * shift.tx + shift.ty * shift.ty + shift.tz * shift.tz;
}

double pbctopo_translation_norm2(box Box, const Int3 &offset) {
  const PbctopoLatticeMetric lattice(Box);
  const auto translation = lattice.scaled_to_cartesian(
      static_cast<double>(offset.x), static_cast<double>(offset.y),
      static_cast<double>(offset.z));
  return translation[0] * translation[0] + translation[1] * translation[1] +
         translation[2] * translation[2];
}

Int3 pbctopo_component_offset_gauge(box Box,
                                    const std::vector<Int3> &offsets) {
  if (offsets.empty())
    return {};

  double mean_x = 0.0;
  double mean_y = 0.0;
  double mean_z = 0.0;
  for (const Int3 &offset : offsets) {
    mean_x += static_cast<double>(offset.x);
    mean_y += static_cast<double>(offset.y);
    mean_z += static_cast<double>(offset.z);
  }
  const double inv_count = 1.0 / static_cast<double>(offsets.size());
  mean_x *= inv_count;
  mean_y *= inv_count;
  mean_z *= inv_count;

  const PbctopoNearestImage nearest =
      PbctopoLatticeMetric(Box).nearest_image_scaled(mean_x, mean_y, mean_z);
  if (!nearest.valid)
    return {};
  Int3 gauge{};
  if (!checked_int3_negate(
          {nearest.image_x, nearest.image_y, nearest.image_z}, gauge)) {
    return {};
  }
  return gauge;
}

bool rebase_component_offsets(std::vector<Int3> &offsets, box Box) {
  const Int3 gauge = pbctopo_component_offset_gauge(Box, offsets);
  std::vector<Int3> rebased;
  rebased.reserve(offsets.size());
  for (const Int3 &offset : offsets) {
    Int3 value{};
    if (!checked_int3_subtract(offset, gauge, value))
      return false;
    rebased.push_back(value);
  }
  offsets = std::move(rebased);
  return true;
}

double pbctopo_component_shift_metric(box Box,
                                      const std::vector<Int3> &offsets) {
  if (offsets.empty())
    return 0.0;
  const Int3 gauge = pbctopo_component_offset_gauge(Box, offsets);
  const PbctopoLatticeMetric lattice(Box);
  double sum_d2 = 0.0;
  for (const Int3 &raw_offset : offsets) {
    Int3 offset{};
    if (!checked_int3_subtract(raw_offset, gauge, offset))
      return std::numeric_limits<double>::infinity();
    const auto translation = lattice.scaled_to_cartesian(
        static_cast<double>(offset.x), static_cast<double>(offset.y),
        static_cast<double>(offset.z));
    sum_d2 += translation[0] * translation[0] +
              translation[1] * translation[1] +
              translation[2] * translation[2];
  }
  return sum_d2 / static_cast<double>(offsets.size());
}

} // namespace titan_pbctopo
