#include "virtual_seed.h"

#include <algorithm>
#include <cmath>

namespace titan_pbctopo {

double pbctopo_normalize_virtual_center(double center) {
  if (!std::isfinite(center))
    return 1.5;
  while (center < 0.5)
    center += 1.0;
  while (center > 2.5)
    center -= 1.0;
  return std::clamp(center, 0.5, 2.5);
}

PbctopoVirtualSeedCenter
pbctopo_center_delta_mic(PbctopoVirtualSeedCenter latest,
                         PbctopoVirtualSeedCenter previous,
                         const PbctopoLatticeMetric &lattice) {
  PbctopoVirtualSeedCenter delta{latest.cx - previous.cx,
                                 latest.cy - previous.cy,
                                 latest.cz - previous.cz};
  const PbctopoNearestImage mic =
      lattice.nearest_image_scaled(delta.cx, delta.cy, delta.cz);
  return {mic.sx, mic.sy, mic.sz};
}

PbctopoVirtualSeedCenter
extrapolate_virtual_seed_center(PbctopoVirtualSeedCenter previous,
                                PbctopoVirtualSeedCenter latest,
                                const PbctopoLatticeMetric &lattice) {
  const PbctopoVirtualSeedCenter delta =
      pbctopo_center_delta_mic(latest, previous, lattice);
  return {pbctopo_normalize_virtual_center(latest.cx + delta.cx),
          pbctopo_normalize_virtual_center(latest.cy + delta.cy),
          pbctopo_normalize_virtual_center(latest.cz + delta.cz)};
}

} // namespace titan_pbctopo
