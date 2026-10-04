#ifndef TITAN_PBCTOPO_VIRTUAL_SEED_H
#define TITAN_PBCTOPO_VIRTUAL_SEED_H

#include "lattice_math.h"

namespace titan_pbctopo {

struct PbctopoVirtualSeedCenter {
  double cx = 1.5;
  double cy = 1.5;
  double cz = 1.5;
};

double pbctopo_normalize_virtual_center(double center);
PbctopoVirtualSeedCenter
pbctopo_center_delta_mic(PbctopoVirtualSeedCenter latest,
                         PbctopoVirtualSeedCenter previous,
                         const PbctopoLatticeMetric &lattice);
PbctopoVirtualSeedCenter
extrapolate_virtual_seed_center(PbctopoVirtualSeedCenter previous,
                                PbctopoVirtualSeedCenter latest,
                                const PbctopoLatticeMetric &lattice);

} // namespace titan_pbctopo

#endif // TITAN_PBCTOPO_VIRTUAL_SEED_H
