#ifndef TITAN_PBCTOPO_CANDIDATE_ORDER_H
#define TITAN_PBCTOPO_CANDIDATE_ORDER_H

#include <cstdint>

namespace titan_pbctopo {

struct PbctopoMetricResolution {
  double absolute = 1.0e-6;
  double relative = 0.0;
};

std::int64_t pbctopo_metric_bucket(double value,
                                   PbctopoMetricResolution resolution);
int compare_pbctopo_metric(double lhs, double rhs,
                            PbctopoMetricResolution resolution = {});

} // namespace titan_pbctopo

#endif
