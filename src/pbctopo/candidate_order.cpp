#include "candidate_order.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace titan_pbctopo {

std::int64_t pbctopo_metric_bucket(double value,
                                   PbctopoMetricResolution resolution) {
  const long double absolute =
      std::max<long double>(std::fabs(resolution.absolute), 1.0e-15L);
  const long double relative =
      std::max<long double>(std::fabs(resolution.relative), 0.0L);
  const long double scaled =
      relative > 0.0L
          ? std::asinh(static_cast<long double>(value) * relative / absolute) /
                std::asinh(relative)
          : static_cast<long double>(value) / absolute;
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min() + 1;
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max() - 1;
  if (scaled <= static_cast<long double>(kMin))
    return kMin;
  if (scaled >= static_cast<long double>(kMax))
    return kMax;
  return static_cast<std::int64_t>(std::llround(scaled));
}

int compare_pbctopo_metric(double lhs, double rhs,
                            PbctopoMetricResolution resolution) {
  const bool lhs_nan = std::isnan(lhs);
  const bool rhs_nan = std::isnan(rhs);
  if (lhs_nan != rhs_nan)
    return lhs_nan ? 1 : -1;
  if (lhs_nan)
    return 0;
  if (lhs == rhs)
    return 0;
  if (std::isinf(lhs) || std::isinf(rhs))
    return lhs < rhs ? -1 : 1;
  const std::int64_t lhs_bucket = pbctopo_metric_bucket(lhs, resolution);
  const std::int64_t rhs_bucket = pbctopo_metric_bucket(rhs, resolution);
  if (lhs_bucket == rhs_bucket)
    return 0;
  return lhs_bucket < rhs_bucket ? -1 : 1;
}

} // namespace titan_pbctopo
