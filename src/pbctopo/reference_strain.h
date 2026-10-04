#ifndef TITAN_PBCTOPO_REFERENCE_STRAIN_H
#define TITAN_PBCTOPO_REFERENCE_STRAIN_H

#include <cmath>

namespace titan_pbctopo {

// Squared Cayley strain: symmetric under exchanging current/reference length,
// quadratic near equilibrium, bounded for severe outliers, and sqrt-free.
inline double pbctopo_symmetric_reference_strain(double d2,
                                                  double ref_d2) noexcept {
  if (!std::isfinite(d2) || !std::isfinite(ref_d2) || d2 < 0.0 ||
      ref_d2 <= 0.0)
    return 0.0;
  const double denominator = d2 + ref_d2;
  if (denominator <= 1.0e-12)
    return 0.0;
  const double residual = (d2 - ref_d2) / denominator;
  return residual * residual;
}

} // namespace titan_pbctopo

#endif
