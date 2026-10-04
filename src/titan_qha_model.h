#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

// Mass-weighted Cartesian covariance eigenvalues are in amu * Angstrom^2.
// Frequencies use the classical covariance relation omega^2 = k_B T / lambda.
inline double titan_qha_ts_kj_mol(const std::vector<double> &eigenvalues,
                                double temperature_k) {
  if (!std::isfinite(temperature_k) || temperature_k <= 0.0)
    throw std::invalid_argument("QHA temperature must be finite and positive");
  constexpr double hbar = 1.054571817e-34; // J s
  constexpr double kb = 1.380649e-23;       // J/K, exact
  constexpr double amu_a2 = 1.66053906660e-47; // kg m^2
  constexpr double gas_kj = 8.31446261815324e-3; // kJ/(mol K)
  double entropy_over_kb = 0.0;
  for (double lambda : eigenvalues) {
    if (!std::isfinite(lambda) || lambda <= 0.0)
      throw std::invalid_argument("QHA requires positive finite retained eigenvalues");
    const double x = hbar / std::sqrt(kb * temperature_k * lambda * amu_a2);
    if (x > 50.0)
      entropy_over_kb += (x + 1.0) * std::exp(-x);
    else
      entropy_over_kb += x / std::expm1(x) - std::log(-std::expm1(-x));
  }
  return gas_kj * temperature_k * entropy_over_kb;
}
