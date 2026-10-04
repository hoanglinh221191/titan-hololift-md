#ifndef TITAN_ELECTROSTATICS_H
#define TITAN_ELECTROSTATICS_H

#include <cmath>
#include <limits>

namespace titan_electrostatics {

// Debye-Huckel screening for a 1:1 aqueous electrolyte.  At 298.15 K,
// kappa = 0.3291 * sqrt(I) A^-1 = 3.291 * sqrt(I) nm^-1.
inline constexpr double kReferenceTemperatureK = 298.15;
inline constexpr double kDebyeKappaPerSqrtMolarNmInverse = 3.291;
inline constexpr double kAmberDebyeConversionInverseAngstrom = 50.33355;
inline constexpr double kAmberIonExclusionFactor = 0.73;

[[nodiscard]] inline double
debye_kappa_nm_inverse(double ionic_strength_molar,
                       double temperature_k = kReferenceTemperatureK) noexcept {
  if (!std::isfinite(ionic_strength_molar) || ionic_strength_molar <= 0.0 ||
      !std::isfinite(temperature_k) || temperature_k <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  return kDebyeKappaPerSqrtMolarNmInverse *
         std::sqrt(ionic_strength_molar) *
         std::sqrt(kReferenceTemperatureK / temperature_k);
}

[[nodiscard]] inline double
debye_length_nm(double ionic_strength_molar,
                double temperature_k = kReferenceTemperatureK) noexcept {
  return 1.0 / debye_kappa_nm_inverse(ionic_strength_molar, temperature_k);
}

[[nodiscard]] inline double
debye_screening(double distance_nm, double ionic_strength_molar,
                double temperature_k = kReferenceTemperatureK) noexcept {
  return std::exp(-distance_nm * debye_kappa_nm_inverse(
                                    ionic_strength_molar, temperature_k));
}

// Amber GB salt correction.  Amber first converts ionic strength to the
// physical Debye kappa using solvent dielectric and temperature, then applies
// a 0.73 ion-exclusion factor.  The final factor of 10 converts A^-1 to nm^-1.
[[nodiscard]] inline double amber_gb_kappa_nm_inverse(
    double ionic_strength_molar, double solvent_dielectric,
    double temperature_k = kReferenceTemperatureK) noexcept {
  if (!std::isfinite(ionic_strength_molar) || ionic_strength_molar < 0.0 ||
      !std::isfinite(solvent_dielectric) || solvent_dielectric <= 0.0 ||
      !std::isfinite(temperature_k) || temperature_k <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  if (ionic_strength_molar == 0.0)
    return 0.0;
  return kAmberDebyeConversionInverseAngstrom *
         std::sqrt(ionic_strength_molar /
                   (solvent_dielectric * temperature_k)) *
         kAmberIonExclusionFactor * 10.0;
}

} // namespace titan_electrostatics

#endif
