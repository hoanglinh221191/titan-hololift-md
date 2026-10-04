#ifndef TITAN_SASA_H
#define TITAN_SASA_H

#include <cstddef>
#include <span>
#include <vector>

class TitanExecutor;

namespace TitanSASA {

inline constexpr int kDefaultPointCount = 96;
inline constexpr double kDefaultProbeRadiusA = 1.4;
inline constexpr double kNonpolarGammaKcalPerMolA2 = 0.0072;
inline constexpr double kKcalToKJ = 4.184;
inline constexpr double kNonpolarGammaKJPerMolA2 =
    kNonpolarGammaKcalPerMolA2 * kKcalToKJ;

struct Atom {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double radius{0.0};
};

struct Options {
  double probe_radius{kDefaultProbeRadiusA};
  double min_cell_size{1.0};
  int point_count{kDefaultPointCount};
};

std::vector<double>
compute_per_atom_sasa(std::span<const Atom> atoms, const Options &options,
                      std::span<const std::size_t> targets = {},
                      TitanExecutor *executor = nullptr);

double sum_sasa(std::span<const double> sasa) noexcept;

double nonpolar_energy_from_delta_sasa_kj(double delta_sasa_a2) noexcept;

} // namespace TitanSASA

#endif // TITAN_SASA_H
