#ifndef TITAN_ENERGY_VALIDATION_H
#define TITAN_ENERGY_VALIDATION_H

#include "MOL.h"

#include <cstddef>
#include <optional>
#include <string_view>

class TitanExecutor;

namespace titan_energy_validation {

inline constexpr double kMinimumPairDistanceNm = 1.0e-10;
inline constexpr double kMinimumGbsaPairDistanceNm = 0.05;
inline constexpr std::string_view kOverlappingAtomsError =
    "invalid_overlapping_atoms";
inline constexpr std::string_view kCloseContactError =
    "invalid_close_contact";
inline constexpr std::string_view kIdenticalGbsaSelectionsError =
    "invalid_gbsa_identical_selections";
inline constexpr std::string_view kNonfiniteCoordinateError =
    "invalid_nonfinite_coordinate";

struct InvalidCoordinate {
  int molecule{};
  std::size_t atom{};
  char axis{'x'};
  double value{};
};

struct OverlappingAtomPair {
  std::size_t mol1_atom{};
  std::size_t mol2_atom{};
  double distance_nm{};
};

[[nodiscard]] std::optional<OverlappingAtomPair>
find_overlapping_atoms(const MOL &mol1, const MOL &mol2,
                       bool identical_selections, TitanExecutor &executor,
                       double minimum_pair_distance_nm =
                           kMinimumPairDistanceNm);

[[nodiscard]] std::optional<InvalidCoordinate>
find_nonfinite_coordinate(const MOL &mol1, const MOL &mol2);

} // namespace titan_energy_validation

#endif
