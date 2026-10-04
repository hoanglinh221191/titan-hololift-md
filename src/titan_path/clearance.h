#ifndef TITAN_PATH_CLEARANCE_H
#define TITAN_PATH_CLEARANCE_H

#include "titan_path/route.h"

#include <cstddef>
#include <limits>
#include <span>

namespace titan_path {

// Public inputs (coordinate/displacement components and hard threshold) must
// have magnitude <= sqrt(DBL_MAX)/16. This conservative representability
// bound is unrelated to the physical clearance threshold.
[[nodiscard]] double translation_clearance_component_limit_angstrom() noexcept;

struct CartesianCoordinatesView {
  std::span<const double> x;
  std::span<const double> y;
  std::span<const double> z;

  [[nodiscard]] std::size_t size() const noexcept { return x.size(); }
};

struct PairClosestApproach {
  double distance_squared{std::numeric_limits<double>::infinity()};
  double fraction{0.0};
};

struct TranslationSweepResult {
  bool passes_threshold{true};
  double witness_distance_angstrom{std::numeric_limits<double>::infinity()};
  double witness_fraction{0.0};
  double endpoint_distance_angstrom{std::numeric_limits<double>::infinity()};
  double witness_start_distance_angstrom{
      std::numeric_limits<double>::infinity()};
  double witness_endpoint_distance_angstrom{
      std::numeric_limits<double>::infinity()};
  std::size_t receptor_index{0};
  std::size_t ligand_index{0};
  std::size_t pair_checks{0};
  bool global_minimum_complete{false};
  bool threshold_violation_hidden_at_endpoints{false};
};

[[nodiscard]] PairClosestApproach closest_approach_during_translation(
    const Vec3 &fixed_point_angstrom, const Vec3 &moving_start_angstrom,
    const Vec3 &displacement_angstrom);

[[nodiscard]] TranslationSweepResult evaluate_translation_sweep(
    const CartesianCoordinatesView &reference,
    const CartesianCoordinatesView &moving_start,
    const Vec3 &displacement_angstrom, double hard_threshold_angstrom);

} // namespace titan_path

#endif
