#ifndef TITAN_PATH_NAVIGATION_H
#define TITAN_PATH_NAVIGATION_H

#include "titan_path/control.h"
#include "titan_path/geometry.h"

#include <cstddef>
#include <array>
#include <optional>
#include <vector>

namespace titan_path {

inline constexpr double kNavigationRecoveryTangentWeight = 0.25;
inline constexpr std::array<double, 3>
    kNavigationOldDirectionBlendFractions{0.25, 0.50, 0.75};

struct NavigationCost {
  double applied_corridor_radius_angstrom{0.0};
  double tube_excess_angstrom{0.0};
  double backtrack_distance_angstrom{0.0};
  double tube_cost{0.0};
  double backtrack_cost{0.0};
  double total_cost{0.0};
};

struct NavigationSeed {
  std::size_t index{0};
  Vec3 direction_unit{};
};

[[nodiscard]] NavigationCost evaluate_navigation_cost(
    double applied_path_distance_angstrom,
    double applied_corridor_radius_angstrom,
    double applied_delta_progress_angstrom, double probe_length_angstrom,
    double tube_weight, double backtrack_weight);

[[nodiscard]] Vec3 build_navigation_guide_direction(
    const Vec3 &current_point_angstrom,
    const PathProjection &current_projection,
    const CorridorSample &current_frame, RouteMode mode,
    double probe_length_angstrom);

[[nodiscard]] std::vector<NavigationSeed> build_navigation_seed_directions(
    const Vec3 &guide_direction,
    const std::optional<Vec3> &old_direction,
    const CorridorSample &current_frame, std::size_t radial_samples,
    double radial_amplitude);

} // namespace titan_path

#endif
