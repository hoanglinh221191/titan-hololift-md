#include "titan_path/navigation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace titan_path {
namespace {

constexpr double kVectorTolerance = 1.0e-12;
constexpr double kPi = 3.14159265358979323846;

Vec3 add(const Vec3 &lhs, const Vec3 &rhs) {
  return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

Vec3 subtract(const Vec3 &lhs, const Vec3 &rhs) {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

Vec3 scale(const Vec3 &value, double factor) {
  return {factor * value[0], factor * value[1], factor * value[2]};
}

Vec3 cross(const Vec3 &lhs, const Vec3 &rhs) {
  return {lhs[1] * rhs[2] - lhs[2] * rhs[1],
          lhs[2] * rhs[0] - lhs[0] * rhs[2],
          lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}

std::optional<Vec3> normalized(const Vec3 &value) {
  const double magnitude = norm(value);
  if (!std::isfinite(magnitude) || magnitude <= kVectorTolerance)
    return std::nullopt;
  return scale(value, 1.0 / magnitude);
}

bool same_direction(const Vec3 &lhs, const Vec3 &rhs) {
  return dot(lhs, rhs) >= 1.0 - 1.0e-12;
}

void append_unique(std::vector<NavigationSeed> &seeds, const Vec3 &direction) {
  const auto unit = normalized(direction);
  if (!unit.has_value())
    return;
  for (const auto &seed : seeds)
    if (same_direction(seed.direction_unit, *unit))
      return;
  seeds.push_back({seeds.size(), *unit});
}

} // namespace

NavigationCost evaluate_navigation_cost(
    double applied_path_distance_angstrom,
    double applied_corridor_radius_angstrom,
    double applied_delta_progress_angstrom, double probe_length_angstrom,
    double tube_weight, double backtrack_weight) {
  for (const double value : {applied_path_distance_angstrom,
                             applied_corridor_radius_angstrom,
                             applied_delta_progress_angstrom,
                             probe_length_angstrom, tube_weight,
                             backtrack_weight}) {
    if (!std::isfinite(value))
      throw std::invalid_argument("navigation inputs must be finite");
  }
  if (applied_path_distance_angstrom < 0.0 ||
      applied_corridor_radius_angstrom < 0.0 ||
      probe_length_angstrom <= 0.0 || tube_weight < 0.0 ||
      backtrack_weight < 0.0) {
    throw std::invalid_argument(
        "navigation distances and weights are outside their valid range");
  }

  NavigationCost result;
  result.applied_corridor_radius_angstrom =
      applied_corridor_radius_angstrom;
  result.tube_excess_angstrom =
      std::max(0.0, applied_path_distance_angstrom -
                        applied_corridor_radius_angstrom);
  result.backtrack_distance_angstrom =
      std::max(0.0, -applied_delta_progress_angstrom);
  const double normalized_tube =
      result.tube_excess_angstrom / probe_length_angstrom;
  const double normalized_backtrack =
      result.backtrack_distance_angstrom / probe_length_angstrom;
  result.tube_cost =
      0.5 * tube_weight * normalized_tube * normalized_tube;
  result.backtrack_cost = 0.5 * backtrack_weight * normalized_backtrack *
                          normalized_backtrack;
  result.total_cost = result.tube_cost + result.backtrack_cost;
  return result;
}

Vec3 build_navigation_guide_direction(
    const Vec3 &current_point_angstrom,
    const PathProjection &current_projection,
    const CorridorSample &current_frame, RouteMode mode,
    double probe_length_angstrom) {
  if (!current_projection.valid || !std::isfinite(probe_length_angstrom) ||
      probe_length_angstrom <= 0.0 ||
      !std::isfinite(current_frame.radius_angstrom) ||
      current_frame.radius_angstrom < 0.0) {
    throw std::invalid_argument("invalid navigation guide inputs");
  }
  const auto tangent = normalized(current_frame.tangent);
  if (!tangent.has_value())
    throw std::invalid_argument("navigation frame tangent is singular");

  const Vec3 offset =
      subtract(current_point_angstrom, current_projection.point_angstrom);
  const double distance = norm(offset);
  const auto return_direction = normalized(scale(offset, -1.0));
  if (mode == RouteMode::Recovery) {
    if (!return_direction.has_value())
      return *tangent;
    const auto recovery = normalized(
        add(*return_direction,
            scale(*tangent, kNavigationRecoveryTangentWeight)));
    if (!recovery.has_value())
      throw std::runtime_error("Recovery navigation guide is singular");
    return *recovery;
  }

  if (!return_direction.has_value() ||
      distance <= current_frame.radius_angstrom)
    return *tangent;
  const double excess = distance - current_frame.radius_angstrom;
  const double return_strength =
      std::clamp(excess / probe_length_angstrom, 0.0, 1.0);
  const auto advance = normalized(
      add(*tangent, scale(*return_direction, return_strength)));
  if (!advance.has_value())
    throw std::runtime_error("Advance navigation guide is singular");
  return *advance;
}

std::vector<NavigationSeed> build_navigation_seed_directions(
    const Vec3 &guide_direction,
    const std::optional<Vec3> &old_direction,
    const CorridorSample &current_frame, std::size_t radial_samples,
    double radial_amplitude) {
  if (!std::isfinite(radial_amplitude) || radial_amplitude < 0.0)
    throw std::invalid_argument(
        "navigation seed amplitude must be finite and non-negative");
  const auto guide = normalized(guide_direction);
  if (!guide.has_value())
    throw std::invalid_argument("navigation guide direction is singular");
  auto basis_1 = normalized(subtract(
      current_frame.normal_1,
      scale(*guide, dot(current_frame.normal_1, *guide))));
  if (!basis_1.has_value()) {
    basis_1 = normalized(subtract(
        current_frame.normal_2,
        scale(*guide, dot(current_frame.normal_2, *guide))));
  }
  if (!basis_1.has_value())
    throw std::invalid_argument("navigation normal frame is singular");
  const auto basis_2 = normalized(cross(*guide, *basis_1));
  if (!basis_2.has_value())
    throw std::invalid_argument("navigation guide ring basis is singular");

  std::vector<NavigationSeed> seeds;
  seeds.reserve(1 + kNavigationOldDirectionBlendFractions.size() +
                radial_samples);
  append_unique(seeds, *guide);
  if (old_direction.has_value()) {
    const auto old = normalized(*old_direction);
    if (old.has_value()) {
      for (const double alpha : kNavigationOldDirectionBlendFractions) {
        append_unique(seeds,
                      add(scale(*old, 1.0 - alpha),
                          scale(*guide, alpha)));
      }
    }
  }
  for (std::size_t radial = 0; radial < radial_samples; ++radial) {
    const double angle = radial_samples == 0
                             ? 0.0
                             : 2.0 * kPi * static_cast<double>(radial) /
                                   static_cast<double>(radial_samples);
    const Vec3 perturbation =
        add(scale(*basis_1, radial_amplitude * std::cos(angle)),
            scale(*basis_2, radial_amplitude * std::sin(angle)));
    append_unique(seeds, add(*guide, perturbation));
  }
  for (std::size_t index = 0; index < seeds.size(); ++index)
    seeds[index].index = index;
  return seeds;
}

} // namespace titan_path
