#include "titan_path/clearance.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace titan_path {
namespace {

constexpr double kDisplacementTolerance = 1.0e-24;
const double kComponentLimit =
    std::sqrt(std::numeric_limits<double>::max()) / 16.0;

void validate_component(double value, const char *name) {
  if (!std::isfinite(value))
    throw std::invalid_argument(std::string(name) + " must be finite");
  if (std::fabs(value) > kComponentLimit)
    throw std::overflow_error(
        std::string(name) + " exceeds the translation-clearance component limit");
}

// The one-time input bound B=sqrt(DBL_MAX)/16 guarantees relative coordinates
// <=2B, endpoint/closest coordinates <=3B, squared norms <=27B^2, and dot
// products <=6B^2. All are finite with a large rounding margin. Keep the hot
// pair kernel unchecked after that proof has been established at entry.
double squared_norm(const Vec3 &value) {
  return value[0] * value[0] + value[1] * value[1] + value[2] * value[2];
}

void validate_view(const CartesianCoordinatesView &view, const char *name) {
  if (view.x.size() != view.y.size() || view.x.size() != view.z.size())
    throw std::invalid_argument(std::string(name) +
                                " coordinate arrays have different sizes");
  if (view.size() == 0)
    throw std::invalid_argument(std::string(name) +
                                " coordinate view must not be empty");
  for (std::size_t index = 0; index < view.size(); ++index) {
    validate_component(view.x[index], name);
    validate_component(view.y[index], name);
    validate_component(view.z[index], name);
  }
}

PairClosestApproach closest_approach_unchecked(
    const Vec3 &fixed_point_angstrom, const Vec3 &moving_start_angstrom,
    const Vec3 &displacement_angstrom, double displacement_squared) {
  const Vec3 relative{moving_start_angstrom[0] - fixed_point_angstrom[0],
                      moving_start_angstrom[1] - fixed_point_angstrom[1],
                      moving_start_angstrom[2] - fixed_point_angstrom[2]};
  double fraction = 0.0;
  if (displacement_squared > kDisplacementTolerance) {
    const double numerator =
        -(relative[0] * displacement_angstrom[0] +
          relative[1] * displacement_angstrom[1] +
          relative[2] * displacement_angstrom[2]);
    // Clamp before dividing: an interior ratio is in (0,1), so division
    // cannot overflow even when a tiny displacement faces a distant point.
    if (numerator >= displacement_squared)
      fraction = 1.0;
    else if (numerator > 0.0)
      fraction = numerator / displacement_squared;
  }
  const Vec3 closest{relative[0] + fraction * displacement_angstrom[0],
                     relative[1] + fraction * displacement_angstrom[1],
                     relative[2] + fraction * displacement_angstrom[2]};
  return {squared_norm(closest), fraction};
}

} // namespace

double translation_clearance_component_limit_angstrom() noexcept {
  return kComponentLimit;
}

PairClosestApproach closest_approach_during_translation(
    const Vec3 &fixed_point_angstrom, const Vec3 &moving_start_angstrom,
    const Vec3 &displacement_angstrom) {
  for (const double value : {fixed_point_angstrom[0], fixed_point_angstrom[1],
                             fixed_point_angstrom[2],
                             moving_start_angstrom[0],
                             moving_start_angstrom[1],
                             moving_start_angstrom[2],
                             displacement_angstrom[0],
                             displacement_angstrom[1],
                             displacement_angstrom[2]}) {
    validate_component(value, "translation-clearance coordinate/displacement");
  }
  return closest_approach_unchecked(fixed_point_angstrom,
                                    moving_start_angstrom,
                                    displacement_angstrom,
                                    squared_norm(displacement_angstrom));
}

TranslationSweepResult evaluate_translation_sweep(
    const CartesianCoordinatesView &reference,
    const CartesianCoordinatesView &moving_start,
    const Vec3 &displacement_angstrom, double hard_threshold_angstrom) {
  validate_view(reference, "reference");
  validate_view(moving_start, "moving");
  if (!std::isfinite(hard_threshold_angstrom) ||
      hard_threshold_angstrom < 0.0)
    throw std::invalid_argument(
        "translation-clearance threshold must be finite and non-negative");
  validate_component(hard_threshold_angstrom, "translation-clearance threshold");
  for (const double value : displacement_angstrom)
    validate_component(value, "translation-clearance displacement");

  TranslationSweepResult result;
  double minimum_squared = std::numeric_limits<double>::infinity();
  double endpoint_minimum_squared = std::numeric_limits<double>::infinity();
  const double displacement_squared = squared_norm(displacement_angstrom);
  const double threshold_squared =
      hard_threshold_angstrom * hard_threshold_angstrom;
  for (std::size_t ligand_index = 0; ligand_index < moving_start.size();
       ++ligand_index) {
    const Vec3 moving{moving_start.x[ligand_index],
                      moving_start.y[ligand_index],
                      moving_start.z[ligand_index]};
    for (std::size_t receptor_index = 0; receptor_index < reference.size();
         ++receptor_index) {
      const Vec3 fixed{reference.x[receptor_index],
                       reference.y[receptor_index],
                       reference.z[receptor_index]};
      const auto closest = closest_approach_unchecked(
          fixed, moving, displacement_angstrom, displacement_squared);
      const Vec3 start_relative{moving[0] - fixed[0], moving[1] - fixed[1],
                                moving[2] - fixed[2]};
      const Vec3 endpoint_relative{
          moving[0] + displacement_angstrom[0] - fixed[0],
          moving[1] + displacement_angstrom[1] - fixed[1],
          moving[2] + displacement_angstrom[2] - fixed[2]};
      const double endpoint_squared = squared_norm(endpoint_relative);
      ++result.pair_checks;
      endpoint_minimum_squared =
          std::min(endpoint_minimum_squared, endpoint_squared);
      if (closest.distance_squared < minimum_squared) {
        minimum_squared = closest.distance_squared;
        result.witness_fraction = closest.fraction;
        result.receptor_index = receptor_index;
        result.ligand_index = ligand_index;
        result.witness_start_distance_angstrom =
            std::sqrt(squared_norm(start_relative));
        result.witness_endpoint_distance_angstrom =
            std::sqrt(endpoint_squared);
      }
      if (hard_threshold_angstrom > 0.0 &&
          closest.distance_squared < threshold_squared) {
        result.passes_threshold = false;
        result.witness_distance_angstrom = std::sqrt(minimum_squared);
        result.endpoint_distance_angstrom =
            std::sqrt(endpoint_minimum_squared);
        result.threshold_violation_hidden_at_endpoints =
            squared_norm(start_relative) >= threshold_squared &&
            endpoint_squared >= threshold_squared && closest.fraction > 0.0 &&
            closest.fraction < 1.0;
        result.global_minimum_complete = false;
        return result;
      }
    }
  }
  result.witness_distance_angstrom = std::sqrt(minimum_squared);
  result.endpoint_distance_angstrom = std::sqrt(endpoint_minimum_squared);
  result.global_minimum_complete = true;
  return result;
}

} // namespace titan_path
