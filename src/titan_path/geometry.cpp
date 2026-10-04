#include "titan_path/geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace titan_path {
namespace {

constexpr double kFrameTolerance = 1.0e-12;

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

Vec3 normalize(const Vec3 &value) {
  const double magnitude = norm(value);
  if (!std::isfinite(magnitude) || magnitude <= kFrameTolerance)
    throw std::runtime_error("cannot normalize a singular corridor vector");
  return scale(value, 1.0 / magnitude);
}

Vec3 project_normal(const Vec3 &normal, const Vec3 &tangent) {
  return normalize(subtract(normal, scale(tangent, dot(normal, tangent))));
}

Vec3 rodrigues(const Vec3 &value, const Vec3 &axis_unit, double angle) {
  return add(add(scale(value, std::cos(angle)),
                 scale(cross(axis_unit, value), std::sin(angle))),
             scale(axis_unit,
                   dot(axis_unit, value) * (1.0 - std::cos(angle))));
}

Vec3 initial_normal(const Vec3 &tangent) {
  const Vec3 axes[3]{{1.0, 0.0, 0.0},
                     {0.0, 1.0, 0.0},
                     {0.0, 0.0, 1.0}};
  std::size_t best = 0;
  for (std::size_t i = 1; i < 3; ++i)
    if (std::fabs(dot(tangent, axes[i])) <
        std::fabs(dot(tangent, axes[best])))
      best = i;
  return normalize(cross(tangent, axes[best]));
}

struct CenterSample {
  double progress{0.0};
  Vec3 center{};
  double radius{0.0};
};

std::vector<CenterSample> resample(const RouteDefinition &route,
                                   double spacing_angstrom) {
  std::vector<double> waypoint_progress(route.waypoints.size(), 0.0);
  for (std::size_t i = 0; i < route.waypoints.size(); ++i) {
    const auto &waypoint = route.waypoints[i];
    for (const double coordinate : waypoint.position_angstrom)
      if (!std::isfinite(coordinate))
        throw std::runtime_error("corridor waypoint coordinates must be finite");
    if (!std::isfinite(waypoint.radius_angstrom) ||
        waypoint.radius_angstrom < 0.0)
      throw std::runtime_error("corridor waypoint radius must be finite and non-negative");
    if (i == 0)
      continue;
    waypoint_progress[i] =
        waypoint_progress[i - 1] +
        norm(subtract(route.waypoints[i].position_angstrom,
                       route.waypoints[i - 1].position_angstrom));
    if (!std::isfinite(waypoint_progress[i]))
      throw std::runtime_error("corridor arc length is outside the finite range");
  }

  const double total = waypoint_progress.back();
  if (total <= kFrameTolerance)
    throw std::runtime_error("corridor arc length must be positive");
  const double spacing_steps = std::floor(total / spacing_angstrom);
  // Both endpoints are reserved in addition to interior spacing samples.
  // Checking in floating point first prevents overflow/undefined conversion
  // and a tiny accepted spacing from requesting an unbounded allocation.
  if (!std::isfinite(spacing_steps) ||
      spacing_steps > static_cast<double>(kMaximumCorridorSamples - 2))
    throw std::runtime_error("corridor resampling exceeds the sample limit");
  const std::size_t full_spacing_steps =
      static_cast<std::size_t>(spacing_steps);
  std::vector<double> sample_progress;
  sample_progress.reserve(full_spacing_steps + 2);
  sample_progress.push_back(0.0);
  for (std::size_t i = 1; i <= full_spacing_steps; ++i) {
    const double progress = static_cast<double>(i) * spacing_angstrom;
    if (progress >= total - kFrameTolerance)
      break;
    sample_progress.push_back(progress);
  }
  sample_progress.push_back(total);
  std::vector<CenterSample> output;
  output.reserve(sample_progress.size());
  std::size_t segment = 0;
  for (const double progress : sample_progress) {
    while (segment + 1 < waypoint_progress.size() - 1 &&
           progress > waypoint_progress[segment + 1])
      ++segment;
    const double begin = waypoint_progress[segment];
    const double end = waypoint_progress[segment + 1];
    const double fraction =
        end <= begin ? 0.0 : std::clamp((progress - begin) / (end - begin),
                                       0.0, 1.0);
    const auto &left = route.waypoints[segment];
    const auto &right = route.waypoints[segment + 1];
    output.push_back(
        {progress,
         add(left.position_angstrom,
             scale(subtract(right.position_angstrom, left.position_angstrom),
                   fraction)),
         left.radius_angstrom +
             fraction * (right.radius_angstrom - left.radius_angstrom)});
  }
  return output;
}

} // namespace

double dot(const Vec3 &lhs, const Vec3 &rhs) {
  return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

double norm(const Vec3 &value) {
  return std::hypot(value[0], value[1], value[2]);
}

CorridorBuildResult
build_parallel_transport_corridor(const RouteDefinition &route,
                                  double spacing_angstrom) {
  CorridorBuildResult result;
  try {
    if (!std::isfinite(spacing_angstrom) || spacing_angstrom <= 0.0)
      throw std::runtime_error("corridor spacing must be finite and positive");
    if (route.waypoints.size() < 2)
      throw std::runtime_error("corridor construction requires two waypoints");
    const auto centers = resample(route, spacing_angstrom);
    if (centers.size() < 2)
      throw std::runtime_error("corridor resampling produced fewer than two samples");

    result.samples.resize(centers.size());
    for (std::size_t i = 0; i < centers.size(); ++i) {
      Vec3 difference{};
      if (i == 0)
        difference = subtract(centers[1].center, centers[0].center);
      else if (i + 1 == centers.size())
        difference = subtract(centers[i].center, centers[i - 1].center);
      else
        difference = subtract(centers[i + 1].center, centers[i - 1].center);
      result.samples[i].progress_angstrom = centers[i].progress;
      result.samples[i].center_angstrom = centers[i].center;
      result.samples[i].radius_angstrom = centers[i].radius;
      result.samples[i].tangent = normalize(difference);
    }

    result.samples[0].normal_1 = initial_normal(result.samples[0].tangent);
    result.samples[0].normal_2 =
        normalize(cross(result.samples[0].tangent,
                        result.samples[0].normal_1));
    for (std::size_t i = 1; i < result.samples.size(); ++i) {
      const Vec3 previous_tangent = result.samples[i - 1].tangent;
      const Vec3 tangent = result.samples[i].tangent;
      const Vec3 rotation_axis = cross(previous_tangent, tangent);
      const double sin_angle = norm(rotation_axis);
      const double cos_angle =
          std::clamp(dot(previous_tangent, tangent), -1.0, 1.0);
      Vec3 transported = result.samples[i - 1].normal_1;
      if (sin_angle > kFrameTolerance) {
        const Vec3 axis_unit = scale(rotation_axis, 1.0 / sin_angle);
        transported = rodrigues(transported, axis_unit,
                                std::atan2(sin_angle, cos_angle));
      } else if (cos_angle < 0.0) {
        throw std::runtime_error(
            "corridor contains an unresolved 180-degree tangent reversal");
      }
      result.samples[i].normal_1 = project_normal(transported, tangent);
      result.samples[i].normal_2 =
          normalize(cross(tangent, result.samples[i].normal_1));
      if (dot(result.samples[i].normal_1,
              result.samples[i - 1].normal_1) < 0.0) {
        result.samples[i].normal_1 = scale(result.samples[i].normal_1, -1.0);
        result.samples[i].normal_2 = scale(result.samples[i].normal_2, -1.0);
      }
    }
    result.ok = true;
  } catch (const std::exception &error) {
    result.error = error.what();
  }
  return result;
}

Vec3 corridor_point(const CorridorSample &sample, double offset_u_angstrom,
                    double offset_v_angstrom) {
  if (!std::isfinite(offset_u_angstrom) ||
      !std::isfinite(offset_v_angstrom))
    throw std::runtime_error("corridor offsets must be finite");
  if (!std::isfinite(sample.radius_angstrom) || sample.radius_angstrom < 0.0)
    throw std::runtime_error("corridor sample radius must be finite and non-negative");
  const double offset_squared = offset_u_angstrom * offset_u_angstrom +
                                 offset_v_angstrom * offset_v_angstrom;
  const double radius_squared = sample.radius_angstrom * sample.radius_angstrom;
  if (!std::isfinite(offset_squared) || !std::isfinite(radius_squared))
    throw std::runtime_error("corridor squared offsets or radius exceed the finite range");
  if (offset_squared > radius_squared + 1.0e-12)
    throw std::runtime_error("corridor offsets exceed the local route radius");
  const auto point = add(sample.center_angstrom,
                         add(scale(sample.normal_1, offset_u_angstrom),
                             scale(sample.normal_2, offset_v_angstrom)));
  for (const double coordinate : point)
    if (!std::isfinite(coordinate))
      throw std::runtime_error("corridor point is outside the finite range");
  return point;
}

CorridorSample
sample_corridor_at_progress(const std::vector<CorridorSample> &samples,
                            double progress_angstrom) {
  if (samples.empty())
    throw std::runtime_error("cannot sample an empty corridor");
  if (!std::isfinite(progress_angstrom))
    throw std::runtime_error("corridor progress must be finite");
  if (samples.size() == 1)
    return samples.front();

  const double progress =
      std::clamp(progress_angstrom, samples.front().progress_angstrom,
                 samples.back().progress_angstrom);
  const auto right = std::lower_bound(
      samples.begin(), samples.end(), progress,
      [](const CorridorSample &sample, double value) {
        return sample.progress_angstrom < value;
      });
  if (right == samples.begin())
    return samples.front();
  if (right == samples.end())
    return samples.back();
  const auto &b = *right;
  const auto &a = *(right - 1);
  const double width = b.progress_angstrom - a.progress_angstrom;
  const double fraction =
      width <= kFrameTolerance
          ? 0.0
          : std::clamp((progress - a.progress_angstrom) / width, 0.0, 1.0);

  CorridorSample output;
  output.progress_angstrom = progress;
  output.center_angstrom =
      add(a.center_angstrom,
          scale(subtract(b.center_angstrom, a.center_angstrom), fraction));
  output.radius_angstrom =
      a.radius_angstrom + fraction * (b.radius_angstrom - a.radius_angstrom);
  output.tangent = normalize(add(scale(a.tangent, 1.0 - fraction),
                                 scale(b.tangent, fraction)));
  const Vec3 blended_normal =
      add(scale(a.normal_1, 1.0 - fraction), scale(b.normal_1, fraction));
  output.normal_1 = project_normal(blended_normal, output.tangent);
  output.normal_2 = normalize(cross(output.tangent, output.normal_1));
  return output;
}

PathProjection project_point_to_polyline(
    std::span<const ArcLengthSegment> segments, const Vec3 &point_angstrom,
    double reference_progress_angstrom, double minimum_progress_angstrom,
    double maximum_progress_angstrom) {
  if (segments.empty())
    throw std::invalid_argument("cannot project onto an empty polyline");
  for (const double coordinate : point_angstrom)
    if (!std::isfinite(coordinate))
      throw std::invalid_argument("projection point must be finite");
  if (!std::isfinite(reference_progress_angstrom) ||
      !std::isfinite(minimum_progress_angstrom) ||
      std::isnan(maximum_progress_angstrom) ||
      minimum_progress_angstrom < 0.0 ||
      maximum_progress_angstrom < minimum_progress_angstrom) {
    throw std::invalid_argument("invalid polyline projection progress window");
  }

  PathProjection best;
  double best_reference_gap = std::numeric_limits<double>::infinity();
  constexpr double kDistanceTieTolerance = 1.0e-9;
  constexpr double kProgressTieTolerance = 1.0e-9;
  for (std::size_t index = 0; index < segments.size(); ++index) {
    const auto &segment = segments[index];
    if (!std::isfinite(segment.s0) || !std::isfinite(segment.length) ||
        segment.length <= kFrameTolerance)
      throw std::invalid_argument("polyline segment must have positive finite length");
    for (const double coordinate : segment.a)
      if (!std::isfinite(coordinate))
        throw std::invalid_argument("polyline segment origin must be finite");
    for (const double coordinate : segment.d)
      if (!std::isfinite(coordinate))
        throw std::invalid_argument("polyline segment vector must be finite");

    const double segment_begin = segment.s0;
    const double segment_end = segment.s0 + segment.length;
    if (!std::isfinite(segment_end))
      throw std::invalid_argument("polyline segment end is outside the finite range");
    if (segment_end + kProgressTieTolerance < minimum_progress_angstrom ||
        segment_begin - kProgressTieTolerance > maximum_progress_angstrom)
      continue;
    const double clipped_begin =
        std::max(segment_begin, minimum_progress_angstrom);
    const double clipped_end = std::min(segment_end, maximum_progress_angstrom);
    if (clipped_end + kProgressTieTolerance < clipped_begin)
      continue;

    const double begin_t =
        std::clamp((clipped_begin - segment_begin) / segment.length, 0.0, 1.0);
    const double end_t =
        std::clamp((clipped_end - segment_begin) / segment.length, 0.0, 1.0);
    const Vec3 offset = subtract(point_angstrom, segment.a);
    for (const double coordinate : offset)
      if (!std::isfinite(coordinate))
        throw std::invalid_argument("polyline offset is outside the finite range");
    const double denominator = dot(segment.d, segment.d);
    if (!std::isfinite(denominator) || denominator <= kFrameTolerance)
      throw std::invalid_argument("polyline segment vector is singular");
    const double numerator = dot(offset, segment.d);
    if (!std::isfinite(numerator))
      throw std::invalid_argument("polyline projection is outside the finite range");
    const double t = std::clamp(numerator / denominator, begin_t, end_t);
    const Vec3 projected = add(segment.a, scale(segment.d, t));
    const double distance = norm(subtract(point_angstrom, projected));
    const double progress = segment_begin + t * segment.length;
    const double reference_gap =
        std::fabs(progress - reference_progress_angstrom);
    if (!std::isfinite(distance) || !std::isfinite(progress) ||
        !std::isfinite(reference_gap))
      throw std::invalid_argument("polyline projection result is outside the finite range");
    const bool closer = !best.valid ||
                        distance < best.distance_angstrom - kDistanceTieTolerance;
    const bool equal_distance =
        best.valid &&
        std::fabs(distance - best.distance_angstrom) <= kDistanceTieTolerance;
    const bool nearer_reference =
        equal_distance &&
        reference_gap < best_reference_gap - kProgressTieTolerance;
    const bool stable_index_tie =
        equal_distance &&
        std::fabs(reference_gap - best_reference_gap) <=
            kProgressTieTolerance &&
        index < best.segment_index;
    if (closer || nearer_reference || stable_index_tie) {
      best.valid = true;
      best.progress_angstrom = progress;
      best.distance_angstrom = distance;
      best.point_angstrom = projected;
      best.segment_index = index;
      best_reference_gap = reference_gap;
    }
  }
  return best;
}

} // namespace titan_path
