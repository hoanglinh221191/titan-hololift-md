#ifndef TITAN_PATH_GEOMETRY_H
#define TITAN_PATH_GEOMETRY_H

#include "titan_path/route.h"

#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace titan_path {

// Bound all resampling allocations before converting a floating-point count.
inline constexpr std::size_t kMaximumCorridorSamples = 1000000;

struct CorridorSample {
  double progress_angstrom{0.0};
  Vec3 center_angstrom{};
  Vec3 tangent{};
  Vec3 normal_1{};
  Vec3 normal_2{};
  double radius_angstrom{0.0};
};

struct CorridorBuildResult {
  bool ok{false};
  std::vector<CorridorSample> samples;
  std::string error;
};

struct ArcLengthSegment {
  double s0{0.0};
  double length{0.0};
  Vec3 a{};
  Vec3 d{};
};

struct PathProjection {
  bool valid{false};
  double progress_angstrom{0.0};
  double distance_angstrom{std::numeric_limits<double>::infinity()};
  Vec3 point_angstrom{};
  std::size_t segment_index{0};
};

[[nodiscard]] CorridorBuildResult
build_parallel_transport_corridor(const RouteDefinition &route,
                                  double spacing_angstrom);

[[nodiscard]] Vec3 corridor_point(const CorridorSample &sample,
                                  double offset_u_angstrom,
                                  double offset_v_angstrom);

[[nodiscard]] CorridorSample
sample_corridor_at_progress(const std::vector<CorridorSample> &samples,
                            double progress_angstrom);

[[nodiscard]] PathProjection project_point_to_polyline(
    std::span<const ArcLengthSegment> segments, const Vec3 &point_angstrom,
    double reference_progress_angstrom,
    double minimum_progress_angstrom = 0.0,
    double maximum_progress_angstrom =
        std::numeric_limits<double>::infinity());

[[nodiscard]] double dot(const Vec3 &lhs, const Vec3 &rhs);
[[nodiscard]] double norm(const Vec3 &value);

} // namespace titan_path

#endif
