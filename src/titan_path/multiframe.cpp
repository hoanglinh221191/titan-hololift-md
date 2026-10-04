#include "titan_path/multiframe.h"
#include "titan_path/clearance.h"
#include "titan_path/continuity.h"
#include "titan_path/registration.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace titan_path {
namespace {
void check(bool ok, const char *reason) {
  if (!ok)
    throw std::invalid_argument(reason);
}
void check_xyz(const std::vector<Vec3> &points, std::size_t minimum) {
  check(points.size() >= minimum, "MULTIFRAME_MISSING_COORDINATES");
  for (const auto &p : points)
    for (double x : p)
      check(std::isfinite(x) && std::abs(x) < 1e6,
            "MULTIFRAME_INVALID_COORDINATE");
}
double shape(const std::vector<Vec3> &points) {
  Eigen::MatrixXd a(points.size(), 3);
  for (std::size_t i = 0; i < points.size(); ++i)
    for (int j = 0; j < 3; ++j)
      a(i, j) = points[i][j];
  a.rowwise() -= a.colwise().mean().eval();
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(a);
  auto s = svd.singularValues();
  return s[0] > 1e-10 ? s[1] / s[0] : 0.;
}
std::vector<ArcLengthSegment> segments(const std::vector<Vec3> &points) {
  std::vector<ArcLengthSegment> result;
  double arc = 0;
  for (std::size_t i = 1; i < points.size(); ++i) {
    Vec3 d{};
    for (int j = 0; j < 3; ++j)
      d[j] = points[i][j] - points[i - 1][j];
    double length = norm(d);
    check(std::isfinite(length) && length > 1e-10,
          "MULTIFRAME_DEGENERATE_ROUTE");
    result.push_back({arc, length, points[i - 1], d});
    arc += length;
  }
  return result;
}
bool ambiguous(const std::vector<ArcLengthSegment> &route, const Vec3 &point,
               const PathProjection &chosen, double previous, double total) {
  std::vector<double> seen;
  for (const auto &segment : route) {
    const auto candidate =
        project_point_to_polyline(std::span(&segment, 1), point, previous);
    bool descent = false;
    for (const auto &incident : route) {
      Vec3 offset{};
      for (int k = 0; k < 3; ++k)
        offset[k] = point[k] - incident.a[k];
      const double raw =
          incident.s0 + dot(offset, incident.d) / incident.length;
      if (std::abs(candidate.progress_angstrom - incident.s0) <= 1e-8 &&
          raw > incident.s0 + 1e-8)
        descent = true;
      if (std::abs(candidate.progress_angstrom - incident.s0 -
                   incident.length) <= 1e-8 &&
          raw < incident.s0 + incident.length - 1e-8)
        descent = true;
    }
    const double s = candidate.progress_angstrom;
    if (descent || std::any_of(seen.begin(), seen.end(), [s](double old) {
          return std::abs(s - old) <= 1e-8;
        }))
      continue;
    seen.push_back(s);
    if (s >= std::max(0., previous - 1.) - 1e-9 &&
        s <= std::min(total, previous + 1.) + 1e-9 &&
        std::abs(s - chosen.progress_angstrom) >= .25 &&
        candidate.distance_angstrom <= chosen.distance_angstrom + .05)
      return true;
  }
  return false;
}
struct Arrays {
  std::vector<double> x, y, z;
  explicit Arrays(const std::vector<Vec3> &p) {
    for (auto a : p) {
      x.push_back(a[0]);
      y.push_back(a[1]);
      z.push_back(a[2]);
    }
  }
  CartesianCoordinatesView view() const { return {x, y, z}; }
};
} // namespace

MultiFrameResult validate_multiframe_direction(const MultiFrameInput &in) {
  check_xyz(in.reference_core, 6);
  check_xyz(in.current_core, 6);
  check_xyz(in.waypoints, 2);
  check_xyz(in.receptor, 1);
  check_xyz(in.ligand, 1);
  check(in.reference_core.size() == in.current_core.size() &&
            in.reference_core.size() <= 10000,
        "MULTIFRAME_CORE_MAPPING_MISMATCH");
  check(in.ligand.size() == in.ligand_masses.size(),
        "MULTIFRAME_MASS_MAPPING_MISMATCH");
  check(!in.previous_mode || *in.previous_mode == RouteMode::Advance ||
            *in.previous_mode == RouteMode::Recovery,
        "MULTIFRAME_INVALID_PREVIOUS_MODE");
  check(in.native_mode == RouteMode::Advance ||
            in.native_mode == RouteMode::Recovery,
        "MULTIFRAME_INVALID_NATIVE_MODE");
  if (in.observed_progress)
    for (double value : *in.observed_progress)
      check(std::isfinite(value) && value >= 0 && value < 1e6,
            "MULTIFRAME_INVALID_OBSERVED_PROGRESS");
  Vec3 com{};
  double mass = 0;
  for (std::size_t i = 0; i < in.ligand.size(); ++i) {
    const double m = in.ligand_masses[i];
    check(std::isfinite(m) && m > 0 && m < 1e6, "MULTIFRAME_INVALID_MASS");
    mass += m;
    for (int j = 0; j < 3; ++j)
      com[j] += m * in.ligand[i][j];
  }
  for (auto &x : com)
    x /= mass;
  for (double x : in.old_direction)
    check(std::isfinite(x) && std::abs(x) < 1e6,
          "MULTIFRAME_INVALID_OLD_VECTOR");
  for (double x : in.displacement)
    check(std::isfinite(x) && std::abs(x) < 1e6, "MULTIFRAME_INVALID_VECTOR");
  check(norm(in.old_direction) > 1e-12, "MULTIFRAME_ZERO_OLD_VECTOR");
  MultiFrameResult result;
  if (std::abs(norm(in.displacement) - .25) > 1e-6) {
    result.reason = "STOP_PROBE_DISPLACEMENT_MISMATCH";
    return result;
  }
  const auto turn =
      project_vector_to_turn_cone(in.displacement, in.old_direction, 180.);
  check(turn.valid, "MULTIFRAME_INVALID_TURN");
  result.turn_deg = turn.requested_angle_deg;
  const Arrays receptor(in.receptor), ligand(in.ligand);
  const auto sweep = evaluate_translation_sweep(receptor.view(), ligand.view(),
                                                in.displacement, 1.5);
  result.clearance = sweep.witness_distance_angstrom;
  if (!sweep.passes_threshold) {
    result.reason = "STOP_CLEARANCE";
    return result;
  }
  for (int f = 0; f < 3; ++f) {
    MultiFrameMetrics row;
    row.name = f == 0 ? "full_core" : (f == 1 ? "split_even" : "split_odd");
    std::vector<Vec3> ref, cur;
    for (std::size_t i = 0; i < in.reference_core.size(); ++i)
      if (f == 0 || i % 2 == std::size_t(f - 1)) {
        ref.push_back(in.reference_core[i]);
        cur.push_back(in.current_core[i]);
      }
    row.shape_ratio = std::min(shape(ref), shape(cur));
    if (row.shape_ratio < .001) {
      row.reason = "STOP_FRAME_REGISTRATION";
      result.frames.push_back(row);
      continue;
    }
    const auto fit = fit_rigid_transform(ref, cur);
    check(fit.valid, "MULTIFRAME_FIT_FAILED");
    row.rmsd = fit.rmsd_angstrom;
    std::vector<Vec3> points;
    for (auto p : in.waypoints)
      points.push_back(apply_transform(fit, p));
    const auto route = segments(points);
    const double total = route.back().s0 + route.back().length;
    PathProjection start;
    try {
      start = in.observed_progress
                  ? project_with_continuity(route, com, (*in.observed_progress)[f], 1., total)
                  : project_point_to_polyline(route, com, 0.);
    } catch (const std::invalid_argument &) {
      row.reason = "STOP_FRAME_CONTINUITY";
      result.frames.push_back(row);
      continue;
    }
    check(start.valid, "MULTIFRAME_PROJECTION_FAILED");
    if (in.observed_progress &&
        std::abs(start.progress_angstrom-(*in.observed_progress)[f]) > .03) {
      row.reason = "STOP_OBSERVED_PROGRESS_MISMATCH";
      result.frames.push_back(row);
      continue;
    }
    row.start_progress = start.progress_angstrom;
    row.start_distance = start.distance_angstrom;
    row.start_available = true;
    row.mode = next_route_mode(in.previous_mode, row.start_distance, 4., 3.);
    Vec3 target = com;
    for (int k = 0; k < 3; ++k)
      target[k] += in.displacement[k];
    if (ambiguous(route, com, start, start.progress_angstrom, total))
      row.reason = "STOP_FRAME_AMBIGUOUS";
    else {
      try {
        const auto end = project_with_continuity(
            route, target, start.progress_angstrom, 1., total);
        row.end_progress = end.progress_angstrom;
        row.end_distance = end.distance_angstrom;
        row.end_available = true;
        if (ambiguous(route, target, end, start.progress_angstrom, total))
          row.reason = "STOP_FRAME_AMBIGUOUS";
        else if (row.mode != in.native_mode)
          row.reason = "STOP_FRAME_MODE_DISAGREEMENT";
        else if (result.turn_deg >
                 (row.mode == RouteMode::Advance ? 20. : 60.) + 1e-6)
          row.reason = "STOP_TURN_CAP";
        else if (row.mode == RouteMode::Advance &&
                 row.end_progress - row.start_progress <= 1e-6)
          row.reason = "STOP_FRAME_FORWARD_PROGRESS";
        else if (row.mode == RouteMode::Advance && row.end_distance > 4. + 1e-9)
          row.reason = "STOP_FRAME_DISTANCE_BOUND";
        else if (row.mode == RouteMode::Recovery &&
                 row.end_distance - row.start_distance >= -1e-6)
          row.reason = "STOP_FRAME_RECOVERY_DESCENT";
        else if (row.mode == RouteMode::Recovery &&
                 row.end_progress - row.start_progress < -1e-6)
          row.reason = "STOP_FRAME_BACKTRACK";
        else {
          row.passed = true;
          row.reason = "OK";
        }
      } catch (const std::invalid_argument &e) {
        if (std::string(e.what()).find(
                "PROJECTION_CONTINUITY_WINDOW_EXHAUSTED") == std::string::npos)
          throw;
        row.reason = "STOP_FRAME_CONTINUITY";
      }
    }
    result.frames.push_back(row);
  }
  result.accepted = std::all_of(result.frames.begin(), result.frames.end(),
                                [](const auto &f) { return f.passed; });
  result.reason = result.accepted ? "ACCEPT_NATIVE_DIRECTION_NO_MD"
                                  : "STOP_FRAME_DISAGREEMENT";
  // A shared turn-cap failure is not disagreement between reference fits.
  // Preserve every frame's detailed result and leave all decision thresholds
  // unchanged; only refine the aggregate reason when all frames agree on it.
  if (std::all_of(result.frames.begin(), result.frames.end(),
                  [](const auto &f) { return f.reason == "STOP_TURN_CAP"; }))
    result.reason = "STOP_TURN_CAP";
  return result;
}
} // namespace titan_path
