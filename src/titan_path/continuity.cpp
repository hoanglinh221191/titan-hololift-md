#include "titan_path/continuity.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace titan_path {
namespace {
double finite_field(const std::map<std::string, std::string> &fields,
                    const std::string &key) {
  const auto found = fields.find(key);
  if (found == fields.end())
    throw std::invalid_argument("previous route status missing " + key);
  std::size_t end = 0;
  const double value = std::stod(found->second, &end);
  if (end != found->second.size() || !std::isfinite(value))
    throw std::invalid_argument("previous route status invalid " + key);
  return value;
}
} // namespace

PreviousRouteState read_previous_route_state(
    std::istream &input, const std::string &route_id,
    const std::string &route_sha256, const std::string &reference_sha256,
    double total_length_angstrom, double current_time_ps) {
  if (!std::isfinite(total_length_angstrom) || total_length_angstrom <= 0.0 ||
      !std::isfinite(current_time_ps) || current_time_ps < 0.0)
    throw std::invalid_argument(
        "continuity requires finite route length and snapshot time");
  std::map<std::string, std::string> fields;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty() || line.front() == '#')
      continue;
    const auto separator = line.find('=');
    if (separator == std::string::npos || separator == 0 ||
        !fields.emplace(line.substr(0, separator), line.substr(separator + 1))
             .second)
      throw std::invalid_argument(
          "previous route status malformed or duplicate field");
  }
  if (input.bad())
    throw std::invalid_argument("cannot read previous route status");
  if (fields["SCHEMA"] != "TITAN_PATH_STATUS_V1" ||
      fields["STATUS"] != "CONTINUE" || fields["ROUTE_KIND"] != "EXPLICIT" ||
      fields["ROUTE_ID"] != route_id ||
      fields["ROUTE_FILE_SHA256"] != route_sha256 ||
      fields["REFERENCE_SHA256"] != reference_sha256)
    throw std::invalid_argument(
        "previous route status identity/state mismatch");
  PreviousRouteState state;
  if (fields["MODE"] == "Advance")
    state.mode = RouteMode::Advance;
  else if (fields["MODE"] == "Recovery")
    state.mode = RouteMode::Recovery;
  else
    throw std::invalid_argument("previous route status invalid MODE");
  state.progress_angstrom = finite_field(fields, "PATH_PROGRESS_A");
  state.snapshot_time_ps = finite_field(fields, "SNAPSHOT_TIME_PS");
  const double old_total = finite_field(fields, "PATH_TOTAL_A");
  if (std::fabs(old_total - total_length_angstrom) > 1e-6 ||
      state.progress_angstrom < 0.0 ||
      state.progress_angstrom > total_length_angstrom + 1e-8 ||
      state.snapshot_time_ps < 0.0 ||
      state.snapshot_time_ps > current_time_ps + 1e-8)
    throw std::invalid_argument(
        "previous route status progress/length/time mismatch");
  state.progress_angstrom =
      std::min(state.progress_angstrom, total_length_angstrom);
  return state;
}

PathProjection
project_with_continuity(std::span<const ArcLengthSegment> segments,
                        const Vec3 &point, double previous_progress_angstrom,
                        double half_window_angstrom,
                        double total_length_angstrom) {
  if (!std::isfinite(previous_progress_angstrom) ||
      !std::isfinite(half_window_angstrom) || half_window_angstrom <= 0.0 ||
      !std::isfinite(total_length_angstrom) || total_length_angstrom <= 0.0 ||
      previous_progress_angstrom < 0.0 ||
      previous_progress_angstrom > total_length_angstrom)
    throw std::invalid_argument("invalid route continuity window");
  const double lo =
      std::max(0.0, previous_progress_angstrom - half_window_angstrom);
  const double hi = std::min(total_length_angstrom,
                             previous_progress_angstrom + half_window_angstrom);
  const auto selected = project_point_to_polyline(
      segments, point, previous_progress_angstrom, lo, hi);
  if (!selected.valid)
    throw std::invalid_argument(
        "PROJECTION_CONTINUITY_WINDOW_EXHAUSTED: no local projection");
  // Check the derivative on each segment incident to the selected point. A
  // projection pinned to an artificial window boundary with a descent outside
  // is clipped; reporting it as a valid progress would understate movement.
  for (const auto &segment : segments) {
    const double end = segment.s0 + segment.length;
    const double boundary = selected.progress_angstrom;
    if (boundary < segment.s0 - 1e-8 || boundary > end + 1e-8)
      continue;
    Vec3 offset{};
    for (std::size_t k = 0; k < 3; ++k)
      offset[k] = point[k] - segment.a[k];
    const double raw = segment.s0 + dot(offset, segment.d) / segment.length;
    if (!std::isfinite(raw))
      throw std::invalid_argument("nonfinite continuity projection");
    const bool clipped_low = lo > 0.0 && std::fabs(boundary - lo) <= 1e-8 &&
                             segment.s0 < lo && raw < lo - 1e-8;
    const bool clipped_high = hi < total_length_angstrom &&
                              std::fabs(boundary - hi) <= 1e-8 && end > hi &&
                              raw > hi + 1e-8;
    if (clipped_low || clipped_high)
      throw std::invalid_argument(
          "PROJECTION_CONTINUITY_WINDOW_EXHAUSTED: clipped local projection");
  }
  return selected;
}
} // namespace titan_path
