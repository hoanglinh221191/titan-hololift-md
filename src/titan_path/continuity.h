#ifndef TITAN_PATH_CONTINUITY_H
#define TITAN_PATH_CONTINUITY_H

#include "titan_path/control.h"
#include "titan_path/geometry.h"
#include <istream>
#include <string>

namespace titan_path {

struct PreviousRouteState {
  RouteMode mode{RouteMode::Advance};
  double progress_angstrom{0.0};
  double snapshot_time_ps{0.0};
};

// Explicit cross-call state; no implicit global nearest-branch reset.
// Throws on duplicate/missing fields, terminal state, identity/time mismatch.
[[nodiscard]] PreviousRouteState
read_previous_route_state(std::istream &input, const std::string &route_id,
                          const std::string &route_sha256,
                          const std::string &reference_sha256,
                          double total_length_angstrom, double current_time_ps);

// A clipped artificial window edge is an uncertain route coordinate, not
// progress. Reject it; never widen the window automatically or emit a NEXT.
[[nodiscard]] PathProjection
project_with_continuity(std::span<const ArcLengthSegment> segments,
                        const Vec3 &point, double previous_progress_angstrom,
                        double half_window_angstrom,
                        double total_length_angstrom);

} // namespace titan_path
#endif
