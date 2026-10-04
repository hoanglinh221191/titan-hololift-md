#include "titan_path/transit.h"
#include "titan_path/clearance.h"
#include "titan_path/continuity.h"
#include "titan_path/control.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace titan_path {
namespace {
constexpr double pi = 3.14159265358979323846;
Vec3 sub(const Vec3 &a, const Vec3 &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Vec3 add(const Vec3 &a, const Vec3 &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
Vec3 scale(const Vec3 &a, double b) { return {a[0] * b, a[1] * b, a[2] * b}; }
Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
Vec3 unit(const Vec3 &v) {
  auto u = scale_vector_to_length(v, 1.0);
  if (!u)
    throw std::invalid_argument("singular transit direction");
  return *u;
}
double squared(const Vec3 &a) {
  const double n = norm(a);
  const double v = n * n;
  if (!std::isfinite(v))
    throw std::invalid_argument("nonfinite transit distance");
  return v;
}
void finite(const Vec3 &v) {
  for (double x : v)
    if (!std::isfinite(x) || std::fabs(x) > 1e100)
      throw std::invalid_argument("invalid transit coordinate");
}
bool hash(const std::string &s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
void validate(const TransitSnapshot &s, const TransitPolicy &p) {
  if (!s.geometry_validated || !hash(s.route_sha256) ||
      !hash(s.snapshot_sha256))
    throw std::invalid_argument("unvalidated transit identity");
  if (s.registered_waypoints.size() < 2 ||
      s.registered_waypoints.size() > 10000 ||
      s.waypoint_radii.size() != s.registered_waypoints.size() ||
      s.receptor.empty() || s.ligand.empty() ||
      s.ligand.size() != s.ligand_masses.size())
    throw std::invalid_argument("invalid transit arrays");
  if (s.receptor.size() > 1000000 || s.ligand.size() > 1000000)
    throw std::invalid_argument("transit atom limit");
  for (const auto &v : s.registered_waypoints)
    finite(v);
  for (const auto &v : s.receptor)
    finite(v);
  for (const auto &v : s.ligand)
    finite(v);
  for (double r : s.waypoint_radii)
    if (!std::isfinite(r) || r < 0)
      throw std::invalid_argument("invalid route radius");
  for (double m : s.ligand_masses)
    if (!std::isfinite(m) || m <= 0)
      throw std::invalid_argument("invalid ligand mass");
  finite(s.old_direction);
  (void)unit(s.old_direction);
  for (double v :
       {p.clearance_angstrom, p.turn_cap_deg, p.lookahead_angstrom,
        p.exit_progress_angstrom, p.merit_decrease_angstrom2,
        p.continuity_half_window_angstrom, p.ambiguity_distance_angstrom,
        p.ambiguity_progress_separation_angstrom, p.max_total_travel_angstrom,
        p.max_seed_displacement_angstrom})
    if (!std::isfinite(v) || v <= 0 || v > 1e50)
      throw std::invalid_argument("invalid transit policy");
  if (p.turn_cap_deg > 180 || p.max_steps == 0 || p.max_steps > 64 ||
      p.beam_width == 0 || p.beam_width > 64 || p.candidate_budget == 0 ||
      p.candidate_budget > 100000)
    throw std::invalid_argument("invalid transit budget");
  double previous = std::numeric_limits<double>::infinity();
  for (double l : p.probe_lengths_angstrom) {
    if (!std::isfinite(l) || l <= 0 || l >= previous ||
        l > p.continuity_half_window_angstrom)
      throw std::invalid_argument("invalid transit probe ladder");
    previous = l;
  }
}
struct SoA {
  std::vector<double> x, y, z;
  explicit SoA(const std::vector<Vec3> &points) {
    for (const auto &p : points) {
      x.push_back(p[0]);
      y.push_back(p[1]);
      z.push_back(p[2]);
    }
  }
  CartesianCoordinatesView view() const { return {x, y, z}; }
};
std::vector<Vec3> directions(const Vec3 &old, const Vec3 &goal,
                             const Vec3 &tangent, const Vec3 &lateral_hint,
                             double cap) {
  std::vector<Vec3> result;
  auto append = [&](const Vec3 &request) {
    const auto turn = project_vector_to_turn_cone(request, old, cap);
    if (!turn.valid)
      return;
    const auto scaled = scale_vector_to_length(turn.vector, 1.0);
    if (!scaled)
      return;
    if (std::none_of(result.begin(), result.end(), [&](const Vec3 &v) {
          return norm(sub(v, *scaled)) < 1e-10;
        }))
      result.push_back(*scaled);
  };
  append(old);
  append(goal);
  append(tangent);
  // Orient the finite cone sampling from geometry, not lab Cartesian axes.
  Vec3 transverse = sub(goal, scale(old, dot(goal, old)));
  if (norm(transverse) <= 1e-10)
    transverse = sub(tangent, scale(old, dot(tangent, old)));
  if (norm(transverse) <= 1e-10)
    transverse = sub(lateral_hint, scale(old, dot(lateral_hint, old)));
  if (norm(transverse) <= 1e-10) {
    Vec3 axis{1, 0, 0};
    if (std::fabs(old[1]) < std::fabs(old[0]))
      axis = {0, 1, 0};
    if (std::fabs(old[2]) < std::fabs(dot(old, axis)))
      axis = {0, 0, 1};
    transverse = cross(old, axis);
  }
  const auto e1 = unit(transverse);
  const auto e2 = unit(cross(old, e1));
  for (int ring = 1; ring <= 2; ++ring) {
    const double theta = cap * pi / 180.0 * ring / 2.0;
    for (int i = 0; i < 8; ++i) {
      const double phi = 2 * pi * i / 8;
      append(add(scale(old, std::cos(theta)),
                 scale(add(scale(e1, std::cos(phi)), scale(e2, std::sin(phi))),
                       std::sin(theta))));
    }
  }
  return result;
}
bool ambiguous(const std::vector<ArcLengthSegment> &segments, const Vec3 &point,
               const PathProjection &selected, double previous,
               const TransitPolicy &p) {
  const double total = segments.back().s0 + segments.back().length;
  const double lo = std::max(0.0, previous - p.continuity_half_window_angstrom),
               hi = std::min(total,
                             previous + p.continuity_half_window_angstrom);
  for (const auto &segment : segments) {
    const auto candidate = project_point_to_polyline(
        std::span<const ArcLengthSegment>(&segment, 1), point, previous);
    if (candidate.progress_angstrom < lo - 1e-9 ||
        candidate.progress_angstrom > hi + 1e-9 ||
        std::fabs(candidate.progress_angstrom - selected.progress_angstrom) <
            p.ambiguity_progress_separation_angstrom ||
        candidate.distance_angstrom >
            selected.distance_angstrom + p.ambiguity_distance_angstrom)
      continue;
    bool descent = false;
    for (const auto &incident : segments) {
      const double raw = incident.s0 + dot(sub(point, incident.a), incident.d) /
                                           incident.length;
      if ((std::fabs(candidate.progress_angstrom - incident.s0) <= 1e-8 &&
           raw > incident.s0 + 1e-8) ||
          (std::fabs(candidate.progress_angstrom - incident.s0 -
                     incident.length) <= 1e-8 &&
           raw < incident.s0 + incident.length - 1e-8)) {
        descent = true;
        break;
      }
    }
    if (!descent)
      return true;
  }
  return false;
}
struct Node {
  std::vector<TransitModelStep> path;
  std::size_t serial{0};
};
bool better(const Node &a, const Node &b) {
  const auto &x = a.path.back();
  const auto &y = b.path.back();
  if (x.merit_angstrom2 != y.merit_angstrom2)
    return x.merit_angstrom2 < y.merit_angstrom2;
  if (x.travel_angstrom != y.travel_angstrom)
    return x.travel_angstrom < y.travel_angstrom;
  return a.serial < b.serial;
}
} // namespace

const char *transit_phase_name(TransitPhase p) noexcept {
  switch (p) {
  case TransitPhase::Approach:
    return "APPROACH";
  case TransitPhase::CornerTransit:
    return "CORNER_TRANSIT";
  case TransitPhase::HandoffAdvance:
    return "HANDOFF_ADVANCE_MODEL_ONLY";
  case TransitPhase::NotApplicable:
    return "NOT_APPLICABLE";
  }
  return "INVALID";
}

TransitPlan plan_bounded_transit(const TransitSnapshot &s,
                                 const TransitPolicy &p) {
  validate(s, p);
  TransitPlan result;
  std::vector<ArcLengthSegment> segments;
  std::vector<double> arc{0.0};
  for (std::size_t i = 1; i < s.registered_waypoints.size(); ++i) {
    const auto d =
        sub(s.registered_waypoints[i], s.registered_waypoints[i - 1]);
    const double length = norm(d);
    if (!std::isfinite(length) || length <= 1e-8)
      throw std::invalid_argument("degenerate transit segment");
    segments.push_back({arc.back(), length, s.registered_waypoints[i - 1], d});
    arc.push_back(arc.back() + length);
  }
  Vec3 com{};
  double mass = 0;
  for (double m : s.ligand_masses)
    mass += m;
  if (!std::isfinite(mass))
    throw std::invalid_argument("ligand mass overflow");
  for (std::size_t i = 0; i < s.ligand.size(); ++i)
    com = add(com, scale(s.ligand[i], s.ligand_masses[i] / mass));
  finite(com);
  const auto initial = project_point_to_polyline(segments, com, 0.0);
  const auto first_tangent = unit(segments.front().d);
  if (initial.progress_angstrom <= 1e-8 &&
      dot(sub(com, s.registered_waypoints.front()), first_tangent) < -1e-8) {
    result.initial_phase = TransitPhase::Approach;
    result.vertex = 0;
  } else
    for (std::size_t vertex = 1; vertex + 1 < arc.size(); ++vertex) {
      if (std::fabs(initial.progress_angstrom - arc[vertex]) <= 1e-8 &&
          dot(unit(segments[vertex - 1].d), unit(segments[vertex].d)) <
              1.0 - 1e-8) {
        result.initial_phase = TransitPhase::CornerTransit;
        result.vertex = vertex;
        break;
      }
    }
  if (result.initial_phase == TransitPhase::NotApplicable) {
    result.reason = "TRANSIT_NOT_APPLICABLE";
    return result;
  }
  const auto outgoing = unit(segments[result.vertex].d);
  const double lookahead =
      std::min(p.lookahead_angstrom, segments[result.vertex].length / 2.0);
  result.fixed_goal =
      add(s.registered_waypoints[result.vertex], scale(outgoing, lookahead));
  const double radial_limit =
      std::max(initial.distance_angstrom, s.waypoint_radii[result.vertex]);
  const SoA receptor(s.receptor);
  const SoA ligand(s.ligand);
  const auto initial_sweep = evaluate_translation_sweep(
      receptor.view(), ligand.view(), {0, 0, 0}, p.clearance_angstrom);
  result.counters.swept_pair_checks += initial_sweep.pair_checks;
  TransitModelStep start{result.initial_phase,
                         com,
                         {0, 0, 0},
                         unit(s.old_direction),
                         initial.progress_angstrom,
                         initial.distance_angstrom,
                         squared(sub(com, result.fixed_goal)),
                         0,
                         0,
                         initial_sweep.witness_distance_angstrom};
  result.model_steps.push_back(start);
  if (!initial_sweep.passes_threshold) {
    result.reason = "INITIAL_CLEARANCE_VIOLATION";
    return result;
  }
  if (ambiguous(segments, com, initial, initial.progress_angstrom, p)) {
    result.reason = "INITIAL_PROJECTION_AMBIGUOUS";
    return result;
  }
  Node best{{start}, 0};
  std::vector<Node> beam{best};
  std::size_t serial = 0;
  for (std::size_t depth = 0; depth < p.max_steps; ++depth) {
    const auto before_level = result.counters;
    std::vector<Node> children, exits;
    for (const auto &node : beam) {
      const auto &state = node.path.back();
      std::vector<Vec3> moving = s.ligand;
      const auto total_shift = sub(state.com, com);
      for (auto &point : moving)
        point = add(point, total_shift);
      const SoA moved(moving);
      Vec3 lateral_hint{};
      for (const auto &fixed : s.receptor) {
        const auto hint = sub(fixed, state.com);
        if (norm(cross(state.direction, hint)) > 1e-10) {
          lateral_hint = hint;
          break;
        }
      }
      for (const auto &direction :
           directions(state.direction, sub(result.fixed_goal, state.com),
                      outgoing, lateral_hint, p.turn_cap_deg)) {
        for (double length : p.probe_lengths_angstrom) {
          ++result.counters.proposals;
          if (result.counters.evaluations >= p.candidate_budget) {
            if (!exits.empty()) {
              std::sort(exits.begin(), exits.end(), better);
              result.found = true;
              result.reason = "GEOMETRY_PLAN_FOUND_BUDGET_LIMITED";
              result.model_steps = std::move(exits.front().path);
            } else {
              result.reason = "TRANSIT_CANDIDATE_BUDGET_EXHAUSTED";
              result.model_steps = best.path;
            }
            return result;
          }
          ++result.counters.evaluations;
          const auto displacement = scale(direction, length);
          const auto next_com = add(state.com, displacement);
          const double travel = state.travel_angstrom + length;
          if (travel > p.max_total_travel_angstrom + 1e-10 ||
              norm(sub(next_com, com)) >
                  p.max_seed_displacement_angstrom + 1e-10) {
            ++result.counters.bound_rejections;
            continue;
          }
          PathProjection projection;
          try {
            projection = project_with_continuity(
                segments, next_com, state.progress_angstrom,
                p.continuity_half_window_angstrom, arc.back());
          } catch (const std::invalid_argument &) {
            ++result.counters.continuity_rejections;
            continue;
          }
          if (ambiguous(segments, next_com, projection, state.progress_angstrom,
                        p)) {
            ++result.counters.ambiguity_rejections;
            continue;
          }
          if (projection.progress_angstrom < state.progress_angstrom - 1e-8) {
            ++result.counters.backtrack_rejections;
            continue;
          }
          if (projection.distance_angstrom > radial_limit + 1e-8) {
            ++result.counters.corridor_rejections;
            continue;
          }
          const double merit = squared(sub(next_com, result.fixed_goal));
          if (merit >= state.merit_angstrom2 - p.merit_decrease_angstrom2) {
            ++result.counters.merit_rejections;
            continue;
          }
          const auto sweep =
              evaluate_translation_sweep(receptor.view(), moved.view(),
                                         displacement, p.clearance_angstrom);
          result.counters.swept_pair_checks += sweep.pair_checks;
          if (!sweep.passes_threshold) {
            ++result.counters.clearance_rejections;
            continue;
          }
          ++result.counters.accepted_edges;
          const double angle =
              std::atan2(
                  norm(cross(state.direction, direction)),
                  std::clamp(dot(state.direction, direction), -1.0, 1.0)) *
              180 / pi;
          if (angle > p.turn_cap_deg + 1e-8)
            throw std::runtime_error("transit cone violation");
          bool exit = projection.progress_angstrom >=
                          arc[result.vertex] + p.exit_progress_angstrom &&
                      projection.segment_index >= result.vertex;
          if (result.initial_phase == TransitPhase::Approach)
            exit =
                exit &&
                dot(sub(next_com, s.registered_waypoints.front()),
                    first_tangent) >= 0 &&
                projection.distance_angstrom <= s.waypoint_radii.front() + 1e-8;
          TransitModelStep step{exit ? TransitPhase::HandoffAdvance
                                     : result.initial_phase,
                                next_com,
                                displacement,
                                direction,
                                projection.progress_angstrom,
                                projection.distance_angstrom,
                                merit,
                                travel,
                                angle,
                                sweep.witness_distance_angstrom};
          Node child = node;
          child.path.push_back(step);
          child.serial = ++serial;
          if (better(child, best))
            best = child;
          if (exit)
            exits.push_back(std::move(child));
          else
            children.push_back(std::move(child));
        }
      }
    }
    if (!exits.empty()) {
      std::sort(exits.begin(), exits.end(), better);
      result.found = true;
      result.reason = "GEOMETRY_PLAN_FOUND";
      result.model_steps = std::move(exits.front().path);
      return result;
    }
    if (children.empty()) {
      result.reason =
          result.counters.clearance_rejections >
                  before_level.clearance_rejections
              ? "TRANSIT_CLEARANCE_EXHAUSTED"
          : result.counters.merit_rejections > before_level.merit_rejections
              ? "TRANSIT_MERIT_EXHAUSTED"
          : result.counters.corridor_rejections >
                  before_level.corridor_rejections
              ? "TRANSIT_CORRIDOR_EXHAUSTED"
              : "TRANSIT_LOCAL_CONSTRAINTS_EXHAUSTED";
      result.model_steps = best.path;
      return result;
    }
    std::sort(children.begin(), children.end(), better);
    if (children.size() > p.beam_width)
      children.resize(p.beam_width);
    beam = std::move(children);
  }
  result.reason = "TRANSIT_STEP_LIMIT";
  result.model_steps = best.path;
  return result;
}
} // namespace titan_path
