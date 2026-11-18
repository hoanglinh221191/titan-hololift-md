#include "titan_path/control.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace titan_path {
namespace {

constexpr double kProgressEpsilon = 1.0e-6;
constexpr double kPi = 3.14159265358979323846;

double vector_norm(const std::array<double, 3> &vector) {
  return std::hypot(vector[0], vector[1], vector[2]);
}

std::array<double, 3> cross(const std::array<double, 3> &lhs,
                            const std::array<double, 3> &rhs) {
  return {lhs[1] * rhs[2] - lhs[2] * rhs[1],
          lhs[2] * rhs[0] - lhs[0] * rhs[2],
          lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}

double unit_vector_angle(const std::array<double, 3> &lhs,
                         const std::array<double, 3> &rhs) {
  const double cosine = std::clamp(
      lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2], -1.0, 1.0);
  return std::atan2(vector_norm(cross(lhs, rhs)), cosine);
}

bool recovery_better(const SelectionCandidate &lhs, std::size_t lhs_index,
                      const SelectionCandidate &rhs, std::size_t rhs_index) {
  if (std::fabs(lhs.path_distance - rhs.path_distance) > 1.0e-9)
    return lhs.path_distance < rhs.path_distance;
  if (std::fabs(lhs.delta_progress - rhs.delta_progress) > kProgressEpsilon)
    return lhs.delta_progress > rhs.delta_progress;
  if (std::fabs(lhs.pull_distance - rhs.pull_distance) > 1.0e-9)
    return lhs.pull_distance < rhs.pull_distance;
  if (std::fabs(lhs.energy_key - rhs.energy_key) > 1.0e-12)
    return lhs.energy_key < rhs.energy_key;
  return lhs_index < rhs_index;
}

bool progress_better(const SelectionCandidate &lhs, std::size_t lhs_index,
                     const SelectionCandidate &rhs, std::size_t rhs_index,
                     bool geometry_only, bool navigation_enabled) {
  if (std::fabs(lhs.delta_progress - rhs.delta_progress) > kProgressEpsilon)
    return lhs.delta_progress > rhs.delta_progress;
  if (navigation_enabled &&
      std::fabs(lhs.navigation_cost - rhs.navigation_cost) > 1.0e-12)
    return lhs.navigation_cost < rhs.navigation_cost;
  if (!geometry_only &&
      std::fabs(lhs.energy_key - rhs.energy_key) > 1.0e-12)
    return lhs.energy_key < rhs.energy_key;
  if (std::fabs(lhs.minimum_clearance - rhs.minimum_clearance) > 1.0e-9)
    return lhs.minimum_clearance > rhs.minimum_clearance;
  if (std::isfinite(lhs.path_distance) && std::isfinite(rhs.path_distance) &&
      std::fabs(lhs.path_distance - rhs.path_distance) > 1.0e-9)
    return lhs.path_distance < rhs.path_distance;
  if (std::fabs(lhs.pull_distance - rhs.pull_distance) > 1.0e-9)
    return lhs.pull_distance < rhs.pull_distance;
  if (std::fabs(lhs.angle_deg - rhs.angle_deg) > 1.0e-9)
    return lhs.angle_deg < rhs.angle_deg;
  return lhs_index < rhs_index;
}

} // namespace

TurnProjectionResult project_vector_to_turn_cone(
    const std::array<double, 3> &requested_vector,
    const std::array<double, 3> &old_vector, double max_turn_deg) {
  if (!std::isfinite(max_turn_deg) || max_turn_deg <= 0.0 ||
      max_turn_deg > 180.0)
    throw std::invalid_argument("maximum turn angle must be in (0, 180]");
  for (const double value : requested_vector)
    if (!std::isfinite(value))
      throw std::invalid_argument("requested turn vector must be finite");
  for (const double value : old_vector)
    if (!std::isfinite(value))
      throw std::invalid_argument("old turn vector must be finite");

  TurnProjectionResult result;
  const double requested_norm = vector_norm(requested_vector);
  const double old_norm = vector_norm(old_vector);
  if (!std::isfinite(requested_norm) || !std::isfinite(old_norm))
    throw std::invalid_argument("turn-vector norm is outside the finite range");
  if (requested_norm <= 1.0e-12)
    return result;
  result.vector = requested_vector;
  result.valid = true;
  if (old_norm <= 1.0e-12)
    return result;

  std::array<double, 3> requested_unit{};
  std::array<double, 3> old_unit{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    requested_unit[axis] = requested_vector[axis] / requested_norm;
    old_unit[axis] = old_vector[axis] / old_norm;
  }
  const double angle = unit_vector_angle(old_unit, requested_unit);
  result.requested_angle_deg = angle * 180.0 / kPi;
  const double max_turn_rad = max_turn_deg * kPi / 180.0;
  if (angle <= max_turn_rad + 1.0e-12) {
    result.applied_angle_deg = result.requested_angle_deg;
    return result;
  }

  // Construct the great-circle tangent directly. Slerp weights divided by
  // sin(acos(dot)) lose precision near antiparallel vectors, allowing the
  // applied direction to exceed the stated turn cap.
  const auto rotation_axis = cross(old_unit, requested_unit);
  auto tangent = cross(rotation_axis, old_unit);
  const double tangent_norm = vector_norm(tangent);
  if (!std::isfinite(tangent_norm) || tangent_norm <= 1.0e-12) {
    result.valid = false;
    return result;
  }
  std::array<double, 3> applied_unit{};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    tangent[axis] /= tangent_norm;
    applied_unit[axis] = std::cos(max_turn_rad) * old_unit[axis] +
                         std::sin(max_turn_rad) * tangent[axis];
  }
  const double applied_norm = vector_norm(applied_unit);
  if (!std::isfinite(applied_norm) || applied_norm <= 1.0e-12) {
    result.valid = false;
    return result;
  }
  for (std::size_t axis = 0; axis < 3; ++axis) {
    applied_unit[axis] /= applied_norm;
    result.vector[axis] = requested_norm * applied_unit[axis];
    if (!std::isfinite(result.vector[axis])) {
      result.valid = false;
      return result;
    }
  }
  const double emitted_norm = vector_norm(result.vector);
  if (!std::isfinite(emitted_norm) || emitted_norm <= 1.0e-12 ||
      std::fabs(emitted_norm / requested_norm - 1.0) > 1.0e-12) {
    result.valid = false;
    return result;
  }
  for (std::size_t axis = 0; axis < 3; ++axis)
    applied_unit[axis] = result.vector[axis] / emitted_norm;
  result.applied_angle_deg =
      unit_vector_angle(old_unit, applied_unit) * 180.0 / kPi;
  if (!std::isfinite(result.applied_angle_deg) ||
      result.applied_angle_deg > max_turn_deg + 1.0e-10) {
    result.valid = false;
    return result;
  }
  result.capped = true;
  return result;
}

std::optional<std::array<double, 3>>
scale_vector_to_length(const std::array<double, 3> &vector,
                       double target_length_angstrom) {
  if (!std::isfinite(target_length_angstrom) || target_length_angstrom <= 0.0)
    throw std::invalid_argument("probe length must be finite and positive");
  for (const double value : vector)
    if (!std::isfinite(value))
      throw std::invalid_argument("probe vector must be finite");
  const double magnitude = vector_norm(vector);
  if (!std::isfinite(magnitude))
    throw std::invalid_argument("probe vector norm is outside the finite range");
  if (magnitude <= 1.0e-12)
    return std::nullopt;
  std::array<double, 3> scaled{};
  for (std::size_t axis = 0; axis < scaled.size(); ++axis)
    scaled[axis] = target_length_angstrom * (vector[axis] / magnitude);
  const double scaled_norm = vector_norm(scaled);
  if (!std::isfinite(scaled_norm) || scaled_norm <= 0.0)
    throw std::invalid_argument("scaled probe norm is outside the finite range");
  return scaled;
}

RouteMode next_route_mode(std::optional<RouteMode> previous_mode,
                          double path_distance_angstrom,
                          double recovery_enter_angstrom,
                          double recovery_exit_angstrom) {
  if (!std::isfinite(path_distance_angstrom) ||
      !std::isfinite(recovery_enter_angstrom) ||
      !std::isfinite(recovery_exit_angstrom) ||
      recovery_exit_angstrom <= 0.0 || recovery_enter_angstrom <= 0.0 ||
      recovery_exit_angstrom > recovery_enter_angstrom) {
    throw std::invalid_argument("invalid route-recovery hysteresis parameters");
  }
  if (previous_mode == RouteMode::Recovery)
    return path_distance_angstrom < recovery_exit_angstrom
               ? RouteMode::Advance
               : RouteMode::Recovery;
  return path_distance_angstrom > recovery_enter_angstrom
             ? RouteMode::Recovery
             : RouteMode::Advance;
}

bool candidate_within_scan_shell(double pull_distance_angstrom,
                                 double minimum_scan_radius_angstrom,
                                 double maximum_scan_radius_angstrom) {
  if (!std::isfinite(pull_distance_angstrom) ||
      !std::isfinite(minimum_scan_radius_angstrom) ||
      !std::isfinite(maximum_scan_radius_angstrom) ||
      pull_distance_angstrom < 0.0 || minimum_scan_radius_angstrom < 0.0 ||
      maximum_scan_radius_angstrom < minimum_scan_radius_angstrom) {
    throw std::invalid_argument("invalid pathway scan-shell parameters");
  }
  return pull_distance_angstrom + 1.0e-9 >= minimum_scan_radius_angstrom &&
         pull_distance_angstrom <= maximum_scan_radius_angstrom + 1.0e-9;
}

CandidateExhaustion classify_candidate_exhaustion(
    std::size_t candidate_proposals, std::size_t candidate_evaluations,
    std::size_t scan_shell_exclusions, std::size_t clearance_rejections,
    std::size_t energy_evaluations) {
  if (candidate_evaluations + scan_shell_exclusions > candidate_proposals ||
      clearance_rejections > candidate_evaluations ||
      energy_evaluations > candidate_evaluations - clearance_rejections) {
    throw std::invalid_argument("inconsistent pathway candidate accounting");
  }
  if (candidate_proposals > 0 && candidate_evaluations == 0 &&
      scan_shell_exclusions == candidate_proposals)
    return CandidateExhaustion::ScanShellExhausted;
  if (candidate_evaluations > 0 &&
      candidate_evaluations + scan_shell_exclusions == candidate_proposals &&
      clearance_rejections == candidate_evaluations &&
      energy_evaluations == 0)
    return CandidateExhaustion::ClearanceExhausted;
  return CandidateExhaustion::None;
}

const char *candidate_exhaustion_name(CandidateExhaustion exhaustion) noexcept {
  switch (exhaustion) {
  case CandidateExhaustion::ScanShellExhausted:
    return "SCAN_SHELL_EXHAUSTED";
  case CandidateExhaustion::ClearanceExhausted:
    return "CLEARANCE_EXHAUSTED";
  case CandidateExhaustion::None:
  default:
    return "NONE";
  }
}

FinalSelectionFailure
classify_final_selection_failure(const SelectionResult &selection,
                                 RouteMode mode) noexcept {
  if (selection.found)
    return FinalSelectionFailure::None;
  if (selection.valid_count == 0)
    return FinalSelectionFailure::NoValidCandidate;
  if (mode == RouteMode::Advance && selection.selectable_count == 0)
    return FinalSelectionFailure::ForwardProgressExhausted;
  if (mode == RouteMode::Recovery && selection.selectable_count == 0)
    return FinalSelectionFailure::RecoveryDescentExhausted;
  return FinalSelectionFailure::EnergySelectionExhausted;
}

const char *
final_selection_failure_name(FinalSelectionFailure failure) noexcept {
  switch (failure) {
  case FinalSelectionFailure::NoValidCandidate:
    return "NO_VALID_CANDIDATE";
  case FinalSelectionFailure::ForwardProgressExhausted:
    return "FORWARD_PROGRESS_EXHAUSTED";
  case FinalSelectionFailure::RecoveryDescentExhausted:
    return "RECOVERY_DESCENT_EXHAUSTED";
  case FinalSelectionFailure::EnergySelectionExhausted:
    return "ENERGY_SELECTION_EXHAUSTED";
  case FinalSelectionFailure::None:
  default:
    return "NONE";
  }
}

static SelectionResult select_epsilon_candidate_impl(
    std::span<const SelectionCandidate> candidates, RouteMode mode,
    double energy_key_epsilon, bool geometry_only,
    bool prefer_maximum_probe_length,
    std::optional<double> navigation_cost_epsilon, bool nonnegative_prefix,
    bool bounded_reroute = false, bool exploratory = false) {
  if (!std::isfinite(energy_key_epsilon) || energy_key_epsilon < 0.0)
    throw std::invalid_argument("energy-key epsilon must be finite and non-negative");
  if (navigation_cost_epsilon.has_value() &&
      (!std::isfinite(*navigation_cost_epsilon) ||
       *navigation_cost_epsilon < 0.0))
    throw std::invalid_argument(
        "navigation-cost epsilon must be finite and non-negative");
  if (navigation_cost_epsilon.has_value() && mode == RouteMode::Recovery)
    throw std::invalid_argument(
        "navigation-cost epsilon is an Advance-only selection stage");

  SelectionResult result;
  std::vector<std::size_t> valid;
  valid.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (!candidates[i].valid || !std::isfinite(candidates[i].energy_key) ||
        !std::isfinite(candidates[i].delta_progress) ||
        !std::isfinite(candidates[i].pull_distance) ||
        !std::isfinite(candidates[i].minimum_clearance) ||
        !std::isfinite(candidates[i].angle_deg) ||
        (navigation_cost_epsilon.has_value() &&
         (!candidates[i].navigation_enabled ||
          !std::isfinite(candidates[i].navigation_cost))) ||
        (mode == RouteMode::Recovery &&
         !std::isfinite(candidates[i].path_distance)))
      continue;
    valid.push_back(i);
  }
  result.valid_count = valid.size();
  if (valid.empty())
    return result;

  std::vector<std::size_t> selectable;
  selectable.reserve(valid.size());
  for (const std::size_t index : valid) {
    if (bounded_reroute && !exploratory && candidates[index].delta_progress < -0.5-kProgressEpsilon)
      continue;
    if (mode == RouteMode::Advance && !bounded_reroute && !exploratory &&
        (nonnegative_prefix ? candidates[index].delta_progress < -kProgressEpsilon
                            : candidates[index].delta_progress <= kProgressEpsilon))
      continue;
    if (mode == RouteMode::Recovery && candidates[index].fixed_probe &&
        std::isfinite(candidates[index].reference_path_distance) &&
        candidates[index].path_distance >=
            candidates[index].reference_path_distance - kProgressEpsilon)
      continue;
    selectable.push_back(index);
  }
  result.mode_eligible_count = selectable.size();
  result.nonforward_count =
      mode == RouteMode::Advance ? result.valid_count - selectable.size() : 0;

  // A shorter trust-region tier may be the first tier that actually advances
  // (or descends toward the route).  Apply the maximum-tier preference only
  // after the mode constraint, otherwise a hard-valid but non-progressing
  // long probe can hide a shorter valid step and create a false STOP.
  if (prefer_maximum_probe_length && !selectable.empty()) {
    double maximum_probe_length = 0.0;
    for (const std::size_t index : selectable) {
      if (!candidates[index].fixed_probe ||
          !std::isfinite(candidates[index].probe_length) ||
          candidates[index].probe_length <= 0.0)
        throw std::invalid_argument(
            "maximum-probe selection requires positive fixed probe lengths");
      maximum_probe_length =
          std::max(maximum_probe_length, candidates[index].probe_length);
    }
    selectable.erase(
        std::remove_if(selectable.begin(), selectable.end(),
                       [&](std::size_t index) {
                         return candidates[index].probe_length <
                                maximum_probe_length - 1.0e-9;
                       }),
        selectable.end());
  }
  result.selectable_count = selectable.size();
  result.maximum_probe_tier_count = selectable.size();
  if (selectable.empty())
    return result;
  for (const std::size_t index : selectable)
    result.minimum_energy_key =
        std::min(result.minimum_energy_key, candidates[index].energy_key);

  std::vector<std::size_t> eligible;
  eligible.reserve(selectable.size());
  if (mode == RouteMode::Recovery || geometry_only) {
    eligible = selectable;
  } else {
    for (const std::size_t index : selectable)
      if (candidates[index].energy_key <=
          result.minimum_energy_key + energy_key_epsilon + 1.0e-12)
        eligible.push_back(index);
  }
  result.eligible_count = eligible.size();
  if (eligible.empty())
    return result;

  std::vector<std::size_t> navigation_eligible;
  if (!navigation_cost_epsilon.has_value()) {
    navigation_eligible = eligible;
  } else {
    for (const std::size_t index : eligible)
      result.minimum_navigation_cost = std::min(
          result.minimum_navigation_cost, candidates[index].navigation_cost);
    for (const std::size_t index : eligible)
      if (candidates[index].navigation_cost <=
          result.minimum_navigation_cost + *navigation_cost_epsilon + 1.0e-12)
        navigation_eligible.push_back(index);
  }
  result.navigation_eligible_count = navigation_eligible.size();
  if (navigation_eligible.empty())
    return result;

  std::size_t best = navigation_eligible.front();
  for (const std::size_t index : navigation_eligible) {
    const bool better =
        mode == RouteMode::Recovery
            ? recovery_better(candidates[index], index, candidates[best], best)
            : progress_better(candidates[index], index, candidates[best], best,
                              geometry_only,
                              navigation_cost_epsilon.has_value());
    if (better)
      best = index;
  }
  result.found = true;
  result.index = best;
  return result;
}

SelectionResult select_epsilon_progress_candidate(
    std::span<const SelectionCandidate> candidates, RouteMode mode,
    double energy_key_epsilon, bool geometry_only,
    bool prefer_maximum_probe_length,
    std::optional<double> navigation_cost_epsilon) {
  return select_epsilon_candidate_impl(candidates, mode, energy_key_epsilon,
      geometry_only, prefer_maximum_probe_length, navigation_cost_epsilon, false);
}

SelectionResult select_epsilon_nonnegative_prefix_candidate(
    std::span<const SelectionCandidate> candidates, double energy_key_epsilon,
    std::optional<double> navigation_cost_epsilon) {
  return select_epsilon_candidate_impl(candidates, RouteMode::Advance,
      energy_key_epsilon, false, false, navigation_cost_epsilon, true);
}

SelectionResult select_epsilon_reroute_candidate(
    std::span<const SelectionCandidate> candidates, double energy_key_epsilon,
    std::optional<double> navigation_cost_epsilon) {
  return select_epsilon_candidate_impl(candidates, RouteMode::Advance,
      energy_key_epsilon, false, false, navigation_cost_epsilon, false, true);
}

SelectionResult select_epsilon_exploratory_candidate(
    std::span<const SelectionCandidate> candidates, double energy_key_epsilon,
    std::optional<double> navigation_cost_epsilon) {
  return select_epsilon_candidate_impl(candidates, RouteMode::Advance,
      energy_key_epsilon, false, false, navigation_cost_epsilon, false, false, true);
}

} // namespace titan_path
