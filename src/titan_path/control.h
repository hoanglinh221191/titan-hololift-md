#ifndef TITAN_PATH_CONTROL_H
#define TITAN_PATH_CONTROL_H

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>

namespace titan_path {

enum class RouteMode { Advance, Recovery };

enum class CandidateExhaustion {
  None,
  ScanShellExhausted,
  ClearanceExhausted,
};

enum class FinalSelectionFailure {
  None,
  NoValidCandidate,
  ForwardProgressExhausted,
  RecoveryDescentExhausted,
  EnergySelectionExhausted,
};

struct SelectionCandidate {
  double energy_key{std::numeric_limits<double>::infinity()};
  double delta_progress{0.0};
  double pull_distance{0.0};
  double minimum_clearance{0.0};
  double angle_deg{180.0};
  bool valid{false};
  double probe_length{0.0};
  bool fixed_probe{false};
  double path_distance{std::numeric_limits<double>::infinity()};
  double reference_path_distance{std::numeric_limits<double>::infinity()};
  bool navigation_enabled{false};
  double navigation_cost{std::numeric_limits<double>::infinity()};
};

struct SelectionResult {
  bool found{false};
  std::size_t index{0};
  double minimum_energy_key{std::numeric_limits<double>::infinity()};
  std::size_t valid_count{0};
  std::size_t mode_eligible_count{0};
  std::size_t maximum_probe_tier_count{0};
  std::size_t selectable_count{0};
  std::size_t nonforward_count{0};
  std::size_t eligible_count{0};
  std::size_t navigation_eligible_count{0};
  double minimum_navigation_cost{std::numeric_limits<double>::infinity()};
};

struct TurnProjectionResult {
  bool valid{false};
  bool capped{false};
  std::array<double, 3> vector{0.0, 0.0, 0.0};
  double requested_angle_deg{0.0};
  double applied_angle_deg{0.0};
};

[[nodiscard]] TurnProjectionResult project_vector_to_turn_cone(
    const std::array<double, 3> &requested_vector,
    const std::array<double, 3> &old_vector, double max_turn_deg);

[[nodiscard]] std::optional<std::array<double, 3>>
scale_vector_to_length(const std::array<double, 3> &vector,
                       double target_length_angstrom);

[[nodiscard]] RouteMode
next_route_mode(std::optional<RouteMode> previous_mode,
                double path_distance_angstrom,
                double recovery_enter_angstrom,
                double recovery_exit_angstrom);

[[nodiscard]] bool candidate_within_scan_shell(
    double pull_distance_angstrom, double minimum_scan_radius_angstrom,
    double maximum_scan_radius_angstrom);

[[nodiscard]] CandidateExhaustion classify_candidate_exhaustion(
    std::size_t candidate_proposals, std::size_t candidate_evaluations,
    std::size_t scan_shell_exclusions, std::size_t clearance_rejections,
    std::size_t energy_evaluations);

[[nodiscard]] const char *
candidate_exhaustion_name(CandidateExhaustion exhaustion) noexcept;

[[nodiscard]] FinalSelectionFailure
classify_final_selection_failure(const SelectionResult &selection,
                                 RouteMode mode) noexcept;

[[nodiscard]] const char *
final_selection_failure_name(FinalSelectionFailure failure) noexcept;

[[nodiscard]] SelectionResult
select_epsilon_progress_candidate(std::span<const SelectionCandidate> candidates,
                                  RouteMode mode,
                                  double energy_key_epsilon,
                                  bool geometry_only,
                                  bool prefer_maximum_probe_length = false,
                                  std::optional<double>
                                      navigation_cost_epsilon = std::nullopt);

// Offline short-path search ONLY. Same objective as Advance, but permits a
// nonnegative-progress prefix. This does not admit a complete path or MD step.
[[nodiscard]] SelectionResult select_epsilon_nonnegative_prefix_candidate(
    std::span<const SelectionCandidate> candidates, double energy_key_epsilon,
    std::optional<double> navigation_cost_epsilon = std::nullopt);

// Opt-in rerouting ONLY: caller validates episode/clearance/goal constraints.
// Retains Advance energy/navigation/progress ranking; permits <=0.5 A retreat.
[[nodiscard]] SelectionResult select_epsilon_reroute_candidate(
    std::span<const SelectionCandidate> candidates, double energy_key_epsilon,
    std::optional<double> navigation_cost_epsilon = std::nullopt);

// Opt-in exploratory domain: progress is ranked, not a hard eligibility gate.
// Caller still owns finite geometry, swept clearance and physical MD admission.
[[nodiscard]] SelectionResult select_epsilon_exploratory_candidate(
    std::span<const SelectionCandidate> candidates, double energy_key_epsilon,
    std::optional<double> navigation_cost_epsilon = std::nullopt);

} // namespace titan_path

#endif
