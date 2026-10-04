#ifndef TITAN_PBE_TABI_SES_QUARTIC_H
#define TITAN_PBE_TABI_SES_QUARTIC_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace TitanPBE::Tabi::analytic_ses {

// Coefficients use ascending order: c[0] + c[1] x + ... + c[4] x^4.
//
// Clean-room implementation note: a certified cubic-discriminant sign first
// establishes the complete set of real derivative roots. Those roots partition
// the quartic into monotone intervals whose boundary signs prove the distinct
// real-root count. Multiple roots are admitted only when an exact dyadic
// subresultant chain localizes a real root of gcd(P,P') to the current critical
// point; other near-tangent/near-complex cases fail closed. Simple roots are
// refined by safeguarded Newton/bisection.
// `certified` therefore covers both root count and residual, rather than being
// a residual-only success flag.
struct QuarticRoots {
  std::array<double, 4> values{};
  std::size_t count{0};
  double max_normalized_residual{0.0};
  bool certified{false};
};

enum class QuarticDiagnosticFailure : std::uint32_t {
  none = 0U,
  invalid_or_degenerate_coefficients = 1U << 0U,
  derivative_cubic_discriminant_unresolved = 1U << 1U,
  derivative_residual_uncertified = 1U << 2U,
  critical_boundary_sign_unresolved = 1U << 3U,
  repeated_root_localization_unavailable = 1U << 4U,
  repeated_root_localization_mismatch = 1U << 5U,
  boundary_nonfinite = 1U << 6U,
  boundary_near_zero = 1U << 7U,
  root_capacity_or_nonfinite = 1U << 8U,
  root_residual_failure = 1U << 9U,
};

enum class QuarticRefinementTermination : std::uint8_t {
  not_bracketed = 0,
  exact_or_residual,
  width_floor_one,
  width_scaled,
  iteration_limit,
};

// Optional audit telemetry from the exact production solver path. It contains
// no heap storage and is populated only when explicitly requested; the normal
// solver API below retains its existing fast path and result semantics.
struct QuarticDiagnostics {
  std::uint32_t failure_mask{0U};
  QuarticDiagnosticFailure first_failure{QuarticDiagnosticFailure::none};
  std::array<double, 5> input_coefficients{};
  std::array<long double, 5> scaled_coefficients{};
  long double coefficient_scale{0.0L};
  long double root_bound{0.0L};
  long double cubic_a{0.0L};
  long double cubic_b{0.0L};
  long double cubic_c{0.0L};
  long double cubic_p{0.0L};
  long double cubic_q{0.0L};
  long double cubic_discriminant{0.0L};
  long double cubic_discriminant_tolerance{0.0L};
  std::array<long double, 3> critical_points{};
  std::array<long double, 3> critical_values{};
  std::array<long double, 3> critical_normalized_residuals{};
  std::array<long double, 3> derivative_normalized_residuals{};
  std::size_t critical_count{0U};
  std::size_t first_failure_boundary_index{
      std::numeric_limits<std::size_t>::max()};
  long double first_failure_point{0.0L};
  long double first_failure_value{0.0L};
  long double first_failure_normalized_residual{0.0L};
  bool first_failure_is_critical{false};
  bool repeated_root_localization_attempted{false};
  bool repeated_root_localization_available{false};
  bool repeated_root_localization_valid{false};
  std::size_t repeated_root_count{0U};
  // Parallel fixed telemetry for roots produced by the production solver.
  // The refined value and residual are captured before the existing binary64
  // cast; the output value and residual capture the unchanged public result.
  std::array<long double, 4> refined_roots{};
  std::array<double, 4> output_roots{};
  std::array<long double, 4> pre_cast_normalized_residuals{};
  std::array<double, 4> post_cast_normalized_residuals{};
  std::array<long double, 4> derivative_magnitudes{};
  std::array<long double, 4> root_condition_proxies{};
  std::array<long double, 4> bracket_lowers{};
  std::array<long double, 4> bracket_uppers{};
  std::array<long double, 4> bracket_lower_values{};
  std::array<long double, 4> bracket_upper_values{};
  std::array<long double, 4> final_bracket_lowers{};
  std::array<long double, 4> final_bracket_uppers{};
  std::array<long double, 4> final_bracket_lower_values{};
  std::array<long double, 4> final_bracket_upper_values{};
  std::array<long double, 4> final_bracket_widths{};
  std::array<long double, 4> termination_normalized_residuals{};
  std::array<QuarticRefinementTermination, 4> refinement_terminations{};
  std::array<std::uint8_t, 4> bracketed_roots{};
  std::array<std::uint8_t, 4> repeated_roots{};
  std::size_t output_root_count{0U};
  double output_max_normalized_residual{0.0};
};

[[nodiscard]] QuarticRoots
solve_real_quartic(const std::array<double, 5> &coefficients) noexcept;

[[nodiscard]] QuarticRoots
solve_real_quartic(const std::array<double, 5> &coefficients,
                   QuarticDiagnostics *diagnostics) noexcept;

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_QUARTIC_H
