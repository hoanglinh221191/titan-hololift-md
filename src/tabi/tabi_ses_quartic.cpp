#include "tabi_ses_quartic.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>

namespace TitanPBE::Tabi::analytic_ses {
namespace {

using Real = long double;
constexpr Real kArithmeticEpsilon = std::numeric_limits<Real>::epsilon();

[[nodiscard]] constexpr std::uint32_t
diagnostic_bit(QuarticDiagnosticFailure failure) noexcept {
  return static_cast<std::uint32_t>(failure);
}

void note_failure(QuarticDiagnostics *diagnostics,
                  QuarticDiagnosticFailure failure) noexcept {
  if (diagnostics == nullptr)
    return;
  diagnostics->failure_mask |= diagnostic_bit(failure);
  if (diagnostics->first_failure == QuarticDiagnosticFailure::none)
    diagnostics->first_failure = failure;
}

enum class ExactMultiplicity {
  square_free,
  repeated_root,
  unavailable,
};

struct DyadicInteger {
  std::uint64_t significand{0U};
  int exponent{0};
  bool negative{false};
};

[[nodiscard]] DyadicInteger exact_dyadic(double value) noexcept {
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
  const bool negative = (bits >> 63U) != 0U;
  const std::uint64_t exponent_bits = (bits >> 52U) & 0x7ffU;
  const std::uint64_t fraction = bits & ((std::uint64_t{1} << 52U) - 1U);
  if (exponent_bits == 0U)
    return {fraction, -1074, negative && fraction != 0U};
  return {(std::uint64_t{1} << 52U) | fraction,
          static_cast<int>(exponent_bits) - 1023 - 52, negative};
}

// Fixed capacity is sufficient for the exact quartic discriminant over the
// complete finite-double exponent range: each common-scale coefficient needs
// at most 2,098 bits and every discriminant monomial has total degree six.
struct ExactInteger {
  static constexpr std::size_t kLimbCount = 512U;
  std::array<std::uint32_t, kLimbCount> limbs{};
  std::size_t used{0U};
  bool negative{false};
  bool overflow{false};

  [[nodiscard]] bool is_zero() const noexcept { return used == 0U; }
};

void normalize(ExactInteger &value) noexcept {
  while (value.used != 0U && value.limbs[value.used - 1U] == 0U)
    --value.used;
  if (value.used == 0U)
    value.negative = false;
}

[[nodiscard]] ExactInteger exact_unsigned(std::uint64_t magnitude,
                                          bool negative = false) noexcept {
  ExactInteger result;
  if (magnitude == 0U)
    return result;
  result.limbs[0] = static_cast<std::uint32_t>(magnitude);
  result.limbs[1] = static_cast<std::uint32_t>(magnitude >> 32U);
  result.used = result.limbs[1] == 0U ? 1U : 2U;
  result.negative = negative;
  return result;
}

[[nodiscard]] ExactInteger exact_shifted(const DyadicInteger &value,
                                         int shift) noexcept {
  ExactInteger result;
  if (value.significand == 0U)
    return result;
  if (shift < 0) {
    result.overflow = true;
    return result;
  }
  const std::size_t word = static_cast<std::size_t>(shift / 32);
  const unsigned bits = static_cast<unsigned>(shift % 32);
  if (word + 2U >= ExactInteger::kLimbCount) {
    result.overflow = true;
    return result;
  }
  const std::uint32_t low = static_cast<std::uint32_t>(value.significand);
  const std::uint32_t high =
      static_cast<std::uint32_t>(value.significand >> 32U);
  if (bits == 0U) {
    result.limbs[word] = low;
    result.limbs[word + 1U] = high;
  } else {
    result.limbs[word] = low << bits;
    result.limbs[word + 1U] = (low >> (32U - bits)) | (high << bits);
    result.limbs[word + 2U] = high >> (32U - bits);
  }
  result.used = word + 3U;
  result.negative = value.negative;
  normalize(result);
  return result;
}

[[nodiscard]] int compare_magnitude(const ExactInteger &a,
                                    const ExactInteger &b) noexcept {
  if (a.used != b.used)
    return a.used < b.used ? -1 : 1;
  for (std::size_t index = a.used; index != 0U; --index) {
    if (a.limbs[index - 1U] != b.limbs[index - 1U])
      return a.limbs[index - 1U] < b.limbs[index - 1U] ? -1 : 1;
  }
  return 0;
}

[[nodiscard]] ExactInteger add_magnitude(const ExactInteger &a,
                                         const ExactInteger &b) noexcept {
  ExactInteger result;
  result.overflow = a.overflow || b.overflow;
  const std::size_t count = std::max(a.used, b.used);
  std::uint64_t carry = 0U;
  for (std::size_t index = 0U; index < count; ++index) {
    const std::uint64_t sum =
        static_cast<std::uint64_t>(a.limbs[index]) + b.limbs[index] + carry;
    result.limbs[index] = static_cast<std::uint32_t>(sum);
    carry = sum >> 32U;
  }
  result.used = count;
  if (carry != 0U) {
    if (count == ExactInteger::kLimbCount)
      result.overflow = true;
    else {
      result.limbs[count] = static_cast<std::uint32_t>(carry);
      result.used = count + 1U;
    }
  }
  return result;
}

[[nodiscard]] ExactInteger
subtract_magnitude(const ExactInteger &larger,
                   const ExactInteger &smaller) noexcept {
  ExactInteger result;
  result.overflow = larger.overflow || smaller.overflow;
  std::uint64_t borrow = 0U;
  for (std::size_t index = 0U; index < larger.used; ++index) {
    const std::uint64_t minuend = larger.limbs[index];
    const std::uint64_t subtrahend =
        static_cast<std::uint64_t>(smaller.limbs[index]) + borrow;
    if (minuend < subtrahend) {
      result.limbs[index] = static_cast<std::uint32_t>(
          (std::uint64_t{1} << 32U) + minuend - subtrahend);
      borrow = 1U;
    } else {
      result.limbs[index] = static_cast<std::uint32_t>(minuend - subtrahend);
      borrow = 0U;
    }
  }
  result.used = larger.used;
  normalize(result);
  return result;
}

[[nodiscard]] ExactInteger add_exact(const ExactInteger &a,
                                     const ExactInteger &b) noexcept {
  if (a.negative == b.negative) {
    ExactInteger result = add_magnitude(a, b);
    result.negative = a.negative && !result.is_zero();
    return result;
  }
  const int comparison = compare_magnitude(a, b);
  if (comparison == 0) {
    ExactInteger result;
    result.overflow = a.overflow || b.overflow;
    return result;
  }
  const ExactInteger &larger = comparison > 0 ? a : b;
  const ExactInteger &smaller = comparison > 0 ? b : a;
  ExactInteger result = subtract_magnitude(larger, smaller);
  result.negative = larger.negative && !result.is_zero();
  return result;
}

[[nodiscard]] ExactInteger negate_exact(ExactInteger value) noexcept {
  if (!value.is_zero())
    value.negative = !value.negative;
  return value;
}

[[nodiscard]] ExactInteger subtract_exact(const ExactInteger &a,
                                          const ExactInteger &b) noexcept {
  return add_exact(a, negate_exact(b));
}

[[nodiscard]] ExactInteger multiply_exact(const ExactInteger &a,
                                          const ExactInteger &b) noexcept {
  ExactInteger result;
  result.overflow = a.overflow || b.overflow;
  if (a.is_zero() || b.is_zero())
    return result;
  if (a.used + b.used > ExactInteger::kLimbCount + 1U) {
    result.overflow = true;
    return result;
  }
  for (std::size_t i = 0U; i < a.used; ++i) {
    std::uint64_t carry = 0U;
    for (std::size_t j = 0U; j < b.used; ++j) {
      const std::size_t index = i + j;
      if (index >= ExactInteger::kLimbCount) {
        result.overflow = true;
        continue;
      }
      const std::uint64_t product =
          static_cast<std::uint64_t>(a.limbs[i]) * b.limbs[j] +
          result.limbs[index] + carry;
      result.limbs[index] = static_cast<std::uint32_t>(product);
      carry = product >> 32U;
    }
    const std::size_t carry_index = i + b.used;
    if (carry != 0U) {
      if (carry_index >= ExactInteger::kLimbCount)
        result.overflow = true;
      else
        result.limbs[carry_index] = static_cast<std::uint32_t>(carry);
    }
  }
  result.used = std::min(ExactInteger::kLimbCount, a.used + b.used);
  result.negative = a.negative != b.negative;
  normalize(result);
  return result;
}

[[nodiscard]] ExactInteger multiply_small_exact(const ExactInteger &value,
                                                unsigned factor) noexcept {
  return multiply_exact(value, exact_unsigned(factor));
}

struct ExactQuarticCoefficients {
  // Descending order: a*x^4 + b*x^3 + c*x^2 + d*x + e.
  std::array<ExactInteger, 5> values{};
  bool available{false};
};

[[nodiscard]] ExactQuarticCoefficients
exact_quartic_coefficients(const std::array<double, 5> &coefficients) noexcept {
  ExactQuarticCoefficients result;
  std::array<DyadicInteger, 5> dyadic{};
  int minimum_exponent = std::numeric_limits<int>::max();
  for (std::size_t index = 0U; index < coefficients.size(); ++index) {
    dyadic[index] = exact_dyadic(coefficients[index]);
    if (dyadic[index].significand != 0U)
      minimum_exponent = std::min(minimum_exponent, dyadic[index].exponent);
  }
  if (minimum_exponent == std::numeric_limits<int>::max())
    return result;
  for (std::size_t high = 0U; high < result.values.size(); ++high) {
    const std::size_t source = 4U - high;
    result.values[high] = exact_shifted(
        dyadic[source], dyadic[source].exponent - minimum_exponent);
    if (result.values[high].overflow)
      return result;
  }
  result.available = true;
  return result;
}

[[nodiscard]] ExactInteger
discriminant_term(const std::array<ExactInteger, 5> &c,
                  const std::array<unsigned, 5> &powers, int factor) noexcept {
  const std::int64_t signed_factor = factor;
  ExactInteger result =
      exact_unsigned(static_cast<std::uint64_t>(
                         signed_factor < 0 ? -signed_factor : signed_factor),
                     signed_factor < 0);
  for (std::size_t coefficient = 0U; coefficient < c.size(); ++coefficient) {
    for (unsigned power = 0U; power < powers[coefficient]; ++power)
      result = multiply_exact(result, c[coefficient]);
  }
  return result;
}

// A quartic has a repeated complex or real root exactly when its discriminant
// is zero. Finite doubles are converted to proportional common-scale integers
// without rounding, so this predicate certifies the actual input polynomial.
[[nodiscard]] ExactMultiplicity exact_quartic_multiplicity(
    const ExactQuarticCoefficients &coefficients) noexcept {
  if (!coefficients.available)
    return ExactMultiplicity::unavailable;

  // a,b,c,d,e correspond to ax^4 + bx^3 + cx^2 + dx + e.
  const std::array<ExactInteger, 5> &c = coefficients.values;
  const std::array<std::array<unsigned, 5>, 16> powers = {{
      {{3, 0, 0, 0, 3}},
      {{2, 1, 0, 1, 2}},
      {{2, 0, 2, 0, 2}},
      {{2, 0, 1, 2, 1}},
      {{2, 0, 0, 4, 0}},
      {{1, 2, 1, 0, 2}},
      {{1, 2, 0, 2, 1}},
      {{1, 1, 2, 1, 1}},
      {{1, 1, 1, 3, 0}},
      {{1, 0, 4, 0, 1}},
      {{1, 0, 3, 2, 0}},
      {{0, 4, 0, 0, 2}},
      {{0, 3, 1, 1, 1}},
      {{0, 3, 0, 3, 0}},
      {{0, 2, 3, 0, 1}},
      {{0, 2, 2, 2, 0}},
  }};
  constexpr std::array<int, 16> factors = {
      256, -192, -128, 144, -27, 144, -6, -80, 18, 16, -4, -27, 18, -4, -4, 1};
  ExactInteger discriminant;
  for (std::size_t term_index = 0U; term_index < powers.size(); ++term_index) {
    const ExactInteger term =
        discriminant_term(c, powers[term_index], factors[term_index]);
    discriminant = add_exact(discriminant, term);
    if (discriminant.overflow)
      return ExactMultiplicity::unavailable;
  }
  return discriminant.is_zero() ? ExactMultiplicity::repeated_root
                                : ExactMultiplicity::square_free;
}

[[nodiscard]] int exact_sign(const ExactInteger &value) noexcept {
  if (value.is_zero())
    return 0;
  return value.negative ? -1 : 1;
}

[[nodiscard]] int exact_bit_length(const ExactInteger &value) noexcept {
  if (value.is_zero())
    return 0;
  return static_cast<int>(32U * (value.used - 1U) +
                          std::bit_width(value.limbs[value.used - 1U]));
}

[[nodiscard]] Real exact_to_scaled_real(const ExactInteger &value,
                                        int scale_bits) noexcept {
  Real result = 0.0L;
  for (std::size_t limb = 0U; limb < value.used; ++limb) {
    const int exponent = static_cast<int>(32U * limb) - scale_bits;
    result += std::ldexp(static_cast<Real>(value.limbs[limb]), exponent);
  }
  return value.negative ? -result : result;
}

struct ExactRepeatedRoots {
  std::array<Real, 2> values{};
  std::size_t count{0U};
  bool available{false};
};

// Localize the real roots of gcd(P,P') with an exact degree-four
// subresultant chain.  The discriminant decides only whether a common root
// exists; Q and R below identify its factor, avoiding the former error of
// applying a global multiplicity result to every shallow critical point.
[[nodiscard]] ExactRepeatedRoots
exact_repeated_real_roots(const std::array<double, 5> &coefficients) noexcept {
  ExactRepeatedRoots result;
  const ExactQuarticCoefficients exact =
      exact_quartic_coefficients(coefficients);
  const ExactMultiplicity multiplicity = exact_quartic_multiplicity(exact);
  if (multiplicity == ExactMultiplicity::unavailable)
    return result;
  result.available = true;
  if (multiplicity == ExactMultiplicity::square_free)
    return result;

  const ExactInteger &a = exact.values[0];
  const ExactInteger &b = exact.values[1];
  const ExactInteger &c = exact.values[2];
  const ExactInteger &d = exact.values[3];
  const ExactInteger &e = exact.values[4];

  // Q = 16*a^2*P - (4*a*x+b)*P' = A*x^2+B*x+C.
  const ExactInteger A =
      subtract_exact(multiply_small_exact(multiply_exact(a, c), 8U),
                     multiply_small_exact(multiply_exact(b, b), 3U));
  const ExactInteger B =
      subtract_exact(multiply_small_exact(multiply_exact(a, d), 12U),
                     multiply_small_exact(multiply_exact(b, c), 2U));
  const ExactInteger C = subtract_exact(
      multiply_small_exact(multiply_exact(a, e), 16U), multiply_exact(b, d));
  if (A.overflow || B.overflow || C.overflow) {
    result.available = false;
    return result;
  }

  std::array<ExactInteger, 3> factor{};
  std::size_t factor_degree = 0U;
  if (!A.is_zero()) {
    // A^2*P' mod Q = U*x+V.  A nonzero remainder yields a linear
    // gcd; a zero remainder means Q itself is the gcd.
    const ExactInteger L =
        subtract_exact(multiply_small_exact(multiply_exact(A, b), 3U),
                       multiply_small_exact(multiply_exact(a, B), 4U));
    const ExactInteger M =
        subtract_exact(multiply_small_exact(multiply_exact(A, c), 2U),
                       multiply_small_exact(multiply_exact(a, C), 4U));
    const ExactInteger U =
        subtract_exact(multiply_exact(A, M), multiply_exact(L, B));
    const ExactInteger V = subtract_exact(
        multiply_exact(multiply_exact(A, A), d), multiply_exact(L, C));
    if (L.overflow || M.overflow || U.overflow || V.overflow) {
      result.available = false;
      return result;
    }
    if (!U.is_zero()) {
      factor[0] = V;
      factor[1] = U;
      factor_degree = 1U;
    } else if (V.is_zero()) {
      factor[0] = C;
      factor[1] = B;
      factor[2] = A;
      factor_degree = 2U;
    } else {
      // This is incompatible with the exact zero discriminant.  Treat any
      // capacity or algebraic inconsistency as uncertified.
      result.available = false;
      return result;
    }
  } else if (!B.is_zero()) {
    factor[0] = C;
    factor[1] = B;
    factor_degree = 1U;
  } else if (C.is_zero()) {
    // Q == 0 implies P is a fourth power.  The only distinct repeated root is
    // the exact root of 4*a*x+b.
    factor[0] = b;
    factor[1] = multiply_small_exact(a, 4U);
    factor_degree = 1U;
  } else {
    result.available = false;
    return result;
  }

  int scale_bits = 0;
  for (std::size_t index = 0U; index <= factor_degree; ++index)
    scale_bits = std::max(scale_bits, exact_bit_length(factor[index]));
  if (scale_bits == 0 || factor[factor_degree].is_zero()) {
    result.available = false;
    return result;
  }
  std::array<Real, 3> f{};
  for (std::size_t index = 0U; index <= factor_degree; ++index)
    f[index] = exact_to_scaled_real(factor[index], scale_bits);

  if (factor_degree == 1U) {
    const Real root = -f[0] / f[1];
    if (!std::isfinite(root)) {
      result.available = false;
      return result;
    }
    result.values[result.count++] = root;
    return result;
  }

  const ExactInteger discriminant = subtract_exact(
      multiply_exact(factor[1], factor[1]),
      multiply_small_exact(multiply_exact(factor[2], factor[0]), 4U));
  if (discriminant.overflow) {
    result.available = false;
    return result;
  }
  const int discriminant_sign = exact_sign(discriminant);
  if (discriminant_sign < 0)
    return result;
  if (discriminant_sign == 0) {
    const Real root = -f[1] / (2.0L * f[2]);
    if (!std::isfinite(root)) {
      result.available = false;
      return result;
    }
    result.values[result.count++] = root;
    return result;
  }

  const Real scaled_discriminant =
      exact_to_scaled_real(discriminant, 2 * scale_bits);
  if (!(scaled_discriminant > 0.0L) || !std::isfinite(scaled_discriminant)) {
    result.available = false;
    return result;
  }
  const Real square_root = std::sqrt(scaled_discriminant);
  const Real q = -0.5L * (f[1] + std::copysign(square_root, f[1]));
  if (!(std::abs(q) > std::numeric_limits<Real>::min())) {
    result.available = false;
    return result;
  }
  result.values[result.count++] = q / f[2];
  result.values[result.count++] = f[0] / q;
  if (!std::isfinite(result.values[0]) || !std::isfinite(result.values[1])) {
    result.available = false;
    result.count = 0U;
    return result;
  }
  if (result.values[1] < result.values[0])
    std::swap(result.values[0], result.values[1]);
  return result;
}

template <std::size_t N>
void sort_prefix(std::array<Real, N> &values, std::size_t count) noexcept {
  // The arrays in this solver contain at most five values.  A tiny insertion
  // sort avoids GCC's false-positive -Warray-bounds diagnostic for
  // std::sort(begin, begin + runtime_count) under -O3/-Werror.
  for (std::size_t index = 1U; index < count; ++index) {
    const Real value = values[index];
    std::size_t position = index;
    while (position > 0U && value < values[position - 1U]) {
      values[position] = values[position - 1U];
      --position;
    }
    values[position] = value;
  }
}

[[nodiscard]] Real evaluate(const std::array<Real, 5> &c, Real x) noexcept {
  return ((((c[4] * x + c[3]) * x + c[2]) * x + c[1]) * x + c[0]);
}

[[nodiscard]] Real derivative(const std::array<Real, 5> &c, Real x) noexcept {
  return (((4.0L * c[4] * x + 3.0L * c[3]) * x + 2.0L * c[2]) * x + c[1]);
}

[[nodiscard]] Real residual_scale(const std::array<Real, 5> &c,
                                  Real x) noexcept {
  const Real ax = std::abs(x);
  Real scale = std::abs(c[4]);
  scale = scale * ax + std::abs(c[3]);
  scale = scale * ax + std::abs(c[2]);
  scale = scale * ax + std::abs(c[1]);
  scale = scale * ax + std::abs(c[0]);
  return std::max(scale, std::numeric_limits<Real>::min());
}

[[nodiscard]] Real normalized_residual(const std::array<Real, 5> &c,
                                       Real x) noexcept {
  return std::abs(evaluate(c, x)) / residual_scale(c, x);
}

struct CubicRoots {
  std::array<Real, 3> values{};
  std::size_t count{0U};
  bool certified{false};
};

[[nodiscard]] Real cubic_residual_scale(Real a3, Real a2, Real a1, Real a0,
                                        Real x) noexcept {
  const Real ax = std::abs(x);
  Real scale = std::abs(a3);
  scale = scale * ax + std::abs(a2);
  scale = scale * ax + std::abs(a1);
  scale = scale * ax + std::abs(a0);
  return std::max(scale, std::numeric_limits<Real>::min());
}

[[nodiscard]] CubicRoots
solve_cubic_real(Real a3, Real a2, Real a1, Real a0,
                 QuarticDiagnostics *diagnostics) noexcept {
  CubicRoots result;
  if (!(std::abs(a3) > std::numeric_limits<Real>::min())) {
    note_failure(diagnostics,
                 QuarticDiagnosticFailure::invalid_or_degenerate_coefficients);
    return result;
  }
  const Real a = a2 / a3;
  const Real b = a1 / a3;
  const Real c = a0 / a3;
  const Real p = b - a * a / 3.0L;
  const Real q = 2.0L * a * a * a / 27.0L - a * b / 3.0L + c;
  const Real half_q = 0.5L * q;
  const Real third_p = p / 3.0L;
  const Real discriminant = half_q * half_q + third_p * third_p * third_p;
  const Real depressed_scale =
      std::max({1.0L, std::abs(a * a), std::abs(b), std::abs(c)});
  const Real p_tolerance = 2048.0L * kArithmeticEpsilon * depressed_scale;
  const Real q_tolerance = 4096.0L * kArithmeticEpsilon * depressed_scale *
                           std::max(1.0L, std::sqrt(depressed_scale));
  const Real discriminant_scale = std::max(
      {1.0L, std::abs(half_q * half_q), std::abs(third_p * third_p * third_p)});
  const Real discriminant_tolerance =
      8192.0L * kArithmeticEpsilon * discriminant_scale;
  const Real shift = a / 3.0L;
  if (diagnostics != nullptr) {
    diagnostics->cubic_a = a;
    diagnostics->cubic_b = b;
    diagnostics->cubic_c = c;
    diagnostics->cubic_p = p;
    diagnostics->cubic_q = q;
    diagnostics->cubic_discriminant = discriminant;
    diagnostics->cubic_discriminant_tolerance = discriminant_tolerance;
  }

  bool discriminant_certified = true;
  if (std::abs(p) <= p_tolerance && std::abs(q) <= q_tolerance) {
    // Cardano is ill-conditioned at a triple root: cbrt(roundoff) produces
    // several nearby fake critical points. Represent the multiplicity cluster
    // once and let the quartic monotone partition own it.
    result.values[result.count++] = -shift;
  } else if (discriminant > discriminant_tolerance) {
    const Real square_root = std::sqrt(discriminant);
    const Real u = std::cbrt(-half_q + square_root);
    const Real v = std::abs(u) > std::numeric_limits<Real>::min()
                       ? -third_p / u
                       : std::cbrt(-half_q - square_root);
    result.values[result.count++] = u + v - shift;
  } else if (discriminant < -discriminant_tolerance) {
    const Real radius = 2.0L * std::sqrt(std::max(0.0L, -third_p));
    const Real denominator = std::sqrt(std::max(
        std::numeric_limits<Real>::min(), -third_p * third_p * third_p));
    const Real argument = std::clamp(-half_q / denominator, -1.0L, 1.0L);
    const Real angle = std::acos(argument);
    for (int index = 0; index < 3; ++index) {
      result.values[result.count++] =
          radius * std::cos((angle + 2.0L * std::numbers::pi_v<Real> * index) /
                            3.0L) -
          shift;
    }
  } else {
    // A zero cubic discriminant is itself a degeneracy.  Return the useful
    // approximations, but do not claim a root-count certificate unless the
    // triple-root branch above established the structure without relying on
    // the sign of an unresolved discriminant.
    discriminant_certified = false;
    note_failure(
        diagnostics,
        QuarticDiagnosticFailure::derivative_cubic_discriminant_unresolved);
    const Real u = std::cbrt(-half_q);
    result.values[result.count++] = 2.0L * u - shift;
    result.values[result.count++] = -u - shift;
  }

  // Refine each analytic approximation against the original cubic.  This is
  // not used to decide how many real roots exist; the certified discriminant
  // sign above owns that decision.
  for (std::size_t index = 0U; index < result.count; ++index) {
    Real &root = result.values[index];
    for (int iteration = 0; iteration < 12; ++iteration) {
      const Real value = ((a3 * root + a2) * root + a1) * root + a0;
      const Real slope = (3.0L * a3 * root + 2.0L * a2) * root + a1;
      if (!(std::abs(slope) > std::numeric_limits<Real>::min()))
        break;
      const Real step = value / slope;
      const Real candidate = root - step;
      if (!std::isfinite(candidate))
        break;
      root = candidate;
      if (std::abs(step) <=
          16.0L * kArithmeticEpsilon * std::max(1.0L, std::abs(root)))
        break;
    }
  }
  const Real zero_tolerance =
      2048.0L * kArithmeticEpsilon *
      std::max(
          {1.0L, std::abs(a), std::sqrt(std::abs(b)), std::cbrt(std::abs(c))});
  for (std::size_t index = 0; index < result.count; ++index) {
    if (std::abs(result.values[index]) <= zero_tolerance)
      result.values[index] = 0.0L;
  }
  sort_prefix(result.values, result.count);
  std::size_t unique_count = 0U;
  for (std::size_t index = 0; index < result.count; ++index) {
    const Real tolerance =
        2048.0L * kArithmeticEpsilon *
        std::max({1.0L, std::abs(result.values[index]),
                  unique_count == 0U
                      ? 0.0L
                      : std::abs(result.values[unique_count - 1U])});
    if (unique_count != 0U &&
        std::abs(result.values[index] - result.values[unique_count - 1U]) <=
            tolerance)
      continue;
    result.values[unique_count++] = result.values[index];
  }
  result.count = unique_count;
  result.certified = discriminant_certified;
  if (diagnostics != nullptr) {
    diagnostics->critical_count = result.count;
    for (std::size_t index = 0U; index < result.count; ++index)
      diagnostics->critical_points[index] = result.values[index];
  }
  for (std::size_t index = 0U; index < result.count; ++index) {
    const Real root = result.values[index];
    const Real value = ((a3 * root + a2) * root + a1) * root + a0;
    const Real residual =
        std::abs(value) / cubic_residual_scale(a3, a2, a1, a0, root);
    if (diagnostics != nullptr)
      diagnostics->derivative_normalized_residuals[index] = residual;
    if (!std::isfinite(root) || residual > 32768.0L * kArithmeticEpsilon) {
      result.certified = false;
      note_failure(diagnostics,
                   QuarticDiagnosticFailure::derivative_residual_uncertified);
    }
  }
  return result;
}

struct RootRefinementProvenance {
  Real lower{0.0L};
  Real upper{0.0L};
  Real lower_value{0.0L};
  Real upper_value{0.0L};
  Real final_lower{0.0L};
  Real final_upper{0.0L};
  Real final_lower_value{0.0L};
  Real final_upper_value{0.0L};
  Real final_width{0.0L};
  Real termination_residual{0.0L};
  QuarticRefinementTermination termination{
      QuarticRefinementTermination::not_bracketed};
  bool bracketed{false};
  bool repeated{false};
};

[[nodiscard]] Real
refine_bracket(const std::array<Real, 5> &c, Real lower, Real upper,
               Real lower_value,
               RootRefinementProvenance *provenance) noexcept {
  if (provenance != nullptr) {
    provenance->lower = lower;
    provenance->upper = upper;
    provenance->lower_value = lower_value;
    provenance->upper_value = evaluate(c, upper);
    provenance->bracketed = true;
  }
  const auto capture_final_bracket = [&](Real termination_root) noexcept {
    if (provenance == nullptr)
      return;
    provenance->final_lower = lower;
    provenance->final_upper = upper;
    provenance->final_lower_value = evaluate(c, lower);
    provenance->final_upper_value = evaluate(c, upper);
    provenance->final_width = upper - lower;
    provenance->termination_residual = normalized_residual(c, termination_root);
  };
  Real root = 0.5L * (lower + upper);
  for (int iteration = 0; iteration < 80; ++iteration) {
    const Real width = upper - lower;
    root = 0.5L * (lower + upper);
    const Real value = evaluate(c, root);
    const Real slope = derivative(c, root);
    Real candidate = root;
    if (std::abs(slope) > std::numeric_limits<Real>::min()) {
      const Real newton = root - value / slope;
      if (newton > lower + 0.1L * width && newton < upper - 0.1L * width &&
          std::isfinite(newton))
        candidate = newton;
    }
    const Real candidate_value = evaluate(c, candidate);
    const Real candidate_residual = normalized_residual(c, candidate);
    if (candidate_value == 0.0L || candidate_residual <= 1.0e-18L) {
      if (provenance != nullptr) {
        capture_final_bracket(candidate);
        provenance->termination =
            QuarticRefinementTermination::exact_or_residual;
      }
      return candidate;
    }
    if ((lower_value < 0.0L && candidate_value > 0.0L) ||
        (lower_value > 0.0L && candidate_value < 0.0L)) {
      upper = candidate;
    } else {
      lower = candidate;
      lower_value = candidate_value;
    }
    root = candidate;
    const Real width_scale = std::max(std::abs(lower), std::abs(upper));
    if (width_scale > 0.0L &&
        upper - lower <=
            64.0L * std::numeric_limits<Real>::epsilon() * width_scale) {
      if (provenance != nullptr) {
        capture_final_bracket(root);
        provenance->termination = QuarticRefinementTermination::width_scaled;
      }
      break;
    }
  }
  if (provenance != nullptr &&
      provenance->termination == QuarticRefinementTermination::not_bracketed) {
    capture_final_bracket(root);
    provenance->termination = QuarticRefinementTermination::iteration_limit;
  }
  return root;
}

} // namespace

template <bool CollectDiagnostics>
QuarticRoots solve_real_quartic_impl(const std::array<double, 5> &coefficients,
                                     QuarticDiagnostics *diagnostics) noexcept {
  if constexpr (!CollectDiagnostics)
    diagnostics = nullptr;
  if constexpr (CollectDiagnostics)
    *diagnostics = {};
  if constexpr (CollectDiagnostics)
    diagnostics->input_coefficients = coefficients;
  QuarticRoots result;
  Real coefficient_scale = 0.0L;
  for (const double value : coefficients) {
    if (!std::isfinite(value)) {
      note_failure(
          diagnostics,
          QuarticDiagnosticFailure::invalid_or_degenerate_coefficients);
      return result;
    }
    coefficient_scale = std::max(coefficient_scale, std::abs(Real{value}));
  }
  if (!(coefficient_scale > 0.0L) ||
      !(std::abs(Real{coefficients[4]}) >
        64.0L * static_cast<Real>(std::numeric_limits<double>::epsilon()) *
            coefficient_scale)) {
    note_failure(diagnostics,
                 QuarticDiagnosticFailure::invalid_or_degenerate_coefficients);
    return result;
  }

  std::array<Real, 5> c{};
  for (std::size_t index = 0; index < c.size(); ++index)
    c[index] = Real{coefficients[index]} / coefficient_scale;
  if (diagnostics != nullptr) {
    diagnostics->coefficient_scale = coefficient_scale;
    diagnostics->scaled_coefficients = c;
  }
  const Real leading = std::abs(c[4]);
  Real bound = 1.0L;
  for (std::size_t index = 0; index < 4U; ++index)
    bound = std::max(bound, 1.0L + std::abs(c[index]) / leading);
  if (diagnostics != nullptr)
    diagnostics->root_bound = bound;

  const CubicRoots critical = solve_cubic_real(4.0L * c[4], 3.0L * c[3],
                                               2.0L * c[2], c[1], diagnostics);
  std::array<Real, 5> points{};
  std::size_t point_count = 0U;
  points[point_count++] = -bound;
  for (std::size_t index = 0; index < critical.count; ++index) {
    if (critical.values[index] > -bound && critical.values[index] < bound)
      points[point_count++] = critical.values[index];
  }
  points[point_count++] = bound;
  sort_prefix(points, point_count);
  if (diagnostics != nullptr) {
    diagnostics->critical_count = point_count - 2U;
    for (std::size_t index = 1U; index + 1U < point_count; ++index)
      diagnostics->critical_points[index - 1U] = points[index];
  }

  std::array<Real, 4> roots{};
  std::array<RootRefinementProvenance, CollectDiagnostics ? 4U : 0U>
      root_provenance{};
  std::size_t root_count = 0U;
  const auto append_root = [&](Real root,
                               const RootRefinementProvenance *provenance) {
    if (!std::isfinite(root) || root_count >= roots.size()) {
      note_failure(diagnostics,
                   QuarticDiagnosticFailure::root_capacity_or_nonfinite);
      return;
    }
    const Real tolerance =
        64.0L * static_cast<Real>(std::numeric_limits<double>::epsilon()) *
        std::max(1.0L, std::abs(root));
    for (std::size_t index = 0; index < root_count; ++index) {
      if (std::abs(roots[index] - root) <= tolerance) {
        if (normalized_residual(c, root) <
            normalized_residual(c, roots[index])) {
          roots[index] = root;
          if constexpr (CollectDiagnostics)
            root_provenance[index] = *provenance;
        }
        return;
      }
    }
    roots[root_count] = root;
    if constexpr (CollectDiagnostics)
      root_provenance[root_count] = *provenance;
    ++root_count;
  };

  std::array<int, 5> boundary_sign{};
  bool sign_certificate = critical.certified;
  // A critical value near zero is the only place where a close real pair or a
  // near-complex pair can masquerade as an even-multiplicity root.  Localize
  // actual common P/P' roots with the exact subresultant chain; every other
  // shallow critical point fails closed.
  constexpr Real kSignTolerance = 4096.0L * kArithmeticEpsilon;
  ExactRepeatedRoots repeated_roots;
  bool repeated_roots_loaded = false;
  bool repeated_localization_valid = true;
  std::array<unsigned char, 5> repeated_boundary{};
  std::array<Real, 5> repeated_boundary_value{};
  const auto load_repeated_root_localization = [&]() {
    if (repeated_roots_loaded)
      return;
    repeated_roots_loaded = true;
    if (diagnostics != nullptr)
      diagnostics->repeated_root_localization_attempted = true;
    repeated_roots = exact_repeated_real_roots(coefficients);
    if (diagnostics != nullptr) {
      diagnostics->repeated_root_localization_available =
          repeated_roots.available;
      diagnostics->repeated_root_count = repeated_roots.count;
    }
    if (!repeated_roots.available) {
      repeated_localization_valid = false;
      note_failure(
          diagnostics,
          QuarticDiagnosticFailure::repeated_root_localization_unavailable);
      return;
    }
    for (std::size_t root_index = 0U; root_index < repeated_roots.count;
         ++root_index) {
      const Real root = repeated_roots.values[root_index];
      std::size_t closest = point_count;
      Real closest_distance = std::numeric_limits<Real>::infinity();
      for (std::size_t point = 1U; point + 1U < point_count; ++point) {
        const Real distance = std::abs(points[point] - root);
        if (distance < closest_distance) {
          closest = point;
          closest_distance = distance;
        }
      }
      Real tolerance =
          262144.0L * kArithmeticEpsilon *
          std::max({1.0L, std::abs(root),
                    closest < point_count ? std::abs(points[closest]) : 0.0L});
      Real neighbor_distance = std::numeric_limits<Real>::infinity();
      if (closest > 1U)
        neighbor_distance =
            std::min(neighbor_distance, points[closest] - points[closest - 1U]);
      if (closest + 2U < point_count)
        neighbor_distance =
            std::min(neighbor_distance, points[closest + 1U] - points[closest]);
      if (std::isfinite(neighbor_distance))
        tolerance = std::min(tolerance, 0.25L * neighbor_distance);
      if (closest == point_count || closest_distance > tolerance ||
          repeated_boundary[closest] != 0U) {
        repeated_localization_valid = false;
        note_failure(
            diagnostics,
            QuarticDiagnosticFailure::repeated_root_localization_mismatch);
        return;
      }
      repeated_boundary[closest] = 1U;
      repeated_boundary_value[closest] = root;
    }
  };
  const auto classify_boundary = [&](std::size_t index, bool is_critical) {
    const Real value = evaluate(c, points[index]);
    const Real residual = std::abs(value) / residual_scale(c, points[index]);
    if (diagnostics != nullptr && is_critical) {
      const std::size_t critical_index = index - 1U;
      diagnostics->critical_values[critical_index] = value;
      diagnostics->critical_normalized_residuals[critical_index] = residual;
    }
    const auto record_boundary_failure =
        [&](QuarticDiagnosticFailure failure) noexcept {
          note_failure(diagnostics, failure);
          if (diagnostics != nullptr &&
              diagnostics->first_failure_boundary_index ==
                  std::numeric_limits<std::size_t>::max()) {
            diagnostics->first_failure_boundary_index = index;
            diagnostics->first_failure_point = points[index];
            diagnostics->first_failure_value = value;
            diagnostics->first_failure_normalized_residual = residual;
            diagnostics->first_failure_is_critical = is_critical;
          }
        };
    if (is_critical && repeated_roots_loaded &&
        repeated_boundary[index] != 0U) {
      if (repeated_localization_valid) {
        boundary_sign[index] = 0;
        if constexpr (CollectDiagnostics) {
          RootRefinementProvenance provenance;
          provenance.repeated = true;
          append_root(repeated_boundary_value[index], &provenance);
        } else {
          append_root(repeated_boundary_value[index], nullptr);
        }
      } else {
        sign_certificate = false;
        boundary_sign[index] = 0;
        record_boundary_failure(
            QuarticDiagnosticFailure::critical_boundary_sign_unresolved);
      }
      return;
    }
    if (is_critical && residual <= kSignTolerance) {
      load_repeated_root_localization();
      if (repeated_localization_valid && repeated_boundary[index] != 0U) {
        boundary_sign[index] = 0;
        if constexpr (CollectDiagnostics) {
          RootRefinementProvenance provenance;
          provenance.repeated = true;
          append_root(repeated_boundary_value[index], &provenance);
        } else {
          append_root(repeated_boundary_value[index], nullptr);
        }
      } else {
        // A small critical value alone is not evidence of tangency.  This
        // includes close simple roots when a genuine repeated root exists at
        // another critical point.
        sign_certificate = false;
        boundary_sign[index] = 0;
        record_boundary_failure(
            QuarticDiagnosticFailure::critical_boundary_sign_unresolved);
        record_boundary_failure(QuarticDiagnosticFailure::boundary_near_zero);
      }
      return;
    }
    if (!std::isfinite(value) || residual <= kSignTolerance) {
      sign_certificate = false;
      boundary_sign[index] = 0;
      record_boundary_failure(
          !std::isfinite(value) ? QuarticDiagnosticFailure::boundary_nonfinite
                                : QuarticDiagnosticFailure::boundary_near_zero);
      if (is_critical)
        note_failure(
            diagnostics,
            QuarticDiagnosticFailure::critical_boundary_sign_unresolved);
      return;
    }
    boundary_sign[index] = value < 0.0L ? -1 : 1;
  };
  classify_boundary(0U, false);
  for (std::size_t index = 1U; index + 1U < point_count; ++index) {
    classify_boundary(index, true);
  }
  classify_boundary(point_count - 1U, false);
  for (std::size_t index = 0; index + 1U < point_count; ++index) {
    if (boundary_sign[index] == 0 || boundary_sign[index + 1U] == 0)
      continue;
    const Real lower_value = evaluate(c, points[index]);
    if (boundary_sign[index] != boundary_sign[index + 1U]) {
      if constexpr (CollectDiagnostics) {
        RootRefinementProvenance provenance;
        const Real root = refine_bracket(c, points[index], points[index + 1U],
                                         lower_value, &provenance);
        append_root(root, &provenance);
      } else {
        append_root(refine_bracket(c, points[index], points[index + 1U],
                                   lower_value, nullptr),
                    nullptr);
      }
    }
  }

  for (std::size_t index = 1U; index < root_count; ++index) {
    const Real root = roots[index];
    RootRefinementProvenance provenance;
    if constexpr (CollectDiagnostics)
      provenance = root_provenance[index];
    std::size_t position = index;
    while (position != 0U && root < roots[position - 1U]) {
      roots[position] = roots[position - 1U];
      if constexpr (CollectDiagnostics)
        root_provenance[position] = root_provenance[position - 1U];
      --position;
    }
    roots[position] = root;
    if constexpr (CollectDiagnostics)
      root_provenance[position] = provenance;
  }
  result.count = root_count;
  // A quartic is monotone between consecutive real derivative roots.  A
  // certified derivative discriminant plus certified signs at every boundary
  // therefore proves the distinct real-root count; this is an equivalent
  // degree-four isolation certificate, not merely a residual check.
  result.certified = critical.certified && sign_certificate && root_count <= 4U;
  for (std::size_t index = 0; index < root_count; ++index) {
    const long double pre_cast_residual = normalized_residual(c, roots[index]);
    result.values[index] = static_cast<double>(roots[index]);
    const double residual = static_cast<double>(
        normalized_residual(c, static_cast<Real>(result.values[index])));
    result.max_normalized_residual =
        std::max(result.max_normalized_residual, residual);
    if (!std::isfinite(result.values[index]) || residual > 2.0e-12) {
      result.certified = false;
      note_failure(diagnostics,
                   QuarticDiagnosticFailure::root_residual_failure);
    }
    if constexpr (CollectDiagnostics) {
      const Real derivative_magnitude = std::abs(derivative(c, roots[index]));
      diagnostics->refined_roots[index] = roots[index];
      diagnostics->output_roots[index] = result.values[index];
      diagnostics->pre_cast_normalized_residuals[index] = pre_cast_residual;
      diagnostics->post_cast_normalized_residuals[index] = residual;
      diagnostics->derivative_magnitudes[index] = derivative_magnitude;
      diagnostics->root_condition_proxies[index] =
          residual_scale(c, roots[index]) /
          std::max(derivative_magnitude, std::numeric_limits<Real>::min());
      diagnostics->bracket_lowers[index] = root_provenance[index].lower;
      diagnostics->bracket_uppers[index] = root_provenance[index].upper;
      diagnostics->bracket_lower_values[index] =
          root_provenance[index].lower_value;
      diagnostics->bracket_upper_values[index] =
          root_provenance[index].upper_value;
      diagnostics->final_bracket_lowers[index] =
          root_provenance[index].final_lower;
      diagnostics->final_bracket_uppers[index] =
          root_provenance[index].final_upper;
      diagnostics->final_bracket_lower_values[index] =
          root_provenance[index].final_lower_value;
      diagnostics->final_bracket_upper_values[index] =
          root_provenance[index].final_upper_value;
      diagnostics->final_bracket_widths[index] =
          root_provenance[index].final_width;
      diagnostics->termination_normalized_residuals[index] =
          root_provenance[index].termination_residual;
      diagnostics->refinement_terminations[index] =
          root_provenance[index].termination;
      diagnostics->bracketed_roots[index] =
          static_cast<std::uint8_t>(root_provenance[index].bracketed);
      diagnostics->repeated_roots[index] =
          static_cast<std::uint8_t>(root_provenance[index].repeated);
    }
  }
  if constexpr (CollectDiagnostics) {
    diagnostics->repeated_root_localization_valid =
        repeated_roots_loaded && repeated_localization_valid;
    diagnostics->output_root_count = result.count;
    diagnostics->output_max_normalized_residual =
        result.max_normalized_residual;
  }
  return result;
}

QuarticRoots
solve_real_quartic(const std::array<double, 5> &coefficients) noexcept {
  return solve_real_quartic_impl<false>(coefficients, nullptr);
}

QuarticRoots solve_real_quartic(const std::array<double, 5> &coefficients,
                                QuarticDiagnostics *diagnostics) noexcept {
  if (diagnostics == nullptr)
    return solve_real_quartic_impl<false>(coefficients, nullptr);
  return solve_real_quartic_impl<true>(coefficients, diagnostics);
}

} // namespace TitanPBE::Tabi::analytic_ses
