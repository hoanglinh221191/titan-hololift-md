#ifndef TITAN_AVX2_MATH_H
#define TITAN_AVX2_MATH_H

// Double-precision AVX2 exp and log at roundoff accuracy (about 1 ulp), for
// kernels whose results must stay within rounding of the scalar libm path.
// Lanes outside the ordinary range fall back to libm, so special values keep
// the scalar behaviour. Callers dispatch on titan_simd_dispatch::runtime_has_avx2().

#include "titan_simd_dispatch.h"

#if TITAN_SIMD_X86
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>

namespace titan_avx2 {

// Range reduction keeps the polynomial argument in [-ln(2)/2, ln(2)/2]; a
// degree-12 Taylor/Horner evaluation is then at roundoff accuracy on that
// interval. Extreme exponents retain the scalar libm behaviour instead of
// relying on denormal exponent-bit construction.
TITAN_TARGET_AVX2
inline __m256d exp_pd(__m256d x) noexcept {
  const __m256d lower = _mm256_set1_pd(-700.0);
  const __m256d upper = _mm256_set1_pd(700.0);
  const int extreme = _mm256_movemask_pd(_mm256_or_pd(
      _mm256_cmp_pd(x, lower, _CMP_LT_OQ),
      _mm256_cmp_pd(x, upper, _CMP_GT_OQ)));
  if (extreme != 0) {
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, x);
    for (double &lane : lanes)
      lane = std::exp(lane);
    return _mm256_load_pd(lanes);
  }

  constexpr double kInvLn2 = 1.44269504088896340735992468100189214;
  constexpr double kLn2Hi = 0.693147180559945286226763982995180413;
  constexpr double kLn2Lo = 2.319046813846299558417771099e-17;
  const __m256d scaled =
      _mm256_mul_pd(x, _mm256_set1_pd(kInvLn2));
  const __m256d rounded = _mm256_round_pd(
      scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m128i exponents = _mm256_cvtpd_epi32(rounded);
  const __m256d exponent_pd = _mm256_cvtepi32_pd(exponents);
  const __m256d reduced = _mm256_sub_pd(
      _mm256_sub_pd(x,
                    _mm256_mul_pd(exponent_pd, _mm256_set1_pd(kLn2Hi))),
      _mm256_mul_pd(exponent_pd, _mm256_set1_pd(kLn2Lo)));

  __m256d polynomial = _mm256_set1_pd(1.0 / 479001600.0); // 1/12!
  polynomial = _mm256_add_pd(
      _mm256_mul_pd(polynomial, reduced),
      _mm256_set1_pd(1.0 / 39916800.0)); // 1/11!
  polynomial = _mm256_add_pd(
      _mm256_mul_pd(polynomial, reduced),
      _mm256_set1_pd(1.0 / 3628800.0)); // 1/10!
  polynomial = _mm256_add_pd(
      _mm256_mul_pd(polynomial, reduced),
      _mm256_set1_pd(1.0 / 362880.0)); // 1/9!
  polynomial = _mm256_add_pd(
      _mm256_mul_pd(polynomial, reduced),
      _mm256_set1_pd(1.0 / 40320.0)); // 1/8!
  polynomial = _mm256_add_pd(
      _mm256_mul_pd(polynomial, reduced),
      _mm256_set1_pd(1.0 / 5040.0)); // 1/7!
  polynomial = _mm256_add_pd(
      _mm256_mul_pd(polynomial, reduced),
      _mm256_set1_pd(1.0 / 720.0)); // 1/6!
  polynomial = _mm256_add_pd(
      _mm256_mul_pd(polynomial, reduced),
      _mm256_set1_pd(1.0 / 120.0)); // 1/5!
  polynomial = _mm256_add_pd(
      _mm256_mul_pd(polynomial, reduced),
      _mm256_set1_pd(1.0 / 24.0)); // 1/4!
  polynomial = _mm256_add_pd(
      _mm256_mul_pd(polynomial, reduced),
      _mm256_set1_pd(1.0 / 6.0)); // 1/3!
  polynomial = _mm256_add_pd(
      _mm256_mul_pd(polynomial, reduced),
      _mm256_set1_pd(0.5));
  polynomial = _mm256_add_pd(_mm256_mul_pd(polynomial, reduced),
                             _mm256_set1_pd(1.0));
  polynomial = _mm256_add_pd(_mm256_mul_pd(polynomial, reduced),
                             _mm256_set1_pd(1.0));

  // 2^k from its exponent field; |x| <= 700 keeps k + 1023 in the normal
  // range, so this equals the scalar construction bit for bit.
  const __m256i biased = _mm256_add_epi64(_mm256_cvtepi32_epi64(exponents),
                                          _mm256_set1_epi64x(1023));
  const __m256d scale = _mm256_castsi256_pd(_mm256_slli_epi64(biased, 52));
  return _mm256_mul_pd(polynomial, scale);
}

// Natural log. x = 2^e m with m folded into [sqrt(1/2), sqrt(2)); then
// log m = 2 atanh(s), s = (m - 1)/(m + 1), |s| <= 0.1716, whose odd series
// through s^21 is at roundoff accuracy there. m - 1 is exact on that
// interval, so the result keeps its relative accuracy near x = 1. Zero,
// negative, subnormal, infinite and NaN lanes use libm.
TITAN_TARGET_AVX2
inline __m256d log_pd(__m256d x) noexcept {
  const __m256d smallest = _mm256_set1_pd(std::numeric_limits<double>::min());
  const __m256d largest = _mm256_set1_pd(std::numeric_limits<double>::max());
  const int unusual = _mm256_movemask_pd(_mm256_or_pd(
      _mm256_cmp_pd(x, smallest, _CMP_NGE_UQ),
      _mm256_cmp_pd(x, largest, _CMP_GT_OQ)));
  if (unusual != 0) {
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, x);
    for (double &lane : lanes)
      lane = std::log(lane);
    return _mm256_load_pd(lanes);
  }

  const __m256i bits = _mm256_castpd_si256(x);
  const __m256i mantissa_bits = _mm256_or_si256(
      _mm256_and_si256(bits, _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL)),
      _mm256_set1_epi64x(0x3FF0000000000000LL));
  __m256d m = _mm256_castsi256_pd(mantissa_bits);
  // The biased exponent field (< 2^11) becomes a double through 2^52.
  const __m256d two52 = _mm256_set1_pd(4503599627370496.0);
  const __m256i field = _mm256_srli_epi64(bits, 52);
  __m256d e = _mm256_sub_pd(
      _mm256_castsi256_pd(
          _mm256_or_si256(field, _mm256_castpd_si256(two52))),
      two52);
  e = _mm256_sub_pd(e, _mm256_set1_pd(1023.0));
  const __m256d fold =
      _mm256_cmp_pd(m, _mm256_set1_pd(1.41421356237309504880), _CMP_GT_OQ);
  m = _mm256_blendv_pd(m, _mm256_mul_pd(m, _mm256_set1_pd(0.5)), fold);
  e = _mm256_add_pd(e, _mm256_and_pd(fold, _mm256_set1_pd(1.0)));

  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d s =
      _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
  const __m256d s2 = _mm256_mul_pd(s, s);
  __m256d p = _mm256_set1_pd(1.0 / 21.0);
  p = _mm256_add_pd(_mm256_mul_pd(p, s2), _mm256_set1_pd(1.0 / 19.0));
  p = _mm256_add_pd(_mm256_mul_pd(p, s2), _mm256_set1_pd(1.0 / 17.0));
  p = _mm256_add_pd(_mm256_mul_pd(p, s2), _mm256_set1_pd(1.0 / 15.0));
  p = _mm256_add_pd(_mm256_mul_pd(p, s2), _mm256_set1_pd(1.0 / 13.0));
  p = _mm256_add_pd(_mm256_mul_pd(p, s2), _mm256_set1_pd(1.0 / 11.0));
  p = _mm256_add_pd(_mm256_mul_pd(p, s2), _mm256_set1_pd(1.0 / 9.0));
  p = _mm256_add_pd(_mm256_mul_pd(p, s2), _mm256_set1_pd(1.0 / 7.0));
  p = _mm256_add_pd(_mm256_mul_pd(p, s2), _mm256_set1_pd(1.0 / 5.0));
  p = _mm256_add_pd(_mm256_mul_pd(p, s2), _mm256_set1_pd(1.0 / 3.0));
  // log m = 2 s + 2 s^3 (1/3 + ...): keep the leading 2 s exact-ish.
  const __m256d two_s = _mm256_add_pd(s, s);
  const __m256d log_m =
      _mm256_add_pd(two_s, _mm256_mul_pd(_mm256_mul_pd(two_s, s2), p));

  constexpr double kLn2Hi = 0.693147180559945286226763982995180413;
  constexpr double kLn2Lo = 2.319046813846299558417771099e-17;
  return _mm256_add_pd(
      _mm256_mul_pd(e, _mm256_set1_pd(kLn2Hi)),
      _mm256_add_pd(log_m, _mm256_mul_pd(e, _mm256_set1_pd(kLn2Lo))));
}

TITAN_TARGET_AVX2
inline double hsum_pd(__m256d v) noexcept {
  __m128d low = _mm256_castpd256_pd128(v);
  const __m128d high = _mm256_extractf128_pd(v, 1);
  low = _mm_add_pd(low, high);
  const __m128d swapped = _mm_unpackhi_pd(low, low);
  return _mm_cvtsd_f64(_mm_add_sd(low, swapped));
}

} // namespace titan_avx2

#endif // TITAN_SIMD_X86

#endif // TITAN_AVX2_MATH_H
