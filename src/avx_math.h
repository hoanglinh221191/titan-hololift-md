#ifndef AVX_MATH_H
#define AVX_MATH_H

#ifdef __AVX__
#include "simd_compat.h"

// Fast AVX exponential function using a polynomial approximation.
// Based on typical high-performance implementations (e.g., fmath, Agner Fog's
// VCL). Accuracy is sufficient for MD force field screening terms.

inline __m256d f_exp_avx(__m256d x) {
  // Constants for exp approximation
  __m256d t_exp_hi = _mm256_set1_pd(709.782712893384);
  __m256d t_exp_lo = _mm256_set1_pd(-708.3964185322641);

  __m256d log2e = _mm256_set1_pd(1.44269504088896340736);
  __m256d ln2 = _mm256_set1_pd(0.69314718055994530942);

  // Coefficients for polynomial approximation (degree 5 or similar)
  __m256d c0 = _mm256_set1_pd(1.0);
  __m256d c1 = _mm256_set1_pd(1.0);
  __m256d c2 = _mm256_set1_pd(0.5);
  __m256d c3 = _mm256_set1_pd(0.166666666666666666667);
  __m256d c4 = _mm256_set1_pd(0.041666666666666666666);
  __m256d c5 = _mm256_set1_pd(0.008333333333333333333);

  // Clamp input to avoid overflow/underflow
  x = _mm256_min_pd(x, t_exp_hi);
  x = _mm256_max_pd(x, t_exp_lo);

  // k = round(x * log2(e))
  __m256d t = _mm256_mul_pd(x, log2e);
  __m256d r = _mm256_round_pd(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

  // r = x - k * ln(2) produces the reduced argument in a small range
  __m256d f = simd_compat::fnmadd_pd(r, ln2, x);

  // Polynomial approximation: exp(f) ~ 1 + f + f^2/2 + f^3/6 + ...
  // using Horner's method: c0 + f * (c1 + f * (c2 + f * (c3 + ...)))

  // p = c5 * f + c4
  __m256d p = simd_compat::fmadd_pd(c5, f, c4);
  // p = p * f + c3
  p = simd_compat::fmadd_pd(p, f, c3);
  // p = p * f + c2
  p = simd_compat::fmadd_pd(p, f, c2);
  // p = p * f + c1
  p = simd_compat::fmadd_pd(p, f, c1);
  // p = p * f + c0
  p = simd_compat::fmadd_pd(p, f, c0);

  // Reconstruct result: 2^k * p
  // Convert float k to int k
  __m128i k_idx_lo =
      _mm256_cvtpd_epi32(r); // Convert 4 doubles to 4 ints (in low 128)

  // Shift k into exponent position (double precision exponent is bits 52-62)
  // Bias is 1023. We need to add k to the exponent.
  // _mm_slli_epi32 shifts left.
  // BUT we are dealing with doubles, so we need 64-bit integer manipulation.
  // Standard trick: exp(x) = p * 2^k
  // We can compute 2^k directly by constructing the double 2^k.

  // Easier approach for 2^k in double:
  // Cast k to __m256i (64 bit integers).
  // Add bias 1023 to k, shift left 52.
  // However, AVX2 instruction set for 64-bit integers is limited compared to
  // AVX-512. Alternative: scale p by 2^k using ldexp equivalent or bitwise
  // addition.

  // Let's use the bit-hack method which is standard in fast math libs.
  // 2^k: exponent bits = k + 1023.

  // 1. Expand 4x32-bit ints to 4x64-bit ints
  __m256i k_64 = _mm256_cvtepi32_epi64(k_idx_lo);

  // 2. Add bias 1023
  __m256i bias = _mm256_set1_epi64x(1023);
  k_64 = _mm256_add_epi64(k_64, bias);

  // 3. Shift left by 52
  k_64 = _mm256_slli_epi64(k_64, 52);

  // 4. Cast to double and multiply
  __m256d two_k = _mm256_castsi256_pd(k_64);

  return _mm256_mul_pd(p, two_k);
}

#endif // __AVX__

#endif // AVX_MATH_H
