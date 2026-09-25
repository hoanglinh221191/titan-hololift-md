#ifndef SIMD_COMPAT_H
#define SIMD_COMPAT_H

#include "titan_build_info.h"

#if TITAN_COMPILED_WITH_AVX
#include <immintrin.h>

namespace simd_compat {

inline __m256d fmadd_pd(__m256d a, __m256d b, __m256d c) noexcept {
#if TITAN_COMPILED_WITH_FMA
  return _mm256_fmadd_pd(a, b, c);
#else
  return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#endif
}

inline __m256d fmsub_pd(__m256d a, __m256d b, __m256d c) noexcept {
#if TITAN_COMPILED_WITH_FMA
  return _mm256_fmsub_pd(a, b, c);
#else
  return _mm256_sub_pd(_mm256_mul_pd(a, b), c);
#endif
}

inline __m256d fnmadd_pd(__m256d a, __m256d b, __m256d c) noexcept {
#if TITAN_COMPILED_WITH_FMA
  return _mm256_fnmadd_pd(a, b, c);
#else
  return _mm256_sub_pd(c, _mm256_mul_pd(a, b));
#endif
}

} // namespace simd_compat

#endif

#endif
