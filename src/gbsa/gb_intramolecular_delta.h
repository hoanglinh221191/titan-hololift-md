// Intramolecular part of the GB binding cycle (ENE-15).
//
// For one molecule, sum over the rows i in [row_begin, row_end) and every
// partner k > i of
//   delta_ik = G_pair(q_i, q_k, r_ik, R_i^complex, R_k^complex; A_complex)
//            - G_pair(q_i, q_k, r_ik, R_i^free,    R_k^free;    A_free),
// G_pair being titan_gb::polar_pair_energy_kj_mol (Still f_GB, Debye-Hueckel
// screening of the solvent term, optional ALPB). Four partners per AVX2
// step, exp through titan_avx2::exp_pd (about 1 ulp of libm), the same
// operation order as the scalar function otherwise, so a sum differs from the
// scalar loop by rounding only. Rows are independent: a caller may split
// [0, n) into row tiles and add the results.
#pragma once

#include "titan_simd_dispatch.h"

#include <cstddef>
#include <immintrin.h>

namespace titan_gbsa {

// Still's f_GB = sqrt(r^2 + RiRj exp(x)), x = -r^2 / (4 RiRj). When x <= -36
// in every lane, RiRj exp(x) <= r^2 exp(-36) / 144 < r^2 * 1.7e-18, below
// half an ulp of r^2 (>= r^2 * 5.5e-17): r^2 + RiRj exp(x) rounds to r^2 for
// any exponential within 30x of the true value, so the exponential can be
// skipped (taken as 0) without changing a bit of f_GB.
TITAN_TARGET_AVX2
inline bool gb_exponential_negligible(__m256d argument) noexcept {
  return _mm256_movemask_pd(_mm256_cmp_pd(
             argument, _mm256_set1_pd(-36.0), _CMP_LE_OQ)) == 0xF;
}

struct GbPairConstants {
  double kappa_nm_inverse{0.0};
  double internal_dielectric{1.0};
  double external_dielectric{78.5};
  bool use_alpb{false};
  double alpb_complex_size_nm{0.0}; // ALPB only
  double alpb_free_size_nm{0.0};    // ALPB only
};

// x, y, z in Angstrom; radii in nm; n atoms. When half_delta_per_atom is not
// null (n entries), 0.5 * delta_ik is added to entries i and k, for the
// per-atom decomposition. Returns the sum of delta_ik (kJ/mol).
double intramolecular_delta_gb(const double *x, const double *y,
                               const double *z, const double *charge,
                               const double *born_complex_nm,
                               const double *born_free_nm, std::size_t n,
                               std::size_t row_begin, std::size_t row_end,
                               const GbPairConstants &constants,
                               double *half_delta_per_atom);

} // namespace titan_gbsa
