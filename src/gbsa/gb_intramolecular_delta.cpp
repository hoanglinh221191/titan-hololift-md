#include "gbsa/gb_intramolecular_delta.h"

#include "titan_avx2_math.h"
#include "titan_gb_model.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace titan_gbsa {
namespace {

struct PairVectors {
  __m256d minus_quarter, minus_kappa, inv_internal, external,
      minus_coulomb, one;
  __m256d alpb_shift_complex, alpb_shift_free, alpb_scale;
};

// polar_pair_energy_kj_mol for four partners, same operation order:
//   f = sqrt(r2 + RiRk exp(-0.25 r2 / RiRk)), s = exp(-kappa f),
//   G = -C qi qk (1/eps_in - s/eps_out) / f            (Still)
//   G = -C qi qk (1/eps_in - s/eps_out) (1/f + ab/A) / (1 + ab)   (ALPB)
TITAN_TARGET_AVX2_FMA
inline __m256d pair_energy(const PairVectors &c, bool use_alpb,
                           __m256d alpb_shift, __m256d qiqk_scaled,
                           __m256d r2, __m256d born_i, __m256d born_k) {
  const __m256d product = _mm256_mul_pd(born_i, born_k);
  const __m256d argument =
      _mm256_div_pd(_mm256_mul_pd(c.minus_quarter, r2), product);
  // far pairs: the exponential is below half an ulp of r2 (header note)
  const __m256d decay = gb_exponential_negligible(argument)
                            ? _mm256_setzero_pd()
                            : titan_avx2::exp_pd(argument);
  const __m256d f_gb =
      _mm256_sqrt_pd(_mm256_add_pd(r2, _mm256_mul_pd(product, decay)));
  const __m256d screened =
      titan_avx2::exp_pd(_mm256_mul_pd(c.minus_kappa, f_gb));
  const __m256d dielectric = _mm256_sub_pd(
      c.inv_internal, _mm256_div_pd(screened, c.external));
  const __m256d numerator = _mm256_mul_pd(qiqk_scaled, dielectric);
  if (!use_alpb)
    return _mm256_div_pd(numerator, f_gb);
  const __m256d inverse_distance =
      _mm256_add_pd(_mm256_div_pd(c.one, f_gb), alpb_shift);
  return _mm256_div_pd(_mm256_mul_pd(numerator, inverse_distance),
                       c.alpb_scale);
}

} // namespace

TITAN_TARGET_AVX2_FMA
double intramolecular_delta_gb(const double *x, const double *y,
                               const double *z, const double *charge,
                               const double *born_complex_nm,
                               const double *born_free_nm, std::size_t n,
                               std::size_t row_begin, std::size_t row_end,
                               const GbPairConstants &k,
                               double *half_delta_per_atom) {
  row_end = std::min(row_end, n);
  if (row_begin >= row_end)
    return 0.0;
  PairVectors c{};
  c.minus_quarter = _mm256_set1_pd(-0.25);
  c.minus_kappa = _mm256_set1_pd(-k.kappa_nm_inverse);
  c.inv_internal = _mm256_set1_pd(1.0 / k.internal_dielectric);
  // the scalar term is screened / eps_out: keep the division
  c.external = _mm256_set1_pd(k.external_dielectric);
  c.minus_coulomb = _mm256_set1_pd(-titan_gb::kCoulombKJMolNmPerE2);
  c.one = _mm256_set1_pd(1.0);
  const double alpha_beta =
      titan_gb::kAlpbAlpha * (k.internal_dielectric / k.external_dielectric);
  c.alpb_shift_complex = _mm256_set1_pd(alpha_beta / k.alpb_complex_size_nm);
  c.alpb_shift_free = _mm256_set1_pd(alpha_beta / k.alpb_free_size_nm);
  c.alpb_scale = _mm256_set1_pd(1.0 + alpha_beta);
  const __m256d scale_nm2 = _mm256_set1_pd(0.01);
  const __m256d half = _mm256_set1_pd(0.5);

  // padding lanes: zero charge, unit radii, far away -> delta exactly 0
  alignas(32) double pad_x[4], pad_y[4], pad_z[4], pad_q[4], pad_rc[4],
      pad_rf[4];
  alignas(32) double pad_half[4];

  double total = 0.0;
  for (std::size_t i = row_begin; i < row_end; ++i) {
    const __m256d xi = _mm256_set1_pd(x[i]);
    const __m256d yi = _mm256_set1_pd(y[i]);
    const __m256d zi = _mm256_set1_pd(z[i]);
    // (-C * q_i) * q_k as in the scalar expression
    const __m256d minus_c_qi =
        _mm256_mul_pd(c.minus_coulomb, _mm256_set1_pd(charge[i]));
    const __m256d rci = _mm256_set1_pd(born_complex_nm[i]);
    const __m256d rfi = _mm256_set1_pd(born_free_nm[i]);
    __m256d row_sum = _mm256_setzero_pd();

    auto block = [&](const double *xk, const double *yk, const double *zk,
                     const double *qk, const double *rck, const double *rfk,
                     double *half_k) {
      const __m256d dx = _mm256_sub_pd(xi, _mm256_loadu_pd(xk));
      const __m256d dy = _mm256_sub_pd(yi, _mm256_loadu_pd(yk));
      const __m256d dz = _mm256_sub_pd(zi, _mm256_loadu_pd(zk));
      const __m256d r2 = _mm256_mul_pd(
          scale_nm2,
          _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx),
                                      _mm256_mul_pd(dy, dy)),
                        _mm256_mul_pd(dz, dz)));
      const __m256d qq = _mm256_mul_pd(minus_c_qi, _mm256_loadu_pd(qk));
      const __m256d delta = _mm256_sub_pd(
          pair_energy(c, k.use_alpb, c.alpb_shift_complex, qq, r2, rci,
                      _mm256_loadu_pd(rck)),
          pair_energy(c, k.use_alpb, c.alpb_shift_free, qq, r2, rfi,
                      _mm256_loadu_pd(rfk)));
      row_sum = _mm256_add_pd(row_sum, delta);
      if (half_k)
        _mm256_storeu_pd(half_k, _mm256_add_pd(_mm256_loadu_pd(half_k),
                                               _mm256_mul_pd(half, delta)));
    };

    std::size_t kk = i + 1;
    for (; kk + 4 <= n; kk += 4)
      block(x + kk, y + kk, z + kk, charge + kk, born_complex_nm + kk,
            born_free_nm + kk,
            half_delta_per_atom ? half_delta_per_atom + kk : nullptr);
    if (kk < n) {
      const std::size_t tail = n - kk;
      for (std::size_t lane = 0; lane < 4; ++lane) {
        const bool real = lane < tail;
        pad_x[lane] = real ? x[kk + lane] : x[i] + 1.0e4;
        pad_y[lane] = real ? y[kk + lane] : y[i];
        pad_z[lane] = real ? z[kk + lane] : z[i];
        pad_q[lane] = real ? charge[kk + lane] : 0.0;
        pad_rc[lane] = real ? born_complex_nm[kk + lane] : 1.0;
        pad_rf[lane] = real ? born_free_nm[kk + lane] : 1.0;
        pad_half[lane] = 0.0;
      }
      block(pad_x, pad_y, pad_z, pad_q, pad_rc, pad_rf,
            half_delta_per_atom ? pad_half : nullptr);
      if (half_delta_per_atom)
        for (std::size_t lane = 0; lane < tail; ++lane)
          half_delta_per_atom[kk + lane] += pad_half[lane];
    }
    const double row = titan_avx2::hsum_pd(row_sum);
    if (half_delta_per_atom)
      half_delta_per_atom[i] += 0.5 * row;
    total += row;
  }
  return total;
}

} // namespace titan_gbsa
