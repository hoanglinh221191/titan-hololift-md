#include "topology/nonbonded_row_kernel.h"

#include "titan_avx2_math.h"
#include "titan_energy_validation.h"

#include <cmath>
#include <limits>

namespace titan_nb {
namespace {

struct PairValues {
  bool ok{true};
  double elec{}, vdw{};
};

// One pair, in the expressions of evaluate_pair.
inline PairValues scalar_pair(const RowInput &in, std::size_t i) {
  PairValues out;
  const double dx = 0.1 * (in.x[i] - in.lx), dy = 0.1 * (in.y[i] - in.ly),
               dz = 0.1 * (in.z[i] - in.lz);
  const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
  const Coefficients &c = in.matrix[in.type_offset[i] + in.lig_type];
  const double charge = in.charge[i] * in.lig_charge;
  if (c.c6 == 0 && c.c12 == 0 && charge == 0)
    return out;
  if (!std::isfinite(r) || r <= titan_energy_validation::kMinimumPairDistanceNm) {
    out.ok = false;
    return out;
  }
  if (in.cutoff > 0 && r > in.cutoff)
    return out;
  const double inv = 1 / r, inv2 = inv * inv, inv6 = inv2 * inv2 * inv2;
  out.vdw = c.c12 * inv6 * inv6 - c.c6 * inv6;
  const double screening = in.kappa > 0 ? std::exp(-in.kappa * r) : 1.0;
  out.elec = in.coulomb * charge * screening * inv;
  return out;
}

} // namespace

bool row_energy(const RowInput &in, double &elec_sum, double &vdw_sum) {
  double elec_total = 0.0, vdw_total = 0.0;
  std::size_t i = 0;
#if TITAN_SIMD_X86 && defined(__AVX2__)
  const __m256d tenth = _mm256_set1_pd(0.1);
  const __m256d lx = _mm256_set1_pd(in.lx), ly = _mm256_set1_pd(in.ly),
                lz = _mm256_set1_pd(in.lz);
  const __m256d lig_charge = _mm256_set1_pd(in.lig_charge);
  const __m256d coulomb = _mm256_set1_pd(in.coulomb);
  const __m256d minus_kappa = _mm256_set1_pd(-in.kappa);
  const __m256d min_distance =
      _mm256_set1_pd(titan_energy_validation::kMinimumPairDistanceNm);
  const __m256d infinity = _mm256_set1_pd(std::numeric_limits<double>::infinity());
  const __m256d cutoff = _mm256_set1_pd(in.cutoff);
  const __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1.0);
  const bool use_cutoff = in.cutoff > 0, use_screening = in.kappa > 0;
  const double *matrix = reinterpret_cast<const double *>(in.matrix);
  const __m128i lig_type = _mm_set1_epi32(static_cast<int>(in.lig_type));
  __m256d elec_acc = zero, vdw_acc = zero;
  for (; i + 4 <= in.count; i += 4) {
    const __m256d dx = _mm256_mul_pd(tenth, _mm256_sub_pd(_mm256_loadu_pd(in.x + i), lx));
    const __m256d dy = _mm256_mul_pd(tenth, _mm256_sub_pd(_mm256_loadu_pd(in.y + i), ly));
    const __m256d dz = _mm256_mul_pd(tenth, _mm256_sub_pd(_mm256_loadu_pd(in.z + i), lz));
    const __m256d r = _mm256_sqrt_pd(_mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)), _mm256_mul_pd(dz, dz)));
    // Coefficients are {c6, c12}: element 2*index and 2*index + 1.
    const __m128i index = _mm_add_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(in.type_offset + i)), lig_type);
    const __m128i twice = _mm_add_epi32(index, index);
    const __m256d c6 = _mm256_i32gather_pd(matrix, twice, 8);
    const __m256d c12 = _mm256_i32gather_pd(matrix, _mm_add_epi32(twice, _mm_set1_epi32(1)), 8);
    const __m256d charge = _mm256_mul_pd(_mm256_loadu_pd(in.charge + i), lig_charge);
    const __m256d silent = _mm256_and_pd(
        _mm256_and_pd(_mm256_cmp_pd(c6, zero, _CMP_EQ_OQ), _mm256_cmp_pd(c12, zero, _CMP_EQ_OQ)),
        _mm256_cmp_pd(charge, zero, _CMP_EQ_OQ));
    const __m256d valid = _mm256_and_pd(_mm256_cmp_pd(r, min_distance, _CMP_GT_OQ),
                                        _mm256_cmp_pd(r, infinity, _CMP_LT_OQ));
    if (_mm256_movemask_pd(_mm256_andnot_pd(valid, _mm256_andnot_pd(silent, _mm256_castsi256_pd(
                                                       _mm256_set1_epi64x(-1))))) != 0)
      return false;
    __m256d keep = _mm256_andnot_pd(silent, valid);
    if (use_cutoff)
      keep = _mm256_and_pd(keep, _mm256_cmp_pd(r, cutoff, _CMP_LE_OQ));
    const __m256d inv = _mm256_div_pd(one, r);
    const __m256d inv2 = _mm256_mul_pd(inv, inv);
    const __m256d inv6 = _mm256_mul_pd(_mm256_mul_pd(inv2, inv2), inv2);
    const __m256d vdw = _mm256_sub_pd(_mm256_mul_pd(_mm256_mul_pd(c12, inv6), inv6),
                                      _mm256_mul_pd(c6, inv6));
    const __m256d screening =
        use_screening ? titan_avx2::exp_pd(_mm256_mul_pd(minus_kappa, r)) : one;
    const __m256d elec = _mm256_mul_pd(
        _mm256_mul_pd(_mm256_mul_pd(coulomb, charge), screening), inv);
    const __m256d vdw_kept = _mm256_and_pd(keep, vdw);
    const __m256d elec_kept = _mm256_and_pd(keep, elec);
    vdw_acc = _mm256_add_pd(vdw_acc, vdw_kept);
    elec_acc = _mm256_add_pd(elec_acc, elec_kept);
    if (in.elec_out != nullptr) {
      _mm256_storeu_pd(in.elec_out + i, elec_kept);
      _mm256_storeu_pd(in.vdw_out + i, vdw_kept);
    }
  }
  elec_total = titan_avx2::hsum_pd(elec_acc);
  vdw_total = titan_avx2::hsum_pd(vdw_acc);
#endif
  for (; i < in.count; ++i) {
    const PairValues pair = scalar_pair(in, i);
    if (!pair.ok)
      return false;
    elec_total += pair.elec;
    vdw_total += pair.vdw;
    if (in.elec_out != nullptr) {
      in.elec_out[i] = pair.elec;
      in.vdw_out[i] = pair.vdw;
    }
  }
  elec_sum += elec_total;
  vdw_sum += vdw_total;
  return true;
}

} // namespace titan_nb
