#include "titan_topology_energy.h"
#include "gromos_nonbonded.h"
#include "functions.h"
#include "titan_electrostatics.h"
#include "titan_simd_dispatch.h"

#ifdef __AVX__
#include "avx_math.h"
#include <immintrin.h>

using namespace std;

namespace {

inline __m256d distance_squared_nm2(__m256d dx, __m256d dy, __m256d dz,
                                    __m256d scale) noexcept {
  __m256d r2 = _mm256_mul_pd(dx, dx);
  r2 = simd_compat::fmadd_pd(dy, dy, r2);
  r2 = simd_compat::fmadd_pd(dz, dz, r2);
  return _mm256_mul_pd(r2, scale);
}

inline __m256d pow6_ratio(__m256d ratio) noexcept {
  const __m256d ratio2 = _mm256_mul_pd(ratio, ratio);
  const __m256d ratio4 = _mm256_mul_pd(ratio2, ratio2);
  return _mm256_mul_pd(ratio4, ratio2);
}

inline void store_pair_block(double *cache_elec, double *cache_vdw,
                             size_t pos_base, __m256d elec,
                             __m256d vdw) noexcept {
  _mm256_storeu_pd(cache_elec + pos_base, elec);
  _mm256_storeu_pd(cache_vdw + pos_base, vdw);
}

inline void zero_cached_pair_if_present(cache_sigma_epsilon &CachedData,
                                        size_t pos) {
  if (!CachedData.elec_ener.empty())
    CachedData.set_elec_vdw(pos, 0., 0.);
}

} // namespace

int cal_energy_amber_intrin_nofilter(
    const MOL &rec, const MOL &lg, const vector<mol_info> &info_rec,
    const vector<mol_info> &info_lg, cache_sigma_epsilon &CachedData,
    size_t begin, size_t end, cache_energy &Energy, size_t num_snap,
    const arguments &Arguments, const Multi_MOL_parameters &rec_parameters,
    const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, nullptr, Arguments);
    return 0;
  }
  (void)info_rec;
  (void)info_lg;
  (void)num_snap;

  Energy.set_energy(0., 0.);

  const size_t rec_size = rec.size();
  const size_t vector_limit = rec_size - (rec_size % 4);
  const bool write_cache = !CachedData.elec_ener.empty();
  double *cache_elec = write_cache ? CachedData.elec_ener.data() : nullptr;
  double *cache_vdw = write_cache ? CachedData.vdw_ener.data() : nullptr;

  const double *rec_x = rec.pos.x.data();
  const double *rec_y = rec.pos.y.data();
  const double *rec_z = rec.pos.z.data();
  const double *rec_charge = rec_parameters.charge.data();
  const double *rec_sigma = rec_parameters.sigma.data();
  const double *rec_epsilon = rec_parameters.epsilon.data();

  const __m256d avx_half = _mm256_set1_pd(0.5);
  const __m256d avx_one = _mm256_set1_pd(1.0);
  const __m256d v_scale = _mm256_set1_pd(0.01);
  const double epsilon_in = Arguments.epsilon_in;

  __m256d avx_sum_elec = _mm256_setzero_pd();
  __m256d avx_sum_vdw = _mm256_setzero_pd();

  for (size_t j = begin; j < end; ++j) {
    const __m256d v_pos_x = _mm256_set1_pd(lg.pos.x[j]);
    const __m256d v_pos_y = _mm256_set1_pd(lg.pos.y[j]);
    const __m256d v_pos_z = _mm256_set1_pd(lg.pos.z[j]);
    const __m256d avx_temp_charge_lg = _mm256_set1_pd(
        138.935457839 * lg_parameters.charge[j] / epsilon_in);
    const __m256d avx_temp_sigma = _mm256_set1_pd(lg_parameters.sigma[j]);
    const __m256d avx_temp_epsilon_lg =
        _mm256_set1_pd(16. * lg_parameters.epsilon[j]);

    for (size_t i = 0; i < vector_limit; i += 4) {
      alignas(32) double sigma_arr[4]{}, epsilon_arr[4]{}, d_arr[4]{},
          vdw_arr[4]{};
      const __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(rec_x + i), v_pos_x);
      const __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(rec_y + i), v_pos_y);
      const __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(rec_z + i), v_pos_z);
      const __m256d r2 = distance_squared_nm2(dx, dy, dz, v_scale);
      const __m256d d = _mm256_sqrt_pd(r2);
      const __m256d inv_d = _mm256_div_pd(avx_one, d);

      __m256d elec = _mm256_mul_pd(avx_temp_charge_lg, _mm256_loadu_pd(rec_charge + i));
      elec = _mm256_mul_pd(elec, inv_d);

      const __m256d epsilon = _mm256_sqrt_pd(
          _mm256_mul_pd(avx_temp_epsilon_lg, _mm256_loadu_pd(rec_epsilon + i)));
      const __m256d sigma =
          _mm256_mul_pd(_mm256_add_pd(avx_temp_sigma, _mm256_loadu_pd(rec_sigma + i)),
                        avx_half);
      _mm256_storeu_pd(sigma_arr, sigma);
      _mm256_storeu_pd(epsilon_arr, epsilon);
      _mm256_storeu_pd(d_arr, d);
      titan_simd_dispatch::lj_mixed_sigma_epsilon_distance_fourlanes(
          sigma_arr, epsilon_arr, d_arr, vdw_arr);
      const __m256d vdw = _mm256_loadu_pd(vdw_arr);

      avx_sum_elec = _mm256_add_pd(avx_sum_elec, elec);
      avx_sum_vdw = _mm256_add_pd(avx_sum_vdw, vdw);

      if (write_cache)
        store_pair_block(cache_elec, cache_vdw, rec_size * j + i, elec, vdw);
    }

    for (size_t i = vector_limit; i < rec_size; ++i) {
      const double d{0.1 * sqrt(square(rec.pos.x[i] - lg.pos.x[j]) +
                                square(rec.pos.y[i] - lg.pos.y[j]) +
                                square(rec.pos.z[i] - lg.pos.z[j]))};
      const double elec_val =
          (138.935457839 * lg_parameters.charge[j] / epsilon_in) *
          rec_parameters.charge[i] / d;
      const double c_6{
          c6(0.5 * (lg_parameters.sigma[j] + rec_parameters.sigma[i]) / d)};
      const double vdw_val =
          sqrt(16. * lg_parameters.epsilon[j] * rec_parameters.epsilon[i]) *
          (square(c_6) - c_6);
      const size_t pos = rec_size * j + i;

      Energy.elec_ener += elec_val;
      Energy.vdw_ener += vdw_val;
      if (write_cache)
        CachedData.set_elec_vdw(pos, elec_val, vdw_val);
    }
  }

  Energy.elec_ener += hsum_double_avx(avx_sum_elec);
  Energy.vdw_ener += hsum_double_avx(avx_sum_vdw);
  return 0;
}

int cal_energy_amber_screening_intrin_nofilter(
    const MOL &rec, const MOL &lg, const vector<mol_info> &info_rec,
    const vector<mol_info> &info_lg, cache_sigma_epsilon &CachedData,
    size_t begin, size_t end, cache_energy &Energy, size_t num_snap,
    const arguments &Arguments, const Multi_MOL_parameters &rec_parameters,
    const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, nullptr, Arguments);
    return 0;
  }
  (void)info_rec;
  (void)info_lg;
  (void)num_snap;

  Energy.set_energy(0., 0.);

  constexpr double kScreeningMinDistance = 1e-10;
  const size_t rec_size = rec.size();
  const size_t vector_limit = rec_size - (rec_size % 4);
  const bool write_cache = !CachedData.elec_ener.empty();
  double *cache_elec = write_cache ? CachedData.elec_ener.data() : nullptr;
  double *cache_vdw = write_cache ? CachedData.vdw_ener.data() : nullptr;

  const double *rec_x = rec.pos.x.data();
  const double *rec_y = rec.pos.y.data();
  const double *rec_z = rec.pos.z.data();
  const double *rec_charge = rec_parameters.charge.data();
  const double *rec_sigma = rec_parameters.sigma.data();
  const double *rec_epsilon = rec_parameters.epsilon.data();

  const double epsilon_in = Arguments.epsilon_in;
  const double lambda_debye =
      titan_electrostatics::debye_length_nm(Arguments.salt_concentration);

  const __m256d avx_half = _mm256_set1_pd(0.5);
  const __m256d avx_one = _mm256_set1_pd(1.0);
  const __m256d avx_min_distance = _mm256_set1_pd(kScreeningMinDistance);
  const __m256d avx_epsilon_in = _mm256_set1_pd(epsilon_in);
  const __m256d avx_neg_inv_lambda = _mm256_set1_pd(-1.0 / lambda_debye);
  const __m256d v_scale = _mm256_set1_pd(0.01);

  __m256d avx_sum_elec = _mm256_setzero_pd();
  __m256d avx_sum_vdw = _mm256_setzero_pd();

  for (size_t j = begin; j < end; ++j) {
    const __m256d v_pos_x = _mm256_set1_pd(lg.pos.x[j]);
    const __m256d v_pos_y = _mm256_set1_pd(lg.pos.y[j]);
    const __m256d v_pos_z = _mm256_set1_pd(lg.pos.z[j]);
    const __m256d avx_temp_charge_lg =
        _mm256_set1_pd(138.935457839 * lg_parameters.charge[j]);
    const __m256d avx_temp_sigma = _mm256_set1_pd(lg_parameters.sigma[j]);
    const __m256d avx_temp_epsilon_lg =
        _mm256_set1_pd(16. * lg_parameters.epsilon[j]);

    for (size_t i = 0; i < vector_limit; i += 4) {
      alignas(32) double sigma_arr[4]{}, epsilon_arr[4]{}, safe_d_arr[4]{},
          vdw_arr[4]{};
      const __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(rec_x + i), v_pos_x);
      const __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(rec_y + i), v_pos_y);
      const __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(rec_z + i), v_pos_z);
      const __m256d r2 = distance_squared_nm2(dx, dy, dz, v_scale);
      const __m256d d = _mm256_sqrt_pd(r2);
      const __m256d valid_mask = _mm256_cmp_pd(d, avx_min_distance, _CMP_GT_OQ);
      const __m256d safe_d = _mm256_blendv_pd(avx_min_distance, d, valid_mask);
      const __m256d inv_d = _mm256_div_pd(avx_one, safe_d);
      const __m256d exp_val = f_exp_avx(_mm256_mul_pd(safe_d, avx_neg_inv_lambda));

      __m256d elec = _mm256_mul_pd(avx_temp_charge_lg, _mm256_loadu_pd(rec_charge + i));
      elec = _mm256_mul_pd(elec, exp_val);
      elec = _mm256_mul_pd(elec, inv_d);
      elec = _mm256_div_pd(elec, avx_epsilon_in);

      const __m256d epsilon = _mm256_sqrt_pd(
          _mm256_mul_pd(avx_temp_epsilon_lg, _mm256_loadu_pd(rec_epsilon + i)));
      const __m256d sigma =
          _mm256_mul_pd(_mm256_add_pd(avx_temp_sigma, _mm256_loadu_pd(rec_sigma + i)),
                        avx_half);
      _mm256_storeu_pd(sigma_arr, sigma);
      _mm256_storeu_pd(epsilon_arr, epsilon);
      _mm256_storeu_pd(safe_d_arr, safe_d);
      titan_simd_dispatch::lj_mixed_sigma_epsilon_distance_fourlanes(
          sigma_arr, epsilon_arr, safe_d_arr, vdw_arr);
      __m256d vdw = _mm256_loadu_pd(vdw_arr);

      elec = _mm256_and_pd(elec, valid_mask);
      vdw = _mm256_and_pd(vdw, valid_mask);

      avx_sum_elec = _mm256_add_pd(avx_sum_elec, elec);
      avx_sum_vdw = _mm256_add_pd(avx_sum_vdw, vdw);

      if (write_cache)
        store_pair_block(cache_elec, cache_vdw, rec_size * j + i, elec, vdw);
    }

    for (size_t i = vector_limit; i < rec_size; ++i) {
      const double d{0.1 * sqrt(square(rec.pos.x[i] - lg.pos.x[j]) +
                                square(rec.pos.y[i] - lg.pos.y[j]) +
                                square(rec.pos.z[i] - lg.pos.z[j]))};
      const size_t pos = rec_size * j + i;
      if (!std::isfinite(d) || d <= kScreeningMinDistance) {
        zero_cached_pair_if_present(CachedData, pos);
        continue;
      }

      const double c_6{
          c6(0.5 * (lg_parameters.sigma[j] + rec_parameters.sigma[i]) / d)};
      const double term_exp = exp(-d / lambda_debye);
      const double elec_val =
          (138.935457839 * lg_parameters.charge[j] * rec_parameters.charge[i] *
           term_exp) /
          (d * epsilon_in);
      const double vdw_val =
          sqrt(16. * lg_parameters.epsilon[j] * rec_parameters.epsilon[i]) *
          (square(c_6) - c_6);

      Energy.elec_ener += elec_val;
      Energy.vdw_ener += vdw_val;
      if (write_cache)
        CachedData.set_elec_vdw(pos, elec_val, vdw_val);
    }
  }

  Energy.elec_ener += hsum_double_avx(avx_sum_elec);
  Energy.vdw_ener += hsum_double_avx(avx_sum_vdw);
  return 0;
}

int cal_energy_gromos_intrin_nofilter(
    const MOL &rec, const MOL &lg, const vector<mol_info> &info_rec,
    const vector<mol_info> &info_lg, cache_sigma_epsilon &CachedData,
    size_t begin, size_t end, cache_energy &Energy, size_t num_snap,
    const arguments &Arguments, const Multi_MOL_parameters &rec_parameters,
    const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, nullptr, Arguments);
    return 0;
  }
  (void)info_rec;
  (void)info_lg;
  (void)num_snap;

  Energy.set_energy(0., 0.);

  const size_t rec_size = rec.size();
  const size_t vector_limit = rec_size - (rec_size % 4);
  const bool write_cache = !CachedData.elec_ener.empty();
  double *cache_elec = write_cache ? CachedData.elec_ener.data() : nullptr;
  double *cache_vdw = write_cache ? CachedData.vdw_ener.data() : nullptr;

  const double *rec_x = rec.pos.x.data();
  const double *rec_y = rec.pos.y.data();
  const double *rec_z = rec.pos.z.data();
  const double *rec_charge = rec_parameters.charge.data();
  const double *rec_sigma = rec_parameters.sigma.data();
  const double *rec_epsilon = rec_parameters.epsilon.data();

  const __m256d avx_one = _mm256_set1_pd(1.0);
  const __m256d avx_epsilon_in = _mm256_set1_pd(Arguments.epsilon_in);
  const __m256d v_scale = _mm256_set1_pd(0.01);

  __m256d avx_sum_elec = _mm256_setzero_pd();
  __m256d avx_sum_vdw = _mm256_setzero_pd();

  for (size_t j = begin; j < end; ++j) {
    const __m256d v_pos_x = _mm256_set1_pd(lg.pos.x[j]);
    const __m256d v_pos_y = _mm256_set1_pd(lg.pos.y[j]);
    const __m256d v_pos_z = _mm256_set1_pd(lg.pos.z[j]);
    const __m256d avx_temp_charge_lg =
        _mm256_set1_pd(138.935457839 * lg_parameters.charge[j]);
    const __m256d avx_temp_sigma_lg = _mm256_set1_pd(lg_parameters.sigma[j]);
    const __m256d avx_temp_epsilon_lg =
        _mm256_set1_pd(lg_parameters.epsilon[j]);

    for (size_t i = 0; i < vector_limit; i += 4) {
      const __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(rec_x + i), v_pos_x);
      const __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(rec_y + i), v_pos_y);
      const __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(rec_z + i), v_pos_z);
      const __m256d r2 = distance_squared_nm2(dx, dy, dz, v_scale);
      const __m256d d = _mm256_sqrt_pd(r2);
      const __m256d inv_d = _mm256_div_pd(avx_one, d);
      const __m256d inv_d6 = pow6_ratio(inv_d);
      const __m256d inv_d12 = _mm256_mul_pd(inv_d6, inv_d6);

      __m256d elec = _mm256_mul_pd(avx_temp_charge_lg, _mm256_loadu_pd(rec_charge + i));
      elec = _mm256_mul_pd(elec, inv_d);
      elec = _mm256_div_pd(elec, avx_epsilon_in);

      __m256d epsilon = _mm256_sqrt_pd(
          _mm256_mul_pd(avx_temp_epsilon_lg, _mm256_loadu_pd(rec_epsilon + i)));
      __m256d sigma = _mm256_sqrt_pd(
          _mm256_mul_pd(avx_temp_sigma_lg, _mm256_loadu_pd(rec_sigma + i)));
      if (CachedData.has_charmm_pair_overrides()) {
        alignas(32) double c6_values[4],c12_values[4];
        for(std::size_t lane=0;lane<4;++lane)
          gromos::pair_params(CachedData,rec_parameters,lg_parameters,i+lane,j,c6_values[lane],c12_values[lane]);
        sigma=_mm256_load_pd(c6_values);epsilon=_mm256_load_pd(c12_values);
      }
      const __m256d vdw = _mm256_sub_pd(_mm256_mul_pd(epsilon, inv_d12),
                                        _mm256_mul_pd(sigma, inv_d6));

      avx_sum_elec = _mm256_add_pd(avx_sum_elec, elec);
      avx_sum_vdw = _mm256_add_pd(avx_sum_vdw, vdw);

      if (write_cache)
        store_pair_block(cache_elec, cache_vdw, rec_size * j + i, elec, vdw);
    }

    for (size_t i = vector_limit; i < rec_size; ++i) {
      const double d{0.1 * sqrt(square(rec.pos.x[i] - lg.pos.x[j]) +
                                square(rec.pos.y[i] - lg.pos.y[j]) +
                                square(rec.pos.z[i] - lg.pos.z[j]))};
      const double elec_val =
          138.935457839 * lg_parameters.charge[j] * rec_parameters.charge[i] /
          (d * Arguments.epsilon_in);
      const double c_6 = c6(d);
      double pair_c6{},pair_c12{};
      gromos::pair_params(CachedData,rec_parameters,lg_parameters,i,j,pair_c6,pair_c12);
      const double vdw_val=pair_c12/square(c_6)-pair_c6/c_6;
      const size_t pos = rec_size * j + i;

      Energy.elec_ener += elec_val;
      Energy.vdw_ener += vdw_val;
      if (write_cache)
        CachedData.set_elec_vdw(pos, elec_val, vdw_val);
    }
  }

  Energy.elec_ener += hsum_double_avx(avx_sum_elec);
  Energy.vdw_ener += hsum_double_avx(avx_sum_vdw);
  return 0;
}

void cal_energy_opls_intrin_nofilter(
    const MOL &rec, const MOL &lg, const vector<mol_info> &info_rec,
    const vector<mol_info> &info_lg, cache_sigma_epsilon &CachedData,
    size_t begin, size_t end, cache_energy &Energy, size_t num_snap,
    const arguments &Arguments, const Multi_MOL_parameters &rec_parameters,
    const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, nullptr, Arguments);
    return;
  }
  (void)info_rec;
  (void)info_lg;
  (void)num_snap;

  Energy.set_energy(0., 0.);

  const size_t rec_size = rec.size();
  const size_t vector_limit = rec_size - (rec_size % 4);
  const bool write_cache = !CachedData.elec_ener.empty();
  double *cache_elec = write_cache ? CachedData.elec_ener.data() : nullptr;
  double *cache_vdw = write_cache ? CachedData.vdw_ener.data() : nullptr;

  const double *rec_x = rec.pos.x.data();
  const double *rec_y = rec.pos.y.data();
  const double *rec_z = rec.pos.z.data();
  const double *rec_charge = rec_parameters.charge.data();
  const double *rec_sigma = rec_parameters.sigma.data();
  const double *rec_epsilon = rec_parameters.epsilon.data();

  const __m256d avx_one = _mm256_set1_pd(1.0);
  const __m256d v_scale = _mm256_set1_pd(0.01);

  __m256d avx_sum_elec = _mm256_setzero_pd();
  __m256d avx_sum_vdw = _mm256_setzero_pd();

  for (size_t j = begin; j < end; ++j) {
    const __m256d v_pos_x = _mm256_set1_pd(lg.pos.x[j]);
    const __m256d v_pos_y = _mm256_set1_pd(lg.pos.y[j]);
    const __m256d v_pos_z = _mm256_set1_pd(lg.pos.z[j]);
    const __m256d avx_temp_charge_lg = _mm256_set1_pd(
        138.935457839 * lg_parameters.charge[j] / Arguments.epsilon_in);
    const __m256d avx_temp_sigma = _mm256_set1_pd(lg_parameters.sigma[j]);
    const __m256d avx_temp_epsilon_lg =
        _mm256_set1_pd(16. * lg_parameters.epsilon[j]);

    for (size_t i = 0; i < vector_limit; i += 4) {
      const __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(rec_x + i), v_pos_x);
      const __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(rec_y + i), v_pos_y);
      const __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(rec_z + i), v_pos_z);
      const __m256d r2 = distance_squared_nm2(dx, dy, dz, v_scale);
      const __m256d d = _mm256_sqrt_pd(r2);
      const __m256d inv_d = _mm256_div_pd(avx_one, d);

      __m256d elec = _mm256_mul_pd(avx_temp_charge_lg, _mm256_loadu_pd(rec_charge + i));
      elec = _mm256_mul_pd(elec, inv_d);

      const __m256d epsilon = _mm256_sqrt_pd(
          _mm256_mul_pd(avx_temp_epsilon_lg, _mm256_loadu_pd(rec_epsilon + i)));
      const __m256d sigma = _mm256_sqrt_pd(
          _mm256_mul_pd(avx_temp_sigma, _mm256_loadu_pd(rec_sigma + i)));
      const __m256d c6 = pow6_ratio(_mm256_mul_pd(sigma, inv_d));
      const __m256d vdw =
          _mm256_mul_pd(epsilon, _mm256_sub_pd(_mm256_mul_pd(c6, c6), c6));

      avx_sum_elec = _mm256_add_pd(avx_sum_elec, elec);
      avx_sum_vdw = _mm256_add_pd(avx_sum_vdw, vdw);

      if (write_cache)
        store_pair_block(cache_elec, cache_vdw, rec_size * j + i, elec, vdw);
    }

    for (size_t i = vector_limit; i < rec_size; ++i) {
      const double d{0.1 * sqrt(square(rec.pos.x[i] - lg.pos.x[j]) +
                                square(rec.pos.y[i] - lg.pos.y[j]) +
                                square(rec.pos.z[i] - lg.pos.z[j]))};
      const double elec_val =
          138.935457839 * lg_parameters.charge[j] * rec_parameters.charge[i] /
          (d * Arguments.epsilon_in);
      const double c_6{
          c6(sqrt(lg_parameters.sigma[j] * rec_parameters.sigma[i]) / d)};
      const double vdw_val =
          sqrt(16. * lg_parameters.epsilon[j] * rec_parameters.epsilon[i]) *
          (square(c_6) - c_6);
      const size_t pos = rec_size * j + i;

      Energy.elec_ener += elec_val;
      Energy.vdw_ener += vdw_val;
      if (write_cache)
        CachedData.set_elec_vdw(pos, elec_val, vdw_val);
    }
  }

  Energy.elec_ener += hsum_double_avx(avx_sum_elec);
  Energy.vdw_ener += hsum_double_avx(avx_sum_vdw);
}

#endif
