#include "titan_topology_energy.h"
#include "gromos_nonbonded.h"
#include "titan_lj_pair_override.h"
#include "cache_sigma_epsilon.h"
#include "charmm_nonbonded.h"
#include "force.h"
#include "functions.h"
#include "martini_nonbonded.h"
#include "miscellaneous.h"
#include "titan_energy_validation.h"
#include "titan_avx2_math.h"
#include "energy/cutoff_blocks.h"
#include "titan_electrostatics.h"
#include "titan_simd_dispatch.h"
#include "uti_func.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __AVX__
#include "avx_math.h"
#include <immintrin.h>

namespace {
inline __m256d load_lane_mask_pd(const double *lane_mask) {
  // _mm256_and_pd expects a bitmask (all-ones/zero), not the IEEE bits of -1.0.
  const __m256d lane_values = _mm256_loadu_pd(lane_mask);
  const __m256d zero = _mm256_setzero_pd();
  return _mm256_cmp_pd(lane_values, zero, _CMP_NEQ_OQ);
}

inline void zero_cached_pair_if_present(cache_sigma_epsilon &CachedData,
                                        size_t pos) {
  if (!CachedData.elec_ener.empty())
    CachedData.set_elec_vdw(pos, 0., 0.);
}

inline __m256d charmm_lj_energy_avx(__m256d sigma_vec, __m256d epsilon_vec,
                                    __m256d distance_vec) {
  const __m256d avx_zero = _mm256_setzero_pd();
  const __m256d avx_four = _mm256_set1_pd(4.0);
  const __m256d sigma_valid = _mm256_cmp_pd(sigma_vec, avx_zero, _CMP_GT_OQ);
  const __m256d epsilon_valid =
      _mm256_cmp_pd(epsilon_vec, avx_zero, _CMP_NEQ_OQ);
  const __m256d valid_mask = _mm256_and_pd(sigma_valid, epsilon_valid);

  const __m256d sr = _mm256_div_pd(sigma_vec, distance_vec);
  const __m256d sr2 = _mm256_mul_pd(sr, sr);
  const __m256d sr6 = _mm256_mul_pd(_mm256_mul_pd(sr2, sr2), sr2);
  const __m256d sr12 = _mm256_mul_pd(sr6, sr6);
  __m256d energy = _mm256_mul_pd(
      _mm256_mul_pd(avx_four, epsilon_vec), _mm256_sub_pd(sr12, sr6));
  energy = _mm256_and_pd(energy, valid_mask);
  return energy;
}
} // namespace

int cal_one_ele(const MOL &rec, const MOL &lg, cache_sigma_epsilon &CachedData,
                const arguments &Arguments, double pos_x, double pos_y,
                double pos_z, double temp_sigma, double temp_charge,
                double temp_epsilon_lg,
                const Multi_MOL_parameters &rec_parameters,
                const Multi_MOL_parameters &lg_parameters,
                const vector<mol_info> &info_rec,
                const vector<mol_info> &info_lg, cache_energy &Energy, size_t i,
                size_t j);

int cal_energy_amber_intrin(const MOL &rec, const MOL &lg,
                            const vector<mol_info> &info_rec,
                            const vector<mol_info> &info_lg,
                            cache_sigma_epsilon &CachedData, size_t begin,
                            size_t end, cache_energy &Energy, size_t num_snap,
                            const arguments &Arguments,
                            const Multi_MOL_parameters &rec_parameters,
                            const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, nullptr, Arguments);
    return 0;
  }
  Energy.set_energy(0., 0.);
  const bool is_charmm =
      Arguments.force_f.find("CHARMM") != std::string::npos;
  const __m256d avx_half = _mm256_set1_pd(0.5);
  const __m256d v_scale = _mm256_set1_pd(0.01);
  const size_t rec_size{rec.size()};
  const size_t size_vector_rec{rec.size() / 4};
  const double epsilon_in = Arguments.epsilon_in;
  const bool write_cache = !CachedData.elec_ener.empty();
  const bool has_cutoff = Arguments.distance_energy_cutoff > 0.;
  const titan_cutoff::CutoffBlocks cutoff_blocks(
      rec, (rec.size() / 4) * 4, Arguments.distance_energy_cutoff); // ENE-03
  const bool check_self = Arguments.self_int;
  const bool check_interchain = Arguments.flag_interchain_energy;
  const bool has_overrides = !is_charmm && CachedData.has_charmm_pair_overrides();
  const __m256d avx_all_lanes = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
  const __m256d avx_cutoff = _mm256_set1_pd(Arguments.distance_energy_cutoff);

  for (size_t j{begin}; j < end; ++j) {
    const double pos_x{lg.pos.x[j]}, pos_y{lg.pos.y[j]}, pos_z{lg.pos.z[j]};
    alignas(32) double elec_temp[4]{}, vdw_temp_d[4]{}, d_arr[4]{},
        sigma_self_arr[4]{}, lane_mask_vals[4]{};

    const __m256d avx_temp_epsilon_lg =
        _mm256_set1_pd(16. * lg_parameters.epsilon[j]);
    const __m256d avx_temp_charge_lg =
        _mm256_set1_pd(138.935457839 * lg_parameters.charge[j] / epsilon_in);
    const __m256d avx_temp_sigma = _mm256_set1_pd(lg_parameters.sigma[j]);

    const __m256d v_pos_x = _mm256_set1_pd(pos_x);
    const __m256d v_pos_y = _mm256_set1_pd(pos_y);
    const __m256d v_pos_z = _mm256_set1_pd(pos_z);

    for (size_t i = 0; i < size_vector_rec * 4; i += 4) {
      if (const size_t culled = cutoff_blocks.skip(i, pos_x, pos_y, pos_z);
          culled > 0) {
        titan_cutoff::CutoffBlocks::zero_cache(CachedData, rec_size * j + i, culled);
        i += culled - 4;
        continue;
      }
      const __m256d v_x = _mm256_loadu_pd(&rec.pos.x[i]);
      const __m256d v_y = _mm256_loadu_pd(&rec.pos.y[i]);
      const __m256d v_z = _mm256_loadu_pd(&rec.pos.z[i]);

      const __m256d dx = _mm256_sub_pd(v_x, v_pos_x);
      const __m256d dy = _mm256_sub_pd(v_y, v_pos_y);
      const __m256d dz = _mm256_sub_pd(v_z, v_pos_z);

      const __m256d dx2 = _mm256_mul_pd(dx, dx);
      const __m256d dy2 = _mm256_mul_pd(dy, dy);
      const __m256d dz2 = _mm256_mul_pd(dz, dz);

      __m256d r2 = _mm256_add_pd(_mm256_add_pd(dx2, dy2), dz2);
      r2 = _mm256_mul_pd(r2, v_scale);
      const __m256d avx_array_d_rec = _mm256_sqrt_pd(r2);

      const __m256d avx_array_charge_rec =
          _mm256_loadu_pd(&rec_parameters.charge[i]);
      __m256d avx_elec_result =
          _mm256_mul_pd(avx_temp_charge_lg, avx_array_charge_rec);
      avx_elec_result = _mm256_div_pd(avx_elec_result, avx_array_d_rec);

      __m256d sigma_self_vec{};
      __m256d vdw_temp3{};
      if (is_charmm) {
        alignas(32) double sigma_lane[4]{};
        alignas(32) double epsilon_lane[4]{};
        charmm::pair_params_block(CachedData, rec_parameters, lg_parameters, i,
                                  4, j, sigma_lane, epsilon_lane);
        sigma_self_vec = _mm256_loadu_pd(sigma_lane);
        const __m256d epsilon_vec = _mm256_loadu_pd(epsilon_lane);
        vdw_temp3 =
            charmm_lj_energy_avx(sigma_self_vec, epsilon_vec, avx_array_d_rec);
      } else {
        alignas(32) double mixed_epsilon_arr[4]{};
        sigma_self_vec = _mm256_loadu_pd(&rec_parameters.sigma[i]);
        sigma_self_vec = _mm256_add_pd(avx_temp_sigma, sigma_self_vec);
        sigma_self_vec = _mm256_mul_pd(sigma_self_vec, avx_half);
        __m256d mixed_epsilon_vec = _mm256_loadu_pd(&rec_parameters.epsilon[i]);
        mixed_epsilon_vec = _mm256_mul_pd(avx_temp_epsilon_lg, mixed_epsilon_vec);
        mixed_epsilon_vec = _mm256_sqrt_pd(mixed_epsilon_vec);
        _mm256_storeu_pd(d_arr, avx_array_d_rec);
        _mm256_storeu_pd(sigma_self_arr, sigma_self_vec);
        _mm256_storeu_pd(mixed_epsilon_arr, mixed_epsilon_vec);
        titan_simd_dispatch::lj_mixed_sigma_epsilon_distance_fourlanes(
            sigma_self_arr, mixed_epsilon_arr, d_arr, vdw_temp_d);
        if (has_overrides)
          titan_lj::override_lj_lanes(CachedData, rec_parameters,
                                      lg_parameters, i, 4, j, d_arr,
                                      vdw_temp_d);
        vdw_temp3 = _mm256_loadu_pd(vdw_temp_d);
      }

      __m256d final_mask = avx_all_lanes;
      if (check_self) {
        _mm256_storeu_pd(d_arr, avx_array_d_rec);
        _mm256_storeu_pd(sigma_self_arr, sigma_self_vec);
        for (int k = 0; k < 4; ++k) {
          bool keep = true;
          if (has_cutoff && d_arr[k] > Arguments.distance_energy_cutoff)
            keep = false;
          if (keep) {
            if (i + static_cast<size_t>(k) == j ||
                d_arr[k] < sigma_self_arr[k]) {
              keep = false;
            } else if (check_interchain) {
              keep = info_lg[j].get_chain() != info_rec[i + k].get_chain();
            }
          }
          lane_mask_vals[k] = keep ? -1.0 : 0.0;
        }
        final_mask = load_lane_mask_pd(lane_mask_vals);
      } else if (has_cutoff) {
        // ENE-04 (22/10/2026): the cutoff mask as one compare instead of a
        // store, a scalar lane loop and a reload; "not greater" keeps NaN
        // lanes exactly as the scalar test did.
        final_mask = _mm256_cmp_pd(avx_array_d_rec, avx_cutoff, _CMP_NGT_UQ);
      }
      avx_elec_result = _mm256_and_pd(avx_elec_result, final_mask);
      vdw_temp3 = _mm256_and_pd(vdw_temp3, final_mask);

      Energy.elec_ener += hsum_double_avx(avx_elec_result);
      Energy.vdw_ener += hsum_double_avx(vdw_temp3);

      if (write_cache) {
        _mm256_storeu_pd(elec_temp, avx_elec_result);
        _mm256_storeu_pd(vdw_temp_d, vdw_temp3);
        CachedData.set_elec_vdw(rec_size * j + i, elec_temp, vdw_temp_d, 4);
      }
    }

    if (size_vector_rec * 4 < rec_size) {
      for (size_t i{size_vector_rec * 4}; i < rec_size; ++i) {
        if (check_self && i == j) {
          zero_cached_pair_if_present(CachedData, rec_size * j + i);
          continue;
        }
        const double d{0.1 * sqrt(square(rec.pos.x[i] - pos_x) +
                                  square(rec.pos.y[i] - pos_y) +
                                  square(rec.pos.z[i] - pos_z))};
        const size_t pos{rec_size * j + i};

        if (has_cutoff && d > Arguments.distance_energy_cutoff) {
          zero_cached_pair_if_present(CachedData, pos);
          continue;
        }
        double sigma_self =
            0.5 * (lg_parameters.sigma[j] + rec_parameters.sigma[i]);
        double vdw_val = 0.0;
        if (is_charmm) {
          double epsilon_ij{};
          charmm::pair_params(CachedData, rec_parameters, lg_parameters, i, j,
                              sigma_self, epsilon_ij);
          vdw_val = charmm::lj_energy(sigma_self, epsilon_ij, d);
        } else if (lg_parameters.sigma[j] < 0. ||
                   rec_parameters.sigma[i] < 0.) {
          const double c_6{
              c6(0.5 * (lg_parameters.sigma[j] + rec_parameters.sigma[i]) / d)};
          vdw_val = sqrt(16. * lg_parameters.epsilon[j] *
                         rec_parameters.epsilon[i]) *
                    (square(c_6) - 0.000);
        } else {
          const double c_6{
              c6(0.5 * (lg_parameters.sigma[j] + rec_parameters.sigma[i]) / d)};
          vdw_val = sqrt(16. * lg_parameters.epsilon[j] *
                         rec_parameters.epsilon[i]) *
                    (square(c_6) - c_6);
        }
        if (has_overrides)
          titan_lj::override_lj_energy(CachedData, rec_parameters,
                                       lg_parameters, i, j, d, vdw_val);

        if (check_self &&
            ((sigma_self > 0.0 && d < sigma_self) ||
             (check_interchain &&
              info_lg[j].get_chain() == info_rec[i].get_chain()))) {
          zero_cached_pair_if_present(CachedData, pos);
          continue;
        }
        const double elec_val =
            (138.935457839 * lg_parameters.charge[j] / epsilon_in) *
            rec_parameters.charge[i] / d;

        if (write_cache) {
          CachedData.set_elec_vdw(pos, elec_val, vdw_val);
          Energy += CachedData.get_energy(pos);
        } else {
          Energy.elec_ener += elec_val;
          Energy.vdw_ener += vdw_val;
        }
      }
    }
  }
  return 0;
}

int cal_one_ele(const MOL &rec, const MOL &lg, cache_sigma_epsilon &CachedData,
                const arguments &Arguments, double pos_x, double pos_y,
                double pos_z, double temp_sigma, double temp_charge,
                double temp_epsilon_lg,
                const Multi_MOL_parameters &rec_parameters,
                const Multi_MOL_parameters &lg_parameters,
                const vector<mol_info> &info_rec,
                const vector<mol_info> &info_lg, cache_energy &Energy, size_t i,
                size_t j) {
  double d{0.1 *
           sqrt(square(rec.pos.x[i] - pos_x) + square(rec.pos.y[i] - pos_y) +
                square(rec.pos.z[i] - pos_z))};
  if (d <= titan_energy_validation::kMinimumPairDistanceNm)
    return 0; // Guard: beads at same position
  double c_6{c6(0.5 * (temp_sigma + rec_parameters[i].get_sigma()) / d)};

  if (Arguments.distance_energy_cutoff > 0. &&
      d > Arguments.distance_energy_cutoff) {
    size_t pos{rec.size() * j + i};
    zero_cached_pair_if_present(CachedData, pos);
    return 0;
  }

  if (Arguments.self_int == true &&
      (d < 0.5 * (temp_sigma + rec_parameters[i].get_sigma()) ||
       (Arguments.flag_interchain_energy == true &&
         info_lg[j].get_chain() == info_rec[i].get_chain()))) {
    zero_cached_pair_if_present(CachedData, rec.size() * j + i);
    return 0;
  }

  if (CachedData.elec_ener.empty()) {
    Energy.elec_ener += temp_charge * rec_parameters[i].get_charge() / d;
    if (temp_sigma < 0. || rec_parameters[i].get_sigma() < 0.)
      Energy.vdw_ener +=
          sqrt(temp_epsilon_lg * rec_parameters[i].get_epsilon()) *
          (square(c_6) - 0.000);
    else
      Energy.vdw_ener +=
          sqrt(temp_epsilon_lg * rec_parameters[i].get_epsilon()) *
          (square(c_6) - c_6);
  } else {
    size_t pos{rec.size() * j + i};
    if (temp_sigma < 0. || rec_parameters[i].get_sigma() < 0.)
      CachedData.set_elec_vdw(
          pos, temp_charge * rec_parameters[i].get_charge() / d,
          sqrt(rec_parameters[i].get_epsilon() * temp_epsilon_lg) *
              (square(c_6) - 0.000));
    else
      CachedData.set_elec_vdw(
          pos, temp_charge * rec_parameters[i].get_charge() / d,
          sqrt(rec_parameters[i].get_epsilon() * temp_epsilon_lg) *
              (square(c_6) - c_6));
    Energy += CachedData.get_energy(pos);
  }
  return 0;
}

int cal_energy_amber_screening_intrin(
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
  Energy.set_energy(0., 0.);
  constexpr double kScreeningMinDistance =
      titan_energy_validation::kMinimumPairDistanceNm;
  const double epsilon_in{Arguments.epsilon_in};
  const double lambda_debye{
      titan_electrostatics::debye_length_nm(Arguments.salt_concentration, Arguments.gbsa_temperature)};
  const bool is_charmm =
      Arguments.force_f.find("CHARMM") != std::string::npos;
  const bool write_cache = !CachedData.elec_ener.empty();
  const bool has_cutoff = Arguments.distance_energy_cutoff > 0.;
  const titan_cutoff::CutoffBlocks cutoff_blocks(
      rec, (rec.size() / 4) * 4, Arguments.distance_energy_cutoff); // ENE-03
  const bool check_self = Arguments.self_int;
  const bool check_interchain = Arguments.flag_interchain_energy;
  const bool has_overrides = !is_charmm && CachedData.has_charmm_pair_overrides();

  const __m256d avx_epsilon_in = _mm256_set1_pd(epsilon_in);
  const __m256d avx_neg_inv_lambda = _mm256_set1_pd(-1.0 / lambda_debye);
  const __m256d avx_one = _mm256_set1_pd(1.0);
  const __m256d avx_half = _mm256_set1_pd(0.5);
  const __m256d avx_min_distance = _mm256_set1_pd(kScreeningMinDistance);

  for (size_t j{begin}; j < end; ++j) {
    const double pos_x{lg.pos.x[j]}, pos_y{lg.pos.y[j]}, pos_z{lg.pos.z[j]};
    const size_t rec_size{rec.size()}, size_vector_rec{rec.size() / 4};
    alignas(32) double elec_temp[4]{}, vdw_temp_d[4]{}, d[4]{},
        sigma_self_arr[4]{}, lane_mask_vals[4]{};

    const __m256d avx_temp_epsilon_lg =
        _mm256_set1_pd(16. * lg_parameters[j].get_epsilon());
    const __m256d avx_temp_charge_lg =
        _mm256_set1_pd(138.935457839 * lg_parameters[j].get_charge());
    const __m256d avx_temp_sigma = _mm256_set1_pd(lg_parameters[j].get_sigma());

    const __m256d v_pos_x = _mm256_set1_pd(pos_x);
    const __m256d v_pos_y = _mm256_set1_pd(pos_y);
    const __m256d v_pos_z = _mm256_set1_pd(pos_z);
    const __m256d v_scale = _mm256_set1_pd(0.01);

    for (size_t i = 0; i < size_vector_rec * 4; i += 4) {
      if (const size_t culled = cutoff_blocks.skip(i, pos_x, pos_y, pos_z);
          culled > 0) {
        titan_cutoff::CutoffBlocks::zero_cache(CachedData, rec_size * j + i, culled);
        i += culled - 4;
        continue;
      }
      const __m256d v_x = _mm256_loadu_pd(&rec.pos.x[i]);
      const __m256d v_y = _mm256_loadu_pd(&rec.pos.y[i]);
      const __m256d v_z = _mm256_loadu_pd(&rec.pos.z[i]);

      const __m256d dx = _mm256_sub_pd(v_x, v_pos_x);
      const __m256d dy = _mm256_sub_pd(v_y, v_pos_y);
      const __m256d dz = _mm256_sub_pd(v_z, v_pos_z);

      const __m256d dx2 = _mm256_mul_pd(dx, dx);
      const __m256d dy2 = _mm256_mul_pd(dy, dy);
      const __m256d dz2 = _mm256_mul_pd(dz, dz);

      __m256d r2 = _mm256_add_pd(_mm256_add_pd(dx2, dy2), dz2);
      r2 = _mm256_mul_pd(r2, v_scale);
      const __m256d avx_array_d_rec = _mm256_sqrt_pd(r2);
      const __m256d avx_valid_mask =
          _mm256_cmp_pd(avx_array_d_rec, avx_min_distance, _CMP_GT_OQ);
      const __m256d avx_safe_d =
          _mm256_blendv_pd(avx_min_distance, avx_array_d_rec, avx_valid_mask);
      const __m256d avx_inv_d = _mm256_div_pd(avx_one, avx_safe_d);

      const __m256d avx_exp_arg = _mm256_mul_pd(avx_safe_d, avx_neg_inv_lambda);
      // ENE-14 (22/10/2026): exp_pd (3e-16) instead of f_exp_avx (3.2e-6).
      const __m256d avx_exp_val = titan_avx2::exp_pd(avx_exp_arg);

      const __m256d avx_array_charge_rec =
          _mm256_loadu_pd(&rec_parameters.charge[i]);

      __m256d avx_elec_result =
          _mm256_mul_pd(avx_temp_charge_lg, avx_array_charge_rec);
      avx_elec_result = _mm256_mul_pd(avx_elec_result, avx_exp_val);
      avx_elec_result = _mm256_mul_pd(avx_elec_result, avx_inv_d);
      avx_elec_result = _mm256_div_pd(avx_elec_result, avx_epsilon_in);

      __m256d sigma_self_vec{};
      __m256d vdw_temp3{};
      if (is_charmm) {
        alignas(32) double sigma_lane[4]{};
        alignas(32) double epsilon_lane[4]{};
        charmm::pair_params_block(CachedData, rec_parameters, lg_parameters, i,
                                  4, j, sigma_lane, epsilon_lane);
        sigma_self_vec = _mm256_loadu_pd(sigma_lane);
        const __m256d epsilon_vec = _mm256_loadu_pd(epsilon_lane);
        vdw_temp3 = charmm_lj_energy_avx(sigma_self_vec, epsilon_vec, avx_safe_d);
      } else {
        alignas(32) double mixed_epsilon_arr[4]{}, safe_d_arr[4]{};
        sigma_self_vec = _mm256_loadu_pd(&rec_parameters.sigma[i]);
        sigma_self_vec = _mm256_add_pd(avx_temp_sigma, sigma_self_vec);
        sigma_self_vec = _mm256_mul_pd(sigma_self_vec, avx_half);
        __m256d mixed_epsilon_vec = _mm256_loadu_pd(&rec_parameters.epsilon[i]);
        mixed_epsilon_vec = _mm256_mul_pd(avx_temp_epsilon_lg, mixed_epsilon_vec);
        mixed_epsilon_vec = _mm256_sqrt_pd(mixed_epsilon_vec);
        _mm256_storeu_pd(safe_d_arr, avx_safe_d);
        _mm256_storeu_pd(sigma_self_arr, sigma_self_vec);
        _mm256_storeu_pd(mixed_epsilon_arr, mixed_epsilon_vec);
        titan_simd_dispatch::lj_mixed_sigma_epsilon_distance_fourlanes(
            sigma_self_arr, mixed_epsilon_arr, safe_d_arr, vdw_temp_d);
        if (has_overrides)
          titan_lj::override_lj_lanes(CachedData, rec_parameters,
                                      lg_parameters, i, 4, j, safe_d_arr,
                                      vdw_temp_d);
        vdw_temp3 = _mm256_loadu_pd(vdw_temp_d);
      }

      _mm256_storeu_pd(d, avx_array_d_rec);
      _mm256_storeu_pd(sigma_self_arr, sigma_self_vec);
      for (int k = 0; k < 4; ++k) {
        bool keep = std::isfinite(d[k]) && d[k] > kScreeningMinDistance;
        if (keep && has_cutoff && d[k] > Arguments.distance_energy_cutoff)
          keep = false;
        if (keep && check_self) {
          if (i + static_cast<size_t>(k) == j ||
              (sigma_self_arr[k] > 0.0 && d[k] < sigma_self_arr[k])) {
            keep = false;
          } else if (check_interchain) {
            keep = info_lg[j].get_chain() != info_rec[i + k].get_chain();
          }
        }
        lane_mask_vals[k] = keep ? -1.0 : 0.0;
      }

      const __m256d final_mask =
          _mm256_and_pd(avx_valid_mask, load_lane_mask_pd(lane_mask_vals));
      avx_elec_result = _mm256_and_pd(avx_elec_result, final_mask);
      vdw_temp3 = _mm256_and_pd(vdw_temp3, final_mask);

      Energy.elec_ener += hsum_double_avx(avx_elec_result);
      Energy.vdw_ener += hsum_double_avx(vdw_temp3);

      if (write_cache) {
        _mm256_storeu_pd(elec_temp, avx_elec_result);
        _mm256_storeu_pd(vdw_temp_d, vdw_temp3);
        CachedData.set_elec_vdw(rec_size * j + i, elec_temp, vdw_temp_d, 4);
      }
    }

    if (size_vector_rec * 4 < rec_size) {
      for (size_t i{size_vector_rec * 4}; i < rec_size; ++i) {
        if (check_self && i == j) {
          zero_cached_pair_if_present(CachedData, rec_size * j + i);
          continue;
        }
        double d{0.1 * sqrt(square(rec.pos.x[i] - pos_x) +
                            square(rec.pos.y[i] - pos_y) +
                            square(rec.pos.z[i] - pos_z))};
        const size_t pos{rec_size * j + i};
        if (!std::isfinite(d) || d <= kScreeningMinDistance) {
          zero_cached_pair_if_present(CachedData, pos);
          continue;
        }

        if (has_cutoff && d > Arguments.distance_energy_cutoff) {
          zero_cached_pair_if_present(CachedData, pos);
          continue;
        }

        double sigma_self =
            0.5 * (lg_parameters[j].get_sigma() + rec_parameters[i].get_sigma());
        double v_val = 0.0;
        if (is_charmm) {
          double epsilon_ij{};
          charmm::pair_params(CachedData, rec_parameters, lg_parameters, i, j,
                              sigma_self, epsilon_ij);
          v_val = charmm::lj_energy(sigma_self, epsilon_ij, d);
        } else {
          const double c_6{c6(sigma_self / d)};
          v_val = sqrt(lg_parameters[j].get_epsilon() * 16. *
                       rec_parameters[i].get_epsilon()) *
                  (square(c_6) - c_6);
        }
        if (has_overrides)
          titan_lj::override_lj_energy(CachedData, rec_parameters,
                                       lg_parameters, i, j, d, v_val);
        if (check_self &&
            ((sigma_self > 0.0 && d < sigma_self) ||
             (check_interchain &&
              info_lg[j].get_chain() == info_rec[i].get_chain()))) {
          zero_cached_pair_if_present(CachedData, pos);
          continue;
        }

        double term_exp = exp(-d / lambda_debye);
        double e_val = (138.935457839 * lg_parameters[j].get_charge() *
                        rec_parameters[i].get_charge() * term_exp) /
                       (d * epsilon_in);

        if (write_cache) {
          CachedData.set_elec_vdw(pos, e_val, v_val);
          Energy += CachedData.get_energy(pos);
        } else {
          Energy.elec_ener += e_val;
          Energy.vdw_ener += v_val;
        }
      }
    }
  }
  return 0;
}

int cal_energy_gromos_intrin(const MOL &rec, const MOL &lg,
                             const vector<mol_info> &info_rec,
                             const vector<mol_info> &info_lg,
                             cache_sigma_epsilon &CachedData, size_t begin,
                             size_t end, cache_energy &Energy, size_t num_snap,
                             const arguments &Arguments,
                             const Multi_MOL_parameters &rec_parameters,
                             const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, nullptr, Arguments);
    return 0;
  }
  Energy.set_energy(0., 0.);
  const __m256d avx_epsilon_in = _mm256_set1_pd(Arguments.epsilon_in);
  const __m256d avx_one = _mm256_set1_pd(1.0);
  const bool write_cache = !CachedData.elec_ener.empty();
  const bool has_cutoff = Arguments.distance_energy_cutoff > 0.;
  const titan_cutoff::CutoffBlocks cutoff_blocks(
      rec, (rec.size() / 4) * 4, Arguments.distance_energy_cutoff); // ENE-03
  const bool check_self = Arguments.self_int;
  const bool check_interchain = Arguments.flag_interchain_energy;

  for (size_t j{begin}; j < end; ++j) {
    const double pos_x{lg.pos.x[j]}, pos_y{lg.pos.y[j]}, pos_z{lg.pos.z[j]};
    const size_t rec_size{rec.size()}, size_vector_rec{rec.size() / 4};
    alignas(32) double elec_temp[4]{}, vdw_temp_d[4]{}, d[4]{},
        lane_mask_vals[4]{};

    const __m256d avx_temp_charge_lg =
        _mm256_set1_pd(138.935457839 * lg_parameters[j].get_charge());
    const __m256d avx_temp_epsilon_lg =
        _mm256_set1_pd(lg_parameters[j].get_epsilon());
    const __m256d avx_temp_sigma_lg =
        _mm256_set1_pd(lg_parameters[j].get_sigma());

    const __m256d v_pos_x = _mm256_set1_pd(pos_x);
    const __m256d v_pos_y = _mm256_set1_pd(pos_y);
    const __m256d v_pos_z = _mm256_set1_pd(pos_z);
    const __m256d v_scale = _mm256_set1_pd(0.01);

    for (size_t i = 0; i < size_vector_rec * 4; i += 4) {
      if (const size_t culled = cutoff_blocks.skip(i, pos_x, pos_y, pos_z);
          culled > 0) {
        titan_cutoff::CutoffBlocks::zero_cache(CachedData, rec_size * j + i, culled);
        i += culled - 4;
        continue;
      }
      const __m256d v_x = _mm256_loadu_pd(&rec.pos.x[i]);
      const __m256d v_y = _mm256_loadu_pd(&rec.pos.y[i]);
      const __m256d v_z = _mm256_loadu_pd(&rec.pos.z[i]);

      const __m256d dx = _mm256_sub_pd(v_x, v_pos_x);
      const __m256d dy = _mm256_sub_pd(v_y, v_pos_y);
      const __m256d dz = _mm256_sub_pd(v_z, v_pos_z);

      const __m256d dx2 = _mm256_mul_pd(dx, dx);
      const __m256d dy2 = _mm256_mul_pd(dy, dy);
      const __m256d dz2 = _mm256_mul_pd(dz, dz);

      __m256d r2 = _mm256_add_pd(_mm256_add_pd(dx2, dy2), dz2);
      r2 = _mm256_mul_pd(r2, v_scale);
      const __m256d avx_array_d_rec = _mm256_sqrt_pd(r2);
      const __m256d avx_inv_d = _mm256_div_pd(avx_one, avx_array_d_rec);

      const __m256d avx_array_charge_rec =
          _mm256_loadu_pd(&rec_parameters.charge[i]);
      __m256d avx_elec_result =
          _mm256_mul_pd(avx_temp_charge_lg, avx_array_charge_rec);
      avx_elec_result = _mm256_mul_pd(avx_elec_result, avx_inv_d);
      avx_elec_result = _mm256_div_pd(avx_elec_result, avx_epsilon_in);

      const __m256d avx_rec_epsilon =
          _mm256_loadu_pd(&rec_parameters.epsilon[i]);
      const __m256d avx_rec_sigma = _mm256_loadu_pd(&rec_parameters.sigma[i]);

      __m256d term_eps =
          _mm256_sqrt_pd(_mm256_mul_pd(avx_temp_epsilon_lg, avx_rec_epsilon));
      __m256d term_sig =
          _mm256_sqrt_pd(_mm256_mul_pd(avx_temp_sigma_lg, avx_rec_sigma));

      if (CachedData.has_charmm_pair_overrides()) {
        alignas(32) double c6_values[4], c12_values[4];
        for (std::size_t lane=0;lane<4;++lane)
          gromos::pair_params(CachedData,rec_parameters,lg_parameters,i+lane,j,c6_values[lane],c12_values[lane]);
        term_sig=_mm256_load_pd(c6_values);term_eps=_mm256_load_pd(c12_values);
      }
      __m256d r2_vdw = _mm256_mul_pd(avx_array_d_rec, avx_array_d_rec);
      __m256d r6 = _mm256_mul_pd(r2_vdw, r2_vdw);
      r6 = _mm256_mul_pd(r6, r2_vdw);

      const __m256d inv_r6 = _mm256_div_pd(avx_one, r6);
      const __m256d inv_r12 = _mm256_mul_pd(inv_r6, inv_r6);

      __m256d vdw_res = _mm256_sub_pd(_mm256_mul_pd(term_eps, inv_r12),
                                      _mm256_mul_pd(term_sig, inv_r6));

      _mm256_storeu_pd(d, avx_array_d_rec);
      for (int k = 0; k < 4; ++k) {
        bool keep = true;
        if (has_cutoff && d[k] > Arguments.distance_energy_cutoff)
          keep = false;
        if (keep && check_self) {
          if (i + static_cast<size_t>(k) == j) {
            keep = false;
          } else if (check_interchain) {
            keep = info_lg[j].get_chain() != info_rec[i + k].get_chain();
          }
        }
        lane_mask_vals[k] = keep ? -1.0 : 0.0;
      }

      const __m256d final_mask = load_lane_mask_pd(lane_mask_vals);
      avx_elec_result = _mm256_and_pd(avx_elec_result, final_mask);
      vdw_res = _mm256_and_pd(vdw_res, final_mask);

      Energy.elec_ener += hsum_double_avx(avx_elec_result);
      Energy.vdw_ener += hsum_double_avx(vdw_res);

      if (write_cache) {
        _mm256_storeu_pd(elec_temp, avx_elec_result);
        _mm256_storeu_pd(vdw_temp_d, vdw_res);
        CachedData.set_elec_vdw(rec_size * j + i, elec_temp, vdw_temp_d, 4);
      }
    }

    if (size_vector_rec * 4 < rec_size) {
      for (size_t i{size_vector_rec * 4}; i < rec_size; ++i) {
        if (check_self && i == j) {
          zero_cached_pair_if_present(CachedData, rec_size * j + i);
          continue;
        }
        double d{0.1 * sqrt(square(rec.pos.x[i] - pos_x) +
                            square(rec.pos.y[i] - pos_y) +
                            square(rec.pos.z[i] - pos_z))};
        const size_t pos{rec_size * j + i};
        if (has_cutoff && d > Arguments.distance_energy_cutoff) {
          zero_cached_pair_if_present(CachedData, pos);
          continue;
        }

        if (check_self &&
            ((i == j) || (check_interchain &&
                        info_lg[j].get_chain() == info_rec[i].get_chain()))) {
          zero_cached_pair_if_present(CachedData, pos);
          continue;
        }

        double temp_ch = 138.935457839 * lg_parameters[j].get_charge() *
                         rec_parameters[i].get_charge() /
                         (d * Arguments.epsilon_in);
        double c_6 = c6(d); // c6(d) = d^6
        double sig_term{}, eps_term{};
        gromos::pair_params(CachedData,rec_parameters,lg_parameters,i,j,sig_term,eps_term);
        double vdw_val = eps_term / square(c_6) - sig_term / c_6;

        if (write_cache) {
          CachedData.set_elec_vdw(pos, temp_ch, vdw_val);
          Energy += CachedData.get_energy(pos);
        } else {
          Energy.elec_ener += temp_ch;
          Energy.vdw_ener += vdw_val;
        }
      }
    }
  }
  return 0;
}

void cal_energy_opls_intrin(const MOL &rec, const MOL &lg,
                            const vector<mol_info> &info_rec,
                            const vector<mol_info> &info_lg,
                            cache_sigma_epsilon &CachedData, size_t begin,
                            size_t end, cache_energy &Energy, size_t num_snap,
                            const arguments &Arguments,
                            const Multi_MOL_parameters &rec_parameters,
                            const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, nullptr, Arguments);
    return;
  }
  Energy.set_energy(0., 0.);
  const __m256d avx_one = _mm256_set1_pd(1.0);
  const bool write_cache = !CachedData.elec_ener.empty();
  const bool has_cutoff = Arguments.distance_energy_cutoff > 0.;
  const titan_cutoff::CutoffBlocks cutoff_blocks(
      rec, (rec.size() / 4) * 4, Arguments.distance_energy_cutoff); // ENE-03
  const bool check_self = Arguments.self_int;
  const bool check_interchain = Arguments.flag_interchain_energy;
  const bool has_overrides = CachedData.has_charmm_pair_overrides();

  for (size_t j{begin}; j < end; ++j) {
    const double pos_x{lg.pos.x[j]}, pos_y{lg.pos.y[j]}, pos_z{lg.pos.z[j]};
    const size_t rec_size{rec.size()}, size_vector_rec{rec.size() / 4};
    alignas(32) double elec_temp[4]{}, vdw_temp_d[4]{}, d[4]{}, c6_arr[4]{},
        lane_mask_vals[4]{};

    const __m256d avx_temp_epsilon_lg =
        _mm256_set1_pd(16. * lg_parameters[j].get_epsilon());
    const __m256d avx_temp_charge_lg = _mm256_set1_pd(
        138.935457839 * lg_parameters[j].get_charge() / Arguments.epsilon_in);
    const __m256d avx_temp_sigma = _mm256_set1_pd(lg_parameters[j].get_sigma());

    const __m256d v_pos_x = _mm256_set1_pd(pos_x);
    const __m256d v_pos_y = _mm256_set1_pd(pos_y);
    const __m256d v_pos_z = _mm256_set1_pd(pos_z);
    const __m256d v_scale = _mm256_set1_pd(0.01);

    for (size_t i = 0; i < size_vector_rec * 4; i += 4) {
      if (const size_t culled = cutoff_blocks.skip(i, pos_x, pos_y, pos_z);
          culled > 0) {
        titan_cutoff::CutoffBlocks::zero_cache(CachedData, rec_size * j + i, culled);
        i += culled - 4;
        continue;
      }
      const __m256d v_x = _mm256_loadu_pd(&rec.pos.x[i]);
      const __m256d v_y = _mm256_loadu_pd(&rec.pos.y[i]);
      const __m256d v_z = _mm256_loadu_pd(&rec.pos.z[i]);

      const __m256d dx = _mm256_sub_pd(v_x, v_pos_x);
      const __m256d dy = _mm256_sub_pd(v_y, v_pos_y);
      const __m256d dz = _mm256_sub_pd(v_z, v_pos_z);

      const __m256d dx2 = _mm256_mul_pd(dx, dx);
      const __m256d dy2 = _mm256_mul_pd(dy, dy);
      const __m256d dz2 = _mm256_mul_pd(dz, dz);

      __m256d r2 = _mm256_add_pd(_mm256_add_pd(dx2, dy2), dz2);
      r2 = _mm256_mul_pd(r2, v_scale);
      const __m256d avx_array_d_rec = _mm256_sqrt_pd(r2);
      const __m256d avx_inv_d = _mm256_div_pd(avx_one, avx_array_d_rec);
      const __m256d avx_inv_d2 = _mm256_mul_pd(avx_inv_d, avx_inv_d);

      const __m256d avx_array_charge_rec =
          _mm256_loadu_pd(&rec_parameters.charge[i]);
      __m256d avx_elec_result =
          _mm256_mul_pd(avx_temp_charge_lg, avx_array_charge_rec);
      avx_elec_result = _mm256_mul_pd(avx_elec_result, avx_inv_d);

      __m256d vdw_temp = _mm256_loadu_pd(&rec_parameters.epsilon[i]);
      vdw_temp = _mm256_sqrt_pd(_mm256_mul_pd(avx_temp_epsilon_lg, vdw_temp));

      __m256d vdw_temp2 = _mm256_loadu_pd(&rec_parameters.sigma[i]);
      vdw_temp2 = _mm256_mul_pd(avx_temp_sigma, vdw_temp2);
      vdw_temp2 = _mm256_sqrt_pd(vdw_temp2); // sqrt(sigma1 * sigma2)

      // Calculate (sigma/d)^6 using d^2
      vdw_temp2 = _mm256_mul_pd(vdw_temp2, vdw_temp2);  // sigma_comb^2
      vdw_temp2 = _mm256_mul_pd(vdw_temp2, avx_inv_d2); // (sigma/d)^2

      __m256d vdw_temp3 = vdw_temp2;
      vdw_temp2 = _mm256_mul_pd(vdw_temp2, vdw_temp2); // ^4
      vdw_temp2 = _mm256_mul_pd(vdw_temp2, vdw_temp3); // ^6 (c6)

      vdw_temp3 = _mm256_sub_pd(_mm256_mul_pd(vdw_temp2, vdw_temp2), vdw_temp2);
      vdw_temp3 = _mm256_mul_pd(vdw_temp, vdw_temp3); // eps * (c12 - c6)

      _mm256_storeu_pd(d, avx_array_d_rec);
      _mm256_storeu_pd(c6_arr, vdw_temp2);
      if (has_overrides) {
        _mm256_storeu_pd(vdw_temp_d, vdw_temp3);
        titan_lj::override_lj_lanes(CachedData, rec_parameters, lg_parameters,
                                    i, 4, j, d, vdw_temp_d);
        vdw_temp3 = _mm256_loadu_pd(vdw_temp_d);
      }
      for (int k = 0; k < 4; ++k) {
        bool keep = true;
        if (check_self &&
            (i + static_cast<size_t>(k) == j || c6_arr[k] > 1.0 ||
             (check_interchain &&
                                 info_lg[j].get_chain() ==
                                     info_rec[i + k].get_chain()))) {
          keep = false;
        }
        if (keep && has_cutoff && d[k] > Arguments.distance_energy_cutoff)
          keep = false;
        lane_mask_vals[k] = keep ? -1.0 : 0.0;
      }

      const __m256d final_mask = load_lane_mask_pd(lane_mask_vals);
      avx_elec_result = _mm256_and_pd(avx_elec_result, final_mask);
      vdw_temp3 = _mm256_and_pd(vdw_temp3, final_mask);

      Energy.elec_ener += hsum_double_avx(avx_elec_result);
      Energy.vdw_ener += hsum_double_avx(vdw_temp3);

      if (write_cache) {
        _mm256_storeu_pd(elec_temp, avx_elec_result);
        _mm256_storeu_pd(vdw_temp_d, vdw_temp3);
        CachedData.set_elec_vdw(rec_size * j + i, elec_temp, vdw_temp_d, 4);
      }
    }

    if (size_vector_rec * 4 < rec_size)
      for (size_t i{size_vector_rec * 4}; i < rec_size; ++i) {
        if (check_self && i == j) {
          zero_cached_pair_if_present(CachedData, rec_size * j + i);
          continue;
        }
        double d{0.1 * sqrt(square(rec.pos.x[i] - pos_x) +
                            square(rec.pos.y[i] - pos_y) +
                            square(rec.pos.z[i] - pos_z))},
            c_6{c6(sqrt(lg_parameters[j].get_sigma() *
                        rec_parameters[i].get_sigma()) /
                   d)};

        const size_t pos{rec_size * j + i};
        if (check_self &&
            (c_6 > 1. || (check_interchain &&
                          info_lg[j].get_chain() == info_rec[i].get_chain()))) {
          zero_cached_pair_if_present(CachedData, pos);
          continue;
        }

        if (has_cutoff && d > Arguments.distance_energy_cutoff) {
          zero_cached_pair_if_present(CachedData, pos);
          continue;
        }

        double e_val = 138.935457839 * lg_parameters[j].get_charge() *
                       rec_parameters[i].get_charge() /
                       (d * Arguments.epsilon_in);
        double v_val = sqrt(lg_parameters[j].get_epsilon() * 16. *
                            rec_parameters[i].get_epsilon()) *
                       (square(c_6) - c_6);
        if (has_overrides)
          titan_lj::override_lj_energy(CachedData, rec_parameters,
                                       lg_parameters, i, j, d, v_val);

        if (write_cache) {
          CachedData.set_elec_vdw(pos, e_val, v_val);
          Energy += CachedData.get_energy(pos);
        } else {
          Energy.elec_ener += e_val;
          Energy.vdw_ener += v_val;
        }
      }
  }
}

// MARTINI: AVX2 energy kernel using the topology-derived LJ pair matrix and
// Martini reaction-field electrostatics. This stays MARTINI-only and does not
// alter all-atom kernels.
int cal_energy_martini_intrin(const MOL &rec, const MOL &lg,
                              const vector<mol_info> &info_rec,
                              const vector<mol_info> &info_lg,
                              cache_sigma_epsilon &CachedData, size_t begin,
                              size_t end, cache_energy &Energy, size_t num_snap,
                              const arguments &Arguments,
                              const Multi_MOL_parameters &rec_parameters,
                              const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, nullptr, Arguments);
    return 0;
  }
  Energy.set_energy(0., 0.);
  const auto elec_cfg = martini::electrostatic_config(Arguments);
  const __m256d v_scale = _mm256_set1_pd(0.01);
  const __m256d avx_one = _mm256_set1_pd(1.0);
  const __m256d avx_zero = _mm256_setzero_pd();
  const __m256d avx_neg_one = _mm256_set1_pd(-1.0);
  const __m256d avx_half = _mm256_set1_pd(0.5);
  const __m256d avx_four = _mm256_set1_pd(4.0);
  const __m256d avx_charge_scale =
      _mm256_set1_pd(martini::kCoulomb / elec_cfg.epsilon_r);
  const __m256d avx_k_rf = _mm256_set1_pd(elec_cfg.k_rf);
  const __m256d avx_c_rf = _mm256_set1_pd(elec_cfg.c_rf);
  const __m256d avx_min_d = _mm256_set1_pd(
      titan_energy_validation::kMinimumPairDistanceNm);
  const size_t rec_size = rec.size();
  const size_t size_vector_rec = rec_size / 4;
  const bool write_cache = !CachedData.elec_ener.empty();
  const bool has_pair_table = CachedData.has_martini_pair_table();
  const int martini_types = CachedData.martini_types();
  const double *pair_sigma = CachedData.martini_sigma_ptr();
  const double *pair_epsilon = CachedData.martini_epsilon_ptr();
  const double *rec_x = rec.pos.x.data();
  const double *rec_y = rec.pos.y.data();
  const double *rec_z = rec.pos.z.data();
  const double *rec_charge = rec_parameters.charge.data();
  const double *rec_sigma = rec_parameters.sigma.data();
  const double *rec_epsilon = rec_parameters.epsilon.data();
  const int32_t *rec_type = rec_parameters.type_id_ptr();

  for (size_t j{begin}; j < end; ++j) {
    const double pos_x{lg.pos.x[j]}, pos_y{lg.pos.y[j]}, pos_z{lg.pos.z[j]};
    const int lg_type = lg_parameters.get_type_id(j);
    const __m256d avx_temp_charge_lg = _mm256_mul_pd(
        avx_charge_scale, _mm256_set1_pd(lg_parameters.charge[j]));
    const __m256d avx_lg_sigma = _mm256_set1_pd(lg_parameters.sigma[j]);
    const __m256d avx_lg_eps = _mm256_set1_pd(lg_parameters.epsilon[j]);
    const __m256d v_pos_x = _mm256_set1_pd(pos_x);
    const __m256d v_pos_y = _mm256_set1_pd(pos_y);
    const __m256d v_pos_z = _mm256_set1_pd(pos_z);
    const int lg_row_offset = lg_type >= 0 ? lg_type * martini_types : 0;

    for (size_t i = 0; i < size_vector_rec * 4; i += 4) {
      const __m256d v_x = _mm256_loadu_pd(&rec_x[i]);
      const __m256d v_y = _mm256_loadu_pd(&rec_y[i]);
      const __m256d v_z = _mm256_loadu_pd(&rec_z[i]);

      const __m256d dx = _mm256_sub_pd(v_x, v_pos_x);
      const __m256d dy = _mm256_sub_pd(v_y, v_pos_y);
      const __m256d dz = _mm256_sub_pd(v_z, v_pos_z);

      const __m256d dx2 = _mm256_mul_pd(dx, dx);
      const __m256d dy2 = _mm256_mul_pd(dy, dy);
      const __m256d dz2 = _mm256_mul_pd(dz, dz);

      __m256d r2 = _mm256_add_pd(_mm256_add_pd(dx2, dy2), dz2);
      r2 = _mm256_mul_pd(r2, v_scale);
      __m256d avx_d = _mm256_sqrt_pd(r2);
      const __m256d avx_d_mask = _mm256_cmp_pd(avx_d, avx_min_d, _CMP_GT_OQ);
      __m256d avx_d_safe =
          _mm256_blendv_pd(avx_one, avx_d, avx_d_mask);

      __m256d sigma_vec = avx_neg_one;
      __m256d epsilon_vec = avx_neg_one;
      // ENE-17(h): ids outside the MARTINI table (an id from another type
      // space) take the per-atom fallback instead of gathering out of bounds.
      if (has_pair_table && lg_type >= 0 && lg_type < martini_types) {
        alignas(16) int pair_idx[4];
        alignas(32) double lookup_mask_vals[4];
        for (int k = 0; k < 4; ++k) {
          const int rec_type_lane = rec_type ? rec_type[i + k] : -1;
          if (rec_type_lane >= 0 && rec_type_lane < martini_types) {
            pair_idx[k] = lg_row_offset + rec_type_lane;
            lookup_mask_vals[k] = -1.0;
          } else {
            pair_idx[k] = 0;
            lookup_mask_vals[k] = 0.0;
          }
        }
#ifdef __AVX2__
        const __m128i gather_idx =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(pair_idx));
        sigma_vec = _mm256_i32gather_pd(pair_sigma, gather_idx, 8);
        epsilon_vec = _mm256_i32gather_pd(pair_epsilon, gather_idx, 8);
#else
        alignas(32) double sigma_lane[4], epsilon_lane[4];
        for (int k = 0; k < 4; ++k) {
          sigma_lane[k] = pair_sigma[pair_idx[k]];
          epsilon_lane[k] = pair_epsilon[pair_idx[k]];
        }
        sigma_vec = _mm256_loadu_pd(sigma_lane);
        epsilon_vec = _mm256_loadu_pd(epsilon_lane);
#endif
        const __m256d lookup_mask = _mm256_loadu_pd(lookup_mask_vals);
        sigma_vec = _mm256_blendv_pd(avx_neg_one, sigma_vec, lookup_mask);
        epsilon_vec = _mm256_blendv_pd(avx_neg_one, epsilon_vec, lookup_mask);
      }

      __m256d fallback_sigma =
          _mm256_mul_pd(_mm256_add_pd(avx_lg_sigma, _mm256_loadu_pd(&rec_sigma[i])),
                        avx_half);
      __m256d fallback_epsilon = _mm256_sqrt_pd(
          _mm256_mul_pd(avx_lg_eps, _mm256_loadu_pd(&rec_epsilon[i])));
      __m256d sigma_missing =
          _mm256_cmp_pd(sigma_vec, avx_zero, _CMP_LT_OQ);
      __m256d epsilon_missing =
          _mm256_cmp_pd(epsilon_vec, avx_zero, _CMP_LT_OQ);
      sigma_vec = _mm256_blendv_pd(sigma_vec, fallback_sigma, sigma_missing);
      epsilon_vec =
          _mm256_blendv_pd(epsilon_vec, fallback_epsilon, epsilon_missing);

      __m256d inv_d = _mm256_div_pd(avx_one, avx_d_safe);
      __m256d avx_elec_result =
          _mm256_mul_pd(avx_temp_charge_lg, _mm256_loadu_pd(&rec_charge[i]));
      __m256d rf_term = _mm256_sub_pd(
          _mm256_add_pd(inv_d, _mm256_mul_pd(avx_k_rf, r2)), avx_c_rf);
      avx_elec_result = _mm256_mul_pd(avx_elec_result, rf_term);

      __m256d sr = _mm256_div_pd(sigma_vec, avx_d_safe);
      __m256d sr2 = _mm256_mul_pd(sr, sr);
      __m256d sr4 = _mm256_mul_pd(sr2, sr2);
      __m256d sr6 = _mm256_mul_pd(sr4, sr2);
      __m256d sr12 = _mm256_mul_pd(sr6, sr6);
      __m256d vdw_result =
          _mm256_mul_pd(_mm256_mul_pd(avx_four, epsilon_vec),
                        _mm256_sub_pd(sr12, sr6));

      alignas(32) double d_arr[4], sigma_arr[4], lane_mask_vals[4];
      _mm256_storeu_pd(d_arr, avx_d);
      _mm256_storeu_pd(sigma_arr, sigma_vec);

      for (int k = 0; k < 4; ++k) {
        bool keep =
            d_arr[k] > titan_energy_validation::kMinimumPairDistanceNm;
        if (keep)
          keep = d_arr[k] <= elec_cfg.cutoff_nm;
        if (keep && Arguments.self_int == true) {
          keep = d_arr[k] >= sigma_arr[k];
          if (keep && Arguments.flag_interchain_energy == true)
            keep = info_lg[j].get_chain() != info_rec[i + k].get_chain();
        }
        lane_mask_vals[k] = keep ? -1.0 : 0.0;
      }

      const __m256d final_mask = load_lane_mask_pd(lane_mask_vals);
      avx_elec_result = _mm256_and_pd(avx_elec_result, final_mask);
      vdw_result = _mm256_and_pd(vdw_result, final_mask);

      Energy.elec_ener += hsum_double_avx(avx_elec_result);
      Energy.vdw_ener += hsum_double_avx(vdw_result);

      if (write_cache) {
        alignas(32) double elec_temp[4], vdw_temp_d[4];
        _mm256_storeu_pd(elec_temp, avx_elec_result);
        _mm256_storeu_pd(vdw_temp_d, vdw_result);
        const size_t pos_base = rec_size * j + i;
        for (int k = 0; k < 4; ++k)
          CachedData.set_elec_vdw(pos_base + k, elec_temp[k], vdw_temp_d[k]);
      }
    }

    if (size_vector_rec * 4 < rec_size) {
      for (size_t i{size_vector_rec * 4}; i < rec_size; ++i) {
        double d{0.1 * sqrt(square(rec.pos.x[i] - pos_x) +
                            square(rec.pos.y[i] - pos_y) +
                            square(rec.pos.z[i] - pos_z))};
        size_t pos{rec_size * j + i};
        if (d <= titan_energy_validation::kMinimumPairDistanceNm) {
          if (write_cache)
            CachedData.set_elec_vdw(pos, 0., 0.);
          continue;
        }

        double sigma_ij{}, epsilon_ij{};
        martini::pair_params(CachedData, rec_parameters, lg_parameters, i, j,
                             sigma_ij, epsilon_ij);
        if (!martini::within_cutoff(elec_cfg, d)) {
          if (write_cache)
            CachedData.set_elec_vdw(pos, 0., 0.);
          continue;
        }
        if (martini::excluded_pair(Arguments, info_rec, info_lg, i, j, d,
                                   sigma_ij)) {
          if (write_cache)
            CachedData.set_elec_vdw(pos, 0., 0.);
          continue;
        }

        const double elec_val = martini::electro_energy(
            elec_cfg, rec_parameters.charge[i], lg_parameters.charge[j], d);
        const double vdw_val = martini::lj_energy(sigma_ij, epsilon_ij, d);

        if (write_cache)
          CachedData.set_elec_vdw(pos, elec_val, vdw_val);
        Energy.elec_ener += elec_val;
        Energy.vdw_ener += vdw_val;
      }
    }
  }
  return 0;
}
#endif
