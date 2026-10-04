#ifndef CHARMM_NONBONDED_H
#define CHARMM_NONBONDED_H

#include "MOL_para.h"
#include "cache_sigma_epsilon.h"
#include <algorithm>
#include <cstdint>
#include <cmath>

namespace charmm {

inline std::uint64_t make_pair_override_key(int32_t type_a, int32_t type_b) {
  const std::uint32_t lo =
      static_cast<std::uint32_t>(std::min(type_a, type_b));
  const std::uint32_t hi =
      static_cast<std::uint32_t>(std::max(type_a, type_b));
  return (static_cast<std::uint64_t>(lo) << 32) |
         static_cast<std::uint64_t>(hi);
}

inline bool lookup_pair_override(const cache_sigma_epsilon &cache,
                                 const Multi_MOL_parameters &rec_parameters,
                                 const Multi_MOL_parameters &lg_parameters,
                                 size_t rec_pos, size_t lg_pos, double &sigma,
                                 double &epsilon) {
  const int32_t rec_type = rec_parameters.get_type_id(rec_pos);
  const int32_t lg_type = lg_parameters.get_type_id(lg_pos);
  if (rec_type < 0 || lg_type < 0)
    return false;

  if (cache.has_charmm_pair_table()) {
    const int ntypes = cache.charmm_types();
    if (rec_type >= ntypes || lg_type >= ntypes)
      return false;

    const size_t idx = static_cast<size_t>(rec_type) *
                           static_cast<size_t>(ntypes) +
                       static_cast<size_t>(lg_type);
    if (idx >= cache.charmm_pair_sigma.size() ||
        idx >= cache.charmm_pair_epsilon.size())
      return false;

    const double sigma_override = cache.charmm_pair_sigma[idx];
    const double epsilon_override = cache.charmm_pair_epsilon[idx];
    if (sigma_override < 0.0 || epsilon_override < 0.0)
      return false;

    sigma = sigma_override;
    epsilon = epsilon_override;
    return true;
  }

  if (!cache.has_charmm_pair_sparse())
    return false;

  const auto key = make_pair_override_key(rec_type, lg_type);
  const auto &keys = cache.charmm_sparse_keys();
  const auto it = std::lower_bound(keys.begin(), keys.end(), key);
  if (it == keys.end() || *it != key)
    return false;

  const size_t idx = static_cast<size_t>(it - keys.begin());
  if (idx >= cache.charmm_sparse_sigma().size() ||
      idx >= cache.charmm_sparse_epsilon().size())
    return false;

  sigma = cache.charmm_sparse_sigma()[idx];
  epsilon = cache.charmm_sparse_epsilon()[idx];
  return true;
}

inline void pair_params(const cache_sigma_epsilon &cache,
                        const Multi_MOL_parameters &rec_parameters,
                        const Multi_MOL_parameters &lg_parameters,
                        size_t rec_pos, size_t lg_pos, double &sigma,
                        double &epsilon) {
  if (lookup_pair_override(cache, rec_parameters, lg_parameters, rec_pos,
                           lg_pos, sigma, epsilon)) {
    return;
  }

  sigma = 0.5 * (rec_parameters[rec_pos].get_sigma() +
                 lg_parameters[lg_pos].get_sigma());
  epsilon = std::sqrt(rec_parameters[rec_pos].get_epsilon() *
                      lg_parameters[lg_pos].get_epsilon());
}

inline void pair_params_block(const cache_sigma_epsilon &cache,
                              const Multi_MOL_parameters &rec_parameters,
                              const Multi_MOL_parameters &lg_parameters,
                              size_t rec_pos_begin, size_t lane_count,
                              size_t lg_pos, double *sigma,
                              double *epsilon) {
  for (size_t lane = 0; lane < lane_count; ++lane) {
    pair_params(cache, rec_parameters, lg_parameters, rec_pos_begin + lane,
                lg_pos, sigma[lane], epsilon[lane]);
  }
}

inline double lj_energy(double sigma, double epsilon, double distance_nm) {
  if (distance_nm <= 0.0 || sigma <= 0.0 || epsilon == 0.0)
    return 0.0;

  const double sr = sigma / distance_nm;
  const double sr2 = sr * sr;
  const double sr6 = sr2 * sr2 * sr2;
  return 4.0 * epsilon * (sr6 * sr6 - sr6);
}

inline double lj_force_prefactor(double sigma, double epsilon,
                                 double distance_nm) {
  if (distance_nm <= 0.0 || sigma <= 0.0 || epsilon == 0.0)
    return 0.0;

  const double sr = sigma / distance_nm;
  const double sr2 = sr * sr;
  const double sr6 = sr2 * sr2 * sr2;
  const double sr12 = sr6 * sr6;
  const double inv_r2 = 1.0 / (distance_nm * distance_nm);
  return 4.0 * epsilon * (12.0 * sr12 - 6.0 * sr6) * inv_r2;
}

} // namespace charmm

#endif
