#ifndef MARTINI_NONBONDED_H
#define MARTINI_NONBONDED_H

#include "cache_sigma_epsilon.h"
#include "force.h"
#include <cmath>
#include <limits>
#include <string>

namespace martini {

constexpr double kCoulomb = 138.935457839;
constexpr double kDefaultCutoffNm = 1.1;

struct ElectrostaticConfig {
  double cutoff_nm{kDefaultCutoffNm};
  double epsilon_r{15.0};
  double epsilon_rf{std::numeric_limits<double>::infinity()};
  double k_rf{0.0};
  double c_rf{0.0};
};

inline ElectrostaticConfig electrostatic_config(const arguments &Arguments) {
  ElectrostaticConfig cfg;
  cfg.cutoff_nm = Arguments.distance_energy_cutoff > 0.0
                      ? Arguments.distance_energy_cutoff
                      : kDefaultCutoffNm;
  cfg.epsilon_r = Arguments.epsilon_in > 0.0 ? Arguments.epsilon_in : 15.0;

  if (std::isinf(cfg.epsilon_rf)) {
    cfg.k_rf = 0.5 / cube(cfg.cutoff_nm);
    cfg.c_rf = 1.5 / cfg.cutoff_nm;
  } else {
    const double denom = 2.0 * cfg.epsilon_rf + cfg.epsilon_r;
    cfg.k_rf = (cfg.epsilon_rf - cfg.epsilon_r) /
               (denom * cube(cfg.cutoff_nm));
    cfg.c_rf = (3.0 * cfg.epsilon_rf) / (denom * cfg.cutoff_nm);
  }

  return cfg;
}

inline void pair_params(const cache_sigma_epsilon &CachedData,
                        const Multi_MOL_parameters &rec_parameters,
                        const Multi_MOL_parameters &lg_parameters,
                        size_t rec_pos, size_t lg_pos, double &sigma_ij,
                        double &epsilon_ij) {
  sigma_ij = -1.0;
  epsilon_ij = -1.0;

  if (CachedData.has_martini_pair_table()) {
    const int rec_type = rec_parameters.get_type_id(rec_pos);
    const int lg_type = lg_parameters.get_type_id(lg_pos);
    if (rec_type >= 0 && lg_type >= 0) {
      const size_t idx = static_cast<size_t>(lg_type) *
                             static_cast<size_t>(CachedData.martini_types()) +
                         static_cast<size_t>(rec_type);
      sigma_ij = CachedData.martini_pair_sigma[idx];
      epsilon_ij = CachedData.martini_pair_epsilon[idx];
    }
  }

  if (sigma_ij < 0.0) {
    sigma_ij =
        0.5 * (lg_parameters.sigma[lg_pos] + rec_parameters.sigma[rec_pos]);
  }
  if (epsilon_ij < 0.0) {
    epsilon_ij = std::sqrt(lg_parameters.epsilon[lg_pos] *
                           rec_parameters.epsilon[rec_pos]);
  }
}

// GROMACS/Martini [ nonbond_params ] stores sigma and epsilon for the
// canonical 12-6 form V = 4 * epsilon * [ (sigma/r)^12 - (sigma/r)^6 ].
inline double lj_energy(double sigma_ij, double epsilon_ij, double d_nm) {
  const double sr6 = c6(sigma_ij / d_nm);
  return 4.0 * epsilon_ij * (square(sr6) - sr6);
}

// Returns the factor F/r_vec_nm for vector force accumulation:
// F_vec = prefactor * r_vec_nm
inline double lj_force_prefactor(double sigma_ij, double epsilon_ij,
                                 double d_nm) {
  const double sr = sigma_ij / d_nm;
  const double sr2 = square(sr);
  const double sr6 = sr2 * sr2 * sr2;
  const double sr12 = sr6 * sr6;
  return 24.0 * epsilon_ij * (2.0 * sr12 - sr6) / square(d_nm);
}

inline double electro_energy(const ElectrostaticConfig &cfg, double qi, double qj,
                             double d_nm) {
  const double pref = kCoulomb * qi * qj / cfg.epsilon_r;
  return pref * (1.0 / d_nm + cfg.k_rf * square(d_nm) - cfg.c_rf);
}

inline double electro_energy(const arguments &Arguments, double qi, double qj,
                             double d_nm) {
  return electro_energy(electrostatic_config(Arguments), qi, qj, d_nm);
}

// Returns the factor F/r_vec_nm for vector force accumulation:
// F_vec = prefactor * r_vec_nm
inline double electro_force_prefactor(const ElectrostaticConfig &cfg, double qi,
                                      double qj, double d_nm) {
  const double pref = kCoulomb * qi * qj / cfg.epsilon_r;
  return pref * (1.0 / cube(d_nm) - 2.0 * cfg.k_rf);
}

inline double electro_force_prefactor(const arguments &Arguments, double qi,
                                      double qj, double d_nm) {
  return electro_force_prefactor(electrostatic_config(Arguments), qi, qj, d_nm);
}

inline bool within_cutoff(const ElectrostaticConfig &cfg, double d_nm) {
  return d_nm <= cfg.cutoff_nm;
}

inline bool within_cutoff(const arguments &Arguments, double d_nm) {
  return within_cutoff(electrostatic_config(Arguments), d_nm);
}

inline bool excluded_pair(const arguments &Arguments,
                          const vector<mol_info> &info_rec,
                          const vector<mol_info> &info_lg, size_t rec_pos,
                          size_t lg_pos, double d_nm, double sigma_ij) {
  if (!Arguments.self_int)
    return false;
  if (d_nm < sigma_ij)
    return true;
  return Arguments.flag_interchain_energy &&
         info_lg[lg_pos].get_chain() == info_rec[rec_pos].get_chain();
}

inline std::string electrostatics_summary(const arguments &Arguments) {
  const auto cfg = electrostatic_config(Arguments);
  return "reaction-field (eps_r=" + std::to_string(cfg.epsilon_r) +
         ", eps_rf=inf, rc=" + std::to_string(cfg.cutoff_nm) + " nm)";
}

} // namespace martini

#endif
