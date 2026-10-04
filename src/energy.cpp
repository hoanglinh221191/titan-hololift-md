#include "titan_topology_energy.h"
#include "gromos_nonbonded.h"
#include "cache_sigma_epsilon.h"
#include "charmm_nonbonded.h"
#include "force.h"
#include "functions.h"
#include "martini_nonbonded.h"
#include "miscellaneous.h"
#include "titan_energy_validation.h"
#include "titan_electrostatics.h"
#include "titan_executor.h"
#include "uti_func.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <set>
#include <string>
#include <string_view>
using std::string_view;
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __AVX__
#include "avx_math.h"
#include <immintrin.h>
#else
#warning AVX is not available.Code will not compile!
#endif
#include <functional>

// AVX is not available.

using namespace std;

namespace {
inline void zero_cached_pair_if_present(cache_sigma_epsilon &CachedData,
                                        size_t pos) {
  if (!CachedData.elec_ener.empty())
    CachedData.set_elec_vdw(pos, 0., 0.);
}
} // namespace

inline int add_energy(const cache_sigma_epsilon &CachedData, size_t i,
                      double &electro_energy, double &van_der_waals_energy,
                      const Multi_MOL_parameters &rec_parameters,
                      const Multi_MOL_parameters &lg_parameters, const MOL &rec,
                      const MOL &lg, size_t snap, const arguments &Arguments,
                      const vector<mol_info> &info_rec,
                      const vector<mol_info> &info_lg, size_t rec_pos = 0,
                      size_t lg_pos = 0) {
  const bool topology_bound = titan_nb::topology_energy_required(CachedData, Arguments);
  if (topology_bound && i >= CachedData.elec_ener.size()) {
    const double dx=0.1*(rec.pos.x[rec_pos]-lg.pos.x[lg_pos]);
    const double dy=0.1*(rec.pos.y[rec_pos]-lg.pos.y[lg_pos]);
    const double dz=0.1*(rec.pos.z[rec_pos]-lg.pos.z[lg_pos]);
    const auto pair=titan_nb::evaluate_pair(CachedData,Arguments,info_rec.at(rec_pos),info_lg.at(lg_pos),
                                           std::sqrt(dx*dx+dy*dy+dz*dz),CachedData.cutoff_dis);
    const double factor=Arguments.self_int?0.5:1.0;
    electro_energy+=factor*pair.elec;van_der_waals_energy+=factor*pair.vdw;
    return 0;
  }
  if (i < CachedData.elec_ener.size()) {
    electro_energy += CachedData.elec_ener[i];
    van_der_waals_energy += CachedData.vdw_ener[i];

    if (CachedData.vdw_ener[i] > 500.0) {
      // Intentionally removed I/O inside hot loop
    }
  } else {
    if (rec_pos >= rec.size() || lg_pos >= lg.size())
      return 1;
    if (Arguments.self_int && rec_pos == lg_pos)
      return 1;
    double dx{rec.pos.x[rec_pos] - lg.pos.x[lg_pos]},
        dy{rec.pos.y[rec_pos] - lg.pos.y[lg_pos]},
        dz{rec.pos.z[rec_pos] - lg.pos.z[lg_pos]},
        charge{138.935457839 * rec_parameters[rec_pos].get_charge() *
               lg_parameters[lg_pos].get_charge() / Arguments.epsilon_in},
        d{0.1 * sqrt(square(dx) + square(dy) + square(dz))};

    const bool is_martini =
        CachedData.force_field_name.find("MARTINI") != string::npos;

    if (CachedData.cutoff_dis > 0. && d > CachedData.cutoff_dis)
      return 1;

    if (!is_martini)
      electro_energy += charge / d;

    double vdw_initial = van_der_waals_energy;
    if (CachedData.force_field_name.find("CHARMM") != string::npos) {
      double sigma_ij{}, epsilon_ij{};
      charmm::pair_params(CachedData, rec_parameters, lg_parameters, rec_pos,
                          lg_pos, sigma_ij, epsilon_ij);

      if (Arguments.self_int == true &&
          ((sigma_ij > 0.0 && d < sigma_ij) ||
           (Arguments.flag_interchain_energy == true &&
            info_lg[lg_pos].get_chain() == info_rec[rec_pos].get_chain())))
        return 1;

      van_der_waals_energy += charmm::lj_energy(sigma_ij, epsilon_ij, d);
    } else if (CachedData.force_field_name.find("AMBER") != string::npos) {
      double temp_vdw{4.0 * sqrt(rec_parameters[rec_pos].get_epsilon() *
                                 lg_parameters[lg_pos].get_epsilon())},
          c_6{c6(0.5 *
                 (lg_parameters[lg_pos].get_sigma() +
                  rec_parameters[rec_pos].get_sigma()) /
                 d)},
          c_12{square(c_6)}, sd{square(1. / d)},
          temp_par_vdw{temp_vdw * (12. * c_12 * sd - 6. * c_6 * sd)};

      if (lg_parameters[lg_pos].get_sigma() < 0. ||
          rec_parameters[rec_pos].get_sigma() < 0.)
        c_6 = 0.000;

      if (Arguments.self_int == true &&
          (c_6 > 1. ||
           (Arguments.flag_interchain_energy == true &&
            info_lg[lg_pos].get_chain() == info_rec[rec_pos].get_chain())))
        return 1;

      van_der_waals_energy += temp_vdw * (c_12 - c_6);
    } else if (CachedData.force_field_name.find("OPLS") != string::npos) {
      double temp_vdw{4.0 * sqrt(rec_parameters[rec_pos].get_epsilon() *
                                 lg_parameters[lg_pos].get_epsilon())},
          c_6{c6(sqrt(lg_parameters[lg_pos].get_sigma() *
                      rec_parameters[rec_pos].get_sigma()) /
                 d)},
          c_12{square(c_6)}, sd{square(1. / d)},
          temp_par_vdw{temp_vdw * (12. * c_12 * sd - 6. * c_6 * sd)};

      if (lg_parameters[lg_pos].get_sigma() < 0. ||
          rec_parameters[rec_pos].get_sigma() < 0.)
        c_6 = 0.000;

      if (Arguments.self_int == true &&
          (c_6 > 1. ||
           (Arguments.flag_interchain_energy == true &&
            info_lg[lg_pos].get_chain() == info_rec[rec_pos].get_chain())))
        return 1;

      van_der_waals_energy += temp_vdw * (c_12 - c_6);
    } else if (CachedData.force_field_name.find("MARTINI") != string::npos) {
      const auto elec_cfg = martini::electrostatic_config(Arguments);
      double sigma_ij{}, epsilon_ij{};
      martini::pair_params(CachedData, rec_parameters, lg_parameters, rec_pos,
                           lg_pos, sigma_ij, epsilon_ij);
      if (!martini::within_cutoff(elec_cfg, d) ||
          martini::excluded_pair(Arguments, info_rec, info_lg, rec_pos, lg_pos,
                                 d, sigma_ij))
        return 1;

      electro_energy += martini::electro_energy(
          elec_cfg, rec_parameters[rec_pos].get_charge(),
          lg_parameters[lg_pos].get_charge(), d);
      van_der_waals_energy += martini::lj_energy(sigma_ij, epsilon_ij, d);
      // NOTE: electro_energy already accumulated above - do NOT add it here.
    } else {
      double sd{1. / square(d)}, temp_elec{sd * charge / Arguments.epsilon_in},
          c_6{c6(d)}, c_12{square(c_6)};
      double sigma{}, epsilon{};
      gromos::pair_params(CachedData, rec_parameters, lg_parameters, rec_pos, lg_pos, sigma, epsilon);
      double temp_vdw{12. * sd * epsilon / c_12 - 6. * sd * sigma / c_6};

      van_der_waals_energy += epsilon / c_12 - sigma / c_6;
    }

    if (van_der_waals_energy - vdw_initial > 500.0) {
      // Intentionally removed I/O inside hot loop
    }
  }

  return 0;
}


int cal_energy_amber_screening(const MOL &rec, const MOL &lg,
                               const vector<mol_info> &info_rec,
                               const vector<mol_info> &info_lg,
                               cache_sigma_epsilon &CachedData, size_t begin,
                               size_t end, cache_energy &Energy,
                               size_t num_snap, const arguments &Arguments,
                               const Multi_MOL_parameters &rec_parameters,
                               const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, nullptr, Arguments);
    return 0;
  }
  constexpr double kScreeningMinDistance =
      titan_energy_validation::kMinimumPairDistanceNm;
  const double epsilon_in{Arguments.epsilon_in};
  const double lambda_debye{
      titan_electrostatics::debye_length_nm(Arguments.salt_concentration)};
  const bool is_charmm =
      Arguments.force_f.find("CHARMM") != string::npos;
  Energy.set_energy(0., 0.);

  for (size_t j{begin}; j < end; ++j) {
    const double temp_charge{138.935457839 * lg_parameters[j].get_charge()};
    const size_t rec_size{rec.size()};
    for (size_t i{}; i < rec_size; ++i) {
      const double d{0.1 * sqrt(square(rec.pos.x[i] - lg.pos.x[j]) +
                                square(rec.pos.y[i] - lg.pos.y[j]) +
                                square(rec.pos.z[i] - lg.pos.z[j]))};

      if (!std::isfinite(d) || d <= kScreeningMinDistance) {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }

      if (Arguments.distance_energy_cutoff > 0. &&
          d > Arguments.distance_energy_cutoff) [[unlikely]] {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }

      double sigma_self = 0.5 *
                          (lg_parameters[j].get_sigma() +
                           rec_parameters[i].get_sigma());
      double v_val = 0.0;
      if (is_charmm) {
        double sigma_ij{}, epsilon_ij{};
        charmm::pair_params(CachedData, rec_parameters, lg_parameters, i, j,
                            sigma_ij, epsilon_ij);
        sigma_self = sigma_ij;
        v_val = charmm::lj_energy(sigma_ij, epsilon_ij, d);
      } else {
        const double temp_epsilon_lg{16. * lg_parameters[j].get_epsilon()};
        const double temp_sigma{lg_parameters[j].get_sigma()};
        const double c_6{
            c6(0.5 * (temp_sigma + rec_parameters[i].get_sigma()) / d)};
        if (temp_sigma < 0. || rec_parameters[i].get_sigma() < 0.)
          v_val = sqrt(temp_epsilon_lg * rec_parameters[i].get_epsilon()) *
                  (square(c_6) - 0.000);
        else
          v_val = sqrt(temp_epsilon_lg * rec_parameters[i].get_epsilon()) *
                  (square(c_6) - c_6);
      }

      if (Arguments.self_int == true &&
          ((sigma_self > 0.0 && d < sigma_self) ||
           (Arguments.flag_interchain_energy == true &&
            info_lg[j].get_chain() == info_rec[i].get_chain()))) {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }

      const double e_val = temp_charge * rec_parameters[i].get_charge() *
                           exp(-d / lambda_debye) / d / epsilon_in;
      if (CachedData.elec_ener.empty()) {
        Energy.elec_ener += e_val;
        Energy.vdw_ener += v_val;
      } else {
        const size_t pos{rec.size() * j + i};
        CachedData.set_elec_vdw(pos, e_val, v_val);
        Energy += CachedData.get_energy(pos);
      }
    }
  }
  return 0;
}

void cal_energy_amber(const MOL &rec, const MOL &lg,
                      const vector<mol_info> &info_rec,
                      const vector<mol_info> &info_lg,
                      cache_sigma_epsilon &CachedData, size_t begin, size_t end,
                      cache_energy &Energy, size_t num_snap,
                      const arguments &Arguments,
                      const Multi_MOL_parameters &rec_parameters,
                      const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, nullptr, Arguments);
    return;
  }
  Energy.set_energy(0., 0.);
  const bool is_charmm =
      Arguments.force_f.find("CHARMM") != string::npos;
  for (size_t j{begin}; j < end; ++j) {
    const double pos_x{lg.pos.x[j]}, pos_y{lg.pos.y[j]}, pos_z{lg.pos.z[j]};
    const size_t rec_size{rec.size()};
    for (size_t i{}; i < rec_size; ++i) {
      if (Arguments.self_int && i == j) {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }
      double d{0.1 * sqrt(square(rec.pos.x[i] - pos_x) +
                          square(rec.pos.y[i] - pos_y) +
                          square(rec.pos.z[i] - pos_z))};

      if (Arguments.distance_energy_cutoff > 0. &&
          d > Arguments.distance_energy_cutoff) [[unlikely]] {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }

      double sigma_self = 0.5 *
                          (lg_parameters[j].get_sigma() +
                           rec_parameters[i].get_sigma());
      double v_val = 0.0;
      if (is_charmm) {
        double sigma_ij{}, epsilon_ij{};
        charmm::pair_params(CachedData, rec_parameters, lg_parameters, i, j,
                            sigma_ij, epsilon_ij);
        sigma_self = sigma_ij;
        v_val = charmm::lj_energy(sigma_ij, epsilon_ij, d);
      } else {
        double temp_epsilon_lg{16. * lg_parameters[j].get_epsilon()};
        double c_6{
            c6(0.5 *
               (lg_parameters[j].get_sigma() + rec_parameters[i].get_sigma()) /
               d)};
        if (lg_parameters[j].get_sigma() < 0. ||
            rec_parameters[i].get_sigma() < 0.)
          v_val = sqrt(temp_epsilon_lg * rec_parameters[i].get_epsilon()) *
                  (square(c_6) - 0.000);
        else
          v_val = sqrt(temp_epsilon_lg * rec_parameters[i].get_epsilon()) *
                  (square(c_6) - c_6);
      }

      if (Arguments.self_int == true &&
          ((sigma_self > 0.0 && d < sigma_self) ||
           (Arguments.flag_interchain_energy == true &&
            info_lg[j].get_chain() == info_rec[i].get_chain()))) {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }

      double temp_charge{138.935457839 * lg_parameters[j].get_charge()};
      double e_val = temp_charge * rec_parameters[i].get_charge() / d /
                     Arguments.epsilon_in;

      if (CachedData.elec_ener.empty()) {
        Energy.elec_ener += e_val;
        Energy.vdw_ener += v_val;
      } else {
        size_t pos{rec.size() * j + i};
        CachedData.set_elec_vdw(pos, e_val, v_val);
        Energy += CachedData.get_energy(pos);
      }
    }
  }
}
void cal_energy_opls(const MOL &rec, const MOL &lg,
                     const vector<mol_info> &info_rec,
                     const vector<mol_info> &info_lg,
                     cache_sigma_epsilon &CachedData, size_t begin, size_t end,
                     cache_energy &Energy, size_t num_snap,
                     const arguments &Arguments,
                     const Multi_MOL_parameters &rec_parameters,
                     const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, nullptr, Arguments);
    return;
  }
  Energy.set_energy(0., 0.);
  for (size_t j{begin}; j < end; ++j) {
    double temp_epsilon_lg{16. * lg_parameters[j].get_epsilon()},
        temp_charge{138.935457839 * lg_parameters[j].get_charge() /
                    Arguments.epsilon_in},
        temp_sigma{lg_parameters[j].get_sigma()}, pos_x{lg.pos.x[j]},
        pos_y{lg.pos.y[j]}, pos_z{lg.pos.z[j]};
    size_t rec_size{rec.size()};

    for (size_t i{}; i < rec_size; ++i) {
      if (Arguments.self_int && i == j) {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }
      double d{0.1 * sqrt(square(rec.pos.x[i] - pos_x) +
                          square(rec.pos.y[i] - pos_y) +
                          square(rec.pos.z[i] - pos_z))},
          c_6{c6(sqrt(temp_sigma * rec_parameters[i].get_sigma()) / d)};

      if (Arguments.self_int == true &&
          (c_6 > 1. || (Arguments.flag_interchain_energy == true &&
                        info_lg[j].get_chain() == info_rec[i].get_chain()))) {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }

      if (Arguments.distance_energy_cutoff > 0. &&
          d > Arguments.distance_energy_cutoff) [[unlikely]] {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }

      if (CachedData.elec_ener.empty()) {
        Energy.elec_ener += temp_charge * rec_parameters[i].get_charge() / d;
        Energy.vdw_ener +=
            sqrt(temp_epsilon_lg * rec_parameters[i].get_epsilon()) *
            (square(c_6) - c_6);
      } else {
        size_t pos{rec.size() * j + i};
        CachedData.set_elec_vdw(
            pos, temp_charge * rec_parameters[i].get_charge() / d,
            sqrt(rec_parameters[i].get_epsilon() * temp_epsilon_lg) *
                (square(c_6) - c_6));
        Energy += CachedData.get_energy(pos);
      }
    }
  }
}

#ifdef __AVX__

#include "avx_math.h"

#endif

void cal_energy_gromos(const MOL &rec, const MOL &lg,
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
  for (size_t j{begin}; j < end; ++j) {
    double temp_charge{138.935457839 * lg_parameters[j].get_charge() /
                       Arguments.epsilon_in},
        temp_epsilon{lg_parameters[j].get_epsilon()},
        temp_sigma{lg_parameters[j].get_sigma()}, pos_x{lg.pos.x[j]},
        pos_y{lg.pos.y[j]}, pos_z{lg.pos.z[j]};
    size_t rec_size{rec.size()};
    for (size_t i{}; i < rec_size; ++i) {
      if (Arguments.self_int && i == j) {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }
      double d{0.1 * sqrt(square(rec.pos.x[i] - pos_x) +
                          square(rec.pos.y[i] - pos_y) +
                          square(rec.pos.z[i] - pos_z))},
          c_6{c6(d)};

      if (Arguments.distance_energy_cutoff > 0. &&
          d > Arguments.distance_energy_cutoff) [[unlikely]] {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }

      if (Arguments.self_int == true &&
          ((i == j) ||
           (Arguments.flag_interchain_energy == true &&
                      info_lg[j].get_chain() == info_rec[i].get_chain()))) {
        zero_cached_pair_if_present(CachedData, rec.size() * j + i);
        continue;
      }

      double pair_c6{}, pair_c12{};
      gromos::pair_params(CachedData, rec_parameters, lg_parameters, i, j, pair_c6, pair_c12);
      if (CachedData.elec_ener.empty()) {
        Energy.elec_ener += temp_charge * rec_parameters[i].get_charge() / d;
        Energy.vdw_ener +=
            pair_c12 / square(c_6) -
            pair_c6 / c_6;
      } else {
        size_t pos{rec.size() * j + i};
        CachedData.set_elec_vdw(
            pos, temp_charge * rec_parameters[i].get_charge() / d,
            pair_c12 / square(c_6) -
                pair_c6 / c_6);
        Energy += CachedData.get_energy(pos);
      }
    }
  }
}

// MARTINI: use the topology-derived pair matrix and the standard GROMACS
// 12-6 form V = 4*epsilon[(sigma/r)^12 - (sigma/r)^6]. Electrostatics use the
// Martini reaction-field form with rc = -denfo or the Martini default 1.1 nm.
void cal_energy_martini(const MOL &rec, const MOL &lg,
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
  const auto elec_cfg = martini::electrostatic_config(Arguments);
  for (size_t j{begin}; j < end; ++j) {
    const double pos_x{lg.pos.x[j]}, pos_y{lg.pos.y[j]}, pos_z{lg.pos.z[j]};
    const size_t rec_size{rec.size()};
    for (size_t i{}; i < rec_size; ++i) {
      double d{0.1 * sqrt(square(rec.pos.x[i] - pos_x) +
                          square(rec.pos.y[i] - pos_y) +
                          square(rec.pos.z[i] - pos_z))};
      if (d <= titan_energy_validation::kMinimumPairDistanceNm) {
        if (!CachedData.elec_ener.empty()) {
          size_t pos{rec.size() * j + i};
          CachedData.set_elec_vdw(pos, 0., 0.);
        }
        continue; // Guard: beads at same position, avoid 0/0 NaN
      }
      double sigma_ij{}, epsilon_ij{};
      martini::pair_params(CachedData, rec_parameters, lg_parameters, i, j,
                           sigma_ij, epsilon_ij);

      if (!martini::within_cutoff(elec_cfg, d)) [[unlikely]] {
        if (!CachedData.elec_ener.empty()) {
          size_t pos{rec.size() * j + i};
          CachedData.set_elec_vdw(pos, 0., 0.);
        }
        continue;
      }

      if (martini::excluded_pair(Arguments, info_rec, info_lg, i, j, d,
                                 sigma_ij)) {
        if (!CachedData.elec_ener.empty()) {
          size_t pos{rec.size() * j + i};
          CachedData.set_elec_vdw(pos, 0., 0.);
        }
        continue;
      }

      const double e_val = martini::electro_energy(
          elec_cfg, rec_parameters[i].get_charge(),
          lg_parameters[j].get_charge(), d);
      const double v_val = martini::lj_energy(sigma_ij, epsilon_ij, d);

      if (CachedData.elec_ener.empty()) {
        Energy.elec_ener += e_val;
        Energy.vdw_ener += v_val;
      } else {
        size_t pos{rec.size() * j + i};
        CachedData.set_elec_vdw(pos, e_val, v_val);
        Energy += CachedData.get_energy(pos);
      }
    }
  }
}
