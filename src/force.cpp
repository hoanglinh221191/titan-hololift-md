#include "titan_topology_energy.h"
#include "gromos_nonbonded.h"
#include "Multi_MOL.h"
#include "cache_sigma_epsilon.h"
#include "charmm_nonbonded.h"
#include "distance_v2.h"
#include "folder.h"
#include "info.h"
#include "martini_nonbonded.h"
#include "miscellaneous.h"
#include "titan_energy_validation.h"
#include "titan_electrostatics.h"
#include "titan_executor.h"
#include "uti_func.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace std;

namespace {
inline void zero_cached_pair_if_present(cache_sigma_epsilon &cached_data,
                                        size_t pos) {
  if (!cached_data.elec_ener.empty())
    cached_data.set_elec_vdw(pos, 0.0, 0.0);
  if (cached_data.get_force(pos).has_value())
    cached_data.set_force(pos, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}
} // namespace

#ifdef __AVX__
#include "avx_math.h"
#include <immintrin.h>

#endif

bool check_in_energy_list(vector<data_out> &lg_ener_inter, string chain,
                          size_t resnr);
void print_force_map(size_t nt, vector<vector<data_out>> &data,
                     fstream &force_file, bool flag_all_mode,
                     const arguments &Arguments, vector<energy_res> &Energy_res,
                     size_t num_snap, info &info_output);

int force_energy_add(const cache_sigma_epsilon &CachedData, size_t i,
                     double &electro_energy, double &van_der_waals_energy,
                     force &force_elec, force &force_vdw,
                     const Multi_MOL_parameters & rec_parameters,
                     const Multi_MOL_parameters & lg_parameters,
                     const Multi_MOL &rec, const Multi_MOL &lg, size_t snap,
                     const arguments &Arguments, size_t rec_pos = 0,
                     size_t lg_pos = 0) {
  const bool topology_bound = titan_nb::topology_energy_required(CachedData, Arguments);
  if (topology_bound && CachedData.elec_ener.empty()) {
    for (std::size_t lane=0;lane<1;++lane) {
      const std::size_t j=lg_pos+lane;
      const double dx=0.1*(rec.m_mol[snap].pos.x[rec_pos]-lg.m_mol[snap].pos.x[j]);
      const double dy=0.1*(rec.m_mol[snap].pos.y[rec_pos]-lg.m_mol[snap].pos.y[j]);
      const double dz=0.1*(rec.m_mol[snap].pos.z[rec_pos]-lg.m_mol[snap].pos.z[j]);
      const auto pair=titan_nb::evaluate_pair(CachedData,Arguments,rec.info.at(rec_pos),lg.info.at(j),
                                             std::sqrt(dx*dx+dy*dy+dz*dz),CachedData.cutoff_dis);
      const double factor=Arguments.self_int?0.5:1.0;
      electro_energy+=factor*pair.elec;van_der_waals_energy+=factor*pair.vdw;
      force_elec+=force(pair.force_elec_prefactor*dx,pair.force_elec_prefactor*dy,pair.force_elec_prefactor*dz);
      force_vdw+=force(pair.force_vdw_prefactor*dx,pair.force_vdw_prefactor*dy,pair.force_vdw_prefactor*dz);
    }
    return 0;
  }
  if (i < CachedData.elec_ener.size()) {
    electro_energy += CachedData.elec_ener[i];
    van_der_waals_energy += CachedData.vdw_ener[i];

    if (CachedData.has_forces()) {
      auto cf = *CachedData.get_force(i);
      force_elec += cf.forces[0];
      force_vdw += cf.forces[1];
    }
  } else {
    const double dx{rec.m_mol[snap].pos.x[rec_pos] -
                    lg.m_mol[snap].pos.x[lg_pos]},
        dy{rec.m_mol[snap].pos.y[rec_pos] -
           lg.m_mol[snap].pos.y[lg_pos]},
        dz{rec.m_mol[snap].pos.z[rec_pos] -
           lg.m_mol[snap].pos.z[lg_pos]},
        charge_prefactor{138.935457839 *
                         rec_parameters[rec_pos].get_charge() *
                         lg_parameters[lg_pos].get_charge() /
                         std::max(1.0e-12, Arguments.epsilon_in)},
        d{0.1 * sqrt(square(dx) + square(dy) + square(dz))};
    const bool is_martini =
        CachedData.force_field_name.find("MARTINI") != string::npos;
    const bool screened =
        Arguments.flag_elec_screening &&
        (CachedData.force_field_name.find("AMBER") != string::npos ||
         CachedData.force_field_name.find("CHARMM") != string::npos);

    if (!std::isfinite(d) ||
        d <= titan_energy_validation::kMinimumPairDistanceNm ||
        (CachedData.cutoff_dis > 0. && d > CachedData.cutoff_dis))
      return 1;

    double screening = 1.0;
    double lambda_debye = 0.0;
    if (screened) {
      lambda_debye = titan_electrostatics::debye_length_nm(
          Arguments.salt_concentration);
      screening = std::exp(-d / lambda_debye);
    }
    const double electro_pair = charge_prefactor * screening / d;
    const double inv_d = 1.0 / d;
    const double electro_force_prefactor =
        charge_prefactor * screening *
        (inv_d * inv_d * inv_d +
         (screened ? inv_d * inv_d / lambda_debye : 0.0));

    if (CachedData.force_field_name.find("CHARMM") != string::npos) {
      double sigma_ij{}, epsilon_ij{};
      charmm::pair_params(CachedData, rec_parameters, lg_parameters, rec_pos,
                          lg_pos, sigma_ij, epsilon_ij);
      if (Arguments.self_int == true &&
          ((sigma_ij > 0.0 && d < sigma_ij) ||
           (Arguments.flag_interchain_energy == true &&
            lg.info[lg_pos].get_chain() == rec.info[rec_pos].get_chain())))
        return 1;
      electro_energy += electro_pair;
      van_der_waals_energy += charmm::lj_energy(sigma_ij, epsilon_ij, d);
      if (Arguments.flag_cal_force == true) {
        const double temp_par_vdw =
            charmm::lj_force_prefactor(sigma_ij, epsilon_ij, d);
        force_elec.add(0.1 * electro_force_prefactor * dx,
                       0.1 * electro_force_prefactor * dy,
                       0.1 * electro_force_prefactor * dz);
        force_vdw.add(0.1 * temp_par_vdw * dx, 0.1 * temp_par_vdw * dy,
                      0.1 * temp_par_vdw * dz);
      }
    } else if (CachedData.force_field_name.find("AMBER") != string::npos) {
      double temp_vdw{4.0 * sqrt(rec_parameters[rec_pos].get_epsilon() *
                                 lg_parameters[lg_pos].get_epsilon())},
          c_6{c6(0.5 *
                 (lg_parameters[lg_pos].get_sigma() +
                  rec_parameters[rec_pos].get_sigma()) /
                 d)},
          c_12{square(c_6)};

      if (lg_parameters[lg_pos].get_sigma() < 0. ||
          rec_parameters[rec_pos].get_sigma() < 0.)
        c_6 = 0.000;

      if (Arguments.self_int == true &&
          (c_6 > 1. ||
           (Arguments.flag_interchain_energy == true &&
            lg.info[lg_pos].get_chain() == rec.info[rec_pos].get_chain())))
        return 1;
      electro_energy += electro_pair;
      van_der_waals_energy += temp_vdw * (c_12 - c_6);
      if (Arguments.flag_cal_force == true) {
        double sd{square(1. / d)},
            temp_par_vdw{temp_vdw * (12. * c_12 * sd - 6. * c_6 * sd)};
        force_elec.add(0.1 * electro_force_prefactor * dx,
                       0.1 * electro_force_prefactor * dy,
                       0.1 * electro_force_prefactor * dz);
        force_vdw.add(0.1 * temp_par_vdw * dx, 0.1 * temp_par_vdw * dy,
                      0.1 * temp_par_vdw * dz);
      }
    } else if (CachedData.force_field_name.find("OPLS") != string::npos) {
      double temp_vdw{4.0 * sqrt(rec_parameters[rec_pos].get_epsilon() *
                                 lg_parameters[lg_pos].get_epsilon())},
          c_6{c6(sqrt(lg_parameters[lg_pos].get_sigma() *
                      rec_parameters[rec_pos].get_sigma()) /
                 d)},
          c_12{square(c_6)};

      if (lg_parameters[lg_pos].get_sigma() < 0. ||
          rec_parameters[rec_pos].get_sigma() < 0.)
        c_6 = 0.000;

      if (Arguments.self_int == true &&
          (c_6 > 1. ||
           (Arguments.flag_interchain_energy == true &&
            lg.info[lg_pos].get_chain() == rec.info[rec_pos].get_chain())))
        return 1;

      electro_energy += electro_pair;
      van_der_waals_energy += temp_vdw * (c_12 - c_6);

      if (Arguments.flag_cal_force == true) {
        double sd{square(1. / d)},
            temp_par_vdw{temp_vdw * (12. * c_12 * sd - 6. * c_6 * sd)};
        force_elec.add(0.1 * electro_force_prefactor * dx,
                       0.1 * electro_force_prefactor * dy,
                       0.1 * electro_force_prefactor * dz);
        force_vdw.add(0.1 * temp_par_vdw * dx, 0.1 * temp_par_vdw * dy,
                      0.1 * temp_par_vdw * dz);
      }
    } else if (is_martini) {
      const auto elec_cfg = martini::electrostatic_config(Arguments);
      double sigma_ij{}, epsilon_ij{};
      martini::pair_params(CachedData, rec_parameters, lg_parameters, rec_pos,
                           lg_pos, sigma_ij, epsilon_ij);

      if (!martini::within_cutoff(elec_cfg, d) ||
          martini::excluded_pair(Arguments, rec.info, lg.info, rec_pos, lg_pos,
                                 d, sigma_ij))
        return 1;

      electro_energy += martini::electro_energy(
          elec_cfg, rec_parameters[rec_pos].get_charge(),
          lg_parameters[lg_pos].get_charge(), d);
      van_der_waals_energy += martini::lj_energy(sigma_ij, epsilon_ij, d);

      if (Arguments.flag_cal_force == true) {
        const double temp_elec = martini::electro_force_prefactor(
            elec_cfg, rec_parameters[rec_pos].get_charge(),
            lg_parameters[lg_pos].get_charge(), d);
        const double temp_vdw =
            martini::lj_force_prefactor(sigma_ij, epsilon_ij, d);
        force_elec.add(0.1 * temp_elec * dx, 0.1 * temp_elec * dy,
                       0.1 * temp_elec * dz);
        force_vdw.add(0.1 * temp_vdw * dx, 0.1 * temp_vdw * dy,
                      0.1 * temp_vdw * dz);
      }
    } else {
      if (Arguments.self_int == true &&
          ((&rec == &lg && rec_pos == lg_pos) ||
           (Arguments.flag_interchain_energy == true &&
            lg.info[lg_pos].get_chain() == rec.info[rec_pos].get_chain())))
        return 1;
      double sd{1. / square(d)}, c_6{c6(d)}, c_12{square(c_6)};
      double sigma{}, epsilon{};
      gromos::pair_params(CachedData,rec_parameters,lg_parameters,rec_pos,lg_pos,sigma,epsilon);
      double temp_vdw{12. * sd * epsilon / c_12 - 6. * sd * sigma / c_6};

      electro_energy += electro_pair;
      van_der_waals_energy += epsilon / c_12 - sigma / c_6;

      if (Arguments.flag_cal_force == true) {
        force_elec.add(0.1 * electro_force_prefactor * dx,
                       0.1 * electro_force_prefactor * dy,
                       0.1 * electro_force_prefactor * dz);
        force_vdw.add(0.1 * temp_vdw * dx, 0.1 * temp_vdw * dy,
                      0.1 * temp_vdw * dz);
      }
    }
  }
  return 0;
}

int helper_print_force(size_t nt, vector<vector<data_out>> &data_lg,
                       vector<vector<data_out>> &data_rec,
                       fstream &lg_ener_inter, fstream &rec_ener_inter,
                       bool flag_all_mode, const arguments &Arguments,
                       vector<energy_res> &Energy_res_mol1,
                       vector<energy_res> &Energy_res_mol2, size_t num_snap,
                       info &info_output) {
  print_force_map(nt, data_lg, lg_ener_inter, flag_all_mode, Arguments,
                  Energy_res_mol2, num_snap, info_output);
  print_force_map(nt, data_rec, rec_ener_inter, flag_all_mode, Arguments,
                  Energy_res_mol1, num_snap, info_output);
  return 0;
}

int wrapper_print_force(size_t nt, vector<vector<data_out>> &data_lg,
                        vector<vector<data_out>> &data_rec, bool flag_all_mode,
                        const arguments &Arguments,
                        vector<energy_res> &Energy_res_mol1,
                        vector<energy_res> &Energy_res_mol2, size_t num_snap,
                        info &info_output) {
  fstream lg_force_inter("force_lg.txt", ios::out | ios::app);
  // lg_ener_inter << "Atom" << setw(12) << "Electro." << setw(12) << "Van der
  // W." << '\n';
  fstream rec_force_inter("force_rec.txt", ios::out | ios::app);
  // rec_ener_inter << "Res" << setw(12) << "Electro." << setw(12) << "Van der
  // W." << '\n';
  /**********************************************************************************/

  if (nt >= 2) {
    thread print1(print_force_map, nt, ref(data_lg), ref(lg_force_inter),
                  flag_all_mode, ref(Arguments), ref(Energy_res_mol2), num_snap,
                  ref(info_output));
    thread print2(print_force_map, nt, ref(data_rec), ref(rec_force_inter),
                  flag_all_mode, ref(Arguments), ref(Energy_res_mol1), num_snap,
                  ref(info_output));
    print1.join();
    print2.join();
  } else {
    helper_print_force(nt, data_lg, data_rec, lg_force_inter, rec_force_inter,
                       flag_all_mode, Arguments, Energy_res_mol1,
                       Energy_res_mol2, num_snap, info_output);
  }

  return 0;
}

void cal_energy_force_amber(const MOL &rec, const MOL &lg,
                            const vector<mol_info> &info_rec,
                            const vector<mol_info> &info_lg,
                            cache_sigma_epsilon &CachedData, size_t begin,
                            size_t end, cache_energy &Energy,
                            cache_force &Force, size_t num_snap,
                            const arguments &Arguments,
                            const Multi_MOL_parameters & rec_parameters,
                            const Multi_MOL_parameters & lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, &Force, Arguments);
    return;
  }
  Energy.set_energy(0., 0.);
  Force.set_force(force(0., 0., 0.), force(0., 0., 0.));
  const bool is_charmm =
      Arguments.force_f.find("CHARMM") != string::npos;
  const double epsilon_in = std::max(1.0e-12, Arguments.epsilon_in);
  const bool screened = Arguments.flag_elec_screening;
  const double lambda_debye =
      screened ? titan_electrostatics::debye_length_nm(
                     Arguments.salt_concentration)
               : 0.0;

  for (size_t j{begin}; j < end; ++j) {
    double temp_charge{138.935457839 * lg_parameters[j].get_charge()};
    for (size_t i{}; i < rec.size(); ++i) {
      double dx{rec.pos.x[i] - lg.pos.x[j]},
          dy{rec.pos.y[i] - lg.pos.y[j]},
          dz{rec.pos.z[i] - lg.pos.z[j]};
      double d{0.1 * sqrt(square(dx) + square(dy) + square(dz))},
          charge_prefactor{temp_charge * rec_parameters[i].get_charge() /
                           epsilon_in};
      const size_t pos{rec.size() * j + i};
      if (!std::isfinite(d) ||
          d <= titan_energy_validation::kMinimumPairDistanceNm) {
        zero_cached_pair_if_present(CachedData, pos);
        continue;
      }
      const double screening = screened ? std::exp(-d / lambda_debye) : 1.0;
      const double charge = charge_prefactor * screening / d;
      const double inv_d = 1.0 / d;
      const double sd{inv_d * inv_d};
      const double temp_elec =
          charge_prefactor * screening *
          (inv_d * sd + (screened ? sd / lambda_debye : 0.0));

      double sigma_self = 0.5 *
                          (lg_parameters[j].get_sigma() +
                           rec_parameters[i].get_sigma());
      double v_val = 0.0;
      double temp_par_vdw = 0.0;
      if (is_charmm) {
        double sigma_ij{}, epsilon_ij{};
        charmm::pair_params(CachedData, rec_parameters, lg_parameters, i, j,
                            sigma_ij, epsilon_ij);
        sigma_self = sigma_ij;
        v_val = charmm::lj_energy(sigma_ij, epsilon_ij, d);
        temp_par_vdw = charmm::lj_force_prefactor(sigma_ij, epsilon_ij, d);
      } else {
        double temp_epsilon_lg{16. * lg_parameters[j].get_epsilon()};
        double temp_sigma_lg{lg_parameters[j].get_sigma()};
        double temp_vdw{
            sqrt(rec_parameters[i].get_epsilon() * temp_epsilon_lg)};
        double c_6{
            c6(0.5 * (temp_sigma_lg + rec_parameters[i].get_sigma()) / d)};
        double c_12{square(c_6)};
        if (temp_sigma_lg < 0. || rec_parameters[i].get_sigma() < 0.)
          c_6 = 0.000;
        temp_par_vdw = temp_vdw * (12. * c_12 * sd - 6. * c_6 * sd);
        v_val = temp_vdw * (c_12 - c_6);
      }

      if (Arguments.distance_energy_cutoff > 0. &&
          d > Arguments.distance_energy_cutoff) {
        zero_cached_pair_if_present(CachedData, pos);
        continue;
      }

      if (Arguments.self_int == true &&
          ((sigma_self > 0.0 && sigma_self > d) ||
           (Arguments.flag_interchain_energy == true &&
            info_lg[j].get_chain() == info_rec[i].get_chain()))) {
        zero_cached_pair_if_present(CachedData, pos);
        continue;
      }

      if (CachedData.elec_ener.empty()) {
        Energy += cache_energy(charge, v_val);
        Force += cache_force(0.1 * temp_elec * dx, 0.1 * temp_elec * dy,
                             0.1 * temp_elec * dz, 0.1 * temp_par_vdw * dx,
                             0.1 * temp_par_vdw * dy, 0.1 * temp_par_vdw * dz);
      } else {
        CachedData.set_elec_vdw(pos, charge, v_val);
        CachedData.set_force(pos, 0.1 * temp_elec * dx, 0.1 * temp_elec * dy,
                             0.1 * temp_elec * dz, 0.1 * temp_par_vdw * dx,
                             0.1 * temp_par_vdw * dy, 0.1 * temp_par_vdw * dz);

        Energy += CachedData.get_energy(pos);
        Force += *CachedData.get_force(pos);
      }
    }
  }
}

void cal_energy_force_opls(const MOL &rec, const MOL &lg,
                           const vector<mol_info> &info_rec,
                           const vector<mol_info> &info_lg,
                           cache_sigma_epsilon &CachedData, size_t begin,
                           size_t end, cache_energy &Energy, cache_force &Force,
                           size_t num_snap, const arguments &Arguments,
                           const Multi_MOL_parameters & rec_parameters,
                           const Multi_MOL_parameters & lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, &Force, Arguments);
    return;
  }
  Energy.set_energy(0., 0.);
  Force.set_force(force(0., 0., 0.), force(0., 0., 0.));
  const double epsilon_in = std::max(1.0e-12, Arguments.epsilon_in);

  for (size_t j{begin}; j < end; ++j) {
    double temp_epsilon_lg{16. * lg_parameters[j].get_epsilon()},
        temp_charge{138.935457839 * lg_parameters[j].get_charge()},
        temp_sigma_lg{lg_parameters[j].get_sigma()};
    for (size_t i{}; i < rec.size(); ++i) {
      double dx{rec.pos.x[i] - lg.pos.x[j]},
          dy{rec.pos.y[i] - lg.pos.y[j]},
          dz{rec.pos.z[i] - lg.pos.z[j]};
      double d{0.1 * sqrt(square(dx) + square(dy) + square(dz))},
          charge_prefactor{temp_charge * rec_parameters[i].get_charge() /
                           epsilon_in};
      const size_t pos{rec.size() * j + i};
      if (!std::isfinite(d) ||
          d <= titan_energy_validation::kMinimumPairDistanceNm) {
        zero_cached_pair_if_present(CachedData, pos);
        continue;
      }
      const double charge = charge_prefactor / d;
      double sd{square(1. / d)}, temp_elec{sd * charge},
          temp_vdw{sqrt(rec_parameters[i].get_epsilon() * temp_epsilon_lg)},
          c_6{c6(sqrt(temp_sigma_lg * rec_parameters[i].get_sigma()) / d)},
          c_12{square(c_6)},
          temp_par_vdw{temp_vdw * (12. * c_12 * sd - 6. * c_6 * sd)};

      if (Arguments.distance_energy_cutoff > 0. &&
          d > Arguments.distance_energy_cutoff) {
        zero_cached_pair_if_present(CachedData, pos);
        continue;
      }

      if (Arguments.self_int == true &&
          (sqrt(temp_sigma_lg * rec_parameters[i].get_sigma()) > d ||
           (Arguments.flag_interchain_energy == true &&
            info_lg[j].get_chain() == info_rec[i].get_chain()))) {
        zero_cached_pair_if_present(CachedData, pos);
        continue;
      }

      if (CachedData.elec_ener.empty()) {
        Energy += cache_energy(charge, temp_vdw * (c_12 - c_6));
        Force += cache_force(0.1 * temp_elec * dx, 0.1 * temp_elec * dy,
                             0.1 * temp_elec * dz, 0.1 * temp_par_vdw * dx,
                             0.1 * temp_par_vdw * dy, 0.1 * temp_par_vdw * dz);
      } else {
        CachedData.set_elec_vdw(pos, charge, temp_vdw * (c_12 - c_6));
        CachedData.set_force(pos, 0.1 * temp_elec * dx, 0.1 * temp_elec * dy,
                             0.1 * temp_elec * dz, 0.1 * temp_par_vdw * dx,
                             0.1 * temp_par_vdw * dy, 0.1 * temp_par_vdw * dz);
        Energy += CachedData.get_energy(pos);
        Force += *CachedData.get_force(pos);
      }
    }
  }
}

void cal_energy_force_gromos(
    const MOL &rec, const MOL &lg, const vector<mol_info> &info_rec,
    const vector<mol_info> &info_lg, cache_sigma_epsilon &CachedData,
    size_t begin, size_t end, cache_energy &Energy, cache_force &Force,
    size_t num_snap, const arguments &Arguments,
    const Multi_MOL_parameters & rec_parameters,
    const Multi_MOL_parameters & lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, &Force, Arguments);
    return;
  }
  Energy.set_energy(0., 0.);
  Force.set_force(force(0., 0., 0.), force(0., 0., 0.));
  const double epsilon_in = std::max(1.0e-12, Arguments.epsilon_in);

  for (size_t j{begin}; j < end; ++j) {
    double temp_charge{138.935457839 * lg_parameters[j].get_charge()},
        temp_epsilon_lg{lg_parameters[j].get_epsilon()},
        temp_sigma_lg{lg_parameters[j].get_sigma()};
    for (size_t i{}; i < rec.size(); ++i) {
      const size_t pos{rec.size() * j + i};
      if (Arguments.self_int == true &&
          ((&rec == &lg && i == j) ||
           (Arguments.flag_interchain_energy == true &&
            info_lg[j].get_chain() == info_rec[i].get_chain()))) {
        zero_cached_pair_if_present(CachedData, pos);
        continue;
      }

      double dx{rec.pos.x[i] - lg.pos.x[j]},
          dy{rec.pos.y[i] - lg.pos.y[j]},
          dz{rec.pos.z[i] - lg.pos.z[j]};

      double d{0.1 * sqrt(square(dx) + square(dy) + square(dz))},
          charge{temp_charge * rec_parameters[i].get_charge() / epsilon_in};
      if (!std::isfinite(d) ||
          d <= titan_energy_validation::kMinimumPairDistanceNm) {
        zero_cached_pair_if_present(CachedData, pos);
        continue;
      }
      double sd{1. / square(d)}, temp_elec{sd * charge / d}, c_6{c6(d)},
          c_12{square(c_6)};
      double sigma{}, epsilon{};
      gromos::pair_params(CachedData,rec_parameters,lg_parameters,i,j,sigma,epsilon);
      double temp_vdw{12. * sd * epsilon / c_12 - 6. * sd * sigma / c_6};

      if (Arguments.distance_energy_cutoff > 0. &&
          d > Arguments.distance_energy_cutoff) {
        zero_cached_pair_if_present(CachedData, pos);
        continue;
      }

      if (CachedData.elec_ener.empty()) {
        Energy += cache_energy(charge / d, epsilon / c_12 - sigma / c_6);
        Force += cache_force(0.1 * temp_elec * dx, 0.1 * temp_elec * dy,
                             0.1 * temp_elec * dz, 0.1 * temp_vdw * dx,
                             0.1 * temp_vdw * dy, 0.1 * temp_vdw * dz);
      } else {
        CachedData.set_elec_vdw(pos, charge / d, epsilon / c_12 - sigma / c_6);
        CachedData.set_force(pos, 0.1 * temp_elec * dx, 0.1 * temp_elec * dy,
                             0.1 * temp_elec * dz, 0.1 * temp_vdw * dx,
                             0.1 * temp_vdw * dy, 0.1 * temp_vdw * dz);
        Energy += CachedData.get_energy(pos);
        Force += *CachedData.get_force(pos);
      }
    }
  }
}

void cal_energy_force_martini(const MOL &rec, const MOL &lg,
                              const vector<mol_info> &info_rec,
                              const vector<mol_info> &info_lg,
                              cache_sigma_epsilon &CachedData, size_t begin,
                              size_t end, cache_energy &Energy,
                              cache_force &Force, size_t num_snap,
                              const arguments &Arguments,
                              const Multi_MOL_parameters &rec_parameters,
                              const Multi_MOL_parameters &lg_parameters) {
  if (titan_nb::topology_energy_required(CachedData, Arguments)) {
    titan_nb::calculate(rec, lg, info_rec, info_lg, CachedData, begin, end,
                        Energy, &Force, Arguments);
    return;
  }
  Energy.set_energy(0., 0.);
  Force.set_force(force(0., 0., 0.), force(0., 0., 0.));
  const auto elec_cfg = martini::electrostatic_config(Arguments);

  for (size_t j{begin}; j < end; ++j) {
    for (size_t i{}; i < rec.size(); ++i) {
      const double dx{rec.pos.x[i] - lg.pos.x[j]};
      const double dy{rec.pos.y[i] - lg.pos.y[j]};
      const double dz{rec.pos.z[i] - lg.pos.z[j]};
      const double d{0.1 * sqrt(square(dx) + square(dy) + square(dz))};

      size_t pos{rec.size() * j + i};
      if (d <= titan_energy_validation::kMinimumPairDistanceNm) {
        if (!CachedData.elec_ener.empty()) {
          CachedData.set_elec_vdw(pos, 0., 0.);
          if (CachedData.has_forces())
            CachedData.set_force(pos, 0., 0., 0., 0., 0., 0.);
        }
        continue;
      }

      double sigma_ij{}, epsilon_ij{};
      martini::pair_params(CachedData, rec_parameters, lg_parameters, i, j,
                           sigma_ij, epsilon_ij);

      if (!martini::within_cutoff(elec_cfg, d) ||
          martini::excluded_pair(Arguments, info_rec, info_lg, i, j, d,
                                 sigma_ij)) {
        if (!CachedData.elec_ener.empty()) {
          CachedData.set_elec_vdw(pos, 0., 0.);
          if (CachedData.has_forces())
            CachedData.set_force(pos, 0., 0., 0., 0., 0., 0.);
        }
        continue;
      }

      const double e_val = martini::electro_energy(
          elec_cfg, rec_parameters[i].get_charge(),
          lg_parameters[j].get_charge(), d);
      const double v_val = martini::lj_energy(sigma_ij, epsilon_ij, d);
      const double elec_pref = martini::electro_force_prefactor(
          elec_cfg, rec_parameters[i].get_charge(),
          lg_parameters[j].get_charge(), d);
      const double vdw_pref =
          martini::lj_force_prefactor(sigma_ij, epsilon_ij, d);

      const double fx_e = 0.1 * elec_pref * dx;
      const double fy_e = 0.1 * elec_pref * dy;
      const double fz_e = 0.1 * elec_pref * dz;
      const double fx_v = 0.1 * vdw_pref * dx;
      const double fy_v = 0.1 * vdw_pref * dy;
      const double fz_v = 0.1 * vdw_pref * dz;

      if (CachedData.elec_ener.empty()) {
        Energy += cache_energy(e_val, v_val);
        Force += cache_force(fx_e, fy_e, fz_e, fx_v, fy_v, fz_v);
      } else {
        CachedData.set_elec_vdw(pos, e_val, v_val);
        if (CachedData.has_forces())
          CachedData.set_force(pos, fx_e, fy_e, fz_e, fx_v, fy_v, fz_v);
        Energy += CachedData.get_energy(pos);
        if (CachedData.has_forces())
          Force += *CachedData.get_force(pos);
        else
          Force += cache_force(fx_e, fy_e, fz_e, fx_v, fy_v, fz_v);
      }
    }
  }
}

int imp_cal_force_lg_atom(vector<data_out> &lg_ener_inter, const Multi_MOL &rec,
                          const Multi_MOL &lg, size_t num_snap, size_t begin,
                          size_t end, const vector<d_min> &dmin,
                          const cache_sigma_epsilon &CachedData,
                          const Multi_MOL_parameters & rec_parameters,
                          const Multi_MOL_parameters & lg_parameters,
                          const arguments &Arguments) {
  for (size_t i = begin; i < end; ++i) {
    double E_E{}, E_V{};
    force f_e_temp{}, f_v_temp{};
    for (size_t j{}; j < rec.m_mol[num_snap].size(); ++j) {
      force_energy_add(CachedData, rec.m_mol[0].size() * i + j, E_E,
                       E_V, f_e_temp, f_v_temp, rec_parameters, lg_parameters,
                       rec, lg, num_snap, Arguments, j, i);
    }
    lg_ener_inter.emplace_back(i + 1, " ", lg.info[i].get_original_res_name(),
                               E_E, E_V, f_e_temp, f_v_temp);
  }
  return 0;
}

int imp_cal_force_lg(vector<data_out> &lg_ener_inter, const Multi_MOL &rec,
                     const Multi_MOL &lg, size_t num_snap, size_t begin,
                     size_t end, const vector<d_min> &dmin,
                     const cache_sigma_epsilon &CachedData,
                     const Multi_MOL_parameters & rec_parameters,
                     const Multi_MOL_parameters & lg_parameters,
                     bool flag_all_mode, const arguments &Arguments) {
  unordered_set<pair<string, size_t>, pair_hash> force_list;

  for (size_t ii = begin; ii < end; ++ii) {
    for (size_t i = 0; i < dmin[ii].base.size(); ++i) {
      if (force_list
              .emplace(dmin[ii].base[i].chain_rec, dmin[ii].base[i].resnr_rec)
              .second == false)
        continue;
      double E_E{}, E_V{};
      force f_e_temp{}, f_v_temp{};
      for (size_t j = dmin[ii].base[i].atom_num - 1;
           j < lg.m_mol[num_snap].size(); ++j) {
        if (lg.info[j].get_resnr() ==
                lg.info[dmin[ii].base[i].atom_num - 1].get_resnr() &&
            lg.info[j].get_chain() ==
                lg.info[dmin[ii].base[i].atom_num - 1].get_chain()) {
          for (size_t k = 0; k < rec.m_mol[num_snap].size(); ++k) {
            if (flag_all_mode == true) {
              force_energy_add(CachedData, rec.m_mol[0].size() * j + k,
                               E_E, E_V, f_e_temp, f_v_temp, rec_parameters,
                               lg_parameters, rec, lg, num_snap, Arguments, k,
                               j);
            } else if (rec.info[k].get_resnr() ==
                           rec.info[dmin[ii].base[i].atom_num_rec - 1]
                               .get_resnr() &&
                       rec.info[k].get_chain() ==
                           rec.info[dmin[ii].base[i].atom_num_rec - 1]
                               .get_chain()) {
              force_energy_add(CachedData, rec.m_mol[0].size() * j + k,
                               E_E, E_V, f_e_temp, f_v_temp, rec_parameters,
                               lg_parameters, rec, lg, num_snap, Arguments, k,
                               j);
            } else {
            }
          }
        } else
          break;
      }
      lg_ener_inter.emplace_back(
          dmin[ii].base[i].resnr_ab, dmin[ii].base[i].chain_ab,
          dmin[ii].base[i].res_name_ab, E_E, E_V, f_e_temp, f_v_temp);
    }
  }

  return 0;
}

int imp_cal_force_rec(vector<data_out> &rec_ener_inter, const Multi_MOL &rec,
                      const Multi_MOL &lg, size_t num_snap, size_t begin,
                      size_t end, const vector<d_min> &dmin,
                      const cache_sigma_epsilon &CachedData,
                      const Multi_MOL_parameters & rec_parameters,
                      const Multi_MOL_parameters & lg_parameters,
                      bool flag_all_mode, const arguments &Arguments) {
  unordered_set<pair<string, size_t>, pair_hash> force_list;

  for (size_t ii{begin}; ii < end; ++ii) {
    for (size_t i{}; i < dmin[ii].base.size(); ++i) {
      if (force_list
              .emplace(dmin[ii].base[i].chain_rec, dmin[ii].base[i].resnr_rec)
              .second == false)
        continue;
      double E_E{}, E_V{};
      force f_e_temp{}, f_v_temp{};
      size_t num = dmin[ii].base[i].atom_num_rec - 1 + 250 > rec.info.size()
                       ? rec.info.size()
                       : dmin[ii].base[i].atom_num_rec - 1 + 250;
      for (size_t j = dmin[ii].base[i].atom_num_rec - 1; j < num; ++j) {
        if (rec.info[j].get_resnr() ==
                rec.info[dmin[ii].base[i].atom_num_rec - 1].get_resnr() &&
            rec.info[j].get_chain() ==
                rec.info[dmin[ii].base[i].atom_num_rec - 1].get_chain()) {
          for (size_t k{}; k < lg.m_mol[num_snap].size(); ++k) {
            if (flag_all_mode == true) {
              force_energy_add(CachedData, rec.m_mol[0].size() * k + j,
                               E_E, E_V, f_e_temp, f_v_temp, rec_parameters,
                               lg_parameters, rec, lg, num_snap, Arguments, j,
                               k);
            } else if (lg.info[k].get_resnr() ==
                           lg.info[dmin[ii].base[i].atom_num - 1].get_resnr() &&
                       lg.info[k].get_chain() ==
                           lg.info[dmin[ii].base[i].atom_num - 1].get_chain()) {
              force_energy_add(CachedData, rec.m_mol[0].size() * k + j,
                               E_E, E_V, f_e_temp, f_v_temp, rec_parameters,
                               lg_parameters, rec, lg, num_snap, Arguments, j,
                               k);
            } else {
            }
          }
        } else
          break;
      }
      rec_ener_inter.emplace_back(
          dmin[ii].base[i].resnr_rec, dmin[ii].base[i].chain_rec,
          dmin[ii].base[i].res_name_rec, E_E, E_V, f_e_temp, f_v_temp);
    }
  }
  return 0;
}

void forceMapOf2Pro(TitanExecutor &executor, const Multi_MOL &rec,
                    const Multi_MOL &lg, size_t num_snap,
                    const vector<d_min> &dmin,
                    const cache_sigma_epsilon &CachedData, size_t nt,
                    bool flag_all_mode, const arguments &Arguments,
                    vector<energy_res> &Energy_res_mol1,
                    vector<energy_res> &Energy_res_mol2, info &info_output,
                    const Multi_MOL_parameters & rec_parameters,
                    const Multi_MOL_parameters & lg_parameters) {
  vector<vector<data_out>> data_lg(nt), data_rec(nt);

  executor.parallel_do(nt, [&](size_t i) {
    imp_cal_force_lg(data_lg[i], rec, lg, num_snap, i * dmin.size() / nt,
                     (i + 1) * dmin.size() / nt, dmin, CachedData,
                     rec_parameters, lg_parameters, flag_all_mode, Arguments);
  });

  executor.parallel_do(nt, [&](size_t i) {
    imp_cal_force_rec(data_rec[i], rec, lg, num_snap, i * dmin.size() / nt,
                      (i + 1) * dmin.size() / nt, dmin, CachedData,
                      rec_parameters, lg_parameters, flag_all_mode, Arguments);
  });

  wrapper_print_force(nt, data_lg, data_rec, flag_all_mode, Arguments,
                      Energy_res_mol1, Energy_res_mol2, num_snap, info_output);
}

void forceMapOfProLig(TitanExecutor &executor, const Multi_MOL &rec,
                      const Multi_MOL &lg, size_t num_snap,
                      const vector<d_min> &dmin,
                      const cache_sigma_epsilon &CachedData, size_t nt,
                      bool flag_all_mode, const arguments &Arguments,
                      vector<energy_res> &Energy_res_mol1,
                      vector<energy_res> &Energy_res_mol2, info &info_output,
                      const Multi_MOL_parameters & rec_parameters,
                      const Multi_MOL_parameters & lg_parameters) {
  vector<vector<data_out>> data_lg(nt), data_rec(nt);

  executor.parallel_do(nt, [&](size_t i) {
    imp_cal_force_lg_atom(
        data_lg[i], rec, lg, num_snap, i * lg.m_mol[num_snap].size() / nt,
        (i + 1) * lg.m_mol[num_snap].size() / nt, dmin, CachedData,
        rec_parameters, lg_parameters, Arguments);
  });

  executor.parallel_do(nt, [&](size_t i) {
    imp_cal_force_rec(data_rec[i], rec, lg, num_snap, i * dmin.size() / nt,
                      (i + 1) * dmin.size() / nt, dmin, CachedData,
                      rec_parameters, lg_parameters, flag_all_mode, Arguments);
  });

  wrapper_print_force(nt, data_lg, data_rec, flag_all_mode, Arguments,
                      Energy_res_mol1, Energy_res_mol2, num_snap, info_output);
}

int forceMapOf2Ligs(TitanExecutor &executor, const Multi_MOL &rec,
                    const Multi_MOL &lg, size_t num_snap,
                    const vector<d_min> &dmin,
                    const cache_sigma_epsilon &CachedData, size_t nt,
                    const arguments &Arguments,
                    vector<energy_res> &Energy_res_mol1,
                    vector<energy_res> &Energy_res_mol2, info &info_output,
                    const Multi_MOL_parameters & rec_parameters,
                    const Multi_MOL_parameters & lg_parameters) {
  vector<vector<data_out>> data_lg(nt), data_rec(nt);

  executor.parallel_do(nt, [&](size_t i) {
    imp_cal_force_lg_atom(data_lg[i], rec, lg, num_snap,
                          i * lg.m_mol[num_snap].size() / nt,
                          (i + 1) * lg.m_mol[num_snap].size() / nt, dmin,
                          CachedData, rec_parameters, lg_parameters, Arguments);
  });

  executor.parallel_do(nt, [&](size_t i) {
    imp_cal_force_lg_atom(data_rec[i], lg, rec, num_snap,
                          i * rec.m_mol[num_snap].size() / nt,
                          (i + 1) * rec.m_mol[num_snap].size() / nt, dmin,
                          CachedData, rec_parameters, lg_parameters, Arguments);
  });

  wrapper_print_force(1, data_lg, data_rec, true, Arguments, Energy_res_mol1,
                      Energy_res_mol2, num_snap, info_output);
  return 0;
}

int wrap_cal_force_map(TitanExecutor &executor, Multi_MOL &mol1,
                       Multi_MOL &mol2, const arguments &Arguments,
                       size_t num_snap, const vector<d_min> &dmin,
                       const cache_sigma_epsilon &CachedData,
                       vector<energy_res> &Energy_res_mol1,
                       vector<energy_res> &Energy_res_mol2, info &info_output) {
  if ((Arguments.flag_topo_1 == true && Arguments.flag_topo_2 == true) ||
      (Arguments.force_map == 3 && Arguments.fmap_type == "atom-atom")) {
    forceMapOf2Ligs(executor, mol1, mol2, num_snap, dmin, CachedData, Arguments.nt,
                    Arguments, Energy_res_mol1, Energy_res_mol2, info_output,
                    mol1.parameters, mol2.parameters);
  } else if ((Arguments.flag_topo_1 == true &&
              Arguments.flag_topo_2 == false) ||
             (Arguments.force_map == 3 &&
              Arguments.fmap_type == "residue-atom")) {
    forceMapOfProLig(executor, mol1, mol2, num_snap, dmin, CachedData,
                     Arguments.nt, Arguments.fmap_level == "all", Arguments,
                     Energy_res_mol1, Energy_res_mol2, info_output,
                     mol1.parameters, mol2.parameters);
  } else if ((Arguments.flag_topo_1 == false &&
              Arguments.flag_topo_2 == true) ||
             (Arguments.force_map == 3 &&
              Arguments.fmap_type == "atom-residue")) {
    forceMapOfProLig(executor, mol2, mol1, num_snap, dmin, CachedData,
                     Arguments.nt, Arguments.fmap_level == "all", Arguments,
                     Energy_res_mol1, Energy_res_mol2, info_output,
                     mol1.parameters, mol2.parameters);
  } else {
    forceMapOf2Pro(executor, mol1, mol2, num_snap, dmin, CachedData,
                   Arguments.nt, Arguments.fmap_level == "all", Arguments,
                   Energy_res_mol1, Energy_res_mol2, info_output,
                   mol1.parameters, mol2.parameters);
  }
  return 0;
}
