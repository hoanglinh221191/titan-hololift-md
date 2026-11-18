#ifndef TITAN_LJ_PAIR_OVERRIDE_H
#define TITAN_LJ_PAIR_OVERRIDE_H

// ENE-11 (21/10/2026). Pair-specific Lennard-Jones parameters
// ([ nonbond_params ], NBFIX, CUFIX) for the kernels that otherwise mix the
// per-atom sigma/epsilon by a combination rule: AMBER (arithmetic sigma,
// geometric epsilon) and OPLS-AA (geometric sigma and epsilon).
//
// The override table is the one CHARMM and GROMOS already use
// (cache_sigma_epsilon::charmm_pair_*, keyed by atom type id): sigma in nm and
// epsilon in kJ/mol for AMBER/CHARMM/OPLS, C6/C12 for GROMOS (gromos_nonbonded.h
// reads it there). A pair without an override keeps the value the kernel
// computed from the combination rule, and the kernels test
// has_charmm_pair_overrides() once per call, so a run without
// [ nonbond_params ] is unchanged bit for bit.

#include "charmm_nonbonded.h"

#include <cstddef>

namespace titan_lj {

// 12-6 energy (kJ/mol) of the pair from its override, 4*eps*((s/r)^12-(s/r)^6).
// Returns false, leaving `energy` untouched, when the pair has no override.
inline bool override_lj_energy(const cache_sigma_epsilon &cache,
                               const Multi_MOL_parameters &rec_parameters,
                               const Multi_MOL_parameters &lg_parameters,
                               std::size_t rec_pos, std::size_t lg_pos,
                               double distance_nm, double &energy) {
  double sigma{}, epsilon{};
  if (!charmm::lookup_pair_override(cache, rec_parameters, lg_parameters,
                                    rec_pos, lg_pos, sigma, epsilon))
    return false;
  energy = charmm::lj_energy(sigma, epsilon, distance_nm);
  return true;
}

// As override_lj_energy, plus the force factor F/r (kJ mol^-1 nm^-2) with the
// same convention as the kernels' temp_par_vdw: F_vec = factor * r_vec[nm].
inline bool override_lj_energy_force(const cache_sigma_epsilon &cache,
                                     const Multi_MOL_parameters &rec_parameters,
                                     const Multi_MOL_parameters &lg_parameters,
                                     std::size_t rec_pos, std::size_t lg_pos,
                                     double distance_nm, double &energy,
                                     double &force_factor) {
  double sigma{}, epsilon{};
  if (!charmm::lookup_pair_override(cache, rec_parameters, lg_parameters,
                                    rec_pos, lg_pos, sigma, epsilon))
    return false;
  energy = charmm::lj_energy(sigma, epsilon, distance_nm);
  force_factor = charmm::lj_force_prefactor(sigma, epsilon, distance_nm);
  return true;
}

// Replaces, in a block of `lane_count` receptor atoms rec_begin.. paired with
// ligand atom lg_pos, the energies of the pairs that have an override.
inline void override_lj_lanes(const cache_sigma_epsilon &cache,
                              const Multi_MOL_parameters &rec_parameters,
                              const Multi_MOL_parameters &lg_parameters,
                              std::size_t rec_begin, std::size_t lane_count,
                              std::size_t lg_pos, const double *distance_nm,
                              double *energy) {
  for (std::size_t lane = 0; lane < lane_count; ++lane)
    override_lj_energy(cache, rec_parameters, lg_parameters, rec_begin + lane,
                       lg_pos, distance_nm[lane], energy[lane]);
}

} // namespace titan_lj

#endif // TITAN_LJ_PAIR_OVERRIDE_H
