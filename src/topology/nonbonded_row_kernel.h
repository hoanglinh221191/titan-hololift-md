#ifndef TITAN_TOPOLOGY_NONBONDED_ROW_KERNEL_H
#define TITAN_TOPOLOGY_NONBONDED_ROW_KERNEL_H

// ENE-10 (22/10/2026). One ligand atom against a run of receptor atoms that
// all belong to other molecule instances (so no exclusion and no listed
// pair applies): Lennard-Jones from the topology type matrix and Coulomb,
// optionally Debye-screened, energies only. AVX2 when the build has it.
// Per pair it is the arithmetic of titan_nb::evaluate_pair for a
// non-MARTINI topology; only the summation order differs (lanes).

#include "topology/nonbonded_topology.h"

#include <cstddef>
#include <cstdint>

namespace titan_nb {

struct RowInput {
  const double *x{}, *y{}, *z{};           // receptor coordinates, Angstrom
  const std::uint32_t *type_offset{};      // receptor type * type_count
  const double *charge{};                  // receptor charges, e
  std::size_t count{};
  double lx{}, ly{}, lz{};                 // ligand atom, Angstrom
  std::uint32_t lig_type{};
  double lig_charge{};
  const Coefficients *matrix{};
  double coulomb{};                        // 138.935... / epsilon_in
  double kappa{};                          // 1/nm, 0 = no screening
  double cutoff{};                         // nm, <= 0 = none
  double *elec_out{}, *vdw_out{};          // per-pair cache rows or nullptr
};

// Adds the row to elec_sum/vdw_sum and returns true; returns false, adding
// nothing, when a pair with non-zero parameters is closer than
// kMinimumPairDistanceNm or not finite (the caller's scalar path reports it).
bool row_energy(const RowInput &in, double &elec_sum, double &vdw_sum);

} // namespace titan_nb

#endif // TITAN_TOPOLOGY_NONBONDED_ROW_KERNEL_H
