#ifndef TITAN_PBCTOPO_STERIC_CLASH_H
#define TITAN_PBCTOPO_STERIC_CLASH_H

#include "types.h"

#include <vector>

namespace titan_pbctopo {

// Explicit zero-LJ sites follow the force field: if either site has an
// explicitly zero Lennard-Jones radius, that pair has no LJ hard core.
inline constexpr const char *kPbctopoZeroLjStericPolicy =
    "force_field_no_hard_core";

struct PbctopoPairClashThreshold {
  double cutoff = 0.0;
  double cutoff2 = 0.0;
  bool used_vdw_radii = false;
  bool explicit_zero_lj_exempt = false;
};

PbctopoPairClashThreshold pbctopo_pair_clash_threshold(
    const PbctopoAtom &lhs, const PbctopoAtom &rhs, double fallback_cutoff);

void prepare_pbctopo_steric_radii(std::vector<PbctopoAtom> &atoms,
                                  double vdw_clash_scale);

double pbctopo_max_clash_cutoff(const std::vector<PbctopoAtom> &atoms,
                                double fallback_cutoff);

bool pbctopo_atoms_clash(double d2, const PbctopoAtom &lhs,
                         const PbctopoAtom &rhs, double fallback_cutoff,
                         PbctopoTimingStats *timing = nullptr);

double pbctopo_pair_soft_overlap_penalty(double d2, const PbctopoAtom &lhs,
                                         const PbctopoAtom &rhs,
                                         double fallback_cutoff);

} // namespace titan_pbctopo

#endif
