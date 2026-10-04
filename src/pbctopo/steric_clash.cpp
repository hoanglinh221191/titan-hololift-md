#include "steric_clash.h"

#include <algorithm>
#include <cmath>

namespace titan_pbctopo {

PbctopoPairClashThreshold pbctopo_pair_clash_threshold(
    const PbctopoAtom &lhs, const PbctopoAtom &rhs, double fallback_cutoff) {
  PbctopoPairClashThreshold threshold;
  threshold.explicit_zero_lj_exempt =
      lhs.steric_radius_source ==
          PbctopoStericRadiusSource::ExplicitZeroLennardJones ||
      rhs.steric_radius_source ==
          PbctopoStericRadiusSource::ExplicitZeroLennardJones;
  if (threshold.explicit_zero_lj_exempt) {
    threshold.used_vdw_radii = true;
    return threshold;
  }
  const bool radii_valid = lhs.has_vdw_radius && rhs.has_vdw_radius &&
                           std::isfinite(lhs.vdw_radius) &&
                           std::isfinite(rhs.vdw_radius) &&
                           lhs.vdw_radius >= 0.0 && rhs.vdw_radius >= 0.0 &&
                           lhs.steric_radius >= 0.0 &&
                           rhs.steric_radius >= 0.0;
  threshold.used_vdw_radii = radii_valid;
  threshold.cutoff =
      radii_valid
          ? std::max(1.0e-6, lhs.steric_radius + rhs.steric_radius)
          : fallback_cutoff;
  threshold.cutoff2 = threshold.cutoff * threshold.cutoff;
  return threshold;
}

void prepare_pbctopo_steric_radii(std::vector<PbctopoAtom> &atoms,
                                  double vdw_clash_scale) {
  for (auto &atom : atoms) {
    atom.steric_radius =
        atom.has_vdw_radius && std::isfinite(atom.vdw_radius) &&
                atom.vdw_radius >= 0.0
            ? vdw_clash_scale * atom.vdw_radius
            : 0.0;
  }
}

double pbctopo_max_clash_cutoff(const std::vector<PbctopoAtom> &atoms,
                                double fallback_cutoff) {
  double max_radius = 0.0;
  for (const auto &atom : atoms) {
    if (atom.has_vdw_radius && std::isfinite(atom.vdw_radius))
      max_radius = std::max(max_radius, atom.steric_radius);
  }
  return std::max(fallback_cutoff, 2.0 * max_radius);
}

bool pbctopo_atoms_clash(double d2, const PbctopoAtom &lhs,
                         const PbctopoAtom &rhs, double fallback_cutoff,
                         PbctopoTimingStats *timing) {
  const auto threshold = pbctopo_pair_clash_threshold(
      lhs, rhs, fallback_cutoff);
  if (timing != nullptr) {
    ++timing->pbctopo_vdw_pair_checks;
    if (threshold.used_vdw_radii)
      ++timing->pbctopo_vdw_radius_pairs;
    else
      ++timing->pbctopo_vdw_radius_fallback_pairs;
    if (threshold.explicit_zero_lj_exempt)
      ++timing->pbctopo_vdw_explicit_zero_lj_exempt_pairs;
  }
  return d2 + 1.0e-12 < threshold.cutoff2;
}

double pbctopo_pair_soft_overlap_penalty(double d2, const PbctopoAtom &lhs,
                                         const PbctopoAtom &rhs,
                                         double fallback_cutoff) {
  if (lhs.steric_radius_source ==
          PbctopoStericRadiusSource::ExplicitZeroLennardJones ||
      rhs.steric_radius_source ==
          PbctopoStericRadiusSource::ExplicitZeroLennardJones)
    return 0.0;
  const bool radii_valid = lhs.has_vdw_radius && rhs.has_vdw_radius &&
                           std::isfinite(lhs.vdw_radius) &&
                           std::isfinite(rhs.vdw_radius) &&
                           lhs.vdw_radius >= 0.0 && rhs.vdw_radius >= 0.0;
  const double soft_cutoff =
      radii_valid ? lhs.vdw_radius + rhs.vdw_radius
                  : std::max(fallback_cutoff, 2.6);
  const double soft_cutoff2 = soft_cutoff * soft_cutoff;
  if (!(soft_cutoff2 > 0.0) || d2 >= soft_cutoff2)
    return 0.0;
  const double normalized_overlap =
      std::clamp(1.0 - std::max(0.0, d2) / soft_cutoff2, 0.0, 1.0);
  return normalized_overlap * normalized_overlap;
}

} // namespace titan_pbctopo
