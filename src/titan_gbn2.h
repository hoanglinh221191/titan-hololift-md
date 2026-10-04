#pragma once

#include "titan_gbn2_tables.h"
#include <cmath>
#include <limits>
#include <string_view>

namespace titan_gb {
// Nguyen, Roe & Simmerling, JCTC 2013, DOI:10.1021/ct3010485.
// DNA/RNA: Nguyen et al., JCTC 2015, DOI:10.1021/acs.jctc.5b00271.
// AmberClassic mdread.F90 / egb.F90 conventions, lengths in Angstrom.
inline constexpr double kGbn2OffsetA = 0.195141;
struct Gbn2Parameters { double screen, alpha, beta, gamma; };

inline bool gbn2_nucleic_residue(std::string_view name) {
  // CHARMM/PDB nucleic names used by TITAN input (including example/4).
  // Normalize these to Amber A/C/G/U before an external sander comparison.
  if (name == "ADE" || name == "CYT" || name == "GUA" || name == "URA" ||
      name == "THY") return true;
  constexpr std::string_view names[] = {
    "A","A3","A5","AN","C","C3","C5","CN","DA","DA3","DA5","DAN",
    "DC","DC3","DC5","DCN","DG","DG3","DG5","DGN","DT","DT3","DT5","DTN",
    "G","G3","G5","8OG","GN","U","U3","U5","UN","AP","DAP","CP",
    "AF2","AF5","AF3","GF2","GF3","GF5","CF2","CFZ","CF3","CF5",
    "UF2","UF5","UF3","UFT","OMC","OMG","MC3","MC5","MG3","MG5",
    "SIC","NAM","NM5","CAP"};
  for (auto n : names) if (name == n) return true;
  return false;
}

inline Gbn2Parameters gbn2_parameters(char e, bool nucleic) {
  if (nucleic) {
    switch (e) {
    case 'H': return {1.696538, .537050, .362861, .116704};
    case 'C': return {1.268902, .331670, .196842, .093422};
    case 'N': return {1.4259728, .686311, .463189, .138722};
    case 'O': return {.1840098, .606344, .463006, .142262};
    case 'S': return {.5, .606344, .463006, .142262};
    case 'P': return {1.5450597, .418365, .290054, .1064245};
    default: return {.5, 1., .8, 4.851};
    }
  }
  switch (e) {
  case 'H': return {1.425952, .788440, .798699, .437334};
  case 'C': return {1.058554, .733756, .506378, .205844};
  case 'N': return {.733599, .503364, .316828, .192915};
  case 'O': return {1.061039, .867814, .876635, .387882};
  case 'S': return {-.703469, .867814, .876635, .387882};
  // Amber assigns the P tanh parameters even outside recognized RNA/DNA.
  // Its screening remains the unoptimized protein/ligand value 0.5.
  case 'P': return {.5, .418365, .290054, .1064245};
  default: return {.5, 1., .8, 4.851};
  }
}

inline double gbn2_neck_integral(double ri, double rj, double r) {
  if (!std::isfinite(ri) || !std::isfinite(rj) || !std::isfinite(r) ||
      ri < 1. || ri > 2. || rj < 1. || rj > 2. || r <= 0.)
    return std::numeric_limits<double>::quiet_NaN();
  if (r >= ri + rj + 6.8) return 0.;
  // Amber uses nearest grid point (not interpolation), Fortran column-major.
  const auto i = static_cast<std::size_t>(std::lround((ri-1.)*20.));
  const auto j = static_cast<std::size_t>(std::lround((rj-1.)*20.));
  const double d = r - neck_d0[j*21+i], d2 = d*d;
  return .826836 * neck_m0[j*21+i] / (1. + d2 + .3*d2*d2*d2);
}

// Amber's far-field expansion also defines the signed sulfur contribution.
// Negative screening MUST NOT be treated as a deleted/zero-volume atom.
inline double gbn2_far_integral(double s, double r) {
  const double u = (s/r)*(s/r);
  return u*s/(r*r) * (1./3. + u*(2./5. + u*(3./7. + u*(4./9. + u*5./11.))));
}

inline double gbn2_born_radius(double radius, double integral,
                               const Gbn2Parameters &p) {
  if (!std::isfinite(radius) || radius <= kGbn2OffsetA ||
      !std::isfinite(integral) || !std::isfinite(p.alpha) ||
      !std::isfinite(p.beta) || !std::isfinite(p.gamma))
    return std::numeric_limits<double>::quiet_NaN();
  const double offset = radius-kGbn2OffsetA;
  const double psi = integral*offset;
  const double inv = 1./offset - std::tanh(psi*(p.alpha-psi*p.beta+psi*psi*p.gamma))/radius;
  if (!std::isfinite(inv) || inv <= 0. || offset <= 0.)
    return std::numeric_limits<double>::quiet_NaN();
  return 1./inv;
}
} // namespace titan_gb
