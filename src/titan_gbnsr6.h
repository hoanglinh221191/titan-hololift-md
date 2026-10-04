#pragma once
#include "tabi/tabi_calderon_aux_surface.h"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace titan_gb {
// Amber GBNSR6/PB constants (variable_module.F90). No 0.73 sander GB factor.
inline double nsr6_kappa_nm_inverse(double molar, double dielectric, double kelvin) {
  if (!std::isfinite(molar) || molar < 0. || !std::isfinite(dielectric) ||
      dielectric <= 0. || !std::isfinite(kelvin) || kelvin <= 0.)
    throw std::invalid_argument("GBNSR6: invalid electrolyte parameters");
  constexpr double kb = 1.3807e-23/1.6606e-27/1.e24*1.e20;
  constexpr double eps0 = 8.8542e-12/(1.6022e-19*1.6022e-19)/1.e30*1.e24*1.6606e-27;
  constexpr double molar_number_density = 6.0220e-4;
  return 10.*std::sqrt(2.*molar*molar_number_density/(dielectric*eps0*kb*kelvin));
}
// Numerical surface R6: R_i^-1 = [integral_S (r-r_i).n/|r-r_i|^6 dS
// / (4*pi)]^(1/3) + B. Forouzesh et al., DOI:10.1021/acs.jcim.7b00192.
// A triangulated TITAN SES is used, not Amber's Cartesian surface quadrature.
// The default B=0.028 A^-1 is calibrated for a 1.4 A solvent probe.
inline double nsr6_inverse_radius(double surface_integral, double offset = .028) {
  if (!std::isfinite(surface_integral) || surface_integral <= 0. ||
      !std::isfinite(offset) || offset < 0.)
    throw std::runtime_error("GBNSR6: nonpositive/nonfinite R6 surface integral");
  return std::cbrt(surface_integral/(4.*std::numbers::pi)) + offset;
}

inline double nsr6_panel_integral(const TitanPBE::Tabi::Panel &p,
                                 double x, double y, double z) {
  // Symmetric degree-2 triangle rule. Refine the SES grid to check convergence.
  if (!p.has_triangle_geometry || !std::isfinite(p.area) || p.area <= 0.)
    throw std::runtime_error("GBNSR6: invalid SES triangle");
  double sum = 0.;
  for (int k=0; k<3; ++k) {
    const double a = k==0 ? 2./3. : 1./6.;
    const double b = k==1 ? 2./3. : 1./6.;
    const double c = k==2 ? 2./3. : 1./6.;
    const double dx = a*p.v0x+b*p.v1x+c*p.v2x-x;
    const double dy = a*p.v0y+b*p.v1y+c*p.v2y-y;
    const double dz = a*p.v0z+b*p.v1z+c*p.v2z-z;
    const double r2 = dx*dx+dy*dy+dz*dz;
    if (!(r2 > 0.) || !std::isfinite(r2))
      throw std::runtime_error("GBNSR6: singular surface quadrature");
    sum += (dx*p.nx+dy*p.ny+dz*p.nz)/(r2*r2*r2);
  }
  return p.area*sum/3.;
}

inline std::vector<double> gbnsr6_born_radii_nm(
    std::span<const TitanPBE::Tabi::Atom> atoms, double spacing = .5,
    TitanExecutor *executor = nullptr) {
  if (atoms.empty()) throw std::invalid_argument("GBNSR6: empty molecule");
  for (const auto &a : atoms)
    if (!std::isfinite(a.radius) || a.radius <= 0. || !std::isfinite(a.x) ||
        !std::isfinite(a.y) || !std::isfinite(a.z))
      throw std::invalid_argument("GBNSR6: invalid atom geometry");
  const auto surface = TitanPBE::Tabi::calderon::build_analytic_ses_surface(
      atoms, spacing, 1.4, executor, 4000000, true, "patch-ray-experimental");
  if (surface.status != "pass" || surface.mesh.panels.empty())
    throw std::runtime_error("GBNSR6 SES rejected: " + surface.status);
  std::vector<double> radii;
  radii.reserve(atoms.size());
  for (const auto &atom : atoms) {
    // Compensated signed sum: concave/reentrant panels can contribute negatively.
    double sum=0., correction=0.;
    for (const auto &panel : surface.mesh.panels) {
      const double term=nsr6_panel_integral(panel,atom.x,atom.y,atom.z)-correction;
      const double next=sum+term;
      correction=(next-sum)-term;
      sum=next;
    }
    radii.push_back(.1/nsr6_inverse_radius(sum));
  }
  return radii;
}
} // namespace titan_gb
