#ifndef TITAN_GB_MODEL_H
#define TITAN_GB_MODEL_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

#if defined(__AVX__)
#include <immintrin.h>
#endif

namespace titan_gb {

// Amber HCT/OBC uses an offset intrinsic radius and an element-specific
// screening factor for the neighbour that performs the descreening.  TITAN
// stores coordinates and input radii in Angstrom, so this layer deliberately
// remains in Angstrom until the final Born radius is converted to nanometres.
inline constexpr double kDielectricOffsetAngstrom = 0.09;
inline constexpr double kCoulombKJMolNmPerE2 = 138.935457839;
// Sigalov et al. analytical linearized Poisson-Boltzmann coefficient.  This
// alpha is dimensionless; the length scale enters separately through A.
inline constexpr double kAlpbAlpha = 0.571412;

// Determinant-based whole-system electrostatic size A_Det (Sigalov et al.,
// Eq. 23).  Coordinates and atomic size parameters are supplied in Angstrom;
// the returned size is in nm for direct use by the GB/ALPB energy kernel.
[[nodiscard]] inline double electrostatic_size_adet_nm(
    std::span<const double> x_angstrom, std::span<const double> y_angstrom,
    std::span<const double> z_angstrom,
    std::span<const double> radii_angstrom) noexcept {
  const std::size_t count = radii_angstrom.size();
  if (count == 0 || x_angstrom.size() != count ||
      y_angstrom.size() != count || z_angstrom.size() != count)
    return std::numeric_limits<double>::quiet_NaN();

  long double total_mass = 0.0L;
  long double center_x = 0.0L;
  long double center_y = 0.0L;
  long double center_z = 0.0L;
  for (std::size_t atom = 0; atom < count; ++atom) {
    const double x = x_angstrom[atom];
    const double y = y_angstrom[atom];
    const double z = z_angstrom[atom];
    const double radius = radii_angstrom[atom];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        !std::isfinite(radius) || radius < 0.0)
      return std::numeric_limits<double>::quiet_NaN();
    // Zero-LJ-radius charged sites contribute no geometric volume, but remain
    // in the electrostatic calculation (for example Amber polar hydrogens).
    const long double radius_ld = static_cast<long double>(radius);
    const long double mass = radius_ld * radius_ld * radius_ld;
    total_mass += mass;
    center_x += mass * static_cast<long double>(x);
    center_y += mass * static_cast<long double>(y);
    center_z += mass * static_cast<long double>(z);
  }
  if (!(total_mass > 0.0L) || !std::isfinite(total_mass))
    return std::numeric_limits<double>::quiet_NaN();
  center_x /= total_mass;
  center_y /= total_mass;
  center_z /= total_mass;

  long double i_xx = 0.0L;
  long double i_yy = 0.0L;
  long double i_zz = 0.0L;
  long double i_xy = 0.0L;
  long double i_xz = 0.0L;
  long double i_yz = 0.0L;
  for (std::size_t atom = 0; atom < count; ++atom) {
    const long double radius =
        static_cast<long double>(radii_angstrom[atom]);
    const long double mass = radius * radius * radius;
    const long double x =
        static_cast<long double>(x_angstrom[atom]) - center_x;
    const long double y =
        static_cast<long double>(y_angstrom[atom]) - center_y;
    const long double z =
        static_cast<long double>(z_angstrom[atom]) - center_z;
    const long double sphere_inertia = 0.4L * radius * radius;
    i_xx += mass * (y * y + z * z + sphere_inertia);
    i_yy += mass * (x * x + z * z + sphere_inertia);
    i_zz += mass * (x * x + y * y + sphere_inertia);
    i_xy -= mass * x * y;
    i_xz -= mass * x * z;
    i_yz -= mass * y * z;
  }

  const long double determinant =
      i_xx * (i_yy * i_zz - i_yz * i_yz) -
      i_xy * (i_xy * i_zz - i_xz * i_yz) +
      i_xz * (i_xy * i_yz - i_xz * i_yy);
  if (!(determinant > 0.0L) || !std::isfinite(determinant))
    return std::numeric_limits<double>::quiet_NaN();

  const long double size_angstrom =
      std::sqrt(5.0L / (2.0L * total_mass)) *
      std::pow(determinant, 1.0L / 6.0L);
  const double size_nm = static_cast<double>(0.1L * size_angstrom);
  return std::isfinite(size_nm) && size_nm > 0.0
             ? size_nm
             : std::numeric_limits<double>::quiet_NaN();
}

[[nodiscard]] inline constexpr double
screening_factor_for_element(char element) noexcept {
  switch (element) {
  case 'H':
    return 0.85;
  case 'C':
    return 0.72;
  case 'N':
    return 0.79;
  case 'O':
    return 0.85;
  case 'F':
    return 0.88;
  case 'P':
    return 0.86;
  case 'S':
    return 0.96;
  default:
    return 0.80;
  }
}

[[nodiscard]] inline constexpr double
standard_heavy_atom_radius_angstrom(char element) noexcept {
  switch (element) {
  case 'C':
    return 1.70;
  case 'N':
    return 1.55;
  case 'O':
  case 'F':
    return 1.50;
  case 'P':
    return 1.85;
  case 'S':
    return 1.80;
  case 'I':
    return 1.98;
  default:
    return 1.50;
  }
}

// Pair contribution to the HCT descreening integral I_i in A^-1.  This is the
// analytical pair integral used by the reference OBC implementation, including
// the Ponder correction for atom i lying completely inside atom j.
//
// rgbmax_angstrom truncates the integral's upper limit, matching Amber's egb
// cutoff (default there: 25 A). The closed form below is G(L) - G(U) with
// G(x) = 1/x + ((s^2 - r^2)/(4r))/x^2 + ln(x)/(2r), so clamping U at rgbmax IS
// the truncated integral; it agrees with Amber's expanded boundary branch to
// ~1e-10 relative and is better conditioned than that expansion. The default
// +inf leaves every arithmetic operation below exactly as it was before the
// parameter existed, so uncut results stay bit-for-bit reproducible.
[[nodiscard]] inline double hct_pair_integral_inverse_angstrom(
    double offset_radius_i_angstrom, double scaled_radius_j_angstrom,
    double distance_angstrom,
    double rgbmax_angstrom =
        std::numeric_limits<double>::infinity()) noexcept {
  if (!std::isfinite(offset_radius_i_angstrom) ||
      !std::isfinite(scaled_radius_j_angstrom) ||
      !std::isfinite(distance_angstrom) || offset_radius_i_angstrom <= 0.0 ||
      scaled_radius_j_angstrom <= 0.0 || distance_angstrom <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();

  double upper_distance = distance_angstrom + scaled_radius_j_angstrom;
  // Written as a guarded comparison rather than std::min so that +inf (and a
  // non-finite value) leaves upper_distance untouched.
  if (rgbmax_angstrom > 0.0 && upper_distance > rgbmax_angstrom)
    upper_distance = rgbmax_angstrom;
  if (offset_radius_i_angstrom >= upper_distance)
    return 0.0;

  const double lower_distance =
      std::max(offset_radius_i_angstrom,
               std::abs(distance_angstrom - scaled_radius_j_angstrom));
  if (!(lower_distance > 0.0) || !(upper_distance > 0.0))
    return std::numeric_limits<double>::quiet_NaN();
  // Reachable only with a finite cutoff: the shell lies entirely outside it.
  if (lower_distance >= upper_distance)
    return 0.0;

  const double inv_r = 1.0 / distance_angstrom;
  const double lower_inv = 1.0 / lower_distance;
  const double upper_inv = 1.0 / upper_distance;
  const double lower_inv2 = lower_inv * lower_inv;
  const double upper_inv2 = upper_inv * upper_inv;
  const double scaled_radius2 =
      scaled_radius_j_angstrom * scaled_radius_j_angstrom;

  double term = lower_inv - upper_inv +
                0.25 * distance_angstrom * (upper_inv2 - lower_inv2) +
                0.5 * inv_r * std::log(upper_inv / lower_inv) +
                0.25 * scaled_radius2 * inv_r *
                    (lower_inv2 - upper_inv2);

  if (offset_radius_i_angstrom <
      scaled_radius_j_angstrom - distance_angstrom) {
    term += 2.0 * (1.0 / offset_radius_i_angstrom - lower_inv);
  }
  return 0.5 * term;
}

[[nodiscard]] inline double
born_radius_angstrom(double intrinsic_radius_angstrom,
                     double hct_integral_inverse_angstrom, int igb) noexcept {
  if (!std::isfinite(intrinsic_radius_angstrom) ||
      !std::isfinite(hct_integral_inverse_angstrom) ||
      intrinsic_radius_angstrom <= kDielectricOffsetAngstrom)
    return std::numeric_limits<double>::quiet_NaN();

  const double offset_radius =
      intrinsic_radius_angstrom - kDielectricOffsetAngstrom;
  double inverse_born = 0.0;
  if (igb == 1) {
    inverse_born = 1.0 / offset_radius - hct_integral_inverse_angstrom;
  } else {
    double alpha = 0.0;
    double beta = 0.0;
    double gamma = 0.0;
    if (igb == 2) {
      alpha = 0.8;
      beta = 0.0;
      gamma = 2.909125;
    } else if (igb == 5) {
      alpha = 1.0;
      beta = 0.8;
      gamma = 4.85;
    } else {
      return std::numeric_limits<double>::quiet_NaN();
    }

    const double psi = hct_integral_inverse_angstrom * offset_radius;
    const double psi2 = psi * psi;
    const double tanh_sum =
        std::tanh(alpha * psi - beta * psi2 + gamma * psi2 * psi);
    inverse_born =
        1.0 / offset_radius - tanh_sum / intrinsic_radius_angstrom;
  }

  if (!std::isfinite(inverse_born) || inverse_born <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  return 1.0 / inverse_born;
}

[[nodiscard]] inline double
effective_gb_distance_nm(double distance_squared_nm2, double born_i_nm,
                         double born_j_nm) noexcept {
  if (!std::isfinite(distance_squared_nm2) ||
      !std::isfinite(born_i_nm) || !std::isfinite(born_j_nm) ||
      distance_squared_nm2 < 0.0 || born_i_nm <= 0.0 || born_j_nm <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  const double product = born_i_nm * born_j_nm;
  return std::sqrt(distance_squared_nm2 +
                   product * std::exp(-0.25 * distance_squared_nm2 / product));
}

// Generalized-Born polar pair term in kJ/mol.  For an unordered non-self pair
// use the returned value directly; for i==j apply the usual factor of 1/2.
[[nodiscard]] inline double polar_pair_energy_kj_mol(
    double charge_i, double charge_j, double distance_squared_nm2,
    double born_i_nm, double born_j_nm, double kappa_nm_inverse,
    double internal_dielectric, double external_dielectric,
    bool use_alpb = false,
    double alpb_electrostatic_size_nm =
        std::numeric_limits<double>::quiet_NaN()) noexcept {
  if (!std::isfinite(charge_i) || !std::isfinite(charge_j) ||
      !std::isfinite(kappa_nm_inverse) || kappa_nm_inverse < 0.0 ||
      !std::isfinite(internal_dielectric) || internal_dielectric <= 0.0 ||
      !std::isfinite(external_dielectric) || external_dielectric <= 0.0 ||
      (use_alpb && (!std::isfinite(alpb_electrostatic_size_nm) ||
                    alpb_electrostatic_size_nm <= 0.0)))
    return std::numeric_limits<double>::quiet_NaN();
  const double f_gb =
      effective_gb_distance_nm(distance_squared_nm2, born_i_nm, born_j_nm);
  if (!std::isfinite(f_gb) || f_gb <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  const double screened = std::exp(-kappa_nm_inverse * f_gb);
  const double dielectric_term =
      1.0 / internal_dielectric - screened / external_dielectric;
  if (use_alpb) {
    // Salt-screened ALPB expression (Sigalov et al., Eq. 25).  The global
    // electrostatic size A corrects the inverse-distance term, while alpha is
    // the dimensionless analytical constant above.
    const double beta = internal_dielectric / external_dielectric;
    const double alpha_beta = kAlpbAlpha * beta;
    const double inverse_distance =
        1.0 / f_gb + alpha_beta / alpb_electrostatic_size_nm;
    return -kCoulombKJMolNmPerE2 * charge_i * charge_j * dielectric_term *
           inverse_distance / (1.0 + alpha_beta);
  }
  return -kCoulombKJMolNmPerE2 * charge_i * charge_j *
         dielectric_term / f_gb;
}

#if defined(__AVX__)
// Four-lane implementation of the same Sigalov Eq. 25 term used above.
// f_gb and exp(-kappa*f_gb) are supplied by the fused GBSA kernel so the
// production vector path and the standalone scientific oracle share one
// implementation boundary.
[[nodiscard]] inline __m256d alpb_polar_pair_energy_from_fgb_avx(
    __m256d charge_product, __m256d f_gb_nm,
    __m256d external_screening, double internal_dielectric,
    double external_dielectric,
    double alpb_electrostatic_size_nm) noexcept {
  if (!std::isfinite(internal_dielectric) || internal_dielectric <= 0.0 ||
      !std::isfinite(external_dielectric) || external_dielectric <= 0.0 ||
      !std::isfinite(alpb_electrostatic_size_nm) ||
      alpb_electrostatic_size_nm <= 0.0) {
    return _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());
  }

  const double alpha_beta =
      kAlpbAlpha * internal_dielectric / external_dielectric;
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d dielectric_term = _mm256_sub_pd(
      _mm256_set1_pd(1.0 / internal_dielectric),
      _mm256_mul_pd(external_screening,
                    _mm256_set1_pd(1.0 / external_dielectric)));
  const __m256d inverse_distance = _mm256_add_pd(
      _mm256_div_pd(one, f_gb_nm),
      _mm256_set1_pd(alpha_beta / alpb_electrostatic_size_nm));
  const __m256d scale = _mm256_set1_pd(
      -kCoulombKJMolNmPerE2 / (1.0 + alpha_beta));
  return _mm256_mul_pd(
      scale,
      _mm256_mul_pd(charge_product,
                    _mm256_mul_pd(dielectric_term, inverse_distance)));
}
#endif

} // namespace titan_gb

#endif
