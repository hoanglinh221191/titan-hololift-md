#ifndef BOX_MATRIX_H
#define BOX_MATRIX_H

#include "box_type.h"
#include "simd_compat.h"
#include <cmath>
#include <immintrin.h>
#include <vector>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct BoxMatrix {
  double m[3][3];
  double inv[3][3];

  static bool valid_box(const box &b) noexcept {
    const double a = b.get_a();
    const double b_len = b.get_b();
    const double c = b.get_c();
    const double alpha = b.get_alpha() * M_PI / 180.0;
    const double beta = b.get_beta() * M_PI / 180.0;
    const double gamma = b.get_gamma() * M_PI / 180.0;
    if (!std::isfinite(a) || !std::isfinite(b_len) || !std::isfinite(c) ||
        !std::isfinite(alpha) || !std::isfinite(beta) ||
        !std::isfinite(gamma) || !(a > 0.0) || !(b_len > 0.0) ||
        !(c > 0.0) || !(alpha > 0.0 && alpha < M_PI) ||
        !(beta > 0.0 && beta < M_PI) ||
        !(gamma > 0.0 && gamma < M_PI))
      return false;
    const double sin_gamma = std::sin(gamma);
    if (std::abs(sin_gamma) <= 1.0e-12)
      return false;
    const double m20 = c * std::cos(beta);
    const double m21 =
        c * (std::cos(alpha) - std::cos(beta) * std::cos(gamma)) /
        sin_gamma;
    const double m22_sq = c * c - m20 * m20 - m21 * m21;
    return std::isfinite(m22_sq) && m22_sq > 1.0e-18;
  }

  BoxMatrix(const box &b) {
    // v1: x-axis
    m[0][0] = b.get_a();
    m[0][1] = 0.0;
    m[0][2] = 0.0;
    // v2: xy-plane
    m[1][0] = b.get_b() * cos(b.get_gamma() * M_PI / 180.0);
    m[1][1] = b.get_b() * sin(b.get_gamma() * M_PI / 180.0);
    m[1][2] = 0.0;
    // v3
    m[2][0] = b.get_c() * cos(b.get_beta() * M_PI / 180.0);
    m[2][1] =
        b.get_c() *
        (cos(b.get_alpha() * M_PI / 180.0) -
         cos(b.get_beta() * M_PI / 180.0) * cos(b.get_gamma() * M_PI / 180.0)) /
        sin(b.get_gamma() * M_PI / 180.0);
    m[2][2] =
        sqrt(b.get_c() * b.get_c() - m[2][0] * m[2][0] - m[2][1] * m[2][1]);

    // Inverse
    double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                 m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                 m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
  }

  void apply_mic(double &dx, double &dy, double &dz) const {
    // Lattice vectors are stored as rows, so r = transpose(m) * f and
    // f = transpose(inv) * r.
    double fx = inv[0][0] * dx + inv[1][0] * dy + inv[2][0] * dz;
    double fy = inv[0][1] * dx + inv[1][1] * dy + inv[2][1] * dz;
    double fz = inv[0][2] * dx + inv[1][2] * dy + inv[2][2] * dz;

    // Shift to [-0.5, 0.5]
    fx -= std::round(fx);
    fy -= std::round(fy);
    fz -= std::round(fz);

    // Back to cartesian
    dx = m[0][0] * fx + m[1][0] * fy + m[2][0] * fz;
    dy = m[0][1] * fx + m[1][1] * fy + m[2][1] * fz;
    dz = m[0][2] * fx + m[1][2] * fy + m[2][2] * fz;
  }

#if TITAN_COMPILED_WITH_AVX
  void apply_mic_avx(__m256d &dx, __m256d &dy, __m256d &dz) const {
    __m256d m00 = _mm256_set1_pd(inv[0][0]);
    __m256d m01 = _mm256_set1_pd(inv[1][0]);
    __m256d m02 = _mm256_set1_pd(inv[2][0]);
    __m256d m10 = _mm256_set1_pd(inv[0][1]);
    __m256d m11 = _mm256_set1_pd(inv[1][1]);
    __m256d m12 = _mm256_set1_pd(inv[2][1]);
    __m256d m20 = _mm256_set1_pd(inv[0][2]);
    __m256d m21 = _mm256_set1_pd(inv[1][2]);
    __m256d m22 = _mm256_set1_pd(inv[2][2]);

    __m256d fx = simd_compat::fmadd_pd(
        m00, dx, simd_compat::fmadd_pd(m01, dy, _mm256_mul_pd(m02, dz)));
    __m256d fy = simd_compat::fmadd_pd(
        m10, dx, simd_compat::fmadd_pd(m11, dy, _mm256_mul_pd(m12, dz)));
    __m256d fz = simd_compat::fmadd_pd(
        m20, dx, simd_compat::fmadd_pd(m21, dy, _mm256_mul_pd(m22, dz)));

    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const auto round_half_away_from_zero = [&](const __m256d value) {
      const __m256d signed_half =
          _mm256_or_pd(half, _mm256_and_pd(value, sign_mask));
      return _mm256_round_pd(_mm256_add_pd(value, signed_half),
                             _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    };
    fx = _mm256_sub_pd(fx, round_half_away_from_zero(fx));
    fy = _mm256_sub_pd(fy, round_half_away_from_zero(fy));
    fz = _mm256_sub_pd(fz, round_half_away_from_zero(fz));

    __m256d r00 = _mm256_set1_pd(m[0][0]);
    __m256d r01 = _mm256_set1_pd(m[0][1]);
    __m256d r02 = _mm256_set1_pd(m[0][2]);
    __m256d r10 = _mm256_set1_pd(m[1][0]);
    __m256d r11 = _mm256_set1_pd(m[1][1]);
    __m256d r12 = _mm256_set1_pd(m[1][2]);
    __m256d r20 = _mm256_set1_pd(m[2][0]);
    __m256d r21 = _mm256_set1_pd(m[2][1]);
    __m256d r22 = _mm256_set1_pd(m[2][2]);

    dx = simd_compat::fmadd_pd(
        r00, fx, simd_compat::fmadd_pd(r10, fy, _mm256_mul_pd(r20, fz)));
    dy = simd_compat::fmadd_pd(
        r01, fx, simd_compat::fmadd_pd(r11, fy, _mm256_mul_pd(r21, fz)));
    dz = simd_compat::fmadd_pd(
        r02, fx, simd_compat::fmadd_pd(r12, fy, _mm256_mul_pd(r22, fz)));
  }
#endif
};

#endif
