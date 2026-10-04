#pragma once

#include "algebra/mat3.h"
#include "algebra/matrix_view.h"
#include "algebra/symmetric_eigen.h"

#include <array>
#include <cassert>
#include <cmath>

namespace titan::algebra {

[[nodiscard]] inline Mat3 quaternion_to_column_rotation(
    const std::array<double, 4> &q_in) noexcept {
  double w = q_in[0];
  double x = q_in[1];
  double y = q_in[2];
  double z = q_in[3];
  const double n2 = w * w + x * x + y * y + z * z;
  if (n2 <= 0.0)
    return Mat3::identity();
  const double inv_n = 1.0 / std::sqrt(n2);
  w *= inv_n;
  x *= inv_n;
  y *= inv_n;
  z *= inv_n;

  Mat3 r{};
  r(0, 0) = 1.0 - 2.0 * (y * y + z * z);
  r(0, 1) = 2.0 * (x * y - w * z);
  r(0, 2) = 2.0 * (x * z + w * y);
  r(1, 0) = 2.0 * (x * y + w * z);
  r(1, 1) = 1.0 - 2.0 * (x * x + z * z);
  r(1, 2) = 2.0 * (y * z - w * x);
  r(2, 0) = 2.0 * (x * z - w * y);
  r(2, 1) = 2.0 * (y * z + w * x);
  r(2, 2) = 1.0 - 2.0 * (x * x + y * y);
  return r;
}

[[nodiscard]] inline Mat3
kabsch_rotation(Matrix3ConstView ref_centered,
                Matrix3ConstView mobile_centered) noexcept {
  assert(ref_centered.rows == mobile_centered.rows);
  if (ref_centered.rows == 0)
    return Mat3::identity();

  double s00 = 0.0, s01 = 0.0, s02 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0;
  double s20 = 0.0, s21 = 0.0, s22 = 0.0;
  for (std::size_t r = 0; r < ref_centered.rows; ++r) {
    const double px = mobile_centered(r, 0);
    const double py = mobile_centered(r, 1);
    const double pz = mobile_centered(r, 2);
    const double qx = ref_centered(r, 0);
    const double qy = ref_centered(r, 1);
    const double qz = ref_centered(r, 2);
    s00 += px * qx;
    s01 += px * qy;
    s02 += px * qz;
    s10 += py * qx;
    s11 += py * qy;
    s12 += py * qz;
    s20 += pz * qx;
    s21 += pz * qy;
    s22 += pz * qz;
  }

  const std::array<double, 16> n = {
      s00 + s11 + s22,
      s12 - s21,
      s20 - s02,
      s01 - s10,
      s12 - s21,
      s00 - s11 - s22,
      s01 + s10,
      s02 + s20,
      s20 - s02,
      s01 + s10,
      -s00 + s11 - s22,
      s12 + s21,
      s01 - s10,
      s02 + s20,
      s12 + s21,
      -s00 - s11 + s22,
  };

  const auto q = dominant_eigenvector_symmetric4(n);
  return transpose(quaternion_to_column_rotation(q));
}

} // namespace titan::algebra
