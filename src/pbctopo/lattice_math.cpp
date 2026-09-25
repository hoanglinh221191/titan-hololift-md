#include "lattice_math.h"

#include "../BoxMatrix.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <tuple>

namespace titan_pbctopo {
namespace {

using LatticeInt = std::int64_t;

constexpr std::size_t kCvpNodeBudget = 1000000;
constexpr double kConditionLimit = 1.0e10;

thread_local PbctopoLatticeQueryAudit g_lattice_query_audit;

PbctopoNearestImage record_lattice_query(PbctopoNearestImage result) noexcept {
  ++g_lattice_query_audit.queries;
  if (!result.valid)
    ++g_lattice_query_audit.failures;
  if (result.ambiguous)
    ++g_lattice_query_audit.ambiguous;
  if (result.node_budget_exhausted)
    ++g_lattice_query_audit.enumeration_budget_failures;
  return result;
}

double row_norm2(const double matrix[3][3], int row) {
  return matrix[row][0] * matrix[row][0] +
         matrix[row][1] * matrix[row][1] +
         matrix[row][2] * matrix[row][2];
}

double row_dot(const double matrix[3][3], int lhs, int rhs) {
  return matrix[lhs][0] * matrix[rhs][0] +
         matrix[lhs][1] * matrix[rhs][1] +
         matrix[lhs][2] * matrix[rhs][2];
}

double determinant3(const double matrix[3][3]) {
  return matrix[0][0] *
             (matrix[1][1] * matrix[2][2] -
              matrix[1][2] * matrix[2][1]) -
         matrix[0][1] *
             (matrix[1][0] * matrix[2][2] -
              matrix[1][2] * matrix[2][0]) +
         matrix[0][2] *
             (matrix[1][0] * matrix[2][1] -
              matrix[1][1] * matrix[2][0]);
}

bool invert3(const double matrix[3][3], double inverse[3][3]) {
  const double determinant = determinant3(matrix);
  const double row_volume_scale =
      std::sqrt(std::max(0.0, row_norm2(matrix, 0) * row_norm2(matrix, 1) *
                                  row_norm2(matrix, 2)));
  if (!std::isfinite(determinant) || !std::isfinite(row_volume_scale) ||
      row_volume_scale <= 0.0 ||
      std::fabs(determinant) <= 1.0e-12 * row_volume_scale)
    return false;
  const double inv_det = 1.0 / determinant;
  inverse[0][0] =
      (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det;
  inverse[0][1] =
      (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det;
  inverse[0][2] =
      (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det;
  inverse[1][0] =
      (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det;
  inverse[1][1] =
      (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det;
  inverse[1][2] =
      (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det;
  inverse[2][0] =
      (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det;
  inverse[2][1] =
      (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det;
  inverse[2][2] =
      (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det;
  return true;
}

bool checked_multiply(LatticeInt lhs, LatticeInt rhs,
                      LatticeInt &product) {
  if (lhs == 0 || rhs == 0) {
    product = 0;
    return true;
  }
  constexpr LatticeInt min_value =
      std::numeric_limits<LatticeInt>::min();
  constexpr LatticeInt max_value =
      std::numeric_limits<LatticeInt>::max();
  if ((lhs == -1 && rhs == min_value) ||
      (rhs == -1 && lhs == min_value)) {
    return false;
  }
  if (lhs > 0) {
    if ((rhs > 0 && lhs > max_value / rhs) ||
        (rhs < 0 && rhs < min_value / lhs)) {
      return false;
    }
  } else {
    if ((rhs > 0 && lhs < min_value / rhs) ||
        (rhs < 0 && lhs < max_value / rhs)) {
      return false;
    }
  }
  product = lhs * rhs;
  return true;
}

bool checked_add(LatticeInt lhs, LatticeInt rhs, LatticeInt &sum) {
  constexpr LatticeInt min_value =
      std::numeric_limits<LatticeInt>::min();
  constexpr LatticeInt max_value =
      std::numeric_limits<LatticeInt>::max();
  if ((rhs > 0 && lhs > max_value - rhs) ||
      (rhs < 0 && lhs < min_value - rhs)) {
    return false;
  }
  sum = lhs + rhs;
  return true;
}

bool checked_subtract(LatticeInt lhs, LatticeInt rhs, LatticeInt &difference) {
  if (rhs == std::numeric_limits<LatticeInt>::min())
    return false;
  return checked_add(lhs, -rhs, difference);
}

bool checked_multiply_subtract(LatticeInt current, LatticeInt multiplier,
                               LatticeInt base, LatticeInt &result) {
  LatticeInt product = 0;
  return checked_multiply(multiplier, base, product) &&
         checked_subtract(current, product, result);
}

bool checked_minor(LatticeInt a, LatticeInt b, LatticeInt c, LatticeInt d,
                   LatticeInt &minor) {
  LatticeInt lhs = 0;
  LatticeInt rhs = 0;
  return checked_multiply(a, d, lhs) && checked_multiply(b, c, rhs) &&
         checked_subtract(lhs, rhs, minor);
}

bool determinant3(const LatticeInt matrix[3][3], LatticeInt &determinant) {
  LatticeInt minor0 = 0;
  LatticeInt minor1 = 0;
  LatticeInt minor2 = 0;
  LatticeInt term0 = 0;
  LatticeInt term1 = 0;
  LatticeInt term2 = 0;
  LatticeInt partial = 0;
  return checked_minor(matrix[1][1], matrix[1][2], matrix[2][1],
                       matrix[2][2], minor0) &&
         checked_minor(matrix[1][0], matrix[1][2], matrix[2][0],
                       matrix[2][2], minor1) &&
         checked_minor(matrix[1][0], matrix[1][1], matrix[2][0],
                       matrix[2][1], minor2) &&
         checked_multiply(matrix[0][0], minor0, term0) &&
         checked_multiply(matrix[0][1], minor1, term1) &&
         checked_multiply(matrix[0][2], minor2, term2) &&
         checked_subtract(term0, term1, partial) &&
         checked_add(partial, term2, determinant);
}

bool cofactor3(const LatticeInt matrix[3][3], int row, int col,
               LatticeInt &cofactor) {
  LatticeInt values[4]{};
  int next = 0;
  for (int r = 0; r < 3; ++r) {
    if (r == row)
      continue;
    for (int c = 0; c < 3; ++c) {
      if (c != col)
        values[next++] = matrix[r][c];
    }
  }
  if (!checked_minor(values[0], values[1], values[2], values[3],
                     cofactor)) {
    return false;
  }
  if (((row + col) & 1) != 0) {
    if (cofactor == std::numeric_limits<LatticeInt>::min())
      return false;
    cofactor = -cofactor;
  }
  return true;
}

bool checked_nearest_integer(double value, LatticeInt &nearest) {
  if (!std::isfinite(value))
    return false;
  const long double rounded =
      std::floor(static_cast<long double>(value) + 0.5L);
  if (rounded <
          static_cast<long double>(std::numeric_limits<LatticeInt>::min()) ||
      rounded >
          static_cast<long double>(std::numeric_limits<LatticeInt>::max())) {
    return false;
  }
  nearest = static_cast<LatticeInt>(rounded);
  return true;
}

bool checked_integer_bounds(long double lower, long double upper,
                            LatticeInt &min_value, LatticeInt &max_value) {
  if (!std::isfinite(lower) || !std::isfinite(upper))
    return false;
  lower = std::ceil(lower);
  upper = std::floor(upper);
  if (lower > upper) {
    min_value = 1;
    max_value = 0;
    return true;
  }
  if (lower <
          static_cast<long double>(std::numeric_limits<LatticeInt>::min()) ||
      upper >
          static_cast<long double>(std::numeric_limits<LatticeInt>::max())) {
    return false;
  }
  min_value = static_cast<LatticeInt>(lower);
  max_value = static_cast<LatticeInt>(upper);
  return true;
}

int physical_orientation_sign(const std::array<double, 3> &cartesian) {
  for (double value : cartesian) {
    if (value > 0.0)
      return 1;
    if (value < 0.0)
      return -1;
  }
  return 1;
}

bool cholesky_upper_from_rows(const double matrix[3][3],
                              double upper[3][3]) {
  double gram[3][3]{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col)
      gram[row][col] = row_dot(matrix, row, col);
  }
  double lower[3][3]{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col <= row; ++col) {
      double value = gram[row][col];
      for (int inner = 0; inner < col; ++inner)
        value -= lower[row][inner] * lower[col][inner];
      if (row == col) {
        if (!std::isfinite(value) || value <= 1.0e-24)
          return false;
        lower[row][col] = std::sqrt(value);
      } else {
        lower[row][col] = value / lower[col][col];
      }
    }
  }
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col)
      upper[row][col] = col >= row ? lower[col][row] : 0.0;
  return true;
}

} // namespace

void pbctopo_reset_lattice_query_audit() noexcept {
  g_lattice_query_audit = {};
}

PbctopoLatticeQueryAudit pbctopo_lattice_query_audit() noexcept {
  return g_lattice_query_audit;
}

void pbctopo_add_lattice_query_audit(
    const PbctopoLatticeQueryAudit &delta) noexcept {
  g_lattice_query_audit.queries += delta.queries;
  g_lattice_query_audit.failures += delta.failures;
  g_lattice_query_audit.ambiguous += delta.ambiguous;
  g_lattice_query_audit.enumeration_budget_failures +=
      delta.enumeration_budget_failures;
}

PbctopoLatticeQueryAudit pbctopo_lattice_query_audit_delta(
    const PbctopoLatticeQueryAudit &before,
    const PbctopoLatticeQueryAudit &after) noexcept {
  const auto delta = [](std::size_t older, std::size_t newer) {
    return newer >= older ? newer - older : newer;
  };
  return {delta(before.queries, after.queries),
          delta(before.failures, after.failures),
          delta(before.ambiguous, after.ambiguous),
          delta(before.enumeration_budget_failures,
                after.enumeration_budget_failures)};
}

bool pbctopo_checked_round_image(double value,
                                 std::int64_t &rounded) noexcept {
  if (!std::isfinite(value))
    return false;
  const long double integral = std::round(static_cast<long double>(value));
  constexpr long double lower =
      static_cast<long double>(std::numeric_limits<std::int64_t>::min());
  constexpr long double upper_exclusive = -lower;
  if (!std::isfinite(integral) || integral < lower ||
      integral >= upper_exclusive) {
    return false;
  }
  rounded = static_cast<std::int64_t>(integral);
  return true;
}

const char *pbctopo_lattice_status_name(PbctopoLatticeStatus status) {
  switch (status) {
  case PbctopoLatticeStatus::Valid:
    return "valid";
  case PbctopoLatticeStatus::ReductionFallback:
    return "reduction_fallback";
  case PbctopoLatticeStatus::InvalidBox:
    return "invalid_box";
  case PbctopoLatticeStatus::NonFiniteMatrix:
    return "nonfinite_matrix";
  case PbctopoLatticeStatus::SingularMatrix:
    return "singular_matrix";
  case PbctopoLatticeStatus::IllConditionedMatrix:
    return "ill_conditioned_matrix";
  case PbctopoLatticeStatus::EnumerationBudgetExceeded:
    return "enumeration_budget_exceeded";
  }
  return "invalid_box";
}

PbctopoLatticeMetric::PbctopoLatticeMetric(const box &Box) {
  if (!BoxMatrix::valid_box(Box)) {
    status_ = PbctopoLatticeStatus::InvalidBox;
    return;
  }
  const BoxMatrix source(Box);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      matrix_[row][col] = source.m[row][col];
      inverse_[row][col] = source.inv[row][col];
    }
  }
  initialize_properties();
}

PbctopoLatticeMetric::PbctopoLatticeMetric(
    std::span<const double, 9> matrix) {
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      matrix_[row][col] =
          matrix[static_cast<std::size_t>(3 * row + col)];
    }
  }
  if (!invert3(matrix_, inverse_)) {
    status_ = std::all_of(matrix.begin(), matrix.end(), [](double value) {
                return std::isfinite(value);
              })
                  ? PbctopoLatticeStatus::SingularMatrix
                  : PbctopoLatticeStatus::NonFiniteMatrix;
    return;
  }
  initialize_properties();
}

PbctopoLatticeMetric::PbctopoLatticeMetric(
    const std::array<double, 6> &gram) {
  if (!std::all_of(gram.begin(), gram.end(),
                   [](double value) { return std::isfinite(value); }) ||
      gram[0] <= 0.0) {
    status_ = PbctopoLatticeStatus::NonFiniteMatrix;
    return;
  }
  matrix_[0][0] = std::sqrt(gram[0]);
  matrix_[1][0] = gram[1] / matrix_[0][0];
  matrix_[2][0] = gram[2] / matrix_[0][0];
  const double diagonal1 =
      gram[3] - matrix_[1][0] * matrix_[1][0];
  if (!std::isfinite(diagonal1) || diagonal1 <= 0.0) {
    status_ = PbctopoLatticeStatus::SingularMatrix;
    return;
  }
  matrix_[1][1] = std::sqrt(diagonal1);
  matrix_[2][1] =
      (gram[4] - matrix_[2][0] * matrix_[1][0]) / matrix_[1][1];
  const double diagonal2 = gram[5] - matrix_[2][0] * matrix_[2][0] -
                           matrix_[2][1] * matrix_[2][1];
  if (!std::isfinite(diagonal2) || diagonal2 <= 0.0) {
    status_ = PbctopoLatticeStatus::SingularMatrix;
    return;
  }
  matrix_[2][2] = std::sqrt(diagonal2);
  if (!invert3(matrix_, inverse_)) {
    status_ = PbctopoLatticeStatus::SingularMatrix;
    return;
  }
  initialize_properties();
}

PbctopoLatticeMetric::PbctopoLatticeMetric(
    const double matrix[3][3], const double inverse[3][3]) {
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      matrix_[row][col] = matrix[row][col];
      inverse_[row][col] = inverse[row][col];
    }
  }
  initialize_properties();
}

void PbctopoLatticeMetric::initialize_properties() {
  auto reset_reduction = [&]() {
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        reduced_matrix_[row][col] = matrix_[row][col];
        reduced_inverse_[row][col] = inverse_[row][col];
        reduced_to_original_[row][col] = row == col ? 1 : 0;
        original_to_reduced_[row][col] = row == col ? 1 : 0;
      }
    }
    basis_reduced_ = false;
  };
  reset_reduction();

  double matrix_norm2 = 0.0;
  double inverse_norm2 = 0.0;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      if (!std::isfinite(matrix_[row][col]) ||
          !std::isfinite(inverse_[row][col])) {
        status_ = PbctopoLatticeStatus::NonFiniteMatrix;
        return;
      }
      matrix_norm2 += matrix_[row][col] * matrix_[row][col];
      inverse_norm2 += inverse_[row][col] * inverse_[row][col];
    }
  }
  const double determinant = determinant3(matrix_);
  const double row_volume_scale =
      std::sqrt(std::max(0.0, row_norm2(matrix_, 0) *
                                  row_norm2(matrix_, 1) *
                                  row_norm2(matrix_, 2)));
  if (!std::isfinite(determinant) || !std::isfinite(row_volume_scale) ||
      row_volume_scale <= 0.0 ||
      std::abs(determinant) <= 1.0e-12 * row_volume_scale) {
    status_ = PbctopoLatticeStatus::SingularMatrix;
    return;
  }
  condition_estimate_ = std::sqrt(matrix_norm2 * inverse_norm2);
  if (!std::isfinite(condition_estimate_) ||
      condition_estimate_ > kConditionLimit) {
    status_ = PbctopoLatticeStatus::IllConditionedMatrix;
    return;
  }
  double inverse_residual_max = 0.0;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      double value = 0.0;
      for (int inner = 0; inner < 3; ++inner)
        value += matrix_[row][inner] * inverse_[inner][col];
      inverse_residual_max =
          std::max(inverse_residual_max,
                   std::abs(value - (row == col ? 1.0 : 0.0)));
    }
  }
  if (!std::isfinite(inverse_residual_max) || inverse_residual_max > 1.0e-7) {
    status_ = PbctopoLatticeStatus::IllConditionedMatrix;
    return;
  }

  valid_ = true;
  status_ = PbctopoLatticeStatus::Valid;
  bool reduction_failed = false;
  bool reduction_converged = false;
  for (int iteration = 0; iteration < 48; ++iteration) {
    bool changed = false;
    for (int lhs = 0; lhs < 2; ++lhs) {
      for (int rhs = lhs + 1; rhs < 3; ++rhs) {
        const double lhs_norm = row_norm2(reduced_matrix_, lhs);
        const double rhs_norm = row_norm2(reduced_matrix_, rhs);
        if (rhs_norm + 1.0e-13 * std::max(1.0, lhs_norm) >= lhs_norm)
          continue;
        for (int axis = 0; axis < 3; ++axis) {
          std::swap(reduced_matrix_[lhs][axis],
                    reduced_matrix_[rhs][axis]);
          std::swap(reduced_to_original_[lhs][axis],
                    reduced_to_original_[rhs][axis]);
        }
        changed = true;
      }
    }
    for (int row = 1; row < 3 && !reduction_failed; ++row) {
      for (int base = row - 1; base >= 0; --base) {
        const double denominator = row_norm2(reduced_matrix_, base);
        if (!std::isfinite(denominator) || denominator <= 1.0e-24)
          continue;
        const double ratio = row_dot(reduced_matrix_, row, base) / denominator;
        if (!std::isfinite(ratio) || std::abs(ratio) > 1000000.0)
          continue;
        const LatticeInt multiplier = static_cast<LatticeInt>(std::llround(ratio));
        if (multiplier == 0)
          continue;

        LatticeInt next_transform[3]{};
        double next_basis[3]{};
        bool update_ok = true;
        for (int axis = 0; axis < 3; ++axis) {
          update_ok = update_ok && checked_multiply_subtract(
                                       reduced_to_original_[row][axis],
                                       multiplier,
                                       reduced_to_original_[base][axis],
                                       next_transform[axis]);
          next_basis[axis] =
              reduced_matrix_[row][axis] -
              static_cast<double>(multiplier) * reduced_matrix_[base][axis];
          update_ok = update_ok && std::isfinite(next_basis[axis]);
        }
        if (!update_ok) {
          reduction_failed = true;
          break;
        }
        for (int axis = 0; axis < 3; ++axis) {
          reduced_to_original_[row][axis] = next_transform[axis];
          reduced_matrix_[row][axis] = next_basis[axis];
        }
        changed = true;
      }
    }
    if (reduction_failed)
      break;
    if (!changed) {
      reduction_converged = true;
      break;
    }
  }

  LatticeInt integer_determinant = 0;
  if (!reduction_failed && reduction_converged &&
      determinant3(reduced_to_original_, integer_determinant) &&
      (integer_determinant == 1 || integer_determinant == -1)) {
    for (int row = 0; row < 3 && !reduction_failed; ++row) {
      for (int col = 0; col < 3; ++col) {
        LatticeInt cofactor = 0;
        if (!cofactor3(reduced_to_original_, col, row, cofactor) ||
            (integer_determinant == -1 &&
             cofactor == std::numeric_limits<LatticeInt>::min())) {
          reduction_failed = true;
          break;
        }
        original_to_reduced_[row][col] =
            integer_determinant == 1 ? cofactor : -cofactor;
      }
    }
  } else {
    reduction_failed = true;
  }

  if (reduction_failed) {
    reset_reduction();
    reduction_fallback_ = true;
    status_ = PbctopoLatticeStatus::ReductionFallback;
  } else {
    basis_reduced_ = false;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        basis_reduced_ = basis_reduced_ ||
                         reduced_to_original_[row][col] !=
                             (row == col ? 1 : 0);
        reduced_inverse_[row][col] = 0.0;
        for (int inner = 0; inner < 3; ++inner) {
          reduced_inverse_[row][col] +=
              inverse_[row][inner] *
              static_cast<double>(original_to_reduced_[inner][col]);
        }
      }
    }
    for (int row = 0; row < 3; ++row) {
      if (!std::isfinite(row_norm2(reduced_matrix_, row)) ||
          row_norm2(reduced_matrix_, row) <= 1.0e-24) {
        reset_reduction();
        reduction_fallback_ = true;
        status_ = PbctopoLatticeStatus::ReductionFallback;
        break;
      }
    }
  }

  orthogonal_basis_ = true;
  for (int lhs = 0; lhs < 3; ++lhs) {
    const double lhs_norm2 = row_norm2(reduced_matrix_, lhs);
    for (int rhs = lhs + 1; rhs < 3; ++rhs) {
      const double rhs_norm2 = row_norm2(reduced_matrix_, rhs);
      const double dot = row_dot(reduced_matrix_, lhs, rhs);
      const double scale = std::sqrt(lhs_norm2 * rhs_norm2);
      if (!std::isfinite(scale) || scale <= 0.0 ||
          std::fabs(dot) > 1.0e-12 * scale) {
        orthogonal_basis_ = false;
      }
    }
  }

  reduced_cholesky_valid_ =
      cholesky_upper_from_rows(reduced_matrix_, reduced_cholesky_upper_);
  // Every nonzero lattice vector sum_i n_i r_i has some n_i != 0 and so a
  // component |n_i| h_i >= h_i along the normal of the face spanned by the
  // other two rows, where h_i = |det| / |r_j x r_k| is the cell height.  A
  // vector shorter than half the smallest height is therefore the unique
  // minimum image of its class, and its reduced fractional coordinates lie
  // inside (-1/2, 1/2), so rounding finds it.
  unique_minimum_d2_bound_ = 0.0;
  const double reduced_det = determinant3(reduced_matrix_);
  double smallest_height = std::numeric_limits<double>::infinity();
  for (int axis = 0; axis < 3; ++axis) {
    const int j = (axis + 1) % 3;
    const int k = (axis + 2) % 3;
    const double cx = reduced_matrix_[j][1] * reduced_matrix_[k][2] -
                      reduced_matrix_[j][2] * reduced_matrix_[k][1];
    const double cy = reduced_matrix_[j][2] * reduced_matrix_[k][0] -
                      reduced_matrix_[j][0] * reduced_matrix_[k][2];
    const double cz = reduced_matrix_[j][0] * reduced_matrix_[k][1] -
                      reduced_matrix_[j][1] * reduced_matrix_[k][0];
    const double face = std::sqrt(cx * cx + cy * cy + cz * cz);
    if (!std::isfinite(face) || face <= 0.0) {
      smallest_height = 0.0;
      break;
    }
    smallest_height = std::min(smallest_height, std::fabs(reduced_det) / face);
  }
  if (reduced_cholesky_valid_ && std::isfinite(smallest_height) &&
      smallest_height > 0.0) {
    unique_minimum_d2_bound_ =
        0.25 * smallest_height * smallest_height * (1.0 - 1.0e-6);
  }
}

std::array<double, 3>
PbctopoLatticeMetric::scaled_to_cartesian(double sx, double sy,
                                           double sz) const {
  return {matrix_[0][0] * sx + matrix_[1][0] * sy + matrix_[2][0] * sz,
          matrix_[0][1] * sx + matrix_[1][1] * sy + matrix_[2][1] * sz,
          matrix_[0][2] * sx + matrix_[1][2] * sy + matrix_[2][2] * sz};
}

std::array<double, 3>
PbctopoLatticeMetric::cartesian_to_scaled(double x, double y, double z) const {
  return {inverse_[0][0] * x + inverse_[1][0] * y + inverse_[2][0] * z,
          inverse_[0][1] * x + inverse_[1][1] * y + inverse_[2][1] * z,
          inverse_[0][2] * x + inverse_[1][2] * y + inverse_[2][2] * z};
}

PbctopoNearestImage PbctopoLatticeMetric::nearest_image_scaled(
    double dsx, double dsy, double dsz) const {
  PbctopoNearestImage best;
  best.d2 = std::numeric_limits<double>::infinity();
  best.lattice_status = status_;
  best.pathological_lattice = !valid_;
  if (!valid_ || !std::isfinite(dsx) || !std::isfinite(dsy) ||
      !std::isfinite(dsz)) {
    return record_lattice_query(best);
  }

  const double source_scaled[3]{dsx, dsy, dsz};
  const auto source_cartesian = scaled_to_cartesian(dsx, dsy, dsz);
  const int orientation_sign = physical_orientation_sign(source_cartesian);
  double reduced_scaled[3]{};
  for (int reduced_axis = 0; reduced_axis < 3; ++reduced_axis) {
    for (int source_axis = 0; source_axis < 3; ++source_axis) {
      reduced_scaled[reduced_axis] +=
          static_cast<double>(original_to_reduced_[source_axis][reduced_axis]) *
          source_scaled[source_axis];
    }
  }

  LatticeInt seed[3]{};
  for (int axis = 0; axis < 3; ++axis) {
    if (!checked_nearest_integer(reduced_scaled[axis], seed[axis])) {
      best.pathological_lattice = true;
      return record_lattice_query(best);
    }
  }

  auto consider = [&](LatticeInt nx, LatticeInt ny, LatticeInt nz) {
    const LatticeInt image_reduced[3]{-nx, -ny, -nz};
    LatticeInt image_original[3]{};
    for (int original_axis = 0; original_axis < 3; ++original_axis) {
      for (int reduced_axis = 0; reduced_axis < 3; ++reduced_axis) {
        LatticeInt product = 0;
        LatticeInt next = 0;
        if (!checked_multiply(
                reduced_to_original_[reduced_axis][original_axis],
                image_reduced[reduced_axis], product) ||
            !checked_add(image_original[original_axis], product, next)) {
          return;
        }
        image_original[original_axis] = next;
      }
    }

    const double reduced_residual[3]{
        reduced_scaled[0] - static_cast<double>(nx),
        reduced_scaled[1] - static_cast<double>(ny),
        reduced_scaled[2] - static_cast<double>(nz)};
    const std::array<double, 3> cart{
        reduced_matrix_[0][0] * reduced_residual[0] +
            reduced_matrix_[1][0] * reduced_residual[1] +
            reduced_matrix_[2][0] * reduced_residual[2],
        reduced_matrix_[0][1] * reduced_residual[0] +
            reduced_matrix_[1][1] * reduced_residual[1] +
            reduced_matrix_[2][1] * reduced_residual[2],
        reduced_matrix_[0][2] * reduced_residual[0] +
            reduced_matrix_[1][2] * reduced_residual[1] +
            reduced_matrix_[2][2] * reduced_residual[2]};
    const double d2 = cart[0] * cart[0] + cart[1] * cart[1] +
                      cart[2] * cart[2];
    if (!std::isfinite(d2))
      return;
    const double tolerance =
        best.valid
            ? 1.0e-13 * std::max({1.0, std::fabs(d2), std::fabs(best.d2)})
            : 0.0;
    const bool strict_better = !best.valid || d2 + tolerance < best.d2;
    const bool tied = best.valid && std::fabs(d2 - best.d2) <= tolerance;
    const auto tie_key = std::tuple{orientation_sign * cart[0],
                                    orientation_sign * cart[1],
                                    orientation_sign * cart[2]};
    const auto best_key = std::tuple{orientation_sign * best.x,
                                     orientation_sign * best.y,
                                     orientation_sign * best.z};
    if (strict_better) {
      best.equivalent_minima = 1;
    } else if (tied) {
      ++best.equivalent_minima;
    } else {
      return;
    }
    if (strict_better || tie_key < best_key) {
      const std::size_t equivalent_minima = best.equivalent_minima;
      best.sx = dsx + static_cast<double>(image_original[0]);
      best.sy = dsy + static_cast<double>(image_original[1]);
      best.sz = dsz + static_cast<double>(image_original[2]);
      best.x = cart[0];
      best.y = cart[1];
      best.z = cart[2];
      best.d2 = d2;
      best.image_x = image_original[0];
      best.image_y = image_original[1];
      best.image_z = image_original[2];
      best.equivalent_minima = equivalent_minima;
      best.valid = true;
    }
  };

  consider(seed[0], seed[1], seed[2]);
  if (!best.valid) {
    best.pathological_lattice = true;
    return record_lattice_query(best);
  }

  // A seed shorter than half the smallest reduced-cell height is the unique
  // minimum image (see unique_minimum_d2_bound_): no other lattice point ties
  // or beats it, so the neighbour scan and the enumeration below would both
  // return exactly this, with one equivalent minimum.  Bonds and contacts
  // end here.  A half-cell tie is at least half a height long and never takes
  // this path; the bound is 0 when the Cholesky factor is missing.
  if (best.d2 < unique_minimum_d2_bound_) {
    best.equivalent_minima = 1;
    best.ambiguous = false;
    return record_lattice_query(best);
  }

  bool half_cell_tie = false;
  for (int axis = 0; axis < 3; ++axis) {
    const double residual =
        reduced_scaled[axis] - static_cast<double>(seed[axis]);
    half_cell_tie = half_cell_tie ||
                    std::abs(std::abs(residual) - 0.5) <= 1.0e-12;
  }
  if (orthogonal_basis_) {
    if (half_cell_tie) {
      best.equivalent_minima = 0;
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz)
            consider(seed[0] + dx, seed[1] + dy, seed[2] + dz);
    }
    best.ambiguous = best.equivalent_minima > 1;
    return record_lattice_query(best);
  }

  for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dz = -1; dz <= 1; ++dz)
        consider(seed[0] + dx, seed[1] + dy, seed[2] + dz);

  if (!reduced_cholesky_valid_) {
    best.valid = false;
    best.pathological_lattice = true;
    best.lattice_status = PbctopoLatticeStatus::SingularMatrix;
    return record_lattice_query(best);
  }
  const auto &upper = reduced_cholesky_upper_;

  best.equivalent_minima = 0;
  LatticeInt selected[3]{};
  std::size_t node_count = 0;
  bool budget_exhausted = false;
  bool bound_failure = false;
  std::function<void(int, double)> enumerate = [&](int level,
                                                   double partial_d2) {
    if (budget_exhausted || bound_failure)
      return;
    if (level < 0) {
      consider(selected[0], selected[1], selected[2]);
      return;
    }
    const double tolerance =
        1.0e-13 * std::max(1.0, std::abs(best.d2));
    const double remaining_d2 = best.d2 + tolerance - partial_d2;
    if (remaining_d2 < 0.0)
      return;
    double tail = 0.0;
    for (int axis = level + 1; axis < 3; ++axis) {
      tail += upper[level][axis] *
              (reduced_scaled[axis] - static_cast<double>(selected[axis]));
    }
    const double diagonal = upper[level][level];
    if (!std::isfinite(diagonal) || diagonal <= 0.0) {
      bound_failure = true;
      return;
    }
    const long double center =
        static_cast<long double>(reduced_scaled[level] + tail / diagonal);
    const long double radius =
        static_cast<long double>(std::sqrt(std::max(0.0, remaining_d2)) /
                                 diagonal);
    LatticeInt min_integer = 0;
    LatticeInt max_integer = 0;
    LatticeInt center_integer = 0;
    if (!checked_integer_bounds(center - radius, center + radius, min_integer,
                                max_integer) ||
        !checked_nearest_integer(static_cast<double>(center), center_integer)) {
      bound_failure = true;
      return;
    }
    if (min_integer > max_integer)
      return;
    center_integer = std::clamp(center_integer, min_integer, max_integer);
    const long double integer_span =
        static_cast<long double>(max_integer) -
        static_cast<long double>(min_integer);
    if (integer_span > static_cast<long double>(kCvpNodeBudget)) {
      budget_exhausted = true;
      return;
    }
    const std::size_t left_span = static_cast<std::size_t>(
        static_cast<long double>(center_integer) -
        static_cast<long double>(min_integer));
    const std::size_t right_span = static_cast<std::size_t>(
        static_cast<long double>(max_integer) -
        static_cast<long double>(center_integer));

    auto visit = [&](LatticeInt integer) {
      if (budget_exhausted || bound_failure)
        return;
      if (++node_count > kCvpNodeBudget) {
        budget_exhausted = true;
        return;
      }
      selected[level] = integer;
      const double residual =
          reduced_scaled[level] - static_cast<double>(integer);
      const double term = diagonal * residual + tail;
      const double next_d2 = partial_d2 + term * term;
      const double next_tolerance =
          1.0e-13 * std::max(1.0, std::abs(best.d2));
      if (next_d2 <= best.d2 + next_tolerance)
        enumerate(level - 1, next_d2);
    };

    visit(center_integer);
    for (std::size_t step = 1;
         !budget_exhausted && !bound_failure; ++step) {
      bool visited = false;
      if (step <= left_span) {
        visit(center_integer - static_cast<LatticeInt>(step));
        visited = true;
      }
      if (step <= right_span) {
        visit(center_integer + static_cast<LatticeInt>(step));
        visited = true;
      }
      if (!visited)
        break;
      if (step >= kCvpNodeBudget) {
        budget_exhausted = true;
        break;
      }
    }
  };
  enumerate(2, 0.0);

  if (budget_exhausted || bound_failure || best.equivalent_minima == 0) {
    best.valid = false;
    best.pathological_lattice = true;
    best.node_budget_exhausted = budget_exhausted;
    best.lattice_status = budget_exhausted
                              ? PbctopoLatticeStatus::EnumerationBudgetExceeded
                              : PbctopoLatticeStatus::SingularMatrix;
    return record_lattice_query(best);
  }
  best.ambiguous = best.equivalent_minima > 1;
  return record_lattice_query(best);
}

bool PbctopoLatticeMetric::image_is_nearest_equivalent_scaled(
    double dsx, double dsy, double dsz, std::int64_t image_x,
    std::int64_t image_y, std::int64_t image_z,
    const PbctopoNearestImage &nearest,
    double relative_tolerance) const {
  if (!valid_ || !nearest.valid || !std::isfinite(nearest.d2) ||
      !std::isfinite(relative_tolerance) || relative_tolerance < 0.0) {
    return false;
  }
  const auto candidate = scaled_to_cartesian(
      dsx + static_cast<double>(image_x),
      dsy + static_cast<double>(image_y),
      dsz + static_cast<double>(image_z));
  const double candidate_d2 = candidate[0] * candidate[0] +
                              candidate[1] * candidate[1] +
                              candidate[2] * candidate[2];
  if (!std::isfinite(candidate_d2))
    return false;
  const double tolerance =
      relative_tolerance * std::max({1.0, candidate_d2, nearest.d2});
  return std::fabs(candidate_d2 - nearest.d2) <= tolerance;
}

PbctopoNearestImage PbctopoLatticeMetric::nearest_image_cartesian(
    double dx, double dy, double dz) const {
  const auto scaled = cartesian_to_scaled(dx, dy, dz);
  return nearest_image_scaled(scaled[0], scaled[1], scaled[2]);
}

double PbctopoLatticeMetric::shortest_basis_length() const {
  if (!valid_)
    return std::numeric_limits<double>::infinity();
  double shortest = std::numeric_limits<double>::infinity();
  for (int row = 0; row < 3; ++row) {
    const double length =
        std::sqrt(reduced_matrix_[row][0] * reduced_matrix_[row][0] +
                  reduced_matrix_[row][1] * reduced_matrix_[row][1] +
                  reduced_matrix_[row][2] * reduced_matrix_[row][2]);
    shortest = std::min(shortest, length);
  }
  return shortest;
}

} // namespace titan_pbctopo
