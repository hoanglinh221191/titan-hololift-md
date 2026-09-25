#ifndef TITAN_PBCTOPO_LATTICE_MATH_H
#define TITAN_PBCTOPO_LATTICE_MATH_H

#include "../box_type.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace titan_pbctopo {

enum class PbctopoLatticeStatus : std::uint8_t {
  Valid,
  ReductionFallback,
  InvalidBox,
  NonFiniteMatrix,
  SingularMatrix,
  IllConditionedMatrix,
  EnumerationBudgetExceeded,
};

const char *pbctopo_lattice_status_name(PbctopoLatticeStatus status);

struct PbctopoNearestImage {
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double d2 = 0.0;
  std::int64_t image_x = 0;
  std::int64_t image_y = 0;
  std::int64_t image_z = 0;
  std::size_t equivalent_minima = 0;
  bool ambiguous = false;
  bool pathological_lattice = false;
  bool node_budget_exhausted = false;
  PbctopoLatticeStatus lattice_status = PbctopoLatticeStatus::InvalidBox;
  bool valid = false;
};

struct PbctopoLatticeQueryAudit {
  std::size_t queries = 0;
  std::size_t failures = 0;
  std::size_t ambiguous = 0;
  std::size_t enumeration_budget_failures = 0;
};

void pbctopo_reset_lattice_query_audit() noexcept;
PbctopoLatticeQueryAudit pbctopo_lattice_query_audit() noexcept;
// Adds counts measured on another thread to this thread's audit, so that work
// moved to executor workers is counted where the serial code counted it.
void pbctopo_add_lattice_query_audit(const PbctopoLatticeQueryAudit &delta) noexcept;
PbctopoLatticeQueryAudit pbctopo_lattice_query_audit_delta(
    const PbctopoLatticeQueryAudit &before,
    const PbctopoLatticeQueryAudit &after) noexcept;
bool pbctopo_checked_round_image(double value,
                                  std::int64_t &rounded) noexcept;

class PbctopoLatticeMetric {
public:
  explicit PbctopoLatticeMetric(const box &Box);
  explicit PbctopoLatticeMetric(std::span<const double, 9> matrix);
  explicit PbctopoLatticeMetric(const std::array<double, 6> &gram);
  PbctopoLatticeMetric(const double matrix[3][3],
                       const double inverse[3][3]);

  std::array<double, 3> scaled_to_cartesian(double sx, double sy,
                                             double sz) const;
  std::array<double, 3> cartesian_to_scaled(double x, double y,
                                             double z) const;
  PbctopoNearestImage nearest_image_scaled(double dsx, double dsy,
                                            double dsz) const;
  PbctopoNearestImage nearest_image_cartesian(double dx, double dy,
                                               double dz) const;
  bool image_is_nearest_equivalent_scaled(
      double dsx, double dsy, double dsz, std::int64_t image_x,
      std::int64_t image_y, std::int64_t image_z,
      const PbctopoNearestImage &nearest,
      double relative_tolerance = 1.0e-12) const;
  double shortest_basis_length() const;
  bool basis_reduced() const noexcept { return basis_reduced_; }
  bool valid() const noexcept { return valid_; }
  bool reduction_fallback() const noexcept { return reduction_fallback_; }
  double condition_estimate() const noexcept { return condition_estimate_; }
  PbctopoLatticeStatus status() const noexcept { return status_; }

private:
  void initialize_properties();

  double matrix_[3][3]{};
  double inverse_[3][3]{};
  double reduced_matrix_[3][3]{};
  double reduced_inverse_[3][3]{};
  std::int64_t reduced_to_original_[3][3]{};
  std::int64_t original_to_reduced_[3][3]{};
  bool orthogonal_basis_ = false;
  bool basis_reduced_ = false;
  bool valid_ = false;
  bool reduction_fallback_ = false;
  double condition_estimate_ = 0.0;
  PbctopoLatticeStatus status_ = PbctopoLatticeStatus::InvalidBox;
  // Upper Cholesky factor of the reduced Gram matrix, computed once instead
  // of on every enumerating query.
  double reduced_cholesky_upper_[3][3]{};
  bool reduced_cholesky_valid_ = false;
  // A query whose rounded reduced-basis seed is shorter than this (a quarter
  // of the smallest reduced-cell height squared, with a safety margin) has
  // the seed as its unique minimum image; 0 when not established.
  double unique_minimum_d2_bound_ = 0.0;
};

} // namespace titan_pbctopo

#endif
