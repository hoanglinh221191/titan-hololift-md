#ifndef TITAN_S2_KERNELS_H
#define TITAN_S2_KERNELS_H

#include "aligned_allocator.h"
#include "box_type.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

class TitanExecutor;

namespace titan_s2_kernel {

struct BondVectorsSoA {
  AlignedDoubleVector x;
  AlignedDoubleVector y;
  AlignedDoubleVector z;
  std::vector<std::uint8_t> valid;

  void clear();
  void reserve(std::size_t n);
  void push(double vx, double vy, double vz, bool is_valid = true);
  [[nodiscard]] std::size_t size() const noexcept { return x.size(); }
};

struct CorrelationRow {
  std::size_t lag_index{0};
  double lag_time_ps{0.0};
  double correlation{std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t samples{0};
};

struct FitResult {
  bool ok{false};
  bool used_tail_fallback{false};
  std::size_t tail_rows{0};
  double raw_plateau{0.0};
  double s2{0.0};
  double tau_ps{0.0};
  double weighted_residual{0.0};
};

// Null or invalid boxes intentionally use raw deltas. The analyzer owns
// warning/reporting for this Phase 6 fallback policy.
void build_bond_vectors_scalar(std::span<const double> from_x,
                               std::span<const double> from_y,
                               std::span<const double> from_z,
                               std::span<const double> to_x,
                               std::span<const double> to_y,
                               std::span<const double> to_z,
                               const box *frame_box, BondVectorsSoA &out);

void build_bond_vectors_optimized(std::span<const double> from_x,
                                  std::span<const double> from_y,
                                  std::span<const double> from_z,
                                  std::span<const double> to_x,
                                  std::span<const double> to_y,
                                  std::span<const double> to_z,
                                  const box *frame_box, BondVectorsSoA &out);

void build_bond_vectors(std::span<const double> from_x,
                        std::span<const double> from_y,
                        std::span<const double> from_z,
                        std::span<const double> to_x,
                        std::span<const double> to_y,
                        std::span<const double> to_z, const box *frame_box,
                        BondVectorsSoA &out);

void rotate_bond_vectors_inplace_scalar(
    BondVectorsSoA &vectors, const std::array<double, 9> &rotation);

void rotate_bond_vectors_inplace_optimized(
    BondVectorsSoA &vectors, const std::array<double, 9> &rotation);

void rotate_bond_vectors_inplace(
    BondVectorsSoA &vectors, const std::array<double, 9> &rotation);

[[nodiscard]] CorrelationRow
compute_correlation_lag_scalar(const BondVectorsSoA &vectors,
                               std::size_t lag);

[[nodiscard]] CorrelationRow
compute_correlation_lag_optimized(const BondVectorsSoA &vectors,
                                  std::size_t lag);

[[nodiscard]] std::vector<CorrelationRow>
compute_correlation_scalar(const BondVectorsSoA &vectors,
                           std::span<const double> time_ps,
                           std::size_t maxlag);

[[nodiscard]] std::vector<CorrelationRow>
compute_correlation(const BondVectorsSoA &vectors,
                    std::span<const double> time_ps, std::size_t maxlag,
                    TitanExecutor &executor);

[[nodiscard]] FitResult
fit_single_exp_plateau(std::span<const CorrelationRow> rows);

} // namespace titan_s2_kernel

#endif // TITAN_S2_KERNELS_H
