#ifndef TITAN_TABI_FMM_M2L_LOCAL_H
#define TITAN_TABI_FMM_M2L_LOCAL_H

#include "tabi_fmm_m2l_cartesian.h"

#include <array>
#include <cstddef>
#include <span>

namespace titan::tabi::fmm_m2l {

inline constexpr int kCartesianLocalSourceOrder = 5;
inline constexpr int kCartesianLocalTargetOrder = 5;
inline constexpr int kCartesianLocalKernelOrder = 10;
inline constexpr int kCartesianCompressedGreenOrder = 11;
inline constexpr std::size_t kCartesianLocalTermMax = 56;
inline constexpr std::size_t kCartesianKernelJetTermCount = 286;
inline constexpr std::size_t kCartesianCompressedGreenJetTermCount = 364;
inline constexpr std::size_t kCartesianJcpKernelJetCount = 8;

enum class JcpLocalField : std::size_t {
  K1 = 0,
  K2 = 1,
  K3Potential = 2,
  K4Potential = 3,
};

struct CartesianJcpLocalP4 {
  std::array<std::array<double, kCartesianLocalTermMax>, 4> fields{};
  bool finite_outputs{true};
};

struct CartesianJcpM2lOperatorP4 {
  std::array<std::array<double, kCartesianKernelJetTermCount>,
             kCartesianJcpKernelJetCount>
      jets{};
  bool finite_outputs{true};
};

struct CartesianJcpCompressedGreenOperatorP4 {
  std::array<double, kCartesianCompressedGreenJetTermCount> coulomb{};
  std::array<double, kCartesianCompressedGreenJetTermCount> yukawa{};
};
static_assert(sizeof(CartesianJcpCompressedGreenOperatorP4) == 5824);

std::size_t cartesian_local_kernel_term_count() noexcept;
std::size_t cartesian_compressed_green_jet_term_count() noexcept;
std::size_t
cartesian_compressed_green_jet_term_count_order(int local_order) noexcept;
std::size_t
cartesian_compressed_green_operator_bytes_order(int local_order) noexcept;
std::size_t cartesian_local_term_count() noexcept;

CartesianJcpLocalP4 translate_cartesian_jcp_m2l_p4(
    const CartesianJcpMoments &moments, double source_center_x,
    double source_center_y, double source_center_z, double target_center_x,
    double target_center_y, double target_center_z, double eps_ratio,
    double kappa);

CartesianJcpM2lOperatorP4 build_cartesian_jcp_m2l_operator_p4(
    double source_center_x, double source_center_y, double source_center_z,
    double target_center_x, double target_center_y, double target_center_z,
    double eps_ratio, double kappa);

CartesianJcpLocalP4 apply_cartesian_jcp_m2l_operator_p4(
    const CartesianJcpMoments &moments,
    const CartesianJcpM2lOperatorP4 &operator_p4);

CartesianJcpLocalP4 apply_cartesian_jcp_m2l_operator_p4_order(
    const CartesianJcpMoments &moments,
    const CartesianJcpM2lOperatorP4 &operator_p4, int local_order);

CartesianJcpCompressedGreenOperatorP4
build_cartesian_jcp_compressed_green_operator_p4(
    double source_center_x, double source_center_y, double source_center_z,
    double target_center_x, double target_center_y, double target_center_z,
    double kappa);

void build_cartesian_jcp_compressed_green_operator_p4_order(
    double source_center_x, double source_center_y, double source_center_z,
    double target_center_x, double target_center_y, double target_center_z,
    double kappa, int local_order, std::span<double> coulomb,
    std::span<double> yukawa);

CartesianJcpLocalP4 apply_cartesian_jcp_compressed_green_operator_p4(
    const CartesianJcpMoments &moments,
    const CartesianJcpCompressedGreenOperatorP4 &operator_p4,
    double eps_ratio);

CartesianJcpLocalP4 apply_cartesian_jcp_compressed_green_operator_p4_order(
    const CartesianJcpMoments &moments,
    const CartesianJcpCompressedGreenOperatorP4 &operator_p4,
    double eps_ratio, int local_order);

CartesianJcpLocalP4 apply_cartesian_jcp_compressed_green_operator_p4_order(
    const CartesianJcpMoments &moments, std::span<const double> coulomb,
    std::span<const double> yukawa, double eps_ratio, int local_order);

// Accumulate a compressed Cartesian JCP translation directly into caller-owned
// local coefficients.  The full-FMM hot path already owns moments and locals
// in the same fixed Cartesian layout, so this avoids constructing and copying
// legacy CartesianJcpMoments/CartesianJcpLocalP4 objects for every M2L pair.
// Translation buffers are validated when the immutable plan is prepared;
// finite-output validation remains an apply-level operation.
bool accumulate_cartesian_jcp_compressed_green_operator_p4_order(
    const std::array<double, kCartesianJcpMomentMax> &phin,
    const std::array<std::array<double, kCartesianJcpMomentMax>, 3>
        &phi_normal,
    std::size_t source_term_count, std::span<const double> coulomb,
    std::span<const double> yukawa, double eps_ratio, int local_order,
    std::array<std::array<double, kCartesianLocalTermMax>, 4>
        &destination) noexcept;

void accumulate_cartesian_jcp_local(
    CartesianJcpLocalP4 &destination,
    const CartesianJcpLocalP4 &source) noexcept;

double evaluate_cartesian_jcp_local_scalar(
    const CartesianJcpLocalP4 &local, JcpLocalField field, double dx,
    double dy, double dz) noexcept;

std::array<double, 3> evaluate_cartesian_jcp_local_gradient(
    const CartesianJcpLocalP4 &local, JcpLocalField field, double dx,
    double dy, double dz,
    int target_order = kCartesianLocalTargetOrder) noexcept;

std::array<double, 4> evaluate_cartesian_jcp_l2p(
    const PrototypePanel &target, double target_center_x,
    double target_center_y, double target_center_z,
    const CartesianJcpLocalP4 &local) noexcept;

std::array<double, 4> evaluate_cartesian_jcp_l2p_order(
    const PrototypePanel &target, double target_center_x,
    double target_center_y, double target_center_z,
    const CartesianJcpLocalP4 &local, int target_order) noexcept;

// [0] at high_order, [1] at low_order, in one pass; equal bit for bit to two
// calls of evaluate_cartesian_jcp_l2p_order().
std::array<std::array<double, 4>, 2> evaluate_cartesian_jcp_l2p_two_orders(
    const PrototypePanel &target, double target_center_x,
    double target_center_y, double target_center_z,
    const CartesianJcpLocalP4 &local, int high_order, int low_order) noexcept;

} // namespace titan::tabi::fmm_m2l

#endif // TITAN_TABI_FMM_M2L_LOCAL_H
