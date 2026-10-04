#ifndef TITAN_TABI_FMM_M2L_CARTESIAN_H
#define TITAN_TABI_FMM_M2L_CARTESIAN_H

#include "tabi_fmm_m2l_prototype.h"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace titan::tabi::fmm_m2l {

inline constexpr std::size_t kCartesianJcpMomentMax = 56;

struct CartesianJcpMoments {
  int order{1};
  std::size_t term_count{4};
  std::array<double, kCartesianJcpMomentMax> phin{};
  std::array<std::array<double, kCartesianJcpMomentMax>, 3> phi_normal{};
};

std::size_t cartesian_moment_term_count(int order);

CartesianJcpMoments build_cartesian_jcp_moments(
    const std::vector<PrototypePanel> &panels,
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values,
    std::span<const std::size_t> source_indices, double center_x,
    double center_y, double center_z, int order);

std::array<double, 4> evaluate_cartesian_jcp_m2p(
    const PrototypePanel &target, double center_x, double center_y,
    double center_z, const CartesianJcpMoments &moments, double eps_ratio,
    double kappa);

} // namespace titan::tabi::fmm_m2l

#endif // TITAN_TABI_FMM_M2L_CARTESIAN_H
