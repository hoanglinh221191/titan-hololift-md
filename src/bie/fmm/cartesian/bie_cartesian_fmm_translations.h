#ifndef TITAN_BIE_CARTESIAN_FMM_TRANSLATIONS_H
#define TITAN_BIE_CARTESIAN_FMM_TRANSLATIONS_H

#include "bie_cartesian_fmm_types.h"

#include <array>
#include <span>

namespace titan::bie::fmm::cartesian {

struct TranslationParityAudit {
  double m2m_relative_error{0.0};
  double l2l_relative_error{0.0};
  bool finite_outputs{true};
};

std::size_t coefficient_count(int order);
std::size_t green_jet_coefficient_count(int order);
const std::vector<MultiIndex> &multi_indices(int order);

JcpMoments build_leaf_moments(std::span<const PanelGeometry> panels,
                              std::span<const double> phi,
                              std::span<const double> phin,
                              std::span<const int> panel_indices, Vec3 center,
                              int order);

void accumulate_m2m(JcpMoments &parent, const JcpMoments &child,
                    Vec3 parent_center, Vec3 child_center) noexcept;

void build_green_translation(Vec3 source_center, Vec3 target_center,
                             double kappa, int order, std::span<double> coulomb,
                             std::span<double> yukawa);

void accumulate_m2l(JcpLocals &target, const JcpMoments &source,
                    std::span<const double> coulomb,
                    std::span<const double> yukawa, double eps_ratio,
                    int order) noexcept;

void accumulate_l2l(JcpLocals &child, const JcpLocals &parent,
                    Vec3 parent_center, Vec3 child_center) noexcept;

std::array<double, kJcpChannelCount>
evaluate_l2p(const PanelGeometry &target, Vec3 center,
             const JcpLocals &local) noexcept;

double evaluate_local_scalar(const JcpLocals &local, std::size_t field,
                             Vec3 offset) noexcept;

std::array<double, 3> evaluate_local_gradient(const JcpLocals &local,
                                              std::size_t field,
                                              Vec3 offset) noexcept;

TranslationParityAudit run_translation_parity_audit(int order);

} // namespace titan::bie::fmm::cartesian

#endif // TITAN_BIE_CARTESIAN_FMM_TRANSLATIONS_H
