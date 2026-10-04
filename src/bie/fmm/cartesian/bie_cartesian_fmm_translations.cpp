#include "bie_cartesian_fmm_translations.h"

#include "../../../tabi/tabi_fmm_m2l_local.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace titan::bie::fmm::cartesian {
namespace {

double integer_power(double value, int exponent) noexcept {
  double result = 1.0;
  for (int i = 0; i < exponent; ++i)
    result *= value;
  return result;
}

double binomial(int n, int k) noexcept {
  if (k < 0 || k > n)
    return 0.0;
  double result = 1.0;
  for (int i = 1; i <= k; ++i)
    result = result * static_cast<double>(n - k + i) / static_cast<double>(i);
  return result;
}

void kahan_add(double value, double &sum, double &compensation) noexcept {
  const double adjusted = value - compensation;
  const double next = sum + adjusted;
  compensation = (next - sum) - adjusted;
  sum = next;
}

int lookup_index(int order, int x, int y, int z) noexcept {
  if (x < 0 || y < 0 || z < 0 || x + y + z > order)
    return -1;
  const auto &indices = multi_indices(order);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (indices[i].x == x && indices[i].y == y && indices[i].z == z)
      return static_cast<int>(i);
  }
  return -1;
}

bool finite_moments(const JcpMoments &moments) noexcept {
  for (std::size_t i = 0; i < moments.term_count; ++i) {
    if (!std::isfinite(moments.phin[i]))
      return false;
    for (const auto &component : moments.phi_normal) {
      if (!std::isfinite(component[i]))
        return false;
    }
  }
  return true;
}

bool finite_locals(const JcpLocals &local) noexcept {
  for (const auto &field : local.fields) {
    for (std::size_t i = 0; i < local.term_count; ++i) {
      if (!std::isfinite(field[i]))
        return false;
    }
  }
  return true;
}

double relative_l2(double diff_squared, double reference_squared) noexcept {
  return std::sqrt(std::max(0.0, diff_squared)) /
         std::max(1.0e-300, std::sqrt(std::max(0.0, reference_squared)));
}

} // namespace

std::size_t coefficient_count(int order) {
  if (order != 1 && order != 4)
    throw std::invalid_argument(
        "BIE Cartesian full FMM expansion order must be 1 or 4");
  return static_cast<std::size_t>((order + 1) * (order + 2) * (order + 3) / 6);
}

std::size_t green_jet_coefficient_count(int order) {
  if (order != 1 && order != 4)
    throw std::invalid_argument(
        "BIE Cartesian full FMM expansion order must be 1 or 4");
  return titan::tabi::fmm_m2l::cartesian_compressed_green_jet_term_count_order(
      order);
}

const std::vector<MultiIndex> &multi_indices(int order) {
  static const std::array<std::vector<MultiIndex>, kMaximumExpansionOrder + 1>
      tables = [] {
        std::array<std::vector<MultiIndex>, kMaximumExpansionOrder + 1>
            result{};
        for (int p = 0; p <= kMaximumExpansionOrder; ++p) {
          for (int degree = 0; degree <= p; ++degree) {
            for (int x = degree; x >= 0; --x) {
              for (int y = degree - x; y >= 0; --y)
                result[static_cast<std::size_t>(p)].push_back(
                    {x, y, degree - x - y});
            }
          }
        }
        return result;
      }();
  const int clamped = std::clamp(order, 0, kMaximumExpansionOrder);
  return tables[static_cast<std::size_t>(clamped)];
}

JcpMoments build_leaf_moments(std::span<const PanelGeometry> panels,
                              std::span<const double> phi,
                              std::span<const double> phin,
                              std::span<const int> panel_indices, Vec3 center,
                              int order) {
  if (phi.size() != panels.size() || phin.size() != panels.size())
    throw std::invalid_argument(
        "BIE Cartesian full FMM density and panel counts differ");
  JcpMoments result;
  result.order = order;
  result.term_count = coefficient_count(order);
  const auto &indices = multi_indices(order);
  std::array<double, kMaximumCoefficientCount> phin_compensation{};
  std::array<std::array<double, kMaximumCoefficientCount>, 3>
      normal_compensation{};
  for (const int raw_index : panel_indices) {
    if (raw_index < 0 || static_cast<std::size_t>(raw_index) >= panels.size())
      throw std::out_of_range("BIE Cartesian full FMM panel index is invalid");
    const std::size_t panel_index = static_cast<std::size_t>(raw_index);
    const PanelGeometry &panel = panels[panel_index];
    const double dx = panel.source.x - center.x;
    const double dy = panel.source.y - center.y;
    const double dz = panel.source.z - center.z;
    const double weighted_phin = panel.area * phin[panel_index];
    const double weighted_phi = panel.area * phi[panel_index];
    const std::array<double, 3> normal{panel.normal.x, panel.normal.y,
                                       panel.normal.z};
    for (std::size_t term = 0; term < indices.size(); ++term) {
      const MultiIndex &index = indices[term];
      const double monomial = integer_power(dx, index.x) *
                              integer_power(dy, index.y) *
                              integer_power(dz, index.z);
      kahan_add(weighted_phin * monomial, result.phin[term],
                phin_compensation[term]);
      for (std::size_t component = 0; component < normal.size(); ++component) {
        kahan_add(weighted_phi * normal[component] * monomial,
                  result.phi_normal[component][term],
                  normal_compensation[component][term]);
      }
    }
  }
  result.finite_outputs = finite_moments(result);
  return result;
}

void accumulate_m2m(JcpMoments &parent, const JcpMoments &child,
                    Vec3 parent_center, Vec3 child_center) noexcept {
  if ((parent.order != 1 && parent.order != 4) || parent.order != child.order ||
      parent.term_count != child.term_count) {
    parent.finite_outputs = false;
    return;
  }
  const int order = parent.order;
  const auto &indices = multi_indices(order);
  const Vec3 shift{child_center.x - parent_center.x,
                   child_center.y - parent_center.y,
                   child_center.z - parent_center.z};
  for (std::size_t alpha_index = 0; alpha_index < indices.size();
       ++alpha_index) {
    const MultiIndex &alpha = indices[alpha_index];
    for (int bx = 0; bx <= alpha.x; ++bx) {
      for (int by = 0; by <= alpha.y; ++by) {
        for (int bz = 0; bz <= alpha.z; ++bz) {
          const int beta_index = lookup_index(order, bx, by, bz);
          if (beta_index < 0)
            continue;
          const double factor = binomial(alpha.x, bx) * binomial(alpha.y, by) *
                                binomial(alpha.z, bz) *
                                integer_power(shift.x, alpha.x - bx) *
                                integer_power(shift.y, alpha.y - by) *
                                integer_power(shift.z, alpha.z - bz);
          const std::size_t beta = static_cast<std::size_t>(beta_index);
          parent.phin[alpha_index] += factor * child.phin[beta];
          for (std::size_t component = 0; component < 3; ++component) {
            parent.phi_normal[component][alpha_index] +=
                factor * child.phi_normal[component][beta];
          }
        }
      }
    }
  }
  parent.finite_outputs =
      parent.finite_outputs && child.finite_outputs && finite_moments(parent);
}

void build_green_translation(Vec3 source_center, Vec3 target_center,
                             double kappa, int order, std::span<double> coulomb,
                             std::span<double> yukawa) {
  const std::size_t required = green_jet_coefficient_count(order);
  if (coulomb.size() < required || yukawa.size() < required)
    throw std::invalid_argument(
        "BIE Cartesian full FMM translation storage is too small");
  titan::tabi::fmm_m2l::build_cartesian_jcp_compressed_green_operator_p4_order(
      source_center.x, source_center.y, source_center.z, target_center.x,
      target_center.y, target_center.z, kappa, order, coulomb.first(required),
      yukawa.first(required));
}

void accumulate_m2l(JcpLocals &target, const JcpMoments &source,
                    std::span<const double> coulomb,
                    std::span<const double> yukawa, double eps_ratio,
                    int order) noexcept {
  if (target.order != order || source.order != order ||
      target.term_count != source.term_count || !source.finite_outputs ||
      !titan::tabi::fmm_m2l::
          accumulate_cartesian_jcp_compressed_green_operator_p4_order(
              source.phin, source.phi_normal, source.term_count, coulomb,
              yukawa, eps_ratio, order, target.fields)) {
    target.finite_outputs = false;
  }
}

void accumulate_l2l(JcpLocals &child, const JcpLocals &parent,
                    Vec3 parent_center, Vec3 child_center) noexcept {
  if ((child.order != 1 && child.order != 4) || child.order != parent.order ||
      child.term_count != parent.term_count) {
    child.finite_outputs = false;
    return;
  }
  const int order = child.order;
  const auto &indices = multi_indices(order);
  const Vec3 shift{child_center.x - parent_center.x,
                   child_center.y - parent_center.y,
                   child_center.z - parent_center.z};
  for (std::size_t alpha_index = 0; alpha_index < indices.size();
       ++alpha_index) {
    const MultiIndex &alpha = indices[alpha_index];
    for (int bx = 0; bx <= alpha.x; ++bx) {
      for (int by = 0; by <= alpha.y; ++by) {
        for (int bz = 0; bz <= alpha.z; ++bz) {
          const int beta_index = lookup_index(order, bx, by, bz);
          if (beta_index < 0)
            continue;
          const double factor = binomial(alpha.x, bx) * binomial(alpha.y, by) *
                                binomial(alpha.z, bz) *
                                integer_power(shift.x, alpha.x - bx) *
                                integer_power(shift.y, alpha.y - by) *
                                integer_power(shift.z, alpha.z - bz);
          for (std::size_t field = 0; field < kJcpChannelCount; ++field) {
            child.fields[field][static_cast<std::size_t>(beta_index)] +=
                factor * parent.fields[field][alpha_index];
          }
        }
      }
    }
  }
  child.finite_outputs =
      child.finite_outputs && parent.finite_outputs && finite_locals(child);
}

double evaluate_local_scalar(const JcpLocals &local, std::size_t field,
                             Vec3 offset) noexcept {
  if (field >= local.fields.size())
    return std::numeric_limits<double>::quiet_NaN();
  const auto &indices = multi_indices(local.order);
  double value = 0.0;
  for (std::size_t term = 0; term < local.term_count; ++term) {
    const MultiIndex &index = indices[term];
    value += local.fields[field][term] * integer_power(offset.x, index.x) *
             integer_power(offset.y, index.y) *
             integer_power(offset.z, index.z);
  }
  return value;
}

std::array<double, 3> evaluate_local_gradient(const JcpLocals &local,
                                              std::size_t field,
                                              Vec3 offset) noexcept {
  std::array<double, 3> gradient{};
  if (field >= local.fields.size()) {
    gradient.fill(std::numeric_limits<double>::quiet_NaN());
    return gradient;
  }
  const auto &indices = multi_indices(local.order);
  for (std::size_t term = 0; term < local.term_count; ++term) {
    const MultiIndex &index = indices[term];
    const double coefficient = local.fields[field][term];
    if (index.x > 0) {
      gradient[0] += static_cast<double>(index.x) * coefficient *
                     integer_power(offset.x, index.x - 1) *
                     integer_power(offset.y, index.y) *
                     integer_power(offset.z, index.z);
    }
    if (index.y > 0) {
      gradient[1] += static_cast<double>(index.y) * coefficient *
                     integer_power(offset.x, index.x) *
                     integer_power(offset.y, index.y - 1) *
                     integer_power(offset.z, index.z);
    }
    if (index.z > 0) {
      gradient[2] += static_cast<double>(index.z) * coefficient *
                     integer_power(offset.x, index.x) *
                     integer_power(offset.y, index.y) *
                     integer_power(offset.z, index.z - 1);
    }
  }
  return gradient;
}

std::array<double, kJcpChannelCount>
evaluate_l2p(const PanelGeometry &target, Vec3 center,
             const JcpLocals &local) noexcept {
  const Vec3 offset{target.target.x - center.x, target.target.y - center.y,
                    target.target.z - center.z};
  std::array<double, kJcpChannelCount> channels{};
  channels[0] = evaluate_local_scalar(local, 0, offset);
  channels[1] = evaluate_local_scalar(local, 1, offset);
  const auto k3 = evaluate_local_gradient(local, 2, offset);
  const auto k4 = evaluate_local_gradient(local, 3, offset);
  channels[2] = target.normal.x * k3[0] + target.normal.y * k3[1] +
                target.normal.z * k3[2];
  channels[3] = target.normal.x * k4[0] + target.normal.y * k4[1] +
                target.normal.z * k4[2];
  return channels;
}

TranslationParityAudit run_translation_parity_audit(int order) {
  TranslationParityAudit audit;
  try {
    const std::size_t terms = coefficient_count(order);
    std::vector<PanelGeometry> panels(12);
    std::vector<double> phi(panels.size());
    std::vector<double> phin(panels.size());
    std::vector<int> all(panels.size());
    std::array<std::vector<int>, 2> child_indices;
    const std::array<Vec3, 2> child_centers{Vec3{-0.55, 0.30, -0.20},
                                            Vec3{0.45, -0.35, 0.25}};
    for (std::size_t i = 0; i < panels.size(); ++i) {
      const std::size_t child = i < panels.size() / 2 ? 0 : 1;
      const double t = static_cast<double>(i + 1);
      panels[i].source = {child_centers[child].x + 0.031 * std::sin(0.7 * t),
                          child_centers[child].y + 0.027 * std::cos(0.5 * t),
                          child_centers[child].z + 0.019 * std::sin(0.9 * t)};
      panels[i].target = panels[i].source;
      panels[i].normal = {0.3 + 0.01 * t, -0.2 + 0.007 * t, 0.8 - 0.009 * t};
      panels[i].area = 0.7 + 0.03 * t;
      phi[i] = std::sin(0.37 * t);
      phin[i] = std::cos(0.29 * t);
      all[i] = static_cast<int>(i);
      child_indices[child].push_back(static_cast<int>(i));
    }
    const Vec3 parent_center{0.08, -0.04, 0.03};
    const JcpMoments direct =
        build_leaf_moments(panels, phi, phin, all, parent_center, order);
    JcpMoments translated;
    translated.order = order;
    translated.term_count = terms;
    for (std::size_t child = 0; child < child_indices.size(); ++child) {
      const JcpMoments child_moments = build_leaf_moments(
          panels, phi, phin, child_indices[child], child_centers[child], order);
      accumulate_m2m(translated, child_moments, parent_center,
                     child_centers[child]);
    }
    double m2m_diff_sq = 0.0;
    double m2m_ref_sq = 0.0;
    for (std::size_t term = 0; term < terms; ++term) {
      const double phin_diff = translated.phin[term] - direct.phin[term];
      m2m_diff_sq += phin_diff * phin_diff;
      m2m_ref_sq += direct.phin[term] * direct.phin[term];
      for (std::size_t component = 0; component < 3; ++component) {
        const double diff = translated.phi_normal[component][term] -
                            direct.phi_normal[component][term];
        m2m_diff_sq += diff * diff;
        m2m_ref_sq += direct.phi_normal[component][term] *
                      direct.phi_normal[component][term];
      }
    }
    audit.m2m_relative_error = relative_l2(m2m_diff_sq, m2m_ref_sq);

    JcpLocals parent_local;
    parent_local.order = order;
    parent_local.term_count = terms;
    for (std::size_t field = 0; field < kJcpChannelCount; ++field) {
      for (std::size_t term = 0; term < terms; ++term) {
        parent_local.fields[field][term] =
            std::sin(0.13 * static_cast<double>((field + 1) * (term + 1)));
      }
    }
    JcpLocals child_local;
    child_local.order = order;
    child_local.term_count = terms;
    const Vec3 local_child_center{0.22, -0.17, 0.14};
    accumulate_l2l(child_local, parent_local, parent_center,
                   local_child_center);
    double l2l_diff_sq = 0.0;
    double l2l_ref_sq = 0.0;
    for (int sample = 0; sample < 6; ++sample) {
      const double s = static_cast<double>(sample + 1);
      const Vec3 child_offset{0.011 * s, -0.009 * s, 0.007 * s};
      const Vec3 parent_offset{
          local_child_center.x + child_offset.x - parent_center.x,
          local_child_center.y + child_offset.y - parent_center.y,
          local_child_center.z + child_offset.z - parent_center.z};
      for (std::size_t field = 0; field < kJcpChannelCount; ++field) {
        const double reference =
            evaluate_local_scalar(parent_local, field, parent_offset);
        const double candidate =
            evaluate_local_scalar(child_local, field, child_offset);
        const double diff = candidate - reference;
        l2l_diff_sq += diff * diff;
        l2l_ref_sq += reference * reference;
        const auto reference_gradient =
            evaluate_local_gradient(parent_local, field, parent_offset);
        const auto candidate_gradient =
            evaluate_local_gradient(child_local, field, child_offset);
        for (std::size_t component = 0; component < 3; ++component) {
          const double gradient_diff =
              candidate_gradient[component] - reference_gradient[component];
          l2l_diff_sq += gradient_diff * gradient_diff;
          l2l_ref_sq +=
              reference_gradient[component] * reference_gradient[component];
        }
      }
    }
    audit.l2l_relative_error = relative_l2(l2l_diff_sq, l2l_ref_sq);
    audit.finite_outputs = direct.finite_outputs && translated.finite_outputs &&
                           parent_local.finite_outputs &&
                           child_local.finite_outputs &&
                           std::isfinite(audit.m2m_relative_error) &&
                           std::isfinite(audit.l2l_relative_error);
  } catch (...) {
    audit.finite_outputs = false;
    audit.m2m_relative_error = std::numeric_limits<double>::infinity();
    audit.l2l_relative_error = std::numeric_limits<double>::infinity();
  }
  return audit;
}

} // namespace titan::bie::fmm::cartesian
