#include "tabi_fmm_m2l_cartesian.h"

#include "tabi_treecode_internal.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace titan::tabi::fmm_m2l {
namespace {

using TitanPBE::Tabi::Panel;
using TitanPBE::Tabi::detail::MomentIndex;
using TitanPBE::Tabi::detail::NodeMoments;
using TitanPBE::Tabi::detail::TaylorSeries;

void kahan_add(double value, double &sum, double &compensation) noexcept {
  const double adjusted = value - compensation;
  const double next = sum + adjusted;
  compensation = (next - sum) - adjusted;
  sum = next;
}

Panel to_production_panel(const PrototypePanel &input) {
  Panel panel;
  panel.x = input.x;
  panel.y = input.y;
  panel.z = input.z;
  panel.nx = input.nx;
  panel.ny = input.ny;
  panel.nz = input.nz;
  panel.area = input.area;
  return panel;
}

NodeMoments to_node_moments(
    const std::array<double, kCartesianJcpMomentMax> &values) {
  NodeMoments result;
  std::copy(values.begin(), values.end(), result.values.begin());
  return result;
}

} // namespace

std::size_t cartesian_moment_term_count(int order) {
  if (order < 1 || order > TitanPBE::Tabi::detail::kTaylorOrderMax) {
    throw std::invalid_argument("Cartesian JCP moment order must be 1..4");
  }
  return TitanPBE::Tabi::detail::moment_indices(order).size();
}

CartesianJcpMoments build_cartesian_jcp_moments(
    const std::vector<PrototypePanel> &panels,
    const std::vector<double> &phi_values,
    const std::vector<double> &phin_values,
    std::span<const std::size_t> source_indices, double center_x,
    double center_y, double center_z, int order) {
  if (phi_values.size() != panels.size() ||
      phin_values.size() != panels.size()) {
    throw std::invalid_argument(
        "Cartesian JCP densities must match the panel count");
  }
  CartesianJcpMoments moments;
  moments.order = order;
  moments.term_count = cartesian_moment_term_count(order);
  const std::vector<MomentIndex> &indices =
      TitanPBE::Tabi::detail::moment_indices(order);
  std::array<double, kCartesianJcpMomentMax> phin_compensation{};
  std::array<std::array<double, kCartesianJcpMomentMax>, 3>
      normal_compensation{};

  for (const std::size_t panel_index : source_indices) {
    if (panel_index >= panels.size()) {
      throw std::out_of_range("Cartesian JCP source panel index is invalid");
    }
    const PrototypePanel &panel = panels[panel_index];
    // A node patch is summed as its one-point source, centroid and mean
    // normal, as in the degree-one moments and the near loop.
    const bool centre = panel.has_source_quadrature_center;
    const double dx = (centre ? panel.source_x : panel.x) - center_x;
    const double dy = (centre ? panel.source_y : panel.y) - center_y;
    const double dz = (centre ? panel.source_z : panel.z) - center_z;
    const double area_phin = panel.area * phin_values[panel_index];
    const double area_phi = panel.area * phi_values[panel_index];
    for (std::size_t term = 0; term < indices.size(); ++term) {
      const MomentIndex &index = indices[term];
      const double monomial =
          TitanPBE::Tabi::detail::pow_int(dx, index.ax) *
          TitanPBE::Tabi::detail::pow_int(dy, index.ay) *
          TitanPBE::Tabi::detail::pow_int(dz, index.az);
      kahan_add(area_phin * monomial, moments.phin[term],
                phin_compensation[term]);
      const std::array<double, 3> normal =
          centre ? std::array<double, 3>{panel.source_nx, panel.source_ny,
                                         panel.source_nz}
                 : std::array<double, 3>{panel.nx, panel.ny, panel.nz};
      for (std::size_t component = 0; component < normal.size(); ++component) {
        kahan_add(area_phi * normal[component] * monomial,
                  moments.phi_normal[component][term],
                  normal_compensation[component][term]);
      }
    }
  }
  return moments;
}

std::array<double, 4> evaluate_cartesian_jcp_m2p(
    const PrototypePanel &target, double center_x, double center_y,
    double center_z, const CartesianJcpMoments &moments, double eps_ratio,
    double kappa) {
  const int order = moments.order;
  if (moments.term_count != cartesian_moment_term_count(order)) {
    throw std::invalid_argument("Cartesian JCP moment metadata is inconsistent");
  }
  const double safe_eps_ratio = std::max(1.0e-12, eps_ratio);
  const double inv_eps_ratio = 1.0 / safe_eps_ratio;
  const double safe_kappa = std::max(0.0, kappa);
  const Panel production_target = to_production_panel(target);
  const NodeMoments phin = to_node_moments(moments.phin);
  std::array<NodeMoments, 3> phi_normal{};
  for (std::size_t component = 0; component < phi_normal.size(); ++component)
    phi_normal[component] = to_node_moments(moments.phi_normal[component]);

  const TaylorSeries green0 = TitanPBE::Tabi::detail::green_series(
      target.x, target.y, target.z, center_x, center_y, center_z, 0.0, order);
  const TaylorSeries greenk = TitanPBE::Tabi::detail::green_series(
      target.x, target.y, target.z, center_x, center_y, center_z, safe_kappa,
      order);
  const TaylorSeries target0 =
      TitanPBE::Tabi::detail::target_normal_green_series(
          production_target, center_x, center_y, center_z, 0.0, order);
  const TaylorSeries targetk =
      TitanPBE::Tabi::detail::target_normal_green_series(
          production_target, center_x, center_y, center_z, safe_kappa, order);

  std::array<double, 4> channels{};
  channels[0] = TitanPBE::Tabi::detail::taylor_sum_series(
      phin, order,
      TitanPBE::Tabi::detail::series_sub(green0, greenk, order));
  channels[2] = TitanPBE::Tabi::detail::taylor_sum_series(
      phin, order,
      TitanPBE::Tabi::detail::series_sub(
          target0,
          TitanPBE::Tabi::detail::series_scale(targetk, inv_eps_ratio, order),
          order));

  for (int component = 0; component < 3; ++component) {
    const TaylorSeries source0 =
        TitanPBE::Tabi::detail::source_normal_component_green_series(
            target.x, target.y, target.z, center_x, center_y, center_z,
            component, 0.0, order);
    const TaylorSeries sourcek =
        TitanPBE::Tabi::detail::source_normal_component_green_series(
            target.x, target.y, target.z, center_x, center_y, center_z,
            component, safe_kappa, order);
    channels[1] += TitanPBE::Tabi::detail::taylor_sum_series(
        phi_normal[static_cast<std::size_t>(component)], order,
        TitanPBE::Tabi::detail::series_sub(
            TitanPBE::Tabi::detail::series_scale(sourcek, safe_eps_ratio,
                                                  order),
            source0, order));

    const TaylorSeries mixed0 =
        TitanPBE::Tabi::detail::mixed_normal_component_green_series(
            production_target, center_x, center_y, center_z, component, 0.0,
            order);
    const TaylorSeries mixedk =
        TitanPBE::Tabi::detail::mixed_normal_component_green_series(
            production_target, center_x, center_y, center_z, component,
            safe_kappa, order);
    channels[3] += TitanPBE::Tabi::detail::taylor_sum_series(
        phi_normal[static_cast<std::size_t>(component)], order,
        TitanPBE::Tabi::detail::series_sub(mixedk, mixed0, order));
  }

  for (const double value : channels) {
    if (!std::isfinite(value))
      throw std::runtime_error("Cartesian JCP M2P produced a non-finite value");
  }
  return channels;
}

} // namespace titan::tabi::fmm_m2l
