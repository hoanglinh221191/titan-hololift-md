#include "tabi_fmm_m2l_local.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace titan::tabi::fmm_m2l {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kJetSide = kCartesianCompressedGreenOrder + 1;
constexpr std::size_t kJetSize =
    static_cast<std::size_t>(kJetSide * kJetSide * kJetSide);

struct MultiIndex {
  int x{0};
  int y{0};
  int z{0};
};

struct Jet8 {
  std::array<double, kJetSize> coeff{};
};

std::size_t cube_index(int x, int y, int z) noexcept {
  return static_cast<std::size_t>((x * kJetSide + y) * kJetSide + z);
}

const std::vector<MultiIndex> &indices_through(int order) {
  static const std::array<std::vector<MultiIndex>,
                          kCartesianCompressedGreenOrder + 1>
      tables = [] {
        std::array<std::vector<MultiIndex>,
                   kCartesianCompressedGreenOrder + 1>
            result{};
        for (int order_index = 0; order_index <= kCartesianCompressedGreenOrder;
             ++order_index) {
          result[static_cast<std::size_t>(order_index)].reserve(
              kCartesianCompressedGreenJetTermCount);
          for (int degree = 0; degree <= order_index; ++degree) {
            for (int x = degree; x >= 0; --x) {
              for (int y = degree - x; y >= 0; --y)
                result[static_cast<std::size_t>(order_index)].push_back(
                    {x, y, degree - x - y});
            }
          }
        }
        return result;
      }();
  const int clamped_order =
      std::clamp(order, 0, kCartesianCompressedGreenOrder);
  return tables[static_cast<std::size_t>(clamped_order)];
}

double binomial(int n, int k) noexcept {
  if (k < 0 || k > n)
    return 0.0;
  double value = 1.0;
  for (int i = 1; i <= k; ++i)
    value = value * static_cast<double>(n - k + i) / static_cast<double>(i);
  return value;
}

double power(double value, int exponent) noexcept {
  double result = 1.0;
  for (int i = 0; i < exponent; ++i)
    result *= value;
  return result;
}

Jet8 constant(double value) noexcept {
  Jet8 result;
  result.coeff[cube_index(0, 0, 0)] = value;
  return result;
}

Jet8 variable(double value, int axis) noexcept {
  Jet8 result = constant(value);
  result.coeff[cube_index(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0,
                           axis == 2 ? 1 : 0)] = 1.0;
  return result;
}

Jet8 add(const Jet8 &a, const Jet8 &b) noexcept {
  Jet8 result;
  for (const MultiIndex &index : indices_through(kCartesianCompressedGreenOrder)) {
    const std::size_t i = cube_index(index.x, index.y, index.z);
    result.coeff[i] = a.coeff[i] + b.coeff[i];
  }
  return result;
}

Jet8 subtract(const Jet8 &a, const Jet8 &b) noexcept {
  Jet8 result;
  for (const MultiIndex &index : indices_through(kCartesianCompressedGreenOrder)) {
    const std::size_t i = cube_index(index.x, index.y, index.z);
    result.coeff[i] = a.coeff[i] - b.coeff[i];
  }
  return result;
}

Jet8 scale(const Jet8 &a, double factor) noexcept {
  Jet8 result;
  for (const MultiIndex &index : indices_through(kCartesianCompressedGreenOrder)) {
    const std::size_t i = cube_index(index.x, index.y, index.z);
    result.coeff[i] = factor * a.coeff[i];
  }
  return result;
}

Jet8 multiply(const Jet8 &a, const Jet8 &b) noexcept {
  Jet8 result;
  for (const MultiIndex &index : indices_through(kCartesianCompressedGreenOrder)) {
    double value = 0.0;
    for (int x = 0; x <= index.x; ++x) {
      for (int y = 0; y <= index.y; ++y) {
        for (int z = 0; z <= index.z; ++z) {
          value += a.coeff[cube_index(x, y, z)] *
                   b.coeff[cube_index(index.x - x, index.y - y,
                                      index.z - z)];
        }
      }
    }
    result.coeff[cube_index(index.x, index.y, index.z)] = value;
  }
  return result;
}

Jet8 reciprocal(const Jet8 &a) noexcept {
  Jet8 result;
  const double a0 = a.coeff[cube_index(0, 0, 0)];
  if (!std::isfinite(a0) || std::abs(a0) <= 1.0e-30)
    return result;
  result.coeff[cube_index(0, 0, 0)] = 1.0 / a0;
  for (const MultiIndex &index : indices_through(kCartesianCompressedGreenOrder)) {
    if (index.x == 0 && index.y == 0 && index.z == 0)
      continue;
    double sum = 0.0;
    for (int x = 0; x <= index.x; ++x) {
      for (int y = 0; y <= index.y; ++y) {
        for (int z = 0; z <= index.z; ++z) {
          if (x == 0 && y == 0 && z == 0)
            continue;
          sum += a.coeff[cube_index(x, y, z)] *
                 result.coeff[cube_index(index.x - x, index.y - y,
                                          index.z - z)];
        }
      }
    }
    result.coeff[cube_index(index.x, index.y, index.z)] = -sum / a0;
  }
  return result;
}

Jet8 square_root(const Jet8 &a) noexcept {
  Jet8 result;
  const double a0 = a.coeff[cube_index(0, 0, 0)];
  if (!std::isfinite(a0) || a0 <= 0.0)
    return result;
  const double root = std::sqrt(a0);
  result.coeff[cube_index(0, 0, 0)] = root;
  for (const MultiIndex &index : indices_through(kCartesianCompressedGreenOrder)) {
    if (index.x == 0 && index.y == 0 && index.z == 0)
      continue;
    double sum = 0.0;
    for (int x = 0; x <= index.x; ++x) {
      for (int y = 0; y <= index.y; ++y) {
        for (int z = 0; z <= index.z; ++z) {
          if ((x == 0 && y == 0 && z == 0) ||
              (x == index.x && y == index.y && z == index.z))
            continue;
          sum += result.coeff[cube_index(x, y, z)] *
                 result.coeff[cube_index(index.x - x, index.y - y,
                                          index.z - z)];
        }
      }
    }
    const std::size_t i = cube_index(index.x, index.y, index.z);
    result.coeff[i] = (a.coeff[i] - sum) / (2.0 * root);
  }
  return result;
}

Jet8 exponential(const Jet8 &a) noexcept {
  Jet8 result;
  result.coeff[cube_index(0, 0, 0)] =
      std::exp(a.coeff[cube_index(0, 0, 0)]);
  for (const MultiIndex &index : indices_through(kCartesianCompressedGreenOrder)) {
    if (index.x == 0 && index.y == 0 && index.z == 0)
      continue;
    int axis = index.x > 0 ? 0 : (index.y > 0 ? 1 : 2);
    const int gx = index.x - (axis == 0 ? 1 : 0);
    const int gy = index.y - (axis == 1 ? 1 : 0);
    const int gz = index.z - (axis == 2 ? 1 : 0);
    double rhs = 0.0;
    for (int x = 0; x <= gx; ++x) {
      for (int y = 0; y <= gy; ++y) {
        for (int z = 0; z <= gz; ++z) {
          const int dx = gx - x + (axis == 0 ? 1 : 0);
          const int dy = gy - y + (axis == 1 ? 1 : 0);
          const int dz = gz - z + (axis == 2 ? 1 : 0);
          const int axis_degree = axis == 0 ? dx : (axis == 1 ? dy : dz);
          rhs += result.coeff[cube_index(x, y, z)] *
                 static_cast<double>(axis_degree) *
                 a.coeff[cube_index(dx, dy, dz)];
        }
      }
    }
    const int degree = axis == 0 ? index.x : (axis == 1 ? index.y : index.z);
    result.coeff[cube_index(index.x, index.y, index.z)] =
        rhs / static_cast<double>(degree);
  }
  return result;
}

struct KernelBase {
  Jet8 ux;
  Jet8 uy;
  Jet8 uz;
  Jet8 r;
  Jet8 inv_r;
  Jet8 inv_r2;
  Jet8 inv_r3;
  Jet8 screened;
};

KernelBase kernel_base(double tx, double ty, double tz, double sx, double sy,
                       double sz, double kappa) noexcept {
  KernelBase base;
  const Jet8 source_x = variable(sx, 0);
  const Jet8 source_y = variable(sy, 1);
  const Jet8 source_z = variable(sz, 2);
  base.ux = subtract(constant(tx), source_x);
  base.uy = subtract(constant(ty), source_y);
  base.uz = subtract(constant(tz), source_z);
  const Jet8 r2 = add(add(multiply(base.ux, base.ux),
                          multiply(base.uy, base.uy)),
                      multiply(base.uz, base.uz));
  if (r2.coeff[cube_index(0, 0, 0)] <= 1.0e-28)
    return base;
  base.r = square_root(r2);
  base.inv_r = reciprocal(base.r);
  base.inv_r2 = multiply(base.inv_r, base.inv_r);
  base.inv_r3 = multiply(base.inv_r2, base.inv_r);
  base.screened = exponential(scale(base.r, -std::max(0.0, kappa)));
  return base;
}

Jet8 green_jet(double tx, double ty, double tz, double sx, double sy,
               double sz, double kappa) noexcept {
  const KernelBase base = kernel_base(tx, ty, tz, sx, sy, sz, kappa);
  return scale(multiply(base.screened, base.inv_r), 1.0 / (4.0 * kPi));
}

Jet8 source_normal_component_jet(double tx, double ty, double tz, double sx,
                                 double sy, double sz, int component,
                                 double kappa) noexcept {
  const double safe_kappa = std::max(0.0, kappa);
  const KernelBase base =
      kernel_base(tx, ty, tz, sx, sy, sz, safe_kappa);
  const Jet8 one_plus_kr =
      add(constant(1.0), scale(base.r, safe_kappa));
  const Jet8 &u = component == 0 ? base.ux : (component == 1 ? base.uy : base.uz);
  return scale(multiply(multiply(base.screened, one_plus_kr),
                        multiply(u, base.inv_r3)),
               1.0 / (4.0 * kPi));
}

std::array<double, kCartesianKernelJetTermCount>
compact_jet(const Jet8 &jet) noexcept {
  std::array<double, kCartesianKernelJetTermCount> compact{};
  const std::vector<MultiIndex> &indices =
      indices_through(kCartesianLocalKernelOrder);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const MultiIndex &index = indices[i];
    compact[i] = jet.coeff[cube_index(index.x, index.y, index.z)];
  }
  return compact;
}

std::array<double, kCartesianCompressedGreenJetTermCount>
compact_green_jet(const Jet8 &jet) noexcept {
  std::array<double, kCartesianCompressedGreenJetTermCount> compact{};
  const std::vector<MultiIndex> &indices =
      indices_through(kCartesianCompressedGreenOrder);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const MultiIndex &index = indices[i];
    compact[i] = jet.coeff[cube_index(index.x, index.y, index.z)];
  }
  return compact;
}

void compact_green_jet_order(const Jet8 &jet, int local_order,
                             std::span<double> compact) noexcept {
  const int order = std::clamp(2 * local_order + 1, 0,
                               kCartesianCompressedGreenOrder);
  const std::vector<MultiIndex> &indices = indices_through(order);
  if (compact.size() < indices.size())
    return;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const MultiIndex &index = indices[i];
    compact[i] = jet.coeff[cube_index(index.x, index.y, index.z)];
  }
}

std::size_t compact_index(int x, int y, int z) noexcept {
  static const std::array<std::size_t, kJetSize> lookup = [] {
    std::array<std::size_t, kJetSize> result{};
    result.fill(kCartesianKernelJetTermCount);
    std::size_t compact = 0;
    for (int degree = 0; degree <= kCartesianLocalKernelOrder; ++degree) {
      for (int ix = degree; ix >= 0; --ix) {
        for (int iy = degree - ix; iy >= 0; --iy)
          result[cube_index(ix, iy, degree - ix - iy)] = compact++;
      }
    }
    return result;
  }();
  return lookup[cube_index(x, y, z)];
}

std::size_t compact_green_index(int x, int y, int z) noexcept {
  static const std::array<std::size_t, kJetSize> lookup = [] {
    std::array<std::size_t, kJetSize> result{};
    result.fill(kCartesianCompressedGreenJetTermCount);
    std::size_t compact = 0;
    for (int degree = 0; degree <= kCartesianCompressedGreenOrder; ++degree) {
      for (int ix = degree; ix >= 0; --ix) {
        for (int iy = degree - ix; iy >= 0; --iy)
          result[cube_index(ix, iy, degree - ix - iy)] = compact++;
      }
    }
    return result;
  }();
  return lookup[cube_index(x, y, z)];
}

struct CompressedTranslationStencil {
  double factor{0.0};
  std::size_t green_index{0};
  std::array<std::size_t, 3> derivative_index{};
  std::array<double, 3> derivative_degree{};
};

const std::array<CompressedTranslationStencil,
                 kCartesianLocalTermMax * kCartesianLocalTermMax> &
compressed_translation_stencils() {
  static const auto stencils = [] {
    std::array<CompressedTranslationStencil,
               kCartesianLocalTermMax * kCartesianLocalTermMax>
        result{};
    const std::vector<MultiIndex> indices =
        indices_through(kCartesianLocalSourceOrder);
    for (std::size_t b = 0; b < indices.size(); ++b) {
      const MultiIndex &beta = indices[b];
      const double sign =
          ((beta.x + beta.y + beta.z) % 2) == 0 ? 1.0 : -1.0;
      for (std::size_t a = 0; a < indices.size(); ++a) {
        const MultiIndex &alpha = indices[a];
        const int gx = alpha.x + beta.x;
        const int gy = alpha.y + beta.y;
        const int gz = alpha.z + beta.z;
        auto &stencil = result[b * kCartesianLocalTermMax + a];
        stencil.factor =
            sign * binomial(gx, alpha.x) * binomial(gy, alpha.y) *
            binomial(gz, alpha.z);
        stencil.green_index = compact_green_index(gx, gy, gz);
        stencil.derivative_index = {
            compact_green_index(gx + 1, gy, gz),
            compact_green_index(gx, gy + 1, gz),
            compact_green_index(gx, gy, gz + 1)};
        stencil.derivative_degree = {
            static_cast<double>(gx + 1), static_cast<double>(gy + 1),
            static_cast<double>(gz + 1)};
      }
    }
    return result;
  }();
  return stencils;
}

bool valid_local_order(int order) noexcept {
  return order >= 1 && order <= kCartesianLocalSourceOrder;
}

void translate_moments_compact(
    const std::array<double, kCartesianJcpMomentMax> &moments,
    const std::array<double, kCartesianKernelJetTermCount> &kernel,
    std::array<double, kCartesianLocalTermMax> &local, int source_order,
    int target_order) {
  const std::vector<MultiIndex> &source =
      indices_through(std::clamp(source_order, 0, kCartesianLocalSourceOrder));
  const std::vector<MultiIndex> &target =
      indices_through(std::clamp(target_order, 0, kCartesianLocalTargetOrder));
  for (std::size_t b = 0; b < target.size(); ++b) {
    const MultiIndex &beta = target[b];
    const double sign = ((beta.x + beta.y + beta.z) % 2) == 0 ? 1.0 : -1.0;
    double value = 0.0;
    for (std::size_t a = 0; a < source.size(); ++a) {
      const MultiIndex &alpha = source[a];
      const int gx = alpha.x + beta.x;
      const int gy = alpha.y + beta.y;
      const int gz = alpha.z + beta.z;
      const std::size_t kernel_index = compact_index(gx, gy, gz);
      const double factor =
          sign * binomial(gx, alpha.x) * binomial(gy, alpha.y) *
          binomial(gz, alpha.z);
      value += moments[a] * factor * kernel[kernel_index];
    }
    local[b] += value;
  }
}

double evaluate_scalar(const std::array<double, kCartesianLocalTermMax> &local,
                       double dx, double dy, double dz,
                       int target_order = kCartesianLocalTargetOrder) noexcept {
  const std::vector<MultiIndex> &indices =
      indices_through(std::clamp(target_order, 0, kCartesianLocalTargetOrder));
  double value = 0.0;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const MultiIndex &index = indices[i];
    value += local[i] * power(dx, index.x) * power(dy, index.y) *
             power(dz, index.z);
  }
  return value;
}

std::array<double, 3>
evaluate_gradient(const std::array<double, kCartesianLocalTermMax> &local,
                  double dx, double dy, double dz,
                  int target_order = kCartesianLocalTargetOrder) noexcept {
  const std::vector<MultiIndex> &indices =
      indices_through(std::clamp(target_order, 0, kCartesianLocalTargetOrder));
  std::array<double, 3> gradient{};
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const MultiIndex &index = indices[i];
    if (index.x > 0)
      gradient[0] += static_cast<double>(index.x) * local[i] *
                     power(dx, index.x - 1) * power(dy, index.y) *
                     power(dz, index.z);
    if (index.y > 0)
      gradient[1] += static_cast<double>(index.y) * local[i] *
                     power(dx, index.x) * power(dy, index.y - 1) *
                     power(dz, index.z);
    if (index.z > 0)
      gradient[2] += static_cast<double>(index.z) * local[i] *
                     power(dx, index.x) * power(dy, index.y) *
                     power(dz, index.z - 1);
  }
  return gradient;
}

} // namespace

std::size_t cartesian_local_kernel_term_count() noexcept {
  return kCartesianKernelJetTermCount;
}

std::size_t cartesian_compressed_green_jet_term_count() noexcept {
  return kCartesianCompressedGreenJetTermCount;
}

std::size_t
cartesian_compressed_green_jet_term_count_order(int local_order) noexcept {
  const int order = std::clamp(2 * local_order + 1, 0,
                               kCartesianCompressedGreenOrder);
  return indices_through(order).size();
}

std::size_t
cartesian_compressed_green_operator_bytes_order(int local_order) noexcept {
  return 2 * cartesian_compressed_green_jet_term_count_order(local_order) *
         sizeof(double);
}

std::size_t cartesian_local_term_count() noexcept {
  return kCartesianLocalTermMax;
}

CartesianJcpLocalP4 translate_cartesian_jcp_m2l_p4(
    const CartesianJcpMoments &moments, double source_center_x,
    double source_center_y, double source_center_z, double target_center_x,
    double target_center_y, double target_center_z, double eps_ratio,
    double kappa) {
  return apply_cartesian_jcp_m2l_operator_p4(
      moments, build_cartesian_jcp_m2l_operator_p4(
                   source_center_x, source_center_y, source_center_z,
                   target_center_x, target_center_y, target_center_z,
                   eps_ratio, kappa));
}

CartesianJcpM2lOperatorP4 build_cartesian_jcp_m2l_operator_p4(
    double source_center_x, double source_center_y, double source_center_z,
    double target_center_x, double target_center_y, double target_center_z,
    double eps_ratio, double kappa) {
  CartesianJcpM2lOperatorP4 result;
  const double safe_eps = std::max(1.0e-12, eps_ratio);
  const Jet8 green0 = green_jet(target_center_x, target_center_y,
                                target_center_z, source_center_x,
                                source_center_y, source_center_z, 0.0);
  const Jet8 greenk = green_jet(target_center_x, target_center_y,
                                target_center_z, source_center_x,
                                source_center_y, source_center_z, kappa);
  result.jets[0] = compact_jet(subtract(green0, greenk));
  result.jets[1] =
      compact_jet(subtract(green0, scale(greenk, 1.0 / safe_eps)));
  for (int component = 0; component < 3; ++component) {
    const Jet8 source0 = source_normal_component_jet(
        target_center_x, target_center_y, target_center_z, source_center_x,
        source_center_y, source_center_z, component, 0.0);
    const Jet8 sourcek = source_normal_component_jet(
        target_center_x, target_center_y, target_center_z, source_center_x,
        source_center_y, source_center_z, component, kappa);
    result.jets[2 + static_cast<std::size_t>(component)] =
        compact_jet(subtract(scale(sourcek, safe_eps), source0));
    result.jets[5 + static_cast<std::size_t>(component)] =
        compact_jet(subtract(sourcek, source0));
  }
  for (const auto &jet : result.jets) {
    for (const double value : jet)
      result.finite_outputs = result.finite_outputs && std::isfinite(value);
  }
  return result;
}

CartesianJcpLocalP4 apply_cartesian_jcp_m2l_operator_p4(
    const CartesianJcpMoments &moments,
    const CartesianJcpM2lOperatorP4 &operator_p4) {
  return apply_cartesian_jcp_m2l_operator_p4_order(
      moments, operator_p4, kCartesianLocalSourceOrder);
}

CartesianJcpLocalP4 apply_cartesian_jcp_m2l_operator_p4_order(
    const CartesianJcpMoments &moments,
    const CartesianJcpM2lOperatorP4 &operator_p4, int local_order) {
  CartesianJcpLocalP4 local;
  const int order = std::clamp(local_order, 1, kCartesianLocalSourceOrder);
  if (!valid_local_order(local_order) || moments.order != order ||
      moments.term_count != cartesian_moment_term_count(order)) {
    local.finite_outputs = false;
    return local;
  }
  translate_moments_compact(
      moments.phin, operator_p4.jets[0],
      local.fields[static_cast<std::size_t>(JcpLocalField::K1)], order,
      order);
  translate_moments_compact(
      moments.phin, operator_p4.jets[1],
      local.fields[static_cast<std::size_t>(JcpLocalField::K3Potential)],
      order, order);
  for (int component = 0; component < 3; ++component) {
    translate_moments_compact(
        moments.phi_normal[static_cast<std::size_t>(component)],
        operator_p4.jets[2 + static_cast<std::size_t>(component)],
        local.fields[static_cast<std::size_t>(JcpLocalField::K2)], order,
        order);
    translate_moments_compact(
        moments.phi_normal[static_cast<std::size_t>(component)],
        operator_p4.jets[5 + static_cast<std::size_t>(component)],
        local.fields[static_cast<std::size_t>(JcpLocalField::K4Potential)],
        order, order);
  }
  local.finite_outputs = operator_p4.finite_outputs;
  for (const auto &field : local.fields) {
    for (const double value : field)
      local.finite_outputs = local.finite_outputs && std::isfinite(value);
  }
  return local;
}

CartesianJcpCompressedGreenOperatorP4
build_cartesian_jcp_compressed_green_operator_p4(
    double source_center_x, double source_center_y, double source_center_z,
    double target_center_x, double target_center_y, double target_center_z,
    double kappa) {
  CartesianJcpCompressedGreenOperatorP4 result;
  result.coulomb = compact_green_jet(
      green_jet(target_center_x, target_center_y, target_center_z,
                source_center_x, source_center_y, source_center_z, 0.0));
  result.yukawa = compact_green_jet(
      green_jet(target_center_x, target_center_y, target_center_z,
                source_center_x, source_center_y, source_center_z, kappa));
  return result;
}

void build_cartesian_jcp_compressed_green_operator_p4_order(
    double source_center_x, double source_center_y, double source_center_z,
    double target_center_x, double target_center_y, double target_center_z,
    double kappa, int local_order, std::span<double> coulomb,
    std::span<double> yukawa) {
  compact_green_jet_order(
      green_jet(target_center_x, target_center_y, target_center_z,
                source_center_x, source_center_y, source_center_z, 0.0),
      local_order, coulomb);
  compact_green_jet_order(
      green_jet(target_center_x, target_center_y, target_center_z,
                source_center_x, source_center_y, source_center_z, kappa),
      local_order, yukawa);
}

CartesianJcpLocalP4 apply_cartesian_jcp_compressed_green_operator_p4(
    const CartesianJcpMoments &moments,
    const CartesianJcpCompressedGreenOperatorP4 &operator_p4,
    double eps_ratio) {
  return apply_cartesian_jcp_compressed_green_operator_p4_order(
      moments, operator_p4, eps_ratio, kCartesianLocalSourceOrder);
}

CartesianJcpLocalP4 apply_cartesian_jcp_compressed_green_operator_p4_order(
    const CartesianJcpMoments &moments,
    const CartesianJcpCompressedGreenOperatorP4 &operator_p4,
    double eps_ratio, int local_order) {
  const std::size_t term_count =
      cartesian_compressed_green_jet_term_count_order(local_order);
  return apply_cartesian_jcp_compressed_green_operator_p4_order(
      moments, std::span<const double>(operator_p4.coulomb.data(), term_count),
      std::span<const double>(operator_p4.yukawa.data(), term_count),
      eps_ratio, local_order);
}

CartesianJcpLocalP4 apply_cartesian_jcp_compressed_green_operator_p4_order(
    const CartesianJcpMoments &moments, std::span<const double> coulomb,
    std::span<const double> yukawa, double eps_ratio, int local_order) {
  CartesianJcpLocalP4 local;
  local.finite_outputs =
      accumulate_cartesian_jcp_compressed_green_operator_p4_order(
          moments.phin, moments.phi_normal, moments.term_count, coulomb,
          yukawa, eps_ratio, local_order, local.fields);
  if (!local.finite_outputs)
    return local;

  for (const double value : coulomb)
    local.finite_outputs = local.finite_outputs && std::isfinite(value);
  for (const double value : yukawa)
    local.finite_outputs = local.finite_outputs && std::isfinite(value);
  for (const auto &field : local.fields) {
    for (const double value : field)
      local.finite_outputs = local.finite_outputs && std::isfinite(value);
  }
  return local;
}

bool accumulate_cartesian_jcp_compressed_green_operator_p4_order(
    const std::array<double, kCartesianJcpMomentMax> &phin,
    const std::array<std::array<double, kCartesianJcpMomentMax>, 3>
        &phi_normal,
    std::size_t source_term_count, std::span<const double> coulomb,
    std::span<const double> yukawa, double eps_ratio, int local_order,
    std::array<std::array<double, kCartesianLocalTermMax>, 4>
        &destination) noexcept {
  const int order = std::clamp(local_order, 1, kCartesianLocalSourceOrder);
  const std::size_t green_term_count =
      cartesian_compressed_green_jet_term_count_order(order);
  const std::size_t term_count = cartesian_moment_term_count(order);
  if (!valid_local_order(local_order) || source_term_count != term_count ||
      coulomb.size() < green_term_count || yukawa.size() < green_term_count) {
    return false;
  }
  const double safe_eps = std::max(1.0e-12, eps_ratio);
  const auto &stencils = compressed_translation_stencils();
  for (std::size_t b = 0; b < term_count; ++b) {
    double k1 = 0.0;
    double k3 = 0.0;
    std::array<double, 3> k2{};
    std::array<double, 3> k4{};
    for (std::size_t a = 0; a < term_count; ++a) {
      const auto &stencil =
          stencils[b * kCartesianLocalTermMax + a];
      const double green0 = coulomb[stencil.green_index];
      const double greenk = yukawa[stencil.green_index];
      k1 += phin[a] * stencil.factor * (green0 - greenk);
      k3 += phin[a] * stencil.factor * (green0 - greenk / safe_eps);
      for (std::size_t component = 0; component < 3; ++component) {
        const double source0 =
            stencil.derivative_degree[component] *
            coulomb[stencil.derivative_index[component]];
        const double sourcek =
            stencil.derivative_degree[component] *
            yukawa[stencil.derivative_index[component]];
        const double moment = phi_normal[component][a];
        k2[component] +=
            moment * stencil.factor * (safe_eps * sourcek - source0);
        // L2P differentiates in the target normal: K4 = d_x(d_y Gk-d_y G0).
        k4[component] +=
            moment * stencil.factor * (sourcek - source0);
      }
    }
    destination[static_cast<std::size_t>(JcpLocalField::K1)][b] += k1;
    destination[static_cast<std::size_t>(JcpLocalField::K3Potential)][b] += k3;
    destination[static_cast<std::size_t>(JcpLocalField::K2)][b] +=
        (k2[0] + k2[1]) + k2[2];
    destination[static_cast<std::size_t>(JcpLocalField::K4Potential)][b] +=
        (k4[0] + k4[1]) + k4[2];
  }
  return true;
}

void accumulate_cartesian_jcp_local(
    CartesianJcpLocalP4 &destination,
    const CartesianJcpLocalP4 &source) noexcept {
  for (std::size_t field = 0; field < destination.fields.size(); ++field) {
    for (std::size_t term = 0; term < destination.fields[field].size();
         ++term)
      destination.fields[field][term] += source.fields[field][term];
  }
  destination.finite_outputs =
      destination.finite_outputs && source.finite_outputs;
}

double evaluate_cartesian_jcp_local_scalar(
    const CartesianJcpLocalP4 &local, JcpLocalField field, double dx,
    double dy, double dz) noexcept {
  return evaluate_scalar(local.fields[static_cast<std::size_t>(field)], dx,
                         dy, dz);
}

std::array<double, 3> evaluate_cartesian_jcp_local_gradient(
    const CartesianJcpLocalP4 &local, JcpLocalField field, double dx,
    double dy, double dz, int target_order) noexcept {
  return evaluate_gradient(local.fields[static_cast<std::size_t>(field)], dx,
                           dy, dz, target_order);
}

std::array<double, 4> evaluate_cartesian_jcp_l2p(
    const PrototypePanel &target, double target_center_x,
    double target_center_y, double target_center_z,
    const CartesianJcpLocalP4 &local) noexcept {
  return evaluate_cartesian_jcp_l2p_order(
      target, target_center_x, target_center_y, target_center_z, local,
      kCartesianLocalTargetOrder);
}

std::array<double, 4> evaluate_cartesian_jcp_l2p_order(
    const PrototypePanel &target, double target_center_x,
    double target_center_y, double target_center_z,
    const CartesianJcpLocalP4 &local, int target_order) noexcept {
  const int order = std::clamp(target_order, 0, kCartesianLocalTargetOrder);
  const double dx = target.x - target_center_x;
  const double dy = target.y - target_center_y;
  const double dz = target.z - target_center_z;
  std::array<double, 4> channels{};
  channels[0] = evaluate_scalar(
      local.fields[static_cast<std::size_t>(JcpLocalField::K1)], dx, dy, dz,
      order);
  channels[1] = evaluate_scalar(
      local.fields[static_cast<std::size_t>(JcpLocalField::K2)], dx, dy, dz,
      order);
  const auto k3_gradient = evaluate_gradient(
      local.fields[static_cast<std::size_t>(JcpLocalField::K3Potential)], dx,
      dy, dz, order);
  const auto k4_gradient = evaluate_gradient(
      local.fields[static_cast<std::size_t>(JcpLocalField::K4Potential)], dx,
      dy, dz, order);
  channels[2] = target.nx * k3_gradient[0] + target.ny * k3_gradient[1] +
                target.nz * k3_gradient[2];
  channels[3] = target.nx * k4_gradient[0] + target.ny * k4_gradient[1] +
                target.nz * k4_gradient[2];
  return channels;
}

} // namespace titan::tabi::fmm_m2l
