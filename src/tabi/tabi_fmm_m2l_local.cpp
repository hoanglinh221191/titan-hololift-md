#include "tabi_fmm_m2l_local.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

// A truncated Taylor jet.  Coefficients are computed through total degree
// `order` only; a coefficient never depends on higher-degree ones, so a jet
// truncated at a lower order has exactly the same leading coefficients.
struct Jet8 {
  std::array<double, kJetSize> coeff{};
  int order{kCartesianCompressedGreenOrder};
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

Jet8 constant(double value,
              int order = kCartesianCompressedGreenOrder) noexcept {
  Jet8 result;
  result.order = order;
  result.coeff[cube_index(0, 0, 0)] = value;
  return result;
}

Jet8 variable(double value, int axis,
              int order = kCartesianCompressedGreenOrder) noexcept {
  Jet8 result = constant(value, order);
  result.coeff[cube_index(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0,
                           axis == 2 ? 1 : 0)] = 1.0;
  return result;
}

Jet8 add(const Jet8 &a, const Jet8 &b) noexcept {
  Jet8 result;
  result.order = std::min(a.order, b.order);
  for (const MultiIndex &index : indices_through(result.order)) {
    const std::size_t i = cube_index(index.x, index.y, index.z);
    result.coeff[i] = a.coeff[i] + b.coeff[i];
  }
  return result;
}

Jet8 subtract(const Jet8 &a, const Jet8 &b) noexcept {
  Jet8 result;
  result.order = std::min(a.order, b.order);
  for (const MultiIndex &index : indices_through(result.order)) {
    const std::size_t i = cube_index(index.x, index.y, index.z);
    result.coeff[i] = a.coeff[i] - b.coeff[i];
  }
  return result;
}

Jet8 scale(const Jet8 &a, double factor) noexcept {
  Jet8 result;
  result.order = a.order;
  for (const MultiIndex &index : indices_through(result.order)) {
    const std::size_t i = cube_index(index.x, index.y, index.z);
    result.coeff[i] = factor * a.coeff[i];
  }
  return result;
}

// The (a, b) coefficient pairs of every product coefficient, in the order of
// the x, y, z loops that used to enumerate them, for the graded index list of
// the highest order; a lower order is a prefix.
struct ConvolutionTable {
  std::vector<std::uint32_t> start;    // per graded index, into a/b
  std::vector<std::uint16_t> a, b;     // cube indices
  std::vector<std::uint16_t> out;      // cube index of each graded index
};

const ConvolutionTable &convolution_table() {
  static const ConvolutionTable table = [] {
    ConvolutionTable t;
    const std::vector<MultiIndex> &indices =
        indices_through(kCartesianCompressedGreenOrder);
    for (const MultiIndex &index : indices) {
      t.start.push_back(static_cast<std::uint32_t>(t.a.size()));
      t.out.push_back(static_cast<std::uint16_t>(
          cube_index(index.x, index.y, index.z)));
      for (int x = 0; x <= index.x; ++x)
        for (int y = 0; y <= index.y; ++y)
          for (int z = 0; z <= index.z; ++z) {
            t.a.push_back(static_cast<std::uint16_t>(cube_index(x, y, z)));
            t.b.push_back(static_cast<std::uint16_t>(
                cube_index(index.x - x, index.y - y, index.z - z)));
          }
    }
    t.start.push_back(static_cast<std::uint32_t>(t.a.size()));
    return t;
  }();
  return table;
}

Jet8 multiply(const Jet8 &a, const Jet8 &b) noexcept {
  Jet8 result;
  result.order = std::min(a.order, b.order);
  const ConvolutionTable &table = convolution_table();
  const std::size_t count = indices_through(result.order).size();
  for (std::size_t i = 0; i < count; ++i) {
    double value = 0.0;
    for (std::uint32_t p = table.start[i]; p < table.start[i + 1]; ++p)
      value += a.coeff[table.a[p]] * b.coeff[table.b[p]];
    result.coeff[table.out[i]] = value;
  }
  return result;
}

Jet8 reciprocal(const Jet8 &a) noexcept {
  Jet8 result;
  result.order = a.order;
  const double a0 = a.coeff[cube_index(0, 0, 0)];
  if (!std::isfinite(a0) || std::abs(a0) <= 1.0e-30)
    return result;
  result.coeff[cube_index(0, 0, 0)] = 1.0 / a0;
  for (const MultiIndex &index : indices_through(result.order)) {
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
  result.order = a.order;
  const double a0 = a.coeff[cube_index(0, 0, 0)];
  if (!std::isfinite(a0) || a0 <= 0.0)
    return result;
  const double root = std::sqrt(a0);
  result.coeff[cube_index(0, 0, 0)] = root;
  for (const MultiIndex &index : indices_through(result.order)) {
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
  result.order = a.order;
  result.coeff[cube_index(0, 0, 0)] =
      std::exp(a.coeff[cube_index(0, 0, 0)]);
  for (const MultiIndex &index : indices_through(result.order)) {
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
  bool separated{false};  // false: coincident centres, every jet left zero
};

// need_inv_r3 = false skips 1/r^2 and 1/r^3, which only the source-normal
// jets use; every other member is computed exactly as with it.
KernelBase kernel_base(double tx, double ty, double tz, double sx, double sy,
                       double sz, double kappa,
                       int order = kCartesianCompressedGreenOrder,
                       bool need_inv_r3 = true) noexcept {
  KernelBase base;
  const Jet8 source_x = variable(sx, 0, order);
  const Jet8 source_y = variable(sy, 1, order);
  const Jet8 source_z = variable(sz, 2, order);
  base.ux = subtract(constant(tx, order), source_x);
  base.uy = subtract(constant(ty, order), source_y);
  base.uz = subtract(constant(tz, order), source_z);
  const Jet8 r2 = add(add(multiply(base.ux, base.ux),
                          multiply(base.uy, base.uy)),
                      multiply(base.uz, base.uz));
  if (r2.coeff[cube_index(0, 0, 0)] <= 1.0e-28)
    return base;
  base.separated = true;
  base.r = square_root(r2);
  base.inv_r = reciprocal(base.r);
  if (need_inv_r3) {
    base.inv_r2 = multiply(base.inv_r, base.inv_r);
    base.inv_r3 = multiply(base.inv_r2, base.inv_r);
  }
  base.screened = exponential(scale(base.r, -std::max(0.0, kappa)));
  return base;
}

// The base of the same geometry at another kappa: only the screening depends
// on kappa, so this equals kernel_base(..., kappa, ...) member for member.
KernelBase with_screening(const KernelBase &base, double kappa) noexcept {
  KernelBase result = base;
  if (base.separated)
    result.screened = exponential(scale(base.r, -std::max(0.0, kappa)));
  return result;
}

// kernel_base() uses kappa only through max(0, kappa) in the screening, so one
// base per kappa serves the Green jet and all three source-normal components.
Jet8 green_jet_from(const KernelBase &base) noexcept {
  return scale(multiply(base.screened, base.inv_r), 1.0 / (4.0 * kPi));
}

Jet8 green_jet(double tx, double ty, double tz, double sx, double sy,
               double sz, double kappa) noexcept {
  return green_jet_from(kernel_base(tx, ty, tz, sx, sy, sz, kappa));
}

// exp(-kappa r) (1 + kappa r), shared by the three components.
Jet8 screened_one_plus_kr(const KernelBase &base, double safe_kappa) noexcept {
  const Jet8 one_plus_kr =
      add(constant(1.0, base.r.order), scale(base.r, safe_kappa));
  return multiply(base.screened, one_plus_kr);
}

// u_c / r^3 does not depend on kappa: the caller forms it once per component.
Jet8 source_normal_u_inv_r3(const KernelBase &base, int component) noexcept {
  const Jet8 &u = component == 0 ? base.ux : (component == 1 ? base.uy : base.uz);
  return multiply(u, base.inv_r3);
}

Jet8 source_normal_component_from(const Jet8 &screened_one_plus,
                                  const Jet8 &u_inv_r3) noexcept {
  return scale(multiply(screened_one_plus, u_inv_r3), 1.0 / (4.0 * kPi));
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

// For each (source order, target order): the kernel index and the factor
// sign * C(gx, ax) * C(gy, ay) * C(gz, az) of every (beta, alpha) pair, row by
// row in beta, computed by the same expression the loop used to evaluate.
struct TranslationTable {
  std::size_t sources{0};
  std::size_t targets{0};
  std::vector<std::uint16_t> kernel_index;
  std::vector<double> factor;
};

const TranslationTable &translation_table(int source_order, int target_order) {
  static const std::array<TranslationTable, (kCartesianLocalSourceOrder + 1) *
                                                (kCartesianLocalTargetOrder + 1)>
      tables = [] {
        std::array<TranslationTable, (kCartesianLocalSourceOrder + 1) *
                                         (kCartesianLocalTargetOrder + 1)>
            result{};
        for (int so = 0; so <= kCartesianLocalSourceOrder; ++so)
          for (int to = 0; to <= kCartesianLocalTargetOrder; ++to) {
            TranslationTable &t = result[static_cast<std::size_t>(
                so * (kCartesianLocalTargetOrder + 1) + to)];
            const std::vector<MultiIndex> &source = indices_through(so);
            const std::vector<MultiIndex> &target = indices_through(to);
            t.sources = source.size();
            t.targets = target.size();
            for (const MultiIndex &beta : target) {
              const double sign =
                  ((beta.x + beta.y + beta.z) % 2) == 0 ? 1.0 : -1.0;
              for (const MultiIndex &alpha : source) {
                const int gx = alpha.x + beta.x;
                const int gy = alpha.y + beta.y;
                const int gz = alpha.z + beta.z;
                t.kernel_index.push_back(
                    static_cast<std::uint16_t>(compact_index(gx, gy, gz)));
                t.factor.push_back(sign * binomial(gx, alpha.x) *
                                   binomial(gy, alpha.y) *
                                   binomial(gz, alpha.z));
              }
            }
          }
        return result;
      }();
  return tables[static_cast<std::size_t>(
      source_order * (kCartesianLocalTargetOrder + 1) + target_order)];
}

void translate_moments_compact(
    const std::array<double, kCartesianJcpMomentMax> &moments,
    const std::array<double, kCartesianKernelJetTermCount> &kernel,
    std::array<double, kCartesianLocalTermMax> &local, int source_order,
    int target_order) {
  const TranslationTable &table = translation_table(
      std::clamp(source_order, 0, kCartesianLocalSourceOrder),
      std::clamp(target_order, 0, kCartesianLocalTargetOrder));
  const std::size_t sources = table.sources;
  for (std::size_t b = 0; b < table.targets; ++b) {
    const std::size_t row = b * sources;
    double value = 0.0;
    for (std::size_t a = 0; a < sources; ++a)
      value += moments[a] * table.factor[row + a] *
               kernel[table.kernel_index[row + a]];
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
  const double safe_kappa = std::max(0.0, kappa);
  // Two kernel bases (kappa = 0 and kappa) instead of one per jet: the same
  // operations, so the operator is unchanged bit for bit.
  // The operator keeps the jets through kCartesianLocalKernelOrder only.
  const KernelBase base0 = kernel_base(
      target_center_x, target_center_y, target_center_z, source_center_x,
      source_center_y, source_center_z, 0.0, kCartesianLocalKernelOrder);
  const KernelBase basek = with_screening(base0, kappa);
  const Jet8 green0 = green_jet_from(base0);
  const Jet8 greenk = green_jet_from(basek);
  result.jets[0] = compact_jet(subtract(green0, greenk));
  result.jets[1] =
      compact_jet(subtract(green0, scale(greenk, 1.0 / safe_eps)));
  const Jet8 screened0 = screened_one_plus_kr(base0, 0.0);
  const Jet8 screenedk = screened_one_plus_kr(basek, safe_kappa);
  for (int component = 0; component < 3; ++component) {
    const Jet8 u_inv_r3 = source_normal_u_inv_r3(base0, component);
    const Jet8 source0 = source_normal_component_from(screened0, u_inv_r3);
    const Jet8 sourcek = source_normal_component_from(screenedk, u_inv_r3);
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
  // Jets through the order compact_green_jet_order() keeps, one geometry for
  // both screenings, no 1/r^3: the kept coefficients are unchanged.
  const int order =
      std::clamp(2 * local_order + 1, 0, kCartesianCompressedGreenOrder);
  const KernelBase base0 =
      kernel_base(target_center_x, target_center_y, target_center_z,
                  source_center_x, source_center_y, source_center_z, 0.0,
                  order, false);
  compact_green_jet_order(green_jet_from(base0), local_order, coulomb);
  compact_green_jet_order(green_jet_from(with_screening(base0, kappa)),
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

std::array<std::array<double, 4>, 2> evaluate_cartesian_jcp_l2p_two_orders(
    const PrototypePanel &target, double target_center_x,
    double target_center_y, double target_center_z,
    const CartesianJcpLocalP4 &local, int high_order, int low_order) noexcept {
  // One pass over the graded terms of the high order: the terms of the low
  // order are its leading terms, so the partial sums at that point are the
  // low-order values.  Powers come from tables built by the same repeated
  // products as power(), so both results equal two separate calls of
  // evaluate_cartesian_jcp_l2p_order() bit for bit.
  const int high = std::clamp(high_order, 0, kCartesianLocalTargetOrder);
  const int low = std::clamp(std::min(low_order, high), 0, kCartesianLocalTargetOrder);
  const std::vector<MultiIndex> &indices = indices_through(high);
  const std::size_t low_count = indices_through(low).size();
  const double dx = target.x - target_center_x;
  const double dy = target.y - target_center_y;
  const double dz = target.z - target_center_z;
  std::array<double, kCartesianLocalTargetOrder + 1> px{}, py{}, pz{};
  px[0] = py[0] = pz[0] = 1.0;
  for (int k = 1; k <= high; ++k) {
    px[static_cast<std::size_t>(k)] = px[static_cast<std::size_t>(k - 1)] * dx;
    py[static_cast<std::size_t>(k)] = py[static_cast<std::size_t>(k - 1)] * dy;
    pz[static_cast<std::size_t>(k)] = pz[static_cast<std::size_t>(k - 1)] * dz;
  }
  const auto &k1 = local.fields[static_cast<std::size_t>(JcpLocalField::K1)];
  const auto &k2 = local.fields[static_cast<std::size_t>(JcpLocalField::K2)];
  const auto &k3 = local.fields[static_cast<std::size_t>(JcpLocalField::K3Potential)];
  const auto &k4 = local.fields[static_cast<std::size_t>(JcpLocalField::K4Potential)];
  double s1 = 0.0, s2 = 0.0;
  std::array<double, 3> g3{}, g4{};
  std::array<std::array<double, 4>, 2> out{};
  const auto finish = [&](std::array<double, 4> &channels) {
    channels[0] = s1;
    channels[1] = s2;
    channels[2] = target.nx * g3[0] + target.ny * g3[1] + target.nz * g3[2];
    channels[3] = target.nx * g4[0] + target.ny * g4[1] + target.nz * g4[2];
  };
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (i == low_count)
      finish(out[1]);
    const MultiIndex &index = indices[i];
    const std::size_t ix = static_cast<std::size_t>(index.x);
    const std::size_t iy = static_cast<std::size_t>(index.y);
    const std::size_t iz = static_cast<std::size_t>(index.z);
    s1 += k1[i] * px[ix] * py[iy] * pz[iz];
    s2 += k2[i] * px[ix] * py[iy] * pz[iz];
    if (index.x > 0) {
      g3[0] += static_cast<double>(index.x) * k3[i] * px[ix - 1] * py[iy] * pz[iz];
      g4[0] += static_cast<double>(index.x) * k4[i] * px[ix - 1] * py[iy] * pz[iz];
    }
    if (index.y > 0) {
      g3[1] += static_cast<double>(index.y) * k3[i] * px[ix] * py[iy - 1] * pz[iz];
      g4[1] += static_cast<double>(index.y) * k4[i] * px[ix] * py[iy - 1] * pz[iz];
    }
    if (index.z > 0) {
      g3[2] += static_cast<double>(index.z) * k3[i] * px[ix] * py[iy] * pz[iz - 1];
      g4[2] += static_cast<double>(index.z) * k4[i] * px[ix] * py[iy] * pz[iz - 1];
    }
  }
  if (low_count >= indices.size())
    finish(out[1]);
  finish(out[0]);
  return out;
}

} // namespace titan::tabi::fmm_m2l
