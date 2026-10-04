#include "tabi_fmm_kifmm_kernels.h"

#include <cmath>
#include <numbers>

namespace titan::tabi::fmm_kifmm {
namespace {

constexpr double kFourPi = 4.0 * std::numbers::pi;
constexpr double kTinyR2 = 1.0e-300;

Vec3 difference(Vec3 a, Vec3 b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 scale(Vec3 v, double s) noexcept {
  return {v.x * s, v.y * s, v.z * s};
}

double norm2(Vec3 v) noexcept {
  return v.x * v.x + v.y * v.y + v.z * v.z;
}

double component(Vec3 v, int c) noexcept {
  if (c == 0) {
    return v.x;
  }
  if (c == 1) {
    return v.y;
  }
  if (c == 2) {
    return v.z;
  }
  return 0.0;
}

Vec3 grad_target_radial(Vec3 target, Vec3 source, double radial_factor) noexcept {
  return scale(difference(target, source), radial_factor);
}

} // namespace

double laplace_G(Vec3 target, Vec3 source) noexcept {
  const double r2 = norm2(difference(target, source));
  if (!(r2 > kTinyR2) || !std::isfinite(r2)) {
    return 0.0;
  }
  return 1.0 / (kFourPi * std::sqrt(r2));
}

double yukawa_G(Vec3 target, Vec3 source, double kappa) noexcept {
  const double r2 = norm2(difference(target, source));
  if (!(r2 > kTinyR2) || !std::isfinite(r2) || !std::isfinite(kappa) ||
      kappa < 0.0) {
    return 0.0;
  }
  const double r = std::sqrt(r2);
  return std::exp(-kappa * r) / (kFourPi * r);
}

Vec3 grad_target_laplace_G(Vec3 target, Vec3 source) noexcept {
  const double r2 = norm2(difference(target, source));
  if (!(r2 > kTinyR2) || !std::isfinite(r2)) {
    return {};
  }
  const double inv_r = 1.0 / std::sqrt(r2);
  const double factor = -1.0 / (kFourPi * r2) * inv_r;
  return grad_target_radial(target, source, factor);
}

Vec3 grad_target_yukawa_G(Vec3 target, Vec3 source,
                          double kappa) noexcept {
  const double r2 = norm2(difference(target, source));
  if (!(r2 > kTinyR2) || !std::isfinite(r2) || !std::isfinite(kappa) ||
      kappa < 0.0) {
    return {};
  }
  const double r = std::sqrt(r2);
  const double factor =
      -std::exp(-kappa * r) * (kappa * r + 1.0) / (kFourPi * r2 * r);
  return grad_target_radial(target, source, factor);
}

Vec3 grad_source_laplace_G(Vec3 target, Vec3 source) noexcept {
  return scale(grad_target_laplace_G(target, source), -1.0);
}

Vec3 grad_source_yukawa_G(Vec3 target, Vec3 source,
                          double kappa) noexcept {
  return scale(grad_target_yukawa_G(target, source, kappa), -1.0);
}

Vec3 grad_target_of_source_gradient_component(KifmmKernel kernel, Vec3 target,
                                               Vec3 source, double kappa,
                                               int source_component) noexcept {
  if (source_component < 0 || source_component > 2) {
    return {};
  }
  const Vec3 u = difference(target, source);
  const double r2 = norm2(u);
  if (!(r2 > kTinyR2) || !std::isfinite(r2) || !std::isfinite(kappa) ||
      (kernel == KifmmKernel::Yukawa && kappa < 0.0)) {
    return {};
  }
  const double r = std::sqrt(r2);
  const double inv_r = 1.0 / r;
  const double inv_r2 = 1.0 / r2;
  const double inv_r3 = inv_r * inv_r2;

  double fp = 0.0;
  double fpp = 0.0;
  switch (kernel) {
  case KifmmKernel::Laplace:
    fp = -1.0 / (kFourPi * r2);
    fpp = 2.0 / (kFourPi * r2 * r);
    break;
  case KifmmKernel::Yukawa:
    fp = -std::exp(-kappa * r) * (kappa * r + 1.0) / (kFourPi * r2);
    fpp = std::exp(-kappa * r) *
          (kappa * kappa * inv_r + 2.0 * kappa * inv_r2 +
           2.0 * inv_r3) /
          kFourPi;
    break;
  }

  Vec3 result{};
  for (int a = 0; a < 3; ++a) {
    const double ua = component(u, a);
    const double uc = component(u, source_component);
    const double delta = a == source_component ? 1.0 : 0.0;
    const double hessian =
        fpp * ua * uc * inv_r2 + fp * (delta * inv_r - ua * uc * inv_r3);
    const double value = -hessian;
    if (a == 0) {
      result.x = value;
    } else if (a == 1) {
      result.y = value;
    } else {
      result.z = value;
    }
  }
  return result;
}

double scalar_kernel_value(KifmmKernel kernel, Vec3 target, Vec3 source,
                           double kappa) noexcept {
  switch (kernel) {
  case KifmmKernel::Laplace:
    return laplace_G(target, source);
  case KifmmKernel::Yukawa:
    return yukawa_G(target, source, kappa);
  }
  return 0.0;
}

Vec3 scalar_kernel_grad_target(KifmmKernel kernel, Vec3 target, Vec3 source,
                               double kappa) noexcept {
  switch (kernel) {
  case KifmmKernel::Laplace:
    return grad_target_laplace_G(target, source);
  case KifmmKernel::Yukawa:
    return grad_target_yukawa_G(target, source, kappa);
  }
  return {};
}

} // namespace titan::tabi::fmm_kifmm
