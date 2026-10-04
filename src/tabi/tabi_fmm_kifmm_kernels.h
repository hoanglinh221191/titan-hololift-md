#ifndef TITAN_TABI_FMM_KIFMM_KERNELS_H
#define TITAN_TABI_FMM_KIFMM_KERNELS_H

#include "tabi_fmm_kifmm_surface.h"

namespace titan::tabi::fmm_kifmm {

enum class KifmmKernel {
  Laplace,
  Yukawa,
};

double laplace_G(Vec3 target, Vec3 source) noexcept;
double yukawa_G(Vec3 target, Vec3 source, double kappa) noexcept;

Vec3 grad_target_laplace_G(Vec3 target, Vec3 source) noexcept;
Vec3 grad_target_yukawa_G(Vec3 target, Vec3 source,
                          double kappa) noexcept;
Vec3 grad_source_laplace_G(Vec3 target, Vec3 source) noexcept;
Vec3 grad_source_yukawa_G(Vec3 target, Vec3 source,
                          double kappa) noexcept;
Vec3 grad_target_of_source_gradient_component(KifmmKernel kernel, Vec3 target,
                                               Vec3 source, double kappa,
                                               int source_component) noexcept;

double scalar_kernel_value(KifmmKernel kernel, Vec3 target, Vec3 source,
                           double kappa) noexcept;
Vec3 scalar_kernel_grad_target(KifmmKernel kernel, Vec3 target, Vec3 source,
                               double kappa) noexcept;

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_KERNELS_H
