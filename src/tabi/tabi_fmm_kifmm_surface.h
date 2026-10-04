#ifndef TITAN_TABI_FMM_KIFMM_SURFACE_H
#define TITAN_TABI_FMM_KIFMM_SURFACE_H

#include <cstddef>
#include <span>
#include <vector>

namespace titan::tabi::fmm_kifmm {

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct KifmmBox {
  Vec3 center{};
  double half_width{0.0};
  int level{0};
  std::size_t node_id{0};
};

struct KifmmSurfacePoint {
  Vec3 x{};
  Vec3 outward_normal{};
  double weight{1.0};
};

struct KifmmSurfaceRule {
  int order{6};
  double equiv_scale{1.05};
  double check_scale{1.60};
  std::vector<KifmmSurfacePoint> unit_equiv{};
  std::vector<KifmmSurfacePoint> unit_check{};
};

std::size_t kifmm_cube_surface_unique_count(int order);

KifmmSurfaceRule make_cube_surface_rule(int order, double equiv_scale,
                                        double check_scale);

KifmmSurfaceRule make_cube_local_surface_rule(int order, double equiv_scale,
                                              double check_scale);

std::vector<KifmmSurfacePoint>
instantiate_surface(std::span<const KifmmSurfacePoint> unit_surface,
                    const KifmmBox &box);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_SURFACE_H
