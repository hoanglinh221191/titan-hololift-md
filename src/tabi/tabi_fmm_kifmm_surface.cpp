#include "tabi_fmm_kifmm_surface.h"

#include <array>
#include <cmath>
#include <set>
#include <stdexcept>

namespace titan::tabi::fmm_kifmm {
namespace {

struct GridKey {
  int ix{0};
  int iy{0};
  int iz{0};

  friend bool operator<(const GridKey &lhs, const GridKey &rhs) noexcept {
    if (lhs.ix != rhs.ix) {
      return lhs.ix < rhs.ix;
    }
    if (lhs.iy != rhs.iy) {
      return lhs.iy < rhs.iy;
    }
    return lhs.iz < rhs.iz;
  }
};

double unit_coord(int index, int order) {
  return -1.0 + 2.0 * static_cast<double>(index) /
                    static_cast<double>(order - 1);
}

Vec3 scale_vec(Vec3 value, double scale) {
  return {scale * value.x, scale * value.y, scale * value.z};
}

Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

void append_point(std::vector<KifmmSurfacePoint> &points,
                  std::set<GridKey> &seen, GridKey key, Vec3 unit,
                  Vec3 normal, double scale) {
  if (!seen.insert(key).second) {
    return;
  }
  points.push_back({scale_vec(unit, scale), normal, 1.0});
}

std::vector<KifmmSurfacePoint> make_unit_cube_surface(int order,
                                                      double scale) {
  if (order < 2) {
    throw std::invalid_argument("KIFMM cube surface order must be at least 2");
  }
  if (!std::isfinite(scale) || scale <= 0.0) {
    throw std::invalid_argument("KIFMM cube surface scale must be positive");
  }

  std::vector<KifmmSurfacePoint> points;
  points.reserve(kifmm_cube_surface_unique_count(order));
  std::set<GridKey> seen;

  const std::array<int, 2> fixed_indices{0, order - 1};
  for (const int ix : fixed_indices) {
    const double x = unit_coord(ix, order);
    const Vec3 normal{x < 0.0 ? -1.0 : 1.0, 0.0, 0.0};
    for (int iy = 0; iy < order; ++iy) {
      for (int iz = 0; iz < order; ++iz) {
        append_point(points, seen, {ix, iy, iz},
                     {x, unit_coord(iy, order), unit_coord(iz, order)},
                     normal, scale);
      }
    }
  }
  for (const int iy : fixed_indices) {
    const double y = unit_coord(iy, order);
    const Vec3 normal{0.0, y < 0.0 ? -1.0 : 1.0, 0.0};
    for (int ix = 0; ix < order; ++ix) {
      for (int iz = 0; iz < order; ++iz) {
        append_point(points, seen, {ix, iy, iz},
                     {unit_coord(ix, order), y, unit_coord(iz, order)},
                     normal, scale);
      }
    }
  }
  for (const int iz : fixed_indices) {
    const double z = unit_coord(iz, order);
    const Vec3 normal{0.0, 0.0, z < 0.0 ? -1.0 : 1.0};
    for (int ix = 0; ix < order; ++ix) {
      for (int iy = 0; iy < order; ++iy) {
        append_point(points, seen, {ix, iy, iz},
                     {unit_coord(ix, order), unit_coord(iy, order), z},
                     normal, scale);
      }
    }
  }
  return points;
}

} // namespace

std::size_t kifmm_cube_surface_unique_count(int order) {
  if (order < 2) {
    throw std::invalid_argument("KIFMM cube surface order must be at least 2");
  }
  const auto q = static_cast<std::size_t>(order);
  return 6ULL * q * q - 12ULL * q + 8ULL;
}

KifmmSurfaceRule make_cube_surface_rule(int order, double equiv_scale,
                                        double check_scale) {
  if (!std::isfinite(equiv_scale) || !std::isfinite(check_scale) ||
      equiv_scale <= 1.0 || check_scale <= equiv_scale) {
    throw std::invalid_argument(
        "KIFMM cube surfaces require 1.0 < equiv_scale < check_scale");
  }

  KifmmSurfaceRule rule;
  rule.order = order;
  rule.equiv_scale = equiv_scale;
  rule.check_scale = check_scale;
  rule.unit_equiv = make_unit_cube_surface(order, equiv_scale);
  rule.unit_check = make_unit_cube_surface(order, check_scale);
  return rule;
}

KifmmSurfaceRule make_cube_local_surface_rule(int order, double equiv_scale,
                                              double check_scale) {
  if (!std::isfinite(equiv_scale) || !std::isfinite(check_scale) ||
      check_scale <= 1.0 || equiv_scale <= check_scale) {
    throw std::invalid_argument(
        "KIFMM local cube surfaces require 1.0 < check_scale < equiv_scale");
  }

  KifmmSurfaceRule rule;
  rule.order = order;
  rule.equiv_scale = equiv_scale;
  rule.check_scale = check_scale;
  rule.unit_equiv = make_unit_cube_surface(order, equiv_scale);
  rule.unit_check = make_unit_cube_surface(order, check_scale);
  return rule;
}

std::vector<KifmmSurfacePoint>
instantiate_surface(std::span<const KifmmSurfacePoint> unit_surface,
                    const KifmmBox &box) {
  if (!std::isfinite(box.half_width) || box.half_width <= 0.0) {
    throw std::invalid_argument("KIFMM box half_width must be positive");
  }
  std::vector<KifmmSurfacePoint> placed;
  placed.reserve(unit_surface.size());
  for (const auto &point : unit_surface) {
    placed.push_back(
        {add(box.center, scale_vec(point.x, box.half_width)),
         point.outward_normal, point.weight});
  }
  return placed;
}

} // namespace titan::tabi::fmm_kifmm
