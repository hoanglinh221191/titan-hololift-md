#ifndef TITAN_GYRATION_H
#define TITAN_GYRATION_H

#include "aligned_allocator.h"
#include "titan_vec3.h"

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace titan_gyration {

using titan_geom::Vec3;

struct PointStats {
  std::size_t count = 0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  double sum_sq = 0.0;
  double sum_x2 = 0.0;
  double sum_y2 = 0.0;
  double sum_z2 = 0.0;
  double sum_xy = 0.0;
  double sum_xz = 0.0;
  double sum_yz = 0.0;
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();

  void add(double x, double y, double z) noexcept;
};

struct GyrationShapeMetrics {
  double kappa2 = std::numeric_limits<double>::infinity();
  double asphericity = std::numeric_limits<double>::infinity();
  double acylindricity = std::numeric_limits<double>::infinity();
};

struct GyrationScratch {
  AlignedDoubleVector xs;
  AlignedDoubleVector ys;
  AlignedDoubleVector zs;

  void reserve(std::size_t point_count);
};

bool avx2_available() noexcept;

PointStats build_point_stats(const std::vector<Vec3> &points,
                             GyrationScratch *scratch = nullptr);
PointStats build_point_stats_soa(const double *xs, const double *ys,
                                 const double *zs, std::size_t count) noexcept;
double compute_bbox_volume(const PointStats &stats) noexcept;
double compute_rg2_from_stats(const PointStats &stats) noexcept;
double compute_rg_from_stats(const PointStats &stats) noexcept;
double compute_rg2(const std::vector<Vec3> &points,
                   GyrationScratch *scratch = nullptr);
std::array<double, 3> compute_principal_rg2(const PointStats &stats);
GyrationShapeMetrics
compute_gyration_shape_metrics(const std::array<double, 3> &principal_rg2);

} // namespace titan_gyration

#endif // TITAN_GYRATION_H
