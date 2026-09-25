#ifndef PBCTOPO_HULL_H
#define PBCTOPO_HULL_H

#include "titan_vec3.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pbctopo_hull {

using titan_geom::Vec3;
using titan_geom::cross;
using titan_geom::dot;
using titan_geom::norm2;

struct PbctopoHullSpan {
  std::size_t begin = 0;
  std::size_t end = 0;
};

struct PbctopoHullFace {
  int a = -1;
  int b = -1;
  int c = -1;
  Vec3 normal{};
  bool valid = false;
};

struct PbctopoHullScratch {
  std::vector<Vec3> shifted_points;
  std::vector<double> xs;
  std::vector<double> ys;
  std::vector<double> zs;
  std::vector<double> face_nx;
  std::vector<double> face_ny;
  std::vector<double> face_nz;
  std::vector<double> face_d;
  std::vector<std::size_t> face_indices;
  std::vector<unsigned char> is_seed;
  std::vector<PbctopoHullFace> faces;
  std::vector<std::size_t> visible_faces;
  std::unordered_map<std::uint64_t, std::pair<int, int>> horizon_edges;

  void reserve(std::size_t point_count);
  void clear_transient();
};

bool avx2_available() noexcept;

void shift_points_by_spans(const std::vector<Vec3> &points,
                           const std::vector<PbctopoHullSpan> &spans,
                           const Vec3 &shift, std::vector<Vec3> &out,
                           PbctopoHullScratch *scratch = nullptr);

bool quickhull_volume(const std::vector<Vec3> &points, double &out_volume,
                      PbctopoHullScratch *scratch = nullptr);

bool quickhull_volume_shifted_spans(
    const std::vector<Vec3> &points,
    const std::vector<PbctopoHullSpan> &spans, const Vec3 &shift,
    double &out_volume, PbctopoHullScratch *scratch = nullptr);

} // namespace pbctopo_hull

#endif
