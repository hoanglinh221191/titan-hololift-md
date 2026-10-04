#include "pbctopo_hull.h"

#include "titan_simd_dispatch.h"

#include <cassert>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace pbctopo_hull {

namespace {

constexpr double kHullEps = 1.0e-9;

struct HullTolerance {
  double distance_eps = kHullEps;
  double area_eps = kHullEps;
};

using HullFace = PbctopoHullFace;

inline std::uint64_t oriented_edge_key(int u, int v) noexcept {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
         static_cast<std::uint32_t>(v);
}

void sync_soa(const std::vector<Vec3> &points, PbctopoHullScratch &scratch) {
  scratch.reserve(points.size());
  scratch.xs.resize(points.size());
  scratch.ys.resize(points.size());
  scratch.zs.resize(points.size());
  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    scratch.xs[idx] = points[idx].x;
    scratch.ys[idx] = points[idx].y;
    scratch.zs[idx] = points[idx].z;
  }
}

void rebuild_face_plane_cache(const std::vector<HullFace> &faces,
                              const std::vector<Vec3> &points,
                              PbctopoHullScratch &scratch) {
  scratch.face_nx.clear();
  scratch.face_ny.clear();
  scratch.face_nz.clear();
  scratch.face_d.clear();
  scratch.face_indices.clear();
  scratch.face_nx.reserve(faces.size());
  scratch.face_ny.reserve(faces.size());
  scratch.face_nz.reserve(faces.size());
  scratch.face_d.reserve(faces.size());
  scratch.face_indices.reserve(faces.size());
  for (std::size_t idx = 0; idx < faces.size(); ++idx) {
    if (!faces[idx].valid)
      continue;
    const auto &face = faces[idx];
    const Vec3 &anchor = points[static_cast<std::size_t>(face.a)];
    scratch.face_nx.push_back(face.normal.x);
    scratch.face_ny.push_back(face.normal.y);
    scratch.face_nz.push_back(face.normal.z);
    scratch.face_d.push_back(-dot(face.normal, anchor));
    scratch.face_indices.push_back(idx);
  }
}

void collect_visible_faces_scalar(const std::vector<HullFace> &faces,
                                  const std::vector<Vec3> &points,
                                  const Vec3 &point,
                                  const HullTolerance &tolerance,
                                  std::vector<std::size_t> &visible_faces) {
  visible_faces.clear();
  for (std::size_t idx = 0; idx < faces.size(); ++idx) {
    if (!faces[idx].valid)
      continue;
    const auto &face = faces[idx];
    const double distance =
        dot(face.normal,
            point - points[static_cast<std::size_t>(face.a)]);
    if (distance > tolerance.distance_eps) {
      visible_faces.push_back(idx);
    }
  }
}

double farthest_line_distance2_scalar(const std::vector<Vec3> &points,
                                      int min_x, int max_x,
                                      const Vec3 &line, int &best_idx) {
  best_idx = -1;
  double best_dist2 = -1.0;
  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    if (static_cast<int>(idx) == min_x || static_cast<int>(idx) == max_x)
      continue;
    const Vec3 rel = points[idx] - points[static_cast<std::size_t>(min_x)];
    const double dist2 = norm2(cross(line, rel));
    if (dist2 > best_dist2) {
      best_dist2 = dist2;
      best_idx = static_cast<int>(idx);
    }
  }
  return best_dist2;
}

double farthest_plane_distance_scalar(const std::vector<Vec3> &points,
                                      int min_x, int max_x, int far_line,
                                      const Vec3 &plane_normal,
                                      int &best_idx) {
  best_idx = -1;
  double best_dist = -1.0;
  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    if (static_cast<int>(idx) == min_x || static_cast<int>(idx) == max_x ||
        static_cast<int>(idx) == far_line) {
      continue;
    }
    const double dist = std::fabs(dot(
        plane_normal, points[idx] - points[static_cast<std::size_t>(min_x)]));
    if (dist > best_dist) {
      best_dist = dist;
      best_idx = static_cast<int>(idx);
    }
  }
  return best_dist;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
double farthest_line_distance2_avx2(const double *xs, const double *ys,
                                    const double *zs, std::size_t point_count,
                                    int min_x, int max_x, const Vec3 &origin,
                                    const Vec3 &line, int &best_idx) noexcept {
  const __m256d ox = _mm256_set1_pd(origin.x);
  const __m256d oy = _mm256_set1_pd(origin.y);
  const __m256d oz = _mm256_set1_pd(origin.z);
  const __m256d lx = _mm256_set1_pd(line.x);
  const __m256d ly = _mm256_set1_pd(line.y);
  const __m256d lz = _mm256_set1_pd(line.z);
  alignas(32) double lane[4];
  best_idx = -1;
  double best_dist2 = -1.0;
  std::size_t idx = 0;
  for (; idx + 4 <= point_count; idx += 4) {
    const __m256d px = _mm256_loadu_pd(xs + idx);
    const __m256d py = _mm256_loadu_pd(ys + idx);
    const __m256d pz = _mm256_loadu_pd(zs + idx);
    const __m256d rx = _mm256_sub_pd(px, ox);
    const __m256d ry = _mm256_sub_pd(py, oy);
    const __m256d rz = _mm256_sub_pd(pz, oz);
    const __m256d cx =
        _mm256_sub_pd(_mm256_mul_pd(ly, rz), _mm256_mul_pd(lz, ry));
    const __m256d cy =
        _mm256_sub_pd(_mm256_mul_pd(lz, rx), _mm256_mul_pd(lx, rz));
    const __m256d cz =
        _mm256_sub_pd(_mm256_mul_pd(lx, ry), _mm256_mul_pd(ly, rx));
    const __m256d dist2 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(cx, cx), _mm256_mul_pd(cy, cy)),
        _mm256_mul_pd(cz, cz));
    _mm256_store_pd(lane, dist2);
    for (int lane_idx = 0; lane_idx < 4; ++lane_idx) {
      const int point_idx = static_cast<int>(idx) + lane_idx;
      if (point_idx == min_x || point_idx == max_x)
        continue;
      if (lane[lane_idx] > best_dist2) {
        best_dist2 = lane[lane_idx];
        best_idx = point_idx;
      }
    }
  }
  for (; idx < point_count; ++idx) {
    const int point_idx = static_cast<int>(idx);
    if (point_idx == min_x || point_idx == max_x)
      continue;
    const double rx = xs[idx] - origin.x;
    const double ry = ys[idx] - origin.y;
    const double rz = zs[idx] - origin.z;
    const double cx = line.y * rz - line.z * ry;
    const double cy = line.z * rx - line.x * rz;
    const double cz = line.x * ry - line.y * rx;
    const double dist2 = cx * cx + cy * cy + cz * cz;
    if (dist2 > best_dist2) {
      best_dist2 = dist2;
      best_idx = point_idx;
    }
  }
  return best_dist2;
}

TITAN_TARGET_AVX2
double farthest_plane_distance_avx2(const double *xs, const double *ys,
                                    const double *zs, std::size_t point_count,
                                    int min_x, int max_x, int far_line,
                                    const Vec3 &origin,
                                    const Vec3 &plane_normal,
                                    int &best_idx) noexcept {
  const __m256d ox = _mm256_set1_pd(origin.x);
  const __m256d oy = _mm256_set1_pd(origin.y);
  const __m256d oz = _mm256_set1_pd(origin.z);
  const __m256d nx = _mm256_set1_pd(plane_normal.x);
  const __m256d ny = _mm256_set1_pd(plane_normal.y);
  const __m256d nz = _mm256_set1_pd(plane_normal.z);
  const __m256d sign_mask =
      _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
  alignas(32) double lane[4];
  best_idx = -1;
  double best_dist = -1.0;
  std::size_t idx = 0;
  for (; idx + 4 <= point_count; idx += 4) {
    const __m256d px = _mm256_loadu_pd(xs + idx);
    const __m256d py = _mm256_loadu_pd(ys + idx);
    const __m256d pz = _mm256_loadu_pd(zs + idx);
    const __m256d rx = _mm256_sub_pd(px, ox);
    const __m256d ry = _mm256_sub_pd(py, oy);
    const __m256d rz = _mm256_sub_pd(pz, oz);
    const __m256d dot_v = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(nx, rx), _mm256_mul_pd(ny, ry)),
        _mm256_mul_pd(nz, rz));
    const __m256d abs_dot = _mm256_and_pd(dot_v, sign_mask);
    _mm256_store_pd(lane, abs_dot);
    for (int lane_idx = 0; lane_idx < 4; ++lane_idx) {
      const int point_idx = static_cast<int>(idx) + lane_idx;
      if (point_idx == min_x || point_idx == max_x || point_idx == far_line)
        continue;
      if (lane[lane_idx] > best_dist) {
        best_dist = lane[lane_idx];
        best_idx = point_idx;
      }
    }
  }
  for (; idx < point_count; ++idx) {
    const int point_idx = static_cast<int>(idx);
    if (point_idx == min_x || point_idx == max_x || point_idx == far_line)
      continue;
    const double dist = std::fabs(plane_normal.x * (xs[idx] - origin.x) +
                                  plane_normal.y * (ys[idx] - origin.y) +
                                  plane_normal.z * (zs[idx] - origin.z));
    if (dist > best_dist) {
      best_dist = dist;
      best_idx = point_idx;
    }
  }
  return best_dist;
}

TITAN_TARGET_AVX2
void collect_visible_faces_avx2(const PbctopoHullScratch &scratch,
                                const Vec3 &point,
                                const HullTolerance &tolerance,
                                std::vector<std::size_t> &visible_faces) noexcept {
  const __m256d px = _mm256_set1_pd(point.x);
  const __m256d py = _mm256_set1_pd(point.y);
  const __m256d pz = _mm256_set1_pd(point.z);
  alignas(32) double lane[4];
  visible_faces.clear();
  const std::size_t face_count = scratch.face_indices.size();
  std::size_t idx = 0;
  for (; idx + 4 <= face_count; idx += 4) {
    const __m256d nx = _mm256_loadu_pd(scratch.face_nx.data() + idx);
    const __m256d ny = _mm256_loadu_pd(scratch.face_ny.data() + idx);
    const __m256d nz = _mm256_loadu_pd(scratch.face_nz.data() + idx);
    const __m256d dd = _mm256_loadu_pd(scratch.face_d.data() + idx);
    const __m256d dist = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(nx, px), _mm256_mul_pd(ny, py)),
        _mm256_add_pd(_mm256_mul_pd(nz, pz), dd));
    _mm256_store_pd(lane, dist);
    for (int lane_idx = 0; lane_idx < 4; ++lane_idx) {
      if (lane[lane_idx] > tolerance.distance_eps)
        visible_faces.push_back(scratch.face_indices[idx + lane_idx]);
    }
  }
  for (; idx < face_count; ++idx) {
    const double dist = scratch.face_nx[idx] * point.x +
                        scratch.face_ny[idx] * point.y +
                        scratch.face_nz[idx] * point.z + scratch.face_d[idx];
    if (dist > tolerance.distance_eps)
      visible_faces.push_back(scratch.face_indices[idx]);
  }
}
#endif

double signed_distance_to_face(const HullFace &face,
                               const std::vector<Vec3> &points,
                               const Vec3 &point) {
  return dot(face.normal, point - points[static_cast<std::size_t>(face.a)]);
}

HullTolerance derive_hull_tolerance(const std::vector<Vec3> &points) {
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  for (const auto &point : points) {
    min_x = std::min(min_x, point.x);
    min_y = std::min(min_y, point.y);
    min_z = std::min(min_z, point.z);
    max_x = std::max(max_x, point.x);
    max_y = std::max(max_y, point.y);
    max_z = std::max(max_z, point.z);
  }
  const double dx = std::max(0.0, max_x - min_x);
  const double dy = std::max(0.0, max_y - min_y);
  const double dz = std::max(0.0, max_z - min_z);
  const double scale = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0);
  return {.distance_eps = kHullEps * scale, .area_eps = kHullEps * scale * scale};
}

bool orient_face(HullFace &face, const std::vector<Vec3> &points,
                 const Vec3 &interior_point, const HullTolerance &tolerance) {
  if (face.a < 0 || face.b < 0 || face.c < 0)
    return false;
  const Vec3 raw_normal =
      cross(points[static_cast<std::size_t>(face.b)] -
                points[static_cast<std::size_t>(face.a)],
            points[static_cast<std::size_t>(face.c)] -
                points[static_cast<std::size_t>(face.a)]);
  const double raw_norm = std::sqrt(norm2(raw_normal));
  if (!std::isfinite(raw_norm) || raw_norm <= tolerance.area_eps)
    return false;
  face.normal = raw_normal * (1.0 / raw_norm);
  const double signed_distance =
      signed_distance_to_face(face, points, interior_point);
  if (signed_distance > tolerance.distance_eps) {
    std::swap(face.b, face.c);
    face.normal = face.normal * -1.0;
  } else if (std::fabs(signed_distance) <= tolerance.distance_eps) {
    return false;
  }
  face.valid = true;
  return true;
}

bool build_initial_tetrahedron(const std::vector<Vec3> &points,
                               PbctopoHullScratch *scratch,
                               std::array<int, 4> &seed,
                               Vec3 &interior_point) {
  if (points.size() < 4)
    return false;

  int min_x = 0;
  int max_x = 0;
  for (std::size_t idx = 1; idx < points.size(); ++idx) {
    if (points[idx].x < points[static_cast<std::size_t>(min_x)].x)
      min_x = static_cast<int>(idx);
    if (points[idx].x > points[static_cast<std::size_t>(max_x)].x)
      max_x = static_cast<int>(idx);
  }
  const Vec3 origin = points[static_cast<std::size_t>(min_x)];
  const Vec3 line = points[static_cast<std::size_t>(max_x)] - origin;
  if (norm2(line) <= kHullEps * kHullEps)
    return false;

  int far_line = -1;
  double max_line_dist2 = -1.0;
  const bool use_avx2 = scratch != nullptr && avx2_available();
  if (use_avx2) {
    sync_soa(points, *scratch);
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    max_line_dist2 = farthest_line_distance2_avx2(
        scratch->xs.data(), scratch->ys.data(), scratch->zs.data(),
        points.size(), min_x, max_x, origin, line, far_line);
#else
    max_line_dist2 = farthest_line_distance2_scalar(points, min_x, max_x, line,
                                                    far_line);
#endif
  } else {
    max_line_dist2 =
        farthest_line_distance2_scalar(points, min_x, max_x, line, far_line);
  }
  if (far_line < 0 || max_line_dist2 <= kHullEps * kHullEps)
    return false;

  const Vec3 plane_normal =
      cross(points[static_cast<std::size_t>(max_x)] - origin,
            points[static_cast<std::size_t>(far_line)] - origin);
  int far_plane = -1;
  double max_plane_dist = -1.0;
  if (use_avx2) {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    max_plane_dist = farthest_plane_distance_avx2(
        scratch->xs.data(), scratch->ys.data(), scratch->zs.data(),
        points.size(), min_x, max_x, far_line, origin, plane_normal,
        far_plane);
#else
    max_plane_dist = farthest_plane_distance_scalar(
        points, min_x, max_x, far_line, plane_normal, far_plane);
#endif
  } else {
    max_plane_dist = farthest_plane_distance_scalar(
        points, min_x, max_x, far_line, plane_normal, far_plane);
  }
  if (far_plane < 0 || max_plane_dist <= kHullEps * kHullEps)
    return false;

  seed = {min_x, max_x, far_line, far_plane};
  interior_point = (points[static_cast<std::size_t>(seed[0])] +
                    points[static_cast<std::size_t>(seed[1])] +
                    points[static_cast<std::size_t>(seed[2])] +
                    points[static_cast<std::size_t>(seed[3])]) *
                   0.25;
  return true;
}

void add_point_to_hull(int point_index, const std::vector<Vec3> &points,
                       std::vector<HullFace> &faces, const Vec3 &interior_point,
                       const HullTolerance &tolerance,
                       PbctopoHullScratch *scratch) {
  std::vector<std::size_t> local_visible_faces;
  auto &visible_faces =
      scratch != nullptr ? scratch->visible_faces : local_visible_faces;
  const Vec3 point = points[static_cast<std::size_t>(point_index)];
  const bool use_avx2 = scratch != nullptr && avx2_available();
  if (use_avx2) {
    rebuild_face_plane_cache(faces, points, *scratch);
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
    collect_visible_faces_avx2(*scratch, point, tolerance, visible_faces);
#else
    collect_visible_faces_scalar(faces, points, point, tolerance,
                                 visible_faces);
#endif
  } else {
    collect_visible_faces_scalar(faces, points, point, tolerance, visible_faces);
  }
  if (visible_faces.empty())
    return;
  std::unordered_map<std::uint64_t, std::pair<int, int>> local_horizon_edges;
  auto &horizon_edges =
      scratch != nullptr ? scratch->horizon_edges : local_horizon_edges;
  horizon_edges.clear();
  auto push_edge = [&](int u, int v) {
    const std::uint64_t reverse_key = oriented_edge_key(v, u);
    const auto it = horizon_edges.find(reverse_key);
    if (it != horizon_edges.end()) {
      horizon_edges.erase(it);
    } else {
      horizon_edges.emplace(oriented_edge_key(u, v), std::make_pair(u, v));
    }
  };

  for (std::size_t face_idx : visible_faces) {
    HullFace &face = faces[face_idx];
    push_edge(face.a, face.b);
    push_edge(face.b, face.c);
    push_edge(face.c, face.a);
    face.valid = false;
  }

  for (const auto &kv : horizon_edges) {
    HullFace face;
    face.a = kv.second.first;
    face.b = kv.second.second;
    face.c = point_index;
    if (orient_face(face, points, interior_point, tolerance))
      faces.push_back(face);
  }
  faces.erase(std::remove_if(faces.begin(), faces.end(),
                             [](const HullFace &face) { return !face.valid; }),
              faces.end());
}

} // namespace

void PbctopoHullScratch::reserve(std::size_t point_count) {
  shifted_points.reserve(point_count);
  xs.reserve(point_count);
  ys.reserve(point_count);
  zs.reserve(point_count);
  face_nx.reserve(std::max<std::size_t>(16, point_count * 2));
  face_ny.reserve(std::max<std::size_t>(16, point_count * 2));
  face_nz.reserve(std::max<std::size_t>(16, point_count * 2));
  face_d.reserve(std::max<std::size_t>(16, point_count * 2));
  face_indices.reserve(std::max<std::size_t>(16, point_count * 2));
  is_seed.reserve(point_count);
  faces.reserve(std::max<std::size_t>(16, point_count * 2));
  visible_faces.reserve(point_count);
  horizon_edges.reserve(std::max<std::size_t>(16, point_count * 2));
}

void PbctopoHullScratch::clear_transient() {
  faces.clear();
  visible_faces.clear();
  horizon_edges.clear();
  face_nx.clear();
  face_ny.clear();
  face_nz.clear();
  face_d.clear();
  face_indices.clear();
}

bool avx2_available() noexcept { return titan_simd_dispatch::runtime_has_avx2(); }

void shift_points_by_spans(const std::vector<Vec3> &points,
                           const std::vector<PbctopoHullSpan> &spans,
                           const Vec3 &shift, std::vector<Vec3> &out,
                           PbctopoHullScratch *scratch) {
  out.resize(points.size());
  std::copy(points.begin(), points.end(), out.begin());
  for (const auto &span : spans) {
    assert(span.begin <= span.end);
    assert(span.end <= out.size());
    if (span.begin >= span.end || span.begin >= out.size())
      continue;
    const std::size_t span_end = std::min(span.end, out.size());
    for (std::size_t idx = span.begin; idx < span_end; ++idx) {
      out[idx].x += shift.x;
      out[idx].y += shift.y;
      out[idx].z += shift.z;
    }
  }
}

bool quickhull_volume(const std::vector<Vec3> &points, double &out_volume,
                      PbctopoHullScratch *scratch) {
  out_volume = 0.0;
  if (points.size() < 4)
    return false;

  PbctopoHullScratch local_scratch;
  PbctopoHullScratch *work = scratch != nullptr ? scratch : &local_scratch;
  work->reserve(points.size());
  work->clear_transient();
  if (avx2_available())
    sync_soa(points, *work);

  std::array<int, 4> seed{};
  Vec3 interior_point{};
  if (!build_initial_tetrahedron(points, work, seed, interior_point))
    return false;
  const HullTolerance tolerance = derive_hull_tolerance(points);

  auto &faces = work->faces;
  faces.clear();
  faces.reserve(points.size() * 2);
  const std::array<std::array<int, 3>, 4> seed_faces = {
      std::array<int, 3>{seed[0], seed[1], seed[2]},
      std::array<int, 3>{seed[0], seed[3], seed[1]},
      std::array<int, 3>{seed[0], seed[2], seed[3]},
      std::array<int, 3>{seed[1], seed[3], seed[2]}};
  for (const auto &seed_face : seed_faces) {
    HullFace face;
    face.a = seed_face[0];
    face.b = seed_face[1];
    face.c = seed_face[2];
    if (!orient_face(face, points, interior_point, tolerance))
      return false;
    faces.push_back(face);
  }

  work->is_seed.assign(points.size(), static_cast<unsigned char>(0));
  for (int seed_idx : seed)
    work->is_seed[static_cast<std::size_t>(seed_idx)] = 1;

  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    if (work->is_seed[idx] != 0)
      continue;
    add_point_to_hull(static_cast<int>(idx), points, faces, interior_point,
                      tolerance, work);
  }

  double signed_volume = 0.0;
  std::size_t valid_face_count = 0;
  for (const auto &face : faces) {
    if (!face.valid)
      continue;
    ++valid_face_count;
    signed_volume +=
        dot(points[static_cast<std::size_t>(face.a)],
            cross(points[static_cast<std::size_t>(face.b)],
                  points[static_cast<std::size_t>(face.c)]));
  }
  if (valid_face_count < 4)
    return false;

  out_volume = std::fabs(signed_volume) / 6.0;
  return std::isfinite(out_volume) && out_volume > kHullEps;
}

bool quickhull_volume_shifted_spans(
    const std::vector<Vec3> &points,
    const std::vector<PbctopoHullSpan> &spans, const Vec3 &shift,
    double &out_volume, PbctopoHullScratch *scratch) {
  PbctopoHullScratch local_scratch;
  PbctopoHullScratch *work = scratch != nullptr ? scratch : &local_scratch;
  shift_points_by_spans(points, spans, shift, work->shifted_points, work);
  return quickhull_volume(work->shifted_points, out_volume, work);
}

} // namespace pbctopo_hull
