#include "tabi_calderon_transfer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace TitanPBE::Tabi::calderon {
namespace {

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

[[nodiscard]] Vec3 operator+(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3 operator-(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3 operator*(double scale, const Vec3 &value) noexcept {
  return {scale * value.x, scale * value.y, scale * value.z};
}

[[nodiscard]] double dot(const Vec3 &a, const Vec3 &b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3 cross(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

[[nodiscard]] double squared_norm(const Vec3 &value) noexcept {
  return dot(value, value);
}

[[nodiscard]] double norm(const Vec3 &value) noexcept {
  return std::sqrt(squared_norm(value));
}

[[nodiscard]] Vec3 normalized(const Vec3 &value) noexcept {
  const double length = norm(value);
  return length > 1.0e-15 ? (1.0 / length) * value : Vec3{};
}

[[nodiscard]] Vec3 panel_point(const Panel &panel) noexcept {
  return {panel.x, panel.y, panel.z};
}

[[nodiscard]] Vec3 panel_normal(const Panel &panel) noexcept {
  return normalized({panel.nx, panel.ny, panel.nz});
}

[[nodiscard]] Vec3 mesh_vertex(const SurfaceMesh &mesh,
                                std::size_t index) noexcept {
  const Vertex &vertex = mesh.vertices[index];
  return {vertex.x, vertex.y, vertex.z};
}

struct ClosestSurfacePoint {
  Vec3 point{};
  Vec3 normal{};
  double distance_squared{std::numeric_limits<double>::infinity()};
  std::size_t matched_index{0};
  std::size_t atom_index{0};
  bool same_owner{false};
  bool owner_ambiguous{false};
  bool positive_alternative{false};
};

struct TrianglePoint {
  Vec3 point{};
  double wa{0.0};
  double wb{0.0};
  double wc{0.0};
};

[[nodiscard]] TrianglePoint closest_point_on_triangle(
    const Vec3 &point, const Vec3 &a, const Vec3 &b, const Vec3 &c) noexcept {
  // Real-Time Collision Detection, point/triangle closest-point regions.
  const Vec3 ab = b - a;
  const Vec3 ac = c - a;
  const Vec3 ap = point - a;
  const double d1 = dot(ab, ap);
  const double d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) {
    return {a, 1.0, 0.0, 0.0};
  }
  const Vec3 bp = point - b;
  const double d3 = dot(ab, bp);
  const double d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) {
    return {b, 0.0, 1.0, 0.0};
  }
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double factor = d1 / std::max(1.0e-300, d1 - d3);
    return {a + factor * ab, 1.0 - factor, factor, 0.0};
  }
  const Vec3 cp = point - c;
  const double d5 = dot(ab, cp);
  const double d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) {
    return {c, 0.0, 0.0, 1.0};
  }
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double factor = d2 / std::max(1.0e-300, d2 - d6);
    return {a + factor * ac, 1.0 - factor, 0.0, factor};
  }
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const Vec3 bc = c - b;
    const double factor = (d4 - d3) /
                          std::max(1.0e-300, (d4 - d3) + (d5 - d6));
    return {b + factor * bc, 0.0, 1.0 - factor, factor};
  }
  const double denominator = 1.0 / std::max(1.0e-300, va + vb + vc);
  const double wb = vb * denominator;
  const double wc = vc * denominator;
  return {a + wb * ab + wc * ac, 1.0 - wb - wc, wb, wc};
}

[[nodiscard]] Vec3 vertex_sdf_normal(const SurfaceMesh &mesh,
                                     std::size_t index) noexcept {
  const Vertex &vertex = mesh.vertices[index];
  return normalized({vertex.nx, vertex.ny, vertex.nz});
}

[[nodiscard]] Vec3 interpolated_sdf_normal(const SurfaceMesh &mesh,
                                           const Triangle &triangle,
                                           const TrianglePoint &point) noexcept {
  return normalized(point.wa * vertex_sdf_normal(mesh, triangle.v0) +
                    point.wb * vertex_sdf_normal(mesh, triangle.v1) +
                    point.wc * vertex_sdf_normal(mesh, triangle.v2));
}

[[nodiscard]] bool triangle_has_owner(const SurfaceMesh &mesh,
                                      const Triangle &triangle,
                                      std::size_t atom_index) noexcept {
  return triangle.atom_index == atom_index ||
         mesh.vertices[triangle.v0].atom_index == atom_index ||
         mesh.vertices[triangle.v1].atom_index == atom_index ||
         mesh.vertices[triangle.v2].atom_index == atom_index;
}

[[nodiscard]] ClosestSurfacePoint closest_auxiliary_surface_point(
    const Vec3 &point, const Vec3 &target_normal, std::size_t target_atom_index,
    const SurfaceMesh &mesh, GeometryNormalMatching matching,
    double distance_slack, double min_normal_dot) {
  ClosestSurfacePoint best;
  std::vector<double> distances(mesh.triangles.size(),
                                std::numeric_limits<double>::infinity());
  std::vector<TrianglePoint> points(mesh.triangles.size());
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
    const Triangle &triangle = mesh.triangles[index];
    if (triangle.v0 >= mesh.vertices.size() || triangle.v1 >= mesh.vertices.size() ||
        triangle.v2 >= mesh.vertices.size()) {
      continue;
    }
    const Vec3 a = mesh_vertex(mesh, triangle.v0);
    const Vec3 b = mesh_vertex(mesh, triangle.v1);
    const Vec3 c = mesh_vertex(mesh, triangle.v2);
    points[index] = closest_point_on_triangle(point, a, b, c);
    distances[index] = squared_norm(point - points[index].point);
    if (distances[index] < best.distance_squared) {
      best.distance_squared = distances[index];
    }
  }
  const double nearest_distance = std::sqrt(best.distance_squared);
  int best_tier = -1;
  double best_dot = -std::numeric_limits<double>::infinity();
  std::size_t distinct_owner = 0;
  std::vector<std::size_t> owners;
  for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
    if (!std::isfinite(distances[index])) {
      continue;
    }
    const double distance = std::sqrt(distances[index]);
    if (distance > nearest_distance + distance_slack) {
      continue;
    }
    const Triangle &triangle = mesh.triangles[index];
    const Vec3 a = mesh_vertex(mesh, triangle.v0);
    const Vec3 b = mesh_vertex(mesh, triangle.v1);
    const Vec3 c = mesh_vertex(mesh, triangle.v2);
    const Vec3 sdf_normal =
        interpolated_sdf_normal(mesh, triangle, points[index]);
    const Vec3 candidate_normal =
        matching == GeometryNormalMatching::OwnerSdfAware
            ? sdf_normal
            : normalized(cross(b - a, c - a));
    const double candidate_dot = dot(target_normal, candidate_normal);
    const double attribution_dot = dot(target_normal, sdf_normal);
    const bool same_owner =
        triangle_has_owner(mesh, triangle, target_atom_index);
    const bool aligned = candidate_dot >= min_normal_dot;
    const int tier = matching == GeometryNormalMatching::OwnerSdfAware
                         ? ((same_owner && aligned) ? 3
                            : aligned              ? 2
                            : same_owner           ? 1
                                                   : 0)
                         : 0;
    if (std::find(owners.begin(), owners.end(), triangle.atom_index) ==
        owners.end()) {
      owners.push_back(triangle.atom_index);
      ++distinct_owner;
    }
    if (attribution_dot >= min_normal_dot) {
      best.positive_alternative = true;
    }
    if (matching == GeometryNormalMatching::FacetNearest &&
        distances[index] > best.distance_squared + 1.0e-24) {
      continue;
    }
    const bool better =
        tier > best_tier ||
        (tier == best_tier && candidate_dot > best_dot + 1.0e-15) ||
        (tier == best_tier && std::abs(candidate_dot - best_dot) <= 1.0e-15 &&
         (distances[index] < best.distance_squared - 1.0e-24 ||
          (std::abs(distances[index] - best.distance_squared) <= 1.0e-24 &&
           index < best.matched_index)));
    if (better) {
      best.point = points[index].point;
      best.normal = candidate_normal;
      best.distance_squared = distances[index];
      best.matched_index = index;
      best.atom_index = triangle.atom_index;
      best.same_owner = same_owner;
      best_tier = tier;
      best_dot = candidate_dot;
    }
  }
  best.owner_ambiguous = distinct_owner > 1;
  return best;
}

[[nodiscard]] ClosestSurfacePoint closest_primary_panel(
    const Vec3 &point, const Vec3 &target_normal, std::size_t target_atom_index,
    std::span<const Panel> primary_panels, GeometryNormalMatching matching,
    double distance_slack, double min_normal_dot) {
  ClosestSurfacePoint best;
  for (std::size_t index = 0; index < primary_panels.size(); ++index) {
    best.distance_squared = std::min(
        best.distance_squared,
        squared_norm(point - panel_point(primary_panels[index])));
  }
  const double nearest_distance = std::sqrt(best.distance_squared);
  int best_tier = -1;
  double best_dot = -std::numeric_limits<double>::infinity();
  std::vector<std::size_t> owners;
  for (std::size_t index = 0; index < primary_panels.size(); ++index) {
    const Panel &panel = primary_panels[index];
    const double distance_squared = squared_norm(point - panel_point(panel));
    const double distance = std::sqrt(distance_squared);
    if (distance > nearest_distance + distance_slack) {
      continue;
    }
    const double candidate_dot = dot(target_normal, panel_normal(panel));
    const bool same_owner = panel.atom_index == target_atom_index;
    const bool aligned = candidate_dot >= min_normal_dot;
    const int tier = matching == GeometryNormalMatching::OwnerSdfAware
                         ? ((same_owner && aligned) ? 3
                            : aligned              ? 2
                            : same_owner           ? 1
                                                   : 0)
                         : 0;
    if (std::find(owners.begin(), owners.end(), panel.atom_index) == owners.end()) {
      owners.push_back(panel.atom_index);
    }
    if (aligned) {
      best.positive_alternative = true;
    }
    if (matching == GeometryNormalMatching::FacetNearest &&
        distance_squared > best.distance_squared + 1.0e-24) {
      continue;
    }
    const bool better =
        tier > best_tier ||
        (tier == best_tier && candidate_dot > best_dot + 1.0e-15) ||
        (tier == best_tier && std::abs(candidate_dot - best_dot) <= 1.0e-15 &&
         (distance_squared < best.distance_squared - 1.0e-24 ||
          (std::abs(distance_squared - best.distance_squared) <= 1.0e-24 &&
           index < best.matched_index)));
    if (better) {
      best.point = panel_point(panel);
      best.normal = panel_normal(panel);
      best.distance_squared = distance_squared;
      best.matched_index = index;
      best.atom_index = panel.atom_index;
      best.same_owner = same_owner;
      best_tier = tier;
      best_dot = candidate_dot;
    }
  }
  best.owner_ambiguous = owners.size() > 1;
  return best;
}

[[nodiscard]] std::vector<std::size_t> sample_indices(std::size_t count,
                                                        std::size_t cap) {
  const std::size_t samples = cap == 0 ? count : std::min(count, cap);
  std::vector<std::size_t> indices;
  indices.reserve(samples);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    indices.push_back((sample * count) / samples);
  }
  return indices;
}

[[nodiscard]] std::vector<Vec3>
build_auxiliary_vertex_normals(const SurfaceMesh &mesh) {
  std::vector<Vec3> normals(mesh.vertices.size());
  for (const Triangle &triangle : mesh.triangles) {
    if (triangle.v0 >= normals.size() || triangle.v1 >= normals.size() ||
        triangle.v2 >= normals.size()) {
      continue;
    }
    const Vec3 a = mesh_vertex(mesh, triangle.v0);
    const Vec3 b = mesh_vertex(mesh, triangle.v1);
    const Vec3 c = mesh_vertex(mesh, triangle.v2);
    const Vec3 weighted_normal = cross(b - a, c - a);
    normals[triangle.v0] = normals[triangle.v0] + weighted_normal;
    normals[triangle.v1] = normals[triangle.v1] + weighted_normal;
    normals[triangle.v2] = normals[triangle.v2] + weighted_normal;
  }
  for (Vec3 &normal : normals) {
    normal = normalized(normal);
  }
  return normals;
}

[[nodiscard]] double relative_l2(std::span<const double> values,
                                  double expected) noexcept {
  double squared = 0.0;
  for (const double value : values) {
    const double difference = value - expected;
    squared += difference * difference;
  }
  return std::sqrt(squared) /
         std::max(1.0e-300, std::sqrt(static_cast<double>(values.size())) *
                                  std::max(1.0, std::abs(expected)));
}

} // namespace

std::vector<double> apply_primary_to_auxiliary_trace_transfer(
    const PrimaryAuxiliaryTransferResult &transfer,
    std::span<const double> primary_values) {
  if (primary_values.size() != transfer.primary_panels) {
    throw std::runtime_error("primary-to-auxiliary transfer input size mismatch");
  }
  std::vector<double> out(transfer.primary_to_auxiliary_p1.size(), 0.0);
  for (std::size_t row = 0; row < out.size(); ++row) {
    for (const TraceTransferWeight &weight :
         transfer.primary_to_auxiliary_p1[row]) {
      if (weight.index >= primary_values.size()) {
        throw std::runtime_error("primary-to-auxiliary transfer index mismatch");
      }
      out[row] += weight.value * primary_values[weight.index];
    }
  }
  return out;
}

std::vector<double> apply_auxiliary_to_primary_adjoint_trace_transfer(
    const PrimaryAuxiliaryTransferResult &transfer,
    std::span<const double> auxiliary_values) {
  if (auxiliary_values.size() != transfer.auxiliary_vertices) {
    throw std::runtime_error("auxiliary-to-primary transfer input size mismatch");
  }
  if (transfer.auxiliary_to_primary_adjoint.size() != transfer.primary_panels) {
    throw std::runtime_error("auxiliary-to-primary transfer stencil mismatch");
  }
  std::vector<double> out(transfer.primary_panels, 0.0);
  for (std::size_t row = 0; row < out.size(); ++row) {
    for (const TraceTransferWeight &weight :
         transfer.auxiliary_to_primary_adjoint[row]) {
      if (weight.index >= auxiliary_values.size()) {
        throw std::runtime_error("auxiliary-to-primary transfer index mismatch");
      }
      out[row] += weight.value * auxiliary_values[weight.index];
    }
  }
  return out;
}

PrimaryAuxiliaryTransferResult build_primary_auxiliary_trace_transfer(
    std::span<const Panel> primary_panels,
    const AuxiliarySurfaceResult &auxiliary,
    const PrimaryAuxiliaryTransferConfig &config) {
  const auto begin = std::chrono::steady_clock::now();
  PrimaryAuxiliaryTransferResult result;
  result.primary_panels = primary_panels.size();
  result.auxiliary_vertices = auxiliary.mesh.vertices.size();
  if (primary_panels.empty() || auxiliary.status != "pass" ||
      !auxiliary.trace_spaces.quality.valid || auxiliary.mesh.triangles.empty() ||
      auxiliary.mesh.vertices.empty() ||
      auxiliary.trace_spaces.p1_vertex_lumped_mass.size() !=
          auxiliary.mesh.vertices.size()) {
    result.status = "invalid_input";
    return result;
  }
  if (config.nearest_primary_panels == 0 ||
      config.nearest_primary_panels > primary_panels.size() ||
      !(config.max_primary_to_auxiliary_distance > 0.0) ||
      !(config.max_auxiliary_to_primary_distance > 0.0) ||
      config.min_normal_dot < -1.0 || config.min_normal_dot > 1.0 ||
      config.min_normal_alignment_fraction < 0.0 ||
      config.min_normal_alignment_fraction > 1.0 ||
      !(config.max_relative_area_mismatch >= 0.0) ||
      !(config.normal_candidate_distance_slack >= 0.0) ||
      !std::isfinite(config.max_primary_to_auxiliary_distance) ||
      !std::isfinite(config.max_auxiliary_to_primary_distance) ||
      !std::isfinite(config.min_normal_dot) ||
      !std::isfinite(config.min_normal_alignment_fraction) ||
      !std::isfinite(config.max_relative_area_mismatch) ||
      !std::isfinite(config.normal_candidate_distance_slack)) {
    result.status = "invalid_config";
    return result;
  }

  try {
    for (const Panel &panel : primary_panels) {
      const Vec3 point = panel_point(panel);
      const Vec3 normal = panel_normal(panel);
      if (!(panel.area > 0.0) || !std::isfinite(panel.area) ||
          !std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z) || squared_norm(normal) < 0.5) {
        result.status = "invalid_primary_panel";
        return result;
      }
      result.primary_area_a2 += panel.area;
    }
    result.auxiliary_area_a2 = auxiliary.trace_spaces.quality.total_area;
    result.relative_area_mismatch =
        std::abs(result.primary_area_a2 - result.auxiliary_area_a2) /
        std::max(1.0e-300, result.auxiliary_area_a2);
    const std::vector<Vec3> auxiliary_normals =
        build_auxiliary_vertex_normals(auxiliary.mesh);
    result.normal_matching_policy =
        config.normal_matching == GeometryNormalMatching::OwnerSdfAware
            ? "owner-sdf-aware"
            : "facet-nearest";

    // Geometry compatibility is measured independently of the sparse transfer
    // interpolation. This keeps a passing algebraic adjointness check from
    // disguising a physically incompatible auxiliary surface.
    const std::vector<std::size_t> primary_samples =
        sample_indices(primary_panels.size(), config.geometry_sample_cap);
    double primary_distance_squared = 0.0;
    double primary_sample_area = 0.0;
    double primary_aligned_area = 0.0;
    std::size_t primary_covered = 0;
    std::size_t primary_normal_aligned = 0;
    std::size_t primary_same_owner = 0;
    std::size_t primary_owner_ambiguous = 0;
    for (const std::size_t index : primary_samples) {
      const Panel &panel = primary_panels[index];
      const ClosestSurfacePoint closest =
          closest_auxiliary_surface_point(
              panel_point(panel), panel_normal(panel), panel.atom_index,
              auxiliary.mesh, config.normal_matching,
              config.normal_candidate_distance_slack, config.min_normal_dot);
      const double distance = std::sqrt(closest.distance_squared);
      const double normal_dot = dot(panel_normal(panel), closest.normal);
      primary_distance_squared += closest.distance_squared;
      primary_sample_area += panel.area;
      result.primary_to_auxiliary_max_distance =
          std::max(result.primary_to_auxiliary_max_distance, distance);
      result.primary_to_auxiliary_min_normal_dot =
          std::min(result.primary_to_auxiliary_min_normal_dot, normal_dot);
      if (distance <= config.max_primary_to_auxiliary_distance) {
        ++primary_covered;
      }
      if (normal_dot >= config.min_normal_dot) {
        ++primary_normal_aligned;
        primary_aligned_area += panel.area;
      } else {
        ++result.primary_negative_normal_samples;
        if (closest.positive_alternative) {
          ++result.primary_positive_alternative_samples;
        }
        if (result.normal_mismatch_samples.size() < config.mismatch_sample_cap) {
          result.normal_mismatch_samples.push_back(
              {"primary-to-auxiliary", index, panel.atom_index,
               closest.matched_index, closest.atom_index, distance, normal_dot,
               closest.same_owner, closest.owner_ambiguous,
               closest.positive_alternative});
        }
      }
      if (closest.same_owner) {
        ++primary_same_owner;
      }
      if (closest.owner_ambiguous) {
        ++primary_owner_ambiguous;
      }
    }
    result.primary_geometry_samples = primary_samples.size();
    result.primary_to_auxiliary_rms_distance =
        std::sqrt(primary_distance_squared /
                  std::max<std::size_t>(1, primary_samples.size()));
    result.primary_coverage_fraction =
        static_cast<double>(primary_covered) /
        std::max<std::size_t>(1, primary_samples.size());
    result.primary_normal_alignment_fraction =
        static_cast<double>(primary_normal_aligned) /
        std::max<std::size_t>(1, primary_samples.size());
    result.primary_area_weighted_normal_alignment_fraction =
        primary_aligned_area / std::max(1.0e-300, primary_sample_area);
    result.primary_same_owner_fraction =
        static_cast<double>(primary_same_owner) /
        std::max<std::size_t>(1, primary_samples.size());
    result.primary_owner_ambiguity_fraction =
        static_cast<double>(primary_owner_ambiguous) /
        std::max<std::size_t>(1, primary_samples.size());

    const std::vector<std::size_t> auxiliary_samples =
        sample_indices(auxiliary.mesh.vertices.size(), config.geometry_sample_cap);
    double auxiliary_distance_squared = 0.0;
    double auxiliary_sample_mass = 0.0;
    double auxiliary_aligned_mass = 0.0;
    std::size_t auxiliary_covered = 0;
    std::size_t auxiliary_normal_aligned = 0;
    std::size_t auxiliary_same_owner = 0;
    std::size_t auxiliary_owner_ambiguous = 0;
    for (const std::size_t index : auxiliary_samples) {
      const Vec3 vertex = mesh_vertex(auxiliary.mesh, index);
      const Vertex &source_vertex = auxiliary.mesh.vertices[index];
      const Vec3 source_normal =
          config.normal_matching == GeometryNormalMatching::OwnerSdfAware
              ? vertex_sdf_normal(auxiliary.mesh, index)
              : auxiliary_normals[index];
      const ClosestSurfacePoint closest = closest_primary_panel(
          vertex, source_normal, source_vertex.atom_index, primary_panels,
          config.normal_matching, config.normal_candidate_distance_slack,
          config.min_normal_dot);
      const double distance = std::sqrt(closest.distance_squared);
      const double normal_dot = dot(source_normal, closest.normal);
      const double sample_mass =
          auxiliary.trace_spaces.p1_vertex_lumped_mass[index];
      auxiliary_distance_squared += closest.distance_squared;
      auxiliary_sample_mass += sample_mass;
      result.auxiliary_to_primary_max_distance =
          std::max(result.auxiliary_to_primary_max_distance, distance);
      result.auxiliary_to_primary_min_normal_dot =
          std::min(result.auxiliary_to_primary_min_normal_dot, normal_dot);
      if (distance <= config.max_auxiliary_to_primary_distance) {
        ++auxiliary_covered;
      }
      if (normal_dot >= config.min_normal_dot) {
        ++auxiliary_normal_aligned;
        auxiliary_aligned_mass += sample_mass;
      } else {
        ++result.auxiliary_negative_normal_samples;
        if (closest.positive_alternative) {
          ++result.auxiliary_positive_alternative_samples;
        }
        if (result.normal_mismatch_samples.size() < config.mismatch_sample_cap) {
          result.normal_mismatch_samples.push_back(
              {"auxiliary-to-primary", index, source_vertex.atom_index,
               closest.matched_index, closest.atom_index, distance, normal_dot,
               closest.same_owner, closest.owner_ambiguous,
               closest.positive_alternative});
        }
      }
      if (closest.same_owner) {
        ++auxiliary_same_owner;
      }
      if (closest.owner_ambiguous) {
        ++auxiliary_owner_ambiguous;
      }
    }
    result.auxiliary_geometry_samples = auxiliary_samples.size();
    result.auxiliary_to_primary_rms_distance =
        std::sqrt(auxiliary_distance_squared /
                  std::max<std::size_t>(1, auxiliary_samples.size()));
    result.auxiliary_coverage_fraction =
        static_cast<double>(auxiliary_covered) /
        std::max<std::size_t>(1, auxiliary_samples.size());
    result.auxiliary_normal_alignment_fraction =
        static_cast<double>(auxiliary_normal_aligned) /
        std::max<std::size_t>(1, auxiliary_samples.size());
    result.auxiliary_area_weighted_normal_alignment_fraction =
        auxiliary_aligned_mass / std::max(1.0e-300, auxiliary_sample_mass);
    result.auxiliary_same_owner_fraction =
        static_cast<double>(auxiliary_same_owner) /
        std::max<std::size_t>(1, auxiliary_samples.size());
    result.auxiliary_owner_ambiguity_fraction =
        static_cast<double>(auxiliary_owner_ambiguous) /
        std::max<std::size_t>(1, auxiliary_samples.size());

    result.primary_to_auxiliary_p1.resize(auxiliary.mesh.vertices.size());
    result.auxiliary_to_primary_adjoint.resize(primary_panels.size());
    for (std::size_t vertex_index = 0;
         vertex_index < auxiliary.mesh.vertices.size(); ++vertex_index) {
      const Vec3 vertex = mesh_vertex(auxiliary.mesh, vertex_index);
      std::vector<std::pair<double, std::size_t>> nearest;
      nearest.reserve(config.nearest_primary_panels);
      for (std::size_t panel_index = 0; panel_index < primary_panels.size();
           ++panel_index) {
        const double distance_squared =
            squared_norm(vertex - panel_point(primary_panels[panel_index]));
        if (nearest.size() < config.nearest_primary_panels) {
          nearest.push_back({distance_squared, panel_index});
          continue;
        }
        auto worst = std::max_element(
            nearest.begin(), nearest.end(),
            [](const auto &left, const auto &right) { return left.first < right.first; });
        if (distance_squared < worst->first) {
          *worst = {distance_squared, panel_index};
        }
      }
      std::sort(nearest.begin(), nearest.end());
      TraceTransferStencil &forward = result.primary_to_auxiliary_p1[vertex_index];
      if (nearest.front().first <= 1.0e-24) {
        forward.push_back({nearest.front().second, 1.0});
      } else {
        double normalization = 0.0;
        for (const auto &[distance_squared, panel_index] : nearest) {
          const double weight = 1.0 / distance_squared;
          forward.push_back({panel_index, weight});
          normalization += weight;
        }
        for (TraceTransferWeight &weight : forward) {
          weight.value /= normalization;
        }
      }
      const double auxiliary_mass =
          auxiliary.trace_spaces.p1_vertex_lumped_mass[vertex_index];
      for (const TraceTransferWeight &weight : forward) {
        const double primary_mass = primary_panels[weight.index].area;
        result.auxiliary_to_primary_adjoint[weight.index].push_back(
            {vertex_index, auxiliary_mass * weight.value / primary_mass});
      }
    }

    std::vector<double> primary_constant(primary_panels.size(), 1.0);
    const std::vector<double> auxiliary_constant =
        apply_primary_to_auxiliary_trace_transfer(result, primary_constant);
    result.forward_constant_relative = relative_l2(auxiliary_constant, 1.0);
    std::vector<double> primary_probe(primary_panels.size());
    std::vector<double> auxiliary_probe(auxiliary.mesh.vertices.size());
    for (std::size_t i = 0; i < primary_probe.size(); ++i) {
      primary_probe[i] = (i % 3 == 0) ? -0.75 : 1.25;
    }
    for (std::size_t i = 0; i < auxiliary_probe.size(); ++i) {
      auxiliary_probe[i] = (i % 5 < 2) ? 0.5 : -1.0;
    }
    const std::vector<double> forward_probe =
        apply_primary_to_auxiliary_trace_transfer(result, primary_probe);
    const std::vector<double> reverse_probe =
        apply_auxiliary_to_primary_adjoint_trace_transfer(result, auxiliary_probe);
    double forward_inner = 0.0;
    double reverse_inner = 0.0;
    for (std::size_t i = 0; i < auxiliary_probe.size(); ++i) {
      forward_inner += auxiliary.trace_spaces.p1_vertex_lumped_mass[i] *
                       forward_probe[i] * auxiliary_probe[i];
    }
    for (std::size_t i = 0; i < primary_probe.size(); ++i) {
      reverse_inner += primary_panels[i].area * primary_probe[i] * reverse_probe[i];
    }
    result.adjoint_relative =
        std::abs(forward_inner - reverse_inner) /
        std::max({1.0e-300, std::abs(forward_inner), std::abs(reverse_inner)});
    const std::vector<double> reverse_constant =
        apply_auxiliary_to_primary_adjoint_trace_transfer(result,
                                                           auxiliary_constant);
    result.adjoint_reverse_constant_relative = relative_l2(reverse_constant, 1.0);

    const bool geometry_coverage = result.primary_coverage_fraction == 1.0 &&
                                   result.auxiliary_coverage_fraction == 1.0;
    const bool geometry_normals =
        result.primary_normal_alignment_fraction >=
            config.min_normal_alignment_fraction &&
        result.auxiliary_normal_alignment_fraction >=
            config.min_normal_alignment_fraction &&
        (!config.require_area_weighted_normal_alignment ||
         (result.primary_area_weighted_normal_alignment_fraction >=
              config.min_normal_alignment_fraction &&
          result.auxiliary_area_weighted_normal_alignment_fraction >=
              config.min_normal_alignment_fraction));
    const bool finite = std::isfinite(result.relative_area_mismatch) &&
                        std::isfinite(result.primary_to_auxiliary_rms_distance) &&
                        std::isfinite(result.auxiliary_to_primary_rms_distance) &&
                        std::isfinite(result.primary_to_auxiliary_min_normal_dot) &&
                        std::isfinite(result.auxiliary_to_primary_min_normal_dot) &&
                        std::isfinite(
                            result.primary_area_weighted_normal_alignment_fraction) &&
                        std::isfinite(
                            result.auxiliary_area_weighted_normal_alignment_fraction) &&
                        std::isfinite(result.forward_constant_relative) &&
                        std::isfinite(result.adjoint_relative) &&
                        std::isfinite(result.adjoint_reverse_constant_relative);
    if (!finite) {
      result.status = "nonfinite";
    } else if (result.forward_constant_relative > 1.0e-13 ||
               result.adjoint_relative > 1.0e-13) {
      result.status = "transfer_algebra_gap";
    } else if (result.relative_area_mismatch > config.max_relative_area_mismatch) {
      result.status = "geometry_area_gap";
    } else if (!geometry_coverage) {
      result.status = "geometry_coverage_gap";
    } else if (!geometry_normals) {
      result.status = "geometry_normal_gap";
    } else {
      result.status = "pass";
    }
  } catch (const std::exception &) {
    result.status = "transfer_failure";
  }
  const auto end = std::chrono::steady_clock::now();
  result.build_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  return result;
}

std::vector<double> apply_mass_compatible_primary_to_auxiliary(
    const MassCompatibleTransferResult &transfer,
    std::span<const double> primary_values) {
  if (primary_values.size() != transfer.primary_panels ||
      transfer.primary_to_auxiliary_p1.size() != transfer.auxiliary_vertices) {
    throw std::runtime_error("mass-compatible forward transfer size mismatch");
  }
  std::vector<double> out(transfer.auxiliary_vertices, 0.0);
  for (std::size_t row = 0; row < out.size(); ++row) {
    for (const TraceTransferWeight &weight :
         transfer.primary_to_auxiliary_p1[row]) {
      if (weight.index >= primary_values.size()) {
        throw std::runtime_error("mass-compatible forward transfer index mismatch");
      }
      out[row] += weight.value * primary_values[weight.index];
    }
  }
  return out;
}

std::vector<double> apply_mass_compatible_auxiliary_to_primary(
    const MassCompatibleTransferResult &transfer,
    std::span<const double> auxiliary_values) {
  if (auxiliary_values.size() != transfer.auxiliary_vertices ||
      transfer.auxiliary_to_primary_balanced_adjoint.size() !=
          transfer.primary_panels) {
    throw std::runtime_error("mass-compatible reverse transfer size mismatch");
  }
  std::vector<double> out(transfer.primary_panels, 0.0);
  for (std::size_t row = 0; row < out.size(); ++row) {
    for (const TraceTransferWeight &weight :
         transfer.auxiliary_to_primary_balanced_adjoint[row]) {
      if (weight.index >= auxiliary_values.size()) {
        throw std::runtime_error("mass-compatible reverse transfer index mismatch");
      }
      out[row] += weight.value * auxiliary_values[weight.index];
    }
  }
  return out;
}

MassCompatibleTransferResult
build_mass_compatible_primary_auxiliary_trace_transfer(
    std::span<const Panel> primary_panels,
    const AuxiliarySurfaceResult &auxiliary,
    const MassCompatibleTransferConfig &config) {
  const auto begin = std::chrono::steady_clock::now();
  MassCompatibleTransferResult result;
  result.primary_panels = primary_panels.size();
  result.auxiliary_vertices = auxiliary.mesh.vertices.size();
  if (config.max_balance_iterations == 0 ||
      !(config.balance_relative_tolerance > 0.0) ||
      !(config.support_floor > 0.0) ||
      !std::isfinite(config.balance_relative_tolerance) ||
      !std::isfinite(config.support_floor)) {
    result.status = "invalid_config";
    return result;
  }

  result.geometry = build_primary_auxiliary_trace_transfer(
      primary_panels, auxiliary, config.geometry);
  result.geometry_status = result.geometry.status;
  if (result.geometry.primary_to_auxiliary_p1.empty() ||
      result.geometry.primary_to_auxiliary_p1.size() !=
          auxiliary.mesh.vertices.size()) {
    result.status = "geometry_transfer_unavailable";
    return result;
  }

  struct TransportEdge {
    std::size_t row{0};
    std::size_t column{0};
    double kernel{0.0};
    double mass{0.0};
  };

  try {
    std::vector<TransportEdge> edges;
    edges.reserve(result.geometry.auxiliary_vertices *
                      config.geometry.nearest_primary_panels +
                  primary_panels.size());
    std::vector<bool> primary_supported(primary_panels.size(), false);
    for (std::size_t row = 0;
         row < result.geometry.primary_to_auxiliary_p1.size(); ++row) {
      for (const TraceTransferWeight &weight :
           result.geometry.primary_to_auxiliary_p1[row]) {
        edges.push_back(
            {row, weight.index, std::max(config.support_floor, weight.value), 0.0});
        primary_supported[weight.index] = true;
      }
    }

    // A panel-centred primary discretisation can contain more panels than the
    // auxiliary P1 vertex set. Guarantee column support with the nearest
    // auxiliary vertex; these local repair edges are reported explicitly.
    for (std::size_t panel_index = 0; panel_index < primary_panels.size();
         ++panel_index) {
      if (primary_supported[panel_index]) {
        continue;
      }
      ++result.unsupported_primary_panels_before_repair;
      std::size_t nearest_vertex = 0;
      double nearest_distance_squared = std::numeric_limits<double>::infinity();
      const Vec3 centre = panel_point(primary_panels[panel_index]);
      for (std::size_t vertex_index = 0;
           vertex_index < auxiliary.mesh.vertices.size(); ++vertex_index) {
        const double distance_squared = squared_norm(
            centre - mesh_vertex(auxiliary.mesh, vertex_index));
        if (distance_squared < nearest_distance_squared) {
          nearest_vertex = vertex_index;
          nearest_distance_squared = distance_squared;
        }
      }
      edges.push_back(
          {nearest_vertex, panel_index, config.support_floor, 0.0});
      ++result.support_repair_edges;
    }
    result.support_edges = edges.size();

    std::vector<double> auxiliary_mass =
        auxiliary.trace_spaces.p1_vertex_lumped_mass;
    std::vector<double> primary_mass(primary_panels.size(), 0.0);
    double auxiliary_total = 0.0;
    double primary_total = 0.0;
    for (const double value : auxiliary_mass) {
      auxiliary_total += value;
    }
    for (std::size_t panel = 0; panel < primary_panels.size(); ++panel) {
      primary_total += primary_panels[panel].area;
    }
    if (!(auxiliary_total > 0.0) || !(primary_total > 0.0)) {
      result.status = "invalid_mass";
      return result;
    }
    result.primary_mass_global_scale = auxiliary_total / primary_total;

    // Sparse local support may have several disconnected components. A single
    // global scale is then infeasible unless every component happens to carry
    // identical primary and auxiliary mass. Balance each local component while
    // retaining the global scale and component spread as diagnostics.
    const std::size_t row_count = auxiliary_mass.size();
    const std::size_t node_count = row_count + primary_mass.size();
    std::vector<std::size_t> parent(node_count);
    std::iota(parent.begin(), parent.end(), 0);
    const auto find_root = [&](auto &&self, std::size_t node) -> std::size_t {
      if (parent[node] != node) {
        parent[node] = self(self, parent[node]);
      }
      return parent[node];
    };
    const auto unite = [&](std::size_t left, std::size_t right) {
      const std::size_t left_root = find_root(find_root, left);
      const std::size_t right_root = find_root(find_root, right);
      if (left_root != right_root) {
        parent[right_root] = left_root;
      }
    };
    for (const TransportEdge &edge : edges) {
      unite(edge.row, row_count + edge.column);
    }
    std::vector<double> component_auxiliary_mass(node_count, 0.0);
    std::vector<double> component_primary_area(node_count, 0.0);
    for (std::size_t row = 0; row < auxiliary_mass.size(); ++row) {
      component_auxiliary_mass[find_root(find_root, row)] += auxiliary_mass[row];
    }
    for (std::size_t panel = 0; panel < primary_panels.size(); ++panel) {
      component_primary_area[find_root(find_root, row_count + panel)] +=
          primary_panels[panel].area;
    }
    std::vector<double> component_scale(node_count, 0.0);
    result.component_mass_scale_min = std::numeric_limits<double>::infinity();
    result.component_mass_scale_max = 0.0;
    for (std::size_t node = 0; node < node_count; ++node) {
      if (find_root(find_root, node) != node) {
        continue;
      }
      if (component_auxiliary_mass[node] == 0.0 &&
          component_primary_area[node] == 0.0) {
        continue;
      }
      if (!(component_auxiliary_mass[node] > 0.0) ||
          !(component_primary_area[node] > 0.0)) {
        result.status = "mass_support_component_gap";
        return result;
      }
      component_scale[node] =
          component_auxiliary_mass[node] / component_primary_area[node];
      result.component_mass_scale_min =
          std::min(result.component_mass_scale_min, component_scale[node]);
      result.component_mass_scale_max =
          std::max(result.component_mass_scale_max, component_scale[node]);
      ++result.support_components;
    }
    for (std::size_t panel = 0; panel < primary_panels.size(); ++panel) {
      const std::size_t root = find_root(find_root, row_count + panel);
      primary_mass[panel] = component_scale[root] * primary_panels[panel].area;
    }

    std::vector<double> log_row_scale(auxiliary_mass.size(), 0.0);
    std::vector<double> log_column_scale(primary_mass.size(), 0.0);
    std::vector<double> row_max(auxiliary_mass.size(),
                                -std::numeric_limits<double>::infinity());
    std::vector<double> column_max(primary_mass.size(),
                                   -std::numeric_limits<double>::infinity());
    std::vector<double> row_sum(auxiliary_mass.size(), 0.0);
    std::vector<double> column_sum(primary_mass.size(), 0.0);
    bool valid_iteration = true;
    for (std::size_t iteration = 0; iteration < config.max_balance_iterations;
         ++iteration) {
      std::fill(row_max.begin(), row_max.end(),
                -std::numeric_limits<double>::infinity());
      for (const TransportEdge &edge : edges) {
        row_max[edge.row] =
            std::max(row_max[edge.row],
                     std::log(edge.kernel) + log_column_scale[edge.column]);
      }
      std::fill(row_sum.begin(), row_sum.end(), 0.0);
      for (const TransportEdge &edge : edges) {
        row_sum[edge.row] += std::exp(std::log(edge.kernel) +
                                     log_column_scale[edge.column] -
                                     row_max[edge.row]);
      }
      for (std::size_t row = 0; row < log_row_scale.size(); ++row) {
        if (!(row_sum[row] > 0.0) || !std::isfinite(row_sum[row]) ||
            !std::isfinite(row_max[row])) {
          valid_iteration = false;
          break;
        }
        log_row_scale[row] = std::log(auxiliary_mass[row]) -
                             row_max[row] - std::log(row_sum[row]);
      }
      if (!valid_iteration) {
        break;
      }

      std::fill(column_max.begin(), column_max.end(),
                -std::numeric_limits<double>::infinity());
      for (const TransportEdge &edge : edges) {
        column_max[edge.column] =
            std::max(column_max[edge.column],
                     std::log(edge.kernel) + log_row_scale[edge.row]);
      }
      std::fill(column_sum.begin(), column_sum.end(), 0.0);
      for (const TransportEdge &edge : edges) {
        column_sum[edge.column] += std::exp(std::log(edge.kernel) +
                                           log_row_scale[edge.row] -
                                           column_max[edge.column]);
      }
      for (std::size_t column = 0; column < log_column_scale.size(); ++column) {
        if (!(column_sum[column] > 0.0) ||
            !std::isfinite(column_sum[column]) ||
            !std::isfinite(column_max[column])) {
          valid_iteration = false;
          break;
        }
        log_column_scale[column] = std::log(primary_mass[column]) -
                                   column_max[column] -
                                   std::log(column_sum[column]);
      }
      if (!valid_iteration) {
        break;
      }

      // Evaluate the true marginals periodically. Log-domain scaling prevents
      // overflow on large, weakly connected real-primary support graphs.
      if (iteration > 9 && (iteration + 1) % 10 != 0 &&
          iteration + 1 != config.max_balance_iterations) {
        continue;
      }
      std::fill(row_sum.begin(), row_sum.end(), 0.0);
      std::fill(column_sum.begin(), column_sum.end(), 0.0);
      for (const TransportEdge &edge : edges) {
        const double log_mass = std::log(edge.kernel) +
                                log_row_scale[edge.row] +
                                log_column_scale[edge.column];
        const double mass = std::exp(log_mass);
        row_sum[edge.row] += mass;
        column_sum[edge.column] += mass;
      }
      double residual = 0.0;
      for (std::size_t row = 0; row < row_sum.size(); ++row) {
        residual = std::max(
            residual, std::abs(row_sum[row] - auxiliary_mass[row]) /
                          std::max(1.0e-300, auxiliary_mass[row]));
      }
      for (std::size_t column = 0; column < column_sum.size(); ++column) {
        residual = std::max(
            residual, std::abs(column_sum[column] - primary_mass[column]) /
                          std::max(1.0e-300, primary_mass[column]));
      }
      result.balance_relative = residual;
      result.balance_iterations = iteration + 1;
      if (residual <= config.balance_relative_tolerance) {
        break;
      }
    }
    if (!valid_iteration || !std::isfinite(result.balance_relative)) {
      result.status = "mass_balance_nonfinite";
      return result;
    }

    result.primary_to_auxiliary_p1.resize(auxiliary_mass.size());
    result.auxiliary_to_primary_balanced_adjoint.resize(primary_mass.size());
    for (TransportEdge &edge : edges) {
      edge.mass = std::exp(std::log(edge.kernel) + log_row_scale[edge.row] +
                           log_column_scale[edge.column]);
      const double forward_weight = edge.mass / auxiliary_mass[edge.row];
      const double reverse_weight = edge.mass / primary_mass[edge.column];
      result.primary_to_auxiliary_p1[edge.row].push_back(
          {edge.column, forward_weight});
      result.auxiliary_to_primary_balanced_adjoint[edge.column].push_back(
          {edge.row, reverse_weight});
      result.max_forward_weight =
          std::max(result.max_forward_weight, forward_weight);
      result.max_reverse_weight =
          std::max(result.max_reverse_weight, reverse_weight);
    }
    for (TraceTransferStencil &stencil : result.primary_to_auxiliary_p1) {
      std::sort(stencil.begin(), stencil.end(),
                [](const TraceTransferWeight &left,
                   const TraceTransferWeight &right) {
                  return left.index < right.index;
                });
    }
    for (TraceTransferStencil &stencil :
         result.auxiliary_to_primary_balanced_adjoint) {
      std::sort(stencil.begin(), stencil.end(),
                [](const TraceTransferWeight &left,
                   const TraceTransferWeight &right) {
                  return left.index < right.index;
                });
    }

    const std::vector<double> primary_constant(primary_panels.size(), 1.0);
    const std::vector<double> auxiliary_constant(auxiliary_mass.size(), 1.0);
    result.forward_constant_relative = relative_l2(
        apply_mass_compatible_primary_to_auxiliary(result, primary_constant), 1.0);
    result.reverse_constant_relative = relative_l2(
        apply_mass_compatible_auxiliary_to_primary(result, auxiliary_constant), 1.0);

    std::vector<double> primary_probe(primary_panels.size());
    std::vector<double> auxiliary_probe(auxiliary_mass.size());
    for (std::size_t i = 0; i < primary_probe.size(); ++i) {
      primary_probe[i] = (i % 3 == 0) ? -0.75 : 1.25;
    }
    for (std::size_t i = 0; i < auxiliary_probe.size(); ++i) {
      auxiliary_probe[i] = (i % 5 < 2) ? 0.5 : -1.0;
    }
    const std::vector<double> forward_probe =
        apply_mass_compatible_primary_to_auxiliary(result, primary_probe);
    const std::vector<double> reverse_probe =
        apply_mass_compatible_auxiliary_to_primary(result, auxiliary_probe);
    double forward_inner = 0.0;
    double reverse_inner = 0.0;
    for (std::size_t i = 0; i < auxiliary_probe.size(); ++i) {
      forward_inner += auxiliary_mass[i] * forward_probe[i] * auxiliary_probe[i];
    }
    for (std::size_t i = 0; i < primary_probe.size(); ++i) {
      reverse_inner += primary_mass[i] * primary_probe[i] * reverse_probe[i];
    }
    result.adjoint_relative =
        std::abs(forward_inner - reverse_inner) /
        std::max({1.0e-300, std::abs(forward_inner), std::abs(reverse_inner)});

    const double algebra_tolerance =
        std::max(1.0e-12, 10.0 * config.balance_relative_tolerance);
    if (result.balance_relative > config.balance_relative_tolerance) {
      result.status = "mass_balance_gap";
    } else if (result.forward_constant_relative > algebra_tolerance ||
               result.reverse_constant_relative > algebra_tolerance ||
               result.adjoint_relative > algebra_tolerance) {
      result.status = "mass_transfer_algebra_gap";
    } else if (result.geometry_status != "pass") {
      result.status = "real_primary_geometry_gap";
    } else {
      result.status = "pass";
    }
  } catch (const std::exception &) {
    result.status = "mass_transfer_failure";
  }
  const auto end = std::chrono::steady_clock::now();
  result.build_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  return result;
}

} // namespace TitanPBE::Tabi::calderon
