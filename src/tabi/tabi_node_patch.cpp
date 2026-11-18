#include "tabi_node_patch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace TitanPBE::Tabi {
namespace {

constexpr std::size_t kInvalid = static_cast<std::size_t>(-1);

struct NodeAccum {
  std::array<double, 3> xyz{};
  double area{0.0};
  double nx_area{0.0};
  double ny_area{0.0};
  double nz_area{0.0};
  double source_x_area{0.0};
  double source_y_area{0.0};
  double source_z_area{0.0};
  double owner_area{-1.0};
  std::size_t atom_index{0};
  bool initialized{false};
  // Certified analytic SES normal at the vertex, when the surface has one.
  std::array<double, 3> analytic_normal{};
  bool has_analytic_normal{false};
};

// The surface engine refuses a vertex whose star turns more than this far
// from its analytic normal (kResolutionSheetCosine); the same bound keeps a
// node from taking the normal of a facing sheet.
constexpr double kAnalyticNormalStarCosine = 0.2;

std::array<double, 3> panel_vertex(const Panel &panel, int local) {
  if (local == 0)
    return {panel.v0x, panel.v0y, panel.v0z};
  if (local == 1)
    return {panel.v1x, panel.v1y, panel.v1z};
  return {panel.v2x, panel.v2y, panel.v2z};
}

bool same_point(const std::array<double, 3> &a,
                const std::array<double, 3> &b) noexcept {
  const double scale = std::max(
      {1.0, std::abs(a[0]), std::abs(a[1]), std::abs(a[2]), std::abs(b[0]),
       std::abs(b[1]), std::abs(b[2])});
  return std::abs(a[0] - b[0]) <= 1.0e-10 * scale &&
         std::abs(a[1] - b[1]) <= 1.0e-10 * scale &&
         std::abs(a[2] - b[2]) <= 1.0e-10 * scale;
}

double triangle_solid_angle(const std::array<double, 3> &a,
                            const std::array<double, 3> &b,
                            const std::array<double, 3> &c, double px,
                            double py, double pz) noexcept {
  const double ax = a[0] - px;
  const double ay = a[1] - py;
  const double az = a[2] - pz;
  const double bx = b[0] - px;
  const double by = b[1] - py;
  const double bz = b[2] - pz;
  const double cx = c[0] - px;
  const double cy = c[1] - py;
  const double cz = c[2] - pz;
  const double la = std::sqrt(ax * ax + ay * ay + az * az);
  const double lb = std::sqrt(bx * bx + by * by + bz * bz);
  const double lc = std::sqrt(cx * cx + cy * cy + cz * cz);
  if (la <= 1.0e-14 || lb <= 1.0e-14 || lc <= 1.0e-14)
    return 0.0;
  const double cross_x = by * cz - bz * cy;
  const double cross_y = bz * cx - bx * cz;
  const double cross_z = bx * cy - by * cx;
  const double numerator = ax * cross_x + ay * cross_y + az * cross_z;
  const double ab = ax * bx + ay * by + az * bz;
  const double bc = bx * cx + by * cy + bz * cz;
  const double ca = cx * ax + cy * ay + cz * az;
  const double denominator = la * lb * lc + ab * lc + bc * la + ca * lb;
  return 2.0 * std::atan2(numerator, denominator);
}

} // namespace

std::vector<Panel>
build_node_patch_panels(const std::vector<Panel> &triangle_panels,
                        const std::string &diagonal_mode) {
  if (triangle_panels.empty())
    return {};

  std::size_t vertex_count = 0;
  for (const Panel &panel : triangle_panels) {
    if (!panel.has_triangle_geometry || panel.v0_index == kInvalid ||
        panel.v1_index == kInvalid || panel.v2_index == kInvalid ||
        !(panel.area > 0.0) || !std::isfinite(panel.area)) {
      throw std::runtime_error(
          "PBE BIE: node-patch discretization requires indexed, positive-area "
          "triangle geometry.");
    }
    if (panel.v0_index == panel.v1_index || panel.v1_index == panel.v2_index ||
        panel.v2_index == panel.v0_index) {
      throw std::runtime_error(
          "PBE BIE: node-patch discretization found a degenerate triangle.");
    }
    vertex_count =
        std::max(vertex_count,
                 1u + std::max({panel.v0_index, panel.v1_index, panel.v2_index}));
  }

  auto geometry = std::make_shared<NodePatchGeometry>();
  geometry->triangles.reserve(triangle_panels.size());
  std::vector<NodeAccum> nodes(vertex_count);
  std::vector<std::vector<std::size_t>> incident(vertex_count);
  std::vector<std::unordered_set<std::size_t>> adjacent(vertex_count);

  for (std::size_t ti = 0; ti < triangle_panels.size(); ++ti) {
    const Panel &panel = triangle_panels[ti];
    const std::array<std::size_t, 3> ids = {panel.v0_index, panel.v1_index,
                                            panel.v2_index};
    NodePatchTriangle triangle;
    triangle.vertices = ids;
    triangle.nx = panel.nx;
    triangle.ny = panel.ny;
    triangle.nz = panel.nz;
    triangle.area = panel.area;
    triangle.atom_index = panel.atom_index;
    geometry->triangles.push_back(triangle);

    for (int local = 0; local < 3; ++local) {
      const std::size_t id = ids[local];
      const auto xyz = panel_vertex(panel, local);
      NodeAccum &node = nodes[id];
      if (node.initialized && !same_point(node.xyz, xyz))
        throw std::runtime_error(
            "PBE BIE: inconsistent coordinates for a shared mesh vertex.");
      node.xyz = xyz;
      node.initialized = true;
      if (!node.has_analytic_normal && panel.analytic_vertex_normals &&
          id < panel.analytic_vertex_normals->size()) {
        const auto &n = (*panel.analytic_vertex_normals)[id];
        if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
          node.analytic_normal = n;
          node.has_analytic_normal = true;
        }
      }
      const double share = panel.area / 3.0;
      node.area += share;
      node.nx_area += share * panel.nx;
      node.ny_area += share * panel.ny;
      node.nz_area += share * panel.nz;
      // The barycentric dual contribution inside one parent triangle is the
      // quadrilateral vertex -> edge midpoint -> triangle centroid -> edge
      // midpoint.  It is exactly two equal-area microtriangles.  Accumulating
      // their centroids preserves the first spatial moment of the dual patch,
      // unlike lumping the full source area at the mesh vertex.
      const auto next = panel_vertex(panel, (local + 1) % 3);
      const auto previous = panel_vertex(panel, (local + 2) % 3);
      const std::array<double, 3> midpoint_next = {
          0.5 * (xyz[0] + next[0]), 0.5 * (xyz[1] + next[1]),
          0.5 * (xyz[2] + next[2])};
      const std::array<double, 3> midpoint_previous = {
          0.5 * (previous[0] + xyz[0]),
          0.5 * (previous[1] + xyz[1]),
          0.5 * (previous[2] + xyz[2])};
      const std::array<double, 3> centroid = {
          (xyz[0] + next[0] + previous[0]) / 3.0,
          (xyz[1] + next[1] + previous[1]) / 3.0,
          (xyz[2] + next[2] + previous[2]) / 3.0};
      const std::array<double, 3> micro_centroid_a = {
          (xyz[0] + midpoint_next[0] + centroid[0]) / 3.0,
          (xyz[1] + midpoint_next[1] + centroid[1]) / 3.0,
          (xyz[2] + midpoint_next[2] + centroid[2]) / 3.0};
      const std::array<double, 3> micro_centroid_b = {
          (xyz[0] + centroid[0] + midpoint_previous[0]) / 3.0,
          (xyz[1] + centroid[1] + midpoint_previous[1]) / 3.0,
          (xyz[2] + centroid[2] + midpoint_previous[2]) / 3.0};
      const double micro_area = panel.area / 6.0;
      node.source_x_area +=
          micro_area * (micro_centroid_a[0] + micro_centroid_b[0]);
      node.source_y_area +=
          micro_area * (micro_centroid_a[1] + micro_centroid_b[1]);
      node.source_z_area +=
          micro_area * (micro_centroid_a[2] + micro_centroid_b[2]);
      if (share > node.owner_area) {
        node.owner_area = share;
        node.atom_index = panel.atom_index;
      }
      incident[id].push_back(ti);
      adjacent[id].insert(ids[(local + 1) % 3]);
      adjacent[id].insert(ids[(local + 2) % 3]);
    }
  }

  geometry->incident_offsets.reserve(vertex_count + 1);
  geometry->adjacent_offsets.reserve(vertex_count + 1);
  geometry->vertex_xyz.reserve(vertex_count);
  geometry->incident_offsets.push_back(0);
  geometry->adjacent_offsets.push_back(0);
  for (std::size_t i = 0; i < vertex_count; ++i) {
    if (!nodes[i].initialized || incident[i].empty())
      throw std::runtime_error(
          "PBE BIE: node-patch mesh contains an unreferenced vertex index.");
    geometry->vertex_xyz.push_back(nodes[i].xyz);
    geometry->incident_triangles.insert(geometry->incident_triangles.end(),
                                        incident[i].begin(), incident[i].end());
    geometry->incident_offsets.push_back(geometry->incident_triangles.size());
    std::vector<std::size_t> neighbors(adjacent[i].begin(), adjacent[i].end());
    std::sort(neighbors.begin(), neighbors.end());
    geometry->adjacent_nodes.insert(geometry->adjacent_nodes.end(),
                                    neighbors.begin(), neighbors.end());
    geometry->adjacent_offsets.push_back(geometry->adjacent_nodes.size());
  }

  std::vector<Panel> patches(vertex_count);
  double patch_area_sum = 0.0;
  for (std::size_t i = 0; i < vertex_count; ++i) {
    const NodeAccum &node = nodes[i];
    const double norm = std::sqrt(node.nx_area * node.nx_area +
                                  node.ny_area * node.ny_area +
                                  node.nz_area * node.nz_area);
    if (!(node.area > 0.0) || !(norm > 1.0e-15) || !std::isfinite(norm))
      throw std::runtime_error(
          "PBE BIE: node-patch construction produced an invalid node.");
    Panel &patch = patches[i];
    patch.x = node.xyz[0];
    patch.y = node.xyz[1];
    patch.z = node.xyz[2];
    patch.source_x = node.source_x_area / node.area;
    patch.source_y = node.source_y_area / node.area;
    patch.source_z = node.source_z_area / node.area;
    patch.has_source_quadrature_center = true;
    patch.nx = node.nx_area / norm;
    patch.ny = node.ny_area / norm;
    patch.nz = node.nz_area / norm;
    // The flat patch takes a third of each incident triangle, so the
    // area-weighted star normal is its mean normal: the one-point source
    // normal, whatever the collocation normal becomes below.
    patch.source_nx = patch.nx;
    patch.source_ny = patch.ny;
    patch.source_nz = patch.nz;
    // The collocation point is the vertex, so its normal is the surface
    // normal there when the surface is analytic: that is the tangent plane
    // the curved cell is built on and the one the jump coefficient assumes.
    // The area-weighted star normal remains for vertices off the analytic
    // surface and for surfaces without one.
    if (node.has_analytic_normal) {
      const auto &n = node.analytic_normal;
      const double length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      const double cosine =
          length > 0.0
              ? (n[0] * patch.nx + n[1] * patch.ny + n[2] * patch.nz) / length
              : -1.0;
      if (cosine >= kAnalyticNormalStarCosine) {
        patch.nx = n[0] / length;
        patch.ny = n[1] / length;
        patch.nz = n[2] / length;
        ++geometry->analytic_normal_nodes;
      } else {
        ++geometry->analytic_normal_rejected;
        ++geometry->averaged_normal_nodes;
      }
    } else {
      ++geometry->averaged_normal_nodes;
    }
    patch.area = node.area;
    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    patch.diagonal = diagonal_mode == "off" ? 0.0 : 1.0;
    patch.solid_angle = diagonal_mode == "off" ? 0.0 : two_pi;
    if (diagonal_mode == "solid-angle") {
      const double offset = std::max(1.0e-7, std::sqrt(node.area) * 1.0e-5);
      const double px = patch.x + patch.nx * offset;
      const double py = patch.y + patch.ny * offset;
      const double pz = patch.z + patch.nz * offset;
      double omega = 0.0;
      for (const std::size_t triangle_index : incident[i]) {
        const NodePatchTriangle &triangle = geometry->triangles[triangle_index];
        omega += triangle_solid_angle(
            geometry->vertex_xyz[triangle.vertices[0]],
            geometry->vertex_xyz[triangle.vertices[1]],
            geometry->vertex_xyz[triangle.vertices[2]], px, py, pz);
      }
      const double solid_angle = std::abs(omega);
      const double diagonal = solid_angle / two_pi;
      if (std::isfinite(diagonal) && diagonal >= 0.10 && diagonal <= 2.50) {
        patch.solid_angle = solid_angle;
        const double clamped = std::clamp(diagonal, 0.25, 1.75);
        if (clamped != diagonal)
          ++geometry->solid_angle_clamps;
        patch.diagonal = clamped;
      } else
        ++geometry->solid_angle_fallbacks;
    }
    patch.atom_index = node.atom_index;
    patch.node_patch_geometry = geometry;
    patch.node_patch_index = i;
    patch_area_sum += patch.area;
  }
  double triangle_area_sum = 0.0;
  for (const Panel &panel : triangle_panels)
    triangle_area_sum += panel.area;
  const double area_scale = std::max(1.0, std::abs(triangle_area_sum));
  if (!std::isfinite(patch_area_sum) ||
      std::abs(patch_area_sum - triangle_area_sum) > 1.0e-12 * area_scale) {
    throw std::runtime_error(
        "PBE BIE: node-patch lumped areas do not preserve surface area.");
  }
  return patches;
}

} // namespace TitanPBE::Tabi
