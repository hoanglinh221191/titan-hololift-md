#ifndef TITAN_PBE_TABI_CARTESIAN_RETRY_FOUNDATION_H
#define TITAN_PBE_TABI_CARTESIAN_RETRY_FOUNDATION_H

#include "tabi_types.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <span>

namespace TitanPBE::Tabi::cartesian_retry {
namespace detail {

inline constexpr std::uint64_t kGeometryHashOffset =
    1469598103934665603ULL;
inline constexpr std::uint64_t kGeometryHashPrime = 1099511628211ULL;

inline void mix(std::uint64_t &hash, std::uint64_t value) noexcept {
  hash ^= value;
  hash *= kGeometryHashPrime;
}

inline bool mix_double(std::uint64_t &hash, double value) noexcept {
  if (!std::isfinite(value))
    return false;
  mix(hash, std::bit_cast<std::uint64_t>(value));
  return true;
}

inline bool mix_xyz(std::uint64_t &hash, double x, double y,
                    double z) noexcept {
  return mix_double(hash, x) && mix_double(hash, y) && mix_double(hash, z);
}

inline bool valid_offsets(std::span<const std::size_t> offsets,
                          std::size_t entries) noexcept {
  if (offsets.empty() || offsets.front() != 0 || offsets.back() != entries)
    return false;
  for (std::size_t index = 1; index < offsets.size(); ++index) {
    if (offsets[index] < offsets[index - 1] || offsets[index] > entries)
      return false;
  }
  return true;
}

inline bool mix_indices(std::uint64_t &hash,
                        std::span<const std::size_t> values) noexcept {
  mix(hash, static_cast<std::uint64_t>(values.size()));
  for (const std::size_t value : values)
    mix(hash, static_cast<std::uint64_t>(value));
  return true;
}

inline bool mix_node_patch_geometry(std::uint64_t &hash,
                                    const NodePatchGeometry &geometry) noexcept {
  const std::size_t vertex_count = geometry.vertex_xyz.size();
  const std::size_t triangle_count = geometry.triangles.size();
  if (vertex_count == 0 ||
      !valid_offsets(geometry.incident_offsets,
                     geometry.incident_triangles.size()) ||
      !valid_offsets(geometry.adjacent_offsets, geometry.adjacent_nodes.size()) ||
      geometry.incident_offsets.size() != vertex_count + 1 ||
      geometry.adjacent_offsets.size() != vertex_count + 1)
    return false;

  mix(hash, 0x4e4f444550415443ULL); // "NODEPATC"
  mix(hash, static_cast<std::uint64_t>(vertex_count));
  for (const auto &vertex : geometry.vertex_xyz) {
    if (!mix_xyz(hash, vertex[0], vertex[1], vertex[2]))
      return false;
  }

  mix(hash, static_cast<std::uint64_t>(triangle_count));
  for (const NodePatchTriangle &triangle : geometry.triangles) {
    for (const std::size_t vertex : triangle.vertices) {
      if (vertex >= vertex_count)
        return false;
      mix(hash, static_cast<std::uint64_t>(vertex));
    }
    if (!mix_xyz(hash, triangle.nx, triangle.ny, triangle.nz) ||
        !mix_double(hash, triangle.area) || !(triangle.area > 0.0))
      return false;
    mix(hash, static_cast<std::uint64_t>(triangle.atom_index));
  }

  for (const std::size_t triangle : geometry.incident_triangles) {
    if (triangle >= triangle_count)
      return false;
  }
  for (const std::size_t node : geometry.adjacent_nodes) {
    if (node >= vertex_count)
      return false;
  }
  if (!mix_indices(hash, geometry.incident_offsets) ||
      !mix_indices(hash, geometry.incident_triangles) ||
      !mix_indices(hash, geometry.adjacent_offsets) ||
      !mix_indices(hash, geometry.adjacent_nodes))
    return false;
  mix(hash, static_cast<std::uint64_t>(geometry.solid_angle_fallbacks));
  return true;
}

} // namespace detail

struct CartesianRetryGeometryFingerprint {
  std::uint64_t value{0};
  bool valid{false};
};

// This is intentionally a value hash, not an ownership hash.  Cartesian
// node-patch execution already requires a single shared topology object per
// panel set; the pointer equality check below validates that contract but no
// pointer or address contributes to the fingerprint.  Thus independently
// allocated, content-equal panel sets produce the same hash.
inline CartesianRetryGeometryFingerprint
fingerprint_cartesian_retry_geometry(std::span<const Panel> panels) noexcept {
  if (panels.empty())
    return {};

  const NodePatchGeometry *const node_patch_geometry =
      panels.front().node_patch_geometry.get();
  if (node_patch_geometry == nullptr)
    return {};
  std::uint64_t hash = detail::kGeometryHashOffset;
  detail::mix(hash, 0x4341525452455452ULL); // "CARTRETR"
  detail::mix(hash, static_cast<std::uint64_t>(panels.size()));
  detail::mix(hash, 1ULL); // node-patch topology is mandatory for retry.

  for (const Panel &panel : panels) {
    if (panel.node_patch_geometry.get() != node_patch_geometry)
      return {};
    const double source_x =
        panel.has_source_quadrature_center ? panel.source_x : panel.x;
    const double source_y =
        panel.has_source_quadrature_center ? panel.source_y : panel.y;
    const double source_z =
        panel.has_source_quadrature_center ? panel.source_z : panel.z;
    if (!detail::mix_xyz(hash, panel.x, panel.y, panel.z) ||
        !detail::mix_xyz(hash, source_x, source_y, source_z) ||
        !detail::mix_xyz(hash, panel.nx, panel.ny, panel.nz) ||
        !detail::mix_double(hash, panel.area) || !(panel.area > 0.0) ||
        !detail::mix_double(hash, panel.diagonal) ||
        !detail::mix_double(hash, panel.solid_angle))
      return {};
    detail::mix(hash, panel.has_source_quadrature_center ? 1ULL : 0ULL);
    detail::mix(hash, static_cast<std::uint64_t>(panel.atom_index));
    detail::mix(hash, panel.has_triangle_geometry ? 1ULL : 0ULL);
    if (panel.has_triangle_geometry) {
      if (!detail::mix_xyz(hash, panel.v0x, panel.v0y, panel.v0z) ||
          !detail::mix_xyz(hash, panel.v1x, panel.v1y, panel.v1z) ||
          !detail::mix_xyz(hash, panel.v2x, panel.v2y, panel.v2z))
        return {};
      detail::mix(hash, static_cast<std::uint64_t>(panel.v0_index));
      detail::mix(hash, static_cast<std::uint64_t>(panel.v1_index));
      detail::mix(hash, static_cast<std::uint64_t>(panel.v2_index));
    }
    detail::mix(hash, static_cast<std::uint64_t>(panel.node_patch_index));
    if (panel.node_patch_index >= node_patch_geometry->vertex_xyz.size())
      return {};
  }

  if (panels.size() != node_patch_geometry->vertex_xyz.size() ||
      !detail::mix_node_patch_geometry(hash, *node_patch_geometry))
    return {};
  return {hash == 0 ? 1 : hash, true};
}

} // namespace TitanPBE::Tabi::cartesian_retry

#endif // TITAN_PBE_TABI_CARTESIAN_RETRY_FOUNDATION_H
