#include "tabi_calderon_trace_space.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace TitanPBE::Tabi::calderon {
namespace {

struct EdgeRecord {
  std::size_t count{0};
  int orientation_sum{0};
};

struct EdgeKey {
  std::size_t first{0};
  std::size_t second{0};

  bool operator==(const EdgeKey &) const noexcept = default;
};

struct EdgeKeyHash {
  std::size_t operator()(const EdgeKey &key) const noexcept {
    std::uint64_t value = static_cast<std::uint64_t>(key.first);
    value ^= static_cast<std::uint64_t>(key.second) +
             0x9e3779b97f4a7c15ULL + (value << 6U) + (value >> 2U);
    return static_cast<std::size_t>(value);
  }
};

using EdgeTable = std::unordered_map<EdgeKey, EdgeRecord, EdgeKeyHash>;

void add_directed_edge(EdgeTable &edges, std::size_t from, std::size_t to) {
  const EdgeKey key{std::min(from, to), std::max(from, to)};
  EdgeRecord &record = edges[key];
  ++record.count;
  record.orientation_sum += from < to ? 1 : -1;
}

} // namespace

TraceSpaces build_closed_surface_trace_spaces(const SurfaceMesh &mesh) {
  TraceSpaces spaces;
  TraceSpaceQuality &quality = spaces.quality;
  quality.vertex_count = mesh.vertices.size();
  quality.triangle_count = mesh.triangles.size();
  if (mesh.vertices.empty() || mesh.triangles.empty()) {
    quality.failure_reason = "empty_mesh";
    return spaces;
  }

  spaces.p0_triangle_mass.resize(mesh.triangles.size(), 0.0);
  spaces.p1_p0_pairing.resize(mesh.triangles.size());
  spaces.p1_vertex_lumped_mass.assign(mesh.vertices.size(), 0.0);
  std::vector<bool> referenced(mesh.vertices.size(), false);
  std::vector<std::size_t> parent(mesh.vertices.size());
  std::iota(parent.begin(), parent.end(), std::size_t{0});
  const auto find_root = [&](std::size_t vertex) {
    while (parent[vertex] != vertex) {
      parent[vertex] = parent[parent[vertex]];
      vertex = parent[vertex];
    }
    return vertex;
  };
  const auto unite = [&](std::size_t first, std::size_t second) {
    first = find_root(first);
    second = find_root(second);
    if (first == second)
      return;
    if (second < first)
      std::swap(first, second);
    parent[second] = first;
  };
  EdgeTable edges;
  edges.reserve(mesh.triangles.size() * 2u);

  for (std::size_t triangle_index = 0; triangle_index < mesh.triangles.size();
       ++triangle_index) {
    const Triangle &triangle = mesh.triangles[triangle_index];
    const std::array<std::size_t, 3> vertices{triangle.v0, triangle.v1,
                                               triangle.v2};
    if (triangle.v0 >= mesh.vertices.size() || triangle.v1 >= mesh.vertices.size() ||
        triangle.v2 >= mesh.vertices.size() || triangle.v0 == triangle.v1 ||
        triangle.v1 == triangle.v2 || triangle.v2 == triangle.v0) {
      quality.failure_reason = "invalid_triangle_connectivity";
      return spaces;
    }
    if (!(triangle.area > 0.0) || !std::isfinite(triangle.area)) {
      quality.failure_reason = "invalid_triangle_area";
      return spaces;
    }

    const double local_pairing = triangle.area / 3.0;
    spaces.p0_triangle_mass[triangle_index] = triangle.area;
    spaces.p1_p0_pairing[triangle_index] = {local_pairing, local_pairing,
                                             local_pairing};
    quality.total_area += triangle.area;
    for (const std::size_t vertex : vertices) {
      referenced[vertex] = true;
      spaces.p1_vertex_lumped_mass[vertex] += local_pairing;
    }
    add_directed_edge(edges, triangle.v0, triangle.v1);
    add_directed_edge(edges, triangle.v1, triangle.v2);
    add_directed_edge(edges, triangle.v2, triangle.v0);
    unite(triangle.v0, triangle.v1);
    unite(triangle.v1, triangle.v2);
  }

  for (const bool is_referenced : referenced) {
    if (!is_referenced) {
      ++quality.unreferenced_vertex_count;
    }
  }
  quality.edge_count = edges.size();
  for (std::size_t vertex = 0; vertex < referenced.size(); ++vertex) {
    if (referenced[vertex] && find_root(vertex) == vertex)
      ++quality.connected_component_count;
  }
  quality.euler_characteristic =
      static_cast<std::int64_t>(quality.vertex_count -
                                quality.unreferenced_vertex_count) -
      static_cast<std::int64_t>(quality.edge_count) +
      static_cast<std::int64_t>(quality.triangle_count);
  for (const auto &[edge, record] : edges) {
    (void)edge;
    if (record.count == 1) {
      ++quality.boundary_edge_count;
    } else if (record.count != 2) {
      ++quality.nonmanifold_edge_count;
    } else if (record.orientation_sum != 0) {
      ++quality.orientation_mismatch_count;
    }
  }

  if (quality.unreferenced_vertex_count != 0) {
    quality.failure_reason = "unreferenced_vertices";
  } else if (quality.boundary_edge_count != 0) {
    quality.failure_reason = "open_surface";
  } else if (quality.nonmanifold_edge_count != 0) {
    quality.failure_reason = "nonmanifold_surface";
  } else if (quality.orientation_mismatch_count != 0) {
    quality.failure_reason = "inconsistent_orientation";
  } else if (!(quality.total_area > 0.0) || !std::isfinite(quality.total_area)) {
    quality.failure_reason = "invalid_total_area";
  } else {
    const std::int64_t genus_numerator =
        2 * static_cast<std::int64_t>(quality.connected_component_count) -
        quality.euler_characteristic;
    if (genus_numerator < 0 || (genus_numerator & 1) != 0) {
      quality.failure_reason = "invalid_topology_invariants";
    } else {
      quality.genus_sum = static_cast<std::size_t>(genus_numerator / 2);
      quality.topology_invariants_valid = true;
      quality.valid = true;
      quality.failure_reason = "pass";
    }
  }
  return spaces;
}

} // namespace TitanPBE::Tabi::calderon
