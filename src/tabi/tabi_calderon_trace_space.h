#ifndef TITAN_PBE_TABI_CALDERON_TRACE_SPACE_H
#define TITAN_PBE_TABI_CALDERON_TRACE_SPACE_H

#include "tabi_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace TitanPBE::Tabi::calderon {

// P1/P0 trace-space data needed before a weak-form Calderon operator can be
// assembled. The builder rejects open or inconsistently oriented surfaces.
struct TraceSpaceQuality {
  bool valid{false};
  std::size_t vertex_count{0};
  std::size_t triangle_count{0};
  std::size_t edge_count{0};
  std::size_t boundary_edge_count{0};
  std::size_t orientation_mismatch_count{0};
  std::size_t nonmanifold_edge_count{0};
  std::size_t unreferenced_vertex_count{0};
  std::size_t connected_component_count{0};
  std::int64_t euler_characteristic{0};
  std::size_t genus_sum{0};
  bool topology_invariants_valid{false};
  double total_area{0.0};
  std::string failure_reason{"not_built"};
};

struct TraceSpaces {
  TraceSpaceQuality quality;
  // The P0 mass is diagonal on triangles. The lumped P1 mass and the
  // triangle-local P1/P0 pairing provide the Riesz-map foundation for C5.
  std::vector<double> p0_triangle_mass;
  std::vector<double> p1_vertex_lumped_mass;
  std::vector<std::array<double, 3>> p1_p0_pairing;
};

[[nodiscard]] TraceSpaces
build_closed_surface_trace_spaces(const SurfaceMesh &mesh);

} // namespace TitanPBE::Tabi::calderon

#endif // TITAN_PBE_TABI_CALDERON_TRACE_SPACE_H
