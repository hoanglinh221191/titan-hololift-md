#ifndef TITAN_TABI_NODE_PATCH_CURVED_H
#define TITAN_TABI_NODE_PATCH_CURVED_H
#include "tabi_types.h"
#include <atomic>
#include <cstdint>
#include <limits>
namespace TitanPBE::Tabi {
struct CurvedDualSample {
  std::array<double, 3> xyz, normal;
  double weight;
};
// Monomials of total degree <= 5 in three variables.  The Taylor order is
// clamped to 5 (tabi_solver.cpp), so no reachable index exceeds this.
inline constexpr std::size_t kCurvedMomentCount = 56;
// Maps the 6x6x6 lattice index ax*36+ay*6+az to a packed slot; -1 marks a
// degree above 5, which the clamp makes unreachable.
inline constexpr std::array<signed char, 216> kMomentSlot = [] {
  std::array<signed char, 216> table{};
  table.fill(-1);
  signed char next = 0;
  for (int ax = 0; ax < 6; ++ax)
    for (int ay = 0; ay + ax < 6; ++ay)
      for (int az = 0; az + ay + ax < 6; ++az)
        table[ax * 36 + ay * 6 + az] = next++;
  return table;
}();
struct CurvedDualCell {
  std::vector<CurvedDualSample> samples;
  std::vector<std::size_t> near_targets;
  std::vector<std::array<double, 4>> near_blocks;
  std::array<double, 3> center{}, normal_integral{};
  // Cartesian geometric moments about center, channel 0=dS, 1..3=n*dS.
  // Only total degree <=5 is used, which is 56 monomials, so the 6x6x6
  // lattice the index ax*36+ay*6+az addresses is 74% empty.  Index through
  // kMomentSlot instead; that saves 4*160*8 = 5120 bytes per cell.  Only the
  // classic treecode reads them (the FMM route's multipoles sum one point
  // per cell), so they are built, four channels, only off the FMM route and
  // are empty on it.
  std::vector<std::array<double, kCurvedMomentCount>> moments;
  double area{}, radius{};
  // The collocation point, which is also the cell's reference point for
  // every distance test (curved near zone, one-point rule, tree bounds, the
  // radius above): the vertex, or for a corner cell - a vertex where the
  // patches do not share a tangent plane, so the normal derivative there has
  // no meaning - a smooth point inside one of its patches (offset), in the
  // incident triangle at host_pos, at barycentric (2/3, 1/6, 1/6) from the
  // vertex: on the diagonal between the cell's two quads there, a sixth of
  // the height from both edges through the vertex.
  std::array<double, 3> reference{};
  bool offset{false};
  std::size_t host_pos{0};
};
struct CurvedKernelCacheEntry {
  std::atomic<std::size_t> key{0};
  std::atomic<unsigned char> ready{0};
  std::array<double, 4> value{};
};
struct CurvedNodePatchGeometry {
  std::vector<CurvedDualCell> cells;
  std::vector<std::array<double, 4>> pair_blocks;
  std::unique_ptr<CurvedKernelCacheEntry[]> sparse_blocks;
  std::size_t sparse_capacity{};
  double eps_ratio{}, kappa{};
  std::size_t storage_bytes{};
  // "curved_dual": exact spherical parents (sphere-patch/vdW).
  // "curved_pn": PN-triangle patches from mesh vertices and node normals,
  // for any closed triangulation such as a watertight SES.
  std::string representation{"curved_dual"};
  // curved_pn: vertices whose analytic normal left a patch steep, folded or
  // not provably apart from another, and which took the star normal for the
  // patches and the collocation alike; of them, those turned further toward
  // their faces; and the repair rounds (pn_repair_normals).
  std::size_t pn_star_normal_vertices{0};
  std::size_t pn_turned_normal_vertices{0};
  std::size_t pn_repair_rounds{0};
  // curved_pn: triangles no vertex normal mends, flat (with their edges
  // straight in the neighbours) so the surface stays valid; the corner
  // vertices this leaves, whose cells collocate at a smooth point inside a
  // patch (CurvedDualCell::offset); and the largest offset from a vertex.
  std::size_t pn_residual_flat_triangles{0};
  std::size_t pn_straightened_edges{0};
  std::size_t pn_corner_vertices{0};
  double pn_max_collocation_offset{0.0};
  // curved_pn: pairs of flat patches without a shared vertex that touch, a
  // defect of the triangulation itself (the lift then refuses).
  std::size_t pn_flat_contacts{0};
  // curved_pn: the largest distance between the two patches of a shared edge,
  // measured after the lift (rounding only; above 1e-9 A the lift throws).
  double pn_max_shared_edge_gap{0.0};
  // Parent geometry, kept so a near pair can be integrated at any order: the
  // sphere of every triangle (curved_dual), or the node normals the PN
  // patches are built from (curved_pn).  pn_straight_edges is empty unless a
  // triangle had to be flattened: the lift repairs normals rather than
  // straightening edges; bit k of a mask straightens the edge from local
  // vertex k to k + 1.
  std::vector<std::array<double, 4>> parent_spheres;
  std::vector<std::array<double, 3>> pn_node_normals;
  std::vector<std::uint8_t> pn_straight_edges;
  // Which pairs are integrated over the curved cell.  Zero, the direct and
  // classic-treecode routes: every pair, with the stored samples or, near the
  // cell, to convergence.  Positive, the FMM route: there the multipole far
  // field sums one point per cell, so a pair is integrated over the curved
  // cell only for self, one-ring, and a target within this many cell radii of
  // the source vertex; every other pair takes the one-point centroid kernel
  // that the packed near loop and the multipole moments use.  Every consumer
  // (near loop, dense audit rows, ILU blocks) then applies one operator.
  double one_point_beyond_radii{0.0};
  // Near-pair evaluations by admissible subdivision on every route, those
  // that could not be resolved (and throw before a partial integral is used),
  // and the smallest distance of a
  // target from a cell among them.  Read into the solve result, and reported
  // when the geometry is released.
  mutable std::atomic<std::size_t> near_adaptive_evaluations{0};
  mutable std::atomic<std::size_t> near_unresolved_evaluations{0};
  mutable std::atomic<double> near_min_gap{
      std::numeric_limits<double>::infinity()};
  // Self blocks by the Gauss sequence 16, 32, 64, 128 (accepted when two
  // orders agree in every coefficient to 1e-6 of that coefficient, or to the
  // rounding of its own sum): those still moving at order 128 (nonzero
  // refuses the lift), the largest last change scaled to a 1e-6 tolerance,
  // and the highest order a block needed.
  std::size_t self_unconverged_blocks{0};
  double self_worst_change{0.0};
  int self_max_order{0};
  // First order of that sequence (TITAN_CURVED_RING_ORDER, default 8).
  int self_base_order{0};
  // Blocks the sequence left moving at 128 that a second rule accepted (the
  // barycentric polar Duffy rule of an off-vertex cell, orders 16..128 as
  // steps 1..4; graded rectangles of a vertex cell, 2^step per side), on two
  // of its own steps agreeing or one agreeing with the order-128 value, and
  // the highest step one needed.  self_unconverged_blocks counts what is
  // still moving after both rules.
  std::size_t self_fallback_blocks{0};
  int self_max_fallback_step{0};
  CurvedNodePatchGeometry() = default;
  CurvedNodePatchGeometry(const CurvedNodePatchGeometry &) = delete;
  CurvedNodePatchGeometry &operator=(const CurvedNodePatchGeometry &) = delete;
  ~CurvedNodePatchGeometry();
};
inline const CurvedDualCell *curved_dual_cell(const Panel &p) noexcept {
  if (!p.node_patch_geometry || !p.node_patch_geometry->curved)
    return nullptr;
  const auto &cells = p.node_patch_geometry->curved->cells;
  return p.node_patch_index < cells.size() ? &cells[p.node_patch_index]
                                           : nullptr;
}
// True when a non-self, non-adjacent pair falls outside the curved near zone
// of the FMM route and so takes the one-point kernel (see
// one_point_beyond_radii).  Always false off the FMM route.
inline bool curved_pair_takes_one_point(const Panel &target,
                                        const Panel &source) noexcept {
  const CurvedDualCell *cell = curved_dual_cell(source);
  if (cell == nullptr)
    return false;
  const double zone = source.node_patch_geometry->curved->one_point_beyond_radii;
  if (!(zone > 0.0))
    return false;
  const double dx = target.x - source.x, dy = target.y - source.y,
               dz = target.z - source.z;
  const double reach = zone * cell->radius;
  return dx * dx + dy * dy + dz * dz > reach * reach;
}
// Returns false for a surface without exact spherical parent geometry.
// one_point_beyond_radii > 0 selects the FMM-route pair rule and turns off
// the eager all-pairs cache, which would hold the curved value of every pair.
bool prepare_curved_node_patches(const std::vector<Panel> &triangles,
                                 std::vector<Panel> &nodes, double eps_ratio,
                                 double kappa, std::size_t pair_cache_budget,
                                 TitanExecutor &executor, bool eager = true,
                                 int sample_order = 4,
                                 double one_point_beyond_radii = 0.0);
std::array<double, 4> curved_node_pair(const Panel &target,
                                       std::size_t target_index,
                                       const Panel &source,
                                       std::size_t source_index,
                                       double eps_ratio, double kappa);
std::size_t
curved_node_storage_bytes(const std::vector<Panel> &panels) noexcept;
} // namespace TitanPBE::Tabi
#endif
