#ifndef TITAN_TABI_NODE_PATCH_CURVED_H
#define TITAN_TABI_NODE_PATCH_CURVED_H
#include "tabi_types.h"
#include <atomic>
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
  // kMomentSlot instead; that saves 4*160*8 = 5120 bytes per cell.
  std::array<std::array<double, kCurvedMomentCount>, 4> moments{};
  double area{}, radius{};
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
};
inline const CurvedDualCell *curved_dual_cell(const Panel &p) noexcept {
  if (!p.node_patch_geometry || !p.node_patch_geometry->curved)
    return nullptr;
  const auto &cells = p.node_patch_geometry->curved->cells;
  return p.node_patch_index < cells.size() ? &cells[p.node_patch_index]
                                           : nullptr;
}
// Returns false for a surface without exact spherical parent geometry.
bool prepare_curved_node_patches(const std::vector<Panel> &triangles,
                                 std::vector<Panel> &nodes, double eps_ratio,
                                 double kappa, std::size_t pair_cache_budget,
                                 TitanExecutor &executor, bool eager = true);
std::array<double, 4> curved_node_pair(const Panel &target,
                                       std::size_t target_index,
                                       const Panel &source,
                                       std::size_t source_index,
                                       double eps_ratio, double kappa);
std::size_t
curved_node_storage_bytes(const std::vector<Panel> &panels) noexcept;
} // namespace TitanPBE::Tabi
#endif
