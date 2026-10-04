#ifndef TITAN_PBE_TABI_SES_PATCH_RAY_H
#define TITAN_PBE_TABI_SES_PATCH_RAY_H

#include "tabi_analytic_ses.h"
#include "tabi_ses_patch_intersections.h"
#include "tabi_ses_patch_provenance.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

class TitanExecutor;

namespace TitanPBE::Tabi::analytic_ses {

struct AxisRayCrossings {
  RayAxis axis{RayAxis::x};
  std::size_t u_count{0};
  std::size_t v_count{0};
  std::vector<std::size_t> offsets;
  std::vector<double> coordinates;
  std::size_t odd_rows{0};

  [[nodiscard]] std::size_t row_count() const noexcept {
    if (u_count != 0U &&
        v_count > std::numeric_limits<std::size_t>::max() / u_count)
      return std::numeric_limits<std::size_t>::max();
    return u_count * v_count;
  }

  [[nodiscard]] std::span<const double>
  row(std::size_t row_index) const noexcept {
    if (row_index + 1U >= offsets.size())
      return {};
    return {coordinates.data() + offsets[row_index],
            offsets[row_index + 1U] - offsets[row_index]};
  }
};

struct PatchRayGrid {
  std::array<AxisRayCrossings, 3> axes;
  std::size_t raw_events{0};
  std::size_t duplicate_events{0};
  // Counts patch-line intersection queries, not distinct analytic patches.
  std::size_t uncertified_intersection_queries{0};
  std::size_t uncertified_rows_repaired{0};
  std::size_t unresolved_uncertified_rows{0};
  std::size_t degenerate_rows_repaired{0};
  // Exact retained vector payload capacities owned by the returned grid.
  // Fixed object/string/allocator metadata is excluded.
  std::size_t retained_bytes{0};
  // Peak builder-owned live vector payload: probe index, event slabs, merge
  // heap/group scratch, CSR/provenance and repair scratch. Executor-owned
  // storage, allocator bookkeeping and fixed objects are excluded.
  std::size_t peak_temporary_bytes{0};
  double build_ms{0.0};
  PatchRayProvenance provenance;
  std::string status{"not_built"};
};

struct PatchRayBuildOptions {
  bool retain_provenance{false};
#ifdef TITAN_TABI_SES_TEST_HOOKS
  bool simulate_allocation_failure{false};
#endif
};

[[nodiscard]] PatchRayProvenance::LatticeIdentity
make_patch_ray_lattice_identity(const Vec3 &lower, double grid_spacing,
                                std::size_t nx, std::size_t ny,
                                std::size_t nz) noexcept;

// Build compact ray crossings by visiting patches, not rays.  The grid lower
// corner and counts are shared with the marching-cell grid.  No ray bucket or
// per-ray patch list is consulted by this function.
[[nodiscard]] PatchRayGrid build_patch_ray_grid(
    const PatchComplex &complex, const Vec3 &lower, double grid_spacing,
    std::size_t nx, std::size_t ny, std::size_t nz,
    TitanExecutor *executor = nullptr,
    PatchRayBuildOptions options = {});

#ifdef TITAN_TABI_SES_TEST_HOOKS
void patch_ray_increment_root_multiplicity_for_test(
    std::uint32_t &multiplicity) noexcept;
void patch_ray_finalize_provenance_for_test(
    PatchRayGrid &grid);
#endif

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_PATCH_RAY_H
