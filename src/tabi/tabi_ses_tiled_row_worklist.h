#ifndef TITAN_PBE_TABI_SES_TILED_ROW_WORKLIST_H
#define TITAN_PBE_TABI_SES_TILED_ROW_WORKLIST_H

#include "tabi_ses_torus_domain_bounds.h"

#include <array>
#include <cstddef>
#include <limits>
#include <string>

class TitanExecutor;

namespace TitanPBE::Tabi::analytic_ses {

struct TiledPatchRayRowWorklistBuildOptions {
  std::size_t tile_u_rows{16U};
  std::size_t tile_v_rows{16U};
  bool reverse_rectangle_schedule_for_audit{false};
  bool reverse_tile_schedule_for_audit{false};
  std::size_t raw_row_pair_limit{std::numeric_limits<std::size_t>::max()};
  std::size_t rectangle_tile_pair_limit{
      std::numeric_limits<std::size_t>::max()};
  std::size_t final_candidate_pair_limit{
      std::numeric_limits<std::size_t>::max()};
  std::size_t per_tile_scratch_pair_limit{
      std::numeric_limits<std::size_t>::max()};
};

// Audit-only exact row CSR built directly from the final D1/D2a/D2b bound
// plan. It does not retain a global raw (row,patch) event buffer and has no
// production call site. Completion is a deterministic construction claim, not
// a mathematical bound, intersection, parity, topology, or surface certificate.
struct PreparedTiledPatchRayRowWorklist {
  PatchComplexFingerprint complex_fingerprint;
  PatchRayProvenance::LatticeIdentity lattice_identity;
  std::array<PatchRayAxisRowWorklist, 3> axes;
  std::size_t total_rows{0U};
  std::size_t tile_u_rows{0U};
  std::size_t tile_v_rows{0U};
  std::size_t bound_rectangles{0U};
  std::size_t total_tiles{0U};
  std::size_t active_tiles{0U};
  std::size_t rectangle_tile_pairs{0U};
  std::size_t raw_row_pairs{0U};
  std::size_t deduplicated_candidate_pairs{0U};
  std::size_t maximum_raw_pairs_per_tile{0U};
  std::size_t torus_domain_plans{0U};
  std::size_t torus_theta_fallback_plans{0U};
  std::size_t torus_full_fallback_plans{0U};
  std::size_t requested_workers{1U};
  std::size_t scheduled_workers{1U};
  std::size_t count_workers_used{0U};
  std::size_t fill_workers_used{0U};
  // Sum of retained scratch vector capacities across every worker after the
  // count/fill phases; it is schedule-observed rather than deterministic.
  std::size_t observed_worker_scratch_bytes{0U};
  // Exact final CSR vector-capacity payload; fixed/status metadata and
  // allocator bookkeeping are excluded.
  std::size_t retained_bytes{0U};
  // Deterministic vector-payload upper bound for the largest builder phase.
  // It excludes allocator metadata/reallocation overlap and is not RSS.
  std::size_t accounted_peak_temporary_upper_bound_bytes{0U};
  double bounds_plan_ms{0.0};
  double tile_plan_ms{0.0};
  double count_ms{0.0};
  double prefix_ms{0.0};
  double fill_ms{0.0};
  double build_ms{0.0};
  bool tiled_worklist_complete{false};
  std::string status{"not_built"};
};

[[nodiscard]] PreparedTiledPatchRayRowWorklist
build_tiled_patch_ray_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    TitanExecutor *executor = nullptr,
    TiledPatchRayRowWorklistBuildOptions options = {});

// Structural and exact-snapshot binding validation only. It does not rebuild
// the bounds/tile plan or rerun patch intersections.
[[nodiscard]] bool validate_tiled_patch_ray_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    const PreparedTiledPatchRayRowWorklist &prepared) noexcept;

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_TILED_ROW_WORKLIST_H
