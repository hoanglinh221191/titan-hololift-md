#ifndef TITAN_PBE_TABI_SES_PATCH_ROW_WORKLIST_H
#define TITAN_PBE_TABI_SES_PATCH_ROW_WORKLIST_H

#include "tabi_ses_patch_mapping.h"

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

class TitanExecutor;

namespace TitanPBE::Tabi::analytic_ses {

// Audit-only exact CSR of analytic patches whose outward-padded AABB projects
// onto each Cartesian ray row. Membership records the completed deterministic
// broad-phase worklist for the chosen bounds contract; it is not a universal
// coverage proof, intersection, parity, topology, or surface certificate and
// is not connected to the production ray builder.
struct PatchRayAxisRowWorklist {
  RayAxis axis{RayAxis::x};
  std::size_t u_count{0U};
  std::size_t v_count{0U};
  std::vector<std::size_t> offsets;
  std::vector<PatchRefId> candidates;

  [[nodiscard]] std::size_t row_count() const noexcept;
  [[nodiscard]] std::span<const PatchRefId>
  row(std::size_t row_index) const noexcept;
};

struct PatchRayRowWorklistBuildOptions {
  // D1 audit scheduling probe: the count pass visits canonical rectangles in
  // reverse while fill remains canonical. It must not change the CSR.
  bool reverse_count_schedule_for_audit{false};
  // Conservative resource guard and deterministic overflow-test hook.
  std::size_t candidate_pair_limit{std::numeric_limits<std::size_t>::max()};
};

struct PreparedPatchRayRowWorklist {
  PatchComplexFingerprint complex_fingerprint;
  PatchRayProvenance::LatticeIdentity lattice_identity;
  std::array<PatchRayAxisRowWorklist, 3> axes;
  std::size_t total_rows{0U};
  std::size_t candidate_pairs{0U};
  std::size_t nonempty_rows{0U};
  std::size_t maximum_row_occupancy{0U};
  std::size_t requested_workers{1U};
  // Workers that actually populated disjoint patch-bound records. Count,
  // prefix, and canonical fill remain serial to avoid workers*rows storage.
  std::size_t bounds_workers_used{1U};
  // Exact owned vector payload capacity. Fixed/status metadata and allocator
  // bookkeeping are excluded.
  std::size_t retained_bytes{0U};
  // Accounted vector payload observed at phase boundaries. It includes the
  // temporary patch rectangles and count/cursor arrays, but excludes allocator
  // metadata and old/new reallocation overlap; it is not RSS.
  std::size_t accounted_peak_temporary_bytes{0U};
  double bounds_ms{0.0};
  double count_ms{0.0};
  double prefix_ms{0.0};
  double fill_ms{0.0};
  double build_ms{0.0};
  // True only when the deterministic projected-AABB worklist build completed.
  // Fixture audits compare it against the legacy broad phase and exact patch
  // intersections; this flag itself does not prove universal coverage.
  bool projected_aabb_worklist_complete{false};
  std::string status{"not_built"};
};

// Exact full padded bound used by the D1 row worklist before its inclusive
// binary-search row mapping. Audit-only downstream builders reuse this helper
// so the fallback broad phase cannot drift from D1.
[[nodiscard]] bool patch_ray_full_padded_bound_for_audit(
    const PatchComplex &complex, PatchKind kind, std::size_t patch_index,
    double grid_spacing, PatchBounds &bounds) noexcept;

[[nodiscard]] PreparedPatchRayRowWorklist
build_patch_ray_row_worklist(const PatchComplex &complex,
                             const PatchMappingGridSpec &spec,
                             TitanExecutor *executor = nullptr,
                             PatchRayRowWorklistBuildOptions options = {});

// Structural/binding validation for immutable audit payloads. It deliberately
// does not re-run patch bounds or intersections.
[[nodiscard]] bool validate_patch_ray_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    const PreparedPatchRayRowWorklist &prepared) noexcept;

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_PATCH_ROW_WORKLIST_H
