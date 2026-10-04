#ifndef TITAN_PBE_TABI_SES_TORUS_INTERVAL_BOUNDS_H
#define TITAN_PBE_TABI_SES_TORUS_INTERVAL_BOUNDS_H

#include "tabi_ses_patch_row_worklist.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

class TitanExecutor;

namespace TitanPBE::Tabi::analytic_ses {

enum class TorusIntervalFallbackReason : std::uint8_t {
  none,
  empty_intervals,
  invalid_interval,
  invalid_frame,
  sub_bound_failure
};

struct TorusExposedIntervalBound {
  PatchRefId ref;
  std::uint32_t interval_index{std::numeric_limits<std::uint32_t>::max()};
  PatchBounds bounds;
  bool fallback_full_torus{false};
};

struct TorusIntervalRowWorklistBuildOptions {
  // Audit scheduling probe. Event insertion is reversed before the canonical
  // sort/dedup and must not affect the prepared CSR.
  bool reverse_rectangle_schedule_for_audit{false};
  std::size_t raw_candidate_pair_limit{std::numeric_limits<std::size_t>::max()};
};

// Audit-only worklist using exposed-theta sub-bounds for toroidal patches and
// the unchanged D1 full bounds for contact/reentrant patches. Completion means
// the deterministic worklist was built for this heuristic bounds contract; it
// is not a mathematical interval, intersection, parity, topology, or surface
// certificate and has no production call site.
struct PreparedTorusIntervalRowWorklist {
  // The two hash words bind the exact snapshot. fingerprint.valid may be false
  // only when a fallback-eligible nonfinite angular interval was retained;
  // all carrier geometry and generated bounds are validated independently.
  PatchComplexFingerprint complex_fingerprint;
  PatchRayProvenance::LatticeIdentity lattice_identity;
  std::array<PatchRayAxisRowWorklist, 3> axes;
  std::vector<std::size_t> torus_interval_offsets;
  std::vector<TorusExposedIntervalBound> torus_interval_bounds;
  std::vector<TorusIntervalFallbackReason> torus_fallback_reasons;
  std::size_t total_rows{0U};
  std::size_t raw_rectangle_candidate_pairs{0U};
  std::size_t candidate_pairs{0U};
  std::size_t nonempty_rows{0U};
  std::size_t maximum_row_occupancy{0U};
  std::size_t exposed_interval_rectangles{0U};
  std::size_t fallback_tori{0U};
  std::array<std::size_t, 5> fallback_reason_counts{};
  std::size_t requested_workers{1U};
  std::size_t bounds_workers_used{1U};
  // Exact owned vector capacity payload; fixed/status metadata and allocator
  // bookkeeping are excluded.
  std::size_t retained_bytes{0U};
  // Accounted vector payload at phase boundaries, excluding allocator
  // metadata and reallocation overlap. It is not RSS.
  std::size_t accounted_peak_temporary_bytes{0U};
  double bounds_ms{0.0};
  double event_ms{0.0};
  double sort_prefix_ms{0.0};
  double build_ms{0.0};
  bool interval_worklist_complete{false};
  std::string status{"not_built"};
};

// Public audit primitive for analytic-extrema fixtures. The result encloses
// the stored torus parameterization over one exposed theta interval using the
// full safe minor-circle extent. It uses libm plus explicit outward padding,
// so success is an audited finite bound rather than a formal interval proof.
[[nodiscard]] bool torus_exposed_interval_bound_for_audit(
    const PatchComplex &complex, std::size_t torus_index,
    std::size_t interval_index, double grid_spacing,
    PatchBounds &bounds) noexcept;

// Shared theta-domain component range used by the D2a and D2b audit bounds.
// It follows the symmetric exposed-interval tolerance used by production.
[[nodiscard]] bool torus_exposed_theta_component_range_for_audit(
    double basis_u_component, double basis_v_component,
    const AngularInterval &interval, double &minimum, double &maximum) noexcept;

[[nodiscard]] PreparedTorusIntervalRowWorklist
build_torus_interval_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    TitanExecutor *executor = nullptr,
    TorusIntervalRowWorklistBuildOptions options = {});

// Structural and exact-snapshot binding validation only. It does not repeat
// the extrema or intersection audits.
[[nodiscard]] bool validate_torus_interval_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    const PreparedTorusIntervalRowWorklist &prepared) noexcept;

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_TORUS_INTERVAL_BOUNDS_H
