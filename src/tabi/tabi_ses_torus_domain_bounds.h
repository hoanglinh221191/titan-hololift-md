#ifndef TITAN_PBE_TABI_SES_TORUS_DOMAIN_BOUNDS_H
#define TITAN_PBE_TABI_SES_TORUS_DOMAIN_BOUNDS_H

#include "tabi_ses_torus_interval_bounds.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

class TitanExecutor;

namespace TitanPBE::Tabi::analytic_ses {

enum class TorusDomainFallbackReason : std::uint8_t {
  none,
  invalid_beta_domain,
  beta_bound_failure,
  theta_worklist_fallback
};

struct TorusDomainIntervalBound {
  PatchRefId ref;
  std::uint32_t interval_index{std::numeric_limits<std::uint32_t>::max()};
  PatchBounds bounds;
};

struct TorusDomainRowWorklistBuildOptions {
  bool reverse_torus_schedule_for_audit{false};
  std::size_t theta_raw_candidate_pair_limit{
      std::numeric_limits<std::size_t>::max()};
  std::size_t candidate_pair_limit{std::numeric_limits<std::size_t>::max()};
};

// Audit-only D2b worklist.  Toroidal rows are filtered by the product of the
// exposed theta domain and the physical directed beta minor arc.  Contact and
// reentrant membership is copied unchanged from D2a/D1.  Completion is a
// deterministic construction claim for the padded-libm bounds, not a formal
// interval, intersection, topology, or surface certificate.
struct PreparedTorusDomainRowWorklist {
  PatchComplexFingerprint complex_fingerprint;
  PatchRayProvenance::LatticeIdentity lattice_identity;
  std::array<PatchRayAxisRowWorklist, 3> axes;
  std::vector<std::size_t> torus_domain_offsets;
  std::vector<TorusDomainIntervalBound> torus_domain_bounds;
  std::vector<TorusDomainFallbackReason> torus_fallback_reasons;
  std::size_t total_rows{0U};
  std::size_t theta_candidate_pairs{0U};
  std::size_t candidate_pairs{0U};
  std::size_t nonempty_rows{0U};
  std::size_t maximum_row_occupancy{0U};
  std::size_t domain_rectangles{0U};
  std::size_t beta_fallback_to_theta_tori{0U};
  std::size_t full_fallback_tori{0U};
  std::array<std::size_t, 4> fallback_reason_counts{};
  std::size_t requested_workers{1U};
  std::size_t bounds_workers_used{1U};
  // Exact owned vector-capacity payload; fixed/status metadata and allocator
  // bookkeeping are excluded.
  std::size_t retained_bytes{0U};
  // Accounted vector payload at phase boundaries, excluding allocator
  // metadata and reallocation overlap. It is not RSS.
  std::size_t accounted_peak_temporary_bytes{0U};
  double theta_worklist_ms{0.0};
  double beta_bounds_ms{0.0};
  double filter_ms{0.0};
  double build_ms{0.0};
  bool domain_worklist_complete{false};
  std::string status{"not_built"};
};

// Public audit primitive used by dense long-double diagnostics.  Success
// returns the intersection of the D2b theta/beta product bound, the D2a theta
// bound, and the D1 full torus bound for one exposed theta interval.
[[nodiscard]] bool torus_domain_interval_bound_for_audit(
    const PatchComplex &complex, std::size_t torus_index,
    std::size_t interval_index, double grid_spacing,
    PatchBounds &bounds) noexcept;

// Audit-only extrema primitive for the exact directed beta acceptance span.
[[nodiscard]] bool torus_beta_trigonometric_ranges_for_audit(
    const TorusBetaMinorArc &arc, double &cos_minimum, double &cos_maximum,
    double &sin_minimum, double &sin_maximum) noexcept;

[[nodiscard]] PreparedTorusDomainRowWorklist build_torus_domain_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    TitanExecutor *executor = nullptr,
    TorusDomainRowWorklistBuildOptions options = {});

// Structural and exact-snapshot binding validation only. It does not repeat
// dense extrema or patch-intersection audits.
[[nodiscard]] bool validate_torus_domain_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    const PreparedTorusDomainRowWorklist &prepared) noexcept;

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_TORUS_DOMAIN_BOUNDS_H
