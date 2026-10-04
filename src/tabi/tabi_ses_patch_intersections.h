#ifndef TITAN_PBE_TABI_SES_PATCH_INTERSECTIONS_H
#define TITAN_PBE_TABI_SES_PATCH_INTERSECTIONS_H

#include "tabi_analytic_ses.h"
#include "tabi_ses_patch_fingerprint.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace TitanPBE::Tabi::analytic_ses {

enum class RayAxis : std::uint8_t { x = 0, y = 1, z = 2 };

struct QuarticDiagnostics;

enum class TorusRootFilterReason : std::uint8_t {
  accepted = 0,
  radial_axis_singular,
  theta_interval,
  tube_consistency,
  beta_minor_arc,
  reentrant_coverage,
  tangential,
};

struct TorusRootFilterDiagnostic {
  TorusRootFilterReason reason{TorusRootFilterReason::accepted};
  double quartic_root{0.0};
  double coordinate{0.0};
  Vec3 point{};
  double theta{0.0};
  double beta{0.0};
  double radial_length{0.0};
  double tube_length{0.0};
  double tube_residual{0.0};
  double directional_normal{0.0};
};

// Optional fixed telemetry for post-quartic torus filtering. It is not stored
// in PatchAxisRoots or any row/worklist payload.
struct TorusPatchAxisDiagnostics {
  std::array<TorusRootFilterDiagnostic, 4> roots{};
  std::size_t root_count{0U};
};

struct PatchBounds {
  Vec3 lower;
  Vec3 upper;
};

struct PatchAxisRoots {
  std::array<double, 4> coordinates{};
  // Fixed audit metadata parallel to coordinates. A set bit proves that the
  // retained primitive root is not a collapsed near-tangent pair and passed
  // the directional-normal transversality filter. It does not retain the
  // derivative sign, so orientation remains uncertified.
  std::array<std::uint8_t, 4> primitive_simplicity_certified{};
  std::size_t count{0};
  bool certified{true};
};

// Audit representation of a conservative closed envelope of the physical
// torus minor arc. For forward spans <= pi it matches the one-sided production
// tolerance. For forward spans > pi it closes the production complement at
// both endpoints, so it may overbound but cannot prune an accepted beta.
struct TorusBetaMinorArc {
  double start{0.0};
  double span{0.0};
};

[[nodiscard]] bool directed_minor_arc_from_endpoints_for_audit(
    double endpoint_a, double endpoint_b, TorusBetaMinorArc &arc) noexcept;

// Narrow regression primitive for the unchanged production complement
// predicate. It is deliberately separate from the conservative audit envelope.
[[nodiscard]] bool angle_on_torus_beta_production_minor_arc_for_audit(
    double angle, double endpoint_a, double endpoint_b) noexcept;

// Reuses the exact beta-endpoint algebra and returns the conservative closed
// envelope described above. This audit helper additionally checks the
// atom/torus carrier consistency needed by beta-domain bounds and fails closed
// when that snapshot cannot be trusted.
[[nodiscard]] bool
torus_beta_minor_arc_for_audit(const PatchComplex &complex,
                               std::size_t torus_index,
                               TorusBetaMinorArc &arc) noexcept;

[[nodiscard]] bool
angle_on_torus_beta_minor_arc_for_audit(double angle,
                                        const TorusBetaMinorArc &arc) noexcept;

struct ProbeCell {
  std::int32_t x{0};
  std::int32_t y{0};
  std::int32_t z{0};
  std::uint32_t begin{0};
  std::uint32_t end{0};
};

// Compact, sorted three-dimensional lookup for reentrant probe blockers.  It
// replaces the ray-bucket scan in the patch-major engine and has no hash table
// or per-cell allocation in the hot path.
struct ReentrantProbeIndex {
  Vec3 origin;
  double cell_size{1.0};
  std::vector<ProbeCell> cells;
  std::vector<std::uint32_t> probe_indices;
};

struct SpatialPatchEntry {
  PatchBounds bounds;
  LocalPatchRef patch;
  std::uint32_t canonical_id{0};
};

struct SpatialPatchNode {
  PatchBounds bounds;
  std::uint32_t begin{0};
  std::uint32_t end{0};
  std::uint32_t left{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t right{std::numeric_limits<std::uint32_t>::max()};

  [[nodiscard]] bool is_leaf() const noexcept {
    return left == std::numeric_limits<std::uint32_t>::max();
  }
};

// Unit-sphere half-space used by the exact bounded contact/reentrant
// projector: dot(normal, direction) >= minimum_dot.
struct SpherePatchConstraint {
  Vec3 normal;
  double minimum_dot{-1.0};
};

enum class ActivePairCacheFailureReason : std::uint8_t {
  none,
  malformed_constraint,
  near_parallel,
  nonfinite_algebra,
  active_residual,
  storage
};

struct ActivePairCacheFailure {
  ActivePairCacheFailureReason reason{ActivePairCacheFailureReason::none};
  PatchKind kind{PatchKind::contact};
  std::size_t patch{std::numeric_limits<std::size_t>::max()};
  std::size_t first{std::numeric_limits<std::size_t>::max()};
  std::size_t second{std::numeric_limits<std::size_t>::max()};
  double first_normal2{0.0};
  double second_normal2{0.0};
  double first_minimum_dot{0.0};
  double second_minimum_dot{0.0};
  double cross_squared{0.0};
  double base_squared{0.0};
  double first_active_residual{0.0};
  double second_active_residual{0.0};
  double minimum_feasibility_margin{0.0};
  bool candidate_feasible{false};
};

// The atom adjacency is only a projection hint.  A deterministic flat AABB
// BVH proves that every omitted patch is farther than the returned hit.
struct LocalPatchProjectionIndex {
  // Identity binding used by downstream analytic certificates.  The pointed
  // PatchComplex must outlive this prepared index.
  const PatchComplex *source_complex{nullptr};
  // Retained only for explicit provenance/mapping audits. The production
  // default avoids the extra O(atoms + edges + patches) fingerprint pass.
  PatchComplexFingerprint source_complex_fingerprint;
  ReentrantProbeIndex reentrant_probes;
  std::vector<std::size_t> atom_offsets;
  std::vector<LocalPatchRef> patches;
  std::vector<SpatialPatchEntry> spatial_entries;
  std::vector<SpatialPatchNode> spatial_nodes;
  std::vector<std::size_t> contact_constraint_offsets;
  std::vector<SpherePatchConstraint> contact_constraints;
  std::vector<std::size_t> reentrant_constraint_offsets;
  std::vector<SpherePatchConstraint> reentrant_constraints;
  // Feasible intersections of two active spherical half-space boundaries.
  // These are prepared once so each query scans O(V) certified corner
  // directions instead of repeating the O(m^3) active-pair feasibility work.
  std::vector<std::size_t> contact_two_active_offsets;
  std::vector<Vec3> contact_two_active_directions;
  std::vector<std::size_t> reentrant_two_active_offsets;
  std::vector<Vec3> reentrant_two_active_directions;
  std::vector<std::uint8_t> contact_domain_empty;
  std::vector<std::uint8_t> reentrant_domain_empty;
  bool bounded_domains_certified{true};
  bool bvh_invariants_certified{false};
  ActivePairCacheFailure active_pair_cache_failure;
  std::size_t torus_reentrant_pairs_audited{0};
  double torus_reentrant_audit_ms{0.0};
  double active_pair_cache_build_ms{0.0};

  [[nodiscard]] std::size_t retained_bytes() const noexcept {
    return atom_offsets.capacity() * sizeof(std::size_t) +
           patches.capacity() * sizeof(LocalPatchRef) +
           spatial_entries.capacity() * sizeof(SpatialPatchEntry) +
           spatial_nodes.capacity() * sizeof(SpatialPatchNode) +
           contact_constraint_offsets.capacity() * sizeof(std::size_t) +
           contact_constraints.capacity() * sizeof(SpherePatchConstraint) +
           reentrant_constraint_offsets.capacity() * sizeof(std::size_t) +
           reentrant_constraints.capacity() * sizeof(SpherePatchConstraint) +
           contact_two_active_offsets.capacity() * sizeof(std::size_t) +
           contact_two_active_directions.capacity() * sizeof(Vec3) +
           reentrant_two_active_offsets.capacity() * sizeof(std::size_t) +
           reentrant_two_active_directions.capacity() * sizeof(Vec3) +
           contact_domain_empty.capacity() * sizeof(std::uint8_t) +
           reentrant_domain_empty.capacity() * sizeof(std::uint8_t) +
           reentrant_probes.cells.capacity() * sizeof(ProbeCell) +
           reentrant_probes.probe_indices.capacity() * sizeof(std::uint32_t);
  }

  [[nodiscard]] std::size_t active_pair_cache_bytes() const noexcept {
    return contact_two_active_offsets.capacity() * sizeof(std::size_t) +
           contact_two_active_directions.capacity() * sizeof(Vec3) +
           reentrant_two_active_offsets.capacity() * sizeof(std::size_t) +
           reentrant_two_active_directions.capacity() * sizeof(Vec3) +
           contact_domain_empty.capacity() * sizeof(std::uint8_t) +
           reentrant_domain_empty.capacity() * sizeof(std::uint8_t);
  }

  // Query-local hint deduplication retains this much additional workspace per
  // worker. It is deliberately separate from immutable index storage above.
  [[nodiscard]] std::size_t stamp_bytes_per_worker() const noexcept {
    return spatial_entries.size() * sizeof(std::uint32_t);
  }
};

struct LocalPatchProjectionIndexBuildOptions {
  bool retain_complex_fingerprint{false};
};

struct LocalProjectionHit {
  Vec3 point;
  Vec3 normal;
  PatchKind kind{PatchKind::contact};
  std::size_t patch_index{0};
  // Deterministic atom attribution for the selected patch.  At seams this is
  // the smallest atom index associated with the patch; it is only a hint for
  // the next query.
  std::size_t atom_index{std::numeric_limits<std::size_t>::max()};
  // Euclidean point-to-projection distance.  This is populated by every
  // projection path.
  double distance{std::numeric_limits<double>::infinity()};
  // Signed-axis-free displacement used only by coordinate-line projection.
  double axis_distance{std::numeric_limits<double>::infinity()};
  // Number of unique exact bounded-patch projector calls. If stamp allocation
  // fails, owner hints are skipped and BVH entries remain unique.
  std::size_t candidate_patches{0};
  // Audit counters indexed by PatchKind (contact, toroidal, reentrant).
  std::array<std::size_t, 3> candidate_patches_by_kind{};
  std::array<std::size_t, 3> empty_domain_short_circuits_by_kind{};
  // Populated only when the audit wrapper explicitly enables timing, so the
  // production surface path does not pay for a clock read per patch.
  std::array<std::uint64_t, 3> projector_nanoseconds_by_kind{};
  // Raw BVH entries eliminated by bounds. This can include a patch already
  // evaluated through an owner hint, so it is not a disjoint partition with
  // candidate_patches.
  std::size_t bound_pruned_patches{0};
  std::size_t spatial_nodes_visited{0};
  std::size_t owner_hint_candidates{0};
  std::size_t stamp_bytes_per_worker{0};
  bool owner_hint_hit{false};
  bool spatial_fallback{false};
  bool axis_fallback{false};
  bool stamp_allocation_failed{false};
  bool traversal_invariant_failed{false};
  bool normal_certified{false};
  bool bounded_closest_certified{false};
  bool found{false};
};

enum class ProjectionHintPolicy : std::uint8_t {
  owner_and_neighbors,
  owner_only,
  bvh_only
};

// Euclidean closest-point projection onto the exposed analytic patch complex.
// The owner patch seeds an upper bound only; a deterministic AABB BVH then
// visits every patch whose bound can tie or improve the result. Interior,
// analytic seam, and corner candidates are included for all three patch kinds.
// bounded_closest_certified is true only when that traversal completes.
[[nodiscard]] LocalProjectionHit project_local_patch_closest(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    std::size_t atom_index, const Vec3 &point, double max_distance) noexcept;

// Test/telemetry audit primitive. It changes only which exact patch
// projections seed the BVH upper bound; geometry, the complete BVH traversal,
// and the legacy axis fallback remain identical. Production callers use
// project_local_patch_closest above.
[[nodiscard]] LocalProjectionHit
project_local_patch_closest_with_hint_policy_for_audit(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    std::size_t atom_index, const Vec3 &point, double max_distance,
    ProjectionHintPolicy hint_policy,
    bool collect_kind_timing = false) noexcept;

// Exact closest point on one bounded analytic patch. This is also the
// patch-level audit primitive used to validate constrained seam regressions;
// it does not perform BVH or owner-hint selection.
[[nodiscard]] LocalProjectionHit
project_patch_bounded_closest(const PatchComplex &complex,
                              const LocalPatchProjectionIndex &index,
                              PatchKind kind, std::size_t patch_index,
                              const Vec3 &point, double max_distance) noexcept;

// Slow exhaustive active-pair oracle for cache regressions. It is never used
// by the production closest-point or surface paths.
[[nodiscard]] LocalProjectionHit
project_patch_bounded_closest_exhaustive_for_audit(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    PatchKind kind, std::size_t patch_index, const Vec3 &point,
    double max_distance) noexcept;

[[nodiscard]] ReentrantProbeIndex
build_reentrant_probe_index(const PatchComplex &complex);

[[nodiscard]] LocalPatchProjectionIndex build_local_patch_projection_index(
    const PatchComplex &complex,
    LocalPatchProjectionIndexBuildOptions options = {});

[[nodiscard]] PatchBounds patch_bounds(const PatchComplex &complex,
                                       PatchKind kind,
                                       std::size_t patch_index) noexcept;

// The exact conservative padding used by the prepared spatial-patch BVH.
// Audit consumers use the same function so a stale or mutated index cannot
// silently retain a closest-point/topology certificate.
[[nodiscard]] PatchBounds
outward_padded_patch_bounds(const PatchComplex &complex, PatchKind kind,
                            std::size_t patch_index) noexcept;

// Intersect one analytic patch with one axis-aligned line.  u and v denote the
// two coordinates orthogonal to the ray: (y,z) for x, (x,z) for y, and (x,y)
// for z.  Only transversal, exposed SES crossings are returned.  Coordinates
// are expressed in the world coordinate of the selected axis.
[[nodiscard]] PatchAxisRoots
intersect_patch_axis(const PatchComplex &complex,
                     const ReentrantProbeIndex &probe_index, PatchKind kind,
                     std::size_t patch_index, RayAxis axis, double u, double v,
                     bool retain_tangential = false) noexcept;

// Audit-only overload that executes the same production intersection path and
// asks the real quartic solver to populate fixed-size diagnostics. Existing
// callers retain the null-diagnostics fast path above.
[[nodiscard]] PatchAxisRoots intersect_patch_axis_with_quartic_diagnostics(
    const PatchComplex &complex, const ReentrantProbeIndex &probe_index,
    PatchKind kind, std::size_t patch_index, RayAxis axis, double u, double v,
    QuarticDiagnostics *diagnostics, bool retain_tangential = false) noexcept;

[[nodiscard]] PatchAxisRoots intersect_patch_axis_with_quartic_diagnostics(
    const PatchComplex &complex, const ReentrantProbeIndex &probe_index,
    PatchKind kind, std::size_t patch_index, RayAxis axis, double u, double v,
    QuarticDiagnostics *quartic_diagnostics,
    TorusPatchAxisDiagnostics *filter_diagnostics,
    bool retain_tangential = false) noexcept;

// Project along x onto the nearest certified analytic patch carried by the
// owning atom or one of its overlap neighbours.  Ties use patch kind/index and
// coordinate, making the result independent of hash or thread order.
[[nodiscard]] LocalProjectionHit
project_local_patch_x(const PatchComplex &complex,
                      const LocalPatchProjectionIndex &index,
                      std::size_t atom_index, const Vec3 &point,
                      double max_axis_distance) noexcept;

[[nodiscard]] LocalProjectionHit
project_local_patch_axis(const PatchComplex &complex,
                         const LocalPatchProjectionIndex &index,
                         std::size_t atom_index, const Vec3 &point,
                         RayAxis axis, double max_axis_distance) noexcept;

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_PATCH_INTERSECTIONS_H
