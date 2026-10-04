#ifndef TITAN_PBE_TABI_CALDERON_AUX_SURFACE_H
#define TITAN_PBE_TABI_CALDERON_AUX_SURFACE_H

#include "tabi_calderon_trace_space.h"
#include "tabi_types.h"

#include <limits>
#include <span>
#include <string>
#include <string_view>

class TitanExecutor;

namespace TitanPBE::Tabi::calderon {

// A closed implicit surface mesh and its topology audit. The same deterministic
// extractor serves the auxiliary Calderon trace and the opt-in primary SES.
struct AuxiliarySurfaceResult {
  SurfaceMesh mesh;
  TraceSpaces trace_spaces;
  std::string status{"not_built"};
  std::string analytic_ses_engine{"current"};
  std::string cavity_policy{"not_applicable"};
  std::string topology_source{"not_built"};
  double build_ms{0.0};
  double patch_build_ms{0.0};
  double ray_build_ms{0.0};
  double topology_ms{0.0};
  double boundary_magnitude_ms{0.0};
  double polygonize_ms{0.0};
  double projection_ms{0.0};
  double local_repair_ms{0.0};
  double orientation_certificate_ms{0.0};
  double self_intersection_audit_ms{0.0};
  double sampled_analytic_audit_ms{0.0};
  double initial_collapse_ms{0.0};
  double stable_sliver_ms{0.0};
  double post_projection_collapse_ms{0.0};
  double local_sliver_ms{0.0};
  double full_projection_ms{0.0};
  double seam_weld_ms{0.0};
  double refused_sliver_repair_ms{0.0};
  double certificate_cleanup_ms{0.0};
  double orientation_repair_ms{0.0};
  double wide_repair_ms{0.0};
  double repair_ms{0.0};
  double current_pre_smooth_repair_ms{0.0};
  double current_component_filter_ms{0.0};
  double current_smooth_analytic_slivers_ms{0.0};
  double current_post_smooth_repair_ms{0.0};
  double current_final_orient_trace_space_ms{0.0};
  std::size_t grid_vertices{0};
  std::size_t odd_ray_rows{0};
  std::size_t patch_ray_events{0};
  std::size_t patch_ray_duplicates{0};
  std::size_t patch_ray_uncertified_intersection_queries{0};
  std::size_t patch_ray_uncertified_rows_repaired{0};
  std::size_t patch_ray_unresolved_uncertified_rows{0};
  std::size_t patch_ray_degenerate_rows_repaired{0};
  std::size_t patch_ray_retained_bytes{0};
  std::size_t patch_ray_peak_temporary_bytes{0};
  std::size_t topology_mixed_blocks{0};
  std::size_t topology_retained_bytes{0};
  std::size_t filled_cavity_regions{0};
  std::size_t filled_cavity_vertices{0};
  std::size_t active_cells{0};
  std::size_t active_cell_bytes{0};
  std::size_t geometric_certificate_vertices{0};
  std::size_t parity_confident_vertices{0};
  std::size_t parity_local_repair_vertices{0};
  std::size_t morphology_fallback_vertices{0};
  std::size_t parity_fallback_vertices{0};
  std::size_t projection_index_bytes{0};
  std::size_t repair_scratch_bytes{0};
  std::size_t projected_vertices{0};
  std::size_t projection_queries{0};
  std::size_t projection_hits{0};
  std::size_t rejected_projection_vertices{0};
  std::size_t rejected_projection_rounds{0};
  std::size_t projection_orientation_reversals{0};
  std::size_t coupled_projection_components_attempted{0};
  std::size_t coupled_projection_components_accepted{0};
  std::size_t coupled_projection_vertices_rescued{0};
  std::size_t coupled_projection_max_component_size{0};
  std::size_t coupled_projection_unresolved_vertices{0};
  std::size_t orientation_repair_rounds{0};
  std::size_t orientation_repair_faces{0};
  std::size_t wide_repair_attempts{0};
  std::size_t topology_certificate_accepts{0};
  std::size_t topology_fallback_audits{0};
  std::size_t welded_vertices{0};
  std::size_t collapsed_edges{0};
  std::size_t flipped_edges{0};
  std::size_t poor_triangles_before{0};
  std::size_t poor_triangles_after{0};
  double triangle_quality_min_before{0.0};
  double triangle_quality_min_after{0.0};
  double max_edge_ratio_before{0.0};
  double max_edge_ratio_after{0.0};
  std::string analytic_audit_status{"not_run"};
  std::size_t analytic_audit_samples{0};
  std::size_t analytic_audit_hits{0};
  std::size_t analytic_audit_misses{0};
  double analytic_projection_distance_rms{0.0};
  double analytic_projection_distance_max{0.0};
  double analytic_patch_residual_rms{0.0};
  double analytic_patch_residual_max{0.0};
  double analytic_normal_angle_rms_deg{0.0};
  double analytic_normal_angle_max_deg{0.0};
  double sampled_hausdorff_proxy{0.0};
  std::size_t analytic_vertex_projection_misses{0};
  double analytic_vertex_residual_max{0.0};
  // Vertices whose certified projection target could not be committed because
  // an incident face would reverse.  The certificate tolerates these while
  // the largest outstanding move stays a small fraction of the spacing, so
  // both numbers are reported rather than collapsed into a pass/fail bit.
  std::size_t analytic_vertex_refused{0};
  double analytic_vertex_refused_distance_max{0.0};
  // The subset refused to keep a one ring from being sheared, rather than
  // because a face would have reversed.  These are the commit's own choice
  // and the only refusals that trade surface position for triangle shape.
  std::size_t analytic_vertex_refused_for_shape{0};
  std::size_t refused_sliver_collapses{0};
  // Inverted faces whose own vertex normals are mutually opposed.  These
  // join two facing sheets of the surface across a gap narrower than the
  // grid can separate, so they are a spacing limit, not a mesh defect.
  std::size_t analytic_sheet_bridge_faces{0};
  bool analytic_vertex_certificate{false};
  std::string analytic_orientation_status{"not_run"};
  std::size_t analytic_orientation_certified_faces{0};
  std::size_t analytic_orientation_centroid_queries{0};
  std::size_t analytic_orientation_uncertified_faces{0};
  std::size_t analytic_inverted_faces{0};
  std::size_t analytic_self_intersections{0};
  std::size_t analytic_self_intersection_first{
      std::numeric_limits<std::size_t>::max()};
  std::size_t analytic_self_intersection_second{
      std::numeric_limits<std::size_t>::max()};
  double analytic_orientation_min_cosine{1.0};
  std::string analytic_normal_quality_status{"not_run"};
  std::size_t analytic_orientation_smooth_faces{0};
  std::size_t analytic_orientation_seam_faces{0};
  std::size_t analytic_normal_cosine_below_025_faces{0};
  std::size_t analytic_normal_cosine_below_050_faces{0};
  double analytic_normal_area_weighted_rms_deg{0.0};
  double analytic_normal_p99_deg{0.0};
  double analytic_smooth_normal_p99_deg{0.0};
  double analytic_seam_normal_p99_deg{0.0};
  double analytic_normal_area_below_cosine_025_fraction{0.0};
  double analytic_normal_area_below_cosine_050_fraction{0.0};
};

// Exact-count broad-phase audit used by the final analytic SES certificate
// and exposed for focused geometry regression tests. Adjacent triangle pairs
// are accepted only when their intersection is confined to their shared
// vertex or edge; overlaps or crossings beyond that simplex are reported.
struct SurfaceSelfIntersectionAudit {
  std::size_t count{0};
  std::size_t first{std::numeric_limits<std::size_t>::max()};
  std::size_t second{std::numeric_limits<std::size_t>::max()};
};

[[nodiscard]] SurfaceSelfIntersectionAudit
audit_surface_self_intersections(const SurfaceMesh &mesh,
                                 double grid_spacing);

#ifdef TITAN_TABI_SES_TEST_HOOKS
namespace testing {

// Exercises the same endpoint-collapse transaction used by the cold analytic
// SES repair path.  Kept behind a test macro so production API/ABI is unchanged.
[[nodiscard]] bool collapse_single_manifold_edge_for_test(
    const SurfaceMesh &source, std::size_t first, std::size_t second,
    bool keep_higher_endpoint);

// Exercises the deterministic projection transaction with caller-supplied
// certified proposals.  This keeps coupled-move regressions independent of a
// particular analytic patch fixture.
[[nodiscard]] bool commit_projection_proposals_for_test(
    SurfaceMesh &mesh, std::span<const Vertex> proposals);

} // namespace testing
#endif

[[nodiscard]] AuxiliarySurfaceResult build_implicit_vdw_auxiliary_surface(
    std::span<const Atom> atoms, double grid_spacing,
    TitanExecutor *executor = nullptr);

[[nodiscard]] AuxiliarySurfaceResult build_implicit_ses_surface(
    std::span<const Atom> atoms, double grid_spacing, double probe_radius,
    TitanExecutor *executor = nullptr,
    std::size_t max_grid_vertices = 4000000,
    bool validate_topology = true);

[[nodiscard]] AuxiliarySurfaceResult build_analytic_ses_surface(
    std::span<const Atom> atoms, double grid_spacing, double probe_radius,
    TitanExecutor *executor = nullptr,
    std::size_t max_grid_vertices = 4000000,
    bool validate_topology = true,
    std::string_view engine = "current");

} // namespace TitanPBE::Tabi::calderon

#endif // TITAN_PBE_TABI_CALDERON_AUX_SURFACE_H
