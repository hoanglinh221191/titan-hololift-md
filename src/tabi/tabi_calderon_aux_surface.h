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
  // patch-ray-resolution: vertices next to an uncertified ray row classified
  // by a certified point 1e-3 h or 1e-2 h away.
  std::size_t resolution_nearby_parity_vertices{0};
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
  // patch-ray-resolution: projections that were not found, refused because
  // they would turn or flatten an incident triangle, refused because the
  // point was on another sheet than the vertex's star, or undone because a
  // triangle pair still intersected; distances in A.
  std::size_t resolution_missing_projection_vertices{0};
  std::size_t resolution_star_refused_vertices{0};
  std::size_t resolution_sheet_refused_vertices{0};
  std::size_t resolution_partial_projection_vertices{0};
  // Triangles whose smallest angle is below 5, 10 and 20 degrees; edges in A.
  std::size_t resolution_angle_below_5{0};
  std::size_t resolution_angle_below_10{0};
  std::size_t resolution_angle_below_20{0};
  double resolution_edge_min{0.0};
  double resolution_edge_mean{0.0};
  std::size_t resolution_short_edges_005{0};
  std::size_t resolution_short_edges_020{0};
  // Isotropic remeshing onto the analytic SES (resolution policy).
  std::size_t resolution_remesh_iterations{0};
  std::size_t resolution_remesh_reverted{0};
  std::size_t resolution_remesh_splits{0};
  std::size_t resolution_remesh_collapses{0};
  std::size_t resolution_remesh_flips{0};
  std::size_t resolution_remesh_off_surface{0};
  std::size_t resolution_remesh_sheet_refused{0};
  // Final vertices carrying the certified analytic normal of the point they
  // sit on, and on-surface vertices whose analytic normal was refused because
  // their star faces another sheet.
  std::size_t resolution_analytic_normal_vertices{0};
  std::size_t resolution_analytic_normal_refused{0};
  // Charge-adaptive remeshing: the charges that set a target edge, and the
  // final vertices whose local target is below the uniform edge.
  std::size_t resolution_refine_charges{0};
  std::size_t resolution_refined_vertices{0};
  double resolution_remesh_edge{0.0};
  double resolution_remesh_ms{0.0};
  std::size_t resolution_reverted_vertices{0};
  std::size_t resolution_revert_rounds{0};
  std::size_t resolution_initial_intersections{0};
  // Grid vertices whose sign was flipped because the solvent slab or the
  // molecular plate they sit in is one cell thick along an axis.
  std::size_t resolution_thin_solvent_filled{0};
  std::size_t resolution_thin_plates_opened{0};
  double resolution_unprojected_distance_max{0.0};
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

// Charge-adaptive remeshing (patch-ray-resolution).  A charge at depth d below
// the surface makes the boundary density vary over about d around its foot
// point, so a uniform edge much longer than d leaves the reaction energy
// unresolved however accurate the quadrature.  The target edge at a surface
// point x becomes clamp(alpha * min_k |x - y_k|, min_edge, edge) over the
// charges y_k with |q_k| >= min_charge: the mesh grades down towards shallow
// charges, and a charge deeper than edge/alpha changes nothing.  alpha <= 0
// keeps the uniform edge.
struct ChargeRefinement {
  double alpha{0.0};
  double min_edge{0.2};
  double min_charge{0.1};
};

[[nodiscard]] AuxiliarySurfaceResult build_analytic_ses_surface(
    std::span<const Atom> atoms, double grid_spacing, double probe_radius,
    TitanExecutor *executor = nullptr,
    std::size_t max_grid_vertices = 4000000,
    bool validate_topology = true,
    std::string_view engine = "current",
    // patch-ray-resolution only: target edge of the isotropic remeshing in A;
    // 0 uses the grid spacing and a negative value keeps the marching mesh.
    double remesh_edge = 0.0,
    // patch-ray-resolution with remeshing only; the charges are atoms' q.
    ChargeRefinement charge_refinement = {},
    // patch-ray-resolution with remeshing only: passes of split, collapse,
    // flip and relaxation.
    int remesh_iterations = 6);

} // namespace TitanPBE::Tabi::calderon

#endif // TITAN_PBE_TABI_CALDERON_AUX_SURFACE_H
