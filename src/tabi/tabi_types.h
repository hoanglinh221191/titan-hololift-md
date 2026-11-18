#ifndef TITAN_PBE_TABI_TYPES_H
#define TITAN_PBE_TABI_TYPES_H

#include "../aligned_allocator.h"
#include "types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class TitanExecutor;

namespace titan::tabi::fmm_m2l {
class FmmM2lAnalyticContext;
}

namespace TitanPBE::Tabi {

struct Atom {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double q{0.0};
  double radius{0.0};
};

struct NodePatchTriangle {
  std::array<std::size_t, 3> vertices{};
  double nx{0.0};
  double ny{0.0};
  double nz{0.0};
  double area{0.0};
  std::size_t atom_index{0};
};

struct CurvedNodePatchGeometry;
struct NodePatchGeometry {
  std::vector<std::array<double, 3>> vertex_xyz;
  std::vector<NodePatchTriangle> triangles;
  std::vector<std::size_t> incident_offsets;
  std::vector<std::size_t> incident_triangles;
  std::vector<std::size_t> adjacent_offsets;
  std::vector<std::size_t> adjacent_nodes;
  std::size_t solid_angle_fallbacks{0};
  std::size_t solid_angle_clamps{0};
  // Where each node normal came from: the analytic SES at the vertex, or the
  // area-weighted star of its triangles.  `analytic_normal_rejected` counts
  // analytic normals turned down because the star faces another sheet.
  std::size_t analytic_normal_nodes{0};
  std::size_t averaged_normal_nodes{0};
  std::size_t analytic_normal_rejected{0};
  std::shared_ptr<const CurvedNodePatchGeometry> curved;
};

// Dimensionless quadrature of one unit-sphere triangle. Shared by all atoms
// using that mesh triangle; storage scales with mesh resolution, not atoms.
struct SpherePatchQuadrature {
  std::vector<std::array<double, 4>> nodes; // unit x,y,z and spherical area
  double self_double_layer{0.0}; // integral dG0/dn_y, including 1/(4*pi)
  std::shared_ptr<const std::vector<std::array<double, 4>>> spheres;
};

struct Panel {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  // Optional quadrature center for source integration.  Ordinary triangle
  // panels use the collocation point above for both roles.  A node-centered
  // dual patch is collocated at the mesh vertex but its area centroid is the
  // consistent zeroth/first-moment source location.
  double source_x{0.0};
  double source_y{0.0};
  double source_z{0.0};
  // With the centroid, the unit mean normal of the patch (its normal
  // integral, normalized).  The collocation normal below is the surface
  // normal at the vertex; on a curved surface the two differ by about the
  // curvature times the vertex-centroid offset, an error of the one-point
  // source rule that does not decay with distance if the vertex normal is
  // used.  Read through panel_source_n*() below.
  double source_nx{0.0};
  double source_ny{0.0};
  double source_nz{0.0};
  bool has_source_quadrature_center{false};
  double nx{0.0};
  double ny{0.0};
  double nz{0.0};
  double area{0.0};
  double diagonal{1.0};
  double solid_angle{0.0};
  std::size_t atom_index{0};
  double v0x{0.0};
  double v0y{0.0};
  double v0z{0.0};
  double v1x{0.0};
  double v1y{0.0};
  double v1z{0.0};
  double v2x{0.0};
  double v2y{0.0};
  double v2z{0.0};
  std::size_t v0_index{static_cast<std::size_t>(-1)};
  std::size_t v1_index{static_cast<std::size_t>(-1)};
  std::size_t v2_index{static_cast<std::size_t>(-1)};
  bool has_triangle_geometry{false};
  std::shared_ptr<const SpherePatchQuadrature> sphere_patch_quadrature;
  // Unit analytic surface normal per mesh vertex, indexed by v*_index and
  // shared by every triangle of one mesh; NaN where the vertex is not on the
  // analytic surface.  Null when the surface has no analytic description.
  std::shared_ptr<const std::vector<std::array<double, 3>>>
      analytic_vertex_normals;
  std::shared_ptr<const NodePatchGeometry> node_patch_geometry;
  std::size_t node_patch_index{static_cast<std::size_t>(-1)};
};

// A panel as a one-point source: the patch centroid and mean normal of a node
// patch, the collocation point and normal of any other panel.
inline double panel_source_x(const Panel &p) noexcept {
  return p.has_source_quadrature_center ? p.source_x : p.x;
}
inline double panel_source_y(const Panel &p) noexcept {
  return p.has_source_quadrature_center ? p.source_y : p.y;
}
inline double panel_source_z(const Panel &p) noexcept {
  return p.has_source_quadrature_center ? p.source_z : p.z;
}
inline double panel_source_nx(const Panel &p) noexcept {
  return p.has_source_quadrature_center ? p.source_nx : p.nx;
}
inline double panel_source_ny(const Panel &p) noexcept {
  return p.has_source_quadrature_center ? p.source_ny : p.ny;
}
inline double panel_source_nz(const Panel &p) noexcept {
  return p.has_source_quadrature_center ? p.source_nz : p.nz;
}

inline const std::array<double, 4> &sphere_patch_sphere(const Panel &panel) {
  return panel.sphere_patch_quadrature->spheres->at(panel.atom_index);
}

inline double jcp_lhs_phi_for_panel(const Panel &panel, double eps_ratio,
                                    double flat_lhs) noexcept {
  (void)panel;
  (void)eps_ratio;
  // The SES is a smooth interface; the analytical trace jump is one half.
  // A polyhedral vertex solid angle is retained as a mesh-quality audit but
  // must not replace that smooth-surface jump coefficient.
  return flat_lhs;
}

inline double jcp_lhs_phin_for_panel(const Panel &panel, double eps_ratio,
                                     double flat_lhs) noexcept {
  (void)panel;
  (void)eps_ratio;
  return flat_lhs;
}

struct Vertex {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double nx{0.0};
  double ny{0.0};
  double nz{0.0};
  std::size_t atom_index{0};
  // True when (nx, ny, nz) is the certified analytic SES normal at a vertex
  // that lies on the surface, false for a star or marching normal.
  bool normal_analytic{false};
};

struct Triangle {
  std::size_t v0{0};
  std::size_t v1{0};
  std::size_t v2{0};
  std::size_t atom_index{0};
  double cx{0.0};
  double cy{0.0};
  double cz{0.0};
  double nx{0.0};
  double ny{0.0};
  double nz{0.0};
  double area{0.0};
};

struct SurfaceMesh {
  std::vector<Vertex> vertices;
  std::vector<Triangle> triangles;
  std::vector<Panel> panels;
  std::size_t exposed_candidates{0};
  std::size_t buried_candidates{0};
  std::size_t solid_angle_fallbacks{0};
  // Diagonals inside the accept band but outside [0.25,1.75] are clamped.
  // Counted apart from fallbacks: a clamp is a silent change, not a reject.
  std::size_t solid_angle_clamps{0};
  std::size_t nonmanifold_edges{0};
  double avg_valence{0.0};
  double diag_min{1.0};
  double diag_max{1.0};
  std::size_t cell_count{0};
  std::size_t max_neighbor_count{0};
  double mean_neighbor_count{0.0};
  double neighbor_build_ms{0.0};
  double mesh_build_ms{0.0};
  double solid_angle_ms{0.0};
  std::string sphere_mesh_mode{"latlong"};
  double quality_total_area_a2{0.0};
  double quality_area_rel_error_vs_4piR2{0.0};
  double quality_triangle_area_min{0.0};
  double quality_triangle_area_mean{0.0};
  double quality_triangle_area_max{0.0};
  double quality_triangle_aspect_max{0.0};
  double quality_normal_radial_dot_min{0.0};
  double quality_normal_radial_dot_mean{0.0};
  double quality_centroid_radial_error_max{0.0};
  double quality_planar_area_a2{0.0};
  double quality_planar_area_rel_error_vs_4piR2{0.0};
  double quality_enclosed_volume_a3{0.0};
  double quality_enclosed_volume_rel_error_vs_4piR3_over3{0.0};
  double quality_triangle_area_cv{0.0};
};

inline constexpr std::size_t kNearBlockCount = 4;
inline constexpr std::size_t kNearPairTypeCount = 4;
inline constexpr std::size_t kNearOrientationCount = 3;
inline constexpr std::size_t kNearBlockAuditBucketCount =
    kNearBlockCount * kNearPairTypeCount * kNearOrientationCount;

struct NearBlockAuditBucket {
  std::size_t accepted{0};
  std::size_t rejected{0};
  std::size_t candidate_worse_than_centroid{0};
  std::size_t reference_nonconverged{0};
  std::size_t orientation_sign_mismatch{0};
  std::size_t strict_ref_reject{0};
  double max_candidate_ref_rel{0.0};
  double sum_candidate_ref_rel{0.0};
  double max_centroid_ref_rel{0.0};
  std::size_t k2_laplace_sign_ok{0};
  double max_k2_yukawa_smooth_ref_rel{0.0};
  double max_k2_orientation_pair_ref_rel{0.0};
  std::size_t sample_count{0};
};

using NearBlockAuditBuckets =
    std::array<NearBlockAuditBucket, kNearBlockAuditBucketCount>;

inline constexpr std::size_t kGeometricGapBucketCount = 4;
inline constexpr std::size_t kGeometricCenterBucketCount = 3;
inline constexpr std::size_t kGeometricKernelCount = 3;
inline constexpr std::size_t kGeometricNearAuditBucketCount =
    kGeometricGapBucketCount * kGeometricCenterBucketCount *
    kGeometricKernelCount;
inline constexpr std::size_t kJcpExpansionChannelCount = 4;

struct GeometricNearAuditBucket {
  std::size_t pairs{0};
  double min_gap{0.0};
  double max_gap_ratio{0.0};
  double sum_gap_ratio{0.0};
  double sum_center_ratio{0.0};
};

using GeometricNearAuditBuckets =
    std::array<GeometricNearAuditBucket, kGeometricNearAuditBucketCount>;

struct Options {
  double eps_in{1.0};
  double eps_out{80.0};
  double salt_m{0.15};
  double temperature_k{298.15};
  int max_iter{10000};
  double residual_tol{1.0e-6};
  int gmres_restart{32};
  std::string tree_preset{"off"};
  std::string sum_mode{"auto"};
  double probe_radius{0.0};
  std::string surface_mode{"vdw"};
  std::string surface_geometry{"triangle"};
  std::string diagonal_mode{"solid-angle"};
  int surface_point_count{96};
  std::string surface_output{"off"};
  bool surface_only{false};
  bool highres_surface{false};
  bool calderon_trace_audit{false};
  std::string calderon_aux_surface{"off"};
  double calderon_aux_grid_spacing{1.25};
  std::string primary_surface{"sphere-patch"};
  // Matches the CLI default in arguments.h; see
  // docs/TITAN_SES_ENGINE_FIX_2026-09-15.md for why it is not `current`.
  std::string analytic_ses_engine{"patch-ray-experimental"};
  // patch-ray-resolution remeshing target edge in A: 0 = grid spacing,
  // negative = keep the marching-tetrahedra mesh.
  double ses_remesh_edge{0.0};
  int ses_remesh_iterations{6};
  // Charge-adaptive remeshing (calderon::ChargeRefinement): alpha, 0 = off,
  // and the smallest target edge in A.
  double ses_charge_refine{0.0};
  double ses_charge_refine_min_edge{0.2};
  double ses_charge_refine_min_charge{0.1};
  double ses_grid_spacing{0.75};
  std::size_t ses_max_grid_vertices{4000000};
  std::string sphere_mesh_mode{"auto"};
  // See Arguments::tabi_sphere_patch_clip.
  bool sphere_patch_clip{true};
  int tree_leaf_size{32};
  int curved_pair_cache_mb{256};
  bool tree_leaf_size_explicit{false};
  double tree_opening_theta{0.45};
  bool tree_opening_theta_explicit{false};
  int taylor_order{3};
  std::string derivative_mode{"analytic"};
  std::string formulation{"jcp2013"};
  std::string discretization{"centroid"};
  std::string energy_projection{"boundary"};
  // Curved surfaces: cells farther than this many cell radii from an atom
  // enter its reaction potential as one aggregate point.  0 sums every sample.
  // A one-point cell leaves an error of order 1/eta^2 that does not fall with
  // the mesh spacing (the dipole kernel is scale free on a surface); 32 keeps
  // it near 1e-4 on the Kirkwood sphere.
  double energy_far_eta{32.0};
  // Gauss order per direction of the stored curved dual-cell samples.
  int curved_sample_order{2};
  // FMM route only: a non-adjacent pair is integrated over the curved cell
  // when the target lies within this many cell radii of the source vertex,
  // and takes the one-point kernel beyond (CurvedNodePatchGeometry).  As with
  // energy_far_eta the one-point error goes as 1/zone^2 independent of the
  // spacing: about 8e-4 at 4, 3e-4 at 6, 1.5e-4 at 8 on the Kirkwood sphere.
  double curved_near_zone_radii{8.0};
  std::string preconditioner{"auto"};
  std::string preconditioner_block_near{"singular"};
  int preconditioner_block_panel_cap{32};
  int preconditioner_overlap_panel_cap{64};
  int preconditioner_ilu_neighbor_cap{192};
  double preconditioner_ilu_drop_tol{0.0};
  std::string preconditioner_ilu_regularization{"adaptive"};
  double preconditioner_ilu_regularization_max{1.0e-2};
  int preconditioner_memory_budget_mb{256};
  int preconditioner_coarse_aggregate_cap{16};
  std::string preconditioner_coarse_base{"diagonal"};
  std::string preconditioner_coarse_verify{"off"};
  std::string near_integration{"auto"};
  std::string quad_buffer{"auto"};
  std::string leaf_near_policy{"auto"};
  std::string leaf_panel_kernel{"auto"};
  std::string adaptive_level{"strict"};
  std::string cluster_policy{"off"};
  std::string leaf_pair_kernel{"auto"};
  std::string leaf_kernel_profile{"off"};
  std::string leaf_work_profile{"off"};
  std::string matvec_cost_profile{"off"};
  std::string jcp_direct_kernel{"baseline"};
  std::string leaf_pair_selection_profile{"off"};
  std::string leaf_pair_probe_cache{"off"};
  std::string leaf_pair_target_group{"scan"};
  std::string leaf_pair_directness_probe{"full-traversal"};
  std::string leaf_pair_directness_cache{"off"};
  std::string leaf_pair_directness_verify{"off"};
  std::string leaf_pair_scheduler{"inline-select"};
  int leaf_pair_worklist_max_active{2048};
  std::string leaf_pair_worklist_verify{"off"};
  std::string leaf_block_verify{"off"};
  std::string leaf_mac{"off"};
  std::string fmm_m2l{"off"};
  std::string fmm_engine{"cartesian"};
  int bie_fmm_cartesian_order{0};
  int kifmm_order{6};
  double kifmm_equiv_scale{1.05};
  double kifmm_check_scale{1.60};
  std::string kifmm_solver{"qr"};
  std::string kifmm_profile{"off"};
  std::string kifmm_full_apply{"serial"};
  std::string kifmm_ilu_graph{"cartesian-near"};
  int kifmm_hierarchy_order{8};
  double kifmm_hierarchy_theta{0.6};
  std::string kifmm_hierarchy_real_panel_guard{"strict-local-check"};
  double kifmm_hierarchy_guard_margin_factor{0.0};
  std::string kifmm_hierarchy_tree_stop_policy{"panel-radius-occupancy"};
  double kifmm_hierarchy_tree_panel_radius_factor{1.5};
  int kifmm_hierarchy_tree_occupancy_stop{32};
  std::string kifmm_hierarchy_m2l_eval_policy{"target-check-batched"};
  std::string kifmm_hierarchy_matrix_cache_policy{"canonical-offset-cache"};
  double kifmm_hierarchy_matrix_cache_limit_mb{0.0};
  std::string kifmm_hierarchy_gemm_batch_policy{"per-pair"};
  std::string kifmm_hierarchy_m2l_parallel_policy{"serial"};
  int kifmm_hierarchy_m2l_parallel_workers{1};
  int cartesian_k4_order{5};
  double cartesian_k4_theta{0.6};
  double cartesian_k4_geometry_theta{0.4};
  double cartesian_k4_tail_tol{0.002};
  std::string interaction_mode{"auto"};
  std::string node_bounds{"moment"};
  std::string tree_builder{"grouped-morton"};
  std::string expansion_center{"legacy"};
  std::string fragmentation_safety{"off"};
  std::string mac_safety{"off"};
  double mac_gap_factor{1.5};
  std::string moment_audit{"off"};
  std::string moment_construction{"translated"};
  std::string moment_sum{"naive"};
  std::string convergence_audit{"off"};
  std::string convergence_safety{"off"};
  double convergence_rho_cap{0.0};
  double mac_aspect_cap{0.0};
  std::string convergence_estimator{"off"};
  double convergence_tail_threshold{1.0e4};
  double convergence_estimator_error_cap{1.0e-3};
  double convergence_unstable_rho_cap{0.25};
  std::string convergence_calibration{"stage3j"};
  std::string convergence_dielectric_amp{"off"};
  std::string convergence_estimator_model{"stage3l"};
  std::string convergence_channel_calibration{"stage3m"};
  std::string convergence_split_policy{"global"};
  std::string convergence_split_selectivity{"legacy"};
  double convergence_split_absolute_floor{0.0};
  std::string convergence_overreject_audit{"off"};
  std::string convergence_overreject_detail{"off"};
  std::string convergence_shape_audit{"off"};
  std::string convergence_kernel_response_audit{"off"};
  std::string channel_missed_semantics{"legacy"};
  std::string accepted_miss_diagnostics{"off"};
  double accepted_miss_absolute_floor{1.0e-7};
  double convergence_k3_accepted_rho_cap{0.0};
  std::string mac_distance_policy{"center"};
  double mac_surface_margin_factor{0.25};
  std::string mac_surface_margin_policy{"all"};
  double mac_surface_margin_aspect_threshold{3.0};
  double mac_surface_margin_rho_trigger{0.55};
  double mac_surface_margin_soft_aspect_threshold{0.0};
  double mac_surface_margin_soft_rho_trigger{0.0};
  double mac_surface_margin_soft_factor{0.10};
  int target_batch_size{1};
  int leaf_pair_min_source_panels{64};
  int leaf_pair_min_targets{4};
  int leaf_pair_max_lookahead{32};
  std::string expansion_audit{"off"};
  std::string expansion_audit_channel{"off"};
  int expansion_audit_sample_count{512};
  int expansion_audit_max_panels{2048};
  std::string matvec_audit{"off"};
  int matvec_audit_sample_count{1024};
  std::string node_patch_operator_audit{"on"};
  std::string fmm_cartesian_audit{"off"};
  std::string fmm_cartesian_error_audit{"off"};
  // Internal-only marker set by the outer Cartesian retry wrapper.  It is
  // deliberately not a CLI option: a normal channel-shadow solve must not
  // pay the retry geometry-fingerprint cost or acquire retry semantics.
  bool fmm_cartesian_retry_runtime{false};
  std::string fmm_local{"off"};
  std::string fmm_local_refine{"off"};
  double fmm_local_refine_tol{4.0e-3};
  std::string fmm_local_cache{"off"};
  int fmm_local_cache_budget_mb{256};
  std::string fmm_local_fallback{"scalar"};
  std::string fmm_local_plan_builder{"legacy-rescan"};
  int fmm_local_refine_probe_count{4};
  std::string fmm_local_refine_probe_mode{"rademacher"};
  std::string fmm_local_geometry_guard{"off"};
  double fmm_local_geometry_theta{0.0};
  // See Arguments::tabi_fmm_local_geometry_sweep (diagnostic only).
  bool fmm_local_geometry_sweep{false};
  std::string progress{"off"};
  int progress_interval_seconds{5};
  std::string true_residual_probe{"restart"};
  std::string true_residual_probe_scope{"full"};
  int true_residual_probe_interval{8};
  int true_residual_sample_count{1024};
};

struct Result {
  std::string status{"ok"};
  bool surface_only{false};
  std::string reaction_energy_source{"tabi_boundary_integral"};
  std::string convergence_reason{"not_started"};
  std::string residual_history{};
  std::string tree_preset{"off"};
  std::size_t panel_count{0};
  std::size_t memory_bytes{0};
  std::size_t peak_memory_bytes{0};
  int surface_point_count_requested{96};
  double surface_area_a2{0.0};
  double panel_density_per_a2{0.0};
  std::string panel_budget_status{"off"};
  ChargeResolutionDiagnostic charge_resolution;
  double surface_build_ms{0.0};
  double surface_neighbor_ms{0.0};
  double surface_mesh_ms{0.0};
  double solid_angle_ms{0.0};
  std::string sphere_mesh_mode{"latlong"};
  double surface_quality_total_area_a2{0.0};
  double surface_quality_area_rel_error_vs_4piR2{0.0};
  double surface_quality_triangle_area_min{0.0};
  double surface_quality_triangle_area_mean{0.0};
  double surface_quality_triangle_area_max{0.0};
  double surface_quality_triangle_aspect_max{0.0};
  double surface_quality_normal_radial_dot_min{0.0};
  double surface_quality_normal_radial_dot_mean{0.0};
  double surface_quality_centroid_radial_error_max{0.0};
  double surface_quality_planar_area_a2{0.0};
  double surface_quality_planar_area_rel_error_vs_4piR2{0.0};
  double surface_quality_enclosed_volume_a3{0.0};
  double surface_quality_enclosed_volume_rel_error_vs_4piR3_over3{0.0};
  double surface_quality_triangle_area_cv{0.0};
  std::string surface_topology_source{"not_built"};
  std::string surface_analytic_ses_engine{"not_applicable"};
  std::string surface_status_detail{"not_built"};
  std::string surface_cavity_policy{"not_applicable"};
  std::size_t surface_odd_ray_rows{0};
  std::size_t surface_patch_ray_unresolved_uncertified_rows{0};
  std::size_t surface_morphology_fallback_vertices{0};
  std::string surface_analytic_orientation_status{"not_run"};
  std::size_t surface_analytic_inverted_faces{0};
  std::size_t surface_analytic_orientation_uncertified_faces{0};
  std::size_t surface_analytic_self_intersections{0};
  std::size_t surface_wide_repair_attempts{0};
  std::size_t surface_topology_certificate_accepts{0};
  std::size_t surface_topology_fallback_audits{0};
  double surface_initial_collapse_ms{0.0};
  double surface_stable_sliver_ms{0.0};
  double surface_full_projection_ms{0.0};
  double surface_orientation_repair_ms{0.0};
  double surface_wide_repair_ms{0.0};
  double surface_self_intersection_audit_ms{0.0};
  std::size_t surface_filled_cavity_regions{0};
  std::size_t surface_filled_cavity_vertices{0};
  std::size_t surface_poor_triangles_before{0};
  std::size_t surface_poor_triangles_after{0};
  double surface_triangle_quality_min_before{0.0};
  double surface_triangle_quality_min_after{0.0};
  double surface_max_edge_ratio_before{0.0};
  double surface_max_edge_ratio_after{0.0};
  std::string surface_analytic_audit_status{"not_run"};
  std::size_t surface_analytic_audit_samples{0};
  std::size_t surface_analytic_audit_hits{0};
  std::size_t surface_analytic_audit_misses{0};
  double surface_analytic_projection_distance_rms{0.0};
  double surface_analytic_projection_distance_max{0.0};
  double surface_analytic_patch_residual_rms{0.0};
  double surface_analytic_patch_residual_max{0.0};
  double surface_analytic_normal_angle_rms_deg{0.0};
  double surface_analytic_normal_angle_max_deg{0.0};
  double surface_sampled_hausdorff_proxy{0.0};
  std::string surface_analytic_normal_quality_status{"not_run"};
  std::size_t surface_analytic_orientation_smooth_faces{0};
  std::size_t surface_analytic_orientation_seam_faces{0};
  std::size_t surface_analytic_normal_cosine_below_025_faces{0};
  std::size_t surface_analytic_normal_cosine_below_050_faces{0};
  double surface_analytic_normal_area_weighted_rms_deg{0.0};
  double surface_analytic_normal_p99_deg{0.0};
  double surface_analytic_smooth_normal_p99_deg{0.0};
  double surface_analytic_seam_normal_p99_deg{0.0};
  double surface_analytic_normal_area_below_cosine_025_fraction{0.0};
  double surface_analytic_normal_area_below_cosine_050_fraction{0.0};
  std::string calderon_trace_space_status{"off"};
  std::size_t calderon_trace_vertices{0};
  std::size_t calderon_trace_edges{0};
  std::size_t calderon_trace_boundary_edges{0};
  std::size_t calderon_trace_orientation_mismatches{0};
  std::size_t calderon_trace_nonmanifold_edges{0};
  std::size_t calderon_trace_unreferenced_vertices{0};
  std::size_t calderon_trace_components{0};
  std::int64_t calderon_trace_euler_characteristic{0};
  std::size_t calderon_trace_genus_sum{0};
  bool calderon_trace_topology_invariants_valid{false};
  std::string calderon_aux_surface_mode{"off"};
  std::string calderon_aux_trace_status{"off"};
  std::size_t calderon_aux_panels{0};
  std::size_t calderon_aux_vertices{0};
  std::size_t calderon_aux_edges{0};
  std::size_t calderon_aux_boundary_edges{0};
  std::size_t calderon_aux_orientation_mismatches{0};
  std::size_t calderon_aux_nonmanifold_edges{0};
  std::size_t calderon_aux_unreferenced_vertices{0};
  std::size_t calderon_aux_grid_vertices{0};
  double calderon_aux_area_a2{0.0};
  double calderon_aux_build_ms{0.0};
  std::size_t exposed_points{0};
  std::size_t buried_points{0};
  double total_charge{0.0};
  int iterations{0};
  double residual{0.0};
  int gmres_restart_requested{32};
  int gmres_restart_effective{32};
  double best_residual{0.0};
  int best_residual_iter{0};
  double final_residual{0.0};
  double best_preconditioned_residual{0.0};
  double final_preconditioned_residual{0.0};
  double true_residual_abs{0.0};
  double true_residual_rel{0.0};
  std::string true_residual_probe_mode{"restart"};
  std::string true_residual_probe_scope{"full"};
  int true_residual_probe_interval{8};
  int true_residual_sample_count{1024};
  std::size_t true_residual_probe_calls{0};
  double true_residual_probe_ms{0.0};
  double sampled_true_residual_rel{0.0};
  std::size_t sampled_true_residual_rows{0};
  std::size_t sampled_probe_calls{0};
  double sampled_probe_ms{0.0};
  std::size_t full_true_residual_confirm_calls{0};
  double full_true_residual_confirm_ms{0.0};
  bool converged{false};
  std::string sum_mode_effective{"direct"};
  double tree_build_ms{0.0};
  std::size_t tree_nodes{0};
  std::size_t tree_leaf_count{0};
  std::size_t near_interactions{0};
  std::size_t far_interactions{0};
  std::string matvec_mode{"direct"};
  double rhs_ms{0.0};
  double gmres_ms{0.0};
  double matvec_ms_total{0.0};
  double matvec_ms_avg{0.0};
  std::size_t matvec_calls{0};
  double tree_vs_direct_matvec_rel{0.0};
  std::size_t matvec_audit_rows{0};
  double matvec_audit_ms{0.0};
  FmmCartesianPreAuditTelemetry fmm_cartesian_pre_audit{};
  FmmCartesianSolutionAuditTelemetry fmm_cartesian_solution_audit{};
  std::array<FmmCartesianChannelAudit, kJcpExpansionChannelCount>
      fmm_cartesian_channel_audit{};
  FmmCartesianRetryTelemetry fmm_cartesian_retry{};
  // Raw per-attempt geometry identity material.  It is populated only by the
  // opt-in retry runtime after node-patch surface construction, then copied
  // into the public retry snapshot by the wrapper.
  std::uint64_t fmm_cartesian_retry_geometry_hash{0};
  bool fmm_cartesian_retry_geometry_valid{false};
  double orthogonalization_ms{0.0};
  double preconditioner_ms{0.0};
  std::string surface_geometry{"triangle"};
  std::size_t mesh_vertices{0};
  std::size_t mesh_triangles{0};
  std::size_t boundary_nodes{0};
  double avg_valence{0.0};
  std::size_t nonmanifold_edges{0};
  std::size_t solid_angle_fallbacks{0};
  std::size_t solid_angle_clamps{0};
  double diag_min{1.0};
  double diag_max{1.0};
  int taylor_order{3};
  std::string derivative_mode{"analytic"};
  double interaction_build_ms{0.0};
  double tree_eval_ms{0.0};
  double tree_moment_ms{0.0};
  double tree_traversal_ms{0.0};
  double near_eval_ms{0.0};
  double far_eval_ms{0.0};
  double singular_near_eval_ms{0.0};
  double leaf_near_eval_ms{0.0};
  double leaf_panel_eval_ms{0.0};
  double leaf_tile_eval_ms{0.0};
  double leaf_residual_eval_ms{0.0};
  double smooth_leaf_node_eval_ms{0.0};
  std::size_t singular_near_interactions{0};
  std::size_t leaf_near_interactions{0};
  std::size_t far_node_interactions{0};
  std::size_t leaf_panel_interactions{0};
  std::size_t leaf_tile_interactions{0};
  std::size_t leaf_tile_panel_interactions{0};
  std::size_t leaf_residual_panel_interactions{0};
  std::size_t smooth_leaf_node_interactions{0};
  std::size_t leaf_pruned_interactions{0};
  std::size_t leaf_mac_candidates{0};
  std::size_t leaf_mac_accepted{0};
  std::size_t leaf_mac_rejected{0};
  std::size_t leaf_mac_far_panels{0};
  std::size_t leaf_mac_direct_panels_saved_estimate{0};
  std::string leaf_work_profile{"off"};
  std::size_t leaf_work_mac_terminal_panels{0};
  std::size_t leaf_work_physical_gap_panels{0};
  std::size_t leaf_work_fragmentation_panels{0};
  std::size_t leaf_work_convergence_panels{0};
  std::size_t leaf_work_convergence_estimator_panels{0};
  std::size_t leaf_work_convergence_rho_panels{0};
  std::size_t leaf_work_convergence_aspect_panels{0};
  std::size_t leaf_work_surface_margin_panels{0};
  std::size_t leaf_work_surface_margin_hard_panels{0};
  std::size_t leaf_work_surface_margin_soft_panels{0};
  std::size_t leaf_work_k3_rho_cap_panels{0};
  std::size_t leaf_work_singular_fallback_panels{0};
  std::size_t leaf_work_scalar_tail_panels{0};
  std::size_t leaf_work_profile_accounted_panels{0};
  double leaf_work_mac_terminal_frac{0.0};
  double leaf_work_convergence_frac{0.0};
  double leaf_work_surface_margin_frac{0.0};
  double leaf_work_singular_fallback_frac{0.0};
  double leaf_work_scalar_tail_frac{0.0};
  std::string matvec_cost_profile{"off"};
  double leaf_panels_per_second{0.0};
  double clean_leaf_panels_per_second{0.0};
  double tree_traversal_fraction{0.0};
  std::size_t adaptive_smooth_leaf_accepted{0};
  std::size_t adaptive_smooth_leaf_rejected{0};
  std::size_t adaptive_span_tile_accepted{0};
  std::size_t adaptive_span_tile_rejected{0};
  std::size_t adaptive_exact_leaf_fallbacks{0};
  std::size_t adaptive_reject_one_ring_overlap{0};
  std::size_t adaptive_reject_error_bound_failed{0};
  std::size_t adaptive_reject_p4_check_failed{0};
  std::size_t adaptive_reject_too_close{0};
  double leaf_work_reduction_ratio{0.0};
  std::size_t target_cluster_interactions{0};
  std::size_t target_cluster_targets{0};
  std::size_t target_cluster_panels_pruned{0};
  std::size_t target_cluster_rejected{0};
  double target_cluster_eval_ms{0.0};
  std::size_t leaf_pair_batch_interactions{0};
  std::size_t leaf_pair_batch_targets{0};
  std::size_t leaf_pair_batch_source_panels{0};
  std::size_t leaf_pair_batch_residual_panels{0};
  std::size_t leaf_pair_batch_fallbacks{0};
  double leaf_pair_batch_ms{0.0};
  std::string leaf_pair_batch_kernel{"scalar"};
  std::size_t leaf_pair_batch_vectorized_sources{0};
  std::size_t leaf_pair_batch_vector_lanes{0};
  double leaf_pair_batch_pairs_per_second{0.0};
  std::string leaf_kernel_profile{"off"};
  std::string jcp_direct_kernel{"baseline"};
  std::size_t leaf_kernel_scalar_tasks{0};
  std::size_t leaf_kernel_batch_tasks{0};
  std::size_t leaf_kernel_source_panel_evals{0};
  double leaf_kernel_scalar_ms{0.0};
  double leaf_kernel_target_avx2_ms{0.0};
  double leaf_kernel_tail_ms{0.0};
  std::size_t leaf_pair_batch_soa_sources{0};
  double leaf_pair_batch_soa_ms{0.0};
  double leaf_pair_batch_scalar_tail_ms{0.0};
  double leaf_pair_batch_clean_source_panels_per_second{0.0};
  std::string leaf_pair_selection_profile{"off"};
  std::string leaf_pair_probe_cache{"off"};
  std::string leaf_pair_target_group{"scan"};
  int leaf_pair_max_lookahead{32};
  std::size_t leaf_pair_selection_candidate_scans{0};
  std::size_t leaf_pair_selection_clean_rejects{0};
  std::size_t leaf_pair_selection_probe_calls{0};
  std::size_t leaf_pair_selection_probe_node_visits{0};
  double leaf_pair_selection_probe_ms{0.0};
  double leaf_pair_selection_ms{0.0};
  std::size_t leaf_pair_selection_probe_cache_hits{0};
  std::size_t leaf_pair_selection_probe_cache_misses{0};
  std::size_t leaf_pair_selection_insufficient_targets{0};
  std::size_t leaf_pair_selection_same_target_leaf_rejects{0};
  std::string leaf_pair_directness_probe{"full-traversal"};
  std::string leaf_pair_directness_cache{"off"};
  std::string leaf_pair_directness_verify{"off"};
  std::size_t leaf_pair_directness_ancestor_probe_calls{0};
  std::size_t leaf_pair_directness_ancestor_probe_node_visits{0};
  double leaf_pair_directness_ancestor_probe_ms{0.0};
  std::size_t leaf_pair_directness_pair_cache_hits{0};
  std::size_t leaf_pair_directness_pair_cache_misses{0};
  std::size_t leaf_pair_directness_all_direct_proofs{0};
  std::size_t leaf_pair_directness_mixed_fallbacks{0};
  std::size_t leaf_pair_directness_verify_checks{0};
  std::size_t leaf_pair_directness_verify_mismatches{0};
  std::string leaf_pair_scheduler{"inline-select"};
  int leaf_pair_worklist_max_active{2048};
  std::string leaf_pair_worklist_verify{"off"};
  std::size_t leaf_pair_worklist_tasks{0};
  std::size_t leaf_pair_worklist_groups{0};
  std::size_t leaf_pair_worklist_batched_groups{0};
  std::size_t leaf_pair_worklist_scalar_flush_tasks{0};
  std::size_t leaf_pair_worklist_forced_flushes{0};
  std::size_t leaf_pair_worklist_active_groups_max{0};
  double leaf_pair_worklist_ms{0.0};
  std::size_t leaf_pair_worklist_verify_checks{0};
  std::size_t leaf_pair_worklist_verify_mismatches{0};
  std::string leaf_block_verify{"off"};
  std::string leaf_mac{"off"};
  std::string fmm_m2l_mode{"off"};
  int fmm_m2l_expansion_order{0};
  std::size_t fmm_m2l_near_pairs{0};
  std::size_t fmm_m2l_far_pairs{0};
  double fmm_m2l_setup_ms{0.0};
  double fmm_m2l_apply_ms{0.0};
  bool fmm_m2l_finite_outputs{true};
  bool fmm_local_gmres_active{false};
  bool bie_cartesian_full_fmm_active{false};
  std::string bie_cartesian_full_fmm_requested{"off"};
  std::string bie_cartesian_full_fmm_effective{"off"};
  int bie_cartesian_full_fmm_order{0};
  int bie_cartesian_full_fmm_leaf_size{0};
  double bie_cartesian_full_fmm_opening_theta{0.0};
  std::uint64_t bie_cartesian_full_fmm_tree_hash{0};
  std::uint64_t bie_cartesian_full_fmm_worklist_hash{0};
  std::uint64_t bie_cartesian_full_fmm_policy_hash{0};
  std::size_t bie_cartesian_full_fmm_nodes{0};
  std::size_t bie_cartesian_full_fmm_far_pairs{0};
  std::size_t bie_cartesian_full_fmm_direct_pairs{0};
  std::uint64_t bie_cartesian_full_fmm_direct_panel_pairs_expected{0};
  std::uint64_t bie_cartesian_full_fmm_direct_panel_pairs_applied_last{0};
  std::uint64_t bie_cartesian_full_fmm_direct_panel_pairs_applied_total{0};
  std::size_t bie_cartesian_full_fmm_cache_entries{0};
  std::size_t bie_cartesian_full_fmm_cache_hits{0};
  std::size_t bie_cartesian_full_fmm_cache_bytes{0};
  std::size_t bie_cartesian_full_fmm_plan_bytes{0};
  std::size_t bie_cartesian_full_fmm_workspace_bytes{0};
  double bie_cartesian_full_fmm_tree_seconds{0.0};
  double bie_cartesian_full_fmm_worklist_seconds{0.0};
  double bie_cartesian_full_fmm_cache_seconds{0.0};
  double bie_cartesian_full_fmm_audit_seconds{0.0};
  double bie_cartesian_full_fmm_p2m_seconds{0.0};
  double bie_cartesian_full_fmm_m2m_seconds{0.0};
  double bie_cartesian_full_fmm_m2l_seconds{0.0};
  double bie_cartesian_full_fmm_l2l_seconds{0.0};
  double bie_cartesian_full_fmm_l2p_seconds{0.0};
  double bie_cartesian_full_fmm_direct_seconds{0.0};
  double bie_cartesian_full_fmm_m2m_parity_error{0.0};
  double bie_cartesian_full_fmm_l2l_parity_error{0.0};
  double fmm_local_prepare_seconds{0.0};
  std::size_t fmm_local_plan_build_count{0};
  std::size_t fmm_local_cache_build_count{0};
  std::size_t fmm_local_gmres_matvec_calls{0};
  double fmm_local_gmres_apply_seconds_total{0.0};
  double fmm_local_gmres_apply_seconds_mean{0.0};
  double fmm_local_moment_seconds_total{0.0};
  double fmm_local_m2l_seconds_total{0.0};
  double fmm_local_fallback_seconds_total{0.0};
  double fmm_local_l2p_seconds_total{0.0};
  double fmm_local_sampled_direct_true_residual{0.0};
  bool fmm_local_operator_finite{true};
  std::string fmm_local_refine_probe_mode{"rademacher"};
  std::size_t fmm_local_refine_probe_count{4};
  bool fmm_local_solution_probe_used{false};
  double fmm_local_linf_max_abs_tail{0.0};
  double fmm_local_linf_max_relative_tail{0.0};
  std::size_t fmm_local_linf_outlier_panels{0};
  std::string fmm_local_geometry_guard{"off"};
  double fmm_local_geometry_theta{0.0};
  std::size_t fmm_local_geometry_guard_rejected_cells{0};
  std::size_t fmm_local_geometry_guard_direct_fallback_pairs{0};
  double fmm_local_geometry_gap_min{0.0};
  double fmm_local_geometry_ratio_max{0.0};
  std::size_t leaf_block_target_leaves{0};
  std::size_t leaf_block_targets{0};
  std::size_t leaf_block_source_groups{0};
  std::size_t leaf_block_batched_groups{0};
  std::size_t leaf_block_scalar_tail_tasks{0};
  double leaf_block_group_ms{0.0};
  double leaf_block_flush_ms{0.0};
  double leaf_block_writeback_ms{0.0};
  std::size_t leaf_block_verify_checks{0};
  std::size_t leaf_block_verify_mismatches{0};
  std::string adaptive_level{"strict"};
  std::string cluster_policy{"off"};
  int target_batch_size{1};
  int leaf_pair_min_source_panels{64};
  int leaf_pair_min_targets{4};
  std::string leaf_near_policy{"exact"};
  std::string leaf_panel_kernel{"scalar"};
  std::string interaction_mode{"cached"};
  std::string node_bounds_mode{"moment"};
  std::string tree_builder{"grouped-morton"};
  std::string expansion_center{"legacy"};
  std::string fragmentation_safety{"off"};
  std::string mac_safety{"off"};
  double mac_gap_factor{1.5};
  std::size_t tree_depth_max{0};
  std::size_t mac_reject_radius_limited{0};
  double mac_accept_rate{0.0};
  double node_radius_max{0.0};
  double node_radius_mean{0.0};
  double node_aspect_ratio_max{0.0};
  double node_aspect_ratio_mean{0.0};
  double leaf_panel_work_per_matvec{0.0};
  double far_node_work_per_matvec{0.0};
  std::size_t far_geometric_unsafe_count{0};
  std::size_t far_geometric_unsafe_panels{0};
  double far_geometric_min_gap{0.0};
  double far_geometric_gap_ratio_min{0.0};
  double far_geometric_gap_ratio_mean{0.0};
  std::size_t far_geometric_audit_samples{0};
  std::size_t mac_physical_gap_rejects{0};
  std::size_t mac_convergence_rho_rejects{0};
  std::size_t mac_convergence_aspect_rejects{0};
  std::size_t mac_convergence_rho_reject_panels{0};
  std::size_t mac_convergence_aspect_reject_panels{0};
  std::size_t mac_surface_margin_rejects{0};
  std::size_t mac_surface_margin_reject_panels{0};
  std::size_t mac_surface_margin_applied{0};
  std::size_t mac_surface_margin_skipped{0};
  std::size_t mac_surface_margin_adaptive_rejects{0};
  std::size_t mac_surface_margin_adaptive_reject_panels{0};
  std::size_t mac_surface_margin_skipped_accepted_far{0};
  std::size_t mac_surface_margin_hard_applied{0};
  std::size_t mac_surface_margin_soft_applied{0};
  std::size_t mac_surface_margin_hard_rejects{0};
  std::size_t mac_surface_margin_hard_reject_panels{0};
  std::size_t mac_surface_margin_soft_rejects{0};
  std::size_t mac_surface_margin_soft_reject_panels{0};
  std::size_t fragmented_far_count{0};
  std::size_t fragmented_far_panels{0};
  std::size_t fragmented_split_rejects{0};
  double fragmentation_center_gap_ratio_max{0.0};
  double fragmentation_center_gap_ratio_mean{0.0};
  std::string moment_audit{"off"};
  std::string moment_construction{"translated"};
  std::string moment_sum{"naive"};
  std::string convergence_audit{"off"};
  std::string convergence_safety{"off"};
  double convergence_rho_cap{0.0};
  double mac_aspect_cap{0.0};
  std::string convergence_estimator{"off"};
  double convergence_tail_threshold{1.0e4};
  double convergence_estimator_error_cap{1.0e-3};
  double convergence_unstable_rho_cap{0.25};
  std::string convergence_calibration{"stage3j"};
  std::string convergence_estimator_model{"stage3l"};
  std::string convergence_channel_calibration{"stage3m"};
  std::string convergence_split_policy{"global"};
  std::string convergence_split_selectivity{"legacy"};
  double convergence_split_absolute_floor{0.0};
  std::string convergence_overreject_audit{"off"};
  std::string convergence_overreject_detail{"off"};
  std::string convergence_shape_audit{"off"};
  std::string convergence_kernel_response_audit{"off"};
  std::string channel_missed_semantics{"legacy"};
  std::string accepted_miss_diagnostics{"off"};
  double accepted_miss_absolute_floor{1.0e-7};
  double convergence_k3_accepted_rho_cap{0.0};
  std::string mac_distance_policy{"center"};
  double mac_surface_margin_factor{0.25};
  std::string mac_surface_margin_policy{"all"};
  double mac_surface_margin_aspect_threshold{3.0};
  double mac_surface_margin_rho_trigger{0.55};
  double mac_surface_margin_soft_aspect_threshold{0.0};
  double mac_surface_margin_soft_rho_trigger{0.0};
  double mac_surface_margin_soft_factor{0.10};
  std::size_t convergence_estimator_samples{0};
  std::size_t convergence_estimator_unstable_far_count{0};
  std::size_t convergence_estimator_unstable_far_panels{0};
  std::size_t convergence_estimator_split_rejects{0};
  std::size_t convergence_estimator_split_reject_panels{0};
  std::size_t convergence_low_abs_reaccepted_far_count{0};
  std::size_t convergence_low_abs_reaccepted_far_panels{0};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_low_abs_reaccepted_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_low_abs_reaccepted_panels{};
  double convergence_estimator_tail_ratio_max{0.0};
  double convergence_estimator_tail_ratio_mean{0.0};
  double convergence_estimator_error_max{0.0};
  double convergence_estimator_error_mean{0.0};
  double convergence_estimator_abs_error_max{0.0};
  double convergence_estimator_abs_error_mean{0.0};
  double convergence_tail_norm_p_max{0.0};
  double convergence_tail_norm_p_mean{0.0};
  double convergence_tail_norm_pm1_max{0.0};
  double convergence_tail_norm_pm1_mean{0.0};
  double convergence_effective_jcp_amp_max{0.0};
  double convergence_effective_jcp_amp_mean{0.0};
  std::string convergence_dielectric_amp{"off"};
  double convergence_dielectric_amp_max{0.0};
  double convergence_dielectric_amp_mean{0.0};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_estimated_error_max{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_estimated_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_effective_jcp_amp_max{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_effective_jcp_amp_mean{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_unstable_far_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_unstable_far_panels{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_split_rejects{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_split_reject_panels{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_high_error_caught{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_high_error_missed{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_high_error_caught_by_split{};
  std::size_t convergence_estimator_high_error_caught{0};
  std::size_t convergence_estimator_high_error_missed{0};
  std::size_t convergence_estimator_high_error_caught_by_split{0};
  std::size_t convergence_estimator_phantom_high_error_count{0};
  double convergence_estimator_phantom_high_error_abs_max{0.0};
  std::size_t convergence_overreject_count{0};
  std::size_t convergence_overreject_panels{0};
  std::size_t convergence_overreject_caught_count{0};
  std::size_t convergence_overreject_caught_panels{0};
  double convergence_overreject_abs_error_max{0.0};
  double convergence_overreject_abs_error_mean{0.0};
  double convergence_overreject_abs_error_p95{0.0};
  double convergence_overreject_abs_error_p99{0.0};
  double convergence_overreject_estimated_abs_error_max{0.0};
  double convergence_overreject_estimated_abs_error_mean{0.0};
  double convergence_overreject_required_scale_max{0.0};
  double convergence_overreject_required_scale_p95{0.0};
  double convergence_overreject_required_scale_p99{0.0};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_overreject_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_overreject_caught_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_overreject_panels{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_overreject_abs_error_max{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_overreject_abs_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_overreject_required_scale_max{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_overreject_required_scale_p99{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      phantom_high_error_channel_count{};
  std::array<double, kJcpExpansionChannelCount>
      phantom_high_error_channel_abs_max{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      accepted_miss_channel_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      accepted_miss_channel_panels{};
  std::array<double, kJcpExpansionChannelCount> accepted_miss_channel_rho_min{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_rho_mean{};
  std::array<double, kJcpExpansionChannelCount> accepted_miss_channel_rho_max{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_tail_ratio_mean{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_tail_ratio_max{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_estimated_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_estimated_error_max{};
  std::string accepted_miss_worst_channel{"none"};
  std::size_t convergence_k3_rho_cap_rejects{0};
  std::size_t convergence_k3_rho_cap_reject_panels{0};
  std::size_t moment_audit_nodes{0};
  double moment_translation_rel_error_max{0.0};
  double moment_translation_rel_error_mean{0.0};
  double moment_translation_offset_max{0.0};
  double moment_translation_offset_mean{0.0};
  double moment_cancellation_ratio_max{0.0};
  double moment_cancellation_ratio_mean{0.0};
  double moment_dynamic_range_max{0.0};
  std::string expansion_audit{"off"};
  std::string expansion_audit_channel{"off"};
  std::size_t far_expansion_audit_pairs{0};
  std::size_t far_expansion_audit_panels{0};
  double far_expansion_direct_ms{0.0};
  double far_expansion_rel_error_max{0.0};
  double far_expansion_rel_error_mean{0.0};
  double far_expansion_rel_error_p95{0.0};
  double far_expansion_rel_error_p99{0.0};
  std::array<double, kJcpExpansionChannelCount>
      far_expansion_channel_rel_error_max{};
  std::array<double, kJcpExpansionChannelCount>
      far_expansion_channel_rel_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      far_expansion_channel_rel_error_p99{};
  std::string far_expansion_worst_channel{"none"};
  bool far_expansion_mean_gate_pass{false};
  double far_convergence_rho_max{0.0};
  double far_convergence_rho_mean{0.0};
  double accepted_far_aspect_ratio_max{0.0};
  double accepted_far_aspect_ratio_mean{0.0};
  std::string quad_buffer_auto_candidate{"off"};
  std::string leaf_tile_status{"off"};
  double leaf_panel_pairs_per_second{0.0};
  std::string near_integration{"centroid"};
  std::string quad_buffer{"off"};
  std::size_t near_self_pairs{0};
  std::size_t near_one_ring_pairs{0};
  std::size_t near_duffy_pairs{0};
  std::size_t near_centroid_pairs{0};
  std::size_t near_fallbacks{0};
  std::size_t near_geometric_buffer_pairs{0};
  std::size_t near_geometric_buffer_gauss_pairs{0};
  std::size_t near_geometric_buffer_adaptive_pairs{0};
  std::size_t near_geometric_buffer_centroid_pairs{0};
  double near_geometric_buffer_min_gap{0.0};
  double near_geometric_buffer_mean_gap_ratio{0.0};
  double near_geometric_buffer_max_edge_mean{0.0};
  double near_integration_ms{0.0};
  std::size_t duffy_subdivisions_max{0};
  double duffy_subdivisions_mean{0.0};
  std::string near_kernel_policy{"centroid"};
  std::size_t near_k1_accepted{0};
  std::size_t near_k1_rejected{0};
  std::size_t near_k2_accepted{0};
  std::size_t near_k2_rejected{0};
  std::size_t near_k3_accepted{0};
  std::size_t near_k3_rejected{0};
  std::size_t near_k4_accepted{0};
  std::size_t near_k4_rejected{0};
  std::size_t near_one_ring_k1_accepted{0};
  std::size_t near_one_ring_k1_rejected{0};
  std::size_t near_one_ring_k2_accepted{0};
  std::size_t near_one_ring_k2_rejected{0};
  std::size_t near_one_ring_k3_accepted{0};
  std::size_t near_one_ring_k3_rejected{0};
  std::size_t near_one_ring_k4_accepted{0};
  std::size_t near_one_ring_k4_rejected{0};
  std::string near_k1_policy{"centroid"};
  std::string near_k2_policy{"centroid"};
  std::string near_k3_policy{"centroid"};
  std::string near_k4_policy{"centroid"};
  NearBlockAuditBuckets near_block_audit{};
  GeometricNearAuditBuckets geometric_near_audit{};
  double energy_kcal_mol{0.0};
  double energy_kj_mol{0.0};
  double energy_kt{0.0};
  std::string formulation{"jcp2013"};
  std::string energy_projection{"boundary"};
  double energy_projection_ms{0.0};
  std::string energy_projection_kernel{"scalar"};
  std::string preconditioner_requested{"auto"};
  std::string preconditioner_effective{"diagonal"};
  std::string preconditioner_block_near{"singular"};
  int preconditioner_block_panel_cap{32};
  int preconditioner_overlap_panel_cap{64};
  int preconditioner_ilu_neighbor_cap_requested{192};
  int preconditioner_ilu_neighbor_cap{192};
  double preconditioner_ilu_drop_tol{0.0};
  std::string preconditioner_ilu_regularization{"adaptive"};
  double preconditioner_ilu_regularization_max{1.0e-2};
  double preconditioner_ilu_regularization_selected{0.0};
  double preconditioner_ilu_regularization_absolute_max{0.0};
  std::size_t preconditioner_ilu_regularization_attempts{0};
  std::size_t preconditioner_ilu_regularization_rejected_attempts{0};
  std::size_t preconditioner_ilu_regularization_shifted_rows{0};
  std::string preconditioner_ilu_unshifted_failure_stage{"off"};
  std::size_t preconditioner_memory_budget_bytes{0};
  std::size_t preconditioner_estimated_bytes{0};
  std::size_t preconditioner_valid_blocks{0};
  std::size_t preconditioner_valid_panels{0};
  std::size_t preconditioner_ill_conditioned_blocks{0};
  std::size_t preconditioner_quality_rejected_blocks{0};
  double preconditioner_quality_probe_max_gain{0.0};
  double preconditioner_quality_probe_rel_residual{0.0};
  double preconditioner_pivot_ratio_min{0.0};
  double preconditioner_factor_growth_max{0.0};
  std::int64_t preconditioner_pivot_ratio_min_row{-1};
  std::int64_t preconditioner_pivot_ratio_min_column{-1};
  std::int64_t preconditioner_factor_growth_row{-1};
  std::int64_t preconditioner_factor_growth_column{-1};
  std::string preconditioner_quality_failure_stage{"off"};
  std::int64_t preconditioner_quality_failure_row{-1};
  std::int64_t preconditioner_quality_failure_column{-1};
  std::int64_t preconditioner_quality_failure_probe{-1};
  double preconditioner_quality_failure_probe_gain{0.0};
  double preconditioner_quality_failure_probe_rel_residual{0.0};
  std::string preconditioner_quality_status{"off"};
  double preconditioner_setup_ms{0.0};
  double preconditioner_graph_setup_ms{0.0};
  bool preconditioner_graph_cache_reused{false};
  double preconditioner_near_assembly_setup_ms{0.0};
  double preconditioner_ilu_factor_setup_ms{0.0};
  double preconditioner_apply_ms{0.0};
  std::size_t preconditioner_apply_calls{0};
  std::size_t preconditioner_sparse_nnz_blocks{0};
  double preconditioner_sparse_avg_degree{0.0};
  std::size_t preconditioner_sparse_max_degree{0};
  int preconditioner_coarse_aggregate_cap{16};
  std::size_t preconditioner_coarse_aggregates{0};
  std::size_t preconditioner_coarse_dimension{0};
  std::string preconditioner_coarse_base{"diagonal"};
  std::size_t preconditioner_coarse_matrix_bytes{0};
  double preconditioner_coarse_pivot_ratio{0.0};
  std::string preconditioner_coarse_status{"off"};
  std::size_t preconditioner_coarse_setup_matvec_calls{0};
  double preconditioner_coarse_setup_ms{0.0};
  double preconditioner_coarse_moments_setup_ms{0.0};
  double preconditioner_coarse_far_setup_ms{0.0};
  double preconditioner_coarse_audit_setup_ms{0.0};
  double preconditioner_coarse_apply_ms{0.0};
  std::size_t preconditioner_coarse_apply_calls{0};
  double preconditioner_coarse_rp_identity_error_max{0.0};
  double preconditioner_coarse_verify_rel_error_max{0.0};
  std::string surface_generator{"titan_sphere_patch"};
  std::string discretization{"centroid"};
  // "curved_dual", "curved_pn", "flat_legacy", or "n/a" when not node-patch.
  std::string node_patch_representation{"n/a"};
  // "analytic=A;averaged=B;rejected=C" node-normal sources, "n/a" otherwise.
  std::string node_patch_normal_source{"n/a"};
  // "key=value;..." PN-lift and near-pair quadrature certificate of a
  // curved node-patch surface (node_patch_certificate in tabi_solver.cpp),
  // "n/a" otherwise.
  std::string node_patch_certificate{"n/a"};
  double rhs_scale{1.0};
  double dielectric_contrast{0.0};
  double operator_contrast{0.0};
  std::size_t unknown_count{0};
  std::string bie_system{"off"};
  double lhs_diag_phi{0.0};
  double lhs_diag_phin{0.0};
  std::string energy_equation{"off"};
  std::string kernel_convention{"off"};
  double source_norm_s1{0.0};
  double source_norm_s2{0.0};
  double phi_norm{0.0};
  double phin_norm{0.0};
  double kappa_inv_a{0.0};
  double coulomb_factor_kcal_mol{0.0};
  double boundary_energy_kcal_mol{0.0};
  double reference_energy_kcal_mol{0.0};
  double selected_energy_scale{1.0};
  bool fallback_used{false};
  std::vector<double> atom_reaction_potential_kj_mol_e;
  std::vector<double> atom_reaction_energy_kj_mol;
  double atom_reaction_energy_closure_kj_mol{0.0};
  double atom_reaction_energy_closure_error_kj_mol{0.0};
  std::vector<Panel> output_panels;
  std::vector<double> output_solution;
  std::vector<double> output_rhs;
};

// The retry wrapper must never let an unaccepted attempt's physical result
// reach the normal energy/output writers.  This is header-visible solely so
// the bounded retry-state smoke can exercise the same sanitization contract
// for either failed attempt without constructing a second molecular solve.
inline Result sanitize_cartesian_retry_failure(Result result,
                                                FmmCartesianRetryTelemetry retry) {
  retry.accepted_attempt = 0;
  result.fmm_cartesian_retry = std::move(retry);
  result.status = "failed";
  result.converged = false;
  result.convergence_reason =
      "cartesian_retry_failed:" + result.fmm_cartesian_retry.final_reason;
  result.reaction_energy_source = "not_computed_cartesian_retry_failed";
  const double nan = std::numeric_limits<double>::quiet_NaN();
  result.energy_kcal_mol = nan;
  result.energy_kj_mol = nan;
  result.energy_kt = nan;
  result.boundary_energy_kcal_mol = nan;
  result.reference_energy_kcal_mol = nan;
  result.selected_energy_scale = nan;
  result.atom_reaction_energy_closure_kj_mol = nan;
  result.atom_reaction_energy_closure_error_kj_mol = nan;
  result.atom_reaction_potential_kj_mol_e.clear();
  result.atom_reaction_energy_kj_mol.clear();
  result.output_panels.clear();
  result.output_solution.clear();
  result.output_rhs.clear();
  return result;
}

struct TreecodeStats {
  double tree_build_ms{0.0};
  std::size_t tree_nodes{0};
  std::size_t tree_leaf_count{0};
  std::size_t near_interactions{0};
  std::size_t far_interactions{0};
  std::string matvec_mode{"direct"};
  int taylor_order{3};
  std::string derivative_mode{"analytic"};
  double interaction_build_ms{0.0};
  double tree_eval_ms{0.0};
  double tree_moment_ms{0.0};
  double tree_traversal_ms{0.0};
  double near_eval_ms{0.0};
  double far_eval_ms{0.0};
  double singular_near_eval_ms{0.0};
  double leaf_near_eval_ms{0.0};
  double leaf_panel_eval_ms{0.0};
  double leaf_tile_eval_ms{0.0};
  double leaf_residual_eval_ms{0.0};
  double smooth_leaf_node_eval_ms{0.0};
  std::size_t singular_near_interactions{0};
  std::size_t leaf_near_interactions{0};
  std::size_t far_node_interactions{0};
  std::size_t leaf_panel_interactions{0};
  std::size_t leaf_tile_interactions{0};
  std::size_t leaf_tile_panel_interactions{0};
  std::size_t leaf_residual_panel_interactions{0};
  std::size_t smooth_leaf_node_interactions{0};
  std::size_t leaf_pruned_interactions{0};
  std::size_t leaf_mac_candidates{0};
  std::size_t leaf_mac_accepted{0};
  std::size_t leaf_mac_rejected{0};
  std::size_t leaf_mac_far_panels{0};
  std::size_t leaf_mac_direct_panels_saved_estimate{0};
  std::string leaf_work_profile{"off"};
  std::size_t leaf_work_mac_terminal_panels{0};
  std::size_t leaf_work_physical_gap_panels{0};
  std::size_t leaf_work_fragmentation_panels{0};
  std::size_t leaf_work_convergence_panels{0};
  std::size_t leaf_work_convergence_estimator_panels{0};
  std::size_t leaf_work_convergence_rho_panels{0};
  std::size_t leaf_work_convergence_aspect_panels{0};
  std::size_t leaf_work_surface_margin_panels{0};
  std::size_t leaf_work_surface_margin_hard_panels{0};
  std::size_t leaf_work_surface_margin_soft_panels{0};
  std::size_t leaf_work_k3_rho_cap_panels{0};
  std::size_t leaf_work_singular_fallback_panels{0};
  std::size_t leaf_work_scalar_tail_panels{0};
  std::size_t leaf_work_profile_accounted_panels{0};
  double leaf_work_mac_terminal_frac{0.0};
  double leaf_work_convergence_frac{0.0};
  double leaf_work_surface_margin_frac{0.0};
  double leaf_work_singular_fallback_frac{0.0};
  double leaf_work_scalar_tail_frac{0.0};
  std::string matvec_cost_profile{"off"};
  double leaf_panels_per_second{0.0};
  double clean_leaf_panels_per_second{0.0};
  double tree_traversal_fraction{0.0};
  std::size_t adaptive_smooth_leaf_accepted{0};
  std::size_t adaptive_smooth_leaf_rejected{0};
  std::size_t adaptive_span_tile_accepted{0};
  std::size_t adaptive_span_tile_rejected{0};
  std::size_t adaptive_exact_leaf_fallbacks{0};
  std::size_t adaptive_reject_one_ring_overlap{0};
  std::size_t adaptive_reject_error_bound_failed{0};
  std::size_t adaptive_reject_p4_check_failed{0};
  std::size_t adaptive_reject_too_close{0};
  double leaf_work_reduction_ratio{0.0};
  std::size_t target_cluster_interactions{0};
  std::size_t target_cluster_targets{0};
  std::size_t target_cluster_panels_pruned{0};
  std::size_t target_cluster_rejected{0};
  double target_cluster_eval_ms{0.0};
  std::size_t leaf_pair_batch_interactions{0};
  std::size_t leaf_pair_batch_targets{0};
  std::size_t leaf_pair_batch_source_panels{0};
  std::size_t leaf_pair_batch_residual_panels{0};
  std::size_t leaf_pair_batch_fallbacks{0};
  double leaf_pair_batch_ms{0.0};
  std::string leaf_pair_batch_kernel{"scalar"};
  std::size_t leaf_pair_batch_vectorized_sources{0};
  std::size_t leaf_pair_batch_vector_lanes{0};
  double leaf_pair_batch_pairs_per_second{0.0};
  std::string leaf_kernel_profile{"off"};
  std::string jcp_direct_kernel{"baseline"};
  std::size_t leaf_kernel_scalar_tasks{0};
  std::size_t leaf_kernel_batch_tasks{0};
  std::size_t leaf_kernel_source_panel_evals{0};
  double leaf_kernel_scalar_ms{0.0};
  double leaf_kernel_target_avx2_ms{0.0};
  double leaf_kernel_tail_ms{0.0};
  std::size_t leaf_pair_batch_soa_sources{0};
  double leaf_pair_batch_soa_ms{0.0};
  double leaf_pair_batch_scalar_tail_ms{0.0};
  double leaf_pair_batch_clean_source_panels_per_second{0.0};
  std::string leaf_pair_selection_profile{"off"};
  std::string leaf_pair_probe_cache{"off"};
  std::string leaf_pair_target_group{"scan"};
  int leaf_pair_max_lookahead{32};
  std::size_t leaf_pair_selection_candidate_scans{0};
  std::size_t leaf_pair_selection_clean_rejects{0};
  std::size_t leaf_pair_selection_probe_calls{0};
  std::size_t leaf_pair_selection_probe_node_visits{0};
  double leaf_pair_selection_probe_ms{0.0};
  double leaf_pair_selection_ms{0.0};
  std::size_t leaf_pair_selection_probe_cache_hits{0};
  std::size_t leaf_pair_selection_probe_cache_misses{0};
  std::size_t leaf_pair_selection_insufficient_targets{0};
  std::size_t leaf_pair_selection_same_target_leaf_rejects{0};
  std::string leaf_pair_directness_probe{"full-traversal"};
  std::string leaf_pair_directness_cache{"off"};
  std::string leaf_pair_directness_verify{"off"};
  std::size_t leaf_pair_directness_ancestor_probe_calls{0};
  std::size_t leaf_pair_directness_ancestor_probe_node_visits{0};
  double leaf_pair_directness_ancestor_probe_ms{0.0};
  std::size_t leaf_pair_directness_pair_cache_hits{0};
  std::size_t leaf_pair_directness_pair_cache_misses{0};
  std::size_t leaf_pair_directness_all_direct_proofs{0};
  std::size_t leaf_pair_directness_mixed_fallbacks{0};
  std::size_t leaf_pair_directness_verify_checks{0};
  std::size_t leaf_pair_directness_verify_mismatches{0};
  std::string leaf_pair_scheduler{"inline-select"};
  int leaf_pair_worklist_max_active{2048};
  std::string leaf_pair_worklist_verify{"off"};
  std::size_t leaf_pair_worklist_tasks{0};
  std::size_t leaf_pair_worklist_groups{0};
  std::size_t leaf_pair_worklist_batched_groups{0};
  std::size_t leaf_pair_worklist_scalar_flush_tasks{0};
  std::size_t leaf_pair_worklist_forced_flushes{0};
  std::size_t leaf_pair_worklist_active_groups_max{0};
  double leaf_pair_worklist_ms{0.0};
  std::size_t leaf_pair_worklist_verify_checks{0};
  std::size_t leaf_pair_worklist_verify_mismatches{0};
  std::string leaf_block_verify{"off"};
  std::string leaf_mac{"off"};
  std::string fmm_m2l_mode{"off"};
  int fmm_m2l_expansion_order{0};
  std::size_t fmm_m2l_near_pairs{0};
  std::size_t fmm_m2l_far_pairs{0};
  double fmm_m2l_setup_ms{0.0};
  double fmm_m2l_apply_ms{0.0};
  bool fmm_m2l_finite_outputs{true};
  std::size_t leaf_block_target_leaves{0};
  std::size_t leaf_block_targets{0};
  std::size_t leaf_block_source_groups{0};
  std::size_t leaf_block_batched_groups{0};
  std::size_t leaf_block_scalar_tail_tasks{0};
  double leaf_block_group_ms{0.0};
  double leaf_block_flush_ms{0.0};
  double leaf_block_writeback_ms{0.0};
  std::size_t leaf_block_verify_checks{0};
  std::size_t leaf_block_verify_mismatches{0};
  std::string adaptive_level{"strict"};
  std::string cluster_policy{"off"};
  int target_batch_size{1};
  int leaf_pair_min_source_panels{64};
  int leaf_pair_min_targets{4};
  std::string leaf_near_policy{"exact"};
  std::string leaf_panel_kernel{"scalar"};
  std::string interaction_mode{"cached"};
  std::string node_bounds_mode{"moment"};
  std::string tree_builder{"grouped-morton"};
  std::string expansion_center{"legacy"};
  std::string fragmentation_safety{"off"};
  std::string mac_safety{"off"};
  double mac_gap_factor{1.5};
  std::size_t tree_depth_max{0};
  std::size_t mac_reject_radius_limited{0};
  double mac_accept_rate{0.0};
  double node_radius_max{0.0};
  double node_radius_mean{0.0};
  double node_aspect_ratio_max{0.0};
  double node_aspect_ratio_mean{0.0};
  double leaf_panel_work_per_matvec{0.0};
  double far_node_work_per_matvec{0.0};
  std::size_t far_geometric_unsafe_count{0};
  std::size_t far_geometric_unsafe_panels{0};
  double far_geometric_min_gap{0.0};
  double far_geometric_gap_ratio_min{0.0};
  double far_geometric_gap_ratio_mean{0.0};
  std::size_t far_geometric_audit_samples{0};
  std::size_t mac_physical_gap_rejects{0};
  std::size_t mac_convergence_rho_rejects{0};
  std::size_t mac_convergence_aspect_rejects{0};
  std::size_t mac_convergence_rho_reject_panels{0};
  std::size_t mac_convergence_aspect_reject_panels{0};
  std::size_t mac_surface_margin_rejects{0};
  std::size_t mac_surface_margin_reject_panels{0};
  std::size_t mac_surface_margin_applied{0};
  std::size_t mac_surface_margin_skipped{0};
  std::size_t mac_surface_margin_adaptive_rejects{0};
  std::size_t mac_surface_margin_adaptive_reject_panels{0};
  std::size_t mac_surface_margin_skipped_accepted_far{0};
  std::size_t mac_surface_margin_hard_applied{0};
  std::size_t mac_surface_margin_soft_applied{0};
  std::size_t mac_surface_margin_hard_rejects{0};
  std::size_t mac_surface_margin_hard_reject_panels{0};
  std::size_t mac_surface_margin_soft_rejects{0};
  std::size_t mac_surface_margin_soft_reject_panels{0};
  std::size_t fragmented_far_count{0};
  std::size_t fragmented_far_panels{0};
  std::size_t fragmented_split_rejects{0};
  double fragmentation_center_gap_ratio_max{0.0};
  double fragmentation_center_gap_ratio_mean{0.0};
  std::string moment_audit{"off"};
  std::string moment_construction{"translated"};
  std::string moment_sum{"naive"};
  std::string convergence_audit{"off"};
  std::string convergence_safety{"off"};
  double convergence_rho_cap{0.0};
  double mac_aspect_cap{0.0};
  std::string convergence_estimator{"off"};
  double convergence_tail_threshold{1.0e4};
  double convergence_estimator_error_cap{1.0e-3};
  double convergence_unstable_rho_cap{0.25};
  std::string convergence_calibration{"stage3j"};
  std::string convergence_estimator_model{"stage3l"};
  std::string convergence_channel_calibration{"stage3m"};
  std::string convergence_split_policy{"global"};
  std::string convergence_split_selectivity{"legacy"};
  double convergence_split_absolute_floor{0.0};
  std::string convergence_overreject_audit{"off"};
  std::string convergence_overreject_detail{"off"};
  std::string convergence_shape_audit{"off"};
  std::string convergence_kernel_response_audit{"off"};
  std::string channel_missed_semantics{"legacy"};
  std::string accepted_miss_diagnostics{"off"};
  double accepted_miss_absolute_floor{1.0e-7};
  double convergence_k3_accepted_rho_cap{0.0};
  std::string mac_distance_policy{"center"};
  double mac_surface_margin_factor{0.25};
  std::string mac_surface_margin_policy{"all"};
  double mac_surface_margin_aspect_threshold{3.0};
  double mac_surface_margin_rho_trigger{0.55};
  double mac_surface_margin_soft_aspect_threshold{0.0};
  double mac_surface_margin_soft_rho_trigger{0.0};
  double mac_surface_margin_soft_factor{0.10};
  std::size_t convergence_estimator_samples{0};
  std::size_t convergence_estimator_unstable_far_count{0};
  std::size_t convergence_estimator_unstable_far_panels{0};
  std::size_t convergence_estimator_split_rejects{0};
  std::size_t convergence_estimator_split_reject_panels{0};
  std::size_t convergence_low_abs_reaccepted_far_count{0};
  std::size_t convergence_low_abs_reaccepted_far_panels{0};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_low_abs_reaccepted_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_low_abs_reaccepted_panels{};
  double convergence_estimator_tail_ratio_max{0.0};
  double convergence_estimator_tail_ratio_mean{0.0};
  double convergence_estimator_error_max{0.0};
  double convergence_estimator_error_mean{0.0};
  double convergence_estimator_abs_error_max{0.0};
  double convergence_estimator_abs_error_mean{0.0};
  double convergence_tail_norm_p_max{0.0};
  double convergence_tail_norm_p_mean{0.0};
  double convergence_tail_norm_pm1_max{0.0};
  double convergence_tail_norm_pm1_mean{0.0};
  double convergence_effective_jcp_amp_max{0.0};
  double convergence_effective_jcp_amp_mean{0.0};
  std::string convergence_dielectric_amp{"off"};
  double convergence_dielectric_amp_max{0.0};
  double convergence_dielectric_amp_mean{0.0};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_estimated_error_max{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_estimated_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_effective_jcp_amp_max{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_effective_jcp_amp_mean{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_unstable_far_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_unstable_far_panels{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_split_rejects{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_split_reject_panels{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_high_error_caught{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_high_error_missed{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_high_error_caught_by_split{};
  std::size_t convergence_estimator_high_error_caught{0};
  std::size_t convergence_estimator_high_error_missed{0};
  std::size_t convergence_estimator_high_error_caught_by_split{0};
  std::size_t convergence_estimator_phantom_high_error_count{0};
  double convergence_estimator_phantom_high_error_abs_max{0.0};
  std::size_t convergence_overreject_count{0};
  std::size_t convergence_overreject_panels{0};
  std::size_t convergence_overreject_caught_count{0};
  std::size_t convergence_overreject_caught_panels{0};
  double convergence_overreject_abs_error_max{0.0};
  double convergence_overreject_abs_error_mean{0.0};
  double convergence_overreject_abs_error_p95{0.0};
  double convergence_overreject_abs_error_p99{0.0};
  double convergence_overreject_estimated_abs_error_max{0.0};
  double convergence_overreject_estimated_abs_error_mean{0.0};
  double convergence_overreject_required_scale_max{0.0};
  double convergence_overreject_required_scale_p95{0.0};
  double convergence_overreject_required_scale_p99{0.0};
  std::vector<double> convergence_overreject_abs_error_samples;
  std::vector<double> convergence_overreject_required_scale_samples;
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_overreject_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_overreject_caught_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_overreject_panels{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_overreject_abs_error_max{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_overreject_abs_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_overreject_abs_error_sum{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_overreject_required_scale_max{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_overreject_required_scale_p99{};
  std::array<std::vector<double>, kJcpExpansionChannelCount>
      convergence_channel_overreject_required_scale_samples;
  std::array<std::size_t, kJcpExpansionChannelCount>
      phantom_high_error_channel_count{};
  std::array<double, kJcpExpansionChannelCount>
      phantom_high_error_channel_abs_max{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      accepted_miss_channel_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      accepted_miss_channel_panels{};
  std::array<double, kJcpExpansionChannelCount> accepted_miss_channel_rho_min{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_rho_mean{};
  std::array<double, kJcpExpansionChannelCount> accepted_miss_channel_rho_max{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_tail_ratio_mean{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_tail_ratio_max{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_estimated_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_estimated_error_max{};
  std::string accepted_miss_worst_channel{"none"};
  std::size_t convergence_k3_rho_cap_rejects{0};
  std::size_t convergence_k3_rho_cap_reject_panels{0};
  std::size_t moment_audit_nodes{0};
  double moment_translation_rel_error_max{0.0};
  double moment_translation_rel_error_mean{0.0};
  double moment_translation_offset_max{0.0};
  double moment_translation_offset_mean{0.0};
  double moment_cancellation_ratio_max{0.0};
  double moment_cancellation_ratio_mean{0.0};
  double moment_dynamic_range_max{0.0};
  std::string expansion_audit{"off"};
  std::string expansion_audit_channel{"off"};
  std::size_t far_expansion_audit_pairs{0};
  std::size_t far_expansion_audit_panels{0};
  double far_expansion_direct_ms{0.0};
  double far_expansion_rel_error_max{0.0};
  double far_expansion_rel_error_mean{0.0};
  double far_expansion_rel_error_p95{0.0};
  double far_expansion_rel_error_p99{0.0};
  std::array<double, kJcpExpansionChannelCount>
      far_expansion_channel_rel_error_max{};
  std::array<double, kJcpExpansionChannelCount>
      far_expansion_channel_rel_error_mean{};
  std::array<double, kJcpExpansionChannelCount>
      far_expansion_channel_rel_error_p99{};
  std::string far_expansion_worst_channel{"none"};
  std::vector<double> far_expansion_rel_error_samples;
  std::array<std::vector<double>, kJcpExpansionChannelCount>
      far_expansion_channel_rel_error_samples;
  bool far_expansion_mean_gate_pass{false};
  double far_convergence_rho_max{0.0};
  double far_convergence_rho_mean{0.0};
  double accepted_far_aspect_ratio_max{0.0};
  double accepted_far_aspect_ratio_mean{0.0};
  std::string leaf_tile_status{"off"};
  double leaf_panel_pairs_per_second{0.0};
  std::size_t matvec_calls{0};
  std::string near_integration{"centroid"};
  std::string quad_buffer{"off"};
  std::size_t near_self_pairs{0};
  std::size_t near_one_ring_pairs{0};
  std::size_t near_duffy_pairs{0};
  std::size_t near_centroid_pairs{0};
  std::size_t near_fallbacks{0};
  std::size_t near_geometric_buffer_pairs{0};
  std::size_t near_geometric_buffer_gauss_pairs{0};
  std::size_t near_geometric_buffer_adaptive_pairs{0};
  std::size_t near_geometric_buffer_centroid_pairs{0};
  double near_geometric_buffer_min_gap{0.0};
  double near_geometric_buffer_mean_gap_ratio{0.0};
  double near_geometric_buffer_max_edge_mean{0.0};
  double near_integration_ms{0.0};
  std::size_t duffy_subdivisions_max{0};
  double duffy_subdivisions_mean{0.0};
  std::string near_kernel_policy{"centroid"};
  std::size_t near_k1_accepted{0};
  std::size_t near_k1_rejected{0};
  std::size_t near_k2_accepted{0};
  std::size_t near_k2_rejected{0};
  std::size_t near_k3_accepted{0};
  std::size_t near_k3_rejected{0};
  std::size_t near_k4_accepted{0};
  std::size_t near_k4_rejected{0};
  std::size_t near_one_ring_k1_accepted{0};
  std::size_t near_one_ring_k1_rejected{0};
  std::size_t near_one_ring_k2_accepted{0};
  std::size_t near_one_ring_k2_rejected{0};
  std::size_t near_one_ring_k3_accepted{0};
  std::size_t near_one_ring_k3_rejected{0};
  std::size_t near_one_ring_k4_accepted{0};
  std::size_t near_one_ring_k4_rejected{0};
  std::string near_k1_policy{"centroid"};
  std::string near_k2_policy{"centroid"};
  std::string near_k3_policy{"centroid"};
  std::string near_k4_policy{"centroid"};
  NearBlockAuditBuckets near_block_audit{};
  GeometricNearAuditBuckets geometric_near_audit{};
};

inline void merge_near_kernel_block_stats(TreecodeStats &dst,
                                          const TreecodeStats &src) noexcept {
  auto merge_geometric_means = [](std::size_t old_count, std::size_t add_count,
                                  double old_value, double add_value) noexcept {
    const std::size_t total = old_count + add_count;
    if (total == 0)
      return 0.0;
    return (old_value * static_cast<double>(old_count) +
            add_value * static_cast<double>(add_count)) /
           static_cast<double>(total);
  };
  const std::size_t old_geometric_count = dst.near_geometric_buffer_pairs;
  dst.near_geometric_buffer_pairs += src.near_geometric_buffer_pairs;
  dst.near_geometric_buffer_gauss_pairs +=
      src.near_geometric_buffer_gauss_pairs;
  dst.near_geometric_buffer_adaptive_pairs +=
      src.near_geometric_buffer_adaptive_pairs;
  dst.near_geometric_buffer_centroid_pairs +=
      src.near_geometric_buffer_centroid_pairs;
  if (src.near_geometric_buffer_pairs > 0) {
    if (old_geometric_count == 0 || dst.near_geometric_buffer_min_gap <= 0.0) {
      dst.near_geometric_buffer_min_gap = src.near_geometric_buffer_min_gap;
    } else if (src.near_geometric_buffer_min_gap > 0.0) {
      dst.near_geometric_buffer_min_gap = std::min(
          dst.near_geometric_buffer_min_gap, src.near_geometric_buffer_min_gap);
    }
    dst.near_geometric_buffer_mean_gap_ratio = merge_geometric_means(
        old_geometric_count, src.near_geometric_buffer_pairs,
        dst.near_geometric_buffer_mean_gap_ratio,
        src.near_geometric_buffer_mean_gap_ratio);
    dst.near_geometric_buffer_max_edge_mean = merge_geometric_means(
        old_geometric_count, src.near_geometric_buffer_pairs,
        dst.near_geometric_buffer_max_edge_mean,
        src.near_geometric_buffer_max_edge_mean);
  }
  dst.near_k1_accepted += src.near_k1_accepted;
  dst.near_k1_rejected += src.near_k1_rejected;
  dst.near_k2_accepted += src.near_k2_accepted;
  dst.near_k2_rejected += src.near_k2_rejected;
  dst.near_k3_accepted += src.near_k3_accepted;
  dst.near_k3_rejected += src.near_k3_rejected;
  dst.near_k4_accepted += src.near_k4_accepted;
  dst.near_k4_rejected += src.near_k4_rejected;
  dst.near_one_ring_k1_accepted += src.near_one_ring_k1_accepted;
  dst.near_one_ring_k1_rejected += src.near_one_ring_k1_rejected;
  dst.near_one_ring_k2_accepted += src.near_one_ring_k2_accepted;
  dst.near_one_ring_k2_rejected += src.near_one_ring_k2_rejected;
  dst.near_one_ring_k3_accepted += src.near_one_ring_k3_accepted;
  dst.near_one_ring_k3_rejected += src.near_one_ring_k3_rejected;
  dst.near_one_ring_k4_accepted += src.near_one_ring_k4_accepted;
  dst.near_one_ring_k4_rejected += src.near_one_ring_k4_rejected;
  auto merge_policy = [](std::size_t accepted, std::size_t rejected,
                         std::string &dst_policy,
                         const std::string &src_policy) noexcept {
    if (accepted > 0 && rejected > 0) {
      dst_policy = "mixed_guarded";
    } else if (accepted > 0 && !src_policy.empty() &&
               src_policy != "centroid") {
      dst_policy = src_policy;
    } else if (rejected > 0 && !src_policy.empty() &&
               src_policy != "centroid") {
      dst_policy = src_policy;
    } else if (rejected > 0) {
      dst_policy = "centroid_guarded";
    }
  };
  merge_policy(dst.near_k1_accepted, dst.near_k1_rejected, dst.near_k1_policy,
               src.near_k1_policy);
  merge_policy(dst.near_k2_accepted, dst.near_k2_rejected, dst.near_k2_policy,
               src.near_k2_policy);
  merge_policy(dst.near_k3_accepted, dst.near_k3_rejected, dst.near_k3_policy,
               src.near_k3_policy);
  merge_policy(dst.near_k4_accepted, dst.near_k4_rejected, dst.near_k4_policy,
               src.near_k4_policy);
  for (std::size_t i = 0; i < dst.near_block_audit.size(); ++i) {
    auto &d = dst.near_block_audit[i];
    const auto &s = src.near_block_audit[i];
    d.accepted += s.accepted;
    d.rejected += s.rejected;
    d.candidate_worse_than_centroid += s.candidate_worse_than_centroid;
    d.reference_nonconverged += s.reference_nonconverged;
    d.orientation_sign_mismatch += s.orientation_sign_mismatch;
    d.strict_ref_reject += s.strict_ref_reject;
    if (s.max_candidate_ref_rel > d.max_candidate_ref_rel)
      d.max_candidate_ref_rel = s.max_candidate_ref_rel;
    d.sum_candidate_ref_rel += s.sum_candidate_ref_rel;
    if (s.max_centroid_ref_rel > d.max_centroid_ref_rel)
      d.max_centroid_ref_rel = s.max_centroid_ref_rel;
    d.k2_laplace_sign_ok += s.k2_laplace_sign_ok;
    if (s.max_k2_yukawa_smooth_ref_rel > d.max_k2_yukawa_smooth_ref_rel)
      d.max_k2_yukawa_smooth_ref_rel = s.max_k2_yukawa_smooth_ref_rel;
    if (s.max_k2_orientation_pair_ref_rel > d.max_k2_orientation_pair_ref_rel)
      d.max_k2_orientation_pair_ref_rel = s.max_k2_orientation_pair_ref_rel;
    d.sample_count += s.sample_count;
  }
  for (std::size_t i = 0; i < dst.geometric_near_audit.size(); ++i) {
    auto &d = dst.geometric_near_audit[i];
    const auto &s = src.geometric_near_audit[i];
    if (s.pairs == 0)
      continue;
    if (d.pairs == 0 || d.min_gap <= 0.0)
      d.min_gap = s.min_gap;
    else if (s.min_gap > 0.0)
      d.min_gap = std::min(d.min_gap, s.min_gap);
    d.max_gap_ratio = std::max(d.max_gap_ratio, s.max_gap_ratio);
    d.sum_gap_ratio += s.sum_gap_ratio;
    d.sum_center_ratio += s.sum_center_ratio;
    d.pairs += s.pairs;
  }
}

struct TreeNode {
  double cx{0.0};
  double cy{0.0};
  double cz{0.0};
  double radius{0.0};
  double area_sum{0.0};
  double min_x{0.0};
  double min_y{0.0};
  double min_z{0.0};
  double max_x{0.0};
  double max_y{0.0};
  double max_z{0.0};
  double max_edge_length{0.0};
  std::size_t first_child{0};
  std::size_t child_count{0};
  std::size_t first_panel{0};
  std::size_t panel_count{0};
  std::size_t depth{0};
  bool fragmented{false};
  double fragmentation_center_gap_ratio{0.0};
};

struct TreeInteractionLists {
  std::vector<std::size_t> singular_offsets;
  std::vector<std::size_t> singular_indices;
  std::vector<std::size_t> leaf_offsets;
  std::vector<std::size_t> leaf_indices;
  std::vector<std::size_t> leaf_tile_offsets;
  std::vector<std::size_t> leaf_tile_node_indices;
  std::vector<std::size_t> leaf_tile_exclusion_offsets;
  std::vector<std::size_t> leaf_tile_exclusion_indices;
  std::vector<std::size_t> leaf_tile_span_offsets;
  std::vector<std::size_t> leaf_tile_span_firsts;
  std::vector<std::size_t> leaf_tile_span_counts;
  std::vector<std::size_t> smooth_leaf_offsets;
  std::vector<std::size_t> smooth_leaf_node_indices;
  std::vector<std::size_t> far_offsets;
  std::vector<std::size_t> far_node_indices;
  std::size_t singular_pair_count{0};
  std::size_t leaf_pair_count{0};
  std::size_t leaf_tile_node_count{0};
  std::size_t leaf_tile_panel_count{0};
  std::size_t smooth_leaf_node_count{0};
  std::size_t leaf_pruned_count{0};
  std::size_t leaf_mac_candidates{0};
  std::size_t leaf_mac_accepted{0};
  std::size_t leaf_mac_rejected{0};
  std::size_t leaf_mac_far_panels{0};
  std::size_t leaf_mac_direct_panels_saved_estimate{0};
  std::size_t adaptive_smooth_leaf_accepted{0};
  std::size_t adaptive_smooth_leaf_rejected{0};
  std::size_t adaptive_span_tile_accepted{0};
  std::size_t adaptive_span_tile_rejected{0};
  std::size_t adaptive_exact_leaf_fallbacks{0};
  std::size_t adaptive_reject_one_ring_overlap{0};
  std::size_t adaptive_reject_error_bound_failed{0};
  std::size_t adaptive_reject_p4_check_failed{0};
  std::size_t adaptive_reject_too_close{0};
  std::size_t target_cluster_interactions{0};
  std::size_t target_cluster_targets{0};
  std::size_t target_cluster_panels_pruned{0};
  std::size_t target_cluster_rejected{0};
  std::size_t far_node_count{0};
  std::size_t mac_reject_radius_limited{0};
  std::size_t far_geometric_unsafe_count{0};
  std::size_t far_geometric_unsafe_panels{0};
  double far_geometric_min_gap{0.0};
  double far_geometric_gap_ratio_min{0.0};
  double far_geometric_gap_ratio_sum{0.0};
  std::size_t far_geometric_audit_samples{0};
  std::size_t mac_physical_gap_rejects{0};
  std::size_t mac_convergence_rho_rejects{0};
  std::size_t mac_convergence_aspect_rejects{0};
  std::size_t mac_convergence_rho_reject_panels{0};
  std::size_t mac_convergence_aspect_reject_panels{0};
  std::size_t mac_surface_margin_rejects{0};
  std::size_t mac_surface_margin_reject_panels{0};
  std::size_t fragmented_far_count{0};
  std::size_t fragmented_far_panels{0};
  std::size_t fragmented_split_rejects{0};
  double fragmentation_center_gap_ratio_sum{0.0};
  double fragmentation_center_gap_ratio_max{0.0};
  double build_ms{0.0};

  [[nodiscard]] bool valid_for(std::size_t panel_count) const noexcept {
    return singular_offsets.size() == panel_count + 1 &&
           leaf_offsets.size() == panel_count + 1 &&
           leaf_tile_offsets.size() == panel_count + 1 &&
           leaf_tile_exclusion_offsets.size() ==
               leaf_tile_node_indices.size() + 1 &&
           leaf_tile_span_offsets.size() == leaf_tile_node_indices.size() + 1 &&
           smooth_leaf_offsets.size() == panel_count + 1 &&
           far_offsets.size() == panel_count + 1;
  }
};

struct TreePanelCache {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> nx;
  std::vector<double> ny;
  std::vector<double> nz;
  std::vector<double> area;
  AlignedDoubleVector ordered_x;
  AlignedDoubleVector ordered_y;
  AlignedDoubleVector ordered_z;
  AlignedDoubleVector ordered_nx;
  AlignedDoubleVector ordered_ny;
  AlignedDoubleVector ordered_nz;
  AlignedDoubleVector ordered_area;
  std::vector<std::size_t> ordered_original_index;

  [[nodiscard]] bool valid_for(std::size_t panel_count) const noexcept {
    return x.size() == panel_count && y.size() == panel_count &&
           z.size() == panel_count && nx.size() == panel_count &&
           ny.size() == panel_count && nz.size() == panel_count &&
           area.size() == panel_count && ordered_x.size() == panel_count &&
           ordered_y.size() == panel_count && ordered_z.size() == panel_count &&
           ordered_nx.size() == panel_count &&
           ordered_ny.size() == panel_count &&
           ordered_nz.size() == panel_count &&
           ordered_area.size() == panel_count &&
           ordered_original_index.size() == panel_count;
  }
};

struct FmmM2lSingleMatvecWorkspace {
  std::vector<double> phi;
  std::vector<double> phin;
  std::vector<double> interaction_phi;
  std::vector<double> interaction_phin;
};

struct TreeWorkspace {
  std::vector<std::size_t> order;
  std::vector<std::uint64_t> morton;
  std::vector<TreeNode> nodes;
  std::vector<std::size_t> child_indices;
  std::size_t root_index{0};
  std::size_t leaf_count{0};
  std::size_t depth_max{0};
  double build_ms{0.0};
  double node_radius_max{0.0};
  double node_radius_mean{0.0};
  double node_aspect_ratio_max{0.0};
  double node_aspect_ratio_mean{0.0};
  std::size_t leaf_size{32};
  double opening_theta{0.45};
  int taylor_order{3};
  std::string derivative_mode{"analytic"};
  std::string leaf_near_policy{"exact"};
  std::string leaf_panel_kernel{"auto"};
  std::string adaptive_level{"strict"};
  std::string cluster_policy{"off"};
  std::string leaf_pair_kernel{"auto"};
  std::string leaf_kernel_profile{"off"};
  std::string leaf_work_profile{"off"};
  std::string matvec_cost_profile{"off"};
  std::string jcp_direct_kernel{"baseline"};
  std::string leaf_pair_selection_profile{"off"};
  std::string leaf_pair_probe_cache{"off"};
  std::string leaf_pair_target_group{"scan"};
  std::string leaf_pair_directness_probe{"full-traversal"};
  std::string leaf_pair_directness_cache{"off"};
  std::string leaf_pair_directness_verify{"off"};
  std::string leaf_pair_scheduler{"inline-select"};
  std::size_t leaf_pair_worklist_max_active{2048};
  std::string leaf_pair_worklist_verify{"off"};
  std::string leaf_block_verify{"off"};
  std::string leaf_mac{"off"};
  std::string fmm_m2l{"off"};
  mutable std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext>
      fmm_m2l_context;
  mutable std::shared_ptr<FmmM2lSingleMatvecWorkspace> fmm_m2l_workspace;
  std::size_t target_batch_size{1};
  std::size_t leaf_pair_min_source_panels{64};
  std::size_t leaf_pair_min_targets{4};
  std::size_t leaf_pair_max_lookahead{32};
  std::string interaction_mode{"cached"};
  std::string node_bounds_mode{"moment"};
  std::string tree_builder{"grouped-morton"};
  std::string quad_buffer{"off"};
  std::string mac_safety{"off"};
  double mac_gap_factor{1.5};
  std::string moment_audit{"off"};
  std::string moment_construction{"translated"};
  std::string moment_sum{"naive"};
  std::string convergence_audit{"off"};
  std::string convergence_safety{"off"};
  double convergence_rho_cap{0.0};
  double mac_aspect_cap{0.0};
  std::string convergence_estimator{"off"};
  double convergence_tail_threshold{1.0e4};
  double convergence_estimator_error_cap{1.0e-3};
  double convergence_unstable_rho_cap{0.25};
  std::string convergence_calibration{"stage3j"};
  std::string convergence_dielectric_amp{"off"};
  std::string convergence_estimator_model{"stage3l"};
  std::string convergence_channel_calibration{"stage3m"};
  std::string convergence_split_policy{"global"};
  std::string convergence_split_selectivity{"legacy"};
  double convergence_split_absolute_floor{0.0};
  std::string convergence_overreject_audit{"off"};
  std::string convergence_overreject_detail{"off"};
  std::string convergence_shape_audit{"off"};
  std::string convergence_kernel_response_audit{"off"};
  std::string channel_missed_semantics{"legacy"};
  std::string accepted_miss_diagnostics{"off"};
  double accepted_miss_absolute_floor{1.0e-7};
  double convergence_k3_accepted_rho_cap{0.0};
  std::string mac_distance_policy{"center"};
  double mac_surface_margin_factor{0.25};
  std::string mac_surface_margin_policy{"all"};
  double mac_surface_margin_aspect_threshold{3.0};
  double mac_surface_margin_rho_trigger{0.55};
  double mac_surface_margin_soft_aspect_threshold{0.0};
  double mac_surface_margin_soft_rho_trigger{0.0};
  double mac_surface_margin_soft_factor{0.10};
  std::string expansion_center{"legacy"};
  std::string fragmentation_safety{"off"};
  std::string expansion_audit{"off"};
  std::string expansion_audit_channel{"off"};
  std::size_t expansion_audit_sample_count{512};
  std::size_t expansion_audit_max_panels{2048};
  TreePanelCache panel_cache;
  TreeInteractionLists interactions;
  std::vector<std::pair<std::size_t, std::size_t>> moment_levels;
  std::vector<std::size_t> target_leaf_index_by_panel;
  std::vector<std::vector<std::size_t>> target_panels_by_leaf;
  std::vector<std::size_t> parent_index_by_node;
  std::vector<std::vector<std::size_t>> leaf_ancestor_path_by_node;
};

std::vector<Panel> build_surface(const std::vector<Atom> &atoms,
                                 const Options &options,
                                 Result *result = nullptr,
                                 TitanExecutor *executor = nullptr);

void apply_direct_operator(const std::vector<Panel> &panels,
                           const std::vector<double> &density, double contrast,
                           double kappa, TitanExecutor &executor,
                           std::vector<double> &out,
                           TreecodeStats *stats = nullptr);

TreeWorkspace build_tree_workspace(
    const std::vector<Panel> &panels, TitanExecutor &executor,
    std::size_t leaf_size = 32, double opening_theta = 0.45,
    const std::string &leaf_near_policy = "auto",
    const std::string &leaf_panel_kernel = "auto",
    const std::string &adaptive_level = "strict",
    const std::string &quad_buffer = "off",
    const std::string &interaction_mode = "cached",
    const std::string &node_bounds_mode = "moment",
    const std::string &tree_builder = "grouped-morton",
    const std::string &mac_safety = "off", double mac_gap_factor = 1.5,
    const std::string &moment_audit = "off",
    const std::string &expansion_center = "legacy",
    const std::string &fragmentation_safety = "off",
    const std::string &expansion_audit = "off",
    int expansion_audit_sample_count = 512,
    int expansion_audit_max_panels = 2048,
    const std::string &moment_construction = "translated",
    const std::string &moment_sum = "naive",
    const std::string &convergence_audit = "off",
    const std::string &convergence_safety = "off",
    double convergence_rho_cap = 0.0, double mac_aspect_cap = 0.0,
    const std::string &convergence_estimator = "off",
    double convergence_tail_threshold = 1.0e4,
    double convergence_estimator_error_cap = 1.0e-3,
    double convergence_unstable_rho_cap = 0.25,
    const std::string &convergence_calibration = "stage3j",
    const std::string &cluster_policy = "off",
    const std::string &convergence_dielectric_amp = "off",
    int target_batch_size = 1, int leaf_pair_min_source_panels = 64,
    int leaf_pair_min_targets = 4,
    const std::string &convergence_estimator_model = "stage3l",
    const std::string &expansion_audit_channel = "off",
    const std::string &convergence_channel_calibration = "stage3m",
    const std::string &channel_missed_semantics = "legacy",
    const std::string &convergence_split_policy = "global",
    const std::string &accepted_miss_diagnostics = "off",
    double convergence_k3_accepted_rho_cap = 0.0,
    const std::string &mac_distance_policy = "center",
    double mac_surface_margin_factor = 0.25,
    const std::string &mac_surface_margin_policy = "all",
    double mac_surface_margin_aspect_threshold = 3.0,
    double mac_surface_margin_rho_trigger = 0.55,
    double accepted_miss_absolute_floor = 1.0e-7,
    const std::string &leaf_pair_kernel = "auto",
    const std::string &leaf_kernel_profile = "off",
    const std::string &leaf_work_profile = "off",
    const std::string &jcp_direct_kernel = "baseline",
    const std::string &leaf_pair_selection_profile = "off",
    const std::string &leaf_pair_probe_cache = "off",
    const std::string &leaf_pair_target_group = "scan",
    int leaf_pair_max_lookahead = 32,
    const std::string &leaf_pair_directness_probe = "full-traversal",
    const std::string &leaf_pair_directness_cache = "off",
    const std::string &leaf_pair_directness_verify = "off",
    const std::string &leaf_pair_scheduler = "inline-select",
    int leaf_pair_worklist_max_active = 2048,
    const std::string &leaf_pair_worklist_verify = "off",
    const std::string &leaf_block_verify = "off",
    double mac_surface_margin_soft_aspect_threshold = 0.0,
    double mac_surface_margin_soft_rho_trigger = 0.0,
    double mac_surface_margin_soft_factor = 0.10,
    const std::string &convergence_split_selectivity = "legacy",
    double convergence_split_absolute_floor = 0.0,
    const std::string &convergence_overreject_audit = "off",
    const std::string &convergence_overreject_detail = "off",
    const std::string &convergence_shape_audit = "off",
    const std::string &convergence_kernel_response_audit = "off",
    const std::string &leaf_mac = "off", const std::string &fmm_m2l = "off");

struct FmmM2lSingleMatvecStats {
  std::size_t near_pairs{0};
  std::size_t far_pairs{0};
  std::size_t near_self_pairs{0};
  std::size_t near_one_ring_pairs{0};
  std::size_t near_duffy_pairs{0};
  std::size_t near_centroid_pairs{0};
  std::size_t near_fallbacks{0};
  std::size_t duffy_subdivisions_max{0};
  double duffy_subdivisions_mean{0.0};
  std::string near_kernel_policy{"centroid"};
  int expansion_order{0};
  double setup_ms{0.0};
  double apply_ms{0.0};
  bool cached{false};
  std::size_t context_nodes{0};
  std::size_t context_work_items{0};
  bool finite_outputs{true};
};

std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext>
ensure_jcp_fmm_m2l_context(
    const std::vector<Panel> &panels, double eps_ratio, double kappa,
    std::size_t leaf_size, double opening_theta, int expansion_order,
    TitanExecutor &executor,
    std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext>
        *context_cache = nullptr);

void apply_jcp_fmm_m2l_single_matvec(
    const std::vector<Panel> &panels, const std::vector<double> &unknowns,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    std::size_t leaf_size, double opening_theta, int expansion_order,
    TitanExecutor &executor, std::vector<double> &out,
    FmmM2lSingleMatvecStats *stats = nullptr, bool coupled = true,
    std::shared_ptr<titan::tabi::fmm_m2l::FmmM2lAnalyticContext>
        *context_cache = nullptr,
    const std::string &near_integration = "centroid",
    const std::string &quad_buffer = "off",
    FmmM2lSingleMatvecWorkspace *workspace = nullptr);

void apply_treecode_operator(const std::vector<Panel> &panels,
                             const std::vector<double> &density,
                             double contrast, double kappa,
                             const TreeWorkspace &tree, TitanExecutor &executor,
                             std::vector<double> &out,
                             TreecodeStats *stats = nullptr);

void apply_jcp_treecode_operator(
    const std::vector<Panel> &panels, const std::vector<double> &unknowns,
    double eps_ratio, double kappa, double lhs_phi, double lhs_phin,
    bool coupled, const TreeWorkspace &tree,
    const std::string &near_integration, const std::string &quad_buffer,
    TitanExecutor &executor, std::vector<double> &out,
    TreecodeStats *stats = nullptr);

Result solve_linear_pb(const std::vector<Atom> &atoms, const Options &options,
                       TitanExecutor &executor);

} // namespace TitanPBE::Tabi

#endif // TITAN_PBE_TABI_TYPES_H
