#ifndef ARGUMENTS_H
#define ARGUMENTS_H
#include <cmath>
#include <limits>
#include <string>
#include <vector>
using namespace std;

// ── Domain sub-structs ──────────────────────────────────────
struct PbcArgs {
  int solv_pbc{0};
  bool solv_pbcn{false};
  bool solv_pbctopo{false};
  bool auto_enabled_pbcn{false};
  bool auto_enabled_pbci{false};
  int solv_pbc_int{0}, solv_pbc_int2{0}, solv_pbc_int3{0};
  bool pbc_aggressive{false};
  bool pbc_mem{false};
  bool no_center{false};
  int chain_ref{-1};
  bool solv_pbc_clust{false};
  vector<char> pbc_clust_axis;
  bool pbc_envi{false};
  vector<string> pbc_envi_chain;
  int pbc_mol2_aggressive{0};
  bool flag_pbc_ref{false};
  string ref_pdb_name{};
  string pbctopo_mode{"vibe"};
  string pbctopo_search_policy{"fast-certified"};
  int pbctopo_audit_max_states{10000000};
  double pbctopo_audit_max_seconds{60.0};
  int pbctopo_audit_max_memory_mb{1024};
  string pbctopo_anchor_backend{"auto"};
  string pbctopo_anchor_metrics{"full"};
  string pbctopo_temporal{"greedy"};
  int pbctopo_temporal_k{4};
  double pbctopo_temporal_penalty{2.0};
  bool pbctopo_evidence_export{false};
  double pbctopo_vdw_clash_scale{0.55};
  double pbctopo_bond_cutoff{0.0};
  bool pbctopo_bond_cutoff_set{false};
  double pbctopo_clash_cutoff{0.0};
  bool pbctopo_clash_cutoff_set{false};
  double pbctopo_contact_cutoff{0.0};
  bool pbctopo_contact_cutoff_set{false};
  string pbctopo_required_contacts_file{};
  double pbctopo_continuity_cutoff{0.0};
  bool pbctopo_continuity_cutoff_set{false};
  int pbctopo_exact_component_limit{0};
  bool pbctopo_exact_component_limit_set{false};
  int pbctopo_beam_width{0};
  bool pbctopo_beam_width_set{false};
  int pbctopo_component_shell_radius{0};
  bool pbctopo_component_shell_radius_set{false};
  bool pbctopo_h9_orbit{false};
};

struct MsmArgs {
  enum class ItsSolverMode {
    Full,
    Schur,
    Partial,
  };

  enum class PcaSolverMode {
    Full,
    Partial,
  };

  bool flag_msm{false};
  bool flag_tica{false};
  // Default/cap for MSM microstates when the user does not pass
  // -msm-clusters explicitly. The engine may auto-reduce this value after
  // TICA finishes if the trajectory is too short for a stable MSM/DHAM fit.
  int msm_clusters{25};
  bool msm_clusters_user_set{false};
  int msm_lag{10};
  int msm_nts{3};
  int msm_kmeans_iter{300};
  int msm_seed{42};
  bool msm_auto_boundary{true};
  int msm_macrostates{-1};
  bool msm_run_pcca{true};
  bool msm_run_its{true};
  vector<int> msm_its_lags{};
  ItsSolverMode msm_its_solver{ItsSolverMode::Full};
  int msm_its_topk{0};
  int msm_its_max_k_full{256};
  int tica_lag{10}, tica_components{5};
  std::string tica_backend{"auto"};
  int tica_rank{24};
  PcaSolverMode pca_solver{PcaSolverMode::Full};
  int pca_topk{0};
  int pca_max_k_full{512};
  bool pca_symmetry_aware{false};
};

struct FeatureExportArgs {
  bool export_features{false};
  std::string source{"auto"};
  bool source_explicit{false};
};

struct PerformanceReportArgs {
  std::string mode{"off"};
};

struct UncertaintyArgs {
  std::string mode{"off"};
  std::string blocks{"auto"};
  int bootstrap_samples{200};
  unsigned int seed{42};
};

struct OutputManifestArgs {
  bool dry_run{false};
  bool output_manifest{false};
  std::string schema_export{"off"};
};

struct GistArgs {
  bool flag_gist{false};
  double gist_resolution{0.5};
  double gist_padding{8.0};
  double gist_temperature{300.0};
  std::string gist_water_model{"TIP3P"};
  int gist_min_count{10};
  double gist_cutoff{9.0};
};

// The only analytic SES engine whose vertices are certified on the analytic
// patches.  `current` signs the field from a 26-direction sampling of the
// probe ball that does not refine with the grid, so its surface never
// converges to the SES; it is kept to reproduce earlier runs and warns when
// it is selected.  See docs/TITAN_SES_ENGINE_FIX_2026-09-15.md.
inline constexpr const char *kTabiDefaultAnalyticSesEngine =
    "patch-ray-experimental";

struct PbeArgs {
  bool flag_pbe{false};
  std::string backend{"grid"};
  double scale{2.0};
  double margin{15.0};
  std::string gridbox{"cuboidal"};
  double int_diel{1.0};
  double ext_diel{80.0};
  double salt{0.15};
  double temp{298.15};
  std::string model{"linear"};
  std::string biomodel{"pbe"};
  std::string dielectric_model{"two-dielectric"};
  std::string boundary{"coulombic"};
  std::string charge_assignment{"trilinear"};
  std::string surface{"vdw"};
  std::string radius_convention{"auto"};
  double probe_radius{1.4};
  std::string ion_access{"dielectric"};
  double ion_radius{2.0};
  int max_iter{10000};
  double rms_tol{1.0e-6};
  double dphi_tol{1.0e-5};
  std::string omega{"auto"};
  double omega_value{0.0};
  std::size_t max_grid_points{25000000ULL};
  std::string map{"off"};
  std::string map_frames{"none"};
  std::string atom_potentials{"off"};
  std::string binding_cycle{"off"};
  std::string binding_receptor_pqr{};
  std::string binding_ligand_pqr{};
  std::string kernel{"auto"};
  std::string focus{"off"};
  std::string multigrid{"off"};
  int focus_levels{1};
  double focus_margin{8.0};
  std::string mg_levels{"auto"};
  int mg_pre{2};
  int mg_post{2};
  int mg_coarse_iter{200};
  int nl_max_iter{80};
  int nl_inner_iter{40};
  int nl_stagnation{8};
  int nl_block{5};
  double nl_dphi{1.0e-4};
  std::string nl_relax{"auto"};
  double nl_relax_value{0.5};
  double nl_clip{20.0};
  std::string nl_continuation{"auto"};
  std::string nl_line_search{"auto"};
  double nl_residual{1.0e-5};
  std::string nl_solver{"picard"};
  std::string nl_inner{"auto"};
  std::string nl_multigrid{"off"};
  std::string nwt_kernel{"auto"};
  std::string nwt_homo_kernel{"auto"};
  std::string nwt_progress{"auto"};
  std::string linear_kernel{"auto"};
  double gaussian_exp{2.0};
  double gaussian_sigma{1.0};
  double density_cutoff{0.759};
  std::string surface_cutoff{"auto"};
  double surface_cutoff_value{0.0};
  double gap_diel{22.0};
  std::string probe_file{};
  std::string probe_residue{};
  std::string probe_site{};
  bool focus_center_user{false};
  double focus_center_x{0.0};
  double focus_center_y{0.0};
  double focus_center_z{0.0};
  bool dgsolv{false};
  bool pbsa{false};
  std::string pbsa_components{"auto"};
  std::string pbsa_decomp{"auto"};
  std::string pbsa_mol1_type{"protein"};
  std::string pbsa_mol2_type{"lig"};
  bool pbsa_ie{true};
  std::string pbsa_residue_frames{"off"};
  std::string pbsa_ie_block{"auto"};
  int pbsa_ie_min_samples{10};
  double pbsa_ie_filter_sigma{2.0};
  std::string pbsa_perf{"summary"};
  std::string pbsa_grid_reuse{"auto"};
  int pbsa_sasa_points{96};
  std::string timing{"off"};
  std::string tabi_diagnostics{"off"};
  std::string tabi_tree_preset{"off"};
  std::string tabi_sum{"auto"};
  std::string tabi_geometry{"triangle"};
  std::string tabi_diagonal{"solid-angle"};
  int tabi_surface_points{96};
  std::string tabi_surface_output{"off"};
  bool tabi_surface_only{false};
  bool tabi_highres_surface{false};
  bool tabi_calderon_trace_audit{false};
  std::string tabi_calderon_aux_surface{"off"};
  double tabi_calderon_aux_grid_spacing{1.25};
  std::string tabi_primary_surface{"sphere-patch"};
  std::string tabi_analytic_ses_engine{kTabiDefaultAnalyticSesEngine};
  double tabi_ses_grid_spacing{0.75};
  std::size_t tabi_ses_max_grid_vertices{4000000};
  std::string tabi_sphere_mesh{"auto"};
  int tabi_tree_leaf_size{32};
  // Budget for the curved node-patch pair cache.  256 MiB holds the dense
  // all-pairs table up to about 2896 nodes; past that a bounded sparse
  // cache is used and uncached pairs are recomputed each matvec.
  int tabi_curved_pair_cache_mb{256};
  bool tabi_tree_leaf_size_explicit{false};
  double tabi_tree_opening_theta{0.45};
  bool tabi_tree_opening_theta_explicit{false};
  int tabi_taylor_order{3};
  std::string tabi_derivatives{"analytic"};
  std::string tabi_formulation{"jcp2013"};
  bool tabi_formulation_explicit{false};
  std::string tabi_discretization{"centroid"};
  bool tabi_discretization_explicit{false};
  std::string tabi_energy_projection{"boundary"};
  std::string tabi_precond{"auto"};
  bool tabi_precond_explicit{false};
  // Preserve the user spelling across post-process auto resolution.  The
  // calibrated Cartesian retry mode is allowed only when ILU(0) was
  // explicitly requested, not when an `auto` default happens to resolve to it.
  std::string tabi_precond_requested{"auto"};
  std::string tabi_precond_block_near{"singular"};
  int tabi_precond_block_panel_cap{32};
  int tabi_precond_overlap_panel_cap{64};
  int tabi_precond_ilu_neighbor_cap{192};
  bool tabi_precond_ilu_neighbor_cap_explicit{false};
  double tabi_precond_ilu_drop_tol{0.0};
  // Legacy parser field. Sparse ILU always runs the automatic
  // ILU(0) -> regularized ILU -> diagonal fallback pipeline.
  std::string tabi_precond_ilu_regularization{"adaptive"};
  double tabi_precond_ilu_regularization_max{1.0e-2};
  int tabi_precond_memory_budget_mb{256};
  int tabi_precond_coarse_aggregate_cap{16};
  std::string tabi_precond_coarse_base{"diagonal"};
  std::string tabi_precond_coarse_verify{"off"};
  std::string tabi_near_integration{"auto"};
  std::string tabi_quad_buffer{"auto"};
  std::string tabi_leaf_near_policy{"auto"};
  std::string tabi_leaf_panel_kernel{"auto"};
  std::string tabi_adaptive_level{"strict"};
  std::string tabi_cluster_policy{"off"};
  std::string tabi_leaf_pair_kernel{"auto"};
  std::string tabi_leaf_kernel_profile{"off"};
  std::string tabi_leaf_work_profile{"off"};
  std::string tabi_matvec_cost_profile{"off"};
  std::string tabi_jcp_direct_kernel{"baseline"};
  std::string tabi_leaf_pair_selection_profile{"off"};
  std::string tabi_leaf_pair_probe_cache{"off"};
  std::string tabi_leaf_pair_target_group{"scan"};
  std::string tabi_leaf_pair_directness_probe{"full-traversal"};
  std::string tabi_leaf_pair_directness_cache{"off"};
  std::string tabi_leaf_pair_directness_verify{"off"};
  std::string tabi_leaf_pair_scheduler{"inline-select"};
  int tabi_leaf_pair_worklist_max_active{2048};
  std::string tabi_leaf_pair_worklist_verify{"off"};
  std::string tabi_leaf_block_verify{"off"};
  std::string tabi_leaf_mac{"off"};
  std::string tabi_fmm_m2l{"auto"};
  bool tabi_fmm_m2l_explicit{false};
  std::string tabi_fmm_engine{"cartesian"};
  int bie_fmm_cartesian_order{0};
  int tabi_kifmm_order{6};
  double tabi_kifmm_equiv_scale{1.05};
  double tabi_kifmm_check_scale{1.60};
  std::string tabi_kifmm_solver{"qr"};
  std::string tabi_kifmm_profile{"off"};
  std::string tabi_kifmm_full_apply{"serial"};
  std::string tabi_kifmm_ilu_graph{"cartesian-near"};
  int tabi_kifmm_hierarchy_order{8};
  double tabi_kifmm_hierarchy_theta{0.6};
  std::string tabi_kifmm_hierarchy_real_panel_guard{"strict-local-check"};
  double tabi_kifmm_hierarchy_guard_margin_factor{0.0};
  std::string tabi_kifmm_hierarchy_tree_stop_policy{"panel-radius-occupancy"};
  double tabi_kifmm_hierarchy_tree_panel_radius_factor{1.5};
  int tabi_kifmm_hierarchy_tree_occupancy_stop{32};
  std::string tabi_kifmm_hierarchy_m2l_eval_policy{"target-check-batched"};
  std::string tabi_kifmm_hierarchy_matrix_cache_policy{
      "canonical-offset-cache"};
  double tabi_kifmm_hierarchy_matrix_cache_limit_mb{0.0};
  std::string tabi_kifmm_hierarchy_gemm_batch_policy{"per-pair"};
  std::string tabi_kifmm_hierarchy_m2l_parallel_policy{"serial"};
  int tabi_kifmm_hierarchy_m2l_parallel_workers{1};
  int tabi_cartesian_k4_order{5};
  double tabi_cartesian_k4_theta{0.6};
  double tabi_cartesian_k4_geometry_theta{0.4};
  double tabi_cartesian_k4_tail_tol{0.002};
  std::string tabi_fmm_cartesian_audit{"off"};
  std::string tabi_fmm_cartesian_error_audit{"off"};
  std::string tabi_fmm_local{"off"};
  std::string tabi_fmm_local_refine{"off"};
  double tabi_fmm_local_refine_tol{4.0e-3};
  std::string tabi_fmm_local_cache{"off"};
  int tabi_fmm_local_cache_budget_mb{256};
  std::string tabi_fmm_local_fallback{"scalar"};
  std::string tabi_fmm_local_plan_builder{"legacy-rescan"};
  int tabi_fmm_local_refine_probe_count{4};
  std::string tabi_fmm_local_refine_probe_mode{"rademacher"};
  std::string tabi_fmm_local_geometry_guard{"off"};
  double tabi_fmm_local_geometry_theta{0.0};
  bool tabi_fmm_local_ilu_cap_sweep{false};
  std::string tabi_interaction_mode{"auto"};
  std::string tabi_node_bounds{"moment"};
  std::string tabi_tree_builder{"grouped-morton"};
  std::string tabi_expansion_center{"legacy"};
  std::string tabi_fragmentation_safety{"off"};
  std::string tabi_mac_safety{"off"};
  double tabi_mac_gap_factor{1.5};
  std::string tabi_moment_audit{"off"};
  std::string tabi_moment_construction{"translated"};
  std::string tabi_moment_sum{"naive"};
  std::string tabi_convergence_audit{"off"};
  std::string tabi_convergence_safety{"off"};
  double tabi_convergence_rho_cap{0.0};
  double tabi_mac_aspect_cap{0.0};
  std::string tabi_convergence_estimator{"off"};
  double tabi_convergence_tail_threshold{1.0e4};
  double tabi_convergence_estimator_error_cap{1.0e-3};
  double tabi_convergence_unstable_rho_cap{0.25};
  std::string tabi_convergence_calibration{"stage3j"};
  std::string tabi_convergence_dielectric_amp{"off"};
  std::string tabi_convergence_estimator_model{"stage3l"};
  std::string tabi_convergence_channel_calibration{"stage3m"};
  std::string tabi_convergence_split_policy{"global"};
  std::string tabi_convergence_split_selectivity{"legacy"};
  double tabi_convergence_split_absolute_floor{0.0};
  std::string tabi_convergence_overreject_audit{"off"};
  std::string tabi_convergence_overreject_detail{"off"};
  std::string tabi_convergence_shape_audit{"off"};
  std::string tabi_convergence_kernel_response_audit{"off"};
  std::string tabi_channel_missed_semantics{"legacy"};
  std::string tabi_accepted_miss_diagnostics{"off"};
  double tabi_accepted_miss_absolute_floor{1.0e-7};
  double tabi_convergence_k3_accepted_rho_cap{0.0};
  std::string tabi_mac_distance_policy{"center"};
  double tabi_mac_surface_margin_factor{0.25};
  std::string tabi_mac_surface_margin_policy{"all"};
  double tabi_mac_surface_margin_aspect_threshold{3.0};
  double tabi_mac_surface_margin_rho_trigger{0.55};
  double tabi_mac_surface_margin_soft_aspect_threshold{0.0};
  double tabi_mac_surface_margin_soft_rho_trigger{0.0};
  double tabi_mac_surface_margin_soft_factor{0.10};
  int tabi_target_batch_size{1};
  int tabi_leaf_pair_min_source_panels{64};
  int tabi_leaf_pair_min_targets{4};
  int tabi_leaf_pair_max_lookahead{32};
  std::string tabi_expansion_audit{"off"};
  std::string tabi_expansion_audit_channel{"off"};
  int tabi_expansion_audit_sample_count{512};
  int tabi_expansion_audit_max_panels{2048};
  std::string tabi_matvec_audit{"off"};
  int tabi_matvec_audit_sample_count{1024};
  std::string tabi_node_patch_operator_audit{"on"};
  std::string tabi_progress{"auto"};
  std::string tabi_pre_solver_progress{"off"};
  int tabi_progress_interval{5};
  int tabi_gmres_restart{32};
  std::string tabi_true_residual_probe{"restart"};
  std::string tabi_true_residual_probe_scope{"full"};
  int tabi_true_residual_probe_interval{8};
  int tabi_true_residual_sample_count{1024};
  std::string stability_audit{"off"};
  std::string stability_mode{"off"};
  std::string grid_anchor{"auto"};
  std::string test_pqr{};
  int test_grid_size{0};
  bool test_center_user{false};
  double test_center_x{0.0};
  double test_center_y{0.0};
  double test_center_z{0.0};
  std::string parity_dump{"off"};
  std::string parity_map_dump{"off"};
  std::string test_charge_grid{"native"};
  std::string test_charge_grid_diff{"off"};
  std::string test_surface_map{"native"};
  std::string test_midpoint_epsilon{"native"};
  std::string test_midpoint_diff{"off"};
  std::string test_vdwms_remap{"auto"};
  std::string test_boundary_map{"native"};
  std::string test_boundary_convention{"native-real"};
  std::string test_field_units{"native"};
  std::string test_operator_convention{"native"};
  std::string test_rhs_map{"native"};
  std::string test_screen_map{"native"};
  std::string test_denom_map{"native"};
  std::string test_nl_source{"titan"};
  bool test_newton_derivative{false};
  std::string test_energy_convention{"auto"};
};

struct DeArgs {
  bool flag_find_pulling_direct{false};
  std::string pull_opt{"de"};
  size_t pull_max_evals{0};
  unsigned int pull_seed{42};
  double de_diff_weight{0.9};
  double de_cross_prob{0.9};
  double de_F{0.5};
  double de_scan_radius{2.};
  double de_min_scan_radius{0.5};
  double de_restrain_angle{90.};
  int de_flag_score_function{1};
  size_t de_pop_num{6480};
  size_t de_algorithm{};
  size_t de_ite{100};
  bool de_adaptive_F{true};
  int de_stagnation_limit{20};
  size_t de_memory_size{10};
  double de_p_best_rate{0.2};
  double de_archive_scale{1.0};
  size_t de_min_pop_num{4};
  bool user_set_de_diff_weight{false};
  bool user_set_de_cross_prob{false};
  bool user_set_de_F{false};
  bool user_set_de_algorithm{false};
  bool user_set_de_adaptive_F{false};
  bool user_set_de_memory_size{false};
  bool user_set_de_p_best_rate{false};
  bool user_set_de_archive_scale{false};
  bool user_set_de_min_pop_num{false};
  bool flag_de_auto_term{false};
  std::string pathway_file{};
  std::string pathway_route_id{};
  std::string path_previous_status{};
  double path_continuity_window{0.0};
  bool user_set_pathway_group{false};
  double path_graph_edge_cutoff{4.0};
  double path_recovery_enter_radius{4.0};
  double path_recovery_exit_radius{3.0};
  double path_recovery_max_turn_angle{60.0};
  double path_corridor_spacing{0.5};
  size_t path_corridor_radial_samples{16};
  double path_registration_max_rmsd{2.0};
  double path_completion_tolerance{1.0};
  double path_completion_min_separation{0.0};
  double path_max_turn_angle{20.0};
  double path_energy_key_epsilon{0.05};
  double path_min_clearance{1.5};
  double path_probe_length{0.0};
  vector<double> path_probe_lengths{};
  bool path_navigation_enabled{false};
  double path_navigation_tube_weight{1.0};
  double path_navigation_backtrack_weight{1.0};
  double path_navigation_cost_epsilon{0.0};
  size_t path_navigation_seed_radial_samples{8};
  double path_navigation_seed_amplitude{0.35};
  std::string path_clearance_mode{"endpoint-center"};
  bool path_geometry_only{false};
  double dis_along_path{4.};
  bool flag_pathway{false};
  size_t group_pathway{};
};

struct LlpsArgs {
  bool flag_llps{false};
  float llps_rcut{0.7f};
  float llps_perc_thr{0.5f};
  int llps_maxlag{500};
  int llps_dimer_size{1};
  bool llps_contacts{false};
  std::string llps_sticker_beads{"BB,SC1,SC2,BB1,BB2,BB3,SC3,SC4"};
  float llps_voxel_size{0.5f};
  bool llps_dynamics{false};
  bool llps_dynamics_detailed{false};
  double llps_lineage_overlap{0.30};
  float llps_prna_rcut{0.0f};
  double llps_msd_fit_start_ns{-1.0};
  double llps_msd_fit_end_ns{-1.0};
  std::string llps_density{"auto"};
  std::string llps_phase_geometry{"none"};
  bool llps_dbscan{true};
  int llps_phase_axis{2};
  double llps_phase_dense{0}, llps_phase_dilute_inner{0}, llps_phase_dilute_outer{0};
  std::string llps_kernel{"auto"};
  std::string llps_rna_compaction{"auto"};
  std::string llps_rna_information{"auto"};
  std::string llps_rna_topology{"auto"};
  bool llps_rna_topology_detailed{true};
  float llps_rna_intra_rcut{0.0f};
  int llps_rna_longrange_min{10};
  float llps_rna_bridge_geom_min{-1.0f};
  int llps_rna_patch_gap_max{1};
  bool llps_rna_clamp_require_loop{true};
  int llps_rna_info_rg_bins{16};
  int llps_rna_info_longrange_bins{16};
  std::string llps_energy_correlation{"auto"};
  bool llps_energy_detailed{false};
  bool llps_paper{false};
  double llps_paper_profile_bin_nm{0.2};
  int llps_paper_blocks{0}; // 0 = auto
  std::string llps_paper_geometry{"auto"};
  std::string llps_paper_fractal{"auto"};
  // DBSCAN bead-level morphology parameters (unit: Angstrom)
  double llps_dbscan_eps{15.0}; // epsilon (Å); default 15 Å ≈ 1.5 nm
  int llps_dbscan_minpts{5};    // min neighbours (self excluded)
  bool llps_fuzzy{false};       // enable Fuzzy EM refinement after DBSCAN
};

struct IfpArgs {
  bool flag_ifp{false};
  bool pbctopo_certified_only{false};
  double cutoff_hbond{3.5};       // D···A distance (Å)
  double cutoff_salt_bridge{4.0}; // charged atom pair (Å)
  double cutoff_hydrophobic{4.0}; // C···C contact (Å)
  double cutoff_pi_pi{5.5};       // ring COM–COM (Å)
  double cutoff_cation_pi{6.0};   // cation–ring COM (Å)
  double cutoff_halogen{4.0};     // X···acceptor (Å)
  std::string output_filename{};  // empty → uses info.get_ifp_filename()
};

struct AwhArgs {
  bool flag_awh_ifp{false};
  std::string xvg_list_file{};
  int xvg_value_column{1};
  int n_bins{20};
  std::string output_prefix{};
};
struct DhamArgs {
  std::string segment_map_file{};  // audited segment/exchange input; counts audit only
  bool flag_dham{false};
  std::string temps_file{};         // file with replica temperatures (K)
  std::string ener_list_file{};     // file listing paths to .xvg energy files
  std::string ensemble{"nvt"};      // nvt | npt
  std::string window_match{"auto"}; // auto | off | strict
  double target_temp{300.0};        // target reweighting temperature (K)
  double target_pressure_bar{1.0};  // target pressure for NPT reweighting (bar)
  int max_iter{5000};
  double tolerance{1e-6};
  int n_replicas{0};        // explicit validation (0 = auto from temps)
  double dt_energy_ps{0.0}; // energy .xvg output freq (ps); 0 = auto from file
};

struct MbarArgs {
  bool flag_mbar{false};
  std::string temps_file{};         // optional ladder temperatures (K)
  std::string ener_list_file{};     // file listing paths to .xvg energy files
  std::string domain{"tica"};       // tica | pca | both
  std::string ensemble{"nvt"};      // nvt | npt
  std::string window_match{"auto"}; // auto | off | strict
  std::string temp_mode{"fixed"};   // fixed | per_frame
  double target_temp{300.0};        // target reweighting temperature (K)
  double target_pressure_bar{1.0};  // target pressure for NPT reweighting (bar)
  double dt_energy_ps{0.0};         // energy .xvg output freq (ps); 0 = auto
  int temp_column{2};               // xvg temperature/state-label column
  int bootstrap_samples{0};         // 0 = off
  int scan_points{0};           // 0/1 = no scan output, >1 = write mbar_cv.dat
  std::string demux_map_file{}; // optional per-frame temperature-state map
};

struct TptArgs {
  bool flag_tpt{false};
  std::string mode{"ensemble"}; // ensemble | kinetic
  std::string source_spec{};    // microstate list/range, e.g. micro:0,1-3
  std::string sink_spec{};      // microstate list/range, e.g. 8-12
  int top_paths{5};
  double flux_cut{0.0};
  bool use_dham{true};
  std::string demux_map_file{}; // optional demux replica-map for kinetic mode
  int bootstrap_samples{0};     // 0 = off
  unsigned int bootstrap_seed{42};
};

struct CommunityArgs {
  bool flag_community{false};
  std::string method{};
};

struct GraphArgs {
  bool flag_graph{false};
  std::vector<std::string> sources{};
  std::string community_method{"auto"};
  int filter_topk{0};
  std::string filter_component{"all"};
};

struct CMapArgs {
  bool flag_cmap{false};
  std::string mode{"sidechain-com"};
  std::string scope{"mol1"};
  double cutoff_ang{6.5};
  bool cutoff_explicit{false};
  std::string domain{"both"};
  bool domain_explicit{false};
  int min_seqsep{3};
  bool min_seqsep_explicit{false};
  bool new_option_explicit{false};
  bool legacy_alias_seen{false};
  bool legacy_alias_enabled{false};
};

struct SegmentArgs {
  std::string config_file{};
};

struct InterfaceAnalysisArgs {
  bool flag_interface_analysis{false};
  std::string contact_mode{"sidechain-com"};
  double cutoff_ang{6.5};
  int state_min_contacts{1};
  int sasa_stride{10};
  int sasa_points{96};
  bool uses_mol2{false};
};

struct QNativeArgs {
  bool flag_qnative{false};
  std::string ref_file{};
  bool ref_explicit{false};
  std::string mode{"sidechain-com"};
  std::string scope{"mol1"};
  double cutoff_ang{6.5};
  bool cutoff_explicit{false};
  std::string domain{"both"};
  bool domain_explicit{false};
  int min_seqsep{3};
  bool min_seqsep_explicit{false};
  bool soft{false};
  bool soft_explicit{false};
  double beta{5.0};
  bool beta_explicit{false};
};

struct DihedralArgs {
  bool flag_dihedral{false};
  std::string kind{"backbone"};
  std::string residues{"all"};
  int stride{1};
  bool export_tica{false};
};

struct MsdArgs {
  bool flag_msd{false};
  std::string selection{"mol1"};
  std::string mode{"com"};
  std::string maxlag{"auto"};
  std::string fit_window{"auto"};
  std::string pbc{"auto"};
  bool rot{false};
  std::string rot_fit_window{"auto"};
};

struct MembraneArgs {
  bool flag_membrane{false};
  std::string group{"auto"};
  std::string lipids{"auto"};
  std::string head{"auto"};
  std::string tail{"auto"};
  std::string normal{"auto"};
  std::string grid{"auto"};
  std::string leaflet{"auto"};
  std::string leaflet_file{};
  int flipflop_persistence{3};
  int resolved_group{-1};
  bool protein_contacts{false};
  double contact_cut{6.5};
  bool enrichment{false};
  std::string curvature{"off"};

  std::string validation_error() const {
    const auto is_nonnegative_integer = [](const std::string &value) {
      if (value.empty())
        return false;
      for (const char c : value) {
        if (c < '0' || c > '9')
          return false;
      }
      return true;
    };
    const auto is_positive_integer = [&](const std::string &value) {
      if (!is_nonnegative_integer(value))
        return false;
      return value.find_first_not_of('0') != std::string::npos;
    };

    if (group != "auto" && !is_nonnegative_integer(group))
      return "-membrane-group must be auto or an integer >= 0.";
    if (normal != "auto" && normal != "x" && normal != "y" && normal != "z")
      return "-membrane-normal must be one of auto, x, y, z.";
    if (grid != "auto") {
      const auto comma = grid.find(',');
      if (comma == std::string::npos ||
          grid.find(',', comma + 1) != std::string::npos ||
          !is_positive_integer(grid.substr(0, comma)) ||
          !is_positive_integer(grid.substr(comma + 1))) {
        return "-membrane-grid must be auto or two positive integers Nx,Ny.";
      }
    }
    if (leaflet != "auto" && leaflet != "upper-lower" && leaflet != "user")
      return "-membrane-leaflet must be one of auto, upper-lower, user.";
    if (flipflop_persistence < 1)
      return "-membrane-flipflop-persistence must be >= 1.";
    if (leaflet == "user" && leaflet_file.empty())
      return "-membrane-leaflet=user requires -membrane-leaflet-file.";
    if (leaflet != "user" && !leaflet_file.empty())
      return "-membrane-leaflet-file is valid only with "
             "-membrane-leaflet=user.";
    if (!(contact_cut > 0.0) || !std::isfinite(contact_cut))
      return "-membrane-contact-cut must be a finite positive distance in A.";
    if (curvature != "off" && curvature != "grid")
      return "-membrane-curvature must be one of off, grid.";
    return {};
  }
};

struct S2Args {
  bool flag_s2{false};
  std::string vector{"both"};
  std::string align{"rms"};
  std::string maxlag{"auto"};
  std::string fit_window{"auto"};
  std::string residues{"all"};
  bool bfactor_pdb{false};

  std::string validation_error() const {
    const auto is_positive_integer = [](const std::string &value) {
      if (value.empty())
        return false;
      for (const char c : value) {
        if (c < '0' || c > '9')
          return false;
      }
      return value.find_first_not_of('0') != std::string::npos;
    };
    const auto parse_double = [](const std::string &value, double &parsed) {
      try {
        size_t parsed_chars = 0;
        parsed = std::stod(value, &parsed_chars);
        return parsed_chars == value.size() && std::isfinite(parsed);
      } catch (...) {
        return false;
      }
    };

    if (vector != "nh" && vector != "ca-c" && vector != "both")
      return "-s2-vector must be one of nh, ca-c, both.";
    if (align == "provided")
      return "-s2-align=provided is not supported in v1; use rms for "
             "first-frame alignment or none.";
    if (align != "rms" && align != "none")
      return "-s2-align must be one of rms, none.";
    if (maxlag != "auto" && !is_positive_integer(maxlag))
      return "-s2-maxlag must be auto or an integer >= 1.";
    if (fit_window != "auto") {
      const auto comma = fit_window.find(',');
      double tmin = 0.0;
      double tmax = 0.0;
      if (comma == std::string::npos ||
          fit_window.find(',', comma + 1) != std::string::npos ||
          !parse_double(fit_window.substr(0, comma), tmin) ||
          !parse_double(fit_window.substr(comma + 1), tmax) || !(tmax > tmin)) {
        return "-s2-fit-window must be auto or tmin,tmax with tmax > tmin.";
      }
    }
    if (residues.empty())
      return "-s2-residues must be 'all' or a non-empty residue list.";
    return {};
  }
};

struct UmapArgs {
  bool flag_umap{false};
  int umap_neighbors{15};
  int umap_components{2};
  int umap_epochs{200};
  double umap_min_dist{0.1};
  double umap_spread{1.0};
  unsigned int umap_seed{42};
  string umap_knn_mode{"auto"};
  bool umap_knn_benchmark{false};
  bool umap_verbose{false};
};

struct DsspArgs {
  bool flag_dssp{false};
  double hbond_energy_cutoff{-0.5};
};

struct RmsArgs {
  bool flag_rms{false};
};

struct RgArgs {
  bool flag_rg2{false};
  int hist_bins{100};
};

struct LjArgs {
  bool flag_lj{false};
  double lj_cluster_cutoff{5.0};
  int lj_min_cluster_size{2};
  double lj_rho_bin{1.0};
  double lj_rho_max{0.0};
  double lj_solid_q6{0.40};
  int lj_solid_min_cn{6};
  double lj_crystal_tol{0.08};
};

struct SqArgs {
  bool flag_sq{false};
  std::vector<std::string> sq_channels{"all"};
  double sq_rdf_bin{1.0};
  double sq_rdf_max{0.0};
  bool sq_rdf_max_auto{true};
  double sq_q_bin{0.05};
  double sq_q_max{0.0};
  bool sq_q_max_auto{true};
};

struct CorrKnnArgs {
  bool flag_corr_knn{false};
  int estimator_k{4};
  int tau_lag{5};
  double pair_cutoff_ang{0.0};
  int pair_topk{0};
  bool pair_use_contact_prior{false};
  double pair_contact_cutoff_ang{8.0};
  bool use_avg_cutoff{true};
  bool compute_te{true};
};

struct ValidationArgs {
  bool flag_validation{false};
  std::vector<int> ck_multiples{2, 3, 4, 5};
  std::vector<int> vamp_components{};
  std::vector<int> vamp_lags{};
  std::vector<int> vamp_clusters{};
  int vamp_folds{5};
  bool auto_pick{false};
  int bootstrap_samples{0};
  unsigned int bootstrap_seed{42};
};

struct HBondKineticsArgs {
  bool flag_hbk{false};
  std::vector<std::string> hbk_kinds{"all"};
  std::string hbk_domain{"both"};
  int hbk_maxlag{500};
  double hbk_hbond_cutoff{0.0};
  bool hbk_hbond_cutoff_explicit{false};
  double hbk_angle_deg{0.0};
  bool hbk_angle_explicit{false};
  double hbk_salt_cutoff{4.0};
  double hbk_water_cutoff{0.0};
  bool hbk_water_cutoff_explicit{false};
};

struct ZtsArgs {
  bool flag_zts{false};
  int zts_images{50};
  int zts_max_iter{2000};
  double zts_step{0.01};
  double zts_tol{1e-5};
  int zts_source{-1};
  int zts_sink{-1};
  double zts_fes_sigma{1.0};
  bool zts_smooth_fes{true};
  std::string zts_init_mode{"linear"};
  std::string zts_domain{"pca"};
  bool zts_domain_explicit{false};
};

// ── Main arguments struct ───────────────────────────────────
struct EDecompArgs {
  bool flag_edecomp{false};
  std::string source{"mm"};
  std::string level{"residue"};
  std::string edge_top{"200"};
  std::string rank_by{"abs-mean"};
  std::string frame_output{"off"};
  std::string perf{"summary"};

  std::string validation_error() const {
    if (source != "mm" && source != "gbsa" && source != "pbsa" &&
        source != "all")
      return "-edecomp-source must be one of mm, gbsa, pbsa, all.";
    if (level != "residue" && level != "atom" && level != "both")
      return "-edecomp-level must be one of residue, atom, both.";
    if (edge_top != "all") {
      if (edge_top.empty())
        return "-edecomp-edge-top must be a positive integer or all.";
      for (const char c : edge_top) {
        if (c < '0' || c > '9')
          return "-edecomp-edge-top must be a positive integer or all.";
      }
      if (edge_top.find_first_not_of('0') == std::string::npos)
        return "-edecomp-edge-top must be a positive integer or all.";
    }
    if (rank_by != "abs-mean" && rank_by != "mean-abs")
      return "-edecomp-rank-by must be one of abs-mean, mean-abs.";
    if (frame_output != "off" && frame_output != "all")
      return "-edecomp-frame-output must be one of off, all.";
    if (perf != "off" && perf != "summary" && perf != "detail")
      return "-edecomp-perf must be one of off, summary, detail.";
    return {};
  }
};

struct HotspotReportArgs {
  bool flag_hotspot_report{false};
  std::string source{"both"};
  std::string rank_by{"combined"};
  bool rank_by_explicit{false};

  std::string validation_error() const {
    if (source != "cas" && source != "edecomp" && source != "both")
      return "-hotspot-source must be one of cas, edecomp, both.";
    if (rank_by != "cas-ddg" && rank_by != "edecomp-abs-mean" &&
        rank_by != "combined")
      return "-hotspot-rank-by must be one of cas-ddg, edecomp-abs-mean, "
             "combined.";
    if (source == "cas" && rank_by == "edecomp-abs-mean")
      return "-hotspot-rank-by=edecomp-abs-mean requires "
             "-hotspot-source=edecomp or both.";
    if (source == "edecomp" && rank_by == "cas-ddg")
      return "-hotspot-rank-by=cas-ddg requires -hotspot-source=cas or both.";
    if (source != "both" && rank_by == "combined")
      return "-hotspot-rank-by=combined requires -hotspot-source=both.";
    return {};
  }
};

struct arguments {
private:
  bool parse_input_file_is_open();
  int tokenize_arg_file(const std::string &filename,
                        std::vector<std::string> &tokens);
  bool parsing_arg_file_tokens{false};

public:
  // Domain sub-structs
  PbcArgs pbc;
  MsmArgs msm;
  FeatureExportArgs feature_export;
  PerformanceReportArgs perf_report;
  UncertaintyArgs uncertainty;
  OutputManifestArgs output_manifest;
  GistArgs gist;
  PbeArgs pbe;
  DeArgs de;
  LlpsArgs llps;
  IfpArgs ifp;
  AwhArgs awh;
  DhamArgs dham;
  MbarArgs mbar;
  TptArgs tpt;
  CommunityArgs community;
  GraphArgs graph;
  CMapArgs cmap;
  SegmentArgs segments;
  InterfaceAnalysisArgs interface_analysis;
  QNativeArgs qnative;
  DihedralArgs dihedral;
  MsdArgs msd;
  MembraneArgs membrane;
  S2Args s2;
  UmapArgs umap;
  DsspArgs dssp;
  RmsArgs rms;
  RgArgs rg;
  LjArgs lj;
  SqArgs sq;
  CorrKnnArgs corr_knn;
  ValidationArgs validation;
  HBondKineticsArgs hbk;
  ZtsArgs zts;
  EDecompArgs edecomp;
  HotspotReportArgs hotspot;
  std::string eq_weight_source{"auto"};
  // continuous: ordinary MD with physically contiguous frames.
  // remd_state: fixed-temperature REMD streams with exchange discontinuities.
  std::string trajectory_semantics{"continuous"};

  // LSP-MD (Kornev, JCTC 2025) — Local Spatial Pattern network analysis
  struct LSPMDArgs {
    bool flag_lspmd = false;    // -lspmd: enable module
    double lspmd_cutoff = 12.0; // -lspmd_cut: Cα–Cα cutoff (Å)
    bool lspmd_use_avg = true;  // use average distance for cutoff check
  };
  LSPMDArgs lspmd;

  // Core flags
  bool flag_gbsa, flag_alpb;
  double qha_temperature{300.0};
  bool ml_predict_experimental{false};
  bool cal_dis_sc_contact, opdb, file_par, new_al_1pdb, entropy, entropy_quasi,
      flag_index, flag_topo_1, flag_topo_2, flag_1_pdb, flag_listfile,
      flag_use_common_topo, flag_nocal, flag_pattern_input, flag_range_input,
      help, flag_network, flag_network_explicit, flag_pca, flag_tda, flag_ml,
      flag_cas, flag_allostery;
  // Transitional engine-facing energy decomposition fields. New CLI input
  // must use EDecompArgs; remove these after the engine migration lands.
  bool flag_all_folders, flag_separation_results, flag_sidechain_hbond,
      flag_cal_force, flag_use_CI, print_force_res, flag_begin_set,
      flag_end_set, flag_maxforce, self_int, flag_sidechain,
      flag_print_sidechain, flag_all_respair;
  bool flag_sub_groups, flag_cal_dis_res_2com, flag_set_type_1, flag_set_type_2,
      flag_cal_dipole_moment, flag_charged_atom_distance_dis,
      flag_angle_dipole_efield, flag_sep_opdb, heat_capacity_cp, flag_cal_vol,
      flag_elec_screening, flag_interchain_energy, flag_cal_mindis,
      mol_1_type_explicit, mol_2_type_explicit, flag_cat_result_multi_infile,
      opdb_name_explicit, flag_groups_explicit;
  int igb;
  double gbnsr6_spacing{0.5};
  int flag_pqr, flag_all_side_chain_ct, begin_mol_2, force_map, argc, group_1,
      group_2, flag_h_bond, group_entropy_com, group_entropy_rec,
      group_entropy_lg, k_neighbor, dt, b_input, e_input, step_input,
      flag_cache, com_dis_energy_2mol, com_dis_2mol, n_wat;
  int fes_grid;
  int allostery_tau;
  int allostery_bins;
  int allostery_knn;
  double allostery_cutoff;
  double allostery_knn_contact_cutoff;
  bool allostery_use_avg_cutoff;
  bool allostery_knn_use_contact_prior;
  bool allostery_compute_te;
  std::string allostery_estimator;
  int allostery_estimator_k;
  double allostery_kde_bandwidth;
  std::string mem_budget_gb_raw{"auto"};
  double mem_budget_gb{-1.0};
  bool low_memory{false};
  std::string gbsa_perf{"summary"};
  int gbsa_sasa_points{96};
  // Cutoff (Angstrom) on the HCT pairwise descreening integral used for the
  // effective Born radii. Amber's egb default is 25.0; TITAN defaults to
  // "no cutoff" so existing results stay reproducible, and -gbsa-rgbmax 25
  // selects Amber parity. Any finite value changes DeltaG_polar.
  double gbsa_rgbmax{std::numeric_limits<double>::infinity()};
  // Nonpolar surface tension in kcal/mol/A^2 applied to the buried SASA.
  // 0.0072 is MMPBSA.py's GB default; sander's gbsa=1 default is 0.005.
  double gbsa_surften{0.0072};
  double gbsa_temperature{298.15};
  std::string gbsa_ie_block{"auto"};
  int gbsa_ie_min_samples{10};
  double gbsa_ie_filter_sigma{0.0};
  std::string network_mode{"auto"};
  double network_sparse_cut{30.0};
  int network_sparse_k{24};
  std::string network_cache_precision{"auto"};
  bool fflib_auto{true};
  std::string fflib_missing{"warn"};
  std::string fflib_hydrogens{"reference"};
  std::string sg_mode{"auto"};
  std::string sg_cache{"auto"};
  std::string sg_memory_gb_raw{"auto"};
  double sg_memory_gb{-1.0};
  std::string sg_timeseries{"auto"};
  std::string sg_kernel{"auto"};
  std::string sg_cutoff{"auto"};
  double sg_cutoff_value_nm{-1.0};
  size_t tda_max_points{200000};
  // XDR trajectory (XTC/TRR)
  string traj_file;
  bool flag_xdr;
  string xdr_type;
  vector<int> sub_group_1, sub_group_2;
  size_t number_of_files, nt, number_charged_distances_bin,
      number_bin_angle_dipole_efield, number_mindis_bin;
  double ext_diel, int_diel, saltcon;
  double d_standard, begin_time, end_time, angle_h, dis_h, limit_dis, eu_par,
      fu_par, dis_clust, d_crit_water, distance_energy_cutoff{-1.1},
      d_crit_minmax_mol{4.}, charged_distances_max, charged_distances_min,
      salt_concentration, epsilon_in;
  string topo_name, topo_name_2, pdb_name, out_pdb_name, index_name, force_f,
      command, path_amino_acid, path_nonbonded, name_out_cal_dis, arg_filename,
      list_filename, fmap_type, input_pattern, fmap_level, out_motif, eu, fu,
      ref_file, tu, input_file_type, mol_1_type, mol_2_type, water_type;

  arguments()
      : // Core bools
        flag_gbsa{false}, flag_alpb{false}, cal_dis_sc_contact{false},
        opdb{false}, file_par{false}, new_al_1pdb{false}, entropy{false},
        entropy_quasi{false}, flag_index{false}, flag_topo_1{false},
        flag_topo_2{false}, flag_1_pdb{false}, flag_listfile{false},
        flag_use_common_topo{false}, flag_nocal{false},
        flag_pattern_input{false}, flag_range_input{false}, help{false},
        flag_network{false}, flag_network_explicit{false}, flag_pca{false},
        flag_tda{false}, flag_ml{false}, flag_cas{false}, flag_allostery{false},
        flag_all_folders{false}, flag_separation_results{false},
        flag_sidechain_hbond{false}, flag_cal_force{false}, flag_use_CI{false},
        print_force_res{false}, flag_begin_set{false}, flag_end_set{false},
        flag_maxforce{false}, self_int{false}, flag_sidechain{true},
        flag_print_sidechain{true}, flag_all_respair{false},
        flag_sub_groups{false}, flag_cal_dis_res_2com{false},
        flag_set_type_1{false}, flag_set_type_2{false},
        flag_cal_dipole_moment{false}, flag_charged_atom_distance_dis{false},
        flag_angle_dipole_efield{false}, flag_sep_opdb{false},
        heat_capacity_cp{false}, flag_cal_vol{false},
        flag_elec_screening{false}, flag_interchain_energy{false},
        flag_cal_mindis{false}, mol_1_type_explicit{false},
        mol_2_type_explicit{false}, flag_cat_result_multi_infile{false},
        opdb_name_explicit{false}, flag_groups_explicit{false},
        // Core ints
        igb{5}, flag_pqr{2}, flag_all_side_chain_ct{1}, begin_mol_2{},
        force_map{2}, argc{1}, group_1{}, group_2{}, flag_h_bond{0},
        group_entropy_com{-1}, group_entropy_rec{-1}, group_entropy_lg{-1},
        k_neighbor{4}, dt{0}, b_input{}, e_input{100000000}, step_input{1},
        flag_cache{1}, com_dis_energy_2mol{4}, com_dis_2mol{0}, n_wat{-1},
        fes_grid{60}, allostery_tau{5}, allostery_bins{-1}, allostery_knn{0},
        allostery_cutoff{0.0}, allostery_knn_contact_cutoff{8.0},
        allostery_use_avg_cutoff{true}, allostery_knn_use_contact_prior{false},
        allostery_compute_te{true}, allostery_estimator{"hist"},
        allostery_estimator_k{4}, allostery_kde_bandwidth{0.0},
        // XDR
        traj_file{}, flag_xdr{false}, xdr_type{},
        // size_t
        number_of_files{1ull}, nt{}, number_charged_distances_bin{400},
        number_bin_angle_dipole_efield{200}, number_mindis_bin{100},
        // doubles
        ext_diel{78.5}, int_diel{1.0}, saltcon{0.15}, d_standard{6.5},
        begin_time{0.}, end_time{100000000000.}, angle_h{135.}, dis_h{3.5},
        limit_dis{10e10}, eu_par{1.0}, fu_par{1.66}, dis_clust{6.5},
        d_crit_water{3.5}, distance_energy_cutoff{-1.1}, d_crit_minmax_mol{4.},
        charged_distances_max{50.}, charged_distances_min{0.},
        salt_concentration{0.15}, epsilon_in{1.00000000},
        // strings
        topo_name{"lig.itp"}, topo_name_2{"no"}, pdb_name{"complex.pdb"},
        out_pdb_name{"out_complex.pdb"}, index_name{"index.ndx"},
        force_f{"AMBER99SB"}, command{"-"}, path_amino_acid{"lib_ff/"},
        path_nonbonded{"lib_ff/"}, name_out_cal_dis{"out_cal_dis_com_"},
        arg_filename{"arg.in"}, list_filename{"input.txt"},
        fmap_type{"residue-atom"}, input_pattern{}, fmap_level{"per-res"},
        out_motif{"*"}, eu{"kJ/mol"}, fu{"pN"}, ref_file{}, tu{"ps"},
        input_file_type{".pdb"}, mol_1_type{"protein"}, mol_2_type{"lig"},
        water_type{"surface"} {}

  int post_process();
  bool needs_mol2_for_cmap_interface() const {
    return cmap.flag_cmap && cmap.scope == "interface";
  }
  bool needs_mol2_for_qnative_interface() const {
    return qnative.flag_qnative && qnative.scope == "interface";
  }
  bool needs_mol2_for_interface_analysis() const {
    return interface_analysis.flag_interface_analysis;
  }
  bool needs_mol2_for_msd() const {
    return msd.flag_msd && msd.selection == "mol2";
  }
  bool needs_membrane_coordinates() const { return membrane.flag_membrane; }
  bool needs_mol2_for_nocal_output() const {
    return flag_nocal && ((flag_groups_explicit && group_1 != group_2) ||
                          begin_mol_2 > 0 || com_dis_2mol != 0);
  }
  bool needs_mol2_coordinates() const {
    return !flag_nocal || needs_mol2_for_nocal_output() ||
           needs_mol2_for_cmap_interface() ||
           needs_mol2_for_qnative_interface() ||
           needs_mol2_for_interface_analysis() || needs_mol2_for_msd();
  }
  string get_build_number() const;
  int parse_arguments(int argc, char **argv);
  int parse_arguments_interactive(int argc);
  int parse_arg_file();
  bool check_topo(string NameofTopoFile, int Topo);
  int help_print();
  void set_number_of_files(size_t num) { number_of_files = num; }
  void setPdbname(string name) { pdb_name = name; }
  string get_pdbOutput() const;
  int recheck_and_change_pdb_input();
  string get_input_file_type() const { return input_file_type; }
  int add_input_file_type() {
    if (pdb_name.length() <= 4 ||
        pdb_name.substr(pdb_name.length() - 4) != get_input_file_type()) {
      pdb_name += get_input_file_type();
      return 0;
    } else
      return 1;
  }
};
#endif
