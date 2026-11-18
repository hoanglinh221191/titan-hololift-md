#ifndef TITAN_PBE_ANALYZER_IMPL_H
#define TITAN_PBE_ANALYZER_IMPL_H

#include "../MOL.h"
#include "../MOL_para.h"
#include "../aligned_allocator.h"
#include "../arguments.h"
#include "../titan_pbe.h"
#include "pbe/grid/grid_solver.h"
#include "pbe/grid/grid_workspace.h"
#include "types.h"
#include "tabi_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

class TitanExecutor;
class EnergyDecompositionProvider;

namespace TitanPBE {

class AnalyzerImpl {
public:
  AnalyzerImpl() = default;

  void configure(const arguments &args);
  [[nodiscard]] bool enabled() const noexcept { return args_.flag_pbe; }
  void set_edecomp_sink(EDecompSink sink) { edecomp_sink_ = std::move(sink); }

  void begin_trajectory(std::size_t trajectory_id,
                        const std::string &source_label);
  void end_trajectory(std::fstream &log);

  FrameResult consume_frame(const MOL &mol1, const Multi_MOL_parameters &params1,
                            const MOL *mol2,
                            const Multi_MOL_parameters *params2,
                            std::size_t trajectory_id,
                            std::size_t frame_index,
                            std::size_t global_index,
                            TitanExecutor &executor, std::fstream &log,
                            const EnergyDecompositionProvider *energy_provider =
                                nullptr);

  void finalize(std::fstream &log);
  // Non-empty when a BIE frame did not succeed: the run wrote its outputs
  // but must not report success.
  [[nodiscard]] std::string failure_summary() const;

private:
  struct Atom {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double q{0.0};
    double radius{0.0};
    std::string element;
    std::string atom_name;
    std::string resname;
    int resnr{0};
    std::string chain;
    int serial{0};
  };

  struct GridShape {
    double ox{0.0};
    double oy{0.0};
    double oz{0.0};
    double h{0.5};
    std::size_t nx{0};
    std::size_t ny{0};
    std::size_t nz{0};
    std::size_t n{0};
    std::size_t memory_bytes{0};
  };

  struct SolveStats {
    int iterations{0};
    double rms{0.0};
    double max_dphi{0.0};
    bool converged{false};
  };

  struct NewtonCorrectionStats {
    SolveStats sweeps{};
    double residual_rms{0.0};
    double residual_max{0.0};
    double jac_diag_min{0.0};
    double jac_diag_max{0.0};
    std::size_t screen_points{0};
    std::size_t active_nonlinear_points{0};
    int inner_levels{1};
    int inner_cycles{0};
    double inner_residual_before{0.0};
    double inner_residual_after{0.0};
    std::string inner_effective{"sor"};
    bool residual_decreased{false};
  };

  struct ChargeDiagnostics {
    double assigned_charge{0.0};
    double moment_error{0.0};
  };

  struct BoundaryDiagnostics {
    double rms{0.0};
    double max_abs{0.0};
  };

  struct BoundaryFormulaSample {
    double native_real_phi{0.0};
    double delphi_grid_phi_raw{0.0};
    double selected_phi{0.0};
    double atom_grid_x{0.0};
    double atom_grid_y{0.0};
    double atom_grid_z{0.0};
    double nearest_atom_grid_distance{0.0};
    double nearest_atom_distance_a{0.0};
    double epsilon_temp{0.0};
    double debye_length{0.0};
    double epkt{1.0};
  };

  struct MapSamplePoint {
    std::string kind{};
    std::size_t rank{0};
    std::size_t id{0};
    double hint{0.0};
  };

  enum class TimingBucket {
    Setup,
    Surface,
    Epsilon,
    Boundary,
    Solver,
    Energy,
    MapIo,
    RpbSource,
    FocusSetup,
    Probe,
    FocusParentSolve,
    FocusChildSetup,
    FocusBoundaryInterp,
    FocusChildSurface,
    FocusChildEpsilon,
    FocusChildSolver,
    ProbeEnergy,
    AuditOutput,
    NwtHalfUpdate,
    NwtRmsReduce,
    NwtOperatorResidual,
    NwtSourceUpdate,
    NwtProgressIo,
    NwtCoeffPack,
    NwtHomoUpdate,
    NwtHeteroUpdate,
    NwtNonlinTerm,
    NwtReduce,
    NwtHomoStencil,
    NwtHomoSaltMath,
    NwtHomoWrite,
    NwtThreadWait,
    NwtBlockOverhead,
    LinearHalfUpdate,
    EpsilonCenterDensity,
    EpsilonMidpointDensity,
    EpsilonIonExclusion,
    EpsilonFinalize,
  };

  struct TimingBuckets {
    double setup_ms{0.0};
    double surface_ms{0.0};
    double epsilon_ms{0.0};
    double boundary_ms{0.0};
    double solver_ms{0.0};
    double energy_ms{0.0};
    double map_io_ms{0.0};
    double rpb_source_ms{0.0};
    double focus_setup_ms{0.0};
    double probe_ms{0.0};
    double focus_parent_solve_ms{0.0};
    double focus_child_setup_ms{0.0};
    double focus_boundary_interp_ms{0.0};
    double focus_child_surface_ms{0.0};
    double focus_child_epsilon_ms{0.0};
    double focus_child_solver_ms{0.0};
    double probe_energy_ms{0.0};
    double audit_output_ms{0.0};
    std::size_t focus_child_atom_count{0};
    double nwt_half_update_ms{0.0};
    double nwt_rms_reduce_ms{0.0};
    double nwt_operator_residual_ms{0.0};
    double nwt_source_update_ms{0.0};
    double nwt_progress_io_ms{0.0};
    double nwt_coeff_pack_ms{0.0};
    double nwt_homo_update_ms{0.0};
    double nwt_hetero_update_ms{0.0};
    double nwt_nonlin_term_ms{0.0};
    double nwt_reduce_ms{0.0};
    double nwt_homo_stencil_ms{0.0};
    double nwt_homo_salt_math_ms{0.0};
    double nwt_homo_write_ms{0.0};
    double nwt_thread_wait_ms{0.0};
    double nwt_block_overhead_ms{0.0};
    double linear_half_update_ms{0.0};
    std::uint64_t nwt_homo_cells{0};
    std::uint64_t nwt_homo_linear_cells{0};
    std::uint64_t nwt_homo_salt_cells{0};
    std::uint64_t nwt_homo_updates{0};
    std::uint64_t nwt_dispatch_calls{0};
    double epsilon_center_density_ms{0.0};
    double epsilon_midpoint_density_ms{0.0};
    double epsilon_ion_exclusion_ms{0.0};
    double epsilon_finalize_ms{0.0};
    double total_ms{0.0};
  };

  struct StabilityAtomSample {
    std::size_t atom_index{0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double q{0.0};
    double radius{0.0};
    double grid_x{0.0};
    double grid_y{0.0};
    double grid_z{0.0};
    double min_integer_grid_units{0.0};
    double min_half_grid_units{0.0};
    double min_grid_plane_A{0.0};
    std::string atom_name;
    std::string resname;
    int resnr{0};
    std::string chain;
    std::string stage_hint{"off"};
  };

  struct StabilityFrameState {
    bool valid{false};
    std::size_t nx{0};
    std::size_t ny{0};
    std::size_t nz{0};
    double h{0.0};
    double origin_x{0.0};
    double origin_y{0.0};
    double origin_z{0.0};
    double center_x{0.0};
    double center_y{0.0};
    double center_z{0.0};
    double energy_kj_mol{0.0};
    std::size_t solute_points{0};
    std::size_t solvent_points{0};
    std::size_t ion_accessible_points{0};
    std::size_t stern_excluded_points{0};
    std::size_t iscm_boundary_points{0};
    std::size_t iscm_charged_boundary_points{0};
    std::size_t vdwms_added{0};
    std::size_t vdwms_removed{0};
    double charge_assigned{0.0};
    double charge_moment_error{0.0};
    double boundary_rms{0.0};
    double boundary_max{0.0};
  };

  struct StabilityMediumState {
    bool valid{false};
    std::size_t nx{0};
    std::size_t ny{0};
    std::size_t nz{0};
    double h{0.0};
    double origin_x{0.0};
    double origin_y{0.0};
    double origin_z{0.0};
    std::vector<unsigned char> inside;
    std::vector<unsigned char> ion_accessible;
    AlignedDoubleVector eps_x;
    AlignedDoubleVector eps_y;
    AlignedDoubleVector eps_z;
  };

  struct StabilityEnsembleState {
    bool valid{false};
    std::size_t samples{0};
    double mean_kj_mol{0.0};
    double stddev_kj_mol{0.0};
    double min_kj_mol{0.0};
    double max_kj_mol{0.0};
  };

  struct StabilityAudit {
    std::string stability_mode{"off"};
    std::string grid_anchor_effective{"per-frame"};
    std::string anchor_status{"off"};
    double anchor_margin_min_A{0.0};
    std::size_t hysteresis_cells_kept{0};
    std::size_t hysteresis_cells_released{0};
    double native_energy_kj_mol{0.0};
    double stable_energy_kj_mol{0.0};
    double energy_delta_vs_native_kj_mol{0.0};
    std::size_t ensemble_samples{0};
    double ensemble_mean_kj_mol{0.0};
    double ensemble_stddev_kj_mol{0.0};
    double ensemble_min_kj_mol{0.0};
    double ensemble_max_kj_mol{0.0};
    double grid_center_delta_A{0.0};
    double grid_origin_delta_A{0.0};
    double energy_delta_kj_mol{0.0};
    double min_integer_grid_units{0.0};
    double min_half_grid_units{0.0};
    double min_grid_plane_A{0.0};
    std::size_t near_integer_atoms_1e6{0};
    std::size_t near_integer_atoms_1e4{0};
    std::size_t near_integer_atoms_1e3{0};
    std::size_t near_half_atoms_1e6{0};
    std::size_t near_half_atoms_1e4{0};
    std::size_t near_half_atoms_1e3{0};
    std::size_t charge_cell_flip_candidates{0};
    std::size_t surface_flip_candidates{0};
    std::size_t midpoint_flip_candidates{0};
    std::size_t ion_access_flip_candidates{0};
    std::int64_t solute_delta{0};
    std::int64_t solvent_delta{0};
    std::int64_t ion_accessible_delta{0};
    std::int64_t stern_excluded_delta{0};
    std::int64_t iscm_boundary_delta{0};
    std::int64_t iscm_charged_boundary_delta{0};
    std::int64_t vdwms_added_delta{0};
    std::int64_t vdwms_removed_delta{0};
    double charge_assigned_delta{0.0};
    double charge_moment_delta{0.0};
    double boundary_rms_delta{0.0};
    double boundary_max_delta{0.0};
    std::string suspect_stage{"first_frame"};
    std::vector<StabilityAtomSample> samples;
  };

  struct NwtHalfWorkspace {
    struct PackedCell {
      std::uint32_t h{0};
      std::uint32_t xp{0};
      std::uint32_t xm{0};
      std::uint32_t yp{0};
      std::uint32_t ym{0};
      std::uint32_t zp{0};
      std::uint32_t zm{0};
      double rhs{0.0};
      double eps_sum{1.0};
      double screen{0.0};
      double ex_p{0.0};
      double ex_m{0.0};
      double ey_p{0.0};
      double ey_m{0.0};
      double ez_p{0.0};
      double ez_m{0.0};
    };
    struct HomoSpan {
      std::uint32_t begin{0};
      std::uint32_t count{0};
    };
    struct HomoTile {
      std::uint32_t begin{0};
      std::uint32_t count{0};
    };

    GridShape shape{};
    bool active{false};
    bool coeff_ready{false};
    bool packed_ready{false};
    bool row_span_ready{false};
    bool salt_tile_ready{false};
    double rhs_scale{0.0};
    std::uint64_t homo_cells{0};
    std::uint64_t homo_linear_cells{0};
    std::uint64_t homo_salt_cells{0};
    AlignedDoubleVector even_phi;
    AlignedDoubleVector odd_phi;
    AlignedDoubleVector rhs_even;
    AlignedDoubleVector rhs_odd;
    AlignedDoubleVector eps_sum_even;
    AlignedDoubleVector eps_sum_odd;
    AlignedDoubleVector screen_even;
    AlignedDoubleVector screen_odd;
    AlignedDoubleVector homo_eps_even;
    AlignedDoubleVector homo_eps_odd;
    std::vector<PackedCell> even_homo_cells;
    std::vector<PackedCell> even_hetero_cells;
    std::vector<PackedCell> odd_homo_cells;
    std::vector<PackedCell> odd_hetero_cells;
    std::vector<HomoSpan> even_homo_linear_spans;
    std::vector<HomoSpan> even_homo_salt_spans;
    std::vector<HomoSpan> odd_homo_linear_spans;
    std::vector<HomoSpan> odd_homo_salt_spans;
    std::vector<HomoTile> even_homo_linear_tiles;
    std::vector<HomoTile> even_homo_salt_tiles;
    std::vector<HomoTile> odd_homo_linear_tiles;
    std::vector<HomoTile> odd_homo_salt_tiles;
    std::vector<double> local_sumsq;
    std::vector<double> local_max;
    std::vector<std::size_t> local_count;
    std::vector<std::size_t> local_clipped;
  };

  struct ParityAudit {
    double eps_sum_min{0.0};
    double eps_sum_max{0.0};
    double eps_sum_mean{0.0};
    double midpoint_eps_sum_min{0.0};
    double midpoint_eps_sum_max{0.0};
    double midpoint_eps_sum_mean{0.0};
    double midpoint_eps_sum_interior_mean{0.0};
    std::size_t screen_points{0};
    double screen_sum{0.0};
    double screen_rms{0.0};
    double screen_max{0.0};
    double source_sum{0.0};
    double source_rms{0.0};
    double source_max{0.0};
    double residual_max_abs{0.0};
    double derivative_max_abs{0.0};
    double derivative_max_rel{0.0};
    std::size_t derivative_samples{0};
  };

  struct ReactionSolve {
    GridShape shape{};
    SolveStats solvent{};
    SolveStats vacuum{};
    int solvent_total_iterations{0};
    int vacuum_total_iterations{0};
    int focus_levels{1};
    double focus_initial_scale{0.0};
    double focus_initial_margin{0.0};
    std::string solver_mode{"sor"};
    int mg_levels{1};
    int mg_cycles{0};
    int mg_pre{0};
    int mg_post{0};
    int mg_coarse_iter{0};
    std::size_t solute_points{0};
    std::size_t solvent_points{0};
    std::size_t ion_accessible_points{0};
    std::size_t stern_excluded_points{0};
    std::size_t peak_memory_bytes{0};
    TimingBuckets pbsa_solvent_timing{};
    TimingBuckets pbsa_vacuum_timing{};
    TimingBuckets pbsa_reuse_timing{};
    double pbsa_energy_ms{0.0};
    double pbsa_reuse_delta_kj_mol{0.0};
    std::size_t pbsa_reuse_surface_hits{0};
    std::size_t pbsa_reuse_operator_hits{0};
    std::string pbe_backend{"grid"};
    double tabi_surface_build_ms{0.0};
    std::size_t tabi_surface_panels{0};
    int tabi_surface_point_count_requested{0};
    double tabi_panel_density_per_a2{0.0};
    std::string tabi_panel_budget_status{"off"};
    ChargeResolutionDiagnostic tabi_charge_resolution;
    std::size_t tabi_surface_exposed_points{0};
    std::size_t tabi_surface_buried_points{0};
    double tabi_tree_build_ms{0.0};
    std::size_t tabi_tree_nodes{0};
    std::size_t tabi_tree_leaf_count{0};
    std::size_t tabi_near_interactions{0};
    std::size_t tabi_far_interactions{0};
    std::string tabi_matvec_mode{"off"};
    std::string tabi_fmm_m2l_mode{"off"};
    int tabi_fmm_m2l_expansion_order{0};
    std::size_t tabi_fmm_m2l_near_pairs{0};
    std::size_t tabi_fmm_m2l_far_pairs{0};
    double tabi_fmm_m2l_setup_ms{0.0};
    double tabi_fmm_m2l_apply_ms{0.0};
    bool tabi_fmm_m2l_finite_outputs{true};
    bool tabi_bie_cartesian_full_fmm_active{false};
    std::string tabi_bie_cartesian_full_fmm_requested{"off"};
    std::string tabi_bie_cartesian_full_fmm_effective{"off"};
    int tabi_bie_cartesian_full_fmm_order{0};
    int tabi_bie_cartesian_full_fmm_leaf_size{0};
    double tabi_bie_cartesian_full_fmm_opening_theta{0.0};
    std::uint64_t tabi_bie_cartesian_full_fmm_tree_hash{0};
    std::uint64_t tabi_bie_cartesian_full_fmm_worklist_hash{0};
    std::uint64_t tabi_bie_cartesian_full_fmm_policy_hash{0};
    std::size_t tabi_bie_cartesian_full_fmm_nodes{0};
    std::size_t tabi_bie_cartesian_full_fmm_far_pairs{0};
    std::size_t tabi_bie_cartesian_full_fmm_direct_pairs{0};
    std::uint64_t tabi_bie_cartesian_full_fmm_direct_panel_pairs_expected{0};
    std::uint64_t tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_last{0};
    std::uint64_t tabi_bie_cartesian_full_fmm_direct_panel_pairs_applied_total{0};
    std::size_t tabi_bie_cartesian_full_fmm_cache_entries{0};
    std::size_t tabi_bie_cartesian_full_fmm_cache_hits{0};
    std::size_t tabi_bie_cartesian_full_fmm_cache_bytes{0};
    std::size_t tabi_bie_cartesian_full_fmm_plan_bytes{0};
    std::size_t tabi_bie_cartesian_full_fmm_workspace_bytes{0};
    double tabi_bie_cartesian_full_fmm_tree_seconds{0.0};
    double tabi_bie_cartesian_full_fmm_worklist_seconds{0.0};
    double tabi_bie_cartesian_full_fmm_cache_seconds{0.0};
    double tabi_bie_cartesian_full_fmm_audit_seconds{0.0};
    double tabi_bie_cartesian_full_fmm_p2m_seconds{0.0};
    double tabi_bie_cartesian_full_fmm_m2m_seconds{0.0};
    double tabi_bie_cartesian_full_fmm_m2l_seconds{0.0};
    double tabi_bie_cartesian_full_fmm_l2l_seconds{0.0};
    double tabi_bie_cartesian_full_fmm_l2p_seconds{0.0};
    double tabi_bie_cartesian_full_fmm_direct_seconds{0.0};
    double tabi_bie_cartesian_full_fmm_m2m_parity_error{0.0};
    double tabi_bie_cartesian_full_fmm_l2l_parity_error{0.0};
    bool tabi_fmm_local_gmres_active{false};
    double tabi_fmm_local_prepare_seconds{0.0};
    std::size_t tabi_fmm_local_plan_build_count{0};
    std::size_t tabi_fmm_local_cache_build_count{0};
    std::size_t tabi_fmm_local_gmres_matvec_calls{0};
    double tabi_fmm_local_gmres_apply_seconds_total{0.0};
    double tabi_fmm_local_gmres_apply_seconds_mean{0.0};
    double tabi_fmm_local_moment_seconds_total{0.0};
    double tabi_fmm_local_m2l_seconds_total{0.0};
    double tabi_fmm_local_fallback_seconds_total{0.0};
    double tabi_fmm_local_l2p_seconds_total{0.0};
    double tabi_fmm_local_sampled_direct_true_residual{0.0};
    bool tabi_fmm_local_operator_finite{true};
    std::string tabi_fmm_local_refine_probe_mode{"rademacher"};
    std::size_t tabi_fmm_local_refine_probe_count{4};
    bool tabi_fmm_local_solution_probe_used{false};
    double tabi_fmm_local_linf_max_abs_tail{0.0};
    double tabi_fmm_local_linf_max_relative_tail{0.0};
    std::size_t tabi_fmm_local_linf_outlier_panels{0};
    std::string tabi_fmm_local_geometry_guard{"off"};
    double tabi_fmm_local_geometry_theta{0.0};
    std::size_t tabi_fmm_local_geometry_guard_rejected_cells{0};
    std::size_t tabi_fmm_local_geometry_guard_direct_fallback_pairs{0};
    double tabi_fmm_local_geometry_gap_min{0.0};
    double tabi_fmm_local_geometry_ratio_max{0.0};
    std::string tabi_status{"off"};
    std::string tabi_convergence_reason{"off"};
    std::string tabi_residual_history{};
    int tabi_gmres_restart_requested{32};
    int tabi_gmres_restart_effective{32};
    double tabi_best_residual{0.0};
    int tabi_best_residual_iter{0};
    double tabi_final_residual{0.0};
    double tabi_best_preconditioned_residual{0.0};
    double tabi_final_preconditioned_residual{0.0};
    double tabi_true_residual_abs{0.0};
    double tabi_true_residual_rel{0.0};
    std::string tabi_true_residual_probe_mode{"off"};
    std::string tabi_true_residual_probe_scope{"full"};
    int tabi_true_residual_probe_interval{0};
    int tabi_true_residual_sample_count{0};
    std::size_t tabi_true_residual_probe_calls{0};
    double tabi_true_residual_probe_ms{0.0};
    double tabi_sampled_true_residual_rel{0.0};
    std::size_t tabi_sampled_true_residual_rows{0};
    std::size_t tabi_sampled_probe_calls{0};
    double tabi_sampled_probe_ms{0.0};
    std::size_t tabi_full_true_residual_confirm_calls{0};
    double tabi_full_probe_ms{0.0};
    std::string tabi_reaction_energy_source{"off"};
    double tabi_rhs_ms{0.0};
    double tabi_gmres_ms{0.0};
    double tabi_matvec_ms_total{0.0};
    double tabi_matvec_ms_avg{0.0};
    std::size_t tabi_matvec_calls{0};
    double tabi_tree_vs_direct_matvec_rel{0.0};
    std::size_t tabi_matvec_audit_rows{0};
    double tabi_matvec_audit_ms{0.0};
    FmmCartesianPreAuditTelemetry tabi_fmm_cartesian_pre_audit{};
    FmmCartesianSolutionAuditTelemetry tabi_fmm_cartesian_solution_audit{};
    std::array<FmmCartesianChannelAudit, kJcpExpansionChannelCount>
        tabi_fmm_cartesian_channel_audit{};
    FmmCartesianRetryTelemetry tabi_fmm_cartesian_retry{};
    double tabi_orthogonalization_ms{0.0};
    double tabi_preconditioner_ms{0.0};
    double tabi_surface_neighbor_ms{0.0};
    double tabi_surface_mesh_ms{0.0};
    double tabi_solid_angle_ms{0.0};
    double tabi_tree_moment_ms{0.0};
    double tabi_tree_traversal_ms{0.0};
    std::string tabi_matvec_cost_profile{"off"};
    double tabi_leaf_panels_per_second{0.0};
    double tabi_clean_leaf_panels_per_second{0.0};
    double tabi_tree_traversal_fraction{0.0};
    double tabi_near_eval_ms{0.0};
    double tabi_far_eval_ms{0.0};
    double tabi_singular_near_eval_ms{0.0};
    double tabi_leaf_near_eval_ms{0.0};
    double tabi_leaf_panel_eval_ms{0.0};
    double tabi_leaf_tile_eval_ms{0.0};
    double tabi_leaf_residual_eval_ms{0.0};
    double tabi_smooth_leaf_node_eval_ms{0.0};
    std::size_t tabi_singular_near_interactions{0};
    std::size_t tabi_leaf_near_interactions{0};
    std::size_t tabi_far_node_interactions{0};
    std::size_t tabi_leaf_panel_interactions{0};
    std::size_t tabi_leaf_tile_interactions{0};
    std::size_t tabi_leaf_tile_panel_interactions{0};
    std::size_t tabi_leaf_residual_panel_interactions{0};
    std::size_t tabi_smooth_leaf_node_interactions{0};
    std::size_t tabi_leaf_pruned_interactions{0};
    std::size_t tabi_leaf_mac_candidates{0};
    std::size_t tabi_leaf_mac_accepted{0};
    std::size_t tabi_leaf_mac_rejected{0};
    std::size_t tabi_leaf_mac_far_panels{0};
    std::size_t tabi_leaf_mac_direct_panels_saved_estimate{0};
    std::string tabi_leaf_work_profile{"off"};
    std::size_t tabi_leaf_work_mac_terminal_panels{0};
    std::size_t tabi_leaf_work_physical_gap_panels{0};
    std::size_t tabi_leaf_work_fragmentation_panels{0};
    std::size_t tabi_leaf_work_convergence_panels{0};
    std::size_t tabi_leaf_work_convergence_estimator_panels{0};
    std::size_t tabi_leaf_work_convergence_rho_panels{0};
    std::size_t tabi_leaf_work_convergence_aspect_panels{0};
    std::size_t tabi_leaf_work_surface_margin_panels{0};
    std::size_t tabi_leaf_work_surface_margin_hard_panels{0};
    std::size_t tabi_leaf_work_surface_margin_soft_panels{0};
    std::size_t tabi_leaf_work_k3_rho_cap_panels{0};
    std::size_t tabi_leaf_work_singular_fallback_panels{0};
    std::size_t tabi_leaf_work_scalar_tail_panels{0};
    std::size_t tabi_leaf_work_profile_accounted_panels{0};
    double tabi_leaf_work_mac_terminal_frac{0.0};
    double tabi_leaf_work_convergence_frac{0.0};
    double tabi_leaf_work_surface_margin_frac{0.0};
    double tabi_leaf_work_singular_fallback_frac{0.0};
    double tabi_leaf_work_scalar_tail_frac{0.0};
    std::size_t tabi_adaptive_smooth_leaf_accepted{0};
    std::size_t tabi_adaptive_smooth_leaf_rejected{0};
    std::size_t tabi_adaptive_span_tile_accepted{0};
    std::size_t tabi_adaptive_span_tile_rejected{0};
    std::size_t tabi_adaptive_exact_leaf_fallbacks{0};
    std::size_t tabi_adaptive_reject_one_ring_overlap{0};
    std::size_t tabi_adaptive_reject_error_bound_failed{0};
    std::size_t tabi_adaptive_reject_p4_check_failed{0};
    std::size_t tabi_adaptive_reject_too_close{0};
    double tabi_leaf_work_reduction_ratio{0.0};
    std::size_t tabi_target_cluster_interactions{0};
    std::size_t tabi_target_cluster_targets{0};
    std::size_t tabi_target_cluster_panels_pruned{0};
    std::size_t tabi_target_cluster_rejected{0};
    double tabi_target_cluster_eval_ms{0.0};
    std::size_t tabi_leaf_pair_batch_interactions{0};
    std::size_t tabi_leaf_pair_batch_targets{0};
    std::size_t tabi_leaf_pair_batch_source_panels{0};
    std::size_t tabi_leaf_pair_batch_residual_panels{0};
    std::size_t tabi_leaf_pair_batch_fallbacks{0};
    double tabi_leaf_pair_batch_ms{0.0};
    std::string tabi_leaf_pair_batch_kernel{"off"};
    std::size_t tabi_leaf_pair_batch_vectorized_sources{0};
    std::size_t tabi_leaf_pair_batch_vector_lanes{0};
    double tabi_leaf_pair_batch_pairs_per_second{0.0};
    std::string tabi_leaf_kernel_profile{"off"};
    std::string tabi_jcp_direct_kernel{"baseline"};
    std::size_t tabi_leaf_kernel_scalar_tasks{0};
    std::size_t tabi_leaf_kernel_batch_tasks{0};
    std::size_t tabi_leaf_kernel_source_panel_evals{0};
    double tabi_leaf_kernel_scalar_ms{0.0};
    double tabi_leaf_kernel_target_avx2_ms{0.0};
    double tabi_leaf_kernel_tail_ms{0.0};
    std::size_t tabi_leaf_pair_batch_soa_sources{0};
    double tabi_leaf_pair_batch_soa_ms{0.0};
    double tabi_leaf_pair_batch_scalar_tail_ms{0.0};
    double tabi_leaf_pair_batch_clean_source_panels_per_second{0.0};
    std::string tabi_leaf_pair_selection_profile{"off"};
    std::string tabi_leaf_pair_probe_cache{"off"};
    std::string tabi_leaf_pair_target_group{"scan"};
    int tabi_leaf_pair_max_lookahead{32};
    std::size_t tabi_leaf_pair_selection_candidate_scans{0};
    std::size_t tabi_leaf_pair_selection_clean_rejects{0};
    std::size_t tabi_leaf_pair_selection_probe_calls{0};
    std::size_t tabi_leaf_pair_selection_probe_node_visits{0};
    double tabi_leaf_pair_selection_probe_ms{0.0};
    double tabi_leaf_pair_selection_ms{0.0};
    std::size_t tabi_leaf_pair_selection_probe_cache_hits{0};
    std::size_t tabi_leaf_pair_selection_probe_cache_misses{0};
    std::size_t tabi_leaf_pair_selection_insufficient_targets{0};
    std::size_t tabi_leaf_pair_selection_same_target_leaf_rejects{0};
    std::string tabi_leaf_pair_directness_probe{"full-traversal"};
    std::string tabi_leaf_pair_directness_cache{"off"};
    std::string tabi_leaf_pair_directness_verify{"off"};
    std::size_t tabi_leaf_pair_directness_ancestor_probe_calls{0};
    std::size_t tabi_leaf_pair_directness_ancestor_probe_node_visits{0};
    double tabi_leaf_pair_directness_ancestor_probe_ms{0.0};
    std::size_t tabi_leaf_pair_directness_pair_cache_hits{0};
    std::size_t tabi_leaf_pair_directness_pair_cache_misses{0};
    std::size_t tabi_leaf_pair_directness_all_direct_proofs{0};
    std::size_t tabi_leaf_pair_directness_mixed_fallbacks{0};
    std::size_t tabi_leaf_pair_directness_verify_checks{0};
    std::size_t tabi_leaf_pair_directness_verify_mismatches{0};
    std::string tabi_leaf_pair_scheduler{"inline-select"};
    int tabi_leaf_pair_worklist_max_active{2048};
    std::string tabi_leaf_pair_worklist_verify{"off"};
    std::size_t tabi_leaf_pair_worklist_tasks{0};
    std::size_t tabi_leaf_pair_worklist_groups{0};
    std::size_t tabi_leaf_pair_worklist_batched_groups{0};
    std::size_t tabi_leaf_pair_worklist_scalar_flush_tasks{0};
    std::size_t tabi_leaf_pair_worklist_forced_flushes{0};
    std::size_t tabi_leaf_pair_worklist_active_groups_max{0};
    double tabi_leaf_pair_worklist_ms{0.0};
    std::size_t tabi_leaf_pair_worklist_verify_checks{0};
    std::size_t tabi_leaf_pair_worklist_verify_mismatches{0};
    std::string tabi_leaf_block_verify{"off"};
    std::size_t tabi_leaf_block_target_leaves{0};
    std::size_t tabi_leaf_block_targets{0};
    std::size_t tabi_leaf_block_source_groups{0};
    std::size_t tabi_leaf_block_batched_groups{0};
    std::size_t tabi_leaf_block_scalar_tail_tasks{0};
    double tabi_leaf_block_group_ms{0.0};
    double tabi_leaf_block_flush_ms{0.0};
    double tabi_leaf_block_writeback_ms{0.0};
    std::size_t tabi_leaf_block_verify_checks{0};
    std::size_t tabi_leaf_block_verify_mismatches{0};
    std::string tabi_tree_preset{"off"};
    std::string tabi_adaptive_level{"off"};
    std::string tabi_cluster_policy{"off"};
    int tabi_target_batch_size{1};
    int tabi_leaf_pair_min_source_panels{64};
    int tabi_leaf_pair_min_targets{4};
    std::string tabi_leaf_near_policy{"off"};
    std::string tabi_leaf_panel_kernel{"off"};
    std::string tabi_interaction_mode{"off"};
    std::string tabi_node_bounds_mode{"off"};
    std::string tabi_tree_builder{"off"};
    std::string tabi_expansion_center{"off"};
    std::string tabi_fragmentation_safety{"off"};
    std::string tabi_mac_safety{"off"};
    double tabi_mac_gap_factor{1.5};
    std::size_t tabi_tree_depth_max{0};
    std::size_t tabi_mac_reject_radius_limited{0};
    double tabi_mac_accept_rate{0.0};
    double tabi_node_radius_max{0.0};
    double tabi_node_radius_mean{0.0};
    double tabi_node_aspect_ratio_max{0.0};
    double tabi_node_aspect_ratio_mean{0.0};
    double tabi_leaf_panel_work_per_matvec{0.0};
    double tabi_far_node_work_per_matvec{0.0};
    std::size_t tabi_far_geometric_unsafe_count{0};
    std::size_t tabi_far_geometric_unsafe_panels{0};
    double tabi_far_geometric_min_gap{0.0};
    double tabi_far_geometric_gap_ratio_min{0.0};
    double tabi_far_geometric_gap_ratio_mean{0.0};
    std::size_t tabi_far_geometric_audit_samples{0};
    std::size_t tabi_mac_physical_gap_rejects{0};
    std::size_t tabi_mac_convergence_rho_rejects{0};
    std::size_t tabi_mac_convergence_aspect_rejects{0};
    std::size_t tabi_mac_convergence_rho_reject_panels{0};
    std::size_t tabi_mac_convergence_aspect_reject_panels{0};
    std::size_t tabi_mac_surface_margin_rejects{0};
    std::size_t tabi_mac_surface_margin_reject_panels{0};
    std::size_t tabi_mac_surface_margin_applied{0};
    std::size_t tabi_mac_surface_margin_skipped{0};
    std::size_t tabi_mac_surface_margin_adaptive_rejects{0};
    std::size_t tabi_mac_surface_margin_adaptive_reject_panels{0};
    std::size_t tabi_mac_surface_margin_skipped_accepted_far{0};
    std::size_t tabi_mac_surface_margin_hard_applied{0};
    std::size_t tabi_mac_surface_margin_soft_applied{0};
    std::size_t tabi_mac_surface_margin_hard_rejects{0};
    std::size_t tabi_mac_surface_margin_hard_reject_panels{0};
    std::size_t tabi_mac_surface_margin_soft_rejects{0};
    std::size_t tabi_mac_surface_margin_soft_reject_panels{0};
    std::size_t tabi_fragmented_far_count{0};
    std::size_t tabi_fragmented_far_panels{0};
    std::size_t tabi_fragmented_split_rejects{0};
    double tabi_fragmentation_center_gap_ratio_max{0.0};
    double tabi_fragmentation_center_gap_ratio_mean{0.0};
    std::string tabi_moment_audit{"off"};
    std::string tabi_moment_construction{"off"};
    std::string tabi_moment_sum{"off"};
    std::string tabi_convergence_audit{"off"};
    std::string tabi_convergence_safety{"off"};
    double tabi_convergence_rho_cap{0.0};
    double tabi_mac_aspect_cap{0.0};
    std::string tabi_convergence_estimator{"off"};
    double tabi_convergence_tail_threshold{1.0e4};
    double tabi_convergence_estimator_error_cap{1.0e-3};
    double tabi_convergence_unstable_rho_cap{0.25};
    std::string tabi_convergence_calibration{"stage3j"};
    std::size_t tabi_convergence_estimator_samples{0};
    std::size_t tabi_convergence_estimator_unstable_far_count{0};
    std::size_t tabi_convergence_estimator_unstable_far_panels{0};
    std::size_t tabi_convergence_estimator_split_rejects{0};
    std::size_t tabi_convergence_estimator_split_reject_panels{0};
    std::size_t tabi_convergence_low_abs_reaccepted_far_count{0};
    std::size_t tabi_convergence_low_abs_reaccepted_far_panels{0};
    double tabi_convergence_estimator_tail_ratio_max{0.0};
    double tabi_convergence_estimator_tail_ratio_mean{0.0};
    double tabi_convergence_estimator_error_max{0.0};
    double tabi_convergence_estimator_error_mean{0.0};
    double tabi_convergence_estimator_abs_error_max{0.0};
    double tabi_convergence_estimator_abs_error_mean{0.0};
    double tabi_convergence_tail_norm_p_max{0.0};
    double tabi_convergence_tail_norm_p_mean{0.0};
    double tabi_convergence_tail_norm_pm1_max{0.0};
    double tabi_convergence_tail_norm_pm1_mean{0.0};
    double tabi_convergence_effective_jcp_amp_max{0.0};
    double tabi_convergence_effective_jcp_amp_mean{0.0};
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
    double tabi_convergence_dielectric_amp_max{0.0};
    double tabi_convergence_dielectric_amp_mean{0.0};
    std::array<double, kJcpExpansionChannelCount>
        tabi_convergence_channel_estimated_error_max{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_convergence_channel_estimated_error_mean{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_convergence_channel_effective_jcp_amp_max{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_convergence_channel_effective_jcp_amp_mean{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_unstable_far_count{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_unstable_far_panels{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_split_rejects{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_split_reject_panels{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_low_abs_reaccepted_count{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_low_abs_reaccepted_panels{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_high_error_caught{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_high_error_missed{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_high_error_caught_by_split{};
    std::size_t tabi_convergence_estimator_high_error_caught{0};
    std::size_t tabi_convergence_estimator_high_error_missed{0};
    std::size_t tabi_convergence_estimator_high_error_caught_by_split{0};
    std::size_t tabi_convergence_estimator_phantom_high_error_count{0};
    double tabi_convergence_estimator_phantom_high_error_abs_max{0.0};
    std::size_t tabi_convergence_overreject_count{0};
    std::size_t tabi_convergence_overreject_panels{0};
    std::size_t tabi_convergence_overreject_caught_count{0};
    std::size_t tabi_convergence_overreject_caught_panels{0};
    double tabi_convergence_overreject_abs_error_max{0.0};
    double tabi_convergence_overreject_abs_error_mean{0.0};
    double tabi_convergence_overreject_abs_error_p95{0.0};
    double tabi_convergence_overreject_abs_error_p99{0.0};
    double tabi_convergence_overreject_estimated_abs_error_max{0.0};
    double tabi_convergence_overreject_estimated_abs_error_mean{0.0};
    double tabi_convergence_overreject_required_scale_max{0.0};
    double tabi_convergence_overreject_required_scale_p95{0.0};
    double tabi_convergence_overreject_required_scale_p99{0.0};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_overreject_count{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_overreject_caught_count{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_convergence_channel_overreject_panels{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_convergence_channel_overreject_abs_error_max{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_convergence_channel_overreject_abs_error_mean{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_convergence_channel_overreject_required_scale_max{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_convergence_channel_overreject_required_scale_p99{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_phantom_high_error_channel_count{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_phantom_high_error_channel_abs_max{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_accepted_miss_channel_count{};
    std::array<std::size_t, kJcpExpansionChannelCount>
        tabi_accepted_miss_channel_panels{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_accepted_miss_channel_rho_min{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_accepted_miss_channel_rho_mean{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_accepted_miss_channel_rho_max{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_accepted_miss_channel_tail_ratio_mean{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_accepted_miss_channel_tail_ratio_max{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_accepted_miss_channel_estimated_error_mean{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_accepted_miss_channel_estimated_error_max{};
    std::string tabi_accepted_miss_worst_channel{"none"};
    std::size_t tabi_convergence_k3_rho_cap_rejects{0};
    std::size_t tabi_convergence_k3_rho_cap_reject_panels{0};
    std::size_t tabi_moment_audit_nodes{0};
    double tabi_moment_translation_rel_error_max{0.0};
    double tabi_moment_translation_rel_error_mean{0.0};
    double tabi_moment_translation_offset_max{0.0};
    double tabi_moment_translation_offset_mean{0.0};
    double tabi_moment_cancellation_ratio_max{0.0};
    double tabi_moment_cancellation_ratio_mean{0.0};
    double tabi_moment_dynamic_range_max{0.0};
    std::string tabi_expansion_audit{"off"};
    std::string tabi_expansion_audit_channel{"off"};
    std::size_t tabi_far_expansion_audit_pairs{0};
    std::size_t tabi_far_expansion_audit_panels{0};
    double tabi_far_expansion_direct_ms{0.0};
    double tabi_far_expansion_rel_error_max{0.0};
    double tabi_far_expansion_rel_error_mean{0.0};
    double tabi_far_expansion_rel_error_p95{0.0};
    double tabi_far_expansion_rel_error_p99{0.0};
    std::array<double, kJcpExpansionChannelCount>
        tabi_far_expansion_channel_rel_error_max{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_far_expansion_channel_rel_error_mean{};
    std::array<double, kJcpExpansionChannelCount>
        tabi_far_expansion_channel_rel_error_p99{};
    std::string tabi_far_expansion_worst_channel{"none"};
    bool tabi_far_expansion_mean_gate_pass{false};
    double tabi_far_convergence_rho_max{0.0};
    double tabi_far_convergence_rho_mean{0.0};
    double tabi_accepted_far_aspect_ratio_max{0.0};
    double tabi_accepted_far_aspect_ratio_mean{0.0};
    std::string tabi_quad_buffer_auto_candidate{"off"};
    std::string tabi_leaf_tile_status{"off"};
    double tabi_leaf_panel_pairs_per_second{0.0};
    std::string tabi_near_integration{"off"};
    std::string tabi_quad_buffer{"off"};
    std::size_t tabi_near_self_pairs{0};
    std::size_t tabi_near_one_ring_pairs{0};
    std::size_t tabi_near_duffy_pairs{0};
    std::size_t tabi_near_centroid_pairs{0};
    std::size_t tabi_near_fallbacks{0};
    std::size_t tabi_near_geometric_buffer_pairs{0};
    std::size_t tabi_near_geometric_buffer_gauss_pairs{0};
    std::size_t tabi_near_geometric_buffer_adaptive_pairs{0};
    std::size_t tabi_near_geometric_buffer_centroid_pairs{0};
    double tabi_near_geometric_buffer_min_gap{0.0};
    double tabi_near_geometric_buffer_mean_gap_ratio{0.0};
    double tabi_near_geometric_buffer_max_edge_mean{0.0};
    double tabi_near_integration_ms{0.0};
    std::size_t tabi_duffy_subdivisions_max{0};
    double tabi_duffy_subdivisions_mean{0.0};
    std::string tabi_near_kernel_policy{"off"};
    std::size_t tabi_near_k1_accepted{0};
    std::size_t tabi_near_k1_rejected{0};
    std::size_t tabi_near_k2_accepted{0};
    std::size_t tabi_near_k2_rejected{0};
    std::size_t tabi_near_k3_accepted{0};
    std::size_t tabi_near_k3_rejected{0};
    std::size_t tabi_near_k4_accepted{0};
    std::size_t tabi_near_k4_rejected{0};
    std::size_t tabi_near_one_ring_k1_accepted{0};
    std::size_t tabi_near_one_ring_k1_rejected{0};
    std::size_t tabi_near_one_ring_k2_accepted{0};
    std::size_t tabi_near_one_ring_k2_rejected{0};
    std::size_t tabi_near_one_ring_k3_accepted{0};
    std::size_t tabi_near_one_ring_k3_rejected{0};
    std::size_t tabi_near_one_ring_k4_accepted{0};
    std::size_t tabi_near_one_ring_k4_rejected{0};
    std::string tabi_near_k1_policy{"off"};
    std::string tabi_near_k2_policy{"off"};
    std::string tabi_near_k3_policy{"off"};
    std::string tabi_near_k4_policy{"off"};
    NearBlockAuditBuckets tabi_near_block_audit{};
    GeometricNearAuditBuckets tabi_geometric_near_audit{};
    std::string tabi_surface_geometry{"off"};
    std::string tabi_sphere_mesh_mode{"off"};
    double tabi_surface_quality_total_area_a2{0.0};
    double tabi_surface_quality_area_rel_error_vs_4piR2{0.0};
    double tabi_surface_quality_triangle_area_min{0.0};
    double tabi_surface_quality_triangle_area_mean{0.0};
    double tabi_surface_quality_triangle_area_max{0.0};
    double tabi_surface_quality_triangle_aspect_max{0.0};
    double tabi_surface_quality_normal_radial_dot_min{0.0};
    double tabi_surface_quality_normal_radial_dot_mean{0.0};
    double tabi_surface_quality_centroid_radial_error_max{0.0};
    double tabi_surface_quality_planar_area_a2{0.0};
    double tabi_surface_quality_planar_area_rel_error_vs_4piR2{0.0};
    double tabi_surface_quality_enclosed_volume_a3{0.0};
    double tabi_surface_quality_enclosed_volume_rel_error_vs_4piR3_over3{0.0};
    double tabi_surface_quality_triangle_area_cv{0.0};
    std::string tabi_surface_topology_source{"not_built"};
    std::string tabi_surface_analytic_ses_engine{"not_applicable"};
    std::string tabi_surface_status_detail{"not_built"};
    std::string tabi_surface_cavity_policy{"not_applicable"};
    std::size_t tabi_surface_odd_ray_rows{0};
    std::size_t tabi_surface_patch_ray_unresolved_uncertified_rows{0};
    std::size_t tabi_surface_morphology_fallback_vertices{0};
    std::string tabi_surface_analytic_orientation_status{"not_run"};
    std::size_t tabi_surface_analytic_inverted_faces{0};
    std::size_t tabi_surface_analytic_orientation_uncertified_faces{0};
    std::size_t tabi_surface_analytic_self_intersections{0};
    std::size_t tabi_surface_wide_repair_attempts{0};
    std::size_t tabi_surface_topology_certificate_accepts{0};
    std::size_t tabi_surface_topology_fallback_audits{0};
    double tabi_surface_initial_collapse_ms{0.0};
    double tabi_surface_stable_sliver_ms{0.0};
    double tabi_surface_full_projection_ms{0.0};
    double tabi_surface_orientation_repair_ms{0.0};
    double tabi_surface_wide_repair_ms{0.0};
    double tabi_surface_self_intersection_audit_ms{0.0};
    std::size_t tabi_surface_filled_cavity_regions{0};
    std::size_t tabi_surface_filled_cavity_vertices{0};
    std::size_t tabi_surface_poor_triangles_before{0};
    std::size_t tabi_surface_poor_triangles_after{0};
    double tabi_surface_triangle_quality_min_before{0.0};
    double tabi_surface_triangle_quality_min_after{0.0};
    double tabi_surface_max_edge_ratio_before{0.0};
    double tabi_surface_max_edge_ratio_after{0.0};
    std::string tabi_surface_analytic_audit_status{"not_run"};
    std::size_t tabi_surface_analytic_audit_samples{0};
    std::size_t tabi_surface_analytic_audit_hits{0};
    std::size_t tabi_surface_analytic_audit_misses{0};
    double tabi_surface_analytic_projection_distance_rms{0.0};
    double tabi_surface_analytic_projection_distance_max{0.0};
    double tabi_surface_analytic_patch_residual_rms{0.0};
    double tabi_surface_analytic_patch_residual_max{0.0};
    double tabi_surface_analytic_normal_angle_rms_deg{0.0};
    double tabi_surface_analytic_normal_angle_max_deg{0.0};
    double tabi_surface_sampled_hausdorff_proxy{0.0};
    std::string tabi_surface_analytic_normal_quality_status{"not_run"};
    std::size_t tabi_surface_analytic_orientation_smooth_faces{0};
    std::size_t tabi_surface_analytic_orientation_seam_faces{0};
    std::size_t tabi_surface_analytic_normal_cosine_below_025_faces{0};
    std::size_t tabi_surface_analytic_normal_cosine_below_050_faces{0};
    double tabi_surface_analytic_normal_area_weighted_rms_deg{0.0};
    double tabi_surface_analytic_normal_p99_deg{0.0};
    double tabi_surface_analytic_smooth_normal_p99_deg{0.0};
    double tabi_surface_analytic_seam_normal_p99_deg{0.0};
    double tabi_surface_analytic_normal_area_below_cosine_025_fraction{0.0};
    double tabi_surface_analytic_normal_area_below_cosine_050_fraction{0.0};
    std::string tabi_calderon_trace_space_status{"off"};
    std::size_t tabi_calderon_trace_vertices{0};
    std::size_t tabi_calderon_trace_edges{0};
    std::size_t tabi_calderon_trace_boundary_edges{0};
    std::size_t tabi_calderon_trace_orientation_mismatches{0};
    std::size_t tabi_calderon_trace_nonmanifold_edges{0};
    std::size_t tabi_calderon_trace_unreferenced_vertices{0};
    std::size_t tabi_calderon_trace_components{0};
    std::int64_t tabi_calderon_trace_euler_characteristic{0};
    std::size_t tabi_calderon_trace_genus_sum{0};
    bool tabi_calderon_trace_topology_invariants_valid{false};
    std::string tabi_calderon_aux_surface_mode{"off"};
    std::string tabi_calderon_aux_trace_status{"off"};
    std::size_t tabi_calderon_aux_panels{0};
    std::size_t tabi_calderon_aux_vertices{0};
    std::size_t tabi_calderon_aux_edges{0};
    std::size_t tabi_calderon_aux_boundary_edges{0};
    std::size_t tabi_calderon_aux_orientation_mismatches{0};
    std::size_t tabi_calderon_aux_nonmanifold_edges{0};
    std::size_t tabi_calderon_aux_unreferenced_vertices{0};
    std::size_t tabi_calderon_aux_grid_vertices{0};
    double tabi_calderon_aux_area_a2{0.0};
    double tabi_calderon_aux_build_ms{0.0};
    std::size_t tabi_mesh_vertices{0};
    std::size_t tabi_mesh_triangles{0};
    std::size_t tabi_boundary_nodes{0};
    double tabi_avg_valence{0.0};
    std::size_t tabi_nonmanifold_edges{0};
    std::size_t tabi_solid_angle_fallbacks{0};
    std::size_t tabi_solid_angle_clamps{0};
    double tabi_diag_min{1.0};
    double tabi_diag_max{1.0};
    int tabi_taylor_order{0};
    std::string tabi_derivative_mode{"off"};
    double tabi_interaction_build_ms{0.0};
    double tabi_tree_eval_ms{0.0};
    std::string tabi_formulation{"off"};
    std::string tabi_energy_projection{"off"};
    double tabi_energy_projection_ms{0.0};
    std::string tabi_energy_projection_kernel{"off"};
    std::string tabi_preconditioner_requested{"off"};
    std::string tabi_preconditioner_effective{"off"};
    std::string tabi_preconditioner_block_near{"off"};
    int tabi_preconditioner_block_panel_cap{32};
    int tabi_preconditioner_overlap_panel_cap{64};
    int tabi_preconditioner_ilu_neighbor_cap_requested{192};
    int tabi_preconditioner_ilu_neighbor_cap{192};
    double tabi_preconditioner_ilu_drop_tol{0.0};
    std::string tabi_preconditioner_ilu_regularization{"off"};
    double tabi_preconditioner_ilu_regularization_max{1.0e-2};
    double tabi_preconditioner_ilu_regularization_selected{0.0};
    double tabi_preconditioner_ilu_regularization_absolute_max{0.0};
    std::size_t tabi_preconditioner_ilu_regularization_attempts{0};
    std::size_t tabi_preconditioner_ilu_regularization_rejected_attempts{0};
    std::size_t tabi_preconditioner_ilu_regularization_shifted_rows{0};
    std::string tabi_preconditioner_ilu_unshifted_failure_stage{"off"};
    std::size_t tabi_preconditioner_memory_budget_bytes{0};
    std::size_t tabi_preconditioner_estimated_bytes{0};
    std::size_t tabi_preconditioner_valid_blocks{0};
    std::size_t tabi_preconditioner_valid_panels{0};
    std::size_t tabi_preconditioner_ill_conditioned_blocks{0};
    std::size_t tabi_preconditioner_quality_rejected_blocks{0};
    double tabi_preconditioner_quality_probe_max_gain{0.0};
    double tabi_preconditioner_quality_probe_rel_residual{0.0};
    double tabi_preconditioner_pivot_ratio_min{0.0};
    double tabi_preconditioner_factor_growth_max{0.0};
    std::int64_t tabi_preconditioner_pivot_ratio_min_row{-1};
    std::int64_t tabi_preconditioner_pivot_ratio_min_column{-1};
    std::int64_t tabi_preconditioner_factor_growth_row{-1};
    std::int64_t tabi_preconditioner_factor_growth_column{-1};
    std::string tabi_preconditioner_quality_failure_stage{"off"};
    std::int64_t tabi_preconditioner_quality_failure_row{-1};
    std::int64_t tabi_preconditioner_quality_failure_column{-1};
    std::int64_t tabi_preconditioner_quality_failure_probe{-1};
    double tabi_preconditioner_quality_failure_probe_gain{0.0};
    double tabi_preconditioner_quality_failure_probe_rel_residual{0.0};
    std::string tabi_preconditioner_quality_status{"off"};
    double tabi_preconditioner_setup_ms{0.0};
    double tabi_preconditioner_graph_setup_ms{0.0};
    bool tabi_preconditioner_graph_cache_reused{false};
    double tabi_preconditioner_near_assembly_setup_ms{0.0};
    double tabi_preconditioner_ilu_factor_setup_ms{0.0};
    double tabi_preconditioner_apply_ms{0.0};
    std::size_t tabi_preconditioner_apply_calls{0};
    std::size_t tabi_preconditioner_sparse_nnz_blocks{0};
    double tabi_preconditioner_sparse_avg_degree{0.0};
    std::size_t tabi_preconditioner_sparse_max_degree{0};
    int tabi_preconditioner_coarse_aggregate_cap{16};
    std::size_t tabi_preconditioner_coarse_aggregates{0};
    std::size_t tabi_preconditioner_coarse_dimension{0};
    std::string tabi_preconditioner_coarse_base{"diagonal"};
    std::size_t tabi_preconditioner_coarse_matrix_bytes{0};
    double tabi_preconditioner_coarse_pivot_ratio{0.0};
    std::string tabi_preconditioner_coarse_status{"off"};
    std::size_t tabi_preconditioner_coarse_setup_matvec_calls{0};
    double tabi_preconditioner_coarse_setup_ms{0.0};
    double tabi_preconditioner_coarse_moments_setup_ms{0.0};
    double tabi_preconditioner_coarse_far_setup_ms{0.0};
    double tabi_preconditioner_coarse_audit_setup_ms{0.0};
    double tabi_preconditioner_coarse_apply_ms{0.0};
    std::size_t tabi_preconditioner_coarse_apply_calls{0};
    double tabi_preconditioner_coarse_rp_identity_error_max{0.0};
    double tabi_preconditioner_coarse_verify_rel_error_max{0.0};
    std::string tabi_surface_generator{"off"};
    std::string tabi_discretization{"off"};
    std::string tabi_node_patch_representation{"n/a"};
    std::string tabi_node_patch_normal_source{"n/a"};
    std::string tabi_node_patch_certificate{"n/a"};
    double tabi_rhs_scale{1.0};
    double tabi_dielectric_contrast{0.0};
    double tabi_operator_contrast{0.0};
    std::size_t tabi_unknown_count{0};
    std::string tabi_bie_system{"off"};
    double tabi_lhs_diag_phi{0.0};
    double tabi_lhs_diag_phin{0.0};
    std::string tabi_energy_equation{"off"};
    std::string tabi_kernel_convention{"off"};
    double tabi_source_norm_s1{0.0};
    double tabi_source_norm_s2{0.0};
    double tabi_phi_norm{0.0};
    double tabi_phin_norm{0.0};
    double tabi_kappa_inv_a{0.0};
    double tabi_coulomb_factor_kcal_mol{0.0};
    double tabi_boundary_energy_kj_mol{0.0};
    double tabi_reference_energy_kj_mol{0.0};
    double tabi_selected_energy_scale{1.0};
    bool tabi_fallback_used{false};
    std::vector<Tabi::Panel> tabi_output_panels;
    std::vector<double> tabi_output_solution;
    std::vector<double> tabi_output_rhs;
    std::string pbe_model_effective{"linear"};
    std::string linear_seed_solver{"none"};
    std::string nl_solver{"picard"};
    std::string nl_inner_solver{"sor"};
    std::string nl_mg_mode{"off"};
    std::string focus_parent_grid{};
    std::string focus_child_grid{};
    std::string probe_status{"off"};
    int newton_inner_iterations{0};
    double newton_step_norm{0.0};
    double newton_residual_before{0.0};
    double newton_residual_after{0.0};
    std::string newton_line_search_status{"off"};
    double newton_correction_residual{0.0};
    double newton_correction_max_residual{0.0};
    double newton_accepted_lambda{0.0};
    int newton_rejected_steps{0};
    double newton_jac_diag_min{0.0};
    double newton_jac_diag_max{0.0};
    std::size_t newton_screen_points{0};
    std::size_t newton_active_nonlinear_points{0};
    std::string newton_stop_reason{"off"};
    int newton_inner_levels{1};
    int newton_inner_cycles{0};
    double newton_inner_residual_before{0.0};
    double newton_inner_residual_after{0.0};
    std::string newton_inner_effective{"off"};
    double gaussian_density_sum{0.0};
    std::size_t gaussian_cutoff_points{0};
    double gaussian_eps_min{0.0};
    double gaussian_eps_max{0.0};
    double gaussian_eps_mean{0.0};
    double gaussian_charge_norm_error{0.0};
    double rpbe_coulomb_norm{0.0};
    double rpbe_grad_coulomb_norm{0.0};
    double rpbe_grad_epsilon_norm{0.0};
    double rpbe_source_rms{0.0};
    double rpbe_source_max{0.0};
    std::string rpbe_source_status{"off"};
    double rhs_scale{0.0};
    double charge_assigned{0.0};
    double charge_moment_error{0.0};
    double boundary_rms{0.0};
    double boundary_max{0.0};
    double linear_residual{0.0};
    int nl_iterations{0};
    int nl_blocks{0};
    double nl_relax{0.5};
    double nl_max_dphi{0.0};
    double nl_residual{0.0};
    std::size_t nl_clipped_points{0};
    int nl_continuation_steps{0};
    int nl_backtracks{0};
    std::string nl_status{"linear"};
    double energy_kcal_mol{0.0};
    double energy_kj_mol{0.0};
    double energy_kt{0.0};
    double solvent_phase_kcal_mol{0.0};
    double solvent_phase_kj_mol{0.0};
    double solvent_phase_kt{0.0};
    double vacuum_phase_kcal_mol{0.0};
    double vacuum_phase_kj_mol{0.0};
    double vacuum_phase_kt{0.0};
    double atom_corrected_kcal_mol{0.0};
    double atom_corrected_kj_mol{0.0};
    double atom_corrected_kt{0.0};
    double grid_corrected_kcal_mol{0.0};
    double grid_corrected_kj_mol{0.0};
    double grid_corrected_kt{0.0};
    double iscm_water_phase_kcal_mol{0.0};
    double iscm_water_phase_kj_mol{0.0};
    double iscm_water_phase_kt{0.0};
    double iscm_vacuum_phase_kcal_mol{0.0};
    double iscm_vacuum_phase_kj_mol{0.0};
    double iscm_vacuum_phase_kt{0.0};
    double iscm_corrected_kcal_mol{0.0};
    double iscm_corrected_kj_mol{0.0};
    double iscm_corrected_kt{0.0};
    double iscm_water_minus_vacuum_kcal_mol{0.0};
    double iscm_water_minus_vacuum_kj_mol{0.0};
    double iscm_water_minus_vacuum_kt{0.0};
    double nonlinear_stress_kt{0.0};
    double nonlinear_stress_kj_mol{0.0};
    double nonlinear_stress_kcal_mol{0.0};
    double nonlinear_osmotic_kt{0.0};
    double nonlinear_osmotic_kj_mol{0.0};
    double nonlinear_osmotic_kcal_mol{0.0};
    double nonlinear_energy_correction_kt{0.0};
    double nonlinear_energy_correction_kj_mol{0.0};
    double nonlinear_energy_correction_kcal_mol{0.0};
    std::string nonlinear_energy_status{"off"};
    std::string iscm_finalization_mode{"off"};
    std::string iscm_position_source{"off"};
    double iscm_position_rms_vs_oracle{0.0};
    std::size_t iscm_position_oracle_matches{0};
    bool iscm_native_independent{true};
    std::size_t iscm_boundary_points{0};
    std::size_t iscm_charged_boundary_points{0};
    double iscm_induced_charge_sum{0.0};
    double iscm_induced_charge_l1{0.0};
    double iscm_induced_charge_l2{0.0};
    double iscm_induced_charge_max{0.0};
    std::string iscm_status{"off"};
    std::size_t delphi_face_crossings{0};
    std::size_t delphi_eps_face_matches{0};
    std::size_t delphi_boundary_grid_points{0};
    std::size_t delphi_charged_boundary_points{0};
    std::string delphi_iscm_status{"off"};
    std::string vdwms_enabled{"off"};
    std::size_t vdwms_boundary_initial{0};
    std::size_t vdwms_boundary_final{0};
    std::size_t vdwms_added{0};
    std::size_t vdwms_removed{0};
    std::size_t vdwms_exposed_points{0};
    double vdwms_midpoint_rms{0.0};
    double coulomb_energy_kcal_mol{0.0};
    double coulomb_energy_kj_mol{0.0};
    double coulomb_energy_kt{0.0};
    std::string reaction_energy_source{"atom"};
    ParityAudit parity{};
    std::vector<double> atom_solvent_potential_kj_mol_e;
    std::vector<double> atom_vacuum_potential_kj_mol_e;
    std::vector<double> atom_reaction_potential_kj_mol_e;
    std::vector<double> atom_reaction_kj_mol;
    double atom_reaction_energy_closure_kj_mol{0.0};
    double atom_reaction_energy_closure_error_kj_mol{0.0};
    double probe_energy_kj_mol{0.0};
    double probe_energy_kcal_mol{0.0};
    double probe_energy_kt{0.0};
    std::size_t probe_matched_atoms{0};
    std::string probe_matched_ids{};
    std::string probe_matched_groups{};
    std::string probe_energy_status{"off"};
    std::string probe_matching_semantics{"delphi_residue_site_groups"};
    bool converged{false};
  };

  struct FieldSolve {
    GridShape shape{};
    SolveStats stats{};
    int total_iterations{0};
    int levels{1};
    double initial_scale{0.0};
    double initial_margin{0.0};
    std::string solver_mode{"sor"};
    int mg_levels{1};
    int mg_cycles{0};
    int mg_pre{0};
    int mg_post{0};
    int mg_coarse_iter{0};
    std::size_t solute_points{0};
    std::size_t solvent_points{0};
    std::size_t ion_accessible_points{0};
    std::size_t stern_excluded_points{0};
    std::size_t peak_memory_bytes{0};
    bool pbsa_reuse_surface_hit{false};
    bool pbsa_reuse_operator_hit{false};
    std::string pbe_model_effective{"linear"};
    std::string linear_seed_solver{"none"};
    std::string nl_solver{"picard"};
    std::string nl_inner_solver{"sor"};
    std::string nl_mg_mode{"off"};
    std::string focus_parent_grid{};
    std::string focus_child_grid{};
    std::string probe_status{"off"};
    int newton_inner_iterations{0};
    double newton_step_norm{0.0};
    double newton_residual_before{0.0};
    double newton_residual_after{0.0};
    std::string newton_line_search_status{"off"};
    double newton_correction_residual{0.0};
    double newton_correction_max_residual{0.0};
    double newton_accepted_lambda{0.0};
    int newton_rejected_steps{0};
    double newton_jac_diag_min{0.0};
    double newton_jac_diag_max{0.0};
    std::size_t newton_screen_points{0};
    std::size_t newton_active_nonlinear_points{0};
    std::string newton_stop_reason{"off"};
    int newton_inner_levels{1};
    int newton_inner_cycles{0};
    double newton_inner_residual_before{0.0};
    double newton_inner_residual_after{0.0};
    std::string newton_inner_effective{"off"};
    double gaussian_density_sum{0.0};
    std::size_t gaussian_cutoff_points{0};
    double gaussian_eps_min{0.0};
    double gaussian_eps_max{0.0};
    double gaussian_eps_mean{0.0};
    double gaussian_charge_norm_error{0.0};
    double rhs_scale{0.0};
    double charge_assigned{0.0};
    double charge_moment_error{0.0};
    double boundary_rms{0.0};
    double boundary_max{0.0};
    double linear_residual{0.0};
    int nl_iterations{0};
    int nl_blocks{0};
    double nl_relax{0.5};
    double nl_max_dphi{0.0};
    double nl_residual{0.0};
    std::size_t nl_clipped_points{0};
    int nl_continuation_steps{0};
    int nl_backtracks{0};
    std::string nl_status{"linear"};
    ParityAudit parity{};
    std::string boundary_convention{"native-real"};
    std::string field_units{"native"};
    std::vector<double> atom_phi;
    AlignedDoubleVector phi_grid;
    AlignedDoubleVector rho_grid;
    AlignedDoubleVector rho_energy_grid;
    AlignedDoubleVector eps_sum_grid;
    AlignedDoubleVector eps_x_grid;
    AlignedDoubleVector eps_y_grid;
    AlignedDoubleVector eps_z_grid;
    AlignedDoubleVector screen_grid;
    std::vector<unsigned char> inside_grid;
    std::string vdwms_enabled{"off"};
    std::size_t vdwms_boundary_initial{0};
    std::size_t vdwms_boundary_final{0};
    std::size_t vdwms_added{0};
    std::size_t vdwms_removed{0};
    std::size_t vdwms_exposed_points{0};
    double vdwms_midpoint_rms{0.0};
    double rpbe_coulomb_norm{0.0};
    double rpbe_grad_coulomb_norm{0.0};
    double rpbe_grad_epsilon_norm{0.0};
    double rpbe_source_rms{0.0};
    double rpbe_source_max{0.0};
    std::string rpbe_source_status{"off"};
    AlignedDoubleVector rpbe_source_grid;
    AlignedDoubleVector rpbe_coulomb_grid;
  };

  struct NonlinearEnergyAudit {
    double electrostatic_stress_kt{0.0};
    double electrostatic_stress_kj_mol{0.0};
    double electrostatic_stress_kcal_mol{0.0};
    double osmotic_pressure_kt{0.0};
    double osmotic_pressure_kj_mol{0.0};
    double osmotic_pressure_kcal_mol{0.0};
    double total_correction_kt{0.0};
    double total_correction_kj_mol{0.0};
    double total_correction_kcal_mol{0.0};
    std::size_t sampled_points{0};
    std::string status{"off"};
  };

  struct RpbAudit {
    double coulomb_norm{0.0};
    double grad_coulomb_norm{0.0};
    double grad_epsilon_norm{0.0};
    double source_rms{0.0};
    double source_max{0.0};
    std::size_t sampled_points{0};
    std::string status{"off"};
  };

  struct ForensicAudit {
    std::size_t face_cross_x{0};
    std::size_t face_cross_y{0};
    std::size_t face_cross_z{0};
    std::size_t eps_face_count_x{0};
    std::size_t eps_face_count_y{0};
    std::size_t eps_face_count_z{0};
    std::size_t internal_face_count_x{0};
    std::size_t internal_face_count_y{0};
    std::size_t internal_face_count_z{0};
    double eps_face_min{0.0};
    double eps_face_max{0.0};
    double eps_face_mean{0.0};
    double eps_face_mean_x{0.0};
    double eps_face_mean_y{0.0};
    double eps_face_mean_z{0.0};
    double vdw_volume_estimate_A3{0.0};
    double rho_sum{0.0};
    double rho_l1{0.0};
    double rho_l2{0.0};
    double rho_mx{0.0};
    double rho_my{0.0};
    double rho_mz{0.0};
    double denom_min{0.0};
    double denom_max{0.0};
    double denom_mean{0.0};
    double screen_sum{0.0};
    double screen_max{0.0};
  };

  struct VdwmsAudit {
    std::string enabled{"off"};
    std::size_t boundary_initial{0};
    std::size_t boundary_final{0};
    std::size_t added{0};
    std::size_t removed{0};
    std::size_t exposed_points{0};
    double midpoint_rms{0.0};
  };

  struct DelphiFaceDebug {
    std::int32_t raw_media{0};
    std::int32_t raw_owner{0};
    std::int32_t raw_encoded{0};
    std::int32_t final_encoded{0};
    std::uint16_t action{0};
    std::uint32_t iteration{0};
    std::int32_t nearest_atom{0};
    double nearest_atom_surface_dist{1.0e300};
    double nearest_exposed_dist{1.0e300};
  };

  struct DelphiVdwMediaMap {
    std::vector<std::int32_t> face_encoded;
    std::vector<std::int32_t> face_media;
    std::vector<std::int32_t> face_owner;
    std::vector<DelphiFaceDebug> face_debug;
    std::vector<unsigned char> iscm_surface_valid;
    std::vector<std::array<double, 3>> iscm_surface_positions;
    VdwmsAudit audit{};
  };

  struct IscmSample {
    std::string kind{};
    std::size_t rank{0};
    std::size_t i{0};
    std::size_t j{0};
    std::size_t k{0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double phi_center{0.0};
    double phi_neighbor_average{0.0};
    double charge_adjustment{0.0};
    double induced_charge{0.0};
    double atom_potential_sum{0.0};
    double energy_contribution_kt{0.0};
  };

  struct IscmAudit {
    std::size_t boundary_points{0};
    std::size_t charged_boundary_points{0};
    double induced_charge_sum{0.0};
    double induced_charge_l1{0.0};
    double induced_charge_l2{0.0};
    double induced_charge_max{0.0};
    double water_phase_kt{0.0};
    double water_phase_kj_mol{0.0};
    double water_phase_kcal_mol{0.0};
    double vacuum_phase_kt{0.0};
    double vacuum_phase_kj_mol{0.0};
    double vacuum_phase_kcal_mol{0.0};
    double corrected_kt{0.0};
    double corrected_kj_mol{0.0};
    double corrected_kcal_mol{0.0};
    double water_minus_vacuum_kt{0.0};
    double water_minus_vacuum_kj_mol{0.0};
    double water_minus_vacuum_kcal_mol{0.0};
    std::string finalization_mode{"off"};
    std::string position_source{"off"};
    double position_rms_vs_oracle{0.0};
    std::size_t position_oracle_matches{0};
    bool native_independent{true};
    std::size_t delphi_face_crossings{0};
    std::size_t delphi_eps_face_matches{0};
    std::string status{"off"};
    std::vector<IscmSample> samples;
  };

  struct PbsaComponents {
    double elec_kj_mol{0.0};
    double vdw_kj_mol{0.0};
    double mm_kj_mol{0.0};
    double sasa_complex_a2{0.0};
    double sasa_mol1_a2{0.0};
    double sasa_mol2_a2{0.0};
    double delta_sasa_a2{0.0};
    double nonpolar_kj_mol{0.0};
    double delta_solv_kj_mol{0.0};
    double no_entropy_kj_mol{0.0};
    std::string status{"ok"};
  };

  struct ScalarStats {
    std::size_t count{0};
    double mean{0.0};
    double m2{0.0};
    double min{0.0};
    double max{0.0};
  };

  struct PbsaEntropyResult {
    std::size_t samples{0};
    std::size_t filtered_samples{0};
    double ess{0.0};
    double minus_tds_kj_mol{0.0};
    std::size_t block_size{0};
    std::size_t block_count{0};
    double block_mean_kj_mol{0.0};
    double block_std_kj_mol{0.0};
    double block_drift_kj_mol{0.0};
    std::string status{"disabled"};
  };

  struct ResidueKey {
    std::string molecule{"unknown"};
    std::string unit_type{"residue"};
    std::size_t chainnum{0};
    std::string chain{" "};
    std::size_t resnr{0};
    std::size_t respos{0};
    std::string resname{"UND"};
    std::size_t atom_index{0};
    std::string atom_name{};
  };

  struct ResidueContribution {
    ResidueKey key{};
    double elec_kj_mol{0.0};
    double vdw_kj_mol{0.0};
    double mm_kj_mol{0.0};
    double polar_kj_mol{0.0};
    double nonpolar_kj_mol{0.0};
    double total_kj_mol{0.0};
  };

  struct ResidueFrameRow : ResidueContribution {
    std::size_t trajectory_id{0};
    std::size_t frame_index{0};
    std::size_t global_index{0};
    double time_ps{0.0};
  };

  struct ResidueStats {
    ResidueKey key{};
    std::size_t frames{0};
    ScalarStats elec{};
    ScalarStats vdw{};
    ScalarStats mm{};
    ScalarStats polar{};
    ScalarStats nonpolar{};
    ScalarStats total{};
    std::vector<double> ie_values_kj_mol;
  };

  struct PbsaEndpointTimingRow {
    std::size_t trajectory_id{0};
    std::size_t frame_index{0};
    std::size_t global_index{0};
    double time_ps{0.0};
    std::string endpoint{};
    std::string phase{};
    std::string variant{"baseline"};
    TimingBuckets timing{};
    std::size_t reuse_surface_hits{0};
    std::size_t reuse_operator_hits{0};
    std::string status{"off"};
  };

  // Fine level of the Galerkin V-cycle; the operator itself is in grid_.
  struct MultigridLevel {
    GridShape shape{};
    AlignedDoubleVector rhs;
    ChargeDiagnostics charge_diag{};
    BoundaryDiagnostics boundary_diag{};
  };

  struct GridMediumOperatorCache {
    bool valid{false};
    std::uint64_t key{0};
    GridShape shape{};
    AlignedDoubleVector eps_x;
    AlignedDoubleVector eps_y;
    AlignedDoubleVector eps_z;
    AlignedDoubleVector denom;
    AlignedDoubleVector screen;
    std::vector<unsigned char> inside;
    std::vector<unsigned char> ion_accessible;
  };

  struct GridReuseStats {
    std::uint64_t workspace_reuse_hits{0};
    std::uint64_t workspace_reallocations{0};
    std::uint64_t medium_cache_hits{0};
    std::uint64_t medium_cache_misses{0};
    std::uint64_t medium_cache_stores{0};
  };

  void collect_atoms(const MOL &mol, const Multi_MOL_parameters &params,
                     std::vector<Atom> &atoms, const char *label) const;
  void collect_test_pqr_atoms(std::vector<Atom> &atoms) const;
  void collect_pqr_atoms(const std::string &pqr_path,
                         std::vector<Atom> &atoms) const;
  [[nodiscard]] std::string
  pqr_path_for_component(const std::string &component) const;
  [[nodiscard]] std::string
  pqr_sha256_for_component(const std::string &component) const;
  void validate_binding_component_union(
      const std::vector<Atom> &complex_atoms,
      const std::vector<Atom> &receptor_atoms,
      const std::vector<Atom> &ligand_atoms) const;
  [[nodiscard]] double intercomponent_coulomb_energy_kcal(
      const std::vector<Atom> &receptor_atoms,
      const std::vector<Atom> &ligand_atoms, double eps) const;
  [[nodiscard]] std::string inferred_element_symbol(const Atom &atom) const;
  [[nodiscard]] double effective_surface_radius(const Atom &atom) const;
  [[nodiscard]] std::string radius_media_source(const Atom &atom) const;
  [[nodiscard]] std::string effective_radius_convention() const;
  [[nodiscard]] bool uses_gaussian_model() const;
  [[nodiscard]] double
  gaussian_influence_radius(const std::vector<Atom> &atoms) const;
  [[nodiscard]] double gaussian_density_at(const std::vector<Atom> &atoms,
                                           double x, double y,
                                           double z) const;
  [[nodiscard]] double gaussian_dielectric(double density, double eps_in,
                                           double eps_out) const;
  [[nodiscard]] std::vector<Atom>
  adjusted_surface_atoms(const std::vector<Atom> &atoms) const;
  GridShape build_shape(const std::vector<Atom> &atoms) const;
  GridShape build_shape(const std::vector<Atom> &atoms, double scale,
                        double margin) const;
  void ensure_workspace(const GridShape &shape, bool reset_medium = true);
  [[nodiscard]] bool pbsa_grid_reuse_gate() const;
  [[nodiscard]] bool grid_medium_operator_cache_allowed(
      const GridShape &shape, const std::vector<Atom> &atoms, double eps_in,
      double eps_out, double kappa, const FrameResult *map_result,
      bool reuse_homogeneous_vacuum) const;
  [[nodiscard]] std::uint64_t grid_medium_operator_cache_key(
      const GridShape &shape, const std::vector<Atom> &atoms, double eps_in,
      double eps_out, double kappa) const;
  [[nodiscard]] bool grid_medium_operator_cache_matches(
      std::uint64_t key, const GridShape &shape) const noexcept;
  bool restore_grid_medium_operator_cache(std::uint64_t key,
                                          const GridShape &shape);
  void store_grid_medium_operator_cache(std::uint64_t key,
                                        const GridShape &shape);
  void classify_surface(const GridShape &shape, const std::vector<Atom> &atoms,
                        TitanExecutor &executor);
  void classify_ion_accessibility(const GridShape &shape,
                                  const std::vector<Atom> &atoms,
                                  TitanExecutor &executor);
  ChargeDiagnostics assign_charges(const GridShape &shape,
                                   const std::vector<Atom> &atoms);
  [[nodiscard]] bool oracle_replay_applies(const GridShape &shape) const;
  void apply_oracle_surface_map(const GridShape &shape);
  [[nodiscard]] ChargeDiagnostics
  assign_oracle_charge_grid(const GridShape &shape,
                            const std::vector<Atom> &atoms);
  void maybe_write_charge_grid_diff(const GridShape &shape,
                                    const std::vector<Atom> &atoms) const;
  void apply_oracle_midpoint_epsilon_map(const GridShape &shape);
  void apply_oracle_operator_maps(const GridShape &shape);
  [[nodiscard]] double delphi_epkt() const noexcept;
  [[nodiscard]] bool delphi_nwt_parity_active() const noexcept;
  [[nodiscard]] bool delphi_field_units_active() const noexcept;
  [[nodiscard]] double field_to_native_factor() const noexcept;
  [[nodiscard]] double effective_rhs_scale(const GridShape &shape) const noexcept;
  [[nodiscard]] double effective_screen_term(const GridShape &shape,
                                             double eps_out, double kappa,
                                             std::size_t id) const noexcept;
  [[nodiscard]] double delphi_relaxation_omega(const GridShape &shape) const;
  [[nodiscard]] double debye_length_from_kappa(double kappa) const noexcept;
  [[nodiscard]] BoundaryFormulaSample boundary_formula_sample(
      const GridShape &shape, const std::vector<Atom> &atoms, std::size_t i,
      std::size_t j, std::size_t k, double eps_out, double kappa) const;
  [[nodiscard]] DelphiVdwMediaMap
  build_delphi_vdw_media_map(const GridShape &shape,
                             const std::vector<Atom> &surface_atoms,
                             TitanExecutor &executor) const;
  void apply_delphi_vdw_media_map(const GridShape &shape,
                                  const DelphiVdwMediaMap &media_map,
                                  double eps_in, double eps_out);
  void maybe_write_midpoint_face_diff(
      const GridShape &shape, const DelphiVdwMediaMap &media_map,
      double eps_in, double eps_out) const;
  [[nodiscard]] BoundaryDiagnostics
  apply_oracle_boundary_map(const GridShape &shape, double eps_out,
                            double kappa);
  void maybe_write_boundary_diff(const GridShape &shape,
                                 const std::vector<Atom> &atoms,
                                 double eps_out, double kappa) const;
  void build_coefficients(const GridShape &shape,
                          const std::vector<Atom> &atoms, double eps_in,
                          double eps_out, double kappa,
                          TitanExecutor &executor);
  BoundaryDiagnostics set_boundary(const GridShape &shape,
                                   const std::vector<Atom> &atoms,
                                   double eps_out, double kappa,
                                   TitanExecutor &executor);
  SolveStats solve_sor(const GridShape &shape, TitanExecutor &executor,
                       int iteration_limit = -1);
  SolveStats solve_sor_half_packed(const GridShape &shape,
                                   TitanExecutor &executor,
                                   int iteration_limit = -1);
  SolveStats solve_nonlinear_sor_block(const GridShape &shape,
                                       const AlignedDoubleVector &source,
                                       int sweeps, double relax,
                                       double screen_scale,
                                       TitanExecutor &executor,
                                       std::size_t &clipped_points);
  SolveStats solve_newton_nwt_sor_block(const GridShape &shape, int sweeps,
                                        double screen_scale,
                                        TitanExecutor &executor,
                                        std::size_t &clipped_points);
  SolveStats solve_newton_nwt_delphi_half_block(const GridShape &shape,
                                                int sweeps,
                                                double screen_scale,
                                                TitanExecutor &executor,
                                                std::size_t &clipped_points);
  [[nodiscard]] bool nwt_half_shape_matches(const GridShape &shape) const
      noexcept;
  void invalidate_nwt_half_workspace() noexcept;
  void prepare_nwt_half_workspace(const GridShape &shape, double rhs_scale);
  void flush_nwt_half_workspace(const GridShape &shape);
  double compute_linear_residual(const GridShape &shape) const;
  [[nodiscard]] double grid_linear_residual_tolerance() const noexcept;
  double compute_nonlinear_residual(const GridShape &shape,
                                    const AlignedDoubleVector &source,
                                    double screen_scale) const;
  std::size_t update_nonlinear_source(const GridShape &shape,
                                      AlignedDoubleVector &source,
                                      double alpha, double clip) const;
  double compute_nonlinear_residual_from_phi(
      const GridShape &shape, const AlignedDoubleVector &field,
      const AlignedDoubleVector &source, double screen_scale) const;
  double compute_nonlinear_residual_vector(
      const GridShape &shape, const AlignedDoubleVector &field,
      const AlignedDoubleVector &source, double screen_scale,
      AlignedDoubleVector &residual, TitanExecutor &executor) const;
  NewtonCorrectionStats
  solve_newton_correction_sor(const GridShape &shape,
                              const AlignedDoubleVector &residual,
                              double residual_rms, double alpha, double clip,
                              double screen_scale, AlignedDoubleVector &delta,
                              int max_sweeps, TitanExecutor &executor) const;
  NewtonCorrectionStats
  solve_newton_correction_vcycle(const GridShape &shape,
                                 const AlignedDoubleVector &residual,
                                 double residual_rms, double alpha,
                                 double clip, double screen_scale,
                                 AlignedDoubleVector &delta,
                                 int max_cycles,
                                 TitanExecutor &executor) const;
  NewtonCorrectionStats
  solve_newton_correction(const GridShape &shape,
                          const AlignedDoubleVector &residual,
                          double residual_rms, double alpha, double clip,
                          double screen_scale, AlignedDoubleVector &delta,
                          int max_iterations, const std::string &inner_solver,
                          TitanExecutor &executor) const;
  FieldSolve solve_newton_nonlinear_field(
      const std::vector<Atom> &atoms, double eps_in, double eps_out,
      double kappa, TitanExecutor &executor, const FrameResult *map_result);
  double grid_reaction_energy_kcal(const GridShape &shape,
                                   const AlignedDoubleVector &solvent_phi,
                                   const AlignedDoubleVector &vacuum_phi) const;
  double grid_phase_energy_kcal(const GridShape &shape,
                                const AlignedDoubleVector &field) const;
  double grid_phase_energy_kcal(const GridShape &shape,
                                const AlignedDoubleVector &field,
                                const AlignedDoubleVector &rho) const;
  double atom_phase_energy_kcal(const std::vector<Atom> &atoms,
                                const std::vector<double> &atom_phi) const;
  double analytic_coulomb_energy_kcal(const std::vector<Atom> &atoms,
                                      double eps) const;
  FieldSolve solve_focused_field(const std::vector<Atom> &atoms, double eps_in,
                                 double eps_out, double kappa,
                                 TitanExecutor &executor,
                                 const FrameResult *map_result);
  FieldSolve solve_vcycle_field(const std::vector<Atom> &atoms, double eps_in,
                                double eps_out, double kappa,
                                TitanExecutor &executor,
                                const FrameResult *map_result);
  FieldSolve solve_linear_field(const std::vector<Atom> &atoms, double eps_in,
                                double eps_out, double kappa,
                                TitanExecutor &executor,
                                const FrameResult *map_result);
  FieldSolve solve_nonlinear_field(const std::vector<Atom> &atoms,
                                   double eps_in, double eps_out, double kappa,
                                   TitanExecutor &executor,
                                   const FrameResult *map_result);
  FieldSolve solve_physical_field(const std::vector<Atom> &atoms,
                                  double eps_in, double eps_out, double kappa,
                                  TitanExecutor &executor,
                                  const FrameResult *map_result);
  FieldSolve solve_grid_field(const std::vector<Atom> &atoms,
                              const GridSolveRequest &request,
                              TitanExecutor &executor);
  void prepare_multigrid_level(MultigridLevel &level, const GridShape &shape,
                               const std::vector<Atom> &atoms, double eps_in,
                               double eps_out, double kappa,
                               bool physical_rhs, bool physical_boundary,
                               TitanExecutor &executor);
  [[nodiscard]] int vcycle_level_cap() const;
  ReactionSolve solve_reaction_field(const std::vector<Atom> &atoms,
                                     TitanExecutor &executor,
                                     const FrameResult *map_result);
  ReactionSolve solve_tabi_reaction_field(const std::vector<Atom> &atoms,
                                          TitanExecutor &executor) const;
  PbsaComponents
  compute_pbsa_components(const MOL &mol1, const Multi_MOL_parameters &params1,
                          const MOL &mol2,
                          const Multi_MOL_parameters &params2,
                          const std::vector<Atom> &mol1_atoms,
                          const std::vector<Atom> &mol2_atoms,
                          const std::vector<Atom> &complex_atoms,
                          double deltaG_solv_kj_mol,
                          TitanExecutor &executor,
                          const EnergyDecompositionProvider *energy_provider)
      const;
  [[nodiscard]] double compute_sasa_total(const std::vector<Atom> &atoms,
                                          TitanExecutor &executor) const;
  [[nodiscard]] std::vector<double>
  compute_sasa_per_atom(const std::vector<Atom> &atoms,
                        TitanExecutor &executor) const;
  [[nodiscard]] std::vector<ResidueFrameRow> compute_pbsa_residue_rows(
      const MOL &mol1, const MOL &mol2, const std::vector<Atom> &mol1_atoms,
      const std::vector<Atom> &mol2_atoms,
      const std::vector<Atom> &complex_atoms,
      const ReactionSolve &complex_solve, const ReactionSolve &mol1_solve,
      const ReactionSolve &mol2_solve, const FrameResult &result,
      TitanExecutor &executor,
      const EnergyDecompositionProvider *energy_provider) const;
  [[nodiscard]] std::string effective_pbsa_decomp() const;
    void accumulate_pbsa_residue_rows(
        const FrameResult &result, const std::vector<ResidueFrameRow> &rows);
    std::vector<Atom> load_probe_atoms(const std::string &path) const;
    std::vector<double> interpolate_atoms(const GridShape &shape,
                                          const std::vector<Atom> &atoms) const;
  bool interpolate_field_value(const GridShape &shape,
                               const AlignedDoubleVector &field, double x,
                               double y, double z, double &value) const;
  void seed_from_focus_source(const GridShape &shape,
                              const GridShape &source_shape,
                              const AlignedDoubleVector &source_phi,
                              TitanExecutor &executor);
  void seed_from_focus_source(const GridShape &shape,
                              const GridShape &source_shape,
                              const AlignedDoubleVector &source_phi,
                              TitanExecutor &executor, bool boundary_only);
  [[nodiscard]] int effective_focus_levels() const noexcept;
  [[nodiscard]] std::string effective_solver_mode() const;
  [[nodiscard]] bool delphi_boundary_convention_active() const noexcept;
  [[nodiscard]] std::string effective_boundary_convention() const;
  [[nodiscard]] double effective_nl_relax() const noexcept;
  [[nodiscard]] double nonlinear_alpha() const noexcept;
  [[nodiscard]] double effective_nonlinear_clip() const noexcept;
  [[nodiscard]] double rhs_scale(const GridShape &shape) const noexcept;
  [[nodiscard]] bool nonlinear_dimensionless_mode() const noexcept;
  [[nodiscard]] double nonlinear_rhs_scale(const GridShape &shape,
                                           double alpha) const noexcept;
  [[nodiscard]] double nonlinear_argument_value(double field_value,
                                                double alpha) const noexcept;
  [[nodiscard]] double nonlinear_source_value(double phi, double alpha,
                                              double clip) const noexcept;
  [[nodiscard]] double nonlinear_source_derivative(double phi, double alpha,
                                                   double clip) const noexcept;
  void scale_nonlinear_field_to_solve_units(AlignedDoubleVector &field,
                                            double alpha) const;
  void scale_nonlinear_field_to_physical_units(AlignedDoubleVector &field,
                                               double alpha) const;
  [[nodiscard]] ParityAudit compute_parity_audit(
      const GridShape &shape, const AlignedDoubleVector *source,
      const AlignedDoubleVector *residual) const;
  [[nodiscard]] ForensicAudit compute_forensic_audit(
      const GridShape &shape) const;
  [[nodiscard]] IscmAudit compute_iscm_audit(
      const GridShape &shape, const std::vector<Atom> &atoms,
      const AlignedDoubleVector &solvent_phi,
      const AlignedDoubleVector &solvent_screen,
      const std::vector<unsigned char> &solvent_inside,
      const AlignedDoubleVector &solvent_eps_x,
      const AlignedDoubleVector &solvent_eps_y,
      const AlignedDoubleVector &solvent_eps_z,
      double grid_vacuum_phase_kcal_mol) const;
  void copy_iscm_audit_to_solve(const IscmAudit &audit,
                                ReactionSolve &solve) const;
  [[nodiscard]] NonlinearEnergyAudit compute_nonlinear_energy_audit(
      const GridShape &shape, const AlignedDoubleVector &solvent_phi,
      const AlignedDoubleVector &solvent_screen) const;
  [[nodiscard]] RpbAudit compute_rpb_source_audit(
      const GridShape &shape, const std::vector<Atom> &atoms,
      TitanExecutor &executor,
      const AlignedDoubleVector &eps_x, const AlignedDoubleVector &eps_y,
      const AlignedDoubleVector &eps_z,
      AlignedDoubleVector *source_out = nullptr,
      AlignedDoubleVector *coulomb_out = nullptr,
      double phase_eps_out = std::numeric_limits<double>::quiet_NaN()) const;
  void maybe_write_rpb_source_audit(const FrameResult &result,
                                    const RpbAudit &audit) const;
  void maybe_write_rpb_source_top(const FrameResult &result,
                                  const GridShape &shape,
                                  const AlignedDoubleVector &source,
                                  const AlignedDoubleVector &coulomb) const;
  void copy_nonlinear_energy_audit_to_solve(
      const NonlinearEnergyAudit &audit, ReactionSolve &solve) const;
  void update_newton_derivative_audit(const GridShape &shape,
                                      const AlignedDoubleVector &source,
                                      const AlignedDoubleVector &delta,
                                      double alpha, double clip,
                                      double screen_scale,
                                      ParityAudit &audit) const;
  void write_energy_header_if_needed();
  void append_energy_row(const FrameResult &result);
  [[nodiscard]] bool pbe_timing_enabled() const noexcept;
  void add_timing_ms(TimingBucket bucket, double ms) const noexcept;
  void write_timing_header_if_needed();
  void append_timing_row(const FrameResult &result,
                         const TimingBuckets &timing);
  void write_tabi_profile_header_if_needed();
  void append_tabi_profile_row(const FrameResult &result,
                               const TimingBuckets &timing);
  void write_tabi_audit_profile_header_if_needed();
  void append_tabi_audit_profile_row(const FrameResult &result,
                                     const TimingBuckets &timing);
  void write_memory_profile_header_if_needed();
  void append_memory_profile_row(const FrameResult &result,
                                 const TimingBuckets &timing);
  void write_tabi_formulation_audit_header_if_needed() const;
  void append_tabi_formulation_audit_row(const FrameResult &result) const;
  void write_tabi_near_block_audit_header_if_needed();
  void append_tabi_near_block_audit_rows(const FrameResult &result);
  void write_tabi_geometric_near_audit_header_if_needed();
  void append_tabi_geometric_near_audit_rows(const FrameResult &result);
  void write_tabi_far_geometric_audit_header_if_needed();
  void append_tabi_far_geometric_audit_row(const FrameResult &result);
  void write_tabi_moment_audit_header_if_needed();
  void append_tabi_moment_audit_row(const FrameResult &result);
  void write_tabi_surface_quality_header_if_needed();
  void append_tabi_surface_quality_row(const FrameResult &result);
  void maybe_write_tabi_surface_solution(const FrameResult &result,
                                         const ReactionSolve &solve) const;
  [[nodiscard]] bool stability_audit_enabled() const noexcept;
  [[nodiscard]] bool stability_detail_enabled() const noexcept;
  [[nodiscard]] std::string effective_grid_anchor() const;
  [[nodiscard]] bool stability_grid_anchor_enabled() const noexcept;
  [[nodiscard]] bool stability_hysteresis_enabled() const noexcept;
  [[nodiscard]] StabilityAudit
  compute_stability_audit(const FrameResult &result,
                          const std::vector<Atom> &atoms,
                          const ReactionSolve &solve) const;
  void maybe_write_stability_audit(const FrameResult &result,
                                   const std::vector<Atom> &atoms,
                                   const ReactionSolve &solve);
  [[nodiscard]] double
  compute_anchor_margin_min_A(const GridShape &shape,
                              const std::vector<Atom> &atoms) const;
  void apply_stability_hysteresis(const GridShape &shape,
                                  const std::vector<Atom> &atoms);
  void store_stability_medium_state(const GridShape &shape);
  void compute_stability_ensemble(const std::vector<Atom> &atoms,
                                  TitanExecutor &executor,
                                  const FrameResult &result);
  void maybe_write_parity_stage(const FrameResult &result,
                                const ReactionSolve &solve) const;
  [[nodiscard]] bool barnase_forensic_enabled() const;
  void maybe_write_barnase_forensic_audits(
      const FrameResult &result, const std::vector<Atom> &atoms,
      const FieldSolve &solvent, const FieldSolve &vacuum,
      const ReactionSolve &solve, const ForensicAudit &solvent_audit) const;
  void maybe_write_iscm_surface_charge_samples(
      const FrameResult &result, const IscmAudit &audit) const;
  void maybe_write_atom_phi_audit(const FrameResult &result,
                                  const std::vector<Atom> &atoms,
                                  const FieldSolve &solvent,
                                  const FieldSolve &vacuum,
                                  const ReactionSolve &solve) const;
  void maybe_write_atom_reaction_potentials(
      const FrameResult &result, const std::vector<Atom> &atoms,
      const ReactionSolve &solve, const std::string &component) const;
  void maybe_write_binding_electrostatics(
      const FrameResult &result, const std::vector<Atom> &complex_atoms,
      const std::vector<Atom> &receptor_atoms,
      const std::vector<Atom> &ligand_atoms,
      const ReactionSolve &complex_solve,
      const ReactionSolve &receptor_solve,
      const ReactionSolve &ligand_solve) const;
  void maybe_write_linear_forensic_points(
      const FrameResult &result, const std::vector<Atom> &atoms,
      const FieldSolve &solvent, const FieldSolve &vacuum,
      const ReactionSolve &solve) const;
  void maybe_write_parity_points(const FrameResult &result,
                                 const GridShape &shape,
                                 const AlignedDoubleVector &source,
                                 const AlignedDoubleVector &residual,
                                 const AlignedDoubleVector &delta,
                                 double alpha, double clip,
                                 double screen_scale) const;
  [[nodiscard]] bool parity_map_dump_enabled() const noexcept;
  [[nodiscard]] std::vector<MapSamplePoint>
  collect_map_sample_points(const GridShape &shape,
                            const std::vector<Atom> &atoms,
                            const ReactionSolve &solve) const;
  void maybe_write_parity_map_dumps(const FrameResult &result,
                                    const std::vector<Atom> &atoms,
                                    const FieldSolve &solvent,
                                    const FieldSolve &vacuum,
                                    const ReactionSolve &solve) const;
  void accumulate_pbsa(const FrameResult &result);
  [[nodiscard]] PbsaEntropyResult compute_pbsa_entropy() const;
  [[nodiscard]] PbsaEntropyResult
  compute_pbsa_entropy_for_values(const std::vector<double> &values) const;
  void write_pbsa_energy_csv(const PbsaEntropyResult &entropy) const;
  void write_pbsa_decomposition(const PbsaEntropyResult &entropy) const;
  void write_pbsa_residue_outputs() const;
  void write_pbsa_unit_outputs() const;
  void write_pbsa_by_trajectory() const;
  void write_pbsa_ie_diagnostics(const PbsaEntropyResult &global_entropy) const;
  void write_pbsa_performance() const;
  void write_pbsa_summary(std::fstream &log);
  static void add_scalar(ScalarStats &stats, double value);
  static double scalar_stddev(const ScalarStats &stats);
  static double scalar_mean_or_nan(const ScalarStats &stats);
  static double scalar_stddev_or_nan(const ScalarStats &stats);
  void maybe_write_maps(const GridShape &shape, const FrameResult &result,
                        bool write_phi_now);
  [[nodiscard]] std::string maybe_write_probe_sites(
      const FrameResult &result, const std::vector<Atom> &atoms,
      const ReactionSolve &solve);
  void write_cube(const std::string &filename, const GridShape &shape,
                  const std::string &field_name, bool epsilon_field,
                  bool ion_field = false) const;
  [[nodiscard]] bool should_write_map() const;
  [[nodiscard]] std::string map_suffix(const FrameResult &result) const;
  [[nodiscard]] GridShape apply_grid_anchor(const GridShape &shape,
                                            double scale,
                                            double margin) const;
  void reset_grid_anchor() const noexcept;

  PbeArgs args_{};
  std::size_t configured_threads_{1};
  std::string current_source_label_{};
  std::size_t current_trajectory_id_{0};

  std::ofstream energy_out_;
  std::ofstream timing_out_;
  std::ofstream tabi_profile_out_;
  std::ofstream tabi_audit_profile_out_;
  std::ofstream memory_profile_out_;
  mutable std::ofstream tabi_formulation_audit_out_;
  std::ofstream tabi_near_block_audit_out_;
  std::ofstream tabi_geometric_near_audit_out_;
  std::ofstream tabi_far_geometric_audit_out_;
  std::ofstream tabi_moment_audit_out_;
  std::ofstream tabi_surface_quality_out_;
  std::ofstream stability_audit_out_;
  std::ofstream stability_samples_out_;
  bool energy_header_written_{false};
  mutable bool parity_stage_header_written_{false};
  mutable bool parity_points_header_written_{false};
  mutable bool forensic_inputs_header_written_{false};
  mutable bool forensic_surface_header_written_{false};
  mutable bool forensic_operator_header_written_{false};
  mutable bool radius_media_audit_header_written_{false};
  mutable bool atom_phi_audit_header_written_{false};
  mutable bool atom_reaction_potential_header_written_{false};
  mutable bool residue_electrostatics_header_written_{false};
  mutable bool component_electrostatics_header_written_{false};
  mutable bool binding_electrostatics_header_written_{false};
  mutable std::map<std::string, std::string> pqr_sha256_cache_;
  mutable bool iscm_surface_samples_header_written_{false};
  mutable bool iscm_energy_terms_header_written_{false};
  mutable bool field_samples_header_written_{false};
  mutable bool boundary_samples_header_written_{false};
  mutable bool charge_grid_audit_header_written_{false};
  mutable bool charge_grid_diff_header_written_{false};
  mutable bool midpoint_face_diff_header_written_{false};
  mutable bool native_midpoint_state_header_written_{false};
  mutable bool vdwms_action_histogram_header_written_{false};
  mutable bool boundary_diff_header_written_{false};
  mutable bool operator_stage_header_written_{false};
  mutable bool sor_trace_header_written_{false};
  mutable bool nwt_trace_header_written_{false};
  bool timing_header_written_{false};
  bool tabi_profile_header_written_{false};
  bool tabi_audit_profile_header_written_{false};
  bool memory_profile_header_written_{false};
  mutable bool tabi_formulation_audit_header_written_{false};
  bool tabi_near_block_audit_header_written_{false};
  bool tabi_geometric_near_audit_header_written_{false};
  bool tabi_far_geometric_audit_header_written_{false};
  bool tabi_moment_audit_header_written_{false};
  bool tabi_surface_quality_header_written_{false};
  bool stability_audit_header_written_{false};
  bool stability_samples_header_written_{false};
  bool wrote_first_map_{false};
  mutable TimingBuckets *active_timing_{nullptr};
  bool pbsa_reuse_vacuum_active_{false};
  GridShape pbsa_reuse_shape_{};
  GridMediumOperatorCache grid_medium_operator_cache_{};
  GridReuseStats grid_reuse_stats_{};
  EDecompSink edecomp_sink_{};
  mutable std::map<std::string, GridShape> grid_anchor_shapes_;
  StabilityFrameState last_stability_state_{};
  StabilityMediumState last_stability_medium_state_{};
  StabilityEnsembleState last_stability_ensemble_{};
  std::size_t stability_hysteresis_cells_kept_frame_{0};
  std::size_t stability_hysteresis_cells_released_frame_{0};
  bool stability_probe_solve_active_{false};

  std::vector<Atom> atoms_;
  GridWorkspace grid_;
  NwtHalfWorkspace nwt_half_workspace_;
  AlignedDoubleVector last_gaussian_density_center_;
  AlignedDoubleVector last_gaussian_density_midpoints_;
  AlignedDoubleVector last_gaussian_eps_center_;
  AlignedDoubleVector last_gaussian_surface_;
  AlignedDoubleVector last_gaussian_grad_surface_;
  std::vector<unsigned char> last_iscm_surface_valid_;
  std::vector<std::array<double, 3>> last_iscm_surface_positions_;
  double last_gaussian_density_sum_{0.0};
  std::size_t last_gaussian_cutoff_points_{0};
  double last_gaussian_eps_min_{0.0};
  double last_gaussian_eps_max_{0.0};
  double last_gaussian_eps_mean_{0.0};
  VdwmsAudit last_vdwms_audit_{};

  std::size_t frames_total_{0};
  std::size_t frames_converged_{0};
  std::size_t frames_failed_{0};
  // Frames whose energy entered energy_mean_ and friends: every grid frame,
  // and the BIE frames that have a finite energy.
  std::size_t energy_frames_{0};
  std::size_t max_grid_points_seen_{0};
  std::size_t max_memory_bytes_seen_{0};
  std::size_t max_tabi_surface_panels_seen_{0};
  std::size_t max_tabi_tree_nodes_seen_{0};
  std::size_t max_tabi_tree_leaf_count_seen_{0};
  std::size_t max_tabi_near_interactions_seen_{0};
  std::size_t max_tabi_far_interactions_seen_{0};
  std::size_t max_tabi_mesh_vertices_seen_{0};
  std::size_t max_tabi_mesh_triangles_seen_{0};
  std::size_t max_tabi_boundary_nodes_seen_{0};
  std::size_t max_tabi_nonmanifold_edges_seen_{0};
  std::size_t max_tabi_solid_angle_fallbacks_seen_{0};
  double min_tabi_diag_seen_{1.0};
  double max_tabi_diag_seen_{1.0};
  std::size_t max_solute_points_seen_{0};
  std::size_t max_solvent_points_seen_{0};
  std::size_t max_ion_accessible_points_seen_{0};
  std::size_t max_stern_excluded_points_seen_{0};
  int max_focus_levels_seen_{1};
  int max_mg_levels_seen_{1};
  int max_mg_cycles_seen_{0};
  std::size_t nonlinear_frames_{0};
  std::size_t nonlinear_converged_{0};
  std::size_t nonlinear_failed_{0};
  int max_nl_blocks_seen_{0};
  int max_nl_iterations_seen_{0};
  std::size_t max_nl_clipped_points_seen_{0};
  double max_nl_dphi_seen_{0.0};
  double max_nl_residual_seen_{0.0};
  int max_newton_inner_iterations_seen_{0};
  double max_newton_step_norm_seen_{0.0};
  double max_newton_residual_before_seen_{0.0};
  double max_newton_residual_after_seen_{0.0};
  double max_newton_correction_residual_seen_{0.0};
  double max_newton_correction_max_residual_seen_{0.0};
  int max_newton_rejected_steps_seen_{0};
  int max_newton_inner_levels_seen_{1};
  int max_newton_inner_cycles_seen_{0};
  double max_newton_inner_residual_before_seen_{0.0};
  double max_newton_inner_residual_after_seen_{0.0};
  double max_linear_residual_seen_{0.0};
  double max_boundary_rms_seen_{0.0};
  double max_boundary_max_seen_{0.0};
  double mg_cycle_mean_{0.0};
  double energy_mean_{0.0};
  double energy_m2_{0.0};
  double energy_min_{0.0};
  double energy_max_{0.0};
  double iter_mean_{0.0};
  std::size_t dgsolv_frames_{0};
  double dgsolv_mean_{0.0};
  double dgsolv_m2_{0.0};
  double dgsolv_min_{0.0};
  double dgsolv_max_{0.0};
  std::size_t pbsa_frames_{0};
  std::size_t pbsa_valid_{0};
  std::size_t pbsa_failed_{0};
  ScalarStats pbsa_elec_stats_{};
  ScalarStats pbsa_vdw_stats_{};
  ScalarStats pbsa_mm_stats_{};
  ScalarStats pbsa_delta_sasa_stats_{};
  ScalarStats pbsa_nonpolar_stats_{};
  ScalarStats pbsa_polar_stats_{};
  ScalarStats pbsa_no_entropy_stats_{};
  std::vector<double> pbsa_ie_values_kj_;
  std::vector<FrameResult> pbsa_rows_;
  std::vector<PbsaEndpointTimingRow> pbsa_endpoint_timing_rows_;
  std::vector<ResidueFrameRow> pbsa_residue_frame_rows_;
  std::map<std::string, ResidueStats> pbsa_residue_stats_;
  std::map<std::string, ResidueStats> pbsa_residue_traj_stats_;
  double pbsa_entropy_output_ms_{0.0};
  double pbsa_residue_closure_max_abs_kj_{0.0};
  double pbsa_residue_closure_elec_max_abs_kj_{0.0};
  double pbsa_residue_closure_vdw_max_abs_kj_{0.0};
  double pbsa_residue_closure_polar_max_abs_kj_{0.0};
  double pbsa_residue_closure_nonpolar_max_abs_kj_{0.0};
};

} // namespace TitanPBE

#endif
