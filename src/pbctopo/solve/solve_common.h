#pragma once

// Internals of the PBCTOPO solver (Multi_MOL::solve_pbc_pbctopo): types,
// constants and helpers shared by the src/pbctopo/solve/*.cpp files.
// Only those files include this header.

#include "Multi_MOL.h"
#include "BoxMatrix.h"
#include "pbctopo/config.h"
#include "pbctopo/anchor_graph.h"
#include "pbctopo/assignment_set_report.h"
#include "pbctopo/candidate_order.h"
#include "pbctopo/candidate_scoring.h"
#include "pbctopo/certification.h"
#include "pbctopo/evidence_audit.h"
#include "pbctopo/evidence_relation_report.h"
#include "pbctopo/component_graph.h"
#include "pbctopo/component_metric.h"
#include "pbctopo/component_refine.h"
#include "pbctopo/component_search.h"
#include "pbctopo/component_shift_domain.h"
#include "pbctopo/llps_local.h"
#include "pbctopo/image_assignment.h"
#include "pbctopo/identifiability.h"
#include "pbctopo/lattice_math.h"
#include "pbctopo/local_graph.h"
#include "pbctopo/logging.h"
#include "pbctopo/observation_binding.h"
#include "pbctopo/observation_layout.h"
#include "pbctopo/observation_schema.h"
#include "pbctopo/periodic_contact_audit.h"
#include "pbctopo/report.h"
#include "pbctopo/required_contacts.h"
#include "pbctopo/reference_strain.h"
#include "pbctopo/selected_report.h"
#include "pbctopo/spatial_lift.h"
#include "pbctopo/steric_clash.h"
#include "pbctopo/temporal_path.h"
#include "pbctopo/temporal_replay.h"
#include "pbctopo/types.h"
#include "pbctopo/uncertainty_band.h"
#include "pbctopo/virtual_seed.h"
#include "pbctopo/virtual_discrete_search.h"
#include "pbctopo/warm_state.h"
#include "pbctopo_hull.h"
#include "titan_gyration.h"
#include "titan_simd_dispatch.h"
#include "titan_executor.h"
#include "titan_ui.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pbctopo_solve {

using namespace std;

using pbctopo_hull::cross;
using pbctopo_hull::dot;
using pbctopo_hull::norm2;
using pbctopo_hull::PbctopoHullScratch;
using pbctopo_hull::PbctopoHullSpan;
using pbctopo_hull::Vec3;
using titan_gyration::build_point_stats;
using titan_gyration::compute_bbox_volume;
using titan_gyration::compute_gyration_shape_metrics;
using titan_gyration::compute_principal_rg2;
using titan_gyration::compute_rg2;
using titan_gyration::compute_rg2_from_stats;
using titan_gyration::GyrationShapeMetrics;
using titan_gyration::PointStats;
using PbctopoAnchorBackend = titan_pbctopo::AnchorBackend;
using PbctopoAnchorMetricMode = titan_pbctopo::AnchorMetricMode;
using PbctopoRuntimeCutoffs = titan_pbctopo::RuntimeCutoffs;
using PbctopoSolveMode = titan_pbctopo::SolveMode;
using PbctopoSearchPolicy = titan_pbctopo::SearchPolicy;
using PbctopoTemporalMode = titan_pbctopo::TemporalMode;
using PbctopoTemporalCandidate = titan_pbctopo::PbctopoTemporalCandidate;
using PbctopoAnchorContactEdge = titan_pbctopo::PbctopoAnchorContactEdge;
using PbctopoAnchorGraphContext = titan_pbctopo::PbctopoAnchorGraphContext;
using PbctopoAnchorGraphScore = titan_pbctopo::PbctopoAnchorGraphScore;
using PbctopoAtom = titan_pbctopo::PbctopoAtom;
using PbctopoComponentCandidate = titan_pbctopo::PbctopoComponentCandidate;
using PbctopoComponentCandidatePrefixStage =
    titan_pbctopo::PbctopoComponentCandidatePrefixStage;
using PbctopoComponentBaseGeometry = titan_pbctopo::PbctopoComponentBaseGeometry;
using PbctopoComponentSearchCallbacks =
    titan_pbctopo::PbctopoComponentSearchCallbacks;
using PbctopoComponentSearchSurvivor =
    titan_pbctopo::PbctopoComponentSearchSurvivor;
using PbctopoContinuityPair = titan_pbctopo::PbctopoContinuityPair;
using PbctopoCutScoreBreakdown = titan_pbctopo::PbctopoCutScoreBreakdown;
using PbctopoCutCandidate = titan_pbctopo::PbctopoCutCandidate;
using PbctopoCycleSyncMetrics = titan_pbctopo::PbctopoCycleSyncMetrics;
using PbctopoCycleSyncResult = titan_pbctopo::PbctopoCycleSyncResult;
using PbctopoEdgeScanSoA = titan_pbctopo::PbctopoEdgeScanSoA;
using PbctopoFailureReason = titan_pbctopo::PbctopoFailureReason;
using PbctopoCertificateScope = titan_pbctopo::PbctopoCertificateScope;
using PbctopoCertificateGraphSource =
    titan_pbctopo::PbctopoCertificateGraphSource;
using PbctopoHardGraphSource = titan_pbctopo::PbctopoHardGraphSource;
using PbctopoSearchEvidenceSource =
    titan_pbctopo::PbctopoSearchEvidenceSource;
using PbctopoFrameMode = titan_pbctopo::PbctopoFrameMode;
using PbctopoFrameReport = titan_pbctopo::PbctopoFrameReport;
using PbctopoFrameStatus = titan_pbctopo::PbctopoFrameStatus;
using PbctopoEvidenceState = titan_pbctopo::PbctopoEvidenceState;
using PbctopoContactHypothesisSet =
    titan_pbctopo::PbctopoContactHypothesisSet;
using PbctopoSelectedEvidenceAudit =
    titan_pbctopo::PbctopoSelectedEvidenceAudit;
using PbctopoGraphEdge = titan_pbctopo::PbctopoGraphEdge;
using PbctopoHardFeasibility = titan_pbctopo::PbctopoHardFeasibility;
using PbctopoImageAssignment = titan_pbctopo::PbctopoImageAssignment;
using PbctopoImageEquationResiduals =
    titan_pbctopo::PbctopoImageEquationResiduals;
using PbctopoIdentifiability = titan_pbctopo::PbctopoIdentifiability;
using PbctopoLocalEdge = titan_pbctopo::PbctopoLocalEdge;
using PbctopoLocalEdgeKind = titan_pbctopo::PbctopoLocalEdgeKind;
using PbctopoComponentLayoutObservation =
    titan_pbctopo::PbctopoComponentLayoutObservation;
using PbctopoLlpsLocalUnits = titan_pbctopo::PbctopoLlpsLocalUnits;
using PbctopoPairShiftCandidate = titan_pbctopo::PbctopoPairShiftCandidate;
using PbctopoPairShiftCandidateSet = titan_pbctopo::PbctopoPairShiftCandidateSet;
using PbctopoPairShiftScore = titan_pbctopo::PbctopoPairShiftScore;
using PbctopoRefineScorePhase = titan_pbctopo::PbctopoRefineScorePhase;
using PbctopoRefineScoringCallbacks =
    titan_pbctopo::PbctopoRefineScoringCallbacks;
using PbctopoScoreVector = titan_pbctopo::PbctopoScoreVector;
using PbctopoShift = titan_pbctopo::PbctopoShift;
using PbctopoShiftedComponent = titan_pbctopo::PbctopoShiftedComponent;
using PbctopoShiftedComponentDescriptor =
    titan_pbctopo::PbctopoShiftedComponentDescriptor;
using PbctopoStage1EdgeContext = titan_pbctopo::PbctopoStage1EdgeContext;
using PbctopoRange = titan_pbctopo::PbctopoRange;
using PbctopoSystemStaticData = titan_pbctopo::PbctopoSystemStaticData;
using PbctopoTrajectoryContext = titan_pbctopo::PbctopoTrajectoryContext;
using PbctopoTimingStats = titan_pbctopo::PbctopoTimingStats;
using PbctopoVirtualSeedCenter = titan_pbctopo::PbctopoVirtualSeedCenter;
using PbctopoVirtualSeedSource = titan_pbctopo::PbctopoVirtualSeedSource;
using PbctopoLatticeMetric = titan_pbctopo::PbctopoLatticeMetric;
using PbctopoLexicographicGap = titan_pbctopo::PbctopoLexicographicGap;
using PbctopoVirtualDiscreteSearchOptions =
    titan_pbctopo::PbctopoVirtualDiscreteSearchOptions;
using PbctopoVirtualDiscreteSearchPlan =
    titan_pbctopo::PbctopoVirtualDiscreteSearchPlan;
using PbctopoVirtualDiscreteSeedHint =
    titan_pbctopo::PbctopoVirtualDiscreteSeedHint;
using PbctopoWarmState = titan_pbctopo::PbctopoWarmState;
using PbctopoComponentLayoutSignature =
    titan_pbctopo::PbctopoComponentLayoutSignature;
using ClashCellKey = titan_pbctopo::ClashCellKey;
using ClashCellKeyHash = titan_pbctopo::ClashCellKeyHash;
using Int3 = titan_pbctopo::Int3;
using titan_pbctopo::is_zero_delta;
using titan_pbctopo::same_delta;
using titan_pbctopo::anchor_graph_metrics_enhanced;
using titan_pbctopo::build_anchor_graph_context;
using titan_pbctopo::build_atom_component_index;
using titan_pbctopo::build_coarse_anchor_subset;
using titan_pbctopo::build_component_graph_edge_candidates;
using titan_pbctopo::build_component_points_from_offsets;
using titan_pbctopo::build_edge_scan_soa;
using titan_pbctopo::build_stage1_edge_context;
using titan_pbctopo::build_tapered_stage1_anchor_subset;
using titan_pbctopo::build_third_anchor_subset;
using titan_pbctopo::clear_component_trial_hints;
using titan_pbctopo::compare_anchor_graph_score;
using titan_pbctopo::compare_pbctopo_metric;
using titan_pbctopo::component_trial_hints_compatible;
using titan_pbctopo::better_component_graph_edge;
using titan_pbctopo::better_component_candidate;
using titan_pbctopo::better_cut_candidate;
using titan_pbctopo::better_preclash_cut_candidate;
using titan_pbctopo::better_stage1_cut_candidate;
using titan_pbctopo::candidate_close_enough_to_skip_search;
using titan_pbctopo::candidate_has_projection_clash_margin;
using titan_pbctopo::candidate_lexicographic_gap;
using titan_pbctopo::candidate_temporal_emission_cost;
using titan_pbctopo::candidate_skip_search_reject_reason;
using titan_pbctopo::component_candidate_cannot_beat_incumbent_prefix;
using titan_pbctopo::component_candidate_is_outlier;
using titan_pbctopo::component_candidate_to_cut_candidate;
using titan_pbctopo::compare_candidate_prefix_before_edge_stretch;
using titan_pbctopo::compare_pbctopo_double_metric;
using titan_pbctopo::compare_principal_rg2;
using titan_pbctopo::compare_shape_kappa;
using titan_pbctopo::compare_shape_moment;
using titan_pbctopo::cut_candidate_cannot_beat_incumbent;
using titan_pbctopo::make_score_vector;
using titan_pbctopo::pbctopo_symmetric_reference_strain;
using titan_pbctopo::preclash_cut_candidate_cannot_beat_incumbent;
using titan_pbctopo::graph_edge_has_zero_shift;
using titan_pbctopo::graph_edge_matches_delta;
using titan_pbctopo::append_atoms_for_system;
using titan_pbctopo::append_metadata_graph_edges_for_system;
using titan_pbctopo::append_topology_bond_edges_for_system;
using titan_pbctopo::build_pbctopo_llps_local_units;
using titan_pbctopo::build_pbctopo_trajectory_context;
using titan_pbctopo::improve_component_offsets_locally;
using titan_pbctopo::make_pbctopo_image_assignment;
using titan_pbctopo::build_heavy_atom_subset;
using titan_pbctopo::build_hard_connected_components_from_local_edges;
using titan_pbctopo::build_local_neighborhood_graph;
using titan_pbctopo::build_projection_anchor_subset;
using titan_pbctopo::build_projection_continuity_subset;
using titan_pbctopo::build_projection_edge_subset;
using titan_pbctopo::build_ranges_for_system;
using titan_pbctopo::merge_pbctopo_llps_unit_source;
using titan_pbctopo::pbctopo_topology_source_label;
using titan_pbctopo::pbctopo_atoms_clash;
using titan_pbctopo::pbctopo_max_clash_cutoff;
using titan_pbctopo::pbctopo_relative_images_identified;
using titan_pbctopo::pbctopo_component_shift_metric;
using titan_pbctopo::pbctopo_candidate_anchor_ambiguous;
using titan_pbctopo::populate_pbctopo_selected_candidate_report;
using titan_pbctopo::prepare_pbctopo_steric_radii;
using titan_pbctopo::certify_pbctopo_candidate;
using titan_pbctopo::evaluate_pbctopo_fixed_certificate_invariants;
using titan_pbctopo::evaluate_pbctopo_image_equations;
using titan_pbctopo::rebase_component_offsets;
using titan_pbctopo::replay_pbctopo_assignment_delta;
using titan_pbctopo::same_component_layout;
using titan_pbctopo::build_component_layout_signature;
using titan_pbctopo::commit_pbctopo_certified_warm_state;
using titan_pbctopo::reset_pbctopo_warm_state;
using titan_pbctopo::same_pbctopo_image_assignment;
using titan_pbctopo::score_component_offset_solution;
using titan_pbctopo::score_component_placed_points_solution;
using titan_pbctopo::search_component_shift_candidate;
using titan_pbctopo::search_component_shift_candidate_beam;
using titan_pbctopo::PbctopoComponentSearchDiagnostics;
using titan_pbctopo::PbctopoComponentSearchLimits;
using titan_pbctopo::PbctopoSearchStopReason;
using titan_pbctopo::kPbctopoExactAssignmentBudget;
using titan_pbctopo::pbctopo_component_assignment_upper_bound;
using titan_pbctopo::pbctopo_search_stop_reason_name;
using titan_pbctopo::solve_component_offsets_from_ranked_edges;
using titan_pbctopo::solve_component_offsets_via_pair_graph;
using titan_pbctopo::unwrap_pbctopo_llps_local_points;
using titan_pbctopo::write_pbctopo_frame_report_header;
using titan_pbctopo::write_pbctopo_frame_report_row;
using titan_pbctopo::write_pbctopo_assignment_set_header;
using titan_pbctopo::write_pbctopo_assignment_set_rows;
using titan_pbctopo::write_pbctopo_candidate_evidence_relation_header;
using titan_pbctopo::write_pbctopo_candidate_evidence_summary_header;
using titan_pbctopo::write_pbctopo_candidate_evidence_frame;
using titan_pbctopo::upsert_pbctopo_candidate_evidence_audit_snapshot;
using titan_pbctopo::write_pbctopo_component_layout_header;
using titan_pbctopo::write_pbctopo_component_layout_rows;
using titan_pbctopo::write_pbctopo_hard_edge_header;
using titan_pbctopo::write_pbctopo_hard_edge_rows;
using titan_pbctopo::extrapolate_virtual_seed_center;
using titan_pbctopo::build_pbctopo_virtual_discrete_search_plan;

constexpr double kDistanceEps = 1.0e-12;
constexpr size_t kPbctopoDenseClashGridMaxCells = 1u << 24;
constexpr size_t kPbctopoExactComponentLimit =
    titan_pbctopo::kDefaultExactComponentLimit;
constexpr size_t kPbctopoMergeModeMaxRawComponents = 16;
constexpr size_t kPbctopoMergedNodeMaxAtoms = 4096;
constexpr size_t kPbctopoMergedNodeMaxRawComponents = 4;
constexpr size_t kPbctopoBeamWidth = titan_pbctopo::kDefaultBeamWidth;
constexpr size_t kPbctopoExactComponentLimitMax =
    titan_pbctopo::kMaxExactComponentLimit;
constexpr size_t kPbctopoBeamWidthMax = titan_pbctopo::kMaxBeamWidth;

struct PbctopoComponentLayoutReportState {
  bool initialized = false;
  std::filesystem::path initialized_cwd;
  vector<PbctopoComponentLayoutObservation> emitted_layouts;
};

struct PbctopoH9ReportState {
  bool initialized = false;
  std::filesystem::path initialized_cwd;
};

struct PbctopoFragment {
  size_t id = 0;
  int owner = 0;
  size_t source_index = 0;
  size_t begin_local = 0;
  size_t end_local = 0;
  size_t base_atom_begin = 0;
  size_t base_atom_end = 0;
  size_t atom_count = 0;
  bool topo_backed = false;
};

struct PbctopoAtomCopy {
  size_t global_atom_id = 0;
  size_t fragment_id = 0;
  size_t node_id = 0;
  int owner = 0;
  size_t local_atom_index = 0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct PbctopoFragmentCopyNode {
  size_t node_id = 0;
  size_t fragment_id = 0;
  size_t shift_index = 26;
  int shift_ix = 0;
  int shift_iy = 0;
  int shift_iz = 0;
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;
  size_t atom_begin = 0;
  size_t atom_end = 0;
  double min_x = 0.0;
  double min_y = 0.0;
  double min_z = 0.0;
  double max_x = 0.0;
  double max_y = 0.0;
  double max_z = 0.0;
};

struct PbctopoOwnerPairSplitDetail {
  int owner_a = -1;
  int owner_b = -1;
  Vec3 raw_delta_scaled{};
  Vec3 minimg_delta_scaled{};
  double raw_norm = 0.0;
  double minimg_norm = 0.0;
  bool valid = false;
};

struct PbctopoTranslationCache {
  Vec3 by_shift[5][5][5]{};
};

struct PbctopoCandidate {
  vector<size_t> node_ids;
  double hull_volume = numeric_limits<double>::infinity();
  double bbox_volume = numeric_limits<double>::infinity();
  double rg2 = numeric_limits<double>::infinity();
  double shape_kappa2 = numeric_limits<double>::infinity();
  double shape_asphericity = numeric_limits<double>::infinity();
  double shape_acylindricity = numeric_limits<double>::infinity();
  double fragment_mst2 = numeric_limits<double>::infinity();
  double fragment_rg2 = numeric_limits<double>::infinity();
  double fragment_kappa2 = numeric_limits<double>::infinity();
  double fragment_asphericity = numeric_limits<double>::infinity();
  double fragment_acylindricity = numeric_limits<double>::infinity();
  double shift_norm2 = numeric_limits<double>::infinity();
  double min_interfragment_d2 = numeric_limits<double>::infinity();
  size_t connectivity_edges = 0;
  bool hull_valid = false;
  bool hard_clash = false;
  bool valid = false;
};

struct PbctopoCutOption {
  double cut = 0.0;
  double gap = 0.0;
};

struct PbctopoSeed2D {
  double cx = 1.5;
  double cy = 1.5;
  double score = -numeric_limits<double>::infinity();
  size_t ix = 0;
  size_t iy = 0;
};

struct PbctopoSeed3D {
  double cx = 1.5;
  double cy = 1.5;
  double cz = 1.5;
  double score = -numeric_limits<double>::infinity();
  size_t ix = 0;
  size_t iy = 0;
  size_t iz = 0;
  PbctopoVirtualSeedSource source = PbctopoVirtualSeedSource::Density;
};

enum class PbctopoComponentMode {
  Disabled,
  Exact,
  MergeThenGraph,
  Beam,
};

struct PbctopoMergedComponentPlan {
  vector<vector<size_t>> merged_components;
  vector<vector<size_t>> merged_to_raw;
  vector<size_t> raw_to_merged;
  vector<Int3> raw_internal_offsets;
};

struct PbctopoGraphSeedQuickScore {
  bool valid = false;
  double continuity_max_d2 = numeric_limits<double>::infinity();
  double continuity_path = numeric_limits<double>::infinity();
  double anchor_max_d2 = numeric_limits<double>::infinity();
  double anchor_mst2 = numeric_limits<double>::infinity();
  PbctopoAnchorGraphScore anchor_graph;
  double principal_rg2_max = numeric_limits<double>::infinity();
  double principal_rg2_mid = numeric_limits<double>::infinity();
  double principal_rg2_min = numeric_limits<double>::infinity();
  double shift_norm2 = numeric_limits<double>::infinity();
};

struct PbctopoCoarseSeedOption {
  uint64_t state_key = 0;
  PbctopoSeed3D seed;
};

struct PbctopoFrameScratch {
  vector<PbctopoAtom> base_atoms;
  vector<PbctopoAtom> projection_atoms;
  vector<PbctopoAtom> component_atoms;
  vector<PbctopoLocalEdge> local_edges;
  vector<PbctopoLocalEdge> component_scoring_edges;
  vector<PbctopoLocalEdge> projection_edges;
  vector<PbctopoContinuityPair> projection_continuity_pairs;
  vector<size_t> projection_anchor_indices;
  vector<std::uint8_t> local_graph_atom_coverage;
  unordered_set<uint64_t> local_edge_set;
  unordered_set<uint64_t> steric_exclusion_set;
  unordered_set<uint64_t> projection_edge_set;
  vector<vector<size_t>> components;
  vector<size_t> atom_component_index;
  vector<PbctopoShift> shifts;
  vector<PbctopoSeed3D> volume_seeds;
  vector<Vec3> warm_points;
  vector<Vec3> coarse_warm_points;
  vector<Vec3> placed_points;
  vector<Vec3> best_safe_points;
  vector<Vec3> verify_points;
  vector<Vec3> component_points;
  vector<PbctopoCutCandidate> coarse_candidates;
  vector<PbctopoCoarseSeedOption> coarse_seed_options;
  vector<PbctopoCutCandidate> coarse_results;
  vector<size_t> projection_index;
  titan_pbctopo::PbctopoVirtualDiscreteSearchWorkspace virtual_search_workspace;
  PbctopoVirtualDiscreteSearchPlan virtual_search_plan;

  void reserve_from_context(const PbctopoTrajectoryContext &context) {
    base_atoms.reserve(context.total_atoms_hint);
    projection_atoms.reserve(context.heavy_atoms_hint);
    component_atoms.reserve(context.total_atoms_hint);
    local_edges.reserve(context.local_edges_hint);
    component_scoring_edges.reserve(context.local_edges_hint + 32);
    projection_edges.reserve(context.projection_edges_hint);
    projection_continuity_pairs.reserve(context.continuity_pair_hint);
    projection_anchor_indices.reserve(context.continuity_pair_hint);
    local_graph_atom_coverage.reserve(context.total_atoms_hint);
    local_edge_set.reserve(context.local_edges_hint * 2 + 1);
    steric_exclusion_set.reserve(context.local_edges_hint + 1);
    projection_edge_set.reserve(context.projection_edges_hint * 2 + 1);
    components.reserve(context.component_hint);
    atom_component_index.reserve(context.total_atoms_hint);
    shifts.reserve(27);
    volume_seeds.reserve(context.coarse_seed_hint);
    warm_points.reserve(context.total_atoms_hint);
    coarse_warm_points.reserve(context.heavy_atoms_hint);
    placed_points.reserve(context.heavy_atoms_hint);
    best_safe_points.reserve(context.total_atoms_hint);
    verify_points.reserve(context.total_atoms_hint);
    component_points.reserve(context.total_atoms_hint);
    coarse_candidates.reserve(context.coarse_candidate_hint);
    coarse_seed_options.reserve(context.coarse_seed_hint);
    coarse_results.reserve(context.coarse_seed_hint);
    projection_index.reserve(context.total_atoms_hint);
  }

  void reset_frame() {
    base_atoms.clear();
    projection_atoms.clear();
    component_atoms.clear();
    local_edges.clear();
    component_scoring_edges.clear();
    projection_edges.clear();
    projection_continuity_pairs.clear();
    projection_anchor_indices.clear();
    local_graph_atom_coverage.clear();
    local_edge_set.clear();
    steric_exclusion_set.clear();
    projection_edge_set.clear();
    components.clear();
    atom_component_index.clear();
    shifts.clear();
    volume_seeds.clear();
    warm_points.clear();
    coarse_warm_points.clear();
    placed_points.clear();
    best_safe_points.clear();
    verify_points.clear();
    component_points.clear();
    coarse_candidates.clear();
    coarse_seed_options.clear();
    coarse_results.clear();
    projection_index.clear();
    virtual_search_plan.x_states.clear();
    virtual_search_plan.y_states.clear();
    virtual_search_plan.z_states.clear();
    virtual_search_plan.states.clear();
  }

  size_t point_capacity_bytes_estimate() const noexcept {
    return (warm_points.capacity() + coarse_warm_points.capacity() +
            placed_points.capacity() + best_safe_points.capacity() +
            verify_points.capacity() + component_points.capacity()) *
           sizeof(Vec3);
  }
};

struct PbctopoFrameLatticeQueryAuditScope {
  PbctopoTimingStats *timing = nullptr;
  const size_t *fatal_failures = nullptr;

  PbctopoFrameLatticeQueryAuditScope(PbctopoTimingStats &stats,
                                     const size_t &frame_fatal_failures)
      : timing(&stats), fatal_failures(&frame_fatal_failures) {
    titan_pbctopo::pbctopo_reset_lattice_query_audit();
  }

  ~PbctopoFrameLatticeQueryAuditScope() {
    if (timing == nullptr)
      return;
    const auto audit = titan_pbctopo::pbctopo_lattice_query_audit();
    timing->pbctopo_lattice_query_failures += audit.failures;
    const size_t fatal =
        fatal_failures != nullptr ? min(*fatal_failures, audit.failures) : 0;
    timing->pbctopo_lattice_fatal_query_failures += fatal;
    timing->pbctopo_lattice_optional_query_failures += audit.failures - fatal;
    timing->pbctopo_lattice_query_ambiguities += audit.ambiguous;
    timing->pbctopo_lattice_query_budget_failures +=
        audit.enumeration_budget_failures;
  }
};

struct PbctopoUltraclashScratch {
  vector<int64_t> cell_x;
  vector<int64_t> cell_y;
  vector<int64_t> cell_z;
  vector<int> head;
  vector<uint32_t> head_epoch;
  vector<int> next;
  uint32_t current_epoch = 0;
};

struct NodeVectorHash {
  size_t operator()(const vector<size_t> &values) const noexcept {
    uint64_t hash = 1469598103934665603ull;
    for (size_t value : values) {
      hash ^= static_cast<uint64_t>(value) + 0x9e3779b97f4a7c15ull +
              (hash << 6) + (hash >> 2);
      hash *= 1099511628211ull;
    }
    return static_cast<size_t>(hash);
  }
};

struct PbctopoRawComponentMetadata {
  size_t raw_index = 0;
  int owner = 0;
  uint64_t chain_key = 0;
  size_t atom_count = 0;
  size_t min_local_atom_index = numeric_limits<size_t>::max();
  size_t max_local_atom_index = 0;
  size_t min_atom_index = numeric_limits<size_t>::max();
  size_t first_atom_index = numeric_limits<size_t>::max();
  size_t last_atom_index = numeric_limits<size_t>::max();
};

struct PbctopoGraphSeedSolveResult {
  const char *seed_name = "baseline";
  vector<Int3> offsets;
  PbctopoGraphSeedQuickScore quick_score;
  bool valid = false;
};

void reset_pbctopo_log_file();

void append_pbctopo_log_line(const string &message);

ofstream open_pbctopo_frame_report();

ofstream open_pbctopo_assignment_set_report();

ofstream open_pbctopo_rejected_assignment_report();

ofstream open_pbctopo_candidate_evidence_summary_report();

ofstream open_pbctopo_candidate_evidence_relation_report();

PbctopoComponentLayoutReportState &
pbctopo_component_layout_report_state();

ofstream open_pbctopo_component_layout_report();

ofstream open_pbctopo_hard_edge_report();

ofstream open_pbctopo_h9_component_representative_report();

ofstream open_pbctopo_h9_evidence_component_report();

ofstream open_pbctopo_h9_orbit_report();

ofstream open_pbctopo_temporal_path_report();

void merge_pbctopo_search_domain_provenance(
    PbctopoCutCandidate &target, const PbctopoCutCandidate &source);

string format_int3(const Int3 &value);

string format_vec3(const Vec3 &value);

string format_component_offsets(const vector<Int3> &offsets);

string format_graph_edges(const vector<PbctopoGraphEdge> &edges);

PbctopoRefineScoringCallbacks make_component_refine_callbacks();

PbctopoRuntimeCutoffs resolve_pbctopo_cutoffs(const arguments &Arguments);

PbctopoAnchorBackend resolve_pbctopo_anchor_backend(const arguments &Arguments);

const char *pbctopo_anchor_backend_name(PbctopoAnchorBackend backend);

PbctopoSolveMode resolve_pbctopo_solve_mode(const arguments &Arguments);

const char *pbctopo_solve_mode_name(PbctopoSolveMode mode);

PbctopoTranslationCache build_virtual_box_translation_cache(box Box);

vector<pair<int, int>>
build_contacting_owner_pairs(const vector<PbctopoLocalEdge> &local_edges,
                             const vector<PbctopoAtom> &atoms, box Box,
                             double contact_cutoff,
                             bool hard_required_only = false);

unordered_map<int, Vec3>
compute_owner_com_cartesian(const vector<Vec3> &placed_points,
                            const vector<PbctopoAtom> &atoms);

unordered_map<int, Vec3>
compute_owner_com_scaled(box Box,
                         const unordered_map<int, Vec3> &owner_com_cartesian);

bool candidate_has_split_contacting_owners(
    const vector<Vec3> &placed_points, const vector<PbctopoAtom> &atoms, box Box,
    const vector<pair<int, int>> &contacting_owner_pairs, size_t &split_owner_pairs,
    double &max_owner_raw_minus_minimg_delta,
    PbctopoOwnerPairSplitDetail *detail = nullptr);

bool reimage_contacting_owner_points_minimum_image(
    vector<Vec3> &placed_points, const vector<PbctopoAtom> &atoms, box Box,
    const vector<pair<int, int>> &contacting_owner_pairs);

void build_owner_spans(const vector<PbctopoAtom> &atoms, int owner,
                       vector<PbctopoHullSpan> &spans);

bool rescue_split_contacting_owner_candidate(
    const vector<Vec3> &seed_points, const vector<PbctopoAtom> &atoms, box Box,
    const vector<pair<int, int>> &contacting_owner_pairs,
    const vector<PbctopoShift> &shifts,
    const PbctopoOwnerPairSplitDetail &split_detail, vector<Vec3> &rescued_points,
    Int3 &applied_shift, double &best_raw_delta_norm);

bool continuity_metrics_ok(
    const vector<Vec3> &placed_points,
    const vector<PbctopoContinuityPair> &continuity_pairs, double hard_cutoff,
    double &path_sum, double &max_d2, double &min_hard_margin_A,
    size_t &margin_atom_a, size_t &margin_atom_b,
    const PbctopoCutCandidate *incumbent = nullptr);

vector<Int3> extract_component_offsets_from_points(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    box Box, const vector<Vec3> &placed_points);

bool align_component_offsets_to_points(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    const PbctopoLatticeMetric &lattice, const vector<Vec3> &placed_points,
    vector<Int3> &offsets);

const char *pbctopo_component_mode_name(PbctopoComponentMode mode);

PbctopoComponentMode select_pbctopo_component_mode(size_t raw_component_count,
                                                   size_t exact_component_limit);

PbctopoMergedComponentPlan build_chain_merged_component_plan(
    const vector<PbctopoAtom> &atoms, box Box,
    const vector<vector<size_t>> &raw_components);

vector<Int3> expand_merged_offsets_to_raw_components(
    const vector<Int3> &merged_offsets, box Box,
    const vector<vector<size_t>> &raw_components,
    const PbctopoMergedComponentPlan &plan);

bool derive_merged_warm_offsets(const vector<Int3> &raw_offsets,
                                const PbctopoMergedComponentPlan &plan,
                                vector<Int3> &merged_offsets);

vector<PbctopoAtom> apply_merged_component_internal_offsets(
    const vector<PbctopoAtom> &atoms,
    const vector<vector<size_t>> &raw_components,
    const PbctopoMergedComponentPlan &plan, box Box);

bool graph_seed_quick_score_better(const PbctopoGraphSeedQuickScore &lhs,
                                   const PbctopoGraphSeedQuickScore &rhs);

PbctopoGraphSeedQuickScore score_component_graph_seed_quick(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    const vector<Vec3> &placed_points, const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, double continuity_cutoff,
    double shift_norm2,
    PbctopoAnchorBackend anchor_backend = PbctopoAnchorBackend::Auto,
    const PbctopoAnchorGraphContext *anchor_context = nullptr);

bool kinematic_virtual_candidate_usable(
    const PbctopoCutCandidate &candidate);

bool has_consecutive_kinematic_seed_history(const PbctopoWarmState &warm_state,
                                            size_t frame_idx);

void reset_kinematic_seed_history(PbctopoWarmState &warm_state,
                                  PbctopoTimingStats &timing);

void update_kinematic_seed_history(PbctopoWarmState &warm_state,
                                   const PbctopoCutCandidate &candidate,
                                   size_t frame_idx,
                                   PbctopoTimingStats &timing);

vector<PbctopoGraphSeedSolveResult> build_large_component_graph_seed_candidates(
    const vector<PbctopoAtom> &atoms, box Box,
    const vector<vector<size_t>> &components, const vector<PbctopoShift> &shifts,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, const PbctopoRuntimeCutoffs &cutoffs,
    const vector<Int3> *warm_component_offsets, PbctopoTimingStats *timing,
    PbctopoAnchorBackend anchor_backend = PbctopoAnchorBackend::Auto,
    const PbctopoAnchorGraphContext *anchor_context = nullptr,
    const vector<titan_pbctopo::PbctopoRequiredContact> *required_contacts =
        nullptr);

bool update_cut_candidate_with_edge_distance(
    PbctopoCutCandidate &candidate, double d2, double ref_d2, double inv_ref_d2,
    double break_cutoff_d2, const PbctopoCutCandidate *edge_prune_threshold,
    PbctopoRefineScorePhase phase);

TITAN_TARGET_AVX2
bool accumulate_edge_scan_avx2(
    PbctopoCutCandidate &candidate, const PbctopoEdgeScanSoA &edge_scan_soa,
    const vector<Vec3> &placed_points,
    const PbctopoStage1EdgeContext *edge_context,
    const PbctopoCutCandidate *edge_prune_threshold,
    PbctopoRefineScorePhase phase);

bool accumulate_edge_scan(
    PbctopoCutCandidate &candidate, const vector<PbctopoLocalEdge> &local_edges,
    const vector<Vec3> &placed_points,
    const PbctopoStage1EdgeContext *edge_context,
    const PbctopoEdgeScanSoA *edge_scan_soa,
    const PbctopoCutCandidate *edge_prune_threshold,
    PbctopoRefineScorePhase phase);

bool edge_scan_cannot_beat_threshold(const PbctopoCutCandidate &partial,
                                     const PbctopoCutCandidate *threshold,
                                     PbctopoRefineScorePhase phase);

vector<PbctopoAtom> apply_spatial_lift(
    const vector<PbctopoAtom> &wrapped_atoms,
    const titan_pbctopo::PbctopoSpatialLiftResult &lift, box Box);

PbctopoComponentCandidate build_component_candidate_seed(
    const PbctopoCutCandidate &source, const vector<vector<size_t>> &components,
    const vector<Vec3> &placed_points);

PbctopoComponentCandidate score_component_assignment(
    const vector<PbctopoAtom> &atoms, const vector<Vec3> &placed_points,
    const vector<vector<size_t>> &components, double min_pair_d2,
    size_t clash_atom_a, size_t clash_atom_b,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, double continuity_cutoff,
    double shift_norm2, const PbctopoComponentCandidate *incumbent = nullptr,
    PbctopoTimingStats *timing = nullptr,
    PbctopoAnchorBackend anchor_backend = PbctopoAnchorBackend::Auto,
    const PbctopoAnchorGraphContext *anchor_context = nullptr);

PbctopoComponentSearchCallbacks make_component_search_callbacks();

bool solve_large_component_merge_mode(
    const vector<PbctopoAtom> &atoms, box Box,
    const vector<vector<size_t>> &raw_components,
    const PbctopoMergedComponentPlan &merge_plan,
    const vector<PbctopoShift> &shifts,
    const vector<PbctopoLocalEdge> &local_edges,
    const unordered_set<uint64_t> &edge_set, vector<Vec3> &placed_points,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices,
    const PbctopoRuntimeCutoffs &cutoffs, PbctopoCutCandidate &best_cut,
    double &shift_norm2, PbctopoTimingStats *timing,
    PbctopoAnchorBackend anchor_backend,
    const vector<Int3> *warm_component_offsets,
    PbctopoWarmState *warm_state,
    const PbctopoCutCandidate *warm_reference_candidate,
    bool allow_exact_search_on_merged,
    vector<Int3> *best_raw_offsets_out = nullptr,
    const PbctopoAnchorGraphContext *anchor_context = nullptr,
    const vector<titan_pbctopo::PbctopoRequiredContact> *required_contacts =
        nullptr);

bool find_virtual_box_shift(double wrapped_coord, double center, int &shift_idx);

void push_unique_xy_seed(vector<PbctopoSeed2D> &best, const PbctopoSeed2D &seed,
                         size_t max_keep);

void push_unique_axis_seed(vector<PbctopoCutOption> &best,
                           const PbctopoCutOption &seed, size_t max_keep);

void push_unique_xyz_seed(vector<PbctopoSeed3D> &best, const PbctopoSeed3D &seed,
                          size_t max_keep);

vector<double> build_projection_density_grid(const vector<PbctopoAtom> &atoms,
                                             size_t mesh_bins, double radius,
                                             char axis_a, char axis_b);

bool projection_density_is_local_maximum(const vector<double> &scores,
                                         size_t mesh_bins, size_t ix,
                                         size_t iy);

bool projection_seed2d_better(const PbctopoSeed2D &lhs,
                              const PbctopoSeed2D &rhs);

void collect_projection_plane_hotspots(const vector<double> &scores,
                                       size_t mesh_bins, size_t top_k,
                                       vector<PbctopoSeed2D> &hotspots);

double projection_grid_score(const vector<double> &scores, size_t mesh_bins,
                             size_t ia, size_t ib);

void append_dense_projection_volume_seeds(const vector<double> &xy_scores,
                                          const vector<double> &xz_scores,
                                          const vector<double> &yz_scores,
                                          size_t mesh_bins, size_t top_k,
                                          vector<PbctopoSeed3D> &seeds);

void collect_projection_volume_seeds(const vector<PbctopoAtom> &atoms,
                                     size_t mesh_bins, size_t top_k,
                                     double radius,
                                     vector<PbctopoSeed3D> &seeds);

[[maybe_unused]] vector<PbctopoSeed2D>
collect_xy_density_seeds(const vector<PbctopoAtom> &atoms, size_t mesh_bins,
                         size_t top_k, double radius);

[[maybe_unused]] vector<PbctopoCutOption>
collect_z_density_seeds(const vector<PbctopoAtom> &atoms, size_t mesh_bins,
                        size_t top_k, double radius);

void push_top_virtual_box_candidate(vector<PbctopoCutCandidate> &top_candidates,
                                    const PbctopoCutCandidate &candidate,
                                    size_t max_keep);

[[maybe_unused]] bool
build_virtual_box_positions(const vector<PbctopoAtom> &atoms, box Box,
                            double center_x, double center_y, double center_z,
                            vector<Vec3> &placed_points, PointStats &stats,
                            double &shift_norm2);

PbctopoCutCandidate evaluate_virtual_box_candidate(
    const vector<PbctopoAtom> &atoms, const vector<PbctopoLocalEdge> &local_edges,
    const unordered_set<uint64_t> &edge_set,
    const PbctopoTranslationCache &translation_cache, box Box, double cut_x,
    double cut_y, double cut_z, double gap_score, vector<Vec3> &placed_points,
    double clash_cutoff,
    const vector<PbctopoContinuityPair> *continuity_pairs = nullptr,
    const vector<size_t> *anchor_indices = nullptr,
    const vector<pair<int, int>> *contacting_owner_pairs = nullptr,
    double continuity_cutoff = 20.0,
    const PbctopoCutCandidate *incumbent = nullptr,
    bool skip_clash = false,
    double frame_time_ps = numeric_limits<double>::quiet_NaN(),
    const char *reject_context = nullptr,
    const PbctopoAnchorGraphContext *anchor_context = nullptr,
    PbctopoVirtualSeedSource seed_source = PbctopoVirtualSeedSource::None,
    const vector<size_t> *atom_component_index = nullptr);

[[maybe_unused]] vector<PbctopoCutOption>
collect_cut_options(const vector<PbctopoAtom> &atoms, char axis,
                    size_t max_options);

[[maybe_unused]] void
build_cut_positions(const vector<PbctopoAtom> &atoms, box Box, double cut_x,
                    double cut_y, double cut_z, vector<Vec3> &placed_points,
                    PointStats &stats, double &shift_norm2);

PbctopoCutCandidate score_cut_candidate(const vector<PbctopoLocalEdge> &local_edges,
                                        const vector<Vec3> &placed_points,
                                        double cut_x, double cut_y, double cut_z,
                                        double gap_score, const PointStats &stats,
                                        double shift_norm2,
                                        const PbctopoStage1EdgeContext *edge_context = nullptr,
                                        const PbctopoEdgeScanSoA *edge_scan_soa = nullptr,
                                        const PbctopoCutCandidate *edge_prune_threshold = nullptr,
                                        const PbctopoCutCandidate *pre_edge_seed = nullptr,
                                        PbctopoRefineScorePhase phase =
                                            PbctopoRefineScorePhase::Exact);

PbctopoCutCandidate score_cut_candidate_principal_only(
    const vector<PbctopoLocalEdge> &local_edges,
    const vector<Vec3> &placed_points, double cut_x, double cut_y, double cut_z,
    double gap_score, const PointStats &stats, double shift_norm2,
    PbctopoCutScoreBreakdown *breakdown = nullptr,
    const PbctopoStage1EdgeContext *edge_context = nullptr,
    const PbctopoEdgeScanSoA *edge_scan_soa = nullptr,
    const PbctopoCutCandidate *edge_prune_threshold = nullptr,
    const PbctopoCutCandidate *pre_edge_seed = nullptr);

void analyze_cut_candidate_ultraclash(const vector<Vec3> &placed_points,
                                       const unordered_set<uint64_t> &edge_set,
                                       PbctopoCutCandidate &candidate,
                                       double cutoff,
                                       const vector<PbctopoAtom> *atoms,
                                       const vector<size_t> *atom_component_index);

void apply_cut_candidate(const vector<PbctopoAtom> &atoms,
                         const vector<Vec3> &placed_points,
                         Multi_MOL &mol1_system, Multi_MOL &mol2_system,
                         size_t frame_index);

void apply_pbctopo_points_direct(const vector<PbctopoAtom> &atoms,
                                 const vector<Vec3> &placed_points,
                                 Multi_MOL &mol1_system,
                                 Multi_MOL &mol2_system,
                                 size_t frame_index);

[[maybe_unused]] void
append_fragments_for_system(const Multi_MOL &system, const MOL &mol, int owner,
                            vector<PbctopoFragment> &fragments,
                            vector<PbctopoAtom> &atoms);

vector<PbctopoShift> build_topology_shifts(box Box, int shell_radius = 1);

[[maybe_unused]] void
build_supercell_for_frame(const vector<PbctopoFragment> &fragments,
                          const vector<PbctopoAtom> &base_atoms,
                          const vector<PbctopoShift> &shifts,
                          vector<PbctopoFragmentCopyNode> &nodes,
                          vector<PbctopoAtomCopy> &atom_copies);

bool are_topology_neighbors(const PbctopoFragment &lhs,
                            const PbctopoFragment &rhs);

[[maybe_unused]] vector<vector<size_t>>
build_fragment_copy_graph(const vector<PbctopoFragment> &fragments,
                          const vector<PbctopoFragmentCopyNode> &nodes,
                          double geometry_cutoff, double topology_cutoff);

[[maybe_unused]] void
sort_fragment_copy_graph(vector<vector<size_t>> &adjacency,
                         const vector<PbctopoFragmentCopyNode> &nodes);

double min_bbox_distance2_to_selected(
    size_t node_id, const vector<size_t> &selected_nodes,
    const vector<PbctopoFragmentCopyNode> &nodes);

[[maybe_unused]] bool
build_candidate_block_from_root(size_t root_node,
                                const vector<PbctopoFragment> &fragments,
                                const vector<PbctopoFragmentCopyNode> &nodes,
                                const vector<vector<size_t>> &adjacency,
                                size_t total_atom_count,
                                vector<size_t> &component_nodes);

vector<size_t> collect_fragment_options(
    size_t fragment_id, const vector<size_t> &selected_nodes,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t max_options);

[[maybe_unused]] void enumerate_connected_block_candidates_recursive(
    const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t max_options_per_fragment,
    vector<int> &assignment, vector<size_t> &selected_nodes,
    unordered_set<vector<size_t>, NodeVectorHash> &seen_components,
    vector<vector<size_t>> &out_components);

[[maybe_unused]] vector<vector<size_t>> enumerate_connected_block_candidates(
    const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t max_options_per_fragment);

double compute_mst2(const vector<Vec3> &points);

size_t count_component_connectivity_edges(
    const vector<size_t> &component_nodes,
    const vector<vector<size_t>> &adjacency);

PbctopoCandidate
score_coarse_node_combination(const vector<size_t> &component_nodes,
                              const vector<PbctopoFragmentCopyNode> &nodes,
                              const vector<vector<size_t>> &adjacency);

bool better_pbctopo_coarse_candidate(const PbctopoCandidate &lhs,
                                     const PbctopoCandidate &rhs);

[[maybe_unused]] void enumerate_small_fragment_shift_combinations_recursive(
    size_t fragment_id, const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, vector<size_t> &selected_nodes,
    vector<PbctopoCandidate> &shortlist, size_t shortlist_limit);

[[maybe_unused]] vector<vector<size_t>>
enumerate_small_fragment_shift_combinations(
    const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t shortlist_limit);

[[maybe_unused]] PbctopoCandidate
score_component_candidate(const vector<size_t> &component_nodes,
                          const vector<PbctopoFragmentCopyNode> &nodes,
                          const vector<PbctopoAtomCopy> &atom_copies,
                          const vector<vector<size_t>> &adjacency);

[[maybe_unused]] bool
analyze_candidate_hard_clash(const vector<size_t> &component_nodes,
                             const vector<PbctopoFragmentCopyNode> &nodes,
                             const vector<PbctopoAtomCopy> &atom_copies,
                             double clash_cutoff,
                             double &min_interfragment_d2);

[[maybe_unused]] bool
better_pbctopo_candidate(const PbctopoCandidate &lhs,
                         const PbctopoCandidate &rhs);

[[maybe_unused]] void
apply_pbctopo_candidate(Multi_MOL &mol1_system, Multi_MOL &mol2_system,
                        size_t frame_index, const PbctopoCandidate &candidate,
                        const vector<PbctopoFragmentCopyNode> &nodes,
                        const vector<PbctopoAtomCopy> &atom_copies);

inline Vec3 point_from_atom_copy(const PbctopoAtomCopy &atom) {
  return {atom.x, atom.y, atom.z};
}

inline uint64_t atom_pair_key(size_t a, size_t b) {
  if (a > b)
    swap(a, b);
  return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

inline void scaled_to_cartesian(box Box, double sx, double sy, double sz,
                         double &x, double &y, double &z) {
  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  x = sx * v1[0] + sy * v2[0] + sz * v3[0];
  y = sx * v1[1] + sy * v2[1] + sz * v3[1];
  z = sx * v1[2] + sy * v2[2] + sz * v3[2];
}

inline PbctopoVirtualSeedCenter
virtual_seed_center_from_candidate(const PbctopoCutCandidate &candidate) {
  return {candidate.cut_x, candidate.cut_y, candidate.cut_z};
}

inline double clamp_virtual_box_center(double center) {
  if (center < 0.5)
    return 0.5;
  if (center > 2.5)
    return 2.5;
  return center;
}

inline double projection_axis_coord(const PbctopoAtom &atom, char axis) {
  if (axis == 'x')
    return atom.sx;
  if (axis == 'y')
    return atom.sy;
  return atom.sz;
}

inline double projection_grid_center(size_t mesh_bins, size_t idx) {
  if (mesh_bins <= 1)
    return 1.5;
  const double step = 2.0 / static_cast<double>(mesh_bins - 1);
  return 0.5 + step * static_cast<double>(idx);
}

inline double axis_gap(double min_a, double max_a, double min_b, double max_b) {
  if (max_a < min_b)
    return min_b - max_a;
  if (max_b < min_a)
    return min_a - max_b;
  return 0.0;
}

inline double bbox_distance2(const PbctopoFragmentCopyNode &lhs,
                      const PbctopoFragmentCopyNode &rhs) {
  const double dx = axis_gap(lhs.min_x, lhs.max_x, rhs.min_x, rhs.max_x);
  const double dy = axis_gap(lhs.min_y, lhs.max_y, rhs.min_y, rhs.max_y);
  const double dz = axis_gap(lhs.min_z, lhs.max_z, rhs.min_z, rhs.max_z);
  return dx * dx + dy * dy + dz * dz;
}

inline double node_shift_norm2(const PbctopoFragmentCopyNode &node) {
  return static_cast<double>(node.shift_ix * node.shift_ix +
                             node.shift_iy * node.shift_iy +
                             node.shift_iz * node.shift_iz);
}

inline Vec3 node_bbox_center(const PbctopoFragmentCopyNode &node) {
  return {(node.min_x + node.max_x) * 0.5, (node.min_y + node.max_y) * 0.5,
          (node.min_z + node.max_z) * 0.5};
}

} // namespace pbctopo_solve
