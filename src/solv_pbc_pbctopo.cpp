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

using namespace std;

namespace {

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

void reset_pbctopo_log_file() {
  titan_pbctopo::reset_log_file();
}

void append_pbctopo_log_line(const string &message) {
  titan_pbctopo::append_log_line(message);
}

ofstream open_pbctopo_frame_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_frame_report.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      write_pbctopo_frame_report_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_assignment_set_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_assignment_candidates.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      write_pbctopo_assignment_set_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_rejected_assignment_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;
  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;
  ofstream out("pbctopo_rejected_candidates.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      titan_pbctopo::write_pbctopo_rejected_assignment_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_candidate_evidence_summary_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_candidate_evidence_summary.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      write_pbctopo_candidate_evidence_summary_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_candidate_evidence_relation_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_candidate_evidence_relations.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      write_pbctopo_candidate_evidence_relation_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

struct PbctopoComponentLayoutReportState {
  bool initialized = false;
  std::filesystem::path initialized_cwd;
  vector<PbctopoComponentLayoutObservation> emitted_layouts;
};

PbctopoComponentLayoutReportState &
pbctopo_component_layout_report_state() {
  static PbctopoComponentLayoutReportState state;
  return state;
}

ofstream open_pbctopo_component_layout_report() {
  namespace fs = std::filesystem;
  PbctopoComponentLayoutReportState &state =
      pbctopo_component_layout_report_state();

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !state.initialized || ec || state.initialized_cwd.empty() ||
      state.initialized_cwd != cwd;

  ofstream out("pbctopo_component_layout.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report) {
      write_pbctopo_component_layout_header(out);
      state.emitted_layouts.clear();
    }
    state.initialized = true;
    state.initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_hard_edge_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_hard_edges.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    if (reset_report)
      write_pbctopo_hard_edge_header(out);
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

struct PbctopoH9ReportState {
  bool initialized = false;
  std::filesystem::path initialized_cwd;
};

template <typename HeaderWriter>
ofstream open_pbctopo_h9_report(const char *filename,
                                PbctopoH9ReportState &state,
                                HeaderWriter &&write_header) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !state.initialized || ec || state.initialized_cwd.empty() ||
      state.initialized_cwd != cwd;
  ofstream out(filename, ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    out.imbue(locale::classic());
    out << setprecision(numeric_limits<double>::max_digits10);
    if (reset_report)
      write_header(out);
    state.initialized = true;
    state.initialized_cwd = cwd;
  }
  return out;
}

ofstream open_pbctopo_h9_component_representative_report() {
  static PbctopoH9ReportState state;
  return open_pbctopo_h9_report(
      "pbctopo_h9_component_representatives.csv", state,
      [](ofstream &out) {
        out << "schema_version,frame,time_ps,component_index,component_id,"
               "atom_count,mass_policy,total_mass,representative_valid,"
               "box_b00,box_b01,box_b02,box_b10,box_b11,box_b12,"
               "box_b20,box_b21,box_b22,sx,sy,sz,x_A,y_A,z_A\n";
      });
}

ofstream open_pbctopo_h9_evidence_component_report() {
  static PbctopoH9ReportState state;
  return open_pbctopo_h9_report(
      "pbctopo_h9_evidence_components.csv", state, [](ofstream &out) {
        out << "schema_version,frame,time_ps,component_index,component_id,"
               "atom_count,evidence_component_label,is_canonical_root,"
               "evidence_connected_component_count,"
               "relative_lattice_degrees_of_freedom\n";
      });
}

ofstream open_pbctopo_h9_orbit_report() {
  static PbctopoH9ReportState state;
  return open_pbctopo_h9_report(
      "pbctopo_h9_orbit.csv", state, [](ofstream &out) {
        out << "schema_version,frame,time_ps,qx,qy,qz,root_label,"
               "shifted_label,within_radius,hard_certified,"
               "evidence_evaluated,evidence_noncontradicted,accepted,"
               "assignment_signature,evidence_state,"
               "evidence_connected_component_count,"
               "relative_lattice_degrees_of_freedom,evidence_edge_count,"
               "compatible_hypothesis_count,compatible_contact_count\n";
      });
}

ofstream open_pbctopo_temporal_path_report() {
  namespace fs = std::filesystem;
  static bool initialized = false;
  static fs::path initialized_cwd;

  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  const bool reset_report =
      !initialized || ec || initialized_cwd.empty() || initialized_cwd != cwd;

  ofstream out("pbctopo_temporal_path.csv",
               ios::out | (reset_report ? ios::trunc : ios::app));
  if (out.is_open()) {
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (reset_report) {
      out << "schema_version,producer_version,coordinate_unit,time_unit,"
             "endianness,floating_format,identity_hash_algorithm,"
             "binding_hash_algorithm,payload_checksum_algorithm,binding_mode,"
             "negative_zero_policy,"
             "nonfinite_policy,observation_scope,temporal_policy_status,frame,"
             "time_ps,selected_rank,cut_x,cut_y,cut_z,emission_cost,"
             "assignment_signature,changed_from_greedy,replay_applied\n";
    }
    initialized = true;
    initialized_cwd = cwd;
  }
  return out;
}

void merge_pbctopo_search_domain_provenance(
    PbctopoCutCandidate &target, const PbctopoCutCandidate &source) {
  const bool target_audit_known = target.search_domain_audit_known;
  const size_t target_radius = target.component_search_shell_radius;
  const bool target_complete = target.search_domain_evidence_complete;
  const bool target_boundary =
      target.selected_assignment_touches_shell_boundary;
  const bool target_expansion_exhausted = target.domain_expansion_exhausted;
  if (source.search_domain_audit_known) {
    const size_t merged_radius =
        max(target_audit_known ? target_radius : 0,
            source.component_search_shell_radius);
    const bool target_is_max_domain =
        target_audit_known && target_radius == merged_radius;
    const bool source_is_max_domain =
        source.component_search_shell_radius == merged_radius;
    target.component_search_shell_radius = merged_radius;
    target.search_domain_evidence_complete =
        (target_is_max_domain && target_complete) ||
        (source_is_max_domain && source.search_domain_evidence_complete);
    target.selected_assignment_touches_shell_boundary =
        (target_is_max_domain && target_boundary) ||
        (source_is_max_domain &&
         source.selected_assignment_touches_shell_boundary);
    target.domain_expansion_exhausted =
        (target_is_max_domain && target_expansion_exhausted) ||
        (source_is_max_domain && source.domain_expansion_exhausted);
    target.search_domain_audit_known = true;
  } else if (!target_audit_known) {
    target.search_domain_evidence_complete = false;
  }
  target.domain_expansion_attempted =
      target.domain_expansion_attempted || source.domain_expansion_attempted;
  target.bounded_domain_only =
      target.bounded_domain_only || source.bounded_domain_only;
}

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

using PbctopoGyrationShapeMetrics = GyrationShapeMetrics;

string format_int3(const Int3 &value) {
  return "(" + to_string(value.x) + "," + to_string(value.y) + "," +
         to_string(value.z) + ")";
}

string format_vec3(const Vec3 &value) {
  return "(" + to_string(value.x) + "," + to_string(value.y) + "," +
         to_string(value.z) + ")";
}

string format_component_offsets(const vector<Int3> &offsets) {
  string formatted = "[";
  for (size_t idx = 0; idx < offsets.size(); ++idx) {
    if (idx > 0)
      formatted += " ";
    formatted += to_string(idx) + ":" + format_int3(offsets[idx]);
  }
  formatted += "]";
  return formatted;
}

string format_graph_edges(const vector<PbctopoGraphEdge> &edges) {
  string formatted = "[";
  for (size_t idx = 0; idx < edges.size(); ++idx) {
    if (idx > 0)
      formatted += " ";
    const auto &edge = edges[idx];
    formatted += to_string(edge.a) + "-" + to_string(edge.b) + ":" +
                 format_int3({edge.dx, edge.dy, edge.dz}) + "/min=" +
                 to_string(edge.min_pair_d2) + "/bbox=" +
                 to_string(edge.bbox_d2) + "/conf=" +
                 to_string(edge.confidence);
  }
  formatted += "]";
  return formatted;
}

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

double compute_mst2(const vector<Vec3> &points);
vector<size_t> collect_fragment_options(
    size_t fragment_id, const vector<size_t> &selected_nodes,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t max_options);
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
bool rescue_split_contacting_owner_candidate(
    const vector<Vec3> &seed_points, const vector<PbctopoAtom> &atoms, box Box,
    const vector<pair<int, int>> &contacting_owner_pairs,
    const vector<PbctopoShift> &shifts,
    const PbctopoOwnerPairSplitDetail &split_detail, vector<Vec3> &rescued_points,
    Int3 &applied_shift, double &best_raw_delta_norm);
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
bool edge_scan_cannot_beat_threshold(const PbctopoCutCandidate &partial,
                                     const PbctopoCutCandidate *threshold,
                                     PbctopoRefineScorePhase phase);
bool continuity_metrics_ok(
    const vector<Vec3> &placed_points,
    const vector<PbctopoContinuityPair> &continuity_pairs, double hard_cutoff,
    double &path_sum, double &max_d2, double &min_hard_margin_A,
    size_t &margin_atom_a, size_t &margin_atom_b,
    const PbctopoCutCandidate *incumbent = nullptr);
void analyze_cut_candidate_ultraclash(const vector<Vec3> &placed_points,
                                       const unordered_set<uint64_t> &edge_set,
                                       PbctopoCutCandidate &candidate,
                                       double cutoff,
                                       const vector<PbctopoAtom> *atoms,
                                       const vector<size_t> *atom_component_index);
PbctopoAnchorBackend resolve_pbctopo_anchor_backend(const arguments &Arguments);
const char *pbctopo_anchor_backend_name(PbctopoAnchorBackend backend);
PbctopoSolveMode resolve_pbctopo_solve_mode(const arguments &Arguments);
const char *pbctopo_solve_mode_name(PbctopoSolveMode mode);
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
PbctopoRefineScoringCallbacks make_component_refine_callbacks() {
  PbctopoRefineScoringCallbacks callbacks;
  callbacks.score_cut_candidate = score_cut_candidate;
  callbacks.score_cut_candidate_principal_only =
      score_cut_candidate_principal_only;
  callbacks.continuity_metrics_ok = continuity_metrics_ok;
  callbacks.analyze_cut_candidate_ultraclash =
      analyze_cut_candidate_ultraclash;
  callbacks.better_cut_candidate = better_cut_candidate;
  callbacks.better_preclash_cut_candidate = better_preclash_cut_candidate;
  callbacks.better_stage1_cut_candidate = better_stage1_cut_candidate;
  callbacks.cut_candidate_cannot_beat_incumbent =
      cut_candidate_cannot_beat_incumbent;
  callbacks.edge_scan_cannot_beat_threshold = edge_scan_cannot_beat_threshold;
  callbacks.candidate_close_enough_to_skip_search =
      candidate_close_enough_to_skip_search;
  return callbacks;
}

[[maybe_unused]] Vec3 point_from_atom_copy(const PbctopoAtomCopy &atom) {
  return {atom.x, atom.y, atom.z};
}

PbctopoRuntimeCutoffs resolve_pbctopo_cutoffs(const arguments &Arguments) {
  return titan_pbctopo::resolve_runtime_cutoffs(Arguments);
}

PbctopoAnchorBackend resolve_pbctopo_anchor_backend(const arguments &Arguments) {
  return titan_pbctopo::resolve_anchor_backend(Arguments);
}

const char *pbctopo_anchor_backend_name(PbctopoAnchorBackend backend) {
  return titan_pbctopo::anchor_backend_name(backend);
}

PbctopoSolveMode resolve_pbctopo_solve_mode(const arguments &Arguments) {
  return titan_pbctopo::resolve_solve_mode(Arguments);
}

const char *pbctopo_solve_mode_name(PbctopoSolveMode mode) {
  return titan_pbctopo::solve_mode_name(mode);
}

uint64_t atom_pair_key(size_t a, size_t b) {
  if (a > b)
    swap(a, b);
  return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

void scaled_to_cartesian(box Box, double sx, double sy, double sz,
                         double &x, double &y, double &z) {
  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  x = sx * v1[0] + sy * v2[0] + sz * v3[0];
  y = sx * v1[1] + sy * v2[1] + sz * v3[1];
  z = sx * v1[2] + sy * v2[2] + sz * v3[2];
}

PbctopoTranslationCache build_virtual_box_translation_cache(box Box) {
  PbctopoTranslationCache cache;
  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  for (int ix = -2; ix <= 2; ++ix) {
    for (int iy = -2; iy <= 2; ++iy) {
      for (int iz = -2; iz <= 2; ++iz) {
        cache.by_shift[ix + 2][iy + 2][iz + 2] = {
            ix * v1[0] + iy * v2[0] + iz * v3[0],
            ix * v1[1] + iy * v2[1] + iz * v3[1],
            ix * v1[2] + iy * v2[2] + iz * v3[2]};
      }
    }
  }
  return cache;
}

vector<pair<int, int>>
build_contacting_owner_pairs(const vector<PbctopoLocalEdge> &local_edges,
                             const vector<PbctopoAtom> &atoms, box Box,
                             double contact_cutoff,
                             bool hard_required_only) {
  vector<pair<int, int>> owner_pairs;
  unordered_set<uint64_t> seen;
  seen.reserve(local_edges.size());
  auto register_owner_pair = [&](int owner_a, int owner_b) {
    if (owner_a == owner_b)
      return false;
    const int lo = min(owner_a, owner_b);
    const int hi = max(owner_a, owner_b);
    const uint64_t key =
        (static_cast<uint64_t>(static_cast<uint32_t>(lo)) << 32) |
        static_cast<uint32_t>(hi);
    if (!seen.insert(key).second)
      return false;
    owner_pairs.emplace_back(lo, hi);
    return true;
  };
  for (const auto &edge : local_edges) {
    if (edge.a >= atoms.size() || edge.b >= atoms.size())
      continue;
    if (hard_required_only &&
        edge.kind != PbctopoLocalEdgeKind::TopologyBond) {
      continue;
    }
    register_owner_pair(atoms[edge.a].owner, atoms[edge.b].owner);
  }

  if (hard_required_only)
    return owner_pairs;

  if (atoms.size() < 2 || contact_cutoff <= kDistanceEps)
    return owner_pairs;

  const PbctopoLatticeMetric lattice(Box);
  if (!lattice.valid())
    return owner_pairs;
  const BoxMatrix box_matrix(Box);
  const double cutoff2 = contact_cutoff * contact_cutoff;
  const double sx_bound =
      min(0.5, contact_cutoff * sqrt(box_matrix.inv[0][0] * box_matrix.inv[0][0] +
                                     box_matrix.inv[1][0] * box_matrix.inv[1][0] +
                                     box_matrix.inv[2][0] * box_matrix.inv[2][0]) +
                    1.0e-9);
  const double sy_bound =
      min(0.5, contact_cutoff * sqrt(box_matrix.inv[0][1] * box_matrix.inv[0][1] +
                                     box_matrix.inv[1][1] * box_matrix.inv[1][1] +
                                     box_matrix.inv[2][1] * box_matrix.inv[2][1]) +
                    1.0e-9);
  const double sz_bound =
      min(0.5, contact_cutoff * sqrt(box_matrix.inv[0][2] * box_matrix.inv[0][2] +
                                     box_matrix.inv[1][2] * box_matrix.inv[1][2] +
                                     box_matrix.inv[2][2] * box_matrix.inv[2][2]) +
                    1.0e-9);
  const int nx = max(1, static_cast<int>(floor(1.0 / max(sx_bound, 1.0e-6))));
  const int ny = max(1, static_cast<int>(floor(1.0 / max(sy_bound, 1.0e-6))));
  const int nz = max(1, static_cast<int>(floor(1.0 / max(sz_bound, 1.0e-6))));
  const double cell_sx = 1.0 / static_cast<double>(nx);
  const double cell_sy = 1.0 / static_cast<double>(ny);
  const double cell_sz = 1.0 / static_cast<double>(nz);
  auto wrap_index = [](int value, int mod) {
    int wrapped = value % mod;
    if (wrapped < 0)
      wrapped += mod;
    return wrapped;
  };

  unordered_map<ClashCellKey, vector<size_t>, ClashCellKeyHash> contact_grid;
  contact_grid.reserve(max<size_t>(64, atoms.size() / 2));
  for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    const auto &atom = atoms[atom_idx];
    if (!atom.is_heavy)
      continue;
    const int cx = min(nx - 1, static_cast<int>(floor(atom.sx / cell_sx)));
    const int cy = min(ny - 1, static_cast<int>(floor(atom.sy / cell_sy)));
    const int cz = min(nz - 1, static_cast<int>(floor(atom.sz / cell_sz)));
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          const ClashCellKey key{wrap_index(cx + dx, nx), wrap_index(cy + dy, ny),
                                 wrap_index(cz + dz, nz)};
          const auto it = contact_grid.find(key);
          if (it == contact_grid.end())
            continue;
          for (size_t other_idx : it->second) {
            const auto &other = atoms[other_idx];
            if (atom.owner == other.owner)
              continue;
            const auto mic = lattice.nearest_image_cartesian(
                other.x - atom.x, other.y - atom.y, other.z - atom.z);
            const double d2 = mic.d2;
            if (d2 > cutoff2)
              continue;
            register_owner_pair(atom.owner, other.owner);
          }
        }
      }
    }
    contact_grid[{cx, cy, cz}].push_back(atom_idx);
  }
  return owner_pairs;
}

unordered_map<int, Vec3>
compute_owner_com_cartesian(const vector<Vec3> &placed_points,
                            const vector<PbctopoAtom> &atoms) {
  unordered_map<int, Vec3> owner_sums;
  unordered_map<int, size_t> owner_counts;
  owner_sums.reserve(8);
  owner_counts.reserve(8);
  const size_t atom_count = min(placed_points.size(), atoms.size());
  for (size_t atom_idx = 0; atom_idx < atom_count; ++atom_idx) {
    const int owner = atoms[atom_idx].owner;
    Vec3 &sum = owner_sums[owner];
    sum.x += placed_points[atom_idx].x;
    sum.y += placed_points[atom_idx].y;
    sum.z += placed_points[atom_idx].z;
    ++owner_counts[owner];
  }
  for (auto &[owner, sum] : owner_sums) {
    const size_t count = owner_counts[owner];
    if (count == 0)
      continue;
    const double inv_count = 1.0 / static_cast<double>(count);
    sum.x *= inv_count;
    sum.y *= inv_count;
    sum.z *= inv_count;
  }
  return owner_sums;
}

unordered_map<int, Vec3>
compute_owner_com_scaled(box Box,
                         const unordered_map<int, Vec3> &owner_com_cartesian) {
  unordered_map<int, Vec3> owner_com_scaled;
  owner_com_scaled.reserve(owner_com_cartesian.size());
  const PbctopoLatticeMetric lattice(Box);
  for (const auto &[owner, point] : owner_com_cartesian) {
    const auto scaled = lattice.cartesian_to_scaled(point.x, point.y, point.z);
    owner_com_scaled[owner] = {scaled[0], scaled[1], scaled[2]};
  }
  return owner_com_scaled;
}

bool candidate_has_split_contacting_owners(
    const vector<Vec3> &placed_points, const vector<PbctopoAtom> &atoms, box Box,
    const vector<pair<int, int>> &contacting_owner_pairs, size_t &split_owner_pairs,
    double &max_owner_raw_minus_minimg_delta,
    PbctopoOwnerPairSplitDetail *detail) {
  split_owner_pairs = 0;
  max_owner_raw_minus_minimg_delta = 0.0;
  if (detail != nullptr)
    *detail = {};
  if (contacting_owner_pairs.empty() || placed_points.empty() || atoms.empty())
    return false;

  const auto owner_com_cartesian = compute_owner_com_cartesian(placed_points, atoms);
  const auto owner_com_scaled = compute_owner_com_scaled(Box, owner_com_cartesian);
  const PbctopoLatticeMetric lattice(Box);
  if (!lattice.valid())
    return false;
  const double box_scale = max(lattice.shortest_basis_length(), 1.0e-12);

  for (const auto &[owner_a, owner_b] : contacting_owner_pairs) {
    const auto it_a = owner_com_scaled.find(owner_a);
    const auto it_b = owner_com_scaled.find(owner_b);
    if (it_a == owner_com_scaled.end() || it_b == owner_com_scaled.end())
      continue;

    const Vec3 raw_delta_scaled{it_b->second.x - it_a->second.x,
                                it_b->second.y - it_a->second.y,
                                it_b->second.z - it_a->second.z};
    const auto mic = lattice.nearest_image_scaled(
        raw_delta_scaled.x, raw_delta_scaled.y, raw_delta_scaled.z);
    if (!mic.valid)
      continue;
    const Vec3 minimg_delta_scaled{mic.sx, mic.sy, mic.sz};
    const Vec3 split_delta{raw_delta_scaled.x - minimg_delta_scaled.x,
                           raw_delta_scaled.y - minimg_delta_scaled.y,
                           raw_delta_scaled.z - minimg_delta_scaled.z};
    const double axis_shift_max =
        max({fabs(split_delta.x), fabs(split_delta.y), fabs(split_delta.z)});
    const auto raw_cart = lattice.scaled_to_cartesian(
        raw_delta_scaled.x, raw_delta_scaled.y, raw_delta_scaled.z);
    const double raw_norm_A = sqrt(raw_cart[0] * raw_cart[0] +
                                   raw_cart[1] * raw_cart[1] +
                                   raw_cart[2] * raw_cart[2]);
    const double minimg_norm_A = sqrt(max(0.0, mic.d2));
    const double norm_gap = (raw_norm_A - minimg_norm_A) / box_scale;
    if (axis_shift_max <= 0.5 || norm_gap <= 0.20)
      continue;

    ++split_owner_pairs;
    if (norm_gap > max_owner_raw_minus_minimg_delta) {
      max_owner_raw_minus_minimg_delta = norm_gap;
      if (detail != nullptr) {
        detail->owner_a = owner_a;
        detail->owner_b = owner_b;
        detail->raw_delta_scaled = raw_delta_scaled;
        detail->minimg_delta_scaled = minimg_delta_scaled;
        detail->raw_norm = raw_norm_A;
        detail->minimg_norm = minimg_norm_A;
        detail->valid = true;
      }
    }
  }

  return split_owner_pairs > 0;
}

bool reimage_contacting_owner_points_minimum_image(
    vector<Vec3> &placed_points, const vector<PbctopoAtom> &atoms, box Box,
    const vector<pair<int, int>> &contacting_owner_pairs) {
  if (contacting_owner_pairs.empty() || placed_points.size() != atoms.size())
    return false;

  const auto owner_com_cartesian = compute_owner_com_cartesian(placed_points, atoms);
  const auto owner_com_scaled = compute_owner_com_scaled(Box, owner_com_cartesian);
  const PbctopoLatticeMetric lattice(Box);
  if (!lattice.valid() || owner_com_scaled.size() < 2)
    return false;

  unordered_map<int, vector<int>> adjacency;
  adjacency.reserve(owner_com_scaled.size());
  for (const auto &[owner_a, owner_b] : contacting_owner_pairs) {
    if (owner_com_scaled.find(owner_a) == owner_com_scaled.end() ||
        owner_com_scaled.find(owner_b) == owner_com_scaled.end()) {
      continue;
    }
    adjacency[owner_a].push_back(owner_b);
    adjacency[owner_b].push_back(owner_a);
  }
  if (adjacency.empty())
    return false;

  vector<int> owners;
  owners.reserve(owner_com_scaled.size());
  for (const auto &[owner, _] : owner_com_scaled)
    owners.push_back(owner);
  sort(owners.begin(), owners.end());

  unordered_map<int, Int3> owner_shifts;
  owner_shifts.reserve(owner_com_scaled.size());
  queue<int> pending;
  for (int root_owner : owners) {
    if (owner_shifts.find(root_owner) != owner_shifts.end())
      continue;
    owner_shifts[root_owner] = {0, 0, 0};
    pending.push(root_owner);
    while (!pending.empty()) {
      const int owner_u = pending.front();
      pending.pop();
      const auto adj_it = adjacency.find(owner_u);
      if (adj_it == adjacency.end())
        continue;
      const Vec3 &com_u = owner_com_scaled.at(owner_u);
      const Int3 shift_u = owner_shifts.at(owner_u);
      for (int owner_v : adj_it->second) {
        const Vec3 &com_v = owner_com_scaled.at(owner_v);
        const auto mic = lattice.nearest_image_scaled(
            com_v.x - com_u.x, com_v.y - com_u.y, com_v.z - com_u.z);
        if (!mic.valid)
          return false;
        Int3 proposed_shift{};
        if (!checked_int3_add(
                shift_u, Int3{mic.image_x, mic.image_y, mic.image_z},
                proposed_shift)) {
          return false;
        }
        const auto [shift_it, inserted] =
            owner_shifts.emplace(owner_v, proposed_shift);
        if (inserted)
          pending.push(owner_v);
      }
    }
  }

  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  bool changed = false;
  for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    const auto shift_it = owner_shifts.find(atoms[atom_idx].owner);
    if (shift_it == owner_shifts.end())
      continue;
    const Int3 &shift = shift_it->second;
    if (shift.x == 0 && shift.y == 0 && shift.z == 0)
      continue;
    placed_points[atom_idx].x += shift.x * v1[0] + shift.y * v2[0] + shift.z * v3[0];
    placed_points[atom_idx].y += shift.x * v1[1] + shift.y * v2[1] + shift.z * v3[1];
    placed_points[atom_idx].z += shift.x * v1[2] + shift.y * v2[2] + shift.z * v3[2];
    changed = true;
  }
  return changed;
}

void build_owner_spans(const vector<PbctopoAtom> &atoms, int owner,
                       vector<PbctopoHullSpan> &spans) {
  spans.clear();
  size_t span_begin = numeric_limits<size_t>::max();
  for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    if (atoms[atom_idx].owner == owner) {
      if (span_begin == numeric_limits<size_t>::max())
        span_begin = atom_idx;
      continue;
    }
    if (span_begin != numeric_limits<size_t>::max()) {
      spans.push_back({span_begin, atom_idx});
      span_begin = numeric_limits<size_t>::max();
    }
  }
  if (span_begin != numeric_limits<size_t>::max())
    spans.push_back({span_begin, atoms.size()});
}

bool rescue_split_contacting_owner_candidate(
    const vector<Vec3> &seed_points, const vector<PbctopoAtom> &atoms, box Box,
    const vector<pair<int, int>> &contacting_owner_pairs,
    const vector<PbctopoShift> &shifts,
    const PbctopoOwnerPairSplitDetail &split_detail, vector<Vec3> &rescued_points,
    Int3 &applied_shift, double &best_raw_delta_norm) {
  rescued_points.clear();
  applied_shift = {};
  best_raw_delta_norm = numeric_limits<double>::infinity();
  if (!split_detail.valid || seed_points.size() != atoms.size() ||
      contacting_owner_pairs.empty() || shifts.empty()) {
    return false;
  }

  const auto owner_com_cartesian = compute_owner_com_cartesian(seed_points, atoms);
  const auto owner_com_scaled = compute_owner_com_scaled(Box, owner_com_cartesian);
  size_t best_shift_idx = numeric_limits<size_t>::max();
  double best_score = numeric_limits<double>::infinity();
  int owner_to_move = -1;
  vector<int> move_candidates;
  move_candidates.reserve(2);
  move_candidates.push_back(split_detail.owner_b);
  if (split_detail.owner_a != split_detail.owner_b)
    move_candidates.push_back(split_detail.owner_a);

  for (int move_owner : move_candidates) {
    const int ref_owner =
        move_owner == split_detail.owner_b ? split_detail.owner_a
                                           : split_detail.owner_b;
    const auto ref_it = owner_com_scaled.find(ref_owner);
    const auto move_it = owner_com_scaled.find(move_owner);
    if (ref_it == owner_com_scaled.end() || move_it == owner_com_scaled.end())
      continue;

    const Vec3 &ref_com = ref_it->second;
    const Vec3 &move_com = move_it->second;
    for (size_t shift_idx = 0; shift_idx < shifts.size(); ++shift_idx) {
      const auto &shift = shifts[shift_idx];
      const Vec3 shifted_move_scaled{move_com.x + static_cast<double>(shift.ix),
                                     move_com.y + static_cast<double>(shift.iy),
                                     move_com.z + static_cast<double>(shift.iz)};
      const Vec3 raw_delta_scaled{shifted_move_scaled.x - ref_com.x,
                                  shifted_move_scaled.y - ref_com.y,
                                  shifted_move_scaled.z - ref_com.z};
      const double raw_norm = sqrt(norm2(raw_delta_scaled));
      double score = raw_norm;
      for (const auto &[owner_a, owner_b] : contacting_owner_pairs) {
        if (owner_a != move_owner && owner_b != move_owner)
          continue;
        const int other_owner = (owner_a == move_owner) ? owner_b : owner_a;
        const auto other_it = owner_com_scaled.find(other_owner);
        if (other_it == owner_com_scaled.end())
          continue;
        Vec3 pair_raw_delta_scaled{};
        if (owner_a == move_owner) {
          pair_raw_delta_scaled = {
              other_it->second.x - shifted_move_scaled.x,
              other_it->second.y - shifted_move_scaled.y,
              other_it->second.z - shifted_move_scaled.z};
        } else {
          pair_raw_delta_scaled = {
              shifted_move_scaled.x - other_it->second.x,
              shifted_move_scaled.y - other_it->second.y,
              shifted_move_scaled.z - other_it->second.z};
        }
        score += 0.25 * sqrt(norm2(pair_raw_delta_scaled));
      }
      if (score + 1.0e-9 < best_score) {
        best_score = score;
        best_raw_delta_norm = raw_norm;
        best_shift_idx = shift_idx;
        applied_shift = {shift.ix, shift.iy, shift.iz};
        owner_to_move = move_owner;
      }
    }
  }

  if (best_shift_idx == numeric_limits<size_t>::max() || owner_to_move < 0)
    return false;
  if (applied_shift.x == 0 && applied_shift.y == 0 && applied_shift.z == 0)
    return false;

  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  const Vec3 translation{
      applied_shift.x * v1[0] + applied_shift.y * v2[0] + applied_shift.z * v3[0],
      applied_shift.x * v1[1] + applied_shift.y * v2[1] + applied_shift.z * v3[1],
      applied_shift.x * v1[2] + applied_shift.y * v2[2] + applied_shift.z * v3[2]};

  rescued_points = seed_points;
  for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    if (atoms[atom_idx].owner != owner_to_move)
      continue;
    rescued_points[atom_idx].x += translation.x;
    rescued_points[atom_idx].y += translation.y;
    rescued_points[atom_idx].z += translation.z;
  }
  return true;
}

bool continuity_metrics_ok(
    const vector<Vec3> &placed_points,
    const vector<PbctopoContinuityPair> &continuity_pairs, double hard_cutoff,
    double &path_sum, double &max_d2, double &min_hard_margin_A,
    size_t &margin_atom_a, size_t &margin_atom_b,
    const PbctopoCutCandidate *incumbent) {
  path_sum = 0.0;
  max_d2 = 0.0;
  min_hard_margin_A = numeric_limits<double>::quiet_NaN();
  margin_atom_a = numeric_limits<size_t>::max();
  margin_atom_b = numeric_limits<size_t>::max();
  if (continuity_pairs.empty())
    return true;

  const double hard_cutoff2 = hard_cutoff * hard_cutoff;
  bool has_hard_margin = false;
  const bool has_incumbent =
      (incumbent != nullptr && incumbent->valid &&
       isfinite(incumbent->continuity_path) &&
       isfinite(incumbent->continuity_max_d2));

  auto consume_pair_metric = [&](double d2,
                                 const PbctopoContinuityPair &pair) -> bool {
    if (pbctopo_continuity_is_hard(pair.kind)) {
      if (d2 > max_d2)
        max_d2 = d2;
      const double margin_A = pbctopo_continuity_hard_margin_A(
          pair, d2, hard_cutoff2);
      if (!has_hard_margin || margin_A < min_hard_margin_A) {
        has_hard_margin = true;
        min_hard_margin_A = margin_A;
        margin_atom_a = pair.a;
        margin_atom_b = pair.b;
      }
      if (pbctopo_continuity_hard_violation(pair, d2, hard_cutoff2))
        return false;
    }
    path_sum += pair.weight *
                pbctopo_symmetric_reference_strain(d2, pair.ref_d2);
    if (has_incumbent &&
        path_sum > incumbent->continuity_path + 1.0e-6) {
      return false;
    }
    return true;
  };

#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  if (titan_simd_dispatch::runtime_has_avx2() &&
      continuity_pairs.size() >= 8) {
    alignas(32) double ax[4];
    alignas(32) double ay[4];
    alignas(32) double az[4];
    alignas(32) double bx[4];
    alignas(32) double by[4];
    alignas(32) double bz[4];
    alignas(32) double d2_lane[4];
    size_t pair_idx = 0;
    for (; pair_idx + 4 <= continuity_pairs.size(); pair_idx += 4) {
      for (size_t lane = 0; lane < 4; ++lane) {
        const auto &pair = continuity_pairs[pair_idx + lane];
        const Vec3 &point_a = placed_points[pair.a];
        const Vec3 &point_b = placed_points[pair.b];
        ax[lane] = point_a.x;
        ay[lane] = point_a.y;
        az[lane] = point_a.z;
        bx[lane] = point_b.x;
        by[lane] = point_b.y;
        bz[lane] = point_b.z;
      }

      const __m256d dx =
          _mm256_sub_pd(_mm256_load_pd(ax), _mm256_load_pd(bx));
      const __m256d dy =
          _mm256_sub_pd(_mm256_load_pd(ay), _mm256_load_pd(by));
      const __m256d dz =
          _mm256_sub_pd(_mm256_load_pd(az), _mm256_load_pd(bz));
      const __m256d d2 = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
          _mm256_mul_pd(dz, dz));
      _mm256_store_pd(d2_lane, d2);

      for (size_t lane = 0; lane < 4; ++lane) {
        if (!consume_pair_metric(d2_lane[lane],
                                 continuity_pairs[pair_idx + lane]))
          return false;
      }
    }
    for (; pair_idx < continuity_pairs.size(); ++pair_idx) {
      const auto &pair = continuity_pairs[pair_idx];
      const Vec3 delta = placed_points[pair.a] - placed_points[pair.b];
      const double d2 = norm2(delta);
      if (!consume_pair_metric(d2, pair))
        return false;
    }
    return true;
  }
#endif

  for (const auto &pair : continuity_pairs) {
    const Vec3 delta = placed_points[pair.a] - placed_points[pair.b];
    const double d2 = norm2(delta);
    if (!consume_pair_metric(d2, pair))
      return false;
  }
  return true;
}

vector<Int3> extract_component_offsets_from_points(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    box Box, const vector<Vec3> &placed_points) {
  vector<Int3> offsets;
  if (placed_points.size() != atoms.size())
    return offsets;
  offsets.resize(components.size());
  const PbctopoLatticeMetric lattice(Box);
  for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
    if (components[comp_idx].empty())
      continue;
    const size_t atom_idx = components[comp_idx].front();
    const double dx = placed_points[atom_idx].x - atoms[atom_idx].x;
    const double dy = placed_points[atom_idx].y - atoms[atom_idx].y;
    const double dz = placed_points[atom_idx].z - atoms[atom_idx].z;
    const auto scaled = lattice.cartesian_to_scaled(dx, dy, dz);
    Int3 offset{};
    if (!titan_pbctopo::pbctopo_checked_round_image(scaled[0], offset.x) ||
        !titan_pbctopo::pbctopo_checked_round_image(scaled[1], offset.y) ||
        !titan_pbctopo::pbctopo_checked_round_image(scaled[2], offset.z)) {
      return {};
    }
    offsets[comp_idx] = offset;
  }
  if (!rebase_component_offsets(offsets, Box))
    return {};
  return offsets;
}

// Offsets from extract_component_offsets_from_points are rebased to the
// closest-lattice-mean gauge, but placed points keep the gauge that placed
// them. Shift every offset by the one lattice vector that puts the first atom
// of the first non-empty component on its placed point. A common shift only
// changes the gauge: the relative assignment, and every other atom's check
// against it, stay as they were.
bool align_component_offsets_to_points(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    const PbctopoLatticeMetric &lattice, const vector<Vec3> &placed_points,
    vector<Int3> &offsets) {
  if (offsets.size() != components.size())
    return false;
  size_t reference_component = components.size();
  for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
    if (!components[comp_idx].empty()) {
      reference_component = comp_idx;
      break;
    }
  }
  if (reference_component == components.size())
    return false;
  const size_t reference_atom = components[reference_component].front();
  if (reference_atom >= atoms.size() || reference_atom >= placed_points.size())
    return false;
  const auto absolute_scaled = lattice.cartesian_to_scaled(
      placed_points[reference_atom].x - atoms[reference_atom].x,
      placed_points[reference_atom].y - atoms[reference_atom].y,
      placed_points[reference_atom].z - atoms[reference_atom].z);
  Int3 absolute_reference_offset{};
  if (!titan_pbctopo::pbctopo_checked_round_image(
          absolute_scaled[0], absolute_reference_offset.x) ||
      !titan_pbctopo::pbctopo_checked_round_image(
          absolute_scaled[1], absolute_reference_offset.y) ||
      !titan_pbctopo::pbctopo_checked_round_image(
          absolute_scaled[2], absolute_reference_offset.z)) {
    return false;
  }
  Int3 global_offset_correction{};
  if (!titan_pbctopo::checked_int3_subtract(absolute_reference_offset,
                                            offsets[reference_component],
                                            global_offset_correction)) {
    return false;
  }
  for (Int3 &offset : offsets) {
    Int3 corrected{};
    if (!titan_pbctopo::checked_int3_add(offset, global_offset_correction,
                                         corrected)) {
      return false;
    }
    offset = corrected;
  }
  return true;
}

const char *pbctopo_component_mode_name(PbctopoComponentMode mode) {
  switch (mode) {
  case PbctopoComponentMode::Disabled:
    return "disabled";
  case PbctopoComponentMode::Exact:
    return "small_graph";
  case PbctopoComponentMode::MergeThenGraph:
    return "merge_then_graph";
  case PbctopoComponentMode::Beam:
    return "beam";
  }
  return "unknown";
}

PbctopoComponentMode select_pbctopo_component_mode(size_t raw_component_count,
                                                   size_t exact_component_limit) {
  if (raw_component_count == 0)
    return PbctopoComponentMode::Disabled;
  const size_t bounded_exact_limit =
      min(exact_component_limit, kPbctopoExactComponentLimitMax);
  if (raw_component_count <= bounded_exact_limit)
    return PbctopoComponentMode::Exact;
  if (raw_component_count <= kPbctopoMergeModeMaxRawComponents)
    return PbctopoComponentMode::MergeThenGraph;
  return PbctopoComponentMode::Beam;
}

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

PbctopoMergedComponentPlan build_chain_merged_component_plan(
    const vector<PbctopoAtom> &atoms, box Box,
    const vector<vector<size_t>> &raw_components) {
  PbctopoMergedComponentPlan plan;
  plan.raw_to_merged.assign(raw_components.size(), numeric_limits<size_t>::max());
  plan.raw_internal_offsets.assign(raw_components.size(), {});
  if (raw_components.empty())
    return plan;

  vector<PbctopoRawComponentMetadata> metadata;
  metadata.reserve(raw_components.size());
  for (size_t raw_idx = 0; raw_idx < raw_components.size(); ++raw_idx) {
    const auto &component = raw_components[raw_idx];
    if (component.empty())
      continue;
    PbctopoRawComponentMetadata meta;
    meta.raw_index = raw_idx;
    meta.atom_count = component.size();
    const auto &first_atom = atoms[component.front()];
    meta.owner = first_atom.owner;
    meta.chain_key = first_atom.chain_key;
    for (size_t atom_idx : component) {
      if (atom_idx >= atoms.size())
        continue;
      meta.min_local_atom_index =
          min(meta.min_local_atom_index, atoms[atom_idx].local_atom_index);
      meta.max_local_atom_index =
          max(meta.max_local_atom_index, atoms[atom_idx].local_atom_index);
      meta.min_atom_index = min(meta.min_atom_index, atom_idx);
      if (meta.first_atom_index == numeric_limits<size_t>::max() ||
          atoms[atom_idx].local_atom_index <
              atoms[meta.first_atom_index].local_atom_index) {
        meta.first_atom_index = atom_idx;
      }
      if (meta.last_atom_index == numeric_limits<size_t>::max() ||
          atoms[atom_idx].local_atom_index >
              atoms[meta.last_atom_index].local_atom_index) {
        meta.last_atom_index = atom_idx;
      }
    }
    metadata.push_back(meta);
  }

  stable_sort(metadata.begin(), metadata.end(),
              [](const PbctopoRawComponentMetadata &lhs,
                 const PbctopoRawComponentMetadata &rhs) {
                if (lhs.owner != rhs.owner)
                  return lhs.owner < rhs.owner;
                if (lhs.chain_key != rhs.chain_key)
                  return lhs.chain_key < rhs.chain_key;
                if (lhs.min_local_atom_index != rhs.min_local_atom_index)
                  return lhs.min_local_atom_index < rhs.min_local_atom_index;
                return lhs.raw_index < rhs.raw_index;
              });

  vector<size_t> current_raw_group;
  vector<size_t> current_atoms;
  vector<size_t> first_atom_by_raw(raw_components.size(),
                                   numeric_limits<size_t>::max());
  vector<size_t> last_atom_by_raw(raw_components.size(),
                                  numeric_limits<size_t>::max());
  for (const auto &meta : metadata) {
    first_atom_by_raw[meta.raw_index] = meta.first_atom_index;
    last_atom_by_raw[meta.raw_index] = meta.last_atom_index;
  }
  const PbctopoLatticeMetric lattice(Box);
  current_raw_group.reserve(kPbctopoMergedNodeMaxRawComponents);
  current_atoms.reserve(kPbctopoMergedNodeMaxAtoms);
  int current_owner = 0;
  uint64_t current_chain_key = 0;
  size_t current_last_local_atom = 0;
  bool has_current_group = false;

  auto flush_group = [&]() {
    if (current_raw_group.empty())
      return;
    Int3 previous_offset{};
    for (size_t pos = 1; pos < current_raw_group.size(); ++pos) {
      const size_t previous_raw = current_raw_group[pos - 1];
      const size_t current_raw = current_raw_group[pos];
      const size_t previous_atom = last_atom_by_raw[previous_raw];
      const size_t current_atom = first_atom_by_raw[current_raw];
      if (previous_atom >= atoms.size() || current_atom >= atoms.size())
        continue;
      const auto mic = lattice.nearest_image_scaled(
          atoms[current_atom].sx -
              (atoms[previous_atom].sx + previous_offset.x),
          atoms[current_atom].sy -
              (atoms[previous_atom].sy + previous_offset.y),
          atoms[current_atom].sz -
              (atoms[previous_atom].sz + previous_offset.z));
      if (!mic.valid)
        continue;
      previous_offset = {mic.image_x, mic.image_y, mic.image_z};
      plan.raw_internal_offsets[current_raw] = previous_offset;
    }
    const size_t merged_idx = plan.merged_components.size();
    sort(current_atoms.begin(), current_atoms.end());
    plan.merged_components.push_back(current_atoms);
    plan.merged_to_raw.push_back(current_raw_group);
    for (size_t raw_idx : current_raw_group) {
      if (raw_idx < plan.raw_to_merged.size())
        plan.raw_to_merged[raw_idx] = merged_idx;
    }
    current_raw_group.clear();
    current_atoms.clear();
    has_current_group = false;
  };

  for (const auto &meta : metadata) {
    const bool contiguous_with_current =
        has_current_group && meta.owner == current_owner &&
        meta.chain_key == current_chain_key &&
        meta.min_local_atom_index == current_last_local_atom + 1;
    const bool fits_current =
        contiguous_with_current &&
        current_raw_group.size() < kPbctopoMergedNodeMaxRawComponents &&
        current_atoms.size() + meta.atom_count <= kPbctopoMergedNodeMaxAtoms;
    if (!fits_current)
      flush_group();
    if (!has_current_group) {
      current_owner = meta.owner;
      current_chain_key = meta.chain_key;
      current_last_local_atom = meta.max_local_atom_index;
      has_current_group = true;
    } else {
      current_last_local_atom = meta.max_local_atom_index;
    }
    current_raw_group.push_back(meta.raw_index);
    current_atoms.insert(current_atoms.end(), raw_components[meta.raw_index].begin(),
                         raw_components[meta.raw_index].end());
  }
  flush_group();

  return plan;
}

vector<Int3> expand_merged_offsets_to_raw_components(
    const vector<Int3> &merged_offsets, box Box,
    const vector<vector<size_t>> &raw_components,
    const PbctopoMergedComponentPlan &plan) {
  vector<Int3> raw_offsets(raw_components.size(), {0, 0, 0});
  for (size_t merged_idx = 0; merged_idx < plan.merged_to_raw.size(); ++merged_idx) {
    const Int3 merged_offset =
        merged_idx < merged_offsets.size() ? merged_offsets[merged_idx] : Int3{};
    for (size_t raw_idx : plan.merged_to_raw[merged_idx]) {
      if (raw_idx < raw_offsets.size()) {
        const Int3 internal = raw_idx < plan.raw_internal_offsets.size()
                                  ? plan.raw_internal_offsets[raw_idx]
                                  : Int3{};
        if (!checked_int3_add(merged_offset, internal,
                              raw_offsets[raw_idx])) {
          return {};
        }
      }
    }
  }
  if (!rebase_component_offsets(raw_offsets, Box))
    return {};
  return raw_offsets;
}

bool derive_merged_warm_offsets(const vector<Int3> &raw_offsets,
                                const PbctopoMergedComponentPlan &plan,
                                vector<Int3> &merged_offsets) {
  if (raw_offsets.empty() || plan.merged_to_raw.empty())
    return false;
  merged_offsets.clear();
  merged_offsets.reserve(plan.merged_to_raw.size());
  for (const auto &group : plan.merged_to_raw) {
    if (group.empty())
      return false;
    const size_t first_raw_idx = group.front();
    if (first_raw_idx >= raw_offsets.size())
      return false;
    const Int3 first_internal =
        first_raw_idx < plan.raw_internal_offsets.size()
            ? plan.raw_internal_offsets[first_raw_idx]
            : Int3{};
    Int3 candidate{};
    if (!checked_int3_subtract(raw_offsets[first_raw_idx], first_internal,
                               candidate)) {
      return false;
    }
    for (size_t raw_idx : group) {
      if (raw_idx >= raw_offsets.size())
        return false;
      const Int3 internal = raw_idx < plan.raw_internal_offsets.size()
                                ? plan.raw_internal_offsets[raw_idx]
                                : Int3{};
      Int3 residual{};
      if (!checked_int3_subtract(raw_offsets[raw_idx], internal, residual))
        return false;
      if (!same_delta(residual, candidate)) {
        return false;
      }
    }
    merged_offsets.push_back(candidate);
  }
  return true;
}

vector<PbctopoAtom> apply_merged_component_internal_offsets(
    const vector<PbctopoAtom> &atoms,
    const vector<vector<size_t>> &raw_components,
    const PbctopoMergedComponentPlan &plan, box Box) {
  vector<PbctopoAtom> aligned = atoms;
  const PbctopoLatticeMetric lattice(Box);
  const size_t raw_count =
      min(raw_components.size(), plan.raw_internal_offsets.size());
  for (size_t raw_idx = 0; raw_idx < raw_count; ++raw_idx) {
    const Int3 offset = plan.raw_internal_offsets[raw_idx];
    for (size_t atom_idx : raw_components[raw_idx]) {
      if (atom_idx >= aligned.size())
        continue;
      auto &atom = aligned[atom_idx];
      atom.sx += offset.x;
      atom.sy += offset.y;
      atom.sz += offset.z;
      const auto cart = lattice.scaled_to_cartesian(atom.sx, atom.sy, atom.sz);
      atom.x = cart[0];
      atom.y = cart[1];
      atom.z = cart[2];
    }
  }
  return aligned;
}

bool graph_seed_quick_score_better(const PbctopoGraphSeedQuickScore &lhs,
                                   const PbctopoGraphSeedQuickScore &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  int metric_cmp = compare_pbctopo_double_metric(
      lhs.continuity_path, rhs.continuity_path, 1.0e-6);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  int anchor_cmp = compare_anchor_graph_score(lhs.anchor_graph,
                                              rhs.anchor_graph);
  if (anchor_cmp == 0) {
    anchor_cmp = compare_pbctopo_double_metric(lhs.anchor_max_d2,
                                               rhs.anchor_max_d2, 1.0e-6);
  }
  if (anchor_cmp == 0) {
    anchor_cmp = compare_pbctopo_double_metric(lhs.anchor_mst2,
                                               rhs.anchor_mst2, 1.0e-6);
  }
  if (anchor_cmp != 0)
    return anchor_cmp < 0;
  const int principal_max_cmp =
      compare_principal_rg2(lhs.principal_rg2_max, rhs.principal_rg2_max);
  if (principal_max_cmp != 0)
    return principal_max_cmp < 0;
  const int principal_mid_cmp =
      compare_principal_rg2(lhs.principal_rg2_mid, rhs.principal_rg2_mid);
  if (principal_mid_cmp != 0)
    return principal_mid_cmp < 0;
  const int principal_min_cmp =
      compare_principal_rg2(lhs.principal_rg2_min, rhs.principal_rg2_min);
  if (principal_min_cmp != 0)
    return principal_min_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.shift_norm2, rhs.shift_norm2,
                                              1.0e-6);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  return false;
}

PbctopoGraphSeedQuickScore score_component_graph_seed_quick(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    const vector<Vec3> &placed_points, const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, double continuity_cutoff,
    double shift_norm2,
    PbctopoAnchorBackend anchor_backend = PbctopoAnchorBackend::Auto,
    const PbctopoAnchorGraphContext *anchor_context = nullptr) {
  PbctopoGraphSeedQuickScore score;
  if (placed_points.size() != atoms.size())
    return score;
  double continuity_min_margin_A = numeric_limits<double>::quiet_NaN();
  size_t continuity_margin_atom_a = numeric_limits<size_t>::max();
  size_t continuity_margin_atom_b = numeric_limits<size_t>::max();
  if (!continuity_metrics_ok(placed_points, continuity_pairs, continuity_cutoff,
                             score.continuity_path, score.continuity_max_d2,
                             continuity_min_margin_A,
                             continuity_margin_atom_a,
                             continuity_margin_atom_b, nullptr)) {
    return score;
  }
  const vector<size_t> half_anchor_indices =
      build_coarse_anchor_subset(atoms, anchor_indices);
  const vector<size_t> third_anchor_indices =
      build_third_anchor_subset(atoms, anchor_indices);
  const vector<size_t> *quick_anchor_indices = &anchor_indices;
  if (anchor_indices.size() > 256 && third_anchor_indices.size() >= 2) {
    quick_anchor_indices = &third_anchor_indices;
  } else if (half_anchor_indices.size() >= 2) {
    quick_anchor_indices = &half_anchor_indices;
  }
  if (!anchor_graph_metrics_enhanced(
          placed_points, *quick_anchor_indices, continuity_pairs,
          score.anchor_mst2, score.anchor_max_d2, score.anchor_graph, nullptr,
          PbctopoRefineScorePhase::Exact, anchor_backend, nullptr,
          anchor_context)) {
    return score;
  }
  const PointStats stats = build_point_stats(placed_points);
  if (stats.count == 0)
    return score;
  const array<double, 3> principal_rg2 = compute_principal_rg2(stats);
  score.principal_rg2_max = principal_rg2[0];
  score.principal_rg2_mid = principal_rg2[1];
  score.principal_rg2_min = principal_rg2[2];
  score.shift_norm2 = shift_norm2;
  score.valid = true;
  return score;
}

PbctopoVirtualSeedCenter
virtual_seed_center_from_candidate(const PbctopoCutCandidate &candidate) {
  return {candidate.cut_x, candidate.cut_y, candidate.cut_z};
}

bool kinematic_virtual_candidate_usable(
    const PbctopoCutCandidate &candidate) {
  return candidate.valid && !candidate.ultra_clash &&
         isfinite(candidate.cut_x) && isfinite(candidate.cut_y) &&
         isfinite(candidate.cut_z);
}

bool has_consecutive_kinematic_seed_history(const PbctopoWarmState &warm_state,
                                            size_t frame_idx) {
  return warm_state.has_kinematic_previous_virtual_candidate &&
         warm_state.has_kinematic_latest_virtual_candidate &&
         warm_state.kinematic_previous_virtual_frame + 1 ==
             warm_state.kinematic_latest_virtual_frame &&
         warm_state.kinematic_latest_virtual_frame + 1 == frame_idx &&
         kinematic_virtual_candidate_usable(
             warm_state.kinematic_previous_virtual_candidate) &&
         kinematic_virtual_candidate_usable(
             warm_state.kinematic_latest_virtual_candidate);
}

void reset_kinematic_seed_history(PbctopoWarmState &warm_state,
                                  PbctopoTimingStats &timing) {
  if (warm_state.has_kinematic_previous_virtual_candidate ||
      warm_state.has_kinematic_latest_virtual_candidate) {
    ++timing.pbctopo_kinematic_seed_history_resets;
  }
  warm_state.kinematic_previous_virtual_candidate = PbctopoCutCandidate{};
  warm_state.kinematic_latest_virtual_candidate = PbctopoCutCandidate{};
  warm_state.kinematic_previous_virtual_frame =
      numeric_limits<size_t>::max();
  warm_state.kinematic_latest_virtual_frame =
      numeric_limits<size_t>::max();
  warm_state.has_kinematic_previous_virtual_candidate = false;
  warm_state.has_kinematic_latest_virtual_candidate = false;
}

void update_kinematic_seed_history(PbctopoWarmState &warm_state,
                                   const PbctopoCutCandidate &candidate,
                                   size_t frame_idx,
                                   PbctopoTimingStats &timing) {
  if (!kinematic_virtual_candidate_usable(candidate)) {
    reset_kinematic_seed_history(warm_state, timing);
    return;
  }

  if (warm_state.has_kinematic_latest_virtual_candidate &&
      warm_state.kinematic_latest_virtual_frame + 1 == frame_idx) {
    warm_state.kinematic_previous_virtual_candidate =
        warm_state.kinematic_latest_virtual_candidate;
    warm_state.kinematic_previous_virtual_frame =
        warm_state.kinematic_latest_virtual_frame;
    warm_state.has_kinematic_previous_virtual_candidate = true;
  } else if (warm_state.has_kinematic_latest_virtual_candidate ||
             warm_state.has_kinematic_previous_virtual_candidate) {
    reset_kinematic_seed_history(warm_state, timing);
  }

  warm_state.kinematic_latest_virtual_candidate = candidate;
  warm_state.kinematic_latest_virtual_frame = frame_idx;
  warm_state.has_kinematic_latest_virtual_candidate = true;
}

struct PbctopoGraphSeedSolveResult {
  const char *seed_name = "baseline";
  vector<Int3> offsets;
  PbctopoGraphSeedQuickScore quick_score;
  bool valid = false;
};

vector<PbctopoGraphSeedSolveResult> build_large_component_graph_seed_candidates(
    const vector<PbctopoAtom> &atoms, box Box,
    const vector<vector<size_t>> &components, const vector<PbctopoShift> &shifts,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, const PbctopoRuntimeCutoffs &cutoffs,
    const vector<Int3> *warm_component_offsets, PbctopoTimingStats *timing,
    PbctopoAnchorBackend anchor_backend = PbctopoAnchorBackend::Auto,
    const PbctopoAnchorGraphContext *anchor_context = nullptr,
    const vector<titan_pbctopo::PbctopoRequiredContact> *required_contacts =
        nullptr) {
  const vector<PbctopoGraphEdge> base_edges = build_component_graph_edge_candidates(
      atoms, Box, components, shifts, cutoffs, timing, warm_component_offsets,
      nullptr, required_contacts);
  vector<PbctopoGraphSeedSolveResult> results;
  results.reserve(3);
  unordered_set<string> seen_offset_signatures;

  auto try_seed = [&](const char *seed_name,
                      const function<void(vector<PbctopoGraphEdge> &)> &sort_fn) {
    vector<PbctopoGraphEdge> ranked_edges = base_edges;
    sort_fn(ranked_edges);
    vector<Int3> offsets;
    if (!solve_component_offsets_from_ranked_edges(components.size(), Box, components,
                                                   ranked_edges, offsets)) {
      return;
    }
    const string signature = format_component_offsets(offsets);
    if (!seen_offset_signatures.insert(signature).second)
      return;
    vector<Vec3> seed_points;
    double seed_shift_norm2 = 0.0;
    build_component_points_from_offsets(atoms, components, Box, offsets, seed_points,
                                        seed_shift_norm2);
    PbctopoGraphSeedSolveResult result;
    result.seed_name = seed_name;
    result.offsets = offsets;
    result.quick_score = score_component_graph_seed_quick(
        atoms, components, seed_points, continuity_pairs, anchor_indices,
        cutoffs.continuity_cutoff, seed_shift_norm2, anchor_backend,
        anchor_context);
    result.valid = result.quick_score.valid;
    results.push_back(std::move(result));
  };

  try_seed("baseline", [&](vector<PbctopoGraphEdge> &ranked_edges) {
    sort(ranked_edges.begin(), ranked_edges.end(), better_component_graph_edge);
  });

  try_seed("nonzero", [&](vector<PbctopoGraphEdge> &ranked_edges) {
    stable_sort(ranked_edges.begin(), ranked_edges.end(),
                [](const PbctopoGraphEdge &lhs, const PbctopoGraphEdge &rhs) {
                  if (!lhs.valid)
                    return false;
                  if (!rhs.valid)
                    return true;
                  if (lhs.ultra_clash != rhs.ultra_clash)
                    return rhs.ultra_clash;
                  const bool lhs_zero = graph_edge_has_zero_shift(lhs);
                  const bool rhs_zero = graph_edge_has_zero_shift(rhs);
                  if (lhs_zero != rhs_zero)
                    return !lhs_zero;
                  int metric_cmp = compare_pbctopo_double_metric(
                      lhs.contact_fraction, rhs.contact_fraction, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.confidence, rhs.confidence, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.soft_penalty_mean, rhs.soft_penalty_mean, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.min_pair_d2, rhs.min_pair_d2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.bbox_d2, rhs.bbox_d2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.shift_norm2, rhs.shift_norm2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  if (lhs.a != rhs.a)
                    return lhs.a < rhs.a;
                  return lhs.b < rhs.b;
                });
  });

  try_seed("warm", [&](vector<PbctopoGraphEdge> &ranked_edges) {
    if (warm_component_offsets == nullptr ||
        warm_component_offsets->size() != components.size()) {
      stable_sort(ranked_edges.begin(), ranked_edges.end(),
                  [](const PbctopoGraphEdge &lhs, const PbctopoGraphEdge &rhs) {
                    if (!lhs.valid)
                      return false;
                    if (!rhs.valid)
                      return true;
                    if (lhs.ultra_clash != rhs.ultra_clash)
                      return rhs.ultra_clash;
                    const bool lhs_zero = graph_edge_has_zero_shift(lhs);
                    const bool rhs_zero = graph_edge_has_zero_shift(rhs);
                    if (lhs_zero != rhs_zero)
                      return !lhs_zero;
                    return better_component_graph_edge(lhs, rhs);
                  });
      return;
    }
    stable_sort(ranked_edges.begin(), ranked_edges.end(),
                [&](const PbctopoGraphEdge &lhs, const PbctopoGraphEdge &rhs) {
                  if (!lhs.valid)
                    return false;
                  if (!rhs.valid)
                    return true;
                  if (lhs.ultra_clash != rhs.ultra_clash)
                    return rhs.ultra_clash;
                  Int3 lhs_expected{};
                  Int3 rhs_expected{};
                  const bool lhs_expected_valid = checked_int3_subtract(
                      (*warm_component_offsets)[lhs.b],
                      (*warm_component_offsets)[lhs.a], lhs_expected);
                  const bool rhs_expected_valid = checked_int3_subtract(
                      (*warm_component_offsets)[rhs.b],
                      (*warm_component_offsets)[rhs.a], rhs_expected);
                  const bool lhs_warm =
                      lhs_expected_valid &&
                      graph_edge_matches_delta(lhs, lhs_expected);
                  const bool rhs_warm =
                      rhs_expected_valid &&
                      graph_edge_matches_delta(rhs, rhs_expected);
                  if (lhs_warm != rhs_warm)
                    return lhs_warm;
                  const bool lhs_zero = graph_edge_has_zero_shift(lhs);
                  const bool rhs_zero = graph_edge_has_zero_shift(rhs);
                  if (lhs_zero != rhs_zero)
                    return !lhs_zero;
                  int metric_cmp = compare_pbctopo_double_metric(
                      lhs.contact_fraction, rhs.contact_fraction, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.confidence, rhs.confidence, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.soft_penalty_mean, rhs.soft_penalty_mean, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.min_pair_d2, rhs.min_pair_d2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.bbox_d2, rhs.bbox_d2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.shift_norm2, rhs.shift_norm2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  if (lhs.a != rhs.a)
                    return lhs.a < rhs.a;
                  return lhs.b < rhs.b;
                });
  });

  return results;
}

bool update_cut_candidate_with_edge_distance(
    PbctopoCutCandidate &candidate, double d2, double ref_d2, double inv_ref_d2,
    double break_cutoff_d2, const PbctopoCutCandidate *edge_prune_threshold,
    PbctopoRefineScorePhase phase) {
  if (d2 > break_cutoff_d2)
    ++candidate.broken_edges;
  (void)inv_ref_d2;
  candidate.edge_stretch +=
      pbctopo_symmetric_reference_strain(d2, ref_d2);
  return edge_prune_threshold == nullptr ||
         !edge_scan_cannot_beat_threshold(candidate, edge_prune_threshold,
                                          phase);
}

#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
TITAN_TARGET_AVX2
bool accumulate_edge_scan_avx2(
    PbctopoCutCandidate &candidate, const PbctopoEdgeScanSoA &edge_scan_soa,
    const vector<Vec3> &placed_points,
    const PbctopoStage1EdgeContext *edge_context,
    const PbctopoCutCandidate *edge_prune_threshold,
    PbctopoRefineScorePhase phase) {
  if (placed_points.empty())
    return true;

  const auto process_block = [&](const size_t *edge_indices,
                                 size_t block_count) -> bool {
    alignas(32) long long atom_a_lane[4]{};
    alignas(32) long long atom_b_lane[4]{};
    alignas(32) double ref_lane[4]{};
    alignas(32) double inv_ref_lane[4]{};
    alignas(32) double cutoff_lane[4]{};
    for (size_t lane = 0; lane < block_count; ++lane) {
      const size_t edge_idx = edge_indices[lane];
      atom_a_lane[lane] = static_cast<long long>(edge_scan_soa.a[edge_idx]);
      atom_b_lane[lane] = static_cast<long long>(edge_scan_soa.b[edge_idx]);
      ref_lane[lane] = edge_scan_soa.ref_d2[edge_idx];
      inv_ref_lane[lane] = edge_scan_soa.inv_ref_d2[edge_idx];
      cutoff_lane[lane] = edge_scan_soa.break_cutoff_d2[edge_idx];
    }
    const auto &a0 = placed_points[static_cast<size_t>(atom_a_lane[0])];
    const auto &a1 = placed_points[static_cast<size_t>(atom_a_lane[1])];
    const auto &a2 = placed_points[static_cast<size_t>(atom_a_lane[2])];
    const auto &a3 = placed_points[static_cast<size_t>(atom_a_lane[3])];
    const auto &b0 = placed_points[static_cast<size_t>(atom_b_lane[0])];
    const auto &b1 = placed_points[static_cast<size_t>(atom_b_lane[1])];
    const auto &b2 = placed_points[static_cast<size_t>(atom_b_lane[2])];
    const auto &b3 = placed_points[static_cast<size_t>(atom_b_lane[3])];
    const __m256d ax = _mm256_set_pd(a3.x, a2.x, a1.x, a0.x);
    const __m256d ay = _mm256_set_pd(a3.y, a2.y, a1.y, a0.y);
    const __m256d az = _mm256_set_pd(a3.z, a2.z, a1.z, a0.z);
    const __m256d bx = _mm256_set_pd(b3.x, b2.x, b1.x, b0.x);
    const __m256d by = _mm256_set_pd(b3.y, b2.y, b1.y, b0.y);
    const __m256d bz = _mm256_set_pd(b3.z, b2.z, b1.z, b0.z);
    const __m256d dx = _mm256_sub_pd(ax, bx);
    const __m256d dy = _mm256_sub_pd(ay, by);
    const __m256d dz = _mm256_sub_pd(az, bz);
    const __m256d d2 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
        _mm256_mul_pd(dz, dz));
    alignas(32) double d2_lane[4];
    _mm256_store_pd(d2_lane, d2);

    for (size_t lane = 0; lane < block_count; ++lane) {
      if (!update_cut_candidate_with_edge_distance(
              candidate, d2_lane[lane], ref_lane[lane], inv_ref_lane[lane],
              cutoff_lane[lane], edge_prune_threshold, phase)) {
        return false;
      }
    }
    return true;
  };

  if (edge_context != nullptr) {
    size_t block[4]{};
    size_t block_count = 0;
    for (size_t edge_idx : edge_context->variable_edge_indices) {
      if (edge_idx >= edge_scan_soa.a.size())
        continue;
      block[block_count++] = edge_idx;
      if (block_count == 4) {
        if (!process_block(block, block_count))
          return false;
        block_count = 0;
      }
    }
    if (block_count != 0 && !process_block(block, block_count))
      return false;
    return true;
  }

  size_t edge_idx = 0;
  size_t block[4]{};
  while (edge_idx + 4 <= edge_scan_soa.a.size()) {
    block[0] = edge_idx + 0;
    block[1] = edge_idx + 1;
    block[2] = edge_idx + 2;
    block[3] = edge_idx + 3;
    if (!process_block(block, 4))
      return false;
    edge_idx += 4;
  }
  for (; edge_idx < edge_scan_soa.a.size(); ++edge_idx) {
    const size_t atom_a = edge_scan_soa.a[edge_idx];
    const size_t atom_b = edge_scan_soa.b[edge_idx];
    const Vec3 delta = placed_points[atom_a] - placed_points[atom_b];
    const double d2 = norm2(delta);
    if (!update_cut_candidate_with_edge_distance(
            candidate, d2, edge_scan_soa.ref_d2[edge_idx],
            edge_scan_soa.inv_ref_d2[edge_idx],
            edge_scan_soa.break_cutoff_d2[edge_idx], edge_prune_threshold,
            phase)) {
      return false;
    }
  }
  return true;
}
#endif

bool accumulate_edge_scan(
    PbctopoCutCandidate &candidate, const vector<PbctopoLocalEdge> &local_edges,
    const vector<Vec3> &placed_points,
    const PbctopoStage1EdgeContext *edge_context,
    const PbctopoEdgeScanSoA *edge_scan_soa,
    const PbctopoCutCandidate *edge_prune_threshold,
    PbctopoRefineScorePhase phase) {
  const bool use_soa =
      edge_scan_soa != nullptr && edge_scan_soa->a.size() == local_edges.size() &&
      edge_scan_soa->b.size() == local_edges.size() &&
      edge_scan_soa->ref_d2.size() == local_edges.size() &&
      edge_scan_soa->inv_ref_d2.size() == local_edges.size() &&
      edge_scan_soa->break_cutoff_d2.size() == local_edges.size();

#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  const size_t selected_edge_count =
      edge_context != nullptr ? edge_context->variable_edge_indices.size()
                              : local_edges.size();
  const bool simd_enabled =
      use_soa && titan_simd_dispatch::runtime_has_avx2() &&
      selected_edge_count >= 16;
  if (simd_enabled) {
    return accumulate_edge_scan_avx2(candidate, *edge_scan_soa, placed_points,
                                     edge_context, edge_prune_threshold,
                                     phase);
  }
#endif

  const auto accumulate_edge = [&](size_t edge_idx) {
    size_t atom_a = 0;
    size_t atom_b = 0;
    double ref_d2 = 0.0;
    double inv_ref_d2 = 0.0;
    double break_cutoff_d2 = 0.0;
    if (use_soa) {
      atom_a = edge_scan_soa->a[edge_idx];
      atom_b = edge_scan_soa->b[edge_idx];
      ref_d2 = edge_scan_soa->ref_d2[edge_idx];
      inv_ref_d2 = edge_scan_soa->inv_ref_d2[edge_idx];
      break_cutoff_d2 = edge_scan_soa->break_cutoff_d2[edge_idx];
    } else {
      const auto &edge = local_edges[edge_idx];
      atom_a = edge.a;
      atom_b = edge.b;
      ref_d2 = edge.ref_d2;
      inv_ref_d2 = edge.inv_ref_d2;
      break_cutoff_d2 = edge.break_cutoff_d2;
    }
    const Vec3 delta = placed_points[atom_a] - placed_points[atom_b];
    const double d2 = norm2(delta);
    return update_cut_candidate_with_edge_distance(
        candidate, d2, ref_d2, inv_ref_d2, break_cutoff_d2,
        edge_prune_threshold, phase);
  };

  if (edge_context != nullptr) {
    for (size_t edge_idx : edge_context->variable_edge_indices) {
      if (edge_idx >= local_edges.size())
        continue;
      if (!accumulate_edge(edge_idx))
        return false;
    }
    return true;
  }

  for (size_t edge_idx = 0; edge_idx < local_edges.size(); ++edge_idx) {
    if (!accumulate_edge(edge_idx))
      return false;
  }
  return true;
}

bool edge_scan_cannot_beat_threshold(const PbctopoCutCandidate &partial,
                                     const PbctopoCutCandidate *threshold,
                                     PbctopoRefineScorePhase phase) {
  if (threshold == nullptr || !threshold->valid)
    return false;
  const int prefix_cmp =
      compare_candidate_prefix_before_edge_stretch(partial, *threshold, phase);
  if (prefix_cmp > 0)
    return true;
  if (prefix_cmp < 0)
    return false;
  return partial.edge_stretch > threshold->edge_stretch + 1.0e-6;
}

vector<PbctopoAtom> apply_spatial_lift(
    const vector<PbctopoAtom> &wrapped_atoms,
    const titan_pbctopo::PbctopoSpatialLiftResult &lift, box Box) {
  vector<PbctopoAtom> unwrapped = wrapped_atoms;
  if (!lift.valid || lift.atom_images.size() != wrapped_atoms.size())
    return unwrapped;
  for (size_t atom_idx = 0; atom_idx < unwrapped.size(); ++atom_idx) {
    auto &atom = unwrapped[atom_idx];
    const auto &image = lift.atom_images[atom_idx];
    atom.sx = wrapped_atoms[atom_idx].sx + static_cast<double>(image.x);
    atom.sy = wrapped_atoms[atom_idx].sy + static_cast<double>(image.y);
    atom.sz = wrapped_atoms[atom_idx].sz + static_cast<double>(image.z);
    scaled_to_cartesian(Box, atom.sx, atom.sy, atom.sz, atom.x, atom.y, atom.z);
  }
  return unwrapped;
}

PbctopoComponentCandidate build_component_candidate_seed(
    const PbctopoCutCandidate &source, const vector<vector<size_t>> &components,
    const vector<Vec3> &placed_points) {
  PbctopoComponentCandidate seed;
  if (!source.valid || placed_points.empty())
    return seed;
  seed.continuity_path = source.continuity_path;
  seed.continuity_max_d2 = source.continuity_max_d2;
  seed.continuity_min_margin_A = source.continuity_min_margin_A;
  seed.continuity_margin_atom_a = source.continuity_margin_atom_a;
  seed.continuity_margin_atom_b = source.continuity_margin_atom_b;
  seed.anchor_mst2 = source.anchor_mst2;
  seed.anchor_max_d2 = source.anchor_max_d2;
  seed.anchor_graph = source.anchor_graph;
  seed.principal_rg2_max = source.principal_rg2_max;
  seed.principal_rg2_mid = source.principal_rg2_mid;
  seed.principal_rg2_min = source.principal_rg2_min;
  seed.bbox_volume = source.bbox_volume;
  seed.rg2 = source.rg2;
  seed.shape_kappa2 = source.shape_kappa2;
  seed.shape_asphericity = source.shape_asphericity;
  seed.shape_acylindricity = source.shape_acylindricity;
  seed.shift_norm2 = source.shift_norm2;
  seed.min_pair_d2 = source.min_pair_d2;
  seed.clash_atom_a = source.clash_atom_a;
  seed.clash_atom_b = source.clash_atom_b;
  seed.ultra_clash = source.ultra_clash;
  vector<Vec3> component_centers;
  component_centers.reserve(components.size());
  for (const auto &component : components) {
    if (component.empty())
      continue;
    Vec3 center{};
    for (size_t atom_idx : component)
      center = center + placed_points[atom_idx];
    component_centers.push_back(
        center * (1.0 / static_cast<double>(component.size())));
  }
  seed.component_rg2 = compute_rg2(component_centers);
  const PointStats component_stats = build_point_stats(component_centers);
  const auto component_principal_rg2 = compute_principal_rg2(component_stats);
  const auto component_shape_metrics =
      compute_gyration_shape_metrics(component_principal_rg2);
  seed.component_kappa2 = component_shape_metrics.kappa2;
  seed.component_asphericity = component_shape_metrics.asphericity;
  seed.component_acylindricity = component_shape_metrics.acylindricity;
  seed.component_mst2 = compute_mst2(component_centers);
  seed.valid = true;
  return seed;
}

PbctopoComponentCandidate score_component_assignment(
    const vector<PbctopoAtom> &atoms, const vector<Vec3> &placed_points,
    const vector<vector<size_t>> &components, double min_pair_d2,
    size_t clash_atom_a, size_t clash_atom_b,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, double continuity_cutoff,
    double shift_norm2, const PbctopoComponentCandidate *incumbent = nullptr,
    PbctopoTimingStats *timing = nullptr,
    PbctopoAnchorBackend anchor_backend = PbctopoAnchorBackend::Auto,
    const PbctopoAnchorGraphContext *anchor_context = nullptr) {
  using clock = chrono::steady_clock;
  const auto started = clock::now();
  auto finish = [&](PbctopoComponentCandidate candidate) {
    if (timing != nullptr) {
      ++timing->component_score_calls;
      timing->component_score_s +=
          chrono::duration<double>(clock::now() - started).count();
    }
    return candidate;
  };
  PbctopoComponentCandidate candidate;
  const bool debug_small_search = (components.size() <= 2);
  if (atoms.empty() || placed_points.size() != atoms.size())
    return finish(candidate);
  candidate.shift_norm2 = shift_norm2;
  candidate.min_pair_d2 = min_pair_d2;
  candidate.clash_atom_a = clash_atom_a;
  candidate.clash_atom_b = clash_atom_b;
  PbctopoCutCandidate incumbent_cut_storage;
  const PbctopoCutCandidate *incumbent_cut = nullptr;
  if (incumbent != nullptr && incumbent->valid && !incumbent->ultra_clash) {
    incumbent_cut_storage = component_candidate_to_cut_candidate(*incumbent);
    incumbent_cut = &incumbent_cut_storage;
  }

  auto stage_started = clock::now();
  if (!continuity_metrics_ok(placed_points, continuity_pairs, continuity_cutoff,
                             candidate.continuity_path,
                             candidate.continuity_max_d2,
                             candidate.continuity_min_margin_A,
                             candidate.continuity_margin_atom_a,
                             candidate.continuity_margin_atom_b,
                             incumbent_cut)) {
    if (timing != nullptr) {
      ++timing->component_score_continuity_rejects;
      timing->component_score_continuity_s +=
          chrono::duration<double>(clock::now() - stage_started).count();
    }
    if (debug_small_search) {
      append_pbctopo_log_line(
          "[pbctopo-debug] component leaf rejected by continuity: max_d2=" +
          to_string(candidate.continuity_max_d2) +
          ", path=" + to_string(candidate.continuity_path));
    }
    return finish(candidate);
  }
  if (timing != nullptr) {
    timing->component_score_continuity_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
  }
  if (component_candidate_cannot_beat_incumbent_prefix(
          candidate, incumbent, PbctopoComponentCandidatePrefixStage::Continuity)) {
    if (timing != nullptr)
      ++timing->component_score_prefix_prunes_after_continuity;
    return finish(candidate);
  }

  stage_started = clock::now();
  if (!anchor_graph_metrics_enhanced(
          placed_points, anchor_indices, continuity_pairs, candidate.anchor_mst2,
          candidate.anchor_max_d2, candidate.anchor_graph, incumbent_cut,
          PbctopoRefineScorePhase::Exact, anchor_backend, nullptr,
          anchor_context)) {
    if (timing != nullptr) {
      ++timing->component_score_anchor_rejects;
      timing->component_score_anchor_s +=
          chrono::duration<double>(clock::now() - stage_started).count();
    }
    if (debug_small_search) {
      append_pbctopo_log_line(
          "[pbctopo-debug] component leaf rejected by anchor-graph: max_d2=" +
          to_string(candidate.anchor_max_d2) +
          ", mst2=" + to_string(candidate.anchor_mst2));
    }
    return finish(candidate);
  }
  if (timing != nullptr) {
    timing->component_score_anchor_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
  }
  if (component_candidate_cannot_beat_incumbent_prefix(
          candidate, incumbent, PbctopoComponentCandidatePrefixStage::Anchor)) {
    if (timing != nullptr)
      ++timing->component_score_prefix_prunes_after_anchor;
    return finish(candidate);
  }

  stage_started = clock::now();
  const PointStats stats = build_point_stats(placed_points);
  if (stats.count == 0)
    return finish(candidate);
  const array<double, 3> principal_rg2 = compute_principal_rg2(stats);
  candidate.principal_rg2_max = principal_rg2[0];
  candidate.principal_rg2_mid = principal_rg2[1];
  candidate.principal_rg2_min = principal_rg2[2];
  const auto shape_metrics = compute_gyration_shape_metrics(principal_rg2);
  candidate.shape_kappa2 = shape_metrics.kappa2;
  candidate.shape_asphericity = shape_metrics.asphericity;
  candidate.shape_acylindricity = shape_metrics.acylindricity;
  candidate.rg2 = compute_rg2_from_stats(stats);
  candidate.bbox_volume = compute_bbox_volume(stats);
  if (timing != nullptr) {
    timing->component_score_stats_principal_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
  }
  if (component_candidate_cannot_beat_incumbent_prefix(
          candidate, incumbent, PbctopoComponentCandidatePrefixStage::Principal)) {
    if (timing != nullptr)
      ++timing->component_score_prefix_prunes_after_principal;
    return finish(candidate);
  }

  stage_started = clock::now();
  vector<Vec3> component_centers;
  component_centers.reserve(components.size());
  for (const auto &component : components) {
    if (component.empty())
      continue;
    Vec3 center{};
    for (size_t atom_idx : component)
      center = center + placed_points[atom_idx];
    component_centers.push_back(
        center * (1.0 / static_cast<double>(component.size())));
  }
  candidate.component_rg2 = compute_rg2(component_centers);
  const PointStats component_stats = build_point_stats(component_centers);
  const auto component_principal_rg2 = compute_principal_rg2(component_stats);
  const auto component_shape_metrics =
      compute_gyration_shape_metrics(component_principal_rg2);
  candidate.component_kappa2 = component_shape_metrics.kappa2;
  candidate.component_asphericity = component_shape_metrics.asphericity;
  candidate.component_acylindricity = component_shape_metrics.acylindricity;
  candidate.component_mst2 = compute_mst2(component_centers);
  if (timing != nullptr) {
    timing->component_score_component_metrics_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
  }
  if (component_candidate_cannot_beat_incumbent_prefix(
          candidate, incumbent,
          PbctopoComponentCandidatePrefixStage::ComponentMetrics)) {
    if (timing != nullptr)
      ++timing->component_score_prefix_prunes_after_component_metrics;
    return finish(candidate);
  }
  if (component_candidate_cannot_beat_incumbent_prefix(
          candidate, incumbent, PbctopoComponentCandidatePrefixStage::Final)) {
    if (timing != nullptr)
      ++timing->component_score_prefix_prunes_after_final;
    return finish(candidate);
  }
  candidate.valid = true;
  if (debug_small_search) {
    append_pbctopo_log_line(
        "[pbctopo-debug] component leaf valid: cont_max_d2=" +
        to_string(candidate.continuity_max_d2) +
        ", cont_path=" + to_string(candidate.continuity_path) +
        ", anchor_max_d2=" + to_string(candidate.anchor_max_d2) +
        ", anchor_mst2=" + to_string(candidate.anchor_mst2) +
        ", princ_max=" + to_string(candidate.principal_rg2_max) +
        ", comp_mst2=" + to_string(candidate.component_mst2) +
        ", kappa2=" + to_string(candidate.shape_kappa2) +
        ", asph=" + to_string(candidate.shape_asphericity));
  }
  return finish(candidate);
}

PbctopoComponentSearchCallbacks make_component_search_callbacks() {
  PbctopoComponentSearchCallbacks callbacks;
  callbacks.build_seed = build_component_candidate_seed;
  callbacks.score_assignment = score_component_assignment;
  callbacks.better_candidate = better_component_candidate;
  return callbacks;
}

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
        nullptr) {
  best_cut = {};
  shift_norm2 = 0.0;
  placed_points.clear();
  if (merge_plan.merged_components.empty())
    return false;

  const vector<PbctopoAtom> merged_atoms =
      apply_merged_component_internal_offsets(atoms, raw_components,
                                              merge_plan, Box);
  if (timing != nullptr) {
    ++timing->component_large_mode_internal_offsets_preserved;
    timing->component_large_mode_internal_offset_nonzero +=
        count_if(merge_plan.raw_internal_offsets.begin(),
                 merge_plan.raw_internal_offsets.end(),
                 [](const Int3 &offset) { return !is_zero_delta(offset); });
  }

  vector<Int3> merged_warm_offsets;
  const vector<Int3> *merged_warm_offsets_ptr =
      (warm_component_offsets != nullptr &&
       derive_merged_warm_offsets(*warm_component_offsets, merge_plan,
                                  merged_warm_offsets))
          ? &merged_warm_offsets
          : nullptr;

  vector<Vec3> best_points_local;
  vector<Int3> best_raw_offsets;
  auto consider_raw_offsets = [&](const vector<Int3> &raw_offsets) {
    vector<Int3> refine_offsets = raw_offsets;
    vector<Vec3> refine_points;
    PbctopoCutCandidate candidate = improve_component_offsets_locally(
        atoms, raw_components, Box, shifts, local_edges, edge_set,
        continuity_pairs, anchor_indices, refine_offsets, cutoffs.clash_cutoff,
        cutoffs.continuity_cutoff, refine_points,
        make_component_refine_callbacks(), anchor_backend, timing, warm_state,
        warm_reference_candidate, anchor_context);
    if (candidate.valid && !candidate.ultra_clash &&
        refine_points.size() == atoms.size() &&
        (!best_cut.valid || better_cut_candidate(candidate, best_cut))) {
      best_cut = candidate;
      best_points_local = refine_points;
      best_raw_offsets = refine_offsets;
    }
  };

  const vector<PbctopoGraphSeedSolveResult> graph_seed_results =
      build_large_component_graph_seed_candidates(
          merged_atoms, Box, merge_plan.merged_components, shifts,
          continuity_pairs, anchor_indices, cutoffs, merged_warm_offsets_ptr, timing,
          anchor_backend, anchor_context, required_contacts);
  if (timing != nullptr)
    timing->component_large_mode_graph_seed_count += graph_seed_results.size();

  const PbctopoGraphSeedSolveResult *selected_seed = nullptr;
  for (const auto &seed_result : graph_seed_results) {
    if (!seed_result.valid)
      continue;
    if (selected_seed == nullptr ||
        graph_seed_quick_score_better(seed_result.quick_score,
                                      selected_seed->quick_score)) {
      selected_seed = &seed_result;
    }
  }
  if (selected_seed == nullptr && !graph_seed_results.empty())
    selected_seed = &graph_seed_results.front();

  if (selected_seed != nullptr &&
      selected_seed->offsets.size() == merge_plan.merged_components.size()) {
    if (timing != nullptr) {
      ++timing->component_large_mode_graph_seed_refined;
      if (string(selected_seed->seed_name) != "baseline")
        ++timing->component_large_mode_graph_seed_nonbaseline_win;
    }
    consider_raw_offsets(expand_merged_offsets_to_raw_components(
        selected_seed->offsets, Box, raw_components, merge_plan));
  }

  if (allow_exact_search_on_merged) {
    vector<Vec3> merged_search_points;
    const PbctopoCutCandidate *search_incumbent =
        (best_cut.valid && !best_cut.ultra_clash) ? &best_cut : warm_reference_candidate;
    const vector<Vec3> *search_incumbent_points =
        (best_cut.valid && !best_cut.ultra_clash &&
         best_points_local.size() == atoms.size())
            ? &best_points_local
            : nullptr;
    if (timing != nullptr)
      ++timing->component_search_frames;
    PbctopoComponentCandidate merged_candidate = search_component_shift_candidate(
        merged_atoms, Box, merge_plan.merged_components, shifts, continuity_pairs,
        anchor_indices, cutoffs, merged_search_points,
        make_component_search_callbacks(), merged_warm_offsets_ptr,
        search_incumbent, search_incumbent_points, timing, anchor_backend,
        anchor_context, nullptr, nullptr, 0, required_contacts);
    if (merged_candidate.valid && !merged_candidate.ultra_clash &&
        merged_search_points.size() == atoms.size()) {
      const vector<Int3> merged_search_offsets = extract_component_offsets_from_points(
          merged_atoms, merge_plan.merged_components, Box,
          merged_search_points);
      if (merged_search_offsets.size() == merge_plan.merged_components.size()) {
        consider_raw_offsets(expand_merged_offsets_to_raw_components(
            merged_search_offsets, Box, raw_components, merge_plan));
      }
    }
  }

  if (!best_cut.valid || best_points_local.size() != atoms.size())
    return false;
  placed_points = best_points_local;
  shift_norm2 = best_cut.shift_norm2;
  if (best_raw_offsets_out != nullptr)
    *best_raw_offsets_out = best_raw_offsets;
  return true;
}

double clamp_virtual_box_center(double center) {
  if (center < 0.5)
    return 0.5;
  if (center > 2.5)
    return 2.5;
  return center;
}

bool find_virtual_box_shift(double wrapped_coord, double center, int &shift_idx) {
  const double low = center - 0.5;
  const double high = center + 0.5;
  for (int candidate = 0; candidate <= 2; ++candidate) {
    const double coord = wrapped_coord + static_cast<double>(candidate);
    if (coord + kDistanceEps >= low && coord < high - kDistanceEps) {
      shift_idx = candidate;
      return true;
    }
  }
  const double mid = 0.5 * (low + high);
  double best_dist = numeric_limits<double>::infinity();
  int best_candidate = 0;
  for (int candidate = 0; candidate <= 2; ++candidate) {
    const double coord = wrapped_coord + static_cast<double>(candidate);
    const double dist = fabs(coord - mid);
    if (dist < best_dist) {
      best_dist = dist;
      best_candidate = candidate;
    }
  }
  shift_idx = best_candidate;
  return true;
}

void push_unique_xy_seed(vector<PbctopoSeed2D> &best, const PbctopoSeed2D &seed,
                         size_t max_keep) {
  for (const auto &existing : best) {
    if (fabs(existing.cx - seed.cx) <= 1.0e-9 &&
        fabs(existing.cy - seed.cy) <= 1.0e-9) {
      return;
    }
  }
  best.push_back(seed);
  sort(best.begin(), best.end(), [](const PbctopoSeed2D &lhs,
                                    const PbctopoSeed2D &rhs) {
    int cmp = compare_pbctopo_double_metric(lhs.score, rhs.score, 1.0e-9);
    if (cmp != 0)
      return cmp > 0;
    cmp = compare_pbctopo_double_metric(lhs.cx, rhs.cx, 1.0e-9);
    if (cmp != 0)
      return cmp < 0;
    return compare_pbctopo_double_metric(lhs.cy, rhs.cy, 1.0e-9) < 0;
  });
  if (best.size() > max_keep)
    best.resize(max_keep);
}

void push_unique_axis_seed(vector<PbctopoCutOption> &best,
                           const PbctopoCutOption &seed, size_t max_keep) {
  for (const auto &existing : best) {
    if (fabs(existing.cut - seed.cut) <= 1.0e-9)
      return;
  }
  best.push_back(seed);
  sort(best.begin(), best.end(),
       [](const PbctopoCutOption &lhs, const PbctopoCutOption &rhs) {
         const int gap_cmp =
             compare_pbctopo_double_metric(lhs.gap, rhs.gap, 1.0e-9);
         if (gap_cmp != 0)
           return gap_cmp > 0;
         return compare_pbctopo_double_metric(lhs.cut, rhs.cut, 1.0e-9) < 0;
       });
  if (best.size() > max_keep)
    best.resize(max_keep);
}

void push_unique_xyz_seed(vector<PbctopoSeed3D> &best, const PbctopoSeed3D &seed,
                          size_t max_keep) {
  for (const auto &existing : best) {
    if (fabs(existing.cx - seed.cx) <= 1.0e-9 &&
        fabs(existing.cy - seed.cy) <= 1.0e-9 &&
        fabs(existing.cz - seed.cz) <= 1.0e-9) {
      return;
    }
  }
  best.push_back(seed);
  sort(best.begin(), best.end(), [](const PbctopoSeed3D &lhs,
                                    const PbctopoSeed3D &rhs) {
    int cmp = compare_pbctopo_double_metric(lhs.score, rhs.score, 1.0e-9);
    if (cmp != 0)
      return cmp > 0;
    cmp = compare_pbctopo_double_metric(lhs.cx, rhs.cx, 1.0e-9);
    if (cmp != 0)
      return cmp < 0;
    cmp = compare_pbctopo_double_metric(lhs.cy, rhs.cy, 1.0e-9);
    if (cmp != 0)
      return cmp < 0;
    return compare_pbctopo_double_metric(lhs.cz, rhs.cz, 1.0e-9) < 0;
  });
  if (best.size() > max_keep)
    best.resize(max_keep);
}

double projection_axis_coord(const PbctopoAtom &atom, char axis) {
  if (axis == 'x')
    return atom.sx;
  if (axis == 'y')
    return atom.sy;
  return atom.sz;
}

vector<double> build_projection_density_grid(const vector<PbctopoAtom> &atoms,
                                             size_t mesh_bins, double radius,
                                             char axis_a, char axis_b) {
  vector<double> scores(mesh_bins * mesh_bins, 0.0);
  if (atoms.empty() || mesh_bins == 0)
    return scores;

  const double span = 2.0;
  const double step =
      (mesh_bins == 1) ? 0.0 : span / static_cast<double>(mesh_bins - 1);
  auto clamp_index = [mesh_bins](long long value) -> size_t {
    if (value < 0)
      return 0;
    const long long max_index = static_cast<long long>(mesh_bins) - 1;
    if (value > max_index)
      return static_cast<size_t>(max_index);
    return static_cast<size_t>(value);
  };

  for (const auto &atom : atoms) {
    const double wrapped_a = projection_axis_coord(atom, axis_a);
    const double wrapped_b = projection_axis_coord(atom, axis_b);
    for (int shift_a = 0; shift_a <= 2; ++shift_a) {
      const double shifted_a = wrapped_a + static_cast<double>(shift_a);
      if (shifted_a + radius < 0.5 || shifted_a - radius > 2.5)
        continue;
      const size_t ia_begin =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    ceil((shifted_a - radius - 0.5) / step)));
      const size_t ia_end =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    floor((shifted_a + radius - 0.5) / step)));
      for (int shift_b = 0; shift_b <= 2; ++shift_b) {
        const double shifted_b = wrapped_b + static_cast<double>(shift_b);
        if (shifted_b + radius < 0.5 || shifted_b - radius > 2.5)
          continue;
        const size_t ib_begin =
            (step <= kDistanceEps)
                ? 0
                : clamp_index(static_cast<long long>(
                      ceil((shifted_b - radius - 0.5) / step)));
        const size_t ib_end =
            (step <= kDistanceEps)
                ? 0
                : clamp_index(static_cast<long long>(
                      floor((shifted_b + radius - 0.5) / step)));
        for (size_t ia = ia_begin; ia <= ia_end; ++ia) {
          const double ca = 0.5 + step * static_cast<double>(ia);
          const double da = fabs(shifted_a - ca);
          if (da > radius)
            continue;
          for (size_t ib = ib_begin; ib <= ib_end; ++ib) {
            const double cb = 0.5 + step * static_cast<double>(ib);
            const double db = fabs(shifted_b - cb);
            if (db > radius)
              continue;
            scores[ia * mesh_bins + ib] += (radius - da) * (radius - db);
          }
        }
      }
    }
  }
  return scores;
}

double projection_grid_center(size_t mesh_bins, size_t idx) {
  if (mesh_bins <= 1)
    return 1.5;
  const double step = 2.0 / static_cast<double>(mesh_bins - 1);
  return 0.5 + step * static_cast<double>(idx);
}

bool projection_density_is_local_maximum(const vector<double> &scores,
                                         size_t mesh_bins, size_t ix,
                                         size_t iy) {
  const double score = scores[ix * mesh_bins + iy];
  if (!(score > 0.0))
    return false;

  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0)
        continue;
      const long long nx_ll = static_cast<long long>(ix) + dx;
      const long long ny_ll = static_cast<long long>(iy) + dy;
      if (nx_ll < 0 || ny_ll < 0)
        continue;
      const size_t nx = static_cast<size_t>(nx_ll);
      const size_t ny = static_cast<size_t>(ny_ll);
      if (nx >= mesh_bins || ny >= mesh_bins)
        continue;
      const double neighbor = scores[nx * mesh_bins + ny];
      if (neighbor > score + 1.0e-12)
        return false;
      if (fabs(neighbor - score) <= 1.0e-12 &&
          (nx < ix || (nx == ix && ny < iy))) {
        return false;
      }
    }
  }
  return true;
}

bool projection_seed2d_better(const PbctopoSeed2D &lhs,
                              const PbctopoSeed2D &rhs) {
  const int score_cmp =
      compare_pbctopo_double_metric(lhs.score, rhs.score, 1.0e-12);
  if (score_cmp != 0)
    return score_cmp > 0;
  if (lhs.ix != rhs.ix)
    return lhs.ix < rhs.ix;
  return lhs.iy < rhs.iy;
}

void collect_projection_plane_hotspots(const vector<double> &scores,
                                       size_t mesh_bins, size_t top_k,
                                       vector<PbctopoSeed2D> &hotspots) {
  hotspots.clear();
  if (mesh_bins == 0 || top_k == 0 || scores.size() != mesh_bins * mesh_bins)
    return;

  vector<PbctopoSeed2D> local_maxima;
  vector<PbctopoSeed2D> ranked_cells;
  local_maxima.reserve(mesh_bins * mesh_bins / 4 + 1);
  ranked_cells.reserve(mesh_bins * mesh_bins);

  for (size_t ix = 0; ix < mesh_bins; ++ix) {
    const double cx = projection_grid_center(mesh_bins, ix);
    for (size_t iy = 0; iy < mesh_bins; ++iy) {
      const double score = scores[ix * mesh_bins + iy];
      if (!(score > 0.0))
        continue;
      PbctopoSeed2D seed{cx, projection_grid_center(mesh_bins, iy), score, ix,
                         iy};
      ranked_cells.push_back(seed);
      if (projection_density_is_local_maximum(scores, mesh_bins, ix, iy))
        local_maxima.push_back(seed);
    }
  }

  sort(local_maxima.begin(), local_maxima.end(), projection_seed2d_better);
  sort(ranked_cells.begin(), ranked_cells.end(), projection_seed2d_better);

  auto append_from = [&](const vector<PbctopoSeed2D> &source,
                         long long min_bin_gap) {
    for (const auto &seed : source) {
      if (hotspots.size() >= top_k)
        break;
      bool too_close = false;
      for (const auto &existing : hotspots) {
        const long long dx =
            llabs(static_cast<long long>(existing.ix) -
                  static_cast<long long>(seed.ix));
        const long long dy =
            llabs(static_cast<long long>(existing.iy) -
                  static_cast<long long>(seed.iy));
        if (max(dx, dy) <= min_bin_gap) {
          too_close = true;
          break;
        }
      }
      if (!too_close)
        hotspots.push_back(seed);
    }
  };

  const long long nms_bin_gap = (mesh_bins >= 10) ? 1ll : 0ll;
  append_from(local_maxima, nms_bin_gap);
  if (hotspots.size() < min(top_k, static_cast<size_t>(4)))
    append_from(ranked_cells, nms_bin_gap);
  if (hotspots.size() < top_k)
    append_from(ranked_cells, 0);
}

double projection_grid_score(const vector<double> &scores, size_t mesh_bins,
                             size_t ia, size_t ib) {
  if (ia >= mesh_bins || ib >= mesh_bins || scores.size() != mesh_bins * mesh_bins)
    return 0.0;
  return scores[ia * mesh_bins + ib];
}

void append_dense_projection_volume_seeds(const vector<double> &xy_scores,
                                          const vector<double> &xz_scores,
                                          const vector<double> &yz_scores,
                                          size_t mesh_bins, size_t top_k,
                                          vector<PbctopoSeed3D> &seeds) {
  for (size_t ix = 0; ix < mesh_bins; ++ix) {
    const double cx = projection_grid_center(mesh_bins, ix);
    for (size_t iy = 0; iy < mesh_bins; ++iy) {
      const double cy = projection_grid_center(mesh_bins, iy);
      const double xy_score = projection_grid_score(xy_scores, mesh_bins, ix, iy);
      for (size_t iz = 0; iz < mesh_bins; ++iz) {
        const double score =
            xy_score + projection_grid_score(xz_scores, mesh_bins, ix, iz) +
            projection_grid_score(yz_scores, mesh_bins, iy, iz);
        push_unique_xyz_seed(
            seeds,
            {cx, cy, projection_grid_center(mesh_bins, iz), score, ix, iy, iz},
            top_k);
      }
    }
  }
}

void collect_projection_volume_seeds(const vector<PbctopoAtom> &atoms,
                                     size_t mesh_bins, size_t top_k,
                                     double radius,
                                     vector<PbctopoSeed3D> &seeds) {
  seeds.clear();
  if (atoms.empty() || mesh_bins == 0 || top_k == 0)
    return;

  const auto xy_scores =
      build_projection_density_grid(atoms, mesh_bins, radius, 'x', 'y');
  const auto xz_scores =
      build_projection_density_grid(atoms, mesh_bins, radius, 'x', 'z');
  const auto yz_scores =
      build_projection_density_grid(atoms, mesh_bins, radius, 'y', 'z');

  vector<PbctopoSeed2D> xy_hotspots;
  vector<PbctopoSeed2D> xz_hotspots;
  vector<PbctopoSeed2D> yz_hotspots;
  const size_t plane_top_k =
      min(mesh_bins * mesh_bins, max<size_t>(top_k * 2, static_cast<size_t>(12)));
  collect_projection_plane_hotspots(xy_scores, mesh_bins, plane_top_k,
                                    xy_hotspots);
  collect_projection_plane_hotspots(xz_scores, mesh_bins, plane_top_k,
                                    xz_hotspots);
  collect_projection_plane_hotspots(yz_scores, mesh_bins, plane_top_k,
                                    yz_hotspots);

  seeds.reserve(top_k);
  auto push_seed = [&](size_t ix, size_t iy, size_t iz, double score) {
    push_unique_xyz_seed(
        seeds,
        {projection_grid_center(mesh_bins, ix), projection_grid_center(mesh_bins, iy),
         projection_grid_center(mesh_bins, iz), score, ix, iy, iz},
        top_k);
  };

  for (const auto &xy_seed : xy_hotspots) {
    for (const auto &xz_seed : xz_hotspots) {
      if (xy_seed.ix != xz_seed.ix)
        continue;
      for (const auto &yz_seed : yz_hotspots) {
        if (xy_seed.iy != yz_seed.ix || xz_seed.iy != yz_seed.iy)
          continue;
        push_seed(xy_seed.ix, xy_seed.iy, xz_seed.iy,
                  xy_seed.score + xz_seed.score + yz_seed.score);
      }
    }
  }

  if (seeds.size() < top_k) {
    for (const auto &xy_seed : xy_hotspots) {
      for (const auto &xz_seed : xz_hotspots) {
        if (xy_seed.ix != xz_seed.ix)
          continue;
        push_seed(xy_seed.ix, xy_seed.iy, xz_seed.iy,
                  xy_seed.score + xz_seed.score +
                      projection_grid_score(yz_scores, mesh_bins, xy_seed.iy,
                                            xz_seed.iy));
      }
    }
  }

  if (seeds.size() < top_k) {
    for (const auto &xy_seed : xy_hotspots) {
      for (const auto &yz_seed : yz_hotspots) {
        if (xy_seed.iy != yz_seed.ix)
          continue;
        push_seed(xy_seed.ix, xy_seed.iy, yz_seed.iy,
                  xy_seed.score + yz_seed.score +
                      projection_grid_score(xz_scores, mesh_bins, xy_seed.ix,
                                            yz_seed.iy));
      }
    }
  }

  if (seeds.size() < top_k) {
    for (const auto &xz_seed : xz_hotspots) {
      for (const auto &yz_seed : yz_hotspots) {
        if (xz_seed.iy != yz_seed.iy)
          continue;
        push_seed(xz_seed.ix, yz_seed.ix, xz_seed.iy,
                  xz_seed.score + yz_seed.score +
                      projection_grid_score(xy_scores, mesh_bins, xz_seed.ix,
                                            yz_seed.ix));
      }
    }
  }

  if (seeds.size() < top_k)
    append_dense_projection_volume_seeds(xy_scores, xz_scores, yz_scores,
                                         mesh_bins, top_k, seeds);
}

[[maybe_unused]] vector<PbctopoSeed2D>
collect_xy_density_seeds(const vector<PbctopoAtom> &atoms, size_t mesh_bins,
                         size_t top_k, double radius) {
  vector<PbctopoSeed2D> seeds;
  if (atoms.empty() || mesh_bins == 0 || top_k == 0)
    return seeds;

  const double span = 2.0;
  const double step =
      (mesh_bins == 1) ? 0.0 : span / static_cast<double>(mesh_bins - 1);
  vector<double> scores(mesh_bins * mesh_bins, 0.0);
  auto clamp_index = [mesh_bins](long long value) -> size_t {
    if (value < 0)
      return 0;
    const long long max_index = static_cast<long long>(mesh_bins) - 1;
    if (value > max_index)
      return static_cast<size_t>(max_index);
    return static_cast<size_t>(value);
  };

  for (const auto &atom : atoms) {
    for (int shift_x = 0; shift_x <= 2; ++shift_x) {
      const double shifted_x = atom.sx + static_cast<double>(shift_x);
      if (shifted_x + radius < 0.5 || shifted_x - radius > 2.5)
        continue;
      const size_t ix_begin =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    ceil((shifted_x - radius - 0.5) / step)));
      const size_t ix_end =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    floor((shifted_x + radius - 0.5) / step)));
      for (int shift_y = 0; shift_y <= 2; ++shift_y) {
        const double shifted_y = atom.sy + static_cast<double>(shift_y);
        if (shifted_y + radius < 0.5 || shifted_y - radius > 2.5)
          continue;
        const size_t iy_begin =
            (step <= kDistanceEps)
                ? 0
                : clamp_index(static_cast<long long>(
                      ceil((shifted_y - radius - 0.5) / step)));
        const size_t iy_end =
            (step <= kDistanceEps)
                ? 0
                : clamp_index(static_cast<long long>(
                      floor((shifted_y + radius - 0.5) / step)));
        for (size_t ix = ix_begin; ix <= ix_end; ++ix) {
          const double cx = 0.5 + step * static_cast<double>(ix);
          const double dx = fabs(shifted_x - cx);
          if (dx > radius)
            continue;
          for (size_t iy = iy_begin; iy <= iy_end; ++iy) {
            const double cy = 0.5 + step * static_cast<double>(iy);
            const double dy = fabs(shifted_y - cy);
            if (dy > radius)
              continue;
            scores[ix * mesh_bins + iy] += (radius - dx) * (radius - dy);
          }
        }
      }
    }
  }

  for (size_t ix = 0; ix < mesh_bins; ++ix) {
    const double cx = 0.5 + step * static_cast<double>(ix);
    for (size_t iy = 0; iy < mesh_bins; ++iy) {
      const double cy = 0.5 + step * static_cast<double>(iy);
      push_unique_xy_seed(seeds, {cx, cy, scores[ix * mesh_bins + iy]}, top_k);
    }
  }
  return seeds;
}

[[maybe_unused]] vector<PbctopoCutOption>
collect_z_density_seeds(const vector<PbctopoAtom> &atoms, size_t mesh_bins,
                        size_t top_k, double radius) {
  vector<PbctopoCutOption> seeds;
  if (atoms.empty() || mesh_bins == 0 || top_k == 0)
    return seeds;

  const double span = 2.0;
  const double step =
      (mesh_bins == 1) ? 0.0 : span / static_cast<double>(mesh_bins - 1);
  vector<double> scores(mesh_bins, 0.0);
  auto clamp_index = [mesh_bins](long long value) -> size_t {
    if (value < 0)
      return 0;
    const long long max_index = static_cast<long long>(mesh_bins) - 1;
    if (value > max_index)
      return static_cast<size_t>(max_index);
    return static_cast<size_t>(value);
  };

  for (const auto &atom : atoms) {
    for (int shift_z = 0; shift_z <= 2; ++shift_z) {
      const double shifted_z = atom.sz + static_cast<double>(shift_z);
      if (shifted_z + radius < 0.5 || shifted_z - radius > 2.5)
        continue;
      const size_t iz_begin =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    ceil((shifted_z - radius - 0.5) / step)));
      const size_t iz_end =
          (step <= kDistanceEps)
              ? 0
              : clamp_index(static_cast<long long>(
                    floor((shifted_z + radius - 0.5) / step)));
      for (size_t iz = iz_begin; iz <= iz_end; ++iz) {
        const double cz = 0.5 + step * static_cast<double>(iz);
        const double dz = fabs(shifted_z - cz);
        if (dz > radius)
          continue;
        scores[iz] += radius - dz;
      }
    }
  }

  for (size_t iz = 0; iz < mesh_bins; ++iz) {
    const double cz = 0.5 + step * static_cast<double>(iz);
    push_unique_axis_seed(seeds, {cz, scores[iz]}, top_k);
  }
  return seeds;
}

void push_top_virtual_box_candidate(vector<PbctopoCutCandidate> &top_candidates,
                                    const PbctopoCutCandidate &candidate,
                                    size_t max_keep) {
  if (!candidate.valid)
    return;
  top_candidates.push_back(candidate);
  sort(top_candidates.begin(), top_candidates.end(), better_cut_candidate);
  if (top_candidates.size() > max_keep)
    top_candidates.resize(max_keep);
}

[[maybe_unused]] bool
build_virtual_box_positions(const vector<PbctopoAtom> &atoms, box Box,
                            double center_x, double center_y, double center_z,
                            vector<Vec3> &placed_points, PointStats &stats,
                            double &shift_norm2) {
  placed_points.resize(atoms.size());
  stats = PointStats{};
  shift_norm2 = 0.0;

  const int global_shift_x = static_cast<int>(floor(center_x)) - 1;
  const int global_shift_y = static_cast<int>(floor(center_y)) - 1;
  const int global_shift_z = static_cast<int>(floor(center_z)) - 1;

  for (size_t idx = 0; idx < atoms.size(); ++idx) {
    const auto &atom = atoms[idx];
    int search_shift_x = -1;
    int search_shift_y = -1;
    int search_shift_z = -1;
    if (!find_virtual_box_shift(atom.sx, center_x, search_shift_x) ||
        !find_virtual_box_shift(atom.sy, center_y, search_shift_y) ||
        !find_virtual_box_shift(atom.sz, center_z, search_shift_z)) {
      return false;
    }

    const int actual_shift_x = search_shift_x - 1 - global_shift_x;
    const int actual_shift_y = search_shift_y - 1 - global_shift_y;
    const int actual_shift_z = search_shift_z - 1 - global_shift_z;

    Vec3 point{};
    scaled_to_cartesian(Box, atom.sx + static_cast<double>(actual_shift_x),
                        atom.sy + static_cast<double>(actual_shift_y),
                        atom.sz + static_cast<double>(actual_shift_z), point.x,
                        point.y, point.z);
    placed_points[idx] = point;
    stats.add(point.x, point.y, point.z);
    shift_norm2 += static_cast<double>(actual_shift_x * actual_shift_x +
                                       actual_shift_y * actual_shift_y +
                                       actual_shift_z * actual_shift_z);
  }

  return stats.count == atoms.size();
}

PbctopoCutCandidate evaluate_virtual_box_candidate(
    const vector<PbctopoAtom> &atoms, const vector<PbctopoLocalEdge> &local_edges,
    const unordered_set<uint64_t> &edge_set,
    const PbctopoTranslationCache &translation_cache, box Box, double cut_x,
    double cut_y, double cut_z, double gap_score, vector<Vec3> &placed_points,
    double clash_cutoff, const vector<PbctopoContinuityPair> *continuity_pairs,
    const vector<size_t> *anchor_indices,
    const vector<pair<int, int>> *contacting_owner_pairs,
    double continuity_cutoff, const PbctopoCutCandidate *incumbent,
    bool skip_clash, double frame_time_ps, const char *reject_context,
    const PbctopoAnchorGraphContext *anchor_context,
    PbctopoVirtualSeedSource seed_source,
    const vector<size_t> *atom_component_index) {
  PbctopoCutCandidate candidate;
  candidate.cut_x = clamp_virtual_box_center(cut_x);
  candidate.cut_y = clamp_virtual_box_center(cut_y);
  candidate.cut_z = clamp_virtual_box_center(cut_z);
  candidate.gap_score = gap_score;
  candidate.virtual_seed_source = seed_source;
  candidate.broken_edges = 0;
  candidate.edge_stretch = 0.0;
  candidate.shift_norm2 = 0.0;
  const PbctopoCutCandidate *active_incumbent =
      (incumbent != nullptr && incumbent->valid && !incumbent->ultra_clash)
          ? incumbent
          : nullptr;
  const PbctopoCutCandidate *metric_incumbent =
      (active_incumbent != nullptr && active_incumbent->broken_edges == 0)
          ? active_incumbent
          : nullptr;

  if (atoms.empty())
    return candidate;

  placed_points.resize(atoms.size());
  PointStats stats;
  vector<Int3> atom_image_offsets(atoms.size());

  const int global_shift_x = static_cast<int>(floor(candidate.cut_x)) - 1;
  const int global_shift_y = static_cast<int>(floor(candidate.cut_y)) - 1;
  const int global_shift_z = static_cast<int>(floor(candidate.cut_z)) - 1;

  for (size_t idx = 0; idx < atoms.size(); ++idx) {
    const auto &atom = atoms[idx];
    int search_shift_x = -1;
    int search_shift_y = -1;
    int search_shift_z = -1;
    if (!find_virtual_box_shift(atom.sx, candidate.cut_x, search_shift_x) ||
        !find_virtual_box_shift(atom.sy, candidate.cut_y, search_shift_y) ||
        !find_virtual_box_shift(atom.sz, candidate.cut_z, search_shift_z)) {
      return PbctopoCutCandidate{};
    }

    const int actual_shift_x = search_shift_x - 1 - global_shift_x;
    const int actual_shift_y = search_shift_y - 1 - global_shift_y;
    const int actual_shift_z = search_shift_z - 1 - global_shift_z;
    const Vec3 &translation =
        translation_cache.by_shift[actual_shift_x + 2][actual_shift_y + 2]
                                  [actual_shift_z + 2];
    Vec3 point{atom.x + translation.x, atom.y + translation.y,
               atom.z + translation.z};
    placed_points[idx] = point;
    stats.add(point.x, point.y, point.z);
    atom_image_offsets[idx] = {actual_shift_x, actual_shift_y, actual_shift_z};
  }

  if (stats.count != atoms.size())
    return PbctopoCutCandidate{};

  const vector<Int3> *metric_offsets = &atom_image_offsets;
  vector<Int3> component_image_offsets;
  vector<unsigned char> component_offset_seen;
  bool component_assignment_consistent =
      atom_component_index != nullptr &&
      atom_component_index->size() == atoms.size();
  if (component_assignment_consistent) {
    size_t component_count = 0;
    for (size_t comp_idx : *atom_component_index) {
      if (comp_idx >= atoms.size()) {
        component_assignment_consistent = false;
        break;
      }
      component_count = max(component_count, comp_idx + 1);
    }
    if (component_assignment_consistent && component_count > 0) {
      component_image_offsets.resize(component_count);
      component_offset_seen.assign(component_count, 0);
      for (size_t atom_idx = 0; atom_idx < atom_image_offsets.size(); ++atom_idx) {
        const size_t comp_idx = (*atom_component_index)[atom_idx];
        if (component_offset_seen[comp_idx] == 0) {
          component_image_offsets[comp_idx] = atom_image_offsets[atom_idx];
          component_offset_seen[comp_idx] = 1;
        } else if (!same_delta(component_image_offsets[comp_idx],
                               atom_image_offsets[atom_idx])) {
          component_assignment_consistent = false;
          break;
        }
      }
      if (component_assignment_consistent &&
          find(component_offset_seen.begin(), component_offset_seen.end(), 0) ==
              component_offset_seen.end()) {
        metric_offsets = &component_image_offsets;
      }
    }
  }
  candidate.shift_norm2 = pbctopo_component_shift_metric(Box, *metric_offsets);

  if (contacting_owner_pairs != nullptr) {
    PbctopoOwnerPairSplitDetail split_detail;
    const bool split_observed = candidate_has_split_contacting_owners(
        placed_points, atoms, Box, *contacting_owner_pairs,
        candidate.split_owner_pairs,
        candidate.max_owner_raw_minus_minimg_delta, &split_detail);
    if (split_observed) {
      if (reject_context != nullptr && split_detail.valid &&
          isfinite(frame_time_ps)) {
        append_pbctopo_log_line(
            "[pbctopo] observe owner_com_image_gap at frame " +
            to_string(static_cast<long long>(frame_time_ps)) + " via " +
            reject_context + ": owners=" + to_string(split_detail.owner_a) + "-" +
            to_string(split_detail.owner_b) +
            ", raw_delta_scaled=" + format_vec3(split_detail.raw_delta_scaled) +
            ", min_image_delta_scaled=" +
            format_vec3(split_detail.minimg_delta_scaled) +
            ", policy=diagnostic_only");
      }
    }
  }

  if (continuity_pairs != nullptr) {
    double continuity_path = numeric_limits<double>::infinity();
    double continuity_max_d2 = numeric_limits<double>::infinity();
    double continuity_min_margin_A = numeric_limits<double>::quiet_NaN();
    size_t continuity_margin_atom_a = numeric_limits<size_t>::max();
    size_t continuity_margin_atom_b = numeric_limits<size_t>::max();
    if (!continuity_metrics_ok(placed_points, *continuity_pairs, continuity_cutoff,
                               continuity_path, continuity_max_d2,
                               continuity_min_margin_A,
                               continuity_margin_atom_a,
                               continuity_margin_atom_b,
                               metric_incumbent)) {
      return PbctopoCutCandidate{};
    }
    candidate.continuity_path = continuity_path;
    candidate.continuity_max_d2 = continuity_max_d2;
    candidate.continuity_min_margin_A = continuity_min_margin_A;
    candidate.continuity_margin_atom_a = continuity_margin_atom_a;
    candidate.continuity_margin_atom_b = continuity_margin_atom_b;
  }
  if (anchor_indices != nullptr) {
    double anchor_mst2 = numeric_limits<double>::infinity();
    double anchor_max_d2 = numeric_limits<double>::infinity();
    PbctopoAnchorGraphScore anchor_graph_score;
    const vector<PbctopoContinuityPair> empty_continuity_pairs;
    const vector<PbctopoContinuityPair> &anchor_continuity_pairs =
        (continuity_pairs != nullptr) ? *continuity_pairs
                                      : empty_continuity_pairs;
    if (!anchor_graph_metrics_enhanced(
            placed_points, *anchor_indices, anchor_continuity_pairs,
            anchor_mst2, anchor_max_d2, anchor_graph_score, metric_incumbent,
            PbctopoRefineScorePhase::Exact, PbctopoAnchorBackend::Prim,
            nullptr, anchor_context)) {
      return PbctopoCutCandidate{};
    }
    candidate.anchor_mst2 = anchor_mst2;
    candidate.anchor_max_d2 = anchor_max_d2;
    candidate.anchor_graph = anchor_graph_score;
  }

  candidate.bbox_volume = compute_bbox_volume(stats);
  candidate.rg2 = compute_rg2_from_stats(stats);
  const array<double, 3> principal_rg2 = compute_principal_rg2(stats);
  candidate.principal_rg2_max = principal_rg2[0];
  candidate.principal_rg2_mid = principal_rg2[1];
  candidate.principal_rg2_min = principal_rg2[2];
  const auto shape_metrics = compute_gyration_shape_metrics(principal_rg2);
  candidate.shape_kappa2 = shape_metrics.kappa2;
  candidate.shape_asphericity = shape_metrics.asphericity;
  candidate.shape_acylindricity = shape_metrics.acylindricity;

  for (const auto &edge : local_edges) {
    const Vec3 delta = placed_points[edge.a] - placed_points[edge.b];
    const double d2 = norm2(delta);
    if (d2 > edge.break_cutoff_d2)
      ++candidate.broken_edges;
    candidate.edge_stretch +=
        pbctopo_symmetric_reference_strain(d2, edge.ref_d2);
  }

  candidate.valid = true;
  if (!skip_clash &&
      preclash_cut_candidate_cannot_beat_incumbent(candidate,
                                                   active_incumbent)) {
    return PbctopoCutCandidate{};
  }
  if (!skip_clash)
    analyze_cut_candidate_ultraclash(placed_points, edge_set, candidate,
                                     clash_cutoff, &atoms,
                                     atom_component_index);
  return candidate;
}

[[maybe_unused]] vector<PbctopoCutOption>
collect_cut_options(const vector<PbctopoAtom> &atoms, char axis,
                    size_t max_options) {
  vector<double> coords;
  coords.reserve(atoms.size() * 2 + 2);
  coords.push_back(0.5);
  coords.push_back(2.5);
  for (const auto &atom : atoms) {
    const double coord =
        (axis == 'x') ? atom.sx : ((axis == 'y') ? atom.sy : atom.sz);
    coords.push_back(coord + 0.5);
    coords.push_back(coord + 1.5);
  }
  if (coords.empty())
    return {{1.5, 2.0}};

  sort(coords.begin(), coords.end());
  vector<PbctopoCutOption> options;
  options.reserve(coords.size());
  for (size_t i = 0; i + 1 < coords.size(); ++i) {
    const double current = coords[i];
    const double next = coords[i + 1];
    const double gap = next - current;
    if (gap <= 1.0e-9)
      continue;
    const double center = 0.5 * (current + next);
    if (center + 1.0e-9 < 0.5 || center - 1.0e-9 > 2.5)
      continue;
    options.push_back({center, gap});
  }

  sort(options.begin(), options.end(),
       [](const PbctopoCutOption &lhs, const PbctopoCutOption &rhs) {
          const int metric_cmp =
              compare_pbctopo_double_metric(lhs.gap, rhs.gap, 1.0e-9);
          if (metric_cmp != 0)
            return metric_cmp > 0;
         return lhs.cut < rhs.cut;
       });

  vector<PbctopoCutOption> best;
  best.reserve(min(max_options, options.size() + 1));
  auto push_unique = [&](double cut, double gap) {
    for (const auto &item : best) {
      if (fabs(item.cut - cut) <= 1.0e-6)
        return;
    }
    best.push_back({cut, gap});
  };

  for (const auto &option : options) {
    push_unique(option.cut, option.gap);
    if (best.size() >= max_options)
      break;
  }
  push_unique(1.5, 0.0);
  return best;
}

[[maybe_unused]] void
build_cut_positions(const vector<PbctopoAtom> &atoms, box Box, double cut_x,
                    double cut_y, double cut_z, vector<Vec3> &placed_points,
                    PointStats &stats, double &shift_norm2) {
  auto select_supercell_coord = [](double wrapped_coord, double center,
                                   double &selected_coord, int &selected_shift) {
    const double lo = center - 0.5;
    const double hi = center + 0.5;
    constexpr double kSelectEps = 1.0e-9;
    for (int copy_idx = 0; copy_idx < 3; ++copy_idx) {
      const double candidate = wrapped_coord + static_cast<double>(copy_idx);
      if (candidate + kSelectEps < lo)
        continue;
      if (candidate >= hi - kSelectEps)
        continue;
      selected_coord = candidate;
      selected_shift = copy_idx - 1;
      return;
    }

    const int copy_idx =
        max(0, min(2, static_cast<int>(ceil(lo - wrapped_coord - kSelectEps))));
    selected_coord = wrapped_coord + static_cast<double>(copy_idx);
    selected_shift = copy_idx - 1;
  };

  placed_points.resize(atoms.size());
  stats = PointStats{};
  shift_norm2 = 0.0;
  for (size_t idx = 0; idx < atoms.size(); ++idx) {
    const auto &atom = atoms[idx];
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    int shift_x = 0;
    int shift_y = 0;
    int shift_z = 0;
    select_supercell_coord(atom.sx, cut_x, sx, shift_x);
    select_supercell_coord(atom.sy, cut_y, sy, shift_y);
    select_supercell_coord(atom.sz, cut_z, sz, shift_z);
    Vec3 point{};
    scaled_to_cartesian(Box, sx, sy, sz, point.x, point.y, point.z);
    placed_points[idx] = point;
    stats.add(point.x, point.y, point.z);

    shift_norm2 += static_cast<double>(shift_x * shift_x + shift_y * shift_y +
                                       shift_z * shift_z);
  }
}

PbctopoCutCandidate score_cut_candidate(const vector<PbctopoLocalEdge> &local_edges,
                                        const vector<Vec3> &placed_points,
                                        double cut_x, double cut_y, double cut_z,
                                        double gap_score, const PointStats &stats,
                                        double shift_norm2,
                                        const PbctopoStage1EdgeContext *edge_context,
                                        const PbctopoEdgeScanSoA *edge_scan_soa,
                                        const PbctopoCutCandidate *edge_prune_threshold,
                                        const PbctopoCutCandidate *pre_edge_seed,
                                        PbctopoRefineScorePhase phase) {
  PbctopoCutCandidate candidate =
      (pre_edge_seed != nullptr) ? *pre_edge_seed : PbctopoCutCandidate{};
  candidate.cut_x = cut_x;
  candidate.cut_y = cut_y;
  candidate.cut_z = cut_z;
  candidate.gap_score = gap_score;
  candidate.broken_edges = 0;
  candidate.edge_stretch = 0.0;
  candidate.valid = false;

  if (stats.count == 0 || placed_points.empty())
    return candidate;

  candidate.bbox_volume = compute_bbox_volume(stats);
  candidate.rg2 = compute_rg2_from_stats(stats);
  const array<double, 3> principal_rg2 = compute_principal_rg2(stats);
  candidate.principal_rg2_max = principal_rg2[0];
  candidate.principal_rg2_mid = principal_rg2[1];
  candidate.principal_rg2_min = principal_rg2[2];
  const auto shape_metrics = compute_gyration_shape_metrics(principal_rg2);
  candidate.shape_kappa2 = shape_metrics.kappa2;
  candidate.shape_asphericity = shape_metrics.asphericity;
  candidate.shape_acylindricity = shape_metrics.acylindricity;
  candidate.shift_norm2 = shift_norm2;

  candidate.broken_edges = (edge_context != nullptr)
                               ? edge_context->fixed_broken_edges
                               : 0;
  candidate.edge_stretch = (edge_context != nullptr)
                               ? edge_context->fixed_edge_stretch
                               : 0.0;

  if (!accumulate_edge_scan(candidate, local_edges, placed_points, edge_context,
                            edge_scan_soa, edge_prune_threshold, phase)) {
    return {};
  }

  candidate.valid = true;
  return candidate;
}

PbctopoCutCandidate score_cut_candidate_principal_only(
    const vector<PbctopoLocalEdge> &local_edges,
    const vector<Vec3> &placed_points, double cut_x, double cut_y, double cut_z,
    double gap_score, const PointStats &stats, double shift_norm2,
    PbctopoCutScoreBreakdown *breakdown,
    const PbctopoStage1EdgeContext *edge_context,
    const PbctopoEdgeScanSoA *edge_scan_soa,
    const PbctopoCutCandidate *edge_prune_threshold,
    const PbctopoCutCandidate *pre_edge_seed) {
  PbctopoCutCandidate candidate =
      (pre_edge_seed != nullptr) ? *pre_edge_seed : PbctopoCutCandidate{};
  candidate.cut_x = cut_x;
  candidate.cut_y = cut_y;
  candidate.cut_z = cut_z;
  candidate.gap_score = gap_score;
  candidate.broken_edges = 0;
  candidate.edge_stretch = 0.0;
  candidate.valid = false;

  if (stats.count == 0 || placed_points.empty())
    return candidate;

  chrono::steady_clock::time_point section_started;
  if (breakdown != nullptr)
    section_started = chrono::steady_clock::now();
  const array<double, 3> principal_rg2 = compute_principal_rg2(stats);
  if (breakdown != nullptr) {
    breakdown->principal_s +=
        chrono::duration<double>(chrono::steady_clock::now() - section_started)
            .count();
    section_started = chrono::steady_clock::now();
  }
  candidate.principal_rg2_max = principal_rg2[0];
  candidate.principal_rg2_mid = principal_rg2[1];
  candidate.principal_rg2_min = principal_rg2[2];
  const auto shape_metrics = compute_gyration_shape_metrics(principal_rg2);
  candidate.shape_kappa2 = shape_metrics.kappa2;
  candidate.shape_asphericity = shape_metrics.asphericity;
  candidate.shape_acylindricity = shape_metrics.acylindricity;
  candidate.shift_norm2 = shift_norm2;

  candidate.broken_edges = (edge_context != nullptr)
                               ? edge_context->fixed_broken_edges
                               : 0;
  candidate.edge_stretch = (edge_context != nullptr)
                               ? edge_context->fixed_edge_stretch
                               : 0.0;

  if (!accumulate_edge_scan(candidate, local_edges, placed_points, edge_context,
                            edge_scan_soa, edge_prune_threshold,
                            PbctopoRefineScorePhase::CoarseStage1)) {
    if (breakdown != nullptr) {
      breakdown->edge_scan_s +=
          chrono::duration<double>(chrono::steady_clock::now() -
                                   section_started)
              .count();
    }
    return {};
  }
  if (breakdown != nullptr) {
    breakdown->edge_scan_s +=
        chrono::duration<double>(chrono::steady_clock::now() - section_started)
            .count();
  }

  candidate.valid = true;
  return candidate;
}


void analyze_cut_candidate_ultraclash(const vector<Vec3> &placed_points,
                                       const unordered_set<uint64_t> &edge_set,
                                       PbctopoCutCandidate &candidate,
                                       double cutoff,
                                       const vector<PbctopoAtom> *atoms,
                                       const vector<size_t> *atom_component_index) {
  thread_local PbctopoUltraclashScratch scratch;
  candidate.min_pair_d2 = numeric_limits<double>::infinity();
  candidate.clash_atom_a = numeric_limits<size_t>::max();
  candidate.clash_atom_b = numeric_limits<size_t>::max();
  candidate.ultra_clash = false;
  if (!candidate.valid || placed_points.size() < 2)
    return;

  const bool filter_heavy =
      atoms != nullptr && atoms->size() == placed_points.size();
  const double grid_cutoff =
      filter_heavy ? pbctopo_max_clash_cutoff(*atoms, cutoff) : cutoff;
  const double inv_grid = 1.0 / grid_cutoff;
  const double cutoff2 = cutoff * cutoff;
  const bool filter_same_component =
      atom_component_index != nullptr &&
      atom_component_index->size() == placed_points.size() &&
      candidate.broken_edges == 0;
  auto accumulate_pair = [&](size_t atom_idx, size_t other_idx) {
    if (filter_same_component &&
        (*atom_component_index)[atom_idx] ==
            (*atom_component_index)[other_idx])
      return false;
    if (edge_set.find(atom_pair_key(atom_idx, other_idx)) != edge_set.end())
      return false;
    const auto &point = placed_points[atom_idx];
    const auto &other = placed_points[other_idx];
    const double dx_atom = point.x - other.x;
    const double dy_atom = point.y - other.y;
    const double dz_atom = point.z - other.z;
    const double d2 = dx_atom * dx_atom + dy_atom * dy_atom + dz_atom * dz_atom;
    if (d2 < candidate.min_pair_d2) {
      candidate.min_pair_d2 = d2;
      candidate.clash_atom_a = atom_idx;
      candidate.clash_atom_b = other_idx;
    }
    const bool clashes =
        filter_heavy
            ? pbctopo_atoms_clash(d2, (*atoms)[atom_idx],
                                   (*atoms)[other_idx], cutoff)
            : d2 + kDistanceEps < cutoff2;
    if (clashes) {
      candidate.ultra_clash = true;
      return true;
    }
    return false;
  };

  auto analyze_with_hash_grid = [&]() {
    unordered_map<ClashCellKey, vector<size_t>, ClashCellKeyHash> clash_grid;
    clash_grid.reserve(max<size_t>(64, placed_points.size()));
    for (size_t atom_idx = 0; atom_idx < placed_points.size(); ++atom_idx) {
      if (filter_heavy && !(*atoms)[atom_idx].is_heavy)
        continue;
      const auto &point = placed_points[atom_idx];
      ClashCellKey own_cell;
      if (!titan_pbctopo::pbctopo_make_cell_key(
              point.x, point.y, point.z, inv_grid, own_cell)) {
        candidate.ultra_clash = true;
        return;
      }

      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            ClashCellKey neighbor;
            if (!titan_pbctopo::pbctopo_offset_cell_key(
                    own_cell, dx, dy, dz, neighbor)) {
              candidate.ultra_clash = true;
              return;
            }
            const auto it = clash_grid.find(neighbor);
            if (it == clash_grid.end())
              continue;
            for (size_t other_idx : it->second) {
              if (accumulate_pair(atom_idx, other_idx))
                return;
            }
          }
        }
      }

      clash_grid[own_cell].push_back(atom_idx);
    }
  };

  scratch.cell_x.resize(placed_points.size());
  scratch.cell_y.resize(placed_points.size());
  scratch.cell_z.resize(placed_points.size());
  int64_t min_cx = numeric_limits<int64_t>::max();
  int64_t min_cy = numeric_limits<int64_t>::max();
  int64_t min_cz = numeric_limits<int64_t>::max();
  int64_t max_cx = numeric_limits<int64_t>::min();
  int64_t max_cy = numeric_limits<int64_t>::min();
  int64_t max_cz = numeric_limits<int64_t>::min();

  for (size_t atom_idx = 0; atom_idx < placed_points.size(); ++atom_idx) {
    if (filter_heavy && !(*atoms)[atom_idx].is_heavy)
      continue;
    const auto &point = placed_points[atom_idx];
    ClashCellKey cell;
    if (!titan_pbctopo::pbctopo_make_cell_key(
            point.x, point.y, point.z, inv_grid, cell)) {
      candidate.ultra_clash = true;
      return;
    }
    scratch.cell_x[atom_idx] = cell.x;
    scratch.cell_y[atom_idx] = cell.y;
    scratch.cell_z[atom_idx] = cell.z;
    min_cx = min(min_cx, cell.x);
    min_cy = min(min_cy, cell.y);
    min_cz = min(min_cz, cell.z);
    max_cx = max(max_cx, cell.x);
    max_cy = max(max_cy, cell.y);
    max_cz = max(max_cz, cell.z);
  }
  if (min_cx == numeric_limits<int64_t>::max())
    return;

  const size_t max_cells = kPbctopoDenseClashGridMaxCells;
  const auto bounded_span = [](int64_t lower, int64_t upper, size_t &span) {
    const long double width = static_cast<long double>(upper) -
                              static_cast<long double>(lower) + 1.0L;
    if (!(width >= 1.0L) || width > static_cast<long double>(max_cells))
      return false;
    span = static_cast<size_t>(width);
    return true;
  };
  size_t nx = 0;
  size_t ny = 0;
  size_t nz = 0;
  if (!bounded_span(min_cx, max_cx, nx) ||
      !bounded_span(min_cy, max_cy, ny) ||
      !bounded_span(min_cz, max_cz, nz) || ny > (max_cells / nx) ||
      nz > (max_cells / (nx * ny))) {
    analyze_with_hash_grid();
    return;
  }

  const size_t total_cells = nx * ny * nz;
  scratch.head.resize(total_cells);
  scratch.head_epoch.resize(total_cells, 0u);
  ++scratch.current_epoch;
  if (scratch.current_epoch == 0u) {
    scratch.current_epoch = 1u;
    fill(scratch.head_epoch.begin(), scratch.head_epoch.end(), 0u);
  }
  scratch.next.resize(placed_points.size());
  const size_t plane_stride = nx * ny;
  auto cell_index = [&](int64_t cx, int64_t cy, int64_t cz) {
    return (static_cast<size_t>(static_cast<long double>(cz) - min_cz) *
            plane_stride) +
           (static_cast<size_t>(static_cast<long double>(cy) - min_cy) * nx) +
           static_cast<size_t>(static_cast<long double>(cx) - min_cx);
  };

  for (size_t atom_idx = 0; atom_idx < placed_points.size(); ++atom_idx) {
    if (filter_heavy && !(*atoms)[atom_idx].is_heavy)
      continue;
    const int64_t cx = scratch.cell_x[atom_idx];
    const int64_t cy = scratch.cell_y[atom_idx];
    const int64_t cz = scratch.cell_z[atom_idx];
    for (int64_t nz_cell = max(min_cz, cz - 1);
         nz_cell <= min(max_cz, cz + 1);
         ++nz_cell) {
      for (int64_t ny_cell = max(min_cy, cy - 1);
           ny_cell <= min(max_cy, cy + 1);
           ++ny_cell) {
        for (int64_t nx_cell = max(min_cx, cx - 1);
             nx_cell <= min(max_cx, cx + 1);
             ++nx_cell) {
          const size_t neighbor_cell_idx = cell_index(nx_cell, ny_cell, nz_cell);
          int other_atom =
              (scratch.head_epoch[neighbor_cell_idx] == scratch.current_epoch)
                  ? scratch.head[neighbor_cell_idx]
                  : -1;
          while (other_atom >= 0) {
            if (accumulate_pair(atom_idx, static_cast<size_t>(other_atom)))
              return;
            other_atom = scratch.next[static_cast<size_t>(other_atom)];
          }
        }
      }
    }

    const size_t own_cell_idx = cell_index(cx, cy, cz);
    scratch.next[atom_idx] =
        (scratch.head_epoch[own_cell_idx] == scratch.current_epoch)
            ? scratch.head[own_cell_idx]
            : -1;
    scratch.head[own_cell_idx] = static_cast<int>(atom_idx);
    scratch.head_epoch[own_cell_idx] = scratch.current_epoch;
  }
}

void apply_cut_candidate(const vector<PbctopoAtom> &atoms,
                         const vector<Vec3> &placed_points,
                         Multi_MOL &mol1_system, Multi_MOL &mol2_system,
                         size_t frame_index) {
  if (atoms.empty() || placed_points.size() != atoms.size())
    return;

  box Box = mol1_system.m_mol[frame_index].Box;
  const PbctopoLatticeMetric lattice(Box);
  double min_sx = numeric_limits<double>::infinity();
  double min_sy = numeric_limits<double>::infinity();
  double min_sz = numeric_limits<double>::infinity();
  double max_sx = -numeric_limits<double>::infinity();
  double max_sy = -numeric_limits<double>::infinity();
  double max_sz = -numeric_limits<double>::infinity();

  for (const auto &point : placed_points) {
    const auto scaled = lattice.cartesian_to_scaled(point.x, point.y, point.z);
    const double sx = scaled[0];
    const double sy = scaled[1];
    const double sz = scaled[2];
    min_sx = min(min_sx, sx);
    min_sy = min(min_sy, sy);
    min_sz = min(min_sz, sz);
    max_sx = max(max_sx, sx);
    max_sy = max(max_sy, sy);
    max_sz = max(max_sz, sz);
  }

  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  const Vec3 recenter_shift{
      (0.5 - 0.5 * (min_sx + max_sx)) * v1[0] +
          (0.5 - 0.5 * (min_sy + max_sy)) * v2[0] +
          (0.5 - 0.5 * (min_sz + max_sz)) * v3[0],
      (0.5 - 0.5 * (min_sx + max_sx)) * v1[1] +
          (0.5 - 0.5 * (min_sy + max_sy)) * v2[1] +
          (0.5 - 0.5 * (min_sz + max_sz)) * v3[1],
      (0.5 - 0.5 * (min_sx + max_sx)) * v1[2] +
          (0.5 - 0.5 * (min_sy + max_sy)) * v2[2] +
          (0.5 - 0.5 * (min_sz + max_sz)) * v3[2]};

  for (size_t idx = 0; idx < atoms.size(); ++idx) {
    const auto &atom = atoms[idx];
    const Vec3 point{placed_points[idx].x + recenter_shift.x,
                     placed_points[idx].y + recenter_shift.y,
                     placed_points[idx].z + recenter_shift.z};
    if (atom.owner == 0) {
      mol1_system.m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
      mol1_system.m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
      mol1_system.m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
    } else if (frame_index < mol2_system.m_mol.size()) {
      mol2_system.m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
      mol2_system.m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
      mol2_system.m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
    }
  }
}

void apply_pbctopo_points_direct(const vector<PbctopoAtom> &atoms,
                                 const vector<Vec3> &placed_points,
                                 Multi_MOL &mol1_system,
                                 Multi_MOL &mol2_system,
                                 size_t frame_index) {
  if (atoms.empty() || placed_points.size() != atoms.size())
    return;

  for (size_t idx = 0; idx < atoms.size(); ++idx) {
    const auto &atom = atoms[idx];
    const Vec3 &point = placed_points[idx];
    if (atom.owner == 0) {
      mol1_system.m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
      mol1_system.m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
      mol1_system.m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
    } else if (frame_index < mol2_system.m_mol.size()) {
      mol2_system.m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
      mol2_system.m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
      mol2_system.m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
    }
  }
}

[[maybe_unused]] void
append_fragments_for_system(const Multi_MOL &system, const MOL &mol, int owner,
                            vector<PbctopoFragment> &fragments,
                            vector<PbctopoAtom> &atoms) {
  const auto ranges = build_ranges_for_system(system, mol);
  for (const auto &range : ranges) {
    PbctopoFragment fragment;
    fragment.id = fragments.size();
    fragment.owner = owner;
    fragment.source_index = range.source_index;
    fragment.begin_local = range.begin;
    fragment.end_local = range.end;
    fragment.base_atom_begin = atoms.size();
    fragment.atom_count = range.end - range.begin;
    fragment.topo_backed = range.topo_backed;

    for (size_t local = range.begin; local < range.end; ++local) {
      double r[3]{mol.pos.x[local], mol.pos.y[local], mol.pos.z[local]};
      double s[3]{};
      mol.Box.to_scaled(r, s);
      for (double &coord : s)
        coord -= floor(coord);
      scaled_to_cartesian(mol.Box, s[0], s[1], s[2], r[0], r[1], r[2]);
      PbctopoAtom atom;
      atom.global_atom_id = atoms.size();
      atom.fragment_id = fragment.id;
      atom.owner = owner;
      atom.local_atom_index = local;
      atom.source_atom_id =
          local < system.info.size() &&
                  system.info[local].has_original_atom_id()
              ? system.info[local].get_original_atom_id()
              : local;
      if (const auto mass = system.topology.atomic_mass(static_cast<std::uint32_t>(local));
          mass.has_value()) {
        atom.atomic_mass = *mass;
        atom.has_atomic_mass = true;
      }
      atom.x = r[0];
      atom.y = r[1];
      atom.z = r[2];
      atom.sx = s[0];
      atom.sy = s[1];
      atom.sz = s[2];
      atoms.push_back(atom);
    }
    fragment.base_atom_end = atoms.size();
    fragments.push_back(fragment);
  }
}

vector<PbctopoShift> build_topology_shifts(box Box, int shell_radius = 1) {
  return titan_pbctopo::build_component_image_shifts(Box, shell_radius);
}

[[maybe_unused]] void
build_supercell_for_frame(const vector<PbctopoFragment> &fragments,
                          const vector<PbctopoAtom> &base_atoms,
                          const vector<PbctopoShift> &shifts,
                          vector<PbctopoFragmentCopyNode> &nodes,
                          vector<PbctopoAtomCopy> &atom_copies) {
  nodes.clear();
  atom_copies.clear();
  nodes.reserve(fragments.size() * shifts.size());
  atom_copies.reserve(base_atoms.size() * shifts.size());

  for (const auto &fragment : fragments) {
    for (const auto &shift : shifts) {
      PbctopoFragmentCopyNode node;
      node.node_id = nodes.size();
      node.fragment_id = fragment.id;
      node.shift_index = shift.shift_index;
      node.shift_ix = shift.ix;
      node.shift_iy = shift.iy;
      node.shift_iz = shift.iz;
      node.tx = shift.tx;
      node.ty = shift.ty;
      node.tz = shift.tz;
      node.atom_begin = atom_copies.size();
      node.min_x = numeric_limits<double>::infinity();
      node.min_y = numeric_limits<double>::infinity();
      node.min_z = numeric_limits<double>::infinity();
      node.max_x = -numeric_limits<double>::infinity();
      node.max_y = -numeric_limits<double>::infinity();
      node.max_z = -numeric_limits<double>::infinity();

      for (size_t atom_idx = fragment.base_atom_begin;
           atom_idx < fragment.base_atom_end; ++atom_idx) {
        const PbctopoAtom &base_atom = base_atoms[atom_idx];
        PbctopoAtomCopy atom_copy;
        atom_copy.global_atom_id = base_atom.global_atom_id;
        atom_copy.fragment_id = fragment.id;
        atom_copy.node_id = node.node_id;
        atom_copy.owner = base_atom.owner;
        atom_copy.local_atom_index = base_atom.local_atom_index;
        atom_copy.x = base_atom.x + shift.tx;
        atom_copy.y = base_atom.y + shift.ty;
        atom_copy.z = base_atom.z + shift.tz;
        node.min_x = min(node.min_x, atom_copy.x);
        node.min_y = min(node.min_y, atom_copy.y);
        node.min_z = min(node.min_z, atom_copy.z);
        node.max_x = max(node.max_x, atom_copy.x);
        node.max_y = max(node.max_y, atom_copy.y);
        node.max_z = max(node.max_z, atom_copy.z);
        atom_copies.push_back(atom_copy);
      }

      node.atom_end = atom_copies.size();
      nodes.push_back(node);
    }
  }
}

bool are_topology_neighbors(const PbctopoFragment &lhs,
                            const PbctopoFragment &rhs) {
  if (!lhs.topo_backed || !rhs.topo_backed || lhs.owner != rhs.owner)
    return false;
  if (lhs.source_index + 1 == rhs.source_index ||
      rhs.source_index + 1 == lhs.source_index)
    return true;
  return false;
}

double axis_gap(double min_a, double max_a, double min_b, double max_b) {
  if (max_a < min_b)
    return min_b - max_a;
  if (max_b < min_a)
    return min_a - max_b;
  return 0.0;
}

double bbox_distance2(const PbctopoFragmentCopyNode &lhs,
                      const PbctopoFragmentCopyNode &rhs) {
  const double dx = axis_gap(lhs.min_x, lhs.max_x, rhs.min_x, rhs.max_x);
  const double dy = axis_gap(lhs.min_y, lhs.max_y, rhs.min_y, rhs.max_y);
  const double dz = axis_gap(lhs.min_z, lhs.max_z, rhs.min_z, rhs.max_z);
  return dx * dx + dy * dy + dz * dz;
}

[[maybe_unused]] vector<vector<size_t>>
build_fragment_copy_graph(const vector<PbctopoFragment> &fragments,
                          const vector<PbctopoFragmentCopyNode> &nodes,
                          double geometry_cutoff, double topology_cutoff) {
  vector<vector<size_t>> adjacency(nodes.size());
  const double geometry_cutoff2 = geometry_cutoff * geometry_cutoff;
  const double topology_cutoff2 = topology_cutoff * topology_cutoff;
  const double max_cutoff = max(geometry_cutoff, topology_cutoff);
  const double inv_cell = 1.0 / max_cutoff;
  unordered_map<ClashCellKey, vector<size_t>, ClashCellKeyHash> cell_nodes;
  cell_nodes.reserve(max<size_t>(64, nodes.size() * 2));

  for (size_t node_id = 0; node_id < nodes.size(); ++node_id) {
    const auto &node = nodes[node_id];
    const int min_cx = static_cast<int>(floor((node.min_x - max_cutoff) * inv_cell));
    const int min_cy = static_cast<int>(floor((node.min_y - max_cutoff) * inv_cell));
    const int min_cz = static_cast<int>(floor((node.min_z - max_cutoff) * inv_cell));
    const int max_cx = static_cast<int>(floor((node.max_x + max_cutoff) * inv_cell));
    const int max_cy = static_cast<int>(floor((node.max_y + max_cutoff) * inv_cell));
    const int max_cz = static_cast<int>(floor((node.max_z + max_cutoff) * inv_cell));

    for (int cx = min_cx; cx <= max_cx; ++cx) {
      for (int cy = min_cy; cy <= max_cy; ++cy) {
        for (int cz = min_cz; cz <= max_cz; ++cz)
          cell_nodes[ClashCellKey{cx, cy, cz}].push_back(node_id);
      }
    }
  }

  unordered_set<uint64_t> seen_pairs;
  seen_pairs.reserve(max<size_t>(128, nodes.size() * 8));
  for (const auto &cell_entry : cell_nodes) {
    const auto &cell_list = cell_entry.second;
    for (size_t a = 0; a < cell_list.size(); ++a) {
      for (size_t b = a + 1; b < cell_list.size(); ++b) {
        size_t lhs_id = cell_list[a];
        size_t rhs_id = cell_list[b];
        if (lhs_id == rhs_id)
          continue;
        if (lhs_id > rhs_id)
          swap(lhs_id, rhs_id);
        const uint64_t pair_key =
            (static_cast<uint64_t>(lhs_id) << 32) | static_cast<uint64_t>(rhs_id);
        if (!seen_pairs.insert(pair_key).second)
          continue;

        const auto &lhs = nodes[lhs_id];
        const auto &rhs = nodes[rhs_id];
        if (lhs.fragment_id == rhs.fragment_id)
          continue;

        const bool topo_related =
            are_topology_neighbors(fragments[lhs.fragment_id],
                                   fragments[rhs.fragment_id]);
        const double cutoff2 = topo_related ? topology_cutoff2 : geometry_cutoff2;
        if (bbox_distance2(lhs, rhs) > cutoff2)
          continue;

        adjacency[lhs_id].push_back(rhs_id);
        adjacency[rhs_id].push_back(lhs_id);
      }
    }
  }
  return adjacency;
}

double node_shift_norm2(const PbctopoFragmentCopyNode &node) {
  return static_cast<double>(node.shift_ix * node.shift_ix +
                             node.shift_iy * node.shift_iy +
                             node.shift_iz * node.shift_iz);
}

[[maybe_unused]] void
sort_fragment_copy_graph(vector<vector<size_t>> &adjacency,
                         const vector<PbctopoFragmentCopyNode> &nodes) {
  for (size_t node_id = 0; node_id < adjacency.size(); ++node_id) {
    auto &neighbors = adjacency[node_id];
    sort(neighbors.begin(), neighbors.end(),
         [&](size_t lhs, size_t rhs) {
           const double lhs_d2 = bbox_distance2(nodes[node_id], nodes[lhs]);
           const double rhs_d2 = bbox_distance2(nodes[node_id], nodes[rhs]);
           if (fabs(lhs_d2 - rhs_d2) > 1.0e-9)
             return lhs_d2 < rhs_d2;
           const double lhs_shift = node_shift_norm2(nodes[lhs]);
           const double rhs_shift = node_shift_norm2(nodes[rhs]);
           if (fabs(lhs_shift - rhs_shift) > 1.0e-9)
             return lhs_shift < rhs_shift;
           return lhs < rhs;
         });
  }
}

double min_bbox_distance2_to_selected(
    size_t node_id, const vector<size_t> &selected_nodes,
    const vector<PbctopoFragmentCopyNode> &nodes) {
  double best_d2 = numeric_limits<double>::infinity();
  for (size_t selected_id : selected_nodes)
    best_d2 = min(best_d2, bbox_distance2(nodes[node_id], nodes[selected_id]));
  return best_d2;
}

[[maybe_unused]] bool
build_candidate_block_from_root(size_t root_node,
                                const vector<PbctopoFragment> &fragments,
                                const vector<PbctopoFragmentCopyNode> &nodes,
                                const vector<vector<size_t>> &adjacency,
                                size_t total_atom_count,
                                vector<size_t> &component_nodes) {
  component_nodes.clear();
  if (root_node >= nodes.size())
    return false;

  vector<int> fragment_assignment(fragments.size(), -1);
  const size_t root_fragment = nodes[root_node].fragment_id;
  fragment_assignment[root_fragment] = static_cast<int>(root_node);
  component_nodes.push_back(root_node);

  size_t component_atom_count = fragments[root_fragment].atom_count;
  if (component_atom_count > total_atom_count)
    return false;

  while (component_nodes.size() < fragments.size()) {
    size_t next_fragment = numeric_limits<size_t>::max();
    size_t next_node = numeric_limits<size_t>::max();
    double next_d2 = numeric_limits<double>::infinity();
    double next_shift = numeric_limits<double>::infinity();

    for (size_t fragment_id = 0; fragment_id < fragments.size(); ++fragment_id) {
      if (fragment_assignment[fragment_id] != -1)
        continue;
      vector<size_t> options =
          collect_fragment_options(fragment_id, component_nodes, nodes, adjacency, 1);
      if (options.empty())
        continue;
      const size_t candidate_node = options.front();
      const double candidate_d2 =
          min_bbox_distance2_to_selected(candidate_node, component_nodes, nodes);
      const double candidate_shift = node_shift_norm2(nodes[candidate_node]);
      if (next_fragment == numeric_limits<size_t>::max() ||
          candidate_d2 < next_d2 - 1.0e-9 ||
          (fabs(candidate_d2 - next_d2) <= 1.0e-9 &&
           candidate_shift < next_shift - 1.0e-9) ||
          (fabs(candidate_d2 - next_d2) <= 1.0e-9 &&
           fabs(candidate_shift - next_shift) <= 1.0e-9 &&
           candidate_node < next_node)) {
        next_fragment = fragment_id;
        next_node = candidate_node;
        next_d2 = candidate_d2;
        next_shift = candidate_shift;
      }
    }

    if (next_fragment == numeric_limits<size_t>::max())
      return false;

    fragment_assignment[next_fragment] = static_cast<int>(next_node);
    component_nodes.push_back(next_node);
    component_atom_count += fragments[next_fragment].atom_count;
    if (component_atom_count > total_atom_count)
      return false;
  }

  if (component_atom_count != total_atom_count ||
      component_nodes.size() != fragments.size()) {
    return false;
  }

  sort(component_nodes.begin(), component_nodes.end());
  return true;
}

vector<size_t> collect_fragment_options(
    size_t fragment_id, const vector<size_t> &selected_nodes,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t max_options) {
  unordered_map<size_t, double> candidate_dist2;
  candidate_dist2.reserve(16);
  for (size_t selected_node : selected_nodes) {
    for (size_t neighbor : adjacency[selected_node]) {
      if (nodes[neighbor].fragment_id != fragment_id)
        continue;
      const double d2 = bbox_distance2(nodes[selected_node], nodes[neighbor]);
      const auto it = candidate_dist2.find(neighbor);
      if (it == candidate_dist2.end() || d2 < it->second)
        candidate_dist2[neighbor] = d2;
    }
  }

  vector<pair<size_t, double>> ranked(candidate_dist2.begin(),
                                      candidate_dist2.end());
  sort(ranked.begin(), ranked.end(),
       [&](const pair<size_t, double> &lhs, const pair<size_t, double> &rhs) {
         int metric_cmp = compare_pbctopo_double_metric(
             lhs.second, rhs.second, 1.0e-9);
         if (metric_cmp != 0)
           return metric_cmp < 0;
         const double lhs_shift = node_shift_norm2(nodes[lhs.first]);
         const double rhs_shift = node_shift_norm2(nodes[rhs.first]);
         metric_cmp = compare_pbctopo_double_metric(lhs_shift, rhs_shift,
                                                     1.0e-9);
         if (metric_cmp != 0)
           return metric_cmp < 0;
         return lhs.first < rhs.first;
       });

  vector<size_t> options;
  options.reserve(min(max_options, ranked.size()));
  for (size_t i = 0; i < ranked.size() && i < max_options; ++i)
    options.push_back(ranked[i].first);
  return options;
}

[[maybe_unused]] void enumerate_connected_block_candidates_recursive(
    const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t max_options_per_fragment,
    vector<int> &assignment, vector<size_t> &selected_nodes,
    unordered_set<vector<size_t>, NodeVectorHash> &seen_components,
    vector<vector<size_t>> &out_components) {
  if (selected_nodes.size() == fragments.size()) {
    vector<size_t> candidate_nodes = selected_nodes;
    sort(candidate_nodes.begin(), candidate_nodes.end());
    if (seen_components.insert(candidate_nodes).second)
      out_components.push_back(std::move(candidate_nodes));
    return;
  }

  size_t next_fragment = numeric_limits<size_t>::max();
  vector<size_t> next_options;
  for (size_t fragment_id = 0; fragment_id < fragments.size(); ++fragment_id) {
    if (assignment[fragment_id] != -1)
      continue;
    vector<size_t> options =
        collect_fragment_options(fragment_id, selected_nodes, nodes, adjacency,
                                 max_options_per_fragment);
    if (options.empty())
      continue;
    if (next_fragment == numeric_limits<size_t>::max() ||
        options.size() < next_options.size()) {
      next_fragment = fragment_id;
      next_options = std::move(options);
    }
  }

  if (next_fragment == numeric_limits<size_t>::max())
    return;

  for (size_t node_id : next_options) {
    assignment[next_fragment] = static_cast<int>(node_id);
    selected_nodes.push_back(node_id);
    enumerate_connected_block_candidates_recursive(
        fragments, nodes, adjacency, max_options_per_fragment, assignment,
        selected_nodes, seen_components, out_components);
    selected_nodes.pop_back();
    assignment[next_fragment] = -1;
  }
}

[[maybe_unused]] vector<vector<size_t>> enumerate_connected_block_candidates(
    const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t max_options_per_fragment) {
  vector<vector<size_t>> out_components;
  if (fragments.empty())
    return out_components;

  unordered_set<vector<size_t>, NodeVectorHash> seen_components;
  const size_t root_fragment = 0;
  for (size_t node_id = 0; node_id < nodes.size(); ++node_id) {
    if (nodes[node_id].fragment_id != root_fragment)
      continue;
    vector<int> assignment(fragments.size(), -1);
    vector<size_t> selected_nodes{node_id};
    assignment[root_fragment] = static_cast<int>(node_id);
    enumerate_connected_block_candidates_recursive(
        fragments, nodes, adjacency, max_options_per_fragment, assignment,
        selected_nodes, seen_components, out_components);
  }

  return out_components;
}

Vec3 node_bbox_center(const PbctopoFragmentCopyNode &node) {
  return {(node.min_x + node.max_x) * 0.5, (node.min_y + node.max_y) * 0.5,
          (node.min_z + node.max_z) * 0.5};
}

double compute_mst2(const vector<Vec3> &points) {
  if (points.size() < 2)
    return 0.0;

  vector<double> best(points.size(), numeric_limits<double>::infinity());
  vector<char> used(points.size(), 0);
  best[0] = 0.0;
  double mst2 = 0.0;

  for (size_t step = 0; step < points.size(); ++step) {
    size_t next = numeric_limits<size_t>::max();
    double next_cost = numeric_limits<double>::infinity();
    for (size_t i = 0; i < points.size(); ++i) {
      if (used[i] != 0)
        continue;
      if (best[i] < next_cost) {
        next = i;
        next_cost = best[i];
      }
    }
    if (next == numeric_limits<size_t>::max())
      break;
    used[next] = 1;
    mst2 += next_cost;
    for (size_t i = 0; i < points.size(); ++i) {
      if (used[i] != 0)
        continue;
      const double d2 = norm2(points[next] - points[i]);
      if (d2 < best[i])
        best[i] = d2;
    }
  }

  return mst2;
}

size_t count_component_connectivity_edges(
    const vector<size_t> &component_nodes,
    const vector<vector<size_t>> &adjacency) {
  if (component_nodes.empty() || adjacency.empty())
    return 0;
  vector<char> node_mask(adjacency.size(), 0);
  for (size_t node_id : component_nodes) {
    if (node_id < node_mask.size())
      node_mask[node_id] = 1;
  }
  size_t edge_count = 0;
  for (size_t node_id : component_nodes) {
    if (node_id >= adjacency.size())
      continue;
    for (size_t neighbor : adjacency[node_id]) {
      if (neighbor >= node_mask.size() || node_mask[neighbor] == 0 ||
          node_id >= neighbor) {
        continue;
      }
      ++edge_count;
    }
  }
  return edge_count;
}

PbctopoCandidate
score_coarse_node_combination(const vector<size_t> &component_nodes,
                              const vector<PbctopoFragmentCopyNode> &nodes,
                              const vector<vector<size_t>> &adjacency) {
  PbctopoCandidate candidate;
  candidate.node_ids = component_nodes;
  if (component_nodes.empty())
    return candidate;

  vector<Vec3> centers;
  centers.reserve(component_nodes.size());
  double min_x = numeric_limits<double>::infinity();
  double min_y = numeric_limits<double>::infinity();
  double min_z = numeric_limits<double>::infinity();
  double max_x = -numeric_limits<double>::infinity();
  double max_y = -numeric_limits<double>::infinity();
  double max_z = -numeric_limits<double>::infinity();
  candidate.shift_norm2 = 0.0;

  for (size_t node_id : component_nodes) {
    const auto &node = nodes[node_id];
    min_x = min(min_x, node.min_x);
    min_y = min(min_y, node.min_y);
    min_z = min(min_z, node.min_z);
    max_x = max(max_x, node.max_x);
    max_y = max(max_y, node.max_y);
    max_z = max(max_z, node.max_z);
    centers.push_back(node_bbox_center(node));
    candidate.shift_norm2 += node_shift_norm2(node);
  }

  candidate.bbox_volume =
      max(0.0, max_x - min_x) * max(0.0, max_y - min_y) * max(0.0, max_z - min_z);
  candidate.rg2 = compute_rg2(centers);
  candidate.fragment_rg2 = candidate.rg2;
  const PointStats center_stats = build_point_stats(centers);
  const auto center_principal_rg2 = compute_principal_rg2(center_stats);
  const auto center_shape_metrics =
      compute_gyration_shape_metrics(center_principal_rg2);
  candidate.shape_kappa2 = center_shape_metrics.kappa2;
  candidate.shape_asphericity = center_shape_metrics.asphericity;
  candidate.shape_acylindricity = center_shape_metrics.acylindricity;
  candidate.fragment_kappa2 = center_shape_metrics.kappa2;
  candidate.fragment_asphericity = center_shape_metrics.asphericity;
  candidate.fragment_acylindricity = center_shape_metrics.acylindricity;
  candidate.fragment_mst2 = compute_mst2(centers);
  candidate.connectivity_edges =
      count_component_connectivity_edges(component_nodes, adjacency);
  candidate.valid = true;
  return candidate;
}

bool better_pbctopo_coarse_candidate(const PbctopoCandidate &lhs,
                                     const PbctopoCandidate &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  if (lhs.connectivity_edges != rhs.connectivity_edges)
    return lhs.connectivity_edges > rhs.connectivity_edges;
  int metric_cmp =
      compare_pbctopo_double_metric(lhs.fragment_mst2, rhs.fragment_mst2);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  const int fragment_kappa_cmp =
      compare_shape_kappa(lhs.fragment_kappa2, rhs.fragment_kappa2);
  if (fragment_kappa_cmp != 0)
    return fragment_kappa_cmp < 0;
  const int fragment_asph_cmp = compare_shape_moment(
      lhs.fragment_asphericity, rhs.fragment_asphericity);
  if (fragment_asph_cmp != 0)
    return fragment_asph_cmp < 0;
  const int fragment_acyl_cmp = compare_shape_moment(
      lhs.fragment_acylindricity, rhs.fragment_acylindricity);
  if (fragment_acyl_cmp != 0)
    return fragment_acyl_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.bbox_volume, rhs.bbox_volume);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  const int shape_kappa_cmp =
      compare_shape_kappa(lhs.shape_kappa2, rhs.shape_kappa2);
  if (shape_kappa_cmp != 0)
    return shape_kappa_cmp < 0;
  const int shape_asph_cmp =
      compare_shape_moment(lhs.shape_asphericity, rhs.shape_asphericity);
  if (shape_asph_cmp != 0)
    return shape_asph_cmp < 0;
  const int shape_acyl_cmp =
      compare_shape_moment(lhs.shape_acylindricity, rhs.shape_acylindricity);
  if (shape_acyl_cmp != 0)
    return shape_acyl_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.shift_norm2, rhs.shift_norm2);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  return lhs.node_ids < rhs.node_ids;
}

[[maybe_unused]] void enumerate_small_fragment_shift_combinations_recursive(
    size_t fragment_id, const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, vector<size_t> &selected_nodes,
    vector<PbctopoCandidate> &shortlist, size_t shortlist_limit) {
  if (fragment_id == fragments.size()) {
    PbctopoCandidate candidate =
        score_coarse_node_combination(selected_nodes, nodes, adjacency);
    shortlist.push_back(std::move(candidate));
    if (shortlist.size() > shortlist_limit * 2) {
      sort(shortlist.begin(), shortlist.end(), better_pbctopo_coarse_candidate);
      shortlist.resize(shortlist_limit);
    }
    return;
  }

  const size_t base_node = fragment_id * 27;
  for (size_t shift_idx = 0; shift_idx < 27; ++shift_idx) {
    selected_nodes.push_back(base_node + shift_idx);
    enumerate_small_fragment_shift_combinations_recursive(
        fragment_id + 1, fragments, nodes, adjacency, selected_nodes,
        shortlist, shortlist_limit);
    selected_nodes.pop_back();
  }
}

[[maybe_unused]] vector<vector<size_t>>
enumerate_small_fragment_shift_combinations(
    const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t shortlist_limit) {
  vector<PbctopoCandidate> shortlist;
  vector<size_t> selected_nodes;
  selected_nodes.reserve(fragments.size());
  enumerate_small_fragment_shift_combinations_recursive(
      0, fragments, nodes, adjacency, selected_nodes, shortlist,
      shortlist_limit);
  sort(shortlist.begin(), shortlist.end(), better_pbctopo_coarse_candidate);
  if (shortlist.size() > shortlist_limit)
    shortlist.resize(shortlist_limit);

  vector<vector<size_t>> components;
  components.reserve(shortlist.size());
  for (auto &candidate : shortlist)
    components.push_back(candidate.node_ids);
  return components;
}

[[maybe_unused]] PbctopoCandidate
score_component_candidate(const vector<size_t> &component_nodes,
                          const vector<PbctopoFragmentCopyNode> &nodes,
                          const vector<PbctopoAtomCopy> &atom_copies,
                          const vector<vector<size_t>> &adjacency) {
  PbctopoCandidate candidate;
  candidate.node_ids = component_nodes;
  if (component_nodes.empty())
    return candidate;

  vector<Vec3> centers;
  PointStats stats;
  candidate.shift_norm2 = 0.0;
  centers.reserve(component_nodes.size());
  for (size_t node_id : component_nodes) {
    const auto &node = nodes[node_id];
    candidate.shift_norm2 +=
        static_cast<double>(node.shift_ix * node.shift_ix + node.shift_iy * node.shift_iy +
                            node.shift_iz * node.shift_iz) *
        static_cast<double>(node.atom_end - node.atom_begin);
    centers.push_back(node_bbox_center(node));
    for (size_t atom_idx = node.atom_begin; atom_idx < node.atom_end; ++atom_idx) {
      const auto &atom = atom_copies[atom_idx];
      stats.add(atom.x, atom.y, atom.z);
    }
  }

  candidate.hull_valid = false;
  candidate.hull_volume = numeric_limits<double>::infinity();
  if (stats.count > 0) {
    candidate.bbox_volume = compute_bbox_volume(stats);
    candidate.rg2 = compute_rg2_from_stats(stats);
    const auto principal_rg2 = compute_principal_rg2(stats);
    const auto shape_metrics = compute_gyration_shape_metrics(principal_rg2);
    candidate.shape_kappa2 = shape_metrics.kappa2;
    candidate.shape_asphericity = shape_metrics.asphericity;
    candidate.shape_acylindricity = shape_metrics.acylindricity;
  }
  candidate.fragment_rg2 = compute_rg2(centers);
  const PointStats center_stats = build_point_stats(centers);
  const auto center_principal_rg2 = compute_principal_rg2(center_stats);
  const auto center_shape_metrics =
      compute_gyration_shape_metrics(center_principal_rg2);
  candidate.fragment_kappa2 = center_shape_metrics.kappa2;
  candidate.fragment_asphericity = center_shape_metrics.asphericity;
  candidate.fragment_acylindricity = center_shape_metrics.acylindricity;
  candidate.fragment_mst2 = compute_mst2(centers);
  candidate.connectivity_edges =
      count_component_connectivity_edges(component_nodes, adjacency);
  candidate.valid = stats.count > 0;
  return candidate;
}

[[maybe_unused]] bool
analyze_candidate_hard_clash(const vector<size_t> &component_nodes,
                             const vector<PbctopoFragmentCopyNode> &nodes,
                             const vector<PbctopoAtomCopy> &atom_copies,
                             double clash_cutoff,
                             double &min_interfragment_d2) {
  min_interfragment_d2 = numeric_limits<double>::infinity();
  if (component_nodes.empty())
    return false;

  size_t total_atoms = 0;
  for (size_t node_id : component_nodes) {
    const auto &node = nodes[node_id];
    total_atoms += node.atom_end - node.atom_begin;
  }

  unordered_map<ClashCellKey, vector<size_t>, ClashCellKeyHash> clash_grid;
  clash_grid.reserve(max<size_t>(64, total_atoms / 2));
  const double inv_grid = 1.0 / clash_cutoff;
  const double clash_cutoff2 = clash_cutoff * clash_cutoff;

  for (size_t node_id : component_nodes) {
    const auto &node = nodes[node_id];
    for (size_t atom_idx = node.atom_begin; atom_idx < node.atom_end; ++atom_idx) {
      const auto &atom = atom_copies[atom_idx];
      const int cx = static_cast<int>(floor(atom.x * inv_grid));
      const int cy = static_cast<int>(floor(atom.y * inv_grid));
      const int cz = static_cast<int>(floor(atom.z * inv_grid));

      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            const ClashCellKey neighbor{cx + dx, cy + dy, cz + dz};
            const auto it = clash_grid.find(neighbor);
            if (it == clash_grid.end())
              continue;
            for (size_t other_idx : it->second) {
              const auto &other = atom_copies[other_idx];
              if (other.fragment_id == atom.fragment_id)
                continue;
              const double dx_atom = atom.x - other.x;
              const double dy_atom = atom.y - other.y;
              const double dz_atom = atom.z - other.z;
              const double d2 =
                  dx_atom * dx_atom + dy_atom * dy_atom + dz_atom * dz_atom;
              if (d2 < min_interfragment_d2)
                min_interfragment_d2 = d2;
              if (d2 + kDistanceEps < clash_cutoff2)
                return true;
            }
          }
        }
      }

      clash_grid[ClashCellKey{cx, cy, cz}].push_back(atom_idx);
    }
  }

  return false;
}

[[maybe_unused]] bool
better_pbctopo_candidate(const PbctopoCandidate &lhs,
                         const PbctopoCandidate &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  if (lhs.hard_clash != rhs.hard_clash)
    return rhs.hard_clash;
  if (lhs.connectivity_edges != rhs.connectivity_edges)
    return lhs.connectivity_edges > rhs.connectivity_edges;
  int metric_cmp = compare_pbctopo_double_metric(lhs.fragment_mst2,
                                                  rhs.fragment_mst2);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  const int fragment_kappa_cmp =
      compare_shape_kappa(lhs.fragment_kappa2, rhs.fragment_kappa2);
  if (fragment_kappa_cmp != 0)
    return fragment_kappa_cmp < 0;
  const int fragment_asph_cmp = compare_shape_moment(
      lhs.fragment_asphericity, rhs.fragment_asphericity);
  if (fragment_asph_cmp != 0)
    return fragment_asph_cmp < 0;
  const int fragment_acyl_cmp = compare_shape_moment(
      lhs.fragment_acylindricity, rhs.fragment_acylindricity);
  if (fragment_acyl_cmp != 0)
    return fragment_acyl_cmp < 0;
  const int shape_kappa_cmp =
      compare_shape_kappa(lhs.shape_kappa2, rhs.shape_kappa2);
  if (shape_kappa_cmp != 0)
    return shape_kappa_cmp < 0;
  const int shape_asph_cmp =
      compare_shape_moment(lhs.shape_asphericity, rhs.shape_asphericity);
  if (shape_asph_cmp != 0)
    return shape_asph_cmp < 0;
  const int shape_acyl_cmp =
      compare_shape_moment(lhs.shape_acylindricity, rhs.shape_acylindricity);
  if (shape_acyl_cmp != 0)
    return shape_acyl_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.bbox_volume,
                                              rhs.bbox_volume);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  if (lhs.hull_valid && rhs.hull_valid &&
      (metric_cmp = compare_pbctopo_double_metric(lhs.hull_volume,
                                                   rhs.hull_volume)) != 0)
    return metric_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.min_interfragment_d2,
                                              rhs.min_interfragment_d2);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.shift_norm2, rhs.shift_norm2);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  return lhs.node_ids < rhs.node_ids;
}

[[maybe_unused]] void
apply_pbctopo_candidate(Multi_MOL &mol1_system, Multi_MOL &mol2_system,
                        size_t frame_index, const PbctopoCandidate &candidate,
                        const vector<PbctopoFragmentCopyNode> &nodes,
                        const vector<PbctopoAtomCopy> &atom_copies) {
  for (size_t node_id : candidate.node_ids) {
    const auto &node = nodes[node_id];
    for (size_t atom_idx = node.atom_begin; atom_idx < node.atom_end; ++atom_idx) {
      const auto &atom_copy = atom_copies[atom_idx];
      if (atom_copy.owner == 0) {
        mol1_system.m_mol[frame_index].pos.x[atom_copy.local_atom_index] =
            atom_copy.x;
        mol1_system.m_mol[frame_index].pos.y[atom_copy.local_atom_index] =
            atom_copy.y;
        mol1_system.m_mol[frame_index].pos.z[atom_copy.local_atom_index] =
            atom_copy.z;
      } else if (frame_index < mol2_system.m_mol.size()) {
        mol2_system.m_mol[frame_index].pos.x[atom_copy.local_atom_index] =
            atom_copy.x;
        mol2_system.m_mol[frame_index].pos.y[atom_copy.local_atom_index] =
            atom_copy.y;
        mol2_system.m_mol[frame_index].pos.z[atom_copy.local_atom_index] =
            atom_copy.z;
      }
    }
  }
}

} // namespace

int Multi_MOL::solve_pbc_pbctopo(Multi_MOL &mol2, arguments &Arguments,
                                 TitanExecutor *executor,
                                 size_t frame_report_offset) {
  using clock = chrono::steady_clock;
  const size_t frame_count =
      mol2.m_mol.empty() ? this->m_mol.size()
                         : min(this->m_mol.size(), mol2.m_mol.size());
  if (frame_count == 0)
    return 0;

  pbctopo_frame_certified_.assign(frame_count, 0);
  pbctopo_frame_status_.assign(frame_count, "missing_status");

  cache_topology_vdw_radii_from_parameters();
  mol2.cache_topology_vdw_radii_from_parameters();

  reset_pbctopo_log_file();
  ofstream pbctopo_frame_report = open_pbctopo_frame_report();
  ofstream pbctopo_assignment_set_report =
      open_pbctopo_assignment_set_report();
  ofstream pbctopo_rejected_assignment_report =
      open_pbctopo_rejected_assignment_report();
  ofstream pbctopo_candidate_evidence_summary_report;
  ofstream pbctopo_candidate_evidence_relation_report;
  if (Arguments.pbc.pbctopo_evidence_export) {
    pbctopo_candidate_evidence_summary_report =
        open_pbctopo_candidate_evidence_summary_report();
    pbctopo_candidate_evidence_relation_report =
        open_pbctopo_candidate_evidence_relation_report();
  }
  ofstream pbctopo_component_layout_report =
      open_pbctopo_component_layout_report();
  ofstream pbctopo_hard_edge_report = open_pbctopo_hard_edge_report();
  const bool h9_orbit_enabled = Arguments.pbc.pbctopo_h9_orbit;
  ofstream h9_component_representative_report;
  ofstream h9_evidence_component_report;
  ofstream h9_orbit_report;
  if (h9_orbit_enabled) {
    h9_component_representative_report =
        open_pbctopo_h9_component_representative_report();
    h9_evidence_component_report =
        open_pbctopo_h9_evidence_component_report();
    h9_orbit_report = open_pbctopo_h9_orbit_report();
  }
  if (!pbctopo_frame_report.is_open()) {
    append_pbctopo_log_line(
        "[pbctopo] observation export unavailable: "
        "artifact=pbctopo_frame_report.csv");
  }
  if (!pbctopo_assignment_set_report.is_open()) {
    append_pbctopo_log_line(
        "[pbctopo] observation export unavailable: "
        "artifact=pbctopo_assignment_candidates.csv");
  }
  if (!pbctopo_rejected_assignment_report.is_open()) {
    append_pbctopo_log_line(
        "[pbctopo] rejected-assignment audit unavailable: "
        "artifact=pbctopo_rejected_candidates.csv");
  }
  if (Arguments.pbc.pbctopo_evidence_export &&
      (!pbctopo_candidate_evidence_summary_report.is_open() ||
       !pbctopo_candidate_evidence_relation_report.is_open())) {
    append_pbctopo_log_line(
        "[pbctopo] observation export unavailable: "
        "artifacts=pbctopo_candidate_evidence_summary.csv|"
        "pbctopo_candidate_evidence_relations.csv");
  }
  if (!pbctopo_component_layout_report.is_open()) {
    append_pbctopo_log_line(
        "[pbctopo] observation export unavailable: "
        "artifact=pbctopo_component_layout.csv");
  }
  if (!pbctopo_hard_edge_report.is_open()) {
    append_pbctopo_log_line(
        "[pbctopo] observation export unavailable: "
        "artifact=pbctopo_hard_edges.csv");
  }
  if (h9_orbit_enabled &&
      (!h9_component_representative_report.is_open() ||
       !h9_evidence_component_report.is_open() ||
       !h9_orbit_report.is_open())) {
    append_pbctopo_log_line(
        "[pbctopo] H9 export unavailable: one or more artifacts could not be "
        "opened");
  }
  vector<PbctopoComponentLayoutObservation> &emitted_component_layouts =
      pbctopo_component_layout_report_state().emitted_layouts;
  size_t fallback_frames = 0;
  size_t valid_candidates = 0;
  PbctopoTimingStats timing;
  PbctopoWarmState warm_state;
  auto reset_uncommitted_warm_state = [&]() {
    reset_kinematic_seed_history(warm_state, timing);
    reset_pbctopo_warm_state(warm_state);
    ++timing.pbctopo_warm_state_resets;
  };
  const PbctopoRuntimeCutoffs runtime_cutoffs =
      resolve_pbctopo_cutoffs(Arguments);
  const PbctopoSolveMode solve_mode = resolve_pbctopo_solve_mode(Arguments);
  const PbctopoSearchPolicy search_policy =
      titan_pbctopo::resolve_search_policy(Arguments);
  const bool audit_exact_search =
      search_policy == PbctopoSearchPolicy::AuditExact;
  constexpr size_t kBytesPerMiB = 1024u * 1024u;
  const size_t audit_memory_mib =
      static_cast<size_t>(Arguments.pbc.pbctopo_audit_max_memory_mb);
  const size_t audit_memory_bytes =
      audit_memory_mib > numeric_limits<size_t>::max() / kBytesPerMiB
          ? numeric_limits<size_t>::max()
          : audit_memory_mib * kBytesPerMiB;
  const PbctopoComponentSearchLimits audit_exact_limits{
      static_cast<size_t>(Arguments.pbc.pbctopo_audit_max_states),
      Arguments.pbc.pbctopo_audit_max_seconds, audit_memory_bytes, true};
  const PbctopoAnchorBackend anchor_backend =
      resolve_pbctopo_anchor_backend(Arguments);
  const PbctopoAnchorMetricMode anchor_metric_mode =
      titan_pbctopo::resolve_anchor_metric_mode(Arguments);
  const PbctopoTemporalMode temporal_mode =
      titan_pbctopo::resolve_temporal_mode(Arguments);
  const size_t requested_exact_component_limit =
      Arguments.pbc.pbctopo_exact_component_limit_set
          ? static_cast<size_t>(Arguments.pbc.pbctopo_exact_component_limit)
          : kPbctopoExactComponentLimit;
  const size_t exact_component_limit =
      min(requested_exact_component_limit, kPbctopoExactComponentLimitMax);
  const size_t requested_beam_width =
      Arguments.pbc.pbctopo_beam_width_set
          ? static_cast<size_t>(Arguments.pbc.pbctopo_beam_width)
          : kPbctopoBeamWidth;
  const size_t beam_width =
      max<size_t>(1, min(requested_beam_width, kPbctopoBeamWidthMax));
  const bool component_shell_radius_explicit =
      Arguments.pbc.pbctopo_component_shell_radius_set;
  const int configured_component_shell_radius =
      component_shell_radius_explicit
          ? Arguments.pbc.pbctopo_component_shell_radius
          : 0;
  if (h9_orbit_enabled &&
      (!component_shell_radius_explicit ||
       configured_component_shell_radius != 2)) {
    TitanUI::print_error(
        "PBCTOPO H9 orbit requires -pbctopo-component-shell-radius=2.");
    append_pbctopo_log_line(
        "[pbctopo] H9 orbit rejected: component shell radius must be "
        "explicitly fixed at 2");
    return 1;
  }
  append_pbctopo_log_line("[pbctopo] begin run: frames=" +
                          to_string(frame_count) + ", pbctopo_mode=" +
                          pbctopo_solve_mode_name(solve_mode) +
                          ", search_policy=" +
                          titan_pbctopo::search_policy_name(search_policy) +
                          ", observation_schema_version=" +
                          to_string(titan_pbctopo::
                                        VIBE_OBSERVATION_SCHEMA_VERSION) +
                          ", observation_scope=" +
                          string(titan_pbctopo::VIBE_OBSERVATION_SCOPE) +
                          ", anchor_backend=" +
                          pbctopo_anchor_backend_name(anchor_backend) +
                          ", anchor_metrics=" +
                          titan_pbctopo::anchor_metric_mode_name(
                              anchor_metric_mode) +
                          ", temporal=" +
                          titan_pbctopo::temporal_mode_name(temporal_mode) +
                          ", temporal_status=" +
                          string(temporal_mode == PbctopoTemporalMode::Viterbi
                                     ? "experimental_diagnostic"
                                     : "production_framewise") +
                          ", vdw_clash_scale=" +
                          to_string(runtime_cutoffs.vdw_clash_scale) +
                          ", bond_cutoff=" +
                          to_string(runtime_cutoffs.bond_cutoff) +
                          ", clash_cutoff=" +
                          to_string(runtime_cutoffs.clash_cutoff) +
                          ", contact_cutoff=" +
                          to_string(runtime_cutoffs.pair_contact_cutoff) +
                          ", continuity_cutoff=" +
                          to_string(runtime_cutoffs.continuity_cutoff) +
                          ", exact_component_limit=" +
                          to_string(exact_component_limit) + ", beam_width=" +
                          to_string(beam_width) +
                          ", component_shell_radius=" +
                          (component_shell_radius_explicit
                               ? to_string(configured_component_shell_radius)
                               : string("auto")));
  const PbctopoTrajectoryContext trajectory_context =
      build_pbctopo_trajectory_context(*this, mol2, frame_count);
  vector<titan_pbctopo::PbctopoRequiredContact> explicit_required_contacts;
  if (!Arguments.pbc.pbctopo_required_contacts_file.empty()) {
    const size_t selected_atom_count =
        trajectory_context.rec_static.atom_count +
        trajectory_context.lig_static.atom_count;
    auto loaded_contacts = titan_pbctopo::load_pbctopo_required_contacts(
        Arguments.pbc.pbctopo_required_contacts_file, selected_atom_count);
    if (!loaded_contacts) {
      append_pbctopo_log_line("[pbctopo] required interface contact error: " +
                              loaded_contacts.error());
      TitanUI::print_error(loaded_contacts.error());
      return 1;
    }
    explicit_required_contacts = std::move(*loaded_contacts);
    timing.pbctopo_required_interface_contacts_loaded =
        explicit_required_contacts.size();
    append_pbctopo_log_line(
        "[pbctopo] required interface contacts loaded: source=user_file" +
        string(", file=") + Arguments.pbc.pbctopo_required_contacts_file +
        ", contacts=" + to_string(explicit_required_contacts.size()) +
        ", atom_indexing=selected_1_based");
  } else {
    append_pbctopo_log_line(
        "[pbctopo] required interface contacts: source=none, "
        "certificate_scope=image_intercomponent_steric");
  }
  const auto build_explicit_evidence_component_edges =
      [&](const auto &atom_to_component, size_t component_count) {
        vector<pair<size_t, size_t>> edges;
        edges.reserve(explicit_required_contacts.size());
        for (const auto &contact : explicit_required_contacts) {
          if (contact.a >= atom_to_component.size() ||
              contact.b >= atom_to_component.size()) {
            continue;
          }
          size_t component_a = atom_to_component[contact.a];
          size_t component_b = atom_to_component[contact.b];
          if (component_a >= component_count || component_b >= component_count ||
              component_a == component_b) {
            continue;
          }
          if (component_b < component_a)
            swap(component_a, component_b);
          edges.emplace_back(component_a, component_b);
        }
        sort(edges.begin(), edges.end());
        edges.erase(unique(edges.begin(), edges.end()), edges.end());
        return edges;
      };
  vector<vector<PbctopoCutCandidate>> temporal_cut_candidates(frame_count);
  vector<PbctopoCutCandidate> greedy_selected_candidates(frame_count);
  struct PbctopoTemporalComponentLayout {
    vector<uint32_t> atom_component_index;
    vector<uint64_t> component_ids;
    vector<double> component_weights;
    size_t component_count = 0;
  };
  vector<PbctopoTemporalComponentLayout> temporal_component_layouts;
  vector<size_t> temporal_frame_layout(
      frame_count, numeric_limits<size_t>::max());
  vector<PbctopoFrameReport> deferred_frame_reports(frame_count);
  vector<unsigned char> deferred_frame_report_valid(frame_count, 0);
  vector<PbctopoContactHypothesisSet> temporal_contact_hypotheses(frame_count);
  size_t evidence_relation_export_groups = 0;
  size_t evidence_relation_export_rows = 0;
  size_t evidence_relation_export_duplicates = 0;
  size_t evidence_relation_export_failures = 0;
  auto write_or_defer_frame_report = [&](size_t frame_idx,
                                          PbctopoFrameReport report) {
    if (frame_idx < pbctopo_frame_status_.size()) {
      pbctopo_frame_status_[frame_idx] =
          titan_pbctopo::pbctopo_frame_status_name(report.status);
      pbctopo_frame_certified_[frame_idx] =
          report.status == PbctopoFrameStatus::Certified ? 1 : 0;
    }
    if (report.component_audit_exact_requested &&
        !report.component_audit_exact_completed &&
        !report.component_audit_exact_incomplete) {
      report.component_audit_exact_incomplete = true;
      report.component_audit_exact_incomplete_reason = "not_attempted";
      ++timing.component_audit_exact_incomplete;
      append_pbctopo_log_line(
          "[pbctopo] audit-exact incomplete at frame " +
          to_string(report.frame) +
          ": reason=not_attempted, action=fail_closed");
    }
    if (temporal_mode == PbctopoTemporalMode::Viterbi) {
      deferred_frame_reports[frame_idx] = report;
      deferred_frame_report_valid[frame_idx] = 1;
    } else if (pbctopo_frame_report.is_open()) {
      write_pbctopo_frame_report_row(pbctopo_frame_report, report);
    }
  };
  const bool has_ligand_static = trajectory_context.lig_static.atom_count > 0;
  const bool static_topology_ready =
      trajectory_context.rec_static.topology_graph_available &&
      (!has_ligand_static ||
       trajectory_context.lig_static.topology_graph_available);
  const bool static_metadata_ready =
      !static_topology_ready &&
      trajectory_context.rec_static.metadata_graph_available &&
      (!has_ligand_static ||
       trajectory_context.lig_static.metadata_graph_available);
  const string static_graph_source =
      static_topology_ready
          ? "gromacs_topology"
          : (static_metadata_ready ? "metadata_chain_residue"
                                   : "geometry_cutoff");
  append_pbctopo_log_line(
      "[pbctopo] local graph prepared: source=" + static_graph_source +
      ", selected_atoms=" +
      to_string(trajectory_context.rec_static.atom_count +
                trajectory_context.lig_static.atom_count) +
      ", topology_bonds=" +
      to_string(trajectory_context.rec_static.topology_bonds_available +
                trajectory_context.lig_static.topology_bonds_available) +
      ", projected_topology_bonds=" +
      to_string(
          trajectory_context.rec_static.projected_topology_bonds.size() +
          trajectory_context.lig_static.projected_topology_bonds.size()) +
      ", metadata_edges=" +
      to_string(trajectory_context.rec_static.projected_metadata_edges.size() +
                trajectory_context.lig_static.projected_metadata_edges.size()) +
      ", vdw_topology_radii=" +
      to_string(trajectory_context.rec_static.vdw_topology_count +
                trajectory_context.lig_static.vdw_topology_count) +
      ", vdw_parameter_radii=" +
      to_string(trajectory_context.rec_static.vdw_parameter_count +
                trajectory_context.lig_static.vdw_parameter_count) +
      ", vdw_zero_lj_radii=" +
      to_string(trajectory_context.rec_static.vdw_zero_lj_count +
                trajectory_context.lig_static.vdw_zero_lj_count) +
      ", vdw_inferred_radii=" +
      to_string(trajectory_context.rec_static.vdw_inferred_count +
                trajectory_context.lig_static.vdw_inferred_count) +
      ", vdw_generic_fallback_radii=" +
      to_string(trajectory_context.rec_static.vdw_generic_fallback_count +
                trajectory_context.lig_static.vdw_generic_fallback_count));
  PbctopoFrameScratch scratch;
  scratch.reserve_from_context(trajectory_context);
  PbctopoHullScratch hull_scratch;
  hull_scratch.reserve(trajectory_context.total_atoms_hint);
  auto &base_atoms = scratch.base_atoms;
  auto &projection_atoms = scratch.projection_atoms;
  auto &component_atoms = scratch.component_atoms;
  auto &local_edges = scratch.local_edges;
  auto &component_scoring_edges = scratch.component_scoring_edges;
  auto &projection_edges = scratch.projection_edges;
  auto &projection_continuity_pairs = scratch.projection_continuity_pairs;
  auto &projection_anchor_indices = scratch.projection_anchor_indices;
  auto &local_graph_atom_coverage = scratch.local_graph_atom_coverage;
  auto &local_edge_set = scratch.local_edge_set;
  auto &steric_exclusion_set = scratch.steric_exclusion_set;
  auto &projection_edge_set = scratch.projection_edge_set;
  auto &components = scratch.components;
  auto &shifts = scratch.shifts;
  auto &volume_seeds = scratch.volume_seeds;
  auto &warm_points = scratch.warm_points;
  auto &coarse_warm_points = scratch.coarse_warm_points;
  auto &placed_points = scratch.placed_points;
  auto &best_safe_points = scratch.best_safe_points;
  auto &verify_points = scratch.verify_points;
  auto &component_points = scratch.component_points;
  auto &coarse_candidates = scratch.coarse_candidates;
  auto &coarse_seed_options = scratch.coarse_seed_options;
  auto &coarse_results = scratch.coarse_results;
  auto &projection_index = scratch.projection_index;
  auto &atom_component_index = scratch.atom_component_index;
  auto &virtual_search_workspace = scratch.virtual_search_workspace;
  auto &virtual_search_plan = scratch.virtual_search_plan;
  vector<PbctopoHullSpan> hull_owner_spans;
  hull_owner_spans.reserve(8);
  vector<Vec3> hull_rescue_points;
  hull_rescue_points.reserve(trajectory_context.total_atoms_hint);
  bool projection_clash_skip_active = false;
  string llps_local_unit_source = "none";
  auto fallback_pbci_frame = [&](size_t frame_index) {
    ++timing.pbci_fallback_frames;
    int chain_ref = Arguments.pbc.chain_ref;
    const size_t chain_count = this->Index.empty() ? 0 : this->Index.size() - 1;
    if (chain_count == 0) {
      chain_ref = 0;
    } else if (chain_ref < 0 ||
               static_cast<size_t>(chain_ref) >= chain_count) {
      chain_ref = 0;
    }

    this->find_best_image_envelope(frame_index, chain_ref, Arguments);
    if (frame_index < mol2.m_mol.size() && !mol2.m_mol[frame_index].pos.x.empty())
      this->find_best_image_of_mol2(mol2, frame_index, chain_ref, Arguments);

    vector<PbctopoAtom> fallback_atoms;
    fallback_atoms.reserve(trajectory_context.total_atoms_hint);
    append_atoms_for_system(this->m_mol[frame_index], &this->info,
                            &trajectory_context.rec_static, 0, fallback_atoms);
    if (frame_index < mol2.m_mol.size() && !mol2.m_mol[frame_index].pos.x.empty()) {
      append_atoms_for_system(mol2.m_mol[frame_index], &mol2.info,
                              &trajectory_context.lig_static, 1, fallback_atoms);
    }
    prepare_pbctopo_steric_radii(fallback_atoms,
                                 runtime_cutoffs.vdw_clash_scale);
    vector<pair<int, int>> fallback_contacting_owner_pairs =
        build_contacting_owner_pairs({}, fallback_atoms,
                                     this->m_mol[frame_index].Box,
                                     runtime_cutoffs.pair_contact_cutoff,
                                     false);
    vector<Vec3> fallback_points;
    fallback_points.reserve(fallback_atoms.size());
    for (const auto &atom : fallback_atoms)
      fallback_points.push_back({atom.x, atom.y, atom.z});
    if (!reimage_contacting_owner_points_minimum_image(
            fallback_points, fallback_atoms, this->m_mol[frame_index].Box,
            fallback_contacting_owner_pairs)) {
      return;
    }
    for (size_t atom_idx = 0; atom_idx < fallback_atoms.size(); ++atom_idx) {
      const auto &atom = fallback_atoms[atom_idx];
      const auto &point = fallback_points[atom_idx];
      if (atom.owner == 0) {
        this->m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
        this->m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
        this->m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
      } else if (frame_index < mol2.m_mol.size()) {
        mol2.m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
        mol2.m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
        mol2.m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
      }
    }
    append_pbctopo_log_line(
        "[pbctopo] fallback pbci=0 reimaged contacting owners at frame " +
        to_string(static_cast<long long>(this->m_mol[frame_index].time_of_snap)));
  };

  for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
    ++timing.frames;
    size_t frame_fatal_lattice_query_failures = 0;
    PbctopoFrameLatticeQueryAuditScope lattice_query_audit_scope(
        timing, frame_fatal_lattice_query_failures);
    const PbctopoWarmState frame_start_warm_state = warm_state;
    const bool kinematic_seed_history_ready =
        has_consecutive_kinematic_seed_history(warm_state, frame_idx);
    const PbctopoCutCandidate kinematic_seed_previous =
        warm_state.kinematic_previous_virtual_candidate;
    const PbctopoCutCandidate kinematic_seed_latest =
        warm_state.kinematic_latest_virtual_candidate;
    PbctopoFrameReport frame_report;
    PbctopoSelectedEvidenceAudit selected_evidence_audit;
    frame_report.frame = frame_report_offset + frame_idx;
    frame_report.time_ps = this->m_mol[frame_idx].time_of_snap;
    frame_report.temporal_inference_experimental =
        temporal_mode == PbctopoTemporalMode::Viterbi;
    frame_report.component_audit_exact_requested = audit_exact_search;
    if (audit_exact_search)
      ++timing.component_audit_exact_frames;
    const auto frame_report_started = clock::now();
    const double frame_graph_s_before = timing.graph_s + timing.component_graph_s;
    const double frame_virtual_s_before =
        timing.event_planes_s + timing.coarse_s + timing.refine_s +
        timing.early_projection_s;
    const double frame_component_s_before =
        timing.component_prep_s + timing.component_graph_s +
        timing.component_search_s + timing.warm_component_s;
    const bool skip_projection_candidate_clash = projection_clash_skip_active;
    if (skip_projection_candidate_clash)
      ++timing.projection_clash_skip_frames;
    scratch.reset_frame();
    auto stage_started = clock::now();
    append_atoms_for_system(this->m_mol[frame_idx], &this->info,
                            &trajectory_context.rec_static, 0, base_atoms);
    if (frame_idx < mol2.m_mol.size() && !mol2.m_mol[frame_idx].pos.x.empty()) {
      append_atoms_for_system(mol2.m_mol[frame_idx], &mol2.info,
                              &trajectory_context.lig_static, 1, base_atoms);
    }
    const auto frame_binding =
        titan_pbctopo::build_pbctopo_frame_binding_observation(
            frame_report.frame, this->m_mol[frame_idx].Box, base_atoms);
    frame_report.trajectory_frame_index =
        frame_binding.trajectory_frame_index;
    frame_report.box_matrix = frame_binding.box_matrix;
    frame_report.box_hash = frame_binding.box_hash;
    frame_report.atom_selection_hash = frame_binding.atom_selection_hash;
    frame_report.canonical_atom_universe_hash =
        frame_binding.canonical_atom_universe_hash;
    frame_report.wrapped_coordinate_hash =
        frame_binding.wrapped_coordinate_hash;
    frame_report.frame_binding_valid = frame_binding.valid;
    prepare_pbctopo_steric_radii(base_atoms,
                                 runtime_cutoffs.vdw_clash_scale);
    timing.append_s += chrono::duration<double>(clock::now() - stage_started).count();

    if (base_atoms.empty()) {
      frame_report.status = PbctopoFrameStatus::Fallback;
      frame_report.mode = PbctopoFrameMode::Pbci;
      frame_report.fallback_reason = PbctopoFailureReason::NoSeed;
      frame_report.time_total_ms =
          chrono::duration<double, milli>(clock::now() - frame_report_started)
              .count();
      write_or_defer_frame_report(frame_idx, frame_report);
      reset_uncommitted_warm_state();
      continue;
    }

    const PbctopoLatticeMetric frame_lattice(this->m_mol[frame_idx].Box);
    ++timing.pbctopo_lattice_audits;
    if (std::isfinite(frame_lattice.condition_estimate())) {
      timing.pbctopo_lattice_condition_max =
          std::max(timing.pbctopo_lattice_condition_max,
                   frame_lattice.condition_estimate());
    }
    if (frame_lattice.reduction_fallback())
      ++timing.pbctopo_lattice_reduction_fallback_frames;
    if (!frame_lattice.valid()) {
      ++timing.pbctopo_lattice_invalid_frames;
      const string invalid_lattice_message =
          "[pbctopo] frame " +
          to_string(static_cast<long long>(
              this->m_mol[frame_idx].time_of_snap)) +
          " left unchanged: invalid lattice status=" +
          titan_pbctopo::pbctopo_lattice_status_name(frame_lattice.status()) +
          ", condition_estimate=" +
          to_string(frame_lattice.condition_estimate());
      append_pbctopo_log_line(invalid_lattice_message);
      TitanUI::print_warning(invalid_lattice_message);
      frame_report.status = PbctopoFrameStatus::Fallback;
      frame_report.mode = PbctopoFrameMode::None;
      frame_report.fallback_reason = PbctopoFailureReason::InvalidLattice;
      frame_report.time_total_ms =
          chrono::duration<double, milli>(clock::now() - frame_report_started)
              .count();
      write_or_defer_frame_report(frame_idx, frame_report);
      reset_uncommitted_warm_state();
      continue;
    }

    stage_started = clock::now();
    build_heavy_atom_subset(base_atoms, projection_atoms);
    const bool heavy_search_omits_atoms =
        projection_atoms.size() < base_atoms.size();
    if (heavy_search_omits_atoms)
      ++timing.pbctopo_heavy_search_frames;
    timing.heavy_subset_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
    stage_started = clock::now();
    const PbctopoTranslationCache translation_cache =
        build_virtual_box_translation_cache(this->m_mol[frame_idx].Box);
    timing.translation_cache_s +=
        chrono::duration<double>(clock::now() - stage_started).count();

    stage_started = clock::now();
    const bool has_ligand_frame =
        frame_idx < mol2.m_mol.size() && !mol2.m_mol[frame_idx].pos.x.empty();
    const bool rec_use_topology =
        trajectory_context.rec_static.topology_graph_available;
    const bool lig_use_topology =
        (has_ligand_frame &&
         trajectory_context.lig_static.topology_graph_available);
    const bool rec_use_metadata =
        !rec_use_topology &&
        trajectory_context.rec_static.metadata_graph_available;
    const bool lig_use_metadata =
        has_ligand_frame &&
        (!trajectory_context.lig_static.topology_graph_available &&
         trajectory_context.lig_static.metadata_graph_available);
    const bool rec_use_geometry = !rec_use_topology && !rec_use_metadata;
    const bool lig_use_geometry =
        has_ligand_frame && !lig_use_topology && !lig_use_metadata;
    const bool any_topology_graph = rec_use_topology || lig_use_topology;
    const bool any_metadata_graph = rec_use_metadata || lig_use_metadata;
    const bool any_geometry_graph = rec_use_geometry || lig_use_geometry;
    if (any_topology_graph && any_metadata_graph) {
      frame_report.certificate_graph_source =
          PbctopoCertificateGraphSource::Mixed;
    } else if (any_topology_graph) {
      frame_report.certificate_graph_source =
          PbctopoCertificateGraphSource::GromacsTopology;
    } else if (any_metadata_graph) {
      frame_report.certificate_graph_source =
          PbctopoCertificateGraphSource::MetadataChainResidue;
    }

    local_edges.clear();
    local_edge_set.clear();
    const size_t projected_static_edges =
        (rec_use_topology
             ? trajectory_context.rec_static.projected_topology_bonds.size()
             : rec_use_metadata
                   ? trajectory_context.rec_static.projected_metadata_edges.size()
                   : 0) +
        (lig_use_topology
             ? trajectory_context.lig_static.projected_topology_bonds.size()
             : lig_use_metadata
                   ? trajectory_context.lig_static.projected_metadata_edges.size()
                   : 0);
    local_edges.reserve(projected_static_edges +
                        (any_geometry_graph ? base_atoms.size() * 2 : 0));
    local_edge_set.reserve(local_edges.capacity() * 2 + 1);

    if (any_topology_graph)
      ++timing.pbctopo_topology_static_graph_frames;
    if (any_metadata_graph)
      ++timing.pbctopo_metadata_graph_frames;

    if (rec_use_topology) {
      append_topology_bond_edges_for_system(
          trajectory_context.rec_static, 0, base_atoms,
          this->m_mol[frame_idx].Box, local_edges, local_edge_set, timing);
    } else if (rec_use_metadata) {
      append_metadata_graph_edges_for_system(trajectory_context.rec_static, 0,
                                             base_atoms, local_edges,
                                             local_edge_set, timing);
    }
    if (lig_use_topology) {
      append_topology_bond_edges_for_system(
          trajectory_context.lig_static,
          trajectory_context.rec_static.atom_count, base_atoms,
          this->m_mol[frame_idx].Box, local_edges, local_edge_set, timing);
    } else if (lig_use_metadata) {
      append_metadata_graph_edges_for_system(
          trajectory_context.lig_static,
          trajectory_context.rec_static.atom_count, base_atoms, local_edges,
          local_edge_set, timing);
    }
    const auto static_graph_done = clock::now();
    timing.pbctopo_topology_edge_append_s +=
        chrono::duration<double>(static_graph_done - stage_started).count();

    if (any_geometry_graph) {
      const size_t edge_count_before_geometry = local_edges.size();
      vector<char> geometry_eligible(base_atoms.size(), 0);
      if (rec_use_geometry) {
        const size_t rec_end = min(trajectory_context.rec_static.atom_count,
                                   geometry_eligible.size());
        fill(geometry_eligible.begin(), geometry_eligible.begin() + rec_end, 1);
      }
      if (lig_use_geometry) {
        const size_t lig_begin = trajectory_context.rec_static.atom_count;
        const size_t lig_end = min(
            geometry_eligible.size(),
            lig_begin + trajectory_context.lig_static.atom_count);
        if (lig_begin < lig_end)
          fill(geometry_eligible.begin() + lig_begin,
               geometry_eligible.begin() + lig_end, 1);
      }
      build_local_neighborhood_graph(base_atoms, this->m_mol[frame_idx].Box,
                                     local_edges, local_edge_set,
                                     runtime_cutoffs.bond_cutoff, true,
                                     geometry_eligible);
      timing.pbctopo_geometry_edges_added +=
          local_edges.size() - edge_count_before_geometry;
      ++timing.pbctopo_geometry_fallback_frames;
    }
    timing.pbctopo_local_graph_geometry_s +=
        chrono::duration<double>(clock::now() - static_graph_done).count();
    timing.graph_s += chrono::duration<double>(clock::now() - stage_started).count();

    size_t hard_connectivity_edge_count = 0;
    size_t topology_edge_count = 0;
    size_t metadata_edge_count = 0;
    size_t soft_geometry_edge_count = 0;
    local_graph_atom_coverage.assign(base_atoms.size(), 0);
    steric_exclusion_set.clear();
    steric_exclusion_set.reserve(local_edges.size() + 1);
    for (const auto &edge : local_edges) {
      const bool edge_indices_valid =
          edge.a < base_atoms.size() && edge.b < base_atoms.size();
      if (titan_pbctopo::pbctopo_edge_is_hard_connectivity(edge.kind)) {
        ++hard_connectivity_edge_count;
        if (edge_indices_valid) {
          local_graph_atom_coverage[edge.a] |= 0x1u;
          local_graph_atom_coverage[edge.b] |= 0x1u;
        }
      }
      if (edge.kind == PbctopoLocalEdgeKind::TopologyBond) {
        ++topology_edge_count;
        if (edge_indices_valid) {
          local_graph_atom_coverage[edge.a] |= 0x2u;
          local_graph_atom_coverage[edge.b] |= 0x2u;
        }
      }
      if (edge.kind == PbctopoLocalEdgeKind::MetadataSkeleton) {
        ++metadata_edge_count;
        if (edge_indices_valid) {
          local_graph_atom_coverage[edge.a] |= 0x4u;
          local_graph_atom_coverage[edge.b] |= 0x4u;
        }
      }
      if (edge.kind == PbctopoLocalEdgeKind::GeometryHeuristic)
        ++soft_geometry_edge_count;
      if (titan_pbctopo::pbctopo_edge_is_steric_exclusion(edge.kind))
        steric_exclusion_set.insert(atom_pair_key(edge.a, edge.b));
    }
    size_t hard_graph_atoms_covered = 0;
    size_t topology_bond_atoms_covered = 0;
    size_t metadata_only_atoms = 0;
    for (const std::uint8_t coverage : local_graph_atom_coverage) {
      hard_graph_atoms_covered += (coverage & 0x1u) != 0;
      topology_bond_atoms_covered += (coverage & 0x2u) != 0;
      metadata_only_atoms +=
          (coverage & 0x4u) != 0 && (coverage & 0x2u) == 0;
    }
    timing.pbctopo_hard_connectivity_edges += hard_connectivity_edge_count;
    timing.pbctopo_soft_geometry_edges += soft_geometry_edge_count;
    timing.pbctopo_steric_exclusion_pairs += steric_exclusion_set.size();

    if (solve_mode == PbctopoSolveMode::LlpsLocal) {
      const PbctopoLlpsLocalUnits llps_units =
          build_pbctopo_llps_local_units(*this, mol2, base_atoms, local_edges);
      llps_local_unit_source =
          merge_pbctopo_llps_unit_source(llps_local_unit_source,
                                         llps_units.source);
      const auto llps_lattice_audit_before =
          titan_pbctopo::pbctopo_lattice_query_audit();
      vector<Vec3> llps_points = unwrap_pbctopo_llps_local_points(
          base_atoms, local_edges, this->m_mol[frame_idx].Box, llps_units,
          timing);
      const auto llps_lattice_audit_delta =
          titan_pbctopo::pbctopo_lattice_query_audit_delta(
              llps_lattice_audit_before,
              titan_pbctopo::pbctopo_lattice_query_audit());
      frame_fatal_lattice_query_failures +=
          llps_lattice_audit_delta.failures;
      if (llps_lattice_audit_delta.failures > 0) {
        const string failure_message =
            "[pbctopo] LLPS-local frame " +
            to_string(static_cast<long long>(
                this->m_mol[frame_idx].time_of_snap)) +
            " left unchanged after a failed lattice-image query";
        append_pbctopo_log_line(failure_message);
        TitanUI::print_warning(failure_message);
        ++fallback_frames;
        frame_report.status = PbctopoFrameStatus::Fallback;
        frame_report.certificate_scope = PbctopoCertificateScope::None;
        frame_report.mode = PbctopoFrameMode::None;
        frame_report.component_count = llps_units.units.size();
        frame_report.fallback_reason =
            PbctopoFailureReason::LatticeQueryFailed;
        frame_report.time_graph_ms =
            max(0.0, (timing.graph_s + timing.component_graph_s -
                      frame_graph_s_before) *
                         1000.0);
        frame_report.time_total_ms =
            chrono::duration<double, milli>(clock::now() -
                                            frame_report_started)
                .count();
        write_or_defer_frame_report(frame_idx, frame_report);
        reset_uncommitted_warm_state();
        continue;
      }
      stage_started = clock::now();
      apply_pbctopo_points_direct(base_atoms, llps_points, *this, mol2,
                                  frame_idx);
      timing.apply_s +=
          chrono::duration<double>(clock::now() - stage_started).count();
      ++timing.apply_calls;
      ++timing.pbctopo_llps_local_frames;
      timing.pbctopo_llps_local_units += llps_units.units.size();
      ++valid_candidates;

      frame_report.status = PbctopoFrameStatus::LocalUnwrap;
      frame_report.evidence_state = PbctopoEvidenceState::NotApplicable;
      frame_report.certificate_scope =
          PbctopoCertificateScope::LocalUnwrap;
      frame_report.mode = PbctopoFrameMode::LlpsLocal;
      frame_report.component_count = llps_units.units.size();
      frame_report.soft_valid_candidate_count = 1;
      frame_report.nonclashing_candidate_count = 0;
      frame_report.unique_certified_assignment_count = 0;
      frame_report.committed_candidate_count = 1;
      frame_report.fallback_reason = PbctopoFailureReason::None;
      frame_report.time_graph_ms =
          max(0.0, (timing.graph_s + timing.component_graph_s -
                    frame_graph_s_before) *
                       1000.0);
      frame_report.time_virtual_ms =
          max(0.0, (timing.coarse_s + timing.refine_s +
                    timing.early_projection_s - frame_virtual_s_before) *
                       1000.0);
      frame_report.time_component_ms =
          max(0.0, (timing.component_prep_s + timing.component_graph_s +
                    timing.component_search_s + timing.warm_component_s -
                    frame_component_s_before) *
                       1000.0);
      frame_report.time_total_ms =
          chrono::duration<double, milli>(clock::now() - frame_report_started)
              .count();
      write_or_defer_frame_report(frame_idx, frame_report);
      reset_uncommitted_warm_state();
      continue;
    }

    stage_started = clock::now();
    build_projection_edge_subset(base_atoms, projection_atoms, local_edges,
                                 projection_edges, projection_edge_set,
                                 projection_index);
    build_projection_continuity_subset(base_atoms, projection_atoms,
                                       trajectory_context.continuity_pairs,
                                       projection_continuity_pairs,
                                       projection_index);
    build_projection_anchor_subset(base_atoms,
                                   trajectory_context.continuity_anchor_indices,
                                   projection_anchor_indices, projection_index);
    const vector<pair<int, int>> local_hard_required_owner_pairs =
        build_contacting_owner_pairs(local_edges, base_atoms,
                                     this->m_mol[frame_idx].Box,
                                     runtime_cutoffs.pair_contact_cutoff,
                                     true);
    const vector<pair<int, int>> projection_hard_required_owner_pairs =
        build_contacting_owner_pairs(projection_edges, projection_atoms,
                                     this->m_mol[frame_idx].Box,
                                     runtime_cutoffs.pair_contact_cutoff,
                                     true);
    timing.pbctopo_hard_required_owner_pairs +=
        local_hard_required_owner_pairs.size();
    timing.projection_edge_s +=
        chrono::duration<double>(clock::now() - stage_started).count();

    PbctopoAnchorGraphContext base_anchor_context =
        build_anchor_graph_context(base_atoms, this->m_mol[frame_idx].Box,
                                   trajectory_context.continuity_pairs,
                                   trajectory_context.continuity_anchor_indices,
                                   runtime_cutoffs.pair_contact_cutoff,
                                   &timing);
    PbctopoAnchorGraphContext projection_anchor_context =
        build_anchor_graph_context(projection_atoms, this->m_mol[frame_idx].Box,
                                   projection_continuity_pairs,
                                   projection_anchor_indices,
                                   runtime_cutoffs.pair_contact_cutoff,
                                   &timing);
    const auto require_complete_eobs = [](const PbctopoAnchorGraphContext &context,
                                          const char *scope) {
      if (context.soft_observed_contact_construction_attempted &&
          !context.soft_observed_contact_construction_complete) {
        throw runtime_error(
            string("PBCTOPO E_obs construction failed closed for ") + scope +
            ": cell_setup_failures=" +
            to_string(context.soft_observed_contact_cell_setup_failures) +
            ", atom_index_failures=" +
            to_string(context.soft_observed_contact_atom_index_failures) +
            ", mic_query_failures=" +
            to_string(context.soft_observed_contact_mic_query_failures));
      }
    };
    require_complete_eobs(base_anchor_context, "all-atom selected system");
    require_complete_eobs(projection_anchor_context, "projection system");
    const bool use_anchor_order =
        anchor_metric_mode == PbctopoAnchorMetricMode::Full ||
        anchor_metric_mode == PbctopoAnchorMetricMode::NoContact;
    const bool use_anchor_contact =
        anchor_metric_mode == PbctopoAnchorMetricMode::Full ||
        anchor_metric_mode == PbctopoAnchorMetricMode::NoOrder;
    base_anchor_context.use_order_metrics = use_anchor_order;
    base_anchor_context.use_contact_metrics = use_anchor_contact;
    projection_anchor_context.use_order_metrics = use_anchor_order;
    projection_anchor_context.use_contact_metrics = use_anchor_contact;

    stage_started = clock::now();
    components = build_hard_connected_components_from_local_edges(
        base_atoms.size(), local_edges);
    atom_component_index =
        build_atom_component_index(base_atoms.size(), components);
    const vector<pair<size_t, size_t>> explicit_evidence_component_edges =
        build_explicit_evidence_component_edges(atom_component_index,
                                                components.size());
    PbctopoIdentifiability frame_identifiability =
        titan_pbctopo::evaluate_pbctopo_identifiability(
            components.size(), {}, explicit_required_contacts,
            atom_component_index);
    const PbctopoHardGraphSource frame_hard_graph_source =
        topology_edge_count > 0 && metadata_edge_count > 0
            ? PbctopoHardGraphSource::MixedHard
            : topology_edge_count > 0
                  ? PbctopoHardGraphSource::ExplicitTopology
                  : metadata_edge_count > 0
                        ? PbctopoHardGraphSource::ValidatedMetadata
                        : PbctopoHardGraphSource::None;
    frame_report.hard_graph_source = frame_hard_graph_source;
    const PbctopoComponentLayoutObservation layout_observation =
        titan_pbctopo::build_pbctopo_component_layout_observation(
            components, base_atoms, local_edges, frame_hard_graph_source);
    titan_pbctopo::PbctopoSpatialLiftResult frame_spatial_lift;
    frame_report.spatial_lift_policy_version =
        titan_pbctopo::PBCTOPO_SPATIAL_LIFT_POLICY_VERSION;
    if (layout_observation.valid && frame_binding.valid) {
      vector<titan_pbctopo::PbctopoSpatialLiftAtom> lift_atoms;
      lift_atoms.reserve(base_atoms.size());
      for (const auto &atom : base_atoms) {
        lift_atoms.push_back(
            {atom.owner, static_cast<uint64_t>(atom.source_atom_id),
             {atom.x, atom.y, atom.z}});
      }
      vector<titan_pbctopo::PbctopoSpatialLiftEdge> lift_edges;
      lift_edges.reserve(layout_observation.hard_edges.size());
      for (const auto &edge : layout_observation.hard_edges)
        lift_edges.push_back({edge.atom_a, edge.atom_b});
      frame_spatial_lift = titan_pbctopo::build_pbctopo_spatial_lift(
          frame_binding.box_matrix, lift_atoms, lift_edges);
      frame_report.spatial_lift_identity = frame_spatial_lift.identity;
      frame_report.spatial_lift_ambiguous_hard_edges =
          frame_spatial_lift.ambiguous_hard_edges;
      frame_report.spatial_lift_cycle_residuals =
          frame_spatial_lift.cycle_residuals;
      frame_report.spatial_lift_valid = frame_spatial_lift.valid;
      if (!frame_spatial_lift.valid) {
        append_pbctopo_log_line(
            "[pbctopo] spatial lift unavailable at frame " +
            to_string(frame_report.frame) + ": reason=" +
            frame_spatial_lift.error);
      }
    }
    if (layout_observation.valid) {
      const auto existing_layout = find_if(
          emitted_component_layouts.begin(), emitted_component_layouts.end(),
          [&](const PbctopoComponentLayoutObservation &existing) {
            return existing.topology_epoch_id ==
                   layout_observation.topology_epoch_id;
          });
      if (existing_layout == emitted_component_layouts.end()) {
        frame_report.topology_epoch_id =
            layout_observation.topology_epoch_id;
        if (pbctopo_component_layout_report.is_open()) {
          write_pbctopo_component_layout_rows(
              pbctopo_component_layout_report, layout_observation);
        }
        if (pbctopo_hard_edge_report.is_open())
          write_pbctopo_hard_edge_rows(pbctopo_hard_edge_report,
                                       layout_observation);
        emitted_component_layouts.push_back(layout_observation);
      } else if (titan_pbctopo::same_pbctopo_component_layout_observation(
                     *existing_layout, layout_observation)) {
        frame_report.topology_epoch_id =
            layout_observation.topology_epoch_id;
      } else {
        append_pbctopo_log_line(
            "[pbctopo] topology epoch hash collision at frame " +
            to_string(frame_report.frame) +
            ": action=topology_epoch_unavailable");
      }
    } else {
      append_pbctopo_log_line(
          "[pbctopo] component membership artifact unavailable at frame " +
          to_string(frame_report.frame) +
          ": action=observation_provenance_incomplete");
    }
    const PbctopoComponentLayoutSignature component_layout =
        build_component_layout_signature(components);
    base_anchor_context.contact_hypotheses.attempted = true;
    if (frame_spatial_lift.valid &&
        frame_spatial_lift.ambiguous_hard_edges == 0 &&
        frame_spatial_lift.cycle_residuals == 0 &&
        frame_spatial_lift.atom_images.size() == base_atoms.size()) {
      vector<Int3> identity_offsets(components.size());
      const auto identity_assignment = make_pbctopo_image_assignment(
          identity_offsets, components, base_atoms);
      base_anchor_context.contact_hypotheses =
          titan_pbctopo::build_pbctopo_contact_hypotheses(
              span<const PbctopoAtom>(base_atoms),
              span<const PbctopoAnchorContactEdge>(
                  base_anchor_context.soft_observed_contact_edges),
              span<const size_t>(atom_component_index),
              span<const Int3>(frame_spatial_lift.atom_images),
              identity_assignment);
    } else {
      base_anchor_context.contact_hypotheses.error =
          "spatial lift is invalid, ambiguous, or inconsistent";
    }
    if (!base_anchor_context.contact_hypotheses.complete) {
      append_pbctopo_log_line(
          "[pbctopo] shift-hypothesis construction unavailable at frame " +
          to_string(frame_report.frame) + ": " +
          base_anchor_context.contact_hypotheses.error);
    }
    if (temporal_mode == PbctopoTemporalMode::Viterbi)
      temporal_contact_hypotheses[frame_idx] =
          base_anchor_context.contact_hypotheses;
    if (temporal_mode == PbctopoTemporalMode::Viterbi &&
        !atom_component_index.empty() &&
        components.size() <= numeric_limits<uint32_t>::max()) {
      vector<uint32_t> compact_layout;
      compact_layout.reserve(atom_component_index.size());
      bool layout_valid = true;
      for (size_t comp_idx : atom_component_index) {
        if (comp_idx >= components.size() ||
            comp_idx > numeric_limits<uint32_t>::max()) {
          layout_valid = false;
          break;
        }
        compact_layout.push_back(static_cast<uint32_t>(comp_idx));
      }
      if (layout_valid) {
        size_t layout_idx = 0;
        for (; layout_idx < temporal_component_layouts.size(); ++layout_idx) {
          if (temporal_component_layouts[layout_idx].component_count ==
                  components.size() &&
              temporal_component_layouts[layout_idx].atom_component_index ==
                  compact_layout) {
            break;
          }
        }
        if (layout_idx == temporal_component_layouts.size()) {
          vector<double> component_weights;
          component_weights.reserve(components.size());
          for (const auto &component : components)
            component_weights.push_back(static_cast<double>(component.size()));
          temporal_component_layouts.push_back(
              {std::move(compact_layout), component_layout.component_hashes,
               std::move(component_weights), components.size()});
        }
        temporal_frame_layout[frame_idx] = layout_idx;
      }
    }
    timing.component_split_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
    const size_t raw_component_count = components.size();
    const PbctopoComponentMode component_mode =
        select_pbctopo_component_mode(raw_component_count,
                                      exact_component_limit);
    const bool can_use_component_logic =
        hard_connectivity_edge_count > 0 &&
        component_mode != PbctopoComponentMode::Disabled;
    if (hard_connectivity_edge_count == 0 && raw_component_count > 1)
      ++timing.pbctopo_component_logic_disabled_no_hard_graph;
    const bool same_component_warm_layout =
        warm_state.has_component_offsets &&
        same_component_layout(components,
                              warm_state.previous_component_layout);
    if (warm_state.has_component_offsets && !same_component_warm_layout)
      ++timing.pbctopo_warm_layout_mismatches;
    if (!same_component_warm_layout)
      clear_component_trial_hints(warm_state);
    const int component_context_shell_radius =
        component_shell_radius_explicit
            ? configured_component_shell_radius
            : (components.size() == 2 ? 2 : 1);
    bool component_context_ready = false;
    auto ensure_component_context = [&]() {
      if (component_context_ready)
        return;
      const auto component_context_started = clock::now();
      component_atoms = apply_spatial_lift(
          base_atoms, frame_spatial_lift, this->m_mol[frame_idx].Box);
      if (can_use_component_logic) {
        shifts = build_topology_shifts(this->m_mol[frame_idx].Box,
                                       component_context_shell_radius);
        timing.component_image_shell_max_radius =
            max(timing.component_image_shell_max_radius,
                static_cast<size_t>(component_context_shell_radius));
        timing.component_image_shell_max_shift_count =
            max(timing.component_image_shell_max_shift_count, shifts.size());
      }
      component_context_ready = true;
      ++timing.component_context_builds;
      timing.component_context_s +=
          chrono::duration<double>(clock::now() - component_context_started).count();
    };
    auto extract_offsets_and_time = [&](const vector<PbctopoAtom> &source_atoms,
                                        const vector<Vec3> &points) {
      const auto extract_started = clock::now();
      auto offsets = extract_component_offsets_from_points(
          source_atoms, components, this->m_mol[frame_idx].Box, points);
      ++timing.component_offset_extracts;
      timing.component_offset_extract_s +=
          chrono::duration<double>(clock::now() - extract_started).count();
      return offsets;
    };
    const PbctopoLatticeMetric certification_lattice(
        this->m_mol[frame_idx].Box);
    if (h9_orbit_enabled &&
        h9_component_representative_report.is_open()) {
      ensure_component_context();
      const vector<Int3> identity_offsets(components.size());
      const PbctopoImageAssignment representative_layout =
          make_pbctopo_image_assignment(identity_offsets, components,
                                         component_atoms);
      double v1[3]{}, v2[3]{}, v3[3]{};
      this->m_mol[frame_idx].Box.get_v1(v1);
      this->m_mol[frame_idx].Box.get_v2(v2);
      this->m_mol[frame_idx].Box.get_v3(v3);
      const auto wrap_unit_interval = [](double value) {
        double wrapped = value - floor(value);
        if (wrapped >= 1.0)
          wrapped = 0.0;
        if (wrapped < 0.0)
          wrapped += 1.0;
        return wrapped == 0.0 ? 0.0 : wrapped;
      };
      for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
        const auto &component = components[comp_idx];
        bool use_mass = !component.empty();
        double total_mass = 0.0;
        for (const size_t atom_idx : component) {
          if (atom_idx >= component_atoms.size() ||
              !component_atoms[atom_idx].has_atomic_mass ||
              !isfinite(component_atoms[atom_idx].atomic_mass) ||
              component_atoms[atom_idx].atomic_mass <= 0.0) {
            use_mass = false;
            break;
          }
          total_mass += component_atoms[atom_idx].atomic_mass;
        }
        if (!isfinite(total_mass) || total_mass <= 0.0)
          use_mass = false;
        Vec3 representative{};
        double denominator = use_mass ? total_mass
                                      : static_cast<double>(component.size());
        bool representative_valid = representative_layout.valid &&
                                    !component.empty() &&
                                    isfinite(denominator) &&
                                    denominator > 0.0 &&
                                    certification_lattice.valid();
        if (representative_valid) {
          for (const size_t atom_idx : component) {
            if (atom_idx >= component_atoms.size()) {
              representative_valid = false;
              break;
            }
            const double weight =
                use_mass ? component_atoms[atom_idx].atomic_mass : 1.0;
            representative.x += weight * component_atoms[atom_idx].x;
            representative.y += weight * component_atoms[atom_idx].y;
            representative.z += weight * component_atoms[atom_idx].z;
          }
        }
        if (representative_valid) {
          representative.x /= denominator;
          representative.y /= denominator;
          representative.z /= denominator;
          representative_valid = isfinite(representative.x) &&
                                 isfinite(representative.y) &&
                                 isfinite(representative.z);
        }
        array<double, 3> scaled{};
        array<double, 3> wrapped_cartesian{};
        if (representative_valid) {
          scaled = certification_lattice.cartesian_to_scaled(
              representative.x, representative.y, representative.z);
          representative_valid = isfinite(scaled[0]) && isfinite(scaled[1]) &&
                                 isfinite(scaled[2]);
        }
        if (representative_valid) {
          scaled[0] = wrap_unit_interval(scaled[0]);
          scaled[1] = wrap_unit_interval(scaled[1]);
          scaled[2] = wrap_unit_interval(scaled[2]);
          wrapped_cartesian = certification_lattice.scaled_to_cartesian(
              scaled[0], scaled[1], scaled[2]);
          representative_valid = isfinite(wrapped_cartesian[0]) &&
                                 isfinite(wrapped_cartesian[1]) &&
                                 isfinite(wrapped_cartesian[2]);
        }
        const uint64_t component_id =
            representative_layout.valid &&
                    comp_idx < representative_layout.component_ids.size()
                ? representative_layout.component_ids[comp_idx]
                : 0;
        h9_component_representative_report
            << "h9_component_representative_v1," << frame_report.frame << ','
            << this->m_mol[frame_idx].time_of_snap << ',' << comp_idx << ','
            << component_id << ',' << component.size() << ','
            << (use_mass ? "mass_weighted" : "atom_count_mean") << ','
            << (use_mass ? total_mass : 0.0) << ','
            << static_cast<int>(representative_valid) << ',' << v1[0] << ','
            << v1[1] << ',' << v1[2] << ',' << v2[0] << ',' << v2[1] << ','
            << v2[2] << ',' << v3[0] << ',' << v3[1] << ',' << v3[2] << ',';
        if (representative_valid) {
          h9_component_representative_report
              << scaled[0] << ',' << scaled[1] << ',' << scaled[2] << ','
              << wrapped_cartesian[0] << ',' << wrapped_cartesian[1] << ','
              << wrapped_cartesian[2];
        }
        h9_component_representative_report << '\n';
      }
    }
    int component_certificate_domain_complete_cache = -1;
    auto component_certificate_domain_complete = [&]() {
      if (component_certificate_domain_complete_cache >= 0)
        return component_certificate_domain_complete_cache != 0;
      ensure_component_context();
      bool complete = certification_lattice.valid() &&
                      component_atoms.size() == base_atoms.size() &&
                      atom_component_index.size() == base_atoms.size() &&
                      !components.empty();
      if (complete) {
        vector<unsigned char> membership_count(base_atoms.size(), 0);
        for (size_t comp_idx = 0; comp_idx < components.size() && complete;
             ++comp_idx) {
          for (size_t atom_idx : components[comp_idx]) {
            if (atom_idx >= base_atoms.size() ||
                atom_component_index[atom_idx] != comp_idx ||
                membership_count[atom_idx] != 0) {
              complete = false;
              break;
            }
            membership_count[atom_idx] = 1;
          }
        }
        complete = complete &&
                   all_of(membership_count.begin(), membership_count.end(),
                          [](unsigned char count) { return count == 1; });
      }
      if (complete) {
        for (const auto &edge : local_edges) {
          if (!titan_pbctopo::pbctopo_edge_is_hard_connectivity(edge.kind))
            continue;
          if (edge.a >= atom_component_index.size() ||
              edge.b >= atom_component_index.size() ||
              atom_component_index[edge.a] != atom_component_index[edge.b]) {
            complete = false;
            break;
          }
        }
      }
      if (complete) {
        vector<Vec3> component_identity_points;
        component_identity_points.reserve(component_atoms.size());
        for (const auto &atom : component_atoms)
          component_identity_points.push_back({atom.x, atom.y, atom.z});
        const auto image_equations = evaluate_pbctopo_image_equations(
            base_atoms, component_identity_points, local_edges,
            certification_lattice);
        vector<Int3> identity_offsets(components.size());
        const auto identity_assignment =
            make_pbctopo_image_assignment(identity_offsets, components,
                                           component_atoms);
        const auto assignment_points =
            titan_pbctopo::evaluate_pbctopo_assignment_point_consistency(
                component_atoms, component_identity_points,
                atom_component_index, identity_assignment,
                certification_lattice);
        complete = identity_assignment.valid &&
                   image_equations.topology_edge_residuals == 0 &&
                   image_equations.metadata_edge_image_residuals == 0 &&
                   image_equations.metadata_edge_distance_residuals == 0 &&
                   image_equations.metadata_edges_checked >=
                       metadata_edge_count &&
                   image_equations.atom_image_residuals == 0 &&
                   image_equations.topology_cycle_residuals == 0 &&
                   image_equations.topological_winding_rank == 0 &&
                   assignment_points.input_valid &&
                   assignment_points.point_residuals == 0;
        frame_report.fixed_certificate_audit =
            evaluate_pbctopo_fixed_certificate_invariants(
                base_atoms, component_identity_points, atom_component_index,
                local_edges, explicit_required_contacts,
                steric_exclusion_set, runtime_cutoffs.clash_cutoff,
                certification_lattice, &timing);
        complete =
            complete && frame_report.fixed_certificate_audit.passed();
      }
      component_certificate_domain_complete_cache = complete ? 1 : 0;
      return complete;
    };
    auto make_frame_component_search_callbacks = [&]() {
      ensure_component_context();
      auto callbacks = make_component_search_callbacks();
      callbacks.input_atoms_cover_certificate_scope =
          component_atoms.size() == base_atoms.size();
      callbacks.hard_certificate_domain_complete =
          component_certificate_domain_complete();
      return callbacks;
    };
    size_t report_soft_valid_candidates = 0;
    size_t report_nonclashing_candidates = 0;
    vector<PbctopoCutCandidate> report_certified_candidates;
    // Failed hard-certificate evaluations are audit records, not members of
    // the certified band.  Neither selection nor certified counts use them.
    vector<PbctopoCutCandidate> report_rejected_candidates;
    vector<titan_pbctopo::PbctopoCandidateEvidenceAuditSnapshot>
        evidence_relation_audit_snapshots;
    string evidence_relation_cache_error;
    bool evidence_relation_frame_written = false;
    const bool evidence_relation_export_enabled =
        pbctopo_candidate_evidence_summary_report.is_open() &&
        pbctopo_candidate_evidence_relation_report.is_open();
    auto cache_candidate_evidence_audit =
        [&](const PbctopoImageAssignment &assignment,
            const PbctopoSelectedEvidenceAudit &audit) {
          if (!evidence_relation_export_enabled ||
              !evidence_relation_cache_error.empty()) {
            return;
          }
          const auto cached =
              upsert_pbctopo_candidate_evidence_audit_snapshot(
                  evidence_relation_audit_snapshots, assignment, audit);
          if (!cached)
            evidence_relation_cache_error = cached.error();
        };
    auto record_report_certified_candidate =
        [&](const PbctopoCutCandidate &candidate) {
          if (!candidate.valid || !candidate.assignment.valid) {
            return;
          }
          if (!candidate.hard_feasibility.certified()) {
            // Export separately, with the failed hard predicates, so a
            // consumer can distinguish a rejection from an unevaluated image.
            const bool known = any_of(
                report_rejected_candidates.begin(),
                report_rejected_candidates.end(),
                [&](const PbctopoCutCandidate &other) {
                  return same_pbctopo_image_assignment(other.assignment,
                                                        candidate.assignment);
                });
            if (!known)
              report_rejected_candidates.push_back(candidate);
            return;
          }
          auto existing = find_if(
              report_certified_candidates.begin(),
              report_certified_candidates.end(),
              [&](const PbctopoCutCandidate &other) {
                return same_pbctopo_image_assignment(other.assignment,
                                                      candidate.assignment);
              });
          if (existing == report_certified_candidates.end()) {
            report_certified_candidates.push_back(candidate);
          } else {
            const PbctopoCutCandidate prior = *existing;
            const bool equivalence_known =
                existing->search_equivalence_known ||
                candidate.search_equivalence_known;
            const bool equivalent_to_search_best =
                (existing->search_equivalence_known &&
                 existing->equivalent_to_search_best) ||
                (candidate.search_equivalence_known &&
                 candidate.equivalent_to_search_best);
            if (better_cut_candidate(candidate, *existing))
              *existing = candidate;
            existing->search_equivalence_known = equivalence_known;
            existing->equivalent_to_search_best =
                equivalent_to_search_best;
            merge_pbctopo_search_domain_provenance(*existing, prior);
            merge_pbctopo_search_domain_provenance(*existing, candidate);
          }
          sort(report_certified_candidates.begin(),
               report_certified_candidates.end(), better_cut_candidate);
        };
    // A candidate's evidence audit in two halves, so that the band loop can
    // run the first on executor workers: compute reads only frame constants;
    // apply writes the evidence cache and the candidate.  nullopt is the input
    // the serial code turned away before auditing.
    using CandidateEvidenceAudit =
        std::expected<titan_pbctopo::PbctopoSelectedEvidenceAudit, string>;
    auto compute_candidate_evidence =
        [&](const PbctopoCutCandidate &candidate,
            const vector<Vec3> &points) -> optional<CandidateEvidenceAudit> {
          if (!candidate.assignment.valid || points.size() != base_atoms.size())
            return nullopt;
          return titan_pbctopo::audit_pbctopo_selected_hypotheses(
              span<const PbctopoAtom>(base_atoms), span<const Vec3>(points),
              base_anchor_context.contact_hypotheses, candidate.assignment, true,
              explicit_evidence_component_edges);
        };
    auto apply_candidate_evidence =
        [&](PbctopoCutCandidate &candidate,
            const optional<CandidateEvidenceAudit> &computed) -> bool {
          if (!computed || !*computed || !(*computed)->evaluated)
            return false;
          const auto &audit = **computed;
          cache_candidate_evidence_audit(candidate.assignment, audit);
          candidate.evidence_relative_relation_signature =
              candidate.assignment.signature;
          candidate.evidence_compatible_hypothesis_hash =
              audit.selected_hypothesis_hash;
          candidate.evidence_compatible_pair_hash =
              audit.selected_compatible_pair_hash;
          candidate.evidence_compatible_hypothesis_count =
              audit.compatible_hypothesis_count;
          candidate.evidence_supported_relation_count =
              audit.supported_relation_count;
          candidate.evidence_compatible_contact_count =
              audit.selected_compatible_contact_count;
          candidate.evidence_no_support_relation_count =
              audit.no_supported_contact_relation_count;
          return audit.state != PbctopoEvidenceState::Contradicted;
        };
    auto annotate_candidate_evidence =
        [&](PbctopoCutCandidate &candidate,
            const vector<Vec3> &points) -> bool {
          return apply_candidate_evidence(
              candidate, compute_candidate_evidence(candidate, points));
        };
    // A candidate's certification in two halves.  compute does the geometry
    // and touches nothing but the candidate and its work record, so the band
    // loop runs it on TitanExecutor workers; apply then makes, on the frame's
    // thread and in band order, every write the serial code made: counters,
    // lattice failures, the rejection ledger, the evidence cache and the
    // certified band.  certify_candidate_for_output is the two back to back.
    struct CandidateCertificationWork {
      bool assignment_valid = false;
      bool lattice_failure = false;
      bool certified = false;
      // Lattice queries from the start of compute to the image equations
      // (the serial failure test), and over all of compute when it ran on a
      // worker, whose thread-local counts must reach the frame's thread.
      titan_pbctopo::PbctopoLatticeQueryAudit certification_lattice_delta;
      titan_pbctopo::PbctopoLatticeQueryAudit worker_lattice_delta;
      bool ran_on_worker = false;
      PbctopoImageEquationResiduals image_equations;
      titan_pbctopo::PbctopoAssignmentPointResiduals assignment_points;
      titan_pbctopo::PbctopoStericAudit steric_audit;
      titan_pbctopo::PbctopoRequiredContactResiduals required_contacts;
      // The only timing counters the steric scan touches (pbctopo_atoms_clash).
      size_t vdw_pair_checks = 0;
      size_t vdw_radius_pairs = 0;
      size_t vdw_radius_fallback_pairs = 0;
      size_t vdw_explicit_zero_lj_exempt_pairs = 0;
      optional<CandidateEvidenceAudit> evidence;
    };
    auto compute_candidate_certification =
        [&](CandidateCertificationWork &work, PbctopoCutCandidate &candidate,
            const vector<Vec3> &points, const vector<Int3> *known_offsets,
            bool on_worker) {
          const auto lattice_before =
              titan_pbctopo::pbctopo_lattice_query_audit();
          auto finish = [&]() {
            if (!on_worker)
              return;
            work.ran_on_worker = true;
            work.worker_lattice_delta =
                titan_pbctopo::pbctopo_lattice_query_audit_delta(
                    lattice_before,
                    titan_pbctopo::pbctopo_lattice_query_audit());
          };
          // A worker runs only with the component context built and the
          // offsets given, so these two frame-state writers stay serial.
          if (!on_worker)
            ensure_component_context();
          vector<Int3> extracted_offsets;
          const vector<Int3> *offsets = known_offsets;
          if (!on_worker &&
              (offsets == nullptr || offsets->size() != components.size())) {
            extracted_offsets =
                extract_offsets_and_time(component_atoms, points);
            offsets = &extracted_offsets;
          }
          // Extracted and search offsets are rebased; certify the points in
          // the gauge that placed them, or every atom reads as a residual.
          vector<Int3> aligned_offsets = *offsets;
          if (align_component_offsets_to_points(component_atoms, components,
                                                certification_lattice, points,
                                                aligned_offsets)) {
            offsets = &aligned_offsets;
          }
          candidate.assignment = make_pbctopo_image_assignment(
              *offsets, components, component_atoms);
          candidate.hard_feasibility.certificate_atoms_total =
              base_atoms.size();
          candidate.hard_feasibility.hard_graph_atoms_covered =
              hard_graph_atoms_covered;
          candidate.hard_feasibility.topology_bond_atoms_covered =
              topology_bond_atoms_covered;
          candidate.hard_feasibility.metadata_only_atoms =
              metadata_only_atoms;
          candidate.hard_feasibility.hard_connectivity_edges_available =
              hard_connectivity_edge_count;
          candidate.hard_feasibility.topology_edges_available =
              topology_edge_count;
          candidate.hard_feasibility.metadata_edges_available =
              metadata_edge_count;
          if (!candidate.assignment.valid) {
            candidate.hard_feasibility = {};
            candidate.hard_feasibility.evaluated = candidate.valid;
            finish();
            return;
          }
          work.assignment_valid = true;
          work.image_equations = evaluate_pbctopo_image_equations(
              base_atoms, points, local_edges, certification_lattice);
          work.certification_lattice_delta =
              titan_pbctopo::pbctopo_lattice_query_audit_delta(
                  lattice_before, titan_pbctopo::pbctopo_lattice_query_audit());
          if (work.certification_lattice_delta.failures > 0) {
            work.lattice_failure = true;
            candidate.hard_feasibility = {};
            candidate.hard_feasibility.evaluated = candidate.valid;
            finish();
            return;
          }
          work.assignment_points =
              titan_pbctopo::evaluate_pbctopo_assignment_point_consistency(
                  component_atoms, points, atom_component_index,
                  candidate.assignment, certification_lattice);
          PbctopoTimingStats steric_timing;
          work.steric_audit = titan_pbctopo::evaluate_pbctopo_final_sterics(
              base_atoms, points, atom_component_index, steric_exclusion_set,
              runtime_cutoffs.clash_cutoff, &steric_timing,
              &certification_lattice);
          work.vdw_pair_checks = steric_timing.pbctopo_vdw_pair_checks;
          work.vdw_radius_pairs = steric_timing.pbctopo_vdw_radius_pairs;
          work.vdw_radius_fallback_pairs =
              steric_timing.pbctopo_vdw_radius_fallback_pairs;
          work.vdw_explicit_zero_lj_exempt_pairs =
              steric_timing.pbctopo_vdw_explicit_zero_lj_exempt_pairs;
          const auto &steric_audit = work.steric_audit;
          candidate.ultra_clash = steric_audit.hard_clashes != 0;
          // The candidate's min_pair_d2 and clash atoms describe its closest
          // pair (analyze_cut_candidate_ultraclash sets them that way), so
          // they are refreshed from the audit's closest pair, not from the
          // pair with the smallest margin.  The margin itself is carried in
          // hard_feasibility.min_steric_margin_A.
          if (isfinite(steric_audit.min_pair_d2) &&
              steric_audit.min_pair_d2 < candidate.min_pair_d2) {
            candidate.min_pair_d2 = steric_audit.min_pair_d2;
            candidate.clash_atom_a = steric_audit.min_pair_atom_a;
            candidate.clash_atom_b = steric_audit.min_pair_atom_b;
          }
          work.required_contacts =
              titan_pbctopo::evaluate_pbctopo_required_contacts(
                  base_atoms, points, local_edges, explicit_required_contacts);
          candidate.geometry_classification =
              titan_pbctopo::classify_pbctopo_final_geometry(
                  points, certification_lattice, 1.0e-6,
                  &work.image_equations);
          work.certified = certify_pbctopo_candidate(
              candidate, work.image_equations, work.assignment_points,
              work.steric_audit, work.required_contacts);
          if (work.certified)
            work.evidence = compute_candidate_evidence(candidate, points);
          finish();
        };
    auto apply_candidate_certification =
        [&](const CandidateCertificationWork &work,
            PbctopoCutCandidate &candidate) {
          if (work.ran_on_worker)
            titan_pbctopo::pbctopo_add_lattice_query_audit(
                work.worker_lattice_delta);
          ++timing.pbctopo_certification_attempts;
          if (!work.assignment_valid) {
            ++timing.pbctopo_certification_assignment_rejects;
            return false;
          }
          frame_fatal_lattice_query_failures +=
              work.certification_lattice_delta.failures;
          if (work.lattice_failure) {
            ++timing.pbctopo_certification_hard_rejects;
            return false;
          }
          timing.pbctopo_vdw_pair_checks += work.vdw_pair_checks;
          timing.pbctopo_vdw_radius_pairs += work.vdw_radius_pairs;
          timing.pbctopo_vdw_radius_fallback_pairs +=
              work.vdw_radius_fallback_pairs;
          timing.pbctopo_vdw_explicit_zero_lj_exempt_pairs +=
              work.vdw_explicit_zero_lj_exempt_pairs;
          ++timing.pbctopo_geometry_classifications;
          if (candidate.geometry_classification.topological_winding_rank > 0) {
            ++timing.pbctopo_winding_rejected_candidates;
            if (candidate.geometry_classification.geometry_class ==
                titan_pbctopo::PbctopoGeometryClass::ExtendedAndWinding) {
              ++timing.pbctopo_extended_winding_rejected_candidates;
            }
          }
          const auto &image_equations = work.image_equations;
          timing.pbctopo_certification_topology_edges_checked +=
              image_equations.topology_edges_checked;
          timing.pbctopo_certification_ambiguous_topology_edges +=
              image_equations.ambiguous_topology_edges;
          timing.pbctopo_certification_topology_edge_residuals +=
              image_equations.topology_edge_residuals;
          timing.pbctopo_certification_topology_cycle_residuals +=
              image_equations.topology_cycle_residuals;
          timing.pbctopo_certification_atom_image_residuals +=
              image_equations.atom_image_residuals;
          timing.pbctopo_certification_assignment_point_residuals +=
              work.assignment_points.point_residuals;
          timing.pbctopo_certification_steric_pair_checks +=
              work.steric_audit.pair_checks;
          timing.pbctopo_certification_steric_clashes +=
              work.steric_audit.hard_clashes;
          timing.pbctopo_certification_required_contacts_checked +=
              work.required_contacts.contacts_checked;
          timing.pbctopo_certification_required_contacts_lost +=
              work.required_contacts.contacts_lost;
          timing.pbctopo_certification_explicit_interface_contacts_checked +=
              work.required_contacts.explicit_contacts_checked;
          timing.pbctopo_certification_explicit_interface_contacts_lost +=
              work.required_contacts.explicit_contacts_lost;
          if (!work.certified) {
            ++timing.pbctopo_certification_hard_rejects;
            // Preserve the rejection in the separate audit ledger.
            record_report_certified_candidate(candidate);
            return false;
          }
          ++timing.pbctopo_certification_successes;
          if (apply_candidate_evidence(candidate, work.evidence))
            record_report_certified_candidate(candidate);
          return true;
        };
    auto certify_candidate_for_output =
        [&](PbctopoCutCandidate &candidate, const vector<Vec3> &points,
            const vector<Int3> *known_offsets = nullptr) {
          CandidateCertificationWork work;
          compute_candidate_certification(work, candidate, points,
                                          known_offsets, false);
          return apply_candidate_certification(work, candidate);
        };
    auto record_temporal_certified_candidate =
        [&](const PbctopoCutCandidate &source_candidate,
            const vector<Vec3> &candidate_points,
            const vector<Int3> *known_offsets = nullptr) {
          if (temporal_mode != PbctopoTemporalMode::Viterbi ||
              !source_candidate.valid || source_candidate.ultra_clash ||
              candidate_points.size() != base_atoms.size()) {
            return;
          }
          PbctopoCutCandidate candidate = source_candidate;
          if (!candidate.hard_feasibility.certified() ||
              !candidate.assignment.valid) {
            if (!certify_candidate_for_output(candidate, candidate_points,
                                              known_offsets)) {
              return;
            }
          }
          if (!annotate_candidate_evidence(candidate, candidate_points))
            return;
          if (candidate.evidence_relative_relation_signature !=
              candidate.assignment.signature)
            return;
          record_report_certified_candidate(candidate);
          auto &frame_candidates = temporal_cut_candidates[frame_idx];
          const auto existing = find_if(
              frame_candidates.begin(), frame_candidates.end(),
              [&](const PbctopoCutCandidate &other) {
                return same_pbctopo_image_assignment(other.assignment,
                                                      candidate.assignment);
              });
          if (existing != frame_candidates.end()) {
            // The first copy keeps its scores, which feed the Viterbi costs,
            // but the search provenance of a later copy of the same
            // assignment is merged as the band and report ledger do. The
            // exported search domain is frame-constant, so a virtual copy
            // recorded before the search's must not hide the search audit.
            const bool equivalence_known =
                existing->search_equivalence_known ||
                candidate.search_equivalence_known;
            const bool equivalent_to_search_best =
                (existing->search_equivalence_known &&
                 existing->equivalent_to_search_best) ||
                (candidate.search_equivalence_known &&
                 candidate.equivalent_to_search_best);
            existing->search_equivalence_known = equivalence_known;
            existing->equivalent_to_search_best = equivalent_to_search_best;
            merge_pbctopo_search_domain_provenance(*existing, candidate);
            ++timing.pbctopo_temporal_assignment_deduped;
            return;
          }
          frame_candidates.push_back(std::move(candidate));
          sort(frame_candidates.begin(), frame_candidates.end(),
               better_cut_candidate);
          const size_t keep =
              static_cast<size_t>(Arguments.pbc.pbctopo_temporal_k);
          if (frame_candidates.size() > keep)
            frame_candidates.resize(keep);
          ++timing.pbctopo_temporal_certified_candidates;
        };
    auto apply_cut_candidate_timed = [&](const vector<Vec3> &points) {
      const auto apply_started = clock::now();
      apply_cut_candidate(base_atoms, points, *this, mol2, frame_idx);
      ++timing.apply_calls;
      timing.apply_s +=
          chrono::duration<double>(clock::now() - apply_started).count();
    };

    constexpr size_t kProjectionTopSeeds = 24;
    constexpr size_t kProjectionKeepCandidates = 3;
    warm_points.reserve(base_atoms.size());
    coarse_warm_points.reserve(projection_atoms.size());
    vector<Vec3> prefetched_safe_points;
    prefetched_safe_points.reserve(base_atoms.size());
    PbctopoCutCandidate prefetched_safe_candidate;
    PbctopoFrameMode prefetched_safe_mode = PbctopoFrameMode::None;
    PbctopoCutCandidate best_verified_virtual_candidate;
    PbctopoCutCandidate best_virtual_geometry_reference;
    PbctopoCutCandidate best_coarse_virtual_geometry_seed;
    vector<Vec3> split_rescue_points;
    split_rescue_points.reserve(base_atoms.size());
    PbctopoOwnerPairSplitDetail split_rescue_detail;
    string split_rescue_source;
    // Owner COM separation is diagnostic-only; it must not drive rescue.
    const bool split_rescue_available = false;
    auto validate_rescue_candidate_points =
        [&](const vector<Vec3> &candidate_points,
            const PbctopoCutCandidate *incumbent_cut,
            PbctopoCutCandidate &candidate_out) {
          if (candidate_points.size() != base_atoms.size())
            return false;
          size_t split_owner_pairs = 0;
          double split_gap = 0.0;
          PbctopoOwnerPairSplitDetail detail;
          candidate_has_split_contacting_owners(
              candidate_points, base_atoms, this->m_mol[frame_idx].Box,
              local_hard_required_owner_pairs, split_owner_pairs, split_gap,
              &detail);

          const vector<Int3> rescue_offsets = extract_component_offsets_from_points(
              base_atoms, components, this->m_mol[frame_idx].Box,
              candidate_points);
          const double rescue_shift_norm2 = pbctopo_component_shift_metric(
              this->m_mol[frame_idx].Box, rescue_offsets);

          const PbctopoLatticeMetric rescue_lattice(
              this->m_mol[frame_idx].Box);
          double min_sx = numeric_limits<double>::infinity();
          double min_sy = numeric_limits<double>::infinity();
          double min_sz = numeric_limits<double>::infinity();
          double max_sx = -numeric_limits<double>::infinity();
          double max_sy = -numeric_limits<double>::infinity();
          double max_sz = -numeric_limits<double>::infinity();
          for (const auto &point : candidate_points) {
            const auto scaled = rescue_lattice.cartesian_to_scaled(
                point.x, point.y, point.z);
            const double sx = scaled[0];
            const double sy = scaled[1];
            const double sz = scaled[2];
            min_sx = min(min_sx, sx);
            min_sy = min(min_sy, sy);
            min_sz = min(min_sz, sz);
            max_sx = max(max_sx, sx);
            max_sy = max(max_sy, sy);
            max_sz = max(max_sz, sz);
          }
          auto normalize_rescue_center = [](double center) {
            if (!isfinite(center))
              return 1.5;
            center = fmod(center - 0.5, 1.0);
            if (center < 0.0)
              center += 1.0;
            return clamp_virtual_box_center(center + 0.5);
          };
          const double rescue_cut_x =
              normalize_rescue_center(0.5 * (min_sx + max_sx));
          const double rescue_cut_y =
              normalize_rescue_center(0.5 * (min_sy + max_sy));
          const double rescue_cut_z =
              normalize_rescue_center(0.5 * (min_sz + max_sz));

          const PointStats rescue_stats = build_point_stats(candidate_points);
          candidate_out = score_cut_candidate(
              component_scoring_edges, candidate_points, rescue_cut_x,
              rescue_cut_y, rescue_cut_z, 0.0, rescue_stats,
              rescue_shift_norm2);
          double continuity_path = numeric_limits<double>::infinity();
          double continuity_max_d2 = numeric_limits<double>::infinity();
          double continuity_min_margin_A =
              numeric_limits<double>::quiet_NaN();
          size_t continuity_margin_atom_a = numeric_limits<size_t>::max();
          size_t continuity_margin_atom_b = numeric_limits<size_t>::max();
          double anchor_mst2 = numeric_limits<double>::infinity();
          double anchor_max_d2 = numeric_limits<double>::infinity();
          PbctopoAnchorGraphScore anchor_graph_score;
          const bool rescue_ok =
              candidate_out.valid &&
              continuity_metrics_ok(candidate_points,
                                    trajectory_context.continuity_pairs,
                                    runtime_cutoffs.continuity_cutoff,
                                    continuity_path, continuity_max_d2,
                                    continuity_min_margin_A,
                                    continuity_margin_atom_a,
                                    continuity_margin_atom_b,
                                    incumbent_cut) &&
              anchor_graph_metrics_enhanced(
                  candidate_points, trajectory_context.continuity_anchor_indices,
                  trajectory_context.continuity_pairs, anchor_mst2,
                  anchor_max_d2, anchor_graph_score, incumbent_cut,
                  PbctopoRefineScorePhase::Exact, anchor_backend, nullptr,
                  &base_anchor_context);
          if (!rescue_ok)
            return false;

          candidate_out.continuity_path = continuity_path;
          candidate_out.continuity_max_d2 = continuity_max_d2;
          candidate_out.continuity_min_margin_A = continuity_min_margin_A;
          candidate_out.continuity_margin_atom_a = continuity_margin_atom_a;
          candidate_out.continuity_margin_atom_b = continuity_margin_atom_b;
          candidate_out.split_owner_pairs = split_owner_pairs;
          candidate_out.max_owner_raw_minus_minimg_delta = split_gap;
          candidate_out.anchor_mst2 = anchor_mst2;
          candidate_out.anchor_max_d2 = anchor_max_d2;
          candidate_out.anchor_graph = anchor_graph_score;
          analyze_cut_candidate_ultraclash(
              candidate_points, steric_exclusion_set, candidate_out,
              runtime_cutoffs.clash_cutoff, &base_atoms,
              &atom_component_index);
          return !candidate_out.ultra_clash;
        };
    if (warm_state.has_projection_candidate) {
      const auto early_projection_started = clock::now();
      const PbctopoCutCandidate warm_projection_reference =
          warm_state.previous_projection_candidate;
      bool warm_projection_accepted = false;
      bool warm_projection_full_evaluated = false;
      PbctopoCutCandidate early_projection;
      PbctopoCutCandidate early_projection_coarse = evaluate_virtual_box_candidate(
          projection_atoms, projection_edges, projection_edge_set,
          translation_cache, this->m_mol[frame_idx].Box,
          warm_state.previous_projection_candidate.cut_x,
          warm_state.previous_projection_candidate.cut_y,
          warm_state.previous_projection_candidate.cut_z,
          warm_state.previous_projection_candidate.gap_score, coarse_warm_points,
          runtime_cutoffs.clash_cutoff, &projection_continuity_pairs,
          &projection_anchor_indices, &projection_hard_required_owner_pairs,
          runtime_cutoffs.continuity_cutoff, nullptr,
          skip_projection_candidate_clash, numeric_limits<double>::quiet_NaN(),
          nullptr, &projection_anchor_context,
          PbctopoVirtualSeedSource::Previous);
      if (skip_projection_candidate_clash)
        ++timing.projection_clash_skip_evals;
      if (early_projection_coarse.valid && !early_projection_coarse.ultra_clash) {
        warm_projection_full_evaluated = true;
        early_projection = evaluate_virtual_box_candidate(
            base_atoms, local_edges, steric_exclusion_set, translation_cache,
            this->m_mol[frame_idx].Box,
            warm_state.previous_projection_candidate.cut_x,
            warm_state.previous_projection_candidate.cut_y,
            warm_state.previous_projection_candidate.cut_z,
            warm_state.previous_projection_candidate.gap_score, warm_points,
            runtime_cutoffs.clash_cutoff, &trajectory_context.continuity_pairs,
            &trajectory_context.continuity_anchor_indices,
            &local_hard_required_owner_pairs, runtime_cutoffs.continuity_cutoff,
            nullptr, false, this->m_mol[frame_idx].time_of_snap,
            "warm_projection", &base_anchor_context,
            PbctopoVirtualSeedSource::Previous, &atom_component_index);
        if (early_projection.valid && !early_projection.ultra_clash &&
          candidate_close_enough_to_skip_search(
              early_projection, warm_projection_reference)) {
          warm_projection_accepted = true;
          ++timing.early_projection_used;
          warm_state.previous_projection_candidate = early_projection;
          warm_state.has_projection_candidate = true;
          if (better_cut_candidate(early_projection,
                                   best_verified_virtual_candidate)) {
            best_verified_virtual_candidate = early_projection;
          }
          if (better_cut_candidate(early_projection, prefetched_safe_candidate)) {
            prefetched_safe_candidate = early_projection;
            prefetched_safe_points = warm_points;
            prefetched_safe_mode = PbctopoFrameMode::EarlyProjection;
            frame_report.mode = PbctopoFrameMode::EarlyProjection;
          }
        }
      }
      if (!warm_projection_accepted) {
        ++timing.pbctopo_warm_projection_rejects;
        const char *reason =
            !early_projection_coarse.valid
                ? "coarse_invalid"
                : (early_projection_coarse.ultra_clash
                       ? "coarse_ultraclash"
                       : (!warm_projection_full_evaluated
                              ? "full_not_evaluated"
                              : candidate_skip_search_reject_reason(
                                    early_projection,
                                    warm_projection_reference)));
        append_pbctopo_log_line(
            "[pbctopo] warm projection rejected at frame " +
            to_string(static_cast<long long>(
                this->m_mol[frame_idx].time_of_snap)) +
            ": reason=" + (reason != nullptr ? reason : "unknown"));
      }
      timing.early_projection_s +=
          chrono::duration<double>(clock::now() - early_projection_started)
              .count();
    }
    if (can_use_component_logic && warm_state.has_component_offsets &&
        same_component_layout(components,
                              warm_state.previous_component_layout)) {
      const auto warm_component_started = clock::now();
      ensure_component_context();
      vector<Int3> warm_offsets = warm_state.previous_component_offsets;
      if (!rebase_component_offsets(warm_offsets,
                                    this->m_mol[frame_idx].Box)) {
        warm_offsets.clear();
      }
      PbctopoCutCandidate early_component = score_component_offset_solution(
          component_atoms, components, this->m_mol[frame_idx].Box, warm_offsets,
          local_edges, steric_exclusion_set, runtime_cutoffs.clash_cutoff,
          trajectory_context.continuity_pairs,
          trajectory_context.continuity_anchor_indices,
          runtime_cutoffs.continuity_cutoff, warm_points,
          make_component_refine_callbacks(),
          &warm_state.previous_projection_candidate, nullptr, true,
          PbctopoRefineScorePhase::Exact, nullptr, nullptr, nullptr,
          anchor_backend, &base_anchor_context);
      PbctopoOwnerPairSplitDetail early_component_split_detail;
      size_t early_component_split_owner_pairs = 0;
      double early_component_split_gap = 0.0;
      const bool early_component_split =
          early_component.valid && !early_component.ultra_clash &&
          candidate_has_split_contacting_owners(
              warm_points, base_atoms, this->m_mol[frame_idx].Box,
              local_hard_required_owner_pairs, early_component_split_owner_pairs,
              early_component_split_gap, &early_component_split_detail);
      early_component.split_owner_pairs = early_component_split_owner_pairs;
      early_component.max_owner_raw_minus_minimg_delta =
          early_component_split_gap;
      if (early_component_split && early_component_split_detail.valid) {
        append_pbctopo_log_line(
            "[pbctopo] observe owner_com_image_gap at frame " +
            to_string(static_cast<long long>(this->m_mol[frame_idx].time_of_snap)) +
            " via warm_component: owners=" +
            to_string(early_component_split_detail.owner_a) + "-" +
            to_string(early_component_split_detail.owner_b) +
            ", raw_delta_scaled=" +
            format_vec3(early_component_split_detail.raw_delta_scaled) +
            ", min_image_delta_scaled=" +
            format_vec3(early_component_split_detail.minimg_delta_scaled) +
            ", policy=diagnostic_only");
      }
      const bool warm_component_accepted =
          early_component.valid && !early_component.ultra_clash &&
          candidate_close_enough_to_skip_search(
              early_component, warm_state.previous_projection_candidate);
      if (warm_component_accepted) {
        ++timing.component_warm_used;
        warm_state.previous_projection_candidate = early_component;
        warm_state.previous_component_offsets = warm_offsets;
        warm_state.previous_component_layout = component_layout;
        warm_state.has_projection_candidate = true;
        warm_state.has_component_offsets = true;
        if (better_cut_candidate(early_component, prefetched_safe_candidate)) {
          prefetched_safe_candidate = early_component;
          prefetched_safe_points = warm_points;
          prefetched_safe_mode = PbctopoFrameMode::Warm;
          frame_report.mode = PbctopoFrameMode::Warm;
        }
      } else {
        ++timing.pbctopo_warm_component_rejects;
        const char *reason = candidate_skip_search_reject_reason(
            early_component, warm_state.previous_projection_candidate);
        append_pbctopo_log_line(
            "[pbctopo] warm component rejected at frame " +
            to_string(static_cast<long long>(
                this->m_mol[frame_idx].time_of_snap)) +
            ": reason=" + (reason != nullptr ? reason : "unknown"));
      }
      timing.warm_component_s +=
          chrono::duration<double>(clock::now() - warm_component_started)
              .count();
    }

    component_scoring_edges = local_edges;
    bool primary_component_graph_used = false;
    if (can_use_component_logic &&
        component_mode == PbctopoComponentMode::Exact) {
      ++timing.component_graph_primary_attempts;
      ensure_component_context();
      PbctopoCutCandidate primary_cut;
      double primary_shift_norm2 = 0.0;
      vector<Int3> primary_tree_offsets;
      vector<Int3> primary_refined_offsets;
      vector<PbctopoGraphEdge> primary_tree_edges;
      PbctopoCycleSyncMetrics primary_cycle_metrics;
      const PbctopoWarmState primary_graph_warm_baseline = warm_state;
      PbctopoWarmState primary_graph_accepted_warm_state;
      bool primary_graph_unidentified = false;
      const PbctopoCutCandidate *primary_warm_reference =
          warm_state.has_projection_candidate &&
                  warm_state.previous_projection_candidate.valid &&
                  !warm_state.previous_projection_candidate.ultra_clash
              ? &warm_state.previous_projection_candidate
              : nullptr;
      auto try_primary_component_shell = [&](int shell_radius) {
        vector<PbctopoShift> expanded_shifts;
        const vector<PbctopoShift> *shell_shifts = &shifts;
        if (component_context_shell_radius != shell_radius) {
          expanded_shifts = build_topology_shifts(
              this->m_mol[frame_idx].Box, shell_radius);
          shell_shifts = &expanded_shifts;
        }
        timing.component_image_shell_max_radius =
            max(timing.component_image_shell_max_radius,
                static_cast<size_t>(shell_radius));
        timing.component_image_shell_max_shift_count =
            max(timing.component_image_shell_max_shift_count,
                shell_shifts->size());
        primary_cut = {};
        primary_shift_norm2 = 0.0;
        primary_tree_offsets.clear();
        primary_refined_offsets.clear();
        primary_tree_edges.clear();
        primary_cycle_metrics = {};
        component_points.clear();
        PbctopoWarmState shell_trial_warm_state =
            primary_graph_warm_baseline;

        const auto primary_graph_started = clock::now();
        const bool graph_ok = solve_component_offsets_via_pair_graph(
            component_atoms, this->m_mol[frame_idx].Box, components,
            *shell_shifts,
            component_scoring_edges, steric_exclusion_set, component_points,
            trajectory_context.continuity_pairs,
            trajectory_context.continuity_anchor_indices, runtime_cutoffs,
            primary_cut, primary_shift_norm2, beam_width, &timing,
            make_component_refine_callbacks(), anchor_backend,
            same_component_warm_layout
                ? &warm_state.previous_component_offsets
                : nullptr,
            &shell_trial_warm_state, primary_warm_reference,
            &primary_tree_offsets,
            &primary_refined_offsets, &primary_tree_edges,
            &primary_cycle_metrics, &base_anchor_context,
            &explicit_required_contacts, &frame_identifiability);
        timing.component_graph_s +=
            chrono::duration<double>(clock::now() - primary_graph_started)
                .count();
        frame_report.soft_pair_cycle_support =
            primary_cycle_metrics.cycle_support;
        frame_report.soft_pair_cycle_mismatch =
            primary_cycle_metrics.cycle_mismatch;
        frame_report.soft_pair_inconsistent_edges =
            primary_cycle_metrics.inconsistent_edge_count;
        frame_report.soft_pair_fundamental_cycle_count =
            primary_cycle_metrics.fundamental_cycle_count;
        frame_report.soft_pair_fundamental_cycle_residual_rms =
            primary_cycle_metrics.fundamental_cycle_residual_rms;
        frame_report.soft_pair_fundamental_cycle_residual_max =
            primary_cycle_metrics.fundamental_cycle_residual_max;

        size_t split_pairs = 0;
        double split_gap = 0.0;
        const bool split =
            graph_ok && primary_cut.valid && !primary_cut.ultra_clash &&
            component_points.size() == base_atoms.size() &&
            candidate_has_split_contacting_owners(
                component_points, base_atoms, this->m_mol[frame_idx].Box,
                local_hard_required_owner_pairs, split_pairs, split_gap);
        if (split) {
          primary_cut.split_owner_pairs = split_pairs;
          primary_cut.max_owner_raw_minus_minimg_delta = split_gap;
        }
        const bool candidate_complete =
            graph_ok && primary_cut.valid && !primary_cut.ultra_clash &&
            component_points.size() == base_atoms.size() &&
            primary_refined_offsets.size() == components.size();
        if (!candidate_complete)
          return false;
        if (!pbctopo_relative_images_identified(frame_identifiability)) {
          primary_graph_unidentified = true;
          return false;
        }
        if (!certify_candidate_for_output(primary_cut, component_points,
                                          &primary_refined_offsets)) {
          return false;
        }
        primary_graph_accepted_warm_state =
            std::move(shell_trial_warm_state);
        return true;
      };

      const int primary_initial_shell_radius =
          component_shell_radius_explicit
              ? configured_component_shell_radius
              : (components.size() == 2 ? 2 : 1);
      bool primary_certified =
          try_primary_component_shell(primary_initial_shell_radius);
      if (!primary_certified && !component_shell_radius_explicit &&
          primary_initial_shell_radius < 2) {
        ++timing.component_image_shell_expansions;
        primary_certified = try_primary_component_shell(2);
        if (primary_certified)
          ++timing.component_image_shell_retry_successes;
      }
      if (primary_certified) {
        warm_state = std::move(primary_graph_accepted_warm_state);
        record_temporal_certified_candidate(primary_cut, component_points,
                                            &primary_refined_offsets);
        prefetched_safe_candidate = primary_cut;
        prefetched_safe_points = component_points;
        prefetched_safe_mode = PbctopoFrameMode::ComponentGraph;
        warm_state.previous_component_offsets = primary_refined_offsets;
        warm_state.previous_component_layout = component_layout;
        warm_state.has_component_offsets = true;
        frame_report.mode = PbctopoFrameMode::ComponentGraph;
        primary_component_graph_used = true;
        ++timing.component_graph_primary_certified;
        ++timing.component_graph_primary_virtual_skips;
        ++timing.component_graph_used;
      } else {
        ++timing.component_graph_primary_fallbacks;
        if (primary_graph_unidentified) {
          ++timing.component_graph_primary_unidentified;
          append_pbctopo_log_line(
              "[pbctopo] primary component graph cannot bypass virtual search "
              "at frame " +
              to_string(static_cast<long long>(
                  this->m_mol[frame_idx].time_of_snap)) +
              ": identified=0, evidence_components=" +
              to_string(
                  frame_identifiability.evidence_connected_component_count) +
              ", unconstrained_relative_components=" +
              to_string(
                  frame_identifiability.unconstrained_relative_components));
        }
      }
    }

    vector<PbctopoVirtualDiscreteSeedHint> virtual_seed_hints;
    if (!primary_component_graph_used) {
      stage_started = clock::now();
      collect_projection_volume_seeds(projection_atoms, 16,
                                      kProjectionTopSeeds, 0.22,
                                      volume_seeds);
    if (warm_state.has_projection_candidate) {
      PbctopoSeed3D previous_seed;
      previous_seed.cx =
          clamp_virtual_box_center(warm_state.previous_projection_candidate.cut_x);
      previous_seed.cy =
          clamp_virtual_box_center(warm_state.previous_projection_candidate.cut_y);
      previous_seed.cz =
          clamp_virtual_box_center(warm_state.previous_projection_candidate.cut_z);
      previous_seed.score =
          max(1.0, warm_state.previous_projection_candidate.gap_score + 1.0);
      previous_seed.source = PbctopoVirtualSeedSource::Previous;
      volume_seeds.push_back(previous_seed);
      ++timing.previous_seed_used;
    }
    if (kinematic_seed_history_ready) {
      const PbctopoVirtualSeedCenter predicted_center =
          extrapolate_virtual_seed_center(
              virtual_seed_center_from_candidate(kinematic_seed_previous),
              virtual_seed_center_from_candidate(kinematic_seed_latest),
              PbctopoLatticeMetric(this->m_mol[frame_idx].Box));
      PbctopoSeed3D kinematic_seed;
      kinematic_seed.cx = predicted_center.cx;
      kinematic_seed.cy = predicted_center.cy;
      kinematic_seed.cz = predicted_center.cz;
      kinematic_seed.score =
          max(1.0, kinematic_seed_latest.gap_score + 0.5);
      kinematic_seed.source = PbctopoVirtualSeedSource::Kinematic;
      volume_seeds.push_back(kinematic_seed);
      ++timing.pbctopo_kinematic_seed_candidates;
    }
    timing.seed_s += chrono::duration<double>(clock::now() - stage_started).count();

    virtual_seed_hints.reserve(volume_seeds.size());
    for (const auto &seed : volume_seeds) {
      virtual_seed_hints.push_back(
          {seed.cx, seed.cy, seed.cz, seed.score, seed.source});
    }
    stage_started = clock::now();
    PbctopoVirtualDiscreteSearchOptions discrete_options;
    discrete_options.top_k_per_axis = 4;
    discrete_options.max_exhaustive_states = 4096;
    build_pbctopo_virtual_discrete_search_plan(
        projection_atoms, projection_edges, this->m_mol[frame_idx].Box,
        virtual_seed_hints, discrete_options, virtual_search_workspace,
        virtual_search_plan);
    const double seam_sweep_s =
        chrono::duration<double>(clock::now() - stage_started).count();
    timing.event_planes_s += seam_sweep_s;
    timing.pbctopo_virtual_seam_sweep_s += seam_sweep_s;
    ++timing.pbctopo_virtual_discrete_frames;
    timing.pbctopo_virtual_axis_states_x +=
        virtual_search_plan.x_states.size();
    timing.pbctopo_virtual_axis_states_y +=
        virtual_search_plan.y_states.size();
    timing.pbctopo_virtual_axis_states_z +=
        virtual_search_plan.z_states.size();
    timing.pbctopo_virtual_sweep_events_x +=
        virtual_search_plan.event_count_x;
    timing.pbctopo_virtual_sweep_events_y +=
        virtual_search_plan.event_count_y;
    timing.pbctopo_virtual_sweep_events_z +=
        virtual_search_plan.event_count_z;
    if (virtual_search_plan.workspace_reused)
      ++timing.pbctopo_virtual_sweep_workspace_reuse_frames;
    timing.pbctopo_virtual_sweep_workspace_peak_bytes =
        max(timing.pbctopo_virtual_sweep_workspace_peak_bytes,
            virtual_search_plan.workspace_bytes_estimate);
    if (numeric_limits<uint64_t>::max() - timing.pbctopo_virtual_state_space <
        virtual_search_plan.total_state_space) {
      timing.pbctopo_virtual_state_space = numeric_limits<uint64_t>::max();
    } else {
      timing.pbctopo_virtual_state_space +=
          virtual_search_plan.total_state_space;
    }
    timing.pbctopo_virtual_states_evaluated +=
        virtual_search_plan.states.size();
    if (virtual_search_plan.search_exhaustive)
      ++timing.pbctopo_virtual_seam_enumeration_exhaustive_frames;
    else
      ++timing.pbctopo_virtual_search_shortlist_frames;
    if (virtual_search_plan.branch_and_bound_used)
      ++timing.pbctopo_virtual_branch_and_bound_frames;
    if (isfinite(virtual_search_plan.omitted_seam_lower_bound)) {
      timing.pbctopo_virtual_omitted_seam_lower_bound_min =
          min(timing.pbctopo_virtual_omitted_seam_lower_bound_min,
              virtual_search_plan.omitted_seam_lower_bound);
    }
    if (isfinite(virtual_search_plan.lower_bound_gap)) {
      timing.pbctopo_virtual_lower_bound_gap_min =
          min(timing.pbctopo_virtual_lower_bound_gap_min,
              virtual_search_plan.lower_bound_gap);
    }
    frame_report.virtual_seam_enumeration_exhaustive =
        virtual_search_plan.search_exhaustive;
    frame_report.virtual_projection_scoring_exhaustive =
        virtual_search_plan.search_exhaustive &&
        virtual_search_plan.states.size() ==
            virtual_search_plan.total_state_space;
    frame_report.virtual_full_atom_verification_exhaustive = false;
    if (frame_report.virtual_projection_scoring_exhaustive)
      ++timing.pbctopo_virtual_projection_scoring_exhaustive_frames;
    if (frame_report.virtual_full_atom_verification_exhaustive)
      ++timing.pbctopo_virtual_full_atom_verification_exhaustive_frames;
    frame_report.virtual_search_exhaustive =
        frame_report.virtual_seam_enumeration_exhaustive &&
        frame_report.virtual_projection_scoring_exhaustive &&
        frame_report.virtual_full_atom_verification_exhaustive;
    if (frame_report.virtual_search_exhaustive)
      ++timing.pbctopo_virtual_search_exhaustive_frames;
    frame_report.virtual_branch_and_bound_used =
        virtual_search_plan.branch_and_bound_used;
    frame_report.virtual_state_space = virtual_search_plan.total_state_space;
    frame_report.virtual_states_evaluated = virtual_search_plan.states.size();
    frame_report.virtual_omitted_seam_lower_bound =
        virtual_search_plan.omitted_seam_lower_bound;
    frame_report.virtual_lower_bound_gap = virtual_search_plan.lower_bound_gap;
    }

    PbctopoCutCandidate best_candidate;
    PbctopoCutCandidate best_safe_candidate;
    bool best_safe_certification_rejected = false;
    size_t safe_candidates = 0;
    bool frame_anchor_ambiguity_counted = false;
    auto record_report_soft_candidate = [&](const PbctopoCutCandidate &candidate) {
      if (!candidate.valid)
        return;
      ++report_soft_valid_candidates;
      if (!candidate.ultra_clash)
        ++report_nonclashing_candidates;
    };
    struct PbctopoOutputCandidateSnapshot {
      PbctopoCutCandidate candidate;
      vector<Int3> offsets;
      PbctopoFrameMode mode = PbctopoFrameMode::None;
    };
    constexpr size_t kInitialOutputCertificationBand = 12;
    constexpr size_t kMaxOutputCertificationBand = 48;
    constexpr size_t kMaxCompleteOutputCertificationBand = 4096;
    size_t output_certification_band_limit = kMaxOutputCertificationBand;
    bool complete_feasible_band_retained = false;
    vector<PbctopoOutputCandidateSnapshot> output_candidate_band;
    output_candidate_band.reserve(kMaxOutputCertificationBand);
    auto record_output_candidate =
        [&](PbctopoCutCandidate candidate, const vector<Vec3> &points,
            PbctopoFrameMode mode,
            const vector<Int3> *known_offsets = nullptr) {
          if (!candidate.valid || candidate.ultra_clash ||
              points.size() != base_atoms.size()) {
            return false;
          }
          ensure_component_context();
          vector<Int3> offsets;
          if (known_offsets != nullptr &&
              known_offsets->size() == components.size()) {
            offsets = *known_offsets;
          } else {
            offsets = extract_offsets_and_time(component_atoms, points);
          }
          if (!align_component_offsets_to_points(component_atoms, components,
                                                 certification_lattice, points,
                                                 offsets)) {
            ++timing.pbctopo_output_certification_band_precheck_rejects;
            return false;
          }
          candidate.assignment =
              make_pbctopo_image_assignment(offsets, components,
                                             component_atoms);
          if (!candidate.assignment.valid) {
            ++timing.pbctopo_output_certification_band_precheck_rejects;
            return false;
          }
          const auto source_point_consistency =
              titan_pbctopo::evaluate_pbctopo_assignment_point_consistency(
                  component_atoms, points, atom_component_index,
                  candidate.assignment, certification_lattice);
          if (!source_point_consistency.input_valid ||
              source_point_consistency.point_residuals != 0) {
            ++timing.pbctopo_output_certification_band_precheck_rejects;
            return false;
          }
          if (!annotate_candidate_evidence(candidate, points))
            return false;
          const PbctopoImageAssignment assignment = candidate.assignment;
          auto existing = find_if(
              output_candidate_band.begin(), output_candidate_band.end(),
              [&](const PbctopoOutputCandidateSnapshot &snapshot) {
                return same_pbctopo_image_assignment(
                    snapshot.candidate.assignment, candidate.assignment);
              });
          if (existing != output_candidate_band.end()) {
            const PbctopoCutCandidate prior = existing->candidate;
            const bool equivalence_known =
                existing->candidate.search_equivalence_known ||
                candidate.search_equivalence_known;
            const bool equivalent_to_search_best =
                (existing->candidate.search_equivalence_known &&
                 existing->candidate.equivalent_to_search_best) ||
                (candidate.search_equivalence_known &&
                 candidate.equivalent_to_search_best);
            if (better_cut_candidate(candidate, existing->candidate)) {
              existing->candidate = candidate;
              existing->offsets = std::move(offsets);
              existing->mode = mode;
            }
            existing->candidate.search_equivalence_known = equivalence_known;
            existing->candidate.equivalent_to_search_best =
                equivalent_to_search_best;
            merge_pbctopo_search_domain_provenance(existing->candidate,
                                                    prior);
            merge_pbctopo_search_domain_provenance(existing->candidate,
                                                    candidate);
          } else {
            output_candidate_band.push_back(
                {candidate, std::move(offsets), mode});
          }
          stable_sort(
              output_candidate_band.begin(), output_candidate_band.end(),
              [](const PbctopoOutputCandidateSnapshot &lhs,
                 const PbctopoOutputCandidateSnapshot &rhs) {
                return better_cut_candidate(lhs.candidate, rhs.candidate);
              });
          if (output_candidate_band.size() > output_certification_band_limit)
            output_candidate_band.resize(output_certification_band_limit);
          size_t offset_bytes = 0;
          for (const auto &snapshot : output_candidate_band) {
            offset_bytes += snapshot.offsets.capacity() * sizeof(Int3);
          }
          timing.pbctopo_output_certification_band_offset_bytes_peak = max(
              timing.pbctopo_output_certification_band_offset_bytes_peak,
              offset_bytes);
          if (base_atoms.size() <=
              numeric_limits<size_t>::max() /
                  (max<size_t>(1, output_candidate_band.size()) *
                   sizeof(Vec3))) {
            const size_t avoided_bytes = output_candidate_band.size() *
                                         base_atoms.size() * sizeof(Vec3);
            timing.pbctopo_output_certification_band_point_bytes_avoided_peak =
                max(timing
                        .pbctopo_output_certification_band_point_bytes_avoided_peak,
                    avoided_bytes);
          }
          return any_of(
              output_candidate_band.begin(), output_candidate_band.end(),
              [&](const PbctopoOutputCandidateSnapshot &snapshot) {
                return same_pbctopo_image_assignment(
                    snapshot.candidate.assignment, assignment);
              });
        };
    auto record_frame_failure = [&](PbctopoFailureReason reason) {
      if (frame_report.fallback_reason == PbctopoFailureReason::None)
        frame_report.fallback_reason = reason;
    };
    auto commit_certified_candidate_to_warm_state =
        [&](const PbctopoCutCandidate &selected_candidate) {
          const PbctopoWarmState trial_warm_state = warm_state;
          if (!commit_pbctopo_certified_warm_state(
                  warm_state, frame_start_warm_state, trial_warm_state,
                  selected_candidate, component_layout)) {
            ++timing.pbctopo_warm_state_commit_rejects;
            return false;
          }
          if (selected_candidate.virtual_seed_source !=
              PbctopoVirtualSeedSource::None) {
            update_kinematic_seed_history(warm_state, selected_candidate,
                                          frame_idx, timing);
          } else {
            reset_kinematic_seed_history(warm_state, timing);
          }
          ++timing.pbctopo_warm_state_commits;
          return true;
        };
    auto apply_selected_evidence_to_frame_identifiability = [&]() {
      titan_pbctopo::apply_pbctopo_final_selected_evidence(
          frame_identifiability, selected_evidence_audit.evaluated,
          selected_evidence_audit.evidence_graph_connected,
          selected_evidence_audit.evidence_component_count,
          selected_evidence_audit.evidence_edge_count,
          selected_evidence_audit.explicit_contact_evidence_edges,
          selected_evidence_audit.scored_contact_evidence_edges,
          selected_evidence_audit.evidence_connected_component_count);
    };
    auto audit_selected_evidence = [&](const vector<Vec3> &selected_points,
                                       const PbctopoImageAssignment &assignment) {
      const auto audit_started = clock::now();
      auto audit = titan_pbctopo::audit_pbctopo_selected_hypotheses(
          span<const PbctopoAtom>(base_atoms), span<const Vec3>(selected_points),
          base_anchor_context.contact_hypotheses, assignment, true,
          explicit_evidence_component_edges);
      timing.pbctopo_selected_evidence_audit_s +=
          chrono::duration<double>(clock::now() - audit_started).count();
      if (!audit) {
        selected_evidence_audit = {};
        append_pbctopo_log_line(
            "[pbctopo] selected evidence audit failed closed at frame " +
            to_string(frame_report.frame) + ": " + audit.error());
        return;
      }
      selected_evidence_audit = *audit;
      cache_candidate_evidence_audit(assignment, *audit);
      apply_selected_evidence_to_frame_identifiability();
      ++timing.pbctopo_selected_evidence_audits;
      timing.pbctopo_selected_evidence_contacts_checked +=
          audit->contacts_checked;
      timing.pbctopo_selected_evidence_contacts_lost += audit->contacts_lost;
    };
    // Status and the warm-state commit must follow the same selected-evidence
    // audit that finalize_frame_report writes over the frame identifiability.
    // The contact graph built before the search can still call a frame
    // disconnected that the audit of the selected assignment finds supported.
    auto selected_relative_images_identified = [&]() {
      PbctopoIdentifiability selected_identifiability = frame_identifiability;
      titan_pbctopo::apply_pbctopo_final_selected_evidence(
          selected_identifiability, selected_evidence_audit.evaluated,
          selected_evidence_audit.evidence_graph_connected,
          selected_evidence_audit.evidence_component_count,
          selected_evidence_audit.evidence_edge_count,
          selected_evidence_audit.explicit_contact_evidence_edges,
          selected_evidence_audit.scored_contact_evidence_edges,
          selected_evidence_audit.evidence_connected_component_count);
      return pbctopo_relative_images_identified(selected_identifiability);
    };
    auto export_h9_partition_and_orbit =
        [&](const PbctopoCutCandidate &base_candidate,
            const vector<Int3> &base_offsets) {
          if (!h9_orbit_enabled || !selected_evidence_audit.evaluated)
            return;
          if (!base_candidate.assignment.valid ||
              selected_evidence_audit.evidence_component_labels.size() !=
                  components.size() ||
              base_candidate.assignment.component_ids.size() !=
                  components.size() ||
              base_offsets.size() != components.size()) {
            append_pbctopo_log_line(
                "[pbctopo] H9 orbit skipped: evidence partition or assignment "
                "layout mismatch at frame " +
                to_string(frame_report.frame));
            return;
          }

          size_t canonical_root = 0;
          for (size_t comp_idx = 1; comp_idx < components.size(); ++comp_idx) {
            if (components[comp_idx].size() >
                    components[canonical_root].size() ||
                (components[comp_idx].size() ==
                     components[canonical_root].size() &&
                 base_candidate.assignment.component_ids[comp_idx] <
                     base_candidate.assignment.component_ids[canonical_root])) {
              canonical_root = comp_idx;
            }
          }
          const size_t root_label =
              selected_evidence_audit
                  .evidence_component_labels[canonical_root];
          if (h9_evidence_component_report.is_open()) {
            for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
              h9_evidence_component_report
                  << "h9_evidence_partition_v1," << frame_report.frame << ','
                  << this->m_mol[frame_idx].time_of_snap << ',' << comp_idx
                  << ',' << base_candidate.assignment.component_ids[comp_idx]
                  << ',' << components[comp_idx].size() << ','
                  << selected_evidence_audit
                         .evidence_component_labels[comp_idx]
                  << ',' << static_cast<int>(comp_idx == canonical_root) << ','
                  << selected_evidence_audit
                         .evidence_connected_component_count
                  << ',' << selected_evidence_audit
                                .relative_lattice_degrees_of_freedom
                  << '\n';
            }
          }

          const bool orbit_applicable =
              selected_evidence_audit.evidence_connected_component_count == 2 &&
              selected_evidence_audit.relative_lattice_degrees_of_freedom == 3 &&
              root_label < 2;
          if (!orbit_applicable || !h9_orbit_report.is_open())
            return;
          const size_t shifted_label = root_label == 0 ? 1 : 0;
          vector<Vec3> orbit_points;
          orbit_points.reserve(component_atoms.size());
          for (int qx = -1; qx <= 1; ++qx) {
            for (int qy = -1; qy <= 1; ++qy) {
              for (int qz = -1; qz <= 1; ++qz) {
                vector<Int3> orbit_offsets = base_offsets;
                bool within_radius = true;
                for (size_t comp_idx = 0; comp_idx < components.size();
                     ++comp_idx) {
                  if (selected_evidence_audit
                          .evidence_component_labels[comp_idx] !=
                      shifted_label) {
                    continue;
                  }
                  Int3 shifted{};
                  if (!titan_pbctopo::checked_int3_add(
                          orbit_offsets[comp_idx], Int3{qx, qy, qz}, shifted)) {
                    within_radius = false;
                    break;
                  }
                  orbit_offsets[comp_idx] = shifted;
                }
                PbctopoCutCandidate orbit_candidate = base_candidate;
                if (within_radius) {
                  const PbctopoImageAssignment provisional_assignment =
                      make_pbctopo_image_assignment(
                          orbit_offsets, components, component_atoms);
                  within_radius = provisional_assignment.valid;
                  for (const Int3 &offset :
                       provisional_assignment.component_offsets) {
                    if (offset.x < -configured_component_shell_radius ||
                        offset.x > configured_component_shell_radius ||
                        offset.y < -configured_component_shell_radius ||
                        offset.y > configured_component_shell_radius ||
                        offset.z < -configured_component_shell_radius ||
                        offset.z > configured_component_shell_radius) {
                      within_radius = false;
                      break;
                    }
                  }
                }
                bool hard_certified = false;
                optional<PbctopoSelectedEvidenceAudit> orbit_audit;
                if (within_radius) {
                  double orbit_shift_norm2 = 0.0;
                  build_component_points_from_offsets(
                      component_atoms, components,
                      this->m_mol[frame_idx].Box, orbit_offsets, orbit_points,
                      orbit_shift_norm2);
                  hard_certified = certify_candidate_for_output(
                      orbit_candidate, orbit_points, &orbit_offsets);
                  if (hard_certified) {
                    auto audited =
                        titan_pbctopo::audit_pbctopo_selected_hypotheses(
                            span<const PbctopoAtom>(base_atoms),
                            span<const Vec3>(orbit_points),
                            base_anchor_context.contact_hypotheses,
                            orbit_candidate.assignment, true,
                            explicit_evidence_component_edges);
                    if (audited)
                      orbit_audit = *audited;
                  }
                }
                const bool evidence_evaluated =
                    orbit_audit.has_value() && orbit_audit->evaluated;
                const bool evidence_noncontradicted =
                    evidence_evaluated &&
                    orbit_audit->state != PbctopoEvidenceState::Contradicted;
                const bool accepted = hard_certified &&
                                      evidence_noncontradicted;
                h9_orbit_report
                    << "h9_local_orbit_v1," << frame_report.frame << ','
                    << this->m_mol[frame_idx].time_of_snap << ',' << qx << ','
                    << qy << ',' << qz << ',' << root_label << ','
                    << shifted_label << ',' << static_cast<int>(within_radius)
                    << ',' << static_cast<int>(hard_certified) << ','
                    << static_cast<int>(evidence_evaluated) << ','
                    << static_cast<int>(evidence_noncontradicted) << ','
                    << static_cast<int>(accepted) << ','
                    << (orbit_candidate.assignment.valid
                            ? orbit_candidate.assignment.signature
                            : 0)
                    << ','
                    << (orbit_audit.has_value()
                            ? titan_pbctopo::pbctopo_evidence_state_name(
                                  orbit_audit->state)
                            : "not_evaluated")
                    << ','
                    << (orbit_audit.has_value()
                            ? orbit_audit->evidence_connected_component_count
                            : 0)
                    << ','
                    << (orbit_audit.has_value()
                            ? orbit_audit->relative_lattice_degrees_of_freedom
                            : 0)
                    << ','
                    << (orbit_audit.has_value()
                            ? orbit_audit->evidence_edge_count
                            : 0)
                    << ','
                    << (orbit_audit.has_value()
                            ? orbit_audit->compatible_hypothesis_count
                            : 0)
                    << ','
                    << (orbit_audit.has_value()
                            ? orbit_audit->selected_compatible_contact_count
                            : 0)
                    << '\n';
              }
            }
          }
        };
    auto finalize_frame_report =
        [&](PbctopoFrameStatus status, PbctopoFrameMode mode,
            PbctopoFailureReason reason,
            const PbctopoCutCandidate *selected_candidate) {
          if (selected_candidate != nullptr && selected_candidate->valid &&
              !selected_evidence_audit.evaluated) {
            status = PbctopoFrameStatus::Fallback;
            reason = PbctopoFailureReason::EvidenceHypothesisUnavailable;
          }
          if (selected_candidate != nullptr && selected_candidate->valid) {
            greedy_selected_candidates[frame_idx] = *selected_candidate;
          }
          frame_report.identifiability = frame_identifiability;
          frame_report.evidence_state = selected_evidence_audit.state;
          frame_report.evidence_consistency_evaluated =
              selected_evidence_audit.evaluated;
          frame_report.evidence_consistent =
              selected_evidence_audit.consistent;
          frame_report.evidence_graph_connected =
              selected_evidence_audit.evaluated
                  ? selected_evidence_audit.evidence_graph_connected
                  : frame_identifiability.evidence_graph_connected;
          titan_pbctopo::apply_pbctopo_final_selected_evidence(
              frame_report.identifiability,
              selected_evidence_audit.evaluated,
              selected_evidence_audit.evidence_graph_connected,
              selected_evidence_audit.evidence_component_count,
              selected_evidence_audit.evidence_edge_count,
              selected_evidence_audit.explicit_contact_evidence_edges,
              selected_evidence_audit.scored_contact_evidence_edges,
              selected_evidence_audit.evidence_connected_component_count);
          if (selected_candidate != nullptr && selected_candidate->valid &&
              selected_evidence_audit.evaluated) {
            const bool final_output_identified =
                pbctopo_relative_images_identified(
                    frame_report.identifiability);
            const bool rescue_provenance =
                status == PbctopoFrameStatus::Rescued ||
                mode == PbctopoFrameMode::HullRescue ||
                mode == PbctopoFrameMode::DistanceRescue;
            if (selected_evidence_audit.state ==
                PbctopoEvidenceState::Contradicted) {
              status = PbctopoFrameStatus::Contradicted;
              reason = PbctopoFailureReason::EvidenceContradicted;
            } else if (selected_evidence_audit.state ==
                       PbctopoEvidenceState::WeakAmbiguous) {
              status = PbctopoFrameStatus::WeakObservation;
              reason = PbctopoFailureReason::EvidenceUnderdetermined;
            } else if (selected_evidence_audit.state ==
                           PbctopoEvidenceState::WeakDisconnected ||
                       !final_output_identified) {
              status = PbctopoFrameStatus::WeakObservation;
              reason = PbctopoFailureReason::EvidenceDisconnected;
            } else {
              status = rescue_provenance ? PbctopoFrameStatus::Rescued
                                         : PbctopoFrameStatus::Certified;
              reason = PbctopoFailureReason::None;
            }
          }
          if (selected_evidence_audit.evaluated) {
            frame_report.soft_observed_contact_pair_hash =
                selected_evidence_audit.observed_pair_hash;
            frame_report.soft_observed_lost_pair_hash =
                selected_evidence_audit.lost_pair_hash;
            frame_report.soft_observed_all_hypotheses_hash =
                selected_evidence_audit.all_hypotheses_hash;
            frame_report.soft_observed_selected_hypothesis_hash =
                selected_evidence_audit.selected_hypothesis_hash;
            frame_report.soft_observed_selected_compatible_pair_hash =
                selected_evidence_audit.selected_compatible_pair_hash;
            frame_report.soft_observed_hypothesis_count =
                selected_evidence_audit.hypothesis_count;
            frame_report.soft_observed_component_relation_count =
                selected_evidence_audit.component_relation_count;
            frame_report.soft_observed_ambiguous_relation_count =
                selected_evidence_audit.ambiguous_component_relation_count;
            frame_report.soft_observed_compatible_hypothesis_count =
                selected_evidence_audit.compatible_hypothesis_count;
            frame_report.soft_observed_selected_compatible_contact_count =
                selected_evidence_audit.selected_compatible_contact_count;
            frame_report.soft_observed_alternative_hypothesis_contact_count =
                selected_evidence_audit.alternative_hypothesis_contact_count;
            frame_report.soft_observed_selected_compatible_contact_loss_count =
                selected_evidence_audit.selected_compatible_contact_loss_count;
            frame_report.soft_observed_no_support_relation_count =
                selected_evidence_audit.no_supported_contact_relation_count;
            frame_report.soft_observed_selected_relation_has_support =
                selected_evidence_audit.selected_relation_has_support;
            frame_report.soft_observed_hypothesis_construction_attempted =
                base_anchor_context.contact_hypotheses.attempted;
            frame_report.soft_observed_hypothesis_construction_complete =
                base_anchor_context.contact_hypotheses.complete;
            frame_report.soft_observed_hypothesis_mic_ambiguity_failures =
                base_anchor_context.contact_hypotheses.mic_ambiguity_failures;
            frame_report.soft_observed_hypothesis_component_mapping_failures =
                base_anchor_context.contact_hypotheses.component_mapping_failures;
            frame_report.soft_observed_hypothesis_internal_edges_ignored =
                base_anchor_context.contact_hypotheses.internal_edges_ignored;
            frame_report.soft_observed_contacts_checked =
                selected_evidence_audit.contacts_checked;
            frame_report.soft_observed_contacts_lost =
                selected_evidence_audit.contacts_lost;
            frame_report.soft_observed_contact_cutoff_A =
                runtime_cutoffs.pair_contact_cutoff;
            frame_report.soft_observed_contact_construction_attempted =
                base_anchor_context.soft_observed_contact_construction_attempted;
            frame_report.soft_observed_contact_construction_complete =
                base_anchor_context.soft_observed_contact_construction_complete;
            frame_report.soft_observed_contact_cell_setup_failures =
                base_anchor_context.soft_observed_contact_cell_setup_failures;
            frame_report.soft_observed_contact_atom_index_failures =
                base_anchor_context.soft_observed_contact_atom_index_failures;
            frame_report.soft_observed_contact_mic_query_failures =
                base_anchor_context.soft_observed_contact_mic_query_failures;
          } else {
            frame_report.soft_observed_contact_pair_hash =
                base_anchor_context.contact_hypotheses.raw_observed_pair_hash;
            frame_report.soft_observed_lost_pair_hash = {};
            frame_report.soft_observed_all_hypotheses_hash =
                base_anchor_context.contact_hypotheses.all_hypotheses_hash;
            frame_report.soft_observed_selected_hypothesis_hash = {};
            frame_report.soft_observed_selected_compatible_pair_hash = {};
            frame_report.soft_observed_hypothesis_count = 0;
            frame_report.soft_observed_component_relation_count = 0;
            frame_report.soft_observed_ambiguous_relation_count = 0;
            frame_report.soft_observed_compatible_hypothesis_count = 0;
            frame_report.soft_observed_selected_compatible_contact_count = 0;
            frame_report.soft_observed_alternative_hypothesis_contact_count = 0;
            frame_report.soft_observed_selected_compatible_contact_loss_count = 0;
            frame_report.soft_observed_no_support_relation_count = 0;
            frame_report.soft_observed_selected_relation_has_support = false;
            frame_report.soft_observed_hypothesis_construction_attempted =
                base_anchor_context.contact_hypotheses.attempted;
            frame_report.soft_observed_hypothesis_construction_complete = false;
            frame_report.soft_observed_hypothesis_mic_ambiguity_failures =
                base_anchor_context.contact_hypotheses.mic_ambiguity_failures;
            frame_report.soft_observed_hypothesis_component_mapping_failures =
                base_anchor_context.contact_hypotheses.component_mapping_failures;
            frame_report.soft_observed_hypothesis_internal_edges_ignored =
                base_anchor_context.contact_hypotheses.internal_edges_ignored;
            frame_report.soft_observed_contacts_checked =
                base_anchor_context.contact_hypotheses.raw_observed_pair_count;
            frame_report.soft_observed_contacts_lost = 0;
            frame_report.soft_observed_contact_cutoff_A =
                base_anchor_context.soft_observed_contact_construction_complete
                    ? runtime_cutoffs.pair_contact_cutoff
                    : numeric_limits<double>::quiet_NaN();
            frame_report.soft_observed_contact_construction_attempted =
                base_anchor_context.soft_observed_contact_construction_attempted;
            frame_report.soft_observed_contact_construction_complete =
                base_anchor_context.soft_observed_contact_construction_complete;
            frame_report.soft_observed_contact_cell_setup_failures =
                base_anchor_context.soft_observed_contact_cell_setup_failures;
            frame_report.soft_observed_contact_atom_index_failures =
                base_anchor_context.soft_observed_contact_atom_index_failures;
            frame_report.soft_observed_contact_mic_query_failures =
                base_anchor_context.soft_observed_contact_mic_query_failures;
          }
          const bool geometry_evidence = soft_geometry_edge_count > 0;
          const bool scored_contact_evidence =
              frame_identifiability.scored_contact_evidence_edges > 0;
          const bool explicit_contact_evidence =
              frame_identifiability.explicit_contact_evidence_edges > 0;
          const size_t evidence_source_count =
              static_cast<size_t>(geometry_evidence) +
              static_cast<size_t>(scored_contact_evidence) +
              static_cast<size_t>(explicit_contact_evidence);
          if (evidence_source_count > 1) {
            frame_report.search_evidence_source =
                PbctopoSearchEvidenceSource::Mixed;
          } else if (explicit_contact_evidence) {
            frame_report.search_evidence_source =
                PbctopoSearchEvidenceSource::ExplicitContacts;
          } else if (scored_contact_evidence) {
            frame_report.search_evidence_source =
                PbctopoSearchEvidenceSource::ScoredContacts;
          } else if (geometry_evidence) {
            frame_report.search_evidence_source =
                PbctopoSearchEvidenceSource::GeometryCutoff;
          }
          frame_report.status = status;
          frame_report.certificate_scope =
              status == PbctopoFrameStatus::Certified ||
                      status == PbctopoFrameStatus::WeakObservation ||
                      status == PbctopoFrameStatus::Rescued ||
                      status == PbctopoFrameStatus::Contradicted
                  ? (explicit_required_contacts.empty()
                         ? PbctopoCertificateScope::ImageIntercomponentSteric
                         : PbctopoCertificateScope::
                               ImageIntercomponentStericInterface)
                  : PbctopoCertificateScope::None;
          if (frame_report.mode == PbctopoFrameMode::None ||
              status != PbctopoFrameStatus::Certified) {
            frame_report.mode = mode;
          }
          if (reason != PbctopoFailureReason::None)
            frame_report.fallback_reason = reason;
          else if (status != PbctopoFrameStatus::Fallback)
            frame_report.fallback_reason = PbctopoFailureReason::None;
          frame_report.component_count = components.size();
          frame_report.soft_valid_candidate_count =
              report_soft_valid_candidates;
          frame_report.nonclashing_candidate_count =
              report_nonclashing_candidates;
          frame_report.unique_certified_assignment_count =
              report_certified_candidates.size();
          frame_report.committed_candidate_count =
              status == PbctopoFrameStatus::Certified ||
                      status == PbctopoFrameStatus::WeakObservation ||
                      status == PbctopoFrameStatus::Rescued
                  ? 1
                  : 0;

          const PbctopoCutCandidate *selected_for_report =
              pbctopo_frame_status_has_selected_assignment(status) &&
                      selected_candidate != nullptr && selected_candidate->valid
                  ? selected_candidate
                  : nullptr;
          const PbctopoCutCandidate *best_for_report =
              report_certified_candidates.empty()
                  ? nullptr
                  : &report_certified_candidates.front();
          const PbctopoCutCandidate *second_for_report =
              report_certified_candidates.size() > 1
                  ? &report_certified_candidates[1]
                  : nullptr;
          if (best_for_report != nullptr && best_for_report->valid) {
            frame_report.pool_best_score =
                make_score_vector(*best_for_report);
            frame_report.best_score = frame_report.pool_best_score;
          }
          if (second_for_report != nullptr && second_for_report->valid) {
            frame_report.pool_second_score =
                make_score_vector(*second_for_report);
            frame_report.second_score = frame_report.pool_second_score;
          }
          if (temporal_mode == PbctopoTemporalMode::Viterbi) {
            for (auto &temporal_candidate :
                 temporal_cut_candidates[frame_idx]) {
              if (pbctopo_candidate_anchor_ambiguous(
                      temporal_candidate, report_certified_candidates)) {
                temporal_candidate.anchor_graph.ambiguity_count =
                    max<size_t>(1,
                                temporal_candidate.anchor_graph.ambiguity_count);
              }
            }
          }
          if (selected_for_report != nullptr) {
            populate_pbctopo_selected_candidate_report(
                frame_report, *selected_for_report, base_atoms,
                runtime_cutoffs.clash_cutoff,
                runtime_cutoffs.continuity_cutoff,
                report_certified_candidates);
            if (frame_report.anchor_graph.ambiguity_count > 0 &&
                !frame_anchor_ambiguity_counted) {
              ++timing.pbctopo_anchor_ambiguous_frames;
              frame_anchor_ambiguity_counted = true;
            }
          } else {
            frame_report.hard_feasibility = {};
            frame_report.hard_feasible = false;
          }
          if (best_for_report != nullptr && best_for_report->valid &&
              second_for_report != nullptr && second_for_report->valid) {
            const PbctopoLexicographicGap score_gap =
                candidate_lexicographic_gap(*best_for_report,
                                            *second_for_report);
            frame_report.score_gap_metric = score_gap.metric;
            frame_report.score_gap = score_gap.gap;
          }
          if (evidence_relation_export_enabled &&
              !evidence_relation_frame_written) {
            evidence_relation_frame_written = true;
            if (!evidence_relation_cache_error.empty()) {
              ++evidence_relation_export_failures;
              append_pbctopo_log_line(
                  "[pbctopo] candidate evidence relation export failed "
                  "closed at frame " +
                  to_string(frame_report.frame) + ": " +
                  evidence_relation_cache_error);
            } else {
              const auto &export_candidates =
                  temporal_mode == PbctopoTemporalMode::Viterbi
                      ? temporal_cut_candidates[frame_idx]
                      : report_certified_candidates;
              const auto exported = write_pbctopo_candidate_evidence_frame(
                  pbctopo_candidate_evidence_summary_report,
                  pbctopo_candidate_evidence_relation_report, frame_report,
                  export_candidates, selected_for_report,
                  temporal_mode == PbctopoTemporalMode::Viterbi,
                  evidence_relation_audit_snapshots);
              if (!exported) {
                ++evidence_relation_export_failures;
                append_pbctopo_log_line(
                    "[pbctopo] candidate evidence relation export failed "
                    "closed at frame " +
                    to_string(frame_report.frame) + ": " + exported.error());
              } else {
                evidence_relation_export_groups +=
                    exported->candidate_groups;
                evidence_relation_export_rows += exported->relation_rows;
                evidence_relation_export_duplicates +=
                    exported->duplicate_candidates_skipped;
              }
            }
          }
          // The certified band is exported only for frames that have one.
          // The rejection ledger records every evaluated rejection whatever
          // the frame's status: a frame at which every candidate failed and
          // fell back is where the ledger matters most, and an empty ledger
          // must mean "nothing was rejected", not "the frame was skipped".
          const bool band_status =
              status == PbctopoFrameStatus::Certified ||
              status == PbctopoFrameStatus::Rescued ||
              status == PbctopoFrameStatus::WeakObservation;
          if ((pbctopo_assignment_set_report.is_open() ||
               pbctopo_rejected_assignment_report.is_open()) &&
              temporal_mode != PbctopoTemporalMode::Viterbi &&
              (band_status || !report_rejected_candidates.empty())) {
            double v1[3]{}, v2[3]{}, v3[3]{};
            this->m_mol[frame_idx].Box.get_v1(v1);
            this->m_mol[frame_idx].Box.get_v2(v2);
            this->m_mol[frame_idx].Box.get_v3(v3);
            const auto normalized_gram =
                titan_pbctopo::pbctopo_temporal_normalized_gram(
                    {v1[0], v1[1], v1[2], v2[0], v2[1], v2[2],
                     v3[0], v3[1], v3[2]});
            vector<PbctopoCutCandidate> report_band =
                report_certified_candidates;
            report_band.insert(report_band.end(),
                               report_rejected_candidates.begin(),
                               report_rejected_candidates.end());
            if (pbctopo_rejected_assignment_report.is_open())
              titan_pbctopo::write_pbctopo_rejected_assignment_rows(
                  pbctopo_rejected_assignment_report,
                  frame_report_offset + frame_idx,
                  this->m_mol[frame_idx].time_of_snap, report_band,
                  normalized_gram, frame_report.topology_epoch_id);
            if (band_status && pbctopo_assignment_set_report.is_open())
              write_pbctopo_assignment_set_rows(
                pbctopo_assignment_set_report,
                frame_report_offset + frame_idx,
                this->m_mol[frame_idx].time_of_snap,
                report_band, selected_for_report,
                frame_report.identifiability, normalized_gram,
                better_cut_candidate, frame_report.topology_epoch_id,
                frame_report
                    .component_soft_score_valid_assignments_within_domain,
                frame_report.spatial_lift_ambiguous_hard_edges,
                frame_report.spatial_lift_cycle_residuals,
                frame_report.status);
          }
          frame_report.time_graph_ms =
              max(0.0, (timing.graph_s + timing.component_graph_s -
                        frame_graph_s_before) *
                           1000.0);
          frame_report.time_virtual_ms =
              max(0.0, (timing.event_planes_s + timing.coarse_s +
                        timing.refine_s + timing.early_projection_s -
                        frame_virtual_s_before) *
                           1000.0);
          frame_report.time_component_ms =
              max(0.0, (timing.component_prep_s + timing.component_graph_s +
                        timing.component_search_s + timing.warm_component_s -
                        frame_component_s_before) *
                           1000.0);
          frame_report.time_total_ms =
              chrono::duration<double, milli>(clock::now() - frame_report_started)
                  .count();
          if (status == PbctopoFrameStatus::WeakObservation)
            ++timing.pbctopo_weak_observation_output_frames;
          if (status == PbctopoFrameStatus::Contradicted)
            ++timing.pbctopo_contradicted_output_frames;
          if ((status == PbctopoFrameStatus::Certified ||
               status == PbctopoFrameStatus::Rescued) &&
              frame_report.geometry_classification.evaluated) {
            if (frame_report.geometry_classification.geometry_class ==
                titan_pbctopo::PbctopoGeometryClass::CompactUnitBox) {
              ++timing.pbctopo_compact_unit_box_certified_frames;
            } else if (frame_report.geometry_classification.geometry_class ==
                       titan_pbctopo::PbctopoGeometryClass::Extended) {
              ++timing.pbctopo_noncompact_certified_frames;
              ++timing.pbctopo_extended_certified_frames;
            }
          }
          if ((status == PbctopoFrameStatus::Certified ||
               status == PbctopoFrameStatus::WeakObservation ||
               status == PbctopoFrameStatus::Rescued) &&
              frame_report.split_owner_pairs > 0) {
            ++timing.pbctopo_owner_com_split_observations;
          }
          write_or_defer_frame_report(frame_idx, frame_report);
        };
    auto accept_rescue_candidate =
        [&](const vector<Vec3> &candidate_points,
            PbctopoCutCandidate candidate) {
      if (!certify_candidate_for_output(candidate, candidate_points)) {
            ++timing.pbctopo_diagnostic_only_candidates;
            record_frame_failure(PbctopoFailureReason::CertificationReject);
        return false;
      }
          ensure_component_context();
          audit_selected_evidence(candidate_points, candidate.assignment);
          if (!selected_evidence_audit.evaluated) {
            warm_state = frame_start_warm_state;
            reset_kinematic_seed_history(warm_state, timing);
            temporal_cut_candidates[frame_idx].clear();
            report_certified_candidates.clear();
            report_rejected_candidates.clear();
            ++timing.pbctopo_diagnostic_only_candidates;
            record_frame_failure(
                PbctopoFailureReason::EvidenceHypothesisUnavailable);
            return false;
          }
          const bool evidence_contradicted =
              selected_evidence_audit.state ==
              PbctopoEvidenceState::Contradicted;
          const bool evidence_weak =
              selected_evidence_audit.state == PbctopoEvidenceState::WeakDisconnected ||
              selected_evidence_audit.state == PbctopoEvidenceState::WeakAmbiguous;
          if (!evidence_contradicted)
            record_temporal_certified_candidate(candidate, candidate_points);
          record_report_soft_candidate(candidate);
          best_safe_candidate = candidate;
          best_candidate = candidate;
          best_safe_points = candidate_points;
          safe_candidates = max<size_t>(safe_candidates, 1);
          projection_clash_skip_active =
              candidate_has_projection_clash_margin(best_safe_candidate,
                                                    runtime_cutoffs.clash_cutoff);
          if (evidence_contradicted) {
            warm_state = frame_start_warm_state;
          } else if (!evidence_weak && selected_relative_images_identified()) {
            if (!commit_certified_candidate_to_warm_state(
                    best_safe_candidate)) {
              ++timing.pbctopo_diagnostic_only_candidates;
              record_frame_failure(PbctopoFailureReason::CertificationReject);
              return false;
            }
          } else {
            warm_state = frame_start_warm_state;
            ++timing.pbctopo_weak_observation_warm_preserved;
          }
          apply_cut_candidate_timed(best_safe_points);
          return true;
        };
    placed_points.reserve(projection_atoms.size());
    best_safe_points.reserve(base_atoms.size());
    coarse_candidates.reserve(12);
    coarse_seed_options.reserve(virtual_search_plan.states.size());
    if (prefetched_safe_candidate.valid && !prefetched_safe_candidate.ultra_clash &&
        prefetched_safe_points.size() == base_atoms.size()) {
      best_candidate = prefetched_safe_candidate;
      best_safe_candidate = prefetched_safe_candidate;
      best_safe_points = prefetched_safe_points;
      safe_candidates = 1;
      record_temporal_certified_candidate(prefetched_safe_candidate,
                                          prefetched_safe_points);
      record_report_soft_candidate(prefetched_safe_candidate);
      record_output_candidate(prefetched_safe_candidate,
                              prefetched_safe_points,
                              prefetched_safe_mode);
    }

    stage_started = clock::now();
    const bool has_kinematic_hint = any_of(
        virtual_seed_hints.begin(), virtual_seed_hints.end(),
        [](const PbctopoVirtualDiscreteSeedHint &hint) {
          return hint.source == PbctopoVirtualSeedSource::Kinematic;
        });
    size_t kinematic_states_evaluated = 0;
    for (size_t state_idx = 0;
         state_idx < virtual_search_plan.states.size(); ++state_idx) {
      const auto &state = virtual_search_plan.states[state_idx];
      PbctopoSeed3D seed;
      seed.cx = state.cx;
      seed.cy = state.cy;
      seed.cz = state.cz;
      // Density and temporal hints order exact states only; they are not part
      // of the physical candidate score.
      seed.score = 0.0;
      seed.ix = state.ix;
      seed.iy = state.iy;
      seed.iz = state.iz;
      seed.source = state.source;
      if (seed.source == PbctopoVirtualSeedSource::Kinematic) {
        ++timing.pbctopo_kinematic_seed_evaluated;
        ++kinematic_states_evaluated;
      }
      coarse_seed_options.push_back(
          {static_cast<uint64_t>(state_idx), seed});
    }
    if (has_kinematic_hint && kinematic_states_evaluated == 0)
      ++timing.pbctopo_kinematic_seed_deduped;
    timing.coarse_candidates += coarse_seed_options.size();
    const size_t requested_threads =
        (Arguments.nt > 1) ? Arguments.nt : static_cast<size_t>(1);
    const size_t executor_capacity =
        (executor != nullptr) ? max<size_t>(1, executor->capacity()) : requested_threads;
    const size_t coarse_threads =
        min(executor_capacity, coarse_seed_options.size());
    atomic<size_t> coarse_projection_skip_evals{0};
    atomic<size_t> coarse_worker_lattice_queries{0};
    atomic<size_t> coarse_worker_lattice_failures{0};
    atomic<size_t> coarse_worker_lattice_ambiguities{0};
    atomic<size_t> coarse_worker_lattice_budget_failures{0};
    auto accumulate_worker_lattice_audit =
        [&](const titan_pbctopo::PbctopoLatticeQueryAudit &before) {
          const auto after =
              titan_pbctopo::pbctopo_lattice_query_audit();
          const auto delta =
              titan_pbctopo::pbctopo_lattice_query_audit_delta(before, after);
          coarse_worker_lattice_queries.fetch_add(delta.queries,
                                                   memory_order_relaxed);
          coarse_worker_lattice_failures.fetch_add(delta.failures,
                                                    memory_order_relaxed);
          coarse_worker_lattice_ambiguities.fetch_add(delta.ambiguous,
                                                       memory_order_relaxed);
          coarse_worker_lattice_budget_failures.fetch_add(
              delta.enumeration_budget_failures, memory_order_relaxed);
        };
    coarse_results.assign(coarse_seed_options.size(), PbctopoCutCandidate{});
    if (executor != nullptr && coarse_threads > 1 && coarse_seed_options.size() >= 4 &&
        projection_atoms.size() >= 2000) {
      executor->parallel_for(
          coarse_seed_options.size(), coarse_threads, 1,
          [&](size_t, size_t begin, size_t end) {
            const auto worker_lattice_audit_before =
                titan_pbctopo::pbctopo_lattice_query_audit();
            vector<Vec3> local_points;
            local_points.reserve(projection_atoms.size());
            for (size_t option_idx = begin; option_idx < end; ++option_idx) {
              const auto &seed = coarse_seed_options[option_idx].seed;
              coarse_results[option_idx] = evaluate_virtual_box_candidate(
                  projection_atoms, projection_edges, projection_edge_set,
                  translation_cache, this->m_mol[frame_idx].Box, seed.cx,
                  seed.cy, seed.cz, seed.score,
                  local_points, runtime_cutoffs.clash_cutoff,
                  &projection_continuity_pairs, &projection_anchor_indices,
                  &projection_hard_required_owner_pairs,
                  runtime_cutoffs.continuity_cutoff, nullptr,
                  skip_projection_candidate_clash,
                  numeric_limits<double>::quiet_NaN(), nullptr,
                  &projection_anchor_context, seed.source);
              if (skip_projection_candidate_clash)
                  coarse_projection_skip_evals.fetch_add(
                    1, memory_order_relaxed);
            }
            accumulate_worker_lattice_audit(worker_lattice_audit_before);
          });
    } else if (coarse_threads > 1 && coarse_seed_options.size() >= 4 &&
               projection_atoms.size() >= 2000) {
      vector<titan::jthread> workers;
      workers.reserve(coarse_threads);
      const size_t chunk =
          (coarse_seed_options.size() + coarse_threads - 1) / coarse_threads;
      for (size_t thread_idx = 0; thread_idx < coarse_threads; ++thread_idx) {
        const size_t begin = thread_idx * chunk;
        const size_t end = min(begin + chunk, coarse_seed_options.size());
        if (begin >= end)
          continue;
        workers.emplace_back([&, begin, end]() {
          const auto worker_lattice_audit_before =
              titan_pbctopo::pbctopo_lattice_query_audit();
          vector<Vec3> local_points;
          local_points.reserve(projection_atoms.size());
          for (size_t option_idx = begin; option_idx < end; ++option_idx) {
            const auto &seed = coarse_seed_options[option_idx].seed;
            coarse_results[option_idx] = evaluate_virtual_box_candidate(
                projection_atoms, projection_edges, projection_edge_set,
                translation_cache, this->m_mol[frame_idx].Box, seed.cx, seed.cy,
                seed.cz, seed.score,
                local_points, runtime_cutoffs.clash_cutoff,
                &projection_continuity_pairs, &projection_anchor_indices,
                &projection_hard_required_owner_pairs,
                runtime_cutoffs.continuity_cutoff, nullptr,
                skip_projection_candidate_clash,
                numeric_limits<double>::quiet_NaN(), nullptr,
                &projection_anchor_context, seed.source);
            if (skip_projection_candidate_clash)
              coarse_projection_skip_evals.fetch_add(
                  1, memory_order_relaxed);
          }
          accumulate_worker_lattice_audit(worker_lattice_audit_before);
        });
      }
    } else {
      for (size_t option_idx = 0; option_idx < coarse_seed_options.size();
           ++option_idx) {
        const auto &seed = coarse_seed_options[option_idx].seed;
        coarse_results[option_idx] = evaluate_virtual_box_candidate(
            projection_atoms, projection_edges, projection_edge_set,
            translation_cache, this->m_mol[frame_idx].Box, seed.cx, seed.cy,
            seed.cz, seed.score,
            placed_points, runtime_cutoffs.clash_cutoff,
            &projection_continuity_pairs, &projection_anchor_indices,
            &projection_hard_required_owner_pairs, runtime_cutoffs.continuity_cutoff,
            nullptr, skip_projection_candidate_clash,
            numeric_limits<double>::quiet_NaN(), nullptr,
            &projection_anchor_context, seed.source);
        if (skip_projection_candidate_clash)
          coarse_projection_skip_evals.fetch_add(1, memory_order_relaxed);
      }
    }
    timing.projection_clash_skip_evals +=
        coarse_projection_skip_evals.load(memory_order_relaxed);
    const size_t worker_lattice_queries =
        coarse_worker_lattice_queries.load(memory_order_relaxed);
    const size_t worker_lattice_failures =
        coarse_worker_lattice_failures.load(memory_order_relaxed);
    const size_t worker_lattice_ambiguities =
        coarse_worker_lattice_ambiguities.load(memory_order_relaxed);
    const size_t worker_lattice_budget_failures =
        coarse_worker_lattice_budget_failures.load(memory_order_relaxed);
    timing.pbctopo_lattice_worker_queries += worker_lattice_queries;
    timing.pbctopo_lattice_worker_query_failures += worker_lattice_failures;
    timing.pbctopo_lattice_worker_query_ambiguities +=
        worker_lattice_ambiguities;
    timing.pbctopo_lattice_worker_query_budget_failures +=
        worker_lattice_budget_failures;
    timing.pbctopo_lattice_query_failures += worker_lattice_failures;
    timing.pbctopo_lattice_optional_query_failures += worker_lattice_failures;
    timing.pbctopo_lattice_query_ambiguities += worker_lattice_ambiguities;
    timing.pbctopo_lattice_query_budget_failures +=
        worker_lattice_budget_failures;
    for (const auto &candidate : coarse_results) {
      if (better_preclash_cut_candidate(
              candidate, best_coarse_virtual_geometry_seed)) {
        best_coarse_virtual_geometry_seed = candidate;
      }
      push_top_virtual_box_candidate(coarse_candidates, candidate,
                                     kProjectionKeepCandidates);
    }
    timing.coarse_s += chrono::duration<double>(clock::now() - stage_started).count();

    stage_started = clock::now();
    verify_points.reserve(base_atoms.size());
    if (best_coarse_virtual_geometry_seed.valid) {
      ++timing.pbctopo_verify_geometry_buffer_reuses;
      ++timing.pbctopo_virtual_geometry_reference_evals;
      PbctopoCutCandidate geometry_reference = evaluate_virtual_box_candidate(
          base_atoms, local_edges, steric_exclusion_set, translation_cache,
          this->m_mol[frame_idx].Box,
          best_coarse_virtual_geometry_seed.cut_x,
          best_coarse_virtual_geometry_seed.cut_y,
          best_coarse_virtual_geometry_seed.cut_z,
          best_coarse_virtual_geometry_seed.gap_score,
          verify_points, runtime_cutoffs.clash_cutoff,
          &trajectory_context.continuity_pairs,
          &trajectory_context.continuity_anchor_indices,
          &local_hard_required_owner_pairs, runtime_cutoffs.continuity_cutoff,
          nullptr, false, this->m_mol[frame_idx].time_of_snap,
          "virtual_geometry_reference", &base_anchor_context,
          best_coarse_virtual_geometry_seed.virtual_seed_source,
          &atom_component_index);
      if (geometry_reference.valid) {
        if (geometry_reference.ultra_clash) {
          ++timing.pbctopo_virtual_geometry_reference_clashing;
        } else {
          best_virtual_geometry_reference = geometry_reference;
          ++timing.pbctopo_virtual_geometry_reference_safe;
          ++safe_candidates;
          record_temporal_certified_candidate(geometry_reference,
                                              verify_points);
          record_report_soft_candidate(geometry_reference);
          record_output_candidate(geometry_reference, verify_points,
                                  PbctopoFrameMode::Virtual);
          best_verified_virtual_candidate = geometry_reference;
          if (better_cut_candidate(geometry_reference, best_safe_candidate)) {
            best_safe_candidate = geometry_reference;
            best_safe_points = verify_points;
            frame_report.mode = PbctopoFrameMode::Virtual;
          }
          if (better_cut_candidate(geometry_reference, best_candidate)) {
            best_candidate = geometry_reference;
          }
        }
      }
    }
    size_t refine_limit = coarse_candidates.size();
    if (coarse_candidates.size() >= 2) {
      const auto &top1 = coarse_candidates[0];
      const auto &top2 = coarse_candidates[1];
      if (top1.broken_edges + 1 < top2.broken_edges ||
          top1.edge_stretch + 1.0e-6 < top2.edge_stretch * 0.85) {
        refine_limit = 1;
      } else if (coarse_candidates.size() >= 3) {
        const auto &top3 = coarse_candidates[2];
        if (top2.broken_edges + 1 < top3.broken_edges ||
            top2.edge_stretch + 1.0e-6 < top3.edge_stretch * 0.85) {
          refine_limit = 2;
        }
      }
    }
    for (size_t coarse_idx = 0; coarse_idx < refine_limit; ++coarse_idx) {
      const auto &coarse = coarse_candidates[coarse_idx];
      const PbctopoCutCandidate *verify_incumbent =
          (best_safe_candidate.valid && !best_safe_candidate.ultra_clash)
              ? &best_safe_candidate
              : nullptr;
      PbctopoCutCandidate verified_candidate = evaluate_virtual_box_candidate(
          base_atoms, local_edges, steric_exclusion_set, translation_cache,
          this->m_mol[frame_idx].Box,
          coarse.cut_x, coarse.cut_y, coarse.cut_z, coarse.gap_score,
          verify_points,
          runtime_cutoffs.clash_cutoff, &trajectory_context.continuity_pairs,
          &trajectory_context.continuity_anchor_indices,
          &local_hard_required_owner_pairs, runtime_cutoffs.continuity_cutoff,
          verify_incumbent, false, this->m_mol[frame_idx].time_of_snap,
          "verified_candidate", &base_anchor_context,
          coarse.virtual_seed_source, &atom_component_index);
      ++timing.refine_candidates;
      if (verified_candidate.valid)
        ++valid_candidates;
      if (!verified_candidate.valid)
        continue;
      const bool virtual_geometry_outlier = component_candidate_is_outlier(
          verified_candidate,
          best_virtual_geometry_reference.valid
              ? &best_virtual_geometry_reference
              : nullptr);
      if (!verified_candidate.ultra_clash) {
        if (virtual_geometry_outlier) {
          ++timing.pbctopo_virtual_geometry_outlier_observations;
          append_pbctopo_log_line(
              "[pbctopo] observe virtual geometry deviation at frame " +
              to_string(static_cast<long long>(
                  this->m_mol[frame_idx].time_of_snap)) +
              ": principal_rg2_max=" +
              to_string(verified_candidate.principal_rg2_max) +
              ", reference_principal_rg2_max=" +
              to_string(best_virtual_geometry_reference.principal_rg2_max) +
              ", kappa2=" + to_string(verified_candidate.shape_kappa2) +
              ", reference_kappa2=" +
              to_string(best_virtual_geometry_reference.shape_kappa2) +
              ", policy=soft_observation");
        }
        if (better_preclash_cut_candidate(
                verified_candidate, best_virtual_geometry_reference)) {
          best_virtual_geometry_reference = verified_candidate;
        }
        ++safe_candidates;
        record_temporal_certified_candidate(verified_candidate,
                                            verify_points);
        record_report_soft_candidate(verified_candidate);
        record_output_candidate(verified_candidate, verify_points,
                                PbctopoFrameMode::Virtual);
        if (better_cut_candidate(verified_candidate,
                                 best_verified_virtual_candidate)) {
          best_verified_virtual_candidate = verified_candidate;
        }
        if (better_cut_candidate(verified_candidate, best_safe_candidate)) {
          best_safe_candidate = verified_candidate;
          best_safe_points = verify_points;
          frame_report.mode = PbctopoFrameMode::Virtual;
        }
      }
      if (better_cut_candidate(verified_candidate, best_candidate)) {
        best_candidate = verified_candidate;
      }
    }
    timing.refine_s += chrono::duration<double>(clock::now() - stage_started).count();
    if (best_verified_virtual_candidate.valid &&
        !best_verified_virtual_candidate.ultra_clash &&
        best_verified_virtual_candidate.virtual_seed_source ==
            PbctopoVirtualSeedSource::Kinematic) {
      ++timing.pbctopo_kinematic_seed_best;
    }

    // Keep component scoring aligned with the original local graph.
    bool component_search_used = primary_component_graph_used;
    stage_started = clock::now();
    if (can_use_component_logic) {
      ++timing.component_frames;
      ensure_component_context();
      timing.component_prep_s +=
          chrono::duration<double>(clock::now() - stage_started).count();
      size_t effective_component_count = components.size();
      string component_graph_fallthrough_reason;
      const string component_mode_label =
          pbctopo_component_mode_name(component_mode);
      const PbctopoCutCandidate *component_warm_reference =
          (best_safe_candidate.valid && !best_safe_candidate.ultra_clash)
              ? &best_safe_candidate
              : ((warm_state.has_projection_candidate &&
                  warm_state.previous_projection_candidate.valid &&
                  !warm_state.previous_projection_candidate.ultra_clash)
                     ? &warm_state.previous_projection_candidate
                     : nullptr);
      auto accept_component_cut = [&](PbctopoCutCandidate &candidate,
                                       const vector<Vec3> &candidate_points,
                                       const vector<Int3> *accepted_offsets,
                                       const string &source_label) {
        if (!candidate.valid || candidate.ultra_clash ||
            candidate_points.size() != base_atoms.size()) {
          return false;
        }
        if (!candidate.hard_feasibility.certified() ||
            !candidate.assignment.valid) {
          if (!certify_candidate_for_output(candidate, candidate_points,
                                            accepted_offsets)) {
            record_frame_failure(PbctopoFailureReason::CertificationReject);
            append_pbctopo_log_line(
                "[pbctopo] component candidate failed independent output "
                "certificate at frame " +
                to_string(static_cast<long long>(
                    this->m_mol[frame_idx].time_of_snap)) +
                " via " + source_label + ": hard_clashes=" +
                to_string(candidate.hard_feasibility.hard_clashes) +
                ", assignment_point_residuals=" +
                to_string(
                    candidate.hard_feasibility.assignment_point_residuals) +
                ", topology_edge_residuals=" +
                to_string(candidate.hard_feasibility.hard_edge_residuals));
            return false;
          }
        }
        const PbctopoCutCandidate *virtual_cloud_reference =
            best_virtual_geometry_reference.valid &&
                    !best_virtual_geometry_reference.ultra_clash
                ? &best_virtual_geometry_reference
                : nullptr;
        if (component_candidate_is_outlier(candidate,
                                           virtual_cloud_reference)) {
          if (source_label.find("search") != string::npos ||
              source_label.find("beam") != string::npos) {
            ++timing.component_search_outlier_observations;
          } else {
            ++timing.component_graph_outlier_observations;
          }
          append_pbctopo_log_line(
              "[pbctopo] observe component geometry deviation at frame " +
              to_string(static_cast<long long>(
                  this->m_mol[frame_idx].time_of_snap)) +
              " via " + source_label + ": principal_rg2_max=" +
              to_string(candidate.principal_rg2_max) +
              ", virtual_principal_rg2_max=" +
              to_string(virtual_cloud_reference->principal_rg2_max) +
              ", kappa2=" + to_string(candidate.shape_kappa2) +
              ", virtual_kappa2=" +
              to_string(virtual_cloud_reference->shape_kappa2) +
              ", asphericity=" +
              to_string(candidate.shape_asphericity) +
              ", virtual_asphericity=" +
              to_string(virtual_cloud_reference->shape_asphericity) +
              ", policy=soft_observation");
        }
        PbctopoOwnerPairSplitDetail split_detail;
        size_t split_owner_pairs = 0;
        double max_owner_raw_minus_minimg_delta = 0.0;
        const bool owner_com_split = candidate_has_split_contacting_owners(
                candidate_points, base_atoms, this->m_mol[frame_idx].Box,
                local_hard_required_owner_pairs, split_owner_pairs,
                max_owner_raw_minus_minimg_delta, &split_detail);
        candidate.split_owner_pairs = split_owner_pairs;
        candidate.max_owner_raw_minus_minimg_delta =
            max_owner_raw_minus_minimg_delta;
        if (owner_com_split) {
          if (split_detail.valid) {
            append_pbctopo_log_line(
                "[pbctopo] observe owner_com_image_gap at frame " +
                to_string(static_cast<long long>(
                    this->m_mol[frame_idx].time_of_snap)) +
                " via " + source_label + ": owners=" +
                to_string(split_detail.owner_a) + "-" +
                to_string(split_detail.owner_b) +
                ", raw_delta_scaled=" +
                format_vec3(split_detail.raw_delta_scaled) +
                ", min_image_delta_scaled=" +
                format_vec3(split_detail.minimg_delta_scaled) +
                ", policy=diagnostic_only");
          }
        }
        bool accepted = false;
        bool better_than_best = false;
        if (!best_safe_candidate.valid ||
            better_cut_candidate(candidate, best_safe_candidate)) {
          best_safe_candidate = candidate;
          if (accepted_offsets != nullptr &&
              accepted_offsets->size() == components.size()) {
            best_safe_candidate.assignment = make_pbctopo_image_assignment(
                *accepted_offsets, components, component_atoms);
          }
          best_safe_points = candidate_points;
          accepted = true;
          better_than_best = true;
        } else if (candidate_close_enough_to_skip_search(candidate,
                                                         best_safe_candidate)) {
          accepted = true;
        }
        if (!accepted)
          return false;
        PbctopoFrameMode candidate_mode = PbctopoFrameMode::ComponentGraph;
        if (source_label.find("beam") != string::npos) {
          candidate_mode = PbctopoFrameMode::ComponentSearchBeam;
        } else if (source_label.find("search") != string::npos) {
          candidate_mode = PbctopoFrameMode::ComponentSearchExact;
        } else if (source_label.find("warm") != string::npos) {
          candidate_mode = PbctopoFrameMode::Warm;
        }
        record_temporal_certified_candidate(candidate, candidate_points,
                                            accepted_offsets);
        record_report_soft_candidate(candidate);
        record_output_candidate(candidate, candidate_points, candidate_mode,
                                accepted_offsets);
        frame_report.mode = candidate_mode;
        component_search_used = true;
        if (accepted_offsets != nullptr &&
            accepted_offsets->size() == components.size()) {
          warm_state.previous_component_offsets = *accepted_offsets;
        } else {
          warm_state.previous_component_offsets =
              extract_offsets_and_time(component_atoms, candidate_points);
        }
        if (better_than_best) {
          append_pbctopo_log_line(
              "[pbctopo] " + source_label + ": mode=" + component_mode_label +
              ", raw_components=" + to_string(raw_component_count) +
              ", effective_components=" +
              to_string(effective_component_count) + ", min_pair_d2=" +
              to_string(candidate.min_pair_d2) + ", kappa2=" +
              to_string(candidate.shape_kappa2) + ", asph=" +
              to_string(candidate.shape_asphericity) + ", bbox=" +
              to_string(candidate.bbox_volume));
        }
        return true;
      };
      auto component_candidate_as_cut =
          [&](const PbctopoComponentCandidate &component_candidate,
              const vector<Vec3> &candidate_points,
              const PbctopoCutCandidate *continuity_incumbent,
              PbctopoCutCandidate &component_as_cut) {
            if (!component_candidate.valid || component_candidate.ultra_clash ||
                candidate_points.size() != base_atoms.size()) {
              return false;
            }
            PointStats component_score_stats;
            for (const auto &point : candidate_points)
              component_score_stats.add(point.x, point.y, point.z);
            double continuity_path = numeric_limits<double>::infinity();
            double continuity_max_d2 = numeric_limits<double>::infinity();
            double continuity_min_margin_A =
                numeric_limits<double>::quiet_NaN();
            size_t continuity_margin_atom_a = numeric_limits<size_t>::max();
            size_t continuity_margin_atom_b = numeric_limits<size_t>::max();
            if (!continuity_metrics_ok(candidate_points,
                                       trajectory_context.continuity_pairs,
                                       runtime_cutoffs.continuity_cutoff,
                                       continuity_path, continuity_max_d2,
                                       continuity_min_margin_A,
                                       continuity_margin_atom_a,
                                       continuity_margin_atom_b,
                                       continuity_incumbent)) {
              return false;
            }
            component_as_cut =
                score_cut_candidate(component_scoring_edges, candidate_points,
                                    1.5, 1.5, 1.5, 0.0, component_score_stats,
                                    component_candidate.shift_norm2);
            component_as_cut.min_pair_d2 = component_candidate.min_pair_d2;
            component_as_cut.continuity_path = continuity_path;
            component_as_cut.continuity_max_d2 = continuity_max_d2;
            component_as_cut.continuity_min_margin_A =
                continuity_min_margin_A;
            component_as_cut.continuity_margin_atom_a =
                continuity_margin_atom_a;
            component_as_cut.continuity_margin_atom_b =
                continuity_margin_atom_b;
            component_as_cut.anchor_mst2 = component_candidate.anchor_mst2;
            component_as_cut.anchor_max_d2 = component_candidate.anchor_max_d2;
            component_as_cut.anchor_graph = component_candidate.anchor_graph;
            component_as_cut.clash_atom_a = component_candidate.clash_atom_a;
            component_as_cut.clash_atom_b = component_candidate.clash_atom_b;
            component_as_cut.component_search_shell_radius =
                component_candidate.component_search_shell_radius;
            component_as_cut.search_domain_audit_known =
                component_candidate.search_domain_audit_known;
            component_as_cut.search_domain_evidence_complete =
                component_candidate.search_domain_evidence_complete;
            component_as_cut.selected_assignment_touches_shell_boundary =
                component_candidate
                    .selected_assignment_touches_shell_boundary;
            component_as_cut.domain_expansion_attempted =
                component_candidate.domain_expansion_attempted;
            component_as_cut.domain_expansion_exhausted =
                component_candidate.domain_expansion_exhausted;
            component_as_cut.bounded_domain_only =
                component_candidate.bounded_domain_only;
            component_as_cut.ultra_clash = component_candidate.ultra_clash;
            component_as_cut.valid = component_candidate.valid;
            return component_as_cut.valid && !component_as_cut.ultra_clash;
          };
      auto accept_component_candidate =
          [&](const PbctopoComponentCandidate &component_candidate,
              const vector<Vec3> &candidate_points,
              const vector<Int3> *accepted_offsets,
              const string &source_label) {
            if (!component_candidate.valid || component_candidate.ultra_clash ||
                candidate_points.size() != base_atoms.size()) {
              return false;
            }
            PbctopoCutCandidate component_as_cut;
            if (!component_candidate_as_cut(
                    component_candidate, candidate_points,
                    &best_safe_candidate, component_as_cut)) {
              record_frame_failure(PbctopoFailureReason::ContinuityReject);
              return false;
            }
            return accept_component_cut(component_as_cut, candidate_points,
                                        accepted_offsets, source_label);
          };
      auto record_component_search_survivors =
          [&](const vector<PbctopoComponentSearchSurvivor> &survivors,
              PbctopoFrameMode mode,
              const PbctopoComponentCandidate *search_best,
              bool search_equivalence_known) {
            timing.component_search_certification_survivors +=
                survivors.size();
            vector<Vec3> survivor_points;
            survivor_points.reserve(base_atoms.size());
            for (const auto &survivor : survivors) {
              if (survivor.component_offsets.size() != components.size())
                continue;
              double shift_norm2 = 0.0;
              build_component_points_from_offsets(
                  component_atoms, components, this->m_mol[frame_idx].Box,
                  survivor.component_offsets, survivor_points, shift_norm2);
              PbctopoComponentCandidate scored_survivor = survivor.candidate;
              scored_survivor.shift_norm2 = shift_norm2;
              PbctopoCutCandidate survivor_as_cut;
              if (!component_candidate_as_cut(scored_survivor, survivor_points,
                                              nullptr, survivor_as_cut)) {
                continue;
              }
              survivor_as_cut.search_equivalence_known =
                  search_equivalence_known && search_best != nullptr &&
                  search_best->valid;
              survivor_as_cut.equivalent_to_search_best =
                  survivor_as_cut.search_equivalence_known &&
                  !better_component_candidate(scored_survivor, *search_best) &&
                  !better_component_candidate(*search_best, scored_survivor);
              if (record_output_candidate(survivor_as_cut, survivor_points,
                                          mode,
                                          &survivor.component_offsets)) {
                ++timing.component_search_certification_band_retained;
              }
            }
          };
      if (same_component_warm_layout) {
        vector<Int3> warm_offsets = warm_state.previous_component_offsets;
        if (!rebase_component_offsets(warm_offsets,
                                      this->m_mol[frame_idx].Box)) {
          warm_offsets.clear();
        }
        PbctopoCutCandidate warm_cut = improve_component_offsets_locally(
            component_atoms, components, this->m_mol[frame_idx].Box, shifts,
            component_scoring_edges, steric_exclusion_set,
            trajectory_context.continuity_pairs,
            trajectory_context.continuity_anchor_indices, warm_offsets,
            runtime_cutoffs.clash_cutoff,
            runtime_cutoffs.continuity_cutoff, component_points,
            make_component_refine_callbacks(), anchor_backend, &timing,
            &warm_state, component_warm_reference, &base_anchor_context);
        PbctopoOwnerPairSplitDetail warm_cut_split_detail;
        size_t warm_cut_split_owner_pairs = 0;
        double warm_cut_split_gap = 0.0;
        const bool warm_cut_split =
            warm_cut.valid && !warm_cut.ultra_clash &&
            candidate_has_split_contacting_owners(
                component_points, base_atoms, this->m_mol[frame_idx].Box,
                local_hard_required_owner_pairs, warm_cut_split_owner_pairs,
                warm_cut_split_gap, &warm_cut_split_detail);
        warm_cut.split_owner_pairs = warm_cut_split_owner_pairs;
        warm_cut.max_owner_raw_minus_minimg_delta = warm_cut_split_gap;
        if (warm_cut_split && warm_cut_split_detail.valid) {
          append_pbctopo_log_line(
              "[pbctopo] observe owner_com_image_gap at frame " +
              to_string(
                  static_cast<long long>(this->m_mol[frame_idx].time_of_snap)) +
              " via warm_component_local_improve: owners=" +
              to_string(warm_cut_split_detail.owner_a) + "-" +
              to_string(warm_cut_split_detail.owner_b) +
              ", raw_delta_scaled=" +
              format_vec3(warm_cut_split_detail.raw_delta_scaled) +
              ", min_image_delta_scaled=" +
              format_vec3(warm_cut_split_detail.minimg_delta_scaled) +
              ", policy=diagnostic_only");
        }
        if (warm_cut.valid && !warm_cut.ultra_clash) {
          bool warm_accepted = false;
          if (!best_safe_candidate.valid ||
              better_cut_candidate(warm_cut, best_safe_candidate)) {
            best_safe_candidate = warm_cut;
            best_safe_points = component_points;
            warm_accepted = true;
            frame_report.mode = PbctopoFrameMode::Warm;
            record_temporal_certified_candidate(warm_cut, component_points,
                                                &warm_offsets);
            record_report_soft_candidate(warm_cut);
            record_output_candidate(warm_cut, component_points,
                                    PbctopoFrameMode::Warm,
                                    &warm_offsets);
          } else if (candidate_close_enough_to_skip_search(
                         warm_cut, best_safe_candidate)) {
            warm_accepted = true;
          }
          if (warm_accepted) {
            ++timing.component_warm_used;
            warm_state.previous_component_offsets = warm_offsets;
          }
        }
      }
      if (component_mode == PbctopoComponentMode::Exact) {
        double component_shift_norm2 = 0.0;
        PbctopoCutCandidate graph_as_cut;
        vector<Int3> graph_tree_offsets;
        vector<Int3> graph_refined_offsets;
        vector<PbctopoGraphEdge> graph_tree_edges;
        PbctopoCycleSyncMetrics graph_cycle_metrics;
        if (!component_search_used) {
          PbctopoWarmState graph_trial_warm_state = warm_state;
          const auto component_graph_started = clock::now();
          const bool graph_ok = solve_component_offsets_via_pair_graph(
              component_atoms, this->m_mol[frame_idx].Box, components, shifts,
              component_scoring_edges, steric_exclusion_set, component_points,
              trajectory_context.continuity_pairs,
              trajectory_context.continuity_anchor_indices, runtime_cutoffs,
              graph_as_cut, component_shift_norm2, beam_width, &timing,
              make_component_refine_callbacks(), anchor_backend,
              same_component_warm_layout ? &warm_state.previous_component_offsets
                                         : nullptr,
              &graph_trial_warm_state, component_warm_reference,
              &graph_tree_offsets,
              &graph_refined_offsets, &graph_tree_edges, &graph_cycle_metrics,
              &base_anchor_context, &explicit_required_contacts,
              &frame_identifiability);
          timing.component_graph_s +=
              chrono::duration<double>(clock::now() - component_graph_started)
                  .count();
          frame_report.soft_pair_cycle_support =
              graph_cycle_metrics.cycle_support;
          frame_report.soft_pair_cycle_mismatch =
              graph_cycle_metrics.cycle_mismatch;
          frame_report.soft_pair_inconsistent_edges =
              graph_cycle_metrics.inconsistent_edge_count;
          frame_report.soft_pair_fundamental_cycle_count =
              graph_cycle_metrics.fundamental_cycle_count;
          frame_report.soft_pair_fundamental_cycle_residual_rms =
              graph_cycle_metrics.fundamental_cycle_residual_rms;
          frame_report.soft_pair_fundamental_cycle_residual_max =
              graph_cycle_metrics.fundamental_cycle_residual_max;
          if (!graph_ok) {
            ++timing.component_graph_failed;
            component_graph_fallthrough_reason = "graph_fail";
            record_frame_failure(PbctopoFailureReason::ComponentGraphFailed);
            clear_component_trial_hints(warm_state);
          }
          if (graph_ok && component_points.size() != base_atoms.size()) {
            ++timing.component_graph_points_mismatch;
            component_graph_fallthrough_reason = "points_mismatch";
          }
          if (graph_ok && component_points.size() == base_atoms.size()) {
            if (!pbctopo_relative_images_identified(frame_identifiability)) {
              ++timing.component_graph_unidentified_output_rejects;
              component_graph_fallthrough_reason = "evidence_disconnected";
              record_frame_failure(PbctopoFailureReason::EvidenceDisconnected);
              append_pbctopo_log_line(
                  "[pbctopo] component graph excluded from output band at "
                  "frame " +
                  to_string(static_cast<long long>(
                      this->m_mol[frame_idx].time_of_snap)) +
                  ": identified=0, evidence_components=" +
                  to_string(frame_identifiability
                                .evidence_connected_component_count) +
                  ", policy=continue_virtual_and_exact_search");
            } else if (accept_component_cut(graph_as_cut, component_points,
                                            &graph_refined_offsets,
                                            "component graph solve")) {
              warm_state = std::move(graph_trial_warm_state);
              ++timing.component_graph_used;
            } else if (!graph_as_cut.valid) {
              ++timing.component_graph_invalid;
              component_graph_fallthrough_reason = "invalid";
              record_frame_failure(PbctopoFailureReason::ProjectionInvalid);
            } else if (graph_as_cut.ultra_clash) {
              ++timing.component_graph_ultraclash;
              component_graph_fallthrough_reason = "ultra_clash";
              record_frame_failure(PbctopoFailureReason::FullVerifyClash);
            } else {
              ++timing.component_graph_not_better_not_close;
              record_frame_failure(PbctopoFailureReason::NotBetterThanIncumbent);
              const char *close_reject_reason =
                  candidate_skip_search_reject_reason(graph_as_cut,
                                                      best_safe_candidate);
              component_graph_fallthrough_reason =
                  string("not_better_not_close/") +
                  (close_reject_reason != nullptr ? close_reject_reason
                                                 : "unknown");
            }
          }
        }

        if (!component_search_used || audit_exact_search ||
            components.size() == 2) {
          ++timing.component_search_frames;
          const PbctopoImageAssignment audit_reference_assignment =
              best_safe_candidate.assignment;
          auto component_clash_atom_summary =
              [&](size_t atom_idx) -> string {
            if (atom_idx >= component_atoms.size())
              return "na";
            size_t component_idx = numeric_limits<size_t>::max();
            for (size_t idx = 0; idx < components.size(); ++idx) {
              if (binary_search(components[idx].begin(), components[idx].end(),
                                atom_idx)) {
                component_idx = idx;
                break;
              }
            }
            const PbctopoAtom &atom = component_atoms[atom_idx];
            return to_string(atom_idx) + "/global=" +
                   to_string(atom.global_atom_id) + "/owner=" +
                   to_string(atom.owner) + "/component=" +
                   (component_idx == numeric_limits<size_t>::max()
                        ? string("na")
                        : to_string(component_idx)) +
                   "/heavy=" + (atom.is_heavy ? "yes" : "no");
          };
          append_pbctopo_log_line(
              "[pbctopo] component graph fell through to search at frame " +
              to_string(static_cast<long long>(
                  this->m_mol[frame_idx].time_of_snap)) +
              ": mode=" + component_mode_label +
              ", raw_components=" + to_string(raw_component_count) +
              ", effective_components=" +
              to_string(effective_component_count) + ", reason=" +
              component_graph_fallthrough_reason + ", anchor_count=" +
              to_string(trajectory_context.continuity_anchor_indices.size()) +
              ", anchor_policy=full" +
              ", graph_valid=" + string(graph_as_cut.valid ? "yes" : "no") +
              ", graph_ultra_clash=" +
              string(graph_as_cut.ultra_clash ? "yes" : "no") +
              ", graph_anchor_max_d2=" + to_string(graph_as_cut.anchor_max_d2) +
              ", ref_anchor_max_d2=" + to_string(best_safe_candidate.anchor_max_d2) +
              ", graph_anchor_mst2=" + to_string(graph_as_cut.anchor_mst2) +
              ", ref_anchor_mst2=" + to_string(best_safe_candidate.anchor_mst2) +
              ", graph_min_pair_d2=" + to_string(graph_as_cut.min_pair_d2) +
              ", graph_clash_atom_a=" +
              component_clash_atom_summary(graph_as_cut.clash_atom_a) +
              ", graph_clash_atom_b=" +
              component_clash_atom_summary(graph_as_cut.clash_atom_b) +
              ", graph_kappa2=" + to_string(graph_as_cut.shape_kappa2) +
              ", graph_asph=" + to_string(graph_as_cut.shape_asphericity) +
              ", graph_bbox=" + to_string(graph_as_cut.bbox_volume));
          append_pbctopo_log_line(
              "[pbctopo] component graph offsets at frame " +
              to_string(static_cast<long long>(
                  this->m_mol[frame_idx].time_of_snap)) +
              ": tree_offsets=" + format_component_offsets(graph_tree_offsets) +
              ", refined_offsets=" + format_component_offsets(graph_refined_offsets) +
              ", tree_edges=" + format_graph_edges(graph_tree_edges));
          const auto component_search_started = clock::now();
          const size_t exact_state_upper_bound =
              pbctopo_component_assignment_upper_bound(
                  components.size(), shifts.size(),
                  kPbctopoExactAssignmentBudget + 1);
          timing.component_search_exact_state_upper_bound =
              max(timing.component_search_exact_state_upper_bound,
                  exact_state_upper_bound);
          const bool use_budgeted_beam =
              !audit_exact_search &&
              exact_state_upper_bound > kPbctopoExactAssignmentBudget;
          const bool retain_complete_feasible_band =
              !use_budgeted_beam &&
              exact_state_upper_bound <= kMaxCompleteOutputCertificationBand;
          const size_t component_search_survivor_limit =
              retain_complete_feasible_band
                  ? max<size_t>(1, exact_state_upper_bound)
                  : kMaxOutputCertificationBand;
          output_certification_band_limit =
              max(output_certification_band_limit,
                  component_search_survivor_limit);
          const PbctopoComponentSearchLimits complete_band_limits{
              0, 0.0, 0, true};
          const PbctopoComponentSearchLimits *component_search_limits =
              audit_exact_search
                  ? &audit_exact_limits
                  : (retain_complete_feasible_band ? &complete_band_limits
                                                   : nullptr);
          if (use_budgeted_beam) {
            ++timing.component_search_exact_budget_skips;
            append_pbctopo_log_line(
                "[pbctopo] replace exact search with beam at frame " +
                to_string(static_cast<long long>(
                    this->m_mol[frame_idx].time_of_snap)) +
                ": assignment_upper_bound=" +
                to_string(exact_state_upper_bound) +
                ", budget=" + to_string(kPbctopoExactAssignmentBudget));
          }
          const auto search_incumbent =
              (best_safe_candidate.valid && !best_safe_candidate.ultra_clash)
                  ? &best_safe_candidate
                  : nullptr;
          const auto search_incumbent_points =
              (search_incumbent != nullptr &&
               best_safe_points.size() == base_atoms.size())
                  ? &best_safe_points
                  : nullptr;
          PbctopoComponentSearchDiagnostics search_diagnostics;
          vector<PbctopoComponentSearchSurvivor> search_survivors;
          PbctopoComponentCandidate component_candidate =
              use_budgeted_beam
                  ? search_component_shift_candidate_beam(
                        component_atoms, this->m_mol[frame_idx].Box,
                        components, shifts,
                        trajectory_context.continuity_pairs,
                        trajectory_context.continuity_anchor_indices,
                        runtime_cutoffs, component_points, beam_width,
                        make_frame_component_search_callbacks(),
                        same_component_warm_layout
                            ? &warm_state.previous_component_offsets
                            : nullptr,
                        search_incumbent, search_incumbent_points, &timing,
                        anchor_backend, &base_anchor_context,
                        kPbctopoExactAssignmentBudget, &search_diagnostics,
                        &search_survivors, component_search_survivor_limit,
                        &explicit_required_contacts)
                  : search_component_shift_candidate(
                        component_atoms, this->m_mol[frame_idx].Box,
                        components, shifts,
                        trajectory_context.continuity_pairs,
                        trajectory_context.continuity_anchor_indices,
                        runtime_cutoffs, component_points,
                        make_frame_component_search_callbacks(),
                        same_component_warm_layout
                            ? &warm_state.previous_component_offsets
                            : nullptr,
                        search_incumbent, search_incumbent_points, &timing,
                        anchor_backend, &base_anchor_context,
                        &search_diagnostics, &search_survivors,
                        component_search_survivor_limit,
                        &explicit_required_contacts,
                        component_search_limits);
          frame_report.component_search_attempted =
              search_diagnostics.attempted;
          frame_report.component_search_completed_under_admissible_bounds =
              search_diagnostics.search_completed_under_admissible_bounds;
          frame_report.component_search_incumbent_bound_pruning_used =
              search_diagnostics.incumbent_bound_pruning_used;
          frame_report.component_search_best_objective_proven =
              search_diagnostics.best_objective_proven;
          frame_report.component_search_equivalent_optima_complete =
              search_diagnostics.equivalent_optima_complete;
          frame_report.component_feasible_assignment_set_enumerated =
              search_diagnostics.feasible_assignment_set_enumerated;
          frame_report.component_feasible_assignments_within_domain =
              search_diagnostics.feasible_assignments_within_domain;
          complete_feasible_band_retained =
              retain_complete_feasible_band &&
              search_diagnostics.feasible_assignment_set_enumerated &&
              search_survivors.size() ==
                  search_diagnostics.feasible_assignments_within_domain;
          frame_report
              .component_soft_score_valid_assignments_within_domain =
              search_diagnostics.soft_score_valid_assignments_within_domain;
          frame_report.component_bounded_domain_feasibility_known =
              search_diagnostics.bounded_domain_feasibility_known;
          frame_report.component_bounded_domain_feasible =
              search_diagnostics.bounded_domain_feasible;
          frame_report.component_search_shell_radius =
              search_diagnostics.component_search_shell_radius;
          frame_report.domain_expansion_attempted =
              search_diagnostics.domain_expansion_attempted;
          frame_report.domain_expansion_exhausted =
              search_diagnostics.domain_expansion_exhausted;
          frame_report.bounded_domain_only =
              search_diagnostics.bounded_domain_only;
          frame_report.component_search_exhaustive =
              search_diagnostics.exhaustive;
          frame_report.component_offset_enumeration_exhaustive =
              search_diagnostics.offset_enumeration_exhaustive;
          frame_report.component_all_atom_feasibility_exhaustive =
              search_diagnostics.all_atom_feasibility_exhaustive;
          frame_report.component_certificate_search_exhaustive =
              search_diagnostics.certificate_search_exhaustive;
          frame_report.output_certificate_search_exhaustive =
              frame_report.output_certificate_search_exhaustive ||
              search_diagnostics.certificate_search_exhaustive;
          titan_pbctopo::apply_pbctopo_search_identifiability(
              frame_identifiability,
              search_diagnostics.certificate_search_exhaustive,
              search_diagnostics.equivalent_optima_complete,
              search_diagnostics.equivalent_best_assignments_within_domain,
              search_diagnostics.feasible_assignment_set_enumerated,
              search_diagnostics.feasible_assignments_within_domain);
          frame_report.beam_survivors_exactly_rescored =
              search_diagnostics.beam_survivors_exactly_rescored;
          frame_report.component_search_optimality_gap_known =
              search_diagnostics.optimality_gap_known;
          frame_report.component_search_optimality_gap =
              search_diagnostics.optimality_gap;
          append_pbctopo_log_line(
              "[pbctopo] component search domain audit at frame " +
              to_string(frame_report.frame) + ": shell_radius=" +
              to_string(search_diagnostics.component_search_shell_radius) +
              ", selected_boundary_known=" +
              string(search_diagnostics.selected_assignment_boundary_known
                         ? "yes"
                         : "no") +
              ", selected_touches_boundary=" +
              string(search_diagnostics.selected_assignment_boundary_known
                         ? (search_diagnostics
                                    .selected_assignment_touches_shell_boundary
                                ? "yes"
                                : "no")
                         : "na") +
              ", expansion_attempted=" +
              string(search_diagnostics.domain_expansion_attempted ? "yes"
                                                                    : "no") +
              ", expansion_exhausted=" +
              string(search_diagnostics.domain_expansion_exhausted ? "yes"
                                                                    : "no") +
              ", bounded_domain_only=" +
              string(search_diagnostics.bounded_domain_only ? "yes" : "no") +
              ", hard_feasible_assignments=" +
              to_string(
                  search_diagnostics.feasible_assignments_within_domain) +
              ", soft_score_valid_assignments=" +
              to_string(search_diagnostics
                            .soft_score_valid_assignments_within_domain));
          timing.component_search_s +=
              chrono::duration<double>(clock::now() - component_search_started)
                  .count();
          frame_report.component_audit_exact_completed =
              audit_exact_search &&
              !search_diagnostics.resource_limit_hit &&
              search_diagnostics.certificate_search_exhaustive &&
              search_diagnostics.feasible_assignment_set_enumerated;
          frame_report.component_audit_exact_incomplete =
              audit_exact_search &&
              !frame_report.component_audit_exact_completed;
          frame_report.component_audit_exact_incomplete_reason =
              !frame_report.component_audit_exact_incomplete
                  ? "none"
                  : search_diagnostics.resource_limit_hit
                        ? pbctopo_search_stop_reason_name(
                              search_diagnostics.stop_reason)
                        : !search_diagnostics.certificate_search_exhaustive
                              ? "certificate_domain_incomplete"
                              : "feasible_set_not_enumerated";
          frame_report.component_audit_exact_states_visited =
              audit_exact_search ? search_diagnostics.states_visited : 0;
          frame_report.component_audit_exact_wall_seconds =
              audit_exact_search ? search_diagnostics.wall_seconds : 0.0;
          frame_report.component_audit_exact_memory_bytes_estimate =
              audit_exact_search
                  ? search_diagnostics.memory_bytes_estimate
                  : 0;
          if (frame_report.component_audit_exact_completed)
            ++timing.component_audit_exact_completed;
          if (frame_report.component_audit_exact_incomplete) {
            ++timing.component_audit_exact_incomplete;
            switch (search_diagnostics.stop_reason) {
            case PbctopoSearchStopReason::StateLimit:
              ++timing.component_audit_exact_state_limit;
              break;
            case PbctopoSearchStopReason::WallTimeLimit:
              ++timing.component_audit_exact_wall_time_limit;
              break;
            case PbctopoSearchStopReason::MemoryLimit:
              ++timing.component_audit_exact_memory_limit;
              break;
            case PbctopoSearchStopReason::None:
              break;
            }
            append_pbctopo_log_line(
                "[pbctopo] audit-exact incomplete at frame " +
                to_string(static_cast<long long>(
                    this->m_mol[frame_idx].time_of_snap)) +
                ": reason=" +
                frame_report.component_audit_exact_incomplete_reason +
                ", states=" +
                to_string(search_diagnostics.states_visited) +
                ", wall_s=" +
                to_string(search_diagnostics.wall_seconds) +
                ", memory_bytes_estimate=" +
                to_string(search_diagnostics.memory_bytes_estimate) +
                ", action=retain_certified_initializer");
          } else {
            record_component_search_survivors(
                search_survivors,
                search_diagnostics.beam_used
                    ? PbctopoFrameMode::ComponentSearchBeam
                    : PbctopoFrameMode::ComponentSearchExact,
                &component_candidate,
                !search_diagnostics.beam_used &&
                    search_diagnostics.certificate_search_exhaustive &&
                    search_diagnostics.equivalent_optima_complete);
          }
          vector<Int3> search_offsets;
          PbctopoImageAssignment audited_assignment;
          if (!frame_report.component_audit_exact_incomplete &&
              component_candidate.valid && !component_candidate.ultra_clash &&
              component_points.size() == component_atoms.size()) {
            search_offsets =
                extract_offsets_and_time(component_atoms, component_points);
            if (search_offsets.size() == components.size()) {
              audited_assignment = make_pbctopo_image_assignment(
                  search_offsets, components, component_atoms);
            }
          }
          if (frame_report.component_audit_exact_completed &&
              audit_reference_assignment.valid && audited_assignment.valid) {
            frame_report.component_audit_exact_reference_comparison_available =
                true;
            frame_report.component_audit_exact_reference_assignment_match =
                same_pbctopo_image_assignment(audit_reference_assignment,
                                               audited_assignment);
            if (frame_report
                    .component_audit_exact_reference_assignment_match) {
              ++timing.component_audit_exact_reference_matches;
            } else {
              ++timing.component_audit_exact_reference_mismatches;
              append_pbctopo_log_line(
                  "[pbctopo] audit-exact differs from certified initializer "
                  "at frame " +
                  to_string(static_cast<long long>(
                      this->m_mol[frame_idx].time_of_snap)) +
                  ": initializer_assignment=" +
                  to_string(audit_reference_assignment.signature) +
                  ", exact_assignment=" +
                  to_string(audited_assignment.signature));
            }
          }
          const vector<Int3> *search_offsets_ptr =
              search_offsets.size() == components.size() ? &search_offsets
                                                         : nullptr;
          if (!frame_report.component_audit_exact_incomplete &&
              accept_component_candidate(
                  component_candidate, component_points, search_offsets_ptr,
                  search_diagnostics.beam_used
                      ? "component beam budget winner"
                      : "component search winner")) {
            warm_state.previous_component_offsets = search_offsets;
            append_pbctopo_log_line(
                "[pbctopo] component search winner offsets at frame " +
                to_string(static_cast<long long>(
                    this->m_mol[frame_idx].time_of_snap)) +
                ": offsets=" + format_component_offsets(search_offsets));
          }
          if (component_search_used || !best_safe_candidate.valid) {
            append_pbctopo_log_line(
                "[pbctopo] component-shift search: mode=" + component_mode_label +
                ", raw_components=" + to_string(raw_component_count) +
                ", effective_components=" +
                to_string(effective_component_count) + ", components=" +
                to_string(components.size()) + ", ultra_clash=" +
                string(component_candidate.ultra_clash ? "yes" : "no") +
                ", min_pair_d2=" + to_string(component_candidate.min_pair_d2) +
                ", kappa2=" + to_string(component_candidate.shape_kappa2) +
                ", asph=" + to_string(component_candidate.shape_asphericity) +
                ", bbox=" + to_string(component_candidate.bbox_volume));
          }
        }
      } else if (component_mode == PbctopoComponentMode::MergeThenGraph) {
        ++timing.component_large_mode_merge_frames;
        const PbctopoMergedComponentPlan merge_plan =
            build_chain_merged_component_plan(
                component_atoms, this->m_mol[frame_idx].Box, components);
        effective_component_count = merge_plan.merged_components.size();
        if (raw_component_count > effective_component_count)
          timing.component_large_mode_merge_count +=
              raw_component_count - effective_component_count;
        const size_t merged_exact_state_upper_bound =
            pbctopo_component_assignment_upper_bound(
                effective_component_count, shifts.size(),
                kPbctopoExactAssignmentBudget + 1);
        timing.component_search_exact_state_upper_bound =
            max(timing.component_search_exact_state_upper_bound,
                merged_exact_state_upper_bound);
        const bool merged_exact_within_budget =
            merged_exact_state_upper_bound <= kPbctopoExactAssignmentBudget;
        const bool allow_exact_search_on_merged =
            effective_component_count <= exact_component_limit &&
            merged_exact_within_budget;
        if (effective_component_count <= exact_component_limit &&
            !merged_exact_within_budget) {
          ++timing.component_search_exact_budget_skips;
          append_pbctopo_log_line(
              "[pbctopo] skip merged exact search at frame " +
              to_string(static_cast<long long>(
                  this->m_mol[frame_idx].time_of_snap)) +
              ": assignment_upper_bound=" +
              to_string(merged_exact_state_upper_bound) +
              ", budget=" + to_string(kPbctopoExactAssignmentBudget) +
              ", policy=component_graph_refine");
        }
        if (!allow_exact_search_on_merged)
          ++timing.component_large_mode_pair_only_frames;

        PbctopoCutCandidate merged_mode_cut;
        vector<Int3> merged_mode_raw_offsets;
        double component_shift_norm2 = 0.0;
        const auto merge_mode_started = clock::now();
        const bool merge_ok = solve_large_component_merge_mode(
            component_atoms, this->m_mol[frame_idx].Box, components, merge_plan,
            shifts, component_scoring_edges, steric_exclusion_set,
            component_points,
            trajectory_context.continuity_pairs,
            trajectory_context.continuity_anchor_indices, runtime_cutoffs,
            merged_mode_cut, component_shift_norm2, &timing, anchor_backend,
            same_component_warm_layout ? &warm_state.previous_component_offsets
                                       : nullptr,
            &warm_state, component_warm_reference, allow_exact_search_on_merged,
            &merged_mode_raw_offsets, &base_anchor_context,
            &explicit_required_contacts);
        timing.component_graph_s +=
            chrono::duration<double>(clock::now() - merge_mode_started).count();
        if (merge_ok &&
            accept_component_cut(merged_mode_cut, component_points,
                                 &merged_mode_raw_offsets,
                                 "component merge-graph solve")) {
          ++timing.component_graph_used;
        } else {
          ++timing.component_large_mode_fallback_frames;
          record_frame_failure(!merge_ok
                                   ? PbctopoFailureReason::ComponentGraphFailed
                                   : PbctopoFailureReason::NotBetterThanIncumbent);
          const char *merge_reject_reason =
              merge_ok ? candidate_skip_search_reject_reason(
                             merged_mode_cut, best_safe_candidate)
                       : nullptr;
          component_graph_fallthrough_reason = !merge_ok
                                                   ? "graph_invalid"
                                                   : string("not_better_not_close/") +
                                                          (merge_reject_reason != nullptr
                                                               ? merge_reject_reason
                                                               : "unknown");
          append_pbctopo_log_line(
              "[pbctopo] large-component merge mode fell back at frame " +
              to_string(static_cast<long long>(
                  this->m_mol[frame_idx].time_of_snap)) +
              ": mode=" + component_mode_label +
              ", raw_components=" + to_string(raw_component_count) +
              ", effective_components=" +
              to_string(effective_component_count) + ", reason=" +
              component_graph_fallthrough_reason);
        }
      } else if (component_mode == PbctopoComponentMode::Beam) {
        ++timing.component_large_mode_beam_frames;
        effective_component_count = components.size();
        const auto beam_started = clock::now();
        PbctopoComponentSearchDiagnostics search_diagnostics;
        vector<PbctopoComponentSearchSurvivor> search_survivors;
        PbctopoComponentCandidate beam_candidate =
            search_component_shift_candidate_beam(
                component_atoms, this->m_mol[frame_idx].Box, components, shifts,
                trajectory_context.continuity_pairs,
                trajectory_context.continuity_anchor_indices, runtime_cutoffs,
                component_points, beam_width,
                make_frame_component_search_callbacks(),
                same_component_warm_layout
                    ? &warm_state.previous_component_offsets
                    : nullptr,
                (best_safe_candidate.valid && !best_safe_candidate.ultra_clash)
                    ? &best_safe_candidate
                    : nullptr,
                (best_safe_candidate.valid && !best_safe_candidate.ultra_clash &&
                 best_safe_points.size() == base_atoms.size())
                    ? &best_safe_points
                    : nullptr,
                &timing, anchor_backend, &base_anchor_context, 0,
                &search_diagnostics, &search_survivors,
                kMaxOutputCertificationBand, &explicit_required_contacts);
        frame_report.component_search_attempted = search_diagnostics.attempted;
        frame_report.component_search_completed_under_admissible_bounds =
            search_diagnostics.search_completed_under_admissible_bounds;
        frame_report.component_search_incumbent_bound_pruning_used =
            search_diagnostics.incumbent_bound_pruning_used;
        frame_report.component_search_best_objective_proven =
            search_diagnostics.best_objective_proven;
        frame_report.component_search_equivalent_optima_complete =
            search_diagnostics.equivalent_optima_complete;
        frame_report.component_feasible_assignment_set_enumerated =
            search_diagnostics.feasible_assignment_set_enumerated;
        frame_report.component_feasible_assignments_within_domain =
            search_diagnostics.feasible_assignments_within_domain;
        frame_report.component_soft_score_valid_assignments_within_domain =
            search_diagnostics.soft_score_valid_assignments_within_domain;
        frame_report.component_bounded_domain_feasibility_known =
            search_diagnostics.bounded_domain_feasibility_known;
        frame_report.component_bounded_domain_feasible =
            search_diagnostics.bounded_domain_feasible;
        frame_report.component_search_shell_radius =
            search_diagnostics.component_search_shell_radius;
        frame_report.domain_expansion_attempted =
            search_diagnostics.domain_expansion_attempted;
        frame_report.domain_expansion_exhausted =
            search_diagnostics.domain_expansion_exhausted;
        frame_report.bounded_domain_only =
            search_diagnostics.bounded_domain_only;
        frame_report.component_search_exhaustive =
            search_diagnostics.exhaustive;
        frame_report.component_offset_enumeration_exhaustive =
            search_diagnostics.offset_enumeration_exhaustive;
        frame_report.component_all_atom_feasibility_exhaustive =
            search_diagnostics.all_atom_feasibility_exhaustive;
        frame_report.component_certificate_search_exhaustive =
            search_diagnostics.certificate_search_exhaustive;
        frame_report.output_certificate_search_exhaustive =
            frame_report.output_certificate_search_exhaustive ||
            search_diagnostics.certificate_search_exhaustive;
        titan_pbctopo::apply_pbctopo_search_identifiability(
            frame_identifiability,
            search_diagnostics.certificate_search_exhaustive,
            search_diagnostics.equivalent_optima_complete,
            search_diagnostics.equivalent_best_assignments_within_domain,
            false, 0);
        frame_report.beam_survivors_exactly_rescored =
            search_diagnostics.beam_survivors_exactly_rescored;
        frame_report.component_search_optimality_gap_known =
            search_diagnostics.optimality_gap_known;
        frame_report.component_search_optimality_gap =
            search_diagnostics.optimality_gap;
        append_pbctopo_log_line(
            "[pbctopo] component beam domain audit at frame " +
            to_string(frame_report.frame) + ": shell_radius=" +
            to_string(search_diagnostics.component_search_shell_radius) +
            ", selected_boundary_known=" +
            string(search_diagnostics.selected_assignment_boundary_known
                       ? "yes"
                       : "no") +
            ", selected_touches_boundary=" +
            string(search_diagnostics.selected_assignment_boundary_known
                       ? (search_diagnostics
                                  .selected_assignment_touches_shell_boundary
                              ? "yes"
                              : "no")
                       : "na") +
            ", bounded_domain_only=" +
            string(search_diagnostics.bounded_domain_only ? "yes" : "no"));
        timing.component_search_s +=
            chrono::duration<double>(clock::now() - beam_started).count();
        record_component_search_survivors(
            search_survivors, PbctopoFrameMode::ComponentSearchBeam,
            &beam_candidate, false);
        if (!accept_component_candidate(beam_candidate, component_points, nullptr,
                                        "component beam solve")) {
          ++timing.component_large_mode_fallback_frames;
          record_frame_failure(beam_candidate.valid && !beam_candidate.ultra_clash
                                   ? PbctopoFailureReason::NotBetterThanIncumbent
                                   : PbctopoFailureReason::BeamEmpty);
          PbctopoCutCandidate beam_as_cut =
              component_candidate_to_cut_candidate(beam_candidate);
          const char *beam_reject_reason =
              (beam_candidate.valid && !beam_candidate.ultra_clash)
                  ? candidate_skip_search_reject_reason(beam_as_cut,
                                                        best_safe_candidate)
                  : nullptr;
          component_graph_fallthrough_reason =
              beam_candidate.valid && !beam_candidate.ultra_clash
                  ? string("not_better_not_close/") +
                        (beam_reject_reason != nullptr
                             ? beam_reject_reason
                             : "unknown")
                  : "beam_no_complete_assignment";
          append_pbctopo_log_line(
              "[pbctopo] large-component beam mode fell back at frame " +
              to_string(static_cast<long long>(
                  this->m_mol[frame_idx].time_of_snap)) +
              ": mode=" + component_mode_label +
              ", raw_components=" + to_string(raw_component_count) +
              ", effective_components=" +
              to_string(effective_component_count) + ", reason=" +
              component_graph_fallthrough_reason + ", min_pair_d2=" +
              to_string(beam_candidate.min_pair_d2) + ", kappa2=" +
              to_string(beam_candidate.shape_kappa2) + ", asph=" +
              to_string(beam_candidate.shape_asphericity) + ", bbox=" +
              to_string(beam_candidate.bbox_volume));
        }
      }
    } else {
      timing.component_prep_s +=
          chrono::duration<double>(clock::now() - stage_started).count();
    }
    warm_state.previous_component_layout = component_layout;
    warm_state.has_component_offsets =
        warm_state.previous_component_offsets.size() == components.size() &&
        !components.empty();
    if (!warm_state.has_component_offsets)
      clear_component_trial_hints(warm_state);

    if (best_safe_candidate.valid && !best_safe_candidate.ultra_clash &&
        best_safe_points.size() == base_atoms.size()) {
      record_output_candidate(
          best_safe_candidate, best_safe_points,
          frame_report.mode == PbctopoFrameMode::None
              ? PbctopoFrameMode::Virtual
              : frame_report.mode);
    }
    bool output_candidate_committed = false;
    size_t output_certificate_rejects = 0;
    size_t output_all_atom_certificate_rejects = 0;
    timing.pbctopo_output_certification_band_candidates +=
        output_candidate_band.size();
    timing.pbctopo_output_certification_band_peak =
        max(timing.pbctopo_output_certification_band_peak,
            output_candidate_band.size());
    if (output_candidate_band.size() > kInitialOutputCertificationBand)
      ++timing.pbctopo_output_certification_band_expanded_frames;
    vector<Vec3> output_certification_points;
    output_certification_points.reserve(base_atoms.size());
    const bool certify_complete_uncertainty_band =
        titan_pbctopo::pbctopo_should_certify_complete_uncertainty_band(
            complete_feasible_band_retained,
            frame_report.component_feasible_assignment_set_enumerated);
    if (certify_complete_uncertainty_band) {
      optional<PbctopoCutCandidate> complete_band_winner;
      optional<PbctopoSelectedEvidenceAudit> complete_band_winner_audit;
      size_t complete_band_winner_index = 0;
      size_t complete_band_winner_all_atom_rejects = 0;
      PbctopoFrameMode complete_band_winner_mode = PbctopoFrameMode::None;
      vector<Int3> complete_band_winner_offsets;
      bool evidence_audit_failed = false;
      // Every band candidate is certified and audited, with no early exit,
      // so the geometry of all of them is computed first -- on the
      // TitanExecutor's workers when it has any -- and the loop below then
      // replays, in band order, exactly what the serial loop did with it.
      // Workers need the component context built and every offset vector
      // complete (the only paths on which certification writes frame state
      // before it finishes); otherwise the loop certifies serially as before.
      const size_t band_size = output_candidate_band.size();
      const size_t band_workers =
          executor != nullptr ? min(executor->capacity(), band_size) : 1;
      const bool band_parallel =
          band_workers > 1 && component_context_ready &&
          all_of(output_candidate_band.begin(), output_candidate_band.end(),
                 [&](const auto &snapshot) {
                   return snapshot.offsets.size() == components.size();
                 });
      vector<PbctopoCutCandidate> band_candidates;
      vector<vector<Vec3>> band_points;
      vector<CandidateCertificationWork> band_work;
      if (band_parallel) {
        band_candidates.resize(band_size);
        band_points.resize(band_size);
        band_work.resize(band_size);
        executor->parallel_for(
            band_size, band_workers, 1,
            [&](size_t, size_t begin, size_t end) {
              for (size_t index = begin; index < end; ++index) {
                const auto &snapshot = output_candidate_band[index];
                // The serial loop builds the points on the frame's thread
                // too, so their lattice queries belong in the carried delta.
                const auto worker_before =
                    titan_pbctopo::pbctopo_lattice_query_audit();
                band_candidates[index] = snapshot.candidate;
                double shift_norm2 = 0.0;
                build_component_points_from_offsets(
                    component_atoms, components, this->m_mol[frame_idx].Box,
                    snapshot.offsets, band_points[index], shift_norm2);
                compute_candidate_certification(
                    band_work[index], band_candidates[index],
                    band_points[index], &snapshot.offsets, true);
                band_work[index].worker_lattice_delta =
                    titan_pbctopo::pbctopo_lattice_query_audit_delta(
                        worker_before,
                        titan_pbctopo::pbctopo_lattice_query_audit());
              }
            });
      }
      for (size_t band_index = 0; band_index < output_candidate_band.size();
           ++band_index) {
        auto &snapshot = output_candidate_band[band_index];
        PbctopoCutCandidate candidate =
            band_parallel ? std::move(band_candidates[band_index])
                          : snapshot.candidate;
        if (band_parallel) {
          output_certification_points = band_points[band_index];
        } else {
          double output_shift_norm2 = 0.0;
          build_component_points_from_offsets(
              component_atoms, components, this->m_mol[frame_idx].Box,
              snapshot.offsets, output_certification_points,
              output_shift_norm2);
        }
        ++timing.pbctopo_output_certification_band_rebuilds;
        const bool certified =
            band_parallel
                ? apply_candidate_certification(band_work[band_index],
                                                candidate)
                : certify_candidate_for_output(candidate,
                                               output_certification_points,
                                               &snapshot.offsets);
        if (!certified) {
          ++output_certificate_rejects;
          ++output_all_atom_certificate_rejects;
          ++timing.pbctopo_output_certification_band_rejects;
          if (heavy_search_omits_atoms) {
            ++timing.pbctopo_heavy_search_all_atom_candidate_rejects;
            if (band_index == 0)
              ++timing.pbctopo_heavy_search_top_candidate_rejects;
          }
          ++timing.pbctopo_diagnostic_only_candidates;
          continue;
        }
        // The audit the serial loop computed here has the same inputs as the
        // one certification made for the evidence annotation, so a worker's
        // copy of that one is reused.
        const auto audit_started = clock::now();
        auto audit =
            band_parallel && band_work[band_index].evidence.has_value()
                ? *band_work[band_index].evidence
                : titan_pbctopo::audit_pbctopo_selected_hypotheses(
                      span<const PbctopoAtom>(base_atoms),
                      span<const Vec3>(output_certification_points),
                      base_anchor_context.contact_hypotheses,
                      candidate.assignment, true,
                      explicit_evidence_component_edges);
        timing.pbctopo_selected_evidence_audit_s +=
            chrono::duration<double>(clock::now() - audit_started).count();
        if (!audit || !audit->evaluated) {
          evidence_audit_failed = true;
          ++timing.pbctopo_output_certification_band_rejects;
          ++timing.pbctopo_diagnostic_only_candidates;
          continue;
        }
        ++timing.pbctopo_selected_evidence_audits;
        timing.pbctopo_selected_evidence_contacts_checked +=
            audit->contacts_checked;
        timing.pbctopo_selected_evidence_contacts_lost +=
            audit->contacts_lost;
        if (audit->state != PbctopoEvidenceState::Contradicted) {
          record_temporal_certified_candidate(candidate,
                                              output_certification_points,
                                              &snapshot.offsets);
        }
        if (!complete_band_winner.has_value()) {
          complete_band_winner = candidate;
          complete_band_winner_audit = *audit;
          cache_candidate_evidence_audit(candidate.assignment, *audit);
          complete_band_winner_index = band_index;
          complete_band_winner_all_atom_rejects =
              output_all_atom_certificate_rejects;
          complete_band_winner_mode = snapshot.mode;
          complete_band_winner_offsets = snapshot.offsets;
        }
      }
      const bool complete_band_exported =
          titan_pbctopo::pbctopo_complete_uncertainty_band_exported(
              !evidence_audit_failed, output_certificate_rejects,
              report_certified_candidates.size(),
              frame_report.component_feasible_assignments_within_domain);
      if (!complete_band_exported) {
        frame_identifiability.feasible_assignment_set_enumerated = false;
        frame_identifiability.feasible_assignments_within_domain = 0;
        // apply_pbctopo_search_identifiability derives evidence_uniqueness_known
        // from feasible_assignment_set_enumerated, so retracting the enumeration
        // here has to retract the claim that uniqueness is known along with it.
        // Leaving it set writes a frame that says the evidence uniqueness is
        // known while reporting no feasible-set audit, a pair that downstream
        // frame-binding validators reject as invalid_candidate_set. It only
        // shows on frames with no evidence-compatible candidate, where the
        // band cannot be exported.
        frame_identifiability.evidence_uniqueness_known = false;
        frame_identifiability.evidence_unique_within_search_domain = false;
      }
      if (complete_band_winner.has_value() &&
          complete_band_winner_audit.has_value()) {
        PbctopoCutCandidate candidate = *complete_band_winner;
        selected_evidence_audit = *complete_band_winner_audit;
        apply_selected_evidence_to_frame_identifiability();
        double output_shift_norm2 = 0.0;
        build_component_points_from_offsets(
            component_atoms, components, this->m_mol[frame_idx].Box,
            complete_band_winner_offsets, output_certification_points,
            output_shift_norm2);
        export_h9_partition_and_orbit(candidate,
                                      complete_band_winner_offsets);
        const bool evidence_contradicted =
            selected_evidence_audit.state ==
            PbctopoEvidenceState::Contradicted;
        const bool evidence_weak =
            selected_evidence_audit.state ==
                PbctopoEvidenceState::WeakDisconnected ||
            selected_evidence_audit.state ==
                PbctopoEvidenceState::WeakAmbiguous;
        const bool output_identified = selected_relative_images_identified();
        bool winner_commit_ok = true;
        if (evidence_contradicted) {
          warm_state = frame_start_warm_state;
        } else if (output_identified && !evidence_weak) {
          if (!commit_certified_candidate_to_warm_state(candidate)) {
            ++output_certificate_rejects;
            ++timing.pbctopo_output_certification_band_rejects;
            winner_commit_ok = false;
          }
        } else {
          warm_state = frame_start_warm_state;
          ++timing.pbctopo_weak_observation_warm_preserved;
        }
        if (winner_commit_ok) {
          best_safe_candidate = candidate;
          best_safe_points = output_certification_points;
          projection_clash_skip_active =
              candidate_has_projection_clash_margin(
                  best_safe_candidate, runtime_cutoffs.clash_cutoff);
          apply_cut_candidate_timed(best_safe_points);
          finalize_frame_report(
              evidence_contradicted
                  ? PbctopoFrameStatus::Contradicted
                  : output_identified && !evidence_weak
                        ? PbctopoFrameStatus::Certified
                        : PbctopoFrameStatus::WeakObservation,
              complete_band_winner_mode == PbctopoFrameMode::None
                  ? PbctopoFrameMode::Virtual
                  : complete_band_winner_mode,
              evidence_contradicted
                  ? PbctopoFailureReason::EvidenceContradicted
                  : output_identified && !evidence_weak
                        ? PbctopoFailureReason::None
                        : selected_evidence_audit.state ==
                                  PbctopoEvidenceState::WeakAmbiguous
                              ? PbctopoFailureReason::EvidenceUnderdetermined
                              : PbctopoFailureReason::EvidenceDisconnected,
              &best_safe_candidate);
          output_candidate_committed = true;
          // Every member ranked above the winner was rejected, so these are
          // the recoveries the first-certifiable loop below counts.
          if (complete_band_winner_index > 0)
            ++timing.pbctopo_output_certification_band_recoveries;
          if (complete_band_winner_index >=
              kInitialOutputCertificationBand) {
            ++timing.pbctopo_output_certification_band_expanded_recoveries;
          }
          if (heavy_search_omits_atoms &&
              complete_band_winner_all_atom_rejects > 0) {
            ++timing.pbctopo_heavy_search_all_atom_ranking_recoveries;
          }
        }
      } else if (evidence_audit_failed) {
        warm_state = frame_start_warm_state;
        reset_kinematic_seed_history(warm_state, timing);
        temporal_cut_candidates[frame_idx].clear();
        report_certified_candidates.clear();
            report_rejected_candidates.clear();
        record_frame_failure(
            PbctopoFailureReason::EvidenceHypothesisUnavailable);
      }
    }
    for (size_t band_index = 0;
         !certify_complete_uncertainty_band &&
         band_index < output_candidate_band.size(); ++band_index) {
      auto &snapshot = output_candidate_band[band_index];
      PbctopoCutCandidate candidate = snapshot.candidate;
      double output_shift_norm2 = 0.0;
      build_component_points_from_offsets(
          component_atoms, components, this->m_mol[frame_idx].Box,
          snapshot.offsets, output_certification_points, output_shift_norm2);
      ++timing.pbctopo_output_certification_band_rebuilds;
      if (!certify_candidate_for_output(candidate, output_certification_points,
                                        &snapshot.offsets)) {
        ++output_certificate_rejects;
        ++output_all_atom_certificate_rejects;
        ++timing.pbctopo_output_certification_band_rejects;
        if (heavy_search_omits_atoms) {
          ++timing.pbctopo_heavy_search_all_atom_candidate_rejects;
          if (band_index == 0)
            ++timing.pbctopo_heavy_search_top_candidate_rejects;
        }
        ++timing.pbctopo_diagnostic_only_candidates;
        continue;
      }
      audit_selected_evidence(output_certification_points,
                              candidate.assignment);
      if (!selected_evidence_audit.evaluated) {
        warm_state = frame_start_warm_state;
        reset_kinematic_seed_history(warm_state, timing);
        temporal_cut_candidates[frame_idx].clear();
        report_certified_candidates.clear();
            report_rejected_candidates.clear();
        ++output_certificate_rejects;
        ++timing.pbctopo_output_certification_band_rejects;
        ++timing.pbctopo_diagnostic_only_candidates;
        record_frame_failure(
            PbctopoFailureReason::EvidenceHypothesisUnavailable);
        continue;
      }
      export_h9_partition_and_orbit(candidate, snapshot.offsets);
      const bool evidence_contradicted =
          selected_evidence_audit.state == PbctopoEvidenceState::Contradicted;
      const bool evidence_weak =
          selected_evidence_audit.state == PbctopoEvidenceState::WeakDisconnected ||
          selected_evidence_audit.state == PbctopoEvidenceState::WeakAmbiguous;
      if (!evidence_contradicted) {
        record_temporal_certified_candidate(candidate,
                                            output_certification_points,
                                            &snapshot.offsets);
      }
      const bool output_identified = selected_relative_images_identified();
      if (evidence_contradicted) {
        warm_state = frame_start_warm_state;
      } else if (output_identified && !evidence_weak) {
        if (!commit_certified_candidate_to_warm_state(candidate)) {
          ++output_certificate_rejects;
          ++timing.pbctopo_output_certification_band_rejects;
          ++timing.pbctopo_diagnostic_only_candidates;
          continue;
        }
      } else {
        warm_state = frame_start_warm_state;
        ++timing.pbctopo_weak_observation_warm_preserved;
      }
      best_safe_candidate = candidate;
      best_safe_points = output_certification_points;
      projection_clash_skip_active = candidate_has_projection_clash_margin(
          best_safe_candidate, runtime_cutoffs.clash_cutoff);
      apply_cut_candidate_timed(best_safe_points);
      finalize_frame_report(
          evidence_contradicted
              ? PbctopoFrameStatus::Contradicted
              : output_identified && !evidence_weak
                    ? PbctopoFrameStatus::Certified
                                  : PbctopoFrameStatus::WeakObservation,
          snapshot.mode == PbctopoFrameMode::None
              ? PbctopoFrameMode::Virtual
              : snapshot.mode,
          evidence_contradicted
              ? PbctopoFailureReason::EvidenceContradicted
              : output_identified && !evidence_weak
                    ? PbctopoFailureReason::None
                    : selected_evidence_audit.state ==
                              PbctopoEvidenceState::WeakAmbiguous
                          ? PbctopoFailureReason::EvidenceUnderdetermined
                          : PbctopoFailureReason::EvidenceDisconnected,
          &best_safe_candidate);
      output_candidate_committed = true;
      if (output_certificate_rejects > 0)
        ++timing.pbctopo_output_certification_band_recoveries;
      if (band_index >= kInitialOutputCertificationBand)
        ++timing.pbctopo_output_certification_band_expanded_recoveries;
      if (heavy_search_omits_atoms &&
          output_all_atom_certificate_rejects > 0) {
        ++timing.pbctopo_heavy_search_all_atom_ranking_recoveries;
      }
      break;
    }
    if (output_candidate_committed)
      continue;
    if (heavy_search_omits_atoms && !output_candidate_band.empty() &&
        output_all_atom_certificate_rejects ==
            output_candidate_band.size()) {
      ++timing.pbctopo_heavy_search_all_atom_band_exhausted;
    }
    if (output_certificate_rejects > 0) {
      best_safe_certification_rejected = true;
      record_frame_failure(PbctopoFailureReason::CertificationReject);
      append_pbctopo_log_line(
          "[pbctopo] rejected output certification band at frame " +
          to_string(static_cast<long long>(
              this->m_mol[frame_idx].time_of_snap)) +
          ": assignments_tested=" +
          to_string(output_certificate_rejects) +
          ", band_size=" + to_string(output_candidate_band.size()));
    }

    if ((!best_safe_candidate.valid || best_safe_candidate.ultra_clash ||
         best_safe_certification_rejected) &&
        split_rescue_available) {
      ensure_component_context();
      ++timing.hull_rescue_attempts;
      PbctopoCutCandidate best_hull_candidate;
      vector<Vec3> best_hull_points;
      best_hull_points.reserve(base_atoms.size());
      Int3 best_hull_shift{};
      int best_hull_owner = -1;
      double best_hull_volume = numeric_limits<double>::infinity();
      size_t hull_valid_candidates = 0;
      vector<int> rescue_owner_candidates;
      rescue_owner_candidates.reserve(2);
      rescue_owner_candidates.push_back(split_rescue_detail.owner_b);
      if (split_rescue_detail.owner_a != split_rescue_detail.owner_b)
        rescue_owner_candidates.push_back(split_rescue_detail.owner_a);
      for (int rescue_owner : rescue_owner_candidates) {
        build_owner_spans(base_atoms, rescue_owner, hull_owner_spans);
        if (hull_owner_spans.empty())
          continue;
        for (const auto &shift : shifts) {
          if (shift.ix == 0 && shift.iy == 0 && shift.iz == 0)
            continue;
          hull_rescue_points.clear();
          pbctopo_hull::shift_points_by_spans(
              split_rescue_points, hull_owner_spans,
              Vec3{shift.tx, shift.ty, shift.tz}, hull_rescue_points,
              &hull_scratch);
          PbctopoCutCandidate hull_candidate;
          const PbctopoCutCandidate *hull_incumbent =
              best_hull_candidate.valid ? &best_hull_candidate : nullptr;
          if (!validate_rescue_candidate_points(hull_rescue_points,
                                                hull_incumbent,
                                                hull_candidate)) {
            continue;
          }
          double hull_volume = numeric_limits<double>::infinity();
          if (!pbctopo_hull::quickhull_volume(hull_rescue_points, hull_volume,
                                              &hull_scratch)) {
            continue;
          }
          ++hull_valid_candidates;
          const bool better_hull =
              !best_hull_candidate.valid ||
              hull_volume + 1.0e-6 < best_hull_volume ||
              (fabs(hull_volume - best_hull_volume) <= 1.0e-6 &&
               better_cut_candidate(hull_candidate, best_hull_candidate));
          if (!better_hull)
            continue;
          best_hull_candidate = hull_candidate;
          best_hull_volume = hull_volume;
          best_hull_shift = {shift.ix, shift.iy, shift.iz};
          best_hull_owner = rescue_owner;
          best_hull_points = hull_rescue_points;
        }
      }
      if (best_hull_candidate.valid) {
        append_pbctopo_log_line(
            "[pbctopo] hull rescue candidate at frame " +
            to_string(static_cast<long long>(
                this->m_mol[frame_idx].time_of_snap)) +
            " from " + split_rescue_source + ": moved_owner=" +
            to_string(best_hull_owner) + ", applied_shift=" +
            format_int3(best_hull_shift) + ", hull_volume=" +
            to_string(best_hull_volume) + ", valid_candidates=" +
            to_string(hull_valid_candidates) + ", avx2=" +
            string(pbctopo_hull::avx2_available() ? "on" : "off"));
        if (accept_rescue_candidate(best_hull_points, best_hull_candidate)) {
          ++timing.hull_rescue_successes;
          const bool rescue_identified = selected_relative_images_identified();
          const bool rescue_evidence_weak =
              selected_evidence_audit.state == PbctopoEvidenceState::WeakDisconnected ||
              selected_evidence_audit.state == PbctopoEvidenceState::WeakAmbiguous;
          finalize_frame_report(rescue_identified && !rescue_evidence_weak
                                    ? (selected_evidence_audit.state ==
                                               PbctopoEvidenceState::Contradicted
                                           ? PbctopoFrameStatus::Contradicted
                                           : PbctopoFrameStatus::Rescued)
                                    : (selected_evidence_audit.state ==
                                               PbctopoEvidenceState::Contradicted
                                           ? PbctopoFrameStatus::Contradicted
                                           : PbctopoFrameStatus::WeakObservation),
                                PbctopoFrameMode::HullRescue,
                                selected_evidence_audit.state ==
                                        PbctopoEvidenceState::Contradicted
                                    ? PbctopoFailureReason::EvidenceContradicted
                                    : rescue_identified
                                          ? PbctopoFailureReason::None
                                          : selected_evidence_audit.state ==
                                                    PbctopoEvidenceState::WeakAmbiguous
                                                ? PbctopoFailureReason::EvidenceUnderdetermined
                                                : PbctopoFailureReason::EvidenceDisconnected,
                                &best_safe_candidate);
          continue;
        }
      }

      ++timing.distance_rescue_attempts;
      vector<Vec3> rescued_points;
      rescued_points.reserve(base_atoms.size());
      Int3 rescue_shift{};
      double rescue_raw_delta_norm = numeric_limits<double>::infinity();
      if (rescue_split_contacting_owner_candidate(
              split_rescue_points, base_atoms, this->m_mol[frame_idx].Box,
              local_hard_required_owner_pairs, shifts, split_rescue_detail,
              rescued_points, rescue_shift, rescue_raw_delta_norm)) {
        PbctopoCutCandidate rescued_candidate;
        if (validate_rescue_candidate_points(rescued_points, nullptr,
                                             rescued_candidate)) {
          append_pbctopo_log_line(
              "[pbctopo] distance rescue candidate at frame " +
              to_string(static_cast<long long>(
                  this->m_mol[frame_idx].time_of_snap)) +
              " from " + split_rescue_source + ": applied_shift=" +
              format_int3(rescue_shift) + ", rescued_raw_delta_norm=" +
              to_string(rescue_raw_delta_norm));
          if (accept_rescue_candidate(rescued_points, rescued_candidate)) {
            ++timing.distance_rescue_successes;
            const bool rescue_identified =
                selected_relative_images_identified();
            const bool rescue_evidence_weak =
                selected_evidence_audit.state == PbctopoEvidenceState::WeakDisconnected ||
                selected_evidence_audit.state == PbctopoEvidenceState::WeakAmbiguous;
            finalize_frame_report(rescue_identified && !rescue_evidence_weak
                                      ? (selected_evidence_audit.state ==
                                                 PbctopoEvidenceState::Contradicted
                                             ? PbctopoFrameStatus::Contradicted
                                             : PbctopoFrameStatus::Rescued)
                                      : (selected_evidence_audit.state ==
                                                 PbctopoEvidenceState::Contradicted
                                             ? PbctopoFrameStatus::Contradicted
                                             : PbctopoFrameStatus::WeakObservation),
                                  PbctopoFrameMode::DistanceRescue,
                                  selected_evidence_audit.state ==
                                          PbctopoEvidenceState::Contradicted
                                      ? PbctopoFailureReason::EvidenceContradicted
                                      : rescue_identified
                                            ? PbctopoFailureReason::None
                                            : selected_evidence_audit.state ==
                                                      PbctopoEvidenceState::WeakAmbiguous
                                                  ? PbctopoFailureReason::EvidenceUnderdetermined
                                                  : PbctopoFailureReason::EvidenceDisconnected,
                                  &best_safe_candidate);
            continue;
          }
        }
      }
    }

    if (best_candidate.valid)
      ++timing.pbctopo_diagnostic_only_candidates;
    const bool lattice_query_failed =
        frame_fatal_lattice_query_failures > 0;
    const PbctopoFailureReason fallback_reason =
        pbctopo_resolve_fallback_reason(lattice_query_failed,
                                        frame_report.fallback_reason);
    const string fallback_message =
        "[pbctopo] frame " +
        to_string(static_cast<long long>(this->m_mol[frame_idx].time_of_snap)) +
        (lattice_query_failed
             ? " left unchanged after a failed lattice-image query: "
             : " fell back to pbci=0 after no certified image assignment: ") +
        "safe_candidates=" +
        to_string(safe_candidates) + ", best.min_pair_d2=" +
        to_string(best_candidate.min_pair_d2) + ", best.kappa2=" +
        to_string(best_candidate.shape_kappa2) + ", best.asph=" +
        to_string(best_candidate.shape_asphericity) + ", best.bbox=" +
        to_string(best_candidate.bbox_volume);
    append_pbctopo_log_line(fallback_message);
    TitanUI::print_warning(fallback_message);
    ++fallback_frames;
    projection_clash_skip_active = false;
    reset_uncommitted_warm_state();
    if (!lattice_query_failed)
      fallback_pbci_frame(frame_idx);
    finalize_frame_report(PbctopoFrameStatus::Fallback,
                          lattice_query_failed ? PbctopoFrameMode::None
                                               : PbctopoFrameMode::Pbci,
                          fallback_reason, nullptr);
  }

  if (temporal_mode == PbctopoTemporalMode::Viterbi) {
    vector<vector<PbctopoTemporalCandidate>> temporal_frames(frame_count);
    for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
      const auto &cuts = temporal_cut_candidates[frame_idx];
      const auto &greedy = greedy_selected_candidates[frame_idx];
      if (cuts.empty() || !greedy.valid || !greedy.assignment.valid ||
          !greedy.hard_feasibility.certified())
        continue;
      ++timing.pbctopo_temporal_candidate_frames;
      timing.pbctopo_temporal_candidates += cuts.size();
      temporal_frames[frame_idx].reserve(cuts.size());
      double v1[3]{}, v2[3]{}, v3[3]{};
      this->m_mol[frame_idx].Box.get_v1(v1);
      this->m_mol[frame_idx].Box.get_v2(v2);
      this->m_mol[frame_idx].Box.get_v3(v3);
      const auto temporal_gram =
          titan_pbctopo::pbctopo_temporal_normalized_gram(
              {v1[0], v1[1], v1[2], v2[0], v2[1], v2[2],
               v3[0], v3[1], v3[2]});
      const size_t layout_idx = temporal_frame_layout[frame_idx];
      for (size_t rank = 0; rank < cuts.size(); ++rank) {
        PbctopoTemporalCandidate candidate;
        candidate.center = {cuts[rank].cut_x, cuts[rank].cut_y,
                            cuts[rank].cut_z};
        candidate.normalized_gram = temporal_gram;
        candidate.metric_available = true;
        candidate.emission_cost =
            candidate_temporal_emission_cost(cuts.front(), cuts[rank]);
        candidate.component_offsets =
            cuts[rank].assignment.component_offsets;
        if (layout_idx < temporal_component_layouts.size()) {
          candidate.component_ids =
              temporal_component_layouts[layout_idx].component_ids;
          candidate.component_weights =
              temporal_component_layouts[layout_idx].component_weights;
        }
        candidate.assignment_signature = cuts[rank].assignment.signature;
        candidate.certified = cuts[rank].hard_feasibility.certified() &&
                              cuts[rank].assignment.valid;
        temporal_frames[frame_idx].push_back(std::move(candidate));
      }
    }

    const auto temporal_result = titan_pbctopo::solve_pbctopo_temporal_viterbi(
        temporal_frames, Arguments.pbc.pbctopo_temporal_penalty);
    ofstream temporal_report = open_pbctopo_temporal_path_report();
    if (!temporal_report.is_open()) {
      append_pbctopo_log_line(
          "[pbctopo] observation export unavailable: "
          "artifact=pbctopo_temporal_path.csv");
    }
    const size_t invalid = numeric_limits<size_t>::max();
    if (temporal_result.valid) {
      for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
        const size_t selected = temporal_result.selected[frame_idx];
        if (selected == invalid ||
            selected >= temporal_cut_candidates[frame_idx].size())
          continue;
        const auto &chosen = temporal_cut_candidates[frame_idx][selected];
        const auto &greedy = greedy_selected_candidates[frame_idx];
        const bool changed = !same_pbctopo_image_assignment(
            chosen.assignment, greedy.assignment);
        if (changed)
          ++timing.pbctopo_temporal_proposed_changes;
        ++timing.pbctopo_temporal_viterbi_frames;
        ++timing.pbctopo_temporal_replay_frames;

        bool replay_applied = !changed;
        vector<PbctopoAtom> replay_atoms;
        optional<titan_pbctopo::PbctopoSelectedEvidenceAudit>
            temporal_selected_audit;
        if (changed) {
          const size_t layout_idx = temporal_frame_layout[frame_idx];
          replay_atoms.reserve(trajectory_context.total_atoms_hint);
          append_atoms_for_system(this->m_mol[frame_idx], &this->info,
                                  &trajectory_context.rec_static, 0,
                                  replay_atoms);
          if (frame_idx < mol2.m_mol.size() &&
              !mol2.m_mol[frame_idx].pos.x.empty()) {
            append_atoms_for_system(mol2.m_mol[frame_idx], &mol2.info,
                                    &trajectory_context.lig_static, 1,
                                    replay_atoms);
          }
          vector<Vec3> applied_points;
          applied_points.reserve(replay_atoms.size());
          for (const auto &atom : replay_atoms)
            applied_points.push_back({atom.x, atom.y, atom.z});
          vector<Vec3> replayed_points;
          if (layout_idx < temporal_component_layouts.size() &&
              replay_pbctopo_assignment_delta(
                  applied_points,
                  temporal_component_layouts[layout_idx].atom_component_index,
                  greedy.assignment, chosen.assignment,
                  this->m_mol[frame_idx].Box, replayed_points)) {
            const auto temporal_explicit_evidence_component_edges =
                build_explicit_evidence_component_edges(
                    temporal_component_layouts[layout_idx].atom_component_index,
                    temporal_component_layouts[layout_idx].component_count);
            const auto temporal_audit_started = clock::now();
            auto audit = titan_pbctopo::audit_pbctopo_selected_hypotheses(
                span<const PbctopoAtom>(replay_atoms),
                span<const Vec3>(replayed_points),
                temporal_contact_hypotheses[frame_idx], chosen.assignment,
                true, temporal_explicit_evidence_component_edges);
            timing.pbctopo_selected_evidence_audit_s +=
                chrono::duration<double>(clock::now() - temporal_audit_started)
                    .count();
            if (audit && audit->evaluated &&
                audit->state != PbctopoEvidenceState::Contradicted) {
              temporal_selected_audit = *audit;
              apply_pbctopo_points_direct(replay_atoms, replayed_points, *this,
                                          mol2, frame_idx);
              replay_applied = true;
              ++timing.pbctopo_temporal_replay_applied;
              ++timing.pbctopo_temporal_path_changes;
            } else {
              ++timing.pbctopo_temporal_replay_skipped;
              if (!audit) {
                append_pbctopo_log_line(
                    "[pbctopo] temporal evidence replay failed closed at frame " +
                    to_string(frame_idx) + ": " + audit.error());
              }
            }
          } else {
            ++timing.pbctopo_temporal_replay_skipped;
          }
        }
        if (deferred_frame_report_valid[frame_idx] != 0) {
          auto &report = deferred_frame_reports[frame_idx];
          const auto &actual_output_candidate =
              changed && replay_applied ? chosen : greedy;
          const bool temporal_output_selected = !changed || replay_applied;
          report.temporal_selected = temporal_output_selected;
          report.temporal_changed_from_greedy = changed && replay_applied;
          report.temporal_assignment_signature =
              temporal_output_selected
                  ? actual_output_candidate.assignment.signature
                  : 0;
          if (changed && replay_applied) {
            const auto &temporal_audit = *temporal_selected_audit;
            ++timing.pbctopo_selected_evidence_audits;
            timing.pbctopo_selected_evidence_contacts_checked +=
                temporal_audit.contacts_checked;
            timing.pbctopo_selected_evidence_contacts_lost +=
                temporal_audit.contacts_lost;
            report.evidence_state = temporal_audit.state;
            report.evidence_consistency_evaluated = temporal_audit.evaluated;
            report.evidence_consistent = temporal_audit.consistent;
            report.evidence_graph_connected =
                temporal_audit.evidence_graph_connected;
            titan_pbctopo::apply_pbctopo_final_selected_evidence(
                report.identifiability, temporal_audit.evaluated,
                temporal_audit.evidence_graph_connected,
                temporal_audit.evidence_component_count,
                temporal_audit.evidence_edge_count,
                temporal_audit.explicit_contact_evidence_edges,
                temporal_audit.scored_contact_evidence_edges,
                temporal_audit.evidence_connected_component_count);
            report.soft_observed_contact_pair_hash =
                temporal_audit.observed_pair_hash;
            report.soft_observed_lost_pair_hash =
                temporal_audit.lost_pair_hash;
            report.soft_observed_all_hypotheses_hash =
                temporal_audit.all_hypotheses_hash;
            report.soft_observed_selected_hypothesis_hash =
                temporal_audit.selected_hypothesis_hash;
            report.soft_observed_selected_compatible_pair_hash =
                temporal_audit.selected_compatible_pair_hash;
            report.soft_observed_hypothesis_count =
                temporal_audit.hypothesis_count;
            report.soft_observed_component_relation_count =
                temporal_audit.component_relation_count;
            report.soft_observed_ambiguous_relation_count =
                temporal_audit.ambiguous_component_relation_count;
            report.soft_observed_compatible_hypothesis_count =
                temporal_audit.compatible_hypothesis_count;
            report.soft_observed_selected_compatible_contact_count =
                temporal_audit.selected_compatible_contact_count;
            report.soft_observed_alternative_hypothesis_contact_count =
                temporal_audit.alternative_hypothesis_contact_count;
            report.soft_observed_selected_compatible_contact_loss_count =
                temporal_audit.selected_compatible_contact_loss_count;
            report.soft_observed_no_support_relation_count =
                temporal_audit.no_supported_contact_relation_count;
            report.soft_observed_selected_relation_has_support =
                temporal_audit.selected_relation_has_support;
            report.soft_observed_contacts_checked =
                temporal_audit.contacts_checked;
            report.soft_observed_contacts_lost = temporal_audit.contacts_lost;
            const bool was_contradicted =
                report.status == PbctopoFrameStatus::Contradicted;
            const bool was_weak =
                report.status == PbctopoFrameStatus::WeakObservation;
            const bool rescue_provenance =
                report.status == PbctopoFrameStatus::Rescued ||
                report.mode == PbctopoFrameMode::HullRescue ||
                report.mode == PbctopoFrameMode::DistanceRescue;
            if (temporal_audit.state ==
                PbctopoEvidenceState::Contradicted) {
              report.status = PbctopoFrameStatus::Contradicted;
              report.fallback_reason =
                  PbctopoFailureReason::EvidenceContradicted;
              report.committed_candidate_count = 0;
            } else if (temporal_audit.state ==
                       PbctopoEvidenceState::WeakDisconnected ||
                       temporal_audit.state ==
                           PbctopoEvidenceState::WeakAmbiguous) {
              report.status = PbctopoFrameStatus::WeakObservation;
              report.fallback_reason =
                  temporal_audit.state == PbctopoEvidenceState::WeakAmbiguous
                      ? PbctopoFailureReason::EvidenceUnderdetermined
                      : PbctopoFailureReason::EvidenceDisconnected;
              report.committed_candidate_count = 1;
            } else {
              report.status = rescue_provenance
                                  ? PbctopoFrameStatus::Rescued
                                  : PbctopoFrameStatus::Certified;
              report.fallback_reason = PbctopoFailureReason::None;
              report.committed_candidate_count = 1;
            }
            const bool is_contradicted =
                report.status == PbctopoFrameStatus::Contradicted;
            const bool is_weak =
                report.status == PbctopoFrameStatus::WeakObservation;
            if (!was_contradicted && is_contradicted) {
              ++timing.pbctopo_contradicted_output_frames;
            } else if (was_contradicted && !is_contradicted &&
                       timing.pbctopo_contradicted_output_frames > 0) {
              --timing.pbctopo_contradicted_output_frames;
            }
            if (!was_weak && is_weak) {
              ++timing.pbctopo_weak_observation_output_frames;
            } else if (was_weak && !is_weak &&
                       timing.pbctopo_weak_observation_output_frames > 0) {
              --timing.pbctopo_weak_observation_output_frames;
            }
            const bool old_anchor_ambiguous =
                report.anchor_graph.ambiguity_count > 0;
            populate_pbctopo_selected_candidate_report(
                report, chosen, replay_atoms, runtime_cutoffs.clash_cutoff,
                runtime_cutoffs.continuity_cutoff,
                temporal_cut_candidates[frame_idx]);
            const bool new_anchor_ambiguous =
                report.anchor_graph.ambiguity_count > 0;
            if (old_anchor_ambiguous != new_anchor_ambiguous) {
              if (new_anchor_ambiguous) {
                ++timing.pbctopo_anchor_ambiguous_frames;
              } else if (timing.pbctopo_anchor_ambiguous_frames > 0) {
                --timing.pbctopo_anchor_ambiguous_frames;
              }
            }
          }
        }
        if (temporal_report.is_open()) {
          temporal_report
                          << titan_pbctopo::VIBE_OBSERVATION_SCHEMA_VERSION
                          << ','
                          << titan_pbctopo::vibe_observation_producer_version()
                          << ','
                          << titan_pbctopo::VIBE_OBSERVATION_COORDINATE_UNIT
                          << ','
                          << titan_pbctopo::VIBE_OBSERVATION_TIME_UNIT << ','
                          << titan_pbctopo::VIBE_OBSERVATION_ENDIANNESS
                          << ','
                          << titan_pbctopo::VIBE_OBSERVATION_FLOATING_FORMAT
                          << ','
                          << titan_pbctopo::
                                 VIBE_OBSERVATION_IDENTITY_HASH_ALGORITHM
                          << ','
                          << titan_pbctopo::
                                 VIBE_OBSERVATION_BINDING_HASH_ALGORITHM
                          << ','
                          << titan_pbctopo::
                                 VIBE_OBSERVATION_PAYLOAD_CHECKSUM_ALGORITHM
                          << ','
                          << titan_pbctopo::VIBE_OBSERVATION_BINDING_MODE << ','
                          << titan_pbctopo::VIBE_OBSERVATION_NEGATIVE_ZERO_POLICY
                          << ','
                          << titan_pbctopo::VIBE_OBSERVATION_NONFINITE_POLICY
                          << ",experimental_temporal_path,"
                             "experimental_viterbi_diagnostic,"
                          << frame_report_offset + frame_idx << ','
                          << titan_pbctopo::format_pbctopo_report_double(
                                 this->m_mol[frame_idx].time_of_snap)
                          << ',' << selected + 1 << ','
                          << titan_pbctopo::format_pbctopo_report_double(
                                 chosen.cut_x)
                          << ','
                          << titan_pbctopo::format_pbctopo_report_double(
                                 chosen.cut_y)
                          << ','
                          << titan_pbctopo::format_pbctopo_report_double(
                                 chosen.cut_z)
                          << ','
                          << titan_pbctopo::format_pbctopo_report_double(
                                 temporal_frames[frame_idx][selected]
                                     .emission_cost)
                          << ',' << chosen.assignment.signature << ','
                          << (changed ? 1 : 0) << ','
                          << (replay_applied ? 1 : 0) << '\n';
        }
      }
      append_pbctopo_log_line(
          "[pbctopo temporal] mode=viterbi, frames=" +
          to_string(timing.pbctopo_temporal_viterbi_frames) +
          ", candidates=" + to_string(timing.pbctopo_temporal_candidates) +
          ", proposed_changes=" +
          to_string(timing.pbctopo_temporal_proposed_changes) +
          ", applied_changes=" +
          to_string(timing.pbctopo_temporal_path_changes) +
          ", replay=immutable_assignment_delta" +
          ", total_cost=" + to_string(temporal_result.total_cost));
    }
    if (pbctopo_frame_report.is_open()) {
      for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
        if (deferred_frame_report_valid[frame_idx] != 0) {
          auto &report = deferred_frame_reports[frame_idx];
          if (frame_idx < pbctopo_frame_status_.size()) {
            pbctopo_frame_status_[frame_idx] =
                titan_pbctopo::pbctopo_frame_status_name(report.status);
            pbctopo_frame_certified_[frame_idx] =
                report.status == PbctopoFrameStatus::Certified ? 1 : 0;
          }
          if (pbctopo_assignment_set_report.is_open() &&
              report.status != PbctopoFrameStatus::Contradicted &&
              !temporal_cut_candidates[frame_idx].empty()) {
            // selected_framewise marks the greedy choice. After a replay the
            // report's assignment_signature names the temporal output, and
            // consumers derive changed-from-greedy by comparing the two.
            const PbctopoCutCandidate *selected_for_report =
                &greedy_selected_candidates[frame_idx];
            double v1[3]{}, v2[3]{}, v3[3]{};
            this->m_mol[frame_idx].Box.get_v1(v1);
            this->m_mol[frame_idx].Box.get_v2(v2);
            this->m_mol[frame_idx].Box.get_v3(v3);
            const auto normalized_gram =
                titan_pbctopo::pbctopo_temporal_normalized_gram(
                    {v1[0], v1[1], v1[2], v2[0], v2[1], v2[2],
                     v3[0], v3[1], v3[2]});
            write_pbctopo_assignment_set_rows(
                pbctopo_assignment_set_report, report.frame, report.time_ps,
                temporal_cut_candidates[frame_idx], selected_for_report,
                report.identifiability, normalized_gram, better_cut_candidate,
                report.topology_epoch_id,
                report.component_soft_score_valid_assignments_within_domain,
                report.spatial_lift_ambiguous_hard_edges,
                report.spatial_lift_cycle_residuals, report.status);
          }
          write_pbctopo_frame_report_row(pbctopo_frame_report,
                                         report);
        }
      }
    }
  }

  append_pbctopo_log_line(
      "[pbctopo evidence relation export] requested=" +
      string(Arguments.pbc.pbctopo_evidence_export ? "1" : "0") +
      ", enabled=" +
      string(pbctopo_candidate_evidence_summary_report.is_open() &&
                     pbctopo_candidate_evidence_relation_report.is_open()
                 ? "1"
                 : "0") +
      ", candidate_groups=" +
      to_string(evidence_relation_export_groups) + ", relation_rows=" +
      to_string(evidence_relation_export_rows) +
      ", duplicate_candidates_skipped=" +
      to_string(evidence_relation_export_duplicates) + ", failures=" +
      to_string(evidence_relation_export_failures));

  timing.pbctopo_frame_point_buffer_peak_bytes_estimate =
      max(timing.pbctopo_frame_point_buffer_peak_bytes_estimate,
          scratch.point_capacity_bytes_estimate());
  if (fallback_frames > 0) {
    const string fallback_summary = "[pbctopo] Fell back to pbci=0 on " +
                                    to_string(fallback_frames) + " frame(s).";
    append_pbctopo_log_line(fallback_summary);
    TitanUI::print_warning(fallback_summary);
  }
  if (timing.pbctopo_lattice_invalid_frames > 0) {
    const string invalid_lattice_summary =
        "[pbctopo] Left " +
        to_string(timing.pbctopo_lattice_invalid_frames) +
        " frame(s) unchanged because their periodic lattice was invalid.";
    append_pbctopo_log_line(invalid_lattice_summary);
    TitanUI::print_warning(invalid_lattice_summary);
  }
  append_pbctopo_log_line(
      "[pbctopo timing] frames=" + to_string(timing.frames) +
      ", append=" + to_string(timing.append_s) + " s, heavy_subset=" +
      to_string(timing.heavy_subset_s) + " s, translation_cache=" +
      to_string(timing.translation_cache_s) + " s, graph=" +
      to_string(timing.graph_s) + " s, pbctopo_local_graph_geometry_s=" +
      to_string(timing.pbctopo_local_graph_geometry_s) +
      " s, pbctopo_topology_edge_append_s=" +
      to_string(timing.pbctopo_topology_edge_append_s) +
      " s, projection_edges=" +
      to_string(timing.projection_edge_s) + " s, component_split=" +
      to_string(timing.component_split_s) + " s, seeds=" +
      to_string(timing.seed_s) + " s, event_planes=" +
      to_string(timing.event_planes_s) +
      " s, pbctopo_density_cut_plane_build=" +
      to_string(timing.pbctopo_density_cut_plane_build_s) +
      " s, pbctopo_virtual_seam_sweep=" +
      to_string(timing.pbctopo_virtual_seam_sweep_s) + " s, coarse=" +
      to_string(timing.coarse_s) + " s, refine=" +
      to_string(timing.refine_s) + " s, early_projection=" +
      to_string(timing.early_projection_s) + " s, warm_component=" +
      to_string(timing.warm_component_s) + " s, component_prep=" +
      to_string(timing.component_prep_s) +
      " s, component_context=" + to_string(timing.component_context_s) +
      " s, component_graph=" + to_string(timing.component_graph_s) +
      " s, component_graph_precompute=" +
      to_string(timing.component_graph_precompute_s) +
      " s, component_graph_lazy_materialize=" +
      to_string(timing.component_graph_lazy_materialize_s) +
      " s, component_graph_spatial_shortlist=" +
      to_string(timing.component_graph_spatial_shortlist_s) +
      " s, component_graph_pair_score=" +
      to_string(timing.component_graph_pair_score_s) +
      " s, component_graph_tree=" +
      to_string(timing.component_graph_tree_s) +
      " s, component_graph_refine=" +
      to_string(timing.component_graph_refine_s) +
      " s, component_graph_refine_rebase=" +
      to_string(timing.component_graph_refine_rebase_s) +
      " s, component_graph_refine_coarse=" +
      to_string(timing.component_graph_refine_coarse_s) +
      " s, component_graph_refine_coarse_stage1=" +
      to_string(timing.component_graph_refine_coarse_stage1_s) +
      " s, component_graph_refine_coarse_stage1_build_points=" +
      to_string(timing.component_graph_refine_coarse_stage1_build_points_s) +
      " s, component_graph_refine_coarse_stage1_continuity=" +
      to_string(timing.component_graph_refine_coarse_stage1_continuity_s) +
      " s, component_graph_refine_coarse_stage1_anchor=" +
      to_string(timing.component_graph_refine_coarse_stage1_anchor_s) +
      " s, component_graph_refine_coarse_stage1_cut_score=" +
      to_string(timing.component_graph_refine_coarse_stage1_cut_score_s) +
      " s, component_graph_refine_coarse_stage1_cut_score_principal=" +
      to_string(timing.component_graph_refine_coarse_stage1_cut_score_principal_s) +
      " s, component_graph_refine_coarse_stage1_cut_score_edge_scan=" +
      to_string(timing.component_graph_refine_coarse_stage1_cut_score_edge_scan_s) +
      " s, component_graph_refine_coarse_stage2=" +
      to_string(timing.component_graph_refine_coarse_stage2_s) +
      " s, component_graph_refine_coarse_stage2_build_points=" +
      to_string(timing.component_graph_refine_coarse_stage2_build_points_s) +
      " s, component_graph_refine_coarse_stage2_continuity=" +
      to_string(timing.component_graph_refine_coarse_stage2_continuity_s) +
      " s, component_graph_refine_coarse_stage2_anchor=" +
      to_string(timing.component_graph_refine_coarse_stage2_anchor_s) +
      " s, component_graph_refine_coarse_stage2_cut_score=" +
      to_string(timing.component_graph_refine_coarse_stage2_cut_score_s) +
      " s, component_graph_refine_build_points=" +
      to_string(timing.component_graph_refine_build_points_s) +
      " s, component_graph_refine_continuity=" +
      to_string(timing.component_graph_refine_continuity_s) +
      " s, component_graph_refine_anchor=" +
      to_string(timing.component_graph_refine_anchor_s) +
      " s, component_graph_refine_anchor_spatial_build=" +
      to_string(timing.component_graph_refine_anchor_spatial_build_s) +
      " s, component_graph_refine_anchor_spatial_query=" +
      to_string(timing.component_graph_refine_anchor_spatial_query_s) +
      " s, component_graph_refine_anchor_dense_fallback=" +
      to_string(timing.component_graph_refine_anchor_dense_fallback_s) +
      " s, pbctopo_anchor_context_build=" +
      to_string(timing.pbctopo_anchor_context_build_s) +
      " s, component_graph_refine_anchor_order=" +
      to_string(timing.component_graph_refine_anchor_order_s) +
      " s, component_graph_refine_anchor_contact=" +
      to_string(timing.component_graph_refine_anchor_contact_s) +
      " s, component_graph_refine_cut_score=" +
      to_string(timing.component_graph_refine_cut_score_s) +
      " s, component_graph_refine_ultraclash=" +
      to_string(timing.component_graph_refine_ultraclash_s) +
      " s, component_graph_refine_stats=" +
      to_string(timing.component_graph_refine_stats_s) +
      " s, component_cycle_sync=" +
      to_string(timing.component_cycle_sync_s) +
      " s, component_search=" + to_string(timing.component_search_s) +
      " s, offset_extract=" + to_string(timing.component_offset_extract_s) +
      " s, apply=" + to_string(timing.apply_s) + " s");
  append_pbctopo_log_line(
      "[pbctopo timing] coarse_candidates=" +
      to_string(timing.coarse_candidates) + ", refine_candidates=" +
      to_string(timing.refine_candidates) + ", projection_clash_skip_frames=" +
      to_string(timing.projection_clash_skip_frames) +
      ", projection_clash_skip_evals=" +
      to_string(timing.projection_clash_skip_evals) + ", component_frames=" +
      to_string(timing.component_frames) + ", component_shift_trials=" +
      to_string(timing.component_shift_trials) +
      ", hull_rescue_attempts=" + to_string(timing.hull_rescue_attempts) +
      ", hull_rescue_successes=" + to_string(timing.hull_rescue_successes) +
      ", distance_rescue_attempts=" +
      to_string(timing.distance_rescue_attempts) +
      ", distance_rescue_successes=" +
      to_string(timing.distance_rescue_successes) +
      ", pbci_fallback_frames=" + to_string(timing.pbci_fallback_frames) +
      ", pbctopo_llps_local_frames=" +
      to_string(timing.pbctopo_llps_local_frames) +
      ", pbctopo_llps_local_units=" +
      to_string(timing.pbctopo_llps_local_units) +
      ", pbctopo_llps_local_unit_source=" + llps_local_unit_source +
      ", pbctopo_llps_local_edges_used=" +
      to_string(timing.pbctopo_llps_local_edges_used) +
      ", pbctopo_llps_local_edges_skipped_cross_unit=" +
      to_string(timing.pbctopo_llps_local_edges_skipped_cross_unit) +
      ", pbctopo_llps_local_disconnected_units=" +
      to_string(timing.pbctopo_llps_local_disconnected_units) +
      ", early_projection_used=" + to_string(timing.early_projection_used) +
      ", previous_seed_used=" + to_string(timing.previous_seed_used) +
      ", pbctopo_kinematic_seed_candidates=" +
      to_string(timing.pbctopo_kinematic_seed_candidates) +
      ", pbctopo_kinematic_seed_evaluated=" +
      to_string(timing.pbctopo_kinematic_seed_evaluated) +
      ", pbctopo_kinematic_seed_deduped=" +
      to_string(timing.pbctopo_kinematic_seed_deduped) +
      ", pbctopo_kinematic_seed_best=" +
      to_string(timing.pbctopo_kinematic_seed_best) +
      ", pbctopo_kinematic_seed_history_resets=" +
      to_string(timing.pbctopo_kinematic_seed_history_resets) +
      ", pbctopo_density_cut_planes_frames=" +
      to_string(timing.pbctopo_density_cut_planes_frames) +
      ", pbctopo_density_cut_planes_x=" +
      to_string(timing.pbctopo_density_cut_planes_x) +
      ", pbctopo_density_cut_planes_y=" +
      to_string(timing.pbctopo_density_cut_planes_y) +
      ", pbctopo_density_cut_planes_z=" +
      to_string(timing.pbctopo_density_cut_planes_z) +
      ", pbctopo_density_cut_planes_dense_equiv_x=" +
      to_string(timing.pbctopo_density_cut_planes_dense_equiv_x) +
      ", pbctopo_density_cut_planes_dense_equiv_y=" +
      to_string(timing.pbctopo_density_cut_planes_dense_equiv_y) +
      ", pbctopo_density_cut_planes_dense_equiv_z=" +
      to_string(timing.pbctopo_density_cut_planes_dense_equiv_z) +
      ", pbctopo_density_cut_plane_collisions=" +
      to_string(timing.pbctopo_density_cut_plane_collisions) +
      ", pbctopo_density_cut_plane_dense_fallbacks=" +
      to_string(timing.pbctopo_density_cut_plane_dense_fallbacks) +
      ", pbctopo_virtual_discrete_frames=" +
      to_string(timing.pbctopo_virtual_discrete_frames) +
      ", pbctopo_virtual_axis_states_x=" +
      to_string(timing.pbctopo_virtual_axis_states_x) +
      ", pbctopo_virtual_axis_states_y=" +
      to_string(timing.pbctopo_virtual_axis_states_y) +
      ", pbctopo_virtual_axis_states_z=" +
      to_string(timing.pbctopo_virtual_axis_states_z) +
      ", pbctopo_virtual_state_space=" +
      to_string(timing.pbctopo_virtual_state_space) +
      ", pbctopo_virtual_states_evaluated=" +
      to_string(timing.pbctopo_virtual_states_evaluated) +
      ", pbctopo_virtual_search_exhaustive_frames=" +
      to_string(timing.pbctopo_virtual_search_exhaustive_frames) +
      ", pbctopo_virtual_seam_enumeration_exhaustive_frames=" +
      to_string(
          timing.pbctopo_virtual_seam_enumeration_exhaustive_frames) +
      ", pbctopo_virtual_projection_scoring_exhaustive_frames=" +
      to_string(
          timing.pbctopo_virtual_projection_scoring_exhaustive_frames) +
      ", pbctopo_virtual_full_atom_verification_exhaustive_frames=" +
      to_string(
          timing.pbctopo_virtual_full_atom_verification_exhaustive_frames) +
      ", pbctopo_virtual_search_shortlist_frames=" +
      to_string(timing.pbctopo_virtual_search_shortlist_frames) +
      ", pbctopo_virtual_branch_and_bound_frames=" +
      to_string(timing.pbctopo_virtual_branch_and_bound_frames) +
      ", pbctopo_virtual_sweep_workspace_reuse_frames=" +
      to_string(timing.pbctopo_virtual_sweep_workspace_reuse_frames) +
      ", pbctopo_virtual_sweep_events_x=" +
      to_string(timing.pbctopo_virtual_sweep_events_x) +
      ", pbctopo_virtual_sweep_events_y=" +
      to_string(timing.pbctopo_virtual_sweep_events_y) +
      ", pbctopo_virtual_sweep_events_z=" +
      to_string(timing.pbctopo_virtual_sweep_events_z) +
      ", pbctopo_virtual_sweep_workspace_peak_bytes=" +
      to_string(timing.pbctopo_virtual_sweep_workspace_peak_bytes) +
      ", pbctopo_verify_geometry_buffer_reuses=" +
      to_string(timing.pbctopo_verify_geometry_buffer_reuses) +
      ", pbctopo_frame_point_buffer_peak_bytes_estimate=" +
      to_string(timing.pbctopo_frame_point_buffer_peak_bytes_estimate) +
      ", pbctopo_virtual_omitted_seam_lower_bound_min=" +
      (isfinite(timing.pbctopo_virtual_omitted_seam_lower_bound_min)
           ? to_string(timing.pbctopo_virtual_omitted_seam_lower_bound_min)
           : string("na")) +
      ", pbctopo_virtual_lower_bound_gap_min=" +
      (isfinite(timing.pbctopo_virtual_lower_bound_gap_min)
           ? to_string(timing.pbctopo_virtual_lower_bound_gap_min)
           : string("na")) +
       ", component_graph_edges=" + to_string(timing.component_graph_edges) +
       ", component_graph_input_shift_domain_max=" +
       to_string(timing.component_graph_input_shift_domain_max) +
       ", component_graph_relative_shift_domain_max=" +
       to_string(timing.component_graph_relative_shift_domain_max) +
       ", component_graph_shift_descriptors=" +
       to_string(timing.component_graph_shift_descriptors) +
       ", component_graph_shift_materialized=" +
       to_string(timing.component_graph_shift_materialized) +
       ", component_graph_bbox_only_scores=" +
       to_string(timing.component_graph_bbox_only_scores) +
       ", component_graph_spatial_shortlist_builds=" +
       to_string(timing.component_graph_spatial_shortlist_builds) +
       ", component_graph_spatial_shortlist_fallbacks=" +
       to_string(timing.component_graph_spatial_shortlist_fallbacks) +
       ", component_graph_spatial_shortlist_pairs=" +
       to_string(timing.component_graph_spatial_shortlist_pairs) +
       ", component_graph_spatial_shortlist_shift_scores_avoided=" +
       to_string(
           timing.component_graph_spatial_shortlist_shift_scores_avoided) +
        ", component_graph_used=" + to_string(timing.component_graph_used) +
        ", component_graph_primary_attempts=" +
        to_string(timing.component_graph_primary_attempts) +
        ", component_graph_primary_certified=" +
        to_string(timing.component_graph_primary_certified) +
        ", component_graph_primary_virtual_skips=" +
        to_string(timing.component_graph_primary_virtual_skips) +
        ", component_graph_primary_fallbacks=" +
        to_string(timing.component_graph_primary_fallbacks) +
        ", component_graph_primary_unidentified=" +
        to_string(timing.component_graph_primary_unidentified) +
        ", component_graph_unidentified_output_rejects=" +
        to_string(timing.component_graph_unidentified_output_rejects) +
        ", component_image_shell_expansions=" +
        to_string(timing.component_image_shell_expansions) +
        ", component_image_shell_retry_successes=" +
        to_string(timing.component_image_shell_retry_successes) +
        ", component_image_shell_max_radius=" +
        to_string(timing.component_image_shell_max_radius) +
        ", component_image_shell_max_shift_count=" +
        to_string(timing.component_image_shell_max_shift_count) +
         ", component_warm_used=" + to_string(timing.component_warm_used) +
       ", component_warm_skip_search=" + to_string(timing.component_warm_skip_search) +
       ", component_search_frames=" + to_string(timing.component_search_frames) +
       ", component_graph_failed=" + to_string(timing.component_graph_failed) +
       ", component_graph_points_mismatch=" +
       to_string(timing.component_graph_points_mismatch) +
       ", component_graph_invalid=" + to_string(timing.component_graph_invalid) +
       ", component_graph_ultraclash=" +
       to_string(timing.component_graph_ultraclash) +
       ", component_graph_not_better_not_close=" +
       to_string(timing.component_graph_not_better_not_close) +
        ", component_refine_warm_shift_hits=" +
        to_string(timing.component_refine_warm_shift_hits) +
       ", component_refine_shortlist_hint_hits=" +
       to_string(timing.component_refine_shortlist_hint_hits) +
       ", component_pair_warm_shift_hits=" +
       to_string(timing.component_pair_warm_shift_hits) +
       ", component_refine_fanout_reduced=" +
       to_string(timing.component_refine_fanout_reduced) +
       ", component_graph_refine_score_calls=" +
       to_string(timing.component_graph_refine_score_calls) +
      ", component_graph_refine_coarse_score_calls=" +
      to_string(timing.component_graph_refine_coarse_score_calls) +
      ", component_graph_refine_coarse_stage1_score_calls=" +
      to_string(timing.component_graph_refine_coarse_stage1_score_calls) +
      ", component_graph_refine_coarse_stage2_score_calls=" +
      to_string(timing.component_graph_refine_coarse_stage2_score_calls) +
      ", component_graph_refine_exact_rescore_calls=" +
      to_string(timing.component_graph_refine_exact_rescore_calls) +
      ", component_graph_refine_preclash_prunes=" +
      to_string(timing.component_graph_refine_preclash_prunes) +
      ", component_graph_refine_trial_build_incremental=" +
      to_string(timing.component_graph_refine_trial_build_incremental) +
      ", component_graph_refine_trial_build_full=" +
      to_string(timing.component_graph_refine_trial_build_full) +
      ", component_graph_refine_trial_build_global_rebase=" +
      to_string(timing.component_graph_refine_trial_build_global_rebase) +
      ", component_graph_refine_stats_cached=" +
      to_string(timing.component_graph_refine_stats_cached) +
      ", component_graph_refine_stats_full=" +
      to_string(timing.component_graph_refine_stats_full) +
      ", component_graph_refine_anchor_spatial_calls=" +
      to_string(timing.component_graph_refine_anchor_spatial_calls) +
      ", component_graph_refine_anchor_spatial_fallbacks=" +
      to_string(timing.component_graph_refine_anchor_spatial_fallbacks) +
      ", component_graph_refine_anchor_spatial_disconnects=" +
      to_string(timing.component_graph_refine_anchor_spatial_disconnects) +
      ", component_graph_refine_anchor_spatial_expansions=" +
      to_string(timing.component_graph_refine_anchor_spatial_expansions) +
      ", component_graph_refine_anchor_spatial_exact_certified=" +
      to_string(timing.component_graph_refine_anchor_spatial_exact_certified) +
      ", component_graph_refine_anchor_spatial_uncertified=" +
      to_string(timing.component_graph_refine_anchor_spatial_uncertified) +
      ", component_graph_refine_anchor_spatial_max_k=" +
      to_string(timing.component_graph_refine_anchor_spatial_max_k) +
      ", component_graph_contact_epoch_scratch_resizes=" +
      to_string(timing.component_graph_contact_epoch_scratch_resizes) +
      ", component_graph_contact_epoch_wrap_clears=" +
      to_string(timing.component_graph_contact_epoch_wrap_clears) +
      ", component_graph_shift_descriptor_domain=" +
      to_string(timing.component_graph_shift_descriptor_domain) +
      ", component_graph_shift_cache_entries_peak=" +
      to_string(timing.component_graph_shift_cache_entries_peak) +
      ", component_graph_shift_cache_dense_entries_equivalent=" +
      to_string(
          timing.component_graph_shift_cache_dense_entries_equivalent) +
      ", component_cycle_sync_attempts=" +
      to_string(timing.component_cycle_sync_attempts) +
      ", component_cycle_sync_used=" +
      to_string(timing.component_cycle_sync_used) +
      ", component_cycle_sync_pair_candidates=" +
      to_string(timing.component_cycle_sync_pair_candidates) +
      ", component_cycle_sync_cycle_support=" +
      to_string(timing.component_cycle_sync_cycle_support) +
      ", component_cycle_sync_cycle_mismatch=" +
      to_string(timing.component_cycle_sync_cycle_mismatch) +
      ", component_cycle_sync_inconsistent_edges=" +
      to_string(timing.component_cycle_sync_inconsistent_edges) +
      ", pbctopo_topology_bonds_available=" +
      to_string(timing.pbctopo_topology_bonds_available) +
      ", pbctopo_topology_bond_edges_added=" +
      to_string(timing.pbctopo_topology_bond_edges_added) +
      ", pbctopo_topology_bond_edges_skipped=" +
      to_string(timing.pbctopo_topology_bond_edges_skipped) +
      ", pbctopo_geometry_edges_added=" +
      to_string(timing.pbctopo_geometry_edges_added) +
      ", pbctopo_hard_connectivity_edges=" +
      to_string(timing.pbctopo_hard_connectivity_edges) +
      ", pbctopo_soft_geometry_edges=" +
      to_string(timing.pbctopo_soft_geometry_edges) +
      ", pbctopo_steric_exclusion_pairs=" +
      to_string(timing.pbctopo_steric_exclusion_pairs) +
      ", pbctopo_component_logic_disabled_no_hard_graph=" +
      to_string(timing.pbctopo_component_logic_disabled_no_hard_graph) +
      ", pbctopo_topology_duplicate_edges=" +
      to_string(timing.pbctopo_topology_duplicate_edges) +
      ", pbctopo_topology_projected_bonds=" +
      to_string(timing.pbctopo_topology_projected_bonds) +
      ", pbctopo_topology_static_graph_frames=" +
      to_string(timing.pbctopo_topology_static_graph_frames) +
      ", pbctopo_metadata_graph_frames=" +
      to_string(timing.pbctopo_metadata_graph_frames) +
      ", pbctopo_metadata_residue_groups=" +
      to_string(timing.pbctopo_metadata_residue_groups) +
      ", pbctopo_metadata_intra_edges=" +
      to_string(timing.pbctopo_metadata_intra_edges) +
      ", pbctopo_metadata_inter_edges=" +
      to_string(timing.pbctopo_metadata_inter_edges) +
      ", pbctopo_metadata_inter_hard_edges=" +
      to_string(timing.pbctopo_metadata_inter_hard_edges) +
      ", pbctopo_metadata_inter_soft_edges=" +
      to_string(timing.pbctopo_metadata_inter_soft_edges) +
      ", pbctopo_geometry_fallback_frames=" +
      to_string(timing.pbctopo_geometry_fallback_frames) +
      ", pbctopo_topology_source=" +
      pbctopo_topology_source_label(timing) +
      ", pbctopo_topology_component_source=" +
      string((timing.pbctopo_topology_static_graph_frames > 0 &&
              timing.pbctopo_metadata_graph_frames > 0)
                 ? "topology+metadata"
                 : ((timing.frames > 0 &&
              timing.pbctopo_topology_static_graph_frames == timing.frames)
                        ? "topology"
                        : ((timing.frames > 0 &&
                            timing.pbctopo_metadata_graph_frames ==
                                timing.frames)
                               ? "metadata"
                               : ((timing.pbctopo_hard_connectivity_edges > 0)
                                      ? "trusted_mixed"
                                      : "none")))) +
      ", pbctopo_anchor_order_edges=" +
      to_string(timing.pbctopo_anchor_order_edges) +
      ", pbctopo_anchor_endpoint_edges=" +
      to_string(timing.pbctopo_anchor_endpoint_edges) +
      ", pbctopo_anchor_contact_pairs=" +
      to_string(timing.pbctopo_anchor_contact_pairs) +
      ", pbctopo_anchor_intra_owner_contact_pairs=" +
      to_string(timing.pbctopo_anchor_intra_owner_contact_pairs) +
      ", pbctopo_soft_observed_contact_pairs=" +
      to_string(timing.pbctopo_soft_observed_contact_pairs) +
      ", pbctopo_hard_required_owner_pairs=" +
      to_string(timing.pbctopo_hard_required_owner_pairs) +
      ", pbctopo_anchor_residue_order_violations=" +
      to_string(timing.pbctopo_anchor_residue_order_violations) +
      ", pbctopo_anchor_ambiguous_frames=" +
      to_string(timing.pbctopo_anchor_ambiguous_frames) +
      ", component_context_builds=" +
      to_string(timing.component_context_builds) +
      ", component_offset_extracts=" +
      to_string(timing.component_offset_extracts) + ", apply_calls=" +
      to_string(timing.apply_calls) +
      ", component_partial_checks=" +
      to_string(timing.component_partial_checks) +
      ", component_partial_prunes=" +
      to_string(timing.component_partial_prunes) +
      ", component_bbox_skips=" + to_string(timing.component_bbox_skips) +
      ", component_score_calls=" +
      to_string(timing.component_score_calls) + ", component_partial_time=" +
      to_string(timing.component_partial_s) + " s, component_score_time=" +
      to_string(timing.component_score_s) + " s");
  append_pbctopo_log_line(
      "[pbctopo timing] component_pair_table_builds=" +
      to_string(timing.component_pair_table_builds) +
      ", component_pair_table_time=" +
      to_string(timing.component_pair_table_s) +
      " s, component_pair_table_prunes=" +
      to_string(timing.component_pair_table_prunes) +
      ", component_pair_table_component_count=" +
      to_string(timing.component_pair_table_component_count) +
      ", component_pair_table_shift_count=" +
      to_string(timing.component_pair_table_shift_count) +
      ", component_pair_table_relative_deltas=" +
      to_string(timing.component_pair_table_relative_deltas) +
      ", component_pair_table_relative_reuses=" +
      to_string(timing.component_pair_table_relative_reuses) +
      ", component_relative_domain_analytic_builds=" +
      to_string(timing.component_relative_domain_analytic_builds) +
      ", component_relative_domain_generic_builds=" +
      to_string(timing.component_relative_domain_generic_builds) +
      ", component_pair_table_dense_entries_equivalent=" +
      to_string(timing.component_pair_table_dense_entries_equivalent) +
      ", component_pair_table_materialized_entries=" +
      to_string(timing.component_pair_table_materialized_entries) +
      ", component_pair_table_bbox_prefiltered=" +
      to_string(timing.component_pair_table_bbox_prefiltered) +
      ", component_pair_table_sparse_rows=" +
      to_string(timing.component_pair_table_sparse_rows) +
      ", component_pair_table_peak_bytes_estimate=" +
      to_string(timing.component_pair_table_peak_bytes_estimate) +
      ", component_search_zero_current_mask_prunes=" +
      to_string(timing.component_search_zero_current_mask_prunes) +
      ", component_search_zero_future_mask_prunes=" +
      to_string(timing.component_search_zero_future_mask_prunes) +
      ", component_search_fixed_minor_frames=" +
      to_string(timing.component_search_fixed_minor_frames) +
      ", component_search_fixed_minor_components=" +
      to_string(timing.component_search_fixed_minor_components) +
      ", component_search_shift_descriptors=" +
      to_string(timing.component_search_shift_descriptors) +
      ", component_search_shift_materialized=" +
      to_string(timing.component_search_shift_materialized) +
      ", component_search_shift_domain_max=" +
      to_string(timing.component_search_shift_domain_max) +
      ", component_search_lazy_materialize_time=" +
      to_string(timing.component_search_lazy_materialize_s) + " s" +
      ", component_search_seed_warm=" +
      to_string(timing.component_search_seed_warm) +
      ", component_search_seed_graph=" +
      to_string(timing.component_search_seed_graph) +
      ", component_search_seed_none=" +
      to_string(timing.component_search_seed_none) +
      ", component_search_prefix_continuity_checks=" +
      to_string(timing.component_search_prefix_continuity_checks) +
      ", component_search_prefix_continuity_prunes=" +
      to_string(timing.component_search_prefix_continuity_prunes) +
      ", component_search_prefix_continuity_hard_rejects=" +
      to_string(timing.component_search_prefix_continuity_hard_rejects) +
      ", component_search_exact_budget_skips=" +
      to_string(timing.component_search_exact_budget_skips) +
      ", component_audit_exact_frames=" +
      to_string(timing.component_audit_exact_frames) +
      ", component_audit_exact_completed=" +
      to_string(timing.component_audit_exact_completed) +
      ", component_audit_exact_incomplete=" +
      to_string(timing.component_audit_exact_incomplete) +
      ", component_audit_exact_state_limit=" +
      to_string(timing.component_audit_exact_state_limit) +
      ", component_audit_exact_wall_time_limit=" +
      to_string(timing.component_audit_exact_wall_time_limit) +
      ", component_audit_exact_memory_limit=" +
      to_string(timing.component_audit_exact_memory_limit) +
      ", component_audit_exact_reference_matches=" +
      to_string(timing.component_audit_exact_reference_matches) +
      ", component_audit_exact_reference_mismatches=" +
      to_string(timing.component_audit_exact_reference_mismatches) +
      ", component_pair_table_all_atom_relative_checks=" +
      to_string(timing.component_pair_table_all_atom_relative_checks) +
      ", component_pair_table_all_atom_relative_rejects=" +
      to_string(timing.component_pair_table_all_atom_relative_rejects) +
      ", component_pair_table_all_atom_materialized=" +
      to_string(timing.component_pair_table_all_atom_materialized) +
      ", component_search_beam_adaptive_retries=" +
      to_string(timing.component_search_beam_adaptive_retries) +
      ", component_search_beam_adaptive_retry_successes=" +
      to_string(timing.component_search_beam_adaptive_retry_successes) +
      ", component_search_beam_adaptive_max_width=" +
      to_string(timing.component_search_beam_adaptive_max_width) +
      ", component_search_exact_state_upper_bound=" +
      to_string(timing.component_search_exact_state_upper_bound) +
      ", component_search_filtered_state_upper_bound=" +
      to_string(timing.component_search_filtered_state_upper_bound) +
      ", component_search_arc_consistency_passes=" +
      to_string(timing.component_search_arc_consistency_passes) +
      ", component_search_arc_consistency_revisions=" +
      to_string(timing.component_search_arc_consistency_revisions) +
      ", component_search_arc_consistency_removals=" +
      to_string(timing.component_search_arc_consistency_removals) +
      ", component_search_arc_consistency_queue_pushes=" +
      to_string(timing.component_search_arc_consistency_queue_pushes) +
      ", component_search_arc_consistency_failures=" +
      to_string(timing.component_search_arc_consistency_failures) +
      ", component_search_exact_recoveries=" +
      to_string(timing.component_search_exact_recoveries) +
      ", component_search_arc_consistency_time=" +
      to_string(timing.component_search_arc_consistency_s) + " s" +
      ", component_search_prefix_continuity_time=" +
      to_string(timing.component_search_prefix_continuity_s) +
      " s" +
      ", component_large_mode_merge_frames=" +
      to_string(timing.component_large_mode_merge_frames) +
      ", component_large_mode_pair_only_frames=" +
      to_string(timing.component_large_mode_pair_only_frames) +
      ", component_large_mode_beam_frames=" +
      to_string(timing.component_large_mode_beam_frames) +
      ", component_large_mode_merge_count=" +
      to_string(timing.component_large_mode_merge_count) +
      ", component_large_mode_internal_offsets_preserved=" +
      to_string(timing.component_large_mode_internal_offsets_preserved) +
      ", component_large_mode_internal_offset_nonzero=" +
      to_string(timing.component_large_mode_internal_offset_nonzero) +
      ", component_large_mode_graph_seed_count=" +
      to_string(timing.component_large_mode_graph_seed_count) +
      ", component_large_mode_graph_seed_refined=" +
      to_string(timing.component_large_mode_graph_seed_refined) +
      ", component_large_mode_graph_seed_nonbaseline_win=" +
      to_string(timing.component_large_mode_graph_seed_nonbaseline_win) +
      ", component_large_mode_beam_expanded=" +
      to_string(timing.component_large_mode_beam_expanded) +
      ", component_large_mode_beam_kept=" +
      to_string(timing.component_large_mode_beam_kept) +
      ", component_large_mode_beam_complete_assignments=" +
      to_string(timing.component_large_mode_beam_complete_assignments) +
      ", component_large_mode_beam_final_rescored=" +
      to_string(timing.component_large_mode_beam_final_rescored) +
      ", component_large_mode_beam_final_rescore_all_frames=" +
      to_string(timing.component_large_mode_beam_final_rescore_all_frames) +
      ", component_search_beam_provisional_children=" +
      to_string(timing.component_search_beam_provisional_children) +
      ", component_search_beam_state_materializations=" +
      to_string(timing.component_search_beam_state_materializations) +
      ", component_search_beam_state_copies_avoided=" +
      to_string(timing.component_search_beam_state_copies_avoided) +
      ", component_search_beam_parent_state_moves=" +
      to_string(timing.component_search_beam_parent_state_moves) +
      ", component_search_certification_survivors=" +
      to_string(timing.component_search_certification_survivors) +
      ", component_search_certification_band_retained=" +
      to_string(timing.component_search_certification_band_retained) +
      ", component_large_mode_fallback_frames=" +
      to_string(timing.component_large_mode_fallback_frames) +
      ", component_search_shift_warm_hits=" +
      to_string(timing.component_search_shift_warm_hits) +
      ", pbctopo_certification_attempts=" +
      to_string(timing.pbctopo_certification_attempts) +
      ", pbctopo_certification_successes=" +
      to_string(timing.pbctopo_certification_successes) +
      ", pbctopo_certification_assignment_rejects=" +
      to_string(timing.pbctopo_certification_assignment_rejects) +
      ", pbctopo_certification_hard_rejects=" +
      to_string(timing.pbctopo_certification_hard_rejects) +
      ", pbctopo_output_certification_band_precheck_rejects=" +
      to_string(timing.pbctopo_output_certification_band_precheck_rejects) +
      ", pbctopo_output_certification_band_rebuilds=" +
      to_string(timing.pbctopo_output_certification_band_rebuilds) +
      ", pbctopo_output_certification_band_offset_bytes_peak=" +
      to_string(timing.pbctopo_output_certification_band_offset_bytes_peak) +
      ", pbctopo_output_certification_band_point_bytes_avoided_peak=" +
      to_string(
          timing.pbctopo_output_certification_band_point_bytes_avoided_peak) +
      ", pbctopo_heavy_search_frames=" +
      to_string(timing.pbctopo_heavy_search_frames) +
      ", pbctopo_heavy_search_all_atom_candidate_rejects=" +
      to_string(timing.pbctopo_heavy_search_all_atom_candidate_rejects) +
      ", pbctopo_heavy_search_top_candidate_rejects=" +
      to_string(timing.pbctopo_heavy_search_top_candidate_rejects) +
      ", pbctopo_heavy_search_all_atom_ranking_recoveries=" +
      to_string(timing.pbctopo_heavy_search_all_atom_ranking_recoveries) +
      ", pbctopo_heavy_search_all_atom_band_exhausted=" +
      to_string(timing.pbctopo_heavy_search_all_atom_band_exhausted) +
      ", pbctopo_runtime_simd_forced_scalar=" +
      to_string(titan_simd_dispatch::runtime_simd_forced_scalar() ? 1 : 0) +
      ", pbctopo_output_certification_band_candidates=" +
      to_string(timing.pbctopo_output_certification_band_candidates) +
      ", pbctopo_output_certification_band_peak=" +
      to_string(timing.pbctopo_output_certification_band_peak) +
      ", pbctopo_output_certification_band_rejects=" +
      to_string(timing.pbctopo_output_certification_band_rejects) +
      ", pbctopo_output_certification_band_recoveries=" +
      to_string(timing.pbctopo_output_certification_band_recoveries) +
      ", pbctopo_output_certification_band_expanded_frames=" +
      to_string(timing.pbctopo_output_certification_band_expanded_frames) +
      ", pbctopo_output_certification_band_expanded_recoveries=" +
      to_string(
          timing.pbctopo_output_certification_band_expanded_recoveries) +
      ", pbctopo_certification_topology_edges_checked=" +
      to_string(timing.pbctopo_certification_topology_edges_checked) +
      ", pbctopo_certification_ambiguous_topology_edges=" +
      to_string(timing.pbctopo_certification_ambiguous_topology_edges) +
      ", pbctopo_certification_topology_edge_residuals=" +
      to_string(timing.pbctopo_certification_topology_edge_residuals) +
      ", pbctopo_certification_topology_cycle_residuals=" +
      to_string(timing.pbctopo_certification_topology_cycle_residuals) +
      ", pbctopo_certification_atom_image_residuals=" +
      to_string(timing.pbctopo_certification_atom_image_residuals) +
      ", pbctopo_certification_assignment_point_residuals=" +
      to_string(timing.pbctopo_certification_assignment_point_residuals) +
      ", pbctopo_certification_steric_pair_checks=" +
      to_string(timing.pbctopo_certification_steric_pair_checks) +
      ", pbctopo_certification_steric_clashes=" +
      to_string(timing.pbctopo_certification_steric_clashes) +
      ", pbctopo_certification_required_contacts_checked=" +
      to_string(timing.pbctopo_certification_required_contacts_checked) +
      ", pbctopo_certification_required_contacts_lost=" +
      to_string(timing.pbctopo_certification_required_contacts_lost) +
      ", pbctopo_required_interface_contacts_loaded=" +
      to_string(timing.pbctopo_required_interface_contacts_loaded) +
      ", component_required_contact_constraint_builds=" +
      to_string(timing.component_required_contact_constraint_builds) +
      ", component_required_contact_constraint_pairs=" +
      to_string(timing.component_required_contact_constraint_pairs) +
      ", component_required_contact_relative_deltas_checked=" +
      to_string(timing.component_required_contact_relative_deltas_checked) +
      ", component_required_contact_relative_deltas_rejected=" +
      to_string(timing.component_required_contact_relative_deltas_rejected) +
      ", component_required_contact_infeasible=" +
      to_string(timing.component_required_contact_infeasible) +
      ", pbctopo_certification_explicit_interface_contacts_checked=" +
      to_string(
          timing.pbctopo_certification_explicit_interface_contacts_checked) +
      ", pbctopo_certification_explicit_interface_contacts_lost=" +
      to_string(
          timing.pbctopo_certification_explicit_interface_contacts_lost) +
      ", pbctopo_geometry_classifications=" +
      to_string(timing.pbctopo_geometry_classifications) +
      ", pbctopo_compact_unit_box_certified_frames=" +
      to_string(timing.pbctopo_compact_unit_box_certified_frames) +
      ", pbctopo_noncompact_certified_frames=" +
      to_string(timing.pbctopo_noncompact_certified_frames) +
      ", pbctopo_extended_certified_frames=" +
      to_string(timing.pbctopo_extended_certified_frames) +
      ", pbctopo_winding_rejected_candidates=" +
      to_string(timing.pbctopo_winding_rejected_candidates) +
      ", pbctopo_extended_winding_rejected_candidates=" +
      to_string(timing.pbctopo_extended_winding_rejected_candidates) +
      ", pbctopo_owner_com_split_observations=" +
      to_string(timing.pbctopo_owner_com_split_observations) +
      ", pbctopo_warm_state_commits=" +
      to_string(timing.pbctopo_warm_state_commits) +
      ", pbctopo_warm_state_resets=" +
      to_string(timing.pbctopo_warm_state_resets) +
      ", pbctopo_warm_state_commit_rejects=" +
      to_string(timing.pbctopo_warm_state_commit_rejects) +
      ", pbctopo_warm_projection_rejects=" +
      to_string(timing.pbctopo_warm_projection_rejects) +
      ", pbctopo_warm_component_rejects=" +
      to_string(timing.pbctopo_warm_component_rejects) +
      ", pbctopo_weak_observation_output_frames=" +
      to_string(timing.pbctopo_weak_observation_output_frames) +
      ", pbctopo_contradicted_output_frames=" +
      to_string(timing.pbctopo_contradicted_output_frames) +
      ", pbctopo_selected_evidence_audits=" +
      to_string(timing.pbctopo_selected_evidence_audits) +
      ", pbctopo_selected_evidence_contacts_checked=" +
      to_string(timing.pbctopo_selected_evidence_contacts_checked) +
      ", pbctopo_selected_evidence_contacts_lost=" +
      to_string(timing.pbctopo_selected_evidence_contacts_lost) +
      ", pbctopo_eobs_construction_attempts=" +
      to_string(timing.pbctopo_eobs_construction_attempts) +
      ", pbctopo_eobs_construction_completed=" +
      to_string(timing.pbctopo_eobs_construction_completed) +
      ", pbctopo_eobs_construction_failures=" +
      to_string(timing.pbctopo_eobs_construction_failures) +
      ", pbctopo_eobs_cell_setup_failures=" +
      to_string(timing.pbctopo_eobs_cell_setup_failures) +
      ", pbctopo_eobs_atom_index_failures=" +
      to_string(timing.pbctopo_eobs_atom_index_failures) +
      ", pbctopo_eobs_mic_query_failures=" +
      to_string(timing.pbctopo_eobs_mic_query_failures) +
      ", pbctopo_selected_evidence_audit_s=" +
      to_string(timing.pbctopo_selected_evidence_audit_s) +
      ", pbctopo_weak_observation_warm_preserved=" +
      to_string(timing.pbctopo_weak_observation_warm_preserved) +
      ", pbctopo_warm_layout_mismatches=" +
      to_string(timing.pbctopo_warm_layout_mismatches) +
      ", pbctopo_diagnostic_only_candidates=" +
      to_string(timing.pbctopo_diagnostic_only_candidates) +
      ", pbctopo_virtual_geometry_reference_evals=" +
      to_string(timing.pbctopo_virtual_geometry_reference_evals) +
      ", pbctopo_virtual_geometry_reference_safe=" +
      to_string(timing.pbctopo_virtual_geometry_reference_safe) +
      ", pbctopo_virtual_geometry_reference_clashing=" +
      to_string(timing.pbctopo_virtual_geometry_reference_clashing) +
      ", pbctopo_virtual_geometry_outlier_rejects=" +
      to_string(timing.pbctopo_virtual_geometry_outlier_rejects) +
      ", pbctopo_virtual_geometry_outlier_observations=" +
      to_string(timing.pbctopo_virtual_geometry_outlier_observations) +
      ", component_graph_outlier_rejects=" +
      to_string(timing.component_graph_outlier_rejects) +
      ", component_search_outlier_rejects=" +
      to_string(timing.component_search_outlier_rejects) +
      ", component_graph_outlier_observations=" +
      to_string(timing.component_graph_outlier_observations) +
      ", component_search_outlier_observations=" +
      to_string(timing.component_search_outlier_observations) +
      ", component_score_continuity_time=" +
      to_string(timing.component_score_continuity_s) +
      " s, component_score_anchor_time=" +
      to_string(timing.component_score_anchor_s) +
      " s, component_score_stats_principal_time=" +
      to_string(timing.component_score_stats_principal_s) +
      " s, component_score_component_metrics_time=" +
      to_string(timing.component_score_component_metrics_s) +
      " s, component_score_continuity_rejects=" +
      to_string(timing.component_score_continuity_rejects) +
      ", component_score_anchor_rejects=" +
      to_string(timing.component_score_anchor_rejects) +
      ", component_score_prefix_prunes_after_continuity=" +
      to_string(timing.component_score_prefix_prunes_after_continuity) +
      ", component_score_prefix_prunes_after_anchor=" +
      to_string(timing.component_score_prefix_prunes_after_anchor) +
      ", component_score_prefix_prunes_after_principal=" +
      to_string(timing.component_score_prefix_prunes_after_principal) +
      ", component_score_prefix_prunes_after_component_metrics=" +
      to_string(timing.component_score_prefix_prunes_after_component_metrics) +
      ", component_score_prefix_prunes_after_final=" +
      to_string(timing.component_score_prefix_prunes_after_final) +
      ", component_search_depth_visits=[d0:" +
      to_string(timing.component_search_depth_visits[0]) + ",d1:" +
      to_string(timing.component_search_depth_visits[1]) + ",d2:" +
      to_string(timing.component_search_depth_visits[2]) + ",d3:" +
      to_string(timing.component_search_depth_visits[3]) + ",d4:" +
      to_string(timing.component_search_depth_visits[4]) + ",d5:" +
      to_string(timing.component_search_depth_visits[5]) + ",d6:" +
      to_string(timing.component_search_depth_visits[6]) + ",d7:" +
      to_string(timing.component_search_depth_visits[7]) + "]");
  append_pbctopo_log_line(
      "[pbctopo math] pair_shift_score=normalized, shift_metric="
      "cartesian_component_mean_A2, gauge=closest_lattice_mean, cycle_residual="
      "fundamental_basis, anchor_metrics=" +
      string(titan_pbctopo::anchor_metric_mode_name(anchor_metric_mode)) +
      ", clash_model=vdw_radius, vdw_pair_checks=" +
      to_string(timing.pbctopo_vdw_pair_checks) +
      ", zero_lj_policy=" +
      string(titan_pbctopo::kPbctopoZeroLjStericPolicy) +
      ", zero_lj_exempt_pairs=" +
      to_string(timing.pbctopo_vdw_explicit_zero_lj_exempt_pairs) +
      ", vdw_radius_pairs=" + to_string(timing.pbctopo_vdw_radius_pairs) +
      ", vdw_fallback_pairs=" +
      to_string(timing.pbctopo_vdw_radius_fallback_pairs) +
      ", temporal_mode=" +
      string(titan_pbctopo::temporal_mode_name(temporal_mode)) +
      ", temporal_candidate_frames=" +
      to_string(timing.pbctopo_temporal_candidate_frames) +
      ", temporal_candidates=" +
      to_string(timing.pbctopo_temporal_candidates) +
      ", temporal_certified_candidates=" +
      to_string(timing.pbctopo_temporal_certified_candidates) +
      ", temporal_assignment_deduped=" +
      to_string(timing.pbctopo_temporal_assignment_deduped) +
      ", temporal_diagnostic_only_frames=" +
      to_string(timing.pbctopo_temporal_diagnostic_only_frames) +
      ", temporal_proposed_changes=" +
      to_string(timing.pbctopo_temporal_proposed_changes) +
      ", temporal_path_changes=" +
      to_string(timing.pbctopo_temporal_path_changes) +
      ", temporal_replay_frames=" +
      to_string(timing.pbctopo_temporal_replay_frames) +
      ", temporal_replay_applied=" +
      to_string(timing.pbctopo_temporal_replay_applied) +
      ", temporal_replay_skipped=" +
      to_string(timing.pbctopo_temporal_replay_skipped) +
      ", pbctopo_lattice_audits=" +
      to_string(timing.pbctopo_lattice_audits) +
      ", pbctopo_lattice_invalid_frames=" +
      to_string(timing.pbctopo_lattice_invalid_frames) +
      ", pbctopo_lattice_reduction_fallback_frames=" +
      to_string(timing.pbctopo_lattice_reduction_fallback_frames) +
      ", pbctopo_lattice_query_failures=" +
      to_string(timing.pbctopo_lattice_query_failures) +
      ", pbctopo_lattice_fatal_query_failures=" +
      to_string(timing.pbctopo_lattice_fatal_query_failures) +
      ", pbctopo_lattice_optional_query_failures=" +
      to_string(timing.pbctopo_lattice_optional_query_failures) +
      ", pbctopo_lattice_query_ambiguities=" +
      to_string(timing.pbctopo_lattice_query_ambiguities) +
      ", pbctopo_lattice_query_budget_failures=" +
      to_string(timing.pbctopo_lattice_query_budget_failures) +
      ", pbctopo_lattice_worker_queries=" +
      to_string(timing.pbctopo_lattice_worker_queries) +
      ", pbctopo_lattice_worker_query_failures=" +
      to_string(timing.pbctopo_lattice_worker_query_failures) +
      ", pbctopo_lattice_worker_query_ambiguities=" +
      to_string(timing.pbctopo_lattice_worker_query_ambiguities) +
      ", pbctopo_lattice_worker_query_budget_failures=" +
      to_string(timing.pbctopo_lattice_worker_query_budget_failures) +
      ", pbctopo_lattice_condition_max=" +
      to_string(timing.pbctopo_lattice_condition_max));
  append_pbctopo_log_line("[pbctopo] Valid candidate evaluations: " +
                          to_string(valid_candidates));
  append_pbctopo_log_line("[pbctopo] end run");
  return 0;
}
