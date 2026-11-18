#pragma once

// The PBCTOPO solver (Multi_MOL::solve_pbc_pbctopo) as two classes:
// PbctopoSolveRun holds what lives for the whole run (reports, timing, warm
// state, temporal buffers, scratch) and runs setup, the frame loop, the
// temporal pass and the final reports; PbctopoFrameSolver holds one frame's
// state and runs its stages in order. Members keep the names and lifetimes
// the function's locals had; the former lambdas are member functions.

#include "pbctopo/solve/solve_common.h"

#include <optional>

class Multi_MOL;
struct arguments;
class TitanExecutor;

namespace pbctopo_solve {

struct PbctopoTemporalComponentLayout {
  vector<uint32_t> atom_component_index;
  vector<uint64_t> component_ids;
  vector<double> component_weights;
  size_t component_count = 0;
};
// A candidate's evidence audit in two halves, so that the band loop can
// run the first on executor workers: compute reads only frame constants;
// apply writes the evidence cache and the candidate.  nullopt is the input
// the serial code turned away before auditing.
using CandidateEvidenceAudit =
    std::expected<titan_pbctopo::PbctopoSelectedEvidenceAudit, string>;
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
struct PbctopoOutputCandidateSnapshot {
  PbctopoCutCandidate candidate;
  vector<Int3> offsets;
  PbctopoFrameMode mode = PbctopoFrameMode::None;
};

class PbctopoFrameSolver;

class PbctopoSolveRun {
public:
  using clock = chrono::steady_clock;

  PbctopoSolveRun(Multi_MOL &self, Multi_MOL &mol2, arguments &Arguments,
                  TitanExecutor *executor, size_t frame_report_offset);
  int run();

  // Stages of the run.
  std::optional<int> setup();
  void run_temporal();
  int finish();

  // Former lambdas of the run.
  void reset_uncommitted_warm_state();
  void write_or_defer_frame_report(size_t frame_idx, PbctopoFrameReport report);
  void fallback_pbci_frame(size_t frame_index);
  template <class AtomToComponent>
  vector<pair<size_t, size_t>> build_explicit_evidence_component_edges(
      const AtomToComponent &atom_to_component, size_t component_count) const {
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
  }

  Multi_MOL &self;
  Multi_MOL &mol2;
  arguments &Arguments;
  TitanExecutor *executor;
  size_t frame_report_offset;

  size_t frame_count{};
  ofstream pbctopo_frame_report{};
  ofstream pbctopo_assignment_set_report{};
  ofstream pbctopo_rejected_assignment_report{};
  ofstream pbctopo_candidate_evidence_summary_report{};
  ofstream pbctopo_candidate_evidence_relation_report{};
  ofstream pbctopo_component_layout_report{};
  ofstream pbctopo_hard_edge_report{};
  bool h9_orbit_enabled{};
  ofstream h9_component_representative_report{};
  ofstream h9_evidence_component_report{};
  ofstream h9_orbit_report{};
  vector<PbctopoComponentLayoutObservation> &emitted_component_layouts;
  size_t fallback_frames{};
  size_t valid_candidates{};
  PbctopoTimingStats timing{};
  PbctopoWarmState warm_state{};
  PbctopoRuntimeCutoffs runtime_cutoffs{};
  PbctopoSolveMode solve_mode{};
  PbctopoSearchPolicy search_policy{};
  bool audit_exact_search{};
  static constexpr size_t kBytesPerMiB = 1024u * 1024u;
  size_t audit_memory_mib{};
  size_t audit_memory_bytes{};
  PbctopoComponentSearchLimits audit_exact_limits{};
  PbctopoAnchorBackend anchor_backend{};
  PbctopoAnchorMetricMode anchor_metric_mode{};
  PbctopoTemporalMode temporal_mode{};
  size_t requested_exact_component_limit{};
  size_t exact_component_limit{};
  size_t requested_beam_width{};
  size_t beam_width{};
  bool component_shell_radius_explicit{};
  int configured_component_shell_radius{};
  PbctopoTrajectoryContext trajectory_context{};
  vector<titan_pbctopo::PbctopoRequiredContact> explicit_required_contacts{};
  vector<vector<PbctopoCutCandidate>> temporal_cut_candidates{};
  vector<PbctopoCutCandidate> greedy_selected_candidates{};
  vector<PbctopoTemporalComponentLayout> temporal_component_layouts{};
  vector<size_t> temporal_frame_layout{};
  vector<PbctopoFrameReport> deferred_frame_reports{};
  vector<unsigned char> deferred_frame_report_valid{};
  vector<PbctopoContactHypothesisSet> temporal_contact_hypotheses{};
  size_t evidence_relation_export_groups{};
  size_t evidence_relation_export_rows{};
  size_t evidence_relation_export_duplicates{};
  size_t evidence_relation_export_failures{};
  bool has_ligand_static{};
  bool static_topology_ready{};
  bool static_metadata_ready{};
  string static_graph_source{};
  PbctopoFrameScratch scratch{};
  PbctopoHullScratch hull_scratch{};
  vector<PbctopoAtom> &base_atoms;
  vector<PbctopoAtom> &projection_atoms;
  vector<PbctopoAtom> &component_atoms;
  vector<PbctopoLocalEdge> &local_edges;
  vector<PbctopoLocalEdge> &component_scoring_edges;
  vector<PbctopoLocalEdge> &projection_edges;
  vector<PbctopoContinuityPair> &projection_continuity_pairs;
  vector<size_t> &projection_anchor_indices;
  vector<std::uint8_t> &local_graph_atom_coverage;
  unordered_set<uint64_t> &local_edge_set;
  unordered_set<uint64_t> &steric_exclusion_set;
  unordered_set<uint64_t> &projection_edge_set;
  vector<vector<size_t>> &components;
  vector<PbctopoShift> &shifts;
  vector<PbctopoSeed3D> &volume_seeds;
  vector<Vec3> &warm_points;
  vector<Vec3> &coarse_warm_points;
  vector<Vec3> &placed_points;
  vector<Vec3> &best_safe_points;
  vector<Vec3> &verify_points;
  vector<Vec3> &component_points;
  vector<PbctopoCutCandidate> &coarse_candidates;
  vector<PbctopoCoarseSeedOption> &coarse_seed_options;
  vector<PbctopoCutCandidate> &coarse_results;
  vector<size_t> &projection_index;
  vector<size_t> &atom_component_index;
  titan_pbctopo::PbctopoVirtualDiscreteSearchWorkspace
      &virtual_search_workspace;
  PbctopoVirtualDiscreteSearchPlan &virtual_search_plan;
  vector<PbctopoHullSpan> hull_owner_spans{};
  vector<Vec3> hull_rescue_points{};
  bool projection_clash_skip_active{};
  string llps_local_unit_source{};
};

class PbctopoFrameSolver {
public:
  using clock = chrono::steady_clock;

  PbctopoFrameSolver(PbctopoSolveRun &run_, size_t frame_idx_);
  void solve();

  // Stages of one frame, in order.
  bool prepare_frame();
  void classify_frame();
  void prepare_component_context();
  void collect_seeds();
  void search_virtual();
  void search_components();
  bool certify_and_commit();

  // Former lambdas of the frame loop.
  void require_complete_eobs(const PbctopoAnchorGraphContext &context,
                             const char *scope);
  void ensure_component_context();
  // The frame's edge nearest images (certification.h), built once on the
  // frame's thread; workers only read it, and use it only once it is built.
  const titan_pbctopo::PbctopoEdgeImageCache *ensure_edge_image_cache();
  vector<Int3> extract_offsets_and_time(const vector<PbctopoAtom> &source_atoms,
                                        const vector<Vec3> &points);
  bool component_certificate_domain_complete();
  PbctopoComponentSearchCallbacks make_frame_component_search_callbacks();
  void
  cache_candidate_evidence_audit(const PbctopoImageAssignment &assignment,
                                 const PbctopoSelectedEvidenceAudit &audit);
  void record_report_certified_candidate(const PbctopoCutCandidate &candidate);
  auto compute_candidate_evidence(const PbctopoCutCandidate &candidate,
                                  const vector<Vec3> &points)
      -> optional<CandidateEvidenceAudit>;
  auto
  apply_candidate_evidence(PbctopoCutCandidate &candidate,
                           const optional<CandidateEvidenceAudit> &computed)
      -> bool;
  auto annotate_candidate_evidence(PbctopoCutCandidate &candidate,
                                   const vector<Vec3> &points) -> bool;
  void compute_candidate_certification(CandidateCertificationWork &work,
                                       PbctopoCutCandidate &candidate,
                                       const vector<Vec3> &points,
                                       const vector<Int3> *known_offsets,
                                       bool on_worker);
  bool apply_candidate_certification(const CandidateCertificationWork &work,
                                     PbctopoCutCandidate &candidate);
  bool
  certify_candidate_for_output(PbctopoCutCandidate &candidate,
                               const vector<Vec3> &points,
                               const vector<Int3> *known_offsets = nullptr);
  void record_temporal_certified_candidate(
      const PbctopoCutCandidate &source_candidate,
      const vector<Vec3> &candidate_points,
      const vector<Int3> *known_offsets = nullptr);
  void apply_cut_candidate_timed(const vector<Vec3> &points);
  bool
  validate_rescue_candidate_points(const vector<Vec3> &candidate_points,
                                   const PbctopoCutCandidate *incumbent_cut,
                                   PbctopoCutCandidate &candidate_out);
  void record_report_soft_candidate(const PbctopoCutCandidate &candidate);
  bool record_output_candidate(PbctopoCutCandidate candidate,
                               const vector<Vec3> &points,
                               PbctopoFrameMode mode,
                               const vector<Int3> *known_offsets = nullptr);
  void record_frame_failure(PbctopoFailureReason reason);
  bool commit_certified_candidate_to_warm_state(
      const PbctopoCutCandidate &selected_candidate);
  void apply_selected_evidence_to_frame_identifiability();
  void apply_periodic_contact_audit(
      const titan_pbctopo::PbctopoPeriodicContactAudit &audit,
      bool relations_valid);
  void audit_selected_evidence(const vector<Vec3> &selected_points,
                               const PbctopoImageAssignment &assignment);
  bool selected_relative_images_identified();
  void export_h9_partition_and_orbit(const PbctopoCutCandidate &base_candidate,
                                     const vector<Int3> &base_offsets);
  void finalize_frame_report(PbctopoFrameStatus status, PbctopoFrameMode mode,
                             PbctopoFailureReason reason,
                             const PbctopoCutCandidate *selected_candidate);
  bool accept_rescue_candidate(const vector<Vec3> &candidate_points,
                               PbctopoCutCandidate candidate);
  void accumulate_worker_lattice_audit(
      const titan_pbctopo::PbctopoLatticeQueryAudit &before);

  // The run's former lambdas, as the frame code calls them.
  void reset_uncommitted_warm_state() {
    return run.reset_uncommitted_warm_state();
  }
  template <class AtomToComponent>
  auto build_explicit_evidence_component_edges(
      const AtomToComponent &atom_to_component, size_t component_count) {
    return run.build_explicit_evidence_component_edges(atom_to_component,
                                                       component_count);
  }
  void write_or_defer_frame_report(size_t frame_idx,
                                   PbctopoFrameReport report) {
    return run.write_or_defer_frame_report(frame_idx, report);
  }
  void fallback_pbci_frame(size_t frame_index) {
    return run.fallback_pbci_frame(frame_index);
  }

  PbctopoSolveRun &run;
  Multi_MOL &self;
  Multi_MOL &mol2;
  arguments &Arguments;
  TitanExecutor *executor;
  size_t frame_report_offset;
  // Run state the frame code uses, under the names it had.
  const size_t &frame_count;
  ofstream &pbctopo_assignment_set_report;
  ofstream &pbctopo_rejected_assignment_report;
  ofstream &pbctopo_candidate_evidence_summary_report;
  ofstream &pbctopo_candidate_evidence_relation_report;
  ofstream &pbctopo_component_layout_report;
  ofstream &pbctopo_hard_edge_report;
  const bool &h9_orbit_enabled;
  ofstream &h9_component_representative_report;
  ofstream &h9_evidence_component_report;
  ofstream &h9_orbit_report;
  vector<PbctopoComponentLayoutObservation> &emitted_component_layouts;
  size_t &fallback_frames;
  size_t &valid_candidates;
  PbctopoTimingStats &timing;
  PbctopoWarmState &warm_state;
  const PbctopoRuntimeCutoffs &runtime_cutoffs;
  const PbctopoSolveMode &solve_mode;
  const bool &audit_exact_search;
  const PbctopoComponentSearchLimits &audit_exact_limits;
  const PbctopoAnchorBackend &anchor_backend;
  const PbctopoAnchorMetricMode &anchor_metric_mode;
  const PbctopoTemporalMode &temporal_mode;
  const size_t &exact_component_limit;
  const size_t &beam_width;
  const bool &component_shell_radius_explicit;
  const int &configured_component_shell_radius;
  const PbctopoTrajectoryContext &trajectory_context;
  vector<titan_pbctopo::PbctopoRequiredContact> &explicit_required_contacts;
  vector<vector<PbctopoCutCandidate>> &temporal_cut_candidates;
  vector<PbctopoCutCandidate> &greedy_selected_candidates;
  vector<PbctopoTemporalComponentLayout> &temporal_component_layouts;
  vector<size_t> &temporal_frame_layout;
  vector<PbctopoContactHypothesisSet> &temporal_contact_hypotheses;
  size_t &evidence_relation_export_groups;
  size_t &evidence_relation_export_rows;
  size_t &evidence_relation_export_duplicates;
  size_t &evidence_relation_export_failures;
  PbctopoFrameScratch &scratch;
  PbctopoHullScratch &hull_scratch;
  vector<PbctopoAtom> &base_atoms;
  vector<PbctopoAtom> &projection_atoms;
  vector<PbctopoAtom> &component_atoms;
  vector<PbctopoLocalEdge> &local_edges;
  vector<PbctopoLocalEdge> &component_scoring_edges;
  vector<PbctopoLocalEdge> &projection_edges;
  vector<PbctopoContinuityPair> &projection_continuity_pairs;
  vector<size_t> &projection_anchor_indices;
  vector<std::uint8_t> &local_graph_atom_coverage;
  unordered_set<uint64_t> &local_edge_set;
  unordered_set<uint64_t> &steric_exclusion_set;
  unordered_set<uint64_t> &projection_edge_set;
  vector<vector<size_t>> &components;
  vector<PbctopoShift> &shifts;
  vector<PbctopoSeed3D> &volume_seeds;
  vector<Vec3> &warm_points;
  vector<Vec3> &coarse_warm_points;
  vector<Vec3> &placed_points;
  vector<Vec3> &best_safe_points;
  vector<Vec3> &verify_points;
  vector<Vec3> &component_points;
  vector<PbctopoCutCandidate> &coarse_candidates;
  vector<PbctopoCoarseSeedOption> &coarse_seed_options;
  vector<PbctopoCutCandidate> &coarse_results;
  vector<size_t> &projection_index;
  vector<size_t> &atom_component_index;
  titan_pbctopo::PbctopoVirtualDiscreteSearchWorkspace
      &virtual_search_workspace;
  PbctopoVirtualDiscreteSearchPlan &virtual_search_plan;
  vector<PbctopoHullSpan> &hull_owner_spans;
  vector<Vec3> &hull_rescue_points;
  bool &projection_clash_skip_active;
  string &llps_local_unit_source;

  size_t frame_idx;
  size_t frame_fatal_lattice_query_failures{};
  optional<PbctopoFrameLatticeQueryAuditScope> lattice_query_audit_scope;
  PbctopoWarmState frame_start_warm_state{};
  bool kinematic_seed_history_ready{};
  PbctopoCutCandidate kinematic_seed_previous{};
  PbctopoCutCandidate kinematic_seed_latest{};
  PbctopoFrameReport frame_report{};
  PbctopoSelectedEvidenceAudit selected_evidence_audit{};
  clock::time_point frame_report_started{};
  double frame_graph_s_before{};
  double frame_virtual_s_before{};
  double frame_component_s_before{};
  bool skip_projection_candidate_clash{};
  clock::time_point stage_started{};
  titan_pbctopo::PbctopoFrameBindingObservation frame_binding{};
  optional<PbctopoLatticeMetric> frame_lattice;
  bool heavy_search_omits_atoms{};
  PbctopoTranslationCache translation_cache{};
  bool has_ligand_frame{};
  bool rec_use_topology{};
  bool lig_use_topology{};
  bool rec_use_metadata{};
  bool lig_use_metadata{};
  bool rec_use_geometry{};
  bool lig_use_geometry{};
  bool any_topology_graph{};
  bool any_metadata_graph{};
  bool any_geometry_graph{};
  size_t projected_static_edges{};
  clock::time_point static_graph_done{};
  size_t hard_connectivity_edge_count{};
  size_t topology_edge_count{};
  size_t metadata_edge_count{};
  size_t soft_geometry_edge_count{};
  size_t hard_graph_atoms_covered{};
  size_t topology_bond_atoms_covered{};
  size_t metadata_only_atoms{};
  vector<pair<int, int>> local_hard_required_owner_pairs{};
  vector<pair<int, int>> projection_hard_required_owner_pairs{};
  PbctopoAnchorGraphContext base_anchor_context{};
  PbctopoAnchorGraphContext projection_anchor_context{};
  bool use_anchor_order{};
  bool use_anchor_contact{};
  vector<pair<size_t, size_t>> explicit_evidence_component_edges{};
  PbctopoIdentifiability frame_identifiability{};
  PbctopoHardGraphSource frame_hard_graph_source{};
  PbctopoComponentLayoutObservation layout_observation{};
  titan_pbctopo::PbctopoSpatialLiftResult frame_spatial_lift{};
  PbctopoComponentLayoutSignature component_layout{};
  size_t raw_component_count{};
  PbctopoComponentMode component_mode{};
  bool can_use_component_logic{};
  bool same_component_warm_layout{};
  int component_context_shell_radius{};
  bool component_context_ready{};
  optional<PbctopoLatticeMetric> certification_lattice;
  optional<titan_pbctopo::PbctopoEdgeImageCache> edge_image_cache;
  int component_certificate_domain_complete_cache{};
  size_t report_soft_valid_candidates{};
  size_t report_nonclashing_candidates{};
  vector<PbctopoCutCandidate> report_certified_candidates{};
  vector<PbctopoCutCandidate> report_rejected_candidates{};
  vector<titan_pbctopo::PbctopoCandidateEvidenceAuditSnapshot>
      evidence_relation_audit_snapshots{};
  string evidence_relation_cache_error{};
  bool evidence_relation_frame_written{};
  bool evidence_relation_export_enabled{};
  static constexpr size_t kProjectionTopSeeds = 24;
  static constexpr size_t kProjectionKeepCandidates = 3;
  vector<Vec3> prefetched_safe_points{};
  PbctopoCutCandidate prefetched_safe_candidate{};
  PbctopoFrameMode prefetched_safe_mode{};
  PbctopoCutCandidate best_verified_virtual_candidate{};
  PbctopoCutCandidate best_virtual_geometry_reference{};
  PbctopoCutCandidate best_coarse_virtual_geometry_seed{};
  vector<Vec3> split_rescue_points{};
  PbctopoOwnerPairSplitDetail split_rescue_detail{};
  string split_rescue_source{};
  bool split_rescue_available{};
  bool primary_component_graph_used{};
  vector<PbctopoVirtualDiscreteSeedHint> virtual_seed_hints{};
  PbctopoCutCandidate best_candidate{};
  PbctopoCutCandidate best_safe_candidate{};
  bool best_safe_certification_rejected{};
  size_t safe_candidates{};
  bool frame_anchor_ambiguity_counted{};
  static constexpr size_t kInitialOutputCertificationBand = 12;
  static constexpr size_t kMaxOutputCertificationBand = 48;
  static constexpr size_t kMaxCompleteOutputCertificationBand = 4096;
  size_t output_certification_band_limit{};
  bool complete_feasible_band_retained{};
  vector<PbctopoOutputCandidateSnapshot> output_candidate_band{};
  bool has_kinematic_hint{};
  size_t kinematic_states_evaluated{};
  size_t requested_threads{};
  size_t executor_capacity{};
  size_t coarse_threads{};
  atomic<size_t> coarse_projection_skip_evals{0};
  atomic<size_t> coarse_worker_lattice_queries{0};
  atomic<size_t> coarse_worker_lattice_failures{0};
  atomic<size_t> coarse_worker_lattice_ambiguities{0};
  atomic<size_t> coarse_worker_lattice_budget_failures{0};
  size_t worker_lattice_queries{};
  size_t worker_lattice_failures{};
  size_t worker_lattice_ambiguities{};
  size_t worker_lattice_budget_failures{};
  size_t refine_limit{};
  bool component_search_used{};
  bool output_candidate_committed{};
  size_t output_certificate_rejects{};
  size_t output_all_atom_certificate_rejects{};
  vector<Vec3> output_certification_points{};
  bool certify_complete_uncertainty_band{};
  bool lattice_query_failed{};
  PbctopoFailureReason fallback_reason{};
  string fallback_message{};
};

} // namespace pbctopo_solve
