#ifndef TITAN_HOLOLIFT_PHASE1_H
#define TITAN_HOLOLIFT_PHASE1_H

#include "lattice_transport.h"
#include "types.h"

#include <array>
#include <cstddef>
#include <expected>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace titan_hololift {

inline constexpr std::uint32_t HOLOLIFT_PHASE1_API_VERSION = 8;
inline constexpr std::uint32_t HOLOLIFT_PHASE1_AUDIT_SEAL_VERSION = 1;

enum class HoloLiftPhase1FrameUse : std::uint8_t {
  TemporalObservation,
  ObservationGapInvalidSpatialLift,
  ObservationGapAmbiguousInternalLift,
  ObservationGapCycleResidual,
  ObservationGapNoCertifiedBand,
  ObservationGapContradictedEvidence,
};

enum class HoloLiftPhase1TransitionGap : std::uint8_t {
  None,
  AmbiguousBasisTransport,
  BasisDiscontinuity,
  BasisMismatch,
};

[[nodiscard]] constexpr std::string_view
hololift_phase1_transition_gap_name(
    HoloLiftPhase1TransitionGap value) noexcept {
  switch (value) {
  case HoloLiftPhase1TransitionGap::None:
    return "none";
  case HoloLiftPhase1TransitionGap::AmbiguousBasisTransport:
    return "ambiguous_basis_transport";
  case HoloLiftPhase1TransitionGap::BasisDiscontinuity:
    return "basis_discontinuity";
  case HoloLiftPhase1TransitionGap::BasisMismatch:
    return "basis_mismatch";
  }
  return "basis_mismatch";
}

[[nodiscard]] constexpr HoloLiftPhase1FrameUse
hololift_phase1_frame_use(const HoloLiftFrameRecord &frame) noexcept {
  if (frame.provenance.status == HoloLiftFrameStatus::Contradicted ||
      frame.provenance.evidence_state == HoloLiftEvidenceState::Contradicted) {
    return HoloLiftPhase1FrameUse::ObservationGapContradictedEvidence;
  }
  if (!frame.spatial_lift_valid || frame.spatial_lift_policy_version == 0 ||
      frame.spatial_lift_identity.empty()) {
    return HoloLiftPhase1FrameUse::ObservationGapInvalidSpatialLift;
  }
  if (frame.spatial_lift_ambiguous_hard_edges != 0) {
    return HoloLiftPhase1FrameUse::ObservationGapAmbiguousInternalLift;
  }
  if (frame.spatial_lift_cycle_residuals != 0)
    return HoloLiftPhase1FrameUse::ObservationGapCycleResidual;
  const bool has_certified_band =
      (frame.provenance.status == HoloLiftFrameStatus::Certified ||
       frame.provenance.status == HoloLiftFrameStatus::Rescued ||
       frame.provenance.status == HoloLiftFrameStatus::WeakObservation) &&
      frame.candidate_set_scope ==
          HoloLiftCandidateSetScope::RetainedCertifiedBand &&
      frame.candidates.count != 0;
  if (!has_certified_band)
    return HoloLiftPhase1FrameUse::ObservationGapNoCertifiedBand;
  return HoloLiftPhase1FrameUse::TemporalObservation;
}

[[nodiscard]] constexpr bool
hololift_phase1_uses_temporal_observation(
    const HoloLiftFrameRecord &frame) noexcept {
  return hololift_phase1_frame_use(frame) ==
         HoloLiftPhase1FrameUse::TemporalObservation;
}

[[nodiscard]] constexpr std::string_view
hololift_phase1_frame_use_name(HoloLiftPhase1FrameUse value) noexcept {
  switch (value) {
  case HoloLiftPhase1FrameUse::TemporalObservation:
    return "temporal_observation";
  case HoloLiftPhase1FrameUse::ObservationGapInvalidSpatialLift:
    return "observation_gap_invalid_spatial_lift";
  case HoloLiftPhase1FrameUse::ObservationGapAmbiguousInternalLift:
    return "observation_gap_ambiguous_internal_lift";
  case HoloLiftPhase1FrameUse::ObservationGapCycleResidual:
    return "observation_gap_cycle_residual";
  case HoloLiftPhase1FrameUse::ObservationGapNoCertifiedBand:
    return "observation_gap_no_certified_band";
  case HoloLiftPhase1FrameUse::ObservationGapContradictedEvidence:
    return "observation_gap_contradicted_evidence";
  }
  return "observation_gap_invalid_spatial_lift";
}

enum class HoloLiftPhase1TimeScaling : std::uint8_t {
  PerFrameDisplacement,
  Diffusive,
  Ballistic,
};

enum class HoloLiftPhase1Loss : std::uint8_t {
  Quadratic,
};

enum class HoloLiftPhase1ComponentWeighting : std::uint8_t {
  AtomCount,
  AtomicMass,
};

class HoloLiftPhase1PreparedFrame;
class HoloLiftPhase1PreparationContext;
struct HoloLiftPhase1FrameSelection;
struct HoloLiftPhase1Result;
struct HoloLiftPhase1TrajectoryFrameView;

using HoloLiftPhase1AuditDigest = std::array<std::byte, 32>;

// A result can enter the scientific artifact writer only through this
// move-only seal. The all-atom audit owns construction; callers can inspect
// but cannot replace the audited result or its trajectory digest.
class HoloLiftPhase1AuditedResult {
public:
  HoloLiftPhase1AuditedResult(HoloLiftPhase1AuditedResult &&) noexcept;
  HoloLiftPhase1AuditedResult &
  operator=(HoloLiftPhase1AuditedResult &&) noexcept;
  ~HoloLiftPhase1AuditedResult();

  HoloLiftPhase1AuditedResult(const HoloLiftPhase1AuditedResult &) = delete;
  HoloLiftPhase1AuditedResult &
  operator=(const HoloLiftPhase1AuditedResult &) = delete;

  [[nodiscard]] bool valid() const noexcept {
    return static_cast<bool>(result_);
  }
  // Returns null after this move-only seal has been moved from.
  [[nodiscard]] const HoloLiftPhase1Result *result_if_valid() const noexcept;
  [[nodiscard]] std::span<const std::byte, 32>
  trajectory_audit_digest_sha256() const noexcept;
  [[nodiscard]] constexpr std::uint32_t audit_seal_version() const noexcept {
    return HOLOLIFT_PHASE1_AUDIT_SEAL_VERSION;
  }

private:
  HoloLiftPhase1AuditedResult(
      std::unique_ptr<HoloLiftPhase1Result> result,
      HoloLiftPhase1AuditDigest trajectory_audit_digest) noexcept;

  std::unique_ptr<HoloLiftPhase1Result> result_;
  HoloLiftPhase1AuditDigest trajectory_audit_digest_{};

  friend std::expected<HoloLiftPhase1AuditedResult, std::string>
  audit_hololift_phase1_all_atom_reconstruction(
      HoloLiftPhase1PreparationContext &, const HoloLiftObservationStore &,
      std::span<const HoloLiftPhase1PreparedFrame>, HoloLiftPhase1Result,
      std::span<const HoloLiftPhase1TrajectoryFrameView>);
};

class HoloLiftPhase1PreparationContext {
public:
  struct Impl;

  // Epoch tables are immutable after construction. Per-frame atom-order
  // mappings use local scratch, so one context may be shared by readers.

  HoloLiftPhase1PreparationContext(HoloLiftPhase1PreparationContext &&) noexcept;
  HoloLiftPhase1PreparationContext &
  operator=(HoloLiftPhase1PreparationContext &&) noexcept;
  ~HoloLiftPhase1PreparationContext();

  HoloLiftPhase1PreparationContext(
      const HoloLiftPhase1PreparationContext &) = delete;
  HoloLiftPhase1PreparationContext &
  operator=(const HoloLiftPhase1PreparationContext &) = delete;

private:
  explicit HoloLiftPhase1PreparationContext(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;

  friend std::expected<HoloLiftPhase1PreparationContext, std::string>
  make_hololift_phase1_preparation_context(const HoloLiftObservationStore &);
  friend class HoloLiftPhase1PreparedFrame;
  friend std::expected<HoloLiftPhase1PreparedFrame, std::string>
  prepare_hololift_phase1_frame(
      HoloLiftPhase1PreparationContext &, const HoloLiftObservationStore &,
      std::size_t, std::size_t, std::span<const double, 9>,
      std::span<const HoloLiftSourceAtomKey>,
      std::span<const std::array<double, 3>>);
  friend std::expected<std::vector<std::array<double, 3>>, std::string>
  reconstruct_hololift_phase1_frame(
      HoloLiftPhase1PreparationContext &, const HoloLiftObservationStore &,
      const HoloLiftPhase1PreparedFrame &,
      const HoloLiftPhase1FrameSelection &,
      std::span<const HoloLiftSourceAtomKey>,
      std::span<const std::array<double, 3>>);
  friend std::expected<HoloLiftPhase1AuditedResult, std::string>
  audit_hololift_phase1_all_atom_reconstruction(
      HoloLiftPhase1PreparationContext &, const HoloLiftObservationStore &,
      std::span<const HoloLiftPhase1PreparedFrame>, HoloLiftPhase1Result,
      std::span<const HoloLiftPhase1TrajectoryFrameView>);
};

std::expected<HoloLiftPhase1PreparationContext, std::string>
make_hololift_phase1_preparation_context(
    const HoloLiftObservationStore &store);

class HoloLiftPhase1PreparedFrame {
public:
  [[nodiscard]] std::size_t frame_index() const noexcept {
    return frame_index_;
  }
  [[nodiscard]] std::size_t trajectory_frame_index() const noexcept {
    return trajectory_frame_index_;
  }
  [[nodiscard]] double time_ps() const noexcept { return time_ps_; }
  [[nodiscard]] HoloLiftHash128 topology_epoch_identity() const noexcept {
    return topology_epoch_identity_;
  }
  [[nodiscard]] HoloLiftHash128 wrapped_coordinate_hash() const noexcept {
    return wrapped_coordinate_hash_;
  }
  [[nodiscard]] HoloLiftHash128 spatial_lift_identity() const noexcept {
    return spatial_lift_identity_;
  }
  [[nodiscard]] std::span<const double, 9> box_matrix() const noexcept {
    return box_matrix_;
  }
  [[nodiscard]] std::span<const double, 6> physical_gram() const noexcept {
    return physical_gram_;
  }
  [[nodiscard]] std::span<const std::array<double, 3>>
  component_fractional_representatives() const noexcept {
    return component_fractional_representatives_;
  }
  [[nodiscard]] std::span<const double> component_weights() const noexcept {
    return component_weights_;
  }
  [[nodiscard]] HoloLiftPhase1ComponentWeighting
  component_weighting() const noexcept {
    return component_weighting_;
  }

private:
  friend std::expected<HoloLiftPhase1PreparedFrame, std::string>
  prepare_hololift_phase1_frame(
      HoloLiftPhase1PreparationContext &, const HoloLiftObservationStore &,
      std::size_t, std::size_t, std::span<const double, 9>,
      std::span<const HoloLiftSourceAtomKey>,
      std::span<const std::array<double, 3>>);

  HoloLiftPhase1PreparedFrame(
      std::size_t frame_index, std::size_t trajectory_frame_index,
      double time_ps, HoloLiftHash128 topology_epoch_identity,
      HoloLiftHash128 wrapped_coordinate_hash,
      HoloLiftHash128 spatial_lift_identity,
      std::array<double, 9> box_matrix,
      std::array<double, 6> physical_gram,
      std::vector<std::array<double, 3>> component_representatives,
      std::vector<double> component_weights,
      HoloLiftPhase1ComponentWeighting component_weighting) noexcept
      : frame_index_(frame_index),
        trajectory_frame_index_(trajectory_frame_index), time_ps_(time_ps),
        topology_epoch_identity_(topology_epoch_identity),
        wrapped_coordinate_hash_(wrapped_coordinate_hash),
        spatial_lift_identity_(spatial_lift_identity),
        box_matrix_(box_matrix), physical_gram_(physical_gram),
        component_fractional_representatives_(
            std::move(component_representatives)),
        component_weights_(std::move(component_weights)),
        component_weighting_(component_weighting) {}

  std::size_t frame_index_ = HOLOLIFT_NO_INDEX;
  std::size_t trajectory_frame_index_ = HOLOLIFT_NO_INDEX;
  double time_ps_ = 0.0;
  HoloLiftHash128 topology_epoch_identity_;
  HoloLiftHash128 wrapped_coordinate_hash_;
  HoloLiftHash128 spatial_lift_identity_;
  std::array<double, 9> box_matrix_{};
  std::array<double, 6> physical_gram_{};
  std::vector<std::array<double, 3>> component_fractional_representatives_;
  std::vector<double> component_weights_;
  HoloLiftPhase1ComponentWeighting component_weighting_ =
      HoloLiftPhase1ComponentWeighting::AtomCount;
};

using HoloLiftPhase1VerifiedFrame = HoloLiftPhase1PreparedFrame;

std::expected<HoloLiftPhase1PreparedFrame, std::string>
prepare_hololift_phase1_frame(
    HoloLiftPhase1PreparationContext &context,
    const HoloLiftObservationStore &store, std::size_t frame_index,
    std::size_t trajectory_frame_index,
    std::span<const double, 9> box_matrix,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates);

std::expected<HoloLiftPhase1VerifiedFrame, std::string>
verify_hololift_phase1_frame(
    const HoloLiftObservationStore &store, std::size_t frame_index,
    std::size_t trajectory_frame_index,
    std::span<const double, 9> box_matrix,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates);

struct HoloLiftPhase1Config {
  double ordinal_rank_weight = 1.0;
  // false: E_t = w_r (rank-1)/(K-1), the linear form of Eq. 1.
  // true:  E_t = w_r [rank != 1], a saturated form that charges every seat
  //        other than the framewise choice the same amount, so a path is
  //        anchored by how often it is the framewise choice, not by how far
  //        down the list it sits when it is not.
  bool ordinal_rank_saturated = false;
  // At a frame whose band holds no evidence-compatible candidate (K_eff = 0)
  // the supplied ordering is a tie-break, not a measurement.  With this set,
  // E_t is zero for every assignment of such a frame, so only the transition
  // term acts there, and the frame is reported as evidence-absent.  This is
  // what the carry band policy sets; the strict policy leaves it off.
  bool evidence_absent_zero_rank = false;
  // Diagnostic admission is opt-in and does not promote provider evidence.
  bool allow_policy_admitted_candidates = false;
  // At exactly one provider-hard-feasible evidence-compatible candidate,
  // charge every other admitted candidate this additional emission cost.
  double unique_evidence_mismatch_weight = 0.0;
  // At one or more provider-hard-feasible evidence-compatible candidates,
  // charge every admitted candidate that is not one of them this additional
  // emission cost. Unlike the unique term it also acts where several
  // candidates carry evidence (a component in contact with two images of its
  // partner), so a large value makes the path follow the evidence with the
  // fewest image changes, and a value w follows an evidence episode only when
  // it outlasts about (cost of leaving and returning) / w frames.  The
  // default, 1000, is far above one cell jump (~50 at unit transition
  // weight): the path leaves the supported images only where no admitted
  // candidate carries evidence.  0 restores the objective of the H1-H9
  // analyses.  Enters the audit digest only when nonzero.
  double supported_evidence_mismatch_weight = 1000.0;
  double physical_transition_weight = 1.0;
  double temporal_scale = 1.0;
  double variance_floor_A2 = 1.0e-6;
  double bounded_anchor_competitor_penalty = 4.0;
  double relative_cost_tolerance = 1.0e-12;
  double basis_transport_max_relative_mismatch = 0.35;
  double basis_transport_ambiguity_tolerance = 1.0e-10;
  double basis_transport_nonidentity_max_relative_mismatch = 0.02;
  double basis_transport_minimum_improvement_ratio = 4.0;
  HoloLiftPhase1TimeScaling time_scaling =
      HoloLiftPhase1TimeScaling::Diffusive;
  HoloLiftPhase1Loss loss = HoloLiftPhase1Loss::Quadratic;
  HoloLiftLatticeBasisPolicy lattice_basis_policy =
      HoloLiftLatticeBasisPolicy::RequireBasisContinuous;
  std::size_t dp_worker_count = 1;
  bool pin_strong_bounded_anchors = false;
};

struct HoloLiftPhase1FrameSelection {
  HoloLiftPhase1FrameUse use =
      HoloLiftPhase1FrameUse::ObservationGapInvalidSpatialLift;
  HoloLiftPhase1TransitionGap preceding_transition_gap =
      HoloLiftPhase1TransitionGap::None;
  std::size_t segment_index = HOLOLIFT_NO_INDEX;
  std::size_t candidate_index = HOLOLIFT_NO_INDEX;
  double emission_cost = 0.0;
  double transition_cost = 0.0;
  double cumulative_segment_cost = 0.0;
  HoloLiftLatticeImage selected_global_gauge_increment;
  HoloLiftLatticeImage cumulative_global_gauge;
  HoloLiftLatticeImage cumulative_global_gauge_input_basis;
  std::array<std::int64_t, 9> lattice_basis_transport_to_segment_reference{
      1, 0, 0, 0, 1, 0, 0, 0, 1};
  double component_displacement_rms_A = 0.0;
  double component_displacement_max_A = 0.0;
  double component_injectivity_radius_lower_bound_A = 0.0;
  double component_injectivity_margin_A = 0.0;
  std::size_t component_nearest_image_violations = 0;
  std::size_t component_injectivity_uncertified = 0;
  std::size_t atoms_audited = 0;
  std::size_t atom_temporal_transitions_audited = 0;
  std::size_t atom_temporal_nearest_image_violations = 0;
  std::size_t atom_temporal_injectivity_uncertified = 0;
  std::size_t independent_temporal_image_ambiguities = 0;
  std::size_t independent_space_time_edges_audited = 0;
  std::size_t independent_space_time_curvature_residuals = 0;
  double lattice_basis_transport_relative_mismatch = 0.0;
  double lattice_basis_transport_identity_relative_mismatch = 0.0;
  double lattice_basis_transport_improvement_ratio = 1.0;
  double maximum_component_representative_error_A = 0.0;
  double maximum_atom_peculiar_displacement_A = 0.0;
  bool global_gauge_increment_ambiguous = false;
  bool lattice_basis_transport_applied = false;
  bool lattice_basis_transport_ambiguous = false;
  bool lattice_basis_transport_identity_fast_path = false;
  bool lattice_basis_transport_local_search_completed = false;
  bool lattice_basis_transport_accepted_under_policy = false;
  bool lattice_basis_transport_explicit_remap_provenance_available = false;
  bool lattice_basis_transport_provenance_sufficient = false;
  bool quadratic_direct_residual_fallback = false;
  bool pinned_strong_bounded_anchor = false;
  bool changed_from_framewise = false;
  bool selected_candidate_provider_unsupported = false;
  bool preceding_transition_provider_unsupported = false;
};

struct HoloLiftPhase1TrajectoryFrameView {
  std::size_t trajectory_frame_index = HOLOLIFT_NO_INDEX;
  std::span<const HoloLiftSourceAtomKey> atom_order;
  std::span<const std::array<double, 3>> wrapped_coordinates;
};

struct HoloLiftPhase1ReconstructionAudit {
  std::size_t temporal_frames_audited = 0;
  std::size_t atoms_audited = 0;
  std::size_t atom_temporal_transitions_audited = 0;
  std::size_t atom_temporal_nearest_image_violations = 0;
  std::size_t atom_temporal_injectivity_uncertified = 0;
  std::size_t independent_temporal_image_ambiguities = 0;
  std::size_t independent_space_time_edges_audited = 0;
  std::size_t independent_space_time_curvature_residuals = 0;
  double maximum_component_representative_error_A = 0.0;
  double maximum_atom_peculiar_displacement_A = 0.0;
  bool selected_frames_complete = false;
  bool atom_temporal_injectivity_audited = false;
  bool independent_space_time_cochain_audited = false;
};

struct HoloLiftPhase1Segment {
  std::size_t frame_begin = 0;
  std::size_t frame_count = 0;
  std::uint32_t topology_epoch_index = HOLOLIFT_NO_DENSE_INDEX;
  double objective = 0.0;
  double second_best_objective = std::numeric_limits<double>::infinity();
  double absolute_path_gap = std::numeric_limits<double>::infinity();
  double relative_path_gap = std::numeric_limits<double>::infinity();
  std::size_t optimal_path_count_capped = 0;
  std::size_t temporal_transition_count = 0;
  std::size_t provider_unsupported_selected_candidate_count = 0;
  std::size_t provider_unsupported_transition_count = 0;
  std::size_t ambiguous_gauge_transition_count = 0;
  std::size_t component_nearest_image_violations = 0;
  std::size_t component_injectivity_uncertified = 0;
  std::size_t gauge_replay_arithmetic_residuals = 0;
  std::size_t temporal_frames_audited = 0;
  std::size_t atoms_audited = 0;
  std::size_t atom_temporal_transitions_audited = 0;
  std::size_t atom_temporal_nearest_image_violations = 0;
  std::size_t atom_temporal_injectivity_uncertified = 0;
  std::size_t independent_temporal_image_ambiguities = 0;
  std::size_t independent_space_time_edges_audited = 0;
  std::size_t independent_space_time_curvature_residuals = 0;
  std::size_t lattice_basis_transports_applied = 0;
  std::size_t quadratic_direct_residual_fallbacks = 0;
  std::size_t pretransformed_candidate_component_images = 0;
  // Runtime-only A2 telemetry; not audit-sealed, replay-validated, or serialized.
  std::size_t peak_live_pretransformed_candidate_component_images = 0;
  std::size_t
      peak_live_pretransformed_candidate_component_image_bytes_estimate = 0;
  std::size_t rolling_pretransformed_candidate_layer_builds = 0;
  // Runtime-only A3 telemetry; not audit-sealed, replay-validated, or
  // serialized.
  std::size_t dp_parallel_layers = 0;
  std::size_t dp_serial_layers = 0;
  std::size_t dp_worker_pool_creations = 0;
  std::size_t dp_worker_pool_creation_failures = 0;
  std::size_t dp_max_active_workers = 0;
  std::size_t peak_layer_state_count = 0;
  std::size_t compact_backpointer_bytes_estimate = 0;
  double maximum_lattice_basis_transport_relative_mismatch = 0.0;
  double maximum_component_representative_error_A = 0.0;
  double maximum_atom_peculiar_displacement_A = 0.0;
  HoloLiftPhase1ComponentWeighting component_weighting =
      HoloLiftPhase1ComponentWeighting::AtomCount;
  bool exact_on_retained_graph = true;
  bool optimal_path_proven_on_complete_bounded_domain = false;
  bool unique_on_configured_retained_state_graph = false;
  bool retained_candidate_bands_complete = false;
  bool bounded_domain_only = false;
  bool anchors_hard_pinned = false;
  bool all_global_gauge_steps_solved = false;
  bool lattice_basis_transport_accepted_under_policy = false;
  bool lattice_basis_transport_provenance_sufficient = false;
  bool local_transport_search_completed = false;
  bool explicit_remap_provenance_available = false;
  bool all_atom_reconstruction_audited = false;
  bool coordinates_materialized_and_replay_consistent = false;
  bool temporal_lift_locally_certified = false;
  bool component_representative_injectivity_audit_passed = false;
  bool gauge_replay_arithmetic_closed = false;
  bool atom_temporal_injectivity_audited = false;
  bool atom_temporal_injectivity_audit_passed = false;
  bool independent_space_time_cochain_audited = false;
  bool independent_space_time_cochain_closed = false;
  bool selected_transition_time_reversal_audited = false;
  bool selected_transition_time_reversal_consistent = false;
};

struct HoloLiftPhase1Certificate {
  bool all_frames_trajectory_bound = false;
  bool all_spatial_lifts_replayed = false;
  bool all_selected_candidates_hard_feasible = false;
  bool all_global_gauge_steps_solved = false;
  bool lattice_basis_transport_accepted_under_policy = false;
  bool lattice_basis_transport_provenance_sufficient = false;
  bool local_transport_search_completed = false;
  bool explicit_remap_provenance_available = false;
  bool component_representative_injectivity_audit_passed = false;
  bool gauge_replay_arithmetic_closed = false;
  bool atom_temporal_injectivity_audited = false;
  bool atom_temporal_injectivity_audit_passed = false;
  bool independent_space_time_cochain_audited = false;
  bool independent_space_time_cochain_closed = false;
  bool all_atom_reconstruction_audited = false;
  bool coordinates_materialized_and_replay_consistent = false;
  bool temporal_lift_locally_certified = false;
  // This is an eligibility gate for bounded external trajectory claims, not an
  // oracle verdict.  It is true only when the conditional local certificate,
  // source coverage, retained-band completeness, and complete bounded-domain
  // optimality all hold simultaneously.
  bool complete_domain_trajectory_claim_eligible = false;
  bool source_trajectory_coverage_complete = false;
  bool retained_candidate_bands_complete = false;
  bool optimal_path_exact_on_retained_graph = false;
  bool optimal_path_proven_on_complete_bounded_domain = false;
  bool unique_on_configured_retained_state_graph = false;
  bool selected_transition_time_reversal_audited = false;
  bool selected_transition_time_reversal_consistent = false;
  bool robust_loss_exact = false;
};

struct HoloLiftPhase1Result {
  std::uint32_t api_version = HOLOLIFT_PHASE1_API_VERSION;
  HoloLiftPhase1Config config;
  std::vector<HoloLiftPhase1FrameSelection> frames;
  std::vector<HoloLiftPhase1Segment> segments;
  double objective = 0.0;
  std::size_t temporal_observation_frames = 0;
  std::size_t observation_gap_frames = 0;
  std::size_t changed_from_framewise_frames = 0;
  std::size_t basis_transport_transition_gaps = 0;
  std::size_t ambiguous_basis_transport_transition_gaps = 0;
  std::size_t basis_discontinuity_transition_gaps = 0;
  std::size_t basis_mismatch_transition_gaps = 0;
  std::size_t quadratic_direct_residual_fallbacks = 0;
  std::size_t pretransformed_candidate_component_images = 0;
  // Runtime-only A2 telemetry; not audit-sealed, replay-validated, or serialized.
  std::size_t peak_live_pretransformed_candidate_component_images = 0;
  std::size_t
      peak_live_pretransformed_candidate_component_image_bytes_estimate = 0;
  std::size_t rolling_pretransformed_candidate_layer_builds = 0;
  // Runtime-only A3 telemetry; not audit-sealed, replay-validated, or
  // serialized.
  std::size_t dp_parallel_layers = 0;
  std::size_t dp_serial_layers = 0;
  std::size_t dp_worker_pool_creations = 0;
  std::size_t dp_worker_pool_creation_failures = 0;
  std::size_t dp_max_active_workers = 0;
  std::size_t source_frame_count = 0;
  std::size_t imported_observation_frame_count = 0;
  bool imported_observation_coverage_complete = false;
  bool source_trajectory_coverage_known = false;
  bool source_trajectory_coverage_complete = false;
  bool temporal_transition_coverage_complete = false;
  bool temporal_coverage_complete = false;
  bool all_segment_objectives_unique = false;
  bool all_observation_bands_complete = false;
  bool exact_on_retained_graph = true;
  HoloLiftPhase1ReconstructionAudit reconstruction_audit;
  HoloLiftPhase1Certificate certificate;
};

std::expected<HoloLiftPhase1Result, std::string>
solve_hololift_phase1_temporal_path(
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    const HoloLiftPhase1Config &config = {});

std::expected<std::vector<std::array<double, 3>>, std::string>
reconstruct_hololift_phase1_frame(
    HoloLiftPhase1PreparationContext &context,
    const HoloLiftObservationStore &store,
    const HoloLiftPhase1PreparedFrame &prepared_frame,
    const HoloLiftPhase1FrameSelection &selection,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates);

// Replays the selected all-atom lift without retaining trajectory coordinates
// in PreparedFrame. Success returns the only type accepted by the scientific
// artifact writer; a mutable solver result cannot self-assert audit status.
std::expected<HoloLiftPhase1AuditedResult, std::string>
audit_hololift_phase1_all_atom_reconstruction(
    HoloLiftPhase1PreparationContext &context,
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    HoloLiftPhase1Result result,
    std::span<const HoloLiftPhase1TrajectoryFrameView> trajectory_frames);

// Recomputes the complete frame -> segment -> global audit hierarchy and the
// deterministic trajectory/assignment digest held by the sealed result.
std::expected<void, std::string> validate_hololift_phase1_audited_result(
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    const HoloLiftPhase1AuditedResult &audited_result);

} // namespace titan_hololift

#endif
