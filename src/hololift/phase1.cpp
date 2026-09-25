#include "phase1.h"

#include "../pbctopo/lattice_math.h"
#include "../titan_sha256.h"
#include "frame_binding.h"
#include "spatial_lift.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace titan_hololift {

#ifdef TITAN_HOLOLIFT_PHASE1_TEST_HOOKS
namespace {
std::atomic<bool> g_force_phase1_dp_pool_creation_failure_for_test{false};
std::atomic<bool> g_force_phase1_dp_pool_creation_bad_alloc_for_test{false};
std::atomic<bool> g_force_phase1_dp_run_setup_exception_for_test{false};
std::atomic<bool> g_force_phase1_dp_worker_exception_for_test{false};
}

namespace testing {
void force_phase1_dp_pool_creation_failure(bool enabled) noexcept {
  g_force_phase1_dp_pool_creation_failure_for_test.store(
      enabled, std::memory_order_relaxed);
}

void force_phase1_dp_pool_creation_bad_alloc(bool enabled) noexcept {
  g_force_phase1_dp_pool_creation_bad_alloc_for_test.store(
      enabled, std::memory_order_relaxed);
}

void force_phase1_dp_run_setup_exception(bool enabled) noexcept {
  g_force_phase1_dp_run_setup_exception_for_test.store(
      enabled, std::memory_order_relaxed);
}

void force_phase1_dp_worker_exception(bool enabled) noexcept {
  g_force_phase1_dp_worker_exception_for_test.store(
      enabled, std::memory_order_relaxed);
}
} // namespace testing
#endif

HoloLiftPhase1AuditedResult::HoloLiftPhase1AuditedResult(
    std::unique_ptr<HoloLiftPhase1Result> result,
    HoloLiftPhase1AuditDigest trajectory_audit_digest) noexcept
    : result_(std::move(result)),
      trajectory_audit_digest_(trajectory_audit_digest) {}

HoloLiftPhase1AuditedResult::HoloLiftPhase1AuditedResult(
    HoloLiftPhase1AuditedResult &&) noexcept = default;

HoloLiftPhase1AuditedResult &HoloLiftPhase1AuditedResult::operator=(
    HoloLiftPhase1AuditedResult &&) noexcept = default;

HoloLiftPhase1AuditedResult::~HoloLiftPhase1AuditedResult() = default;

const HoloLiftPhase1Result *
HoloLiftPhase1AuditedResult::result_if_valid() const noexcept {
  return result_.get();
}

std::span<const std::byte, 32>
HoloLiftPhase1AuditedResult::trajectory_audit_digest_sha256() const noexcept {
  return trajectory_audit_digest_;
}

namespace {

constexpr std::uint64_t kPhase1AuditSealHashDomain =
    0x484c41315345414cULL; // "HLA1SEAL"

class Phase1AuditSealHash {
public:
  Phase1AuditSealHash() noexcept {
    hash_.word(kPhase1AuditSealHashDomain);
    hash_.word(HOLOLIFT_PHASE1_AUDIT_SEAL_VERSION);
  }

  void boolean(bool value) noexcept { hash_.byte(value ? 1U : 0U); }
  void count(std::size_t value) noexcept {
    hash_.word(static_cast<std::uint64_t>(value));
  }
  void word(std::uint64_t value) noexcept { hash_.word(value); }
  void real(double value) noexcept { hash_.real(value); }
  void identity(HoloLiftHash128 value) noexcept {
    hash_.word(value.hi);
    hash_.word(value.lo);
  }
  void image(const HoloLiftLatticeImage &value) noexcept {
    hash_.word(static_cast<std::uint64_t>(value.x));
    hash_.word(static_cast<std::uint64_t>(value.y));
    hash_.word(static_cast<std::uint64_t>(value.z));
  }
  [[nodiscard]] HoloLiftPhase1AuditDigest finish() const noexcept {
    return hash_.finish256();
  }

private:
  titan_hash::Sha256Builder hash_;
};

bool audit_count_add(std::size_t &target, std::size_t value) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - target)
    return false;
  target += value;
  return true;
}

bool valid_audit_maximum(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

bool frame_transition_audit_is_zero(
    const HoloLiftPhase1FrameSelection &selection) noexcept {
  return selection.atom_temporal_transitions_audited == 0 &&
         selection.atom_temporal_nearest_image_violations == 0 &&
         selection.atom_temporal_injectivity_uncertified == 0 &&
         selection.independent_temporal_image_ambiguities == 0 &&
         selection.independent_space_time_edges_audited == 0 &&
         selection.independent_space_time_curvature_residuals == 0 &&
         selection.maximum_atom_peculiar_displacement_A == 0.0;
}

std::expected<void, std::string> validate_all_atom_audit_hierarchy(
    const HoloLiftObservationStore &store,
    const HoloLiftPhase1Result &result) {
  if (result.frames.size() != store.frames().size())
    return std::unexpected("Phase 1 audit hierarchy frame count mismatch");

  HoloLiftPhase1ReconstructionAudit expected;
  std::vector<bool> frame_seen(result.frames.size(), false);
  bool expected_atom_audited = true;
  bool expected_cochain_audited = true;
  for (std::size_t segment_index = 0;
       segment_index < result.segments.size(); ++segment_index) {
    const auto &segment = result.segments[segment_index];
    if (segment.frame_begin > result.frames.size() ||
        segment.frame_count > result.frames.size() - segment.frame_begin ||
        segment.topology_epoch_index >= store.topology_epochs().size() ||
        segment.temporal_transition_count !=
            (segment.frame_count == 0 ? 0 : segment.frame_count - 1)) {
      return std::unexpected("Phase 1 audit segment range is invalid");
    }
    const auto &epoch =
        store.topology_epochs()[segment.topology_epoch_index];
    const std::size_t atom_count = epoch.atom_count;
    const std::size_t edge_count = epoch.hard_edges.count;
    std::size_t segment_atoms = 0;
    std::size_t segment_atom_transitions = 0;
    std::size_t segment_nearest_violations = 0;
    std::size_t segment_injectivity_uncertified = 0;
    std::size_t segment_temporal_ambiguities = 0;
    std::size_t segment_edges = 0;
    std::size_t segment_curvature_residuals = 0;
    std::size_t provider_unsupported_candidates = 0;
    std::size_t provider_unsupported_transitions = 0;
    bool previous_provider_unsupported = false;
    double segment_component_maximum = 0.0;
    double segment_atom_maximum = 0.0;
    for (std::size_t offset = 0; offset < segment.frame_count; ++offset) {
      const std::size_t frame_index = segment.frame_begin + offset;
      const auto &selection = result.frames[frame_index];
      const auto &frame = store.frames()[frame_index];
      if (frame_seen[frame_index] ||
          selection.use != HoloLiftPhase1FrameUse::TemporalObservation ||
          selection.segment_index != segment_index ||
          frame.topology_epoch_index != segment.topology_epoch_index ||
          selection.atoms_audited != atom_count ||
          !valid_audit_maximum(
              selection.maximum_component_representative_error_A) ||
          !valid_audit_maximum(
              selection.maximum_atom_peculiar_displacement_A)) {
        return std::unexpected(
            "Phase 1 frame audit does not match its temporal segment");
      }
      frame_seen[frame_index] = true;
      if (selection.candidate_index >= store.candidates().size())
        return std::unexpected("Phase 1 provider-support candidate is invalid");
      const bool provider_unsupported =
          !store.candidates()[selection.candidate_index].evidence.hard_feasible;
      const bool transition_unsupported = offset > 0 &&
          (provider_unsupported || previous_provider_unsupported);
      if (selection.selected_candidate_provider_unsupported != provider_unsupported ||
          selection.preceding_transition_provider_unsupported != transition_unsupported)
        return std::unexpected("Phase 1 provider-support flags disagree with the store");
      provider_unsupported_candidates += provider_unsupported ? 1 : 0;
      provider_unsupported_transitions += transition_unsupported ? 1 : 0;
      previous_provider_unsupported = provider_unsupported;
      const std::size_t expected_atom_transitions =
          offset == 0 ? 0 : atom_count;
      const std::size_t expected_edges = offset == 0 ? 0 : edge_count;
      if (selection.atom_temporal_transitions_audited !=
              expected_atom_transitions ||
          selection.independent_space_time_edges_audited != expected_edges ||
          (offset == 0 && !frame_transition_audit_is_zero(selection))) {
        return std::unexpected(
            "Phase 1 segment start/transition audit is inconsistent");
      }
      if (!audit_count_add(segment_atoms, selection.atoms_audited) ||
          !audit_count_add(segment_atom_transitions,
                           selection.atom_temporal_transitions_audited) ||
          !audit_count_add(segment_nearest_violations,
                           selection.atom_temporal_nearest_image_violations) ||
          !audit_count_add(segment_injectivity_uncertified,
                           selection.atom_temporal_injectivity_uncertified) ||
          !audit_count_add(segment_temporal_ambiguities,
                           selection.independent_temporal_image_ambiguities) ||
          !audit_count_add(segment_edges,
                           selection.independent_space_time_edges_audited) ||
          !audit_count_add(
              segment_curvature_residuals,
              selection.independent_space_time_curvature_residuals)) {
        return std::unexpected("Phase 1 frame audit counter overflow");
      }
      segment_component_maximum = std::max(
          segment_component_maximum,
          selection.maximum_component_representative_error_A);
      segment_atom_maximum = std::max(
          segment_atom_maximum,
          selection.maximum_atom_peculiar_displacement_A);
    }
    const bool segment_atom_audited = segment.temporal_transition_count > 0;
    const bool segment_atom_passed =
        segment_atom_audited && segment_nearest_violations == 0 &&
        segment_injectivity_uncertified == 0 &&
        segment_temporal_ambiguities == 0;
    const bool segment_cochain_closed =
        segment_atom_audited && segment_temporal_ambiguities == 0 &&
        segment_curvature_residuals == 0;
    if (segment.temporal_frames_audited != segment.frame_count ||
        segment.provider_unsupported_selected_candidate_count != provider_unsupported_candidates ||
        segment.provider_unsupported_transition_count != provider_unsupported_transitions ||
        (segment.temporal_lift_locally_certified && provider_unsupported_candidates != 0) ||
        segment.atoms_audited != segment_atoms ||
        segment.atom_temporal_transitions_audited !=
            segment_atom_transitions ||
        segment.atom_temporal_nearest_image_violations !=
            segment_nearest_violations ||
        segment.atom_temporal_injectivity_uncertified !=
            segment_injectivity_uncertified ||
        segment.independent_temporal_image_ambiguities !=
            segment_temporal_ambiguities ||
        segment.independent_space_time_edges_audited != segment_edges ||
        segment.independent_space_time_curvature_residuals !=
            segment_curvature_residuals ||
        segment.maximum_component_representative_error_A !=
            segment_component_maximum ||
        segment.maximum_atom_peculiar_displacement_A !=
            segment_atom_maximum ||
        !segment.all_atom_reconstruction_audited ||
        !segment.coordinates_materialized_and_replay_consistent ||
        segment.atom_temporal_injectivity_audited != segment_atom_audited ||
        segment.atom_temporal_injectivity_audit_passed !=
            segment_atom_passed ||
        segment.independent_space_time_cochain_audited !=
            segment_atom_audited ||
        segment.independent_space_time_cochain_closed !=
            segment_cochain_closed) {
      return std::unexpected(
          "Phase 1 segment audit is not the reduction of its frames");
    }
    if (!audit_count_add(expected.temporal_frames_audited,
                         segment.temporal_frames_audited) ||
        !audit_count_add(expected.atoms_audited, segment.atoms_audited) ||
        !audit_count_add(expected.atom_temporal_transitions_audited,
                         segment.atom_temporal_transitions_audited) ||
        !audit_count_add(expected.atom_temporal_nearest_image_violations,
                         segment.atom_temporal_nearest_image_violations) ||
        !audit_count_add(expected.atom_temporal_injectivity_uncertified,
                         segment.atom_temporal_injectivity_uncertified) ||
        !audit_count_add(expected.independent_temporal_image_ambiguities,
                         segment.independent_temporal_image_ambiguities) ||
        !audit_count_add(expected.independent_space_time_edges_audited,
                         segment.independent_space_time_edges_audited) ||
        !audit_count_add(
            expected.independent_space_time_curvature_residuals,
            segment.independent_space_time_curvature_residuals)) {
      return std::unexpected("Phase 1 segment audit counter overflow");
    }
    expected.maximum_component_representative_error_A = std::max(
        expected.maximum_component_representative_error_A,
        segment.maximum_component_representative_error_A);
    expected.maximum_atom_peculiar_displacement_A = std::max(
        expected.maximum_atom_peculiar_displacement_A,
        segment.maximum_atom_peculiar_displacement_A);
    expected_atom_audited &= segment_atom_audited;
    expected_cochain_audited &= segment_atom_audited;
  }

  for (std::size_t frame_index = 0; frame_index < result.frames.size();
       ++frame_index) {
    const auto &selection = result.frames[frame_index];
    if (selection.use == HoloLiftPhase1FrameUse::TemporalObservation) {
      if (!frame_seen[frame_index])
        return std::unexpected("Phase 1 temporal frame was not audited");
      continue;
    }
    if (selection.atoms_audited != 0 ||
        selection.maximum_component_representative_error_A != 0.0 ||
        !frame_transition_audit_is_zero(selection)) {
      return std::unexpected("Phase 1 observation gap carries audit data");
    }
  }

  expected.selected_frames_complete =
      expected.temporal_frames_audited == result.temporal_observation_frames;
  expected.atom_temporal_injectivity_audited =
      !result.segments.empty() && expected_atom_audited;
  expected.independent_space_time_cochain_audited =
      !result.segments.empty() && expected_cochain_audited;
  const auto &actual = result.reconstruction_audit;
  if (actual.temporal_frames_audited != expected.temporal_frames_audited ||
      actual.atoms_audited != expected.atoms_audited ||
      actual.atom_temporal_transitions_audited !=
          expected.atom_temporal_transitions_audited ||
      actual.atom_temporal_nearest_image_violations !=
          expected.atom_temporal_nearest_image_violations ||
      actual.atom_temporal_injectivity_uncertified !=
          expected.atom_temporal_injectivity_uncertified ||
      actual.independent_temporal_image_ambiguities !=
          expected.independent_temporal_image_ambiguities ||
      actual.independent_space_time_edges_audited !=
          expected.independent_space_time_edges_audited ||
      actual.independent_space_time_curvature_residuals !=
          expected.independent_space_time_curvature_residuals ||
      actual.maximum_component_representative_error_A !=
          expected.maximum_component_representative_error_A ||
      actual.maximum_atom_peculiar_displacement_A !=
          expected.maximum_atom_peculiar_displacement_A ||
      actual.selected_frames_complete != expected.selected_frames_complete ||
      actual.atom_temporal_injectivity_audited !=
          expected.atom_temporal_injectivity_audited ||
      actual.independent_space_time_cochain_audited !=
          expected.independent_space_time_cochain_audited) {
    return std::unexpected(
        "Phase 1 global audit is not the reduction of its segments");
  }
  const bool expected_atom_pass =
      expected.atom_temporal_injectivity_audited &&
      expected.atom_temporal_nearest_image_violations == 0 &&
      expected.atom_temporal_injectivity_uncertified == 0 &&
      expected.independent_temporal_image_ambiguities == 0;
  const bool expected_cochain_closed =
      expected.independent_space_time_cochain_audited &&
      expected.independent_temporal_image_ambiguities == 0 &&
      expected.independent_space_time_curvature_residuals == 0;
  if (!result.certificate.all_atom_reconstruction_audited ||
      result.certificate.coordinates_materialized_and_replay_consistent !=
          expected.selected_frames_complete ||
      result.certificate.atom_temporal_injectivity_audited !=
          expected.atom_temporal_injectivity_audited ||
      result.certificate.atom_temporal_injectivity_audit_passed !=
          expected_atom_pass ||
      result.certificate.independent_space_time_cochain_audited !=
          expected.independent_space_time_cochain_audited ||
      result.certificate.independent_space_time_cochain_closed !=
          expected_cochain_closed) {
    return std::unexpected(
        "Phase 1 certificate does not match the sealed atom audit");
  }
  return {};
}

std::expected<HoloLiftPhase1AuditDigest, std::string>
compute_phase1_audit_digest(
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    const HoloLiftPhase1Result &result) {
  if (prepared_frames.size() != store.frames().size() ||
      result.frames.size() != store.frames().size()) {
    return std::unexpected("Phase 1 audit digest frame coverage mismatch");
  }
  Phase1AuditSealHash hash;
  hash.word(result.api_version);
  hash.boolean(result.config.ordinal_rank_saturated);
  hash.boolean(result.config.evidence_absent_zero_rank);
  hash.boolean(result.config.allow_policy_admitted_candidates);
  hash.real(result.config.unique_evidence_mismatch_weight);
  // Only when set, so that the digests of configurations without it are the
  // ones recorded before it existed.
  if (result.config.supported_evidence_mismatch_weight != 0.0) {
    hash.boolean(true);
    hash.real(result.config.supported_evidence_mismatch_weight);
  }
  hash.count(result.frames.size());
  hash.count(result.segments.size());
  for (std::size_t frame_index = 0; frame_index < store.frames().size();
       ++frame_index) {
    const auto &frame = store.frames()[frame_index];
    const auto &prepared = prepared_frames[frame_index];
    const auto &selection = result.frames[frame_index];
    if (prepared.frame_index() != frame_index ||
        prepared.trajectory_frame_index() != frame.trajectory_frame_index ||
        prepared.topology_epoch_identity() != frame.topology_epoch_identity ||
        prepared.wrapped_coordinate_hash() != frame.wrapped_coordinate_hash ||
        prepared.spatial_lift_identity() != frame.spatial_lift_identity ||
        hash_hololift_box_matrix(prepared.box_matrix()) != frame.box_hash) {
      return std::unexpected(
          "Phase 1 audit digest prepared-frame binding mismatch");
    }
    hash.count(frame_index);
    hash.count(frame.trajectory_frame_index);
    hash.identity(frame.box_hash);
    hash.identity(frame.atom_selection_hash);
    hash.identity(frame.canonical_atom_universe_hash);
    hash.identity(frame.wrapped_coordinate_hash);
    hash.identity(frame.topology_epoch_identity);
    hash.identity(frame.spatial_lift_identity);
    hash.word(static_cast<std::uint64_t>(selection.use));
    hash.count(selection.segment_index);
    hash.count(selection.candidate_index);
    hash.boolean(selection.selected_candidate_provider_unsupported);
    hash.boolean(selection.preceding_transition_provider_unsupported);
    if (selection.use == HoloLiftPhase1FrameUse::TemporalObservation) {
      if (selection.candidate_index >= store.candidates().size())
        return std::unexpected(
            "Phase 1 audit digest candidate index is invalid");
      hash.identity(
          store.candidates()[selection.candidate_index].assignment_identity);
    } else {
      hash.identity({});
    }
    hash.image(selection.cumulative_global_gauge);
    hash.count(selection.atoms_audited);
    hash.count(selection.atom_temporal_transitions_audited);
    hash.count(selection.atom_temporal_nearest_image_violations);
    hash.count(selection.atom_temporal_injectivity_uncertified);
    hash.count(selection.independent_temporal_image_ambiguities);
    hash.count(selection.independent_space_time_edges_audited);
    hash.count(selection.independent_space_time_curvature_residuals);
    hash.real(selection.maximum_component_representative_error_A);
    hash.real(selection.maximum_atom_peculiar_displacement_A);
  }
  for (const auto &segment : result.segments) {
    hash.count(segment.frame_begin);
    hash.count(segment.frame_count);
    hash.count(segment.provider_unsupported_selected_candidate_count);
    hash.count(segment.provider_unsupported_transition_count);
    hash.word(segment.topology_epoch_index);
    hash.count(segment.temporal_frames_audited);
    hash.count(segment.atoms_audited);
    hash.count(segment.atom_temporal_transitions_audited);
    hash.count(segment.atom_temporal_nearest_image_violations);
    hash.count(segment.atom_temporal_injectivity_uncertified);
    hash.count(segment.independent_temporal_image_ambiguities);
    hash.count(segment.independent_space_time_edges_audited);
    hash.count(segment.independent_space_time_curvature_residuals);
    hash.real(segment.maximum_component_representative_error_A);
    hash.real(segment.maximum_atom_peculiar_displacement_A);
    hash.boolean(segment.all_atom_reconstruction_audited);
    hash.boolean(segment.coordinates_materialized_and_replay_consistent);
    hash.boolean(segment.atom_temporal_injectivity_audited);
    hash.boolean(segment.atom_temporal_injectivity_audit_passed);
    hash.boolean(segment.independent_space_time_cochain_audited);
    hash.boolean(segment.independent_space_time_cochain_closed);
    hash.boolean(segment.temporal_lift_locally_certified);
  }
  const auto &audit = result.reconstruction_audit;
  hash.count(audit.temporal_frames_audited);
  hash.count(audit.atoms_audited);
  hash.count(audit.atom_temporal_transitions_audited);
  hash.count(audit.atom_temporal_nearest_image_violations);
  hash.count(audit.atom_temporal_injectivity_uncertified);
  hash.count(audit.independent_temporal_image_ambiguities);
  hash.count(audit.independent_space_time_edges_audited);
  hash.count(audit.independent_space_time_curvature_residuals);
  hash.real(audit.maximum_component_representative_error_A);
  hash.real(audit.maximum_atom_peculiar_displacement_A);
  hash.boolean(audit.selected_frames_complete);
  hash.boolean(audit.atom_temporal_injectivity_audited);
  hash.boolean(audit.independent_space_time_cochain_audited);
  const auto &certificate = result.certificate;
  hash.boolean(certificate.all_atom_reconstruction_audited);
  hash.boolean(certificate.coordinates_materialized_and_replay_consistent);
  hash.boolean(certificate.atom_temporal_injectivity_audited);
  hash.boolean(certificate.atom_temporal_injectivity_audit_passed);
  hash.boolean(certificate.independent_space_time_cochain_audited);
  hash.boolean(certificate.independent_space_time_cochain_closed);
  hash.boolean(certificate.temporal_lift_locally_certified);
  hash.boolean(certificate.complete_domain_trajectory_claim_eligible);
  return hash.finish();
}

struct SourceAtomKeyHash {
  [[nodiscard]] std::size_t
  operator()(const HoloLiftSourceAtomKey &key) const noexcept {
    const std::uint64_t owner = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(key.owner));
    const std::uint64_t mixed =
        key.source_atom_id ^
        (owner + 0x9e3779b97f4a7c15ULL + (key.source_atom_id << 6) +
         (key.source_atom_id >> 2));
    return static_cast<std::size_t>(mixed);
  }
};

struct Phase1EpochCache {
  HoloLiftHash128 topology_epoch_identity;
  std::vector<double> component_weights;
  std::vector<std::size_t> component_atom_counts;
  HoloLiftPhase1ComponentWeighting component_weighting =
      HoloLiftPhase1ComponentWeighting::AtomCount;
  std::unordered_map<HoloLiftSourceAtomKey, std::uint32_t, SourceAtomKeyHash>
      component_by_atom;
  std::unordered_map<HoloLiftSourceAtomKey, double, SourceAtomKeyHash>
      weight_by_atom;
  std::unordered_map<HoloLiftSourceAtomKey, std::uint32_t, SourceAtomKeyHash>
      canonical_dense_by_atom;
  std::vector<HoloLiftSourceAtomKey> canonical_atoms;
};

struct Phase1AtomMapping {
  std::vector<std::uint32_t> component_by_input_atom;
  std::vector<std::uint32_t> canonical_dense_by_input_atom;
  std::vector<std::uint32_t> input_index_by_canonical_dense_atom;
  std::vector<double> weight_by_input_atom;
};

} // namespace

struct HoloLiftPhase1PreparationContext::Impl {
  const HoloLiftObservationStore *store = nullptr;
  std::vector<Phase1EpochCache> epochs;
};

namespace {

struct Phase1CandidateState {
  std::size_t candidate_index = HOLOLIFT_NO_INDEX;
  double emission_cost = 0.0;
  bool pinned_anchor = false;
};

struct Phase1Transition {
  double cost = std::numeric_limits<double>::infinity();
  HoloLiftLatticeImage gauge_increment;
  double component_displacement_rms_A = 0.0;
  double component_displacement_max_A = 0.0;
  double component_injectivity_radius_lower_bound_A = 0.0;
  double component_injectivity_margin_A = 0.0;
  std::size_t component_nearest_image_violations = 0;
  std::size_t component_injectivity_uncertified = 0;
  bool gauge_increment_ambiguous = false;
  bool quadratic_direct_residual_fallback = false;
};

struct Phase1TransportedFrameMetadata {
  const HoloLiftPhase1PreparedFrame *source = nullptr;
  HoloLiftLatticeBasisTransport transport;
  std::array<double, 6> physical_gram{};
  std::size_t candidate_begin = 0;
  std::size_t candidate_count = 0;
  std::size_t component_count = 0;
};

struct Phase1MaterializedFrameLayer {
  const Phase1TransportedFrameMetadata *metadata = nullptr;
  std::vector<std::array<double, 3>> component_representatives;
  std::size_t candidate_begin = 0;
  std::size_t candidate_count = 0;
  std::vector<HoloLiftLatticeImage> candidate_component_images_reference;
};

struct Phase1TransitionContext {
  const Phase1MaterializedFrameLayer *previous = nullptr;
  const Phase1MaterializedFrameLayer *current = nullptr;
  std::array<double, 6> gram{};
  titan_pbctopo::PbctopoLatticeMetric metric;
  std::vector<std::array<double, 3>> base_component_delta;
  std::vector<double> component_weights;
  double total_weight = 0.0;
  double variance = 0.0;
  double injectivity_radius_lower_bound_A = 0.0;

  explicit Phase1TransitionContext(const std::array<double, 6> &input_gram)
      : gram(input_gram), metric(input_gram) {}
};

struct Phase1PathSlot {
  double cost = std::numeric_limits<double>::infinity();
  std::size_t predecessor_state = HOLOLIFT_NO_INDEX;
  std::size_t predecessor_slot = HOLOLIFT_NO_INDEX;
  std::size_t optimal_path_count_capped = 0;
  bool valid = false;
};

struct Phase1CompactBackpointer {
  std::uint32_t predecessor_state = HOLOLIFT_NO_DENSE_INDEX;
  std::uint8_t predecessor_slot = 0;
  bool valid = false;
};

struct Phase1BackpointerLayer {
  std::vector<std::array<Phase1CompactBackpointer, 2>> paths;
};

// Below this amount of deterministic transition work, synchronization costs
// dominate the small retained candidate bands used by the common path.
constexpr std::size_t kPhase1DpParallelWorkThreshold = 8192;

std::size_t phase1_dp_work_estimate(std::size_t previous_states,
                                    std::size_t current_states,
                                    std::size_t component_count) noexcept {
  constexpr std::size_t maximum =
      std::numeric_limits<std::size_t>::max();
  if (previous_states == 0 || current_states == 0 || component_count == 0)
    return 0;
  if (previous_states > maximum / current_states)
    return maximum;
  const std::size_t state_pairs = previous_states * current_states;
  if (state_pairs > maximum / component_count)
    return maximum;
  return state_pairs * component_count;
}

class Phase1DpExecutor {
public:
  struct Failure {
    enum class Kind { Setup, WorkerTask };

    Kind kind = Kind::Setup;
    std::exception_ptr exception;
  };

  static std::unique_ptr<Phase1DpExecutor>
  create(std::size_t worker_capacity, bool &creation_failed) {
    creation_failed = false;
    std::unique_ptr<Phase1DpExecutor> executor;
    try {
      executor = std::unique_ptr<Phase1DpExecutor>(new Phase1DpExecutor);
      executor->workers_.reserve(worker_capacity);
      for (std::size_t worker = 0; worker < worker_capacity; ++worker) {
#ifdef TITAN_HOLOLIFT_PHASE1_TEST_HOOKS
        if (worker == 1 &&
            g_force_phase1_dp_pool_creation_bad_alloc_for_test.load(
                std::memory_order_relaxed)) {
          throw std::bad_alloc{};
        }
        if (worker == 1 &&
            g_force_phase1_dp_pool_creation_failure_for_test.load(
                std::memory_order_relaxed)) {
          throw std::system_error(
              std::make_error_code(std::errc::resource_unavailable_try_again));
        }
#endif
        executor->workers_.emplace_back(
            [executor_ptr = executor.get(), worker] {
              executor_ptr->worker_loop(worker);
            });
      }
    } catch (const std::system_error &) {
      creation_failed = true;
      if (executor)
        executor->shutdown();
      return {};
    } catch (const std::bad_alloc &) {
      creation_failed = true;
      if (executor)
        executor->shutdown();
      return {};
    }
    return executor;
  }

  Phase1DpExecutor(const Phase1DpExecutor &) = delete;
  Phase1DpExecutor &operator=(const Phase1DpExecutor &) = delete;

  ~Phase1DpExecutor() { shutdown(); }

  [[nodiscard]] std::size_t capacity() const noexcept {
    return workers_.size();
  }

  template <typename Task>
  std::expected<void, Failure>
  run(std::size_t active_workers, std::size_t task_count,
      Task &&task) {
    active_workers = std::min(active_workers, workers_.size());
    if (active_workers == 0 || task_count == 0)
      return {};
    {
      std::lock_guard lock(mutex_);
      try {
#ifdef TITAN_HOLOLIFT_PHASE1_TEST_HOOKS
        if (g_force_phase1_dp_run_setup_exception_for_test.load(
                std::memory_order_relaxed)) {
          throw std::bad_alloc{};
        }
#endif
        task_ = std::forward<Task>(task);
      } catch (...) {
        std::function<void(std::size_t)> empty;
        task_.swap(empty);
        return std::unexpected(
            Failure{Failure::Kind::Setup, std::current_exception()});
      }
      task_exception_ = nullptr;
      task_failed_.store(false, std::memory_order_relaxed);
      task_count_ = task_count;
      active_workers_ = active_workers;
      next_task_.store(0, std::memory_order_relaxed);
      remaining_workers_.store(active_workers, std::memory_order_relaxed);
      ++generation_;
    }
    work_ready_.notify_all();

    std::unique_lock lock(mutex_);
    work_done_.wait(lock, [&] {
      return remaining_workers_.load(std::memory_order_acquire) == 0;
    });
    const std::exception_ptr task_exception = task_exception_;
    task_ = {};
    task_exception_ = nullptr;
    task_count_ = 0;
    active_workers_ = 0;
    if (task_exception)
      return std::unexpected(
          Failure{Failure::Kind::WorkerTask, task_exception});
    return {};
  }

private:
  Phase1DpExecutor() = default;

  void worker_loop(std::size_t worker_index) {
    std::size_t observed_generation = 0;
    while (true) {
      std::size_t task_count = 0;
      {
        std::unique_lock lock(mutex_);
        work_ready_.wait(lock, [&] {
          return stopping_ || generation_ != observed_generation;
        });
        if (stopping_)
          return;
        observed_generation = generation_;
        if (worker_index >= active_workers_)
          continue;
        task_count = task_count_;
      }

      while (true) {
        if (task_failed_.load(std::memory_order_acquire))
          break;
        const std::size_t task_index =
            next_task_.fetch_add(1, std::memory_order_relaxed);
        if (task_index >= task_count)
          break;
        try {
#ifdef TITAN_HOLOLIFT_PHASE1_TEST_HOOKS
          if (task_index == 0 &&
              g_force_phase1_dp_worker_exception_for_test.load(
                  std::memory_order_relaxed)) {
            throw std::runtime_error(
                "injected Phase 1 DP worker task failure");
          }
#endif
          task_(task_index);
        } catch (...) {
          {
            std::lock_guard lock(mutex_);
            if (!task_exception_)
              task_exception_ = std::current_exception();
          }
          task_failed_.store(true, std::memory_order_release);
          break;
        }
      }
      if (remaining_workers_.fetch_sub(1, std::memory_order_acq_rel) == 1)
        work_done_.notify_one();
    }
  }

  void shutdown() noexcept {
    {
      std::lock_guard lock(mutex_);
      if (stopping_)
        return;
      stopping_ = true;
      ++generation_;
    }
    work_ready_.notify_all();
    for (auto &worker : workers_) {
      if (worker.joinable())
        worker.join();
    }
    workers_.clear();
  }

  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  std::vector<std::jthread> workers_;
  std::function<void(std::size_t)> task_;
  std::exception_ptr task_exception_;
  std::atomic<std::size_t> next_task_{0};
  std::atomic<std::size_t> remaining_workers_{0};
  std::atomic<bool> task_failed_{false};
  std::size_t task_count_ = 0;
  std::size_t active_workers_ = 0;
  std::size_t generation_ = 0;
  bool stopping_ = false;
};

HoloLiftPhase1TransitionGap transition_gap_from_transport_failure(
    HoloLiftLatticeBasisTransportFailure failure) noexcept {
  switch (failure) {
  case HoloLiftLatticeBasisTransportFailure::Ambiguous:
    return HoloLiftPhase1TransitionGap::AmbiguousBasisTransport;
  case HoloLiftLatticeBasisTransportFailure::BasisDiscontinuity:
    return HoloLiftPhase1TransitionGap::BasisDiscontinuity;
  case HoloLiftLatticeBasisTransportFailure::InvalidPolicy:
  case HoloLiftLatticeBasisTransportFailure::InvalidInputBox:
  case HoloLiftLatticeBasisTransportFailure::TransformOutOfRange:
  case HoloLiftLatticeBasisTransportFailure::NoUnimodularCandidate:
  case HoloLiftLatticeBasisTransportFailure::ContinuityMismatch:
  case HoloLiftLatticeBasisTransportFailure::AutomaticResidualGate:
  case HoloLiftLatticeBasisTransportFailure::AutomaticImprovementGate:
  case HoloLiftLatticeBasisTransportFailure::InvalidSelectedTransport:
    return HoloLiftPhase1TransitionGap::BasisMismatch;
  }
  return HoloLiftPhase1TransitionGap::BasisMismatch;
}

class KahanSum {
public:
  void add(long double value) noexcept {
    const long double corrected = value - correction_;
    const long double next = sum_ + corrected;
    correction_ = (next - sum_) - corrected;
    sum_ = next;
  }

  [[nodiscard]] long double value() const noexcept { return sum_; }

private:
  long double sum_ = 0.0L;
  long double correction_ = 0.0L;
};

long double gram_bilinear(const std::array<double, 6> &gram,
                          const std::array<long double, 3> &lhs,
                          const std::array<long double, 3> &rhs) noexcept {
  return lhs[0] * (static_cast<long double>(gram[0]) * rhs[0] +
                   static_cast<long double>(gram[1]) * rhs[1] +
                   static_cast<long double>(gram[2]) * rhs[2]) +
         lhs[1] * (static_cast<long double>(gram[1]) * rhs[0] +
                   static_cast<long double>(gram[3]) * rhs[1] +
                   static_cast<long double>(gram[4]) * rhs[2]) +
         lhs[2] * (static_cast<long double>(gram[2]) * rhs[0] +
                   static_cast<long double>(gram[4]) * rhs[1] +
                   static_cast<long double>(gram[5]) * rhs[2]);
}

long double gram_quadratic(
    const std::array<double, 6> &gram,
    const std::array<long double, 3> &vector) noexcept {
  return gram_bilinear(gram, vector, vector);
}

bool phase1_cost_less(double lhs, double rhs, double tolerance) noexcept {
  const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
  return lhs < rhs - tolerance * scale;
}

bool phase1_cost_equal(double lhs, double rhs, double tolerance) noexcept {
  const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
  return std::fabs(lhs - rhs) <= tolerance * scale;
}

std::size_t capped_path_count_add(std::size_t lhs, std::size_t rhs) noexcept {
  return lhs >= 2 || rhs >= 2 || lhs + rhs >= 2 ? 2 : lhs + rhs;
}

void insert_path(std::array<Phase1PathSlot, 2> &slots,
                 const Phase1PathSlot &candidate, double tolerance) {
  if (!candidate.valid)
    return;
  if (!slots[0].valid) {
    slots[0] = candidate;
    return;
  }
  if (phase1_cost_less(candidate.cost, slots[0].cost, tolerance)) {
    const Phase1PathSlot old_best = slots[0];
    const Phase1PathSlot old_second = slots[1];
    slots[0] = candidate;
    slots[1] = {};
    insert_path(slots, old_best, tolerance);
    insert_path(slots, old_second, tolerance);
    return;
  }
  if (phase1_cost_equal(candidate.cost, slots[0].cost, tolerance)) {
    slots[0].optimal_path_count_capped = capped_path_count_add(
        slots[0].optimal_path_count_capped,
        candidate.optimal_path_count_capped);
    return;
  }
  if (!slots[1].valid ||
      phase1_cost_less(candidate.cost, slots[1].cost, tolerance)) {
    slots[1] = candidate;
  } else if (phase1_cost_equal(candidate.cost, slots[1].cost, tolerance)) {
    slots[1].optimal_path_count_capped = capped_path_count_add(
        slots[1].optimal_path_count_capped,
        candidate.optimal_path_count_capped);
  }
}

std::array<double, 6>
physical_gram(std::span<const double, 9> box_matrix) noexcept {
  auto dot_rows = [&](std::size_t lhs, std::size_t rhs) {
    return box_matrix[lhs * 3] * box_matrix[rhs * 3] +
           box_matrix[lhs * 3 + 1] * box_matrix[rhs * 3 + 1] +
           box_matrix[lhs * 3 + 2] * box_matrix[rhs * 3 + 2];
  };
  return {dot_rows(0, 0), dot_rows(0, 1), dot_rows(0, 2),
          dot_rows(1, 1), dot_rows(1, 2), dot_rows(2, 2)};
}

std::array<double, 6>
average_physical_gram(const Phase1TransportedFrameMetadata &previous,
                      const Phase1TransportedFrameMetadata &current) noexcept {
  std::array<double, 6> gram{};
  const auto &lhs = previous.physical_gram;
  const auto &rhs = current.physical_gram;
  for (std::size_t index = 0; index < gram.size(); ++index)
    gram[index] = 0.5 * (lhs[index] + rhs[index]);
  return gram;
}

double shortest_lattice_vector_lower_bound(
    const std::array<double, 6> &gram) noexcept {
  const double cofactor00 = gram[3] * gram[5] - gram[4] * gram[4];
  const double cofactor11 = gram[0] * gram[5] - gram[2] * gram[2];
  const double cofactor22 = gram[0] * gram[3] - gram[1] * gram[1];
  const double determinant =
      gram[0] * cofactor00 -
      gram[1] * (gram[1] * gram[5] - gram[2] * gram[4]) +
      gram[2] * (gram[1] * gram[4] - gram[2] * gram[3]);
  if (!std::isfinite(determinant) || determinant <= 0.0)
    return 0.0;
  const double inverse_trace =
      (cofactor00 + cofactor11 + cofactor22) / determinant;
  if (!std::isfinite(inverse_trace) || inverse_trace <= 0.0)
    return 0.0;
  // sigma_min(B) is a rigorous lower bound for every nonzero lattice vector;
  // 1/sqrt(trace(G^-1)) is a conservative lower bound for sigma_min(B).
  return 1.0 / std::sqrt(inverse_trace);
}

std::expected<std::vector<Phase1TransportedFrameMetadata>, std::string>
build_transported_segment_frame_metadata(
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    std::size_t frame_begin, std::size_t frame_end,
    const HoloLiftPhase1Config &config) {
  if (frame_begin >= frame_end || frame_end > prepared_frames.size())
    return std::unexpected("Phase 1 transported-frame range is invalid");

  std::vector<Phase1TransportedFrameMetadata> result;
  result.reserve(frame_end - frame_begin);
  for (std::size_t frame_index = frame_begin; frame_index < frame_end;
       ++frame_index) {
    const auto &source = prepared_frames[frame_index];
    Phase1TransportedFrameMetadata frame;
    frame.source = &source;
    if (result.empty()) {
      std::copy(source.box_matrix().begin(), source.box_matrix().end(),
                frame.transport.transported_box.begin());
      frame.transport.identity_fast_path = true;
      frame.transport.accepted_under_policy = true;
      frame.transport.provenance_sufficient_for_certification = true;
    } else {
      const auto transport = find_hololift_lattice_basis_transport(
          result.back().transport.transported_box, source.box_matrix(),
          config.lattice_basis_policy,
          config.basis_transport_max_relative_mismatch,
          config.basis_transport_ambiguity_tolerance,
          config.basis_transport_nonidentity_max_relative_mismatch,
          config.basis_transport_minimum_improvement_ratio);
      if (!transport)
        return std::unexpected("Phase 1 basis transport failed: " +
                               transport.error());
      frame.transport = *transport;
    }
    frame.physical_gram = physical_gram(frame.transport.transported_box);
    const auto &source_frame = store.frames()[source.frame_index()];
    frame.candidate_begin = source_frame.candidates.begin;
    frame.candidate_count = source_frame.candidates.count;
    frame.component_count =
        source.component_fractional_representatives().size();
    if (!source_frame.candidates.valid_for(store.candidates().size()) ||
        frame.candidate_count >
            std::numeric_limits<std::size_t>::max() /
                std::max<std::size_t>(1, frame.component_count)) {
      return std::unexpected(
          "Phase 1 transported candidate range is invalid");
    }
    result.push_back(std::move(frame));
  }
  return result;
}

std::expected<Phase1MaterializedFrameLayer, std::string>
materialize_transported_frame_layer(
    const HoloLiftObservationStore &store,
    const Phase1TransportedFrameMetadata &metadata,
    std::size_t candidate_begin, std::size_t candidate_count) {
  if (!metadata.source || candidate_begin < metadata.candidate_begin) {
    return std::unexpected(
        "Phase 1 rolling candidate layer range is invalid");
  }
  const std::size_t candidate_offset =
      candidate_begin - metadata.candidate_begin;
  if (candidate_offset > metadata.candidate_count ||
      candidate_count > metadata.candidate_count - candidate_offset ||
      candidate_count > std::numeric_limits<std::size_t>::max() /
                            std::max<std::size_t>(1, metadata.component_count)) {
    return std::unexpected(
        "Phase 1 rolling candidate layer range is invalid");
  }

  Phase1MaterializedFrameLayer layer;
  layer.metadata = &metadata;
  layer.candidate_begin = candidate_begin;
  layer.candidate_count = candidate_count;
  layer.component_representatives.reserve(metadata.component_count);
  for (const auto &representative :
       metadata.source->component_fractional_representatives()) {
    const auto transformed =
        hololift_fractional_to_reference(representative, metadata.transport);
    if (!std::all_of(transformed.begin(), transformed.end(),
                     [](double value) { return std::isfinite(value); })) {
      return std::unexpected(
          "Phase 1 transported component representative is not finite");
    }
    layer.component_representatives.push_back(transformed);
  }
  if (layer.component_representatives.size() != metadata.component_count) {
    return std::unexpected(
        "Phase 1 rolling component representative range is invalid");
  }

  layer.candidate_component_images_reference.reserve(
      candidate_count * metadata.component_count);
  for (std::size_t local = 0; local < candidate_count; ++local) {
    const std::size_t candidate_index = candidate_begin + local;
    const auto &candidate = store.candidates()[candidate_index];
    if (!candidate.component_images.valid_for(
            store.component_images().size()) ||
        candidate.component_images.count != metadata.component_count) {
      return std::unexpected(
          "Phase 1 transported candidate images are invalid");
    }
    const auto images = store.component_images().subspan(
        candidate.component_images.begin, candidate.component_images.count);
    for (const auto &image : images) {
      const auto reference_image =
          hololift_lattice_image_to_reference(image, metadata.transport);
      if (!reference_image)
        return std::unexpected(reference_image.error());
      layer.candidate_component_images_reference.push_back(*reference_image);
    }
  }
  return layer;
}

std::expected<Phase1MaterializedFrameLayer, std::string>
materialize_transported_frame_layer(
    const HoloLiftObservationStore &store,
    const Phase1TransportedFrameMetadata &metadata) {
  return materialize_transported_frame_layer(
      store, metadata, metadata.candidate_begin, metadata.candidate_count);
}

std::expected<Phase1MaterializedFrameLayer, std::string>
materialize_selected_transported_frame_layer(
    const HoloLiftObservationStore &store,
    const Phase1TransportedFrameMetadata &metadata,
    std::size_t candidate_index) {
  return materialize_transported_frame_layer(store, metadata, candidate_index,
                                             1);
}

std::span<const HoloLiftLatticeImage> transported_candidate_images(
    const Phase1MaterializedFrameLayer &frame,
    std::size_t candidate_index) noexcept {
  if (candidate_index < frame.candidate_begin ||
      candidate_index >= frame.candidate_begin + frame.candidate_count ||
      !frame.metadata || frame.metadata->component_count == 0) {
    return {};
  }
  const std::size_t local = candidate_index - frame.candidate_begin;
  const std::size_t begin = local * frame.metadata->component_count;
  if (begin > frame.candidate_component_images_reference.size() ||
      frame.metadata->component_count >
          frame.candidate_component_images_reference.size() - begin) {
    return {};
  }
  return std::span<const HoloLiftLatticeImage>(
      frame.candidate_component_images_reference)
      .subspan(begin, frame.metadata->component_count);
}

bool checked_int64_add(std::int64_t lhs, std::int64_t rhs,
                       std::int64_t &sum) noexcept {
  constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
  constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
  if ((rhs > 0 && lhs > maximum - rhs) ||
      (rhs < 0 && lhs < minimum - rhs)) {
    return false;
  }
  sum = lhs + rhs;
  return true;
}

bool checked_image_add(const HoloLiftLatticeImage &lhs,
                       const HoloLiftLatticeImage &rhs,
                       HoloLiftLatticeImage &result) noexcept {
  return checked_int64_add(lhs.x, rhs.x, result.x) &&
         checked_int64_add(lhs.y, rhs.y, result.y) &&
         checked_int64_add(lhs.z, rhs.z, result.z);
}

bool checked_int64_subtract(std::int64_t lhs, std::int64_t rhs,
                            std::int64_t &difference) noexcept {
  if (rhs == std::numeric_limits<std::int64_t>::min())
    return false;
  return checked_int64_add(lhs, -rhs, difference);
}

bool checked_image_subtract(const HoloLiftLatticeImage &lhs,
                            const HoloLiftLatticeImage &rhs,
                            HoloLiftLatticeImage &result) noexcept {
  return checked_int64_subtract(lhs.x, rhs.x, result.x) &&
         checked_int64_subtract(lhs.y, rhs.y, result.y) &&
         checked_int64_subtract(lhs.z, rhs.z, result.z);
}

bool checked_image_negate(const HoloLiftLatticeImage &source,
                          HoloLiftLatticeImage &result) noexcept {
  const HoloLiftLatticeImage zero{};
  return checked_image_subtract(zero, source, result);
}

bool image_is_exact_binary64(const HoloLiftLatticeImage &image) noexcept {
  constexpr std::int64_t maximum_exact_integer = 9007199254740991LL;
  const auto exact = [](std::int64_t value) {
    return value >= -maximum_exact_integer && value <= maximum_exact_integer;
  };
  return exact(image.x) && exact(image.y) && exact(image.z);
}

std::expected<double, std::string>
checked_image_delta(std::int64_t current, std::int64_t previous) {
  constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
  constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
  if ((previous > 0 && current < minimum + previous) ||
      (previous < 0 && current > maximum + previous)) {
    return std::unexpected("Phase 1 component image difference overflows");
  }
  const std::int64_t delta = current - previous;
  constexpr std::int64_t maximum_exact_integer = 9007199254740991LL;
  if (delta < -maximum_exact_integer || delta > maximum_exact_integer) {
    return std::unexpected(
        "Phase 1 component image difference exceeds exact binary64 range");
  }
  return static_cast<double>(delta);
}

std::expected<Phase1AtomMapping, std::string>
atom_mapping(const HoloLiftPhase1PreparationContext::Impl &context,
             const HoloLiftObservationStore &store,
             const HoloLiftFrameRecord &frame,
             std::span<const HoloLiftSourceAtomKey> atom_order) {
  if (frame.topology_epoch_index >= context.epochs.size())
    return std::unexpected("Phase 1 topology epoch cache is missing");
  const auto &cache = context.epochs[frame.topology_epoch_index];
  Phase1AtomMapping mapping;
  mapping.input_index_by_canonical_dense_atom.assign(
      atom_order.size(), HOLOLIFT_NO_DENSE_INDEX);
  mapping.component_by_input_atom.reserve(atom_order.size());
  mapping.canonical_dense_by_input_atom.reserve(atom_order.size());
  mapping.weight_by_input_atom.reserve(atom_order.size());
  for (std::size_t input_index = 0; input_index < atom_order.size();
       ++input_index) {
    const auto &atom = atom_order[input_index];
    const auto component = cache.component_by_atom.find(atom);
    const auto canonical = cache.canonical_dense_by_atom.find(atom);
    const auto weight = cache.weight_by_atom.find(atom);
    if (component == cache.component_by_atom.end() ||
        canonical == cache.canonical_dense_by_atom.end() ||
        weight == cache.weight_by_atom.end()) {
      return std::unexpected(
          "Phase 1 atom selection is absent from topology epoch");
    }
    if (input_index >= HOLOLIFT_NO_DENSE_INDEX ||
        canonical->second >=
            mapping.input_index_by_canonical_dense_atom.size() ||
        mapping.input_index_by_canonical_dense_atom[canonical->second] !=
            HOLOLIFT_NO_DENSE_INDEX) {
      return std::unexpected(
          "Phase 1 atom selection is not a canonical one-to-one mapping");
    }
    mapping.component_by_input_atom.push_back(component->second);
    mapping.canonical_dense_by_input_atom.push_back(canonical->second);
    mapping.weight_by_input_atom.push_back(weight->second);
    mapping.input_index_by_canonical_dense_atom[canonical->second] =
        static_cast<std::uint32_t>(input_index);
  }
  if (mapping.component_by_input_atom.size() !=
          store.topology_epochs()[frame.topology_epoch_index].atom_count ||
      std::find(mapping.input_index_by_canonical_dense_atom.begin(),
                mapping.input_index_by_canonical_dense_atom.end(),
                HOLOLIFT_NO_DENSE_INDEX) !=
          mapping.input_index_by_canonical_dense_atom.end()) {
    return std::unexpected("Phase 1 atom/component mapping size mismatch");
  }
  return mapping;
}

std::expected<std::vector<Phase1CandidateState>, std::string>
build_phase1_candidate_states(const HoloLiftObservationStore &store,
                              const HoloLiftFrameRecord &frame,
                              const HoloLiftPhase1Config &config) {
  if (!frame.candidates.valid_for(store.candidates().size()))
    return std::unexpected("Phase 1 frame candidate range is invalid");

  std::vector<std::size_t> candidate_indices;
  candidate_indices.reserve(frame.candidates.count);
  std::size_t strong_anchor_index = HOLOLIFT_NO_INDEX;
  for (std::size_t local = 0; local < frame.candidates.count; ++local) {
    const std::size_t candidate_index = frame.candidates.begin + local;
    const auto &candidate = store.candidates()[candidate_index];
    const bool provider_admissible =
        candidate.evidence.hard_feasible &&
        candidate.observation_class != HoloLiftObservationClass::NotHardFeasible;
    const bool policy_admissible =
        config.allow_policy_admitted_candidates &&
        candidate.policy_admission != HoloLiftPolicyAdmission::None;
    if (!provider_admissible && !policy_admissible) {
      continue;
    }
    if (candidate.observation_class ==
        HoloLiftObservationClass::StrongGlobalAnchor) {
      return std::unexpected(
          "Phase 1 does not accept unsupported global anchors");
    }
    if (candidate.rank == 0)
      return std::unexpected("Phase 1 candidate rank is zero");
    if (candidate.observation_class ==
        HoloLiftObservationClass::StrongBoundedAnchor) {
      if (strong_anchor_index != HOLOLIFT_NO_INDEX)
        return std::unexpected(
            "Phase 1 frame contains multiple strong bounded anchors");
      strong_anchor_index = candidate_index;
    }
    candidate_indices.push_back(candidate_index);
  }
  if (candidate_indices.empty())
    return std::unexpected(
        "Phase 1 temporal observation has no admitted candidate");

  std::vector<Phase1CandidateState> states;
  states.reserve(candidate_indices.size());
  const double rank_denominator =
      static_cast<double>(std::max<std::size_t>(1, candidate_indices.size() - 1));
  bool evidence_absent = true;
  std::size_t supported_count = 0;
  std::size_t unique_supported_index = HOLOLIFT_NO_INDEX;
  for (const std::size_t candidate_index : candidate_indices) {
    const auto &candidate = store.candidates()[candidate_index];
    if (candidate.evidence.hard_feasible &&
        candidate.evidence_compatible_hypothesis_count > 0) {
      evidence_absent = false;
      ++supported_count;
      unique_supported_index = candidate_index;
    }
  }
  const bool zero_rank_frame =
      config.evidence_absent_zero_rank && evidence_absent;
  for (const std::size_t candidate_index : candidate_indices) {
    const auto &candidate = store.candidates()[candidate_index];
    const double normalized_rank =
        zero_rank_frame ? 0.0
        : config.ordinal_rank_saturated
            ? (candidate.rank > 1 ? 1.0 : 0.0)
            : static_cast<double>(candidate.rank - 1) / rank_denominator;
    double emission = config.ordinal_rank_weight * normalized_rank;
    if (supported_count == 1 && candidate_index != unique_supported_index)
      emission += config.unique_evidence_mismatch_weight;
    if (supported_count >= 1 &&
        !(candidate.evidence.hard_feasible &&
          candidate.evidence_compatible_hypothesis_count > 0))
      emission += config.supported_evidence_mismatch_weight;
    if (strong_anchor_index != HOLOLIFT_NO_INDEX &&
        candidate_index != strong_anchor_index) {
      emission += config.bounded_anchor_competitor_penalty;
    }
    if (!std::isfinite(emission))
      return std::unexpected("Phase 1 emission cost is not finite");
    states.push_back({candidate_index, emission, false});
  }

  if (config.pin_strong_bounded_anchors &&
      strong_anchor_index != HOLOLIFT_NO_INDEX) {
    const auto found =
        std::find_if(states.begin(), states.end(), [&](const auto &state) {
          return state.candidate_index == strong_anchor_index;
        });
    if (found == states.end())
      return std::unexpected("Phase 1 strong anchor is not hard feasible");
    Phase1CandidateState pinned = *found;
    pinned.pinned_anchor = true;
    states.assign(1, pinned);
  }
  return states;
}

std::expected<double, std::string>
transition_variance(double delta_time_ps,
                    const HoloLiftPhase1Config &config) {
  if (!std::isfinite(delta_time_ps) || delta_time_ps <= 0.0)
    return std::unexpected("Phase 1 transition time step is not positive");
  double time_factor = 1.0;
  switch (config.time_scaling) {
  case HoloLiftPhase1TimeScaling::PerFrameDisplacement:
    break;
  case HoloLiftPhase1TimeScaling::Diffusive:
    time_factor = delta_time_ps;
    break;
  case HoloLiftPhase1TimeScaling::Ballistic:
    time_factor = delta_time_ps * delta_time_ps;
    break;
  }
  const double variance =
      config.variance_floor_A2 +
      config.temporal_scale * config.temporal_scale * time_factor;
  if (!std::isfinite(variance) || variance <= 0.0)
    return std::unexpected("Phase 1 transition variance is invalid");
  return variance;
}

std::expected<Phase1TransitionContext, std::string>
build_phase1_transition_context(
    const HoloLiftObservationStore &store,
    const HoloLiftTopologyEpochRecord &epoch,
    const Phase1MaterializedFrameLayer &previous_frame,
    const Phase1MaterializedFrameLayer &current_frame,
    const HoloLiftPhase1Config &config) {
  const auto components = store.components().subspan(
      epoch.components.begin, epoch.components.count);
  if (!previous_frame.metadata || !current_frame.metadata ||
      !previous_frame.metadata->source || !current_frame.metadata->source ||
      previous_frame.component_representatives.size() != components.size() ||
      current_frame.component_representatives.size() != components.size()) {
    return std::unexpected(
        "Phase 1 transition context does not match topology epoch");
  }
  const auto gram = average_physical_gram(*previous_frame.metadata,
                                          *current_frame.metadata);
  Phase1TransitionContext result(gram);
  if (!result.metric.valid())
    return std::unexpected("Phase 1 transition lattice metric is invalid");
  result.previous = &previous_frame;
  result.current = &current_frame;
  result.base_component_delta.reserve(components.size());
  result.component_weights.reserve(components.size());
  const auto previous_weights =
      previous_frame.metadata->source->component_weights();
  const auto current_weights =
      current_frame.metadata->source->component_weights();
  if (previous_frame.metadata->source->component_weighting() !=
          current_frame.metadata->source->component_weighting() ||
      previous_weights.size() != components.size() ||
      current_weights.size() != components.size() ||
      !std::equal(previous_weights.begin(), previous_weights.end(),
                  current_weights.begin(), current_weights.end())) {
    return std::unexpected(
        "Phase 1 transition component weighting is inconsistent");
  }
  for (std::size_t index = 0; index < components.size(); ++index) {
    const double weight = previous_weights[index];
    if (!std::isfinite(weight) || weight <= 0.0)
      return std::unexpected("Phase 1 component has zero atom weight");
    result.component_weights.push_back(weight);
    result.total_weight += weight;
    result.base_component_delta.push_back(
        {current_frame.component_representatives[index][0] -
             previous_frame.component_representatives[index][0],
         current_frame.component_representatives[index][1] -
             previous_frame.component_representatives[index][1],
         current_frame.component_representatives[index][2] -
             previous_frame.component_representatives[index][2]});
  }
  if (!std::isfinite(result.total_weight) || result.total_weight <= 0.0)
    return std::unexpected("Phase 1 transition total weight is invalid");
  const auto variance = transition_variance(
      std::fabs(current_frame.metadata->source->time_ps() -
                previous_frame.metadata->source->time_ps()),
      config);
  if (!variance)
    return std::unexpected(variance.error());
  result.variance = *variance;
  const double previous_shortest =
      shortest_lattice_vector_lower_bound(
          previous_frame.metadata->physical_gram);
  const double current_shortest =
      shortest_lattice_vector_lower_bound(
          current_frame.metadata->physical_gram);
  result.injectivity_radius_lower_bound_A =
      0.5 * std::min(previous_shortest, current_shortest);
  if (!std::isfinite(result.injectivity_radius_lower_bound_A) ||
      result.injectivity_radius_lower_bound_A <= 0.0) {
    return std::unexpected(
        "Phase 1 injectivity-radius lower bound is invalid");
  }
  return result;
}

std::expected<Phase1Transition, std::string> phase1_transition_cost(
    const HoloLiftObservationStore &store,
    const Phase1TransitionContext &context,
    std::size_t previous_candidate_index,
    std::size_t current_candidate_index,
    const HoloLiftPhase1Config &config, bool audit_components) {
  const std::size_t component_count = context.base_component_delta.size();
  if (!context.previous || !context.current ||
      context.component_weights.size() != component_count ||
      previous_candidate_index >= store.candidates().size() ||
      current_candidate_index >= store.candidates().size()) {
    return std::unexpected(
        "Phase 1 transition inputs do not match topology epoch");
  }
  const auto previous_images = transported_candidate_images(
      *context.previous, previous_candidate_index);
  const auto current_images = transported_candidate_images(
      *context.current, current_candidate_index);
  if (previous_images.size() != component_count ||
      current_images.size() != component_count) {
    return std::unexpected(
        "Phase 1 pretransformed candidate images are invalid");
  }
  const auto component_delta = [&](std::size_t index)
      -> std::expected<std::array<double, 3>, std::string> {
    const auto dx =
        checked_image_delta(current_images[index].x, previous_images[index].x);
    const auto dy =
        checked_image_delta(current_images[index].y, previous_images[index].y);
    const auto dz =
        checked_image_delta(current_images[index].z, previous_images[index].z);
    if (!dx || !dy || !dz)
      return std::unexpected(!dx ? dx.error() : (!dy ? dy.error() : dz.error()));
    std::array<double, 3> delta{
        context.base_component_delta[index][0] + *dx,
        context.base_component_delta[index][1] + *dy,
        context.base_component_delta[index][2] + *dz};
    if (!std::all_of(delta.begin(), delta.end(),
                     [](double value) { return std::isfinite(value); })) {
      return std::unexpected(
          "Phase 1 component displacement is not finite");
    }
    return delta;
  };
  std::array<KahanSum, 3> weighted_delta;
  KahanSum weighted_raw_distance2;
  std::vector<std::array<double, 3>> audited_deltas;
  if (audit_components)
    audited_deltas.reserve(component_count);
  for (std::size_t index = 0; index < component_count; ++index) {
    const double weight = context.component_weights[index];
    const auto delta = component_delta(index);
    if (!delta)
      return std::unexpected(delta.error());
    for (std::size_t axis = 0; axis < 3; ++axis) {
      weighted_delta[axis].add(static_cast<long double>(weight) *
                               static_cast<long double>((*delta)[axis]));
    }
    const std::array<long double, 3> delta_ld{
        static_cast<long double>((*delta)[0]),
        static_cast<long double>((*delta)[1]),
        static_cast<long double>((*delta)[2])};
    const long double distance2 = gram_quadratic(context.gram, delta_ld);
    if (!std::isfinite(distance2))
      return std::unexpected("Phase 1 physical moment is not finite");
    weighted_raw_distance2.add(static_cast<long double>(weight) * distance2);
    if (audit_components)
      audited_deltas.push_back(*delta);
  }
  const long double weight_sum =
      static_cast<long double>(context.total_weight);
  const std::array<double, 3> weighted_mean{
      static_cast<double>(weighted_delta[0].value() / weight_sum),
      static_cast<double>(weighted_delta[1].value() / weight_sum),
      static_cast<double>(weighted_delta[2].value() / weight_sum)};
  const auto nearest = context.metric.nearest_image_scaled(
      weighted_mean[0], weighted_mean[1], weighted_mean[2]);
  if (!nearest.valid)
    return std::unexpected("Phase 1 global-gauge CVP failed");

  Phase1Transition result;
  result.gauge_increment =
      {nearest.image_x, nearest.image_y, nearest.image_z};
  result.gauge_increment_ambiguous = nearest.ambiguous;
  result.component_injectivity_radius_lower_bound_A =
      context.injectivity_radius_lower_bound_A;
  const std::array<long double, 3> gauge{
      static_cast<long double>(result.gauge_increment.x),
      static_cast<long double>(result.gauge_increment.y),
      static_cast<long double>(result.gauge_increment.z)};
  const std::array<long double, 3> weighted_delta_vector{
      weighted_delta[0].value(), weighted_delta[1].value(),
      weighted_delta[2].value()};
  const long double cross_term =
      2.0L * gram_bilinear(context.gram, gauge, weighted_delta_vector);
  const long double gauge_term =
      weight_sum * gram_quadratic(context.gram, gauge);
  long double weighted_residual2 =
      weighted_raw_distance2.value() + cross_term + gauge_term;
  const long double cancellation_numerator =
      std::fabs(weighted_raw_distance2.value()) + std::fabs(cross_term) +
      std::fabs(gauge_term);
  const long double cancellation_denominator = std::max(
      std::fabs(weighted_residual2),
      std::numeric_limits<long double>::epsilon() *
          std::max(1.0L, cancellation_numerator));
  constexpr long double kDirectResidualCancellationRatio = 1.0e10L;
  if (cancellation_numerator / cancellation_denominator >
          kDirectResidualCancellationRatio ||
      weighted_residual2 < 0.0L) {
    KahanSum direct_residual2;
    for (std::size_t index = 0; index < component_count; ++index) {
      std::array<double, 3> delta{};
      if (audit_components) {
        delta = audited_deltas[index];
      } else {
        const auto rebuilt_delta = component_delta(index);
        if (!rebuilt_delta)
          return std::unexpected(rebuilt_delta.error());
        delta = *rebuilt_delta;
      }
      const std::array<long double, 3> residual{
          static_cast<long double>(delta[0]) + gauge[0],
          static_cast<long double>(delta[1]) + gauge[1],
          static_cast<long double>(delta[2]) + gauge[2]};
      direct_residual2.add(
          static_cast<long double>(context.component_weights[index]) *
          gram_quadratic(context.gram, residual));
    }
    weighted_residual2 = direct_residual2.value();
    result.quadratic_direct_residual_fallback = true;
  }
  const long double moment_scale =
      std::max(1.0L, cancellation_numerator);
  if (!std::isfinite(weighted_residual2) ||
      weighted_residual2 < -1.0e-14L * moment_scale) {
    return std::unexpected("Phase 1 quadratic moment became invalid");
  }
  weighted_residual2 = std::max(0.0L, weighted_residual2);
  double maximum_residual2 = 0.0;
  if (audit_components) {
    for (std::size_t index = 0; index < component_count; ++index) {
      const auto &delta = audited_deltas[index];
      const auto cartesian = context.metric.scaled_to_cartesian(
          delta[0] + static_cast<double>(result.gauge_increment.x),
          delta[1] + static_cast<double>(result.gauge_increment.y),
          delta[2] + static_cast<double>(result.gauge_increment.z));
      const double residual2 = cartesian[0] * cartesian[0] +
                               cartesian[1] * cartesian[1] +
                               cartesian[2] * cartesian[2];
      if (!std::isfinite(residual2))
        return std::unexpected("Phase 1 physical residual is not finite");
      maximum_residual2 = std::max(maximum_residual2, residual2);
      const double residual = std::sqrt(std::max(0.0, residual2));
      const double injectivity_tolerance =
          1.0e-12 * std::max(
                         1.0,
                         result.component_injectivity_radius_lower_bound_A);
      if (residual >= result.component_injectivity_radius_lower_bound_A -
                          injectivity_tolerance) {
        ++result.component_injectivity_uncertified;
      }
      const auto component_nearest =
          context.metric.nearest_image_scaled(delta[0], delta[1], delta[2]);
      if (!component_nearest.valid ||
          !context.metric.image_is_nearest_equivalent_scaled(
              delta[0], delta[1], delta[2], result.gauge_increment.x,
              result.gauge_increment.y, result.gauge_increment.z,
              component_nearest)) {
        ++result.component_nearest_image_violations;
      }
    }
  }
  const double mean_residual2 =
      static_cast<double>(weighted_residual2 / weight_sum);
  result.cost =
      config.physical_transition_weight * mean_residual2 / context.variance;
  result.component_displacement_rms_A =
      std::sqrt(std::max(0.0, mean_residual2));
  result.component_displacement_max_A =
      std::sqrt(std::max(0.0, maximum_residual2));
  result.component_injectivity_margin_A =
      result.component_injectivity_radius_lower_bound_A -
      result.component_displacement_max_A;
  if (!std::isfinite(result.cost))
    return std::unexpected("Phase 1 transition cost is not finite");
  return result;
}

std::expected<std::vector<std::array<double, 3>>, std::string>
reconstruct_selected_coordinates(
    const HoloLiftObservationStore &store,
    const HoloLiftPhase1PreparedFrame &prepared_frame,
    const HoloLiftPhase1FrameSelection &selection,
    std::span<const std::uint32_t> component_by_input_atom,
    std::span<const HoloLiftLatticeImage> spatial_images,
    std::span<const std::array<double, 3>> wrapped_coordinates) {
  if (selection.candidate_index >= store.candidates().size() ||
      component_by_input_atom.size() != wrapped_coordinates.size() ||
      spatial_images.size() != wrapped_coordinates.size()) {
    return std::unexpected(
        "Phase 1 reconstruction scratch inputs are invalid");
  }
  const auto &candidate = store.candidates()[selection.candidate_index];
  if (!candidate.component_images.valid_for(store.component_images().size()))
    return std::unexpected(
        "Phase 1 reconstruction component-image range is invalid");
  const auto component_images = store.component_images().subspan(
      candidate.component_images.begin, candidate.component_images.count);
  const titan_pbctopo::PbctopoLatticeMetric lattice(
      prepared_frame.box_matrix());
  if (!lattice.valid())
    return std::unexpected("Phase 1 reconstruction lattice is invalid");

  std::vector<std::array<double, 3>> reconstructed;
  reconstructed.reserve(wrapped_coordinates.size());
  for (std::size_t atom_index = 0; atom_index < wrapped_coordinates.size();
       ++atom_index) {
    const std::uint32_t component = component_by_input_atom[atom_index];
    if (component >= component_images.size())
      return std::unexpected("Phase 1 reconstruction mapping is invalid");
    HoloLiftLatticeImage component_and_gauge;
    HoloLiftLatticeImage total_image;
    if (!checked_image_add(component_images[component],
                           selection.cumulative_global_gauge_input_basis,
                           component_and_gauge) ||
        !checked_image_add(spatial_images[atom_index], component_and_gauge,
                           total_image) ||
        !image_is_exact_binary64(total_image)) {
      return std::unexpected("Phase 1 reconstructed atom image overflows");
    }
    const auto &coordinate = wrapped_coordinates[atom_index];
    const auto scaled = lattice.cartesian_to_scaled(
        coordinate[0], coordinate[1], coordinate[2]);
    const auto cartesian = lattice.scaled_to_cartesian(
        scaled[0] + static_cast<double>(total_image.x),
        scaled[1] + static_cast<double>(total_image.y),
        scaled[2] + static_cast<double>(total_image.z));
    if (!std::all_of(cartesian.begin(), cartesian.end(),
                     [](double value) { return std::isfinite(value); })) {
      return std::unexpected(
          "Phase 1 reconstructed atom coordinate is not finite");
    }
    reconstructed.push_back(cartesian);
  }
  return reconstructed;
}

std::expected<HoloLiftPhase1Segment, std::string> solve_phase1_segment(
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    std::size_t frame_begin, std::size_t frame_end,
    const HoloLiftPhase1Config &config, std::size_t segment_index,
    std::vector<HoloLiftPhase1FrameSelection> &selections) {
  if (frame_begin >= frame_end || frame_end > store.frames().size())
    return std::unexpected("Phase 1 segment range is invalid");
  const auto &first_frame = store.frames()[frame_begin];
  const auto &epoch =
      store.topology_epochs()[first_frame.topology_epoch_index];
  const std::size_t layer_count = frame_end - frame_begin;
  auto transported_frame_metadata = build_transported_segment_frame_metadata(
      store, prepared_frames, frame_begin, frame_end, config);
  if (!transported_frame_metadata)
    return std::unexpected(transported_frame_metadata.error());
  std::vector<Phase1BackpointerLayer> backpointer_layers(layer_count);
  bool retained_bands_complete = true;
  bool bounded_domain_only = true;
  std::size_t peak_layer_state_count = 0;
  std::size_t peak_live_pretransformed_candidate_component_images = 0;
  std::size_t rolling_pretransformed_candidate_layer_builds = 0;
  std::size_t dp_parallel_layers = 0;
  std::size_t dp_serial_layers = 0;
  std::size_t dp_worker_pool_creations = 0;
  std::size_t dp_worker_pool_creation_failures = 0;
  std::size_t dp_max_active_workers = 0;
  const unsigned hardware_concurrency = std::thread::hardware_concurrency();
  const std::size_t hardware_worker_limit =
      hardware_concurrency == 0
          ? 1
          : static_cast<std::size_t>(hardware_concurrency);
  const std::size_t configured_worker_limit =
      std::min(config.dp_worker_count, hardware_worker_limit);
  std::unique_ptr<Phase1DpExecutor> dp_executor;
  bool dp_pool_creation_disabled = false;

  for (std::size_t frame_index = frame_begin; frame_index < frame_end;
       ++frame_index) {
    const auto &frame = store.frames()[frame_index];
    if (frame.topology_epoch_index != first_frame.topology_epoch_index)
      return std::unexpected("Phase 1 segment crosses a topology epoch");
    retained_bands_complete =
        retained_bands_complete && frame.evidence.credible_alternative_set_complete;
    bounded_domain_only =
        bounded_domain_only && frame.search_domain.bounded_domain_only;
  }

  auto previous_states =
      build_phase1_candidate_states(store, first_frame, config);
  if (!previous_states)
    return std::unexpected(previous_states.error());
  if (previous_states->size() >= HOLOLIFT_NO_DENSE_INDEX)
    return std::unexpected("Phase 1 layer exceeds compact backpointer range");
  std::vector<std::array<Phase1PathSlot, 2>> previous_paths(
      previous_states->size());
  peak_layer_state_count = previous_states->size();
  for (std::size_t state = 0; state < previous_states->size(); ++state) {
    auto &slot = previous_paths[state][0];
    slot.cost = (*previous_states)[state].emission_cost;
    slot.optimal_path_count_capped = 1;
    slot.valid = true;
  }
  auto previous_layer = materialize_transported_frame_layer(
      store, transported_frame_metadata->front());
  if (!previous_layer)
    return std::unexpected(previous_layer.error());
  ++rolling_pretransformed_candidate_layer_builds;
  peak_live_pretransformed_candidate_component_images =
      previous_layer->candidate_component_images_reference.size();

  for (std::size_t row = 1; row < layer_count; ++row) {
    const auto &current_frame = store.frames()[frame_begin + row];
    auto current_states =
        build_phase1_candidate_states(store, current_frame, config);
    if (!current_states)
      return std::unexpected(current_states.error());
    if (current_states->size() >= HOLOLIFT_NO_DENSE_INDEX)
      return std::unexpected("Phase 1 layer exceeds compact backpointer range");
    peak_layer_state_count =
        std::max(peak_layer_state_count, current_states->size());
    auto current_layer = materialize_transported_frame_layer(
        store, (*transported_frame_metadata)[row]);
    if (!current_layer)
      return std::unexpected(current_layer.error());
    ++rolling_pretransformed_candidate_layer_builds;
    if (current_layer->candidate_component_images_reference.size() >
        std::numeric_limits<std::size_t>::max() -
            previous_layer->candidate_component_images_reference.size()) {
      return std::unexpected(
          "Phase 1 live pretransformed image counter overflows");
    }
    peak_live_pretransformed_candidate_component_images = std::max(
        peak_live_pretransformed_candidate_component_images,
        previous_layer->candidate_component_images_reference.size() +
            current_layer->candidate_component_images_reference.size());
    auto transition_context = build_phase1_transition_context(
        store, epoch, *previous_layer, *current_layer, config);
    if (!transition_context)
      return std::unexpected(transition_context.error());
    std::vector<std::array<Phase1PathSlot, 2>> current_paths(
        current_states->size());
    const auto evaluate_current_state = [&](std::size_t current_state)
        -> std::optional<std::string> {
      const std::size_t current_candidate_index =
          (*current_states)[current_state].candidate_index;
      for (std::size_t previous_state = 0;
           previous_state < previous_states->size(); ++previous_state) {
        const std::size_t previous_candidate_index =
            (*previous_states)[previous_state].candidate_index;
        const auto transition = phase1_transition_cost(
            store, *transition_context, previous_candidate_index,
            current_candidate_index, config, false);
        if (!transition)
          return transition.error();
        for (std::size_t previous_slot = 0; previous_slot < 2;
             ++previous_slot) {
          const auto &prior = previous_paths[previous_state][previous_slot];
          if (!prior.valid)
            continue;
          Phase1PathSlot candidate;
          candidate.cost = prior.cost + transition->cost +
                           (*current_states)[current_state].emission_cost;
          candidate.predecessor_state = previous_state;
          candidate.predecessor_slot = previous_slot;
          candidate.optimal_path_count_capped =
              prior.optimal_path_count_capped;
          candidate.valid = std::isfinite(candidate.cost);
          if (!candidate.valid)
            return std::string("Phase 1 path cost is not finite");
          insert_path(current_paths[current_state], candidate,
                      config.relative_cost_tolerance);
        }
      }
      return std::nullopt;
    };
    std::vector<std::optional<std::string>> state_errors(
        current_states->size());
    const std::size_t desired_active_workers =
        std::min(configured_worker_limit, current_states->size());
    const std::size_t work_estimate = phase1_dp_work_estimate(
        previous_states->size(), current_states->size(),
        transition_context->base_component_delta.size());
    const bool parallel_layer =
        desired_active_workers > 1 &&
        work_estimate >= kPhase1DpParallelWorkThreshold;
    if (parallel_layer && !dp_executor && !dp_pool_creation_disabled) {
      bool creation_failed = false;
      dp_executor =
          Phase1DpExecutor::create(desired_active_workers, creation_failed);
      if (dp_executor) {
        ++dp_worker_pool_creations;
      } else if (creation_failed) {
        ++dp_worker_pool_creation_failures;
        dp_pool_creation_disabled = true;
      }
    }
    if (!parallel_layer || !dp_executor) {
      ++dp_serial_layers;
      for (std::size_t current_state = 0;
           current_state < current_states->size(); ++current_state) {
        state_errors[current_state] = evaluate_current_state(current_state);
      }
    } else {
      const std::size_t active_workers = std::min(
          dp_executor->capacity(), current_states->size());
      const auto execution = dp_executor->run(
          active_workers, current_states->size(),
          [&](std::size_t current_state) {
            state_errors[current_state] =
                evaluate_current_state(current_state);
          });
      if (!execution) {
        const char *failure_prefix =
            execution.error().kind == Phase1DpExecutor::Failure::Kind::Setup
                ? "Phase 1 DP executor setup failed: "
                : "Phase 1 DP worker task failed: ";
        try {
          std::rethrow_exception(execution.error().exception);
        } catch (const std::exception &error) {
          return std::unexpected(std::string(failure_prefix) + error.what());
        } catch (...) {
          return std::unexpected(std::string(failure_prefix) +
                                 "non-standard exception");
        }
      }
      ++dp_parallel_layers;
      dp_max_active_workers =
          std::max(dp_max_active_workers, active_workers);
    }
    for (const auto &error : state_errors) {
      if (error)
        return std::unexpected(*error);
    }
    auto &compact_layer = backpointer_layers[row];
    compact_layer.paths.resize(current_paths.size());
    for (std::size_t state = 0; state < current_paths.size(); ++state) {
      for (std::size_t slot = 0; slot < 2; ++slot) {
        const auto &source = current_paths[state][slot];
        if (!source.valid)
          continue;
        if (source.predecessor_state >= HOLOLIFT_NO_DENSE_INDEX ||
            source.predecessor_slot >= 2) {
          return std::unexpected("Phase 1 compact backpointer overflow");
        }
        compact_layer.paths[state][slot] = {
            static_cast<std::uint32_t>(source.predecessor_state),
            static_cast<std::uint8_t>(source.predecessor_slot), true};
      }
    }
    *previous_states = std::move(*current_states);
    previous_paths = std::move(current_paths);
    *previous_layer = std::move(*current_layer);
  }
  *previous_layer = Phase1MaterializedFrameLayer{};

  std::array<Phase1PathSlot, 2> final_paths{};
  for (std::size_t state = 0; state < previous_states->size(); ++state) {
    for (std::size_t slot_index = 0; slot_index < 2; ++slot_index) {
      Phase1PathSlot candidate = previous_paths[state][slot_index];
      if (!candidate.valid)
        continue;
      candidate.predecessor_state = state;
      candidate.predecessor_slot = slot_index;
      insert_path(final_paths, candidate, config.relative_cost_tolerance);
    }
  }
  if (!final_paths[0].valid)
    return std::unexpected("Phase 1 segment has no finite temporal path");

  std::vector<std::size_t> selected_state(layer_count, HOLOLIFT_NO_INDEX);
  std::vector<std::size_t> selected_slot(layer_count, HOLOLIFT_NO_INDEX);
  selected_state.back() = final_paths[0].predecessor_state;
  selected_slot.back() = final_paths[0].predecessor_slot;
  for (std::size_t row = layer_count - 1; row > 0; --row) {
    const auto &path =
        backpointer_layers[row].paths[selected_state[row]][selected_slot[row]];
    if (!path.valid || path.predecessor_state == HOLOLIFT_NO_DENSE_INDEX ||
        path.predecessor_slot >= 2) {
      return std::unexpected("Phase 1 backpointer chain is incomplete");
    }
    selected_state[row - 1] = path.predecessor_state;
    selected_slot[row - 1] = path.predecessor_slot;
  }

  HoloLiftPhase1Segment segment;
  segment.frame_begin = frame_begin;
  segment.frame_count = frame_end - frame_begin;
  segment.topology_epoch_index = first_frame.topology_epoch_index;
  segment.component_weighting =
      transported_frame_metadata->front().source->component_weighting();
  segment.objective = final_paths[0].cost;
  segment.optimal_path_count_capped =
      final_paths[0].optimal_path_count_capped;
  segment.unique_on_configured_retained_state_graph =
      segment.optimal_path_count_capped == 1;
  segment.retained_candidate_bands_complete = retained_bands_complete;
  segment.bounded_domain_only = bounded_domain_only;
  segment.optimal_path_proven_on_complete_bounded_domain =
      segment.exact_on_retained_graph && retained_bands_complete &&
      bounded_domain_only;
  segment.peak_layer_state_count = peak_layer_state_count;
  segment.peak_live_pretransformed_candidate_component_images =
      peak_live_pretransformed_candidate_component_images;
  if (peak_live_pretransformed_candidate_component_images >
      std::numeric_limits<std::size_t>::max() /
          sizeof(HoloLiftLatticeImage)) {
    return std::unexpected(
        "Phase 1 pretransformed image byte estimate overflows");
  }
  segment.peak_live_pretransformed_candidate_component_image_bytes_estimate =
      peak_live_pretransformed_candidate_component_images *
      sizeof(HoloLiftLatticeImage);
  segment.rolling_pretransformed_candidate_layer_builds =
      rolling_pretransformed_candidate_layer_builds;
  segment.dp_parallel_layers = dp_parallel_layers;
  segment.dp_serial_layers = dp_serial_layers;
  segment.dp_worker_pool_creations = dp_worker_pool_creations;
  segment.dp_worker_pool_creation_failures =
      dp_worker_pool_creation_failures;
  segment.dp_max_active_workers = dp_max_active_workers;
  segment.lattice_basis_transport_accepted_under_policy = true;
  segment.lattice_basis_transport_provenance_sufficient = true;
  segment.local_transport_search_completed = true;
  for (std::size_t row = 0; row < transported_frame_metadata->size(); ++row) {
    const auto &frame = (*transported_frame_metadata)[row];
    if (frame.candidate_count >
        std::numeric_limits<std::size_t>::max() /
            std::max<std::size_t>(1, frame.component_count)) {
      return std::unexpected(
          "Phase 1 logical pretransformed image counter overflows");
    }
    const std::size_t logical_image_count =
        frame.candidate_count * frame.component_count;
    if (logical_image_count >
        std::numeric_limits<std::size_t>::max() -
            segment.pretransformed_candidate_component_images) {
      return std::unexpected(
          "Phase 1 pretransformed image counter overflows");
    }
    segment.pretransformed_candidate_component_images +=
        logical_image_count;
    if (frame.transport.applied)
      ++segment.lattice_basis_transports_applied;
    segment.maximum_lattice_basis_transport_relative_mismatch = std::max(
        segment.maximum_lattice_basis_transport_relative_mismatch,
        frame.transport.relative_mismatch);
    segment.lattice_basis_transport_accepted_under_policy &=
        frame.transport.accepted_under_policy && !frame.transport.ambiguous;
    segment.lattice_basis_transport_provenance_sufficient &=
        frame.transport.provenance_sufficient_for_certification;
    if (row != 0) {
      segment.local_transport_search_completed &=
          frame.transport.local_search_completed;
    }
    segment.explicit_remap_provenance_available |=
        frame.transport.explicit_remap_provenance_available;
  }
  for (const auto &layer : backpointer_layers) {
    segment.compact_backpointer_bytes_estimate +=
        layer.paths.size() *
        sizeof(std::array<Phase1CompactBackpointer, 2>);
  }
  if (segment.optimal_path_count_capped >= 2) {
    segment.second_best_objective = segment.objective;
  } else if (final_paths[1].valid) {
    segment.second_best_objective = final_paths[1].cost;
  }
  if (std::isfinite(segment.second_best_objective)) {
    segment.absolute_path_gap =
        std::max(0.0, segment.second_best_objective - segment.objective);
    segment.relative_path_gap =
        segment.absolute_path_gap / std::max(1.0, std::fabs(segment.objective));
  }

  HoloLiftLatticeImage cumulative_gauge{};
  bool component_injectivity_ok = true;
  bool time_reversal_consistent = true;
  double selected_path_cost = 0.0;
  std::optional<Phase1MaterializedFrameLayer> previous_selected_layer;
  for (std::size_t row = 0; row < layer_count; ++row) {
    const std::size_t frame_index = frame_begin + row;
    auto row_states = build_phase1_candidate_states(
        store, store.frames()[frame_index], config);
    if (!row_states || selected_state[row] >= row_states->size()) {
      return std::unexpected(row_states
                                 ? "Phase 1 selected state is out of range"
                                 : row_states.error());
    }
    const auto &state = (*row_states)[selected_state[row]];
    auto &selection = selections[frame_index];
    selection.use = HoloLiftPhase1FrameUse::TemporalObservation;
    selection.segment_index = segment_index;
    selection.candidate_index = state.candidate_index;
    selection.selected_candidate_provider_unsupported =
        !store.candidates()[state.candidate_index].evidence.hard_feasible;
    selection.preceding_transition_provider_unsupported =
        row > 0 &&
        (selection.selected_candidate_provider_unsupported ||
         selections[frame_index - 1].selected_candidate_provider_unsupported);
    segment.provider_unsupported_selected_candidate_count +=
        selection.selected_candidate_provider_unsupported ? 1 : 0;
    segment.provider_unsupported_transition_count +=
        selection.preceding_transition_provider_unsupported ? 1 : 0;
    selection.emission_cost = state.emission_cost;
    selection.pinned_strong_bounded_anchor = state.pinned_anchor;
    const auto &transported_frame = (*transported_frame_metadata)[row];
    auto current_selected_layer = materialize_selected_transported_frame_layer(
        store, transported_frame, state.candidate_index);
    if (!current_selected_layer)
      return std::unexpected(current_selected_layer.error());
    selection.lattice_basis_transport_to_segment_reference =
        transported_frame.transport.to_reference;
    selection.lattice_basis_transport_relative_mismatch =
        transported_frame.transport.relative_mismatch;
    selection.lattice_basis_transport_identity_relative_mismatch =
        transported_frame.transport.identity_relative_mismatch;
    selection.lattice_basis_transport_improvement_ratio =
        transported_frame.transport.improvement_ratio;
    selection.lattice_basis_transport_applied =
        transported_frame.transport.applied;
    selection.lattice_basis_transport_ambiguous =
        transported_frame.transport.ambiguous;
    selection.lattice_basis_transport_identity_fast_path =
        transported_frame.transport.identity_fast_path;
    selection.lattice_basis_transport_local_search_completed =
        transported_frame.transport.local_search_completed;
    selection.lattice_basis_transport_accepted_under_policy =
        transported_frame.transport.accepted_under_policy;
    selection.lattice_basis_transport_explicit_remap_provenance_available =
        transported_frame.transport.explicit_remap_provenance_available;
    selection.lattice_basis_transport_provenance_sufficient =
        transported_frame.transport.provenance_sufficient_for_certification;
    selection.changed_from_framewise =
        state.candidate_index !=
        store.frames()[frame_index].selected_candidate_index;
    segment.anchors_hard_pinned =
        segment.anchors_hard_pinned || state.pinned_anchor;
    selected_path_cost += state.emission_cost;
    if (row == 0) {
      selection.cumulative_global_gauge = cumulative_gauge;
      const auto input_gauge = hololift_lattice_image_from_reference(
          cumulative_gauge, transported_frame.transport);
      if (!input_gauge)
        return std::unexpected(input_gauge.error());
      selection.cumulative_global_gauge_input_basis = *input_gauge;
      selection.cumulative_segment_cost = selected_path_cost;
      previous_selected_layer = std::move(*current_selected_layer);
      continue;
    }

    if (!previous_selected_layer) {
      return std::unexpected(
          "Phase 1 selected rolling layer chain is incomplete");
    }
    const auto selected_transition_context = build_phase1_transition_context(
        store, epoch, *previous_selected_layer, *current_selected_layer,
        config);
    if (!selected_transition_context)
      return std::unexpected(selected_transition_context.error());
    const auto audit = phase1_transition_cost(
        store, *selected_transition_context,
        selections[frame_index - 1].candidate_index,
        selection.candidate_index, config, true);
    if (!audit)
      return std::unexpected(audit.error());
    selection.transition_cost = audit->cost;
    selected_path_cost += audit->cost;
    selection.cumulative_segment_cost = selected_path_cost;
    selection.selected_global_gauge_increment = audit->gauge_increment;
    if (!checked_image_add(cumulative_gauge, audit->gauge_increment,
                           cumulative_gauge)) {
      return std::unexpected("Phase 1 cumulative global gauge overflows");
    }
    selection.cumulative_global_gauge = cumulative_gauge;
    const auto input_gauge = hololift_lattice_image_from_reference(
        cumulative_gauge, transported_frame.transport);
    if (!input_gauge)
      return std::unexpected(input_gauge.error());
    selection.cumulative_global_gauge_input_basis = *input_gauge;
    selection.component_displacement_rms_A =
        audit->component_displacement_rms_A;
    selection.component_displacement_max_A =
        audit->component_displacement_max_A;
    selection.component_injectivity_radius_lower_bound_A =
        audit->component_injectivity_radius_lower_bound_A;
    selection.component_injectivity_margin_A =
        audit->component_injectivity_margin_A;
    selection.component_nearest_image_violations =
        audit->component_nearest_image_violations;
    selection.component_injectivity_uncertified =
        audit->component_injectivity_uncertified;
    selection.global_gauge_increment_ambiguous =
        audit->gauge_increment_ambiguous;
    selection.quadratic_direct_residual_fallback =
        audit->quadratic_direct_residual_fallback;
    if (audit->quadratic_direct_residual_fallback)
      ++segment.quadratic_direct_residual_fallbacks;
    ++segment.temporal_transition_count;
    segment.component_nearest_image_violations +=
        audit->component_nearest_image_violations;
    segment.component_injectivity_uncertified +=
        audit->component_injectivity_uncertified;
    if (audit->gauge_increment_ambiguous)
      ++segment.ambiguous_gauge_transition_count;
    component_injectivity_ok =
        component_injectivity_ok && !audit->gauge_increment_ambiguous &&
        audit->component_nearest_image_violations == 0 &&
        audit->component_injectivity_uncertified == 0;

    const std::size_t previous_candidate_index =
        selections[frame_index - 1].candidate_index;
    const std::size_t current_candidate_index = selection.candidate_index;
    const auto reverse_context = build_phase1_transition_context(
        store, epoch, *current_selected_layer, *previous_selected_layer,
        config);
    const auto reverse = reverse_context
                             ? phase1_transition_cost(
                                   store, *reverse_context,
                                   current_candidate_index,
                                   previous_candidate_index, config, false)
                             : std::expected<Phase1Transition, std::string>(
                                   std::unexpected(reverse_context.error()));
    HoloLiftLatticeImage expected_reverse_gauge;
    if (!reverse || !checked_image_negate(audit->gauge_increment,
                                          expected_reverse_gauge) ||
        reverse->gauge_increment != expected_reverse_gauge ||
        reverse->gauge_increment_ambiguous !=
            audit->gauge_increment_ambiguous ||
        !phase1_cost_equal(reverse->cost, audit->cost,
                           config.relative_cost_tolerance)) {
      time_reversal_consistent = false;
    }

    const auto previous_images = transported_candidate_images(
        *previous_selected_layer, previous_candidate_index);
    const auto current_images = transported_candidate_images(
        *current_selected_layer, current_candidate_index);
    if (previous_images.size() != current_images.size() ||
        current_images.size() != epoch.components.count) {
      return std::unexpected(
          "Phase 1 selected pretransformed image range is invalid");
    }
    for (std::size_t component = 0; component < current_images.size();
         ++component) {
      HoloLiftLatticeImage previous_total;
      HoloLiftLatticeImage current_total;
      HoloLiftLatticeImage observed_delta;
      HoloLiftLatticeImage candidate_delta;
      HoloLiftLatticeImage expected_delta;
      const bool closed =
          checked_image_add(previous_images[component],
                            selections[frame_index - 1]
                                .cumulative_global_gauge,
                            previous_total) &&
          checked_image_add(current_images[component],
                            selection.cumulative_global_gauge,
                            current_total) &&
          checked_image_subtract(current_total, previous_total,
                                 observed_delta) &&
          checked_image_subtract(current_images[component],
                                 previous_images[component],
                                 candidate_delta) &&
          checked_image_add(candidate_delta, audit->gauge_increment,
                            expected_delta) &&
          observed_delta == expected_delta;
      if (!closed)
        ++segment.gauge_replay_arithmetic_residuals;
    }
    previous_selected_layer = std::move(*current_selected_layer);
  }
  if (!phase1_cost_equal(selected_path_cost, segment.objective,
                         config.relative_cost_tolerance)) {
    return std::unexpected("Phase 1 compact backpointer objective replay failed");
  }
  segment.all_global_gauge_steps_solved = true;
  segment.coordinates_materialized_and_replay_consistent = false;
  segment.temporal_lift_locally_certified = false;
  segment.component_representative_injectivity_audit_passed =
      component_injectivity_ok;
  segment.gauge_replay_arithmetic_closed =
      segment.gauge_replay_arithmetic_residuals == 0;
  segment.selected_transition_time_reversal_audited =
      segment.temporal_transition_count > 0;
  segment.selected_transition_time_reversal_consistent =
      segment.selected_transition_time_reversal_audited &&
      time_reversal_consistent;
  return segment;
}

} // namespace

HoloLiftPhase1PreparationContext::HoloLiftPhase1PreparationContext(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

HoloLiftPhase1PreparationContext::HoloLiftPhase1PreparationContext(
    HoloLiftPhase1PreparationContext &&) noexcept = default;

HoloLiftPhase1PreparationContext &
HoloLiftPhase1PreparationContext::operator=(
    HoloLiftPhase1PreparationContext &&) noexcept = default;

HoloLiftPhase1PreparationContext::~HoloLiftPhase1PreparationContext() = default;

std::expected<HoloLiftPhase1PreparationContext, std::string>
make_hololift_phase1_preparation_context(
    const HoloLiftObservationStore &store) {
  auto implementation = std::make_unique<HoloLiftPhase1PreparationContext::Impl>();
  implementation->store = &store;
  implementation->epochs.resize(store.topology_epochs().size());
  for (std::size_t epoch_index = 0;
       epoch_index < store.topology_epochs().size(); ++epoch_index) {
    const auto &epoch = store.topology_epochs()[epoch_index];
    if (!epoch.components.valid_for(store.components().size()))
      return std::unexpected("Phase 1 topology component range is invalid");
    auto &cache = implementation->epochs[epoch_index];
    cache.topology_epoch_identity = epoch.topology_epoch_identity;
    cache.component_by_atom.reserve(epoch.atom_count);
    const auto components = store.components().subspan(
        epoch.components.begin, epoch.components.count);
    const auto atom_masses = store.exact_atom_masses();
    bool use_atomic_mass =
        atom_masses.size() == store.exact_atom_memberships().size();
    if (use_atomic_mass) {
      for (const auto &component : components) {
        if (!component.exact_atom_membership.valid_for(atom_masses.size())) {
          return std::unexpected(
              "Phase 1 component mass range is invalid");
        }
        const auto masses = atom_masses.subspan(
            component.exact_atom_membership.begin,
            component.exact_atom_membership.count);
        if (std::any_of(masses.begin(), masses.end(), [](double mass) {
              return !std::isfinite(mass) || mass <= 0.0;
            })) {
          use_atomic_mass = false;
          break;
        }
      }
    }
    cache.component_weighting =
        use_atomic_mass ? HoloLiftPhase1ComponentWeighting::AtomicMass
                        : HoloLiftPhase1ComponentWeighting::AtomCount;
    cache.component_weights.reserve(components.size());
    cache.component_atom_counts.reserve(components.size());
    for (std::size_t local_component = 0;
         local_component < components.size(); ++local_component) {
      const auto &component = components[local_component];
      if (!component.exact_atom_membership.valid_for(
              store.exact_atom_memberships().size()) ||
          component.exact_atom_membership.count == 0) {
        return std::unexpected("Phase 1 component membership is invalid");
      }
      cache.component_atom_counts.push_back(
          component.exact_atom_membership.count);
      const auto members = store.exact_atom_memberships().subspan(
          component.exact_atom_membership.begin,
          component.exact_atom_membership.count);
      long double component_weight = 0.0L;
      for (std::size_t member_index = 0; member_index < members.size();
           ++member_index) {
        const auto &atom = members[member_index];
        const double atom_weight =
            use_atomic_mass
                ? atom_masses[component.exact_atom_membership.begin +
                              member_index]
                : 1.0;
        component_weight += static_cast<long double>(atom_weight);
        if (!cache.component_by_atom
                 .emplace(atom, static_cast<std::uint32_t>(local_component))
                 .second) {
          return std::unexpected(
              "Phase 1 topology epoch contains duplicate atom membership");
        }
        if (!cache.weight_by_atom.emplace(atom, atom_weight).second) {
          return std::unexpected(
              "Phase 1 topology epoch contains duplicate atom weights");
        }
      }
      const double stored_weight = static_cast<double>(component_weight);
      if (!std::isfinite(stored_weight) || stored_weight <= 0.0)
        return std::unexpected("Phase 1 component weight is invalid");
      cache.component_weights.push_back(stored_weight);
    }
    if (cache.component_by_atom.size() != epoch.atom_count)
      return std::unexpected("Phase 1 topology epoch atom count mismatch");
    if (epoch.atom_count >= HOLOLIFT_NO_DENSE_INDEX)
      return std::unexpected(
          "Phase 1 topology epoch exceeds canonical dense-index range");
    cache.canonical_atoms.reserve(cache.component_by_atom.size());
    for (const auto &[atom, component] : cache.component_by_atom) {
      static_cast<void>(component);
      cache.canonical_atoms.push_back(atom);
    }
    std::sort(cache.canonical_atoms.begin(), cache.canonical_atoms.end(),
              [](const auto &lhs, const auto &rhs) {
                return std::tie(lhs.owner, lhs.source_atom_id) <
                       std::tie(rhs.owner, rhs.source_atom_id);
              });
    cache.canonical_dense_by_atom.reserve(cache.canonical_atoms.size());
    for (std::size_t dense_index = 0;
         dense_index < cache.canonical_atoms.size(); ++dense_index) {
      cache.canonical_dense_by_atom.emplace(
          cache.canonical_atoms[dense_index],
          static_cast<std::uint32_t>(dense_index));
    }
  }
  return HoloLiftPhase1PreparationContext(std::move(implementation));
}

std::expected<HoloLiftPhase1PreparedFrame, std::string>
prepare_hololift_phase1_frame(
    HoloLiftPhase1PreparationContext &context,
    const HoloLiftObservationStore &store, std::size_t frame_index,
    std::size_t trajectory_frame_index,
    std::span<const double, 9> box_matrix,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates) {
  if (!context.impl_ || context.impl_->store != &store)
    return std::unexpected("Phase 1 preparation context/store mismatch");
  if (frame_index >= store.frames().size())
    return std::unexpected("Phase 1 frame index is out of range");
  const auto &frame = store.frames()[frame_index];
  if (frame.topology_epoch_index >= store.topology_epochs().size())
    return std::unexpected("Phase 1 frame topology epoch is invalid");
  const auto &epoch = store.topology_epochs()[frame.topology_epoch_index];
  if (!epoch.hard_edges.valid_for(store.hard_edges().size()))
    return std::unexpected("Phase 1 topology hard-edge range is invalid");
  const auto hard_edges = store.hard_edges().subspan(
      epoch.hard_edges.begin, epoch.hard_edges.count);
  const auto binding = validate_hololift_frame_binding(
      epoch, frame, trajectory_frame_index, box_matrix, atom_order,
      wrapped_coordinates);
  if (!binding)
    return std::unexpected("Phase 1 frame binding failed: " + binding.error());
  const auto spatial = replay_hololift_spatial_lift(
      epoch, frame, hard_edges, atom_order, wrapped_coordinates);
  if (!spatial)
    return std::unexpected("Phase 1 spatial replay failed: " + spatial.error());
  if (spatial->identity != frame.spatial_lift_identity ||
      spatial->ambiguous_hard_edges !=
          frame.spatial_lift_ambiguous_hard_edges ||
      spatial->cycle_residuals != frame.spatial_lift_cycle_residuals) {
    return std::unexpected(
        "Phase 1 spatial replay disagrees with observation metadata");
  }
  const auto mapping = atom_mapping(*context.impl_, store, frame, atom_order);
  if (!mapping)
    return std::unexpected(mapping.error());
  auto &cache = context.impl_->epochs[frame.topology_epoch_index];
  if (cache.component_weights.empty())
    return std::unexpected("Phase 1 topology epoch has no components");

  const titan_pbctopo::PbctopoLatticeMetric lattice(box_matrix);
  if (!lattice.valid())
    return std::unexpected("Phase 1 prepared-frame lattice is invalid");

  std::vector<std::array<long double, 3>> sums(cache.component_weights.size());
  std::vector<std::size_t> counts(cache.component_weights.size(), 0);
  for (std::size_t canonical_index = 0;
       canonical_index < mapping->input_index_by_canonical_dense_atom.size();
       ++canonical_index) {
    const std::size_t atom_index =
        mapping->input_index_by_canonical_dense_atom[canonical_index];
    if (atom_index >= atom_order.size() ||
        atom_index >= mapping->component_by_input_atom.size() ||
        atom_index >= mapping->weight_by_input_atom.size()) {
      return std::unexpected(
          "Phase 1 canonical atom mapping is invalid");
    }
    const std::uint32_t component =
        mapping->component_by_input_atom[atom_index];
    if (component >= sums.size() || atom_index >= spatial->atom_images.size())
      return std::unexpected("Phase 1 spatial/component mapping is invalid");
    const long double atom_weight =
        static_cast<long double>(mapping->weight_by_input_atom[atom_index]);
    const auto &coordinate = wrapped_coordinates[atom_index];
    const auto scaled = lattice.cartesian_to_scaled(
        coordinate[0], coordinate[1], coordinate[2]);
    const auto &image = spatial->atom_images[atom_index];
    sums[component][0] +=
        atom_weight * (static_cast<long double>(scaled[0]) +
                       static_cast<long double>(image.x));
    sums[component][1] +=
        atom_weight * (static_cast<long double>(scaled[1]) +
                       static_cast<long double>(image.y));
    sums[component][2] +=
        atom_weight * (static_cast<long double>(scaled[2]) +
                       static_cast<long double>(image.z));
    ++counts[component];
  }

  std::vector<std::array<double, 3>> representatives(sums.size());
  for (std::size_t component = 0; component < sums.size(); ++component) {
    if (counts[component] != cache.component_atom_counts[component]) {
      return std::unexpected("Phase 1 component atom count mismatch");
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      representatives[component][axis] = static_cast<double>(
          sums[component][axis] /
          static_cast<long double>(cache.component_weights[component]));
      if (!std::isfinite(representatives[component][axis]))
        return std::unexpected(
            "Phase 1 component representative is not finite");
    }
  }
  std::array<double, 9> stored_box{};
  std::copy(box_matrix.begin(), box_matrix.end(), stored_box.begin());
  return HoloLiftPhase1PreparedFrame(
      frame_index, trajectory_frame_index, frame.time_ps,
      frame.topology_epoch_identity, frame.wrapped_coordinate_hash,
      frame.spatial_lift_identity, stored_box, physical_gram(box_matrix),
      std::move(representatives), cache.component_weights,
      cache.component_weighting);
}

std::expected<HoloLiftPhase1VerifiedFrame, std::string>
verify_hololift_phase1_frame(
    const HoloLiftObservationStore &store, std::size_t frame_index,
    std::size_t trajectory_frame_index,
    std::span<const double, 9> box_matrix,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates) {
  auto context = make_hololift_phase1_preparation_context(store);
  if (!context)
    return std::unexpected(context.error());
  return prepare_hololift_phase1_frame(
      *context, store, frame_index, trajectory_frame_index, box_matrix,
      atom_order, wrapped_coordinates);
}

std::expected<HoloLiftPhase1Result, std::string>
solve_hololift_phase1_temporal_path(
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    const HoloLiftPhase1Config &config) {
  if (!std::isfinite(config.ordinal_rank_weight) ||
      config.ordinal_rank_weight < 0.0 ||
      !std::isfinite(config.unique_evidence_mismatch_weight) ||
      config.unique_evidence_mismatch_weight < 0.0 ||
      !std::isfinite(config.supported_evidence_mismatch_weight) ||
      config.supported_evidence_mismatch_weight < 0.0 ||
      !std::isfinite(config.physical_transition_weight) ||
      config.physical_transition_weight < 0.0 ||
      !std::isfinite(config.temporal_scale) || config.temporal_scale <= 0.0 ||
      !std::isfinite(config.variance_floor_A2) ||
      config.variance_floor_A2 < 0.0 ||
      !std::isfinite(config.bounded_anchor_competitor_penalty) ||
      config.bounded_anchor_competitor_penalty < 0.0 ||
      !std::isfinite(config.relative_cost_tolerance) ||
      config.relative_cost_tolerance <= 0.0 ||
      config.relative_cost_tolerance > 1.0e-3 ||
      !std::isfinite(config.basis_transport_max_relative_mismatch) ||
      config.basis_transport_max_relative_mismatch <= 0.0 ||
      !std::isfinite(config.basis_transport_ambiguity_tolerance) ||
      config.basis_transport_ambiguity_tolerance <= 0.0 ||
      !std::isfinite(
          config.basis_transport_nonidentity_max_relative_mismatch) ||
      config.basis_transport_nonidentity_max_relative_mismatch <= 0.0 ||
      !std::isfinite(config.basis_transport_minimum_improvement_ratio) ||
      config.basis_transport_minimum_improvement_ratio <= 1.0 ||
      config.dp_worker_count == 0 || config.dp_worker_count > 1024 ||
      (config.lattice_basis_policy !=
           HoloLiftLatticeBasisPolicy::RequireBasisContinuous &&
       config.lattice_basis_policy !=
           HoloLiftLatticeBasisPolicy::TransportUnimodularBasis) ||
      config.loss != HoloLiftPhase1Loss::Quadratic) {
    return std::unexpected("Phase 1 configuration is invalid");
  }
  if (prepared_frames.size() != store.frames().size()) {
    return std::unexpected(
        "Phase 1 requires one prepared trajectory binding per frame");
  }
  for (std::size_t index = 0; index < prepared_frames.size(); ++index) {
    const auto &frame = store.frames()[index];
    const auto &prepared = prepared_frames[index];
    if (prepared.frame_index() != index ||
        prepared.trajectory_frame_index() != frame.trajectory_frame_index ||
        prepared.time_ps() != frame.time_ps ||
        prepared.topology_epoch_identity() != frame.topology_epoch_identity ||
        prepared.wrapped_coordinate_hash() != frame.wrapped_coordinate_hash ||
        prepared.spatial_lift_identity() != frame.spatial_lift_identity ||
        prepared.component_fractional_representatives().size() !=
            store.topology_epochs()[frame.topology_epoch_index]
                .components.count) {
      return std::unexpected(
          "Phase 1 prepared frame does not match observation store");
    }
  }

  HoloLiftPhase1Result result;
  result.config = config;
  result.frames.resize(store.frames().size());
  result.all_observation_bands_complete = true;
  result.all_segment_objectives_unique = true;
  result.certificate.all_frames_trajectory_bound = true;
  result.certificate.all_spatial_lifts_replayed = true;

  std::size_t frame_index = 0;
  while (frame_index < store.frames().size()) {
    const auto use = hololift_phase1_frame_use(store.frames()[frame_index]);
    result.frames[frame_index].use = use;
    if (use != HoloLiftPhase1FrameUse::TemporalObservation) {
      ++result.observation_gap_frames;
      ++frame_index;
      continue;
    }

    const std::size_t segment_begin = frame_index;
    const std::uint32_t epoch_index =
        store.frames()[frame_index].topology_epoch_index;
    std::array<double, 9> previous_reference_box{};
    std::copy(prepared_frames[frame_index].box_matrix().begin(),
              prepared_frames[frame_index].box_matrix().end(),
              previous_reference_box.begin());
    ++frame_index;
    while (frame_index < store.frames().size()) {
      const auto &frame = store.frames()[frame_index];
      const auto &previous = store.frames()[frame_index - 1];
      if (hololift_phase1_frame_use(frame) !=
              HoloLiftPhase1FrameUse::TemporalObservation ||
          frame.source_relation !=
              HoloLiftSourceFrameRelation::ContiguousSourceFrame ||
          frame.topology_epoch_index != epoch_index ||
          frame.time_ps <= previous.time_ps) {
        break;
      }
      const auto transport = evaluate_hololift_lattice_basis_transport(
          previous_reference_box, prepared_frames[frame_index].box_matrix(),
          config.lattice_basis_policy,
          config.basis_transport_max_relative_mismatch,
          config.basis_transport_ambiguity_tolerance,
          config.basis_transport_nonidentity_max_relative_mismatch,
          config.basis_transport_minimum_improvement_ratio);
      if (!transport) {
        const auto gap = transition_gap_from_transport_failure(
            transport.error().failure);
        result.frames[frame_index].preceding_transition_gap = gap;
        ++result.basis_transport_transition_gaps;
        if (gap ==
            HoloLiftPhase1TransitionGap::AmbiguousBasisTransport) {
          ++result.ambiguous_basis_transport_transition_gaps;
        } else if (gap ==
                   HoloLiftPhase1TransitionGap::BasisDiscontinuity) {
          ++result.basis_discontinuity_transition_gaps;
        } else {
          ++result.basis_mismatch_transition_gaps;
        }
        break;
      }
      previous_reference_box = transport->transported_box;
      ++frame_index;
    }

    const std::size_t segment_index = result.segments.size();
    auto segment = solve_phase1_segment(
        store, prepared_frames, segment_begin, frame_index, config,
        segment_index, result.frames);
    if (!segment)
      return std::unexpected(segment.error());
    result.objective += segment->objective;
    result.temporal_observation_frames += segment->frame_count;
    result.all_segment_objectives_unique =
        result.all_segment_objectives_unique &&
        segment->unique_on_configured_retained_state_graph;
    result.all_observation_bands_complete =
        result.all_observation_bands_complete &&
        segment->retained_candidate_bands_complete;
    result.segments.push_back(*segment);
  }

  bool all_selected_hard_feasible = true;
  bool all_gauge_steps_solved = true;
  bool all_basis_transports_accepted = true;
  bool all_basis_transport_provenance_sufficient = true;
  bool all_local_transport_searches_completed = true;
  bool any_explicit_remap_provenance = false;
  bool all_component_injectivity_audits_passed = true;
  bool all_gauge_replay_arithmetic_closed = true;
  bool all_selected_transition_reversal_audited = true;
  bool all_selected_transition_reversal_consistent = true;
  bool all_segments_bounded_domain_only = true;
  for (const auto &selection : result.frames) {
    if (selection.changed_from_framewise)
      ++result.changed_from_framewise_frames;
    if (selection.use != HoloLiftPhase1FrameUse::TemporalObservation)
      continue;
    all_selected_hard_feasible =
        all_selected_hard_feasible &&
        selection.candidate_index < store.candidates().size() &&
        store.candidates()[selection.candidate_index].evidence.hard_feasible;
  }
  for (const auto &segment : result.segments) {
    result.quadratic_direct_residual_fallbacks +=
        segment.quadratic_direct_residual_fallbacks;
    result.pretransformed_candidate_component_images +=
        segment.pretransformed_candidate_component_images;
    result.peak_live_pretransformed_candidate_component_images = std::max(
        result.peak_live_pretransformed_candidate_component_images,
        segment.peak_live_pretransformed_candidate_component_images);
    result
        .peak_live_pretransformed_candidate_component_image_bytes_estimate =
        std::max(
            result
                .peak_live_pretransformed_candidate_component_image_bytes_estimate,
            segment
                .peak_live_pretransformed_candidate_component_image_bytes_estimate);
    result.rolling_pretransformed_candidate_layer_builds +=
        segment.rolling_pretransformed_candidate_layer_builds;
    if (!audit_count_add(result.dp_parallel_layers,
                         segment.dp_parallel_layers) ||
        !audit_count_add(result.dp_serial_layers, segment.dp_serial_layers) ||
        !audit_count_add(result.dp_worker_pool_creations,
                         segment.dp_worker_pool_creations) ||
        !audit_count_add(result.dp_worker_pool_creation_failures,
                         segment.dp_worker_pool_creation_failures)) {
      return std::unexpected("Phase 1 DP executor telemetry overflows");
    }
    result.dp_max_active_workers =
        std::max(result.dp_max_active_workers, segment.dp_max_active_workers);
    all_gauge_steps_solved =
        all_gauge_steps_solved && segment.all_global_gauge_steps_solved;
    all_basis_transports_accepted =
        all_basis_transports_accepted &&
        segment.lattice_basis_transport_accepted_under_policy;
    all_basis_transport_provenance_sufficient =
        all_basis_transport_provenance_sufficient &&
        segment.lattice_basis_transport_provenance_sufficient;
    all_local_transport_searches_completed =
        all_local_transport_searches_completed &&
        segment.local_transport_search_completed;
    any_explicit_remap_provenance =
        any_explicit_remap_provenance ||
        segment.explicit_remap_provenance_available;
    all_component_injectivity_audits_passed =
        all_component_injectivity_audits_passed &&
        segment.component_representative_injectivity_audit_passed;
    all_gauge_replay_arithmetic_closed =
        all_gauge_replay_arithmetic_closed &&
        segment.gauge_replay_arithmetic_closed;
    all_selected_transition_reversal_audited =
        all_selected_transition_reversal_audited &&
        segment.selected_transition_time_reversal_audited;
    all_selected_transition_reversal_consistent =
        all_selected_transition_reversal_consistent &&
        segment.selected_transition_time_reversal_consistent;
    all_segments_bounded_domain_only =
        all_segments_bounded_domain_only && segment.bounded_domain_only;
  }

  const auto &coverage = store.source_coverage();
  result.source_frame_count = coverage.source_frame_count;
  result.imported_observation_frame_count = store.frames().size();
  result.imported_observation_coverage_complete =
      result.observation_gap_frames == 0 &&
      result.temporal_observation_frames == store.frames().size();
  result.source_trajectory_coverage_known = coverage.known();
  result.source_trajectory_coverage_complete =
      coverage.complete() && result.imported_observation_coverage_complete;
  result.temporal_transition_coverage_complete =
      result.source_trajectory_coverage_complete &&
      result.segments.size() == 1 && !result.segments.empty();
  result.temporal_coverage_complete =
      result.temporal_transition_coverage_complete;
  if (result.segments.empty()) {
    result.all_segment_objectives_unique = false;
    result.all_observation_bands_complete = false;
    all_gauge_steps_solved = false;
    all_basis_transports_accepted = false;
    all_basis_transport_provenance_sufficient = false;
    all_local_transport_searches_completed = false;
    any_explicit_remap_provenance = false;
    all_component_injectivity_audits_passed = false;
    all_gauge_replay_arithmetic_closed = false;
    all_selected_transition_reversal_audited = false;
    all_selected_transition_reversal_consistent = false;
    all_segments_bounded_domain_only = false;
    all_selected_hard_feasible = false;
  }
  result.exact_on_retained_graph =
      !result.segments.empty() &&
      std::all_of(result.segments.begin(), result.segments.end(),
                  [](const auto &segment) {
                    return segment.exact_on_retained_graph;
                  });

  result.certificate.all_selected_candidates_hard_feasible =
      all_selected_hard_feasible;
  result.certificate.all_global_gauge_steps_solved = all_gauge_steps_solved;
  result.certificate.lattice_basis_transport_accepted_under_policy =
      all_basis_transports_accepted;
  result.certificate.lattice_basis_transport_provenance_sufficient =
      all_basis_transport_provenance_sufficient;
  result.certificate.local_transport_search_completed =
      all_local_transport_searches_completed;
  result.certificate.explicit_remap_provenance_available =
      any_explicit_remap_provenance;
  result.certificate.component_representative_injectivity_audit_passed =
      all_component_injectivity_audits_passed;
  result.certificate.gauge_replay_arithmetic_closed =
      all_gauge_replay_arithmetic_closed;
  result.certificate.atom_temporal_injectivity_audited = false;
  result.certificate.atom_temporal_injectivity_audit_passed = false;
  result.certificate.independent_space_time_cochain_audited = false;
  result.certificate.independent_space_time_cochain_closed = false;
  result.certificate.all_atom_reconstruction_audited = false;
  result.certificate.coordinates_materialized_and_replay_consistent = false;
  result.certificate.temporal_lift_locally_certified = false;
  result.certificate.complete_domain_trajectory_claim_eligible = false;
  result.certificate.source_trajectory_coverage_complete =
      result.source_trajectory_coverage_complete;
  result.certificate.retained_candidate_bands_complete =
      result.all_observation_bands_complete;
  result.certificate.optimal_path_exact_on_retained_graph =
      result.exact_on_retained_graph;
  result.certificate.optimal_path_proven_on_complete_bounded_domain =
      result.exact_on_retained_graph && result.all_observation_bands_complete &&
      all_segments_bounded_domain_only && result.temporal_coverage_complete;
  result.certificate.unique_on_configured_retained_state_graph =
      result.all_segment_objectives_unique;
  result.certificate.selected_transition_time_reversal_audited =
      all_selected_transition_reversal_audited;
  result.certificate.selected_transition_time_reversal_consistent =
      all_selected_transition_reversal_audited &&
      all_selected_transition_reversal_consistent;
  result.certificate.robust_loss_exact =
      config.loss == HoloLiftPhase1Loss::Quadratic;
  return result;
}

std::expected<std::vector<std::array<double, 3>>, std::string>
reconstruct_hololift_phase1_frame(
    HoloLiftPhase1PreparationContext &context,
    const HoloLiftObservationStore &store,
    const HoloLiftPhase1PreparedFrame &prepared_frame,
    const HoloLiftPhase1FrameSelection &selection,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates) {
  if (selection.use != HoloLiftPhase1FrameUse::TemporalObservation ||
      selection.candidate_index >= store.candidates().size() ||
      prepared_frame.frame_index() >= store.frames().size()) {
    return std::unexpected("Phase 1 reconstruction selection is invalid");
  }
  const auto &frame = store.frames()[prepared_frame.frame_index()];
  if (selection.candidate_index < frame.candidates.begin ||
      selection.candidate_index >= frame.candidates.begin +
                                       frame.candidates.count) {
    return std::unexpected(
        "Phase 1 reconstruction candidate does not belong to frame");
  }
  const auto &epoch = store.topology_epochs()[frame.topology_epoch_index];
  const auto hard_edges = store.hard_edges().subspan(
      epoch.hard_edges.begin, epoch.hard_edges.count);
  const auto binding = validate_hololift_frame_binding(
      epoch, frame, prepared_frame.trajectory_frame_index(),
      prepared_frame.box_matrix(), atom_order, wrapped_coordinates);
  if (!binding)
    return std::unexpected("Phase 1 reconstruction binding failed: " +
                           binding.error());
  const auto spatial = replay_hololift_spatial_lift(
      epoch, frame, hard_edges, atom_order, wrapped_coordinates);
  if (!spatial)
    return std::unexpected(spatial.error());
  if (spatial->identity != prepared_frame.spatial_lift_identity() ||
      spatial->identity != frame.spatial_lift_identity ||
      spatial->ambiguous_hard_edges !=
          frame.spatial_lift_ambiguous_hard_edges ||
      spatial->cycle_residuals != frame.spatial_lift_cycle_residuals) {
    return std::unexpected(
        "Phase 1 reconstruction spatial replay mismatch");
  }
  const auto mapping = atom_mapping(*context.impl_, store, frame, atom_order);
  if (!mapping)
    return std::unexpected(mapping.error());
  return reconstruct_selected_coordinates(
      store, prepared_frame, selection, mapping->component_by_input_atom,
      spatial->atom_images, wrapped_coordinates);
}

std::expected<HoloLiftPhase1AuditedResult, std::string>
audit_hololift_phase1_all_atom_reconstruction(
    HoloLiftPhase1PreparationContext &context,
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    HoloLiftPhase1Result result,
    std::span<const HoloLiftPhase1TrajectoryFrameView> trajectory_frames) {
  if (!context.impl_ || context.impl_->store != &store)
    return std::unexpected("Phase 1 reconstruction audit context mismatch");
  if (prepared_frames.size() != store.frames().size() ||
      trajectory_frames.size() != store.frames().size() ||
      result.frames.size() != store.frames().size() ||
      result.api_version != HOLOLIFT_PHASE1_API_VERSION) {
    return std::unexpected(
        "Phase 1 reconstruction audit frame coverage is invalid");
  }

  HoloLiftPhase1Result working_result = std::move(result);
  for (auto &selection : working_result.frames) {
    selection.atoms_audited = 0;
    selection.atom_temporal_transitions_audited = 0;
    selection.atom_temporal_nearest_image_violations = 0;
    selection.atom_temporal_injectivity_uncertified = 0;
    selection.independent_temporal_image_ambiguities = 0;
    selection.independent_space_time_edges_audited = 0;
    selection.independent_space_time_curvature_residuals = 0;
    selection.maximum_component_representative_error_A = 0.0;
    selection.maximum_atom_peculiar_displacement_A = 0.0;
  }
  for (auto &segment : working_result.segments) {
    segment.temporal_frames_audited = 0;
    segment.atoms_audited = 0;
    segment.atom_temporal_transitions_audited = 0;
    segment.atom_temporal_nearest_image_violations = 0;
    segment.atom_temporal_injectivity_uncertified = 0;
    segment.independent_temporal_image_ambiguities = 0;
    segment.independent_space_time_edges_audited = 0;
    segment.independent_space_time_curvature_residuals = 0;
    segment.maximum_component_representative_error_A = 0.0;
    segment.maximum_atom_peculiar_displacement_A = 0.0;
    segment.all_atom_reconstruction_audited = false;
    segment.coordinates_materialized_and_replay_consistent = false;
    segment.temporal_lift_locally_certified = false;
    segment.atom_temporal_injectivity_audited = false;
    segment.atom_temporal_injectivity_audit_passed = false;
    segment.independent_space_time_cochain_audited = false;
    segment.independent_space_time_cochain_closed = false;
  }
  working_result.reconstruction_audit = {};
  working_result.certificate.all_atom_reconstruction_audited = false;
  working_result.certificate.coordinates_materialized_and_replay_consistent =
      false;
  working_result.certificate.temporal_lift_locally_certified = false;
  working_result.certificate.atom_temporal_injectivity_audited = false;
  working_result.certificate.atom_temporal_injectivity_audit_passed = false;
  working_result.certificate.independent_space_time_cochain_audited = false;
  working_result.certificate.independent_space_time_cochain_closed = false;
  HoloLiftPhase1ReconstructionAudit audit;
  struct SelectedAtomAuditFrame {
    std::size_t frame_index = HOLOLIFT_NO_INDEX;
    std::size_t segment_index = HOLOLIFT_NO_INDEX;
    HoloLiftHash128 topology_epoch_identity;
    std::array<double, 6> physical_gram{};
    std::vector<std::array<double, 3>> wrapped_reference;
    std::vector<std::array<double, 3>> selected_reference;
    std::vector<HoloLiftLatticeImage> spatial_images_reference;
  };
  SelectedAtomAuditFrame previous_atom_frame;
  bool has_previous_atom_frame = false;
  std::vector<std::size_t> audited_segment_frames(
      working_result.segments.size(), 0);
  std::vector<std::size_t> audited_segment_transitions(
      working_result.segments.size(), 0);
  for (std::size_t frame_index = 0; frame_index < store.frames().size();
       ++frame_index) {
    auto &selection = working_result.frames[frame_index];
    if (selection.use != HoloLiftPhase1FrameUse::TemporalObservation)
      continue;
    if (selection.segment_index >= working_result.segments.size() ||
        prepared_frames[frame_index].frame_index() != frame_index ||
        trajectory_frames[frame_index].trajectory_frame_index !=
            prepared_frames[frame_index].trajectory_frame_index()) {
      return std::unexpected(
          "Phase 1 reconstruction audit selection binding is invalid");
    }
    auto &segment = working_result.segments[selection.segment_index];
    const auto &atom_order = trajectory_frames[frame_index].atom_order;
    const auto &wrapped_coordinates =
        trajectory_frames[frame_index].wrapped_coordinates;
    if (atom_order.size() != wrapped_coordinates.size()) {
      return std::unexpected(
          "Phase 1 reconstruction audit atom arrays differ in size");
    }
    const auto &frame = store.frames()[frame_index];
    const auto &epoch = store.topology_epochs()[frame.topology_epoch_index];
    const auto hard_edges = store.hard_edges().subspan(
        epoch.hard_edges.begin, epoch.hard_edges.count);
    const auto binding = validate_hololift_frame_binding(
        epoch, frame,
        prepared_frames[frame_index].trajectory_frame_index(),
        prepared_frames[frame_index].box_matrix(), atom_order,
        wrapped_coordinates);
    if (!binding)
      return std::unexpected(
          "Phase 1 reconstruction audit frame binding failed: " +
          binding.error());
    const auto spatial = replay_hololift_spatial_lift(
        epoch, frame, hard_edges, atom_order, wrapped_coordinates);
    if (!spatial)
      return std::unexpected("Phase 1 atom audit spatial replay failed: " +
                             spatial.error());
    if (spatial->identity !=
            prepared_frames[frame_index].spatial_lift_identity() ||
        spatial->identity != frame.spatial_lift_identity ||
        spatial->ambiguous_hard_edges !=
            frame.spatial_lift_ambiguous_hard_edges ||
        spatial->cycle_residuals != frame.spatial_lift_cycle_residuals) {
      return std::unexpected(
          "Phase 1 atom audit spatial replay mismatch");
    }
    const auto mapping = atom_mapping(*context.impl_, store, frame, atom_order);
    if (!mapping)
      return std::unexpected(mapping.error());
    const auto reconstructed = reconstruct_selected_coordinates(
        store, prepared_frames[frame_index], selection,
        mapping->component_by_input_atom, spatial->atom_images,
        wrapped_coordinates);
    if (!reconstructed)
      return std::unexpected(reconstructed.error());
    const auto &candidate = store.candidates()[selection.candidate_index];
    if (!candidate.component_images.valid_for(store.component_images().size()) ||
        candidate.component_images.count != epoch.components.count) {
      return std::unexpected(
          "Phase 1 reconstruction audit candidate range is invalid");
    }
    const auto component_images = store.component_images().subspan(
        candidate.component_images.begin, candidate.component_images.count);

    const titan_pbctopo::PbctopoLatticeMetric lattice(
        prepared_frames[frame_index].box_matrix());
    if (!lattice.valid())
      return std::unexpected(
          "Phase 1 reconstruction audit lattice is invalid");

    const auto transport = make_hololift_lattice_basis_transport(
        selection.lattice_basis_transport_to_segment_reference,
        prepared_frames[frame_index].box_matrix());
    if (!transport)
      return std::unexpected("Phase 1 reconstruction basis replay failed: " +
                             transport.error());
    const auto expected_input_gauge = hololift_lattice_image_from_reference(
        selection.cumulative_global_gauge, *transport);
    if (!expected_input_gauge ||
        *expected_input_gauge !=
            selection.cumulative_global_gauge_input_basis) {
      return std::unexpected(
          "Phase 1 reconstruction input-basis gauge replay failed");
    }
    SelectedAtomAuditFrame atom_frame;
    atom_frame.frame_index = frame_index;
    atom_frame.segment_index = selection.segment_index;
    atom_frame.topology_epoch_identity = epoch.topology_epoch_identity;
    atom_frame.physical_gram = physical_gram(transport->transported_box);
    const std::size_t canonical_atom_count =
        mapping->input_index_by_canonical_dense_atom.size();
    atom_frame.wrapped_reference.resize(canonical_atom_count);
    atom_frame.selected_reference.resize(canonical_atom_count);
    atom_frame.spatial_images_reference.resize(canonical_atom_count);

    std::vector<std::array<long double, 3>> component_sums(
        epoch.components.count);
    std::vector<long double> component_weight_sums(epoch.components.count,
                                                   0.0L);
    std::vector<std::size_t> component_counts(epoch.components.count, 0);
    for (std::size_t atom_index = 0; atom_index < reconstructed->size();
         ++atom_index) {
      const std::uint32_t canonical_index =
          mapping->canonical_dense_by_input_atom[atom_index];
      const std::uint32_t component =
          mapping->component_by_input_atom[atom_index];
      if (canonical_index >= canonical_atom_count ||
          component >= component_sums.size() ||
          atom_index >= mapping->weight_by_input_atom.size())
        return std::unexpected(
            "Phase 1 reconstruction audit atom mapping is invalid");
      const long double atom_weight =
          static_cast<long double>(mapping->weight_by_input_atom[atom_index]);
      const auto scaled = lattice.cartesian_to_scaled(
          (*reconstructed)[atom_index][0], (*reconstructed)[atom_index][1],
          (*reconstructed)[atom_index][2]);
      if (!std::all_of(scaled.begin(), scaled.end(),
                       [](double value) { return std::isfinite(value); })) {
        return std::unexpected(
            "Phase 1 reconstruction audit produced non-finite coordinates");
      }
      for (std::size_t axis = 0; axis < 3; ++axis)
        component_sums[component][axis] += atom_weight * scaled[axis];
      component_weight_sums[component] += atom_weight;
      ++component_counts[component];

      const auto &wrapped = wrapped_coordinates[atom_index];
      const auto wrapped_input = lattice.cartesian_to_scaled(
          wrapped[0], wrapped[1], wrapped[2]);
      const auto wrapped_reference =
          hololift_fractional_to_reference(wrapped_input, *transport);
      const auto spatial_reference = hololift_lattice_image_to_reference(
          spatial->atom_images[atom_index], *transport);
      const auto component_reference = hololift_lattice_image_to_reference(
          component_images[component], *transport);
      if (!spatial_reference || !component_reference)
        return std::unexpected(!spatial_reference
                                   ? spatial_reference.error()
                                   : component_reference.error());
      HoloLiftLatticeImage component_and_gauge_reference;
      HoloLiftLatticeImage total_reference_image;
      if (!checked_image_add(*component_reference,
                             selection.cumulative_global_gauge,
                             component_and_gauge_reference) ||
          !checked_image_add(*spatial_reference,
                             component_and_gauge_reference,
                             total_reference_image) ||
          !image_is_exact_binary64(total_reference_image)) {
        return std::unexpected(
            "Phase 1 atom audit reference-basis image overflows");
      }
      std::array<double, 3> selected_reference{};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const std::int64_t image =
            axis == 0 ? total_reference_image.x
                      : (axis == 1 ? total_reference_image.y
                                   : total_reference_image.z);
        selected_reference[axis] =
            wrapped_reference[axis] + static_cast<double>(image);
      }
      const auto reconstructed_reference =
          hololift_fractional_to_reference(scaled, *transport);
      for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(wrapped_reference[axis]) ||
            !std::isfinite(selected_reference[axis]) ||
            std::fabs(reconstructed_reference[axis] -
                      selected_reference[axis]) > 1.0e-10) {
          return std::unexpected(
              "Phase 1 atom audit disagrees with basis-transport replay");
        }
      }
      atom_frame.wrapped_reference[canonical_index] = wrapped_reference;
      atom_frame.selected_reference[canonical_index] = selected_reference;
      atom_frame.spatial_images_reference[canonical_index] =
          *spatial_reference;
    }

    const auto prepared_representatives =
        prepared_frames[frame_index].component_fractional_representatives();
    const auto prepared_weights =
        prepared_frames[frame_index].component_weights();
    const auto components = store.components().subspan(
        epoch.components.begin, epoch.components.count);
    const auto gram = prepared_frames[frame_index].physical_gram();
    const double length_scale = std::sqrt(
        std::max({gram[0], gram[3], gram[5], 1.0}));
    const double tolerance_A = 1.0e-9 * length_scale;
    for (std::size_t component = 0; component < components.size();
         ++component) {
      if (component_counts[component] !=
              components[component].exact_atom_membership.count ||
          component >= prepared_weights.size() ||
          std::fabs(static_cast<double>(component_weight_sums[component]) -
                    prepared_weights[component]) >
              1.0e-12 * std::max(1.0, prepared_weights[component])) {
        return std::unexpected(
            "Phase 1 reconstruction audit component weight is invalid");
      }
      HoloLiftLatticeImage component_and_gauge;
      if (!checked_image_add(component_images[component],
                             selection.cumulative_global_gauge_input_basis,
                             component_and_gauge) ||
          !image_is_exact_binary64(component_and_gauge)) {
        return std::unexpected(
            "Phase 1 reconstruction audit component image overflows");
      }
      std::array<double, 3> difference{};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const double observed = static_cast<double>(
            component_sums[component][axis] /
            component_weight_sums[component]);
        const std::int64_t image =
            axis == 0 ? component_and_gauge.x
                      : (axis == 1 ? component_and_gauge.y
                                   : component_and_gauge.z);
        const double expected = prepared_representatives[component][axis] +
                                static_cast<double>(image);
        difference[axis] = observed - expected;
      }
      const auto cartesian_difference = lattice.scaled_to_cartesian(
          difference[0], difference[1], difference[2]);
      const double error_A = std::sqrt(
          cartesian_difference[0] * cartesian_difference[0] +
          cartesian_difference[1] * cartesian_difference[1] +
          cartesian_difference[2] * cartesian_difference[2]);
      if (!std::isfinite(error_A) || error_A > tolerance_A) {
        return std::unexpected(
            "Phase 1 all-atom reconstruction disagrees with component lift");
      }
      audit.maximum_component_representative_error_A = std::max(
          audit.maximum_component_representative_error_A, error_A);
      selection.maximum_component_representative_error_A = std::max(
          selection.maximum_component_representative_error_A, error_A);
      segment.maximum_component_representative_error_A = std::max(
          segment.maximum_component_representative_error_A, error_A);
    }

    if (has_previous_atom_frame &&
        previous_atom_frame.segment_index == atom_frame.segment_index) {
      if (previous_atom_frame.topology_epoch_identity !=
              atom_frame.topology_epoch_identity ||
          previous_atom_frame.selected_reference.size() !=
              canonical_atom_count ||
          previous_atom_frame.wrapped_reference.size() !=
              canonical_atom_count ||
          previous_atom_frame.spatial_images_reference.size() !=
              canonical_atom_count) {
        return std::unexpected(
            "Phase 1 atom temporal audit topology changed within segment");
      }
      std::array<double, 6> transition_gram{};
      for (std::size_t index = 0; index < transition_gram.size(); ++index) {
        transition_gram[index] =
            0.5 * (previous_atom_frame.physical_gram[index] +
                   atom_frame.physical_gram[index]);
      }
      const titan_pbctopo::PbctopoLatticeMetric temporal_metric(
          transition_gram);
      if (!temporal_metric.valid())
        return std::unexpected(
            "Phase 1 atom temporal audit metric is invalid");
      const double injectivity_radius_A =
          0.5 * std::min(
                    shortest_lattice_vector_lower_bound(
                        previous_atom_frame.physical_gram),
                    shortest_lattice_vector_lower_bound(
                        atom_frame.physical_gram));
      if (!std::isfinite(injectivity_radius_A) ||
          injectivity_radius_A <= 0.0) {
        return std::unexpected(
            "Phase 1 atom temporal injectivity radius is invalid");
      }

      std::vector<HoloLiftLatticeImage> independent_temporal_images;
      independent_temporal_images.reserve(canonical_atom_count);
      for (std::size_t atom_index = 0; atom_index < canonical_atom_count;
           ++atom_index) {
        std::array<double, 3> selected_delta{};
        std::array<double, 3> wrapped_delta{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
          selected_delta[axis] =
              atom_frame.selected_reference[atom_index][axis] -
              previous_atom_frame.selected_reference[atom_index][axis];
          wrapped_delta[axis] =
              atom_frame.wrapped_reference[atom_index][axis] -
              previous_atom_frame.wrapped_reference[atom_index][axis];
        }
        const auto selected_nearest = temporal_metric.nearest_image_scaled(
            selected_delta[0], selected_delta[1], selected_delta[2]);
        const auto independent_nearest = temporal_metric.nearest_image_scaled(
            wrapped_delta[0], wrapped_delta[1], wrapped_delta[2]);
        if (!selected_nearest.valid || !independent_nearest.valid) {
          return std::unexpected(
              "Phase 1 atom temporal closest-image audit failed");
        }
        if (!temporal_metric.image_is_nearest_equivalent_scaled(
                selected_delta[0], selected_delta[1], selected_delta[2], 0,
                0, 0, selected_nearest)) {
          ++selection.atom_temporal_nearest_image_violations;
          ++segment.atom_temporal_nearest_image_violations;
          ++audit.atom_temporal_nearest_image_violations;
        }
        const auto cartesian_delta = temporal_metric.scaled_to_cartesian(
            selected_delta[0], selected_delta[1], selected_delta[2]);
        const double displacement_A = std::sqrt(
            cartesian_delta[0] * cartesian_delta[0] +
            cartesian_delta[1] * cartesian_delta[1] +
            cartesian_delta[2] * cartesian_delta[2]);
        if (!std::isfinite(displacement_A))
          return std::unexpected(
              "Phase 1 atom temporal displacement is not finite");
        selection.maximum_atom_peculiar_displacement_A = std::max(
            selection.maximum_atom_peculiar_displacement_A, displacement_A);
        segment.maximum_atom_peculiar_displacement_A = std::max(
            segment.maximum_atom_peculiar_displacement_A, displacement_A);
        audit.maximum_atom_peculiar_displacement_A = std::max(
            audit.maximum_atom_peculiar_displacement_A, displacement_A);
        const double injectivity_tolerance =
            1.0e-12 * std::max(1.0, injectivity_radius_A);
        if (displacement_A >=
            injectivity_radius_A - injectivity_tolerance) {
          ++selection.atom_temporal_injectivity_uncertified;
          ++segment.atom_temporal_injectivity_uncertified;
          ++audit.atom_temporal_injectivity_uncertified;
        }
        if (independent_nearest.ambiguous) {
          ++selection.independent_temporal_image_ambiguities;
          ++segment.independent_temporal_image_ambiguities;
          ++audit.independent_temporal_image_ambiguities;
        }
        independent_temporal_images.push_back(
            {independent_nearest.image_x, independent_nearest.image_y,
             independent_nearest.image_z});
      }
      segment.atom_temporal_transitions_audited += canonical_atom_count;
      selection.atom_temporal_transitions_audited += canonical_atom_count;
      audit.atom_temporal_transitions_audited += canonical_atom_count;

      for (const auto &edge : hard_edges) {
        if (edge.atom_a >= canonical_atom_count ||
            edge.atom_b >= canonical_atom_count) {
          return std::unexpected(
              "Phase 1 independent cochain edge is out of range");
        }
        HoloLiftLatticeImage previous_spatial_edge;
        HoloLiftLatticeImage current_spatial_edge;
        HoloLiftLatticeImage previous_path;
        HoloLiftLatticeImage current_path;
        HoloLiftLatticeImage curvature;
        const bool closed =
            checked_image_subtract(
                previous_atom_frame.spatial_images_reference[edge.atom_b],
                previous_atom_frame.spatial_images_reference[edge.atom_a],
                previous_spatial_edge) &&
            checked_image_subtract(
                atom_frame.spatial_images_reference[edge.atom_b],
                atom_frame.spatial_images_reference[edge.atom_a],
                current_spatial_edge) &&
            checked_image_add(previous_spatial_edge,
                              independent_temporal_images[edge.atom_b],
                              previous_path) &&
            checked_image_add(current_spatial_edge,
                              independent_temporal_images[edge.atom_a],
                              current_path) &&
            checked_image_subtract(previous_path, current_path, curvature);
        if (!closed)
          return std::unexpected(
              "Phase 1 independent space-time cochain overflows");
        if (curvature != HoloLiftLatticeImage{}) {
          ++selection.independent_space_time_curvature_residuals;
          ++segment.independent_space_time_curvature_residuals;
          ++audit.independent_space_time_curvature_residuals;
        }
        ++segment.independent_space_time_edges_audited;
        ++selection.independent_space_time_edges_audited;
        ++audit.independent_space_time_edges_audited;
      }
      ++audited_segment_transitions[selection.segment_index];
    }

    ++audit.temporal_frames_audited;
    audit.atoms_audited += reconstructed->size();
    selection.atoms_audited = reconstructed->size();
    ++segment.temporal_frames_audited;
    segment.atoms_audited += reconstructed->size();
    ++audited_segment_frames[selection.segment_index];
    previous_atom_frame = std::move(atom_frame);
    has_previous_atom_frame = true;
  }

  if (audit.temporal_frames_audited !=
      working_result.temporal_observation_frames)
    return std::unexpected(
        "Phase 1 reconstruction audit omitted a temporal observation");
  for (std::size_t segment_index = 0;
       segment_index < working_result.segments.size(); ++segment_index) {
    if (audited_segment_frames[segment_index] !=
        working_result.segments[segment_index].frame_count) {
      return std::unexpected(
          "Phase 1 reconstruction audit omitted a segment frame");
    }
    if (audited_segment_transitions[segment_index] !=
        working_result.segments[segment_index].temporal_transition_count) {
      return std::unexpected(
          "Phase 1 atom audit omitted a selected temporal transition");
    }
  }

  audit.selected_frames_complete = true;
  audit.atom_temporal_injectivity_audited =
      std::all_of(working_result.segments.begin(),
                  working_result.segments.end(),
                  [](const auto &segment) {
                    return segment.temporal_transition_count > 0;
                  });
  audit.independent_space_time_cochain_audited =
      audit.atom_temporal_injectivity_audited;
  working_result.reconstruction_audit = audit;
  for (auto &segment : working_result.segments) {
    segment.all_atom_reconstruction_audited = true;
    segment.coordinates_materialized_and_replay_consistent = true;
    segment.atom_temporal_injectivity_audited =
        segment.temporal_transition_count > 0;
    segment.atom_temporal_injectivity_audit_passed =
        segment.atom_temporal_injectivity_audited &&
        segment.atom_temporal_nearest_image_violations == 0 &&
        segment.atom_temporal_injectivity_uncertified == 0 &&
        segment.independent_temporal_image_ambiguities == 0;
    segment.independent_space_time_cochain_audited =
        segment.temporal_transition_count > 0;
    segment.independent_space_time_cochain_closed =
        segment.independent_space_time_cochain_audited &&
        segment.independent_temporal_image_ambiguities == 0 &&
        segment.independent_space_time_curvature_residuals == 0;
    segment.temporal_lift_locally_certified =
        segment.provider_unsupported_selected_candidate_count == 0 &&
        segment.provider_unsupported_transition_count == 0 &&
        segment.all_global_gauge_steps_solved &&
        segment.lattice_basis_transport_accepted_under_policy &&
        segment.lattice_basis_transport_provenance_sufficient &&
        segment.temporal_transition_count > 0 &&
        segment.ambiguous_gauge_transition_count == 0 &&
        segment.component_nearest_image_violations == 0 &&
        segment.component_injectivity_uncertified == 0 &&
        segment.component_representative_injectivity_audit_passed &&
        segment.gauge_replay_arithmetic_closed &&
        segment.atom_temporal_injectivity_audit_passed &&
        segment.independent_space_time_cochain_audited &&
        segment.independent_space_time_cochain_closed &&
        segment.selected_transition_time_reversal_audited &&
        segment.selected_transition_time_reversal_consistent;
  }
  working_result.certificate.all_atom_reconstruction_audited = true;
  working_result.certificate.coordinates_materialized_and_replay_consistent =
      audit.selected_frames_complete;
  working_result.certificate.atom_temporal_injectivity_audited =
      audit.atom_temporal_injectivity_audited;
  working_result.certificate.atom_temporal_injectivity_audit_passed =
      audit.atom_temporal_injectivity_audited &&
      audit.atom_temporal_nearest_image_violations == 0 &&
      audit.atom_temporal_injectivity_uncertified == 0 &&
      audit.independent_temporal_image_ambiguities == 0;
  working_result.certificate.independent_space_time_cochain_audited =
      audit.independent_space_time_cochain_audited;
  working_result.certificate.independent_space_time_cochain_closed =
      audit.independent_space_time_cochain_audited &&
      audit.independent_temporal_image_ambiguities == 0 &&
      audit.independent_space_time_curvature_residuals == 0;
  working_result.certificate.temporal_lift_locally_certified =
      working_result.temporal_transition_coverage_complete &&
      std::all_of(working_result.segments.begin(),
                  working_result.segments.end(),
                  [](const auto &segment) {
                    return segment.temporal_lift_locally_certified;
                  });
  working_result.certificate.complete_domain_trajectory_claim_eligible =
      working_result.certificate.temporal_lift_locally_certified &&
      working_result.certificate.all_frames_trajectory_bound &&
      working_result.certificate.all_spatial_lifts_replayed &&
      working_result.certificate.all_selected_candidates_hard_feasible &&
      working_result.certificate.source_trajectory_coverage_complete &&
      working_result.certificate.retained_candidate_bands_complete &&
      working_result.certificate.optimal_path_exact_on_retained_graph &&
      working_result.certificate
          .optimal_path_proven_on_complete_bounded_domain;
  const auto hierarchy =
      validate_all_atom_audit_hierarchy(store, working_result);
  if (!hierarchy)
    return std::unexpected(hierarchy.error());
  const auto digest =
      compute_phase1_audit_digest(store, prepared_frames, working_result);
  if (!digest)
    return std::unexpected(digest.error());
  return HoloLiftPhase1AuditedResult(
      std::make_unique<HoloLiftPhase1Result>(std::move(working_result)),
      *digest);
}

std::expected<void, std::string> validate_hololift_phase1_audited_result(
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    const HoloLiftPhase1AuditedResult &audited_result) {
  const auto *result = audited_result.result_if_valid();
  if (result == nullptr)
    return std::unexpected("Phase 1 audited result is moved-from");
  const auto hierarchy = validate_all_atom_audit_hierarchy(store, *result);
  if (!hierarchy)
    return hierarchy;
  const auto expected_digest =
      compute_phase1_audit_digest(store, prepared_frames, *result);
  if (!expected_digest)
    return std::unexpected(expected_digest.error());
  if (!std::equal(expected_digest->begin(), expected_digest->end(),
                  audited_result.trajectory_audit_digest_sha256().begin())) {
    return std::unexpected(
        "Phase 1 trajectory/all-atom audit digest mismatch");
  }
  return {};
}

} // namespace titan_hololift
