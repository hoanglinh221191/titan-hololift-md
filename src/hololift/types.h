#ifndef TITAN_HOLOLIFT_TYPES_H
#define TITAN_HOLOLIFT_TYPES_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace titan_hololift {

inline constexpr std::uint32_t HOLOLIFT_PHASE0_API_VERSION = 10;
inline constexpr std::size_t HOLOLIFT_NO_INDEX =
    std::numeric_limits<std::size_t>::max();
inline constexpr std::uint32_t HOLOLIFT_NO_DENSE_INDEX =
    std::numeric_limits<std::uint32_t>::max();

struct HoloLiftHash128 {
  std::uint64_t lo = 0;
  std::uint64_t hi = 0;

  [[nodiscard]] bool empty() const noexcept { return lo == 0 && hi == 0; }
  bool operator==(const HoloLiftHash128 &) const = default;
};

struct HoloLiftHash128Hasher {
  [[nodiscard]] std::size_t
  operator()(const HoloLiftHash128 &value) const noexcept {
    const std::uint64_t mixed =
        value.lo ^ (value.hi + 0x9e3779b97f4a7c15ULL +
                    (value.lo << 6) + (value.lo >> 2));
    return static_cast<std::size_t>(mixed);
  }
};

struct HoloLiftRange {
  std::size_t begin = 0;
  std::size_t count = 0;

  [[nodiscard]] bool valid_for(std::size_t total) const noexcept {
    return begin <= total && count <= total - begin;
  }

  bool operator==(const HoloLiftRange &) const = default;
};

struct HoloLiftSourceContract {
  std::uint32_t schema_version = 0;
  std::string producer_version;
  std::string package_id;
  std::string trajectory_id;
  std::string coordinate_unit;
  std::string time_unit;
  std::string endianness;
  std::string floating_format;
  std::string identity_hash_algorithm;
  std::string binding_hash_algorithm;
  std::string payload_checksum_algorithm;
  std::string binding_mode;
  std::string negative_zero_policy;
  std::string nonfinite_policy;
  std::string observation_scope;
};

enum class HoloLiftSourceCoverageScope : std::uint8_t {
  Unknown,
  CompleteSourceManifest,
};

struct HoloLiftSourceCoverage {
  HoloLiftSourceCoverageScope scope = HoloLiftSourceCoverageScope::Unknown;
  std::size_t source_frame_count = 0;
  std::size_t imported_observation_frame_count = 0;
  std::size_t skipped_local_unwrap_frame_count = 0;
  std::size_t skipped_fallback_frame_count = 0;

  [[nodiscard]] bool known() const noexcept {
    return scope == HoloLiftSourceCoverageScope::CompleteSourceManifest;
  }

  [[nodiscard]] bool complete() const noexcept {
    return known() && source_frame_count != 0 &&
           imported_observation_frame_count == source_frame_count &&
           skipped_local_unwrap_frame_count == 0 &&
           skipped_fallback_frame_count == 0;
  }

  bool operator==(const HoloLiftSourceCoverage &) const = default;
};

struct HoloLiftSourceAtomKey {
  int owner = 0;
  std::uint64_t source_atom_id = 0;

  bool operator==(const HoloLiftSourceAtomKey &) const = default;
};

// Numeric identities are scoped by the enclosing store's package/trajectory
// ids and, within that store, by the exact topology epoch.
struct HoloLiftScopedComponentKey {
  HoloLiftHash128 topology_epoch_identity;
  HoloLiftHash128 component_identity;

  bool operator==(const HoloLiftScopedComponentKey &) const = default;
};

struct HoloLiftScopedAssignmentKey {
  HoloLiftHash128 topology_epoch_identity;
  HoloLiftHash128 assignment_identity;

  bool operator==(const HoloLiftScopedAssignmentKey &) const = default;
};

struct HoloLiftLatticeImage {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;

  bool operator==(const HoloLiftLatticeImage &) const = default;
};

struct HoloLiftComponentRecord {
  std::uint64_t component_id = 0;
  int owner = 0;
  bool mixed_owner = false;
  HoloLiftSourceAtomKey root_atom;
  HoloLiftRange exact_atom_membership;
  // The 64-bit component_id is the VIBE source signature. This 128-bit value
  // is the authoritative HoloLift identity.
  HoloLiftHash128 component_identity;
};

enum class HoloLiftHardGraphSource : std::uint8_t {
  None,
  ExplicitTopology,
  ValidatedMetadata,
  MixedHard,
};

enum class HoloLiftHardEdgeKind : std::uint8_t {
  TopologyBond,
  ValidatedMetadata,
};

// Endpoints are dense indices into the lexicographically sorted union of the
// epoch's exact atom memberships. The edge orientation is canonical (a < b).
// Lattice-image deltas are frame-dependent and are intentionally not stored in
// this static topology record.
struct HoloLiftHardEdgeRecord {
  std::uint32_t atom_a = 0;
  std::uint32_t atom_b = 0;
  HoloLiftHardEdgeKind kind = HoloLiftHardEdgeKind::TopologyBond;

  bool operator==(const HoloLiftHardEdgeRecord &) const = default;
};

struct HoloLiftTopologyEpochRecord {
  std::uint64_t topology_epoch_id = 0;
  std::uint64_t layout_signature = 0;
  HoloLiftRange components;
  HoloLiftRange hard_edges;
  HoloLiftHardGraphSource hard_graph_source = HoloLiftHardGraphSource::None;
  HoloLiftHash128 layout_identity;
  HoloLiftHash128 topology_epoch_identity;
  HoloLiftHash128 canonical_atom_universe_hash;
  std::size_t atom_count = 0;
  std::size_t hard_graph_cycle_rank = 0;
};

struct HoloLiftComponentImage {
  std::uint64_t component_id = 0;
  HoloLiftLatticeImage image;

  bool operator==(const HoloLiftComponentImage &) const = default;
};

enum class HoloLiftObservationClass : std::uint8_t {
  NotHardFeasible,
  WeakObservation,
  AmbiguousAnchor,
  StrongBoundedAnchor,
  StrongGlobalAnchor,
};

enum class HoloLiftCandidateSetScope : std::uint8_t {
  Unknown,
  RetainedCertifiedBand,
};

enum class HoloLiftFrameStatus : std::uint8_t {
  Certified,
  LocalUnwrap,
  Rescued,
  Fallback,
  WeakObservation,
  Contradicted,
};

enum class HoloLiftEvidenceState : std::uint8_t {
  NotEvaluated,
  NotApplicable,
  SupportedConnected,
  WeakDisconnected,
  Contradicted,
  WeakAmbiguous,
};

enum class HoloLiftCertificateScope : std::uint8_t {
  None,
  LocalUnwrap,
  ImageIntercomponentSteric,
  ImageIntercomponentStericInterface,
};

enum class HoloLiftCertificateGraphSource : std::uint8_t {
  None,
  GromacsTopology,
  MetadataChainResidue,
  GeometryCutoff,
  Mixed,
};

enum class HoloLiftSearchEvidenceSource : std::uint8_t {
  None,
  GeometryCutoff,
  ScoredContacts,
  ExplicitContacts,
  Mixed,
};

enum class HoloLiftTemporalPolicyStatus : std::uint8_t {
  FramewiseProduction,
  ExperimentalViterbiDiagnostic,
};

enum class HoloLiftSourceFrameRelation : std::uint8_t {
  SegmentStart,
  ContiguousSourceFrame,
};

struct HoloLiftFrameProvenance {
  HoloLiftFrameStatus status = HoloLiftFrameStatus::Fallback;
  HoloLiftCertificateScope certificate_scope =
      HoloLiftCertificateScope::None;
  HoloLiftCertificateGraphSource certificate_graph_source =
      HoloLiftCertificateGraphSource::None;
  HoloLiftHardGraphSource hard_graph_source =
      HoloLiftHardGraphSource::None;
  HoloLiftSearchEvidenceSource search_evidence_source =
      HoloLiftSearchEvidenceSource::None;
  HoloLiftTemporalPolicyStatus temporal_policy =
      HoloLiftTemporalPolicyStatus::FramewiseProduction;
  HoloLiftEvidenceState evidence_state = HoloLiftEvidenceState::NotEvaluated;
  bool evidence_consistency_evaluated = false;
  bool evidence_consistent = false;
  bool evidence_graph_connected = false;
  HoloLiftHash128 soft_observed_contact_pair_hash;
  HoloLiftHash128 soft_observed_lost_pair_hash;
  HoloLiftHash128 soft_observed_all_hypotheses_hash;
  HoloLiftHash128 soft_observed_selected_hypothesis_hash;
  HoloLiftHash128 soft_observed_selected_compatible_pair_hash;
  std::size_t soft_observed_hypothesis_count = 0;
  std::size_t soft_observed_component_relation_count = 0;
  std::size_t soft_observed_ambiguous_relation_count = 0;
  std::size_t soft_observed_compatible_hypothesis_count = 0;
  std::size_t soft_observed_selected_compatible_contact_count = 0;
  std::size_t soft_observed_alternative_hypothesis_contact_count = 0;
  std::size_t soft_observed_selected_compatible_contact_loss_count = 0;
  std::size_t soft_observed_no_support_relation_count = 0;
  bool soft_observed_selected_relation_has_support = false;
  bool soft_observed_hypothesis_construction_attempted = false;
  bool soft_observed_hypothesis_construction_complete = false;
  std::size_t soft_observed_hypothesis_mic_ambiguity_failures = 0;
  std::size_t soft_observed_hypothesis_component_mapping_failures = 0;
  std::size_t soft_observed_hypothesis_internal_edges_ignored = 0;
  std::size_t soft_observed_contacts_checked = 0;
  std::size_t soft_observed_contacts_lost = 0;
  double soft_observed_contact_cutoff_A =
      std::numeric_limits<double>::quiet_NaN();
  bool soft_observed_contact_construction_attempted = false;
  bool soft_observed_contact_construction_complete = false;
  std::size_t soft_observed_contact_cell_setup_failures = 0;
  std::size_t soft_observed_contact_atom_index_failures = 0;
  std::size_t soft_observed_contact_mic_query_failures = 0;
  std::uint64_t framewise_assignment_signature = 0;
  std::uint64_t output_assignment_signature = 0;
  std::uint64_t temporal_assignment_signature = 0;
  bool temporal_selected = false;
  bool temporal_changed_from_framewise = false;
  bool source_frame_report_present = false;
  HoloLiftHash128 framewise_assignment_identity;
  HoloLiftHash128 output_assignment_identity;
  HoloLiftHash128 temporal_assignment_identity;
};

enum class HoloLiftDomainCompleteness : std::uint8_t {
  Unknown,
  BoundedExhaustive,
  // Reserved until an outside-domain infeasibility proof is represented.
  GlobalCertified,
};

struct HoloLiftFrameSearchDomain {
  std::size_t shell_radius = 0;
  bool audit_known = false;
  bool evidence_complete = false;
  bool expansion_attempted = false;
  bool expansion_exhausted = false;
  bool bounded_domain_only = false;
  HoloLiftDomainCompleteness completeness =
      HoloLiftDomainCompleteness::Unknown;

  bool operator==(const HoloLiftFrameSearchDomain &) const = default;
};

struct HoloLiftCandidateDomainEvidence {
  bool boundary_known = false;
  bool touches_shell_boundary = false;

  bool operator==(const HoloLiftCandidateDomainEvidence &) const = default;
};

struct HoloLiftCandidateEvidence {
  bool selected_framewise = false;
  bool hard_feasible = false;
  bool search_equivalence_known = false;
  bool equivalent_to_search_best = false;
  bool equivalent_under_output_order = false;
};

enum class HoloLiftFeasibleSetSemantics : std::uint8_t {
  NotEnumerated,
  HardConstraintsWithinSearchDomain,
};

struct HoloLiftFrameEvidence {
  bool identified = false;
  bool uniqueness_search_exhaustive = false;
  bool objective_uniqueness_known = false;
  bool objective_unique_within_search_domain = false;
  bool evidence_uniqueness_known = false;
  bool evidence_unique_within_search_domain = false;
  bool feasible_assignment_set_enumerated = false;
  HoloLiftFeasibleSetSemantics feasible_set_semantics =
      HoloLiftFeasibleSetSemantics::NotEnumerated;
  bool equivalent_set_complete = false;
  bool credible_alternative_set_complete = false;
  // Legacy wire name: for a HoloLift policy-augmented store this counts
  // the full optimization band; per-candidate hard_feasible is authoritative.
  std::size_t retained_certified_assignment_count = 0;
  std::size_t search_equivalent_assignments_exported = 0;
  std::size_t equivalent_best_assignments_within_domain = 0;
  std::size_t feasible_assignments_within_domain = 0;
  std::size_t soft_score_valid_assignments_within_domain = 0;
  std::size_t search_domain_assignments_exported = 0;
};

// This is a compact audit summary, not the complete VIBE comparator vector.
// Candidate rank remains the authoritative framewise ordering evidence.
struct HoloLiftVibeScoreSummary {
  std::size_t broken_edges = std::numeric_limits<std::size_t>::max();
  double continuity_max_d2 = std::numeric_limits<double>::quiet_NaN();
  double continuity_path = std::numeric_limits<double>::quiet_NaN();
  double anchor_max_d2 = std::numeric_limits<double>::quiet_NaN();
  double anchor_mst2 = std::numeric_limits<double>::quiet_NaN();
  double min_pair_d2 = std::numeric_limits<double>::quiet_NaN();
  double shift_norm2 = std::numeric_limits<double>::quiet_NaN();
};

enum class HoloLiftPolicyAdmission : std::uint8_t {
  None,
  CarryPreviousFrame,
  RestoreDeclaredDomain,
};

// Recorded + hard_feasible=false means a provider-rejected assignment.
// NotEvaluated must never be interpreted as a provider certificate.
enum class HoloLiftProviderEvaluation : std::uint8_t {
  Recorded,
  NotEvaluated,
};

struct HoloLiftCandidateRecord {
  std::size_t rank = 0;
  std::uint64_t layout_signature = 0;
  std::uint64_t assignment_signature = 0;
  std::uint64_t evidence_relative_relation_signature = 0;
  HoloLiftHash128 evidence_compatible_hypothesis_hash;
  HoloLiftHash128 evidence_compatible_pair_hash;
  std::size_t evidence_compatible_hypothesis_count = 0;
  std::size_t evidence_supported_relation_count = 0;
  std::size_t evidence_compatible_contact_count = 0;
  std::size_t evidence_no_support_relation_count = 0;
  HoloLiftRange component_images;
  HoloLiftObservationClass observation_class =
      HoloLiftObservationClass::NotHardFeasible;
  HoloLiftCandidateEvidence evidence;
  HoloLiftCandidateDomainEvidence domain_evidence;
  HoloLiftVibeScoreSummary vibe_score_summary;
  HoloLiftHash128 layout_identity;
  HoloLiftHash128 assignment_identity;
  // True for a candidate that an evidence-absent policy added to the band
  // (an image carried from the preceding frame, or an image of the provider's
  // search cube the provider did not list).  The provider never sets it.
  bool carried = false;
  HoloLiftPolicyAdmission policy_admission = HoloLiftPolicyAdmission::None;
  HoloLiftProviderEvaluation provider_evaluation =
      HoloLiftProviderEvaluation::Recorded;
  std::size_t policy_source_frame_index = HOLOLIFT_NO_INDEX;
  HoloLiftHash128 policy_source_assignment_identity;
};

struct HoloLiftFrameRecord {
  std::size_t frame = 0;
  double time_ps = 0.0;
  std::size_t source_sequence_index = 0;
  std::size_t trajectory_frame_index = 0;
  HoloLiftSourceFrameRelation source_relation =
      HoloLiftSourceFrameRelation::SegmentStart;
  std::uint32_t topology_epoch_index = HOLOLIFT_NO_DENSE_INDEX;
  std::uint64_t topology_epoch_id = 0;
  std::uint64_t layout_signature = 0;
  std::array<double, 6> normalized_gram{
      1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  std::array<double, 9> box_matrix{};
  HoloLiftHash128 box_hash;
  HoloLiftHash128 atom_selection_hash;
  HoloLiftHash128 canonical_atom_universe_hash;
  HoloLiftHash128 wrapped_coordinate_hash;
  bool frame_binding_valid = false;
  std::uint32_t spatial_lift_policy_version = 0;
  HoloLiftHash128 spatial_lift_identity;
  std::size_t spatial_lift_ambiguous_hard_edges = 0;
  std::size_t spatial_lift_cycle_residuals = 0;
  bool spatial_lift_valid = false;
  HoloLiftRange candidates;
  std::size_t selected_candidate_index = HOLOLIFT_NO_INDEX;
  HoloLiftCandidateSetScope candidate_set_scope =
      HoloLiftCandidateSetScope::Unknown;
  HoloLiftFrameProvenance provenance;
  HoloLiftFrameEvidence evidence;
  HoloLiftFrameSearchDomain search_domain;
  HoloLiftHash128 topology_epoch_identity;
  HoloLiftHash128 layout_identity;
};

// Mutable import/build state. HoloLift algorithms consume only the immutable
// store returned by finalize_hololift_observation_store().
struct HoloLiftObservationStoreBuilder {
  std::uint32_t api_version = HOLOLIFT_PHASE0_API_VERSION;
  HoloLiftSourceContract source_contract;
  HoloLiftSourceCoverage source_coverage;
  std::vector<HoloLiftTopologyEpochRecord> topology_epochs;
  std::vector<HoloLiftComponentRecord> components;
  std::vector<HoloLiftSourceAtomKey> exact_atom_memberships;
  // Optional atomic masses aligned one-to-one with exact_atom_memberships.
  // A zero entry means that mass metadata is unavailable for that atom.
  std::vector<double> exact_atom_masses;
  std::vector<HoloLiftHardEdgeRecord> hard_edges;
  std::vector<HoloLiftFrameRecord> frames;
  std::vector<HoloLiftCandidateRecord> candidates;
  // Images follow the canonical component order of the frame's topology epoch.
  std::vector<HoloLiftLatticeImage> component_images;
};

struct HoloLiftObservationStoreFinalizer;

class HoloLiftObservationStore {
public:
  HoloLiftObservationStore(HoloLiftObservationStore &&) noexcept = default;
  HoloLiftObservationStore &
  operator=(HoloLiftObservationStore &&) noexcept = default;
  HoloLiftObservationStore(const HoloLiftObservationStore &) = delete;
  HoloLiftObservationStore &
  operator=(const HoloLiftObservationStore &) = delete;

  [[nodiscard]] std::uint32_t api_version() const noexcept {
    return storage_.api_version;
  }
  [[nodiscard]] const HoloLiftSourceContract &
  source_contract() const noexcept {
    return storage_.source_contract;
  }
  [[nodiscard]] const HoloLiftSourceCoverage &
  source_coverage() const noexcept {
    return storage_.source_coverage;
  }
  [[nodiscard]] std::span<const HoloLiftTopologyEpochRecord>
  topology_epochs() const noexcept {
    return storage_.topology_epochs;
  }
  [[nodiscard]] std::span<const HoloLiftComponentRecord>
  components() const noexcept {
    return storage_.components;
  }
  [[nodiscard]] std::span<const HoloLiftSourceAtomKey>
  exact_atom_memberships() const noexcept {
    return storage_.exact_atom_memberships;
  }
  [[nodiscard]] std::span<const double> exact_atom_masses() const noexcept {
    return storage_.exact_atom_masses;
  }
  [[nodiscard]] std::span<const HoloLiftHardEdgeRecord>
  hard_edges() const noexcept {
    return storage_.hard_edges;
  }
  [[nodiscard]] std::span<const HoloLiftFrameRecord>
  frames() const noexcept {
    return storage_.frames;
  }
  [[nodiscard]] std::span<const HoloLiftCandidateRecord>
  candidates() const noexcept {
    return storage_.candidates;
  }
  [[nodiscard]] std::span<const HoloLiftLatticeImage>
  component_images() const noexcept {
    return storage_.component_images;
  }

private:
  friend struct HoloLiftObservationStoreFinalizer;

  explicit HoloLiftObservationStore(
      HoloLiftObservationStoreBuilder &&storage) noexcept
      : storage_(std::move(storage)) {}

  HoloLiftObservationStoreBuilder storage_;
};

} // namespace titan_hololift

#endif
