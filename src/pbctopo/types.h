#ifndef TITAN_PBCTOPO_TYPES_H
#define TITAN_PBCTOPO_TYPES_H

#include "../pbctopo_hull.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace titan_pbctopo {

using pbctopo_hull::Vec3;

struct PbctopoHash128 {
  std::uint64_t lo = 0;
  std::uint64_t hi = 0;

  [[nodiscard]] bool empty() const noexcept { return lo == 0 && hi == 0; }
  bool operator==(const PbctopoHash128 &) const = default;
};

enum class PbctopoEvidenceState : std::uint8_t {
  NotEvaluated,
  NotApplicable,
  SupportedConnected,
  WeakDisconnected,
  Contradicted,
  WeakAmbiguous,
};

struct PbctopoAnchorGraphScore {
  double order_max_d2 = std::numeric_limits<double>::infinity();
  double order_path_d2 = std::numeric_limits<double>::infinity();
  double order_path_mean_d2 = std::numeric_limits<double>::infinity();
  double endpoint_image_gap = std::numeric_limits<double>::infinity();
  std::size_t residue_order_violations = 0;
  std::size_t contact_map_lost = 0;
  double contact_map_lost_fraction = std::numeric_limits<double>::infinity();
  double contact_map_max_d2 = std::numeric_limits<double>::infinity();
  double mst_mean_d2 = std::numeric_limits<double>::infinity();
  std::size_t order_edge_count = 0;
  std::size_t contact_pair_count = 0;
  std::size_t ambiguity_count = 0;
};

enum class PbctopoVirtualSeedSource : std::uint8_t {
  None,
  Density,
  Previous,
  Kinematic
};

struct Int3 {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;

  bool operator==(const Int3 &) const = default;
};

enum class PbctopoDomainCompleteness : std::uint8_t {
  Unknown,
  BoundedExhaustive,
  GlobalCertified,
};

constexpr const char *pbctopo_domain_completeness_name(
    PbctopoDomainCompleteness value) noexcept {
  switch (value) {
  case PbctopoDomainCompleteness::Unknown:
    return "unknown";
  case PbctopoDomainCompleteness::BoundedExhaustive:
    return "bounded_exhaustive";
  case PbctopoDomainCompleteness::GlobalCertified:
    return "global_certified";
  }
  return "unknown";
}

inline bool checked_int64_add(std::int64_t lhs, std::int64_t rhs,
                              std::int64_t &result) noexcept {
  constexpr auto min_value = std::numeric_limits<std::int64_t>::min();
  constexpr auto max_value = std::numeric_limits<std::int64_t>::max();
  if ((rhs > 0 && lhs > max_value - rhs) ||
      (rhs < 0 && lhs < min_value - rhs)) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

inline bool checked_int64_subtract(std::int64_t lhs, std::int64_t rhs,
                                   std::int64_t &result) noexcept {
  constexpr auto min_value = std::numeric_limits<std::int64_t>::min();
  constexpr auto max_value = std::numeric_limits<std::int64_t>::max();
  if ((rhs > 0 && lhs < min_value + rhs) ||
      (rhs < 0 && lhs > max_value + rhs)) {
    return false;
  }
  result = lhs - rhs;
  return true;
}

inline bool checked_int3_add(const Int3 &lhs, const Int3 &rhs,
                             Int3 &result) noexcept {
  return checked_int64_add(lhs.x, rhs.x, result.x) &&
         checked_int64_add(lhs.y, rhs.y, result.y) &&
         checked_int64_add(lhs.z, rhs.z, result.z);
}

inline bool checked_int3_subtract(const Int3 &lhs, const Int3 &rhs,
                                  Int3 &result) noexcept {
  return checked_int64_subtract(lhs.x, rhs.x, result.x) &&
         checked_int64_subtract(lhs.y, rhs.y, result.y) &&
         checked_int64_subtract(lhs.z, rhs.z, result.z);
}

inline bool checked_int3_negate(const Int3 &value, Int3 &result) noexcept {
  constexpr auto min_value = std::numeric_limits<std::int64_t>::min();
  if (value.x == min_value || value.y == min_value || value.z == min_value)
    return false;
  result = {-value.x, -value.y, -value.z};
  return true;
}

enum class PbctopoGeometryClass : std::uint8_t {
  Unknown,
  CompactUnitBox,
  Extended,
  Winding,
  ExtendedAndWinding,
};

enum class PbctopoStericRadiusSource : std::uint8_t {
  Unavailable,
  ForceFieldTopology,
  ForceFieldParameter,
  ExplicitZeroLennardJones,
  InferredElement,
  GenericFallback,
};

struct PbctopoGeometryClassification {
  std::array<double, 3> scaled_span{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()};
  std::array<double, 3> cartesian_extent_A{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()};
  std::array<Int3, 3> topological_winding_basis{};
  std::size_t topological_winding_rank = 0;
  PbctopoGeometryClass geometry_class = PbctopoGeometryClass::Unknown;
  bool fits_current_basis_fundamental_parallelepiped = false;
  // Legacy alias retained for report compatibility.
  bool fits_one_fundamental_cell = false;
  bool evaluated = false;
};

inline bool is_zero_delta(const Int3 &delta) {
  return delta.x == 0 && delta.y == 0 && delta.z == 0;
}

inline bool same_delta(const Int3 &lhs, const Int3 &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

struct PbctopoImageAssignment {
  std::vector<Int3> component_offsets;
  std::vector<std::uint64_t> component_ids;
  Int3 global_gauge{};
  std::uint64_t layout_signature = 0;
  std::uint64_t signature = 0;
  bool valid = false;
};

struct PbctopoRequiredContact {
  std::size_t a = 0;
  std::size_t b = 0;
  double max_distance_A = 0.0;
  double weight = 1.0;
};

struct PbctopoHardFeasibility {
  std::size_t certificate_atoms_total = 0;
  std::size_t hard_graph_atoms_covered = 0;
  std::size_t topology_bond_atoms_covered = 0;
  std::size_t metadata_only_atoms = 0;
  std::size_t hard_connectivity_edges_available = 0;
  std::size_t topology_edges_available = 0;
  std::size_t metadata_edges_available = 0;
  std::size_t topology_edges_checked = 0;
  std::size_t ambiguous_topology_edges = 0;
  std::size_t topology_edge_residuals = 0;
  std::size_t metadata_edges_checked = 0;
  std::size_t ambiguous_metadata_edges = 0;
  std::size_t metadata_edge_image_residuals = 0;
  std::size_t metadata_edge_distance_residuals = 0;
  double max_metadata_edge_distance_error_A = 0.0;
  std::size_t hard_edge_residuals = 0;
  std::size_t atom_image_residuals = 0;
  double max_atom_image_error = 0.0;
  std::size_t assignment_point_residuals = 0;
  double max_assignment_point_error = 0.0;
  std::size_t topology_cycle_residuals = 0;
  std::size_t unsupported_winding_rank = 0;
  std::size_t steric_pairs_checked = 0;
  std::size_t hard_clashes = 0;
  std::size_t metadata_internal_pairs_checked = 0;
  std::size_t metadata_internal_clashes = 0;
  std::size_t steric_force_field_atoms = 0;
  std::size_t steric_explicit_zero_lj_atoms = 0;
  std::size_t steric_inferred_atoms = 0;
  std::size_t steric_generic_fallback_atoms = 0;
  std::size_t steric_unavailable_atoms = 0;
  double min_steric_margin_A = std::numeric_limits<double>::infinity();
  std::size_t required_contacts_checked = 0;
  std::size_t required_contacts_lost = 0;
  std::size_t explicit_interface_contacts_checked = 0;
  std::size_t explicit_interface_contacts_lost = 0;
  double explicit_interface_weight_checked = 0.0;
  double explicit_interface_weight_lost = 0.0;
  bool assignment_available = false;
  bool steric_audit_available = false;
  bool steric_decision_complete = false;
  bool steric_pair_enumeration_complete = false;
  std::uint8_t steric_termination_reason = 2;
  bool steric_scan_complete = false;
  bool evaluated = false;

  [[nodiscard]] bool certified() const noexcept {
    return evaluated && assignment_available &&
           topology_edges_checked >= topology_edges_available &&
           metadata_edges_checked >= metadata_edges_available &&
           hard_edge_residuals == 0 &&
           atom_image_residuals == 0 && assignment_point_residuals == 0 &&
           topology_cycle_residuals == 0 && unsupported_winding_rank == 0 &&
           steric_audit_available && steric_decision_complete &&
           steric_pair_enumeration_complete && steric_scan_complete &&
           hard_clashes == 0 &&
           metadata_internal_clashes == 0 &&
           required_contacts_lost == 0;
  }
};

struct PbctopoFixedCertificateAudit {
  std::size_t metadata_internal_pairs_checked = 0;
  std::size_t metadata_internal_clashes = 0;
  std::size_t required_contacts_checked = 0;
  std::size_t required_contacts_lost = 0;
  bool radius_model_valid = false;
  bool internal_steric_scan_complete = false;
  bool required_contact_scan_complete = false;
  bool input_valid = false;
  bool evaluated = false;

  [[nodiscard]] bool passed() const noexcept {
    return evaluated && input_valid && radius_model_valid &&
           internal_steric_scan_complete && metadata_internal_clashes == 0 &&
           required_contact_scan_complete && required_contacts_lost == 0;
  }
};

struct PbctopoAtom {
  std::size_t global_atom_id = 0;
  std::size_t fragment_id = 0;
  int owner = 0;
  std::size_t local_atom_index = 0;
  std::size_t source_atom_id = 0;
  std::uint64_t chain_key = 0;
  int residue_id = std::numeric_limits<int>::min();
  bool has_topology_key = false;
  bool has_topology_graph = false;
  bool is_heavy = true;
  bool has_vdw_radius = false;
  PbctopoStericRadiusSource steric_radius_source =
      PbctopoStericRadiusSource::Unavailable;
  double vdw_radius = 0.0;
  double steric_radius = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  double atomic_mass = 0.0;
  bool has_atomic_mass = false;
};

struct PbctopoCutCandidate {
  double cut_x = 0.0;
  double cut_y = 0.0;
  double cut_z = 0.0;
  double gap_score = 0.0;
  double bbox_volume = std::numeric_limits<double>::infinity();
  double rg2 = std::numeric_limits<double>::infinity();
  double shape_kappa2 = std::numeric_limits<double>::infinity();
  double shape_asphericity = std::numeric_limits<double>::infinity();
  double shape_acylindricity = std::numeric_limits<double>::infinity();
  double shift_norm2 = std::numeric_limits<double>::infinity();
  double min_pair_d2 = std::numeric_limits<double>::infinity();
  double continuity_path = std::numeric_limits<double>::infinity();
  std::uint64_t evidence_relative_relation_signature = 0;
  PbctopoHash128 evidence_compatible_hypothesis_hash;
  PbctopoHash128 evidence_compatible_pair_hash;
  std::size_t evidence_compatible_hypothesis_count = 0;
  std::size_t evidence_supported_relation_count = 0;
  std::size_t evidence_compatible_contact_count = 0;
  std::size_t evidence_no_support_relation_count = 0;
  double continuity_max_d2 = std::numeric_limits<double>::infinity();
  double continuity_min_margin_A =
      std::numeric_limits<double>::quiet_NaN();
  std::size_t continuity_margin_atom_a =
      std::numeric_limits<std::size_t>::max();
  std::size_t continuity_margin_atom_b =
      std::numeric_limits<std::size_t>::max();
  double anchor_mst2 = std::numeric_limits<double>::infinity();
  double anchor_max_d2 = std::numeric_limits<double>::infinity();
  PbctopoAnchorGraphScore anchor_graph;
  double principal_rg2_max = std::numeric_limits<double>::infinity();
  double principal_rg2_mid = std::numeric_limits<double>::infinity();
  double principal_rg2_min = std::numeric_limits<double>::infinity();
  double edge_stretch = std::numeric_limits<double>::infinity();
  double max_owner_raw_minus_minimg_delta = 0.0;
  std::size_t broken_edges = std::numeric_limits<std::size_t>::max();
  std::size_t split_owner_pairs = 0;
  std::size_t clash_atom_a = std::numeric_limits<std::size_t>::max();
  std::size_t clash_atom_b = std::numeric_limits<std::size_t>::max();
  PbctopoVirtualSeedSource virtual_seed_source =
      PbctopoVirtualSeedSource::None;
  PbctopoImageAssignment assignment;
  PbctopoHardFeasibility hard_feasibility;
  PbctopoGeometryClassification geometry_classification;
  bool search_equivalence_known = false;
  bool equivalent_to_search_best = false;
  std::size_t component_search_shell_radius = 0;
  bool search_domain_audit_known = false;
  bool search_domain_evidence_complete = false;
  bool selected_assignment_touches_shell_boundary = false;
  bool domain_expansion_attempted = false;
  bool domain_expansion_exhausted = false;
  bool bounded_domain_only = false;
  PbctopoDomainCompleteness domain_completeness =
      PbctopoDomainCompleteness::Unknown;
  bool ultra_clash = false;
  bool valid = false;
};

enum class PbctopoLocalEdgeKind : std::uint8_t {
  GeometryHeuristic,
  TopologyBond,
  MetadataSkeleton
};

constexpr bool pbctopo_edge_is_hard_connectivity(
    PbctopoLocalEdgeKind kind) noexcept {
  return kind == PbctopoLocalEdgeKind::TopologyBond ||
         kind == PbctopoLocalEdgeKind::MetadataSkeleton;
}

constexpr bool pbctopo_edge_is_steric_exclusion(
    PbctopoLocalEdgeKind kind) noexcept {
  return kind == PbctopoLocalEdgeKind::TopologyBond;
}

struct PbctopoLocalEdge {
  std::size_t a = 0;
  std::size_t b = 0;
  double ref_d2 = 0.0;
  double inv_ref_d2 = 0.0;
  double break_cutoff_d2 = 0.0;
  PbctopoLocalEdgeKind kind = PbctopoLocalEdgeKind::GeometryHeuristic;
};

enum class PbctopoContinuityKind : std::uint8_t {
  TopologyHard,
  MetadataHard,
  MetadataSoft,
  ObservedSoft
};

constexpr bool
pbctopo_continuity_is_hard(PbctopoContinuityKind kind) noexcept {
  return kind == PbctopoContinuityKind::TopologyHard ||
         kind == PbctopoContinuityKind::MetadataHard;
}

struct PbctopoContinuityPair {
  std::size_t a = 0;
  std::size_t b = 0;
  double ref_d2 = 1.0;
  double inv_ref_d2 = 1.0;
  PbctopoContinuityKind kind = PbctopoContinuityKind::TopologyHard;
  double weight = 1.0;
  double hard_cutoff_d2 = std::numeric_limits<double>::infinity();
};

inline bool pbctopo_continuity_hard_violation(
    const PbctopoContinuityPair &pair, double d2,
    double fallback_hard_cutoff_d2) noexcept {
  if (!pbctopo_continuity_is_hard(pair.kind))
    return false;
  const double cutoff_d2 = pair.hard_cutoff_d2 < fallback_hard_cutoff_d2
                               ? pair.hard_cutoff_d2
                               : fallback_hard_cutoff_d2;
  return d2 > cutoff_d2;
}

inline double pbctopo_continuity_hard_margin_A(
    const PbctopoContinuityPair &pair, double d2,
    double fallback_hard_cutoff_d2) noexcept {
  const double cutoff_d2 =
      pair.hard_cutoff_d2 < fallback_hard_cutoff_d2
          ? pair.hard_cutoff_d2
          : fallback_hard_cutoff_d2;
  return std::sqrt(std::max(0.0, cutoff_d2)) -
         std::sqrt(std::max(0.0, d2));
}

struct PbctopoComponentCandidate {
  double continuity_path = std::numeric_limits<double>::infinity();
  double continuity_max_d2 = std::numeric_limits<double>::infinity();
  double continuity_min_margin_A =
      std::numeric_limits<double>::quiet_NaN();
  std::size_t continuity_margin_atom_a =
      std::numeric_limits<std::size_t>::max();
  std::size_t continuity_margin_atom_b =
      std::numeric_limits<std::size_t>::max();
  double anchor_mst2 = std::numeric_limits<double>::infinity();
  double anchor_max_d2 = std::numeric_limits<double>::infinity();
  PbctopoAnchorGraphScore anchor_graph;
  double principal_rg2_max = std::numeric_limits<double>::infinity();
  double principal_rg2_mid = std::numeric_limits<double>::infinity();
  double principal_rg2_min = std::numeric_limits<double>::infinity();
  double shape_kappa2 = std::numeric_limits<double>::infinity();
  double shape_asphericity = std::numeric_limits<double>::infinity();
  double shape_acylindricity = std::numeric_limits<double>::infinity();
  double component_mst2 = std::numeric_limits<double>::infinity();
  double component_rg2 = std::numeric_limits<double>::infinity();
  double component_kappa2 = std::numeric_limits<double>::infinity();
  double component_asphericity = std::numeric_limits<double>::infinity();
  double component_acylindricity = std::numeric_limits<double>::infinity();
  double bbox_volume = std::numeric_limits<double>::infinity();
  double rg2 = std::numeric_limits<double>::infinity();
  double shift_norm2 = std::numeric_limits<double>::infinity();
  double min_pair_d2 = std::numeric_limits<double>::infinity();
  std::size_t clash_atom_a = std::numeric_limits<std::size_t>::max();
  std::size_t clash_atom_b = std::numeric_limits<std::size_t>::max();
  std::size_t component_search_shell_radius = 0;
  bool search_domain_audit_known = false;
  bool search_domain_evidence_complete = false;
  bool selected_assignment_touches_shell_boundary = false;
  bool domain_expansion_attempted = false;
  bool domain_expansion_exhausted = false;
  bool bounded_domain_only = false;
  PbctopoDomainCompleteness domain_completeness =
      PbctopoDomainCompleteness::Unknown;
  bool ultra_clash = false;
  bool valid = false;
};

struct PbctopoShift {
  std::size_t shift_index = 26;
  int ix = 0;
  int iy = 0;
  int iz = 0;
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;
};

struct PbctopoPairShiftScore {
  std::size_t shift_index = 0;
  int ix = 0;
  int iy = 0;
  int iz = 0;
  std::size_t contact_atoms = 0;
  std::size_t lhs_contact_atoms = 0;
  std::size_t rhs_contact_atoms = 0;
  std::size_t contact_pairs = 0;
  double contact_fraction = 0.0;
  double soft_penalty = 0.0;
  double soft_penalty_mean = 0.0;
  double min_pair_d2 = std::numeric_limits<double>::infinity();
  double bbox_d2 = std::numeric_limits<double>::infinity();
  double shift_norm2 = std::numeric_limits<double>::infinity();
  double quality_score = -std::numeric_limits<double>::infinity();
  bool ultra_clash = false;
  bool valid = false;
  bool complete = false;
};

struct PbctopoGraphEdge {
  std::size_t a = 0;
  std::size_t b = 0;
  int dx = 0;
  int dy = 0;
  int dz = 0;
  std::size_t option_rank = 0;
  std::size_t contact_atoms = 0;
  std::size_t contact_pairs = 0;
  double contact_fraction = 0.0;
  double soft_penalty = 0.0;
  double soft_penalty_mean = 0.0;
  double min_pair_d2 = std::numeric_limits<double>::infinity();
  double bbox_d2 = std::numeric_limits<double>::infinity();
  double shift_norm2 = std::numeric_limits<double>::infinity();
  double confidence = 0.0;
  int cycle_support = 0;
  int cycle_mismatch = 0;
  bool ultra_clash = false;
  bool valid = false;
};

struct PbctopoCutScoreBreakdown {
  double principal_s = 0.0;
  double edge_scan_s = 0.0;
};

struct PbctopoStage1EdgeContext {
  std::size_t fixed_broken_edges = 0;
  double fixed_edge_stretch = 0.0;
  std::vector<std::size_t> variable_edge_indices;
};

struct PbctopoEdgeScanSoA {
  std::vector<std::size_t> a;
  std::vector<std::size_t> b;
  std::vector<double> ref_d2;
  std::vector<double> inv_ref_d2;
  std::vector<double> break_cutoff_d2;
};

struct PbctopoComponentPairShiftEntry {
  double min_pair_d2 = std::numeric_limits<double>::infinity();
  double bbox_d2 = std::numeric_limits<double>::infinity();
  std::size_t clash_atom_a = std::numeric_limits<std::size_t>::max();
  std::size_t clash_atom_b = std::numeric_limits<std::size_t>::max();
  bool hard_clash = false;
};

struct PbctopoPairShiftCandidate {
  Int3 delta;
  PbctopoPairShiftScore score;
  double weight = 0.0;
};

struct PbctopoPairShiftCandidateSet {
  std::size_t a = 0;
  std::size_t b = 0;
  std::vector<PbctopoPairShiftCandidate> candidates;
};

struct PbctopoIdentifiability {
  std::size_t evidence_component_count = 0;
  std::size_t evidence_edge_count = 0;
  std::size_t explicit_contact_evidence_edges = 0;
  std::size_t scored_contact_evidence_edges = 0;
  std::size_t evidence_connected_component_count = 0;
  std::size_t unconstrained_relative_components = 0;
  std::size_t relative_lattice_degrees_of_freedom = 0;
  std::size_t equivalent_best_assignments_within_domain = 0;
  std::size_t feasible_assignments_within_domain = 0;
  bool evidence_graph_connected = false;
  bool identified = false;
  bool objective_uniqueness_known = false;
  bool objective_unique_within_search_domain = false;
  bool feasible_assignment_set_enumerated = false;
  bool evidence_uniqueness_known = false;
  bool evidence_unique_within_search_domain = false;
  // Legacy aliases for objective uniqueness.
  bool unique_within_search_domain = false;
  bool uniqueness_search_exhaustive = false;
};

struct PbctopoCycleSyncMetrics {
  std::size_t cycle_support = 0;
  double cycle_mismatch = 0.0;
  std::size_t inconsistent_edge_count = 0;
  std::size_t fundamental_cycle_count = 0;
  double fundamental_cycle_residual_rms = 0.0;
  double fundamental_cycle_residual_max = 0.0;
};

struct PbctopoCycleSyncResult {
  bool valid = false;
  std::vector<Int3> offsets;
  PbctopoCycleSyncMetrics metrics;
};

struct PbctopoComponentLayoutSignature {
  std::size_t atom_count = 0;
  std::vector<std::uint64_t> component_hashes;
  bool valid = false;

  bool operator==(const PbctopoComponentLayoutSignature &) const = default;
};

struct PbctopoWarmState {
  PbctopoCutCandidate previous_projection_candidate;
  PbctopoCutCandidate kinematic_previous_virtual_candidate;
  PbctopoCutCandidate kinematic_latest_virtual_candidate;
  PbctopoComponentLayoutSignature previous_component_layout;
  std::vector<Int3> previous_component_offsets;
  std::vector<std::vector<Int3>> previous_stage1_delta_hints_by_component;
  std::vector<std::vector<Int3>> previous_stage2_delta_hints_by_component;
  std::size_t kinematic_previous_virtual_frame =
      std::numeric_limits<std::size_t>::max();
  std::size_t kinematic_latest_virtual_frame =
      std::numeric_limits<std::size_t>::max();
  bool has_projection_candidate = false;
  bool has_kinematic_previous_virtual_candidate = false;
  bool has_kinematic_latest_virtual_candidate = false;
  bool has_component_offsets = false;
  bool has_component_trial_hints = false;
};

struct ClashCellKey {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;

  bool operator==(const ClashCellKey &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }
};

struct ClashCellKeyHash {
  std::size_t operator()(const ClashCellKey &key) const noexcept {
    const auto mix = [](std::uint64_t value) noexcept {
      value += 0x9e3779b97f4a7c15ull;
      value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
      value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
      return value ^ (value >> 31);
    };
    const std::uint64_t hx = mix(static_cast<std::uint64_t>(key.x));
    const std::uint64_t hy = mix(static_cast<std::uint64_t>(key.y));
    const std::uint64_t hz = mix(static_cast<std::uint64_t>(key.z));
    return static_cast<std::size_t>(hx ^ (hy << 1) ^ (hz >> 1));
  }
};

inline bool pbctopo_make_cell_key(double x, double y, double z,
                                  double inv_cell,
                                  ClashCellKey &key) noexcept {
  if (!(inv_cell > 0.0) || !std::isfinite(inv_cell))
    return false;
  constexpr std::int64_t kCoordinateGuard = 1024;
  const long double min_coordinate = static_cast<long double>(
      std::numeric_limits<std::int64_t>::min() + kCoordinateGuard);
  const long double max_coordinate = static_cast<long double>(
      std::numeric_limits<std::int64_t>::max() - kCoordinateGuard);
  auto convert = [&](double coordinate, std::int64_t &cell) noexcept {
    if (!std::isfinite(coordinate))
      return false;
    const long double scaled = static_cast<long double>(coordinate) *
                               static_cast<long double>(inv_cell);
    const long double floored = std::floor(scaled);
    if (!std::isfinite(floored) || floored < min_coordinate ||
        floored > max_coordinate)
      return false;
    cell = static_cast<std::int64_t>(floored);
    return true;
  };
  return convert(x, key.x) && convert(y, key.y) && convert(z, key.z);
}

inline bool pbctopo_offset_cell_key(const ClashCellKey &key, int dx, int dy,
                                    int dz,
                                    ClashCellKey &offset) noexcept {
  const auto checked_add = [](std::int64_t value, int delta,
                              std::int64_t &result) noexcept {
    if (delta > 0 &&
        value > std::numeric_limits<std::int64_t>::max() - delta)
      return false;
    if (delta < 0 &&
        value < std::numeric_limits<std::int64_t>::min() - delta)
      return false;
    result = value + static_cast<std::int64_t>(delta);
    return true;
  };
  return checked_add(key.x, dx, offset.x) &&
         checked_add(key.y, dy, offset.y) &&
         checked_add(key.z, dz, offset.z);
}

struct PbctopoShiftedComponent {
  std::vector<Vec3> points;
  std::vector<std::size_t> atom_indices;
  std::vector<ClashCellKey> clash_keys;
  std::vector<ClashCellKey> contact_keys;
  std::unordered_map<ClashCellKey, std::vector<std::size_t>, ClashCellKeyHash>
      clash_grid;
  std::unordered_map<ClashCellKey, std::vector<std::size_t>, ClashCellKeyHash>
      contact_grid;
  bool spatial_keys_valid = true;
  Vec3 center{};
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  double translation_norm2 = 0.0;
};

struct PbctopoShiftedComponentDescriptor {
  Vec3 center{};
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  double translation_norm2 = 0.0;
};

struct PbctopoComponentBaseGeometry {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<std::size_t> atom_indices;
  Vec3 center{};
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
};

struct PbctopoTimingStats {
  std::size_t frames = 0;
  std::size_t coarse_candidates = 0;
  std::size_t refine_candidates = 0;
  std::size_t projection_clash_skip_frames = 0;
  std::size_t projection_clash_skip_evals = 0;
  std::size_t component_frames = 0;
  std::size_t component_shift_trials = 0;
  std::size_t component_graph_refine_coarse_score_calls = 0;
  std::size_t component_graph_refine_coarse_stage1_score_calls = 0;
  std::size_t component_graph_refine_coarse_stage2_score_calls = 0;
  std::size_t component_graph_refine_exact_rescore_calls = 0;
  std::size_t component_partial_checks = 0;
  std::size_t component_partial_prunes = 0;
  std::size_t component_bbox_skips = 0;
  std::size_t component_score_calls = 0;
  std::size_t component_pair_table_builds = 0;
  std::size_t component_pair_table_prunes = 0;
  std::size_t component_pair_table_all_atom_relative_checks = 0;
  std::size_t component_pair_table_all_atom_relative_rejects = 0;
  std::size_t component_pair_table_all_atom_materialized = 0;
  std::size_t component_pair_table_component_count = 0;
  std::size_t component_pair_table_shift_count = 0;
  std::size_t component_pair_table_relative_deltas = 0;
  std::size_t component_pair_table_relative_reuses = 0;
  std::size_t component_relative_domain_analytic_builds = 0;
  std::size_t component_relative_domain_generic_builds = 0;
  std::size_t component_pair_table_dense_entries_equivalent = 0;
  std::size_t component_pair_table_materialized_entries = 0;
  std::size_t component_pair_table_bbox_prefiltered = 0;
  std::size_t component_pair_table_sparse_rows = 0;
  std::size_t component_pair_table_peak_bytes_estimate = 0;
  std::size_t component_search_zero_current_mask_prunes = 0;
  std::size_t component_search_zero_future_mask_prunes = 0;
  std::size_t component_search_fixed_minor_components = 0;
  std::size_t component_search_fixed_minor_frames = 0;
  std::size_t component_search_shift_descriptors = 0;
  std::size_t component_search_shift_materialized = 0;
  std::size_t component_search_shift_domain_max = 0;
  std::size_t component_search_shift_warm_hits = 0;
  std::size_t component_search_seed_warm = 0;
  std::size_t component_search_seed_graph = 0;
  std::size_t component_search_seed_none = 0;
  std::size_t component_search_prefix_continuity_checks = 0;
  std::size_t component_search_prefix_continuity_prunes = 0;
  std::size_t component_search_prefix_continuity_hard_rejects = 0;
  std::size_t component_search_exact_budget_skips = 0;
  std::size_t component_audit_exact_frames = 0;
  std::size_t component_audit_exact_completed = 0;
  std::size_t component_audit_exact_incomplete = 0;
  std::size_t component_audit_exact_state_limit = 0;
  std::size_t component_audit_exact_wall_time_limit = 0;
  std::size_t component_audit_exact_memory_limit = 0;
  std::size_t component_audit_exact_reference_matches = 0;
  std::size_t component_audit_exact_reference_mismatches = 0;
  std::size_t component_search_exact_state_upper_bound = 0;
  std::size_t component_search_filtered_state_upper_bound = 0;
  std::size_t component_search_arc_consistency_passes = 0;
  std::size_t component_search_arc_consistency_revisions = 0;
  std::size_t component_search_arc_consistency_removals = 0;
  std::size_t component_search_arc_consistency_queue_pushes = 0;
  std::size_t component_search_arc_consistency_failures = 0;
  std::size_t component_search_exact_recoveries = 0;
  std::size_t component_large_mode_merge_frames = 0;
  std::size_t component_large_mode_pair_only_frames = 0;
  std::size_t component_large_mode_beam_frames = 0;
  std::size_t component_large_mode_merge_count = 0;
  std::size_t component_large_mode_internal_offsets_preserved = 0;
  std::size_t component_large_mode_internal_offset_nonzero = 0;
  std::size_t component_large_mode_graph_seed_count = 0;
  std::size_t component_large_mode_graph_seed_refined = 0;
  std::size_t component_large_mode_graph_seed_nonbaseline_win = 0;
  std::size_t component_large_mode_beam_expanded = 0;
  std::size_t component_large_mode_beam_kept = 0;
  std::size_t component_large_mode_beam_complete_assignments = 0;
  std::size_t component_large_mode_beam_final_rescored = 0;
  std::size_t component_large_mode_beam_final_rescore_all_frames = 0;
  std::size_t component_search_beam_provisional_children = 0;
  std::size_t component_search_beam_state_materializations = 0;
  std::size_t component_search_beam_state_copies_avoided = 0;
  std::size_t component_search_beam_parent_state_moves = 0;
  std::size_t component_search_beam_adaptive_retries = 0;
  std::size_t component_search_beam_adaptive_retry_successes = 0;
  std::size_t component_search_beam_adaptive_max_width = 0;
  std::size_t component_search_certification_survivors = 0;
  std::size_t component_search_certification_band_retained = 0;
  std::size_t component_large_mode_fallback_frames = 0;
  std::size_t pbctopo_certification_attempts = 0;
  std::size_t pbctopo_certification_successes = 0;
  std::size_t pbctopo_certification_assignment_rejects = 0;
  std::size_t pbctopo_certification_hard_rejects = 0;
  std::size_t pbctopo_output_certification_band_candidates = 0;
  std::size_t pbctopo_output_certification_band_peak = 0;
  std::size_t pbctopo_output_certification_band_rejects = 0;
  std::size_t pbctopo_output_certification_band_recoveries = 0;
  std::size_t pbctopo_output_certification_band_expanded_frames = 0;
  std::size_t pbctopo_output_certification_band_expanded_recoveries = 0;
  std::size_t pbctopo_output_certification_band_precheck_rejects = 0;
  std::size_t pbctopo_output_certification_band_rebuilds = 0;
  std::size_t pbctopo_output_certification_band_offset_bytes_peak = 0;
  std::size_t pbctopo_output_certification_band_point_bytes_avoided_peak = 0;
  std::size_t pbctopo_heavy_search_frames = 0;
  std::size_t pbctopo_heavy_search_all_atom_candidate_rejects = 0;
  std::size_t pbctopo_heavy_search_top_candidate_rejects = 0;
  std::size_t pbctopo_heavy_search_all_atom_ranking_recoveries = 0;
  std::size_t pbctopo_heavy_search_all_atom_band_exhausted = 0;
  std::size_t pbctopo_certification_topology_edges_checked = 0;
  std::size_t pbctopo_certification_ambiguous_topology_edges = 0;
  std::size_t pbctopo_certification_topology_edge_residuals = 0;
  std::size_t pbctopo_certification_topology_cycle_residuals = 0;
  std::size_t pbctopo_certification_atom_image_residuals = 0;
  std::size_t pbctopo_certification_assignment_point_residuals = 0;
  std::size_t pbctopo_certification_steric_pair_checks = 0;
  std::size_t pbctopo_certification_steric_clashes = 0;
  std::size_t pbctopo_certification_required_contacts_checked = 0;
  std::size_t pbctopo_certification_required_contacts_lost = 0;
  std::size_t pbctopo_required_interface_contacts_loaded = 0;
  std::size_t component_required_contact_constraint_builds = 0;
  std::size_t component_required_contact_constraint_pairs = 0;
  std::size_t component_required_contact_relative_deltas_checked = 0;
  std::size_t component_required_contact_relative_deltas_rejected = 0;
  std::size_t component_required_contact_infeasible = 0;
  std::size_t pbctopo_certification_explicit_interface_contacts_checked = 0;
  std::size_t pbctopo_certification_explicit_interface_contacts_lost = 0;
  std::size_t pbctopo_geometry_classifications = 0;
  std::size_t pbctopo_compact_unit_box_certified_frames = 0;
  std::size_t pbctopo_noncompact_certified_frames = 0;
  std::size_t pbctopo_extended_certified_frames = 0;
  std::size_t pbctopo_winding_rejected_candidates = 0;
  std::size_t pbctopo_extended_winding_rejected_candidates = 0;
  std::size_t pbctopo_owner_com_split_observations = 0;
  std::size_t pbctopo_warm_state_commits = 0;
  std::size_t pbctopo_warm_state_resets = 0;
  std::size_t pbctopo_warm_state_commit_rejects = 0;
  std::size_t pbctopo_warm_projection_rejects = 0;
  std::size_t pbctopo_warm_component_rejects = 0;
  std::size_t pbctopo_weak_observation_output_frames = 0;
  std::size_t pbctopo_contradicted_output_frames = 0;
  std::size_t pbctopo_selected_evidence_audits = 0;
  std::size_t pbctopo_selected_evidence_contacts_checked = 0;
  std::size_t pbctopo_selected_evidence_contacts_lost = 0;
  std::size_t pbctopo_eobs_construction_attempts = 0;
  std::size_t pbctopo_eobs_construction_completed = 0;
  std::size_t pbctopo_eobs_construction_failures = 0;
  std::size_t pbctopo_eobs_cell_setup_failures = 0;
  std::size_t pbctopo_eobs_atom_index_failures = 0;
  std::size_t pbctopo_eobs_mic_query_failures = 0;
  double pbctopo_selected_evidence_audit_s = 0.0;
  std::size_t pbctopo_weak_observation_warm_preserved = 0;
  std::size_t pbctopo_warm_layout_mismatches = 0;
  std::size_t pbctopo_diagnostic_only_candidates = 0;
  std::size_t pbctopo_virtual_geometry_reference_evals = 0;
  std::size_t pbctopo_virtual_geometry_reference_safe = 0;
  std::size_t pbctopo_virtual_geometry_reference_clashing = 0;
  std::size_t pbctopo_virtual_geometry_outlier_rejects = 0;
  std::size_t pbctopo_virtual_geometry_outlier_observations = 0;
  std::size_t component_graph_outlier_rejects = 0;
  std::size_t component_search_outlier_rejects = 0;
  std::size_t component_graph_outlier_observations = 0;
  std::size_t component_search_outlier_observations = 0;
  std::size_t component_score_continuity_rejects = 0;
  std::size_t component_score_anchor_rejects = 0;
  std::size_t component_score_prefix_prunes_after_continuity = 0;
  std::size_t component_score_prefix_prunes_after_anchor = 0;
  std::size_t component_score_prefix_prunes_after_principal = 0;
  std::size_t component_score_prefix_prunes_after_component_metrics = 0;
  std::size_t component_score_prefix_prunes_after_final = 0;
  std::size_t component_graph_edges = 0;
  std::size_t component_graph_input_shift_domain_max = 0;
  std::size_t component_graph_relative_shift_domain_max = 0;
  std::size_t component_graph_shift_descriptors = 0;
  std::size_t component_graph_shift_descriptor_domain = 0;
  std::size_t component_graph_shift_materialized = 0;
  std::size_t component_graph_shift_cache_entries_peak = 0;
  std::size_t component_graph_shift_cache_dense_entries_equivalent = 0;
  std::size_t component_graph_bbox_only_scores = 0;
  std::size_t component_graph_spatial_shortlist_builds = 0;
  std::size_t component_graph_spatial_shortlist_fallbacks = 0;
  std::size_t component_graph_spatial_shortlist_pairs = 0;
  std::size_t component_graph_spatial_shortlist_shift_scores_avoided = 0;
  std::size_t component_graph_contact_epoch_scratch_resizes = 0;
  std::size_t component_graph_contact_epoch_wrap_clears = 0;
  std::size_t component_graph_used = 0;
  std::size_t component_graph_primary_attempts = 0;
  std::size_t component_graph_primary_certified = 0;
  std::size_t component_graph_primary_virtual_skips = 0;
  std::size_t component_graph_primary_fallbacks = 0;
  std::size_t component_graph_primary_unidentified = 0;
  std::size_t component_graph_unidentified_output_rejects = 0;
  std::size_t component_image_shell_expansions = 0;
  std::size_t component_image_shell_retry_successes = 0;
  std::size_t component_image_shell_max_radius = 1;
  std::size_t component_image_shell_max_shift_count = 27;
  std::size_t component_warm_used = 0;
  std::size_t component_warm_skip_search = 0;
  std::size_t component_search_frames = 0;
  std::size_t component_graph_failed = 0;
  std::size_t component_graph_points_mismatch = 0;
  std::size_t component_graph_invalid = 0;
  std::size_t component_graph_ultraclash = 0;
  std::size_t component_graph_not_better_not_close = 0;
  std::size_t component_refine_warm_shift_hits = 0;
  std::size_t component_refine_shortlist_hint_hits = 0;
  std::size_t component_pair_warm_shift_hits = 0;
  std::size_t component_refine_fanout_reduced = 0;
  std::size_t hull_rescue_attempts = 0;
  std::size_t hull_rescue_successes = 0;
  std::size_t distance_rescue_attempts = 0;
  std::size_t distance_rescue_successes = 0;
  std::size_t pbci_fallback_frames = 0;
  std::size_t pbctopo_llps_local_frames = 0;
  std::size_t pbctopo_llps_local_units = 0;
  std::size_t pbctopo_llps_local_edges_used = 0;
  std::size_t pbctopo_llps_local_edges_skipped_cross_unit = 0;
  std::size_t pbctopo_llps_local_disconnected_units = 0;
  std::size_t component_graph_refine_score_calls = 0;
  std::size_t component_graph_refine_preclash_prunes = 0;
  std::size_t component_graph_refine_trial_build_incremental = 0;
  std::size_t component_graph_refine_trial_build_full = 0;
  std::size_t component_graph_refine_trial_build_global_rebase = 0;
  std::size_t component_graph_refine_stats_cached = 0;
  std::size_t component_graph_refine_stats_full = 0;
  std::size_t component_graph_refine_anchor_spatial_calls = 0;
  std::size_t component_graph_refine_anchor_spatial_fallbacks = 0;
  std::size_t component_graph_refine_anchor_spatial_disconnects = 0;
  std::size_t component_graph_refine_anchor_spatial_expansions = 0;
  std::size_t component_graph_refine_anchor_spatial_exact_certified = 0;
  std::size_t component_graph_refine_anchor_spatial_uncertified = 0;
  std::size_t component_graph_refine_anchor_spatial_max_k = 0;
  std::size_t pbctopo_anchor_order_edges = 0;
  std::size_t pbctopo_anchor_endpoint_edges = 0;
  std::size_t pbctopo_anchor_contact_pairs = 0;
  std::size_t pbctopo_anchor_intra_owner_contact_pairs = 0;
  std::size_t pbctopo_soft_observed_contact_pairs = 0;
  std::size_t pbctopo_hard_required_owner_pairs = 0;
  std::size_t pbctopo_anchor_residue_order_violations = 0;
  std::size_t pbctopo_anchor_ambiguous_frames = 0;
  std::size_t pbctopo_vdw_pair_checks = 0;
  std::size_t pbctopo_vdw_radius_pairs = 0;
  std::size_t pbctopo_vdw_radius_fallback_pairs = 0;
  std::size_t pbctopo_vdw_explicit_zero_lj_exempt_pairs = 0;
  std::size_t pbctopo_temporal_candidate_frames = 0;
  std::size_t pbctopo_temporal_candidates = 0;
  std::size_t pbctopo_temporal_certified_candidates = 0;
  std::size_t pbctopo_temporal_assignment_deduped = 0;
  std::size_t pbctopo_temporal_viterbi_frames = 0;
  std::size_t pbctopo_temporal_diagnostic_only_frames = 0;
  std::size_t pbctopo_temporal_proposed_changes = 0;
  std::size_t pbctopo_temporal_path_changes = 0;
  std::size_t pbctopo_temporal_replay_frames = 0;
  std::size_t pbctopo_temporal_replay_applied = 0;
  std::size_t pbctopo_temporal_replay_skipped = 0;
  std::size_t pbctopo_lattice_audits = 0;
  std::size_t pbctopo_lattice_invalid_frames = 0;
  std::size_t pbctopo_lattice_reduction_fallback_frames = 0;
  std::size_t pbctopo_lattice_query_failures = 0;
  std::size_t pbctopo_lattice_fatal_query_failures = 0;
  std::size_t pbctopo_lattice_optional_query_failures = 0;
  std::size_t pbctopo_lattice_query_ambiguities = 0;
  std::size_t pbctopo_lattice_query_budget_failures = 0;
  std::size_t pbctopo_lattice_worker_queries = 0;
  std::size_t pbctopo_lattice_worker_query_failures = 0;
  std::size_t pbctopo_lattice_worker_query_ambiguities = 0;
  std::size_t pbctopo_lattice_worker_query_budget_failures = 0;
  std::size_t component_cycle_sync_attempts = 0;
  std::size_t component_cycle_sync_used = 0;
  std::size_t component_cycle_sync_pair_candidates = 0;
  std::size_t component_cycle_sync_cycle_support = 0;
  double component_cycle_sync_cycle_mismatch = 0.0;
  std::size_t component_cycle_sync_inconsistent_edges = 0;
  std::size_t pbctopo_topology_bonds_available = 0;
  std::size_t pbctopo_topology_bond_edges_added = 0;
  std::size_t pbctopo_topology_bond_edges_skipped = 0;
  std::size_t pbctopo_geometry_edges_added = 0;
  std::size_t pbctopo_hard_connectivity_edges = 0;
  std::size_t pbctopo_soft_geometry_edges = 0;
  std::size_t pbctopo_steric_exclusion_pairs = 0;
  std::size_t pbctopo_component_logic_disabled_no_hard_graph = 0;
  std::size_t pbctopo_topology_duplicate_edges = 0;
  std::size_t pbctopo_topology_projected_bonds = 0;
  std::size_t pbctopo_topology_static_graph_frames = 0;
  std::size_t pbctopo_metadata_graph_frames = 0;
  std::size_t pbctopo_metadata_residue_groups = 0;
  std::size_t pbctopo_metadata_intra_edges = 0;
  std::size_t pbctopo_metadata_inter_edges = 0;
  std::size_t pbctopo_metadata_inter_hard_edges = 0;
  std::size_t pbctopo_metadata_inter_soft_edges = 0;
  std::size_t pbctopo_geometry_fallback_frames = 0;
  std::size_t component_context_builds = 0;
  std::size_t component_offset_extracts = 0;
  std::size_t apply_calls = 0;
  std::size_t early_projection_used = 0;
  std::size_t previous_seed_used = 0;
  std::size_t pbctopo_kinematic_seed_candidates = 0;
  std::size_t pbctopo_kinematic_seed_evaluated = 0;
  std::size_t pbctopo_kinematic_seed_deduped = 0;
  std::size_t pbctopo_kinematic_seed_best = 0;
  std::size_t pbctopo_kinematic_seed_history_resets = 0;
  std::size_t pbctopo_density_cut_planes_frames = 0;
  std::size_t pbctopo_density_cut_planes_x = 0;
  std::size_t pbctopo_density_cut_planes_y = 0;
  std::size_t pbctopo_density_cut_planes_z = 0;
  std::size_t pbctopo_density_cut_planes_dense_equiv_x = 0;
  std::size_t pbctopo_density_cut_planes_dense_equiv_y = 0;
  std::size_t pbctopo_density_cut_planes_dense_equiv_z = 0;
  std::size_t pbctopo_density_cut_plane_collisions = 0;
  std::size_t pbctopo_density_cut_plane_dense_fallbacks = 0;
  std::size_t pbctopo_virtual_discrete_frames = 0;
  std::size_t pbctopo_virtual_axis_states_x = 0;
  std::size_t pbctopo_virtual_axis_states_y = 0;
  std::size_t pbctopo_virtual_axis_states_z = 0;
  std::uint64_t pbctopo_virtual_state_space = 0;
  std::size_t pbctopo_virtual_states_evaluated = 0;
  std::size_t pbctopo_virtual_search_exhaustive_frames = 0;
  std::size_t pbctopo_virtual_seam_enumeration_exhaustive_frames = 0;
  std::size_t pbctopo_virtual_projection_scoring_exhaustive_frames = 0;
  std::size_t pbctopo_virtual_full_atom_verification_exhaustive_frames = 0;
  std::size_t pbctopo_virtual_search_shortlist_frames = 0;
  std::size_t pbctopo_virtual_branch_and_bound_frames = 0;
  std::size_t pbctopo_virtual_sweep_workspace_reuse_frames = 0;
  std::size_t pbctopo_virtual_sweep_events_x = 0;
  std::size_t pbctopo_virtual_sweep_events_y = 0;
  std::size_t pbctopo_virtual_sweep_events_z = 0;
  std::size_t pbctopo_virtual_sweep_workspace_peak_bytes = 0;
  std::size_t pbctopo_verify_geometry_buffer_reuses = 0;
  std::size_t pbctopo_frame_point_buffer_peak_bytes_estimate = 0;
  std::array<std::size_t, 8> component_search_depth_visits{};
  double append_s = 0.0;
  double heavy_subset_s = 0.0;
  double translation_cache_s = 0.0;
  double graph_s = 0.0;
  double pbctopo_local_graph_geometry_s = 0.0;
  double pbctopo_topology_edge_append_s = 0.0;
  double projection_edge_s = 0.0;
  double component_split_s = 0.0;
  double event_planes_s = 0.0;
  double pbctopo_density_cut_plane_build_s = 0.0;
  double pbctopo_virtual_seam_sweep_s = 0.0;
  double pbctopo_virtual_omitted_seam_lower_bound_min =
      std::numeric_limits<double>::infinity();
  double pbctopo_virtual_lower_bound_gap_min =
      std::numeric_limits<double>::infinity();
  double seed_s = 0.0;
  double coarse_s = 0.0;
  double refine_s = 0.0;
  double component_prep_s = 0.0;
  double component_search_s = 0.0;
  double component_graph_s = 0.0;
  double component_graph_precompute_s = 0.0;
  double component_graph_lazy_materialize_s = 0.0;
  double component_graph_spatial_shortlist_s = 0.0;
  double component_graph_pair_score_s = 0.0;
  double component_graph_tree_s = 0.0;
  double component_graph_refine_s = 0.0;
  double component_graph_refine_rebase_s = 0.0;
  double component_graph_refine_coarse_s = 0.0;
  double component_graph_refine_coarse_stage1_s = 0.0;
  double component_graph_refine_coarse_stage1_build_points_s = 0.0;
  double component_graph_refine_coarse_stage1_continuity_s = 0.0;
  double component_graph_refine_coarse_stage1_anchor_s = 0.0;
  double component_graph_refine_coarse_stage1_cut_score_s = 0.0;
  double component_graph_refine_coarse_stage1_cut_score_principal_s = 0.0;
  double component_graph_refine_coarse_stage1_cut_score_edge_scan_s = 0.0;
  double component_graph_refine_coarse_stage2_s = 0.0;
  double component_graph_refine_coarse_stage2_build_points_s = 0.0;
  double component_graph_refine_coarse_stage2_continuity_s = 0.0;
  double component_graph_refine_coarse_stage2_anchor_s = 0.0;
  double component_graph_refine_coarse_stage2_cut_score_s = 0.0;
  double component_graph_refine_build_points_s = 0.0;
  double component_graph_refine_continuity_s = 0.0;
  double component_graph_refine_anchor_s = 0.0;
  double component_graph_refine_anchor_spatial_build_s = 0.0;
  double component_graph_refine_anchor_spatial_query_s = 0.0;
  double component_graph_refine_anchor_dense_fallback_s = 0.0;
  double pbctopo_anchor_context_build_s = 0.0;
  double component_graph_refine_anchor_order_s = 0.0;
  double component_graph_refine_anchor_contact_s = 0.0;
  double component_graph_refine_cut_score_s = 0.0;
  double component_graph_refine_ultraclash_s = 0.0;
  double component_graph_refine_stats_s = 0.0;
  double component_cycle_sync_s = 0.0;
  double early_projection_s = 0.0;
  double warm_component_s = 0.0;
  double component_partial_s = 0.0;
  double component_score_s = 0.0;
  double component_pair_table_s = 0.0;
  double component_search_lazy_materialize_s = 0.0;
  double component_score_continuity_s = 0.0;
  double component_search_prefix_continuity_s = 0.0;
  double component_search_arc_consistency_s = 0.0;
  double component_score_anchor_s = 0.0;
  double component_score_stats_principal_s = 0.0;
  double component_score_component_metrics_s = 0.0;
  double component_context_s = 0.0;
  double component_offset_extract_s = 0.0;
  double apply_s = 0.0;
  double pbctopo_lattice_condition_max = 0.0;
};

enum class PbctopoRefineScorePhase {
  Exact,
  CoarseStage1,
  CoarseStage2,
};

struct PbctopoAnchorContactEdge {
  std::size_t a = 0;
  std::size_t b = 0;
  double cutoff_d2 = std::numeric_limits<double>::infinity();
  Int3 nearest_image{};
  std::size_t equivalent_nearest_images = 0;
  bool nearest_image_valid = false;
};

struct PbctopoContactHypothesisEdge {
  std::size_t a = 0;
  std::size_t b = 0;
  double cutoff_d2 = std::numeric_limits<double>::infinity();
};

struct PbctopoContactShiftHypothesis {
  std::size_t component_a = 0;
  std::size_t component_b = 0;
  std::uint64_t component_id_a = 0;
  std::uint64_t component_id_b = 0;
  Int3 relative_shift{};
  std::vector<PbctopoContactHypothesisEdge> edges;
};

struct PbctopoContactHypothesisSet {
  std::vector<PbctopoContactShiftHypothesis> hypotheses;
  PbctopoHash128 all_hypotheses_hash;
  PbctopoHash128 raw_observed_pair_hash;
  std::size_t raw_observed_pair_count = 0;
  std::size_t component_relation_count = 0;
  std::size_t ambiguous_component_relation_count = 0;
  std::size_t mic_ambiguity_failures = 0;
  std::size_t component_mapping_failures = 0;
  std::size_t internal_edges_ignored = 0;
  bool attempted = false;
  bool complete = false;
  std::string error;
};

struct PbctopoAnchorGraphContext {
  std::vector<PbctopoContinuityPair> ordered_edges;
  std::vector<PbctopoContinuityPair> endpoint_edges;
  std::vector<PbctopoAnchorContactEdge> soft_observed_contact_edges;
  PbctopoContactHypothesisSet contact_hypotheses;
  std::size_t anchor_count = 0;
  std::size_t residue_order_violations = 0;
  bool use_order_metrics = true;
  bool use_contact_metrics = true;
  bool soft_observed_contact_construction_attempted = false;
  bool soft_observed_contact_construction_complete = false;
  std::size_t soft_observed_contact_cell_setup_failures = 0;
  std::size_t soft_observed_contact_atom_index_failures = 0;
  std::size_t soft_observed_contact_mic_query_failures = 0;
  double box_m[3][3]{};
  double box_inv[3][3]{};
  bool has_box = false;
};

} // namespace titan_pbctopo

#endif
