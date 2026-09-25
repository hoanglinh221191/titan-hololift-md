#ifndef TITAN_PBCTOPO_CERTIFICATION_H
#define TITAN_PBCTOPO_CERTIFICATION_H

#include "lattice_math.h"
#include "types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_set>

namespace titan_pbctopo {

struct PbctopoImageEquationResiduals {
  std::size_t topology_edges_checked = 0;
  std::size_t ambiguous_topology_edges = 0;
  std::size_t topology_edge_residuals = 0;
  std::size_t metadata_edges_checked = 0;
  std::size_t ambiguous_metadata_edges = 0;
  std::size_t metadata_edge_image_residuals = 0;
  std::size_t metadata_edge_distance_residuals = 0;
  double max_metadata_edge_distance_error_A = 0.0;
  std::size_t atom_image_residuals = 0;
  std::size_t topology_cycle_residuals = 0;
  std::array<Int3, 3> topological_winding_basis{};
  std::size_t topological_winding_rank = 0;
  double max_atom_image_error = 0.0;
  bool input_valid = false;
};

struct PbctopoRequiredContactResiduals {
  std::size_t contacts_checked = 0;
  std::size_t contacts_lost = 0;
  std::size_t topology_contacts_checked = 0;
  std::size_t topology_contacts_lost = 0;
  std::size_t explicit_contacts_checked = 0;
  std::size_t explicit_contacts_lost = 0;
  double explicit_weight_checked = 0.0;
  double explicit_weight_lost = 0.0;
  bool explicit_constraints_requested = false;
  bool input_valid = false;
};

struct PbctopoAssignmentPointResiduals {
  std::size_t point_residuals = 0;
  double max_point_error = 0.0;
  bool input_valid = false;
};

enum class PbctopoStericTerminationReason : std::uint8_t {
  Completed,
  FirstHardClash,
  InvalidInput,
  NumericalFailure,
};

constexpr const char *pbctopo_steric_termination_reason_name(
    PbctopoStericTerminationReason reason) noexcept {
  switch (reason) {
  case PbctopoStericTerminationReason::Completed:
    return "completed";
  case PbctopoStericTerminationReason::FirstHardClash:
    return "first_hard_clash";
  case PbctopoStericTerminationReason::InvalidInput:
    return "invalid_input";
  case PbctopoStericTerminationReason::NumericalFailure:
    return "numerical_failure";
  }
  return "invalid_input";
}

struct PbctopoStericAudit {
  std::size_t pair_checks = 0;
  std::size_t hard_clashes = 0;
  std::size_t metadata_internal_pairs_checked = 0;
  std::size_t metadata_internal_clashes = 0;
  std::size_t force_field_radius_atoms = 0;
  std::size_t explicit_zero_lj_radius_atoms = 0;
  std::size_t inferred_radius_atoms = 0;
  std::size_t generic_fallback_radius_atoms = 0;
  std::size_t unavailable_radius_atoms = 0;
  // The pair with the smallest steric margin (the certificate's deciding
  // pair) and its squared distance.
  double min_margin_A = std::numeric_limits<double>::infinity();
  double min_margin_pair_d2 = std::numeric_limits<double>::infinity();
  std::size_t clash_atom_a = std::numeric_limits<std::size_t>::max();
  std::size_t clash_atom_b = std::numeric_limits<std::size_t>::max();
  // The closest intercomponent pair among the pairs the steric search
  // examined, which is not the same pair in general.  Only pairs within the
  // search radius are examined, so this is the minimum over that radius, not
  // a global minimum; it is infinite when no pair came within it.
  double min_pair_d2 = std::numeric_limits<double>::infinity();
  std::size_t min_pair_atom_a = std::numeric_limits<std::size_t>::max();
  std::size_t min_pair_atom_b = std::numeric_limits<std::size_t>::max();
  bool decision_complete = false;
  bool pair_enumeration_complete = false;
  PbctopoStericTerminationReason termination_reason =
      PbctopoStericTerminationReason::InvalidInput;
  // Legacy alias retained for the schema-6 hard-certificate field.
  bool scan_complete = false;
  bool input_valid = false;
};

PbctopoImageEquationResiduals evaluate_pbctopo_image_equations(
    std::span<const PbctopoAtom> wrapped_atoms,
    std::span<const Vec3> placed_points,
    std::span<const PbctopoLocalEdge> local_edges,
    const PbctopoLatticeMetric &lattice,
    double image_tolerance = 1.0e-6);

PbctopoRequiredContactResiduals evaluate_pbctopo_required_contacts(
    std::span<const PbctopoAtom> atoms,
    std::span<const Vec3> placed_points,
    std::span<const PbctopoLocalEdge> local_edges,
    std::span<const PbctopoRequiredContact> explicit_contacts = {});

PbctopoAssignmentPointResiduals
evaluate_pbctopo_assignment_point_consistency(
    std::span<const PbctopoAtom> component_reference_atoms,
    std::span<const Vec3> placed_points,
    std::span<const std::size_t> atom_component_index,
    const PbctopoImageAssignment &assignment,
    const PbctopoLatticeMetric &lattice,
    double image_tolerance = 1.0e-6);

PbctopoStericAudit evaluate_pbctopo_final_sterics(
    std::span<const PbctopoAtom> atoms,
    std::span<const Vec3> placed_points,
    std::span<const std::size_t> atom_component_index,
    const std::unordered_set<std::uint64_t> &excluded_pairs,
    double fallback_cutoff,
    PbctopoTimingStats *timing = nullptr,
    const PbctopoLatticeMetric *wrapped_lattice = nullptr);

PbctopoFixedCertificateAudit evaluate_pbctopo_fixed_certificate_invariants(
    std::span<const PbctopoAtom> wrapped_atoms,
    std::span<const Vec3> component_reference_points,
    std::span<const std::size_t> atom_component_index,
    std::span<const PbctopoLocalEdge> local_edges,
    std::span<const PbctopoRequiredContact> explicit_contacts,
    const std::unordered_set<std::uint64_t> &excluded_pairs,
    double fallback_cutoff, const PbctopoLatticeMetric &wrapped_lattice,
    PbctopoTimingStats *timing = nullptr);

PbctopoGeometryClassification classify_pbctopo_final_geometry(
    std::span<const Vec3> placed_points,
    const PbctopoLatticeMetric &lattice,
    double unit_box_tolerance = 1.0e-6,
    const PbctopoImageEquationResiduals *image_equations = nullptr);

PbctopoHardFeasibility evaluate_pbctopo_hard_feasibility(
    const PbctopoCutCandidate &candidate,
    const PbctopoImageEquationResiduals &image_equations,
    const PbctopoAssignmentPointResiduals &assignment_points,
    const PbctopoStericAudit &steric_audit,
    PbctopoRequiredContactResiduals required_contacts = {});

bool certify_pbctopo_candidate(PbctopoCutCandidate &candidate,
                               const PbctopoImageEquationResiduals &image_equations,
                               const PbctopoAssignmentPointResiduals &assignment_points,
                               const PbctopoStericAudit &steric_audit,
                               PbctopoRequiredContactResiduals required_contacts = {});

} // namespace titan_pbctopo

#endif
