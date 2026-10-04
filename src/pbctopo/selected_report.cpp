#include "selected_report.h"

#include "candidate_scoring.h"
#include "image_assignment.h"
#include "steric_clash.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace titan_pbctopo {

bool pbctopo_candidate_anchor_ambiguous(
    const PbctopoCutCandidate &candidate,
    std::span<const PbctopoCutCandidate> certified_pool) {
  for (const auto &pool_candidate : certified_pool) {
    if (candidate.assignment.valid && pool_candidate.assignment.valid &&
        same_pbctopo_image_assignment(candidate.assignment,
                                       pool_candidate.assignment)) {
      continue;
    }
    if (candidate_anchor_prefix_equivalent(candidate, pool_candidate) &&
        (better_cut_candidate(candidate, pool_candidate) ||
         better_cut_candidate(pool_candidate, candidate))) {
      return true;
    }
  }
  return false;
}

bool populate_pbctopo_selected_candidate_report(
    PbctopoFrameReport &report, const PbctopoCutCandidate &candidate,
    std::span<const PbctopoAtom> atoms, double fallback_clash_cutoff,
    double continuity_cutoff,
    std::span<const PbctopoCutCandidate> certified_pool) {
  (void)continuity_cutoff;
  report.hard_feasibility = candidate.hard_feasibility;
  report.hard_feasible = candidate.valid && candidate.hard_feasibility.certified();
  report.assignment_signature = candidate.assignment.signature;
  report.component_layout_signature = candidate.assignment.layout_signature;
  report.assignment_component_count = candidate.assignment.component_offsets.size();
  report.selected_assignment_boundary_known =
      candidate.search_domain_audit_known;
  report.selected_assignment_touches_shell_boundary =
      candidate.search_domain_audit_known &&
      candidate.selected_assignment_touches_shell_boundary;
  if (candidate.search_domain_audit_known) {
    report.component_search_shell_radius =
        candidate.component_search_shell_radius;
  }
  report.domain_expansion_attempted =
      report.domain_expansion_attempted ||
      candidate.domain_expansion_attempted;
  report.domain_expansion_exhausted =
      report.domain_expansion_exhausted ||
      candidate.domain_expansion_exhausted;
  report.bounded_domain_only =
      report.bounded_domain_only || candidate.bounded_domain_only;
  report.domain_completeness = candidate.domain_completeness;

  const auto &hard = candidate.hard_feasibility;
  const bool topology_valid =
      hard.topology_edges_available > 0 &&
      hard.topology_edges_checked >= hard.topology_edges_available &&
      hard.topology_edge_residuals == 0;
  const bool metadata_valid =
      hard.metadata_edges_available > 0 &&
      hard.metadata_edges_checked >= hard.metadata_edges_available &&
      hard.metadata_edge_image_residuals == 0 &&
      hard.metadata_edge_distance_residuals == 0;
  if (topology_valid && metadata_valid) {
    report.hard_graph_source = PbctopoHardGraphSource::MixedHard;
  } else if (topology_valid) {
    report.hard_graph_source = PbctopoHardGraphSource::ExplicitTopology;
  } else if (metadata_valid) {
    report.hard_graph_source = PbctopoHardGraphSource::ValidatedMetadata;
  } else {
    report.hard_graph_source = PbctopoHardGraphSource::None;
  }

  report.geometry_classification = candidate.geometry_classification;
  report.current_basis_fundamental_parallelepiped_certified =
      report.hard_feasible && candidate.geometry_classification.evaluated &&
      candidate.geometry_classification
          .fits_current_basis_fundamental_parallelepiped;
  report.compact_unit_box_certified =
      report.hard_feasible && candidate.geometry_classification.evaluated &&
      candidate.geometry_classification.geometry_class ==
          PbctopoGeometryClass::CompactUnitBox;
  report.split_owner_pairs = candidate.split_owner_pairs;
  report.split_owner_gap_scaled = candidate.max_owner_raw_minus_minimg_delta;

  report.anchor_graph = candidate.anchor_graph;
  const bool anchor_ambiguous =
      pbctopo_candidate_anchor_ambiguous(candidate, certified_pool);
  if (anchor_ambiguous)
    report.anchor_graph.ambiguity_count =
        std::max<std::size_t>(1, report.anchor_graph.ambiguity_count);
  report.selected_score = make_score_vector(candidate);
  report.selected_score.anchor_graph.ambiguity_count =
      report.anchor_graph.ambiguity_count;

  const double nan = std::numeric_limits<double>::quiet_NaN();
  report.clash_margin_A = nan;
  report.continuity_margin_A = nan;
  report.continuity_margin_atom_a = candidate.continuity_margin_atom_a;
  report.continuity_margin_atom_b = candidate.continuity_margin_atom_b;
  report.anchor_max_edge_A = nan;
  if (candidate.hard_feasibility.evaluated &&
      candidate.hard_feasibility.steric_audit_available &&
      std::isfinite(candidate.hard_feasibility.min_steric_margin_A)) {
    // The certified audit's own minimum margin, over every pair it checked.
    report.clash_margin_A = candidate.hard_feasibility.min_steric_margin_A;
  } else if (std::isfinite(candidate.min_pair_d2)) {
    // No hard audit: the margin of the closest pair is the best available.
    double active_clash_cutoff = fallback_clash_cutoff;
    if (candidate.clash_atom_a < atoms.size() &&
        candidate.clash_atom_b < atoms.size()) {
      active_clash_cutoff =
          pbctopo_pair_clash_threshold(atoms[candidate.clash_atom_a],
                                       atoms[candidate.clash_atom_b],
                                       fallback_clash_cutoff)
              .cutoff;
    }
    report.clash_margin_A =
        std::sqrt(std::max(0.0, candidate.min_pair_d2)) - active_clash_cutoff;
  }
  if (std::isfinite(candidate.continuity_min_margin_A))
    report.continuity_margin_A = candidate.continuity_min_margin_A;
  if (std::isfinite(candidate.anchor_max_d2)) {
    report.anchor_max_edge_A =
        std::sqrt(std::max(0.0, candidate.anchor_max_d2));
  }
  return anchor_ambiguous;
}

} // namespace titan_pbctopo
