#include "candidate_scoring.h"

#include "anchor_graph.h"
#include "candidate_order.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace titan_pbctopo {
namespace {

constexpr PbctopoMetricResolution kPrincipalRgResolution{5.0, 0.0};
constexpr PbctopoMetricResolution kShapeKappaResolution{0.02, 0.0};
constexpr PbctopoMetricResolution kShapeMomentResolution{5.0, 0.0};

bool principal_rg2_significantly_worse(double value, double reference) {
  return compare_pbctopo_metric(value, reference, kPrincipalRgResolution) > 0;
}

bool shape_kappa_significantly_worse(double value, double reference) {
  return compare_pbctopo_metric(value, reference, kShapeKappaResolution) > 0;
}

bool shape_moment_significantly_worse(double value, double reference) {
  return compare_pbctopo_metric(value, reference, kShapeMomentResolution) > 0;
}

template <typename Candidate>
bool candidate_has_continuity_score(const Candidate &candidate) {
  return std::isfinite(candidate.continuity_max_d2) &&
         std::isfinite(candidate.continuity_path);
}

template <typename Candidate>
int compare_candidate_continuity_score(const Candidate &lhs,
                                       const Candidate &rhs) {
  const bool lhs_available = candidate_has_continuity_score(lhs);
  const bool rhs_available = candidate_has_continuity_score(rhs);
  if (lhs_available != rhs_available)
    return lhs_available ? -1 : 1;
  return lhs_available
             ? compare_pbctopo_double_metric(lhs.continuity_path,
                                             rhs.continuity_path)
             : 0;
}

template <typename Candidate>
int compare_candidate_anchor_prefix(const Candidate &lhs,
                                    const Candidate &rhs) {
  const int graph_cmp = compare_anchor_graph_score(lhs.anchor_graph,
                                                   rhs.anchor_graph);
  if (graph_cmp != 0)
    return graph_cmp;
  const bool lhs_has_anchor =
      std::isfinite(lhs.anchor_max_d2) && std::isfinite(lhs.anchor_mst2);
  const bool rhs_has_anchor =
      std::isfinite(rhs.anchor_max_d2) && std::isfinite(rhs.anchor_mst2);
  if (lhs_has_anchor != rhs_has_anchor)
    return lhs_has_anchor ? -1 : 1;
  if (!lhs_has_anchor)
    return 0;
  int cmp = compare_pbctopo_double_metric(lhs.anchor_max_d2,
                                          rhs.anchor_max_d2);
  if (cmp != 0)
    return cmp;
  return compare_pbctopo_double_metric(lhs.anchor_mst2, rhs.anchor_mst2);
}

template <typename Candidate>
bool anchor_prefix_equivalent(const Candidate &lhs, const Candidate &rhs) {
  if (lhs.broken_edges != rhs.broken_edges)
    return false;
  if (compare_candidate_continuity_score(lhs, rhs) != 0)
    return false;
  return compare_candidate_anchor_prefix(lhs, rhs) == 0;
}

} // namespace

int compare_pbctopo_double_metric(double lhs, double rhs, double eps) {
  return compare_pbctopo_metric(lhs, rhs, {eps, 0.0});
}

int compare_principal_rg2(double lhs, double rhs) {
  return compare_pbctopo_metric(lhs, rhs, kPrincipalRgResolution);
}

int compare_shape_kappa(double lhs, double rhs) {
  return compare_pbctopo_metric(lhs, rhs, kShapeKappaResolution);
}

int compare_shape_moment(double lhs, double rhs) {
  return compare_pbctopo_metric(lhs, rhs, kShapeMomentResolution);
}

PbctopoScoreVector make_score_vector(const PbctopoCutCandidate &candidate) {
  PbctopoScoreVector score;
  score.valid = candidate.valid;
  score.ultra_clash = candidate.ultra_clash;
  score.broken_edges = candidate.broken_edges;
  score.continuity_max_d2 = candidate.continuity_max_d2;
  score.continuity_path = candidate.continuity_path;
  score.anchor_max_d2 = candidate.anchor_max_d2;
  score.anchor_mst2 = candidate.anchor_mst2;
  score.anchor_graph = candidate.anchor_graph;
  score.principal_rg2_max = candidate.principal_rg2_max;
  score.principal_rg2_mid = candidate.principal_rg2_mid;
  score.principal_rg2_min = candidate.principal_rg2_min;
  score.shape_kappa2 = candidate.shape_kappa2;
  score.shape_asphericity = candidate.shape_asphericity;
  score.shape_acylindricity = candidate.shape_acylindricity;
  score.edge_stretch = candidate.edge_stretch;
  score.shift_norm2 = candidate.shift_norm2;
  score.gap_score = candidate.gap_score;
  return score;
}

PbctopoCutCandidate component_candidate_to_cut_candidate(
    const PbctopoComponentCandidate &candidate) {
  PbctopoCutCandidate cut;
  cut.continuity_path = candidate.continuity_path;
  cut.continuity_max_d2 = candidate.continuity_max_d2;
  cut.continuity_min_margin_A = candidate.continuity_min_margin_A;
  cut.continuity_margin_atom_a = candidate.continuity_margin_atom_a;
  cut.continuity_margin_atom_b = candidate.continuity_margin_atom_b;
  cut.anchor_mst2 = candidate.anchor_mst2;
  cut.anchor_max_d2 = candidate.anchor_max_d2;
  cut.anchor_graph = candidate.anchor_graph;
  cut.principal_rg2_max = candidate.principal_rg2_max;
  cut.principal_rg2_mid = candidate.principal_rg2_mid;
  cut.principal_rg2_min = candidate.principal_rg2_min;
  cut.bbox_volume = candidate.bbox_volume;
  cut.rg2 = candidate.rg2;
  cut.shape_kappa2 = candidate.shape_kappa2;
  cut.shape_asphericity = candidate.shape_asphericity;
  cut.shape_acylindricity = candidate.shape_acylindricity;
  cut.broken_edges = 0;
  cut.edge_stretch = 0.0;
  cut.shift_norm2 = candidate.shift_norm2;
  cut.min_pair_d2 = candidate.min_pair_d2;
  cut.clash_atom_a = candidate.clash_atom_a;
  cut.clash_atom_b = candidate.clash_atom_b;
  cut.component_search_shell_radius =
      candidate.component_search_shell_radius;
  cut.search_domain_audit_known = candidate.search_domain_audit_known;
  cut.search_domain_evidence_complete =
      candidate.search_domain_evidence_complete;
  cut.selected_assignment_touches_shell_boundary =
      candidate.selected_assignment_touches_shell_boundary;
  cut.domain_expansion_attempted = candidate.domain_expansion_attempted;
  cut.domain_expansion_exhausted = candidate.domain_expansion_exhausted;
  cut.bounded_domain_only = candidate.bounded_domain_only;
  cut.domain_completeness = candidate.domain_completeness;
  cut.ultra_clash = candidate.ultra_clash;
  cut.valid = candidate.valid;
  return cut;
}

bool candidate_anchor_prefix_equivalent(const PbctopoCutCandidate &lhs,
                                        const PbctopoCutCandidate &rhs) {
  return anchor_prefix_equivalent(lhs, rhs);
}

bool component_candidate_is_outlier(const PbctopoCutCandidate &candidate,
                                    const PbctopoCutCandidate *reference) {
  if (!candidate.valid || candidate.ultra_clash || reference == nullptr ||
      !reference->valid) {
    return false;
  }
  return principal_rg2_significantly_worse(candidate.principal_rg2_max,
                                           reference->principal_rg2_max) &&
         (shape_moment_significantly_worse(candidate.shape_asphericity,
                                           reference->shape_asphericity) ||
          shape_kappa_significantly_worse(candidate.shape_kappa2,
                                          reference->shape_kappa2));
}

const char *candidate_skip_search_reject_reason(
    const PbctopoCutCandidate &candidate,
    const PbctopoCutCandidate &reference) {
  if (!candidate.valid || candidate.ultra_clash)
    return "invalid_or_ultraclash";
  if (!reference.valid || reference.ultra_clash)
    return nullptr;
  if (candidate.broken_edges > reference.broken_edges)
    return "broken_edges";
  if (std::isfinite(candidate.continuity_path) &&
      std::isfinite(reference.continuity_path) &&
      candidate.continuity_path >
          std::max(reference.continuity_path * 1.05,
                   reference.continuity_path + 0.01))
    return "continuity_path";
  if (candidate.anchor_graph.residue_order_violations >
      reference.anchor_graph.residue_order_violations)
    return "anchor_residue_order_violations";
  if (std::isfinite(candidate.anchor_graph.endpoint_image_gap) &&
      std::isfinite(reference.anchor_graph.endpoint_image_gap) &&
      candidate.anchor_graph.endpoint_image_gap >
          reference.anchor_graph.endpoint_image_gap + 1.0e-6)
    return "anchor_endpoint_image_gap";
  if (std::isfinite(candidate.anchor_graph.order_max_d2) &&
      std::isfinite(reference.anchor_graph.order_max_d2) &&
      candidate.anchor_graph.order_max_d2 >
          reference.anchor_graph.order_max_d2 + 1.0e-6)
    return "anchor_order_max_d2";
  if (std::isfinite(candidate.anchor_graph.order_path_d2) &&
      std::isfinite(reference.anchor_graph.order_path_d2) &&
      candidate.anchor_graph.order_path_d2 >
          reference.anchor_graph.order_path_d2 + 1.0e-6)
    return "anchor_order_path_d2";
  if (std::isfinite(candidate.anchor_max_d2) &&
      std::isfinite(reference.anchor_max_d2) &&
      candidate.anchor_max_d2 >
          std::max(reference.anchor_max_d2 * 1.10,
                   reference.anchor_max_d2 + 25.0))
    return "anchor_max_d2";
  if (std::isfinite(candidate.anchor_mst2) &&
      std::isfinite(reference.anchor_mst2) &&
      candidate.anchor_mst2 >
          std::max(reference.anchor_mst2 * 1.10,
                   reference.anchor_mst2 + 100.0))
    return "anchor_mst2";
  if (principal_rg2_significantly_worse(candidate.principal_rg2_max,
                                        reference.principal_rg2_max))
    return "principal_rg2_max";
  if (principal_rg2_significantly_worse(candidate.principal_rg2_mid,
                                        reference.principal_rg2_mid))
    return "principal_rg2_mid";
  if (principal_rg2_significantly_worse(candidate.principal_rg2_min,
                                        reference.principal_rg2_min))
    return "principal_rg2_min";
  if (shape_kappa_significantly_worse(candidate.shape_kappa2,
                                      reference.shape_kappa2))
    return "shape_kappa2";
  if (shape_moment_significantly_worse(candidate.shape_asphericity,
                                       reference.shape_asphericity))
    return "shape_asphericity";
  if (shape_moment_significantly_worse(candidate.shape_acylindricity,
                                       reference.shape_acylindricity))
    return "shape_acylindricity";
  if (candidate.edge_stretch >
      std::max(reference.edge_stretch * 1.10,
               reference.edge_stretch + 0.05))
    return "edge_stretch";
  if (std::isfinite(reference.min_pair_d2) &&
      candidate.min_pair_d2 + 1.0e-6 <
          std::min(reference.min_pair_d2 * 0.90,
                   reference.min_pair_d2 - 0.05))
    return "min_pair_d2";
  return nullptr;
}

bool candidate_close_enough_to_skip_search(
    const PbctopoCutCandidate &candidate,
    const PbctopoCutCandidate &reference) {
  return candidate_skip_search_reject_reason(candidate, reference) == nullptr;
}

bool candidate_has_projection_clash_margin(
    const PbctopoCutCandidate &candidate, double clash_cutoff) {
  if (!candidate.valid || candidate.ultra_clash ||
      !std::isfinite(candidate.min_pair_d2))
    return false;
  const double margin_distance =
      std::max(clash_cutoff * 1.10, clash_cutoff + 0.20);
  return candidate.min_pair_d2 >= margin_distance * margin_distance;
}

bool better_stage1_cut_candidate(const PbctopoCutCandidate &lhs,
                                 const PbctopoCutCandidate &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  if (lhs.broken_edges != rhs.broken_edges)
    return lhs.broken_edges < rhs.broken_edges;
  int cmp = compare_candidate_continuity_score(lhs, rhs);
  if (cmp != 0)
    return cmp < 0;
  cmp = compare_candidate_anchor_prefix(lhs, rhs);
  if (cmp != 0)
    return cmp < 0;
  if ((cmp = compare_principal_rg2(lhs.principal_rg2_max,
                                   rhs.principal_rg2_max)) != 0 ||
      (cmp = compare_principal_rg2(lhs.principal_rg2_mid,
                                   rhs.principal_rg2_mid)) != 0 ||
      (cmp = compare_principal_rg2(lhs.principal_rg2_min,
                                   rhs.principal_rg2_min)) != 0 ||
      (cmp = compare_shape_kappa(lhs.shape_kappa2, rhs.shape_kappa2)) != 0 ||
      (cmp = compare_shape_moment(lhs.shape_asphericity,
                                  rhs.shape_asphericity)) != 0 ||
      (cmp = compare_shape_moment(lhs.shape_acylindricity,
                                  rhs.shape_acylindricity)) != 0 ||
      (cmp = compare_pbctopo_double_metric(lhs.edge_stretch,
                                           rhs.edge_stretch)) != 0)
    return cmp < 0;
  cmp = compare_pbctopo_double_metric(lhs.gap_score, rhs.gap_score, 1.0e-9);
  return cmp > 0;
}

bool better_preclash_cut_candidate(const PbctopoCutCandidate &lhs,
                                   const PbctopoCutCandidate &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  if (lhs.broken_edges != rhs.broken_edges)
    return lhs.broken_edges < rhs.broken_edges;
  int cmp = compare_candidate_continuity_score(lhs, rhs);
  if (cmp != 0)
    return cmp < 0;
  cmp = compare_candidate_anchor_prefix(lhs, rhs);
  if (cmp != 0)
    return cmp < 0;
  if ((cmp = compare_principal_rg2(lhs.principal_rg2_max,
                                   rhs.principal_rg2_max)) != 0 ||
      (cmp = compare_principal_rg2(lhs.principal_rg2_mid,
                                   rhs.principal_rg2_mid)) != 0 ||
      (cmp = compare_principal_rg2(lhs.principal_rg2_min,
                                   rhs.principal_rg2_min)) != 0 ||
      (cmp = compare_shape_kappa(lhs.shape_kappa2, rhs.shape_kappa2)) != 0 ||
      (cmp = compare_shape_moment(lhs.shape_asphericity,
                                  rhs.shape_asphericity)) != 0 ||
      (cmp = compare_shape_moment(lhs.shape_acylindricity,
                                  rhs.shape_acylindricity)) != 0 ||
      (cmp = compare_pbctopo_double_metric(lhs.edge_stretch,
                                           rhs.edge_stretch)) != 0 ||
      (cmp = compare_pbctopo_double_metric(lhs.shift_norm2,
                                           rhs.shift_norm2)) != 0)
    return cmp < 0;
  cmp = compare_pbctopo_double_metric(lhs.gap_score, rhs.gap_score, 1.0e-9);
  return cmp > 0;
}

bool better_cut_candidate(const PbctopoCutCandidate &lhs,
                          const PbctopoCutCandidate &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  if (lhs.ultra_clash != rhs.ultra_clash)
    return rhs.ultra_clash;
  if (lhs.broken_edges != rhs.broken_edges)
    return lhs.broken_edges < rhs.broken_edges;
  int cmp = compare_candidate_continuity_score(lhs, rhs);
  if (cmp != 0)
    return cmp < 0;
  cmp = compare_candidate_anchor_prefix(lhs, rhs);
  if (cmp != 0)
    return cmp < 0;
  if ((cmp = compare_principal_rg2(lhs.principal_rg2_max,
                                   rhs.principal_rg2_max)) != 0 ||
      (cmp = compare_principal_rg2(lhs.principal_rg2_mid,
                                   rhs.principal_rg2_mid)) != 0 ||
      (cmp = compare_principal_rg2(lhs.principal_rg2_min,
                                   rhs.principal_rg2_min)) != 0 ||
      (cmp = compare_shape_kappa(lhs.shape_kappa2, rhs.shape_kappa2)) != 0 ||
      (cmp = compare_shape_moment(lhs.shape_asphericity,
                                  rhs.shape_asphericity)) != 0 ||
      (cmp = compare_shape_moment(lhs.shape_acylindricity,
                                  rhs.shape_acylindricity)) != 0 ||
      (cmp = compare_pbctopo_double_metric(lhs.edge_stretch,
                                           rhs.edge_stretch)) != 0)
    return cmp < 0;
  cmp = compare_pbctopo_double_metric(lhs.min_pair_d2, rhs.min_pair_d2,
                                      1.0e-9);
  if (cmp != 0)
    return cmp > 0;
  if ((cmp = compare_pbctopo_double_metric(lhs.shift_norm2,
                                           rhs.shift_norm2)) != 0 ||
      (cmp = compare_pbctopo_double_metric(lhs.cut_x, rhs.cut_x, 1.0e-9)) !=
          0 ||
      (cmp = compare_pbctopo_double_metric(lhs.cut_y, rhs.cut_y, 1.0e-9)) !=
          0)
    return cmp < 0;
  return compare_pbctopo_double_metric(lhs.cut_z, rhs.cut_z, 1.0e-9) < 0;
}

bool better_component_candidate(const PbctopoComponentCandidate &lhs,
                                const PbctopoComponentCandidate &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  if (lhs.ultra_clash != rhs.ultra_clash)
    return rhs.ultra_clash;
  int cmp = compare_candidate_continuity_score(lhs, rhs);
  if (cmp != 0)
    return cmp < 0;
  cmp = compare_candidate_anchor_prefix(lhs, rhs);
  if (cmp != 0)
    return cmp < 0;
  if ((cmp = compare_principal_rg2(lhs.principal_rg2_max,
                                   rhs.principal_rg2_max)) != 0 ||
      (cmp = compare_principal_rg2(lhs.principal_rg2_mid,
                                   rhs.principal_rg2_mid)) != 0 ||
      (cmp = compare_principal_rg2(lhs.principal_rg2_min,
                                   rhs.principal_rg2_min)) != 0 ||
      (cmp = compare_pbctopo_double_metric(lhs.component_mst2,
                                           rhs.component_mst2)) != 0 ||
      (cmp = compare_shape_kappa(lhs.component_kappa2,
                                 rhs.component_kappa2)) != 0 ||
      (cmp = compare_shape_moment(lhs.component_asphericity,
                                  rhs.component_asphericity)) != 0 ||
      (cmp = compare_shape_moment(lhs.component_acylindricity,
                                  rhs.component_acylindricity)) != 0 ||
      (cmp = compare_shape_kappa(lhs.shape_kappa2, rhs.shape_kappa2)) != 0 ||
      (cmp = compare_shape_moment(lhs.shape_asphericity,
                                  rhs.shape_asphericity)) != 0 ||
      (cmp = compare_shape_moment(lhs.shape_acylindricity,
                                  rhs.shape_acylindricity)) != 0 ||
      (cmp = compare_pbctopo_double_metric(lhs.shift_norm2,
                                           rhs.shift_norm2)) != 0)
    return cmp < 0;
  cmp = compare_pbctopo_double_metric(lhs.min_pair_d2, rhs.min_pair_d2,
                                      1.0e-9);
  return cmp > 0;
}

PbctopoLexicographicGap candidate_lexicographic_gap(
    const PbctopoCutCandidate &best, const PbctopoCutCandidate &second) {
  if (!best.valid || !second.valid)
    return {"na", std::numeric_limits<double>::quiet_NaN()};
  if (best.ultra_clash != second.ultra_clash)
    return {"ultra_clash", 1.0};
  if (best.broken_edges != second.broken_edges)
    return {"broken_edges", static_cast<double>(second.broken_edges) -
                                static_cast<double>(best.broken_edges)};

  auto lower = [](const char *name, double lhs, double rhs, int cmp) {
    return cmp == 0 ? PbctopoLexicographicGap{}
                    : PbctopoLexicographicGap{name, rhs - lhs};
  };
  auto metric = [&](const char *name, double lhs, double rhs,
                    double eps = 1.0e-6) {
    return lower(name, lhs, rhs,
                 compare_pbctopo_double_metric(lhs, rhs, eps));
  };
  auto quantized = [&](const char *name, double lhs, double rhs,
                       PbctopoMetricResolution resolution) {
    return lower(name, lhs, rhs,
                 compare_pbctopo_metric(lhs, rhs, resolution));
  };
  auto distinct = [](const PbctopoLexicographicGap &candidate,
                     PbctopoLexicographicGap &out) {
    if (candidate.metric == "equivalent")
      return false;
    out = candidate;
    return true;
  };

  PbctopoLexicographicGap result;
  const bool best_has_cont = std::isfinite(best.continuity_max_d2) &&
                             std::isfinite(best.continuity_path);
  const bool second_has_cont = std::isfinite(second.continuity_max_d2) &&
                               std::isfinite(second.continuity_path);
  if (best_has_cont != second_has_cont)
    return {"continuity_available", best_has_cont ? 1.0 : -1.0};
  if (best_has_cont) {
    if (distinct(metric("continuity_path", best.continuity_path,
                        second.continuity_path), result))
      return result;
  }
  if (best.anchor_graph.residue_order_violations !=
      second.anchor_graph.residue_order_violations) {
    return {"anchor_residue_order_violations",
            static_cast<double>(second.anchor_graph.residue_order_violations) -
                static_cast<double>(best.anchor_graph.residue_order_violations)};
  }
  if (distinct(metric("anchor_endpoint_image_gap",
                      best.anchor_graph.endpoint_image_gap,
                      second.anchor_graph.endpoint_image_gap), result))
    return result;
  if (distinct(metric("anchor_order_max_d2", best.anchor_graph.order_max_d2,
                      second.anchor_graph.order_max_d2), result) ||
      distinct(metric("anchor_order_path_d2",
                      best.anchor_graph.order_path_d2,
                      second.anchor_graph.order_path_d2), result) ||
      distinct(metric("anchor_max_d2", best.anchor_max_d2,
                      second.anchor_max_d2), result) ||
      distinct(metric("anchor_mst2", best.anchor_mst2, second.anchor_mst2),
               result) ||
      distinct(quantized("principal_rg2_max", best.principal_rg2_max,
                         second.principal_rg2_max, kPrincipalRgResolution),
               result) ||
      distinct(quantized("principal_rg2_mid", best.principal_rg2_mid,
                         second.principal_rg2_mid, kPrincipalRgResolution),
               result) ||
      distinct(quantized("principal_rg2_min", best.principal_rg2_min,
                         second.principal_rg2_min, kPrincipalRgResolution),
               result) ||
      distinct(quantized("shape_kappa2", best.shape_kappa2,
                         second.shape_kappa2, kShapeKappaResolution), result) ||
      distinct(quantized("shape_asphericity", best.shape_asphericity,
                         second.shape_asphericity, kShapeMomentResolution),
               result) ||
      distinct(quantized("shape_acylindricity", best.shape_acylindricity,
                         second.shape_acylindricity,
                         kShapeMomentResolution), result) ||
      distinct(metric("edge_stretch", best.edge_stretch,
                      second.edge_stretch), result))
    return result;
  const int pair_cmp = compare_pbctopo_double_metric(
      best.min_pair_d2, second.min_pair_d2, 1.0e-9);
  if (pair_cmp != 0)
    return {"min_pair_d2", best.min_pair_d2 - second.min_pair_d2};
  if (distinct(metric("shift_norm2", best.shift_norm2, second.shift_norm2),
               result) ||
      distinct(metric("cut_x", best.cut_x, second.cut_x, 1.0e-9), result) ||
      distinct(metric("cut_y", best.cut_y, second.cut_y, 1.0e-9), result) ||
      distinct(metric("cut_z", best.cut_z, second.cut_z, 1.0e-9), result))
    return result;
  return result;
}

double candidate_temporal_emission_cost(
    const PbctopoCutCandidate &frame_best,
    const PbctopoCutCandidate &candidate) {
  if (!frame_best.valid || !candidate.valid)
    return std::numeric_limits<double>::infinity();
  if (!better_cut_candidate(frame_best, candidate) &&
      !better_cut_candidate(candidate, frame_best)) {
    return 0.0;
  }

  const PbctopoLexicographicGap gap =
      candidate_lexicographic_gap(frame_best, candidate);
  struct Tier {
    const char *name;
    double weight;
  };
  constexpr Tier tiers[] = {
      {"ultra_clash", 1.0},
      {"broken_edges", 0.95},
      {"continuity_available", 0.90},
      {"continuity_path", 0.86},
      {"anchor_residue_order_violations", 0.82},
      {"anchor_endpoint_image_gap", 0.80},
      {"anchor_order_max_d2", 0.74},
      {"anchor_order_path_d2", 0.72},
      {"anchor_max_d2", 0.70},
      {"anchor_mst2", 0.68},
      {"principal_rg2_max", 0.50},
      {"principal_rg2_mid", 0.48},
      {"principal_rg2_min", 0.46},
      {"shape_kappa2", 0.42},
      {"shape_asphericity", 0.40},
      {"shape_acylindricity", 0.38},
      {"edge_stretch", 0.34},
      {"min_pair_d2", 0.30},
      {"shift_norm2", 0.24},
      {"cut_x", 0.12},
      {"cut_y", 0.10},
      {"cut_z", 0.08},
  };
  double tier_weight = 0.05;
  for (const auto &tier : tiers) {
    if (gap.metric == tier.name) {
      tier_weight = tier.weight;
      break;
    }
  }
  const double magnitude = std::isfinite(gap.gap) ? std::fabs(gap.gap) : 1.0;
  const double bounded_gap = magnitude / (1.0 + magnitude);
  return tier_weight + 0.05 * bounded_gap;
}

bool preclash_cut_candidate_cannot_beat_incumbent(
    const PbctopoCutCandidate &candidate,
    const PbctopoCutCandidate *incumbent) {
  if (!candidate.valid || incumbent == nullptr || !incumbent->valid ||
      incumbent->ultra_clash)
    return false;
  return cut_candidate_cannot_beat_incumbent(candidate, incumbent);
}

bool cut_candidate_cannot_beat_incumbent(
    const PbctopoCutCandidate &partial,
    const PbctopoCutCandidate *incumbent) {
  if (incumbent == nullptr || !incumbent->valid || incumbent->ultra_clash ||
      partial.ultra_clash)
    return false;
  if (partial.broken_edges > incumbent->broken_edges)
    return true;
  if (std::isfinite(partial.continuity_path) &&
      std::isfinite(incumbent->continuity_path) &&
      partial.continuity_path > incumbent->continuity_path + 1.0e-6)
    return true;
  if (compare_anchor_graph_score(partial.anchor_graph,
                                 incumbent->anchor_graph) > 0 &&
      std::isfinite(partial.anchor_max_d2) &&
      std::isfinite(incumbent->anchor_max_d2) &&
      partial.anchor_max_d2 > incumbent->anchor_max_d2 + 1.0e-6 &&
      partial.anchor_mst2 > incumbent->anchor_mst2 + 1.0e-6)
    return true;
  return partial.broken_edges == incumbent->broken_edges &&
         partial.edge_stretch > incumbent->edge_stretch + 1.0e-6;
}

int compare_candidate_prefix_before_edge_stretch(
    const PbctopoCutCandidate &lhs, const PbctopoCutCandidate &rhs,
    PbctopoRefineScorePhase phase) {
  (void)phase;
  if (lhs.broken_edges != rhs.broken_edges)
    return lhs.broken_edges < rhs.broken_edges ? -1 : 1;
  int cmp = compare_candidate_continuity_score(lhs, rhs);
  if (cmp != 0)
    return cmp;
  return compare_candidate_anchor_prefix(lhs, rhs);
}

bool component_candidate_cannot_beat_incumbent_prefix(
    const PbctopoComponentCandidate &candidate,
    const PbctopoComponentCandidate *incumbent,
    PbctopoComponentCandidatePrefixStage stage) {
  if (incumbent == nullptr || !incumbent->valid || incumbent->ultra_clash ||
      candidate.ultra_clash)
    return false;

  const int continuity_cmp =
      compare_candidate_continuity_score(candidate, *incumbent);
  if (continuity_cmp != 0)
    return continuity_cmp > 0;
  if (stage == PbctopoComponentCandidatePrefixStage::Continuity)
    return false;

  int cmp = compare_candidate_anchor_prefix(candidate, *incumbent);
  if (cmp != 0)
    return cmp > 0;
  return false;
}

} // namespace titan_pbctopo
