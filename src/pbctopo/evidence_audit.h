#ifndef TITAN_PBCTOPO_EVIDENCE_AUDIT_H
#define TITAN_PBCTOPO_EVIDENCE_AUDIT_H

#include "evidence_support_policy.h"
#include "types.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace titan_pbctopo {

inline constexpr std::string_view PBCTOPO_EOBS_PAIR_HASH_ALGORITHM =
    "sha256-128";
inline constexpr std::string_view PBCTOPO_EOBS_PAIR_HASH_DOMAIN =
    "titan-pbctopo-eobs-pairs-v1";
inline constexpr std::string_view PBCTOPO_EOBS_HYPOTHESIS_HASH_DOMAIN =
    "titan-pbctopo-eobs-shift-hypotheses-v1";
inline constexpr std::string_view PBCTOPO_EOBS_CUTOFF_POLICY =
    "wrapped-mic-heavy-closest8-per-owner-fragment-pair-v1";
inline constexpr std::uint32_t PBCTOPO_EOBS_AUDIT_POLICY_VERSION = 1;
inline constexpr std::size_t PBCTOPO_EOBS_MAX_PAIRS_PER_UNIT_PAIR = 8;
inline constexpr double PBCTOPO_EOBS_SELECTION_TIE_TOLERANCE_D2_A2 = 1.0e-9;
inline constexpr double PBCTOPO_EOBS_LOSS_TOLERANCE_D2_A2 = 1.0e-6;

struct PbctopoCanonicalAtomPair {
  std::uint64_t a = 0;
  std::uint64_t b = 0;

  auto operator<=>(const PbctopoCanonicalAtomPair &) const = default;
};

struct PbctopoSelectedEvidenceAudit {
  PbctopoEvidenceState state = PbctopoEvidenceState::NotEvaluated;
  bool evaluated = false;
  bool consistent = false;
  bool evidence_graph_connected = false;
  std::size_t evidence_component_count = 0;
  std::size_t evidence_edge_count = 0;
  std::size_t explicit_contact_evidence_edges = 0;
  std::size_t scored_contact_evidence_edges = 0;
  std::size_t evidence_connected_component_count = 0;
  std::size_t unconstrained_relative_components = 0;
  std::size_t relative_lattice_degrees_of_freedom = 0;
  std::size_t contacts_checked = 0;
  std::size_t contacts_lost = 0;
  PbctopoHash128 observed_pair_hash;
  PbctopoHash128 lost_pair_hash;
  PbctopoHash128 all_hypotheses_hash;
  PbctopoHash128 selected_hypothesis_hash;
  PbctopoHash128 selected_compatible_pair_hash;
  std::size_t hypothesis_count = 0;
  std::size_t component_relation_count = 0;
  std::size_t ambiguous_component_relation_count = 0;
  std::size_t compatible_hypothesis_count = 0;
  std::size_t supported_relation_count = 0;
  std::size_t selected_compatible_contact_count = 0;
  std::size_t alternative_hypothesis_contact_count = 0;
  std::size_t selected_compatible_contact_loss_count = 0;
  std::size_t no_supported_contact_relation_count = 0;
  bool selected_relation_has_support = false;
  std::vector<PbctopoSelectedEvidenceRelationAudit> relation_audits;
};

[[nodiscard]] PbctopoContactHypothesisSet
build_pbctopo_contact_hypotheses(
    std::span<const PbctopoAtom> atoms,
    std::span<const PbctopoAnchorContactEdge> observed_edges,
    std::span<const std::size_t> atom_component_index,
    std::span<const Int3> spatial_lift_atom_images,
    const PbctopoImageAssignment &identity_assignment);

[[nodiscard]] std::expected<PbctopoHash128, std::string>
hash_pbctopo_contact_hypotheses(
    std::span<const PbctopoContactShiftHypothesis> hypotheses);

[[nodiscard]] std::expected<PbctopoHash128, std::string>
hash_pbctopo_canonical_atom_pairs(
    std::span<const PbctopoCanonicalAtomPair> pairs);

template <class Point>
[[nodiscard]] std::expected<PbctopoSelectedEvidenceAudit, std::string>
audit_pbctopo_selected_evidence(
    std::span<const PbctopoAtom> atoms, std::span<const Point> placed_points,
    std::span<const PbctopoAnchorContactEdge> observed_edges,
    bool hard_feasible, bool evidence_graph_connected) {
  PbctopoSelectedEvidenceAudit result;
  result.evidence_graph_connected = evidence_graph_connected;
  if (!hard_feasible)
    return result;
  if (atoms.size() != placed_points.size())
    return std::unexpected("selected evidence audit atom/point size mismatch");

  std::vector<PbctopoCanonicalAtomPair> observed_pairs;
  std::vector<PbctopoCanonicalAtomPair> lost_pairs;
  observed_pairs.reserve(observed_edges.size());
  lost_pairs.reserve(observed_edges.size());
  for (const auto &edge : observed_edges) {
    if (edge.a >= atoms.size() || edge.b >= atoms.size() || edge.a == edge.b ||
        !std::isfinite(edge.cutoff_d2) || edge.cutoff_d2 < 0.0) {
      return std::unexpected("selected evidence audit contains an invalid edge");
    }
    const std::uint64_t global_a =
        static_cast<std::uint64_t>(atoms[edge.a].global_atom_id);
    const std::uint64_t global_b =
        static_cast<std::uint64_t>(atoms[edge.b].global_atom_id);
    const PbctopoCanonicalAtomPair pair{std::min(global_a, global_b),
                                        std::max(global_a, global_b)};
    observed_pairs.push_back(pair);
    const double dx = placed_points[edge.a].x - placed_points[edge.b].x;
    const double dy = placed_points[edge.a].y - placed_points[edge.b].y;
    const double dz = placed_points[edge.a].z - placed_points[edge.b].z;
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (!std::isfinite(d2))
      return std::unexpected("selected evidence audit found non-finite distance");
    if (d2 > edge.cutoff_d2 + PBCTOPO_EOBS_LOSS_TOLERANCE_D2_A2)
      lost_pairs.push_back(pair);
  }
  std::sort(observed_pairs.begin(), observed_pairs.end());
  std::sort(lost_pairs.begin(), lost_pairs.end());
  if (std::adjacent_find(observed_pairs.begin(), observed_pairs.end()) !=
      observed_pairs.end()) {
    return std::unexpected("selected evidence audit contains duplicate pairs");
  }
  const auto observed_hash =
      hash_pbctopo_canonical_atom_pairs(observed_pairs);
  if (!observed_hash)
    return std::unexpected(observed_hash.error());
  const auto lost_hash = hash_pbctopo_canonical_atom_pairs(lost_pairs);
  if (!lost_hash)
    return std::unexpected(lost_hash.error());
  result.observed_pair_hash = *observed_hash;
  result.lost_pair_hash = *lost_hash;
  result.evaluated = true;
  result.contacts_checked = observed_pairs.size();
  result.contacts_lost = lost_pairs.size();
  result.consistent = lost_pairs.empty();
  result.state = !result.consistent
                     ? PbctopoEvidenceState::Contradicted
                     : evidence_graph_connected
                           ? PbctopoEvidenceState::SupportedConnected
                           : PbctopoEvidenceState::WeakDisconnected;
  return result;
}

template <class Point>
[[nodiscard]] std::expected<PbctopoSelectedEvidenceAudit, std::string>
audit_pbctopo_selected_hypotheses(
    std::span<const PbctopoAtom> atoms, std::span<const Point> placed_points,
    const PbctopoContactHypothesisSet &hypothesis_set,
    const PbctopoImageAssignment &selected_assignment, bool hard_feasible,
    std::span<const std::pair<std::size_t, std::size_t>>
        explicit_component_edges = {}) {
  PbctopoSelectedEvidenceAudit result;
  if (!hard_feasible)
    return result;
  if (!hypothesis_set.complete)
    return result;
  if (!selected_assignment.valid || atoms.size() != placed_points.size() ||
      selected_assignment.component_offsets.size() !=
          selected_assignment.component_ids.size()) {
    return std::unexpected("selected hypothesis audit input mismatch");
  }

  result.all_hypotheses_hash = hypothesis_set.all_hypotheses_hash;
  result.observed_pair_hash = hypothesis_set.raw_observed_pair_hash;
  result.contacts_checked = hypothesis_set.raw_observed_pair_count;
  result.hypothesis_count = hypothesis_set.hypotheses.size();
  result.component_relation_count = hypothesis_set.component_relation_count;
  result.ambiguous_component_relation_count =
      hypothesis_set.ambiguous_component_relation_count;
  result.evidence_component_count = selected_assignment.component_ids.size();

  auto sorted_component_ids = selected_assignment.component_ids;
  std::sort(sorted_component_ids.begin(), sorted_component_ids.end());
  if (std::adjacent_find(sorted_component_ids.begin(),
                         sorted_component_ids.end()) !=
      sorted_component_ids.end()) {
    return std::unexpected(
        "selected hypothesis audit contains duplicate component IDs");
  }

  auto offset_for_id = [&](std::uint64_t id) -> const Int3 * {
    for (std::size_t idx = 0; idx < selected_assignment.component_ids.size(); ++idx) {
      if (selected_assignment.component_ids[idx] == id)
        return &selected_assignment.component_offsets[idx];
    }
    return nullptr;
  };

  std::vector<PbctopoContactShiftHypothesis> compatible;
  std::vector<PbctopoCanonicalAtomPair> observed_pairs;
  std::vector<PbctopoCanonicalAtomPair> lost_pairs;
  std::vector<std::pair<std::size_t, std::size_t>> connected_edges;
  std::vector<std::pair<std::size_t, std::size_t>> explicit_edges;
  std::vector<std::pair<std::size_t, std::size_t>> scored_edges;
  explicit_edges.reserve(explicit_component_edges.size());
  connected_edges.reserve(explicit_component_edges.size() +
                           hypothesis_set.component_relation_count);
  result.relation_audits.reserve(explicit_component_edges.size() +
                                 hypothesis_set.component_relation_count);
  const auto canonicalize_component_edges = [](auto &edges) {
    for (auto &[a, b] : edges) {
      if (b < a)
        std::swap(a, b);
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  };
  for (auto [a, b] : explicit_component_edges) {
    if (a >= result.evidence_component_count ||
        b >= result.evidence_component_count || a == b) {
      return std::unexpected(
          "selected hypothesis audit contains an invalid explicit component "
          "edge");
    }
    if (b < a)
      std::swap(a, b);
    explicit_edges.emplace_back(a, b);
  }
  canonicalize_component_edges(explicit_edges);

  bool unsupported_relation = false;
  std::size_t relation_begin = 0;
  while (relation_begin < hypothesis_set.hypotheses.size()) {
    std::size_t relation_end = relation_begin + 1;
    const auto &first = hypothesis_set.hypotheses[relation_begin];
    while (relation_end < hypothesis_set.hypotheses.size() &&
           hypothesis_set.hypotheses[relation_end].component_id_a ==
               first.component_id_a &&
           hypothesis_set.hypotheses[relation_end].component_id_b ==
               first.component_id_b) {
      ++relation_end;
    }
    if (first.component_id_a >= first.component_id_b ||
        first.component_a >= result.evidence_component_count ||
        first.component_b >= result.evidence_component_count ||
        first.component_a == first.component_b ||
        selected_assignment.component_ids[first.component_a] !=
            first.component_id_a ||
        selected_assignment.component_ids[first.component_b] !=
            first.component_id_b) {
      return std::unexpected(
          "selected hypothesis audit contains an invalid component relation");
    }
    const Int3 *offset_a = offset_for_id(first.component_id_a);
    const Int3 *offset_b = offset_for_id(first.component_id_b);
    if (offset_a == nullptr || offset_b == nullptr)
      return std::unexpected("selected assignment misses an evidence component");
    Int3 selected_relative{};
    if (!checked_int3_subtract(*offset_b, *offset_a, selected_relative))
      return std::unexpected("selected evidence relative shift overflow");

    const std::pair<std::size_t, std::size_t> component_pair{
        std::min(first.component_a, first.component_b),
        std::max(first.component_a, first.component_b)};
    std::vector<Int3> observed_relative_shifts;
    observed_relative_shifts.reserve(relation_end - relation_begin);
    for (std::size_t idx = relation_begin; idx < relation_end; ++idx) {
      const auto &hypothesis = hypothesis_set.hypotheses[idx];
      if (hypothesis.component_a != first.component_a ||
          hypothesis.component_b != first.component_b ||
          hypothesis.component_id_a != first.component_id_a ||
          hypothesis.component_id_b != first.component_id_b) {
        return std::unexpected(
            "selected hypothesis relation has inconsistent component binding");
      }
      observed_relative_shifts.push_back(hypothesis.relative_shift);
    }
    const bool explicit_relation =
        std::binary_search(explicit_edges.begin(), explicit_edges.end(),
                           component_pair);
    PbctopoSelectedEvidenceRelationAudit relation;
    relation.component_a = component_pair.first;
    relation.component_b = component_pair.second;
    relation.component_id_a = first.component_id_a;
    relation.component_id_b = first.component_id_b;
    relation.evaluated = true;
    relation.observed_hypothesis_relation = true;
    relation.support = evaluate_pbctopo_selected_compatibility_support(
        explicit_relation, observed_relative_shifts, selected_relative);

    for (std::size_t idx = relation_begin; idx < relation_end; ++idx) {
      const auto &hypothesis = hypothesis_set.hypotheses[idx];
      if (!same_delta(hypothesis.relative_shift, selected_relative))
      {
        result.alternative_hypothesis_contact_count += hypothesis.edges.size();
        relation.alternative_hypothesis_contact_count +=
            hypothesis.edges.size();
        continue;
      }
      compatible.push_back(hypothesis);
      relation.compatible_contact_count += hypothesis.edges.size();
      for (const auto &edge : hypothesis.edges) {
        if (edge.a >= atoms.size() || edge.b >= atoms.size() || edge.a == edge.b)
          return std::unexpected("selected hypothesis contains an invalid edge");
        const auto global_a = static_cast<std::uint64_t>(atoms[edge.a].global_atom_id);
        const auto global_b = static_cast<std::uint64_t>(atoms[edge.b].global_atom_id);
        const PbctopoCanonicalAtomPair pair{std::min(global_a, global_b),
                                            std::max(global_a, global_b)};
        observed_pairs.push_back(pair);
        const double dx = placed_points[edge.a].x - placed_points[edge.b].x;
        const double dy = placed_points[edge.a].y - placed_points[edge.b].y;
        const double dz = placed_points[edge.a].z - placed_points[edge.b].z;
        const double d2 = dx * dx + dy * dy + dz * dz;
        if (!std::isfinite(d2))
          return std::unexpected("selected hypothesis audit found non-finite distance");
        if (d2 > edge.cutoff_d2 + PBCTOPO_EOBS_LOSS_TOLERANCE_D2_A2) {
          lost_pairs.push_back(pair);
          ++relation.compatible_contact_loss_count;
        }
      }
    }
    const bool matched = relation.support.compatibility_match;
    if (matched)
      scored_edges.push_back(component_pair);
    if (relation.support.active_at_threshold)
      connected_edges.push_back(component_pair);
    unsupported_relation = unsupported_relation || !matched;
    if (matched) ++result.supported_relation_count;
    if (!matched) ++result.no_supported_contact_relation_count;
    result.relation_audits.push_back(std::move(relation));
    relation_begin = relation_end;
  }

  for (const auto &component_pair : explicit_edges) {
    const auto existing = std::find_if(
        result.relation_audits.begin(), result.relation_audits.end(),
        [&](const PbctopoSelectedEvidenceRelationAudit &relation) {
          return relation.component_a == component_pair.first &&
                 relation.component_b == component_pair.second;
        });
    if (existing != result.relation_audits.end())
      continue;

    PbctopoSelectedEvidenceRelationAudit relation;
    relation.component_a = component_pair.first;
    relation.component_b = component_pair.second;
    relation.component_id_a =
        selected_assignment.component_ids[component_pair.first];
    relation.component_id_b =
        selected_assignment.component_ids[component_pair.second];
    if (relation.component_id_b < relation.component_id_a)
      std::swap(relation.component_id_a, relation.component_id_b);
    relation.evaluated = true;
    relation.support = evaluate_pbctopo_selected_compatibility_support(
        true, std::span<const Int3>{}, Int3{});
    connected_edges.push_back(component_pair);
    result.relation_audits.push_back(std::move(relation));
  }

  std::sort(result.relation_audits.begin(), result.relation_audits.end(),
            [](const auto &lhs, const auto &rhs) {
              return std::pair{lhs.component_a, lhs.component_b} <
                     std::pair{rhs.component_a, rhs.component_b};
            });
  if (std::adjacent_find(
          result.relation_audits.begin(), result.relation_audits.end(),
          [](const auto &lhs, const auto &rhs) {
            return lhs.component_a == rhs.component_a &&
                   lhs.component_b == rhs.component_b;
          }) != result.relation_audits.end()) {
    return std::unexpected(
        "selected hypothesis audit contains duplicate relation records");
  }

  std::sort(observed_pairs.begin(), observed_pairs.end());
  std::sort(lost_pairs.begin(), lost_pairs.end());
  if (std::adjacent_find(observed_pairs.begin(), observed_pairs.end()) !=
          observed_pairs.end() ||
      std::adjacent_find(lost_pairs.begin(), lost_pairs.end()) !=
          lost_pairs.end()) {
    return std::unexpected("selected hypothesis audit contains duplicate pairs");
  }
  auto observed_hash = hash_pbctopo_canonical_atom_pairs(observed_pairs);
  auto lost_hash = hash_pbctopo_canonical_atom_pairs(lost_pairs);
  auto compatible_hashable = compatible;
  for (auto &hypothesis : compatible_hashable) {
    for (auto &edge : hypothesis.edges) {
      edge.a = atoms[edge.a].global_atom_id;
      edge.b = atoms[edge.b].global_atom_id;
      if (edge.b < edge.a) std::swap(edge.a, edge.b);
    }
    std::sort(hypothesis.edges.begin(), hypothesis.edges.end(),
              [](const auto &a, const auto &b) {
                return std::pair{a.a, a.b} < std::pair{b.a, b.b};
              });
  }
  std::expected<PbctopoHash128, std::string> selected_hash =
      compatible_hashable.empty()
          ? std::expected<PbctopoHash128, std::string>(PbctopoHash128{})
          : hash_pbctopo_contact_hypotheses(compatible_hashable);
  if (!observed_hash) return std::unexpected(observed_hash.error());
  if (!lost_hash) return std::unexpected(lost_hash.error());
  if (!selected_hash) return std::unexpected(selected_hash.error());
  result.selected_compatible_pair_hash = compatible.empty()
                                             ? PbctopoHash128{}
                                             : *observed_hash;
  result.lost_pair_hash = *lost_hash;
  result.selected_hypothesis_hash = *selected_hash;
  result.selected_compatible_contact_count = observed_pairs.size();
  result.contacts_lost = lost_pairs.size();
  result.selected_compatible_contact_loss_count = lost_pairs.size();
  result.compatible_hypothesis_count = compatible.size();
  result.selected_relation_has_support = !compatible.empty();

  canonicalize_component_edges(scored_edges);
  canonicalize_component_edges(connected_edges);
  const std::size_t active_relation_count =
      static_cast<std::size_t>(std::count_if(
          result.relation_audits.begin(), result.relation_audits.end(),
          [](const auto &relation) {
            return relation.support.active_at_threshold;
          }));
  const std::size_t explicit_relation_count =
      static_cast<std::size_t>(std::count_if(
          result.relation_audits.begin(), result.relation_audits.end(),
          [](const auto &relation) {
            return relation.support.explicit_relation;
          }));
  const std::size_t compatible_relation_count =
      static_cast<std::size_t>(std::count_if(
          result.relation_audits.begin(), result.relation_audits.end(),
          [](const auto &relation) {
            return relation.support.compatibility_match;
          }));
  const std::size_t observed_relation_count =
      static_cast<std::size_t>(std::count_if(
          result.relation_audits.begin(), result.relation_audits.end(),
          [](const auto &relation) {
            return relation.observed_hypothesis_relation;
          }));
  const bool support_thresholds_replay = std::all_of(
      result.relation_audits.begin(), result.relation_audits.end(),
      [](const auto &relation) {
        return relation.evaluated &&
               std::isfinite(relation.support.support_primary) &&
               std::isfinite(relation.support.decision_threshold) &&
               relation.support.active_at_threshold ==
                   (relation.support.support_primary >=
                    relation.support.decision_threshold);
      });
  if (!support_thresholds_replay ||
      active_relation_count != connected_edges.size() ||
      explicit_relation_count != explicit_edges.size() ||
      compatible_relation_count != scored_edges.size() ||
      observed_relation_count != hypothesis_set.component_relation_count) {
    return std::unexpected(
        "selected hypothesis support records do not replay graph aggregates");
  }
  result.explicit_contact_evidence_edges = explicit_edges.size();
  result.scored_contact_evidence_edges = scored_edges.size();
  result.evidence_edge_count = connected_edges.size();

  std::vector<unsigned char> seen(selected_assignment.component_ids.size(), 0);
  std::vector<std::vector<std::size_t>> adjacency(seen.size());
  for (const auto &[a, b] : connected_edges) {
    if (a >= seen.size() || b >= seen.size()) {
      return std::unexpected(
          "selected hypothesis audit contains an invalid component edge");
    }
    adjacency[a].push_back(b);
    adjacency[b].push_back(a);
  }
  for (std::size_t start = 0; start < seen.size(); ++start) {
    if (seen[start] != 0)
      continue;
    ++result.evidence_connected_component_count;
    std::vector<std::size_t> stack{start};
    seen[start] = 1;
    while (!stack.empty()) {
      const auto node = stack.back();
      stack.pop_back();
      for (const auto next : adjacency[node]) {
        if (seen[next] == 0) {
          seen[next] = 1;
          stack.push_back(next);
        }
      }
    }
  }
  result.evidence_graph_connected =
      result.evidence_component_count > 0 &&
      result.evidence_connected_component_count == 1;
  result.unconstrained_relative_components =
      result.evidence_connected_component_count > 0
          ? result.evidence_connected_component_count - 1
          : 0;
  result.relative_lattice_degrees_of_freedom =
      3 * result.unconstrained_relative_components;
  result.evaluated = true;
  result.consistent = lost_pairs.empty();
  if (!result.consistent) {
    result.state = PbctopoEvidenceState::Contradicted;
  } else if (unsupported_relation ||
             hypothesis_set.ambiguous_component_relation_count != 0) {
    result.state = PbctopoEvidenceState::WeakAmbiguous;
  } else if (result.evidence_graph_connected) {
    result.state = PbctopoEvidenceState::SupportedConnected;
  } else {
    result.state = PbctopoEvidenceState::WeakDisconnected;
  }
  return result;
}

} // namespace titan_pbctopo

#endif
