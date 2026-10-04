#include "identifiability.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace titan_pbctopo {
namespace {

std::uint64_t component_pair_key(std::size_t a, std::size_t b) {
  const std::size_t lo = std::min(a, b);
  const std::size_t hi = std::max(a, b);
  return (static_cast<std::uint64_t>(lo) << 32) |
         static_cast<std::uint64_t>(hi);
}

std::size_t find_root(std::vector<std::size_t> &parent, std::size_t value) {
  while (parent[value] != value) {
    parent[value] = parent[parent[value]];
    value = parent[value];
  }
  return value;
}

void unite(std::vector<std::size_t> &parent, std::vector<unsigned char> &rank,
           std::size_t a, std::size_t b) {
  a = find_root(parent, a);
  b = find_root(parent, b);
  if (a == b)
    return;
  if (rank[a] < rank[b])
    std::swap(a, b);
  parent[b] = a;
  if (rank[a] == rank[b])
    ++rank[a];
}

} // namespace

void apply_pbctopo_search_identifiability(
    PbctopoIdentifiability &identifiability,
    bool certificate_search_exhaustive, bool equivalent_optima_complete,
    std::size_t equivalent_best_assignments_within_domain,
    bool feasible_assignment_set_enumerated,
    std::size_t feasible_assignments_within_domain) {
  identifiability.objective_uniqueness_known =
      certificate_search_exhaustive && equivalent_optima_complete;
  identifiability.equivalent_best_assignments_within_domain =
      identifiability.objective_uniqueness_known
          ? equivalent_best_assignments_within_domain
          : 0;
  identifiability.objective_unique_within_search_domain =
      identifiability.identified &&
      identifiability.objective_uniqueness_known &&
      identifiability.equivalent_best_assignments_within_domain == 1;

  identifiability.feasible_assignment_set_enumerated =
      certificate_search_exhaustive &&
      feasible_assignment_set_enumerated;
  identifiability.feasible_assignments_within_domain =
      identifiability.feasible_assignment_set_enumerated
          ? feasible_assignments_within_domain
          : 0;
  identifiability.evidence_uniqueness_known =
      identifiability.feasible_assignment_set_enumerated;
  identifiability.evidence_unique_within_search_domain =
      identifiability.identified &&
      identifiability.evidence_uniqueness_known &&
      identifiability.feasible_assignments_within_domain == 1;

  identifiability.uniqueness_search_exhaustive =
      identifiability.objective_uniqueness_known;
  identifiability.unique_within_search_domain =
      identifiability.objective_unique_within_search_domain;
}

void apply_pbctopo_final_selected_evidence(
    PbctopoIdentifiability &identifiability, bool evidence_audit_evaluated,
    bool evidence_graph_connected, std::size_t evidence_component_count,
    std::size_t evidence_edge_count,
    std::size_t explicit_contact_evidence_edges,
    std::size_t scored_contact_evidence_edges,
    std::size_t evidence_connected_component_count) {
  if (!evidence_audit_evaluated)
    return;

  identifiability.evidence_component_count = evidence_component_count;
  identifiability.evidence_edge_count = evidence_edge_count;
  identifiability.explicit_contact_evidence_edges =
      explicit_contact_evidence_edges;
  identifiability.scored_contact_evidence_edges = scored_contact_evidence_edges;
  identifiability.evidence_connected_component_count =
      evidence_connected_component_count;
  identifiability.unconstrained_relative_components =
      evidence_connected_component_count > 0
          ? evidence_connected_component_count - 1
          : 0;
  identifiability.relative_lattice_degrees_of_freedom =
      3 * identifiability.unconstrained_relative_components;
  identifiability.evidence_graph_connected = evidence_graph_connected;
  identifiability.identified = evidence_graph_connected;
  identifiability.objective_unique_within_search_domain =
      identifiability.identified &&
      identifiability.objective_uniqueness_known &&
      identifiability.equivalent_best_assignments_within_domain == 1;
  identifiability.evidence_unique_within_search_domain =
      identifiability.identified &&
      identifiability.evidence_uniqueness_known &&
      identifiability.feasible_assignment_set_enumerated &&
      identifiability.feasible_assignments_within_domain == 1;
  identifiability.uniqueness_search_exhaustive =
      identifiability.objective_uniqueness_known;
  identifiability.unique_within_search_domain =
      identifiability.objective_unique_within_search_domain;
}

PbctopoIdentifiability evaluate_pbctopo_identifiability(
    std::size_t component_count,
    std::span<const PbctopoPairShiftCandidateSet> pair_candidates,
    std::span<const PbctopoRequiredContact> required_contacts,
    std::span<const std::size_t> atom_component_index,
    bool uniqueness_search_exhaustive,
    std::size_t equivalent_best_assignments_within_domain,
    bool feasible_assignment_set_enumerated,
    std::size_t feasible_assignments_within_domain) {
  PbctopoIdentifiability result;
  result.evidence_component_count = component_count;
  if (component_count == 0)
    return result;

  std::vector<std::size_t> parent(component_count);
  std::iota(parent.begin(), parent.end(), 0);
  std::vector<unsigned char> rank(component_count, 0);
  std::unordered_set<std::uint64_t> evidence_edges;
  std::unordered_set<std::uint64_t> explicit_edges;
  std::unordered_set<std::uint64_t> scored_edges;

  for (const auto &contact : required_contacts) {
    if (contact.a >= atom_component_index.size() ||
        contact.b >= atom_component_index.size()) {
      continue;
    }
    const std::size_t a = atom_component_index[contact.a];
    const std::size_t b = atom_component_index[contact.b];
    if (a >= component_count || b >= component_count || a == b)
      continue;
    const std::uint64_t key = component_pair_key(a, b);
    evidence_edges.insert(key);
    explicit_edges.insert(key);
    unite(parent, rank, a, b);
  }

  for (const auto &pair_set : pair_candidates) {
    const auto support =
        evaluate_pbctopo_scored_pair_evidence_support(pair_set);
    if (pair_set.a >= component_count || pair_set.b >= component_count ||
        pair_set.a == pair_set.b || !support.active_at_threshold) {
      continue;
    }
    const std::uint64_t key = component_pair_key(pair_set.a, pair_set.b);
    evidence_edges.insert(key);
    scored_edges.insert(key);
    unite(parent, rank, pair_set.a, pair_set.b);
  }

  std::unordered_set<std::size_t> roots;
  roots.reserve(component_count);
  for (std::size_t component = 0; component < component_count; ++component)
    roots.insert(find_root(parent, component));

  result.evidence_edge_count = evidence_edges.size();
  result.explicit_contact_evidence_edges = explicit_edges.size();
  result.scored_contact_evidence_edges = scored_edges.size();
  result.evidence_connected_component_count = roots.size();
  result.evidence_graph_connected = roots.size() == 1;
  result.unconstrained_relative_components = roots.size() - 1;
  result.relative_lattice_degrees_of_freedom =
      3 * result.unconstrained_relative_components;
  result.identified = result.evidence_graph_connected;
  apply_pbctopo_search_identifiability(
      result, uniqueness_search_exhaustive, uniqueness_search_exhaustive,
      equivalent_best_assignments_within_domain,
      feasible_assignment_set_enumerated,
      feasible_assignments_within_domain);
  return result;
}

} // namespace titan_pbctopo
