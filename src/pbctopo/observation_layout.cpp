#include "observation_layout.h"

#include "image_assignment.h"
#include "observation_binding.h"
#include "observation_schema.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <locale>
#include <sstream>
#include <utility>

namespace titan_pbctopo {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_byte(std::uint64_t &hash, std::uint8_t value) {
  hash ^= value;
  hash *= kFnvPrime;
}

void hash_word(std::uint64_t &hash, std::uint64_t word) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    hash_byte(hash, static_cast<std::uint8_t>(word >> shift));
}

std::string serialize_source_atom_ids(
    const std::vector<std::size_t> &source_atom_ids) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  for (std::size_t idx = 0; idx < source_atom_ids.size(); ++idx) {
    if (idx != 0)
      out << '|';
    out << source_atom_ids[idx];
  }
  return out.str();
}

std::string serialize_exact_membership(
    const std::vector<PbctopoSourceAtomKey> &membership) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  for (std::size_t idx = 0; idx < membership.size(); ++idx) {
    if (idx != 0)
      out << '|';
    out << membership[idx].owner << ':' << membership[idx].source_atom_id;
  }
  return out.str();
}

std::string serialize_exact_atom_masses(
    const std::vector<double> &masses) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out.precision(std::numeric_limits<double>::max_digits10);
  for (std::size_t idx = 0; idx < masses.size(); ++idx) {
    if (idx != 0)
      out << '|';
    out << masses[idx];
  }
  return out.str();
}

bool source_atom_key_less(const PbctopoSourceAtomKey &lhs,
                          const PbctopoSourceAtomKey &rhs) noexcept {
  if (lhs.owner != rhs.owner)
    return lhs.owner < rhs.owner;
  return lhs.source_atom_id < rhs.source_atom_id;
}

bool hard_edge_less(const PbctopoObservationHardEdge &lhs,
                    const PbctopoObservationHardEdge &rhs) noexcept {
  if (lhs.atom_a != rhs.atom_a)
    return lhs.atom_a < rhs.atom_a;
  if (lhs.atom_b != rhs.atom_b)
    return lhs.atom_b < rhs.atom_b;
  return static_cast<std::uint8_t>(lhs.kind) <
         static_cast<std::uint8_t>(rhs.kind);
}

bool hard_graph_source_matches_edges(
    PbctopoHardGraphSource source,
    std::span<const PbctopoObservationHardEdge> edges) noexcept {
  bool has_topology = false;
  bool has_metadata = false;
  for (const auto &edge : edges) {
    has_topology = has_topology ||
                   edge.kind ==
                       PbctopoObservationHardEdgeKind::TopologyBond;
    has_metadata = has_metadata ||
                   edge.kind ==
                       PbctopoObservationHardEdgeKind::ValidatedMetadata;
  }
  switch (source) {
  case PbctopoHardGraphSource::None:
    return !has_topology && !has_metadata;
  case PbctopoHardGraphSource::ExplicitTopology:
    return has_topology && !has_metadata;
  case PbctopoHardGraphSource::ValidatedMetadata:
    return !has_topology && has_metadata;
  case PbctopoHardGraphSource::MixedHard:
    return has_topology && has_metadata;
  }
  return false;
}

} // namespace

PbctopoComponentLayoutObservation build_pbctopo_component_layout_observation(
    std::span<const std::vector<std::size_t>> components,
    std::span<const PbctopoAtom> atoms,
    std::span<const PbctopoLocalEdge> local_edges,
    PbctopoHardGraphSource hard_graph_source) {
  PbctopoComponentLayoutObservation layout;
  if (components.empty() || atoms.empty() ||
      atoms.size() >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return layout;
  }

  const std::vector<Int3> identity_offsets(components.size());
  const PbctopoImageAssignment assignment =
      make_pbctopo_image_assignment(identity_offsets, components, atoms);
  if (!assignment.valid ||
      assignment.component_ids.size() != components.size()) {
    return layout;
  }

  layout.layout_signature = assignment.layout_signature;
  layout.hard_graph_source = hard_graph_source;
  layout.components.reserve(components.size());
  std::vector<PbctopoSourceAtomKey> all_atom_keys;
  all_atom_keys.reserve(atoms.size());
  for (std::size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
    const auto &component = components[comp_idx];
    if (component.empty())
      return {};
    PbctopoComponentMembership membership;
    membership.component_id = assignment.component_ids[comp_idx];
    membership.exact_atom_membership.reserve(component.size());
    std::vector<std::pair<PbctopoSourceAtomKey, double>> weighted_atoms;
    weighted_atoms.reserve(component.size());
    for (const std::size_t atom_idx : component) {
      if (atom_idx >= atoms.size())
        return {};
      weighted_atoms.push_back(
          {{atoms[atom_idx].owner, atoms[atom_idx].source_atom_id},
           atoms[atom_idx].has_atomic_mass &&
                   std::isfinite(atoms[atom_idx].atomic_mass) &&
                   atoms[atom_idx].atomic_mass > 0.0
               ? atoms[atom_idx].atomic_mass
               : 0.0});
    }
    std::sort(weighted_atoms.begin(), weighted_atoms.end(),
              [](const auto &lhs, const auto &rhs) {
                return source_atom_key_less(lhs.first, rhs.first);
              });
    membership.exact_atom_masses.reserve(weighted_atoms.size());
    for (const auto &[atom, mass] : weighted_atoms) {
      membership.exact_atom_membership.push_back(atom);
      membership.exact_atom_masses.push_back(mass);
    }
    if (std::adjacent_find(membership.exact_atom_membership.begin(),
                           membership.exact_atom_membership.end()) !=
        membership.exact_atom_membership.end()) {
      return {};
    }
    membership.owner = membership.exact_atom_membership.front().owner;
    membership.root_atom_id =
        membership.exact_atom_membership.front().source_atom_id;
    membership.source_atom_ids.reserve(
        membership.exact_atom_membership.size());
    for (const auto &atom : membership.exact_atom_membership) {
      membership.mixed_owner =
          membership.mixed_owner || atom.owner != membership.owner;
      membership.source_atom_ids.push_back(atom.source_atom_id);
      all_atom_keys.push_back(atom);
    }
    std::sort(membership.source_atom_ids.begin(),
              membership.source_atom_ids.end());
    layout.components.push_back(std::move(membership));
  }
  std::sort(all_atom_keys.begin(), all_atom_keys.end(), source_atom_key_less);
  if (all_atom_keys.size() != atoms.size() ||
      std::adjacent_find(all_atom_keys.begin(), all_atom_keys.end()) !=
          all_atom_keys.end()) {
    return {};
  }
  layout.canonical_atom_universe_hash =
      hash_pbctopo_canonical_atom_universe(atoms);
  if (layout.canonical_atom_universe_hash.empty())
    return {};

  std::sort(layout.components.begin(), layout.components.end(),
            [](const PbctopoComponentMembership &lhs,
               const PbctopoComponentMembership &rhs) {
              return lhs.component_id < rhs.component_id;
            });
  if (std::adjacent_find(
          layout.components.begin(), layout.components.end(),
          [](const PbctopoComponentMembership &lhs,
             const PbctopoComponentMembership &rhs) {
            return lhs.component_id == rhs.component_id;
          }) != layout.components.end()) {
    return {};
  }

  std::vector<std::pair<PbctopoSourceAtomKey, std::size_t>> atom_order;
  atom_order.reserve(atoms.size());
  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    atom_order.push_back(
        {{atoms[atom_idx].owner, atoms[atom_idx].source_atom_id}, atom_idx});
  }
  std::sort(atom_order.begin(), atom_order.end(), [](const auto &lhs,
                                                      const auto &rhs) {
    return source_atom_key_less(lhs.first, rhs.first);
  });
  std::vector<std::uint32_t> dense_atom_index(atoms.size());
  for (std::size_t dense_idx = 0; dense_idx < atom_order.size(); ++dense_idx) {
    if (dense_idx != 0 &&
        atom_order[dense_idx - 1].first == atom_order[dense_idx].first) {
      return {};
    }
    dense_atom_index[atom_order[dense_idx].second] =
        static_cast<std::uint32_t>(dense_idx);
  }
  layout.hard_edges.reserve(local_edges.size());
  for (const auto &edge : local_edges) {
    if (!pbctopo_edge_is_hard_connectivity(edge.kind))
      continue;
    if (edge.a >= atoms.size() || edge.b >= atoms.size() || edge.a == edge.b)
      return {};
    std::uint32_t atom_a = dense_atom_index[edge.a];
    std::uint32_t atom_b = dense_atom_index[edge.b];
    if (atom_b < atom_a)
      std::swap(atom_a, atom_b);
    const auto kind = edge.kind == PbctopoLocalEdgeKind::TopologyBond
                          ? PbctopoObservationHardEdgeKind::TopologyBond
                          : PbctopoObservationHardEdgeKind::ValidatedMetadata;
    layout.hard_edges.push_back({atom_a, atom_b, kind});
  }
  std::sort(layout.hard_edges.begin(), layout.hard_edges.end(),
            hard_edge_less);
  layout.hard_edges.erase(
      std::unique(layout.hard_edges.begin(), layout.hard_edges.end()),
      layout.hard_edges.end());
  if (!hard_graph_source_matches_edges(layout.hard_graph_source,
                                       layout.hard_edges)) {
    return {};
  }

  std::uint64_t epoch = kFnvOffset;
  hash_word(epoch, VIBE_OBSERVATION_SCHEMA_VERSION);
  hash_word(epoch, static_cast<std::uint8_t>(layout.hard_graph_source));
  hash_word(epoch, layout.layout_signature);
  hash_word(epoch, layout.components.size());
  for (const auto &component : layout.components) {
    hash_word(epoch, component.component_id);
    hash_word(epoch, component.exact_atom_membership.size());
    for (const auto &atom : component.exact_atom_membership) {
      hash_word(epoch, static_cast<std::uint64_t>(
                           static_cast<std::int64_t>(atom.owner)));
      hash_word(epoch, atom.source_atom_id);
    }
  }
  hash_word(epoch, layout.hard_edges.size());
  for (const auto &edge : layout.hard_edges) {
    hash_word(epoch, edge.atom_a);
    hash_word(epoch, edge.atom_b);
    hash_word(epoch, static_cast<std::uint8_t>(edge.kind));
  }
  layout.topology_epoch_id = epoch == 0 ? kFnvOffset : epoch;
  layout.valid = true;
  return layout;
}

bool same_pbctopo_component_layout_observation(
    const PbctopoComponentLayoutObservation &lhs,
    const PbctopoComponentLayoutObservation &rhs) {
  return lhs.valid && rhs.valid &&
         lhs.topology_epoch_id == rhs.topology_epoch_id &&
         lhs.layout_signature == rhs.layout_signature &&
         lhs.canonical_atom_universe_hash ==
             rhs.canonical_atom_universe_hash &&
         lhs.hard_graph_source == rhs.hard_graph_source &&
         lhs.components == rhs.components && lhs.hard_edges == rhs.hard_edges;
}

void write_pbctopo_component_layout_header(std::ostream &out) {
  out << "schema_version,producer_version,coordinate_unit,time_unit,"
         "endianness,floating_format,identity_hash_algorithm,"
         "binding_hash_algorithm,payload_checksum_algorithm,binding_mode,"
         "negative_zero_policy,"
         "nonfinite_policy,observation_scope,topology_epoch_id,layout_signature,"
         "canonical_atom_universe_hash,"
         "component_id,component_index,owner,root_atom_id,"
         "component_atom_count,sorted_source_atom_ids,"
         "exact_atom_membership,exact_atom_masses,hard_graph_source\n";
}

void write_pbctopo_component_layout_rows(
    std::ostream &out, const PbctopoComponentLayoutObservation &layout) {
  if (!layout.valid)
    return;
  for (std::size_t idx = 0; idx < layout.components.size(); ++idx) {
    const auto &component = layout.components[idx];
    out << VIBE_OBSERVATION_SCHEMA_VERSION << ','
        << vibe_observation_producer_version() << ','
        << VIBE_OBSERVATION_COORDINATE_UNIT << ','
        << VIBE_OBSERVATION_TIME_UNIT << ','
        << VIBE_OBSERVATION_ENDIANNESS << ','
        << VIBE_OBSERVATION_FLOATING_FORMAT << ','
        << VIBE_OBSERVATION_IDENTITY_HASH_ALGORITHM << ','
        << VIBE_OBSERVATION_BINDING_HASH_ALGORITHM << ','
        << VIBE_OBSERVATION_PAYLOAD_CHECKSUM_ALGORITHM << ','
        << VIBE_OBSERVATION_BINDING_MODE << ','
        << VIBE_OBSERVATION_NEGATIVE_ZERO_POLICY << ','
        << VIBE_OBSERVATION_NONFINITE_POLICY
        << ",static_component_membership," << layout.topology_epoch_id << ','
        << layout.layout_signature << ','
        << format_pbctopo_hash128(layout.canonical_atom_universe_hash) << ','
        << component.component_id << ','
        << idx << ',';
    if (component.mixed_owner)
      out << "mixed";
    else
      out << component.owner;
    out << ',' << component.root_atom_id << ','
        << component.exact_atom_membership.size() << ','
        << serialize_source_atom_ids(component.source_atom_ids) << ','
        << serialize_exact_membership(component.exact_atom_membership) << ','
        << serialize_exact_atom_masses(component.exact_atom_masses) << ','
        << pbctopo_hard_graph_source_name(layout.hard_graph_source) << '\n';
  }
}

const char *pbctopo_observation_hard_edge_kind_name(
    PbctopoObservationHardEdgeKind kind) noexcept {
  switch (kind) {
  case PbctopoObservationHardEdgeKind::TopologyBond:
    return "topology_bond";
  case PbctopoObservationHardEdgeKind::ValidatedMetadata:
    return "validated_metadata";
  }
  return "invalid";
}

void write_pbctopo_hard_edge_header(std::ostream &out) {
  out << "schema_version,producer_version,coordinate_unit,time_unit,"
         "endianness,floating_format,identity_hash_algorithm,"
         "binding_hash_algorithm,payload_checksum_algorithm,binding_mode,"
         "negative_zero_policy,"
         "nonfinite_policy,observation_scope,topology_epoch_id,layout_signature,"
         "edge_index,atom_a,atom_b,edge_kind,hard_graph_source\n";
}

void write_pbctopo_hard_edge_rows(
    std::ostream &out, const PbctopoComponentLayoutObservation &layout) {
  if (!layout.valid)
    return;
  for (std::size_t idx = 0; idx < layout.hard_edges.size(); ++idx) {
    const auto &edge = layout.hard_edges[idx];
    out << VIBE_OBSERVATION_SCHEMA_VERSION << ','
        << vibe_observation_producer_version() << ','
        << VIBE_OBSERVATION_COORDINATE_UNIT << ','
        << VIBE_OBSERVATION_TIME_UNIT << ','
        << VIBE_OBSERVATION_ENDIANNESS << ','
        << VIBE_OBSERVATION_FLOATING_FORMAT << ','
        << VIBE_OBSERVATION_IDENTITY_HASH_ALGORITHM << ','
        << VIBE_OBSERVATION_BINDING_HASH_ALGORITHM << ','
        << VIBE_OBSERVATION_PAYLOAD_CHECKSUM_ALGORITHM << ','
        << VIBE_OBSERVATION_BINDING_MODE << ','
        << VIBE_OBSERVATION_NEGATIVE_ZERO_POLICY << ','
        << VIBE_OBSERVATION_NONFINITE_POLICY << ",static_hard_edges,"
        << layout.topology_epoch_id << ',' << layout.layout_signature << ','
        << idx << ',' << edge.atom_a << ',' << edge.atom_b << ','
        << pbctopo_observation_hard_edge_kind_name(edge.kind) << ','
        << pbctopo_hard_graph_source_name(layout.hard_graph_source) << '\n';
  }
}

} // namespace titan_pbctopo
