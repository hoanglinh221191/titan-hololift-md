#ifndef TITAN_PBCTOPO_OBSERVATION_LAYOUT_H
#define TITAN_PBCTOPO_OBSERVATION_LAYOUT_H

#include "report.h"

#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace titan_pbctopo {

struct PbctopoSourceAtomKey {
  int owner = 0;
  std::size_t source_atom_id = 0;

  bool operator==(const PbctopoSourceAtomKey &) const = default;
};

struct PbctopoComponentMembership {
  std::uint64_t component_id = 0;
  int owner = 0;
  bool mixed_owner = false;
  std::size_t root_atom_id = 0;
  std::vector<std::size_t> source_atom_ids;
  std::vector<PbctopoSourceAtomKey> exact_atom_membership;
  // Aligned with exact_atom_membership; zero denotes unavailable mass.
  std::vector<double> exact_atom_masses;

  bool operator==(const PbctopoComponentMembership &) const = default;
};

enum class PbctopoObservationHardEdgeKind : std::uint8_t {
  TopologyBond,
  ValidatedMetadata,
};

struct PbctopoObservationHardEdge {
  std::uint32_t atom_a = 0;
  std::uint32_t atom_b = 0;
  PbctopoObservationHardEdgeKind kind =
      PbctopoObservationHardEdgeKind::TopologyBond;

  bool operator==(const PbctopoObservationHardEdge &) const = default;
};

struct PbctopoComponentLayoutObservation {
  std::uint64_t topology_epoch_id = 0;
  std::uint64_t layout_signature = 0;
  PbctopoHash128 canonical_atom_universe_hash;
  PbctopoHardGraphSource hard_graph_source = PbctopoHardGraphSource::None;
  std::vector<PbctopoComponentMembership> components;
  std::vector<PbctopoObservationHardEdge> hard_edges;
  bool valid = false;
};

PbctopoComponentLayoutObservation build_pbctopo_component_layout_observation(
    std::span<const std::vector<std::size_t>> components,
    std::span<const PbctopoAtom> atoms,
    std::span<const PbctopoLocalEdge> local_edges,
    PbctopoHardGraphSource hard_graph_source);

bool same_pbctopo_component_layout_observation(
    const PbctopoComponentLayoutObservation &lhs,
    const PbctopoComponentLayoutObservation &rhs);

void write_pbctopo_component_layout_header(std::ostream &out);

void write_pbctopo_component_layout_rows(
    std::ostream &out, const PbctopoComponentLayoutObservation &layout);

const char *pbctopo_observation_hard_edge_kind_name(
    PbctopoObservationHardEdgeKind kind) noexcept;

void write_pbctopo_hard_edge_header(std::ostream &out);

void write_pbctopo_hard_edge_rows(
    std::ostream &out, const PbctopoComponentLayoutObservation &layout);

} // namespace titan_pbctopo

#endif
