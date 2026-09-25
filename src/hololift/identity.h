#ifndef TITAN_HOLOLIFT_IDENTITY_H
#define TITAN_HOLOLIFT_IDENTITY_H

#include "types.h"

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace titan_hololift {

struct HoloLiftCanonicalAssignment {
  std::uint64_t layout_signature = 0;
  std::uint64_t assignment_signature = 0;
  HoloLiftHash128 layout_identity;
  HoloLiftHash128 assignment_identity;
  HoloLiftLatticeImage global_gauge;
  std::vector<HoloLiftComponentImage> component_images;
};

class HoloLiftIdentityRegistry {
public:
  // Inputs are canonical exact records; this guard detects one numeric lookup
  // key being reused for different exact content across topology epochs.
  std::expected<void, std::string> register_component(
      std::uint64_t component_id,
      std::span<const HoloLiftSourceAtomKey> exact_membership);

  std::expected<void, std::string> register_layout(
      std::uint64_t layout_signature,
      std::span<const std::uint64_t> ordered_component_ids);

private:
  std::unordered_map<std::uint64_t,
                     std::vector<HoloLiftSourceAtomKey>>
      component_memberships_;
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>
      layout_components_;
};

std::expected<std::uint64_t, std::string>
hololift_component_id(std::span<const HoloLiftSourceAtomKey> membership);

std::expected<HoloLiftHash128, std::string>
hololift_component_identity128(
    std::span<const HoloLiftSourceAtomKey> membership);

std::expected<std::uint64_t, std::string>
hololift_layout_signature(
    std::span<const HoloLiftComponentRecord> components);

std::expected<HoloLiftHash128, std::string>
hololift_layout_identity128(
    std::span<const HoloLiftComponentRecord> components);

std::expected<std::uint64_t, std::string> hololift_topology_epoch_id(
    std::uint32_t source_schema_version,
    HoloLiftHardGraphSource hard_graph_source,
    std::uint64_t layout_signature,
    std::span<const HoloLiftComponentRecord> components,
    std::span<const HoloLiftSourceAtomKey> exact_atom_memberships,
    std::span<const HoloLiftHardEdgeRecord> hard_edges);

std::expected<HoloLiftHash128, std::string> hololift_topology_epoch_identity128(
    std::uint32_t source_schema_version,
    HoloLiftHardGraphSource hard_graph_source,
    HoloLiftHash128 layout_identity,
    std::span<const HoloLiftComponentRecord> canonical_components,
    std::span<const HoloLiftHardEdgeRecord> hard_edges);

std::expected<std::uint64_t, std::string>
hololift_assignment_signature(
    std::uint64_t layout_signature,
    std::span<const HoloLiftComponentImage> canonical_component_images);

// O(C) fast path for epochs already validated in strict component-id order.
std::expected<std::uint64_t, std::string>
hololift_assignment_signature_canonical(
    std::uint64_t layout_signature,
    std::span<const HoloLiftComponentRecord> canonical_components,
    std::span<const HoloLiftLatticeImage> canonical_component_images);

std::expected<HoloLiftHash128, std::string>
hololift_assignment_identity128_canonical(
    HoloLiftHash128 layout_identity,
    std::span<const HoloLiftComponentRecord> canonical_components,
    std::span<const HoloLiftLatticeImage> canonical_component_images);

std::expected<HoloLiftCanonicalAssignment, std::string>
canonicalize_hololift_assignment(
    std::uint64_t layout_signature,
    std::span<const HoloLiftComponentRecord> components,
    std::span<const HoloLiftComponentImage> raw_component_images);

} // namespace titan_hololift

#endif
