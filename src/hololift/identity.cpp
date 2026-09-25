#include "identity.h"

#include "../titan_sha256.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace titan_hololift {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
class IdentityHash128Builder {
public:
  explicit IdentityHash128Builder(std::uint64_t domain) noexcept {
    hash_.word(domain);
  }

  void word(std::uint64_t value) noexcept { hash_.word(value); }

  void hash128(HoloLiftHash128 value) noexcept {
    word(value.lo);
    word(value.hi);
  }

  [[nodiscard]] HoloLiftHash128 finish() const noexcept {
    const auto digest = hash_.finish128();
    return {digest[0], digest[1]};
  }

private:
  titan_hash::Sha256Builder hash_;
};

void hash_byte(std::uint64_t &hash, std::uint8_t value) noexcept {
  hash ^= value;
  hash *= kFnvPrime;
}

void hash_word(std::uint64_t &hash, std::uint64_t value) noexcept {
  for (unsigned shift = 0; shift < 64; shift += 8)
    hash_byte(hash, static_cast<std::uint8_t>(value >> shift));
}

bool atom_key_less(const HoloLiftSourceAtomKey &lhs,
                   const HoloLiftSourceAtomKey &rhs) noexcept {
  if (lhs.owner != rhs.owner)
    return lhs.owner < rhs.owner;
  return lhs.source_atom_id < rhs.source_atom_id;
}

bool hard_edge_less(const HoloLiftHardEdgeRecord &lhs,
                    const HoloLiftHardEdgeRecord &rhs) noexcept {
  if (lhs.atom_a != rhs.atom_a)
    return lhs.atom_a < rhs.atom_a;
  if (lhs.atom_b != rhs.atom_b)
    return lhs.atom_b < rhs.atom_b;
  return static_cast<std::uint8_t>(lhs.kind) <
         static_cast<std::uint8_t>(rhs.kind);
}

bool valid_hard_graph_source(HoloLiftHardGraphSource source) noexcept {
  switch (source) {
  case HoloLiftHardGraphSource::None:
  case HoloLiftHardGraphSource::ExplicitTopology:
  case HoloLiftHardGraphSource::ValidatedMetadata:
  case HoloLiftHardGraphSource::MixedHard:
    return true;
  }
  return false;
}

bool valid_hard_edge_kind(HoloLiftHardEdgeKind kind) noexcept {
  return kind == HoloLiftHardEdgeKind::TopologyBond ||
         kind == HoloLiftHardEdgeKind::ValidatedMetadata;
}

bool checked_subtract(std::int64_t lhs, std::int64_t rhs,
                      std::int64_t &result) noexcept {
  constexpr std::int64_t min_value = std::numeric_limits<std::int64_t>::min();
  constexpr std::int64_t max_value = std::numeric_limits<std::int64_t>::max();
  if ((rhs > 0 && lhs < min_value + rhs) ||
      (rhs < 0 && lhs > max_value + rhs)) {
    return false;
  }
  result = lhs - rhs;
  return true;
}

bool checked_subtract(const HoloLiftLatticeImage &lhs,
                      const HoloLiftLatticeImage &rhs,
                      HoloLiftLatticeImage &result) noexcept {
  return checked_subtract(lhs.x, rhs.x, result.x) &&
         checked_subtract(lhs.y, rhs.y, result.y) &&
         checked_subtract(lhs.z, rhs.z, result.z);
}

std::expected<std::vector<const HoloLiftComponentRecord *>, std::string>
sorted_components(std::span<const HoloLiftComponentRecord> components) {
  if (components.empty())
    return std::unexpected("HoloLift component set is empty");
  std::vector<const HoloLiftComponentRecord *> sorted;
  sorted.reserve(components.size());
  for (const auto &component : components) {
    if (component.component_id == 0)
      return std::unexpected("HoloLift component id is zero");
    sorted.push_back(&component);
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto *lhs, const auto *rhs) {
    return lhs->component_id < rhs->component_id;
  });
  for (std::size_t idx = 1; idx < sorted.size(); ++idx) {
    if (sorted[idx - 1]->component_id == sorted[idx]->component_id)
      return std::unexpected("duplicate HoloLift component id");
  }
  return sorted;
}

} // namespace

std::expected<void, std::string>
HoloLiftIdentityRegistry::register_component(
    std::uint64_t component_id,
    std::span<const HoloLiftSourceAtomKey> exact_membership) {
  if (component_id == 0 || exact_membership.empty()) {
    return std::unexpected(
        "invalid HoloLift component identity registration");
  }
  const auto found = component_memberships_.find(component_id);
  if (found == component_memberships_.end()) {
    component_memberships_.emplace(
        component_id,
        std::vector<HoloLiftSourceAtomKey>(
            exact_membership.begin(), exact_membership.end()));
    return {};
  }
  if (found->second.size() != exact_membership.size() ||
      !std::equal(found->second.begin(), found->second.end(),
                  exact_membership.begin())) {
    return std::unexpected(
        "HoloLift component id collision across topology epochs");
  }
  return {};
}

std::expected<void, std::string>
HoloLiftIdentityRegistry::register_layout(
    std::uint64_t layout_signature,
    std::span<const std::uint64_t> ordered_component_ids) {
  if (layout_signature == 0 || ordered_component_ids.empty()) {
    return std::unexpected(
        "invalid HoloLift layout identity registration");
  }
  const auto found = layout_components_.find(layout_signature);
  if (found == layout_components_.end()) {
    layout_components_.emplace(
        layout_signature,
        std::vector<std::uint64_t>(
            ordered_component_ids.begin(),
            ordered_component_ids.end()));
    return {};
  }
  if (found->second.size() != ordered_component_ids.size() ||
      !std::equal(found->second.begin(), found->second.end(),
                  ordered_component_ids.begin())) {
    return std::unexpected(
        "HoloLift layout signature collision across topology epochs");
  }
  return {};
}

std::expected<std::uint64_t, std::string>
hololift_component_id(std::span<const HoloLiftSourceAtomKey> membership) {
  if (membership.empty())
    return std::unexpected("HoloLift component membership is empty");
  for (std::size_t idx = 1; idx < membership.size(); ++idx) {
    if (!atom_key_less(membership[idx - 1], membership[idx])) {
      return std::unexpected(
          "HoloLift exact atom membership is not strictly sorted");
    }
  }
  std::uint64_t hash = kFnvOffset;
  hash_word(hash, membership.size());
  for (const auto &atom : membership) {
    hash_word(hash, static_cast<std::uint64_t>(
                        static_cast<std::int64_t>(atom.owner)));
    hash_word(hash, atom.source_atom_id);
  }
  return hash == 0 ? kFnvOffset : hash;
}

std::expected<HoloLiftHash128, std::string>
hololift_component_identity128(
    std::span<const HoloLiftSourceAtomKey> membership) {
  if (membership.empty())
    return std::unexpected("HoloLift component membership is empty");
  for (std::size_t idx = 1; idx < membership.size(); ++idx) {
    if (!atom_key_less(membership[idx - 1], membership[idx])) {
      return std::unexpected(
          "HoloLift exact atom membership is not strictly sorted");
    }
  }
  IdentityHash128Builder hash(0x484f4c4f434f4d50ULL); // HOLOCOMP
  hash.word(membership.size());
  for (const auto &atom : membership) {
    hash.word(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(atom.owner)));
    hash.word(atom.source_atom_id);
  }
  return hash.finish();
}

std::expected<std::uint64_t, std::string> hololift_layout_signature(
    std::span<const HoloLiftComponentRecord> components) {
  auto sorted = sorted_components(components);
  if (!sorted)
    return std::unexpected(sorted.error());
  std::uint64_t hash = kFnvOffset;
  hash_word(hash, sorted->size());
  for (const auto *component : *sorted)
    hash_word(hash, component->component_id);
  return hash == 0 ? kFnvOffset : hash;
}

std::expected<HoloLiftHash128, std::string>
hololift_layout_identity128(
    std::span<const HoloLiftComponentRecord> components) {
  auto sorted = sorted_components(components);
  if (!sorted)
    return std::unexpected(sorted.error());
  IdentityHash128Builder hash(0x484f4c4f4c41594fULL); // HOLOLAYO
  hash.word(sorted->size());
  for (const auto *component : *sorted) {
    if (component->component_identity.empty()) {
      return std::unexpected(
          "HoloLift component lacks a 128-bit identity");
    }
    hash.hash128(component->component_identity);
  }
  return hash.finish();
}

std::expected<std::uint64_t, std::string> hololift_topology_epoch_id(
    std::uint32_t source_schema_version,
    HoloLiftHardGraphSource hard_graph_source,
    std::uint64_t layout_signature,
    std::span<const HoloLiftComponentRecord> components,
    std::span<const HoloLiftSourceAtomKey> exact_atom_memberships,
    std::span<const HoloLiftHardEdgeRecord> hard_edges) {
  if (source_schema_version == 0)
    return std::unexpected("HoloLift source schema version is zero");
  if (!valid_hard_graph_source(hard_graph_source))
    return std::unexpected("HoloLift hard graph source is invalid");
  if (layout_signature == 0)
    return std::unexpected("HoloLift layout signature is zero");
  auto sorted = sorted_components(components);
  if (!sorted)
    return std::unexpected(sorted.error());

  std::uint64_t hash = kFnvOffset;
  hash_word(hash, source_schema_version);
  hash_word(hash, static_cast<std::uint8_t>(hard_graph_source));
  hash_word(hash, layout_signature);
  hash_word(hash, sorted->size());
  std::size_t epoch_atom_count = 0;
  for (const auto *component : *sorted) {
    if (!component->exact_atom_membership.valid_for(
            exact_atom_memberships.size()) ||
        component->exact_atom_membership.count == 0) {
      return std::unexpected("invalid HoloLift component membership range");
    }
    const auto membership = exact_atom_memberships.subspan(
        component->exact_atom_membership.begin,
        component->exact_atom_membership.count);
    const auto component_id = hololift_component_id(membership);
    if (!component_id)
      return std::unexpected(component_id.error());
    if (*component_id != component->component_id) {
      return std::unexpected(
          "HoloLift component id does not match exact membership");
    }
    if (component->exact_atom_membership.count >
        std::numeric_limits<std::size_t>::max() - epoch_atom_count) {
      return std::unexpected("HoloLift topology epoch atom count overflow");
    }
    epoch_atom_count += component->exact_atom_membership.count;
    hash_word(hash, component->component_id);
    hash_word(hash, membership.size());
    for (const auto &atom : membership) {
      hash_word(hash, static_cast<std::uint64_t>(
                          static_cast<std::int64_t>(atom.owner)));
      hash_word(hash, atom.source_atom_id);
    }
  }
  if (epoch_atom_count >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return std::unexpected("HoloLift topology epoch has too many atoms");
  }
  hash_word(hash, hard_edges.size());
  for (std::size_t idx = 0; idx < hard_edges.size(); ++idx) {
    const auto &edge = hard_edges[idx];
    if (edge.atom_a >= edge.atom_b ||
        edge.atom_b >= epoch_atom_count ||
        !valid_hard_edge_kind(edge.kind)) {
      return std::unexpected("invalid canonical HoloLift hard edge");
    }
    if (idx != 0 && !hard_edge_less(hard_edges[idx - 1], edge)) {
      return std::unexpected(
          "HoloLift hard edges are not strictly canonical");
    }
    hash_word(hash, edge.atom_a);
    hash_word(hash, edge.atom_b);
    hash_word(hash, static_cast<std::uint8_t>(edge.kind));
  }
  return hash == 0 ? kFnvOffset : hash;
}

std::expected<HoloLiftHash128, std::string> hololift_topology_epoch_identity128(
    std::uint32_t source_schema_version,
    HoloLiftHardGraphSource hard_graph_source,
    HoloLiftHash128 layout_identity,
    std::span<const HoloLiftComponentRecord> canonical_components,
    std::span<const HoloLiftHardEdgeRecord> hard_edges) {
  if (source_schema_version == 0 || layout_identity.empty() ||
      !valid_hard_graph_source(hard_graph_source) ||
      canonical_components.empty()) {
    return std::unexpected(
        "incomplete HoloLift 128-bit topology epoch identity input");
  }
  std::size_t atom_count = 0;
  for (std::size_t idx = 0; idx < canonical_components.size(); ++idx) {
    const auto &component = canonical_components[idx];
    if (component.component_identity.empty() ||
        component.exact_atom_membership.count == 0 ||
        (idx != 0 && canonical_components[idx - 1].component_id >=
                         component.component_id) ||
        component.exact_atom_membership.count >
            std::numeric_limits<std::size_t>::max() - atom_count) {
      return std::unexpected(
          "invalid canonical component in HoloLift 128-bit epoch identity");
    }
    atom_count += component.exact_atom_membership.count;
  }
  if (atom_count >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return std::unexpected("HoloLift topology epoch has too many atoms");
  }

  IdentityHash128Builder hash(0x484f4c4f45504f43ULL); // HOLOEPOC
  hash.word(source_schema_version);
  hash.word(static_cast<std::uint8_t>(hard_graph_source));
  hash.hash128(layout_identity);
  hash.word(canonical_components.size());
  for (const auto &component : canonical_components)
    hash.hash128(component.component_identity);
  hash.word(hard_edges.size());
  for (std::size_t idx = 0; idx < hard_edges.size(); ++idx) {
    const auto &edge = hard_edges[idx];
    if (edge.atom_a >= edge.atom_b || edge.atom_b >= atom_count ||
        !valid_hard_edge_kind(edge.kind) ||
        (idx != 0 && !hard_edge_less(hard_edges[idx - 1], edge))) {
      return std::unexpected("invalid canonical HoloLift hard edge");
    }
    hash.word(edge.atom_a);
    hash.word(edge.atom_b);
    hash.word(static_cast<std::uint8_t>(edge.kind));
  }
  return hash.finish();
}

std::expected<std::uint64_t, std::string> hololift_assignment_signature(
    std::uint64_t layout_signature,
    std::span<const HoloLiftComponentImage> canonical_component_images) {
  if (layout_signature == 0)
    return std::unexpected("HoloLift assignment layout signature is zero");
  if (canonical_component_images.empty())
    return std::unexpected("HoloLift component-image assignment is empty");
  for (std::size_t idx = 0; idx < canonical_component_images.size(); ++idx) {
    if (canonical_component_images[idx].component_id == 0) {
      return std::unexpected("HoloLift assignment component id is zero");
    }
    if (idx != 0 &&
        canonical_component_images[idx - 1].component_id >=
            canonical_component_images[idx].component_id) {
      return std::unexpected(
          "HoloLift assignment component ids are not strictly sorted");
    }
  }

  std::uint64_t hash = kFnvOffset;
  hash_word(hash, canonical_component_images.size());
  hash_word(hash, layout_signature);
  for (const auto &entry : canonical_component_images) {
    hash_word(hash, entry.component_id);
    hash_word(hash, static_cast<std::uint64_t>(entry.image.x));
    hash_word(hash, static_cast<std::uint64_t>(entry.image.y));
    hash_word(hash, static_cast<std::uint64_t>(entry.image.z));
  }
  return hash == 0 ? kFnvOffset : hash;
}

std::expected<HoloLiftHash128, std::string>
hololift_assignment_identity128_canonical(
    HoloLiftHash128 layout_identity,
    std::span<const HoloLiftComponentRecord> canonical_components,
    std::span<const HoloLiftLatticeImage> canonical_component_images) {
  if (layout_identity.empty() || canonical_components.empty() ||
      canonical_component_images.size() != canonical_components.size()) {
    return std::unexpected(
        "incomplete HoloLift 128-bit assignment identity input");
  }
  IdentityHash128Builder hash(0x484f4c4f4153534eULL); // HOLOASSN
  hash.hash128(layout_identity);
  hash.word(canonical_components.size());
  for (std::size_t idx = 0; idx < canonical_components.size(); ++idx) {
    const auto &component = canonical_components[idx];
    const auto &image = canonical_component_images[idx];
    if (component.component_identity.empty() ||
        (idx != 0 && canonical_components[idx - 1].component_id >=
                         component.component_id)) {
      return std::unexpected(
          "HoloLift canonical components are not strictly identified");
    }
    hash.hash128(component.component_identity);
    hash.word(static_cast<std::uint64_t>(image.x));
    hash.word(static_cast<std::uint64_t>(image.y));
    hash.word(static_cast<std::uint64_t>(image.z));
  }
  return hash.finish();
}

std::expected<std::uint64_t, std::string>
hololift_assignment_signature_canonical(
    std::uint64_t layout_signature,
    std::span<const HoloLiftComponentRecord> canonical_components,
    std::span<const HoloLiftLatticeImage> canonical_component_images) {
  if (layout_signature == 0)
    return std::unexpected("HoloLift assignment layout signature is zero");
  if (canonical_components.empty() ||
      canonical_component_images.size() != canonical_components.size()) {
    return std::unexpected("HoloLift assignment/component count mismatch");
  }
  for (std::size_t idx = 0; idx < canonical_components.size(); ++idx) {
    if (canonical_components[idx].component_id == 0 ||
        (idx != 0 && canonical_components[idx - 1].component_id >=
                         canonical_components[idx].component_id)) {
      return std::unexpected(
          "HoloLift canonical components are not strictly sorted");
    }
  }

  std::uint64_t hash = kFnvOffset;
  hash_word(hash, canonical_component_images.size());
  hash_word(hash, layout_signature);
  for (std::size_t idx = 0; idx < canonical_components.size(); ++idx) {
    const auto &image = canonical_component_images[idx];
    hash_word(hash, canonical_components[idx].component_id);
    hash_word(hash, static_cast<std::uint64_t>(image.x));
    hash_word(hash, static_cast<std::uint64_t>(image.y));
    hash_word(hash, static_cast<std::uint64_t>(image.z));
  }
  return hash == 0 ? kFnvOffset : hash;
}

std::expected<HoloLiftCanonicalAssignment, std::string>
canonicalize_hololift_assignment(
    std::uint64_t layout_signature,
    std::span<const HoloLiftComponentRecord> components,
    std::span<const HoloLiftComponentImage> raw_component_images) {
  auto sorted = sorted_components(components);
  if (!sorted)
    return std::unexpected(sorted.error());
  if (raw_component_images.size() != sorted->size()) {
    return std::unexpected(
        "HoloLift assignment/component count mismatch");
  }

  HoloLiftCanonicalAssignment result;
  result.layout_signature = layout_signature;
  result.component_images.assign(raw_component_images.begin(),
                                 raw_component_images.end());
  std::sort(result.component_images.begin(), result.component_images.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.component_id < rhs.component_id;
            });
  for (std::size_t idx = 0; idx < sorted->size(); ++idx) {
    if (result.component_images[idx].component_id !=
        (*sorted)[idx]->component_id) {
      return std::unexpected(
          "HoloLift assignment does not match topology-epoch components");
    }
  }

  const HoloLiftComponentRecord *root = sorted->front();
  for (const auto *component : *sorted) {
    if (component->exact_atom_membership.count >
            root->exact_atom_membership.count ||
        (component->exact_atom_membership.count ==
             root->exact_atom_membership.count &&
         component->component_id < root->component_id)) {
      root = component;
    }
  }
  const auto root_image = std::lower_bound(
      result.component_images.begin(), result.component_images.end(),
      root->component_id, [](const HoloLiftComponentImage &entry,
                             std::uint64_t component_id) {
        return entry.component_id < component_id;
      });
  if (root_image == result.component_images.end() ||
      root_image->component_id != root->component_id) {
    return std::unexpected("HoloLift root component image is missing");
  }
  result.global_gauge = root_image->image;
  for (auto &entry : result.component_images) {
    HoloLiftLatticeImage canonical;
    if (!checked_subtract(entry.image, result.global_gauge, canonical)) {
      return std::unexpected(
          "HoloLift lattice image overflow during gauge normalization");
    }
    entry.image = canonical;
  }
  const auto signature = hololift_assignment_signature(
      layout_signature, result.component_images);
  if (!signature)
    return std::unexpected(signature.error());
  result.assignment_signature = *signature;
  const auto layout_identity = hololift_layout_identity128(components);
  if (!layout_identity)
    return std::unexpected(layout_identity.error());
  std::vector<HoloLiftComponentRecord> canonical_components;
  std::vector<HoloLiftLatticeImage> canonical_images;
  canonical_components.reserve(sorted->size());
  canonical_images.reserve(result.component_images.size());
  for (const auto *component : *sorted)
    canonical_components.push_back(*component);
  for (const auto &entry : result.component_images)
    canonical_images.push_back(entry.image);
  const auto assignment_identity = hololift_assignment_identity128_canonical(
      *layout_identity, canonical_components, canonical_images);
  if (!assignment_identity)
    return std::unexpected(assignment_identity.error());
  result.layout_identity = *layout_identity;
  result.assignment_identity = *assignment_identity;
  return result;
}

} // namespace titan_hololift
