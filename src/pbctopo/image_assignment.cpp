#include "image_assignment.h"

#include <algorithm>
#include <cstdint>

namespace titan_pbctopo {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_word(std::uint64_t &hash, std::uint64_t word) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    hash ^= static_cast<std::uint8_t>(word >> shift);
    hash *= kFnvPrime;
  }
}

std::uint64_t component_identity_legacy(
    const std::vector<std::size_t> &component) {
  std::vector<std::size_t> sorted_atoms = component;
  std::sort(sorted_atoms.begin(), sorted_atoms.end());
  std::uint64_t hash = kFnvOffset;
  hash_word(hash, static_cast<std::uint64_t>(sorted_atoms.size()));
  for (std::size_t atom_idx : sorted_atoms)
    hash_word(hash, static_cast<std::uint64_t>(atom_idx));
  return hash == 0 ? kFnvOffset : hash;
}

std::uint64_t component_identity(
    const std::vector<std::size_t> &component,
    std::span<const PbctopoAtom> atoms) {
  struct SourceAtomKey {
    int owner = 0;
    std::size_t source_atom_id = 0;
  };
  std::vector<SourceAtomKey> sorted_atoms;
  sorted_atoms.reserve(component.size());
  for (const std::size_t atom_idx : component) {
    if (atom_idx >= atoms.size())
      return 0;
    sorted_atoms.push_back(
        {atoms[atom_idx].owner, atoms[atom_idx].source_atom_id});
  }
  std::sort(sorted_atoms.begin(), sorted_atoms.end(),
            [](const SourceAtomKey &lhs, const SourceAtomKey &rhs) {
              if (lhs.owner != rhs.owner)
                return lhs.owner < rhs.owner;
              return lhs.source_atom_id < rhs.source_atom_id;
            });
  std::uint64_t hash = kFnvOffset;
  hash_word(hash, static_cast<std::uint64_t>(sorted_atoms.size()));
  for (const auto &atom : sorted_atoms) {
    hash_word(hash, static_cast<std::uint64_t>(
                        static_cast<std::int64_t>(atom.owner)));
    hash_word(hash, static_cast<std::uint64_t>(atom.source_atom_id));
  }
  return hash == 0 ? kFnvOffset : hash;
}

template <typename ComponentIdentityFn>
PbctopoImageAssignment make_assignment_impl(
    std::span<const Int3> component_offsets,
    std::span<const std::vector<std::size_t>> components,
    ComponentIdentityFn &&component_identity_fn) {
  PbctopoImageAssignment assignment;
  if (component_offsets.empty() ||
      component_offsets.size() != components.size()) {
    return assignment;
  }

  assignment.component_ids.reserve(components.size());
  for (const auto &component : components) {
    if (component.empty())
      return {};
    const std::uint64_t component_id = component_identity_fn(component);
    if (component_id == 0)
      return {};
    assignment.component_ids.push_back(component_id);
  }

  std::vector<std::uint64_t> sorted_component_ids = assignment.component_ids;
  std::sort(sorted_component_ids.begin(), sorted_component_ids.end());
  if (std::adjacent_find(sorted_component_ids.begin(),
                         sorted_component_ids.end()) !=
      sorted_component_ids.end()) {
    return {};
  }
  std::uint64_t layout_signature = kFnvOffset;
  hash_word(layout_signature,
            static_cast<std::uint64_t>(sorted_component_ids.size()));
  for (std::uint64_t component_id : sorted_component_ids)
    hash_word(layout_signature, component_id);
  assignment.layout_signature =
      layout_signature == 0 ? kFnvOffset : layout_signature;

  std::size_t root = 0;
  for (std::size_t idx = 1; idx < components.size(); ++idx) {
    if (components[idx].size() > components[root].size() ||
        (components[idx].size() == components[root].size() &&
         assignment.component_ids[idx] < assignment.component_ids[root])) {
      root = idx;
    }
  }

  assignment.global_gauge = component_offsets[root];
  assignment.component_offsets.reserve(component_offsets.size());
  std::uint64_t signature = kFnvOffset;
  hash_word(signature, static_cast<std::uint64_t>(component_offsets.size()));
  hash_word(signature, assignment.layout_signature);
  struct CanonicalComponentOffset {
    std::uint64_t component_id = 0;
    Int3 offset{};
  };
  std::vector<CanonicalComponentOffset> canonical_entries;
  canonical_entries.reserve(component_offsets.size());
  for (std::size_t idx = 0; idx < component_offsets.size(); ++idx) {
    Int3 canonical{};
    if (!checked_int3_subtract(component_offsets[idx],
                               assignment.global_gauge, canonical)) {
      return {};
    }
    assignment.component_offsets.push_back(canonical);
    canonical_entries.push_back({assignment.component_ids[idx], canonical});
  }
  std::sort(canonical_entries.begin(), canonical_entries.end(),
            [](const CanonicalComponentOffset &lhs,
               const CanonicalComponentOffset &rhs) {
              return lhs.component_id < rhs.component_id;
            });
  for (const auto &entry : canonical_entries) {
    hash_word(signature, entry.component_id);
    hash_word(signature, static_cast<std::uint64_t>(entry.offset.x));
    hash_word(signature, static_cast<std::uint64_t>(entry.offset.y));
    hash_word(signature, static_cast<std::uint64_t>(entry.offset.z));
  }
  assignment.signature = signature == 0 ? kFnvOffset : signature;
  assignment.valid = true;
  return assignment;
}

} // namespace

PbctopoImageAssignment make_pbctopo_image_assignment(
    std::span<const Int3> component_offsets,
    std::span<const std::vector<std::size_t>> components) {
  return make_assignment_impl(
      component_offsets, components,
      [](const std::vector<std::size_t> &component) {
        return component_identity_legacy(component);
      });
}

PbctopoImageAssignment make_pbctopo_image_assignment(
    std::span<const Int3> component_offsets,
    std::span<const std::vector<std::size_t>> components,
    std::span<const PbctopoAtom> atoms) {
  return make_assignment_impl(
      component_offsets, components,
      [atoms](const std::vector<std::size_t> &component) {
        return component_identity(component, atoms);
      });
}

bool same_pbctopo_image_assignment(const PbctopoImageAssignment &lhs,
                                    const PbctopoImageAssignment &rhs) {
  if (!lhs.valid || !rhs.valid || lhs.signature != rhs.signature ||
      lhs.layout_signature != rhs.layout_signature ||
      lhs.component_offsets.size() != rhs.component_offsets.size() ||
      lhs.component_ids.size() != lhs.component_offsets.size() ||
      rhs.component_ids.size() != rhs.component_offsets.size()) {
    return false;
  }
  struct Entry {
    std::uint64_t component_id = 0;
    Int3 offset{};
  };
  std::vector<Entry> lhs_entries;
  std::vector<Entry> rhs_entries;
  lhs_entries.reserve(lhs.component_offsets.size());
  rhs_entries.reserve(rhs.component_offsets.size());
  for (std::size_t idx = 0; idx < lhs.component_offsets.size(); ++idx)
    lhs_entries.push_back({lhs.component_ids[idx], lhs.component_offsets[idx]});
  for (std::size_t idx = 0; idx < rhs.component_offsets.size(); ++idx)
    rhs_entries.push_back({rhs.component_ids[idx], rhs.component_offsets[idx]});
  const auto by_id = [](const Entry &a, const Entry &b) {
    return a.component_id < b.component_id;
  };
  std::sort(lhs_entries.begin(), lhs_entries.end(), by_id);
  std::sort(rhs_entries.begin(), rhs_entries.end(), by_id);
  for (std::size_t idx = 0; idx < lhs_entries.size(); ++idx) {
    if (lhs_entries[idx].component_id != rhs_entries[idx].component_id ||
        !same_delta(lhs_entries[idx].offset, rhs_entries[idx].offset)) {
      return false;
    }
  }
  return true;
}

} // namespace titan_pbctopo
