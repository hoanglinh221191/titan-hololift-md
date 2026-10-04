#include "evidence_audit.h"

#include "titan_sha256.h"

#include <algorithm>
#include <limits>
#include <tuple>

namespace titan_pbctopo {

std::expected<PbctopoHash128, std::string>
hash_pbctopo_canonical_atom_pairs(
    std::span<const PbctopoCanonicalAtomPair> pairs) {
  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    if (pairs[idx].a >= pairs[idx].b)
      return std::unexpected("contact pair is not canonical");
    if (idx != 0 && !(pairs[idx - 1] < pairs[idx]))
      return std::unexpected("contact pairs are not strictly sorted");
  }
  titan_hash::Sha256Builder hash;
  hash.update(PBCTOPO_EOBS_PAIR_HASH_DOMAIN);
  hash.word(static_cast<std::uint64_t>(pairs.size()));
  for (const auto &pair : pairs) {
    hash.word(pair.a);
    hash.word(pair.b);
  }
  const auto digest = hash.finish128();
  return PbctopoHash128{digest[0], digest[1]};
}

namespace {
void hash_i32(titan_hash::Sha256Builder &hash, std::int64_t value) {
  const auto bits = static_cast<std::uint32_t>(static_cast<std::int32_t>(value));
  for (unsigned shift = 0; shift < 32; shift += 8)
    hash.byte(static_cast<std::uint8_t>(bits >> shift));
}
}

std::expected<PbctopoHash128, std::string> hash_pbctopo_contact_hypotheses(
    std::span<const PbctopoContactShiftHypothesis> hypotheses) {
  titan_hash::Sha256Builder hash;
  hash.update(PBCTOPO_EOBS_HYPOTHESIS_HASH_DOMAIN);
  hash.word(static_cast<std::uint64_t>(hypotheses.size()));
  const PbctopoContactShiftHypothesis *previous_hypothesis = nullptr;
  for (const auto &hypothesis : hypotheses) {
    if (hypothesis.component_id_a >= hypothesis.component_id_b)
      return std::unexpected("contact hypothesis component IDs are not canonical");
    if (previous_hypothesis != nullptr) {
      const auto previous_key = std::tie(
          previous_hypothesis->component_id_a,
          previous_hypothesis->component_id_b,
          previous_hypothesis->relative_shift.x,
          previous_hypothesis->relative_shift.y,
          previous_hypothesis->relative_shift.z);
      const auto current_key = std::tie(
          hypothesis.component_id_a, hypothesis.component_id_b,
          hypothesis.relative_shift.x, hypothesis.relative_shift.y,
          hypothesis.relative_shift.z);
      if (!(previous_key < current_key))
        return std::unexpected("contact hypotheses are not strictly sorted");
    }
    for (const auto value : {hypothesis.relative_shift.x,
                             hypothesis.relative_shift.y,
                             hypothesis.relative_shift.z}) {
      if (value < std::numeric_limits<std::int32_t>::min() ||
          value > std::numeric_limits<std::int32_t>::max())
        return std::unexpected("contact hypothesis shift exceeds int32");
    }
    hash.word(hypothesis.component_id_a);
    hash.word(hypothesis.component_id_b);
    hash_i32(hash, hypothesis.relative_shift.x);
    hash_i32(hash, hypothesis.relative_shift.y);
    hash_i32(hash, hypothesis.relative_shift.z);
    hash.word(static_cast<std::uint64_t>(hypothesis.edges.size()));
    PbctopoCanonicalAtomPair previous{};
    bool has_previous = false;
    for (const auto &edge : hypothesis.edges) {
      if (edge.a == edge.b)
        return std::unexpected("contact hypothesis contains a self edge");
      const PbctopoCanonicalAtomPair pair{
          static_cast<std::uint64_t>(std::min(edge.a, edge.b)),
          static_cast<std::uint64_t>(std::max(edge.a, edge.b))};
      if (has_previous && !(previous < pair))
        return std::unexpected("contact hypothesis pairs are not sorted");
      hash.word(pair.a);
      hash.word(pair.b);
      previous = pair;
      has_previous = true;
    }
    previous_hypothesis = &hypothesis;
  }
  const auto digest = hash.finish128();
  return PbctopoHash128{digest[0], digest[1]};
}

PbctopoContactHypothesisSet
build_pbctopo_contact_hypotheses(
    std::span<const PbctopoAtom> atoms,
    std::span<const PbctopoAnchorContactEdge> observed_edges,
    std::span<const std::size_t> atom_component_index,
    std::span<const Int3> spatial_lift_atom_images,
    const PbctopoImageAssignment &identity_assignment) {
  PbctopoContactHypothesisSet result;
  result.attempted = true;
  if (!identity_assignment.valid || atoms.size() != atom_component_index.size() ||
      atoms.size() != spatial_lift_atom_images.size() ||
      identity_assignment.component_ids.size() !=
          identity_assignment.component_offsets.size()) {
    result.error = "contact hypothesis input mismatch";
    ++result.component_mapping_failures;
    return result;
  }
  std::vector<PbctopoCanonicalAtomPair> raw_pairs;
  raw_pairs.reserve(observed_edges.size());
  for (const auto &edge : observed_edges) {
    if (edge.a >= atoms.size() || edge.b >= atoms.size() || edge.a == edge.b) {
      result.error = "contact hypothesis contains an invalid raw edge";
      ++result.component_mapping_failures;
      return result;
    }
    const auto ga = static_cast<std::uint64_t>(atoms[edge.a].global_atom_id);
    const auto gb = static_cast<std::uint64_t>(atoms[edge.b].global_atom_id);
    raw_pairs.push_back({std::min(ga, gb), std::max(ga, gb)});
  }
  std::sort(raw_pairs.begin(), raw_pairs.end());
  auto raw_hash = hash_pbctopo_canonical_atom_pairs(raw_pairs);
  if (!raw_hash) {
    result.error = raw_hash.error();
    ++result.component_mapping_failures;
    return result;
  }
  result.raw_observed_pair_hash = *raw_hash;
  result.raw_observed_pair_count = raw_pairs.size();
  for (const auto &edge : observed_edges) {
    if (edge.a >= atoms.size() || edge.b >= atoms.size() || edge.a == edge.b ||
        !edge.nearest_image_valid || edge.equivalent_nearest_images != 1) {
      ++result.mic_ambiguity_failures;
      result.error = "contact hypothesis requires one exact nearest image";
      return result;
    }
    std::size_t ca = atom_component_index[edge.a];
    std::size_t cb = atom_component_index[edge.b];
    if (ca >= identity_assignment.component_ids.size() ||
        cb >= identity_assignment.component_ids.size())
    {
      ++result.component_mapping_failures;
      result.error = "contact hypothesis has invalid component mapping";
      return result;
    }
    if (ca == cb) {
      ++result.internal_edges_ignored;
      continue;
    }
    Int3 lift_delta{};
    Int3 relative{};
    if (!checked_int3_subtract(spatial_lift_atom_images[edge.b],
                               spatial_lift_atom_images[edge.a], lift_delta) ||
        !checked_int3_subtract(edge.nearest_image, lift_delta, relative))
    {
      result.error = "contact hypothesis relative shift overflow";
      return result;
    }
    std::uint64_t id_a = identity_assignment.component_ids[ca];
    std::uint64_t id_b = identity_assignment.component_ids[cb];
    std::size_t atom_a = edge.a;
    std::size_t atom_b = edge.b;
    if (id_b < id_a) {
      std::swap(id_a, id_b);
      std::swap(ca, cb);
      std::swap(atom_a, atom_b);
      Int3 negated{};
      if (!checked_int3_negate(relative, negated))
      {
        result.error = "contact hypothesis shift negate overflow";
        return result;
      }
      relative = negated;
    }
    auto it = std::find_if(result.hypotheses.begin(), result.hypotheses.end(),
      [&](const PbctopoContactShiftHypothesis &h) {
        return h.component_id_a == id_a && h.component_id_b == id_b &&
               same_delta(h.relative_shift, relative);
      });
    if (it == result.hypotheses.end()) {
      result.hypotheses.push_back({ca, cb, id_a, id_b, relative, {}});
      it = std::prev(result.hypotheses.end());
    }
    it->edges.push_back({atom_a, atom_b, edge.cutoff_d2});
  }
  std::sort(result.hypotheses.begin(), result.hypotheses.end(),
    [](const auto &a, const auto &b) {
      return std::tie(a.component_id_a, a.component_id_b, a.relative_shift.x,
                      a.relative_shift.y, a.relative_shift.z) <
             std::tie(b.component_id_a, b.component_id_b, b.relative_shift.x,
                      b.relative_shift.y, b.relative_shift.z);
    });
  for (auto &hypothesis : result.hypotheses) {
    std::sort(hypothesis.edges.begin(), hypothesis.edges.end(),
      [&](const auto &a, const auto &b) {
        const auto pa = std::minmax(atoms[a.a].global_atom_id,
                                    atoms[a.b].global_atom_id);
        const auto pb = std::minmax(atoms[b.a].global_atom_id,
                                    atoms[b.b].global_atom_id);
        return pa < pb;
      });
  }
  std::size_t begin = 0;
  while (begin < result.hypotheses.size()) {
    std::size_t end = begin + 1;
    while (end < result.hypotheses.size() &&
           result.hypotheses[end].component_id_a == result.hypotheses[begin].component_id_a &&
           result.hypotheses[end].component_id_b == result.hypotheses[begin].component_id_b)
      ++end;
    ++result.component_relation_count;
    if (end - begin > 1) ++result.ambiguous_component_relation_count;
    begin = end;
  }
  // Hashing uses global atom IDs, not local vector indices.
  auto hashable = result.hypotheses;
  for (auto &hypothesis : hashable) {
    for (auto &edge : hypothesis.edges) {
      edge.a = atoms[edge.a].global_atom_id;
      edge.b = atoms[edge.b].global_atom_id;
      if (edge.b < edge.a) std::swap(edge.a, edge.b);
    }
    std::sort(hypothesis.edges.begin(), hypothesis.edges.end(),
              [](const auto &a, const auto &b) {
                return std::tie(a.a, a.b) < std::tie(b.a, b.b);
              });
  }
  auto digest = hash_pbctopo_contact_hypotheses(hashable);
  if (!digest) {
    result.error = digest.error();
    return result;
  }
  result.all_hypotheses_hash = *digest;
  result.complete = true;
  return result;
}

} // namespace titan_pbctopo
