#include "observation_binding.h"

#include "observation_schema.h"
#include "../titan_sha256.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>
#include <vector>

namespace titan_pbctopo {
namespace {

constexpr std::uint64_t kBoxHashDomain = 0x504243424f583031ULL;
constexpr std::uint64_t kSelectionHashDomain = 0x50424353454c3031ULL;
constexpr std::uint64_t kCoordinateHashDomain = 0x5042434352443031ULL;

class Hash128Builder {
public:
  explicit Hash128Builder(std::uint64_t domain) noexcept { hash_.word(domain); }

  void word(std::uint64_t value) noexcept { hash_.word(value); }

  void real(double value) noexcept { hash_.real(value); }

  [[nodiscard]] PbctopoHash128 finish() const noexcept {
    const auto digest = hash_.finish128();
    return {digest[0], digest[1]};
  }

private:
  titan_hash::Sha256Builder hash_;
};

bool finite_box(std::span<const double, 9> matrix) noexcept {
  for (double value : matrix) {
    if (!std::isfinite(value))
      return false;
  }
  const double determinant =
      matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
      matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
      matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
  return std::isfinite(determinant) && std::fabs(determinant) > 1.0e-18;
}

} // namespace

PbctopoHash128
hash_pbctopo_box_matrix(std::span<const double, 9> box_matrix) noexcept {
  Hash128Builder hash(kBoxHashDomain);
  hash.word(box_matrix.size());
  for (double value : box_matrix)
    hash.real(value);
  return hash.finish();
}

PbctopoHash128 hash_pbctopo_canonical_atom_universe(
    std::span<const PbctopoAtom> atoms) {
  if (atoms.empty())
    return {};
  std::vector<std::size_t> order(atoms.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
    return std::tie(atoms[lhs].owner, atoms[lhs].source_atom_id) <
           std::tie(atoms[rhs].owner, atoms[rhs].source_atom_id);
  });
  Hash128Builder hash(VIBE_OBSERVATION_ATOM_UNIVERSE_HASH_DOMAIN);
  hash.word(order.size());
  for (std::size_t dense_idx = 0; dense_idx < order.size(); ++dense_idx) {
    const auto &atom = atoms[order[dense_idx]];
    if (dense_idx != 0) {
      const auto &previous = atoms[order[dense_idx - 1]];
      if (previous.owner == atom.owner &&
          previous.source_atom_id == atom.source_atom_id) {
        return {};
      }
    }
    hash.word(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(atom.owner)));
    hash.word(atom.source_atom_id);
  }
  return hash.finish();
}

PbctopoFrameBindingObservation build_pbctopo_frame_binding_observation(
    std::size_t trajectory_frame_index, box Box,
    std::span<const PbctopoAtom> wrapped_atoms) {
  PbctopoFrameBindingObservation result;
  result.trajectory_frame_index = trajectory_frame_index;
  if (wrapped_atoms.empty())
    return result;

  double v1[3]{};
  double v2[3]{};
  double v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  result.box_matrix = {v1[0], v1[1], v1[2], v2[0], v2[1],
                       v2[2], v3[0], v3[1], v3[2]};
  if (!finite_box(result.box_matrix))
    return result;

  Hash128Builder selection_hash(kSelectionHashDomain);
  Hash128Builder coordinate_hash(kCoordinateHashDomain);
  selection_hash.word(wrapped_atoms.size());
  coordinate_hash.word(wrapped_atoms.size());
  for (const auto &atom : wrapped_atoms) {
    if (!std::isfinite(atom.x) || !std::isfinite(atom.y) ||
        !std::isfinite(atom.z)) {
      return {};
    }
    const auto owner = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(atom.owner));
    selection_hash.word(owner);
    selection_hash.word(atom.source_atom_id);
    coordinate_hash.real(atom.x);
    coordinate_hash.real(atom.y);
    coordinate_hash.real(atom.z);
  }
  result.box_hash = hash_pbctopo_box_matrix(result.box_matrix);
  result.atom_selection_hash = selection_hash.finish();
  result.canonical_atom_universe_hash =
      hash_pbctopo_canonical_atom_universe(wrapped_atoms);
  result.wrapped_coordinate_hash = coordinate_hash.finish();
  result.valid = !result.box_hash.empty() &&
                  !result.atom_selection_hash.empty() &&
                  !result.canonical_atom_universe_hash.empty() &&
                  !result.wrapped_coordinate_hash.empty();
  return result;
}

std::string format_pbctopo_hash128(PbctopoHash128 hash) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << hash.hi
      << std::setw(16) << hash.lo;
  return out.str();
}

} // namespace titan_pbctopo
