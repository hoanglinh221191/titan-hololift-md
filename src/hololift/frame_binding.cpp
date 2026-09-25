#include "frame_binding.h"

#include "../pbctopo/observation_schema.h"
#include "../titan_sha256.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <system_error>
#include <tuple>
#include <vector>

namespace titan_hololift {
namespace {

constexpr std::uint64_t kBoxHashDomain = 0x504243424f583031ULL;
constexpr std::uint64_t kSelectionHashDomain = 0x50424353454c3031ULL;
constexpr std::uint64_t kCoordinateHashDomain = 0x5042434352443031ULL;

class Hash128Builder {
public:
  explicit Hash128Builder(std::uint64_t domain) noexcept { hash_.word(domain); }

  void word(std::uint64_t value) noexcept { hash_.word(value); }

  void real(double value) noexcept { hash_.real(value); }

  [[nodiscard]] HoloLiftHash128 finish() const noexcept {
    const auto digest = hash_.finish128();
    return {digest[0], digest[1]};
  }

private:
  titan_hash::Sha256Builder hash_;
};

} // namespace

HoloLiftHash128
hash_hololift_box_matrix(std::span<const double, 9> box_matrix) noexcept {
  Hash128Builder hash(kBoxHashDomain);
  hash.word(box_matrix.size());
  for (double value : box_matrix)
    hash.real(value);
  return hash.finish();
}

HoloLiftHash128 hash_hololift_atom_selection(
    std::span<const HoloLiftSourceAtomKey> atom_order) noexcept {
  Hash128Builder hash(kSelectionHashDomain);
  hash.word(atom_order.size());
  for (const auto &atom : atom_order) {
    hash.word(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(atom.owner)));
    hash.word(atom.source_atom_id);
  }
  return hash.finish();
}

std::expected<HoloLiftHash128, std::string>
hash_hololift_canonical_atom_universe(
    std::span<const HoloLiftSourceAtomKey> atom_order) {
  if (atom_order.empty())
    return std::unexpected("HoloLift atom universe is empty");
  std::vector<std::size_t> order(atom_order.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
    return std::tie(atom_order[lhs].owner, atom_order[lhs].source_atom_id) <
           std::tie(atom_order[rhs].owner, atom_order[rhs].source_atom_id);
  });
  Hash128Builder hash(
      titan_pbctopo::VIBE_OBSERVATION_ATOM_UNIVERSE_HASH_DOMAIN);
  hash.word(order.size());
  for (std::size_t dense_idx = 0; dense_idx < order.size(); ++dense_idx) {
    const auto &atom = atom_order[order[dense_idx]];
    if (dense_idx != 0 && atom == atom_order[order[dense_idx - 1]]) {
      return std::unexpected("HoloLift atom universe contains duplicates");
    }
    hash.word(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(atom.owner)));
    hash.word(atom.source_atom_id);
  }
  return hash.finish();
}

HoloLiftHash128 hash_hololift_wrapped_coordinates(
    std::span<const std::array<double, 3>> coordinates) noexcept {
  Hash128Builder hash(kCoordinateHashDomain);
  hash.word(coordinates.size());
  for (const auto &coordinate : coordinates) {
    hash.real(coordinate[0]);
    hash.real(coordinate[1]);
    hash.real(coordinate[2]);
  }
  return hash.finish();
}

std::expected<void, std::string> validate_hololift_frame_binding(
    const HoloLiftTopologyEpochRecord &epoch,
    const HoloLiftFrameRecord &observation,
    std::size_t trajectory_frame_index,
    std::span<const double, 9> box_matrix,
    std::span<const HoloLiftSourceAtomKey> atom_order,
    std::span<const std::array<double, 3>> wrapped_coordinates) {
  if (!observation.frame_binding_valid)
    return std::unexpected("HoloLift observation has no valid frame binding");
  if (trajectory_frame_index != observation.trajectory_frame_index) {
    return std::unexpected(
        "HoloLift trajectory-frame index does not match the observation");
  }
  if (atom_order.empty() || atom_order.size() != wrapped_coordinates.size()) {
    return std::unexpected(
        "HoloLift atom selection and wrapped coordinates do not align");
  }
  for (double value : box_matrix) {
    if (!std::isfinite(value))
      return std::unexpected("HoloLift trajectory box is not finite");
  }
  for (const auto &coordinate : wrapped_coordinates) {
    if (!std::isfinite(coordinate[0]) || !std::isfinite(coordinate[1]) ||
        !std::isfinite(coordinate[2])) {
      return std::unexpected(
          "HoloLift wrapped trajectory coordinates are not finite");
    }
  }
  if (hash_hololift_box_matrix(box_matrix) != observation.box_hash)
    return std::unexpected("HoloLift trajectory box hash mismatch");
  if (hash_hololift_atom_selection(atom_order) !=
      observation.atom_selection_hash) {
    return std::unexpected("HoloLift trajectory atom-selection hash mismatch");
  }
  const auto atom_universe_hash =
      hash_hololift_canonical_atom_universe(atom_order);
  if (!atom_universe_hash)
    return std::unexpected(atom_universe_hash.error());
  if (*atom_universe_hash != observation.canonical_atom_universe_hash ||
      *atom_universe_hash != epoch.canonical_atom_universe_hash) {
    return std::unexpected(
        "HoloLift trajectory atom universe does not match its topology epoch");
  }
  if (hash_hololift_wrapped_coordinates(wrapped_coordinates) !=
      observation.wrapped_coordinate_hash) {
    return std::unexpected(
        "HoloLift wrapped-coordinate hash mismatch");
  }
  return {};
}

std::string format_hololift_hash128(HoloLiftHash128 hash) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << hash.hi
      << std::setw(16) << hash.lo;
  return out.str();
}

std::expected<HoloLiftHash128, std::string>
parse_hololift_hash128(std::string_view value) {
  if (value.size() != 32)
    return std::unexpected("HoloLift hash must contain 32 hexadecimal digits");
  HoloLiftHash128 result;
  const auto hi = std::from_chars(value.data(), value.data() + 16, result.hi,
                                  16);
  const auto lo = std::from_chars(value.data() + 16, value.data() + 32,
                                  result.lo, 16);
  if (hi.ec != std::errc{} || hi.ptr != value.data() + 16 ||
      lo.ec != std::errc{} || lo.ptr != value.data() + 32 || result.empty()) {
    return std::unexpected("invalid or empty HoloLift 128-bit hash");
  }
  return result;
}

} // namespace titan_hololift
