#include "component_search_pair_table.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <unordered_set>
#include <utility>

namespace titan_pbctopo {

std::size_t PbctopoInt3Hash::operator()(const Int3 &value) const noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  auto mix = [&](std::int64_t coordinate) {
    const std::uint64_t word = static_cast<std::uint64_t>(coordinate);
    for (unsigned shift = 0; shift < 64; shift += 8) {
      hash ^= static_cast<std::uint8_t>(word >> shift);
      hash *= 1099511628211ull;
    }
  };
  mix(value.x);
  mix(value.y);
  mix(value.z);
  return static_cast<std::size_t>(hash);
}

PbctopoShiftMask::PbctopoShiftMask(std::size_t shift_count) {
  reset(shift_count);
}

PbctopoShiftMask PbctopoShiftMask::full(std::size_t shift_count) {
  PbctopoShiftMask mask(shift_count);
  mask.set_all();
  return mask;
}

PbctopoShiftMask PbctopoShiftMask::only(std::size_t shift_count,
                                        std::size_t shift_idx) {
  PbctopoShiftMask mask(shift_count);
  mask.set_only(shift_idx);
  return mask;
}

void PbctopoShiftMask::reset(std::size_t shift_count) {
  shift_count_ = shift_count;
  inline_words_.fill(0ull);
  if (uses_inline_storage()) {
    words_.clear();
  } else {
    words_.assign(word_count_for(shift_count_), 0ull);
  }
}

void PbctopoShiftMask::set_all() {
  if (shift_count_ == 0) {
    clear_all();
    return;
  }
  if (uses_inline_storage())
    inline_words_.fill(~0ull);
  else
    std::fill(words_.begin(), words_.end(), ~0ull);
  trim_unused_bits();
}

void PbctopoShiftMask::clear_all() {
  inline_words_.fill(0ull);
  std::fill(words_.begin(), words_.end(), 0ull);
}

void PbctopoShiftMask::set_only(std::size_t shift_idx) {
  clear_all();
  set(shift_idx);
}

void PbctopoShiftMask::set(std::size_t shift_idx) {
  if (shift_idx >= shift_count_)
    return;
  word_at(shift_idx / kBitsPerWord) |=
      (1ull << (shift_idx % kBitsPerWord));
}

void PbctopoShiftMask::clear(std::size_t shift_idx) {
  if (shift_idx >= shift_count_)
    return;
  word_at(shift_idx / kBitsPerWord) &=
      ~(1ull << (shift_idx % kBitsPerWord));
}

bool PbctopoShiftMask::test(std::size_t shift_idx) const {
  if (shift_idx >= shift_count_)
    return false;
  return (word_at(shift_idx / kBitsPerWord) &
          (1ull << (shift_idx % kBitsPerWord))) != 0ull;
}

bool PbctopoShiftMask::empty() const {
  const std::size_t word_count = word_count_for(shift_count_);
  for (std::size_t idx = 0; idx < word_count; ++idx) {
    if (word_at(idx) != 0ull)
      return false;
  }
  return true;
}

std::size_t PbctopoShiftMask::count() const {
  std::size_t total = 0;
  const std::size_t word_count = word_count_for(shift_count_);
  for (std::size_t idx = 0; idx < word_count; ++idx)
    total += static_cast<std::size_t>(std::popcount(word_at(idx)));
  return total;
}

bool PbctopoShiftMask::intersect_with(const PbctopoShiftMask &other) {
  bool changed = false;
  if (shift_count_ != other.shift_count_) {
    PbctopoShiftMask filtered(shift_count_);
    for_each_set_bit([&](std::size_t shift_idx) {
      if (other.test(shift_idx))
        filtered.set(shift_idx);
    });
    changed = filtered.count() != count();
    *this = std::move(filtered);
    return changed;
  }

  const std::size_t word_count = word_count_for(shift_count_);
  for (std::size_t idx = 0; idx < word_count; ++idx) {
    const std::uint64_t previous = word_at(idx);
    word_at(idx) &= other.word_at(idx);
    changed = changed || previous != word_at(idx);
  }
  return changed;
}

std::size_t PbctopoShiftMask::storage_bytes_estimate() const {
  return sizeof(PbctopoShiftMask) + words_.capacity() * sizeof(std::uint64_t);
}

std::size_t PbctopoShiftMask::word_count_for(std::size_t shift_count) {
  return (shift_count + kBitsPerWord - 1) / kBitsPerWord;
}

std::uint64_t PbctopoShiftMask::word_at(std::size_t index) const {
  return uses_inline_storage() ? inline_words_[index] : words_[index];
}

std::uint64_t &PbctopoShiftMask::word_at(std::size_t index) {
  return uses_inline_storage() ? inline_words_[index] : words_[index];
}

void PbctopoShiftMask::trim_unused_bits() {
  if (shift_count_ == 0)
    return;
  const std::size_t used_bits = shift_count_ % kBitsPerWord;
  if (used_bits == 0)
    return;
  const std::uint64_t mask = (1ull << used_bits) - 1ull;
  word_at(word_count_for(shift_count_) - 1) &= mask;
}

void PbctopoComponentPairShiftTable::reset(
    std::size_t new_component_count, std::size_t new_shift_count) {
  component_count = new_component_count;
  shift_count = new_shift_count;
  materialized_entries_ = 0;
  bbox_prefiltered_ = 0;
  relative_mode_ = false;
  shifts_.clear();
  empty_mask_.reset(shift_count);
  relative_rows_.clear();
  constrained_neighbors_.assign(component_count, {});
  rows_.clear();
  rows_.resize(component_count * component_count);
  for (Row &entry_row : rows_) {
    entry_row.by_lhs_shift.assign(shift_count, {});
    entry_row.compat_by_lhs_shift.clear();
    entry_row.compat_by_lhs_shift.reserve(shift_count);
    for (std::size_t shift_idx = 0; shift_idx < shift_count; ++shift_idx)
      entry_row.compat_by_lhs_shift.push_back(PbctopoShiftMask::full(shift_count));
    entry_row.by_relative_shift.clear();
    entry_row.materialized_entries = 0;
    entry_row.has_hard_clash = false;
  }
}

void PbctopoComponentPairShiftTable::reset(
    std::size_t new_component_count,
    const std::vector<PbctopoShift> &shifts) {
  component_count = new_component_count;
  shift_count = shifts.size();
  materialized_entries_ = 0;
  bbox_prefiltered_ = 0;
  relative_mode_ = true;
  shifts_.clear();
  shifts_.reserve(shifts.size());
  for (const auto &shift : shifts)
    shifts_.push_back({shift.ix, shift.iy, shift.iz});
  empty_mask_.reset(shift_count);
  rows_.clear();
  relative_rows_.clear();
  constrained_neighbors_.assign(component_count, {});
  empty_row_ = {};
}

void PbctopoComponentPairShiftTable::store_relative_pair(
    std::size_t lhs_comp, std::size_t rhs_comp, const Int3 &rhs_minus_lhs,
    const PbctopoComponentPairShiftEntry &entry) {
  if (!relative_mode_ || lhs_comp >= component_count ||
      rhs_comp >= component_count || lhs_comp == rhs_comp) {
    return;
  }
  Row &entry_row = row(lhs_comp, rhs_comp);
  Int3 canonical_delta = rhs_minus_lhs;
  if (lhs_comp > rhs_comp &&
      !checked_int3_negate(rhs_minus_lhs, canonical_delta)) {
    return;
  }
  const auto [position, inserted] =
      entry_row.by_relative_shift.insert_or_assign(canonical_delta, entry);
  (void)position;
  entry_row.invalidate_relative_compat_cache();
  if (inserted) {
    ++entry_row.materialized_entries;
    ++materialized_entries_;
  }
  if (entry.hard_clash && !entry_row.has_hard_clash) {
    entry_row.has_hard_clash = true;
    register_constraint_pair(lhs_comp, rhs_comp);
  }
}

void PbctopoComponentPairShiftTable::mark_relative_incompatible(
    std::size_t lhs_comp, std::size_t rhs_comp,
    const Int3 &rhs_minus_lhs) {
  if (!relative_mode_ || lhs_comp >= component_count ||
      rhs_comp >= component_count || lhs_comp == rhs_comp) {
    return;
  }
  Int3 canonical_delta = rhs_minus_lhs;
  if (lhs_comp > rhs_comp &&
      !checked_int3_negate(rhs_minus_lhs, canonical_delta)) {
    return;
  }
  Row &entry_row = row(lhs_comp, rhs_comp);
  PbctopoComponentPairShiftEntry entry;
  const auto existing = entry_row.by_relative_shift.find(canonical_delta);
  if (existing != entry_row.by_relative_shift.end())
    entry = existing->second;
  entry.hard_clash = true;
  store_relative_pair(lhs_comp, rhs_comp, rhs_minus_lhs, entry);
}

void PbctopoComponentPairShiftTable::store_directed_pair(
    std::size_t lhs_comp, std::size_t lhs_shift, std::size_t rhs_comp,
    std::size_t rhs_shift, const PbctopoComponentPairShiftEntry &entry) {
  if (lhs_comp >= component_count || rhs_comp >= component_count ||
      lhs_shift >= shift_count || rhs_shift >= shift_count) {
    return;
  }

  if (relative_mode_) {
    Int3 delta{};
    if (!checked_int3_subtract(shifts_[rhs_shift], shifts_[lhs_shift], delta))
      return;
    store_relative_pair(lhs_comp, rhs_comp, delta, entry);
    return;
  }

  Row &entry_row = row(lhs_comp, rhs_comp);
  auto &bucket = entry_row.by_lhs_shift[lhs_shift];
  for (PbctopoSparseShiftPair &sparse_pair : bucket) {
    if (sparse_pair.rhs_shift == rhs_shift) {
      sparse_pair.entry = entry;
      if (entry.hard_clash)
        entry_row.compat_by_lhs_shift[lhs_shift].clear(rhs_shift);
      else
        entry_row.compat_by_lhs_shift[lhs_shift].set(rhs_shift);
      if (entry.hard_clash && !entry_row.has_hard_clash) {
        entry_row.has_hard_clash = true;
        register_constraint_pair(lhs_comp, rhs_comp);
      }
      return;
    }
  }

  bucket.push_back({rhs_shift, entry});
  ++entry_row.materialized_entries;
  ++materialized_entries_;
  if (entry.hard_clash)
    entry_row.compat_by_lhs_shift[lhs_shift].clear(rhs_shift);
  if (entry.hard_clash && !entry_row.has_hard_clash) {
    entry_row.has_hard_clash = true;
    register_constraint_pair(lhs_comp, rhs_comp);
  }
}

const PbctopoComponentPairShiftEntry &
PbctopoComponentPairShiftTable::entry_or_default(
    std::size_t lhs_comp, std::size_t lhs_shift, std::size_t rhs_comp,
    std::size_t rhs_shift) const {
  if (lhs_comp >= component_count || rhs_comp >= component_count ||
      lhs_shift >= shift_count || rhs_shift >= shift_count ||
      lhs_comp == rhs_comp) {
    return default_entry_;
  }
  if (relative_mode_) {
    const Row &entry_row = row(lhs_comp, rhs_comp);
    Int3 delta{};
    if (!checked_int3_subtract(shifts_[rhs_shift], shifts_[lhs_shift], delta))
      return default_entry_;
    if (lhs_comp > rhs_comp) {
      Int3 reversed{};
      if (!checked_int3_negate(delta, reversed))
        return default_entry_;
      delta = reversed;
    }
    const auto found = entry_row.by_relative_shift.find(delta);
    return found == entry_row.by_relative_shift.end() ? default_entry_
                                                       : found->second;
  }
  const Row &entry_row = row(lhs_comp, rhs_comp);
  const auto &bucket = entry_row.by_lhs_shift[lhs_shift];
  for (const PbctopoSparseShiftPair &sparse_pair : bucket) {
    if (sparse_pair.rhs_shift == rhs_shift)
      return sparse_pair.entry;
  }
  return default_entry_;
}

PbctopoShiftMask PbctopoComponentPairShiftTable::compat_mask(
    std::size_t lhs_comp, std::size_t lhs_shift, std::size_t rhs_comp) const {
  if (lhs_comp >= component_count || rhs_comp >= component_count ||
      lhs_shift >= shift_count || lhs_comp == rhs_comp) {
    return empty_mask_;
  }
  if (relative_mode_) {
    const Row &entry_row = row(lhs_comp, rhs_comp);
    if (entry_row.materialized_entries == 0)
      return PbctopoShiftMask::full(shift_count);

    RelativeCompatCache &cache =
        lhs_comp < rhs_comp ? entry_row.relative_compat_lower_to_upper
                            : entry_row.relative_compat_upper_to_lower;
    if (cache.by_lhs_shift.size() != shift_count) {
      cache.by_lhs_shift.resize(shift_count);
      cache.valid_lhs_shifts.reset(shift_count);
    }
    if (cache.valid_lhs_shifts.test(lhs_shift))
      return cache.by_lhs_shift[lhs_shift];

    PbctopoShiftMask mask = PbctopoShiftMask::full(shift_count);
    for (std::size_t rhs_shift = 0; rhs_shift < shift_count; ++rhs_shift) {
      Int3 delta{};
      if (!checked_int3_subtract(shifts_[rhs_shift], shifts_[lhs_shift],
                                 delta)) {
        mask.clear(rhs_shift);
        continue;
      }
      if (lhs_comp > rhs_comp) {
        Int3 reversed{};
        if (!checked_int3_negate(delta, reversed)) {
          mask.clear(rhs_shift);
          continue;
        }
        delta = reversed;
      }
      const auto found = entry_row.by_relative_shift.find(delta);
      if (found != entry_row.by_relative_shift.end() &&
          found->second.hard_clash) {
        mask.clear(rhs_shift);
      }
    }
    cache.by_lhs_shift[lhs_shift] = std::move(mask);
    cache.valid_lhs_shifts.set(lhs_shift);
    return cache.by_lhs_shift[lhs_shift];
  }
  return row(lhs_comp, rhs_comp).compat_by_lhs_shift[lhs_shift];
}

std::size_t PbctopoComponentPairShiftTable::dense_entries_equivalent() const {
  std::size_t result = 1;
  for (const std::size_t factor :
       {component_count, component_count, shift_count, shift_count}) {
    if (factor != 0 && result > std::numeric_limits<std::size_t>::max() / factor)
      return std::numeric_limits<std::size_t>::max();
    result *= factor;
  }
  return result;
}

std::size_t PbctopoComponentPairShiftTable::sparse_rows() const {
  if (relative_mode_)
    return relative_rows_.size();
  std::size_t non_empty = 0;
  for (const Row &entry_row : rows_) {
    if (entry_row.materialized_entries > 0)
      ++non_empty;
  }
  return non_empty;
}

std::size_t PbctopoComponentPairShiftTable::peak_bytes_estimate() const {
  std::size_t bytes = sizeof(PbctopoComponentPairShiftTable);
  bytes += rows_.capacity() * sizeof(Row);
  bytes += relative_rows_.size() * (sizeof(std::size_t) + sizeof(Row) +
                                    2 * sizeof(void *));
  bytes += relative_rows_.bucket_count() * sizeof(void *);
  bytes += shifts_.capacity() * sizeof(Int3);
  bytes += constrained_neighbors_.capacity() * sizeof(std::vector<std::size_t>);
  for (const auto &neighbors : constrained_neighbors_)
    bytes += neighbors.capacity() * sizeof(std::size_t);
  auto add_row_bytes = [&](const Row &entry_row) {
    bytes += entry_row.by_lhs_shift.capacity() *
             sizeof(std::vector<PbctopoSparseShiftPair>);
    bytes += entry_row.compat_by_lhs_shift.capacity() *
             sizeof(PbctopoShiftMask);
    bytes += entry_row.by_relative_shift.size() *
             (sizeof(Int3) + sizeof(PbctopoComponentPairShiftEntry) +
              2 * sizeof(void *));
    bytes += entry_row.by_relative_shift.bucket_count() * sizeof(void *);
    for (const auto &bucket : entry_row.by_lhs_shift)
      bytes += bucket.capacity() * sizeof(PbctopoSparseShiftPair);
    for (const PbctopoShiftMask &mask : entry_row.compat_by_lhs_shift)
      bytes += mask.storage_bytes_estimate();
    auto add_relative_cache_bytes = [&](const RelativeCompatCache &cache) {
      bytes += cache.by_lhs_shift.capacity() * sizeof(PbctopoShiftMask);
      for (const PbctopoShiftMask &mask : cache.by_lhs_shift) {
        bytes += mask.storage_bytes_estimate() - sizeof(PbctopoShiftMask);
      }
      bytes += cache.valid_lhs_shifts.storage_bytes_estimate() -
               sizeof(PbctopoShiftMask);
    };
    add_relative_cache_bytes(entry_row.relative_compat_lower_to_upper);
    add_relative_cache_bytes(entry_row.relative_compat_upper_to_lower);
  };
  for (const Row &entry_row : rows_)
    add_row_bytes(entry_row);
  for (const auto &[row_key, entry_row] : relative_rows_) {
    (void)row_key;
    add_row_bytes(entry_row);
  }
  return bytes;
}

const std::vector<std::size_t> &
PbctopoComponentPairShiftTable::constrained_neighbors(
    std::size_t component) const {
  static const std::vector<std::size_t> empty;
  return component < constrained_neighbors_.size()
             ? constrained_neighbors_[component]
             : empty;
}

std::size_t PbctopoComponentPairShiftTable::row_index(
    std::size_t lhs_comp, std::size_t rhs_comp) const {
  if (relative_mode_) {
    const std::size_t lo = std::min(lhs_comp, rhs_comp);
    const std::size_t hi = std::max(lhs_comp, rhs_comp);
    return lo * (2 * component_count - lo - 1) / 2 + (hi - lo - 1);
  }
  return lhs_comp * component_count + rhs_comp;
}

PbctopoComponentPairShiftTable::Row &
PbctopoComponentPairShiftTable::row(std::size_t lhs_comp,
                                    std::size_t rhs_comp) {
  if (relative_mode_)
    return relative_rows_[row_index(lhs_comp, rhs_comp)];
  return rows_[row_index(lhs_comp, rhs_comp)];
}

const PbctopoComponentPairShiftTable::Row &
PbctopoComponentPairShiftTable::row(std::size_t lhs_comp,
                                    std::size_t rhs_comp) const {
  if (relative_mode_) {
    const auto found = relative_rows_.find(row_index(lhs_comp, rhs_comp));
    return found == relative_rows_.end() ? empty_row_ : found->second;
  }
  return rows_[row_index(lhs_comp, rhs_comp)];
}

void PbctopoComponentPairShiftTable::register_constraint_pair(
    std::size_t lhs_comp, std::size_t rhs_comp) {
  if (lhs_comp >= component_count || rhs_comp >= component_count ||
      lhs_comp == rhs_comp)
    return;
  auto add_neighbor = [&](std::size_t component, std::size_t neighbor) {
    auto &neighbors = constrained_neighbors_[component];
    if (std::find(neighbors.begin(), neighbors.end(), neighbor) ==
        neighbors.end()) {
      neighbors.push_back(neighbor);
    }
  };
  add_neighbor(lhs_comp, rhs_comp);
  add_neighbor(rhs_comp, lhs_comp);
}

bool pbctopo_enforce_pair_arc_consistency(
    const PbctopoComponentPairShiftTable &table,
    std::vector<PbctopoShiftMask> &domains, std::size_t *passes,
    std::size_t *revisions, std::size_t *removals,
    std::size_t *queue_pushes) {
  if (domains.size() != table.component_count)
    return false;
  for (const auto &domain : domains) {
    if (domain.shift_count() != table.shift_count || domain.empty())
      return false;
  }

  if (passes != nullptr)
    ++*passes;
  using Arc = std::pair<std::size_t, std::size_t>;
  std::deque<Arc> queue;
  std::unordered_set<std::size_t> queued;
  auto enqueue = [&](std::size_t lhs, std::size_t rhs) {
    if (lhs == rhs)
      return;
    const std::size_t key = lhs * table.component_count + rhs;
    if (!queued.insert(key).second)
      return;
    queue.emplace_back(lhs, rhs);
    if (queue_pushes != nullptr)
      ++*queue_pushes;
  };
  for (std::size_t lhs = 0; lhs < table.component_count; ++lhs)
    for (std::size_t rhs : table.constrained_neighbors(lhs))
      enqueue(lhs, rhs);

  std::vector<std::size_t> unsupported;
  unsupported.reserve(table.shift_count);
  while (!queue.empty()) {
    const auto [lhs_comp, rhs_comp] = queue.front();
    queue.pop_front();
    queued.erase(lhs_comp * table.component_count + rhs_comp);
    if (revisions != nullptr)
      ++*revisions;
    unsupported.clear();
    domains[lhs_comp].for_each_set_bit([&](std::size_t lhs_shift) {
      PbctopoShiftMask supported = domains[rhs_comp];
      supported.intersect_with(
          table.compat_mask(lhs_comp, lhs_shift, rhs_comp));
      if (supported.empty())
        unsupported.push_back(lhs_shift);
    });
    if (unsupported.empty())
      continue;
    for (std::size_t lhs_shift : unsupported)
      domains[lhs_comp].clear(lhs_shift);
    if (removals != nullptr)
      *removals += unsupported.size();
    if (domains[lhs_comp].empty())
      return false;
    for (std::size_t neighbor : table.constrained_neighbors(lhs_comp)) {
      if (neighbor != rhs_comp)
        enqueue(neighbor, lhs_comp);
    }
  }
  return true;
}

} // namespace titan_pbctopo
