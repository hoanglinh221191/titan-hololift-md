#ifndef TITAN_PBCTOPO_COMPONENT_SEARCH_PAIR_TABLE_H
#define TITAN_PBCTOPO_COMPONENT_SEARCH_PAIR_TABLE_H

#include "types.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace titan_pbctopo {

class PbctopoShiftMask {
public:
  PbctopoShiftMask() = default;
  explicit PbctopoShiftMask(std::size_t shift_count);

  static PbctopoShiftMask full(std::size_t shift_count);
  static PbctopoShiftMask only(std::size_t shift_count, std::size_t shift_idx);

  void reset(std::size_t shift_count);
  void set_all();
  void clear_all();
  void set_only(std::size_t shift_idx);
  void set(std::size_t shift_idx);
  void clear(std::size_t shift_idx);
  bool test(std::size_t shift_idx) const;
  bool empty() const;
  std::size_t count() const;
  bool intersect_with(const PbctopoShiftMask &other);
  std::size_t shift_count() const { return shift_count_; }
  std::size_t storage_bytes_estimate() const;

  template <typename Fn> void for_each_set_bit(Fn &&fn) const {
    const std::size_t word_count = word_count_for(shift_count_);
    for (std::size_t word_idx = 0; word_idx < word_count; ++word_idx) {
      std::uint64_t word = word_at(word_idx);
      while (word != 0ull) {
        const std::size_t bit =
            static_cast<std::size_t>(std::countr_zero(word));
        word &= (word - 1ull);
        const std::size_t shift_idx = word_idx * kBitsPerWord + bit;
        if (shift_idx < shift_count_)
          fn(shift_idx);
      }
    }
  }

private:
  static constexpr std::size_t kBitsPerWord = 64;
  static constexpr std::size_t kInlineWordCount = 2;
  static constexpr std::size_t kInlineShiftLimit =
      kInlineWordCount * kBitsPerWord;

  bool uses_inline_storage() const {
    return shift_count_ <= kInlineShiftLimit;
  }
  static std::size_t word_count_for(std::size_t shift_count);
  std::uint64_t word_at(std::size_t index) const;
  std::uint64_t &word_at(std::size_t index);
  void trim_unused_bits();

  std::size_t shift_count_ = 0;
  std::array<std::uint64_t, kInlineWordCount> inline_words_{};
  std::vector<std::uint64_t> words_;
};

struct PbctopoSparseShiftPair {
  std::size_t rhs_shift = 0;
  PbctopoComponentPairShiftEntry entry;
};

struct PbctopoInt3Hash {
  std::size_t operator()(const Int3 &value) const noexcept;
};

class PbctopoComponentPairShiftTable {
public:
  std::size_t component_count = 0;
  std::size_t shift_count = 0;

  void reset(std::size_t new_component_count, std::size_t new_shift_count);
  void reset(std::size_t new_component_count,
             const std::vector<PbctopoShift> &shifts);
  void store_relative_pair(std::size_t lhs_comp, std::size_t rhs_comp,
                           const Int3 &rhs_minus_lhs,
                           const PbctopoComponentPairShiftEntry &entry);
  void mark_relative_incompatible(std::size_t lhs_comp,
                                  std::size_t rhs_comp,
                                  const Int3 &rhs_minus_lhs);
  void store_directed_pair(std::size_t lhs_comp, std::size_t lhs_shift,
                           std::size_t rhs_comp, std::size_t rhs_shift,
                           const PbctopoComponentPairShiftEntry &entry);
  const PbctopoComponentPairShiftEntry &
  entry_or_default(std::size_t lhs_comp, std::size_t lhs_shift,
                   std::size_t rhs_comp, std::size_t rhs_shift) const;
  const PbctopoComponentPairShiftEntry &
  at(std::size_t lhs_comp, std::size_t lhs_shift, std::size_t rhs_comp,
     std::size_t rhs_shift) const {
    return entry_or_default(lhs_comp, lhs_shift, rhs_comp, rhs_shift);
  }

  PbctopoShiftMask compat_mask(std::size_t lhs_comp,
                               std::size_t lhs_shift,
                               std::size_t rhs_comp) const;
  void add_bbox_prefiltered(std::size_t count) { bbox_prefiltered_ += count; }

  std::size_t dense_entries_equivalent() const;
  std::size_t materialized_entries() const { return materialized_entries_; }
  std::size_t bbox_prefiltered() const { return bbox_prefiltered_; }
  std::size_t sparse_rows() const;
  std::size_t peak_bytes_estimate() const;
  const std::vector<std::size_t> &
  constrained_neighbors(std::size_t component) const;

private:
  struct RelativeCompatCache {
    std::vector<PbctopoShiftMask> by_lhs_shift;
    PbctopoShiftMask valid_lhs_shifts;

    void invalidate() { valid_lhs_shifts.clear_all(); }
  };

  struct Row {
    std::vector<std::vector<PbctopoSparseShiftPair>> by_lhs_shift;
    std::unordered_map<Int3, PbctopoComponentPairShiftEntry, PbctopoInt3Hash>
        by_relative_shift;
    std::vector<PbctopoShiftMask> compat_by_lhs_shift;
    mutable RelativeCompatCache relative_compat_lower_to_upper;
    mutable RelativeCompatCache relative_compat_upper_to_lower;
    std::size_t materialized_entries = 0;
    bool has_hard_clash = false;

    void invalidate_relative_compat_cache() {
      relative_compat_lower_to_upper.invalidate();
      relative_compat_upper_to_lower.invalidate();
    }
  };

  std::size_t row_index(std::size_t lhs_comp, std::size_t rhs_comp) const;
  Row &row(std::size_t lhs_comp, std::size_t rhs_comp);
  const Row &row(std::size_t lhs_comp, std::size_t rhs_comp) const;
  void register_constraint_pair(std::size_t lhs_comp, std::size_t rhs_comp);

  std::vector<Row> rows_;
  std::unordered_map<std::size_t, Row> relative_rows_;
  std::vector<std::vector<std::size_t>> constrained_neighbors_;
  Row empty_row_;
  std::vector<Int3> shifts_;
  bool relative_mode_ = false;
  PbctopoComponentPairShiftEntry default_entry_;
  PbctopoShiftMask empty_mask_;
  std::size_t materialized_entries_ = 0;
  std::size_t bbox_prefiltered_ = 0;
};

bool pbctopo_enforce_pair_arc_consistency(
    const PbctopoComponentPairShiftTable &table,
    std::vector<PbctopoShiftMask> &domains, std::size_t *passes = nullptr,
    std::size_t *revisions = nullptr, std::size_t *removals = nullptr,
    std::size_t *queue_pushes = nullptr);

} // namespace titan_pbctopo

#endif
