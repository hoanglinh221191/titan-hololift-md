#include "tabi_ses_bilevel_grid.h"

#include <algorithm>
#include <stdexcept>

namespace TitanPBE::Tabi::analytic_ses {

BilevelBitGrid::BilevelBitGrid(std::size_t nx, std::size_t ny, std::size_t nz,
                               bool initial_value)
    : nx_(nx), ny_(ny), nz_(nz), nbx_((nx + kBlock - 1U) / kBlock),
      nby_((ny + kBlock - 1U) / kBlock),
      nbz_((nz + kBlock - 1U) / kBlock),
      tags_(nbx_ * nby_ * nbz_,
            initial_value ? Tag::all_one : Tag::all_zero),
      fine_indices_(tags_.size(), kNoWord) {}

std::size_t BilevelBitGrid::block_index(std::size_t x, std::size_t y,
                                        std::size_t z) const noexcept {
  return x / kBlock + nbx_ * (y / kBlock + nby_ * (z / kBlock));
}

std::uint64_t BilevelBitGrid::bit_mask(std::size_t x, std::size_t y,
                                      std::size_t z) noexcept {
  const std::size_t bit = (x % kBlock) +
                          kBlock * ((y % kBlock) + kBlock * (z % kBlock));
  return std::uint64_t{1} << bit;
}

std::uint64_t BilevelBitGrid::valid_mask(std::size_t bx, std::size_t by,
                                        std::size_t bz) const noexcept {
  std::uint64_t result = 0;
  const std::size_t x_count = std::min(kBlock, nx_ - bx * kBlock);
  const std::size_t y_count = std::min(kBlock, ny_ - by * kBlock);
  const std::size_t z_count = std::min(kBlock, nz_ - bz * kBlock);
  for (std::size_t z = 0; z < z_count; ++z)
    for (std::size_t y = 0; y < y_count; ++y)
      for (std::size_t x = 0; x < x_count; ++x)
        result |= std::uint64_t{1} << (x + kBlock * (y + kBlock * z));
  return result;
}

std::uint32_t BilevelBitGrid::allocate_word(std::uint64_t initial) {
  if (!free_words_.empty()) {
    const std::uint32_t index = free_words_.back();
    free_words_.pop_back();
    mixed_words_[index] = initial;
    return index;
  }
  if (mixed_words_.size() >= kNoWord)
    throw std::overflow_error("BilevelBitGrid fine-word index overflow");
  mixed_words_.push_back(initial);
  return static_cast<std::uint32_t>(mixed_words_.size() - 1U);
}

void BilevelBitGrid::release_word(std::uint32_t word) {
  if (word != kNoWord)
    free_words_.push_back(word);
}

bool BilevelBitGrid::get(std::size_t x, std::size_t y,
                         std::size_t z) const noexcept {
  if (x >= nx_ || y >= ny_ || z >= nz_)
    return false;
  const std::size_t block = block_index(x, y, z);
  if (tags_[block] == Tag::all_zero)
    return false;
  if (tags_[block] == Tag::all_one)
    return true;
  return (mixed_words_[fine_indices_[block]] & bit_mask(x, y, z)) != 0U;
}

void BilevelBitGrid::set(std::size_t x, std::size_t y, std::size_t z,
                         bool value) {
  if (x >= nx_ || y >= ny_ || z >= nz_)
    throw std::out_of_range("BilevelBitGrid coordinate out of range");
  const std::size_t block = block_index(x, y, z);
  const Tag desired = value ? Tag::all_one : Tag::all_zero;
  if (tags_[block] == desired)
    return;
  if (tags_[block] != Tag::mixed) {
    const std::uint64_t initial =
        tags_[block] == Tag::all_one ? ~std::uint64_t{0} : std::uint64_t{0};
    fine_indices_[block] = allocate_word(initial);
    tags_[block] = Tag::mixed;
  }
  std::uint64_t &word = mixed_words_[fine_indices_[block]];
  const std::uint64_t mask = bit_mask(x, y, z);
  if (value)
    word |= mask;
  else
    word &= ~mask;
}

void BilevelBitGrid::compact() {
  for (std::size_t bz = 0; bz < nbz_; ++bz) {
    for (std::size_t by = 0; by < nby_; ++by) {
      for (std::size_t bx = 0; bx < nbx_; ++bx) {
        const std::size_t block = bx + nbx_ * (by + nby_ * bz);
        if (tags_[block] != Tag::mixed)
          continue;
        const std::uint32_t word_index = fine_indices_[block];
        const std::uint64_t word = mixed_words_[word_index];
        const std::uint64_t valid = valid_mask(bx, by, bz);
        if ((word & valid) == 0U) {
          tags_[block] = Tag::all_zero;
        } else if ((word & valid) == valid) {
          tags_[block] = Tag::all_one;
        } else {
          continue;
        }
        fine_indices_[block] = kNoWord;
        release_word(word_index);
      }
    }
  }
}

std::size_t BilevelBitGrid::retained_bytes() const noexcept {
  return tags_.capacity() * sizeof(Tag) +
         fine_indices_.capacity() * sizeof(std::uint32_t) +
         mixed_words_.capacity() * sizeof(std::uint64_t) +
         free_words_.capacity() * sizeof(std::uint32_t);
}

} // namespace TitanPBE::Tabi::analytic_ses
