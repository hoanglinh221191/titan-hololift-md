#ifndef TITAN_PBE_TABI_SES_BILEVEL_GRID_H
#define TITAN_PBE_TABI_SES_BILEVEL_GRID_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace TitanPBE::Tabi::analytic_ses {

// A sparse boolean volume with one 4x4x4 bit word per mixed block.  Uniform
// blocks carry only a one-byte tag and a sentinel fine index.  This layout is
// independently implemented for TITAN and is intended for topology, exterior,
// and cavity masks; it has no external surface-library dependency.
class BilevelBitGrid {
public:
  BilevelBitGrid() = default;
  BilevelBitGrid(std::size_t nx, std::size_t ny, std::size_t nz,
                 bool initial_value = false);

  [[nodiscard]] std::size_t nx() const noexcept { return nx_; }
  [[nodiscard]] std::size_t ny() const noexcept { return ny_; }
  [[nodiscard]] std::size_t nz() const noexcept { return nz_; }
  [[nodiscard]] std::size_t size() const noexcept { return nx_ * ny_ * nz_; }

  [[nodiscard]] bool get(std::size_t x, std::size_t y,
                         std::size_t z) const noexcept;
  void set(std::size_t x, std::size_t y, std::size_t z, bool value);
  void compact();

  [[nodiscard]] std::size_t mixed_blocks() const noexcept {
    return mixed_words_.size() - free_words_.size();
  }
  [[nodiscard]] std::size_t retained_bytes() const noexcept;

private:
  static constexpr std::size_t kBlock = 4U;
  static constexpr std::uint32_t kNoWord =
      std::numeric_limits<std::uint32_t>::max();
  enum class Tag : std::uint8_t { all_zero = 0, all_one = 1, mixed = 2 };

  [[nodiscard]] std::size_t block_index(std::size_t x, std::size_t y,
                                        std::size_t z) const noexcept;
  [[nodiscard]] static std::uint64_t bit_mask(std::size_t x, std::size_t y,
                                              std::size_t z) noexcept;
  [[nodiscard]] std::uint64_t valid_mask(std::size_t bx, std::size_t by,
                                         std::size_t bz) const noexcept;
  [[nodiscard]] std::uint32_t allocate_word(std::uint64_t initial);
  void release_word(std::uint32_t word);

  std::size_t nx_{0};
  std::size_t ny_{0};
  std::size_t nz_{0};
  std::size_t nbx_{0};
  std::size_t nby_{0};
  std::size_t nbz_{0};
  std::vector<Tag> tags_;
  std::vector<std::uint32_t> fine_indices_;
  std::vector<std::uint64_t> mixed_words_;
  std::vector<std::uint32_t> free_words_;
};

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_BILEVEL_GRID_H
