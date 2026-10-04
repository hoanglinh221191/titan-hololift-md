#ifndef TITAN_PBE_GRID_SES_MASK_H
#define TITAN_PBE_GRID_SES_MASK_H

#include <cstddef>
#include <cstdint>
#include <span>

class TitanExecutor;

namespace TitanPBE {

struct SesMaskShape {
  std::size_t nx{0};
  std::size_t ny{0};
  std::size_t nz{0};
  double spacing{0.0};

  [[nodiscard]] constexpr std::size_t points() const noexcept {
    return nx * ny * nz;
  }
};

// Convert a probe-expanded molecular mask into a solvent-excluded mask by
// eroding it with a Euclidean ball of the same probe radius.
void erode_expanded_mask_to_ses(const SesMaskShape &shape,
                                std::span<const std::uint8_t> expanded_mask,
                                double probe_radius,
                                std::span<std::uint8_t> ses_mask,
                                TitanExecutor *executor);

} // namespace TitanPBE

#endif // TITAN_PBE_GRID_SES_MASK_H
