#ifndef TITAN_PBE_GRID_COMMON_H
#define TITAN_PBE_GRID_COMMON_H

#include "titan_executor.h"

#include <algorithm>
#include <cstddef>

namespace TitanPBE {

inline std::size_t idx3(std::size_t i, std::size_t j, std::size_t k,
                        std::size_t nx, std::size_t ny) noexcept {
  return (k * ny + j) * nx + i;
}

inline std::size_t effective_workers(const TitanExecutor &executor,
                                     std::size_t work_items) noexcept {
  if (work_items == 0)
    return 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(executor.capacity(), work_items));
}

} // namespace TitanPBE

#endif // TITAN_PBE_GRID_COMMON_H
