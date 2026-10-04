#include "component_shift_domain.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace titan_pbctopo {
namespace {

constexpr int kLegacyImageCoeff[27][3] = {
    {1, 0, 0},   {-1, 0, 0},  {0, 1, 0},   {0, -1, 0},  {0, 0, 1},
    {0, 0, -1},  {1, 1, 0},   {1, -1, 0},  {-1, 1, 0},  {-1, -1, 0},
    {1, 0, 1},   {1, 0, -1},  {-1, 0, 1},  {-1, 0, -1}, {0, 1, 1},
    {0, -1, 1},  {0, 1, -1},  {0, -1, -1}, {1, 1, 1},   {-1, 1, 1},
    {1, -1, 1},  {1, 1, -1},  {-1, -1, 1}, {-1, 1, -1}, {1, -1, -1},
    {-1, -1, -1}, {0, 0, 0}};

struct RelativeShiftKeyHash {
  std::size_t operator()(const Int3 &value) const noexcept {
    std::size_t seed = static_cast<std::size_t>(value.x);
    seed ^= static_cast<std::size_t>(value.y) + 0x9e3779b9U + (seed << 6) +
            (seed >> 2);
    seed ^= static_cast<std::size_t>(value.z) + 0x9e3779b9U + (seed << 6) +
            (seed >> 2);
    return seed;
  }
};

} // namespace

std::vector<PbctopoShift> build_component_image_shifts(box Box,
                                                        int shell_radius) {
  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);

  shell_radius = std::clamp(shell_radius, 1, 3);
  const std::size_t width = static_cast<std::size_t>(2 * shell_radius + 1);
  std::vector<PbctopoShift> shifts;
  shifts.reserve(width * width * width);
  auto append = [&](int ix, int iy, int iz) {
    const std::size_t index = shifts.size();
    shifts.push_back({index,
                      ix,
                      iy,
                      iz,
                      ix * v1[0] + iy * v2[0] + iz * v3[0],
                      ix * v1[1] + iy * v2[1] + iz * v3[1],
                      ix * v1[2] + iy * v2[2] + iz * v3[2]});
  };

  for (const auto &coeff : kLegacyImageCoeff)
    append(coeff[0], coeff[1], coeff[2]);
  for (int shell = 2; shell <= shell_radius; ++shell) {
    for (int ix = -shell; ix <= shell; ++ix) {
      for (int iy = -shell; iy <= shell; ++iy) {
        for (int iz = -shell; iz <= shell; ++iz) {
          if (std::max({std::abs(ix), std::abs(iy), std::abs(iz)}) == shell)
            append(ix, iy, iz);
        }
      }
    }
  }

  std::stable_sort(shifts.begin(), shifts.end(),
                   [](const PbctopoShift &lhs, const PbctopoShift &rhs) {
                     const double lhs_norm2 =
                         lhs.tx * lhs.tx + lhs.ty * lhs.ty + lhs.tz * lhs.tz;
                     const double rhs_norm2 =
                         rhs.tx * rhs.tx + rhs.ty * rhs.ty + rhs.tz * rhs.tz;
                     if (std::abs(lhs_norm2 - rhs_norm2) > 1.0e-12)
                       return lhs_norm2 < rhs_norm2;
                     return lhs.shift_index < rhs.shift_index;
                   });
  for (std::size_t index = 0; index < shifts.size(); ++index)
    shifts[index].shift_index = index;
  return shifts;
}

std::vector<PbctopoRelativeShiftMultiplicity>
build_component_relative_shift_domain(
    const std::vector<PbctopoShift> &shifts,
    bool *used_analytic_construction) {
  if (used_analytic_construction != nullptr)
    *used_analytic_construction = false;
  if (!shifts.empty()) {
    int min_x = shifts.front().ix;
    int max_x = shifts.front().ix;
    int min_y = shifts.front().iy;
    int max_y = shifts.front().iy;
    int min_z = shifts.front().iz;
    int max_z = shifts.front().iz;
    const PbctopoShift *basis_x = nullptr;
    const PbctopoShift *basis_y = nullptr;
    const PbctopoShift *basis_z = nullptr;
    for (const auto &shift : shifts) {
      min_x = std::min(min_x, shift.ix);
      max_x = std::max(max_x, shift.ix);
      min_y = std::min(min_y, shift.iy);
      max_y = std::max(max_y, shift.iy);
      min_z = std::min(min_z, shift.iz);
      max_z = std::max(max_z, shift.iz);
      if (shift.ix == 1 && shift.iy == 0 && shift.iz == 0)
        basis_x = &shift;
      if (shift.ix == 0 && shift.iy == 1 && shift.iz == 0)
        basis_y = &shift;
      if (shift.ix == 0 && shift.iy == 0 && shift.iz == 1)
        basis_z = &shift;
    }
    const std::int64_t span_x = static_cast<std::int64_t>(max_x) - min_x;
    const std::int64_t span_y = static_cast<std::int64_t>(max_y) - min_y;
    const std::int64_t span_z = static_cast<std::int64_t>(max_z) - min_z;
    if (span_x > std::numeric_limits<int>::max() ||
        span_y > std::numeric_limits<int>::max() ||
        span_z > std::numeric_limits<int>::max()) {
      return {};
    }
    const std::size_t width_x =
        static_cast<std::size_t>(span_x) + 1;
    const std::size_t width_y =
        static_cast<std::size_t>(span_y) + 1;
    const std::size_t width_z =
        static_cast<std::size_t>(span_z) + 1;
    const bool centered =
        static_cast<std::int64_t>(min_x) ==
            -static_cast<std::int64_t>(max_x) &&
        static_cast<std::int64_t>(min_y) ==
            -static_cast<std::int64_t>(max_y) &&
        static_cast<std::int64_t>(min_z) ==
            -static_cast<std::int64_t>(max_z);
    const bool candidate_complete =
        centered && basis_x != nullptr && basis_y != nullptr &&
        basis_z != nullptr &&
        width_x * width_y * width_z == shifts.size();
    bool complete = candidate_complete;
    std::vector<unsigned char> occupied;
    if (candidate_complete) {
      occupied.assign(shifts.size(), 0);
      for (const auto &shift : shifts) {
        const std::size_t x = static_cast<std::size_t>(shift.ix - min_x);
        const std::size_t y = static_cast<std::size_t>(shift.iy - min_y);
        const std::size_t z = static_cast<std::size_t>(shift.iz - min_z);
        const std::size_t index = (x * width_y + y) * width_z + z;
        if (index >= occupied.size() || occupied[index] != 0) {
          complete = false;
          break;
        }
        occupied[index] = 1;
      }
    }
    if (complete) {
      std::vector<PbctopoRelativeShiftMultiplicity> relative;
      relative.reserve((2 * width_x - 1) * (2 * width_y - 1) *
                       (2 * width_z - 1));
      for (int dx = min_x - max_x; dx <= max_x - min_x; ++dx) {
        for (int dy = min_y - max_y; dy <= max_y - min_y; ++dy) {
          for (int dz = min_z - max_z; dz <= max_z - min_z; ++dz) {
            PbctopoShift shift;
            shift.ix = dx;
            shift.iy = dy;
            shift.iz = dz;
            shift.tx = dx * basis_x->tx + dy * basis_y->tx +
                       dz * basis_z->tx;
            shift.ty = dx * basis_x->ty + dy * basis_y->ty +
                       dz * basis_z->ty;
            shift.tz = dx * basis_x->tz + dy * basis_y->tz +
                       dz * basis_z->tz;
            const std::size_t multiplicity =
                (width_x - static_cast<std::size_t>(std::abs(dx))) *
                (width_y - static_cast<std::size_t>(std::abs(dy))) *
                (width_z - static_cast<std::size_t>(std::abs(dz)));
            relative.push_back({shift, multiplicity});
          }
        }
      }
      std::stable_sort(
          relative.begin(), relative.end(),
          [](const PbctopoRelativeShiftMultiplicity &lhs,
             const PbctopoRelativeShiftMultiplicity &rhs) {
            const double lhs_norm2 = lhs.shift.tx * lhs.shift.tx +
                                     lhs.shift.ty * lhs.shift.ty +
                                     lhs.shift.tz * lhs.shift.tz;
            const double rhs_norm2 = rhs.shift.tx * rhs.shift.tx +
                                     rhs.shift.ty * rhs.shift.ty +
                                     rhs.shift.tz * rhs.shift.tz;
            if (std::abs(lhs_norm2 - rhs_norm2) > 1.0e-12)
              return lhs_norm2 < rhs_norm2;
            if (lhs.shift.ix != rhs.shift.ix)
              return lhs.shift.ix < rhs.shift.ix;
            if (lhs.shift.iy != rhs.shift.iy)
              return lhs.shift.iy < rhs.shift.iy;
            return lhs.shift.iz < rhs.shift.iz;
          });
      for (std::size_t index = 0; index < relative.size(); ++index)
        relative[index].shift.shift_index = index;
      if (used_analytic_construction != nullptr)
        *used_analytic_construction = true;
      return relative;
    }
  }
  std::vector<PbctopoRelativeShiftMultiplicity> relative;
  std::unordered_map<Int3, std::size_t, RelativeShiftKeyHash> index_by_delta;
  index_by_delta.reserve(shifts.size() * 4 + 1);
  for (const auto &lhs : shifts) {
    for (const auto &rhs : shifts) {
      Int3 delta{};
      if (!checked_int3_subtract(Int3{rhs.ix, rhs.iy, rhs.iz},
                                 Int3{lhs.ix, lhs.iy, lhs.iz}, delta)) {
        return {};
      }
      if (delta.x < std::numeric_limits<int>::min() ||
          delta.x > std::numeric_limits<int>::max() ||
          delta.y < std::numeric_limits<int>::min() ||
          delta.y > std::numeric_limits<int>::max() ||
          delta.z < std::numeric_limits<int>::min() ||
          delta.z > std::numeric_limits<int>::max()) {
        return {};
      }
      const auto found = index_by_delta.find(delta);
      if (found != index_by_delta.end()) {
        ++relative[found->second].multiplicity;
        continue;
      }
      PbctopoShift relative_shift;
      relative_shift.shift_index = relative.size();
      relative_shift.ix = static_cast<int>(delta.x);
      relative_shift.iy = static_cast<int>(delta.y);
      relative_shift.iz = static_cast<int>(delta.z);
      relative_shift.tx = rhs.tx - lhs.tx;
      relative_shift.ty = rhs.ty - lhs.ty;
      relative_shift.tz = rhs.tz - lhs.tz;
      index_by_delta.emplace(delta, relative.size());
      relative.push_back({relative_shift, 1});
    }
  }
  return relative;
}

} // namespace titan_pbctopo
