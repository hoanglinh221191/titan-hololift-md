#pragma once
#ifndef TITAN_NEIGHBOR_GRID_H
#define TITAN_NEIGHBOR_GRID_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace titan_neighbor {

struct OrthorhombicGridSpec {
  double cutoff{0.0};
  double len_x{0.0};
  double len_y{0.0};
  double len_z{0.0};
  double shift_x{0.0};
  double shift_y{0.0};
  double shift_z{0.0};
  bool periodic{true};
};

class HalfShellGrid {
public:
  template <typename CoordFn, typename ActiveFn, typename PairFn>
  void for_each_pair(std::size_t point_count, const OrthorhombicGridSpec &spec,
                     CoordFn coord_fn, ActiveFn active_fn,
                     PairFn pair_fn) {
    if (point_count == 0 || spec.cutoff <= 0.0 || spec.len_x <= 0.0 ||
        spec.len_y <= 0.0 || spec.len_z <= 0.0) {
      head_.clear();
      next_.assign(point_count, -1);
      return;
    }

    const int nx = std::max(1, static_cast<int>(spec.len_x / spec.cutoff));
    const int ny = std::max(1, static_cast<int>(spec.len_y / spec.cutoff));
    const int nz = std::max(1, static_cast<int>(spec.len_z / spec.cutoff));
    const double cell_x = spec.len_x / static_cast<double>(nx);
    const double cell_y = spec.len_y / static_cast<double>(ny);
    const double cell_z = spec.len_z / static_cast<double>(nz);

    // With one/two periodic cells, opposite stencil offsets alias the same
    // cell and a half shell no longer assigns each unordered pair once.
    const bool aliased_periodic_shell = spec.periodic && (nx < 3 || ny < 3 || nz < 3);

    head_.assign(static_cast<std::size_t>(nx * ny * nz), -1);
    next_.assign(point_count, -1);

    const auto wrap_index = [](int idx, int dim) noexcept {
      return ((idx % dim) + dim) % dim;
    };
    const auto clamp_index = [](int idx, int dim) noexcept {
      if (idx < 0)
        return 0;
      if (idx >= dim)
        return dim - 1;
      return idx;
    };
    const auto cell_of = [&](double x, double y, double z) noexcept {
      int ix =
          static_cast<int>(std::floor((x + spec.shift_x) / cell_x));
      int iy =
          static_cast<int>(std::floor((y + spec.shift_y) / cell_y));
      int iz =
          static_cast<int>(std::floor((z + spec.shift_z) / cell_z));
      if (spec.periodic) {
        ix = wrap_index(ix, nx);
        iy = wrap_index(iy, ny);
        iz = wrap_index(iz, nz);
      } else {
        ix = clamp_index(ix, nx);
        iy = clamp_index(iy, ny);
        iz = clamp_index(iz, nz);
      }
      return ix * ny * nz + iy * nz + iz;
    };

    for (std::size_t i = 0; i < point_count; ++i) {
      if (!active_fn(i))
        continue;
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      coord_fn(i, x, y, z);
      const int cell = cell_of(x, y, z);
      next_[i] = head_[static_cast<std::size_t>(cell)];
      head_[static_cast<std::size_t>(cell)] = static_cast<int>(i);
    }

    for (std::size_t i = 0; i < point_count; ++i) {
      if (!active_fn(i))
        continue;

      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      coord_fn(i, x, y, z);
      int ix = static_cast<int>(std::floor((x + spec.shift_x) / cell_x));
      int iy = static_cast<int>(std::floor((y + spec.shift_y) / cell_y));
      int iz = static_cast<int>(std::floor((z + spec.shift_z) / cell_z));
      if (spec.periodic) {
        ix = wrap_index(ix, nx);
        iy = wrap_index(iy, ny);
        iz = wrap_index(iz, nz);
      } else {
        ix = clamp_index(ix, nx);
        iy = clamp_index(iy, ny);
        iz = clamp_index(iz, nz);
      }

      std::array<int, 27> visited_cells{};
      std::size_t visited_count = 0;
      const std::size_t offset_count = aliased_periodic_shell ? 27 : kHalfShellOffsets.size();
      for (std::size_t oi = 0; oi < offset_count; ++oi) {
        const std::array<int, 3> offset = aliased_periodic_shell
            ? std::array<int, 3>{static_cast<int>(oi / 9) - 1,
                                 static_cast<int>((oi / 3) % 3) - 1,
                                 static_cast<int>(oi % 3) - 1}
            : kHalfShellOffsets[oi];
        const int raw_ix = ix + offset[0];
        const int raw_iy = iy + offset[1];
        const int raw_iz = iz + offset[2];
        if (!spec.periodic &&
            (raw_ix < 0 || raw_ix >= nx || raw_iy < 0 || raw_iy >= ny ||
             raw_iz < 0 || raw_iz >= nz)) {
          continue;
        }
        const int cell_ix =
            spec.periodic ? wrap_index(raw_ix, nx) : raw_ix;
        const int cell_iy =
            spec.periodic ? wrap_index(raw_iy, ny) : raw_iy;
        const int cell_iz =
            spec.periodic ? wrap_index(raw_iz, nz) : raw_iz;
        const int cell = cell_ix * ny * nz + cell_iy * nz + cell_iz;
        if (aliased_periodic_shell) {
          if (std::find(visited_cells.begin(), visited_cells.begin() + visited_count,
                        cell) != visited_cells.begin() + visited_count)
            continue;
          visited_cells[visited_count++] = cell;
        }

        for (int j = head_[static_cast<std::size_t>(cell)]; j != -1;
             j = next_[static_cast<std::size_t>(j)]) {
          if ((aliased_periodic_shell ||
               (offset[0] == 0 && offset[1] == 0 && offset[2] == 0)) &&
              static_cast<std::size_t>(j) <= i) {
            continue;
          }
          pair_fn(i, static_cast<std::size_t>(j));
        }
      }
    }
  }

private:
  static constexpr std::array<std::array<int, 3>, 14> kHalfShellOffsets{{
      {{0, 0, 0}},   {{1, 0, 0}},  {{-1, 1, 0}}, {{0, 1, 0}},
      {{1, 1, 0}},   {{-1, -1, 1}}, {{0, -1, 1}}, {{1, -1, 1}},
      {{-1, 0, 1}},  {{0, 0, 1}}, {{1, 0, 1}},  {{-1, 1, 1}},
      {{0, 1, 1}},   {{1, 1, 1}},
  }};

  std::vector<int> head_;
  std::vector<int> next_;
};

} // namespace titan_neighbor

#endif // TITAN_NEIGHBOR_GRID_H
