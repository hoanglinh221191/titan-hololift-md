#include "titan_surface.h"

#include "aligned_allocator.h"
#include "titan_executor.h"
#include "titan_simd_dispatch.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace TitanSurface {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr std::uint32_t kMortonMax = (1u << 21) - 1u;
constexpr std::size_t kAutoOptimizedTargetThreshold = 128;

struct SpherePoint {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct GridInfo {
  double min_x{0.0};
  double min_y{0.0};
  double min_z{0.0};
  double max_x{0.0};
  double max_y{0.0};
  double max_z{0.0};
  double cell{1.0};
  double probe{0.0};
  std::size_t bx{1};
  std::size_t by{1};
  std::size_t bz{1};
  std::size_t cell_count{1};
};

struct AtomOrderEntry {
  std::uint64_t morton{0};
  std::size_t original_index{0};
  std::size_t cell_index{0};
};

struct OptimizedAtoms {
  AlignedDoubleVector x;
  AlignedDoubleVector y;
  AlignedDoubleVector z;
  AlignedDoubleVector radius;
  AlignedDoubleVector surface_radius;
  AlignedDoubleVector cutoff_sq;
  std::vector<std::size_t> original_index;
  std::vector<std::size_t> sorted_index_by_original;
  std::vector<std::size_t> cell_index_by_original;
};

const std::vector<SpherePoint> &sphere_points(int point_count) {
  static const std::vector<SpherePoint> default_points = [] {
    std::vector<SpherePoint> out(kDefaultPointCount);
    const double inc = kPi * (3.0 - std::sqrt(5.0));
    const double off = 2.0 / static_cast<double>(kDefaultPointCount);
    for (int k = 0; k < kDefaultPointCount; ++k) {
      const double y = static_cast<double>(k) * off - 1.0 + 0.5 * off;
      const double r = std::sqrt(std::max(0.0, 1.0 - y * y));
      const double phi = static_cast<double>(k) * inc;
      out[static_cast<std::size_t>(k)] = {std::cos(phi) * r, y,
                                          std::sin(phi) * r};
    }
    return out;
  }();
  if (point_count == kDefaultPointCount)
    return default_points;

  static thread_local std::vector<SpherePoint> custom_points;
  static thread_local int custom_count = 0;
  if (custom_count != point_count) {
    custom_points.assign(static_cast<std::size_t>(point_count), {});
    const double inc = kPi * (3.0 - std::sqrt(5.0));
    const double off = 2.0 / static_cast<double>(point_count);
    for (int k = 0; k < point_count; ++k) {
      const double y = static_cast<double>(k) * off - 1.0 + 0.5 * off;
      const double r = std::sqrt(std::max(0.0, 1.0 - y * y));
      const double phi = static_cast<double>(k) * inc;
      custom_points[static_cast<std::size_t>(k)] = {std::cos(phi) * r, y,
                                                    std::sin(phi) * r};
    }
    custom_count = point_count;
  }
  return custom_points;
}

std::size_t effective_workers(const TitanExecutor *executor,
                              std::size_t work_items) noexcept {
  if (executor == nullptr || work_items == 0)
    return 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(executor->capacity(), work_items));
}

double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) noexcept {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::uint64_t expand_bits_21(std::uint32_t value) noexcept {
  value = std::min(value, kMortonMax);
  std::uint64_t out = 0;
  for (std::uint32_t bit = 0; bit < 21; ++bit)
    out |= static_cast<std::uint64_t>((value >> bit) & 1u) << (3u * bit);
  return out;
}

std::uint64_t morton_code(std::size_t ix, std::size_t iy,
                          std::size_t iz) noexcept {
  return (expand_bits_21(static_cast<std::uint32_t>(
              std::min<std::size_t>(ix, kMortonMax)))
          << 2u) |
         (expand_bits_21(static_cast<std::uint32_t>(
              std::min<std::size_t>(iy, kMortonMax)))
          << 1u) |
         expand_bits_21(static_cast<std::uint32_t>(
             std::min<std::size_t>(iz, kMortonMax)));
}

std::size_t round_up4(std::size_t value) noexcept {
  return (value + 3u) & ~std::size_t{3};
}

GridInfo make_grid_info(std::span<const Atom> atoms,
                        const Options &options) {
  GridInfo grid;
  grid.probe =
      options.surface_mode == "vdw" ? 0.0 : std::max(0.0, options.probe_radius);

  double max_surface_radius = 0.0;
  grid.min_x = std::numeric_limits<double>::max();
  grid.min_y = std::numeric_limits<double>::max();
  grid.min_z = std::numeric_limits<double>::max();
  grid.max_x = std::numeric_limits<double>::lowest();
  grid.max_y = std::numeric_limits<double>::lowest();
  grid.max_z = std::numeric_limits<double>::lowest();
  for (const Atom &atom : atoms) {
    const double surface_radius = std::max(0.0, atom.radius) + grid.probe;
    max_surface_radius = std::max(max_surface_radius, surface_radius);
    grid.min_x = std::min(grid.min_x, atom.x);
    grid.min_y = std::min(grid.min_y, atom.y);
    grid.min_z = std::min(grid.min_z, atom.z);
    grid.max_x = std::max(grid.max_x, atom.x);
    grid.max_y = std::max(grid.max_y, atom.y);
    grid.max_z = std::max(grid.max_z, atom.z);
  }

  grid.cell = std::max(options.min_cell_size, 2.0 * max_surface_radius);
  grid.bx = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::floor((grid.max_x - grid.min_x) / grid.cell)) +
             1);
  grid.by = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::floor((grid.max_y - grid.min_y) / grid.cell)) +
             1);
  grid.bz = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::floor((grid.max_z - grid.min_z) / grid.cell)) +
             1);
  grid.cell_count = grid.bx * grid.by * grid.bz;
  return grid;
}

std::size_t cell_index(const GridInfo &grid, double x, double y,
                       double z) noexcept {
  const std::size_t ix = std::min<std::size_t>(
      grid.bx - 1, static_cast<std::size_t>(
                       std::max(0.0, std::floor((x - grid.min_x) / grid.cell))));
  const std::size_t iy = std::min<std::size_t>(
      grid.by - 1, static_cast<std::size_t>(
                       std::max(0.0, std::floor((y - grid.min_y) / grid.cell))));
  const std::size_t iz = std::min<std::size_t>(
      grid.bz - 1, static_cast<std::size_t>(
                       std::max(0.0, std::floor((z - grid.min_z) / grid.cell))));
  return (iz * grid.by + iy) * grid.bx + ix;
}

void cell_coords(const GridInfo &grid, std::size_t cell, std::size_t &ix,
                 std::size_t &iy, std::size_t &iz) noexcept {
  ix = cell % grid.bx;
  iy = (cell / grid.bx) % grid.by;
  iz = cell / (grid.bx * grid.by);
}

std::vector<std::size_t>
make_targets(std::span<const Atom> atoms, std::span<const std::size_t> targets) {
  std::vector<std::size_t> out;
  if (targets.empty()) {
    out.resize(atoms.size());
    std::iota(out.begin(), out.end(), std::size_t{0});
  } else {
    out.assign(targets.begin(), targets.end());
  }
  return out;
}

void finalize_panels(Workspace &workspace,
                     std::vector<std::vector<Panel>> &panels_by_atom) {
  std::size_t panel_count = 0;
  for (const auto &atom_panels : panels_by_atom)
    panel_count += atom_panels.size();
  workspace.panels.reserve(panel_count);
  for (auto &atom_panels : panels_by_atom) {
    workspace.panels.insert(workspace.panels.end(),
                            std::make_move_iterator(atom_panels.begin()),
                            std::make_move_iterator(atom_panels.end()));
  }
}

void finalize_diagnostics(Workspace &workspace,
                          const std::vector<std::size_t> &exposed_by_atom,
                          const std::vector<std::size_t> &buried_by_atom,
                          const std::vector<std::size_t> &neighbor_counts) {
  workspace.total_area =
      std::accumulate(workspace.area_by_atom.begin(),
                      workspace.area_by_atom.end(), 0.0);
  workspace.diagnostics.exposed_points =
      std::accumulate(exposed_by_atom.begin(), exposed_by_atom.end(),
                      std::size_t{0});
  workspace.diagnostics.buried_points =
      std::accumulate(buried_by_atom.begin(), buried_by_atom.end(),
                      std::size_t{0});
  const std::size_t neighbor_sum =
      std::accumulate(neighbor_counts.begin(), neighbor_counts.end(),
                      std::size_t{0});
  workspace.diagnostics.max_neighbor_count =
      neighbor_counts.empty()
          ? 0
          : *std::max_element(neighbor_counts.begin(), neighbor_counts.end());
  workspace.diagnostics.mean_neighbor_count =
      workspace.diagnostics.target_count > 0
          ? static_cast<double>(neighbor_sum) /
                static_cast<double>(workspace.diagnostics.target_count)
          : 0.0;
}

Workspace build_workspace_legacy(std::span<const Atom> atoms,
                                 const Options &options,
                                 std::span<const std::size_t> targets,
                                 TitanExecutor *executor) {
  const auto build_begin = std::chrono::steady_clock::now();
  Workspace workspace;
  workspace.area_by_atom.assign(atoms.size(), 0.0);
  workspace.diagnostics.acceleration_effective = "legacy";

  const int point_count =
      options.point_count > 0 ? options.point_count : kDefaultPointCount;
  workspace.diagnostics.point_count = point_count;
  workspace.diagnostics.target_count =
      targets.empty() ? atoms.size() : targets.size();

  if (atoms.empty()) {
    if (options.emit_neighbor_workspace)
      workspace.neighbor_offsets.assign(1, 0);
    workspace.diagnostics.build_ms =
        elapsed_ms(build_begin, std::chrono::steady_clock::now());
    workspace.diagnostics.exposure_ms = workspace.diagnostics.build_ms;
    return workspace;
  }

  const auto &sphere = sphere_points(point_count);
  const GridInfo grid = make_grid_info(atoms, options);
  workspace.diagnostics.cell_count = grid.cell_count;

  std::vector<std::vector<std::size_t>> bins(grid.cell_count);
  auto bin_index = [&](double x, double y, double z) {
    return cell_index(grid, x, y, z);
  };

  for (std::size_t i = 0; i < atoms.size(); ++i)
    bins[bin_index(atoms[i].x, atoms[i].y, atoms[i].z)].push_back(i);

  std::vector<std::size_t> all_targets = make_targets(atoms, targets);
  targets = std::span<const std::size_t>(all_targets.data(),
                                         all_targets.size());

  std::vector<std::size_t> exposed_by_atom(atoms.size(), 0);
  std::vector<std::size_t> buried_by_atom(atoms.size(), 0);
  std::vector<std::size_t> neighbor_counts(atoms.size(), 0);
  std::vector<std::vector<Panel>> panels_by_atom;
  std::vector<std::vector<std::size_t>> neighbors_by_atom;
  if (options.emit_panels)
    panels_by_atom.resize(atoms.size());
  if (options.emit_neighbor_workspace)
    neighbors_by_atom.resize(atoms.size());

  const auto exposure_begin = std::chrono::steady_clock::now();
  auto compute_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t target_pos = begin; target_pos < end; ++target_pos) {
      const std::size_t i = targets[target_pos];
      if (i >= atoms.size())
        continue;
      const Atom &atom = atoms[i];
      const double ri = std::max(0.0, atom.radius) + grid.probe;
      if (ri <= 0.0)
        continue;

      const double point_area =
          4.0 * kPi * ri * ri / static_cast<double>(point_count);
      const std::size_t base = bin_index(atom.x, atom.y, atom.z);
      std::size_t bix = 0;
      std::size_t biy = 0;
      std::size_t biz = 0;
      cell_coords(grid, base, bix, biy, biz);
      const std::size_t x0 = (bix == 0) ? 0 : bix - 1;
      const std::size_t y0 = (biy == 0) ? 0 : biy - 1;
      const std::size_t z0 = (biz == 0) ? 0 : biz - 1;
      const std::size_t x1 = std::min(grid.bx - 1, bix + 1);
      const std::size_t y1 = std::min(grid.by - 1, biy + 1);
      const std::size_t z1 = std::min(grid.bz - 1, biz + 1);

      std::vector<std::size_t> neighbors;
      neighbors.reserve(64);
      for (std::size_t kk = z0; kk <= z1; ++kk) {
        for (std::size_t jj = y0; jj <= y1; ++jj) {
          for (std::size_t ii = x0; ii <= x1; ++ii) {
            const auto &cell_atoms =
                bins[(kk * grid.by + jj) * grid.bx + ii];
            for (std::size_t j : cell_atoms) {
              if (j == i)
                continue;
              const Atom &other = atoms[j];
              const double rj = std::max(0.0, other.radius) + grid.probe;
              const double dx = atom.x - other.x;
              const double dy = atom.y - other.y;
              const double dz = atom.z - other.z;
              if (dx * dx + dy * dy + dz * dz < (ri + rj) * (ri + rj))
                neighbors.push_back(j);
            }
          }
        }
      }

      neighbor_counts[i] = neighbors.size();
      if (options.emit_neighbor_workspace)
        neighbors_by_atom[i] = neighbors;

      std::vector<Panel> atom_panels;
      if (options.emit_panels)
        atom_panels.reserve(sphere.size());

      const std::size_t full_groups = sphere.size() / 4;
      std::vector<unsigned> buried_mask(full_groups, 0u);
      std::vector<std::array<double, 4>> px(full_groups);
      std::vector<std::array<double, 4>> py(full_groups);
      std::vector<std::array<double, 4>> pz(full_groups);
      for (std::size_t g = 0; g < full_groups; ++g) {
        for (int lane = 0; lane < 4; ++lane) {
          const SpherePoint &pt =
              sphere[g * 4 + static_cast<std::size_t>(lane)];
          px[g][static_cast<std::size_t>(lane)] = atom.x + pt.x * ri;
          py[g][static_cast<std::size_t>(lane)] = atom.y + pt.y * ri;
          pz[g][static_cast<std::size_t>(lane)] = atom.z + pt.z * ri;
        }
      }

      for (std::size_t j : neighbors) {
        const Atom &other = atoms[j];
        const double rj = std::max(0.0, other.radius) + grid.probe;
        const double cutoff_sq = rj * rj;
        for (std::size_t g = 0; g < full_groups; ++g) {
          buried_mask[g] |= titan_simd_dispatch::point_vs_four_cutoff_mask(
              other.x, other.y, other.z, px[g].data(), py[g].data(),
              pz[g].data(), cutoff_sq, -1.0, false);
        }
      }

      std::size_t exposed = 0;
      auto emit_panel = [&](const SpherePoint &pt, double x, double y,
                            double z) {
        if (!options.emit_panels)
          return;
        atom_panels.push_back(
            Panel{x, y, z, pt.x, pt.y, pt.z, point_area, i});
      };

      for (std::size_t g = 0; g < full_groups; ++g) {
        const unsigned mask = buried_mask[g];
        for (int lane = 0; lane < 4; ++lane) {
          if ((mask & (1u << lane)) != 0u)
            continue;
          ++exposed;
          const std::size_t point_index =
              g * 4 + static_cast<std::size_t>(lane);
          emit_panel(sphere[point_index], px[g][static_cast<std::size_t>(lane)],
                     py[g][static_cast<std::size_t>(lane)],
                     pz[g][static_cast<std::size_t>(lane)]);
        }
      }

      const std::size_t remainder_begin = full_groups * 4;
      for (std::size_t p = remainder_begin; p < sphere.size(); ++p) {
        const SpherePoint &pt = sphere[p];
        const double sx = atom.x + pt.x * ri;
        const double sy = atom.y + pt.y * ri;
        const double sz = atom.z + pt.z * ri;
        bool buried = false;
        for (std::size_t j : neighbors) {
          const Atom &other = atoms[j];
          const double rj = std::max(0.0, other.radius) + grid.probe;
          const double dx = sx - other.x;
          const double dy = sy - other.y;
          const double dz = sz - other.z;
          if (dx * dx + dy * dy + dz * dz < rj * rj) {
            buried = true;
            break;
          }
        }
        if (!buried) {
          ++exposed;
          emit_panel(pt, sx, sy, sz);
        }
      }

      exposed_by_atom[i] = exposed;
      buried_by_atom[i] = sphere.size() - exposed;
      workspace.area_by_atom[i] = static_cast<double>(exposed) * point_area;
      if (options.emit_panels)
        panels_by_atom[i] = std::move(atom_panels);
    }
  };

  if (executor != nullptr && targets.size() > 1) {
    const std::size_t workers = effective_workers(executor, targets.size());
    executor->parallel_for(targets.size(), workers, 8,
                           [&](std::size_t worker, std::size_t begin,
                               std::size_t end) {
                             (void)worker;
                             compute_range(begin, end);
                           });
  } else {
    compute_range(0, targets.size());
  }
  workspace.diagnostics.exposure_ms =
      elapsed_ms(exposure_begin, std::chrono::steady_clock::now());

  finalize_diagnostics(workspace, exposed_by_atom, buried_by_atom,
                       neighbor_counts);

  if (options.emit_panels)
    finalize_panels(workspace, panels_by_atom);

  if (options.emit_neighbor_workspace) {
    workspace.neighbor_offsets.assign(atoms.size() + 1, 0);
    for (std::size_t i = 0; i < neighbors_by_atom.size(); ++i)
      workspace.neighbor_offsets[i + 1] =
          workspace.neighbor_offsets[i] + neighbors_by_atom[i].size();
    workspace.neighbors.reserve(workspace.neighbor_offsets.back());
    for (const auto &neighbors : neighbors_by_atom)
      workspace.neighbors.insert(workspace.neighbors.end(), neighbors.begin(),
                                 neighbors.end());
  }

  workspace.diagnostics.build_ms =
      elapsed_ms(build_begin, std::chrono::steady_clock::now());
  return workspace;
}

OptimizedAtoms build_reordered_atoms(std::span<const Atom> atoms,
                                     const GridInfo &grid) {
  std::vector<AtomOrderEntry> entries;
  entries.reserve(atoms.size());
  std::vector<std::size_t> cell_by_original(atoms.size(), 0);
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const std::size_t cell = cell_index(grid, atoms[i].x, atoms[i].y,
                                        atoms[i].z);
    std::size_t ix = 0;
    std::size_t iy = 0;
    std::size_t iz = 0;
    cell_coords(grid, cell, ix, iy, iz);
    entries.push_back({morton_code(ix, iy, iz), i, cell});
    cell_by_original[i] = cell;
  }
  std::stable_sort(entries.begin(), entries.end(),
                   [](const AtomOrderEntry &lhs, const AtomOrderEntry &rhs) {
                     if (lhs.morton != rhs.morton)
                       return lhs.morton < rhs.morton;
                     return lhs.original_index < rhs.original_index;
                   });

  OptimizedAtoms reordered;
  reordered.x.resize(atoms.size());
  reordered.y.resize(atoms.size());
  reordered.z.resize(atoms.size());
  reordered.radius.resize(atoms.size());
  reordered.surface_radius.resize(atoms.size());
  reordered.cutoff_sq.resize(atoms.size());
  reordered.original_index.resize(atoms.size());
  reordered.sorted_index_by_original.assign(atoms.size(), 0);
  reordered.cell_index_by_original = std::move(cell_by_original);

  for (std::size_t sorted = 0; sorted < entries.size(); ++sorted) {
    const std::size_t original = entries[sorted].original_index;
    const Atom &atom = atoms[original];
    const double surface_radius = std::max(0.0, atom.radius) + grid.probe;
    reordered.x[sorted] = atom.x;
    reordered.y[sorted] = atom.y;
    reordered.z[sorted] = atom.z;
    reordered.radius[sorted] = std::max(0.0, atom.radius);
    reordered.surface_radius[sorted] = surface_radius;
    reordered.cutoff_sq[sorted] = surface_radius * surface_radius;
    reordered.original_index[sorted] = original;
    reordered.sorted_index_by_original[original] = sorted;
  }
  return reordered;
}

void build_cell_csr(const OptimizedAtoms &atoms, const GridInfo &grid,
                    std::vector<std::size_t> &cell_offsets,
                    std::vector<std::size_t> &cell_atoms_sorted) {
  cell_offsets.assign(grid.cell_count + 1, 0);
  for (std::size_t sorted = 0; sorted < atoms.original_index.size(); ++sorted) {
    const std::size_t original = atoms.original_index[sorted];
    ++cell_offsets[atoms.cell_index_by_original[original] + 1];
  }
  for (std::size_t cell = 0; cell < grid.cell_count; ++cell)
    cell_offsets[cell + 1] += cell_offsets[cell];
  cell_atoms_sorted.assign(atoms.original_index.size(), 0);
  std::vector<std::size_t> cursor = cell_offsets;
  for (std::size_t sorted = 0; sorted < atoms.original_index.size(); ++sorted) {
    const std::size_t original = atoms.original_index[sorted];
    const std::size_t cell = atoms.cell_index_by_original[original];
    cell_atoms_sorted[cursor[cell]++] = sorted;
  }
}

template <typename Fn>
void for_each_neighbor_candidate(const OptimizedAtoms &reordered,
                                 const GridInfo &grid,
                                 const std::vector<std::size_t> &cell_offsets,
                                 const std::vector<std::size_t> &cell_atoms,
                                 std::size_t original_index, Fn &&fn) {
  const std::size_t target_sorted =
      reordered.sorted_index_by_original[original_index];
  const double ri = reordered.surface_radius[target_sorted];
  const std::size_t base = reordered.cell_index_by_original[original_index];
  std::size_t bix = 0;
  std::size_t biy = 0;
  std::size_t biz = 0;
  cell_coords(grid, base, bix, biy, biz);
  const std::size_t x0 = (bix == 0) ? 0 : bix - 1;
  const std::size_t y0 = (biy == 0) ? 0 : biy - 1;
  const std::size_t z0 = (biz == 0) ? 0 : biz - 1;
  const std::size_t x1 = std::min(grid.bx - 1, bix + 1);
  const std::size_t y1 = std::min(grid.by - 1, biy + 1);
  const std::size_t z1 = std::min(grid.bz - 1, biz + 1);

  for (std::size_t kk = z0; kk <= z1; ++kk) {
    for (std::size_t jj = y0; jj <= y1; ++jj) {
      for (std::size_t ii = x0; ii <= x1; ++ii) {
        const std::size_t cell = (kk * grid.by + jj) * grid.bx + ii;
        for (std::size_t pos = cell_offsets[cell];
             pos < cell_offsets[cell + 1]; ++pos) {
          const std::size_t candidate_sorted = cell_atoms[pos];
          if (candidate_sorted == target_sorted)
            continue;
          const double dx =
              reordered.x[target_sorted] - reordered.x[candidate_sorted];
          const double dy =
              reordered.y[target_sorted] - reordered.y[candidate_sorted];
          const double dz =
              reordered.z[target_sorted] - reordered.z[candidate_sorted];
          const double rj = reordered.surface_radius[candidate_sorted];
          const double cutoff = ri + rj;
          if (dx * dx + dy * dy + dz * dz < cutoff * cutoff)
            fn(candidate_sorted);
        }
      }
    }
  }
}

Workspace build_workspace_reordered_padded(std::span<const Atom> atoms,
                                           const Options &options,
                                           std::span<const std::size_t> targets,
                                           TitanExecutor *executor) {
  const auto build_begin = std::chrono::steady_clock::now();
  Workspace workspace;
  workspace.area_by_atom.assign(atoms.size(), 0.0);
  workspace.diagnostics.acceleration_effective = "reordered_padded";

  const int point_count =
      options.point_count > 0 ? options.point_count : kDefaultPointCount;
  workspace.diagnostics.point_count = point_count;
  workspace.diagnostics.target_count =
      targets.empty() ? atoms.size() : targets.size();

  if (atoms.empty()) {
    if (options.emit_neighbor_workspace)
      workspace.neighbor_offsets.assign(1, 0);
    workspace.diagnostics.build_ms =
        elapsed_ms(build_begin, std::chrono::steady_clock::now());
    return workspace;
  }

  const auto &sphere = sphere_points(point_count);
  const GridInfo grid = make_grid_info(atoms, options);
  workspace.diagnostics.cell_count = grid.cell_count;
  workspace.diagnostics.reordered_atoms = atoms.size();

  const OptimizedAtoms reordered = build_reordered_atoms(atoms, grid);
  std::vector<std::size_t> cell_offsets;
  std::vector<std::size_t> cell_atoms_sorted;
  build_cell_csr(reordered, grid, cell_offsets, cell_atoms_sorted);

  const std::vector<std::size_t> target_indices = make_targets(atoms, targets);
  std::vector<std::size_t> neighbor_counts(atoms.size(), 0);
  const auto neighbor_begin = std::chrono::steady_clock::now();
  auto count_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t target_pos = begin; target_pos < end; ++target_pos) {
      const std::size_t original = target_indices[target_pos];
      if (original >= atoms.size())
        continue;
      std::size_t count = 0;
      for_each_neighbor_candidate(
          reordered, grid, cell_offsets, cell_atoms_sorted, original,
          [&](std::size_t) { ++count; });
      neighbor_counts[original] = count;
    }
  };
  if (executor != nullptr && target_indices.size() > 1) {
    const std::size_t workers = effective_workers(executor, target_indices.size());
    executor->parallel_for(target_indices.size(), workers, 16,
                           [&](std::size_t, std::size_t begin,
                               std::size_t end) { count_range(begin, end); });
  } else {
    count_range(0, target_indices.size());
  }

  std::vector<std::size_t> padded_offsets(atoms.size() + 1, 0);
  for (std::size_t i = 0; i < atoms.size(); ++i)
    padded_offsets[i + 1] = padded_offsets[i] + round_up4(neighbor_counts[i]);
  std::vector<std::int64_t> padded_neighbors(padded_offsets.back(), 0);
  std::vector<double> padded_cutoff_sq(padded_offsets.back(), -1.0);

  if (options.emit_neighbor_workspace) {
    workspace.neighbor_offsets.assign(atoms.size() + 1, 0);
    for (std::size_t i = 0; i < atoms.size(); ++i)
      workspace.neighbor_offsets[i + 1] =
          workspace.neighbor_offsets[i] + neighbor_counts[i];
    workspace.neighbors.assign(workspace.neighbor_offsets.back(), 0);
  }

  auto fill_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t target_pos = begin; target_pos < end; ++target_pos) {
      const std::size_t original = target_indices[target_pos];
      if (original >= atoms.size())
        continue;
      std::size_t padded_pos = padded_offsets[original];
      std::size_t output_pos =
          options.emit_neighbor_workspace ? workspace.neighbor_offsets[original]
                                          : 0;
      for_each_neighbor_candidate(
          reordered, grid, cell_offsets, cell_atoms_sorted, original,
          [&](std::size_t candidate_sorted) {
            padded_neighbors[padded_pos] =
                static_cast<std::int64_t>(candidate_sorted);
            padded_cutoff_sq[padded_pos] = reordered.cutoff_sq[candidate_sorted];
            ++padded_pos;
            if (options.emit_neighbor_workspace) {
              workspace.neighbors[output_pos] =
                  reordered.original_index[candidate_sorted];
              ++output_pos;
            }
          });
    }
  };
  if (executor != nullptr && target_indices.size() > 1) {
    const std::size_t workers = effective_workers(executor, target_indices.size());
    executor->parallel_for(target_indices.size(), workers, 16,
                           [&](std::size_t, std::size_t begin,
                               std::size_t end) { fill_range(begin, end); });
  } else {
    fill_range(0, target_indices.size());
  }
  workspace.diagnostics.neighbor_build_ms =
      elapsed_ms(neighbor_begin, std::chrono::steady_clock::now());
  workspace.diagnostics.padded_neighbor_slots = padded_neighbors.size();

  std::vector<std::size_t> exposed_by_atom(atoms.size(), 0);
  std::vector<std::size_t> buried_by_atom(atoms.size(), 0);
  std::vector<std::vector<Panel>> panels_by_atom;
  if (options.emit_panels)
    panels_by_atom.resize(atoms.size());

  const auto exposure_begin = std::chrono::steady_clock::now();
  auto expose_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t target_pos = begin; target_pos < end; ++target_pos) {
      const std::size_t original = target_indices[target_pos];
      if (original >= atoms.size())
        continue;
      const std::size_t target_sorted =
          reordered.sorted_index_by_original[original];
      const double ri = reordered.surface_radius[target_sorted];
      if (ri <= 0.0)
        continue;

      const double point_area =
          4.0 * kPi * ri * ri / static_cast<double>(point_count);
      const std::size_t neighbor_begin_pos = padded_offsets[original];
      const std::size_t neighbor_end_pos = padded_offsets[original + 1];
      std::vector<Panel> atom_panels;
      if (options.emit_panels)
        atom_panels.reserve(sphere.size());

      std::size_t exposed = 0;
      for (std::size_t p = 0; p < sphere.size(); ++p) {
        const SpherePoint &pt = sphere[p];
        const double sx = reordered.x[target_sorted] + pt.x * ri;
        const double sy = reordered.y[target_sorted] + pt.y * ri;
        const double sz = reordered.z[target_sorted] + pt.z * ri;
        bool buried = false;
        for (std::size_t pos = neighbor_begin_pos; pos < neighbor_end_pos;
             pos += 4) {
          const unsigned mask =
              titan_simd_dispatch::point_vs_gathered4_variable_cutoff_mask(
                  sx, sy, sz, reordered.x.data(), reordered.y.data(),
                  reordered.z.data(), padded_neighbors.data() + pos,
                  padded_cutoff_sq.data() + pos, -1.0, false);
          if (mask != 0u) {
            buried = true;
            break;
          }
        }
        if (buried)
          continue;
        ++exposed;
        if (options.emit_panels)
          atom_panels.push_back(
              Panel{sx, sy, sz, pt.x, pt.y, pt.z, point_area, original});
      }

      exposed_by_atom[original] = exposed;
      buried_by_atom[original] = sphere.size() - exposed;
      workspace.area_by_atom[original] =
          static_cast<double>(exposed) * point_area;
      if (options.emit_panels)
        panels_by_atom[original] = std::move(atom_panels);
    }
  };
  if (executor != nullptr && target_indices.size() > 1) {
    const std::size_t workers = effective_workers(executor, target_indices.size());
    executor->parallel_for(target_indices.size(), workers, 16,
                           [&](std::size_t, std::size_t begin,
                               std::size_t end) { expose_range(begin, end); });
  } else {
    expose_range(0, target_indices.size());
  }
  workspace.diagnostics.exposure_ms =
      elapsed_ms(exposure_begin, std::chrono::steady_clock::now());

  finalize_diagnostics(workspace, exposed_by_atom, buried_by_atom,
                       neighbor_counts);
  if (options.emit_panels)
    finalize_panels(workspace, panels_by_atom);

  workspace.diagnostics.build_ms =
      elapsed_ms(build_begin, std::chrono::steady_clock::now());
  return workspace;
}

bool should_use_optimized(const Options &options,
                          std::size_t target_count) noexcept {
  if (options.acceleration == "legacy")
    return false;
  if (options.acceleration == "reordered_padded")
    return true;
  if (options.acceleration == "auto")
    return target_count >= kAutoOptimizedTargetThreshold;
  return false;
}

} // namespace

Workspace build_workspace(std::span<const Atom> atoms, const Options &options,
                          std::span<const std::size_t> targets,
                          TitanExecutor *executor) {
  const std::size_t target_count = targets.empty() ? atoms.size() : targets.size();
  if (should_use_optimized(options, target_count))
    return build_workspace_reordered_padded(atoms, options, targets, executor);
  return build_workspace_legacy(atoms, options, targets, executor);
}

} // namespace TitanSurface
