#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <cstdio>
#endif

#include "analyzer_impl.h"

#include "file_sha256.h"

#include "tabi_types.h"
#include "titan_energy_decomposition.h"
#include "titan_executor.h"
#include "titan_interaction_entropy.h"
#include "titan_sasa.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "analyzer_impl_detail.h"

namespace TitanPBE {

using analyzer_impl_detail::csv_double;
using analyzer_impl_detail::csv_escape;
using analyzer_impl_detail::csv_header_map;
using analyzer_impl_detail::csv_index;
using analyzer_impl_detail::effective_workers;
using analyzer_impl_detail::idx3;
using analyzer_impl_detail::kPi;
using analyzer_impl_detail::split_csv_simple;
using analyzer_impl_detail::trim_copy;

namespace {

enum DelphiFaceAction : std::uint16_t {
  kDelphiFaceUnchanged = 0,
  kDelphiFaceRawVdw = 1,
  kDelphiFaceExposedSolvent = 2,
  kDelphiFaceContactSolvent = 3,
  kDelphiFaceCavityMedia = 4,
  kDelphiFaceSignRemap = 5,
  kDelphiFaceDelphiContactSign = 6,
  kDelphiFaceDelphiCavityRemap = 7,
  kDelphiFaceSignPreserve = 8,
};

const char *delphi_face_action_name(std::uint16_t action) noexcept {
  switch (action) {
  case kDelphiFaceRawVdw:
    return "raw_vdw";
  case kDelphiFaceExposedSolvent:
    return "exposed_solvent";
  case kDelphiFaceContactSolvent:
    return "contact_solvent";
  case kDelphiFaceCavityMedia:
    return "cavity_media";
  case kDelphiFaceSignRemap:
    return "sign_remap";
  case kDelphiFaceDelphiContactSign:
    return "delphi_contact_sign";
  case kDelphiFaceDelphiCavityRemap:
    return "delphi_cavity_remap";
  case kDelphiFaceSignPreserve:
    return "sign_preserve";
  default:
    return "unchanged";
  }
}

struct AtomSpatialBins {
  double min_x{0.0};
  double min_y{0.0};
  double min_z{0.0};
  double cell{1.0};
  std::size_t bx{1};
  std::size_t by{1};
  std::size_t bz{1};
  double max_radius{0.0};
  std::vector<std::vector<std::size_t>> bins;
  bool empty{true};
};

template <typename AtomRange>
AtomSpatialBins build_atom_bins(const AtomRange &atoms, double inflate,
                                double min_cell) {
  AtomSpatialBins out;
  if (atoms.empty())
    return out;

  out.empty = false;
  out.min_x = std::numeric_limits<double>::max();
  out.min_y = std::numeric_limits<double>::max();
  out.min_z = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::lowest();
  double max_y = std::numeric_limits<double>::lowest();
  double max_z = std::numeric_limits<double>::lowest();
  for (const auto &atom : atoms) {
    out.max_radius = std::max(out.max_radius, atom.radius + inflate);
    out.min_x = std::min(out.min_x, atom.x);
    out.min_y = std::min(out.min_y, atom.y);
    out.min_z = std::min(out.min_z, atom.z);
    max_x = std::max(max_x, atom.x);
    max_y = std::max(max_y, atom.y);
    max_z = std::max(max_z, atom.z);
  }
  out.cell = std::max({out.max_radius, min_cell, 1.0e-6});
  out.bx = std::max<std::size_t>(
      1,
      static_cast<std::size_t>(std::floor((max_x - out.min_x) / out.cell)) + 1);
  out.by = std::max<std::size_t>(
      1,
      static_cast<std::size_t>(std::floor((max_y - out.min_y) / out.cell)) + 1);
  out.bz = std::max<std::size_t>(
      1,
      static_cast<std::size_t>(std::floor((max_z - out.min_z) / out.cell)) + 1);
  out.bins.assign(out.bx * out.by * out.bz, {});

  auto bin_index = [&](double x, double y, double z) {
    const std::size_t ix = std::min<std::size_t>(
        out.bx - 1, static_cast<std::size_t>(
                        std::max(0.0, std::floor((x - out.min_x) / out.cell))));
    const std::size_t iy = std::min<std::size_t>(
        out.by - 1, static_cast<std::size_t>(
                        std::max(0.0, std::floor((y - out.min_y) / out.cell))));
    const std::size_t iz = std::min<std::size_t>(
        out.bz - 1, static_cast<std::size_t>(
                        std::max(0.0, std::floor((z - out.min_z) / out.cell))));
    return (iz * out.by + iy) * out.bx + ix;
  };
  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx)
    out.bins[bin_index(atoms[atom_idx].x, atoms[atom_idx].y, atoms[atom_idx].z)]
        .push_back(atom_idx);
  return out;
}

template <typename AtomRange> double max_atom_radius(const AtomRange &atoms) {
  double radius = 0.0;
  for (const auto &atom : atoms)
    radius = std::max(radius, atom.radius);
  return radius;
}

template <typename AtomRange>
bool point_inside_inflated_atom(const AtomSpatialBins &bins,
                                const AtomRange &atoms, double x, double y,
                                double z, double inflate) {
  if (bins.empty)
    return false;
  const std::size_t bix = std::min<std::size_t>(
      bins.bx - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((x - bins.min_x) / bins.cell))));
  const std::size_t biy = std::min<std::size_t>(
      bins.by - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((y - bins.min_y) / bins.cell))));
  const std::size_t biz = std::min<std::size_t>(
      bins.bz - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((z - bins.min_z) / bins.cell))));
  const std::size_t x0 = (bix == 0) ? 0 : bix - 1;
  const std::size_t y0 = (biy == 0) ? 0 : biy - 1;
  const std::size_t z0 = (biz == 0) ? 0 : biz - 1;
  const std::size_t x1 = std::min(bins.bx - 1, bix + 1);
  const std::size_t y1 = std::min(bins.by - 1, biy + 1);
  const std::size_t z1 = std::min(bins.bz - 1, biz + 1);
  for (std::size_t kk = z0; kk <= z1; ++kk) {
    for (std::size_t jj = y0; jj <= y1; ++jj) {
      for (std::size_t ii = x0; ii <= x1; ++ii) {
        const auto &cell_atoms = bins.bins[(kk * bins.by + jj) * bins.bx + ii];
        for (std::size_t atom_idx : cell_atoms) {
          const auto &atom = atoms[atom_idx];
          const double radius = atom.radius + inflate;
          const double dx = x - atom.x;
          const double dy = y - atom.y;
          const double dz = z - atom.z;
          if (dx * dx + dy * dy + dz * dz <= radius * radius)
            return true;
        }
      }
    }
  }
  return false;
}

template <typename AtomRange>
bool point_inside_vdw_half_grid(const AtomSpatialBins &bins,
                                const AtomRange &atoms, double x, double y,
                                double z) {
  if (bins.empty)
    return false;
  const std::size_t bix = std::min<std::size_t>(
      bins.bx - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((x - bins.min_x) / bins.cell))));
  const std::size_t biy = std::min<std::size_t>(
      bins.by - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((y - bins.min_y) / bins.cell))));
  const std::size_t biz = std::min<std::size_t>(
      bins.bz - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((z - bins.min_z) / bins.cell))));
  const std::size_t x0 = (bix == 0) ? 0 : bix - 1;
  const std::size_t y0 = (biy == 0) ? 0 : biy - 1;
  const std::size_t z0 = (biz == 0) ? 0 : biz - 1;
  const std::size_t x1 = std::min(bins.bx - 1, bix + 1);
  const std::size_t y1 = std::min(bins.by - 1, biy + 1);
  const std::size_t z1 = std::min(bins.bz - 1, biz + 1);
  for (std::size_t kk = z0; kk <= z1; ++kk) {
    for (std::size_t jj = y0; jj <= y1; ++jj) {
      for (std::size_t ii = x0; ii <= x1; ++ii) {
        const auto &cell_atoms = bins.bins[(kk * bins.by + jj) * bins.bx + ii];
        for (std::size_t atom_idx : cell_atoms) {
          const auto &atom = atoms[atom_idx];
          const double radius = atom.radius;
          const double dx = x - atom.x;
          const double dy = y - atom.y;
          const double dz = z - atom.z;
          if (dx * dx + dy * dy + dz * dz < radius * radius)
            return true;
        }
      }
    }
  }
  return false;
}

template <typename AtomRange>
double segment_inside_fraction(const AtomSpatialBins &bins,
                               const AtomRange &atoms, double x0, double y0,
                               double z0, double x1, double y1, double z1,
                               double inflate) {
  if (bins.empty)
    return 0.0;

  const double sx = x1 - x0;
  const double sy = y1 - y0;
  const double sz = z1 - z0;
  const double a = sx * sx + sy * sy + sz * sz;
  if (a <= 0.0)
    return 0.0;

  const double pad = bins.max_radius + 1.0e-12;
  auto clamp_cell = [](double value, std::size_t n) {
    return static_cast<std::size_t>(
        std::clamp<double>(std::floor(value), 0.0, static_cast<double>(n - 1)));
  };
  const std::size_t ix0 =
      clamp_cell((std::min(x0, x1) - pad - bins.min_x) / bins.cell, bins.bx);
  const std::size_t ix1 =
      clamp_cell((std::max(x0, x1) + pad - bins.min_x) / bins.cell, bins.bx);
  const std::size_t iy0 =
      clamp_cell((std::min(y0, y1) - pad - bins.min_y) / bins.cell, bins.by);
  const std::size_t iy1 =
      clamp_cell((std::max(y0, y1) + pad - bins.min_y) / bins.cell, bins.by);
  const std::size_t iz0 =
      clamp_cell((std::min(z0, z1) - pad - bins.min_z) / bins.cell, bins.bz);
  const std::size_t iz1 =
      clamp_cell((std::max(z0, z1) + pad - bins.min_z) / bins.cell, bins.bz);

  std::vector<std::pair<double, double>> intervals;
  for (std::size_t iz = iz0; iz <= iz1; ++iz) {
    for (std::size_t iy = iy0; iy <= iy1; ++iy) {
      for (std::size_t ix = ix0; ix <= ix1; ++ix) {
        const auto &cell_atoms = bins.bins[(iz * bins.by + iy) * bins.bx + ix];
        for (std::size_t atom_idx : cell_atoms) {
          const auto &atom = atoms[atom_idx];
          const double radius = atom.radius + inflate;
          const double px = x0 - atom.x;
          const double py = y0 - atom.y;
          const double pz = z0 - atom.z;
          const double b = 2.0 * (px * sx + py * sy + pz * sz);
          const double c = px * px + py * py + pz * pz - radius * radius;
          const double disc = b * b - 4.0 * a * c;
          if (disc < 0.0)
            continue;
          const double root = std::sqrt(std::max(0.0, disc));
          double t0 = (-b - root) / (2.0 * a);
          double t1 = (-b + root) / (2.0 * a);
          if (t1 < 0.0 || t0 > 1.0)
            continue;
          t0 = std::clamp(t0, 0.0, 1.0);
          t1 = std::clamp(t1, 0.0, 1.0);
          if (t1 > t0)
            intervals.emplace_back(t0, t1);
        }
      }
    }
  }
  if (intervals.empty())
    return 0.0;

  std::sort(intervals.begin(), intervals.end());
  double covered = 0.0;
  double begin = intervals.front().first;
  double end = intervals.front().second;
  for (std::size_t i = 1; i < intervals.size(); ++i) {
    if (intervals[i].first <= end) {
      end = std::max(end, intervals[i].second);
    } else {
      covered += end - begin;
      begin = intervals[i].first;
      end = intervals[i].second;
    }
  }
  covered += end - begin;
  return std::clamp(covered, 0.0, 1.0);
}

} // namespace

AnalyzerImpl::DelphiVdwMediaMap
AnalyzerImpl::build_delphi_vdw_media_map(const GridShape &shape,
                                         const std::vector<Atom> &surface_atoms,
                                         TitanExecutor &executor) const {
  DelphiVdwMediaMap out;
  out.face_encoded.assign(shape.n * 3, 0);
  out.face_media.assign(shape.n * 3, 0);
  out.face_owner.assign(shape.n * 3, 0);
  out.face_debug.assign(shape.n * 3, {});
  out.audit.enabled = "raw";
  if (shape.n == 0 || surface_atoms.empty())
    return out;

  const std::int32_t delphi_epsilon_dim =
      static_cast<std::int32_t>(surface_atoms.size() + 2);
  auto face_index = [](std::size_t id, int axis) {
    return id * 3 + static_cast<std::size_t>(axis);
  };
  auto decode_media_from_value = [&](std::int32_t value) -> std::int32_t {
    return std::abs(value) / delphi_epsilon_dim;
  };
  auto decode_entity_from_value = [&](std::int32_t value) -> std::int32_t {
    if (value == 0)
      return 0;
    return value % delphi_epsilon_dim;
  };
  auto encode_delphi_face = [&](std::int32_t entity,
                                std::int32_t media) -> std::int32_t {
    const std::int32_t sign = entity < 0 ? -1 : 1;
    return sign * (std::abs(entity) + std::abs(media) * delphi_epsilon_dim);
  };
  auto sync_face_from_encoded = [&](std::size_t fid) {
    if (fid >= out.face_encoded.size())
      return;
    const std::int32_t value = out.face_encoded[fid];
    out.face_media[fid] = decode_media_from_value(value);
    out.face_owner[fid] = decode_entity_from_value(value);
    if (fid < out.face_debug.size())
      out.face_debug[fid].final_encoded = value;
  };
  auto set_face_encoded = [&](std::size_t fid, std::int32_t value) {
    if (fid >= out.face_encoded.size())
      return;
    out.face_encoded[fid] = value;
    sync_face_from_encoded(fid);
  };

  constexpr std::array<std::array<int, 3>, 27> kDelphiNeighborOffsets{{
      {{0, 0, 0}},   {{-1, 0, 0}}, {{1, 0, 0}},   {{0, -1, 0}},
      {{0, 1, 0}},   {{0, 0, -1}}, {{0, 0, 1}},   {{1, 0, 1}},
      {{-1, 0, 1}},  {{0, 1, 1}},  {{0, -1, 1}},  {{-1, -1, 0}},
      {{1, -1, 0}},  {{1, 1, 0}},  {{-1, 1, 0}},  {{-1, 0, -1}},
      {{1, 0, -1}},  {{0, 1, -1}}, {{0, -1, -1}}, {{-1, -1, -1}},
      {{1, -1, -1}}, {{1, 1, -1}}, {{-1, 1, -1}}, {{-1, 1, 1}},
      {{1, 1, 1}},   {{1, -1, 1}}, {{-1, -1, 1}},
  }};
  const double grid_spacing_extension = 0.5 * shape.h * std::sqrt(3.0);
  const double delphi_voxel_side = max_atom_radius(surface_atoms) +
                                   args_.ion_radius + grid_spacing_extension;
  double atom_min_x = surface_atoms.front().x;
  double atom_min_y = surface_atoms.front().y;
  double atom_min_z = surface_atoms.front().z;
  double atom_max_x = surface_atoms.front().x;
  double atom_max_y = surface_atoms.front().y;
  double atom_max_z = surface_atoms.front().z;
  for (const Atom &atom : surface_atoms) {
    atom_min_x = std::min(atom_min_x, atom.x);
    atom_min_y = std::min(atom_min_y, atom.y);
    atom_min_z = std::min(atom_min_z, atom.z);
    atom_max_x = std::max(atom_max_x, atom.x);
    atom_max_y = std::max(atom_max_y, atom.y);
    atom_max_z = std::max(atom_max_z, atom.z);
  }
  const double delphi_voxel_padding = delphi_voxel_side + 0.1;
  const double raw_voxel_origin_x = atom_min_x - delphi_voxel_padding;
  const double raw_voxel_origin_y = atom_min_y - delphi_voxel_padding;
  const double raw_voxel_origin_z = atom_min_z - delphi_voxel_padding;
  const auto raw_voxel_count = [&](double lo, double hi) -> std::size_t {
    return static_cast<std::size_t>(std::max<double>(
        0.0,
        std::floor((hi + delphi_voxel_padding - (lo - delphi_voxel_padding)) /
                   delphi_voxel_side)));
  };
  const std::size_t raw_vx = raw_voxel_count(atom_min_x, atom_max_x);
  const std::size_t raw_vy = raw_voxel_count(atom_min_y, atom_max_y);
  const std::size_t raw_vz = raw_voxel_count(atom_min_z, atom_max_z);
  const std::size_t raw_sx = raw_vx + 1;
  const std::size_t raw_sy = raw_vy + 1;
  const std::size_t raw_sz = raw_vz + 1;
  const double raw_voxel_scale = 1.0 / std::max(1.0e-12, delphi_voxel_side);
  auto raw_voxel_flat = [&](std::size_t vx, std::size_t vy, std::size_t vz) {
    return (vz * raw_sy + vy) * raw_sx + vx;
  };
  auto raw_atom_central = [&](const Atom &atom, std::size_t &vx,
                              std::size_t &vy, std::size_t &vz) {
    auto coord = [&](double value, double origin,
                     std::size_t high) -> std::size_t {
      const double scaled = (value - origin) * raw_voxel_scale;
      if (scaled <= 0.0)
        return 0;
      return std::min<std::size_t>(high, static_cast<std::size_t>(scaled));
    };
    vx = coord(atom.x, raw_voxel_origin_x, raw_vx);
    vy = coord(atom.y, raw_voxel_origin_y, raw_vy);
    vz = coord(atom.z, raw_voxel_origin_z, raw_vz);
  };
  std::vector<std::vector<std::size_t>> raw_voxel_atoms(raw_sx * raw_sy *
                                                        raw_sz);
  for (const auto &offset : kDelphiNeighborOffsets) {
    for (std::size_t atom_idx = 0; atom_idx < surface_atoms.size();
         ++atom_idx) {
      const Atom &atom = surface_atoms[atom_idx];
      if (atom.radius <= 0.0)
        continue;
      std::size_t cx = 0, cy = 0, cz = 0;
      raw_atom_central(atom, cx, cy, cz);
      const int tx = static_cast<int>(cx) + offset[0];
      const int ty = static_cast<int>(cy) + offset[1];
      const int tz = static_cast<int>(cz) + offset[2];
      if (tx < 0 || ty < 0 || tz < 0 || tx > static_cast<int>(raw_vx) ||
          ty > static_cast<int>(raw_vy) || tz > static_cast<int>(raw_vz))
        continue;
      raw_voxel_atoms[raw_voxel_flat(static_cast<std::size_t>(tx),
                                     static_cast<std::size_t>(ty),
                                     static_cast<std::size_t>(tz))]
          .push_back(atom_idx);
    }
  }
  auto raw_grid_voxel = [&](double x, double y, double z, std::size_t &vx,
                            std::size_t &vy, std::size_t &vz) {
    auto coord = [&](double value, double origin,
                     std::size_t high) -> std::size_t {
      const double scaled = (value - origin) * raw_voxel_scale;
      if (scaled <= 0.0)
        return 0;
      return std::min<std::size_t>(high, static_cast<std::size_t>(scaled));
    };
    vx = coord(x, raw_voxel_origin_x, raw_vx);
    vy = coord(y, raw_voxel_origin_y, raw_vy);
    vz = coord(z, raw_voxel_origin_z, raw_vz);
  };
  const std::size_t workers = effective_workers(executor, shape.nz);
  executor.parallel_for(
      shape.nz, workers, 1,
      [&](std::size_t, std::size_t z_begin, std::size_t z_end) {
        for (std::size_t k = z_begin; k < z_end; ++k) {
          const double z = shape.oz + static_cast<double>(k) * shape.h;
          for (std::size_t j = 0; j < shape.ny; ++j) {
            const double y = shape.oy + static_cast<double>(j) * shape.h;
            for (std::size_t i = 0; i < shape.nx; ++i) {
              const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
              const double x = shape.ox + static_cast<double>(i) * shape.h;
              const bool valid_x = i + 1 < shape.nx;
              const bool valid_y = j + 1 < shape.ny;
              const bool valid_z = k + 1 < shape.nz;
              bool done_x = !valid_x;
              bool done_y = !valid_y;
              bool done_z = !valid_z;
              std::size_t gvx = 0, gvy = 0, gvz = 0;
              raw_grid_voxel(x, y, z, gvx, gvy, gvz);
              for (const auto &offset : kDelphiNeighborOffsets) {
                const int tx = static_cast<int>(gvx) + offset[0];
                const int ty = static_cast<int>(gvy) + offset[1];
                const int tz = static_cast<int>(gvz) + offset[2];
                if (tx < 0 || ty < 0 || tz < 0 ||
                    tx > static_cast<int>(raw_vx) ||
                    ty > static_cast<int>(raw_vy) ||
                    tz > static_cast<int>(raw_vz))
                  continue;
                const auto &bucket = raw_voxel_atoms[raw_voxel_flat(
                    static_cast<std::size_t>(tx), static_cast<std::size_t>(ty),
                    static_cast<std::size_t>(tz))];
                for (std::size_t atom_idx : bucket) {
                  const Atom &atom = surface_atoms[atom_idx];
                  const double rdx = x - atom.x;
                  const double rdy = y - atom.y;
                  const double rdz = z - atom.z;
                  const double radius2 = atom.radius * atom.radius;
                  const std::int32_t owner =
                      static_cast<std::int32_t>(atom_idx + 2);
                  if (!done_x) {
                    const double dx = rdx + 0.5 * shape.h;
                    if (dx * dx + rdy * rdy + rdz * rdz < radius2) {
                      out.face_encoded[face_index(id, 0)] =
                          encode_delphi_face(owner, std::int32_t{1});
                      done_x = true;
                    }
                  }
                  if (!done_y) {
                    const double dy = rdy + 0.5 * shape.h;
                    if (rdx * rdx + dy * dy + rdz * rdz < radius2) {
                      out.face_encoded[face_index(id, 1)] =
                          encode_delphi_face(owner, std::int32_t{1});
                      done_y = true;
                    }
                  }
                  if (!done_z) {
                    const double dz = rdz + 0.5 * shape.h;
                    if (rdx * rdx + rdy * rdy + dz * dz < radius2) {
                      out.face_encoded[face_index(id, 2)] =
                          encode_delphi_face(owner, std::int32_t{1});
                      done_z = true;
                    }
                  }
                  if (done_x && done_y && done_z)
                    break;
                }
                if (done_x && done_y && done_z)
                  break;
              }
            }
          }
        }
      });

  for (std::size_t fid = 0; fid < out.face_media.size(); ++fid) {
    sync_face_from_encoded(fid);
    DelphiFaceDebug &debug = out.face_debug[fid];
    debug.raw_media = out.face_media[fid];
    debug.raw_owner = out.face_owner[fid];
    debug.raw_encoded = out.face_encoded[fid];
    debug.action =
        debug.raw_media != 0 ? kDelphiFaceRawVdw : kDelphiFaceUnchanged;
  }

  const std::size_t plane = shape.nx * shape.ny;
  auto positive_face = [&](std::size_t i, std::size_t j, std::size_t k,
                           int axis) -> std::int32_t {
    if (i >= shape.nx || j >= shape.ny || k >= shape.nz)
      return 0;
    const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
    return out.face_media[face_index(id, axis)];
  };
  auto face_media6 = [&](std::size_t i, std::size_t j, std::size_t k,
                         std::array<std::int32_t, 6> &faces) {
    const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
    faces[0] = out.face_media[face_index(id, 0)];
    faces[1] = out.face_media[face_index(id, 1)];
    faces[2] = out.face_media[face_index(id, 2)];
    faces[3] = i > 0 ? out.face_media[face_index(id - 1, 0)] : 0;
    faces[4] = j > 0 ? out.face_media[face_index(id - shape.nx, 1)] : 0;
    faces[5] = k > 0 ? out.face_media[face_index(id - plane, 2)] : 0;
  };
  auto collect_boundary_voxels = [&]() {
    std::vector<std::size_t> voxels;
    if (shape.nx < 3 || shape.ny < 3 || shape.nz < 3)
      return voxels;
    std::array<std::int32_t, 6> faces{};
    for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
      for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
        for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
          face_media6(i, j, k, faces);
          bool has_solute = false;
          bool has_solvent = false;
          for (std::int32_t value : faces) {
            has_solute = has_solute || value != 0;
            has_solvent = has_solvent || value == 0;
          }
          if (has_solute && has_solvent)
            voxels.push_back(idx3(i, j, k, shape.nx, shape.ny));
        }
      }
    }
    return voxels;
  };

  std::vector<std::size_t> boundary_voxels = collect_boundary_voxels();
  out.audit.boundary_initial = boundary_voxels.size();
  if (args_.test_vdwms_remap == "off" || args_.probe_radius <= 0.0) {
    out.audit.enabled = args_.test_vdwms_remap == "off" ? "off" : "zero_probe";
    out.audit.boundary_final = out.audit.boundary_initial;
    return out;
  }

  out.audit.enabled = "auto";
  const AtomSpatialBins probe_bins = build_atom_bins(
      surface_atoms, args_.probe_radius, std::max(shape.h, delphi_voxel_side));
  auto probe_surface_occluded = [&](double x, double y, double z,
                                    std::size_t closest_atom_idx) {
    if (probe_bins.empty || closest_atom_idx >= surface_atoms.size())
      return false;
    const Atom &closest = surface_atoms[closest_atom_idx];
    const double vx = x - closest.x;
    const double vy = y - closest.y;
    const double vz = z - closest.z;
    const double distance = std::sqrt(vx * vx + vy * vy + vz * vz);
    if (distance <= 1.0e-12)
      return true;
    const double surface_radius = closest.radius + args_.probe_radius;
    const double sx = closest.x + vx * (surface_radius / distance);
    const double sy = closest.y + vy * (surface_radius / distance);
    const double sz = closest.z + vz * (surface_radius / distance);
    const int radius_bins =
        static_cast<int>(std::ceil(std::max(probe_bins.max_radius, shape.h) /
                                   std::max(1.0e-12, probe_bins.cell))) +
        1;
    const int bx0 =
        static_cast<int>(std::floor((sx - probe_bins.min_x) / probe_bins.cell));
    const int by0 =
        static_cast<int>(std::floor((sy - probe_bins.min_y) / probe_bins.cell));
    const int bz0 =
        static_cast<int>(std::floor((sz - probe_bins.min_z) / probe_bins.cell));
    constexpr double kSasShrink = 0.99999;
    for (int dz = -radius_bins; dz <= radius_bins; ++dz) {
      const int bz = bz0 + dz;
      if (bz < 0 || bz >= static_cast<int>(probe_bins.bz))
        continue;
      for (int dy = -radius_bins; dy <= radius_bins; ++dy) {
        const int by = by0 + dy;
        if (by < 0 || by >= static_cast<int>(probe_bins.by))
          continue;
        for (int dx = -radius_bins; dx <= radius_bins; ++dx) {
          const int bx = bx0 + dx;
          if (bx < 0 || bx >= static_cast<int>(probe_bins.bx))
            continue;
          const auto &bucket =
              probe_bins.bins[(static_cast<std::size_t>(bz) * probe_bins.by +
                               static_cast<std::size_t>(by)) *
                                  probe_bins.bx +
                              static_cast<std::size_t>(bx)];
          for (std::size_t atom_idx : bucket) {
            if (atom_idx == closest_atom_idx)
              continue;
            const Atom &atom = surface_atoms[atom_idx];
            const double radius = atom.radius + args_.probe_radius;
            const double ddx = sx - atom.x;
            const double ddy = sy - atom.y;
            const double ddz = sz - atom.z;
            const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if (r2 < radius * radius * kSasShrink)
              return true;
          }
        }
      }
    }
    return false;
  };
  std::vector<unsigned char> vdwms_accessible_atoms(surface_atoms.size(), 1U);
  struct ClosestSurfaceAtom {
    std::ptrdiff_t index{-1};
    double surface_dist{std::numeric_limits<double>::infinity()};
  };
  auto closest_surface_atom = [&](double x, double y,
                                  double z) -> ClosestSurfaceAtom {
    ClosestSurfaceAtom best;
    if (probe_bins.empty)
      return best;
    const int radius_bins =
        static_cast<int>(std::ceil(std::max(probe_bins.max_radius, shape.h) /
                                   std::max(1.0e-12, probe_bins.cell))) +
        1;
    const int bx0 =
        static_cast<int>(std::floor((x - probe_bins.min_x) / probe_bins.cell));
    const int by0 =
        static_cast<int>(std::floor((y - probe_bins.min_y) / probe_bins.cell));
    const int bz0 =
        static_cast<int>(std::floor((z - probe_bins.min_z) / probe_bins.cell));
    for (int dz = -radius_bins; dz <= radius_bins; ++dz) {
      const int bz = bz0 + dz;
      if (bz < 0 || bz >= static_cast<int>(probe_bins.bz))
        continue;
      for (int dy = -radius_bins; dy <= radius_bins; ++dy) {
        const int by = by0 + dy;
        if (by < 0 || by >= static_cast<int>(probe_bins.by))
          continue;
        for (int dx = -radius_bins; dx <= radius_bins; ++dx) {
          const int bx = bx0 + dx;
          if (bx < 0 || bx >= static_cast<int>(probe_bins.bx))
            continue;
          const auto &bucket =
              probe_bins.bins[(static_cast<std::size_t>(bz) * probe_bins.by +
                               static_cast<std::size_t>(by)) *
                                  probe_bins.bx +
                              static_cast<std::size_t>(bx)];
          for (std::size_t atom_idx : bucket) {
            if (atom_idx < vdwms_accessible_atoms.size() &&
                vdwms_accessible_atoms[atom_idx] == 0U)
              continue;
            const Atom &atom = surface_atoms[atom_idx];
            const double ddx = x - atom.x;
            const double ddy = y - atom.y;
            const double ddz = z - atom.z;
            const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
            const double surface_distance = std::sqrt(r2) - atom.radius;
            if (surface_distance < best.surface_dist) {
              best.surface_dist = surface_distance;
              best.index = static_cast<std::ptrdiff_t>(atom_idx);
            }
          }
        }
      }
    }
    return best;
  };
  struct ExposedPointBins {
    bool empty{true};
    double min_x{0.0};
    double min_y{0.0};
    double min_z{0.0};
    double cell{1.0};
    std::size_t bx{0};
    std::size_t by{0};
    std::size_t bz{0};
    std::vector<std::array<double, 3>> points;
    std::vector<std::vector<std::size_t>> bins;
  };
  auto build_exposed_point_bins = [&]() {
    ExposedPointBins bins;
    if (surface_atoms.empty() || args_.probe_radius <= 0.0)
      return bins;
    constexpr int kInitialVertexCount = 12;
    constexpr int kRefinementLevel = 5;
    constexpr int kMaxVertices = 520;
    constexpr int kMaxEdges = 1040;
    constexpr double kSasShrink = 0.99999;
    auto probe_point_occupier = [&](double sx, double sy,
                                    double sz) -> std::int32_t {
      for (std::size_t atom_idx = 0; atom_idx < surface_atoms.size();
           ++atom_idx) {
        const Atom &atom = surface_atoms[atom_idx];
        const double radius = atom.radius + args_.probe_radius;
        const double ddx = sx - atom.x;
        const double ddy = sy - atom.y;
        const double ddz = sz - atom.z;
        const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
        if (r2 < radius * radius * kSasShrink)
          return static_cast<std::int32_t>(atom_idx + 1);
      }
      return 0;
    };
    struct SasEdge {
      int first{0};
      int second{0};
    };
    std::vector<std::array<double, 3>> vertex_array(kMaxVertices + 1);
    std::vector<SasEdge> edge_array(kMaxEdges + 1);
    std::vector<int> edges(kMaxEdges + 1, 0);
    const double angle_increment =
        2.0 * kPi / static_cast<double>(kInitialVertexCount);
    for (int vertex_index = 1; vertex_index <= kInitialVertexCount;
         ++vertex_index) {
      const double angle =
          static_cast<double>(vertex_index - 1) * angle_increment;
      vertex_array[static_cast<std::size_t>(vertex_index)] = {
          std::cos(angle), std::sin(angle), 0.0};
      edge_array[static_cast<std::size_t>(vertex_index)] = {
          vertex_index,
          vertex_index != kInitialVertexCount ? vertex_index + 1 : 1};
    }
    int current_vertex_count = kInitialVertexCount;
    int current_edge_count = kInitialVertexCount;
    int start_edge_index = 1;
    int end_edge_index = 0;
    for (int level = 1; level <= kRefinementLevel; ++level) {
      start_edge_index = end_edge_index + 1;
      end_edge_index = current_edge_count;
      for (int edge_index = start_edge_index; edge_index <= end_edge_index;
           ++edge_index) {
        const int v1 = edge_array[static_cast<std::size_t>(edge_index)].first;
        const int v2 = edge_array[static_cast<std::size_t>(edge_index)].second;
        std::array<double, 3> midpoint{
            vertex_array[static_cast<std::size_t>(v1)][0] +
                vertex_array[static_cast<std::size_t>(v2)][0],
            vertex_array[static_cast<std::size_t>(v1)][1] +
                vertex_array[static_cast<std::size_t>(v2)][1],
            vertex_array[static_cast<std::size_t>(v1)][2] +
                vertex_array[static_cast<std::size_t>(v2)][2]};
        const double magnitude =
            std::sqrt(midpoint[0] * midpoint[0] + midpoint[1] * midpoint[1] +
                      midpoint[2] * midpoint[2]);
        if (magnitude <= 1.0e-30)
          continue;
        ++current_vertex_count;
        vertex_array[static_cast<std::size_t>(current_vertex_count)] = {
            midpoint[0] / magnitude, midpoint[1] / magnitude,
            midpoint[2] / magnitude};
        ++current_edge_count;
        edges[static_cast<std::size_t>(edge_index)] = current_edge_count;
        edge_array[static_cast<std::size_t>(current_edge_count)] = {
            v1, current_vertex_count};
        ++current_edge_count;
        edge_array[static_cast<std::size_t>(current_edge_count)] = {
            current_vertex_count, v2};
      }
    }
    current_edge_count = end_edge_index;
    for (int edge_index = start_edge_index; edge_index <= current_edge_count;
         ++edge_index)
      edges[static_cast<std::size_t>(edge_index)] = -1;

    bins.points.reserve(surface_atoms.size() * 64);
    std::vector<unsigned char> has_pair(surface_atoms.size(), 0U);
    std::vector<unsigned char> accessible(surface_atoms.size(), 0U);
    for (std::size_t a = 0; a < surface_atoms.size(); ++a) {
      const Atom &atom_a = surface_atoms[a];
      const double ra = atom_a.radius + args_.probe_radius;
      if (atom_a.radius <= 0.0 || ra <= 0.0)
        continue;
      for (std::size_t b = a + 1; b < surface_atoms.size(); ++b) {
        const Atom &atom_b = surface_atoms[b];
        const double rb = atom_b.radius + args_.probe_radius;
        if (atom_b.radius <= 0.0 || rb <= 0.0)
          continue;
        const double dx = atom_b.x - atom_a.x;
        const double dy = atom_b.y - atom_a.y;
        const double dz = atom_b.z - atom_a.z;
        const double d2 = dx * dx + dy * dy + dz * dz;
        const double sum_r = ra + rb;
        const double diff_r = std::abs(ra - rb);
        if (sum_r * sum_r - d2 <= 0.01 || d2 <= diff_r * diff_r ||
            d2 <= 1.0e-16)
          continue;
        has_pair[a] = 1U;
        has_pair[b] = 1U;
        const double d = std::sqrt(d2);
        const double prefactor = 1.0 + (ra * ra - rb * rb) / d2;
        const double cx = atom_a.x + 0.5 * prefactor * dx;
        const double cy = atom_a.y + 0.5 * prefactor * dy;
        const double cz = atom_a.z + 0.5 * prefactor * dz;
        const double term1 = std::max(0.0, sum_r * sum_r - d2);
        const double term2 = std::max(0.0, d2 - diff_r * diff_r);
        const double circle_r = 0.5 * std::sqrt(term1) * std::sqrt(term2) / d;
        if (circle_r <= 0.0)
          continue;
        std::array<double, 3> circle_x{1.0, 0.0, 0.0};
        std::array<double, 3> circle_y{0.0, 1.0, 0.0};
        const double xy_projection = std::sqrt(dx * dx + dy * dy);
        if (xy_projection > 1.0e-8) {
          const double tangential_x = -dy / xy_projection;
          const double tangential_y = dx / xy_projection;
          const double cosine_theta = dz / d;
          const double sine_theta =
              std::sqrt(std::max(0.0, 1.0 - cosine_theta * cosine_theta));
          const double one_minus_cosine = 1.0 - cosine_theta;
          const double temp = one_minus_cosine * tangential_x;
          const double sine_tangential_x = sine_theta * tangential_x;
          const double sine_tangential_y = sine_theta * tangential_y;
          circle_x = {temp * tangential_x + cosine_theta, temp * tangential_y,
                      -sine_tangential_y};
          circle_y = {temp * tangential_y,
                      one_minus_cosine * tangential_y * tangential_y +
                          cosine_theta,
                      sine_tangential_x};
        }
        auto circle_point = [&](const std::array<double, 3> &unit_point) {
          return std::array<double, 3>{
              cx + circle_r *
                       (unit_point[0] * circle_x[0] +
                        unit_point[1] * circle_y[0] + unit_point[2] * 0.0),
              cy + circle_r *
                       (unit_point[0] * circle_x[1] +
                        unit_point[1] * circle_y[1] + unit_point[2] * 0.0),
              cz + circle_r *
                       (unit_point[0] * circle_x[2] +
                        unit_point[1] * circle_y[2] + unit_point[2] * 1.0)};
        };
        std::vector<std::int32_t> vertex_occupation(kMaxVertices + 1, 0);
        std::vector<int> edge_state(kMaxEdges + 1, 0);
        int num_visible_vertices = 0;
        for (int vertex_index = 1; vertex_index <= kInitialVertexCount;
             ++vertex_index) {
          const auto point = circle_point(
              vertex_array[static_cast<std::size_t>(vertex_index)]);
          const std::int32_t occupation =
              probe_point_occupier(point[0], point[1], point[2]);
          vertex_occupation[static_cast<std::size_t>(vertex_index)] =
              occupation;
          if (occupation == 0) {
            ++num_visible_vertices;
            bins.points.push_back(point);
            vertex_occupation[static_cast<std::size_t>(vertex_index)] = 0;
          }
        }

        int edge_state_count = 0;
        for (int edge_index = kInitialVertexCount; edge_index >= 1;
             --edge_index) {
          const int v1 = edge_array[static_cast<std::size_t>(edge_index)].first;
          const int v2 =
              edge_array[static_cast<std::size_t>(edge_index)].second;
          const std::int32_t occ1 =
              vertex_occupation[static_cast<std::size_t>(v1)];
          const std::int32_t occ2 =
              vertex_occupation[static_cast<std::size_t>(v2)];
          if (occ1 > 0 && occ1 == occ2)
            continue;
          ++edge_state_count;
          edge_state[static_cast<std::size_t>(edge_state_count)] = edge_index;
        }

        while (edge_state_count > 0) {
          const int edge_index =
              edge_state[static_cast<std::size_t>(edge_state_count)];
          --edge_state_count;
          const int endpoint1 =
              edge_array[static_cast<std::size_t>(edge_index)].first;
          const int endpoint2 =
              edge_array[static_cast<std::size_t>(edge_index)].second;
          const std::int32_t vertex_index1 =
              vertex_occupation[static_cast<std::size_t>(endpoint1)];
          const std::int32_t vertex_index2 =
              vertex_occupation[static_cast<std::size_t>(endpoint2)];
          const int child_edge = edges[static_cast<std::size_t>(edge_index)];
          if (vertex_index1 > static_cast<std::int32_t>(surface_atoms.size()) ||
              vertex_index2 > static_cast<std::int32_t>(surface_atoms.size()))
            continue;
          const int refined_vertex_index = edge_index + kInitialVertexCount;
          if (refined_vertex_index < 1 ||
              refined_vertex_index > current_vertex_count)
            continue;
          const auto point = circle_point(
              vertex_array[static_cast<std::size_t>(refined_vertex_index)]);
          const std::int32_t occ_mid =
              probe_point_occupier(point[0], point[1], point[2]);
          bool handled = false;
          if (vertex_index1 != 0 && occ_mid == vertex_index1) {
            vertex_occupation[static_cast<std::size_t>(refined_vertex_index)] =
                vertex_index1;
            if (child_edge > 0) {
              ++edge_state_count;
              edge_state[static_cast<std::size_t>(edge_state_count)] =
                  child_edge + 1;
            }
            handled = true;
          }
          if (!handled && vertex_index2 != 0 && occ_mid == vertex_index2) {
            vertex_occupation[static_cast<std::size_t>(refined_vertex_index)] =
                vertex_index2;
            if (child_edge > 0) {
              ++edge_state_count;
              edge_state[static_cast<std::size_t>(edge_state_count)] =
                  child_edge;
            }
            handled = true;
          }
          if (!handled && occ_mid != 0) {
            vertex_occupation[static_cast<std::size_t>(refined_vertex_index)] =
                occ_mid;
            if (child_edge > 0) {
              ++edge_state_count;
              edge_state[static_cast<std::size_t>(edge_state_count)] =
                  child_edge + 1;
              ++edge_state_count;
              edge_state[static_cast<std::size_t>(edge_state_count)] =
                  child_edge;
            }
            handled = true;
          }
          if (handled)
            continue;

          ++num_visible_vertices;
          vertex_occupation[static_cast<std::size_t>(refined_vertex_index)] = 0;
          bins.points.push_back(point);
          if (child_edge > 0) {
            const int child_a = child_edge;
            const int child_b = child_edge + 1;
            if ((edges[static_cast<std::size_t>(child_b)] > 0) ||
                vertex_index2 > 0) {
              ++edge_state_count;
              edge_state[static_cast<std::size_t>(edge_state_count)] = child_b;
            }
            if ((edges[static_cast<std::size_t>(child_a)] > 0) ||
                vertex_index1 > 0) {
              ++edge_state_count;
              edge_state[static_cast<std::size_t>(edge_state_count)] = child_a;
            }
          }
        }
        if (num_visible_vertices > 0) {
          accessible[a] = 1U;
          accessible[b] = 1U;
        }
      }
    }
    for (std::size_t atom_idx = 0; atom_idx < surface_atoms.size();
         ++atom_idx) {
      if (has_pair[atom_idx] == 0U)
        accessible[atom_idx] = 1U;
    }
    vdwms_accessible_atoms = std::move(accessible);
    if (bins.points.empty())
      return bins;
    double max_x = bins.points.front()[0];
    double max_y = bins.points.front()[1];
    double max_z = bins.points.front()[2];
    bins.min_x = max_x;
    bins.min_y = max_y;
    bins.min_z = max_z;
    for (const auto &point : bins.points) {
      bins.min_x = std::min(bins.min_x, point[0]);
      bins.min_y = std::min(bins.min_y, point[1]);
      bins.min_z = std::min(bins.min_z, point[2]);
      max_x = std::max(max_x, point[0]);
      max_y = std::max(max_y, point[1]);
      max_z = std::max(max_z, point[2]);
    }
    bins.cell = std::max(shape.h, args_.probe_radius);
    bins.bx = std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(
                                           (max_x - bins.min_x) / bins.cell)) +
                                           1);
    bins.by = std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(
                                           (max_y - bins.min_y) / bins.cell)) +
                                           1);
    bins.bz = std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(
                                           (max_z - bins.min_z) / bins.cell)) +
                                           1);
    bins.bins.assign(bins.bx * bins.by * bins.bz, {});
    for (std::size_t point_idx = 0; point_idx < bins.points.size();
         ++point_idx) {
      const auto &point = bins.points[point_idx];
      const std::size_t ix = std::min<std::size_t>(
          bins.bx - 1,
          static_cast<std::size_t>(
              std::max(0.0, std::floor((point[0] - bins.min_x) / bins.cell))));
      const std::size_t iy = std::min<std::size_t>(
          bins.by - 1,
          static_cast<std::size_t>(
              std::max(0.0, std::floor((point[1] - bins.min_y) / bins.cell))));
      const std::size_t iz = std::min<std::size_t>(
          bins.bz - 1,
          static_cast<std::size_t>(
              std::max(0.0, std::floor((point[2] - bins.min_z) / bins.cell))));
      bins.bins[(iz * bins.by + iy) * bins.bx + ix].push_back(point_idx);
    }
    bins.empty = false;
    return bins;
  };
  const ExposedPointBins exposed_bins = build_exposed_point_bins();
  auto nearest_exposed_distance = [&](double x, double y, double z) {
    if (exposed_bins.empty)
      return std::numeric_limits<double>::infinity();
    const int radius_bins =
        static_cast<int>(std::ceil(args_.probe_radius /
                                   std::max(1.0e-12, exposed_bins.cell))) +
        1;
    const int bx0 = static_cast<int>(
        std::floor((x - exposed_bins.min_x) / exposed_bins.cell));
    const int by0 = static_cast<int>(
        std::floor((y - exposed_bins.min_y) / exposed_bins.cell));
    const int bz0 = static_cast<int>(
        std::floor((z - exposed_bins.min_z) / exposed_bins.cell));
    double best = std::numeric_limits<double>::infinity();
    for (int dz = -radius_bins; dz <= radius_bins; ++dz) {
      const int bz = bz0 + dz;
      if (bz < 0 || bz >= static_cast<int>(exposed_bins.bz))
        continue;
      for (int dy = -radius_bins; dy <= radius_bins; ++dy) {
        const int by = by0 + dy;
        if (by < 0 || by >= static_cast<int>(exposed_bins.by))
          continue;
        for (int dx = -radius_bins; dx <= radius_bins; ++dx) {
          const int bx = bx0 + dx;
          if (bx < 0 || bx >= static_cast<int>(exposed_bins.bx))
            continue;
          const auto &bucket =
              exposed_bins
                  .bins[(static_cast<std::size_t>(bz) * exposed_bins.by +
                         static_cast<std::size_t>(by)) *
                            exposed_bins.bx +
                        static_cast<std::size_t>(bx)];
          for (std::size_t point_idx : bucket) {
            const auto &point = exposed_bins.points[point_idx];
            const double ddx = x - point[0];
            const double ddy = y - point[1];
            const double ddz = z - point[2];
            const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if (r2 < best)
              best = r2;
          }
        }
      }
    }
    return std::sqrt(best);
  };
  auto nearest_exposed_point = [&](double x, double y, double z,
                                   std::array<double, 3> &best_point) {
    if (exposed_bins.empty)
      return std::numeric_limits<double>::infinity();
    const int radius_bins =
        static_cast<int>(std::ceil(args_.probe_radius /
                                   std::max(1.0e-12, exposed_bins.cell))) +
        1;
    const int bx0 = static_cast<int>(
        std::floor((x - exposed_bins.min_x) / exposed_bins.cell));
    const int by0 = static_cast<int>(
        std::floor((y - exposed_bins.min_y) / exposed_bins.cell));
    const int bz0 = static_cast<int>(
        std::floor((z - exposed_bins.min_z) / exposed_bins.cell));
    double best = std::numeric_limits<double>::infinity();
    for (int dz = -radius_bins; dz <= radius_bins; ++dz) {
      const int bz = bz0 + dz;
      if (bz < 0 || bz >= static_cast<int>(exposed_bins.bz))
        continue;
      for (int dy = -radius_bins; dy <= radius_bins; ++dy) {
        const int by = by0 + dy;
        if (by < 0 || by >= static_cast<int>(exposed_bins.by))
          continue;
        for (int dx = -radius_bins; dx <= radius_bins; ++dx) {
          const int bx = bx0 + dx;
          if (bx < 0 || bx >= static_cast<int>(exposed_bins.bx))
            continue;
          const auto &bucket =
              exposed_bins
                  .bins[(static_cast<std::size_t>(bz) * exposed_bins.by +
                         static_cast<std::size_t>(by)) *
                            exposed_bins.bx +
                        static_cast<std::size_t>(bx)];
          for (std::size_t point_idx : bucket) {
            const auto &point = exposed_bins.points[point_idx];
            const double ddx = x - point[0];
            const double ddy = y - point[1];
            const double ddz = z - point[2];
            const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if (r2 < best) {
              best = r2;
              best_point = point;
            }
          }
        }
      }
    }
    return std::sqrt(best);
  };

  struct DelphiFaceRef {
    std::size_t id{0};
    int axis{0};
    bool valid{false};
  };
  auto face_ref_for_neighbor = [&](std::size_t i, std::size_t j, std::size_t k,
                                   int neighbor) -> DelphiFaceRef {
    const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
    switch (neighbor) {
    case 1:
      return {id, 0, i + 1 < shape.nx};
    case 2:
      return {id, 1, j + 1 < shape.ny};
    case 3:
      return {id, 2, k + 1 < shape.nz};
    case 4:
      return {i > 0 ? id - 1 : id, 0, i > 0};
    case 5:
      return {j > 0 ? id - shape.nx : id, 1, j > 0};
    case 6:
      return {k > 0 ? id - plane : id, 2, k > 0};
    default:
      return {};
    }
  };
  auto face_midpoint = [&](std::size_t i, std::size_t j, std::size_t k,
                           int neighbor) {
    std::array<double, 3> point{shape.ox + static_cast<double>(i) * shape.h,
                                shape.oy + static_cast<double>(j) * shape.h,
                                shape.oz + static_cast<double>(k) * shape.h};
    if (neighbor == 1)
      point[0] += 0.5 * shape.h;
    else if (neighbor == 2)
      point[1] += 0.5 * shape.h;
    else if (neighbor == 3)
      point[2] += 0.5 * shape.h;
    else if (neighbor == 4)
      point[0] -= 0.5 * shape.h;
    else if (neighbor == 5)
      point[1] -= 0.5 * shape.h;
    else if (neighbor == 6)
      point[2] -= 0.5 * shape.h;
    return point;
  };
  auto face_flat = [&](const DelphiFaceRef &ref) {
    return face_index(ref.id, ref.axis);
  };
  auto decode_index = [&](std::size_t id, std::size_t &i, std::size_t &j,
                          std::size_t &k) {
    k = id / plane;
    const std::size_t rem = id - k * plane;
    j = rem / shape.nx;
    i = rem - j * shape.nx;
  };
  struct BoundaryState {
    bool boundary{false};
    bool external{false};
  };
  auto classify_boundary = [&](std::size_t i, std::size_t j,
                               std::size_t k) -> BoundaryState {
    BoundaryState state;
    if (i == 0 || j == 0 || k == 0 || i + 1 >= shape.nx || j + 1 >= shape.ny ||
        k + 1 >= shape.nz)
      return state;
    std::array<std::int32_t, 6> faces{};
    face_media6(i, j, k, faces);
    state.external = std::any_of(faces.begin(), faces.end(),
                                 [](std::int32_t v) { return v == 0; });
    state.boundary = faces[0] != faces[5];
    for (std::size_t n = 1; n < faces.size(); ++n)
      state.boundary = state.boundary || faces[n] != faces[n - 1];
    return state;
  };

  std::vector<std::uint32_t> boundary_rank(shape.n, 0U);
  std::vector<unsigned char> boundary_external(shape.n, 0U);
  std::vector<std::size_t> boundary_queue;
  boundary_queue.reserve(boundary_voxels.size() * 2);
  std::size_t discovered_boundary = 0;
  std::size_t external_boundary = 0;
  if (shape.nx >= 3 && shape.ny >= 3 && shape.nz >= 3) {
    for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
      for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
        for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
          const BoundaryState state = classify_boundary(i, j, k);
          if (!state.boundary)
            continue;
          const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
          boundary_rank[id] = static_cast<std::uint32_t>(++discovered_boundary);
          boundary_external[id] = state.external ? 1U : 0U;
          if (state.external) {
            ++external_boundary;
            boundary_queue.push_back(id);
          }
        }
      }
    }
  }
  out.audit.boundary_initial = discovered_boundary;

  std::size_t added = 0;
  std::size_t removed = 0;
  std::size_t exposed = 0;
  std::size_t contact = 0;
  auto update_boundary_status = [&](std::size_t id, bool enqueue_new) {
    std::size_t i = 0, j = 0, k = 0;
    decode_index(id, i, j, k);
    const BoundaryState state = classify_boundary(i, j, k);
    if (!state.boundary) {
      if (boundary_rank[id] != 0U) {
        if (boundary_external[id] != 0U && external_boundary > 0)
          --external_boundary;
        boundary_rank[id] = 0U;
        boundary_external[id] = 0U;
        ++removed;
      }
      return;
    }
    if (boundary_rank[id] == 0U) {
      boundary_rank[id] = static_cast<std::uint32_t>(++discovered_boundary);
      boundary_external[id] = state.external ? 1U : 0U;
      if (state.external)
        ++external_boundary;
      if (enqueue_new)
        boundary_queue.push_back(id);
      return;
    }
    if (boundary_external[id] != (state.external ? 1U : 0U)) {
      if (state.external)
        ++external_boundary;
      else if (external_boundary > 0)
        --external_boundary;
      boundary_external[id] = state.external ? 1U : 0U;
    }
  };
  auto update_adjacent_boundary_status = [&](std::size_t i, std::size_t j,
                                             std::size_t k, int neighbor) {
    const std::size_t center_id = idx3(i, j, k, shape.nx, shape.ny);
    update_boundary_status(center_id, false);
    std::size_t ni = i, nj = j, nk = k;
    if (neighbor == 1)
      ++ni;
    else if (neighbor == 2)
      ++nj;
    else if (neighbor == 3)
      ++nk;
    else if (neighbor == 4)
      --ni;
    else if (neighbor == 5)
      --nj;
    else if (neighbor == 6)
      --nk;
    if (ni > 0 && nj > 0 && nk > 0 && ni + 1 < shape.nx && nj + 1 < shape.ny &&
        nk + 1 < shape.nz)
      update_boundary_status(idx3(ni, nj, nk, shape.nx, shape.ny), true);
  };
  auto set_contact_solvent = [&](const DelphiFaceRef &ref, std::int32_t entity,
                                 std::uint16_t action, std::uint32_t iteration,
                                 const ClosestSurfaceAtom &nearest_atom,
                                 double nearest_exposed_dist) {
    (void)entity;
    const std::size_t fid = face_flat(ref);
    const std::int32_t original = out.face_encoded[fid];
    std::int32_t original_entity = decode_entity_from_value(original);
    const std::int32_t original_media = decode_media_from_value(original);
    if (original_entity >= 0) {
      if (original_entity == 0)
        original_entity = 1;
      set_face_encoded(fid,
                       encode_delphi_face(-original_entity, original_media));
    }
    if (fid < out.face_debug.size()) {
      DelphiFaceDebug &debug = out.face_debug[fid];
      debug.action = action;
      debug.iteration = iteration;
      debug.nearest_exposed_dist = nearest_exposed_dist;
      if (nearest_atom.index >= 0) {
        debug.nearest_atom = static_cast<std::int32_t>(nearest_atom.index + 1);
        debug.nearest_atom_surface_dist = nearest_atom.surface_dist;
      }
    }
  };
  auto set_cavity_media = [&](const DelphiFaceRef &ref, std::uint32_t iteration,
                              const ClosestSurfaceAtom &nearest_atom,
                              double nearest_exposed_dist) {
    const std::size_t fid = face_flat(ref);
    if (decode_media_from_value(out.face_encoded[fid]) == 0)
      ++added;
    const std::int32_t media =
        nearest_atom.index >= 0 ? std::int32_t{1} : std::int32_t{1};
    set_face_encoded(fid, encode_delphi_face(std::int32_t{1}, media));
    if (fid < out.face_debug.size()) {
      DelphiFaceDebug &debug = out.face_debug[fid];
      debug.action = kDelphiFaceDelphiCavityRemap;
      debug.iteration = iteration;
      debug.nearest_exposed_dist = nearest_exposed_dist;
      if (nearest_atom.index >= 0) {
        debug.nearest_atom = static_cast<std::int32_t>(nearest_atom.index + 1);
        debug.nearest_atom_surface_dist = nearest_atom.surface_dist;
      }
    }
  };

  const std::size_t max_boundary_iterations =
      std::max<std::size_t>(32, shape.nx + shape.ny + shape.nz);
  std::size_t batch_begin = 0;
  std::size_t iterations = 0;
  while (batch_begin < boundary_queue.size() &&
         iterations < max_boundary_iterations) {
    const std::size_t batch_end = boundary_queue.size();
    ++iterations;
    for (std::size_t q = batch_begin; q < batch_end; ++q) {
      const std::size_t center_id = boundary_queue[q];
      if (center_id >= shape.n || boundary_rank[center_id] == 0U)
        continue;
      std::size_t i = 0, j = 0, k = 0;
      decode_index(center_id, i, j, k);
      std::array<std::int32_t, 7> neighbor_entities{};
      std::array<std::int32_t, 7> neighbor_media{};
      bool rm_boundary_pt = false;
      for (int neighbor = 1; neighbor <= 6; ++neighbor) {
        const DelphiFaceRef ref = face_ref_for_neighbor(i, j, k, neighbor);
        if (!ref.valid)
          continue;
        const std::size_t fid = face_flat(ref);
        neighbor_entities[static_cast<std::size_t>(neighbor)] =
            decode_entity_from_value(out.face_encoded[fid]);
        neighbor_media[static_cast<std::size_t>(neighbor)] =
            decode_media_from_value(out.face_encoded[fid]);
        const std::int32_t entity =
            neighbor_entities[static_cast<std::size_t>(neighbor)];
        if (entity > 1 &&
            entity <= static_cast<std::int32_t>(surface_atoms.size() + 1))
          rm_boundary_pt = true;
      }
      for (int neighbor = 1; neighbor <= 6; ++neighbor) {
        const DelphiFaceRef ref = face_ref_for_neighbor(i, j, k, neighbor);
        if (!ref.valid)
          continue;
        const std::int32_t entity =
            neighbor_entities[static_cast<std::size_t>(neighbor)];
        const std::int32_t media =
            neighbor_media[static_cast<std::size_t>(neighbor)];
        const bool should_process =
            entity == 0 ||
            (rm_boundary_pt &&
             entity > static_cast<std::int32_t>(surface_atoms.size() + 1)) ||
            (media == 0 && entity > 0);
        if (!should_process)
          continue;
        const auto midpoint = face_midpoint(i, j, k, neighbor);
        const double exposed_dist =
            nearest_exposed_distance(midpoint[0], midpoint[1], midpoint[2]);
        if (exposed_dist < args_.probe_radius) {
          set_contact_solvent(ref, 1, kDelphiFaceExposedSolvent,
                              static_cast<std::uint32_t>(iterations),
                              ClosestSurfaceAtom{}, exposed_dist);
          ++exposed;
          continue;
        }
        const ClosestSurfaceAtom contact_atom =
            closest_surface_atom(midpoint[0], midpoint[1], midpoint[2]);
        if (contact_atom.index >= 0 &&
            !probe_surface_occluded(
                midpoint[0], midpoint[1], midpoint[2],
                static_cast<std::size_t>(contact_atom.index))) {
          set_contact_solvent(
              ref, static_cast<std::int32_t>(contact_atom.index + 2),
              kDelphiFaceContactSolvent, static_cast<std::uint32_t>(iterations),
              contact_atom, exposed_dist);
          ++contact;
        } else {
          set_cavity_media(ref, static_cast<std::uint32_t>(iterations),
                           contact_atom, exposed_dist);
          update_adjacent_boundary_status(i, j, k, neighbor);
        }
      }
      update_boundary_status(center_id, false);
    }
    batch_begin = batch_end;
  }

  for (std::size_t fid = 0; fid < out.face_encoded.size(); ++fid) {
    const std::int32_t encoded = out.face_encoded[fid];
    if (encoded < 0) {
      const std::int32_t signed_entity = encoded % delphi_epsilon_dim;
      std::int32_t final_value = -encoded;
      if (signed_entity == -1)
        final_value -= 1;
      set_face_encoded(fid, final_value);
      if (fid < out.face_debug.size() &&
          out.face_debug[fid].action == kDelphiFaceUnchanged)
        out.face_debug[fid].action = kDelphiFaceSignPreserve;
    } else {
      sync_face_from_encoded(fid);
    }
  }

  out.audit.added = added;
  out.audit.removed = removed;
  out.audit.exposed_points = exposed_bins.points.empty()
                                 ? exposed + contact
                                 : exposed_bins.points.size();
  std::size_t final_boundary = 0;
  for (std::uint32_t rank : boundary_rank) {
    if (rank != 0U)
      ++final_boundary;
  }
  out.audit.boundary_final = final_boundary;
  const double face_count =
      static_cast<double>(std::max<std::size_t>(1, shape.n * 3));
  out.audit.midpoint_rms =
      (args_.ext_diel - args_.int_diel) *
      std::sqrt(static_cast<double>(added + removed) / face_count);
  out.iscm_surface_valid.assign(shape.n, 0U);
  out.iscm_surface_positions.assign(shape.n, {0.0, 0.0, 0.0});
  for (std::size_t id = 0; id < shape.n; ++id) {
    if (boundary_rank[id] == 0U)
      continue;
    std::size_t i = 0, j = 0, k = 0;
    decode_index(id, i, j, k);
    const std::array<double, 3> grid_point{
        shape.ox + static_cast<double>(i) * shape.h,
        shape.oy + static_cast<double>(j) * shape.h,
        shape.oz + static_cast<double>(k) * shape.h};
    std::array<double, 3> surface_point = grid_point;
    bool scaled = false;
    if (id < boundary_external.size() && boundary_external[id] == 0U &&
        surface_atoms.size() >= 2) {
      std::size_t closest1 = std::numeric_limits<std::size_t>::max();
      std::size_t closest2 = std::numeric_limits<std::size_t>::max();
      double min1 = std::numeric_limits<double>::max();
      double min2 = std::numeric_limits<double>::max();
      for (std::size_t atom_idx = 0; atom_idx < surface_atoms.size();
           ++atom_idx) {
        const Atom &atom = surface_atoms[atom_idx];
        const double dx = grid_point[0] - atom.x;
        const double dy = grid_point[1] - atom.y;
        const double dz = grid_point[2] - atom.z;
        const double distance_sq =
            dx * dx + dy * dy + dz * dz - atom.radius * atom.radius;
        if (distance_sq < min1) {
          closest2 = closest1;
          min2 = min1;
          closest1 = atom_idx;
          min1 = distance_sq;
        } else if (distance_sq <= min2) {
          closest2 = atom_idx;
          min2 = distance_sq;
        }
      }
      if (closest1 != std::numeric_limits<std::size_t>::max() &&
          closest2 != std::numeric_limits<std::size_t>::max() &&
          closest1 != closest2) {
        const Atom &atom1 = surface_atoms[closest1];
        const Atom &atom2 = surface_atoms[closest2];
        const double dx12 = atom2.x - atom1.x;
        const double dy12 = atom2.y - atom1.y;
        const double dz12 = atom2.z - atom1.z;
        const double dist12_sq = dx12 * dx12 + dy12 * dy12 + dz12 * dz12;
        if (dist12_sq > 1.0e-30) {
          const double factor = 0.5 * (min2 - min1) / dist12_sq;
          surface_point = {grid_point[0] + factor * dx12,
                           grid_point[1] + factor * dy12,
                           grid_point[2] + factor * dz12};
          scaled = true;
        }
      }
    }
    if (scaled) {
      out.iscm_surface_valid[id] = 1U;
      out.iscm_surface_positions[id] = surface_point;
      continue;
    }
    const ClosestSurfaceAtom nearest_atom =
        closest_surface_atom(grid_point[0], grid_point[1], grid_point[2]);
    if (nearest_atom.index >= 0 &&
        static_cast<std::size_t>(nearest_atom.index) < surface_atoms.size()) {
      const Atom &atom =
          surface_atoms[static_cast<std::size_t>(nearest_atom.index)];
      const double dx = grid_point[0] - atom.x;
      const double dy = grid_point[1] - atom.y;
      const double dz = grid_point[2] - atom.z;
      const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
      bool solvent_exposed = true;
      if (args_.probe_radius > 0.0) {
        solvent_exposed = !probe_surface_occluded(
            grid_point[0], grid_point[1], grid_point[2],
            static_cast<std::size_t>(nearest_atom.index));
      }
      if (distance > 1.0e-30 && solvent_exposed) {
        surface_point = {atom.x + dx * (atom.radius / distance),
                         atom.y + dy * (atom.radius / distance),
                         atom.z + dz * (atom.radius / distance)};
        scaled = true;
      }
    }
    if (!scaled && args_.probe_radius > 0.0 && !exposed_bins.empty) {
      std::array<double, 3> exposed_point{0.0, 0.0, 0.0};
      const double exposed_distance = nearest_exposed_point(
          grid_point[0], grid_point[1], grid_point[2], exposed_point);
      if (std::isfinite(exposed_distance) && exposed_distance > 1.0e-30) {
        const double factor = args_.probe_radius / exposed_distance;
        surface_point = {
            exposed_point[0] + (grid_point[0] - exposed_point[0]) * factor,
            exposed_point[1] + (grid_point[1] - exposed_point[1]) * factor,
            exposed_point[2] + (grid_point[2] - exposed_point[2]) * factor};
      }
    }
    out.iscm_surface_valid[id] = 1U;
    out.iscm_surface_positions[id] = surface_point;
  }
  (void)positive_face;
  return out;
}

void AnalyzerImpl::apply_delphi_vdw_media_map(
    const GridShape &shape, const DelphiVdwMediaMap &media_map, double eps_in,
    double eps_out) {
  if (media_map.face_media.size() != shape.n * 3 ||
      media_map.face_encoded.size() != shape.n * 3)
    throw std::runtime_error(
        "PBE: internal DelPhi VDW media map has wrong size.");
  for (std::size_t id = 0; id < shape.n; ++id) {
    grid_.eps_x[id] = media_map.face_media[id * 3] != 0 ? eps_in : eps_out;
    grid_.eps_y[id] = media_map.face_media[id * 3 + 1] != 0 ? eps_in : eps_out;
    grid_.eps_z[id] = media_map.face_media[id * 3 + 2] != 0 ? eps_in : eps_out;
  }
  maybe_write_midpoint_face_diff(shape, media_map, eps_in, eps_out);
  last_vdwms_audit_ = media_map.audit;
  last_iscm_surface_valid_ = media_map.iscm_surface_valid;
  last_iscm_surface_positions_ = media_map.iscm_surface_positions;
}

void AnalyzerImpl::maybe_write_midpoint_face_diff(
    const GridShape &shape, const DelphiVdwMediaMap &media_map, double eps_in,
    double eps_out) const {
  if (args_.test_midpoint_diff == "off")
    return;
  if (args_.test_pqr.empty())
    throw std::runtime_error(
        "PBE: -pbe-test-midpoint-diff requires -pbe-test-pqr.");
  if (!oracle_replay_applies(shape))
    return;
  if (media_map.face_media.size() != shape.n * 3)
    throw std::runtime_error(
        "PBE: internal DelPhi VDW media map has wrong size for midpoint diff.");
  if (std::abs(eps_in - args_.int_diel) > 1.0e-10 ||
      std::abs(eps_out - args_.ext_diel) > 1.0e-10) {
    return;
  }

  std::ifstream in("pbe_oracle_midpoint_epsilon.csv");
  if (!in.is_open()) {
    throw std::runtime_error(
        "PBE: -pbe-test-midpoint-diff requires "
        "pbe_oracle_midpoint_epsilon.csv in the run directory.");
  }
  std::string line;
  if (!std::getline(in, line))
    throw std::runtime_error("PBE: empty pbe_oracle_midpoint_epsilon.csv.");
  const auto header = csv_header_map(split_csv_simple(line));
  const auto fixture_it = header.find("fixture");
  auto fixture_matches = [&](const std::vector<std::string> &fields) {
    if (fixture_it == header.end() || fixture_it->second >= fields.size())
      return true;
    const std::string fixture = fields[fixture_it->second];
    if (fixture.empty() || current_source_label_.empty())
      return true;
    if (fixture == current_source_label_)
      return true;
    if (current_source_label_.rfind(fixture + "__", 0) == 0 ||
        fixture.rfind(current_source_label_ + "__", 0) == 0)
      return true;
    // TITAN trajectory labels are often input filenames, while pyDelPhi oracle
    // rows use fixture names. The oracle file is copied per run, so keep the
    // rows comparable instead of dropping the entire file on label mismatch.
    return true;
  };

  auto face_index = [](std::size_t id, int axis) {
    return id * 3 + static_cast<std::size_t>(axis);
  };
  auto axis_name = [](int axis) -> const char * {
    if (axis == 0)
      return "x";
    if (axis == 1)
      return "y";
    return "z";
  };

  if (args_.test_midpoint_diff == "full") {
    std::ofstream state_out;
    if (!native_midpoint_state_header_written_) {
      state_out.open("pbe_native_midpoint_state.csv",
                     std::ios::out | std::ios::trunc);
      if (!state_out.is_open())
        throw std::runtime_error(
            "PBE: failed to create pbe_native_midpoint_state.csv.");
      state_out << "fixture,i,j,k,raw_idx_x,raw_idx_y,raw_idx_z,idx_x,idx_y,"
                   "idx_z,raw_media_x,raw_media_y,raw_media_z,media_x,"
                   "media_y,media_z,eps_x,eps_y,eps_z\n";
      native_midpoint_state_header_written_ = true;
    } else {
      state_out.open("pbe_native_midpoint_state.csv",
                     std::ios::out | std::ios::app);
      if (!state_out.is_open())
        throw std::runtime_error(
            "PBE: failed to append pbe_native_midpoint_state.csv.");
    }
    state_out << std::setprecision(17);
    const std::string fixture =
        current_source_label_.empty() ? "unknown" : current_source_label_;
    for (std::size_t i = 0; i < shape.nx; ++i) {
      for (std::size_t j = 0; j < shape.ny; ++j) {
        for (std::size_t k = 0; k < shape.nz; ++k) {
          const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
          std::int32_t raw_idx[3]{0, 0, 0};
          std::int32_t final_idx[3]{0, 0, 0};
          std::int32_t raw_media[3]{0, 0, 0};
          std::int32_t final_media[3]{0, 0, 0};
          double eps[3]{eps_out, eps_out, eps_out};
          for (int axis = 0; axis < 3; ++axis) {
            const std::size_t fid = face_index(id, axis);
            if (fid < media_map.face_debug.size()) {
              raw_idx[axis] = media_map.face_debug[fid].raw_encoded;
              raw_media[axis] = media_map.face_debug[fid].raw_media;
            }
            if (fid < media_map.face_encoded.size())
              final_idx[axis] = media_map.face_encoded[fid];
            if (fid < media_map.face_media.size())
              final_media[axis] = media_map.face_media[fid];
            eps[axis] = final_media[axis] != 0 ? eps_in : eps_out;
          }
          state_out << csv_escape(fixture) << ',' << i << ',' << j << ',' << k
                    << ',' << raw_idx[0] << ',' << raw_idx[1] << ','
                    << raw_idx[2] << ',' << final_idx[0] << ',' << final_idx[1]
                    << ',' << final_idx[2] << ',' << raw_media[0] << ','
                    << raw_media[1] << ',' << raw_media[2] << ','
                    << final_media[0] << ',' << final_media[1] << ','
                    << final_media[2] << ',' << eps[0] << ',' << eps[1] << ','
                    << eps[2] << '\n';
        }
      }
    }

    std::map<std::tuple<std::uint16_t, std::int32_t, std::int32_t>, std::size_t>
        histogram;
    for (std::size_t fid = 0; fid < media_map.face_debug.size(); ++fid) {
      const auto &debug = media_map.face_debug[fid];
      const std::int32_t final_media =
          fid < media_map.face_media.size() ? media_map.face_media[fid] : 0;
      ++histogram[std::make_tuple(debug.action, debug.raw_media, final_media)];
    }
    std::ofstream hist_out;
    if (!vdwms_action_histogram_header_written_) {
      hist_out.open("pbe_vdwms_action_histogram.csv",
                    std::ios::out | std::ios::trunc);
      if (!hist_out.is_open())
        throw std::runtime_error(
            "PBE: failed to create pbe_vdwms_action_histogram.csv.");
      hist_out << "fixture,action,raw_media,final_media,count\n";
      vdwms_action_histogram_header_written_ = true;
    } else {
      hist_out.open("pbe_vdwms_action_histogram.csv",
                    std::ios::out | std::ios::app);
      if (!hist_out.is_open())
        throw std::runtime_error(
            "PBE: failed to append pbe_vdwms_action_histogram.csv.");
    }
    for (const auto &[key, count] : histogram) {
      const auto [action, raw_media, final_media] = key;
      hist_out << csv_escape(fixture) << ',' << delphi_face_action_name(action)
               << ',' << raw_media << ',' << final_media << ',' << count
               << '\n';
    }
  }

  std::ofstream out;
  if (!midpoint_face_diff_header_written_) {
    out.open("pbe_midpoint_face_diff.csv", std::ios::out | std::ios::trunc);
    if (!out.is_open())
      throw std::runtime_error(
          "PBE: failed to create pbe_midpoint_face_diff.csv.");
    out << "i,j,k,axis,native_eps,oracle_eps,native_media,oracle_media,"
           "native_owner,native_encoded,oracle_encoded,face_x_A,face_y_A,"
           "face_z_A,queue_iteration,"
           "remap_action,nearest_atom,nearest_atom_surface_dist,"
           "nearest_exposed_dist,raw_media,raw_owner,raw_encoded,final_"
           "encoded\n";
    midpoint_face_diff_header_written_ = true;
  } else {
    out.open("pbe_midpoint_face_diff.csv", std::ios::out | std::ios::app);
    if (!out.is_open())
      throw std::runtime_error(
          "PBE: failed to append pbe_midpoint_face_diff.csv.");
  }
  out << std::setprecision(17);

  auto face_coord = [&](std::size_t i, std::size_t j, std::size_t k, int axis) {
    std::array<double, 3> point{shape.ox + static_cast<double>(i) * shape.h,
                                shape.oy + static_cast<double>(j) * shape.h,
                                shape.oz + static_cast<double>(k) * shape.h};
    if (axis == 0)
      point[0] += 0.5 * shape.h;
    else if (axis == 1)
      point[1] += 0.5 * shape.h;
    else
      point[2] += 0.5 * shape.h;
    return point;
  };
  auto oracle_media = [&](double eps_value) {
    if (std::abs(eps_value - eps_in) <= 1.0e-9)
      return 1;
    if (std::abs(eps_value - eps_out) <= 1.0e-9)
      return 0;
    return -1;
  };
  auto write_optional_double = [&](double value) {
    if (std::isfinite(value) && std::abs(value) < 1.0e299)
      out << value;
    else
      out << "NA";
  };

  const bool full = args_.test_midpoint_diff == "full";
  const std::size_t sample_limit =
      full ? std::numeric_limits<std::size_t>::max() : std::size_t{50};
  std::size_t compared_rows = 0;
  std::size_t written_rows = 0;
  std::size_t skipped_fixture_rows = 0;
  while (std::getline(in, line)) {
    if (trim_copy(line).empty())
      continue;
    const auto fields = split_csv_simple(line);
    if (!fixture_matches(fields)) {
      ++skipped_fixture_rows;
      continue;
    }
    std::size_t i = 0, j = 0, k = 0;
    double oracle_eps[3]{0.0, 0.0, 0.0};
    double oracle_encoded[3]{std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN()};
    if (!csv_index(fields, header, "i", i) ||
        !csv_index(fields, header, "j", j) ||
        !csv_index(fields, header, "k", k)) {
      throw std::runtime_error(
          "PBE: pbe_oracle_midpoint_epsilon.csv row is missing i,j,k.");
    }
    if (!csv_double(fields, header, {"eps_x", "epsilon_x", "ex"},
                    oracle_eps[0]) ||
        !csv_double(fields, header, {"eps_y", "epsilon_y", "ey"},
                    oracle_eps[1]) ||
        !csv_double(fields, header, {"eps_z", "epsilon_z", "ez"},
                    oracle_eps[2])) {
      throw std::runtime_error("PBE: pbe_oracle_midpoint_epsilon.csv row is "
                               "missing eps_x/eps_y/eps_z.");
    }
    if (i >= shape.nx || j >= shape.ny || k >= shape.nz)
      throw std::runtime_error(
          "PBE: pbe_oracle_midpoint_epsilon.csv index is outside the grid.");
    (void)csv_double(fields, header, {"idx_x", "index_x", "encoded_x"},
                     oracle_encoded[0]);
    (void)csv_double(fields, header, {"idx_y", "index_y", "encoded_y"},
                     oracle_encoded[1]);
    (void)csv_double(fields, header, {"idx_z", "index_z", "encoded_z"},
                     oracle_encoded[2]);
    const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
    ++compared_rows;
    for (int axis = 0; axis < 3; ++axis) {
      const std::size_t fid = face_index(id, axis);
      const int native_media = media_map.face_media[fid] != 0 ? 1 : 0;
      const double native_eps = native_media != 0 ? eps_in : eps_out;
      if (std::abs(native_eps - oracle_eps[axis]) <= 1.0e-12)
        continue;
      if (written_rows >= sample_limit)
        continue;
      const auto point = face_coord(i, j, k, axis);
      const DelphiFaceDebug debug = fid < media_map.face_debug.size()
                                        ? media_map.face_debug[fid]
                                        : DelphiFaceDebug{};
      out << i << ',' << j << ',' << k << ',' << axis_name(axis) << ','
          << native_eps << ',' << oracle_eps[axis] << ',' << native_media << ','
          << oracle_media(oracle_eps[axis]) << ',' << media_map.face_owner[fid]
          << ','
          << (fid < media_map.face_encoded.size() ? media_map.face_encoded[fid]
                                                  : std::int32_t{0})
          << ',';
      write_optional_double(oracle_encoded[axis]);
      out << ',' << point[0] << ',' << point[1] << ',' << point[2] << ','
          << debug.iteration << ',' << delphi_face_action_name(debug.action)
          << ',' << debug.nearest_atom << ',';
      write_optional_double(debug.nearest_atom_surface_dist);
      out << ',';
      write_optional_double(debug.nearest_exposed_dist);
      out << ',' << debug.raw_media << ',' << debug.raw_owner << ','
          << debug.raw_encoded << ',' << debug.final_encoded << '\n';
      ++written_rows;
    }
  }
  if (compared_rows == 0) {
    const std::string skipped =
        skipped_fixture_rows > 0 ? " fixture filter removed all rows." : "";
    throw std::runtime_error(
        "PBE: pbe_oracle_midpoint_epsilon.csv has no comparable rows for "
        "midpoint diff." +
        skipped);
  }
}


} // namespace TitanPBE
