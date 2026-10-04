#include "BoxMatrix.h"
#include "functions.h"
#include "pbctopo_hull.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <thread>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

int init_rejected_images(vector<size_t> &rejected_images, arguments &Arguments);
bool check_in_rejected_images(vector<size_t> &rejected_images, size_t im);

int gen_chain_list_cell(size_t chain, vector<vector<size_t>> &data);

namespace {

constexpr int kEnvelopeVoxelBias = 1 << 20;
constexpr double kEnvelopeGrid = 2.5;
constexpr double kClashGrid = 2.5;
constexpr double kHardClashCutoff = 1.10;
constexpr double kSoftClashCutoff = 1.35;
constexpr double kContactCutoff = 4.50;
constexpr double kContactFloor = 1.50;
constexpr double kHardClashCutoff2 = kHardClashCutoff * kHardClashCutoff;
constexpr double kSoftClashCutoff2 = kSoftClashCutoff * kSoftClashCutoff;
constexpr double kContactCutoff2 = kContactCutoff * kContactCutoff;
constexpr double kContactFloor2 = kContactFloor * kContactFloor;
constexpr int kFaceOffsets[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                    {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
constexpr int kLegacyImageCoeff[27][3] = {
    {1, 0, 0},   {-1, 0, 0}, {0, 1, 0},   {0, -1, 0}, {0, 0, 1},
    {0, 0, -1},  {1, 1, 0},  {1, -1, 0},  {-1, 1, 0}, {-1, -1, 0},
    {1, 0, 1},   {1, 0, -1}, {-1, 0, 1},  {-1, 0, -1}, {0, 1, 1},
    {0, -1, 1},  {0, 1, -1}, {0, -1, -1}, {1, 1, 1},  {-1, 1, 1},
    {1, -1, 1},  {1, 1, -1}, {-1, -1, 1}, {-1, 1, -1}, {1, -1, -1},
    {-1, -1, -1}, {0, 0, 0}};

struct EnvelopeChainGeom {
  size_t atom_count = 0;
  double mean_x = 0.0;
  double mean_y = 0.0;
  double mean_z = 0.0;
  double env_center_x = 0.0;
  double env_center_y = 0.0;
  double env_center_z = 0.0;
  vector<array<double, 3>> atom_cart;
  vector<array<double, 3>> core_centers;
  vector<array<double, 3>> surface_centers;
};

struct EnvelopePlacementScore {
  size_t moving = static_cast<size_t>(-1);
  size_t anchor = static_cast<size_t>(-1);
  size_t image_index = 26;
  long sx = 0;
  long sy = 0;
  long sz = 0;
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;
  size_t hard_clash = std::numeric_limits<size_t>::max();
  size_t soft_clash = std::numeric_limits<size_t>::max();
  size_t overlap_cells = std::numeric_limits<size_t>::max();
  size_t contact_atoms = 0;
  size_t contact_cells = 0;
  double center_d2 = std::numeric_limits<double>::infinity();
  bool valid = false;
};

struct EnvelopeScene {
  unordered_set<uint64_t> core_cells;
  unordered_map<uint64_t, vector<array<double, 3>>> clash_atoms;
  unordered_map<uint64_t, vector<array<double, 3>>> surface_points;
  double center_sum_x = 0.0;
  double center_sum_y = 0.0;
  double center_sum_z = 0.0;
  size_t atom_total = 0;
};

using pbctopo_hull::PbctopoHullScratch;
using pbctopo_hull::Vec3;

struct HullChainGeom {
  size_t atom_count = 0;
  vector<Vec3> atom_points;
};

struct HullAssemblyScene {
  unordered_map<uint64_t, vector<Vec3>> clash_atoms;
  vector<Vec3> points;
};

struct HullPlacementScore {
  size_t moving = static_cast<size_t>(-1);
  size_t image_index = 26;
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;
  double hull_volume = std::numeric_limits<double>::infinity();
  double min_d2 = std::numeric_limits<double>::infinity();
  bool min_d2_ready = false;
  bool valid = false;
};

struct LegacyImageShift {
  size_t image_index = 26;
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;
};

uint64_t pack_envelope_cell(int ix, int iy, int iz) {
  const uint64_t ux = static_cast<uint64_t>(ix + kEnvelopeVoxelBias);
  const uint64_t uy = static_cast<uint64_t>(iy + kEnvelopeVoxelBias);
  const uint64_t uz = static_cast<uint64_t>(iz + kEnvelopeVoxelBias);
  return (ux << 42) | (uy << 21) | uz;
}

void unpack_envelope_cell(uint64_t key, int &ix, int &iy, int &iz) {
  constexpr uint64_t mask = (1ULL << 21) - 1ULL;
  ix = static_cast<int>((key >> 42) & mask) - kEnvelopeVoxelBias;
  iy = static_cast<int>((key >> 21) & mask) - kEnvelopeVoxelBias;
  iz = static_cast<int>(key & mask) - kEnvelopeVoxelBias;
}

int envelope_grid_index(double value, double spacing) {
  return static_cast<int>(std::floor(value / spacing));
}

bool better_envelope_placement(const EnvelopePlacementScore &lhs,
                               const EnvelopePlacementScore &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  if (lhs.hard_clash != rhs.hard_clash)
    return lhs.hard_clash < rhs.hard_clash;
  if (lhs.soft_clash != rhs.soft_clash)
    return lhs.soft_clash < rhs.soft_clash;
  const bool lhs_has_contact = lhs.contact_atoms > 0 || lhs.contact_cells > 0;
  const bool rhs_has_contact = rhs.contact_atoms > 0 || rhs.contact_cells > 0;
  if (lhs_has_contact != rhs_has_contact)
    return lhs_has_contact;
  if (lhs.overlap_cells != rhs.overlap_cells)
    return lhs.overlap_cells < rhs.overlap_cells;
  if (lhs.contact_atoms != rhs.contact_atoms)
    return lhs.contact_atoms > rhs.contact_atoms;
  if (lhs.contact_cells != rhs.contact_cells)
    return lhs.contact_cells > rhs.contact_cells;
  if (lhs.center_d2 != rhs.center_d2)
    return lhs.center_d2 < rhs.center_d2;
  if (lhs.image_index != rhs.image_index)
    return lhs.image_index < rhs.image_index;
  return lhs.anchor < rhs.anchor;
}

bool better_coarse_envelope_placement(const EnvelopePlacementScore &lhs,
                                      const EnvelopePlacementScore &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  const bool lhs_has_contact = lhs.contact_cells > 0;
  const bool rhs_has_contact = rhs.contact_cells > 0;
  if (lhs_has_contact != rhs_has_contact)
    return lhs_has_contact;
  if (lhs.overlap_cells != rhs.overlap_cells)
    return lhs.overlap_cells < rhs.overlap_cells;
  if (lhs.contact_cells != rhs.contact_cells)
    return lhs.contact_cells > rhs.contact_cells;
  if (lhs.center_d2 != rhs.center_d2)
    return lhs.center_d2 < rhs.center_d2;
  if (lhs.image_index != rhs.image_index)
    return lhs.image_index < rhs.image_index;
  return lhs.anchor < rhs.anchor;
}

bool is_clean_envelope_placement(const EnvelopePlacementScore &score) {
  return score.valid && score.hard_clash == 0 && score.soft_clash == 0 &&
         score.overlap_cells == 0 &&
         (score.contact_atoms > 0 || score.contact_cells > 0);
}

EnvelopeChainGeom build_envelope_chain_geom(
    vector<array<double, 3>> atom_cart) {
  EnvelopeChainGeom geom;
  geom.atom_count = atom_cart.size();
  geom.atom_cart = std::move(atom_cart);
  if (geom.atom_cart.empty())
    return geom;

  double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  unordered_set<uint64_t> local_core;
  local_core.reserve(geom.atom_cart.size() * 2);

  for (const auto &atom : geom.atom_cart) {
    sum_x += atom[0];
    sum_y += atom[1];
    sum_z += atom[2];
    min_x = std::min(min_x, atom[0]);
    min_y = std::min(min_y, atom[1]);
    min_z = std::min(min_z, atom[2]);
    max_x = std::max(max_x, atom[0]);
    max_y = std::max(max_y, atom[1]);
    max_z = std::max(max_z, atom[2]);
    local_core.insert(pack_envelope_cell(
        envelope_grid_index(atom[0], kEnvelopeGrid),
        envelope_grid_index(atom[1], kEnvelopeGrid),
        envelope_grid_index(atom[2], kEnvelopeGrid)));
  }

  const double inv_n = 1.0 / static_cast<double>(geom.atom_cart.size());
  geom.mean_x = sum_x * inv_n;
  geom.mean_y = sum_y * inv_n;
  geom.mean_z = sum_z * inv_n;
  geom.env_center_x = 0.5 * (min_x + max_x);
  geom.env_center_y = 0.5 * (min_y + max_y);
  geom.env_center_z = 0.5 * (min_z + max_z);

  geom.core_centers.reserve(local_core.size());
  geom.surface_centers.reserve(local_core.size());
  for (uint64_t key : local_core) {
    int ix = 0, iy = 0, iz = 0;
    unpack_envelope_cell(key, ix, iy, iz);
    const array<double, 3> center = {
        (static_cast<double>(ix) + 0.5) * kEnvelopeGrid,
        (static_cast<double>(iy) + 0.5) * kEnvelopeGrid,
        (static_cast<double>(iz) + 0.5) * kEnvelopeGrid};
    geom.core_centers.push_back(center);

    bool surface = false;
    for (const auto &off : kFaceOffsets) {
      if (local_core.find(
              pack_envelope_cell(ix + off[0], iy + off[1], iz + off[2])) ==
          local_core.end()) {
        surface = true;
        break;
      }
    }
    if (surface)
      geom.surface_centers.push_back(center);
  }
  return geom;
}

HullChainGeom build_hull_chain_geom(vector<Vec3> atom_points) {
  HullChainGeom geom;
  geom.atom_count = atom_points.size();
  geom.atom_points = std::move(atom_points);
  return geom;
}

void reserve_envelope_scene(EnvelopeScene &scene, size_t atom_hint,
                            size_t surface_hint) {
  scene.core_cells.reserve(std::max<size_t>(1, atom_hint * 2));
  scene.clash_atoms.reserve(std::max<size_t>(1, atom_hint * 2));
  scene.surface_points.reserve(std::max<size_t>(1, surface_hint * 2));
}

void reserve_hull_scene(HullAssemblyScene &scene, size_t atom_hint) {
  scene.clash_atoms.reserve(std::max<size_t>(1, atom_hint * 2));
  scene.points.reserve(atom_hint);
}

void add_chain_to_envelope_scene(EnvelopeScene &scene,
                                 const EnvelopeChainGeom &geom, double tx,
                                 double ty, double tz) {
  if (geom.atom_count == 0)
    return;

  scene.atom_total += geom.atom_count;
  scene.center_sum_x +=
      static_cast<double>(geom.atom_count) * (geom.env_center_x + tx);
  scene.center_sum_y +=
      static_cast<double>(geom.atom_count) * (geom.env_center_y + ty);
  scene.center_sum_z +=
      static_cast<double>(geom.atom_count) * (geom.env_center_z + tz);

  for (const auto &atom : geom.atom_cart) {
    const double rx = atom[0] + tx;
    const double ry = atom[1] + ty;
    const double rz = atom[2] + tz;
    scene.clash_atoms[pack_envelope_cell(
        envelope_grid_index(rx, kClashGrid), envelope_grid_index(ry, kClashGrid),
        envelope_grid_index(rz, kClashGrid))]
        .push_back({rx, ry, rz});
  }

  for (const auto &center : geom.core_centers) {
    scene.core_cells.insert(pack_envelope_cell(
        envelope_grid_index(center[0] + tx, kEnvelopeGrid),
        envelope_grid_index(center[1] + ty, kEnvelopeGrid),
        envelope_grid_index(center[2] + tz, kEnvelopeGrid)));
  }

  for (const auto &center : geom.surface_centers) {
    const double px = center[0] + tx;
    const double py = center[1] + ty;
    const double pz = center[2] + tz;
    scene.surface_points[pack_envelope_cell(
        envelope_grid_index(px, kContactCutoff),
        envelope_grid_index(py, kContactCutoff),
        envelope_grid_index(pz, kContactCutoff))]
        .push_back({px, py, pz});
  }
}

void add_chain_to_hull_scene(HullAssemblyScene &scene, const HullChainGeom &geom,
                             double tx, double ty, double tz) {
  if (geom.atom_count == 0)
    return;
  for (const Vec3 &atom : geom.atom_points) {
    const Vec3 shifted{atom.x + tx, atom.y + ty, atom.z + tz};
    scene.points.push_back(shifted);
    scene.clash_atoms[pack_envelope_cell(
        envelope_grid_index(shifted.x, kClashGrid),
        envelope_grid_index(shifted.y, kClashGrid),
        envelope_grid_index(shifted.z, kClashGrid))]
        .push_back(shifted);
  }
}

bool shifted_chain_has_exact_hard_clash(const HullChainGeom &geom,
                                        const HullAssemblyScene &scene,
                                        double tx, double ty, double tz) {
  if (geom.atom_points.empty() || scene.points.empty())
    return false;

  for (const Vec3 &atom : geom.atom_points) {
    const double rx = atom.x + tx;
    const double ry = atom.y + ty;
    const double rz = atom.z + tz;
    const int hx = envelope_grid_index(rx, kClashGrid);
    const int hy = envelope_grid_index(ry, kClashGrid);
    const int hz = envelope_grid_index(rz, kClashGrid);
    for (int ox = -1; ox <= 1; ++ox) {
      for (int oy = -1; oy <= 1; ++oy) {
        for (int oz = -1; oz <= 1; ++oz) {
          const auto it = scene.clash_atoms.find(
              pack_envelope_cell(hx + ox, hy + oy, hz + oz));
          if (it == scene.clash_atoms.end())
            continue;
          for (const Vec3 &other : it->second) {
            const double dx = rx - other.x;
            const double dy = ry - other.y;
            const double dz = rz - other.z;
            const double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < kHardClashCutoff2)
              return true;
          }
        }
      }
    }
  }

  return false;
}

double compute_shifted_chain_min_d2(const HullChainGeom &geom,
                                    const HullAssemblyScene &scene, double tx,
                                    double ty, double tz) {
  double min_d2 = std::numeric_limits<double>::infinity();
  if (geom.atom_points.empty() || scene.points.empty())
    return min_d2;

  for (const Vec3 &atom : geom.atom_points) {
    const double rx = atom.x + tx;
    const double ry = atom.y + ty;
    const double rz = atom.z + tz;
    for (const Vec3 &other : scene.points) {
      const double dx = rx - other.x;
      const double dy = ry - other.y;
      const double dz = rz - other.z;
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < min_d2)
        min_d2 = d2;
    }
  }

  return min_d2;
}

void build_shifted_hull_candidate_points(const HullAssemblyScene &scene,
                                         const HullChainGeom &geom, double tx,
                                         double ty, double tz,
                                         vector<Vec3> &candidate_points) {
  candidate_points.clear();
  candidate_points.reserve(scene.points.size() + geom.atom_points.size());
  candidate_points.insert(candidate_points.end(), scene.points.begin(),
                          scene.points.end());
  for (const Vec3 &atom : geom.atom_points) {
    candidate_points.push_back({atom.x + tx, atom.y + ty, atom.z + tz});
  }
}

double compute_convex_hull_volume_or_zero(const vector<Vec3> &points,
                                          PbctopoHullScratch &scratch) {
  if (points.size() < 4)
    return 0.0;

  double hull_volume = 0.0;
  if (pbctopo_hull::quickhull_volume(points, hull_volume, &scratch))
    return hull_volume;
  return 0.0;
}

void ensure_hull_candidate_min_d2(HullPlacementScore &candidate,
                                  const HullChainGeom &geom,
                                  const HullAssemblyScene &scene) {
  if (candidate.min_d2_ready)
    return;
  candidate.min_d2 =
      compute_shifted_chain_min_d2(geom, scene, candidate.tx, candidate.ty,
                                   candidate.tz);
  candidate.min_d2_ready = true;
}

bool better_hull_placement(HullPlacementScore &lhs, HullPlacementScore &rhs,
                           const vector<HullChainGeom> &geom,
                           const HullAssemblyScene &scene) {
  constexpr double kHullVolumeTol = 1.0e-9;
  constexpr double kMinD2Tol = 1.0e-12;
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  if (lhs.hull_volume + kHullVolumeTol < rhs.hull_volume)
    return true;
  if (rhs.hull_volume + kHullVolumeTol < lhs.hull_volume)
    return false;

  if (lhs.moving < geom.size())
    ensure_hull_candidate_min_d2(lhs, geom[lhs.moving], scene);
  if (rhs.moving < geom.size())
    ensure_hull_candidate_min_d2(rhs, geom[rhs.moving], scene);

  if (lhs.min_d2 > rhs.min_d2 + kMinD2Tol)
    return true;
  if (rhs.min_d2 > lhs.min_d2 + kMinD2Tol)
    return false;
  if (lhs.image_index != rhs.image_index)
    return lhs.image_index < rhs.image_index;
  return lhs.moving < rhs.moving;
}

HullPlacementScore evaluate_hull_placement_candidate(
    const HullChainGeom &geom, const HullAssemblyScene &scene, double tx,
    double ty, double tz, size_t moving, size_t image_index,
    vector<Vec3> &candidate_points, PbctopoHullScratch &hull_scratch) {
  HullPlacementScore candidate;
  candidate.moving = moving;
  candidate.image_index = image_index;
  candidate.tx = tx;
  candidate.ty = ty;
  candidate.tz = tz;

  if (geom.atom_points.empty())
    return candidate;
  if (shifted_chain_has_exact_hard_clash(geom, scene, tx, ty, tz))
    return candidate;

  build_shifted_hull_candidate_points(scene, geom, tx, ty, tz, candidate_points);
  candidate.hull_volume =
      compute_convex_hull_volume_or_zero(candidate_points, hull_scratch);
  candidate.valid = true;
  return candidate;
}

EnvelopePlacementScore score_chain_envelope_placement(
    const EnvelopeChainGeom &geom, const EnvelopeScene &scene, double tx,
    double ty, double tz, size_t moving, size_t anchor, size_t image_index,
    const EnvelopePlacementScore *incumbent = nullptr) {
  EnvelopePlacementScore candidate;
  candidate.moving = moving;
  candidate.anchor = anchor;
  candidate.image_index = image_index;
  candidate.tx = tx;
  candidate.ty = ty;
  candidate.tz = tz;
  candidate.valid = geom.atom_count > 0 && scene.atom_total > 0;
  if (!candidate.valid)
    return candidate;
  candidate.hard_clash = 0;
  candidate.soft_clash = 0;
  candidate.overlap_cells = 0;
  candidate.contact_atoms = 0;
  candidate.contact_cells = 0;
  candidate.center_d2 = 0.0;

  for (const auto &atom : geom.atom_cart) {
    const double rx = atom[0] + tx;
    const double ry = atom[1] + ty;
    const double rz = atom[2] + tz;
    const int hx = envelope_grid_index(rx, kClashGrid);
    const int hy = envelope_grid_index(ry, kClashGrid);
    const int hz = envelope_grid_index(rz, kClashGrid);

    bool atom_has_contact = false;
    for (int ox = -1; ox <= 1; ++ox) {
      for (int oy = -1; oy <= 1; ++oy) {
        for (int oz = -1; oz <= 1; ++oz) {
          const auto it = scene.clash_atoms.find(
              pack_envelope_cell(hx + ox, hy + oy, hz + oz));
          if (it == scene.clash_atoms.end())
            continue;
          for (const auto &other : it->second) {
            const double drx = rx - other[0];
            const double dry = ry - other[1];
            const double drz = rz - other[2];
            const double d2 = drx * drx + dry * dry + drz * drz;
            if (d2 < kHardClashCutoff2) {
              ++candidate.hard_clash;
              if (incumbent != nullptr && incumbent->valid &&
                  candidate.hard_clash > incumbent->hard_clash) {
                candidate.overlap_cells = 0;
                candidate.contact_atoms = 0;
                candidate.contact_cells = 0;
                candidate.center_d2 = std::numeric_limits<double>::infinity();
                return candidate;
              }
            } else if (d2 < kSoftClashCutoff2) {
              ++candidate.soft_clash;
              if (incumbent != nullptr && incumbent->valid &&
                  candidate.hard_clash == incumbent->hard_clash &&
                  candidate.soft_clash > incumbent->soft_clash) {
                candidate.overlap_cells = 0;
                candidate.contact_atoms = 0;
                candidate.contact_cells = 0;
                candidate.center_d2 = std::numeric_limits<double>::infinity();
                return candidate;
              }
            } else if (d2 < kContactCutoff2) {
              atom_has_contact = true;
            }
          }
        }
      }
    }
    if (atom_has_contact)
      ++candidate.contact_atoms;
  }

  candidate.overlap_cells = 0;
  for (const auto &center : geom.core_centers) {
    if (scene.core_cells.find(pack_envelope_cell(
            envelope_grid_index(center[0] + tx, kEnvelopeGrid),
            envelope_grid_index(center[1] + ty, kEnvelopeGrid),
            envelope_grid_index(center[2] + tz, kEnvelopeGrid))) !=
        scene.core_cells.end()) {
      ++candidate.overlap_cells;
    }
  }

  candidate.contact_cells = 0;
  for (const auto &center : geom.surface_centers) {
    const double px = center[0] + tx;
    const double py = center[1] + ty;
    const double pz = center[2] + tz;
    const int hx = envelope_grid_index(px, kContactCutoff);
    const int hy = envelope_grid_index(py, kContactCutoff);
    const int hz = envelope_grid_index(pz, kContactCutoff);

    bool has_surface_contact = false;
    for (int ox = -1; ox <= 1 && !has_surface_contact; ++ox) {
      for (int oy = -1; oy <= 1 && !has_surface_contact; ++oy) {
        for (int oz = -1; oz <= 1 && !has_surface_contact; ++oz) {
          const auto it = scene.surface_points.find(
              pack_envelope_cell(hx + ox, hy + oy, hz + oz));
          if (it == scene.surface_points.end())
            continue;
          for (const auto &other : it->second) {
            const double drx = px - other[0];
            const double dry = py - other[1];
            const double drz = pz - other[2];
            const double d2 = drx * drx + dry * dry + drz * drz;
            if (d2 >= kContactFloor2 && d2 <= kContactCutoff2) {
              has_surface_contact = true;
              break;
            }
          }
        }
      }
    }
    if (has_surface_contact)
      ++candidate.contact_cells;
  }

  const double inv_scene_atoms = 1.0 / static_cast<double>(scene.atom_total);
  const double scene_center_x = scene.center_sum_x * inv_scene_atoms;
  const double scene_center_y = scene.center_sum_y * inv_scene_atoms;
  const double scene_center_z = scene.center_sum_z * inv_scene_atoms;
  const double dcx = (geom.env_center_x + tx) - scene_center_x;
  const double dcy = (geom.env_center_y + ty) - scene_center_y;
  const double dcz = (geom.env_center_z + tz) - scene_center_z;
  candidate.center_d2 = dcx * dcx + dcy * dcy + dcz * dcz;
  return candidate;
}

EnvelopePlacementScore score_chain_envelope_placement_coarse(
    const EnvelopeChainGeom &geom, const EnvelopeScene &scene, double tx,
    double ty, double tz, size_t moving, size_t anchor, size_t image_index) {
  EnvelopePlacementScore candidate;
  candidate.moving = moving;
  candidate.anchor = anchor;
  candidate.image_index = image_index;
  candidate.tx = tx;
  candidate.ty = ty;
  candidate.tz = tz;
  candidate.valid = geom.atom_count > 0 && scene.atom_total > 0;
  if (!candidate.valid)
    return candidate;

  candidate.hard_clash = 0;
  candidate.soft_clash = 0;
  candidate.contact_atoms = 0;
  candidate.overlap_cells = 0;
  candidate.contact_cells = 0;

  for (const auto &center : geom.core_centers) {
    if (scene.core_cells.find(pack_envelope_cell(
            envelope_grid_index(center[0] + tx, kEnvelopeGrid),
            envelope_grid_index(center[1] + ty, kEnvelopeGrid),
            envelope_grid_index(center[2] + tz, kEnvelopeGrid))) !=
        scene.core_cells.end()) {
      ++candidate.overlap_cells;
    }
  }

  for (const auto &center : geom.surface_centers) {
    const double px = center[0] + tx;
    const double py = center[1] + ty;
    const double pz = center[2] + tz;
    const int hx = envelope_grid_index(px, kContactCutoff);
    const int hy = envelope_grid_index(py, kContactCutoff);
    const int hz = envelope_grid_index(pz, kContactCutoff);

    bool has_surface_contact = false;
    for (int ox = -1; ox <= 1 && !has_surface_contact; ++ox) {
      for (int oy = -1; oy <= 1 && !has_surface_contact; ++oy) {
        for (int oz = -1; oz <= 1 && !has_surface_contact; ++oz) {
          const auto it = scene.surface_points.find(
              pack_envelope_cell(hx + ox, hy + oy, hz + oz));
          if (it == scene.surface_points.end())
            continue;
          for (const auto &other : it->second) {
            const double drx = px - other[0];
            const double dry = py - other[1];
            const double drz = pz - other[2];
            const double d2 = drx * drx + dry * dry + drz * drz;
            if (d2 >= kContactFloor2 && d2 <= kContactCutoff2) {
              has_surface_contact = true;
              break;
            }
          }
        }
      }
    }
    if (has_surface_contact)
      ++candidate.contact_cells;
  }

  const double inv_scene_atoms = 1.0 / static_cast<double>(scene.atom_total);
  const double scene_center_x = scene.center_sum_x * inv_scene_atoms;
  const double scene_center_y = scene.center_sum_y * inv_scene_atoms;
  const double scene_center_z = scene.center_sum_z * inv_scene_atoms;
  const double dcx = (geom.env_center_x + tx) - scene_center_x;
  const double dcy = (geom.env_center_y + ty) - scene_center_y;
  const double dcz = (geom.env_center_z + tz) - scene_center_z;
  candidate.center_d2 = dcx * dcx + dcy * dcy + dcz * dcz;
  return candidate;
}

void insert_top_envelope_candidate(vector<EnvelopePlacementScore> &top_scores,
                                   const EnvelopePlacementScore &candidate,
                                   size_t limit) {
  if (!candidate.valid || limit == 0)
    return;
  auto pos = top_scores.begin();
  for (; pos != top_scores.end(); ++pos) {
    if (better_envelope_placement(candidate, *pos))
      break;
  }
  top_scores.insert(pos, candidate);
  if (top_scores.size() > limit)
    top_scores.pop_back();
}

void insert_top_coarse_envelope_candidate(
    vector<EnvelopePlacementScore> &top_scores,
    const EnvelopePlacementScore &candidate, size_t limit) {
  if (!candidate.valid || limit == 0)
    return;
  auto pos = top_scores.begin();
  for (; pos != top_scores.end(); ++pos) {
    if (better_coarse_envelope_placement(candidate, *pos))
      break;
  }
  top_scores.insert(pos, candidate);
  if (top_scores.size() > limit)
    top_scores.pop_back();
}

bool exact_envelope_candidate_has_hard_clash(const EnvelopeChainGeom &geom,
                                             const EnvelopeScene &scene,
                                             double tx, double ty, double tz,
                                             double &min_d2_out) {
  min_d2_out = std::numeric_limits<double>::infinity();
  if (geom.atom_cart.empty() || scene.clash_atoms.empty())
    return false;

  for (const auto &atom : geom.atom_cart) {
    const double rx = atom[0] + tx;
    const double ry = atom[1] + ty;
    const double rz = atom[2] + tz;
    const int hx = envelope_grid_index(rx, kClashGrid);
    const int hy = envelope_grid_index(ry, kClashGrid);
    const int hz = envelope_grid_index(rz, kClashGrid);
    for (int ox = -1; ox <= 1; ++ox) {
      for (int oy = -1; oy <= 1; ++oy) {
        for (int oz = -1; oz <= 1; ++oz) {
          const auto it = scene.clash_atoms.find(
              pack_envelope_cell(hx + ox, hy + oy, hz + oz));
          if (it == scene.clash_atoms.end())
            continue;
          for (const auto &other : it->second) {
            const double dx = rx - other[0];
            const double dy = ry - other[1];
            const double dz = rz - other[2];
            const double d2 = dx * dx + dy * dy + dz * dz;
            min_d2_out = std::min(min_d2_out, d2);
            if (d2 < kHardClashCutoff2)
              return true;
          }
        }
      }
    }
  }
  return false;
}

vector<EnvelopePlacementScore> refine_envelope_candidate_shortlist(
    const EnvelopeChainGeom &geom, const EnvelopeScene &scene,
    const vector<EnvelopePlacementScore> &coarse_top_scores, size_t limit) {
  vector<EnvelopePlacementScore> refined_scores;
  refined_scores.reserve(std::min(limit, coarse_top_scores.size()));
  for (const auto &coarse_candidate : coarse_top_scores) {
    const EnvelopePlacementScore refined_candidate =
        score_chain_envelope_placement(
            geom, scene, coarse_candidate.tx, coarse_candidate.ty,
            coarse_candidate.tz, coarse_candidate.moving,
            coarse_candidate.anchor, coarse_candidate.image_index,
            refined_scores.empty() ? nullptr : &refined_scores.front());
    insert_top_envelope_candidate(refined_scores, refined_candidate, limit);
  }
  return refined_scores;
}

EnvelopePlacementScore finalize_envelope_candidate_selection(
    const EnvelopeChainGeom &geom, const EnvelopeScene &scene,
    const vector<EnvelopePlacementScore> &top_scores) {
  if (top_scores.empty())
    return {};

  EnvelopePlacementScore best = top_scores.front();
  EnvelopePlacementScore refined_best;
  bool have_refined = false;
  double refined_min_d2 = -1.0;

  for (const auto &candidate : top_scores) {
    double min_d2 = std::numeric_limits<double>::infinity();
    if (exact_envelope_candidate_has_hard_clash(geom, scene, candidate.tx,
                                                candidate.ty, candidate.tz,
                                                min_d2)) {
      continue;
    }

    if (!have_refined || better_envelope_placement(candidate, refined_best) ||
        (!better_envelope_placement(refined_best, candidate) &&
         min_d2 > refined_min_d2)) {
      refined_best = candidate;
      refined_min_d2 = min_d2;
      have_refined = true;
    }
  }

  return have_refined ? refined_best : best;
}

vector<LegacyImageShift> build_legacy_image_shifts(box box_in) {
  double v1[3]{}, v2[3]{}, v3[3]{};
  box_in.get_v1(v1);
  box_in.get_v2(v2);
  box_in.get_v3(v3);

  vector<LegacyImageShift> shifts;
  shifts.reserve(27);
  shifts.push_back({26, 0.0, 0.0, 0.0});
  for (size_t idx = 0; idx < 26; ++idx) {
    const int *coeff = kLegacyImageCoeff[idx];
    shifts.push_back({idx,
                      coeff[0] * v1[0] + coeff[1] * v2[0] + coeff[2] * v3[0],
                      coeff[1] * v2[1] + coeff[2] * v3[1],
                      coeff[2] * v3[2]});
  }
  return shifts;
}

} // namespace

// Issue 5: O(N) scan — acceptable since list_image has at most 26 elements.
inline bool add_list_images(vector<size_t> &list_image, size_t &&i) {
  for (auto &image : list_image)
    if (image == i)
      return false;
  list_image.emplace_back(i);
  return true;
}

namespace {

struct CartesianChainRange {
  size_t begin{};
  size_t end{};
};

vector<CartesianChainRange>
build_cartesian_chain_ranges(const vector<class index> &Index, size_t atom_count) {
  vector<CartesianChainRange> ranges;
  ranges.reserve(Index.size());
  for (size_t k = 0; k + 1 < Index.size(); ++k) {
    const size_t begin = Index[k].get_res_pos_e(0) - 1;
    const size_t end = Index[k + 1].get_res_pos_e(0) - 1;
    if (begin >= atom_count || end > atom_count || begin >= end)
      continue;
    ranges.push_back({begin, end});
  }
  if (ranges.empty() && atom_count > 0)
    ranges.push_back({0, atom_count});
  return ranges;
}

vector<EnvelopeChainGeom>
build_envelope_chain_geometries(const MOL &mol,
                                const vector<CartesianChainRange> &ranges) {
  vector<EnvelopeChainGeom> geom;
  geom.reserve(ranges.size());
  for (const auto &range : ranges) {
    vector<array<double, 3>> atom_cart;
    atom_cart.reserve(range.end - range.begin);
    for (size_t i = range.begin; i < range.end; ++i)
      atom_cart.push_back({mol.pos.x[i], mol.pos.y[i], mol.pos.z[i]});
    geom.push_back(build_envelope_chain_geom(std::move(atom_cart)));
  }
  return geom;
}

void apply_range_shift(MOL &mol, const CartesianChainRange &range, double tx,
                       double ty, double tz);

array<unsigned char, 27>
build_rejected_image_mask(const vector<size_t> &rejected_images);

vector<HullChainGeom>
build_hull_chain_geometries(const MOL &mol,
                            const vector<CartesianChainRange> &ranges) {
  vector<HullChainGeom> geom;
  geom.reserve(ranges.size());
  for (const auto &range : ranges) {
    vector<Vec3> atom_points;
    atom_points.reserve(range.end - range.begin);
    for (size_t i = range.begin; i < range.end; ++i) {
      atom_points.push_back({mol.pos.x[i], mol.pos.y[i], mol.pos.z[i]});
    }
    geom.push_back(build_hull_chain_geom(std::move(atom_points)));
  }
  return geom;
}

int assemble_chains_with_hull(MOL &mol, const vector<class index> &Index,
                              int seed_chain,
                              const vector<size_t> &rejected_images,
                              const vector<char> *skip_mask_ptr = nullptr) {
  const auto ranges = build_cartesian_chain_ranges(Index, mol.pos.x.size());
  if (ranges.empty())
    return 0;

  vector<char> skip_mask(ranges.size(), 0);
  if (skip_mask_ptr != nullptr) {
    const size_t limit = std::min(skip_mask.size(), skip_mask_ptr->size());
    for (size_t i = 0; i < limit; ++i)
      skip_mask[i] = (*skip_mask_ptr)[i];
  }

  size_t active_chain_count = 0;
  size_t active_atom_hint = 0;
  const auto geom = build_hull_chain_geometries(mol, ranges);
  for (size_t i = 0; i < geom.size(); ++i) {
    if (skip_mask[i])
      continue;
    ++active_chain_count;
    active_atom_hint += geom[i].atom_count;
  }
  if (active_chain_count == 0)
    return 0;

  if (seed_chain < 0 || seed_chain >= static_cast<int>(ranges.size()) ||
      skip_mask[static_cast<size_t>(seed_chain)] != 0) {
    seed_chain = 0;
    while (static_cast<size_t>(seed_chain) < skip_mask.size() &&
           skip_mask[static_cast<size_t>(seed_chain)] != 0) {
      ++seed_chain;
    }
    if (static_cast<size_t>(seed_chain) >= skip_mask.size())
      return 0;
  }

  const auto candidate_shifts = build_legacy_image_shifts(mol.Box);
  const auto rejected_mask = build_rejected_image_mask(rejected_images);

  vector<char> assembled(ranges.size(), 0);
  vector<array<double, 3>> chosen_shift(ranges.size(), {0.0, 0.0, 0.0});
  HullAssemblyScene scene;
  reserve_hull_scene(scene, active_atom_hint);
  assembled[static_cast<size_t>(seed_chain)] = 1;
  add_chain_to_hull_scene(scene, geom[static_cast<size_t>(seed_chain)], 0.0, 0.0,
                          0.0);

  size_t remaining = active_chain_count - 1;
  vector<Vec3> candidate_points;
  candidate_points.reserve(active_atom_hint);
  PbctopoHullScratch hull_scratch;
  hull_scratch.reserve(active_atom_hint);

  while (remaining > 0) {
    HullPlacementScore best_score;
    for (size_t moving = 0; moving < ranges.size(); ++moving) {
      if (skip_mask[moving] || assembled[moving])
        continue;

      for (const auto &shift : candidate_shifts) {
        if (shift.image_index < rejected_mask.size() &&
            rejected_mask[shift.image_index] != 0) {
          continue;
        }

        HullPlacementScore candidate = evaluate_hull_placement_candidate(
            geom[moving], scene, shift.tx, shift.ty, shift.tz, moving,
            shift.image_index, candidate_points, hull_scratch);
        if (!candidate.valid)
          continue;
        if (better_hull_placement(candidate, best_score, geom, scene))
          best_score = candidate;
      }
    }

    if (!best_score.valid) {
      for (size_t moving = 0; moving < ranges.size(); ++moving) {
        if (skip_mask[moving] || assembled[moving])
          continue;
        HullPlacementScore candidate = evaluate_hull_placement_candidate(
            geom[moving], scene, 0.0, 0.0, 0.0, moving, 26, candidate_points,
            hull_scratch);
        if (!candidate.valid)
          continue;
        if (better_hull_placement(candidate, best_score, geom, scene))
          best_score = candidate;
      }
    }

    if (!best_score.valid) {
      for (size_t moving = 0; moving < ranges.size(); ++moving) {
        if (skip_mask[moving] || assembled[moving])
          continue;
        assembled[moving] = 1;
        --remaining;
        break;
      }
      continue;
    }

    chosen_shift[best_score.moving] = {best_score.tx, best_score.ty,
                                       best_score.tz};
    assembled[best_score.moving] = 1;
    add_chain_to_hull_scene(scene, geom[best_score.moving], best_score.tx,
                            best_score.ty, best_score.tz);
    --remaining;
  }

  for (size_t chain = 0; chain < ranges.size(); ++chain) {
    if (skip_mask[chain])
      continue;
    apply_range_shift(mol, ranges[chain], chosen_shift[chain][0],
                      chosen_shift[chain][1], chosen_shift[chain][2]);
  }

  return 0;
}

array<unsigned char, 27>
build_rejected_image_mask(const vector<size_t> &rejected_images) {
  array<unsigned char, 27> mask{};
  for (size_t rejected : rejected_images) {
    if (rejected < mask.size())
      mask[rejected] = 1;
  }
  return mask;
}

bool init_scaled_box_vectors(box box_in, double v1[3], double v2[3],
                             double v3[3]) {
  box_in.get_v1(v1);
  box_in.get_v2(v2);
  box_in.get_v3(v3);
  return v1[0] != 0.0 && v2[1] != 0.0 && v3[2] != 0.0;
}

void init_fixbox_box_center(const double v1[3], const double v2[3],
                            const double v3[3], double box_center[3]) {
  box_center[0] = 0.5 * (v1[0] + v2[0] + v3[0]);
  box_center[1] = 0.5 * (v1[1] + v2[1] + v3[1]);
  box_center[2] = 0.5 * (v1[2] + v2[2] + v3[2]);
}

void cartesian_to_fixbox_scaled(const double v1[3], const double v2[3],
                                const double v3[3], const double box_center[3],
                                double rx, double ry, double rz, double &sx,
                                double &sy, double &sz) {
  const double crx = rx - box_center[0];
  const double cry = ry - box_center[1];
  const double crz = rz - box_center[2];
  sz = crz / v3[2];
  sy = (cry - v3[1] * sz) / v2[1];
  sx = (crx - v2[0] * sy - v3[0] * sz) / v1[0];
}

void scaled_to_cartesian(const double v1[3], const double v2[3],
                         const double v3[3], double sx, double sy, double sz,
                         double &rx, double &ry, double &rz) {
  rx = v1[0] * sx + v2[0] * sy + v3[0] * sz;
  ry = v2[1] * sy + v3[1] * sz;
  rz = v3[2] * sz;
}

void fixbox_scaled_to_cartesian(const double v1[3], const double v2[3],
                                const double v3[3], const double box_center[3],
                                double sx, double sy, double sz, double &rx,
                                double &ry, double &rz) {
  scaled_to_cartesian(v1, v2, v3, sx, sy, sz, rx, ry, rz);
  rx += box_center[0];
  ry += box_center[1];
  rz += box_center[2];
}

void unwrap_scaled_chain_ranges(vector<double> &sx, vector<double> &sy,
                                vector<double> &sz, const vector<class index> &Index,
                                size_t atom_count) {
  for (size_t k = 0; k + 1 < Index.size(); ++k) {
    const size_t begin = Index[k].get_res_pos_e(0) - 1;
    const size_t end = Index[k + 1].get_res_pos_e(0) - 1;
    if (begin >= atom_count || end > atom_count || begin >= end)
      continue;

    double ref_x = sx[begin];
    double ref_y = sy[begin];
    double ref_z = sz[begin];
    for (size_t i = begin + 1; i < end; ++i) {
      sx[i] -= std::round(sx[i] - ref_x);
      sy[i] -= std::round(sy[i] - ref_y);
      sz[i] -= std::round(sz[i] - ref_z);
      ref_x = sx[i];
      ref_y = sy[i];
      ref_z = sz[i];
    }
  }
}

void unwrap_mol_chains_in_scaled_space(MOL &mol, const vector<class index> &Index,
                                       size_t atom_count) {
  if (atom_count == 0 || atom_count > mol.pos.x.size())
    atom_count = mol.pos.x.size();
  if (atom_count == 0)
    return;

  double v1[3]{}, v2[3]{}, v3[3]{};
  if (!init_scaled_box_vectors(mol.Box, v1, v2, v3))
    return;

  vector<double> sx(atom_count), sy(atom_count), sz(atom_count);
  for (size_t i = 0; i < atom_count; ++i) {
    const double r[3]{mol.pos.x[i], mol.pos.y[i], mol.pos.z[i]};
    double s[3]{};
    mol.Box.to_scaled(r, s);
    sx[i] = s[0];
    sy[i] = s[1];
    sz[i] = s[2];
  }

  unwrap_scaled_chain_ranges(sx, sy, sz, Index, atom_count);

  for (size_t i = 0; i < atom_count; ++i) {
    scaled_to_cartesian(v1, v2, v3, sx[i], sy[i], sz[i], mol.pos.x[i],
                        mol.pos.y[i], mol.pos.z[i]);
  }
}

struct FixBoxChainRange {
  size_t begin{};
  size_t end{};
  bool ligand{false};
  bool assemble{true};
};

vector<FixBoxChainRange>
build_fixbox_chain_ranges(const vector<class index> &Index, size_t atom_count,
                          bool ligand_flag) {
  vector<FixBoxChainRange> ranges;
  ranges.reserve(Index.size());
  for (size_t k = 0; k + 1 < Index.size(); ++k) {
    const size_t begin = Index[k].get_res_pos_e(0) - 1;
    const size_t end = Index[k + 1].get_res_pos_e(0) - 1;
    if (begin >= atom_count || end > atom_count || begin >= end)
      continue;
    ranges.push_back({begin, end, ligand_flag, true});
  }
  if (ranges.empty() && atom_count > 0)
    ranges.push_back({0, atom_count, ligand_flag, true});
  return ranges;
}

long fixbox_nint(double x) { return static_cast<long>(std::floor(x + 0.5)); }

double fixbox_cart_dist_sq(const double v1[3], const double v2[3],
                           const double v3[3], double dsx, double dsy,
                           double dsz) {
  const double rx = v1[0] * dsx + v2[0] * dsy + v3[0] * dsz;
  const double ry = v2[1] * dsy + v3[1] * dsz;
  const double rz = v3[2] * dsz;
  return rx * rx + ry * ry + rz * rz;
}

void apply_fixbox_pbcn_frame(vector<double> &sx_rec, vector<double> &sy_rec,
                             vector<double> &sz_rec, vector<double> &sx_lig,
                             vector<double> &sy_lig, vector<double> &sz_lig,
                             bool has_ligand, const vector<class index> &rec_index,
                             const vector<class index> &lig_index, int chain_ref,
                             const arguments &Arguments, const double v1[3],
                             const double v2[3], const double v3[3]) {
  (void)chain_ref;
  vector<FixBoxChainRange> rec_ranges =
      build_fixbox_chain_ranges(rec_index, sx_rec.size(), false);
  vector<FixBoxChainRange> lig_ranges;
  if (has_ligand && !sx_lig.empty())
    lig_ranges = build_fixbox_chain_ranges(lig_index, sx_lig.size(), true);

  if (Arguments.pbc.pbc_envi == true) {
    for (size_t k = 0; k < rec_ranges.size() && k + 1 < rec_index.size(); ++k) {
      for (const auto &chain_name : Arguments.pbc.pbc_envi_chain) {
        if (rec_index[k].get_chain() == chain_name) {
          rec_ranges[k].assemble = false;
          break;
        }
      }
    }
  }

  vector<FixBoxChainRange> chains;
  chains.reserve(rec_ranges.size() + lig_ranges.size());
  chains.insert(chains.end(), rec_ranges.begin(), rec_ranges.end());
  chains.insert(chains.end(), lig_ranges.begin(), lig_ranges.end());
  if (chains.empty())
    return;

  vector<long> dnx(chains.size(), 0);
  vector<long> dny(chains.size(), 0);
  vector<long> dnz(chains.size(), 0);
  vector<char> assembled(chains.size(), 0);

  auto select_sx = [&](const FixBoxChainRange &range) -> vector<double> & {
    return range.ligand ? sx_lig : sx_rec;
  };
  auto select_sy = [&](const FixBoxChainRange &range) -> vector<double> & {
    return range.ligand ? sy_lig : sy_rec;
  };
  auto select_sz = [&](const FixBoxChainRange &range) -> vector<double> & {
    return range.ligand ? sz_lig : sz_rec;
  };
  auto select_sx_const =
      [&](const FixBoxChainRange &range) -> const vector<double> & {
    return range.ligand ? sx_lig : sx_rec;
  };
  auto select_sy_const =
      [&](const FixBoxChainRange &range) -> const vector<double> & {
    return range.ligand ? sy_lig : sy_rec;
  };
  auto select_sz_const =
      [&](const FixBoxChainRange &range) -> const vector<double> & {
    return range.ligand ? sz_lig : sz_rec;
  };

  struct PairChoice {
    size_t moving = static_cast<size_t>(-1);
    size_t anchor = static_cast<size_t>(-1);
    long sx = 0;
    long sy = 0;
    long sz = 0;
    double d2 = std::numeric_limits<double>::infinity();
    bool valid = false;
  };

  auto build_anchor_list = [&](const vector<size_t> &stage_indices,
                               bool allow_all_assembled_anchors) {
    vector<size_t> anchors;
    anchors.reserve(chains.size());
    for (size_t idx = 0; idx < chains.size(); ++idx) {
      if (!assembled[idx])
        continue;
      if (!allow_all_assembled_anchors &&
          std::find(stage_indices.begin(), stage_indices.end(), idx) ==
              stage_indices.end()) {
        continue;
      }
      anchors.push_back(idx);
    }
    return anchors;
  };

  auto find_best_pair = [&](const vector<size_t> &stage_indices,
                            bool allow_all_assembled_anchors) {
    const vector<size_t> anchors =
        build_anchor_list(stage_indices, allow_all_assembled_anchors);
    PairChoice best;
    for (size_t moving : stage_indices) {
      if (assembled[moving])
        continue;
      const FixBoxChainRange &moving_range = chains[moving];
      const vector<double> &sx_m = select_sx_const(moving_range);
      const vector<double> &sy_m = select_sy_const(moving_range);
      const vector<double> &sz_m = select_sz_const(moving_range);
      for (size_t anchor : anchors) {
        const FixBoxChainRange &anchor_range = chains[anchor];
        const vector<double> &sx_a = select_sx_const(anchor_range);
        const vector<double> &sy_a = select_sy_const(anchor_range);
        const vector<double> &sz_a = select_sz_const(anchor_range);
        for (size_t i = moving_range.begin; i < moving_range.end; ++i) {
          for (size_t j = anchor_range.begin; j < anchor_range.end; ++j) {
            const double dsx = sx_a[j] - sx_m[i];
            const double dsy = sy_a[j] - sy_m[i];
            const double dsz = sz_a[j] - sz_m[i];
            const long nx = fixbox_nint(dsx);
            const long ny = fixbox_nint(dsy);
            const long nz = fixbox_nint(dsz);
            const double d2 = fixbox_cart_dist_sq(
                v1, v2, v3, dsx - static_cast<double>(nx),
                dsy - static_cast<double>(ny), dsz - static_cast<double>(nz));
            if (d2 < best.d2) {
              best.moving = moving;
              best.anchor = anchor;
              best.sx = dnx[anchor] + nx;
              best.sy = dny[anchor] + ny;
              best.sz = dnz[anchor] + nz;
              best.d2 = d2;
              best.valid = true;
            }
          }
        }
      }
    }
    return best;
  };

  auto assemble_stage = [&](const vector<size_t> &stage_indices,
                            bool allow_all_assembled_anchors,
                            bool seed_if_empty) {
    if (stage_indices.empty())
      return;

    bool has_anchor = false;
    for (size_t idx = 0; idx < chains.size(); ++idx) {
      if (assembled[idx] &&
          (allow_all_assembled_anchors ||
           std::find(stage_indices.begin(), stage_indices.end(), idx) !=
               stage_indices.end())) {
        has_anchor = true;
        break;
      }
    }
    if (!has_anchor && seed_if_empty) {
      assembled[stage_indices.front()] = 1;
    }

    size_t remaining = 0;
    for (size_t idx : stage_indices)
      remaining += assembled[idx] ? 0u : 1u;

    while (remaining > 0) {
      const PairChoice best =
          find_best_pair(stage_indices, allow_all_assembled_anchors);
      if (!best.valid) {
        for (size_t idx : stage_indices) {
          if (!assembled[idx]) {
            assembled[idx] = 1;
            --remaining;
            break;
          }
        }
        continue;
      }

      dnx[best.moving] = best.sx;
      dny[best.moving] = best.sy;
      dnz[best.moving] = best.sz;
      assembled[best.moving] = 1;
      --remaining;
    }
  };

  vector<size_t> rec_stage;
  rec_stage.reserve(rec_ranges.size());
  for (size_t idx = 0; idx < rec_ranges.size(); ++idx) {
    if (chains[idx].assemble)
      rec_stage.push_back(idx);
  }

  vector<size_t> lig_stage;
  lig_stage.reserve(lig_ranges.size());
  for (size_t idx = rec_ranges.size(); idx < chains.size(); ++idx) {
    if (chains[idx].assemble)
      lig_stage.push_back(idx);
  }

  // Titan maps to a simple implicit FixBox moldef:
  //   A Receptor
  //   A Ligand   (if present)
  //   C Receptor Receptor Receptor
  //   P None None None
  // The optional chain-unwrapping prep happens outside this core because
  // FixBox itself assumes whole molecules in the input trajectory.
  if (!rec_stage.empty())
    assembled[rec_stage.front()] = 1;
  else if (!lig_stage.empty())
    assembled[lig_stage.front()] = 1;

  if (!rec_stage.empty())
    assemble_stage(rec_stage, false, false);
  if (!lig_stage.empty())
    assemble_stage(lig_stage, true, rec_stage.empty());

  for (size_t idx = 0; idx < chains.size(); ++idx) {
    if (!assembled[idx])
      continue;
    const FixBoxChainRange &range = chains[idx];
    vector<double> &sx = select_sx(range);
    vector<double> &sy = select_sy(range);
    vector<double> &sz = select_sz(range);
    for (size_t i = range.begin; i < range.end; ++i) {
      sx[i] += static_cast<double>(dnx[idx]);
      sy[i] += static_cast<double>(dny[idx]);
      sz[i] += static_cast<double>(dnz[idx]);
    }
  }

  if (!Arguments.pbc.no_center) {
    const vector<size_t> &center_stage = !rec_stage.empty() ? rec_stage : lig_stage;
    for (int k = 0; k < 3; ++k) {
      double smin = std::numeric_limits<double>::infinity();
      double smax = -std::numeric_limits<double>::infinity();
      bool have_center_group = false;
      for (size_t idx : center_stage) {
        if (!assembled[idx])
          continue;
        have_center_group = true;
        const FixBoxChainRange &range = chains[idx];
        const vector<double> &s =
            (k == 0) ? select_sx_const(range)
                     : ((k == 1) ? select_sy_const(range) : select_sz_const(range));
        for (size_t i = range.begin; i < range.end; ++i) {
          smin = std::min(smin, s[i]);
          smax = std::max(smax, s[i]);
        }
      }
      if (!have_center_group)
        continue;

      const double smid = 0.5 * (smin + smax);
      if (k == 0) {
        for (double &v : sx_rec)
          v -= smid;
        for (double &v : sx_lig)
          v -= smid;
      } else if (k == 1) {
        for (double &v : sy_rec)
          v -= smid;
        for (double &v : sy_lig)
          v -= smid;
      } else {
        for (double &v : sz_rec)
          v -= smid;
        for (double &v : sz_lig)
          v -= smid;
      }
    }
  }
}

void apply_range_shift(MOL &mol, const CartesianChainRange &range, double tx,
                       double ty, double tz) {
  if (tx == 0.0 && ty == 0.0 && tz == 0.0)
    return;
  for (size_t i = range.begin; i < range.end; ++i) {
    mol.pos.x[i] += tx;
    mol.pos.y[i] += ty;
    mol.pos.z[i] += tz;
  }
}

int assemble_chains_with_envelope(MOL &mol, const vector<class index> &Index,
                                  int seed_chain,
                                  const vector<size_t> &rejected_images) {
  return assemble_chains_with_hull(mol, Index, seed_chain, rejected_images,
                                   nullptr);
}

int place_molecule_with_envelope(const MOL &anchor, const vector<class index> &Index,
                                 MOL &moving,
                                 const vector<size_t> &rejected_images) {
  if (anchor.pos.x.empty() || moving.pos.x.empty())
    return 0;

  const auto anchor_ranges =
      build_cartesian_chain_ranges(Index, anchor.pos.x.size());
  if (anchor_ranges.empty())
    return 0;

  const auto anchor_geom = build_hull_chain_geometries(anchor, anchor_ranges);
  const auto image_shifts = build_legacy_image_shifts(moving.Box);
  const auto rejected_mask = build_rejected_image_mask(rejected_images);

  vector<CartesianChainRange> moving_ranges{{0, moving.pos.x.size()}};
  const auto moving_geom = build_hull_chain_geometries(moving, moving_ranges);
  size_t anchor_atom_hint = 0;
  for (const auto &chain_geom : anchor_geom) {
    anchor_atom_hint += chain_geom.atom_count;
  }

  HullAssemblyScene scene;
  reserve_hull_scene(scene, anchor_atom_hint);
  for (const auto &chain_geom : anchor_geom)
    add_chain_to_hull_scene(scene, chain_geom, 0.0, 0.0, 0.0);

  vector<Vec3> candidate_points;
  candidate_points.reserve(anchor_atom_hint + moving_geom[0].atom_count);
  PbctopoHullScratch hull_scratch;
  hull_scratch.reserve(anchor_atom_hint + moving_geom[0].atom_count);

  HullPlacementScore best_score;
  for (const auto &shift : image_shifts) {
    if (shift.image_index < rejected_mask.size() &&
        rejected_mask[shift.image_index] != 0)
      continue;
    HullPlacementScore candidate = evaluate_hull_placement_candidate(
        moving_geom[0], scene, shift.tx, shift.ty, shift.tz, 0,
        shift.image_index, candidate_points, hull_scratch);
    if (!candidate.valid)
      continue;
    if (better_hull_placement(candidate, best_score, moving_geom, scene))
      best_score = candidate;
  }

  if (!best_score.valid) {
    HullPlacementScore identity_candidate = evaluate_hull_placement_candidate(
        moving_geom[0], scene, 0.0, 0.0, 0.0, 0, 26, candidate_points,
        hull_scratch);
    if (identity_candidate.valid)
      best_score = identity_candidate;
  }

  if (!best_score.valid)
    return 0;

  apply_range_shift(moving, moving_ranges[0], best_score.tx, best_score.ty,
                    best_score.tz);
  return 0;
}

} // namespace

size_t MOL::find_chain_ref_pbc(arguments &Arguments,
                               vector<class index> &Index) {
  vector<vector<double>> cogs;
  for (size_t i{}; i < Index.size(); ++i) {
    double cog[3]{};
    this->find_cog_chain(i, Arguments, Index, cog);
    vector<double> cog_t{cog[0], cog[1], cog[2]};
    cogs.emplace_back(cog_t);
  }

  double cog_allchains[3]{};
  for (size_t i{}; i < cogs.size(); ++i) {
    cog_allchains[0] += cogs[i][0];
    cog_allchains[1] += cogs[i][1];
    cog_allchains[2] += cogs[i][2];
  }

  cog_allchains[0] /= cogs.size();
  cog_allchains[1] /= cogs.size();
  cog_allchains[2] /= cogs.size();

  double mindis{10e9};
  size_t pos{};
  for (size_t i{}; i < cogs.size(); ++i) {
    double dis{sqrt(square(cog_allchains[0] - cogs[i][0]) +
                    square(cog_allchains[1] - cogs[i][1]) +
                    square(cog_allchains[2] - cogs[i][2]))};
    if (mindis > dis) {
      mindis = dis;
      pos = i;
    }
  }
  return pos;
}

int MOL::find_images_suitable(size_t chain, size_t res,
                              std::vector<class index> &Index,
                              std::vector<size_t> &list_image) const {
  double com[3]{};
  size_t atom_count{};
  size_t limit(res == Index[chain].get_size_v() - 1
                   ? Index[chain + 1].get_res_pos_e(0) - 1
                   : Index[chain].get_res_pos_e(res + 1) - 1);
  for (size_t i = Index[chain].get_res_pos_e(res) - 1; i < limit; ++i) {
    com[0] += this->pos.x[i];
    com[1] += this->pos.y[i];
    com[2] += this->pos.z[i];
    ++atom_count;
  }
  com[0] /= atom_count;
  com[1] /= atom_count;
  com[2] /= atom_count;
  add_list_images(list_image, 26);

  // FixBox (Baptista et al. 2022): use scaled coordinates s = B⁻¹·com
  // For orthorhombic: s[i] = com[i]/L[i], so s[i]>0.5 ↔ com[i]>L[i]/2
  double s[3]{};
  Box.to_scaled(com, s);

  if (s[0] > 0.5) {
    add_list_images(list_image, 0);
    add_list_images(list_image, 6);
    add_list_images(list_image, 7);
    add_list_images(list_image, 10);
    add_list_images(list_image, 11);
    add_list_images(list_image, 14);
    add_list_images(list_image, 16);
    add_list_images(list_image, 18);
    add_list_images(list_image, 20);
    add_list_images(list_image, 21);
    add_list_images(list_image, 24);
  } else {
    add_list_images(list_image, 1);
    add_list_images(list_image, 8);
    add_list_images(list_image, 9);
    add_list_images(list_image, 12);
    add_list_images(list_image, 13);
    add_list_images(list_image, 19);
    add_list_images(list_image, 22);
    add_list_images(list_image, 23);
    add_list_images(list_image, 25);
  }
  if (s[1] > 0.5) {
    add_list_images(list_image, 2);
    add_list_images(list_image, 6);
    add_list_images(list_image, 8);
    add_list_images(list_image, 14);
    add_list_images(list_image, 16);
    add_list_images(list_image, 18);
    add_list_images(list_image, 19);
    add_list_images(list_image, 21);
    add_list_images(list_image, 23);
  } else {
    add_list_images(list_image, 3);
    add_list_images(list_image, 7);
    add_list_images(list_image, 9);
    add_list_images(list_image, 15);
    add_list_images(list_image, 17);
    add_list_images(list_image, 20);
    add_list_images(list_image, 22);
    add_list_images(list_image, 24);
    add_list_images(list_image, 25);
  }
  if (s[2] > 0.5) {
    add_list_images(list_image, 4);
    add_list_images(list_image, 10);
    add_list_images(list_image, 12);
    add_list_images(list_image, 14);
    add_list_images(list_image, 15);
    add_list_images(list_image, 18);
    add_list_images(list_image, 19);
    add_list_images(list_image, 20);
    add_list_images(list_image, 22);
  } else {
    add_list_images(list_image, 5);
    add_list_images(list_image, 11);
    add_list_images(list_image, 13);
    add_list_images(list_image, 16);
    add_list_images(list_image, 17);
    add_list_images(list_image, 21);
    add_list_images(list_image, 23);
    add_list_images(list_image, 24);
    add_list_images(list_image, 25);
  }
  return 0;
}

double MOL::find_min_dis_2res(size_t chain, size_t chain2, size_t res1,
                              size_t res2, Multi_MOL &images,
                              std::vector<class index> &Index,
                              size_t pos_image) const {
  double com[3]{}, com2[3]{};
  size_t atom_count{};
  size_t lim1 = (res1 == Index[chain].get_size_v() - 1
                     ? Index[chain + 1].get_res_pos_e(0) - 1
                     : Index[chain].get_res_pos_e(res1 + 1) - 1);
  for (size_t i = Index[chain].get_res_pos_e(res1) - 1; i < lim1; ++i) {
    com[0] += this->pos.x[i];
    com[1] += this->pos.y[i];
    com[2] += this->pos.z[i];
    ++atom_count;
  }
  com[0] /= atom_count;
  com[1] /= atom_count;
  com[2] /= atom_count;

  size_t limit = (res2 == Index[chain2].get_size_v() - 1
                      ? Index[chain2 + 1].get_res_pos_e(0) - 1
                      : Index[chain2].get_res_pos_e(res2 + 1) - 1);
  atom_count = 0;
  for (size_t i = Index[chain2].get_res_pos_e(res2) - 1; i < limit; ++i) {
    com2[0] += images.m_mol[pos_image].pos.x[i];
    com2[1] += images.m_mol[pos_image].pos.y[i];
    com2[2] += images.m_mol[pos_image].pos.z[i];
    ++atom_count;
  }
  com2[0] /= atom_count;
  com2[1] /= atom_count;
  com2[2] /= atom_count;
  return sqrt(square(com[0] - com2[0]) + square(com[1] - com2[1]) +
              square(com[2] - com2[2]));
}

double MOL::find_dis_res_point(size_t chain, size_t res, Multi_MOL &images,
                               std::vector<class index> &Index, size_t pos_image,
                               std::span<const double> point) const {
  double com[3]{};
  size_t atom_count{};
  size_t limit = (res == Index[chain].get_size_v() - 1
                      ? Index[chain + 1].get_res_pos_e(0) - 1
                      : Index[chain].get_res_pos_e(res + 1) - 1);
  for (size_t i = Index[chain].get_res_pos_e(res) - 1; i < limit; ++i) {
    com[0] += images.m_mol[pos_image].pos.x[i];
    com[1] += images.m_mol[pos_image].pos.y[i];
    com[2] += images.m_mol[pos_image].pos.z[i];
    ++atom_count;
  }
  com[0] /= atom_count;
  com[1] /= atom_count;
  com[2] /= atom_count;
  return sqrt(square(com[0] - point[0]) + square(com[1] - point[1]) +
              square(com[2] - point[2]));
}

int Multi_MOL::solv_pbc_envi(size_t ind, arguments &Arguments,
                             vector<size_t> &rejected_images, size_t k,
                             Multi_MOL &pbc_images,
                             std::span<double> com_ref_chain) {

  for (size_t i{}; i < Index[k].get_size_v(); ++i) {
    double dis_min{10e10};
    size_t pos{};

    for (size_t j{}; j < pbc_images.m_mol.size(); ++j) {
      if (check_in_rejected_images(rejected_images, j) == true)
        continue;

      double d{this->m_mol[ind].find_dis_res_point(k, i, pbc_images, Index, j,
                                                   com_ref_chain)};

      if (d < dis_min) {
        dis_min = d;
        pos = j;
      }
    }

    size_t limit =
        (i == Index[k].get_size_v() - 1 ? Index[k + 1].get_res_pos_e(0) - 1
                                        : Index[k].get_res_pos_e(i + 1) - 1);
    for (size_t j = Index[k].get_res_pos_e(i) - 1; j < limit; ++j) {
      this->m_mol[ind].pos.x[j] = pbc_images.m_mol[pos].pos.x[j];
      this->m_mol[ind].pos.y[j] = pbc_images.m_mol[pos].pos.y[j];
      this->m_mol[ind].pos.z[j] = pbc_images.m_mol[pos].pos.z[j];
    }
  }
  return 0;
}

int MOL::find_cog_chain(size_t k, arguments &Arguments, vector<class index> &Index,
                        std::span<double> center_of_geometry) {
  // Zero-init before accumulating so double-call is safe
  center_of_geometry[0] = center_of_geometry[1] = center_of_geometry[2] = 0.0;
  size_t atom_num{};
  for (size_t i{Index[k].get_res_pos_e(0) - 1};
       i < Index[k + 1].get_res_pos_e(0) - 1; ++i) {
    center_of_geometry[0] += this->pos.x[i];
    center_of_geometry[1] += this->pos.y[i];
    center_of_geometry[2] += this->pos.z[i];
    ++atom_num;
  }
  if (atom_num > 0) {
    center_of_geometry[0] /= atom_num;
    center_of_geometry[1] /= atom_num;
    center_of_geometry[2] /= atom_num;
  }
  return 0;
}

double Multi_MOL::find_min_dis2(size_t ind, size_t ind2, size_t chain,
                                size_t chain2, Multi_MOL &images) const {
  double mindistance{10e12};

  for (size_t i{Index[chain].get_res_pos_e(0) - 1};
       i < Index[chain + 1].get_res_pos_e(0) - 1; ++i) {
    for (size_t j{images.Index[chain2].get_res_pos_e(0) - 1};
         j < images.Index[chain2 + 1].get_res_pos_e(0) - 1; ++j) {
      double distance{
          sqrt(square(m_mol[ind].pos.x[i] - images.m_mol[ind2].pos.x[j]) +
               square(m_mol[ind].pos.y[i] - images.m_mol[ind2].pos.y[j]) +
               square(m_mol[ind].pos.z[i] - images.m_mol[ind2].pos.z[j]))};
      if (distance < mindistance)
        mindistance = distance;
    }
  }
  return mindistance;
}

double Multi_MOL::find_min_dis(size_t ind, size_t ind2, size_t chain,
                               size_t chain2, Multi_MOL &images) const {
  double com[3]{}, com2[3]{};
  size_t atom_num{}, atom_num_chain2{};

  for (size_t i{Index[chain].get_res_pos_e(0) - 1};
       i < Index[chain + 1].get_res_pos_e(0) - 1; ++i) {
    com[0] += m_mol[ind].pos.x[i];
    com[1] += m_mol[ind].pos.y[i];
    com[2] += m_mol[ind].pos.z[i];
    ++atom_num;
  }
  com[0] /= atom_num;
  com[1] /= atom_num;
  com[2] /= atom_num;

  for (size_t i{images.Index[chain2].get_res_pos_e(0) - 1};
       i < images.Index[chain2 + 1].get_res_pos_e(0) - 1; ++i) {
    com2[0] += images.m_mol[ind2].pos.x[i];
    com2[1] += images.m_mol[ind2].pos.y[i];
    com2[2] += images.m_mol[ind2].pos.z[i];
    ++atom_num_chain2;
  }

  com2[0] /= atom_num_chain2;
  com2[1] /= atom_num_chain2;
  com2[2] /= atom_num_chain2;

  return sqrt(square(com[0] - com2[0]) + square(com[1] - com2[1]) +
              square(com[2] - com2[2]));
}

int MOL::create_pbc_images(arguments &Arguments, Multi_MOL &out_mol,
                           const vector<class index> &Index) {

  for (int i{}; i < 27; ++i) {
    double tran_vec[3]{}, tran_vec_2[3]{}, tran_vec_3[3]{};
    if (i == 0)
      Box.get_v1(tran_vec);
    else if (i == 1) {
      Box.get_v1(tran_vec);
      tran_vec[0] *= -1.;
    } else if (i == 2)
      Box.get_v2(tran_vec_2);
    else if (i == 3) {
      Box.get_v2(tran_vec_2);
      tran_vec_2[0] *= -1.;
      tran_vec_2[1] *= -1.;
    } else if (i == 4)
      Box.get_v3(tran_vec_3);
    else if (i == 5) {
      Box.get_v3(tran_vec_3);
      tran_vec_3[0] *= -1.;
      tran_vec_3[1] *= -1.;
      tran_vec_3[2] *= -1.;
    } else if (i == 6) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
    } else if (i == 7) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
      tran_vec_2[0] *= -1.;
      tran_vec_2[1] *= -1.;
    } else if (i == 8) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
      tran_vec[0] *= -1.;
      tran_vec[1] *= -1.;
    } else if (i == 9) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
      tran_vec[0] *= -1.;
      tran_vec[1] *= -1.;
      tran_vec_2[0] *= -1.;
      tran_vec_2[1] *= -1.;
    } else if (i == 10) {
      Box.get_v1(tran_vec);
      Box.get_v3(tran_vec_3);
    } else if (i == 11) {
      Box.get_v1(tran_vec);
      Box.get_v3(tran_vec_3);
      tran_vec_3[0] *= -1.;
      tran_vec_3[1] *= -1.;
      tran_vec_3[2] *= -1.;
    } else if (i == 12) {
      Box.get_v1(tran_vec);
      Box.get_v3(tran_vec_3);
      tran_vec[0] *= -1.;
      tran_vec[1] *= -1.;
      tran_vec[2] *= -1.;
    } else if (i == 13) {
      Box.get_v1(tran_vec);
      Box.get_v3(tran_vec_3);
      tran_vec[0] *= -1.;
      tran_vec[1] *= -1.;
      tran_vec[2] *= -1.;
      tran_vec_3[0] *= -1.;
      tran_vec_3[1] *= -1.;
      tran_vec_3[2] *= -1.;
    } else if (i == 14) {
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
    } else if (i == 15) {
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
      tran_vec_2[0] *= -1.;
      tran_vec_2[1] *= -1.;
      tran_vec_2[2] *= -1.;
    } else if (i == 16) {
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
      tran_vec_3[0] *= -1.;
      tran_vec_3[1] *= -1.;
      tran_vec_3[2] *= -1.;
    } else if (i == 17) {
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
      tran_vec_2[0] *= -1.;
      tran_vec_2[1] *= -1.;
      tran_vec_2[2] *= -1.;
      tran_vec_3[0] *= -1.;
      tran_vec_3[1] *= -1.;
      tran_vec_3[2] *= -1.;
    } else if (i == 18) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
    } else if (i == 19) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
      tran_vec[0] *= -1.;
      tran_vec[1] *= -1.;
      tran_vec[2] *= -1.;
    } else if (i == 20) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
      tran_vec_2[0] *= -1.;
      tran_vec_2[1] *= -1.;
      tran_vec_2[2] *= -1.;
    } else if (i == 21) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
      tran_vec_3[0] *= -1.;
      tran_vec_3[1] *= -1.;
      tran_vec_3[2] *= -1.;
    } else if (i == 22) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
      tran_vec[0] *= -1.;
      tran_vec[1] *= -1.;
      tran_vec[2] *= -1.;
      tran_vec_2[0] *= -1.;
      tran_vec_2[1] *= -1.;
      tran_vec_2[2] *= -1.;
    } else if (i == 23) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
      tran_vec[0] *= -1.;
      tran_vec[1] *= -1.;
      tran_vec[2] *= -1.;
      tran_vec_3[0] *= -1.;
      tran_vec_3[1] *= -1.;
      tran_vec_3[2] *= -1.;
    } else if (i == 24) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
      tran_vec_2[0] *= -1.;
      tran_vec_2[1] *= -1.;
      tran_vec_2[2] *= -1.;
      tran_vec_3[0] *= -1.;
      tran_vec_3[1] *= -1.;
      tran_vec_3[2] *= -1.;
    } else if (i == 25) {
      Box.get_v1(tran_vec);
      Box.get_v2(tran_vec_2);
      Box.get_v3(tran_vec_3);
      tran_vec[0] *= -1.;
      tran_vec[1] *= -1.;
      tran_vec[2] *= -1.;
      tran_vec_2[0] *= -1.;
      tran_vec_2[1] *= -1.;
      tran_vec_2[2] *= -1.;
      tran_vec_3[0] *= -1.;
      tran_vec_3[1] *= -1.;
      tran_vec_3[2] *= -1.;
    } else {
    }
    for (size_t j{}; j < this->size(); ++j) {
      out_mol.emplace(i, this->Box, this->get_time(), this->get_force_f(),
                      mol_info(), [&] {
                        MOL m;
                        // Assumption: v1=(a,0,0), v2=(bx,by,0), v3=(cx,cy,cz)
                        // — standard GROMACS triclinic form. v1[1]=v1[2]=0.
                        m.pos.x.emplace_back(this->pos.x[j] + tran_vec[0] +
                                             tran_vec_2[0] + tran_vec_3[0]);
                        m.pos.y.emplace_back(this->pos.y[j] + tran_vec_2[1] +
                                             tran_vec_3[1]);
                        m.pos.z.emplace_back(this->pos.z[j] + tran_vec_3[2]);
                        return m;
                      }());
    }
    out_mol.Index = Index;
  }
  return 0;
}
int MOL::find_best_image_of_residues(Multi_MOL &pbc_images, size_t chain,
                                     vector<class index> &Index,
                                     vector<size_t> &rejected_images) {
  for (size_t j{}; j < Index[chain].get_size_v(); ++j) {
    size_t limit = (j == Index[chain].get_size_v() - 1
                        ? Index[chain + 1].get_res_pos_e(0) - 1
                        : Index[chain].get_res_pos_e(j + 1) - 1);
    // if (limit == this->size()) return 0;

    for (size_t i = Index[chain].get_res_pos_e(j) - 1; i < limit - 1; ++i) {
      double distance_min{10000000.};
      size_t pos{};
      for (size_t k{}; k < pbc_images.m_mol.size(); ++k) {
        if (check_in_rejected_images(rejected_images, k) == true)
          continue;

        double d{this->dis_between_atom_and_point(
            i, pbc_images.m_mol[k].pos.x[i + 1],
            pbc_images.m_mol[k].pos.y[i + 1],
            pbc_images.m_mol[k].pos.z[i + 1])};
        if (d < distance_min) {
          distance_min = d;
          pos = k;
        }
      }
      this->pos.x[i + 1] = pbc_images.m_mol[pos].pos.x[i + 1];
      this->pos.y[i + 1] = pbc_images.m_mol[pos].pos.y[i + 1];
      this->pos.z[i + 1] = pbc_images.m_mol[pos].pos.z[i + 1];
    }
  }
  return 0;
}
int Multi_MOL::impl_solve_pbc_chain(size_t ind, arguments &Arguments,
                                    vector<size_t> &rejected_images) {
  (void)Arguments;
  (void)rejected_images;
  unwrap_mol_chains_in_scaled_space(this->m_mol[ind], Index,
                                    this->m_mol[ind].pos.x.size());
  return 0;
}
int Multi_MOL::find_best_image(size_t ind, int chain_ref,
                               arguments &Arguments) {
  vector<size_t> rejected_images;

  if (Arguments.pbc.solv_pbc_clust == true)
    init_rejected_images(rejected_images, Arguments);

  this->impl_solve_pbc_chain(ind, Arguments, rejected_images);

  if (Arguments.pbc.pbc_envi == false) {
    if (chain_ref >= static_cast<int>(Index.size() - 1))
      chain_ref = 0;
    return assemble_chains_with_envelope(this->m_mol[ind], Index, chain_ref,
                                         rejected_images);
  }

  Multi_MOL pbc_images;
  pbc_images.m_mol.reserve(27);
  this->m_mol[ind].create_pbc_images(Arguments, pbc_images, Index);

  if (chain_ref >= Index.size() - 1)
    chain_ref = 0;

  double centerofrefchain[3]{};
  if (Arguments.pbc.pbc_envi == true)
    this->m_mol[ind].find_cog_chain(chain_ref, Arguments, this->Index,
                                    centerofrefchain);

  for (size_t k{}; k < Index.size() - 1; ++k) {
    if (k == chain_ref)
      continue;

    if (Arguments.pbc.pbc_envi == true) {
      bool in_evi{false};
      for (size_t c{}; c < Arguments.pbc.pbc_envi_chain.size(); ++c)
        if (Index[k].get_chain() == Arguments.pbc.pbc_envi_chain[c]) {
          in_evi = true;
          break;
        }
      if (in_evi == true) {
        this->solv_pbc_envi(ind, Arguments, rejected_images, k, pbc_images,
                            centerofrefchain);
        continue;
      }
    }

    // FixBox (Baptista et al. 2022, Eq. 12): CoM pre-filter + atom-level
    // verification. CoM distance selects top-3 candidates (fast, O(N_chains)),
    // then atom-level min distance picks the true best (O(N²) only 3 times).
    constexpr size_t TOP_K = 3;
    struct Candidate {
      size_t img_idx;
      double com_dist;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(pbc_images.m_mol.size());

    for (size_t i{}; i < pbc_images.m_mol.size(); ++i) {
      if (Arguments.pbc.solv_pbc_clust == true &&
          check_in_rejected_images(rejected_images, i) == true)
        continue;
      double d_com = this->find_min_dis(ind, i, chain_ref, k, pbc_images);
      candidates.push_back({i, d_com});
    }

    // Sort by CoM distance, keep top-3
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) {
                return a.com_dist < b.com_dist;
              });
    if (candidates.size() > TOP_K)
      candidates.resize(TOP_K);

    // Atom-level verification on top-3 (FixBox Eq. 12)
    double distance_min{10000000.};
    size_t pos{};
    bool flag_min{false};

    for (const auto &cand : candidates) {
      double d =
          this->find_min_dis2(ind, cand.img_idx, chain_ref, k, pbc_images);
      if (d < distance_min) {
        distance_min = d;
        pos = cand.img_idx;
        flag_min = true;
      }
    }

    if (flag_min == true)
      for (size_t i{Index[k].get_res_pos_e(0) - 1};
           i < Index[k + 1].get_res_pos_e(0) - 1; ++i) {
        m_mol[ind].pos.x[i] = pbc_images.m_mol[pos].pos.x[i];
        m_mol[ind].pos.y[i] = pbc_images.m_mol[pos].pos.y[i];
        m_mol[ind].pos.z[i] = pbc_images.m_mol[pos].pos.z[i];
      }
  }
  return 0;
}

int Multi_MOL::find_best_image_envelope(size_t ind, int chain_ref,
                                        arguments &Arguments) {
  vector<size_t> rejected_images;

  if (Arguments.pbc.solv_pbc_clust == true)
    init_rejected_images(rejected_images, Arguments);

  this->impl_solve_pbc_chain(ind, Arguments, rejected_images);

  const size_t chain_count = Index.empty() ? 0 : Index.size() - 1;
  if (chain_count == 0)
    return 0;
  if (chain_ref < 0 || static_cast<size_t>(chain_ref) >= chain_count)
    chain_ref = 0;

  vector<char> envi_chain(chain_count, 0);
  for (size_t k = 0; k < chain_count; ++k) {
    if (Arguments.pbc.pbc_envi == true) {
      for (size_t c{}; c < Arguments.pbc.pbc_envi_chain.size(); ++c) {
        if (Index[k].get_chain() == Arguments.pbc.pbc_envi_chain[c]) {
          envi_chain[k] = 1;
          break;
        }
      }
    }
  }

  assemble_chains_with_hull(this->m_mol[ind], Index, chain_ref, rejected_images,
                            &envi_chain);

  if (Arguments.pbc.pbc_envi == true) {
    Multi_MOL pbc_images;
    pbc_images.m_mol.reserve(27);
    this->m_mol[ind].create_pbc_images(Arguments, pbc_images, Index);

    double centerofrefchain[3]{};
    this->m_mol[ind].find_cog_chain(static_cast<size_t>(chain_ref), Arguments,
                                    this->Index, centerofrefchain);

    for (size_t k = 0; k < chain_count; ++k) {
      if (!envi_chain[k])
        continue;
      this->solv_pbc_envi(ind, Arguments, rejected_images, k, pbc_images,
                          centerofrefchain);
    }
  }
  return 0;
}

int take_index_chain(vector<vector<force>> &center_of_geo,
                     size_t index_chain_ref,
                     vector<pair<size_t, size_t>> &list_ind) {
  for (size_t i{}; i < center_of_geo.size(); ++i)
    for (size_t j{}; j < center_of_geo[i].size(); ++j) {
      if (index_chain_ref != i)
        list_ind.emplace_back(i, j);
    }
  return 0;
}

int Multi_MOL::find_best_image_new(size_t ind, size_t chain_ref,
                                   arguments &Arguments,
                                   vector<vector<size_t>> &list_chain_combi) {
  (void)list_chain_combi;
  return this->find_best_image_envelope(ind, static_cast<int>(chain_ref),
                                        Arguments);
}

int Multi_MOL::find_best_image_of_mol2(Multi_MOL &mol2, size_t ind,
                                       int chain_ref, arguments &Arguments) {
  if (ind >= mol2.m_mol.size() || mol2.m_mol[ind].pos.x.empty())
    return 0;
  vector<size_t> rejected_images;
  if (Arguments.pbc.solv_pbc_clust == true)
    init_rejected_images(rejected_images, Arguments);

  if (Arguments.pbc.pbc_mem == false)
    mol2.find_best_image_envelope(ind, chain_ref, Arguments);

  if (Arguments.pbc.pbc_mol2_aggressive == 0)
    return place_molecule_with_envelope(this->m_mol[ind], Index,
                                        mol2.m_mol[ind], rejected_images);

  place_molecule_with_envelope(this->m_mol[ind], Index, mol2.m_mol[ind],
                               rejected_images);
  return place_molecule_with_envelope(this->m_mol[ind], Index, mol2.m_mol[ind],
                                      rejected_images);
}

int Multi_MOL::aggressive_find_best_image_of_mol2(Multi_MOL &mol2, size_t ind,
                                                  int chain_ref,
                                                  arguments &Arguments) {
  (void)chain_ref;
  if (ind >= mol2.m_mol.size() || mol2.m_mol[ind].pos.x.empty())
    return 0;
  // if (Arguments.pbc.pbc_mem == false)
  // mol2.find_best_image(ind, 0, Arguments);

  Multi_MOL pbc_images;
  if (Arguments.pbc.pbc_mol2_aggressive == 2) {
    pbc_images.m_mol.reserve(27);
    mol2.m_mol[ind].create_pbc_images(Arguments, pbc_images, Index);

    for (size_t j{}; j < mol2.m_mol[ind].pos.x.size(); ++j) {
      double distance_min{10e12};
      size_t pos{};
      for (size_t i{}; i < pbc_images.m_mol.size(); ++i) {
        double d{10e13};

        d = this->m_mol[ind].cal_minimum_distance_mol_atom(pbc_images.m_mol[i],
                                                           Arguments, j);

        if (d < distance_min) {
          distance_min = d;
          pos = i;
        }
      }

      mol2.m_mol[ind].pos.x[j] = pbc_images.m_mol[pos].pos.x[j];
      mol2.m_mol[ind].pos.y[j] = pbc_images.m_mol[pos].pos.y[j];
      mol2.m_mol[ind].pos.z[j] = pbc_images.m_mol[pos].pos.z[j];
    }

    double dmin{10e12};
    size_t pos_min{};

    for (size_t j{}; j < mol2.m_mol[ind].pos.x.size(); ++j) {
      double d{this->m_mol[ind].cal_minimum_distance_mol_atom(mol2.m_mol[ind],
                                                              Arguments, j)};
      if (d < dmin) {
        dmin = d;
        pos_min = j;
      }
    }

    pbc_images.m_mol.clear();
    mol2.m_mol[ind].create_pbc_images(Arguments, pbc_images, Index);

    for (size_t j{}; j < mol2.m_mol[ind].pos.x.size(); ++j) {
      double distance_min{10e12};
      size_t pos{};
      for (size_t i{}; i < pbc_images.m_mol.size(); ++i) {
        double d{10e13};

        d = sqrt(square(pbc_images.m_mol[i].pos.x[j] -
                        mol2.m_mol[ind].pos.x[pos_min]) +
                 square(pbc_images.m_mol[i].pos.y[j] -
                        mol2.m_mol[ind].pos.y[pos_min]) +
                 square(pbc_images.m_mol[i].pos.z[j] -
                        mol2.m_mol[ind].pos.z[pos_min]));

        if (d < distance_min) {
          distance_min = d;
          pos = i;
        }
      }

      mol2.m_mol[ind].pos.x[j] = pbc_images.m_mol[pos].pos.x[j];
      mol2.m_mol[ind].pos.y[j] = pbc_images.m_mol[pos].pos.y[j];
      mol2.m_mol[ind].pos.z[j] = pbc_images.m_mol[pos].pos.z[j];
    }

    double com_mol2[3]{};
    mol2.m_mol[ind].get_centroid(com_mol2);

    dmin = 10e12;
    pos_min = 0;
    for (size_t j{}; j < mol2.m_mol[ind].pos.x.size(); ++j) {
      double d{sqrt(square(mol2.m_mol[ind].pos.x[j] - com_mol2[0]) +
                    square(mol2.m_mol[ind].pos.y[j] - com_mol2[1]) +
                    square(mol2.m_mol[ind].pos.z[j] - com_mol2[2]))};
      if (d < dmin) {
        pos_min = j;
        dmin = d;
      }
    }

    pbc_images.m_mol.clear();
    mol2.m_mol[ind].create_pbc_images(Arguments, pbc_images, Index);

    for (size_t j{}; j < mol2.m_mol[ind].pos.x.size(); ++j) {
      double distance_min{10e12};
      size_t pos{};
      for (size_t i{}; i < pbc_images.m_mol.size(); ++i) {
        double d{10e13};

        d = sqrt(square(pbc_images.m_mol[i].pos.x[j] -
                        mol2.m_mol[ind].pos.x[pos_min]) +
                 square(pbc_images.m_mol[i].pos.y[j] -
                        mol2.m_mol[ind].pos.y[pos_min]) +
                 square(pbc_images.m_mol[i].pos.z[j] -
                        mol2.m_mol[ind].pos.z[pos_min]));

        if (d < distance_min) {
          distance_min = d;
          pos = i;
        }
      }

      mol2.m_mol[ind].pos.x[j] = pbc_images.m_mol[pos].pos.x[j];
      mol2.m_mol[ind].pos.y[j] = pbc_images.m_mol[pos].pos.y[j];
      mol2.m_mol[ind].pos.z[j] = pbc_images.m_mol[pos].pos.z[j];
    }
  }

  if (Arguments.pbc.pbc_mol2_aggressive == 3) {
    for (size_t j{}; j < mol2.m_mol[ind].pos.x.size(); ++j) {
      double v1[3]{}, v2[3]{}, v3[3]{};

      box Box_temp;
      Box_temp.set_box_type(this->m_mol[ind].Box.get_box_type());

      double mx = mol2.m_mol[ind].pos.x[j];
      double my = mol2.m_mol[ind].pos.y[j];
      double mz = mol2.m_mol[ind].pos.z[j];

      Box_temp.convert_to_v(mx, my, mz, v1, v2, v3);

      if (fabs(mz) > m_mol[ind].Box.get_c()) {
        mol2.m_mol[ind].pos.z[j] = mz - mz * m_mol[ind].Box.get_c() / fabs(mz);
      } else {
        if (my - v3[1] > this->m_mol[ind].Box.get_b()) {
          mol2.m_mol[ind].pos.y[j] = my - m_mol[ind].Box.get_b();
        }
        if (mx - v3[0] > this->m_mol[ind].Box.get_a()) {
          mol2.m_mol[ind].pos.x[j] = mx - m_mol[ind].Box.get_a();
        }
      }
    }
  }

  vector<size_t> rejected_images;
  if (Arguments.pbc.solv_pbc_clust == true)
    init_rejected_images(rejected_images, Arguments);

  return place_molecule_with_envelope(this->m_mol[ind], Index, mol2.m_mol[ind],
                                      rejected_images);
}

int Multi_MOL::impl_solv_pbc_new(Multi_MOL &mol2, size_t begin, size_t end,
                                 arguments &Arguments,
                                 const amino_acid &amine) {
  vector<vector<size_t>> data;
  if (Arguments.pbc.solv_pbc_int3 == 1) {
    gen_chain_list_cell(Index.size() - 1, data);
  }
  if (Arguments.flag_nocal == false) {
    for (size_t i{begin}; i < end; ++i) {
      int chain_ref{Arguments.pbc.chain_ref};
      if (chain_ref < 0)
        chain_ref = static_cast<int>(
            this->m_mol[i].find_chain_ref_pbc(Arguments, this->Index));

      if (Arguments.pbc.solv_pbc_int3 == 0)
        this->find_best_image_envelope(i, chain_ref, Arguments);
      else
        this->find_best_image_new(i, chain_ref, Arguments, data);

      if (Arguments.pbc.pbc_mol2_aggressive == 0)
        this->find_best_image_of_mol2(mol2, i, chain_ref, Arguments);
      else {
        this->aggressive_find_best_image_of_mol2(mol2, i, chain_ref, Arguments);
      }

      if (Arguments.pbc.pbc_aggressive == true) {
        if (Arguments.pbc.solv_pbc_int3 == 0)
          this->find_best_image_envelope(i, chain_ref, Arguments);
        else
          this->find_best_image_new(i, chain_ref, Arguments, data);
        this->aggressive_find_best_image_of_mol2(mol2, i, chain_ref, Arguments);
      }
    }
  } else {
    map<size_t, size_t> chains_connect;
    for (size_t i{begin}; i < end; ++i) {
      if (Arguments.pbc.flag_pbc_ref == true) {
        this->solv_pbc_ref(Arguments, amine, chains_connect, i, begin);
        continue;
      }
      int chain_ref{Arguments.pbc.chain_ref};
      if (chain_ref < 0)
        chain_ref = static_cast<int>(
            this->m_mol[i].find_chain_ref_pbc(Arguments, this->Index));

      if (Arguments.pbc.solv_pbc_int3 == 0)
        this->find_best_image_envelope(i, chain_ref, Arguments);
      else
        this->find_best_image_new(i, chain_ref, Arguments, data);
      if (Arguments.pbc.pbc_aggressive == true) {
        this->find_best_image_envelope(i, chain_ref, Arguments);
        this->find_best_image_envelope(i, chain_ref, Arguments);
      }
    }
  }
  return 0;
}
int Multi_MOL::solve_pbc_new(Multi_MOL &mol2, arguments &Arguments,
                             const amino_acid &amine) {
  vector<thread> impl_threads(Arguments.nt);
  for (size_t i{}; i < impl_threads.size(); ++i) {
    impl_threads[i] = thread(&Multi_MOL::impl_solv_pbc_new, &(*this), ref(mol2),
                             i * m_mol.size() / impl_threads.size(),
                             (i + 1) * m_mol.size() / impl_threads.size(),
                             ref(Arguments), ref(amine));
  }
  for (auto &t : impl_threads)
    t.join();
  return 0;
}

int Multi_MOL::solv_pbc_pbcn(Multi_MOL &mol2, arguments &Arguments) {
  {
    auto worker = [&](int start_frame, int end_frame) {
      if (start_frame >= end_frame ||
          start_frame >= static_cast<int>(m_mol.size())) {
        return;
      }

      const size_t N_rec = m_mol[start_frame].pos.x.size();
      const bool has_ligand =
          start_frame < static_cast<int>(mol2.m_mol.size()) &&
          !mol2.m_mol[start_frame].pos.x.empty();
      const size_t N_lig =
          has_ligand ? mol2.m_mol[start_frame].pos.x.size() : 0;
      vector<double> sx_rec(N_rec), sy_rec(N_rec), sz_rec(N_rec);
      vector<double> sx_lig(N_lig), sy_lig(N_lig), sz_lig(N_lig);

      for (int frame = start_frame; frame < end_frame; ++frame) {
        if (frame >= static_cast<int>(m_mol.size()))
          continue;

        MOL &rec = m_mol[frame];
        MOL *lig_ptr =
            frame < static_cast<int>(mol2.m_mol.size()) ? &mol2.m_mol[frame]
                                                        : nullptr;
        if (rec.pos.x.size() != N_rec)
          continue;
        if (has_ligand &&
            (lig_ptr == nullptr || lig_ptr->pos.x.size() != N_lig)) {
          continue;
        }

        double v1[3], v2[3], v3[3];
        if (!init_scaled_box_vectors(rec.Box, v1, v2, v3))
          continue;
        double box_center[3]{};
        init_fixbox_box_center(v1, v2, v3, box_center);

        for (size_t i = 0; i < N_rec; ++i) {
          cartesian_to_fixbox_scaled(v1, v2, v3, box_center, rec.pos.x[i],
                                     rec.pos.y[i], rec.pos.z[i], sx_rec[i],
                                     sy_rec[i], sz_rec[i]);
        }
        if (has_ligand) {
          MOL &lig = *lig_ptr;
          for (size_t i = 0; i < N_lig; ++i) {
            cartesian_to_fixbox_scaled(v1, v2, v3, box_center, lig.pos.x[i],
                                       lig.pos.y[i], lig.pos.z[i], sx_lig[i],
                                       sy_lig[i], sz_lig[i]);
          }
        }

        // Titan preprocessing only: make each chain whole before entering the
        // exact FixBox core below. Standalone FixBox assumes whole molecules
        // are already present in the input trajectory.
        unwrap_scaled_chain_ranges(sx_rec, sy_rec, sz_rec, Index, N_rec);
        if (has_ligand)
          unwrap_scaled_chain_ranges(sx_lig, sy_lig, sz_lig, mol2.Index, N_lig);

        apply_fixbox_pbcn_frame(sx_rec, sy_rec, sz_rec, sx_lig, sy_lig, sz_lig,
                                has_ligand, Index, mol2.Index,
                                Arguments.pbc.chain_ref, Arguments, v1, v2, v3);

        for (size_t i = 0; i < N_rec; ++i) {
          fixbox_scaled_to_cartesian(v1, v2, v3, box_center, sx_rec[i], sy_rec[i],
                                     sz_rec[i], rec.pos.x[i], rec.pos.y[i],
                                     rec.pos.z[i]);
        }
        if (has_ligand) {
          MOL &lig = *lig_ptr;
          for (size_t i = 0; i < N_lig; ++i) {
            fixbox_scaled_to_cartesian(v1, v2, v3, box_center, sx_lig[i], sy_lig[i],
                                       sz_lig[i], lig.pos.x[i], lig.pos.y[i],
                                       lig.pos.z[i]);
          }
        }
      }
    };

    const int num_threads =
        Arguments.nt > 0 ? static_cast<int>(Arguments.nt) : 1;
    vector<thread> threads(num_threads);
    const int chunk = static_cast<int>(m_mol.size()) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
      const int start = t * chunk;
      const int end = (t == num_threads - 1) ? static_cast<int>(m_mol.size())
                                             : (t + 1) * chunk;
      threads[t] = thread(worker, start, end);
    }
    for (auto &th : threads) {
      if (th.joinable())
        th.join();
    }
    return 0;
  }

  // FixBox Algorithm (Baptista et al., J. Chem. Inf. Model. 2022, 62, 4435)
  // Replaces the former COM-heuristic with proper scaled-space unwrapping
  // and minimum-image assembly.

  auto worker = [&](int start_frame, int end_frame) {
    if (start_frame >= end_frame ||
        start_frame >= static_cast<int>(m_mol.size()))
      return;

    // Step 0: Allocate once per thread — N_rec/N_lig are constant across MD
    const size_t N_rec = m_mol[start_frame].pos.x.size();
    const bool has_ligand =
        start_frame < static_cast<int>(mol2.m_mol.size()) &&
        !mol2.m_mol[start_frame].pos.x.empty();
    const size_t N_lig = has_ligand ? mol2.m_mol[start_frame].pos.x.size() : 0;
    std::vector<double> sx_rec(N_rec), sy_rec(N_rec), sz_rec(N_rec);
    std::vector<double> sx_lig(N_lig), sy_lig(N_lig), sz_lig(N_lig);

    for (int frame = start_frame; frame < end_frame; ++frame) {
      if (frame >= static_cast<int>(m_mol.size()))
        continue;

      MOL &rec = m_mol[frame];
      MOL *lig_ptr =
          frame < static_cast<int>(mol2.m_mol.size()) ? &mol2.m_mol[frame]
                                                      : nullptr;

      // ISSUE 3: skip corrupt frames with mismatched atom count
      if (rec.pos.x.size() != N_rec)
        continue;
      if (has_ligand) {
        if (lig_ptr == nullptr || lig_ptr->pos.x.size() != N_lig)
          continue;
      }

      // ── Step 1: Build B⁻¹ and convert to scaled coordinates ──
      double v1[3], v2[3], v3[3];
      rec.Box.get_v1(v1);
      rec.Box.get_v2(v2);
      rec.Box.get_v3(v3);

      // Analytical B⁻¹ (back-substitution, upper triangular B):
      //   B = [[v1[0], v2[0], v3[0]],
      //        [    0, v2[1], v3[1]],
      //        [    0,     0, v3[2]]]
      auto cart_to_scaled = [&](double rx, double ry, double rz,
                                double &sx, double &sy, double &sz) {
        sz = rz / v3[2];
        sy = (ry - v3[1] * sz) / v2[1];
        sx = (rx - v2[0] * sy - v3[0] * sz) / v1[0];
      };

      auto scaled_to_cart = [&](double sx, double sy, double sz,
                                double &rx, double &ry, double &rz) {
        rx = v1[0] * sx + v2[0] * sy + v3[0] * sz;
        ry = v2[1] * sy + v3[1] * sz;
        rz = v3[2] * sz;
      };

      for (size_t i = 0; i < N_rec; ++i)
        cart_to_scaled(rec.pos.x[i], rec.pos.y[i], rec.pos.z[i], sx_rec[i],
                       sy_rec[i], sz_rec[i]);
      if (has_ligand) {
        MOL &lig = *lig_ptr;
        for (size_t i = 0; i < N_lig; ++i)
          cart_to_scaled(lig.pos.x[i], lig.pos.y[i], lig.pos.z[i], sx_lig[i],
                         sy_lig[i], sz_lig[i]);
      }

      // ── Step 2: Unwrap molecules in scaled space (FixBox Eq. 13–17) ──
      unwrap_scaled_chain_ranges(sx_rec, sy_rec, sz_rec, Index, N_rec);
      if (has_ligand)
        unwrap_scaled_chain_ranges(sx_lig, sy_lig, sz_lig, mol2.Index, N_lig);

      // ── Step 3: Assemble ligand chains onto the receptor using a FixBox-style tree ──
      {
        constexpr size_t kExactRefineTopK = 4;

        struct ChainRange {
          size_t begin;
          size_t end;
          bool ligand;
        };

        struct ChainScaledGeom {
          double mean_sx = 0.0;
          double mean_sy = 0.0;
          double mean_sz = 0.0;
          double env_center_sx = 0.0;
          double env_center_sy = 0.0;
          double env_center_sz = 0.0;
        };

        auto build_chain_ranges = [](const vector<class index> &chains,
                                     size_t atom_count) {
          vector<ChainRange> ranges;
          ranges.reserve(chains.size());
          for (size_t k = 0; k + 1 < chains.size(); ++k) {
            size_t begin = chains[k].get_res_pos_e(0) - 1;
            size_t end = chains[k + 1].get_res_pos_e(0) - 1;
            if (begin >= atom_count || end > atom_count || begin >= end)
              continue;
            ranges.push_back({begin, end, false});
          }
          if (ranges.empty() && atom_count > 0)
            ranges.push_back({0, atom_count, false});
          return ranges;
        };

        vector<ChainRange> rec_ranges = build_chain_ranges(Index, N_rec);
        vector<ChainRange> lig_ranges;
        if (has_ligand && N_lig > 0)
          lig_ranges = build_chain_ranges(mol2.Index, N_lig);
        for (ChainRange &range : lig_ranges)
          range.ligand = true;

        vector<ChainRange> chains;
        chains.reserve(rec_ranges.size() + lig_ranges.size());
        chains.insert(chains.end(), rec_ranges.begin(), rec_ranges.end());
        chains.insert(chains.end(), lig_ranges.begin(), lig_ranges.end());

        if (!chains.empty()) {
          const size_t rec_chain_count = rec_ranges.size();
          vector<long> shift_x(chains.size(), 0);
          vector<long> shift_y(chains.size(), 0);
          vector<long> shift_z(chains.size(), 0);
          vector<char> assembled(chains.size(), 0);

          auto select_sx = [&](const ChainRange &range) -> const vector<double> & {
            return range.ligand ? sx_lig : sx_rec;
          };
          auto select_sy = [&](const ChainRange &range) -> const vector<double> & {
            return range.ligand ? sy_lig : sy_rec;
          };
          auto select_sz = [&](const ChainRange &range) -> const vector<double> & {
            return range.ligand ? sz_lig : sz_rec;
          };

          auto shift_to_cart = [&](long sx_shift, long sy_shift, long sz_shift,
                                   double &tx, double &ty, double &tz) {
            tx = v1[0] * static_cast<double>(sx_shift) +
                 v2[0] * static_cast<double>(sy_shift) +
                 v3[0] * static_cast<double>(sz_shift);
            ty = v2[1] * static_cast<double>(sy_shift) +
                 v3[1] * static_cast<double>(sz_shift);
            tz = v3[2] * static_cast<double>(sz_shift);
          };

          auto pack_shift = [](long sx_shift, long sy_shift,
                               long sz_shift) -> uint64_t {
            const uint64_t ux =
                static_cast<uint64_t>(sx_shift + kEnvelopeVoxelBias);
            const uint64_t uy =
                static_cast<uint64_t>(sy_shift + kEnvelopeVoxelBias);
            const uint64_t uz =
                static_cast<uint64_t>(sz_shift + kEnvelopeVoxelBias);
            return (ux << 42) | (uy << 21) | uz;
          };

          vector<ChainScaledGeom> scaled_geom(chains.size());
          vector<EnvelopeChainGeom> cart_geom(chains.size());
          size_t total_atoms = 0;
          size_t total_surface_cells = 0;
          for (size_t idx = 0; idx < chains.size(); ++idx) {
            const ChainRange &range = chains[idx];
            const vector<double> &sx = select_sx(range);
            const vector<double> &sy = select_sy(range);
            const vector<double> &sz = select_sz(range);
            double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
            double min_x = std::numeric_limits<double>::infinity();
            double min_y = std::numeric_limits<double>::infinity();
            double min_z = std::numeric_limits<double>::infinity();
            double max_x = -std::numeric_limits<double>::infinity();
            double max_y = -std::numeric_limits<double>::infinity();
            double max_z = -std::numeric_limits<double>::infinity();
            vector<array<double, 3>> atom_cart;
            atom_cart.reserve(range.end - range.begin);
            for (size_t i = range.begin; i < range.end; ++i) {
              sum_x += sx[i];
              sum_y += sy[i];
              sum_z += sz[i];
              min_x = std::min(min_x, sx[i]);
              min_y = std::min(min_y, sy[i]);
              min_z = std::min(min_z, sz[i]);
              max_x = std::max(max_x, sx[i]);
              max_y = std::max(max_y, sy[i]);
              max_z = std::max(max_z, sz[i]);
              double rx = 0.0, ry = 0.0, rz = 0.0;
              scaled_to_cart(sx[i], sy[i], sz[i], rx, ry, rz);
              atom_cart.push_back({rx, ry, rz});
            }
            const double inv_n =
                1.0 / static_cast<double>(std::max<size_t>(1, range.end - range.begin));
            scaled_geom[idx].mean_sx = sum_x * inv_n;
            scaled_geom[idx].mean_sy = sum_y * inv_n;
            scaled_geom[idx].mean_sz = sum_z * inv_n;
            scaled_geom[idx].env_center_sx = 0.5 * (min_x + max_x);
            scaled_geom[idx].env_center_sy = 0.5 * (min_y + max_y);
            scaled_geom[idx].env_center_sz = 0.5 * (min_z + max_z);

            cart_geom[idx] = build_envelope_chain_geom(std::move(atom_cart));
            total_atoms += cart_geom[idx].atom_count;
            total_surface_cells += cart_geom[idx].surface_centers.size();
          }

          auto assemble_stage = [&](size_t begin_idx, size_t end_idx,
                                    bool allow_all_assembled_anchors,
                                    size_t preferred_seed) {
            if (begin_idx >= end_idx)
              return;

            bool has_eligible_anchor = false;
            for (size_t idx = 0; idx < chains.size(); ++idx) {
              if (!assembled[idx])
                continue;
              if (!allow_all_assembled_anchors &&
                  (idx < begin_idx || idx >= end_idx)) {
                continue;
              }
              has_eligible_anchor = true;
              break;
            }

            if (!has_eligible_anchor) {
              size_t seed = preferred_seed;
              if (seed < begin_idx || seed >= end_idx)
                seed = begin_idx;
              assembled[seed] = 1;
            }

            size_t remaining = 0;
            for (size_t idx = begin_idx; idx < end_idx; ++idx)
              remaining += assembled[idx] ? 0u : 1u;

            while (remaining > 0) {
              EnvelopeScene scene;
              reserve_envelope_scene(scene, total_atoms, total_surface_cells);
              for (size_t idx = 0; idx < chains.size(); ++idx) {
                if (!assembled[idx])
                  continue;
                double tx = 0.0, ty = 0.0, tz = 0.0;
                shift_to_cart(shift_x[idx], shift_y[idx], shift_z[idx], tx, ty,
                              tz);
                add_chain_to_envelope_scene(scene, cart_geom[idx], tx, ty, tz);
              }
              if (scene.atom_total == 0)
                break;

              EnvelopePlacementScore best_score;
              for (size_t moving = begin_idx; moving < end_idx; ++moving) {
                if (assembled[moving])
                  continue;

                for (size_t anchor = 0; anchor < chains.size(); ++anchor) {
                  if (!assembled[anchor])
                    continue;
                  if (!allow_all_assembled_anchors &&
                      (anchor < begin_idx || anchor >= end_idx)) {
                    continue;
                  }

                  vector<EnvelopePlacementScore> pair_top_scores;
                  unordered_set<uint64_t> seen_shifts;
                  seen_shifts.reserve(2048);
                  size_t candidate_order = 0;

                  auto try_candidate = [&](long sx_shift, long sy_shift,
                                           long sz_shift) {
                    if (!seen_shifts
                             .insert(pack_shift(sx_shift, sy_shift, sz_shift))
                             .second) {
                      return;
                    }
                    double tx = 0.0, ty = 0.0, tz = 0.0;
                    shift_to_cart(sx_shift, sy_shift, sz_shift, tx, ty, tz);
                    EnvelopePlacementScore candidate =
                        score_chain_envelope_placement(
                            cart_geom[moving], scene, tx, ty, tz, moving,
                            anchor, candidate_order,
                            pair_top_scores.empty() ? nullptr
                                                    : &pair_top_scores.front());
                    candidate.sx = sx_shift;
                    candidate.sy = sy_shift;
                    candidate.sz = sz_shift;
                    candidate.image_index = candidate_order++;
                    insert_top_envelope_candidate(pair_top_scores, candidate,
                                                  kExactRefineTopK);
                  };

                  try_candidate(shift_x[moving], shift_y[moving],
                                shift_z[moving]);

                  const double base_sx[3] = {
                      (scaled_geom[anchor].env_center_sx +
                       static_cast<double>(shift_x[anchor])) -
                          scaled_geom[moving].env_center_sx,
                      (scaled_geom[anchor].mean_sx +
                       static_cast<double>(shift_x[anchor])) -
                          scaled_geom[moving].mean_sx,
                      static_cast<double>(shift_x[anchor])};
                  const double base_sy[3] = {
                      (scaled_geom[anchor].env_center_sy +
                       static_cast<double>(shift_y[anchor])) -
                          scaled_geom[moving].env_center_sy,
                      (scaled_geom[anchor].mean_sy +
                       static_cast<double>(shift_y[anchor])) -
                          scaled_geom[moving].mean_sy,
                      static_cast<double>(shift_y[anchor])};
                  const double base_sz[3] = {
                      (scaled_geom[anchor].env_center_sz +
                       static_cast<double>(shift_z[anchor])) -
                          scaled_geom[moving].env_center_sz,
                      (scaled_geom[anchor].mean_sz +
                       static_cast<double>(shift_z[anchor])) -
                          scaled_geom[moving].mean_sz,
                      static_cast<double>(shift_z[anchor])};

                  for (int base_id = 0; base_id < 3; ++base_id) {
                    const long base_kx = std::lround(base_sx[base_id]);
                    const long base_ky = std::lround(base_sy[base_id]);
                    const long base_kz = std::lround(base_sz[base_id]);
                    for (int radius = 0; radius <= 3; ++radius) {
                      for (int dx = -radius; dx <= radius; ++dx) {
                        for (int dy = -radius; dy <= radius; ++dy) {
                          for (int dz = -radius; dz <= radius; ++dz) {
                            try_candidate(base_kx + static_cast<long>(dx),
                                          base_ky + static_cast<long>(dy),
                                          base_kz + static_cast<long>(dz));
                          }
                        }
                      }
                      if (!pair_top_scores.empty() &&
                          is_clean_envelope_placement(pair_top_scores.front())) {
                        break;
                      }
                    }
                    if (!pair_top_scores.empty() &&
                        is_clean_envelope_placement(pair_top_scores.front())) {
                      break;
                    }
                  }

                  const EnvelopePlacementScore pair_best =
                      finalize_envelope_candidate_selection(cart_geom[moving],
                                                            scene,
                                                            pair_top_scores);
                  if (better_envelope_placement(pair_best, best_score))
                    best_score = pair_best;
                }
              }

              if (!best_score.valid || best_score.moving >= chains.size()) {
                for (size_t moving = begin_idx; moving < end_idx; ++moving) {
                  if (!assembled[moving]) {
                    assembled[moving] = 1;
                    --remaining;
                    break;
                  }
                }
                continue;
              }

              shift_x[best_score.moving] = best_score.sx;
              shift_y[best_score.moving] = best_score.sy;
              shift_z[best_score.moving] = best_score.sz;
              assembled[best_score.moving] = 1;
              --remaining;
            }
          };

          if (rec_chain_count > 0)
            assemble_stage(0, rec_chain_count, false, 0);
          if (chains.size() > rec_chain_count)
            assemble_stage(rec_chain_count, chains.size(), true,
                           rec_chain_count);
          if (rec_chain_count == 0 && !chains.empty())
            assemble_stage(0, chains.size(), false, 0);

          for (size_t idx = 0; idx < chains.size(); ++idx) {
            if (!assembled[idx])
              continue;
            const ChainRange &range = chains[idx];
            vector<double> &sx = range.ligand ? sx_lig : sx_rec;
            vector<double> &sy = range.ligand ? sy_lig : sy_rec;
            vector<double> &sz = range.ligand ? sz_lig : sz_rec;
            for (size_t i = range.begin; i < range.end; ++i) {
              sx[i] += static_cast<double>(shift_x[idx]);
              sy[i] += static_cast<double>(shift_y[idx]);
              sz[i] += static_cast<double>(shift_z[idx]);
            }
          }
        }
      }

      if (false && has_ligand && N_lig > 0) {
        struct ChainRange {
          size_t begin;
          size_t end;
          bool ligand;
        };

        struct ChainGeom {
          double mean_sx;
          double mean_sy;
          double mean_sz;
          double env_center_sx;
          double env_center_sy;
          double env_center_sz;
        };

        struct PlacementScore {
          size_t moving = static_cast<size_t>(-1);
          size_t anchor = static_cast<size_t>(-1);
          long sx = 0;
          long sy = 0;
          long sz = 0;
          size_t hard_clash = std::numeric_limits<size_t>::max();
          size_t soft_clash = std::numeric_limits<size_t>::max();
          size_t overlap_cells = std::numeric_limits<size_t>::max();
          size_t contact_atoms = 0;
          size_t contact_cells = 0;
          double center_d2 = std::numeric_limits<double>::infinity();
          bool valid = false;
        };

        auto cart_dist_sq_from_scaled = [&](double dsx, double dsy,
                                            double dsz) -> double {
          double rx = v1[0] * dsx + v2[0] * dsy + v3[0] * dsz;
          double ry = v2[1] * dsy + v3[1] * dsz;
          double rz = v3[2] * dsz;
          return rx * rx + ry * ry + rz * rz;
        };

        auto build_chain_ranges = [](const vector<class index> &chains,
                                     size_t atom_count) {
          vector<ChainRange> ranges;
          ranges.reserve(chains.size());
          for (size_t k = 0; k + 1 < chains.size(); ++k) {
            size_t begin = chains[k].get_res_pos_e(0) - 1;
            size_t end = chains[k + 1].get_res_pos_e(0) - 1;
            if (begin >= atom_count || end > atom_count || begin >= end)
              continue;
            ranges.push_back({begin, end, false});
          }
          if (ranges.empty() && atom_count > 0)
            ranges.push_back({0, atom_count, false});
          return ranges;
        };

        vector<ChainRange> rec_ranges = build_chain_ranges(Index, N_rec);
        vector<ChainRange> lig_ranges = build_chain_ranges(mol2.Index, N_lig);
        for (ChainRange &range : lig_ranges)
          range.ligand = true;

        vector<ChainRange> chains;
        chains.reserve(rec_ranges.size() + lig_ranges.size());
        chains.insert(chains.end(), rec_ranges.begin(), rec_ranges.end());
        chains.insert(chains.end(), lig_ranges.begin(), lig_ranges.end());
        if (chains.empty())
          return;

        const size_t rec_chain_count = rec_ranges.size();
        vector<long> shift_x(chains.size(), 0);
        vector<long> shift_y(chains.size(), 0);
        vector<long> shift_z(chains.size(), 0);
        vector<char> assembled(chains.size(), 0);

        auto select_sx = [&](const ChainRange &range) -> const vector<double> & {
          return range.ligand ? sx_lig : sx_rec;
        };
        auto select_sy = [&](const ChainRange &range) -> const vector<double> & {
          return range.ligand ? sy_lig : sy_rec;
        };
        auto select_sz = [&](const ChainRange &range) -> const vector<double> & {
          return range.ligand ? sz_lig : sz_rec;
        };

        vector<ChainGeom> geom(chains.size());
        for (size_t idx = 0; idx < chains.size(); ++idx) {
          const ChainRange &range = chains[idx];
          const vector<double> &sx = select_sx(range);
          const vector<double> &sy = select_sy(range);
          const vector<double> &sz = select_sz(range);
          double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
          double min_x = std::numeric_limits<double>::infinity();
          double min_y = std::numeric_limits<double>::infinity();
          double min_z = std::numeric_limits<double>::infinity();
          double max_x = -std::numeric_limits<double>::infinity();
          double max_y = -std::numeric_limits<double>::infinity();
          double max_z = -std::numeric_limits<double>::infinity();
          for (size_t i = range.begin; i < range.end; ++i) {
            sum_x += sx[i];
            sum_y += sy[i];
            sum_z += sz[i];
            min_x = std::min(min_x, sx[i]);
            min_y = std::min(min_y, sy[i]);
            min_z = std::min(min_z, sz[i]);
            max_x = std::max(max_x, sx[i]);
            max_y = std::max(max_y, sy[i]);
            max_z = std::max(max_z, sz[i]);
          }
          const double inv_n =
              1.0 / static_cast<double>(std::max<size_t>(1, range.end - range.begin));
          geom[idx].mean_sx = sum_x * inv_n;
          geom[idx].mean_sy = sum_y * inv_n;
          geom[idx].mean_sz = sum_z * inv_n;
          geom[idx].env_center_sx = 0.5 * (min_x + max_x);
          geom[idx].env_center_sy = 0.5 * (min_y + max_y);
          geom[idx].env_center_sz = 0.5 * (min_z + max_z);
        }

        constexpr int voxel_bias = 1 << 20;
        const double envelope_grid = 2.5;
        const double clash_grid = 2.5;
        const double hard_clash_cutoff = 1.10;
        const double soft_clash_cutoff = 1.35;
        const double contact_cutoff = 4.50;
        const double contact_floor = 1.50;
        const double hard_clash_cutoff2 = hard_clash_cutoff * hard_clash_cutoff;
        const double soft_clash_cutoff2 = soft_clash_cutoff * soft_clash_cutoff;
        const double contact_cutoff2 = contact_cutoff * contact_cutoff;
        const double contact_floor2 = contact_floor * contact_floor;
        static const int face_offsets[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                               {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        vector<array<int, 3>> shell_offsets;
        shell_offsets.reserve(124);
        for (int dx = -2; dx <= 2; ++dx)
          for (int dy = -2; dy <= 2; ++dy)
            for (int dz = -2; dz <= 2; ++dz)
              if (dx != 0 || dy != 0 || dz != 0)
                shell_offsets.push_back({dx, dy, dz});

        auto pack_cell = [&](int ix, int iy, int iz) -> uint64_t {
          const uint64_t ux = static_cast<uint64_t>(ix + voxel_bias);
          const uint64_t uy = static_cast<uint64_t>(iy + voxel_bias);
          const uint64_t uz = static_cast<uint64_t>(iz + voxel_bias);
          return (ux << 42) | (uy << 21) | uz;
        };

        auto unpack_cell = [&](uint64_t key, int &ix, int &iy, int &iz) {
          constexpr uint64_t mask = (1ULL << 21) - 1ULL;
          ix = static_cast<int>((key >> 42) & mask) - voxel_bias;
          iy = static_cast<int>((key >> 21) & mask) - voxel_bias;
          iz = static_cast<int>(key & mask) - voxel_bias;
        };

        auto grid_index = [](double value, double spacing) -> int {
          return static_cast<int>(std::floor(value / spacing));
        };

        auto shift_to_cart = [&](long sx_shift, long sy_shift, long sz_shift,
                                 double &tx, double &ty, double &tz) {
          tx = v1[0] * static_cast<double>(sx_shift) +
               v2[0] * static_cast<double>(sy_shift) +
               v3[0] * static_cast<double>(sz_shift);
          ty = v2[1] * static_cast<double>(sy_shift) +
               v3[1] * static_cast<double>(sz_shift);
          tz = v3[2] * static_cast<double>(sz_shift);
        };

        vector<vector<array<double, 3>>> chain_core_centers(chains.size());
        vector<vector<array<double, 3>>> chain_surface_centers(chains.size());
        for (size_t idx = 0; idx < chains.size(); ++idx) {
          const ChainRange &range = chains[idx];
          const vector<double> &sx = select_sx(range);
          const vector<double> &sy = select_sy(range);
          const vector<double> &sz = select_sz(range);
          unordered_set<uint64_t> local_core;
          local_core.reserve((range.end - range.begin) * 2);
          for (size_t i = range.begin; i < range.end; ++i) {
            double rx = 0.0, ry = 0.0, rz = 0.0;
            scaled_to_cart(sx[i], sy[i], sz[i], rx, ry, rz);
            local_core.insert(pack_cell(grid_index(rx, envelope_grid),
                                        grid_index(ry, envelope_grid),
                                        grid_index(rz, envelope_grid)));
          }

          chain_core_centers[idx].reserve(local_core.size());
          chain_surface_centers[idx].reserve(local_core.size());
          for (uint64_t key : local_core) {
            int ix = 0, iy = 0, iz = 0;
            unpack_cell(key, ix, iy, iz);
            const array<double, 3> center = {
                (static_cast<double>(ix) + 0.5) * envelope_grid,
                (static_cast<double>(iy) + 0.5) * envelope_grid,
                (static_cast<double>(iz) + 0.5) * envelope_grid};
            chain_core_centers[idx].push_back(center);

            bool surface = false;
            for (const auto &off : face_offsets) {
              if (local_core.find(pack_cell(ix + off[0], iy + off[1],
                                            iz + off[2])) == local_core.end()) {
                surface = true;
                break;
              }
            }
            if (surface)
              chain_surface_centers[idx].push_back(center);
          }
        }

        auto better_placement = [&](const PlacementScore &lhs,
                                    const PlacementScore &rhs) -> bool {
          if (!lhs.valid)
            return false;
          if (!rhs.valid)
            return true;
          if (lhs.hard_clash != rhs.hard_clash)
            return lhs.hard_clash < rhs.hard_clash;
          if (lhs.soft_clash != rhs.soft_clash)
            return lhs.soft_clash < rhs.soft_clash;
          if (lhs.contact_atoms != rhs.contact_atoms)
            return lhs.contact_atoms > rhs.contact_atoms;
          if (lhs.contact_cells != rhs.contact_cells)
            return lhs.contact_cells > rhs.contact_cells;
          if (lhs.overlap_cells != rhs.overlap_cells)
            return lhs.overlap_cells < rhs.overlap_cells;
          if (lhs.center_d2 != rhs.center_d2)
            return lhs.center_d2 < rhs.center_d2;
          return lhs.anchor < rhs.anchor;
        };

        auto is_clean_placement = [&](const PlacementScore &score) -> bool {
          return score.valid && score.hard_clash == 0 && score.soft_clash == 0 &&
                 (score.contact_atoms > 0 || score.contact_cells > 0);
        };

        auto assemble_stage = [&](size_t begin_idx, size_t end_idx,
                                  bool allow_all_assembled_anchors) {
          if (begin_idx >= end_idx)
            return;

          auto has_eligible_anchor = [&]() {
            for (size_t i = 0; i < chains.size(); ++i) {
              if (!assembled[i])
                continue;
              if (!allow_all_assembled_anchors &&
                  (i < begin_idx || i >= end_idx))
                continue;
              return true;
            }
            return false;
          };

          if (!has_eligible_anchor())
            assembled[begin_idx] = 1;

          size_t remaining = 0;
          for (size_t i = begin_idx; i < end_idx; ++i)
            remaining += assembled[i] ? 0u : 1u;

          while (remaining > 0) {
            unordered_set<uint64_t> assembled_core;
            unordered_map<uint64_t, vector<array<double, 3>>> assembled_atoms;
            size_t assembled_atom_total = 0;
            double assembled_center_sx = 0.0;
            double assembled_center_sy = 0.0;
            double assembled_center_sz = 0.0;

            for (size_t idx = 0; idx < chains.size(); ++idx) {
              if (!assembled[idx])
                continue;
              const ChainRange &range = chains[idx];
              const vector<double> &sx = select_sx(range);
              const vector<double> &sy = select_sy(range);
              const vector<double> &sz = select_sz(range);
              const size_t atom_count = range.end - range.begin;
              assembled_atom_total += atom_count;
              assembled_center_sx +=
                  static_cast<double>(atom_count) *
                  (geom[idx].env_center_sx + static_cast<double>(shift_x[idx]));
              assembled_center_sy +=
                  static_cast<double>(atom_count) *
                  (geom[idx].env_center_sy + static_cast<double>(shift_y[idx]));
              assembled_center_sz +=
                  static_cast<double>(atom_count) *
                  (geom[idx].env_center_sz + static_cast<double>(shift_z[idx]));

              for (size_t i = range.begin; i < range.end; ++i) {
                double rx = 0.0, ry = 0.0, rz = 0.0;
                scaled_to_cart(sx[i] + static_cast<double>(shift_x[idx]),
                               sy[i] + static_cast<double>(shift_y[idx]),
                               sz[i] + static_cast<double>(shift_z[idx]), rx, ry,
                               rz);
                assembled_core.insert(
                    pack_cell(grid_index(rx, envelope_grid),
                              grid_index(ry, envelope_grid),
                              grid_index(rz, envelope_grid)));
                assembled_atoms[pack_cell(grid_index(rx, clash_grid),
                                          grid_index(ry, clash_grid),
                                          grid_index(rz, clash_grid))]
                    .push_back({rx, ry, rz});
              }
            }

            if (assembled_atom_total == 0)
              break;

            const double inv_assembled =
                1.0 / static_cast<double>(assembled_atom_total);
            assembled_center_sx *= inv_assembled;
            assembled_center_sy *= inv_assembled;
            assembled_center_sz *= inv_assembled;

            unordered_set<uint64_t> assembled_shell;
            assembled_shell.reserve(assembled_core.size() * 6);
            for (uint64_t key : assembled_core) {
              int ix = 0, iy = 0, iz = 0;
              unpack_cell(key, ix, iy, iz);
              bool surface = false;
              for (const auto &off : face_offsets) {
                if (assembled_core.find(
                        pack_cell(ix + off[0], iy + off[1], iz + off[2])) ==
                    assembled_core.end()) {
                  surface = true;
                  break;
                }
              }
              if (!surface)
                continue;
              for (const auto &off : shell_offsets) {
                const uint64_t shell_key =
                    pack_cell(ix + off[0], iy + off[1], iz + off[2]);
                if (assembled_core.find(shell_key) == assembled_core.end())
                  assembled_shell.insert(shell_key);
              }
            }

            unordered_map<uint64_t, vector<array<double, 3>>>
                assembled_surface_points;
            for (size_t idx = 0; idx < chains.size(); ++idx) {
              if (!assembled[idx])
                continue;
              double tx = 0.0, ty = 0.0, tz = 0.0;
              shift_to_cart(shift_x[idx], shift_y[idx], shift_z[idx], tx, ty, tz);
              for (const auto &center : chain_surface_centers[idx]) {
                const double px = center[0] + tx;
                const double py = center[1] + ty;
                const double pz = center[2] + tz;
                assembled_surface_points[pack_cell(
                    grid_index(px, contact_cutoff), grid_index(py, contact_cutoff),
                    grid_index(pz, contact_cutoff))]
                    .push_back({px, py, pz});
              }
            }

            PlacementScore best_score;

            for (size_t m = begin_idx; m < end_idx; ++m) {
              if (assembled[m])
                continue;
              for (size_t n = 0; n < chains.size(); ++n) {
                if (!assembled[n])
                  continue;
                if (!allow_all_assembled_anchors &&
                    (n < begin_idx || n >= end_idx))
                  continue;

                PlacementScore pair_best;
                const double base_sx[3] = {
                    (geom[n].env_center_sx + static_cast<double>(shift_x[n])) -
                        geom[m].env_center_sx,
                    (geom[n].mean_sx + static_cast<double>(shift_x[n])) -
                        geom[m].mean_sx,
                    static_cast<double>(shift_x[n])};
                const double base_sy[3] = {
                    (geom[n].env_center_sy + static_cast<double>(shift_y[n])) -
                        geom[m].env_center_sy,
                    (geom[n].mean_sy + static_cast<double>(shift_y[n])) -
                        geom[m].mean_sy,
                    static_cast<double>(shift_y[n])};
                const double base_sz[3] = {
                    (geom[n].env_center_sz + static_cast<double>(shift_z[n])) -
                        geom[m].env_center_sz,
                    (geom[n].mean_sz + static_cast<double>(shift_z[n])) -
                        geom[m].mean_sz,
                    static_cast<double>(shift_z[n])};

                for (int base_id = 0; base_id < 3; ++base_id) {
                  const long base_kx = std::lround(base_sx[base_id]);
                  const long base_ky = std::lround(base_sy[base_id]);
                  const long base_kz = std::lround(base_sz[base_id]);
                  for (int search_radius = 1; search_radius <= 3; ++search_radius) {
                    for (int dx = -search_radius; dx <= search_radius; ++dx) {
                      for (int dy = -search_radius; dy <= search_radius; ++dy) {
                        for (int dz = -search_radius; dz <= search_radius; ++dz) {
                        PlacementScore candidate;
                        candidate.moving = m;
                        candidate.anchor = n;
                        candidate.sx = base_kx + static_cast<long>(dx);
                        candidate.sy = base_ky + static_cast<long>(dy);
                        candidate.sz = base_kz + static_cast<long>(dz);
                        candidate.valid = true;

                        const ChainRange &range = chains[m];
                        const vector<double> &sx = select_sx(range);
                        const vector<double> &sy = select_sy(range);
                        const vector<double> &sz = select_sz(range);
                        unordered_set<uint64_t> moving_core;
                        moving_core.reserve((range.end - range.begin) * 2);

                        size_t hard_clash = 0;
                        size_t soft_clash = 0;
                        size_t contact_atoms = 0;
                        for (size_t i = range.begin; i < range.end; ++i) {
                          double rx = 0.0, ry = 0.0, rz = 0.0;
                          scaled_to_cart(
                              sx[i] + static_cast<double>(candidate.sx),
                              sy[i] + static_cast<double>(candidate.sy),
                              sz[i] + static_cast<double>(candidate.sz), rx, ry,
                              rz);

                          moving_core.insert(
                              pack_cell(grid_index(rx, envelope_grid),
                                        grid_index(ry, envelope_grid),
                                        grid_index(rz, envelope_grid)));

                          const int hx = grid_index(rx, clash_grid);
                          const int hy = grid_index(ry, clash_grid);
                          const int hz = grid_index(rz, clash_grid);
                          bool atom_has_contact = false;
                          for (int ox = -1; ox <= 1; ++ox) {
                            for (int oy = -1; oy <= 1; ++oy) {
                              for (int oz = -1; oz <= 1; ++oz) {
                                const auto it = assembled_atoms.find(
                                    pack_cell(hx + ox, hy + oy, hz + oz));
                                if (it == assembled_atoms.end())
                                  continue;
                                for (const auto &other : it->second) {
                                  const double drx = rx - other[0];
                                  const double dry = ry - other[1];
                                  const double drz = rz - other[2];
                                  const double d2 =
                                      drx * drx + dry * dry + drz * drz;
                                  if (d2 < hard_clash_cutoff2)
                                    ++hard_clash;
                                  else if (d2 < soft_clash_cutoff2)
                                    ++soft_clash;
                                  else if (d2 < contact_cutoff2)
                                    atom_has_contact = true;
                                }
                              }
                            }
                          }
                          if (atom_has_contact)
                            ++contact_atoms;
                        }

                        double tx = 0.0, ty = 0.0, tz = 0.0;
                        shift_to_cart(candidate.sx, candidate.sy, candidate.sz,
                                      tx, ty, tz);

                        size_t overlap_cells = 0;
                        for (uint64_t key : moving_core) {
                          if (assembled_core.find(key) != assembled_core.end())
                            ++overlap_cells;
                        }

                        size_t contact_cells = 0;
                        for (const auto &center : chain_surface_centers[m]) {
                          const double px = center[0] + tx;
                          const double py = center[1] + ty;
                          const double pz = center[2] + tz;
                          const int hx = grid_index(px, contact_cutoff);
                          const int hy = grid_index(py, contact_cutoff);
                          const int hz = grid_index(pz, contact_cutoff);
                          bool has_surface_contact = false;
                          for (int ox = -1; ox <= 1 && !has_surface_contact; ++ox) {
                            for (int oy = -1; oy <= 1 && !has_surface_contact;
                                 ++oy) {
                              for (int oz = -1; oz <= 1 && !has_surface_contact;
                                   ++oz) {
                                const auto it = assembled_surface_points.find(
                                    pack_cell(hx + ox, hy + oy, hz + oz));
                                if (it == assembled_surface_points.end())
                                  continue;
                                for (const auto &other : it->second) {
                                  const double drx = px - other[0];
                                  const double dry = py - other[1];
                                  const double drz = pz - other[2];
                                  const double d2 =
                                      drx * drx + dry * dry + drz * drz;
                                  if (d2 >= contact_floor2 &&
                                      d2 <= contact_cutoff2) {
                                    has_surface_contact = true;
                                    break;
                                  }
                                }
                              }
                            }
                          }
                          if (has_surface_contact)
                            ++contact_cells;
                        }

                        candidate.hard_clash = hard_clash;
                        candidate.soft_clash = soft_clash;
                        candidate.overlap_cells = overlap_cells;
                        candidate.contact_atoms = contact_atoms;
                        candidate.contact_cells = contact_cells;
                        candidate.center_d2 = cart_dist_sq_from_scaled(
                            (geom[m].env_center_sx +
                             static_cast<double>(candidate.sx)) -
                                assembled_center_sx,
                            (geom[m].env_center_sy +
                             static_cast<double>(candidate.sy)) -
                                assembled_center_sy,
                            (geom[m].env_center_sz +
                             static_cast<double>(candidate.sz)) -
                                assembled_center_sz);

                        if (better_placement(candidate, pair_best))
                          pair_best = candidate;
                        }
                      }
                    }
                    if (is_clean_placement(pair_best))
                      break;
                  }
                  if (is_clean_placement(pair_best))
                    break;
                }

                if (better_placement(pair_best, best_score))
                  best_score = pair_best;
              }
            }

            if (!best_score.valid || best_score.moving >= chains.size()) {
              for (size_t m = begin_idx; m < end_idx; ++m) {
                if (!assembled[m]) {
                  assembled[m] = 1;
                  --remaining;
                  break;
                }
              }
              continue;
            }

            shift_x[best_score.moving] = best_score.sx;
            shift_y[best_score.moving] = best_score.sy;
            shift_z[best_score.moving] = best_score.sz;
            assembled[best_score.moving] = 1;
            --remaining;
          }
        };

        if (rec_chain_count > 0) {
          assemble_stage(0, rec_chain_count, false);
          assemble_stage(rec_chain_count, chains.size(), true);
        } else {
          assemble_stage(0, chains.size(), false);
        }

        for (size_t idx = 0; idx < chains.size(); ++idx) {
          if (!assembled[idx])
            continue;
          const ChainRange &range = chains[idx];
          vector<double> &sx = range.ligand ? sx_lig : sx_rec;
          vector<double> &sy = range.ligand ? sy_lig : sy_rec;
          vector<double> &sz = range.ligand ? sz_lig : sz_rec;
          for (size_t i = range.begin; i < range.end; ++i) {
            sx[i] += static_cast<double>(shift_x[idx]);
            sy[i] += static_cast<double>(shift_y[idx]);
            sz[i] += static_cast<double>(shift_z[idx]);
          }
        }
      }

      // ── Step 4: Center Receptor bounding box (FixBox Step 3) ──
      if (!Arguments.pbc.no_center) {
        double sxmin = *std::min_element(sx_rec.begin(), sx_rec.begin() + N_rec);
        double sxmax = *std::max_element(sx_rec.begin(), sx_rec.begin() + N_rec);
        double symin = *std::min_element(sy_rec.begin(), sy_rec.begin() + N_rec);
        double symax = *std::max_element(sy_rec.begin(), sy_rec.begin() + N_rec);
        double szmin = *std::min_element(sz_rec.begin(), sz_rec.begin() + N_rec);
        double szmax = *std::max_element(sz_rec.begin(), sz_rec.begin() + N_rec);

        // Shift to center bounding box at (0.5, 0.5, 0.5)
        double csx = 0.5 - 0.5 * (sxmin + sxmax);
        double csy = 0.5 - 0.5 * (symin + symax);
        double csz = 0.5 - 0.5 * (szmin + szmax);

        for (size_t i = 0; i < N_rec; ++i) {
          sx_rec[i] += csx;
          sy_rec[i] += csy;
          sz_rec[i] += csz;
        }
        if (has_ligand) {
          for (size_t i = 0; i < N_lig; ++i) {
            sx_lig[i] += csx;
            sy_lig[i] += csy;
            sz_lig[i] += csz;
          }
        }
      }

      // ── Step 5: Convert back to Cartesian ──
      for (size_t i = 0; i < N_rec; ++i)
        scaled_to_cart(sx_rec[i], sy_rec[i], sz_rec[i], rec.pos.x[i],
                       rec.pos.y[i], rec.pos.z[i]);
      if (has_ligand) {
        MOL &lig = *lig_ptr;
        for (size_t i = 0; i < N_lig; ++i)
          scaled_to_cart(sx_lig[i], sy_lig[i], sz_lig[i], lig.pos.x[i],
                         lig.pos.y[i], lig.pos.z[i]);
      }
    }
  };

  // Threading — identical structure to original
  int num_threads = Arguments.nt > 0 ? static_cast<int>(Arguments.nt) : 1;
  std::vector<std::thread> threads(num_threads);
  int chunk = static_cast<int>(m_mol.size()) / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    int start = t * chunk;
    int end = (t == num_threads - 1) ? static_cast<int>(m_mol.size())
                                     : (t + 1) * chunk;
    threads[t] = std::thread(worker, start, end);
  }

  for (auto &th : threads) {
    if (th.joinable())
      th.join();
  }

  return 0;
}
