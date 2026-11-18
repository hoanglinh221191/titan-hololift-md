#include "tabi_calderon_aux_surface.h"
#include "tabi_calderon_trace_space.h"
#include "tabi_types.h"
#include "tabi_near.h"
#include "tabi_sphere_clip.h"

#include "../titan_surface.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace TitanPBE::Tabi {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kVertexQuantize = 1.0e8;

struct UnitVertex {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct UnitTriangle {
  std::size_t a{0};
  std::size_t b{0};
  std::size_t c{0};
};

struct UnitSphereMesh {
  std::vector<UnitVertex> vertices;
  std::vector<UnitTriangle> triangles;
};

struct AtomCellGrid {
  double min_x{0.0};
  double min_y{0.0};
  double min_z{0.0};
  double cell{1.0};
  std::size_t nx{1};
  std::size_t ny{1};
  std::size_t nz{1};
  std::size_t cell_count{1};
};

struct QuantizedKey {
  std::int64_t x{0};
  std::int64_t y{0};
  std::int64_t z{0};

  bool operator==(const QuantizedKey &other) const noexcept {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct QuantizedKeyHash {
  std::size_t operator()(const QuantizedKey &key) const noexcept {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](std::int64_t value) {
      h ^= static_cast<std::uint64_t>(value) + 0x9e3779b97f4a7c15ull +
           (h << 6u) + (h >> 2u);
      h *= 1099511628211ull;
    };
    mix(key.x);
    mix(key.y);
    mix(key.z);
    return static_cast<std::size_t>(h);
  }
};

struct EdgeKey {
  std::size_t a{0};
  std::size_t b{0};

  bool operator==(const EdgeKey &other) const noexcept {
    return a == other.a && b == other.b;
  }
};

struct EdgeKeyHash {
  std::size_t operator()(const EdgeKey &edge) const noexcept {
    std::uint64_t h = 1469598103934665603ull;
    h ^= static_cast<std::uint64_t>(edge.a) + 0x9e3779b97f4a7c15ull +
         (h << 6u) + (h >> 2u);
    h *= 1099511628211ull;
    h ^= static_cast<std::uint64_t>(edge.b) + 0x9e3779b97f4a7c15ull +
         (h << 6u) + (h >> 2u);
    h *= 1099511628211ull;
    return static_cast<std::size_t>(h);
  }
};

bool finite_atom(const Atom &atom) {
  return std::isfinite(atom.x) && std::isfinite(atom.y) &&
         std::isfinite(atom.z) && std::isfinite(atom.radius) &&
         atom.radius > 0.0;
}

double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) noexcept {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

double norm3(double x, double y, double z) noexcept {
  return std::sqrt(std::max(0.0, x * x + y * y + z * z));
}

double spherical_triangle_area_unit(const UnitVertex &a, const UnitVertex &b,
                                    const UnitVertex &c) noexcept {
  const double bx_cx = b.y * c.z - b.z * c.y;
  const double by_cx = b.z * c.x - b.x * c.z;
  const double bz_cx = b.x * c.y - b.y * c.x;
  const double det = a.x * bx_cx + a.y * by_cx + a.z * bz_cx;
  const double ab = a.x * b.x + a.y * b.y + a.z * b.z;
  const double bc = b.x * c.x + b.y * c.y + b.z * c.z;
  const double ca = c.x * a.x + c.y * a.y + c.z * a.z;
  const double denom = 1.0 + ab + bc + ca;
  return std::abs(2.0 * std::atan2(det, denom));
}

QuantizedKey quantized_key(double x, double y, double z) noexcept {
  return {static_cast<std::int64_t>(std::llround(x * kVertexQuantize)),
          static_cast<std::int64_t>(std::llround(y * kVertexQuantize)),
          static_cast<std::int64_t>(std::llround(z * kVertexQuantize))};
}

int mesh_rings_for_point_count(int point_count) noexcept {
  // Legacy latlong policy: 64 rings / 16128 triangles per atom at the cap.
  // Requests >= 3969 are saturated even with highres_surface; use icosphere
  // for further refinement. Keep explicit latlong reproducible.
  const int safe_points = std::max(8, point_count);
  const int rings =
      static_cast<int>(std::lround(std::sqrt(static_cast<double>(safe_points) *
                                             1.0))) +
      1;
  return std::clamp(rings, 4, 64);
}

UnitVertex normalized_vertex(double x, double y, double z) noexcept {
  const double len = norm3(x, y, z);
  if (len <= 1.0e-14)
    return {0.0, 0.0, 1.0};
  return {x / len, y / len, z / len};
}

UnitSphereMesh make_latlong_unit_sphere_mesh(int point_count) {
  const int rings = mesh_rings_for_point_count(point_count);
  const int segments = std::max(8, rings * 2);
  UnitSphereMesh mesh;
  mesh.vertices.reserve(2 + static_cast<std::size_t>(rings - 1) *
                                static_cast<std::size_t>(segments));
  mesh.vertices.push_back({0.0, 0.0, 1.0});
  for (int ring = 1; ring < rings; ++ring) {
    const double theta = kPi * static_cast<double>(ring) /
                         static_cast<double>(rings);
    const double z = std::cos(theta);
    const double r = std::sin(theta);
    for (int seg = 0; seg < segments; ++seg) {
      const double phi = kTwoPi * static_cast<double>(seg) /
                         static_cast<double>(segments);
      mesh.vertices.push_back({r * std::cos(phi), r * std::sin(phi), z});
    }
  }
  const std::size_t south = mesh.vertices.size();
  mesh.vertices.push_back({0.0, 0.0, -1.0});

  auto ring_index = [&](int ring, int seg) -> std::size_t {
    const int wrapped = (seg % segments + segments) % segments;
    return 1 + static_cast<std::size_t>(ring - 1) *
                   static_cast<std::size_t>(segments) +
           static_cast<std::size_t>(wrapped);
  };

  for (int seg = 0; seg < segments; ++seg)
    mesh.triangles.push_back({0, ring_index(1, seg),
                              ring_index(1, seg + 1)});

  for (int ring = 1; ring < rings - 1; ++ring) {
    for (int seg = 0; seg < segments; ++seg) {
      const std::size_t a = ring_index(ring, seg);
      const std::size_t b = ring_index(ring, seg + 1);
      const std::size_t c = ring_index(ring + 1, seg);
      const std::size_t d = ring_index(ring + 1, seg + 1);
      mesh.triangles.push_back({a, c, b});
      mesh.triangles.push_back({b, c, d});
    }
  }

  for (int seg = 0; seg < segments; ++seg)
    mesh.triangles.push_back({ring_index(rings - 1, seg), south,
                              ring_index(rings - 1, seg + 1)});
  return mesh;
}

UnitSphereMesh make_icosphere_unit_sphere_mesh(int point_count) {
  UnitSphereMesh mesh;
  const double phi = (1.0 + std::sqrt(5.0)) * 0.5;
  const std::array<UnitVertex, 12> base_vertices = {
      normalized_vertex(-1.0, phi, 0.0), normalized_vertex(1.0, phi, 0.0),
      normalized_vertex(-1.0, -phi, 0.0), normalized_vertex(1.0, -phi, 0.0),
      normalized_vertex(0.0, -1.0, phi), normalized_vertex(0.0, 1.0, phi),
      normalized_vertex(0.0, -1.0, -phi), normalized_vertex(0.0, 1.0, -phi),
      normalized_vertex(phi, 0.0, -1.0), normalized_vertex(phi, 0.0, 1.0),
      normalized_vertex(-phi, 0.0, -1.0), normalized_vertex(-phi, 0.0, 1.0)};
  mesh.vertices.assign(base_vertices.begin(), base_vertices.end());
  const std::array<UnitTriangle, 20> base_triangles = {
      UnitTriangle{0, 11, 5}, UnitTriangle{0, 5, 1},
      UnitTriangle{0, 1, 7},  UnitTriangle{0, 7, 10},
      UnitTriangle{0, 10, 11}, UnitTriangle{1, 5, 9},
      UnitTriangle{5, 11, 4}, UnitTriangle{11, 10, 2},
      UnitTriangle{10, 7, 6}, UnitTriangle{7, 1, 8},
      UnitTriangle{3, 9, 4},  UnitTriangle{3, 4, 2},
      UnitTriangle{3, 2, 6},  UnitTriangle{3, 6, 8},
      UnitTriangle{3, 8, 9},  UnitTriangle{4, 9, 5},
      UnitTriangle{2, 4, 11}, UnitTriangle{6, 2, 10},
      UnitTriangle{8, 6, 7},  UnitTriangle{9, 8, 1}};
  mesh.triangles.assign(base_triangles.begin(), base_triangles.end());

  int subdivisions = 0;
  std::size_t target_triangles = 20;
  const std::size_t safe_target =
      static_cast<std::size_t>(std::max(8, point_count));
  while (target_triangles < safe_target && subdivisions < 6) {
    target_triangles *= 4;
    ++subdivisions;
  }

  auto midpoint = [&](std::size_t a, std::size_t b,
                      std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash>
                          &cache) -> std::size_t {
    EdgeKey key{std::min(a, b), std::max(a, b)};
    const auto found = cache.find(key);
    if (found != cache.end())
      return found->second;
    const UnitVertex &va = mesh.vertices[a];
    const UnitVertex &vb = mesh.vertices[b];
    const UnitVertex mid =
        normalized_vertex(0.5 * (va.x + vb.x), 0.5 * (va.y + vb.y),
                          0.5 * (va.z + vb.z));
    const std::size_t index = mesh.vertices.size();
    mesh.vertices.push_back(mid);
    cache.emplace(key, index);
    return index;
  };

  for (int level = 0; level < subdivisions; ++level) {
    std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash> midpoint_cache;
    midpoint_cache.reserve(mesh.triangles.size() * 3);
    std::vector<UnitTriangle> refined;
    refined.reserve(mesh.triangles.size() * 4);
    for (const UnitTriangle &tri : mesh.triangles) {
      const std::size_t ab = midpoint(tri.a, tri.b, midpoint_cache);
      const std::size_t bc = midpoint(tri.b, tri.c, midpoint_cache);
      const std::size_t ca = midpoint(tri.c, tri.a, midpoint_cache);
      refined.push_back({tri.a, ab, ca});
      refined.push_back({tri.b, bc, ab});
      refined.push_back({tri.c, ca, bc});
      refined.push_back({ab, bc, ca});
    }
    mesh.triangles = std::move(refined);
  }
  return mesh;
}

std::string resolve_sphere_mesh_mode(const std::string &requested) {
  if (requested == "icosphere")
    return "icosphere";
  return "latlong";
}

const UnitSphereMesh &cached_unit_sphere_mesh(int point_count,
                                             const std::string &mesh_mode) {
  const std::string mode = resolve_sphere_mesh_mode(mesh_mode);
  static const UnitSphereMesh default_mesh =
      make_latlong_unit_sphere_mesh(TitanSurface::kDefaultPointCount);
  if (mode == "latlong" && point_count == TitanSurface::kDefaultPointCount)
    return default_mesh;
  static thread_local UnitSphereMesh local_latlong_mesh;
  static thread_local UnitSphereMesh local_icosphere_mesh;
  static thread_local int local_latlong_points = 0;
  static thread_local int local_icosphere_points = 0;
  if (mode == "icosphere") {
    if (local_icosphere_points != point_count) {
      local_icosphere_mesh = make_icosphere_unit_sphere_mesh(point_count);
      local_icosphere_points = point_count;
    }
    return local_icosphere_mesh;
  }
  if (local_latlong_points != point_count) {
    local_latlong_mesh = make_latlong_unit_sphere_mesh(point_count);
    local_latlong_points = point_count;
  }
  return local_latlong_mesh;
}

AtomCellGrid make_atom_cell_grid(const std::vector<Atom> &atoms,
                                 const std::vector<double> &surface_radii) {
  AtomCellGrid grid;
  if (atoms.empty()) {
    return grid;
  }

  double max_radius = 0.0;
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  grid.min_x = std::numeric_limits<double>::infinity();
  grid.min_y = std::numeric_limits<double>::infinity();
  grid.min_z = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    max_radius = std::max(max_radius, surface_radii[i]);
    grid.min_x = std::min(grid.min_x, atoms[i].x);
    grid.min_y = std::min(grid.min_y, atoms[i].y);
    grid.min_z = std::min(grid.min_z, atoms[i].z);
    max_x = std::max(max_x, atoms[i].x);
    max_y = std::max(max_y, atoms[i].y);
    max_z = std::max(max_z, atoms[i].z);
  }

  grid.cell = std::max(1.0e-6, 2.0 * max_radius);
  grid.nx = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::floor((max_x - grid.min_x) / grid.cell)) +
             1);
  grid.ny = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::floor((max_y - grid.min_y) / grid.cell)) +
             1);
  grid.nz = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::floor((max_z - grid.min_z) / grid.cell)) +
             1);
  grid.cell_count = grid.nx * grid.ny * grid.nz;
  return grid;
}

std::size_t atom_cell_index(const AtomCellGrid &grid, double x, double y,
                            double z) noexcept {
  const std::size_t ix = std::min<std::size_t>(
      grid.nx - 1, static_cast<std::size_t>(
                       std::max(0.0, std::floor((x - grid.min_x) / grid.cell))));
  const std::size_t iy = std::min<std::size_t>(
      grid.ny - 1, static_cast<std::size_t>(
                       std::max(0.0, std::floor((y - grid.min_y) / grid.cell))));
  const std::size_t iz = std::min<std::size_t>(
      grid.nz - 1, static_cast<std::size_t>(
                       std::max(0.0, std::floor((z - grid.min_z) / grid.cell))));
  return (iz * grid.ny + iy) * grid.nx + ix;
}

void atom_cell_coords(const AtomCellGrid &grid, std::size_t cell,
                      std::size_t &ix, std::size_t &iy,
                      std::size_t &iz) noexcept {
  ix = cell % grid.nx;
  const std::size_t yz = cell / grid.nx;
  iy = yz % grid.ny;
  iz = yz / grid.ny;
}

std::vector<std::vector<std::size_t>>
build_atom_neighbor_lists(const std::vector<Atom> &atoms,
                          const std::vector<double> &surface_radii,
                          SurfaceMesh &mesh) {
  const auto begin = std::chrono::steady_clock::now();
  std::vector<std::vector<std::size_t>> neighbors(atoms.size());
  if (atoms.empty()) {
    mesh.neighbor_build_ms = elapsed_ms(begin, std::chrono::steady_clock::now());
    return neighbors;
  }

  const AtomCellGrid grid = make_atom_cell_grid(atoms, surface_radii);
  mesh.cell_count = grid.cell_count;
  std::vector<std::size_t> cell_offsets(grid.cell_count + 1, 0);
  std::vector<std::size_t> atom_cell(atoms.size(), 0);
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    atom_cell[i] = atom_cell_index(grid, atoms[i].x, atoms[i].y, atoms[i].z);
    ++cell_offsets[atom_cell[i] + 1];
  }
  for (std::size_t c = 0; c < grid.cell_count; ++c)
    cell_offsets[c + 1] += cell_offsets[c];

  std::vector<std::size_t> fill = cell_offsets;
  std::vector<std::size_t> cell_atoms(atoms.size(), 0);
  for (std::size_t i = 0; i < atoms.size(); ++i)
    cell_atoms[fill[atom_cell[i]]++] = i;

  std::size_t neighbor_sum = 0;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    std::size_t ix = 0;
    std::size_t iy = 0;
    std::size_t iz = 0;
    atom_cell_coords(grid, atom_cell[i], ix, iy, iz);
    const std::size_t x0 = ix == 0 ? 0 : ix - 1;
    const std::size_t y0 = iy == 0 ? 0 : iy - 1;
    const std::size_t z0 = iz == 0 ? 0 : iz - 1;
    const std::size_t x1 = std::min(grid.nx - 1, ix + 1);
    const std::size_t y1 = std::min(grid.ny - 1, iy + 1);
    const std::size_t z1 = std::min(grid.nz - 1, iz + 1);
    auto &list = neighbors[i];
    for (std::size_t zz = z0; zz <= z1; ++zz) {
      for (std::size_t yy = y0; yy <= y1; ++yy) {
        for (std::size_t xx = x0; xx <= x1; ++xx) {
          const std::size_t cell = (zz * grid.ny + yy) * grid.nx + xx;
          for (std::size_t pos = cell_offsets[cell];
               pos < cell_offsets[cell + 1]; ++pos) {
            const std::size_t j = cell_atoms[pos];
            if (i == j)
              continue;
            const double cutoff = surface_radii[i] + surface_radii[j];
            const double dx = atoms[i].x - atoms[j].x;
            const double dy = atoms[i].y - atoms[j].y;
            const double dz = atoms[i].z - atoms[j].z;
            if (dx * dx + dy * dy + dz * dz < cutoff * cutoff)
              list.push_back(j);
          }
        }
      }
    }
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
    neighbor_sum += list.size();
    mesh.max_neighbor_count = std::max(mesh.max_neighbor_count, list.size());
  }

  mesh.mean_neighbor_count =
      atoms.empty() ? 0.0
                    : static_cast<double>(neighbor_sum) /
                          static_cast<double>(atoms.size());
  mesh.neighbor_build_ms = elapsed_ms(begin, std::chrono::steady_clock::now());
  return neighbors;
}

bool point_buried(double x, double y, double z,
                  const std::vector<Atom> &atoms,
                  const std::vector<double> &surface_radii,
                  const std::vector<std::size_t> &neighbors) noexcept {
  for (const std::size_t j : neighbors) {
    const Atom &other = atoms[j];
    const double dx = x - other.x;
    const double dy = y - other.y;
    const double dz = z - other.z;
    const double r = surface_radii[j];
    if (dx * dx + dy * dy + dz * dz < r * r)
      return true;
  }
  return false;
}

double triangle_solid_angle(const Vertex &a, const Vertex &b, const Vertex &c,
                            double px, double py, double pz) noexcept {
  const double ax = a.x - px;
  const double ay = a.y - py;
  const double az = a.z - pz;
  const double bx = b.x - px;
  const double by = b.y - py;
  const double bz = b.z - pz;
  const double cx = c.x - px;
  const double cy = c.y - py;
  const double cz = c.z - pz;
  const double la = norm3(ax, ay, az);
  const double lb = norm3(bx, by, bz);
  const double lc = norm3(cx, cy, cz);
  if (la <= 1.0e-14 || lb <= 1.0e-14 || lc <= 1.0e-14)
    return 0.0;
  const double cross_x = by * cz - bz * cy;
  const double cross_y = bz * cx - bx * cz;
  const double cross_z = bx * cy - by * cx;
  const double numerator = ax * cross_x + ay * cross_y + az * cross_z;
  const double ab = ax * bx + ay * by + az * bz;
  const double bc = bx * cx + by * cy + bz * cz;
  const double ca = cx * ax + cy * ay + cz * az;
  const double denominator = la * lb * lc + ab * lc + bc * la + ca * lb;
  return 2.0 * std::atan2(numerator, denominator);
}

std::size_t add_vertex(SurfaceMesh &mesh,
                       std::unordered_multimap<QuantizedKey, std::size_t,
                                               QuantizedKeyHash> &lookup,
                       const Atom &atom, std::size_t atom_index,
                       const UnitVertex &unit, double radius) {
  const double x = atom.x + unit.x * radius;
  const double y = atom.y + unit.y * radius;
  const double z = atom.z + unit.z * radius;
  const QuantizedKey key = quantized_key(x, y, z);
  const auto range = lookup.equal_range(key);
  for (auto found = range.first; found != range.second; ++found) {
    const auto &v = mesh.vertices[found->second];
    // A hash bucket is not geometric identity. Clipped seam vertices can be
    // less than 1e-8 A apart while belonging to different exposed patches.
    if (norm3(v.x - x, v.y - y, v.z - z) <= 1e-12)
      return found->second;
  }
  const std::size_t index = mesh.vertices.size();
  mesh.vertices.push_back({x, y, z, unit.x, unit.y, unit.z, atom_index});
  lookup.emplace(key, index);
  return index;
}

void add_edge(std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash> &edges,
              std::size_t a, std::size_t b) {
  if (a > b)
    std::swap(a, b);
  ++edges[{a, b}];
}

void finalize_solid_angle_diagonal(SurfaceMesh &mesh,
                                   const std::string &diagonal_mode) {
  const auto begin = std::chrono::steady_clock::now();
  if (mesh.panels.empty()) {
    mesh.diag_min = 1.0;
    mesh.diag_max = 1.0;
    mesh.solid_angle_ms = elapsed_ms(begin, std::chrono::steady_clock::now());
    return;
  }

  std::vector<std::vector<std::size_t>> triangles_by_vertex(
      mesh.vertices.size());
  std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash> edge_counts;
  edge_counts.reserve(mesh.triangles.size() * 3);
  for (std::size_t t = 0; t < mesh.triangles.size(); ++t) {
    const Triangle &tri = mesh.triangles[t];
    triangles_by_vertex[tri.v0].push_back(t);
    triangles_by_vertex[tri.v1].push_back(t);
    triangles_by_vertex[tri.v2].push_back(t);
    add_edge(edge_counts, tri.v0, tri.v1);
    add_edge(edge_counts, tri.v1, tri.v2);
    add_edge(edge_counts, tri.v2, tri.v0);
  }

  std::size_t valence_sum = 0;
  std::size_t active_vertices = 0;
  for (const auto &incident : triangles_by_vertex) {
    if (!incident.empty()) {
      valence_sum += incident.size();
      ++active_vertices;
    }
  }
  mesh.avg_valence = active_vertices > 0
                         ? static_cast<double>(valence_sum) /
                               static_cast<double>(active_vertices)
                         : 0.0;
  mesh.nonmanifold_edges = 0;
  for (const auto &[edge, count] : edge_counts) {
    (void)edge;
    if (count > 2)
      ++mesh.nonmanifold_edges;
  }

  mesh.diag_min = std::numeric_limits<double>::infinity();
  mesh.diag_max = -std::numeric_limits<double>::infinity();
  std::vector<std::vector<std::size_t>> patches(mesh.triangles.size());
  if (diagonal_mode == "solid-angle") {
    std::vector<std::size_t> patch;
    for (std::size_t t = 0; t < mesh.triangles.size(); ++t) {
      const Triangle &tri = mesh.triangles[t];
      patch.clear();
      patch.push_back(t);
      for (const std::size_t vertex : {tri.v0, tri.v1, tri.v2}) {
        const auto &incident = triangles_by_vertex[vertex];
        patch.insert(patch.end(), incident.begin(), incident.end());
      }
      std::sort(patch.begin(), patch.end());
      patch.erase(std::unique(patch.begin(), patch.end()), patch.end());
      patches[t] = patch;
    }
  }
  for (std::size_t t = 0; t < mesh.triangles.size(); ++t) {
    Panel &panel = mesh.panels[t];
    double diagonal = 1.0;
    double solid_angle = kTwoPi;
    if (diagonal_mode == "off") {
      diagonal = 0.0;
      solid_angle = 0.0;
    } else if (diagonal_mode == "solid-angle") {
      const double offset = std::max(1.0e-7, std::sqrt(panel.area) * 1.0e-5);
      const double px = panel.x + panel.nx * offset;
      const double py = panel.y + panel.ny * offset;
      const double pz = panel.z + panel.nz * offset;
      double omega = 0.0;
      for (const std::size_t patch_tri_index : patches[t]) {
        const Triangle &patch_tri = mesh.triangles[patch_tri_index];
        omega += triangle_solid_angle(mesh.vertices[patch_tri.v0],
                                      mesh.vertices[patch_tri.v1],
                                      mesh.vertices[patch_tri.v2], px, py, pz);
      }
      solid_angle = std::abs(omega);
      diagonal = solid_angle / kTwoPi;
      if (!std::isfinite(diagonal) || diagonal < 0.10 || diagonal > 2.50) {
        ++mesh.solid_angle_fallbacks;
        diagonal = 1.0;
        solid_angle = kTwoPi;
      } else {
        const double clamped = std::clamp(diagonal, 0.25, 1.75);
        if (clamped != diagonal)
          ++mesh.solid_angle_clamps;
        diagonal = clamped;
      }
    }
    panel.diagonal = diagonal;
    panel.solid_angle = solid_angle;
    mesh.diag_min = std::min(mesh.diag_min, diagonal);
    mesh.diag_max = std::max(mesh.diag_max, diagonal);
  }
  if (!std::isfinite(mesh.diag_min))
    mesh.diag_min = 1.0;
  if (!std::isfinite(mesh.diag_max))
    mesh.diag_max = 1.0;
  mesh.solid_angle_ms = elapsed_ms(begin, std::chrono::steady_clock::now());
}

SurfaceMesh build_triangle_mesh(const std::vector<Atom> &atoms,
                                const std::vector<std::size_t> &original_indices,
                                const Options &options) {
  const double probe =
      options.surface_mode == "sas" ? std::max(0.0, options.probe_radius) : 0.0;
  const int surface_point_max = options.highres_surface ? 65536 : 4096;
  const int point_count =
      std::clamp(options.surface_point_count, 8, surface_point_max);
  const std::string sphere_mesh_mode =
      resolve_sphere_mesh_mode(options.sphere_mesh_mode == "auto" &&
                                      options.surface_mode == "vdw"
                                  ? "icosphere"
                                  : options.sphere_mesh_mode);
  const UnitSphereMesh &unit_mesh =
      cached_unit_sphere_mesh(point_count, sphere_mesh_mode);
  // Salt is deliberately not a condition here.  The quadrature nodes are pure
  // geometry; the screened kernels reuse them at solve time, where kappa is
  // known.  Gating on salt_m <= 0 left every ionic-strength run on the
  // uncorrected flat self term.
  const bool curved_poisson = options.surface_mode == "vdw" &&
      options.discretization == "centroid" &&
      options.formulation != "tabi2-nodepatch" &&
      options.near_integration != "centroid" && !options.surface_only;
  const bool curved_node_geometry = options.surface_mode == "vdw" &&
      (options.discretization == "node-patch" || options.formulation == "tabi2-nodepatch");
  std::vector<std::shared_ptr<const SpherePatchQuadrature>> sphere_quadrature(
      curved_poisson ? unit_mesh.triangles.size() : 0);
  std::shared_ptr<std::vector<std::array<double, 4>>> sphere_geometry;
  if ((curved_poisson || curved_node_geometry) && !original_indices.empty()) {
    sphere_geometry = std::make_shared<std::vector<std::array<double, 4>>>(
        *std::max_element(original_indices.begin(), original_indices.end()) + 1);
    for (std::size_t i = 0; i < atoms.size(); ++i)
      (*sphere_geometry)[original_indices[i]] =
          {atoms[i].x, atoms[i].y, atoms[i].z, atoms[i].radius};
  }
  std::shared_ptr<SpherePatchQuadrature> node_parent_geometry;
  if(curved_node_geometry){
    node_parent_geometry=std::make_shared<SpherePatchQuadrature>();
    node_parent_geometry->spheres=sphere_geometry;
  }

  std::vector<double> surface_radii(atoms.size(), 0.0);
  for (std::size_t i = 0; i < atoms.size(); ++i)
    surface_radii[i] = std::max(0.0, atoms[i].radius) + probe;

  SurfaceMesh mesh;
  mesh.sphere_mesh_mode = sphere_mesh_mode;
  const std::vector<std::vector<std::size_t>> neighbors =
      build_atom_neighbor_lists(atoms, surface_radii, mesh);
  const auto mesh_begin = std::chrono::steady_clock::now();
  mesh.vertices.reserve(atoms.size() * unit_mesh.vertices.size());
  mesh.triangles.reserve(atoms.size() * unit_mesh.triangles.size());
  mesh.panels.reserve(atoms.size() * unit_mesh.triangles.size());
  std::unordered_multimap<QuantizedKey, std::size_t, QuantizedKeyHash> vertex_lookup;
  vertex_lookup.reserve(atoms.size() * unit_mesh.vertices.size());
  std::unordered_map<std::size_t, double> area_by_atom;
  std::unordered_map<std::size_t, double> radius_by_atom;
  double quality_area_min = std::numeric_limits<double>::infinity();
  double quality_area_max = 0.0;
  double quality_area_sum = 0.0;
  double quality_area_square_sum = 0.0;
  double quality_planar_area_sum = 0.0;
  double quality_signed_volume_sum = 0.0;
  double quality_aspect_max = 0.0;
  double quality_radial_dot_min = std::numeric_limits<double>::infinity();
  double quality_radial_dot_sum = 0.0;
  double quality_centroid_radial_error_max = 0.0;
  std::size_t quality_count = 0;

  for (std::size_t atom_index = 0; atom_index < atoms.size(); ++atom_index) {
    const Atom &atom = atoms[atom_index];
    const double radius = surface_radii[atom_index];
    if (radius <= 0.0)
      continue;
    std::vector<SphereClip::Cap> caps;
    caps.reserve(neighbors[atom_index].size());
    bool covered = false;
    // Without clipping (-pbe-bie-sphere-patch-clip=off) the triangles stay
    // whole and the buried test below decides each one, as before the
    // clipping was introduced.
    const bool clip = options.surface_mode == "vdw" && options.sphere_patch_clip;
    if (clip) {
      for (const auto j : neighbors[atom_index]) {
        const auto &other = atoms[j];
        const double dx = other.x - atom.x, dy = other.y - atom.y,
                     dz = other.z - atom.z;
        const double d = norm3(dx, dy, dz), r = surface_radii[j];
        if (d == 0.0 && r == radius) {
          // Coincident duplicate balls have one boundary, owned by the first.
          if (j < atom_index)
            covered = true;
          continue;
        }
        if (d + radius <= r) {
          covered = true;
          break;
        }
        if (d >= radius + r || d + r <= radius)
          continue;
        caps.push_back(
            {{dx / d, dy / d, dz / d},
             std::clamp((d * d + radius * radius - r * r) / (2 * d * radius),
                        -1., 1.)});
      }
    }
    if (covered) {
      mesh.buried_candidates += unit_mesh.triangles.size();
      continue;
    }
    std::vector<SphereClip::Triangle> pieces;
    pieces.reserve(16);
    for (const UnitTriangle &unit_tri : unit_mesh.triangles) {
      const UnitVertex &base_a = unit_mesh.vertices[unit_tri.a];
      const UnitVertex &base_b = unit_mesh.vertices[unit_tri.b];
      const UnitVertex &base_c = unit_mesh.vertices[unit_tri.c];
      const SphereClip::Triangle base{{{base_a.x, base_a.y, base_a.z},
                                       {base_b.x, base_b.y, base_b.z},
                                       {base_c.x, base_c.y, base_c.z}}};
      pieces.clear();
      if (caps.empty())
        pieces.push_back(base);
      else
        SphereClip::clip(base, caps, pieces);
      if (pieces.empty())
        ++mesh.buried_candidates;
      for (const auto &piece : pieces) {
        const UnitVertex ua{piece[0][0], piece[0][1], piece[0][2]};
        const UnitVertex ub{piece[1][0], piece[1][1], piece[1][2]};
        const UnitVertex uc{piece[2][0], piece[2][1], piece[2][2]};
        const double ax = atom.x + ua.x * radius;
        const double ay = atom.y + ua.y * radius;
        const double az = atom.z + ua.z * radius;
        const double bx = atom.x + ub.x * radius;
        const double by = atom.y + ub.y * radius;
        const double bz = atom.z + ub.z * radius;
        const double cx = atom.x + uc.x * radius;
        const double cy = atom.y + uc.y * radius;
        const double cz = atom.z + uc.z * radius;
        double panel_ux = (ua.x + ub.x + uc.x) / 3.0;
        double panel_uy = (ua.y + ub.y + uc.y) / 3.0;
        double panel_uz = (ua.z + ub.z + uc.z) / 3.0;
        const double panel_len = norm3(panel_ux, panel_uy, panel_uz);
        if (panel_len <= 1.0e-14)
          continue;
        panel_ux /= panel_len;
        panel_uy /= panel_len;
        panel_uz /= panel_len;
        const double px = atom.x + panel_ux * radius;
        const double py = atom.y + panel_uy * radius;
        const double pz = atom.z + panel_uz * radius;

        if (!clip) {
          const bool centroid_buried = point_buried(
              px, py, pz, atoms, surface_radii, neighbors[atom_index]);
          const int buried_vertices =
              static_cast<int>(point_buried(ax, ay, az, atoms, surface_radii,
                                            neighbors[atom_index])) +
              static_cast<int>(point_buried(bx, by, bz, atoms, surface_radii,
                                            neighbors[atom_index])) +
              static_cast<int>(point_buried(cx, cy, cz, atoms, surface_radii,
                                            neighbors[atom_index]));
          if (centroid_buried || buried_vertices == 3) {
            ++mesh.buried_candidates;
            continue;
          }
        }
        ++mesh.exposed_candidates;

        std::size_t ia = add_vertex(mesh, vertex_lookup, atom,
                                    original_indices[atom_index], ua, radius);
        std::size_t ib = add_vertex(mesh, vertex_lookup, atom,
                                    original_indices[atom_index], ub, radius);
        std::size_t ic = add_vertex(mesh, vertex_lookup, atom,
                                    original_indices[atom_index], uc, radius);

        const Vertex &va = mesh.vertices[ia];
        const Vertex &vb = mesh.vertices[ib];
        const Vertex &vc = mesh.vertices[ic];
        double ux = vb.x - va.x;
        double uy = vb.y - va.y;
        double uz = vb.z - va.z;
        double vx = vc.x - va.x;
        double vy = vc.y - va.y;
        double vz = vc.z - va.z;
        double nx = uy * vz - uz * vy;
        double ny = uz * vx - ux * vz;
        double nz = ux * vy - uy * vx;
        double normal_len = norm3(nx, ny, nz);
        if (normal_len <= 1.0e-14)
          continue;
        const double planar_area = 0.5 * normal_len;
        double area =
            spherical_triangle_area_unit(ua, ub, uc) * radius * radius;
        if (!std::isfinite(area) || area <= 1.0e-14)
          area = 0.5 * normal_len;
        const double edge_ab = norm3(vb.x - va.x, vb.y - va.y, vb.z - va.z);
        const double edge_bc = norm3(vc.x - vb.x, vc.y - vb.y, vc.z - vb.z);
        const double edge_ca = norm3(va.x - vc.x, va.y - vc.y, va.z - vc.z);
        const double edge_min = std::min({edge_ab, edge_bc, edge_ca});
        const double edge_max = std::max({edge_ab, edge_bc, edge_ca});
        nx /= normal_len;
        ny /= normal_len;
        nz /= normal_len;
        const double flat_px = (va.x + vb.x + vc.x) / 3.0;
        const double flat_py = (va.y + vb.y + vc.y) / 3.0;
        const double flat_pz = (va.z + vb.z + vc.z) / 3.0;
        const double radial_x = flat_px - atom.x;
        const double radial_y = flat_py - atom.y;
        const double radial_z = flat_pz - atom.z;
        bool face_flipped = false;
        if (nx * radial_x + ny * radial_y + nz * radial_z < 0.0) {
          std::swap(ib, ic);
          nx = -nx;
          ny = -ny;
          nz = -nz;
          face_flipped = true;
        }
        const double radial_len = norm3(radial_x, radial_y, radial_z);
        const double radial_dot =
            radial_len > 1.0e-14
                ? (nx * radial_x + ny * radial_y + nz * radial_z) / radial_len
                : 1.0;
        quality_area_min = std::min(quality_area_min, area);
        quality_area_max = std::max(quality_area_max, area);
        quality_area_sum += area;
        quality_area_square_sum += area * area;
        quality_planar_area_sum += planar_area;
        const double signed_volume = (va.x * (vb.y * vc.z - vb.z * vc.y) -
                                      va.y * (vb.x * vc.z - vb.z * vc.x) +
                                      va.z * (vb.x * vc.y - vb.y * vc.x)) /
                                     6.0;
        quality_signed_volume_sum +=
            face_flipped ? -signed_volume : signed_volume;
        if (edge_min > 1.0e-14)
          quality_aspect_max =
              std::max(quality_aspect_max, edge_max / edge_min);
        quality_radial_dot_min = std::min(quality_radial_dot_min, radial_dot);
        quality_radial_dot_sum += radial_dot;
        quality_centroid_radial_error_max = std::max(
            quality_centroid_radial_error_max, std::abs(radial_len - radius));
        ++quality_count;
        area_by_atom[original_indices[atom_index]] += area;
        radius_by_atom[original_indices[atom_index]] = radius;

        Triangle tri;
        tri.v0 = ia;
        tri.v1 = ib;
        tri.v2 = ic;
        tri.atom_index = original_indices[atom_index];
        tri.cx = px;
        tri.cy = py;
        tri.cz = pz;
        tri.nx = nx;
        tri.ny = ny;
        tri.nz = nz;
        tri.area = area;
        mesh.triangles.push_back(tri);
        Panel panel;
        panel.x = px;
        panel.y = py;
        panel.z = pz;
        panel.nx = panel_ux;
        panel.ny = panel_uy;
        panel.nz = panel_uz;
        panel.area = area;
        panel.diagonal = 1.0;
        panel.solid_angle = kTwoPi;
        panel.atom_index = tri.atom_index;
        panel.v0x = va.x;
        panel.v0y = va.y;
        panel.v0z = va.z;
        panel.v1x = vb.x;
        panel.v1y = vb.y;
        panel.v1z = vb.z;
        panel.v2x = vc.x;
        panel.v2y = vc.y;
        panel.v2z = vc.z;
        panel.v0_index = ia;
        panel.v1_index = ib;
        panel.v2_index = ic;
        panel.has_triangle_geometry = true;
        if(curved_node_geometry) panel.sphere_patch_quadrature=node_parent_geometry;
        if (curved_poisson) {
          const auto triangle_index =
              static_cast<std::size_t>(&unit_tri - unit_mesh.triangles.data());
          if (piece == base) {
            auto &quadrature = sphere_quadrature[triangle_index];
            if (!quadrature)
              quadrature = prepare_sphere_patch_quadrature(
                  {ua.x, ua.y, ua.z}, {ub.x, ub.y, ub.z}, {uc.x, uc.y, uc.z},
                  sphere_geometry);
            panel.sphere_patch_quadrature = quadrature;
          } else {
            panel.sphere_patch_quadrature = prepare_sphere_patch_quadrature(
                {ua.x, ua.y, ua.z}, {ub.x, ub.y, ub.z}, {uc.x, uc.y, uc.z},
                sphere_geometry, true);
          }
        }
        mesh.panels.push_back(panel);
      }
    }
  }

  mesh.mesh_build_ms = elapsed_ms(mesh_begin, std::chrono::steady_clock::now());
  mesh.quality_total_area_a2 = quality_area_sum;
  if (quality_count > 0) {
    mesh.quality_triangle_area_min =
        std::isfinite(quality_area_min) ? quality_area_min : 0.0;
    mesh.quality_triangle_area_mean =
        quality_area_sum / static_cast<double>(quality_count);
    mesh.quality_triangle_area_max = quality_area_max;
    mesh.quality_triangle_aspect_max = quality_aspect_max;
    mesh.quality_normal_radial_dot_min =
        std::isfinite(quality_radial_dot_min) ? quality_radial_dot_min : 0.0;
    mesh.quality_normal_radial_dot_mean =
        quality_radial_dot_sum / static_cast<double>(quality_count);
    mesh.quality_centroid_radial_error_max =
        quality_centroid_radial_error_max;
    const double area_mean = quality_area_sum / static_cast<double>(quality_count);
    const double area_variance = std::max(
        0.0, quality_area_square_sum / static_cast<double>(quality_count) -
                 area_mean * area_mean);
    mesh.quality_triangle_area_cv =
        area_mean > 1.0e-14 ? std::sqrt(area_variance) / area_mean : 0.0;
  }
  mesh.quality_planar_area_a2 = quality_planar_area_sum;
  mesh.quality_enclosed_volume_a3 = std::abs(quality_signed_volume_sum);
  if (area_by_atom.size() == 1) {
    const std::size_t atom_index = area_by_atom.begin()->first;
    const auto radius_it = radius_by_atom.find(atom_index);
    const double radius =
        radius_it != radius_by_atom.end() ? radius_it->second : 0.0;
    const double exact_area = 4.0 * kPi * radius * radius;
    if (exact_area > 1.0e-14) {
      mesh.quality_area_rel_error_vs_4piR2 =
          std::abs(quality_area_sum - exact_area) / exact_area;
      mesh.quality_planar_area_rel_error_vs_4piR2 =
          std::abs(quality_planar_area_sum - exact_area) / exact_area;
    }
    const double exact_volume = (4.0 / 3.0) * kPi * radius * radius * radius;
    if (exact_volume > 1.0e-14) {
      mesh.quality_enclosed_volume_rel_error_vs_4piR3_over3 =
          std::abs(mesh.quality_enclosed_volume_a3 - exact_volume) /
          exact_volume;
    }
  }
  finalize_solid_angle_diagonal(mesh, options.diagonal_mode);
  return mesh;
}

std::vector<Panel> build_point_panel_surface(const std::vector<Atom> &atoms,
                                             const Options &options,
                                             Result *result,
                                             double elapsed_surface_ms,
                                             TitanExecutor *executor) {
  std::vector<TitanSurface::Atom> surface_atoms;
  std::vector<std::size_t> original_indices;
  surface_atoms.reserve(atoms.size());
  original_indices.reserve(atoms.size());
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const Atom &atom = atoms[i];
    if (!finite_atom(atom))
      continue;
    surface_atoms.push_back({atom.x, atom.y, atom.z, atom.radius});
    original_indices.push_back(i);
  }

  TitanSurface::Options surface_options;
  surface_options.probe_radius = options.probe_radius;
  surface_options.min_cell_size = 1.0;
  const int surface_point_max = options.highres_surface ? 65536 : 4096;
  surface_options.point_count =
      std::clamp(options.surface_point_count, 8, surface_point_max);
  surface_options.surface_mode =
      options.surface_mode == "sas" ? std::string("sas") : std::string("vdw");
  surface_options.emit_panels = true;
  surface_options.emit_neighbor_workspace = false;

  TitanSurface::Workspace workspace =
      TitanSurface::build_workspace(surface_atoms, surface_options, {},
                                    executor);

  std::vector<Panel> panels;
  panels.reserve(workspace.panels.size());
  for (const TitanSurface::Panel &surface_panel : workspace.panels) {
    Panel panel;
    panel.x = surface_panel.x;
    panel.y = surface_panel.y;
    panel.z = surface_panel.z;
    panel.nx = surface_panel.nx;
    panel.ny = surface_panel.ny;
    panel.nz = surface_panel.nz;
    panel.area = surface_panel.area;
    panel.diagonal = options.diagonal_mode == "off" ? 0.0 : 1.0;
    panel.solid_angle = panel.diagonal > 0.0 ? kTwoPi : 0.0;
    panel.atom_index = surface_panel.atom_index < original_indices.size()
                           ? original_indices[surface_panel.atom_index]
                           : surface_panel.atom_index;
    panels.push_back(panel);
  }

  if (result != nullptr) {
    result->surface_geometry = "point-panel";
    result->sphere_mesh_mode = "off";
    result->panel_count = panels.size();
    result->surface_area_a2 = workspace.total_area;
    result->surface_build_ms = elapsed_surface_ms + workspace.diagnostics.build_ms;
    result->surface_neighbor_ms =
        workspace.diagnostics.neighbor_build_ms;
    result->surface_mesh_ms = workspace.diagnostics.exposure_ms;
    result->solid_angle_ms = 0.0;
    result->exposed_points = workspace.diagnostics.exposed_points;
    result->buried_points = workspace.diagnostics.buried_points;
    result->boundary_nodes = panels.size();
    result->diag_min = options.diagonal_mode == "off" ? 0.0 : 1.0;
    result->diag_max = options.diagonal_mode == "off" ? 0.0 : 1.0;
  }
  return panels;
}

void populate_implicit_mesh_quality(SurfaceMesh &mesh) {
  double area_sum = 0.0;
  double area_square_sum = 0.0;
  double area_min = std::numeric_limits<double>::infinity();
  double area_max = 0.0;
  double aspect_max = 0.0;
  double signed_volume = 0.0;
  for (const Triangle &triangle : mesh.triangles) {
    area_sum += triangle.area;
    area_square_sum += triangle.area * triangle.area;
    area_min = std::min(area_min, triangle.area);
    area_max = std::max(area_max, triangle.area);
    const Vertex &a = mesh.vertices[triangle.v0];
    const Vertex &b = mesh.vertices[triangle.v1];
    const Vertex &c = mesh.vertices[triangle.v2];
    const double ab = norm3(a.x - b.x, a.y - b.y, a.z - b.z);
    const double bc = norm3(b.x - c.x, b.y - c.y, b.z - c.z);
    const double ca = norm3(c.x - a.x, c.y - a.y, c.z - a.z);
    const double edge_min = std::min({ab, bc, ca});
    const double edge_max = std::max({ab, bc, ca});
    if (edge_min > 1.0e-14) {
      aspect_max = std::max(aspect_max, edge_max / edge_min);
    }
    signed_volume +=
        (a.x * (b.y * c.z - b.z * c.y) -
         a.y * (b.x * c.z - b.z * c.x) +
         a.z * (b.x * c.y - b.y * c.x)) /
        6.0;
  }
  mesh.quality_total_area_a2 = area_sum;
  mesh.quality_planar_area_a2 = area_sum;
  mesh.quality_enclosed_volume_a3 = std::abs(signed_volume);
  mesh.quality_triangle_area_min =
      std::isfinite(area_min) ? area_min : 0.0;
  mesh.quality_triangle_area_max = area_max;
  mesh.quality_triangle_aspect_max = aspect_max;
  if (!mesh.triangles.empty()) {
    const double count = static_cast<double>(mesh.triangles.size());
    const double mean = area_sum / count;
    const double variance =
        std::max(0.0, area_square_sum / count - mean * mean);
    mesh.quality_triangle_area_mean = mean;
    mesh.quality_triangle_area_cv =
        mean > 1.0e-14 ? std::sqrt(variance) / mean : 0.0;
  }
}

} // namespace

std::vector<Panel> build_surface(const std::vector<Atom> &atoms,
                                 const Options &options, Result *result,
                                 TitanExecutor *executor) {
  const auto surface_begin = std::chrono::steady_clock::now();
  if (options.surface_geometry == "point-panel") {
    return build_point_panel_surface(atoms, options, result, 0.0, executor);
  }

  std::vector<Atom> finite_atoms;
  std::vector<std::size_t> original_indices;
  finite_atoms.reserve(atoms.size());
  original_indices.reserve(atoms.size());
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    if (!finite_atom(atoms[i]))
      continue;
    finite_atoms.push_back(atoms[i]);
    original_indices.push_back(i);
  }

  SurfaceMesh mesh;
  calderon::TraceSpaces calderon_trace;
  calderon::AuxiliarySurfaceResult primary_surface;
  if (options.primary_surface == "grid-ses" ||
      options.primary_surface == "analytic-ses") {
    const bool analytic = options.primary_surface == "analytic-ses";
    primary_surface =
        analytic ? calderon::build_analytic_ses_surface(
                       finite_atoms, options.ses_grid_spacing,
                       options.probe_radius, executor,
                       options.ses_max_grid_vertices, true,
                       options.analytic_ses_engine, options.ses_remesh_edge,
                       calderon::ChargeRefinement{
                           options.ses_charge_refine,
                           options.ses_charge_refine_min_edge,
                           options.ses_charge_refine_min_charge},
                       options.ses_remesh_iterations)
                 : calderon::build_implicit_ses_surface(
                       finite_atoms, options.ses_grid_spacing,
                       options.probe_radius, executor,
                       options.ses_max_grid_vertices, true);
    if (primary_surface.status != "pass") {
      std::string hint;
      // Say what the status means and what to change.  These two are the
      // ones a correct input can still hit, and both are about the spacing
      // rather than about anything wrong with the molecule.
      if (primary_surface.status == "analytic_sheet_bridge_below_spacing") {
        hint = "  " +
               std::to_string(primary_surface.analytic_sheet_bridge_faces) +
               " triangle(s) join two facing sheets of the surface across a"
               " gap narrower than one grid cell, so a single cell holds"
               " both sheets and no repair can separate them.  Reduce"
               " -pbe-bie-ses-grid-spacing (currently " +
               std::to_string(options.ses_grid_spacing) + " A).";
      } else if (primary_surface.status == "analytic_vertex_commit_refused") {
        hint = "  " +
               std::to_string(primary_surface.analytic_vertex_refused) +
               " vertex/vertices could not be moved onto their certified"
               " analytic target without reversing an incident triangle; the"
               " largest outstanding move is " +
               std::to_string(
                   primary_surface.analytic_vertex_refused_distance_max) +
               " A.  Reduce -pbe-bie-ses-grid-spacing.";
      }
      throw std::runtime_error(
          "TABI " + options.primary_surface +
          " surface failed watertight topology gate: " +
          primary_surface.status + hint);
    }
    if (analytic && options.analytic_ses_engine == "patch-ray-resolution") {
      // The resolution policy accepts what the certified engine refuses, so
      // say how much of the surface is on the analytic patches.
      const auto &s = primary_surface;
      const std::size_t vertices = s.mesh.vertices.size();
      std::cerr << "TABI SES patch-ray-resolution: h=" << options.ses_grid_spacing
                << " A vertices=" << vertices
                << " triangles=" << s.mesh.triangles.size()
                << " marching_vertices=" << s.projection_queries
                << " projected=" << s.projected_vertices << " ("
                << (s.projection_queries == 0U
                        ? 0.0
                        : 100.0 * static_cast<double>(s.projected_vertices) /
                              static_cast<double>(s.projection_queries))
                << "%) partial=" << s.resolution_partial_projection_vertices
                << " star_refused=" << s.resolution_star_refused_vertices
                << " sheet_refused=" << s.resolution_sheet_refused_vertices
                << " missing=" << s.resolution_missing_projection_vertices
                << " reverted=" << s.resolution_reverted_vertices
                << " revert_rounds=" << s.resolution_revert_rounds
                << " initial_intersections="
                << s.resolution_initial_intersections
                << " odd_ray_rows=" << s.odd_ray_rows
                << " uncertified_ray_rows="
                << s.patch_ray_unresolved_uncertified_rows
                << " nearby_parity_vertices="
                << s.resolution_nearby_parity_vertices
                << " fallback_sign_vertices=" << s.morphology_fallback_vertices
                << " thin_solvent_filled=" << s.resolution_thin_solvent_filled
                << " thin_plates_opened=" << s.resolution_thin_plates_opened
                << " unprojected_distance_max_A="
                << s.resolution_unprojected_distance_max
                << " projection_rms_A=" << s.analytic_projection_distance_rms
                << " min_angle_below_5/10/20_deg=" << s.resolution_angle_below_5
                << '/' << s.resolution_angle_below_10 << '/'
                << s.resolution_angle_below_20
                << " remesh_edge_A=" << s.resolution_remesh_edge
                << " remesh_iterations=" << s.resolution_remesh_iterations
                << " remesh_undone=" << s.resolution_remesh_reverted
                << " splits=" << s.resolution_remesh_splits
                << " collapses=" << s.resolution_remesh_collapses
                << " remesh_flips=" << s.resolution_remesh_flips
                << " off_surface=" << s.resolution_remesh_off_surface
                << " remesh_sheet_refused=" << s.resolution_remesh_sheet_refused
                << " refine_charges=" << s.resolution_refine_charges
                << " refined_vertices=" << s.resolution_refined_vertices
                << " analytic_normals=" << s.resolution_analytic_normal_vertices
                << " analytic_normal_refused="
                << s.resolution_analytic_normal_refused
                << " edges_below_0.05/0.2_A=" << s.resolution_short_edges_005
                << '/' << s.resolution_short_edges_020
                << " edge_min_A=" << s.resolution_edge_min
                << " edge_mean_A=" << s.resolution_edge_mean
                << " build_s=" << s.build_ms / 1000.0
                << " (patches_s=" << s.patch_build_ms / 1000.0
                << " rays_s=" << s.ray_build_ms / 1000.0
                << " topology_s=" << s.topology_ms / 1000.0
                << " polygonize_s=" << s.polygonize_ms / 1000.0
                << " projection_s=" << s.projection_ms / 1000.0
                << " remesh_s=" << s.resolution_remesh_ms / 1000.0 << ")\n";
    }
    if (!analytic) {
      const auto &s = primary_surface;
      std::cerr << "TABI Grid-SES: h=" << options.ses_grid_spacing
                << " A vertices=" << s.mesh.vertices.size()
                << " triangles=" << s.mesh.triangles.size()
                << " projected=" << s.projected_vertices << " ("
                << (s.projection_queries == 0U
                        ? 0.0
                        : 100.0 * static_cast<double>(s.projected_vertices) /
                              static_cast<double>(s.projection_queries))
                << "%) build_s=" << s.build_ms / 1000.0 << "\n";
    }
    mesh = std::move(primary_surface.mesh);
    calderon_trace = std::move(primary_surface.trace_spaces);
    mesh.sphere_mesh_mode = options.primary_surface;
    if (analytic && (options.analytic_ses_engine == "patch-ray-experimental" ||
                     options.analytic_ses_engine == "patch-ray-resolution")) {
      // Reuse the established profile columns as a stable high-level split:
      // neighbor = patch graph + compact ray CSR, mesh = topology + active
      // polygonization + projection/repair. surface_build_ms remains total.
      mesh.neighbor_build_ms =
          primary_surface.patch_build_ms + primary_surface.ray_build_ms;
      mesh.mesh_build_ms = primary_surface.topology_ms +
                           primary_surface.polygonize_ms +
                           primary_surface.repair_ms;
    } else {
      mesh.mesh_build_ms = primary_surface.build_ms;
    }
    for (Vertex &vertex : mesh.vertices) {
      if (vertex.atom_index < original_indices.size()) {
        vertex.atom_index = original_indices[vertex.atom_index];
      }
    }
    // Hand the analytic vertex normals to the triangles, where node-patch
    // construction can find them by vertex index.  Only the surface engine
    // knows which vertices lie on the SES - the analytic engines' certified
    // projections, or Grid-SES vertices moved onto it from their nearest
    // probe position - so this is the last point at which the information
    // exists.
    if (std::any_of(mesh.vertices.begin(), mesh.vertices.end(),
                    [](const Vertex &v) { return v.normal_analytic; })) {
      constexpr double nan = std::numeric_limits<double>::quiet_NaN();
      auto normals = std::make_shared<std::vector<std::array<double, 3>>>(
          mesh.vertices.size(), std::array<double, 3>{nan, nan, nan});
      for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const Vertex &v = mesh.vertices[i];
        if (v.normal_analytic)
          (*normals)[i] = {v.nx, v.ny, v.nz};
      }
      std::shared_ptr<const std::vector<std::array<double, 3>>> shared =
          std::move(normals);
      for (Panel &panel : mesh.panels)
        panel.analytic_vertex_normals = shared;
    }
    for (Triangle &triangle : mesh.triangles) {
      if (triangle.atom_index < original_indices.size()) {
        triangle.atom_index = original_indices[triangle.atom_index];
      }
    }
    for (Panel &panel : mesh.panels) {
      if (panel.atom_index < original_indices.size()) {
        panel.atom_index = original_indices[panel.atom_index];
      }
    }
    populate_implicit_mesh_quality(mesh);
    if (options.surface_only) {
      for (Panel &panel : mesh.panels) {
        panel.diagonal = 0.0;
        panel.solid_angle = 0.0;
      }
      mesh.diag_min = 0.0;
      mesh.diag_max = 0.0;
      mesh.solid_angle_ms = 0.0;
    } else {
      finalize_solid_angle_diagonal(mesh, options.diagonal_mode);
    }
  } else {
    mesh = build_triangle_mesh(finite_atoms, original_indices, options);
    if (options.calderon_trace_audit) {
      calderon_trace = calderon::build_closed_surface_trace_spaces(mesh);
    }
  }
  calderon::AuxiliarySurfaceResult calderon_auxiliary;
  if (options.calderon_aux_surface == "implicit-vdw") {
    calderon_auxiliary = calderon::build_implicit_vdw_auxiliary_surface(
        finite_atoms, options.calderon_aux_grid_spacing);
  }
  if (result != nullptr) {
    result->surface_geometry = "triangle";
    result->panel_count = mesh.panels.size();
    result->surface_area_a2 = std::accumulate(
        mesh.panels.begin(), mesh.panels.end(), 0.0,
        [](double sum, const Panel &panel) { return sum + panel.area; });
    result->sphere_mesh_mode = mesh.sphere_mesh_mode;
    result->surface_quality_total_area_a2 = mesh.quality_total_area_a2;
    result->surface_quality_area_rel_error_vs_4piR2 =
        mesh.quality_area_rel_error_vs_4piR2;
    result->surface_quality_triangle_area_min =
        mesh.quality_triangle_area_min;
    result->surface_quality_triangle_area_mean =
        mesh.quality_triangle_area_mean;
    result->surface_quality_triangle_area_max =
        mesh.quality_triangle_area_max;
    result->surface_quality_triangle_aspect_max =
        mesh.quality_triangle_aspect_max;
    result->surface_quality_normal_radial_dot_min =
        mesh.quality_normal_radial_dot_min;
    result->surface_quality_normal_radial_dot_mean =
        mesh.quality_normal_radial_dot_mean;
    result->surface_quality_centroid_radial_error_max =
        mesh.quality_centroid_radial_error_max;
    result->surface_quality_planar_area_a2 =
        mesh.quality_planar_area_a2;
    result->surface_quality_planar_area_rel_error_vs_4piR2 =
        mesh.quality_planar_area_rel_error_vs_4piR2;
    result->surface_quality_enclosed_volume_a3 =
        mesh.quality_enclosed_volume_a3;
    result->surface_quality_enclosed_volume_rel_error_vs_4piR3_over3 =
        mesh.quality_enclosed_volume_rel_error_vs_4piR3_over3;
    result->surface_quality_triangle_area_cv =
        mesh.quality_triangle_area_cv;
    result->surface_topology_source = primary_surface.topology_source;
    result->surface_analytic_ses_engine =
        primary_surface.analytic_ses_engine;
    result->surface_status_detail = primary_surface.status;
    if (options.primary_surface == "sphere-patch" && options.surface_mode == "vdw") {
      result->surface_topology_source = options.sphere_patch_clip
                                            ? "sphere_patch_cap_clip"
                                            : "sphere_patch_whole_triangles";
      result->surface_status_detail = "approximate_exposed_patches";
    }
    result->surface_cavity_policy = primary_surface.cavity_policy;
    result->surface_odd_ray_rows = primary_surface.odd_ray_rows;
    result->surface_patch_ray_unresolved_uncertified_rows =
        primary_surface.patch_ray_unresolved_uncertified_rows;
    result->surface_morphology_fallback_vertices =
        primary_surface.morphology_fallback_vertices;
    result->surface_analytic_orientation_status =
        primary_surface.analytic_orientation_status;
    result->surface_analytic_inverted_faces =
        primary_surface.analytic_inverted_faces;
    result->surface_analytic_orientation_uncertified_faces =
        primary_surface.analytic_orientation_uncertified_faces;
    result->surface_analytic_self_intersections =
        primary_surface.analytic_self_intersections;
    result->surface_wide_repair_attempts =
        primary_surface.wide_repair_attempts;
    result->surface_topology_certificate_accepts =
        primary_surface.topology_certificate_accepts;
    result->surface_topology_fallback_audits =
        primary_surface.topology_fallback_audits;
    result->surface_initial_collapse_ms = primary_surface.initial_collapse_ms;
    result->surface_stable_sliver_ms = primary_surface.stable_sliver_ms;
    result->surface_full_projection_ms = primary_surface.full_projection_ms;
    result->surface_orientation_repair_ms =
        primary_surface.orientation_repair_ms;
    result->surface_wide_repair_ms = primary_surface.wide_repair_ms;
    result->surface_self_intersection_audit_ms =
        primary_surface.self_intersection_audit_ms;
    result->surface_filled_cavity_regions =
        primary_surface.filled_cavity_regions;
    result->surface_filled_cavity_vertices =
        primary_surface.filled_cavity_vertices;
    result->surface_poor_triangles_before =
        primary_surface.poor_triangles_before;
    result->surface_poor_triangles_after =
        primary_surface.poor_triangles_after;
    result->surface_triangle_quality_min_before =
        primary_surface.triangle_quality_min_before;
    result->surface_triangle_quality_min_after =
        primary_surface.triangle_quality_min_after;
    result->surface_max_edge_ratio_before =
        primary_surface.max_edge_ratio_before;
    result->surface_max_edge_ratio_after =
        primary_surface.max_edge_ratio_after;
    result->surface_analytic_audit_status =
        primary_surface.analytic_audit_status;
    result->surface_analytic_audit_samples =
        primary_surface.analytic_audit_samples;
    result->surface_analytic_audit_hits = primary_surface.analytic_audit_hits;
    result->surface_analytic_audit_misses =
        primary_surface.analytic_audit_misses;
    result->surface_analytic_projection_distance_rms =
        primary_surface.analytic_projection_distance_rms;
    result->surface_analytic_projection_distance_max =
        primary_surface.analytic_projection_distance_max;
    result->surface_analytic_patch_residual_rms =
        primary_surface.analytic_patch_residual_rms;
    result->surface_analytic_patch_residual_max =
        primary_surface.analytic_patch_residual_max;
    result->surface_analytic_normal_angle_rms_deg =
        primary_surface.analytic_normal_angle_rms_deg;
    result->surface_analytic_normal_angle_max_deg =
        primary_surface.analytic_normal_angle_max_deg;
    result->surface_sampled_hausdorff_proxy =
        primary_surface.sampled_hausdorff_proxy;
    result->surface_analytic_normal_quality_status =
        primary_surface.analytic_normal_quality_status;
    result->surface_analytic_orientation_smooth_faces =
        primary_surface.analytic_orientation_smooth_faces;
    result->surface_analytic_orientation_seam_faces =
        primary_surface.analytic_orientation_seam_faces;
    result->surface_analytic_normal_cosine_below_025_faces =
        primary_surface.analytic_normal_cosine_below_025_faces;
    result->surface_analytic_normal_cosine_below_050_faces =
        primary_surface.analytic_normal_cosine_below_050_faces;
    result->surface_analytic_normal_area_weighted_rms_deg =
        primary_surface.analytic_normal_area_weighted_rms_deg;
    result->surface_analytic_normal_p99_deg =
        primary_surface.analytic_normal_p99_deg;
    result->surface_analytic_smooth_normal_p99_deg =
        primary_surface.analytic_smooth_normal_p99_deg;
    result->surface_analytic_seam_normal_p99_deg =
        primary_surface.analytic_seam_normal_p99_deg;
    result->surface_analytic_normal_area_below_cosine_025_fraction =
        primary_surface.analytic_normal_area_below_cosine_025_fraction;
    result->surface_analytic_normal_area_below_cosine_050_fraction =
        primary_surface.analytic_normal_area_below_cosine_050_fraction;
    result->calderon_trace_space_status = calderon_trace.quality.failure_reason;
    result->calderon_trace_vertices = calderon_trace.quality.vertex_count;
    result->calderon_trace_edges = calderon_trace.quality.edge_count;
    result->calderon_trace_boundary_edges =
        calderon_trace.quality.boundary_edge_count;
    result->calderon_trace_orientation_mismatches =
        calderon_trace.quality.orientation_mismatch_count;
    result->calderon_trace_nonmanifold_edges =
        calderon_trace.quality.nonmanifold_edge_count;
    result->calderon_trace_unreferenced_vertices =
        calderon_trace.quality.unreferenced_vertex_count;
    result->calderon_trace_components =
        calderon_trace.quality.connected_component_count;
    result->calderon_trace_euler_characteristic =
        calderon_trace.quality.euler_characteristic;
    result->calderon_trace_genus_sum = calderon_trace.quality.genus_sum;
    result->calderon_trace_topology_invariants_valid =
        calderon_trace.quality.topology_invariants_valid;
    result->calderon_aux_surface_mode = options.calderon_aux_surface;
    result->calderon_aux_trace_status =
        options.calderon_aux_surface == "off" ? "off" : calderon_auxiliary.status;
    result->calderon_aux_panels = calderon_auxiliary.mesh.panels.size();
    result->calderon_aux_vertices = calderon_auxiliary.trace_spaces.quality.vertex_count;
    result->calderon_aux_edges = calderon_auxiliary.trace_spaces.quality.edge_count;
    result->calderon_aux_boundary_edges =
        calderon_auxiliary.trace_spaces.quality.boundary_edge_count;
    result->calderon_aux_orientation_mismatches =
        calderon_auxiliary.trace_spaces.quality.orientation_mismatch_count;
    result->calderon_aux_nonmanifold_edges =
        calderon_auxiliary.trace_spaces.quality.nonmanifold_edge_count;
    result->calderon_aux_unreferenced_vertices =
        calderon_auxiliary.trace_spaces.quality.unreferenced_vertex_count;
    result->calderon_aux_grid_vertices = calderon_auxiliary.grid_vertices;
    result->calderon_aux_area_a2 =
        calderon_auxiliary.trace_spaces.quality.total_area;
    result->calderon_aux_build_ms = calderon_auxiliary.build_ms;
    result->surface_build_ms =
        elapsed_ms(surface_begin, std::chrono::steady_clock::now());
    result->surface_neighbor_ms = mesh.neighbor_build_ms;
    result->surface_mesh_ms = mesh.mesh_build_ms;
    result->solid_angle_ms = mesh.solid_angle_ms;
    result->exposed_points = mesh.exposed_candidates;
    result->buried_points = mesh.buried_candidates;
    result->mesh_vertices = mesh.vertices.size();
    result->mesh_triangles = mesh.triangles.size();
    result->boundary_nodes = mesh.panels.size();
    result->avg_valence = mesh.avg_valence;
    result->nonmanifold_edges = mesh.nonmanifold_edges;
    result->solid_angle_fallbacks = mesh.solid_angle_fallbacks;
    result->solid_angle_clamps = mesh.solid_angle_clamps;
    result->diag_min = mesh.diag_min;
    result->diag_max = mesh.diag_max;
  }
  return std::move(mesh.panels);
}

} // namespace TitanPBE::Tabi
